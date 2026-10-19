"""Pull raw statcast pitch data and snapshot it to parquet. Network lives here
and nowhere else in the pipeline -- every later stage is a pure function of
these files.

    python -m basenet.data.ingest --seasons 2021 2022 2023 2024 2025
    python -m basenet.data.ingest --seasons 2024 --force

Idempotent: a season already snapshotted is skipped unless --force. Also
resolves and upserts the MLBAM id crosswalk (data/reference/player_ids.parquet)
for every batter/pitcher id seen in the seasons just pulled.

Also snapshots MLB StatsAPI's regular-season schedule per season
(data/raw/schedule/season=YYYY/schedule_YYYY.parquet): official final scores
and real venue ids per game_pk. It is a source *independent* of statcast's
pitch feed, so pa.py can check its derived labels against it rather than
against themselves.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import pandas as pd

from . import config
from .ids import build_id_crosswalk, upsert_crosswalk
from .paths import ensure_dir, raw_schedule_season_dir, raw_statcast_season_dir, StageTimer

STATSAPI_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&gameType=R"
)

# One row per schedule entry, NOT per game: a postponed game appears once
# with its original date (codedGameState 'D') and again when it's played.
# Kept raw here; pa.py filters to played games (see config.PLAYED_GAME_STATES).
SCHEDULE_COLUMNS = [
    "game_pk", "official_date", "season", "game_type", "coded_game_state",
    "detailed_state", "home_team_id", "home_team_name", "away_team_id",
    "away_team_name", "home_score", "away_score", "venue_id", "venue_name",
    "double_header", "game_number",
]


def _season_snapshot_path(season: int):
    return raw_statcast_season_dir(season) / f"statcast_{season}.parquet"


def season_already_ingested(season: int) -> bool:
    return _season_snapshot_path(season).exists()


def ingest_season(season: int, force: bool = False) -> pd.DataFrame | None:
    """Pull one season of statcast, filter to regular season, write snapshot.

    Returns the written DataFrame, or None if skipped (already present, not forced).
    """
    out_path = _season_snapshot_path(season)
    if out_path.exists() and not force:
        print(f"[ingest] season={season} already snapshotted at {out_path}, skipping "
              f"(use --force to re-pull)")
        return None

    import pybaseball as pb

    pb.cache.enable()  # our own parquet snapshot below is the durable copy;
                        # pybaseball's cache just avoids re-hitting Savant on retries.

    start_dt, end_dt = config.SEASON_DATE_RANGES[season]
    with StageTimer(f"ingest season={season}").run() as t:
        df = pb.statcast(start_dt=start_dt, end_dt=end_dt)
        t.rows_in = len(df)

        # Defensive filter even though SEASON_DATE_RANGES is already tight:
        # this is what actually guarantees "regular season only" (drops any
        # spring training / All-Star / postseason game_pks Savant returns
        # for dates near the boundary).
        df = df[df["game_type"] == config.REGULAR_SEASON_GAME_TYPE].copy()
        df["season"] = season
        # pybaseball.statcast() concatenates one DataFrame per ~5-day chunk;
        # `game_date` comes back as `object` dtype with a genuine mix of
        # str/Timestamp/datetime.date across chunks (observed on a real
        # full-season pull), which pyarrow refuses to write. Normalize once,
        # here, so every later stage can assume a real datetime64 column.
        df["game_date"] = pd.to_datetime(df["game_date"])
        t.rows_out = len(df)

        ensure_dir(out_path.parent)
        df.to_parquet(out_path, index=False)
        print(f"[ingest] wrote {out_path} ({len(df):,} rows, "
              f"{df['game_pk'].nunique():,} games)")

    return df


def _schedule_snapshot_path(season: int):
    return raw_schedule_season_dir(season) / f"schedule_{season}.parquet"


def parse_schedule(payload: dict, season: int) -> pd.DataFrame:
    """StatsAPI /schedule JSON -> flat SCHEDULE_COLUMNS frame (pure; no network)."""
    rows = []
    for date in payload.get("dates", []):
        for g in date["games"]:
            home, away = g["teams"]["home"], g["teams"]["away"]
            rows.append({
                "game_pk": g["gamePk"],
                "official_date": g.get("officialDate"),
                "season": season,
                "game_type": g.get("gameType"),
                "coded_game_state": g["status"].get("codedGameState"),
                "detailed_state": g["status"].get("detailedState"),
                "home_team_id": home["team"]["id"],
                "home_team_name": home["team"].get("name"),
                "away_team_id": away["team"]["id"],
                "away_team_name": away["team"].get("name"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
                "venue_id": g.get("venue", {}).get("id"),
                "venue_name": g.get("venue", {}).get("name"),
                "double_header": g.get("doubleHeader"),
                "game_number": g.get("gameNumber"),
            })
    df = pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)
    df["official_date"] = pd.to_datetime(df["official_date"])
    # unplayed games carry no score: nullable ints, not floats
    for c in ("home_score", "away_score", "venue_id"):
        df[c] = df[c].astype("Int64")
    return df


def ingest_schedule(season: int, force: bool = False) -> pd.DataFrame | None:
    """Pull one season's regular-season schedule from StatsAPI and snapshot it.
    Returns the written frame, or None if skipped (already present, not forced).
    """
    out_path = _schedule_snapshot_path(season)
    if out_path.exists() and not force:
        print(f"[ingest] schedule season={season} already snapshotted at {out_path}, skipping")
        return None
    with StageTimer(f"ingest schedule season={season}").run() as t:
        with urllib.request.urlopen(STATSAPI_SCHEDULE_URL.format(season=season), timeout=60) as resp:
            payload = json.load(resp)
        df = parse_schedule(payload, season)
        t.rows_in = t.rows_out = len(df)
        ensure_dir(out_path.parent)
        df.to_parquet(out_path, index=False)
        print(f"[ingest] wrote {out_path} ({len(df):,} schedule rows, "
              f"{df['game_pk'].nunique():,} distinct game_pks)")
    return df


def update_id_crosswalk(df: pd.DataFrame) -> None:
    ids = set(df["batter"].dropna().astype(int)) | set(df["pitcher"].dropna().astype(int))
    resolved = build_id_crosswalk(ids)
    upsert_crosswalk(resolved)
    print(f"[ingest] crosswalk updated: {len(resolved):,} ids resolved this run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True,
                         choices=config.SEASONS)
    parser.add_argument("--force", action="store_true",
                         help="re-pull and overwrite even if a snapshot already exists")
    parser.add_argument("--skip-crosswalk", action="store_true",
                         help="skip the id-crosswalk resolution step (faster iteration)")
    args = parser.parse_args(argv)

    for season in args.seasons:
        df = ingest_season(season, force=args.force)
        if df is not None and not args.skip_crosswalk:
            update_id_crosswalk(df)
        ingest_schedule(season, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
