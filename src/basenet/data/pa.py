"""Raw statcast pitches -> one row per plate appearance.

    python -m basenet.data.pa --seasons 2021 2022 2023 2024 2025

Pure function of `data/raw/statcast/season=YYYY/*.parquet` (no network).
Re-running from scratch must reproduce the same output byte-for-byte given
the same raw snapshot.

Column notes (see DATA.md for the authoritative per-column doc):

- `order_slot` is derived by ranking each team's own plate appearances within
  a game by `at_bat_number` (which is globally monotonic across the whole
  game, both teams interleaved) and taking `((rank - 1) % 9) + 1`. This is
  not an approximation of "track who replaced whom" -- a fixed 9-slot
  rotation *is* the actual rule (a substitute bats in his predecessor's
  slot), so this reproduces the true slot exactly, with one caveat: it
  assumes the pitch feed's PA order is complete and correct, which fails
  silently in the vanishingly rare case of a team batting out of turn. We
  additionally log how often a team uses more than 9 distinct batters in a
  game (very common, due to pinch hitters/subs) as a sanity diagnostic --
  that is expected and is not the same thing as a slot-derivation error.
- `times_through_order` is taken directly from statcast's own
  `n_thruorder_pitcher` rather than recomputed.
- `park_id` is a **team proxy** (`home_team`), not a true venue id -- the
  standard statcast pull carries no venue/ballpark id column. This
  misattributes park effects whenever a team's home games are played
  somewhere else (e.g. the Rays' 2025 season at Steinbrenner Field after
  Tropicana Field's hurricane damage; international series in London/Mexico
  City/Seoul/Tokyo). Kept as the home-team code (it's what `batting_team ==
  park_id` relies on); the real ballpark is `venue_id`, StatsAPI's venue id
  per game_pk from the schedule snapshot.
- `final_home_score`/`final_away_score` (the label source) are StatsAPI's
  **official** final scores (`data/raw/schedule/`), not statcast's. Statcast
  can't always see the winning run: a walk-off wild pitch ends the game on
  a pitch with no `events` value (so terminal-pitch-only scores record a
  tie), and a walk-off balk has no pitch row at all. Statcast's own final
  (max post-pitch score over every pitch) is cross-checked against the
  official one and the stage fails on any disagreement other than that
  pitchless walk-off -- see `check_finals_against_schedule`.
- `is_doubleheader_game_2` is derived by ranking a team-pair's game_pks on a
  shared date by `game_pk` ascending. statcast's timestamp field for this
  (`sv_id`) is entirely null in modern data, so we fall back to game_pk
  order, which is chronological in practice but not a guaranteed MLBAM
  contract. Documented approximation.
"""
from __future__ import annotations

import argparse

import duckdb
import pandas as pd

from . import config
from .paths import (
    interim_pa_season_dir, raw_schedule_season_dir, raw_statcast_season_dir, ensure_dir, StageTimer,
)

PA_COLUMNS = [
    "game_pk", "game_date", "season", "park_id", "venue_id", "is_doubleheader_game_2",
    "inning", "half", "batting_team", "pitching_team",
    "batter_id", "pitcher_id", "batter_stands", "pitcher_throws",
    "order_slot", "times_through_order", "outcome", "runs_on_play",
    # batted-ball type (ground_ball/fly_ball/line_drive/popup), populated
    # only when `outcome` is a ball put in play (OUT_IP/1B/2B/3B/HR/ROE) --
    # feeds GB%/FB% in features.py.
    "bb_type",
    # quality of contact / run value of the PA's terminal pitch -- feed the
    # xwOBA / hard-hit / barrel features in features.py (outcomes of THIS
    # PA: only ever used through strictly-earlier windows).
    "woba_value", "woba_denom", "launch_speed", "launch_speed_angle", "xwoba_contact",
    # internal/auxiliary -- see DATA.md
    "at_bat_number", "final_home_score", "final_away_score",
]


def _events_case_sql() -> str:
    whens = "\n            ".join(
        f"WHEN '{event}' THEN '{outcome}'"
        for event, outcome in config.EVENTS_TO_OUTCOME.items()
    )
    return f"CASE events\n            {whens}\n            ELSE 'OTHER'\n        END"


def _assert_events_covered(con: duckdb.DuckDBPyConnection, raw_path: str) -> None:
    known = set(config.EVENTS_TO_OUTCOME) | config.DROPPED_EVENTS
    seen = con.execute(
        f"SELECT DISTINCT events FROM read_parquet('{raw_path}') WHERE events IS NOT NULL"
    ).df()["events"].tolist()
    unknown = sorted(set(seen) - known)
    if unknown:
        raise ValueError(
            f"Unmapped statcast `events` value(s) encountered: {unknown}. "
            f"Add them to config.EVENTS_TO_OUTCOME or config.DROPPED_EVENTS -- "
            f"silently bucketing unknown outcomes into OTHER would hide a schema change."
        )


def _reconcile_frame(pa_df: pd.DataFrame) -> pd.DataFrame:
    per_team = pa_df.groupby(["game_pk", "batting_team"], as_index=False)["runs_on_play"].sum()
    finals = pa_df.groupby("game_pk", as_index=False).agg(
        park_id=("park_id", "first"),
        final_home_score=("final_home_score", "first"),
        final_away_score=("final_away_score", "first"),
    )
    merged = per_team.merge(finals, on="game_pk")
    is_home = merged["batting_team"] == merged["park_id"]
    merged["expected"] = merged["final_home_score"].where(is_home, merged["final_away_score"])
    # The one legitimate way PA-attributed runs can fall short of the final:
    # a home walk-off on a non-PA play (wild pitch, passed ball, balk,
    # steal of home). The game ends mid-PA, so exactly one run -- the
    # winning run, scored from a tie -- belongs to no plate appearance.
    merged["non_pa_walkoff"] = (
        is_home
        & (merged["expected"] - merged["runs_on_play"] == 1)
        & (merged["runs_on_play"] == merged["final_away_score"])
    )
    return merged


def reconcile_runs(pa_df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of (game_pk, batting_team) rows whose summed
    `runs_on_play` does not equal that team's final score, excluding
    non-PA walk-offs (see `_reconcile_frame`). An empty frame means every
    run in `pa_df` is attributed to a plate appearance or explained.

    This checks *run attribution to PAs*, not the label itself: both sides
    come from statcast's own score columns. The label is checked against an
    independent source in `check_finals_against_schedule`.

    `park_id` doubles as the home-team code (see module docstring), so
    `batting_team == park_id` identifies the home team's rows without
    needing a separate home_team column in the persisted table.
    """
    merged = _reconcile_frame(pa_df)
    return merged[(merged["runs_on_play"] != merged["expected"]) & ~merged["non_pa_walkoff"]]


def count_non_pa_walkoffs(pa_df: pd.DataFrame) -> int:
    return int(_reconcile_frame(pa_df)["non_pa_walkoff"].sum())


def load_played_schedule(season: int) -> pd.DataFrame:
    """StatsAPI schedule snapshot -> one row per *played* game_pk. A
    suspended-then-resumed game is listed once per date it was on the
    schedule (same game_pk, same final score); dedupe, but refuse to pick
    between conflicting scores.
    """
    path = raw_schedule_season_dir(season) / f"schedule_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python -m basenet.data.ingest --seasons {season}` "
            f"(pulls the StatsAPI schedule used to validate labels)."
        )
    sched = pd.read_parquet(path)
    sched = sched[sched["coded_game_state"].isin(config.PLAYED_GAME_STATES)]
    n_scores = sched.groupby("game_pk")[["home_score", "away_score"]].nunique()
    conflicting = n_scores[(n_scores > 1).any(axis=1)]
    if not conflicting.empty:
        raise RuntimeError(f"[pa] schedule lists conflicting final scores for game_pk(s) "
                           f"{conflicting.index.tolist()[:10]}")
    return sched.drop_duplicates("game_pk", keep="last").reset_index(drop=True)


def check_finals_against_schedule(pa_df: pd.DataFrame, schedule: pd.DataFrame) -> tuple[list[str], int]:
    """Statcast-derived final scores (`statcast_final_*`) vs StatsAPI's
    official ones, plus game coverage in both directions. Returns
    (human-readable problems, n_pitchless_walkoffs); no problems = ok.

    The one tolerated disagreement: a walk-off balk (or similar) scores the
    winning run with no pitch thrown, so statcast has no row for it at all
    and its last score is a tie the official record breaks by exactly one
    home run.
    """
    problems = []
    ours = pa_df.drop_duplicates("game_pk")[["game_pk", "statcast_final_home_score",
                                             "statcast_final_away_score"]].rename(
        columns={"statcast_final_home_score": "final_home_score",
                 "statcast_final_away_score": "final_away_score"})
    merged = ours.merge(schedule[["game_pk", "home_score", "away_score"]], on="game_pk", how="outer",
                        indicator=True)
    only_statcast = merged[merged["_merge"] == "left_only"]["game_pk"].tolist()
    only_schedule = merged[merged["_merge"] == "right_only"]["game_pk"].tolist()
    if only_statcast:
        problems.append(f"{len(only_statcast)} statcast game(s) not played per schedule: {only_statcast[:10]}")
    if only_schedule:
        problems.append(f"{len(only_schedule)} played game(s) missing from statcast: {only_schedule[:10]}")
    both = merged[merged["_merge"] == "both"]
    disagree = (both["final_home_score"] != both["home_score"]) | (both["final_away_score"] != both["away_score"])
    pitchless_walkoff = (
        (both["final_home_score"] == both["final_away_score"])
        & (both["away_score"] == both["final_away_score"])
        & (both["home_score"] == both["final_home_score"] + 1)
    )
    bad = both[disagree & ~pitchless_walkoff]
    if not bad.empty:
        problems.append(
            f"{len(bad)} game(s) whose final score disagrees with StatsAPI:\n"
            + bad[["game_pk", "final_home_score", "home_score", "final_away_score", "away_score"]]
            .head(20).to_string()
        )
    return problems, int((disagree & pitchless_walkoff).sum())


def _log_lineup_size_diagnostic(pa_df: pd.DataFrame) -> None:
    distinct_batters = pa_df.groupby(["game_pk", "batting_team"])["batter_id"].nunique()
    over9 = (distinct_batters > 9).mean()
    print(f"[pa] {over9:.1%} of team-games used more than 9 distinct batters "
          f"(pinch hitters/subs -- expected, informational only)")


def build_pa_table(season: int) -> pd.DataFrame:
    raw_path = str(raw_statcast_season_dir(season) / f"statcast_{season}.parquet")
    con = duckdb.connect()
    _assert_events_covered(con, raw_path)

    sql = f"""
    WITH terminal AS (
        SELECT *
        FROM read_parquet('{raw_path}')
        WHERE events IS NOT NULL
          AND events NOT IN ({", ".join(f"'{e}'" for e in config.DROPPED_EVENTS)})
    ),
    pa AS (
        SELECT
            game_pk,
            game_date,
            season,
            home_team AS park_id,
            home_team,
            away_team,
            inning,
            inning_topbot AS half,
            CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS batting_team,
            CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS pitching_team,
            at_bat_number,
            batter AS batter_id,
            pitcher AS pitcher_id,
            stand AS batter_stands,
            p_throws AS pitcher_throws,
            n_thruorder_pitcher AS times_through_order,
            -- NOT `post_bat_score - bat_score` on this same row: a wild pitch,
            -- passed ball, balk, or stolen base can score a run without ever
            -- producing its own `events` value or a same-row score delta --
            -- observed directly in 2024 data (e.g. game_pk 744809, at_bat 8:
            -- bat_score is already post-run on ITS OWN first pitch, because
            -- the run scored on a pitch delivered during at_bat 7, between
            -- that PA's own last recorded row and this one). The team's own
            -- consecutive terminal-pitch post_bat_score values are still
            -- monotonic and complete, so diffing against the immediately
            -- preceding same-team PA (LAG, defaulting to 0 for a team's
            -- first PA of the game) correctly attributes every run scored,
            -- regardless of mechanism, and is what makes runs reconcile
            -- exactly against the box score.
            post_bat_score - LAG(post_bat_score, 1, 0)
                OVER (PARTITION BY game_pk,
                      CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END
                      ORDER BY at_bat_number)
                AS runs_on_play,
            {_events_case_sql()} AS outcome,
            bb_type,
            woba_value, woba_denom, launch_speed, launch_speed_angle,
            estimated_woba_using_speedangle AS xwoba_contact
        FROM terminal
    ),
    slotted AS (
        SELECT *,
            ((ROW_NUMBER() OVER (PARTITION BY game_pk, batting_team ORDER BY at_bat_number) - 1) % 9) + 1
                AS order_slot
        FROM pa
    ),
    -- every pitch, not just `terminal`: a walk-off on a non-PA play (wild
    -- pitch, balk, ...) ends the game on a pitch with no `events` value.
    finals AS (
        SELECT game_pk,
               MAX(post_home_score) AS final_home_score,
               MAX(post_away_score) AS final_away_score
        FROM read_parquet('{raw_path}')
        GROUP BY game_pk
    ),
    dh AS (
        SELECT DISTINCT game_pk, game_date, home_team, away_team FROM slotted
    ),
    dh_ranked AS (
        SELECT game_pk,
               (DENSE_RANK() OVER (PARTITION BY game_date, home_team, away_team ORDER BY game_pk) = 2)
                   AS is_doubleheader_game_2
        FROM dh
    )
    SELECT
        s.game_pk, s.game_date, s.season, s.park_id, dh_ranked.is_doubleheader_game_2,
        s.inning, s.half, s.batting_team, s.pitching_team,
        s.batter_id, s.pitcher_id, s.batter_stands, s.pitcher_throws,
        s.order_slot, s.times_through_order, s.outcome, s.runs_on_play, s.bb_type,
        s.woba_value, s.woba_denom, s.launch_speed, s.launch_speed_angle, s.xwoba_contact,
        s.at_bat_number, finals.final_home_score, finals.final_away_score,
        s.home_team, s.away_team
    FROM slotted s
    JOIN finals ON finals.game_pk = s.game_pk
    JOIN dh_ranked ON dh_ranked.game_pk = s.game_pk
    ORDER BY s.game_pk, s.at_bat_number
    """
    df = con.execute(sql).df()

    # The label is the OFFICIAL final score; statcast's own is kept only to
    # cross-check it (see check_finals_against_schedule for why they can
    # legitimately differ).
    df = df.rename(columns={"final_home_score": "statcast_final_home_score",
                            "final_away_score": "statcast_final_away_score"})
    schedule = load_played_schedule(season)
    problems, n_pitchless = check_finals_against_schedule(df, schedule)
    if problems:
        raise RuntimeError(
            f"[pa] labels FAILED the StatsAPI cross-check for season {season}:\n" + "\n".join(problems)
        )
    print(f"[pa] {n_pitchless} game(s) ended on a walk-off with no pitch thrown (e.g. balk) -- "
          f"official score used; statcast has no row for that run")
    official = schedule[["game_pk", "home_score", "away_score", "venue_id"]].rename(
        columns={"home_score": "final_home_score", "away_score": "final_away_score"})
    df = df.merge(official, on="game_pk", how="left", validate="many_to_one")
    df[["final_home_score", "final_away_score", "venue_id"]] = \
        df[["final_home_score", "final_away_score", "venue_id"]].astype(int)

    triple_headers = (
        df.drop_duplicates("game_pk")
        .groupby(["game_date", "home_team", "away_team"])["game_pk"]
        .nunique()
    )
    over_two = triple_headers[triple_headers > 2]
    if not over_two.empty:
        print(f"[pa] WARNING: {len(over_two)} team-pair/date group(s) had >2 games "
              f"(doubleheader game_pk-ordinal approximation is unreliable there): "
              f"{over_two.index.tolist()[:5]}")

    _log_lineup_size_diagnostic(df)

    bad = reconcile_runs(df)
    if not bad.empty:
        raise RuntimeError(
            f"[pa] runs reconciliation FAILED for {len(bad)} team-game(s) in season {season}:\n"
            f"{bad[['game_pk', 'batting_team', 'runs_on_play', 'expected']].to_string()}"
        )
    print(f"[pa] {count_non_pa_walkoffs(df)} game(s) ended on a non-PA walk-off "
          f"(winning run on no plate appearance -- expected, rare)")

    return df[PA_COLUMNS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True, choices=config.SEASONS)
    args = parser.parse_args(argv)

    for season in args.seasons:
        with StageTimer(f"pa season={season}").run() as t:
            raw_rows = duckdb.connect().execute(
                f"SELECT COUNT(*) FROM read_parquet('{raw_statcast_season_dir(season)}/*.parquet')"
            ).fetchone()[0]
            t.rows_in = raw_rows
            df = build_pa_table(season)
            t.rows_out = len(df)
            out_dir = ensure_dir(interim_pa_season_dir(season))
            out_path = out_dir / f"pa_{season}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"[pa] wrote {out_path} ({len(df):,} PAs, {df['game_pk'].nunique():,} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
