"""Assemble the final training table: two rows per game (see module-level
note below on why that's not the classic duplicated-row mistake).

    python -m basenet.data.matchups --seasons 2021 2022 2023 2024 2025

Inputs: `data/interim/pa/season=YYYY/*.parquet` and
`data/interim/player_daily/player_daily_YYYY.parquet` for the requested
seasons (player_daily must already cover any earlier seasons needed for its
own trailing windows -- that happened in features.py, not here).

**v1 caveat (repeated from the spec, load-bearing):** lineups and starters
are the *actual* ones that appeared, not lineups as they'd be knowable a few
hours pre-game. This table is for training; it is not what a real
production pre-game inference pipeline would have available at prediction
time.

Row identity: `batting_is_home` distinguishes the two rows a `game_pk`
produces (batting_team == park_id iff home). Both rows are still
statistically coupled (park, weather, umpire, day) -- see
`GROUP_KEY` and `datasets.chronological_split`: splits must group on
`game_pk`, never split its two rows apart.

Two design points worth calling out explicitly (both settled after review):

- **Expected PA per lineup slot is measured, not configured.** It's an
  empirical, leakage-safe quantity computed from our own PA table (see
  `compute_slot_pa_table`) -- it varies by season and run environment, so a
  number typed into a model config would be a guess frozen at authoring
  time. `config.DEFAULT_PA_PER_SLOT` still exists, but only as the
  bootstrap used before enough same-season sample accumulates.
- **The bullpen is a set of individual pitchers, not a pre-aggregated
  composite.** `resolve_bullpen_set` produces `bullpen{k}_*` columns for
  k in 1..config.MAX_BULLPEN_SIZE (mask + IP share + full feature vector
  each), which `datasets.py` stacks with `sp_*` (always index 0) into the
  (N, K, F_p) tensor `PitcherEncoder` actually consumes. This replaced an
  earlier IP-share-weighted composite once the model's actual interface
  turned out to want per-pitcher granularity with masking.
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from . import config
from . import paths
from .features import HITTER_FEATURE_COLUMNS, PITCHER_FEATURE_COLUMNS, PITCHER_STATE_COLUMNS
from .paths import ensure_dir, interim_pa_season_dir, raw_statcast_season_dir, StageTimer

GROUP_KEY = "game_pk"  # never split at row level -- see module docstring

N_SLOTS = config.N_LINEUP_SLOTS


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_pa(seasons: list[int]) -> pd.DataFrame:
    pa_paths = [str(interim_pa_season_dir(s) / f"pa_{s}.parquet") for s in seasons]
    con = duckdb.connect()
    lst = "[" + ", ".join(f"'{p}'" for p in pa_paths) + "]"
    df = con.execute(f"SELECT * FROM read_parquet({lst})").df()
    df["game_date"] = pd.to_datetime(df["game_date"])  # see _load_pitch_counts for why
    return df


def _load_player_daily(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = paths.INTERIM_PLAYER_DAILY_DIR / f"player_daily_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(
            "No player_daily files found. Run `python -m basenet.data.features` first."
        )
    return pd.concat(frames, ignore_index=True)


def _load_pitcher_post_game(seasons: list[int]) -> pd.DataFrame:
    frames = [pd.read_parquet(p) for s in seasons
              if (p := paths.INTERIM_PITCHER_POST_GAME_DIR / f"pitcher_post_game_{s}.parquet").exists()]
    if not frames:
        raise FileNotFoundError(
            "No pitcher_post_game files found. Run `python -m basenet.data.features` first."
        )
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _load_pitch_counts(seasons: list[int]) -> pd.DataFrame:
    """pitcher, game_pk, game_date, n_pitches -- for the bullpen availability heuristic."""
    con = duckdb.connect()
    raw_paths = [str(raw_statcast_season_dir(s) / f"statcast_{s}.parquet") for s in seasons]
    lst = "[" + ", ".join(f"'{p}'" for p in raw_paths) + "]"
    df = con.execute(f"""
        SELECT pitcher AS pitcher_id, game_pk, game_date, COUNT(*) AS n_pitches,
               CASE WHEN inning_topbot='Top' THEN home_team ELSE away_team END AS pitching_team
        FROM read_parquet({lst})
        GROUP BY pitcher, game_pk, game_date, pitching_team
    """).df()
    # normalize to datetime64[ns]: DuckDB may hand back a coarser date unit,
    # and pd.Timedelta arithmetic against that emits a noisy deprecation
    # warning (and would eventually error) further down in this module.
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ---------------------------------------------------------------------------
# Starting lineups / starting pitchers
# ---------------------------------------------------------------------------

def starting_lineups(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batting_team, order_slot): the batter with that
    slot's first plate appearance of the game -- i.e. the actual starter in
    that slot, which (unlike later substitutions) is genuinely knowable
    before first pitch.
    """
    first_pa = pa.sort_values("at_bat_number").drop_duplicates(
        subset=["game_pk", "batting_team", "order_slot"], keep="first"
    )
    return first_pa[["game_pk", "batting_team", "order_slot", "batter_id",
                      "batter_stands", "game_date"]]


def starting_pitchers(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, pitching_team): the pitcher of record for that
    team's first plate appearance of the game.
    """
    first_pa = pa.sort_values("at_bat_number").drop_duplicates(
        subset=["game_pk", "pitching_team"], keep="first"
    )
    return first_pa[["game_pk", "pitching_team", "pitcher_id", "pitcher_throws", "game_date"]]


def _incomplete_lineup_game_pks(lineups: pd.DataFrame) -> set[int]:
    counts = lineups.groupby(["game_pk", "batting_team"])["order_slot"].nunique()
    bad = counts[counts < N_SLOTS]
    if not bad.empty:
        print(f"[matchups] dropping {bad.index.get_level_values('game_pk').nunique()} "
              f"team-game(s) with fewer than {N_SLOTS} distinct starting slots recorded "
              f"(rain-shortened / data gaps)")
    return set(bad.index.get_level_values("game_pk"))


# ---------------------------------------------------------------------------
# Expected PA per lineup slot -- measured from our own PA table, not a fixed
# guess (see config.DEFAULT_PA_PER_SLOT's docstring on why this matters).
# ---------------------------------------------------------------------------

def compute_slot_pa_table(pa: pd.DataFrame) -> pd.DataFrame:
    """(season, game_date, slot) -> leakage-safe expected PA for that slot:
    the running mean PAs-per-team-game at that slot, through the end of the
    PRIOR calendar date (same day-granularity, leakage-safe pattern as
    features.py's league means -- see that module's docstring for why day-
    level, not per-PA, granularity is the right tradeoff here too). Falls
    back to config.DEFAULT_PA_PER_SLOT until config.SLOT_PA_MIN_TEAM_GAMES
    of real same-season sample has accumulated.
    """
    con = duckdb.connect()
    con.register("pa", pa)
    sql = """
    WITH slot_game AS (
        SELECT game_pk, batting_team, order_slot AS slot, game_date, season,
               COUNT(*) AS pa_count
        FROM pa
        GROUP BY game_pk, batting_team, order_slot, game_date, season
    ),
    daily AS (
        SELECT season, game_date, slot, SUM(pa_count) AS total_pa, COUNT(*) AS n_team_games
        FROM slot_game
        GROUP BY season, game_date, slot
    )
    SELECT season, game_date, slot,
        COALESCE(SUM(total_pa) OVER (PARTITION BY season, slot ORDER BY game_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS cum_pa,
        COALESCE(SUM(n_team_games) OVER (PARTITION BY season, slot ORDER BY game_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS cum_team_games
    FROM daily
    ORDER BY season, slot, game_date
    """
    df = con.execute(sql).df()

    def _resolve(row) -> float:
        if row["cum_team_games"] >= config.SLOT_PA_MIN_TEAM_GAMES:
            return row["cum_pa"] / row["cum_team_games"]
        return config.DEFAULT_PA_PER_SLOT[int(row["slot"])]

    df["expected_pa"] = df.apply(_resolve, axis=1)
    return df[["season", "game_date", "slot", "expected_pa"]]


# ---------------------------------------------------------------------------
# Bullpen availability + composite (interface open for a learned IP policy)
# ---------------------------------------------------------------------------

def estimate_bullpen_ip_shares(
    available_ids: list[int], total_ip: float = config.BULLPEN_EXPECTED_TOTAL_IP
) -> dict[int, float]:
    """v1 heuristic: split `total_ip` uniformly across available relievers.
    Swap this out for a learned policy later without touching callers --
    the interface (list of ids in, {id: ip_share} out) stays the same.
    """
    if not available_ids:
        return {}
    share = total_ip / len(available_ids)
    return {pid: share for pid in available_ids}


def bullpen_candidates(
    pa: pd.DataFrame, pitch_counts: pd.DataFrame, target_games: pd.DataFrame
) -> pd.DataFrame:
    """For each (game_pk, pitching_team, game_date) in `target_games`, the
    list of relievers deemed available: pitched for that team within the
    trailing roster-lookback window, is not that game's starter, and hasn't
    exceeded the pitch-count availability thresholds in the trailing 1-2
    days. See config.BULLPEN_* for the tunable constants and the module
    docstring for the roster-membership caveat.

    `bullpen_ids` is ordered most- to least-recently-used (within the
    lookback window) and truncated to config.MAX_BULLPEN_SIZE -- real MLB
    bullpens run ~5-8 arms, so this order is a *deterministic* tiebreak for
    which pitchers survive truncation, not a claim that recency is
    meaningful to the (permutation-invariant) pooling that consumes them.
    `n_candidates` (pre-truncation) is carried through so truncation
    frequency is visible in the output, not just in the printed summary
    below.
    """
    sp = starting_pitchers(pa)[["game_pk", "pitching_team", "pitcher_id"]].rename(
        columns={"pitcher_id": "sp_id"}
    )
    tg = target_games.merge(sp, on=["game_pk", "pitching_team"], how="left")

    out_rows = []
    n_truncated = 0
    n_starters_excluded = 0
    starts = sp.rename(columns={"sp_id": "pitcher_id"}).assign(is_start=True)
    pc = pitch_counts.merge(starts, on=["game_pk", "pitching_team", "pitcher_id"], how="left")
    pc["is_start"] = pc["is_start"].eq(True)  # NaN (not a start) -> False
    pc = pc.sort_values("game_date")
    # `unit="D"` (rather than the `days=` keyword) sidesteps a pandas/numpy
    # deprecation warning about "generic"-unit timedeltas on this stack;
    # semantics are identical.
    lookback = pd.Timedelta(config.BULLPEN_ROSTER_LOOKBACK_DAYS, unit="D")
    one_day = pd.Timedelta(1, unit="D")
    two_days = pd.Timedelta(2, unit="D")
    for row in tg.itertuples(index=False):
        team_hist = pc[
            (pc.pitching_team == row.pitching_team)
            & (pc.game_date < row.game_date)
            & (pc.game_date >= row.game_date - lookback)
        ]
        # Rotation starters pitched for this team recently too, but they
        # aren't bullpen arms: drop anyone whose appearances in the window
        # were mostly starts. Openers / swingmen (mostly relief) stay.
        start_share = team_hist.groupby("pitcher_id")["is_start"].mean()
        starters = set(start_share[start_share >= config.BULLPEN_STARTER_SHARE_EXCLUDE].index)
        n_starters_excluded += len(starters - {row.sp_id})
        candidate_ids = set(team_hist.pitcher_id) - {row.sp_id} - starters
        avail = []
        for pid in candidate_ids:
            recent = pc[(pc.pitcher_id == pid) & (pc.game_date < row.game_date)]
            p1d = recent[recent.game_date >= row.game_date - one_day]["n_pitches"].sum()
            p2d = recent[recent.game_date >= row.game_date - two_days]["n_pitches"].sum()
            if p1d <= config.BULLPEN_MAX_PITCHES_1D and p2d <= config.BULLPEN_MAX_PITCHES_2D:
                avail.append(pid)

        last_seen = team_hist[team_hist["pitcher_id"].isin(avail)].groupby("pitcher_id")["game_date"].max()
        avail_sorted = sorted(avail, key=lambda pid: last_seen[pid], reverse=True)
        if len(avail_sorted) > config.MAX_BULLPEN_SIZE:
            n_truncated += 1
        bullpen_ids = avail_sorted[: config.MAX_BULLPEN_SIZE]

        out_rows.append({"game_pk": row.game_pk, "pitching_team": row.pitching_team,
                          "bullpen_ids": bullpen_ids, "n_candidates": len(avail_sorted)})

    result = pd.DataFrame(out_rows)
    if len(result):
        rate = n_truncated / len(result)
        print(f"[matchups] bullpen candidate pool exceeded MAX_BULLPEN_SIZE="
              f"{config.MAX_BULLPEN_SIZE} in {n_truncated}/{len(result)} team-games "
              f"({rate:.1%}) -- kept the {config.MAX_BULLPEN_SIZE} most recently used; "
              f"median pool {result['n_candidates'].median():.0f}; "
              f"{n_starters_excluded / len(result):.1f} rotation starter(s)/team-game excluded")
    return result


# ---------------------------------------------------------------------------
# Feature resolution
# ---------------------------------------------------------------------------

_DAY_NS = 24 * 3600 * 1_000_000_000


def _sort_key(game_date: pd.Series, game_pk: pd.Series) -> pd.Series:
    """(game_date, game_pk) as one int64 -- date dominates, game_pk breaks
    same-day ties (the pipeline-wide ordering convention) -- so "strictly
    before this game" is a single merge_asof key."""
    return pd.to_datetime(game_date).astype("int64") + game_pk.astype("int64") % _DAY_NS

def _resolve_exact(ids_df: pd.DataFrame, id_col: str, player_daily: pd.DataFrame,
                    role: str, feature_cols: list[str], prefix: str) -> pd.DataFrame:
    """Equi-join on (player_id, role, game_date, game_pk) -- valid because
    every id here is guaranteed to have actually appeared in that exact
    game (starting lineup batters, starting pitchers).
    """
    pd_role = player_daily[player_daily["role"] == role]
    merged = ids_df.merge(
        pd_role, left_on=[id_col, "game_date", "game_pk"],
        right_on=["player_id", "game_date", "game_pk"], how="left",
    )
    missing = merged[feature_cols[0]].isna().sum()
    if missing:
        raise RuntimeError(
            f"[matchups] {missing} {role} row(s) failed to resolve pre-game features via "
            f"exact (player_id, game_date, game_pk) join -- player_daily is out of sync with pa."
        )
    return merged.rename(columns={c: f"{prefix}{c}" for c in feature_cols})


def _asof_prior_state(targets: pd.DataFrame, post_game: pd.DataFrame) -> pd.DataFrame:
    """For each (pitcher_id, game_date, game_pk) target, that pitcher's
    post-game state (PITCHER_STATE_COLUMNS) from his latest appearance
    strictly before the target game -- i.e. his state going into it. See
    features.build_pitcher_post_game. Returns rows aligned to `targets`.
    """
    t = targets[["pitcher_id", "game_date", "game_pk"]].copy()
    t["_order"] = np.arange(len(t))
    t["sort_key"] = _sort_key(t["game_date"], t["game_pk"])
    s = post_game[["player_id", "game_date", "game_pk"] + PITCHER_STATE_COLUMNS].copy()
    s["sort_key"] = _sort_key(s["game_date"], s["game_pk"])
    merged = pd.merge_asof(
        t.sort_values("sort_key"), s.drop(columns=["game_date", "game_pk"]).sort_values("sort_key"),
        on="sort_key", left_by="pitcher_id", right_by="player_id",
        direction="backward", allow_exact_matches=False,
    ).sort_values("_order")
    return merged[PITCHER_STATE_COLUMNS].reset_index(drop=True)


def reliever_fatigue(targets: pd.DataFrame, pitch_counts: pd.DataFrame) -> pd.DataFrame:
    """PITCHER_FATIGUE_COLUMNS for each (pitcher_id, game_date, game_pk)
    target, relative to the TARGET game -- same definitions (and the same
    debut defaults) as features._pitch_recency_features uses for starters:
    last outing = latest appearance strictly before in (game_date, game_pk)
    order; trailing 3d = pitches on the 3 calendar days before game_date.
    """
    t = targets[["pitcher_id", "game_date", "game_pk"]].copy()
    t["game_date"] = pd.to_datetime(t["game_date"])
    t["_order"] = np.arange(len(t))
    t["sort_key"] = _sort_key(t["game_date"], t["game_pk"])
    pc = pitch_counts[["pitcher_id", "game_date", "game_pk", "n_pitches"]].copy()
    pc["sort_key"] = _sort_key(pc["game_date"], pc["game_pk"])
    last = pd.merge_asof(
        t.sort_values("sort_key"),
        pc.rename(columns={"game_date": "prev_date"}).drop(columns="game_pk").sort_values("sort_key"),
        on="sort_key", by="pitcher_id", direction="backward", allow_exact_matches=False,
    ).sort_values("_order").reset_index(drop=True)

    con = duckdb.connect()
    con.register("t", t[["_order", "pitcher_id", "game_date"]])
    con.register("pc", pc[["pitcher_id", "game_date", "n_pitches"]])
    trailing = con.execute("""
        SELECT t._order, COALESCE(SUM(pc.n_pitches), 0) AS pitch_count_trailing_3d
        FROM t LEFT JOIN pc
          ON pc.pitcher_id = t.pitcher_id
         AND pc.game_date >= t.game_date - INTERVAL 3 DAY
         AND pc.game_date < t.game_date
        GROUP BY t._order ORDER BY t._order
    """).df()

    return pd.DataFrame({
        "days_rest": (last["game_date"] - last["prev_date"]).dt.days
                     .fillna(config.DEFAULT_DAYS_REST).to_numpy(dtype=float),
        "pitch_count_last_outing": last["n_pitches"].fillna(0).to_numpy(dtype=float),
        "pitch_count_trailing_3d": trailing["pitch_count_trailing_3d"].to_numpy(dtype=float),
    })


def resolve_bullpen_set(bullpen_df: pd.DataFrame, pitcher_post_game: pd.DataFrame,
                         pitch_counts: pd.DataFrame, throws_map: pd.Series) -> pd.DataFrame:
    """One row per (game_pk, pitching_team), with `bullpen{k}_*` columns for
    k in 1..config.MAX_BULLPEN_SIZE: pitcher_id, mask (real vs padding),
    ip_share, throws, and every PITCHER_FEATURE_COLUMNS entry, individually
    -- this is the pitcher-SET representation `datasets.py` stacks (with
    `sp_*`, always index 0) into the (N, K, F_p) tensor PitcherEncoder
    expects, rather than pre-collapsing the bullpen into one composite
    vector. Padding slots (mask=False) get finite (never NaN) 0.0 features
    -- their value is irrelevant once multiplied by a zero mask*weight
    downstream, but NaN would poison the product (0 * NaN = NaN) instead of
    zeroing it.

    Each reliever's features describe him going INTO the target game: state
    from his latest prior post-game row, fatigue recomputed against the
    target date. (An earlier version read his latest prior *pre-game* row,
    i.e. his state before his last outing -- one appearance stale, with
    days_rest/pitch counts describing the wrong day.)
    """
    K = config.MAX_BULLPEN_SIZE
    long = bullpen_df[["game_pk", "pitching_team", "game_date", "bullpen_ids"]].explode("bullpen_ids")
    long = long.dropna(subset=["bullpen_ids"]).rename(columns={"bullpen_ids": "pitcher_id"})
    long["pitcher_id"] = long["pitcher_id"].astype("int64")
    long["game_date"] = pd.to_datetime(long["game_date"])
    long = long.reset_index(drop=True)
    long["k"] = long.groupby(["game_pk", "pitching_team"]).cumcount() + 1

    state = _asof_prior_state(long, pitcher_post_game)
    missing = state[PITCHER_STATE_COLUMNS[0]].isna()
    if missing.any():
        raise RuntimeError(
            f"[matchups] {int(missing.sum())} bullpen candidate(s) have no prior post-game state -- "
            f"pitcher_post_game is out of sync with the raw pitch counts."
        )
    feats = pd.concat([state, reliever_fatigue(long, pitch_counts)], axis=1)
    long = pd.concat([long, feats[PITCHER_FEATURE_COLUMNS].astype(float)], axis=1)

    shares = {
        (r.game_pk, r.pitching_team): estimate_bullpen_ip_shares(list(r.bullpen_ids))
        for r in bullpen_df.itertuples(index=False)
    }
    long["ip_share"] = [shares[(g, t)][p] for g, t, p in zip(long.game_pk, long.pitching_team, long.pitcher_id)]
    long["throws"] = long["pitcher_id"].map(throws_map).fillna("R")

    out = bullpen_df[["game_pk", "pitching_team"]].copy()
    for k in range(1, K + 1):
        prefix = f"bullpen{k}_"
        sub = long[long["k"] == k][["game_pk", "pitching_team", "pitcher_id", "ip_share", "throws"]
                                   + PITCHER_FEATURE_COLUMNS]
        sub = sub.rename(columns={c: f"{prefix}{c}" for c in sub.columns if c not in ("game_pk", "pitching_team")})
        out = out.merge(sub, on=["game_pk", "pitching_team"], how="left", validate="one_to_one")
        out[f"{prefix}mask"] = out[f"{prefix}pitcher_id"].notna()
        out[f"{prefix}pitcher_id"] = out[f"{prefix}pitcher_id"].fillna(-1).astype("int64")
        out[f"{prefix}ip_share"] = out[f"{prefix}ip_share"].fillna(0.0)
        out[f"{prefix}throws"] = out[f"{prefix}throws"].fillna("R")  # arbitrary: masked out downstream
        out[[f"{prefix}{c}" for c in PITCHER_FEATURE_COLUMNS]] = \
            out[[f"{prefix}{c}" for c in PITCHER_FEATURE_COLUMNS]].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_matchups(seasons: list[int]) -> pd.DataFrame:
    pa = _load_pa(seasons)
    player_daily = _load_player_daily(seasons)
    pitch_counts = _load_pitch_counts(seasons)

    lineups = starting_lineups(pa)
    bad_pks = _incomplete_lineup_game_pks(lineups)
    lineups = lineups[~lineups["game_pk"].isin(bad_pks)]
    pitchers = starting_pitchers(pa)
    pitchers = pitchers[~pitchers["game_pk"].isin(bad_pks)]

    games = pa[~pa["game_pk"].isin(bad_pks)].drop_duplicates("game_pk")[
        ["game_pk", "game_date", "season", "park_id", "final_home_score",
         "final_away_score", "is_doubleheader_game_2", "venue_id"]
    ]

    # derive both team codes per game once: park_id is the home team by
    # construction (see pa.py), the other team is whichever of
    # batting_team/pitching_team isn't park_id.
    team_codes = pa[~pa["game_pk"].isin(bad_pks)].groupby("game_pk").apply(
        lambda g: pd.Series({
            "home_team": g["park_id"].iloc[0],
            "away_team": ((set(g["batting_team"]) | set(g["pitching_team"])) - {g["park_id"].iloc[0]}).pop(),
        }),
        include_groups=False,
    ).reset_index()
    games = games.merge(team_codes, on="game_pk")

    matchup_rows = []
    for is_home in (True, False):
        side = games.copy()
        side["batting_team"] = side["home_team"] if is_home else side["away_team"]
        side["pitching_team"] = side["away_team"] if is_home else side["home_team"]
        side["batting_is_home"] = is_home
        side["runs_scored"] = side["final_home_score"] if is_home else side["final_away_score"]
        matchup_rows.append(side)
    long = pd.concat(matchup_rows, ignore_index=True)
    long["runs_scored"] = np.where(long["batting_is_home"], long["final_home_score"], long["final_away_score"])

    # --- lineup: pivot starting_lineups to slot1..slot9 columns, then resolve features ---
    lu = lineups.rename(columns={"batting_team": "_bt"})
    base = long[["game_pk", "batting_team", "pitching_team", "batting_is_home", "game_date",
                 "season", "park_id", "venue_id", "runs_scored", "is_doubleheader_game_2"]].copy()

    slot_pa_table = compute_slot_pa_table(pa)
    base_season_date = base[["game_pk", "game_date", "season"]]

    per_slot_frames = []
    for slot in range(1, N_SLOTS + 1):
        slot_lu = lu[lu["order_slot"] == slot]
        merged = base.merge(
            slot_lu[["game_pk", "_bt", "batter_id", "batter_stands", "game_date"]],
            left_on=["game_pk", "batting_team", "game_date"],
            right_on=["game_pk", "_bt", "game_date"], how="left",
        )
        resolved = _resolve_exact(
            merged[["game_pk", "batting_team", "game_date", "batter_id", "batter_stands"]],
            "batter_id", player_daily, "hitter", HITTER_FEATURE_COLUMNS, f"slot{slot}_",
        )
        resolved = resolved.rename(columns={"batter_id": f"slot{slot}_batter_id",
                                             "batter_stands": f"slot{slot}_stands"})
        # empirical, leakage-safe expected PA for THIS slot/season/date --
        # not the static config.DEFAULT_PA_PER_SLOT bootstrap (see
        # compute_slot_pa_table). game_pk uniquely determines (season,
        # game_date), so joining through it is a proper keyed merge rather
        # than trusting two independently-built frames to stay row-aligned.
        slot_pa_for_slot = slot_pa_table[slot_pa_table["slot"] == slot][["season", "game_date", "expected_pa"]]
        # dedupe to one row per game_pk first: `base` has 2 rows/game_pk
        # (home+away sides) sharing the same (season, game_date), and a
        # non-unique right side would fan out the next merge into
        # duplicate rows.
        game_pk_expected_pa = base_season_date.drop_duplicates("game_pk").merge(
            slot_pa_for_slot, on=["season", "game_date"], how="left"
        )[["game_pk", "expected_pa"]].fillna({"expected_pa": config.DEFAULT_PA_PER_SLOT[slot]})
        resolved = resolved.merge(game_pk_expected_pa, on="game_pk", how="left")
        resolved = resolved.rename(columns={"expected_pa": f"slot{slot}_expected_pa"})
        keep_cols = [f"slot{slot}_batter_id", f"slot{slot}_stands", f"slot{slot}_expected_pa"] + \
                    [f"slot{slot}_{c}" for c in HITTER_FEATURE_COLUMNS]
        per_slot_frames.append(resolved[keep_cols])

    result = base.reset_index(drop=True)
    for frame in per_slot_frames:
        result = pd.concat([result, frame.reset_index(drop=True)], axis=1)

    # --- starting pitcher ---
    sp_merged = base.merge(
        pitchers.rename(columns={"pitching_team": "_pt"})[["game_pk", "_pt", "pitcher_id", "pitcher_throws", "game_date"]],
        left_on=["game_pk", "pitching_team", "game_date"],
        right_on=["game_pk", "_pt", "game_date"], how="left",
    )
    sp_resolved = _resolve_exact(
        sp_merged[["game_pk", "pitching_team", "game_date", "pitcher_id", "pitcher_throws"]],
        "pitcher_id", player_daily, "pitcher", PITCHER_FEATURE_COLUMNS, "sp_",
    )
    sp_resolved = sp_resolved.rename(columns={"pitcher_id": "sp_id", "pitcher_throws": "sp_throws"})
    sp_cols = ["sp_id", "sp_throws"] + [f"sp_{c}" for c in PITCHER_FEATURE_COLUMNS]
    result = pd.concat([result, sp_resolved[sp_cols].reset_index(drop=True)], axis=1)

    # --- bullpen: individual masked/weighted pitcher slots, not a composite
    # (see resolve_bullpen_set docstring -- this is the set PitcherEncoder
    # actually expects) ---
    target_games = base[["game_pk", "pitching_team", "game_date"]].drop_duplicates()
    bp = bullpen_candidates(pa, pitch_counts, target_games)
    throws_map = pa.drop_duplicates("pitcher_id").set_index("pitcher_id")["pitcher_throws"]
    bp_resolved = resolve_bullpen_set(
        bp.merge(target_games, on=["game_pk", "pitching_team"]), _load_pitcher_post_game(seasons),
        pitch_counts, throws_map,
    )
    bullpen_cols = []
    for k in range(1, config.MAX_BULLPEN_SIZE + 1):
        bullpen_cols += [f"bullpen{k}_pitcher_id", f"bullpen{k}_mask",
                         f"bullpen{k}_ip_share", f"bullpen{k}_throws"]
        bullpen_cols += [f"bullpen{k}_{c}" for c in PITCHER_FEATURE_COLUMNS]
    result = result.merge(
        bp_resolved[["game_pk", "pitching_team"] + bullpen_cols],
        on=["game_pk", "pitching_team"], how="left",
    )
    # traceability, parallel to lineup_player_ids below -- real (unpadded) ids only.
    result = result.merge(
        bp[["game_pk", "pitching_team", "bullpen_ids", "n_candidates"]].rename(
            columns={"bullpen_ids": "bullpen_pitcher_ids"}
        ),
        on=["game_pk", "pitching_team"], how="left",
    )

    lineup_id_cols = [f"slot{s}_batter_id" for s in range(1, N_SLOTS + 1)]
    result["lineup_player_ids"] = result[lineup_id_cols].values.tolist()

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True, choices=config.SEASONS)
    args = parser.parse_args(argv)

    for season in args.seasons:
        with StageTimer(f"matchups season={season}").run() as t:
            df = build_matchups([season])
            t.rows_in = df["game_pk"].nunique()
            t.rows_out = len(df)
            out_dir = ensure_dir(paths.PROCESSED_MATCHUPS_DIR)
            out_path = out_dir / f"matchups_{season}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"[matchups] wrote {out_path} ({len(df):,} rows, "
                  f"{df['game_pk'].nunique():,} games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
