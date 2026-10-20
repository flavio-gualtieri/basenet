"""Rolling, leakage-safe player feature tables.

    python -m basenet.data.features --seasons 2021 2022 2023 2024 2025

Inputs:
  - every `data/interim/pa/season=YYYY/*.parquet` found on disk (not just the
    requested --seasons): trailing-300 windows must be able to see across a
    season boundary (an April game's "trailing 300 PA" mostly consists of
    PAs from the previous September), so this stage always builds off the
    full available PA history and simply *writes* one output file per
    requested season.
  - `data/raw/statcast/season=YYYY/*.parquet` directly, for two pitch-level
    aggregates the PA table cannot carry at PA granularity: pitcher
    contact%/whiff% (need every pitch of a plate appearance, not just the
    terminal one) and pitch counts per outing. This is the one place outside
    ingest.py that reads raw pitch data, and it is still a pure function of
    the immutable raw snapshot -- no network.

Leakage contract
-----------------
`player_daily` is keyed on (player_id, role, game_date, game_pk) [role added
beyond the natural (player_id, game_date, game_pk) key so two-way
players -- Ohtani pitches and hits in the same seasons we're ingesting -- get
independent hitter and pitcher rows instead of colliding]. Every stat stored
for a row is computed using ONLY PAs/pitches strictly before that row's
(game_date, game_pk) in the ordering key (game_date, game_pk, at_bat_number).
Concretely: we compute cumulative windows at PA granularity ordered by that
key, then keep only the first PA of each player-game, whose "N PRECEDING"
window by construction stops at the previous game and touches none of the
PAs in the current one.

Because matchups.py (v1) only ever needs a player's features for a game he
is *known* to have appeared in (the real lineup/pitchers who actually
played), resolving "features as of before this game" is a plain equi-join of
`matchups` onto `player_daily` on (player_id, role, game_date, game_pk) --
no separate "most recent prior row" lookup is required.

Shrinkage target: player rates are shrunk toward the *running, leakage-safe*
season-to-date league mean for that stat/role, computed through the end of
the prior calendar date (see `_daily_league_means`). Using a whole-season
league constant would itself leak future-season information into early-
season games; day-level (not per-PA) granularity for the league prior is a
deliberate, documented simplification -- see module docstring in DATA.md.
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd

from . import config
from . import paths
from .paths import ensure_dir, StageTimer

ORDER_KEY = "game_date, game_pk, at_bat_number"

# Non-id columns present on every `player_daily` row, per role -- exported so
# matchups.py / datasets.py build their column lists off these instead of
# duplicating them (see _finalize_hitter_daily / _finalize_pitcher_daily).
HITTER_BASE_STATS = ("k_pct", "bb_pct", "hbp_pct", "x1b_rate", "x2b_rate",
                     "x3b_rate", "hr_rate", "babip", "gb_pct", "fb_pct")
PITCHER_BASE_STATS = ("k_pct", "bb_pct", "hr_per_bf", "gb_pct", "contact_pct", "whiff_pct")
# Quality of contact (for hitters: produced; for pitchers: allowed) -- the
# "contact" feature group (see datasets.FEATURE_GROUPS).
CONTACT_STATS = ("xwoba", "hard_hit_pct", "barrel_pct")

HITTER_CONTACT_COLUMNS: list[str] = [f"{w}_{s}" for w in ("std", "trailing") for s in CONTACT_STATS]
PITCHER_CONTACT_COLUMNS: list[str] = list(HITTER_CONTACT_COLUMNS)

HITTER_FEATURE_COLUMNS: list[str] = (
    ["std_pa", "trailing_pa", "is_cold_start", "days_since_last_game"]
    + [f"std_{s}" for s in HITTER_BASE_STATS]
    + [f"trailing_{s}" for s in HITTER_BASE_STATS]
    + HITTER_CONTACT_COLUMNS
)
PITCHER_FEATURE_COLUMNS: list[str] = (
    ["std_bf", "trailing_bf", "is_cold_start", "days_rest",
     "pitch_count_last_outing", "pitch_count_trailing_3d"]
    + [f"std_{s}" for s in PITCHER_BASE_STATS]
    + [f"trailing_{s}" for s in PITCHER_BASE_STATS]
    + PITCHER_CONTACT_COLUMNS
)

# Platoon splits ("platoon" group): rates vs each opposing hand, over every
# earlier PA in the ingested window, shrunk toward the player's own overall
# rate (see _attach_platoon). Stored for both hands; matchups.py picks the
# one that applies to each matchup (hitter vs today's starter's hand, starter
# vs each hitter's side). Not part of *_FEATURE_COLUMNS: they're resolved
# per matchup, not per player.
HITTER_PLATOON_STATS = ("k_pct", "bb_pct", "hr_rate", "xwoba")
PITCHER_PLATOON_STATS = ("k_pct", "bb_pct", "hr_per_bf", "xwoba")
HITTER_PLATOON_COLUMNS: list[str] = [f"plat_vs_{h}_{s}" for h in ("l", "r") for s in HITTER_PLATOON_STATS]
PITCHER_PLATOON_COLUMNS: list[str] = [f"plat_vs_{h}_{s}" for h in ("l", "r") for s in PITCHER_PLATOON_STATS]

# What matchups.resolve_matchup_platoon leaves per lineup slot (slot{k}_...):
# the hitter's split vs today's starter's hand, then the starter's vs his side.
SLOT_PLATOON_COLUMNS: list[str] = ([f"plat_{s}" for s in HITTER_PLATOON_STATS]
                                   + [f"sp_plat_{s}" for s in PITCHER_PLATOON_STATS])

# Starter workload ("workload" group), pitcher rows only: see _attach_workload.
PITCHER_WORKLOAD_COLUMNS: list[str] = ["n_recent_starts", "bf_per_start"]
# Fatigue/recency features are relative to the *target* game's date, so for
# a reliever who isn't pitching in that game they can't be read off any
# stored row -- matchups.py recomputes them per target game (same
# definitions as `_pitch_recency_features`). Everything else is "state".
PITCHER_FATIGUE_COLUMNS: list[str] = ["days_rest", "pitch_count_last_outing", "pitch_count_trailing_3d"]
PITCHER_STATE_COLUMNS: list[str] = [c for c in PITCHER_FEATURE_COLUMNS if c not in PITCHER_FATIGUE_COLUMNS]

# (numerator indicator column, denominator indicator column) per stat, in
# terms of the per-PA fact columns built by `_build_fact_table`.
HITTER_STAT_SPECS: dict[str, tuple[str, str]] = {
    "k_pct": ("k", "pa"),
    "bb_pct": ("bb", "pa"),
    "hbp_pct": ("hbp", "pa"),
    "x1b_rate": ("x1b", "pa"),
    "x2b_rate": ("x2b", "pa"),
    "x3b_rate": ("x3b", "pa"),
    "hr_rate": ("hr", "pa"),
    "babip": ("babip_hit", "babip_bip"),
    "gb_pct": ("gb", "bip_typed"),
    "fb_pct": ("fb", "bip_typed"),
    "xwoba": ("xwoba_num", "woba_den"),
    "hard_hit_pct": ("hard_hit", "ev_bip"),
    "barrel_pct": ("barrel", "ev_bip"),
}
PITCHER_STAT_SPECS: dict[str, tuple[str, str]] = {
    "k_pct": ("k", "bf"),
    "bb_pct": ("bb", "bf"),
    "hr_per_bf": ("hr", "bf"),
    "gb_pct": ("gb", "bip_typed"),
    "contact_pct": ("contact", "swings"),
    "whiff_pct": ("whiffs", "swings"),
    "xwoba": ("xwoba_num", "woba_den"),
    "hard_hit_pct": ("hard_hit", "ev_bip"),
    "barrel_pct": ("barrel", "ev_bip"),
}

HITTER_INDICATOR_SQL: dict[str, str] = {
    "k": "CASE WHEN outcome='K' THEN 1 ELSE 0 END",
    "bb": "CASE WHEN outcome='BB' THEN 1 ELSE 0 END",
    "hbp": "CASE WHEN outcome='HBP' THEN 1 ELSE 0 END",
    "x1b": "CASE WHEN outcome='1B' THEN 1 ELSE 0 END",
    "x2b": "CASE WHEN outcome='2B' THEN 1 ELSE 0 END",
    "x3b": "CASE WHEN outcome='3B' THEN 1 ELSE 0 END",
    "hr": "CASE WHEN outcome='HR' THEN 1 ELSE 0 END",
    # BABIP excludes HR from both hits and the ball-in-play denominator.
    "babip_hit": "CASE WHEN outcome IN ('1B','2B','3B') THEN 1 ELSE 0 END",
    "babip_bip": "CASE WHEN outcome IN ('1B','2B','3B','OUT_IP','ROE') THEN 1 ELSE 0 END",
    "bip_typed": "CASE WHEN bb_type IS NOT NULL THEN 1 ELSE 0 END",
    "gb": "CASE WHEN bb_type='ground_ball' THEN 1 ELSE 0 END",
    "fb": "CASE WHEN bb_type='fly_ball' THEN 1 ELSE 0 END",
    # xwOBA: statcast's expected wOBA from exit velocity + launch angle on
    # balls in play, the actual wOBA value otherwise (K, BB, HBP -- and the
    # rare batted ball with no tracked EV), over wOBA's own denominator
    # (excludes sac bunts, catcher's interference, ...).
    "xwoba_num": ("CASE WHEN woba_denom = 1 THEN COALESCE("
                  "CASE WHEN bb_type IS NOT NULL THEN xwoba_contact END, woba_value, 0) ELSE 0 END"),
    "woba_den": "COALESCE(woba_denom, 0)",
    "ev_bip": "CASE WHEN bb_type IS NOT NULL AND launch_speed IS NOT NULL THEN 1 ELSE 0 END",
    "hard_hit": f"CASE WHEN bb_type IS NOT NULL AND launch_speed >= {config.HARD_HIT_MPH} THEN 1 ELSE 0 END",
    "barrel": (f"CASE WHEN bb_type IS NOT NULL AND launch_speed_angle = {config.BARREL_LAUNCH_SPEED_ANGLE} "
               f"THEN 1 ELSE 0 END"),
}
# Pitcher PA-level indicators are a subset of the same outcome vocabulary,
# just grouped by pitcher_id instead of batter_id.
PITCHER_INDICATOR_SQL: dict[str, str] = {
    name: HITTER_INDICATOR_SQL[name]
    for name in ("k", "bb", "hr", "bip_typed", "gb", "xwoba_num", "woba_den", "ev_bip", "hard_hit", "barrel")
}


# ---------------------------------------------------------------------------
# Loading raw / PA data
# ---------------------------------------------------------------------------

def _available_pa_paths() -> list[str]:
    return sorted(str(p) for p in paths.INTERIM_PA_DIR.glob("season=*/*.parquet"))


def _pa_paths_sql_list(pa_paths: list[str]) -> str:
    return "[" + ", ".join(f"'{p}'" for p in pa_paths) + "]"


def _con_with_pa(con: duckdb.DuckDBPyConnection) -> None:
    pa_paths = _available_pa_paths()
    if not pa_paths:
        raise FileNotFoundError(
            "No PA parquet files found under data/interim/pa/. Run `python -m basenet.data.pa` first."
        )
    con.execute(f"CREATE OR REPLACE VIEW pa AS SELECT * FROM read_parquet({_pa_paths_sql_list(pa_paths)})")


# ---------------------------------------------------------------------------
# Per-PA fact tables + cumulative windows
# ---------------------------------------------------------------------------

def _window_expr(indicator: str, frame: str, extra_partition: str = "") -> str:
    # COALESCE: SQL SUM() over an empty window frame (a player's/league's
    # very first row) is NULL, not 0 -- e.g. a season-opener has no prior
    # rows for the STD window. Every count in this pipeline means "0 seen
    # so far", never "unknown", so it must not propagate as NaN.
    return f"COALESCE(SUM({indicator}) OVER (PARTITION BY player_id{extra_partition} ORDER BY {ORDER_KEY} {frame}), 0) "


def _window_frames(trailing_n: int, post_game: bool) -> tuple[str, str, str]:
    """(std_frame, trailing_frame, in-game row order) for one row per
    player-game. Pre-game (default): read the window at the game's FIRST PA,
    excluding it -- touches only earlier games. Post-game: read it at the
    game's LAST PA, including it -- the player's state after that game,
    which is his pre-game state for any later game he didn't play in
    between (how matchups.py resolves relievers)."""
    if post_game:
        return ("ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
                f"ROWS BETWEEN {trailing_n - 1} PRECEDING AND CURRENT ROW",
                "at_bat_number DESC")
    return ("ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING",
            f"ROWS BETWEEN {trailing_n} PRECEDING AND 1 PRECEDING",
            "at_bat_number")


def _build_role_daily(
    con: duckdb.DuckDBPyConnection,
    player_col: str,
    n_alias: str,
    indicators: dict[str, str],
    role: str,
    trailing_n: int,
    post_game: bool = False,
) -> pd.DataFrame:
    """One row per (player_id, role, game_date, game_pk): cumulative counts
    for every indicator, in both the STD (season-to-date) and trailing-N
    windows -- pre-game by default, post-game if `post_game` (see
    `_window_frames`). Reads from the `pa` view registered by `_con_with_pa`.
    """
    ind_cols = ", ".join(f"{sql} AS {name}" for name, sql in indicators.items())
    std_frame, trail_frame, in_game_order = _window_frames(trailing_n, post_game)

    std_cols = ", ".join(
        _window_expr(name, std_frame, ", season") + f"AS std_{name}" for name in indicators
    )
    trail_cols = ", ".join(
        _window_expr(name, trail_frame) + f"AS trailing_{name}" for name in indicators
    )
    std_n = _window_expr("1", std_frame, ", season") + f"AS std_{n_alias}"
    trail_n = _window_expr("1", trail_frame) + f"AS trailing_{n_alias}"

    sql = f"""
    WITH facts AS (
        SELECT
            {player_col} AS player_id,
            game_pk, game_date, season, at_bat_number,
            1 AS n,
            {ind_cols}
        FROM pa
    ),
    windowed AS (
        SELECT *,
            {std_n}, {trail_n}, {std_cols}, {trail_cols},
            ROW_NUMBER() OVER (PARTITION BY player_id, game_pk ORDER BY {in_game_order}) AS _rn_in_game
        FROM facts
    )
    SELECT * EXCLUDE (_rn_in_game, at_bat_number, n, {", ".join(indicators)})
    FROM windowed
    WHERE _rn_in_game = 1
    """
    df = con.execute(sql).df()
    df.insert(1, "role", role)
    return df


def _build_hitter_daily(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return _build_role_daily(con, "batter_id", "pa", HITTER_INDICATOR_SQL, role="hitter",
                             trailing_n=config.TRAILING_PA_WINDOW)


def _build_pitcher_pa_daily(con: duckdb.DuckDBPyConnection, post_game: bool = False) -> pd.DataFrame:
    return _build_role_daily(con, "pitcher_id", "bf", PITCHER_INDICATOR_SQL, role="pitcher",
                             trailing_n=config.TRAILING_BF_WINDOW, post_game=post_game)


# ---------------------------------------------------------------------------
# Pitch-level aggregates (contact%/whiff%, pitch counts): raw statcast only
# ---------------------------------------------------------------------------

def _available_raw_pitch_paths() -> list[str]:
    """Every raw season on disk, not just the requested ones -- same reason
    as `_available_pa_paths`: trailing windows (contact%/whiff%) and the
    last-outing pitch count reach back across the season boundary. Reading
    only the requested seasons would silently count a prior season's PAs
    as zero swings."""
    raw_paths = sorted(str(p) for p in paths.RAW_STATCAST_DIR.glob("season=*/*.parquet"))
    if not raw_paths:
        raise FileNotFoundError("No raw statcast parquet found. Run `python -m basenet.data.ingest` first.")
    return raw_paths


def _pa_level_swing_whiff(con: duckdb.DuckDBPyConnection, raw_paths: list[str]) -> pd.DataFrame:
    """Swings/whiffs summed to one row per (game_pk, at_bat_number) -- i.e.
    per PA, matching the PA table's grain -- so they can feed the same
    cumulative-window machinery as the other pitcher indicators.
    """
    swing_set = ", ".join(f"'{d}'" for d in config.SWING_DESCRIPTIONS)
    whiff_set = ", ".join(f"'{d}'" for d in config.WHIFF_DESCRIPTIONS)
    sql = f"""
    SELECT
        game_pk, at_bat_number,
        SUM(CASE WHEN description IN ({swing_set}) THEN 1 ELSE 0 END) AS swings,
        SUM(CASE WHEN description IN ({whiff_set}) THEN 1 ELSE 0 END) AS whiffs
    FROM read_parquet({_pa_paths_sql_list(raw_paths)})
    GROUP BY game_pk, at_bat_number
    """
    return con.execute(sql).df()


def _pitch_counts_per_outing(con: duckdb.DuckDBPyConnection, raw_paths: list[str]) -> pd.DataFrame:
    """One row per (pitcher, game_pk): total pitches thrown that game, plus
    the calendar date, for the "last outing" / "trailing 3 day" features.
    """
    sql = f"""
    SELECT pitcher AS player_id, game_pk, game_date,
           COUNT(*) AS n_pitches
    FROM read_parquet({_pa_paths_sql_list(raw_paths)})
    GROUP BY pitcher, game_pk, game_date
    """
    return con.execute(sql).df()


def _pitch_recency_features(pitch_counts: pd.DataFrame) -> pd.DataFrame:
    """LAG-based pitch_count_last_outing + a small self-join for the trailing
    3-calendar-day pitch total, both computed strictly before each outing.
    """
    con = duckdb.connect()
    con.register("pc", pitch_counts)
    sql = """
    WITH ordered AS (
        SELECT *,
            LAG(n_pitches) OVER (PARTITION BY player_id ORDER BY game_date, game_pk) AS pitch_count_last_outing,
            LAG(game_date) OVER (PARTITION BY player_id ORDER BY game_date, game_pk) AS prev_game_date
        FROM pc
    ),
    trailing_pitches AS (
        SELECT a.player_id, a.game_pk,
               COALESCE(SUM(b.n_pitches), 0) AS pitch_count_trailing_3d
        FROM pc a
        LEFT JOIN pc b
          ON a.player_id = b.player_id
         AND b.game_date >= a.game_date - INTERVAL 3 DAY
         AND b.game_date < a.game_date
        GROUP BY a.player_id, a.game_pk
    )
    SELECT o.player_id, o.game_pk, o.game_date,
           o.pitch_count_last_outing,
           DATE_DIFF('day', o.prev_game_date, o.game_date) AS days_rest,
           t.pitch_count_trailing_3d
    FROM ordered o
    JOIN trailing_pitches t ON t.player_id = o.player_id AND t.game_pk = o.game_pk
    """
    return con.execute(sql).df()


# ---------------------------------------------------------------------------
# Leakage-safe running league means (day granularity, see module docstring)
# ---------------------------------------------------------------------------

def _daily_league_means(
    con: duckdb.DuckDBPyConnection, player_col: str, indicators: dict[str, str], role: str,
    n_alias: str,
) -> pd.DataFrame:
    """Same STD-style cumulative window as `_build_role_daily`, but pooled
    across every player of `role` (no PARTITION BY player_id) and collapsed
    to one row per calendar date -- the running, leakage-safe league-mean
    prior used to shrink individual players' rates (see module docstring).
    `n_alias` ("pa" or "bf") names the total-count column so it lines up
    with the same denominator name STAT_SPECS uses at the player level.
    """
    ind_cols = ", ".join(f"{sql} AS {name}" for name, sql in indicators.items())
    sum_cols = ", ".join(f"SUM({name}) AS {name}" for name in indicators)
    cum_cols = ", ".join(
        f"COALESCE(SUM({name}) OVER (PARTITION BY season ORDER BY game_date "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS lg_{name}"
        for name in indicators
    )
    sql = f"""
    WITH facts AS (
        SELECT game_date, season, 1 AS n, {ind_cols}
        FROM pa
    ),
    daily AS (
        SELECT game_date, season, SUM(n) AS n, {sum_cols}
        FROM facts
        GROUP BY game_date, season
    )
    SELECT game_date, season,
           COALESCE(SUM(n) OVER (PARTITION BY season ORDER BY game_date
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS lg_{n_alias},
           {cum_cols}
    FROM daily
    ORDER BY season, game_date
    """
    df = con.execute(sql).df()
    df["role"] = role
    return df


def _attach_league_rates(daily_league: pd.DataFrame, specs: dict[str, tuple[str, str]],
                          fallback: dict[str, float], n_alias: str) -> pd.DataFrame:
    n_col = f"lg_{n_alias}"
    out = daily_league[["season", "game_date", "role", n_col]].copy()
    for stat, (num, den) in specs.items():
        num_col, den_col = f"lg_{num}", f"lg_{den}"
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = daily_league[num_col] / daily_league[den_col].replace(0, np.nan)
        rate = rate.fillna(fallback[stat])
        rate = rate.where(daily_league[n_col] >= config.LEAGUE_MEAN_MIN_N, fallback[stat])
        out[f"league_{stat}"] = rate
    return out


# ---------------------------------------------------------------------------
# Shrinkage + assembly
# ---------------------------------------------------------------------------

def _apply_shrinkage(df: pd.DataFrame, specs: dict[str, tuple[str, str]],
                      shrink_k: dict[str, float], window: str) -> None:
    """Adds `{window}_{stat}` shrunk-rate columns in place, using the raw
    `{window}_{numerator}` / `{window}_{denominator}` counts already present
    plus a `league_{stat}` column (joined on beforehand).
    """
    for stat, (num, den) in specs.items():
        num_col, den_col = f"{window}_{num}", f"{window}_{den}"
        k = shrink_k[stat]
        n = df[den_col].fillna(0)
        numerator = df[num_col].fillna(0)
        league = df[f"league_{stat}"]
        df[f"{window}_{stat}"] = (numerator + k * league) / (n + k)


# ---------------------------------------------------------------------------
# Platoon splits + starter workload (pre-game rows only)
# ---------------------------------------------------------------------------

# Per-PA indicators the platoon windows need ("n" = one PA / batter faced),
# and each platoon stat's (numerator, denominator) in those terms.
_PLATOON_INDICATOR_SQL: dict[str, str] = {
    "n": "1", **{k: HITTER_INDICATOR_SQL[k] for k in ("k", "bb", "hr", "xwoba_num", "woba_den")},
}
_PLATOON_NUM_DEN: dict[str, tuple[str, str]] = {
    "k_pct": ("k", "n"), "bb_pct": ("bb", "n"), "hr_rate": ("hr", "n"), "hr_per_bf": ("hr", "n"),
    "xwoba": ("xwoba_num", "woba_den"),
}


def _platoon_counts(con: duckdb.DuckDBPyConnection, player_col: str, hand_col: str) -> pd.DataFrame:
    """Per (player_id, game_pk): cumulative indicator counts vs each opposing
    hand (`hand_col`: pitcher_throws for hitters, batter_stands for
    pitchers) over every PA strictly before this game -- the whole ingested
    window, not one season (splits need the sample). Read at the game's
    first PA, like the pre-game STD window."""
    cols = ", ".join(
        f"COALESCE(SUM(CASE WHEN opp_hand = '{hand}' THEN {sql} ELSE 0 END) OVER (PARTITION BY player_id "
        f"ORDER BY {ORDER_KEY} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS vs_{hand.lower()}_{name}"
        for name, sql in _PLATOON_INDICATOR_SQL.items() for hand in ("L", "R")
    )
    return con.execute(f"""
        WITH facts AS (SELECT {player_col} AS player_id, {hand_col} AS opp_hand, * FROM pa),
        windowed AS (
            SELECT player_id, game_pk, {cols},
                ROW_NUMBER() OVER (PARTITION BY player_id, game_pk ORDER BY at_bat_number) AS _rn
            FROM facts
        )
        SELECT * EXCLUDE (_rn) FROM windowed WHERE _rn = 1
    """).df()


def _attach_platoon(con: duckdb.DuckDBPyConnection, daily: pd.DataFrame, role: str) -> pd.DataFrame:
    """Adds plat_vs_{l,r}_{stat}: the rate vs that hand, shrunk toward the
    player's own overall rate over the same window with
    config.PLATOON_SHRINKAGE_K -- which is itself shrunk toward the running
    league rate (`league_{stat}`, already on `daily`) with the stat's usual
    k. A player with no PAs vs a hand gets his overall rate; one with no PAs
    at all gets the league rate."""
    if role == "hitter":
        player_col, hand_col, stats, shrink_k = "batter_id", "pitcher_throws", HITTER_PLATOON_STATS, config.HITTER_SHRINKAGE_K
    else:
        player_col, hand_col, stats, shrink_k = "pitcher_id", "batter_stands", PITCHER_PLATOON_STATS, config.PITCHER_SHRINKAGE_K
    counts = _platoon_counts(con, player_col, hand_col)
    daily = daily.merge(counts, on=["player_id", "game_pk"], how="left")
    kp = config.PLATOON_SHRINKAGE_K
    for stat in stats:
        num, den = _PLATOON_NUM_DEN[stat]
        l_num, r_num = daily[f"vs_l_{num}"].fillna(0), daily[f"vs_r_{num}"].fillna(0)
        l_den, r_den = daily[f"vs_l_{den}"].fillna(0), daily[f"vs_r_{den}"].fillna(0)
        k = shrink_k[stat]
        overall = (l_num + r_num + k * daily[f"league_{stat}"]) / (l_den + r_den + k)
        daily[f"plat_vs_l_{stat}"] = (l_num + kp * overall) / (l_den + kp)
        daily[f"plat_vs_r_{stat}"] = (r_num + kp * overall) / (r_den + kp)
    return daily.drop(columns=[c for c in counts.columns if c.startswith("vs_")])


def _attach_workload(con: duckdb.DuckDBPyConnection, daily: pd.DataFrame) -> pd.DataFrame:
    """Adds n_recent_starts / bf_per_start to pitcher rows: batters faced per
    start over the pitcher's last STARTER_BF_LOOKBACK_STARTS starts strictly
    before this game (any season), shrunk toward STARTER_BF_PRIOR with
    STARTER_BF_PRIOR_K starts. A start = the first pitcher his team used in a
    game (an opener counts, and his short outings pull the estimate down --
    which is the point)."""
    n = config.STARTER_BF_LOOKBACK_STARTS
    df = con.execute(f"""
        WITH app AS (
            SELECT pitcher_id AS player_id, game_pk, game_date, pitching_team, COUNT(*) AS bf
            FROM pa GROUP BY 1, 2, 3, 4
        ),
        first_pitcher AS (
            SELECT game_pk, pitching_team, arg_min(pitcher_id, at_bat_number) AS sp_id FROM pa GROUP BY 1, 2
        ),
        starts AS (
            SELECT a.player_id, a.game_pk, a.game_date, a.bf FROM app a
            JOIN first_pitcher f ON f.game_pk = a.game_pk AND f.pitching_team = a.pitching_team
                                AND f.sp_id = a.player_id
        ),
        prior AS (
            SELECT a.player_id, a.game_pk, s.bf,
                   ROW_NUMBER() OVER (PARTITION BY a.player_id, a.game_pk
                                      ORDER BY s.game_date DESC, s.game_pk DESC) AS rk
            FROM app a JOIN starts s
              ON s.player_id = a.player_id
             AND (s.game_date < a.game_date OR (s.game_date = a.game_date AND s.game_pk < a.game_pk))
        )
        SELECT player_id, game_pk, COUNT(*) AS n_recent_starts, SUM(bf) AS _bf_sum
        FROM prior WHERE rk <= {n} GROUP BY 1, 2
    """).df()
    daily = daily.merge(df, on=["player_id", "game_pk"], how="left")
    daily["n_recent_starts"] = daily["n_recent_starts"].fillna(0)
    k = config.STARTER_BF_PRIOR_K
    daily["bf_per_start"] = (daily["_bf_sum"].fillna(0) + k * config.STARTER_BF_PRIOR) / (daily["n_recent_starts"] + k)
    return daily.drop(columns="_bf_sum")


def _finalize_hitter_daily(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    daily = _build_hitter_daily(con)
    league = _daily_league_means(con, "batter_id", HITTER_INDICATOR_SQL, role="hitter", n_alias="pa")
    league_rates = _attach_league_rates(league, HITTER_STAT_SPECS, config.LEAGUE_MEAN_FALLBACK_HITTER, "pa")
    daily = daily.merge(league_rates.drop(columns=["role", "lg_pa"]), on=["season", "game_date"], how="left")

    _apply_shrinkage(daily, HITTER_STAT_SPECS, config.HITTER_SHRINKAGE_K, "std")
    _apply_shrinkage(daily, HITTER_STAT_SPECS, config.HITTER_SHRINKAGE_K, "trailing")
    daily = _attach_platoon(con, daily, "hitter")

    # No PAs anywhere in the ingested window -- a true debut (or first
    # appearance since the window starts). NOT std_pa == 0, which is every
    # player's first game of every season; that case is visible to the model
    # through std_pa itself.
    daily["is_cold_start"] = daily["trailing_pa"].fillna(0) == 0
    daily = daily.sort_values(["player_id", "game_date", "game_pk"])
    prev_date = daily.groupby("player_id")["game_date"].shift(1)
    daily["days_since_last_game"] = (daily["game_date"] - prev_date).dt.days
    daily["days_since_last_game"] = daily["days_since_last_game"].fillna(
        config.DEFAULT_DAYS_SINCE_LAST_GAME
    )

    keep = ["player_id", "role", "game_pk", "game_date", "season", "std_pa", "trailing_pa",
            "is_cold_start", "days_since_last_game"]
    keep += [f"std_{s}" for s in HITTER_STAT_SPECS] + [f"trailing_{s}" for s in HITTER_STAT_SPECS]
    keep += HITTER_PLATOON_COLUMNS
    return daily[keep]


def _finalize_pitcher_daily(con: duckdb.DuckDBPyConnection,
                            post_game: bool = False) -> pd.DataFrame:
    """Pre-game pitcher rows (default; PITCHER_FEATURE_COLUMNS) or, with
    `post_game`, each pitcher's state *after* each game (PITCHER_STATE_COLUMNS
    only -- fatigue features are relative to a target date, see
    PITCHER_FATIGUE_COLUMNS)."""
    daily = _build_pitcher_pa_daily(con, post_game=post_game)
    league = _daily_league_means(con, "pitcher_id", PITCHER_INDICATOR_SQL, role="pitcher", n_alias="bf")

    raw_paths = _available_raw_pitch_paths()
    swing_whiff = _pa_level_swing_whiff(con, raw_paths)
    con.register("swing_whiff", swing_whiff)
    con.execute("CREATE OR REPLACE VIEW pa_with_swings AS "
                "SELECT pa.*, sw.swings, sw.whiffs FROM pa "
                "LEFT JOIN swing_whiff sw USING (game_pk, at_bat_number)")
    std_frame, trail_frame, in_game_order = _window_frames(config.TRAILING_BF_WINDOW, post_game)
    # Same cumulative-window shape as `_build_role_daily`, but reading from
    # `pa_with_swings` (which the generic view-based helper can't target)
    # since swings/whiffs only exist joined onto the PA table, not in it.
    contact_daily = con.execute(f"""
        WITH facts AS (
            SELECT pitcher_id AS player_id, game_pk, game_date, season, at_bat_number,
                   COALESCE(swings, 0) AS swings, COALESCE(whiffs, 0) AS whiffs
            FROM pa_with_swings
        ),
        windowed AS (
            SELECT *,
                COALESCE(SUM(swings) OVER (PARTITION BY player_id, season ORDER BY {ORDER_KEY}
                    {std_frame}), 0) AS std_swings,
                COALESCE(SUM(whiffs) OVER (PARTITION BY player_id, season ORDER BY {ORDER_KEY}
                    {std_frame}), 0) AS std_whiffs,
                COALESCE(SUM(swings) OVER (PARTITION BY player_id ORDER BY {ORDER_KEY}
                    {trail_frame}), 0) AS trailing_swings,
                COALESCE(SUM(whiffs) OVER (PARTITION BY player_id ORDER BY {ORDER_KEY}
                    {trail_frame}), 0) AS trailing_whiffs,
                ROW_NUMBER() OVER (PARTITION BY player_id, game_pk ORDER BY {in_game_order}) AS _rn
            FROM facts
        )
        SELECT player_id, game_pk, std_swings, std_whiffs, trailing_swings, trailing_whiffs
        FROM windowed WHERE _rn = 1
    """).df()
    contact_daily["std_contact"] = contact_daily["std_swings"] - contact_daily["std_whiffs"]
    contact_daily["trailing_contact"] = contact_daily["trailing_swings"] - contact_daily["trailing_whiffs"]
    daily = daily.merge(contact_daily, on=["player_id", "game_pk"], how="left")

    # contact_pct/whiff_pct's numerator/denominator (swings/whiffs) live only
    # in `contact_daily`, not in the PA-outcome `league` table above -- rather
    # than a second full league-wide pass over raw pitches just for these
    # two, their league prior is the static fallback constant directly
    # (its influence is already small relative to the per-player k; see
    # config.LEAGUE_MEAN_FALLBACK_PITCHER's docstring on this being a
    # bootstrap-only anchor).
    pa_level_specs = {k: v for k, v in PITCHER_STAT_SPECS.items() if k not in ("contact_pct", "whiff_pct")}
    league_rates = _attach_league_rates(league, pa_level_specs, config.LEAGUE_MEAN_FALLBACK_PITCHER, "bf")
    for stat in ("contact_pct", "whiff_pct"):
        league_rates[f"league_{stat}"] = config.LEAGUE_MEAN_FALLBACK_PITCHER[stat]
    daily = daily.merge(league_rates.drop(columns=["role", "lg_bf"]), on=["season", "game_date"], how="left")

    _apply_shrinkage(daily, PITCHER_STAT_SPECS, config.PITCHER_SHRINKAGE_K, "std")
    _apply_shrinkage(daily, PITCHER_STAT_SPECS, config.PITCHER_SHRINKAGE_K, "trailing")

    daily["is_cold_start"] = daily["trailing_bf"].fillna(0) == 0  # see _finalize_hitter_daily

    if post_game:
        keep = ["player_id", "role", "game_pk", "game_date", "season"] + PITCHER_STATE_COLUMNS
        return daily[keep]

    pitch_counts = _pitch_counts_per_outing(con, raw_paths)
    recency = _pitch_recency_features(pitch_counts)
    daily = daily.merge(
        recency[["player_id", "game_pk", "pitch_count_last_outing", "days_rest", "pitch_count_trailing_3d"]],
        on=["player_id", "game_pk"], how="left",
    )
    daily["pitch_count_last_outing"] = daily["pitch_count_last_outing"].fillna(0)
    daily["pitch_count_trailing_3d"] = daily["pitch_count_trailing_3d"].fillna(0)
    daily["days_rest"] = daily["days_rest"].fillna(config.DEFAULT_DAYS_REST)
    daily = _attach_platoon(con, daily, "pitcher")
    daily = _attach_workload(con, daily)

    keep = ["player_id", "role", "game_pk", "game_date", "season", "std_bf", "trailing_bf",
            "is_cold_start", "days_rest", "pitch_count_last_outing", "pitch_count_trailing_3d"]
    keep += [f"std_{s}" for s in PITCHER_STAT_SPECS] + [f"trailing_{s}" for s in PITCHER_STAT_SPECS]
    keep += PITCHER_PLATOON_COLUMNS + PITCHER_WORKLOAD_COLUMNS
    return daily[keep]


def build_player_daily(seasons: list[int]) -> pd.DataFrame:
    """Builds player_daily using every PA file on disk (for correct trailing
    windows across season boundaries), returning only rows for `seasons`.
    """
    con = duckdb.connect()
    _con_with_pa(con)
    hitters = _finalize_hitter_daily(con)
    pitchers = _finalize_pitcher_daily(con)
    combined = pd.concat([hitters, pitchers], ignore_index=True)
    return combined[combined["season"].isin(seasons)].reset_index(drop=True)


def build_pitcher_post_game(seasons: list[int]) -> pd.DataFrame:
    """Each pitcher's state (PITCHER_STATE_COLUMNS) *after* each of his games,
    keyed (player_id, game_date, game_pk). For a target game a pitcher did
    NOT appear in (a bullpen candidate), his latest row strictly before that
    game is exactly his pre-game state -- no stale "as of before his last
    outing" lag. Same leakage contract: only PAs up to and including that
    earlier, already-completed game.
    """
    con = duckdb.connect()
    _con_with_pa(con)
    df = _finalize_pitcher_daily(con, post_game=True)
    return df[df["season"].isin(seasons)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True, choices=config.SEASONS)
    args = parser.parse_args(argv)

    with StageTimer(f"features seasons={args.seasons}").run() as t:
        pa_rows_in = sum(
            duckdb.connect().execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]
            for p in _available_pa_paths()
        )
        t.rows_in = pa_rows_in
        df = build_player_daily(args.seasons)
        post = build_pitcher_post_game(args.seasons)
        t.rows_out = len(df)

    for season in args.seasons:
        out_dir = ensure_dir(paths.INTERIM_PLAYER_DAILY_DIR)
        out_path = out_dir / f"player_daily_{season}.parquet"
        df[df["season"] == season].to_parquet(out_path, index=False)
        print(f"[features] wrote {out_path} ({(df['season'] == season).sum():,} rows)")

        out_dir = ensure_dir(paths.INTERIM_PITCHER_POST_GAME_DIR)
        out_path = out_dir / f"pitcher_post_game_{season}.parquet"
        post[post["season"] == season].to_parquet(out_path, index=False)
        print(f"[features] wrote {out_path} ({(post['season'] == season).sum():,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
