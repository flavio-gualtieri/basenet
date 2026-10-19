"""Seasons, windows, shrinkage constants, and feature lists for the data pipeline.

Nothing in this file touches the network or the filesystem — it is pure
constants so every stage (and every test) imports the same numbers.
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

# 2020 is excluded per spec: 60-game season, expanded playoffs, universal DH
# introduced mid-stream, no-fans park effects — not representative.
SEASONS: Final[list[int]] = [2021, 2022, 2023, 2024, 2025]

# Wide, deliberately generous request windows for `pybaseball.statcast`.
# These are NOT the authoritative regular-season boundary — `pa.py` filters
# on `game_type == 'R'` for that, which is exact regardless of the calendar
# quirks of any given year (lockout-delayed start in 2022, Seoul/Tokyo
# season-opening series in 2024/2025, tie-breaker games, hurricane makeups,
# etc.). These bounds only exist to keep the raw pull from wastefully
# fetching months of off-season emptiness.
SEASON_DATE_RANGES: Final[dict[int, tuple[str, str]]] = {
    2021: ("2021-03-15", "2021-10-10"),
    2022: ("2022-03-15", "2022-10-10"),
    2023: ("2023-03-15", "2023-10-10"),
    2024: ("2024-03-10", "2024-10-10"),
    2025: ("2025-03-10", "2025-10-10"),
}

REGULAR_SEASON_GAME_TYPE: Final[str] = "R"

# StatsAPI schedule `codedGameState` values that mean "this game was played
# and has an official final score" ('F' covers both "Final" and "Completed
# Early"). Postponed ('D') and cancelled ('C') entries are excluded.
PLAYED_GAME_STATES: Final[set[str]] = {"F"}

# ---------------------------------------------------------------------------
# Rolling feature windows (features.py)
# ---------------------------------------------------------------------------

TRAILING_PA_WINDOW: Final[int] = 300  # hitters: trailing N plate appearances
TRAILING_BF_WINDOW: Final[int] = 300  # pitchers: trailing N batters faced

# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage: shrunk = (n*observed + k*league_mean) / (n + k)
#
# These `k` values set how many PA/BF of league-average data a player's rate
# is "worth" before we start believing their observed rate over the league
# mean. They are approximate starting points informed by general sabermetric
# stabilization-point research (e.g. Russell Carleton's work for Baseball
# Prospectus) — NOT refit against this project's data. Treat them as
# hyperparameters to tune once real backtests exist, not as fixed truth.
# ---------------------------------------------------------------------------

HITTER_SHRINKAGE_K: Final[dict[str, float]] = {
    "k_pct": 60.0,      # strikeout rate stabilizes fast
    "bb_pct": 120.0,
    "hbp_pct": 240.0,   # rare event, noisy
    "x1b_rate": 290.0,
    "x2b_rate": 500.0,
    "x3b_rate": 1000.0,  # very rare
    "hr_rate": 150.0,
    "babip": 800.0,      # notoriously slow to stabilize (spec calls this out explicitly)
    "gb_pct": 80.0,
    "fb_pct": 80.0,
}

PITCHER_SHRINKAGE_K: Final[dict[str, float]] = {
    "k_pct": 70.0,
    "bb_pct": 170.0,
    "hr_per_bf": 400.0,  # HR rate allowed is one of the noisiest pitcher stats
    "gb_pct": 70.0,
    "contact_pct": 100.0,
    "whiff_pct": 100.0,
}

# Bootstrap-only league-average priors, rough modern-era (2021-2025) anchors.
# The real shrinkage target is the running, leakage-safe season-to-date
# league mean computed in features.py (through the end of the PRIOR calendar
# date only); these constants are the fallback used exclusively when that
# running sample is empty (n_league < LEAGUE_MEAN_MIN_N), i.e. only for
# games on the very first date(s) of the very first ingested season, when
# there is no in-pipeline league history yet to compute a real one from.
# Their influence vanishes within the season's first few days as real
# league volume accumulates -- do not read them as calibrated targets.
LEAGUE_MEAN_MIN_N: Final[int] = 100
LEAGUE_MEAN_FALLBACK_HITTER: Final[dict[str, float]] = {
    "k_pct": 0.23, "bb_pct": 0.085, "hbp_pct": 0.01,
    "x1b_rate": 0.145, "x2b_rate": 0.045, "x3b_rate": 0.004, "hr_rate": 0.032,
    "babip": 0.29, "gb_pct": 0.44, "fb_pct": 0.35,
}
LEAGUE_MEAN_FALLBACK_PITCHER: Final[dict[str, float]] = {
    "k_pct": 0.23, "bb_pct": 0.085, "hr_per_bf": 0.032,
    "gb_pct": 0.44, "contact_pct": 0.77, "whiff_pct": 0.24,
}

# ---------------------------------------------------------------------------
# Plate-appearance outcome vocabulary + statcast `events` -> outcome mapping.
# ---------------------------------------------------------------------------

OUTCOME_LABELS: Final[list[str]] = [
    "K", "BB", "HBP", "OUT_IP", "1B", "2B", "3B", "HR", "ROE", "OTHER",
]

# statcast's `events` column, populated only on the last pitch of a PA.
# Rows with events in {"truncated_pa", "game_advisory"} are dropped from the
# PA table entirely (no real outcome was recorded) -- see pa.py.
EVENTS_TO_OUTCOME: Final[dict[str, str]] = {
    "strikeout": "K",
    "strikeout_double_play": "K",
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
    "field_error": "ROE",
    "field_out": "OUT_IP",
    "force_out": "OUT_IP",
    "double_play": "OUT_IP",
    "grounded_into_double_play": "OUT_IP",
    "sac_fly": "OUT_IP",
    "sac_bunt": "OUT_IP",
    "sac_fly_double_play": "OUT_IP",
    "sac_bunt_double_play": "OUT_IP",
    "triple_play": "OUT_IP",
    # A fielder's choice means the batter reached base but is not the out;
    # it doesn't cleanly fit OUT_IP (batter safe) or any hit code. It's a
    # small fraction of PAs (~1%); we bucket it (and catcher's interference)
    # as OTHER rather than inventing a category the spec's enum doesn't have.
    "fielders_choice": "OTHER",
    "fielders_choice_out": "OTHER",
    "catcher_interf": "OTHER",
}

# events values with no real PA outcome -- dropped, not mapped to OTHER.
DROPPED_EVENTS: Final[set[str]] = {"truncated_pa", "game_advisory"}

# Pitch-level `description` taxonomy used for pitcher contact%/whiff%
# (features.py). foul_tip counts as contact (bat touched the ball), not a
# whiff, matching Baseball Savant's own convention.
SWING_DESCRIPTIONS: Final[set[str]] = {
    "foul", "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul_tip", "foul_bunt", "missed_bunt",
}
WHIFF_DESCRIPTIONS: Final[set[str]] = {
    "swinging_strike", "swinging_strike_blocked", "missed_bunt",
}

# ---------------------------------------------------------------------------
# Lineup slot structure (datasets.py)
# ---------------------------------------------------------------------------

N_LINEUP_SLOTS: Final[int] = 9

# Bootstrap-only fallback for expected-PA-per-slot: approximate long-run MLB
# averages, used ONLY when the leakage-safe empirical estimate (computed
# from our own PA table in matchups.py -- see compute_slot_pa_table) has too
# small a sample to trust yet, i.e. a season's first date(s), same pattern
# as LEAGUE_MEAN_FALLBACK_HITTER/PITCHER. This table itself is NOT what gets
# fed to the model: expected PA per slot varies by season/run-environment
# and is a measured quantity, not a hyperparameter -- see matchups.py.
DEFAULT_PA_PER_SLOT: Final[dict[int, float]] = {
    1: 4.62, 2: 4.47, 3: 4.37, 4: 4.28, 5: 4.18,
    6: 4.08, 7: 3.98, 8: 3.90, 9: 3.85,
}
# Minimum leakage-safe sample (team-games, summed across all 9 slots' prior
# occurrences through the end of the prior calendar date) before trusting
# the empirical per-slot PA estimate over the static bootstrap above.
SLOT_PA_MIN_TEAM_GAMES: Final[int] = 20

# ---------------------------------------------------------------------------
# Bullpen availability heuristic (matchups.py)
# ---------------------------------------------------------------------------

# A reliever is considered UNavailable for game date D if he threw more than
# this many pitches within the trailing 1 (resp. 2) calendar day(s) before D.
BULLPEN_MAX_PITCHES_1D: Final[int] = 30
BULLPEN_MAX_PITCHES_2D: Final[int] = 50

# A pitcher counts as "on this team's bullpen staff" for a given game if he
# has pitched (in relief) for that team at least once in the trailing window
# below. This is a proxy for roster membership -- statcast carries no
# transactions/40-man feed -- and will miss e.g. a reliever activated from
# the IL with no recent appearances, or double-count someone traded within
# the window who pitched for both clubs. Documented, not solved, in v1.
BULLPEN_ROSTER_LOOKBACK_DAYS: Final[int] = 30

# Rotation starters also "pitched for this team recently". A candidate is
# excluded from the bullpen if at least this share of his appearances in
# the lookback window were starts -- removes the rotation, keeps openers
# and swingmen (mostly relief, the occasional start).
BULLPEN_STARTER_SHARE_EXCLUDE: Final[float] = 0.5

# v1 expected-innings heuristic: split this many total innings uniformly
# across available relievers. Left as a heuristic per spec; the interface
# (`matchups.estimate_bullpen_ip_shares`) is written so a learned policy can
# be swapped in without changing callers.
BULLPEN_EXPECTED_TOTAL_IP: Final[float] = 3.5

# Complementary v1 heuristic: the starter's own expected innings, used only
# as his "weight" in the pitcher-set tensor (see below) -- a 9-inning game
# split roughly SP + bullpen. Same caveat as BULLPEN_EXPECTED_TOTAL_IP: a
# placeholder for a learned policy, not a calibrated target.
SP_EXPECTED_IP: Final[float] = 5.5

# ---------------------------------------------------------------------------
# Pitcher-SET tensor shape (datasets.py): index 0 is always the starter,
# indices 1..MAX_BULLPEN_SIZE are available relievers ordered most- to
# least-recently-used, padded/masked when fewer are available and
# TRUNCATED (dropping the least-recently-used) when more are. Sized by
# recall against who actually pitched in relief (2024): the median pool is
# ~11 (the 30-day lookback also catches optioned/injured arms), and 8 slots
# kept 85.8% of actual relievers vs 92.4% at 12 and a 92.8% ceiling with no
# truncation. matchups.py logs the truncation rate.
# ---------------------------------------------------------------------------
MAX_BULLPEN_SIZE: Final[int] = 12

# Fallback rest-day value when a player has no earlier appearance in the
# data at all (their debut in the window we've ingested) -- paired with
# is_cold_start=True so the model can learn to discount it rather than
# reading it as a real, extremely-well-rested pitcher.
DEFAULT_DAYS_SINCE_LAST_GAME: Final[int] = 4
DEFAULT_DAYS_REST: Final[int] = 5
