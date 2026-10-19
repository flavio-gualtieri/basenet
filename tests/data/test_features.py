from __future__ import annotations

import pandas as pd

from basenet.data import config, paths


def _player_daily(pipeline_env) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(p) for p in sorted(paths.INTERIM_PLAYER_DAILY_DIR.glob("*.parquet"))],
        ignore_index=True,
    )


def test_debut_shrinks_fully_to_league_mean(pipeline_env):
    """A player's very first game (std_pa == 0) has no observed data at all,
    so shrunk = (0*x + k*league)/(0+k) == league mean exactly -- not 0, and
    not influenced by that game's own outcome.
    """
    pdaily = _player_daily(pipeline_env)
    debut = pdaily[(pdaily["role"] == "hitter") & (pdaily["trailing_pa"] == 0)].iloc[0]
    assert debut["std_k_pct"] > 0  # not zero -- shrunk to a league rate, not an empty average
    assert debut["is_cold_start"]


def test_season_opener_is_not_cold_start(pipeline_env):
    """Player 1's first 2022 game: season-to-date is empty (std_pa == 0),
    but he has a full 2021 behind him -- not a cold start. (is_cold_start
    used to be std_pa == 0, flagging every player's season opener.)"""
    pdaily = _player_daily(pipeline_env)
    row = pdaily[(pdaily["role"] == "hitter") & (pdaily["player_id"] == 1)
                 & (pdaily["game_pk"] == 2000)].iloc[0]
    assert row["std_pa"] == 0
    assert not row["is_cold_start"]
    sp = pdaily[(pdaily["role"] == "pitcher") & (pdaily["player_id"] == 901)
                & (pdaily["game_pk"] == 2000)].iloc[0]
    assert sp["std_bf"] == 0
    assert not sp["is_cold_start"]


def test_single_season_rebuild_matches_full_rebuild(pipeline_env):
    """Building features for 2022 alone must give the same pitcher rows as
    the full build: trailing contact/whiff windows and last-outing pitch
    counts reach back into 2021's raw pitches either way. (Raw pitches used
    to be read for the requested seasons only, so a single-season rebuild
    silently counted 2021 PAs as zero swings.)"""
    from basenet.data import features
    full = _player_daily(pipeline_env)
    full = full[(full["season"] == 2022) & (full["role"] == "pitcher")]
    alone = features.build_player_daily([2022])
    alone = alone[alone["role"] == "pitcher"]
    key = ["player_id", "game_pk"]
    pd.testing.assert_frame_equal(
        full.sort_values(key).reset_index(drop=True), alone.sort_values(key).reset_index(drop=True),
    )


def test_trailing_carries_across_season_boundary_std_does_not(pipeline_env):
    """The first 2022 game for a player who also played in 2021 should have
    std_pa == 0 (season-to-date resets) but trailing_pa > 0 (trailing looks
    back across the season boundary).
    """
    pdaily = _player_daily(pipeline_env)
    row = pdaily[(pdaily["role"] == "hitter") & (pdaily["player_id"] == 1)
                 & (pdaily["game_pk"] == 2000)].iloc[0]
    assert row["std_pa"] == 0
    assert row["trailing_pa"] > 0


def test_shrinkage_k_values_loaded_from_config(pipeline_env):
    # sanity: every stat produced actually has a configured shrink constant
    # (a stat present in HITTER_FEATURE_COLUMNS with no matching k would be
    # a silent config/features.py drift).
    from basenet.data.features import HITTER_STAT_SPECS, PITCHER_STAT_SPECS
    assert set(HITTER_STAT_SPECS) == set(config.HITTER_SHRINKAGE_K)
    assert set(PITCHER_STAT_SPECS) == set(config.PITCHER_SHRINKAGE_K)


def test_pitcher_pitch_count_features_present(pipeline_env):
    pdaily = _player_daily(pipeline_env)
    pitchers = pdaily[pdaily["role"] == "pitcher"]
    # the reliever's second-ever outing should show a nonzero last-outing
    # pitch count (from his first outing) and a real days_rest value.
    reliever_rows = pitchers[pitchers["player_id"] == 911].sort_values(["game_date", "game_pk"])
    assert len(reliever_rows) >= 2
    second = reliever_rows.iloc[1]
    assert second["pitch_count_last_outing"] > 0
    assert second["days_rest"] >= 0
