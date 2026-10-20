"""Optional feature groups (datasets.FEATURE_GROUPS): quality of contact,
platoon splits, starter workload, schedule and weather context."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from basenet.data import config, datasets, features, matchups, paths, pa as pa_module
from basenet.data.features import (
    HITTER_FEATURE_COLUMNS, HITTER_PLATOON_COLUMNS, HITTER_PLATOON_STATS, PITCHER_FEATURE_COLUMNS,
    PITCHER_PLATOON_COLUMNS, PITCHER_PLATOON_STATS, PITCHER_WORKLOAD_COLUMNS,
)

ALL_GROUPS = tuple(datasets.FEATURE_GROUPS)


def _player_daily() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted(paths.INTERIM_PLAYER_DAILY_DIR.glob("*.parquet"))],
                     ignore_index=True)


# ---------------------------------------------------------------------------
# Leakage: a game's own outcomes never reach its pre-game features
# ---------------------------------------------------------------------------

def test_new_features_ignore_the_games_own_outcomes(pipeline_env):
    """Rewrite game 1003's contact quality (every batted ball 120 mph,
    xwOBA 2.0) and rebuild: every pre-game feature for game 1003 -- contact,
    platoon, workload, everything -- must be unchanged, while a later game
    (2000) must see the difference (so the mutation was real and visible)."""
    before = _player_daily()
    raw_path = paths.raw_statcast_season_dir(2021) / "statcast_2021.parquet"
    raw = pd.read_parquet(raw_path)
    in_play = (raw["game_pk"] == 1003) & raw["launch_speed"].notna()
    raw.loc[in_play, ["launch_speed", "estimated_woba_using_speedangle", "launch_speed_angle"]] = [120.0, 2.0, 6]
    raw.to_parquet(raw_path, index=False)
    pa_module.build_pa_table(2021).to_parquet(paths.interim_pa_season_dir(2021) / "pa_2021.parquet", index=False)
    after = features.build_player_daily([2021, 2022, 2023])

    key = ["player_id", "role", "game_pk"]
    cols = sorted(set(HITTER_FEATURE_COLUMNS + HITTER_PLATOON_COLUMNS + PITCHER_FEATURE_COLUMNS
                      + PITCHER_PLATOON_COLUMNS + PITCHER_WORKLOAD_COLUMNS))
    b = before.set_index(key).sort_index()
    a = after.set_index(key).sort_index()
    same_game = b.index.get_level_values("game_pk") == 1003
    pd.testing.assert_frame_equal(b.loc[same_game, cols], a.loc[same_game, cols], check_dtype=False)
    later = b.index.get_level_values("game_pk") == 2000
    hitters = b.index.get_level_values("role") == "hitter"
    assert not np.allclose(b.loc[later & hitters, "trailing_xwoba"], a.loc[later & hitters, "trailing_xwoba"])
    assert not np.allclose(b.loc[later & hitters, "trailing_hard_hit_pct"],
                           a.loc[later & hitters, "trailing_hard_hit_pct"])


# ---------------------------------------------------------------------------
# Quality of contact, platoon, workload
# ---------------------------------------------------------------------------

def test_debut_contact_and_platoon_fall_back_to_league(pipeline_env):
    """No history at all: every contact rate is the league prior, and both
    platoon splits equal the (league) overall rate."""
    d = _player_daily()
    debut = d[(d["role"] == "hitter") & (d["trailing_pa"] == 0)].iloc[0]
    assert debut["std_xwoba"] == pytest.approx(config.LEAGUE_MEAN_FALLBACK_HITTER["xwoba"])
    for s in HITTER_PLATOON_STATS:
        assert debut[f"plat_vs_l_{s}"] == pytest.approx(debut[f"plat_vs_r_{s}"])
    assert debut["plat_vs_l_k_pct"] == pytest.approx(debut["std_k_pct"])


def test_starter_workload_counts_only_earlier_starts(pipeline_env):
    """AAA's starter 901 starts every 2021 game, facing 9 batters each time.
    Going into his 4th start (game 1003) he has 3 earlier starts; reliever
    911 has none, so he sits at the prior."""
    d = _player_daily()
    sp = d[(d["role"] == "pitcher") & (d["player_id"] == 901) & (d["game_pk"] == 1003)].iloc[0]
    assert sp["n_recent_starts"] == 3
    k, prior = config.STARTER_BF_PRIOR_K, config.STARTER_BF_PRIOR
    assert sp["bf_per_start"] == pytest.approx((3 * 9 + k * prior) / (3 + k))
    rp = d[(d["role"] == "pitcher") & (d["player_id"] == 911) & (d["game_pk"] == 1003)].iloc[0]
    assert rp["n_recent_starts"] == 0
    assert rp["bf_per_start"] == pytest.approx(prior)


def test_matchup_platoon_picks_the_applicable_hand():
    """Standalone: the hitter's split follows the STARTER's hand; the
    starter's split follows each HITTER's side."""
    row = {"sp_throws": "L"}
    for s in PITCHER_PLATOON_STATS:
        row[f"sp_plat_vs_l_{s}"], row[f"sp_plat_vs_r_{s}"] = 0.11, 0.22
    for k in range(1, 10):
        row[f"slot{k}_stands"] = "R" if k % 2 else "L"
        for s in HITTER_PLATOON_STATS:
            row[f"slot{k}_plat_vs_l_{s}"], row[f"slot{k}_plat_vs_r_{s}"] = 0.33, 0.44
    out = matchups.resolve_matchup_platoon(pd.DataFrame([row])).iloc[0]
    assert not [c for c in out.index if "plat_vs_" in c]  # source columns dropped
    assert out["slot1_plat_xwoba"] == 0.33       # vs the LEFT-handed starter
    assert out["slot1_sp_plat_xwoba"] == 0.22    # slot 1 bats right
    assert out["slot2_sp_plat_xwoba"] == 0.11    # slot 2 bats left


# ---------------------------------------------------------------------------
# Schedule + weather context
# ---------------------------------------------------------------------------

def test_team_schedule_features():
    """Standalone: rest days (doubleheader = 0, opener = cap), travel, and
    day-after-night only when the previous game was the night before."""
    games = [(1, dt.date(2099, 4, 1), 10, "night"), (2, dt.date(2099, 4, 2), 10, "day"),
             (3, dt.date(2099, 4, 2), 10, "night"), (4, dt.date(2099, 4, 5), 20, "day")]
    pa_df = pd.DataFrame([{"game_pk": g, "game_date": pd.Timestamp(d), "batting_team": "T", "venue_id": v}
                          for g, d, v, _ in games])
    day_night = pd.Series({g: dn for g, _, _, dn in games})
    out = matchups._team_schedule_features(pa_df, day_night).set_index("game_pk")
    assert out["days_since_last_game"].tolist() == [config.TEAM_REST_DAYS_CAP, 1, 0, 3]
    assert out["traveled"].tolist() == [0, 0, 0, 1]
    assert out["day_after_night"].tolist() == [0, 1, 0, 0]


def test_parse_weather():
    sched = pd.DataFrame({
        "game_pk": [1, 2, 3, 4, 5],
        "weather_condition": ["Clear", "Cloudy", "Roof Closed", "Sunny", "Rain"],
        "weather_temp": [85.0, 55.0, 0.0, np.nan, 61.0],
        "weather_wind": ["12 mph, Out To CF", "7 mph, In From LF", "0 mph, None", "9 mph, L To R", "Calm"],
    })
    w = matchups.parse_weather(sched).set_index("game_pk")
    assert w["wind_out_mph"].tolist() == [12, 0, 0, 0, 0]
    assert w["wind_in_mph"].tolist() == [0, 7, 0, 0, 0]
    assert w["is_indoor"].tolist() == [0, 0, 1, 0, 0]
    assert w["is_precip"].tolist() == [0, 0, 0, 0, 1]
    # the indoor 0F placeholder and the missing temp both become the default
    assert w["temp_f"].tolist() == [85, 55, config.DEFAULT_TEMP_F, config.DEFAULT_TEMP_F, 61]


def test_schedule_context_reaches_matchups(pipeline_env):
    m = datasets.load_matchups([2021, 2022])
    nightcap = m[(m["game_pk"] == 1001)].iloc[0]       # DH game 2, a day game
    assert nightcap["is_day_game"] == 1
    assert nightcap["bat_days_since_last_game"] == 0
    assert nightcap["bat_day_after_night"] == 0        # same date, not "the night before"
    roof = m[m["game_pk"] == 2001].iloc[0]             # see conftest.ROOF_CLOSED_GAME_PK
    assert (roof["is_indoor"], roof["temp_f"], roof["wind_out_mph"]) == (1, config.DEFAULT_TEMP_F, 0)
    aaa_home = m[(m["park_id"] == "AAA") & (m["game_pk"] != 2001)]
    assert (aaa_home["wind_out_mph"] == 10).all() and (aaa_home["temp_f"] == 80).all()


# ---------------------------------------------------------------------------
# Tensors: the switch
# ---------------------------------------------------------------------------

def test_groups_change_widths_and_tensors_match(pipeline_env):
    df = datasets.load_matchups([2021, 2022, 2023])
    base = datasets.feature_dims(())
    full = datasets.feature_dims(ALL_GROUPS)
    assert all(f > b for f, b in zip(full[:1] + full[2:], base[:1] + base[2:]))  # lineup, context grow
    assert full[1] == base[1] + len(features.PITCHER_CONTACT_COLUMNS)
    t = datasets.build_tensors(df, groups=ALL_GROUPS)
    assert t["lineup"].shape[2] == full[0]
    assert t["pitchers"].shape[2] == full[1]
    assert t["context"].shape[1] == full[2]
    for key in ("lineup", "pitchers", "context"):
        assert not np.isnan(t[key]).any(), key


def test_workload_sets_starter_and_bullpen_innings(pipeline_env):
    df = datasets.load_matchups([2021, 2022, 2023])
    w = datasets.build_tensors(df, groups=("workload",))["pitcher_weights"]
    expected_sp = np.clip(df["sp_bf_per_start"].to_numpy() / config.BATTERS_PER_INNING,
                          1.0, config.TEAM_INNINGS_PER_GAME - 1.0)
    np.testing.assert_allclose(w[:, 0], expected_sp)
    has_bullpen = df["bullpen1_mask"].to_numpy(dtype=bool)
    np.testing.assert_allclose(w[has_bullpen].sum(axis=1), config.TEAM_INNINGS_PER_GAME)
    # without the group: the old flat starter weight
    assert (datasets.build_tensors(df)["pitcher_weights"][:, 0] == config.SP_EXPECTED_IP).all()


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError, match="unknown feature group"):
        datasets.feature_dims(("contact", "vibes"))
