from __future__ import annotations

import pandas as pd
import pytest

from basenet.data import paths, pa as pa_module


def _pa_2021(pipeline_env) -> pd.DataFrame:
    return pd.read_parquet(paths.interim_pa_season_dir(2021) / "pa_2021.parquet")


def test_outcome_mapping(pipeline_env):
    df = _pa_2021(pipeline_env)
    assert set(df["outcome"].unique()) <= set(
        ["K", "BB", "HBP", "OUT_IP", "1B", "2B", "3B", "HR", "ROE", "OTHER"]
    )
    # our synthetic generator never emits a fielder's-choice/catcher's-interference
    # PA, but every outcome it *does* emit should be present at least once,
    # confirming the events->outcome CASE mapping actually fired for each.
    assert {"K", "BB", "HBP", "OUT_IP", "1B", "2B", "3B", "HR", "ROE"} <= set(df["outcome"].unique())


def test_order_slot_rotates_1_through_9(pipeline_env):
    df = _pa_2021(pipeline_env)
    g = df[(df["game_pk"] == 1000) & (df["batting_team"] == "AAA")].sort_values("at_bat_number")
    assert list(g["order_slot"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9] * 2  # 2 trips through the order


def test_doubleheader_game_2_flag(pipeline_env):
    df = _pa_2021(pipeline_env)
    game1 = df[df["game_pk"] == 1000]["is_doubleheader_game_2"]
    game2 = df[df["game_pk"] == 1001]["is_doubleheader_game_2"]
    assert (game1 == False).all()  # noqa: E712
    assert (game2 == True).all()  # noqa: E712
    # a non-doubleheader game is never flagged
    game3 = df[df["game_pk"] == 1002]["is_doubleheader_game_2"]
    assert (game3 == False).all()  # noqa: E712


def test_times_through_order_resets_per_pitcher(pipeline_env):
    df = _pa_2021(pipeline_env)
    g = df[(df["game_pk"] == 1000) & (df["batting_team"] == "BBB")].sort_values("at_bat_number")
    # inning 1 faced by one pitcher (n_thru=1 throughout, since the synthetic
    # game never gives one pitcher more than a single 9-batter trip), inning
    # 2 faced by the reliever, also his first time through.
    assert (g["times_through_order"] == 1).all()


def test_unmapped_event_fails_loudly(pipeline_env, monkeypatch):
    """A statcast `events` value outside EVENTS_TO_OUTCOME/DROPPED_EVENTS
    must raise, not silently fall into OTHER -- see pa.py's
    _assert_events_covered.
    """
    raw_path = paths.raw_statcast_season_dir(2021) / "statcast_2021.parquet"
    df = pd.read_parquet(raw_path)
    df.loc[df.index[0], "events"] = "some_future_statcast_event_type"
    df.to_parquet(raw_path, index=False)

    with pytest.raises(ValueError, match="Unmapped statcast"):
        pa_module.build_pa_table(2021)


def test_non_pa_walkoff_scores_the_winning_run(pipeline_env):
    """Game 1003 ends on a walk-off wild pitch (no `events` on that pitch).
    The final score must include it (3-2), not stop at the last terminal
    pitch (which would record a 2-2 tie), and reconciliation must accept it.
    """
    from conftest import NON_PA_WALKOFF_GAME_PK
    df = _pa_2021(pipeline_env)
    game = df[df["game_pk"] == NON_PA_WALKOFF_GAME_PK].iloc[0]
    assert (game["final_home_score"], game["final_away_score"]) == (3, 2)
    assert pa_module.reconcile_runs(df).empty
    assert pa_module.count_non_pa_walkoffs(df) == 1


def test_reconcile_still_flags_unexplained_run_gaps(pipeline_env):
    """The walk-off exemption is narrow: a gap for the away team, or a home
    gap of more than the single winning run, is still a failure."""
    df = _pa_2021(pipeline_env)
    away_gap = df.copy()
    away_gap.loc[away_gap["game_pk"] == 1000, "final_away_score"] += 1
    assert set(pa_module.reconcile_runs(away_gap)["game_pk"]) == {1000}

    home_gap = df.copy()
    home_gap.loc[home_gap["game_pk"] == 1000, "final_home_score"] += 2
    assert set(pa_module.reconcile_runs(home_gap)["game_pk"]) == {1000}


def test_label_disagreeing_with_schedule_fails_loudly(pipeline_env):
    sched_path = paths.raw_schedule_season_dir(2021) / "schedule_2021.parquet"
    sched = pd.read_parquet(sched_path)
    sched.loc[sched["game_pk"] == 1002, "away_score"] += 1
    sched.to_parquet(sched_path, index=False)
    with pytest.raises(RuntimeError, match="StatsAPI cross-check"):
        pa_module.build_pa_table(2021)


def test_pitchless_walkoff_takes_official_score(pipeline_env):
    """A walk-off balk scores the winning run with no pitch thrown, so
    statcast ends tied; the official schedule's +1 home run is the label."""
    sched_path = paths.raw_schedule_season_dir(2021) / "schedule_2021.parquet"
    sched = pd.read_parquet(sched_path)
    sched.loc[sched["game_pk"] == 1002, "home_score"] += 1  # 1002 ends 2-2 in statcast
    sched.to_parquet(sched_path, index=False)
    df = pa_module.build_pa_table(2021)
    game = df[df["game_pk"] == 1002].iloc[0]
    assert (game["final_home_score"], game["final_away_score"]) == (3, 2)
    assert pa_module.reconcile_runs(df).empty


def test_game_missing_from_statcast_fails_loudly(pipeline_env):
    sched_path = paths.raw_schedule_season_dir(2021) / "schedule_2021.parquet"
    sched = pd.read_parquet(sched_path)
    extra = sched.iloc[[0]].assign(game_pk=9999)
    pd.concat([sched, extra]).to_parquet(sched_path, index=False)
    with pytest.raises(RuntimeError, match="missing from statcast"):
        pa_module.build_pa_table(2021)
