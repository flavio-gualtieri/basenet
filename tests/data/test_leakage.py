"""The six non-negotiable leakage tests: the invariants that make the
matchups table safe to train on. validate.py re-runs the real-data analogues
(runs reconcile, two rows per game) against the actual data/ tree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from basenet.data import config, datasets, pa as pa_module


# ---------------------------------------------------------------------------
# 1. test_no_future_features
# ---------------------------------------------------------------------------

def test_no_future_features(pipeline_env):
    """Recompute std_pa/trailing_pa (hitters) and std_bf/trailing_bf
    (pitchers) from scratch, using only PAs strictly before each
    player_daily row's own (game_date, game_pk), and assert equality with
    the stored value -- for every row, not just a sample (the synthetic
    fixture is small enough to check exhaustively).
    """
    from basenet.data import paths
    pa_all = pd.concat(
        [pd.read_parquet(p) for p in sorted(paths.INTERIM_PA_DIR.glob("season=*/*.parquet"))],
        ignore_index=True,
    )
    player_daily = pd.concat(
        [pd.read_parquet(p) for p in sorted(paths.INTERIM_PLAYER_DAILY_DIR.glob("*.parquet"))],
        ignore_index=True,
    )

    def strictly_before(df, game_date, game_pk):
        return df[(df["game_date"] < game_date) | ((df["game_date"] == game_date) & (df["game_pk"] < game_pk))]

    checked = 0
    for row in player_daily.itertuples(index=False):
        if row.role == "hitter":
            prior = strictly_before(pa_all, row.game_date, row.game_pk)
            prior = prior[prior["batter_id"] == row.player_id]
            std_prior = prior[prior["season"] == row.season]
            assert len(std_prior) == row.std_pa, (
                f"std_pa leak/miscount for batter {row.player_id} at "
                f"({row.game_date}, {row.game_pk}): recomputed {len(std_prior)} != stored {row.std_pa}"
            )
            trailing_prior = prior.sort_values(["game_date", "game_pk", "at_bat_number"]).tail(
                config.TRAILING_PA_WINDOW
            )
            assert len(trailing_prior) == row.trailing_pa
            checked += 1
        else:
            prior = strictly_before(pa_all, row.game_date, row.game_pk)
            prior = prior[prior["pitcher_id"] == row.player_id]
            std_prior = prior[prior["season"] == row.season]
            assert len(std_prior) == row.std_bf, (
                f"std_bf leak/miscount for pitcher {row.player_id} at "
                f"({row.game_date}, {row.game_pk}): recomputed {len(std_prior)} != stored {row.std_bf}"
            )
            trailing_prior = prior.sort_values(["game_date", "game_pk", "at_bat_number"]).tail(
                config.TRAILING_BF_WINDOW
            )
            assert len(trailing_prior) == row.trailing_bf
            checked += 1
    assert checked == len(player_daily)

    # Complementary check: a player's very first-ever appearance (debut,
    # zero prior PAs/BF in the whole window) must be flagged is_cold_start --
    # not something that could only be known by looking at *this* game's
    # own outcome -- and nothing else is.
    hitter_debuts = player_daily[(player_daily["role"] == "hitter") & (player_daily["trailing_pa"] == 0)]
    assert len(hitter_debuts) > 0  # the fixture must actually exercise this case
    assert (hitter_debuts["is_cold_start"] == True).all()  # noqa: E712
    assert player_daily["is_cold_start"].sum() == (
        (player_daily["role"] == "hitter") & (player_daily["trailing_pa"] == 0)
        | (player_daily["role"] == "pitcher") & (player_daily["trailing_bf"] == 0)
    ).sum()


# ---------------------------------------------------------------------------
# 2. test_game_grouping
# ---------------------------------------------------------------------------

def test_game_grouping(pipeline_env):
    """No game_pk appears in more than one split."""
    train_pks, val_pks, test_pks = datasets.chronological_split([2021], 2022, 2023)
    assert train_pks & val_pks == set()
    assert train_pks & test_pks == set()
    assert val_pks & test_pks == set()
    # and both rows of each game_pk really do land together (same game_pk
    # can't be half in train and half in val since matchups is loaded by
    # whole season, but assert the row-level property explicitly too)
    all_matchups = datasets.load_matchups([2021, 2022, 2023])
    for split_pks in (train_pks, val_pks, test_pks):
        subset = all_matchups[all_matchups["game_pk"].isin(split_pks)]
        assert (subset.groupby("game_pk").size() == 2).all()


# ---------------------------------------------------------------------------
# 3. test_chronological
# ---------------------------------------------------------------------------

def test_chronological(pipeline_env):
    """max(train date) < min(val date) < min(test date)."""
    train_df = datasets.load_matchups([2021])
    val_df = datasets.load_matchups([2022])
    test_df = datasets.load_matchups([2023])

    assert train_df["game_date"].max() < val_df["game_date"].min()
    assert val_df["game_date"].max() < test_df["game_date"].min()

    # chronological_split itself must not raise for a valid ordering...
    datasets.chronological_split([2021], 2022, 2023)
    # ...and must refuse an invalid (overlapping/out-of-order) one.
    with pytest.raises(ValueError):
        datasets.chronological_split([2022], 2021, 2023)


# ---------------------------------------------------------------------------
# 4. test_scaler_fit_scope
# ---------------------------------------------------------------------------

def test_scaler_fit_scope(pipeline_env, tmp_path):
    """Scaler statistics are unchanged when val/test rows are removed from
    (or altered in) the input -- i.e. the scaler only ever sees train.
    """
    from basenet.data import paths
    scaler_1 = datasets.fit_and_persist_split([2021], 2022, 2023, out_dir=tmp_path / "split1")

    # Mutate the val season's matchups on disk (as if val data were entirely
    # different) and refit: the persisted scaler must be byte-for-byte the
    # same, since it must never have looked at val/test in the first place.
    val_path = paths.PROCESSED_MATCHUPS_DIR / "matchups_2022.parquet"
    val_df = pd.read_parquet(val_path)
    mutated = val_df.copy()
    numeric_cols = [c for c in mutated.columns if c.startswith("slot1_std_")]
    mutated[numeric_cols] = mutated[numeric_cols] * 1000.0 + 999.0
    mutated.to_parquet(val_path, index=False)

    scaler_2 = datasets.fit_and_persist_split([2021], 2022, 2023, out_dir=tmp_path / "split2")

    for key in datasets.SCALED_KEYS:
        np.testing.assert_array_equal(scaler_1.mean[key], scaler_2.mean[key])
        np.testing.assert_array_equal(scaler_1.std[key], scaler_2.std[key])


# ---------------------------------------------------------------------------
# 5. test_runs_reconcile
# ---------------------------------------------------------------------------

def test_runs_reconcile(pipeline_env):
    """Per-game summed runs_on_play equals the final box score for every
    game in every season.
    """
    from basenet.data import paths
    for p in sorted(paths.INTERIM_PA_DIR.glob("season=*/*.parquet")):
        df = pd.read_parquet(p)
        bad = pa_module.reconcile_runs(df)
        assert bad.empty, f"runs reconciliation failed in {p}:\n{bad}"


# ---------------------------------------------------------------------------
# 6. test_two_rows_per_game
# ---------------------------------------------------------------------------

def test_two_rows_per_game(pipeline_env):
    """Every game_pk in matchups has exactly two rows, with distinct
    batting_team, and their labels sum to the game's total runs.
    """
    from basenet.data import paths
    for season in (2021, 2022, 2023):
        df = pd.read_parquet(paths.PROCESSED_MATCHUPS_DIR / f"matchups_{season}.parquet")
        counts = df.groupby("game_pk").size()
        assert (counts == 2).all(), counts[counts != 2]

        for game_pk, g in df.groupby("game_pk"):
            assert g["batting_team"].nunique() == 2
            total_runs = g["runs_scored"].sum()
            # cross-check against the box score directly from the PA table
            pa_df = pd.read_parquet(paths.interim_pa_season_dir(season) / f"pa_{season}.parquet")
            game_pa = pa_df[pa_df["game_pk"] == game_pk].iloc[0]
            expected_total = game_pa["final_home_score"] + game_pa["final_away_score"]
            assert total_runs == expected_total
