from __future__ import annotations

import numpy as np

from basenet.data import config, datasets


def test_tensor_shapes(pipeline_env):
    df = datasets.load_matchups([2021, 2022, 2023])
    tensors = datasets.build_tensors(df)
    n = len(df)
    assert tensors["lineup"].shape == (n, config.N_LINEUP_SLOTS, datasets.F_HITTER)
    assert tensors["pa_weights"].shape == (n, config.N_LINEUP_SLOTS)
    assert tensors["pitchers"].shape == (n, datasets.K_PITCHERS, datasets.F_PITCHER)
    assert tensors["pitcher_mask"].shape == (n, datasets.K_PITCHERS)
    assert tensors["pitcher_weights"].shape == (n, datasets.K_PITCHERS)
    assert tensors["context"].shape == (n, datasets.F_CONTEXT)
    assert tensors["y"].shape == (n,)
    assert tensors["game_pk"].shape == (n,)
    assert not np.isnan(tensors["lineup"]).any()
    assert not np.isnan(tensors["pitchers"]).any()
    # index 0 of the pitcher set is always the starter
    assert tensors["pitcher_mask"][:, 0].all()
    assert (tensors["pitcher_weights"][:, 0] == config.SP_EXPECTED_IP).all()


def test_expected_pa_channel_breaks_slot_symmetry(pipeline_env):
    df = datasets.load_matchups([2021])
    tensors = datasets.build_tensors(df)
    expected_pa_idx = len(datasets.HITTER_FEATURE_COLUMNS)  # channel order: features..., expected_pa, hand
    slot1_expected_pa = tensors["lineup"][:, 0, expected_pa_idx]
    slot9_expected_pa = tensors["lineup"][:, 8, expected_pa_idx]
    assert np.allclose(slot1_expected_pa, config.DEFAULT_PA_PER_SLOT[1])
    assert np.allclose(slot9_expected_pa, config.DEFAULT_PA_PER_SLOT[9])
    assert not np.allclose(slot1_expected_pa, slot9_expected_pa)


def test_scaler_zero_means_after_transform_on_train(pipeline_env):
    train_pks, val_pks, test_pks = datasets.chronological_split([2021], 2022, 2023)
    train_df = datasets.load_matchups([2021])
    park_vocab = datasets.build_park_vocab(datasets.load_matchups([2021, 2022, 2023]))
    train_tensors = datasets.build_tensors(train_df, park_vocab)
    scaler = datasets.fit_scaler(train_tensors, park_vocab)
    scaled = scaler.transform(train_tensors)
    assert abs(scaled["context"].mean()) < 1e-6
    assert abs(scaled["lineup"].mean()) < 1e-6


def test_scaler_save_load_roundtrip(pipeline_env, tmp_path):
    train_df = datasets.load_matchups([2021])
    park_vocab = datasets.build_park_vocab(train_df)
    tensors = datasets.build_tensors(train_df, park_vocab)
    scaler = datasets.fit_scaler(tensors, park_vocab)
    scaler.save(tmp_path / "scaler")
    loaded = datasets.Scaler.load(tmp_path / "scaler")
    for key in datasets.SCALED_KEYS:
        np.testing.assert_array_equal(scaler.mean[key], loaded.mean[key])
        np.testing.assert_array_equal(scaler.std[key], loaded.std[key])
    assert loaded.park_vocab == park_vocab


def test_pitcher_scaler_ignores_padding_rows(pipeline_env):
    """The scaler's 'pitchers' stats must be computed only from real
    (mask=True) slots -- padding rows are a fixed 0.0 placeholder, and
    averaging them in would pull every stat toward 0 regardless of the
    real pitchers' actual values.
    """
    train_df = datasets.load_matchups([2021])
    tensors = datasets.build_tensors(train_df)
    scaler = datasets.fit_scaler(tensors, datasets.build_park_vocab(train_df))

    mask = tensors["pitcher_mask"]
    flat = tensors["pitchers"].reshape(-1, tensors["pitchers"].shape[-1])
    real_only_mean = flat[mask.reshape(-1)].mean(axis=0)
    np.testing.assert_allclose(scaler.mean["pitchers"], real_only_mean)


def test_park_index_is_train_fit_with_unknown_bucket(pipeline_env):
    """Park is an int64 index (for an embedding), not a scaled float in
    `context`; the vocab comes from train only, and a park train never saw
    maps to UNKNOWN_PARK (0) rather than getting an untrained index."""
    train_df = datasets.load_matchups([2021])
    vocab = datasets.build_park_vocab(train_df, min_rows=1)  # synthetic venues have only a few games
    assert datasets.UNKNOWN_PARK not in vocab.values()
    assert sorted(vocab.values()) == list(range(1, len(vocab) + 1))

    tensors = datasets.build_tensors(train_df, vocab)
    assert tensors["park"].dtype == np.int64
    assert (tensors["park"] >= 1).all()
    assert "park" not in datasets.SCALED_KEYS

    unseen = train_df.head(4).assign(**{datasets.PARK_KEY: "NEW"})
    assert (datasets.build_tensors(unseen, vocab)["park"] == datasets.UNKNOWN_PARK).all()


def test_rarely_seen_park_shares_unknown_index(pipeline_env):
    """2021: AAA's park hosts 3 games (6 rows), BBB's hosts 1 (2 rows). With
    min_rows=3 only AAA's venue gets its own index; BBB's maps to unknown."""
    train_df = datasets.load_matchups([2021])
    vocab = datasets.build_park_vocab(train_df, min_rows=3)
    assert list(vocab) == ["101"]  # synthetic venue_id = home team id (AAA=1) + 100
    park = datasets.build_tensors(train_df, vocab)["park"]
    is_bbb_home = (train_df["park_id"] == "BBB").to_numpy()
    assert (park[is_bbb_home] == datasets.UNKNOWN_PARK).all()
    assert (park[~is_bbb_home] == 1).all()
