"""End-to-end smoke test of training on the synthetic pipeline output: the
model side (loader -> BaseNet -> GLM baseline -> early stopping -> checkpoint)
wires together and produces finite metrics. Lives under tests/data/ only to
reuse its `pipeline_env` fixture."""
from __future__ import annotations

import json
import math

import torch

from basenet import train as train_module
from basenet.loader import load_split
from basenet.model import BaseNet
from basenet.model_config import load_feature_groups, load_model_config


def test_basenet_forward_shapes_and_unknown_park(pipeline_env):
    # same config for the tensors and the model's widths, as train.py does
    split = load_split([2021], 2022, 2023, device="cpu", feature_groups=load_feature_groups())
    cfg = load_model_config(n_parks=len(split.scaler.park_vocab))
    net = BaseNet(cfg)
    log_lambda = net(split.val)
    assert log_lambda.shape == (len(split.val),)
    assert torch.isfinite(log_lambda).all()
    # the unknown-park row is a fixed zero vector, not a trainable one
    assert torch.all(net.park_embedding.weight[0] == 0)


def test_train_runs_and_restores_best_epoch(pipeline_env, tmp_path):
    summary = train_module.train(
        [2021], 2022, 2023, epochs=6, batch_size=4, lr=1e-3, seed=0, out_dir=tmp_path,
        device="cpu", patience=2,
    )
    assert 1 <= summary["best_epoch"] <= summary["epochs_run"] <= 6
    for key in ("best_val_loss", "test_loss", "glm_val_nll", "glm_test_nll", "baseline_val_nll"):
        assert math.isfinite(summary[key]), key
    ckpt = torch.load(tmp_path / "basenet.pt", weights_only=False)
    assert ckpt["epoch"] == summary["best_epoch"]
    assert ckpt["val_loss"] == summary["best_val_loss"]
    assert json.loads((tmp_path / "summary.json").read_text())["best_epoch"] == summary["best_epoch"]


def test_train_negbin_reports_dispersion_for_model_and_baselines(pipeline_env, tmp_path):
    import yaml
    from basenet.model_config import DEFAULT_CONFIG_PATH
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    cfg["model"]["head"]["distribution"] = "negbin"
    cfg_path = tmp_path / "negbin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    summary = train_module.train(
        [2021], 2022, 2023, epochs=4, batch_size=4, lr=1e-3, seed=0, out_dir=tmp_path / "run",
        device="cpu", patience=2, config_path=cfg_path,
    )
    assert summary["distribution"] == "negbin"
    for key in ("dispersion_r", "baseline_dispersion_r", "glm_dispersion_r"):
        assert summary[key] is not None and summary[key] > 0, key
    for key in ("best_val_loss", "test_loss", "glm_val_nll", "baseline_val_nll"):
        assert math.isfinite(summary[key]) and summary[key] > 0, key  # full NLL is positive
    assert 0 < summary["best_val_geo_p"] < 1


def test_train_with_every_feature_group(pipeline_env, tmp_path):
    """configs.yaml `features.groups` drives both the tensors and the
    model's input widths -- they must agree for every group at once."""
    import yaml
    from basenet.data import datasets
    from basenet.model_config import DEFAULT_CONFIG_PATH
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    cfg["features"]["groups"] = list(datasets.FEATURE_GROUPS)
    cfg_path = tmp_path / "all_groups.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    summary = train_module.train(
        [2021], 2022, 2023, epochs=3, batch_size=4, lr=1e-3, seed=0, out_dir=tmp_path / "run",
        device="cpu", patience=2, config_path=cfg_path,
    )
    assert summary["feature_groups"] == sorted(datasets.FEATURE_GROUPS)
    assert math.isfinite(summary["best_val_loss"]) and math.isfinite(summary["glm_val_nll"])


def test_config_without_feature_groups_is_rejected(tmp_path):
    import pytest
    import yaml
    from basenet.model_config import DEFAULT_CONFIG_PATH
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    del cfg["features"]
    cfg_path = tmp_path / "no_features.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="features.groups"):
        load_model_config(n_parks=3, path=cfg_path)
