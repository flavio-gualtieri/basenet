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
from basenet.model_config import load_model_config


def test_basenet_forward_shapes_and_unknown_park(pipeline_env):
    split = load_split([2021], 2022, 2023, device="cpu")
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
