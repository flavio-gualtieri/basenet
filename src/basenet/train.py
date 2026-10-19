"""Train BaseNet on a chronological train/val/test split.

    python -m basenet.train
    python -m basenet.train --epochs 100 --batch-size 256 --lr 3e-4
    python -m basenet.train --weight-decay 0.1 --patience 15 --config my_configs.yaml

Loads `configs.yaml` via `model_config.load_model_config()`, builds
`BaseNet`, trains with full-in-memory minibatches (see `loader.py` -- no
`Dataset`/`DataLoader`), evaluates on val every epoch, reports final
train/val/test metrics against two baselines -- a constant prediction (the
training mean `runs_scored`; the floor) and a linear Poisson GLM on the same
inputs (see baseline.py; the bar BaseNet actually has to clear) -- and saves
a checkpoint + a training-curve figure.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: this writes a file, never opens a window
import matplotlib.pyplot as plt
import torch

from .baseline import fit_glm, PoissonGLM
from .loader import load_split, SplitTensors
from .model import BaseNet
from .model_config import DEFAULT_CONFIG_PATH, load_model_config

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "runs"


def _mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def _baseline_metrics(train_y: torch.Tensor, target: SplitTensors) -> dict[str, float]:
    """Constant-prediction baseline: always predict the training set's mean
    runs_scored, regardless of matchup. Doesn't touch the model at all --
    exists purely so the trained model's own numbers have something to beat.
    """
    mean_runs = train_y.mean().clamp_min(1e-6)
    log_lambda = torch.log(mean_runs).expand_as(target.y)
    criterion = torch.nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
    nll = criterion(log_lambda, target.y).item()
    mae = _mae(mean_runs.expand_as(target.y), target.y)
    return {"nll": nll, "mae": mae}


@torch.no_grad()
def _glm_metrics(glm: PoissonGLM, target: SplitTensors) -> dict[str, float]:
    log_lambda = glm(target)
    y = target.y.cpu()
    criterion = torch.nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
    return {"nll": criterion(log_lambda, y).item(), "mae": _mae(torch.exp(log_lambda), y)}


def train(
    seasons_train: list[int], season_val: int, season_test: int,
    epochs: int, batch_size: int, lr: float, seed: int, out_dir: Path,
    device: str | None = None, weight_decay: float = 0.01, patience: int = 10,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict:
    """Trains for up to `epochs`, stopping early once val loss hasn't improved
    for `patience` epochs, and restores the best-val-loss weights before the
    test evaluation and checkpoint (val picks the epoch; test is only ever
    scored once, on that model)."""
    torch.manual_seed(seed)

    split = load_split(seasons_train, season_val, season_test, device=device)
    cfg = load_model_config(n_parks=len(split.scaler.park_vocab), path=config_path)
    net = BaseNet(cfg).to(split.train.lineup.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}

    n_train = len(split.train)
    history: dict[str, list[float]] = {"epoch": [], "train_loss": [], "val_loss": [], "val_mae": []}

    print(f"[train] device={split.train.lineup.device} "
          f"params={sum(p.numel() for p in net.parameters()):,} "
          f"train={n_train:,} val={len(split.val):,} test={len(split.test):,}")

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch = split.train[idx]
            optimizer.zero_grad()
            loss = net.loss(batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / n_train

        net.eval()
        with torch.no_grad():
            val_loss = net.loss(split.val).item()
            val_mae = _mae(net.predict_runs(split.val), split.val.y)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        improved = val_loss < best["val_loss"]
        if improved:
            best = {"val_loss": val_loss, "epoch": epoch,
                    "state": copy.deepcopy(net.state_dict())}
        print(f"[train] epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_mae={val_mae:.4f}{'  *' if improved else ''}")
        if epoch - best["epoch"] >= patience:
            print(f"[train] early stop: no val improvement for {patience} epochs "
                  f"(best epoch {best['epoch']})")
            break

    wall = time.perf_counter() - t0

    net.load_state_dict(best["state"])
    net.eval()
    with torch.no_grad():
        best_val_mae = _mae(net.predict_runs(split.val), split.val.y)
    with torch.no_grad():
        test_loss = net.loss(split.test).item()
        test_mae = _mae(net.predict_runs(split.test), split.test.y)

    baseline_val = _baseline_metrics(split.train.y, split.val)
    baseline_test = _baseline_metrics(split.train.y, split.test)
    glm = fit_glm(split.train, n_parks=len(split.scaler.park_vocab))
    glm_val = _glm_metrics(glm, split.val)
    glm_test = _glm_metrics(glm, split.test)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "basenet.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "cfg": dataclasses.asdict(cfg),
        "seasons_train": seasons_train, "season_val": season_val, "season_test": season_test,
        "epoch": best["epoch"], "val_loss": best["val_loss"],
    }, ckpt_path)

    fig_path = out_dir / "training_curve.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train")
    ax1.plot(history["epoch"], history["val_loss"], label="val")
    ax1.axhline(baseline_val["nll"], color="gray", linestyle="--", label="val baseline (constant)")
    ax1.axhline(glm_val["nll"], color="black", linestyle=":", label="val baseline (GLM)")
    ax1.axvline(best["epoch"], color="tab:green", alpha=0.4, label=f"best epoch ({best['epoch']})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Poisson NLL")
    ax1.set_title("loss")
    ax1.legend()
    ax2.plot(history["epoch"], history["val_mae"], color="tab:orange", label="val MAE")
    ax2.axhline(baseline_val["mae"], color="gray", linestyle="--", label="val baseline (constant)")
    ax2.axhline(glm_val["mae"], color="black", linestyle=":", label="val baseline (GLM)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("|pred - actual| runs")
    ax2.set_title("val MAE (runs)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    summary = {
        "wall_time_s": round(wall, 1),
        "epochs": epochs, "epochs_run": history["epoch"][-1], "best_epoch": best["epoch"],
        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay, "patience": patience,
        "seed": seed, "config": str(config_path),
        # val/test metrics below are all for the restored best-epoch model
        "best_val_loss": best["val_loss"], "best_val_mae": best_val_mae,
        "test_loss": test_loss, "test_mae": test_mae,
        "baseline_val_nll": baseline_val["nll"], "baseline_val_mae": baseline_val["mae"],
        "baseline_test_nll": baseline_test["nll"], "baseline_test_mae": baseline_test["mae"],
        "glm_val_nll": glm_val["nll"], "glm_val_mae": glm_val["mae"],
        "glm_test_nll": glm_test["nll"], "glm_test_mae": glm_test["mae"],
        "checkpoint": str(ckpt_path), "figure": str(fig_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons-train", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--season-val", type=int, default=2024)
    parser.add_argument("--season-test", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=150, help="max epochs (early stopping usually ends sooner)")
    parser.add_argument("--patience", type=int, default=15,
                        help="stop after this many epochs without a val-loss improvement")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    summary = train(
        args.seasons_train, args.season_val, args.season_test,
        args.epochs, args.batch_size, args.lr, args.seed, args.out_dir, args.device,
        weight_decay=args.weight_decay, patience=args.patience, config_path=args.config,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
