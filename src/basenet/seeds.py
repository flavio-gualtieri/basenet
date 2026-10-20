"""Train BaseNet over several seeds and report mean +/- sd, so a gap between
BaseNet and the baselines can be read against seed-to-seed noise.

    python -m basenet.seeds
    python -m basenet.seeds --seeds 0 1 2 3 4 --out-dir runs/seeds

Each seed is a full `train.train()` run (its own early stopping, best-epoch
restore and single test evaluation) written to `<out-dir>/seed_<s>/`. The
aggregate goes to `<out-dir>/seeds_summary.json` and is printed as a markdown
table. The test set is still never used to choose anything: every seed's
model is picked on val, and the spread is only reported, not selected on.

The baselines (constant, GLM) don't depend on the seed -- they're fit by
deterministic LBFGS -- so their sd is ~0 and they're reported once.
`basenet_minus_glm` is the paired per-seed difference in test NLL (negative
= BaseNet better).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .model_config import DEFAULT_CONFIG_PATH
from .train import DEFAULT_OUT_DIR, train

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# (summary key, label) for every metric aggregated across seeds.
_METRICS = [
    ("test_loss", "BaseNet test NLL"), ("test_geo_p", "BaseNet test P(actual)"),
    ("test_mae", "BaseNet test MAE"), ("best_val_loss", "BaseNet val NLL"),
    ("glm_test_nll", "GLM test NLL"), ("glm_test_geo_p", "GLM test P(actual)"),
    ("glm_test_mae", "GLM test MAE"), ("baseline_test_nll", "Constant test NLL"),
    ("baseline_test_geo_p", "Constant test P(actual)"), ("baseline_test_mae", "Constant test MAE"),
    ("best_epoch", "best epoch"),
]


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def aggregate(summaries: list[dict]) -> dict:
    out = {key: _mean_sd([s[key] for s in summaries]) for key, _ in _METRICS}
    diffs = [s["test_loss"] - s["glm_test_nll"] for s in summaries]
    out["basenet_minus_glm_test_nll"] = {
        **_mean_sd(diffs), "per_seed": diffs,
        "seeds_basenet_better": sum(d < 0 for d in diffs),
    }
    return out


def to_markdown(agg: dict, n_seeds: int) -> str:
    lines = [f"| metric (n={n_seeds} seeds) | mean | sd |", "|---|---|---|"]
    for key, label in _METRICS:
        m, sd = agg[key]["mean"], agg[key]["sd"]
        lines.append(f"| {label} | {m:.4f} | {sd:.4f} |")
    d = agg["basenet_minus_glm_test_nll"]
    lines.append(f"| BaseNet - GLM test NLL | {d['mean']:+.4f} | {d['sd']:.4f} |")
    lines.append(f"\nBaseNet beat the GLM on test NLL in {d['seeds_basenet_better']}/{n_seeds} seeds.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--seasons-train", type=int, nargs="+", default=[2021, 2022, 2023])
    parser.add_argument("--season-val", type=int, default=2024)
    parser.add_argument("--season-test", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "seeds")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    summaries = []
    for seed in args.seeds:
        print(f"[seeds] === seed {seed} ===")
        summaries.append(train(
            args.seasons_train, args.season_val, args.season_test,
            args.epochs, args.batch_size, args.lr, seed, args.out_dir / f"seed_{seed}", args.device,
            weight_decay=args.weight_decay, patience=args.patience, config_path=args.config,
        ))

    agg = aggregate(summaries)
    result = {
        "seeds": args.seeds, "distribution": summaries[0]["distribution"],
        "feature_groups": summaries[0]["feature_groups"],
        "aggregate": agg,
        "geo_p_from_mean_nll": {k: math.exp(-agg[k]["mean"])
                                for k in ("test_loss", "glm_test_nll", "baseline_test_nll")},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "seeds_summary.json").write_text(json.dumps(result, indent=2))
    print(to_markdown(agg, len(args.seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
