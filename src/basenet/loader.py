"""Loads a chronological train/val/test split as PyTorch tensors, fully in
memory, scaled and device-placed -- ready to slice and feed to the model
directly.

Deliberately skips `torch.utils.data.Dataset`/`DataLoader`: that machinery
exists to stream data that doesn't fit in memory. The whole dataset here is
~24k rows across 5 seasons; loading it onto the device once and indexing it
(`split.train.lineup[idx]`, a boolean mask, `torch.randperm` for shuffling,
...) is simpler and has fewer moving parts while the model side is still
being shaken out.

    from basenet.loader import load_split
    split = load_split(seasons_train=[2021, 2022, 2023], season_val=2024, season_test=2025)
    split.train.lineup            # (N_train, 9, F_h), scaled, on `device`
    split.train.pa_weights        # (N_train, 9), RAW (never scaled)
    split.train.pitchers          # (N_train, K, F_p), scaled
    split.train.pitcher_mask      # (N_train, K), bool
    split.train.pitcher_weights   # (N_train, K), RAW expected innings
    split.train.context           # (N_train, F_c), scaled
    split.train.park              # (N_train,), int64 park index (0 = unseen in train)
    split.train.y                 # (N_train,)
    split.train.game_pk           # (N_train,) -- kept on CPU, bookkeeping only
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import datasets

_FIELD_NAMES = (
    "lineup", "pa_weights", "pitchers", "pitcher_mask", "pitcher_weights",
    "context", "park", "y", "game_pk",
)


def _pick_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class SplitTensors:
    lineup: torch.Tensor
    pa_weights: torch.Tensor
    pitchers: torch.Tensor
    pitcher_mask: torch.Tensor
    pitcher_weights: torch.Tensor
    context: torch.Tensor
    park: torch.Tensor     # int64 park index, never scaled
    y: torch.Tensor
    game_pk: torch.Tensor  # int64, kept on CPU -- bookkeeping, never fed to the model

    def __len__(self) -> int:
        return self.lineup.shape[0]

    def __getitem__(self, idx: torch.Tensor) -> "SplitTensors":
        """Slices every field by the same index -- what makes this usable
        as a minibatch container. `game_pk` lives on CPU while every other
        field lives on the model's device, so the index has to be moved to
        match each field's own device rather than assumed to work for both.
        """
        device = self.lineup.device
        idx_dev = idx.to(device)
        idx_cpu = idx.to("cpu")
        return SplitTensors(
            lineup=self.lineup[idx_dev],
            pa_weights=self.pa_weights[idx_dev],
            pitchers=self.pitchers[idx_dev],
            pitcher_mask=self.pitcher_mask[idx_dev],
            pitcher_weights=self.pitcher_weights[idx_dev],
            context=self.context[idx_dev],
            park=self.park[idx_dev],
            y=self.y[idx_dev],
            game_pk=self.game_pk[idx_cpu],
        )


@dataclass
class Split:
    train: SplitTensors
    val: SplitTensors
    test: SplitTensors
    scaler: datasets.Scaler


def _to_split_tensors(arrs: dict[str, np.ndarray], device: torch.device) -> SplitTensors:
    kwargs = {}
    for name in _FIELD_NAMES:
        arr = arrs[name]
        if name == "game_pk":
            kwargs[name] = torch.as_tensor(arr)  # CPU, native dtype
        elif name == "park":
            kwargs[name] = torch.as_tensor(arr, dtype=torch.long, device=device)
        elif name == "pitcher_mask":
            kwargs[name] = torch.as_tensor(arr, dtype=torch.bool, device=device)
        else:
            kwargs[name] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return SplitTensors(**kwargs)


def load_split(
    seasons_train: list[int], season_val: int, season_test: int,
    device: str | torch.device | None = None, feature_groups=(),
) -> Split:
    """Fits the chronological split + scaler on `seasons_train` (persisting
    both, same as calling `datasets.fit_and_persist_split` directly), then
    builds, scales, and device-places tensors for all three splits.
    """
    dev = _pick_device(device)
    scaler = datasets.fit_and_persist_split(seasons_train, season_val, season_test,
                                            feature_groups=feature_groups)
    train_pks, val_pks, test_pks = datasets.chronological_split(seasons_train, season_val, season_test)

    all_df = datasets.load_matchups(seasons_train + [season_val, season_test])

    parts: dict[str, SplitTensors] = {}
    for name, pks in (("train", train_pks), ("val", val_pks), ("test", test_pks)):
        df = all_df[all_df["game_pk"].isin(pks)]
        tensors = datasets.build_tensors(df, scaler.park_vocab, feature_groups)
        tensors = scaler.transform(tensors)
        parts[name] = _to_split_tensors(tensors, dev)

    return Split(train=parts["train"], val=parts["val"], test=parts["test"], scaler=scaler)
