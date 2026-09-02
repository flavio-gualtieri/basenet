"""
input
    lineup: (B, 9, F_h)     B games, 9 lineup slots, F_h features each
                            (F_h includes expected-PA-per-slot as a plain
                            data-supplied feature -- see matchups.py; it is
                            measured from prior seasons' PA table, not a
                            model-owned constant)
    pa_weights: (B, 9)      only required when pooling="pa_weighted" -- the
                            same expected-PA quantity, passed as its own
                            tensor rather than read back out of `lineup`, so
                            the encoder stays decoupled from F_h's column
                            order. Caller extracts it from the data layer's
                            own per-row values (they vary by season/date),
                            not from a fixed table.

output
    (B, H)
"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from .base import Encoder


@dataclass(frozen=True)
class HitterEncoderConfig:
    in_dim: int                       # from the feature manifest, not yaml
    hidden_dim: int = 64
    num_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.1
    norm: Literal["group", "batch"] = "group"
    pooling: Literal["mean", "pa_weighted"] = "pa_weighted"

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric padding")
        # Receptive field after n layers of kernel k is 1 + n*(k-1).
        # Once it reaches 9, every slot already sees the whole lineup.
        rf = 1 + self.num_layers * (self.kernel_size - 1)
        if rf > 9 + (self.kernel_size - 1):
            raise ValueError(
                f"num_layers={self.num_layers} gives receptive field {rf} "
                f"over a 9-slot lineup; this is wasted depth"
            )


class CircularConvBlock(nn.Module):

    def __init__(self, dim: int, kernel_size: int, dropout: float, norm: str):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        )
        if norm == "group":
            self.norm = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim)
        else:
            self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 9)
        return x + self.drop(self.act(self.norm(self.conv(x))))


class HitterEncoder(Encoder):
    input_type = "hitter"

    def build_encoder(self) -> nn.Module:
        c = self.cfg
        self.proj = nn.Conv1d(c.in_dim, c.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            CircularConvBlock(c.hidden_dim, c.kernel_size, c.dropout, c.norm)
            for _ in range(c.num_layers)
        ])
        return nn.ModuleList([self.proj, *self.blocks])

    @property
    def output_dim(self) -> int:
        return self.cfg.hidden_dim

    def forward(self, lineup: torch.Tensor, pa_weights: torch.Tensor | None = None) -> torch.Tensor:
        """lineup: (B, 9, F_h) -> (B, hidden_dim).

        `pa_weights`: (B, 9), required iff pooling="pa_weighted" -- the
        data layer's own per-row expected-PA-per-slot values (see
        matchups.py / config.DEFAULT_PA_PER_SLOT's docstring for where
        these come from). Normalized internally; raw expected-PA counts
        are fine as-is, no need to pre-normalize before calling.
        """
        B, S, _ = lineup.shape
        if S != 9:
            raise ValueError(f"expected 9 lineup slots, got {S}")

        x = lineup.transpose(1, 2)          # (B, F, 9) channels-first
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)

        if self.cfg.pooling == "pa_weighted":
            if pa_weights is None:
                raise ValueError(
                    "pooling='pa_weighted' requires pa_weights (B, 9); "
                    "pass the data layer's per-row expected-PA-per-slot values"
                )
            w = pa_weights / pa_weights.sum(dim=1, keepdim=True)
            w = w.view(B, 1, 9)
            return (x * w).sum(dim=2)
        return x.mean(dim=2)
