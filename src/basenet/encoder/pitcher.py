"""
input
    x: (B, K, F_p)     B games, K pitchers (index 0 = starter, 1..K-1 =
                        available bullpen, padded), F_p features each
    weights: (B, K)    expected innings for that slot -- SP and bullpen
                        pitchers share the same scale (see
                        config.SP_EXPECTED_IP / BULLPEN_EXPECTED_TOTAL_IP),
                        so the pooled average is a true innings-weighted mix
    mask: (B, K)       True where a real pitcher, False where padding

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
class PitcherEncoderConfig:
    in_dim: int
    num_layers: int = 2
    hidden_dim: int = 48
    dropout: float = 0.0
    keep_sp_separate: bool = True
    pooling: Literal["ip_weighted", "mean"] = "ip_weighted"


class PitcherEncoder(Encoder):
    input_type = "pitcher"

    def build_encoder(self):
        c = self.cfg
        layers: list[nn.Module] = []
        d_in = c.in_dim

        for _ in range(c.num_layers):
            layers += [
                nn.Linear(d_in, c.hidden_dim),
                nn.LayerNorm(c.hidden_dim),
                nn.GELU(),
                nn.Dropout(c.dropout),
            ]
            d_in = c.hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.net(x)

        w = mask.to(h.dtype)
        if self.cfg.pooling == "ip_weighted":
            w = w * weights
        w = w.unsqueeze(-1)                              # (B, K, 1)

        pooled = (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)

        if self.cfg.keep_sp_separate:
            pooled = torch.cat([h[:, 0], pooled], dim=-1)
        return pooled

    @property
    def output_dim(self) -> int:
        return self.cfg.hidden_dim * (2 if self.cfg.keep_sp_separate else 1)
