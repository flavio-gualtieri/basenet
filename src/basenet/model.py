"""BaseNet: wires HitterEncoder + PitcherEncoder + ScoreHead into one model.

    from basenet.model_config import load_model_config
    from basenet.model import BaseNet

    cfg = load_model_config(n_parks=len(split.scaler.park_vocab))  # reads configs.yaml, injects data-derived dims
    net = BaseNet(cfg).to(device)
    log_lambda = net(batch)            # batch: loader.SplitTensors
    loss = net.loss(batch)
    runs = net.predict_runs(batch)

BaseNet takes its shape entirely from `cfg: ModelConfig` and imports nothing
from `basenet.data` or `basenet.model_config` itself -- every number it
needs (feature widths, context width) already arrived pre-resolved inside
`cfg`. The one thing it computes itself is the head's `in_dim`, which
can't be a yaml value: it depends on `hidden_dim`/`keep_sp_separate`
choices baked into the two encoders, so it's only knowable once they exist.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder.hitter import HitterEncoder
from .encoder.pitcher import PitcherEncoder
from .head.head import ScoreHead
from .loader import SplitTensors
from .model_config import ModelConfig


class BaseNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.hitter_encoder = HitterEncoder(cfg.hitter)
        self.pitcher_encoder = PitcherEncoder(cfg.pitcher)
        # +1 row for UNKNOWN_PARK (index 0): padding_idx keeps it a fixed
        # zero vector, so a park unseen in training contributes nothing
        # instead of an untrained random vector.
        self.park_embedding = nn.Embedding(cfg.n_parks + 1, cfg.head.park_embedding_dim, padding_idx=0)
        head_in_dim = (
            self.hitter_encoder.output_dim + self.pitcher_encoder.output_dim + cfg.context_dim
            + cfg.head.park_embedding_dim
        )
        self.head = ScoreHead(cfg.head, in_dim=head_in_dim)

    def forward(self, batch: SplitTensors) -> torch.Tensor:
        """Returns log-lambda, shape (B,). Not runs -- see ScoreHead."""
        hitter_vec = self.hitter_encoder(batch.lineup, batch.pa_weights)
        pitcher_vec = self.pitcher_encoder(batch.pitchers, batch.pitcher_weights, batch.pitcher_mask)
        context = torch.cat([batch.context, self.park_embedding(batch.park)], dim=-1)
        return self.head(hitter_vec, pitcher_vec, context)

    def loss(self, batch: SplitTensors) -> torch.Tensor:
        """Poisson NLL against `batch.y`. Mirrors ScoreHead.loss() one level up."""
        return self.head.loss(self(batch), batch.y)

    @torch.no_grad()
    def predict_runs(self, batch: SplitTensors) -> torch.Tensor:
        """Expected runs, shape (B,). Inference only -- never feed to loss()."""
        return torch.exp(self(batch))
