from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..distributions import DISTRIBUTIONS, Distribution, nll


@dataclass(frozen=True)
class HeadConfig:
    """Configuration for the run-prediction head.

    hidden_dims determines both the width and the depth of the MLP; there is
    no separate num_layers knob.
    """

    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    log_lambda_min: float = -5.0
    log_lambda_max: float = 4.0
    park_embedding_dim: int = 4  # learned park vector concatenated onto the head input
    # Run-count distribution around the predicted mean: "poisson" (variance =
    # mean) or "negbin" (one learned, global dispersion r: variance = mean +
    # mean^2 / r). See basenet.distributions.
    distribution: Distribution = "poisson"

    def __post_init__(self) -> None:
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {DISTRIBUTIONS}, got {self.distribution!r}")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if self.park_embedding_dim < 0:
            raise ValueError(f"park_embedding_dim must be >= 0, got {self.park_embedding_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.log_lambda_min >= self.log_lambda_max:
            raise ValueError("log_lambda_min must be less than log_lambda_max")


class ScoreHead(nn.Module):
    """Maps encoded hitters, pitchers, and game context to expected runs,
    plus (for distribution="negbin") a global dispersion.

    The forward pass returns **log-lambda** (log expected runs), not runs.
    This is deliberate: the NLL consumes the log mean directly, and
    predicting in log space means the network can emit any real number while
    the implied mean stays positive. Use predict_runs() to obtain expected
    runs. The mean is the same quantity under either distribution; negbin
    only widens the distribution around it via `log_r` (a learned scalar,
    stored as a log so r stays positive).

    Shapes
    ------
    hitter_vec  : (B, H_h)
    pitcher_vec : (B, H_p)
    context     : (B, F_c)   (BaseNet passes context + park embedding here)
    returns     : (B,)   log-lambda
    """

    def __init__(self, cfg: HeadConfig, in_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_dim = in_dim
        self.net = self._build_net(in_dim, cfg)
        if cfg.distribution == "negbin":
            # r = 5 to start (variance ~ 1.9x the mean at 4.5 runs); learned from there.
            self.log_r = nn.Parameter(torch.tensor(math.log(5.0)))
        else:
            self.register_parameter("log_r", None)

    @property
    def dispersion(self) -> float | None:
        """The learned negbin dispersion r, or None for a Poisson head."""
        return None if self.log_r is None else float(torch.exp(self.log_r.detach()))

    @staticmethod
    def _build_net(in_dim: int, cfg: HeadConfig) -> nn.Module:
        layers: list[nn.Module] = []
        d_in = in_dim
        for dim in cfg.hidden_dims:
            layers += [
                nn.Linear(d_in, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
            d_in = dim
        layers.append(nn.Linear(d_in, 1))
        return nn.Sequential(*layers)

    def forward(
        self,
        hitter_vec: torch.Tensor,
        pitcher_vec: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Returns log-lambda of shape (B,). Not runs."""
        x = torch.cat([hitter_vec, pitcher_vec, context], dim=-1)
        log_lambda = self.net(x).squeeze(-1)
        return log_lambda.clamp(self.cfg.log_lambda_min, self.cfg.log_lambda_max)

    @torch.no_grad()
    def predict_runs(
        self,
        hitter_vec: torch.Tensor,
        pitcher_vec: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Expected runs, shape (B,). Inference only — never feed to loss()."""
        return torch.exp(self(hitter_vec, pitcher_vec, context))

    def loss(self, log_lambda: torch.Tensor, runs: torch.Tensor) -> torch.Tensor:
        """Full NLL (nats/team-game, including log y!) under cfg.distribution.
        Expects log-lambda from forward(), not runs."""
        if log_lambda.shape != runs.shape:
            raise ValueError(
                f"shape mismatch: log_lambda {tuple(log_lambda.shape)} vs "
                f"runs {tuple(runs.shape)} — broadcasting would silently "
                f"produce an (B, B) loss matrix"
            )
        return nll(log_lambda, runs, self.cfg.distribution, self.log_r)