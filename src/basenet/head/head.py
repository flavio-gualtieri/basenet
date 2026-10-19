from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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

    def __post_init__(self) -> None:
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if self.park_embedding_dim < 0:
            raise ValueError(f"park_embedding_dim must be >= 0, got {self.park_embedding_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.log_lambda_min >= self.log_lambda_max:
            raise ValueError("log_lambda_min must be less than log_lambda_max")


class ScoreHead(nn.Module):
    """Maps encoded hitters, pitchers, and game context to a Poisson rate.

    The forward pass returns **log-lambda**, not runs. This is deliberate:
    the Poisson NLL consumes the log rate directly, and predicting in log
    space means the network can emit any real number while the implied rate
    stays positive. Use predict_runs() to obtain expected runs.

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
        self.criterion = nn.PoissonNLLLoss(
            log_input=True, full=False, reduction="mean"
        )

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
        """Poisson NLL. Expects log-lambda from forward(), not runs."""
        if log_lambda.shape != runs.shape:
            raise ValueError(
                f"shape mismatch: log_lambda {tuple(log_lambda.shape)} vs "
                f"runs {tuple(runs.shape)} — broadcasting would silently "
                f"produce an (B, B) loss matrix"
            )
        return self.criterion(log_lambda, runs.float())