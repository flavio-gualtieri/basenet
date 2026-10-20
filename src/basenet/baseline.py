"""Baselines on the same tensors BaseNet sees, under the same distribution.

- `fit_constant`: always predict the training mean (the floor).
- `fit_glm`: a linear count regression -- the bar a deep model actually has
  to clear. If BaseNet can't beat a linear model on its own inputs, its
  extra capacity isn't buying anything.

Both take the same `distribution` as BaseNet's head and, for "negbin", fit
their own dispersion r on train. Comparing a negbin BaseNet against Poisson
baselines would credit the model with what is really just the better
distribution shape.

    from basenet.baseline import fit_glm
    glm = fit_glm(split.train, n_parks=len(split.scaler.park_vocab), distribution="negbin")
    log_mu = glm(split.val)
    glm.log_r            # None for poisson

Design matrix (`glm_design`): mean over the 9 lineup slots of the hitter
features, the starter's vector, the mask-weighted mean of the real bullpen
slots, and the context vector -- all already scaled by the train-fit Scaler
-- plus a one-hot park indicator (the column for UNKNOWN_PARK, 0, is dropped:
an unseen park gets no park effect, same as BaseNet's zero embedding).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Distribution, nll
from .loader import SplitTensors

_INIT_LOG_R = math.log(5.0)


def glm_design(batch: SplitTensors, n_parks: int) -> torch.Tensor:
    """(B, D) design matrix, on CPU (the fit is small; LBFGS is happiest there)."""
    lineup = batch.lineup.cpu().mean(dim=1)
    pitchers = batch.pitchers.cpu()
    sp = pitchers[:, 0]
    bp_mask = batch.pitcher_mask.cpu()[:, 1:].to(pitchers.dtype).unsqueeze(-1)
    bullpen = (pitchers[:, 1:] * bp_mask).sum(dim=1) / bp_mask.sum(dim=1).clamp_min(1.0)
    park = F.one_hot(batch.park.cpu(), num_classes=n_parks + 1)[:, 1:].to(lineup.dtype)
    return torch.cat([lineup, sp, bullpen, batch.context.cpu(), park], dim=-1)


def _make_log_r(distribution: Distribution) -> nn.Parameter | None:
    return nn.Parameter(torch.tensor(_INIT_LOG_R)) if distribution == "negbin" else None


class ConstantBaseline(nn.Module):
    """Same expected runs for every row; negbin adds a fitted dispersion."""

    def __init__(self, mean_runs: float, distribution: Distribution) -> None:
        super().__init__()
        self.distribution = distribution
        self.register_buffer("log_mu", torch.tensor(math.log(max(mean_runs, 1e-6))))
        self.log_r = _make_log_r(distribution)

    def forward(self, batch: SplitTensors) -> torch.Tensor:
        """log expected runs, shape (B,), on CPU."""
        return self.log_mu.expand(len(batch))


class CountGLM(nn.Module):
    def __init__(self, in_dim: int, n_parks: int, distribution: Distribution) -> None:
        super().__init__()
        self.n_parks = n_parks
        self.distribution = distribution
        self.linear = nn.Linear(in_dim, 1)
        self.log_r = _make_log_r(distribution)

    def forward(self, batch: SplitTensors) -> torch.Tensor:
        """log expected runs, shape (B,), on CPU."""
        return self.linear(glm_design(batch, self.n_parks)).squeeze(-1)


def _lbfgs(params: list[torch.Tensor], loss_fn, max_iter: int) -> None:
    opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)


def fit_constant(train: SplitTensors, distribution: Distribution = "poisson") -> ConstantBaseline:
    """The mean is the training mean (the MLE under either distribution); for
    negbin, r is then fit on train."""
    y = train.y.cpu()
    model = ConstantBaseline(y.mean().item(), distribution)
    if model.log_r is not None:
        _lbfgs([model.log_r], lambda: nll(model.log_mu.expand_as(y), y, distribution, model.log_r), 100)
    return model.eval()


def fit_glm(train: SplitTensors, n_parks: int, distribution: Distribution = "poisson",
            l2: float = 1e-2, max_iter: int = 500) -> CountGLM:
    """Full-batch L-BFGS on the full NLL + `l2` * ||w||^2 (bias and r
    unpenalized; bias initialized at log(mean runs) so the fit starts from
    the constant baseline)."""
    x = glm_design(train, n_parks)
    y = train.y.cpu()
    glm = CountGLM(x.shape[1], n_parks, distribution)
    with torch.no_grad():
        glm.linear.weight.zero_()
        glm.linear.bias.fill_(float(torch.log(y.mean().clamp_min(1e-6))))

    def loss():
        return nll(glm.linear(x).squeeze(-1), y, distribution, glm.log_r) + l2 * glm.linear.weight.pow(2).sum()

    _lbfgs(list(glm.parameters()), loss, max_iter)
    return glm.eval()
