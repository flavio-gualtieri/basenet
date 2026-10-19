"""Linear Poisson regression (GLM) baseline on the same tensors BaseNet sees.

The constant-mean baseline in train.py is the floor; this is the bar a deep
model actually has to clear. If BaseNet can't beat a linear model on its own
inputs, its extra capacity isn't buying anything.

    from basenet.baseline import fit_glm
    glm = fit_glm(split.train, n_parks=len(split.scaler.park_vocab))
    log_lambda = glm(split.val)

Design matrix (`glm_design`): mean over the 9 lineup slots of the hitter
features, the starter's vector, the mask-weighted mean of the real bullpen
slots, and the context vector -- all already scaled by the train-fit Scaler
-- plus a one-hot park indicator (the column for UNKNOWN_PARK, 0, is dropped:
an unseen park gets no park effect, same as BaseNet's zero embedding).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loader import SplitTensors


def glm_design(batch: SplitTensors, n_parks: int) -> torch.Tensor:
    """(B, D) design matrix, on CPU (the fit is small; LBFGS is happiest there)."""
    lineup = batch.lineup.cpu().mean(dim=1)
    pitchers = batch.pitchers.cpu()
    sp = pitchers[:, 0]
    bp_mask = batch.pitcher_mask.cpu()[:, 1:].to(pitchers.dtype).unsqueeze(-1)
    bullpen = (pitchers[:, 1:] * bp_mask).sum(dim=1) / bp_mask.sum(dim=1).clamp_min(1.0)
    park = F.one_hot(batch.park.cpu(), num_classes=n_parks + 1)[:, 1:].to(lineup.dtype)
    return torch.cat([lineup, sp, bullpen, batch.context.cpu(), park], dim=-1)


class PoissonGLM(nn.Module):
    def __init__(self, in_dim: int, n_parks: int) -> None:
        super().__init__()
        self.n_parks = n_parks
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, batch: SplitTensors) -> torch.Tensor:
        """log-lambda, shape (B,), on CPU."""
        return self.linear(glm_design(batch, self.n_parks)).squeeze(-1)


def fit_glm(train: SplitTensors, n_parks: int, l2: float = 1e-2, max_iter: int = 500) -> PoissonGLM:
    """Full-batch L-BFGS on Poisson NLL + `l2` * ||w||^2 (bias unpenalized,
    initialized at log(mean runs) so the fit starts from the constant baseline)."""
    x = glm_design(train, n_parks)
    y = train.y.cpu()
    glm = PoissonGLM(x.shape[1], n_parks)
    with torch.no_grad():
        glm.linear.weight.zero_()
        glm.linear.bias.fill_(float(torch.log(y.mean().clamp_min(1e-6))))
    criterion = nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
    opt = torch.optim.LBFGS(glm.parameters(), max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = criterion(glm.linear(x).squeeze(-1), y) + l2 * glm.linear.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    glm.eval()
    return glm
