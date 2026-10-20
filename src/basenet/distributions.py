"""Count distributions for a team's runs in a game, as FULL log-likelihoods.

"Full" means including the -log(y!) term that torch's PoissonNLLLoss drops
by default. It's constant in the model's parameters (so gradients are the
same either way), but without it Poisson and negative-binomial losses sit on
different scales and can't be compared. With it, exp(-mean NLL) is the
geometric-mean probability a model put on the run totals that actually
happened.

    poisson:  mean mu, variance mu
    negbin:   mean mu, variance mu + mu^2 / r   (dispersion r > 0; r -> inf is Poisson)

Real run totals are overdispersed (variance ~2x the mean), which a Poisson
can't represent: it puts far too little probability on shutouts and
blowouts. See DATA.md / the head docstring.
"""
from __future__ import annotations

from typing import Literal

import torch

Distribution = Literal["poisson", "negbin"]
DISTRIBUTIONS: tuple[str, ...] = ("poisson", "negbin")


def log_prob(log_mu: torch.Tensor, y: torch.Tensor, distribution: Distribution,
             log_r: torch.Tensor | None = None) -> torch.Tensor:
    """Elementwise log P(y | mean exp(log_mu)[, dispersion exp(log_r)])."""
    y = y.to(log_mu.dtype)
    log_y_fact = torch.lgamma(y + 1)
    if distribution == "poisson":
        return y * log_mu - torch.exp(log_mu) - log_y_fact
    if distribution == "negbin":
        if log_r is None:
            raise ValueError("negbin needs log_r (log dispersion)")
        r = torch.exp(log_r)
        log_r_plus_mu = torch.logaddexp(log_r, log_mu)  # log(r + mu), stable
        return (torch.lgamma(y + r) - torch.lgamma(r) - log_y_fact
                + r * (log_r - log_r_plus_mu) + y * (log_mu - log_r_plus_mu))
    raise ValueError(f"unknown distribution {distribution!r}; expected one of {DISTRIBUTIONS}")


def nll(log_mu: torch.Tensor, y: torch.Tensor, distribution: Distribution,
        log_r: torch.Tensor | None = None) -> torch.Tensor:
    """Mean full negative log-likelihood (nats per team-game)."""
    return -log_prob(log_mu, y, distribution, log_r).mean()


def pmf_table(log_mu: torch.Tensor, distribution: Distribution, log_r: torch.Tensor | None = None,
              max_runs: int = 40) -> torch.Tensor:
    """(B, max_runs + 1) probabilities of 0..max_runs runs for each row --
    for calibration tables and win probabilities."""
    k = torch.arange(max_runs + 1, dtype=log_mu.dtype, device=log_mu.device)
    return torch.exp(log_prob(log_mu[:, None], k[None, :], distribution, log_r))
