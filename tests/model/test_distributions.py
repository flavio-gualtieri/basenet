"""Count-distribution math (basenet.distributions) against scipy, plus head wiring."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats

from basenet.distributions import log_prob, nll, pmf_table
from basenet.head.head import HeadConfig, ScoreHead

Y = torch.tensor([0, 1, 3, 4, 7, 12, 20], dtype=torch.float64)
MU = torch.tensor([4.5, 2.0, 3.3, 5.1, 6.0, 4.4, 4.4], dtype=torch.float64)


def test_poisson_log_prob_matches_scipy():
    ours = log_prob(torch.log(MU), Y, "poisson").numpy()
    np.testing.assert_allclose(ours, stats.poisson.logpmf(Y.numpy(), MU.numpy()), rtol=1e-10)


def test_poisson_full_nll_is_torch_loss_plus_log_y_factorial():
    """Same gradients as torch's default PoissonNLLLoss -- only the constant
    log(y!) term differs, which is what makes it comparable to negbin."""
    torch_loss = torch.nn.PoissonNLLLoss(log_input=True, full=False)(torch.log(MU), Y)
    assert nll(torch.log(MU), Y, "poisson").item() == pytest.approx(
        torch_loss.item() + torch.lgamma(Y + 1).mean().item(), rel=1e-10)


@pytest.mark.parametrize("r", [0.7, 3.8, 50.0])
def test_negbin_log_prob_matches_scipy(r):
    """scipy's nbinom(n=r, p=r/(r+mu)) has mean mu and variance mu + mu^2/r."""
    ours = log_prob(torch.log(MU), Y, "negbin", torch.tensor(math.log(r), dtype=torch.float64)).numpy()
    ref = stats.nbinom.logpmf(Y.numpy(), r, r / (r + MU.numpy()))
    np.testing.assert_allclose(ours, ref, rtol=1e-9)


def test_negbin_approaches_poisson_as_r_grows():
    big_r = torch.tensor(math.log(1e7), dtype=torch.float64)
    np.testing.assert_allclose(log_prob(torch.log(MU), Y, "negbin", big_r).numpy(),
                               log_prob(torch.log(MU), Y, "poisson").numpy(), atol=1e-5)


@pytest.mark.parametrize("dist,log_r", [("poisson", None), ("negbin", torch.tensor(math.log(3.8)))])
def test_pmf_table_sums_to_one_with_mean_mu(dist, log_r):
    mu = torch.tensor([2.0, 4.5, 7.0])
    table = pmf_table(torch.log(mu), dist, log_r, max_runs=80)
    np.testing.assert_allclose(table.sum(1).numpy(), 1.0, atol=1e-5)
    np.testing.assert_allclose((table * torch.arange(81)).sum(1).numpy(), mu.numpy(), rtol=1e-4)


def test_head_dispersion_parameter_only_for_negbin():
    poisson = ScoreHead(HeadConfig(distribution="poisson"), in_dim=8)
    negbin = ScoreHead(HeadConfig(distribution="negbin"), in_dim=8)
    assert poisson.log_r is None and poisson.dispersion is None
    assert isinstance(negbin.log_r, torch.nn.Parameter)
    assert negbin.dispersion == pytest.approx(5.0)
    # the dispersion is trained: it gets a gradient from the loss
    log_mu = negbin(torch.randn(6, 3), torch.randn(6, 3), torch.randn(6, 2))
    negbin.loss(log_mu, torch.tensor([0., 2., 3., 5., 9., 14.])).backward()
    assert negbin.log_r.grad is not None and negbin.log_r.grad != 0


def test_head_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="distribution"):
        HeadConfig(distribution="gaussian")
