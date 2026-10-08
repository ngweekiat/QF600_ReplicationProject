"""Properties of the trading-speed matrix m (Lemma 1) — jkmp.portfolio.trading_speed."""

import numpy as np
import pytest

from jkmp.portfolio.trading_speed import lemma1_residual, trading_speed

RNG = np.random.default_rng(42)
N = 5
A = RNG.normal(size=(N, N)) * 0.03
SIGMA = A @ A.T + np.diag(np.full(N, 0.002))  # monthly covariance, vols ≈ 5–10%
LAM = RNG.uniform(1e-10, 1e-9, N)             # price impact for $0.2–2B daily volume
KW = dict(gamma=10.0, rf=0.002, mu=0.007)


def test_m_goes_to_zero_as_wealth_goes_to_zero():
    m = trading_speed(SIGMA, LAM, wealth=1e-6, **KW)
    assert np.abs(m).max() < 1e-8  # tiny investor: trades fully to the aim


def test_m_goes_to_identity_as_wealth_goes_to_infinity():
    # I − m shrinks like √(1/wealth) (scalar case: m̃ = 1 − √x + …), so convergence is slow but steady.
    gaps = [np.abs(trading_speed(SIGMA, LAM, wealth=w, **KW) - np.eye(N)).max() for w in (1e12, 1e16, 1e20, 1e24)]
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3]
    assert gaps[-1] < 1e-7  # huge investor: barely trades


@pytest.mark.parametrize("wealth", [1e9, 1e10, 1e11])
def test_eigenvalues_in_unit_interval_and_fixed_point(wealth):
    m = trading_speed(SIGMA, LAM, wealth=wealth, **KW)
    eig = np.linalg.eigvals(m)
    assert np.abs(eig.imag).max() < 1e-10
    assert ((eig.real > 0) & (eig.real < 1)).all()
    m_tilde = trading_speed(SIGMA, LAM, wealth=wealth, return_tilde=True, **KW)
    np.testing.assert_allclose(m_tilde, m_tilde.T)
    # R's 10 iterations leave a small Lemma 1 residual (m within ~1e-3 of the fixed point); more converge.
    assert lemma1_residual(m_tilde, SIGMA, LAM, wealth, **KW) < 1e-3
    converged = trading_speed(SIGMA, LAM, wealth=wealth, iterations=300, return_tilde=True, **KW)
    assert lemma1_residual(converged, SIGMA, LAM, wealth, **KW) < 1e-9
    np.testing.assert_allclose(m_tilde, converged, atol=2e-3)


def test_larger_investor_trades_more_slowly():
    eig = [np.linalg.eigvals(trading_speed(SIGMA, LAM, wealth=w, **KW)).real.mean() for w in (1e9, 1e10, 1e11)]
    assert eig[0] < eig[1] < eig[2]
