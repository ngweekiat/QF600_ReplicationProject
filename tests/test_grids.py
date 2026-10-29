"""Frontier grid helpers (jkmp.portfolio.grids, jkmp.evaluation.frontier): wealth rescaling equals the
wealth recursion, Markowitz/Factor scaling hits the vol target with costs growing in the square of the
scale, and the trading-speed matrix depends on (γ, w) only through γ/w."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.costs import wealth_path
from jkmp.evaluation.frontier import scaled_frontier
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.grids import grid_points, point_key, scaled_wealth
from jkmp.portfolio.trading_speed import trading_speed

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
RNG = np.random.default_rng(2)


def test_scaled_wealth_equals_recursion():
    market = pd.DataFrame({"eom_ret": pd.date_range("2020-01-31", periods=24, freq="ME"),
                           "tret": RNG.normal(0.01, 0.04, 24)})
    end = dt.date(2021, 12, 31)
    base = wealth_path(1e10, end, market)
    for level in (1e9, 2.5e11):
        np.testing.assert_allclose(scaled_wealth(base, level, 1e10)["wealth"], wealth_path(level, end, market)["wealth"],
                                   rtol=1e-13)


def test_point_keys_and_grid_order():
    assert point_key(10, 1e10) == "gamma10_wealth10B" and point_key(1, 2.5e11) == "gamma1_wealth250B"
    pts = grid_points([1, 5], [1e9, 1e10])
    assert pts == [(1.0, 1e9), (5.0, 1e9), (1.0, 1e10), (5.0, 1e10)] and len(set(map(lambda p: point_key(*p), pts))) == 4


def test_scaled_frontier_hits_vol_target_and_costs_scale_quadratically():
    dates = pd.date_range("2018-01-31", periods=30, freq="ME")  # returns end 2020-07, before the holdout
    ids = np.arange(1, 6)
    panel = pd.DataFrame([{"id": i, "eom": d, "ret_ld1": RNG.normal(0, 0.06), "lambda": 1e-10 * i}
                          for d in dates for i in ids])
    wealth = pd.DataFrame({"eom": dates, "wealth": 1e10, "mu_ld1": 0.01})
    w = pd.DataFrame([{"id": i, "eom": d, "w": RNG.normal(0, 0.5), "w_start": RNG.normal(0, 0.5), "exit": False,
                       "type": "Markowitz-ML"} for d in dates for i in ids])
    out = scaled_frontier(w, panel, wealth, [0.05, 0.1, 0.2], gamma=10.0, dates=load_config().dates)
    np.testing.assert_allclose(out["Vol"], [0.05, 0.1, 0.2], rtol=1e-10)
    base = portfolio_returns(w, panel, wealth)
    k = 0.1 / (base["r"].std(ddof=1) * np.sqrt(12))
    assert out.loc[1, "TC"] == pytest.approx(base["tc"].mean() * 12 * k**2, rel=1e-10)
    assert out.loc[1, "R"] == pytest.approx(base["r"].mean() * 12 * k, rel=1e-10)


def test_trading_speed_depends_on_gamma_over_wealth_only():
    a = RNG.normal(size=(4, 4)) * 0.05
    sigma = a @ a.T + np.diag(RNG.uniform(0.002, 0.005, 4))
    lam = RNG.uniform(1e-11, 1e-10, 4)
    m1 = trading_speed(sigma, lam, 1e10, 10.0, 0.001, 0.007)
    m2 = trading_speed(sigma, lam, 1e9, 1.0, 0.001, 0.007)       # same γ/w
    m3 = trading_speed(sigma, lam, 1e10, 1.0, 0.001, 0.007)
    np.testing.assert_allclose(m1, m2, rtol=1e-10)
    assert np.abs(m1 - m3).max() > 1e-3
