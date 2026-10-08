"""Static-ML (jkmp.portfolio.static_ml): Λ = 0 reduces to Markowitz-ML; one recursion step by hand;
hyperparameter selection by cumulative utility, applied to the next return year."""

import numpy as np
import pandas as pd
import pytest

from jkmp.covariance import CovParts
from jkmp.portfolio.ml_benchmarks import markowitz_targets
from jkmp.portfolio.static_ml import month_inputs, select_hypers, sigma_gam_adj, static_weights

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

RNG = np.random.default_rng(7)
DATES = pd.date_range("2020-01-31", periods=3, freq="ME")
IDS = np.array([1, 2, 3])
GAMMA = 10.0


def _setup(lam):
    rows = []
    for t, eom in enumerate(DATES):
        for i in IDS:
            rows.append({"id": i, "eom": eom, "me": 100.0 * i, "in_universe": True, "lambda": lam,
                         "pred_ld1": 0.01 * (i - 2) + 0.002 * t, "tr_ld1": 0.01 * i, "ret_ld1": 0.0})
    panel = pd.DataFrame(rows)
    wealth = pd.DataFrame({"eom": DATES, "wealth": 1e10, "mu_ld1": 0.01})
    cov = {eom: CovParts(eom, IDS, ("f",), RNG.normal(size=(3, 1)) * 0.05, np.array([[0.002]]),
                         np.array([0.004, 0.003, 0.005])) for eom in DATES}
    return panel, wealth, cov


def test_sigma_gam_adj_cov_add():
    s = np.array([[2.0, 0.5], [0.5, 1.0]])
    np.testing.assert_allclose(sigma_gam_adj(s, 2.0), [[6.0, 0.5], [0.5, 3.0]])


def test_static_ml_with_zero_lambda_equals_markowitz_ml():
    panel, wealth, cov = _setup(lam=0.0)
    inputs = month_inputs(panel, wealth, cov, DATES)
    w = static_weights(inputs, panel, wealth, GAMMA, lambda _: (1.0, 1.0, 0.0), charge_exit_tc=True)
    mk = markowitz_targets(panel.assign(eom=panel["eom"]), cov, GAMMA)
    merged = w.merge(mk, on=["id", "eom"], suffixes=("_static", "_mk"))
    assert len(merged) == 9
    np.testing.assert_allclose(merged["w_static"], merged["w_mk"], rtol=1e-10, atol=1e-14)


def test_one_recursion_step_by_hand():
    panel, wealth, cov = _setup(lam=2e-10)
    inputs = month_inputs(panel, wealth, cov, DATES)
    k, u, g = 1 / 3, 0.5, 1.0
    w = static_weights(inputs, panel, wealth, GAMMA, lambda _: (k, u, g), charge_exit_tc=True)
    first = w[w["eom"] == DATES[0]].set_index("id")
    np.testing.assert_allclose(first["w_start"], [100 / 600, 200 / 600, 300 / 600])        # VW start
    S = GAMMA * cov[DATES[0]].full(IDS)
    S = S + np.diag(np.diag(S)) * g
    L = np.diag(np.full(3, 1e10 * k * 2e-10))
    mu = np.array([-0.01, 0.0, 0.01])
    expected = np.linalg.solve(S + L, u * mu + L @ first["w_start"].to_numpy())
    np.testing.assert_allclose(first["w"].to_numpy(), expected, rtol=1e-10)
    second = w[w["eom"] == DATES[1]].set_index("id")                                        # drifted start
    np.testing.assert_allclose(second["w_start"], expected * (1 + 0.01 * IDS) / 1.01, rtol=1e-12)


def test_select_hypers_uses_cumulative_utility_for_next_year():
    months = pd.date_range("2015-01-31", "2016-12-31", freq="ME")
    rows = []
    for hp_no, (k, r_2015, r_2016) in enumerate([(1.0, 0.010, 0.000), (0.2, 0.000, 0.030)]):
        r = np.where(months.year == 2015, r_2015, r_2016)
        rows.append(pd.DataFrame({"eom_ret": months, "r": r, "tc": 0.0, "hp_no": hp_no, "k": k, "u": 1.0, "g": 0.0}))
    best = select_hypers(pd.concat(rows), gamma=10).set_index("year")
    assert best.loc[2016, "k"] == 1.0   # better through Dec 2015 → used in 2016
    assert best.loc[2017, "k"] == 0.2   # cumulative 2015–2016 favours hp 1 → used in 2017
