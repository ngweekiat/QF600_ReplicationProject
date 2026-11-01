"""Benchmark-relative statistics (jkmp.evaluation.metrics active_*) and γ_TE calibration (jkmp.portfolio.active)."""

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.covariance import CovParts
from jkmp.evaluation.metrics import (active_returns, active_stats, active_utility_flow, active_weights,
                                     ex_ante_exposures, prob_greater)
from jkmp.portfolio.active import calibrate

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
DATES = load_config().dates
RNG = np.random.default_rng(11)
N, T = 4, 120
FORM = pd.date_range("1990-01-31", periods=T, freq="ME")


def _setup():
    rb = RNG.normal(0.007, 0.04, T)
    r = 0.9 * rb + RNG.normal(0.002, 0.01, T)
    pf = pd.DataFrame({"eom": FORM, "eom_ret": FORM + pd.offsets.MonthEnd(1), "r": r, "tc": 0.0005})
    bench = pf.assign(r=rb, tc=0.0001)
    me = np.array([4.0, 3.0, 2.0, 1.0])
    a = np.array([0.02, -0.01, 0.0, -0.01])
    panel = pd.DataFrame([{"id": i, "eom": e, "me": me[i]} for e in FORM for i in range(N)])
    weights = pd.DataFrame([{"id": i, "eom": e, "w": me[i] / me.sum() + a[i] - (0.3 if i == 3 else 0.0),
                             "w_start": 0.0, "exit": False} for e in FORM for i in range(N)])
    X = RNG.normal(size=(N, 2))
    cov = {e: CovParts(e, np.arange(N), ("sec_a", "theme_x"), X, np.diag([0.002, 0.001]), np.full(N, 0.003))
           for e in FORM}
    return pf, bench, weights, panel, cov, X


def test_active_returns_and_utility():
    pf, bench, *_ = _setup()
    act = active_returns(pf, bench, DATES)
    np.testing.assert_allclose(act["active"], pf["r"] - bench["r"])
    np.testing.assert_allclose(act["active_net"], (pf["r"] - 0.0005) - (bench["r"] - 0.0001))
    u = active_utility_flow(act, 50.0)
    np.testing.assert_allclose(u, act["active_net"] - 25.0 * (act["active"] - act["active"].mean()) ** 2)
    assert prob_greater(pd.Series([0.01, 0.02, 0.015])) > 0.99
    assert prob_greater(pd.Series([1.0, 2.0, 3.0]), pd.Series([6.0, 7.5, 7.8])) < 1e-3
    with pytest.raises(ValueError, match="different return months"):
        active_returns(pf, bench.iloc[1:], DATES)


def test_active_stats_by_hand():
    pf, bench, weights, panel, cov, X = _setup()
    s = active_stats(pf, bench, weights, panel, cov, 50.0, DATES)
    act = active_returns(pf, bench, DATES)
    sd = act["active"].std(ddof=1)
    assert s["ir_net"] == pytest.approx(act["active_net"].mean() / sd * np.sqrt(12))
    assert s["te_realised"] == pytest.approx(sd * np.sqrt(12))
    assert s["beta"] == pytest.approx(np.polyfit(bench["r"], pf["r"], 1)[0], rel=1e-10)
    a = weights.loc[weights["eom"] == FORM[0], "w"].to_numpy() - np.array([0.4, 0.3, 0.2, 0.1])
    sigma = X @ np.diag([0.002, 0.001]) @ X.T + np.diag(np.full(N, 0.003))
    assert s["te_ex_ante"] == pytest.approx(np.sqrt(a @ sigma @ a) * np.sqrt(12), rel=1e-10)
    assert s["neg_weight_share"] == pytest.approx(0.25)                     # stock 3: 0.1 − 0.01 − 0.3 < 0
    ex = ex_ante_exposures(active_weights(weights, panel), cov)
    np.testing.assert_allclose(ex.loc[0, ["sec_a", "theme_x"]].to_numpy(dtype=float), X.T @ a, rtol=1e-12)


def test_calibrate_interpolates_log_te_on_log_gamma():
    points = pd.DataFrame({"gamma_te": [10.0, 100.0, 1000.0], "te_ex_ante": [0.08, 0.02, 0.005]})
    out = calibrate(points, [0.02, 0.01]).set_index("te_target")["gamma_te"]
    assert out[0.02] == pytest.approx(100.0)
    assert out[0.01] == pytest.approx(np.exp(np.interp(np.log(0.01), np.log([0.005, 0.02]), np.log([1000.0, 100.0]))),
                                      rel=1e-3)                          # rounded to 4 significant figures
    with pytest.raises(ValueError, match="outside"):
        calibrate(points, [0.2])
