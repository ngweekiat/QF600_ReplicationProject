"""Markowitz-ML and Factor-ML target weights (jkmp.portfolio.ml_benchmarks)."""

import numpy as np
import pandas as pd

from jkmp.covariance import CovParts
from jkmp.portfolio.ml_benchmarks import factor_targets, markowitz_targets

EOM = pd.Timestamp("2020-01-31")


def test_factor_ml_quintiles_value_weighted_legs():
    u = pd.DataFrame({"id": range(10), "eom": EOM, "me": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.0],
                      "pred_ld1": [0.01, 0.09, 0.03, 0.05, 0.02, 0.08, 0.04, 0.07, 0.06, 0.10]})
    w = factor_targets(u, n_portfolios=5).set_index("id")["w"]
    # R's ecdf cut: long if rank ≥ 0.8 → ranks 1.0, 0.9, 0.8 = ids 9, 1, 5 (3 stocks), weights by me 10 : 2 : 6.
    np.testing.assert_allclose(w[[9, 1, 5]], [10 / 18, 2 / 18, 6 / 18])
    # Short if rank ≤ 0.2 → ranks 0.1, 0.2 = ids 0, 4 (2 stocks), me 1 : 5.
    np.testing.assert_allclose(w[[0, 4]], [-1 / 6, -5 / 6])
    assert (w.drop([9, 1, 5, 0, 4]) == 0).all()
    assert w[w > 0].sum() == np.float64(1.0) or abs(w[w > 0].sum() - 1) < 1e-12
    assert abs(w[w < 0].sum() + 1) < 1e-12


def test_markowitz_ml_solves_gamma_sigma_w_equals_mu():
    rng = np.random.default_rng(0)
    parts = CovParts(EOM, np.array([1, 2, 3]), ("f",), rng.normal(size=(3, 1)) * 0.05,
                     np.array([[1.0]]), np.array([0.004, 0.002, 0.003]))
    u = pd.DataFrame({"id": [1, 2, 3], "eom": EOM, "pred_ld1": [0.01, -0.005, 0.002]})
    w = markowitz_targets(u, {EOM: parts}, gamma=10)["w"].to_numpy()
    np.testing.assert_allclose(10 * parts.full() @ w, u["pred_ld1"])
