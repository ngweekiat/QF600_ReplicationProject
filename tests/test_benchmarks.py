"""Benchmark target weights (jkmp.portfolio.benchmarks)."""

import numpy as np
import pandas as pd

from jkmp.covariance import CovParts
from jkmp.portfolio.benchmarks import equal_weight_targets, market_targets, min_variance_targets

EOM = pd.Timestamp("2020-01-31")
UNIVERSE = pd.DataFrame({"id": [1, 2, 3], "eom": EOM, "me": [600.0, 300.0, 100.0]})


def test_market_and_equal_weights():
    assert market_targets(UNIVERSE)["w"].tolist() == [0.6, 0.3, 0.1]
    np.testing.assert_allclose(equal_weight_targets(UNIVERSE)["w"], [1 / 3] * 3)


def test_min_variance_weights_sum_to_one_and_equalise_marginal_risk():
    rng = np.random.default_rng(1)
    parts = CovParts(EOM, np.array([1, 2, 3]), ("f",), rng.normal(size=(3, 1)) * 0.05,
                     np.array([[1.0]]), np.array([0.004, 0.002, 0.003]))
    w = min_variance_targets(UNIVERSE, {EOM: parts})["w"].to_numpy()
    assert w.sum() == np.float64(1.0) or abs(w.sum() - 1) < 1e-12
    marginal = parts.full() @ w
    np.testing.assert_allclose(marginal, marginal[0])  # Σw ∝ 1
