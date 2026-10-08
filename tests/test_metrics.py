"""Table 2 statistics by hand (jkmp.evaluation.metrics; R pf_summary)."""

import numpy as np
import pandas as pd
import pytest

from jkmp.evaluation.metrics import table2, utility_flow

PF = pd.DataFrame({"type": "A", "r": [0.01, 0.02, -0.01], "tc": [0.001, 0.002, 0.0],
                   "turnover": [0.1, 0.2, 0.3], "inv": [1.0, 1.2, 1.4], "shorting": [0.0, 0.1, 0.2]})


def test_table2_by_hand_uses_gross_variance():
    row = table2(PF, gamma=10).loc["A"]
    mean, var = 0.02 / 3, 4.666666666666667e-4 / 2   # mean and sample variance (n−1) of gross r
    sd = np.sqrt(var)
    assert row["n"] == 3
    assert row["R"] == pytest.approx(mean * 12)                         # 0.08
    assert row["Vol"] == pytest.approx(sd * np.sqrt(12))                # 0.0529
    assert row["SR_gross"] == pytest.approx(mean / sd * np.sqrt(12))    # 1.512
    assert row["TC"] == pytest.approx(0.012)
    assert row["R_TC"] == pytest.approx(0.068)
    assert row["SR_net"] == pytest.approx((mean - 0.001) / sd * np.sqrt(12))  # 1.285 — gross sd
    assert row["Utility"] == pytest.approx((mean - 0.5 * 10 * var - 0.001) * 12)  # 0.054
    assert row["Utility"] == pytest.approx(0.054)
    assert row["Turnover"] == pytest.approx(0.2) and row["Lev"] == pytest.approx(1.2)
    assert row["Shorting"] == pytest.approx(0.1)
    # With net-return variance the utility would differ: the port must not use it.
    net_var = (PF["r"] - PF["tc"]).var()
    assert row["Utility"] != pytest.approx((mean - 0.5 * 10 * net_var - 0.001) * 12)


def test_utility_flow_averages_to_utility():
    u = utility_flow(PF, gamma=10)
    # Flow uses the population variance (mean of squared deviations), the summary the sample variance.
    assert u.mean() * 12 == pytest.approx((PF["r"].mean() - 0.5 * 10 * PF["r"].var(ddof=0) - PF["tc"].mean()) * 12)
