"""Table 2 statistics by hand (jkmp.evaluation.metrics; R pf_summary)."""

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.evaluation.metrics import table2, utility_flow

DATES = load_config().dates

PF = pd.DataFrame({"type": "A", "eom_ret": pd.date_range("2020-01-31", periods=3, freq="ME"), "r": [0.01, 0.02, -0.01], "tc": [0.001, 0.002, 0.0],
                   "turnover": [0.1, 0.2, 0.3], "inv": [1.0, 1.2, 1.4], "shorting": [0.0, 0.1, 0.2]})


def test_table2_by_hand_uses_gross_variance():
    row = table2(PF, gamma=10, dates=DATES).loc["A"]
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


def _two(n=240, seed=3):
    rng = np.random.default_rng(seed)
    months = pd.date_range("1981-01-31", periods=n, freq="ME")
    a = pd.DataFrame({"type": "A", "eom_ret": months, "r": rng.normal(0.01, 0.03, n), "tc": 0.0})
    b = pd.DataFrame({"type": "B", "eom_ret": months, "r": 0.5 * a["r"] + rng.normal(0.0, 0.03, n), "tc": 0.002})
    return pd.concat([a, b], ignore_index=True)


def test_prob_outperformance_matches_r_formula_and_is_complementary():
    from scipy.stats import norm

    from jkmp.evaluation.metrics import prob_outperformance
    pfs = _two()
    p = prob_outperformance(pfs, 10, DATES, ["A", "B"])
    u = utility_flow(pfs, 10)
    d = u[pfs["type"] == "A"].to_numpy() - u[pfs["type"] == "B"].to_numpy()
    assert p.loc["A", "B"] == pytest.approx(norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))))
    assert p.loc["A", "B"] + p.loc["B", "A"] == pytest.approx(1.0) and np.isnan(p.loc["A", "A"])


def test_correlations_and_subperiods():
    from jkmp.evaluation.metrics import return_correlations, subperiod_table
    pfs = _two().assign(turnover=0.1, inv=1.0, shorting=0.0)
    c = return_correlations(pfs, DATES, ["A", "B"])
    assert c.loc["A", "B"] == pytest.approx(np.corrcoef(pfs.loc[pfs.type == "A", "r"], pfs.loc[pfs.type == "B", "r"])[0, 1])
    t = subperiod_table(pfs, 10, DATES, {"early": ("1981-01-31", "1990-12-31"), "late": ("1991-01-31", "2000-12-31")},
                        ["Utility", "TC"])
    early = table2(pfs[pfs["eom_ret"] <= "1990-12-31"], 10, DATES)
    assert t.loc["A", ("Utility", "early")] == pytest.approx(early.loc["A", "Utility"])
    assert list(t.columns) == [("Utility", "early"), ("Utility", "late"), ("TC", "early"), ("TC", "late")]
