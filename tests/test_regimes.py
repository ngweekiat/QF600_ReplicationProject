"""Regimes (jkmp.evaluation.regimes): FRED parsing and monthly alignment, Fed / VIX / recession labels,
statistics by regime, cumulative utility, and the VIX re-costing (tc linear in Λ)."""

import numpy as np
import pandas as pd
import pytest

from jkmp.evaluation.metrics import summary_stats, utility_flow
from jkmp.evaluation.regimes import (cumulative_utility, fetch_fred, label_regimes, monthly_series, recost_vix,
                                     regime_table, vix_cost_scale)

RNG = np.random.default_rng(9)
MONTHS = pd.date_range("2020-01-31", periods=12, freq="ME")


def test_fetch_fred_parses_csv_and_drops_missing(tmp_path):
    (tmp_path / "VIXCLS.csv").write_text("observation_date,VIXCLS\n2020-01-02,12.5\n2020-01-03,.\n2020-01-06,13.0\n")
    out = fetch_fred(f"file://{tmp_path}/{{series}}.csv", "VIXCLS")
    assert out["value"].tolist() == [12.5, 13.0] and out["date"].dt.day.tolist() == [2, 6]


def _fred():
    days = pd.bdate_range("2019-09-01", "2020-12-31")
    vix = pd.DataFrame({"date": days, "value": 20 + 10 * np.sin(np.arange(len(days)) / 30)})
    first = pd.date_range("2019-09-01", "2020-12-01", freq="MS")
    ff = np.array([2.0, 1.8, 1.6, 1.6, 1.6, 1.6, 1.6, 0.1, 0.1, 0.1, 0.1, 0.4, 0.8, 1.2, 1.2, 1.2])
    return {"VIXCLS": vix, "FEDFUNDS": pd.DataFrame({"date": first, "value": ff}),
            "USREC": pd.DataFrame({"date": first, "value": (first.month.isin([3, 4]) & (first.year == 2020)).astype(float)})}


def test_monthly_series_aligns_to_month_end():
    fred = _fred()
    m = monthly_series(fred).set_index("eom")
    jan = fred["VIXCLS"][fred["VIXCLS"]["date"].dt.to_period("M") == pd.Period("2020-01")]["value"].mean()
    assert m.loc["2020-01-31", "vix"] == pytest.approx(jan)
    assert m.loc["2020-03-31", "usrec"] == 1.0 and m.loc["2020-02-29", "fedfunds"] == 1.6


def test_labels_fed_stance_vix_terciles_and_recession():
    lab = label_regimes(monthly_series(_fred()), MONTHS, fed_change_months=3, fed_threshold=0.10).set_index("eom")
    assert lab.loc["2020-01-31", "fed_regime"] == "cutting"   # 1.6 vs 1.8 in Oct 2019
    assert lab.loc["2020-02-29", "fed_regime"] == "on hold"   # 1.6 vs 1.6 in Nov 2019
    assert lab.loc["2020-04-30", "fed_regime"] == "cutting"   # 0.1 vs 1.6
    assert lab.loc["2020-09-30", "fed_regime"] == "hiking"    # 0.8 vs 0.1 in Jun 2020
    assert lab["vix_regime"].value_counts().tolist() == [4, 4, 4]
    assert lab.loc[lab["vix"].idxmax(), "vix_regime"] == "high VIX"
    assert lab["recession"].value_counts()["recession"] == 2
    with pytest.raises(ValueError, match="missing"):
        label_regimes(monthly_series(_fred()), pd.date_range("2021-01-31", periods=2, freq="ME"), 3, 0.1)


def _pfs():
    rows = [pd.DataFrame({"type": t, "eom": MONTHS - pd.offsets.MonthEnd(1), "eom_ret": MONTHS,
                          "r": RNG.normal(0.01, 0.04, 12), "tc": RNG.uniform(0, 0.002, 12),
                          "turnover": RNG.uniform(0, 1, 12), "inv": 1.0, "shorting": 0.0}) for t in ("A", "B")]
    return pd.concat(rows, ignore_index=True)


def test_regime_table_matches_summary_stats_on_subsets():
    pfs = _pfs()
    lab = label_regimes(monthly_series(_fred()), MONTHS, 3, 0.10)
    t = regime_table(pfs, lab, gamma=10)
    sub = pfs.merge(lab.rename(columns={"eom": "eom_ret"}), on="eom_ret")
    want = summary_stats(sub[(sub["vix_regime"] == "high VIX") & (sub["type"] == "A")], 10)
    got = t[(t["dimension"] == "vix_regime") & (t["regime"] == "high VIX") & (t["type"] == "A")].iloc[0]
    assert got["Utility"] == pytest.approx(want["Utility"]) and got["n"] == 4


def test_cumulative_utility_sums_monthly_flows():
    pfs = _pfs()
    cum = cumulative_utility(pfs, 10)
    a = pfs[pfs["type"] == "A"]
    assert cum[cum["type"] == "A"]["cum_u"].iloc[-1] == pytest.approx(utility_flow(a, 10).sum())


def test_vix_recost_is_linear_in_lambda():
    pfs = _pfs()
    form = pd.DatetimeIndex(sorted(pfs["eom"].unique()))
    same = recost_vix(pfs, pd.Series(1.0, index=form), 10)
    np.testing.assert_allclose(same["TC_vix"], same["TC"]) and np.testing.assert_allclose(same["dUtility"], 0)
    double = recost_vix(pfs, pd.Series(2.0, index=form), 10)
    np.testing.assert_allclose(double["TC_vix"], 2 * double["TC"])
    np.testing.assert_allclose(double["dUtility"], -double["TC"])          # utility falls by the extra cost
    monthly = pd.DataFrame({"eom": form, "vix": np.linspace(10, 30, len(form))})
    s = vix_cost_scale(monthly, form, "window")
    assert s.mean() == pytest.approx(1.0)
