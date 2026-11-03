"""Macro stress tests (jkmp.evaluation.macro_stress): target splicing, hike / cut / hold months, the S1 / S2 / S3
classification (including the 2026 path), window sums, the current-book factor P&L, the S3 mixture, and the
Addendum A rate betas, December surprises and cost to exit."""

import numpy as np
import pandas as pd
import pytest

from jkmp.covariance import CovParts
from jkmp.evaluation.macro_stress import (book_exposure, book_window_returns, classify_events, december_surprises,
                                          distribution_stats, exit_cost, mixture, month_end_changes,
                                          monthly_factor_returns, policy_moves, policy_target, rate_betas,
                                          rate_components, rate_scenario, scenario_starts, simulate, summarise_windows, tc_ratio,
                                          window_sums)


def _daily(points):
    """Daily target series from (date, value) change points, held until the next point."""
    days = pd.date_range(points[0][0], "2026-11-02", freq="D")
    s = pd.Series(np.nan, index=days)
    for d, v in points:
        s[pd.Timestamp(d)] = v
    return s.ffill()


PATH_2024_26 = [("2024-08-01", 5.50), ("2024-09-18", 5.00), ("2024-11-07", 4.75), ("2024-12-18", 4.50),
                ("2025-09-17", 4.25), ("2025-10-29", 4.00), ("2025-12-10", 3.75), ("2026-09-17", 4.00)]


def test_splice_takes_later_series_from_its_first_date():
    old = pd.DataFrame({"date": pd.date_range("2008-12-10", "2008-12-20"), "value": 1.0})
    new = pd.DataFrame({"date": pd.date_range("2008-12-16", "2008-12-31"), "value": 0.25})
    s = policy_target({"A": old, "B": new}, ["A", "B"])
    assert s[:"2008-12-15"].eq(1.0).all() and s["2008-12-16":].eq(0.25).all() and not s.index.duplicated().any()


def test_moves_from_month_end_targets():
    m = policy_moves(_daily(PATH_2024_26), 0.0).set_index("eom")
    assert m.loc["2024-09-30", "move"] == "cut" and m.loc["2024-10-31", "move"] == "hold"
    assert m.loc["2026-09-30", "move"] == "hike" and m.loc["2026-09-30", "change"] == pytest.approx(0.25)
    assert m.index[0] == pd.Timestamp("2024-09-30")  # first (partial) month dropped


def test_classifies_the_2026_path():
    e = classify_events(policy_moves(_daily(PATH_2024_26), 0.0), outcome_months=2).set_index("eom")
    assert e.loc["2026-09-30", "s1"] and e.loc["2026-09-30", "prev_move"] == "cut"   # first hike after cuts
    assert e.loc["2026-10-31", "s2"] and not e.loc["2026-09-30", "s2"]              # hold right after it
    assert pd.isna(e.loc["2026-10-31", "s3_outcome"])                                # December not yet observed
    assert not e["s1"].drop(pd.Timestamp("2026-09-30")).any()


def test_s3_outcome_and_window_offsets():
    path = [("2000-01-01", 1.0), ("2000-03-15", 1.25), ("2000-05-15", 1.5), ("2000-07-10", 1.75)]
    e = classify_events(policy_moves(_daily(path), 0.0), outcome_months=2).set_index("eom")
    # Mar hike, Apr hold (S2; May hike -> "hike"), Jun hold after May hike (S2; Jul hike -> "hike")
    assert e.loc["2000-04-30", "s2"] and e.loc["2000-04-30", "s3_outcome"] == "hike"
    assert e.loc["2000-08-31", "s2"] and e.loc["2000-08-31", "s3_outcome"] == "no hike"
    starts = scenario_starts(e.reset_index(), offset_months=2)
    assert pd.Timestamp("2000-06-30") in starts["S3 next decision: hike"]       # Apr + 2
    assert pd.Timestamp("2000-10-31") in starts["S3 next decision: no hike"]    # Aug + 2


def test_window_sums_skip_incomplete_windows_and_tc_ratio():
    months = pd.date_range("2020-01-31", periods=6, freq="ME")
    monthly = pd.DataFrame({"type": "A", "eom_ret": months, "net": np.arange(6.0), "tc": [1.0, 1, 2, 2, 1, 1]})
    w = window_sums(monthly, [months[1], months[4]], 3, ["net", "tc"])
    assert len(w) == 1 and w.iloc[0]["net"] == 1 + 2 + 3 and w.iloc[0]["tc"] == 5
    assert tc_ratio(w, monthly).iloc[0]["tc_ratio"] == pytest.approx((5 / 3) / (8 / 6))
    s = summarise_windows(w, "net").iloc[0]
    assert s["n"] == 1 and s["mean"] == 6 and s["share_negative"] == 0


def test_book_exposure_and_window_pnl():
    ids = np.array([1, 2, 3])
    X = np.array([[1.0, 0.5], [1.0, -0.5], [1.0, 0.0]])
    F = np.diag([0.002, 0.001])
    parts = CovParts(eom=pd.Timestamp("2025-11-30"), ids=ids, factors=("sec_a", "theme_b"), X=X, F=F,
                     ivar=np.array([0.01, 0.02, 0.03]))
    a = np.array([0.02, -0.01, -0.01])
    b, idio = book_exposure(pd.DataFrame({"id": ids, "eom": parts.eom, "a": a}), parts)
    assert b.to_numpy() == pytest.approx(X.T @ a) and idio == pytest.approx((parts.ivar * a**2).sum())
    days = pd.bdate_range("2026-01-01", "2026-03-31")
    daily = pd.DataFrame({"sec_a": 0.001, "theme_b": -0.002}, index=days)
    fm = monthly_factor_returns(daily, ["sec_a", "theme_b"])
    pnl = book_window_returns(b, fm, [pd.Timestamp("2026-01-31"), pd.Timestamp("2026-03-31")], 2)
    assert list(pnl.index) == [pd.Timestamp("2026-01-31")]  # Mar + 1 month is outside the data
    assert pnl.iloc[0] == pytest.approx(fm.iloc[:2].to_numpy().sum(axis=0) @ b.to_numpy())


def test_simulation_and_mixture_recover_the_mixed_mean():
    rng = np.random.default_rng(1)
    hike = simulate(pd.Series([0.01, 0.03]), idio_var=1e-6, length=3, cost=0.001, n=40_000, rng=rng)
    pause = simulate(pd.Series([-0.02]), idio_var=1e-6, length=3, cost=0.001, n=40_000, rng=rng)
    assert hike.mean() == pytest.approx(0.019, abs=5e-4) and pause.mean() == pytest.approx(-0.021, abs=5e-4)
    mix = mixture(hike, pause, 0.8, rng)
    assert mix.mean() == pytest.approx(0.8 * 0.019 + 0.2 * -0.021, abs=1e-3)
    st = distribution_stats(mix)
    assert st["p05"] <= st["p50"] <= st["p95"] and st["es05"] <= st["p05"]
    with pytest.raises(ValueError):
        simulate(pd.Series(dtype=float), 1e-6, 3, 0.0, 10, rng)


def test_month_end_changes():
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    level = pd.Series(1.0, index=days)
    level["2020-02-14":] = 1.25
    level["2020-03-20":] = 1.0
    dy = month_end_changes(pd.DataFrame({"date": days, "value": level.to_numpy()}))
    assert list(dy.index) == [pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-31")]
    assert dy.tolist() == pytest.approx([0.25, -0.25])


def test_rate_betas_recover_known_slopes_and_standard_errors():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2000-01-31", periods=600, freq="ME")
    dy = pd.Series(rng.normal(0, 0.2, len(idx)), index=idx, name="dy")
    noise = rng.normal(0, 0.01, (len(idx), 2))
    ret = pd.DataFrame({"f1": 0.001 - 0.05 * dy + noise[:, 0], "f2": 0.02 * dy + noise[:, 1]}, index=idx)
    rb = rate_betas(ret, dy, idx[0], idx[-1])
    assert rb.n == 600 and rb.beta.to_numpy() == pytest.approx([-0.05, 0.02], abs=0.006)
    sxx = ((dy - dy.mean()) ** 2).sum()
    assert np.sqrt(rb.cov_beta[0, 0]) == pytest.approx(np.sqrt(rb.resid_cov[0, 0] / sxx))
    assert np.sqrt(rb.resid_cov[1, 1]) == pytest.approx(0.01, rel=0.1)
    with pytest.raises(ValueError):
        rate_betas(ret, dy, idx[0], idx[10])


def test_december_surprises_are_zero_in_expectation():
    s = december_surprises(0.8, 0.25)
    assert s["hike (as expected)"] == pytest.approx(0.05) and s["no hike (dovish surprise)"] == pytest.approx(-0.20)
    assert 0.8 * s["hike (as expected)"] + 0.2 * s["no hike (dovish surprise)"] == pytest.approx(0.0)


def test_rate_scenario_expectation_and_range():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(5)
    dy = pd.Series(rng.normal(0, 0.2, 120), index=idx, name="dy")
    ret = pd.DataFrame({"a": -0.04 * dy + rng.normal(0, 0.01, 120), "b": rng.normal(0, 0.01, 120)}, index=idx)
    rb = rate_betas(ret, dy, idx[0], idx[-1])
    b = pd.Series({"b": 0.5, "a": 2.0})
    out = rate_scenario(b, rb, idio_var=1e-5, shock=1.0, interval=0.9)
    s = 2.0 * rb.beta["a"] + 0.5 * rb.beta["b"]
    assert out["sensitivity"] == pytest.approx(s) and out["expected"] == pytest.approx(s)
    assert out["low"] < out["expected"] < out["high"]
    assert out["expected"] - out["low"] == pytest.approx(out["high"] - out["expected"])


def test_exit_cost_quadratic_and_split():
    pos = pd.Series({1: 0.02, 2: -0.01})
    lam = pd.Series({1: 2e-10, 2: 4e-10, 3: 1.0})
    one = exit_cost(pos, lam, wealth=1e10, multiplier=1.0, months=1)
    assert one == pytest.approx(0.5 * 1e10 * (2e-10 * 0.02**2 + 4e-10 * 0.01**2))
    assert exit_cost(pos, lam, 1e10, 3.0, 1) == pytest.approx(3 * one)
    assert exit_cost(pos, lam, 1e10, 1.0, 3) == pytest.approx(one / 3)
    with pytest.raises(ValueError):
        exit_cost(pd.Series({9: 0.1}), lam, 1e10, 1.0, 1)


def test_rate_components_reproduce_rate_scenario_for_any_shock():
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(8)
    dy = pd.Series(rng.normal(0, 0.2, 120), index=idx, name="dy")
    ret = pd.DataFrame({"a": -0.03 * dy + rng.normal(0, 0.01, 120), "b": 0.01 * dy + rng.normal(0, 0.02, 120)}, index=idx)
    rb = rate_betas(ret, dy, idx[0], idx[-1])
    b = pd.Series({"a": 1.5, "b": -0.4})
    c = rate_components(b, rb, idio_var=2e-5)
    assert sum(c["contributions"].values()) == pytest.approx(c["sensitivity"])
    for shock in (-0.2, 0.05, 1.0):
        sc = rate_scenario(b, rb, 2e-5, shock, 0.9)
        half = 1.6448536269514722 * np.sqrt(c["resid_var"] + (c["se_per_pp"] * shock) ** 2)
        assert sc["expected"] == pytest.approx(c["sensitivity"] * shock)
        assert sc["high"] - sc["expected"] == pytest.approx(half)
