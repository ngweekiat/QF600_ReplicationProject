"""Macro stress tests (jkmp.evaluation.macro_stress): target splicing, hike / cut / hold months, the S1 / S2 / S3
classification (including the 2026 path), window sums, the current-book factor P&L and the S3 mixture."""

import numpy as np
import pandas as pd
import pytest

from jkmp.covariance import CovParts
from jkmp.evaluation.macro_stress import (book_exposure, book_window_returns, classify_events, distribution_stats,
                                          mixture, monthly_factor_returns, policy_moves, policy_target,
                                          scenario_starts, simulate, summarise_windows, tc_ratio, window_sums)


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
