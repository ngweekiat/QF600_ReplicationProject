"""Monitor flags (jkmp.evaluation.monitor) and the deflated Sharpe ratio (jkmp.evaluation.metrics)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from jkmp.config import load_config
from jkmp.evaluation.metrics import deflated_sharpe
from jkmp.evaluation.monitor import CHECKS, half_life_steps, monitor_table, plot_dashboard

CFG = load_config()
T = 60
FORM = pd.date_range("1990-01-31", periods=T, freq="ME")


def _inputs(drift=0.002):
    rng = np.random.default_rng(8)
    rb = rng.normal(0.007, 0.04, T)
    a = rng.normal(drift, 0.005, T)
    act = pd.DataFrame({"eom": FORM, "eom_ret": FORM + pd.offsets.MonthEnd(1), "r": rb + a, "r_b": rb, "tc": 0.0002,
                        "active": a, "active_net": a - 0.0002})
    ex = pd.DataFrame({"eom": FORM, "te": 0.005, "theme_value": np.where(np.arange(T) == 10, 1.5, 0.2)})
    ic = pd.Series(np.r_[np.full(30, 0.05), np.full(30, -0.05)], index=act["eom_ret"])
    tc_real = pd.Series(np.r_[np.full(50, 0.0001), np.full(10, 0.0005)], index=act["eom_ret"])
    return act, ex, ic, tc_real


def test_flags_fire_where_constructed():
    act, ex, ic, tc_real = _inputs()
    hl = half_life_steps(pd.DataFrame({"year": [1990, 1991, 1992, 1993, 1994], "h": [3, 5, None, None, 3]}),
                         [3, 5, 10, None])
    m = monitor_table(act, ex, ic, tc_real, hl, CFG.monitor, CFG.dates).set_index("eom_ret")
    assert m["flag_theme"].sum() == 1 and m["flag_theme"].iloc[10]
    assert not m["flag_ic"].iloc[:41].any() and m["flag_ic"].iloc[-1]          # IC < 0 for 6+ rolling months
    assert m["flag_tc"].sum() == 8                                              # 10 bad months, flagged from the 3rd
    assert m.loc[m.index.year == 1992, "flag_half_life"].all()                  # 5 -> inf is two steps
    assert not m.loc[m.index.year == 1991, "flag_half_life"].any()
    assert m["ir"].iloc[:35].isna().all() and (m["ir"].iloc[35:] > 0).all()     # positive drift
    expect = (act["active"] / 0.005).rolling(36).std().to_numpy()               # ex-ante TE = true active sd
    np.testing.assert_allclose(m["bias"].to_numpy(), expect, equal_nan=True)
    assert abs(m["bias"].iloc[-1] - 1) < 0.35
    neg = monitor_table(*_inputs(drift=-0.003), None, CFG.monitor, CFG.dates)
    assert neg["flag_ir"].iloc[-1] and neg["flag_drawdown"].iloc[-1]


def test_dashboard_writes_png_and_html(tmp_path):
    m = monitor_table(*_inputs(), None, CFG.monitor, CFG.dates)
    png, html = plot_dashboard(m, CFG.monitor, "test", tmp_path / "dash")
    assert png.stat().st_size > 0 and "flagged months" in html.read_text()
    assert set(CHECKS) <= set(m.columns)


def test_deflated_sharpe_single_trial_is_psr_and_falls_with_trials():
    r = pd.Series(np.random.default_rng(3).normal(0.01, 0.04, 480))
    one = deflated_sharpe(r, [0.25], 1)
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    psr = norm.cdf(sr * np.sqrt(479) / np.sqrt(1 - (z**3).mean() * sr + ((z**4).mean() - 1) / 4 * sr**2))
    assert one["dsr"] == pytest.approx(psr) and one["sr0"] == 0.0
    many = deflated_sharpe(r, [0.05, 0.10, 0.20, 0.25, 0.15], 5)
    assert many["sr0"] > 0 and many["dsr"] < one["dsr"]
