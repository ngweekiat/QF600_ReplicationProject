"""Offline unit tests for jkmp.costs and jkmp.data.prepare (synthetic data)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from jkmp.costs import drift_factor, price_impact_lambda, wealth_path
from jkmp.data.prepare import add_timing_columns, ff12_industry, flag_history

EOMS = pd.date_range("2024-09-30", periods=4, freq="ME")  # Sep, Oct, Nov, Dec 2024


def _market(returns):
    return pd.DataFrame({"eom_ret": EOMS[1:], "tret": returns})


def test_wealth_path_rolls_back_exactly_and_ends_at_wealth_end():
    w = wealth_path(1.0e10, dt.date(2024, 12, 31), _market([0.10, -0.05, 0.02]))
    assert w["eom"].tolist() == list(EOMS)
    assert w["wealth"].iloc[-1] == 1.0e10
    # Forward check: w_t = w_{t-1} (1 + R_t).
    np.testing.assert_allclose(w["wealth"].iloc[:-1].to_numpy() * (1 + np.array([0.10, -0.05, 0.02])),
                               w["wealth"].iloc[1:].to_numpy())
    assert w["mu_ld1"].iloc[:-1].tolist() == [0.10, -0.05, 0.02] and np.isnan(w["mu_ld1"].iloc[-1])


def test_wealth_path_requires_returns_through_end():
    with pytest.raises(ValueError):
        wealth_path(1.0e10, dt.date(2025, 1, 31), _market([0.01, 0.01, 0.01]))


def test_timing_columns_use_the_right_months():
    wealth = wealth_path(1.0, dt.date(2024, 12, 31), _market([0.10, -0.05, 0.02]))
    rf = pd.Series([0.001, 0.002, 0.003, 0.004], index=EOMS)
    chars = pd.DataFrame({"id": [1], "eom": [EOMS[1]], "ret_exc": [0.05], "ret_exc_lead1m": [0.07]})  # t = Oct
    row = add_timing_columns(chars, rf, wealth).iloc[0]
    assert row["ret_ld1"] == 0.07
    assert row["tr_ld1"] == pytest.approx(0.07 + 0.003)  # RF of November (t+1)
    assert row["tr_ld0"] == pytest.approx(0.05 + 0.002)  # RF of October (t)
    assert row["mu_ld1"] == -0.05 and row["mu_ld0"] == 0.10  # market in Nov / in Oct


def test_flag_history_needs_consecutive_months():
    eoms = pd.date_range("2023-01-31", periods=14, freq="ME")
    full = pd.DataFrame({"id": 1, "eom": eoms})
    gap = pd.DataFrame({"id": 2, "eom": eoms.delete(5)})  # one missing month
    flags = flag_history(pd.concat([full, gap], ignore_index=True), n_lags=12)
    assert flags[:14].tolist() == [False] * 12 + [True, True]
    assert not flags[14:].any()


def test_drift_factor():
    g = drift_factor(pd.Series([0.10, np.nan]), pd.Series([0.05, 0.05]))
    assert g.iloc[0] == pytest.approx(1.10 / 1.05) and g.iloc[1] == 1.0  # R: missing g -> 1


def test_ff12_industry():
    sic = pd.Series([2834, 3714, 6021, 7372, 9999, 3693])
    assert ff12_industry(sic).tolist() == ["Hlth", "Durbl", "Money", "BusEq", "Other", "Hlth"]


def test_price_impact_lambda():
    # Paper example: $5m traded in a $500m/day stock costs 0.5 * 0.2/500m * (5m)^2 = $5,000.
    lam = price_impact_lambda(pd.Series([500e6]), 0.1).iloc[0]
    assert 0.5 * lam * 5e6**2 == pytest.approx(5000.0)
