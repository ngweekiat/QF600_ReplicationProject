"""Trading-cost inputs and the investor's wealth path.

* ``price_impact_lambda`` — diagonal of Kyle's lambda (Eq. 35): Λ_i = 2·π / dolvol_i = 0.2 / dolvol_i.
* ``drift_factor`` — g_t = (1 + tr_ld0) / (1 + mu_ld0), how a weight drifts over month t (Eq. 6).
* ``wealth_path`` — the fund grows with the market (Sec. 5.1.3), so its size is a stable share of
  the market; wealth is labelled by its value at the end date (``investor.wealth_end_date``). Port of
  ``wealth_func`` in reference/jkmp_r/1 - Prepare Data.R.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def price_impact_lambda(dolvol: pd.Series, price_impact: float) -> pd.Series:
    """Λ_i = 2·π / V_i, with V_i the expected daily dollar volume (R: ``lambda := 2/dolvol*settings$pi``).

    With π = 0.1, trading 1% of daily volume moves the price by 0.1% (Frazzini et al., 2018).
    """
    return 2.0 * price_impact / dolvol


def drift_factor(tr_ld0: pd.Series, mu_ld0: pd.Series) -> pd.Series:
    """Weight growth over month t, g_t = (1 + r_f + r_t) / (1 + g^w_t)  (Eq. 6).

    A weight π_{t-1} grows to g_t·π_{t-1} before trading at t; the fund grows with the market, so
    g^w_t = mu_ld0. As in R (``pfml_input_fun``), a missing g is set to 1, i.e. the stock is assumed
    to have earned the fund's return.
    """
    return ((1.0 + tr_ld0) / (1.0 + mu_ld0)).fillna(1.0)


def wealth_path(wealth_end: float, end: dt.date, market: pd.DataFrame) -> pd.DataFrame:
    """Investor wealth at each month-end, equal to ``wealth_end`` at the anchor ``end``: rolled back
    before it and forward after it, over every month in ``market``.

    ``market`` has one row per month: ``eom_ret`` (month-end the return is realised) and ``tret``
    (market *total* return over that month). Returns columns ``eom``, ``wealth`` and ``mu_ld1``,
    the market total return over the month after ``eom`` (NaN at the last month).

    w_t = w_{t-1} (1 + R_{m,t})  (Sec. 5.1.3): forward after the anchor, w_{t-1} = w_t / (1 + R_{m,t})
    before it. The R code only rolls back (its sample ends at the anchor), with w_{t-1} = w_t (1 - R_{m,t}),
    a first-order approximation that compounds to a noticeably lower early-sample wealth; we use the exact
    recursion (docs/deviations.md D9, D14).
    """
    end = pd.Timestamp(end)
    mkt = market[["eom_ret", "tret"]].sort_values("eom_ret")
    if not (mkt["eom_ret"] == end).any():
        raise ValueError(f"market returns must run through the wealth end date {end.date()}")
    if mkt["tret"].isna().any():
        raise ValueError("market returns contain missing values")
    eom = pd.DatetimeIndex([mkt["eom_ret"].iloc[0] - pd.offsets.MonthEnd(1), *mkt["eom_ret"]])
    tret = mkt["tret"].to_numpy()
    # Growth index at each month-end; wealth is the index rescaled to wealth_end at the anchor.
    level = np.concatenate([[1.0], np.cumprod(1.0 + tret)])
    return pd.DataFrame({
        "eom": eom,
        "wealth": wealth_end * level / level[eom.get_loc(end)],
        "mu_ld1": np.append(tret, np.nan),
    })
