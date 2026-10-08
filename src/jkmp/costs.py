"""Trading-cost inputs and the investor's wealth path.

* ``price_impact_lambda`` — diagonal of Kyle's lambda (Eq. 35): Λ_i = 2·π / dolvol_i = 0.2 / dolvol_i.
* ``drift_factor`` — g_t = (1 + tr_ld0) / (1 + mu_ld0), how a weight drifts over month t (Eq. 6).
* ``wealth_path`` — the fund grows with the market (Sec. 5.1.3), so its size is a stable share of
  the market; wealth is labelled by its value at the end date (here end-2025). Port of
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
    """Investor wealth at each month-end, rolled backwards from ``wealth_end`` at ``end``.

    ``market`` has one row per month: ``eom_ret`` (month-end the return is realised) and ``tret``
    (market *total* return over that month). Returns columns ``eom``, ``wealth`` and ``mu_ld1``,
    the market total return over the month after ``eom`` (NaN at ``end``).

    w_t = w_{t-1} (1 + R_{m,t})  (Sec. 5.1.3), so  w_{t-1} = w_t / (1 + R_{m,t}).
    The R code uses w_{t-1} = w_t (1 - R_{m,t}), a first-order approximation that compounds to a
    noticeably lower early-sample wealth; we use the exact recursion (docs/deviations.md).
    """
    end = pd.Timestamp(end)
    mkt = market.loc[market["eom_ret"] <= end, ["eom_ret", "tret"]].sort_values("eom_ret")
    if mkt.empty or mkt["eom_ret"].iloc[-1] != end:
        raise ValueError(f"market returns must run through the wealth end date {end.date()}")
    if mkt["tret"].isna().any():
        raise ValueError("market returns contain missing values")
    # Wealth at the start of each return month: divide out that month and every later one.
    growth_after = np.cumprod((1.0 + mkt["tret"].to_numpy())[::-1])[::-1]
    out = pd.DataFrame({
        "eom": (mkt["eom_ret"] - pd.offsets.MonthEnd(1)).to_numpy(),
        "wealth": wealth_end / growth_after,
        "mu_ld1": mkt["tret"].to_numpy(),
    })
    last = pd.DataFrame({"eom": [end], "wealth": [wealth_end], "mu_ld1": [np.nan]})
    return pd.concat([out, last], ignore_index=True)
