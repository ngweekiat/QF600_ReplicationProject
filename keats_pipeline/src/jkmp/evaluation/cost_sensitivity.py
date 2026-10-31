"""Trading-cost sensitivity by re-costing fixed weights (extension, research log trial 3). No re-training: the
weights of a run are kept and only Λ changes, so tc = (w/2)·Σ λ_i s_i (Δw_i)² with a scale s per stock-month.

* VIX: s_t = VIX_t / mean VIX (``regimes.vix_cost_scale``), the same for every stock in month t.
* Stock volatility: s_i,t = σ_i,t / σ̄_i,t, σ_i,t = √Σ_ii at t and σ̄_i,t the mean of σ_i over the stock's last
  ``window`` month-ends up to and including t (past data only). Stocks without Σ at t (exits) keep s = 1.

``capacity_aum``: the wealth at which the net information ratio first falls below a threshold, by linear
interpolation in log wealth over the wealth grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from jkmp.covariance import CovParts
from jkmp.portfolio.accounting import portfolio_returns


def vol_cost_scale(cov_parts: Mapping[pd.Timestamp, CovParts], window: int) -> pd.Series:
    """s_i,t = σ_i,t / σ̄_i,t indexed by (id, eom); σ̄ over the last ``window`` month-ends ≤ t with a Σ for the stock."""
    vol = pd.concat([pd.DataFrame({"id": c.ids, "eom": eom, "vol": c.vol()}) for eom, c in sorted(cov_parts.items())])
    vol = vol.sort_values(["id", "eom"])
    vbar = vol.groupby("id")["vol"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return pd.Series((vol["vol"] / vbar).to_numpy(), index=pd.MultiIndex.from_frame(vol[["id", "eom"]]),
                     name="cost_scale")


def recost(weights: pd.DataFrame, panel: pd.DataFrame, wealth: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """``portfolio_returns`` with Λ multiplied by ``scale``: indexed by eom (one value a month) or by (id, eom).
    Stock-months without a scale keep Λ."""
    p = panel[["id", "eom", "ret_ld1", "lambda"]]
    if isinstance(scale.index, pd.MultiIndex):
        s = pd.Series(scale.reindex(pd.MultiIndex.from_frame(p[["id", "eom"]])).to_numpy(), index=p.index)
    else:
        s = p["eom"].map(scale)
    return portfolio_returns(weights, p.assign(**{"lambda": p["lambda"] * s.fillna(1.0)}), wealth)


def capacity_aum(wealth: Sequence[float], ir: Sequence[float], threshold: float) -> float:
    """First wealth where IR crosses below ``threshold`` (log-linear interpolation between grid points).
    NaN-free edge cases: −inf if already below at the smallest wealth, +inf if never below on the grid."""
    w, r = np.asarray(wealth, dtype=float), np.asarray(ir, dtype=float)
    order = np.argsort(w)
    w, r = w[order], r[order]
    if r[0] < threshold:
        return float("-inf")
    for k in range(1, len(w)):
        if r[k] < threshold:
            f = (r[k - 1] - threshold) / (r[k - 1] - r[k])
            return float(np.exp(np.log(w[k - 1]) + f * (np.log(w[k]) - np.log(w[k - 1]))))
    return float("inf")
