"""Table 2 statistics. Port of ``pf_summary`` and ``utility_t`` in reference/jkmp_r/6 - Base analysis.R.

Input: monthly performance rows (``jkmp.portfolio.accounting.portfolio_returns``) with columns
``type, r, tc, turnover, inv, shorting``. As in R, volatility, the net Sharpe ratio and utility all use
the variance of the **gross** return r (not r − tc); sample moments use n − 1 like R's ``sd``/``var``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TABLE2_COLUMNS = ["n", "R", "Vol", "SR_gross", "TC", "R_TC", "SR_net", "Utility", "Turnover", "Lev", "Shorting"]


def utility_flow(pf: pd.DataFrame, gamma: float) -> pd.Series:
    """Monthly realised utility r − tc − ½γ(r − r̄)², r̄ the mean gross return of each ``type``."""
    rbar = pf.groupby("type")["r"].transform("mean")
    return pf["r"] - pf["tc"] - 0.5 * gamma * (pf["r"] - rbar) ** 2


def summary_stats(pf: pd.DataFrame, gamma: float) -> pd.Series:
    """One method's Table 2 row (annualised except Turnover, Lev and Shorting)."""
    r, tc = pf["r"], pf["tc"]
    sd = r.std(ddof=1)
    return pd.Series({
        "n": len(pf),
        "R": r.mean() * 12,
        "Vol": sd * np.sqrt(12),
        "SR_gross": r.mean() / sd * np.sqrt(12),
        "TC": tc.mean() * 12,
        "R_TC": (r - tc).mean() * 12,
        "SR_net": (r - tc).mean() / sd * np.sqrt(12),
        "Utility": (r.mean() - 0.5 * r.var(ddof=1) * gamma - tc.mean()) * 12,
        "Turnover": pf["turnover"].mean(),
        "Lev": pf["inv"].mean(),
        "Shorting": pf["shorting"].mean(),
    })


def table2(pfs: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Table 2 rows, one per ``type`` (in order of first appearance)."""
    rows = {t: summary_stats(g, gamma) for t, g in pfs.groupby("type", sort=False)}
    out = pd.DataFrame(rows).T[TABLE2_COLUMNS]
    out["n"] = out["n"].astype(int)
    return out.rename_axis("type")
