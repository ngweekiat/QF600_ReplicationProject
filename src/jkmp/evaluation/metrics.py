"""Table 2 statistics, Table 3 (probability of outperformance) and Table 4 (correlations). Ports of ``pf_summary``,
``utility_t``, ``prob_outperformance`` and the return correlations in reference/jkmp_r/6 - Base analysis.R.

Input: monthly performance rows (``jkmp.portfolio.accounting.portfolio_returns``) with columns
``type, r, tc, turnover, inv, shorting``. As in R, volatility, the net Sharpe ratio and utility all use
the variance of the **gross** return r (not r − tc); sample moments use n − 1 like R's ``sd``/``var``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from jkmp.config import DatesConfig
from jkmp.evaluation.holdout import check_return_months

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


def table2(pfs: pd.DataFrame, gamma: float, dates: DatesConfig) -> pd.DataFrame:
    """Table 2 rows, one per ``type`` (in order of first appearance); refuses locked holdout months."""
    check_return_months(pfs, dates, "table2")
    rows = {t: summary_stats(g, gamma) for t, g in pfs.groupby("type", sort=False)}
    out = pd.DataFrame(rows).T[TABLE2_COLUMNS]
    out["n"] = out["n"].astype(int)
    return out.rename_axis("type")


def _wide(pfs: pd.DataFrame, value: pd.Series, types: Sequence[str]) -> pd.DataFrame:
    """Return month × type; every type must cover the same months."""
    wide = pfs.assign(value=value).pivot(index="eom_ret", columns="type", values="value")[list(types)]
    if wide.isna().any().any():
        raise ValueError("methods cover different return months")
    return wide


def prob_outperformance(pfs: pd.DataFrame, gamma: float, dates: DatesConfig, types: Sequence[str]) -> pd.DataFrame:
    """Table 3: posterior probability that the row method's mean utility flow beats the column method's.
    R ``prob_outperformance``: d_t = u_row,t − u_col,t, P = 1 − Φ(0; mean d, sd(d)/√T), with u_t = ``utility_flow``."""
    check_return_months(pfs, dates, "prob_outperformance")
    u = _wide(pfs, utility_flow(pfs, gamma), types)
    out = pd.DataFrame(np.nan, index=list(types), columns=list(types))
    for a in types:
        for b in types:
            if a != b:
                d = u[a] - u[b]
                out.loc[a, b] = 1.0 - norm.cdf(0.0, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
    return out.rename_axis("type")


def return_correlations(pfs: pd.DataFrame, dates: DatesConfig, types: Sequence[str]) -> pd.DataFrame:
    """Table 4: Pearson correlation of the monthly gross returns r (R: ``cor(method = "pearson")``)."""
    check_return_months(pfs, dates, "return_correlations")
    return _wide(pfs, pfs["r"], types).corr(method="pearson").rename_axis("type")


def subperiod_table(pfs: pd.DataFrame, gamma: float, dates: DatesConfig,
                    periods: Mapping[str, tuple[str, str]], columns: Sequence[str]) -> pd.DataFrame:
    """Table 2 ``columns`` within each (label: (first, last) return month) period; type × (column, period)."""
    parts = {label: table2(pfs[pfs["eom_ret"].between(pd.Timestamp(a), pd.Timestamp(b))], gamma, dates)[list(columns)]
             for label, (a, b) in periods.items()}
    out = pd.concat(parts, axis=1).swaplevel(axis=1)
    return out[[(c, label) for c in columns for label in periods]]
