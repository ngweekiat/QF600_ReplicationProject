"""Series behind the report figures (scripts/17_report_figures.py): cumulative return / utility paths (the paper's
Fig. 2), rolling information ratios and the standard error of an annualised information ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jkmp.evaluation.metrics import utility_flow


def cumulative_paths(pfs: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Per ``type`` and return month: cumulative sums of the gross return r, the net return r − tc and the utility
    flow r − tc − ½γ(r − r̄)² (JKMP Fig. 2: additive, not compounded)."""
    d = pfs.sort_values(["type", "eom_ret"]).reset_index(drop=True)
    d = d.assign(net=d["r"] - d["tc"], util=utility_flow(d, gamma))
    g = d.groupby("type", sort=False)
    return d.assign(cum_gross=g["r"].cumsum(), cum_net=g["net"].cumsum(),
                    cum_util=g["util"].cumsum())[["type", "eom_ret", "cum_gross", "cum_net", "cum_util"]]


def rolling_ir(active: pd.DataFrame, window: int) -> pd.Series:
    """Annualised rolling net information ratio: mean(active_net) / sd(active) over ``window`` months, × √12
    (the denominator is the gross active volatility, as in ``metrics.active_stats``)."""
    a = active.sort_values("eom_ret").set_index("eom_ret")
    m = a["active_net"].rolling(window).mean()
    s = a["active"].rolling(window).std(ddof=1)
    return (m / s * np.sqrt(12)).rename("rolling_ir")


def ir_standard_error(ir: float, n_months: int) -> float:
    """Approximate standard error of an annualised Sharpe / information ratio from n monthly observations
    (Lo, 2002, iid case): √((1 + IR²/2) / years)."""
    return float(np.sqrt((1.0 + 0.5 * ir**2) / (n_months / 12.0)))
