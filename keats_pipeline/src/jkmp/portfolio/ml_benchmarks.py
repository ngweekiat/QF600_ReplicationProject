"""Benchmarks built on the one-month return predictions ``pred_ld1``: Markowitz-ML and Factor-ML.

Ports ``tpf_implement`` and ``factor_ml_implement`` (0 - Portfolio choice functions.R). Both trade to
their targets in full each month; costs come from ``jkmp.portfolio.accounting``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from jkmp.covariance import CovParts
from jkmp.portfolio.accounting import add_start_weights, portfolio_returns


def markowitz_targets(universe: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts], gamma: float) -> pd.DataFrame:
    """Eq. 32: w = (1/γ) Σ⁻¹ μ with μ = ``pred_ld1``, via a Cholesky solve (R: ``solve(sigma) %*% pred_ld1``)."""
    out = []
    for eom, g in universe.groupby("eom", sort=True):
        ids = g["id"].to_numpy()
        w = cho_solve(cho_factor(cov_parts[eom].full(ids)), g["pred_ld1"].to_numpy()) / gamma
        out.append(pd.DataFrame({"id": ids, "eom": eom, "w": w}))
    return pd.concat(out, ignore_index=True)


def factor_targets(universe: pd.DataFrame, n_portfolios: int) -> pd.DataFrame:
    """Long the top 1/n of predicted returns, short the bottom 1/n, value-weighted within each leg.

    R: ``er_prank = ecdf(er)(er)``; long if ``er_prank >= 1 - 1/n``, short if ``<= 1/n``; each leg
    sums to +1 / −1. We use quintiles (n = 5) since the universe has ~80 stocks (R: deciles). As in R,
    the ``>=`` cut can give the long leg one more stock than the short leg (with 80 stocks: 17 vs 16).
    """
    prank = universe.groupby("eom")["pred_ld1"].rank(method="max", pct=True)
    pos = np.select([prank >= 1 - 1 / n_portfolios, prank <= 1 / n_portfolios], [1.0, -1.0], default=0.0)
    d = universe[["id", "eom", "me"]].assign(pos=pos)
    leg_me = d.groupby(["eom", "pos"])["me"].transform("sum")
    return d[["id", "eom"]].assign(w=(d["me"] / leg_me * d["pos"]).to_numpy())


def run_ml_benchmarks(panel: pd.DataFrame, wealth: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts],
                      dates: Sequence[pd.Timestamp], gamma: float, n_portfolios: int,
                      charge_exit_tc: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weights and monthly performance for Markowitz-ML and Factor-ML; ``panel`` must carry ``pred_ld1``."""
    dates = pd.DatetimeIndex(sorted(dates))
    universe = panel.loc[panel["in_universe"] & panel["eom"].isin(dates)]
    if universe["pred_ld1"].isna().any():
        bad = universe[universe["pred_ld1"].isna()]
        raise ValueError(f"{len(bad)} universe stock-months lack pred_ld1, e.g. {bad.iloc[0][['id', 'eom']].tolist()}")
    targets = {"Markowitz-ML": markowitz_targets(universe, cov_parts, gamma),
               "Factor-ML": factor_targets(universe, n_portfolios)}
    weights, pfs = [], []
    for name, tg in targets.items():
        w = add_start_weights(tg, panel, wealth, dates, charge_exit_tc)
        weights.append(w.assign(type=name))
        pfs.append(portfolio_returns(w, panel, wealth).assign(type=name))
    return pd.concat(weights, ignore_index=True), pd.concat(pfs, ignore_index=True)
