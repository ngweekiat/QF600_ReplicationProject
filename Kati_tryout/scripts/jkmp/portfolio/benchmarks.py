"""Benchmarks without a return model: Market, 1/N and Minimum variance.

Ports ``mkt_implement``, ``ew_implement`` and ``mv_implement`` (0 - Portfolio choice functions.R): the
target weights are traded to in full each month; costs come from ``jkmp.portfolio.accounting``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from jkmp.covariance import CovParts
from jkmp.portfolio.accounting import add_start_weights, portfolio_returns


def market_targets(universe: pd.DataFrame) -> pd.DataFrame:
    """Value weights me / Σ me over the universe each month."""
    w = universe["me"] / universe.groupby("eom")["me"].transform("sum")
    return universe[["id", "eom"]].assign(w=w.to_numpy())


def equal_weight_targets(universe: pd.DataFrame) -> pd.DataFrame:
    """1/N over the universe each month (DeMiguel et al., 2009)."""
    return universe[["id", "eom"]].assign(w=(1.0 / universe.groupby("eom")["id"].transform("size")).to_numpy())


def min_variance_targets(universe: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts]) -> pd.DataFrame:
    """Ex-ante minimum-variance weights Σ⁻¹1 / (1ᵀΣ⁻¹1), via a Cholesky solve (no explicit inverse)."""
    out = []
    for eom, g in universe.groupby("eom", sort=True):
        ids = g["id"].to_numpy()
        x = cho_solve(cho_factor(cov_parts[eom].full(ids)), np.ones(len(ids)))
        out.append(pd.DataFrame({"id": ids, "eom": eom, "w": x / x.sum()}))
    return pd.concat(out, ignore_index=True)


def run_benchmarks(panel: pd.DataFrame, wealth: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts],
                   dates: Sequence[pd.Timestamp], charge_exit_tc: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weights and monthly performance (``type`` column) for the three benchmarks over ``dates``."""
    dates = pd.DatetimeIndex(sorted(dates))
    missing = [d for d in dates if d not in cov_parts]
    if missing:
        raise ValueError(f"no covariance for {len(missing)} dates, e.g. {missing[0].date()}")
    universe = panel.loc[panel["in_universe"] & panel["eom"].isin(dates)]
    targets = {
        "Market": market_targets(universe),
        "1/N": equal_weight_targets(universe),
        "Minimum variance": min_variance_targets(universe, cov_parts),
    }
    weights, pfs = [], []
    for name, tg in targets.items():
        w = add_start_weights(tg, panel, wealth, dates, charge_exit_tc)
        weights.append(w.assign(type=name))
        pfs.append(portfolio_returns(w, panel, wealth).assign(type=name))
    return pd.concat(weights, ignore_index=True), pd.concat(pfs, ignore_index=True)
