"""Statistics of the Kati_tryout iteration chain on the replication (research log trials 6–7): the measures of its
results slides and the pretest diagnostics of iterations 4.2 and 4.3."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jkmp.evaluation.bootstrap import circular_block_indices, executed_score, net_sharpe


def chain_stats(pf: pd.DataFrame, market: pd.DataFrame) -> dict[str, float]:
    """Kati_tryout's results table for one strategy vs Market (both ``eom_ret, r, tc``, same months): annual net
    excess return, net Sharpe (gross-return volatility), worst additive drawdown, annual net active return, the
    information ratio on the net-active SD, and annual trading cost."""
    d = pf[["eom_ret", "r", "tc"]].merge(market[["eom_ret", "r", "tc"]], on="eom_ret", suffixes=("", "_m"),
                                         validate="one_to_one")
    if len(d) != len(pf) or len(d) != len(market):
        raise ValueError("strategy and Market cover different months")
    net = (d["r"] - d["tc"]).to_numpy()
    active = net - (d["r_m"] - d["tc_m"]).to_numpy()
    cum = np.cumsum(net)
    dd = cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return {"months": len(d), "net_return": 12 * net.mean(), "net_sharpe": float(net_sharpe(d["r"].to_numpy(), d["tc"].to_numpy())),
            "worst_additive_dd": float(dd.min()), "active_return": 12 * active.mean(),
            "ir_net_active_sd": float(np.sqrt(12) * active.mean() / active.std(ddof=1)), "costs": 12 * d["tc"].mean()}


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def best_stays_best(net: np.ndarray, gross: np.ndarray, gamma: float, reps: int, block: int, seed: int,
                    chunk: int = 100) -> float:
    """Share of circular block resamples (paired across candidates) in which the sample-best candidate keeps the
    highest executed score (ties allowed). ``net``/``gross``: months × candidates."""
    best = int(np.argmax(executed_score(net, gross, gamma, axis=0)))
    idx = circular_block_indices(len(net), reps, min(block, len(net)), np.random.default_rng(seed))
    hits = 0
    for i in range(0, reps, chunk):
        sub = idx[i:i + chunk]
        s = executed_score(net[sub], gross[sub], gamma, axis=1)           # (chunk × candidates)
        hits += int((s[:, best] >= s.max(axis=1) - 1e-15).sum())
    return hits / reps
