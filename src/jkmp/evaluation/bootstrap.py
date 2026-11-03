"""Paired circular block bootstrap (port of the team's Kati_tryout ``analysis/ml_comparison/inference.py``):
the acceptance test of the Kati_tryout iteration chain (research log trials 6–7) and the standard error of an
executed validation score (trial 7).
"""

from __future__ import annotations

import numpy as np


def circular_block_indices(n: int, replicates: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Circular blocks of ``block_length`` with uniform random starts, concatenated and truncated to n."""
    if n < 2 or replicates < 1 or not 1 <= block_length <= n:
        raise ValueError("require n >= 2, replicates >= 1 and 1 <= block_length <= n")
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(replicates, blocks))
    return ((starts[..., None] + np.arange(block_length)) % n).reshape(replicates, -1)[:, :n]


def net_sharpe(r: np.ndarray, tc: np.ndarray, axis: int = -1) -> np.ndarray:
    """Paper-convention net Sharpe: √12 · mean(r − tc) / sd(r) (gross-return volatility, ddof = 1)."""
    return np.sqrt(12) * (r - tc).mean(axis=axis) / r.std(axis=axis, ddof=1)


def p_better_sharpe(r_a: np.ndarray, tc_a: np.ndarray, r_b: np.ndarray, tc_b: np.ndarray, reps: int, block: int,
                    seed: int) -> dict[str, float]:
    """Share of paired resamples in which strategy a's net Sharpe exceeds b's (ties at half weight), and the 95%
    percentile interval of the difference. Identical series give P = 0.5 by the tie rule."""
    if len(r_a) != len(r_b):
        raise ValueError("paired series must have the same length")
    idx = circular_block_indices(len(r_a), reps, block, np.random.default_rng(seed))
    d = net_sharpe(r_a[idx], tc_a[idx]) - net_sharpe(r_b[idx], tc_b[idx])
    tied = np.isclose(d, 0.0, rtol=0.0, atol=1e-12)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"p_better": float(np.mean((d > 1e-12) + 0.5 * tied)), "diff": float(net_sharpe(r_a, tc_a) - net_sharpe(r_b, tc_b)),
            "ci_low": float(lo), "ci_high": float(hi)}


def executed_score(net_active: np.ndarray, gross_active: np.ndarray, gamma: float, axis: int = -1) -> np.ndarray:
    """Annual executed active utility 12·[mean(net active) − (γ/2)·sample variance(gross active)]."""
    return 12 * (net_active.mean(axis=axis) - 0.5 * gamma * gross_active.var(axis=axis, ddof=1))


def score_standard_error(net_active: np.ndarray, gross_active: np.ndarray, gamma: float, reps: int, block: int,
                         seed: int) -> float:
    """Bootstrap standard error of ``executed_score``: circular block resamples of the paired monthly series, with
    the mean and the variance recomputed in each resample."""
    idx = circular_block_indices(len(net_active), reps, min(block, len(net_active)), np.random.default_rng(seed))
    return float(np.std(executed_score(net_active[idx], gross_active[idx], gamma), ddof=1))
