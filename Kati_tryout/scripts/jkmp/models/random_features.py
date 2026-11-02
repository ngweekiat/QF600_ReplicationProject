"""Random Fourier features (Eqs. 38–39). Port of ``rff`` in reference/jkmp_r/0 - Return prediction functions.R.

    W ~ N(0, g·I)  (k × p/2; g is a VARIANCE, as in R's ``mvrnorm(Sigma = g*diag(k))``)
    RF(s) = p^{-1/2} [cos(sW), sin(sW)]

R calls ``set.seed(seed)`` before each g, so every g shares the same standard-normal draws scaled by
√g; a model with p features uses the first p/2 cos and the first p/2 sin columns of the p_max draw.
NumPy's generator differs from R's, so the draws themselves are not R's (docs/deviations.md).
"""

from __future__ import annotations

import numpy as np


def base_draws(n_inputs: int, p_max: int, seed: int) -> np.ndarray:
    """Standard-normal matrix Z (n_inputs × p_max/2), shared by every g."""
    if p_max % 2:
        raise ValueError("p_max must be even")
    return np.random.default_rng(seed).standard_normal((n_inputs, p_max // 2))


def weights_for(z: np.ndarray, variance: float) -> np.ndarray:
    """W = √g · Z, i.e. W ~ N(0, g·I)."""
    return np.sqrt(variance) * z


def rff_raw(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Unscaled features [cos(xW), sin(xW)] for every column of W (n × 2·W.shape[1])."""
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
        xw = x @ w
    if not np.isfinite(xw).all():
        raise FloatingPointError("non-finite random-feature projections")
    return np.hstack([np.cos(xw), np.sin(xw)])


def column_subset(p: int, p_max: int) -> np.ndarray:
    """Columns of ``rff_raw`` (p_max features) that make up the first p features: cos 1..p/2, sin 1..p/2."""
    if p % 2 or p > p_max:
        raise ValueError(f"p must be even and <= {p_max}")
    half, half_max = p // 2, p_max // 2
    return np.concatenate([np.arange(half), half_max + np.arange(half)])


def rff(x: np.ndarray, w: np.ndarray, p: int) -> np.ndarray:
    """RF(x) with p features: p^{-1/2} [cos(xW[:, :p/2]), sin(xW[:, :p/2])] (R scales by ``p^(-0.5)``)."""
    return rff_raw(x, w[:, : p // 2]) / np.sqrt(p)
