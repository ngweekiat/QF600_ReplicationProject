"""Closed-form ridge with an unpenalised intercept, for a whole λ grid from one eigendecomposition.

Replaces R's ``glmnet(alpha = 0, standardize = FALSE)``, which minimises

    (1/2n) Σ (y − β₀ − xβ)² + (λ/2) ‖β‖²   ⇒   β(λ) = (C + λI)⁻¹ c,  β₀ = ȳ − x̄β,

with C = X̃ᵀX̃/n and c = X̃ᵀỹ/n on centred data. With C = V diag(d) Vᵀ every λ costs one matrix
product: β(λ) = V diag(1/(d + λ)) Vᵀ c.

All quantities come from additive sufficient statistics (``Moments``), so a training window is a sum
of monthly moments and a validation window is a difference of two running sums; validation MSE for
every λ is computed from the validation moments without materialising the validation design matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass
class Moments:
    """Sums over rows of a design X (n × P) and target y."""

    n: int
    sx: np.ndarray    # Σ x        (P,)
    sy: float         # Σ y
    sxx: np.ndarray   # Σ x xᵀ     (P, P)
    sxy: np.ndarray   # Σ x y      (P,)
    syy: float        # Σ y²

    @classmethod
    def zeros(cls, n_cols: int) -> "Moments":
        return cls(0, np.zeros(n_cols), 0.0, np.zeros((n_cols, n_cols)), np.zeros(n_cols), 0.0)

    @classmethod
    def of(cls, x: np.ndarray, y: np.ndarray) -> "Moments":
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            m = cls(len(y), x.sum(axis=0), float(y.sum()), x.T @ x, x.T @ y, float(y @ y))
        if not (np.isfinite(m.sxx).all() and np.isfinite(m.sxy).all()):
            raise FloatingPointError("non-finite moments")
        return m

    def __add__(self, o: "Moments") -> "Moments":
        return Moments(self.n + o.n, self.sx + o.sx, self.sy + o.sy, self.sxx + o.sxx, self.sxy + o.sxy, self.syy + o.syy)

    def __sub__(self, o: "Moments") -> "Moments":
        return Moments(self.n - o.n, self.sx - o.sx, self.sy - o.sy, self.sxx - o.sxx, self.sxy - o.sxy, self.syy - o.syy)

    def select(self, cols: np.ndarray, scale: float) -> "Moments":
        """Moments of the design ``scale · X[:, cols]``."""
        return Moments(self.n, self.sx[cols] * scale, self.sy, self.sxx[np.ix_(cols, cols)] * scale**2,
                       self.sxy[cols] * scale, self.syy)


def ridge_path(train: Moments, lambdas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients (P × L) and intercepts (L,) for every λ, from one eigendecomposition.

    λ = 0 gives OLS; directions with a numerically zero eigenvalue (and λ = 0) get zero weight, i.e.
    the minimum-norm least-squares solution.
    """
    n = train.n
    xbar, ybar = train.sx / n, train.sy / n
    C = train.sxx / n - np.outer(xbar, xbar)
    c = train.sxy / n - xbar * ybar
    d, V = eigh(0.5 * (C + C.T))
    d = np.clip(d, 0.0, None)
    lam = np.asarray(lambdas, dtype=float)
    denom = d[:, None] + lam[None, :]
    tiny = denom <= 1e-12 * max(d.max(), 1e-300)
    inv = np.where(tiny, 0.0, 1.0 / np.where(tiny, 1.0, denom))
    with np.errstate(all="ignore"):
        coef = V @ (inv * (V.T @ c)[:, None])
        intercept = ybar - xbar @ coef
    if not (np.isfinite(coef).all() and np.isfinite(intercept).all()):
        raise FloatingPointError("non-finite ridge coefficients")
    return coef, intercept


def mse_from_moments(m: Moments, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """Mean squared error of y − β₀ − xβ over the rows summarised by ``m``, for each column of ``coef``."""
    with np.errstate(all="ignore"):
        quad = np.einsum("pl,pl->l", coef, m.sxx @ coef)
        sse = (m.syy - 2 * intercept * m.sy - 2 * (m.sxy @ coef) + m.n * intercept**2
               + 2 * intercept * (m.sx @ coef) + quad)
    return sse / m.n
