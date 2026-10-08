"""Trading-speed matrix m (Lemma 1). Port of ``m_func`` (reference/jkmp_r/0 - Portfolio choice functions.R).

With Λ diagonal, Lemma 1's fixed point reads

    m̃ = (x + y − m̃ ∘ G)⁻¹,      m = Λ^{-1/2} m̃ Λ^{1/2},

    G = 11ᵀ + Σ / (1 + r_f + μ)²         second moment of the weight growth g (ḡ = 1)
    x = w⁻¹ Λ^{-1/2} γΣ Λ^{-1/2}         risk relative to trading cost
    y = I + diag(diag G)

and ∘ is the elementwise product. The optimal portfolio trades a fraction I − m towards the aim each
month (Eq. 20): m → 0 (trade fully) as wealth → 0 and m → I (do not trade) as wealth → ∞.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve, sqrtm


def _parts(sigma: np.ndarray, lam: np.ndarray, wealth: float, gamma: float, rf: float, mu: float):
    lam = np.asarray(lam, dtype=float)
    if (lam <= 0).any():
        raise ValueError("lambda must be positive")
    n = len(lam)
    G = np.ones((n, n)) + sigma / (1.0 + rf + mu) ** 2
    s = 1.0 / np.sqrt(lam)  # diagonal of Λ^{-1/2}
    x = (gamma / wealth) * (s[:, None] * sigma * s[None, :])
    y = np.diag(1.0 + np.diag(G))
    return lam, G, x, y


def trading_speed(sigma: np.ndarray, lam: np.ndarray, wealth: float, gamma: float, rf: float, mu: float,
                  iterations: int = 10, return_tilde: bool = False) -> np.ndarray:
    """m (n × n) for monthly covariance ``sigma``, diagonal price impact ``lam`` (vector) and ``wealth``.

    ``rf`` and ``mu`` are the monthly risk-free rate and expected portfolio growth (``investor.mu``);
    callers pass the RF of the return month t+1 (docs/deviations.md D13; R passes RF of month t).

    Start value (R): m̃₀ = ½(σ̂ − √(σ̂² − 4I)) with σ̂ = x + 2I, the solution when G = 11ᵀ. It is
    computed in the algebraically identical, cancellation-free form 2·(σ̂ + √(σ̂² − 4I))⁻¹, with the
    square root from ``scipy.linalg.sqrtm``. Then ``iterations`` fixed-point steps, each solving
    (x + y − m̃∘G) m̃ = I with a symmetric solver (no explicit inverse).
    """
    lam, G, x, y = _parts(sigma, lam, wealth, gamma, rf, mu)
    n = len(lam)
    eye = np.eye(n)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
        sig_hat = x + 2.0 * eye
        root = np.real(sqrtm(sig_hat @ sig_hat - 4.0 * eye))
        m_tilde = solve(sig_hat + root, 2.0 * eye, assume_a="sym")
        for _ in range(iterations):
            m_tilde = solve(x + y - m_tilde * G, eye, assume_a="sym")
            m_tilde = 0.5 * (m_tilde + m_tilde.T)  # keep exact symmetry against round-off
    if not np.isfinite(m_tilde).all():
        raise FloatingPointError("non-finite trading-speed matrix")
    if return_tilde:
        return m_tilde
    return (1.0 / np.sqrt(lam))[:, None] * m_tilde * np.sqrt(lam)[None, :]


def lemma1_residual(m_tilde: np.ndarray, sigma: np.ndarray, lam: np.ndarray, wealth: float, gamma: float,
                    rf: float, mu: float) -> float:
    """max |(x + y − m̃∘G) m̃ − I|: zero at the Lemma 1 fixed point."""
    _, G, x, y = _parts(sigma, lam, wealth, gamma, rf, mu)
    with np.errstate(all="ignore"):
        r = (x + y - m_tilde * G) @ m_tilde - np.eye(len(m_tilde))
    return float(np.abs(r).max())
