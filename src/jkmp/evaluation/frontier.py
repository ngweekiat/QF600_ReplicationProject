"""Figure 1 (implementable efficient frontier) and the capacity chart. Port of the plotting parts of
``6 - Implementable efficient frontier.R``.

Fig. 1A: net excess return (R_TC) vs volatility at the baseline wealth, one line per method across γ,
starting at the origin (γ → ∞ holds nothing); Markowitz-ML and Factor-ML are their base weights scaled
to target volatilities (R ``vol_range``), and the Markowitz-ML gross Sharpe line is the no-cost reference.
Fig. 1B: the Portfolio-ML frontier at each wealth level. Capacity: utility vs wealth at one γ.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.evaluation.metrics import summary_stats  # noqa: E402
from jkmp.portfolio.accounting import portfolio_returns  # noqa: E402

COLOURS = {"Portfolio-ML": "C0", "Static-ML*": "C1", "Static-ML": "C1", "Markowitz-ML": "C2", "Factor-ML": "C3",
           "Portfolio-ML (linear)": "C4"}
MAIN = ["Portfolio-ML", "Portfolio-ML (linear)", "Static-ML*"]  # methods with a γ frontier


def scaled_frontier(weights: pd.DataFrame, panel: pd.DataFrame, wealth: pd.DataFrame,
                    vol_targets: Sequence[float], gamma: float) -> pd.DataFrame:
    """R: scale a method's weights *and* inherited weights by vol_target / its realised vol, rerun
    ``pf_ts_fun``; trading costs grow with the square of the scale. One row per (type, vol_target)."""
    rows = []
    for t, w in weights.groupby("type", sort=False):
        base = portfolio_returns(w, panel, wealth)
        base_vol = base["r"].std(ddof=1) * np.sqrt(12)
        for v in vol_targets:
            k = v / base_vol
            pf = portfolio_returns(w.assign(w=w["w"] * k, w_start=w["w_start"] * k), panel, wealth)
            rows.append(summary_stats(pf, gamma).rename(None).to_frame().T.assign(type=t, vol_target=v))
    return pd.concat(rows, ignore_index=True)


def _with_origin(points: pd.DataFrame) -> pd.DataFrame:
    """Sort a method's γ points by volatility and prepend the origin (γ → ∞)."""
    origin = pd.DataFrame({"Vol": [0.0], "R_TC": [0.0], "gamma": [np.inf]})
    return pd.concat([origin, points.sort_values("Vol")], ignore_index=True)


def _save(fig, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    fig.savefig(out[0], dpi=200, bbox_inches="tight")
    fig.savefig(out[1], bbox_inches="tight")
    plt.close(fig)
    return out


def _limits(points: pd.DataFrame) -> tuple[float, float, float]:
    xmax = max(0.1, 1.3 * points["Vol"].max())
    lo, hi = min(0.0, points["R_TC"].min()), max(0.0, points["R_TC"].max())
    pad = 0.3 * max(hi - lo, 0.01)
    return xmax, lo - pad, hi + pad


def plot_fig1a(summary: pd.DataFrame, scaled: pd.DataFrame, wealth: float, gamma_base: float,
               markowitz_sr_gross: float | None, path: Path) -> list[Path]:
    s = summary[summary["wealth"] == wealth]
    main = s[s["type"].isin(MAIN)]
    xmax, ylo, yhi = _limits(main)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.linspace(0, xmax, 200)
    if markowitz_sr_gross is not None:
        ax.plot(x, markowitz_sr_gross * x, color=COLOURS["Markowitz-ML"], lw=1, ls=":",
                label="Markowitz-ML (gross, no TC)")
    for t, g in scaled.groupby("type", sort=False):
        ax.plot(g["Vol"], g["R_TC"], color=COLOURS.get(t, "grey"), lw=1.2, label=f"{t} (scaled)")
    for t, offset in [("Portfolio-ML", (-6, 6)), ("Static-ML*", (4, -10)), ("Portfolio-ML (linear)", None)]:
        pts = _with_origin(main[main["type"] == t])
        if len(pts) == 1:  # method not run
            continue
        ax.plot(pts["Vol"], pts["R_TC"], "-o" if offset else "--o", color=COLOURS[t], ms=4, label=t)
        for _, r in (pts.iloc[1:].iterrows() if offset else []):  # γ labels on the two main lines only
            ax.annotate(f"{r['gamma']:g}", (r["Vol"], r["R_TC"]), textcoords="offset points", xytext=offset,
                        fontsize=7, color=COLOURS[t], ha="right" if offset[0] < 0 else "left")
        base = pts[pts["gamma"] == gamma_base]
        if len(base):  # indifference curve through the γ_base point: R_TC = U + γ/2·σ²
            u = base["R_TC"].iloc[0] - 0.5 * gamma_base * base["Vol"].iloc[0] ** 2
            ax.plot(x, u + 0.5 * gamma_base * x**2, color=COLOURS[t], lw=0.8, ls="--", alpha=0.5)
    one = s[(s["type"] == "Static-ML") & (s["gamma"] == gamma_base)]
    if len(one):
        ax.plot(one["Vol"], one["R_TC"], "s", color="black", ms=5, label=f"Static-ML (one layer, γ={gamma_base:g})")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set(xlim=(0, xmax), ylim=(ylo, yhi), xlabel="Volatility (annualised)",
           ylabel="Excess return net of trading cost (annualised)",
           title=f"Implementable efficient frontier, wealth ${wealth / 1e9:g}B (labels: γ)")
    ax.legend(fontsize=7, loc="best")
    return _save(fig, path)


def plot_fig1b(summary: pd.DataFrame, markowitz_sr_gross: float | None, path: Path) -> list[Path]:
    s = summary[summary["type"] == "Portfolio-ML"]
    xmax, ylo, yhi = _limits(s)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = np.linspace(0, xmax, 200)
    if markowitz_sr_gross is not None:
        ax.plot(x, markowitz_sr_gross * x, color="grey", lw=1, ls=":", label="Markowitz-ML (gross, no TC)")
    cmap = plt.get_cmap("viridis")
    levels = sorted(s["wealth"].unique())
    for i, level in enumerate(levels):
        pts = _with_origin(s[s["wealth"] == level])
        ax.plot(pts["Vol"], pts["R_TC"], "-o", ms=4, color=cmap(i / max(len(levels) - 1, 1)),
                label=f"${level / 1e9:g}B")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set(xlim=(0, xmax), ylim=(ylo, yhi), xlabel="Volatility (annualised)",
           ylabel="Excess return net of trading cost (annualised)",
           title="Portfolio-ML frontier by wealth (one point per γ)")
    ax.legend(title="Wealth by 2025", fontsize=7, title_fontsize=7)
    return _save(fig, path)


def plot_capacity(summary: pd.DataFrame, gamma: float, path: Path) -> list[Path]:
    s = summary[(summary["gamma"] == gamma) & summary["type"].isin(MAIN)]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for t in [m for m in MAIN if m in set(s["type"])]:
        g = s[s["type"] == t].sort_values("wealth")
        ax.plot(g["wealth"] / 1e9, g["Utility"], "--o" if "linear" in t else "-o", color=COLOURS[t], ms=4, label=t)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xscale("log")
    ax.set(xlabel="Wealth by 2025 ($B, log scale)", ylabel="Realised utility (annualised, net of TC)",
           title=f"Capacity: utility vs wealth, γ = {gamma:g}")
    ax.legend(fontsize=8)
    return _save(fig, path)
