"""Active theme exposures over time for benchmark-relative Portfolio-ML (extension, D34)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.features import THEME_PREFIX  # noqa: E402


def plot_active_exposures(exposures: pd.DataFrame, path: Path, window: int = 12) -> list[Path]:
    """One panel per theme: the active exposure Xᵀa (theme loadings are standardised scores) of each method,
    as a ``window``-month rolling mean. ``exposures``: eom, type, theme_* columns (``ex_ante_exposures``)."""
    themes = [c for c in exposures.columns if c.startswith(THEME_PREFIX)]
    fig, axes = plt.subplots(2, (len(themes) + 1) // 2, figsize=(12, 5.6), sharex=True)
    for ax, theme in zip(axes.flat, themes):
        for t, g in exposures.groupby("type", sort=False):
            s = g.set_index("eom")[theme].sort_index().rolling(window).mean()
            ax.plot(s.index, s, lw=1.0, label=t)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(theme.removeprefix(THEME_PREFIX).replace("_", " "), fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(themes):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=6)
    fig.supylabel(f"Active theme exposure Xᵀa ({window}-month rolling mean)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    fig.savefig(out[0], dpi=200, bbox_inches="tight")
    fig.savefig(out[1], bbox_inches="tight")
    plt.close(fig)
    return out
