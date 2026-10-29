"""Chosen hyperparameters over time (the paper's Internet Appendix Fig. D.1) and how often picks sit on a grid edge.

Portfolio-ML: g (RFF variance), p (number of random features) and λ (ridge), picked each December
(``outputs/portfolio_ml/{rff,raw}/chosen.csv``). Static-ML*: k (multiplier on Λ), u (shrinkage of predicted
returns) and v = 1 + g (variance add-on, R ``cov_add``), from ``outputs/static_ml/chosen.csv``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def edge_share(picks: Mapping[str, tuple[pd.Series, Sequence[float]]]) -> pd.DataFrame:
    """For each hyperparameter (name: (picks, grid)): number of picks and the share at the grid's lowest and
    highest value. A grid with one value has no edge to report."""
    rows = []
    for name, (values, grid) in picks.items():
        v, lo, hi = np.asarray(values, dtype=float), float(min(grid)), float(max(grid))
        rows.append({"hyperparameter": name, "n": len(v), "grid_min": lo, "grid_max": hi,
                     "share_at_min": float(np.isclose(v, lo).mean()) if lo < hi else np.nan,
                     "share_at_max": float(np.isclose(v, hi).mean()) if lo < hi else np.nan})
    return pd.DataFrame(rows)


def plot_hyperparameters(pfml: Mapping[str, pd.DataFrame], static: pd.DataFrame, path: Path) -> list[Path]:
    """Step plots by return year: Portfolio-ML g, p, log λ (one line per signal set in ``pfml``, label → chosen
    frame) and Static-ML* k, u, v = 1 + g."""
    log_l = lambda c: np.log(c["l"].where(c["l"] > 0))  # noqa: E731 — λ = 0 has no log; drawn as a gap
    panels = [("Portfolio-ML: g", lambda c: c["g"]), ("Portfolio-ML: p", lambda c: c["p"]),
              ("Portfolio-ML: log λ", log_l),
              ("Static-ML*: k", None), ("Static-ML*: u", None), ("Static-ML*: v = 1 + g", None)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    for ax, (title, get) in zip(axes.flat, panels):
        if get is not None:
            for i, (label, c) in enumerate(pfml.items()):
                if label.endswith("(linear)") and title != "Portfolio-ML: log λ":
                    continue  # the linear version only tunes λ
                ax.step(c["year"], get(c), where="post", lw=1.2, color=f"C{0 if i == 0 else 4}", label=label)
            if title == "Portfolio-ML: p":
                ax.set_yscale("log", base=2)
        else:
            col = {"k": "k", "u": "u", "v": "g"}[title.split(": ")[1][0]]
            y = static[col] + (1.0 if col == "g" else 0.0)
            ax.step(static["year"], y, where="post", lw=1.2, color="C1", label="Static-ML*")
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 2].legend(fontsize=7)
    fig.supxlabel("Return year the choice applies to (picked the December before)", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    fig.savefig(out[0], dpi=200, bbox_inches="tight")
    fig.savefig(out[1], bbox_inches="tight")
    plt.close(fig)
    return out
