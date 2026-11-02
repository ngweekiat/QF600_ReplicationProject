"""Monthly monitoring of the live active strategy (extension; scripts/11_monitor.py). Every check uses only data
up to the month it flags. One row per return month with the statistic and its flag:

* ``ir``: rolling ``ir_window``-month net information ratio < 0.
* ``drawdown``: drawdown of the cumulative net active return > ``drawdown_te_multiple`` × the TE budget.
* ``beta``: rolling ``ir_window``-month beta to the benchmark outside ``beta_limits``.
* ``theme``: largest |active theme exposure Xᵀa| at formation > ``theme_limit``.
* ``ic``: rolling ``ic_window``-month mean rank IC < 0 for ``ic_negative_months`` consecutive months.
* ``tc``: realised trading cost (stock-vol-scaled Λ) above the model's prediction for ``tc_months`` months in a row.
* ``half_life``: the December pick of the half-life moves more than one grid step from the previous year's.
* ``bias``: bias statistic sd(active_t / ex-ante TE_t) over ``bias_window`` months outside 1 ± √(2/T).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.config import DatesConfig, MonitorConfig  # noqa: E402
from jkmp.evaluation.holdout import check_return_months  # noqa: E402
from jkmp.features import THEME_PREFIX  # noqa: E402

MONTH = pd.offsets.MonthEnd(1)
CHECKS = ["ir", "drawdown", "beta", "theme", "ic", "tc", "half_life", "bias"]


def _run_length(flag: pd.Series) -> pd.Series:
    """Consecutive True count ending at each row."""
    grp = (~flag).cumsum()
    return flag.astype(int).groupby(grp).cumsum()


def half_life_steps(chosen: pd.DataFrame, grid: Sequence[float | None]) -> pd.Series:
    """Per return year: grid steps the chosen half-life moved from the previous year (None/inf = last step)."""
    order = [np.inf if h is None else float(h) for h in grid]
    pos = chosen.set_index("year")["h"].map(lambda h: order.index(np.inf if pd.isna(h) else float(h))).sort_index()
    return pos.diff().abs()


def monitor_table(active: pd.DataFrame, exposures: pd.DataFrame, rank_ic: pd.Series, tc_realised: pd.Series,
                  hl_steps: pd.Series | None, cfg: MonitorConfig, dates: DatesConfig) -> pd.DataFrame:
    """``active``: eom, eom_ret, r, r_b, tc, active, active_net (``metrics.active_returns``); ``exposures``: eom,
    te, theme_* (``metrics.ex_ante_exposures``); ``rank_ic``: monthly IC by return month; ``tc_realised``: re-costed
    tc by return month; ``hl_steps``: ``half_life_steps`` by return year (None if h is not tuned)."""
    check_return_months(active, dates, "monitor")
    d = active.sort_values("eom_ret").set_index("eom_ret")
    ex = exposures.assign(eom_ret=exposures["eom"] + MONTH).set_index("eom_ret").reindex(d.index)
    themes = [c for c in exposures.columns if c.startswith(THEME_PREFIX)]
    w, sq12 = cfg.ir_window, np.sqrt(12)
    out = pd.DataFrame(index=d.index)
    out["ir"] = d["active_net"].rolling(w).mean() / d["active"].rolling(w).std() * sq12
    cum = d["active_net"].cumsum()
    out["drawdown"] = cum.cummax().clip(lower=0) - cum
    out["beta"] = d["r"].rolling(w).cov(d["r_b"]) / d["r_b"].rolling(w).var()
    out["theme"] = ex[themes].abs().max(axis=1)
    out["ic"] = rank_ic.reindex(d.index).rolling(cfg.ic_window).mean()
    out["tc"] = tc_realised.reindex(d.index) - d["tc"]
    out["half_life"] = (d.index.year.map(hl_steps) if hl_steps is not None else np.nan)
    z = d["active"] / ex["te"]
    out["bias"] = z.rolling(cfg.bias_window).std()
    band = np.sqrt(2 / cfg.bias_window)
    flags = pd.DataFrame({
        "ir": out["ir"] < 0,
        "drawdown": out["drawdown"] > cfg.drawdown_te_multiple * cfg.te_budget,
        "beta": (out["beta"] < cfg.beta_limits[0]) | (out["beta"] > cfg.beta_limits[1]),
        "theme": out["theme"] > cfg.theme_limit,
        "ic": _run_length(out["ic"] < 0) >= cfg.ic_negative_months,
        "tc": _run_length(out["tc"] > 0) >= cfg.tc_months,
        "half_life": pd.Series(out["half_life"], index=out.index) > 1,
        "bias": (out["bias"] < 1 - band) | (out["bias"] > 1 + band),
    }, index=out.index)
    return out.join(flags.add_prefix("flag_")).reset_index()


def plot_dashboard(table: pd.DataFrame, cfg: MonitorConfig, title: str, path: Path) -> list[Path]:
    """One page: the eight statistics with their limits and flagged months shaded; plus an HTML version with the
    flag summary table. Returns the PNG and HTML paths."""
    band = np.sqrt(2 / cfg.bias_window)
    limits = {"ir": [0.0], "drawdown": [cfg.drawdown_te_multiple * cfg.te_budget], "beta": list(cfg.beta_limits),
              "theme": [cfg.theme_limit], "ic": [0.0], "tc": [0.0], "half_life": [1.0], "bias": [1 - band, 1 + band]}
    names = {"ir": f"Rolling {cfg.ir_window}m net IR", "drawdown": "Drawdown, cum. net active",
             "beta": f"Rolling {cfg.ir_window}m beta", "theme": "Max |theme exposure|",
             "ic": f"Rolling {cfg.ic_window}m rank IC", "tc": "Realised − predicted TC",
             "half_life": "Half-life jump (grid steps)", "bias": f"TE bias stat ({cfg.bias_window}m)"}
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    x = pd.to_datetime(table["eom_ret"])
    for ax, c in zip(axes.flat, CHECKS):
        ax.plot(x, table[c], lw=1.0, color="C0")
        for lim in limits[c]:
            ax.axhline(lim, color="C3", lw=0.7, ls="--")
        flagged = table[f"flag_{c}"].fillna(False).to_numpy()
        ax.fill_between(x, 0, 1, where=flagged, transform=ax.get_xaxis_transform(), color="C3", alpha=0.15, lw=0)
        ax.set_title(f"{names[c]} — flagged {int(flagged.sum())} of {len(flagged)} months", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    png, html = path.with_suffix(".png"), path.with_suffix(".html")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    summary = pd.DataFrame({"check": [names[c] for c in CHECKS],
                            "flagged months": [int(table[f"flag_{c}"].fillna(False).sum()) for c in CHECKS],
                            "flagged in last month": [bool(table[f"flag_{c}"].fillna(False).iloc[-1]) for c in CHECKS]})
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Strategy monitor</title>"
                    f"<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}}"
                    f"table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px}}"
                    f"img{{max-width:100%}}</style></head><body><h1>{title}</h1>{summary.to_html(index=False)}"
                    f"<p><img src='{png.name}' alt='monitor charts'></p></body></html>")
    return [png, html]
