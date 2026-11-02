"""Market regimes over the test window (extension; not in the paper or the R code).

FRED series (public CSV, cached in data/raw/fred_{series}.parquet):
  USREC    NBER recession indicator, monthly (1 = recession)
  VIXCLS   CBOE VIX daily close → monthly mean
  FEDFUNDS effective federal funds rate, monthly average (%)
Labels per month-end: ``recession``, ``vix_regime`` (low / mid / high tercile of monthly VIX over the
evaluation window) and ``fed_regime`` (hiking / cutting / on hold from the change in FEDFUNDS over the
last ``fed_change_months`` months, with a ±``fed_threshold`` pp band).

Performance rows are labelled by the month the return is realised (``eom_ret``). The VIX re-costing
scales each month's trading cost by VIX_t / mean VIX at the formation month t, when the trade happens:
the weights are unchanged and tc = (w/2) ΔπᵀΛΔπ is linear in Λ, so Λ_t → s_t Λ gives tc_t → s_t tc_t.
"""

from __future__ import annotations

import io
import logging
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.config import DatesConfig  # noqa: E402
from jkmp.evaluation.holdout import check_return_months  # noqa: E402
from jkmp.evaluation.metrics import summary_stats, utility_flow  # noqa: E402

log = logging.getLogger(__name__)

MONTH = pd.offsets.MonthEnd(1)
REGIME_COLUMNS = ("vix_regime", "fed_regime", "recession")


# --- FRED -------------------------------------------------------------------------------------------

def fetch_fred(url_template: str, series: str, timeout: int = 60) -> pd.DataFrame:
    """One FRED series as ``date, value`` (missing observations dropped)."""
    raise FileNotFoundError(f"Missing local FRED cache for {series}; downloads are disabled.")
    url = url_template.format(series=series)
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = pd.read_csv(io.BytesIO(resp.read()))
    raw.columns = ["date", "value"]
    out = pd.DataFrame({"date": pd.to_datetime(raw["date"]), "value": pd.to_numeric(raw["value"], errors="coerce")})
    return out.dropna().reset_index(drop=True)


def load_fred(url_template: str, series: Sequence[str], raw_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    """Cached FRED downloads: ``raw_dir/fred_{series}.parquet``."""
    out = {}
    for s in series:
        path = raw_dir / f"fred_{s}.parquet"
        if force or not path.exists():
            fetch_fred(url_template, s).to_parquet(path, index=False)
        out[s] = pd.read_parquet(path)
    return out


def monthly_series(fred: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """``eom, usrec, vix, fedfunds`` by month-end (VIX: mean of daily closes in the month)."""
    def by_month(df: pd.DataFrame, name: str) -> pd.Series:
        return df.groupby(df["date"].dt.to_period("M"))["value"].mean().rename(name)
    parts = [by_month(fred["USREC"], "usrec"), by_month(fred["VIXCLS"], "vix"), by_month(fred["FEDFUNDS"], "fedfunds")]
    m = pd.concat(parts, axis=1)
    m.index = m.index.to_timestamp(how="end").normalize()
    return m.rename_axis("eom").reset_index()


# --- labels ------------------------------------------------------------------------------------------

def label_regimes(monthly: pd.DataFrame, months: Sequence[pd.Timestamp], fed_change_months: int,
                  fed_threshold: float) -> pd.DataFrame:
    """Regime labels for ``months``; VIX terciles are cut on those months only."""
    m = monthly.sort_values("eom").set_index("eom")
    d_ff = m["fedfunds"] - m["fedfunds"].shift(fed_change_months)
    fed = np.select([d_ff > fed_threshold, d_ff < -fed_threshold], ["hiking", "cutting"], default="on hold")
    m = m.assign(fed_change=d_ff, fed_regime=pd.Series(fed, index=m.index).where(d_ff.notna()))
    sub = m.reindex(pd.DatetimeIndex(months))
    if sub[["vix", "fedfunds", "usrec"]].isna().any().any():
        missing = sub.index[sub[["vix", "fedfunds", "usrec"]].isna().any(axis=1)]
        raise ValueError(f"FRED data missing for {len(missing)} months, e.g. {missing[0].date()}")
    sub["vix_regime"] = pd.qcut(sub["vix"], 3, labels=["low VIX", "mid VIX", "high VIX"]).astype(str)
    sub["recession"] = np.where(sub["usrec"] > 0.5, "recession", "expansion")
    return sub.rename_axis("eom").reset_index()


# --- analysis ----------------------------------------------------------------------------------------

def cumulative_utility(pfs: pd.DataFrame, gamma: float, dates: DatesConfig) -> pd.DataFrame:
    """Monthly utility flow r − tc − ½γ(r − r̄)² (Table 2's utility, month by month) and its running sum."""
    check_return_months(pfs, dates, "cumulative_utility")
    d = pfs.sort_values(["type", "eom_ret"]).reset_index(drop=True)
    d["u"] = utility_flow(d, gamma)
    d["cum_u"] = d.groupby("type", sort=False)["u"].cumsum()
    return d[["type", "eom_ret", "u", "cum_u"]]


def regime_table(pfs: pd.DataFrame, labels: pd.DataFrame, gamma: float, dates: DatesConfig) -> pd.DataFrame:
    """Table 2 statistics within each regime (rows labelled by the return month)."""
    check_return_months(pfs, dates, "regime_table")
    d = pfs.merge(labels[["eom", *REGIME_COLUMNS]].rename(columns={"eom": "eom_ret"}), on="eom_ret", how="left")
    if d[list(REGIME_COLUMNS)].isna().any().any():
        raise ValueError("some return months have no regime label")
    rows = []
    for dim in REGIME_COLUMNS:
        for (regime, t), g in d.groupby([dim, "type"], sort=False):
            rows.append(summary_stats(g, gamma).rename(None).to_frame().T.assign(dimension=dim, regime=regime, type=t))
    out = pd.concat(rows, ignore_index=True)
    first = ["dimension", "regime", "type"]
    return out[first + [c for c in out.columns if c not in first]]


def vix_cost_scale(monthly: pd.DataFrame, months: Sequence[pd.Timestamp], norm: str) -> pd.Series:
    """s_t = VIX_t / mean VIX, with the mean over ``months`` ('window') or all VIX history ('full')."""
    v = monthly.set_index("eom")["vix"]
    base = v.reindex(pd.DatetimeIndex(months)).mean() if norm == "window" else v.mean()
    return (v / base).rename("vix_scale")


def recost_vix(pfs: pd.DataFrame, scale: pd.Series, gamma: float, dates: DatesConfig) -> pd.DataFrame:
    """Each method's trades re-costed with Λ_t = s_t·Λ (s at the formation month): Table 2 rows for the
    original and re-costed costs side by side."""
    check_return_months(pfs, dates, "recost_vix")
    s = pfs["eom"].map(scale)
    if s.isna().any():
        raise ValueError("VIX scale missing for some formation months")
    re = pfs.assign(tc=pfs["tc"] * s)
    rows = []
    for t, g in pfs.groupby("type", sort=False):
        a, b = summary_stats(g, gamma), summary_stats(re[re["type"] == t], gamma)
        rows.append({"type": t, "TC": a["TC"], "TC_vix": b["TC"], "R_TC": a["R_TC"], "R_TC_vix": b["R_TC"],
                     "Utility": a["Utility"], "Utility_vix": b["Utility"], "dUtility": b["Utility"] - a["Utility"],
                     "TC_ratio": b["TC"] / a["TC"] if a["TC"] > 0 else np.nan,
                     "corr_turnover_vix": _corr(g["turnover"], g["eom"].map(scale))})
    return pd.DataFrame(rows)


def _corr(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


# --- figure ------------------------------------------------------------------------------------------

def plot_cumulative_utility(cum: pd.DataFrame, episodes, types: Sequence[str], gamma: float,
                            path: Path, dates: DatesConfig) -> list[Path]:
    check_return_months(cum, dates, "plot_cumulative_utility")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for ep in episodes:
        ax.axvspan(pd.Timestamp(ep.start), pd.Timestamp(ep.end) + pd.Timedelta(days=1), color="grey", alpha=0.18, lw=0)
        ax.text(pd.Timestamp(ep.start), 1.0, f" {ep.label}", transform=ax.get_xaxis_transform(), fontsize=6.5,
                va="top", ha="left", rotation=90, color="dimgrey")
    for t in types:
        g = cum[cum["type"] == t]
        if len(g):
            start = pd.DataFrame({"eom_ret": [g["eom_ret"].min() - MONTH], "cum_u": [0.0]})
            g = pd.concat([start, g[["eom_ret", "cum_u"]]])
            ax.plot(g["eom_ret"], g["cum_u"], lw=1.4 if "ML" in t else 1.0, label=t)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set(ylabel=f"Cumulative realised utility (γ = {gamma:g}, monthly flows)", title=f"Cumulative utility, {cum['eom_ret'].min():%Y}–{cum['eom_ret'].max():%Y}")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    fig.savefig(out[0], dpi=200, bbox_inches="tight")
    fig.savefig(out[1], bbox_inches="tight")
    plt.close(fig)
    return out
