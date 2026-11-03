"""Macro stress tests: Fed-policy scenarios for frozen strategies (extension; evaluation only, not a trial).

Pre-registered in ``docs/macro_stress.md``. Policy months come from the FOMC target (FRED ``DFEDTAR`` spliced
with ``DFEDTARU``): a month is a hike / cut / hold by the change in its month-end target. Scenarios:

  S1  first hike after cuts      hike in t and the last non-hold month before t was a cut      (2026-09)
  S2  hold right after a hike    hold in t and hike in t−1                                      (2026-10)
  S3  the next decision          S2 months split by outcome: any hike in t+1 .. t+k or not      (2026-12)

Two lenses. Historical: sums of saved monthly returns over each analogue window (return months).
Current book: the last book's factor exposures b = Xᵀa applied to the risk model's monthly factor returns in
each analogue window, plus idiosyncratic noise; S3 mixes the two outcome sets with the market-implied P(hike).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from jkmp.covariance import CovParts
from jkmp.evaluation.metrics import ex_ante_exposures

MONTH = pd.offsets.MonthEnd(1)


# --- policy months -----------------------------------------------------------------------------------

def policy_target(fred: Mapping[str, pd.DataFrame], series: Sequence[str]) -> pd.Series:
    """Daily target rate (%), splicing ``series`` in order: each later series takes over from its first date."""
    parts = []
    for i, s in enumerate(series):
        d = fred[s].set_index("date")["value"].sort_index()
        if i + 1 < len(series):
            d = d[d.index < fred[series[i + 1]]["date"].min()]
        parts.append(d)
    out = pd.concat(parts)
    if out.index.duplicated().any():
        raise ValueError("overlapping target series after splicing")
    return out.rename("target")


def policy_moves(target: pd.Series, threshold: float) -> pd.DataFrame:
    """``eom, target, change, move`` per calendar month: month-end target, its change on the previous month-end,
    and hike / cut / hold (|change| > ``threshold``). The first month (no previous month-end) is dropped."""
    m = target.groupby(target.index.to_period("M")).last()
    m.index = m.index.to_timestamp(how="end").normalize()
    change = m.diff()
    move = np.select([change > threshold, change < -threshold], ["hike", "cut"], default="hold")
    out = pd.DataFrame({"eom": m.index, "target": m.to_numpy(), "change": change.to_numpy(), "move": move})
    return out.iloc[1:].reset_index(drop=True)


def classify_events(moves: pd.DataFrame, outcome_months: int) -> pd.DataFrame:
    """Adds ``prev_move`` (last non-hold month strictly before), the S1 / S2 flags and, for S2 months,
    ``s3_outcome``: "hike" if any hike in the next ``outcome_months`` months, "no hike" if none, missing if
    those months are not yet observed."""
    d = moves.sort_values("eom").reset_index(drop=True)
    if not (d["eom"].iloc[1:].to_numpy() == (d["eom"].iloc[:-1] + MONTH).to_numpy()).all():
        raise ValueError("policy months must be consecutive")
    nonhold = d["move"].where(d["move"] != "hold")
    d["prev_move"] = nonhold.ffill().shift()
    d["s1"] = (d["move"] == "hike") & (d["prev_move"] == "cut")
    d["s2"] = (d["move"] == "hold") & (d["move"].shift() == "hike")
    ahead = pd.concat([d["move"].shift(-j) for j in range(1, outcome_months + 1)], axis=1)
    outcome = np.where(ahead.eq("hike").any(axis=1), "hike", "no hike")
    known = ahead.notna().all(axis=1)
    d["s3_outcome"] = pd.Series(outcome, index=d.index).where(d["s2"] & known)
    return d


def scenario_starts(events: pd.DataFrame, offset_months: int) -> dict[str, pd.DatetimeIndex]:
    """First return month of each analogue window: S1, S2 at the event month; S3 sets ``offset_months`` later."""
    e = events
    s3 = e.loc[e["s3_outcome"].notna(), ["eom", "s3_outcome"]]
    shift = pd.offsets.MonthEnd(offset_months)
    return {"S1 first hike after cuts": pd.DatetimeIndex(e.loc[e["s1"], "eom"]),
            "S2 hold after a hike": pd.DatetimeIndex(e.loc[e["s2"], "eom"]),
            "S3 next decision: hike": pd.DatetimeIndex(s3.loc[s3["s3_outcome"] == "hike", "eom"]) + shift,
            "S3 next decision: no hike": pd.DatetimeIndex(s3.loc[s3["s3_outcome"] == "no hike", "eom"]) + shift}


# --- historical lens ---------------------------------------------------------------------------------

def window_sums(monthly: pd.DataFrame, starts: Sequence[pd.Timestamp], length: int,
                value_cols: Sequence[str]) -> pd.DataFrame:
    """Sum of ``value_cols`` over return months start .. start + length − 1, per ``type`` and start.
    ``monthly`` has ``type, eom_ret`` and the value columns; incomplete windows are skipped."""
    rows = []
    for t, g in monthly.groupby("type", sort=False):
        s = g.set_index("eom_ret")[list(value_cols)].sort_index()
        for start in starts:
            idx = pd.date_range(start, periods=length, freq="ME")
            if idx.isin(s.index).all():
                rows.append({"type": t, "start": start, "h": length, **s.loc[idx].sum().to_dict()})
    return pd.DataFrame(rows, columns=["type", "start", "h", *value_cols])


def summarise_windows(windows: pd.DataFrame, value: str) -> pd.DataFrame:
    """Per ``type`` and ``h``: number of windows, mean, median, min, max and share below zero of ``value``."""
    g = windows.groupby(["type", "h"], sort=False)[value]
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(), "min": g.min(), "max": g.max(),
                         "share_negative": g.apply(lambda x: float((x < 0).mean()))}).reset_index()


def tc_ratio(windows: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Mean monthly trading cost inside the windows relative to each method's mean over all saved months."""
    overall = monthly.groupby("type")["tc"].mean()
    w = windows.assign(tc_month=windows["tc"] / windows["h"])
    inside = w.groupby(["type", "h"])["tc_month"].mean()
    out = inside.to_frame("tc_window").reset_index()
    out["tc_all"] = out["type"].map(overall)
    out["tc_ratio"] = out["tc_window"] / out["tc_all"]
    return out


# --- current-book lens -------------------------------------------------------------------------------

def book_exposure(active: pd.DataFrame, parts: CovParts) -> tuple[pd.Series, float]:
    """Factor exposures b = Xᵀa of one month's active book (``id, eom, a``) and its monthly idiosyncratic
    variance Σ ivar·a² (= ex-ante active variance − bᵀFb)."""
    ex = ex_ante_exposures(active, {parts.eom: parts}).iloc[0]
    b = ex[list(parts.factors)].astype(float)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
        idio = float(ex["te"] ** 2 - b.to_numpy() @ parts.F @ b.to_numpy())
    if not np.isfinite(idio) or idio < -1e-12:
        raise FloatingPointError("invalid idiosyncratic active variance")
    return b, max(idio, 0.0)


def monthly_factor_returns(daily: pd.DataFrame, factors: Sequence[str]) -> pd.DataFrame:
    """Daily factor returns summed to calendar months (index: month-end), the horizon of a held book."""
    d = daily[list(factors)]
    m = d.groupby(d.index.to_period("M")).sum()
    m.index = m.index.to_timestamp(how="end").normalize()
    return m


def book_window_returns(b: pd.Series, factor_monthly: pd.DataFrame, starts: Sequence[pd.Timestamp],
                        length: int) -> pd.Series:
    """Gross factor P&L of a fixed book over each complete window: Σ_m bᵀf_m, indexed by start month."""
    f = factor_monthly[b.index]
    out = {}
    for start in starts:
        idx = pd.date_range(start, periods=length, freq="ME")
        if idx.isin(f.index).all():
            out[start] = float(f.loc[idx].to_numpy().sum(axis=0) @ b.to_numpy())
    return pd.Series(out, dtype=float, name="gross")


def simulate(window_returns: pd.Series, idio_var: float, length: int, cost: float, n: int,
             rng: np.random.Generator) -> np.ndarray:
    """Net active return draws: a window drawn with replacement + N(0, length·idio_var) − ``cost``."""
    if window_returns.empty:
        raise ValueError("no analogue windows to draw from")
    pick = rng.integers(len(window_returns), size=n)
    return window_returns.to_numpy()[pick] + rng.normal(0.0, np.sqrt(length * idio_var), n) - cost


def mixture(draws_a: np.ndarray, draws_b: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Each draw from ``draws_a`` with probability ``p``, else the same position of ``draws_b``."""
    if len(draws_a) != len(draws_b):
        raise ValueError("mixture components need the same number of draws")
    return np.where(rng.random(len(draws_a)) < p, draws_a, draws_b)


def distribution_stats(x: np.ndarray) -> dict[str, float]:
    """Mean, sd, 5th / 50th / 95th percentiles, P(< 0) and the expected shortfall at 5%."""
    q05, q50, q95 = np.quantile(x, [0.05, 0.5, 0.95])
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "p05": float(q05), "p50": float(q50),
            "p95": float(q95), "p_negative": float((x < 0).mean()), "es05": float(x[x <= q05].mean())}
