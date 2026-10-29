"""Stage 1: the screened stock-month panel. Port of reference/jkmp_r/1 - Prepare Data.R.

Timing (one row per stock ``id`` and formation month-end ``eom`` = t):

* ``ret_ld1`` — excess return over month t+1 (what a portfolio formed at t earns)
* ``tr_ld1``  — total return over month t+1 (drifts weights from t to t+1)
* ``tr_ld0``  — total return over month t (the growth term g_t)
* ``mu_ld1`` / ``mu_ld0`` — market total return over month t+1 / month t (wealth growth)

Also on each row: ``lambda`` (Λ_i), ``g`` (weight drift over month t), ranked characteristics,
``theme_*`` scores, ``valid_data`` (passes every screen and has 12 months of history) and
``in_universe`` (member of the top-80 buffered universe, ``jkmp.data.universe``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jkmp import costs, features
from jkmp.config import Config
from jkmp.data import universe
from jkmp.data.wrds_pull import RAW_FILES

log = logging.getLogger(__name__)

MONTH = pd.offsets.MonthEnd(1)

# Fama-French 12 industries from SIC codes (R: the case_when in 1 - Prepare Data.R; first match wins).
FF12_RANGES: list[tuple[str, list[tuple[int, int]]]] = [
    ("NoDur", [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ("Durbl", [(2500, 2519), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751),
               (3792, 3792), (3900, 3939), (3990, 3999)]),
    ("Manuf", [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629),
               (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799),
               (3830, 3839), (3860, 3899)]),
    ("Enrgy", [(1200, 1399), (2900, 2999)]),
    ("Chems", [(2800, 2829), (2840, 2899)]),
    ("BusEq", [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ("Telcm", [(4800, 4899)]),
    ("Utils", [(4900, 4949)]),
    ("Shops", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Hlth", [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ("Money", [(6000, 6999)]),
]


# --------------------------------------------------------------------------- market inputs


def risk_free_monthly(ff_monthly: pd.DataFrame) -> pd.Series:
    """Ken French RF by month-end: the T-bill return over that month (decimal)."""
    return ff_monthly.set_index("date")["rf"].rename_axis("eom")


def market_total_returns(ff_monthly: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Market total return (Mkt-RF + RF) per return month in ``start .. end``: columns eom_ret, tret.

    R uses JKP's value-weighted US market (``market_returns.csv``); Ken French's market is the same
    concept — value-weighted CRSP common stocks (docs/deviations.md).
    """
    ff = ff_monthly[ff_monthly["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return pd.DataFrame({"eom_ret": ff["date"], "tret": ff["mkt_rf"] + ff["rf"]}).reset_index(drop=True)


# --------------------------------------------------------------------------- panel construction


def add_timing_columns(chars: pd.DataFrame, rf: pd.Series, wealth: pd.DataFrame) -> pd.DataFrame:
    """Add ret_ld1, tr_ld1, tr_ld0, mu_ld1, mu_ld0 to JKP rows (needs ``ret_exc``, ``ret_exc_lead1m``).

    JKP's ``ret_exc`` at month t is the CRSP total return over t minus RF of month t, so total
    returns add back the RF of the *return* month: t+1 for ``tr_ld1``, t for ``tr_ld0``.
    (R adds RF of month t to ``tr_ld1``; see docs/deviations.md.)
    """
    needed = pd.Index(chars["eom"]).union(pd.Index(chars["eom"] + MONTH))
    missing = needed.difference(rf.index)
    in_sample = missing[missing <= rf.index.max()]
    if len(in_sample):
        raise ValueError(f"risk-free rate missing for {len(in_sample)} months, e.g. {in_sample[0].date()}")
    out = chars.copy()
    mu = wealth.set_index("eom")["mu_ld1"]
    out["ret_ld1"] = out["ret_exc_lead1m"]
    out["tr_ld1"] = out["ret_ld1"] + (out["eom"] + MONTH).map(rf)
    out["tr_ld0"] = out["ret_exc"] + out["eom"].map(rf)
    out["mu_ld1"] = out["eom"].map(mu)
    out["mu_ld0"] = (out["eom"] - MONTH).map(mu)
    return out


def apply_screens(panel: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows failing each screen in turn (R order); return the panel and a per-screen log."""
    feats = list(cfg.features.names)
    u, d = cfg.universe, cfg.dates
    min_feat = int(np.floor(len(feats) * u.min_feature_pct))  # R: floor(length(features)*feat_pct)
    steps = [
        ("common stock (exchange, share code)",
         lambda p: p["crsp_exchcd"].isin(u.exchanges) & p["crsp_shrcd"].isin(u.share_codes)),
        ("sample dates", lambda p: p["eom"].between(pd.Timestamp(d.warmup_start), pd.Timestamp(d.data_end))),
        ("non-missing me", lambda p: p["me"].notna()),
        ("returns at t and t+1", lambda p: p["tr_ld0"].notna() & p["tr_ld1"].notna()),
        ("dolvol > 0", lambda p: p["dolvol"].gt(0)),  # NaN fails too
        ("SIC code", lambda p: p["sic"].notna()),
        (f"at least {min_feat} of {len(feats)} features", lambda p: p[feats].notna().sum(axis=1) >= min_feat),
    ]
    n_start, me_start = len(panel), panel["me"].sum()
    rows = []
    for name, keep in steps:
        mask = keep(panel)
        excluded = int((~mask).sum())
        rows.append({"screen": name, "rows_before": len(panel), "excluded": excluded,
                     "pct_excluded": 100.0 * excluded / max(len(panel), 1)})
        log.info("   %-40s excludes %6.2f%% of the observations", name, rows[-1]["pct_excluded"])
        panel = panel[mask]
    log.info("   Screened panel keeps %.2f%% of the observations and %.2f%% of the market cap",
             100.0 * len(panel) / n_start, 100.0 * panel["me"].sum() / me_start)
    return panel.reset_index(drop=True), pd.DataFrame(rows)


def flag_history(panel: pd.DataFrame, n_lags: int) -> pd.Series:
    """True where the row ``n_lags`` back for the same stock is exactly ``n_lags`` months earlier.

    R: ``eom_lag := shift(eom, n = lb)``, ``month_diff == lb``, with lb = lb_hor + 1 = 12, applied
    after the screens — so the stock must pass every screen in each of the last 13 months.
    """
    ordered = panel.sort_values(["id", "eom"])
    lag = ordered.groupby("id")["eom"].shift(n_lags)
    month_diff = (ordered["eom"].dt.year - lag.dt.year) * 12 + (ordered["eom"].dt.month - lag.dt.month)
    return month_diff.eq(n_lags).reindex(panel.index)


def ff12_industry(sic: pd.Series) -> pd.Series:
    out = pd.Series("Other", index=sic.index, dtype="object")
    assigned = pd.Series(False, index=sic.index)
    for name, ranges in FF12_RANGES:
        hit = pd.Series(False, index=sic.index)
        for lo, hi in ranges:
            hit |= sic.between(lo, hi)
        hit &= ~assigned
        out[hit] = name
        assigned |= hit
    return out


@dataclass
class PreparedData:
    panel: pd.DataFrame        # screened stock-months with timing, costs, features, themes, universe flags
    wealth: pd.DataFrame       # eom, wealth, mu_ld1
    screen_log: pd.DataFrame   # rows excluded by each screen
    membership: pd.DataFrame   # universe decisions per reconstitution
    universe_stats: pd.DataFrame


def build_panel(cfg: Config) -> PreparedData:
    """Read data/raw and build every Stage 1 output."""
    raw = cfg.paths.raw
    feats = list(cfg.features.names)
    chars = pd.read_parquet(raw / RAW_FILES["jkp"])
    ff = pd.read_parquet(raw / RAW_FILES["ff_monthly"])

    wealth = costs.wealth_path(
        cfg.investor.wealth, cfg.investor.wealth_end_date,
        market_total_returns(ff, start=pd.Timestamp(cfg.dates.warmup_start), end=pd.Timestamp(cfg.dates.data_end)),
    )
    dolvol_scale = 1e6 if cfg.trading_costs.dolvol_units == "millions" else 1.0
    chars["dolvol"] = chars[cfg.trading_costs.dolvol_column] * dolvol_scale

    panel = add_timing_columns(chars, risk_free_monthly(ff), wealth)
    panel, screen_log = apply_screens(panel, cfg)

    n_lags = cfg.universe.lookback_months + 1  # R: lb <- pf_set$lb_hor + 1
    panel["valid_data"] = flag_history(panel, n_lags)
    log.info("   %d-month history flag marks %.2f%% of rows invalid", n_lags, 100.0 * (~panel["valid_data"]).mean())

    panel[feats] = features.rank_features(panel, feats, cfg.features.percentile_rank, cfg.features.impute_value)
    directions = features.load_theme_directions(raw, cfg.features.themes)
    themes = features.theme_scores(panel[feats], directions)
    panel[themes.columns] = themes
    panel["ff12"] = ff12_industry(panel["sic"])
    panel["lambda"] = costs.price_impact_lambda(panel["dolvol"], cfg.trading_costs.price_impact)
    panel["g"] = costs.drift_factor(panel["tr_ld0"], panel["mu_ld0"])

    u = cfg.universe
    panel, membership, stats = universe.build_universe(
        panel, u.target_n, u.entry_rank, u.exit_rank, u.reconstitution_month)

    columns = ["id", "eom", "sic", "ff12", "size_grp", "crsp_exchcd", "crsp_shrcd", "me", "dolvol", "lambda",
               "ret_exc", "ret_ld1", "tr_ld0", "tr_ld1", "mu_ld0", "mu_ld1", "g",
               "valid_data", "recon_eom", "in_universe", *feats, *themes.columns]
    panel = panel[columns].sort_values(["eom", "id"], ignore_index=True)
    return PreparedData(panel, wealth, screen_log, membership, stats)
