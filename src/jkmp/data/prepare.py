"""Prepare raw files for the Python replication's timing, screens and universe.

Reference: 1 - Prepare Data.R. The documented Python changes include next-month
RF for next-month total returns, exact wealth roll-back and annual rank buffers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from jkmp.config import Config
from jkmp.costs import drift_factor, price_impact_lambda, wealth_path
from jkmp.data.universe import build_universe
from jkmp.data.wrds_pull import RAW_FILES
from jkmp.features import load_theme_directions, rank_features, theme_scores

# Ordered, inclusive SIC ranges from reference/jkmp_r/1 - Prepare Data.R.
FF12_RANGES = [
    ("NoDur", [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]),
    ("Durbl", [(2500, 2519), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716),
               (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]),
    ("Manuf", [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569),
               (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
               (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)]),
    ("Enrgy", [(1200, 1399), (2900, 2999)]),
    ("Chems", [(2800, 2829), (2840, 2899)]),
    ("BusEq", [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]),
    ("Telcm", [(4800, 4899)]),
    ("Utils", [(4900, 4949)]),
    ("Shops", [(5000, 5999), (7200, 7299), (7600, 7699)]),
    ("Hlth", [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]),
    ("Money", [(6000, 6999)]),
]


def ff12_industry(sic: pd.Series) -> pd.Series:
    values = pd.to_numeric(sic, errors="coerce")
    result = pd.Series("Other", index=sic.index, dtype=object)
    assigned = pd.Series(False, index=sic.index)
    for name, ranges in FF12_RANGES:
        match = pd.Series(False, index=sic.index)
        for low, high in ranges:
            match |= values.between(low, high)
        match &= ~assigned
        result.loc[match] = name
        assigned |= match
    return result


def risk_free_monthly(ff: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(ff["date"]) + pd.offsets.MonthEnd(0)
    if dates.duplicated().any():
        raise ValueError("monthly factors have duplicate months")
    return pd.Series(pd.to_numeric(ff["rf"]).to_numpy(dtype=float), index=dates, name="rf").sort_index()


def market_total_returns(ff: pd.DataFrame, start=None) -> pd.DataFrame:
    out = pd.DataFrame({
        "eom_ret": pd.to_datetime(ff["date"]) + pd.offsets.MonthEnd(0),
        "tret": (pd.to_numeric(ff["mkt_rf"]) + pd.to_numeric(ff["rf"])).to_numpy(dtype=float),
    }).sort_values("eom_ret").reset_index(drop=True)
    if out["eom_ret"].duplicated().any():
        raise ValueError("monthly market returns have duplicate months")
    if start is not None:
        out = out.loc[out["eom_ret"] >= pd.Timestamp(start)].reset_index(drop=True)
    return out


def add_timing_columns(chars: pd.DataFrame, rf: pd.Series, wealth: pd.DataFrame) -> pd.DataFrame:
    out = chars.copy()
    next_eom = out["eom"] + pd.offsets.MonthEnd(1)
    out["ret_ld1"] = pd.to_numeric(out["ret_exc_lead1m"]).astype("float64")
    out["tr_ld1"] = out["ret_ld1"] + next_eom.map(rf)
    out["tr_ld0"] = pd.to_numeric(out["ret_exc"]).astype("float64") + out["eom"].map(rf)
    market = wealth.set_index("eom")["mu_ld1"]
    out["mu_ld1"] = out["eom"].map(market)
    previous_market = market.copy()
    previous_market.index += pd.offsets.MonthEnd(1)
    out["mu_ld0"] = out["eom"].map(previous_market)
    out["g"] = drift_factor(out["tr_ld0"], out["mu_ld0"])
    return out


def flag_history(panel: pd.DataFrame, n_lags: int) -> pd.Series:
    """True only with current + n_lags consecutive screened months; preserve row order."""
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")
    if panel.duplicated(["id", "eom"]).any():
        raise ValueError("history input has duplicate stock-months")
    rows = panel[["id", "eom"]].copy().reset_index(drop=True)
    rows["position"] = np.arange(len(rows))
    rows = rows.sort_values(["id", "eom"])
    ordinal = rows["eom"].dt.year * 12 + rows["eom"].dt.month
    lag = ordinal.groupby(rows["id"]).shift(n_lags)
    flags = ordinal.sub(lag).eq(n_lags).to_numpy()
    output = np.zeros(len(rows), dtype=bool)
    output[rows["position"].to_numpy()] = flags
    return pd.Series(output, index=panel.index, name="valid_data")


@dataclass
class PreparedData:
    panel: pd.DataFrame
    wealth: pd.DataFrame
    membership: pd.DataFrame
    screen_log: pd.DataFrame
    universe_stats: pd.DataFrame


def build_panel(cfg: Config) -> PreparedData:
    raw = cfg.paths.raw
    needed = [raw / RAW_FILES[k] for k in ("jkp", "ff_monthly", "factor_details", "cluster_labels")]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Run scripts/00_download.py pull first. Missing: " + ", ".join(missing))
    panel = pd.read_parquet(raw / RAW_FILES["jkp"])
    required = ["id", "eom", "me", "sic", "size_grp", "ret_exc", "ret_exc_lead1m",
                "crsp_exchcd", "crsp_shrcd", cfg.trading_costs.dolvol_column,
                *cfg.universe.jkp_flags, *cfg.features.names]
    absent = sorted(set(required) - set(panel))
    if absent:
        raise ValueError(f"raw JKP file is missing required columns: {absent}")
    panel["eom"] = pd.to_datetime(panel["eom"]).dt.normalize()
    panel["id"] = pd.to_numeric(panel["id"], errors="raise").astype("int64")
    if panel.duplicated(["id", "eom"]).any():
        raise ValueError("raw JKP file has duplicate stock-months")
    numeric = set(required) - {"id", "eom", "size_grp"}
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").astype("float64")
    ff = pd.read_parquet(raw / RAW_FILES["ff_monthly"])
    market = market_total_returns(ff, start=cfg.dates.warmup_start)
    wealth = wealth_path(cfg.investor.wealth, cfg.investor.wealth_end_date, market)
    panel = add_timing_columns(panel, risk_free_monthly(ff), wealth)
    screens = []

    def keep(name, mask):
        nonlocal panel
        before = len(panel)
        panel = panel.loc[mask.fillna(False)].copy()
        screens.append({"screen": name, "before": before, "excluded": before - len(panel), "after": len(panel)})

    keep("dates", panel["eom"].between(pd.Timestamp(cfg.dates.warmup_start),
                                     pd.Timestamp(cfg.dates.test_end) - pd.offsets.MonthEnd(1)))
    keep("crsp_id", panel["id"].between(1, 99999))
    if "excntry" in panel:
        keep("country", panel["excntry"].eq("USA"))
    keep("exchange", panel["crsp_exchcd"].isin(cfg.universe.exchanges))
    keep("share_code", panel["crsp_shrcd"].isin(cfg.universe.share_codes))
    for flag in cfg.universe.jkp_flags:
        keep(flag, panel[flag].eq(1))
    keep("market_cap", panel["me"].notna() & np.isfinite(panel["me"]) & panel["me"].gt(0))
    keep("returns", panel[["ret_ld1", "tr_ld1", "tr_ld0"]].notna().all(axis=1)
         & np.isfinite(panel[["ret_ld1", "tr_ld1", "tr_ld0"]]).all(axis=1))
    keep("market_timing", panel[["mu_ld0", "mu_ld1"]].notna().all(axis=1))
    volume = cfg.trading_costs.dolvol_column
    keep("dollar_volume", panel[volume].notna() & np.isfinite(panel[volume]) & panel[volume].gt(0))
    keep("sic", panel["sic"].notna())
    panel[list(cfg.features.names)] = panel[list(cfg.features.names)].replace([np.inf, -np.inf], np.nan)
    keep("feature_coverage", panel[list(cfg.features.names)].notna().mean(axis=1).ge(cfg.universe.min_feature_pct))
    if panel.empty:
        raise ValueError("no stock-months survived the configured screens")
    panel = panel.sort_values(["id", "eom"]).reset_index(drop=True)
    # R checks one extra month for the previous period's lagged portfolio signals (R9).
    panel["valid_data"] = flag_history(panel, cfg.universe.lookback_months + 1)
    panel["dolvol"] = panel[volume] * (1e6 if cfg.trading_costs.dolvol_units == "millions" else 1.0)
    panel["lambda"] = price_impact_lambda(panel["dolvol"], cfg.trading_costs.price_impact)
    panel["ff12"] = ff12_industry(panel["sic"])
    ranks = rank_features(panel, cfg.features.names, cfg.features.percentile_rank, cfg.features.impute_value)
    directions = load_theme_directions(raw, cfg.features.themes)
    panel[list(cfg.features.names)] = ranks
    panel = pd.concat([panel, theme_scores(ranks, directions)], axis=1)
    panel, membership, stats = build_universe(
        panel, cfg.universe.target_n, cfg.universe.entry_rank, cfg.universe.exit_rank,
        month=cfg.universe.reconstitution_month,
    )
    return PreparedData(panel, wealth, membership, pd.DataFrame(screens), stats)
