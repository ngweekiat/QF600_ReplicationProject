"""Offline adapters for the saved Keats monthly panels.

The original portfolio mathematics lives in this folder's private jkmp snapshot.
These adapters change data loading, membership, timing and output paths only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from jkmp import costs, features
from jkmp.config import load_config
from jkmp.data.prepare import add_timing_columns, ff12_industry, market_total_returns, risk_free_monthly
from jkmp.data.wrds_pull import RAW_FILES

MONTH = pd.offsets.MonthEnd(1)
UNIVERSES = ("us_large", "us_mid", "eu_large")
VARIANTS = ("base", "active_pfml", "universe_250", "halflife")


def settings():
    return json.loads((ROOT / "settings.json").read_text())


def source_dir():
    return (ROOT / settings()["source"]).resolve()


def config_path(universe, variant="base"):
    from script_groups import normalise_group
    if universe not in UNIVERSES:raise ValueError("Unknown market")
    group=normalise_group(variant)
    return ROOT/"configs"/universe/f"{group}.yaml"


def region(universe):
    return "EU" if universe == "eu_large" else "US"


def assert_new_tree(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Refusing to write outside the new workflow: {path}")


def write_json(path, value):
    assert_new_tree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def recover_current_returns(rows):
    """R^excess_t is the prior row's t+1 return only for consecutive calendar months.

    Never use the current row's forward return to drift inherited weights.
    A screened-out month is a real gap, not a zero return or forward fill.
    """
    out = rows.sort_values(["id", "eom"]).reset_index(drop=True).copy()
    previous_date = out.groupby("id", sort=False)["eom"].shift()
    previous_return = out.groupby("id", sort=False)["ret_exc_lead1m"].shift()
    out["ret_exc"] = previous_return.where(previous_date + MONTH == out["eom"])
    return out


def load_monthly_supplement(universe):
    path = ROOT / f"data/inputs/monthly_returns_{region(universe)}.parquet"
    if not path.exists():
        return None
    provenance = json.loads(path.with_suffix(".json").read_text())
    if provenance.get("currency") != "USD" or provenance.get("return_type") != "excess":
        raise ValueError("Monthly supplements must declare USD excess returns")
    frame = numeric_ids(pd.read_parquet(path))
    needed = {"ret_exc", "ret_exc_lead1m"}
    if not needed.issubset(frame.columns):
        raise ValueError("Monthly supplement lacks required return columns")
    return frame


def fill_forward_gaps(rows, supplement):
    """Use exact source observations only; never invent a post-delisting return."""
    if supplement is None:
        return rows
    direct = supplement.set_index(["id", "eom"])["ret_exc_lead1m"]
    adjacent = supplement[["id", "eom", "ret_exc"]].copy()
    adjacent["eom"] = adjacent.eom - MONTH
    adjacent = adjacent.set_index(["id", "eom"])["ret_exc"]
    keys = pd.MultiIndex.from_frame(rows[["id", "eom"]])
    fill = pd.Series(direct.reindex(keys).to_numpy(dtype=float, na_value=np.nan), index=rows.index)
    fill = fill.fillna(pd.Series(adjacent.reindex(keys).to_numpy(dtype=float, na_value=np.nan), index=rows.index))
    out = rows.copy()
    out["ret_exc_lead1m"] = out.ret_exc_lead1m.fillna(fill)
    return out


def zero_missing_held_returns(panel, rf):
    """User-selected assumption: missing held-stock EXCESS returns are zero.

    Preserve the observed series and flag assumptions. Total returns then equal
    the next month's risk-free return; missing current returns are not changed.
    """
    out = panel.copy()
    out["ret_ld1_original"] = out["ret_ld1"]
    missing = out["in_universe"] & out["ret_ld1"].isna()
    out["ret_ld1_imputed_zero"] = missing
    next_rf = (out.loc[missing, "eom"] + MONTH).map(rf)
    if next_rf.isna().any():
        raise ValueError("Cannot apply zero excess-return assumption without next-month RF")
    out.loc[missing, "ret_ld1"] = 0.0
    out.loc[missing, "tr_ld1"] = next_rf.to_numpy()
    return out


def numeric_ids(frame):
    out = frame.copy()
    ids = pd.to_numeric(out["id"], errors="raise")
    if ids.isna().any() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError("Expected non-missing, integral JKP security identifiers")
    out["id"] = ids.astype("int64")
    out["eom"] = pd.to_datetime(out["eom"])
    if not out["eom"].dt.is_month_end.all() or out.duplicated(["id", "eom"]).any():
        raise ValueError("Keats rows must be unique by security and calendar month-end")
    return out


def broaden_membership(screened, universe):
    """Apply Keats' annual buffered rule at 250 names, preserving the mid-cap centre.

    Large caps: ranks 1..250, entry <=220, retention <=300.
    Mid caps: ranks 376..625 (centre 500.5), entry 406..595, retention 326..675.
    Retained positions may exceed target, as in the saved Keats 80-stock universes.
    Missing a month removes membership until the next December.
    """
    middle = universe == "us_mid"
    lo, hi = (376, 625) if middle else (1, 250)
    entry_lo, entry_hi = (406, 595) if middle else (1, 220)
    keep_lo, keep_hi = (326, 675) if middle else (1, 300)
    dec = screened[screened.eom.dt.month.eq(12)].copy()
    # Keats uses method='first' on the original screened row order.
    dec["mcap_rank"] = dec.groupby("eom")["me"].rank(ascending=False, method="first")
    previous, memberships = set(), []
    for eom, group in dec.groupby("eom", sort=True):
        ranks = group.set_index("id")["mcap_rank"]
        retained = {i for i in previous if i in ranks.index and keep_lo <= ranks[i] <= keep_hi}
        candidates = ranks[ranks.between(entry_lo, entry_hi)].drop(list(retained), errors="ignore")
        if middle:
            candidates = candidates.iloc[np.argsort(abs(candidates.to_numpy() - (lo + hi) / 2), kind="stable")]
            candidates = candidates.iloc[:max(0, 250 - len(retained))]
        statuses = {i: "retained" for i in retained} | {i: "new" for i in candidates.index}
        fill = ranks.drop(list(statuses), errors="ignore")
        fill = fill.iloc[np.argsort(abs(fill.to_numpy() - (lo + hi) / 2), kind="stable")]
        statuses.update({i: "fill" for i in fill.index[:max(0, 250 - len(statuses))]})
        selected = group[group.id.isin(statuses)][["id", "mcap_rank"]].copy()
        selected["form_year"] = eom.year
        selected["status"] = selected.id.map(statuses)
        memberships.append(selected)
        previous = set(statuses)
    membership = pd.concat(memberships, ignore_index=True)
    rows = screened[["id", "eom"]].copy()
    rows["form_year"] = np.where(rows.eom.dt.month.eq(12), rows.eom.dt.year, rows.eom.dt.year - 1)
    rows = rows.merge(membership, on=["id", "form_year"], validate="many_to_one")
    rows = rows.sort_values(["id", "form_year", "eom"])
    prior = rows.groupby(["id", "form_year"])["eom"].shift()
    broken = prior.notna() & (prior + MONTH != rows.eom)
    rows = rows[~broken.groupby([rows.id, rows.form_year]).cummax()]
    return rows


def missing_inputs(universe):
    cfg = load_config(config_path(universe))
    daily = ROOT / settings()["daily_inputs"][region(universe)]
    meta = daily.with_suffix(".json")
    missing = []
    for path, purpose in ((daily, "Daily stock-level USD TOTAL returns: id,date,ret; id is the JKP security id"),
                          (meta, 'Daily provenance: currency=USD, return_type=total, identifier=jkp_id, source'),
                          (ROOT / settings()["reference_data"] / RAW_FILES["ff_daily"], "Daily USD risk-free rate")):
        if not path.exists():
            missing.append({"path": str(path), "purpose": purpose})
    for series in cfg.regimes.fred_series:
        candidate = cfg.paths.raw / f"fred_{series}.parquet"
        reference = ROOT / settings()["reference_data"] / candidate.name
        shared = ROOT / f"data/inputs/fred_{series}.parquet"
        if not candidate.exists() and not reference.exists() and not shared.exists():
            missing.append({"path": str(candidate), "purpose": f"Optional cached FRED {series} for stages 10 and 13"})
    return missing


def audit():
    report = {"downloads_allowed": False, "source": str(source_dir()), "files": {}, "universes": {}}
    for path in sorted(source_dir().glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        dates = pd.read_parquet(path, columns=["eom"])["eom"]
        report["files"][path.name] = {"rows": parquet.metadata.num_rows, "columns": parquet.schema.names,
            "first_formation_month": str(dates.min().date()), "last_formation_month": str(dates.max().date()),
            "frequency": "monthly", "has_daily_stock_returns": False}
    for universe in UNIVERSES:
        cfg = load_config(config_path(universe))
        report["universes"][universe] = {"features": len(cfg.features.names),
            "missing_inputs": missing_inputs(universe), "test": [str(cfg.dates.test_start), str(cfg.dates.test_end)],
            "separate_holdout_available": False}
    write_json(ROOT / "outputs/input_audit.json", report)
    print(json.dumps({u: report["universes"][u] for u in UNIVERSES}, indent=2))
    return report


def prepare(universe, variant="base"):
    cfg = load_config(config_path(universe, variant))
    source = source_dir()
    fingerprint = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                   for p in (source / f"screened_{region(universe)}.parquet", source / f"panel_{universe}.parquet")}
    supplement_path = ROOT / f"data/inputs/monthly_returns_{region(universe)}.parquet"
    if supplement_path.exists():
        fingerprint[supplement_path.name] = {"size": supplement_path.stat().st_size,
                                              "mtime_ns": supplement_path.stat().st_mtime_ns}
    if (cfg.paths.processed / "preparation.json").exists():
        previous = json.loads((cfg.paths.processed / "preparation.json").read_text())
        current_config = json.loads(json.dumps(cfg.as_dict(), default=str))
        fields = ("dates", "features", "trading_costs")
        if previous.get("snapshot") != fingerprint or any(
            previous["config"][key] != current_config[key] for key in fields
        ) or any(previous["config"]["investor"][key] != current_config["investor"][key]
                 for key in ("wealth", "wealth_end_date")):
            raise ValueError("Prepared data is stale. Existing files retained; choose a new output folder before rebuilding.")
        if previous.get("missing_forward_return_policy", "stop") != settings().get("missing_forward_return_policy", "stop"):
            raise ValueError("Prepared return policy differs; retain existing files and prepare into a new output folder")
        print(f"Already prepared: {cfg.paths.processed} (existing files retained)")
        return
    assert_new_tree(cfg.paths.processed)
    assert_new_tree(cfg.paths.raw)
    chars = list(cfg.features.names)
    # Load the full screened region so training and cross-sectional ranks do not use only held names.
    use = ["id", "eom", "sic", "size_grp", "me", "ret_exc_lead1m", "hist_months", *chars]
    use = list(dict.fromkeys(use))
    raw = numeric_ids(pd.read_parquet(source / f"screened_{region(universe)}.parquet", columns=use))
    supplement = load_monthly_supplement(universe)
    raw = fill_forward_gaps(raw, supplement)
    if variant in ("universe_250", "halflife"):
        members = broaden_membership(raw, universe)
    else:
        members = numeric_ids(pd.read_parquet(source / f"panel_{universe}.parquet",
            columns=["id", "eom", "form_year", "mcap_rank", "status", *[c + "_rank" for c in chars]]))

    cfg.paths.raw.mkdir(parents=True, exist_ok=True)
    reference = (ROOT / settings()["reference_data"]).resolve()
    # Tiny reference files are copied once; the large source panels stay in their original location.
    for name in (RAW_FILES["ff_daily"], RAW_FILES["factor_details"], RAW_FILES["cluster_labels"]):
        dest = cfg.paths.raw / name
        if not dest.exists() and (reference / name).exists():
            shutil.copy2(reference / name, dest)
    for series in cfg.regimes.fred_series:
        path = reference / f"fred_{series}.parquet"
        if not path.exists():
            path = ROOT / f"data/inputs/fred_{series}.parquet"
        if path.exists() and not (cfg.paths.raw / path.name).exists():
            shutil.copy2(path, cfg.paths.raw / path.name)
    ff = pd.read_parquet(reference / RAW_FILES["ff_monthly"])[["date", "rf"]]
    ff["date"] = pd.to_datetime(ff.date)
    if ff.date.duplicated().any():
        raise ValueError("Duplicate monthly RF dates")

    # A regional value-weighted market wealth path, built with formation-month market caps.
    # This avoids using the US market as the European wealth-growth benchmark.
    market_rows = raw[raw.ret_exc_lead1m.notna() & raw.me.gt(0)].copy()
    market_rows["weighted"] = market_rows.me * market_rows.ret_exc_lead1m
    monthly = market_rows.groupby("eom")[["weighted", "me"]].sum()
    market = (monthly.weighted / monthly.me).rename("mkt_rf").reset_index()
    market["date"] = market.eom + MONTH
    market = market[["date", "mkt_rf"]].merge(ff, on="date", how="left", validate="one_to_one")
    market = market[market.date.le(pd.Timestamp(cfg.dates.data_end))]
    if market.rf.isna().any():
        raise ValueError("Monthly RF missing for a regional market-return month")
    market.to_parquet(cfg.paths.raw / RAW_FILES["ff_monthly"], index=False)
    wealth = costs.wealth_path(cfg.investor.wealth, cfg.investor.wealth_end_date,
        market_total_returns(market, market.date.min(), market.date.max()))
    # RF at the very first formation month is needed in addition to the forward-return RF dates.
    timed = recover_current_returns(raw)
    if supplement is not None:
        known = supplement.set_index(["id", "eom"])["ret_exc"]
        keys = pd.MultiIndex.from_frame(timed[["id", "eom"]])
        timed["ret_exc"] = timed.ret_exc.fillna(pd.Series(known.reindex(keys).to_numpy(dtype=float, na_value=np.nan), index=timed.index))
    timed = add_timing_columns(timed, risk_free_monthly(ff), wealth)
    timed["dolvol"] = timed["dolvol_126d"].astype(float)
    if timed.dolvol.isna().any() or not timed.dolvol.gt(0).all():
        raise ValueError("Expected positive Keats USD daily dollar volume")
    timed["lambda"] = costs.price_impact_lambda(timed.dolvol, cfg.trading_costs.price_impact)
    ranked = raw[chars].groupby(raw.eom).rank(pct=True).fillna(0.5)
    # recover_current_returns sorted the rows; align by explicit keys, never by row position.
    ranked = pd.concat([raw[["id", "eom"]], ranked], axis=1)
    if variant not in ("universe_250", "halflife"):
        check = members.merge(ranked, on=["id", "eom"], validate="one_to_one")
        if len(check) != len(members):
            raise ValueError("Saved member rows not found in the screened region")
        for char in chars:
            if not np.allclose(check[char].to_numpy(dtype=float), check[char + "_rank"].to_numpy(dtype=float), atol=1e-12):
                raise ValueError(f"Saved Keats ranks disagree with regional ranks for {char}")
    timed = timed.drop(columns=chars).merge(ranked, on=["id", "eom"], validate="one_to_one")
    directions = features.load_theme_directions(cfg.paths.raw, cfg.features.themes)
    theme = features.theme_scores(timed[chars], directions)
    timed[theme.columns] = theme
    # Keats permits ff49/gics when SIC is missing. Do not drop saved members or invent a sector.
    timed["ff12"] = ff12_industry(timed.sic)
    timed.loc[timed.sic.isna(), "ff12"] = "Other"
    member_keys = members[["id", "eom", "form_year", "mcap_rank", "status"]].copy()
    timed = timed.merge(member_keys, on=["id", "eom"], how="left", validate="one_to_one")
    timed["in_universe"] = timed.form_year.notna()
    from jkmp.data.lag_history import exclude_unavailable
    timed, history_exclusions = exclude_unavailable(timed, universe, cfg.universe.lookback_months)
    timed["recon_eom"] = pd.to_datetime(timed.form_year.astype("Int64").astype("string") + "-12-31")
    if settings().get("missing_forward_return_policy") == "zero_excess":
        timed = zero_missing_held_returns(timed, risk_free_monthly(ff))
    timed["valid_data"] = timed.hist_months.ge(12) & timed.ret_ld1.notna() & timed.tr_ld0.notna()
    timed["g"] = costs.drift_factor(timed.tr_ld0, timed.mu_ld0)
    # The final formation month earns a 2026 return and cannot enter the declared 2020-25 test.
    timed = timed[(timed.eom + MONTH).le(pd.Timestamp(cfg.dates.test_end))].copy()
    for col in [*chars, *theme.columns, "me", "dolvol", "lambda", "ret_ld1", "ret_exc", "tr_ld0", "tr_ld1", "g"]:
        timed[col] = pd.to_numeric(timed[col], errors="raise").astype(float)
    if not np.isfinite(timed[chars].to_numpy()).all():
        raise ValueError("Nonfinite ranked ML input")
    # Top-500 covariance estimation pool belongs to the same region as the traded universe.
    top = raw.assign(rank=raw.groupby("eom")["me"].rank(ascending=False, method="first"))
    top = top[top["rank"].le(cfg.covariance.factor_universe_top_n)][["id", "eom"]]
    top.to_parquet(cfg.paths.raw / f"top{cfg.covariance.factor_universe_top_n}_by_month.parquet", index=False)
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    timed = timed.sort_values(["eom", "id"], ignore_index=True)
    timed.to_parquet(cfg.paths.processed / "panel.parquet", index=False)
    history_exclusions.to_csv(cfg.paths.processed / "history_exclusions.csv", index=False)
    if "ret_ld1_imputed_zero" in timed:
        timed.loc[timed.ret_ld1_imputed_zero, ["id", "eom", "ret_ld1_original", "ret_ld1", "tr_ld1"]].to_csv(
            cfg.paths.processed / "zero_return_imputations.csv", index=False)
    wealth.to_parquet(cfg.paths.processed / "wealth.parquet", index=False)
    member_keys.to_parquet(cfg.paths.processed / "universe_membership.parquet", index=False)
    counts = timed[timed.in_universe].groupby("eom").size().rename("n_members")
    counts.to_csv(cfg.paths.processed / "universe_stats.csv")
    screen = pd.DataFrame([{"note": "Keats screens preserved; no additional selection on forward returns",
        "screened_rows": len(raw), "prepared_rows": len(timed), "invalid_current_return": int(timed.tr_ld0.isna().sum())}])
    screen.to_csv(cfg.paths.processed / "screen_log.csv", index=False)
    report = {"universe": universe, "variant": variant, "features": len(chars), "rows": len(timed),
        "members": int(timed.in_universe.sum()), "valid_training_rows": int(timed.valid_data.sum()),
        "missing_current_return_rows": int(timed.tr_ld0.isna().sum()),
        "missing_SIC_member_rows": int((timed.in_universe & timed.sic.isna()).sum()),
        "missing_forward_return_policy": settings().get("missing_forward_return_policy", "stop"),
        "zero_imputed_forward_returns": int(timed.get("ret_ld1_imputed_zero", pd.Series(False, index=timed.index)).sum()),
        "history_excluded_stock_months": len(history_exclusions),
        "ranks": "Keats average percentile ranks, imputed to 0.5; all 50 saved member ranks verified",
        "wealth_market": "formation-cap-weighted screened regional USD excess returns + USD RF",
        "wealth_anchor": str(cfg.investor.wealth_end_date), "dollar_volume_units": "USD, not millions",
        "snapshot": fingerprint, "config": cfg.as_dict(), "missing_inputs": missing_inputs(universe)}
    write_json(cfg.paths.processed / "preparation.json", report)
    from jkmp.imputation_audit import report_if_enabled
    report_if_enabled(cfg)
    print(f"Prepared {universe}/{variant}: {len(timed):,} rows, {int(timed.in_universe.sum()):,} member rows; {len(chars)} features")


def covariance(universe, variant="base"):
    from jkmp import covariance as cov
    cfg = load_config(config_path(universe, variant))
    out = cfg.paths.processed / "covariance"
    if (out / "cov_factor.parquet").exists():
        print(f"Existing covariance retained: {out}")
        return
    daily_path = ROOT / settings()["daily_inputs"][region(universe)]
    if not daily_path.exists() or not daily_path.with_suffix(".json").exists():
        raise FileNotFoundError(f"Missing {daily_path} and/or its provenance JSON. No downloads were attempted. "
            "Provide stock-level USD total returns (id,date,ret); see README.md and outputs/input_audit.json.")
    provenance = json.loads(daily_path.with_suffix(".json").read_text())
    expected = {"currency": "USD", "return_type": "total", "identifier": "jkp_id"}
    if any(provenance.get(k) != v for k, v in expected.items()) or not provenance.get("source"):
        raise ValueError("Daily return provenance must declare USD total returns and JKP security ids")
    daily = pd.read_parquet(daily_path, columns=["id", "date", "ret"])
    daily["date"] = pd.to_datetime(daily.date)
    daily["id"] = pd.to_numeric(daily.id, errors="raise")
    if daily.id.isna().any() or not np.equal(daily.id, np.floor(daily.id)).all():
        raise ValueError("Daily ids must be integral JKP security ids")
    daily["id"] = daily.id.astype("int64")
    if daily.duplicated(["id", "date"]).any() or not np.isfinite(daily.ret).all() or (daily.ret < -1).any():
        raise ValueError("Invalid/duplicate daily return observations")
    panel = pd.read_parquet(cfg.paths.processed / "panel.parquet")
    # Covariance depends on historical daily returns/loadings, not current/forward monthly targets.
    # Return completeness is checked by the portfolio accounting/model stages that actually use it.
    top = pd.read_parquet(cfg.paths.raw / f"top{cfg.covariance.factor_universe_top_n}_by_month.parquet")
    needed = set(top.id) | set(panel.loc[panel.in_universe, "id"])
    absent = needed - set(daily.id)
    if absent:
        raise ValueError(f"Daily file has no observations for {len(absent)} required stocks (e.g. {sorted(absent)[:5]})")
    # 'permno' is only the original covariance function's input-column name. EU values remain JKP ids.
    daily = daily[daily.id.isin(needed)].rename(columns={"id": "permno"})
    result = cov.build_covariance(panel, top, daily, pd.read_parquet(cfg.paths.raw / RAW_FILES["ff_daily"]), cfg)
    expected_months = pd.date_range("2013-12-31", "2025-11-30", freq="ME")
    missing_months = expected_months.difference(pd.DatetimeIndex(result.parts))
    if len(missing_months):
        raise ValueError(f"Daily data does not support {len(missing_months)} required risk months; "
                         f"first missing {missing_months[0].date()}. No partial covariance was saved.")
    cov.save_covariance(result.parts, out)
    result.factor_returns.to_parquet(out / "factor_returns.parquet")
    result.diagnostics.to_csv(out / "cov_diagnostics.csv", index=False)
    write_json(out / "provenance.json", provenance)


def validate_prepared():
    """Publish the observed data gaps instead of claiming a complete backtest."""
    summary = []
    for universe in UNIVERSES:
        cfg = load_config(config_path(universe))
        path = cfg.paths.processed / "panel.parquet"
        if not path.exists():
            continue
        cols = ["id", "eom", "in_universe", "tr_ld0", "ret_ld1", "sic"]
        if "ret_ld1_imputed_zero" in pq.ParquetFile(path).schema.names:
            cols.append("ret_ld1_imputed_zero")
        panel = pd.read_parquet(path, columns=cols)
        scored = panel[panel.in_universe & panel.eom.ge(pd.Timestamp("2013-12-31"))].copy()
        gaps = scored[scored.tr_ld0.isna() | scored.ret_ld1.isna()].copy()
        gaps["missing_current_month_return"] = gaps.tr_ld0.isna()
        gaps["missing_next_month_return"] = gaps.ret_ld1.isna()
        gap_path = ROOT / "outputs" / universe / "monthly_return_gaps.csv"
        gap_path.parent.mkdir(parents=True, exist_ok=True)
        gaps[["id", "eom", "missing_current_month_return", "missing_next_month_return"]].to_csv(gap_path, index=False)
        summary.append({"universe": universe, "prepared_rows": len(panel),
            "member_rows": int(panel.in_universe.sum()),
            "held_missing_current_return_since_2013_12": int(scored.tr_ld0.isna().sum()),
            "held_missing_forward_return_since_2013_12": int(scored.ret_ld1.isna().sum()),
            "zero_imputed_forward_returns_all_history": int(panel.get("ret_ld1_imputed_zero", pd.Series(False, index=panel.index)).sum()),
            "gap_details": str(gap_path.relative_to(ROOT)), "full_backtest_complete": False})
    write_json(ROOT / "outputs/prepared_validation.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def originals_unchanged():
    manifest = json.loads((ROOT / "original_manifest.json").read_text())
    changed = [name for name, digest in manifest.items()
               if hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise AssertionError(f"Original source files changed: {changed}")
    return True


def ensure_prediction_coverage(cfg):
    """Fill absent held-stock forecasts from already fitted models, never fabricate returns."""
    import datetime as dt
    from uuid import uuid4
    from jkmp.models.return_model import load_models
    p=cfg.paths.processed
    prediction_path=p/"predictions.parquet"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Run the return-model stage first: {prediction_path}")
    cache=p/"return_model/prediction_coverage.json"
    def signature():
        return {name:{"size":(p/name).stat().st_size,"mtime_ns":(p/name).stat().st_mtime_ns}
                for name in ("panel.parquet","predictions.parquet")}
    before=signature()
    if cache.exists() and json.loads(cache.read_text()).get("inputs")==before:
        return
    columns=["id","eom","in_universe",*cfg.features.names]
    held=pd.read_parquet(p/"panel.parquet",columns=columns,filters=[("in_universe","=",True)])
    return_dates=held.eom+MONTH
    held=held[return_dates.between(pd.Timestamp(cfg.return_model.pred_years[0],1,31),pd.Timestamp(cfg.dates.test_end))]
    predictions=pd.read_parquet(prediction_path)
    present=pd.MultiIndex.from_frame(predictions[["id","eom"]])
    missing=held[~pd.MultiIndex.from_frame(held[["id","eom"]]).isin(present)].copy()
    if len(missing):
        models,z=load_models(p/"return_model/models.npz")
        missing["year"]=(missing.eom+MONTH).dt.year
        additions=[]
        for year,group in missing.groupby("year"):
            if int(year) not in models:
                raise ValueError(f"No fitted return model for held-stock forecast year {year}")
            predicted=models[int(year)].predict(group[list(cfg.features.names)].to_numpy(dtype=float),z)
            additions.append(group[["id","eom","year"]].assign(pred_ld1=predicted))
        extra=pd.concat(additions,ignore_index=True)
        archive=p/"return_model/previous_predictions"/(dt.datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid4().hex[:8])
        archive.mkdir(parents=True)
        shutil.copy2(prediction_path,archive/prediction_path.name)
        pd.concat([predictions,extra],ignore_index=True).sort_values(["eom","id"]).to_parquet(prediction_path,index=False)
        extra.to_csv(p/"return_model/forecast_coverage_additions.csv",index=False)
        print("Additional held-stock forecasts from existing fitted models (no new training labels):",flush=True)
        print(extra.to_string(index=False),flush=True)
    write_json(cache,{"inputs":signature(),"missing_held_forecasts":0,"added_forecasts":len(missing)})
