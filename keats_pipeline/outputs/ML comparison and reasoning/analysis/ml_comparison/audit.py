"""Read-only audit of saved Keats inputs, dates, weights and portfolio accounting.

Nothing in this module fits a model or writes beneath ``keats_pipeline``.  Saved
weights are checked with an independent direct calculation of accounting.py's
formulas.  Date checks establish internal consistency, not point-in-time vendor
availability, which cannot be established from these saved research snapshots.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


MONTH = pd.offsets.MonthEnd(1)
TOLERANCE = 1e-10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root: Path, value) -> Path | None:
    if value is None or (not isinstance(value, Path) and pd.isna(value)) or str(value).strip() == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _config(path: Path, seen=None) -> dict:
    """Resolve YAML inheritance independently without importing a stage runner."""
    seen = set() if seen is None else seen
    if path.resolve() in seen:
        raise ValueError(f"recursive config inheritance: {path}")
    seen.add(path.resolve())
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = value.pop("base", None)
    merged = _config(path.parent / base, seen) if base else {}
    def merge(a, b):
        for key, item in b.items():
            if isinstance(item, dict) and isinstance(a.get(key), dict):
                merge(a[key], item)
            else:
                a[key] = item
        return a
    return merge(merged, value)


def independent_accounting(weights: pd.DataFrame, panel: pd.DataFrame,
                           wealth: pd.DataFrame) -> pd.DataFrame:
    """Independent Eq. 4-style direct sums; turnover is full L1 trading, not half."""
    c = weights.merge(panel[["id", "eom", "ret_ld1", "lambda"]],
                      on=["id", "eom"], how="left", validate="many_to_one")
    c = c.merge(wealth[["eom", "wealth"]], on="eom", how="left", validate="many_to_one")
    exits = c["exit"].fillna(False).astype(bool)
    required = c[["lambda", "wealth", "w", "w_start"]].isna().any(axis=1)
    required |= (~exits) & c["ret_ld1"].isna()
    if required.any():
        raise ValueError(f"{int(required.sum())} weights missing return/cost/wealth inputs")
    delta = c["w"] - c["w_start"]
    contributions = pd.DataFrame({"eom": c["eom"],
        "r": c["w"] * c["ret_ld1"].where(~exits, 0.0),
        "tc": 0.5 * c["wealth"] * c["lambda"] * delta ** 2,
        "turnover": delta.abs(), "inv": c["w"].abs(),
        "shorting": (-c["w"].clip(upper=0))})
    return contributions.groupby("eom", sort=True).sum().reset_index()


def covariance_diagnostics(load: pd.DataFrame, fac: pd.DataFrame,
                           months: set[pd.Timestamp]) -> tuple[pd.DataFrame, dict]:
    """Exact eigenvalue/conditioning checks on saved test-month covariance matrices."""
    names = [c for c in fac.columns if c not in ("eom", "factor")]
    records, cache = [], {}
    previous = None
    previous_f = None
    for eom, f in fac[fac["eom"].isin(months)].groupby("eom", sort=True):
        L = load[load["eom"].eq(eom)].sort_values("id")
        F = f.set_index("factor").loc[names, names].to_numpy(dtype=float)
        X = L[names].to_numpy(dtype=float)
        ivar = L["ivar"].to_numpy(dtype=float)
        sigma = X @ F @ X.T + np.diag(ivar)
        finite = bool(np.isfinite(sigma).all())
        symmetry_error = float(np.max(np.abs(sigma - sigma.T))) if finite else np.nan
        eig = np.linalg.eigvalsh((sigma + sigma.T) / 2) if finite else np.full(len(L), np.nan)
        feig = np.linalg.eigvalsh((F + F.T) / 2) if finite else np.full(len(F), np.nan)
        minimum = float(eig.min()) if len(eig) else np.nan
        maximum = float(eig.max()) if len(eig) else np.nan
        relative_change = (float(np.linalg.norm(F - previous_f, "fro") / np.linalg.norm(previous_f, "fro"))
                           if previous_f is not None and np.linalg.norm(previous_f, "fro") else np.nan)
        records.append({"eom": eom, "n_stocks": len(L), "n_factors": len(names),
            "finite": finite, "symmetry_max_abs": symmetry_error,
            "minimum_eigenvalue": minimum, "maximum_eigenvalue": maximum,
            "condition_number": maximum / minimum if minimum > 0 else np.inf,
            "positive_definite": finite and minimum > 0,
            "factor_minimum_eigenvalue": float(feig.min()) if len(feig) else np.nan,
            "negative_idiosyncratic_variances": int((ivar < 0).sum()),
            "factor_covariance_relative_frobenius_change": relative_change,
            "previous_eom": previous})
        cache[eom] = (pd.Index(L["id"]), sigma)
        previous, previous_f = eom, F
    return pd.DataFrame(records), cache


def _source_type(record: dict, weights: pd.DataFrame, saved: pd.DataFrame):
    kind = record.get("source_type", record["strategy"])
    if "type" in weights:
        if kind in set(weights["type"]):
            return weights[weights["type"].eq(kind)].copy()
        # A saved fixed Static-ML return path has no saved fixed-ML weights.
        if len(weights["type"].unique()) == 1 and "type" not in saved:
            return weights.copy()
        return weights.iloc[:0].copy()
    return weights.copy()


def _records_frame(rows: list, filename: Path, columns=None) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty and columns:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(filename, index=False)
    return frame


def history_input_status(root: Path, applied_exclusions: list[dict]) -> list[dict]:
    """List actual unavailable source-history records separately from applied exclusions.

    A documented source gap is not evidence that a not-yet-prepared portfolio
    has actually excluded a stock. In particular, the EU250 prepared dataset
    need not exist merely because README describes its intended policy.
    """
    rows = []
    for region in ["US", "EU"]:
        path = Path(root) / f"keats_pipeline/data/inputs/unavailable_lag_history_{region}.csv"
        if not path.is_file():
            continue
        data = pd.read_csv(path)
        for record in data.to_dict("records"):
            events = [event for event in applied_exclusions
                      if int(event["stock_id"]) == int(record["id"])
                      and str(event["dataset"]).startswith("eu_" if region == "EU" else "us_")]
            rows.append(dict(region=region, stock_id=int(record["id"]),
                missing_history_month_end=record["eom"], source_path=str(path),
                applied_exclusion_records=len(events),
                applied_datasets=";".join(sorted({event["dataset"] for event in events})),
                application_status="Applied exclusion present in saved prepared ledger" if events else
                    "Source gap documented; no saved applied exclusion ledger event",
                interpretation="Unavailable source history is retained as evidence; no applied exclusion is invented."))
    return rows


def non_executable_saved_month(record: dict, risk_free: float) -> dict | None:
    """Flag nonpositive funded growth using total, not excess, return thresholds."""
    gross_total = record["r"] + risk_free
    net_total = record["r"] - record["tc"] + risk_free
    if not np.isfinite(risk_free) or (gross_total > -1 and net_total > -1):
        return None
    return dict(dataset=record["dataset"], strategy=record["strategy"],
        eom=record["eom"], eom_ret=record["eom_ret"], gross_excess_return=record["r"],
        gross_total_return=gross_total, transaction_cost=record["tc"], risk_free=risk_free,
        net_total_return=net_total, gross_excess_below_minus_one=record["r"] <= -1,
        gross_total_below_minus_one=gross_total <= -1, net_total_below_minus_one=net_total <= -1,
        interpretation="Saved unconstrained/quadratic-cost path; not executable after depletion of investor equity")


def run_audit(root: Path, destination: Path, inventory: pd.DataFrame,
              monthly: pd.DataFrame) -> dict:
    """Audit saved sources, write reproducible CSVs/notes, and return a summary.

    ``root`` is the repository root; destination is a NEW comparison-run audit
    directory. Inventory uses the parent's dataset/strategy source catalog.
    """
    root, destination = Path(root).resolve(), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    findings, sources, configs, exclusions, zeroes = [], [], [], [], []
    accounting_rows, risk_rows, cov_rows, panel_rows, timing_rows, diagnostics = [], [], [], [], [], []
    extreme_costs, infeasible_months, history_coverage = [], [], []
    seen_sources = set()
    audited_processed = set()

    def finding(dataset, strategy, check, status, detail):
        findings.append(dict(dataset=dataset, strategy=strategy, check=check,
                             status=status, detail=str(detail)))

    def source(path, role):
        if path is None or not path.is_file() or path.resolve() in seen_sources:
            return
        seen_sources.add(path.resolve())
        sources.append(dict(path=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            role=role, bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
            sha256=file_sha256(path)))

    for (dataset, strategy), m in monthly.groupby(["dataset", "strategy"], sort=True):
        duplicate = int(m.duplicated("eom_ret").sum())
        numeric = [c for c in ["r", "tc", "turnover", "inv", "shorting"] if c in m]
        bad = int((~np.isfinite(m[numeric].to_numpy(dtype=float))).sum())
        finding(dataset, strategy, "unique_monthly_rows", "PASS" if duplicate == 0 else "FAIL", f"duplicate return dates={duplicate}")
        finding(dataset, strategy, "finite_saved_metrics", "PASS" if bad == 0 else "FAIL", f"nonfinite values={bad}")
        negative = int((m["tc"] < -TOLERANCE).sum())
        finding(dataset, strategy, "nonnegative_costs", "PASS" if negative == 0 else "FAIL", f"negative costs={negative}")
        expected = pd.date_range(m.eom_ret.min(), m.eom_ret.max(), freq="ME")
        absent = expected.difference(m.eom_ret)
        finding(dataset, strategy, "consecutive_test_months", "PASS" if len(absent) == 0 else "FAIL", f"missing={','.join(d.strftime('%Y-%m') for d in absent)}")
        if "eom" in m:
            bad_dates = int((m["eom"] + MONTH != m["eom_ret"]).sum())
            finding(dataset, strategy, "formation_to_return_timing", "PASS" if bad_dates == 0 else "FAIL", f"eom_ret != eom+one month={bad_dates}")
        # An identity check only; independent stock-level reconstruction below is stronger.
        if "net" in m:
            difference = float(np.max(np.abs(m["r"] - m["tc"] - m["net"])))
            finding(dataset, strategy, "gross_minus_cost_equals_saved_net", "PASS" if difference < TOLERANCE else "FAIL", difference)

    by_processed = inventory.groupby("processed_path", dropna=False, sort=False)
    for processed_value, records in by_processed:
        processed = _path(root, processed_value)
        datasets = sorted(records.dataset.unique())
        dataset_label = ";".join(datasets)
        if processed is None or not (processed / "panel.parquet").is_file():
            finding(dataset_label, "ALL", "panel_available", "SKIPPED", str(processed))
            continue
        audited_processed.add(processed.resolve())
        panel_path = processed / "panel.parquet"
        schema = pq.read_schema(panel_path)
        available = set(schema.names)
        cols = [c for c in ["id", "eom", "ret_ld1", "tr_ld1", "lambda", "dolvol", "me", "in_universe", "valid_data", "ret_ld1_original", "ret_ld1_imputed_zero"] if c in available]
        panel_full = pd.read_parquet(panel_path, columns=cols)
        duplicate_panel = int(panel_full.duplicated(["id", "eom"]).sum())
        finding(dataset_label, "ALL", "panel_unique_security_month", "PASS" if duplicate_panel == 0 else "FAIL", f"duplicate rows={duplicate_panel}")
        mset = monthly[monthly["dataset"].isin(datasets)]
        months = set(pd.to_datetime(mset["eom"]).unique())
        months = {pd.Timestamp(d) for d in months}
        panel = panel_full[panel_full["eom"].isin(months)].copy()
        member = panel[panel["in_universe"]]
        if "dolvol" in panel:
            theoretical_lambda = 0.2 / panel.dolvol
            delta_lambda = float((panel["lambda"] - theoretical_lambda).abs().max())
            finding(dataset_label,"ALL","lambda_equals_point_two_over_saved_usd_volume",
                    "PASS" if np.allclose(panel["lambda"],theoretical_lambda,atol=1e-14,rtol=1e-10) else "FAIL",
                    f"max absolute discrepancy={delta_lambda}; this verifies consistency with saved USD labels, not independent venue unit validation")
        risk_free = (panel.tr_ld1-panel.ret_ld1).groupby(panel.eom).median()
        missing_rf = int(pd.Index(sorted(months)).difference(risk_free.dropna().index).size)
        finding(dataset_label, "ALL", "risk_free_for_funded_growth_checks",
                "PASS" if missing_rf == 0 else "SKIPPED",
                f"Formation months lacking a saved implied next-month risk-free return={missing_rf}; funded-growth checks use r+RF, not excess r alone")
        for record in mset.to_dict("records"):
            rf=float(risk_free.get(record["eom"], np.nan))
            infeasible = non_executable_saved_month(record, rf)
            if infeasible is not None:
                infeasible_months.append(infeasible)
        panel_rows.append(dict(dataset=dataset_label, processed_path=str(processed),
            rows=len(panel_full), securities=panel_full.id.nunique(),
            first_formation=panel_full.eom.min(), last_formation=panel_full.eom.max(),
            duplicate_security_months=duplicate_panel, test_member_rows=len(member),
            minimum_test_stocks=int(member.groupby("eom").size().min()),
            maximum_test_stocks=int(member.groupby("eom").size().max()),
            missing_member_returns=int(member["ret_ld1"].isna().sum()),
            missing_member_cost_coefficients=int(member["lambda"].isna().sum()),
            zero_imputed_full_history=int(panel_full.get("ret_ld1_imputed_zero", pd.Series(False,index=panel_full.index)).sum()),
            zero_imputed_test=int(panel.get("ret_ld1_imputed_zero", pd.Series(False,index=panel.index)).sum())))
        flagged = panel_full.loc[panel_full.get("ret_ld1_imputed_zero", pd.Series(False,index=panel_full.index)).astype(bool)]
        if len(flagged):
            bad_zero = int((~np.isclose(flagged.ret_ld1, 0.0, atol=1e-14, rtol=0)).sum())
            original_observed = int(flagged.ret_ld1_original.notna().sum()) if "ret_ld1_original" in flagged else len(flagged)
            finding(dataset_label, "ALL", "zero_return_flags_preserve_missing_originals",
                    "PASS" if bad_zero == 0 and original_observed == 0 else "FAIL",
                    f"flagged history rows={len(flagged)}; nonzero assumed excess={bad_zero}; originally observed but flagged={original_observed}")
            observed = panel_full.loc[~panel_full.ret_ld1_imputed_zero.astype(bool)]
            observed_rf = (observed.tr_ld1-observed.ret_ld1).groupby(observed.eom).median()
            expected_rf = flagged.eom.map(observed_rf)
            finite = np.isfinite(expected_rf)
            mismatch = int((~np.isclose(flagged.loc[finite, "tr_ld1"], expected_rf[finite], atol=1e-12, rtol=0)).sum())
            finding(dataset_label, "ALL", "assumed_total_return_matches_other_stocks_implied_risk_free",
                    "PASS" if mismatch == 0 and finite.all() else "FAIL" if mismatch else "SKIPPED",
                    f"total-return mismatches={mismatch}; rows without independent same-month observed-stock RF={int((~finite).sum())}")
        source(panel_path, "prepared panel")
        for name in ["preparation.json", "screen_log.csv", "universe_stats.csv"]:
            source(processed/name,"saved data preparation provenance")
        for name, target in [("history_exclusions.csv", exclusions), ("zero_return_imputations.csv", zeroes)]:
            path = processed / name
            if path.is_file():
                source(path, name)
                try:
                    d = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    d = pd.DataFrame()
                if len(d):
                    if name == "zero_return_imputations.csv":
                        keys = set(zip(flagged.id.astype(int), pd.to_datetime(flagged.eom)))
                        ledger_keys = set(zip(d.id.astype(int), pd.to_datetime(d.eom)))
                        finding(dataset_label, "ALL", "zero_return_flag_ledger_agreement",
                                "PASS" if keys == ledger_keys and len(d) == len(flagged) else "FAIL",
                                f"flagged rows={len(flagged)}; ledger rows={len(d)}; differing security-month keys={len(keys.symmetric_difference(ledger_keys))}")
                    d.insert(0, "dataset", dataset_label)
                    target.extend(d.to_dict("records"))
        del panel_full
        wealth_path = processed / "wealth.parquet"
        wealth = pd.read_parquet(wealth_path) if wealth_path.is_file() else None
        source(wealth_path, "wealth and screened regional market growth")
        covdir = processed / "covariance"
        cov_cache = {}
        if (covdir / "cov_loadings.parquet").is_file() and (covdir / "cov_factor.parquet").is_file():
            load = pd.read_parquet(covdir / "cov_loadings.parquet")
            fac = pd.read_parquet(covdir / "cov_factor.parquet")
            cv, cov_cache = covariance_diagnostics(load, fac, months)
            cv.insert(0, "dataset", dataset_label)
            cov_rows.extend(cv.to_dict("records"))
            source(covdir / "cov_loadings.parquet", "saved covariance loadings/idiosyncratic variance")
            source(covdir / "cov_factor.parquet", "saved factor covariance")
            absent = months.difference(cov_cache)
            finding(dataset_label, "ALL", "covariance_test_coverage", "PASS" if not absent else "FAIL", f"missing formation months={len(absent)}")
            badcov = int((~cv.positive_definite | ~cv.finite | cv.symmetry_max_abs.gt(1e-10) | cv.negative_idiosyncratic_variances.gt(0)).sum())
            finding(dataset_label, "ALL", "covariance_positive_definite", "PASS" if not badcov else "FAIL", f"invalid matrices={badcov}; inspected={len(cv)}")
            for name in ["provenance.json", "cov_diagnostics.csv"]:
                path = covdir / name
                source(path, "covariance provenance/diagnostics")
                if name.endswith("csv") and path.is_file():
                    d = pd.read_csv(path); d.insert(0,"dataset",dataset_label)
                    diagnostics.extend(d.to_dict("records"))
        else:
            finding(dataset_label, "ALL", "covariance_validity", "SKIPPED", "No saved covariance pieces")

        pred_path = processed / "predictions.parquet"
        if pred_path.is_file():
            preds = pd.read_parquet(pred_path)
            source(pred_path, "saved one-month predictions")
            duplicates = int(preds.duplicated(["id", "eom"]).sum())
            badyears = int(((pd.to_datetime(preds.eom) + MONTH).dt.year != preds.year).sum()) if "year" in preds else -1
            nonfinite = int((~np.isfinite(preds["pred_ld1"])).sum())
            finding(dataset_label, "return model", "prediction_dates_and_finiteness", "PASS" if duplicates == 0 and badyears == 0 and nonfinite == 0 else "FAIL", f"duplicates={duplicates}; wrong return year={badyears}; nonfinite={nonfinite}")
            covered = member[["id", "eom"]].merge(preds[["id", "eom"]], on=["id", "eom"], how="left", indicator=True)
            missing_pred = int(covered._merge.ne("both").sum())
            finding(dataset_label, "return model", "held_prediction_coverage", "PASS" if missing_pred == 0 else "FAIL", f"missing held security-month predictions={missing_pred}")
            del preds
        else:
            finding(dataset_label, "return model", "saved_predictions", "SKIPPED", "No predictions.parquet")

        weights_cache = {}
        for record in records.to_dict("records"):
            dataset, strategy = record["dataset"], record["strategy"]
            cfg_path = _path(root, record.get("config_path"))
            if cfg_path and cfg_path.is_file():
                source(cfg_path, "effective variant config")
                cfg = _config(cfg_path)
                flags=cfg.get("portfolio_ml",{})
                override_note=""
                source_type=record.get("source_type",strategy)
                if "active" in str(source_type).lower() and record.get("variant")=="base":
                    active_path=root/"keats_pipeline/configs"/record["universe"]/"active_pfml.yaml"
                    if active_path.is_file():
                        source(active_path,"active-run effective configuration")
                        flags=_config(active_path).get("portfolio_ml",{})
                        override_note="Saved active-run path implies active_pfml.yaml; TE risk aversion from calibrated folder, not absolute gamma=10"
                if record.get("variant")=="universe_250" and "/portfolio_ml/" in str(record.get("pf_path","")).replace("\\","/"):
                    flags=dict(flags,benchmark_relative=False,add_constant=True)
                    override_note="run_all standard u250_pfml explicitly overrides benchmark_relative=false and add_constant=true"
                configs.append(dict(dataset=dataset, strategy=strategy, config_path=str(cfg_path),
                    validation_start=cfg.get("dates",{}).get("validation_start"),
                    validation_end=cfg.get("dates",{}).get("validation_end"),
                    test_start=cfg.get("dates",{}).get("test_start"),test_end=cfg.get("dates",{}).get("test_end"),
                    return_model_validation_years=cfg.get("return_model",{}).get("val_years"),
                    baseline_gamma=cfg.get("investor",{}).get("gamma"), baseline_wealth=cfg.get("investor",{}).get("wealth"),
                    charge_exit_tc=cfg.get("portfolio",{}).get("charge_exit_tc"),
                    feature_count=sum(len(v) for v in cfg.get("features",{}).get("themes",{}).values()),
                    price_impact=cfg.get("trading_costs",{}).get("price_impact"),
                    portfolio_ml_benchmark_relative=flags.get("benchmark_relative"),
                    portfolio_ml_add_constant=flags.get("add_constant"),
                    effective_override_notes=override_note,
                    separate_holdout=cfg.get("dates",{}).get("holdout_start") is not None))
            saved = monthly[(monthly.dataset.eq(dataset)) & monthly.strategy.eq(strategy)]
            weight_path = _path(root, record.get("weights_path"))
            if weight_path is None or not weight_path.is_file() or wealth is None:
                finding(dataset, strategy, "independent_accounting", "SKIPPED", "No corresponding saved weights or wealth")
                continue
            if weight_path not in weights_cache:
                weights_cache[weight_path] = pd.read_parquet(weight_path)
            w = _source_type(record, weights_cache[weight_path], saved)
            if w.empty:
                finding(dataset, strategy, "independent_accounting", "SKIPPED", "No saved weights for this source type (fixed Static-ML weights are not retained)")
                continue
            w = w[w.eom.isin(saved.eom)]
            source(weight_path, "saved portfolio weights")
            duplicates = int(w.duplicated(["id", "eom"]).sum())
            finding(dataset, strategy, "unique_weights", "PASS" if duplicates == 0 else "FAIL", f"duplicates={duplicates}")
            if duplicates:
                continue
            working_wealth = wealth.copy()
            scale = record.get("wealth_scale", 1.0)
            if pd.isna(scale):
                scale = 1.0
            working_wealth["wealth"] *= float(scale)
            try:
                independent = independent_accounting(w, panel, working_wealth)
                comp = saved.merge(independent, on="eom", suffixes=("_saved", "_reconstructed"), validate="one_to_one")
                errors = {field: float(np.max(np.abs(comp[field + "_saved"] - comp[field + "_reconstructed"]))) for field in ["r", "tc", "turnover", "inv", "shorting"]}
                metrics=["r", "tc", "turnover", "inv", "shorting"]
                equal=all(np.allclose(comp[field+"_saved"],comp[field+"_reconstructed"],atol=TOLERANCE,rtol=1e-10) for field in metrics)
                status = "PASS" if len(comp) == len(saved) and equal else "FAIL"
                finding(dataset, strategy, "independent_accounting", status, f"months={len(comp)}/{len(saved)}; max errors={errors}")
                accounting_rows.append(dict(dataset=dataset,strategy=strategy,n_months=len(comp),status=status,**{"max_abs_error_"+k:v for k,v in errors.items()}))
            except (ValueError, KeyError) as error:
                finding(dataset, strategy, "independent_accounting", "FAIL", str(error))
            # Trace large costs to actual holdings, listing volume and wealth;
            # no winsorisation or deletion of economically impossible months.
            if "dolvol" in panel:
                c=w.merge(panel[["id","eom","dolvol","lambda","ret_ld1"]],on=["id","eom"],how="left")
                c=c.merge(working_wealth[["eom","wealth"]],on="eom",how="left")
                c["delta_w"]=c.w-c.w_start
                c["tc_contribution"]=0.5*c.wealth*c["lambda"]*c.delta_w**2
                c["trade_dollars"]=c.wealth*c.delta_w.abs()
                c["daily_dollar_volume_participation"]=c.trade_dollars/c.dolvol
                c=c.nlargest(20,"tc_contribution")
                membership=root/"configs/Keats universe code/data/universes Keats"/f"membership_{record['universe']}.csv"
                if membership.is_file():
                    names=pd.read_csv(membership)
                    names=names[[v for v in ["id","form_year","conm","excntry","mcap_rank"] if v in names]].drop_duplicates(["id","form_year"])
                    c["form_year"]=np.where(c.eom.dt.month.eq(12),c.eom.dt.year,c.eom.dt.year-1)
                    c=c.merge(names,on=["id","form_year"],how="left",validate="many_to_one")
                    source(membership,"original company/listing membership labels")
                c.insert(0,"dataset",dataset);c.insert(1,"strategy",strategy)
                extreme_costs.extend(c.to_dict("records"))
            # Stored inherited weights checked independently against prior saved target weights.
            prior = w[["id", "eom", "w"]].merge(panel[["id", "eom", "tr_ld1"]], on=["id", "eom"], how="left")
            prior = prior.merge(wealth[["eom", "mu_ld1"]], on="eom", how="left")
            prior["expected_start"] = prior.w * (1 + prior.tr_ld1) / (1 + prior.mu_ld1)
            prior["eom"] += MONTH
            joined = w.merge(prior[["id", "eom", "expected_start"]], on=["id", "eom"], how="left")
            later = joined.eom.gt(w.eom.min())
            compare = joined.loc[later].copy()
            # New entrants have zero start. Missing preceding held return must not be filled.
            compare["expected_start"] = compare.expected_start.fillna(0.0)
            delta = float((compare.w_start - compare.expected_start).abs().max()) if len(compare) else np.nan
            finding(dataset, strategy, "inherited_weight_drift", "PASS" if np.isfinite(delta) and delta < TOLERANCE else "FAIL", f"max absolute error excluding first saved month={delta}")
            for eom, g in w[w.w.ne(0)].groupby("eom", sort=True):
                if eom not in cov_cache:
                    continue
                ids, sigma = cov_cache[eom]
                if not set(g.id).issubset(set(ids)):
                    finding(dataset,strategy,"forecast_risk_coverage","FAIL",f"held IDs lack covariance at {eom}")
                    continue
                a = pd.Series(g.w.to_numpy(), index=g.id).reindex(ids, fill_value=0.0).to_numpy()
                variance = float(a @ sigma @ a)
                observed = saved.loc[saved.eom.eq(eom)].iloc[0]
                risk_rows.append(dict(dataset=dataset,strategy=strategy,eom=eom,eom_ret=observed.eom_ret,
                    forecast_variance_monthly=variance,forecast_volatility_annual=np.sqrt(max(variance,0))*np.sqrt(12),
                    gross_excess_return=observed.r,raw_squared_return_to_forecast_variance=(observed.r ** 2 / variance if variance>0 else np.nan)))

    # Include input assumptions even for prepared 250-stock datasets whose model
    # outputs are not yet available. Do not scan earlier archives as new inputs.
    for universe in ["us_large", "us_mid", "eu_large"]:
        for folder, label in [("with_zero_returns", "base"), ("universe_250", "u250")]:
            processed = root / "keats_pipeline/data/processed" / universe / folder
            ledger = processed / "history_exclusions.csv"
            history_coverage.append(dict(dataset=f"{universe}/{label}", processed_path=str(processed),
                prepared_panel_exists=(processed / "panel.parquet").is_file(),
                applied_history_ledger_exists=ledger.is_file(),
                evaluated_primary_portfolio=processed.resolve() in audited_processed,
                status="Saved applied-exclusion ledger available" if ledger.is_file() else
                    "SKIPPED: no prepared panel or applied ledger; source gap evidence is reported separately",
                interpretation="A missing dataset/ledger is not treated as an empty completed exclusion policy."))
            if processed.resolve() in audited_processed:
                continue
            for name, target in [("history_exclusions.csv", exclusions), ("zero_return_imputations.csv", zeroes)]:
                path = processed / name
                if not path.is_file():
                    continue
                source(path, "prepared-only input assumption/exclusion ledger; no portfolio evaluation available")
                try:
                    d = pd.read_csv(path)
                except pd.errors.EmptyDataError:
                    d = pd.DataFrame()
                if len(d):
                    d.insert(0,"dataset",f"{universe}/{label}")
                    d["scope"]="prepared inputs only; no saved primary portfolio"
                    target.extend(d.to_dict("records"))
    unavailable_history = history_input_status(root, exclusions)
    for region in ["US", "EU"]:
        source(root / f"keats_pipeline/data/inputs/unavailable_lag_history_{region}.csv",
               "documented unavailable source history; distinct from applied portfolio exclusions")
    # Inspect saved chosen-hyperparameter timestamps without recomputing choices.
    chosen_paths = set()
    for record in inventory.to_dict("records"):
        p = _path(root,record.get("pf_path"))
        if p:
            candidate = p.parent/"chosen.csv"
            if candidate.is_file():
                chosen_paths.add(candidate)
        processed = _path(root,record.get("processed_path"))
        if processed and (processed/"return_model/chosen.csv").is_file():
            chosen_paths.add(processed/"return_model/chosen.csv")
    for path in sorted(chosen_paths):
        d = pd.read_csv(path)
        source(path,"saved yearly hyperparameter choices")
        if "year" not in d:
            continue
        for row in d.to_dict("records"):
            year = int(row["year"])
            selection = pd.Timestamp(row["eom_ret"]) if "eom_ret" in row else pd.Timestamp(year-1,12,31)
            timing_rows.append(dict(path=str(path.relative_to(root)),year=year,
                selection_last_return_month=selection,first_predicted_return_month=pd.Timestamp(year,1,31),
                selected_before_predicted_year=selection < pd.Timestamp(year,1,1),
                timestamp_evidence="saved chosen.csv" if "eom_ret" in row else "code-implied expanding return-model cutoff; fit timestamp not retained"))
    for source_path in [root/"keats_pipeline/configs/keats_base.yaml", root/"keats_pipeline/settings.json",root/"keats_pipeline/data/inputs/manifest.json"]:
        source(source_path,"common configuration/input manifest")
    for filename in ["models/return_model.py","portfolio/static_ml.py","portfolio/portfolio_ml.py","portfolio/accounting.py","covariance.py"]:
        source(root/"keats_pipeline/src/jkmp"/filename,"timing/accounting implementation inspected read-only")
    source(root/"keats_pipeline/workflow.py","data preparation and cost units inspected read-only")
    source(root/"keats_pipeline/run_all.py","saved experiment runner overrides inspected read-only")

    source_frame = _records_frame(sources,destination/"audited_source_hashes.csv")
    findings_frame = _records_frame(findings,destination/"data_quality_checks.csv")
    _records_frame(configs,destination/"effective_configurations.csv")
    _records_frame(panel_rows,destination/"prepared_panel_coverage.csv")
    _records_frame(exclusions,destination/"all_history_exclusions.csv",["dataset","stock_id","formation_month_end","return_month_end","missing_history_month_end","reason"])
    _records_frame(history_coverage,destination/"history_exclusion_coverage.csv")
    _records_frame(unavailable_history,destination/"all_unavailable_lag_history_inputs.csv")
    _records_frame(zeroes,destination/"all_zero_return_assumptions.csv",["dataset","id","eom","ret_ld1_original","ret_ld1","tr_ld1"])
    _records_frame(accounting_rows,destination/"independent_accounting.csv")
    _records_frame(cov_rows,destination/"covariance_validity_and_stability.csv")
    _records_frame(diagnostics,destination/"saved_covariance_diagnostics.csv")
    _records_frame(extreme_costs,destination/"largest_stock_month_cost_contributors.csv")
    _records_frame(infeasible_months,destination/"non_executable_saved_months.csv")
    risk = _records_frame(risk_rows,destination/"forecast_versus_realised_risk_monthly.csv")
    if len(risk):
        calibration=[]
        for (dataset,strategy), g in risk.groupby(["dataset","strategy"]):
            realised = float(g.gross_excess_return.std(ddof=1)*np.sqrt(12))
            forecast = float(np.sqrt(g.forecast_variance_monthly.mean()*12))
            calibration.append(dict(dataset=dataset,strategy=strategy,n=len(g),
                realised_gross_volatility_annual=realised,rms_forecast_volatility_annual=forecast,
                realised_to_forecast_volatility=realised/forecast if forecast else np.nan,
                raw_barra_bias_sqrt_mean_return_squared_over_forecast_variance=float(np.sqrt(g.raw_squared_return_to_forecast_variance.mean())),
                caveat="Changing portfolio weights; descriptive calibration, no independent model refit"))
        _records_frame(calibration,destination/"forecast_versus_realised_risk_summary.csv")
    timing = _records_frame(timing_rows,destination/"hyperparameter_timing.csv")
    if len(timing):
        failed_timing=int((~timing.selected_before_predicted_year).sum())
        finding("ALL","ALL","saved_hyperparameter_year_separation","PASS" if failed_timing==0 else "FAIL",f"failed choices={failed_timing}; choices={len(timing)}")
    finding("ALL","ALL","point_in_time_vendor_availability","SKIPPED","Saved vendor research snapshots do not retain publication vintages; timestamp consistency cannot prove absence of revised fundamentals or survivorship bias")
    finding("ALL","ALL","fresh_holdout","SKIPPED","2021-2025 is within the declared 2020-2025 test; no separate unseen holdout")
    finding("ALL","ALL","lookahead_refit_counterfactual","SKIPPED","No retraining authorized; saved predictions and code/date checks only, no future-return perturbation rerun")
    finding("ALL","ALL","alternative_covariance_reoptimisation","SKIPPED","One saved factor covariance specification; no alternative covariance fits/reoptimisation authorized")
    findings_frame = _records_frame(findings,destination/"data_quality_checks.csv")
    git = subprocess.run(["git","status","--short"],cwd=root,capture_output=True,text=True,check=False)
    (destination/"working_tree_at_audit.txt").write_text(git.stdout,encoding="utf-8")
    counts = findings_frame.status.value_counts().to_dict()
    summary=dict(check_counts={str(k):int(v) for k,v in counts.items()},source_files_hashed=len(source_frame),
        history_exclusion_records=len(exclusions),zero_return_assumption_records=len(zeroes),
        unavailable_lag_history_input_records=len(unavailable_history),
        unavailable_source_gaps_without_applied_exclusion=sum(r["applied_exclusion_records"] == 0 for r in unavailable_history),
        datasets=sorted(inventory.dataset.unique().tolist()),accounting_reconstructed=len(accounting_rows),
        covariance_matrices_inspected=len(cov_rows))
    summary["non_executable_saved_months"]=len(infeasible_months)
    (destination/"audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (destination/"exclusions_assumptions_and_changes.md").write_text(_NOTES,encoding="utf-8")
    (destination/"hyperparameter_dictionary.csv").write_text(_HYPERS,encoding="utf-8")
    return summary


_NOTES = """# Exclusions, assumptions, changes and limits

This comparison reads saved outputs. It does not fit, tune, retrain, change or
repair a model. Its new audit files are separate from every source result.
The working-tree file lists the state at audit time, including the newly added
comparison code. Existing tracked modifications in keats_pipeline/README.md,
apply_history_policy.py, configs/us_mid/base.yaml, scripts/13_breadth.py,
src/jkmp/portfolio/recursion.py and tests/test_execution_safety.py predate this
comparison and must not be represented as comparison changes. Source checksums
record the exact saved predictions, covariance pieces, configs and weights read.

## Existing input assumptions and exclusions

- Missing next-month held-stock excess returns were explicitly set to zero in
  the pre-existing workflow. Their total return equals the next-month USD
  risk-free return. They are assumptions, not observations. Every available
  occurrence is copied to `all_zero_return_assumptions.csv`; the prepared-panel
  summary separates complete history from the evaluated test months. Dataset
  variants can share the same economic event; rows must not be summed as unique
  source events across variants. Unfilled broad-training missing values are not
  fabricated by this evaluation.
- Unavailable required historical signal stock/months were excluded in the
  pre-existing workflow rather than assigned invented history. Exact IDs,
  missing history and formation/return dates are copied to
  `all_history_exclusions.csv`. No further security exclusion is imposed here.
  Applied-ledger coverage is separately saved in `history_exclusion_coverage.csv`.
  The raw source history files document three missing security/months: US IDs
  88818 and 91135, and European ID 320230501, all December 2010. At the audited
  snapshot, the saved US-mid250 ledger contains two applied December 2011
  exclusions; EU250 has not been prepared and has no applied exclusion ledger.
  Therefore the applied count is two, not three. The documented European source
  gap is retained in `all_unavailable_lag_history_inputs.csv`, with no invented
  applied portfolio event. Earlier README policy descriptions are not evidence
  that an unrun preparation stage has completed.
- Features were previously average-percentile-ranked over the broad screened
  same-region population; missing feature ranks were assigned 0.5. All 50
  characteristics and the saved memberships are retained. Supplementary lagged
  history is used only as lagged information, not additional fitted targets.
- US common-stock screens include NYSE and Nasdaq (CRSP 1 and 3), share codes
  10/11/12, primary/common/observed/main-exchange flags, positive market cap and
  dollar volume, industry/history and characteristic coverage. US mid-cap is
  centred on market-cap ranks 461-540. European large-cap spans developed
  Europe, not UK-only. Buffers and exits make monthly counts vary.
- Equity returns and investor wealth are USD. European daily risk inputs use
  Compustat Global price/dividend/FX reconstruction and JKP security matching;
  US risk inputs use CRSP CIZ total returns with its delisting treatment.
  Original provenance and reconciliation files remain authoritative. Daily risk
  coverage/reconciliation does not prove every monthly delisting payoff is
  correct. Missing held forward returns remain the explicit zero assumption.
- A stock disappearing from the panel is treated by original accounting as
  settled through its reported terminal return, with no additional sale. This
  assumption is not independently verified against terminal cash proceeds.
  Exit cost rows are retained only where an exiting security still has a panel
  row and saved Lambda. Shorting fees, borrow constraints, fees, taxes and
  unmodelled execution costs are absent.

## Actual timing and estimation

- All date labels refer to earned returns: formation December 2019 earns
  January 2020. Declared portfolio validation is 2015-2019; reported test/OOS
  is 2020-2025. Historical signals/risk need earlier warmup.
- Return-model fitting targets begin January 2005. It refits yearly. For
  predicted return year y, its five-year tuning validation is y-5 through y-1;
  the earlier expanding rows train the candidate coefficients, and selected
  coefficients refit through December y-1. Thus the 2015 forecasts initially
  train on 2005-2009, validate 2010-2014 and refit on 2005-2014. For 2020 this
  becomes candidate training 2005-2014, validation 2015-2019, refit 2005-2019.
  A simplistic fixed training 2005-2014 label does not describe all annual fits.
- Static-ML* and Portfolio-ML select each following year's hyperparameters
  from cumulative saved realised validation utility ending in the previous
  December. Earlier test observations can enter later walk-forward fits and
  selections as they become available. This is sequential OOS, not a frozen
  untouched holdout. Saved date/year separation is audited; exact historical
  model execution timestamps and vintage vendor data are not retained.
- Covariance uses prior/current formation-date information, factor daily
  return rows dated no later than eom, a 2,520-trading-day window, EWMA
  correlation half-life 378 days and variance/idiosyncratic half-life 126 days.
  Daily estimates are multiplied by 21 for monthly units. Missing idiosyncratic
  volatility is filled by existing size-group/overall medians. Exact saved
  positive definiteness and conditioning are checked on each evaluated month;
  this does not establish that the forecast is well calibrated.
- No additional unseen holdout follows December 2025. No post-2025 result or
  independently verified point-in-time publication vintage is available.

## Accounting conventions and independent checks

- r is saved monthly gross EXCESS return; tc is a positive monthly fraction
  of investor wealth. Net excess is r-tc. Turnover is sum(abs(w-w_start)),
  the full L1 trade amount, not conventional half-turnover. Gross leverage is
  sum(abs(w)); shorting is sum(max(-w,0)). Trading cost is
  0.5 * wealth_t * sum(Lambda_i,t * (w_i,t-w_start_i,t)^2).
- Funded-growth diagnostics add the saved next-month risk-free return:
  gross_total=r+RF and net_total=r-tc+RF. The audit flags growth <=0 using those
  total returns. A separate synthetic excess-index flag r-tc<=-1 does not by
  itself establish funded bankruptcy and must not be counted interchangeably.
- Independent reconstruction checks gross return, cost, turnover, leverage and
  shorting against saved weights and exact saved inputs to tolerance 1e-10.
  Inherited weights are checked as prior target*(1+stock total return)/
  (1+screened regional market total return). The first saved month is excluded
  from that drift comparison because its prior state may predate saved weights.
- Fixed Static-ML returns are saved but its fixed-parameter weights are not
  retained; Static-ML* has separate saved weights. Fixed Static-ML accounting,
  concentration and weight comparisons are marked unavailable, not inferred
  from Static-ML* weights.
- The screened regional value-weighted market is the wealth growth proxy,
  anchored to USD10bn at end-2019. It is not an external investable index.
- Paper Table 2 volatility, net Sharpe denominator and variance penalty use
  GROSS return standard deviation/variance. Annual means/costs/utility use 12;
  volatility and Sharpe use sqrt(12). Sample variance uses n-1. The Table 3
  paper utility-flow centring has a finite-sample n versus n-1 distinction,
  which the statistical module must identify separately.

## Differences and inference limits relative to Jensen et al.

These small buffered 80/250-stock US-size and European universes, 50 features,
2005-start fitting, five-year return-model tuning window, 2020-2025 test and
USD10bn end-2019 wealth path differ from the paper sample and investment
universe. Raw-characteristic linear Portfolio-ML, benchmark-relative active
Portfolio-ML, half-life, regime and breadth experiments are extensions. RFF
complexity grids and computational choices may differ; effective configs
and hyperparameter dictionary record their actual meanings. Findings are
descriptive tests of this saved implementation, not a numerical replication
of the paper's levels. Across-region/size gaps confound universe, liquidity,
signals, risk estimates and return paths; they cannot identify causal model
quality. Multiple variants sharing inputs are not independent replications.

Missing outputs, unavailable seed return paths, multi-horizon forecasts,
alternative covariance fits, original fixed Static-ML weights and fresh
holdout tests must be SKIPPED rather than simulated or filled. No result
selection using this report should be represented as new untouched OOS.
"""

_HYPERS = '''component,parameter,definition,unit,selection
return_model,p,Number of cosine and sine random Fourier features,feature count,Annual minimum historical validation MSE
return_model,g,Variance multiplier of Gaussian RFF frequencies; frequencies=sqrt(g)*Z,dimensionless,Annual minimum historical validation MSE
return_model,lam,Ridge regularisation penalty lambda (not trading-cost Lambda),dimensionless under implementation scaling,Annual minimum historical validation MSE
return_model,val_years,Length of rolling historical tuning window before refit,years,Fixed config (five)
static_ml,k,Trading-cost multiplier k=1/phi in wealth*k*Lambda,dimensionless,Static=1; Static-star chosen by cumulative utility
static_ml,u,Forecast shrinkage/scaling multiplier u*mu,dimensionless,Static=1; Static-star chosen by cumulative utility
static_ml,g,Variance inflation adding g*diag(gamma*Sigma) to gamma*Sigma,dimensionless,Static=0; Static-star chosen by cumulative utility
portfolio_ml,p,Number of transformed signal columns/RFF complexity,feature count,Annual cumulative utility selection
portfolio_ml,g,RFF frequency variance multiplier; raw linear signal variant has no RFF transform,dimensionless,Annual cumulative utility selection
portfolio_ml,l,Ridge penalty on portfolio-aim coefficients beta,dimensionless under implementation scaling,Annual cumulative utility selection
portfolio_ml,h,Exponential training observation half-life; infinity means no decay,years,Only searched where half-life variant saved
portfolio_ml,add_constant,Include constant characteristic signal,boolean,Fixed config/runner override
portfolio_ml,benchmark_relative,Learn/trade active positions around universe VW reference,boolean,Fixed config/runner override
investor,gamma,Mean-variance risk aversion for absolute-return objective,reciprocal return,Fixed baseline 10 or saved grid
investor,gamma_te,Mean-variance risk aversion for active tracking-risk objective,reciprocal return,Historical TE calibration where available
investor,wealth,Dollar investor wealth anchored at configured end date,USD,Fixed baseline 10000000000 or saved AUM grid
trading_costs,price_impact,Scalar used to build diagonal cost coefficient from dollar volume,dimensionless,Fixed config
portfolio,charge_exit_tc,Charge liquidation cost when exiting ID still has contemporaneous panel data,boolean,Fixed config
'''
