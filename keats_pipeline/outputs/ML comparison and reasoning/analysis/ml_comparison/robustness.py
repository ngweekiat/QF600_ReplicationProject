"""Evaluation-only stress tests of saved Keats portfolios.

No function in this module fits a model, calls a portfolio optimiser, downloads
data, or writes into the Keats tree.  Fixed-weight re-costing and already
optimised grid results deliberately have different output files.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

COST_FACTORS = (0.0, 0.5, 1.0, 2.0, 3.0)
GAMMAS = (1.0, 5.0, 10.0, 20.0, 100.0)
AUM_LEVELS = (1e9, 5e9, 10e9, 25e9, 50e9, 100e9, 250e9)


def saved_stats(frame: pd.DataFrame, gamma: float = 10.0,
                cost_factor: float = 1.0) -> dict:
    """Paper moments: mean*12, sd(gross)*sqrt(12), gross-variance utility.

    Portfolio turnover is the full sum of absolute traded weights, never halved.
    SR_net_paper divides mean net return by *gross* return standard deviation.
    """
    if len(frame) < 2:
        return {"n_months": len(frame)}
    r = np.asarray(frame.r, dtype=float)
    tc = np.asarray(frame.tc, dtype=float) * cost_factor
    if not np.isfinite(r).all() or not np.isfinite(tc).all():
        raise ValueError("Nonfinite saved return or transaction cost")
    vol = float(np.std(r, ddof=1))
    return {"n_months": len(r), "start": str(pd.Timestamp(frame.eom_ret.min()).date()),
            "end": str(pd.Timestamp(frame.eom_ret.max()).date()),
            "R": 12 * r.mean(), "Vol": np.sqrt(12) * vol,
            "TC": 12 * tc.mean(), "R_TC": 12 * (r - tc).mean(),
            "net_excess_leq_minus_one_months": int(np.sum((r - tc) <= -1.0)),
            "SR_net_paper": np.sqrt(12) * (r - tc).mean() / vol if vol > 0 else np.nan,
            "Utility": 12 * ((r - tc).mean() - gamma * 0.5 * vol**2),
            "Turnover": frame.turnover.mean() if "turnover" in frame else np.nan,
            "Lev": frame.inv.mean() if "inv" in frame else np.nan}


def cost_sensitivity(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, strategy), g in monthly.groupby(["dataset", "strategy"], sort=True):
        for factor in COST_FACTORS:
            rows.append({"dataset": dataset, "strategy": strategy,
                         "cost_factor": factor, "evaluation_gamma": 10.0,
                         "portfolio_policy": "fixed_saved_weights",
                         **saved_stats(g, cost_factor=factor)})
    out = pd.DataFrame(rows)
    if len(out):
        out["utility_rank"] = out.groupby(["dataset", "cost_factor"]).Utility.rank(ascending=False, method="min")
        out["SR_rank"] = out.groupby(["dataset", "cost_factor"]).SR_net_paper.rank(ascending=False, method="min")
    return out


def gamma_sensitivity(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = [{"dataset": dataset, "strategy": strategy, "evaluation_gamma": gamma,
             "portfolio_policy": "fixed_saved_weights_evaluated_at_new_gamma",
             **saved_stats(g, gamma=gamma)}
            for (dataset, strategy), g in monthly.groupby(["dataset", "strategy"], sort=True)
            for gamma in GAMMAS]
    out = pd.DataFrame(rows)
    if len(out):
        out["utility_rank"] = out.groupby(["dataset", "evaluation_gamma"]).Utility.rank(ascending=False, method="min")
    return out


def fixed_weight_aum(monthly: pd.DataFrame, base_wealth: dict[str, float]) -> pd.DataFrame:
    """Quadratic dollar impact / wealth is linear in wealth at fixed weights.

    These are hypothetical scenarios: the saved weight policy and return path
    remain fixed.  They are not the implementable efficient frontier.
    """
    rows = []
    for (dataset, strategy), g in monthly.groupby(["dataset", "strategy"], sort=True):
        anchor = base_wealth[dataset]
        for wealth in AUM_LEVELS:
            rows.append({"dataset": dataset, "strategy": strategy,
                         "aum_anchor_usd": wealth, "saved_aum_anchor_usd": anchor,
                         "cost_scale": wealth / anchor,
                         "portfolio_policy": "hypothetical_fixed_weights_linear_impact_recosting",
                         **saved_stats(g, cost_factor=wealth / anchor)})
    return pd.DataFrame(rows)


def _csv(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


def _test_window(monthly: pd.DataFrame, dataset: str) -> pd.DatetimeIndex:
    rows = monthly.loc[monthly.dataset.eq(dataset), "eom_ret"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(rows).unique()))


def _load_cfg(row):
    from jkmp.config import load_config
    return load_config(str(row.config_path))


def saved_input_catalog(root: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    """Independent input availability; does not add a strategy or portfolio row.

    Prepared predictions can exist while covariance/portfolio stages have not
    completed. Such inputs support forecast diagnostics and saved model-choice
    stability, but never performance rankings or hypothetical portfolio returns.
    """
    from jkmp.config import load_config
    universes = sorted(set(inventory.universe)) if "universe" in inventory else sorted(
        {str(value).split('/')[0] for value in inventory.dataset})
    evaluated = set(inventory.dataset)
    rows = []
    for universe in universes:
        for variant in ("base", "universe_250"):
            config_path = root / "keats_pipeline/configs" / universe / f"{variant}.yaml"
            if not config_path.is_file():
                continue
            cfg = load_config(str(config_path))
            processed = cfg.paths.processed
            predictions, panel = processed / "predictions.parquet", processed / "panel.parquet"
            # A panel alone cannot support forecast diagnostics, and missing
            # forecasts must never be manufactured from another universe.
            if not predictions.is_file() or not panel.is_file():
                continue
            dataset = f"{universe}/{cfg.paths.outputs.name}"
            scope = "saved_primary_portfolio_inputs" if dataset in evaluated else "input_only_no_saved_portfolio"
            rows.append(dict(dataset=dataset, universe=universe, variant=variant,
                processed_path=str(processed), config_path=str(config_path),
                predictions_path=str(predictions), panel_path=str(panel), scope=scope,
                intended_test_start=str(cfg.dates.test_start), intended_test_end=str(cfg.dates.test_end),
                saved_return_model_choices=(processed / "return_model/chosen.csv").is_file(),
                saved_covariance=(processed / "covariance/cov_loadings.parquet").is_file(),
                interpretation="Input diagnostics only when no primary portfolio; no weights or returns inferred."))
    return pd.DataFrame(rows, columns=["dataset", "universe", "variant", "processed_path", "config_path",
        "predictions_path", "panel_path", "scope", "intended_test_start", "intended_test_end",
        "saved_return_model_choices", "saved_covariance", "interpretation"])


def existing_optimised_grids(root: Path, destination: Path, monthly: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Read complete saved grid/capacity point files, never trust summary alone."""
    rows, skipped, coverage = [], [], []
    for universe in ("us_large", "us_mid", "eu_large"):
        base = root / "keats_pipeline/outputs" / universe / "base/grids"
        dataset = universe + "/base"
        window = _test_window(monthly, dataset)
        if len(window) == 0:
            continue
        from jkmp.config import load_config
        cfg = load_config(str(root / "keats_pipeline/configs" / universe / "base.yaml"))
        for gamma in cfg.investor.gamma_grid:
            for wealth in cfg.investor.wealth_grid:
                point = base / f"gamma{gamma:g}_wealth{wealth / 1e9:g}B"
                for filename in ("pf_static_ml.parquet", "pf_portfolio_ml_rff.parquet", "pf_portfolio_ml_raw.parquet"):
                    file = point / filename
                    coverage.append({"dataset": dataset, "optimised_gamma": gamma,
                                     "aum_anchor_usd": wealth, "expected_file": str(file.relative_to(root)),
                                     "saved_file_exists": file.is_file()})
                    if not file.is_file():
                        skipped.append({"test": "saved_optimised_grid", "dataset": dataset,
                                        "input": str(file.relative_to(root)), "status": "SKIPPED",
                                        "reason": "Expected optimised point file not saved; no reoptimisation performed"})
        for point in sorted(base.glob("gamma*_wealth*B")):
            match = re.fullmatch(r"gamma([\d.e+-]+)_wealth([\d.e+-]+)B", point.name)
            if not match:
                continue
            gamma, wealth = float(match[1]), float(match[2]) * 1e9
            for file in sorted(point.glob("pf_*.parquet")):
                data = pd.read_parquet(file)
                for strategy, g in data.groupby("type"):
                    g = g.loc[pd.to_datetime(g.eom_ret).isin(window)]
                    valid = not g.eom_ret.duplicated().any() and set(pd.to_datetime(g.eom_ret)) == set(window)
                    if not valid:
                        skipped.append({"test": "saved_optimised_grid", "dataset": dataset,
                                        "input": str(file), "status": "SKIPPED",
                                        "reason": "Saved point does not cover every common test month exactly once"})
                        continue
                    rows.append({"dataset": dataset, "strategy": strategy,
                                 "optimised_gamma": gamma, "aum_anchor_usd": wealth,
                                 "optimised_gamma_te": np.nan, "evaluation_gamma": gamma,
                                 "optimisation_objective": "absolute_mean_variance",
                                 "portfolio_policy": "previously_optimised_saved_grid_point",
                                 "input": str(file.relative_to(root)), **saved_stats(g, gamma)})
    # Capacity results are saved active strategies rather than baseline gamma grids.
    for universe in ("us_large", "us_mid", "eu_large"):
        for variant in ("base", "u250", "u250_hl"):
            dataset = f"{universe}/{variant}"
            window = _test_window(monthly, dataset)
            if len(window) == 0:
                continue
            capacity_files = sorted((root / "keats_pipeline/outputs" / universe / variant / "active/capacity").glob("gte*_w*B/pf.parquet"))
            if not capacity_files:
                skipped.append({"test": "saved_active_capacity", "dataset": dataset, "status": "SKIPPED",
                                "reason": "No saved optimised active-capacity point; fixed-weight scenarios are hypothetical only"})
            for file in capacity_files:
                match = re.fullmatch(r"gte([\d.e+-]+)_w([\d.e+-]+)B", file.parent.name)
                if not match:
                    continue
                gamma, wealth = float(match[1]), float(match[2]) * 1e9
                data = pd.read_parquet(file)
                for strategy, g in data.groupby("type"):
                    g = g.loc[pd.to_datetime(g.eom_ret).isin(window)]
                    if g.eom_ret.duplicated().any() or set(pd.to_datetime(g.eom_ret)) != set(window):
                        skipped.append({"test": "saved_active_capacity", "dataset": dataset,
                                        "input": str(file), "status": "SKIPPED", "reason": "Incomplete common month coverage"})
                        continue
                    rows.append({"dataset": dataset, "strategy": f"{strategy} (active capacity)",
                                 "optimised_gamma": np.nan, "optimised_gamma_te": gamma,
                                 "evaluation_gamma": 10.0, "aum_anchor_usd": wealth,
                                 "optimisation_objective": "benchmark_relative_mean_variance",
                                 "portfolio_policy": "previously_optimised_active_capacity_point",
                                 "input": str(file.relative_to(root)), **saved_stats(g, 10.0)})
    result = pd.DataFrame(rows)
    _csv(result, destination / "saved_optimised_gamma_aum.csv")
    _csv(pd.DataFrame(coverage), destination / "saved_grid_file_coverage.csv")
    return result, skipped


def _portfolio_diagnostics(weights: pd.DataFrame, panel: pd.DataFrame, wealth: pd.DataFrame,
                           strategy: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Holdings and additive P&L attribution, not independently tradable subportfolios."""
    w = weights.merge(panel, on=["id", "eom"], how="left", validate="many_to_one")
    w = w.merge(wealth[["eom", "wealth"]], on="eom", how="left", validate="many_to_one")
    w["abs_weight"] = w.w.abs()
    w["abs_trade"] = (w.w - w.w_start).abs()
    w["gross_contribution"] = np.where(w.exit, 0.0, w.w * w.ret_ld1)
    w["tc_contribution"] = 0.5 * w.wealth * w["lambda"] * (w.w - w.w_start)**2
    meta = {"dataset": dataset, "strategy": strategy}
    records = []
    for date, group in w.groupby("eom", sort=True):
        held = group.loc[~group.exit & group.w.ne(0)]
        gross = float(held.abs_weight.sum())
        normal = held.abs_weight / gross if gross else held.abs_weight * np.nan
        positive = float(held.loc[held.w.gt(0), "w"].sum())
        negative = float(-held.loc[held.w.lt(0), "w"].sum())
        records.append({**meta, "eom": date, "eom_ret": date + pd.offsets.MonthEnd(1),
                        "n_nonzero": len(held), "long_count": int(held.w.gt(0).sum()),
                        "short_count": int(held.w.lt(0).sum()), "exit_count": int(group.exit.sum()),
                        "gross_leverage": gross, "net_investment": positive - negative,
                        "long_weight": positive, "short_weight": negative,
                        "abs_weight_hhi": float((normal**2).sum()) if gross else np.nan,
                        "largest_absolute_weight": held.abs_weight.max() if len(held) else 0,
                        "top10_share_of_gross": held.abs_weight.nlargest(10).sum() / gross if gross else np.nan,
                        "abs_trade_turnover": group.abs_trade.sum(),
                        "december_formation": bool(date.month == 12),
                        "missing_size_or_liquidity_rows": int(group[["me", "dolvol"]].isna().any(axis=1).sum())})
    groups = []
    for column in ("size_bucket", "liquidity_bucket"):
        group = w.groupby(["eom", column], dropna=False, observed=True).agg(
            gross_exposure=("abs_weight", "sum"), signed_exposure=("w", "sum"),
            turnover=("abs_trade", "sum"), gross_return_contribution=("gross_contribution", "sum"),
            tc_contribution=("tc_contribution", "sum"), n_rows=("id", "size"),
            median_raw_me=("me", "median"), median_raw_dollar_volume=("dolvol", "median")).reset_index()
        group = group.rename(columns={column: "bucket"})
        group["group_variable"] = column
        group["dataset"], group["strategy"] = dataset, strategy
        group["eom_ret"] = group.eom + pd.offsets.MonthEnd(1)
        groups.append(group)
    sector = w.groupby(["eom", "ff12"], dropna=False).agg(
        gross_exposure=("abs_weight", "sum"), signed_exposure=("w", "sum"),
        turnover=("abs_trade", "sum"), gross_return_contribution=("gross_contribution", "sum"),
        tc_contribution=("tc_contribution", "sum")).reset_index()
    sector["dataset"], sector["strategy"] = dataset, strategy
    sector["eom_ret"] = sector.eom + pd.offsets.MonthEnd(1)
    return pd.DataFrame(records), pd.concat(groups, ignore_index=True), sector


def holdings_diagnostics(inventory: pd.DataFrame, monthly: pd.DataFrame, destination: Path) -> tuple[pd.DataFrame, list]:
    holdings, groups, sectors, skipped = [], [], [], []
    columns = ["id", "eom", "in_universe", "me", "dolvol", "lambda", "ret_ld1", "ff12"]
    for (dataset, processed), items in inventory.groupby(["dataset", "processed_path"], sort=True):
        window = _test_window(monthly, dataset)
        formation = window - pd.offsets.MonthEnd(1)
        panel_file, wealth_file = Path(processed) / "panel.parquet", Path(processed) / "wealth.parquet"
        if not panel_file.exists() or not wealth_file.exists():
            skipped.append({"test": "holdings_diagnostics", "dataset": dataset, "status": "SKIPPED",
                            "reason": "Prepared panel/wealth unavailable"})
            continue
        panel = pd.read_parquet(panel_file, columns=columns)
        panel = panel.loc[panel.eom.isin(formation)].copy()
        # Quantiles are within the saved investable membership, not cross-dataset.
        for raw, bucket in (("me", "size_bucket"), ("dolvol", "liquidity_bucket")):
            member_rank = panel.loc[panel.in_universe, ["id", "eom", raw]].copy()
            ranks = member_rank.groupby("eom")[raw].rank(pct=True, method="average")
            member_rank[bucket] = np.select([ranks.le(1 / 3), ranks.le(2 / 3)],
                                            ["low", "middle"], default="high")
            member_rank.loc[member_rank[raw].isna(), bucket] = "missing"
            panel = panel.merge(member_rank[["id", "eom", bucket]], on=["id", "eom"], how="left", validate="one_to_one")
            panel[bucket] = panel[bucket].fillna("outside_current_membership")
        wealth = pd.read_parquet(wealth_file, columns=["eom", "wealth"])
        cache = {}
        for item in items.itertuples():
            file = Path(str(item.weights_path))
            if not file.is_file():
                skipped.append({"test": "holdings_diagnostics", "dataset": dataset, "strategy": item.strategy,
                                "status": "SKIPPED", "reason": "Saved weights unavailable"})
                continue
            if str(file) not in cache:
                cache[str(file)] = pd.read_parquet(file)
            weights = cache[str(file)]
            source_type = getattr(item, "source_type", item.strategy)
            if "type" in weights:
                weights = weights.loc[weights.type.eq(source_type)]
            weights = weights.loc[weights.eom.isin(formation)]
            if weights.empty:
                skipped.append({"test": "holdings_diagnostics", "dataset": dataset, "strategy": item.strategy,
                                "status": "SKIPPED", "reason": f"Saved weight file has no type={source_type} on common months; no missing strategy weights reconstructed"})
                continue
            h, g, s = _portfolio_diagnostics(weights, panel, wealth, item.strategy, dataset)
            holdings.append(h)
            groups.append(g)
            sectors.append(s)
    h = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()
    g = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    s = pd.concat(sectors, ignore_index=True) if sectors else pd.DataFrame()
    _csv(h, destination / "construction_monthly.csv")
    _csv(g, destination / "size_liquidity_attribution_monthly.csv")
    _csv(s, destination / "industry_exposures_monthly.csv")
    if len(h):
        cols = [c for c in h.select_dtypes(include=["number"]).columns if c not in ("december_formation",)]
        _csv(h.groupby(["dataset", "strategy"])[cols].mean().reset_index(), destination / "construction_summary.csv")
        _csv(h.groupby(["dataset", "strategy", "december_formation"])[["abs_trade_turnover", "exit_count", "n_nonzero"]].agg(["mean", "count"]).reset_index(), destination / "reconstitution_turnover.csv")
    if len(g):
        summary = g.groupby(["dataset", "strategy", "group_variable", "bucket"], dropna=False).agg(
            n_months_present=("eom", "nunique"), gross_exposure_sum=("gross_exposure", "sum"),
            signed_exposure_sum=("signed_exposure", "sum"), turnover_sum=("turnover", "sum"),
            gross_contribution_sum=("gross_return_contribution", "sum"),
            tc_contribution_sum=("tc_contribution", "sum")).reset_index()
        denominators = h.groupby(["dataset", "strategy"]).eom.nunique().rename("n_months")
        summary = summary.merge(denominators, on=["dataset", "strategy"], validate="many_to_one")
        summary["mean_gross_exposure"] = summary.gross_exposure_sum / summary.n_months
        summary["mean_signed_exposure"] = summary.signed_exposure_sum / summary.n_months
        summary["mean_monthly_turnover"] = summary.turnover_sum / summary.n_months
        summary["annual_gross_return_contribution"] = summary.gross_contribution_sum / summary.n_months * 12
        summary["annual_tc_contribution"] = summary.tc_contribution_sum / summary.n_months * 12
        _csv(summary, destination / "size_liquidity_attribution_summary.csv")
    return h, skipped


def prediction_diagnostics(inventory: pd.DataFrame, monthly: pd.DataFrame, destination: Path) -> tuple[pd.DataFrame, list]:
    rows, feature_rows, skipped = [], [], []
    for (dataset, processed), items in inventory.groupby(["dataset", "processed_path"], sort=True):
        predictions_file, panel_file = Path(processed) / "predictions.parquet", Path(processed) / "panel.parquet"
        if not predictions_file.exists():
            skipped.append({"test": "one_month_rank_ic", "dataset": dataset, "status": "SKIPPED",
                            "reason": "Saved predictions unavailable"})
            continue
        cfg = _load_cfg(items.iloc[0])
        scope = str(items.iloc[0].get("scope", "saved_primary_portfolio_inputs"))
        window = _test_window(monthly, dataset)
        if not len(window):
            if scope != "input_only_no_saved_portfolio":
                raise ValueError(f"Missing aligned portfolio calendar for {dataset}; input-only fallback was not authorised by catalog scope")
            # This is a forecast-diagnostic calendar, not a fabricated monthly
            # performance series. Only actually saved prediction rows are used.
            window = pd.date_range(pd.Timestamp(cfg.dates.test_start), pd.Timestamp(cfg.dates.test_end), freq="ME")
        formation = window - pd.offsets.MonthEnd(1)
        predictions = pd.read_parquet(predictions_file)
        predictions = predictions.loc[predictions.eom.isin(formation)]
        columns = ["id", "eom", "in_universe", "ret_ld1", "ret_ld1_imputed_zero", *cfg.features.names]
        panel = pd.read_parquet(panel_file, columns=columns)
        panel = panel.loc[panel.eom.isin(formation)]
        data = predictions.merge(panel, on=["id", "eom"], how="inner", validate="one_to_one")
        for date, group in data.groupby("eom", sort=True):
            for population in ("broad_saved_forecast_population", "held_universe_members"):
                subset = group if population.startswith("broad") else group.loc[group.in_universe]
                for target in ("including_user_zero_assumptions", "observed_targets_only"):
                    sample = subset if target.startswith("including") else subset.loc[~subset.ret_ld1_imputed_zero.fillna(False)]
                    sample = sample.loc[np.isfinite(sample.pred_ld1) & np.isfinite(sample.ret_ld1)]
                    ic = sample.pred_ld1.corr(sample.ret_ld1, method="spearman") if len(sample) > 2 and sample.pred_ld1.nunique() > 1 and sample.ret_ld1.nunique() > 1 else np.nan
                    rows.append({"dataset": dataset, "eom": date, "eom_ret": date + pd.offsets.MonthEnd(1),
                                 "scope": scope,
                                 "model": "saved_return_prediction_model", "horizon_months": 1,
                                 "population": population, "target_policy": target,
                                 "n_stocks": len(sample), "rank_ic": ic,
                                 "rmse": np.sqrt(np.mean((sample.pred_ld1 - sample.ret_ld1)**2)) if len(sample) else np.nan})
            # Association diagnostics, not fitted-model feature importance.
            member = group.loc[group.in_universe & ~group.ret_ld1_imputed_zero.fillna(False)]
            for theme, features in cfg.features.themes.items():
                available = [f for f in features if f in member]
                if not available:
                    continue
                signal = member[available].mean(axis=1)
                valid = signal.notna() & member.ret_ld1.notna()
                ic = signal[valid].corr(member.loc[valid, "ret_ld1"], method="spearman") if valid.sum() > 2 and signal[valid].nunique() > 1 else np.nan
                feature_rows.append({"dataset": dataset, "eom_ret": date + pd.offsets.MonthEnd(1),
                                     "scope": scope,
                                     "theme": theme, "n_stocks": int(valid.sum()),
                                     "theme_rank_association_ic": ic,
                                     "interpretation": "simple_mean_of_saved_rank_features_vs_observed_next_month_return; not_model_importance"})
    ic = pd.DataFrame(rows)
    features = pd.DataFrame(feature_rows)
    _csv(ic, destination / "one_month_prediction_diagnostics.csv")
    if len(ic):
        _csv(ic.groupby(["dataset", "scope", "model", "horizon_months", "population", "target_policy"]).agg(
            n_months=("eom_ret", "nunique"), mean_rank_ic=("rank_ic", "mean"),
            sample_start=("eom_ret", "min"), sample_end=("eom_ret", "max"),
            n_valid_ic_months=("rank_ic", "count"),
            median_rank_ic=("rank_ic", "median"), mean_cross_section_rmse=("rmse", "mean"),
            mean_n_stocks=("n_stocks", "mean")).reset_index(), destination / "prediction_diagnostics_summary.csv")
    _csv(features, destination / "theme_predictive_associations.csv")
    if len(features):
        _csv(features.groupby(["dataset", "scope", "theme"]).agg(n_months=("eom_ret", "nunique"),
             sample_start=("eom_ret", "min"), sample_end=("eom_ret", "max"),
             mean_theme_association_ic=("theme_rank_association_ic", "mean")).reset_index(),
             destination / "theme_predictive_associations_summary.csv")
    return ic, skipped


def saved_importance(root: Path, destination: Path) -> list:
    skipped = []
    for universe in ("us_large", "us_mid", "eu_large"):
        source = root / "keats_pipeline/outputs" / universe / "base/feature_importance"
        found = False
        for name in ("importance.csv", "importance_by_seed.csv"):
            if (source / name).is_file():
                out = destination / "saved_feature_importance" / universe / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, out)
                found = True
        if not found:
            skipped.append({"test": "saved_economic_feature_importance", "dataset": universe + "/base",
                            "status": "SKIPPED", "reason": "No saved feature permutation result; no new counterfactual portfolios generated"})
    return skipped


def hyperparameter_stability(inventory: pd.DataFrame, destination: Path,
                             input_catalog: pd.DataFrame | None = None) -> None:
    picks, rows, fixed = [], [], []
    seen = set()
    primary = inventory.copy()
    primary["scope"] = "saved_primary_portfolio_inputs"
    primary["input_only"] = False
    records = [primary]
    if input_catalog is not None and len(input_catalog):
        extra = input_catalog.loc[input_catalog.scope.eq("input_only_no_saved_portfolio")].copy()
        extra["input_only"] = True
        records.append(extra)
    for item in pd.concat(records, ignore_index=True).itertuples():
        # A static_ml/pf.parquet file contains BOTH fixed Static-ML and tuned
        # Static-ML*. Its sibling chosen.csv belongs ONLY to the latter.
        # Attaching those selections to fixed Static-ML falsely makes that
        # strategy look tuned and understates stability of its fixed rule.
        strategy_name = getattr(item, "strategy", "return_prediction_model")
        source_type = getattr(item, "source_type", strategy_name)
        files = [(Path(item.processed_path) / "return_model/chosen.csv", "return_prediction_model")]
        if item.input_only:
            pass # No portfolio or portfolio-choice file exists for this catalog row.
        elif source_type == "Static-ML":
            fixed.append({"dataset": item.dataset, "strategy": item.strategy,
                          "k": 1.0, "u": 1.0, "g": 0.0,
                          "source": "jkmp.portfolio.static_ml.run_static_ml fixed tuple (1,1,0)",
                          "interpretation": "fixed rule; no annual portfolio hyperparameter selection; inherited validation path begins before test"})
        else:
            files.insert(0, (Path(item.pf_path).parent / "chosen.csv", item.strategy))
        for file, strategy in files:
            key = (item.dataset, strategy, str(file))
            if key in seen or not file.is_file():
                continue
            seen.add(key)
            frame = pd.read_csv(file)
            frame.insert(0, "dataset", item.dataset)
            frame.insert(1, "strategy", strategy)
            frame["source"] = str(file)
            frame["scope"] = item.scope
            picks.append(frame)
            for name in ("g", "p", "l", "lam", "h", "k", "u"):
                if name not in frame:
                    continue
                values = frame[name]
                changes = values.iloc[1:].to_numpy() != values.iloc[:-1].to_numpy()
                rows.append({"dataset": item.dataset, "strategy": strategy, "parameter": name,
                             "scope": item.scope,
                             "n_selections": len(values), "n_distinct": values.nunique(dropna=False),
                             "n_year_to_year_changes": int(changes.sum()),
                             "switch_fraction": float(changes.mean()) if len(changes) else np.nan,
                             "mode": values.mode().iloc[0] if len(values.mode()) else np.nan,
                             "interpretation": "saved_annual_choices; sample_includes_pretest_choices_when_saved"})
    _csv(pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(), destination / "saved_hyperparameter_choices.csv")
    _csv(pd.DataFrame(rows), destination / "hyperparameter_stability.csv")
    _csv(pd.DataFrame(fixed), destination / "fixed_hyperparameter_rules.csv")


def robustness_charts(destination: Path) -> int:
    """Render publication files from the new evaluation tables only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts = destination / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
                         "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})
    count = 0

    def save(fig, name):
        nonlocal count
        fig.savefig(charts / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(charts / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        count += 1

    for source, x, xlabel, name in (
        ("fixed_weight_cost_sensitivity.csv", "cost_factor", "Saved transaction-cost multiplier", "cost_sensitivity"),
        ("fixed_weight_gamma_evaluation.csv", "evaluation_gamma", "Evaluation risk aversion gamma", "gamma_evaluation")):
        data = pd.read_csv(destination / source)
        for dataset, group in data.groupby("dataset", sort=True):
            fig, ax = plt.subplots(figsize=(13.5, 5.5))
            colors = plt.get_cmap("tab20")
            for index, (strategy, g) in enumerate(group.groupby("strategy", sort=True)):
                ax.plot(g[x], g.Utility, marker="o", markersize=3, linewidth=1.6,
                        color=colors(index % 20), label=strategy)
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_yscale("symlog", linthresh=.1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Annual mean-variance utility (decimal; symmetric log)")
            ax.set_title(f"{dataset} | {group.start.iloc[0]} to {group.end.iloc[0]} | fixed saved weights")
            ax.grid(alpha=.2)
            ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
            fig.subplots_adjust(right=.71, bottom=.19)
            fig.text(.01, .02, "Paper gross-variance utility. Cost scenarios: gamma=10. Gamma scenarios change evaluation only.\n"
                     "Extreme costs are retained algebraically; net excess <= -100% invalidates the synthetic excess index, not proof of funded bankruptcy.", fontsize=8)
            save(fig, f"{name}_{dataset.replace('/', '_')}")

    actual_file = destination / "saved_optimised_gamma_aum.csv"
    hypothetical = pd.read_csv(destination / "hypothetical_fixed_weight_aum.csv")
    actual = pd.read_csv(actual_file)
    if len(actual):
        for dataset, fixed in hypothetical.groupby("dataset", sort=True):
            points = actual.loc[actual.dataset.eq(dataset) & actual.optimised_gamma.eq(10)
                                & actual.portfolio_policy.eq("previously_optimised_saved_grid_point")]
            if points.empty:
                continue
            fig, ax = plt.subplots(figsize=(12.5, 5.7))
            names = [s for s in ("Portfolio-ML", "Portfolio-ML (linear)", "Static-ML", "Static-ML*")
                     if s in set(points.strategy) & set(fixed.strategy)]
            for index, strategy in enumerate(names):
                saved = points.loc[points.strategy.eq(strategy)].sort_values("aum_anchor_usd")
                scaled = fixed.loc[fixed.strategy.eq(strategy)].sort_values("aum_anchor_usd")
                color = f"C{index}"
                ax.plot(saved.aum_anchor_usd / 1e9, saved.Utility, "o-", color=color,
                        label=f"{strategy}: saved optimised points")
                ax.plot(scaled.aum_anchor_usd / 1e9, scaled.Utility, "--", color=color,
                        label=f"{strategy}: hypothetical fixed-weight re-costing")
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=.1)
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_xlabel("AUM anchor at 2019-12-31 (USD billions; log)")
            ax.set_ylabel("Annual utility at gamma=10 (decimal; symmetric log)")
            ax.set_title(f"{dataset} | {fixed.start.iloc[0]} to {fixed.end.iloc[0]} | AUM sensitivity")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
            fig.subplots_adjust(right=.65, bottom=.18)
            fig.text(.01, .02, "Solid: portfolios previously optimised at gamma=10 for each saved AUM. Dashed: baseline saved weights, cost scaled linearly with AUM.\n"
                     "No new optimisation; absent saved points are not interpolated into data tables.", fontsize=8)
            save(fig, f"aum_optimised_vs_hypothetical_{dataset.replace('/', '_')}")

    for file in sorted((destination / "saved_feature_importance").glob("*/importance.csv")):
        importance = pd.read_csv(file)
        models = list(importance.type.unique())
        fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 6.5), squeeze=False)
        for ax, model in zip(axes[0], models):
            sample = importance.loc[importance.type.eq(model) & importance.theme.ne("none")].sort_values("fi")
            errors = 1.96 * sample.fi_se if "fi_se" in sample else None
            ax.errorbar(sample.fi, np.arange(len(sample)), xerr=errors, fmt="o", capsize=3)
            ax.set_yticks(np.arange(len(sample)), sample.theme)
            ax.axvline(0, color="black", linewidth=.8)
            ax.grid(axis="x", alpha=.2)
            ax.set_xlabel("Annual utility loss after permutation")
            ax.set_title(str(model))
        fig.suptitle(f"{file.parent.name}/base | 2020–2025 | saved economic feature importance", y=1.01)
        fig.tight_layout()
        fig.text(.01, -.06, "Error bars: 1.96 x permutation-seed standard error, not training uncertainty. PF-ML: saved net-utility policy.\n"
                 "Markowitz: separate gamma=100, zero-wealth/no-TC construction; importance evaluated at gamma=10.\n"
                 "Negative loss means permutation improved realised utility. This compares different policies/objectives, not the main Markowitz portfolio.", fontsize=8)
        save(fig, f"saved_economic_feature_importance_{file.parent.name}")
    return count


def robustness_metric_dictionary() -> pd.DataFrame:
    entries = [
        ("cost_factor", "tc_scenario = cost_factor * tc_saved", "multiplier", "0, 0.5, 1, 2, 3; saved weights and gross returns unchanged."),
        ("evaluation_gamma", "U=12*(mean(r-tc)-gamma*sample_var(r)/2)", "risk-aversion parameter", "Evaluates saved total-return policy; does not refit or optimise weights."),
        ("optimised_gamma", "gamma parsed from existing absolute-return grid point/config", "risk-aversion parameter", "Parameter used when the original saved absolute-return point was optimised; missing for active capacity."),
        ("optimised_gamma_te", "gamma_TE parsed from saved active-capacity path", "tracking-error risk-aversion parameter", "Benchmark-relative optimisation parameter; not comparable to absolute-return gamma. Total-return evaluation_gamma remains 10."),
        ("optimisation_objective", "absolute_mean_variance or benchmark_relative_mean_variance", "objective label", "Separates originally optimised mandates; gamma and gamma_TE are not pooled."),
        ("aum_anchor_usd", "wealth at configured anchor date, 2019-12-31", "USD", "AUM anchor; actual time-varying wealth follows the original saved market-linked path."),
        ("cost_scale", "new_AUM_anchor / saved_AUM_anchor", "multiplier", "Linear fractional-cost scaling under quadratic dollar impact at fixed weight trades."),
        ("SR_net_paper", "sqrt(12)*mean(r-tc)/sample_sd(r)", "ratio", "Alias of Table 2 SR_net; gross-volatility denominator."),
        ("utility_rank", "descending rank of Utility within dataset and scenario", "ordinal", "Includes all available saved models and benchmarks, ties use minimum rank."),
        ("SR_rank", "descending rank of SR_net_paper within dataset and cost scenario", "ordinal", "Ranks a different criterion from utility; no causal model comparison."),
        ("net_excess_leq_minus_one_months", "sum( r - tc_scenario <= -1 )", "months", "Invalid synthetic compounded excess index; funded bankruptcy requires total return including risk-free return."),
        ("abs_weight_hhi", "sum_i((abs(w_i)/sum_j(abs(w_j)))^2)", "[0,1]", "Concentration relative to gross leverage; NaN if all weights zero."),
        ("top10_share_of_gross", "sum(10 largest abs(w_i))/sum_i(abs(w_i))", "fraction", "Concentration in ten largest positions, including longs and shorts by absolute weight."),
        ("n_nonzero", "count(not exit and w != 0)", "stocks/month", "Actual nonzero portfolio positions, distinct from universe membership count."),
        ("long_count / short_count", "count(w>0) / count(w<0), excluding exit rows", "stocks/month", "Long/short security counts; no new leverage or borrowing constraint imposed."),
        ("long_weight / short_weight", "sum(max(w,0)) / sum(max(-w,0))", "weight", "Long and absolute short exposure; gross leverage is their sum."),
        ("net_investment", "sum(w)", "weight", "Signed invested exposure; may differ from one for unconstrained strategies."),
        ("size_bucket", "average percentile rank of raw me within current universe, thirds", "low / middle / high", "me in USD millions; exits outside membership get a separate label."),
        ("liquidity_bucket", "average percentile rank of raw dolvol within current universe, thirds", "low / middle / high", "dolvol in USD; ranked dolvol_126d feature is not a dollar amount."),
        ("mean_gross_exposure", "sum_t sum_i_in_bucket abs(w_i,t) / T", "gross weight", "Additive exposure; absent buckets contribute zero exposure for that month."),
        ("mean_signed_exposure", "sum_t sum_i_in_bucket w_i,t / T", "signed weight", "Additive long/short exposure; not a standalone tradable subgroup portfolio."),
        ("annual_gross_return_contribution", "12/T * sum_t sum_i_in_bucket w_i,t * ret_ld1_i,t", "decimal/year", "Additive arithmetic attribution, not subgroup standalone return."),
        ("annual_tc_contribution", "12/T * sum_t sum_i_in_bucket wealth_t*lambda_i,t*(w_i,t-w_start_i,t)^2/2", "fraction/year", "Additive saved-impact attribution; includes exit cost where stored."),
        ("mean_monthly_turnover", "sum_t sum_i_in_bucket abs(w_i,t-w_start_i,t) / T", "fraction/month", "Full traded-weight turnover, not divided by two; bucket sum recovers portfolio turnover."),
        ("ff12", "saved Fama-French 12 industry classification", "industry code", "Uses existing panel labels; exposures preserve signed and gross weights."),
        ("rank_ic", "Spearman(pred_ld1,ret_ld1) across stocks each month", "[-1,1]", "One-month saved forecast accuracy; held and broad populations and zero/observed targets separate."),
        ("rmse", "sqrt(mean_i((pred_ld1_i-ret_ld1_i)^2))", "monthly excess-return units", "Cross-sectional prediction error; monthly RMSE summaries are not pooled observation RMSE."),
        ("theme_rank_association_ic", "Spearman(mean(saved ranked features in theme),observed ret_ld1)", "[-1,1]", "Descriptive association only; not fitted-model importance, permutation importance, or causal effect."),
        ("fi", "saved annual utility_unshuffled - utility_theme_permuted, mean over seeds", "decimal utility/year", "Economic importance: PF-ML saved net policy; separate Markowitz gamma=100 zero-wealth/no-TC construction evaluated at utility gamma=10, not main Markowitz portfolio."),
        ("fi_se", "sample_sd(seed importance) / sqrt(number_of_seeds)", "decimal utility/year", "Permutation Monte Carlo standard error; not refitted-model uncertainty."),
        ("switch_fraction", "count(consecutive annual picks differ)/(n_picks-1)", "fraction", "All saved yearly picks including pretest years; no inference about optimal new hyperparameters."),
        ("december_formation", "formation month == December", "boolean", "Reconstitution-season diagnostic; does not isolate causal reconstitution turnover."),
        ("scope", "saved_primary_portfolio_inputs or input_only_no_saved_portfolio", "availability label", "Input-only predictions/choices never enter portfolio performance, rankings, cost/AUM scenarios or significance tests; no weights or returns inferred."),
    ]
    return pd.DataFrame(entries, columns=["metric", "formula", "unit", "interpretation_and_method"])


def run_robustness(root: Path, destination: Path, inventory: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    destination = Path(destination) / "robustness"
    destination.mkdir(parents=True, exist_ok=True)
    _csv(robustness_metric_dictionary(), destination / "metric_dictionary.csv")
    base = {}
    for dataset, items in inventory.groupby("dataset"):
        base[dataset] = float(_load_cfg(items.iloc[0]).investor.wealth)
    _csv(cost_sensitivity(monthly), destination / "fixed_weight_cost_sensitivity.csv")
    _csv(gamma_sensitivity(monthly), destination / "fixed_weight_gamma_evaluation.csv")
    _csv(fixed_weight_aum(monthly, base), destination / "hypothetical_fixed_weight_aum.csv")
    grid, skipped = existing_optimised_grids(root, destination, monthly)
    h, missing = holdings_diagnostics(inventory, monthly, destination)
    skipped.extend(missing)
    catalog = saved_input_catalog(root, inventory)
    _csv(catalog, destination / "saved_input_catalog.csv")
    for dataset in sorted(set(inventory.dataset) - set(catalog.dataset)):
        skipped.append({"test": "one_month_rank_ic", "dataset": dataset, "status": "SKIPPED",
                        "reason": "Saved panel-plus-prediction input pair unavailable; no forecast invented"})
    ic, missing = prediction_diagnostics(catalog, monthly, destination)
    skipped.extend(missing)
    skipped.extend(saved_importance(root, destination))
    hyperparameter_stability(inventory, destination, input_catalog=catalog)
    chart_count = robustness_charts(destination)
    for test, reason in (
        ("multi_horizon_alpha_decay_2_to_12_months", "Saved predictions provide only pred_ld1 (one month); forward-horizon forecasts/decay simulations unavailable"),
        ("fitted_model_random_seed_sensitivity", "Feature importance seeds are permutation seeds, not independently fitted model seeds; no independent refits generated"),
        ("true_predictive_permutation_importance", "Saved importance is utility-based, including gross Markowitz economic importance; theme rank IC is association, not fitted-model importance"),
        ("alternative_covariance_reoptimisation", "No alternative fitted covariance portfolios generated; saved covariance validity/forecast diagnostics reported in audit"),
        ("new_liquidity_constraints", "Existing saved portfolios are diagnosed, not reoptimised under new participation/liquidity constraints")):
        skipped.append({"test": test, "dataset": "all", "status": "SKIPPED", "reason": reason})
    _csv(pd.DataFrame(skipped), destination / "skipped_tests.csv")
    (destination / "ASSUMPTIONS.md").write_text(
        "# Robustness assumptions and interpretation\n\n"
        "All input portfolios are saved results; no model was fit, retuned or reoptimised by these scripts. "
        "Dates are return months from the common sample supplied by the comparison runner. Monetary figures are USD.\n\n"
        "* Paper SR_net uses gross return standard deviation. Utility uses sample gross-return variance with ddof=1. "
        "Annualisation is mean/cost/utility times 12 and volatility times sqrt(12). Turnover is the full sum of absolute trades.\n"
        "* Cost scenarios multiply saved costs at fixed weights. Gamma scenarios change the evaluation penalty only. "
        "Hypothetical AUM scenarios multiply costs by new anchor wealth / saved anchor wealth; quadratic dollar impact "
        "divided by wealth scales linearly at fixed weights. They are not optimised frontier points.\n"
        "* net_excess_leq_minus_one_months flags net excess return <= -100%. Algebraic means, variance, utility, and Sharpe "
        "are retained without trimming. This invalidates a synthetic compounded excess index; funded insolvency must be "
        "checked using total returns including the risk-free return and is separately reported by the audit.\n"
        "* Saved unconstrained leverage and local-listing liquidity can produce extreme market-impact costs. "
        "No new leverage/participation constraint, listing consolidation, currency correction, or winsorisation is imposed. "
        "A large cost calculated correctly from saved inputs remains an economic investability limitation.\n"
        "* Saved gamma/AUM point files are independently read and required to cover all common test months. "
        "Active capacity uses its separately calibrated gamma_TE; Utility there is evaluated at gamma=10 for total returns, "
        "not the benchmark-relative optimisation objective. Baseline grid Utility is evaluated at its optimised gamma.\n"
        "* Size and liquidity thirds use actual me and dolvol within current saved membership each month, with average "
        "percentile ties. me is in source USD millions; dolvol is USD dollar volume. Ranked dolvol_126d is not treated as dollars. "
        "Exiting stocks are labelled outside_current_membership. Contributions sum to a portfolio; they are not independent "
        "tradable subgroup portfolios. No missing group observation is silently relabelled low liquidity.\n"
        "* Construction HHI/top-10 use absolute weights divided by gross leverage; signed exposure preserves long/short direction. "
        "December formation turnover is a reconstitution-season diagnostic, not a causal attribution. Industry labels are saved ff12.\n"
        "* Rank IC is Spearman correlation across stock predictions and saved next-month excess returns. "
        "Observed-only targets exclude explicit zero imputations. Broad and held populations are separate; "
        "only one-month forecasts exist. Theme association IC uses a simple mean of existing ranked features; "
        "it is neither fitted-model importance nor causal evidence.\n"
        "* saved_input_catalog.csv independently discovers existing panel-plus-prediction inputs. "
        "input_only_no_saved_portfolio rows use the configured test calendar only for forecast and saved return-model "
        "hyperparameter diagnostics; they do not enter portfolio statistics, performance rankings, cost scenarios, or "
        "portfolio significance tests. Saved membership is an intended universe, not proof of actual portfolio holdings.\n"
        "* Saved feature importance is utility loss after within-month theme permutation, not pure predictive accuracy; "
        "Markowitz uses a separate gamma=100, zero-wealth/no-TC construction and evaluates importance utility at gamma=10; "
        "it is not the main saved Markowitz-ML portfolio. PF-ML uses its net-utility policy, so magnitudes across models are not directly comparable. "
        "Permutation seeds assess that diagnostic's Monte Carlo error, not model training stability.\n"
        "* Hyperparameter switch fractions include all saved annual choices, including pretest choices. "
        "Distinct strategy picks that share the same return model are repeated with the dataset identifier for traceability. "
        "Fixed Static-ML uses (k,u,g)=(1,1,0), has no annually selected portfolio hyperparameters, and inherits its "
        "validation-path positions rather than the tuned Static-ML* test-window initialisation. Its rule is saved separately.\n",
        encoding="utf-8")
    return {"saved_optimised_points": len(grid), "holdings_months": len(h),
            "input_only_datasets": int(catalog.scope.eq("input_only_no_saved_portfolio").sum()) if len(catalog) else 0,
            "prediction_diagnostic_rows": len(ic), "skipped_tests": len(skipped),
            "chart_pairs_png_pdf": chart_count}
