"""Visible, stage-specific disclosure of zero-imputed next-month returns.

Reporting is opt-in at script entry points so importing/loading configurations
in libraries does not normally print tables or create output files.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_stage = None
_reported = set()


def reporting_enabled():
    return _stage is not None


def enable_reporting(stage):
    global _stage, _reported
    _stage = str(stage)
    _reported = set()


@contextmanager
def reporting_context(stage):
    global _stage, _reported
    previous = (_stage, _reported)
    enable_reporting(stage)
    try:
        yield
    finally:
        _stage, _reported = previous


def build_table(cfg):
    """Read the actual prepared-data audit; do not infer assumptions from NaNs."""
    source = cfg.paths.processed / "zero_return_imputations.csv"
    if not source.exists():
        return None
    data = pd.read_csv(source, parse_dates=["eom"])
    required = {"id", "eom", "ret_ld1_original", "ret_ld1", "tr_ld1"}
    if not required.issubset(data):
        raise ValueError(f"Incomplete zero-return audit: {source}")
    data["eom"] = pd.to_datetime(data.eom)
    if not data.ret_ld1.eq(0).all() or data.ret_ld1_original.notna().any():
        raise ValueError(f"Audit must identify genuinely missing source returns assumed zero: {source}")
    table = data.rename(columns={"id": "stock_id", "eom": "formation_month_end",
        "ret_ld1_original": "source_excess_return", "ret_ld1": "assumed_excess_return",
        "tr_ld1": "resulting_total_return"}).copy()
    table["stock_id"] = pd.to_numeric(table.stock_id, errors="raise").astype("int64")
    table["return_month_end"] = table.formation_month_end + pd.offsets.MonthEnd(1)
    universe = cfg.paths.raw.name
    table["universe"] = universe
    table["company_label"] = "Name unavailable locally"
    # Membership labels can be current company names; the JKP security id is the stable identifier.
    membership = ROOT.parent / "configs/Keats universe code/data/universes Keats" / f"membership_{universe}.csv"
    if membership.exists() and len(table):
        names = pd.read_csv(membership, usecols=["id", "form_year", "conm"])
        names = names.dropna(subset=["id", "form_year", "conm"])
        names["stock_id"] = pd.to_numeric(names.id, errors="raise").astype("int64")
        labels = names.groupby(["stock_id", "form_year"]).conm.agg(
            lambda values: " / ".join(sorted(set(values.astype(str)))))
        formation_year = table.formation_month_end.dt.year - table.formation_month_end.dt.month.ne(12).astype(int)
        keys = pd.MultiIndex.from_arrays([table.stock_id, formation_year])
        table["company_label"] = labels.reindex(keys).to_numpy()
        table["company_label"] = table.company_label.fillna("Name unavailable locally")
    columns = ["universe", "stock_id", "company_label", "formation_month_end", "return_month_end",
               "source_excess_return", "assumed_excess_return", "resulting_total_return"]
    return table[columns].sort_values(["return_month_end", "stock_id"], ignore_index=True)


def report_if_enabled(cfg):
    if not reporting_enabled():
        return
    source = cfg.paths.processed / "zero_return_imputations.csv"
    key = (_stage, str(cfg.paths.processed.resolve()), str(cfg.paths.outputs.resolve()))
    if key in _reported:
        return
    table = build_table(cfg)
    if table is None:
        print(f"\nZERO-RETURN AUDIT | {_stage} | {cfg.paths.raw.name}: prepared audit not yet available.", flush=True)
        return  # Preparation can create it later in this same run.
    destination = cfg.paths.outputs.resolve() / "imputation_audit"
    if not destination.is_relative_to(ROOT.resolve()):
        raise ValueError("Refusing to write the stage imputation audit outside the isolated Keats workflow")
    destination.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", _stage).strip("_") or "stage"
    path = destination / f"{filename}.csv"
    saved = table.copy()
    for column in ("formation_month_end", "return_month_end"):
        saved[column] = saved[column].dt.strftime("%Y-%m-%d")
    saved.to_csv(path, index=False, na_rep="MISSING")
    print(f"\nZERO-RETURN ASSUMPTIONS | {_stage} | {cfg.paths.raw.name}", flush=True)
    print(f"Prepared input: {cfg.paths.processed}", flush=True)
    print(f"{len(table)} input-dataset observations have MISSING next-month EXCESS returns assumed to be 0%.", flush=True)
    if len(table):
        shown = saved.copy()
        shown["source_excess_return"] = "MISSING"
        for column in ("assumed_excess_return", "resulting_total_return"):
            shown[column] = shown[column].map(lambda value: f"{value:.6%}")
        print(shown.to_string(index=False, max_rows=None, max_cols=None), flush=True)
        print("These are assumptions, not observed zero returns. Total return equals the next-month risk-free rate.", flush=True)
        print("The table covers the full prepared history, including warmup; not every row necessarily affects this stage.", flush=True)
        print("Company labels come from local membership files; use stock_id as the stable security identifier.", flush=True)
    stage_name = Path(_stage.split()[0]).name
    if stage_name == "02" or stage_name.startswith("02_"):
        print("Covariance uses historical daily returns; these next-month values are not risk-estimation inputs.", flush=True)
    print(f"Saved stage audit: {path}\n", flush=True)
    exclusions = cfg.paths.processed / "history_exclusions.csv"
    if exclusions.exists():
        excluded = pd.read_csv(exclusions)
        if len(excluded):
            print(f"HISTORY EXCLUSIONS | {_stage}: {len(excluded)} stock/month exclusions", flush=True)
            print(excluded.to_string(index=False, max_rows=None, max_cols=None), flush=True)
            print("Membership resumes only when the required historical rows are available; no synthetic lag history was inserted.", flush=True)
        excluded.to_csv(destination / f"{filename}_history_exclusions.csv", index=False)
    _reported.add(key)
