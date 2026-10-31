"""Ensure every printed disclosure identifies the actual affected securities/months."""
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest

from workflow import ROOT, config_path
from jkmp.config import load_config
from jkmp.imputation_audit import build_table, report_if_enabled, reporting_context


@pytest.mark.parametrize("universe,count", [("us_large", 15), ("us_mid", 21), ("eu_large", 20)])
def test_console_and_saved_audit_identify_every_actual_zero_assumption(universe, count, capsys):
    cfg = load_config(config_path(universe))
    actual = build_table(cfg)
    assert len(actual) == count
    assert actual.stock_id.dtype == "int64"
    assert actual.company_label.ne("Name unavailable locally").any()
    assert actual.source_excess_return.isna().all()
    assert actual.assumed_excess_return.eq(0).all()
    assert (actual.return_month_end == actual.formation_month_end + pd.offsets.MonthEnd(1)).all()
    stage = f"test_disclosure_{universe}"
    with reporting_context(stage):
        report_if_enabled(cfg)
        report_if_enabled(cfg)  # One disclosure per dataset/output for this stage, not every config read.
    printed = capsys.readouterr().out
    assert printed.count("ZERO-RETURN ASSUMPTIONS") == 1
    assert "assumptions, not observed zero returns" in printed
    for row in actual.itertuples():
        assert str(row.stock_id) in printed
        assert row.formation_month_end.strftime("%Y-%m-%d") in printed
        assert row.return_month_end.strftime("%Y-%m-%d") in printed
        assert row.company_label in printed
    saved = pd.read_csv(cfg.paths.outputs / "imputation_audit" / f"{stage}.csv")
    assert len(saved) == count
    assert saved.source_excess_return.eq("MISSING").all()
    assert saved.assumed_excess_return.eq(0).all()


def test_config_loader_reports_automatically_in_script_context(capsys):
    with reporting_context("test_config_disclosure"):
        load_config(config_path("us_large"))
    assert "ZERO-RETURN ASSUMPTIONS" in capsys.readouterr().out


def test_covariance_output_distinguishes_dataset_assumptions_from_risk_inputs(capsys):
    with reporting_context("02 us_large base"):
        load_config(config_path("us_large"))
    assert "not risk-estimation inputs" in capsys.readouterr().out


def test_te_budget_002_does_not_misidentify_prepare_as_covariance(capsys):
    with reporting_context("prepare us_large base te=0.02"):
        load_config(config_path("us_large"))
    assert "not risk-estimation inputs" not in capsys.readouterr().out


def test_empty_audit_is_explicit_and_saves_headers(capsys):
    root = ROOT / "data/audit_tests" / uuid4().hex
    processed = root / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(columns=["id", "eom", "ret_ld1_original", "ret_ld1", "tr_ld1"]).to_csv(
        processed / "zero_return_imputations.csv", index=False)
    cfg = SimpleNamespace(paths=SimpleNamespace(processed=processed, raw=root / "us_large", outputs=root / "outputs"))
    with reporting_context("test_no_assumptions"):
        report_if_enabled(cfg)
    assert "0 input-dataset observations" in capsys.readouterr().out
    saved = pd.read_csv(cfg.paths.outputs / "imputation_audit/test_no_assumptions.csv")
    assert saved.empty
    assert "stock_id" in saved


def test_inconsistent_audit_is_not_silently_printed_as_a_missing_source():
    root = ROOT / "data/audit_tests" / uuid4().hex
    root.mkdir(parents=True)
    pd.DataFrame({"id": [1], "eom": ["2020-01-31"], "ret_ld1_original": [0.12],
                  "ret_ld1": [0.0], "tr_ld1": [0.001]}).to_csv(root / "zero_return_imputations.csv", index=False)
    cfg = SimpleNamespace(paths=SimpleNamespace(processed=root, raw=root / "us_large", outputs=root / "outputs"))
    with pytest.raises(ValueError, match="genuinely missing"):
        build_table(cfg)
