"""Independent regression checks for subtle provenance and attribution errors."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

from analysis.ml_comparison.paths import REPOSITORY_ROOT
ROOT = REPOSITORY_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "keats_pipeline/src"))

from analysis.ml_comparison.data import discover
from analysis.ml_comparison.robustness import hyperparameter_stability, holdings_diagnostics
from analysis.ml_comparison import robustness as robustness_module


def test_active_config_override_is_each_saved_path_gamma_not_last_discovered_gamma():
    output = ROOT / "keats_pipeline/outputs/us_large/base/active"
    if not list(output.glob("gte*/pf.parquet")):
        pytest.skip("Saved active portfolio outputs unavailable in checkout")
    inventory, _, _ = discover(ROOT, ["us_large"])
    active = inventory[inventory.source_type.eq("Portfolio-ML (active)")]
    for item in active.itertuples():
        gamma_te = float(Path(item.pf_path).parent.name.removeprefix("gte"))
        overrides = json.loads(item.effective_overrides)
        assert f"investor.gamma_te={gamma_te:g}" in overrides
        assert Path(item.config_path).name == "active_pfml.yaml"


def test_fixed_static_does_not_inherit_tuned_static_annual_choices(tmp_path):
    source = tmp_path / "static_ml"
    source.mkdir()
    pd.DataFrame({"year": [2020, 2021], "k": [2., 3.], "u": [1., 2.], "g": [4., 5.]}).to_csv(source / "chosen.csv", index=False)
    processed = tmp_path / "processed"
    processed.mkdir()
    inventory = pd.DataFrame([
        dict(dataset="test/base", strategy=strategy, source_type=strategy,
             pf_path=str(source / "pf.parquet"), processed_path=str(processed))
        for strategy in ["Static-ML", "Static-ML*"]])
    destination = tmp_path / "new_evaluation"
    destination.mkdir()
    hyperparameter_stability(inventory, destination)
    picks = pd.read_csv(destination / "saved_hyperparameter_choices.csv")
    assert set(picks.strategy) == {"Static-ML*"}
    fixed = pd.read_csv(destination / "fixed_hyperparameter_rules.csv")
    assert fixed.strategy.tolist() == ["Static-ML"]
    assert fixed.iloc[0][["k", "u", "g"]].to_numpy(float) == pytest.approx([1., 1., 0.])


def test_exit_bucket_attribution_annualises_over_all_test_months(tmp_path):
    """A one-month exit bucket must include zero contribution in other months."""
    processed = tmp_path / "processed"
    processed.mkdir()
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.DataFrame([
        dict(id=1, eom=date, in_universe=True, me=100., dolvol=100.,
             **{"lambda": 1e-10}, ret_ld1=.01, ff12=1)
        for date in dates])
    panel = pd.concat([panel, pd.DataFrame([
        dict(id=2, eom=dates[0], in_universe=False, me=50., dolvol=50.,
             **{"lambda": 1e-10}, ret_ld1=np.nan, ff12=2)])], ignore_index=True)
    panel.to_parquet(processed / "panel.parquet", index=False)
    pd.DataFrame({"eom": dates, "wealth": [1e9, 1e9]}).to_parquet(processed / "wealth.parquet", index=False)
    weights = pd.DataFrame([
        dict(id=1, eom=date, w=1., w_start=1., exit=False, type="ML")
        for date in dates])
    weights = pd.concat([weights, pd.DataFrame([
        dict(id=2, eom=dates[0], w=0., w_start=.1, exit=True, type="ML")])], ignore_index=True)
    weight_file = tmp_path / "weights.parquet"
    weights.to_parquet(weight_file, index=False)
    inv = pd.DataFrame([dict(dataset="test/base", strategy="ML", source_type="ML",
                             processed_path=str(processed), weights_path=str(weight_file))])
    monthly = pd.DataFrame(dict(dataset="test/base", strategy="ML", eom_ret=dates + pd.offsets.MonthEnd(1)))
    dest = tmp_path / "evaluation"
    dest.mkdir()
    _, skipped = holdings_diagnostics(inv, monthly, dest)
    assert not skipped
    summary = pd.read_csv(dest / "size_liquidity_attribution_summary.csv")
    exits = summary[summary.bucket.eq("outside_current_membership")]
    assert len(exits) == 2 # same exit attribution in the separate size/liquidity decompositions
    assert exits.n_months_present.eq(1).all()
    assert exits.n_months.eq(2).all()
    assert exits.mean_monthly_turnover.to_numpy() == pytest.approx([.05, .05])
    assert exits.annual_tc_contribution.to_numpy() == pytest.approx([.003, .003])


def test_input_only_forecasts_and_choices_do_not_require_or_create_portfolio_returns(tmp_path, monkeypatch):
    from types import SimpleNamespace
    processed = tmp_path / "p"
    processed.mkdir()
    formation = pd.to_datetime(["2019-12-31", "2020-01-31"])
    panel = pd.DataFrame([dict(id=stock, eom=date, in_universe=True,
        ret_ld1=stock / 100, ret_ld1_imputed_zero=False, x=stock / 3)
        for date in formation for stock in [1, 2, 3]])
    panel.to_parquet(processed / "panel.parquet", index=False)
    panel[["id", "eom"]].assign(pred_ld1=panel.ret_ld1 / 2).to_parquet(processed / "predictions.parquet", index=False)
    model = processed / "return_model"
    model.mkdir()
    pd.DataFrame({"year": [2020], "p": [8], "g": [.5], "lam": [1.]}).to_csv(model / "chosen.csv", index=False)
    catalog = pd.DataFrame([dict(dataset="test/u250", processed_path=str(processed),
        scope="input_only_no_saved_portfolio", config_path="unused")])
    cfg = SimpleNamespace(features=SimpleNamespace(names=["x"], themes={"size": ["x"]}),
        dates=SimpleNamespace(test_start="2020-01-31", test_end="2020-02-29"))
    monkeypatch.setattr(robustness_module, "_load_cfg", lambda row: cfg)
    monthly = pd.DataFrame(columns=["dataset", "strategy", "eom_ret"])
    destination = tmp_path / "e"
    destination.mkdir()
    diagnostics, skipped = robustness_module.prediction_diagnostics(catalog, monthly, destination)
    assert not skipped
    assert len(diagnostics) == 8
    assert diagnostics.scope.eq("input_only_no_saved_portfolio").all()
    assert diagnostics.eom_ret.nunique() == 2
    assert diagnostics.rank_ic.to_numpy() == pytest.approx(np.ones(8))
    primary = pd.DataFrame(columns=["dataset", "strategy", "source_type", "pf_path", "processed_path"])
    hyperparameter_stability(primary, destination, input_catalog=catalog)
    picks = pd.read_csv(destination / "saved_hyperparameter_choices.csv")
    assert picks.strategy.tolist() == ["return_prediction_model"]
    assert picks.scope.eq("input_only_no_saved_portfolio").all()
    assert monthly.empty # This diagnostic never fabricates performance observations.


def test_input_catalog_discovers_saved_us_mid_250_without_adding_primary_performance():
    saved = ROOT / "keats_pipeline/data/processed/us_mid/universe_250/predictions.parquet"
    if not saved.is_file():
        pytest.skip("Prepared-only US mid 250 forecasts unavailable in checkout")
    inventory, monthly, _ = discover(ROOT)
    count_before = len(monthly)
    catalog = robustness_module.saved_input_catalog(ROOT, inventory)
    extra = catalog.loc[catalog.dataset.eq("us_mid/u250")]
    assert len(extra) == 1
    assert extra.iloc[0].scope == "input_only_no_saved_portfolio"
    assert extra.iloc[0].saved_return_model_choices
    assert not extra.iloc[0].saved_covariance
    assert "us_mid/u250" not in set(inventory.dataset)
    assert len(monthly) == count_before
    assert all(Path(p).is_file() for p in catalog.predictions_path)
    eu = ROOT / "keats_pipeline/data/processed/eu_large/universe_250/predictions.parquet"
    if not eu.is_file():
        assert "eu_large/u250" not in set(catalog.dataset)
