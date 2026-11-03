"""Check the declared experiment and protect the original pipeline's files."""
import json
from pathlib import Path

import pytest

from workflow import ROOT, UNIVERSES, config_path, originals_unchanged
from jkmp.config import ConfigError, load_config
from run_all import plan
from script_groups import normalise_group, script_path


def test_private_engine_and_all_configured_inputs_are_local():
    import jkmp.config
    import analysis.ml_comparison.cli as comparison

    expected = json.loads((ROOT / "data/characteristics/recommended_50.json").read_text())
    assert len(expected) == len(set(expected)) == 50
    assert Path(jkmp.config.__file__).resolve().is_relative_to(ROOT / "scripts")
    assert comparison.KATI == ROOT
    choices = [(u, "base") for u in UNIVERSES]
    choices += [("us_large", f'iteration_{i}') for i in range(1,11)]
    for market, group in choices:
        cfg = load_config(config_path(market, group))
        assert set(cfg.features.names) == set(expected)
        assert len(cfg.features.names) == 50
        for target in (cfg.paths.raw, cfg.paths.processed, cfg.paths.outputs):
            assert target.resolve().is_relative_to(ROOT)
        assert cfg.dates.holdout_start is None and cfg.dates.holdout_end is None
    assert originals_unchanged()


@pytest.mark.parametrize("field", ["raw", "processed", "outputs"])
def test_configuration_refuses_to_write_into_original_keats(field):
    target = ROOT.parent / "keats_pipeline" / "outputs" / "forbidden_test"
    with pytest.raises(ConfigError, match="inside its own folder"):
        load_config(config_path("us_large"), {f"paths.{field}": str(target)})
    assert not target.exists()


def test_plan_keeps_all_base_markets_and_only_us_high_cap_iterations():
    jobs = plan(UNIVERSES, "full", "all")
    names = {job.name for job in jobs}
    assert len(names) == len(jobs)
    assert all(set(job.dependencies).issubset(names) for job in jobs)
    for market in UNIVERSES:
        for stage in ("portfolio", "tables", "market_reference"):
            assert f"{market}.base.{stage}" in names
        assert (f"{market}.base.static" in names)==(market=="us_large")
    for group in (f'iteration_{i}' for i in range(1,11)):
        assert any(f"us_large.{group}." in name for name in names)
    assert not any(job.universe in ("us_mid", "eu_large") and job.variant != "base" for job in jobs)
    numerical = {"03", "04", "05", "06", "07", "10", "11", "12", "13", "14", "monitor","experiment"}
    for job in jobs:
        if job.stage in numerical:
            path = script_path(job.stage, normalise_group(job.variant))
            assert path.is_file() and path.resolve().is_relative_to(ROOT / "scripts")


def test_final_plan_has_no_hardcoded_external_source_dependency():
    settings = json.loads((ROOT / "settings.json").read_text())
    for key in ("source", "reference_data"):
        assert (ROOT / settings[key]).resolve().is_relative_to(ROOT / "data")
    assert not settings["downloads_allowed"]
