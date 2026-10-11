import dataclasses
import datetime as dt
import math

import pytest

from jkmp.config import PROJECT_ROOT, ConfigError, load_config


def test_base_config_loads():
    cfg = load_config()
    assert cfg.investor.gamma == 10.0
    assert cfg.investor.wealth == 1.0e10
    assert cfg.universe.target_n == 80
    assert cfg.dates.test_end == dt.date(2025, 12, 31)
    assert len(cfg.features.themes) == 13
    assert len(cfg.features.names) == 50
    assert cfg.paths.raw == PROJECT_ROOT / "data" / "raw"


def test_ridge_grid_matches_r():
    values = load_config().ridge_grid.values
    assert len(values) == 101 and values[0] == 0.0
    assert values[1] == pytest.approx(math.exp(-10))
    assert values[-1] == pytest.approx(math.exp(10))


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.investor.gamma = 5.0  # type: ignore[misc]


def test_overrides_strings_and_mapping():
    assert load_config(overrides=["investor.gamma=5"]).investor.gamma == 5.0
    assert load_config(overrides={"investor.wealth": 1.0e9}).investor.wealth == 1.0e9
    cfg = load_config(overrides=["dates.warmup_start=2003-01-31"])
    assert cfg.dates.warmup_start == dt.date(2003, 1, 31)


@pytest.mark.parametrize(
    "override",
    [
        "investor.gama=5",              # typo -> unknown key
        "investor.wealth=1e9",          # PyYAML reads 1e9 as a string
        "universe.entry_rank=90",       # entry_rank > target_n
        "dates.test_start=2019-06-30",  # overlaps validation
        "portfolio_ml.p_grid=[63]",     # odd number of random features
    ],
)
def test_bad_overrides_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=[override])


def test_experiment_config_inherits_base(tmp_path):
    lin = load_config(PROJECT_ROOT / "configs" / "experiments" / "linear_pfml.yaml")
    base = load_config()
    assert lin.portfolio_ml.use_rff is False and base.portfolio_ml.use_rff is True
    assert lin.investor == base.investor and lin.portfolio_ml.p_grid == base.portfolio_ml.p_grid
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"base: {PROJECT_ROOT / 'configs' / 'base.yaml'}\nportfolio_ml:\n  use_rf: false\n")
    with pytest.raises(ConfigError, match="use_rf"):
        load_config(bad)
