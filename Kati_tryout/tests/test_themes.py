"""Provisional assignments are explicit inputs, not silently published classifications."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from workflow import ROOT, config_path
from jkmp.config import load_config
from jkmp.features import load_theme_directions, theme_scores


def test_frozen_group_override_is_complete_and_refuses_relabelled_config():
    cfg = load_config(config_path("us_large"))
    directions = load_theme_directions(cfg.paths.raw, cfg.features.themes)
    assert len(directions) == 50
    assert directions.feature.is_unique
    assert directions.direction.isin([-1, 1]).all()
    metadata=pd.read_csv(ROOT/'data/characteristics/selected_feature_metadata.csv')
    assert directions.published_group.sum() == metadata.published_jkp_153.sum()
    assert (~directions.published_group).sum() == (~metadata.published_jkp_153).sum()
    assert directions.direction_source.notna().all()
    changed = {name: list(values) for name, values in cfg.features.themes.items()}
    group = next(iter(changed))
    changed["incorrect_theme"] = [changed[group].pop()]
    with pytest.raises(ValueError, match="theme differs"):
        load_theme_directions(cfg.paths.raw, changed)


def test_signed_theme_exposures_do_not_mutate_raw_ml_ranks():
    ranks = pd.DataFrame({"a": [.2, .9], "b": [.8, .1]})
    original = ranks.copy(deep=True)
    directions = pd.DataFrame({"feature": ["a", "b"], "theme": ["quality", "quality"], "direction": [1, -1]})
    result = theme_scores(ranks, directions)
    np.testing.assert_allclose(result.theme_quality, [.2, .9])
    pd.testing.assert_frame_equal(ranks, original)
