"""Offline tests for jkmp.features."""

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.features import (
    load_theme_directions, normalise_theme_label, rank_features, standardize_by_month, theme_scores,
)

EOM = pd.Timestamp("2024-12-31")


def test_rank_features_matches_r_ecdf_zero_rule_and_imputation():
    p = pd.DataFrame({"eom": [EOM] * 5, "x": [3.0, 1.0, 0.0, np.nan, 1.0]})
    out = rank_features(p, ["x"])["x"].tolist()
    # ecdf over the 4 non-missing values {3, 1, 0, 1}: 3 -> 1.0, 1 -> 0.75 (ties take max); 0 -> 0 (R rule).
    assert out == [1.0, 0.75, 0.0, 0.5, 0.75]


def test_rank_features_is_per_month():
    p = pd.DataFrame({"eom": [EOM, EOM, EOM + pd.offsets.MonthEnd(1)], "x": [1.0, 2.0, 100.0]})
    assert rank_features(p, ["x"])["x"].tolist() == [0.5, 1.0, 1.0]


def test_theme_scores_flip_negative_directions():
    ranks = pd.DataFrame({"a": [0.2, 0.9], "b": [0.4, 0.1], "c": [0.7, 0.3]})
    directions = pd.DataFrame({"feature": ["a", "b", "c"], "theme": ["t1", "t1", "t2"], "direction": [1.0, -1.0, -1.0]})
    out = theme_scores(ranks, directions)
    assert list(out.columns) == ["theme_t1", "theme_t2"]
    np.testing.assert_allclose(out["theme_t1"], [(0.2 + 0.6) / 2, (0.9 + 0.9) / 2])
    np.testing.assert_allclose(out["theme_t2"], [0.3, 0.7])


def test_standardize_by_month():
    x = pd.DataFrame({"v": [1.0, 2.0, 3.0, 10.0, 20.0]})
    eom = pd.Series([EOM] * 3 + [EOM + pd.offsets.MonthEnd(1)] * 2)
    z = standardize_by_month(x, eom)["v"]
    np.testing.assert_allclose(z, [-1.0, 0.0, 1.0, -np.sqrt(0.5), np.sqrt(0.5)])


def test_normalise_theme_label():
    assert normalise_theme_label("Short-Term Reversal") == "short_term_reversal"


def test_load_theme_directions_against_jkp_files():
    cfg = load_config()
    if not (cfg.paths.raw / "cluster_labels.parquet").exists():
        pytest.skip("raw data not downloaded")
    d = load_theme_directions(cfg.paths.raw, cfg.features.themes)
    assert len(d) == 18 and set(d["direction"]) == {1.0, -1.0}
    assert d.set_index("feature").loc["be_me", "direction"] == 1.0      # value: high B/M is good
    assert d.set_index("feature").loc["ret_1_0", "direction"] == -1.0   # short-term reversal
