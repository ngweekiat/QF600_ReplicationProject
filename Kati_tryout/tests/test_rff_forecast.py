"""RFF identities and a synthetic walk-forward fit, independent of cached results."""
import numpy as np
import pandas as pd
import pytest

from workflow import config_path
from jkmp.config import load_config
from jkmp.models.random_features import base_draws, weights_for, rff, rff_raw, column_subset
from jkmp.models.return_model import fit_return_model


def test_fourier_pairs_scaling_and_nested_model_sizes():
    x = np.array([[0., .2, 1.], [.5, .8, .3]])
    z = base_draws(3, 8, seed=12)
    np.testing.assert_array_equal(z, base_draws(3, 8, seed=12))
    w = weights_for(z, .25)
    np.testing.assert_allclose(w, .5 * z)
    projection = x @ w
    expected = np.column_stack([np.cos(projection[:, :2]), np.sin(projection[:, :2])]) / 2
    np.testing.assert_allclose(rff(x, w, 4), expected)
    np.testing.assert_allclose(rff_raw(x, w)[:, column_subset(4, 8)] / 2, expected)
    np.testing.assert_allclose((expected ** 2).sum(axis=1), .5)
    with pytest.raises(FloatingPointError, match="non-finite"):
        rff_raw(np.array([[np.nan, 0., 1.]]), w)


def test_forecaster_learns_coefficients_without_seeing_forecast_year_returns():
    cfg = load_config(config_path("us_large"), {
        "return_model.pred_years": [2015, 2015],
        "return_model.p_grid": [2, 4],
        "return_model.rff_variance_grid": [.05],
        "ridge_grid.n": 3,
    })
    features = list(cfg.features.names)
    dates = pd.date_range("2004-12-31", "2015-11-30", freq="ME")
    rng = np.random.default_rng(58)
    x = rng.uniform(0, 1, (len(dates) * 6, len(features)))
    panel = pd.DataFrame(x, columns=features)
    panel["id"] = np.tile(np.arange(1, 7), len(dates))
    panel["eom"] = np.repeat(dates, 6)
    panel["valid_data"] = True
    panel["in_universe"] = panel.id.le(3)
    panel["ret_ld1"] = .08 * x[:, 0] - .03 * x[:, 1] + rng.normal(0, .002, len(x))

    first = fit_return_model(panel, cfg)
    future_changed = panel.copy()
    future_changed.loc[future_changed.eom.ge("2014-12-31"), "ret_ld1"] = 99.
    second = fit_return_model(future_changed, cfg)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    pd.testing.assert_frame_equal(first.chosen, second.chosen)
    assert first.models[2015].coef.shape == (first.models[2015].p,)
    assert np.linalg.norm(first.models[2015].coef) > 0
    assert first.predictions.pred_ld1.std() > 0
    assert first.chosen.n_test.iloc[0] == 72

    # A changed past target changes the fit: these are learned coefficients,
    # rather than fixed characteristic weights or only a random projection.
    past_changed = panel.copy()
    past_changed.loc[past_changed.eom.lt("2014-12-31"), "ret_ld1"] *= -1
    third = fit_return_model(past_changed, cfg)
    np.testing.assert_allclose(third.predictions.pred_ld1, -first.predictions.pred_ld1, rtol=1e-10, atol=1e-12)
