"""RFF identities and a synthetic walk-forward fit, independent of cached results."""
import numpy as np
import pandas as pd
import pytest

from workflow import config_path
from jkmp.config import load_config
from jkmp.models.random_features import base_draws, weights_for, rff, rff_raw, column_subset
from jkmp.models.return_model import fit_return_model, validation_rows, tuning_marker


def test_validation_rows_follow_historical_membership():
    rows = pd.DataFrame(dict(id=[1, 2, 1, 2],
        eom=pd.to_datetime(["2013-12-31", "2013-12-31", "2014-12-31", "2014-12-31"]),
        in_universe=[True, False, False, True], ret_ld1=[.01, 99., 99., .02]))
    selected = validation_rows(rows, "universe")
    assert list(zip(selected.id, selected.eom.dt.year)) == [(1, 2013), (2, 2014)]
    assert validation_rows(rows, "all") is rows
    with pytest.raises(ValueError, match="historical in_universe"):
        validation_rows(rows.drop(columns="in_universe"), "universe")
    with pytest.raises(ValueError, match="No eligible"):
        validation_rows(rows.assign(in_universe=False), "universe")


def test_market_configs_and_checkpoints_are_independent():
    from run_all import Job, output_paths
    paths=[]
    for market in ("us_large", "us_mid", "eu_large"):
        cfg=load_config(config_path(market))
        assert cfg.return_model.validation_scope=="universe"
        marker=tuning_marker(cfg)
        assert marker.parent==cfg.paths.processed/"return_model"
        assert marker in output_paths(Job(f"{market}.base.returns", "03", market))
        legacy=load_config(config_path(market), {"return_model.validation_scope":"all"})
        assert marker!=tuning_marker(legacy)
        paths.append(marker)
    assert len(set(paths))==3
    for i in range(1,11):
        assert load_config(config_path("us_large",f"iteration_{i}")).return_model.validation_scope=="universe"


def test_base_and_iteration_ridge_grid_checkpoints():
    from jkmp.strategies import portfolio_tuning_marker
    from run_all import Job, output_paths
    for market in ("us_large","us_mid","eu_large"):
        cfg=load_config(config_path(market))
        assert len(cfg.ridge_grid.values)==101 and cfg.ridge_grid.values[0]==0.0
        different=load_config(config_path(market),{"ridge_grid.n":3})
        assert len(different.ridge_grid.values)==4
        assert tuning_marker(cfg)!=tuning_marker(different)
        assert portfolio_tuning_marker(cfg)!=portfolio_tuning_marker(different)
        assert portfolio_tuning_marker(cfg) in output_paths(Job(market,"07",market))
    for i in (1,2,3):
        cfg=load_config(config_path("us_large",f"iteration_{i}"))
        assert len(cfg.ridge_grid.values)==101


def test_restore_does_not_reuse_stale_outputs_with_an_old_marker(tmp_path):
    import os
    from run_all import cached_outputs_ready
    output=tmp_path/"table.csv"
    old_marker=tmp_path/"sources_old.json"
    new_marker=tmp_path/"sources_new.json"
    output.write_text("old result")
    old_marker.write_text("old settings")
    os.utime(output,ns=(100,100));os.utime(old_marker,ns=(200,200))
    assert cached_outputs_ready([output,old_marker])
    output.write_text("new result")
    os.utime(output,ns=(300,300))
    new_marker.write_text("new settings")
    os.utime(new_marker,ns=(400,400))
    assert not cached_outputs_ready([output,old_marker])
    assert cached_outputs_ready([output,new_marker])


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
    assert first.chosen.validation_scope.iloc[0] == "universe"
    assert first.chosen.n_val.iloc[0] == 5 * 12 * 3
    assert first.chosen.n_val_regional.iloc[0] == 5 * 12 * 6

    # A changed past target changes the fit: these are learned coefficients,
    # rather than fixed characteristic weights or only a random projection.
    past_changed = panel.copy()
    past_changed.loc[past_changed.eom.lt("2014-12-31"), "ret_ld1"] *= -1
    third = fit_return_model(past_changed, cfg)
    np.testing.assert_allclose(third.predictions.pred_ld1, -first.predictions.pred_ld1, rtol=1e-10, atol=1e-12)
