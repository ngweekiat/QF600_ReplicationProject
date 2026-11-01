import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from workflow import ROOT, UNIVERSES, VARIANTS, config_path, covariance, originals_unchanged, recover_current_returns
from jkmp.config import ConfigError, load_config
from jkmp.evaluation.holdout import evaluation_window


def test_current_return_uses_previous_consecutive_month_and_never_crosses_ids():
    rows = pd.DataFrame({"id": [1, 1, 1, 2, 2],
        "eom": pd.to_datetime(["2019-12-31", "2020-01-31", "2020-03-31", "2019-12-31", "2020-01-31"]),
        "ret_exc_lead1m": [0.1, 0.2, 0.9, -0.1, -0.2]})
    out = recover_current_returns(rows).set_index(["id", "eom"])
    assert out.loc[(1, pd.Timestamp("2020-01-31")), "ret_exc"] == 0.1
    assert out.loc[(2, pd.Timestamp("2020-01-31")), "ret_exc"] == -0.1
    assert pd.isna(out.loc[(1, pd.Timestamp("2020-03-31")), "ret_exc"])


def test_all_configs_are_isolated_50_feature_configs_and_have_no_fake_holdout():
    expected = set(json.loads((ROOT / "feature_names.json").read_text()))
    for universe in UNIVERSES:
        for variant in VARIANTS:
            cfg = load_config(config_path(universe, variant))
            assert len(cfg.features.names) == 50
            assert set(cfg.features.names) == expected
            for path in (cfg.paths.raw, cfg.paths.processed, cfg.paths.outputs):
                assert path.resolve().is_relative_to(ROOT)
            assert evaluation_window(cfg.dates) == (pd.Timestamp("2020-01-31"), pd.Timestamp("2025-12-31"))
            assert cfg.dates.holdout_start is None
            with pytest.raises(ConfigError, match="No separate Keats holdout"):
                load_config(config_path(universe, variant), ["dates.holdout_locked=false"])


def test_covariance_stops_before_any_network_call_for_missing_daily_data():
    from dataclasses import replace
    from uuid import uuid4
    from workflow import settings
    offline = settings()
    offline["daily_inputs"] = {r: f"data/test_missing_daily_{r}.parquet" for r in ("US", "EU")}
    with patch("workflow.settings", return_value=offline), patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be used")):
        for universe in UNIVERSES:
            cfg = load_config(config_path(universe))
            cfg = replace(cfg, paths=replace(cfg.paths, processed=ROOT / "data/test_missing_covariance" / uuid4().hex))
            with patch("workflow.load_config", return_value=cfg):
                with pytest.raises(FileNotFoundError, match="No downloads were attempted"):
                    covariance(universe)


def test_offline_fred_cache_miss_does_not_download():
    from jkmp.evaluation.regimes import load_fred
    with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be used")):
        with pytest.raises(FileNotFoundError, match="downloads are disabled"):
            load_fred("https://example.invalid/{series}", ["VIXCLS"], ROOT / "data/test_missing_fred")


def test_wrds_and_reference_downloads_are_disabled():
    from jkmp.data.wrds_pull import connect, _http_get
    with patch("urllib.request.urlopen", side_effect=AssertionError("Network must not be used")):
        with pytest.raises(RuntimeError, match="Downloads disabled"):
            connect()
        with pytest.raises(RuntimeError, match="Downloads disabled"):
            _http_get("https://example.invalid")




def test_original_sources_unchanged_and_every_script_compiles():
    assert originals_unchanged()
    for path in ROOT.rglob("*.py"):
        if ".deps" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("universe", UNIVERSES)
def test_prepared_membership_ranks_timing_and_units_match_local_sources(universe):
    cfg = load_config(config_path(universe))
    prepared_path = cfg.paths.processed / "panel.parquet"
    if not prepared_path.exists():
        pytest.skip("Run the offline prepare stage before integration verification")
    from workflow import source_dir
    chars = list(cfg.features.names)
    source = pd.read_parquet(source_dir() / f"panel_{universe}.parquet",
        columns=["id", "eom", "ret_exc_lead1m", "dolvol_126d", *[f + "_rank" for f in chars]])
    source = source[(source.eom + pd.offsets.MonthEnd(1)).le(pd.Timestamp(cfg.dates.test_end))]
    source = source.rename(columns={"dolvol_126d": "raw_dolvol"})
    prepared = pd.read_parquet(prepared_path,
        columns=["id", "eom", "in_universe", "lambda", "ret_ld1", "tr_ld1", *chars])
    held = prepared[prepared.in_universe].merge(source, on=["id", "eom"], validate="one_to_one")
    assert len(held) == len(source)
    available = held.ret_exc_lead1m.notna()
    assert np.allclose(held.loc[available, "ret_ld1"], held.loc[available, "ret_exc_lead1m"].to_numpy(dtype=float))
    assert np.allclose(held["lambda"], 0.2 / held.raw_dolvol.to_numpy(dtype=float))
    for char in chars:
        assert np.allclose(held[char], held[char + "_rank"].to_numpy(dtype=float))
    rf = pd.read_parquet(ROOT.parent / "data/raw/ff_factors_monthly.parquet").set_index("date").rf
    assert np.allclose(held.tr_ld1, held.ret_ld1 + (held.eom + pd.offsets.MonthEnd(1)).map(rf), equal_nan=True)


@pytest.mark.parametrize("universe", UNIVERSES)
def test_small_model_fit_on_actual_keats_members_does_not_use_future_returns(universe):
    from jkmp.models.return_model import fit_return_model
    cfg = load_config(config_path(universe), {
        "return_model.pred_years": [2015, 2015], "return_model.p_grid": [2, 4],
        "return_model.rff_variance_grid": [0.05], "ridge_grid.n": 3,
    })
    path = cfg.paths.processed / "panel.parquet"
    if not path.exists():
        pytest.skip("Prepare the local data first")
    panel = pd.read_parquet(path, columns=["id", "eom", "valid_data", "ret_ld1", *cfg.features.names],
                            filters=[("in_universe", "=", True)])
    result = fit_return_model(panel, cfg)
    altered = panel.copy()
    altered.loc[altered.eom.ge(pd.Timestamp("2014-12-31")), "ret_ld1"] = 99.0
    other = fit_return_model(altered, cfg)
    pd.testing.assert_frame_equal(result.predictions, other.predictions)
    assert result.chosen.n_train.gt(0).all()
    assert result.chosen.n_val.gt(0).all()
    assert result.chosen.n_test.gt(0).all()


def test_midcap_breadth_stays_near_rank_500_and_drops_after_gap():
    from workflow import broaden_membership
    dates = pd.to_datetime(["2019-12-31", "2020-01-31", "2020-03-31"])
    frame = pd.DataFrame([(i, date, 1001 - i) for date in dates for i in range(1, 1001)], columns=["id", "eom", "me"])
    result = broaden_membership(frame, "us_mid")
    december = result[result.eom.eq(dates[0])]
    assert len(december) == 250
    assert december.id.min() == 376
    assert december.id.max() == 625
    assert not result.eom.eq(dates[2]).any()
