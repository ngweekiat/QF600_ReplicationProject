"""Pure checks for supplementary data alignment; tests never connect to WRDS."""
import pandas as pd
import numpy as np
import pytest

from workflow import fill_forward_gaps


def test_zero_policy_preserves_observed_returns_and_flags_only_missing_holdings():
    from workflow import zero_missing_held_returns
    frame = pd.DataFrame({"id": [1, 2, 3], "eom": pd.to_datetime(["2020-01-31"] * 3),
        "in_universe": [True, True, False], "ret_ld1": [np.nan, 0.12, np.nan],
        "tr_ld1": [np.nan, 0.123, np.nan]})
    rf = pd.Series([0.003], index=pd.to_datetime(["2020-02-29"]))
    result = zero_missing_held_returns(frame, rf)
    assert result.ret_ld1_imputed_zero.tolist() == [True, False, False]
    assert result.ret_ld1.iloc[0] == 0.0
    assert result.tr_ld1.iloc[0] == 0.003
    assert result.ret_ld1.iloc[1] == 0.12
    assert pd.isna(result.ret_ld1.iloc[2])
    assert pd.isna(result.ret_ld1_original.iloc[0])
    assert pd.isna(frame.ret_ld1.iloc[0])


@pytest.mark.parametrize("universe,expected", [("us_large", 15), ("us_mid", 21), ("eu_large", 20)])
def test_actual_zero_policy_has_expected_flags_without_overwriting_source(universe, expected):
    from workflow import ROOT, config_path
    from jkmp.config import load_config
    cfg = load_config(config_path(universe))
    if not cfg.paths.processed.name == "with_zero_returns":
        pytest.skip("Zero policy migration has not completed")
    current = pd.read_parquet(cfg.paths.processed / "panel.parquet",
        columns=["id", "eom", "in_universe", "ret_ld1", "ret_ld1_original", "ret_ld1_imputed_zero"])
    old_path = ROOT / f"data/processed/{universe}/with_supplements/panel.parquet"
    original = pd.read_parquet(old_path, columns=["id", "eom", "ret_ld1"])
    assert len(current) == len(original)
    assert int(current.ret_ld1_imputed_zero.sum()) == expected
    assert current.loc[current.ret_ld1_imputed_zero, "ret_ld1"].eq(0).all()
    assert not current.loc[current.in_universe, "ret_ld1"].isna().any()
    np.testing.assert_allclose(current.ret_ld1_original, original.ret_ld1, equal_nan=True)
    observed = original.ret_ld1.notna()
    np.testing.assert_allclose(current.loc[observed, "ret_ld1"], original.loc[observed, "ret_ld1"])


def test_source_supplements_only_fill_missing_returns_and_respect_calendar_months():
    rows = pd.DataFrame({"id": [1, 1, 1, 2], "eom": pd.to_datetime([
        "2020-01-31", "2020-02-29", "2020-03-31", "2020-01-31"]),
        "ret_exc_lead1m": pd.array([None, 0.1, None, None], dtype="Float64")})
    supplement = pd.DataFrame({"id": [1, 1, 1], "eom": pd.to_datetime([
        "2020-01-31", "2020-02-29", "2020-03-31"]),
        "ret_exc": pd.array([None, 0.2, 0.3], dtype="Float64"),
        "ret_exc_lead1m": pd.array([None, 0.9, None], dtype="Float64")})
    result = fill_forward_gaps(rows, supplement)
    assert result.ret_exc_lead1m.iloc[0] == 0.2
    assert result.ret_exc_lead1m.iloc[1] == 0.1  # Original observed value always wins.
    assert pd.isna(result.ret_exc_lead1m.iloc[2])  # No April observation is available.
    assert pd.isna(result.ret_exc_lead1m.iloc[3])  # Never cross security ids.


@pytest.mark.parametrize("region", ["US", "EU"])
def test_daily_input_has_declared_ids_dates_and_finite_total_returns(region):
    from workflow import ROOT
    import json
    import pyarrow.parquet as pq
    path = ROOT / f"data/inputs/daily_{region}.parquet"
    if not path.exists():
        pytest.skip("Explicit acquisition has not run")
    info = json.loads(path.with_suffix(".json").read_text())
    assert info["currency"] == "USD"
    assert info["return_type"] == "total"
    assert info["identifier"] == "jkp_id"
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=250000):
        frame = batch.to_pandas()
        assert frame.id.notna().all()
        assert frame.date.notna().all()
        assert frame.date.le(pd.Timestamp("2025-12-31")).all()
        assert np.isfinite(frame.ret).all()
        assert frame.ret.ge(-1).all()
        count += len(frame)
    assert count == info["rows"]
