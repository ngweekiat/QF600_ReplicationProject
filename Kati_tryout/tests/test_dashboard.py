"""Keep the results viewer honest about missing fits and source identities."""
import json

import numpy as np
import pandas as pd
import pytest

from dashboard import build_dashboard, collect_series


def test_new_build_displays_unavailable_results_without_substituting_keats(tmp_path):
    original = tmp_path / "keats_pipeline/outputs/base/us_large/static_ml"
    original.mkdir(parents=True)
    (original / "pf.parquet").write_bytes(b"not a Kati result")
    kati = tmp_path / "Kati_tryout"
    kati.mkdir()
    assert collect_series(kati) == []
    build_dashboard(kati)
    status = json.loads((kati / "outputs/dashboard_status.json").read_text())
    assert status["completed_series"] == 0
    assert status["models_fitted_by_dashboard"] is False
    assert "No completed Kati model results yet" in (kati / "outputs/dashboard.html").read_text()


def test_viewer_uses_net_returns_and_costs_from_completed_local_file(tmp_path):
    folder = tmp_path / "outputs/base/us_large/static_ml"
    folder.mkdir(parents=True)
    frame = pd.DataFrame(dict(eom_ret=pd.date_range("2020-01-31", periods=3, freq="ME"),
        r=[.02, -.01, .03], tc=[.001, .002, .003], turnover=[.1, .2, .3], type="Static-ML"))
    frame.to_parquet(folder / "pf.parquet", index=False)
    rows = collect_series(tmp_path)
    assert len(rows) == 1
    assert np.isclose(rows[0]["summary"]["annual_net"], 12*(frame.r-frame.tc).mean())
    assert np.isclose(rows[0]["summary"]["annual_cost"], 12*frame.tc.mean())
    assert np.isclose(rows[0]["points"][0]["net"], .019)
    frame["type"] = "Factor-ML"
    frame.to_parquet(folder / "pf.parquet", index=False)
    with pytest.raises(ValueError, match="strategy identity"):
        collect_series(tmp_path)
