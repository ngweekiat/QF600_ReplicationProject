"""Reconstructed data pipeline: offline downloads, cache recovery, and prepared panel."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jkmp.config import PROJECT_ROOT, load_config
from jkmp.data import prepare, wrds_pull


def _config(tmp_path):
    return load_config(overrides={
        "paths.raw": str(tmp_path), "universe.target_n": 4,
        "universe.entry_rank": 3, "universe.exit_rank": 5,
    })


def _raw(cfg):
    months = pd.date_range(cfg.dates.warmup_start, cfg.dates.test_end, freq="ME")
    rows = pd.DataFrame({
        "id": np.tile(np.arange(1, 6), len(months)),
        "eom": np.repeat(months, 5),
    })
    rows["me"] = 100.0 - rows["id"]
    rows["sic"], rows["size_grp"], rows["excntry"] = 7372.0, "large", "USA"
    rows["ret_exc"], rows["ret_exc_lead1m"] = 0.02, 0.03
    rows["crsp_exchcd"], rows["crsp_shrcd"] = 1.0, 10.0
    rows[cfg.trading_costs.dolvol_column] = 1e8
    for flag in cfg.universe.jkp_flags:
        rows[flag] = 1.0
    for feature in cfg.features.names:
        if feature != cfg.trading_costs.dolvol_column:
            rows[feature] = rows["id"] / 10.0
    return rows


def _public_files(cfg):
    cfg.paths.raw.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(cfg.dates.warmup_start, cfg.dates.test_end, freq="ME")
    ff = pd.DataFrame({"date": months, "mkt_rf": 0.009, "rf": 0.001, "smb": 0.0, "hml": 0.0})
    ff.to_parquet(cfg.paths.raw / wrds_pull.RAW_FILES["ff_monthly"], index=False)
    details = pd.DataFrame({"abr_jkp": cfg.features.names, "direction": 1.0})
    labels = pd.DataFrame([
        {"characteristic": f, "cluster": theme.replace("_", " ")}
        for theme, features in cfg.features.themes.items() for f in features
    ])
    details.to_parquet(cfg.paths.raw / wrds_pull.RAW_FILES["factor_details"], index=False)
    labels.to_parquet(cfg.paths.raw / wrds_pull.RAW_FILES["cluster_labels"], index=False)
    return {k: cfg.paths.raw / wrds_pull.RAW_FILES[k] for k in ("ff_monthly", "factor_details", "cluster_labels")}


class FakeWRDS:
    def __init__(self, cfg):
        self.jkp = _raw(cfg)
        self.daily = self.jkp[["id", "eom"]].rename(columns={"id": "permno", "eom": "dlycaldt"})
        self.daily["dlyret"] = 0.02
        self.queries = []
        self.fail = False

    def raw_sql(self, sql, **kwargs):
        source = self.jkp if '"contrib"."global_factor"' in sql else self.daily
        if "LIMIT 0" in sql:
            return source.head(0)
        if self.fail:
            raise RuntimeError("simulated network interruption")
        self.queries.append((sql, kwargs["params"]))
        params = kwargs["params"]
        date = "eom" if source is self.jkp else "dlycaldt"
        selected = source.loc[source[date].between(pd.Timestamp(params["start"]), pd.Timestamp(params["end"]))].copy()
        if source is self.jkp:
            assert params["country"] == "USA"
            assert params["exchanges"] == (1, 3)
            assert params["shares"] == (10, 11, 12)
            assert '"common" = 1' in sql and '"obs_main" = 1' in sql
            assert '"id" BETWEEN 1 AND 99999' in sql
        else:
            selected = selected.loc[selected["permno"].isin(params["ids"])]
            selected = selected.rename(columns={"dlycaldt": "date", "dlyret": "ret"})
        return iter([selected])


def test_pull_all_and_cache_contract(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(wrds_pull, "pull_public", lambda cfg, force=False: _public_files(cfg))
    db = FakeWRDS(cfg)
    outputs = wrds_pull.pull_all(db, cfg)
    assert all(path.exists() for path in outputs.values())
    jkp = pd.read_parquet(outputs["jkp"])
    daily = pd.read_parquet(outputs["daily"])
    assert jkp["id"].dtype == np.dtype("int64")
    assert set(daily) == {"permno", "date", "ret"}
    assert daily["date"].min() == pd.Timestamp(cfg.dates.warmup_start)
    counts = pd.read_csv(outputs["counts"])
    assert (counts["jkp_rows"] == 5).all()
    assert (counts["daily_permnos"] == 5).all()
    queries = len(db.queries)
    wrds_pull.pull_all(db, cfg)
    assert len(db.queries) == queries  # completed licensed datasets are reused
    changed = load_config(overrides={"paths.raw": str(tmp_path), "covariance.factor_universe_top_n": 4})
    with pytest.raises(ValueError, match="different settings"):
        wrds_pull.pull_all(db, changed)


def test_interrupted_forced_refresh_does_not_reuse_stale_data(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(wrds_pull, "pull_public", lambda cfg, force=False: _public_files(cfg))
    db = FakeWRDS(cfg)
    wrds_pull.pull_all(db, cfg)
    db.fail = True
    with pytest.raises(RuntimeError, match="interruption"):
        wrds_pull.pull_all(db, cfg, force=True)
    manifest = json.loads((tmp_path / "download_manifest.json").read_text())
    assert manifest["completed"] == []
    assert not list(tmp_path.glob("*.part"))
    db.fail = False
    db.jkp["me"] = 42.0
    wrds_pull.pull_all(db, cfg)
    assert (pd.read_parquet(tmp_path / wrds_pull.RAW_FILES["jkp"])["me"] == 42.0).all()


def test_build_panel_from_downloaded_formats(tmp_path):
    cfg = _config(tmp_path)
    _public_files(cfg)
    raw = _raw(cfg)
    raw.loc[raw["id"] == 1, "be_me"] = 0.0
    raw.loc[raw["id"] == 2, "be_me"] = np.nan  # one missing feature still passes the coverage screen
    raw.to_parquet(tmp_path / wrds_pull.RAW_FILES["jkp"], index=False)
    data = prepare.build_panel(cfg)
    panel = data.panel
    assert panel["eom"].max() == pd.Timestamp("2025-11-30")
    assert panel.loc[panel["valid_data"], "eom"].min() == pd.Timestamp("1994-01-31")
    assert panel.loc[panel["in_universe"], "eom"].min() == pd.Timestamp("1995-01-31")
    assert (panel.loc[panel["id"] == 1, "be_me"] == 0.0).all()
    assert (panel.loc[panel["id"] == 2, "be_me"] == 0.5).all()
    row = panel.loc[(panel["id"] == 1) & (panel["eom"] == pd.Timestamp("2020-01-31"))].iloc[0]
    assert row["tr_ld1"] == pytest.approx(0.031)
    assert row["lambda"] == pytest.approx(2e-9)
    assert row["ff12"] == "BusEq"
    assert np.isfinite(panel.filter(like="theme_").to_numpy()).all()
    assert data.wealth["wealth"].iloc[-1] == cfg.investor.wealth
    assert set(data.membership["status"]) <= {"kept", "added", "filled"}


def test_screened_gap_breaks_history(tmp_path):
    cfg = _config(tmp_path)
    _public_files(cfg)
    raw = _raw(cfg)
    gap = (raw["id"] == 1) & (raw["eom"] == pd.Timestamp("1995-03-31"))
    raw.loc[gap, cfg.trading_costs.dolvol_column] = 0.0
    raw.to_parquet(tmp_path / wrds_pull.RAW_FILES["jkp"], index=False)
    data = prepare.build_panel(cfg)
    after = data.panel[(data.panel["id"] == 1) & data.panel["eom"].between("1995-04-30", "1996-03-31")]
    assert not after["valid_data"].any()
    assert not after["in_universe"].any()
    assert data.screen_log.set_index("screen").loc["dollar_volume", "excluded"] == 1


def test_stream_failure_preserves_existing_file(tmp_path):
    path = tmp_path / "daily.parquet"
    original = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("1993-01-02")], "ret": [0.1]})
    original.to_parquet(path, index=False)

    class Interrupted:
        def raw_sql(self, *args, **kwargs):
            def chunks():
                yield original
                raise RuntimeError("interrupted")
            return chunks()

    with pytest.raises(RuntimeError, match="interrupted"):
        wrds_pull._stream_years(Interrupted(), "", {}, "date", "1993-01-01", "1993-12-31",
                               path, wrds_pull._normalise_daily)
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert not path.with_suffix(".parquet.part").exists()


def test_stream_rejects_duplicate_across_batches(tmp_path):
    frame = pd.DataFrame({"permno": [1], "date": [pd.Timestamp("1993-01-02")], "ret": [0.1]})

    class Duplicated:
        def raw_sql(self, *args, **kwargs):
            return iter([frame, frame])

    with pytest.raises(ValueError, match="chunk boundary"):
        wrds_pull._stream_years(Duplicated(), "", {}, "date", "1993-01-01", "1993-12-31",
                               tmp_path / "daily.parquet", wrds_pull._normalise_daily)


def test_french_missing_values_and_malformed_archive():
    payload = io.BytesIO()
    with wrds_pull.zipfile.ZipFile(payload, "w") as z:
        z.writestr("factors.csv", ",Mkt-RF,SMB,HML,RF\n200201,-99.99,0,0,0.1\n")
    data = wrds_pull.read_french_zip(payload.getvalue())
    assert pd.isna(data["mkt_rf"].iloc[0])
    assert data["rf"].iloc[0] == pytest.approx(0.001)
    with pytest.raises(wrds_pull.zipfile.BadZipFile):
        wrds_pull.read_french_zip(b"not a ZIP")


def test_connect_uses_normal_auth_and_closes_on_failure(monkeypatch, tmp_path):
    import wrds
    calls = []

    class Connection:
        def __init__(self, **options):
            calls.append(options)

        def close(self):
            calls.append("closed")

    monkeypatch.setenv("WRDS_USERNAME", "test_user")
    monkeypatch.setattr(wrds_pull, "LOCAL_LOGIN_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(wrds, "Connection", Connection)
    with pytest.raises(RuntimeError, match="test failure"):
        with wrds_pull.connect():
            raise RuntimeError("test failure")
    assert calls == [{"wrds_username": "test_user"}, "closed"]


def test_public_runner_does_not_connect_to_wrds(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("download_runner", PROJECT_ROOT / "scripts/00_download.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []
    monkeypatch.setattr(wrds_pull, "pull_public", lambda cfg, force=False: calls.append(cfg.paths.raw) or {})
    monkeypatch.setattr(wrds_pull, "connect", lambda: pytest.fail("public downloads must not connect"))
    monkeypatch.setattr(sys, "argv", ["00_download.py", "pull", "--public-only", "--override", f"paths.raw={tmp_path}"])
    runner.main()
    assert calls == [tmp_path]


def test_public_download_handles_mixed_excel_metadata(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    details = pd.DataFrame({"abr_jkp": cfg.features.names, "direction": 1,
                            "time_to_create": [1, "A"] + [None] * (len(cfg.features.names) - 2)})
    details.loc[details["abr_jkp"] == "ret_1_0", "direction"] = -1
    labels = pd.DataFrame([{"characteristic": f, "cluster": t.replace("_", " ")}
                           for t, features in cfg.features.themes.items() for f in features])
    workbook = io.BytesIO()
    details.to_excel(workbook, index=False)

    def zipped(date):
        buffer = io.BytesIO()
        with wrds_pull.zipfile.ZipFile(buffer, "w") as z:
            z.writestr("factors.csv", f",Mkt-RF,SMB,HML,RF\n{date},1,0,0,0.1\n")
        return buffer.getvalue()

    blobs = {
        cfg.data_sources.factor_details_url: workbook.getvalue(),
        cfg.data_sources.cluster_labels_url: labels.to_csv(index=False).encode(),
        cfg.data_sources.ff_factors_url: zipped("200201"),
        cfg.data_sources.ff_factors_daily_url: zipped("20020102"),
    }
    monkeypatch.setattr(wrds_pull, "_fetch", lambda url: blobs[url])
    paths = wrds_pull.pull_public(cfg)
    saved = pd.read_parquet(paths["factor_details"])
    assert saved["time_to_create"].iloc[:2].tolist() == ["1", "A"]
    assert saved.set_index("abr_jkp").loc["ret_1_0", "direction"] == -1
    assert (tmp_path / "Factor Details.xlsx").exists()
    assert not list(tmp_path.glob("*.part"))
    monkeypatch.setattr(wrds_pull, "_fetch", lambda url: pytest.fail("cached public files should be reused"))
    wrds_pull.pull_public(cfg)


def test_local_login_supplies_password_without_environment_username(tmp_path, monkeypatch):
    login = tmp_path / "login.json"
    login.write_text(json.dumps({"username": "test_user", "password": "dummy-test-password"}))
    monkeypatch.setattr(wrds_pull, "LOCAL_LOGIN_PATH", login)
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    assert wrds_pull._login_options() == {"wrds_username": "test_user", "wrds_password": "dummy-test-password"}


def test_username_override_does_not_use_another_accounts_password(tmp_path, monkeypatch):
    login = tmp_path / "login.json"
    login.write_text(json.dumps({"username": "test_user", "password": "dummy-test-password"}))
    monkeypatch.setattr(wrds_pull, "LOCAL_LOGIN_PATH", login)
    monkeypatch.setenv("WRDS_USERNAME", "different_user")
    assert wrds_pull._login_options() == {"wrds_username": "different_user"}


def test_local_login_rejects_missing_password(tmp_path, monkeypatch):
    login = tmp_path / "login.json"
    login.write_text(json.dumps({"username": "test_user"}))
    monkeypatch.setattr(wrds_pull, "LOCAL_LOGIN_PATH", login)
    with pytest.raises(ValueError, match="non-empty password"):
        wrds_pull._login_options()


@pytest.mark.parametrize("fail", [False, True])
def test_saved_login_reports_connection_error_without_prompt(monkeypatch, fail):
    import wrds

    calls = []

    class Connection:
        def __init__(self, **options):
            calls.append(options)

        def _Connection__make_sa_engine_conn(self, raise_err=False):
            assert raise_err
            calls.append("connect")
            if fail:
                raise RuntimeError("original connection error")

        def load_library_list(self):
            calls.append("libraries")

        def close(self):
            calls.append("closed")

    options = {"wrds_username": "test_user", "wrds_password": "dummy-password"}
    monkeypatch.setattr(wrds_pull, "_login_options", lambda: options.copy())
    monkeypatch.setattr(wrds, "Connection", Connection)
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("unexpected prompt"))
    if fail:
        with pytest.raises(RuntimeError, match="original connection error"):
            with wrds_pull.connect():
                pytest.fail("must not yield after failed login")
        assert calls == [{"autoconnect": False, **options}, "connect"]
    else:
        with wrds_pull.connect():
            pass
        assert calls == [{"autoconnect": False, **options}, "connect", "libraries", "closed"]
