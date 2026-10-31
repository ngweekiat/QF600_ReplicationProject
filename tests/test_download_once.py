"""The attended runner must not retry authentication or use persisted credentials."""

import importlib.util
import sys

import pytest

from jkmp.config import PROJECT_ROOT


@pytest.mark.parametrize("fail", [False, True])
def test_attended_download_attempts_login_once(monkeypatch, fail):
    spec = importlib.util.spec_from_file_location("download_once", PROJECT_ROOT / "scripts/download_once.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    calls = []

    class Connection:
        def __init__(self, **kwargs):
            assert kwargs == dict(autoconnect=False, wrds_username="test_user", wrds_password="dummy")

        def _Connection__make_sa_engine_conn(self, raise_err=False):
            assert raise_err
            calls.append("login")
            if fail:
                raise RuntimeError("authentication failed")

        def close(self):
            calls.append("closed")

    monkeypatch.setattr(sys, "argv", ["download_once.py", "--wrds-username", "test_user"])
    monkeypatch.setattr(runner.getpass, "getpass", lambda prompt: "dummy")
    monkeypatch.setattr(runner.wrds, "Connection", Connection)
    monkeypatch.setattr(runner, "pull_all", lambda db, cfg: calls.append("download") or {})
    if fail:
        with pytest.raises(RuntimeError, match="authentication failed"):
            runner.main()
        assert calls == ["login"]
    else:
        runner.main()
        assert calls == ["login", "download", "closed"]
