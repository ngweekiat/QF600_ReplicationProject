"""Offline tests for the pure helpers in jkmp.data.wrds_pull (no WRDS or network access)."""

import io
import subprocess
import sys
import types
import zipfile

import pandas as pd
import pytest

from jkmp.data import wrds_pull


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_Factors.CSV", text)
    return buf.getvalue()


MONTHLY = """This file was created by CMPT_ME_BEME_RETS using the 202512 CRSP database.
The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RF
200201,  -1.44,   1.20,   3.44,   0.14
200202,  -2.29,  -0.50,   2.31,   0.13

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
2002, -22.76,   4.20,   8.11,   1.61
"""


def test_read_french_monthly_drops_annual_and_converts_percent():
    df = wrds_pull.read_french_zip(_zip(MONTHLY))
    assert list(df.columns) == ["date", "mkt_rf", "smb", "hml", "rf"]
    assert df["date"].tolist() == [pd.Timestamp("2002-01-31"), pd.Timestamp("2002-02-28")]
    assert df["rf"].tolist() == pytest.approx([0.0014, 0.0013])


def test_read_french_daily():
    daily = ",Mkt-RF,SMB,HML,RF\n20020102,   0.50,  -0.10,   0.20,   0.007\n20020103,   0.90,   0.30,  -0.40,   0.007\n"
    df = wrds_pull.read_french_zip(_zip(daily))
    assert df["date"].tolist() == [pd.Timestamp("2002-01-02"), pd.Timestamp("2002-01-03")]
    assert df["mkt_rf"].tolist() == pytest.approx([0.005, 0.009])


def test_top_n_by_month_ranks_within_month_and_skips_missing_me():
    jkp = pd.DataFrame({
        "eom": pd.to_datetime(["2002-01-31"] * 4 + ["2002-02-28"] * 2),
        "id": [1, 2, 3, 4, 1, 2],
        "me": [10.0, 30.0, None, 20.0, 5.0, 50.0],
    })
    top = wrds_pull.top_n_by_month(jkp, n=2)
    jan = top[top["eom"] == "2002-01-31"]
    assert jan["id"].tolist() == [2, 4] and jan["me_rank"].tolist() == [1, 2]
    assert top[top["eom"] == "2002-02-28"]["id"].tolist() == [2, 1]


def test_row_counts_by_month():
    jkp = pd.DataFrame({
        "eom": pd.to_datetime(["2002-01-31", "2002-01-31", "2002-02-28"]),
        "id": [1, 2, 1], "me": [1.0, None, 1.0], "ret_exc_lead1m": [0.1, 0.2, None],
    })
    top = wrds_pull.top_n_by_month(jkp, n=5)
    daily = pd.DataFrame({
        "permno": [1, 1, 2], "date": pd.to_datetime(["2002-01-02", "2002-01-03", "2002-02-01"]), "ret": [0.0, 0.1, None],
    })
    counts = wrds_pull.row_counts_by_month(jkp, top, daily).set_index("eom")
    assert counts.loc["2002-01-31", ["jkp_rows", "me_nonnull", "top_n", "daily_rows", "daily_permnos"]].tolist() == [2, 1, 1, 2, 1]
    assert counts.loc["2002-02-28", "ret_lead1m_nonnull"] == 0


@pytest.mark.parametrize("bad", ["eom; drop table x", "Eom", "a.b", ""])
def test_identifier_rejects_non_identifiers(bad):
    with pytest.raises(ValueError):
        wrds_pull._identifier(bad)


def test_qualified_table():
    assert wrds_pull._qualified("crsp.dsf_v2") == '"crsp"."dsf_v2"'


def test_connect_reads_dotenv_and_shell_wins(tmp_path, monkeypatch):
    from jkmp.data import wrds_pull

    seen = {}
    monkeypatch.setitem(sys.modules, "wrds", types.SimpleNamespace(Connection=lambda **kw: seen.update(kw) or "db"))
    for var in ("WRDS_USERNAME", "WRDS_API_KEY", "WRDS_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("WRDS_USERNAME=from_file\nWRDS_API_KEY=file_key\n")
    monkeypatch.setenv("WRDS_API_KEY", "shell_key")
    assert wrds_pull.connect(env) == "db"
    assert seen == {"wrds_username": "from_file", "api_key": "shell_key"}
    monkeypatch.delenv("WRDS_USERNAME")                      # load_dotenv set it; undo for other tests
    seen.clear()
    assert wrds_pull.connect(tmp_path / "missing.env") == "db" and seen == {"api_key": "shell_key"}


def test_library_imports_without_wrds():
    # wrds and sqlalchemy blocked: every jkmp module must still import (only connect() needs them).
    code = ("import sys; sys.modules['wrds'] = None; sys.modules['sqlalchemy'] = None\n"
            "import pkgutil, importlib, jkmp\n"
            "for m in pkgutil.walk_packages(jkmp.__path__, 'jkmp.'): importlib.import_module(m.name)\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
