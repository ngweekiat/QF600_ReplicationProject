from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.ml_comparison.audit import (independent_accounting, covariance_diagnostics, _config,
    history_input_status, non_executable_saved_month)


def test_independent_accounting_charges_exit_and_full_l1_turnover():
    date = pd.Timestamp("2020-01-31")
    w = pd.DataFrame({"id":[1,2,3],"eom":date,"w":[.5,-.2,0.],
                      "w_start":[.4,-.1,.1],"exit":[False,False,True]})
    p = pd.DataFrame({"id":[1,2,3],"eom":date,"ret_ld1":[.1,-.05,np.nan],
                      "lambda":[.01,.02,.03]})
    wealth=pd.DataFrame({"eom":[date],"wealth":[10.]})
    out=independent_accounting(w,p,wealth).iloc[0]
    assert out.r == pytest.approx(.06)
    assert out.tc == pytest.approx(.003)
    assert out.turnover == pytest.approx(.3)
    assert out.inv == pytest.approx(.7)
    assert out.shorting == pytest.approx(.2)
    with pytest.raises(ValueError,match="missing return"):
        independent_accounting(w,p.assign(ret_ld1=np.nan),wealth)


def test_saved_covariance_detects_negative_eigenvalue_and_exact_conditioning():
    dates=pd.to_datetime(["2020-01-31","2020-02-29"])
    load=pd.DataFrame({"eom":[dates[0],dates[0],dates[1],dates[1]],
        "id":[1,2,1,2],"f":[1.,0.,1.,0.],"ivar":[1.,2.,-2.,2.]})
    fac=pd.DataFrame({"eom":dates,"factor":["f","f"],"f":[1.,1.]})
    out,cache=covariance_diagnostics(load,fac,set(dates))
    assert out.iloc[0].positive_definite
    assert out.iloc[0].condition_number == pytest.approx(1.)
    assert not out.iloc[1].positive_definite
    assert out.iloc[1].minimum_eigenvalue == pytest.approx(-1.)
    assert out.iloc[1].negative_idiosyncratic_variances == 1
    np.testing.assert_allclose(cache[dates[0]][1],np.diag([2.,2.]))


def test_config_resolves_inheritance_without_running_model(tmp_path):
    (tmp_path/"base.yaml").write_text("dates:\n  test_start: '2020-01-31'\ninvestor:\n  gamma: 10\n")
    (tmp_path/"variant.yaml").write_text("base: base.yaml\ninvestor:\n  gamma: 20\n")
    result=_config(tmp_path/"variant.yaml")
    assert result["dates"]["test_start"] == "2020-01-31"
    assert result["investor"]["gamma"] == 20
    (tmp_path/"cycle.yaml").write_text("base: cycle.yaml\n")
    with pytest.raises(ValueError,match="recursive"):
        _config(tmp_path/"cycle.yaml")


def test_documented_history_gap_does_not_invent_unprepared_exclusion(tmp_path):
    inputs = tmp_path / "keats_pipeline/data/inputs"
    inputs.mkdir(parents=True)
    (inputs / "unavailable_lag_history_US.csv").write_text("id,eom\n88818,2010-12-31\n91135,2010-12-31\n")
    (inputs / "unavailable_lag_history_EU.csv").write_text("id,eom\n320230501,2010-12-31\n")
    applied = [{"dataset": "us_mid/u250", "stock_id": stock} for stock in [88818, 91135]]
    rows = history_input_status(tmp_path, applied)
    assert len(rows) == 3
    assert sum(row["applied_exclusion_records"] for row in rows) == 2
    european = next(row for row in rows if row["region"] == "EU")
    assert european["stock_id"] == 320230501
    assert european["applied_exclusion_records"] == 0
    assert "no saved applied" in european["application_status"]
    assert european["applied_datasets"] == ""


def test_funded_growth_uses_total_return_instead_of_excess_threshold():
    record = dict(dataset="example/base", strategy="test", eom=pd.Timestamp("2020-01-31"),
        eom_ret=pd.Timestamp("2020-02-29"), r=-1.005, tc=0.)
    # Excess index is nonpositive, but RF rescues the actual funded growth.
    assert non_executable_saved_month(record, .01) is None
    result = non_executable_saved_month(dict(record, tc=.01), .01)
    assert result["net_total_return"] == pytest.approx(-1.005)
    assert result["net_total_below_minus_one"]
    assert not result["gross_total_below_minus_one"]
    assert non_executable_saved_month(record, np.nan) is None
