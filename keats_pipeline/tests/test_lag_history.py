from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from workflow import ROOT
from jkmp.data import lag_history


def test_exclusion_is_only_for_affected_stock_month_and_history_can_resume(monkeypatch):
    root=ROOT/"data/history_tests"/uuid4().hex
    (root/"data/inputs").mkdir(parents=True)
    pd.DataFrame({"id":[7],"eom":["2010-12-31"]}).to_csv(root/"data/inputs/unavailable_lag_history_US.csv",index=False)
    monkeypatch.setattr(lag_history,"ROOT",root)
    source=pd.DataFrame({"id":[7,7,8],"eom":pd.to_datetime(["2011-12-31","2012-01-31","2011-12-31"]),"in_universe":True})
    result,events=lag_history.exclude_unavailable(source,"us_mid",11)
    assert result.in_universe.tolist()==[False,True,True]
    assert len(result)==len(source)
    assert source.in_universe.all()
    assert events.stock_id.tolist()==[7]
    assert events.missing_history_month_end.iloc[0]==pd.Timestamp("2010-12-31")


def test_lag_completion_never_replaces_observed_rows_or_invents_missing_history(monkeypatch):
    root=ROOT/"data/history_tests"/uuid4().hex
    (root/"data/inputs").mkdir(parents=True)
    monkeypatch.setattr(lag_history,"ROOT",root)
    day=pd.Timestamp("2010-12-31")
    existing=pd.DataFrame({"id":[1],"eom":[day],"g":[1.1],"ret_ld1":[.02],"lambda":[.01],"feature":[.3]})
    extra=pd.DataFrame({"id":[1,2],"eom":[day,day],"g":[1.,1.2],"ret_ld1":[np.nan,np.nan],"lambda":[np.nan,np.nan],"feature":[.9,.7]})
    extra.to_parquet(root/"data/inputs/lag_history_US.parquet",index=False)
    need=pd.DataFrame({"id":[1,2],"eom":[day,day]})
    result=lag_history.complete_rows(existing,need,["feature"])
    assert result.set_index("id").loc[1,"feature"]==.3
    assert result.set_index("id").loc[2,"feature"]==.7
    with pytest.raises(ValueError,match="unavailable"):
        lag_history.complete_rows(existing,pd.DataFrame({"id":[3],"eom":[day]}),["feature"])


@pytest.mark.parametrize("region,rows",[("US",930),("EU",441)])
def test_actual_supplement_is_only_real_lag_history_not_return_targets(region,rows):
    path=ROOT/f"data/inputs/lag_history_{region}.parquet"
    if not path.exists(): pytest.skip("Explicit history acquisition not yet completed")
    actual=pd.read_parquet(path)
    source=pd.read_parquet(ROOT/f"data/inputs/lag_history_source_{region}.parquet")
    assert len(actual)==len(source)==rows
    pd.testing.assert_frame_equal(actual[["id","eom"]],source[["id","eom"]])
    assert actual.lag_history_only.all()
    assert actual.ret_ld1.isna().all()
    assert actual["lambda"].isna().all()
    assert np.isfinite(actual.g).all()
    features=[c for c in actual if c in source and c not in ("id","eom","ret_exc")]
    assert len(features)==50
    assert actual[features].ge(0).all().all() and actual[features].le(1).all().all()
