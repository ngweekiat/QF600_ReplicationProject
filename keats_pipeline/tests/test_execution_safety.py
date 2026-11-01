from importlib.util import module_from_spec, spec_from_file_location
from uuid import uuid4

import numpy as np
import pandas as pd

from workflow import ROOT
from jkmp.covariance import CovParts
from jkmp.portfolio.portfolio_ml import PfmlMoments, save_moments
from jkmp.portfolio.static_ml import run_static_ml


def test_static_subset_optimization_matches_original_with_entry_and_exit():
    original = ROOT.parent / "src/jkmp/portfolio/static_ml.py"
    spec = spec_from_file_location("original_static_for_equivalence", original)
    reference = module_from_spec(spec)
    spec.loader.exec_module(reference)
    recursion_spec=spec_from_file_location("original_recursion_for_equivalence",ROOT.parent/"src/jkmp/portfolio/recursion.py")
    original_recursion=module_from_spec(recursion_spec)
    import sys
    sys.modules[recursion_spec.name]=original_recursion
    recursion_spec.loader.exec_module(original_recursion)
    reference.recursive_weights=original_recursion.recursive_weights
    dates = pd.date_range("2014-12-31", "2016-11-30", freq="ME")
    rows, cov = [], {}
    for index, date in enumerate(dates):
        ids = [1, 2, 3] if index < 15 else [1, 3, 4]
        cov[date] = CovParts(date, np.array(ids), ("f",), np.ones((3, 1)) * .1,
                            np.array([[.002]]), np.array([.003, .004, .005]))
        for stock in [1, 2, 3, 4, *range(100, 120)]:
            rows.append(dict(id=stock, eom=date, in_universe=stock in ids, me=100.0*stock,
                pred_ld1=.001*stock, ret_ld1=.002*stock, tr_ld1=.002*stock+.001, **{"lambda":1e-10}))
    panel = pd.DataFrame(rows)
    wealth = pd.DataFrame(dict(eom=dates, wealth=1e10, mu_ld1=.003))
    grid = [(1.,1.,0.), (.2,.25,1.), (1.,.5,1.)]
    expected = reference.run_static_ml(panel, wealth, cov, dates, dates[12:], 10., grid, True)
    actual = run_static_ml(panel, wealth, cov, dates, dates[12:], 10., True)
    pd.testing.assert_frame_equal(actual["pf"],expected["pf"].loc[lambda x:x.type.eq("Static-ML")].reset_index(drop=True),check_exact=False,rtol=1e-11,atol=1e-13)
    assert actual["weights"].type.eq("Static-ML").all()
    assert len(actual["pf_oos"])==len(dates)




def test_rebuilding_moments_archives_existing_files_without_deleting():
    directory = ROOT / "data/safety_tests" / uuid4().hex
    directory.mkdir(parents=True)
    old = directory / "2019-12-31.npz"
    payload = b"existing cache content must be retained"
    old.write_bytes(payload)
    moments = PfmlMoments(columns=(), rff_w=None, meta={})
    save_moments(moments, directory)
    archived = list((directory / "previous_runs").rglob(old.name))
    assert len(archived) == 1
    assert archived[0].read_bytes() == payload


def test_fast_recursion_matches_original_entries_exits_and_zero_positions():
    import sys
    from jkmp.portfolio.recursion import recursive_weights,month_inputs
    spec=spec_from_file_location("original_recursion_direct",ROOT.parent/"src/jkmp/portfolio/recursion.py")
    reference=module_from_spec(spec)
    sys.modules[spec.name]=reference
    spec.loader.exec_module(reference)
    dates=pd.date_range("2020-01-31",periods=4,freq="ME")
    rows,cov=[],{}
    for index,date in enumerate(dates):
        ids=[1,2] if index<2 else [2,3]
        cov[date]=CovParts(date,np.array(ids),("f",),np.ones((2,1))*.1,np.array([[.002]]),np.array([.003,.004]))
        for stock in [1,2,3]:
            rows.append(dict(id=stock,eom=date,in_universe=stock in ids,me=100.*stock,
                pred_ld1=.001*stock,tr_ld1=.01*stock,ret_ld1=.01*stock-.001,**{"lambda":1e-10}))
    panel=pd.DataFrame(rows)
    wealth=pd.DataFrame(dict(eom=dates,wealth=1e10,mu_ld1=.005))
    inputs=month_inputs(panel,wealth,cov,dates)
    for charge in [True,False]:
        for zero in [True,False]:
            def target(date,x,held):
                return np.zeros_like(held) if zero else .8*held+.01*x.ids
            actual=recursive_weights(inputs,panel,wealth,target,charge)
            expected=reference.recursive_weights(inputs,panel,wealth,target,charge)
            pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-12,atol=1e-14)
