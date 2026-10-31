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
    actual = run_static_ml(panel, wealth, cov, dates, dates[12:], 10., grid, True)
    for key in expected:
        pd.testing.assert_frame_equal(actual[key], expected[key], check_exact=False, rtol=1e-11, atol=1e-13)


def test_batched_static_validation_matches_original_without_exit_costs():
    original = ROOT.parent / "src/jkmp/portfolio/static_ml.py"
    spec = spec_from_file_location("original_static_grid_reference", original)
    reference = module_from_spec(spec)
    spec.loader.exec_module(reference)
    from jkmp.portfolio.static_ml import month_inputs, validate
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    records, cov = [], {}
    for index,date in enumerate(dates):
        ids = [1,2] if index < 2 else [2,3]
        cov[date] = CovParts(date, np.array(ids), ("f",), np.ones((2,1))*.1,
                            np.array([[.002]]), np.array([.003,.004]))
        for stock in [1,2,3]:
            records.append(dict(id=stock,eom=date,in_universe=stock in ids,me=100.*stock,
                pred_ld1=.001*stock,ret_ld1=.002*stock,tr_ld1=.002*stock+.001,**{"lambda":1e-10}))
    panel=pd.DataFrame(records)
    wealth=pd.DataFrame(dict(eom=dates,wealth=1e10,mu_ld1=.003))
    inputs=month_inputs(panel,wealth,cov,dates)
    grid=[(1.,1.,0.),(1.,0.,0.),(1.,.25,0.),(.2,.5,1.)]
    for charge in (False,True):
        actual=validate(inputs,panel,wealth,10.,grid,charge)
        expected=reference.validate(inputs,panel,wealth,10.,grid,charge)
        pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-11,atol=1e-13)


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
