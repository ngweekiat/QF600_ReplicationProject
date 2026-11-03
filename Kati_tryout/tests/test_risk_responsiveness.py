"""Risk experiment isolation, causal inputs, and pretest-only selection."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from jkmp.covariance import CovParts,idiosyncratic_vol
import risk_responsiveness as risk


def test_residuals_use_previous_month_exposures_and_contemporaneous_factors():
    dates=pd.to_datetime(['2020-01-02','2020-02-03'])
    daily=pd.DataFrame(dict(id=[1,1],date=dates,ret=[.005,.009]))
    load=pd.DataFrame(dict(id=[1,1],eom=pd.to_datetime(['2019-12-31','2020-01-31']),a=[1.,3.],b=[2.,4.]))
    factors=pd.DataFrame(dict(a=[.01,.02],b=[-.02,.01],n_obs=[500,500],r2=[.5,.5]),index=dates)
    rf=pd.Series(.001,index=dates)
    got=risk.residuals_from_saved_factors(daily,load,factors,rf)
    np.testing.assert_allclose(got.res,[.034,-.092],atol=1e-14)
    factors.iloc[1,0]=100.
    later=risk.residuals_from_saved_factors(daily,load,factors,rf)
    assert later.res.iloc[0]==got.res.iloc[0]


def test_idiosyncratic_estimator_is_causal_under_future_shocks_and_truncation():
    days=pd.bdate_range('2017-01-02',periods=390)
    residual=pd.DataFrame(dict(id=1,date=days,res=.01*np.sin(np.arange(len(days)))))
    cutoff=days[290]
    a=idiosyncratic_vol(residual,days,63,63,200,252)
    changed=residual.copy();changed.loc[changed.date>cutoff,'res']=100.
    b=idiosyncratic_vol(changed,days,63,63,200,252)
    truncated=idiosyncratic_vol(residual.loc[residual.date<=cutoff],days[days<=cutoff],63,63,200,252)
    pd.testing.assert_frame_equal(a.loc[a.eom<=cutoff].reset_index(drop=True),b.loc[b.eom<=cutoff].reset_index(drop=True))
    pd.testing.assert_frame_equal(a.loc[a.eom<=cutoff].reset_index(drop=True),truncated.loc[truncated.eom<=cutoff].reset_index(drop=True))


def test_variance_assembly_preserves_factor_risk_ids_and_size_group_imputation():
    date=pd.Timestamp('2020-01-31')
    part=CovParts(date,np.array([1,2]),('factor',),np.array([[1.],[2.]]),np.array([[.01]]),np.array([.02,.03]))
    L=pd.DataFrame(dict(id=[1,2,3],eom=date,size_grp=['large']*3,in_top=[True,True,True],in_universe=[True,True,False]))
    vol=pd.DataFrame(dict(id=[1,3],eom=date,res_vol=[.01,.03]))
    new=risk.assemble_variances(vol,L,{date:part},21)[date]
    np.testing.assert_array_equal(new.X,part.X);np.testing.assert_array_equal(new.F,part.F)
    np.testing.assert_allclose(new.ivar,[.01**2*21,.02**2*21])
    assert risk.choose_risk([dict(half_life_days=h,validation_utility=.01) for h in [63,126,252]]).half_life_days==126


def test_pretest_risk_choice_freezes_before_test_and_resumes(tmp_path,monkeypatch):
    import iteration_sequence as seq
    from selective_trading import save_result
    monkeypatch.setattr(risk,'ROOT',tmp_path);monkeypatch.setattr(seq,'ROOT',tmp_path)
    monkeypatch.setattr(risk,'prepare_risk',lambda *a:None)
    raw=tmp_path/'raw';processed=tmp_path/'processed'
    for p in [tmp_path/'scripts/jkmp/covariance.py',tmp_path/'data/inputs/daily_US.parquet',raw/'ff_factors_daily.parquet',raw/'top500_by_month.parquet',processed/'covariance/factor_returns.parquet']:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'stable-source')
    previous=tmp_path/'outputs/iteration_6/us_large';previous.mkdir(parents=True)
    (previous/'decision.json').write_text(json.dumps(dict(last_accepted_after=4)))
    reference=tmp_path/'outputs/iteration_4/us_large';reference.mkdir(parents=True)
    for name in ('pf.parquet','weights.parquet'):(reference/name).write_bytes(b'unchanged-reference')
    pd.DataFrame(dict(gamma_te=[174.1])).to_csv(reference/'calibrated.csv',index=False)
    pd.DataFrame(dict(gamma_te=[100.,250.],te_ex_ante=[.0165,.0066])).to_csv(reference/'calibration.csv',index=False)
    folder=tmp_path/'outputs/iteration_7/us_large';folder.mkdir(parents=True)
    class Fake:
        signature='stable'
        cfg=SimpleNamespace(covariance=SimpleNamespace(half_life_idio_days=126),paths=SimpleNamespace(raw=raw,processed=processed),
                            dates=SimpleNamespace(validation_end='2019-12-31'))
        def __init__(self):self.test_calls=0
        def fit(self,state,path,*,calibration=False,gamma_te=None,validation_result=False):
            if calibration and not validation_result:return dict(gamma_te=gamma_te,te_ex_ante=1.65/gamma_te,months=48,last_return='2019-12-31')
            if not calibration:
                self.test_calls+=1
                frozen=json.loads((folder/'pretest_selection.json').read_text())
                assert frozen['selected_half_life_days']==63 and not frozen['test_returns_used']
            dates=pd.date_range('2016-01-31' if calibration else '2020-01-31',periods=48 if calibration else 72,freq='ME')
            mean={63:.01,126:.005,252:.008}[state['idio_half_life_days']]
            pf=pd.DataFrame(dict(eom_ret=dates,r=mean+.01*np.sin(np.arange(len(dates))),tc=.0001,turnover=.1,inv=1.,shorting=0.))
            result=dict(pf=pf,weights=pd.DataFrame({'w':[1.]}),chosen=pd.DataFrame({'year':[2016]}))
            if not calibration:save_result(result,path)
            return result
    engine=Fake();state=dict(benchmark_relative=True,cost_multiple=1.,te_target=.01,idio_half_life_days=126)
    spec=dict(idio_half_life_candidates=[63,126,252]);original=dict(state)
    risk.run_risk_trial(engine,state,folder,spec)
    assert engine.test_calls==1 and state['idio_half_life_days']==63
    rows=pd.read_csv(folder/'risk_half_life_validation.csv')
    assert pd.to_datetime(rows.last_return).max()==pd.Timestamp('2019-12-31')
    risk.run_risk_trial(engine,original,folder,spec)
    assert engine.test_calls==1 and original['idio_half_life_days']==63
