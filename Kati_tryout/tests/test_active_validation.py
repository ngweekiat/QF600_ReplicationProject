"""Batched shadow execution and past-only executed-utility selection."""
from types import SimpleNamespace
import dataclasses
import numpy as np
import pandas as pd
import pytest

from active_validation import replay_candidates,utility,bootstrap_utility,select_executed
from jkmp.portfolio.portfolio_ml import pfml_weights
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.accounting import portfolio_returns


@pytest.mark.parametrize('charge_exits',[True,False])
def test_batched_candidate_books_equal_original_scalar_execution_across_refit_and_exit(tmp_path,charge_exits):
    from test_portfolio_math import _weights_setup
    panel,wealth,cov,rf,_=_weights_setup(1e9)
    panel['eom']=panel.eom-pd.offsets.MonthEnd(2);wealth['eom']=wealth.eom-pd.offsets.MonthEnd(2)
    cov={d-pd.offsets.MonthEnd(2):dataclasses.replace(v,eom=d-pd.offsets.MonthEnd(2)) for d,v in cov.items()}
    rf.index=rf.index-pd.offsets.MonthEnd(2)
    dates=pd.DatetimeIndex(sorted(cov))
    panel.loc[panel.id.eq(3)&panel.eom.ge(dates[2]),'in_universe']=False
    signals={d:(g.id.to_numpy(),np.column_stack([np.sin(g.id+d.month),np.cos(g.id-d.month)])) for d,g in
             panel.loc[panel.in_universe].groupby('eom',sort=True)}
    coefficients={2019:{2:np.array([[.03,.02],[-.01,.04]])},2020:{2:np.array([[-.04,.01],[.02,-.03]])}}
    result=SimpleNamespace(signals=signals,coefs=coefficients,tag='toy',meta={'use_rff':False},columns=['x','y'])
    catalog=pd.DataFrame(dict(cid=[0,1],p=[2,2]))
    cfg=SimpleNamespace(investor=SimpleNamespace(mu=.007,m_iterations=10),portfolio=SimpleNamespace(charge_exit_tc=charge_exits))
    batched=replay_candidates(panel,wealth,cov,rf,dates,result,catalog,cfg,174.1,tmp_path)
    for cid in (0,1):
        aims=pd.concat([pd.DataFrame(dict(id=ids,eom=d,w_aim=S@coefficients[(d+pd.offsets.MonthEnd(1)).year][2][:,cid]))
            for d,(ids,S) in signals.items()],ignore_index=True)
        weights=pfml_weights(month_inputs(panel,wealth,cov,dates,mu_col=None),aims,panel,wealth,rf,
            gamma=174.1,mu=.007,iterations=10,charge_exit_tc=charge_exits,benchmark_relative=True)
        pf=portfolio_returns(weights,panel,wealth).sort_values('eom_ret')
        vector=batched.loc[batched.cid.eq(cid)].sort_values('eom_ret')
        np.testing.assert_allclose(vector[['r','tc','turnover','inv','shorting']],pf[['r','tc','turnover','inv','shorting']],rtol=1e-10,atol=1e-12)
    assert len(list(tmp_path.glob('*.npz')))==len(dates)


def toy_flows():
    rng=np.random.default_rng(91);dates=pd.date_range('2015-01-31','2021-12-31',freq='ME')
    catalog=pd.DataFrame(dict(cid=[0,1,2,3],tag=['g1']*4,g=[.1]*4,p=[8,8,16,32],
        l_no=[0,1,2,3],l=[10.,20.,1.,0.],h=[np.inf]*4))
    rows=[]
    for cid in catalog.cid:
        x=rng.normal(.0001,.002,len(dates));z=x-rng.uniform(0,.00003,len(dates))
        rows.append(pd.DataFrame(dict(cid=cid,eom_ret=dates,active_gross=x,active_net=z)))
    return pd.concat(rows,ignore_index=True),catalog


def test_future_return_poisoning_cannot_change_prior_annual_choices_or_SE():
    flows,catalog=toy_flows();rule=dict(gamma_te=174.1,bootstrap_reps=250,bootstrap_block_months=6,bootstrap_seed=20261107)
    altered=flows.copy();future=altered.eom_ret.gt('2019-12-31')
    altered.loc[future,['active_gross','active_net']]=1e8
    for stable in (False,True):
        a,aa=select_executed(flows,catalog,rule,stable,max_year=2020)
        b,bb=select_executed(altered,catalog,rule,stable,max_year=2020)
        pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(aa,bb)
        assert (a.year==a.eom_ret.dt.year+1).all()


def test_sample_variance_utility_and_bootstrap_recompute_variance():
    x=np.array([.01,-.02,.004,.007,-.003,.012]);z=x-.0001;gamma=174.1
    expected=12*(sum(z)/len(z)-gamma/2*sum((x-x.mean())**2)/(len(x)-1))
    assert utility(x,z,gamma)==pytest.approx(expected)
    rule=dict(bootstrap_reps=100,bootstrap_block_months=3,bootstrap_seed=20261107)
    rng=np.random.default_rng(rule['bootstrap_seed']+2019)
    starts=rng.integers(0,len(x),size=(100,2));scores=[]
    for row in starts:
        ix=[(int(s)+j)%len(x) for s in row for j in range(3)]
        scores.append(12*(z[ix].mean()-gamma/2*np.var(x[ix],ddof=1)))
    assert bootstrap_utility(x,z,gamma,rule,2019)==pytest.approx(np.std(scores,ddof=1),rel=1e-12)


def test_one_se_simplicity_order_does_not_choose_larger_equal_scoring_models():
    flows,catalog=toy_flows();common=flows.loc[flows.cid.eq(0)].drop(columns='cid')
    identical=pd.concat([common.assign(cid=c) for c in catalog.cid],ignore_index=True)
    rule=dict(gamma_te=174.1,bootstrap_reps=100,bootstrap_block_months=6,bootstrap_seed=20261107)
    choice,audit=select_executed(identical,catalog,rule,True,max_year=2020)
    assert choice.cid.eq(1).all() and audit.eligible_count.eq(4).all()
