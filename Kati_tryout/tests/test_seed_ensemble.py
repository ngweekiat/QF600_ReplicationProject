"""Ensemble execution and quadratic costs against literal synthetic stock trades."""
import numpy as np
import pandas as pd
import pytest

from seed_ensemble import average_aims,audit_combined_orders
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.portfolio_ml import pfml_weights
from jkmp.portfolio.accounting import portfolio_returns


@pytest.mark.parametrize('charge_exits',[True,False])
def test_average_targets_equals_average_executed_weights_and_correct_costs(charge_exits):
    from test_portfolio_math import _weights_setup,W_DATES
    panel,wealth,cov,rf,_=_weights_setup(1e9)
    panel.loc[panel.id.eq(3)&panel.eom.ge(W_DATES[2]),'in_universe']=False
    kwargs=dict(gamma=174.1,mu=.007,iterations=10,charge_exit_tc=charge_exits,benchmark_relative=True)
    inputs=month_inputs(panel,wealth,cov,W_DATES,mu_col=None)
    members={}
    for seed in range(1,6):
        aims=panel.loc[panel.in_universe&panel.eom.isin(W_DATES),['id','eom']].copy()
        # An abrupt change mimics a coefficient jump at an annual refit.
        aims['w_aim']=.04*np.sin(aims.id*seed+aims.eom.dt.month)*np.where(aims.eom.ge(W_DATES[2]),2.,1.)
        w=pfml_weights(inputs,aims,panel,wealth,rf,**kwargs)
        members[seed]=dict(aims=aims,weights=w,pf=portfolio_returns(w,panel,wealth))
    aims=average_aims(members)
    actual=pfml_weights(inputs,aims,panel,wealth,rf,**kwargs)
    audit=audit_combined_orders(members,actual,panel,wealth)
    expected=sum(r['pf'].tc.to_numpy() for r in members.values())/5
    np.testing.assert_allclose(audit.mean_member_cost,expected,rtol=1e-11,atol=1e-12)
    assert audit.variance_cost_saving.ge(-1e-12).all()
    assert audit.variance_cost_saving.sum()>0
    np.testing.assert_allclose(audit.mean_member_cost-audit.ensemble_cost,audit.variance_cost_saving,rtol=1e-10,atol=1e-12)
    pf=portfolio_returns(actual,panel,wealth)
    np.testing.assert_allclose(pf.r,np.mean([r['pf'].r for r in members.values()],axis=0),rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(pf.tc,audit.ensemble_cost,rtol=1e-11,atol=1e-12)


def test_aim_averaging_requires_identical_unique_keys():
    a=pd.DataFrame(dict(id=[1,2],eom=[pd.Timestamp('2019-12-31')]*2,w_aim=[.1,-.1]))
    with pytest.raises(ValueError,match='keys differ'):
        average_aims({1:dict(aims=a),2:dict(aims=a.iloc[:1])})
    with pytest.raises(ValueError,match='Invalid member'):
        average_aims({1:dict(aims=pd.concat([a,a]))})
    b=a.copy();b.w_aim=-a.w_aim
    assert average_aims({1:dict(aims=a),2:dict(aims=b)}).w_aim.eq(0).all()


def test_identical_seed_members_have_no_artificial_netting_gain():
    date=pd.Timestamp('2019-12-31')
    w=pd.DataFrame(dict(eom=[date],id=[1],exit=[False],w=[.5],w_start=[.4]))
    panel=pd.DataFrame(dict(eom=[date],id=[1],**{'lambda':[.01]}))
    wealth=pd.DataFrame(dict(eom=[date],wealth=[100.]))
    audit=audit_combined_orders({s:dict(weights=w.copy()) for s in range(1,6)},w,panel,wealth)
    assert audit.variance_cost_saving.iloc[0]==pytest.approx(0,abs=1e-15)
    assert audit.ensemble_cost.iloc[0]==pytest.approx(.005)
