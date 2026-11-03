"""Affine-cost derivation against literal implementation, including exits and drift."""
import numpy as np
import pandas as pd
import pytest

from benchmark_netting import AffineTradeBasis
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.portfolio_ml import pfml_weights,ridge_coefs
from jkmp.portfolio.accounting import portfolio_returns


@pytest.mark.parametrize('charge_exits',[True,False])
@pytest.mark.parametrize('beta',[np.array([0.,0.]),np.array([.07,-.03])])
def test_affine_cost_matches_existing_recursive_execution(charge_exits,beta):
    from test_portfolio_math import _weights_setup,W_DATES
    panel,wealth,cov,rf,_=_weights_setup(1e9)
    panel.loc[panel.id.eq(3)&panel.eom.ge(W_DATES[2]),'in_universe']=False
    basis=AffineTradeBasis(panel,wealth,cov,rf,10.,.007,10,charge_exits)
    rows=[];aims=[];costs=[]
    for j,date in enumerate(W_DATES):
        ids=panel.loc[panel.eom.eq(date)&panel.in_universe,'id'].to_numpy()
        signal=np.column_stack([np.sin(ids+j),np.cos(ids-j)])
        curve=basis.step(date,ids,signal)
        rows.append(pd.DataFrame(dict(id=curve.ids,eom=date,w=curve.base_weight+curve.response_weight@beta,
            w_start=curve.base_start+curve.response_start@beta,exit=curve.exit)))
        aims.append(pd.DataFrame(dict(id=ids,eom=date,w_aim=signal@beta)))
        c0,c,C=curve.coefficients()
        assert np.linalg.eigvalsh(C).min()>-1e-12
        costs.append(c0+c@beta+.5*beta@C@beta)
        assert costs[-1]==pytest.approx(curve.literal_cost(beta),rel=1e-11,abs=1e-13)
        # The gradient sign and 1/2 factor are checked by central differences.
        for k in range(2):
            step=np.zeros(2);step[k]=1e-6
            gradient=(curve.literal_cost(step)-curve.literal_cost(-step))/(2e-6)
            assert gradient==pytest.approx(c[k],rel=1e-7,abs=1e-10)
    actual=pfml_weights(month_inputs(panel,wealth,cov,W_DATES,mu_col=None),pd.concat(aims),panel,wealth,rf,
                       gamma=10.,mu=.007,iterations=10,charge_exit_tc=charge_exits,benchmark_relative=True)
    predicted=pd.concat(rows).sort_values(['eom','id']).reset_index(drop=True)
    actual=actual.sort_values(['eom','id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(predicted[['id','eom','exit']],actual[['id','eom','exit']])
    np.testing.assert_allclose(predicted[['w','w_start']],actual[['w','w_start']],rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(portfolio_returns(actual,panel,wealth).tc,costs,rtol=1e-10,atol=1e-12)
    assert int(actual.exit.sum())==(1 if charge_exits else 0)


def test_current_forward_returns_do_not_enter_current_cost_curve():
    from test_portfolio_math import _weights_setup,W_DATES
    panel,wealth,cov,rf,_=_weights_setup(1e9)
    changed=panel.copy();changed.loc[changed.eom.eq(W_DATES[1]),'tr_ld1']=10.
    a=AffineTradeBasis(panel,wealth,cov,rf,10.,.007,10)
    b=AffineTradeBasis(changed,wealth,cov,rf,10.,.007,10)
    ids=np.array([1,2,3]);signal=np.column_stack([ids/10,np.sin(ids)])
    for date in W_DATES[:2]:
        first=a.step(date,ids,signal);second=b.step(date,ids,signal)
        for x,y in zip(first.coefficients(),second.coefficients()):np.testing.assert_array_equal(x,y)


def test_cross_term_changes_only_linear_cost_in_ridge_objective():
    # Finite differences above establish the cost gradient. This independently
    # verifies the closed-form optimiser after subtracting that gradient.
    risk=np.array([[2.,.1],[.1,1.]]);T=np.array([[1.,-.5],[.2,.4]])
    base=np.array([.1,-.2]);scale=np.array([.3,.7]);r=np.array([.2,.1]);ridge=.5
    C=T.T@(scale[:,None]*T);c=T.T@(scale*base)
    got=ridge_coefs(risk+C,r-c,[ridge])[:,0]
    gradient=r-(risk+ridge*np.eye(2))@got-T.T@(scale*(base+T@got))
    np.testing.assert_allclose(gradient,0.,atol=1e-12)
