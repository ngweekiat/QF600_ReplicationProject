"""Independent saddle-point solutions for the diagnostic neutrality geometry."""
import numpy as np
import pytest
from scipy.linalg import null_space
from anchor_neutrality_diagnostic import project


@pytest.mark.parametrize('seed',[1,19,73])
def test_nested_projection_matches_direct_primal_dual_optimisation(seed):
    rng=np.random.default_rng(seed);n=18
    A=rng.normal(size=(n,n));sigma=A@A.T+np.eye(n)*.1
    S=np.eye(3)[np.arange(n)%3]
    b=rng.uniform(.01,1,size=n);b/=b.sum();a=rng.normal(size=n)*.01
    beta,joint,risk=project(a,b,sigma,S)
    C=np.column_stack([sigma@b,S]);k=C.shape[1]
    saddle=np.block([[sigma,C],[C.T,np.zeros((k,k))]])
    solution=np.linalg.solve(saddle,np.r_[sigma@a,np.zeros(k)])[:n]
    np.testing.assert_allclose(joint,solution,rtol=1e-9,atol=1e-11)
    assert abs(b@sigma@beta)<1e-11
    np.testing.assert_allclose(C.T@joint,0,atol=1e-10)
    assert risk['beta_projection_variance']>=0 and risk['conditional_sector_projection_variance']>=0
    assert joint@sigma@joint<=beta@sigma@beta+1e-12<=a@sigma@a+2e-12
    np.testing.assert_allclose(a@sigma@a,risk['beta_projection_variance']+
        risk['conditional_sector_projection_variance']+risk['remaining_variance'],atol=1e-12)
    np.testing.assert_allclose(risk['conditional_sector_projection_variance'],
        risk['net_stock_budget_projection_variance']+risk['sector_contrast_projection_variance'],atol=1e-12)


def test_already_neutral_positions_remain_unchanged_under_both_constraints():
    rng=np.random.default_rng(31);n=15
    A=rng.normal(size=(n,n));sigma=A@A.T+np.eye(n)
    b=np.ones(n)/n;S=np.eye(3)[np.arange(n)%3]
    basis=null_space(np.column_stack([sigma@b,S]).T)
    a=basis@rng.normal(size=basis.shape[1])
    beta,joint,risk=project(a,b,sigma,S)
    np.testing.assert_allclose(beta,a,atol=1e-11)
    np.testing.assert_allclose(joint,a,atol=1e-11)
    assert risk['beta_projection_variance']<1e-20
    assert risk['conditional_sector_projection_variance']<1e-20
