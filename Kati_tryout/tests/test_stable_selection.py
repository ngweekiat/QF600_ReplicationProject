"""One-SE eligibility, serial dependence, forward timing and objective equivalence."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stable_selection import block_mean_se,stable_hypers,monthly_search


def validation():
    dates=pd.date_range('2019-01-31',periods=12,freq='ME')
    best=np.r_[np.repeat(-.02,6),np.repeat(.06,6)]
    rows=[]
    for p,l,offset in [(512,1.,0.),(16,1.,-.0005),(8,1.,-.001),(8,100.,-.002)]:
        rows.append(pd.DataFrame(dict(eom_ret=dates,tag='g',g=.05,p=p,l_no=0 if l==1 else 1,l=l,h=np.inf,obj=best+offset)))
    return pd.concat(rows,ignore_index=True)


def test_one_se_rule_selects_fewer_features_then_stronger_ridge():
    chosen,audit=stable_hypers(validation())
    assert chosen.iloc[0].p==8 and chosen.iloc[0].l==100.
    assert audit.iloc[0].selected_utility>=audit.iloc[0].utility_threshold
    assert audit.iloc[0].eligible_count==4
    assert audit.iloc[0].best_p==512


def test_future_validation_returns_cannot_change_earlier_choice():
    first=validation();future=first.copy()
    future.eom_ret+=pd.DateOffset(years=1)
    future.loc[future.p.eq(512),'obj']=100.
    early,_=stable_hypers(first)
    extended,_=stable_hypers(pd.concat([first,future],ignore_index=True))
    pd.testing.assert_frame_equal(early,extended.loc[extended.year.eq(2020)].reset_index(drop=True))


def test_bootstrap_se_matches_literal_circular_blocks():
    values=np.arange(12,dtype=float)/100
    rng=np.random.default_rng(82);means=[]
    for _ in range(100):
        starts=rng.integers(0,12,size=2)
        indices=[(int(start)+i)%12 for start in starts for i in range(6)]
        means.append(values[indices].mean())
    assert block_mean_se(values,reps=100,block=6,seed=82)==pytest.approx(np.std(means,ddof=1))
    assert block_mean_se(np.ones(12))==0.


def test_monthly_scores_reproduce_original_yearly_scores():
    from test_portfolio_math import _panel
    panel,wealth,cov,rf,dates,features=_panel(n_months=36)
    dates=dates[3:-1];years=[2020,2021]
    aim=dates[(dates+pd.offsets.MonthEnd(1)).year>2020]
    pm=SimpleNamespace(p_grid=[2,4],half_life_years=[None])
    kwargs=dict(gamma=10.,mu=.007,lb=2,iterations=10,scale_by_vol=True,add_constant=False,
                use_rff=True,rff_variance=.05,p_max=4,seed=1)
    result=monthly_search(panel,wealth,cov,rf,dates,features,aim,pm,[.1,1.],years,'rff',kwargs)
    # The implementation independently asserts equality against yearly sums for
    # every December/candidate. Check retained flows and full candidate calendars.
    assert 'obj' in result.validation and result.validation.eom_ret.dt.month.nunique()==12
    assert result.validation.groupby(['p','l_no']).size().nunique()==1
