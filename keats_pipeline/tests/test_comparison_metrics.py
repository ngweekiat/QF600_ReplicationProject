import numpy as np
import pandas as pd
import pytest
from analysis.ml_comparison.data import align
from analysis.ml_comparison.metrics import annual,paths_and_rolling,statistics,table

def sample_frame(n=24):
    dates=pd.date_range('2020-01-31',periods=n,freq='ME')
    return pd.DataFrame(dict(dataset='us/base',strategy='known',eom_ret=dates,
        eom=dates-pd.offsets.MonthEnd(1),r=np.resize([-.1,.2,.01,-.02],n),
        tc=np.resize([.001,.002,.003,.004],n),turnover=.4,inv=1.2,shorting=.1))


def test_statistics_independent_paper_formulas_and_net_identity():
    g=sample_frame()
    actual=statistics(g,gamma=5)
    r,tc=g.r.to_numpy(),g.tc.to_numpy()
    mu,variance=r.mean(),((r-r.mean())**2).sum()/(len(r)-1)
    expected=dict(R=12*mu,Vol=np.sqrt(12*variance),SR_gross=np.sqrt(12)*mu/np.sqrt(variance),
        TC=12*tc.mean(),R_TC=12*(mu-tc.mean()),SR_net=np.sqrt(12)*(mu-tc.mean())/np.sqrt(variance),
        Utility=12*((mu-tc.mean())-2.5*variance),Turnover=.4,Lev=1.2,Shorting=.1)
    for key,value in expected.items():
        assert actual[key]==pytest.approx(value,rel=1e-13,abs=1e-14)
    assert actual['R']-actual['TC']==pytest.approx(actual['R_TC'])
    assert actual['net_sharpe_net_sd']!=pytest.approx(actual['SR_net'])


def test_drawdown_includes_initial_unit_capital_and_compounding_is_explicit():
    g=sample_frame(4).assign(r=[-.1,.2,-.25,.1],tc=0.)
    result=statistics(g)
    wealth=np.array([.9,1.08,.81,.891])
    expected_dd=wealth/np.array([1.,1.08,1.08,1.08])-1
    assert result['max_drawdown_excess']==pytest.approx(-.25)
    assert result['cumulative_net_excess']==pytest.approx(.891-1)
    assert result['cumulative_net_excess_arithmetic']==pytest.approx(-.05)
    paths=paths_and_rolling(g,window=3)
    np.testing.assert_allclose(paths.drawdown_excess,expected_dd)
    assert paths.iloc[0].drawdown_excess==pytest.approx(-.1)
    assert paths.iloc[2].rolling_net_sharpe_paper==pytest.approx(
        np.sqrt(12)*g.r.iloc[:3].mean()/g.r.iloc[:3].std(ddof=1))


def test_nonpositive_growth_factor_never_creates_fake_wealth_recovery():
    g=sample_frame(4).assign(r=[.1,-1.1,.2,.3],tc=0.)
    stats=statistics(g)
    assert not stats['geometric_excess_path_valid']
    assert np.isnan(stats['max_drawdown_excess'])
    assert np.isnan(stats['cagr_excess'])
    assert np.isnan(stats['cumulative_net_excess'])
    path=paths_and_rolling(g,window=2)
    assert path.iloc[0].wealth_index_excess==pytest.approx(1.1)
    assert path.iloc[1:].wealth_index_excess.isna().all()
    assert path.iloc[1:].drawdown_excess.isna().all()
    assert path.iloc[-1].cumulative_net_excess_arithmetic==pytest.approx(-.5)
    assert path.iloc[-1].cumulative_utility_paper==pytest.approx(
        (g.r-5*(g.r-g.r.mean())**2).sum())


def test_alignment_identical_within_datasets_and_common_across_without_pooling():
    a=sample_frame()
    b=sample_frame().assign(strategy='other').iloc[2:]
    eu=sample_frame().assign(dataset='eu/base').iloc[4:]
    within,across,logs=align(pd.concat([a,b,eu],ignore_index=True))
    assert within.groupby(['dataset','strategy']).size().to_dict()=={
        ('eu/base','known'):20,('us/base','known'):22,('us/base','other'):22}
    assert across.groupby(['dataset','strategy']).size().eq(20).all()
    assert across.eom_ret.min()==pd.Timestamp('2020-05-31')
    assert logs.query("dataset=='us/base' and strategy=='known'").iloc[0].excluded_months==2
    assert len(table(across))==3
    assert annual(across).year.nunique()==2


def test_alignment_rejects_duplicates_calendar_gaps_and_unfilled_short_samples():
    a=sample_frame()
    with pytest.raises(ValueError,match='Duplicate'):
        align(pd.concat([a,a.iloc[[0]]]))
    with pytest.raises(ValueError,match='Calendar gaps'):
        align(a.drop(index=[5]))
    with pytest.raises(ValueError,match='Fewer than 12'):
        align(a.iloc[:4])
