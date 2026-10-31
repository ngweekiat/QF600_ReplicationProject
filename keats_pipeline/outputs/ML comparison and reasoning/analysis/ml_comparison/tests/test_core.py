"""Independent algebra, timing, source-preservation and saved-result checks."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from analysis.ml_comparison.paths import REPOSITORY_ROOT
ROOT=REPOSITORY_ROOT
sys.path.insert(0,str(ROOT/'keats_pipeline/src'))

from analysis.ml_comparison.data import align, discover, sha256, snapshot
from analysis.ml_comparison.metrics import annual, paths_and_rolling, statistics, table


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


def test_snapshot_is_read_only_and_detects_changed_source_bytes(tmp_path):
    source=tmp_path/'keats_pipeline/src/jkmp/model.py'
    source.parent.mkdir(parents=True)
    source.write_text('# old model bytes\n')
    output=tmp_path/'keats_pipeline/outputs/us/base/pf.csv'
    output.parent.mkdir(parents=True)
    output.write_text('r,tc\n0.1,0.01\n')
    before=snapshot(tmp_path)
    again=snapshot(tmp_path)
    pd.testing.assert_frame_equal(before,again)
    assert source.read_text()=='# old model bytes\n'
    output.write_text('r,tc\n0.2,0.01\n')
    after=snapshot(tmp_path)
    assert before.set_index('path').loc[str(output.relative_to(tmp_path)),'sha256']!=after.set_index('path').loc[str(output.relative_to(tmp_path)),'sha256']


def test_independent_two_sided_turnover_and_exit_cost_accounting():
    from jkmp.portfolio.accounting import portfolio_returns
    date=pd.Timestamp('2020-01-31')
    weights=pd.DataFrame(dict(id=[1,2,3],eom=date,w=[.7,-.2,0.],w_start=[.6,-.1,.1],exit=[False,False,True]))
    panel=pd.DataFrame(dict(id=[1,2,3],eom=date,ret_ld1=[.1,.05,np.nan],**{'lambda':[.01,.02,.03]}))
    wealth=pd.DataFrame(dict(eom=[date],wealth=[100.]))
    pf=portfolio_returns(weights,panel,wealth).iloc[0]
    assert pf.r==pytest.approx(.06)
    assert pf.tc==pytest.approx(.03)
    assert pf.turnover==pytest.approx(.3) # sum |delta w|; no half-turnover
    assert pf.inv==pytest.approx(.9)
    assert pf.shorting==pytest.approx(.2)
    assert pf.eom_ret==pd.Timestamp('2020-02-29')


@pytest.mark.parametrize('universe',['us_large','us_mid','eu_large'])
def test_saved_table2_matches_independently_recomputed_saved_monthly_returns(universe):
    output=ROOT/'keats_pipeline/outputs'/universe/'base'
    saved=output/'tables/table2_vs_paper.csv'
    if not saved.exists():
        pytest.skip('Saved source outputs are unavailable in this checkout.')
    reference=pd.read_csv(saved,header=[0,1],index_col=0)
    frames=[pd.read_parquet(output/name) for name in [
        'benchmarks/pf_test.parquet','static_ml/pf.parquet','portfolio_ml/rff/pf.parquet','portfolio_ml/raw/pf.parquet']]
    monthly=pd.concat(frames,ignore_index=True)
    for strategy,g in monthly.groupby('type'):
        # Recompute directly, without calling either comparison or project metric function.
        r,tc=g.r.to_numpy(float),g.tc.to_numpy(float)
        mean=r.sum()/len(r)
        variance=((r-mean)**2).sum()/(len(r)-1)
        expected=dict(R=12*mean,Vol=np.sqrt(12*variance),SR_gross=np.sqrt(12)*mean/np.sqrt(variance),
            TC=12*tc.mean(),R_TC=12*(r-tc).mean(),SR_net=np.sqrt(12)*(r-tc).mean()/np.sqrt(variance),
            Utility=12*((r-tc).mean()-5*variance),Turnover=g.turnover.mean(),Lev=g.inv.mean())
        stats=statistics(g)
        for key,value in expected.items():
            assert reference.loc[strategy,(key,'ours')]==pytest.approx(value,rel=2e-12,abs=2e-13)
            assert stats[key]==pytest.approx(value,rel=2e-12,abs=2e-13)


def test_discovery_is_read_only_and_active_te_labels_have_saved_evidence():
    if not (ROOT/'keats_pipeline/outputs/us_large/base/benchmarks/pf_test.parquet').exists():
        pytest.skip('Saved source outputs are unavailable in this checkout.')
    source=ROOT/'keats_pipeline/outputs/us_large/base/benchmarks/pf_test.parquet'
    before=sha256(source)
    inventory,monthly,missing=discover(ROOT,universes=['us_large'])
    assert sha256(source)==before
    assert not monthly.duplicated(['dataset','strategy','eom_ret']).any()
    assert (monthly.eom+pd.offsets.MonthEnd(1)).eq(monthly.eom_ret).all()
    assert inventory.feature_count.eq(50).all()
    if 'Portfolio-ML (active; TE 2%)' in inventory.strategy.to_list():
        row=inventory[inventory.strategy.eq('Portfolio-ML (active; TE 2%)')].iloc[0]
        assert '/gte29.29/' in row.pf_path.replace('\\','/')
        assert row.source_type=='Portfolio-ML (active)'
