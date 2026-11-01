from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from characteristics.statistics import paired_columns,monthly_return_stats,pairwise_matrices,hac_mean,bh_adjust
from characteristics.selection import diverse_pick
from characteristics.analysis import before_test,walk_forward_univariate


def test_pairwise_moments_match_pandas_with_different_missingness():
    x=pd.DataFrame({'a':[1.,2.,np.nan,8.,5.],'b':[np.nan,1.,4.,8.,3.],'c':[1.,1.,1.,1.,1.]})
    corr,cov,n=pairwise_matrices(x.to_numpy(),min_pairs=2)
    np.testing.assert_allclose(corr,x.corr().to_numpy(),equal_nan=True,atol=1e-12)
    np.testing.assert_allclose(cov,x.cov().to_numpy(),equal_nan=True,atol=1e-12)
    assert n[0,1]==3


def test_spearman_ranks_target_only_on_each_features_available_pairs():
    d=pd.DataFrame({'a':[1.,np.nan,3.,3.,8.,2.],'b':[np.nan,2.,4.,1.,7.,5.],
        'observed_ret_ld1':[.1,.9,np.nan,.2,.4,-.1]})
    out=monthly_return_stats(d,['a','b'],min_pairs=3).set_index('characteristic')
    for c in ['a','b']:
        sample=d[[c,'observed_ret_ld1']].dropna()
        assert out.loc[c,'rank_ic']==pytest.approx(spearmanr(sample[c],sample.observed_ret_ld1).statistic)
        assert out.loc[c,'pearson']==pytest.approx(sample.corr().iloc[0,1])
        assert out.loc[c,'raw_characteristic_return_covariance']==pytest.approx(sample.cov().iloc[0,1])


def test_constant_features_are_not_fake_predictors():
    n,c,v=paired_columns(np.ones((5,1)),np.arange(5.),2)
    assert np.isnan(c[0])
    assert v[0]==0


def test_source_nullable_values_are_preserved_as_missing_not_zero():
    d=pd.DataFrame({'a':[1.,pd.NA,3.,4.],'observed_ret_ld1':[.1,.9,.3,.4]})
    out=monthly_return_stats(d,['a'],min_pairs=3).iloc[0]
    assert out.n_stocks==3
    assert out.pearson==pytest.approx(1.)
    corr,cov,n=pairwise_matrices(np.array([[1.,2.],[pd.NA,3.],[3.,6.]],dtype=object),min_pairs=2)
    assert n[0,1]==2
    assert corr[0,1]==pytest.approx(1.)


def test_hac_constant_and_known_fdr_values():
    h=hac_mean([.02]*12)
    assert h['mean']==pytest.approx(.02)
    assert h['se']==pytest.approx(0,abs=1e-16)
    np.testing.assert_allclose(bh_adjust([.01,.04,.03,np.nan]),[.03,.04,.04,np.nan],equal_nan=True)


def test_hac_missing_months_preserve_calendar_lags():
    x=np.array([1.,np.nan,-1.,2.,np.nan,0.,3.,-2.])
    h=hac_mean(x);valid=np.isfinite(x);n=valid.sum();z=np.where(valid,x-np.nanmean(x),0.)
    variance=z@z
    for k in range(1,h['lags']+1):variance+=2*(1-k/(h['lags']+1))*sum(z[k:]*z[:-k])
    assert h['se']==pytest.approx(np.sqrt(max(variance,0))/n)
    assert h['se']!=pytest.approx(hac_mean(x[valid])['se'])


def test_diverse_pick_keeps_theme_representation_and_penalizes_duplicate():
    scores=pd.DataFrame(dict(characteristic=['a','b','c','d'],theme=['Value','Value','Risk','Risk'],
        eligible=True,published_jkp_153=True,screen_score=[1.,.99,.8,.7]))
    corr=pd.DataFrame(np.eye(4),index=scores.characteristic,columns=scores.characteristic)
    corr.loc['a','b']=corr.loc['b','a']=1.
    chosen=diverse_pick(scores,corr,3,min_theme=1,score_col='screen_score',redundancy_penalty=.5)
    assert set(chosen.theme)=={'Value','Risk'}
    assert set(chosen.characteristic)=={'a','c','d'}


def test_diverse_pick_never_relaxes_infeasible_theme_caps_silently():
    s=pd.DataFrame(dict(characteristic=['a','b'],theme=['Value','Value'],eligible=True,published_jkp_153=True,screen_score=[1.,.9]))
    c=pd.DataFrame(np.eye(2),index=s.characteristic,columns=s.characteristic)
    with pytest.raises(ValueError,match='infeasible'):
        diverse_pick(s,c,2,min_theme=1,max_theme=1)


def test_final_selection_requires_published_anchor_despite_higher_proxy_score():
    s=pd.DataFrame(dict(characteristic=['official','proxy1','proxy2'],theme=['Accruals']*3,
        eligible=True,published_jkp_153=[True,False,False],screen_score=[.1,.9,.8]))
    c=pd.DataFrame(np.eye(3),index=s.characteristic,columns=s.characteristic)
    chosen=diverse_pick(s,c,2,min_theme=2,published_anchor=True)
    assert set(chosen.characteristic)=={'official','proxy1'}
    s['published_jkp_153']=False
    with pytest.raises(ValueError,match='No eligible published'):
        diverse_pick(s,c,2,min_theme=2,published_anchor=True)


def test_return_dates_not_formation_year_determine_selection_cutoff():
    f=pd.DataFrame({'eom':pd.to_datetime(['2019-10-31','2019-11-30','2019-12-31','2020-01-31']),
        'return_month':pd.to_datetime(['2019-11-30','2019-12-31','2020-01-31','2020-02-29']),
        'observed_ret_ld1':[.1,.2,.3,.4]})
    before=before_test(f)
    mutated=f.copy();mutated.loc[mutated.return_month.ge('2020-01-31'),'observed_ret_ld1']=999
    pd.testing.assert_frame_equal(before,before_test(mutated))
    assert len(before)==2


def test_walk_forward_fit_does_not_use_future_validation_targets():
    rng=np.random.default_rng(123)
    dates=pd.date_range('2004-12-31','2019-11-30',freq='ME')
    f=pd.DataFrame({'eom':np.repeat(dates,8),'a':rng.normal(size=len(dates)*8)})
    f['return_month']=f.eom+pd.offsets.MonthEnd(1)
    f['observed_ret_ld1']=.01*f.a+rng.normal(0,.03,len(f))
    _,first=walk_forward_univariate(f,['a'])
    f.loc[f.return_month.dt.year>=2017,'observed_ret_ld1']*=100
    _,second=walk_forward_univariate(f,['a'])
    pd.testing.assert_frame_equal(first.loc[first.year<2017],second.loc[second.year<2017])
    assert (first.last_training_return_year<first.year).all()


def test_live_schema_registry_retains_published_price_and_excludes_returns():
    from characteristics.acquire import DATA,registry,EXCLUDED_FIELDS
    path=DATA/'source_schema.csv'
    if not path.exists():pytest.skip('Source metadata not downloaded in this checkout')
    r=registry(pd.read_csv(path))
    assert 'prc' in set(r.characteristic)
    assert not set(r.characteristic)&EXCLUDED_FIELDS
    assert r.published_jkp_153.sum()==153
    assert r.characteristic.nunique()==404


def test_frozen_screen_has_200_qualified_names_and_all_13_themes():
    from characteristics.acquire import DATA
    path=DATA/'download_200_selection.csv'
    if not path.exists():pytest.skip('Screen not run in this checkout')
    d=pd.read_csv(path)
    assert len(d)==d.characteristic.nunique()==200
    assert d.theme.nunique()==13
    assert d.minimum_universe_coverage.ge(.5).all()
    assert d.minimum_valid_months.ge(60).all()


def test_lag_download_preserves_unavailable_keys_and_reuses_cache(tmp_path,monkeypatch):
    import characteristics.acquire as acquire
    monkeypatch.setattr(acquire,'KEATS',tmp_path)
    monkeypatch.setattr(acquire,'DATA',tmp_path/'new_data')
    acquire.DATA.mkdir();acquire.write_json(acquire.DATA/'download_200.json',['a'])
    inputs=tmp_path/'data/inputs';inputs.mkdir(parents=True)
    for region in ('US','EU'):
        pd.DataFrame({'id':[1],'eom':pd.to_datetime(['2010-01-31'])}).to_parquet(inputs/f'lag_history_source_{region}.parquet')
        pd.DataFrame({'id':[2],'eom':['2010-01-31']}).to_csv(inputs/f'unavailable_lag_history_{region}.csv',index=False)
    class Source:
        calls=0
        def raw_sql(self,*args,**kwargs):
            self.calls+=1
            return pd.DataFrame({'id':[1],'eom':pd.to_datetime(['2010-01-31']),'a':[.7]})
    source=Source();acquire.download_lag_features(source)
    for region in ('US','EU'):
        d=pd.read_parquet(acquire.DATA/f'lag_history/{region}.parquet').set_index('id')
        assert d.loc[1,'source_row_found'] and d.loc[1,'a']==.7
        assert not d.loc[2,'source_row_found'] and pd.isna(d.loc[2,'a'])
    acquire.download_lag_features(source)
    assert source.calls==2
