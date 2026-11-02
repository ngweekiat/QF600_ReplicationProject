"""Independent scope and statistical checks for the association-only shortlist."""
import numpy as np
import pandas as pd
import pytest

from characteristics.us_high_cap_selection import load_sample, association_scores
from characteristics.statistics import pairwise_matrices
from characteristics.us_high_cap_selection import verified_jkp_registry


def test_jkp_membership_requires_both_published_references(tmp_path):
    reference=tmp_path/'reference';reference.mkdir()
    pd.DataFrame({'characteristic':['verified','cluster_only'],'cluster':['Value','Quality']}).to_parquet(reference/'cluster_labels.parquet',index=False)
    pd.DataFrame({'abr_jkp':['verified','details_only'],'direction':[1,-1],'name_new':['Verified','Other']}).to_parquet(reference/'factor_details.parquet',index=False)
    registry=pd.DataFrame({'characteristic':['verified','cluster_only','details_only','intruder'],
                           'published_theme':['Wrong']*4,'direction':[-1]*4,'name_new':['Wrong']*4})
    selected,_=verified_jkp_registry(registry,data=tmp_path)
    assert selected.characteristic.tolist()==['verified']
    assert selected.published_theme.tolist()==['Value']
    assert selected.direction.tolist()==[1]
    assert selected.jkp_membership_verified.all()


def test_pairwise_matrices_match_pandas_with_missing_observations():
    rng=np.random.default_rng(151)
    values=rng.normal(size=(250,5))
    values[rng.random(values.shape)<.1]=np.nan
    corr,cov,n=pairwise_matrices(values,min_pairs=120)
    expected=pd.DataFrame(values)
    np.testing.assert_allclose(corr,expected.corr(min_periods=120),atol=1e-12)
    np.testing.assert_allclose(cov,expected.cov(min_periods=120),atol=1e-12)
    assert np.array_equal(n,np.isfinite(values).astype(int).T@np.isfinite(values).astype(int))


def test_loader_excludes_us_mid_europe_and_post_2014_before_statistics(tmp_path):
    (tmp_path/'pilot').mkdir()
    dates=pd.to_datetime(['2005-01-31','2005-01-31','2005-01-31','2015-01-31'])
    meta=pd.DataFrame(dict(id=[1,2,3,1],return_month=dates,member_us_large=[True,False,False,True],
        member_us_mid=[False,True,False,False],member_eu_large=[False,False,True,False],observed_ret_ld1=[.1,100.,200.,300.]))
    meta['eom']=meta.return_month-pd.offsets.MonthEnd()
    meta.to_parquet(tmp_path/'metadata_US.parquet',index=False)
    raw=meta[['id','eom']].copy();raw['signal']=[1.,1000.,2000.,3000.]
    raw.to_parquet(tmp_path/'pilot/US_000.parquet',index=False)
    pd.DataFrame({'characteristic':['signal']}).to_csv(tmp_path/'candidate_registry.csv',index=False)
    frame,names,_,_=load_sample(data=tmp_path)
    assert names==['signal'] and frame.id.tolist()==[1] and frame.signal.tolist()==[1.]
    assert frame.return_month.max()==pd.Timestamp('2005-01-31')


def test_scores_refuse_non_us_high_cap_statistics():
    with pytest.raises(ValueError,match='only US high-cap'):
        association_scores(pd.DataFrame({'universe':['us_mid']}))


def test_scores_refuse_test_outcomes():
    with pytest.raises(ValueError,match='Out-of-scope'):
        association_scores(pd.DataFrame({'universe':['us_large'],'return_month':[pd.Timestamp('2020-01-31')]}))
