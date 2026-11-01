"""Transparent predictive diagnostics with pairwise missing-data handling."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def numeric_array(values):
    a=np.asarray(values)
    if a.dtype.kind=='O':
        return pd.to_numeric(pd.Series(a.ravel()),errors='raise').to_numpy(dtype=float,na_value=np.nan).reshape(a.shape)
    return np.asarray(a,float)


def paired_columns(x,y,min_pairs=20):
    """Pearson and covariance for each column with its own valid target pairs."""
    x=numeric_array(x);y=numeric_array(y)
    if y.ndim==1:y=np.broadcast_to(y[:,None],x.shape)
    if x.shape!=y.shape:raise ValueError('Target matrix and features do not align')
    m=np.isfinite(x)&np.isfinite(y)
    n=m.sum(axis=0)
    a=np.where(m,x,0.);b=np.where(m,y,0.)
    sx=a.sum(axis=0);sy=b.sum(axis=0)
    den=np.maximum(n,1)
    cross=(a*b).sum(axis=0)-sx*sy/den
    xx=(a*a).sum(axis=0)-sx*sx/den
    yy=(b*b).sum(axis=0)-sy*sy/den
    corr=np.divide(cross,np.sqrt(np.maximum(xx,0)*np.maximum(yy,0)),out=np.full(x.shape[1],np.nan),where=(xx>0)&(yy>0)&(n>=min_pairs))
    cov=np.divide(cross,n-1,out=np.full(x.shape[1],np.nan),where=n>=max(2,min_pairs))
    return n,corr,cov


def monthly_return_stats(frame,features,min_pairs=20):
    """Exact pairwise Spearman ranks and raw/rank-return covariance, no imputation."""
    x=frame[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    y=pd.to_numeric(frame.observed_ret_ld1,errors='coerce').replace([np.inf,-np.inf],np.nan)
    n,pearson,cov=paired_columns(x.to_numpy(dtype=float,na_value=np.nan),y.to_numpy(dtype=float,na_value=np.nan),min_pairs)
    joint=x.notna().to_numpy()&y.notna().to_numpy()[:,None]
    a=x.where(joint).rank(method='average',pct=True)
    b=pd.DataFrame(np.broadcast_to(y.to_numpy()[:,None],x.shape),index=x.index,columns=x.columns).where(joint).rank(method='average',pct=True)
    _,ic,rankcov=paired_columns(a.to_numpy(),b.to_numpy(),min_pairs)
    _,_,rank_ycov=paired_columns(a.to_numpy(),y.to_numpy(),min_pairs)
    top=a.ge(.8).to_numpy()&joint;bottom=a.le(.2).to_numpy()&joint
    yt=np.broadcast_to(y.to_numpy()[:,None],x.shape)
    nt=top.sum(axis=0);nb=bottom.sum(axis=0)
    high=np.divide(np.where(top,yt,0.).sum(axis=0),nt,out=np.full(len(features),np.nan),where=nt>0)
    low=np.divide(np.where(bottom,yt,0.).sum(axis=0),nb,out=np.full(len(features),np.nan),where=nb>0)
    spread=high-low;spread[n<min_pairs]=np.nan
    return pd.DataFrame(dict(characteristic=features,n_stocks=n,pearson=pearson,rank_ic=ic,
        raw_characteristic_return_covariance=cov,rank_rank_covariance=rankcov,
        ranked_characteristic_return_covariance=rank_ycov,gross_quintile_spread=spread,
        observed_feature_count=x.notna().sum().to_numpy(),population_size=len(frame)))


def hac_mean(values):
    x=np.asarray(values,float);valid=np.isfinite(x);n=int(valid.sum())
    if n<3:return dict(n=n,mean=np.nan,se=np.nan,t=np.nan,p=np.nan,ci_low=np.nan,ci_high=np.nan,lags=0)
    mu=x[valid].mean();z=np.where(valid,x-mu,0.)
    lags=min(len(x)-1,int(np.floor(4*(n/100)**(2/9))))
    # Missing ICs retain their calendar positions: zero score contributions do
    # not impute outcomes, and two separated months never become adjacent.
    variance=float(z@z)
    for k in range(1,lags+1):variance+=2*(1-k/(lags+1))*float(z[k:]@z[:-k])
    se=np.sqrt(max(variance,0))/n
    t=mu/se if se>0 else (0. if mu==0 else np.sign(mu)*np.inf)
    return dict(n=n,mean=mu,se=se,t=t,p=float(2*norm.sf(abs(t))),
        ci_low=mu-1.95996398454*se,ci_high=mu+1.95996398454*se,lags=lags)


def bh_adjust(values):
    p=np.asarray(values,float);result=np.full(len(p),np.nan)
    ix=np.flatnonzero(np.isfinite(p));order=ix[np.argsort(p[ix])]
    if not len(order):return result
    v=p[order]*len(order)/np.arange(1,len(order)+1)
    result[order]=np.minimum(1.,np.minimum.accumulate(v[::-1])[::-1]);return result


def pairwise_matrices(x,min_pairs=24):
    """Pairwise complete covariance/correlation; no zero filling of observed data."""
    a=numeric_array(x);m=np.isfinite(a).astype(float);v=np.where(np.isfinite(a),a,0.)
    n=m.T@m;s=v.T@m;q=(v*v).T@m
    denom=np.maximum(n,1.)
    cross=v.T@v-s*s.T/denom
    variance=q-s*s/denom
    corr=np.divide(cross,np.sqrt(np.maximum(variance,0.)*np.maximum(variance.T,0.)),
        out=np.full(n.shape,np.nan),where=(variance>0)&(variance.T>0)&(n>=min_pairs))
    cov=np.divide(cross,n-1,out=np.full(n.shape,np.nan),where=n>=min_pairs)
    return corr,cov,n


def panel_metrics(panel,features,universe,min_pairs=20,prior_state=None,return_state=False):
    rows=[];prior,prior_date=(None,None) if prior_state is None else prior_state
    for date,g in panel.groupby('eom',sort=True):
        result=monthly_return_stats(g,features,min_pairs)
        result['universe']=universe;result['formation_month']=date;result['return_month']=date+pd.offsets.MonthEnd(1)
        ranks=g.set_index('id')[features].replace([np.inf,-np.inf],np.nan).rank(pct=True)
        result['rank_persistence']=np.nan;result['mean_abs_rank_change']=np.nan
        if prior is not None and date==prior_date+pd.offsets.MonthEnd(1):
            ids=ranks.index.intersection(prior.index)
            _,persist,_=paired_columns(ranks.loc[ids].to_numpy(),prior.loc[ids].to_numpy(),min_pairs)
            result['rank_persistence']=persist
            changes=np.abs(ranks.loc[ids].to_numpy()-prior.loc[ids].to_numpy())
            count=np.isfinite(changes).sum(axis=0)
            result['mean_abs_rank_change']=np.divide(np.nansum(changes,axis=0),count,
                out=np.full(len(features),np.nan),where=count>0)
        prior,prior_date=ranks,date;rows.append(result)
    out=pd.concat(rows,ignore_index=True)
    return (out,(prior,prior_date)) if return_state else out


def summarize(monthly):
    rows=[]
    for (universe,char),g in monthly.groupby(['universe','characteristic']):
        ic=hac_mean(g.rank_ic)
        rows.append(dict(universe=universe,characteristic=char,months=len(g),valid_ic_months=ic['n'],
            coverage=g.observed_feature_count.sum()/g.population_size.sum(),
            mean_rank_ic=ic['mean'],ic_hac_t=ic['t'],ic_hac_se=ic['se'],ic_hac_p=ic['p'],
            ic_ci_low=ic['ci_low'],ic_ci_high=ic['ci_high'],mean_pearson=g.pearson.mean(),
            mean_raw_return_covariance=g.raw_characteristic_return_covariance.mean(),
            mean_rank_return_covariance=g.ranked_characteristic_return_covariance.mean(),
            mean_gross_spread_annual=12*g.gross_quintile_spread.mean(),
            rank_persistence=g.rank_persistence.mean(),rank_change_proxy=g.mean_abs_rank_change.mean(),
            sample_start=g.return_month.min(),sample_end=g.return_month.max()))
    out=pd.DataFrame(rows)
    out['ic_hac_p_bh']=out.groupby('universe').ic_hac_p.transform(lambda v:bh_adjust(v.to_numpy()))
    return out
