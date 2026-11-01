"""Offline validation, per-stock diagnostics and a frozen common set of 50."""
from __future__ import annotations

import html
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .acquire import DATA,OUTPUT,KEATS,UNIVERSES,digest,write_json
from .statistics import panel_metrics,summarize,paired_columns,pairwise_matrices
from .selection import diverse_pick


def before_test(frame):
    """All selection uses realised return months; December 2019 formation is test."""
    return frame.loc[frame.return_month.between('2005-01-31','2019-12-31')].copy()


def collect(features,wait=False):
    """Stream regional partitions; never load all broad 200-feature panels at once."""
    holdings={u:[] for u in UNIVERSES};broad=[];integrity=[]
    for region in ['US','EU']:
        metadata=pd.read_parquet(DATA/f'metadata_{region}.parquet')
        state=None
        for year in sorted(metadata.eom.dt.year.unique()):
            path=DATA/f'panels/{region}_{year}.parquet'
            if not path.exists() and wait:print(f'Waiting for acquisition: {region} {year}',flush=True)
            while not path.exists() and wait:time.sleep(5)
            if not path.exists():raise ValueError(f'Full download incomplete: {region} {year}')
            keys=metadata.loc[metadata.eom.dt.year.eq(year)]
            # Count all original dates, including warmup/test, without using test outcomes.
            import pyarrow.parquet as pq
            filemeta=pq.read_metadata(path)
            if filemeta.num_rows!=len(keys):raise AssertionError('Download row count differs from original panel')
            integrity.append(dict(region=region,year=year,download_rows=filemeta.num_rows,original_rows=len(keys),
                file=str(path.relative_to(DATA)),test_values_used_for_selection=False))
            eligible=before_test(keys)
            if eligible.empty:continue
            raw=pd.read_parquet(path)
            raw[features]=raw[features].apply(pd.to_numeric,errors='raise').astype('float64').replace([np.inf,-np.inf],np.nan)
            if raw.duplicated(['id','eom']).any():raise ValueError('Duplicate downloaded stock-month')
            frame=eligible.merge(raw,on=['id','eom'],how='left',validate='one_to_one')
            if len(frame)!=len(eligible):raise AssertionError('Selection join changed original panel keys')
            stat,state=panel_metrics(frame,features,f'broad_{region}',prior_state=state,return_state=True)
            broad.append(stat)
            for u in UNIVERSES:
                if (u.startswith('eu'))==(region=='EU'):
                    holdings[u].append(frame.loc[frame[f'member_{u}']].copy())
            print(f'Analysed broad {region} {year}: {len(frame):,} pre-test stock-months',flush=True)
    for region in ['US','EU']:
        expected=set(pd.read_parquet(DATA/f'metadata_{region}.parquet',columns=['eom']).eom.dt.year)
        got={r['year'] for r in integrity if r['region']==region}
        if expected!=got:raise ValueError(f'Full download incomplete for {region}: missing years {sorted(expected-got)}')
    hold={u:pd.concat(parts,ignore_index=True) for u,parts in holdings.items()}
    return hold,pd.concat(broad,ignore_index=True),pd.DataFrame(integrity)


def walk_forward_univariate(frame,features):
    """One-feature linear forecasts of observed returns; no existing model refit.

    Each validation year is fit only to earlier realised-return years. Different
    features use their own observed feature/target pairs; counts are explicit.
    This is a predictive diagnostic, not Static-ML or Portfolio-ML performance.
    """
    ranks=frame.groupby('eom')[features].rank(pct=True).to_numpy(float)-.5
    y=frame.observed_ret_ld1.to_numpy(float);years=frame.return_month.dt.year.to_numpy()
    sse=np.zeros(len(features));null_sse=np.zeros(len(features));counts=np.zeros(len(features),int);records=[]
    for year in range(2015,2020):
        train=(years<year)&(years>=2005);val=years==year
        x=ranks[train];target=y[train]
        valid=np.isfinite(x)&np.isfinite(target)[:,None]
        n=valid.sum(axis=0);a=np.where(valid,x,0.);b=np.where(valid,target[:,None],0.)
        nn=np.maximum(n,1);sx=a.sum(axis=0);sy=b.sum(axis=0)
        xx=(a*a).sum(axis=0)-sx*sx/nn;xy=(a*b).sum(axis=0)-sx*sy/nn
        beta=np.divide(xy,xx+1e-8,out=np.zeros(len(features)),where=(n>=60)&(xx>1e-12))
        intercept=sy/nn-beta*sx/nn
        xv=ranks[val];yv=y[val,None]
        observed=np.isfinite(xv)&np.isfinite(yv)&(n>=60)[None,:]
        pred=intercept[None,:]+xv*beta[None,:]
        err=np.where(observed,(yv-pred)**2,0.).sum(axis=0)
        base=np.where(observed,(yv-sy[None,:]/nn[None,:])**2,0.).sum(axis=0)
        count=observed.sum(axis=0);sse+=err;null_sse+=base;counts+=count
        for j,c in enumerate(features):records.append(dict(year=year,characteristic=c,n_train=int(n[j]),n_validation=int(count[j]),
            last_training_return_year=year-1,validation_r2=(1-err[j]/base[j]) if base[j]>0 else np.nan,
            slope=float(beta[j]),sse=float(err[j]),historical_mean_benchmark_sse=float(base[j])))
    score=np.divide(sse,null_sse,out=np.full(len(features),np.nan),where=null_sse>0)
    return pd.DataFrame(dict(characteristic=features,walk_forward_validation_r2=1-score,validation_pairs=counts)),pd.DataFrame(records)


def per_stock(frame,features,universe):
    rows=[]
    for stock,g in frame.groupby('id',sort=True):
        x=g[features].replace([np.inf,-np.inf],np.nan);y=g.observed_ret_ld1.to_numpy(float)
        n,pearson,cov=paired_columns(x.to_numpy(),y,min_pairs=24)
        joint=x.notna().to_numpy()&np.isfinite(y)[:,None]
        xr=x.where(joint).rank(pct=True)
        yr=pd.DataFrame(np.broadcast_to(y[:,None],x.shape),index=x.index,columns=x.columns).where(joint).rank(pct=True)
        _,spearman,_=paired_columns(xr.to_numpy(),yr.to_numpy(),min_pairs=24)
        rows.append(pd.DataFrame(dict(universe=universe,id=stock,characteristic=features,n_observed_pairs=n,
            temporal_pearson=pearson,temporal_spearman=spearman,raw_return_covariance=cov,
            sample_start=g.return_month.min(),sample_end=g.return_month.max(),
            scope='pre-2020 held-month time-series association, minimum 24 pairs; not causal or a fitted ML forecast')))
    return pd.concat(rows,ignore_index=True)


def final_scores(train,val,monthly,diagnostics,registry,screen):
    rows=[]
    dscore=screen.set_index('characteristic')
    yearly=monthly.loc[monthly.return_month.dt.year.ge(2015)].assign(year=lambda x:x.return_month.dt.year).groupby(['universe','characteristic','year']).rank_ic.mean()
    for char,v in val.groupby('characteristic'):
        tr=train.loc[train.characteristic.eq(char)]
        direction=int(dscore.loc[char,'learned_training_direction'])
        yearly_ic=yearly.loc[(slice(None),char,slice(None))].dropna()
        r2=diagnostics.loc[diagnostics.characteristic.eq(char)].walk_forward_validation_r2
        rows.append(dict(characteristic=char,training_direction=direction,
            training_mean_rank_ic=tr.mean_rank_ic.mean(),validation_mean_rank_ic=v.mean_rank_ic.mean(),
            validation_oriented_ic=direction*v.mean_rank_ic.mean(),
            minimum_universe_oriented_validation_ic=float((direction*v.mean_rank_ic).min()),
            minimum_training_coverage=tr.coverage.min(),minimum_validation_coverage=v.coverage.min(),
            validation_direction_stability=float((direction*yearly_ic>0).mean()),
            mean_persistence=pd.concat([tr.rank_persistence,v.rank_persistence]).mean(),
            mean_walk_forward_r2=r2.mean(),eligible=bool(tr.coverage.min()>=.6 and v.coverage.min()>=.6 and
                tr.valid_ic_months.min()>=60 and v.valid_ic_months.min()>=30)))
    s=pd.DataFrame(rows)
    s['final_score']=(.35*s.validation_oriented_ic.rank(pct=True)+.15*s.minimum_universe_oriented_validation_ic.rank(pct=True)
        +.15*s.minimum_validation_coverage.rank(pct=True)+.10*s.validation_direction_stability.rank(pct=True)
        +.10*s.mean_persistence.fillna(0).rank(pct=True)+.15*s.mean_walk_forward_r2.fillna(-np.inf).rank(pct=True))
    return s.merge(registry,on='characteristic',validate='one_to_one')


def selected_input_panels(features):
    """Paper-style rank inputs for later explicit refitting; never replaces current data."""
    coverage=[]
    for path in sorted((DATA/'panels').glob('*.parquet')):
        destination=DATA/'selected50_ranked'/path.name
        d=pd.read_parquet(path,columns=['id','eom',*features])
        d[features]=d[features].apply(pd.to_numeric,errors='raise').astype('float64').replace([np.inf,-np.inf],np.nan)
        for c in features:coverage.append(dict(region=path.stem.split('_')[0],formation_year=int(path.stem.split('_')[1]),
            characteristic=c,observed=int(d[c].notna().sum()),stock_months=len(d),coverage=float(d[c].notna().mean()),
            used_to_reselect_features=False))
        if destination.exists():
            import pyarrow.parquet as pq
            present=set(pq.read_schema(destination).names)
            if not set(features).issubset(present):raise ValueError('Existing ranked export has a different recommendation; retained without overwriting')
            continue
        observed=d[features].notna().sum(axis=1)
        ranks=d.groupby('eom')[features].rank(pct=True).fillna(.5)
        out=pd.concat([d[['id','eom']],ranks],axis=1)
        out['observed_characteristic_count']=observed
        out['at_least_half_features_observed']=observed.ge(25)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.part.parquet');out.to_parquet(temporary,index=False,compression='zstd');temporary.replace(destination)
    pd.DataFrame(coverage).to_csv(OUTPUT/'selected50_full_period_feature_coverage.csv',index=False)
    print('Exported separate paper-style ranked 50-feature inputs for all saved dates.',flush=True)


def analyse(wait=False):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    features=json.loads((DATA/'download_200.json').read_text())
    registry=pd.read_csv(DATA/'candidate_registry_with_themes.csv').loc[lambda x:x.characteristic.isin(features)]
    screen=pd.read_csv(OUTPUT/'screening/all_candidate_training_scores.csv')
    hold,broad,checks=collect(features,wait)
    checks.to_csv(OUTPUT/'stock_date_integrity.csv',index=False)
    broad.to_parquet(OUTPUT/'broad_regional_monthly_diagnostics.parquet',index=False)
    summarize(broad).to_csv(OUTPUT/'broad_regional_characteristic_summary.csv',index=False)
    monthly=[];one=[];stock=[];corrs=[]
    for universe,frame in hold.items():
        if frame.return_month.max()>pd.Timestamp('2019-12-31'):raise AssertionError('Test leakage into analysis')
        m=panel_metrics(frame,features,universe);monthly.append(m)
        diag,annual=walk_forward_univariate(frame,features);diag['universe']=universe;one.append(diag)
        annual['universe']=universe;annual.to_csv(OUTPUT/f'{universe}_annual_univariate_forecasts.csv',index=False)
        stock.append(per_stock(frame,features,universe))
        ranks=frame.groupby('eom')[features].rank(pct=True)
        corr,cov,n=pairwise_matrices(ranks.to_numpy(),min_pairs=120)
        raw_corr,raw_cov,_=pairwise_matrices(frame[features].to_numpy(),min_pairs=120)
        for name,array in [('rank_correlation',corr),('rank_covariance',cov),('raw_correlation',raw_corr),('raw_covariance',raw_cov),('pair_counts',n)]:
            pd.DataFrame(array,index=features,columns=features).to_csv(OUTPUT/f'{universe}_feature_{name}.csv')
        corrs.append(corr)
    monthly=pd.concat(monthly,ignore_index=True);diag=pd.concat(one,ignore_index=True)
    monthly.to_parquet(OUTPUT/'all_universe_monthly_metrics.parquet',index=False)
    diag.to_csv(OUTPUT/'univariate_walk_forward_validation.csv',index=False)
    pd.concat(stock,ignore_index=True).to_parquet(OUTPUT/'per_stock_per_characteristic.parquet',index=False)
    training=summarize(monthly.loc[monthly.return_month.dt.year.le(2014)])
    validation=summarize(monthly.loc[monthly.return_month.dt.year.ge(2015)])
    training.to_csv(OUTPUT/'training_characteristic_results.csv',index=False)
    validation.to_csv(OUTPUT/'validation_characteristic_results.csv',index=False)
    n=np.isfinite(corrs).sum(axis=0)
    matrix=np.divide(np.nansum(corrs,axis=0),n,out=np.full(np.shape(corrs)[1:],np.nan),where=n>0)
    corr=pd.DataFrame(matrix,index=features,columns=features)
    corr.to_csv(OUTPUT/'balanced_pretest_rank_correlation.csv')
    scores=final_scores(training,validation,monthly,diag,registry,screen)
    scores.sort_values('final_score',ascending=False).to_csv(OUTPUT/'all_capped_characteristic_scores.csv',index=False)
    selected=diverse_pick(scores,corr,50,min_theme=2,max_theme=6,score_col='final_score',redundancy_penalty=.20,published_anchor=True)
    selected.to_csv(OUTPUT/'recommended_50.csv',index=False)
    write_json(OUTPUT/'recommended_50.json',selected.characteristic.tolist())
    write_json(DATA/'recommended_50.json',selected.characteristic.tolist())
    themes={theme.lower().replace(' ','_').replace('-','_'):g.characteristic.tolist() for theme,g in selected.groupby('theme')}
    write_json(OUTPUT/'proposed_feature_configuration.json',dict(features=themes,
        applied_to_existing_models=False,selection_return_cutoff='2019-12-31',
        caution='Empirical proxy theme assignments and learned directions for nonpublished characteristics require the supplied mapping; not silently activated in current configs.'))
    selected[['characteristic','theme','theme_assignment_method','direction','training_direction']].to_csv(OUTPUT/'selected_theme_directions.csv',index=False)
    original=json.loads((KEATS/'feature_names.json').read_text())
    pd.DataFrame(dict(characteristic=sorted(set(original)|set(selected.characteristic)))).assign(
        original_50=lambda x:x.characteristic.isin(original),recommended_50=lambda x:x.characteristic.isin(selected.characteristic)).to_csv(OUTPUT/'original_vs_recommended_50.csv',index=False)
    # Honest redundancy ledger rather than suppressing highly similar picks.
    pairs=[]
    for i,a in enumerate(selected.characteristic):
        for b in selected.characteristic.iloc[i+1:]:pairs.append(dict(characteristic_a=a,characteristic_b=b,rank_correlation=corr.loc[a,b]))
    pd.DataFrame(pairs).assign(absolute_correlation=lambda x:x.rank_correlation.abs()).sort_values('absolute_correlation',ascending=False).to_csv(OUTPUT/'selected_feature_redundancy.csv',index=False)
    selected_input_panels(selected.characteristic.tolist())
    from .verification import verify
    verify(hold,features,monthly)
    from .report import write_report
    write_report(selected,scores,validation,diag,checks)
    print('CHARACTERISTIC ANALYSIS COMPLETE; exactly 50 recommended, no test return used.',flush=True)
    return selected
