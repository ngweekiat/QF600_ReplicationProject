"""Pre-2020, cross-universe screening and diversity-aware feature selection."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .acquire import DATA,OUTPUT,UNIVERSES,write_json
from .statistics import panel_metrics,summarize,pairwise_matrices


def load_pilot():
    registry=pd.read_csv(DATA/'candidate_registry.csv')
    features=registry.characteristic.tolist();samples={}
    for region in ('US','EU'):
        data=pd.concat([pd.read_parquet(p) for p in sorted((DATA/'pilot').glob(f'{region}_*.parquet'))],ignore_index=True)
        if (DATA/f'pilot/prc_{region}.parquet').exists():
            extra=pd.read_parquet(DATA/f'pilot/prc_{region}.parquet').rename(columns={'prc':'prc_supplement'})
            data=data.merge(extra,on=['id','eom'],how='left',validate='one_to_one')
            data['prc']=data.get('prc',pd.Series(np.nan,index=data.index)).combine_first(data.prc_supplement)
            data=data.drop(columns='prc_supplement')
        if not set(features).issubset(data):raise ValueError('Pilot candidate data incomplete; do not silently omit a characteristic')
        data[features]=data[features].apply(pd.to_numeric,errors='raise').astype('float64').replace([np.inf,-np.inf],np.nan)
        meta=pd.read_parquet(DATA/f'metadata_{region}.parquet')
        meta=meta.loc[meta.return_month.between('2005-01-31','2014-12-31')]
        for universe in UNIVERSES:
            if (universe.startswith('eu'))!=(region=='EU'):continue
            keys=meta.loc[meta[f'member_{universe}']]
            samples[universe]=keys.merge(data,on=['id','eom'],how='left',validate='one_to_one')
    return registry,features,samples


def equal_universe_score(summary,monthly):
    """No large dataset dominates; signs estimated only from the training window."""
    if set(summary.universe.unique())!=set(UNIVERSES):raise ValueError('All three universe summaries required')
    rows=[]
    yearly=monthly.assign(year=monthly.return_month.dt.year).groupby(['universe','characteristic','year']).rank_ic.mean()
    for char,g in summary.groupby('characteristic'):
        if len(g)!=3:raise ValueError('Incomplete characteristic universe coverage')
        mean=float(g.mean_rank_ic.mean());direction=1 if mean>=0 else -1
        yr=yearly.loc[(slice(None),char,slice(None))].dropna()
        stable=float((yr*direction>0).mean()) if len(yr) else 0.
        coverage=float(g.coverage.min());valid=int(g.valid_ic_months.min())
        rows.append(dict(characteristic=char,balanced_mean_rank_ic=mean,absolute_balanced_ic=abs(mean),
            learned_training_direction=direction,minimum_universe_coverage=coverage,minimum_valid_months=valid,
            directional_year_stability=stable,mean_rank_persistence=float(g.rank_persistence.mean()),
            minimum_universe_ic_in_learned_direction=float((g.mean_rank_ic*direction).min()),
            eligible=(coverage>=.50 and valid>=60)))
    out=pd.DataFrame(rows)
    out['screen_score']=.55*out.absolute_balanced_ic.rank(pct=True)+.20*out.minimum_universe_coverage.rank(pct=True)+.15*out.directional_year_stability.rank(pct=True)+.10*out.mean_rank_persistence.fillna(0).rank(pct=True)
    return out


def balanced_correlations(samples,features):
    corrs=[];covs=[];counts=[]
    for universe,sample in samples.items():
        ranks=sample.groupby('eom')[features].rank(pct=True)
        corr,cov,n=pairwise_matrices(ranks.to_numpy(),min_pairs=120)
        corrs.append(corr);covs.append(cov);counts.append(n)
        folder=OUTPUT/'screening';folder.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(corr,index=features,columns=features).to_csv(folder/f'{universe}_rank_feature_correlation.csv')
        pd.DataFrame(cov,index=features,columns=features).to_csv(folder/f'{universe}_rank_feature_covariance.csv')
    # Equal weight to the 3 requested investment universes; undefined remains NaN.
    count=np.isfinite(corrs).sum(axis=0)
    corr=np.divide(np.nansum(corrs,axis=0),count,out=np.full(np.shape(corrs)[1:],np.nan),where=count>0)
    return pd.DataFrame(corr,index=features,columns=features)


def assign_themes(registry,corr):
    result=registry.copy();groups=result.loc[result.published_theme.notna()].groupby('published_theme').characteristic.apply(list).to_dict()
    themes=[];methods=[];strengths=[]
    for row in result.itertuples():
        if pd.notna(row.published_theme):themes.append(row.published_theme);methods.append('published JKP cluster');strengths.append(np.nan)
        else:
            proxy={theme:float(corr.loc[row.characteristic,chars].abs().mean()) for theme,chars in groups.items()}
            theme=max(proxy,key=lambda k:proxy[k] if np.isfinite(proxy[k]) else -np.inf)
            themes.append(theme);methods.append('empirical training correlation proxy; not published semantic cluster');strengths.append(proxy[theme])
    result['theme']=themes;result['theme_assignment_method']=methods;result['theme_assignment_strength']=strengths
    return result


def diverse_pick(scores,corr,count,min_theme=2,max_theme=None,score_col='screen_score',redundancy_penalty=.15,published_anchor=False):
    """Greedy transparent coverage/diversity selection, not a global-optimality claim."""
    pool=scores.loc[scores.eligible].copy().set_index('characteristic',drop=False)
    if len(pool)<count:raise ValueError(f'Only {len(pool)} coverage-qualified candidates for requested {count}')
    chosen=[];reasons={};theme_counts={}
    def add(char,reason):
        chosen.append(char);theme=pool.loc[char,'theme'];theme_counts[theme]=theme_counts.get(theme,0)+1;reasons[char]=reason
    def best(chars):
        values={}
        for char in chars:
            redundant=float(corr.loc[char,chosen].abs().max()) if chosen else 0.
            if not np.isfinite(redundant):redundant=0.
            # Small documented preference for published interpretable cluster membership.
            values[char]=float(pool.loc[char,score_col])+.02*bool(pool.loc[char,'published_jkp_153'])-redundancy_penalty*redundant
        return max(values,key=lambda c:(values[c],c))
    for theme in sorted(pool.theme.unique()):
        chars=pool.index[pool.theme.eq(theme)].tolist()
        if published_anchor:
            anchors=[c for c in chars if pool.loc[c,'published_jkp_153']]
            if not anchors:raise ValueError(f'No eligible published representative for theme {theme}')
            if len(chosen)>=count:raise ValueError('Theme anchors make requested subset infeasible')
            add(best(anchors),'published JKP representative for economic theme')
        for _ in range(max(0,min(min_theme,len(chars))-theme_counts.get(theme,0))):
            available=[c for c in chars if c not in chosen]
            if not available or len(chosen)>=count:break
            add(best(available),'minimum economic-theme representation')
    while len(chosen)<count:
        available=[c for c in pool.index if c not in chosen and (max_theme is None or theme_counts.get(pool.loc[c,'theme'],0)<max_theme)]
        if not available:raise ValueError('Theme cap makes requested subset infeasible; no silent relaxation')
        add(best(available),'balanced predictive score with redundancy penalty')
    selected=pool.loc[chosen].copy();selected['selection_order']=np.arange(1,len(selected)+1);selected['selection_reason']=[reasons[c] for c in chosen]
    return selected.reset_index(drop=True)


def preselect():
    registry,features,samples=load_pilot()
    print('Computing training-only monthly evidence for',len(features),'candidates',flush=True)
    folder=OUTPUT/'screening';folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'all_candidate_monthly_metrics.parquet'
    if cache.exists():
        monthly=pd.read_parquet(cache)
        if set(monthly.characteristic)!=set(features) or set(monthly.universe)!=set(UNIVERSES) or monthly.return_month.max()>pd.Timestamp('2014-12-31'):
            raise ValueError('Cached screening metrics do not match the candidate set/training cutoff')
        print('Reuse complete training-only monthly metrics.',flush=True)
    else:
        parts=[]
        for universe,sample in samples.items():
            print('Screen',universe,len(sample),'stock-months',flush=True)
            parts.append(panel_metrics(sample,features,universe))
        monthly=pd.concat(parts,ignore_index=True)
    summary=summarize(monthly)
    monthly.to_parquet(folder/'all_candidate_monthly_metrics.parquet',index=False)
    summary.to_csv(folder/'all_candidate_universe_summary.csv',index=False)
    score=equal_universe_score(summary,monthly)
    corr=balanced_correlations(samples,features)
    corr.to_csv(folder/'balanced_training_feature_correlation.csv')
    reg=assign_themes(registry,corr);reg.to_csv(DATA/'candidate_registry_with_themes.csv',index=False)
    score=score.merge(reg,on='characteristic',validate='one_to_one')
    score.sort_values('screen_score',ascending=False).to_csv(folder/'all_candidate_training_scores.csv',index=False)
    n=min(200,int(score.eligible.sum()))
    chosen=diverse_pick(score,corr,n,min_theme=2,redundancy_penalty=.10)
    chosen.to_csv(DATA/'download_200_selection.csv',index=False)
    write_json(DATA/'download_200.json',chosen.characteristic.tolist())
    write_json(folder/'screening_decision.json',dict(candidates=len(features),eligible=int(score.eligible.sum()),retained=n,
        training_only_return_start='2005-01-31',training_only_return_end='2014-12-31',test_outcomes_used=False,
        validation_outcomes_used=False,selection_method='equal-universe score, minimum 2 per theme when available, greedy redundancy penalty',
        caveat='Best means best under this declared screen, not a proven universal optimum.'))
    print(f'PRESELECTED {n} from {len(features)}; coverage eligible {int(score.eligible.sum())}',flush=True)
    return chosen
