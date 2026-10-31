"""Descriptive comparisons of aligned saved portfolios and weights."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import statistics


def correlations(monthly,destination):
    rows=[]
    for dataset,g in monthly.groupby('dataset'):
        folder=destination/'tables'/dataset.replace('/','__')
        folder.mkdir(parents=True,exist_ok=True)
        for kind,value in [('gross',g.r),('net',g.r-g.tc)]:
            wide=g.assign(value=value).pivot(index='eom_ret',columns='strategy',values='value')
            corr=wide.corr()
            corr.to_csv(folder/f'table4_{kind}_correlation.csv')
            for a,b in combinations(corr.index,2):
                rows.append(dict(dataset=dataset,kind=kind,strategy_a=a,strategy_b=b,
                    correlation=corr.loc[a,b],n=len(wide),sample_start=wide.index.min(),sample_end=wide.index.max()))
    return pd.DataFrame(rows)


def weight_similarity(inventory,monthly):
    cache={}; rows=[]; skips=[]
    for dataset,inv in inventory.groupby('dataset'):
        weights={}
        for row in inv.itertuples():
            p=Path(row.weights_path) if row.weights_path else None
            if p is None or not p.exists():
                skips.append(dict(dataset=dataset,strategy=row.strategy,status='SKIPPED',reason='No saved portfolio weights.'))
                continue
            if str(p) not in cache:cache[str(p)]=pd.read_parquet(p)
            w=cache[str(p)]
            if 'type' in w:w=w.loc[w.type.eq(row.source_type)]
            if w.empty:
                skips.append(dict(dataset=dataset,strategy=row.strategy,status='SKIPPED',reason='Performance series has no corresponding saved weight type.'))
                continue
            dates=monthly.loc[(monthly.dataset==dataset)&(monthly.strategy==row.strategy),'eom']
            w=w.loc[w.eom.isin(dates)].pivot(index='eom',columns='id',values='w').fillna(0.)
            weights[row.strategy]=w
        for a,b in combinations(weights,2):
            wa,wb=weights[a],weights[b]
            dates=wa.index.intersection(wb.index); ids=wa.columns.union(wb.columns)
            x=wa.reindex(index=dates,columns=ids,fill_value=0.).to_numpy()
            y=wb.reindex(index=dates,columns=ids,fill_value=0.).to_numpy()
            denom=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
            cosine=np.divide(np.sum(x*y,axis=1),denom,out=np.full(len(dates),np.nan),where=denom>0)
            rows.append(dict(dataset=dataset,strategy_a=a,strategy_b=b,n_months=len(dates),
                sample_start=dates.min(),sample_end=dates.max(),mean_cosine=float(np.nanmean(cosine)),
                mean_l1_weight_difference=float(np.abs(x-y).sum(axis=1).mean())))
    return pd.DataFrame(rows),pd.DataFrame(skips)


def regimes(root,monthly,gamma=10.):
    """Ex-post conditioning only; local Market sign and fixed VIX 20 threshold."""
    vix_path=root/'keats_pipeline/data/inputs/fred_VIXCLS.parquet'
    vix=pd.read_parquet(vix_path) if vix_path.exists() else pd.DataFrame()
    if not vix.empty:
        vix['eom']=pd.to_datetime(vix.date)+pd.offsets.MonthEnd(0)
        vix=vix.groupby('eom').value.mean()
    rec_path=root/'keats_pipeline/data/inputs/fred_USREC.parquet'
    rec=pd.read_parquet(rec_path) if rec_path.exists() else pd.DataFrame()
    if not rec.empty:
        rec['eom_ret']=pd.to_datetime(rec.date)+pd.offsets.MonthEnd(0)
        rec=rec.groupby('eom_ret').value.last()
    rows=[]
    for dataset,d in monthly.groupby('dataset'):
        market=d.loc[d.strategy=='Market'].set_index('eom_ret').r
        for strategy,g in d.groupby('strategy'):
            labels={'calendar_half':np.where(g.eom_ret.dt.year<=2022,'2020–2022','2023–2025'),
                    'calendar_stress':np.where(g.eom_ret.between('2020-02-01','2020-04-30'),'COVID Feb–Apr 2020',
                        np.where(g.eom_ret.dt.year==2022,'Calendar 2022','Other months'))}
            if len(market):
                local=g.eom_ret.map(market)
                labels['contemporaneous_local_market_sign']=pd.Series(np.where(local>=0,'Nonnegative','Negative'),index=g.index).where(local.notna())
            if not vix.empty:
                vx=g.eom.map(vix)
                labels['prior_month_US_VIX_20']=pd.Series(np.where(vx>=20,'VIX >=20','VIX <20'),index=g.index).where(vx.notna())
            if not rec.empty:
                rc=g.eom_ret.map(rec)
                labels['US_recession_context']=pd.Series(np.where(rc>=.5,'US recession','US expansion'),index=g.index).where(rc.notna())
            for dimension,values in labels.items():
                for label,part in g.assign(regime=values).dropna(subset=['regime']).groupby('regime'):
                    if len(part)>=2:
                        rows.append(dict(dataset=dataset,strategy=strategy,dimension=dimension,regime=label,
                            conditioning='descriptive, ex post; US indicators are context for Europe',**statistics(part,gamma)))
    return pd.DataFrame(rows)


def performance_decomposition(table):
    """Exact differences: net-return advantage = gross advantage + cost saving."""
    rows=[]
    for dataset,g in table.groupby('dataset'):
        for a,b in combinations(g.itertuples(),2):
            rows.append(dict(dataset=dataset,strategy_a=a.strategy,strategy_b=b.strategy,
                gross_return_difference=a.R-b.R,cost_saving=b.TC-a.TC,
                net_return_difference=a.R_TC-b.R_TC,utility_difference=a.Utility-b.Utility,
                turnover_difference=a.Turnover-b.Turnover,
                risk_penalty_difference=(a.R_TC-a.Utility)-(b.R_TC-b.Utility),
                interpretation='Accounting decomposition; not causal attribution.'))
    return pd.DataFrame(rows)


def cross_dataset(table):
    rows=[]
    for strategy,g in table.groupby('strategy'):
        if len(g)<2:continue
        for a,b in combinations(g.itertuples(),2):
            rows.append(dict(strategy=strategy,dataset_a=a.dataset,dataset_b=b.dataset,
                n_a=a.n,n_b=b.n,sample_start=a.sample_start,sample_end=a.sample_end,
                net_return_difference=a.R_TC-b.R_TC,utility_difference=a.Utility-b.Utility,
                net_sharpe_difference=a.SR_net-b.SR_net,
                inference='Descriptive matched-calendar comparison; region, membership, liquidity, currency and fitted model may differ. No causal isolation.'))
    return pd.DataFrame(rows)
