"""Iteration 7: causal idiosyncratic-risk EWMA tuning with pretest-only selection.

Reuse original factor returns, exposures and factor covariance. Reconstruct daily
residuals, independently match the original 126-day variances, then vary only
idiosyncratic EWMA half-life. All data and previous portfolios remain read-only.
"""
import gc
import hashlib
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from analysis.ml_comparison.metrics import statistics
from jkmp.covariance import CovParts,idiosyncratic_vol,monthly_loadings,save_covariance
from jkmp.portfolio.active import calibrate
from selective_trading import save_result
from workflow import ROOT

log=logging.getLogger(__name__)
CACHE=ROOT/'data/processed/us_large/iteration_7_covariance'


def residuals_from_saved_factors(daily,loadings,factors,rf):
    """Known daily factor returns and previous-month exposures; no new regression."""
    names=[c for c in factors if c not in ('n_obs','r2')]
    rows=[]
    for year,frame in daily.groupby(daily.date.dt.year,sort=True):
        frame=frame.assign(eom=(frame.date.dt.to_period('M')-1).dt.end_time.dt.normalize())
        left=frame.eom.min();right=frame.eom.max()
        L=loadings.loc[loadings.eom.between(left,right),['id','eom',*names]]
        d=frame.merge(L,on=['id','eom'],how='inner',validate='many_to_one').dropna(subset=['ret',*names])
        d=d.loc[d.date.isin(factors.index)&d.date.isin(rf.index)]
        X=d[names].to_numpy();F=factors.loc[d.date,names].to_numpy()
        res=d.ret.to_numpy()-rf.loc[d.date].to_numpy()-np.einsum('ij,ij->i',X,F)
        rows.append(pd.DataFrame(dict(id=d.id.to_numpy(),date=d.date.to_numpy(),res=res)))
        log.info('Reconstructed causal residuals %s: %d rows',year,len(d))
    return pd.concat(rows,ignore_index=True)


def assemble_variances(vol,loadings,base,days_per_month):
    """Retain original pool/size-group imputation and all original X/F/stock IDs."""
    by_vol={d:g.set_index('id').res_vol for d,g in vol.groupby('eom')}
    by_load={d:g.set_index('id') for d,g in loadings.groupby('eom')}
    result={}
    for eom,part in base.items():
        L=by_load[eom]
        pool=L.loc[L.in_top|L.in_universe,['size_grp']].join(by_vol.get(eom,pd.Series(dtype=float,name='res_vol')))
        fill=pool.groupby('size_grp').res_vol.transform('median').fillna(pool.res_vol.median())
        variance=pool.res_vol.fillna(fill).loc[part.ids].to_numpy()**2*days_per_month
        if not np.isfinite(variance).all() or (variance<0).any():raise ValueError('Invalid idiosyncratic variances')
        result[eom]=CovParts(eom,part.ids.copy(),part.factors,part.X.copy(),part.F.copy(),variance)
    return result


def prepare_risk(engine,folder,signature,candidates):
    from iteration_sequence import digest,write_json
    CACHE.mkdir(parents=True,exist_ok=True)
    marker=CACHE/'provenance.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('signature')==signature and all(p.is_file() and digest(p)==h for p,h in
            [(CACHE/name,value) for name,value in old.get('hashes',{}).items()]):
            log.info('Reuse independently verified risk candidates')
            shutil.copy2(CACHE/'control_verification.json',folder/'risk_control_verification.json')
            return
    cfg=engine.cfg;p=cfg.paths.processed
    raw=ROOT/'data/inputs/daily_US.parquet'
    top=pd.read_parquet(cfg.paths.raw/'top500_by_month.parquet')
    themes=[n for n in pq.ParquetFile(p/'panel.parquet').schema.names if n.startswith('theme_')]
    columns=['id','eom','size_grp','ff12','in_universe',*themes]
    panel=pd.read_parquet(p/'panel.parquet',columns=columns)
    needed=sorted(set(top.id)|set(panel.loc[panel.in_universe,'id']))
    daily=pd.read_parquet(raw,columns=['id','date','ret'],filters=[('id','in',needed)])
    daily['date']=pd.to_datetime(daily.date)
    L=monthly_loadings(panel,top,daily.id.unique(),cfg.covariance.sectors)
    del panel;gc.collect()
    factors=pd.read_parquet(p/'covariance/factor_returns.parquet')
    if 'date' in factors:factors=factors.set_index('date')
    rf=pd.read_parquet(cfg.paths.raw/'ff_factors_daily.parquet').set_index('date').rf
    residual=residuals_from_saved_factors(daily,L,factors,rf)
    del daily;gc.collect()
    c=cfg.covariance
    checks=[];hashes={}
    for half_life in candidates:
        log.info('Build idiosyncratic risk half-life %s days',half_life)
        vol=idiosyncratic_vol(residual,factors.index,half_life,c.idio_initial_obs,c.idio_min_obs,c.idio_min_window)
        parts=assemble_variances(vol,L,engine.cov,c.days_per_month)
        maximum=0.
        for d,part in parts.items():
            original=engine.cov[d]
            if not (np.array_equal(part.ids,original.ids) and np.array_equal(part.X,original.X)
                    and np.array_equal(part.F,original.F)):raise AssertionError('Non-idiosyncratic risk changed')
            if half_life==c.half_life_idio_days:
                np.testing.assert_allclose(part.ivar,original.ivar,rtol=1e-9,atol=1e-12)
                maximum=max(maximum,float(np.max(np.abs(part.ivar-original.ivar))))
        target=CACHE/f'hl_{half_life}'
        if half_life==c.half_life_idio_days:
            # Use bitwise original control after independently matching its reconstruction.
            target.mkdir(exist_ok=True)
            for name in ('cov_factor.parquet','cov_loadings.parquet'):shutil.copy2(p/'covariance'/name,target/name)
        else:save_covariance(parts,target)
        for name in ('cov_factor.parquet','cov_loadings.parquet'):
            path=target/name;hashes[path.relative_to(CACHE).as_posix()]=digest(path)
        checks.append(dict(half_life_days=half_life,months=len(parts),same_ids_exposures_factor_covariance=True,
                           original_control_matched=half_life==c.half_life_idio_days,max_control_abs_error=maximum))
        del vol,parts;gc.collect()
    write_json(CACHE/'control_verification.json',dict(status='PASS',checks=checks,
        residual_source='original saved daily factor returns and lagged monthly loadings',
        covariance_window_days=c.window_days,correlation_half_life=c.half_life_corr_days,factor_variance_half_life=c.half_life_var_days))
    write_json(marker,dict(signature=signature,hashes=hashes))
    shutil.copy2(CACHE/'control_verification.json',folder/'risk_control_verification.json')


def choose_risk(rows,control=126):
    table=pd.DataFrame(rows)
    if not np.isfinite(table[['half_life_days','validation_utility']]).all().all():raise ValueError('Invalid pretest scores')
    table['change']=table.half_life_days.ne(control)
    return table.sort_values(['validation_utility','change','half_life_days'],ascending=[False,True,True],kind='stable').iloc[0]


def run_risk_trial(engine,state,folder,spec):
    from iteration_sequence import digest,write_json
    if not state['benchmark_relative'] or state['cost_multiple']!=1. or state.get('training_cost_multiplier',1.)!=1.:
        raise ValueError('Risk trial requires original actual and learning costs on the accepted active setup')
    previous=json.loads((ROOT/'outputs/iteration_6/us_large/decision.json').read_text())
    reference=ROOT/f'outputs/iteration_{previous["last_accepted_after"]}/us_large'
    c=engine.cfg.covariance
    paths=[Path(__file__),ROOT/'scripts/jkmp/covariance.py',ROOT/'data/inputs/daily_US.parquet',
           engine.cfg.paths.raw/'ff_factors_daily.parquet',engine.cfg.paths.raw/'top500_by_month.parquet',
           engine.cfg.paths.processed/'covariance/factor_returns.parquet',reference/'pf.parquet',reference/'weights.parquet']
    key=hashlib.sha256(json.dumps(dict(engine=engine.signature,state=state,spec=spec,
        sources={str(p):digest(p) for p in paths}),sort_keys=True).encode()).hexdigest()
    marker=folder/'risk_fit_manifest.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('signature')==key and all((folder/n).is_file() and digest(folder/n)==h for n,h in old['hashes'].items()):
            state.update(old['selected_state']);log.info('Reuse verified iteration 7');return pd.read_parquet(folder/'pf.parquet')
    prepare_risk(engine,folder,key,spec['idio_half_life_candidates'])
    rows=[]
    for half_life in spec['idio_half_life_candidates']:
        trial=folder/'pretest_trials'/f'hl_{half_life}';trial.mkdir(parents=True,exist_ok=True)
        trial_state={**state,'idio_half_life_days':int(half_life)}
        stamp=dict(signature=key,half_life=half_life);trial_marker=trial/'completion.json'
        if trial_marker.exists():
            old=json.loads(trial_marker.read_text())
            if old.get('stamp')==stamp and all((trial/n).is_file() and digest(trial/n)==h for n,h in old['hashes'].items()):
                rows.append(old['row']);continue
        if half_life==c.half_life_idio_days:
            gamma=float(pd.read_csv(reference/'calibrated.csv').gamma_te.iloc[0])
            shutil.copy2(reference/'calibration.csv',trial/'calibration.csv')
        else:
            points=[]
            for gamma in (100.,250.,50.,500.,25.,1000.,10.,2500.):
                points.append(engine.fit(trial_state,trial,calibration=True,gamma_te=gamma))
                pd.DataFrame(points).to_csv(trial/'calibration.csv',index=False)
                if len(points)>1 and min(p['te_ex_ante'] for p in points)<=state['te_target']<=max(p['te_ex_ante'] for p in points):break
            gamma=float(calibrate(pd.DataFrame(points),[state['te_target']]).gamma_te.iloc[0])
        pd.DataFrame([dict(te_target=state['te_target'],gamma_te=gamma)]).to_csv(trial/'calibrated.csv',index=False)
        result=engine.fit(trial_state,trial,calibration=True,gamma_te=gamma,validation_result=True)
        pf=result['pf'];dates=pd.DatetimeIndex(pf.eom_ret)
        if dates.max()>pd.Timestamp(engine.cfg.dates.validation_end):raise ValueError('Test returns entered risk selection')
        save_result(result,trial);score=statistics(pf)
        row=dict(half_life_days=half_life,gamma_te=gamma,validation_utility=score['Utility'],
            validation_net_return=score['R_TC'],validation_cost=score['TC'],validation_turnover=score['Turnover'],
            validation_net_sharpe=score['SR_net'],validation_months=len(pf),first_return=str(dates.min().date()),
            last_return=str(dates.max().date()),actual_cost_multiple=state['cost_multiple'])
        rows.append(row)
        write_json(trial_marker,dict(stamp=stamp,row=row,hashes={n:digest(trial/n) for n in ('pf.parquet','weights.parquet','chosen.csv')}))
        pd.DataFrame(rows).to_csv(folder/'risk_half_life_validation.csv',index=False)
    chosen=choose_risk(rows,c.half_life_idio_days);half_life=int(chosen.half_life_days)
    state['idio_half_life_days']=half_life
    write_json(folder/'pretest_selection.json',dict(selected_half_life_days=half_life,gamma_te=float(chosen.gamma_te),
        candidates=spec['idio_half_life_candidates'],test_returns_used=False,frozen_for_all_test_years=True,
        validation_last_return=str(chosen.last_return),selection='maximum pretest actual-cost utility; ties unchanged 126 then smaller',
        actual_cost_multiple=1.,training_cost_multiplier=1.))
    for name in ('calibration.csv','calibrated.csv'):shutil.copy2(folder/f'pretest_trials/hl_{half_life}'/name,folder/name)
    if half_life==c.half_life_idio_days:
        for name in ('pf.parquet','weights.parquet','chosen.csv','aims.parquet','aim_coefs.parquet','validation.parquet'):shutil.copy2(reference/name,folder/name)
    else:engine.fit(state,folder,gamma_te=float(chosen.gamma_te))
    plot_risk_trials(pd.DataFrame(rows),folder)
    write_json(marker,dict(signature=key,selected_state=state,hashes={n:digest(folder/n) for n in ('pf.parquet','weights.parquet','chosen.csv')}))
    return pd.read_parquet(folder/'pf.parquet')


def plot_risk_trials(table,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4));x=np.arange(len(table))
    for ax,col,title in zip(axes,['validation_utility','validation_cost','validation_turnover'],
                           ['Annual utility at gamma 10','Annual actual trading cost','Monthly turnover']):
        ax.bar(x,100*table[col],color='#267a9e');ax.set_xticks(x,[f'{h} days' for h in table.half_life_days])
        ax.set_title(title);ax.set_ylabel('Percent');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Iteration 7 pretest selection | 2016-2019 | Only stock-specific risk weighting changes')
    fig.tight_layout();out=folder/'charts';out.mkdir(exist_ok=True);fig.savefig(out/'pretest_risk_half_life.png',dpi=170);plt.close(fig)
