"""Iteration 10: fixed equal-weight RFF seed ensemble, with net orders executed once.

All members use identical gamma, data, equations, initialisation and grids.
Each independently selects its annual hyperparameters from prior returns.
Average stock aims, not coefficients from incompatible random feature bases.
"""
from __future__ import annotations
import copy
import dataclasses
import hashlib
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from workflow import ROOT
from jkmp.portfolio.grids import grid_inputs
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.portfolio_ml import pfml_weights
from jkmp.portfolio.accounting import portfolio_returns,add_start_weights
from jkmp.portfolio.benchmarks import market_targets
from jkmp.portfolio.active import calibrate
from jkmp.evaluation.metrics import active_weights,ex_ante_exposures
from analysis.ml_comparison.metrics import statistics
from selective_trading import save_result

log=logging.getLogger(__name__)
MONTH=pd.offsets.MonthEnd(1)
RESULT_FILES=('pf.parquet','weights.parquet','chosen.csv','aims.parquet','aim_coefs.parquet','validation.parquet')


def average_aims(members):
    """Reject different/duplicate stock-month keys; every member has equal weight."""
    frames=[];expected=None
    for seed,result in members.items():
        frame=result['aims'][['eom','id','w_aim']].copy()
        if frame.duplicated(['eom','id']).any() or not np.isfinite(frame.w_aim).all():
            raise ValueError('Invalid member aims')
        keys=pd.MultiIndex.from_frame(frame[['eom','id']]).sort_values()
        if expected is None:expected=keys
        elif not expected.equals(keys):raise ValueError('Ensemble stock-month keys differ')
        frames.append(frame.assign(seed=seed))
    if not frames:raise ValueError('Empty ensemble')
    return pd.concat(frames).groupby(['eom','id'],as_index=False).w_aim.mean()


def audit_combined_orders(members,weights,panel,wealth):
    """Check affine execution and Jensen cost identity against literal stock orders."""
    keys=['eom','id','exit'];frames=[]
    for seed,result in members.items():
        w=result['weights'][keys+['w','w_start']].copy()
        if w.duplicated(keys).any():raise ValueError('Duplicate member orders')
        frames.append(w.set_index(keys).rename(columns={'w':f'w_{seed}','w_start':f'start_{seed}'}))
    joined=pd.concat(frames,axis=1).sort_index()
    # An omitted exit is an exactly zero position. Missing held rows are invalid.
    if joined.loc[~joined.index.get_level_values('exit')].isna().any().any():
        raise ValueError('Missing held row in a seed member')
    joined=joined.fillna(0.)
    actual=weights.set_index(keys)[['w','w_start']]
    if not actual.index.difference(joined.index).empty:raise ValueError('Unexpected ensemble order')
    actual=actual.reindex(joined.index)
    if actual.loc[~actual.index.get_level_values('exit')].isna().any().any():
        raise ValueError('Missing held ensemble row')
    actual=actual.fillna(0.)
    W=np.column_stack([joined[f'w_{s}'] for s in members])
    H=np.column_stack([joined[f'start_{s}'] for s in members])
    D=W-H;mean_w=W.mean(axis=1);mean_h=H.mean(axis=1);mean_d=D.mean(axis=1)
    np.testing.assert_allclose(actual.w,mean_w,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(actual.w_start,mean_h,rtol=1e-10,atol=1e-12)
    stock=actual.reset_index().merge(panel[['id','eom','lambda']],on=['id','eom'],validate='many_to_one').merge(
        wealth[['eom','wealth']],on='eom',validate='many_to_one')
    if stock[['lambda','wealth']].isna().any().any():raise ValueError('Missing cost inputs')
    scale=.5*stock.wealth.to_numpy()*stock['lambda'].to_numpy()
    direct=scale*(stock.w-stock.w_start).to_numpy()**2
    average=scale*np.mean(D**2,axis=1)
    gap=scale*np.mean((D-mean_d[:,None])**2,axis=1)
    np.testing.assert_allclose(average-direct,gap,rtol=1e-9,atol=1e-12)
    rows=pd.DataFrame(dict(eom=stock.eom,ensemble_cost=direct,mean_member_cost=average,
        variance_cost_saving=gap,order_identity_error=np.abs((stock.w-stock.w_start).to_numpy()-mean_d),
        weight_identity_error=np.abs(stock.w.to_numpy()-mean_w)))
    out=rows.groupby('eom',as_index=False).agg(ensemble_cost=('ensemble_cost','sum'),
        mean_member_cost=('mean_member_cost','sum'),variance_cost_saving=('variance_cost_saving','sum'),
        max_order_identity_error=('order_identity_error','max'),max_weight_identity_error=('weight_identity_error','max'))
    out['cost_identity_error']=out.mean_member_cost-out.ensemble_cost-out.variance_cost_saving
    return out


def combine_members(engine,state,members,gamma,calibration):
    cfg=engine.cfg;panel=engine.panel_for(state,ROOT/'outputs/iteration_10/us_large');cov=engine.covariance_for(panel)
    inp=grid_inputs(cfg,panel,engine.wealth,cov,engine.rf)
    dates=inp.pfml_dates[inp.pfml_dates+MONTH<=pd.Timestamp(cfg.dates.validation_end)] if calibration else inp.eval_dates
    if calibration:dates=dates[(dates+MONTH).year>inp.hp_years[0]]
    aims=average_aims(members)
    weights=pfml_weights(month_inputs(inp.panel,engine.wealth,cov,dates,mu_col=None),aims,inp.panel,
        engine.wealth,engine.rf,gamma=gamma,mu=cfg.investor.mu,iterations=cfg.investor.m_iterations,
        charge_exit_tc=cfg.portfolio.charge_exit_tc,benchmark_relative=True)
    pf=portfolio_returns(weights,inp.panel,engine.wealth).assign(type='Portfolio-ML')
    audit=audit_combined_orders(members,weights,inp.panel,engine.wealth)
    np.testing.assert_allclose(pf.tc,audit.ensemble_cost,rtol=1e-11,atol=1e-12)
    mean_r=np.mean([r['pf'].sort_values('eom_ret').r.to_numpy() for r in members.values()],axis=0)
    np.testing.assert_allclose(pf.r,mean_r,rtol=1e-10,atol=1e-12)
    result=dict(aims=aims,weights=weights.assign(type='Portfolio-ML'),pf=pf)
    for name in ('chosen','aim_coefs','validation'):
        result[name]=pd.concat([r[name].assign(seed=s,ensemble_weight=1/len(members)) for s,r in members.items()],ignore_index=True)
    result['cost_netting_audit']=audit
    return result


def load_result(folder):
    return {name:pd.read_csv(folder/'chosen.csv') if name=='chosen' else pd.read_parquet(folder/f'{name}.parquet')
            for name in ('pf','weights','chosen','aims','aim_coefs','validation')}


def fit_ensemble(engine,state,folder,spec,gamma,calibration,key):
    from iteration_sequence import digest,write_json
    if (state.get('training_cost_multiplier',1.)!=1. or 'idio_half_life_days' in state
            or state.get('selection_rule')=='one_se' or 'cost_learning' in state or state['half_lives']!=[None]):
        raise ValueError('Ensemble must exclude rejected iteration changes')
    members={}
    for seed in spec['seeds']:
        path=folder/'members'/f'seed_{seed}';path.mkdir(parents=True,exist_ok=True)
        marker=path/'completion.json'
        signature=hashlib.sha256(f'{key}|{gamma}|{calibration}|{seed}'.encode()).hexdigest()
        valid=False
        if marker.exists():
            old=json.loads(marker.read_text())
            valid=old.get('signature')==signature and all((path/n).is_file() and digest(path/n)==h for n,h in old.get('hashes',{}).items())
        if not valid:
            member=copy.copy(engine);member.cfg=dataclasses.replace(engine.cfg,seed=int(seed))
            log.info('Ensemble seed=%d gamma_TE=%g pretest=%s',seed,gamma,calibration)
            result=member.fit(state,path,calibration=calibration,gamma_te=gamma,validation_result=calibration)
            save_result(result,path)
            write_json(marker,dict(signature=signature,seed=seed,gamma_te=gamma,pretest=calibration,
                hashes={n:digest(path/n) for n in RESULT_FILES}))
        members[seed]=load_result(path)
        # Parquet/CSV round trip restores timestamps used by the generic verifier.
        members[seed]['chosen']['eom_ret']=pd.to_datetime(members[seed]['chosen'].eom_ret)
    result=combine_members(engine,state,members,gamma,calibration)
    save_result(result,folder);result['cost_netting_audit'].to_csv(folder/'cost_netting_audit.csv',index=False)
    return result,members


def pretest_diagnostic(engine,state,folder,result,members,gamma):
    from iteration_sequence import write_json
    dates=pd.DatetimeIndex(result['pf'].eom)
    panel=engine.panel_for(state,folder)
    targets=market_targets(panel.loc[panel.in_universe&panel.eom.isin(dates)])
    market_w=add_start_weights(targets,panel,engine.wealth,dates,engine.cfg.portfolio.charge_exit_tc)
    market=portfolio_returns(market_w,panel,engine.wealth)
    market.to_parquet(folder/'market_reference.parquet',index=False)
    market_w.to_parquet(folder/'market_reference_weights.parquet',index=False)
    metrics=[];active={};weight_vectors={};pairs=[]
    for seed,r in members.items():
        pf=r['pf'].sort_values('eom_ret');d=pd.DatetimeIndex(pf.eom_ret)
        if not d.equals(pd.DatetimeIndex(market.eom_ret)) or d.max()>pd.Timestamp(engine.cfg.dates.validation_end):
            raise ValueError('Diagnostic requires only common pretest months')
        metrics.append(dict(seed=seed,**statistics(pf)))
        active[str(seed)]=(pf.r.to_numpy()-pf.tc.to_numpy())-(market.r-market.tc).to_numpy()
        weight_vectors[seed]=r['weights'].set_index(['eom','id','exit']).w
    for a,b in combinations(members,2):
        joined=pd.concat([weight_vectors[a],weight_vectors[b]],axis=1).fillna(0.).to_numpy()
        pairs.append(dict(seed_a=a,seed_b=b,weight_difference_rms=float(np.sqrt(np.mean((joined[:,0]-joined[:,1])**2))),
            max_absolute_weight_difference=float(np.max(np.abs(joined[:,0]-joined[:,1]))),
            net_active_return_correlation=float(np.corrcoef(active[str(a)],active[str(b)])[0,1])))
    pd.DataFrame(metrics).to_csv(folder/'seed_performance.csv',index=False)
    pairs=pd.DataFrame(pairs);pairs.to_csv(folder/'pairwise_seed_sensitivity.csv',index=False)
    pd.DataFrame(active,index=market.eom_ret).corr().to_csv(folder/'active_return_correlation.csv')
    control=ROOT/'outputs/iteration_6/us_large/pretest_trials/cost_1/pf.parquet'
    original=pd.read_parquet(control).sort_values('eom_ret')
    seed1=members[engine.cfg.seed]['pf'].sort_values('eom_ret')
    np.testing.assert_allclose(seed1[['r','tc']],original[['r','tc']],rtol=1e-10,atol=1e-12)
    summary=dict(pretest_only=True,first_return=str(market.eom_ret.min().date()),last_return=str(market.eom_ret.max().date()),
        months=len(market),gamma_te=gamma,seeds=list(members),equal_weights=True,seeds_selected_by_performance=False,
        seed_1_matches_original_control=True,mean_pairwise_weight_difference_rms=float(pairs.weight_difference_rms.mean()),
        median_active_return_correlation=float(pairs.net_active_return_correlation.median()),
        numerically_identical=bool(pairs.max_absolute_weight_difference.max()<=1e-12),
        annual_cost_saving_vs_mean_member=float(12*result['cost_netting_audit'].variance_cost_saving.mean()),
        hypothesis='Different random feature draws may contain diversifiable approximation noise; dispersion alone does not establish an alpha improvement')
    write_json(folder/'diagnostic.json',summary)
    plot_diagnostic(folder,pairs,pd.DataFrame(active).corr(),summary)
    return summary


def plot_diagnostic(folder,pairs,correlation,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    im=axes[0].imshow(correlation,vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=axes[0],label='Net active return correlation')
    axes[0].set_xticks(range(len(correlation)),correlation.columns);axes[0].set_yticks(range(len(correlation)),correlation.index)
    for i in range(len(correlation)):
        for j in range(len(correlation)):axes[0].text(j,i,f'{correlation.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    axes[0].set(title='Pretest seed sensitivity',xlabel='Fixed RFF seed',ylabel='Fixed RFF seed')
    axes[1].bar(range(len(pairs)),10000*pairs.weight_difference_rms,color='#217c91')
    axes[1].set_xticks(range(len(pairs)),[f'{r.seed_a}-{r.seed_b}' for r in pairs.itertuples()],rotation=35)
    axes[1].set(title='Stock-weight differences across seeds',ylabel='RMS weight difference (basis points)',xlabel='Seed pair')
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Iteration 10 | 2016-2019 only | Equal-weight seeds 1-5; no seed cherry-picking')
    fig.text(.5,.015,'Seed dispersion motivates a test; it does not establish future outperformance.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.94));out=folder/'charts';out.mkdir(exist_ok=True)
    fig.savefig(out/'pretest_seed_sensitivity.png',dpi=180);plt.close(fig)


def run_ensemble(engine,state,folder,spec):
    from iteration_sequence import digest,write_json
    if len(spec['seeds'])!=5 or len(set(spec['seeds']))!=5 or spec['seeds']!=list(range(engine.cfg.seed,engine.cfg.seed+5)):
        raise ValueError('Use exactly five predetermined consecutive seeds, including the original')
    key=hashlib.sha256(json.dumps(dict(engine=engine.signature,state=state,spec=spec,
        source=digest(Path(__file__)),core=digest(ROOT/'scripts/jkmp/portfolio/portfolio_ml.py')),sort_keys=True).encode()).hexdigest()
    marker=folder/'ensemble_fit_manifest.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('signature')==key and all((folder/n).is_file() and digest(folder/n)==h for n,h in old['hashes'].items()):
            log.info('Reuse verified iteration 10');return pd.read_parquet(folder/'pf.parquet')
    reference_gamma=float(pd.read_csv(ROOT/'outputs/iteration_4/us_large/calibrated.csv').gamma_te.iloc[0])
    diagnostic=folder/'pretest_seed_sensitivity';diagnostic.mkdir(parents=True,exist_ok=True)
    result,members=fit_ensemble(engine,state,diagnostic,spec,reference_gamma,True,key)
    diagnostic_summary=pretest_diagnostic(engine,state,diagnostic,result,members,reference_gamma)
    points=[];panel=engine.panel_for(state,folder);cov=engine.covariance_for(panel)
    for gamma in spec['calibration_gamma_grid']:
        trial=folder/'calibration_trials'/f'gamma_{gamma:g}';trial.mkdir(parents=True,exist_ok=True)
        result,_=fit_ensemble(engine,state,trial,spec,gamma,True,key)
        ex=ex_ante_exposures(active_weights(result['weights'],panel),cov)
        dates=pd.DatetimeIndex(result['pf'].eom_ret)
        if dates.max()>pd.Timestamp(engine.cfg.dates.validation_end):raise ValueError('Test returns entered TE calibration')
        points.append(dict(gamma_te=gamma,te_ex_ante=float(ex.te.mean()*np.sqrt(12)),months=len(dates),last_return=str(dates.max().date())))
        pd.DataFrame(points).to_csv(folder/'calibration.csv',index=False)
        if len(points)>1 and min(p['te_ex_ante'] for p in points)<=state['te_target']<=max(p['te_ex_ante'] for p in points):break
    chosen=calibrate(pd.DataFrame(points),[state['te_target']]);chosen.to_csv(folder/'calibrated.csv',index=False)
    gamma=float(chosen.gamma_te.iloc[0])
    write_json(folder/'pretest_selection.json',dict(seeds=spec['seeds'],seed_weights=[.2]*5,seed_count_selected=False,
        seeds_selected_by_performance=False,gamma_te=gamma,te_target=state['te_target'],test_returns_used=False,
        calibration_last_return=max(p['last_return'] for p in points),same_gamma_for_all_members=True,
        combination='equal stock-level aims, followed by one original recursive execution',
        learning_and_actual_cost_equations_unchanged=True,pretest_sensitivity=diagnostic_summary))
    fit_ensemble(engine,state,folder,spec,gamma,False,key)
    names=(*RESULT_FILES,'cost_netting_audit.csv','pretest_selection.json')
    write_json(marker,dict(signature=key,hashes={n:digest(folder/n) for n in names}))
    return pd.read_parquet(folder/'pf.parquet')
