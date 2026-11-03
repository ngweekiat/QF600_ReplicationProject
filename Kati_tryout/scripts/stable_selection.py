"""Iteration 8: one-standard-error selection of simpler Portfolio-ML candidates.

Original RFF signals, coefficient equations, candidate grids and implementation
functions are reused. Only annual candidate selection changes. Monthly utility
flows reproduce the existing December cumulative score exactly; uncertainty is
estimated from past flows of that year's best candidate, using circular blocks.
"""
import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from jkmp.portfolio.portfolio_ml import (build_yearly_moments,build_moments,search_yearly,
    model_columns,aim_portfolios,pfml_weights)
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.grids import grid_inputs,_signal_kwargs
from jkmp.portfolio.active import calibrate
from jkmp.evaluation.metrics import active_weights,ex_ante_exposures
from selective_trading import save_result
from workflow import ROOT

log=logging.getLogger(__name__)
MONTH=pd.offsets.MonthEnd(1)
DEFAULT_RULE=dict(bootstrap_reps=2000,block_months=6,seed=20261103,se_multiple=1.)


def block_mean_se(values,reps=2000,block=6,seed=20261103):
    x=np.asarray(values,dtype=float)
    if len(x)<2 or not np.isfinite(x).all():raise ValueError('SE requires at least two finite historical utility observations')
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(x),size=(reps,int(np.ceil(len(x)/block))))
    indices=((starts[:,:,None]+np.arange(block))%len(x)).reshape(reps,-1)[:,:len(x)]
    return float(x[indices].mean(axis=1).std(ddof=1))


def stable_hypers(validation,rule=None):
    """At each December: mean >= best mean - SE(best); then p up, lambda down, g up.

    'lambda down' means descending sort: stronger shrinkage among equal-size
    models. No return after the selection month contributes to that choice.
    """
    rule={**DEFAULT_RULE,**(rule or {})}
    keys=['tag','g','p','l_no','l','h'];rows=[];audit=[]
    for date in sorted(validation.loc[validation.eom_ret.dt.month.eq(12),'eom_ret'].unique()):
        date=pd.Timestamp(date)
        past=validation.loc[validation.eom_ret.le(date)].sort_values('eom_ret')
        scores=past.groupby(keys,sort=False,dropna=False).agg(cum_obj=('obj','mean'),months=('obj','size')).reset_index()
        if scores.months.nunique()!=1:raise ValueError('Candidate utility histories differ')
        best=scores.loc[scores.cum_obj.idxmax()]
        mask=np.ones(len(past),dtype=bool)
        for key in keys:mask &= past[key].eq(best[key]).to_numpy()
        flow=past.loc[mask].sort_values('eom_ret').obj.to_numpy()
        se=block_mean_se(flow,rule['bootstrap_reps'],rule['block_months'],rule['seed']+date.year)
        threshold=float(best.cum_obj-rule['se_multiple']*se)
        eligible=scores.loc[scores.cum_obj.ge(threshold-1e-15)]
        selected=eligible.sort_values(['p','l','g','tag'],ascending=[True,False,True,True],kind='stable').iloc[0]
        row={key:selected[key] for key in keys}
        row.update(eom_ret=date,year=date.year+1,cum_obj=float(selected.cum_obj))
        rows.append(row)
        audit.append(dict(year=date.year+1,selection_month=date,validation_last_return=date,
            history_months=len(flow),candidate_count=len(scores),eligible_count=len(eligible),
            best_p=int(best.p),best_l=float(best.l),best_g=float(best.g),best_utility=float(best.cum_obj),
            bootstrap_se=se,utility_threshold=threshold,selected_p=int(selected.p),selected_l=float(selected.l),
            selected_g=float(selected.g),selected_utility=float(selected.cum_obj),
            bootstrap_reps=rule['bootstrap_reps'],block_months=rule['block_months'],
            bootstrap_seed=rule['seed']+date.year))
    return pd.DataFrame(rows),pd.DataFrame(audit)


def monthly_search(panel,wealth,cov,rf,dates,features,aim,pm,lambdas,years,tag,kwargs):
    """Stream the original monthly objective; do not retain monthly dense matrices."""
    ym=build_yearly_moments(panel,wealth,cov,rf,dates,features,signal_dates=aim,**kwargs)
    result=search_yearly(ym,tag,pm.p_grid,lambdas,years,pm.half_life_years)
    original=result.validation.copy();del ym
    flows=[]
    def collect(mm):
        year=(mm.eom+MONTH).year
        if year not in result.coefs:return
        for p,B in result.coefs[year].items():
            columns=model_columns(result,p)
            with np.errstate(all='ignore'):
                obj=mm.r_tilde[columns]@B-.5*np.einsum('kl,kl->l',B,mm.denom[np.ix_(columns,columns)]@B)
            if not np.isfinite(obj).all():raise ValueError('Nonfinite monthly candidate utility')
            flows.append(pd.DataFrame(dict(eom_ret=mm.eom+MONTH,tag=tag,g=result.g,p=p,
                l_no=np.arange(len(lambdas)),l=np.asarray(lambdas),h=np.inf,obj=obj)))
    build_moments(panel,wealth,cov,rf,dates,features,sink=collect,**kwargs)
    monthly=pd.concat(flows,ignore_index=True).sort_values(['p','l_no','eom_ret'])
    monthly['cum_obj']=monthly.groupby(['p','l_no'],sort=False)['obj'].transform(lambda s:s.expanding().mean())
    december=monthly.loc[monthly.eom_ret.dt.month.eq(12)]
    joined=original.merge(december,on=['eom_ret','p','l_no','tag'],suffixes=('_yearly','_monthly'),validate='one_to_one')
    if len(joined)!=len(original):raise AssertionError('Monthly utility reconstruction lost candidates')
    np.testing.assert_allclose(joined.cum_obj_yearly,joined.cum_obj_monthly,rtol=1e-10,atol=1e-12)
    result.validation=monthly
    return result


def implement(results,panel,wealth,cov,rf,aim,evaluation,cfg,gamma,rule):
    validation=pd.concat([r.validation for r in results.values()],ignore_index=True)
    chosen,audit=stable_hypers(validation,rule)
    years={(d+MONTH).year for d in aim}
    aims,coefs=aim_portfolios(results,chosen.loc[chosen.year.isin(years)],aim)
    inputs=month_inputs(panel,wealth,cov,evaluation,mu_col=None)
    weights=pfml_weights(inputs,aims,panel,wealth,rf,gamma=gamma,mu=cfg.investor.mu,
        iterations=cfg.investor.m_iterations,charge_exit_tc=cfg.portfolio.charge_exit_tc,benchmark_relative=True)
    pf=portfolio_returns(weights,panel,wealth).assign(type='Portfolio-ML')
    return dict(validation=validation,chosen=chosen,aims=aims,aim_coefs=coefs,
                weights=weights.assign(type='Portfolio-ML'),pf=pf,selection_audit=audit)


def fit_stable(engine,state,folder,*,calibration=False,gamma_te=None,validation_result=False):
    if state.get('training_cost_multiplier',1.)!=1. or 'idio_half_life_days' in state or state['half_lives']!=[None]:
        raise ValueError('Iteration 8 must exclude the rejected cost/decay/risk changes')
    cfg=engine.cfg;panel=engine.panel_for(state,folder);cov=engine.covariance_for(panel)
    pm=dataclasses.replace(cfg.portfolio_ml,benchmark_relative=True,add_constant=False)
    cfg=dataclasses.replace(cfg,portfolio_ml=pm)
    inp=grid_inputs(cfg,panel,engine.wealth,cov,engine.rf)
    if calibration:
        end=pd.Timestamp(cfg.dates.validation_end)
        dates=inp.pfml_dates[inp.pfml_dates+MONTH<=end]
        years=[y for y in inp.hp_years if y<=end.year]
    else:dates=inp.pfml_dates;years=inp.hp_years
    aim=dates[(dates+MONTH).year>years[0]]
    evaluation=aim if calibration else inp.eval_dates
    results={}
    for g in pm.rff_variance_grid:
        tag=f'rff_g{g:.4f}'
        log.info('Stable selection gamma_TE=%g %s; %d formation months',gamma_te,tag,len(dates))
        kwargs=dict(gamma=gamma_te,mu=cfg.investor.mu,lb=cfg.universe.lookback_months,
            iterations=cfg.investor.m_iterations,scale_by_vol=pm.scale_by_vol,add_constant=False,
            **_signal_kwargs(cfg,g))
        results[tag]=monthly_search(inp.panel,engine.wealth,cov,engine.rf,dates,cfg.features.names,aim,
            pm,cfg.ridge_grid.values,years,tag,kwargs)
    result=implement(results,inp.panel,engine.wealth,cov,engine.rf,aim,evaluation,cfg,gamma_te,state.get('one_se_rule'))
    if calibration:
        if validation_result:return result
        ex=ex_ante_exposures(active_weights(result['weights'],inp.panel),cov)
        return dict(gamma_te=gamma_te,te_ex_ante=float(ex.te.mean()*np.sqrt(12)),months=len(evaluation),last_return=str((evaluation[-1]+MONTH).date()))
    save_result(result,folder)
    result['selection_audit'].to_csv(folder/'selection_audit.csv',index=False)
    return result


def run_stable(engine,state,folder,spec):
    from iteration_sequence import digest,write_json
    key=hashlib.sha256(json.dumps(dict(engine=engine.signature,state=state,spec=spec,
        source=digest(Path(__file__))),sort_keys=True).encode()).hexdigest()
    marker=folder/'stable_fit_manifest.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('signature')==key and all((folder/n).is_file() and digest(folder/n)==h for n,h in saved['hashes'].items()):
            log.info('Reuse verified iteration 8');return pd.read_parquet(folder/'pf.parquet')
    points=[]
    for gamma in spec['calibration_gamma_grid']:
        trial=folder/'calibration_trials'/f'gamma_{gamma:g}';trial.mkdir(parents=True,exist_ok=True)
        trial_marker=trial/'completion.json'
        if trial_marker.exists() and json.loads(trial_marker.read_text()).get('signature')==key:
            old=json.loads(trial_marker.read_text())
            if all((trial/n).is_file() and digest(trial/n)==h for n,h in old['hashes'].items()):
                point=old['point'];points.append(point)
                log.info('Reuse pretest stable-selection calibration gamma=%g',gamma)
                if len(points)>1 and min(p['te_ex_ante'] for p in points)<=state['te_target']<=max(p['te_ex_ante'] for p in points):break
                continue
        result=fit_stable(engine,state,trial,calibration=True,gamma_te=gamma,validation_result=True)
        save_result(result,trial);result['selection_audit'].to_csv(trial/'selection_audit.csv',index=False)
        cov=engine.covariance_for(engine.panel_for(state,trial))
        ex=ex_ante_exposures(active_weights(result['weights'],engine.panel_for(state,trial)),cov)
        dates=pd.DatetimeIndex(result['pf'].eom_ret)
        if dates.max()>pd.Timestamp(engine.cfg.dates.validation_end):raise ValueError('Test returns entered TE calibration')
        point=dict(gamma_te=gamma,te_ex_ante=float(ex.te.mean()*np.sqrt(12)),months=len(dates),last_return=str(dates.max().date()))
        points.append(point);pd.DataFrame(points).to_csv(folder/'calibration.csv',index=False)
        names=('pf.parquet','weights.parquet','chosen.csv','selection_audit.csv')
        write_json(trial_marker,dict(signature=key,point=point,hashes={n:digest(trial/n) for n in names}))
        log.info('Stable-selection pretest gamma=%g: annual ex-ante TE %.6f',gamma,point['te_ex_ante'])
        if len(points)>1 and min(p['te_ex_ante'] for p in points)<=state['te_target']<=max(p['te_ex_ante'] for p in points):break
    calibrated=calibrate(pd.DataFrame(points),[state['te_target']])
    calibrated.to_csv(folder/'calibrated.csv',index=False);gamma=float(calibrated.gamma_te.iloc[0])
    write_json(folder/'pretest_selection.json',dict(rule=state['one_se_rule'],gamma_te=gamma,
        test_returns_used_for_rule_or_TE=False,rule_frozen=True,annual_choices_use_only_prior_returns=True,
        selection='within one block-bootstrap SE of best historical mean utility; fewer RFF, stronger ridge, smaller g',
        calibration_last_return=max(p['last_return'] for p in points)))
    fit_stable(engine,state,folder,gamma_te=gamma)
    plot_audit(pd.read_csv(folder/'selection_audit.csv'),folder)
    names=('pf.parquet','weights.parquet','chosen.csv','selection_audit.csv')
    write_json(marker,dict(signature=key,hashes={n:digest(folder/n) for n in names}))
    return pd.read_parquet(folder/'pf.parquet')


def plot_audit(audit,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    used=audit.loc[audit.year.between(2020,2025)]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(used.year,used.best_p,'o-',label='Highest-utility candidate')
    axes[0].plot(used.year,used.selected_p,'s-',label='One-SE choice');axes[0].set_ylabel('RFF features')
    axes[1].plot(used.year,used.selected_l,'o-');axes[1].set_yscale('symlog',linthresh=.01);axes[1].set_ylabel('Selected ridge penalty')
    axes[0].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('Return year');ax.grid(alpha=.2)
    fig.suptitle('Iteration 8 | Choices use validation returns before each forecast year')
    fig.tight_layout();out=folder/'charts';out.mkdir(exist_ok=True);fig.savefig(out/'annual_selection_robustness.png',dpi=170);plt.close(fig)
