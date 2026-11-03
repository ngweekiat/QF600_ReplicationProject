"""Iteration 9: recursive benchmark-plus-active trading costs in coefficient learning.

For fixed coefficients, existing execution is affine: w_t=b_t+Q_t beta.
Its orders are d_t=dB_t+T_t beta, including exits. Thus costs are exactly
c0 + c' beta + .5 beta' C beta, c=W T' Lambda dB, C=W T' Lambda T.
Only the learning cost representation changes. Original return/risk moment
proxies, ridge search, execution and actual cost accounting are retained.
"""
from dataclasses import dataclass
import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from jkmp.portfolio.portfolio_ml import (build_moments,YearSums,YearlyMoments,
    search_yearly,run_portfolio_ml,pfml_weights)
from jkmp.portfolio.trading_speed import trading_speed
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.grids import grid_inputs,_signal_kwargs
from jkmp.portfolio.active import calibrate
from jkmp.evaluation.metrics import active_weights,ex_ante_exposures
from selective_trading import save_result
from workflow import ROOT

log=logging.getLogger(__name__)
MONTH=pd.offsets.MonthEnd(1)


@dataclass
class CostCurve:
    eom: pd.Timestamp
    ids: np.ndarray
    base_weight: np.ndarray
    base_start: np.ndarray
    response_weight: np.ndarray
    response_start: np.ndarray
    exit: np.ndarray
    lam: np.ndarray
    wealth: float

    @property
    def base_order(self):return self.base_weight-self.base_start
    @property
    def order_response(self):return self.response_weight-self.response_start

    def coefficients(self):
        T=self.order_response;b=self.base_order;scale=self.wealth*self.lam
        constant=.5*float(np.sum(scale*b*b))
        linear=T.T@(scale*b)
        quadratic=T.T@(scale[:,None]*T)
        return constant,linear,.5*(quadratic+quadratic.T)

    def literal_cost(self,beta):
        order=(self.base_weight+self.response_weight@beta)-(self.base_start+self.response_start@beta)
        return .5*float(np.sum(self.wealth*self.lam*order**2))


class AffineTradeBasis:
    """Same VW start, drift, membership, exits and trading rule as recursive_weights."""
    def __init__(self,panel,wealth,cov,rf,gamma,mu,iterations,charge_exit_tc=True):
        cols=['id','me','lambda','tr_ld1']
        self.panel={d:g.set_index('id')[cols[1:]] for d,g in panel[['eom',*cols]].groupby('eom',sort=False)}
        self.wealth=wealth.set_index('eom');self.cov=cov;self.rf=rf
        self.gamma=gamma;self.mu=mu;self.iterations=iterations;self.charge_exit_tc=charge_exit_tc
        self.previous_ids=None;self.previous_base=None;self.previous_response=None;self.last=None

    def step(self,date,ids,signal):
        date=pd.Timestamp(date);ids=np.asarray(ids);P=signal.shape[1]
        if self.last is not None and date!=self.last+MONTH:raise ValueError('Trade basis requires consecutive months')
        p=self.panel[date];current=p.reindex(ids)
        if current[['me','lambda']].isna().any().any():raise ValueError('Missing benchmark cost inputs')
        market=current.me.to_numpy()/current.me.sum()
        if self.previous_ids is None:
            held=market.copy();response_start=np.zeros((len(ids),P));exit_ids=np.array([],dtype=ids.dtype)
            exit_base=np.zeros(0);exit_response=np.zeros((0,P))
        else:
            positions=pd.Index(self.previous_ids).get_indexer(ids);found=positions>=0
            held=np.zeros(len(ids));response_start=np.zeros((len(ids),P))
            held[found]=self.previous_base[positions[found]]
            response_start[found]=self.previous_response[positions[found]]
            leaving=~pd.Index(self.previous_ids).isin(ids)
            leaving &= pd.Index(self.previous_ids).isin(p.index)
            if not self.charge_exit_tc:leaving[:]=False
            exit_ids=self.previous_ids[leaving];exit_base=self.previous_base[leaving]
            exit_response=self.previous_response[leaving]
        wealth=float(self.wealth.loc[date,'wealth'])
        m=trading_speed(self.cov[date].full(ids),current['lambda'].to_numpy(),wealth,self.gamma,
                        float(self.rf[date+MONTH]),self.mu,self.iterations)
        base=market+m@(held-market)
        response=m@response_start+signal-m@signal
        curve=CostCurve(date,np.r_[ids,exit_ids],np.r_[base,np.zeros(len(exit_ids))],np.r_[held,exit_base],
            np.vstack([response,np.zeros_like(exit_response)]),np.vstack([response_start,exit_response]),
            np.r_[np.zeros(len(ids),dtype=bool),np.ones(len(exit_ids),dtype=bool)],
            p['lambda'].reindex(np.r_[ids,exit_ids]).to_numpy(),wealth)
        # This forward return is used ONLY to form the next month's inheritance,
        # after that return has occurred; it never enters this month's cost curve.
        growth=(1+current.tr_ld1.to_numpy())/(1+float(self.wealth.loc[date,'mu_ld1']))
        if not np.isfinite(growth).all():raise ValueError('Missing inherited-weight growth')
        self.previous_ids=ids.copy();self.previous_base=base*growth
        self.previous_response=response*growth[:,None];self.last=date
        return curve


def search_netting(panel,wealth,cov,rf,dates,features,aim,pm,lambdas,years,tag,kwargs,cfg,verify_zero=True,expected_baseline=None):
    """Stream affine cost curves; keep original return and risk proxies unchanged."""
    basis=AffineTradeBasis(panel,wealth,cov,rf,kwargs['gamma'],cfg.investor.mu,
                          cfg.investor.m_iterations,cfg.portfolio.charge_exit_tc)
    sums={};signals={};audit=[];baselines=[];keep=set(aim);rng=np.random.default_rng(20261106)
    def collect(mm):
        curve=basis.step(mm.eom,mm.ids,mm.signal)
        constant,linear,C=curve.coefficients()
        maximum=0.
        for scale in (0.,.0001,.001):
            beta=scale*rng.normal(size=len(linear))
            predicted=constant+linear@beta+.5*beta@C@beta
            literal=curve.literal_cost(beta)
            np.testing.assert_allclose(predicted,literal,rtol=1e-10,atol=1e-12)
            maximum=max(maximum,abs(float(predicted-literal)))
        year=(mm.eom+MONTH).year
        if year not in sums:sums[year]=YearSums(0,np.zeros_like(mm.r_tilde),np.zeros_like(mm.denom))
        sums[year].n+=1
        sums[year].r_sum+=mm.r_tilde-linear
        sums[year].d_sum+=mm.risk+C
        if mm.eom in keep:signals[mm.eom]=(mm.ids,mm.signal)
        audit.append(dict(eom=mm.eom,tag=tag,baseline_cost=constant,linear_norm=float(np.linalg.norm(linear)),
             cost_quadratic_trace=float(np.trace(C)),original_cost_quadratic_trace=float(np.trace(mm.denom-mm.risk)),
             cost_polynomial_error=maximum,polynomial_probes=3,exit_rows=int(curve.exit.sum()),
             prior_formation_input=None if not audit else str((mm.eom-MONTH).date())))
        baselines.append(pd.DataFrame(dict(id=curve.ids,eom=mm.eom,w=curve.base_weight,w_start=curve.base_start,exit=curve.exit)))
    shell=build_moments(panel,wealth,cov,rf,dates,features,sink=collect,**kwargs)
    ym=YearlyMoments(shell.columns,{**shell.meta,'cost_learning':'affine_recursive_net_orders'},sums,signals,shell.rff_w)
    result=search_yearly(ym,tag,pm.p_grid,lambdas,years,pm.half_life_years)
    audit=pd.DataFrame(audit);baseline=pd.concat(baselines,ignore_index=True)
    order=['eom','id','exit']
    if verify_zero:
        zero=baseline.loc[~baseline.exit,['id','eom']].assign(w_aim=0.)
        inputs=month_inputs(panel,wealth,cov,dates,mu_col=None)
        independent=pfml_weights(inputs,zero,panel,wealth,rf,gamma=kwargs['gamma'],mu=cfg.investor.mu,
            iterations=cfg.investor.m_iterations,charge_exit_tc=cfg.portfolio.charge_exit_tc,benchmark_relative=True)
        a=baseline.sort_values(order).reset_index(drop=True);b=independent.sort_values(order).reset_index(drop=True)
        pd.testing.assert_frame_equal(a[order],b[order])
        np.testing.assert_allclose(a[['w','w_start']],b[['w','w_start']],rtol=1e-10,atol=1e-12)
    else:
        pd.testing.assert_frame_equal(baseline.sort_values(order).reset_index(drop=True),
                                      expected_baseline.sort_values(order).reset_index(drop=True))
    baseline_pf=portfolio_returns(baseline,panel,wealth)
    np.testing.assert_allclose(baseline_pf.tc,audit.baseline_cost,rtol=1e-10,atol=1e-12)
    return result,audit,baseline,baseline_pf


def fit_netting(engine,state,folder,*,calibration=False,gamma_te=None,validation_result=False):
    if state.get('training_cost_multiplier',1.)!=1. or 'idio_half_life_days' in state or state.get('selection_rule')=='one_se' or state['half_lives']!=[None]:
        raise ValueError('Netting trial must exclude rejected changes')
    cfg=engine.cfg;panel=engine.panel_for(state,folder);cov=engine.covariance_for(panel)
    pm=dataclasses.replace(cfg.portfolio_ml,benchmark_relative=True,add_constant=False)
    cfg=dataclasses.replace(cfg,portfolio_ml=pm);inp=grid_inputs(cfg,panel,engine.wealth,cov,engine.rf)
    if calibration:
        end=pd.Timestamp(cfg.dates.validation_end)
        dates=inp.pfml_dates[inp.pfml_dates+MONTH<=end];years=[y for y in inp.hp_years if y<=end.year]
    else:dates=inp.pfml_dates;years=inp.hp_years
    aim=dates[(dates+MONTH).year>years[0]];evaluation=aim if calibration else inp.eval_dates
    results={};audits=[];baseline=None;baseline_pf=None
    for g in pm.rff_variance_grid:
        tag=f'rff_g{g:.4f}';log.info('Netting fit gamma_TE=%g %s; %d months',gamma_te,tag,len(dates))
        kwargs=dict(gamma=gamma_te,mu=cfg.investor.mu,lb=cfg.universe.lookback_months,
            iterations=cfg.investor.m_iterations,scale_by_vol=pm.scale_by_vol,add_constant=False,**_signal_kwargs(cfg,g))
        result,audit,current,current_pf=search_netting(inp.panel,engine.wealth,cov,engine.rf,dates,cfg.features.names,
            aim,pm,cfg.ridge_grid.values,years,tag,kwargs,cfg,verify_zero=baseline is None,expected_baseline=baseline)
        results[tag]=result;audits.append(audit)
        if baseline is None:baseline=current;baseline_pf=current_pf
    result=run_portfolio_ml(results,inp.panel,engine.wealth,cov,engine.rf,aim,evaluation,gamma=gamma_te,
        mu=cfg.investor.mu,iterations=cfg.investor.m_iterations,charge_exit_tc=cfg.portfolio.charge_exit_tc,benchmark_relative=True)
    result.update(cost_audit=pd.concat(audits,ignore_index=True),baseline_weights=baseline,baseline_pf=baseline_pf)
    if calibration:
        if validation_result:return result
        ex=ex_ante_exposures(active_weights(result['weights'],inp.panel),cov)
        return dict(gamma_te=gamma_te,te_ex_ante=float(ex.te.mean()*np.sqrt(12)),months=len(evaluation),last_return=str((evaluation[-1]+MONTH).date()))
    save_result(result,folder);result['cost_audit'].to_csv(folder/'cost_audit.csv',index=False)
    return result


def run_netting(engine,state,folder,spec):
    from iteration_sequence import digest,write_json
    key=hashlib.sha256(json.dumps(dict(engine=engine.signature,state=state,spec=spec,
        source=digest(Path(__file__))),sort_keys=True).encode()).hexdigest()
    marker=folder/'netting_fit_manifest.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('signature')==key and all((folder/n).is_file() and digest(folder/n)==h for n,h in old['hashes'].items()):
            log.info('Reuse verified iteration 9');return pd.read_parquet(folder/'pf.parquet')
    points=[]
    for gamma in spec['calibration_gamma_grid']:
        trial=folder/'calibration_trials'/f'gamma_{gamma:g}';trial.mkdir(parents=True,exist_ok=True)
        done=trial/'completion.json'
        if done.exists():
            old=json.loads(done.read_text())
            if old.get('signature')==key and all((trial/n).is_file() and digest(trial/n)==h for n,h in old['hashes'].items()):
                points.append(old['point'])
                if len(points)>1 and min(p['te_ex_ante'] for p in points)<=state['te_target']<=max(p['te_ex_ante'] for p in points):break
                continue
        result=fit_netting(engine,state,trial,calibration=True,gamma_te=gamma,validation_result=True)
        save_result(result,trial);result['cost_audit'].to_csv(trial/'cost_audit.csv',index=False)
        panel=engine.panel_for(state,trial);cov=engine.covariance_for(panel)
        ex=ex_ante_exposures(active_weights(result['weights'],panel),cov)
        dates=pd.DatetimeIndex(result['pf'].eom_ret)
        if dates.max()>pd.Timestamp(engine.cfg.dates.validation_end):raise ValueError('Test returns entered calibration')
        point=dict(gamma_te=gamma,te_ex_ante=float(ex.te.mean()*np.sqrt(12)),months=len(dates),last_return=str(dates.max().date()))
        points.append(point);pd.DataFrame(points).to_csv(folder/'calibration.csv',index=False)
        names=('pf.parquet','weights.parquet','chosen.csv','cost_audit.csv')
        write_json(done,dict(signature=key,point=point,hashes={n:digest(trial/n) for n in names}))
        if len(points)>1 and min(p['te_ex_ante'] for p in points)<=state['te_target']<=max(p['te_ex_ante'] for p in points):break
    chosen=calibrate(pd.DataFrame(points),[state['te_target']]);chosen.to_csv(folder/'calibrated.csv',index=False)
    gamma=float(chosen.gamma_te.iloc[0])
    write_json(folder/'pretest_selection.json',dict(gamma_te=gamma,te_target=state['te_target'],test_returns_used=False,
        cost_representation='exact affine recursive net orders for fixed coefficients, including exits',
        return_risk_proxies_unchanged=True,actual_cost_formula_unchanged=True,
        limitations='annual coefficient-jump and original return/risk learning approximations retained; realised accounting exact',
        calibration_last_return=max(p['last_return'] for p in points)))
    fit_netting(engine,state,folder,gamma_te=gamma)
    names=('pf.parquet','weights.parquet','chosen.csv','cost_audit.csv')
    write_json(marker,dict(signature=key,hashes={n:digest(folder/n) for n in names}))
    return pd.read_parquet(folder/'pf.parquet')
