"""Executed active-utility selection and one-SE refinement after iteration 4.

This branch preserves original coefficients, signals, costs, annual fit dates
and the frozen gamma_TE. It changes selection, then adds a fixed simplicity
heuristic. Actual future policy switching costs remain charged by execution.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from workflow import ROOT
from iteration_sequence import ExperimentEngine,digest,write_json,paired_probability,plot_comparison,verify_accounting
from jkmp.portfolio.grids import grid_inputs,_signal_kwargs
from jkmp.portfolio.portfolio_ml import (build_yearly_moments,save_yearly_moments,load_yearly_moments,
    search_yearly,model_columns,aim_portfolios,pfml_weights)
from jkmp.portfolio.trading_speed import trading_speed
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.accounting import add_start_weights,portfolio_returns
from jkmp.portfolio.benchmarks import market_targets
from jkmp.evaluation.metrics import active_weights,ex_ante_exposures
from analysis.ml_comparison.metrics import statistics
from selective_trading import save_result

log=logging.getLogger(__name__)
MONTH=pd.offsets.MonthEnd(1)
PROTOCOL=ROOT/'configs/us_high_cap_substages.json'
CANONICAL=ROOT/'outputs/iteration_4.2/us_large'


def utility(gross_active,net_active,gamma):
    x=np.asarray(gross_active,dtype=float);z=np.asarray(net_active,dtype=float)
    if len(x)<2 or x.shape!=z.shape or not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError('Utility requires matched finite histories of at least two months')
    return 12*(z.mean(axis=0)-.5*gamma*x.var(axis=0,ddof=1))


def bootstrap_utility(x,z,gamma,rule,year):
    x=np.asarray(x);z=np.asarray(z);n=len(x);reps=rule['bootstrap_reps'];block=rule['bootstrap_block_months']
    rng=np.random.default_rng(rule['bootstrap_seed']+year)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    u=12*(z[ix].mean(axis=1)-.5*gamma*x[ix].var(axis=1,ddof=1))
    return float(u.std(ddof=1))


def select_executed(flows,catalog,rule,stable=False,max_year=2025):
    x=flows.pivot(index='eom_ret',columns='cid',values='active_gross').sort_index()
    z=flows.pivot(index='eom_ret',columns='cid',values='active_net').sort_index()
    catalog=catalog.sort_values('cid').reset_index(drop=True)
    if not np.array_equal(x.columns,catalog.cid) or not x.columns.equals(z.columns) or not x.index.equals(z.index):
        raise ValueError('Executed candidate keys differ')
    if x.isna().any().any() or z.isna().any().any():raise ValueError('Incomplete candidate calendars')
    chosen=[];audit=[]
    for date in x.index[x.index.month==12]:
        date=pd.Timestamp(date)
        if date.year+1>max_year:continue
        a=x.loc[:date].to_numpy();n=z.loc[:date].to_numpy()
        score=utility(a,n,rule['gamma_te']);best_no=int(np.argmax(score));best=catalog.iloc[best_no]
        se=bootstrap_utility(a[:,best_no],n[:,best_no],rule['gamma_te'],rule,date.year) if stable else 0.
        threshold=float(score[best_no]-rule.get('one_se_multiple',1.)*se)
        eligible=catalog.loc[score>=threshold-1e-15].copy()
        if stable:row=eligible.sort_values(['p','l','g','cid'],ascending=[True,False,True,True],kind='stable').iloc[0]
        else:row=best
        cid=int(row.cid);location=int(np.flatnonzero(catalog.cid.to_numpy()==cid)[0])
        selected=row.to_dict();selected.update(year=date.year+1,eom_ret=date,cum_obj=float(score[location]))
        chosen.append(selected)
        audit.append(dict(year=date.year+1,selection_month=date,history_months=len(a),candidate_count=len(score),
            best_cid=int(best.cid),best_utility=float(score[best_no]),bootstrap_se=se,utility_threshold=threshold,
            eligible_count=len(eligible),selected_cid=cid,selected_p=int(row.p),selected_l=float(row.l),
            selected_g=float(row.g),selected_utility=float(score[location]),stable_rule=stable,
            bootstrap_reps=rule['bootstrap_reps'] if stable else 0,
            bootstrap_seed=rule['bootstrap_seed']+date.year if stable else 0,
            block_months=rule['bootstrap_block_months'] if stable else 0))
    return pd.DataFrame(chosen),pd.DataFrame(audit)


def source_key(engine,rule):
    files=[Path(__file__),PROTOCOL,*[ROOT/f'scripts/jkmp/portfolio/{name}.py' for name in
        ('portfolio_ml','trading_speed','accounting','recursion','grids')]]
    return hashlib.sha256(json.dumps(dict(engine=engine.signature,config=engine.cfg.as_dict(),rule=rule,
        sources={str(p):digest(p) for p in files}),sort_keys=True,default=str).encode()).hexdigest()


def prepare_results(engine,rule,mode,key):
    folder=CANONICAL/'candidate_cache'/mode;folder.mkdir(parents=True,exist_ok=True)
    cfg=engine.cfg;state=json.loads((ROOT/'outputs/iteration_4/us_large/decision.json').read_text())['candidate_state']
    panel=engine.panel_for(state,folder);cov=engine.covariance_for(panel)
    pm=dataclasses.replace(cfg.portfolio_ml,benchmark_relative=True,add_constant=False,half_life_years=(None,))
    cfg=dataclasses.replace(cfg,portfolio_ml=pm);inp=grid_inputs(cfg,panel,engine.wealth,cov,engine.rf)
    end=pd.Timestamp(rule['pretest_end'] if mode=='pretest' else rule['test_end'])
    dates=inp.pfml_dates[inp.pfml_dates+MONTH<=end]
    years=[y for y in inp.hp_years if y<=end.year]
    signals=dates[(dates+MONTH).year>=years[0]]
    results={};catalog=[];proxy=[];offset=0
    for g in pm.rff_variance_grid:
        tag=f'rff_g{g:.4f}';cache=folder/f'{tag}_moments.npz';manifest=folder/f'{tag}_moments.json'
        if manifest.exists() and cache.exists() and json.loads(manifest.read_text()).get('signature')==key:
            if digest(cache)!=json.loads(manifest.read_text())['sha256']:raise ValueError('Corrupt moment cache')
            ym=load_yearly_moments(cache)
        else:
            log.info('Build original moments %s %s through %s',mode,tag,end.date())
            ym=build_yearly_moments(inp.panel,engine.wealth,cov,engine.rf,dates,cfg.features.names,signal_dates=signals,
                gamma=rule['gamma_te'],mu=cfg.investor.mu,lb=cfg.universe.lookback_months,
                iterations=cfg.investor.m_iterations,scale_by_vol=pm.scale_by_vol,add_constant=False,**_signal_kwargs(cfg,g))
            save_yearly_moments(ym,cache);write_json(manifest,dict(signature=key,sha256=digest(cache)))
        result=search_yearly(ym,tag,pm.p_grid,cfg.ridge_grid.values,years,(None,));del ym
        results[tag]=result
        rows=[]
        for p in pm.p_grid:
            for no,l in enumerate(cfg.ridge_grid.values):
                rows.append(dict(cid=offset+len(rows),tag=tag,g=g,p=p,l_no=no,l=float(l),h=np.inf))
        cat=pd.DataFrame(rows);offset+=len(cat);catalog.append(cat)
        proxy.append(result.validation.merge(cat[['cid','tag','p','l_no']],on=['tag','p','l_no'],validate='many_to_one'))
    catalog=pd.concat(catalog,ignore_index=True);catalog.to_csv(folder/'catalog.csv',index=False)
    pd.concat(proxy,ignore_index=True).to_csv(folder/'proxy_scores.csv',index=False)
    return folder,cfg,inp,signals,results,catalog


def replay_candidates(panel,wealth,cov,rf,dates,result,catalog,cfg,gamma,orders_dir):
    """Vectorised original execution; columns are fixed-hyperparameter shadow books."""
    inputs=month_inputs(panel,wealth,cov,dates,mu_col=None)
    by_date={d:g.set_index('id') for d,g in panel.loc[panel.eom.isin(dates),['eom','id','lambda','ret_ld1','tr_ld1']].groupby('eom')}
    capital=wealth.set_index('eom');last_ids=None;last_start=None;frames=[]
    orders_dir.mkdir(parents=True,exist_ok=True);M=len(catalog)
    for j,date in enumerate(dates):
        x=inputs[date];ids=x.ids;current=by_date[date];p=current.reindex(ids)
        me=x.panel_t.set_index('id').me.reindex(ids).to_numpy();b=me/me.sum()
        if last_ids is None:
            held=np.repeat(b[:,None],M,axis=1);exit_ids=np.array([],dtype=ids.dtype);exit_start=np.zeros((0,M))
        else:
            pos=pd.Index(last_ids).get_indexer(ids);found=pos>=0;held=np.zeros((len(ids),M));held[found]=last_start[pos[found]]
            leaving=~pd.Index(last_ids).isin(ids)&pd.Index(last_ids).isin(current.index)
            if not cfg.portfolio.charge_exit_tc:leaving[:]=False
            exit_ids=last_ids[leaving];exit_start=last_start[leaving]
        signal_ids,S=result.signals[date]
        np.testing.assert_array_equal(ids,signal_ids)
        year=(date+MONTH).year
        aims=np.hstack([S[:,model_columns(result,int(pp))]@result.coefs[year][int(pp)] for pp in catalog.p.drop_duplicates()])
        assert aims.shape==held.shape
        m=trading_speed(x.sigma,x.lam,x.wealth,gamma,float(rf[date+MONTH]),cfg.investor.mu,cfg.investor.m_iterations)
        W=b[:,None]+m@(held-b[:,None])+aims-m@aims
        all_W=np.vstack([W,np.zeros_like(exit_start)]);H=np.vstack([held,exit_start])
        all_ids=np.r_[ids,exit_ids];ex=np.r_[np.zeros(len(ids),dtype=bool),np.ones(len(exit_ids),dtype=bool)]
        ret=np.r_[p.ret_ld1.to_numpy(),np.zeros(len(exit_ids))]
        lam=current['lambda'].reindex(all_ids).to_numpy();D=all_W-H
        if not np.isfinite(np.r_[all_W.ravel(),H.ravel(),ret,lam]).all():raise ValueError('Nonfinite candidate book')
        r=ret@all_W;tc=.5*x.wealth*(lam[:,None]*D**2).sum(axis=0)
        frame=pd.DataFrame(dict(eom_ret=date+MONTH,cid=catalog.cid.to_numpy(),r=r,tc=tc,
            turnover=np.abs(D).sum(axis=0),inv=np.abs(all_W).sum(axis=0),shorting=np.maximum(-all_W,0).sum(axis=0)))
        frames.append(frame)
        np.savez_compressed(orders_dir/f'{date:%Y-%m-%d}.npz',ids=all_ids,W=all_W,H=H,exit=ex,ret=ret,lam=lam,
            wealth=x.wealth,cids=catalog.cid.to_numpy())
        growth=(1+p.tr_ld1.to_numpy())/(1+float(capital.loc[date,'mu_ld1']))
        if not np.isfinite(growth).all():raise ValueError('Missing stock/wealth drift')
        last_ids=ids;last_start=W*growth[:,None]
        if j%12==11:log.info('Executed %s through %s (%d candidate columns)',result.tag,date.date(),M)
    return pd.concat(frames,ignore_index=True)


def candidate_cache(engine,rule,mode,key):
    folder,cfg,inp,dates,results,catalog=prepare_results(engine,rule,mode,key)
    marker=folder/'completion.json';names=('candidate_flows.parquet','market_pf.parquet','catalog.csv','proxy_scores.csv')
    if marker.exists():
        old=json.loads(marker.read_text())
        if old.get('signature')==key and all((folder/n).exists() and digest(folder/n)==h for n,h in old['hashes'].items()):
            log.info('Reuse executed candidate cashflows: %s',mode)
            return folder,cfg,inp,dates,results,catalog,pd.read_parquet(folder/'candidate_flows.parquet')
    tg=market_targets(inp.panel.loc[inp.panel.in_universe&inp.panel.eom.isin(dates)])
    w=add_start_weights(tg,inp.panel,engine.wealth,dates,cfg.portfolio.charge_exit_tc)
    market=portfolio_returns(w,inp.panel,engine.wealth);market.to_parquet(folder/'market_pf.parquet',index=False)
    w.to_parquet(folder/'market_weights.parquet',index=False)
    frames=[]
    for tag,res in results.items():
        frames.append(replay_candidates(inp.panel,engine.wealth,engine.covariance_for(inp.panel),engine.rf,dates,res,
            catalog.loc[catalog.tag.eq(tag)],cfg,rule['gamma_te'],folder/'orders'/tag))
    flows=pd.concat(frames,ignore_index=True).merge(catalog,on='cid',validate='many_to_one').merge(
        market[['eom_ret','r','tc']].rename(columns={'r':'r_b','tc':'tc_b'}),on='eom_ret',validate='many_to_one')
    flows['active_gross']=flows.r-flows.r_b;flows['active_net']=(flows.r-flows.tc)-(flows.r_b-flows.tc_b)
    flows=flows.sort_values(['eom_ret','cid']).reset_index(drop=True)
    flows.to_parquet(folder/'candidate_flows.parquet',index=False)
    write_json(marker,dict(signature=key,hashes={n:digest(folder/n) for n in names},
        candidate_count=len(catalog),months=len(dates),coefficient_fit_unchanged=True))
    return folder,cfg,inp,dates,results,catalog,flows


def execute_policy(engine,cfg,inp,dates,results,chosen,folder):
    aims,coefs=aim_portfolios(results,chosen,dates)
    cov=engine.covariance_for(inp.panel);gamma=json.loads(PROTOCOL.read_text())['gamma_te']
    weights=pfml_weights(month_inputs(inp.panel,engine.wealth,cov,dates,mu_col=None),aims,inp.panel,engine.wealth,
        engine.rf,gamma=gamma,mu=cfg.investor.mu,iterations=cfg.investor.m_iterations,
        charge_exit_tc=cfg.portfolio.charge_exit_tc,benchmark_relative=True)
    pf=portfolio_returns(weights,inp.panel,engine.wealth).assign(type='Portfolio-ML')
    folder.mkdir(parents=True,exist_ok=True)
    save_result(dict(pf=pf,weights=weights.assign(type='Portfolio-ML'),aims=aims,aim_coefs=coefs,chosen=chosen),folder)
    ex=ex_ante_exposures(active_weights(weights,inp.panel),cov)
    ex.to_csv(folder/'ex_ante_exposures.csv',index=False)
    return pf


def report_stage(stage,engine,rule,chosen,audit,pf):
    folder=ROOT/f'outputs/iteration_{stage}/us_large'
    reference=ROOT/('outputs/iteration_4/us_large' if stage=='4.2' else 'outputs/iteration_4.2/us_large')
    previous=pd.read_parquet(reference/'pf.parquet');market=pd.read_parquet(ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet')
    frames={'Candidate':pf,'Previous stage':previous,'Iteration 4':pd.read_parquet(ROOT/'outputs/iteration_4/us_large/pf.parquet'),
        'Base Portfolio ML':pd.read_parquet(ROOT/'outputs/base/us_large/portfolio_ml/rff/pf.parquet'),
        'Static ML*':pd.read_parquet(ROOT/'outputs/base/us_large/static_ml_star/pf.parquet'),'Market':market}
    table=pd.DataFrame([dict(strategy=k,**statistics(v)) for k,v in frames.items()]);table.to_csv(folder/'performance_summary.csv',index=False)
    pd.concat([v.assign(strategy=k) for k,v in frames.items()],ignore_index=True).to_csv(folder/'monthly_returns.csv',index=False)
    pd.DataFrame([dict(strategy=k,year=y,**statistics(g)) for k,v in frames.items() for y,g in v.groupby(v.eom_ret.dt.year)]).to_csv(folder/'annual_performance.csv',index=False)
    plot_comparison(frames,folder/'charts',f'Iteration {stage}: executed active validation'+(' + one-SE simplicity' if stage=='4.3' else ''))
    compare=paired_probability(pf,previous,rule['p_test_reps'],rule['p_test_block_months'],rule['p_test_seed'])
    secondary=paired_probability(pf,frames['Iteration 4'],rule['p_test_reps'],rule['p_test_block_months'],rule['p_test_seed'])
    parent=json.loads((ROOT/'outputs/iteration_4/us_large/decision.json').read_text())
    inherited=list(parent['included_after'])+(['4.2'] if stage=='4.3' else [])
    if 2 in inherited:raise ValueError('Iteration 2 must not enter this Base-anchored branch')
    decision=dict(number=stage,name='Executed active-utility validation' if stage=='4.2' else 'One-SE executed active-utility selection',
        reference='Iteration 4' if stage=='4.2' else 'Iteration 4.2',p_better=compare['probability'],bootstrap=compare,
        secondary_p_vs_iteration_4=secondary,threshold=None,p_is_conventional_p_value=False,
        accepted=True,retention_is_statistical_acceptance=False,verdict='Retained as research stage; P(test) descriptive',
        inherited_iterations=inherited,included_after=[*inherited,stage],
        candidate_state=dict(benchmark_relative=True,gamma_te=rule['gamma_te'],selector='executed_one_se' if stage=='4.3' else 'executed_max_utility'),
        base_parent_excludes_iteration_2=True,prior_trial_results_preserved=True,independent_unseen_confirmation=False)
    write_json(folder/'decision.json',decision);write_json(folder/'protocol.json',rule)
    chosen.to_csv(folder/'chosen.csv',index=False);audit.to_csv(folder/'selection_audit.csv',index=False)
    ex=pd.read_csv(folder/'ex_ante_exposures.csv')
    write_json(folder/'risk_summary.json',dict(frozen_gamma_te=rule['gamma_te'],nominal_original_te_target=.01,
        mean_test_ex_ante_te=float(np.sqrt(12)*ex.te.mean()),risk_target_recalibrated=False))
    p=compare['probability'];ptext='N/A (identical)' if p is None else f'{p:.2%}'
    text=f'# Iteration {stage}: {decision["name"]}\n\nInherits {inherited}. Compared with {decision["reference"]}.\n\n'
    text+=f'P(test): {ptext}. Retained as a research stage under the user instruction; there is no 80% gate and this is not statistical acceptance.\n\n'
    text+='The coefficient of active risk remains 174.1; actual tracking error can differ from the original nominal 1% target. Annual choices use only preceding returns.\n\n'
    text+='Candidate books start in 2015 and carry their own annual coefficient changes; the actual test policy starts in 2020 and charges its own hyperparameter-switch costs. These inherited-state differences remain a validation approximation.\n\n'
    text+='For 4.3, the one-SE set is a fixed robustness heuristic, not a hypothesis test or a multiple-comparison correction. P(test) is paired circular-block bootstrap frequency of a higher paper-convention net Sharpe, not a conventional p-value or future-outperformance probability.\n\n'
    text+='The current chain starts from Base and excludes iteration 2. Earlier trial results remain historical evidence. This branch extends benchmark-relative validation; it is not exact paper replication or independent unseen confirmation.\n'
    (folder/'SUMMARY.md').write_text(text,encoding='utf-8');(folder/'README.md').write_text(text,encoding='utf-8')
    log.info('Stage %s: P(test) %s; candidate net SR %.6f',stage,ptext,table.iloc[0].SR_net)


def preserve_snapshot(folder):
    path=folder/'prior_source_hashes.json'
    if path.exists():return json.loads(path.read_text())
    files=[]
    for group in ['base',*[f'iteration_{i}' for i in range(1,11)]]:
        files.extend(p for p in (ROOT/'outputs'/group).rglob('*') if p.is_file() and p.name in
            {'pf.parquet','weights.parquet','chosen.csv','aim_coefs.parquet','decision.json'})
    snapshot={str(p.relative_to(ROOT)):digest(p) for p in files};write_json(path,snapshot);return snapshot


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['4.2','4.3','both'],default='both')
    args=parser.parse_args(argv);logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    rule=json.loads(PROTOCOL.read_text());CANONICAL.mkdir(parents=True,exist_ok=True)
    before=preserve_snapshot(CANONICAL);engine=ExperimentEngine();key=source_key(engine,rule)
    reference_gamma=float(pd.read_csv(ROOT/'outputs/iteration_4/us_large/calibrated.csv').gamma_te.iloc[0])
    if rule['gamma_te']!=reference_gamma:raise ValueError('This branch retains the original gamma_TE')
    pre=candidate_cache(engine,rule,'pretest',key);_,cfg,inp,dates,results,catalog,flows=pre
    if pd.Timestamp(flows.eom_ret.max())>pd.Timestamp(rule['pretest_end']):raise ValueError('Future returns in diagnosis')
    for stage in ('4.2','4.3'):
        folder=ROOT/f'outputs/iteration_{stage}/us_large';folder.mkdir(parents=True,exist_ok=True)
        chosen,audit=select_executed(flows,catalog,rule,stable=stage=='4.3',max_year=2020)
        audit.to_csv(folder/'diagnostic_selection_audit.csv',index=False)
        chosen.to_csv(folder/'diagnostic_chosen.csv',index=False)
    from active_validation_rationale import build_rationales
    build_rationales(pre[0],ROOT/'outputs/iteration_4.2/us_large/diagnostic_selection_audit.csv',
        ROOT/'outputs/iteration_4.3/us_large/diagnostic_selection_audit.csv')
    # Only after rationale images and protocol exist do we build the test-period histories.
    full=candidate_cache(engine,rule,'full',key);_,cfg,inp,dates,results,catalog,flows=full
    for stage in (('4.2',) if args.stage=='4.2' else ('4.2','4.3')):
        folder=ROOT/f'outputs/iteration_{stage}/us_large';chosen,audit=select_executed(flows,catalog,rule,stable=stage=='4.3')
        eligible_dates=dates[(dates+MONTH).year>=2016]
        pre_dates=eligible_dates[eligible_dates+MONTH<=pd.Timestamp(rule['pretest_end'])]
        execute_policy(engine,cfg,inp,pre_dates,results,chosen,folder/'pretest')
        pf=execute_policy(engine,cfg,inp,inp.eval_dates,results,chosen,folder)
        report_stage(stage,engine,rule,chosen,audit,pf)
        verify_accounting(folder,inp.panel,engine.wealth)
        names=['pf.parquet','weights.parquet','chosen.csv','aim_coefs.parquet','selection_audit.csv']
        write_json(folder/'fit_manifest.json',dict(signature=key,stage=stage,hashes={n:digest(folder/n) for n in names}))
    changed=[p for p,h in before.items() if digest(ROOT/p)!=h]
    if changed:raise AssertionError(f'Earlier portfolios changed: {changed}')
    write_json(CANONICAL/'preservation_verification.json',dict(status='PASS',files_checked=len(before),changed=[]))
    return 0


if __name__=='__main__':raise SystemExit(main())
