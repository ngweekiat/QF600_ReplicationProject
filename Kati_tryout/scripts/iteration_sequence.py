"""Five sequential US high-cap experiments, with test-only P(better) and explicit inheritance.

Existing base models, input dates and characteristics are read-only. The retained
strategy changes only when a candidate's paired bootstrap Sharpe comparison reaches
the declared threshold. Rejected candidates remain fully reported.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import norm

from workflow import ROOT, config_path
from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.data.prepare import risk_free_monthly
from jkmp.portfolio.grids import grid_inputs, _signal_kwargs
from jkmp.portfolio.portfolio_ml import build_yearly_moments, search_yearly, run_portfolio_ml
from jkmp.portfolio.active import calibrate
from jkmp.evaluation.metrics import active_weights, ex_ante_exposures
from analysis.ml_comparison.inference import _bootstrap_differences
from analysis.ml_comparison.metrics import statistics

log=logging.getLogger(__name__)
PROTOCOL=ROOT/'configs/us_high_cap_iterations.json'
OUT=ROOT/'outputs/iteration_summary'
BASE=ROOT/'outputs/base/us_large/portfolio_ml/rff'
MONTH=pd.offsets.MonthEnd(1)
STYLE='<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:35px auto;color:#203044;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd5de;padding:9px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#203451;color:white}img{max-width:100%}.note{background:#edf3f8;padding:15px}a{color:#126084}</style>'


def write_json(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Output outside Kati')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.writing.json')
    temporary.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def default_state(cfg):
    return dict(gamma=cfg.investor.gamma,exit_rank=cfg.universe.exit_rank,cost_multiple=1.0,
                benchmark_relative=False,te_target=None,half_lives=[None])


def candidate_state(previous,overrides):
    return {**previous,**overrides}


def inheritance(previous,number,accepted):
    return [*previous,number] if accepted and number not in previous else list(previous)


def paired_probability(candidate,reference,reps=10000,block=6,seed=20261102):
    """Paired bootstrap support, not a posterior or a conventional p-value."""
    a=candidate.sort_values('eom_ret').set_index('eom_ret')
    b=reference.sort_values('eom_ret').set_index('eom_ret')
    if not a.index.equals(b.index) or len(a)<2:raise ValueError('Paired calendars differ')
    if np.array_equal(a[['r','tc']],b[['r','tc']]):
        return dict(probability=None,ci_low=0.,ci_high=0.,identical=True,valid_reps=reps)
    _,delta=_bootstrap_differences(a.r.to_numpy(),a.tc.to_numpy(),b.r.to_numpy(),b.tc.to_numpy(),
                                  10.,reps,block,np.random.default_rng(seed))
    delta=delta[np.isfinite(delta)]
    if len(delta)!=reps:raise ValueError('Undefined bootstrap Sharpe replicates')
    # Half weight to numerical ties; equality cannot manufacture evidence.
    tied=np.isclose(delta,0.,rtol=0.,atol=1e-12)
    p=float(np.mean((delta>1e-12).astype(float)+.5*tied))
    lo,hi=np.quantile(delta,[.025,.975])
    return dict(probability=p,ci_low=float(lo),ci_high=float(hi),identical=False,valid_reps=len(delta))


def buffered_membership(screened,exit_rank,target_n=80,entry_rank=70):
    """Keats source timing: December formation selection, held December--November.

    Rankings use contemporaneous market cap and original screened row order.
    No forward target is consulted. Missing calendar rows terminate a membership.
    """
    dec=screened[screened.eom.dt.month.eq(12)].copy()
    dec['mcap_rank']=dec.groupby('eom').me.rank(ascending=False,method='first')
    previous=set();decisions=[]
    for eom,g in dec.groupby('eom',sort=True):
        ranks=g.set_index('id').mcap_rank
        retained={i for i in previous if i in ranks.index and ranks[i]<=exit_rank}
        entrants=ranks[ranks.le(entry_rank)].drop(list(retained),errors='ignore')
        chosen=retained|set(entrants.index)
        fill=ranks.drop(list(chosen),errors='ignore').sort_values(kind='stable')
        chosen |= set(fill.index[:max(0,target_n-len(chosen))])
        decisions.append(g[g.id.isin(chosen)][['id','mcap_rank']].assign(form_year=eom.year))
        previous=chosen
    membership=pd.concat(decisions,ignore_index=True)
    rows=screened[['id','eom']].copy()
    rows['form_year']=np.where(rows.eom.dt.month.eq(12),rows.eom.dt.year,rows.eom.dt.year-1)
    rows=rows.merge(membership,on=['id','form_year'],validate='many_to_one').sort_values(['id','form_year','eom'])
    prior=rows.groupby(['id','form_year']).eom.shift()
    gap=prior.notna()&(prior+MONTH!=rows.eom)
    return rows[~gap.groupby([rows.id,rows.form_year]).cummax()]


class ExperimentEngine:
    def __init__(self):
        self.cfg=load_config(config_path('us_large'))
        p=self.cfg.paths.processed
        # Portfolio fitting uses only stocks ever held under a declared rule.
        # Keep that exact union at read time, not million-row copies per trial.
        members=pd.read_parquet(p/'panel.parquet',columns=['id'],filters=[('in_universe','=',True)])
        ids=set(members.id.unique())
        source=pd.read_parquet(ROOT/'data/source/screened_US.parquet',columns=['id','eom','me'])
        declared=json.loads(PROTOCOL.read_text())['iterations']
        self._memberships={}
        for rank in {s.get('override',{}).get('exit_rank') for s in declared}-{None}:
            membership=buffered_membership(source,rank,self.cfg.universe.target_n,self.cfg.universe.entry_rank)
            self._memberships[rank]=membership
            ids.update(membership.id.unique())
        ids=sorted(ids)
        columns=list(dict.fromkeys(['id','eom','me','in_universe','lambda','g','tr_ld1','ret_ld1',*self.cfg.features.names]))
        self.panel=pd.read_parquet(p/'panel.parquet',columns=columns,filters=[('id','in',ids)])
        preds=pd.read_parquet(p/'predictions.parquet',columns=['id','eom','pred_ld1'],filters=[('id','in',ids)])
        self.panel=self.panel.merge(preds,on=['id','eom'],how='left',validate='one_to_one')
        self.wealth=pd.read_parquet(p/'wealth.parquet')
        self.cov=load_covariance(p/'covariance')
        self.rf=risk_free_monthly(pd.read_parquet(self.cfg.paths.raw/'ff_factors_monthly.parquet'))
        self._panels={self.cfg.universe.exit_rank:self.panel}
        self._expanded_cov=None
        paths=[p/'panel.parquet',p/'wealth.parquet',p/'covariance/cov_loadings.parquet',p/'covariance/cov_factor.parquet',BASE/'pf.parquet',PROTOCOL,Path(__file__)]
        self.signature=hashlib.sha256(json.dumps({str(x):digest(x) for x in paths},sort_keys=True).encode()).hexdigest()

    def panel_for(self,state,folder):
        rank=state['exit_rank']
        if rank not in self._panels:
            from jkmp.data.lag_history import exclude_unavailable
            from workflow import zero_missing_held_returns
            if rank not in self._memberships:raise ValueError('Exit-rank change is not in the declared protocol')
            membership=self._memberships[rank]
            keys=pd.MultiIndex.from_frame(membership[['id','eom']])
            result=self.panel.copy()
            result['in_universe']=pd.MultiIndex.from_frame(result[['id','eom']]).isin(keys)
            result,excluded=exclude_unavailable(result,'us_large',self.cfg.universe.lookback_months)
            result=zero_missing_held_returns(result,self.rf)
            if not result.loc[result.in_universe,['g','lambda','ret_ld1','me']].notna().all().all():
                raise ValueError('New buffered members lack required observed current inputs')
            folder.mkdir(parents=True,exist_ok=True)
            membership.to_csv(folder/'membership.csv',index=False)
            excluded.to_csv(folder/'history_exclusions.csv',index=False)
            self._panels[rank]=result
        return self._panels[rank].assign(**{'lambda':self._panels[rank]['lambda']*state['cost_multiple']})

    def covariance_for(self,panel):
        missing=any(not set(g.id).issubset(set(self.cov[d].ids)) for d,g in
                    panel[panel.in_universe&panel.eom.isin(self.cov)].groupby('eom'))
        if not missing:return self.cov
        if self._expanded_cov is None:
            path=ROOT/'data/processed/us_large/universe_250/covariance'
            if not path.exists():raise FileNotFoundError('Extended buffered members require the existing wider covariance cache')
            extended=load_covariance(path)
            for d,c in self.cov.items():
                common=np.intersect1d(c.ids,extended[d].ids)
                np.testing.assert_allclose(c.full(common),extended[d].full(common),rtol=1e-10,atol=1e-12)
            self._expanded_cov=extended
        return self._expanded_cov

    def fit(self,state,folder,*,calibration=False,gamma_te=None):
        cfg=self.cfg
        panel=self.panel_for(state,folder)
        cov=self.covariance_for(panel)
        pm=dataclasses.replace(cfg.portfolio_ml,benchmark_relative=state['benchmark_relative'],
             add_constant=not state['benchmark_relative'],half_life_years=tuple(state['half_lives']))
        cfg=dataclasses.replace(cfg,portfolio_ml=pm)
        inp=grid_inputs(cfg,panel,self.wealth,cov,self.rf)
        gamma=float(gamma_te if state['benchmark_relative'] else state['gamma'])
        if calibration:
            end=pd.Timestamp(cfg.dates.validation_end)
            dates=inp.pfml_dates[inp.pfml_dates+MONTH<=end]
            years=[y for y in inp.hp_years if y<=end.year]
            aim=dates[(dates+MONTH).year>years[0]]
        else:
            dates=inp.pfml_dates;years=inp.hp_years
            aim=dates[(dates+MONTH).year>years[0]]
        evaluation=aim if calibration else inp.eval_dates
        results={}
        for g in pm.rff_variance_grid:
            tag=f'rff_g{g:.4f}'
            log.info('Fit %s: gamma=%g %s; %d months',folder.name,gamma,tag,len(dates))
            ym=build_yearly_moments(inp.panel,self.wealth,cov,self.rf,dates,cfg.features.names,
                signal_dates=aim,gamma=gamma,mu=cfg.investor.mu,lb=cfg.universe.lookback_months,
                iterations=cfg.investor.m_iterations,scale_by_vol=pm.scale_by_vol,
                add_constant=pm.add_constant,**_signal_kwargs(cfg,g))
            results[tag]=search_yearly(ym,tag,pm.p_grid,cfg.ridge_grid.values,years,pm.half_life_years)
            del ym
        res=run_portfolio_ml(results,inp.panel,self.wealth,cov,self.rf,aim,evaluation,
            gamma=gamma,mu=cfg.investor.mu,iterations=cfg.investor.m_iterations,
            charge_exit_tc=cfg.portfolio.charge_exit_tc,benchmark_relative=state['benchmark_relative'])
        if calibration:
            ex=ex_ante_exposures(active_weights(res['weights'],inp.panel),cov)
            return dict(gamma_te=gamma,te_ex_ante=float(ex.te.mean()*np.sqrt(12)),
                        months=len(evaluation),last_return=str((evaluation[-1]+MONTH).date()))
        for name,frame in res.items():
            if name=='chosen':frame.to_csv(folder/'chosen.csv',index=False)
            else:frame.to_parquet(folder/f'{name}.parquet',index=False)
        return res

    def run_candidate(self,state,folder):
        folder.mkdir(parents=True,exist_ok=True)
        key=hashlib.sha256(json.dumps(dict(state=state,source=self.signature),sort_keys=True).encode()).hexdigest()
        marker=folder/'fit_manifest.json'
        files=['pf.parquet','weights.parquet','chosen.csv']
        if marker.exists():
            old=json.loads(marker.read_text())
            if old.get('signature')==key and all((folder/f).exists() and digest(folder/f)==old.get('hashes',{}).get(f) for f in files):
                log.info('Reuse verified candidate %s',folder)
                return pd.read_parquet(folder/'pf.parquet')
        gamma_te=None
        if state['benchmark_relative']:
            points=[]
            # Use a bracket, not an OOS-selected TE risk penalty.
            for g in (100.,250.,50.,500.,25.,1000.,10.,2500.):
                point=self.fit(state,folder,calibration=True,gamma_te=g)
                points.append(point)
                table=pd.DataFrame(points)
                table.to_csv(folder/'calibration.csv',index=False)
                if len(points)>1 and table.te_ex_ante.min()<=state['te_target']<=table.te_ex_ante.max():break
            chosen=calibrate(pd.DataFrame(points),[state['te_target']])
            chosen.to_csv(folder/'calibrated.csv',index=False)
            gamma_te=float(chosen.gamma_te.iloc[0])
        self.fit(state,folder,gamma_te=gamma_te)
        write_json(marker,dict(signature=key,state=state,gamma_te=gamma_te,
                              hashes={f:digest(folder/f) for f in files}))
        return pd.read_parquet(folder/'pf.parquet')


def verify_accounting(folder,panel,wealth):
    """Reconstruct each stock's return/cost directly, independently of the model optimiser."""
    w=pd.read_parquet(folder/'weights.parquet')
    pf=pd.read_parquet(folder/'pf.parquet').sort_values('eom_ret').set_index('eom_ret')
    d=w.merge(panel[['id','eom','ret_ld1','lambda']],on=['id','eom'],how='left',validate='many_to_one').merge(wealth[['eom','wealth']],on='eom',validate='many_to_one')
    if d['lambda'].isna().any() or d.loc[~d.exit,'ret_ld1'].isna().any():raise ValueError('Incomplete accounting inputs')
    dw=d.w-d.w_start
    d=d.assign(r=d.w*d.ret_ld1.where(~d.exit,0),tc=.5*d.wealth*d['lambda']*dw**2,
               inv=d.w.abs(),shorting=d.w.clip(upper=0).abs(),turnover=dw.abs())
    recomputed=d.groupby('eom')[['r','tc','inv','shorting','turnover']].sum()
    recomputed.index+=MONTH
    np.testing.assert_allclose(recomputed.loc[pf.index],pf[recomputed.columns],rtol=1e-10,atol=1e-12)
    if not pf.index.equals(pd.date_range('2020-01-31','2025-12-31',freq='ME')):raise ValueError('Expected all 72 existing test months')
    write_json(folder/'accounting_verification.json',dict(status='PASS',months=len(pf),metrics=list(recomputed.columns)))


def plot_comparison(frames,out,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.1))
    market=frames['Market'].sort_values('eom_ret')
    mb=(1+market.r-market.tc).cumprod().to_numpy()
    for label,pf in frames.items():
        d=pf.sort_values('eom_ret');net=d.r-d.tc;path=(1+net).cumprod().to_numpy()
        peak=np.maximum.accumulate(np.r_[1.,path])[1:]
        axes[0].plot(d.eom_ret,100*(path-1),label=label)
        axes[1].plot(d.eom_ret,100*(path/peak-1),label=label)
        if label!='Market':axes[2].plot(d.eom_ret,100*(path/mb-1),label=label)
    for ax in axes:ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25);ax.set_ylabel('%')
    axes[0].set_title('Compounded net excess-return index')
    axes[1].set_title('Drawdown from previous peak')
    axes[2].set_title('Cumulative relative excess index vs Market');axes[2].axhline(0,color='grey',lw=.8)
    axes[0].legend(fontsize=7);fig.suptitle(title);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'net_returns_and_costs.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,3.5))
    for label,pf in frames.items():
        d=pf.sort_values('eom_ret');axes[0].plot(d.eom_ret,100*d.tc.cumsum(),label=label)
        axes[1].plot(d.eom_ret,d.turnover,label=label)
    axes[0].set_title('Cumulative modelled costs (%)');axes[1].set_title('Monthly turnover')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(out/f'costs_and_turnover.{suffix}',dpi=160)
    plt.close(fig)


def write_iteration_report(number,decision,candidate,reference,base,static,market):
    folder=ROOT/f'outputs/iteration_{number}/us_large'
    frames={'Candidate':candidate,'Previous accepted':reference,'Base Portfolio ML':base,'Static ML*':static,'Market':market}
    table=pd.DataFrame([dict(strategy=name,**statistics(pf)) for name,pf in frames.items()])
    table.to_csv(folder/'performance_summary.csv',index=False)
    pd.concat([p.assign(strategy=n) for n,p in frames.items()],ignore_index=True).to_csv(folder/'monthly_returns.csv',index=False)
    pd.DataFrame([dict(strategy=n,year=y,**statistics(g)) for n,p in frames.items() for y,g in p.groupby(p.eom_ret.dt.year)]).to_csv(folder/'annual_performance.csv',index=False)
    plot_comparison(frames,folder/'charts',f'Iteration {number}: {decision["name"]}')
    inherited=decision['inherited_iterations']
    narrative=f'Iteration 1 (annual refitting) is already part of iteration {number} and Base.'
    narrative+=f' Other inherited accepted changes: {[i for i in inherited if i!=1] or "none"}.'
    narrative+=f' Compared against: {decision["reference"]}. Verdict: {decision["verdict"]}.'
    narrative+=f' Included in subsequent accepted setup: {decision["accepted"]}.'
    if decision['candidate_state']['benchmark_relative']:
        narrative+=' The tilt uses calibrated gamma_TE; an inherited gamma-5 decision is not an extra multiplier of active risk.'
    text=f'# Iteration {number}: {decision["name"]}\n\n{narrative}\n\nP(better net Sharpe): {decision["p_better"] if decision["p_better"] is not None else "N/A (already in Base or identical)"}. Acceptance threshold: 80%.\n\n'+METHODS
    (folder/'SUMMARY.md').write_text(text,encoding='utf-8')
    (folder/'README.md').write_text(text,encoding='utf-8')
    body=f'<h1>Iteration {number}: {html.escape(decision["name"])}</h1><p class="note">{html.escape(narrative)}</p><p>P(better): {decision["p_better"] if decision["p_better"] is not None else "N/A"}</p>'
    body+=table.to_html(index=False,float_format=lambda v:f'{v:.5g}')+'<img src="charts/net_returns_and_costs.png"><img src="charts/costs_and_turnover.png">'
    body+='<p><a href="SUMMARY.md">Summary and methods</a> | <a href="decision.json">Decision/inheritance</a> | <a href="performance_summary.csv">Performance CSV</a> | <a href="monthly_returns.csv">Monthly returns</a> | <a href="../../iteration_summary/index.html">All five experiments</a></p>'
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+body,encoding='utf-8')


METHODS='''All comparisons use the existing 72 test return months, January 2020--December 2025. Kati OOS labels overlap this test history; no separate unseen confirmation is claimed. Annual expanding coefficient refits and monthly portfolio updates are preserved. P(better) is the fraction of 10,000 paired circular-block bootstrap samples (six-month blocks, seed 20261102) with a positive net-Sharpe difference, with half weight to numerical ties. It is not a frequentist p-value or a Bayesian probability of future outperformance. Net Sharpe uses gross-return volatility, following the paper convention. Acceptance is P(better) >= 80% against the last accepted version, not against Market. Identical strategies are not accepted as an improvement. Iteration 1 is retained as an existing design, without a statistical claim. Drawdown is descriptive and adds no rejection gate. Decisions use already inspected test outcomes; subsequent reported performance on those same months is selected-sample evidence, not independent validation. The experiment log retains rejected trials. Nested hyperparameter searches and earlier research choices are not fully corrected by these five comparisons. Compounded excess-return paths and drawdowns are synthetic excess indices, not total investor wealth. TE 1% is an ex-ante pretest calibration target, not a hard realised cap. Base markets, stock screens, selected 50, dates, costs and risk equations are retained except for each explicitly tested override. Exit 120 changes buffering within the same US large-cap candidate population and may change the stock count; costs x2 are applied both when fitting and when trading. The original wider covariance cache may supply extra buffered stocks, only after equality checks against common base covariance entries. No 250-stock breadth experiment is part of this five-step sequence.'''


def report_saved(upto=5):
    protocol=json.loads(PROTOCOL.read_text())
    decisions=[]
    base=pd.read_parquet(BASE/'pf.parquet')
    market=pd.read_parquet(ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet')
    for spec in protocol['iterations'][:upto]:
        folder=ROOT/f'outputs/iteration_{spec["number"]}/us_large'
        if (folder/'decision.json').exists():decisions.append(json.loads((folder/'decision.json').read_text()))
    if not decisions:return 0
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for d in decisions:
        pf=pd.read_parquet(ROOT/f'outputs/iteration_{d["number"]}/us_large/pf.parquet')
        s=statistics(pf)
        rows.append(dict(iteration=d['number'],change=d['name'],Sharpe=s['SR_net'],P_better=d['p_better'],
            worst_drawdown=s['max_drawdown_excess'],verdict=d['verdict'],included=d['accepted'],
            reference=d['reference'],inherited_iterations=','.join(map(str,d['inherited_iterations'])),
            final_included_iterations=','.join(map(str,d['included_after'])),annual_net_return=s['R_TC']))
    table=pd.DataFrame(rows);table.to_csv(OUT/'statistical_tests_all_iterations.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chart_table=table[['iteration','change','Sharpe','P_better','worst_drawdown','verdict']].copy()
    chart_table['Sharpe']=chart_table.Sharpe.map(lambda x:f'{x:.3f}')
    chart_table['P_better']=chart_table.P_better.map(lambda x:'N/A' if pd.isna(x) else f'{x:.1%}')
    chart_table['worst_drawdown']=chart_table.worst_drawdown.map(lambda x:f'{x:.1%}')
    chart_table['verdict']=chart_table.verdict.map(lambda x:'Already in Base; retained as design' if 'Annual refitting' in x else 'Kept' if x=='Kept' else 'Not included')
    fig,ax=plt.subplots(figsize=(15,3.8));ax.axis('off')
    chart=ax.table(cellText=chart_table.values,colLabels=['Step','Change','Net Sharpe','P(better)','Worst drawdown','Verdict'],
                   colWidths=[.04,.24,.09,.09,.12,.35],loc='center',cellLoc='left')
    chart.auto_set_font_size(False);chart.set_fontsize(10);chart.scale(1,2)
    for (row,col),cell in chart.get_celld().items():
        cell.set_edgecolor('#ccd4dd')
        if row==0:cell.set_facecolor('#203451');cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#f0f4f8' if row%2 else 'white')
    fig.suptitle('Statistical tests across all five iterations | US high-cap, 2020–2025',fontsize=14)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(OUT/f'statistical_tests_all_iterations.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    final_number=decisions[-1]['last_accepted_after']
    final_path=BASE if final_number==1 else ROOT/f'outputs/iteration_{final_number}/us_large'
    final=pd.read_parquet(final_path/'pf.parquet')
    final_folder=OUT/'final';final_folder.mkdir(exist_ok=True)
    for file in ('pf.parquet','weights.parquet'):shutil.copy2(final_path/file,final_folder/file)
    final_cmp=paired_probability(final,market,protocol['bootstrap_reps'],protocol['bootstrap_block_months'],protocol['bootstrap_seed'])
    net=final.r.to_numpy()-final.tc.to_numpy();mn=market.r.to_numpy()-market.tc.to_numpy();active=net-mn
    te=float(active.std(ddof=1)*np.sqrt(12))
    final_stats=pd.DataFrame([dict(strategy=n,**statistics(p)) for n,p in [('Final Portfolio ML',final),('Market',market)]])
    final_stats.to_csv(final_folder/'performance_summary.csv',index=False)
    write_json(final_folder/'summary.json',dict(last_accepted_iteration=final_number,included_iterations=decisions[-1]['included_after'],
        p_beats_market=final_cmp['probability'],active_return=float(12*active.mean()),tracking_error=te,
        information_ratio=float(12*active.mean()/te) if te>0 else None,independent_unseen_confirmation=False))
    plot_comparison({'Final Portfolio ML':final,'Market':market},final_folder/'charts','Final accepted strategy vs Market, 2020--2025')
    (final_folder/'SUMMARY.md').write_text('# Final accepted US high-cap Portfolio ML\n\nIncluded iterations: '+str(decisions[-1]['included_after'])+'\n\n'+METHODS,encoding='utf-8')
    write_json(OUT/'run_status.json',dict(completed_iterations=len(decisions),total_iterations=5,
        sequence_complete=len(decisions)==5,final_is_provisional=len(decisions)<5))
    display=table[['iteration','change','Sharpe','P_better','worst_drawdown','verdict']].copy()
    display['P_better']=display.P_better.map(lambda x:'N/A' if pd.isna(x) else f'{x:.1%}')
    display['worst_drawdown']=display.worst_drawdown.map(lambda x:f'{x:.1%}')
    links=''.join(f'<li><a href="../iteration_{d["number"]}/us_large/index.html">Iteration {d["number"]}: {html.escape(d["name"])}</a> — {html.escape(d["verdict"])}</li>' for d in decisions)
    body='<h1>Statistical tests across all five US high-cap iterations</h1>'+display.to_html(index=False,float_format=lambda x:f'{x:.3f}')
    body+='<ul>'+links+'</ul><h2>Final accepted strategy vs Market</h2>'+final_stats.to_html(index=False,float_format=lambda x:f'{x:.5g}')
    body+='<img src="final/charts/net_returns_and_costs.png"><p><a href="statistical_tests_all_iterations.csv">Experiment table CSV</a> | <a href="final/summary.json">Final active metrics and P versus Market</a> | <a href="METHODS.md">Definitions and limitations</a></p>'
    body+='<p class="note">'+html.escape(METHODS)+'</p>'
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8">'+STYLE+body,encoding='utf-8')
    (OUT/'METHODS.md').write_text(METHODS,encoding='utf-8')
    # Replace the former four-version landing page with the current five-step report.
    alias=ROOT/'outputs/base/overall analysis/us_high_cap_progression';alias.mkdir(parents=True,exist_ok=True)
    (alias/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=../../../iteration_summary/index.html"><a href="../../../iteration_summary/index.html">Current five-experiment decisions and final strategy</a>',encoding='utf-8')
    progression=[dict(strategy='Base Portfolio ML',**statistics(base))]
    for d in decisions:
        progression.append(dict(strategy=f'Iteration {d["number"]}: {d["name"]}',**statistics(pd.read_parquet(ROOT/f'outputs/iteration_{d["number"]}/us_large/pf.parquet'))))
    pd.DataFrame(progression).to_csv(alias/'performance_summary.csv',index=False)
    print(display.to_string(index=False),flush=True)
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upto',type=int,choices=range(1,6),default=5)
    parser.add_argument('--reports-only',action='store_true')
    args=parser.parse_args(argv)
    if args.reports_only:return report_saved()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    protocol=json.loads(PROTOCOL.read_text());engine=ExperimentEngine()
    base=pd.read_parquet(BASE/'pf.parquet')
    static=pd.read_parquet(ROOT/'outputs/base/us_large/static_ml_star/pf.parquet')
    market=pd.read_parquet(ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet')
    accepted_state=default_state(engine.cfg);accepted=[1];previous=base;previous_number=1
    base_hashes={str(p):digest(p) for p in [BASE/'pf.parquet',ROOT/'outputs/base/us_large/static_ml_star/pf.parquet',ROOT/'outputs/base/us_mid/portfolio_ml/rff/pf.parquet',ROOT/'outputs/base/eu_large/portfolio_ml/rff/pf.parquet']}
    for spec in protocol['iterations'][:args.upto]:
        n=spec['number'];folder=ROOT/f'outputs/iteration_{n}/us_large';folder.mkdir(parents=True,exist_ok=True)
        before=list(accepted);reference=previous;reference_label='Base (annual refitting)' if previous_number==1 else f'Iteration {previous_number}'
        if n==1:
            state=dict(accepted_state)
            for name in ('pf.parquet','weights.parquet','chosen.csv'):shutil.copy2(BASE/name,folder/name)
            candidate=base;test=dict(probability=None,ci_low=0.,ci_high=0.,identical=True,valid_reps=0)
            keep=True;verdict='Annual refitting already included in Base; retained as design'
        else:
            state=candidate_state(accepted_state,spec['override'])
            candidate=engine.run_candidate(state,folder)
            test=paired_probability(candidate,reference,protocol['bootstrap_reps'],protocol['bootstrap_block_months'],protocol['bootstrap_seed'])
            keep=test['probability'] is not None and test['probability']>=protocol['acceptance_threshold'] and not test['identical']
            verdict='Kept' if keep else ('Not included: identical to previous accepted version' if test['identical'] else 'Not included: P(better) below 80%')
        verify_accounting(folder,engine.panel_for(state,folder),engine.wealth)
        if keep:
            accepted_state=state;accepted=inheritance(accepted,n,True);previous=candidate;previous_number=n
        decision=dict(number=n,name=spec['name'],reference=reference_label,inherited_iterations=before,
            p_better=test['probability'],bootstrap=test,threshold=protocol['acceptance_threshold'],
            accepted=bool(keep),enabled_in_next_accepted_setup=bool(keep),included_after=accepted,
            last_accepted_after=previous_number,verdict=verdict,candidate_state=state,accepted_state_after=accepted_state,
            evaluation_start=str(engine.cfg.dates.test_start),evaluation_end=str(engine.cfg.dates.test_end),
            p_is_conventional_p_value=False,independent_unseen_confirmation=False)
        write_json(folder/'decision.json',decision)
        write_iteration_report(n,decision,candidate,reference,base,static,market)
        report_saved(upto=n)
    if any(digest(Path(p))!=h for p,h in base_hashes.items()):raise AssertionError('Base portfolios changed during iteration work')
    write_json(OUT/'verification.json',dict(status='PASS',base_portfolios_unchanged=True,iterations=args.upto,
        acceptance_threshold=protocol['acceptance_threshold'],test_months=72,source_signature=engine.signature))
    return 0


if __name__=='__main__':raise SystemExit(main())
