"""Pretest-only Base diagnosis and current/historical branch presentation reports.

The pretest command reads past-only saved Base aims, replays original execution,
and never reads a 2020+ portfolio outcome. The results command runs after fits.
"""
from pathlib import Path
import argparse
import json
import shutil
from textwrap import fill
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from workflow import ROOT
from iteration_sequence import ExperimentEngine,default_state,digest,write_json,verify_accounting
from iteration_2_diagnostic import regression_hac,paired_bootstrap
from jkmp.portfolio.grids import grid_inputs
from jkmp.portfolio.portfolio_ml import pfml_weights
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.accounting import add_start_weights,portfolio_returns
from jkmp.portfolio.benchmarks import market_targets
from analysis.ml_comparison.metrics import statistics

NAVY='#1c304d';TEAL='#00868b';ORANGE='#dd842b'
OUT=ROOT/'outputs/iteration_4/us_large'
BASE=ROOT/'outputs/base/us_large/portfolio_ml/rff'
HISTORY=ROOT/'outputs/revision_history/before_iteration4_base_parent'

def savefig(fig,folder,name):
    folder.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.texts:
        box=text.get_window_extent(renderer);available=.965*fig.bbox.width-box.x0
        if box.width>available:text.set_fontsize(text.get_fontsize()*available/box.width)
    for suffix in ('png','pdf'):fig.savefig(folder/f'{name}.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)

def style(table):
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#72798a');cell.set_linewidth(.6);cell.PAD=.025
        cell.set_facecolor(NAVY if r==0 else ('#eef2f7' if r%2 else 'white'))
        cell.set_text_props(color='white' if r==0 else '#253140',weight='bold' if r==0 else 'normal')

def pretest():
    engine=ExperimentEngine();cfg=engine.cfg;state=default_state(cfg)
    panel=engine.panel_for(state,OUT/'pretest_base');cov=engine.covariance_for(panel)
    inp=grid_inputs(cfg,panel,engine.wealth,cov,engine.rf)
    dates=inp.pfml_dates[(inp.pfml_dates>=pd.Timestamp('2015-12-31')) &
                         (inp.pfml_dates+pd.offsets.MonthEnd(1)<=pd.Timestamp('2019-12-31'))]
    # The saved aims include other dates, but only preceding annual choices and
    # formation months returning in 2016-2019 can enter the diagnostic replay.
    aims=pd.read_parquet(BASE/'aims.parquet');aims=aims.loc[aims.eom.isin(dates)]
    chosen=pd.read_csv(BASE/'chosen.csv',parse_dates=['eom_ret'])
    relevant=chosen[chosen.year.isin((dates+pd.offsets.MonthEnd(1)).year)]
    assert relevant.eom_ret.lt(pd.to_datetime(relevant.year.astype(str)+'-01-01')).all()
    weights=pfml_weights(month_inputs(inp.panel,engine.wealth,cov,dates,mu_col=None),aims,inp.panel,
        engine.wealth,engine.rf,gamma=cfg.investor.gamma,mu=cfg.investor.mu,
        iterations=cfg.investor.m_iterations,charge_exit_tc=cfg.portfolio.charge_exit_tc)
    candidate=portfolio_returns(weights,inp.panel,engine.wealth).sort_values('eom_ret').reset_index(drop=True)
    targets=market_targets(inp.panel.loc[inp.panel.in_universe&inp.panel.eom.isin(dates)])
    mw=add_start_weights(targets,inp.panel,engine.wealth,dates,cfg.portfolio.charge_exit_tc)
    market=portfolio_returns(mw,inp.panel,engine.wealth).sort_values('eom_ret').reset_index(drop=True)
    assert candidate.eom_ret.equals(market.eom_ret) and candidate.eom_ret.max()<=pd.Timestamp('2019-12-31')
    folder=OUT/'pretest_base';folder.mkdir(exist_ok=True)
    candidate.to_parquet(folder/'pf.parquet',index=False);weights.to_parquet(folder/'weights.parquet',index=False)
    market.to_parquet(folder/'market_pf.parquet',index=False)
    stock=weights.merge(inp.panel[['id','eom','ret_ld1','lambda']],on=['id','eom'],validate='many_to_one').merge(engine.wealth[['eom','wealth']],on='eom',validate='many_to_one')
    assert len(stock)==len(weights) and stock['lambda'].notna().all() and stock.loc[~stock.exit,'ret_ld1'].notna().all()
    literal=pd.DataFrame(dict(eom=stock.eom,r=stock.w*stock.ret_ld1.where(~stock.exit,0),
        tc=.5*stock.wealth*stock['lambda']*(stock.w-stock.w_start)**2)).groupby('eom').sum()
    literal.index+=pd.offsets.MonthEnd(1)
    assert candidate.eom_ret.equals(pd.Series(pd.date_range('2016-01-31','2019-12-31',freq='ME'),name='eom_ret'))
    np.testing.assert_allclose(literal.loc[candidate.eom_ret,['r','tc']],candidate[['r','tc']],rtol=1e-10,atol=1e-12)
    x=(market.r-market.tc).to_numpy();y=(candidate.r-candidate.tc).to_numpy()
    coef,se,residual=regression_hac(x,y);beta=float(coef[1]);alpha=float(12*coef[0])
    annual_market=12*x.mean();annual_net=12*y.mean();gross=12*candidate.r.mean();cost=12*candidate.tc.mean()
    gap=annual_market-annual_net;exposure=(1-beta)*annual_market
    np.testing.assert_allclose(gap,exposure-alpha,atol=1e-12)
    intervals=paired_bootstrap(x,y,reps=10000,block=6)
    sensitivity=[dict(block=b,**{k:list(map(float,v)) for k,v in paired_bootstrap(x,y,block=b).items()}) for b in (3,6,12)]
    stats=dict(months=len(x),start=str(candidate.eom_ret.min().date()),end=str(candidate.eom_ret.max().date()),
        beta=beta,annual_alpha=alpha,annual_net_base=annual_net,annual_gross_base=gross,annual_market=annual_market,
        annual_cost=cost,net_shortfall=gap,exposure_component=exposure,negative_alpha_component=-alpha,
        r_squared=float(1-residual@residual/np.sum((y-y.mean())**2)),
        intervals={k:list(map(float,v)) for k,v in intervals.items()},block_sensitivity=sensitivity,
        iteration_2_used=False,iteration_4_or_later_outcomes_read=False,independent_unseen_evidence=False)
    output=OUT/'pretest_diagnosis';output.mkdir(exist_ok=True)
    write_json(output/'statistics.json',stats)
    pd.DataFrame(dict(eom_ret=candidate.eom_ret,base_net=y,market_net=x)).to_csv(output/'monthly_diagnostic.csv',index=False)
    fig=plt.figure(figsize=(15,9),facecolor='white')
    fig.text(.035,.955,'3 | PRETEST DIAGNOSIS',fontsize=12,weight='bold',color=TEAL)
    fig.text(.035,.905,'Before iteration 4: why test a market-anchored portfolio?',fontsize=21,weight='bold',color=NAVY)
    fig.text(.035,.86,f'US high-cap | 2016-2019 only | {len(x)} paired months | Base gamma 10 | No iteration 2 or 2020+ outcomes',fontsize=11,color='#535e6b')
    axes=[fig.add_axes([.075+j*.31,.51,.27,.27]) for j in range(3)]
    values=100*np.array([annual_net,gross,annual_market]);axes[0].bar(range(3),values,color=[ORANGE,'#e9b976',NAVY])
    axes[0].set_xticks(range(3),['Base net','Same weights\nzero costs','Market net']);axes[0].set(title='Can saved costs explain the gap?',ylabel='Annual excess return (%)')
    for i,v in enumerate(values):axes[0].annotate(f'{v:.2f}%',(i,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
    axes[0].margins(y=.2)
    axes[1].scatter(100*x,100*y,color=ORANGE,alpha=.7,s=13);xx=np.array([x.min(),x.max()]);axes[1].plot(100*xx,100*(coef[0]+beta*xx),color=NAVY)
    axes[1].set(title=f'Market participation: beta {beta:.3f}',xlabel='Market monthly net excess return (%)',ylabel='Base monthly net excess return (%)')
    axes[2].bar(range(2),100*np.array([exposure,-alpha]),color=['#267f9e','#b84c3e'])
    axes[2].set_xticks(range(2),['Market-exposure\ncomponent','Fitted-alpha\noffset']);axes[2].set(title='Descriptive return-gap decomposition',ylabel='Annual gap (percentage points)')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.tick_params(labelsize=8);ax.title.set_fontsize(10);ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
    rows=[['Market beta',f'{beta:.3f}',f'95% paired-block interval [{intervals["beta"][0]:.3f}, {intervals["beta"][1]:.3f}]'],
          ['Annual fitted alpha',f'{alpha:.2%}',f'95% interval [{intervals["alpha"][0]:.2%}, {intervals["alpha"][1]:.2%}]'],
          ['Annual Market minus Base',f'{100*gap:.2f} pp',f'95% interval [{100*intervals["gap"][0]:.2f}, {100*intervals["gap"][1]:.2f}] pp'],
          ['Saved-cost removal at fixed weights',f'{100*cost:.2f} pp',f'Remaining gap {100*(annual_market-gross):.2f} pp; no reoptimisation']]
    ax=fig.add_axes([.035,.29,.93,.16]);ax.axis('off');style(ax.table(cellText=rows,colLabels=['Diagnostic','Base pretest observation','Statistical / economic interpretation'],colWidths=[.32,.22,.46],cellLoc='left',colLoc='left',bbox=[0,0,1,1]))
    fig.text(.035,.23,'Hypothesis: retain the Market starting point and let ML learn small, cost-aware active stock tilts.',fontsize=11,weight='bold',color=NAVY)
    fig.text(.035,.18,'Beta is well below one, but the return-gap interval includes zero. Saved costs account for most of the observed shortfall.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.13,'The anchor tests a benchmark-relative mandate. Low beta is not inherently wrong; this projection is descriptive, not causal.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.08,'Intervals use 10,000 paired circular six-month blocks; 3/6/12-month sensitivity is saved. Forty-eight months give limited precision.',fontsize=9.3,color='#454f5b')
    fig.text(.035,.04,'Retrospective pretest-only diagnosis. No iteration-4 outcomes enter this image; no improvement is assumed.',fontsize=9,color='#68717d')
    savefig(fig,OUT/'charts','pretest_rationale')
    write_json(output/'verification.json',dict(status='PASS',pretest_only=True,last_return=stats['end'],
        original_base_execution_replayed=True,stock_accounting_verified=True,regression_decomposition_verified=True,
        literal_bootstrap_OLS_checks=400,source_hashes={str(p):digest(p) for p in (BASE/'aims.parquet',BASE/'chosen.csv')},
        future_aim_rows_excluded=True,iteration_2_used=False,later_outcomes_read=False))
    (output/'METHODS.md').write_text('Past-only saved Base aims are replayed through original execution at gamma 10, with a fresh common December-2015 VW start. Only January 2016-December 2019 returns enter diagnosis. Aims outside formation dates are excluded. Market uses the same dates, universe and accounting. OLS, Bartlett HAC and paired block intervals are descriptive; low beta is not inherently a flaw under an absolute-risk mandate. Zero-cost comparison holds weights fixed. Source files may contain other dates, but no later strategy return or aim enters the calculation.\n',encoding='utf-8')
    print(json.dumps(stats,indent=2))

DESCRIPTIONS={1:'Annual expanding coefficient refits, already included in Base.',2:'Gamma 10 to 5: a standalone historical trial; excluded from the revised anchor chain.',3:'Exit rank 120 and costs x2: historical trial after gamma 5.',4:'Start directly from Base; add Market plus pretest-calibrated 1% TE-target tilts. No gamma-5 parent.',5:'Recent-history weighting.',6:'Training-only selective-trading cost preference.',7:'Stock-specific risk responsiveness.',8:'Simpler selection using the original learning proxy.',9:'Benchmark-aware recursive trade-cost learning.',10:'Equal-weight five-seed RFF aim ensemble.','4.2':'On iteration 4: choose annual settings by executed historical active utility.','4.3':'On 4.2: prefer simpler candidates within one bootstrap SE of the best executed utility.'}

def results():
    from iteration_results_panels import build
    from active_validation_reports import build_reports
    for number in (4,'4.2','4.3'):build(number)
    build_reports();rows=[]
    for number in (1,2,3,4,'4.2','4.3',5,6,7,8,9,10):
        folder=ROOT/f'outputs/iteration_{number}/us_large';decision=json.loads((folder/'decision.json').read_text());s=statistics(pd.read_parquet(folder/'pf.parquet'))
        current=number in (1,4,'4.2','4.3')
        verdict=('Retained as design' if number==1 else 'Retained; no P gate' if isinstance(number,str) else decision['verdict'])
        historical=not current and (number in (2,3) or 2 in decision['inherited_iterations'])
        if not current:verdict=('Kept standalone; excluded' if number==2 else 'Not included; historical' if historical and not decision['accepted'] else decision['verdict']+'; outside chain')
        rows.append(dict(iteration=str(number),description=DESCRIPTIONS[number],net_sharpe=s['SR_net'],p_better=decision['p_better'],
            additive_dd_pp=100*s['max_drawdown_excess'],annual_net_return=s['R_TC'],verdict=verdict,
            current_chain=current,historical_trial=historical,reference=decision['reference'],ancestry=','.join(map(str,decision['included_after']))))
    data=pd.DataFrame(rows);folder=ROOT/'outputs/iteration_summary';folder.mkdir(exist_ok=True)
    data.to_csv(folder/'all_trials_current_and_historical.csv',index=False)
    data[data.current_chain].to_csv(folder/'kept_current_chain.csv',index=False)
    legacy=data.rename(columns={'description':'change','net_sharpe':'Sharpe','p_better':'P_better','ancestry':'final_included_iterations'})
    legacy=legacy.assign(worst_drawdown=data.additive_dd_pp/100,included=data.current_chain,path_convention='additive')
    legacy.to_csv(folder/'statistical_tests_all_iterations.csv',index=False)
    data.to_csv(folder/'statistical_tests_with_descriptions.csv',index=False)
    for subset,name,title in [(data,'all_trials_with_descriptions','All trials: current branch and historical experiments'),(data[data.current_chain],'kept_current_chain','Current retained chain: Base → 4 → 4.2 → 4.3')]:
        fig=plt.figure(figsize=(18,4+len(subset)*.58),facecolor='white')
        fig.text(.025,.95,title,fontsize=22,weight='bold',color=NAVY)
        fig.text(.025,.91,'Kati_tryout | US high-cap | 2020-2025 | Additive returns | Iteration 2 excluded from current chain',fontsize=11,color='#535e6b')
        values=[[r.iteration,fill(r.description,63),f'{r.net_sharpe:.6f}','N/A' if pd.isna(r.p_better) else f'{r.p_better:.2%}',f'{r.additive_dd_pp:.2f}',fill(r.verdict,27)] for r in subset.itertuples()]
        ax=fig.add_axes([.025,.23,.95,.62]);ax.axis('off')
        table=ax.table(cellText=values,colLabels=['Iteration','What changed','Net Sharpe','P(test)','Additive DD (pp)','Verdict / scope'],colWidths=[.065,.415,.095,.085,.10,.24],cellLoc='left',colLoc='left',bbox=[0,0,1,1]);style(table)
        for index,row in enumerate(subset.itertuples(),start=1):
            color=TEAL if row.current_chain and row.iteration!='1' else '#bd351e' if row.verdict.startswith('Not included') else NAVY
            table[index,5].set_text_props(color=color,weight='bold')
        fig.text(.025,.17,'Current retained chain: 1 (annual refitting), 4, 4.2, 4.3. Gamma-5 iteration 2 is not inherited.',fontsize=11,weight='bold',color=NAVY)
        fig.text(.025,.125,'P references: 4 vs Base; 4.2 vs 4; 4.3 vs 4.2. P is bootstrap support, not a conventional p-value.',fontsize=10,color='#454f5b')
        fig.text(.025,.08,'4 uses the 80% rule; 4.2 and 4.3 are retained for research without that gate. Historical trials retain their original reference/P.',fontsize=10,color='#454f5b')
        fig.text(.025,.04,'Historical trials 2, 3 and 5-10 were not rerun or rebased. The same inspected test period is not independent unseen confirmation.',fontsize=9,color='#68717d')
        savefig(fig,folder,name)
    links=''.join(f'<li>Iteration {s}: <a href="../iteration_{s}/us_large/charts/{p}.png">{label}</a></li>' for s in ('4','4.2','4.3') for p,label in [('pretest_rationale' if s=='4' else 'rationale','Pretest diagnosis'),('market_comparison_panel','Results vs Market')])
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;color:#1c304d}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}</style><h1>Revised Base-anchored iteration chain</h1><p>Current retained chain: 1, 4, 4.2, 4.3. Iteration 2 is excluded. Other trials are preserved as historical comparisons, not rebased results.</p><ul>'+links+'</ul><img src="all_trials_with_descriptions.png"><img src="kept_current_chain.png">'+data.to_html(index=False)+'<p><a href="all_trials_current_and_historical.csv">All trials CSV</a> | <a href="kept_current_chain.csv">Current retained chain CSV</a></p>',encoding='utf-8')
    # Keep earlier summary filenames coherent with the new landing page.
    for old in ('statistical_tests_with_descriptions','statistical_tests_all_iterations'):
        for ext in ('png','pdf'):shutil.copy2(folder/f'all_trials_with_descriptions.{ext}',folder/f'{old}.{ext}')
    shutil.copy2(folder/'all_trials_with_descriptions.png',folder/'statistical_tests_screenshot.png')
    final=folder/'final';final.mkdir(exist_ok=True)
    for name in ('pf.parquet','weights.parquet'):shutil.copy2(ROOT/'outputs/iteration_4.3/us_large'/name,final/name)
    from iteration_sequence import plot_comparison
    frames={'Final Portfolio ML':pd.read_parquet(final/'pf.parquet'),'Market':pd.read_parquet(ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet')}
    pd.DataFrame([dict(strategy=k,**statistics(v)) for k,v in frames.items()]).to_csv(final/'performance_summary.csv',index=False)
    plot_comparison(frames,final/'charts','Retained research chain 1, 4, 4.2, 4.3 vs Market')
    (final/'SUMMARY.md').write_text('Current retained research chain: 1, 4, 4.2, 4.3. Iteration 2 is excluded. Substages 4.2 and 4.3 have descriptive P(test), without an 80% gate; retention is not evidence of superior Sharpe. Historical trial results remain separate.\n',encoding='utf-8')
    write_json(final/'summary.json',dict(research_branch=True,included_iterations=[1,4,'4.2','4.3'],last_retained_stage='4.3',p_retention_gate_for_substages=False,independent_unseen_confirmation=False))
    write_json(folder/'revision_status.json',dict(status='PASS',current_chain=[1,4,'4.2','4.3'],historical_trials_not_rebased=[2,3,5,6,7,8,9,10]))
    write_json(folder/'run_status.json',dict(completed_iterations=12,total_iterations=12,sequence_complete=True,final_is_provisional=False,research_branch=True,rerun_stages=[4,'4.2','4.3'],historical_trials_not_rebased=[2,3,5,6,7,8,9,10]))
    print(data.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pretest',action='store_true');args=parser.parse_args()
    pretest() if args.pretest else results()
