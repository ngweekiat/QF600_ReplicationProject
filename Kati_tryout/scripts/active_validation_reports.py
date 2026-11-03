"""Read-only economic comparison of iteration 4 and its 4.2/4.3 validation branch.

Investment models and rationale images are never fitted or changed here.
Reports distinguish absolute gamma-10 utility from active gamma-174.1 utility.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflow import ROOT,config_path
from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.metrics import active_weights,ex_ante_exposures
from analysis.ml_comparison.metrics import statistics

GAMMA=174.1
OUT=ROOT/'outputs/iteration_4_substages_summary'
CALENDAR=pd.date_range('2020-01-31','2025-12-31',freq='ME')
MARKET=ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet'
BLOCK_START='<!-- active-validation-economics -->'
BLOCK_END='<!-- /active-validation-economics -->'


def _digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _portfolio(path):
    frame=pd.read_parquet(path).sort_values('eom_ret').reset_index(drop=True)
    if not pd.DatetimeIndex(frame.eom_ret).equals(CALENDAR):raise ValueError(f'Incomplete/misaligned test calendar: {path}')
    if not np.isfinite(frame[['r','tc']]).all().all():raise ValueError('Nonfinite portfolio accounting')
    return frame


def _active(frame,market,forecast_te):
    g=(frame.r-market.r).to_numpy();n=((frame.r-frame.tc)-(market.r-market.tc)).to_numpy()
    extra=(frame.tc-market.tc).to_numpy()
    annual_gross=float(12*g.mean());annual_net=float(12*n.mean());annual_cost=float(12*extra.mean())
    gross_te=float(np.sqrt(12)*g.std(ddof=1));net_te=float(np.sqrt(12)*n.std(ddof=1))
    active_utility=float(annual_net-.5*GAMMA*gross_te**2)
    np.testing.assert_allclose(annual_gross-annual_cost,annual_net,rtol=1e-11,atol=1e-12)
    # Literal monthly sample-variance calculation independently checks annual utility.
    sample_var=sum((float(v)-float(np.mean(g)))**2 for v in g)/(len(g)-1)
    np.testing.assert_allclose(active_utility,12*np.mean(n)-.5*GAMMA*12*sample_var,rtol=1e-11,atol=1e-12)
    return dict(annual_active_gross=annual_gross,annual_active_net=annual_net,annual_extra_cost=annual_cost,
        realized_gross_TE=gross_te,realized_net_TE=net_te,IRnet=annual_net/gross_te if gross_te>0 else np.nan,
        active_utility=active_utility,active_utility_gamma=GAMMA,mean_ex_ante_TE=forecast_te,
        nominal_original_TE_target=.01,risk_target_recalibrated=False,months=len(frame),
        information_ratio_denominator='sample SD of gross active returns, annualised',
        benchmark='same-universe Market, own modelled costs included',independent_unseen_confirmation=False)


def _forecast_te(stages):
    """Recompute 4 directly; use/reconcile each new stage's own exposure file."""
    cfg=load_config(config_path('us_large'));p=cfg.paths.processed
    weight_path=ROOT/'outputs/iteration_4/us_large/weights.parquet'
    weights=pd.read_parquet(weight_path)
    panel=pd.read_parquet(p/'panel.parquet',columns=['id','eom','me'],filters=[
        ('id','in',weights.id.unique().tolist()),('eom','>=',weights.eom.min()),('eom','<=',weights.eom.max())])
    covariance=load_covariance(p/'covariance')
    exposures=ex_ante_exposures(active_weights(weights,panel),covariance).sort_values('eom')
    if not pd.DatetimeIndex(exposures.eom+pd.offsets.MonthEnd(1)).equals(CALENDAR):raise ValueError('Stage-4 forecast-risk calendar mismatch')
    means={'4':float(np.sqrt(12)*exposures.te.mean()),'Market':0.}
    crosscheck=ROOT/'outputs/iteration_4/us_large/diagnostic_for_neutrality/test/monthly_risk_and_replay.csv'
    neutral_crosscheck=False
    if crosscheck.exists():
        direct=pd.read_csv(crosscheck,parse_dates=['eom']).sort_values('eom')
        if not exposures.eom.reset_index(drop=True).equals(direct.eom.reset_index(drop=True)):raise ValueError('Stored risk cross-check calendar mismatch')
        np.testing.assert_allclose(exposures.te,np.sqrt(direct.active_variance),rtol=1e-9,atol=1e-12)
        neutral_crosscheck=True
    for stage in ('4.2','4.3'):
        folder=stages[stage]
        ex=pd.read_csv(folder/'ex_ante_exposures.csv',parse_dates=['eom']).sort_values('eom')
        if not pd.DatetimeIndex(ex.eom+pd.offsets.MonthEnd(1)).equals(CALENDAR):raise ValueError('Substage risk calendar mismatch')
        if not np.isfinite(ex.te).all() or ex.te.lt(0).any():raise ValueError('Invalid saved tracking risk')
        means[stage]=float(np.sqrt(12)*ex.te.mean())
        risk=json.loads((folder/'risk_summary.json').read_text())
        np.testing.assert_allclose(means[stage],risk['mean_test_ex_ante_te'],rtol=1e-11,atol=1e-12)
        if risk['frozen_gamma_te']!=GAMMA or risk['risk_target_recalibrated']:raise ValueError('Unexpected active-risk policy')
    return means,exposures,neutral_crosscheck


def build_reports():
    stages={stage:ROOT/f'outputs/iteration_{stage}/us_large' for stage in ('4','4.2','4.3')}
    cfg=load_config(config_path('us_large'))
    source_paths=[MARKET,*[folder/name for folder in stages.values() for name in ('pf.parquet','decision.json','performance_summary.csv')],
        stages['4']/'weights.parquet',cfg.paths.processed/'panel.parquet',
        cfg.paths.processed/'covariance/cov_loadings.parquet',cfg.paths.processed/'covariance/cov_factor.parquet',
        *[stages[s]/name for s in ('4.2','4.3') for name in ('ex_ante_exposures.csv','risk_summary.json')]]
    before={str(p):_digest(p) for p in source_paths}
    frames={stage:_portfolio(folder/'pf.parquet') for stage,folder in stages.items()};frames['Market']=_portfolio(MARKET)
    forecast,ex4,neutral_crosscheck=_forecast_te(stages)
    OUT.mkdir(parents=True,exist_ok=True)
    ex4.to_csv(OUT/'iteration_4_recomputed_ex_ante_exposures.csv',index=False)
    comparisons=[]
    for stage in ('4','4.2','4.3'):
        frame=frames[stage];standard=statistics(frame);decision=json.loads((stages[stage]/'decision.json').read_text())
        root_table=pd.read_csv(stages[stage]/'performance_summary.csv').set_index('strategy')
        current=root_table.loc['Candidate']
        for key in ('R_TC','SR_net','TC','Utility'):
            np.testing.assert_allclose(standard[key],current[key],rtol=1e-10,atol=1e-12)
        comparisons.append(dict(stage=stage,strategy=f'Iteration {stage}',**standard,
            P_test_vs_reference=decision['p_better'],P_test_reference=decision['reference'],
            P_test_is_retention_gate=stage=='4',mean_ex_ante_TE=forecast[stage],
            nominal_original_TE_target=.01,risk_target_recalibrated=False))
    comparisons.append(dict(stage='Market',strategy='Market',**statistics(frames['Market']),
        P_test_vs_reference=np.nan,P_test_reference='Reference',P_test_is_retention_gate=False,mean_ex_ante_TE=0.,
        nominal_original_TE_target=np.nan,risk_target_recalibrated=False))
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'performance_comparison.csv',index=False)
    active_all=[]
    for stage in ('4.2','4.3'):
        reference='4' if stage=='4.2' else '4.2'
        records=[]
        for role,key in [('Candidate',stage),('Previous stage',reference),('Iteration 4','4'),('Market','Market')]:
            records.append(dict(role=role,strategy='Market' if key=='Market' else f'Iteration {key}',
                **_active(frames[key],frames['Market'],forecast[key])))
        table=pd.DataFrame(records);table.to_csv(stages[stage]/'active_performance.csv',index=False)
        active_all.append(table.assign(report_stage=stage))
        candidate=table.iloc[0]
        paragraph=(f'Benchmark-relative economics for the same 72 test months: annual net active return '
            f'{candidate.annual_active_net:+.4%}, annual extra costs {candidate.annual_extra_cost:+.4%}, '
            f'realised gross tracking error {candidate.realized_gross_TE:.4%}, and net information ratio '
            f'{candidate.IRnet:.4f}. Active utility is {candidate.active_utility:+.4%} per year at gamma_TE 174.1. '
            f'Mean forecast tracking error is {candidate.mean_ex_ante_TE:.4%}; the nominal original 1% target '
            'was not forced or recalibrated. This active utility differs from the absolute gamma-10 Utility '
            'in performance_summary.csv. Differences in active risk and costs do not establish stock-selection alpha. '
            'P(test) is descriptive for this research branch, without an 80% gate. '
            'Detailed figures are in active_performance.csv and ../../iteration_4_substages_summary/charts/result_comparison.png.')
        path=stages[stage]/'SUMMARY.md'
        text=path.read_text(encoding='utf-8')
        if BLOCK_START in text:
            start=text.index(BLOCK_START);end=text.index(BLOCK_END,start)+len(BLOCK_END)
            text=text[:start]+text[end:]
        path.write_text(text.rstrip()+'\n\n'+BLOCK_START+'\n\n'+paragraph+'\n\n'+BLOCK_END+'\n',encoding='utf-8')
    pd.concat(active_all,ignore_index=True).to_csv(OUT/'active_performance.csv',index=False)
    pd.concat([f.assign(strategy='Market' if key=='Market' else f'Iteration {key}') for key,f in frames.items()],ignore_index=True).to_csv(OUT/'monthly_returns.csv',index=False)
    draw(frames,comparison)
    (OUT/'METHODS.md').write_text(METHODS,encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1400px;margin:35px auto;font:16px/1.5 system-ui;color:#1c304d}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #ccd4dd}</style><h1>Iteration 4 and validation substages</h1><p>Current chain: Base → 4 → 4.2 → 4.3; iteration 2 excluded. For 4.2 and 4.3 P(test) is descriptive, without an 80% gate. Other trials retain historical references.</p><img src="charts/result_comparison.png"><p><a href="performance_comparison.csv">Absolute performance</a> | <a href="active_performance.csv">Benchmark-relative performance</a> | <a href="METHODS.md">Metric definitions and limitations</a></p>'+comparison[['strategy','R_TC','SR_net','TC','mean_ex_ante_TE','P_test_vs_reference','P_test_reference']].to_html(index=False,float_format=lambda v:f'{v:.6g}'),encoding='utf-8')
    after={str(p):_digest(p) for p in source_paths}
    if before!=after:raise ValueError('Investment result/input source changed while reporting')
    verification=dict(status='PASS',source_files_unchanged=True,sources=before,calendar_months=72,
        benchmark_relative_accounting_identities_checked=8,original_stage4_ex_ante_TE_recomputed=True,
        optional_existing_geometry_risk_crosscheck_passed=neutral_crosscheck,substage_forecast_risk_summaries_reconciled=True,
        root_standard_performance_tables_reconciled=True,models_fitted=False,rationale_images_modified=False,
        active_utility_gamma=GAMMA,standard_absolute_utility_gamma=10.)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(comparison[['strategy','R_TC','SR_net','TC','mean_ex_ante_TE','P_test_vs_reference']].to_string(index=False))
    return 0


def draw(frames,table):
    fig=plt.figure(figsize=(15,8.6),dpi=180,facecolor='white');navy='#1c304d'
    fig.text(.035,.955,'3 | RESULTS',fontsize=12,weight='bold',color='#00868b')
    fig.text(.035,.9,'Iteration 4 to 4.2 and 4.3: validation refinements after the market anchor',fontsize=21,weight='bold',color=navy)
    fig.text(.035,.855,'US high-cap | 2020-2025 | 72 paired months | Frozen gamma_TE 174.1 | P(test) is descriptive for 4.2 / 4.3',fontsize=11,color='#535e6b')
    axes=[fig.add_axes([.075,.47,.40,.31]),fig.add_axes([.56,.47,.40,.31])]
    colors={'4':'#507fa2','4.2':'#dd842b','4.3':'#00868b','Market':'#1c304d'}
    benchmark=frames['Market'];bn=(benchmark.r-benchmark.tc).to_numpy()
    for key,frame in frames.items():
        label='Market' if key=='Market' else f'Iteration {key}'
        if key!='Market':axes[0].plot(frame.eom_ret,100*np.cumsum((frame.r-frame.tc).to_numpy()-bn),label=label,color=colors[key],lw=1.8)
        axes[1].plot(frame.eom_ret,100*frame.tc.cumsum(),label=label,color=colors[key],lw=1.6,ls='--' if key=='Market' else '-')
    axes[0].axhline(0,lw=.8,color='#777');axes[0].set(title='Cumulative net return minus Market (additive)',ylabel='Percentage points')
    axes[1].set(title='Cumulative actual modelled trading costs',ylabel='Percentage points')
    for ax in axes:ax.grid(alpha=.18);ax.legend(fontsize=8);ax.tick_params(labelsize=9);ax.title.set_fontsize(11)
    rows=[]
    for row in table.itertuples():
        p='Reference' if pd.isna(row.P_test_vs_reference) else f'{row.P_test_vs_reference:.2%}'
        rows.append([row.strategy,f'{row.R_TC:.4%}',f'{row.SR_net:.6f}',f'{row.TC:.4%}',p,f'{row.mean_ex_ante_TE:.4%}'])
    ax=fig.add_axes([.035,.245,.93,.155]);ax.axis('off')
    t=ax.table(cellText=rows,colLabels=['Strategy','Net excess return / year','Paper-net Sharpe','Trading costs / year','P(test), stated reference','Mean ex-ante TE'],
        colWidths=[.18,.19,.15,.17,.18,.13],cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    t.auto_set_font_size(False);t.set_fontsize(9)
    for (r,c),cell in t.get_celld().items():
        cell.set_linewidth(.6);cell.set_edgecolor('#68717d');cell.PAD=.025
        cell.set_facecolor(navy if r==0 else ('#eef2f7' if r%2 else 'white'))
        cell.set_text_props(color='white' if r==0 else '#253140',weight='bold' if r==0 else 'normal')
    fig.text(.035,.197,'4.3 trades less and takes less active risk; the headline Sharpe differences remain small.',fontsize=11,weight='bold',color=navy)
    fig.text(.035,.156,'P references: iteration 4 versus Base; 4.2 versus 4; 4.3 versus 4.2. Iteration 2 is excluded. No 80% gate applies to 4.2 / 4.3.',fontsize=10,color='#454f5b')
    fig.text(.035,.116,'The nominal original 1% TE target is not forced: risk aversion remains frozen, so realised and forecast active risks can change.',fontsize=10,color='#454f5b')
    fig.text(.035,.078,'Standard Utility uses absolute returns and gamma 10. active_performance.csv separately reports active utility at gamma_TE 174.1.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.039,'P(test) is block-bootstrap support, not a conventional p-value or a future probability. These inspected returns are not independent unseen confirmation.',fontsize=9.2,color='#68717d')
    output=OUT/'charts';output.mkdir(exist_ok=True)
    for suffix in ('png','pdf'):fig.savefig(output/f'result_comparison.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)


METHODS='''# Benchmark-relative economic comparison after iteration 4

The original saved 2020-2025 Market portfolio and each strategy are aligned on
all 72 return months. Gross active return is r_strategy-r_Market. Net active
return is (r_strategy-tc_strategy)-(r_Market-tc_Market). Extra cost is
tc_strategy-tc_Market; annual gross active minus annual extra cost must equal
annual net active. Annual means multiply by 12. All volatility/TE figures use
sample standard deviations (ddof=1) times sqrt(12).

IRnet uses annual net active return divided by realised GROSS active-return
volatility. Realised NET TE is supplied separately. Active utility is annual
net active return minus gamma_TE/2 times realised gross TE squared, at frozen
gamma_TE 174.1. This differs from the standard performance_summary.csv Utility:
that table measures absolute net excess return minus gamma/2 times absolute
gross-return variance at common reporting gamma 10. They are distinct economic
objectives and must not be interchanged or compared as the same statistic.

Mean ex-ante TE for iteration 4 is recomputed from saved stock weights, current
market-cap benchmark weights and the existing covariance model, using original
active_weights/ex_ante_exposures functions. If a previous geometry diagnostic
exists, its active variances are only an optional independent cross-check,
not a required reporting input. The 4.2/4.3 figures read each stage's own saved
exposure file and reconcile its risk_summary.json. Mean forecast TE is the
mean monthly forecast volatility times sqrt(12), not sqrt(mean variance).

Gamma_TE 174.1 remains frozen. The original 1% pretest calibration target is
nominal context; these substages do not force or recalibrate risk back to 1%.
Different levels of active risk, cost and return prevent attributing an observed
headline result solely to improved stock selection or forecasting skill.

P(test) is read from verified stage decisions: iteration 4 compares with
Base; 4.2 with iteration 4; 4.3 with 4.2. The original integer experiment
gate is not changed. For the two new substages retention has no 80% gate and
does not constitute statistical acceptance. P(test) describes paired block
bootstrap support for higher paper-convention net Sharpe; it is not a standard
frequentist p-value, posterior probability, or probability of future success.

Paths add monthly net excess returns (percentage points), and the cost panel
adds actual saved modelled quadratic costs. No investment fits are launched.
Rationale images remain separate and are not modified with these test results.
The same inspected 2020-2025 history is reused, so results are not independent
unseen confirmation. Standard performance tables, accounting identities,
forecast-risk summaries and input/result source hashes are reconciled.
'''


if __name__=='__main__':raise SystemExit(build_reports())
