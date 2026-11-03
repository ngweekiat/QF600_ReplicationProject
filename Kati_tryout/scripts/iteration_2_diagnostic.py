"""Diagnose iteration 2 using only its returns and Market; no later model outcomes.

This retrospective diagnostic motivates a market-anchor experiment but cannot
establish its success. It reads exactly two portfolio files and writes separate
charts, statistics, sensitivity checks and source hashes under iteration 2.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/iteration_2/us_large/diagnostic_for_market_anchor'


def regression_hac(x,y,lags=6):
    X=np.column_stack([np.ones(len(x)),x]);inverse=np.linalg.inv(X.T@X)
    coef=np.linalg.lstsq(X,y,rcond=None)[0];residual=y-X@coef
    scores=X*residual[:,None];meat=scores.T@scores
    for lag in range(1,lags+1):
        cross=scores[lag:].T@scores[:-lag]
        meat+=(1-lag/(lags+1))*(cross+cross.T)
    covariance=inverse@meat@inverse*len(x)/(len(x)-2)
    distance=np.abs(np.arange(len(x))[:,None]-np.arange(len(x))[None,:])
    kernel=np.maximum(0.,1-distance/(lags+1))
    direct=inverse@(scores.T@kernel@scores)@inverse*len(x)/(len(x)-2)
    np.testing.assert_allclose(covariance,direct,rtol=1e-11,atol=1e-13)
    se=np.sqrt(np.diag(covariance))
    return coef,se,residual


def paired_bootstrap(x,y,block=6,reps=10000,seed=20261104):
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(x),size=(reps,int(np.ceil(len(x)/block))))
    index=((starts[:,:,None]+np.arange(block))%len(x)).reshape(reps,-1)[:,:len(x)]
    a=x[index];b=y[index];dx=a-a.mean(axis=1)[:,None];dy=b-b.mean(axis=1)[:,None]
    beta=(dx*dy).sum(axis=1)/(dx*dx).sum(axis=1)
    alpha=12*(b.mean(axis=1)-beta*a.mean(axis=1));gap=12*(a-b).mean(axis=1)
    # Independent literal OLS check on a subset of the SAME paired samples.
    for i in range(100):
        coefficients=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x[index[i]]]),y[index[i]],rcond=None)[0]
        np.testing.assert_allclose([beta[i],alpha[i]],[coefficients[1],12*coefficients[0]],rtol=1e-10,atol=1e-12)
    return dict(beta=np.quantile(beta,[.025,.975]),alpha=np.quantile(alpha,[.025,.975]),
                gap=np.quantile(gap,[.025,.975]))


def main():
    sources=[ROOT/'outputs/iteration_2/us_large/pf.parquet',
             ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet']
    candidate,market=[pd.read_parquet(p).sort_values('eom_ret').reset_index(drop=True) for p in sources]
    calendar=pd.date_range('2020-01-31','2025-12-31',freq='ME')
    assert pd.DatetimeIndex(candidate.eom_ret).equals(calendar) and candidate.eom_ret.equals(market.eom_ret)
    y=(candidate.r-candidate.tc).to_numpy();x=(market.r-market.tc).to_numpy()
    coef,se,residual=regression_hac(x,y)
    beta=float(coef[1]);alpha=float(12*coef[0]);market_return=float(12*x.mean());net=float(12*y.mean())
    gross=float(12*candidate.r.mean());cost=float(12*candidate.tc.mean());gap=market_return-net
    exposure_gap=(1-beta)*market_return;residual_gap=-alpha
    r2=float(1-residual@residual/((y-y.mean())@(y-y.mean())))
    direct_beta=float(((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum())
    np.testing.assert_allclose([beta,alpha,gap],[direct_beta,12*(y.mean()-direct_beta*x.mean()),exposure_gap+residual_gap],rtol=1e-12,atol=1e-12)
    sensitivity=[]
    for block in (3,6,12):
        intervals=paired_bootstrap(x,y,block)
        sensitivity.append(dict(block_months=block,**{f'{name}_{side}':float(value) for name,values in intervals.items() for side,value in zip(('low','high'),values)}))
    central=next(r for r in sensitivity if r['block_months']==6)
    p_beta_one=float(2*norm.sf(abs((beta-1)/se[1])))
    p_alpha=float(2*norm.sf(abs(coef[0]/se[0])))
    stats=dict(months=72,annual_net_iteration_2=net,annual_gross_iteration_2=gross,annual_net_market=market_return,
        annual_cost_iteration_2=cost,annual_net_gap_market_minus_iteration_2=gap,beta=beta,annual_alpha=alpha,r_squared=r2,
        correlation=float(np.corrcoef(x,y)[0,1]),exposure_gap_component=exposure_gap,negative_alpha_component=residual_gap,
        exposure_share_of_gap=exposure_gap/gap,cost_removal_upper_bound_at_fixed_weights=cost,
        gap_even_without_saved_costs=market_return-gross,HAC_lags=6,HAC_beta_se=float(se[1]),
        HAC_annual_alpha_se=float(12*se[0]),HAC_p_beta_equals_one=p_beta_one,HAC_p_alpha_equals_zero=p_alpha,
        bootstrap_reps=10000,bootstrap_seed=20261104,bootstrap_sensitivity=sensitivity,
        later_iteration_results_read=False,causal_attribution=False,independent_unseen_evidence=False)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    pd.DataFrame(sensitivity).to_csv(OUT/'block_bootstrap_sensitivity.csv',index=False)
    pd.DataFrame(dict(eom_ret=calendar,iteration_2_net=y,market_net=x,regression_fitted=coef[0]+beta*x,
                      regression_residual=residual)).to_csv(OUT/'monthly_diagnostic.csv',index=False)
    (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        source_count=2,later_iteration_results_read=False,paired_calendar_months=72,
        direct_OLS_and_decomposition_verified=True,HAC_covariance_independently_verified=True,
        literal_bootstrap_OLS_checks=300,beta_below_one_in_all_block_sensitivities=all(c['beta_high']<1 for c in sensitivity),
        net_shortfall_positive_in_all_block_sensitivities=all(c['gap_low']>0 for c in sensitivity)),indent=2),encoding='utf-8')
    draw(stats,central,x,y,coef)
    methods='''# Iteration 2: rationale for testing a market anchor

This diagnostic reads ONLY iteration 2 and the existing US high-cap Market
monthly portfolio files, January 2020-December 2025, 72 paired observations.
It does not read any later iteration's outcomes. It is a retrospective
diagnosis, not a pre-registered prediction that a proposed change will succeed.

Both regression series are monthly USD excess returns after each portfolio's
own saved modelled trading costs. OLS: iteration_2_net = alpha + beta*market_net
+ residual. Alpha annualises as 12 times the intercept. R-squared is the sample
variance fraction explained by this one-factor projection, not causation or a
complete risk attribution. Negative alpha can reflect omitted exposures, cost,
stock selection or estimation uncertainty; it is not isolated stock-picking skill.
Low market beta is not inherently a defect under an absolute-risk objective.
The proposed anchor is motivated by a benchmark-relative investment mandate,
not by proof that a fully invested market exposure is optimal for every investor.

HAC uses Bartlett/Newey-West lags 6, normal-reference two-sided inference, and
finite-sample n/(n-2) correction. Bootstrap CIs use 10,000 paired circular-block
resamples, 6-month blocks, seed 20261104; 3/12-month sensitivity is retained.
These are unadjusted descriptive intervals on already inspected test data,
not a correction for all previous research choices. Market-beta uncertainty
and alpha uncertainty are reported separately.

The gap identity is market mean - strategy mean = (1-beta)*market mean - alpha.
It is an accounting decomposition of the fitted regression, NOT a causal
estimate of the gain achievable by changing the strategy. Trading-cost removal
holds saved positions/gross returns fixed; it is not a frictionless refit.

Proposed economic experiment: retain market participation through a benchmark
anchor, then let ML choose bounded cost-aware stock tilts. This addresses an
exposure mismatch but cannot establish that learned tilts add value. A separate
evaluation is required. No later iteration's performance appears in the images.
'''
    (OUT/'METHODS.md').write_text(methods,encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k!='bootstrap_sensitivity'},indent=2))
    return 0


def draw(s,ci,x,y,coef):
    navy='#1c304d';teal='#00868b';orange='#dd842b'
    fig=plt.figure(figsize=(13.5,8.4),dpi=180,facecolor='white')
    fig.text(.035,.953,'3 | DIAGNOSIS',fontsize=11,weight='bold',color=teal)
    fig.text(.035,.902,'Iteration 2: why test a market-anchored portfolio?',fontsize=21,weight='bold',color=navy)
    fig.text(.035,.855,'US high-cap | 2020-2025 | 72 paired months | Only iteration 2 and Market data',fontsize=11,color='#535e6b')
    axes=[fig.add_axes([.075+j*.31,.48,.27,.29]) for j in range(3)]
    values=100*np.array([s['annual_net_iteration_2'],s['annual_gross_iteration_2'],s['annual_net_market']])
    axes[0].bar([0,1,2],values,color=[orange,'#e5ba82',navy],width=.6)
    axes[0].set_xticks([0,1,2],['Iteration 2\nnet','Same weights\nzero costs','Market\nnet'],fontsize=8)
    axes[0].set_title('Costs alone cannot close the gap',fontsize=10);axes[0].set_ylabel('Annual excess return (%)',fontsize=9)
    for i,value in enumerate(values):axes[0].text(i,value+.4,f'{value:.2f}%',ha='center',fontsize=9)
    axes[0].set_ylim(0,18)
    axes[1].scatter(100*x,100*y,s=15,alpha=.6,color=orange)
    line=np.linspace(x.min(),x.max(),100)
    axes[1].plot(100*line,100*(coef[0]+coef[1]*line),color=navy,lw=1.5)
    axes[1].set_title('Market participation is substantially lower',fontsize=10)
    axes[1].set_xlabel('Market monthly net excess return (%)',fontsize=8);axes[1].set_ylabel('Iteration 2 monthly net excess return (%)',fontsize=8)
    axes[1].text(.04,.95,f"Beta = {s['beta']:.3f}\nR-squared = {s['r_squared']:.1%}",transform=axes[1].transAxes,va='top',fontsize=9)
    components=100*np.array([s['exposure_gap_component'],s['negative_alpha_component']])
    axes[2].bar([0,1],components,color=['#267a9e','#b94d3e'],width=.6)
    axes[2].set_xticks([0,1],['Lower market\nexposure','Negative fitted\nalpha'],fontsize=8)
    axes[2].set_title('Regression decomposition of the return gap',fontsize=9)
    axes[2].set_ylabel('Annual return gap (percentage points)',fontsize=8)
    for i,value in enumerate(components):axes[2].text(i,value+.15,f'{value:.2f} pp',ha='center',fontsize=9)
    axes[2].set_ylim(0,max(components)*1.2)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.tick_params(labelsize=8)
    rows=[
        ['Market beta',f"{s['beta']:.3f}",'1.000',f"95% block CI: [{ci['beta_low']:.3f}, {ci['beta_high']:.3f}]"],
        ['Annual fitted alpha',f"{s['annual_alpha']:.2%}",'0%',f"95% block CI: [{ci['alpha_low']:.2%}, {ci['alpha_high']:.2%}]"],
        ['Annual net-return shortfall',f"{100*s['annual_net_gap_market_minus_iteration_2']:.2f} pp",'Market minus model',f"Gap CI: [{100*ci['gap_low']:.2f}, {100*ci['gap_high']:.2f}] pp"],
        ['Maximum saved-cost removal',f"+{100*s['annual_cost_iteration_2']:.2f} pp",'Fixed weights',f"Remaining annual gap: {100*s['gap_even_without_saved_costs']:.2f} pp"],
    ]
    ax=fig.add_axes([.035,.24,.93,.175]);ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Diagnostic','Iteration 2','Reference','Statistical / economic check'],
        colWidths=[.28,.16,.14,.42],cellLoc='left',colLoc='left',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(9.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.6);cell.set_edgecolor('#68717d');cell.PAD=.025
        cell.set_facecolor(navy if r==0 else ('#eef2f7' if r%2 else 'white'))
        cell.set_text_props(color='white' if r==0 else '#253140',weight='bold' if r==0 else 'normal')
    fig.text(.035,.192,'Hypothesis to test: preserve Market exposure; let ML choose small, cost-aware stock tilts.',fontsize=11,weight='bold',color=navy)
    fig.text(.035,.153,'Lower market exposure accounts for about '+f"{s['exposure_share_of_gap']:.0%}"+' of the observed gap in this regression identity; it is not a causal gain estimate.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.117,'Fitted alpha is negative but uncertain (its 95% interval includes zero). Anchoring alone does not establish stock-selection skill.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.078,'This motivates testing iteration 4 for a benchmark-relative mandate; improvement is unproven. No later results are displayed.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.042,'Retrospective analysis. Bootstrap intervals: 10,000 paired six-month blocks. Decomposition is descriptive; no causal or unseen-data claim.',fontsize=8.5,color='#68717d')
    fig.savefig(OUT/'iteration_2_gap_diagnostic.png',dpi=180,facecolor='white');plt.close(fig)
    pd.DataFrame(rows,columns=['Diagnostic','Iteration 2','Reference','Statistical / economic check']).to_csv(OUT/'presentation_table.csv',index=False)


if __name__=='__main__':raise SystemExit(main())
