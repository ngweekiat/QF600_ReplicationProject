"""Base-only economic/statistical rationale for testing gamma 5 as iteration 2.

No iteration-2 or later outcomes are read. This is a retrospective diagnosis,
not evidence that lowering risk aversion must improve performance.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/base/us_large/diagnostic_for_gamma5'


def bootstrap(gross,cost,market,block,reps=10000,seed=20261105):
    rng=np.random.default_rng(seed);n=len(gross)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    index=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    r=gross[index];c=cost[index];m=market[index]
    net=12*(r-c).mean(axis=1);vol=np.sqrt(12)*r.std(axis=1,ddof=1)
    sharpe=net/vol
    market_vol=np.sqrt(12)*m.std(axis=1,ddof=1)
    for i in range(100):
        # Literal independent loop over the same blocks, rather than the array formula.
        ix=[(int(start)+j)%n for start in starts[i] for j in range(block)][:n]
        a=np.array([gross[j] for j in ix]);b=np.array([cost[j] for j in ix])
        expected_net=12*sum(float(v) for v in a-b)/n
        expected_var=sum((float(v)-float(a.mean()))**2 for v in a)/(n-1)
        np.testing.assert_allclose([net[i],vol[i],sharpe[i]],
            [expected_net,np.sqrt(12*expected_var),expected_net/np.sqrt(12*expected_var)],rtol=1e-11,atol=1e-12)
    result=dict(block_months=block)
    for name,array in dict(annual_net_return=net,net_sharpe=sharpe,annual_gross_vol=vol,
                            gross_vol_ratio=vol/market_vol).items():
        low,high=np.quantile(array,[.025,.975]);result[name+'_low']=float(low);result[name+'_high']=float(high)
    return result


def main():
    sources=[ROOT/'outputs/base/us_large/portfolio_ml/rff/pf.parquet',
             ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    base,market=[pd.read_parquet(p).sort_values('eom_ret').reset_index(drop=True) for p in sources]
    dates=pd.date_range('2020-01-31','2025-12-31',freq='ME')
    assert pd.DatetimeIndex(base.eom_ret).equals(dates) and base.eom_ret.equals(market.eom_ret)
    r=base.r.to_numpy();c=base.tc.to_numpy();m=market.r.to_numpy()
    assert np.isfinite(np.r_[r,c,m]).all() and (c>=0).all()
    gross=float(12*r.mean());cost=float(12*c.mean());net=float(12*(r-c).mean())
    variance=float(12*r.var(ddof=1));vol=float(np.sqrt(variance));mvol=float(np.sqrt(12)*m.std(ddof=1))
    risk10=10/2*variance;risk5=5/2*variance
    stats=dict(months=len(base),annual_gross_return=gross,annual_cost=cost,annual_net_return=net,
        net_sharpe=net/vol,annual_gross_volatility=vol,market_annual_gross_volatility=mvol,
        base_to_market_volatility_ratio=vol/mvol,cost_fraction_of_positive_gross_return=cost/gross,
        same_weights_risk_charge_gamma10=risk10,same_weights_risk_charge_gamma5=risk5,
        base_realised_utility_gamma10=net-risk10,base_realised_utility_gamma5_same_weights=net-risk5,
        risk_charge_example_is_not_a_return_gain=True,iteration_2_or_later_outcomes_read=False,
        independent_unseen_evidence=False,bootstrap_reps=10000,bootstrap_seed=20261105)
    sensitivity=[bootstrap(r,c,m,b) for b in (3,6,12)]
    central=next(s for s in sensitivity if s['block_months']==6)
    # Independent signed monthly utility flow and annual accounting identities.
    flows=(r-c)-10/2*(r-r.mean())**2*len(r)/(len(r)-1)
    np.testing.assert_allclose([gross-cost,12*flows.mean(),risk5/risk10],
        [net,stats['base_realised_utility_gamma10'],.5],rtol=1e-12,atol=1e-12)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(sensitivity).to_csv(OUT/'block_bootstrap_sensitivity.csv',index=False)
    pd.DataFrame(dict(eom_ret=dates,base_gross=r,base_cost=c,base_net=r-c,market_gross=m,
        market_cost=market.tc)).to_csv(OUT/'monthly_diagnostic.csv',index=False)
    draw(stats,central)
    methods='''# Base diagnosis: why test iteration 2 (gamma 10 to 5)?

Inputs are ONLY Base US high-cap Portfolio ML and the existing Market monthly
portfolio files, 72 common return months, January 2020-December 2025. No iteration
2 or later portfolio, weights, coefficients, decision or performance is read.
The source files are hashed before and after reporting. No model is fitted.

Annual mean excess return is 12 times the monthly mean. Costs are actual saved
quadratic-impact costs, also annualised by 12. Gross-return volatility uses
sqrt(12) times the sample standard deviation (ddof=1). Paper-convention net
Sharpe is annual mean net excess return divided by gross-return volatility.
The cost/gross ratio is descriptive accounting for this positive-gross sample;
it is not a causal explanation or an assumed saving attainable by refitting.

Bootstrap intervals are marginal 95% percentile intervals from 10,000 paired
circular block resamples, six-month blocks, seed 20261105. All monthly series
use the same sampled indices. Three- and twelve-month sensitivities are saved.
The intervals describe an already inspected test sample and are not adjusted
for prior model or research selection. They do not establish a gamma treatment
effect. A market volatility comparison is a reference, not an optimal risk target.

For the existing fixed positions, the annual mean-variance risk charge is
gamma/2 * annual sample variance of gross returns. Changing gamma from 10 to 5
halves this charge. The saved Base realised utility at gamma 10 is reconciled
independently from monthly flows. The gamma-5 charge is ONLY a same-positions
objective illustration; it is not the return, utility gain or realised risk of
the refitted iteration-2 portfolio. Utilities under different preferences are
not evidence of investment performance improvement. Lower risk aversion is an
investor-preference change, not an econometrically estimated correction.

The economic hypothesis is to test whether the original model's risk-cost
trade-off is suitable at this scale. Lower gamma reduces the price assigned
to risk; all stocks, characteristics, dates, learning formula and actual cost
assumptions remain fixed. A refit is necessary: weights need not double,
costs may increase, and net Sharpe can worsen. The observed weak net edge and
low realised volatility justify diagnosis, not a claim of excessive risk aversion
or a forecast that gamma 5 will succeed. In particular, stronger exposure cannot
repair a weak signal by itself. An investor content with Base risk has no
statistical obligation to choose a lower gamma.

The image deliberately excludes iteration-2 outcomes and its acceptance P.
It motivates a controlled test, not retrospective proof of success. There is
no new independent holdout and no causal or unseen-data claim.
'''
    (OUT/'METHODS.md').write_text(methods,encoding='utf-8')
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    assert hashes==after
    verification=dict(status='PASS',source_files=hashes,source_count=2,models_fitted=False,
        source_portfolios_unchanged=True,iteration_2_or_later_outcomes_read=False,
        paired_calendar_months=len(base),literal_bootstrap_checks=300,
        gross_minus_cost_and_utility_identity_verified=True,
        gamma5_example_is_same_weights_not_a_model_result=True)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(statistics=stats,central_intervals=central,verification=verification),indent=2))
    return 0


def draw(s,ci):
    navy='#1c304d';teal='#00868b';orange='#dd842b'
    fig=plt.figure(figsize=(13.5,8.7),dpi=180,facecolor='white')
    fig.text(.035,.955,'3 | DIAGNOSIS',fontsize=11,weight='bold',color=teal)
    fig.text(.035,.907,'Base: why test lower risk aversion in iteration 2?',fontsize=21,weight='bold',color=navy)
    fig.text(.035,.861,'US high-cap | 2020-2025 | 72 paired months | Only Base and Market data; no iteration-2 results',fontsize=10.5,color='#535e6b')
    axes=[fig.add_axes([.075+j*.31,.49,.27,.28]) for j in range(3)]
    values=[100*s['annual_gross_return'],-100*s['annual_cost'],100*s['annual_net_return']]
    axes[0].bar([0,1,2],values,bottom=[0,100*s['annual_gross_return'],0],color=[navy,'#b94d3e',orange],width=.6)
    axes[0].plot([.3,.7],[100*s['annual_gross_return']]*2,color='#8992a0',lw=.8)
    axes[0].plot([1.3,1.7],[100*s['annual_net_return']]*2,color='#8992a0',lw=.8)
    axes[0].set_xticks([0,1,2],['Base gross','Trading costs','Base net'],fontsize=8)
    axes[0].set_title('Costs absorb most of the small gross edge',fontsize=9.5)
    axes[0].set_ylabel('Annual excess return / cost (pp)',fontsize=8.5)
    for i,(value,top) in enumerate(zip(values,[values[0],values[0],values[2]])):
        axes[0].text(i,top+.045,f'{value:+.2f}',ha='center',fontsize=9)
    axes[0].set_ylim(0,1.08)
    vol=100*np.array([s['annual_gross_volatility'],s['market_annual_gross_volatility']])
    axes[1].bar([0,1],vol,color=[orange,navy],width=.6)
    axes[1].set_xticks([0,1],['Base Portfolio ML','Market'],fontsize=8)
    axes[1].set_title('Base takes substantially less realised risk',fontsize=9.5)
    axes[1].set_ylabel('Annual gross-return volatility (%)',fontsize=8.5)
    for i,v in enumerate(vol):axes[1].text(i,v+.5,f'{v:.2f}%',ha='center',fontsize=9)
    axes[1].set_ylim(0,21)
    charges=100*np.array([s['same_weights_risk_charge_gamma10'],s['same_weights_risk_charge_gamma5']])
    axes[2].bar([0,1],charges,color=[navy,teal],width=.6)
    axes[2].set_xticks([0,1],['Gamma 10\nBase preference','Gamma 5\nIllustration only'],fontsize=8)
    axes[2].set_title('Lower gamma halves the price assigned to risk',fontsize=9)
    axes[2].set_ylabel('Annual risk charge: gamma x variance / 2 (pp)',fontsize=8)
    for i,v in enumerate(charges):axes[2].text(i,v+.06,f'{v:.2f} pp',ha='center',fontsize=9)
    axes[2].set_ylim(0,2.7)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.tick_params(labelsize=8)
    rows=[
        ['Annual net excess return',f"{s['annual_net_return']:.2%}",'Zero net edge',f"95% block CI: [{ci['annual_net_return_low']:.2%}, {ci['annual_net_return_high']:.2%}]"],
        ['Net Sharpe (gross-return SD)',f"{s['net_sharpe']:.3f}",'Zero',f"95% block CI: [{ci['net_sharpe_low']:.3f}, {ci['net_sharpe_high']:.3f}]"],
        ['Base / Market volatility',f"{s['base_to_market_volatility_ratio']:.1%}",'Market = 100%',f"95% block CI: [{ci['gross_vol_ratio_low']:.1%}, {ci['gross_vol_ratio_high']:.1%}]"],
        ['Costs / positive gross return',f"{s['cost_fraction_of_positive_gross_return']:.1%}",'Saved Base positions','Accounting ratio; not a causal or forecast gain'],
        ['Gamma 10 to 5 risk charge',f'{charges[0]:.2f} to {charges[1]:.2f} pp','Same positions only','Preference illustration; no return gain assumed'],
    ]
    ax=fig.add_axes([.035,.252,.93,.185]);ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Diagnostic','Base / illustration','Reference','Statistical / economic check'],
        colWidths=[.285,.165,.15,.40],cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.6);cell.set_edgecolor('#68717d');cell.PAD=.025
        cell.set_facecolor(navy if r==0 else ('#eef2f7' if r%2 else 'white'))
        cell.set_text_props(color='white' if r==0 else '#253140',weight='bold' if r==0 else 'normal')
    fig.text(.035,.203,'Hypothesis to test: refit at gamma 5; keep stocks, inputs, dates and the trading-cost formula fixed.',fontsize=10.7,weight='bold',color=navy)
    fig.text(.035,.163,'The net edge is weak and uncertain. Lower gamma tests the risk-cost trade-off; it cannot create predictive skill.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.126,'Lower volatility is not itself a flaw. Gamma is an investor preference; Market volatility is not an optimal risk target.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.089,'A refit is required: weights need not double, costs may rise and Sharpe may worsen. No iteration-2 outcome or P(better) is shown.',fontsize=9.1,color='#454f5b')
    fig.text(.035,.045,'Retrospective diagnosis, not proof of improvement. 10,000 paired six-month-block resamples; no causal or independent unseen-data claim.',fontsize=8.5,color='#68717d')
    for suffix in ('png','pdf'):fig.savefig(OUT/f'base_rationale_for_iteration_2.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    pd.DataFrame(rows,columns=['Diagnostic','Base / illustration','Reference','Statistical / economic check']).to_csv(OUT/'presentation_table.csv',index=False)


if __name__=='__main__':raise SystemExit(main())
