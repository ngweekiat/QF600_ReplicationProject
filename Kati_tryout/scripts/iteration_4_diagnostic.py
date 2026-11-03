"""Diagnose iteration 4 and propose a trade-netting experiment, without fitting it.

Only iteration 4, its Market reference, saved trade weights and existing cost
inputs are read. No subsequent experiment's results support this diagnosis.
"""
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iteration_2_diagnostic import regression_hac
from analysis.ml_comparison.inference import newey_west_mean
from jkmp.portfolio.portfolio_ml import month_moments

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/iteration_4/us_large/diagnostic_for_trade_netting'
BP=10000.


def cost_decomposition(portfolio_weights,market_weights,panel,wealth):
    """Exact identity for two saved implemented books, including exit orders."""
    p=portfolio_weights.assign(dp=portfolio_weights.w-portfolio_weights.w_start)
    m=market_weights.assign(dm=market_weights.w-market_weights.w_start)
    trades=p[['id','eom','dp']].merge(m[['id','eom','dm']],on=['id','eom'],how='outer',validate='one_to_one')
    trades=trades.fillna({'dp':0.,'dm':0.}).merge(panel,on=['id','eom'],how='left',validate='many_to_one')
    trades=trades.merge(wealth[['eom','wealth']],on='eom',validate='many_to_one')
    if not np.isfinite(trades[['dp','dm','lambda','wealth']]).all().all():raise ValueError('Missing cost-accounting inputs')
    trades['da']=trades.dp-trades.dm
    scale=trades.wealth*trades['lambda']
    pieces=pd.DataFrame(dict(eom=trades.eom,market_cost=.5*scale*trades.dm**2,
        isolated_active_cost=.5*scale*trades.da**2,interaction=scale*trades.dm*trades.da,
        actual_cost=.5*scale*trades.dp**2)).groupby('eom').sum()
    np.testing.assert_allclose(pieces.actual_cost,pieces.market_cost+pieces.isolated_active_cost+pieces.interaction,rtol=1e-11,atol=1e-14)
    pieces.index+=pd.offsets.MonthEnd(1)
    return pieces


def bootstrap(monthly,block=6,reps=10000,seed=20261105):
    n=len(monthly);rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    names=['gross_active','extra_cost','net_active','interaction','isolated_active_cost','actual_cost']
    mean=12*monthly[names].to_numpy()[idx].mean(axis=1)
    x=monthly.market_net.to_numpy()[idx];y=monthly.portfolio_net.to_numpy()[idx]
    dx=x-x.mean(axis=1)[:,None];dy=y-y.mean(axis=1)[:,None]
    beta=(dx*dy).sum(axis=1)/(dx*dx).sum(axis=1)
    alpha=12*(y.mean(axis=1)-beta*x.mean(axis=1))
    for i in range(100):
        literal=12*monthly.iloc[idx[i]][names].to_numpy().mean(axis=0)
        np.testing.assert_allclose(mean[i],literal,rtol=1e-12,atol=1e-14)
    result={name:np.quantile(mean[:,i],[.025,.975]).tolist() for i,name in enumerate(names)}
    result.update(beta=np.quantile(beta,[.025,.975]).tolist(),alpha=np.quantile(alpha,[.025,.975]).tolist())
    return result


def waterfall(ax,values,labels,colors,title):
    running=0.
    for i,value in enumerate(values):
        if i==len(values)-1:bottom=0.;height=value
        else:bottom=running;height=value;running+=value
        ax.bar(i,height,bottom=bottom,color=colors[i],width=.65)
        endpoint=bottom+height
        ax.text(i,endpoint+(.18 if height>=0 else -.25),f'{value:+.2f}',ha='center',va='bottom' if height>=0 else 'top',fontsize=9)
    ax.set_xticks(range(len(labels)),labels,fontsize=8)
    ax.axhline(0,color='#777777',lw=.7);ax.set_title(title,fontsize=10)
    ax.set_ylabel('Basis points per year',fontsize=9);ax.grid(axis='y',alpha=.17)


def draw(stats,ci):
    navy='#1c304d';teal='#00868b';blue='#267a9e';red='#b94d3e';green='#359568'
    fig=plt.figure(figsize=(13.5,8.5),dpi=180,facecolor='white')
    fig.text(.035,.955,'3 | DIAGNOSIS',fontsize=11,weight='bold',color=teal)
    fig.text(.035,.903,'After iteration 4: can trade-aware learning add value?',fontsize=20,weight='bold',color=navy)
    fig.text(.035,.856,'US high-cap | 2020-2025 | 72 paired months | Iteration 4 and Market only | 1 bp = 0.01 percentage point',fontsize=10,color='#535e6b')
    axes=[fig.add_axes(position) for position in ([.07,.49,.25,.28],[.382,.49,.25,.28],[.72,.49,.245,.28])]
    waterfall(axes[0],[stats['gross_active']*BP,-stats['extra_cost']*BP,stats['net_active']*BP],
              ['Gross edge\nvs Market','Additional\ncost','Net edge\nvs Market'],[blue,red,navy],
              'Small gross advantage, larger extra costs')
    axes[0].set_ylim(-3.3,2.1)
    waterfall(axes[1],[stats['market_cost']*BP,stats['isolated_active_cost']*BP,stats['interaction']*BP,stats['actual_cost']*BP],
              ['Market\ncost','Isolated\nactive cost','Trade\ninteraction','Actual\ntotal cost'],[navy,blue,green,teal],
              'Netting already matters for actual costs')
    axes[1].set_ylim(-.3,10.5)
    labels=['Gross edge','Extra cost','Net edge','Interaction']
    keys=['gross_active','extra_cost','net_active','interaction']
    for i,key in enumerate(keys):
        lo,hi=np.array(ci[key])*BP;point=stats[key]*BP
        axes[2].plot([lo,hi],[i,i],lw=2,color=[blue,red,navy,green][i]);axes[2].plot(point,i,'o',color=[blue,red,navy,green][i])
    axes[2].set_yticks(range(4),labels,fontsize=8);axes[2].invert_yaxis();axes[2].axvline(0,color='#777777',lw=.8)
    axes[2].set_title('95% paired six-month-block intervals',fontsize=9)
    axes[2].set_xlabel('Annual mean (basis points)',fontsize=9);axes[2].grid(axis='x',alpha=.17)
    rows=[
        ['Market beta',f"{stats['beta']:.4f}",f"[{ci['beta'][0]:.4f}, {ci['beta'][1]:.4f}]",'Market exposure is already close to 1'],
        ['Annual net edge vs Market',f"{stats['net_active']*BP:+.2f} bp",f"[{ci['net_active'][0]*BP:+.2f}, {ci['net_active'][1]*BP:+.2f}] bp",'No reliable positive active return'],
        ['Annual additional cost',f"{stats['extra_cost']*BP:+.2f} bp",f"[{ci['extra_cost'][0]*BP:+.2f}, {ci['extra_cost'][1]*BP:+.2f}] bp",'Costs exceed the observed gross edge'],
        ['Trade-interaction effect',f"{stats['interaction']*BP:+.2f} bp",f"[{ci['interaction'][0]*BP:+.2f}, {ci['interaction'][1]*BP:+.2f}] bp",'Offsets already save cost; they are not free alpha'],
    ]
    ax=fig.add_axes([.035,.25,.93,.17]);ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Diagnostic','Annual / fitted value','95% block-bootstrap CI','Economic interpretation'],
        colWidths=[.25,.17,.25,.33],cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#68717d');cell.set_linewidth(.6);cell.PAD=.025
        cell.set_facecolor(navy if r==0 else ('#eef2f7' if r%2 else 'white'))
        cell.set_text_props(color='white' if r==0 else '#253140',weight='bold' if r==0 else 'normal')
    fig.text(.035,.202,'Proposed iteration 9: benchmark-aware trade netting during learning.',fontsize=12,weight='bold',color=navy)
    fig.text(.035,.161,'Learn using benchmark-rebalancing flows and their interaction with ML orders; charge the same costs on actual net trades.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.124,'Reported costs are already correct. The hypothesis targets an approximation in the benchmark-relative learning objective.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.087,'This refines our active extension; improvement is unproven and exact paper replication is not claimed. No iteration 9 outcomes are used here.',fontsize=9.3,color='#454f5b')
    fig.text(.035,.047,'Retrospective diagnosis; source books and costs are fixed. Confidence intervals do not correct earlier research selection. 3/12-month sensitivity retained.',fontsize=8.5,color='#68717d')
    fig.savefig(OUT/'iteration_4_next_iteration_diagnostic.png',dpi=180,facecolor='white');plt.close(fig)
    pd.DataFrame(rows,columns=['Diagnostic','Value','95% CI','Interpretation']).to_csv(OUT/'presentation_table.csv',index=False)


def main():
    data=ROOT/'data/processed/us_large/with_zero_returns';market=ROOT/'outputs/base/overall analysis/market_references/us_large'
    strategy=ROOT/'outputs/iteration_4/us_large'
    sources=[strategy/'pf.parquet',market/'pf.parquet',strategy/'weights.parquet',market/'weights.parquet',data/'panel.parquet',data/'wealth.parquet']
    p,m=[pd.read_parquet(path).sort_values('eom_ret').reset_index(drop=True) for path in sources[:2]]
    dates=pd.date_range('2020-01-31','2025-12-31',freq='ME')
    assert pd.DatetimeIndex(p.eom_ret).equals(dates) and p.eom_ret.equals(m.eom_ret)
    weights=[pd.read_parquet(path) for path in sources[2:4]]
    panel=pd.read_parquet(sources[4],columns=['id','eom','lambda']);wealth=pd.read_parquet(sources[5])
    pieces=cost_decomposition(*weights,panel,wealth).loc[dates]
    np.testing.assert_allclose(pieces.actual_cost,p.tc,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(pieces.market_cost,m.tc,rtol=1e-10,atol=1e-12)
    monthly=pieces.copy()
    monthly['market_net']=(m.r-m.tc).to_numpy();monthly['portfolio_net']=(p.r-p.tc).to_numpy()
    monthly['gross_active']=(p.r-m.r).to_numpy();monthly['extra_cost']=(p.tc-m.tc).to_numpy()
    monthly['net_active']=monthly.portfolio_net-monthly.market_net
    np.testing.assert_allclose(monthly.net_active,monthly.gross_active-monthly.extra_cost,rtol=1e-12,atol=1e-14)
    coef,se,residual=regression_hac(monthly.market_net.to_numpy(),monthly.portfolio_net.to_numpy())
    x=monthly.market_net.to_numpy();y=monthly.portfolio_net.to_numpy()
    direct_beta=((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum()
    np.testing.assert_allclose(coef,[y.mean()-direct_beta*x.mean(),direct_beta],rtol=1e-11,atol=1e-13)
    annual={k:float(12*monthly[k].mean()) for k in ['gross_active','extra_cost','net_active','market_cost','isolated_active_cost','interaction','actual_cost']}
    stats=dict(**annual,beta=float(coef[1]),annual_fitted_alpha=float(12*coef[0]),
        realised_TE=float(monthly.net_active.std(ddof=1)*np.sqrt(12)),
        net_IR=float(12*monthly.net_active.mean()/(monthly.net_active.std(ddof=1)*np.sqrt(12))),
        annual_portfolio_net=float(12*monthly.portfolio_net.mean()),annual_market_net=float(12*monthly.market_net.mean()),
        break_even_total_cost_at_unchanged_gross=annual['market_cost']+annual['gross_active'],
        negative_interaction_months=int(monthly.interaction.lt(0).sum()),months=72,
        cost_identity='TC_P = TC_M + TC_(P-M trades) + W * sum(lambda * dM * dA)',
        HAC_net_active=newey_west_mean(monthly.net_active.to_numpy(),lags=6))
    sensitivity=[];central=None
    for block in (3,6,12):
        ci=bootstrap(monthly,block=block)
        if block==6:central=ci
        for name,values in ci.items():sensitivity.append(dict(block_months=block,metric=name,low=values[0],high=values[1]))
    stats['bootstrap_sensitivity']=sensitivity
    OUT.mkdir(parents=True,exist_ok=True)
    monthly.rename_axis('eom_ret').to_csv(OUT/'monthly_diagnostic.csv')
    pd.DataFrame(sensitivity).to_csv(OUT/'block_bootstrap_sensitivity.csv',index=False)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    draw(stats,central)
    code=inspect.getsource(month_moments)
    verification=dict(status='PASS',source_files={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        source_count=6,later_iteration_results_read=False,calendar_months=72,monthly_stock_costs_reconstructed=True,
        monthly_trade_cost_identity_verified=True,paired_return_cost_identity_verified=True,
        literal_bootstrap_mean_checks=300,direct_regression_coefficient_check=True,
        extra_cost_positive_all_block_sensitivities=all(v['low']>0 for v in sensitivity if v['metric']=='extra_cost'),
        netting_effect_negative_all_block_sensitivities=all(v['high']<0 for v in sensitivity if v['metric']=='interaction'),
        learning_moment_signature=str(inspect.signature(month_moments)),
        learning_source_sha256=hashlib.sha256(code.encode()).hexdigest(),new_iteration_fitted=False)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    (OUT/'METHODS.md').write_text(METHODS,encoding='utf-8')
    print(json.dumps({k:v for k,v in stats.items() if k!='bootstrap_sensitivity'},indent=2))
    return 0


METHODS='''# Iteration 4 diagnostic: benchmark-aware trade netting

Only iteration 4 and its Market reference are analysed: 72 paired return months,
January 2020-December 2025. No later experiment outcome is read, fitted or used
to estimate a gain. These are retrospective, already inspected test results.

Gross active return is r_P-r_M; additional cost is TC_P-TC_M; net active return
is their difference. Annual means multiply by 12; values multiplied by 10,000
are basis points per year. Monthly regressions use net USD excess returns and
the same HAC check as the iteration-2 diagnostic. Confidence intervals use
10,000 paired circular-block bootstrap samples, seed 20261105; the image uses
6-month blocks and CSVs retain 3/12-month sensitivity. Intervals are unadjusted
for prior research selection, not independent unseen confirmation.

For each stock-month including exits, define dP=wP-wP_start, dM=wM-wM_start and
dA=dP-dM. With original W and lambda, the exact cost identity is:

TC_P = 0.5 W sum(lambda*dM^2) + 0.5 W sum(lambda*dA^2)
       + W sum(lambda*dM*dA).

Every month is independently reconciled to both saved cost books. The interaction
may be negative because orders offset; total actual costs remain nonnegative.
This is an accounting decomposition of SAVED trades, not proof that one component
can be removed without changing orders, returns or future costs. The gap-closing
cost target holds observed gross return fixed and is not an achievable-profit claim.

Code inspection: the active learning moment uses Omega returns and a quadratic
Delta-Omega cost matrix without an explicit benchmark-dependent linear cost term.
The active execution recursion does use the benchmark and charges actual net
trades correctly. Its documented learning approximation ignores benchmark
rebalancing. No reported-cost error is alleged.

Proposal: derive the learning objective for total recursive benchmark-plus-active
orders, retaining its benchmark-dependent cost interaction. The implementable
baseline must be generated by the SAME recursion and initialisation, not blindly
substituting the fully rebalanced Market reference's trades. The descriptive
decomposition above uses that reference for accounting clarity; it is not the
derivation of the proposed training matrix.

This refines our benchmark-relative extension; it preserves the quadratic impact
functional form but changes its active learning approximation. It is not a claim
to match the paper's default configuration exactly, nor a new scalar cost penalty.
The cost interaction already observed is not available a second time as free alpha.

No iteration 9 is implemented or run by this script. Any proposed trial should
start from accepted iteration 4, keep stocks, 50 inputs, dates, actual costs and
RFF/ridge grids fixed, calibrate 1% ex-ante TE before test, and apply the existing
80% paired-bootstrap P(better net Sharpe) rule without inheriting rejected changes.
'''


if __name__=='__main__':raise SystemExit(main())
