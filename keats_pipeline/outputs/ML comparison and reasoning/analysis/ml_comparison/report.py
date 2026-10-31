"""Readable research summary and publication-resolution saved charts."""
from __future__ import annotations

import html
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def charts(destination,monthly,table,paths):
    plt.rcParams.update({'font.size':9,'figure.dpi':130,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
    target=destination/'charts';target.mkdir(exist_ok=True)
    written=[]
    def save(fig,name):
        for ext in ['png','pdf']:
            fig.savefig(target/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig);written.append(name)
    for dataset,t in table.groupby('dataset'):
        tag=dataset.replace('/','__')
        dates=f'{t.iloc[0].sample_start} to {t.iloc[0].sample_end}'
        g=monthly.loc[monthly.dataset==dataset]
        fig,ax=plt.subplots(figsize=(10,6))
        for row in t.itertuples():
            ax.scatter(row.Vol*100,row.R_TC*100,label=row.strategy,s=35,
                marker='x' if not row.geometric_excess_path_valid else 'o')
        ax.set(xlabel='Annualised gross volatility (%)',ylabel='Annualised net excess return (%)',
            title=f'{dataset}: saved portfolio operating points | {dates}')
        if t.R_TC.min() < -1:ax.set_yscale('symlog',linthresh=10)
        ax.axhline(0,color='grey',lw=.6);ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
        fig.text(.1,-.02,'Saved baseline weights; gamma=10 evaluation, $10B anchor. Points are not a continuous optimised frontier.\nCrosses: at least one net monthly return <= −100%; algebraic metrics only.',fontsize=8)
        save(fig,f'{tag}_saved_frontier_points')
        if t.Vol.max()>4*t.Vol.median():
            # A second explicitly cropped view keeps ordinary-risk portfolios
            # legible without deleting the extreme strategy from the full view.
            focus=t.loc[t.Vol<=4*t.Vol.median()]
            fig,ax=plt.subplots(figsize=(10,6))
            for row in focus.itertuples():
                ax.scatter(row.Vol*100,row.R_TC*100,label=row.strategy,s=35,
                    marker='x' if not row.geometric_excess_path_valid else 'o')
            ax.set(xlabel='Annualised gross volatility (%)',ylabel='Annualised net excess return (%)',
                title=f'{dataset}: focused ordinary-risk view | {dates}')
            if focus.R_TC.min()<-1:ax.set_yscale('symlog',linthresh=10)
            ax.axhline(0,color='grey',lw=.6);ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
            omitted=', '.join(t.loc[~t.index.isin(focus.index),'strategy'])
            fig.text(.1,-.02,f'View limit: volatility <=4 times cross-strategy median. Outside view: {omitted}.\nAll observations remain in the full-view plot and tables; gamma=10, $10B anchor.',fontsize=8)
            save(fig,f'{tag}_saved_frontier_points_focused')
        fig,ax=plt.subplots(figsize=(10,5))
        for strategy,p in paths.loc[paths.dataset==dataset].groupby('strategy'):
            ax.plot(p.eom_ret,p.wealth_index_excess,label=strategy)
        ax.set(xlabel='Return month',ylabel='Compounded excess index (initial=1)',title=f'{dataset} | {dates}')
        ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
        fig.text(.1,-.01,'Modelled costs; synthetic excess index, not total investor wealth. Paths stop if a month loses 100% or more.',fontsize=8)
        save(fig,f'{tag}_cumulative_net_excess')
        for field,label in [('rolling_net_sharpe_paper','24-month paper net Sharpe'),('rolling_utility','24-month annual utility'),('drawdown_excess','Excess-index drawdown')]:
            fig,ax=plt.subplots(figsize=(10,5))
            for strategy,p in paths.loc[paths.dataset==dataset].groupby('strategy'):
                ax.plot(p.eom_ret,p[field],label=strategy)
            ax.set(title=f'{dataset}: {label} | {dates}',xlabel='Return month',ylabel=label)
            if field!='drawdown_excess' and np.nanmax(np.abs(paths.loc[paths.dataset==dataset,field]))>10:
                ax.set_yscale('symlog',linthresh=1)
            ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
            fig.text(.1,-.01,'Gamma=10; monthly costs deducted. Sharpe uses gross-return volatility. No model refitting.',fontsize=8)
            save(fig,f'{tag}_{field}')
        for kind,value in [('gross',g.r),('net',g.r-g.tc)]:
            corr=g.assign(value=value).pivot(index='eom_ret',columns='strategy',values='value').corr()
            fig,ax=plt.subplots(figsize=(10,8))
            im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
            ax.set_xticks(range(len(corr)),corr.columns,rotation=70,ha='right',fontsize=7)
            ax.set_yticks(range(len(corr)),corr.index,fontsize=7)
            ax.set_title(f'{dataset}: {kind} monthly return correlations | {dates}')
            fig.colorbar(im,ax=ax,label='Pearson correlation')
            save(fig,f'{tag}_table4_{kind}')
    return written


def _optional_csv(destination,relative):
    path=destination/relative
    if not path.is_file():return pd.DataFrame()
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()


def _audit_csv(destination,name):
    path='audit/'+name if (destination/'audit'/name).is_file() else name
    return _optional_csv(destination,path)


def frontier_charts(destination):
    """Figure 1 operating points from actual saved optimisation, never rescaling."""
    grid=_optional_csv(destination,'robustness/saved_optimised_gamma_aum.csv')
    if grid.empty:return
    grid=grid.loc[grid.portfolio_policy.eq('previously_optimised_saved_grid_point')&
        np.isclose(grid.aum_anchor_usd,1e10)&grid.optimised_gamma.isin([1,5,10,20,100])]
    target=destination/'charts';target.mkdir(exist_ok=True)
    for dataset,g in grid.groupby('dataset'):
        from matplotlib.lines import Line2D
        fig,ax=plt.subplots(figsize=(10,6))
        markers={1:'o',5:'s',10:'^',20:'D',100:'X'}
        strategy_handles=[]
        for i,(strategy,p) in enumerate(g.groupby('strategy')):
            p=p.sort_values('optimised_gamma')
            color=f'C{i}'
            strategy_handles.append(Line2D([0],[0],marker='o',color='none',markerfacecolor=color,
                markeredgecolor=color,label=strategy,linestyle='none'))
            # A path between observed points is descriptive, not interpolation
            # of optimal portfolios. No connecting lines across missing gammas.
            for row in p.itertuples():
                ax.scatter(row.Vol*100,row.R_TC*100,color=color,s=45,
                    marker=markers[int(row.optimised_gamma)],alpha=.8,edgecolor='white',linewidth=.4)
        if g.R_TC.min()<-1:ax.set_yscale('symlog',linthresh=10)
        if g.Vol.max()>4*g.Vol.median():ax.set_xscale('symlog',linthresh=10)
        dates=f'{g.start.min()} to {g.end.max()}'
        ax.set(xlabel='Annualised gross volatility (%)',ylabel='Annualised net excess return (%)',
            title=f'{dataset}: actual saved optimised frontier points | {dates}')
        ax.axhline(0,color='grey',lw=.6)
        strategy_legend=ax.legend(handles=strategy_handles,bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8,title='Saved strategy')
        ax.add_artist(strategy_legend)
        gamma_handles=[Line2D([0],[0],marker=markers[level],color='grey',linestyle='none',
            label=f'gamma={level:g}') for level in sorted(g.optimised_gamma.unique())]
        ax.legend(handles=gamma_handles,bbox_to_anchor=(1.02,.62),loc='upper left',fontsize=8,title='Optimised risk aversion')
        fig.text(.1,-.02,'Fixed $10B anchor; independently saved optimisation at labelled gamma. No hypothetical weight scaling.\nOnly available points shown; incomplete grids remain incomplete. Scatter points do not assert a continuous frontier.',fontsize=8)
        for ext in ['png','pdf']:
            fig.savefig(target/f"{dataset.replace('/','__')}_actual_optimised_frontier_10B.{ext}",bbox_inches='tight',dpi=300)
        plt.close(fig)


def _numerical_answers(destination,table,pairs,annual):
    """Use generated data, rather than fixed conclusions, for all ten answers."""
    from .inference import newey_west_mean
    def pct(x):return f'{100*x:,.2f}%'
    def pp(x):return f'{100*x:,.2f} percentage points'
    def pair(dataset,a,b):
        if pairs is None or pairs.empty:return None
        p=pairs.loc[(pairs.dataset==dataset)&(((pairs.strategy_a==a)&(pairs.strategy_b==b))|
            ((pairs.strategy_a==b)&(pairs.strategy_b==a)))]
        if p.empty:return None
        z=p.iloc[0];sgn=1 if z.strategy_a==a else -1
        low,high=(z.utility_boot_ci_low,z.utility_boot_ci_high) if sgn==1 else (-z.utility_boot_ci_high,-z.utility_boot_ci_low)
        return f'{pp(sgn*z.utility_diff_annual)}/year (95% block CI {pp(low)} to {pp(high)}; Holm HAC p={z.utility_hac_p_holm:.4g})'
    ml=table.loc[table.strategy.str.contains('ML',regex=False)]
    winners=ml.loc[ml.groupby('dataset').Utility.idxmax()]
    lines=['## Answers to the research questions','',
        '### 1. Which strategy is best after costs and risk?','',
        '; '.join(f'{r.dataset}: {r.strategy}, utility {pct(r.Utility)}/year and paper net Sharpe {r.SR_net:.3f}' for r in winners.itertuples())+'.',
        'These are conditional rankings of saved portfolios. '+
        (f'All {len(winners)} dataset-specific winning ML utilities are negative at gamma=10. ' if winners.Utility.lt(0).all() else '')+
        'The zero-excess-risk cash alternative would have zero utility algebraically, but is not inserted as an implemented strategy. Compare passive benchmarks above, rather than equating best ML with best investment.',
        '', '### 2. Is the ranking consistent across datasets?','',
        f'There are {winners.strategy.nunique()} distinct ML winners across {len(winners)} dataset samples. '+
        'Baseline Portfolio-ML utility is '+', '.join(f'{r.dataset} {pct(r.Utility)}/year' for r in table.loc[table.strategy.eq('Portfolio-ML')].itertuples())+'. '
        'Common dates do not equal common inputs: membership, liquidity, fitted forecasts, country exposure and regional training populations differ. No pooled return or causal regional/model-effect estimate is formed (`tables/cross_dataset_common_calendar.csv`).',
        '', '### 3. Are the differences statistically supported?','']
    if pairs is not None and len(pairs):
        lines.append(f'Across {len(pairs)} unordered within-dataset pairs, {(pairs.utility_hac_p_holm<.05).sum()} utility and {(pairs.sharpe_hac_p_holm<.05).sum()} paper-net-Sharpe differences have Holm-adjusted HAC p<0.05. Families are separate by dataset and metric. Much of this separation involves extreme saved costs, so it is not evidence that all close ML competitors differ reliably.')
        for dataset,t in ml.groupby('dataset'):
            t=t.sort_values('Utility',ascending=False)
            effect=pair(dataset,t.iloc[0].strategy,t.iloc[1].strategy)
            if effect:lines.append(f'- {dataset}, best versus next-best ML ({t.iloc[0].strategy} versus {t.iloc[1].strategy}): {effect}.')
        lines.append('The paper posterior-style probabilities above are not these frequentist p-values. CIs condition on already selected portfolios, not a new untouched holdout. Block lengths 3/6/12 and joint-bootstrap ranking stability are saved; posterior-like certainty can coexist with weak HAC support in this short dependent sample.')
    else:lines.append('SKIPPED in this run: paired inference was not requested. Run `--only inference` or the full evaluation.')
    stability=_optional_csv(destination,'tables/bootstrap_ranking_stability.csv')
    if not stability.empty:
        selected=stability.loc[stability.ranking_metric.eq('utility')&stability.observed_rank.eq(1)]
        lines.append('Among all models and benchmarks, conditional joint-bootstrap frequencies of rank one are '+
            '; '.join(f'{r.dataset} {r.strategy}: {100*r.bootstrap_probability_best_ties_split:.1f}% (95% rank interval {int(r.rank_ci_low)}–{int(r.rank_ci_high)})' for r in selected.itertuples())+'. These are resampling stability frequencies, not Bayesian or future-win probabilities.')
    lines.extend(['', '### 4. Does Portfolio-ML beat return-prediction approaches?',''])
    for dataset,t in table.groupby('dataset'):
        base=t.loc[t.strategy.eq('Portfolio-ML')]
        if base.empty:continue
        a=base.iloc[0]
        competitors=t.loc[t.strategy.isin(['Markowitz-ML','Factor-ML','Static-ML','Static-ML*'])]
        wins=int((a.Utility>competitors.Utility).sum())
        effect=pair(dataset,'Portfolio-ML','Static-ML*')
        static=t.loc[t.strategy.eq('Static-ML*')]
        contrast=effect or (f'{pp(a.Utility-static.iloc[0].Utility)}/year' if len(static) else 'SKIPPED: no saved tuned Static-ML* series')
        lines.append(f'- {dataset}: baseline Portfolio-ML has higher utility than {wins}/{len(competitors)} saved standard return-prediction portfolio variants. Versus tuned Static-ML*: '+contrast+'.')
    lines.extend(['Portfolio-ML therefore has no universal dominance over tuned Static-ML*. Linear and active Portfolio-ML are separate saved variants; the active mandate is benchmark-relative rather than the same absolute utility objective.',
        '', '### 5. How much comes from trading costs or turnover?',''])
    for dataset,t in table.groupby('dataset'):
        a=t.loc[t.strategy.eq('Portfolio-ML')];b=t.loc[t.strategy.eq('Static-ML*')]
        if a.empty or b.empty:continue
        a,b=a.iloc[0],b.iloc[0]
        lines.append(f'- {dataset}, Portfolio-ML versus Static-ML*: annual net-return difference {pp(a.R_TC-b.R_TC)} = gross-return difference {pp(a.R-b.R)} + cost saving {pp(b.TC-a.TC)}; mean monthly turnover {pct(a.Turnover)} versus {pct(b.Turnover)}.')
    lines.append('This exact accounting decomposition is not causal attribution. The utility difference also includes the gross-variance penalty. Squared trade size, venue liquidity and wealth, not turnover alone, determine the modelled impact (`tables/return_cost_decomposition.csv`).')
    lines.extend(['', '### 6. How sensitive are AUM, gamma, liquidity and covariance assumptions?',''])
    cost=_optional_csv(destination,'robustness/fixed_weight_cost_sensitivity.csv')
    if not cost.empty:
        for dataset,g in cost.loc[cost.strategy.str.contains('ML',regex=False)].groupby('dataset'):
            choices=[]
            for multiplier in [0.,1.,3.]:
                p=g.loc[np.isclose(g.cost_factor,multiplier)]
                if len(p):
                    r=p.loc[p.Utility.idxmax()]
                    choices.append(f'{multiplier:g}x: {r.strategy} {pct(r.Utility)}/year')
            lines.append(f'- Cost re-pricing, {dataset}: '+ '; '.join(choices)+'. These portfolios were not re-optimised.')
    gamma=_optional_csv(destination,'robustness/fixed_weight_gamma_evaluation.csv')
    if not gamma.empty:
        for dataset,g in gamma.loc[gamma.strategy.str.contains('ML',regex=False)].groupby('dataset'):
            choices=[]
            for level in [1.,100.]:
                p=g.loc[np.isclose(g.evaluation_gamma,level)]
                if len(p):
                    r=p.loc[p.Utility.idxmax()];choices.append(f'gamma={level:g}: {r.strategy} ({pct(r.Utility)}/year)')
            lines.append(f'- Fixed saved weights, {dataset}: '+'; '.join(choices)+'. Changing evaluation gamma does not retune weights.')
    aum=_optional_csv(destination,'robustness/hypothetical_fixed_weight_aum.csv')
    grid=_optional_csv(destination,'robustness/saved_optimised_gamma_aum.csv')
    if not grid.empty:
        lines.append(f'- There are {len(grid)} complete saved optimised grid/capacity strategy points. The actual fixed-$10B frontier charts use saved gamma points only, without scaling or interpolation. AUM paths at a common gamma are separated from fixed-weight scenarios.')
    if not aum.empty:
        for dataset,g in aum.loc[aum.strategy.eq('Portfolio-ML')].groupby('dataset'):
            p=g.sort_values('aum_anchor_usd');lo,hi=p.iloc[0],p.iloc[-1]
            line=f'- Fixed-weight Portfolio-ML, {dataset}: utility {pct(lo.Utility)} at ${lo.aum_anchor_usd/1e9:g}B and {pct(hi.Utility)} at ${hi.aum_anchor_usd/1e9:g}B (gamma=10).'
            if not grid.empty:
                actual=grid.loc[grid.dataset.eq(dataset)&grid.strategy.eq('Portfolio-ML')&np.isclose(grid.optimised_gamma,10)&grid.portfolio_policy.eq('previously_optimised_saved_grid_point')].sort_values('aum_anchor_usd')
                if len(actual):
                    x,y=actual.iloc[0],actual.iloc[-1]
                    line+=f' Actual saved re-optimised points span {pct(x.Utility)} at ${x.aum_anchor_usd/1e9:g}B to {pct(y.Utility)} at ${y.aum_anchor_usd/1e9:g}B; return/weight changes make this a different experiment.'
            lines.append(line)
    liq=_optional_csv(destination,'robustness/size_liquidity_attribution_summary.csv')
    if not liq.empty:
        p=liq.loc[liq.dataset.eq('eu_large/base')&liq.strategy.eq('Markowitz-ML')&liq.group_variable.eq('liquidity_bucket')]
        low=p.loc[p.bucket.eq('low')]
        if len(low) and p.annual_tc_contribution.sum()>0:
            share=low.annual_tc_contribution.sum()/p.annual_tc_contribution.sum()
            lines.append(f'- European Markowitz-ML: the formation-month low-liquidity third contributes {100*share:.1f}% of annual modelled TC. This groups actual raw USD volume, not rank-normalised liquidity features.')
    cov=_audit_csv(destination,'covariance_validity_and_stability.csv')
    if not cov.empty:
        lines.append(f'- All {int(cov.positive_definite.sum())}/{len(cov)} examined covariance matrices are positive definite; maximum condition number is {cov.condition_number.max():,.1f}. This numerical check does not validate liquidity inputs, eliminate forecast error, or constitute an alternative covariance estimator.')
    risk=_audit_csv(destination,'forecast_versus_realised_risk_summary.csv')
    if not risk.empty:
        p=risk.loc[risk.strategy.eq('Portfolio-ML')]
        lines.append('- Baseline Portfolio-ML realised-to-RMS-forecast gross-volatility ratios: '+', '.join(f'{r.dataset} {r.realised_to_forecast_volatility:.2f}' for r in p.itertuples())+'. Descriptive forecast calibration uses changing saved positions; no confidence claim or refit is implied.')
    lines.extend(['', '### 7. Do characteristics explain performance differences?',''])
    ic=_optional_csv(destination,'robustness/one_month_prediction_diagnostics.csv')
    if not ic.empty:
        p=ic.loc[ic.population.eq('held_universe_members')&ic.target_policy.eq('observed_targets_only')]
        for dataset,g in p.groupby('dataset'):
            h=newey_west_mean(g.sort_values('eom_ret').rank_ic.to_numpy(float))
            scope=' Input-only diagnostic; no saved primary portfolio performance is available for this dataset.' if 'scope' in g and g.scope.astype(str).str.contains('input_only').any() else ''
            lines.append(f'- {dataset}: mean held-universe one-month rank IC {h["mean"]:.4f}; unadjusted exploratory HAC 95% CI [{h["mean"]-1.96*h["se"]:.4f}, {h["mean"]+1.96*h["se"]:.4f}], {len(g)} months. Imputed-zero targets are excluded from this diagnostic only; headline portfolios retain their saved assumptions.'+scope)
    for path in sorted((destination/'robustness/saved_feature_importance').glob('*/importance.csv')):
        fi=pd.read_csv(path)
        for strategy,g in fi.loc[fi.theme.ne('none')].groupby('type'):
            r=g.loc[g.fi.idxmax()]
            policy='gross utility; Markowitz construction gamma=100 and zero TC' if strategy=='Markowitz-ML' else 'net utility after saved modelled TC'
            lines.append(f'- Saved economic permutation importance, {path.parent.name}/base {strategy}: largest positive loss is {r.theme}, {pp(r.fi)} annual {policy}, seed standard error {pp(r.fi_se)} across {int(r.n_seeds)} permutations. Evaluation utility gamma=10. This seed SE measures permutation variability conditional on one model, not independent training uncertainty; the gross Markowitz diagnostic differs from its headline portfolio.')
    lines.append('Pure predictive permutation importance is unavailable. Theme–return rank associations are saved as descriptive diagnostics, not substituted for predictive importance. Observational exposures and permutations do not establish a causal characteristic explanation; 2–12 month decay is SKIPPED because multi-horizon predictions are not saved.')
    lines.extend(['', '### 8. Is performance stable through time and regimes?',''])
    regimes=_optional_csv(destination,'tables/regime_performance.csv')
    if not regimes.empty:
        halves=regimes.loc[regimes.strategy.eq('Portfolio-ML')&regimes.dimension.eq('calendar_half')]
        for dataset,g in halves.groupby('dataset'):
            g=g.sort_values('sample_start')
            lines.append(f'- {dataset}, baseline Portfolio-ML annual utility: '+ ' versus '.join(f'{r.sample_start[:4]}–{r.sample_end[:4]} {pct(r.Utility)}' for r in g.itertuples())+'. These are 36-month descriptive splits, not independently selected holdouts.')
    for winner in winners.itertuples():
        d=annual.loc[annual.dataset.eq(winner.dataset)&annual.strategy.str.contains('ML',regex=False)]
        yearly=d.loc[d.groupby('year').Utility.idxmax()]
        count=yearly.strategy.eq(winner.strategy).sum()
        lines.append(f'- {winner.dataset}: the full-period winning ML strategy, {winner.strategy}, is also the annual utility winner in {count}/{len(yearly)} individual years.')
    lines.append('The annual-winner changes and half-sample shifts do not support blanket time stability. VIX, market-sign, recession and calendar-stress rows and 24-month rolling charts are available. These ex-post partitions are exploratory; US recession/VIX data are context for Europe, not a European recession classification or proof of a trading signal.')
    lines.extend(['', '### 9. Which results support or contradict the paper?',''])
    # The local paper transcription is descriptive and uses a different sample.
    from .paths import REPOSITORY_ROOT
    paper_path=REPOSITORY_ROOT/'docs/paper_tables/table2.csv'
    if paper_path.is_file():
        paper=pd.read_csv(paper_path)
        p=paper.loc[paper.type.eq('Portfolio-ML')]
        if len(p):
            r=p.iloc[0]
            lines.append(f'The locally transcribed paper Table 2 reports Portfolio-ML utility {pct(r.Utility)}/year and net Sharpe {r.SR_net:.2f} for 1981–2020. Here baseline Portfolio-ML utility ranges from {pct(table.loc[table.strategy.eq("Portfolio-ML"),"Utility"].min())} to {pct(table.loc[table.strategy.eq("Portfolio-ML"),"Utility"].max())}/year in 2020–2025. This does not reproduce the paper’s positive magnitude; dates, training history, universe size, membership, liquidity and characteristics differ, so it is not a like-for-like rejection.')
    pf=table.loc[table.strategy.eq('Portfolio-ML')]
    mark=table.loc[table.strategy.eq('Markowitz-ML')]
    joined=pf.merge(mark,on='dataset',suffixes=('_pf','_mark'))
    if len(joined):
        lines.append(f'Portfolio-ML has lower annual TC and higher utility than saved Markowitz-ML in {int(((joined.TC_pf<joined.TC_mark)&(joined.Utility_pf>joined.Utility_mark)).sum())}/{len(joined)} datasets. That supports the economic relevance of cost-aware construction in these saved outputs, while the absence of universal superiority to Static-ML* and passive benchmarks limits the broader conclusion. Extreme unconstrained costs amplify the apparent advantage.')
    lines.extend(['', '### 10. What limits this replication?',''])
    nonexe=_audit_csv(destination,'non_executable_saved_months.csv')
    if not nonexe.empty:
        net_total=nonexe.net_total_below_minus_one.astype(str).str.lower().eq('true').sum()
        gross=nonexe.gross_excess_below_minus_one.astype(str).str.lower().eq('true').sum()
        lines.append(f'The audit retains {len(nonexe)} flagged saved strategy-month rows: {net_total} have net TOTAL return at or below −100% after adding the saved risk-free rate; {gross} have gross EXCESS return at or below −100% (overlap possible). These row counts are not interchangeable with the synthetic excess-index validity flag. The unconstrained model and quadratic impact paths are not executable after investor equity depletion.')
    contributors=_audit_csv(destination,'largest_stock_month_cost_contributors.csv')
    if not contributors.empty:
        r=contributors.loc[contributors.tc_contribution.idxmax()]
        company=r.get('conm',f'id {r.id}')
        lines.append(f'Largest recorded stock-month cost contributor: {r.dataset}, {r.strategy}, {company} (ID {r.id}), formation {r.eom}, modelled cost {r.tc_contribution:,.2f} times investor wealth, daily dollar volume ${r.dolvol:,.0f}, target/inherited weight {r.w:.3f}/{r.w_start:.3f}. Exact accounting reproduction does not externally validate this listing’s volume or economic feasibility.')
    lines.append('There is no separate unseen holdout, no independent training-seed paths, no 2–12 month saved forecast panel, no alternative saved covariance estimator and no observed execution-cost validation. Missing results remain SKIPPED. Zero-return imputations and unavailable-history exclusions are enumerated in the audit; no additional catastrophic observations are silently removed. Statistical uncertainty cannot repair these design and data limits.')
    return lines


def summary(destination,table,pairs,annual,inventory,missing,run_settings):
    frontier_charts(destination)
    out=['# Keats ML comparison and reasoning','',
        f"Run: {run_settings['created_sgt']} (Asia/Singapore). Saved outputs only; no model fitting, portfolio optimisation or downloaded inputs.",'',
        '## Scope and interpretation','',
        'All main comparisons use identical saved 2020–2025 return months within each dataset. Cross-dataset tables use the common calendar without pooling stock or return observations. The European universe is Europe-wide, not UK-only. Base portfolios target roughly 80 stocks; the saved US-large 250-stock experiment is also evaluated.',
        '', 'Initial return-model history begins in 2005. Portfolio validation forecasts cover 2015–2019; 2020–2025 is the declared walk-forward test. Models and portfolio hyperparameters were updated annually using past observations. There is no separate unseen holdout, and post-test research selection limits claims of pristine OOS validation.',
        '', 'Paper conventions: arithmetic annualisation (monthly mean ×12, gross SD ×√12), net Sharpe with gross SD, gamma=10 sample-variance mean–variance utility, two-sided monthly turnover. Costs are modelled quadratic price impact. Drawdown/CAGR/Sortino/Calmar are evaluation extensions with explicit excess-return definitions.',
        '', '## Findings','']
    def pct(x):return f'{x*100:,.2f}%'
    for dataset,t in table.groupby('dataset'):
        ml=t.loc[t.strategy.str.contains('ML',regex=False)]
        best=ml.sort_values('Utility',ascending=False).iloc[0]
        benchmark=t.sort_values('Utility',ascending=False).iloc[0]
        sr=ml.sort_values('SR_net',ascending=False).iloc[0]
        out.extend([f'### {dataset}','',
            f"Best saved ML utility: **{best.strategy}**, {pct(best.Utility)}/year, net excess return {pct(best.R_TC)}, gross volatility {pct(best.Vol)}, paper net Sharpe {best.SR_net:.3f}, annual TC {pct(best.TC)}, mean monthly turnover {pct(best.Turnover)}. The highest ML Sharpe is {sr.strategy} ({sr.SR_net:.3f}).",
            f"Across ML and benchmarks, highest utility is {benchmark.strategy} ({pct(benchmark.Utility)}/year). Sample: {int(best.n)} months, {best.sample_start} to {best.sample_end}.",''])
        if pairs is not None and len(pairs):
            comparisons=pairs.loc[(pairs.dataset==dataset)&((pairs.strategy_a==best.strategy)|(pairs.strategy_b==best.strategy))]
            # Compare the winning ML against each standard ML and Market; disclose all pair details in CSV.
            for other in ['Portfolio-ML','Static-ML*','Static-ML','Markowitz-ML','Factor-ML','Market']:
                if other==best.strategy:continue
                z=comparisons.loc[((comparisons.strategy_a==other)|(comparisons.strategy_b==other))]
                if z.empty:continue
                z=z.iloc[0];sgn=1 if z.strategy_a==best.strategy else -1
                diff=sgn*z.utility_diff_annual
                lo,hi=(z.utility_boot_ci_low,z.utility_boot_ci_high) if sgn==1 else (-z.utility_boot_ci_high,-z.utility_boot_ci_low)
                prob=z.posterior_probability_a_gt_b if sgn==1 else 1-z.posterior_probability_a_gt_b
                out.append(f"- Versus {other}: utility difference {pct(diff)}/year; paired block-bootstrap 95% CI [{pct(lo)}, {pct(hi)}]; HAC Holm-adjusted p={z.utility_hac_p_holm:.4g}; paper posterior-style probability={prob:.3f}.")
            out.append('')
    out.extend(_numerical_answers(destination,table,pairs,annual))
    out.extend(['', '## Material cautions','',
        'Some saved EXCESS returns net of modelled costs fall at or below -100% in a month. Their Table 2 arithmetic statistics are preserved and flagged; synthetic excess-index paths stop at that point, and full-sample geometric drawdown/CAGR/Calmar are unavailable. This excess-index criterion is distinct from funded equity depletion, which the audit checks using saved risk-free rates and net TOTAL returns. These cost/input issues need resolution at the model stage before interpreting an economic ranking as investable. This evaluation does not repair or refit models.',
        '', 'No cash portfolio has been added as an implemented strategy. At gamma=10, negative utility means that zero excess return with zero risk would have higher utility algebraically; it does not establish feasibility of another strategy under identical constraints.',
        '', '## Files and reproducibility','',
        '- `tables/table2_consolidated.csv`: paper statistics plus extensions and validity flags.',
        '- `tables/table3_pairwise_tests.csv` and dataset folders: paired inference and probability matrices. `tables/bootstrap_ranking_stability.csv` reports conditional resampling rank stability when inference is requested.',
        '- `tables/table4_correlations_long.csv` and dataset folders: gross/net correlations.',
        '- `tables/annual_performance.csv`, `tables/rolling_and_cumulative.csv`, `tables/regime_performance.csv`.',
        '- `robustness/`: re-costing, AUM, gamma, saved grid, liquidity, construction, prediction and feature diagnostics.',
        '- `audit/`: reconstruction checks, covariance/timing/data-quality findings, exclusions and source provenance.',
        '- `metric_dictionary.csv`, `EXCLUSIONS_ASSUMPTIONS_AND_CHANGES.md`, `validation/`, `run_settings.json`.',
        '- `charts/`: 300-dpi PNG and vector PDF. Axes use symmetric log scales when saved costs span many orders of magnitude.',
        '', 'Reference: Jensen, Kelly, Malamud and Pedersen (2026), [Machine Learning and the Implementable Efficient Frontier](https://doi.org/10.1093/rfs/hhag022). Exact implementation conventions were checked against the read-only authors’ R code in `reference/jkmp_r/0 - General functions.R`, `6 - Base analysis.R`, and local Keats accounting/metrics functions.'])
    text='\n'.join(out)+'\n'
    (destination/'RESEARCH_SUMMARY.md').write_text(text,encoding='utf-8')
    # A standalone readable index needs no web service and links all local artifacts.
    body=[]
    def inline(line):
        line=html.escape(line)
        line=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',line)
        line=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',line)
        return re.sub(r'`([^`]+)`',r'<code>\1</code>',line)
    for line in out:
        if line.startswith('### '):body.append('<h3>'+inline(line[4:])+'</h3>')
        elif line.startswith('## '):body.append('<h2>'+inline(line[3:])+'</h2>')
        elif line.startswith('# '):body.append('<h1>'+inline(line[2:])+'</h1>')
        elif line:body.append('<p>'+inline(line)+'</p>')
    files=sorted(p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file() and p.name!='index.html')
    links=''.join(f'<li><a href="{html.escape(f,quote=True)}">{html.escape(f)}</a></li>' for f in files)
    key_charts=sorted((destination/'charts').glob('*actual_optimised_frontier_10B.png'))
    gallery=''.join(f'<a href="charts/{html.escape(p.name)}"><img style="width:100%;max-width:1000px" src="charts/{html.escape(p.name)}" alt="{html.escape(p.stem)}"></a>' for p in key_charts)
    (destination/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Keats ML comparison</title><style>body{max-width:1100px;margin:40px auto;font:16px/1.6 system-ui;color:#142533}h1,h2{color:#12576a}li{margin:5px 0}code{background:#eef3f5;padding:2px 4px}</style>'+''.join(body)+'<h2>Actual saved optimised frontier points</h2>'+gallery+'<h2>All artifacts</h2><ul>'+links+'</ul>',encoding='utf-8')
