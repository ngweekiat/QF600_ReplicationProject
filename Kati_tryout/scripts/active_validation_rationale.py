"""Pretest-only rationale images for the separate iteration-4 validation branch.

Reads candidate validation flows/proxy scores and pretest selection audits only.
No fitted 4.2/4.3 test portfolio or 2020+ outcome may enter these images.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
PRETEST_END=pd.Timestamp('2019-12-31')
PRETEST_START=pd.Timestamp('2015-01-31')
GAMMA=174.1
NAVY='#1c304d';TEAL='#00868b';ORANGE='#dd842b'


def _digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _window(dates,label):
    dates=pd.DatetimeIndex(pd.to_datetime(dates))
    if dates.isna().any() or dates.min()<PRETEST_START or dates.max()>PRETEST_END:
        raise ValueError(f'{label} contains a non-pretest date')
    return dates


def _utility(x,z):
    return 12*(np.mean(z,axis=0)-.5*GAMMA*np.var(x,axis=0,ddof=1))


def _bootstrap_scores(x,z,reps,block,seed):
    """All candidates use identical time blocks; recompute mean AND variance."""
    n=len(x);rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    counts=np.bincount((np.arange(reps)[:,None]*n+ix).ravel(),minlength=reps*n).reshape(reps,n).astype(float)
    mean_x=counts@x/n;mean_z=counts@z/n
    var=(counts@(x*x)-n*mean_x**2)/(n-1)
    if np.min(var)<-1e-10:raise ValueError('Bootstrap variance is materially negative')
    score=12*(mean_z-.5*GAMMA*np.maximum(var,0.))
    for i in range(min(50,reps)):
        # Literal index-loop check is independent of the frequency-matrix method.
        literal=[(int(a)+j)%n for a in starts[i] for j in range(block)][:n]
        np.testing.assert_allclose(score[i],_utility(x[literal],z[literal]),rtol=1e-8,atol=1e-11)
    return score


def _format_model(row):
    return f"p={int(row.p)}, lambda={float(row.l):.3g}, g={float(row.g):.3g}"


def _style(table):
    table.auto_set_font_size(False);table.set_fontsize(8.8)
    for (r,c),cell in table.get_celld().items():
        cell.set_linewidth(.6);cell.set_edgecolor('#68717d');cell.PAD=.025
        cell.set_facecolor(NAVY if r==0 else ('#eef2f7' if r%2 else 'white'))
        cell.set_text_props(color='white' if r==0 else '#253140',weight='bold' if r==0 else 'normal')


def _save_figure(fig,folder):
    charts=folder/'charts';charts.mkdir(parents=True,exist_ok=True)
    # Keep every presentation paragraph inside the same right-hand margin.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for label in fig.texts:
        box=label.get_window_extent(renderer)
        available=.965*fig.bbox.width-box.x0
        if box.width>available:
            label.set_fontsize(label.get_fontsize()*available/box.width)
    for suffix in ('png','pdf'):fig.savefig(charts/f'rationale.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)


def build_rationales(cache,audit42path,audit43path):
    """Build both rationale images without reading any strategy test outcomes."""
    cache=Path(cache);audit42path=Path(audit42path);audit43path=Path(audit43path)
    sources=[cache/'candidate_flows.parquet',cache/'proxy_scores.csv',cache/'catalog.csv',audit42path,audit43path]
    before={str(p):_digest(p) for p in sources}
    flows=pd.read_parquet(sources[0]);proxy=pd.read_csv(sources[1],parse_dates=['eom_ret'])
    catalog=pd.read_csv(sources[2]).sort_values('cid').reset_index(drop=True)
    a42=pd.read_csv(audit42path,parse_dates=['selection_month']).sort_values('selection_month')
    a43=pd.read_csv(audit43path,parse_dates=['selection_month']).sort_values('selection_month')
    _window(flows.eom_ret,'Candidate flows');_window(proxy.eom_ret,'Proxy scores')
    if flows.duplicated(['cid','eom_ret']).any() or catalog.cid.duplicated().any():raise ValueError('Duplicate candidate keys')
    for name,audit in [('4.2',a42),('4.3',a43)]:
        _window(audit.selection_month,f'{name} audit')
        if not audit.year.eq(audit.selection_month.dt.year+1).all():raise ValueError('Selection is not prior-year')
    x=flows.pivot(index='eom_ret',columns='cid',values='active_gross').sort_index().reindex(columns=catalog.cid)
    z=flows.pivot(index='eom_ret',columns='cid',values='active_net').sort_index().reindex(columns=catalog.cid)
    costs=flows.pivot(index='eom_ret',columns='cid',values='tc').sort_index().reindex(columns=catalog.cid)
    if not x.index.equals(z.index) or not x.index.equals(costs.index):raise ValueError('Candidate calendars differ')
    if not np.isfinite(np.r_[x.to_numpy().ravel(),z.to_numpy().ravel(),costs.to_numpy().ravel()]).all():
        raise ValueError('Incomplete/nonfinite validation flows')
    if not x.index.equals(pd.date_range(x.index.min(),x.index.max(),freq='ME')):raise ValueError('Nonconsecutive validation months')
    rank_rows=[];robust_rows=[];last=None;winner_counts=None;literal_se_checks=0
    for date in x.index[x.index.month==12]:
        date=pd.Timestamp(date);X=x.loc[:date].to_numpy();Z=z.loc[:date].to_numpy();C=costs.loc[:date].to_numpy()
        score=_utility(X,Z);best_no=int(np.argmax(score));best=catalog.iloc[best_no]
        p=proxy.loc[proxy.eom_ret.eq(date)].set_index('cid').reindex(catalog.cid)
        if p.cum_obj.isna().any():raise ValueError('Proxy candidates missing at a December')
        proxy_no=int(np.argmax(p.cum_obj.to_numpy()));original=catalog.iloc[proxy_no]
        rank=pd.Series(score).rank(ascending=False,method='average').to_numpy()
        proxy_rank=pd.Series(p.cum_obj.to_numpy()).rank(ascending=False,method='average').to_numpy()
        rho=float(spearmanr(proxy_rank,rank).statistic)
        actual_audit=a42.loc[a42.selection_month.eq(date)]
        stable_audit=a43.loc[a43.selection_month.eq(date)]
        if len(actual_audit)!=1 or len(stable_audit)!=1:raise ValueError('A December pretest audit is missing or duplicated')
        check42=actual_audit.iloc[0];check43=stable_audit.iloc[0]
        np.testing.assert_allclose([check42.best_utility,check43.best_utility],[score[best_no]]*2,rtol=1e-9,atol=1e-12)
        if int(check42.best_cid)!=int(best.cid) or int(check43.best_cid)!=int(best.cid):raise ValueError('Executed best candidate disagrees with audit')
        rank_rows.append(dict(selection_month=date,forecast_year=date.year+1,history_months=len(X),
            candidates=len(catalog),spearman_proxy_vs_executed=rho,proxy_selected_cid=int(original.cid),
            executed_best_cid=int(best.cid),same_selected_cid=int(original.cid)==int(best.cid),
            proxy_winner_executed_rank=float(rank[proxy_no]),
            proxy_winner_executed_percentile=float((len(catalog)-rank[proxy_no])/(len(catalog)-1)),
            proxy_winner_executed_utility=float(score[proxy_no]),best_executed_utility=float(score[best_no]),
            validation_utility_gap=float(score[best_no]-score[proxy_no]),
            proxy_winner_annual_active_gross=float(12*X[:,proxy_no].mean()),
            executed_winner_annual_active_gross=float(12*X[:,best_no].mean()),
            proxy_winner_annual_cost=float(12*C[:,proxy_no].mean()),
            executed_winner_annual_cost=float(12*C[:,best_no].mean()),
            proxy_winner_annual_active_net=float(12*Z[:,proxy_no].mean()),
            executed_winner_annual_active_net=float(12*Z[:,best_no].mean())))
        reps=int(check43.bootstrap_reps);block=int(check43.block_months);seed=int(check43.bootstrap_seed)
        sampled=_bootstrap_scores(X,Z,reps,block,seed)
        se=float(np.std(sampled[:,best_no],ddof=1));literal_se_checks+=min(50,reps)
        np.testing.assert_allclose(se,check43.bootstrap_se,rtol=1e-8,atol=1e-12)
        threshold=float(score[best_no]-se)
        np.testing.assert_allclose(threshold,check43.utility_threshold,rtol=1e-8,atol=1e-12)
        eligible=catalog.loc[score>=threshold-1e-15]
        selected=eligible.sort_values(['p','l','g','cid'],ascending=[True,False,True,True],kind='stable').iloc[0]
        selected_no=int(np.flatnonzero(catalog.cid.to_numpy()==selected.cid)[0])
        if len(eligible)!=int(check43.eligible_count) or int(selected.cid)!=int(check43.selected_cid):raise ValueError('One-SE choice disagrees with audit')
        np.testing.assert_allclose(score[selected_no],check43.selected_utility,rtol=1e-9,atol=1e-12)
        maximum=sampled.max(axis=1);tied=sampled>=maximum[:,None]-1e-12
        credit=tied/tied.sum(axis=1)[:,None]
        robust_rows.append(dict(selection_month=date,forecast_year=date.year+1,history_months=len(X),candidate_count=len(catalog),
            best_cid=int(best.cid),best_p=int(best.p),best_l=float(best.l),best_g=float(best.g),best_utility=float(score[best_no]),
            bootstrap_se=se,utility_threshold=threshold,eligible_count=len(eligible),
            selected_cid=int(selected.cid),selected_p=int(selected.p),selected_l=float(selected.l),selected_g=float(selected.g),
            selected_utility=float(score[selected_no]),best_in_bootstrap_top_set_frequency=float(tied[:,best_no].mean()),
            best_tie_adjusted_bootstrap_win_frequency=float(credit[:,best_no].mean()),bootstrap_reps=reps,block_months=block,bootstrap_seed=seed))
        last=dict(date=date,score=score,rank=rank,proxy_rank=proxy_rank,proxy_no=proxy_no,best_no=best_no,
            proxy=original,best=best,selected=selected,selected_no=selected_no)
        winner_counts=catalog.assign(top_set_frequency=tied.mean(axis=0),tie_adjusted_win_frequency=credit.mean(axis=0))
    ranks=pd.DataFrame(rank_rows);robust=pd.DataFrame(robust_rows)
    if last is None:raise ValueError('No December pretest comparisons')
    folder42=audit42path.parent;folder43=audit43path.parent
    for folder in (folder42,folder43):(folder/'rationale').mkdir(parents=True,exist_ok=True)
    ranks.to_csv(folder42/'rationale/december_ranking_diagnostic.csv',index=False)
    robust.to_csv(folder43/'rationale/selection_uncertainty_diagnostic.csv',index=False)
    winner_counts.to_csv(folder43/'rationale/final_pretest_bootstrap_winner_stability.csv',index=False)
    stats42=dict(candidate_count=len(catalog),validation_months=len(x),last_outcome=str(x.index.max().date()),
        mean_december_rank_correlation=float(ranks.spearman_proxy_vs_executed.mean()),
        same_winner_count=int(ranks.same_selected_cid.sum()),december_comparisons=len(ranks),
        final_proxy_winner_executed_rank=float(ranks.iloc[-1].proxy_winner_executed_rank),
        final_proxy_winner_executed_percentile=float(ranks.iloc[-1].proxy_winner_executed_percentile),
        final_validation_utility_gap=float(ranks.iloc[-1].validation_utility_gap),
        motivation='Investigate executed validation ranking; a same-sample score gap is not proof of future improvement',
        current_proxy_computation_error_claimed=False,proxy_and_executed_score_levels_compared=False,
        future_strategy_outcomes_read=False,independent_unseen_evidence=False,gamma_te=GAMMA)
    stats43=dict(candidate_count=len(catalog),validation_months=len(x),last_outcome=str(x.index.max().date()),
        final_eligible_count=int(robust.iloc[-1].eligible_count),final_best_p=int(robust.iloc[-1].best_p),
        final_selected_p=int(robust.iloc[-1].selected_p),
        final_best_in_bootstrap_top_set_frequency=float(robust.iloc[-1].best_in_bootstrap_top_set_frequency),
        final_best_utility=float(robust.iloc[-1].best_utility),final_bootstrap_se=float(robust.iloc[-1].bootstrap_se),
        final_selected_utility=float(robust.iloc[-1].selected_utility),
        motivation='Test a fixed simplicity heuristic under executed-validation uncertainty, not formal indistinguishability',
        one_SE_is_hypothesis_test=False,one_SE_corrects_multiple_testing=False,
        future_strategy_outcomes_read=False,independent_unseen_evidence=False,gamma_te=GAMMA)
    draw42(ranks,last,catalog,folder42);draw43(robust,folder43)
    after={str(p):_digest(p) for p in sources}
    if before!=after:raise ValueError('Rationale input files changed while reporting')
    for folder,stats in ((folder42,stats42),(folder43,stats43)):
        (folder/'rationale/statistics.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
        (folder/'rationale/METHODS.md').write_text(METHODS,encoding='utf-8')
        verification=dict(status='PASS',sources=before,inputs_unchanged=True,latest_return='2019-12-31',
            model_fits_launched=False,PF_test_outcomes_read=False,pretest_audit_choices_reconciled=True,
            literal_all_candidate_bootstrap_sample_checks=literal_se_checks,bootstrap_best_SE_reconciled=True)
        (folder/'rationale/verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    return dict(stage42=stats42,stage43=stats43)


def draw42(rows,last,catalog,folder):
    fig=plt.figure(figsize=(15,9),dpi=180,facecolor='white')
    fig.text(.035,.955,'3 | PRETEST DIAGNOSIS',fontsize=12,weight='bold',color=TEAL)
    fig.text(.035,.905,'Iteration 4.2: does the learning-score winner survive actual execution?',fontsize=21,weight='bold',color=NAVY)
    fig.text(.035,.863,f"2015-2019 validation only | {len(catalog):,} fixed candidates | Original coefficients and execution | No 2020+ strategy outcomes",fontsize=10.5,color='#535e6b')
    axes=[fig.add_axes([.075+j*.31,.50,.27,.275]) for j in range(3)]
    years=rows.selection_month.dt.year.to_numpy()
    axes[0].bar(years,rows.spearman_proxy_vs_executed,color='#377f9c',width=.65)
    axes[0].set(title='Do the two rankings agree?',ylabel='Spearman rank correlation (descriptive)',ylim=(-1,1))
    axes[0].set_xticks(years,[str(y) for y in years]);axes[0].axhline(0,color='#777',lw=.7)
    n=len(catalog);px=100*(n-last['proxy_rank'])/(n-1);py=100*(n-last['rank'])/(n-1)
    axes[1].scatter(px,py,s=8,alpha=.18,color='#467d9d');axes[1].plot([0,100],[0,100],ls='--',lw=.8,color='#777')
    for index,color,label in [(last['proxy_no'],ORANGE,'Proxy-selected'),(last['best_no'],TEAL,'Executed-score best')]:
        axes[1].scatter(px[index],py[index],s=65,color=color,edgecolor='white',label=label,zorder=4)
    axes[1].set(title='Final pretest December: candidate ranks',xlabel='Proxy-score percentile (100 = best)',ylabel='Executed-utility percentile (100 = best)')
    axes[1].legend(fontsize=7,loc='lower right')
    values=100*np.array([last['score'][last['proxy_no']],last['score'][last['best_no']]])
    axes[2].bar(range(2),values,color=[ORANGE,TEAL],width=.6)
    axes[2].set_xticks(range(2),['Proxy-selected','Executed-score best'],fontsize=8)
    axes[2].set(title='Same-sample executed active utility',ylabel='Annual active utility (%)')
    axes[2].margins(y=.18)
    axes[2].axhline(0,color='#777',lw=.7)
    for i,v in enumerate(values):axes[2].annotate(f'{v:.3f}%',(i,v),xytext=(0,5 if v>=0 else -13),textcoords='offset points',ha='center',fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.tick_params(labelsize=8);ax.title.set_fontsize(10);ax.xaxis.label.set_fontsize(8);ax.yaxis.label.set_fontsize(8)
    final=rows.iloc[-1]
    table_rows=[
        ['December choices changed',f'{int((~rows.same_selected_cid).sum())} of {len(rows)}','Different choice is not evidence of a future gain'],
        ['Final proxy winner: executed rank',f"{final.proxy_winner_executed_rank:.1f} / {len(catalog)}",_format_model(last['proxy'])],
        ['Final executed winner',_format_model(last['best']),'Same candidate grid; only validation scoring changes'],
        ['Annual gross active return: proxy / executed best',f'{10000*final.proxy_winner_annual_active_gross:+.1f} / {10000*final.executed_winner_annual_active_gross:+.1f} bp','Validation sample, not held-out realised improvement'],
        ['Annual actual trading cost: proxy / executed best',f'{10000*final.proxy_winner_annual_cost:.1f} / {10000*final.executed_winner_annual_cost:.1f} bp','Costs from each candidate book\'s own inherited positions'],
    ]
    ax=fig.add_axes([.035,.25,.93,.17]);ax.axis('off')
    t=ax.table(cellText=table_rows,colLabels=['Economic / statistical diagnostic','Pretest observation','Interpretation'],colWidths=[.40,.25,.35],cellLoc='left',colLoc='left',bbox=[0,0,1,1]);_style(t)
    correlation=rows.spearman_proxy_vs_executed.mean()
    changed=int((~rows.same_selected_cid).sum())
    if correlation>.95 and rows.iloc[-1].same_selected_cid:
        conclusion=f'Rankings largely agree; only {changed} of {len(rows)} choices differ and the final winner is the same. The motivation is modest.'
    else:
        conclusion='The selection rules disagree: test whether executed validation selects a more useful active portfolio.' if changed else 'The winners agree: the measured rationale for changing selection is weak in this pretest sample.'
    fig.text(.035,.202,conclusion,fontsize=10.8,weight='bold',color=NAVY)
    fig.text(.035,.16,f'Mean December rank correlation = {correlation:.3f}. Rankings across correlated candidates are descriptive; no cross-candidate IID p-value is claimed.',fontsize=9.2,color='#454f5b')
    fig.text(.035,.121,'Executed score = 12 x [mean(net active return) - gamma/2 x sample variance(gross active return)]; gamma = 174.1.',fontsize=9.3,color='#454f5b')
    fig.text(.035,.082,'The original score is exact for its own quadratic proxy. Score levels are not compared; choosing a same-sample winner guarantees no future benefit.',fontsize=9.1,color='#454f5b')
    fig.text(.035,.041,'Rationale precedes the 4.2 test fit. These are reused calibration/validation data, not independent unseen confirmation.',fontsize=9,color='#68717d')
    _save_figure(fig,folder)
    pd.DataFrame(table_rows,columns=['Diagnostic','Pretest observation','Interpretation']).to_csv(folder/'rationale/presentation_table.csv',index=False)


def draw43(rows,folder):
    fig=plt.figure(figsize=(15,9),dpi=180,facecolor='white')
    fig.text(.035,.955,'3 | PRETEST DIAGNOSIS',fontsize=12,weight='bold',color=TEAL)
    fig.text(.035,.905,'Iteration 4.3: is the executed-validation winner stable enough to trust?',fontsize=21,weight='bold',color=NAVY)
    fig.text(.035,.863,'2015-2019 validation only | Same executed candidate books as 4.2 | A fixed one-SE simplicity heuristic | No test-stage outcomes',fontsize=10.5,color='#535e6b')
    axes=[fig.add_axes([.075+j*.31,.50,.27,.275]) for j in range(3)]
    years=rows.selection_month.dt.year.to_numpy();x=np.arange(len(rows))
    axes[0].errorbar(x,100*rows.best_utility,yerr=100*rows.bootstrap_se,fmt='o-',color=NAVY,capsize=4,label='Best score +/- one bootstrap SE')
    axes[0].scatter(x,100*rows.selected_utility,color=TEAL,marker='s',label='Simpler eligible choice',zorder=4)
    axes[0].set_xticks(x,years);axes[0].set(title='Utility separation versus estimation variation',ylabel='Annual executed active utility (%)')
    axes[0].legend(fontsize=6.8,loc='best')
    axes[1].bar(x,rows.eligible_count,color='#377f9c',width=.6)
    axes[1].set_xticks(x,years);axes[1].set(title='How many candidates lie within one SE?',ylabel='Eligible candidates (not independent tests)')
    for i,value in enumerate(rows.eligible_count):axes[1].annotate(str(int(value)),(i,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    axes[1].set_ylim(0,max(1,int(rows.eligible_count.max()))*1.2)
    axes[2].plot(x-.035,rows.best_p,'o-',color=ORANGE,label='Highest executed utility',lw=1.3,markersize=6)
    axes[2].plot(x+.035,rows.selected_p,'s-',color=TEAL,label='One-SE simpler choice',lw=1.3,markersize=6)
    axes[2].set_xticks(x,years);axes[2].set_yscale('log',base=2)
    ticks=sorted(set(rows.best_p)|set(rows.selected_p));axes[2].set_yticks(ticks,[str(int(p)) for p in ticks])
    axes[2].set_ylim(min(ticks)/1.2,max(ticks)*1.3)
    axes[2].set(title='Complexity of the annual choice',ylabel='RFF features p (log-2 scale)')
    axes[2].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.tick_params(labelsize=8);ax.title.set_fontsize(10);ax.yaxis.label.set_fontsize(8)
    final=rows.iloc[-1]
    table_rows=[
        ['Final pretest eligible set',f'{int(final.eligible_count)} / {int(final.candidate_count)}','Eligibility is a fixed one-SE heuristic, not a hypothesis test'],
        ['Best remains bootstrap-best (ties allowed)',f'{final.best_in_bootstrap_top_set_frequency:.1%}','Resampling stability; not probability of future success'],
        ['Final best / selected RFF count',f'{int(final.best_p)} / {int(final.selected_p)}','Tie-break: fewer features, stronger ridge, smaller RFF variance'],
        ['Final best / selected ridge penalty',f'{final.best_l:.3g} / {final.selected_l:.3g}','Coefficients and candidate grid are unchanged'],
        ['Annual validation utility sacrificed',f'{10000*(final.best_utility-final.selected_utility):.2f} bp','A possible robustness trade-off; no future gain assumed'],
    ]
    ax=fig.add_axes([.035,.25,.93,.17]);ax.axis('off')
    t=ax.table(cellText=table_rows,colLabels=['Uncertainty diagnostic','Pretest observation','Interpretation'],colWidths=[.39,.20,.41],cellLoc='left',colLoc='left',bbox=[0,0,1,1]);_style(t)
    conclusion='Several choices are near the best score: test a simpler choice before trusting a narrow validation winner.' if rows.eligible_count.gt(1).any() else 'Only the best is eligible: the one-SE refinement has little measured effect in this pretest sample.'
    fig.text(.035,.202,conclusion,fontsize=10.8,weight='bold',color=NAVY)
    fig.text(.035,.16,'The bootstrap recomputes both mean active return and active-return variance in every paired six-month-block resample.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.121,'One SE is a selection heuristic, not evidence of statistical equivalence or a correction for selecting among many candidates.',fontsize=9.3,color='#454f5b')
    fig.text(.035,.082,'Short early histories contain few effective time blocks. The candidate set and simplicity rule are fixed before test outcomes.',fontsize=9.1,color='#454f5b')
    fig.text(.035,.041,'Rationale precedes the 4.3 test fit. Retention has no 80% gate; later P(test) is descriptive, not independent unseen confirmation.',fontsize=9,color='#68717d')
    _save_figure(fig,folder)
    pd.DataFrame(table_rows,columns=['Diagnostic','Pretest observation','Interpretation']).to_csv(folder/'rationale/presentation_table.csv',index=False)


METHODS='''# Pretest-only rationale for the iteration-4 validation branch

Only candidate_flows.parquet, proxy_scores.csv, catalog.csv and the two pretest
diagnostic_selection_audit.csv files are read. Returns must fall within
January 2015-December 2019. No 4.2/4.3 fitted test portfolio or 2020+ return
outcome is read. Forecast-year labels 2016-2020 denote choices made at the end
of the preceding December. Every input is hashed before and after rendering.

The executed candidate score is 12 * [mean(active_net) - gamma/2 * sample
variance(active_gross, ddof=1)], with frozen gamma_TE 174.1. All candidate
histories through a selection December use the same dates. The original
quadratic proxy is exact for its own learning representation. Its LEVEL is
not compared with executed utility: the image compares candidate RANKS and
the executed utility/cost/active return of the respective selected candidates.
Spearman correlations are descriptive. Candidates are highly correlated and
do not form independent statistical observations; no cross-candidate IID
significance p-value is calculated. A same-sample best-versus-proxy-selected
utility gap is selected by construction and is not a future improvement.

The 4.3 rule uses the fixed annual audit bootstrap seed, rep count and block
length. Candidate active-net means AND sample active-gross variances are
recomputed in every paired circular-block resample. The frequency-matrix
implementation is checked against literal indexed bootstrap samples. The
best candidate's bootstrap standard deviation is its SE. Eligibility is
executed utility >= best utility - one SE. The chosen candidate is the
eligible candidate with fewest RFF features, then strongest ridge penalty,
then smaller RFF variance, then candidate ID. Audit thresholds, best score,
eligible counts, selected IDs and utilities are independently reconciled.

Bootstrap top-set frequency is how often the original best candidate remains
in the highest bootstrap-score set, allowing absolute utility ties within
1e-12. A separate tie-divided frequency is saved as well. These frequencies
describe validation-ranking instability conditional on existing candidate
returns; they are neither frequentist hypothesis tests nor probabilities of
future outperformance. No coefficients, risk forecasts or training histories
are re-estimated inside the bootstrap. The same rows are resampled for every
candidate to preserve contemporaneous dependence. Early 12-month histories
have few effective six-month blocks, limiting uncertainty resolution.

The one-SE rule is a fixed robustness/simplicity heuristic. It is not a proof
that eligible models are statistically indistinguishable, an equivalence
test, or a full multiple-testing correction. Fewer features or stronger ridge
may help or hurt. No test-stage outcome determines the rule. Both rationale
images use already-available calibration/validation data and do not constitute
independent unseen confirmation. Later P(test) is computed descriptively and
does not have an 80% retention gate for these two substages.
'''


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'outputs/iteration_4.2/us_large/candidate_cache/pretest')
    parser.add_argument('--audit42',type=Path,default=ROOT/'outputs/iteration_4.2/us_large/diagnostic_selection_audit.csv')
    parser.add_argument('--audit43',type=Path,default=ROOT/'outputs/iteration_4.3/us_large/diagnostic_selection_audit.csv')
    args=parser.parse_args()
    print(json.dumps(build_rationales(args.cache,args.audit42,args.audit43),indent=2))
