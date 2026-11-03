"""Read-only diagnosis of beta/sector neutrality after the saved market anchor.

Uses iteration 4 and its identical 1x-cost pretest replay. Projections/replayed
targets are diagnostic counterfactuals, not fitted or accepted new iterations.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflow import ROOT,config_path
from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.portfolio.accounting import add_start_weights,portfolio_returns
from analysis.ml_comparison.metrics import statistics

OUT=ROOT/'outputs/iteration_4/us_large/diagnostic_for_neutrality'
MONTH=pd.offsets.MonthEnd(1)
SEED=20261106
NAMES=['Stage 1: saved market anchor','Beta-neutral projection','Beta + sector-neutral projection']


def project(a,b,sigma,S):
    """Nested minimum-Sigma-distance projections; no forward returns used."""
    vb=float(b@sigma@b);va=float(a@sigma@a)
    beta=float(b@sigma@a/vb);z=a-beta*b
    invS=np.linalg.solve(sigma,S)
    Qs=invS-np.outer(b,b@S)/vb
    G=S.T@Qs;sector_correction=Qs@np.linalg.solve(G,S.T@z)
    joint=z-sector_correction
    # Complete sector neutrality also fixes net stock investment. Isolate that
    # budget component before attributing the remainder to sector contrasts.
    one=np.ones(len(a));Qone=np.linalg.solve(sigma,one)-b*b.sum()/vb
    budget_correction=Qone*(z.sum()/Qone.sum())
    after_budget=z-budget_correction;sector_only_correction=after_budget-joint
    vbudget=float(budget_correction@sigma@budget_correction)
    vsector_only=float(sector_only_correction@sigma@sector_only_correction)
    # Independent joint constrained solution, with unit risk-scaled columns.
    C=np.column_stack([sigma@b,S]);invC=np.linalg.solve(sigma,C)
    scale=np.sqrt(np.diag(C.T@invC));D=C/scale;invD=invC/scale
    direct=a-invD@np.linalg.solve(D.T@invD,D.T@a)
    np.testing.assert_allclose(joint,direct,rtol=1e-9,atol=2e-11)
    np.testing.assert_allclose([b@sigma@z,b@sigma@joint],0,atol=2e-12)
    np.testing.assert_allclose(S.T@joint,0,atol=2e-11)
    vmarket=beta*beta*vb;vsector=float(sector_correction@sigma@sector_correction);vleft=float(joint@sigma@joint)
    np.testing.assert_allclose(va,vmarket+vsector+vleft,rtol=1e-9,atol=1e-14)
    np.testing.assert_allclose(vsector,vbudget+vsector_only,rtol=1e-9,atol=1e-14)
    return z,joint,dict(active_variance=va,benchmark_variance=vb,active_beta=beta,
        beta_projection_variance=vmarket,conditional_sector_projection_variance=vsector,
        net_stock_budget_projection_variance=vbudget,sector_contrast_projection_variance=vsector_only,
        remaining_variance=vleft,sector_gross_deviation=float(np.abs(S.T@a).sum()),
        sector_gross_deviation_after_beta=float(np.abs(S.T@z).sum()),
        max_constraint_error=float(max(abs(b@sigma@joint),np.max(abs(S.T@joint)))))


def independent_accounting(w,pf,panel,wealth):
    d=w.merge(panel[['id','eom','ret_ld1','lambda']],on=['id','eom'],validate='many_to_one').merge(
        wealth[['eom','wealth']],on='eom',validate='many_to_one')
    assert d[['lambda','wealth']].notna().all().all() and d.loc[~d.exit,'ret_ld1'].notna().all()
    literal=pd.DataFrame(dict(eom=d.eom,r=d.w*d.ret_ld1.where(~d.exit,0.),
        tc=.5*d.wealth*d['lambda']*(d.w-d.w_start)**2,turnover=(d.w-d.w_start).abs())).groupby('eom').sum()
    actual=pf.set_index('eom')
    np.testing.assert_allclose(literal.loc[actual.index],actual[literal.columns],rtol=1e-10,atol=1e-12)


def replay(phase,saved,pf,panel,wealth,cov):
    dates=pd.DatetimeIndex(pf.eom).sort_values();targets=[[],[],[]];risk=[];sector_rows=[]
    for date,g in saved.loc[~saved.exit].groupby('eom',sort=True):
        c=cov[date];ids=g.id.to_numpy();p=panel.loc[panel.eom.eq(date)].set_index('id').reindex(ids)
        b=p.me.to_numpy()/p.me.sum();a=g.w.to_numpy()-b;sigma=c.full(ids)
        names=[s for s in c.factors if s.startswith('sec_')]
        S=c.X[c._rows(ids)][:,[c.factors.index(s) for s in names]]
        assert len(names)==6 and np.array_equal(S.sum(axis=1),np.ones(len(ids)))
        z,joint,record=project(a,b,sigma,S)
        record.update(eom=date,eom_ret=date+MONTH,phase=phase);risk.append(record)
        for name,exposure,after in zip(names,S.T@a,S.T@z):
            sector_rows.append(dict(phase=phase,eom=date,sector=name,active_weight=exposure,after_beta_active_weight=after))
        for collection,active in zip(targets,[a,z,joint]):
            collection.append(pd.DataFrame(dict(id=ids,eom=date,w=b+active)))
    folder=OUT/phase;folder.mkdir(parents=True,exist_ok=True)
    books={};metrics=[]
    for number,(label,rows) in enumerate(zip(NAMES,targets)):
        tg=pd.concat(rows,ignore_index=True)
        w=add_start_weights(tg,panel,wealth,dates,True)
        p=portfolio_returns(w,panel,wealth)
        independent_accounting(w,p,panel,wealth)
        if number==0:
            np.testing.assert_allclose(p[['r','tc','turnover']],pf.sort_values('eom')[['r','tc','turnover']],rtol=1e-10,atol=1e-12)
        w.to_parquet(folder/f'book_{number}_weights.parquet',index=False)
        p.to_parquet(folder/f'book_{number}_pf.parquet',index=False)
        books[number]=p.sort_values('eom_ret').reset_index(drop=True)
        metrics.append(dict(phase=phase,book=number,strategy=label,fitted_new_model=False,**statistics(p)))
    risk=pd.DataFrame(risk).sort_values('eom').reset_index(drop=True)
    for number in (1,2):
        risk[f'net_difference_{number}']=(books[number].r-books[number].tc)-(books[0].r-books[0].tc)
        risk[f'cost_difference_{number}']=books[number].tc-books[0].tc
    risk['net_difference_3']=risk.net_difference_2-risk.net_difference_1
    risk['cost_difference_3']=risk.cost_difference_2-risk.cost_difference_1
    risk.to_csv(folder/'monthly_risk_and_replay.csv',index=False)
    pd.DataFrame(sector_rows).to_csv(folder/'sector_active_weights.csv',index=False)
    pd.DataFrame(metrics).to_csv(folder/'mechanical_replay_performance.csv',index=False)
    return risk,pd.DataFrame(metrics)


def bootstrap(rows,block,reps=10000):
    n=len(rows);rng=np.random.default_rng(SEED);starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    ix=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    variance=rows.active_variance.to_numpy()[ix].sum(axis=1)
    values=dict(beta_share=rows.beta_projection_variance.to_numpy()[ix].sum(axis=1)/variance,
        conditional_sector_share=rows.conditional_sector_projection_variance.to_numpy()[ix].sum(axis=1)/variance,
        budget_share=rows.net_stock_budget_projection_variance.to_numpy()[ix].sum(axis=1)/variance,
        sector_contrast_share=rows.sector_contrast_projection_variance.to_numpy()[ix].sum(axis=1)/variance,
        mean_active_beta=rows.active_beta.to_numpy()[ix].mean(axis=1))
    for number in (1,2,3):
        for kind in ('net','cost'):
            values[f'annual_{kind}_difference_{number}']=12*rows[f'{kind}_difference_{number}'].to_numpy()[ix].mean(axis=1)
    for i in range(100):
        literal=[(int(start)+j)%n for start in starts[i] for j in range(block)][:n];sample=rows.iloc[literal]
        for name,expected in dict(beta_share=sample.beta_projection_variance.sum()/sample.active_variance.sum(),
            conditional_sector_share=sample.conditional_sector_projection_variance.sum()/sample.active_variance.sum(),
            sector_contrast_share=sample.sector_contrast_projection_variance.sum()/sample.active_variance.sum(),
            annual_net_difference_1=12*sample.net_difference_1.mean(),annual_net_difference_2=12*sample.net_difference_2.mean(),
            annual_net_difference_3=12*sample.net_difference_3.mean()).items():
            np.testing.assert_allclose(values[name][i],expected,rtol=1e-11,atol=1e-13)
    out=dict(block_months=block)
    for key,value in values.items():
        low,high=np.quantile(value,[.025,.975]);out[key+'_low']=float(low);out[key+'_high']=float(high)
    return out


def main():
    cfg=load_config(config_path('us_large'));p=cfg.paths.processed
    paths=dict(test_weights=ROOT/'outputs/iteration_4/us_large/weights.parquet',
        test_pf=ROOT/'outputs/iteration_4/us_large/pf.parquet',
        pretest_weights=ROOT/'outputs/iteration_6/us_large/pretest_trials/cost_1/weights.parquet',
        pretest_pf=ROOT/'outputs/iteration_6/us_large/pretest_trials/cost_1/pf.parquet',
        pretest_provenance=ROOT/'outputs/iteration_6/us_large/pretest_trials/cost_1/completion.json',
        stage1_calibration=ROOT/'outputs/iteration_4/us_large/calibrated.csv',panel=p/'panel.parquet',
        wealth=p/'wealth.parquet',cov_loadings=p/'covariance/cov_loadings.parquet',cov_factor=p/'covariance/cov_factor.parquet')
    hashes={str(v.relative_to(ROOT)):hashlib.file_digest(v.open('rb'),'sha256').hexdigest() for v in paths.values()}
    provenance=json.loads(paths['pretest_provenance'].read_text())['row']
    gamma=float(pd.read_csv(paths['stage1_calibration']).gamma_te.iloc[0])
    assert provenance['training_cost_multiplier']==1 and provenance['actual_cost_multiple']==1 and provenance['gamma_te']==gamma
    pre_w=pd.read_parquet(paths['pretest_weights']);test_w=pd.read_parquet(paths['test_weights'])
    pre_pf=pd.read_parquet(paths['pretest_pf']).sort_values('eom');test_pf=pd.read_parquet(paths['test_pf']).sort_values('eom')
    assert pd.DatetimeIndex(pre_pf.eom_ret).equals(pd.date_range('2016-01-31','2019-12-31',freq='ME'))
    assert pd.DatetimeIndex(test_pf.eom_ret).equals(pd.date_range('2020-01-31','2025-12-31',freq='ME'))
    ids=list(set(pre_w.id)|set(test_w.id))
    panel=pd.read_parquet(paths['panel'],columns=['id','eom','me','ret_ld1','tr_ld1','lambda'],
        filters=[('id','in',ids),('eom','>=',pre_w.eom.min()),('eom','<=',test_w.eom.max())])
    wealth=pd.read_parquet(paths['wealth']);cov=load_covariance(p/'covariance')
    OUT.mkdir(parents=True,exist_ok=True);stats={};tables=[];sensitivity=[]
    for phase,w,pf in [('pretest',pre_w,pre_pf),('test',test_w,test_pf)]:
        risk,table=replay(phase,w,pf,panel,wealth,cov);tables.append(table)
        variance=risk.active_variance.sum()
        s=dict(months=len(risk),mean_total_ex_ante_beta=float(1+risk.active_beta.mean()),
            min_total_ex_ante_beta=float(1+risk.active_beta.min()),max_total_ex_ante_beta=float(1+risk.active_beta.max()),
            beta_share=float(risk.beta_projection_variance.sum()/variance),
            conditional_sector_share=float(risk.conditional_sector_projection_variance.sum()/variance),
            budget_share=float(risk.net_stock_budget_projection_variance.sum()/variance),
            sector_contrast_share=float(risk.sector_contrast_projection_variance.sum()/variance),
            remaining_share=float(risk.remaining_variance.sum()/variance),
            mean_gross_sector_deviation=float(risk.sector_gross_deviation.mean()),
            mean_gross_sector_deviation_after_beta=float(risk.sector_gross_deviation_after_beta.mean()),
            max_projection_constraint_error=float(risk.max_constraint_error.max()))
        for number in (1,2,3):
            s[f'annual_net_difference_{number}']=float(12*risk[f'net_difference_{number}'].mean())
            s[f'annual_cost_difference_{number}']=float(12*risk[f'cost_difference_{number}'].mean())
        for block in (3,6,12):sensitivity.append(dict(phase=phase,**bootstrap(risk,block)))
        stats[phase]=s
    stats.update(new_models_fitted=False,projections_are_formal_iteration_results=False,
        no_TE_rescaling=True,shares_are_projection_geometry_not_causal_attribution=True,
        pretest_is_calibration_sample=True,independent_unseen_evidence=False,
        bootstrap_reps=10000,bootstrap_seed=SEED,
        technical_source='https://docs.mosek.com/portfolio-cookbook/benchmarkrel.html')
    pd.DataFrame(sensitivity).to_csv(OUT/'block_bootstrap_sensitivity.csv',index=False)
    pd.concat(tables,ignore_index=True).to_csv(OUT/'mechanical_replay_performance.csv',index=False)
    (OUT/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf-8')
    draw(stats,pd.DataFrame(sensitivity),pd.concat(tables,ignore_index=True))
    (OUT/'METHODS.md').write_text(METHODS,encoding='utf-8')
    after={str(v.relative_to(ROOT)):hashlib.file_digest(v.open('rb'),'sha256').hexdigest() for v in paths.values()}
    assert hashes==after
    (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',sources=hashes,source_files_unchanged=True,
        months=120,independent_projection_KKT_checks=120,risk_decomposition_identities=120,
        independently_accounted_books=6,original_stage1_replays_matched=2,literal_bootstrap_samples_checked=600,
        forward_returns_used_to_construct_projection=False,new_models_fitted=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(stats,indent=2));return 0


def draw(s,intervals,performance):
    navy='#1c304d';teal='#00868b';orange='#dd842b';colors=[orange,'#9467a8','#447caa','#9ba6b3']
    fig=plt.figure(figsize=(15,9.3),dpi=180,facecolor='white')
    fig.text(.035,.955,'3 | DIAGNOSIS',fontsize=12,weight='bold',color=teal)
    fig.text(.035,.905,'After the market anchor: do beta and sector neutrality deserve a test?',fontsize=21,weight='bold',color=navy)
    fig.text(.035,.863,'US high-cap | Pretest: 2016-2019 (48 months) | Inspected test: 2020-2025 (72 months) | No new ML fits',fontsize=11,color='#535e6b')
    axes=[fig.add_axes([.075+j*.31,.505,.27,.27]) for j in range(3)]
    risk_keys=['beta_share','budget_share','sector_contrast_share','remaining_share']
    risk_names=['Benchmark-beta component','Net stock/cash budget','Sector contrasts after beta/budget','Remaining active risk']
    for j,(key,name) in enumerate(zip(risk_keys,risk_names)):
        bottom=np.zeros(2) if j==0 else sum(100*np.array([s[p][k] for p in ('pretest','test')]) for k in risk_keys[:j])
        values=100*np.array([s[p][key] for p in ('pretest','test')])
        axes[0].bar([0,1],values,bottom=bottom,color=colors[j],label=name.replace('\n',' '),width=.6)
        for x,value,base in zip([0,1],values,bottom):
            if value>7:axes[0].text(x,base+value/2,f'{value:.1f}%',ha='center',va='center',fontsize=9,color='white' if j<3 else navy)
    axes[0].set_xticks([0,1],['Pretest','Test']);axes[0].set_ylim(0,103)
    axes[0].set_title('Where neutrality removes modelled active risk',fontsize=10)
    axes[0].set_ylabel('Share of summed predicted active variance (%)',fontsize=8)
    axes[0].legend(fontsize=6.8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False)
    labels=['Pretest','Test'];b=np.array([s[p]['mean_total_ex_ante_beta'] for p in ('pretest','test')])
    axes[1].plot([0,1],b,'o',color=navy,markersize=7)
    axes[1].vlines([0,1],[s[p]['min_total_ex_ante_beta'] for p in ('pretest','test')],
        [s[p]['max_total_ex_ante_beta'] for p in ('pretest','test')],color='#6d8ba6',lw=3)
    axes[1].axhline(1,color=orange,ls='--',lw=1,label='Benchmark beta = 1')
    axes[1].set_xticks([0,1],labels);axes[1].set_xlim(-.5,1.5)
    axes[1].set_title('The anchor is already close to beta one',fontsize=10)
    axes[1].set_ylabel('Ex-ante total beta: dot = mean; line = range',fontsize=8)
    axes[1].legend(fontsize=7,loc='lower left')
    for x,value in enumerate(b):axes[1].annotate(f'{value:.4f}',(x,value),xytext=(8,5),textcoords='offset points',fontsize=9)
    for j,(phase,color) in enumerate([('pretest',navy),('test',teal)]):
        costs=[10000*performance.loc[performance.phase.eq(phase)&performance.book.eq(i),'TC'].iloc[0] for i in (0,1,2)]
        axes[2].bar(np.arange(3)+(j-.5)*.32,costs,width=.32,color=color,label=phase.capitalize())
    axes[2].set_xticks(range(3),['Saved\nanchor','Beta-neutral\nreplay','Beta + sector\nreplay'],fontsize=8)
    axes[2].set_title('Economic check: actual costs of projected targets',fontsize=9.5)
    axes[2].set_ylabel('Annual trading cost (basis points)',fontsize=8)
    axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.tick_params(labelsize=8)
    rows=[]
    for key,title in [('beta_share','Beta constraint: modelled variance removed'),
                      ('conditional_sector_share','Sector + budget constraints: additional removal'),
                      ('sector_contrast_share','Of that: sector contrasts after beta / budget')]:
        def fmt(phase):
            ci=intervals.loc[intervals.phase.eq(phase)&intervals.block_months.eq(6)].iloc[0]
            return f'{s[phase][key]:.1%} [{ci[key+"_low"]:.1%}, {ci[key+"_high"]:.1%}]'
        rows.append([title,fmt('pretest'),fmt('test'),'Model geometry, not expected return improvement'])
    for i,title in [(1,'Beta replay: annual net difference vs anchor'),(3,'Sector replay: incremental net difference vs beta')]:
        def fmt(phase):
            ci=intervals.loc[intervals.phase.eq(phase)&intervals.block_months.eq(6)].iloc[0];key=f'annual_net_difference_{i}'
            return f'{10000*s[phase][key]:+.1f} [{10000*ci[key+"_low"]:+.1f}, {10000*ci[key+"_high"]:+.1f}] bp'
        rows.append([title,fmt('pretest'),fmt('test'),'Mechanical replay only; no refit or TE rescaling'])
    ax=fig.add_axes([.035,.25,.93,.17]);ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Diagnostic (95% block intervals)','Pretest: calibration sample','Test: already inspected','Interpretation'],
        colWidths=[.33,.22,.22,.23],cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(8.6)
    for (r,c),cell in table.get_celld().items():
        cell.PAD=.02;cell.set_edgecolor('#68717d');cell.set_linewidth(.6)
        cell.set_facecolor(navy if r==0 else ('#eef2f7' if r%2 else 'white'))
        cell.set_text_props(color='white' if r==0 else '#253140',weight='bold' if r==0 else 'normal')
    fig.text(.035,.207,'Stage 2: beta neutrality is a modest exposure-control test; the data does not establish a major beta problem.',fontsize=11,weight='bold',color=navy)
    fig.text(.035,.17,'Stage 3: sector neutrality also fixes the stock/cash budget and raises replay costs; improvement is not established.',fontsize=11,weight='bold',color=navy)
    fig.text(.035,.128,'Risk shares use nested covariance-metric projections: beta, stock/cash budget, then sector contrasts. They are not causal attribution.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.093,'Replays fully rebalance projected targets; they are NOT constrained Portfolio-ML results. Remaining risk also includes style exposures.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.057,'Only consistent constrained learning/execution and pretest calibration can test a new iteration. No 80% acceptance verdict is claimed here.',fontsize=9.5,color='#454f5b')
    fig.text(.035,.023,'10,000 paired six-month-block resamples; 3/12-month sensitivity saved. Conditional on existing estimates; no independent unseen or guaranteed-gain claim.',fontsize=8.2,color='#68717d')
    for ext in ('png','pdf'):fig.savefig(OUT/f'neutrality_stage_diagnostic.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    pd.DataFrame(rows,columns=['Diagnostic','Pretest','Test','Interpretation']).to_csv(OUT/'presentation_table.csv',index=False)


METHODS='''# Diagnose two refinements after the market anchor

Stage 1 is saved iteration 4. The 2016-2019 sample uses its existing replay at
gamma_TE 174.1, archived as the iteration-6 1x-cost control. Its provenance must
confirm original costs and gamma. This is NOT iteration-6's selected 2x model.
The pretest sample was used for TE calibration and is not independent validation.
2020-2025 is already inspected research data, not a fresh holdout.

## Risk geometry (formation-month information only)

b is the current market-cap-weight benchmark, a=w-b, and Sigma is the existing
formation-month covariance. The active benchmark beta is q=(b'Sigma a)/(b'Sigma b).
The beta-neutral projection is z=a-q b. Its variance falls by q^2 b'Sigma b.
Sector indicators S use the six existing covariance-model sectors. Within the
beta-neutral subspace, Q=Sigma^-1-b b'/(b'Sigma b) and the joint projection is
z_joint=z-Q S (S'Q S)^-1 S'z. Both b'Sigma z_joint and S'z_joint are zero.
All six sector constraints also force net active stock investment to zero;
this restricts stock/cash exposure as well as sector allocations.

Variance is decomposed into beta-constraint removal, additional sector-constraint
removal conditional on beta neutrality, and remaining variance. The components
are orthogonal under the covariance metric and sum to the original variance.
Shares are ratios of summed monthly predicted variances, not average monthly
percentages. Constraint order and the risk model matter. They do not uniquely
identify economic causes, expected alpha, or causal profits. Sector-neutral
remaining risk still includes style and individual-stock exposures. Ex-ante
beta is not the full-period realised regression beta reported in other charts.
Every month is independently checked against a risk-scaled joint KKT solution.
To avoid calling a stock/cash allocation effect a sector effect, the additional
joint projection is subdivided: first impose 1'a=0 within the beta-neutral
subspace, then neutralise sector contrasts. These two orthogonal variance
components sum to the joint sector/budget removal. The intermediate budget-only
projection is an attribution calculation, not another investment experiment.

## Mechanical economic replay, NOT a trained new iteration

Targets b+z and b+z_joint are projected from the saved stage-1 shadow-book weights.
Each replay is a separate portfolio, fully rebalanced to those monthly targets.
Its OWN preceding weights are drifted using the original stock and market wealth
growth, and original exits and quadratic costs are applied. Initialisation is
the same first-month VW book. No scaling back to a 1% TE target is performed.
This target replay differs from the original ML partial-adjustment execution;
it is neither an exact constrained optimisation nor a learned stage-2/3 result.
It estimates the consequences of this PARTICULAR mechanical overlay, including
costs, not a bound on all possible implementations or a forecast of a refit.
The saved original targets must reproduce original returns, costs and turnover.
All replay stock returns and costs are reconstructed independently.

Net/cost differences use 12 times the mean monthly difference versus stage 1.
The incremental sector comparison subtracts the beta-neutral replay, so it
isolates the next diagnostic step; the joint-minus-anchor difference is retained
in the statistics files. This conditional comparison does not assume a future
beta-neutral ML experiment will pass. Rejected changes must not be inherited.
The 95% percentile intervals use 10,000 paired circular-block resamples with
six-month blocks, seed 20261106; 3/12-month sensitivity is also retained. They
condition on the saved weights/covariances, do not re-estimate models, and are
not corrected for past research selection or multiple displayed comparisons.
Positive risk shares are mechanical, not significance tests for harmful bets.
There is no P(better) acceptance decision for these diagnostic replays.

Interpretation: a low beta-risk share weakens, rather than proves, the rationale
for beta neutrality. Sector exposure can be intentional and rewarded. A large
sector projection share motivates attribution/mandate scrutiny, not necessarily
a hard constraint. A proper new experiment must integrate constraints into
learning and execution, retain actual costs, calibrate using pretest data only,
and compare with the last accepted strategy. These are project extensions.

Technical reference: https://docs.mosek.com/portfolio-cookbook/benchmarkrel.html
The existing paper replication and characteristic/date/stock universes are not
changed. No new investment model is fitted or accepted by this diagnostic.
'''


if __name__=='__main__':raise SystemExit(main())
