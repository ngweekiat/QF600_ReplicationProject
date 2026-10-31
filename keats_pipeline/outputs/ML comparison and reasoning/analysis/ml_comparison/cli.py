"""Evaluate immutable saved Keats outputs. No model runners are imported or invoked."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .paths import CODE_ROOT, COMPARISON_ROOT, REPOSITORY_ROOT

ROOT=REPOSITORY_ROOT
sys.path.insert(0,str(ROOT/'keats_pipeline/src'))

from .comparison import correlations,cross_dataset,performance_decomposition,regimes,weight_similarity
from .data import align,discover,snapshot
from .inference import bootstrap_rank_stability,compare_pairs
from .metrics import annual,metric_dictionary,paths_and_rolling,ranks,table
from .report import charts,summary


def save(frame,path,index=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=index)


def matrices(pairs,destination):
    for dataset,g in pairs.groupby('dataset'):
        names=sorted(set(g.strategy_a)|set(g.strategy_b))
        for field,complement in [('posterior_probability_a_gt_b',True),('utility_hac_p_holm',False),('sharpe_hac_p_holm',False)]:
            out=pd.DataFrame(np.nan,index=names,columns=names)
            for row in g.itertuples():
                value=getattr(row,field)
                out.loc[row.strategy_a,row.strategy_b]=value
                out.loc[row.strategy_b,row.strategy_a]=1-value if complement else value
            out.index.name='row_strategy'
            save(out,destination/'tables'/dataset.replace('/','__')/f'table3_{field}.csv',index=True)


def independent_validation(root,destination,monthly,t2,pairs):
    """Recompute from arrays independently of all project reporting functions."""
    findings=[]
    for (dataset,strategy),g in monthly.groupby(['dataset','strategy']):
        r,tc=g.r.to_numpy(float),g.tc.to_numpy(float)
        n=len(r);mu=sum(r)/n;cost=sum(tc)/n
        variance=sum((x-mu)**2 for x in r)/(n-1);sd=variance**.5
        expected={'R':12*mu,'Vol':np.sqrt(12)*sd,'SR_gross':np.sqrt(12)*mu/sd,
            'TC':12*cost,'R_TC':12*(mu-cost),'SR_net':np.sqrt(12)*(mu-cost)/sd,
            'Utility':12*(mu-cost-5*variance),'Turnover':sum(g.turnover)/n,'Lev':sum(g.inv)/n}
        row=t2.loc[(t2.dataset==dataset)&(t2.strategy==strategy)].iloc[0]
        err=max(abs(row[k]-value)/max(1.,abs(value)) for k,value in expected.items())
        identity=abs(row.R-row.TC-row.R_TC)
        findings.append(dict(dataset=dataset,strategy=strategy,check='independent_array_formula_recomputation',
            status='PASS' if err<1e-11 and identity<1e-8 else 'FAIL',max_scaled_error=err,gross_minus_cost_minus_net=identity))
        if pairs is not None and not pairs.empty:
            for z in pairs.loc[(pairs.dataset==dataset)&(pairs.strategy_a==strategy)].itertuples():
                other=t2.loc[(t2.dataset==dataset)&(t2.strategy==z.strategy_b)].iloc[0]
                error=abs(z.utility_diff_annual-(row.Utility-other.Utility))
                findings.append(dict(dataset=dataset,strategy=f'{strategy} vs {z.strategy_b}',check='paired_effect_matches_Table2_difference',
                    status='PASS' if error<1e-8*max(1,abs(z.utility_diff_annual)) else 'FAIL',max_scaled_error=error/max(1,abs(z.utility_diff_annual))))
    out=pd.DataFrame(findings)
    save(out,destination/'validation/independent_metric_checks.csv')
    if (out.status=='FAIL').any():raise AssertionError('Independent metric recomputation failed.')
    return out


def assumptions(destination,inventory,missing,settings):
    text=f'''# Exclusions, assumptions and changes

Run timestamp: {settings['created_sgt']} (Asia/Singapore).

## Scope and changes made for this task

- Independent evaluation code and analysis outputs are together under `keats_pipeline/outputs/ML comparison and reasoning/`; code is in its `analysis/ml_comparison/` subdirectory.
- No existing ML implementation, configuration, prediction, portfolio, covariance estimate or source input was edited or fitted by the evaluation scripts.
- The previously authorised full-run process tree (PID 56932 and child 92028) was stopped before this analysis to prevent additional fitting. Its already saved partial outputs are retained. Status logs can still say running because stopping a process does not finalise its state file.
- Six pre-existing modified files were present before this task: Keats README, apply_history_policy.py, US-mid base.yaml, scripts/13_breadth.py, portfolio/recursion.py and test_execution_safety.py. Those are prior work, not comparison changes. Their current bytes are checked before/after this evaluation.
- The user's latest instruction moved both code and analysis into `keats_pipeline/outputs/ML comparison and reasoning/`. Historical run logs retain their original recorded paths; `RELOCATION.md` explains the mapping. New runs use the relocated folder.

## Sample and exclusions

- Primary sample is the declared test return window January 2020–December 2025. Models' saved validation returns are not included in headline test rankings.
- Within each dataset, every strategy uses identical observed return months; cross-dataset comparisons use the common overlap. No missing return month is filled. `tables/date_alignment_exclusions.csv` lists every date excluded by alignment.
- Only available saved strategies are compared. Grid/capacity and feature-permutation experiments are kept in separate diagnostics and are never counted as additional independently trained models.
- Unavailable strategies/variants and absent weights are marked SKIPPED, rather than inferred from other universes. UK-only and independently changed feature-set datasets are not available.
- Primary universe labels preserve base (about 80 names), US-large 250-stock, and any saved active-TE variants. No pooling of stocks, returns, currencies or incompatible mandates occurs.

## Existing upstream assumptions retained, not newly introduced

- Missing held-stock next-month excess returns were set to zero by prior user instruction. The previously reported full-history counts are 15 US-large, 21 US-mid, 20 European observations; selected 250-stock memberships can have different affected rows. The audit copies actual flags for every evaluated prepared panel, including original missing values, dates and IDs. These are assumptions, not observed zero returns; total returns equal the next-month USD risk-free rate.
- Unavailable lagged history caused specific stock-month exclusions in previously prepared 250-stock panels. Saved exclusion logs identify IDs, missing lag months and dates. Existing feature-missing values retain the pipeline's neutral 0.5 rank convention.
- The actual saved application ledgers contain two US-mid 250-stock exclusions. A third source history gap concerns the unprepared European 250-stock dataset; it is documented separately as pending, rather than counted as an applied exclusion. The audit preserves both ledgers and preparation-coverage status.
- Saved regional membership and training populations are used without rebuilding universes. US universes include NYSE and Nasdaq; European large-cap includes multiple countries, not UK only.
- Daily European returns are USD-adjusted under existing source construction; venue volume, listing selection and delisting treatment are not independently established by an accounting identity. Missing terminal returns and surviving data gaps limit survivorship claims.
- All portfolios use existing market-linked wealth paths with the configured $10B anchor at end-2019; covariance, feature ranks, lagged signals, liquidity coefficients and modelled impact are reused exactly.

## Metric, inference and scenario assumptions

- Paper Table 2: annual mean/cost/utility ×12, gross volatility ×sqrt(12), sample variance ddof=1, paper net Sharpe uses GROSS SD, monthly two-sided turnover (no half factor). Main evaluation gamma=10, even for active models with a distinct calibrated gamma_TE objective.
- Table 3 paper utility-flow probability uses the exact normal approximation from authors' R code. It is labelled a posterior-style probability, not a frequentist p-value. HAC tests and block bootstraps are separate evaluation extensions. The population- versus sample-variance finite-sample difference is explicit.
- Paired circular moving-block bootstrap: {settings['bootstrap_reps']} replications, fixed seed {settings['seed']}, primary block length 6 months, sensitivity lengths 3 and 12. Bootstrap resamples all paired values jointly and re-estimates means/variances.
- Bartlett Newey–West lag choice floor(4*(T/100)^(2/9)), capped T−1; normal-reference two-sided inference. Holm correction applies within each dataset separately for utility and Sharpe pair families. It does not correct undisclosed prior research trials or every exploratory scenario.
- Re-costing 0/0.5/1/2/3 times existing TC, fixed-weight AUM scaling, and evaluation gamma 1/5/10/20/100 do not re-optimise or retune models. Actual saved optimised grid points are labelled separately and incomplete grids remain incomplete.
- Raw dollar volume is used for liquidity diagnostics when available; rank-normalised features are not treated as dollar amounts. Formation months determine trade costs; realised next-month returns determine performance.
- Synthetic compounded excess-return indices are not total investor wealth. Any net month <=−100% invalidates full-sample geometric metrics; paths stop at that month. Paper arithmetic statistics retain all such observations with validity flags. No winsorisation or selective removal is applied.
- Regimes are descriptive. Local market up/down uses the contemporaneous saved Market return; VIX is prior-month mean with fixed threshold 20; USREC is current-vintage US context, including for Europe. Calendar stress windows and rolling statistics are not trading signals.
- Modelled quadratic impact can exceed investor wealth when leverage/turnover is unbounded or venue volume is tiny. Saved portfolio rankings in such cases are algebraic diagnostics, not feasible executed-investment claims. No external liquidity/participation/borrowing constraint is imposed after the fact.

## Unavailable tests

- Independent fresh holdout after 2025; multi-horizon 2–12 month predictions; independently refitted model random-seed return paths; unrun gamma/AUM grid points; alternative covariance estimators not already saved; externally observed execution costs and constraint-aware re-optimisation.
- Look-ahead checks inspect saved prediction/selection/covariance dates and source logic. No future-return perturbation refit is performed under the no-retraining instruction; historical point-in-time data vintages cannot be proven from final saved values alone.
- See `audit/`, `robustness/skipped_tests.csv`, and `tables/missing_saved_outputs.csv` for dataset-specific detail and all retained exceptions.

## Verification

- Each paper metric is independently recomputed from arrays and matched to reused project functions; accounting is independently reconstructed from available weights, panel returns, wealth, liquidity and inherited weights.
- Source SHA256 inventories before and after evaluate saved code/config/results; evaluated PF/weight files and data provenance are recorded. No model runner is invoked.
- Automated tests exercise posterior direction, paired bootstrap, HAC arithmetic, formula annualisation, turnover/exit semantics, geometric invalidity, common dates, missing values and preservation. Test output is in `validation/`.
'''
    (destination/'EXCLUSIONS_ASSUMPTIONS_AND_CHANGES.md').write_text(text,encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['all','performance','inference','robustness','audit'],default='all')
    parser.add_argument('--universe',choices=['all','us_large','us_mid','eu_large'],default='all')
    parser.add_argument('--output',type=Path,help='New, non-existing directory under keats_pipeline/outputs/ML comparison and reasoning')
    parser.add_argument('--bootstrap-reps',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=20261101)
    parser.add_argument('--run-tests',action='store_true',help='Run comparison and existing Keats tests; save the log under validation/')
    args=parser.parse_args(argv)
    now=datetime.now(ZoneInfo('Asia/Singapore'))
    output_root=COMPARISON_ROOT
    dest=(args.output or output_root/f"r{now:%Y%m%d_%H%M%S}").resolve()
    if not dest.is_relative_to(output_root.resolve()) or dest==output_root.resolve():
        parser.error('Use a new subdirectory of keats_pipeline/outputs/ML comparison and reasoning.')
    dest.mkdir(parents=True,exist_ok=False)
    settings=dict(created_sgt=now.isoformat(),only=args.only,universe=args.universe,bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,primary_block_months=6,block_sensitivities=[3,12],gamma=10,output=str(dest),no_retraining=True,
        automated_tests_requested=args.run_tests)
    (dest/'run_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print(f'Output: {dest}',flush=True)
    before=snapshot(ROOT);save(before,dest/'validation/source_hashes_before.csv')
    inv,raw,missing=discover(ROOT,None if args.universe=='all' else [args.universe])
    monthly,across,alignment=align(raw)
    save(inv,dest/'tables/source_inventory.csv');save(missing,dest/'tables/missing_saved_outputs.csv')
    save(alignment,dest/'tables/date_alignment_exclusions.csv');save(monthly,dest/'tables/aligned_monthly_saved_returns.csv')
    t2=table(monthly);yearly=annual(monthly);paths=paths_and_rolling(monthly)
    save(t2,dest/'tables/table2_consolidated.csv');save(ranks(t2),dest/'tables/model_rankings.csv')
    save(table(across),dest/'tables/table2_common_cross_dataset_calendar.csv')
    save(cross_dataset(table(across)),dest/'tables/cross_dataset_common_calendar.csv')
    save(yearly,dest/'tables/annual_performance.csv');save(paths,dest/'tables/rolling_and_cumulative.csv')
    save(performance_decomposition(t2),dest/'tables/return_cost_decomposition.csv')
    from .robustness import robustness_metric_dictionary
    save(pd.concat([metric_dictionary(),robustness_metric_dictionary()],ignore_index=True)
        .drop_duplicates('metric'),dest/'metric_dictionary.csv')
    print(f'Discovered {len(inv)} strategy/dataset series; {len(monthly)} aligned monthly observations.',flush=True)
    pairs=None
    if args.only in ['all','performance']:
        save(correlations(monthly,dest),dest/'tables/table4_correlations_long.csv')
        similar,skips=weight_similarity(inv,monthly)
        save(similar,dest/'tables/portfolio_weight_similarity.csv');save(skips,dest/'tables/weight_similarity_skips.csv')
        save(regimes(ROOT,monthly),dest/'tables/regime_performance.csv')
        charts(dest,monthly,t2,paths)
        print('Performance, correlations, regimes and charts complete.',flush=True)
    if args.only in ['all','inference']:
        pairs=compare_pairs(monthly,bootstrap_reps=args.bootstrap_reps,seed=args.seed)
        save(pairs,dest/'tables/table3_pairwise_tests.csv');matrices(pairs,dest)
        save(bootstrap_rank_stability(monthly,bootstrap_reps=args.bootstrap_reps,seed=args.seed),
            dest/'tables/bootstrap_ranking_stability.csv')
        sensitivities=[]
        for block in [3,12]:
            sensitivities.append(compare_pairs(monthly,bootstrap_reps=args.bootstrap_reps,block_length=block,seed=args.seed))
        save(pd.concat(sensitivities,ignore_index=True),dest/'tables/table3_block_length_sensitivity.csv')
        print(f'Paired inference complete: {len(pairs)} pairs, three bootstrap block lengths.',flush=True)
    if args.only in ['all','robustness']:
        from .robustness import run_robustness
        result=run_robustness(ROOT,dest,inv,monthly)
        print(f'Robustness complete: {result}',flush=True)
    if args.only in ['all','audit']:
        from .audit import run_audit
        result=run_audit(ROOT,dest/'audit',inv,monthly)
        print(f'Integrity audit complete: {result}',flush=True)
    independent_validation(ROOT,dest,monthly,t2,pairs)
    assumptions(dest,inv,missing,settings)
    if args.run_tests:
        # Short paths avoid Windows MAX_PATH failures in pytest fixture names.
        temp=output_root/f't_{uuid.uuid4().hex[:6]}'
        temp_argument=('\\\\?\\'+str(temp.resolve())) if sys.platform=='win32' else str(temp)
        command=[sys.executable,'-m','pytest',str(CODE_ROOT/'tests'),'keats_pipeline/tests',
            '-q','-p','no:cacheprovider','--basetemp',temp_argument]
        tested=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (dest/'validation/automated_tests.log').write_text(tested.stdout+'\n'+tested.stderr,encoding='utf-8')
        (dest/'validation/automated_tests_command.json').write_text(json.dumps(command,indent=2),encoding='utf-8')
        print(tested.stdout.strip(),flush=True)
        if tested.returncode:raise AssertionError('Automated tests failed; see validation/automated_tests.log')
    code_hashes=pd.DataFrame([dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(CODE_ROOT.rglob('*.py'))])
    save(code_hashes,dest/'validation/evaluation_code_hashes.csv')
    after=snapshot(ROOT);save(after,dest/'validation/source_hashes_after.csv')
    preservation=before.merge(after,on='path',how='outer',suffixes=('_before','_after'),indicator=True)
    preservation['preserved']=preservation['_merge'].eq('both')&preservation.sha256_before.eq(preservation.sha256_after)
    save(preservation,dest/'validation/source_preservation.csv')
    if not preservation.preserved.all():raise AssertionError('Existing source changed during read-only evaluation; inspect source_preservation.csv.')
    test_status='Automated tests passed; see automated_tests.log.' if args.run_tests else 'Automated tests not requested in this run; use --run-tests.'
    (dest/'validation/README.md').write_text(f'Independent metric checks passed. All {len(preservation)} original source files retained identical SHA256 hashes.\n{test_status}\nAudit FAIL/WARN/SKIPPED records describe input limitations and must also be reviewed.\n',encoding='utf-8')
    summary(dest,t2,pairs,yearly,inv,missing,settings)
    # New pointer per run preserves older run directories and pointers.
    (output_root/f'OPEN_{now:%Y%m%d_%H%M%S}.md').write_text(f'# Keats ML comparison\n\n[Open report]({dest.name}/index.html)\n\n[Research summary]({dest.name}/RESEARCH_SUMMARY.md)\n\n[Exclusions, assumptions and changes]({dest.name}/EXCLUSIONS_ASSUMPTIONS_AND_CHANGES.md)\n',encoding='utf-8')
    print(f'COMPLETE. Existing source files preserved: {len(preservation)}. Report: {dest / "index.html"}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
