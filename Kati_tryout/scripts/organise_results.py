"""Index saved base/iteration outputs without fitting or completing experiments."""
from pathlib import Path
import hashlib
import html
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SUBSTAGES=('iteration_4.2','iteration_4.3')
GROUPS=('base',*(f'iteration_{i}' for i in range(1,5)),*SUBSTAGES,*(f'iteration_{i}' for i in range(5,11)))
MARKETS=('us_large','us_mid','eu_large')
DESCRIPTIONS={
    'base':'US high-cap Static ML* and default Portfolio ML in all three markets; overall analysis adds Market references.',
    'iteration_1':'Annual refitting, already in Base; retained as design, P(better) N/A.',
    'iteration_2':'Gamma 5 against the previous accepted version.',
    'iteration_3':'Exit rank 120 and costs x2, inheriting accepted changes.',
    'iteration_4':'Market plus pretest-calibrated 1% TE-target ML tilt.',
    'iteration_4.2':'Execution-based active-utility validation on the existing market anchor; P(test) is descriptive, without an 80% retention gate.',
    'iteration_4.3':'Uncertainty-aware selection using executed active-utility validation flows; inherits research stage 4.2, with descriptive P(test).',
    'iteration_5':'Recent-history weighting with 3/5/10-year half-lives and equal weighting.',
    'iteration_6':'More selective trading: pretest-selected training cost penalty; actual costs unchanged.',
    'iteration_7':'Stock-specific risk responsiveness: pretest-selected 63/126/252-day idiosyncratic volatility half-life.',
    'iteration_8':'Stable one-SE model selection: simpler candidates within one historical block-bootstrap SE.',
    'iteration_9':'Benchmark-aware trade netting: affine recursive cost learning; unchanged actual accounting.',
    'iteration_10':'RFF seed ensemble: equal target tilts from five fixed seeds, executed once with original costs.'}


def markets_for_group(group):
    """Only publish current scope; never recreate removed historical folders."""
    return MARKETS if group=='base' else ('us_large',)


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def active_label(path,root):
    calibration=root/'active/calibrated.csv'
    if calibration.exists():
        for row in pd.read_csv(calibration).itertuples():
            if f'gte{row.gamma_te:g}'==path.parent.name:return f'Portfolio-ML, TE {row.te_target:.0%}'
    return f'Portfolio-ML, {path.parent.name}'


def discover(group,market):
    root=ROOT/'outputs'/group/market
    files=[]
    if group=='base':
        files=([(root/'static_ml_star/pf.parquet','experiment',None)] if market=='us_large' else [])+[(root/'portfolio_ml/rff/pf.parquet','experiment',None)]
    else:
        files=[(root/'pf.parquet','experiment',f'Portfolio ML iteration {group.split("_")[1]}')]
        base=ROOT/'outputs/base/us_large'
        files += [(base/'portfolio_ml/rff/pf.parquet','reference','Base Portfolio ML'),
                  (base/'static_ml_star/pf.parquet','reference','Static ML*'),
                  (ROOT/'outputs/base/overall analysis/market_references/us_large/pf.parquet','reference','Market')]
        if group in SUBSTAGES:
            files.append((ROOT/'outputs/iteration_4/us_large/pf.parquet','reference','Iteration 4: market anchor'))
            if group=='iteration_4.3':
                files.append((ROOT/'outputs/iteration_4.2/us_large/pf.parquet','reference','Iteration 4.2: executed validation'))
    frames=[];inventory=[]
    for path,role,label in files:
        if not path.exists():continue
        data=pd.read_parquet(path)
        for name,d in data.groupby('type',sort=False):
            if group=='base':
                from jkmp.strategies import base_strategies
                if name not in base_strategies(market):raise ValueError('Other method in base')
            d=d.copy();d['strategy']=label or name;d['role']=role;d['group']=group;d['market']=market
            frames.append(d)
            inventory.append(dict(group=group,market=market,strategy=label or name,role=role,
                file=str(path.relative_to(ROOT/'outputs')),sha256=sha256(path),months=len(d)))
    columns=['type','eom','eom_ret','r','tc','turnover','inv','shorting','strategy','role','group','market']
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)),inventory


def stats(data,market_root):
    rows=[];benchmark=None
    path=market_root/'benchmarks/pf_test.parquet'
    if path.exists():benchmark=pd.read_parquet(path).query("type == 'Market'").set_index('eom_ret')
    for (strategy,role),d in data.groupby(['strategy','role'],sort=False):
        r=d.r.to_numpy(float);tc=d.tc.to_numpy(float);sd=np.std(r,ddof=1)
        row=dict(strategy=strategy,role=role,months=len(d),first_return=str(pd.Timestamp(d.eom_ret.min()).date()),
            last_return=str(pd.Timestamp(d.eom_ret.max()).date()),annual_gross_return=12*r.mean(),annual_cost=12*tc.mean(),
            annual_net_return=12*(r-tc).mean(),annual_gross_volatility=np.sqrt(12)*sd,
            net_sharpe_paper=np.sqrt(12)*(r-tc).mean()/sd if sd>0 else np.nan,
            mean_leverage=d.inv.mean(),mean_monthly_turnover=d.turnover.mean())
        if benchmark is not None:
            b=benchmark.reindex(d.eom_ret)
            if b[['r','tc']].notna().all().all():
                active=r-b.r.to_numpy();active_net=(r-tc)-(b.r-b.tc).to_numpy();te=active.std(ddof=1)*np.sqrt(12)
                row.update(annual_active_net=12*active_net.mean(),tracking_error_realised=te,
                    information_ratio_net=12*active_net.mean()/te if te>0 else np.nan,
                    benchmark_cost_convention='net strategy minus net same-universe Market')
        rows.append(row)
    return pd.DataFrame(rows)


def plot(data,destination,title):
    if data.empty:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for label,d in data.groupby('strategy'):
        d=d.sort_values('eom_ret')
        axes[0].plot(d.eom_ret,100*d.r.cumsum(),label=label)
        axes[1].plot(d.eom_ret,100*d.tc.cumsum(),label=label)
    axes[0].set(title='Cumulative gross excess return (arithmetic)',ylabel='Percentage points')
    axes[1].set(title='Cumulative modelled trading cost',ylabel='Percentage points')
    if data.tc.max()>1:axes[1].set_yscale('symlog',linthresh=1);axes[1].set_title('Cumulative cost (symlog)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(title);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(destination.with_suffix('.'+suffix),dpi=170)
    plt.close(fig)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only-group',choices=GROUPS,default=None)
    args=parser.parse_args(argv)
    availability=[];inventory=[]
    style='<style>body{max-width:1300px;margin:36px auto;padding:0 20px;font:15px/1.5 system-ui;color:#21323e}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ced8df;padding:8px}th{background:#e8eff4}.scroll{overflow-x:auto}img{width:100%}</style>'
    for group in ([args.only_group] if args.only_group else GROUPS):
        group_root=ROOT/'outputs'/group;group_root.mkdir(exist_ok=True)
        for market in markets_for_group(group):
            root=group_root/market;reports=root/'reports';reports.mkdir(parents=True,exist_ok=True)
            data,files=discover(group,market);inventory.extend(files)
            experiment=data.loc[data.role.eq('experiment')];reference=data.loc[data.role.ne('experiment')]
            if group=='base':status='available' if experiment.strategy.nunique()==(2 if market=='us_large' else 1) else 'not run' if experiment.empty else 'partial'
            elif not experiment.empty:status='saved experiment results available'
            elif not reference.empty:status='preparatory/reference outputs only; experiment unfinished'
            elif any(p.is_file() for p in root.rglob('*') if 'reports' not in p.relative_to(root).parts and p.name not in ('status.json','README.md','index.html')):status='input audits/references only; experiment unfinished'
            else:status='not run; no saved experiment performance'
            record=dict(group=group,market=market,status=status,experiment_series=int(experiment.strategy.nunique()),
                reference_series=int(reference.strategy.nunique()),new_experiments_run=False,
                scheduled_in_current_scope=group=='base' or market=='us_large',
                part_of_integer_acceptance_sequence=group.startswith('iteration_') and group not in SUBSTAGES,
                p_test_is_retention_gate=group not in ('base','iteration_1',*SUBSTAGES))
            if group!='base' and (root/'decision.json').exists():
                decision=json.loads((root/'decision.json').read_text())
                record.update(verdict=decision['verdict'],p_better=decision['p_better'],
                    included=decision['accepted'],inherited_iterations=decision['inherited_iterations'])
            availability.append(record)
            (root/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            table=stats(data,root)
            if not data.empty:
                data.to_csv(reports/'available_monthly_returns.csv',index=False)
                table.to_csv(reports/'available_performance.csv',index=False)
            plot(experiment,reports/'charts/experiment_returns_and_costs',f'{group} | {market} | saved experiment results')
            plot(reference,reports/'charts/reference_returns_and_costs',f'{group} | {market} | reference/preparatory results')
            intro=f'<h1>{group}: {market}</h1><p>{html.escape(DESCRIPTIONS[group])}</p><p><strong>Status: {html.escape(status)}</strong></p>'
            text='Only freshly fitted Kati outputs are indexed here. This reporting step does not fit models. Test returns are 2020-2025; characteristic selection used US high-cap 2005-2014 only. Iterations apply only to US high-cap Portfolio ML. Reference/preparatory statistics are not a completed iteration result.'
            body=intro+'<p>'+text+'</p>'
            saved_decision=json.loads((root/'decision.json').read_text()) if (root/'decision.json').exists() else {}
            historical=group=='iteration_2' or (group not in ('base','iteration_1','iteration_4',*SUBSTAGES) and 2 in saved_decision.get('inherited_iterations',[]))
            if historical:
                body+='<p><strong>Historical trial:</strong> not part of the revised Base → 4 → 4.2 → 4.3 chain. Its saved ancestry and P(test) retain the original reference; this trial was not rebased or rerun.</p>'
            if group=='iteration_4' and (root/'charts/pretest_rationale.png').exists():
                body+='<h2>Pretest-only rationale: Base to Market anchor</h2><img src="charts/pretest_rationale.png"><p><a href="pretest_diagnosis/statistics.json">2016-2019 statistics</a> | <a href="pretest_diagnosis/METHODS.md">Methods and limits</a></p>'
            diagnostic=ROOT/'outputs/base/us_large/diagnostic_for_gamma5'
            if market=='us_large' and group in ('base','iteration_2') and (diagnostic/'base_rationale_for_iteration_2.png').exists():
                prefix='diagnostic_for_gamma5/' if group=='base' else '../../base/us_large/diagnostic_for_gamma5/'
                body+='<h2>Base diagnosis: why test lower risk aversion in iteration 2?</h2><p>Only Base and Market data are used; no iteration-2 outcomes. Lower gamma is a preference experiment, not a proven improvement.</p>'
                body+=f'<img src="{prefix}base_rationale_for_iteration_2.png"><p><a href="{prefix}base_rationale_for_iteration_2.pdf">Download PDF</a> | <a href="{prefix}statistics.json">Statistics</a> | <a href="{prefix}METHODS.md">Methods and limitations</a> | <a href="{prefix}verification.json">Independent checks</a></p>'
            if group!='base' and (root/'decision.json').exists():
                decision=json.loads((root/'decision.json').read_text())
                probability=decision['p_better']
                ptext=('N/A (no distinct comparison)' if group in SUBSTAGES else 'N/A (already in Base)') if probability is None else f'{probability:.2%}'
                body+=f'<p><b>{html.escape(decision["verdict"])}</b>; P(better) = {ptext}. Inherited iterations: {decision["inherited_iterations"]}. Included after this decision: {decision["included_after"]}.</p><p><a href="SUMMARY.md">Inheritance and decision summary</a> | <a href="../../iteration_summary/index.html">Original ten-experiment sequence and final comparison</a></p>'
                if group in SUBSTAGES:
                    body+='<p class="note"><b>Separate iteration-4 research branch.</b> P(test) is reported and explained, but has no 80% retention gate for these stages. Retaining a research stage is not a declaration of superior performance. These outputs do not change the original integer-sequence accepted strategy.</p><p><a href="../../../docs/ITERATION_4_SUBSTAGES.md">Substage scope and interpretation</a></p>'
                    if (root/'active_performance.csv').exists():
                        body+='<p><a href="active_performance.csv">Benchmark-relative active performance CSV</a> | <a href="../../iteration_4_substages_summary/index.html">Joint iteration 4 / 4.2 / 4.3 economic comparison</a></p>'
                    if (ROOT/'outputs/iteration_4_substages_summary/charts/result_comparison.png').exists():
                        body+='<h2>Executed research-stage results</h2><img src="../../iteration_4_substages_summary/charts/result_comparison.png"><p>Result figures are separate from the pretest-only rationale. Standard absolute utility and benchmark-relative active utility use different risk parameters and objectives.</p>'
                    if (root/'charts/rationale.png').exists():
                        body+='<h2>Statistical and economic rationale</h2><img src="charts/rationale.png"><p>The rationale must be read with its stated data window and limitations; it is not evidence that the subsequent stage must improve.</p>'
                body+='<img src="charts/net_returns_and_costs.png" alt="Candidate, previous accepted version, Base, Static ML star and Market: returns, drawdowns and relative performance"><p><a href="performance_summary.csv">Detailed performance CSV</a> | <a href="monthly_returns.csv">Monthly series</a> | <a href="decision.json">Decision and exact inherited settings</a></p>'
                if group=='iteration_6':
                    body+='<h2>Pretest selection: learning-only cost preference</h2><p>2016-2019 validation selects one penalty before the test comparison. Actual execution costs remain unchanged.</p><img src="charts/pretest_cost_preference.png"><p><a href="cost_preference_validation.csv">Every pretest trial</a> | <a href="pretest_selection.json">Frozen selection and calibrated risk coefficient</a></p><img src="charts/market_comparison_panel.png">'
                if group=='iteration_7':
                    body+='<h2>Pretest selection: stock-specific risk responsiveness</h2><p>Only the idiosyncratic volatility half-life changes. All factor risks, stocks and cost formulas are fixed.</p><img src="charts/pretest_risk_half_life.png"><p><a href="risk_half_life_validation.csv">Every pretest trial</a> | <a href="pretest_selection.json">Frozen selection</a> | <a href="risk_control_verification.json">Original 126-day control and covariance isolation checks</a></p><img src="charts/market_comparison_panel.png">'
                if group=='iteration_8':
                    body+='<h2>Annual robustness selection</h2><p>Fewer RFF features and stronger ridge are preferred within one historical block-bootstrap standard error of the best utility. Only prior returns enter each choice.</p><img src="charts/annual_selection_robustness.png"><p><a href="selection_audit.csv">Every annual choice, SE and threshold</a> | <a href="pretest_selection.json">Frozen rule and pretest TE calibration</a></p><img src="charts/market_comparison_panel.png">'
                if group=='iteration_9':
                    body+='<h2>Benchmark-aware trade netting</h2><p>Only coefficient-learning costs change. The recursive benchmark and active order responses are learned together; original returns, risk proxies and actual stock accounting stay unchanged. The affine cost identity is exact for fixed coefficients, with annual coefficient jumps still charged through actual accounting.</p><p><a href="cost_audit.csv">Every historical cost-identity check</a> | <a href="pretest_selection.json">Pretest-only TE calibration and scope</a> | <a href="../../../docs/ITERATION_9_BENCHMARK_NETTING.md">Derivation and limitations</a></p><img src="charts/market_comparison_panel.png">'
                if group=='iteration_10':
                    body+='<h2>Fixed five-seed ensemble</h2><p>Seeds 1-5 receive equal weights. Every member uses the original annual validation rule and the same calibrated active-risk coefficient. Average stock aims, then execute once.</p><img src="pretest_seed_sensitivity/charts/pretest_seed_sensitivity.png"><p><a href="pretest_seed_sensitivity/diagnostic.json">Pretest sensitivity and original-seed control</a> | <a href="cost_netting_audit.csv">Combined-order cost identity</a> | <a href="pretest_selection.json">Frozen seeds and pretest calibration</a> | <a href="../../../docs/ITERATION_10_SEED_ENSEMBLE.md">Protocol</a></p><img src="charts/market_comparison_panel.png">'
                if group=='iteration_2' and (root/'diagnostic_for_market_anchor/iteration_2_gap_diagnostic.png').exists():
                    body+='<h2>Diagnostic: rationale for a market anchor</h2><p>Uses only iteration 2 and Market; no later model results.</p><img src="diagnostic_for_market_anchor/iteration_2_gap_diagnostic.png"><p><a href="diagnostic_for_market_anchor/statistics.json">Regression, HAC and bootstrap statistics</a> | <a href="diagnostic_for_market_anchor/METHODS.md">Methods and limits</a></p>'
                if group=='iteration_4' and (root/'diagnostic_for_trade_netting/iteration_4_next_iteration_diagnostic.png').exists():
                    body+='<h2>Diagnostic: rationale for benchmark-aware trade netting</h2><p>Uses only iteration 4 and Market; no iteration-9 outcomes enter this diagnostic.</p><img src="diagnostic_for_trade_netting/iteration_4_next_iteration_diagnostic.png"><p><a href="diagnostic_for_trade_netting/statistics.json">Stock-cost decomposition and inference</a> | <a href="diagnostic_for_trade_netting/METHODS.md">Methods, source audit and proposal limits</a></p>'
                if group=='iteration_4' and (root/'diagnostic_for_neutrality/neutrality_stage_diagnostic.png').exists():
                    body+='<h2>Diagnose beta and sector neutrality after the market anchor</h2><p>Separates pretest and inspected test data, beta exposure, stock/cash budget and sector contrasts. Economic projections are diagnostic replays, not fitted new iterations or acceptance decisions.</p><img src="diagnostic_for_neutrality/neutrality_stage_diagnostic.png"><p><a href="diagnostic_for_neutrality/neutrality_stage_diagnostic.pdf">Download PDF</a> | <a href="diagnostic_for_neutrality/statistics.json">Risk and economic statistics</a> | <a href="diagnostic_for_neutrality/block_bootstrap_sensitivity.csv">3/6/12-month bootstrap sensitivity</a> | <a href="diagnostic_for_neutrality/METHODS.md">Methods and limitations</a> | <a href="diagnostic_for_neutrality/verification.json">Independent checks and source hashes</a></p>'
                if group=='iteration_4':
                    links=''.join(f'<li><a href="../../{stage}/us_large/index.html">{stage}: {html.escape(DESCRIPTIONS[stage])}</a></li>' for stage in SUBSTAGES if (ROOT/'outputs'/stage/'us_large/index.html').exists())
                    if links:body+='<h2>Deeper validation substages</h2><p>Separate research stages with descriptive P(test), outside the original 80%-gate sequence.</p><ul>'+links+'</ul>'
            if not table.empty:body+='<h2>Available saved performance</h2><p>Returns and costs are annual decimal fractions; leverage and turnover are ratios.</p><div class="scroll">'+table.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</div>'
            if (root/'active/summary.csv').exists():body+='<p><a href="active/summary.csv">Original iteration summary (unchanged)</a></p>'
            for role in ('experiment','reference'):
                image=f'reports/charts/{role}_returns_and_costs.png'
                if (root/image).exists():body+=f'<img src="{image}" alt="{role} results">'
            (root/'index.html').write_text('<!doctype html><meta charset="utf-8">'+style+body,encoding='utf-8')
            (root/'README.md').write_text(f'# {group}: {market}\n\n{DESCRIPTIONS[group]}\n\n**Status: {status}.**\n\n{text}\n\nOpen index.html. CSV, Parquet and chart files remain in their respective run folders. Newly generated descriptive summaries/charts are under reports/.\n',encoding='utf-8')
            if group!='base' and (root/'SUMMARY.md').exists():
                summary=(root/'SUMMARY.md').read_text(encoding='utf-8')
                (root/'README.md').write_text(summary,encoding='utf-8')
                (group_root/'SUMMARY.md').write_text(summary+'\n\nDetailed results: us_large/index.html\n',encoding='utf-8')
        rows=pd.DataFrame(availability).query('group == @group')
        rows.to_csv(group_root/'availability.csv',index=False)
        links=''.join(f'<li><a href="{m}/index.html">{m}</a></li>' for m in markets_for_group(group))
        # Base keeps its richer validated two-method comparison as index.html.
        if group!='base' or not (group_root/'index.html').exists():
            (group_root/'index.html').write_text('<!doctype html><meta charset="utf-8">'+style+f'<h1>{group}</h1><p>{DESCRIPTIONS[group]}</p>'+rows.to_html(index=False)+f'<ul>{links}</ul><p>Only completed Kati results are shown; experiments without new-feature outputs remain unfinished.</p>',encoding='utf-8')
    out=ROOT/'outputs'
    if args.only_group:
        previous=out/'availability_by_group_and_market.csv'
        if previous.exists():availability += pd.read_csv(previous).query('group != @args.only_group').to_dict('records')
        previous_inventory=out/'organisation_audit/result_source_inventory.csv'
        if previous_inventory.exists():inventory += pd.read_csv(previous_inventory).query('group != @args.only_group').to_dict('records')
    pd.DataFrame(availability).to_csv(out/'availability_by_group_and_market.csv',index=False)
    (out/'organisation_audit').mkdir(exist_ok=True)
    pd.DataFrame(inventory).to_csv(out/'organisation_audit/result_source_inventory.csv',index=False)
    links='<li><a href="us_high_cap_selection/index.html">US high-cap 2005–2014 characteristic selection, correlation and covariance</a></li><li><a href="dashboard.html">Interactive results explorer</a></li>'+''.join(f'<li><a href="{g}/index.html">{g}</a>: {DESCRIPTIONS[g]}</li>' for g in GROUPS)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8">'+style+'<h1>Kati results by run group</h1><ul>'+links+'</ul>'+pd.DataFrame(availability).to_html(index=False)+'<p>Base contains US high-cap Static ML* and regional Portfolio ML. Iteration scope follows Keats; results require fresh fits with the new 50. Open dashboard.html for interactive comparisons.</p>',encoding='utf-8')
    print(pd.DataFrame(availability).to_string(index=False))
    from comparison_analysis import add_navigation
    add_navigation(include_iterations=args.only_group!="base")
    return 0


if __name__=='__main__':raise SystemExit(main())
