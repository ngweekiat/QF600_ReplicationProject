"""Human-readable selection rationale, limitations and portable charts."""
from pathlib import Path
import html
import json
import re
import numpy as np
import pandas as pd

from .acquire import DATA,OUTPUT,KEATS,UNIVERSES,digest,write_json
from .paths import TRYOUT_ROOT


def write_report(selected,scores,validation,diagnostics,checks):
    labels={'us_large':'US large cap','us_mid':'US mid cap','eu_large':'European large cap'}
    screen=json.loads((OUTPUT/'screening/screening_decision.json').read_text())
    selected=selected.sort_values(['theme','final_score'],ascending=[True,False])
    themes=selected.groupby('theme').agg(features=('characteristic','count'),published_representatives=('published_jkp_153','sum'),
        minimum_validation_coverage=('minimum_validation_coverage','min')).reset_index()
    themes.to_csv(OUTPUT/'selected_theme_coverage.csv',index=False)
    display=selected[['characteristic','description','theme','theme_assignment_method','training_mean_rank_ic',
        'validation_oriented_ic','minimum_validation_coverage','mean_persistence','mean_walk_forward_r2','selection_reason']]
    display.to_csv(OUTPUT/'recommended_50_explained.csv',index=False)
    per_universe=validation.loc[validation.characteristic.isin(selected.characteristic)].merge(
        selected[['characteristic','training_direction','theme']],on='characteristic',validate='many_to_one')
    per_universe.to_csv(OUTPUT/'selected_50_validation_by_universe.csv',index=False)
    summary=['# Shared characteristics for the Keats ML strategies','',
        '## Recommendation','',
        f'Recommend one common set of **50 characteristics**, spanning **{selected.theme.nunique()} theme groups**, for a separately versioned Static-ML / Portfolio-ML experiment. Published features keep official clusters; other features carry explicitly inferred proxy labels. The existing models and their 50-feature inputs are unchanged.',
        '',f'The WRDS schema contains {screen["candidates"]} eligible stock-characteristic fields. The pre-2015 screen finds {screen["eligible"]} meeting the stated coverage/valid-month rules and retains {screen["retained"]} for full-history acquisition. The cap is a declared scoring-and-diversity screen, not a claim of the globally optimal 200.',
        '',f'The full capped files preserve {int(checks.download_rows.sum()):,} regional stock-months across {len(checks)} yearly partitions, including warmup and 2020-2025 values stored for later use. Selection analyses use only return months 2005-2019. Test return months 2020-2025 do not influence either shortlist.',
        '', '## Why this follows the paper in spirit','',
        'Section 5.1.2 of the supplied paper (printed page 3056; PDF page 22) uses **115** characteristics, after excluding poorly covered early features from the JKP research set of 153. It ranks monthly characteristics to [0,1] and uses 0.5 for missing signals. Section 3.3 defines economic feature importance through lost investor utility, not a raw correlation coefficient.',
        '', 'Our requested 50-feature limit is a computational/design simplification, not an exact replication of 115. We preserve broad economic representation, cross-market coverage, redundancy control and signal persistence. Predictive diagnostics alone cannot establish that these are the utility-optimal inputs for either ML portfolio model.',
        '', '## Selection method','',
        '1. Preserve the exact existing US and European regional training keys, historical portfolio memberships and observed next-month USD excess-return targets. Do not substitute newly downloaded or assumed-zero target values.',
        '2. Screen every candidate on all three portfolios\' held stock-months during return years 2005-2014. Each universe receives equal weight, irrespective of stock or row counts. At least 50% observed characteristic coverage in every universe and at least 60 valid monthly ICs are required.',
        '3. The first score combines predictive rank IC (55%), worst-universe coverage (20%), training-year sign stability (15%) and rank persistence (10%). Retain at most 200, with theme representation and a redundancy penalty. Published JKP membership receives a small 0.02 interpretability preference.',
        '4. Use returns 2015-2019 to assess validation IC, cross-universe directional consistency and simple annual walk-forward one-feature regressions. Forecast coefficients for year y use only return years before y. The regression diagnostic does not refit either existing ML model.',
        '5. The final score uses balanced validation IC (35%), weakest-universe oriented IC (15%), minimum validation coverage (15%), yearly direction stability (10%), persistence (10%) and average one-feature validation R-squared (15%). Require at least 60% coverage in every universe in both training and validation, at least 60 training IC months and 30 validation IC months.',
        '6. Choose 50 greedily, requiring at least one eligible published JKP representative in every theme, at least two total representatives per theme and no more than six per theme. Penalise correlation with already chosen inputs (0.20). This transparent rule trades marginal signal strength against duplication; it does not solve a guaranteed global optimum.',
        '', '## How to read the results','',
        '- **Rank IC** is monthly cross-sectional Spearman correlation with next-month observed excess returns. Signed validation IC uses the direction learned before 2015. Negative validation values indicate that the training direction did not carry forward.',
        '- **Pearson correlation and raw covariance** are also supplied, but raw covariance depends on units. It is not meaningful to rank market equity against a profitability ratio by raw covariance magnitude.',
        '- **Feature covariance/correlation** measures redundancy between characteristics. It is different from the saved stock-return risk covariance used by Static-ML and Portfolio-ML.',
        '- Pairwise missing-data feature covariance matrices need not be positive semidefinite and must not be substituted for the portfolio risk matrix. Individual cross-sectional correlations are not sector/country-neutral and may reflect common exposures.',
        '- **Walk-forward R-squared** compares a single-characteristic linear forecast against its past observed-return mean on the same available validation pairs. Negative values mean that diagnostic did worse than the historical-mean benchmark.',
        '- **HAC confidence intervals and BH false-discovery adjustments** describe individual IC evidence. They do not correct all model-selection optimism or make the selected 50 a fresh OOS result.',
        '- **Persistence and rank change** are turnover-related proxies, not actual trading costs or economic feature importance. Gross quintile spreads are uncosted descriptive diagnostics; they are not implementable portfolios or ML backtests.',
        '', '## Economic reading of the current evidence','',
        'Debt-to-market and net-debt-to-market measures rank highly in the declared balanced validation score. A high debt-to-price ratio combines capital structure with market valuation: its association may reflect cheapness, financial distress, country/industry exposure or some combination. It does not establish that adding debt causes higher returns. Profitability and quality measures supply different operating-performance information, while momentum, risk, seasonality and accrual signals broaden the inputs beyond one valuation family.',
        '', 'Individual forecast R-squared values are small, and many HAC intervals include zero. Some representatives are retained for theme coverage despite weaker or inconsistent standalone evidence. Correlated valuation measures still appear together: the redundancy penalty reduces duplication but does not impose an absolute correlation cutoff. Whether the joint set improves investor utility after risk, turnover and trading costs remains a question for a separately fitted Static-ML/Portfolio-ML experiment.',
        '', '## Evidence for each market','']
    for universe in UNIVERSES:
        sub=per_universe.loc[per_universe.universe.eq(universe)].copy()
        sub['oriented_ic']=sub.mean_rank_ic*sub.training_direction
        best=sub.sort_values('oriented_ic',ascending=False).head(5)
        summary.append(f'### {labels[universe]}')
        for row in best.itertuples():
            lo,hi=(row.ic_ci_low,row.ic_ci_high) if row.training_direction>0 else (-row.ic_ci_high,-row.ic_ci_low)
            summary.append(f'- {row.characteristic}: validation IC in the training direction {row.oriented_ic:.4f}; exploratory HAC 95% CI [{lo:.4f}, {hi:.4f}], coverage {100*row.coverage:.1f}%, BH-adjusted IC p={row.ic_hac_p_bh:.4g}.')
        summary.append('')
    summary.extend(['## Recommended 50',''])
    for theme,g in selected.groupby('theme'):
        summary.append(f'- **{theme} ({len(g)})**: '+', '.join(g.characteristic))
    summary.extend(['','## Assumptions limitations and changes','',
        f'- Existing Keats panels, memberships, returns, model fits and configurations are unchanged. Experimental code: `{TRYOUT_ROOT}`; selected data directory: `{DATA}`; analysis directory: `{OUTPUT}`.',
        f'- Candidate exclusions: {screen["candidates"]} admitted characteristics; {screen["candidates"]-screen["eligible"]} fail the initial coverage/valid-month gate; {screen["eligible"]-screen["retained"]} qualified candidates fall outside the 200 cap. Of the 200 fully acquired, {int((~scores.eligible).sum())} fail the final gate; {int(scores.eligible.sum())-50} qualified candidates are not retained in the final 50. The separate all-404 decision ledger records each decision.',
        '- Source-field exclusions: identifiers, metadata, observation flags, realised/forward returns and auxiliary fields are excluded from the predictor registry. All 444 source fields and their admission decisions are listed in data/source_field_admission_audit.csv; 404 are predictors and 40 are not.',
        '- Initial gate: at least 50% observed feature coverage and 60 valid monthly ICs in every universe during 2005-2014. Final gate: at least 60% coverage in every universe in both 2005-2014 and 2015-2019, plus 60 training and 30 validation IC months. Weaker standalone predictive evidence is not by itself an exclusion if a signal is retained for representation.',
        '- Return years before 2005 and from 2020 onward are excluded from all selection scores. Monthly IC needs at least 20 observed feature/return pairs; per-stock time-series diagnostics need 24 pairs; feature redundancy matrices need 120 pairs. Insufficient or constant-variable correlations stay undefined, rather than being reported as zero.',
        '- The initial exact-key SQL pilot stalled. Its completed chunks were retained; acquisition resumed with simpler yearly predicates, TCP keepalive and statement timeouts. A price supplement preserves the published `prc` predictor outside the contiguous source characteristic block.',
        '- Every source missing value stays missing in raw files. Analysis excludes assumed-zero return targets by using saved original observed targets; genuinely observed zero returns remain eligible. Pairwise feature/return availability counts, missing source keys and original membership boundaries are explicit. Historical portfolio membership is respected rather than filling nonmembership months into a held-stock sample.',
        '- Downloaded data are a current WRDS vintage, not proof of historically unrevised accounting values. Existing screened populations were originally formed using the old 50 inputs. This conditioning and prior inspection of old test results limit causal and pristine-OOS claims.',
        '- Published characteristic definitions and clusters are fixed external research metadata, not demonstrated to have been available in 2019. They may embed later research information. The no-test-use guarantee concerns direct use of Keats\' 2020-2025 return outcomes, not a claim of historically pristine feature discovery.',
        '- Univariate screening can miss features useful only through nonlinear interactions. No correlation-based shortlist can establish the globally best joint predictor or utility-optimal Portfolio-ML input set.',
        '- Published 153-theme labels are preserved. Extra characteristics receive an explicitly labelled empirical theme proxy from pre-2015 correlations with published groups; this is not an official JKP semantic classification.',
        '- Review found that the preliminary proxy-based selection left Accruals with no published representative. The final rule requires a published anchor in every group. Preliminary recommendation files and ranked exports are retained separately for an audit trail; they are not the final recommendation.',
        '- The final 50 is validation-selected. Its 2015-2019 metrics are selection evidence, not an unbiased estimate of future gains. No test performance claim or new ML-portfolio outperformance is made.',
        '- Separate ranked 50-feature panels cover all original dates and broad stocks. Missing characteristics are 0.5 only in these explicitly labelled ranked exports; raw observations and returns remain unchanged. Coverage flags are supplied. Existing ML settings have not been activated.',
        '- Portfolio-ML supplemental lag keys are acquired separately under `data/lag_history`. They are input history only, never additional return-target samples. Unavailable source lag observations retain explicit missing-row flags. They require consistent lag preparation during a future refit.',
        '- Three original supplemental lag keys remain unavailable: US IDs 88818 and 91135 and European ID 320230501, all at formation month 2010-12-31. Their characteristic values are not fabricated. All 2,886,029 main regional stock-month keys were found in the source.',
        '- HAC inference preserves missing ICs in their calendar positions. Final reconciliation recomputes summaries from saved monthly observations and independently reproduces the frozen recommendation. These uncertainty estimates remain exploratory after feature selection.',
        '- Using this list in Static-ML/Portfolio-ML requires an isolated future preparation/refit, correct theme-direction handling (especially nonpublished inputs), and regenerated consistent risk inputs. No such model refit is performed here.',
        '', '## Files to open','',
        '- `recommended_50_explained.csv`: shared recommendation, names, theme provenance and selection evidence.',
        '- `selected_50_validation_by_universe.csv`: individual-market ICs, HAC intervals, coverage and persistence.',
        '- `all_capped_characteristic_scores.csv` and `screening/all_candidate_training_scores.csv`: complete capped/all-source rankings and exclusions.',
        '- `all_404_selection_and_exclusion_ledger.csv`: one explicit stage decision and reason for every admitted characteristic.',
        '- `ASSUMPTIONS_EXCLUSIONS_AND_CHANGES.md`: consolidated assumptions, exclusions and changes.',
        '- `*_feature_rank_correlation.csv`, `*_feature_raw_covariance.csv`, and `selected_feature_redundancy.csv`: redundancy and covariance evidence.',
        '- `per_stock_per_characteristic.parquet`: time-series associations for every eligible held stock/characteristic, not causal prediction claims.',
        '- `broad_regional_characteristic_summary.csv`: separate training-population diagnostics.',
        '- `data/candidate_registry.csv`, `download_manifest.csv` and yearly `panels/`: raw data, definitions, coverage and SHA256 provenance.',
        '', 'Sources: supplied Jensen et al. research paper; WRDS contrib.global_factor; JKP source definitions and [official WRDS guide](https://jkpfactors.com/jkp-wrds-guide).'])
    text='\n'.join(summary)+'\n';(OUTPUT/'RESEARCH_SUMMARY.md').write_text(text,encoding='utf-8')
    methods=text.split('## Assumptions limitations and changes',1)[1].split('## Files to open',1)[0]
    (OUTPUT/'ASSUMPTIONS_EXCLUSIONS_AND_CHANGES.md').write_text('# Assumptions exclusions and changes\n'+methods,encoding='utf-8')
    body=[]
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
        value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
        return re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',value)
    listing=False
    for line in summary:
        if line.startswith('- '):
            if not listing:body.append('<ul>');listing=True
            body.append('<li>'+inline(line[2:])+'</li>');continue
        if listing:body.append('</ul>');listing=False
        if line.startswith('### '):body.append('<h3>'+inline(line[4:])+'</h3>')
        elif line.startswith('## '):body.append('<h2>'+inline(line[3:])+'</h2>')
        elif line.startswith('# '):body.append('<h1>'+inline(line[2:])+'</h1>')
        elif line:body.append('<p>'+inline(line)+'</p>')
    if listing:body.append('</ul>')
    caption='<p>Shared candidate set for all three Keats universes. IC values are correlations; coverage is a fraction. Selection cutoff is return month December 2019.</p>'
    (OUTPUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Keats characteristic selection</title><style>body{max-width:1300px;margin:40px auto;padding:0 20px;font:15px/1.5 system-ui;color:#17232c}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #ccd6dd}th{background:#e7eff5}h1,h2{color:#163f52}li{margin:8px 0}code{background:#f1f4f7;padding:2px 4px}.table-scroll{overflow-x:auto}</style>'+''.join(body)+caption+'<div class="table-scroll">'+display.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'</div>',encoding='utf-8')
    charts(selected,per_universe)
    write_json(OUTPUT/'completion.json',dict(status='complete',recommended=50,downloaded=screen['retained'],
        source_candidates=screen['candidates'],themes=selected.theme.nunique(),original_data_unchanged=True,
        test_returns_used=False,models_retrained=False,source_stock_months=int(checks.download_rows.sum())))


def charts(selected,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUTPUT/'charts';folder.mkdir(exist_ok=True)
    chars=selected.sort_values('final_score',ascending=False).characteristic.tolist()
    corr=pd.read_csv(OUTPUT/'balanced_pretest_rank_correlation.csv',index_col=0).loc[chars,chars]
    plt.rcParams.update({'font.size':9,'savefig.dpi':240})
    fig,ax=plt.subplots(figsize=(14,12))
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(50),chars,rotation=90,fontsize=6);ax.set_yticks(range(50),chars,fontsize=6)
    ax.set_title('Recommended 50 characteristic rank correlations | 2005-2019\nEqual weight to US large cap, US mid cap and European large cap; diagnostic, not a stock risk matrix')
    fig.colorbar(im,ax=ax,label='Average pairwise rank correlation')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'selected_50_correlations.{ext}',bbox_inches='tight')
    plt.close(fig)
    top=chars[:15];fig,ax=plt.subplots(figsize=(10,8))
    for i,u in enumerate(UNIVERSES):
        g=validation.loc[validation.universe.eq(u)].set_index('characteristic').loc[top]
        direction=g.training_direction.to_numpy()
        mean=g.mean_rank_ic.to_numpy()*direction
        low=np.where(direction>0,g.ic_ci_low.to_numpy(),-g.ic_ci_high.to_numpy())
        high=np.where(direction>0,g.ic_ci_high.to_numpy(),-g.ic_ci_low.to_numpy())
        y=np.arange(len(top))+(i-1)*.22
        ax.errorbar(mean,y,xerr=np.vstack([mean-low,high-mean]),fmt='o',markersize=3,label=u,alpha=.8)
    ax.set_yticks(range(len(top)),top);ax.invert_yaxis();ax.axvline(0,color='grey',lw=.7)
    ax.set(xlabel='Validation rank IC in pre-2015 training direction',title='Top 15 recommended characteristics | validation return years 2015-2019')
    ax.legend(fontsize=8);fig.text(.08,-.015,'Exploratory HAC 95% intervals, not selection-adjusted. Final feature set was selected using this validation period.',fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'validation_ic_intervals.{ext}',bbox_inches='tight')
    plt.close(fig)
