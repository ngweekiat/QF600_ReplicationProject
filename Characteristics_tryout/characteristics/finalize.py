"""Rebuild summaries and independently reconcile the completed frozen run."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .acquire import DATA,OUTPUT,KEATS,UNIVERSES,EXCLUDED_FIELDS,digest,write_json
from .statistics import summarize
from .analysis import selected_input_panels,final_scores
from .selection import diverse_pick
from .report import write_report
from .paths import TRYOUT_ROOT,relative_link


def finalize():
    features=json.loads((DATA/'download_200.json').read_text())
    chosen=pd.read_csv(OUTPUT/'recommended_50.csv')
    monthly=pd.read_parquet(OUTPUT/'all_universe_monthly_metrics.parquet')
    train=summarize(monthly.loc[monthly.return_month.dt.year.le(2014)])
    val=summarize(monthly.loc[monthly.return_month.dt.year.ge(2015)])
    train.to_csv(OUTPUT/'training_characteristic_results.csv',index=False)
    val.to_csv(OUTPUT/'validation_characteristic_results.csv',index=False)
    summarize(pd.read_parquet(OUTPUT/'broad_regional_monthly_diagnostics.parquet')).to_csv(
        OUTPUT/'broad_regional_characteristic_summary.csv',index=False)
    summarize(pd.read_parquet(OUTPUT/'screening/all_candidate_monthly_metrics.parquet')).to_csv(
        OUTPUT/'screening/all_candidate_universe_summary.csv',index=False)
    diag=pd.read_csv(OUTPUT/'univariate_walk_forward_validation.csv')
    registry=pd.read_csv(DATA/'candidate_registry_with_themes.csv').loc[lambda x:x.characteristic.isin(features)]
    screen=pd.read_csv(OUTPUT/'screening/all_candidate_training_scores.csv')
    scores=final_scores(train,val,monthly,diag,registry,screen)
    corr=pd.read_csv(OUTPUT/'balanced_pretest_rank_correlation.csv',index_col=0)
    reproduced=diverse_pick(scores,corr,50,min_theme=2,max_theme=6,score_col='final_score',redundancy_penalty=.20,published_anchor=True)
    if reproduced.characteristic.tolist()!=chosen.characteristic.tolist():
        raise AssertionError('Frozen recommendation did not reproduce; files retained for investigation')
    scores.sort_values('final_score',ascending=False).to_csv(OUTPUT/'all_capped_characteristic_scores.csv',index=False)
    ledger=screen.copy()
    ledger['downloaded_full_history']=ledger.characteristic.isin(features)
    ledger['recommended_50']=ledger.characteristic.isin(chosen.characteristic)
    final_eligible=scores.set_index('characteristic').eligible
    ledger['final_eligible']=ledger.characteristic.map(final_eligible)
    def decision(row):
        if row.recommended_50:return 'Recommended: passed coverage gates and balanced greedy selection'
        if not row.eligible:return 'Excluded at initial screen: below 50% coverage in at least one universe or fewer than 60 valid IC months'
        if not row.downloaded_full_history:return 'Not retained in 200 cap: initial balanced score, theme representation and redundancy rule'
        if not row.final_eligible:return 'Excluded at final gate: below 60% training/validation coverage, 60 training IC months or 30 validation IC months in at least one universe'
        return 'Not retained in final 50: balanced validation score, theme quotas and redundancy rule'
    ledger['decision_reason']=ledger.apply(decision,axis=1)
    ledger.to_csv(OUTPUT/'all_404_selection_and_exclusion_ledger.csv',index=False)
    audit=[]
    def check(name,passed,detail):
        audit.append(dict(check=name,status='PASS' if passed else 'FAIL',detail=detail))
    check('independently_reproduced_shared_50',True,'Fresh summaries and saved monthly metrics reproduce the ordered recommendation exactly')
    check('selection_return_dates',monthly.return_month.between('2005-01-31','2019-12-31').all(),'No return outcome from 2020 onward used')
    counts=chosen.groupby('theme').size()
    check('theme_representation',len(counts)==13 and counts.between(2,6).all(),counts.to_json())
    anchors=chosen.groupby('theme').published_jkp_153.sum()
    check('published_theme_anchors',anchors.ge(1).all(),'At least one eligible published JKP characteristic in each of the 13 groups')
    check('predictor_only',not set(features)&EXCLUDED_FIELDS,'No source return or identifier admitted as a characteristic')
    check('all_selected_eligible',chosen.eligible.all(),'Training and validation coverage/valid-month gates met')
    manifest=pd.read_csv(DATA/'download_manifest.csv')
    for region in ('US','EU'):
        meta=pd.read_parquet(DATA/f'metadata_{region}.parquet',columns=['id','eom'])
        for year,g in meta.groupby(meta.eom.dt.year):
            path=DATA/f'panels/{region}_{year}.parquet'
            d=pd.read_parquet(path,columns=['id','eom','source_row_found',*features])
            left=pd.MultiIndex.from_frame(g[['id','eom']]);right=pd.MultiIndex.from_frame(d[['id','eom']])
            same=len(left)==len(right) and right.is_unique and left.difference(right).empty and right.difference(left).empty
            source_record=manifest.loc[(manifest.region==region)&(manifest.year==year)].iloc[0]
            check(f'exact_keys_and_source_hash_{region}_{year}',same and digest(path)==source_record.sha256,
                  f'{len(d):,} original keys; {int((~d.source_row_found).sum())} missing source rows')
            a=d[features].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float,na_value=np.nan)
            check(f'no_infinite_source_values_{region}_{year}',not np.isinf(a).any(),'Raw NaNs remain missing; no infinite inputs')
    for universe in UNIVERSES:
        for kind in ('rank_correlation','rank_covariance','raw_correlation','raw_covariance','pair_counts'):
            a=pd.read_csv(OUTPUT/f'{universe}_feature_{kind}.csv',index_col=0).to_numpy(float)
            check(f'symmetric_{universe}_{kind}',np.allclose(a,a.T,equal_nan=True,rtol=1e-10,atol=1e-12),'Pairwise-complete diagnostic matrix')
    for row in pd.read_csv(DATA/'original_panel_inventory.csv').itertuples():
        check(f'original_panel_preserved_{row.universe}',digest(KEATS/row.path)==row.sha256,'Original SHA256 unchanged')
    lag_manifest=pd.read_csv(DATA/'lag_history_manifest.csv')
    for row in lag_manifest.itertuples():
        d=pd.read_parquet(DATA/row.path)
        original=pd.read_parquet(KEATS/f'data/inputs/lag_history_source_{row.region}.parquet',columns=['id','eom'])
        missing=pd.read_csv(KEATS/f'data/inputs/unavailable_lag_history_{row.region}.csv',parse_dates=['eom'])
        requested=pd.concat([original,missing],ignore_index=True).drop_duplicates(['id','eom'])
        left=pd.MultiIndex.from_frame(requested);right=pd.MultiIndex.from_frame(d[['id','eom']])
        same=len(left)==len(right) and right.is_unique and left.difference(right).empty and right.difference(left).empty
        check(f'supplemental_lag_keys_and_hash_{row.region}',same and digest(DATA/row.path)==row.sha256,
              f'{row.found}/{row.requested} source rows; {row.missing_source_rows} explicitly unavailable; input history only')
    inventory=[]
    for path in sorted((DATA/'pilot').glob('*.parquet')):
        schema=pq.read_schema(path)
        inventory.append(dict(path=str(path.relative_to(DATA)),rows=pq.read_metadata(path).num_rows,
            characteristic_columns=sum(c in schema.names for c in pd.read_csv(DATA/'candidate_registry.csv').characteristic),sha256=digest(path),
            role='price supplement' if path.stem.startswith('prc_') else 'exact-key training-period screening'))
    pd.DataFrame(inventory).to_csv(DATA/'pilot_file_inventory.csv',index=False)
    fields=pd.read_csv(DATA/'source_schema.csv');fields['included_predictor']=fields.name.isin(pd.read_csv(DATA/'candidate_registry.csv').characteristic)
    fields['reason']=np.where(fields.included_predictor,'documented characteristic block or published JKP characteristic',
        'identifier, observation flag, industry/currency metadata, realised/forward return, or auxiliary field')
    fields.to_csv(DATA/'source_field_admission_audit.csv',index=False)
    original=json.loads((KEATS/'feature_names.json').read_text())
    pd.DataFrame(dict(characteristic=sorted(set(original)|set(chosen.characteristic)))).assign(
        original_50=lambda x:x.characteristic.isin(original),recommended_50=lambda x:x.characteristic.isin(chosen.characteristic)).to_csv(
        OUTPUT/'original_vs_recommended_50.csv',index=False)
    selected_input_panels(chosen.characteristic.tolist())
    initial_path=OUTPUT/'review_archive/initial_proxy_only_50.csv'
    if initial_path.exists():
        initial=pd.read_csv(initial_path)
        changes=pd.DataFrame(dict(characteristic=sorted(set(initial.characteristic)|set(chosen.characteristic))))
        changes['preliminary_proxy_only_50']=changes.characteristic.isin(initial.characteristic)
        changes['final_published_anchor_50']=changes.characteristic.isin(chosen.characteristic)
        changes['rule_change']='Require at least one eligible published JKP representative in each theme; no test returns inspected'
        changes.to_csv(OUTPUT/'selection_review_changes.csv',index=False)
    for path in sorted((DATA/'selected50_ranked').glob('*.parquet')):
        d=pd.read_parquet(path)
        a=d[chosen.characteristic.tolist()].to_numpy(float)
        check(f'ranked_input_range_{path.stem}',np.isfinite(a).all() and (a>=0).all() and (a<=1).all(),'All exported ranks within [0,1]; source missing ranks explicitly 0.5')
    out=pd.DataFrame(audit);out.to_csv(OUTPUT/'final_reconciliation_checks.csv',index=False)
    if out.status.eq('FAIL').any():raise AssertionError('Final reconciliation failed')
    write_json(OUTPUT/'final_reconciliation_summary.json',dict(status='PASS',checks=len(out),
        counts=out.status.value_counts().to_dict(),reproduced_recommendation=True,raw_stock_months=int(manifest.requested.sum())))
    checks=pd.read_csv(OUTPUT/'stock_date_integrity.csv')
    write_report(chosen,scores,val,diag,checks)
    start=['# Keats characteristic results — start here','',
        f'Complete: {len(features)} characteristics downloaded for {int(manifest.requested.sum()):,} original regional stock-months; exactly 50 recommended for all three Keats universes.',
        '', '| Open | Contents |','|---|---|',
        '| [Readable report](index.html) | Methods, economic interpretation and the 50-feature table |',
        '| [Recommended 50](recommended_50_explained.csv) | Codes, descriptions, themes and evidence |',
        '| [Market results](selected_50_validation_by_universe.csv) | Separate US large, US mid and European large results |',
        '| [All exclusions](all_404_selection_and_exclusion_ledger.csv) | Stage decision for every candidate |',
        '| [Assumptions and changes](ASSUMPTIONS_EXCLUSIONS_AND_CHANGES.md) | Consolidated limitations and changes |',
        '| [Verification](final_reconciliation_summary.json) | Exact keys, source hashes, matrices and reproduced recommendation |',
        '| [Test log](validation/pytest_results.txt) | Regression and characteristic tests |',
        f'| [Data folder]({relative_link(OUTPUT,DATA)}/) | Raw 200-feature partitions, metadata, ranked 50-feature exports and lag history |',
        f'| [Code]({relative_link(OUTPUT,TRYOUT_ROOT/"README.md")}) | Acquisition, analysis and verification commands |','',
        'Selection uses return years 2005–2014 for the 404-to-200 screen and 2015–2019 for the final 50. Downloaded dates include the entire existing 1994–2025 formation history. Returns from 2020 onward are not used to select either list.',
        '',f'The {len(out)} final reconciliation checks pass. Separate independent formula/source checks are in verification_summary.json. The published-anchor rule guarantees representation from all 13 original JKP groups; extra inputs retain clearly labelled proxy groups.',
        '', 'Small predictive associations are not evidence of net portfolio profits. Existing Static-ML and Portfolio-ML fits are unchanged; the recommendation requires a separate future refit.',
        '',f'Portfolio-ML lag history: {int(lag_manifest.found.sum()):,} available observations and {int(lag_manifest.missing_source_rows.sum())} explicit unavailable source keys.',
        '', 'The review_archive folder and data/selected50_ranked_initial_proxy_selection preserve the preliminary selection for an audit trail. Use recommended_50.csv and data/selected50_ranked for the final version.']
    (OUTPUT/'START_HERE.md').write_text('\n'.join(start)+'\n',encoding='utf-8')
    (DATA/'README.md').write_text(f'''# Keats characteristic data

`panels/`: exact original broad regional stock-month keys, 200 frozen source characteristics, source-row availability, and a separately named source forward-return audit field. These source forward returns are not used as selection targets. The US-large and US-mid panels share regional training keys and are stored once as US.

`metadata_US.parquet` / `metadata_EU.parquet`: original observed next-month targets, original memberships and saved market/liquidity inputs. Formation month is `eom`; target return month is `eom + 1 month`. Formation dates run from December 1994 through November 2025.

`selected50_ranked/`: final shared 50 features, ranked within each monthly broad regional population. Only these exports use 0.5 for missing features; raw NaNs stay missing. Observation counts and at-least-half-observed flags are explicit.

`lag_history/`: the same 200 raw features for original supplemental Portfolio-ML lag keys. Missing source rows are flagged. These keys are input history, not additional return targets; they need consistent normalization and lag handling at a future refit.

`pilot/`: pre-2015 held-stock observations used to screen all 404 candidates. The older 403-feature US chunks are complemented by an explicit price supplement; source values are not fabricated.

`source_field_admission_audit.csv` distinguishes characteristics from identifiers, return targets and auxiliary source columns. `candidate_registry.csv` contains source definitions. `download_manifest.csv`, `pilot_file_inventory.csv` and `lag_history_manifest.csv` provide source-file provenance and SHA256 hashes.

Final results and exclusions: `{relative_link(DATA,OUTPUT/"START_HERE.md")}`. Code: `{relative_link(DATA,TRYOUT_ROOT/"README.md")}`. Licensed WRDS data are local and covered by the existing Git ignore rules.

`selected50_ranked_initial_proxy_selection/` is an archived preliminary version, retained for review history. It is not the final recommendation.
''',encoding='utf-8')
    print(f'FINAL RECONCILIATION COMPLETE: {len(out)} checks PASS; recommendation independently reproduced.',flush=True)
