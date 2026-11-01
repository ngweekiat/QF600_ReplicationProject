"""Check output separation, restored bytes, original inputs and local report links."""
from pathlib import Path
from html.parser import HTMLParser
import ast
import json
import pandas as pd

from workflow import ROOT,UNIVERSES,config_path,originals_unchanged
from jkmp.config import load_config
from organise_results import sha256


def main():
    checks=[]
    def check(name,scope,passed,detail=''):
        checks.append(dict(check=name,scope=scope,status='PASS' if passed else 'FAIL',detail=detail))
    groups=('base','iteration_1','iteration_2','iteration_3')
    features=set(json.loads((ROOT/'feature_names.json').read_text()))
    for market in UNIVERSES:
        for group in groups:
            cfg=load_config(config_path(market,group))
            check('separate_group_output_path',f'{group}/{market}',cfg.paths.outputs==ROOT/'outputs'/group/market)
            check('original_50_features',f'{group}/{market}',len(cfg.features.names)==50 and set(cfg.features.names)==features)
        pf=[]
        for rel,method in [('static_ml/pf.parquet','Static-ML'),('portfolio_ml/rff/pf.parquet','Portfolio-ML')]:
            data=pd.read_parquet(ROOT/'outputs/base'/market/rel)
            check('base_method_and_calendar',f'{market}/{method}',set(data.type)=={method} and len(data)==72)
            pf.append(data)
        table=pd.read_csv(ROOT/'outputs/base'/market/'table2_test.csv')
        check('base_table_has_only_two_methods',market,set(table.type)=={'Static-ML','Portfolio-ML'} and len(table)==2)
    # Every restored native CSV/Parquet/chart is compared to its saved original hash.
    restoration=json.loads((ROOT/'outputs/organisation_audit/restored_files.json').read_text())
    for row in restoration:
        if row['role'].startswith('original iteration'):
            path=ROOT/row['destination']
            check('original_iteration_artifact_unchanged',row['destination'],path.is_file() and sha256(path)==row['sha256'])
    backup=Path('C:/Users/katiravan/Downloads/keats_removed_models_20261101/code_before/keats_pipeline')
    same=[('src/jkmp/portfolio/portfolio_ml.py','src/jkmp/portfolio/portfolio_ml.py'),
          ('src/jkmp/portfolio/active.py','src/jkmp/portfolio/active.py'),
          ('src/jkmp/portfolio/iteration_static_ml.py','src/jkmp/portfolio/static_ml.py'),
          ('src/jkmp/evaluation/cost_sensitivity.py','src/jkmp/evaluation/cost_sensitivity.py')]
    for current,original in same:
        check('original_iteration_math_unchanged',current,sha256(ROOT/current)==sha256(backup/original))
    # Original prepared inputs and daily-risk/forecast caches are not rewritten.
    hashes=pd.read_csv(ROOT/'outputs/model_cleanup/retained_input_hashes_before.csv')
    for row in hashes.itertuples():check('retained_input_hash_unchanged',row.path,sha256(ROOT/row.path)==row.sha256)
    allowed={'Static-ML','Portfolio-ML'}
    for path in (ROOT/'outputs/base').rglob('*.csv'):
        try:data=pd.read_csv(path)
        except pd.errors.EmptyDataError:continue
        for column in ('type','strategy','strategy_a','strategy_b'):
            if column in data:
                check('base_csv_method_scope',str(path.relative_to(ROOT))+' '+column,set(data[column].dropna()).issubset(allowed))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            self.links.extend(v for k,v in attrs if k in ('href','src'))
    pages=[ROOT/'outputs/index.html']
    pages += [ROOT/'outputs'/group/'index.html' for group in groups]
    pages += [ROOT/'outputs'/group/market/'index.html' for group in groups for market in UNIVERSES]
    for page in pages:
        parser=Links();parser.feed(page.read_text(encoding='utf-8'))
        missing=[x for x in parser.links if not x.startswith(('http:','https:','#')) and not (page.parent/x).exists()]
        check('report_links_exist',str(page.relative_to(ROOT)),not missing,str(missing))
    availability=pd.read_csv(ROOT/'outputs/availability_by_group_and_market.csv')
    check('all_12_group_market_combinations', 'availability',len(availability)==12 and not availability.duplicated(['group','market']).any())
    check('no_missing_experiments_run','availability',availability.new_experiments_run.eq(False).all())
    check('iteration_1_saved_runs_not_mislabeled', 'availability',availability.query("group=='iteration_1' and market=='us_large'").experiment_series.eq(3).all())
    check('iteration_2_and_3_not_falsely_completed','availability',availability.query("group in ['iteration_2','iteration_3']").experiment_series.eq(0).all())
    for path in (ROOT/'scripts').rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    mapping=json.loads((ROOT/'scripts/SCRIPT_MAP.json').read_text())
    check('script_map_has_no_missing_entries','script map',all((ROOT/row['script']).is_file() for row in mapping))
    check('original_external_pipeline_unchanged','original manifest',originals_unchanged())
    out=ROOT/'outputs/organisation_audit'
    table=pd.DataFrame(checks);table.to_csv(out/'organisation_checks.csv',index=False)
    summary=dict(status='PASS' if table.status.eq('PASS').all() else 'FAIL',checks=len(table),
                 counts=table.status.value_counts().to_dict(),new_missing_experiments_run=False)
    (out/'organisation_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    if summary['status']!='PASS':raise AssertionError('Organisation verification failed; inspect organisation_checks.csv')
    return 0


if __name__=='__main__':raise SystemExit(main())
