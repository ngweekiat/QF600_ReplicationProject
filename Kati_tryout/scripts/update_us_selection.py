"""Install the US high-cap association shortlist and archive old-feature fits.

This is an offline input migration after selection/acquisition. It never calls
the return forecaster, covariance estimator or either portfolio ML fit.
"""
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
CHAR=PROJECT/'Characteristics_tryout'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def checked_move(source,destination):
    source=source.resolve();destination=destination.resolve()
    if not source.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):raise ValueError('Archive target outside Kati')
    if not source.exists():return
    if destination.exists():raise ValueError(f'Archive destination already exists: {destination}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(source),str(destination))


def refresh_shared_manifest():
    """Retain existing hashes and register the shared inputs used by new fits."""
    path=ROOT/'data/SNAPSHOT_MANIFEST.json'
    manifest=json.loads(path.read_text())
    for item in manifest['files']:
        if digest(ROOT/item['path'])!=item['sha256']:raise AssertionError('Existing frozen input changed')
    present={item['path'] for item in manifest['files']}
    for folder in ('inputs','reference','characteristics'):
        for p in (ROOT/'data'/folder).rglob('*'):
            if not p.is_file():continue
            relative=str(p.relative_to(ROOT)).replace('\\','/')
            if relative not in present:
                manifest['files'].append(dict(path=relative,sha256=digest(p),bytes=p.stat().st_size,role='Shared source input or self-contained US association sample'))
    json_file(path,manifest)
    return len(manifest['files'])


def portable_analysis():
    """Keep all analysis source and measurement data local to Kati."""
    origin=CHAR/'characteristics'
    source=(origin/'us_high_cap_selection.py').read_text(encoding='utf-8')
    source=source.replace('from .paths import DATA, OUTPUT',
        "ROOT=Path(__file__).resolve().parents[1]\nDATA=ROOT/'data/characteristics/us_high_cap_inputs'\nOUTPUT=ROOT/'outputs'")
    source=source.replace('from .selection import assign_themes, diverse_pick','from selection_rules import assign_themes, diverse_pick')
    source=source.replace('from .statistics import panel_metrics, summarize, pairwise_matrices, paired_columns',
        'from association_statistics import panel_metrics, summarize, pairwise_matrices, paired_columns')
    (ROOT/'scripts/us_high_cap_selection.py').write_text(source,encoding='utf-8')
    shutil.copy2(origin/'statistics.py',ROOT/'scripts/association_statistics.py')
    original=(origin/'selection.py').read_text(encoding='utf-8');tree=ast.parse(original)
    funcs=[ast.get_source_segment(original,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('assign_themes','diverse_pick')]
    (ROOT/'scripts/selection_rules.py').write_text('"""Frozen transparent group/quota/redundancy rules; no ML fit."""\nimport numpy as np\nimport pandas as pd\n\n'+'\n\n'.join(funcs)+'\n',encoding='utf-8')
    shutil.copy2(CHAR/'data/us_high_cap_selection/us_high_cap_sample.parquet',ROOT/'data/characteristics/us_high_cap_inputs/us_high_cap_sample.parquet')
    shutil.copy2(CHAR/'data/us_high_cap_selection/candidate_registry.csv',ROOT/'data/characteristics/us_high_cap_inputs/candidate_registry.csv')


def update():
    result=CHAR/'outputs/us_high_cap_selection'
    selected=pd.read_csv(result/'recommended_50.csv')
    names=selected.characteristic.tolist()
    if len(names)!=len(set(names)) or len(names)!=50:raise ValueError('Expected exactly 50 selected characteristics')
    decision=json.loads((result/'selection_manifest.json').read_text())
    if decision['other_markets_used'] or decision['models_fitted'] or decision['selection_return_end']!='2014-12-31':raise ValueError('Selection must be association-only, US high-cap, 2005–2014')
    current=ROOT/'data/characteristics/recommended_50.json'
    if current.exists() and json.loads(current.read_text())==names:
        print('Current US high-cap 50 inputs already installed.');return 0
    missing=[x for x in names if x not in json.loads((CHAR/'data/download_200.json').read_text())]
    supplemental=CHAR/'data/us_high_cap_selection/supplemental'
    if missing and not (supplemental/'manifest.json').exists():raise FileNotFoundError('Acquire the missing new-characteristic columns before updating Kati')
    stage=ROOT/'data/.us_high_cap_selection_build'
    if stage.exists():raise ValueError('A prior staging folder exists; inspect it before another migration')
    stage.mkdir()
    files=[];audits=[]
    def register(path,role):
        files.append(dict(path='data/'+str(path.relative_to(stage)).replace('\\','/'),sha256=digest(path),bytes=path.stat().st_size,role=role))
    def save(frame,path,role):
        path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False,compression='zstd');register(path,role)
    chars=stage/'characteristics';chars.mkdir()
    shutil.copy2(result/'recommended_50.json',chars/'recommended_50.json');register(chars/'recommended_50.json','US high-cap-only 2005–2014 association shortlist')
    shutil.copy2(result/'recommended_50.csv',chars/'selected_feature_metadata.csv');register(chars/'selected_feature_metadata.csv','Selection reasons, group status and association diagnostics')
    directions=selected[['characteristic','theme','theme_assignment_method','published_jkp_153','direction','training_direction']].copy()
    directions['model_theme']=directions.theme.str.lower().str.replace(' ','_',regex=False).str.replace('-','_',regex=False)
    directions['model_theme_direction']=np.where(directions.published_jkp_153,directions.direction,directions.training_direction)
    directions['model_direction_source']=np.where(directions.published_jkp_153,'published JKP direction','2005–2014 US high-cap empirical sign for provisional group')
    if not directions.model_theme_direction.isin([-1,1]).all():raise ValueError('Invalid theme direction')
    directions.to_csv(chars/'theme_directions.csv',index=False);register(chars/'theme_directions.csv','Published signs or US high-cap empirical signs; raw ML ranks unflipped')
    for region,markets in [('US',('us_large','us_mid')),('EU',('eu_large',))]:
        print('Build new input snapshot:',region,flush=True)
        old_source=ROOT/f'data/source/screened_{region}.parquet'
        basecols=['id','eom','sic','size_grp','me','ret_exc_lead1m','hist_months','dolvol_126d']
        base=pd.read_parquet(old_source,columns=basecols)
        cached=[x for x in names if x not in missing]
        raw=pd.concat([pd.read_parquet(p,columns=['id','eom',*cached]) for p in sorted((CHAR/'data/panels').glob(f'{region}_*.parquet'))],ignore_index=True)
        if missing:
            extra=pd.concat([pd.read_parquet(p,columns=['id','eom',*missing]) for p in sorted(supplemental.glob(f'{region}_*.parquet'))],ignore_index=True)
            raw=raw.merge(extra,on=['id','eom'],how='left',validate='one_to_one',indicator=True)
            if not raw._merge.eq('both').all():raise ValueError('Missing supplemental keys')
            raw=raw.drop(columns='_merge')
        raw=raw[['id','eom',*names]]
        if raw.duplicated(['id','eom']).any():raise ValueError('Duplicate raw characteristic keys')
        raw[names]=raw[names].apply(pd.to_numeric,errors='raise').astype(float).replace([np.inf,-np.inf],np.nan)
        raw=base[['id','eom']].merge(raw,on=['id','eom'],validate='one_to_one',how='left',indicator=True)
        if not raw._merge.eq('both').all():raise ValueError('Original broad regional keys not covered')
        raw=raw.drop(columns='_merge')
        if 'dolvol_126d' in names:
            np.testing.assert_allclose(base.dolvol_126d,raw.dolvol_126d,equal_nan=True,rtol=1e-12,atol=1e-10)
        save(raw,chars/f'raw50_{region}.parquet','Actual full-history selected50 observations; only US high-cap evidence selected names')
        frame=base.drop(columns=[x for x in names if x in base]).merge(raw,on=['id','eom'],validate='one_to_one')
        save(frame,stage/f'source/screened_{region}.parquet','Original infrastructure with newly selected raw inputs')
        ranks=frame.groupby('eom')[names].rank(method='average',pct=True).fillna(.5)
        rankframe=pd.concat([frame[['id','eom']],ranks.add_suffix('_rank')],axis=1)
        for market in markets:
            member=pd.read_parquet(ROOT/f'data/source/panel_{market}.parquet',columns=['id','eom','form_year','mcap_rank','status'])
            rebuilt=member.merge(rankframe,on=['id','eom'],how='left',validate='one_to_one',indicator=True)
            if not rebuilt._merge.eq('both').all():raise ValueError('Original member keys absent')
            save(rebuilt.drop(columns='_merge'),stage/f'source/panel_{market}.parquet','Unchanged membership and independent new-feature ranks')
        old_lag=pd.read_parquet(ROOT/f'data/inputs/lag_history_{region}.parquet',columns=['id','eom','g','growth_imputed_neutral'])
        lag=pd.read_parquet(CHAR/f'data/lag_history/{region}.parquet',columns=['id','eom',*cached])
        if missing:
            extra=pd.read_parquet(supplemental/f'lag_{region}.parquet',columns=['id','eom',*missing])
            lag=lag.merge(extra,on=['id','eom'],how='left',validate='one_to_one')
        lag=old_lag.merge(lag,on=['id','eom'],how='left',validate='one_to_one',indicator=True)
        if not lag._merge.eq('both').all():raise ValueError('Actual lag keys absent from characteristic source')
        lag=lag.drop(columns='_merge')
        save(lag[['id','eom',*names]],chars/f'lag_raw50_{region}.parquet','Actual selected-feature lag observations')
        normalized=lag[['id','eom','g','growth_imputed_neutral']].copy()
        for name in names:normalized[name]=.5
        group_indices=frame.groupby('eom').groups
        for date,g in lag.groupby('eom'):
            reference=frame.loc[group_indices[date],names]
            for name in names:
                ref=np.sort(reference[name].dropna().to_numpy(float));values=g[name].to_numpy(float)
                valid=np.isfinite(values);a=np.full(len(g),.5)
                if len(ref):a[valid]=(np.searchsorted(ref,values[valid],side='left')+np.searchsorted(ref,values[valid],side='right')+2)/(2*(len(ref)+1))
                normalized.loc[g.index,name]=a
        normalized['ret_ld1']=np.nan;normalized['lambda']=np.nan;normalized['lag_history_only']=True
        save(normalized,stage/f'inputs/lag_history_{region}.parquet','New signal ranks, unchanged existing lag growth infrastructure')
        audits.append(dict(region=region,stock_months=len(frame),exact_broad_keys_preserved=True,absent_source_rows=0,
            formation_start=str(frame.eom.min().date()),formation_end=str(frame.eom.max().date())))
        del frame,raw,ranks,rankframe,base,lag,normalized
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_previous_50'
    archive_data=ROOT/'data/archives'/stamp;archive_outputs=ROOT/'outputs/archive'/stamp
    archive_data.mkdir(parents=True);archive_outputs.mkdir(parents=True)
    # The original feature set, coefficients, fitted caches and results move
    # together. They are never relabelled as fits for the new selected 50.
    shutil.copytree(ROOT/'configs',archive_data/'configs')
    shutil.copy2(ROOT/'settings.json',archive_data/'settings.json')
    if (ROOT/'Kati_tryout.ipynb').exists():shutil.copy2(ROOT/'Kati_tryout.ipynb',archive_data/'Kati_tryout.ipynb')
    shutil.copytree(ROOT/'scripts',archive_data/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    fitted=[p for p in (ROOT/'data/processed').rglob('*') if p.is_file() and p.name in ('models.npz','aim_coefs.parquet','chosen.csv')]
    fitted_hashes=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in fitted]
    output_hashes=[dict(path=str(p.relative_to(ROOT/'outputs')),sha256=digest(p)) for p in (ROOT/'outputs').rglob('*')
        if p.is_file() and 'archive' not in p.relative_to(ROOT/'outputs').parts and p.name in ('pf.parquet','weights.parquet','aim_coefs.parquet','chosen.csv')]
    moved=[]
    for name in ('characteristics','source','processed','raw','figure1'):
        p=ROOT/'data'/name
        if p.exists():checked_move(p,archive_data/name);moved.append('data/'+name)
    if (ROOT/'data/SNAPSHOT_MANIFEST.json').exists():checked_move(ROOT/'data/SNAPSHOT_MANIFEST.json',archive_data/'SNAPSHOT_MANIFEST.json')
    for p in list((ROOT/'outputs').iterdir()):
        if p.name=='archive':continue
        checked_move(p,archive_outputs/p.name)
    # Return/current risk inputs stay shared; only feature-dependent lag ranks change.
    for region in ('US','EU'):
        for suffix in ('.parquet','.json'):
            p=ROOT/f'data/inputs/lag_history_{region}{suffix}'
            if p.exists():checked_move(p,archive_data/f'inputs/{p.name}')
    for name in ('characteristics','source'):
        checked_move(stage/name,ROOT/'data'/name)
    for p in (stage/'inputs').iterdir():checked_move(p,ROOT/'data/inputs'/p.name)
    for region in ('US','EU'):
        json_file(ROOT/f'data/inputs/lag_history_{region}.json',dict(selection_scope='US high-cap 2005–2014',models_fitted=False,
            rank_policy='Monthly regional percentile ranks; actual inserted lag observation; neutral0.5 for missing feature',
            growth_policy='Prior actual lag infrastructure preserved by exact key'))
    themes={t:g.characteristic.tolist() for t,g in directions.groupby('model_theme',sort=False)}
    for p in (ROOT/'configs').rglob('*.yaml'):
        cfg=yaml.safe_load(p.read_text());cfg.setdefault('features',{})['themes']=themes
        p.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    settings=json.loads((ROOT/'settings.json').read_text())
    settings.update(features='US high-cap-only association-selected 50',selection_scope='us_large',selection_return_start='2005-01-31',
        selection_return_end='2014-12-31',selection_models_fitted=False,old_feature_archive=stamp)
    json_file(ROOT/'settings.json',settings)
    (ROOT/'data/characteristics/us_high_cap_inputs').mkdir()
    portable_analysis()
    shutil.copytree(result,ROOT/'outputs/us_high_cap_selection')
    for item in files:
        path=ROOT/item['path']
        if digest(path)!=item['sha256']:raise AssertionError('Staged input changed during installation')
    for item in fitted_hashes:
        relative=Path(item['path']).relative_to('data')
        if digest(archive_data/relative)!=item['sha256']:raise AssertionError('Archived fit coefficient hash changed')
    for item in output_hashes:
        if digest(archive_outputs/item['path'])!=item['sha256']:raise AssertionError('Archived portfolio or coefficient output changed')
    json_file(ROOT/'data/SNAPSHOT_MANIFEST.json',dict(status='PASS',pipeline='Kati_tryout',features=names,feature_count=50,
        selection_scope='us_large',selection_return_start='2005-01-31',selection_return_end='2014-12-31',models_refitted=False,
        runtime_requires_sibling_projects=False,files=files,regional_audits=audits,previous_feature_archive=stamp,
        theme_direction_policy=dict(published_direction_count=int(directions.published_jkp_153.sum()),training_direction_proxy_count=int((~directions.published_jkp_153).sum()))))
    migration=dict(status='PASS',new_selection=decision,archive_data=str(archive_data.relative_to(ROOT)),archive_results=str(archive_outputs.relative_to(ROOT)),
        preserved_fit_hashes=fitted_hashes,preserved_portfolio_output_hashes=output_hashes,models_fitted=False,old_fits_preserved=True,old_feature_data_archived=moved)
    json_file(ROOT/'outputs/validation/us_high_cap_selection_migration.json',migration)
    (archive_outputs/'README.md').write_text('# Previous characteristic-set results\n\nThese fits use the earlier three-market-selected 50. They were archived intact when switching to the US high-cap-only association shortlist. They are not results for the new 50.\n\nInputs, original configurations and code/notebook are under `'+str(archive_data.relative_to(ROOT))+'`.\n',encoding='utf-8')
    print('Installed new50; previous data/fits archived at',archive_data,'and',archive_outputs,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(update())
