"""Read-only WRDS acquisition; never changes existing Keats panels or models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .paths import KEATS,PROJECT,DATA,OUTPUT
UNIVERSES=('us_large','us_mid','eu_large')
EXCLUDED_FIELDS={'id','permno','permco','gvkey','iid','excntry','exch_main','common','primary_sec',
    'bidask','crsp_shrcd','crsp_exchcd','comp_tpci','comp_exchg','curcd','fx','date','eom','adjfct',
    'shares','me','me_company','prc_local','prc_high','prc_low','dolvol','tvol','ret','ret_local',
    'ret_exc','ret_lag_dif','source_crsp','ret_exc_lead1m','obs_main','gics','sic','naics','ff49','size_grp','me_lag1'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.writing.json')
    tmp.write_text(json.dumps(data,indent=2,default=str)+'\n',encoding='utf-8');tmp.replace(path)


def registry(schema):
    """Source's documented numeric characteristic block; IDs/returns are forbidden."""
    schema=schema.copy()
    start=int(schema.index[schema.name.eq('market_equity')][0])
    end=int(schema.index[schema.name.eq('qmj')][0])
    published=pd.read_parquet((DATA/'reference/cluster_labels.parquet' if (DATA/'reference/cluster_labels.parquet').exists() else PROJECT/'data/raw/cluster_labels.parquet')).characteristic
    # Price is outside the contiguous block but is an explicitly published JKP
    # characteristic. Never discard a published predictor as generic metadata.
    frame=schema.loc[schema.index.to_series().between(start,end)|schema.name.isin(published)].rename(columns={'name':'characteristic'})
    if set(frame.characteristic)&EXCLUDED_FIELDS:raise ValueError('Non-characteristic field in predictor block')
    if not frame.dtype.eq('double precision').all():raise ValueError('Unexpected nonnumeric candidate schema')
    if frame.characteristic.duplicated().any():raise ValueError('Duplicate characteristic names')
    details=pd.read_parquet((DATA/'reference/factor_details.parquet' if (DATA/'reference/factor_details.parquet').exists() else PROJECT/'data/raw/factor_details.parquet'))
    details=details.loc[details.abr_jkp.notna(),['abr_jkp','name_new','direction']]
    clusters=pd.read_parquet((DATA/'reference/cluster_labels.parquet' if (DATA/'reference/cluster_labels.parquet').exists() else PROJECT/'data/raw/cluster_labels.parquet')).rename(columns={'cluster':'published_theme'})
    frame=frame.merge(details,left_on='characteristic',right_on='abr_jkp',how='left',validate='one_to_one')
    frame=frame.merge(clusters,on='characteristic',how='left',validate='one_to_one').drop(columns=['abr_jkp'])
    if not frame.characteristic.str.fullmatch('[a-z][a-z0-9_]*').all():raise ValueError('Unexpected source SQL identifier')
    frame['published_jkp_153']=frame.published_theme.notna()
    frame['source']='WRDS contrib.global_factor source column description'
    return frame


def ensure_source_schema(db):
    """Fetch missing column metadata only during an explicit acquisition."""
    path=DATA/'source_schema.csv'
    if path.exists():
        return
    schema=db.raw_sql('''select a.attname as name,
        pg_catalog.format_type(a.atttypid,a.atttypmod) as dtype,
        col_description(a.attrelid,a.attnum) as description
        from pg_attribute a
        where a.attrelid='contrib.global_factor'::regclass
          and a.attnum>0 and not a.attisdropped
        order by a.attnum''')
    if schema.name.duplicated().any() or not {'id','eom','market_equity','qmj','prc'}.issubset(schema.name):
        raise ValueError('Unexpected WRDS characteristic schema')
    DATA.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.part.csv')
    schema.to_csv(temporary,index=False);temporary.replace(path)


def prepare_local():
    """Exact historical training population and memberships, observed saved targets."""
    DATA.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    if not (DATA/'source_schema.csv').exists():
        raise FileNotFoundError('No source_schema.csv in the selected data directory; run acquire or select the existing cache with --data-dir')
    schema=pd.read_csv(DATA/'source_schema.csv')
    regs=registry(schema);regs.to_csv(DATA/'candidate_registry.csv',index=False)
    fields=schema.copy();fields['included_predictor']=fields.name.isin(regs.characteristic)
    fields['reason']=np.where(fields.included_predictor,'documented characteristic block or published JKP characteristic',
        'identifier, observation flag, industry/currency metadata, realised/forward return, or auxiliary field; not an admitted predictor')
    fields.to_csv(DATA/'source_field_admission_audit.csv',index=False)
    proofs=[];frames={}
    for universe in UNIVERSES:
        path=KEATS/f'data/processed/{universe}/with_zero_returns/panel.parquet'
        d=pd.read_parquet(path,columns=['id','eom','in_universe','ret_ld1','ret_ld1_original',
                                      'ret_ld1_imputed_zero','me','dolvol','lambda'])
        if d.duplicated(['id','eom']).any():raise ValueError(f'Duplicate existing panel keys: {universe}')
        if not d.eom.dt.is_month_end.all():raise ValueError('Non-month-end panel date')
        proofs.append(dict(universe=universe,path=str(path.relative_to(KEATS)),sha256=digest(path),
            rows=len(d),member_rows=int(d.in_universe.sum()),securities=d.id.nunique(),
            first_formation=d.eom.min(),last_formation=d.eom.max()))
        frames[universe]=d
    for region,universes in [('US',['us_large','us_mid']),('EU',['eu_large'])]:
        target=DATA/f'metadata_{region}.parquet'
        if target.exists():continue
        first=frames[universes[0]]
        meta=first[['id','eom','ret_ld1_original','me','dolvol','lambda']].rename(columns={'ret_ld1_original':'observed_ret_ld1'})
        meta=meta.copy();meta['id']=meta.id.astype('int64');meta['return_month']=meta.eom+pd.offsets.MonthEnd(1)
        for u in UNIVERSES:
            if u in universes:
                other=frames[u][['id','eom','in_universe','ret_ld1_original']]
                merged=meta[['id','eom','observed_ret_ld1']].merge(other,on=['id','eom'],how='left',validate='one_to_one',indicator=True)
                if len(meta)!=len(other) or not merged['_merge'].eq('both').all():raise ValueError('US regional panel key sets differ')
                if not np.allclose(merged.observed_ret_ld1,merged.ret_ld1_original,equal_nan=True):
                    raise ValueError('Original observed regional targets differ across membership variants')
                meta[f'member_{u}']=merged.in_universe.to_numpy(dtype=bool)
            else:meta[f'member_{u}']=False
        meta.to_parquet(target,index=False)
    pd.DataFrame(proofs).to_csv(DATA/'original_panel_inventory.csv',index=False)
    write_json(DATA/'design.json',dict(source='WRDS contrib.global_factor',candidate_count=len(regs),
        cap=200,recommendation_count=50,screen_return_start='2005-01-31',screen_return_end='2014-12-31',
        validation_return_start='2015-01-31',validation_return_end='2019-12-31',
        untouched_test_start='2020-01-31',untouched_test_end='2025-12-31',
        selection_target='existing observed next-month USD excess return; assumed-zero targets excluded',
        scope='both existing broad regional training panels and all three held-stock universes',
        source_vintage=pd.Timestamp.now(tz='UTC').isoformat(),
        reference_paper=dict(pdf='research paper 1.pdf',signals=115,feature_themes=13,
            printed_page_3056='115 features; drops poorly covered early features from JKP153; rank to [0,1], missing=0.5',
            economic_feature_importance='utility value loss after removing signals; distinct from predictive correlation'),
        preselection_score='0.55 abs equally weighted signed mean monthly rank IC percentile +0.20 minimum universe coverage percentile +0.15 yearly directional stability percentile +0.10 persistence percentile',
        final_selection='pre-2020 balanced-universe evidence with theme quotas and redundancy penalty; never ranks test returns',
        missing_characteristics='source NaNs preserved; no source zeros inserted',
        preselection_pilot='all held stock-months across all three universes in 2005-2014; full capped data then includes broad training populations'))
    return regs


def download_pilot(db,batch_size=1000):
    """Screen every source candidate before 2015; no test/validation outcomes."""
    regs=pd.read_csv(DATA/'candidate_registry.csv')
    columns=','.join('"'+c+'"' for c in regs.characteristic)
    audit=[]
    for region in ('US','EU'):
        meta=pd.read_parquet(DATA/f'metadata_{region}.parquet')
        held=meta[[f'member_{u}' for u in UNIVERSES]].any(axis=1)
        sample=meta.loc[held & meta.return_month.between('2005-01-31','2014-12-31'),['id','eom']]
        existing=sorted((DATA/'pilot').glob(f'{region}_*.parquet'))
        if existing:
            covered=pd.concat([pd.read_parquet(p,columns=['id','eom']) for p in existing],ignore_index=True)
            if covered.duplicated(['id','eom']).any():raise ValueError('Cached pilot keys overlap')
            sample=sample.merge(covered.assign(cached=True),on=['id','eom'],how='left',validate='one_to_one')
            sample=sample.loc[sample.cached.isna(),['id','eom']]
        for year,g in sample.groupby(sample.eom.dt.year,sort=True):
            path=DATA/f'pilot/{region}_year{year}.parquet'
            if path.exists():raise ValueError('Uncovered keys collide with an existing pilot partition')
            t=time.monotonic();path.parent.mkdir(parents=True,exist_ok=True)
            # Annual id/range predicates avoid enormous tuple-IN query plans.
            d=db.raw_sql(f'select id,eom,{columns} from contrib.global_factor where id in %(ids)s and eom between %(first)s and %(last)s order by id,eom',
                params={'ids':tuple(int(i) for i in g.id.unique()),'first':g.eom.min().date(),'last':g.eom.max().date()},date_cols=['eom'])
            if d.duplicated(['id','eom']).any():raise ValueError('Source has duplicate pilot keys')
            d['id']=d.id.astype('int64')
            d=g[['id','eom']].merge(d,on=['id','eom'],how='left',validate='one_to_one')
            temporary=path.with_suffix('.part.parquet');d.to_parquet(temporary,index=False);temporary.replace(path)
            audit.append(dict(path=str(path.relative_to(DATA)),rows=len(d),sha256=digest(path),seconds=time.monotonic()-t))
            print(f'Pilot {region} {year}: {len(d):,} rows, {time.monotonic()-t:.1f}s',flush=True)
    if audit:pd.DataFrame(audit).to_csv(DATA/'pilot_download_manifest.csv',index=False)
    print('PILOT DOWNLOAD COMPLETE',flush=True)


def download_full(db):
    """Download at most 200 preselected features for the exact existing panels."""
    features=json.loads((DATA/'download_200.json').read_text())
    if len(features)>200 or set(features)&EXCLUDED_FIELDS:raise ValueError('Invalid selected predictor list')
    columns=','.join('"'+c+'"' for c in features)
    audit=[]
    for region in ('US','EU'):
        meta=pd.read_parquet(DATA/f'metadata_{region}.parquet')
        for year,g in meta.groupby(meta.eom.dt.year,sort=True):
            path=DATA/f'panels/{region}_{year}.parquet'
            if path.exists():
                import pyarrow.parquet as pq
                if not set(features).issubset(pq.read_schema(path).names):raise ValueError('Cached partition has different features')
                print(f'Reuse {path.name}',flush=True);continue
            t=time.monotonic();path.parent.mkdir(parents=True,exist_ok=True)
            d=db.raw_sql(f'select id,eom,ret_exc_lead1m as source_ret_exc_lead1m,{columns} from contrib.global_factor '
                'where id in %(ids)s and eom between %(first)s and %(last)s order by id,eom',
                params={'ids':tuple(int(i) for i in g.id.unique()),'first':g.eom.min().date(),'last':g.eom.max().date()},date_cols=['eom'])
            if d.duplicated(['id','eom']).any():raise ValueError(f'Duplicate source keys: {region} {year}')
            d['id']=d.id.astype('int64');d['source_row_found']=True
            matched=g[['id','eom']].merge(d,on=['id','eom'],how='left',validate='one_to_one')
            matched['source_row_found']=matched.source_row_found.fillna(False).astype(bool)
            if len(matched)!=len(g):raise AssertionError('Requested key preservation failed')
            temporary=path.with_suffix('.part.parquet');matched.to_parquet(temporary,index=False,compression='zstd');temporary.replace(path)
            record=dict(path=str(path.relative_to(DATA)),region=region,year=int(year),requested=len(g),
                found=int(matched.source_row_found.sum()),missing_source_rows=int((~matched.source_row_found).sum()),
                features=len(features),sha256=digest(path),seconds=round(time.monotonic()-t,2))
            audit.append(record)
            write_json(path.with_suffix('.json'),record)
            print(f'Full {region} {year}: {len(matched):,} rows, {record["missing_source_rows"]} source gaps, {path.stat().st_size/1e6:.1f}MB, {record["seconds"]}s',flush=True)
    records=[json.loads(p.read_text()) for p in sorted((DATA/'panels').glob('*.json'))]
    pd.DataFrame(records).to_csv(DATA/'download_manifest.csv',index=False)
    print('FULL DOWNLOAD COMPLETE',flush=True)


def complete_pilot_price(db):
    """Repair an older pilot lacking published price, without re-fetching 403 columns."""
    for region in ('US','EU'):
        paths=sorted((DATA/'pilot').glob(f'{region}_*.parquet'))
        if not paths or all('prc' in pd.read_parquet(p).columns for p in paths):continue
        target=DATA/f'pilot/prc_{region}.parquet'
        if target.exists():continue
        meta=pd.read_parquet(DATA/f'metadata_{region}.parquet')
        held=meta[[f'member_{u}' for u in UNIVERSES]].any(axis=1)
        k=meta.loc[held & meta.return_month.between('2005-01-31','2014-12-31')]
        d=db.raw_sql('select id,eom,prc from contrib.global_factor where id in %(ids)s and eom between %(first)s and %(last)s',
            params={'ids':tuple(int(i) for i in k.id.unique()),'first':k.eom.min().date(),'last':k.eom.max().date()},date_cols=['eom'])
        if d.duplicated(['id','eom']).any():raise ValueError('Duplicate pilot price keys')
        d['id']=d.id.astype('int64');d.to_parquet(target,index=False)
        print(f'Pilot price {region}: {len(d):,} rows',flush=True)


def download_lag_features(db):
    """Supplement original Portfolio-ML lag keys; never use them as target samples."""
    features=json.loads((DATA/'download_200.json').read_text())
    columns=','.join('"'+c+'"' for c in features)
    audit=[]
    for region in ('US','EU'):
        destination=DATA/f'lag_history/{region}.parquet'
        keys=pd.read_parquet(KEATS/f'data/inputs/lag_history_source_{region}.parquet',columns=['id','eom'])
        unavailable=pd.read_csv(KEATS/f'data/inputs/unavailable_lag_history_{region}.csv',parse_dates=['eom'])
        keys=pd.concat([keys,unavailable],ignore_index=True).drop_duplicates(['id','eom'])
        keys['id']=keys.id.astype('int64')
        if not destination.exists():
            rows=[]
            for year,g in keys.groupby(keys.eom.dt.year):
                source=db.raw_sql(f'select id,eom,{columns} from contrib.global_factor '
                    'where id in %(ids)s and eom between %(first)s and %(last)s',
                    params={'ids':tuple(int(i) for i in g.id.unique()),'first':g.eom.min().date(),
                            'last':g.eom.max().date()},date_cols=['eom'])
                source['id']=source.id.astype('int64');source['source_row_found']=True
                if source.duplicated(['id','eom']).any():raise ValueError('Duplicate supplemental lag keys')
                matched=g.merge(source,on=['id','eom'],how='left',validate='one_to_one')
                matched['source_row_found']=matched.source_row_found.eq(True)
                matched[features]=matched[features].apply(pd.to_numeric,errors='raise').astype('float64')
                rows.append(matched)
            d=pd.concat(rows,ignore_index=True)
            destination.parent.mkdir(parents=True,exist_ok=True)
            temp=destination.with_suffix('.part.parquet');d.to_parquet(temp,index=False,compression='zstd');temp.replace(destination)
        else:d=pd.read_parquet(destination)
        matched=keys.merge(d[['id','eom']],on=['id','eom'],how='outer',indicator=True,validate='one_to_one')
        if not matched['_merge'].eq('both').all() or not set(features).issubset(d.columns):
            raise ValueError('Supplemental lag cache does not match requested keys/features')
        audit.append(dict(region=region,requested=len(keys),found=int(d.source_row_found.sum()),
            missing_source_rows=int((~d.source_row_found).sum()),features=len(features),
            path=str(destination.relative_to(DATA)),sha256=digest(destination),used_for_selection=False))
        print(f'Lag supplement {region}: {len(d):,} keys, {audit[-1]["missing_source_rows"]} source gaps',flush=True)
    pd.DataFrame(audit).to_csv(DATA/'lag_history_manifest.csv',index=False)
