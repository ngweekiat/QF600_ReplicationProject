"""Independent checks of source overlaps, dates, formulas and preservation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .acquire import DATA,OUTPUT,KEATS,digest,write_json,UNIVERSES
from .selection import load_pilot
from .statistics import numeric_array


def verify(hold,features,monthly):
    audit=[]
    inventory=pd.read_csv(DATA/'original_panel_inventory.csv')
    for row in inventory.itertuples():
        path=KEATS/row.path
        same=digest(path)==row.sha256
        audit.append(dict(check='original_prepared_panel_sha256_unchanged',scope=row.universe,status='PASS' if same else 'FAIL',detail=str(path)))
    if monthly.return_month.max()>pd.Timestamp('2019-12-31'):raise AssertionError('Test outcome in selection analysis')
    audit.append(dict(check='test_return_quarantine',scope='all selection',status='PASS',detail='No monthly selection outcome after 2019-12-31'))
    # Independent library calculations on randomly sampled month/features.
    rng=np.random.default_rng(20261101)
    for universe,frame in hold.items():
        months=sorted(frame.eom.unique())
        for month in rng.choice(months,size=min(8,len(months)),replace=False):
            g=frame.loc[frame.eom.eq(month)]
            for c in rng.choice(features,size=min(5,len(features)),replace=False):
                a=g[[c,'observed_ret_ld1']].apply(pd.to_numeric,errors='coerce').dropna()
                observed=monthly.loc[(monthly.universe==universe)&monthly.formation_month.eq(month)&monthly.characteristic.eq(c)].iloc[0]
                if len(a)<20:continue
                expected=spearmanr(a[c],a.observed_ret_ld1).statistic
                raw=a.cov().iloc[0,1]
                same=(np.isclose(observed.rank_ic,expected,equal_nan=True,atol=1e-12) and np.isclose(observed.raw_characteristic_return_covariance,raw,equal_nan=True,rtol=1e-10,atol=1e-12))
                audit.append(dict(check='independent_scipy_spearman_and_pandas_covariance',scope=f'{universe} {pd.Timestamp(month).date()} {c}',status='PASS' if same else 'FAIL',detail=f'pairs={len(a)}'))
    # Pilot and final panels come from separate WRDS queries; compare actual raw cells.
    registry,allfeatures,pilot=load_pilot()
    source_rows=[]
    for universe,frame in pilot.items():
        sample=frame.sample(min(200,len(frame)),random_state=20261101)
        region='EU' if universe.startswith('eu') else 'US'
        for year,group in sample.groupby(sample.eom.dt.year):
            full=pd.read_parquet(DATA/f'panels/{region}_{year}.parquet',columns=['id','eom',*features])
            comp=group[['id','eom',*features]].merge(full,on=['id','eom'],how='left',validate='one_to_one',suffixes=('_pilot','_full'))
            x=numeric_array(comp[[c+'_pilot' for c in features]].to_numpy())
            y=numeric_array(comp[[c+'_full' for c in features]].to_numpy())
            same=np.isclose(x,y,equal_nan=True,rtol=1e-10,atol=1e-12)
            source_rows.append(dict(universe=universe,year=int(year),cells=int(x.size),observed_cells=int(np.isfinite(x).sum()),differences=int((~same).sum())))
    pd.DataFrame(source_rows).to_csv(OUTPUT/'independent_source_overlap_checks.csv',index=False)
    differences=sum(r['differences'] for r in source_rows)
    audit.append(dict(check='independent_WRDS_pilot_vs_full_raw_cells',scope='600 sampled stock-months',status='PASS' if differences==0 else 'WARN',detail=f'{sum(r["cells"] for r in source_rows):,} cells, {differences} source-vintage differences; no overwriting performed'))
    chosen=pd.read_csv(OUTPUT/'recommended_50.csv')
    audit.append(dict(check='exact_shared_50_no_duplicates',scope='recommendation',status='PASS' if len(chosen)==50 and chosen.characteristic.nunique()==50 else 'FAIL',detail=f'themes={chosen.theme.nunique()}'))
    for u in UNIVERSES:
        covered=hold[u].groupby('id').size()
        region='EU' if u.startswith('eu') else 'US'
        meta=pd.read_parquet(DATA/f'metadata_{region}.parquet',columns=['id',f'member_{u}'])
        ids=sorted(meta.loc[meta[f'member_{u}'],'id'].unique())
        pd.DataFrame(dict(id=ids)).assign(universe=u,pretest_held_months=lambda x:x.id.map(covered).fillna(0).astype(int)).to_csv(OUTPUT/f'{u}_stock_pretest_coverage.csv',index=False)
    out=pd.DataFrame(audit);out.to_csv(OUTPUT/'verification_checks.csv',index=False)
    if out.status.eq('FAIL').any():raise AssertionError('Independent characteristic verification failed')
    write_json(OUTPUT/'verification_summary.json',dict(status='PASS',checks=len(out),counts=out.status.value_counts().to_dict(),
        independent_source_cells=sum(r['cells'] for r in source_rows),original_panels_preserved=True,test_returns_used=False))
