"""Fetch only newly selected source columns; never fit or change an ML model."""
from __future__ import annotations
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd

from .paths import DATA, OUTPUT, KEATS
from .us_high_cap_selection import sha


def acquire_new_columns():
    selected=json.loads((OUTPUT/'us_high_cap_selection/recommended_50.json').read_text())
    cached=json.loads((DATA/'download_200.json').read_text())
    missing=[x for x in selected if x not in cached]
    if not all(re.fullmatch('[a-z][a-z0-9_]*',x) for x in missing):raise ValueError('Invalid predictor identifier')
    folder=DATA/'us_high_cap_selection/supplemental'
    folder.mkdir(parents=True,exist_ok=True)
    if not missing:return 0
    sys.path.insert(0,str(KEATS))
    from acquire_missing import credentials
    import wrds
    user,password=credentials()
    db=wrds.Connection(autoconnect=False,verbose=False,wrds_username=user,wrds_password=password,
        wrds_connect_args={'sslmode':'require','connect_timeout':30,'keepalives':1,'options':'-c statement_timeout=600000'})
    print(f'Connecting to WRDS for {len(missing)} new columns. Approve Duo if requested.',flush=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            db._Connection__make_sa_engine_conn(raise_err=True)
        records=[]
        cols=','.join('"'+x+'"' for x in missing)
        def fetch(keys,path):
            if path.exists():
                d=pd.read_parquet(path)
                if not set(missing).issubset(d):raise ValueError('Supplemental cache has another feature set')
                a=pd.MultiIndex.from_frame(keys[['id','eom']]);b=pd.MultiIndex.from_frame(d[['id','eom']])
                if len(a)!=len(b) or not a.difference(b).empty or not b.difference(a).empty:raise ValueError('Supplemental cache keys differ')
                print('Reuse',path.name,flush=True)
            else:
                d=db.raw_sql(f'select id,eom,{cols} from contrib.global_factor where id in %(ids)s '
                    'and eom between %(first)s and %(last)s order by id,eom',
                    params={'ids':tuple(int(x) for x in keys.id.unique()),'first':keys.eom.min().date(),'last':keys.eom.max().date()},date_cols=['eom'])
                d['id']=d.id.astype('int64');d['source_row_found']=True
                if d.duplicated(['id','eom']).any():raise ValueError('Duplicate WRDS source keys')
                d=keys[['id','eom']].merge(d,on=['id','eom'],how='left',validate='one_to_one')
                d['source_row_found']=d.source_row_found.eq(True)
                temporary=path.with_suffix('.part.parquet');d.to_parquet(temporary,index=False,compression='zstd');temporary.replace(path)
                print(f'Fetched {path.name}: {len(d):,} exact keys',flush=True)
            records.append(dict(path=str(path.relative_to(DATA)),rows=len(d),source_rows=int(d.source_row_found.sum()),sha256=sha(path),columns=missing))
        for region in ('US','EU'):
            metadata=pd.read_parquet(DATA/f'metadata_{region}.parquet',columns=['id','eom'])
            for year,keys in metadata.groupby(metadata.eom.dt.year,sort=True):fetch(keys,folder/f'{region}_{year}.parquet')
            keys=pd.read_parquet(DATA/f'lag_history/{region}.parquet',columns=['id','eom'])
            # Query all requested lag rows; unavailable rows remain explicitly absent.
            lagparts=[]
            for year,g in keys.groupby(keys.eom.dt.year,sort=True):
                p=folder/f'lag_{region}_{year}.parquet';fetch(g,p);lagparts.append(pd.read_parquet(p))
            pd.concat(lagparts,ignore_index=True).to_parquet(folder/f'lag_{region}.parquet',index=False)
        (folder/'manifest.json').write_text(json.dumps(dict(status='complete',models_fitted=False,new_columns=missing,files=records),indent=2)+'\n',encoding='utf-8')
        return 0
    except Exception as error:
        print('Source acquisition stopped:',type(error).__name__,'; existing caches and ML results retained.',flush=True)
        return 2
    finally:
        with contextlib.suppress(Exception):db.close()


if __name__=='__main__':raise SystemExit(acquire_new_columns())
