"""Explicit characteristic acquisition and offline analysis, isolated from Keats outputs."""
from pathlib import Path
import argparse
import contextlib
import io
import json
import os
import sys


def acquire(lags_only=False):
    from .acquire import (DATA,KEATS,prepare_local,download_pilot,complete_pilot_price,
                          download_full,download_lag_features,ensure_source_schema,write_json)
    sys.path.insert(0,str(KEATS))
    from acquire_missing import credentials
    import wrds
    username,password=credentials()
    db=wrds.Connection(autoconnect=False,verbose=False,wrds_username=username,wrds_password=password,
        wrds_connect_args={'sslmode':'require','connect_timeout':30,'keepalives':1,
            'keepalives_idle':30,'keepalives_interval':10,'keepalives_count':3,
            'options':'-c statement_timeout=600000'})
    print('Connecting once to WRDS; approve Duo if requested.',flush=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            db._Connection__make_sa_engine_conn(raise_err=True)
        print('WRDS connected; preserving existing cached data.',flush=True)
        if lags_only:
            download_lag_features(db)
            return 0
        ensure_source_schema(db)
        prepare_local()
        download_pilot(db)
        complete_pilot_price(db)
        from .selection import preselect
        if not (DATA/'download_200.json').exists():preselect()
        else:print('Reuse the frozen pre-2015 download shortlist.',flush=True)
        download_full(db)
        download_lag_features(db)
        write_json(DATA/'acquisition_status.json',dict(status='complete',credentials_saved=False))
        return 0
    except Exception as err:
        print('Acquisition stopped:',type(err).__name__,'Cached data retained; no source panels changed.',flush=True)
        write_json(DATA/'acquisition_status.json',dict(status='incomplete',error_type=type(err).__name__,credentials_saved=False))
        return 2
    finally:
        db.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['paths','acquire','acquire-lags','screen','analyse','finalize','prepare'])
    parser.add_argument('--wait',action='store_true',help='Analyse partitions while explicit acquisition completes them')
    parser.add_argument('--data-dir',type=Path,help='Existing cache or new data directory (default: Characteristics_tryout/data)')
    parser.add_argument('--output-dir',type=Path,help='Analysis destination (default: Characteristics_tryout/outputs)')
    args=parser.parse_args(argv)
    if args.data_dir is not None:os.environ['CHARACTERISTICS_DATA_DIR']=str(args.data_dir.expanduser().resolve())
    if args.output_dir is not None:os.environ['CHARACTERISTICS_OUTPUT_DIR']=str(args.output_dir.expanduser().resolve())
    # Import only after parsing path options, so every module sees one set of paths.
    from .paths import TRYOUT_ROOT,PROJECT,KEATS,DATA,OUTPUT
    if args.stage=='paths':
        print(json.dumps(dict(tryout=str(TRYOUT_ROOT),project=str(PROJECT),keats_inputs=str(KEATS),data=str(DATA),outputs=str(OUTPUT)),indent=2))
        return 0
    if args.stage=='acquire':return acquire()
    if args.stage=='acquire-lags':return acquire(lags_only=True)
    if args.stage=='prepare':
        from .acquire import prepare_local
        prepare_local();return 0
    if args.stage=='screen':
        from .selection import preselect
        preselect();return 0
    if args.stage=='finalize':
        from .finalize import finalize
        finalize();return 0
    from .analysis import analyse
    analyse(wait=args.wait);return 0


if __name__=='__main__':raise SystemExit(main())
