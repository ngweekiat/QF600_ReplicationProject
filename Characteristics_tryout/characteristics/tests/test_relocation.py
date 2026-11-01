from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import pandas as pd

from characteristics import acquire

ROOT=Path(__file__).resolve().parents[2]


def run_paths(cwd,*options):
    env=dict(os.environ)
    env.pop('CHARACTERISTICS_DATA_DIR',None)
    env.pop('CHARACTERISTICS_OUTPUT_DIR',None)
    result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'run.py'),'paths',*map(str,options)],
        cwd=cwd,env=env,text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def test_default_paths_are_isolated_and_independent_of_working_directory(tmp_path):
    result=run_paths(tmp_path)
    assert Path(result['data'])==ROOT/'data'
    assert Path(result['outputs'])==ROOT/'outputs'
    assert Path(result['keats_inputs'])==ROOT.parent/'keats_pipeline'
    assert not Path(result['data']).is_relative_to(Path(result['keats_inputs']))


def test_existing_cache_and_output_options_are_applied_before_imports(tmp_path):
    data=tmp_path/'cache';outputs=tmp_path/'reports'
    result=run_paths(tmp_path,'--data-dir',data,'--output-dir',outputs)
    assert Path(result['data'])==data and Path(result['outputs'])==outputs
    assert not data.exists() and not outputs.exists()


def test_calculation_and_selection_modules_keep_the_archived_source_bytes():
    source=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
    hashes={row['file']:row['source_sha256'] for row in source['files']}
    for name in ('statistics.py','selection.py','analysis.py','verification.py'):
        assert hashlib.sha256((ROOT/'characteristics'/name).read_bytes()).hexdigest()==hashes[name]


def test_first_acquisition_fetches_schema_once_and_reuses_it(tmp_path,monkeypatch):
    monkeypatch.setattr(acquire,'DATA',tmp_path)
    class Source:
        calls=0
        def raw_sql(self,sql):
            self.calls+=1
            assert 'pg_attribute' in sql and 'contrib.global_factor' in sql
            return pd.DataFrame({'name':['id','eom','market_equity','qmj','prc'],
                'dtype':['bigint','date','double precision','double precision','double precision'],
                'description':['identifier','month','size','quality','price']})
    source=Source();acquire.ensure_source_schema(source);acquire.ensure_source_schema(source)
    assert source.calls==1
    assert pd.read_csv(tmp_path/'source_schema.csv').name.tolist()==['id','eom','market_equity','qmj','prc']
