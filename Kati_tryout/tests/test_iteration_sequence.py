"""Acceptance, paired statistics and accepted-only ancestry, without real-data model fits."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import iteration_sequence as seq


def pf(offset=0.):
    rng=np.random.default_rng(100)
    return pd.DataFrame(dict(eom_ret=pd.date_range('2020-01-31',periods=72,freq='ME'),
        r=rng.normal(.005,.03,72)+offset,tc=.001,turnover=.1,inv=1.,shorting=.1))


def test_bootstrap_is_paired_reproducible_and_reverses():
    a=pf(.003);b=pf()
    first=seq.paired_probability(a,b,reps=500)
    second=seq.paired_probability(a,b,reps=500)
    reverse=seq.paired_probability(b,a,reps=500)
    assert first==second
    assert first['probability']>.95
    assert first['probability']+reverse['probability']==pytest.approx(1)
    assert seq.paired_probability(a,a)['probability'] is None
    with pytest.raises(ValueError,match='calendars'):
        seq.paired_probability(a,b.iloc[1:])


def test_buffer_does_not_use_future_returns():
    rows=pd.DataFrame([dict(id=i,eom=pd.Timestamp(y,12,31),me=100-i,ret_ld1=0.)
                       for y in (2018,2019,2020) for i in range(1,11)])
    a=seq.buffered_membership(rows,5,target_n=3,entry_rank=2)
    b=seq.buffered_membership(rows.assign(ret_ld1=999.),5,target_n=3,entry_rank=2)
    pd.testing.assert_frame_equal(a,b)


def test_rejected_experiments_never_enter_next_candidate(tmp_path,monkeypatch):
    monkeypatch.setattr(seq,'ROOT',tmp_path)
    monkeypatch.setattr(seq,'OUT',tmp_path/'outputs/iteration_summary')
    base=tmp_path/'outputs/base/us_large/portfolio_ml/rff';base.mkdir(parents=True)
    monkeypatch.setattr(seq,'BASE',base)
    for relative in ('portfolio_ml/rff','static_ml_star'):
        folder=tmp_path/'outputs/base/us_large'/relative;folder.mkdir(parents=True,exist_ok=True)
        pf().to_parquet(folder/'pf.parquet')
    for market in ('us_mid','eu_large'):
        folder=tmp_path/f'outputs/base/{market}/portfolio_ml/rff';folder.mkdir(parents=True)
        pf().to_parquet(folder/'pf.parquet')
    market=tmp_path/'outputs/base/overall analysis/market_references/us_large';market.mkdir(parents=True)
    pf().to_parquet(market/'pf.parquet')
    (base/'weights.parquet').write_bytes(b'unchanged')
    (base/'chosen.csv').write_text('year,lambda\n2020,1\n')
    protocol=json.loads(seq.PROTOCOL.read_text())
    file=tmp_path/'protocol.json';file.write_text(json.dumps(protocol))
    monkeypatch.setattr(seq,'PROTOCOL',file)
    calls=[]
    class Fake:
        signature='test'
        cfg=SimpleNamespace(investor=SimpleNamespace(gamma=10.),universe=SimpleNamespace(exit_rank=100),
                            dates=SimpleNamespace(test_start='2020-01-31',test_end='2025-12-31'))
        wealth=None
        def run_candidate(self,state,folder):calls.append(dict(state));return pf(len(calls)*.001)
        def panel_for(self,state,folder):return None
    monkeypatch.setattr(seq,'ExperimentEngine',Fake)
    probabilities=iter([.9,.3,.95,.1])
    monkeypatch.setattr(seq,'paired_probability',lambda *a,**k:dict(probability=next(probabilities),identical=False))
    monkeypatch.setattr(seq,'verify_accounting',lambda *a:None)
    monkeypatch.setattr(seq,'write_iteration_report',lambda *a:None)
    monkeypatch.setattr(seq,'report_saved',lambda *a,**k:0)
    assert seq.main(['--upto','5'])==0
    assert calls[1]['gamma']==5. and calls[1]['exit_rank']==120
    assert calls[2]['gamma']==10. and calls[2]['exit_rank']==100 and calls[2]['cost_multiple']==1.
    assert calls[3]['benchmark_relative'] and calls[3]['half_lives']==[3,5,10,None]
    d=json.loads((tmp_path/'outputs/iteration_4/us_large/decision.json').read_text())
    assert d['inherited_iterations']==[1] and d['included_after']==[1,4]
    d=json.loads((tmp_path/'outputs/iteration_5/us_large/decision.json').read_text())
    assert not d['accepted'] and d['included_after']==[1,4]
    calls.clear()
    monkeypatch.setattr(seq,'paired_probability',lambda *a,**k:dict(probability=.95,identical=False))
    assert seq.main(['--upto','4','--only','4'])==0
    assert len(calls)==1 and calls[0]['gamma']==10. and calls[0]['benchmark_relative']
    assert json.loads((tmp_path/'outputs/iteration_4/us_large/decision.json').read_text())['inherited_iterations']==[1]


def test_frontier_moment_reuse_never_reuses_changed_risk_or_observations():
    from figure1 import moments_compatible
    import copy
    old=dict(config=dict(ridge_grid={'n':100},return_model={},investor={'gamma':10}),
             sources={'/panel.parquet':'a','/predictions.parquet':'b','/portfolio_ml.py':'c'})
    current=copy.deepcopy(old);current['config']['ridge_grid']={'n':200};current['sources']['/predictions.parquet']='new'
    assert moments_compatible(old,current)
    current['config']['investor']['gamma']=5
    assert not moments_compatible(old,current)
    current=copy.deepcopy(old);current['sources']['/panel.parquet']='new'
    assert not moments_compatible(old,current)


def test_filtered_reader_preserves_all_declared_members_and_feature_values():
    engine=seq.ExperimentEngine()
    cfg=engine.cfg
    keys=pd.MultiIndex.from_frame(engine.panel[['id','eom']])
    held=pd.read_parquet(cfg.paths.processed/'panel.parquet',columns=['id','eom'],filters=[('in_universe','=',True)])
    assert pd.MultiIndex.from_frame(held).difference(keys).empty
    for members in engine._memberships.values():
        assert pd.MultiIndex.from_frame(members[['id','eom']]).difference(keys).empty
    date=pd.Timestamp('2019-12-31')
    columns=['id','eom',*cfg.features.names]
    original=pd.read_parquet(cfg.paths.processed/'panel.parquet',columns=columns,filters=[('eom','=',date)])
    selected=engine.panel[engine.panel.eom.eq(date)][columns]
    original=original[original.id.isin(selected.id)].sort_values('id').reset_index(drop=True)
    pd.testing.assert_frame_equal(selected.sort_values('id').reset_index(drop=True),original)
