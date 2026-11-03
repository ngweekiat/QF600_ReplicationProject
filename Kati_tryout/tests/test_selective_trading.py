"""Learning-cost preference, pretest selection, and true-cost accounting separation."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jkmp.portfolio.portfolio_ml import build_moments
import selective_trading as selective


def test_training_penalty_changes_only_cost_part_of_learning_moments():
    from test_portfolio_math import _panel
    panel,wealth,cov,rf,dates,features=_panel()
    args=(panel,wealth,cov,rf,dates[3:],features)
    kw=dict(gamma=10.,mu=.007,lb=2,iterations=10,use_rff=False,add_constant=False)
    original=build_moments(*args,**kw)
    control=build_moments(*args,**kw,training_cost_multiplier=1.)
    stronger=build_moments(*args,**kw,training_cost_multiplier=2.)
    for d,first in original.months.items():
        a,b=control.months[d],stronger.months[d]
        np.testing.assert_array_equal(first.denom,a.denom)
        np.testing.assert_allclose(b.denom,first.risk+2*(first.denom-first.risk),atol=1e-12)
        np.testing.assert_array_equal(first.r_tilde,b.r_tilde)
        np.testing.assert_array_equal(first.risk,b.risk)
        np.testing.assert_array_equal(first.signal,b.signal)
    with pytest.raises(ValueError,match='>= 1'):
        build_moments(*args,**kw,training_cost_multiplier=.5)


def test_pretest_choice_uses_utility_and_conservative_tie_break():
    rows=[dict(training_cost_multiplier=c,validation_utility=u) for c,u in [(2.,.01),(1.,.01),(1.5,.02)]]
    assert selective.choose_trial(rows).training_cost_multiplier==1.5
    rows[2]['validation_utility']=.01
    assert selective.choose_trial(rows).training_cost_multiplier==1.


def test_master_charts_support_all_six_iterations_plus_base(tmp_path):
    from comparison_analysis import charts
    from analysis.ml_comparison.metrics import statistics
    dates=pd.date_range('2020-01-31',periods=30,freq='ME')
    frames=[pd.DataFrame(dict(dataset='all',strategy=f'Model {i}',eom_ret=dates,
        r=.005+.01*np.sin(np.arange(len(dates))+i),tc=.0001,turnover=.1,inv=1.,shorting=0.))
        for i in range(11)]
    monthly=pd.concat(frames,ignore_index=True)
    stats=pd.DataFrame([dict(strategy=f.strategy.iloc[0],**statistics(f)) for f in frames])
    charts(monthly,stats,tmp_path)
    assert (tmp_path/'charts/rolling_and_drawdown.png').exists()


def test_selection_is_frozen_before_test_and_actual_costs_stay_one(tmp_path,monkeypatch):
    import iteration_sequence as seq
    monkeypatch.setattr(selective,'ROOT',tmp_path)
    monkeypatch.setattr(seq,'ROOT',tmp_path)
    math=tmp_path/'scripts/jkmp/portfolio/portfolio_ml.py';math.parent.mkdir(parents=True)
    math.write_text('fake version')
    original=tmp_path/'outputs/iteration_4/us_large';original.mkdir(parents=True)
    pd.DataFrame({'gamma_te':[174.1]}).to_csv(original/'calibrated.csv',index=False)
    folder=tmp_path/'outputs/iteration_6/us_large';folder.mkdir(parents=True)
    class Fake:
        signature='constant-source'
        cfg=SimpleNamespace(dates=SimpleNamespace(validation_end='2019-12-31'))
        def __init__(self):self.test_calls=0
        def fit(self,state,path,*,calibration=False,gamma_te=None,validation_result=False):
            assert state['cost_multiple']==1.
            if calibration and not validation_result:
                return dict(gamma_te=gamma_te,te_ex_ante=1.65/gamma_te,months=48,last_return='2019-12-31')
            if not calibration:
                self.test_calls+=1
                chosen=json.loads((folder/'pretest_selection.json').read_text())
                assert chosen['selected_multiplier']==1.5 and not chosen['test_returns_used']
            dates=pd.date_range('2016-01-31' if calibration else '2020-01-31',periods=48 if calibration else 72,freq='ME')
            mean={1.:.005,1.5:.008,2.:.006}[state['training_cost_multiplier']]
            pf=pd.DataFrame(dict(eom_ret=dates,r=mean+.01*np.sin(np.arange(len(dates))),tc=.0001,turnover=.1,inv=1.,shorting=0.))
            result=dict(pf=pf,weights=pd.DataFrame({'w':[1.]}),chosen=pd.DataFrame({'year':[2016]}))
            if not calibration:selective.save_result(result,path)
            return result
    engine=Fake()
    state=dict(benchmark_relative=True,te_target=.01,cost_multiple=1.,training_cost_multiplier=1.)
    spec=dict(training_cost_candidates=[1.,1.5,2.])
    result=selective.run_selective(engine,state,folder,spec)
    assert state['training_cost_multiplier']==1.5 and engine.test_calls==1
    assert result.tc.eq(.0001).all()
    trial=pd.read_csv(folder/'cost_preference_validation.csv')
    assert pd.to_datetime(trial.last_return).max()==pd.Timestamp('2019-12-31')
    assert trial.actual_cost_multiple.eq(1.).all()
