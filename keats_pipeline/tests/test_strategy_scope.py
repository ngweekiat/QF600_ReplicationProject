from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest

from workflow import ROOT,UNIVERSES,config_path
from jkmp.config import ConfigError,load_config
from jkmp.portfolio.portfolio_ml import YearSums,coefs_from_sums,select_hypers,model_columns
from jkmp.strategies import STRATEGIES,require_strategies


def test_each_base_market_has_two_methods_and_separate_iteration_configs():
    for universe in UNIVERSES:
        cfg=load_config(config_path(universe))
        assert cfg.reporting.main_types==STRATEGIES
        assert cfg.regimes.plot_types==STRATEGIES
        assert sorted(p.name for p in (ROOT/'configs'/universe).glob('*.yaml'))==['base.yaml','iteration_1.yaml','iteration_2.yaml','iteration_3.yaml']
        assert tuple(cfg.features.names)==tuple(json.loads((ROOT/'feature_names.json').read_text()))
        with pytest.raises(ConfigError):load_config(config_path(universe),['portfolio_ml.benchmark_relative=true'])


def test_public_result_guard_rejects_extra_methods_and_duplicate_months():
    frame=pd.DataFrame({'type':['Static-ML'],'eom_ret':pd.to_datetime(['2020-01-31'])})
    with pytest.raises(ValueError):require_strategies(frame,complete=True)
    with pytest.raises(ValueError):require_strategies(pd.concat([frame,frame],ignore_index=True))
    with pytest.raises(ValueError):require_strategies(frame.assign(type='unsupported'))


def test_portfolio_ml_refits_use_only_earlier_return_years():
    owner=SimpleNamespace(columns=['constant','cos','sin'],meta={'use_rff':True,'p_max':2,'add_orig':False,'add_constant':True})
    sums={y:YearSums(12,np.array([.1,.2,.3])*12,np.eye(3)*12) for y in range(2010,2022)}
    original=coefs_from_sums(sums,owner,[2],[.1],[2015,2021])
    changed=dict(sums);changed[2020]=YearSums(12,np.array([.9,.8,.7])*12,np.eye(3)*12)
    altered=coefs_from_sums(changed,owner,[2],[.1],[2015,2021])
    np.testing.assert_array_equal(original[2015][2],altered[2015][2])
    assert not np.allclose(original[2021][2],altered[2021][2])
    changed[2021]=YearSums(12,np.array([999.,999.,999.]),np.eye(3)*12)
    future=coefs_from_sums(changed,owner,[2],[.1],[2021])
    np.testing.assert_array_equal(altered[2021][2],future[2021][2])


def test_portfolio_ml_choice_is_used_in_following_return_year():
    v=pd.DataFrame({'eom_ret':pd.to_datetime(['2019-12-31','2019-12-31','2020-12-31']),
        'cum_obj':[.1,.2,.3],'tag':['a','b','a'],'g':[.05,.1,.05],'p':[2,2,2],'l_no':[0,0,0],'l':[.1,.1,.1]})
    chosen=select_hypers(v)
    assert chosen.year.tolist()==[2020,2021]
    assert chosen.tag.tolist()==['b','a']


def test_base_rejects_experimental_options_while_iterations_keep_them():
    with pytest.raises(ConfigError):load_config(config_path("us_large"),["portfolio_ml.use_rff=false"])
    for variant in ("active_pfml","universe_250","halflife"):
        cfg=load_config(config_path("us_large",variant))
        assert cfg.portfolio_ml.benchmark_relative
    assert load_config(config_path("us_large","halflife")).portfolio_ml.half_life_years==(3.,5.,10.,None)
