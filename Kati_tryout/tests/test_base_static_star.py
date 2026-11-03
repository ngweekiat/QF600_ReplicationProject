"""Base scope, lagged annual tuning, and independently reconstructed Static ML* trades."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from jkmp.covariance import CovParts
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.static_ml_star import hyper_grid, run_static_ml_star, select_hypers, static_weights, validate
from jkmp.strategies import base_strategies
from run_all import plan
from workflow import UNIVERSES


def example():
    dates=pd.date_range("2018-12-31",periods=25,freq="ME")
    rng=np.random.default_rng(91)
    rows=[];cov={}
    for i,date in enumerate(dates):
        ids=np.arange(1,4)
        cov[date]=CovParts(date,ids,("f",),np.ones((3,1)),np.array([[.0003]]),np.array([.002,.003,.004]))
        for stock in ids:
            # Membership change forces a real exit and a new position.
            rows.append(dict(id=stock,eom=date,me=100.+stock,**{"lambda":1e-10*stock},
                pred_ld1=.004*stock,ret_ld1=rng.normal(.005,.03),tr_ld1=rng.normal(.006,.03),
                in_universe=stock in ((1,2) if i<13 else (1,3))))
    panel=pd.DataFrame(rows)
    wealth=pd.DataFrame(dict(eom=dates,wealth=1e9,mu_ld1=.006))
    return dates,panel,wealth,cov


@pytest.mark.parametrize("charge_exit",[False,True])
def test_batched_validation_matches_independent_recursive_accounting(charge_exit):
    dates,panel,wealth,cov=example()
    inputs=month_inputs(panel,wealth,cov,dates)
    grid=hyper_grid([1.,.2],[0.,.5,1.],[0.,2.])
    observed=validate(inputs,panel,wealth,10.,grid,charge_exit)
    fields=["r","tc","turnover","inv","shorting"]
    for i,hp in enumerate(grid):
        weights=static_weights(inputs,panel,wealth,10.,lambda date:hp,charge_exit)
        expected=portfolio_returns(weights,panel,wealth).sort_values("eom")
        actual=observed.loc[observed.hp_no.eq(i)].sort_values("eom")
        np.testing.assert_allclose(actual[fields],expected[fields],rtol=1e-11,atol=1e-13)


def test_future_outcomes_do_not_change_prior_year_choice_or_weights():
    dates,panel,wealth,cov=example();grid=hyper_grid([1.,.2],[.25,1.],[0.,2.])
    evaluation=dates[(dates+pd.offsets.MonthEnd(1)).year==2020]
    first=run_static_ml_star(panel,wealth,cov,dates,evaluation,10.,grid,True)
    changed=panel.copy()
    future=(changed.eom+pd.offsets.MonthEnd(1)).dt.year>=2020
    changed.loc[future,"ret_ld1"]=-2.
    second=run_static_ml_star(changed,wealth,cov,dates,evaluation,10.,grid,True)
    pd.testing.assert_frame_equal(first["weights"],second["weights"])
    pd.testing.assert_frame_equal(first["chosen"].query("year<=2020"),second["chosen"].query("year<=2020"))
    assert set(first["pf"].type)=={"Static-ML*"}
    assert first["chosen"].eom_ret.dt.year.add(1).eq(first["chosen"].year).all()


def test_december_choice_matches_literal_paper_cumulative_objective():
    _,panel,wealth,cov=example();dates=sorted(cov);grid=hyper_grid([1.,.2],[.25,1.],[0.,2.])
    validation=validate(month_inputs(panel,wealth,cov,dates),panel,wealth,10.,grid,True)
    chosen=select_hypers(validation,10.)
    for row in chosen.itertuples():
        scores=[]
        for i in range(len(grid)):
            d=validation.loc[validation.hp_no.eq(i)&validation.eom_ret.le(row.eom_ret)].sort_values("eom_ret")
            variance=[np.mean(d.r.iloc[:n]**2)-np.mean(d.r.iloc[:n])**2 for n in range(1,len(d)+1)]
            scores.append(np.mean(d.r-d.tc-5*np.array(variance)))
        assert row.hp_no==int(np.argmax(scores))
        assert row.cum_obj==pytest.approx(max(scores))


def test_base_scope_and_iterations_are_separate():
    jobs=plan(UNIVERSES,"full","all")
    assert [j.universe for j in jobs if j.stage=="05"]==["us_large"]
    assert base_strategies("us_large")==("Static-ML*","Portfolio-ML")
    assert base_strategies("us_mid")==base_strategies("eu_large")==("Portfolio-ML",)
    assert len([j for j in jobs if j.stage=="experiment"])==10
    from comparison_analysis import base_sources,iteration_baseline_sources
    sources=base_sources()
    assert len([s for s in sources if s["expected_type"]=="Static-ML*"])==1
    assert not any(s["expected_type"]=="Static-ML" for s in sources)
    assert any(s["expected_type"]=="Static-ML*" for s in iteration_baseline_sources())
    from jkmp.config import load_config
    from workflow import config_path
    for i in range(1,11):
        cfg=load_config(config_path("us_large",f"iteration_{i}"))
        assert cfg.reporting.main_types==("Static-ML*","Portfolio-ML")
        assert cfg.static_ml.k_grid==(1.,1/3,.2,.1,.05)
        assert cfg.static_ml.g_grid==(0.,1.,2.,3.)


def test_portfolio_cache_reuse_rejects_changed_data_or_model_settings():
    from figure1 import portfolio_compatible
    import copy
    old=dict(config=dict(reporting={},static_ml={},regimes={},portfolio_ml={"p_grid":[8]}),sources={"panel.parquet":"aaa","/static_ml.py":"old","/portfolio_ml.py":"bbb"})
    new=copy.deepcopy(old);new["config"]["reporting"]={"main_types":["Portfolio-ML"]}
    new["sources"].pop("/static_ml.py");new["sources"]["/static_ml_star.py"]="new"
    assert portfolio_compatible(old,new)
    for part in ("data","model"):
        changed=copy.deepcopy(new)
        if part=="data":changed["sources"]["panel.parquet"]="changed"
        else:changed["config"]["portfolio_ml"]["p_grid"]=[16]
        assert not portfolio_compatible(old,changed)


def test_combined_figure_contains_only_all_retained_frontiers(tmp_path,monkeypatch):
    import figure1
    monkeypatch.setattr(figure1,"OUT",tmp_path)
    monkeypatch.setattr(figure1,"provenance",lambda universe:{"signature":"verified"})
    monkeypatch.setattr(figure1,"load_config",lambda path:SimpleNamespace(investor=SimpleNamespace(gamma=10.,wealth=1e10)))
    monkeypatch.setattr(figure1,"market_dir",lambda universe:tmp_path/universe/"market")
    for universe in UNIVERSES:
        folder=tmp_path/universe;folder.mkdir()
        (folder/"market").mkdir()
        pf=pd.DataFrame(dict(r=[.01,.02,-.01],tc=[.001]*3,turnover=[.1]*3,inv=[1.]*3,shorting=[0.]*3))
        pf.to_parquet(folder/"market/pf.parquet",index=False)
        rows=[dict(type=method,gamma=g,wealth=w,Vol=.2/g**.5,R_TC=.1/g**.5)
              for method,_ in figure1.frontier_methods(universe) for w in (0.,1e10) for g in (1.,10.,100.)]
        pd.DataFrame(rows).to_csv(folder/"frontier_summary.csv",index=False)
        (folder/"completion.json").write_text(json.dumps(dict(signature="verified")))
    assert figure1.plot_combined()
    result=pd.read_csv(tmp_path/"combined_frontier_summary.csv")
    assert len(result)==24
    assert set(result.loc[result.type.eq("Static-ML*"),"universe"])=={"us_large"}
    assert (tmp_path/"charts/figure1_combined.pdf").stat().st_size>1000
