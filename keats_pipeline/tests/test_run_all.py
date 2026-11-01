from run_all import Job,output_paths,plan
from workflow import UNIVERSES,VARIANTS,config_path
from script_groups import GROUPS,script_path
import pytest


def test_grouped_plans_keep_dependencies_in_order_and_base_does_not_run_iterations():
    base=plan(UNIVERSES,"full","base")
    assert all(x.variant=="base" for x in base)
    assert "04" not in {x.stage for x in base}
    for group in (*GROUPS,"all"):
        previous=set()
        for job in plan(UNIVERSES,"full",group):
            assert job.name not in previous and set(job.dependencies)<=previous
            previous.add(job.name)
            if job.stage in ("03","04","05","06","07","10","11","12","13","14","monitor"):
                assert script_path(job.stage,job.variant).exists()
    assert set(VARIANTS)=={"base","active_pfml","universe_250","halflife"}


def test_static_cache_requires_its_actual_fixed_weights_and_continuous_history():
    paths=output_paths(Job("static","05","us_large"))
    assert {p.name for p in paths}=={"pf.parquet","weights.parquet","pf_oos.parquet","weights_oos.parquet"}


def test_unknown_variant_config_is_rejected():
    with pytest.raises(ValueError):config_path("us_large","removed_variant")


def test_base_market_and_active_stages_cannot_write_into_base():
    with pytest.raises(ValueError):script_path("04","base")
    with pytest.raises(ValueError):script_path("12","base")


def test_iteration_references_stay_in_the_correct_run_group():
    from script_groups import baseline_reference_dir
    from jkmp.config import load_config
    for universe in UNIVERSES:
        base=load_config(config_path(universe,"base")).paths.outputs
        second=load_config(config_path(universe,"iteration_2")).paths.outputs
        assert baseline_reference_dir(load_config(config_path(universe,"iteration_1")))==base
        assert baseline_reference_dir(load_config(config_path(universe,"iteration_2")))==second
        assert baseline_reference_dir(load_config(config_path(universe,"iteration_3")))==second


def test_breadth_te_argument_is_forwarded_without_becoming_te_target(monkeypatch):
    import run
    calls=[]
    monkeypatch.setattr(run.sys,'argv',['run.py','13','--group','iteration_2','--te','0.01'])
    monkeypatch.setattr(run,'run_stage',lambda *args:calls.append(args))
    monkeypatch.setattr(run,'originals_unchanged',lambda:True)
    assert run.main()==0
    assert calls[0][2]=='iteration_2'
    assert calls[0][3]==['--te','0.01']
