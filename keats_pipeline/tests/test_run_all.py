from run_all import Job, output_paths, plan
from workflow import UNIVERSES


def test_full_plan_dependencies_exist_before_their_jobs():
    jobs = plan(UNIVERSES, "full")
    assert len(jobs) == 114
    previous = set()
    for job in jobs:
        assert job.name not in previous
        assert set(job.dependencies) <= previous
        previous.add(job.name)


def test_grid_cache_checks_the_selected_signal_variant_not_just_shared_summary():
    rff = output_paths(Job("rff", "08", "us_large"))
    linear = output_paths(Job("linear", "08", "us_large", "linear_pfml"))
    assert len(rff) == len(linear) == 26
    assert rff[0] == linear[0]
    assert all(p.name == "pf_portfolio_ml_rff.parquet" for p in rff[1:])
    assert all(p.name == "pf_portfolio_ml_raw.parquet" for p in linear[1:])
