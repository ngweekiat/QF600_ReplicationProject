"""Authoritative mapping from run groups to script entry points."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
GROUPS=("base","iteration_1","iteration_2","iteration_3")
VARIANTS={"base":"base","iteration_1":"active_pfml","iteration_2":"universe_250","iteration_3":"halflife"}
ALIASES={**{x:x for x in GROUPS},**{v:k for k,v in VARIANTS.items()}}


def normalise_group(value):
    if value not in ALIASES:raise ValueError(f"Unknown group: {value}")
    return ALIASES[value]


def baseline_reference_dir(cfg):
    """Route reference portfolios without copying base results into iteration reports."""
    output=cfg.paths.outputs
    group=output.parent.name
    if group=='iteration_1':return ROOT/'outputs/base'/output.name
    if group=='iteration_3':return ROOT/'outputs/iteration_2'/output.name
    return output


def script_path(stage,group):
    group=normalise_group(group)
    common={"03":"shared/03_return_model.py","04":"shared/04_benchmarks.py"}
    base={"05":"base/05_static_ml.py","06":"shared/06_pfml_inputs.py","07":"base/07_portfolio_ml.py","10":"base/10_regimes.py","11":"base/11_tables.py"}
    if stage=="04":
        if group=="base":raise ValueError("Market is an iteration reference; base reports contain only the two ML strategies")
        return ROOT/"scripts"/("iteration_2/reference_benchmarks.py" if group=="iteration_2" else common[stage])
    if stage in common:return ROOT/"scripts"/common[stage]
    if group=="base" and stage in base:return ROOT/"scripts"/base[stage]
    if stage=="05" and group=="iteration_2":return ROOT/"scripts/iteration_2/reference_static_ml.py"
    if stage=="06" and group=="iteration_2":return ROOT/"scripts/shared/06_pfml_inputs.py"
    if stage=="07" and group=="iteration_2":return ROOT/"scripts/iteration_2/reference_portfolio_ml.py"
    if stage=="12" and group!="base":return ROOT/"scripts"/group/"run.py"
    if stage=="13" and group=="iteration_2":return ROOT/"scripts/iteration_2/compare_breadth.py"
    if stage=="14" and group=="iteration_3":return ROOT/"scripts/iteration_3/compare_halflife.py"
    if stage=="monitor" and group=="iteration_2":return ROOT/"scripts/iteration_2/monitor.py"
    raise ValueError(f"Stage {stage} is not part of {group}")
