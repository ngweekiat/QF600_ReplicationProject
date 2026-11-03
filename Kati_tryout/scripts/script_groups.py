"""Authoritative mapping from run groups to script entry points."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
GROUPS=("base",*(f"iteration_{i}" for i in range(1,11)))
VARIANTS={group:group for group in GROUPS}
ALIASES={**{x:x for x in GROUPS},**{v:k for k,v in VARIANTS.items()}}


def normalise_group(value):
    if value not in ALIASES:raise ValueError(f"Unknown group: {value}")
    return ALIASES[value]


def baseline_reference_dir(cfg):
    """Route reference portfolios without copying base results into iteration reports."""
    output=cfg.paths.outputs
    group=output.parent.name
    return ROOT/'outputs/base'/output.name


def script_path(stage,group):
    group=normalise_group(group)
    common={"03":"shared/03_return_model.py","04":"shared/04_benchmarks.py"}
    base={"05":"base/05_static_ml.py","06":"shared/06_pfml_inputs.py","07":"base/07_portfolio_ml.py","10":"base/10_regimes.py","11":"base/11_tables.py"}
    if stage=="04":
        if group=="base":raise ValueError("Market is an iteration reference; base reports contain only the two ML strategies")
        return ROOT/"scripts"/common[stage]
    if stage in common:return ROOT/"scripts"/common[stage]
    if group=="base" and stage in base:return ROOT/"scripts"/base[stage]
    if stage=="06" and group=="iteration_2":return ROOT/"scripts/shared/06_pfml_inputs.py"
    if stage in ("12","experiment") and group!="base":return ROOT/"scripts"/group/"run.py"
    if stage=="13" and group=="iteration_2":return ROOT/"scripts/iteration_2/compare_breadth.py"
    if stage=="14" and group=="iteration_3":return ROOT/"scripts/iteration_3/compare_halflife.py"
    if stage=="monitor" and group=="iteration_2":return ROOT/"scripts/iteration_2/monitor.py"
    raise ValueError(f"Stage {stage} is not part of {group}")
