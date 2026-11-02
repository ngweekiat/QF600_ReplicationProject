"""Offline Kati stages: three base markets and US high-cap Portfolio ML iterations."""
from __future__ import annotations
import argparse,importlib.util,sys
for _stream in (sys.stdout,sys.stderr):
    if hasattr(_stream,"reconfigure"):_stream.reconfigure(encoding="utf-8")
from workflow import ROOT,UNIVERSES,VARIANTS,audit,config_path,covariance,originals_unchanged,prepare,validate_prepared,write_json
from script_groups import GROUPS,VARIANTS as GROUP_VARIANTS,normalise_group,script_path
SCRIPTS={x:x for x in ("03","04","05","06","07","10","11","12","13","14","monitor")}

def calibrated_gamma(universe, variant, target=0.02):
    import pandas as pd
    from jkmp.config import load_config
    owner = "universe_250" if variant in ("universe_250", "halflife") else "active_pfml"
    cfg = load_config(config_path(universe, owner))
    path = cfg.paths.outputs / "active/calibrated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Calibrate this universe first; missing {path}. Never reuse the original gamma_TE.")
    points = pd.read_csv(path)
    selected = points[(points.te_target - target).abs().lt(1e-12)]
    if len(selected) != 1:
        raise ValueError(f"No unique calibration for TE target {target}")
    return float(selected.gamma_te.iloc[0])



def run_stage(stage,universe,variant="base",extra=(),target=.02):
    if stage=="select-characteristics":
        from us_high_cap_selection import main
        return main(list(extra))
    if stage=="dashboard":
        from dashboard import build_dashboard
        return build_dashboard()
    if stage=="verify":
        from verify import main
        return main(list(extra))
    group=normalise_group(variant);variant=GROUP_VARIANTS[group]
    if group!="base" and universe!="us_large" and stage not in ("audit","00","validate","compare","organise","overall","figure1"):
        raise ValueError("Iterations are scoped to US high-cap Portfolio ML only; no other-market output is created.")
    from jkmp.imputation_audit import enable_reporting
    enable_reporting(f"{group} {stage} {universe} te={target:g}")
    if stage=="figure1":
        from figure1 import main
        return main(["--selected-universe",universe,*extra])
    if stage=="overall":
        from comparison_analysis import main
        return main(list(extra))
    if stage=="market":
        from comparison_analysis import build_market_reference
        return build_market_reference(universe)
    if stage in ("audit","00"):return audit()
    if stage=="validate":return validate_prepared()
    if stage in ("prepare","01"):return prepare(universe,variant)
    if stage=="02":return covariance(universe,variant)
    if stage=="15":
        report=dict(universe=universe,available=False,reason="Kati test is 2020-2025; no separate unseen holdout exists.")
        write_json(ROOT/"outputs"/group/universe/"availability/holdout.json",report);print(report["reason"]);return
    if stage=="compare":
        if group!="base":raise ValueError("Use organise for existing iteration results")
        sys.path.insert(0,str(ROOT/"scripts"))
        from analysis.ml_comparison.cli import main
        return main(list(extra))
    if stage=="organise":
        from organise_results import main
        return main(list(extra))
    from jkmp.config import load_config
    cfg_path=config_path(universe,variant);cfg=load_config(cfg_path)
    if stage=="05" and group=="base" and universe!="us_large":
        raise ValueError("Static-ML* is enabled only for US high-cap base")
    required=[cfg.paths.processed/"panel.parquet"]
    if stage!="03":required.append(cfg.paths.processed/"covariance/cov_factor.parquet")
    missing=[str(p) for p in required if not p.exists()]
    if missing:raise FileNotFoundError("Run upstream stages first: "+", ".join(missing))
    if stage in ("04","05","12"):
        from workflow import ensure_prediction_coverage
        ensure_prediction_coverage(cfg)
    script=script_path(stage,group);args=list(extra)
    if stage in ("03","04","05","06","07","10","11","12"):
        args += ["--config",str(cfg_path)]
    if stage=="12" and args and args[0] in ("run","capacity") and not any("investor.gamma_te=" in a for a in args):
        args += ["--override",f"investor.gamma_te={calibrated_gamma(universe,variant,target):g}"]
    sys.argv=[str(script),*args];sys.path.insert(0,str(ROOT/"scripts"))
    if stage=="12":
        import runpy
        return runpy.run_path(str(script),run_name="__main__")
    spec=importlib.util.spec_from_file_location("keats_stage_"+stage,script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if stage=="13":module.CONFIGS={"80-stock":str(config_path(universe,"active_pfml")),"250-stock":str(config_path(universe,"universe_250"))}
    if stage=="14":module.REF=str(config_path(universe,"universe_250"));module.VAR=str(config_path(universe,"halflife"))
    if stage=="monitor":module.SELECTED_CONFIG=config_path(universe,"universe_250")
    module.main()


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("stage",choices=["audit","validate","prepare","00","01","02","15","compare","organise","market","overall","figure1","dashboard","verify","select-characteristics",*SCRIPTS])
    parser.add_argument("--universe",choices=[*UNIVERSES,"all"],default="us_large")
    parser.add_argument("--group",choices=GROUPS,default="base")
    parser.add_argument("--variant",choices=VARIANTS,default=None,help="Legacy alias for a group")
    parser.add_argument("--te-target",type=float,default=.02)
    args,extra=parser.parse_known_args();group=normalise_group(args.variant or args.group)
    if args.stage not in SCRIPTS and args.stage not in ("compare","overall","figure1","organise") and extra:parser.error(f"Unexpected arguments: {extra}")
    universes=UNIVERSES if args.universe=="all" else [args.universe]
    if group!="base" and args.universe=="all":universes=["us_large"]
    if args.stage in ("audit","validate","00","compare","organise","overall","dashboard","verify","select-characteristics"):universes=["us_large"]
    if args.stage=="figure1":universes=[args.universe]
    failed=[]
    for universe in universes:
        try:
            result=run_stage(args.stage,universe,group,extra,args.te_target)
            if isinstance(result,int) and not isinstance(result,bool) and result!=0:failed.append(universe)
        except (FileNotFoundError,ValueError) as error:print(f"STOPPED {universe}/{group}: {error}",file=sys.stderr);failed.append(universe)
    originals_unchanged();return 2 if failed else 0

if __name__=="__main__":raise SystemExit(main())
