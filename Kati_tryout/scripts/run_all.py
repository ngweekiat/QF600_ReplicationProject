"""Run the Kati workflow in dependency order with persistent logs and checkpoints.

Examples:
    python Kati_tryout/scripts/run_all.py --scope core
    python Kati_tryout/scripts/run_all.py --group iteration_3 --scope full
    python Kati_tryout/scripts/run_all.py --group iteration_3 --scope full --universe us_large

Completed outputs are retained. Failed jobs block their dependants; independent
universes/stages continue. Rerunning resumes incomplete work. Nothing is committed,
pushed or downloaded by this runner.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from workflow import ROOT, UNIVERSES, config_path, originals_unchanged
from jkmp.config import load_config


@dataclass
class Job:
    name: str
    stage: str
    universe: str
    variant: str = "base"
    extra: tuple[str, ...] = ()
    target: float = 0.02
    dependencies: tuple[str, ...] = ()


def plan(universes,scope,group="base",skip_figure1=False):
    from script_groups import normalise_group
    level=3 if group=="all" else ("base","iteration_1","iteration_2","iteration_3").index(normalise_group(group))
    jobs=[Job("audit","audit","all"),Job("prepare","prepare","all",dependencies=("audit",)),Job("validate","validate","all",dependencies=("prepare",))]
    tables=[]
    for u in universes:
        def add(label,stage,variant="base",extra=(),target=.02,deps=()):
            name=f"{u}.{label}";jobs.append(Job(name,stage,u,variant,tuple(extra),target,tuple(deps)));return name
        cov=add("base.covariance","02",deps=("validate",));ret=add("base.returns","03",deps=("validate",))
        static=add("base.static","05",deps=(cov,ret)) if u=="us_large" else None
        inputs=add("base.inputs","06",deps=(cov,));pf=add("base.portfolio","07",deps=(inputs,))
        table=add("base.tables","11",deps=tuple(x for x in (static,pf) if x));tables.append(table)
        if scope=="full":add("base.regimes","10",deps=(table,));add("base.availability","15",deps=(table,))
        add("base.market_reference","market",deps=(cov,))
        if level<1 or u!="us_large":continue
        market=add("iteration_1.market_reference","04","active_pfml",deps=(cov,ret))
        cal1=add("iteration_1.calibration","12","active_pfml",("calibrate",),deps=(cov,ret,market))
        runs1=[add(f"iteration_1.run_{t:g}","12","active_pfml",("run",),t,deps=(cal1,)) for t in (.01,.02,.04)]
        cap1=add("iteration_1.capacity","12","active_pfml",("capacity",),deps=(cal1,))
        add("iteration_1.report","12","active_pfml",("report",),deps=(*runs1,pf,market))
        add("iteration_1.overall_analysis","overall","active_pfml",("--only-group","iteration_1"),deps=(*runs1,pf,static))
        if level<2:continue
        prep2=add("iteration_2.prepare","prepare","universe_250")
        cov2=add("iteration_2.covariance","02","universe_250",deps=(prep2,));ret2=add("iteration_2.returns","03","universe_250",deps=(prep2,))
        bench2=add("iteration_2.reference_benchmarks","04","universe_250",deps=(cov2,ret2))
        cal2=add("iteration_2.calibration","12","universe_250",("calibrate",),deps=(cov2,ret2,bench2))
        runs2=[add(f"iteration_2.run_{t:g}","12","universe_250",("run",),t,deps=(cal2,)) for t in (.01,.02,.04)]
        cap2=add("iteration_2.capacity","12","universe_250",("capacity",),deps=(cal2,))
        add("iteration_2.overall_analysis","overall","universe_250",("--only-group","iteration_2"),deps=(*runs2,runs1[1],pf,static))
        add("iteration_2.breadth","13","universe_250",deps=(cap1,cap2,runs1[1],runs2[1]))
        if scope=="full":add("iteration_2.monitor","monitor","universe_250",deps=(runs2[1],bench2,ret2))
        if level<3:continue
        run3=add("iteration_3.run","12","halflife",("run",),deps=(cal2,cov2,ret2))
        add("iteration_3.report","14","halflife",deps=(run3,runs2[1],bench2))
        add("iteration_3.overall_analysis","overall","halflife",("--only-group","iteration_3"),deps=(run3,runs1[1],runs2[1],pf,static))
        add("final_comparison","overall","halflife",("--only-group","final"),deps=(run3,runs1[1],runs2[1],pf))
    jobs.append(Job("base.comparison","compare","all",dependencies=tuple(tables)))
    if set(universes)==set(UNIVERSES):
        jobs.append(Job("base.overall_analysis","overall","all",extra=("--only-group","base"),dependencies=tuple(tables)+tuple(f"{u}.base.market_reference" for u in universes)))
        if level==3:
            jobs.append(Job("overall.master_comparison","overall","all",extra=("--only-group","master"),dependencies=("base.overall_analysis","us_large.final_comparison")))
    if scope=="full" and not skip_figure1:
        chosen="all" if set(universes)==set(UNIVERSES) else universes[0]
        extras=("--jobs","3")+(("--include-iterations",) if level==3 and "us_large" in universes else ())
        deps=tuple(tables)+(("us_large.final_comparison",) if level==3 and "us_large" in universes else ())
        jobs.append(Job("figure1.frontiers","figure1",chosen,extra=extras,dependencies=deps))
    jobs.append(Job("organisation","organise","all",extra=("--only-group","base") if group=="base" else ()))
    jobs.append(Job("dashboard","dashboard","all"))
    return jobs


def output_paths(job):
    if job.stage in ("audit","validate","15","compare","organise","overall","figure1"):return []
    cfg=load_config(config_path(job.universe,job.variant));p,o=cfg.paths.processed,cfg.paths.outputs
    if job.stage=="prepare":return [p/"preparation.json",p/"panel.parquet"]
    if job.stage=="02":return [p/"covariance"/n for n in ("cov_factor.parquet","cov_loadings.parquet","cov_diagnostics.csv","provenance.json")]
    if job.stage=="03":
        from jkmp.models.return_model import tuning_marker
        return [p/n for n in ("predictions.parquet","return_model/models.npz","return_model/chosen.csv","return_model/diagnostics.csv")]+[tuning_marker(cfg)]
    if job.stage=="04":return [o/"benchmarks"/n for n in ("pf_test.parquet","weights_test.parquet","table2_test.csv")]
    if job.stage=="market":return [ROOT/"outputs/base/overall analysis/market_references"/job.universe/n for n in ("pf.parquet","weights.parquet","provenance.json")]
    if job.stage=="05":
        from jkmp.portfolio.static_ml_star import tuning_grid_marker
        return [o/"static_ml_star"/n for n in ("pf.parquet","weights.parquet","validation.parquet","chosen.csv")]+[tuning_grid_marker(cfg)]
    if job.stage=="06":return [p/"pfml"/f"rff_g{g:.4f}"/n for g in cfg.portfolio_ml.rff_variance_grid for n in ("meta.json","2025-11-30.npz")]
    if job.stage=="07":
        from jkmp.strategies import portfolio_tuning_marker
        return [o/"portfolio_ml/rff"/n for n in ("pf.parquet","weights.parquet","aim_coefs.parquet","chosen.csv")]+[portfolio_tuning_marker(cfg)]
    if job.stage in ("10","11"):
        from jkmp.strategies import base_sources_marker
        folder="regimes" if job.stage=="10" else "tables"
        names=("regime_table.csv","model_scope.json") if job.stage=="10" else ("table2.csv","table3.csv","table4.csv","subperiods.csv","model_scope.json")
        return [o/folder/n for n in names]+[base_sources_marker(o,folder)]
    if job.stage=="13":return [o/"breadth/capacity_aum.csv"]
    if job.stage=="14":return [o/"halflife/summary.csv"]
    if job.stage=="monitor":return [o/"monitor/monitor.csv"]
    if job.stage=="12":
        if job.extra[0]=="calibrate":return [o/"active/calibrated.csv"]
        if job.extra[0]=="report":return [o/"active/summary.csv"]
        from run import calibrated_gamma
        try:g=calibrated_gamma(job.universe,job.variant,job.target)
        except (FileNotFoundError,ValueError):return []
        if job.extra[0]=="run":return [o/"active"/f"gte{g:g}"/n for n in ("pf.parquet","weights.parquet","chosen.csv")]
        return [o/"active/capacity"/f"gte{g:g}_w{w/1e9:g}B/pf.parquet" for w in cfg.investor.wealth_grid]
    return []


def cached_outputs_ready(paths):
    """A historical marker cannot validate files overwritten by a later run.

    This matters when restoring an earlier parameter choice: its old marker
    may still exist even though the shared output files describe another fit.
    Versioned completion markers are written after their output files.
    """
    if not paths or not all(p.is_file() and p.stat().st_size for p in paths):
        return False
    marker=paths[-1]
    if marker.name.startswith(("tuning_","sources_")):
        return marker.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in paths[:-1])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("core", "full"), default="core")
    parser.add_argument("--universe", choices=(*UNIVERSES, "all"), default="all")
    parser.add_argument("--group",choices=("base","iteration_1","iteration_2","iteration_3","all"),default="base")
    parser.add_argument("--list-jobs",action="store_true",help="Show dependencies without running experiments")
    parser.add_argument("--skip-figure1",action="store_true",help="Run full diagnostics without fitting the separate frontier grid")
    args = parser.parse_args()
    if args.group!="base" and args.universe not in ("all","us_large"):
        parser.error("Iterations apply only to US high-cap Portfolio ML. Use --universe all for all base markets plus US high-cap iterations.")
    universes = UNIVERSES if args.universe == "all" else [args.universe]
    jobs = plan(universes, args.scope, args.group,args.skip_figure1)
    if args.list_jobs:
        for job in jobs:print(job.name,job.stage,job.variant,"depends on",job.dependencies)
        return 0
    logs = ROOT / "outputs" / (args.group if args.group!="all" else "organisation_audit") / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = logs / f"{args.group}_{args.scope}_{args.universe}_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
    names={job.name for job in jobs}
    historical={k:v for k,v in state["jobs"].items() if k not in names}
    if historical:state.setdefault("historical_out_of_scope_jobs",{}).update(historical)
    state["jobs"]={k:v for k,v in state["jobs"].items() if k in names}
    state["iteration_scope"]="us_large Portfolio ML only; base retains selected markets"
    statuses = {}
    def save():
        state["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        temporary = state_path.with_suffix(".writing.json")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                       OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for index, job in enumerate(jobs, 1):
        unmet = [name for name in job.dependencies if statuses.get(name) not in ("complete", "reused")]
        if unmet:
            statuses[job.name] = "blocked"
            state["jobs"][job.name] = {"status": "blocked", "dependencies": unmet}
            save()
            print(f"[{index}/{len(jobs)}] BLOCKED {job.name}: {', '.join(unmet)}", flush=True)
            continue
        markers = output_paths(job) if job.universe != "all" else []
        if cached_outputs_ready(markers):
            statuses[job.name] = "reused"
            state["jobs"][job.name] = {"status": "reused", "outputs": [str(p) for p in markers]}
            save()
            print(f"[{index}/{len(jobs)}] REUSE {job.name}", flush=True)
            continue
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = logs / f"{job.name}_{stamp}.log"
        command = [sys.executable, "-u", str(ROOT / "scripts/run.py"), job.stage, "--universe", job.universe,
                   "--variant", job.variant, "--te-target", str(job.target), *job.extra]
        if job.stage=="compare":command += ["--selected-universe",args.universe]
        started = time.monotonic()
        state["jobs"][job.name] = {"status": "running", "command": command, "log": str(log)}
        save()
        print(f"[{index}/{len(jobs)}] RUN {job.name} -> {log.name}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        status = "complete" if process.returncode == 0 else "failed"
        statuses[job.name] = status
        state["jobs"][job.name].update(status=status, exit_code=process.returncode,
                                          seconds=round(time.monotonic() - started, 2))
        save()
        print(f"[{index}/{len(jobs)}] {status.upper()} {job.name} ({time.monotonic()-started:.1f}s)", flush=True)
        if process.returncode:
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]), flush=True)
    originals_unchanged()
    failed = [name for name, value in statuses.items() if value in ("failed", "blocked")]
    print(f"Run finished: {len(jobs)-len(failed)}/{len(jobs)} jobs complete/reused; state: {state_path}", flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
