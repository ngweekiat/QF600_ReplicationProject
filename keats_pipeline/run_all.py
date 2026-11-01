"""Run the Keats workflow in dependency order with persistent logs and checkpoints.

Examples:
    python keats_pipeline/run_all.py --scope core
    python keats_pipeline/run_all.py --group iteration_3 --scope full
    python keats_pipeline/run_all.py --group iteration_3 --scope full --universe us_large

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


def plan(universes,scope,group="base"):
    from script_groups import normalise_group
    level=3 if group=="all" else ("base","iteration_1","iteration_2","iteration_3").index(normalise_group(group))
    jobs=[Job("audit","audit","all"),Job("prepare","prepare","all",dependencies=("audit",)),Job("validate","validate","all",dependencies=("prepare",))]
    tables=[]
    for u in universes:
        def add(label,stage,variant="base",extra=(),target=.02,deps=()):
            name=f"{u}.{label}";jobs.append(Job(name,stage,u,variant,tuple(extra),target,tuple(deps)));return name
        cov=add("base.covariance","02",deps=("validate",));ret=add("base.returns","03",deps=("validate",))
        static=add("base.static","05",deps=(cov,ret));inputs=add("base.inputs","06",deps=(cov,));pf=add("base.portfolio","07",deps=(inputs,))
        table=add("base.tables","11",deps=(static,pf));tables.append(table)
        if scope=="full":add("base.regimes","10",deps=(table,));add("base.availability","15",deps=(table,))
        if level<1:continue
        market=add("iteration_1.market_reference","04","active_pfml",deps=(cov,ret))
        cal1=add("iteration_1.calibration","12","active_pfml",("calibrate",),deps=(cov,ret,market))
        runs1=[add(f"iteration_1.run_{t:g}","12","active_pfml",("run",),t,deps=(cal1,)) for t in (.01,.02,.04)]
        cap1=add("iteration_1.capacity","12","active_pfml",("capacity",),deps=(cal1,))
        add("iteration_1.report","12","active_pfml",("report",),deps=(*runs1,pf,market))
        if level<2:continue
        prep2=add("iteration_2.prepare","prepare","universe_250")
        cov2=add("iteration_2.covariance","02","universe_250",deps=(prep2,));ret2=add("iteration_2.returns","03","universe_250",deps=(prep2,))
        bench2=add("iteration_2.reference_benchmarks","04","universe_250",deps=(cov2,ret2))
        add("iteration_2.reference_static","05","universe_250",deps=(bench2,))
        baseline=("--override","portfolio_ml.benchmark_relative=false","--override","portfolio_ml.add_constant=true")
        inputs2=add("iteration_2.reference_inputs","06","universe_250",baseline,deps=(cov2,))
        pf2=add("iteration_2.reference_portfolio","07","universe_250",baseline,deps=(inputs2,))
        cal2=add("iteration_2.calibration","12","universe_250",("calibrate",),deps=(cov2,ret2,bench2))
        runs2=[add(f"iteration_2.run_{t:g}","12","universe_250",("run",),t,deps=(cal2,)) for t in (.01,.02,.04)]
        cap2=add("iteration_2.capacity","12","universe_250",("capacity",),deps=(cal2,))
        add("iteration_2.report","12","universe_250",("report",),deps=(*runs2,pf2,bench2))
        add("iteration_2.breadth","13","universe_250",deps=(cap1,cap2,runs1[1],runs2[1]))
        if scope=="full":add("iteration_2.monitor","monitor","universe_250",deps=(runs2[1],bench2,ret2))
        if level<3:continue
        run3=add("iteration_3.run","12","halflife",("run",),deps=(cal2,cov2,ret2))
        add("iteration_3.report","14","halflife",deps=(run3,runs2[1],bench2))
    jobs.append(Job("base.comparison","compare","all",dependencies=tuple(tables)))
    jobs.append(Job("organisation","organise","all"))
    return jobs


def output_paths(job):
    if job.stage in ("audit","validate","15","compare","organise"):return []
    cfg=load_config(config_path(job.universe,job.variant));p,o=cfg.paths.processed,cfg.paths.outputs
    if job.stage=="prepare":return [p/"preparation.json",p/"panel.parquet"]
    if job.stage=="02":return [p/"covariance"/n for n in ("cov_factor.parquet","cov_loadings.parquet","cov_diagnostics.csv","provenance.json")]
    if job.stage=="03":return [p/n for n in ("predictions.parquet","return_model/models.npz","return_model/chosen.csv","return_model/diagnostics.csv")]
    if job.stage=="04":return [o/"benchmarks"/n for n in ("pf_test.parquet","weights_test.parquet","table2_test.csv")]
    if job.stage=="05":return [o/"static_ml"/n for n in (("pf.parquet","weights.parquet","pf_oos.parquet","weights_oos.parquet") if job.variant=="base" else ("pf.parquet","weights.parquet","chosen.csv"))]
    if job.stage=="06":return [p/"pfml"/f"rff_g{g:.4f}"/n for g in cfg.portfolio_ml.rff_variance_grid for n in ("meta.json","2025-11-30.npz")]
    if job.stage=="07":return [o/"portfolio_ml/rff"/n for n in ("pf.parquet","weights.parquet","aim_coefs.parquet","chosen.csv")]
    if job.stage=="10":return [o/"regimes/regime_table.csv"]
    if job.stage=="11":return [o/"tables"/n for n in ("table2.csv","table3.csv","table4.csv","subperiods.csv")]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("core", "full"), default="core")
    parser.add_argument("--universe", choices=(*UNIVERSES, "all"), default="all")
    parser.add_argument("--group",choices=("base","iteration_1","iteration_2","iteration_3","all"),default="base")
    parser.add_argument("--list-jobs",action="store_true",help="Show dependencies without running experiments")
    args = parser.parse_args()
    universes = UNIVERSES if args.universe == "all" else [args.universe]
    jobs = plan(universes, args.scope, args.group)
    if args.list_jobs:
        for job in jobs:print(job.name,job.stage,job.variant,"depends on",job.dependencies)
        return 0
    logs = ROOT / "outputs" / (args.group if args.group!="all" else "organisation_audit") / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = logs / f"{args.group}_{args.scope}_{args.universe}_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
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
        if markers and all(path.is_file() and path.stat().st_size for path in markers):
            statuses[job.name] = "reused"
            state["jobs"][job.name] = {"status": "reused", "outputs": [str(p) for p in markers]}
            save()
            print(f"[{index}/{len(jobs)}] REUSE {job.name}", flush=True)
            continue
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log = logs / f"{job.name}_{stamp}.log"
        command = [sys.executable, "-u", str(ROOT / "run.py"), job.stage, "--universe", job.universe,
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
