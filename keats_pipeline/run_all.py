"""Run the Keats workflow in dependency order with persistent logs and checkpoints.

Examples:
    python keats_pipeline/run_all.py --scope core
    python keats_pipeline/run_all.py --scope full
    python keats_pipeline/run_all.py --scope full --universe us_large

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


def plan(universes, scope):
    jobs = [Job("audit", "audit", "all"), Job("prepare", "prepare", "all", dependencies=("audit",)),
            Job("validate", "validate", "all", dependencies=("prepare",))]
    for universe in universes:
        def add(label, stage, variant="base", extra=(), target=0.02, deps=()):
            name = f"{universe}.{label}"
            jobs.append(Job(name, stage, universe, variant, tuple(extra), target,
                            tuple(f"{universe}.{d}" for d in deps)))
            return name
        add("covariance", "02", deps=())
        jobs[-1].dependencies = ("prepare",)
        add("returns", "03")
        jobs[-1].dependencies = ("prepare",)
        add("benchmarks", "04", deps=("covariance", "returns"))
        add("static", "05", deps=("benchmarks",))
        add("inputs_rff", "06", deps=("covariance",))
        add("pfml_rff", "07", deps=("inputs_rff",))
        add("inputs_linear", "06", "linear_pfml", deps=("covariance",))
        add("pfml_linear", "07", "linear_pfml", deps=("inputs_linear",))
        add("tables", "11", deps=("benchmarks", "returns", "static", "pfml_rff", "pfml_linear"))
        if scope == "core":
            continue
        add("grids_rff", "08", deps=("benchmarks",))
        add("grids_linear", "08", "linear_pfml", deps=("grids_rff",))
        add("importance", "09", deps=("benchmarks", "returns", "pfml_rff", "pfml_linear"))
        add("regimes", "10", deps=("benchmarks", "static", "pfml_rff", "pfml_linear"))
        add("u250_prepare", "prepare", "universe_250")
        add("u250_covariance", "02", "universe_250", deps=("u250_prepare",))
        add("u250_returns", "03", "universe_250", deps=("u250_prepare",))
        add("u250_benchmarks", "04", "universe_250", deps=("u250_covariance", "u250_returns"))
        add("u250_static", "05", "universe_250", deps=("u250_benchmarks",))
        baseline = ("--override", "portfolio_ml.benchmark_relative=false", "--override", "portfolio_ml.add_constant=true")
        add("u250_inputs", "06", "universe_250", extra=baseline, deps=("u250_covariance",))
        add("u250_pfml", "07", "universe_250", extra=baseline, deps=("u250_inputs",))
        for variant, prefix in (("active_pfml", "active"), ("universe_250", "u250_active")):
            upstream = ("covariance", "returns") if prefix == "active" else ("u250_covariance", "u250_returns")
            add(f"{prefix}_calibrate", "12", variant, extra=("calibrate",), deps=upstream)
            for target in (0.01, 0.02, 0.04):
                add(f"{prefix}_run_{target:g}", "12", variant, extra=("run",), target=target,
                    deps=(f"{prefix}_calibrate",))
            add(f"{prefix}_capacity", "12", variant, extra=("capacity",), deps=(f"{prefix}_calibrate",))
            base_pf = "pfml_rff" if prefix == "active" else "u250_pfml"
            bench = "benchmarks" if prefix == "active" else "u250_benchmarks"
            add(f"{prefix}_report", "12", variant, extra=("report",),
                deps=(base_pf, bench, *(f"{prefix}_run_{t:g}" for t in (0.01, 0.02, 0.04))))
        add("breadth", "13", deps=("active_capacity", "u250_active_capacity", "active_report", "u250_active_report"))
        add("halflife_run", "12", "halflife", extra=("run",), deps=("u250_active_calibrate",))
        add("halflife_report", "14", deps=("halflife_run", "u250_active_run_0.02", "u250_benchmarks"))
        add("monitor", "monitor", deps=("u250_active_run_0.02", "u250_benchmarks", "u250_returns"))
        add("holdout_availability", "15")
    return jobs


def output_paths(job):
    if job.stage in ("audit", "validate", "15"):
        return []
    cfg = load_config(config_path(job.universe, job.variant))
    p, o = cfg.paths.processed, cfg.paths.outputs
    sig = "raw" if job.variant == "linear_pfml" else "rff"
    if job.stage == "prepare":
        return [p / "preparation.json", p / "panel.parquet"]
    if job.stage == "02":
        return [p / "covariance" / n for n in ("cov_factor.parquet", "cov_loadings.parquet", "cov_diagnostics.csv", "provenance.json")]
    if job.stage == "03":
        return [p / n for n in ("predictions.parquet", "return_model/models.npz", "return_model/chosen.csv", "return_model/diagnostics.csv")]
    if job.stage == "04":
        return [o / "benchmarks" / n for n in ("pf_test.parquet", "weights_test.parquet", "table2_test.csv")]
    if job.stage == "05":
        return [o / "static_ml" / n for n in ("pf.parquet", "chosen.csv", "weights.parquet")]
    if job.stage == "06":
        tags = ["raw"] if sig == "raw" else [f"rff_g{g:.4f}" for g in cfg.portfolio_ml.rff_variance_grid]
        return [p / "pfml" / tag / "2025-11-30.npz" for tag in tags]
    if job.stage == "07":
        return [o / "portfolio_ml" / sig / n for n in ("pf.parquet", "weights.parquet", "aim_coefs.parquet", "chosen.csv")]
    if job.stage == "08":
        from jkmp.portfolio.grids import point_key
        return [o / "grids/summary.csv", *[
            o / "grids" / point_key(gamma, wealth) / f"pf_portfolio_ml_{sig}.parquet"
            for gamma in cfg.investor.gamma_grid for wealth in cfg.investor.wealth_grid]]
    names = {"08": "grids/summary.csv", "09": "feature_importance/importance.csv", "10": "regimes/regime_table.csv",
             "11": "tables/grid_edges.csv", "13": "u250/breadth/capacity_aum.csv", "14": "u250_hl/halflife/summary.csv"}
    if job.stage in names:
        if job.stage in ("13", "14"):
            return [ROOT / "outputs" / job.universe / names[job.stage]]
        return [o / names[job.stage]]
    if job.stage == "monitor":
        return [ROOT / "outputs" / job.universe / "u250/monitor/monitor.csv"]
    if job.stage == "12":
        if job.extra[0] == "calibrate":
            return [o / "active/calibrated.csv"]
        if job.extra[0] == "report":
            return [o / "active/summary.csv"]
        from run import calibrated_gamma
        try:
            gamma = calibrated_gamma(job.universe, job.variant, job.target)
        except (FileNotFoundError, ValueError):
            return []
        if job.extra[0] == "run":
            return [o / "active" / f"gte{gamma:g}" / n for n in ("pf.parquet", "weights.parquet", "chosen.csv")]
        return [o / "active/capacity" / f"gte{gamma:g}_w{wealth/1e9:g}B" / "pf.parquet" for wealth in cfg.investor.wealth_grid]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("core", "full"), default="core")
    parser.add_argument("--universe", choices=(*UNIVERSES, "all"), default="all")
    args = parser.parse_args()
    universes = UNIVERSES if args.universe == "all" else [args.universe]
    jobs = plan(universes, args.scope)
    logs = ROOT / "outputs/run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = logs / f"{args.scope}_{args.universe}_state.json"
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
