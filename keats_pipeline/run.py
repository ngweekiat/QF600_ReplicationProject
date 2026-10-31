"""Run isolated Keats stages without downloading data.

python keats_pipeline/run.py audit
python keats_pipeline/run.py prepare --universe all
python keats_pipeline/run.py 03 --universe us_large
python keats_pipeline/run.py 12 --universe us_mid --variant active_pfml calibrate
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

from workflow import ROOT, UNIVERSES, VARIANTS, audit, config_path, covariance, originals_unchanged, prepare, validate_prepared, write_json

SCRIPTS = {
    "03": "03_return_model.py", "04": "04_benchmarks.py", "05": "05_static_ml.py",
    "06": "06_pfml_inputs.py", "07": "07_portfolio_ml.py", "08": "08_grids.py",
    "09": "09_feature_importance.py", "10": "10_regimes.py", "11": "11_tables.py",
    "monitor": "11_monitor.py", "12": "12_active_pfml.py", "13": "13_breadth.py",
    "14": "14_halflife.py", "15": "15_holdout.py",
}


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


def run_stage(stage, universe, variant, extra, target):
    from jkmp.imputation_audit import enable_reporting
    operation = extra[0] if extra and not extra[0].startswith("--") else ""
    enable_reporting(f"{stage} {operation} {universe} {variant} te={target:g}")
    if stage in ("audit", "00"):
        return audit()
    if stage == "validate":
        return validate_prepared()
    if stage in ("prepare", "01"):
        return prepare(universe, variant)
    if stage == "02":
        return covariance(universe, variant)
    if stage == "15":
        report = {"universe": universe, "available": False,
                  "reason": "Keats test period already includes 2020-2025. No separate unseen holdout exists.",
                  "original_2021_2025_comparison_reused": False, "downloads_attempted": False}
        write_json(ROOT / "outputs" / universe / "holdout_availability.json", report)
        print(report["reason"])
        return
    if stage == "12" and variant in ("base", "linear_pfml"):
        variant = "active_pfml"
    script = ROOT / "scripts" / SCRIPTS[stage]
    if stage == "monitor":
        script = ROOT / "monitor_stage.py"
    args = list(extra)
    from jkmp.config import load_config
    if stage in ("03", "04", "05", "06", "07", "08", "09", "10", "11", "12"):
        cfg_path = config_path(universe, variant)
        args += ["--config", str(cfg_path)]
        cfg = load_config(cfg_path)
        required = [cfg.paths.processed / "panel.parquet"]
        if stage != "03":
            required.append(cfg.paths.processed / "covariance/cov_factor.parquet")
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Run upstream stages first; missing: " + ", ".join(missing))
        if stage in ("04","05","08","09","12"):
            from workflow import ensure_prediction_coverage
            ensure_prediction_coverage(cfg)
        if stage == "12" and ("run" in args or "capacity" in args):
            if not any("investor.gamma_te=" in a for a in args):
                args += ["--override", f"investor.gamma_te={calibrated_gamma(universe, variant, target):g}"]
    sys.argv = [str(script), *args]
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("keats_stage_" + stage, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if stage == "13":
        module.CONFIGS = {"80-stock": f"configs/{universe}/active_pfml.yaml",
                          "250-stock": f"configs/{universe}/universe_250.yaml"}
    if stage == "14":
        module.REF = f"configs/{universe}/universe_250.yaml"
        module.VAR = f"configs/{universe}/halflife.yaml"
    if stage == "monitor":
        module.SELECTED_CONFIG = config_path(universe, "universe_250")
    module.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["audit", "validate", "prepare", "00", "01", "02", *SCRIPTS])
    parser.add_argument("--universe", choices=[*UNIVERSES, "all"], default="us_large")
    parser.add_argument("--variant", choices=VARIANTS, default="base")
    parser.add_argument("--te-target", type=float, default=0.02)
    args, extra = parser.parse_known_args()
    if args.stage not in SCRIPTS and extra:
        parser.error(f"Unexpected arguments: {extra}")
    universes = UNIVERSES if args.universe == "all" else [args.universe]
    if args.stage in ("audit", "validate", "00"):
        universes = ["us_large"]
    failed = []
    for universe in universes:
        try:
            run_stage(args.stage, universe, args.variant, extra, args.te_target)
        except (FileNotFoundError, ValueError) as error:
            print(f"STOPPED {universe}: {error}", file=sys.stderr)
            failed.append(universe)
    originals_unchanged()
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
