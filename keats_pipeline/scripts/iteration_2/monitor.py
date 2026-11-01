"""Monitor a Keats test run without reusing the original holdout results."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.metrics import active_returns, active_weights, ex_ante_exposures
from jkmp.evaluation.monitor import half_life_steps, monitor_table, plot_dashboard
from jkmp.models.return_model import monthly_rank_ic
from jkmp.imputation_audit import enable_reporting, reporting_enabled
if not reporting_enabled():
    enable_reporting(Path(__file__).name)

SELECTED_CONFIG = ROOT / "configs/us_large/iteration_2.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", default="2020-01-31")
    args = parser.parse_args()
    cfg = load_config(SELECTED_CONFIG)
    cal = pd.read_csv(cfg.paths.outputs / "active/calibrated.csv")
    gamma = float(cal.loc[(cal.te_target - 0.02).abs().lt(1e-12), "gamma_te"].iloc[0])
    run = cfg.paths.outputs / "active" / f"gte{gamma:g}"
    panel = pd.read_parquet(cfg.paths.processed / "panel.parquet")
    cov = load_covariance(cfg.paths.processed / "covariance")
    pf = pd.read_parquet(run / "pf.parquet")
    weights = pd.read_parquet(run / "weights.parquet")
    benchmark = pd.read_parquet(cfg.paths.outputs / "benchmarks/pf_test.parquet").query("type == 'Market'").assign(tc=0.0)
    active = active_returns(pf, benchmark, cfg.dates)
    exposure = ex_ante_exposures(active_weights(weights, panel), cov)
    ic = monthly_rank_ic(pd.read_parquet(cfg.paths.processed / "predictions.parquet"), panel, cfg.dates)
    chosen = pd.read_csv(run / "chosen.csv")
    hl = half_life_steps(chosen, cfg.portfolio_ml.half_life_years) if "h" in chosen else None
    # Real execution costs are unobserved: this check must stay missing, not substitute modelled costs.
    realised = pd.Series(float("nan"), index=pd.DatetimeIndex(pf.eom_ret))
    table = monitor_table(active, exposure, ic, realised, hl, cfg.monitor, cfg.dates)
    out = cfg.paths.outputs / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "monitor.csv", index=False)
    shown = table[table.eom_ret.ge(pd.Timestamp(args.start))]
    plot_dashboard(shown, cfg.monitor, "Keats 2020-2025 test monitor; no separate holdout", out / "dashboard")


if __name__ == "__main__":
    main()
