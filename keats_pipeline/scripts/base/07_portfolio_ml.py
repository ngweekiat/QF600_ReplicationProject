"""Fit and evaluate the sole default Portfolio-ML strategy with expanding annual updates."""

from __future__ import annotations

from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import configure
configure(__file__)

import argparse
import logging

import numpy as np
import pandas as pd

from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.data.prepare import risk_free_monthly
from jkmp.evaluation.holdout import evaluation_window
from jkmp.evaluation.metrics import table2
from jkmp.portfolio.portfolio_ml import load_moments, return_year, run_portfolio_ml, search, signal_set

MONTH = pd.offsets.MonthEnd(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    if cfg.paths.outputs.parent.name != "base":raise ValueError("This script writes base results; choose the matching iteration script")
    processed, outputs, pm = cfg.paths.processed, cfg.paths.outputs, cfg.portfolio_ml
    sig, tags, label = signal_set(True,pm.rff_variance_grid)
    start, end = evaluation_window(cfg.dates)
    hp_years = list(range(cfg.return_model.pred_years[0], end.year + 1))

    results, aim_dates = {}, None
    for tag in tags:
        moments = load_moments(processed / "pfml" / tag, with_risk=False)
        aim_dates = [d for d in sorted(moments.months) if hp_years[0] < return_year(d) <= hp_years[-1]]
        results[tag] = search(moments, tag, pm.p_grid, cfg.ridge_grid.values, hp_years, aim_dates)
        del moments

    panel = pd.read_parquet(processed / "panel.parquet")
    wealth = pd.read_parquet(processed / "wealth.parquet")
    cov = load_covariance(processed / "covariance")
    rf = risk_free_monthly(pd.read_parquet(cfg.paths.raw / "ff_factors_monthly.parquet"))
    ret = pd.DatetimeIndex(aim_dates) + MONTH
    eval_dates = pd.DatetimeIndex(aim_dates)[(ret >= start) & (ret <= end)]
    res = run_portfolio_ml(results, panel, wealth, cov, rf, aim_dates, eval_dates, gamma=cfg.investor.gamma,
                           mu=cfg.investor.mu, iterations=cfg.investor.m_iterations,
                           charge_exit_tc=cfg.portfolio.charge_exit_tc)

    out = outputs / "portfolio_ml" / sig
    out.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "aims", "aim_coefs", "weights", "pf"):
        res[name].to_parquet(out / f"{name}.parquet", index=False)
    res["chosen"].to_csv(out / "chosen.csv", index=False)

    pfs = [outputs / "static_ml" / "pf.parquet",
           outputs / "portfolio_ml" / "rff" / "pf.parquet"]
    t2 = table2(pd.concat([pd.read_parquet(f) for f in pfs if f.exists()], ignore_index=True), cfg.investor.gamma,
            cfg.dates)
    t2.to_csv(outputs / "table2_test.csv")

    pd.set_option("display.width", 200)
    chosen = res["chosen"]
    print(f"\n{label} choices by return year:")
    print(chosen.assign(log_l=np.log(chosen["l"].where(chosen["l"] > 0)))  # NaN for λ = 0
          [["year", "tag", "p", "log_l", "cum_obj"]].to_string(index=False))
    s = res["aims"].groupby("eom")["w_aim"].agg(lev=lambda w: w.abs().sum(), net="sum")
    print(f"\nAims {s.index[0].date()} .. {s.index[-1].date()}: median Σ|w| {s['lev'].median():.2f}, "
          f"median Σw {s['net'].median():.2f}")
    print(f"\nTable 2 — test: returns {eval_dates[0] + MONTH:%Y-%m} to {eval_dates[-1] + MONTH:%Y-%m}, "
          f"wealth ${cfg.investor.wealth / 1e9:.0f}B by {cfg.investor.wealth_end_date:%Y}, gamma {cfg.investor.gamma:g}")
    print(t2.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
