"""Benchmarks and their Table 2 rows: Market, 1/N, Minimum variance, Markowitz-ML, Factor-ML.

Windows are defined by the months in which returns are realised (formation month t earns over t+1),
each started fresh from value weights (as R does per window):
  test : returns in the evaluation window (dates.test_* or, with dates.evaluate: holdout, dates.holdout_*)
  oos  : returns in return_model.pred_years, up to the last month the holdout lock allows
Writes outputs/benchmarks/{weights,pf,table2}_{window}.* and prints both Table 2s.

    python scripts/04_benchmarks.py [--override portfolio.charge_exit_tc=false]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.holdout import evaluation_window, last_open_month
from jkmp.evaluation.metrics import table2
from jkmp.portfolio.benchmarks import run_benchmarks
from jkmp.portfolio.ml_benchmarks import run_ml_benchmarks

MONTH = pd.offsets.MonthEnd(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    processed = cfg.paths.processed
    preds = pd.read_parquet(processed / "predictions.parquet")[["id", "eom", "pred_ld1"]]
    panel = pd.read_parquet(processed / "panel.parquet").merge(preds, on=["id", "eom"], how="left")
    wealth = pd.read_parquet(processed / "wealth.parquet")
    cov = load_covariance(processed / "covariance")

    formation = pd.DatetimeIndex(sorted(cov))
    returns = formation + MONTH
    first, last = cfg.return_model.pred_years
    windows = {
        "test": formation[(returns >= evaluation_window(cfg.dates)[0]) & (returns <= evaluation_window(cfg.dates)[1])],
        "oos": formation[(returns >= pd.Timestamp(first, 1, 31))
                         & (returns <= min(pd.Timestamp(last, 12, 31), last_open_month(cfg.dates)))],
    }
    out = cfg.paths.outputs / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    exit_tc = cfg.portfolio.charge_exit_tc
    for name, dates in windows.items():
        w1, pf1 = run_benchmarks(panel, wealth, cov, dates, exit_tc)
        w2, pf2 = run_ml_benchmarks(panel, wealth, cov, dates, cfg.investor.gamma, cfg.factor_ml.n_portfolios, exit_tc)
        weights, pf = pd.concat([w1, w2], ignore_index=True), pd.concat([pf1, pf2], ignore_index=True)
        t2 = table2(pf, cfg.investor.gamma, cfg.dates)
        weights.to_parquet(out / f"weights_{name}.parquet", index=False)
        pf.to_parquet(out / f"pf_{name}.parquet", index=False)
        t2.to_csv(out / f"table2_{name}.csv")
        print(f"\nTable 2 — {name}: returns {dates[0] + MONTH:%Y-%m} to {dates[-1] + MONTH:%Y-%m} "
              f"({len(dates)} months), wealth ${cfg.investor.wealth / 1e9:.0f}B by {cfg.investor.wealth_end_date:%Y}, "
              f"gamma {cfg.investor.gamma:g}, charge_exit_tc={exit_tc}")
        print(t2.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
