"""Build the default Portfolio-ML Fourier-signal moments from past stock history."""

from __future__ import annotations

from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import configure
configure(__file__)

import argparse
import logging
import time

import pandas as pd

from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.data.prepare import risk_free_monthly
from jkmp.evaluation.holdout import evaluation_window
from jkmp.portfolio.portfolio_ml import build_moments, save_moments, signal_set

MONTH = pd.offsets.MonthEnd(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    processed = cfg.paths.processed
    panel = pd.read_parquet(processed / "panel.parquet")
    wealth = pd.read_parquet(processed / "wealth.parquet")
    cov = load_covariance(processed / "covariance")
    rf = risk_free_monthly(pd.read_parquet(cfg.paths.raw / "ff_factors_monthly.parquet"))

    lb = cfg.universe.lookback_months
    first = min(cov) + MONTH * (lb + 1)
    dates = pd.DatetimeIndex([d for d in sorted(cov) if first <= d <= evaluation_window(cfg.dates)[1] - MONTH])
    pm = cfg.portfolio_ml
    common = dict(gamma=cfg.investor.gamma, mu=cfg.investor.mu, lb=lb, iterations=cfg.investor.m_iterations,
                  scale_by_vol=pm.scale_by_vol)
    _, tags, _ = signal_set(True,pm.rff_variance_grid)
    runs=[(tag,dict(rff_variance=g,p_max=max(pm.p_grid),seed=cfg.seed))
          for tag,g in zip(tags,pm.rff_variance_grid)]
    logging.info("PF-ML inputs for %d formation months, %s .. %s", len(dates), dates[0].date(), dates[-1].date())
    for tag, kw in runs:
        start = time.time()
        res = build_moments(panel, wealth, cov, rf, dates, cfg.features.names, **common, **kw)
        save_moments(res, processed / "pfml" / tag)
        logging.info("%s: %d months, %d signals, %.1f min", tag, len(res.months), len(res.columns),
                     (time.time() - start) / 60)


if __name__ == "__main__":
    main()
