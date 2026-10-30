"""Portfolio-ML inputs (R pfml_input_fun): per formation month r̃, Σ̃ = risk + tc, and the signals S_t.

Formation months run from the first Σ + (lb+1) months (R dates_m2) to the end of the evaluation window − 1 month.
  portfolio_ml.use_rff = true  : random Fourier features at p_max = max(portfolio_ml.p_grid), one run per
                                 RFF variance g -> data/processed/pfml/rff_g{g:.4f}/
  portfolio_ml.use_rff = false : the ranked characteristics themselves (linear Portfolio-ML,
                                 configs/experiments/linear_pfml.yaml) -> data/processed/pfml/raw/
Each directory holds meta.json, rff_w.npy (RFF only) and one YYYY-MM-DD.npz per month
(ids, r_tilde, denom, risk, signal); load with jkmp.portfolio.portfolio_ml.load_moments().

    python scripts/06_pfml_inputs.py [--config configs/experiments/linear_pfml.yaml]
"""

from __future__ import annotations

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
    _, tags, _ = signal_set(pm.use_rff, pm.rff_variance_grid)
    runs = ([(tag, dict(use_rff=True, rff_variance=g, p_max=max(pm.p_grid), seed=cfg.seed,
                        add_orig=pm.add_original_features)) for tag, g in zip(tags, pm.rff_variance_grid)]
            if pm.use_rff else [("raw", dict(use_rff=False))])
    logging.info("PF-ML inputs for %d formation months, %s .. %s", len(dates), dates[0].date(), dates[-1].date())
    for tag, kw in runs:
        start = time.time()
        res = build_moments(panel, wealth, cov, rf, dates, cfg.features.names, **common, **kw)
        save_moments(res, processed / "pfml" / tag)
        logging.info("%s: %d months, %d signals, %.1f min", tag, len(res.months), len(res.columns),
                     (time.time() - start) / 60)


if __name__ == "__main__":
    main()
