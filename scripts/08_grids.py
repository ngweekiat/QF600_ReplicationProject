"""Implementable efficient frontier: Static-ML* and Portfolio-ML at every γ × wealth, then Fig. 1A, 1B and capacity.

The Portfolio-ML signal set follows portfolio_ml.use_rff; run once with configs/base.yaml and once with
configs/experiments/linear_pfml.yaml to add Portfolio-ML (linear) — the second run reuses Static-ML results.
The summary and figures use only Static-ML* and Portfolio-ML, including when older cached results exist.

Loops investor.gamma_grid × investor.wealth_grid (25 points). Each point re-tunes every hyperparameter
(Static-ML* (k, u, g); Portfolio-ML (g, p, λ) and β) and runs over the test window, in parallel with
joblib (grids.n_jobs workers). Portfolio-ML moments are cached per point in data/processed/grids/{key}/;
finished points in outputs/grids/{key}/ are reused unless --force.
Needs scripts 01–03 (panel, covariance, predictions); no benchmark results required.
Writes outputs/grids/summary.csv and outputs/figures/{fig1a_frontier,fig1b_frontier_wealth,capacity}.{png,pdf}.

    python scripts/08_grids.py [--config configs/experiments/linear_pfml.yaml] [--force] [--plots-only]
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
from joblib import Parallel, delayed

from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.data.prepare import risk_free_monthly
from jkmp.evaluation.frontier import plot_capacity, plot_fig1a, plot_fig1b
from jkmp.evaluation.comparison import selected_methods
from jkmp.portfolio.grids import grid_inputs, grid_points, load_point, run_point, summarise

MONTH = pd.offsets.MonthEnd(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    parser.add_argument("--force", action="store_true", help="recompute finished points")
    parser.add_argument("--plots-only", action="store_true", help="only redraw from outputs/grids/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    processed, outputs, inv = cfg.paths.processed, cfg.paths.outputs, cfg.investor
    out_root, cache_root = outputs / "grids", processed / "grids"
    points = grid_points(inv.gamma_grid, inv.wealth_grid)

    preds = pd.read_parquet(processed / "predictions.parquet")[["id", "eom", "pred_ld1"]]
    panel = pd.read_parquet(processed / "panel.parquet").merge(preds, on=["id", "eom"], how="left")
    wealth = pd.read_parquet(processed / "wealth.parquet")
    cov = load_covariance(processed / "covariance")
    rf = risk_free_monthly(pd.read_parquet(cfg.paths.raw / "ff_factors_monthly.parquet"))
    inp = grid_inputs(cfg, panel, wealth, cov, rf)
    del panel

    if not args.plots_only:
        start = time.time()
        logging.info("%d grid points on %d workers (Portfolio-ML signals: %s); test window %s .. %s", len(points),
                     cfg.grids.n_jobs, "RFF" if cfg.portfolio_ml.use_rff else "raw characteristics",
                     (inp.eval_dates[0] + MONTH).date(), (inp.eval_dates[-1] + MONTH).date())
        Parallel(n_jobs=cfg.grids.n_jobs, verbose=10)(
            delayed(run_point)(inp, cfg, g, w, cache_root, out_root, args.force) for g, w in points)
        logging.info("grid done in %.1f min", (time.time() - start) / 60)
    summary = summarise(pd.concat([load_point(out_root, g, w) for g, w in points], ignore_index=True))
    summary = selected_methods(summary)
    summary.to_csv(out_root / "summary.csv", index=False)

    scaled = pd.DataFrame(columns=["type", "Vol", "R_TC"])
    sr_gross = None

    figs = outputs / "figures"
    written = (plot_fig1a(summary, scaled, inv.wealth, inv.gamma, sr_gross, figs / "fig1a_frontier")
               + plot_fig1b(summary, sr_gross, figs / "fig1b_frontier_wealth")
               + plot_capacity(summary, inv.gamma, figs / "capacity"))
    logging.info("wrote %s", ", ".join(p.name for p in written))

    pd.set_option("display.width", 200)
    cols = ["type", "gamma", "wealth", "R_TC", "Vol", "SR_net", "Utility", "TC", "Turnover", "Lev"]
    show = summary[summary["type"].isin(["Portfolio-ML", "Portfolio-ML (linear)", "Static-ML*"])
                   & (summary["gamma"] == inv.gamma)].sort_values(["type", "wealth"])
    print(show[cols].assign(wealth=show["wealth"] / 1e9).rename(columns={"wealth": "wealth_$B"})
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
