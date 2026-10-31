"""Portfolio-ML (R pfml_implement): coefficient search, (g, p, λ) choice, aim portfolios, weights, Table 2.

hp years run from return_model.pred_years[0] to the last year of the evaluation window. For hp year y,
β = (Σ̄ + λI)⁻¹ r̄ uses every month with returns realised by Dec y−1; each December the (g, p, λ) with the
highest cumulative realised utility since the first hp year is used for the next return year, so aims start a
year after it.
Weights π_t = m·π_start + (I − m)·aim_t run over the test window from VW initial weights.
Reads data/processed/pfml/{tag}/ from scripts/06_pfml_inputs.py.
  portfolio_ml.use_rff = true  : search over g (rff_variance_grid), p (p_grid) and λ (ridge_grid) -> "rff"
  portfolio_ml.use_rff = false : the ranked characteristics, λ only, "Portfolio-ML (linear)" -> "raw"
                                 (configs/experiments/linear_pfml.yaml)
Writes outputs/portfolio_ml/{rff|raw}/{validation,aims,aim_coefs,weights,pf}.parquet + chosen.csv, and
rebuilds outputs/table2_test.csv from every test-window pf file present (benchmarks, Static-ML, Portfolio-ML).

    python scripts/07_portfolio_ml.py [--config configs/experiments/linear_pfml.yaml]
"""

from __future__ import annotations

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
    processed, outputs, pm = cfg.paths.processed, cfg.paths.outputs, cfg.portfolio_ml
    sig, tags, label = signal_set(pm.use_rff, pm.rff_variance_grid)
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
                           charge_exit_tc=cfg.portfolio.charge_exit_tc, label=label)

    out = outputs / "portfolio_ml" / sig
    out.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "aims", "aim_coefs", "weights", "pf"):
        res[name].to_parquet(out / f"{name}.parquet", index=False)
    res["chosen"].to_csv(out / "chosen.csv", index=False)

    pfs = [outputs / "benchmarks" / "pf_test.parquet", outputs / "static_ml" / "pf.parquet",
           outputs / "portfolio_ml" / "rff" / "pf.parquet", outputs / "portfolio_ml" / "raw" / "pf.parquet"]
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
