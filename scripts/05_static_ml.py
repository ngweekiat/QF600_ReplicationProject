"""Static-ML and Static-ML*: validate (k, u, g) from return year 2015, pick each December, evaluate on the test window.

Reports Static-ML* only; no benchmark stage is required.
Writes outputs/static_ml/{validation,chosen,weights,pf}.* and outputs/table2_test.csv.

    python scripts/05_static_ml.py
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.metrics import table2
from jkmp.evaluation.comparison import load_comparison, selected_methods
from jkmp.portfolio.static_ml import hyper_grid, run_static_ml

MONTH = pd.offsets.MonthEnd(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    processed, outputs = cfg.paths.processed, cfg.paths.outputs
    preds = pd.read_parquet(processed / "predictions.parquet")[["id", "eom", "pred_ld1"]]
    panel = pd.read_parquet(processed / "panel.parquet").merge(preds, on=["id", "eom"], how="left")
    wealth = pd.read_parquet(processed / "wealth.parquet")
    cov = load_covariance(processed / "covariance")

    formation = pd.DatetimeIndex(sorted(cov))
    ret = formation + MONTH
    first_year = cfg.return_model.pred_years[0]
    val_dates = formation[(ret >= pd.Timestamp(first_year, 1, 31)) & (ret <= pd.Timestamp(cfg.dates.test_end))]
    eval_dates = formation[(ret >= pd.Timestamp(cfg.dates.test_start)) & (ret <= pd.Timestamp(cfg.dates.test_end))]
    s = cfg.static_ml
    res = run_static_ml(panel, wealth, cov, val_dates, eval_dates, cfg.investor.gamma,
                        hyper_grid(s.k_grid, s.u_grid, s.g_grid), cfg.portfolio.charge_exit_tc)
    res["pf"] = selected_methods(res["pf"])

    out = outputs / "static_ml"
    out.mkdir(parents=True, exist_ok=True)
    res["validation"].to_parquet(out / "validation.parquet", index=False)
    res["chosen"].to_csv(out / "chosen.csv", index=False)
    res["weights"].to_parquet(out / "weights.parquet", index=False)
    res["pf"].to_parquet(out / "pf.parquet", index=False)

    t2 = table2(load_comparison(outputs), cfg.investor.gamma)
    t2.to_csv(outputs / "table2_test.csv")
    pd.set_option("display.width", 200)
    print("\nStatic-ML* choices (k, u, g) by return year:")
    print(res["chosen"][["year", "k", "u", "g", "cum_obj"]].to_string(index=False))
    print(f"\nTable 2 — test: returns {eval_dates[0] + MONTH:%Y-%m} to {eval_dates[-1] + MONTH:%Y-%m}, "
          f"wealth ${cfg.investor.wealth / 1e9:.0f}B by {cfg.investor.wealth_end_date:%Y}, gamma {cfg.investor.gamma:g}")
    print(t2.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
