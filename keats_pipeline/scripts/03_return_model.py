"""Stage 3 runner: one-month return predictions pred_ld1 (random-feature ridge, yearly expanding refit).

Writes data/processed/predictions.parquet and return_model/{hp_search,chosen,diagnostics}.*

    python scripts/03_return_model.py
"""

from __future__ import annotations

from _bootstrap import configure
configure(__file__)

import argparse
import logging

import pandas as pd

from jkmp.config import load_config
from jkmp.evaluation.holdout import last_open_month
from jkmp.models.return_model import fit_return_model, prediction_diagnostics, save_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    processed = cfg.paths.processed
    panel = pd.read_parquet(processed / "panel.parquet")

    result = fit_return_model(panel, cfg)
    preds = result.predictions
    open_preds = preds[preds["eom"] + pd.offsets.MonthEnd(1) <= last_open_month(cfg.dates)]  # no holdout diagnostics
    diag = prediction_diagnostics(open_preds, panel, cfg.dates)
    out = processed / "return_model"
    out.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(processed / "predictions.parquet", index=False)
    result.hp_search.to_parquet(out / "hp_search.parquet", index=False)
    result.chosen.to_csv(out / "chosen.csv", index=False)
    save_models(result, out / "models.npz")
    diag.to_csv(out / "diagnostics.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nChosen hyperparameters by return year:")
    print(result.chosen[["year", "g", "p", "log_lam", "val_mse", "n_train", "n_val", "n_test"]].to_string(index=False))
    print("\nOut-of-sample diagnostics (R² vs zero forecast; mean monthly rank IC):")
    print(diag.pivot(index="year", columns="scope", values=["r2_oos", "rank_ic"]).round(4).to_string())


if __name__ == "__main__":
    main()
