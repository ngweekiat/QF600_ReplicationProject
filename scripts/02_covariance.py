"""Stage 2 runner: Barra-style covariance for every universe month-end.

Reads data/processed/panel.parquet and data/raw/{top500_by_month, crsp_daily, ff_factors_daily}.parquet;
writes data/processed/covariance/{cov_loadings,cov_factor}.parquet, factor_returns.parquet and
cov_diagnostics.csv.

    python scripts/02_covariance.py
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from jkmp import covariance
from jkmp.config import load_config
from jkmp.data.wrds_pull import RAW_FILES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    raw, processed = cfg.paths.raw, cfg.paths.processed

    result = covariance.build_covariance(
        panel=pd.read_parquet(processed / "panel.parquet"),
        top=pd.read_parquet(raw / f"top{cfg.covariance.factor_universe_top_n}_by_month.parquet"),
        daily=pd.read_parquet(raw / RAW_FILES["daily"]),
        ff_daily=pd.read_parquet(raw / RAW_FILES["ff_daily"]),
        cfg=cfg,
    )
    out = processed / "covariance"
    covariance.save_covariance(result.parts, out)
    result.factor_returns.to_parquet(out / "factor_returns.parquet")
    result.diagnostics.to_csv(out / "cov_diagnostics.csv", index=False)
    d = result.diagnostics
    logging.info("Wrote %d month-ends (%s to %s) to %s; ivar imputed for %d of %d stock-months",
                 len(d), d["eom"].min().date(), d["eom"].max().date(), out,
                 int(d["n_ivar_imputed"].sum()), int(d["n_stocks"].sum()))


if __name__ == "__main__":
    main()
