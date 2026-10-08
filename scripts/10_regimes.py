"""Regimes over the test window: FRED USREC / VIXCLS / FEDFUNDS labels, cumulative utility with shaded
episodes, Table 2 statistics by regime and method, and every method's trades re-costed with Λ ∝ VIX.

Uses the test-window performance of every method present (outputs/benchmarks/pf_test.parquet,
outputs/static_ml/pf.parquet, outputs/portfolio_ml/{rff,raw}/pf.parquet). FRED files are cached in
data/raw/fred_{series}.parquet (--force re-downloads).
Writes outputs/regimes/{labels,regime_table,vix_recost,cumulative_utility}.csv and
outputs/figures/cumulative_utility.{png,pdf}.

    python scripts/10_regimes.py [--force]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from jkmp.config import load_config
from jkmp.evaluation import regimes as rg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    parser.add_argument("--force", action="store_true", help="re-download the FRED series")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    outputs, rc, gamma = cfg.paths.outputs, cfg.regimes, cfg.investor.gamma

    files = [outputs / "benchmarks" / "pf_test.parquet", outputs / "static_ml" / "pf.parquet",
             outputs / "portfolio_ml" / "rff" / "pf.parquet", outputs / "portfolio_ml" / "raw" / "pf.parquet"]
    pfs = pd.concat([pd.read_parquet(f) for f in files if f.exists()], ignore_index=True)
    pfs = pfs[["type", "eom", "eom_ret", "r", "tc", "turnover", "inv", "shorting"]]
    ret_months = pd.DatetimeIndex(sorted(pfs["eom_ret"].unique()))
    form_months = pd.DatetimeIndex(sorted(pfs["eom"].unique()))

    fred = rg.load_fred(cfg.data_sources.fred_csv_url, rc.fred_series, cfg.paths.raw, force=args.force)
    monthly = rg.monthly_series(fred)
    labels = rg.label_regimes(monthly, ret_months, rc.fed_change_months, rc.fed_threshold)
    table = rg.regime_table(pfs, labels, gamma)
    scale = rg.vix_cost_scale(monthly, form_months, rc.vix_norm)
    recost = rg.recost_vix(pfs, scale, gamma)
    cum = rg.cumulative_utility(pfs, gamma)

    out = outputs / "regimes"
    out.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out / "labels.csv", index=False)
    table.to_csv(out / "regime_table.csv", index=False)
    recost.to_csv(out / "vix_recost.csv", index=False)
    cum.to_csv(out / "cumulative_utility.csv", index=False)
    written = rg.plot_cumulative_utility(cum, rc.episodes, rc.plot_types, gamma,
                                         outputs / "figures" / "cumulative_utility")
    logging.info("wrote %s", ", ".join(p.name for p in written))

    pd.set_option("display.width", 220)
    order = list(dict.fromkeys(pfs["type"]))
    print("Months per regime (return months):")
    for dim in rg.REGIME_COLUMNS:
        print(f"  {dim}: {labels[dim].value_counts().to_dict()}")
    for col in ("Utility", "TC", "Turnover"):
        piv = table.pivot_table(index=["dimension", "regime"], columns="type", values=col)[order]
        print(f"\n{col} by regime (annualised; Turnover monthly):")
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nVIX re-costing (Λ_t = Λ · VIX_t / mean VIX, mean over the {rc.vix_norm}):")
    print(recost.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
