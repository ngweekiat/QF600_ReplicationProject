"""Baseline tables over the test window: Table 2 next to the paper's, Table 3 (probability of outperformance),
Table 4 (return correlations), Table 2 by sub-period, hyperparameter grid-edge shares and Fig. D.1.

Reads the test-window performance of scripts 04, 05 and 07 (both signal sets), the picks of 03, 05 and 07,
and the paper's values in docs/paper_tables/. Writes outputs/tables/{table2_vs_paper,table3,table4,
subperiods,grid_edges}.csv and outputs/figures/figD1_hyperparameters.{png,pdf}.

    python scripts/11_tables.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from jkmp.config import PROJECT_ROOT, load_config
from jkmp.evaluation.hyperparams import edge_share, plot_hyperparameters
from jkmp.evaluation.metrics import prob_outperformance, return_correlations, subperiod_table, table2

PAPER = PROJECT_ROOT / "docs" / "paper_tables"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    outputs, processed, gamma, rep = cfg.paths.outputs, cfg.paths.processed, cfg.investor.gamma, cfg.reporting
    files = [outputs / "benchmarks" / "pf_test.parquet", outputs / "static_ml" / "pf.parquet",
             outputs / "portfolio_ml" / "rff" / "pf.parquet", outputs / "portfolio_ml" / "raw" / "pf.parquet"]
    pfs = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out = outputs / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731

    t2 = table2(pfs, gamma, cfg.dates)
    paper = pd.read_csv(PAPER / "table2.csv", index_col="type")
    cols = ["R", "Vol", "SR_gross", "TC", "R_TC", "SR_net", "Utility", "Turnover", "Lev"]
    side = pd.concat({"ours": t2[cols], "paper": paper[cols].reindex(t2.index)}, axis=1).swaplevel(axis=1)
    side = side[[(c, w) for c in cols for w in ("ours", "paper")]]
    side.to_csv(out / "table2_vs_paper.csv")
    print(f"Table 2, returns {pfs['eom_ret'].min():%Y-%m} .. {pfs['eom_ret'].max():%Y-%m}, ours vs paper:")
    print(side.to_string(float_format=fmt))

    t3 = prob_outperformance(pfs, gamma, cfg.dates, rep.main_types)
    t4 = return_correlations(pfs, cfg.dates, rep.main_types)
    t3.to_csv(out / "table3.csv")
    t4.to_csv(out / "table4.csv")
    print("\nTable 3 — P(row has higher mean utility than column):")
    print(t3.to_string(float_format=fmt))
    print("\nTable 4 — correlation of gross returns:")
    print(t4.to_string(float_format=lambda v: f"{v:.2f}"))

    periods = {p.label: (p.start, p.end) for p in rep.subperiods}
    sub = subperiod_table(pfs, gamma, cfg.dates, periods, ["Utility", "SR_net", "TC"])
    sub.to_csv(out / "subperiods.csv")
    print("\nTable 2 by sub-period:")
    print(sub.to_string(float_format=fmt))

    pm, st, rm = cfg.portfolio_ml, cfg.static_ml, cfg.return_model
    lambdas = np.asarray(cfg.ridge_grid.values)
    chosen = {"Portfolio-ML": pd.read_csv(outputs / "portfolio_ml" / "rff" / "chosen.csv"),
              "Portfolio-ML (linear)": pd.read_csv(outputs / "portfolio_ml" / "raw" / "chosen.csv")}
    static = pd.read_csv(outputs / "static_ml" / "chosen.csv")
    model = pd.read_csv(processed / "return_model" / "chosen.csv")
    pf, lin = chosen["Portfolio-ML"], chosen["Portfolio-ML (linear)"]
    edges = edge_share({
        "Portfolio-ML g": (pf["g"], pm.rff_variance_grid), "Portfolio-ML p": (pf["p"], pm.p_grid),
        "Portfolio-ML λ": (pf["l"], lambdas), "Portfolio-ML (linear) λ": (lin["l"], lambdas),
        "Static-ML* k": (static["k"], st.k_grid), "Static-ML* u": (static["u"], st.u_grid),
        "Static-ML* v = 1 + g": (static["g"] + 1.0, [g + 1.0 for g in st.g_grid]),
        "Return model g": (model["g"], rm.rff_variance_grid), "Return model p": (model["p"], rm.p_grid),
        "Return model λ": (model["lam"], lambdas)})
    edges.to_csv(out / "grid_edges.csv", index=False)
    print("\nShare of yearly picks on a grid edge:")
    print(edges.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    written = plot_hyperparameters(chosen, static, outputs / "figures" / "figD1_hyperparameters")
    logging.info("wrote %s", ", ".join(p.name for p in written))


if __name__ == "__main__":
    main()
