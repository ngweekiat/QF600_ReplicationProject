"""Feature importance by theme (Fig. 5): permute each theme's characteristics across universe stocks within
each month, keep W, β and hyperparameters, and measure the drop in realised utility over the test window.

  Portfolio-ML (net of TC) and, if built, Portfolio-ML (linear): same aim coefficients as scripts/07.
  Markowitz-ML (gross): pred_ld1 recomputed with each year's fitted return model (scripts/03 models.npz),
  w = Σ⁻¹μ/γ at feature_importance.markowitz_gamma with zero wealth, as R's tpf_cf_fun.
Writes outputs/feature_importance/{pf.parquet, importance.csv} and outputs/figures/fig5_feature_importance.{png,pdf}.

    python scripts/09_feature_importance.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.data.prepare import risk_free_monthly
from jkmp.evaluation import feature_importance as fi
from jkmp.models.return_model import load_models
from jkmp.portfolio.grids import grid_inputs
from jkmp.portfolio.recursion import month_inputs

LABELS = {"rff": "Portfolio-ML", "raw": "Portfolio-ML (linear)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="YAML config (default: configs/base.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    processed, outputs, inv = cfg.paths.processed, cfg.paths.outputs, cfg.investor
    preds = pd.read_parquet(processed / "predictions.parquet")[["id", "eom", "pred_ld1"]]
    panel = pd.read_parquet(processed / "panel.parquet").merge(preds, on=["id", "eom"], how="left")
    wealth = pd.read_parquet(processed / "wealth.parquet")
    cov = load_covariance(processed / "covariance")
    rf = risk_free_monthly(pd.read_parquet(cfg.paths.raw / "ff_factors_monthly.parquet"))
    inp = grid_inputs(cfg, panel, wealth, cov, rf)
    del panel
    dates, panel = inp.eval_dates, inp.panel

    universe = fi._universe(panel, dates)
    perms = fi.permutations(universe, cfg.seed)
    themes = fi.theme_list(cfg.features.themes)
    inputs = month_inputs(panel, wealth, cov, dates, mu_col=None)
    common = dict(gamma=inv.gamma, mu=inv.mu, iterations=inv.m_iterations, charge_exit_tc=cfg.portfolio.charge_exit_tc)

    runs, types = [], []
    for sig, label in LABELS.items():
        path = outputs / "portfolio_ml" / sig / "aim_coefs.parquet"
        if not path.exists():
            logging.info("skip %s: %s not found", label, path)
            continue
        coefs = pd.read_parquet(path)
        coefs = coefs[coefs["eom"].isin(dates)]
        specs = {tag: fi.load_signal_spec(processed / "pfml" / tag) for tag in coefs["tag"].unique()}
        for theme, feats in themes:
            pf = fi.pfml_counterfactual(feats, universe, perms, coefs, specs, inputs, panel, wealth, cov, rf, **common)
            runs.append(pf.assign(type=label, theme=theme))
        types.append(label)
        base = runs[-len(themes)]
        ref = pd.read_parquet(outputs / "portfolio_ml" / sig / "pf.parquet")
        gap = float(np.abs(base.sort_values("eom")["r"].to_numpy() - ref.sort_values("eom")["r"].to_numpy()).max())
        logging.info("%s unshuffled vs scripts/07: max |Δr| = %.2e", label, gap)

    models, z = load_models(processed / "return_model" / "models.npz")
    wealth0 = wealth.assign(wealth=0.0)  # R: wealth_func(wealth_end = 0) -> no trading costs
    for theme, feats in themes:
        pf = fi.markowitz_counterfactual(feats, universe, perms, cfg.features.names, models, z, panel, wealth0, cov,
                                         gamma=cfg.feature_importance.markowitz_gamma,
                                         charge_exit_tc=cfg.portfolio.charge_exit_tc)
        runs.append(pf.assign(type="Markowitz-ML", theme=theme))
    types.append("Markowitz-ML")

    pfs = pd.concat(runs, ignore_index=True)
    imp = fi.importance(pfs, inv.gamma)
    out = outputs / "feature_importance"
    out.mkdir(parents=True, exist_ok=True)
    pfs.to_parquet(out / "pf.parquet", index=False)
    imp.to_csv(out / "importance.csv", index=False)
    written = fi.plot_importance(imp, types, outputs / "figures" / "fig5_feature_importance")
    logging.info("wrote %s", ", ".join(p.name for p in written))
    pd.set_option("display.width", 200)
    print(imp.pivot(index="theme", columns="type", values="fi")[types].round(4).to_string())
    print("\nUtility of the unshuffled runs:", imp[imp["theme"] == fi.BASE].set_index("type")["utility"].round(4).to_dict())


if __name__ == "__main__":
    main()
