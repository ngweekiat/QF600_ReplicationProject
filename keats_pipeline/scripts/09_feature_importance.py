"""Feature importance by theme (Fig. 5): permute each theme's characteristics across universe stocks within
each month, keep W, β and hyperparameters, and measure the drop in realised utility over the test window.

  Portfolio-ML (net of TC) and, if built, Portfolio-ML (linear): same aim coefficients as scripts/07.
  Markowitz-ML (gross): pred_ld1 recomputed with each year's fitted return model (scripts/03 models.npz),
  w = Σ⁻¹μ/γ at feature_importance.markowitz_gamma with zero wealth, as R's tpf_cf_fun.
Repeated for feature_importance.n_seeds permutation seeds; importance = the mean over seeds, with its standard
error. Writes outputs/feature_importance/{pf.parquet, importance_by_seed.csv, importance.csv} and outputs/figures/fig5_feature_importance.{png,pdf}.

    python scripts/09_feature_importance.py
"""

from __future__ import annotations

from _bootstrap import configure
configure(__file__)

import argparse
import logging
import time

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
    seeds = cfg.feature_importance.seeds(cfg.seed)
    perms = {seed: fi.permutations(universe, seed) for seed in seeds}
    themes = fi.theme_list(cfg.features.themes)
    shuffled_themes = [(t, f) for t, f in themes if f is not None]
    inputs = month_inputs(panel, wealth, cov, dates, mu_col=None)
    common = dict(gamma=inv.gamma, mu=inv.mu, iterations=inv.m_iterations, charge_exit_tc=cfg.portfolio.charge_exit_tc)
    logging.info("%d permutation seeds (%d .. %d), %d themes, %d months", len(seeds), seeds[0], seeds[-1],
                 len(shuffled_themes), len(dates))

    runs, types = [], []
    for sig, label in LABELS.items():
        path = outputs / "portfolio_ml" / sig / "aim_coefs.parquet"
        if not path.exists():
            logging.info("skip %s: %s not found", label, path)
            continue
        coefs = pd.read_parquet(path)
        coefs = coefs[coefs["eom"].isin(dates)]
        specs = {tag: fi.load_signal_spec(processed / "pfml" / tag) for tag in coefs["tag"].unique()}
        # The unshuffled run does not depend on the permutation: compute it once, reuse it for every seed.
        base = fi.pfml_counterfactual(None, universe, perms[seeds[0]], coefs, specs, inputs, panel, wealth, cov, rf,
                                      **common)
        ref = pd.read_parquet(outputs / "portfolio_ml" / sig / "pf.parquet")
        gap = float(np.abs(base.sort_values("eom")["r"].to_numpy() - ref.sort_values("eom")["r"].to_numpy()).max())
        logging.info("%s unshuffled vs scripts/07: max |Δr| = %.2e", label, gap)
        for seed in seeds:
            start = time.time()
            runs.append(base.assign(type=label, theme=fi.BASE, seed=seed))
            for theme, feats in shuffled_themes:
                pf = fi.pfml_counterfactual(feats, universe, perms[seed], coefs, specs, inputs, panel, wealth, cov, rf,
                                            **common)
                runs.append(pf.assign(type=label, theme=theme, seed=seed))
            logging.info("%s seed %d: %.1f s", label, seed, time.time() - start)
        types.append(label)

    models, z = load_models(processed / "return_model" / "models.npz")
    wealth0 = wealth.assign(wealth=0.0)  # R: wealth_func(wealth_end = 0) -> no trading costs
    mkw = dict(gamma=cfg.feature_importance.markowitz_gamma, charge_exit_tc=cfg.portfolio.charge_exit_tc)
    base = fi.markowitz_counterfactual(None, universe, perms[seeds[0]], cfg.features.names, models, z, panel, wealth0,
                                       cov, **mkw)
    for seed in seeds:
        start = time.time()
        runs.append(base.assign(type="Markowitz-ML", theme=fi.BASE, seed=seed))
        for theme, feats in shuffled_themes:
            pf = fi.markowitz_counterfactual(feats, universe, perms[seed], cfg.features.names, models, z, panel,
                                             wealth0, cov, **mkw)
            runs.append(pf.assign(type="Markowitz-ML", theme=theme, seed=seed))
        logging.info("Markowitz-ML seed %d: %.1f s", seed, time.time() - start)
    types.append("Markowitz-ML")

    pfs = pd.concat(runs, ignore_index=True)
    by_seed = fi.importance(pfs, inv.gamma, cfg.dates)
    imp = fi.summarise_seeds(by_seed)
    out = outputs / "feature_importance"
    out.mkdir(parents=True, exist_ok=True)
    pfs.to_parquet(out / "pf.parquet", index=False)
    by_seed.to_csv(out / "importance_by_seed.csv", index=False)
    imp.to_csv(out / "importance.csv", index=False)
    written = fi.plot_importance(imp, types, outputs / "figures" / "fig5_feature_importance")
    logging.info("wrote %s", ", ".join(p.name for p in written))
    pd.set_option("display.width", 200)
    print(f"Importance (mean over {len(seeds)} seeds; standard error in brackets):")
    cell = imp.assign(cell=imp["fi"].map("{:.4f}".format) + " (" + imp["fi_se"].map("{:.4f}".format) + ")")
    print(cell.pivot(index="theme", columns="type", values="cell")[types].to_string())
    print("\nUtility of the unshuffled runs:", imp[imp["theme"] == fi.BASE].set_index("type")["utility"].round(4).to_dict())

if __name__ == "__main__":
    main()
