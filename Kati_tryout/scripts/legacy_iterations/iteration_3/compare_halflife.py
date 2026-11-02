"""Trial 4 (docs/research_log.md): time-decayed β (configs/experiments/halflife.yaml) vs trial 3's 250-stock active
Portfolio-ML — the pre-registered net-IR comparison against the paper index, the half-life chosen each year,
active theme exposures over time, and net IR by decade.

    python scripts/14_halflife.py

Writes outputs/u250_hl/halflife/{summary,chosen_h,by_decade}.csv and outputs/u250_hl/figures/halflife_exposures.{png,pdf}.
"""

from __future__ import annotations

from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import configure
configure(__file__)

import argparse
import logging

import numpy as np
import pandas as pd

from jkmp.config import PROJECT_ROOT, load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.active import plot_active_exposures
from jkmp.evaluation.metrics import active_returns, active_stats, active_weights, ex_ante_exposures, prob_greater

REF, VAR = "configs/us_large/iteration_2.yaml", "configs/us_large/iteration_3.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ref, var = load_config(PROJECT_ROOT / REF), load_config(PROJECT_ROOT / VAR)
    dates = var.dates
    cal = pd.read_csv(ref.paths.outputs / "active/calibrated.csv")
    g = float(cal.loc[(cal.te_target - 0.02).abs().lt(1e-12), "gamma_te"].iloc[0])
    panel = pd.read_parquet(var.paths.processed / "panel.parquet")
    cov = load_covariance(var.paths.processed / "covariance")
    bench = (pd.read_parquet(ref.paths.outputs / "benchmarks" / "pf_test.parquet").query("type == 'Market'")
             .assign(tc=0.0))  # paper index (trial 3 decision)
    runs = {"trial 3 (h = ∞)": ref.paths.outputs / "active" / f"gte{g:g}",
            "trial 4 (h tuned)": var.paths.outputs / "active" / f"gte{g:g}"}
    out = var.paths.outputs / "halflife"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    stats, scaled, decades, exposures = {}, {}, {}, []
    for name, d in runs.items():
        pf, w = pd.read_parquet(d / "pf.parquet"), pd.read_parquet(d / "weights.parquet")
        stats[name] = active_stats(pf, bench, w, panel, cov, g, dates)
        act = active_returns(pf, bench, dates)
        scaled[name] = (act["active_net"] / act["active"].std(ddof=1)).reset_index(drop=True)
        dec = act["eom_ret"].dt.year // 10 * 10
        decades[name] = act.groupby(dec).apply(
            lambda x: x["active_net"].mean() / x["active"].std(ddof=1) * np.sqrt(12), include_groups=False)
        exposures.append(ex_ante_exposures(active_weights(w, panel), cov).assign(type=name))
    p_improve = prob_greater(scaled["trial 4 (h tuned)"], scaled["trial 3 (h = ∞)"])
    summary = pd.DataFrame(stats).T
    summary.to_csv(out / "summary.csv")
    by_decade = pd.DataFrame(decades).rename_axis("decade")
    by_decade.to_csv(out / "by_decade.csv")

    chosen = pd.read_csv(runs["trial 4 (h tuned)"] / "chosen.csv")
    test = chosen[chosen["year"].between(dates.test_start.year, dates.test_end.year)]
    share_inf = float(np.isinf(test["h"]).mean())
    test[["year", "tag", "p", "l", "h"]].to_csv(out / "chosen_h.csv", index=False)
    exp = pd.concat(exposures, ignore_index=True)
    written = plot_active_exposures(exp, var.paths.outputs / "figures" / "halflife_exposures")
    logging.info("wrote %s", ", ".join(p.name for p in written))

    print("Active Portfolio-ML, 250-stock, 2% budget, vs the paper index, 2020–2025:")
    print(summary[["active_net", "te_ex_ante", "te_realised", "ir_net", "p_ir_net_pos", "beta", "tc"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nNet IR by decade:")
    print(by_decade.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nChosen half-life by return year (inf = no decay):")
    print(test.groupby("h")["year"].apply(lambda s: f"{len(s)} years: {s.min()}–{s.max()}").to_string())
    print(" ".join(f"{y}:{'∞' if np.isinf(h) else int(h)}" for y, h in zip(test["year"], test["h"])))
    exp["decade"] = (exp["eom"] + pd.offsets.MonthEnd(1)).dt.year // 10 * 10
    print("\nMean active low-risk / quality exposure by decade:")
    print(exp.pivot_table(index="decade", columns="type", values=["theme_low_risk", "theme_quality"])
          .to_string(float_format=lambda v: f"{v:.3f}"))
    killed_inf = share_inf > 0.5
    keep = p_improve >= 0.80 and not killed_inf
    print(f"\nP(net IR trial 4 > trial 3) = {p_improve:.4f}; h = ∞ chosen in {share_inf:.0%} of test years "
          f"-> {'KEEP' if keep else 'KILL'} (pre-registered: P >= 0.80 and h = ∞ in <= 50% of years)")


if __name__ == "__main__":
    main()
