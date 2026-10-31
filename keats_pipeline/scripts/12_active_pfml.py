"""Benchmark-relative Portfolio-ML (research log trial 2, docs/deviations.md D34).

    python scripts/12_active_pfml.py calibrate --config configs/experiments/active_pfml.yaml
        γ_TE grid (investor.gamma_te_grid) -> mean ex-ante TE over the validation years only; interpolated
        γ_TE for each budget in investor.te_targets -> outputs/active/{calibration,calibrated}.csv
    python scripts/12_active_pfml.py run --config configs/experiments/active_pfml.yaml --override investor.gamma_te=X
        one budget over the test window -> outputs/active/gte{X}/{pf,weights,chosen,aims}.*
    python scripts/12_active_pfml.py report --config configs/experiments/active_pfml.yaml
        active statistics vs the VW universe for every calibrated budget and for the baseline Portfolio-ML,
        the pre-registered keep / kill probabilities, and active theme exposures over time
        -> outputs/active/summary.csv, exposures.parquet, outputs/figures/active_theme_exposure.{png,pdf}
    python scripts/12_active_pfml.py capacity --config ... --override investor.gamma_te=X
        the same γ_TE at every investor.wealth_grid level (wealth path rescaled; γ_TE not re-calibrated)
        -> outputs/active/capacity/gte{X}_w{W}B/{pf,weights}.parquet

Needs scripts 01–05 and 07 (panel, Σ, predictions, benchmarks incl. the Market row, baseline Portfolio-ML).
"""

from __future__ import annotations

from _bootstrap import configure
configure(__file__)

import argparse
import dataclasses
import logging
import time

import pandas as pd

from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.data.prepare import risk_free_monthly
from jkmp.evaluation.metrics import (active_returns, active_stats, active_utility_flow, active_weights,
                                     ex_ante_exposures, prob_greater)
from jkmp.evaluation.active import plot_active_exposures
from jkmp.portfolio.active import LABEL, calibrate, calibration_point, run_active
from jkmp.portfolio.grids import grid_inputs, scaled_wealth

MONTH = pd.offsets.MonthEnd(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["calibrate", "run", "capacity", "report"])
    parser.add_argument("--config", default=None, help="YAML config (configs/experiments/active_pfml.yaml)")
    parser.add_argument("--override", action="append", default=[], help="dotted override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.override)
    processed, outputs, inv = cfg.paths.processed, cfg.paths.outputs, cfg.investor
    out = outputs / "active"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    preds = pd.read_parquet(processed / "predictions.parquet")[["id", "eom", "pred_ld1"]]
    panel = pd.read_parquet(processed / "panel.parquet").merge(preds, on=["id", "eom"], how="left")
    wealth = pd.read_parquet(processed / "wealth.parquet")
    cov = load_covariance(processed / "covariance")
    rf = risk_free_monthly(pd.read_parquet(cfg.paths.raw / "ff_factors_monthly.parquet"))
    inp = grid_inputs(cfg, panel, wealth, cov, rf)
    del panel

    if args.step == "calibrate":
        rows = []
        for g in inv.gamma_te_grid:
            start = time.time()
            rows.append(calibration_point(inp, cfg, g))
            logging.info("gamma_te %g: ex-ante TE %.4f (%.1f min)", g, rows[-1]["te_ex_ante"], (time.time() - start) / 60)
        points = pd.DataFrame(rows)
        points.to_csv(out / "calibration.csv", index=False)
        chosen = calibrate(points, inv.te_targets)
        chosen.to_csv(out / "calibrated.csv", index=False)
        print(points.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\nCalibrated gamma_te per ex-ante TE budget (validation years only):")
        print(chosen.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        return

    if args.step == "run":
        start = time.time()
        res = run_active(inp, cfg, inv.gamma_te, inp.eval_dates, inp.hp_years)
        run_dir = out / f"gte{inv.gamma_te:g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("aims", "weights", "pf"):
            res[name].to_parquet(run_dir / f"{name}.parquet", index=False)
        res["chosen"].to_csv(run_dir / "chosen.csv", index=False)
        logging.info("gamma_te %g: test window %s .. %s in %.1f min -> %s", inv.gamma_te,
                     (inp.eval_dates[0] + MONTH).date(), (inp.eval_dates[-1] + MONTH).date(),
                     (time.time() - start) / 60, run_dir)
        return

    if args.step == "capacity":
        for level in inv.wealth_grid:
            start = time.time()
            point = dataclasses.replace(inp, wealth=scaled_wealth(inp.wealth, level, inv.wealth))
            res = run_active(point, cfg, inv.gamma_te, inp.eval_dates, inp.hp_years)
            run_dir = out / "capacity" / f"gte{inv.gamma_te:g}_w{level / 1e9:g}B"
            run_dir.mkdir(parents=True, exist_ok=True)
            for name in ("weights", "pf"):
                res[name].to_parquet(run_dir / f"{name}.parquet", index=False)
            logging.info("wealth $%gB: %.1f min -> %s", level / 1e9, (time.time() - start) / 60, run_dir)
        return

    # report
    budgets = pd.read_csv(out / "calibrated.csv")
    bench_all = pd.read_parquet(outputs / "benchmarks" / "pf_test.parquet")
    bench = bench_all[bench_all["type"] == "Market"]  # the value-weighted universe
    base_pf = pd.read_parquet(outputs / "portfolio_ml" / "rff" / "pf.parquet")
    base_w = pd.read_parquet(outputs / "portfolio_ml" / "rff" / "weights.parquet")
    rows, exposures = [], []
    base_ex = ex_ante_exposures(active_weights(base_w, inp.panel), cov).assign(type="Portfolio-ML (baseline)")
    exposures.append(base_ex)
    for b in budgets.itertuples(index=False):
        run_dir = out / f"gte{b.gamma_te:g}"
        pf, w = pd.read_parquet(run_dir / "pf.parquet"), pd.read_parquet(run_dir / "weights.parquet")
        s = active_stats(pf, bench, w, inp.panel, cov, b.gamma_te, cfg.dates)
        sb = active_stats(base_pf, bench, base_w, inp.panel, cov, b.gamma_te, cfg.dates)
        u = active_utility_flow(active_returns(pf, bench, cfg.dates), b.gamma_te)
        ub = active_utility_flow(active_returns(base_pf, bench, cfg.dates), b.gamma_te)
        p_beat = prob_greater(u.reset_index(drop=True), ub.reset_index(drop=True))
        label = f"{LABEL}, TE {b.te_target:.0%}"
        rows.append({**s.to_dict(), "type": label, "te_target": b.te_target, "gamma_te": b.gamma_te,
                     "p_beats_baseline": p_beat})
        rows.append({**sb.to_dict(), "type": f"Portfolio-ML (baseline) at γ_TE of TE {b.te_target:.0%}",
                     "te_target": b.te_target, "gamma_te": b.gamma_te, "p_beats_baseline": float("nan")})
        exposures.append(ex_ante_exposures(active_weights(w, inp.panel), cov).assign(type=label))
    summary = pd.DataFrame(rows).set_index("type")
    summary.to_csv(out / "summary.csv")
    exp = pd.concat(exposures, ignore_index=True)
    exp.to_parquet(out / "exposures.parquet", index=False)
    written = plot_active_exposures(exp, outputs / "figures" / "active_theme_exposure")
    logging.info("wrote %s", ", ".join(p.name for p in written))

    m = bench
    print(f"Benchmark (VW universe) own trading: turnover {m['turnover'].mean():.4f} a month, "
          f"TC {m['tc'].mean() * 12:.5f} a year")
    cols = ["gamma_te", "active", "active_net", "te_ex_ante", "te_realised", "ir_gross", "ir_net", "p_ir_net_pos",
            "beta", "neg_weight_share", "active_utility", "p_beats_baseline", "tc"]
    print(summary[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    act = summary[summary.index.str.startswith(LABEL)]
    keep = act[act["p_beats_baseline"] >= 0.80]
    killed = not (act["p_ir_net_pos"] >= 0.80).any()
    print(f"\nPre-registered rule: keep budgets {list(keep['te_target'])} (P(beats baseline) >= 80%); "
          f"kill = {killed} (no budget with P(net IR > 0) >= 80%)")


if __name__ == "__main__":
    main()
