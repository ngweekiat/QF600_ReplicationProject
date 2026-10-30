"""Monthly monitor of the final strategy (trial 3: active Portfolio-ML, top-250, 2% TE budget) — the eight checks of
jkmp.evaluation.monitor over 1981–2025 (test-window run followed by the holdout run), and a one-page dashboard.

    python scripts/11_monitor.py [--from 2011-01-31]

Needs the holdout unlocked (configs/experiments/holdout_final.yaml) and the runs of scripts/12 for
configs/experiments/universe_250.yaml (1981–2020) and holdout_final.yaml (2021–2025). Writes
outputs/holdout_final/monitor/{monitor.csv, dashboard.png, dashboard.html}.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from jkmp.config import PROJECT_ROOT, load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.cost_sensitivity import recost, vol_cost_scale
from jkmp.evaluation.metrics import active_returns, active_weights, ex_ante_exposures
from jkmp.evaluation.monitor import CHECKS, half_life_steps, monitor_table, plot_dashboard
from jkmp.models.return_model import monthly_rank_ic

TEST, HOLD = "configs/experiments/universe_250.yaml", "configs/experiments/holdout_final.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", default="2011-01-31", help="first return month shown on the dashboard")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    test_cfg, cfg = load_config(PROJECT_ROOT / TEST), load_config(PROJECT_ROOT / HOLD)
    g = cfg.investor.gamma_te
    panel = pd.read_parquet(cfg.paths.processed / "panel.parquet")
    wealth = pd.read_parquet(cfg.paths.processed / "wealth.parquet")
    cov = load_covariance(cfg.paths.processed / "covariance")

    pfs, ws, benches, chosen = [], [], [], []
    for c in (test_cfg, cfg):
        run = c.paths.outputs / "active" / f"gte{g:g}"
        pfs.append(pd.read_parquet(run / "pf.parquet"))
        ws.append(pd.read_parquet(run / "weights.parquet"))
        chosen.append(pd.read_csv(run / "chosen.csv"))
        benches.append(pd.read_parquet(c.paths.outputs / "benchmarks" / "pf_test.parquet").query("type == 'Market'"))
    pf, w, bench = pd.concat(pfs), pd.concat(ws), pd.concat(benches).assign(tc=0.0)  # paper index
    act = active_returns(pf, bench, cfg.dates)
    ex = ex_ante_exposures(active_weights(w, panel), cov)
    ic = monthly_rank_ic(pd.read_parquet(cfg.paths.processed / "predictions.parquet"), panel, cfg.dates)
    realised = recost(w, panel, wealth, vol_cost_scale(cov, cfg.reporting.cost_vol_window)).set_index("eom_ret")["tc"]
    ch = pd.concat(chosen).drop_duplicates("year", keep="last")
    hl = half_life_steps(ch, cfg.portfolio_ml.half_life_years) if "h" in ch else None

    table = monitor_table(act, ex, ic, realised, hl, cfg.monitor, cfg.dates)
    out = cfg.paths.outputs / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "monitor.csv", index=False)
    shown = table[table["eom_ret"] >= pd.Timestamp(args.start)].reset_index(drop=True)
    title = (f"Active Portfolio-ML, top-250, TE budget {cfg.monitor.te_budget:.0%} — monthly monitor "
             f"{shown['eom_ret'].min():%Y-%m} .. {shown['eom_ret'].max():%Y-%m} (holdout from 2021-01)")
    written = plot_dashboard(shown, cfg.monitor, title, out / "dashboard")
    logging.info("wrote %s", ", ".join(p.name for p in written))

    pd.set_option("display.width", 200)
    for label, part in (("1981–2020", table[table["eom_ret"] <= pd.Timestamp(cfg.dates.test_end)]),
                        ("2021–2025", table[table["eom_ret"] > pd.Timestamp(cfg.dates.test_end)])):
        print(f"{label}: flagged months per check:",
              {c: int(part[f"flag_{c}"].fillna(False).sum()) for c in CHECKS}, f"of {len(part)}")
    last = table.iloc[-1]
    print(f"Last month {last['eom_ret']:%Y-%m}: " + ", ".join(f"{c}={last[c]:.3f}{' FLAG' if last[f'flag_{c}'] else ''}"
                                                          for c in CHECKS if pd.notna(last[c])))


if __name__ == "__main__":
    main()
