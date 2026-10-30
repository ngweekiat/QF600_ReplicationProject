"""Holdout 2021–2025 (run once; Phase 7): Table 2 and Table 3 for the baseline methods and the final kept version,
the 1981–2020 vs 2021–2025 comparison (post-publication decay), the final model's active statistics in both
periods, and its deflated Sharpe ratio (Bailey and López de Prado, 2014) over the selection period 1981–2020 with
the trial count of docs/research_log.md.

    python scripts/15_holdout.py

Needs the holdout runs of scripts 03–07 (configs/experiments/holdout{,_linear}.yaml) and 04, 12
(holdout_final.yaml), plus the 1981–2020 runs. Writes outputs/holdout/tables/*.csv.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from jkmp.config import PROJECT_ROOT, load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.metrics import (active_returns, active_stats, deflated_sharpe, prob_outperformance, table2)

FINAL = "Active Portfolio-ML (top-250, TE 2%)"
TYPES = ["Portfolio-ML", "Portfolio-ML (linear)", "Static-ML*", FINAL, "Factor-ML", "Markowitz-ML"]
N_TRIALS = 6  # docs/research_log.md: trials 1, 2a, 2b, 2c, 3, 4


def _pfs(outputs, final_run) -> pd.DataFrame:
    files = [outputs / "benchmarks" / "pf_test.parquet", outputs / "static_ml" / "pf.parquet",
             outputs / "portfolio_ml" / "rff" / "pf.parquet", outputs / "portfolio_ml" / "raw" / "pf.parquet"]
    parts = [pd.read_parquet(f) for f in files if f.exists()]
    parts.append(pd.read_parquet(final_run / "pf.parquet").assign(type=FINAL))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = {k: load_config(PROJECT_ROOT / f"configs/experiments/{f}.yaml") for k, f in
           (("hold", "holdout"), ("final_hold", "holdout_final"), ("final_test", "universe_250"))}
    base = load_config()
    g = cfg["final_hold"].investor.gamma_te
    hold_dates, gamma = cfg["hold"].dates, base.investor.gamma
    out = cfg["hold"].paths.outputs / "tables"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    fmt = lambda v: f"{v:.3f}"  # noqa: E731

    test = _pfs(base.paths.outputs, cfg["final_test"].paths.outputs / "active" / f"gte{g:g}")
    hold = _pfs(cfg["hold"].paths.outputs, cfg["final_hold"].paths.outputs / "active" / f"gte{g:g}")
    t2h = table2(hold, gamma, hold_dates)
    t2t = table2(test, gamma, base.dates)
    t2h.to_csv(out / "table2_holdout.csv")
    print(f"Table 2 — holdout, returns {hold['eom_ret'].min():%Y-%m} .. {hold['eom_ret'].max():%Y-%m}:")
    print(t2h.to_string(float_format=fmt))
    t3 = prob_outperformance(hold, gamma, hold_dates, TYPES)
    t3.to_csv(out / "table3_holdout.csv")
    print("\nTable 3 — holdout, P(row has higher mean utility than column):")
    print(t3.to_string(float_format=fmt))

    cols = ["R", "Vol", "SR_net", "TC", "Utility"]
    decay = pd.concat({"1981–2020": t2t[cols], "2021–2025": t2h[cols].reindex(t2t.index)}, axis=1).swaplevel(axis=1)
    decay = decay[[(c, p) for c in cols for p in ("1981–2020", "2021–2025")]]
    decay.to_csv(out / "decay.csv")
    print("\nPost-publication comparison, 1981–2020 vs 2021–2025:")
    print(decay.to_string(float_format=fmt))

    rows = {}
    for label, key in (("1981–2020", "final_test"), ("2021–2025", "final_hold")):
        c = cfg[key]
        run = c.paths.outputs / "active" / f"gte{g:g}"
        bench = pd.read_parquet(c.paths.outputs / "benchmarks" / "pf_test.parquet").query("type == 'Market'").assign(tc=0.0)
        rows[label] = active_stats(pd.read_parquet(run / "pf.parquet"), bench, pd.read_parquet(run / "weights.parquet"),
                                   pd.read_parquet(c.paths.processed / "panel.parquet"),
                                   load_covariance(c.paths.processed / "covariance"), g, c.dates)
        if label == "1981–2020":
            act = active_returns(pd.read_parquet(run / "pf.parquet"), bench, c.dates)
    active = pd.DataFrame(rows).T
    active.to_csv(out / "final_active.csv")
    print(f"\n{FINAL} vs the paper index (top-250 VW):")
    print(active[["active_net", "te_ex_ante", "te_realised", "ir_gross", "ir_net", "p_ir_net_pos", "beta", "tc"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    # Deflated Sharpe ratio over the selection period, on the TE-scaled net active return (its Sharpe = net IR).
    trial_ir = _trial_irs(base)
    dsr = deflated_sharpe(act["active_net"] / act["active"].std(ddof=1), np.asarray(trial_ir) / np.sqrt(12), N_TRIALS)
    pd.Series(dsr).to_csv(out / "deflated_sharpe.csv")
    print(f"\nDeflated Sharpe ratio (net IR, 1981–2020; {N_TRIALS} trials, trial IRs {np.round(trial_ir, 3).tolist()}):")
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in dsr.items()})


def _trial_irs(base) -> list[float]:
    """Annual net IRs of the six trials vs their benchmarks, from the trial summaries (research log)."""
    o = base.paths.outputs
    t2 = pd.read_csv(o / "active" / "summary.csv", index_col=0)
    b3 = pd.read_csv(o / "u250" / "breadth" / "summary.csv", index_col=0)
    h4 = pd.read_csv(o / "u250_hl" / "halflife" / "summary.csv", index_col=0)
    baseline = float(t2.loc[t2.index.str.startswith("Portfolio-ML (baseline)"), "ir_net"].iloc[0])
    actives = t2.loc[t2.index.str.startswith("Portfolio-ML (active)"), "ir_net"].tolist()
    return [baseline, *actives, float(b3.loc["top-250", "ir_net"]), float(h4.loc["trial 4 (h tuned)", "ir_net"])]


if __name__ == "__main__":
    main()
