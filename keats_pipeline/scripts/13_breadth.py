"""Trial 3 (docs/research_log.md): 250-stock vs 80-stock active Portfolio-ML — the pre-registered net-IR comparison,
rank IC by size bucket, capacity curves and the cost sensitivity (re-costed weights, no re-training).

    python scripts/13_breadth.py --te 0.02

Reads the reference (configs/experiments/active_pfml.yaml: outputs/active/) and the variant
(configs/experiments/universe_250.yaml: outputs/u250/active/), each with its calibrated γ_TE for the budget
--te, its `run` and its `capacity` outputs (scripts/12_active_pfml.py). Writes outputs/u250/breadth/*.csv and
outputs/u250/figures/capacity_ir.{png,pdf}.
"""

from __future__ import annotations

from _bootstrap import configure
configure(__file__)

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.config import PROJECT_ROOT, load_config  # noqa: E402
from jkmp.covariance import load_covariance  # noqa: E402
from jkmp.evaluation import regimes as rg  # noqa: E402
from jkmp.evaluation.cost_sensitivity import capacity_aum, recost, vol_cost_scale  # noqa: E402
from jkmp.evaluation.metrics import active_returns, active_stats, prob_greater  # noqa: E402
from jkmp.models.return_model import rank_ic_by_size  # noqa: E402

CONFIGS = {"80-stock": "configs/us_large/active_pfml.yaml", "250-stock": "configs/us_large/universe_250.yaml"}


def load_side(name: str, te: float) -> dict:
    cfg = load_config(PROJECT_ROOT / CONFIGS[name])
    out = cfg.paths.outputs / "active"
    cal = pd.read_csv(out / "calibrated.csv")
    gamma_te = float(cal.loc[(cal["te_target"] - te).abs() < 1e-12, "gamma_te"].iloc[0])
    bench = pd.read_parquet(cfg.paths.outputs / "benchmarks" / "pf_test.parquet").query("type == 'Market'")
    return dict(cfg=cfg, gamma_te=gamma_te, run=out / f"gte{gamma_te:g}", capacity=out / "capacity", bench=bench,
                panel=pd.read_parquet(cfg.paths.processed / "panel.parquet"),
                wealth=pd.read_parquet(cfg.paths.processed / "wealth.parquet"),
                cov=load_covariance(cfg.paths.processed / "covariance"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--te", type=float, default=0.02, help="TE budget of the compared runs (a calibrated target)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sides = {name: load_side(name, args.te) for name in CONFIGS}
    cfg = sides["250-stock"]["cfg"]
    dates, rep, inv = cfg.dates, cfg.reporting, cfg.investor
    out = cfg.paths.outputs / "breadth"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    # 1. Pre-registered decision: net IR improvement of 250-stock over 80-stock
    stats, scaled = {}, {}
    for name, s in sides.items():
        pf, w = pd.read_parquet(s["run"] / "pf.parquet"), pd.read_parquet(s["run"] / "weights.parquet")
        stats[name] = active_stats(pf, s["bench"], w, s["panel"], s["cov"], s["gamma_te"], dates)
        act = active_returns(pf, s["bench"], dates)
        scaled[name] = (act["active_net"] / act["active"].std(ddof=1)).reset_index(drop=True)
    p_improve = prob_greater(scaled["250-stock"], scaled["80-stock"])
    summary = pd.DataFrame(stats).T.assign(gamma_te=[sides[n]["gamma_te"] for n in stats])
    summary.to_csv(out / "summary.csv")
    print(f"Active Portfolio-ML at the {args.te:.0%} TE budget, 2020–2025:")
    print(summary[["gamma_te", "active_net", "te_ex_ante", "te_realised", "ir_gross", "ir_net", "p_ir_net_pos",
                   "beta", "neg_weight_share", "tc"]].to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nP(net IR 250-stock > 80-stock) = {p_improve:.4f} -> {'KEEP' if p_improve >= 0.80 else 'KILL'} "
          f"(pre-registered bar 0.80)")

    # 2. Reference-model rank IC by size bucket. Held-specific zero targets can
    # change fitted forecasts between the 80- and 250-stock variants.
    base = sides["80-stock"]
    preds = pd.read_parquet(base["cfg"].paths.processed / "predictions.parquet")
    preds = preds[preds["eom"] + pd.offsets.MonthEnd(1) <= pd.Timestamp(dates.test_end)]
    ic = rank_ic_by_size(preds, base["panel"], rep.ic_size_buckets, dates)
    ic.to_csv(out / "rank_ic_by_size.csv", index=False)
    print("\nReference 80-stock model: mean monthly rank IC by market-cap rank bucket (return years 2015–2025):")
    print(ic.pivot(index="decade", columns="bucket", values="rank_ic").to_string(float_format=lambda v: f"{v:.4f}"))

    # 3. Capacity and cost sensitivity: net IR vs wealth, base costs, VIX-scaled (2020–2025 (VIX subset)), stock-vol-scaled
    vix = rg.monthly_series(rg.load_fred(cfg.data_sources.fred_csv_url, cfg.regimes.fred_series, cfg.paths.raw,
                                         force=False))
    rows = []
    vix_start = pd.Timestamp(rep.cost_vix_start)
    for name, s in sides.items():
        vol_scale = vol_cost_scale(s["cov"], rep.cost_vol_window)
        for level in inv.wealth_grid:
            d = s["capacity"] / f"gte{s['gamma_te']:g}_w{level / 1e9:g}B"
            w = pd.read_parquet(d / "weights.parquet")
            wealth = s["wealth"].assign(wealth=s["wealth"]["wealth"] * level / inv.wealth)
            # "paper index": the benchmark earns its return at no cost (the capacity headline). "traded VW": the
            # benchmark pays its own rebalancing cost at this wealth (∝ wealth), the definition pre-registered at $10B;
            # at large wealth it rewards a fund for skipping the rebalancing it cannot afford.
            benches = {"paper index": s["bench"].assign(tc=0.0),
                       "traded VW": s["bench"].assign(tc=s["bench"]["tc"] * level / inv.wealth)}
            form = pd.DatetimeIndex(sorted(w["eom"].unique()))
            late = form[form + pd.offsets.MonthEnd(1) >= vix_start]
            vix_scale = rg.vix_cost_scale(vix, late, "window")
            runs = {"base": recost(w, s["panel"], wealth, pd.Series(1.0, index=form)),
                    "stock vol": recost(w, s["panel"], wealth, vol_scale)}
            for (model, pf), (bname, bench) in ((r, b) for r in runs.items() for b in benches.items()):
                st = active_stats(pf, bench, w, s["panel"], s["cov"], s["gamma_te"], dates)
                rows.append({"universe": name, "benchmark": bname, "wealth": level, "costs": model,
                             "period": "2020–2025", "ir_net": st["ir_net"], "tc": st["tc"], "te": st["te_realised"]})
            w_late = w[w["eom"].isin(late)]
            for (model, scale), (bname, bench) in ((m, b) for m in (("base", pd.Series(1.0, index=late)), ("VIX", vix_scale))
                                                    for b in benches.items()):
                pf = recost(w_late, s["panel"], wealth, scale)
                st = active_stats(pf, bench[bench["eom"].isin(late)], w_late, s["panel"], s["cov"], s["gamma_te"], dates)
                rows.append({"universe": name, "benchmark": bname, "wealth": level, "costs": model,
                             "period": "2020–2025 (VIX subset)", "ir_net": st["ir_net"], "tc": st["tc"], "te": st["te_realised"]})
    cap = pd.DataFrame(rows)
    cap.to_csv(out / "capacity_curve.csv", index=False)
    aum = (cap.groupby(["benchmark", "universe", "period", "costs"])
           .apply(lambda g: capacity_aum(g["wealth"], g["ir_net"], rep.capacity_ir), include_groups=False)
           .rename("capacity_aum").reset_index())
    aum.to_csv(out / "capacity_aum.csv", index=False)
    print("\nNet IR by wealth ($B), cost model and period:")
    print(cap.assign(wealth=cap["wealth"] / 1e9).pivot_table(index=["benchmark", "universe", "period", "costs"], columns="wealth",
                                                             values="ir_net").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nCapacity: AUM where net IR falls below {rep.capacity_ir} ($B; inf = not reached by $250B):")
    print(aum.assign(capacity_aum=aum["capacity_aum"] / 1e9).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, period in zip(axes, ["2020–2025", "2020–2025 (VIX subset)"]):
        for (name, model), g in cap[(cap["period"] == period) & (cap["benchmark"] == "paper index")].groupby(["universe", "costs"]):
            ax.plot(g["wealth"] / 1e9, g["ir_net"], marker="o", ms=3, lw=1.2,
                    ls="-" if model == "base" else "--", color="C0" if name == "80-stock" else "C3", label=f"{name}, {model}")
        ax.axhline(rep.capacity_ir, color="grey", lw=0.6, ls=":")
        ax.set(xscale="log", title=f"{period}, vs the paper index", xlabel="Wealth at end-2019 ($B, log scale)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel(f"Net information ratio (TE budget {args.te:.0%})")
    fig.tight_layout()
    figs = cfg.paths.outputs / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"capacity_ir.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
