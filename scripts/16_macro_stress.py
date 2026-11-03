"""Macro stress tests: Fed-policy scenarios (evaluation of frozen strategies, not a trial; docs/macro_stress.md).

S1 first hike after cuts (2026-09), S2 hold right after a hike (2026-10), S3 the next decision (2026-12, mixed with
the market-implied P(hike)). Historical lens: the saved monthly returns of the baseline methods and of the final
version (net active vs the paper index) over every analogue window, 1981–2025. Current-book lens: the final
version's 2025-11 book's factor exposures applied to each analogue window's factor returns.

    python scripts/16_macro_stress.py [--force]

Needs the 1981–2020 and holdout runs (scripts 04–07, 12 with universe_250.yaml / holdout*.yaml). FRED target series
are cached in data/raw/fred_{series}.parquet (--force re-downloads). Writes outputs/macro_stress/*.csv and
outputs/figures/macro_stress.{png,pdf}.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.config import PROJECT_ROOT, load_config  # noqa: E402
from jkmp.covariance import load_covariance  # noqa: E402
from jkmp.evaluation import macro_stress as ms  # noqa: E402
from jkmp.evaluation.holdout import check_return_months  # noqa: E402
from jkmp.evaluation.metrics import active_returns, active_weights  # noqa: E402
from jkmp.evaluation.regimes import load_fred  # noqa: E402

FINAL = "Final version (net active vs paper index)"
INDEX = "Paper index (top-250 VW)"
PLOT_TYPES = [FINAL, "Portfolio-ML", "Static-ML*"]
SERIES = ("#2a78d6", "#eb6834")       # categorical slots 1, 2 (dataviz reference palette, light mode)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _baseline(outputs) -> pd.DataFrame:
    files = [outputs / "benchmarks" / "pf_test.parquet", outputs / "static_ml" / "pf.parquet",
             outputs / "portfolio_ml" / "rff" / "pf.parquet"]
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _final(cfg, run, dates) -> pd.DataFrame:
    pf = pd.read_parquet(run / "pf.parquet")
    bench = pd.read_parquet(cfg.paths.outputs / "benchmarks" / "pf_test.parquet").query("type == 'Market'").assign(tc=0.0)
    act = active_returns(pf, bench, dates)
    return pd.concat([act.assign(type=FINAL, net=act["active_net"])[["type", "eom_ret", "net", "tc"]],
                      act.assign(type=INDEX, net=act["r_b"], tc=0.0)[["type", "eom_ret", "net", "tc"]]])


def _plot(summary: pd.DataFrame, s3: np.ndarray, s3_stats: dict, p: float, h: int, out) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    scen = ["S1 first hike after cuts", "S2 hold after a hike"]
    x, width = np.arange(len(PLOT_TYPES)), 0.36
    for i, (sc, col) in enumerate(zip(scen, SERIES)):
        vals = [summary.query("scenario == @sc and type == @t and h == 3")["mean"].iloc[0] * 100 for t in PLOT_TYPES]
        n = int(summary.query("scenario == @sc and h == 3")["n"].iloc[0])
        bars = ax1.bar(x + (i - 0.5) * (width + 0.02), vals, width, color=col, label=f"{sc} (n = {n})", zorder=2)
        for b, v in zip(bars, vals):
            ax1.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3 if v >= 0 else -11),
                         textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax1.axhline(0, color=MUTED, lw=0.8)
    ax1.set_xticks(x, ["Final version\n(net active)", "Portfolio-ML\n(net)", "Static-ML*\n(net)"], color=INK)
    ax1.set_ylabel("Mean 3-month return, percentage points", color=MUTED, fontsize=9)
    ax1.set_title("Historical analogues, 1982–2025", loc="left", color=INK, fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax2.hist(s3 * 100, bins=60, color=SERIES[0], edgecolor="#fcfcfb", linewidth=0.5, zorder=2)
    top = ax2.get_ylim()[1]
    for v, lab, ls, dx, ha in ((s3_stats["mean"], "mean", "-", 5, "left"),
                               (s3_stats["p05"], "5th pct", "--", -5, "right")):
        ax2.axvline(v * 100, color=INK, lw=1.2, ls=ls)
        ax2.annotate(f"{lab} {v * 100:+.2f}", (v * 100, top * 0.93), xytext=(dx, 0), textcoords="offset points",
                     ha=ha, fontsize=8.5, color=INK, bbox={"fc": "#fcfcfb", "ec": "none", "pad": 1})
    ax2.set_xlabel(f"Net active return over {h} months, percentage points", color=MUTED, fontsize=9)
    ax2.set_ylabel("Simulated draws", color=MUTED, fontsize=9)
    ax2.set_title(f"S3 December decision, current book (P(hike) = {p:.0%})", loc="left", color=INK, fontsize=11)
    fig.suptitle("Fed-policy stress test: final strategy and baseline methods", x=0.01, ha="left", color=INK,
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"macro_stress.{suffix}", dpi=180, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download the FRED target series")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    base = load_config()
    cfg = {k: load_config(PROJECT_ROOT / f"configs/experiments/{f}.yaml") for k, f in
           (("hold", "holdout"), ("final_test", "universe_250"), ("final_hold", "holdout_final"))}
    m, dates = base.macro_stress, cfg["hold"].dates   # holdout already run once: read-only, descriptive
    out = base.paths.outputs / "macro_stress"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base.seed)

    fred = load_fred(base.data_sources.fred_csv_url, m.target_series, base.paths.raw, force=args.force)
    events = ms.classify_events(ms.policy_moves(ms.policy_target(fred, m.target_series), m.move_threshold),
                                m.s3_outcome_months)
    events.to_csv(out / "policy_events.csv", index=False)
    starts = ms.scenario_starts(events, m.s3_offset_months)

    g = cfg["final_hold"].investor.gamma_te
    base_pf = pd.concat([_baseline(base.paths.outputs), _baseline(cfg["hold"].paths.outputs)], ignore_index=True)
    base_pf = base_pf[base_pf["type"].isin(m.methods)].assign(net=lambda d: d["r"] - d["tc"])
    final = pd.concat([_final(cfg[k], cfg[k].paths.outputs / "active" / f"gte{g:g}", dates)
                       for k in ("final_test", "final_hold")], ignore_index=True)
    monthly = pd.concat([base_pf[["type", "eom_ret", "net", "tc"]], final], ignore_index=True)
    check_return_months(monthly, dates, "macro_stress")

    horizons = {sc: (m.horizons if sc.startswith(("S1", "S2")) else (m.s3_horizon,)) for sc in starts}
    windows = pd.concat([ms.window_sums(monthly, starts[sc], h, ["net", "tc"]).assign(scenario=sc)
                         for sc in starts for h in horizons[sc]], ignore_index=True)
    summary = pd.concat([ms.summarise_windows(w, "net").assign(scenario=sc)
                         for sc, w in windows.groupby("scenario", sort=False)], ignore_index=True)
    s3h = summary[summary["scenario"].str.startswith("S3")].pivot_table(index=["type", "h"], columns="scenario",
                                                                          values="mean").reset_index()
    s3h["mean"] = m.p_hike * s3h["S3 next decision: hike"] + (1 - m.p_hike) * s3h["S3 next decision: no hike"]
    summary = pd.concat([summary, s3h[["type", "h", "mean"]].assign(scenario=f"S3 mixed (P(hike) = {m.p_hike:g})")],
                        ignore_index=True)
    tcr = pd.concat([ms.tc_ratio(w, monthly).assign(scenario=sc) for sc, w in windows.groupby("scenario", sort=False)],
                    ignore_index=True)
    windows.to_csv(out / "windows_historical.csv", index=False)
    summary.to_csv(out / "summary_historical.csv", index=False)
    tcr.to_csv(out / "tc_ratio.csv", index=False)

    fh = cfg["final_hold"]
    run = fh.paths.outputs / "active" / f"gte{g:g}"
    w = pd.read_parquet(run / "weights.parquet")
    last = w["eom"].max()
    panel = pd.read_parquet(fh.paths.processed / "panel.parquet", columns=["id", "eom", "me"],
                            filters=[("eom", "==", last)])
    parts = load_covariance(fh.paths.processed / "covariance")[last]
    b, idio = ms.book_exposure(active_weights(w[w["eom"] == last], panel), parts)
    b.rename("exposure").to_frame().assign(eom=last, idio_var_monthly=idio).to_csv(out / "current_book_exposure.csv")
    daily = pd.read_parquet(fh.paths.processed / "covariance" / "factor_returns.parquet")
    fmon = ms.monthly_factor_returns(daily, parts.factors)

    rows, draws, contrib = [], {}, []
    for sc in starts:
        for h in horizons[sc]:
            gross = ms.book_window_returns(b, fmon, starts[sc], h)
            cost = float(tcr.query("scenario == @sc and type == @FINAL and h == @h")["tc_window"].iloc[0]) * h
            draws[(sc, h)] = ms.simulate(gross, idio, h, cost, m.n_draws, rng)
            rows.append({"scenario": sc, "h": h, "n_windows": len(gross), "cost": cost,
                         **ms.distribution_stats(draws[(sc, h)])})
            f = fmon[b.index]
            per = np.mean([f.loc[pd.date_range(s, periods=h, freq="ME")].sum().to_numpy() for s in gross.index], axis=0)
            contrib.append(pd.Series(per * b.to_numpy(), index=b.index).rename((sc, h)))
    h3 = m.s3_horizon
    s3 = ms.mixture(draws[("S3 next decision: hike", h3)], draws[("S3 next decision: no hike", h3)], m.p_hike, rng)
    s3_stats = ms.distribution_stats(s3)
    rows.append({"scenario": f"S3 mixed (P(hike) = {m.p_hike:g})", "h": h3, "n_windows": np.nan, "cost": np.nan,
                 **s3_stats})
    book = pd.DataFrame(rows)
    book.to_csv(out / "scenarios_current_book.csv", index=False)
    pd.concat(contrib, axis=1).to_csv(out / "current_book_factor_contributions.csv")

    figures = base.paths.outputs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot(summary, s3, s3_stats, m.p_hike, h3, figures)

    pd.set_option("display.width", 220)
    print("Recent policy months:")
    print(events[events["eom"] >= "2026-06-30"][["eom", "target", "move", "prev_move", "s1", "s2", "s3_outcome"]]
          .to_string(index=False))
    print(f"\nAnalogue windows: { {k: len(v) for k, v in starts.items()} }")
    print("\nHistorical lens, mean cumulative net return per window (decimal):")
    print(summary.pivot_table(index=["scenario", "h"], columns="type", values="mean").to_string(float_format="{:+.4f}".format))
    print("\nTrading cost inside windows vs all months (ratio):")
    print(tcr.pivot_table(index=["scenario", "h"], columns="type", values="tc_ratio").to_string(float_format="{:.2f}".format))
    print(f"\nCurrent book ({last.date()}) exposures:\n{b.to_string(float_format='{:+.4f}'.format)}")
    print(f"monthly idiosyncratic active vol {np.sqrt(idio):.4%}")
    print("\nCurrent-book lens, net active return (decimal):")
    print(book.to_string(index=False, float_format="{:+.4f}".format))


if __name__ == "__main__":
    main()
