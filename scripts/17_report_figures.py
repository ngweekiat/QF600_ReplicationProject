"""Report and presentation figures, numbered in report order, from saved outputs (nothing is re-fitted).

New figures: replication vs JKMP Table 2, the paper's Fig. 2 (cumulative gross / net / utility), Table 3 as heatmaps,
post-publication decay, the iteration path, the final strategy's cumulative active return, and the rate-sensitivity /
cost-to-exit stress. Existing figures (Fig. 1A, 1B, 5, capacity, monitor, Fed scenarios, appendix) are copied in.

    python scripts/17_report_figures.py

Needs scripts 04–16 to have run. Writes outputs/figures/report/NN_name.{png,pdf} and INDEX.md (captions).
"""

from __future__ import annotations

import argparse
import logging
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from jkmp.config import PROJECT_ROOT, load_config  # noqa: E402
from jkmp.evaluation.figure_data import cumulative_paths, ir_standard_error, rolling_ir  # noqa: E402
from jkmp.evaluation.metrics import active_returns  # noqa: E402

FINAL = "Active Portfolio-ML (top-250, TE 2%)"
SURF, INK, MUTED, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
# Fixed colour per method in every figure (dataviz reference palette, light mode; colour follows the entity).
COL = {"Portfolio-ML": "#2a78d6", "Static-ML*": "#eb6834", "Portfolio-ML (linear)": "#1baf7a",
       "Static-ML": "#eda100", "Factor-ML": "#e87ba4", "Markowitz-ML": "#008300", FINAL: "#6250d6", "Market": REF}
SHORT = {FINAL: "Final version (active, top-250)"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left", "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": MUTED,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "legend.frameon": False, "lines.linewidth": 2.0})


def _style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _end_labels(ax, items, gap=0.05, dx=6):
    """Direct labels at line ends, nudged vertically so no two are closer than ``gap`` of the axis height."""
    if not items:
        return
    log = ax.get_yscale() == "log"
    f = (lambda v: np.log10(v)) if log else (lambda v: v)
    lo, hi = (f(v) for v in ax.get_ylim())
    step = gap * (hi - lo)
    items = sorted(items, key=lambda it: f(it[1]))
    pos = [f(it[1]) for it in items]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + step)
    for (x, y, text), p in zip(items, pos):
        ax.annotate(text, (x, 10 ** p if log else p), xytext=(dx, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK, annotation_clip=False)


def _save(fig, out, name, title, caption, index):
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=200)
    plt.close(fig)
    index.append((name, title, caption))


def fig_replication(t2, out, index):
    core = ["Market", "1/N", "Static-ML", "Static-ML*", "Portfolio-ML", "Portfolio-ML (linear)"]
    blind = ["Minimum variance", "Factor-ML"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1.1, 0.8]})
    for ax, (metric, rows, label) in zip(axes, [("Utility", core, "Realised utility (γ = 10, annual)"),
                                                ("SR_net", core, "Net Sharpe ratio"),
                                                ("Utility", blind, "Utility, cost-blind methods")]):
        y = np.arange(len(rows))[::-1]
        ours, paper = t2.loc[rows, (metric, "ours")].to_numpy(float), t2.loc[rows, (metric, "paper")].to_numpy(float)
        for yi, o, p in zip(y, ours, paper):
            if np.isfinite(p):
                ax.plot([o, p], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        ax.scatter(paper, y, s=70, facecolors=SURF, edgecolors=INK, linewidths=1.6, zorder=3, label="JKMP Table 2")
        ax.scatter(ours, y, s=70, color=INK, zorder=4, label="Ours")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(y, rows if ax is not axes[1] else [""] * len(rows), color=INK)
        ax.set_xlabel(label)
        _style(ax, "x")
    axes[1].legend(loc="upper right", fontsize=9)
    _save(fig, out, "01_replication_vs_paper", "Replication vs JKMP Table 2, 1981–2020 ($10B, γ = 10)",
          "Ours (filled) vs the paper (hollow) on the paper's test window. Market and the cost-blind methods match "
          "closely; the cost-aware methods have the paper's signs but smaller magnitudes. Markowitz-ML is off-scale "
          "in both (ours: utility −41.5; the paper reports an extreme value as '−'). Portfolio-ML (linear) is our "
          "extension (no paper value).", index)


def fig_paths(paths, out, index):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    panels = [("cum_gross", "Gross return", ["Portfolio-ML", "Static-ML*", "Portfolio-ML (linear)", "Factor-ML"]),
              ("cum_net", "Return net of trading cost", ["Portfolio-ML", "Static-ML*", "Portfolio-ML (linear)"]),
              ("cum_util", "Net of trading cost and risk (utility)",
               ["Portfolio-ML", "Static-ML*", "Portfolio-ML (linear)"])]
    for ax, (col, title, types) in zip(axes, panels):
        ends = []
        for t in types:
            d = paths[paths["type"] == t]
            ax.plot(d["eom_ret"], d[col], color=COL[t], label=t)
            ends.append((d["eom_ret"].iloc[-1], d[col].iloc[-1], f"{d[col].iloc[-1]:.2f}"))
        _end_labels(ax, ends, dx=4)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, color=INK)
        _style(ax)
    axes[0].set_ylabel("Cumulative performance (sum of monthly returns)")
    axes[0].legend(fontsize=8.5, loc="upper left")
    _save(fig, out, "03_performance_over_time", "Performance over time, 1981–2020 (JKMP Fig. 2)",
          "Cumulative sums of monthly gross returns, returns net of trading cost, and utility (net of cost and of "
          "½γ × squared deviation, γ = 10). Factor-ML appears only gross: net of cost it loses about 2.6 a year. "
          "Markowitz-ML is omitted (net −41 a year). Additive, as in the paper.", index)


def fig_table3(ours, paper, out, index):
    cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f0efec", "#2a78d6"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tab, title) in zip(axes, [(ours, "Ours, 1981–2020"), (paper, "JKMP Table 3, 1981–2020")]):
        v = tab.to_numpy(float)
        ax.imshow(np.where(np.isnan(v), 0.5, v), cmap=cmap, vmin=0, vmax=1)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color=GRID))
                    continue
                txt = f"{v[i, j]:.1%}".replace("100.0%", "100%").replace("0.0%", "0%")
                ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                        color="white" if abs(v[i, j] - 0.5) > 0.38 else INK)
        ax.set_xticks(range(len(tab.columns)), tab.columns, rotation=25, ha="right", color=INK)
        ax.set_yticks(range(len(tab.index)), tab.index, color=INK)
        ax.set_title(title, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    _save(fig, out, "04_outperformance_probabilities",
          "Probability that the row method has higher mean utility than the column method",
          "JKMP's Table 3 rule (posterior with a flat prior, normal utility differences). The cost-aware methods beat "
          "Factor-ML and Markowitz-ML with certainty in both; unlike the paper, our Portfolio-ML is not separated from "
          "Static-ML* (55% vs 99.9%), and our linear version beats both. The paper has Multiperiod-ML*, which we "
          "did not replicate.", index)


def fig_decay(decay, out, index):
    types = ["Market", "Static-ML*", "Portfolio-ML", "Portfolio-ML (linear)", FINAL]
    names = {"Portfolio-ML (linear)": "Portfolio-ML (linear)", FINAL: "Final version"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, (metric, label) in zip(axes, [("Utility", "Realised utility (γ = 10, annual)"),
                                          ("SR_net", "Net Sharpe ratio")]):
        ends = []
        for t in types:
            a, b = decay.loc[t, (metric, "1981–2020")], decay.loc[t, (metric, "2021–2025")]
            ax.plot([0, 1], [a, b], color=COL[t], marker="o", ms=7, label=SHORT.get(t, t))
            value = f"{b:+.3f}" if metric == "Utility" else f"{b:.2f}"
            ends.append((1, b, f"{names.get(t, t)}  {value}"))
        ax.axhline(0, color=MUTED, lw=0.8)
        _end_labels(ax, ends, gap=0.06)
        ax.set_xticks([0, 1], ["Test 1981–2020", "Holdout 2021–2025\n(run once)"], color=INK)
        ax.set_xlim(-0.15, 1.75)
        ax.set_ylabel(label)
        _style(ax)
    _save(fig, out, "07_holdout_decay", "Post-publication check: test window vs the untouched holdout",
          "Each line joins a method's 1981–2020 value to its 2021–2025 value. The holdout was run once, after every "
          "model choice was frozen. Market's Sharpe rises in a strong bull market; the ML methods' risk-adjusted "
          "results fall, but the cost-aware methods stay positive or near zero.", index)


def fig_iterations(rows, out, index):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        se = ir_standard_error(r["ir"], r["n"])
        colour = COL[FINAL] if r["final"] else REF
        ax.barh(yi, r["ir"], height=0.55, color=colour, hatch="//" if r["status"] == "killed" else None,
                edgecolor=SURF if r["status"] != "killed" else MUTED, linewidth=0.6, zorder=2)
        ax.errorbar(r["ir"], yi, xerr=1.96 * se, color=INK, lw=1, capsize=3, zorder=3)
        ax.annotate(f"{r['ir']:+.2f}  {r['status']}", (r["ir"] + 1.96 * se, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r["label"] for r in rows], color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Net information ratio vs the benchmark (annual; bars ±1.96 s.e.)")
    ax.set_xlim(-0.8, 1.75)
    _style(ax, "x")
    _save(fig, out, "08_iteration_path", "Iterations: each pre-registered trial and its decision",
          "Net information ratio of each trial against its value-weighted benchmark (trial 3 onward: the paper index, "
          "no benchmark cost). Violet: the kept path to the final version and its holdout. Error bars use the iid "
          "standard error √((1 + IR²/2)/years), so the 5-year holdout is much less precise than 1981–2020.", index)


def fig_final(final_act, top80_act, window, holdout_start, out, index):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    for ax in (a1, a2):
        ax.axvspan(holdout_start, final_act["eom_ret"].max(), color=GRID, zorder=0)
        _style(ax)
    a1.plot(top80_act["eom_ret"], top80_act["active_net"].cumsum() * 100, color=REF, lw=1.6,
            label="Top-80, TE 2% (trial 2b)")
    a1.plot(final_act["eom_ret"], final_act["active_net"].cumsum() * 100, color=COL[FINAL],
            label="Final version: top-250, TE 2% (trial 3)")
    a1.set_ylabel("Cumulative net active return\n(percentage points, additive)")
    a1.legend(fontsize=9, loc="upper left")
    a1.text(holdout_start, 0.04, "  Holdout,\n  run once", transform=a1.get_xaxis_transform(), fontsize=9,
            color=MUTED, va="bottom")
    ir = rolling_ir(final_act, window)
    a2.plot(ir.index, ir, color=COL[FINAL], lw=1.5)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel(f"Rolling {window}-month\nnet information ratio")
    _save(fig, out, "09_final_strategy_active_return",
          "Final strategy vs the paper index, 1981–2025",
          "Benchmark-relative Portfolio-ML on the top-250 universe at a 2% tracking-error budget, net of its trading "
          "cost, against the value-weighted top-250 index without trading cost. Grey: the top-80 version it replaced. "
          "Net IR 0.65 over 1981–2020 and 0.34 over the holdout (shaded).", index)


def fig_rates_exit(betas, exit_tab, base_wealth, out, index):
    pretty = {"theme_low_risk": "Low risk", "theme_value": "Value", "theme_investment": "Investment",
              "theme_quality": "Quality", "theme_momentum": "Momentum", "theme_short_term_reversal": "Short-term reversal",
              "sec_consumer": "Consumer", "sec_industrials": "Industrials", "sec_tech_telecom": "Tech & telecom",
              "sec_energy_utilities": "Energy & utilities", "sec_health": "Health", "sec_finance": "Finance",
              "index": "Paper index (top-250)"}
    b = betas[betas["window"] == betas["window"].iloc[0]].set_index("column")
    order = [c for c in pretty if c.startswith("theme")] + [c for c in pretty if c.startswith("sec")] + ["index"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
    y = np.arange(len(order))[::-1]
    for yi, c in zip(y, order):
        v, se = b.loc[c, "beta"] * 100, b.loc[c, "se"] * 100
        a1.errorbar(v, yi, xerr=1.96 * se, fmt="o", ms=6, color=INK if c != "index" else REF, capsize=3, lw=1)
    a1.set_yticks(y, [pretty[c] for c in order], color=INK)
    a1.axvline(0, color=MUTED, lw=0.8)
    a1.set_xlabel("Monthly return per +1 pp in the 2-year yield (%; ±1.96 s.e.)")
    a1.set_title(f"Rate sensitivity, {b['window'].iloc[0]}", color=INK)
    _style(a1, "x")
    e = exit_tab[exit_tab["months"] == 1]
    a2.set_xscale("log")
    a2.set_yscale("log")
    ends = []
    for pos, colour in (("active book -> index", COL[FINAL]), ("everything -> cash", REF)):
        for k, ls in ((1.0, "-"), (3.0, "--")):
            d = e[(e["position"] == pos) & (e["multiplier"] == k)].sort_values("wealth_2020")
            a2.plot(d["wealth_2020"] / 1e9, d["cost_fraction"] * 100, color=colour, ls=ls, marker="o", ms=4)
            ends.append((d["wealth_2020"].iloc[-1] / 1e9, d["cost_fraction"].iloc[-1] * 100,
                         f"{'Active book' if pos.startswith('active') else 'Everything'}, costs ×{k:g}"))
    a2.axvline(base_wealth / 1e9, color=MUTED, lw=0.8, ls=":")
    a2.text(base_wealth / 1e9, 0.03, " base case ($10B)", transform=a2.get_xaxis_transform(), fontsize=8.5,
            color=MUTED)
    a2.set_xlim(right=1300)
    _end_labels(a2, ends)
    a2.set_xlabel("Fund size at end-2020 ($B, log scale)")
    a2.set_ylabel("Cost to exit at once (% of fund, log scale)")
    a2.set_title("Liquidity stress: cost to exit, 2025-11 book", color=INK)
    _style(a2, "both")
    _save(fig, out, "13_rate_sensitivity_and_exit_cost", "Stress test: rate sensitivity and the cost of de-risking",
          "Left: OLS betas of the risk model's factors and of the index on monthly 2-year Treasury yield changes. Only "
          "the low-risk theme is clearly rate-sensitive; the current active book is close to rate-neutral "
          "(−0.02% per +1 pp). Right: cost of trading the active book back to the index (violet) or everything to cash "
          "(grey) at once, at normal (solid) and tripled (dashed) costs; spreading over 3 months divides by 3. Larger "
          "sizes hold the $10B book fixed, so they are upper bounds.", index)


COPIES = [("02_frontier_by_method", "figures/fig1a_frontier", "Implementable efficient frontier by method, $10B (JKMP Fig. 1A)",
           "Each point is a risk aversion γ; methods are re-tuned at every point."),
          ("05_frontier_by_wealth", "figures/fig1b_frontier_wealth", "Portfolio-ML frontier by investor size (JKMP Fig. 1B)",
           "Larger investors face a lower implementable frontier."),
          ("06_feature_importance", "figures/fig5_feature_importance", "Economic feature importance (JKMP Fig. 5)",
           "Drop in realised utility when a theme's characteristics are permuted (20 seeds, ±1.96 s.e.)."),
          ("10_capacity", "u250/figures/capacity_ir", "Capacity: net information ratio vs fund size (trial 3)",
           "Top-250 vs top-80 at a 2% TE budget; dotted line: the 0.3 capacity floor."),
          ("11_monitor_dashboard", "holdout_final/monitor/dashboard", "Monitoring dashboard, holdout 2021–2025",
           "Monthly risk flags for the final version."),
          ("12_macro_stress_fed_scenarios", "figures/macro_stress", "Stress test: Fed-policy scenarios",
           "Historical analogues of the 2026 hike and hold, and the December scenario for the current book."),
          ("A1_hyperparameters", "figures/figD1_hyperparameters", "Appendix: chosen hyperparameters over time (JKMP Fig. D.1)", ""),
          ("A2_active_theme_exposure", "figures/active_theme_exposure", "Appendix: active theme exposures (trial 2)", "")]


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    base = load_config()
    hold = load_config(PROJECT_ROOT / "configs/experiments/holdout.yaml")
    o = base.paths.outputs
    out = o / "figures" / "report"
    out.mkdir(parents=True, exist_ok=True)
    index: list[tuple[str, str, str]] = []

    t2 = pd.read_csv(o / "tables" / "table2_vs_paper.csv", header=[0, 1], index_col=0)
    fig_replication(t2, out, index)
    pfs = pd.concat([pd.read_parquet(o / f) for f in ("benchmarks/pf_test.parquet", "static_ml/pf.parquet",
                                                      "portfolio_ml/rff/pf.parquet", "portfolio_ml/raw/pf.parquet")])
    fig_paths(cumulative_paths(pfs[pfs["type"].isin(COL)], base.investor.gamma), out, index)
    fig_table3(pd.read_csv(o / "tables" / "table3.csv", index_col=0),
               pd.read_csv(PROJECT_ROOT / "docs" / "paper_tables" / "table3.csv", index_col=0), out, index)
    fig_decay(pd.read_csv(o / "holdout" / "tables" / "decay.csv", header=[0, 1], index_col=0), out, index)

    act = pd.read_csv(o / "active" / "summary.csv")
    active_ir = act.set_index("type")["ir_net"]
    hl = pd.read_csv(o / "u250_hl" / "halflife" / "summary.csv", index_col=0)["ir_net"]
    hold_ir = pd.read_csv(o / "holdout" / "tables" / "final_active.csv", index_col=0)
    rows = [{"label": "Baseline Portfolio-ML vs VW (top-80)", "n": 480, "status": "reference", "final": False,
             "ir": active_ir["Portfolio-ML (baseline) at γ_TE of TE 2%"]},
            *[{"label": f"Trial 2{c}: benchmark-relative, TE {te}%", "n": 480, "status": "kept", "final": False,
               "ir": active_ir[f"Portfolio-ML (active), TE {te}%"]} for c, te in (("a", 1), ("b", 2), ("c", 4))],
            {"label": "Trial 3: top-250 universe, TE 2%", "n": 480, "status": "kept: final", "final": True,
             "ir": hl["trial 3 (h = ∞)"]},
            {"label": "Trial 4: half-life weighting of β", "n": 480, "status": "killed", "final": False,
             "ir": hl["trial 4 (h tuned)"]},
            {"label": "Holdout 2021–2025: final version", "n": int(hold_ir.loc["2021–2025", "n"]), "status": "holdout",
             "final": True, "ir": hold_ir.loc["2021–2025", "ir_net"]}]
    fig_iterations(rows, out, index)

    def _active(run, outputs):
        bench = pd.read_parquet(outputs / "benchmarks" / "pf_test.parquet").query("type == 'Market'").assign(tc=0.0)
        return active_returns(pd.read_parquet(run / "pf.parquet"), bench, hold.dates)
    final_act = pd.concat([_active(o / "u250" / "active" / "gte49.09", o / "u250"),
                           _active(o / "holdout_final" / "active" / "gte49.09", o / "holdout_final")], ignore_index=True)
    top80 = _active(o / "active" / "gte12.82", o)
    fig_final(final_act, top80, base.monitor.ir_window, pd.Timestamp(hold.dates.holdout_start), out, index)

    fig_rates_exit(pd.read_csv(o / "macro_stress" / "rate_betas.csv"), pd.read_csv(o / "macro_stress" / "exit_cost.csv"),
                   base.investor.wealth, out, index)

    for name, src, title, caption in COPIES:
        for suffix in ("png", "pdf"):
            if (o / f"{src}.{suffix}").exists():
                shutil.copy2(o / f"{src}.{suffix}", out / f"{name}.{suffix}")
        index.append((name, title, caption))
    lines = ["# Report figures", "", "Generated by `scripts/17_report_figures.py` from saved outputs. PNG at 200 dpi, plus PDF.", ""]
    for name, title, caption in sorted(index):
        lines += [f"## {name}", f"**{title}**", "", caption, ""]
    (out / "INDEX.md").write_text("\n".join(lines))
    print("\n".join(sorted(n for n, _, _ in index)))


if __name__ == "__main__":
    main()
