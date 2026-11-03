"""Saved-output cross-market and US high-cap iteration comparisons.

No investment model is fitted here. Market references use the existing
market-target and portfolio-accounting routines on each base stock basket.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from workflow import ROOT, UNIVERSES, config_path
from jkmp.config import load_config

sys.path.insert(0, str(ROOT / "scripts"))
from analysis.ml_comparison.data import sha256
from analysis.ml_comparison.metrics import statistics, annual, paths_and_rolling, metric_dictionary
from analysis.ml_comparison.inference import compare_pairs, bootstrap_rank_stability, newey_west_mean

BASE_OUT = ROOT / "outputs/base/overall analysis"
NAMES = {"us_large": "US high-cap", "us_mid": "US mid-cap", "eu_large": "EU high-cap"}
BASE_PF = "US high-cap Portfolio ML base"
STATIC = "US high-cap Static ML"
STATIC_STAR = "US high-cap Static ML*"
MARKET = "US high-cap Market"
ITERATIONS = {i: f"US high-cap Portfolio ML iteration {i}" for i in range(1,11)}
STYLE = """<style>body{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#23313c}h1{font-size:30px}h2{margin-top:32px}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:9px;border:1px solid #d5dee5;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9eff4}.scroll{overflow:auto}img{max-width:100%;margin:18px 0}a{color:#125981}.note{background:#eef3f7;padding:15px;border-radius:6px}</style>"""


def market_dir(universe):
    return BASE_OUT / "market_references" / universe


def build_market_reference(universe):
    """Compute only Market, never unused equal-weight/min-variance strategies."""
    from jkmp.portfolio.benchmarks import market_targets
    from jkmp.portfolio.accounting import add_start_weights, portfolio_returns
    cfg = load_config(config_path(universe))
    panel_path = cfg.paths.processed / "panel.parquet"
    wealth_path = cfg.paths.processed / "wealth.parquet"
    panel, wealth = pd.read_parquet(panel_path), pd.read_parquet(wealth_path)
    dates = pd.date_range(cfg.dates.test_start, cfg.dates.test_end, freq="ME") - pd.offsets.MonthEnd(1)
    universe_rows = panel.loc[panel.in_universe & panel.eom.isin(dates)]
    targets = market_targets(universe_rows)
    weights = add_start_weights(targets, panel, wealth, dates, cfg.portfolio.charge_exit_tc)
    pf = portfolio_returns(weights, panel, wealth).assign(type="Market")
    folder = market_dir(universe)
    folder.mkdir(parents=True, exist_ok=True)
    weights.assign(type="Market").to_parquet(folder / "weights.parquet", index=False)
    pf.to_parquet(folder / "pf.parquet", index=False)
    provenance = dict(universe=universe, method="value-weighted existing base stock basket",
        panel_sha256=sha256(panel_path), wealth_sha256=sha256(wealth_path),
        pf_sha256=sha256(folder / "pf.parquet"), cost_convention="saved modelled quadratic costs included",
        initialisation="same existing add_start_weights convention, first test formation month",
        external_index=False, models_fitted=False)
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return 0


def base_sources():
    records = []
    for universe in UNIVERSES:
        root = ROOT / "outputs/base" / universe
        for method, rel, kind in (("Portfolio ML", "portfolio_ml/rff/pf.parquet", "Portfolio-ML"),
                                  ("Static ML*", "static_ml_star/pf.parquet", "Static-ML*"),
                                  ("Market", None, "Market")):
            if kind=="Static-ML*" and universe!="us_large":continue
            label = f"{NAMES[universe]} {method}" + (" base" if method == "Portfolio ML" else "")
            records.append(dict(strategy=label, universe=universe, group="base",
                path=market_dir(universe) / "pf.parquet" if rel is None else root / rel,
                expected_type=kind, te_target=np.nan))
    return records


def iteration_source(number, target=None):
    return dict(strategy=ITERATIONS[number], universe="us_large", group=f"iteration_{number}",
        path=ROOT / f"outputs/iteration_{number}/us_large/pf.parquet",
        expected_type="Portfolio-ML", te_target=.01 if number in (4,5) else np.nan)


def iteration_comparison_sources(number):
    """Current iteration plus all earlier versions; never include a later one."""
    if number not in range(1,11):raise ValueError("Iteration must be 1 through 10")
    return iteration_baseline_sources()+[iteration_source(i) for i in range(1,number+1)]


def iteration_baseline_sources():
    """Current US high-cap base references, including Static ML*."""
    return [r for r in base_sources() if r['universe']=='us_large']


def load_sources(records):
    """Require the entire declared test calendar; never fill, trim or deduplicate."""
    frames, inventory = [], []
    fields = ["r", "tc", "turnover", "inv", "shorting"]
    for record in records:
        path = record["path"]
        pf = pd.read_parquet(path).copy()
        if set(pf.type) != {record["expected_type"]}:
            raise ValueError(f"Unexpected strategy type: {path}")
        pf["eom_ret"], pf["eom"] = pd.to_datetime(pf.eom_ret), pd.to_datetime(pf.eom)
        cfg = load_config(config_path(record["universe"]))
        expected = pd.date_range(cfg.dates.test_start, cfg.dates.test_end, freq="ME")
        pf = pf.sort_values("eom_ret")
        if not pd.DatetimeIndex(pf.eom_ret).equals(expected):
            raise ValueError(f"Incomplete or duplicate test months: {path}")
        if not (pf.eom + pd.offsets.MonthEnd(1)).eq(pf.eom_ret).all():
            raise ValueError(f"Formation/return mismatch: {path}")
        if not np.isfinite(pf[fields].to_numpy(float)).all():
            raise ValueError(f"Missing or nonfinite performance: {path}")
        frames.append(pf.assign(strategy=record["strategy"], universe=record["universe"], group=record["group"]))
        inventory.append({**{k: v for k, v in record.items() if k != "expected_type"},
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path), "months": len(pf),
            "first_return": str(pf.eom_ret.min().date()), "last_return": str(pf.eom_ret.max().date()),
            "excluded_months": 0})
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["strategy", "eom_ret"]).any():
        raise ValueError("Repeated comparison source")
    return result, pd.DataFrame(inventory)


def relative_metrics(monthly, benchmark):
    """Paper IR uses gross-active SD; net-active SD IR is separately reported."""
    b = benchmark.set_index("eom_ret")
    rows = []
    for label, d in monthly.groupby("strategy", sort=False):
        d = d.set_index("eom_ret").sort_index()
        ref = b.reindex(d.index)
        if ref[["r", "tc"]].isna().any().any():
            raise ValueError("Benchmark dates differ")
        gross = (d.r - ref.r).to_numpy()
        net = (d.r - d.tc - ref.r + ref.tc).to_numpy()
        te_g, te_n = np.std(gross, ddof=1) * np.sqrt(12), np.std(net, ddof=1) * np.sqrt(12)
        test = newey_west_mean(net)
        rows.append(dict(strategy=label, benchmark=benchmark.strategy.iloc[0], months=len(d),
            active_gross_return=12 * gross.mean(), active_net_return=12 * net.mean(),
            tracking_error_gross=te_g, tracking_error_net=te_n,
            information_ratio_paper=12 * net.mean() / te_g if te_g > 1e-12 else np.nan,
            information_ratio_net_sd=12 * net.mean() / te_n if te_n > 1e-12 else np.nan,
            beta_net=np.cov((d.r-d.tc), (ref.r-ref.tc), ddof=1)[0, 1] / (ref.r-ref.tc).var(ddof=1),
            months_beating_benchmark=int((net > 0).sum()), active_net_hac_p=test["p"],
            active_net_hac_ci_low=12 * (test["mean"] - 1.959964 * test["se"]),
            active_net_hac_ci_high=12 * (test["mean"] + 1.959964 * test["se"]),
            inference_note="Unadjusted descriptive HAC test; paired utility and Sharpe files apply Holm"))
    return pd.DataFrame(rows)


def charts(monthly, stats, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = out / "charts"
    folder.mkdir(exist_ok=True)
    palette=plt.get_cmap('tab20').colors
    colors={label:palette[i%len(palette)] for i,label in enumerate(monthly.strategy.unique())}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, d in monthly.groupby("strategy", sort=False):
        d = d.sort_values("eom_ret")
        axes[0].plot(d.eom_ret, 100 * (d.r-d.tc).cumsum(), label=label, color=colors[label])
        axes[1].plot(d.eom_ret, 100 * d.tc.cumsum(), color=colors[label])
    axes[0].set(ylabel="Percentage points", title="Cumulative net excess return   arithmetic")
    axes[1].set(ylabel="Percentage points", title="Cumulative modelled trading costs")
    for ax,values in ((axes[0],monthly.r-monthly.tc),(axes[1],monthly.tc)):
        if abs(values).max()>.5:
            ax.set_yscale("symlog",linthresh=10)
            ax.set_title(ax.get_title()+"   symmetric log scale")
    axes[0].legend(fontsize=9, loc="best")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / "net_returns_and_costs.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(stats))
    axes[0].barh(x, 100*stats.R_TC, color=[colors[s] for s in stats.strategy])
    axes[0].set_yticks(x, stats.strategy, fontsize=9); axes[0].set_xlabel("Annual net excess return (%)")
    axes[1].barh(x, stats.SR_net, color=[colors[s] for s in stats.strategy])
    axes[1].set_yticks(x, [""]*len(x)); axes[1].set_xlabel("Paper net Sharpe")
    if stats.R_TC.abs().max()>1:
        axes[0].set_xscale("symlog",linthresh=10);axes[0].set_xlabel("Annual net excess return (%; symmetric log scale)")
    if stats.SR_net.abs().max()>5:
        axes[1].set_xscale("symlog",linthresh=1);axes[1].set_xlabel("Paper net Sharpe (symmetric log scale)")
    for ax in axes: ax.grid(axis="x", alpha=.2); ax.axvline(0, color="#555", lw=.7)
    fig.tight_layout(); fig.savefig(folder / "return_and_sharpe.png", dpi=160); plt.close(fig)
    rolling = paths_and_rolling(monthly)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for label, d in rolling.groupby("strategy", sort=False):
        axes[0].plot(d.eom_ret, d.rolling_net_sharpe_paper, label=label, color=colors[label])
        axes[1].plot(d.eom_ret, 100*d.drawdown_excess, color=colors[label])
    axes[0].set_title("Rolling 24-month paper net Sharpe"); axes[0].legend(fontsize=8)
    axes[1].set(title="Additive net excess-return drawdown", ylabel="Percentage points")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder / "rolling_and_drawdown.png", dpi=160); plt.close(fig)


def overview_table(stats):
    show = stats[["strategy", "n", "R", "TC", "R_TC", "Vol", "SR_net", "Utility", "Lev", "Turnover"]].copy()
    for c in ("R", "TC", "R_TC", "Vol", "Utility"): show[c] *= 100
    return show.rename(columns={"R":"Gross %", "TC":"Cost %", "R_TC":"Net %", "Vol":"Risk %", "SR_net":"Net Sharpe", "Utility":"Utility %", "Lev":"Leverage"})


def economic_findings(stats, monthly, benchmark, inference):
    ranked = stats.sort_values("Utility", ascending=False)
    best = ranked.iloc[0]
    findings = [f"On the saved 2020-2025 sample, {best.strategy} has the highest mean-variance utility ({best.Utility:.2%} per year at evaluation gamma 10). This is an in-sample comparison of historical backtests, not a forecast of future leadership."]
    for row in stats.itertuples():
        findings.append(f"{row.strategy}: gross excess return {row.R:.2%}, modelled cost {row.TC:.2%}, net excess return {row.R_TC:.2%} per year; paper net Sharpe {row.SR_net:.3f}, gross leverage {row.Lev:.2f} and monthly turnover {row.Turnover:.2%}.")
    rel = relative_metrics(monthly, benchmark)
    for r in rel.loc[rel.strategy.ne(benchmark.strategy.iloc[0])].itertuples():
        findings.append(f"Against {r.benchmark}, {r.strategy} has annual net return difference {r.active_net_return:+.2%} and paper information ratio {r.information_ratio_paper:.3f}. This difference includes market exposure and universe effects.")
    central = inference.loc[inference.bootstrap_block_months.eq(6)]
    significant = int((central.utility_hac_p_holm < .05).sum())
    findings.append(f"Of {len(central)} pairwise utility comparisons, {significant} have Holm-adjusted HAC p below 0.05. Read paired confidence intervals before describing a difference as robust. Six years of monthly results provide limited evidence; the tests do not undo prior research selection.")
    if all(label in set(monthly.strategy) for label in (ITERATIONS[2], ITERATIONS[3])):
        a=monthly.loc[monthly.strategy.eq(ITERATIONS[2])].sort_values("eom_ret")
        b=monthly.loc[monthly.strategy.eq(ITERATIONS[3])].sort_values("eom_ret")
        if np.array_equal(a[["r","tc"]].to_numpy(),b[["r","tc"]].to_numpy()):
            findings.append("Iterations 2 and 3 have exactly identical saved monthly returns and costs over 2020-2025. The half-life search selected no decay in every test year; iteration 3 produces no incremental performance in this test period.")
    if (monthly.r-monthly.tc).le(-1).any():
        findings.append("At least one strategy has a monthly net excess return at or below -100%. Additive paths remain arithmetic model diagnostics, not feasible investor wealth.")
    return findings


def write_report(monthly, inventory, out, title, benchmark, reps=2000, context=""):
    out.mkdir(parents=True, exist_ok=True)
    monthly = monthly.assign(dataset=title)
    stats = pd.DataFrame([dict(strategy=s, **statistics(d)) for s, d in monthly.groupby("strategy", sort=False)])
    monthly.to_csv(out / "monthly_returns.csv", index=False)
    inventory.to_csv(out / "source_inventory.csv", index=False)
    stats.to_csv(out / "performance_summary.csv", index=False)
    annual(monthly).to_csv(out / "annual_performance.csv", index=False)
    paths_and_rolling(monthly).to_csv(out / "rolling_and_cumulative.csv", index=False)
    periods = []
    for start, end in ((2020, 2022), (2023, 2025)):
        for s, d in monthly.loc[monthly.eom_ret.dt.year.between(start, end)].groupby("strategy", sort=False):
            periods.append(dict(period=f"{start}-{end}", strategy=s, **statistics(d)))
    pd.DataFrame(periods).to_csv(out / "subperiod_performance.csv", index=False)
    for field, filename in (("r", "gross_return_correlation.csv"), ("net", "net_return_correlation.csv")):
        monthly.assign(net=monthly.r-monthly.tc).pivot(index="eom_ret", columns="strategy", values=field).corr().to_csv(out / filename)
    inference = pd.concat([compare_pairs(monthly, bootstrap_reps=reps, block_length=b) for b in (3, 6, 12)], ignore_index=True)
    inference.to_csv(out / "paired_inference.csv", index=False)
    bootstrap_rank_stability(monthly, bootstrap_reps=reps).to_csv(out / "bootstrap_rank_stability.csv", index=False)
    relative_metrics(monthly, benchmark).to_csv(out / "versus_market.csv", index=False)
    sensitivities = []
    for multiple in (0, .5, 1, 2):
        for s, d in monthly.groupby("strategy", sort=False):
            sensitivities.append(dict(strategy=s, cost_multiple=multiple, **statistics(d.assign(tc=d.tc*multiple))))
    pd.DataFrame(sensitivities).to_csv(out / "fixed_weight_cost_sensitivity.csv", index=False)
    metric_dictionary().to_csv(out / "metric_dictionary.csv", index=False)
    # Independent direct formulas, separate from the reused metric implementation.
    checks = []
    for row in stats.itertuples():
        d = monthly.loc[monthly.strategy.eq(row.strategy)]
        r = d.r.to_numpy(); net = (d.r-d.tc).to_numpy(); variance = sum((r-sum(r)/len(r))**2)/(len(r)-1)
        passed = all(np.isclose(a, b, rtol=1e-11, atol=1e-12) for a,b in (
            (row.R_TC, 12*sum(net)/len(net)), (row.Utility, 12*(sum(net)/len(net)-5*variance)),
            (row.Vol, np.sqrt(12*variance)), (row.TC, 12*sum(d.tc)/len(d))))
        checks.append(dict(strategy=row.strategy, status="PASS" if passed else "FAIL", months=len(d), excluded_months=0))
    pd.DataFrame(checks).to_csv(out / "independent_verification.csv", index=False)
    if not all(c["status"] == "PASS" for c in checks): raise AssertionError("Comparison formula check failed")
    charts(monthly, stats, out)
    notes = economic_findings(stats, monthly, benchmark, inference)
    methods = """All main sources must contain exactly the same 72 return months from January 2020 through December 2025. No missing return is filled and no month is silently dropped. Returns are USD excess returns; subtract modelled trading costs before net comparisons. Reported annual returns are arithmetic monthly means times 12. Utility uses common evaluation gamma 10 and gross-return sample variance. Paper net Sharpe divides mean net returns by gross-return volatility; the CSV also supplies net-volatility Sharpe. The benchmark is the existing value-weighted stock basket, not a downloaded commercial index. Its own modelled trading costs are included. Cumulative returns add monthly net excess returns, matching the paper replication code. Drawdown is the drop from the additive path's running peak including initial zero; multiply by 100 for percentage points, not percent loss of capital. This drawdown is a reporting extension of the additive chart convention; CAGR and geometric Calmar are omitted. Models update annually on expanding past history; this is not a separate unseen holdout. Paired HAC tests, Holm adjustment within each report and utility/Sharpe family, and paired circular block bootstraps (2000 default replications; 3, 6 and 12 months) describe uncertainty conditional on these saved portfolios. Cross-report testing and earlier research choices are not corrected. Cost scenarios scale saved costs at fixed weights, without reoptimisation. Original missing-return assumptions and initialization conventions are retained. Cross-region rankings reflect different stock universes, opportunity sets and risk exposures; they do not isolate forecasting skill or currency-neutral alpha."""
    md = f"# {title}\n\n{context}\n\n## Economic interpretation\n\n" + "\n\n".join("- "+n for n in notes)
    md += "\n\n## Methods assumptions and exclusions\n\n"+methods+"\n\nSee index.html for readable tables and charts. CSV values are decimal fractions unless the metric dictionary says otherwise.\n"
    (out / "README.md").write_text(md, encoding="utf-8")
    (out / "METHODS_ASSUMPTIONS_AND_EXCLUSIONS.md").write_text(methods+"\n\n"+context, encoding="utf-8")
    links = "".join(f'<li><a href="{f.name}">{html.escape(f.stem.replace("_", " "))}</a></li>' for f in sorted(out.glob("*.csv")))
    body = f"<h1>{html.escape(title)}</h1><p>{html.escape(context)}</p><p class='note'>2020–2025 test return months. Percent columns are annualised; Sharpe, leverage and turnover are ratios. All costs are model estimates.</p>"
    body += '<h2>Performance</h2><div class="scroll">'+overview_table(stats).to_html(index=False, float_format=lambda x:f"{x:,.4g}")+"</div>"
    body += "<h2>Economic interpretation</h2><ul>"+"".join("<li>"+html.escape(n)+"</li>" for n in notes)+"</ul>"
    for f in sorted((out / "charts").glob("*.png")): body += f'<img src="charts/{f.name}" alt="{html.escape(f.stem.replace("_", " "))}">'
    body += "<h2>Methods and limits</h2><p>"+html.escape(methods)+"</p><h2>Data and detailed tests</h2><ul>"+links+"</ul>"
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+"</title>"+STYLE+body, encoding="utf-8")
    return stats


def run_reports(only="all", reps=2000):
    if only in ('final',*(f'iteration_{i}' for i in range(1,11))):
        from iteration_sequence import report_saved
        return report_saved()
    if only=='all':
        from iteration_sequence import report_saved
        report_saved()
        run_reports('base',reps)
        return run_reports('master',reps)
    records = base_sources() if only in ("base","master","all") else iteration_baseline_sources()
    needed = UNIVERSES if only in ("base", "master", "all") else ("us_large",)
    records = [r for r in records if r["universe"] in needed]
    for universe in needed:
        # Rebuild cheaply from source accounting so cached Market cannot go stale.
        build_market_reference(universe)
    data, inventory = load_sources(records)
    benchmark = data.loc[data.strategy.eq(MARKET)]
    if only in ("base", "all"):
        wanted = [r["strategy"] for r in records if r["expected_type"]!="Market" or r["strategy"]==MARKET]
        keep = data.strategy.isin(wanted)
        write_report(data.loc[keep], inventory.loc[inventory.strategy.isin(wanted)], BASE_OUT,
            "Base overall comparison", benchmark, reps,
            "Four base ML results: US high-cap Static ML* and Portfolio ML in US high-cap, US mid-cap and EU high-cap, plus US high-cap Market. Static ML is excluded for US mid-cap and EU high-cap. Same-region Market comparisons are supplied separately.")
        all_stats = pd.DataFrame([dict(strategy=s, **statistics(d)) for s,d in data.groupby("strategy", sort=False)])
        all_stats.to_csv(BASE_OUT / "all_base_models_and_regional_markets.csv", index=False)
        regional = pd.concat([relative_metrics(data.loc[data.universe.eq(u)], data.loc[data.universe.eq(u)&data.type.eq("Market")]).assign(universe=u) for u in UNIVERSES])
        regional.to_csv(BASE_OUT / "same_region_market_comparison.csv", index=False)
        data.to_csv(BASE_OUT / "all_base_monthly_returns.csv", index=False)
    if only in ("master","all"):
        chosen=[r for r in records if r["expected_type"]!="Market" or r["strategy"]==MARKET]
        combined, inv=load_sources(chosen+[iteration_source(i) for i in range(1,11)])
        write_report(combined,inv,BASE_OUT/"all_models_and_iterations",
            "All retained base ML models and US high-cap Portfolio ML iterations",benchmark,reps,
            "Unchanged three-market Base and ten US high-cap sequential experiments. Rejected candidates remain visible but are not inherited. See outputs/iteration_summary for P(better), accepted ancestry and the final strategy.")
    add_navigation(include_iterations=only in ("all","iteration_1","iteration_2","iteration_3"))
    return 0


def add_navigation(include_iterations=True):
    entries = [("Base overall comparison", "outputs/base/overall analysis/index.html"),
               ("Figure 1 implementable frontiers", "outputs/base/overall analysis/figure1/index.html"),
               ("All base ML models and iterations", "outputs/base/overall analysis/all_models_and_iterations/index.html"),
               ("Final US high-cap Portfolio ML progression", "outputs/base/overall analysis/us_high_cap_progression/index.html")]
    entries += [(f"US high-cap iteration {i}", f"outputs/iteration_{i}/us_large/index.html") for i in range(1,11)]
    entries.append(('Ten experiment decisions and final strategy','outputs/iteration_summary/index.html'))
    for title, rel in entries:
        if not include_iterations and rel.startswith("outputs/iteration_"):continue
        page = ROOT / rel
        if not page.exists(): continue
        import os
        links = " · ".join(f'<a href="{os.path.relpath(ROOT/r,page.parent).replace(chr(92),"/")}">{html.escape(t)}</a>' for t,r in entries if (ROOT/r).exists())
        text = page.read_text(encoding="utf-8")
        if "<!-- comparison-navigation -->" in text: text = text.split("<!-- comparison-navigation -->")[0]
        extra="".join(f'<li><a href="{p.name}">{html.escape(p.stem.replace("_"," "))}</a></li>' for p in sorted(page.parent.glob("*.csv")) if f'href="{p.name}"' not in text)
        if extra:text+="<h2>Additional comparisons</h2><ul>"+extra+"</ul>"
        page.write_text(text+"<!-- comparison-navigation --><h2>Related comparisons</h2><p>"+links+"</p>", encoding="utf-8")
    for page in [ROOT/"outputs/index.html", ROOT/"outputs/base/index.html"] + ([ROOT/f"outputs/iteration_{i}/index.html" for i in (1,2,3)] if include_iterations else []):
        if not page.exists(): continue
        import os
        href = os.path.relpath(BASE_OUT/"index.html", page.parent).replace("\\", "/")
        text = page.read_text(encoding="utf-8")
        if "<!-- overall-analysis-link -->" in text: text = text.split("<!-- overall-analysis-link -->")[0]
        page.write_text(text+f'<!-- overall-analysis-link --><h2>Overall comparisons</h2><p><a href="{href}">Base overall analysis and US high-cap iteration comparisons</a></p>', encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-group", choices=("all", "base", *(f'iteration_{i}' for i in range(1,11)), "final", "master"), default="all")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 1: parser.error("bootstrap-reps must be positive")
    return run_reports(args.only_group, args.bootstrap_reps)


if __name__ == "__main__":
    raise SystemExit(main())
