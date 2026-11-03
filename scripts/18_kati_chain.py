"""The team's Kati_tryout iteration chain on the replication (research log K4 = trial 2a, trials 6–7).

K4 Market anchor (benchmark-relative Portfolio-ML, top-80, 1% TE) → K4.2 executed validation → K4.3 simpler
selection, judged on 1981–2020 by Kati_tryout's paired block bootstrap of net Sharpe (gate 80%, no waivers).

    python scripts/18_kati_chain.py verify --config configs/experiments/active_pfml.yaml
        the executed-book engine and the choice-driven run must reproduce stored trial 2a exactly
    python scripts/18_kati_chain.py run    --config configs/experiments/active_pfml.yaml
        execute all 1,414 candidates from 1972, select (executed best; one-SE simplest), run K4.2 and K4.3
        -> outputs/kati_chain/{books.npz, candidates.csv, chosen_*.csv, k42/, k43/, pretest_*.csv}
    python scripts/18_kati_chain.py report --config configs/experiments/active_pfml.yaml
        acceptance tests, results table and slides -> outputs/kati_chain/{summary.csv, tests.csv, figures/}
    python scripts/18_kati_chain.py window --config configs/experiments/kati_chain_2020.yaml
        the same chain and baseline on Kati_tryout's window 2020-01 .. 2025-12 (walk-forward through 2025, each
        strategy from value weights at 2019-12); descriptive -> outputs/kati_chain_2020/
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.config import PROJECT_ROOT, load_config  # noqa: E402
from jkmp.covariance import load_covariance  # noqa: E402
from jkmp.data.prepare import risk_free_monthly  # noqa: E402
from jkmp.evaluation.bootstrap import p_better_sharpe  # noqa: E402
from jkmp.evaluation.chain_report import best_stays_best, chain_stats, rank_correlation  # noqa: E402
from jkmp.portfolio.active import active_search, active_with_choices, pfml_search  # noqa: E402
from jkmp.portfolio.executed import (active_series, candidate_aims, candidate_table, choose,  # noqa: E402
                                     executed_books, executed_scores, one_se_pick)
from jkmp.portfolio.grids import grid_inputs  # noqa: E402
from jkmp.portfolio.portfolio_ml import run_portfolio_ml, select_hypers  # noqa: E402
from jkmp.portfolio.recursion import month_inputs  # noqa: E402

MONTH = pd.offsets.MonthEnd(1)
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"Base Portfolio-ML": "#9bb3c6", "Market": INK, "step": "#6250d6"}
NAMES = {"K4": "K4 Market anchor", "K4.2": "K4.2 Executed validation", "K4.3": "K4.3 Simpler selection"}


def _gamma_te(cfg) -> float:
    c = pd.read_csv(cfg.paths.outputs / "active" / "calibrated.csv")
    row = c[np.isclose(c["te_target"], cfg.kati_chain.te_target)]
    if row.empty:
        raise ValueError(f"no calibrated gamma_te for TE {cfg.kati_chain.te_target} (run scripts/12 calibrate)")
    return float(row["gamma_te"].iloc[0])


def _proxy_chosen(results, years):
    validation = pd.concat([r.validation for r in results.values()], ignore_index=True)
    ch = select_hypers(validation)
    return ch[ch["year"].isin(years)], validation


def _decembers(first_year, last_year):
    return [pd.Timestamp(y, 12, 31) for y in range(first_year, last_year + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["verify", "run", "report", "window"])
    parser.add_argument("--config", required=True, help="configs/experiments/active_pfml.yaml (kati_chain_2020.yaml for window)")
    parser.add_argument("--start", default="2020-01-31", help="window step: first return month")
    parser.add_argument("--end", default="2025-12-31", help="window step: last return month")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    kc, out = cfg.kati_chain, cfg.paths.outputs / "kati_chain"
    out.mkdir(parents=True, exist_ok=True)
    gamma = _gamma_te(cfg)
    stored = cfg.paths.outputs / "active" / f"gte{gamma:g}"
    pd.set_option("display.width", 220)

    if args.step == "report":
        return report(cfg, out, gamma, stored)

    processed = cfg.paths.processed
    preds = pd.read_parquet(processed / "predictions.parquet")[["id", "eom", "pred_ld1"]]
    panel = pd.read_parquet(processed / "panel.parquet").merge(preds, on=["id", "eom"], how="left")
    cov = load_covariance(processed / "covariance")
    rf = risk_free_monthly(pd.read_parquet(cfg.paths.raw / "ff_factors_monthly.parquet"))
    inp = grid_inputs(cfg, panel, pd.read_parquet(processed / "wealth.parquet"), cov, rf)
    del panel
    inv, pf_cfg = cfg.investor, cfg.portfolio
    test_years = sorted({(d + MONTH).year for d in inp.eval_dates})

    if args.step == "verify":
        results = active_search(inp, cfg, gamma, inp.eval_dates, inp.hp_years)
        chosen, _ = _proxy_chosen(results, test_years)
        res = active_with_choices(results, chosen, inp, cfg, gamma, inp.eval_dates, "K4")
        ref = pd.read_parquet(stored / "pf.parquet").sort_values("eom_ret")
        a = res["pf"].sort_values("eom_ret")
        d1 = {c: float(np.abs(a[c].to_numpy() - ref[c].to_numpy()).max()) for c in ("r", "tc", "turnover")}
        aims_by = {e: g.set_index("id")["w_aim"] for e, g in res["aims"].groupby("eom")}
        books = executed_books(month_inputs(inp.panel, inp.wealth, inp.cov, inp.eval_dates, mu_col=None),
                               lambda e, ids: aims_by[e].reindex(ids).to_numpy()[:, None], inp.panel, inp.wealth,
                               inp.rf, gamma=gamma, mu=inv.mu, iterations=inv.m_iterations,
                               charge_exit_tc=pf_cfg.charge_exit_tc)
        d2 = {c: float(np.abs(books[c][:, 0] - ref[c].to_numpy()).max()) for c in ("r", "tc", "turnover")}
        (out / "verify.json").write_text(json.dumps({"gamma_te": gamma, "choice_run_vs_stored": d1,
                                                     "engine_vs_stored": d2}, indent=2))
        print(f"choice-driven run vs stored trial 2a: {d1}\nexecuted engine vs stored trial 2a: {d2}")
        if max(d1.values()) > 1e-12 or max(d2.values()) > 1e-10:
            raise AssertionError("the K4.2 machinery does not reproduce trial 2a")
        return

    # run / window: execute every candidate from the first aim month, then select and implement
    formation = pd.DatetimeIndex(sorted(inp.cov))
    ret = formation + MONTH
    eval_dates = inp.eval_dates
    if args.step == "window":
        out = cfg.paths.outputs / "kati_chain_2020"
        out.mkdir(parents=True, exist_ok=True)
        eval_dates = formation[(ret >= pd.Timestamp(args.start)) & (ret <= pd.Timestamp(args.end))]
        test_years = sorted({(d + MONTH).year for d in eval_dates})
    full = formation[(ret.year > inp.hp_years[0]) & (ret <= eval_dates[-1] + MONTH)]
    results = active_search(inp, cfg, gamma, full, inp.hp_years)
    table = candidate_table(results, cfg.ridge_grid.values)
    table.to_csv(out / "candidates.csv", index=False)
    books = executed_books(month_inputs(inp.panel, inp.wealth, inp.cov, full, mu_col=None),
                           candidate_aims(results, table), inp.panel, inp.wealth, inp.rf, gamma=gamma, mu=inv.mu,
                           iterations=inv.m_iterations, charge_exit_tc=pf_cfg.charge_exit_tc)
    np.savez_compressed(out / "books.npz", eom_ret=books["eom_ret"].to_numpy(), r=books["r"], tc=books["tc"],
                        turnover=books["turnover"])
    bench = _market(cfg, oos=True)
    net, gross = active_series(books, bench)

    first_dec, last_dec = books["eom_ret"][0].year, test_years[-1] - 1
    decisions = _decembers(first_dec, last_dec)
    scores = executed_scores(books, net, gross, decisions, gamma)
    proxy_ch, validation = _proxy_chosen(results, range(first_dec + 1, last_dec + 2))
    v = validation.merge(table, on=["tag", "p", "l_no"], how="inner")
    picks42, picks43, rows = [], [], []
    for i, d in enumerate(decisions):
        mask = books["eom_ret"] <= d
        k43, info = one_se_pick(table, scores[i], net[mask], gross[mask], gamma, kc.se_reps, kc.se_block, kc.se_seed)
        picks42.append(info["best"])
        picks43.append(k43)
        if d <= pd.Timestamp(kc.pretest_end):
            proxy = v[v["eom_ret"] == d].set_index("cand")["cum_obj"].reindex(table["cand"]).to_numpy()
            pw = int(proxy_ch.loc[proxy_ch["eom_ret"] == d, ["tag", "p", "l_no"]].merge(table, on=["tag", "p", "l_no"])["cand"].iloc[0])
            ex_rank = int((scores[i] > scores[i][pw]).sum()) + 1
            m_net, m_gross = net[mask], gross[mask]
            rows.append({"december": d.date(), "months": int(mask.sum()), "rank_corr": rank_correlation(proxy, scores[i]),
                         "proxy_winner": pw, "executed_winner": info["best"], "choice_differs": pw != info["best"],
                         "executed_rank_of_proxy_winner": ex_rank,
                         "gross_active_proxy": 12 * m_gross[:, pw].mean(), "gross_active_executed": 12 * m_gross[:, info["best"]].mean(),
                         "tc_proxy": 12 * books["tc"][mask][:, pw].mean(), "tc_executed": 12 * books["tc"][mask][:, info["best"]].mean(),
                         "score_best": scores[i][info["best"]], "se_best": info["se"], "n_eligible": info["n_eligible"],
                         "best_stays_best": best_stays_best(m_net, m_gross, gamma, kc.se_reps, kc.se_block, kc.se_seed),
                         "p_best": int(table.loc[info["best"], "p"]), "p_one_se": int(table.loc[k43, "p"]),
                         "l_best": table.loc[info["best"], "l"], "l_one_se": table.loc[k43, "l"],
                         "score_one_se": scores[i][k43]})
    pd.DataFrame(rows).to_csv(out / "pretest_diagnostics.csv", index=False)
    test_dec = [d for d in decisions if d.year + 1 in test_years]
    sel = [i for i, d in enumerate(decisions) if d in test_dec]
    for name, picks in (("k42", picks42), ("k43", picks43)):
        chosen = choose(table, [picks[i] for i in sel], test_dec, [scores[i][picks[i]] for i in sel])
        chosen.to_csv(out / f"chosen_{name}.csv", index=False)
        res = active_with_choices(results, chosen, inp, cfg, gamma, eval_dates, name)
        (out / name).mkdir(exist_ok=True)
        for f in ("pf", "weights", "aims"):
            res[f].to_parquet(out / name / f"{f}.parquet", index=False)
    if args.step == "run":
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        return
    # window: K4 with the original selection, the total-return baseline, Market; then the same report
    proxy, _ = _proxy_chosen(results, test_years)
    k4 = active_with_choices(results, proxy, inp, cfg, gamma, eval_dates, "K4")
    stored_aims = pd.read_parquet(stored / "aims.parquet")
    a20 = k4["aims"][k4["aims"]["eom"].isin(stored_aims["eom"])].merge(stored_aims, on=["id", "eom"], suffixes=("", "_test"))
    gap = float((a20["w_aim"] - a20["w_aim_test"]).abs().max()) if len(a20) else float("nan")
    logging.info("K4 aims for 2020 vs the stored 1981-2020 run: %d rows, max |diff| %.2e", len(a20), gap)
    base_cfg = load_config(PROJECT_ROOT / "configs/experiments/holdout.yaml")
    base_res = pfml_search(inp, base_cfg, base_cfg.investor.gamma, eval_dates, inp.hp_years)
    base = run_portfolio_ml(base_res, inp.panel, inp.wealth, inp.cov, inp.rf, eval_dates, eval_dates,
                            gamma=base_cfg.investor.gamma, mu=inv.mu, iterations=inv.m_iterations,
                            charge_exit_tc=pf_cfg.charge_exit_tc, label="Portfolio-ML")
    for name, res in (("k4", k4), ("base", base)):
        (out / name).mkdir(exist_ok=True)
        res["pf"].to_parquet(out / name / "pf.parquet", index=False)
        res["weights"].to_parquet(out / name / "weights.parquet", index=False)
    market = _market(cfg, oos=False)
    market = market[market["eom_ret"].between(pd.Timestamp(args.start), pd.Timestamp(args.end))]
    (out / "window.json").write_text(json.dumps({"start": args.start, "end": args.end, "gamma_te": gamma,
                                                 "k4_aims_2020_vs_test_max_abs_diff": gap}, indent=2))
    pfs = {k: pd.read_parquet(out / d / "pf.parquet") for k, d in (("Base", "base"), ("K4", "k4"), ("K4.2", "k42"), ("K4.3", "k43"))}
    summarise(out, pfs, market, cfg.kati_chain, f"{args.start[:4]}–{args.end[:4]}", pretest=False)


def _market(cfg, oos: bool) -> pd.DataFrame:
    """The value-weighted top-80 universe with its own trading cost: 1971–2020 (``oos``) or 1981–2020 from the test
    run, continued with the holdout run's 2021–2025 when the config is unlocked (value weights restart at 2020-12)."""
    main = load_config().paths.outputs / "benchmarks"
    b = pd.read_parquet(main / ("pf_oos.parquet" if oos else "pf_test.parquet")).query("type == 'Market'")
    if not cfg.dates.holdout_locked:
        h = load_config(PROJECT_ROOT / "configs/experiments/holdout.yaml").paths.outputs / "benchmarks" / "pf_test.parquet"
        b = pd.concat([b, pd.read_parquet(h).query("type == 'Market'")], ignore_index=True)
    if b["eom_ret"].duplicated().any():
        raise ValueError("overlapping Market rows")
    return b.sort_values("eom_ret").reset_index(drop=True)


def report(cfg, out, gamma, stored) -> None:
    o = cfg.paths.outputs
    market = pd.read_parquet(o / "benchmarks" / "pf_test.parquet").query("type == 'Market'")
    pfs = {"Base": pd.read_parquet(o / "portfolio_ml" / "rff" / "pf.parquet"), "K4": pd.read_parquet(stored / "pf.parquet"),
           "K4.2": pd.read_parquet(out / "k42" / "pf.parquet"), "K4.3": pd.read_parquet(out / "k43" / "pf.parquet")}
    summarise(out, pfs, market, cfg.kati_chain, "1981–2020", pretest=True)


def summarise(out, pfs, market, kc, window: str, pretest: bool) -> None:
    """Kati_tryout's acceptance tests and results table for Base → K4 → K4.2 → K4.3 over one window, and its slides."""
    pfs = {k: v.sort_values("eom_ret").reset_index(drop=True) for k, v in pfs.items()}
    if any(not v["eom_ret"].equals(market.sort_values("eom_ret")["eom_ret"].reset_index(drop=True)) for v in pfs.values()):
        raise ValueError("strategies and Market cover different months")
    m = market.sort_values("eom_ret").reset_index(drop=True)

    def p_test(a, b):
        return p_better_sharpe(a["r"].to_numpy(), a["tc"].to_numpy(), b["r"].to_numpy(), b["tc"].to_numpy(),
                               kc.accept_reps, kc.accept_block, kc.accept_seed)

    rows, tests, accepted, ref = {}, [], [], "Base"
    for step, reference in (("K4", "Base"), ("K4.2", "K4"), ("K4.3", "K4.2")):
        t = p_test(pfs[step], pfs[reference])
        tm = p_test(pfs[step], m)
        passed = t["p_better"] >= kc.accept_threshold
        tests.append({"step": step, "reference": reference, **t, "accepted": passed,
                      "p_better_vs_market": tm["p_better"]})
        if passed:
            accepted.append(step)
    for k, v in {**pfs, "Market": m}.items():
        rows[k] = chain_stats(v, m)
    summary, tests = pd.DataFrame(rows).T, pd.DataFrame(tests)
    summary.to_csv(out / "summary.csv")
    tests.to_csv(out / "tests.csv", index=False)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n" + tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    final = accepted[-1] if accepted else "Base"
    print(f"\nAccepted in sequence (gate {kc.accept_threshold:.0%}, no waivers): {accepted or 'none'}; final: {final}")
    (out / "decision.json").write_text(json.dumps({"window": window, "accepted": accepted, "final": final}, indent=2))
    figures(out, pfs, m, summary, tests, window, pretest)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figures(out, pfs, market, summary, tests, window: str, pretest: bool) -> None:
    fd = out / "figures"
    fd.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                         "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    mnet = (market["r"] - market["tc"]).to_numpy()
    for step in ("K4", "K4.2", "K4.3"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
        for label, pf, colour, lw in (("Base Portfolio-ML", pfs["Base"], COL["Base Portfolio-ML"], 1.2),
                                      ("Market", market, COL["Market"], 1.6), (NAMES[step], pfs[step], COL["step"], 2.0)):
            net = (pf["r"] - pf["tc"]).to_numpy()
            cum = np.cumsum(net) * 100
            axes[0].plot(pf["eom_ret"], cum, color=colour, lw=lw, label=label)
            axes[1].plot(pf["eom_ret"], cum - np.maximum.accumulate(np.r_[0.0, cum])[1:], color=colour, lw=lw)
        rel = np.cumsum((pfs[step]["r"] - pfs[step]["tc"]).to_numpy() - mnet) * 100
        axes[2].plot(pfs[step]["eom_ret"], rel, color=COL["step"], lw=2)
        axes[2].fill_between(pfs[step]["eom_ret"], 0, rel, color=COL["step"], alpha=0.12)
        axes[2].axhline(0, color=MUTED, lw=0.8)
        for ax, t in zip(axes, ("Cumulative net return (additive)", "Additive drawdown from peak",
                                "Cumulative return minus Market")):
            ax.set_title(t, loc="left", color=INK)
            ax.set_ylabel("Percentage points")
            _style(ax)
        axes[0].legend(frameon=False, fontsize=8)
        s, t = summary.loc[step], tests[tests["step"] == step].iloc[0]
        fig.suptitle(f"{NAMES[step]} vs the market, {window} — net return {s['net_return']:.2%}/yr (Market "
                     f"{summary.loc['Market', 'net_return']:.2%}), net Sharpe {s['net_sharpe']:.3f}, active "
                     f"{s['active_return']:+.2%}/yr, P(better vs {t['reference']}) {t['p_better']:.1%}"
                     f"{' — accepted' if t['accepted'] else ' — not accepted'}", x=0.01, ha="left", fontsize=10.5,
                     fontweight="bold", color=INK)
        fig.tight_layout()
        fig.savefig(fd / f"{step.replace('.', '_')}_vs_market.png", dpi=180)
        plt.close(fig)
    if not pretest:
        return
    d = pd.read_csv(out / "pretest_diagnostics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    axes[0].bar(d["december"].str[:4], d["rank_corr"], color=COL["step"])
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("K4.2: rank correlation, original vs executed score", loc="left", color=INK)
    axes[1].errorbar(d["december"].str[:4], d["score_best"] * 100, yerr=d["se_best"] * 100, fmt="o", color=INK,
                     capsize=3, label="Best executed score ± 1 SE")
    axes[1].scatter(d["december"].str[:4], d["score_one_se"] * 100, color=COL["step"], marker="s", zorder=3,
                    label="One-SE simpler choice")
    axes[1].set_title("K4.3: best score vs estimation noise", loc="left", color=INK)
    axes[1].set_ylabel("Annual executed active utility (%)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(d["december"].str[:4], d["p_best"], marker="o", color=INK, label="Highest executed score")
    axes[2].plot(d["december"].str[:4], d["p_one_se"], marker="s", color=COL["step"], label="One-SE simpler choice")
    axes[2].set_yscale("log", base=2)
    axes[2].set_title("K4.3: RFF features of the annual choice", loc="left", color=INK)
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        _style(ax)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Pretest diagnostics, Decembers 1972–1980 (no test-window outcomes)", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(fd / "pretest_diagnostics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
