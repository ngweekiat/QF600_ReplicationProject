# Validation report: Portfolio-ML research framework

Independent validation (SR 11-7-style effective challenge) of the repository against `docs/research_framework.md`
(2026-10-29), judged in the superseded form S1–S3 of `docs/validation_prompt.md`. Date: 2026-10-30.

**Method.**
- Read-only, apart from writing this file. No pipeline script (00–15) was re-run, and the holdout was not regenerated.
- Ran `.venv/bin/pytest` and git read commands.
- Recomputed results with two short Python scripts that read only the saved parquet/CSV files under `outputs/` and
  `data/processed*/`. The formulas were written out independently and not imported from `jkmp`; the DSR recomputation
  takes the trial IRs from the saved trial summaries.
- Line numbers refer to the working tree as found: branch `rebaseline-1952`, HEAD `30bada7`, with uncommitted changes.

---

## 1. Verdict

1. **The framework is substantially implemented in its superseded form.** Every iteration, the active-book machinery,
   the breadth and capacity analysis, the half-life search, the holdout guard, the deflated Sharpe ratio and the monitor
   exist in code, have tests (156 pass), and produced saved outputs.
2. **The reported numbers can be trusted as computed.** Every quantity I recomputed from the saved monthly files matches
   the reported value: Table 2 to ≤ 7e-17, and the IRs, TEs, betas, keep probabilities, capacity, rank IC and DSR to the
   printed precision. No calculation error was found.
3. **The governance around those numbers is weaker than the framework requires:**
   - the go/no-go thresholds were never pre-registered;
   - the robustness battery was not run;
   - none of the iteration or holdout code is committed or tagged;
   - the "untouched" 2021–2025 holdout is not clean, because the hypotheses of all three iterations were motivated by
     the pre-rebaseline 2020–2025 results.
4. **On the framework's own go/no-go thresholds, the final version (Trial 3) does not pass.** On the holdout it fails
   the 80% probability bar (77.8%), and the deflated Sharpe ratio is 0.90 against a 0.95 bar. Neither failure appears in
   `docs/results_summary.md` as a go/no-go decision.

---

## 2. Traceability matrix

| ID | Requirement (short) | Verdict | Evidence |
|---|---|---|---|
| A1 | (S1) Timeline, holdout keys, 1952–2025 data, \$10B anchored 2020-12-31 rolled back and forward; start-weight sensitivity | **Met (superseded form)** | `configs/base.yaml:36-46` (warmup 1952-12, validation 1971–1980, test 1981–2020, holdout 2021–2025, `data_end`, `holdout_locked: true`, `evaluate: test`); `base.yaml:116` anchor; `src/jkmp/costs.py` `wealth_path` (forward and back). Recomputed: `wealth.parquet` 1952-11 → 2025-12, exactly 1.0e10 at 2020-12-31, 817 months rolled back and 60 forward; panel 1952-12 → 2025-11. Start sensitivity (utility 1981–2020 → 1982–2020): Portfolio-ML (linear) 0.0257 → 0.0272; Portfolio-ML 0.0187 → 0.0202; Static-ML\* 0.0176 → 0.0190; Static-ML −0.0112 → −0.0083. **The ranking does not change.** The first 12 months are negative for every method (−0.03 to −0.12), but the shift is uniform. |
| A2 | Grids p ∋ 8, 16, 32; u ∋ 0, 0.1; edge shares reported; flag > 25% on an edge | **Partial** | `base.yaml:156` p_grid {8..512} and `base.yaml:169` u_grid {0, 0.1, ..}: Met. `outputs/tables/grid_edges.csv` exists (n = 50 picks, 1971–2020). The targeted edges are fixed: Portfolio-ML p 8% min / 2% max; Static-ML\* u 12% / 4%. **Flags > 25%:** Portfolio-ML g 28% / 72%, and return-model g 20% / 80% (2-point grids, so 100% of picks are on an edge by construction); Static-ML\* k 42% at min (k = 0.2); Static-ML\* v = 1 + g 22% / 48% (70% on edges); return-model p 66% at max (512). Not addressed. |
| A3 | Feature importance over 20 seeds with s.e.; Fig. 5 error bars | **Met** | `outputs/feature_importance/importance.csv` has columns `n_seeds` = 20 and `fi_se`; `src/jkmp/evaluation/feature_importance.py:167` `xerr=1.96*fi_se`; `outputs/figures/fig5_feature_importance.{png,pdf}` exist; D31 (`docs/deviations.md:34`). |
| A4 | Lazy `import wrds`; tests without WRDS; CLAUDE.md current; no WRDS CSV tracked, source reference instead | **Partial** | Lazy imports: `src/jkmp/data/wrds_pull.py:82` (`import wrds` in `connect`) and `:96` (`import sqlalchemy` in `_set_timeout`); the module-level import sits under `TYPE_CHECKING` (`:30`); no other `wrds`/`sqlalchemy` import in `src/`. Tests: 156 pass, but `wrds` is installed in `.venv`, so "without WRDS installed" is not directly verified. `git ls-files` has no WRDS CSV (only `docs/paper_tables/*.csv` and `report/figures/*.csv`). **But:** CLAUDE.md still describes the old design at `CLAUDE.md:17` (Σ 2003–2025 from 1993 data), `:22` (predictions 2015–2025), `:25` (validation from 2015), `:171`, `:174`, `:184` (1993 data), `:185` (\$10B at end-2025), and `:214` (hp years 2015–2025). `CLAUDE.md:173` still lists `jkp_usa_top500.csv` as a "Deliverable CSV", and the file exists locally (`data/raw/jkp_usa_top500.csv`, 191 MB, git-ignored). The framework asked for it to be replaced with a source reference. |
| B1 | Every trial pre-registered (hypothesis, change, metric, keep, kill, date) before its result; timing evidence | **Partial** | `docs/research_log.md:13`, `:38`, `:104` and the Trial 4 pre-registration each sit above their result blocks and carry a date (2026-10-30). Trials 2 and 3 name all six fields. Trials 3 and 4 give a "Motivation" with no labelled "Hypothesis"; it is implied. **Order in time is not provable:** the log is not in git, and its mtime (08:58) post-dates every output. File mtimes are consistent with the stated sequence: `active_pfml.yaml` 02:34 < `active/calibration.csv` 02:41 < runs 02:42; `universe_250.yaml` 02:47 < u250 calibration 03:08 < run 03:10; `halflife.yaml` 03:30 < run 03:33. |
| B2 | One change at a time; list changed keys; flag unregistered changes | **Met** | Trial 2 (`active_pfml.yaml:5-7`): `portfolio_ml.add_constant` false, `portfolio_ml.benchmark_relative` true, plus `investor.gamma_te` via `--override` (pre-registered calibration, log `:47-50`). Trial 3 (`universe_250.yaml:4-11`): `paths.processed`, `paths.outputs`, `universe.target_n/entry_rank/exit_rank`, plus `gamma_te` = 49.09 via override. The γ_TE re-calibration **is pre-registered** (`research_log.md:112`). Trial 4 (`halflife.yaml:3-9`): `paths.outputs`, `gamma_te` 49.09 (unchanged from Trial 3), `portfolio_ml.half_life_years`. No unregistered key. Two notes. (a) Trial 3's 49.09 lives only in the command line: `universe_250.yaml` inherits `gamma_te: 100` from base. (b) Trial 3 changes the realised TE level as well as breadth (ex-ante 2.36% vs 3.63%); checked separately, it still beats top-80 at the 1% budget (IR 0.646 vs 0.464, P = 0.946). |
| B3 | (S1) Holdout guard on every evaluation path; list leaks; tests | **Partial** | Guard: `src/jkmp/evaluation/holdout.py:31-50` and `evaluation_window` `:53-61`. Called in `metrics.table2:52`, `prob_outperformance:70`, `return_correlations:83` and `active_returns:101`, so also `active_stats`, scripts 12–15 and the capacity curves. Also called in `grids.py:54,60`, `frontier.py:36`, `feature_importance.py:139`, `regimes.py:97,106,129,152`, `monitor.py:52`, `covariance.vol_bias:354`, and `return_model.py:169,186,196`; `prediction_years` is capped at `:94`. Tests: `tests/test_holdout.py` (5 tests) pass. **Unguarded helpers that can produce holdout-month returns while locked:** `portfolio.accounting.portfolio_returns`, `evaluation.cost_sensitivity.recost`, and `metrics.summary_stats` / `utility_flow` (public, no `dates`). Every script routes them through a guarded function before printing, but nothing prevents a direct call. The panel holds `ret_ld1` through 2025-11. |
| B4 | Keep rule P = Φ(mean d / (sd/√T)) ≥ 80%; formula, ddof, T; recompute Trials 2–4 | **Met** | `metrics.prob_greater` (`:115-118`) and `prob_outperformance` (`:77`) compute 1 − Φ(0; mean d, sd(d)/√T), which equals Φ(mean/(sd/√T)), with ddof = 1 and T = 480. Recomputed: Trial 2, P(beats baseline) = 1.0000 at 1%, 2% and 4% (log `:79-81`: 100.0%). Trial 3, P = 0.9841 traded VW / 0.9846 paper index (log `:139`: 98.4% / 98.5%). Trial 4, P = 0.0256 (log `:183`: 2.6%). |
| B5 | Trial count complete; DSR output exists | **Partial** | The count of 6 is kept (log `:7`), and calibration runs used ex-ante TE only (see C2). `outputs/holdout/tables/deflated_sharpe.csv` exists: DSR 0.8995, recomputed 0.8995. **Gaps.** (a) `N_TRIALS = 6` is hard-coded (`scripts/15_holdout.py:26`), not read from the log. (b) The trial-SR set mixes benchmark definitions (Trials 2–3 vs traded VW, Trial 4 vs paper index) and includes the baseline's total-return strategy scored as an active book (IR −0.27). That one point dominates V[SR]: without it, SR₀ falls from 0.44 to 0.15 a year and the DSR rises from 0.90 to 0.999, which flips the go/no-go criterion. (c) The pre-rebaseline 1993-design runs that motivated the iterations are not counted. |
| B6 | Robustness battery per kept change: 20 RFF seeds, sub-periods, γ ∈ {5, 10, 20}, wealth \$1B–\$250B | **Not met** | RFF seeds: not run for Trial 2 or 3 (the 20 seeds in D31 are permutation seeds for feature importance). Sub-periods: Trial 2 none; Trial 3 only capacity over 1990–2020 and IR by decade, the latter produced inside Trial 4 (`outputs/u250_hl/halflife/by_decade.csv`). γ: Trial 2 varies the TE budget (γ_TE 31.19 / 12.82 / 4.331), not γ ∈ {5, 10, 20}; Trial 3 none. Wealth: Trial 3 (top-250) and the top-80 2% reference, \$1B–\$250B (`outputs/u250/breadth/capacity_curve.csv`). **1 of 4 checks run, for Trial 3 only.** |
| C1 | `add_constant` flag in `scale_signals`; default reproduces baseline | **Met** | `src/jkmp/portfolio/portfolio_ml.py:94-103`, with default `True` (`base.yaml:160`) and `config.py:239`. Test `tests/test_portfolio_ml.py:397`. The log's claim that the current code reproduces the stored baseline exactly (`research_log.md`, holdout entry) was not re-run (rule 2). |
| C2 | `gamma_te`, grid and targets in config; calibration on 1972–1980 ex-ante TE only | **Met** | `base.yaml:119-121`; `config.py:179-190`. `src/jkmp/portfolio/active.py:53-64`: hp years ≤ `validation_end`, aim dates with return ≤ 1980-12, output = mean ex-ante √(aᵀΣa) only. Realised returns enter only through the walk-forward selection on ≤ 1980 data, as in any run; no test-window return is read. `outputs/active/calibration.csv`: 108 months from 1972-01. |
| C3 | a_t = m·a_inh + (I − m)A_t, π = π^VW + a; Δπ = Δa; benchmark turnover/cost reported | **Met** | `portfolio_ml.py:573-577` returns `vw + m@(held − vw) + aim − m@aim`. Test `tests/test_portfolio_ml.py:407-440` asserts Δπ = Δa to 1e-15. Benchmark recomputed: turnover 1.94% a month, TC 0.097% a year. **The log misstates it as "TC 0.001% a year"** (`research_log.md:84`; the value is 0.00097 as a fraction). |
| C4 | Active return, ex-ante and realised TE, gross and net IR, β, Xᵀa, negative-weight share; recompute | **Met** | `metrics.active_stats:149-167`, `ex_ante_exposures:130-146`. Recomputed vs `outputs/active/summary.csv`: net IR 0.5038 / 0.4153 / 0.3397, realised TE 2.465% / 4.635% / 8.032%, β 0.9655 / 0.9273 / 0.8513 for 1% / 2% / 4%. All match to 4 dp, and match log `:79-81`. |
| C5 | Long-only limitation stated | **Not met** | `grep -i "long-only\|long only"` over `src`, `scripts`, `docs` and `CLAUDE.md` finds it only in the framework and the validation prompt. The negative-weight share (11–28% for Trial 2, 22% for Trial 3) is reported but not framed as a limitation of a long-only mandate. |
| C6 | Kill criterion evaluated as pre-registered | **Met** | Kill rule at `research_log.md:61-62`; decision at `:87-88`. Recomputed P(net IR > 0) = 0.9993 / 0.9954 / 0.9831, all ≥ 0.80, so not killed. |
| D1 | `universe_250.yaml` target 250, scaled buffer; own VW benchmark; factors from top 500 | **Met** | `universe_250.yaml:8-11` (250 / 220 / 300). Benchmark is the Market row of `outputs/u250/benchmarks/pf_test.parquet`. `covariance.factor_universe_top_n: 500` is unchanged (`base.yaml:84`), and Σ was rebuilt in `data/processed_u250/covariance/`. Minor: recomputed 238–267 members a month vs "250–268" in D35 (`deviations.md:38`). |
| D2 | Run time measured at N = 250 first; N = 500 decision documented | **Met** | `research_log.md:128-130` (0.9 s per year of inputs, 1.5 min per calibration point, N = 500 about 13 min, not run as an un-pre-registered trial), and `:152`. |
| D3 | Rank IC by size bucket by decade; recompute | **Met** | Recomputed from `data/processed_u250/{predictions,panel}.parquet` with scipy Spearman: top-80 0.0311 overall (1970s 0.0629, 2010s 0.0100, 2020 −0.0994); ranks 81–250 0.0431 (1970s 0.0773, 2010s 0.0116, 2020 −0.0634). This matches `outputs/u250/breadth/rank_ic_by_size.csv` and log `:140-141`. |
| D4 | Capacity curve; AUM where net IR < 0.3 (interpolation) | **Met** | `cost_sensitivity.capacity_aum:43-55` interpolates linearly in log wealth. Recomputed top-250 paper-index net IR \$1B 0.773, \$10B 0.646, \$50B 0.456, \$100B 0.328, \$250B −0.039, giving a capacity of **\$107.31B** (reported 107.3B). |
| D5 | Cost sensitivity evaluation-only; VIX Λ 1990–2020; stock-vol Λ with σ̄ ≤ t; capacity under all three | **Met** | `cost_sensitivity.py:32-40` re-costs fixed weights, with no refit. `:23-29`: σ̄ is a rolling mean over each stock's Σ month-ends ≤ t, sorted by eom, so there is no look-ahead (test `tests/test_cost_sensitivity.py:39`). Note `min_periods=1`, so early months have s ≈ 1. Capacity is reported for base, stock-vol and VIX (VIX 1990–2020 only) in `outputs/u250/breadth/capacity_aum.csv`. |
| D6 | (S2) Benchmark change documented; Trial 3 decision under both | **Met** | Documented at `research_log.md:146-151` and D35 (`deviations.md:38`). Recomputed P = 0.9841 (traded VW) and 0.9846 (paper index): KEEP under both. |
| E1 | Discounted sums in `coefs_from_sums` and `december_utility`; δ = 0.5^(1/h); discounted counts | **Deviates — documented** | `portfolio_ml.py:331`: weight 0.5^(−(s−s₀)/h) = δ^(−(s−s₀)), applied to r_sum, d_sum and n (`:335`). The common factor cancels, so normalisation uses the discounted counts. `december_utility` (`:402`) is deliberately **not** discounted, so that every h is scored on one yardstick. Documented in D36 (`deviations.md:39`) and the Trial 4 pre-registration. |
| E2 | `half_life_years` {3, 5, 10, ∞}; h chosen each December with (g, p, λ) by the same score | **Met** | `halflife.yaml:9`; `search_yearly:488-501` (an `h` column per β set); `select_hypers:504-513` takes the arg-max of `cum_obj` across (g, p, λ, h). |
| E3 | Test that h = ∞ reproduces β exactly | **Met** | `tests/test_portfolio_ml.py:443` (`assert_array_equal`, bit for bit). Ran it: passes. |
| E4 | Chosen h saved; kill applied (35/40, P = 2.6%) | **Met** | `outputs/u250_hl/active/gte49.09/chosen.csv` and `outputs/u250_hl/halflife/chosen_h.csv`. Recomputed: h = ∞ in 35/40 test years, h = 10 in 5, P = 0.0256. Kill rule in code at `scripts/14_halflife.py:78-79`; decision at `research_log.md:183`. |
| F1 | Go/no-go thresholds recorded before the baseline | **Not met** | No go/no-go section or thresholds in `docs/research_log.md`, `docs/results_summary.md` or `CLAUDE.md`. Searches for "go/no-go" and "go-no-go" over `docs/` hit only the framework and the validation prompt. |
| F2 | Evaluate go/no-go for Trial 3 on the holdout and 1981–2020 | **Partial (computed here; 2 FAIL on holdout)** | See section 4. |
| F3 | Monitor: six checks with the framework's flag rules; limits defined; flags sensible; dashboard | **Partial** | `src/jkmp/evaluation/monitor.py:47-79` implements 8 checks. Limits are defined in `base.yaml:182-191` (β [0.8, 1.2], \|Xᵀa\| < 1.0, IR window 36, IC 12/6, TC 3, bias 36). The IR, drawdown, β, theme, IC and bias rules match the framework wording (`:70-77`); the bias band is 1 ± √(2/36). Dashboard: `outputs/holdout_final/monitor/dashboard.{png,html}`. **Problems:** (a) "Realised" trading cost is a model re-cost with the stock-vol Λ (`scripts/11_monitor.py:50`), and "predicted" is the run's own base-Λ cost (`monitor.py:64`). The check tests whether σ_i,t > σ̄_i,t, not realised vs predicted cost. Recomputed: tc stat > 0 in 48.5% of months, `flag_tc` 192/480 (1981–2020) and 27/60 (2021–2025), first firing 1981-03. Uninformative as specified. (b) The half-life check is vacuous for the final model, which has no h search: the statistic is 0 in every month. (c) The IC flag fires 51/480 and 15/60 times; the bias flag fires 125/480 in 1981–2020 (consistent with realised TE 1.32× ex-ante) and 0/60 in the holdout. Last month 2025-12: rolling IR −0.213 FLAG; β 0.986, max theme 0.20 and bias 1.01 within limits. |
| G1 | Scripts for data, model, validation/backtest, monitoring | **Met** | `scripts/00_download.py`–`02` (data), `03`–`07` (models), `08`–`10`, `11_tables`, `12`–`15` (validation/backtest), `11_monitor.py` (monitoring). Two scripts share the number 11. |
| G2 | Datasets: CSV or link plus source reference; nothing WRDS-licensed redistributed | **Partial** | Source references for tables and URLs: `base.yaml:20-27`. Nothing WRDS-licensed is tracked. However, `data/raw/jkp_usa_top500.csv` (JKP/WRDS rows, 191 MB) is generated by `00_download.py`, and `CLAUDE.md:173` calls it a "Deliverable CSV". No public-data CSV or link deliverable is defined. |
| G3 | Report and slides present; report reflects the current design | **Not met / Pending (S3)** | `report/QF600_Replication_Report.pdf` (untracked, mtime 2026-10-08, 23 pages): text has "2015" ×11, "2020–2025" ×9 and "1993" ×4; no "Executive summary". It describes the pre-rebaseline design. No slide file was found within three directory levels. |
| H1 | pytest | **Met** | `.venv/bin/pytest`: **156 passed** in 28.6 s. |
| H2 | Uncommitted files results depend on; tags reproduce results | **Not met** | `git status`: 17 modified tracked files, including `src/jkmp/portfolio/portfolio_ml.py`, `evaluation/metrics.py`, `evaluation/holdout.py`, `config.py`, `scripts/04-07`, `configs/base.yaml` and `docs/research_log.md`. 25 untracked files, including all iteration and holdout configs, `scripts/11_monitor.py` and `12`–`15`, `evaluation/{active,cost_sensitivity,monitor}.py`, `portfolio/active.py` and their tests. Only tag: `baseline-v1` → `30bada7`. `git diff --stat baseline-v1 -- src scripts`: 11 files, +267/−47. No per-trial or final tags exist, so Trials 2–4 and the holdout cannot be reproduced from version control. |
| H3 | Recompute Table 2 (5 methods) for 1981–2020 vs `table2_vs_paper.csv`, tolerance 1e-6 | **Met** | Max \|diff\| over R, Vol, TC, SR_net and Utility: Market 2.8e-17, Factor-ML 5.6e-17, Static-ML\* 2.8e-17, Portfolio-ML 6.6e-17, Portfolio-ML (linear) 4.6e-17 (n = 480 each). |

---

## 3. Recomputed numbers vs reported

| Quantity | Reported (where) | Recomputed | Match? |
|---|---|---|---|
| Table 2 Utility: Market / Factor-ML / Static-ML\* / PF-ML / PF-ML linear, 1981–2020 | −0.0351 / −2.6410 / 0.0176 / 0.0187 / 0.0257 (`outputs/tables/table2_vs_paper.csv`) | −0.035071 / −2.641023 / 0.017608 / 0.018696 / 0.025671 | Yes (≤ 7e-17) |
| Table 2 SR_net: PF-ML / Static-ML\* | 0.6116 / 0.6322 (same file) | 0.611634 / 0.632153 | Yes |
| Utility 1982–2020 (start sensitivity), PF-ML / Static-ML\* / Static-ML | not reported | 0.0202 / 0.0190 / −0.0083 (ranking unchanged) | n/a |
| Trial 2 net IR, 1% / 2% / 4% | 0.5038 / 0.4153 / 0.3397 (`outputs/active/summary.csv`; log 0.50 / 0.42 / 0.34) | 0.5038 / 0.4153 / 0.3397 | Yes |
| Trial 2 realised TE | 2.465% / 4.635% / 8.032% (summary; log 2.47 / 4.63 / 8.03) | 2.465% / 4.635% / 8.032% | Yes |
| Trial 2 β to VW | 0.9655 / 0.9273 / 0.8513 | 0.9655 / 0.9273 / 0.8513 | Yes |
| Trial 2 P(net IR > 0) | 0.9993 / 0.9954 / 0.9831 | same | Yes |
| Trial 2 P(beats baseline) | 1.000 ×3 (log `:79-81`) | 1.0000 ×3 | Yes |
| VW top-80 benchmark TC a year | "0.001% a year" (`research_log.md:84`) | 0.097% a year (0.00097) | **No: unit error in the log** (the CSV value is correct) |
| Trial 3 net IR top-250 / top-80, traded VW | 0.666 / 0.415 (`outputs/u250/breadth/summary.csv`) | 0.6659 / 0.4153 | Yes |
| Trial 3 P(keep), traded VW / paper index | 98.4% / 98.5% (log `:139`) | 0.9841 / 0.9846 | Yes |
| Trial 3 vs top-80 at 1% budget (TE-level confound check) | not reported | IR 0.646 vs 0.464, P = 0.946 | n/a (decision robust) |
| Rank IC top-80 / 81–250, 1971–2020 | 0.031 / 0.043 (log `:140`) | 0.0311 / 0.0431 | Yes |
| Capacity top-250, paper index, base, 1981–2020 | \$107B (`capacity_aum.csv`: 107.306e9) | \$107.31B | Yes |
| Trial 4 net IR (h tuned) / Trial 3 (paper index) | 0.580 / 0.646 (log) | 0.5799 / 0.6464 | Yes |
| Trial 4 P(improve); share of h = ∞ | 2.6%; 35 of 40 (88%) | 0.0256; 35/40 | Yes |
| Final, holdout net IR / realised TE / P(IR > 0) | 0.344 / 1.84% / 77.8% (`outputs/holdout/tables/final_active.csv`) | 0.3437 / 1.84% / 0.7782 (60 months, 2021-01..2025-12) | Yes |
| DSR (1981–2020, N = 6) | 0.8995, SR₀ 0.438 a year (`deflated_sharpe.csv`) | 0.8995, SR₀ 0.4379 | Yes |
| DSR excluding the baseline trial (N = 5 active variants) | not reported | 0.9988, SR₀ 0.154 | n/a (sensitivity) |
| Monitor, last month (2025-12) rolling 36m IR | −0.21 FLAG (Phase 7 report) | −0.2132, `flag_ir` True | Yes |
| Universe size, top-250 | "250–268 members" (D35) | 238–267 a month | Minor mismatch |

---

## 4. Go/no-go scorecard (F2): final version, Trial 3 (active PF-ML, top-250, γ_TE = 49.09, paper index)

The thresholds come from the framework's illustrative table. They were **never pre-registered** (F1), so this scorecard
applies them after the fact.

| Criterion | Threshold | 1981–2020 (decision window) | 2021–2025 (holdout) | Verdict |
|---|---|---|---|---|
| Net information ratio | ≥ 0.3 | 0.646: PASS | 0.344: PASS | **PASS** |
| P(beat index), Table 3 rule, P(mean net active > 0) | ≥ 80% | 99.998%: PASS | 77.8%: FAIL | **FAIL (holdout)** |
| Deflated Sharpe ratio | > 0.95 | 0.8995: FAIL (0.9988 if the baseline trial is excluded from V[SR]) | NOT COMPUTED (by design: DSR on the selection sample) | **FAIL** |
| Realised TE within ±20% of the 2% budget (1.6–2.4%) | ±20% | 3.12%: FAIL (+56%) | 1.84%: PASS | **PASS on the holdout (framework's window); FAIL on 1981–2020** |
| Capacity (AUM where net IR < 0.3) ≥ \$10B, every cost model | ≥ \$10B | base \$107B, stock-vol \$106B, VIX \$56B (1990–2020): PASS | NOT COMPUTED | **PASS (1981–2020 only)** |

**Result: 2 of 5 criteria fail**: holdout probability (77.8% vs 80%) and DSR (0.90 vs 0.95). Under the framework's
thresholds the recommendation would be **no-go, or "not proven"**. The DSR verdict depends on an unregistered choice: it
passes if the total-return baseline is left out of the trial-SR variance.

---

## 5. Gaps, ranked by severity

### Critical (a result could be wrong or misread)

**C-1. The holdout is not a clean out-of-sample test.**
- *What.*
  - The hypotheses of all three iterations come from the pre-rebaseline run, whose test window was 2020–2025 and
    overlaps the 2021–2025 holdout:
    - Trial 2's β ≈ 0.20, α −2.6% (`research_log.md:40`);
    - Trial 3's "IC 0.002 inside the top 80" (`:106`, `:141`);
    - Trial 4's "low-risk tilt reversed after 2020 in the old run" (`:161`).
  - `outputs/_archive_1993/` still holds those results.
  - In the Phase 7 session the Dec-2025 wealth level was also printed before the holdout run, which reveals the
    market's holdout return. This is not recorded anywhere in the repo.
- *Evidence.* The cited log lines. `grep -n "18.47\|slip"` over `docs/` and `CLAUDE.md` finds nothing.
- *Why it matters.* The holdout statements, "nothing here fed back into any model choice" and "post-publication decay",
  overstate how independent the 2021–2025 numbers are. Choices made with knowledge of 2020–2025 behaviour (benchmark-
  relative book, breadth) can flatter the holdout.
- *Smallest fix.* Add a "Holdout contamination" paragraph to `docs/research_log.md` (holdout entry) and to
  `docs/results_summary.md`. It should name the three motivating facts from the old 2020–2025 run and the Dec-2025 wealth
  disclosure, and call the holdout "partially informed".

### Major (requirement unmet)

**M-1. Go/no-go was never pre-registered or decided (F1, F2).**
- No thresholds were logged before the baseline, and the final report states no go/no-go decision.
- On the framework's thresholds the final version fails two of five (section 4).
- *Fix.* Record the thresholds in the log, marked "post hoc, after the holdout". Add the section 4 scorecard and an
  explicit recommendation to `results_summary.md`.

**M-2. Results are not reproducible from version control (H2).**
- All iteration, holdout and monitor code and configs are uncommitted, as is the research log.
- `src/jkmp/portfolio/portfolio_ml.py` and `metrics.py` differ from `baseline-v1`.
- No `it1`–`it3` or `final` tags exist.
- *Fix.* Commit the tree as it stands, which is the code the log says reproduces every stored run. Tag it `final`, and
  record `git rev-parse HEAD` in the research log. Per-trial tags can point at that commit, with the config and override
  for each trial.

**M-3. The robustness battery was not run (B6).**
- None of the four checks (RFF seeds, sub-periods, γ ∈ {5, 10, 20}, wealth) was run for Trial 2.
- For Trial 3 only the wealth check was run.
- *Fix.* At minimum, run Trial 3 at 5 RFF seeds and report IR by sub-period. This is a new evaluation of a frozen model,
  so it should be pre-registered as such.

**M-4. The DSR depends on an unregistered choice of trial set (B5).**
- `N_TRIALS = 6` is hard-coded (`scripts/15_holdout.py:26`).
- The trial SRs mix two benchmark definitions and include the total-return baseline (IR −0.27). Excluding it moves the
  DSR from 0.90 to 0.999 and flips the go/no-go criterion.
- *Fix.* Document the trial-set choice in D37, report both DSR values, and read N from one constant shared with the log.

**M-5. The monitor's trading-cost check does not compare realised with predicted cost (F3).**
- Both sides are model costs: vol-scaled Λ against base Λ (`scripts/11_monitor.py:50`, `monitor.py:64`).
- The flag fires in 40–45% of months.
- *Fix.* Rename it "cost-stress check" in the docstring, D37 and the dashboard, and state that realised execution costs
  are unavailable. Optionally require a materiality threshold (e.g. > 10% above prediction).

**M-6. The long-only limitation is not stated (C5).**
- Negative-weight shares of 11–28% are reported, but nowhere is it said that a long-only mandate would need a
  constraint that breaks the closed form.
- *Fix.* One sentence in D34 and in `results_summary.md`.

**M-7. The WRDS-derived CSV is still labelled a deliverable (A4, G2).**
- `CLAUDE.md:173` calls `data/raw/jkp_usa_top500.csv` a "Deliverable CSV" (191 MB of JKP/WRDS rows).
- The framework required replacing it with a source reference.
- *Fix.* Change the CLAUDE.md row to a source reference (WRDS `contrib.global_factor`, the filters, and the pull command).
  Make sure the file is excluded from any submission zip.

**M-8. The report does not reflect the current design (G3; pending per S3), and no slides were found.**
- *Fix.* Out of scope for code validation. Record it as pending in `results_summary.md`.

### Minor (documentation)

- **m-1. CLAUDE.md is stale** at `:17, :22, :25, :171, :174, :184, :185, :214` (1993 data, 2015–2025 years, \$10B at
  end-2025). *Fix:* update those rows to the 1952–2025 / 1971–2020 design.
- **m-2. `research_log.md:84` unit error.** Benchmark TC is 0.1% a year, not 0.001%. *Fix:* edit the number.
- **m-3. D35 says "250–268 members".** Recomputed 238–267. *Fix:* correct the range.
- **m-4. Trial 3's γ_TE = 49.09 is only a command-line override.** `universe_250.yaml` inherits 100. *Fix:* pin
  `investor.gamma_te: 49.09` in a note or in the run command recorded in the log; `halflife.yaml` already pins it.
- **m-5. Grid-edge shares above 25% are not discussed.** Affected: Static-ML\* k 42% at 0.2, v 70% on edges, return-model
  p 66% at 512, and 2-point g grids. *Fix:* a sentence in D33, or widen the grids in a future trial.
- **m-6. The holdout guard is not on the return generators** (`accounting.portfolio_returns`, `cost_sensitivity.recost`,
  `metrics.summary_stats`). *Fix:* add a `dates` argument and call the guard there, or document that scripts must route
  through guarded functions.
- **m-7. The monitor's half-life check is vacuous** for the final model, which has no h search. The IC series starts in
  1981, so the rolling 12-month IC is NaN until 1981-12. *Fix:* note both in D37.
- **m-8. Pre-registrations for Trials 3 and 4 lack a labelled "Hypothesis".** The log is not version-controlled, so its
  timing cannot be proven. *Fix:* label the hypotheses, and commit the log after each pre-registration in future.
- **m-9. Two scripts are numbered 11** (`11_tables.py`, `11_monitor.py`). *Fix:* renumber one.

---

## 6. What could not be verified, and why

1. **Pre-registration timing (B1).**
   - The research log is untracked and was last modified after every output, so it carries no history.
   - All entries are dated only "2026-10-30", without times.
   - File mtimes are consistent with the stated order but cannot prove it.
2. **That the current code reproduces the stored runs (C1, H2).**
   - Rule 2 forbids re-running scripts 00–15. The log's "max |Δr| = 0" claim is unverified.
   - The baseline outputs pre-date code changes since `baseline-v1`: +267/−47 lines in `src` and `scripts`.
3. **That tests pass without WRDS installed (A4).** `wrds` is installed in `.venv`. Imports are lazy by inspection, but I
   did not uninstall the package to test this.
4. **The holdout runs themselves (F2, B3).**
   - I read the saved holdout outputs; I did not regenerate them.
   - Whether any un-saved holdout number was viewed before the run cannot be established from files, apart from the
     Dec-2025 wealth disclosure noted in C-1, which comes from the session record rather than the repo.
5. **Capacity on the holdout, and the DSR on the holdout.** Neither was computed by the team. Computing either would
   require a new run, which this validation's rules exclude.
6. **The paper-side values in `table2_vs_paper.csv`.** I checked "ours" against the pf files. I did not re-check the
   "paper" column against the PDF of the paper.
7. **Covariance and return-model internals** (Barra Σ, RFF ridge). These were out of scope for this framework
   validation. They are covered only by the existing tests.
