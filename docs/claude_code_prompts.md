# Claude Code Prompt Framework — Rebaseline to the Paper's Timeline, Then Three Iterations

Run these prompts in order from the project root (`QF600_Replication/`), one phase per Claude Code session.
Each phase ends at a **STOP gate**: Claude Code reports, and you review and commit before pasting the next prompt.
Work on a branch (`git switch -c rebaseline-1952`) and commit after each approved phase.

| Phase | Prompt | Produces |
|---|---|---|
| 0 | Orientation (read-only) | Change plan for every date-dependent line, compute estimate |
| 1 | Config, holdout guard, data re-pull from 1952 | New config, deviations updated, raw data 1952–2025 |
| 2 | Panel, universe, covariance from 1952 | Processed data, Nasdaq-1972 check, diagnostics |
| 3 | Baseline: return model, benchmarks, Static-ML\*, Portfolio-ML over 1981–2020 | Table 2/3 vs paper, baseline fixes |
| 4 | Iteration 1: benchmark-relative Portfolio-ML | Active-return results, keep/drop decision |
| 5 | Iteration 2: breadth vs capacity (top 250) + cost sensitivity | Capacity curve, keep/drop decision |
| 6 | Iteration 3: time-decayed estimation | Half-life results, keep/drop decision |
| 7 | Holdout 2021–2025 (run once), monitoring script, report updates | Final tables, `scripts/11_monitor.py` |

---

## Standing instructions (paste at the top of every phase)

```text
STANDING INSTRUCTIONS — apply to every task in this session.

Project: Python replication of Jensen, Kelly, Malamud & Pedersen (2026), "Machine Learning and the
Implementable Efficient Frontier" (RFS 39: 3035–3078). Read CLAUDE.md first and follow its Rules and
Conventions exactly. In particular:
- Logic lives in src/jkmp/; scripts only wire config -> library -> files.
- Every parameter comes from configs/base.yaml via jkmp.config.load_config(); new parameters go into the
  YAML AND the matching dataclass in src/jkmp/config.py. Experiments are configs/experiments/*.yaml
  overrides that list only the keys they change.
- Never use np.linalg.inv. Where the paper is ambiguous, follow reference/jkmp_r/ and cite the R file and
  function. Never edit reference/jkmp_r/.
- Every deviation from the paper or the R code goes in docs/deviations.md in the same change.
- Timing: never use information dated after eom to form a portfolio at eom.

Study design (the baseline matches the paper's timeline):
- Data from 1952-12; estimation only 1952–1970; validation head start 1971–1980;
  out-of-sample test 1981-01 .. 2020-12 (480 return months); wealth $10B at 2020-12-31.
- HOLDOUT 2021-01 .. 2025-12 is LOCKED. Do not compute, print, plot or inspect any performance number
  with a return month after 2020-12 unless the prompt explicitly says the holdout is unlocked.
  Model fitting may use data up to the end of each walk-forward year as usual.
- Walk-forward exactly as the paper (Sec. 5.3): beta and the return model re-estimated every year on an
  expanding window; hyperparameters picked each December by cumulative validation utility since 1971.

Research log: before running any iteration, append its pre-registration to docs/research_log.md
(hypothesis, change, metric, keep threshold, kill criterion, date). After running, append the result and
the decision. Every variant run counts as a trial; keep a running trial count in the log.

Working style:
- Make the smallest change that does the job. Run .venv/bin/pytest after every code change; add tests for
  new logic (synthetic data, no WRDS needed).
- Long jobs: time one year or one decade first, report the estimate, then run the full job.
- At the STOP gate, report: what changed (files), test results, the key numbers, anything surprising,
  and open questions. Then stop and wait. Do not start the next phase.
```

---

## Phase 0 — Orientation (read-only)

```text
[Paste STANDING INSTRUCTIONS]

TASK (read-only — make no edits):
1. Read CLAUDE.md, configs/base.yaml, src/jkmp/config.py, docs/deviations.md (D1, D4, D8, D9, D23,
   D26, D27, D29, D30), and scripts 00–10.
2. List every place where the current dates (1993 warm-up, 2015 validation start, 2020–2025 test,
   2025-12-31 wealth anchor) are used, directly or through the config. Known spots include:
   - dates.test_end doubles as the end of the data pull (data/wrds_pull.py, data/prepare.py) — the holdout
     needs data to 2025-12 while the test ends 2020-12, so the data end must become its own key;
   - eval_dates start at dates.test_start in scripts/05, 07 and portfolio/grids.py;
   - hp years run from return_model.pred_years[0] to dates.test_end.year (scripts/07);
   - the wealth path is rolled BACK from wealth_end_date (costs.wealth_path) — with the anchor at
     2020-12-31 and data to 2025-12, months after the anchor need wealth rolled FORWARD;
   - hard-coded labels such as "2020–2025" in evaluation/regimes.py and script docstrings.
3. Estimate the new sizes and run times: JKP rows 1952–2025, CRSP daily rows for every stock ever in the
   top 500, covariance regressions over ~17,000 trading days, Portfolio-ML inputs (per g) for ~3x the
   current months. Use the current logs in outputs/logs/ to scale.
4. Propose the config keys you will add (suggested: dates.data_end, dates.holdout_start,
   dates.holdout_end, dates.holdout_locked) and how the holdout guard will work (one function in
   src/jkmp that every evaluation path calls, raising if a return month > test_end appears while locked).

STOP: report the list, the estimates and the proposal. Do not edit anything.
```

---

## Phase 1 — Config, holdout guard and data re-pull

```text
[Paste STANDING INSTRUCTIONS]

TASK:
1. configs/base.yaml (and the dataclasses in config.py):
   dates.warmup_start: 1952-12-31
   dates.validation_start: 1971-01-31,  dates.validation_end: 1980-12-31
   dates.test_start: 1981-01-31,        dates.test_end: 2020-12-31
   dates.data_end: 2025-12-31 (new)     dates.holdout_start: 2021-01-31, dates.holdout_end: 2025-12-31 (new)
   dates.holdout_locked: true (new)
   return_model.val_years: 10           return_model.pred_years: [1971, 2025]
   investor.wealth_end_date: 2020-12-31 (wealth $10B, as the paper)
   Remove or re-label train_start/train_end so nothing implies a fixed split (the design is walk-forward).
2. Point the data pull and the panel's date screen at dates.data_end instead of dates.test_end.
3. Make costs.wealth_path roll forward after the anchor (w_{t+1} = w_t (1 + R_m,t+1)) as well as back,
   with a test.
4. Add the holdout guard (one function, called by metrics/table2, the grids, feature importance, regimes
   and any plotting path) and a test that it raises while locked.
5. Update docs/deviations.md: D4 (sample period now matches the paper; holdout added), D9 (wealth anchor),
   D23 (10 validation years, as R), D27/D29 (hp schedule from 1971, selection from 1981, as R). Mark D1 as
   "under review" — its justification (sample starts after Nasdaq entered CRSP in 1972) no longer holds.
6. Run `.venv/bin/python scripts/00_download.py pull --force` (WRDS credentials from the environment).
   Report row counts by decade from data/raw/row_counts_by_month.csv.

STOP gate.
```

---

## Phase 2 — Panel, universe and covariance from 1952

```text
[Paste STANDING INSTRUCTIONS]

TASK:
1. Run scripts/01_prepare.py. Report: screened rows by decade, universe size by year, and name turnover
   by year from data/processed/universe_stats.csv.
2. D1 check: compare universe turnover in 1971–1975 with the surrounding decade. If 1972–73 shows a
   jump caused by Nasdaq stocks entering CRSP, propose (do not yet implement) either NYSE-only before
   1974 or NYSE-only throughout as the paper does, and quantify how many top-80 names are affected.
3. Run scripts/02_covariance.py on one decade first (time it), then in full. The 10-year window means the
   first Sigma is around 1963; confirm this and that every month from the first Portfolio-ML input month
   onward has Sigma for every universe stock. Report cov_diagnostics.csv summaries by decade
   (median vol, R², imputations) and the realised/predicted volatility ratio by decade.
4. Run pytest.

STOP gate — include the D1 recommendation.
```

---

## Phase 3 — Baseline over 1981–2020, plus baseline fixes

```text
[Paste STANDING INSTRUCTIONS]

TASK A — baseline fixes (correct the setup; these are not iterations):
1. Grid edges: extend portfolio_ml.p_grid with {8, 16, 32} and static_ml.u_grid with {0, 0.1}; log as
   deviations. After the run, report how often each pick sits on a grid edge.
2. Feature importance: average over 20 seeds and report standard errors (scripts/09,
   evaluation/feature_importance.py).
3. Move `import wrds` inside the pull functions in data/wrds_pull.py so the library and tests import
   without WRDS installed.

TASK B — baseline run (test window 1981–2020; holdout stays locked):
4. Run scripts 03, 04, 05, 06, 07 (RFF and linear configs), then 08 and 09. Time one year of 06 before
   running it in full.
5. Produce Table 2 side by side with the paper's Table 2 (values in docs/QF600 Replication Plan.md or the
   paper), Table 3 (posterior probabilities, the paper's method), Table 4 (correlations), Figure 1A/1B
   and Figure 5 — all over 1981–2020.
6. Report a Figure D.1-style plot of the chosen hyperparameters over time (Portfolio-ML g, p, lambda;
   Static-ML* k, u, v).
7. Sub-period table: 1981–2000 vs 2001–2020 for Table 2's Utility, SR_net and TC.
8. Update CLAUDE.md Status and the report's results section notes (do not rewrite the report).

STOP gate — this is the BASELINE every iteration is measured against. Tag the commit `baseline-v1`.
```

---

## Phase 4 — Iteration 1: benchmark-relative Portfolio-ML

```text
[Paste STANDING INSTRUCTIONS]

CONTEXT: In the old 2020–2025 run, Portfolio-ML had beta 0.20 and alpha −2.6% a year against the value-
weighted universe, and 0.82 correlation with minimum variance: the total-return objective (gamma = 10)
lets the model choose its own market exposure, and the vol-scaled constant column is an inverse-volatility
long book. A real $10B large-cap mandate is benchmark-relative with a tracking-error budget.

PRE-REGISTER in docs/research_log.md first (hypothesis, metric = net information ratio vs the value-
weighted universe, keep if the posterior probability that mean monthly active utility beats the
baseline's is >= 80% over 1981–2020, kill if net IR is not positive with >= 80% probability at any
tracking-error budget).

CHANGE (one experiment: configs/experiments/active_pfml.yaml):
  pi_t = pi_t^VW + a_t,  a_t = Omega_t beta
  max_beta E[ a_t' r_{t+1} − (w_t/2) Δa_t' Λ_t Δa_t − (gamma_TE/2) a_t' Σ_t a_t ]
1. portfolio_ml.scale_signals: flag portfolio_ml.add_constant (default true = current behaviour).
2. New investor.gamma_te; calibrate it so ex-ante tracking error is 1%, 2% and 4% a year (three runs).
   month_moments needs no formula change — r̃ = Ω'r is the active return.
3. pfml_weights: run the recursion on active weights, a_t = m g a_{t−1} + (I − m) A_t, then add the
   value weights from portfolio/benchmarks.py. Justify in a comment and a deviation entry why Δπ ≈ Δa
   (value weights drift with g; report the benchmark's own turnover and cost to show it).
4. evaluation/metrics.py: active return, ex-ante and realised tracking error, gross and net information
   ratio, beta to the benchmark, share of stocks with negative total weight.
5. Report active exposure to each theme (from the covariance loadings X) over time.

STOP gate — give the keep/drop decision per the pre-registration and update the research log.
```

---

## Phase 5 — Iteration 2: breadth vs capacity (+ cost sensitivity)

```text
[Paste STANDING INSTRUCTIONS]

CONTEXT: In the old run the return forecasts had rank IC 0.002 inside the top 80 but 0.07 across all
screened stocks. Fundamental law: IR ≈ TC × IC × √BR. Build on the version kept after Phase 4.

PRE-REGISTER first (metric = net IR; keep if posterior probability of improvement over the Phase-4
version >= 80% over 1981–2020; kill otherwise — the finding is then that the 18 characteristics, not the
universe, are the bottleneck).

CHANGE (configs/experiments/universe_250.yaml):
1. universe.target_n: 250 with entry_rank / exit_rank scaled (e.g. 220 / 300); benchmark = this
   universe's own value-weighted portfolio. Covariance factors are already estimated on the top 500.
2. Time one year of scripts/06 at N = 250 first (m is N x N per month, 13 signal months). Report the
   estimate before the full run. Only consider N = 500 if the full 250 run takes under ~8 hours.
3. Rank IC by size bucket (top 80, 81–250) per decade.
4. Capacity curve: net IR against wealth from $1B to $250B (scripts/08 grid) at the chosen TE budget.
5. COST SENSITIVITY (evaluation only — re-cost the final weights, no retraining):
   (a) VIX-scaled Lambda, reusing evaluation/regimes.py (VIX starts 1990; use 1990–2020 for this part);
   (b) stock-level Lambda_i,t × sigma_i,t / sigma_bar_i,t, with sigma_bar from past data only.
   Report the capacity estimate (AUM where net IR falls below 0.3) under the base model, (a) and (b).

STOP gate — keep/drop decision; update the research log.
```

---

## Phase 6 — Iteration 3: time-decayed estimation

```text
[Paste STANDING INSTRUCTIONS]

CONTEXT: beta is estimated from equally weighted means of r̃ and Σ̃ over every month since the start,
so the model is slow to unlearn a premium that decays (the low-risk tilt reversed after 2020 in the old
run). Build on the version kept after Phase 5.

PRE-REGISTER first (keep if posterior probability of improvement >= 80% over 1981–2020; kill if h = ∞
is chosen in most years or the gain misses the bar).

CHANGE:
  r̄_y = Σ_{s<y} δ^{y−s} r̃_s / Σ_{s<y} δ^{y−s},  Σ̄_y likewise,  δ = 0.5^{1/h}
1. coefs_from_sums and december_utility in portfolio/portfolio_ml.py: discounted running sums
   (YearSums already holds yearly sums).
2. New portfolio_ml.half_life_years: [3, 5, 10, null] (null = ∞ = current model). Add h to search and
   select_hypers so it is picked each December with (g, p, lambda) — out of sample by construction.
3. Test: h = ∞ reproduces the Phase-5 betas exactly.
4. Report the chosen h by year, theme exposures over time (especially low risk), and the decade
   sub-periods.

STOP gate — keep/drop decision; update the research log.
```

---

## Phase 7 — Holdout, monitoring script and report updates

```text
[Paste STANDING INSTRUCTIONS — and add: THE HOLDOUT IS UNLOCKED FOR THIS PHASE ONLY.]

TASK:
1. Set dates.holdout_locked: false in a new configs/experiments/holdout.yaml (never in base.yaml).
   Run ONCE, with no changes to any model choice: the baseline (tag baseline-v1) and the final kept
   version, over 2021-01 .. 2025-12, continuing the walk-forward. Report Table 2 and Table 3 for the
   holdout, and the 1981–2020 vs 2021–2025 comparison (post-publication decay).
2. Deflated Sharpe ratio (Bailey & López de Prado, 2014) of the final model, using the trial count from
   docs/research_log.md.
3. scripts/11_monitor.py (+ logic in src/jkmp/evaluation/monitor.py): monthly checks with flags —
   rolling 36-month IR < 0 or drawdown > 2 × TE budget; beta/theme exposure outside limits; rolling 12-
   month rank IC < 0 for 6 months; realised vs predicted trading cost above prediction for 3 months;
   chosen half-life jump > 1 grid step; bias statistic of the TE forecast outside 1 ± √(2/T).
   Output a one-page PNG/HTML dashboard for the slide appendix.
4. Update CLAUDE.md, docs/deviations.md and write docs/results_summary.md (tables + one paragraph per
   iteration: hypothesis, result, decision). Do not rewrite the report itself.

STOP gate — final report.
```

---

### Notes for you (not for Claude Code)

- The keep rule uses 1981–2020 because the paper reports there; the 2021–2025 holdout is the honest
  out-of-sample check, touched once. State this in the report's methodology.
- If Phase 2 shows the 1972 Nasdaq jump matters, decide D1 before Phase 3 — it changes the baseline.
- If compute bites in Phase 5, run Iteration 2 on 1981–2020 only at N = 250 and say so.
