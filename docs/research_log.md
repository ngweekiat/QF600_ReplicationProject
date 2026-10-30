# Research log

Every run of a variant counts as a trial. Each entry is pre-registered (hypothesis, change, metric, keep
threshold, kill criterion, date) **before** it runs; the result and the decision are appended after.
Test window: return months 1981-01 .. 2020-12. The holdout 2021-01 .. 2025-12 stays locked until Phase 7.

**Running trial count: 6** (trial 1 baseline; trials 2a–2c active Portfolio-ML at three TE budgets; trial 3 top-250; trial 4 half-life)

---

## Trial 1 — Baseline (`baseline-v1`)

**Pre-registration (2026-10-30, before any run)**

- **Purpose.** The reference every later iteration is measured against. Not an iteration: the design is
  the paper's timeline with our scale-down (top-80 NYSE + Nasdaq universe, 18 features in 6 themes; see
  `docs/deviations.md`), with the Phase 3 baseline fixes: grid edges p ∈ {8 .. 512}, u ∈ {0 .. 1} (D33),
  feature importance over 20 seeds (D31).
- **Hypothesis (replication).** Over 1981–2020 at \$10B and γ = 10: (H1) Portfolio-ML has the highest
  realised utility of all methods; (H2) Markowitz-ML has the highest gross Sharpe ratio and a large negative
  net return; (H3) Factor-ML and Minimum variance have negative utility because of trading costs;
  (H4) Static-ML\* has positive utility, below Portfolio-ML; (H5) Table 3: P(Portfolio-ML > Static-ML\*) ≥ 0.95.
- **Metric.** Table 2 utility (annualised, gross-variance utility as R `pf_summary`), SR_net, TC; Table 3
  posterior probabilities.
- **Keep threshold / kill criterion.** None — a baseline is not kept or killed. It is *accepted* as the
  reference if the plumbing checks pass: Market R ≈ 0.08, Vol ≈ 0.15, TC ≈ 0 (Replication Plan §6);
  Portfolio-ML unshuffled feature-importance run reproduces scripts/07 (max |Δr| < 1e-10); no holdout
  month in any output. H1–H5 are reported as replicated or not; a failure is a finding, not a reason to
  re-tune.
- **Decisions fixed before the run.** D1 keep NYSE + Nasdaq; Σ scaling ×21 as R (R3), despite the
  pre-2000 under-prediction found in Phase 2 (candidate iteration, not changed here); the return model is
  fitted for 1971–2020 only while the holdout is locked.

---

## Trial 2 — Benchmark-relative Portfolio-ML (`configs/experiments/active_pfml.yaml`)

**Pre-registration (2026-10-30, before any run of this variant)**

- **Motivation.** In the old 2020–2025 run Portfolio-ML had β ≈ 0.20 and α ≈ −2.6% a year against the
  value-weighted universe and 0.82 correlation with minimum variance: the total-return objective lets the
  model pick its own market exposure, and the vol-scaled constant column is an inverse-volatility long book.
  A real \$10B large-cap mandate is benchmark-relative with a tracking-error budget.
- **Change (one experiment, three budgets).** π_t = π_t^VW + a_t, a_t = Ω_t β, with β maximising
  E[a_tᵀr_{t+1} − (w_t/2) Δa_tᵀΛ_tΔa_t − (γ_TE/2) a_tᵀΣ_t a_t]: the Portfolio-ML machinery with γ → γ_TE
  (in Σ̃ and in m), no constant signal column (`portfolio_ml.add_constant: false`), and the weight
  recursion on active weights a_t = m a_inh + (I − m) A_t, a_inh = π_start − π^VW_t, then π_t = π^VW_t + a_t
  (so the traded Δπ equals Δa exactly; the objective's ΔΩ ignores the benchmark's own drift, reported as
  the Market row's turnover and cost). RFF signals, same (g, p, λ) grids and December selection as the
  baseline. γ_TE is calibrated **before the test window**: mean ex-ante tracking error √(aᵀΣa)·√12 of the
  implemented active weights over return months 1972–1980 = 1%, 2% and 4% a year (log-log interpolation
  over `investor.gamma_te_grid`; calibration runs look only at ex-ante TE, never at returns).
- **Hypothesis.** H1: net information ratio vs the VW universe is positive. H2: the active book's mean
  monthly active utility u_t = (active net return)_t − (γ_TE/2)(active_t − mean active)² beats the baseline
  Portfolio-ML's active utility (its return minus the VW universe's, same formula and γ_TE).
- **Metric.** Net information ratio = mean(active return − (tc − tc_VW)) / sd(active return), annualised,
  1981–2020; also gross IR, realised and ex-ante TE, β to the VW universe, share of stocks with negative
  total weight, active theme exposures (Xᵀa).
- **Keep threshold.** For a budget, keep if P(mean active utility of the variant > the baseline's) ≥ 80%
  (R's posterior formula, Table 3) over 1981–2020. Report all three budgets.
- **Kill criterion.** Kill the idea if at no budget is P(net IR > 0) ≥ 80%
  (P = Φ(mean net active / (sd/√T))).
- **Trials.** The three evaluated budgets are trials 2a, 2b, 2c (count → 4). Calibration runs are listed but
  their test-window performance is not computed.
- **Caveat recorded in advance.** The baseline's active return has a large tracking error, so at high γ_TE its
  active utility is strongly negative and H2 is an easy bar; H1 (net IR > 0) is the informative test.

**Calibration (2026-10-30, validation return months 1972-01 .. 1980-12 only; no test-window number computed)**

Mean ex-ante TE (annual) by γ_TE: 1 → 6.44%, 2.5 → 5.27%, 5 → 3.72%, 10 → 2.43%, 25 → 1.18%, 50 → 0.70%,
100 → 0.42%, 250 → 0.22% (monotone). An initial timing point at γ_TE = 100 (grid then 10 .. 2500) showed
the budgets lay below the grid; the grid was shifted to 1 .. 250 before calibrating. Calibrated (log-log
interpolation): **TE 1% → γ_TE = 31.19, 2% → 12.82, 4% → 4.331.** These three runs are trials 2a–2c.

**Result (2026-10-30, test window 1981-01 .. 2020-12, 480 months; `outputs/active/summary.csv`)**

| Budget (γ_TE) | Active net (ann.) | Ex-ante TE | Realised TE | Net IR | P(net IR > 0) | β to VW | Neg. weights | P(beats baseline) |
|---|---|---|---|---|---|---|---|---|
| 1% (31.19) | 1.24% | 1.95% | 2.47% | 0.50 | 99.9% | 0.97 | 11% | 100.0% |
| 2% (12.82) | 1.92% | 3.63% | 4.63% | 0.42 | 99.5% | 0.93 | 20% | 100.0% |
| 4% (4.331) | 2.73% | 6.45% | 8.03% | 0.34 | 98.3% | 0.85 | 28% | 100.0% |
| Baseline Portfolio-ML, vs VW | −4.00% | 13.7% | 14.6% | −0.27 | 4.2% | 0.10 | 40% | — |

Benchmark's own trading (VW universe): turnover 1.9% a month, TC 0.001% a year — the Δπ ≈ Δa gap in the
objective is negligible. Active TC 0.14% / 0.31% / 0.57% a year.

- **Decision: KEEP at all three budgets** (pre-registered rule: P(mean active utility > baseline's) ≥ 80%;
  observed 100.0% each). **Not killed**: P(net IR > 0) ≥ 80% at every budget.
- **As anticipated in the pre-registration, H2 is an easy bar** — the baseline is not a benchmark-relative
  strategy (β 0.10, 14.6% TE), so its active utility at these γ_TE is −0.09 to −0.37 a year. H1 (net IR > 0) is
  the informative result, and it holds at every budget, with IR falling as the budget grows.
- **Surprises.** (1) Out-of-sample ex-ante TE is about 1.6–2× the budget calibrated on 1972–1980, and realised TE
  is a further 1.25–1.28× ex-ante (Σ under-predicts risk before 2000, Phase 2); a live mandate would need to
  re-calibrate γ_TE each year (e.g. on the trailing ex-ante TE) — a candidate follow-up, not run. (2) The
  active book's tilts are quality and low risk (mean Xᵀa 0.28 / 0.23 at 1%, 0.97 / 0.83 at 4%), then momentum
  and investment; almost no value or short-term reversal. (3) In total-return terms at γ = 10 the active runs
  have *lower* utility than the baseline (−0.018 / −0.010 / −0.007 vs 0.019): full market beta costs γ/2·σ²_m.
  The variant answers a different mandate; it does not replace the baseline for the paper's objective.

---

## Trial 3 — Breadth: top-250 universe for active Portfolio-ML (`configs/experiments/universe_250.yaml`)

**Pre-registration (2026-10-30, before any run of this variant)**

- **Motivation.** Old run: rank IC 0.002 inside the top 80 vs 0.07 across all screened stocks. Fundamental law
  IR ≈ TC × IC × √BR: a wider universe adds breadth and (if IC is higher among smaller stocks) skill.
- **Reference ("the Phase-4 version").** Active Portfolio-ML at the **2% TE budget** (γ_TE = 12.82, top-80, net
  IR 0.42). Chosen because 2% was the middle budget fixed before any result, not the best-performing one.
- **Change.** `universe.target_n: 250`, buffer `entry_rank: 220`, `exit_rank: 300`; everything else as the
  reference (active book, no constant, RFF, same grids). Benchmark = the top-250 universe's own VW portfolio.
  Covariance factors stay estimated on the top 500. γ_TE re-calibrated to 2% ex-ante TE on 1972–1980 at N = 250
  (validation years only). Separate data/outputs folders (`data/processed_u250`, `outputs/u250`).
- **Metric.** Net IR vs the universe's own VW benchmark, 1981–2020.
- **Keep** if the posterior probability of improvement over the reference is ≥ 80%: P(mean d_t > 0) with
  d_t = an_t^250 / sd(a^250) − an_t^80 / sd(a^80) (monthly net active returns scaled by each book's TE, so mean
  d·√12 = IR_250 − IR_80), R's posterior rule. **Kill** otherwise — the finding is then that the 18
  characteristics, not the universe, are the bottleneck.
- **Also reported (not part of the decision).** Rank IC by size bucket (top 80, 81–250 by market-cap rank among
  screened stocks) per decade; capacity curve (net IR vs wealth \$1B–\$250B at the 2% γ_TE, both universes);
  cost sensitivity by re-costing the final weights — (a) Λ_t × VIX_t / mean VIX 1990–2020 (1990–2020 only),
  (b) Λ_i,t × σ_i,t / σ̄_i,t with σ̄ the stock's mean σ over the past 60 months (data ≤ t only); capacity = AUM
  where net IR falls below 0.3 (log-wealth interpolation) under the base costs, (a) and (b).
- **Trials.** One evaluated variant (trial 3, count → 5). Capacity/cost runs are evaluation of the same variant
  and the reference at other wealth levels, not new trials. N = 500 only if the full N = 250 run < ~8 hours.

**Calibration at N = 250 (validation return months 1972–1980 only).** Ex-ante TE by γ_TE: 1 → 7.77%, 2.5 → 7.33%,
5 → 6.32%, 10 → 4.92%, 25 → 3.18%, 50 → 1.97%, 100 → 1.20%, 250 → 0.57%. Calibrated: 1% → 125, **2% → 49.09**,
4% → 15.44. Timing: one year of PF-ML inputs at N = 250 = 0.9 s; a calibration point 1.5 min. N = 500 would be
~8× per point (~13 min) — well under the 8-hour gate, considered but not run (it would be a new trial).

**Result (2026-10-30, 1981–2020; `outputs/u250/breadth/`)**

| 2% TE budget | γ_TE | Net IR | Active net | Ex-ante TE | Realised TE | β | TC |
|---|---|---|---|---|---|---|---|
| top-80 (reference) | 12.82 | 0.42 | 1.92% | 3.63% | 4.63% | 0.93 | 0.31% |
| **top-250** | 49.09 | **0.67** | 2.08% | 2.36% | 3.12% | 0.96 | 0.29% |

- **Decision: KEEP top-250.** P(net IR top-250 > top-80) = 98.4% (bar 80%); 98.5% with the paper-index benchmark.
- **Rank IC by size** (mean monthly, 1971–2020): top-80 0.031, ranks 81–250 0.043; higher in 81–250 in every decade
  (1970s 0.063 / 0.077 … 2010s 0.010 / 0.012; 2020 alone −0.099 / −0.063). The old run's "IC 0.002 inside the top 80"
  does not recur on 1971–2020; breadth (√BR) and somewhat higher IC both contribute.
- **Capacity** (net IR vs wealth at γ_TE fixed; AUM where net IR < 0.3), **against the paper index**: top-250 \$107B
  (1981–2020) / \$55B (1990–2020); top-80 \$31B / \$7B. Cost sensitivity barely moves it: stock-vol-scaled Λ \$106B /
  \$30B; VIX-scaled Λ (1990–2020) \$56B / \$7B.
- **Caveat found in the analysis (metric definition).** The pre-registered net active return subtracts the
  benchmark's own trading cost (tc_VW). At \$10B that is 0.1% a year and immaterial (decision unchanged), but it
  grows ∝ wealth (2.4% a year at \$250B) and makes net IR *rise* with wealth (2.1–2.3 at \$250B): a fund too large
  to rebalance is credited with the benchmark rebalancing it skips. Capacity is therefore reported against the
  paper index (tc_b = 0); the traded-VW curve is kept in `capacity_curve.csv` for transparency. Future trials use
  the paper index as the benchmark.
- **N = 500** was feasible (~13 min per run vs the 8-hour gate) but not run: it would be a new, un-pre-registered trial.

---

## Trial 4 — Time-decayed estimation of β (`configs/experiments/halflife.yaml`)

**Pre-registration (2026-10-30, before any run of this variant)**

- **Motivation.** β is estimated from equally weighted means of r̃ and Σ̃ over every month since the start, so the
  model is slow to unlearn a premium that decays (the low-risk tilt reversed after 2020 in the old run).
- **Reference ("the Phase-5 version").** Trial 3: active Portfolio-ML, top-250, γ_TE = 49.09 (2% budget), net IR
  0.646 vs the paper index (0.666 vs the traded VW benchmark).
- **Change.** r̄_y = Σ_{s<y} δ^{y−s} r̃_s / Σ_{s<y} δ^{y−s} n_s, Σ̄_y likewise, δ = 0.5^{1/h} (yearly sums), with
  h ∈ {3, 5, 10, ∞} (`portfolio_ml.half_life_years`; ∞ = the reference model) picked each December jointly with
  (g, p, λ) by the **same** selection score as before — the equally weighted cumulative validation utility since
  1971 — so the choice of h is out of sample and every h is judged on one yardstick. γ_TE fixed at 49.09 (not
  re-calibrated, so only one thing changes).
- **Metric.** Net IR vs the paper index (benchmark without trading cost; decided in trial 3), 1981–2020.
- **Keep** if P(improvement over the reference) ≥ 80% (same IR-difference test as trial 3). **Kill** if h = ∞ is
  chosen in most (> 50%) of the test years 1981–2020, or the gain misses the bar.
- **Also reported.** Chosen h by year; theme exposures over time (especially low risk); net IR by decade.
- **Trials.** One evaluated variant (trial 4, count → 6).

**Result (2026-10-30, 1981–2020, vs the paper index; `outputs/u250_hl/halflife/`)**

| | Net IR | Active net | Realised TE | β | 1980s | 1990s | 2000s | 2010s | 2020 |
|---|---|---|---|---|---|---|---|---|---|
| trial 3 (h = ∞) | 0.646 | 2.02% | 3.12% | 0.96 | 0.92 | 0.57 | 0.57 | 0.77 | −1.09 |
| trial 4 (h tuned) | 0.580 | 1.81% | 3.13% | 0.97 | 0.92 | 0.53 | 0.31 | 0.77 | −1.09 |

- Chosen h: ∞ in 35 of 40 test years; h = 10 in 1998, 2000, 2004, 2005, 2007; h = 3 and 5 never.
- **Decision: KILL.** P(net IR trial 4 > trial 3) = 2.6% (bar 80%) and h = ∞ is chosen in 88% of test years (> 50%) —
  both kill conditions hold. The years where the validation score preferred h = 10 are the years that cost IR (2000s
  0.57 → 0.31). Low-risk exposure falls in the 2000s either way (0.47 → 0.10 / 0.07); the slow decay of the old run
  is not repaired by discounting the estimation window, on 1981–2020.
- **Final kept version: trial 3** — active Portfolio-ML, top-250, γ_TE = 49.09 (2% budget), paper-index benchmark.

---

## Holdout 2021-01 .. 2025-12 — run once (Phase 7)

**Statement before the run (2026-10-30).** All model choices are frozen: the baseline (`baseline-v1`, trial 1) and the
final kept version (trial 3: active Portfolio-ML, top-250, γ_TE = 49.09, paper-index benchmark). Before unlocking,
the current code was checked to reproduce both stored 1981–2020 runs exactly (max |Δr| = max |Δtc| = 0). The holdout
continues the walk-forward (return model refits for 2021–2025, β and December picks keep updating on data through
the previous year); each method starts from value weights at 2020-12 as every window does. Configs:
`configs/experiments/holdout{,_linear,_final}.yaml`. Nothing below may feed back into a model choice.

**Result (run once, 2026-10-30; `outputs/holdout/tables/`).** Final version vs the paper index (top-250 VW):
net IR 0.646 (1981–2020) → **0.344 (2021–2025)**, active net 2.02% → 0.63% a year, realised TE 3.1% → 1.8%, β 0.96 → 1.00,
P(net IR > 0) = 77.8% over 60 months. Baseline Portfolio-ML: utility 0.019 → −0.004, SR_net 0.61 → 0.25. Holdout Table 3:
no method beats another among Portfolio-ML, its linear version, Static-ML\* and the final version with P ≥ 80% (all
0.16–0.84); all four beat Factor-ML and Markowitz-ML with P = 1.00. Deflated Sharpe ratio of the final version over
1981–2020 with 6 trials: SR₀ = 0.44 a year (annualised), DSR = **0.90**. Nothing here fed back into any model choice.
