# Research log

Every run of a variant counts as a trial. Each entry is pre-registered (hypothesis, change, metric, keep
threshold, kill criterion, date) **before** it runs; the result and the decision are appended after.
Test window: return months 1981-01 .. 2020-12. The holdout 2021-01 .. 2025-12 stays locked until Phase 7.

**Running trial count: 1**

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
