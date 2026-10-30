# Results summary

Python replication of Jensen, Kelly, Malamud & Pedersen (2026) on the paper's timeline (data 1952-12, validation
1971–1980, **test 1981–2020**, 480 months, \$10B at end-2020, γ = 10), scaled down to a top-80 NYSE + Nasdaq universe and
18 characteristics in 6 themes (`docs/deviations.md`). Three pre-registered iterations, then a holdout (2021–2025) run once.
Trial count: 6 (`docs/research_log.md`). Baseline = git tag `baseline-v1`.

## Baseline vs the paper (Table 2, 1981–2020)

| Method | Utility (ours / paper) | SR_net | TC | Turnover |
|---|---|---|---|---|
| Market | −0.035 / −0.033 | 0.51 / 0.51 | 0.001 / 0.000 | 0.02 / 0.01 |
| 1/N | −0.046 / −0.051 | 0.48 / 0.54 | 0.006 / 0.004 | 0.06 / 0.07 |
| Minimum variance | −0.201 / −0.771 | −1.07 / −6.41 | 0.21 / 0.82 | 0.63 / 1.35 |
| Factor-ML | −2.64 / −2.49 | −17.1 / −18.1 | 2.59 / 2.51 | 2.31 / 2.79 |
| Markowitz-ML | −41.5 / "−" | −142 / "−" | 41.3 / "+" | 7.8 / 79.4 |
| Static-ML | −0.011 / −0.165 | 0.56 / 0.79 | 0.014 / 0.044 | 0.20 / 0.80 |
| Static-ML\* | 0.018 / 0.030 | 0.63 / 0.81 | 0.007 / 0.035 | 0.16 / 0.68 |
| **Portfolio-ML** | **0.019 / 0.086** | 0.61 / 1.33 | 0.004 / 0.006 | 0.08 / 0.25 |
| Portfolio-ML (linear) | 0.026 / — | 0.76 / — | 0.002 / — | 0.06 / — |

Plumbing replicates (Market, 1/N, Factor-ML cost). The ranking replicates qualitatively — cost-aware methods have
positive utility, Markowitz-ML, Factor-ML and minimum variance strongly negative — but Portfolio-ML's edge does not: its
utility is a quarter of the paper's and P(Portfolio-ML > Static-ML\*) = 0.55 (paper 0.999). The linear version beats both
(P = 0.96 vs Portfolio-ML). Likely causes: 18 vs 115 characteristics and an 80-stock large-cap universe (lower predictability:
Markowitz-ML gross SR 1.0 vs 2.06), and a covariance that under-predicts portfolio risk by 30–46% before 1990. Feature
importance (20 seeds, s.e. ≤ 0.0006): quality (0.013) and momentum (0.005) matter most for Portfolio-ML; shuffling low risk
*raises* its utility (−0.0045). Sub-periods: Portfolio-ML utility 0.016 (1981–2000) / 0.021 (2001–2020); Static-ML\* 0.030 / 0.005.

## Iterations (decided on 1981–2020 as pre-registered)

| Trial | Change | Metric | Result | Decision |
|---|---|---|---|---|
| 1 | Baseline | Table 2 | above | reference |
| 2a–c | Benchmark-relative Portfolio-ML, TE 1/2/4% | net IR vs VW | 0.50 / 0.42 / 0.34; P(IR > 0) ≥ 98% | **keep** |
| 3 | Top-250 universe (2% TE) | net IR vs own VW | 0.67 vs 0.42; P(better) = 98% | **keep → final** |
| 4 | Half-life β, h ∈ {3, 5, 10, ∞} | net IR | 0.58 vs 0.65; h = ∞ in 88% of years | **kill** |

**Trial 2 — benchmark-relative Portfolio-ML.** *Hypothesis:* a \$10B large-cap mandate is benchmark-relative; the
total-return model picks its own beta (0.10 vs the VW universe) and holds an inverse-vol long book through the constant
signal. *Change:* active book a = Ωβ on top of value weights, active risk aversion γ_TE calibrated on 1972–1980 to 1/2/4%
ex-ante TE, no constant. *Result:* net IR 0.50 / 0.42 / 0.34, β 0.85–0.97, tilts to quality and low risk; out-of-sample
TE ran 1.6–2× the calibrated budget. *Decision:* keep (the comparison with the baseline's active utility was an easy bar;
the informative result is a positive net IR at every budget).

**Trial 3 — breadth.** *Hypothesis:* IR ≈ TC·IC·√BR; a wider universe adds breadth and higher IC. *Change:* top-250
universe (buffer 220/300) with its own VW benchmark; γ_TE re-calibrated (2% → 49.09). *Result:* net IR 0.67 vs 0.42
(P = 98%), realised TE 3.1%; rank IC 0.043 in ranks 81–250 vs 0.031 in the top 80, higher in every decade. Capacity (net IR
< 0.3, vs the paper index): \$107B vs \$31B (1981–2020), \$55B vs \$7B (1990–2020); VIX- and stock-vol-scaled costs move it
by < 2%. *Decision:* keep — the final version. *Caveat:* a net IR that subtracts the benchmark's own trading cost rises with
wealth (it credits a fund for skipping rebalancing), so capacity is measured against the paper index.

**Trial 4 — time-decayed β.** *Hypothesis:* equal-weighted β estimation is slow to unlearn decaying premia. *Change:*
yearly-discounted means of r̃, Σ̃ with half-life h picked each December (same selection score). *Result:* h = ∞ chosen in
35 of 40 years; where h = 10 was chosen (1998–2007) the 2000s IR fell from 0.57 to 0.31; overall 0.58 vs 0.65
(P(better) = 2.6%). *Decision:* kill.

## Holdout 2021–2025 (run once; `outputs/holdout/tables/`)

| Method | Utility 1981–2020 → 2021–2025 | SR_net | TC |
|---|---|---|---|
| Market | −0.035 → 0.009 | 0.51 → 0.85 | 0.001 → 0.000 |
| Static-ML\* | 0.018 → 0.009 | 0.63 → 0.46 | 0.007 → 0.002 |
| Portfolio-ML | 0.019 → −0.004 | 0.61 → 0.25 | 0.004 → 0.002 |
| Portfolio-ML (linear) | 0.026 → 0.001 | 0.76 → 0.30 | 0.002 → 0.002 |
| **Final: active, top-250, 2% TE** | −0.010 → 0.010 | 0.67 → 0.83 | 0.003 → 0.001 |

Final version vs its benchmark: **net IR 0.65 → 0.34**, active net 2.0% → 0.6% a year, TE 3.1% → 1.8%,
P(IR > 0) = 78% (60 months). Post-publication decay is visible in every ML method; the cost-aware methods remain
positive or near zero while Factor-ML and Markowitz-ML stay deeply negative. In the holdout no pair among Portfolio-ML,
its linear version, Static-ML\* and the final version is separated with P ≥ 80%. **Deflated Sharpe ratio** of the final
version (1981–2020, 6 trials): 0.90 — the in-sample net IR of 0.64 clears the multiple-testing hurdle (0.44) with 90% probability.

## Monitor (`scripts/11_monitor.py`, `outputs/holdout_final/monitor/dashboard.{png,html}`)

As of 2025-12 the final version is flagged on the **rolling 36-month net IR (−0.21)**; beta (0.99), theme exposures (max 0.20),
the TE bias statistic (1.01) and the half-life are within limits. Holdout flags (of 60 months): IR 20, drawdown 9, IC 15,
realised-vs-predicted cost 27. The cost check fires in 40% of test months, so as specified it is too sensitive to be
actionable.

## Caveats

Decisions were taken on the test window 1981–2020, as each phase prompt specified, not on a separate validation window;
the deflated Sharpe ratio is the correction offered. 18 characteristics (a teammate branch with 50 exists and was not
merged). Σ under-predicts portfolio risk before 1990 (×21 daily-to-monthly scaling, as R). The holdout restarts every
method from value weights at 2020-12.
