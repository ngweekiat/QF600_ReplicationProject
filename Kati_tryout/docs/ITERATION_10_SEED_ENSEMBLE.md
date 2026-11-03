# Iteration 10: fixed equal-weight RFF seed ensemble

Starts from accepted iteration 4 (ancestry 1, 2, 4). Rejected iterations 5-9
are absent. Five seeds are declared before any new fit: **1, 2, 3, 4, 5**, each
with weight **20%**. Seed 1 is the original seed. No seed or ensemble size is
chosen using pretest or test performance. This is a separately labelled ensemble
extension, not a claim that the paper used this particular combination.

## Procedure

1. Diagnose seed sensitivity on usable pretest portfolio returns, 2016-2019,
   using the iteration-4 gamma_TE (174.1) for all five members. Report pairwise
   stock-weight differences, net active-return correlations and cost cancellation.
   Reproduce the saved seed-1 pretest control. Dispersion is descriptive evidence
   of random-feature sensitivity, not proof of alpha or a performance-based gate.
2. Each seed independently uses the original annual validation-utility selection
   over the same RFF variance, feature-count and 101-penalty ridge grids. Year Y
   settings and coefficients use only returns through Y-1. No one-SE rule,
   recent weighting, alternative risk half-life or iteration-9 cost learning.
3. For each stock and month, average the five **aim weights**. Coefficients from
   different random feature bases are not directly comparable and are not averaged.
4. Recalibrate the same 1% pretest ex-ante tracking-error target on the combined
   portfolio using the original bracket and log-interpolation rule. All members
   in any fit share the same gamma. There is no seed-specific risk normalisation
   or post-test rescaling. Interpolation retains the existing approximate target.
5. Execute the combined aims through the original monthly recursion once, from
   the same initial VW book, with the same drift, exits, wealth and impact costs.
   Save every member as an audit, but report the combined executed strategy.
6. Compare 72 months, 2020-2025, with iteration 4. Keep only if paired six-month
   circular-block bootstrap P(better paper-net Sharpe) >= 80% (10,000 samples,
   original seed 20261102). No drawdown gate. Report block lengths 3/6/12 too.

## Independent identities

With common inputs and initialisation, the execution recursion is affine in the
aim weights, even across annual coefficient changes. Therefore the combined
target and inherited weights equal the means of member weights. For member
orders d_j and their mean d_bar, quadratic impact obeys:

```
TC_combined = .5 W sum(lambda * d_bar^2)
mean(TC_members) - TC_combined
    = .5 W sum(lambda * mean_j((d_j - d_bar)^2)) >= 0
```

This cost saving is relative to the average of five separately executed books,
not a guarantee of savings or outperformance versus iteration 4. Costs and
Sharpe ratios are never averaged to manufacture ensemble performance. Actual
costs are reconstructed from saved combined orders. Synthetic tests cover exits,
coefficient jumps, cancellation, identical members and missing member keys.

Stocks, dates, selected 50 inputs, original learning/risk equations, annual refit
timing and actual cost assumptions remain fixed. Pretest diagnostics are completed
before calibration and the test fit. The 2020-2025 history has already been used
for research decisions; this is not independent unseen confirmation. The extra
model fits do not provide five independent samples of market returns.

Run from the repository root:

```powershell
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/iteration_10/run.py
```

Code: `scripts/seed_ensemble.py`. Outputs: `outputs/iteration_10/us_large/`.
Every pretest gamma fit and every fixed-seed member is retained separately.

## Completed result: 2020-2025

| Measure | Accepted iteration 4 | Iteration 10 |
| --- | ---: | ---: |
| Annual net excess return | 15.13536% | 15.08129% |
| Paper-convention net Sharpe | 0.870264 | 0.868582 |
| Annual actual trading costs | 0.059453% | 0.065569% |
| Worst additive drawdown (percentage points) | -27.422 | -27.512 |

**P(better versus iteration 4) = 28.61%; not included.** The paired bootstrap
95% interval for the Sharpe difference is [-0.008333, +0.004116]. Accepted
ancestry remains **1, 2, 4**. No seed was removed or reweighted after seeing
these results, and no further ensemble size or seed list was tried.

The pretest median pairwise net active-return correlation is 0.79984, with
stock-weight RMS differences averaging 36.12 basis points. The original seed-1
pretest control is reproduced. The ensemble's shared gamma_TE is 161.5,
calibrated using pretest gamma 100 and 250 and the original interpolation.

Economically, combining trades saves **1.286 basis points per year** versus
the mean cost of executing the five members separately at the same gamma.
Mean separate-member cost is 7.843 bp/year; combined cost is 6.557 bp/year.
However, iteration 4 costs only 5.945 bp/year. Relative to iteration 4,
iteration 10 has 4.796 bp/year lower gross returns plus 0.612 bp/year more
costs, reducing net returns by 5.407 bp/year. Diversifying random-feature
approximations did not improve this realised sample, despite correct trade
netting. The savings identity is a comparison with the average of separate
members, not evidence that the ensemble beats the retained strategy.

All 90 code tests and 251 independent saved-result checks pass, covering
stock accounting for every member and combined book, annual choices, pretest
timing, bootstrap probabilities, and the quadratic cost identity. These are
selected-test results, not independent unseen confirmation.
