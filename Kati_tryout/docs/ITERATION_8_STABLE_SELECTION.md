# Iteration 8: stable model selection

Start from the last accepted setup, currently iteration 4 (ancestry 1, 2, 4).
Rejected decay, learning-cost and risk-half-life changes are absent. Keep stocks,
50 characteristics, dates, RFF/ridge candidate grids, coefficients, risk model,
costs and monthly portfolio implementation equations unchanged.

Change only the annual model-selection rule. Each December:

1. Reconstruct monthly candidate utility using the original moment objective,
   with coefficients fitted only on returns before their validation year.
   Independently require cumulative scores to match the original yearly sums.
2. Find the highest historical mean utility among the same 1,414 candidates.
3. Estimate the standard error of that best candidate's past mean utility using
   2,000 circular six-month-block bootstrap draws, seed 20261103 plus selection
   calendar year. The one-SE band is fixed, not tuned on test results.
4. Retain candidates whose mean is at least best mean minus that SE. Choose
   fewer RFF features, then stronger ridge, then smaller RFF variance.
5. Use that choice for the next return year. Later annual choices may use earlier
   realised test returns, as in the original expanding workflow, but never future
   returns. The selection rule itself is frozen before the test comparison.

This is a heuristic robustness extension, not a significance test or the paper's
original maximum-utility selection. Selection utility is the original candidate
moment objective, not a newly invented return forecast or final account curve.
Bootstrap bands do not correct all model-selection bias or prior research trials.

Recalibrate the inherited 1% ex-ante TE target using this rule and pretest returns
only (existing usable 2016-2019 portfolio returns; 2015 starts candidate validation).
The gamma bracket is declared in the JSON protocol before test. If no declared
setting brackets 1%, report calibration infeasibility rather than pretending the
risk target was met. Otherwise evaluate one calibrated 2020-2025 portfolio against
iteration 4, with the same 80% P(better paper-net Sharpe) rule, 10,000 paired
six-month-block bootstrap draws. Drawdown adds no gate. No unseen confirmation is
claimed from these repeatedly inspected test months.

Run `python -X utf8 Kati_tryout/scripts/iteration_8/run.py`.
Source: `scripts/stable_selection.py`. Outputs: `outputs/iteration_8/us_large/`,
including monthly candidate validation, annual `selection_audit.csv`, calibration
trials, chosen models, weights, returns, charts and the decision summary. All
earlier model fits remain read-only.

## Completed result

Pretest TE calibration selects gamma_TE **66.15** under the fixed one-SE rule.
For test years 2020-2023 it selects **32** RFF features; for 2024-2025 it selects
**8**, versus the maximum-utility candidate's 512 in those comparisons.

| 2020-2025 test metric | Iteration 4 | Iteration 8 |
|---|---:|---:|
| Annual net excess return | 15.1354% | 15.2777% |
| Paper net Sharpe | 0.870264 | 0.873828 |
| Annual actual trading cost | 0.05945% | 0.02084% |
| Monthly turnover | 4.0359% | 1.9949% |
| Worst additive drawdown | -27.4224 pp | -28.1814 pp |

The rule reduces complexity and turnover, but **P(better) is only 59.67%** versus
iteration 4. **Not included** under the 80% rule; final ancestry remains **1, 2, 4**.
All annual SEs, eligibility sets and tie-break choices are independently verified.
The 2026 choice in the audit uses December 2025 information and is not part of
the 2020-2025 evaluated portfolios.
