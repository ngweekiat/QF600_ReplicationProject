# Iteration 6: more selective trading

## Hypothesis and fixed setup

Iteration 4's learned stock tilts may trade away their small gross advantage.
Test whether a stronger preference against turnover when learning targets improves
net performance. This is an explicitly declared economic tuning extension of
Portfolio ML, not a claim that this extra preference is in the paper's default.

Start from the accepted state after iteration 5: currently iterations **1, 2, 4**.
Keep the selected 50, US large-cap membership/buffering, dates, original lambda,
$10bn reference wealth, RFF and ridge grids, annual refitting and monthly trading.
Rejected iteration 3's changed exit rank/costs and iteration 5's decay are absent.

## One change

For coefficient learning only, replace `risk + cost` with `risk + c * cost`, where
`c` is one of **1, 1.5, 2**. The 1x candidate is the unchanged control. Trading
speed, signal construction and realised-cost accounting retain the original
lambda. The cost multiplier is a preference in learning, not a claim about higher
actual execution costs. If weights are identical, reported costs are identical.

Each candidate retains the existing pretest 1% ex-ante tracking-error calibration.
The old gamma-5 decision is not an additional multiplier of active risk.

## Predeclared selection and evaluation

1. Run candidate-specific TE calibration without returns after December 2019.
2. Evaluate pretest implemented portfolios at **actual 1x costs**. As in existing
   active calibration, usable validation returns are January 2016-December 2019
   (48 months); 2015 supplies the first December hyperparameter choice. No dates
   or upstream training history are shortened.
3. Select maximum annual net mean-variance utility, common evaluation gamma 10;
   break exact ties in favour of the smaller preference. Retain every trial.
4. Freeze the chosen scalar and its calibrated gamma_TE before producing the
   2020-2025 candidate. Annual RFF/ridge choices and coefficient refits continue
   using previously realised history, exactly as in the inherited model.
5. Compare the candidate against the previous accepted strategy (currently 4),
   using 10,000 paired six-month circular-block bootstrap samples. Keep only if
   P(better paper-net Sharpe) >= 80%. Identical control results are not improvements.

Drawdown adds no acceptance gate. Reports use additive returns and percentage-point
drawdowns. The test has already guided earlier choices, so results remain
exploratory selected-sample evidence, not independent unseen confirmation.

## Files and command

`scripts/iteration_6/run.py` is the entry point; `scripts/selective_trading.py`
selects the pretest preference and records all trials. The learning extension is
in `scripts/jkmp/portfolio/portfolio_ml.py`, defaulting to exactly 1 for all other
runs. Protocol settings are in `configs/us_high_cap_iterations.json`.

```powershell
python -X utf8 Kati_tryout/scripts/iteration_6/run.py
```

Outputs: `outputs/iteration_6/us_large/`, including `pretest_trials/`,
`cost_preference_validation.csv`, `pretest_selection.json`, weights, monthly
returns, charts and a decision/inheritance summary. Earlier saved iterations are
checked and reused; this command fits only iteration 6. Rejected results remain.

## Current completed result

Pretest utility selects **2x** training cost preference (gamma_TE **145.7**).
On 2020-2025, paper net Sharpe rises from **0.870264** to **0.870916** and
annual net excess return from **15.1354%** to **15.1450%**. Annual actual costs
fall from **0.05945%** to **0.05084%**, and monthly turnover from **4.0359%** to
**3.6810%**. Additive worst drawdown changes from **-27.4224 pp** to **-27.4690 pp**.
P(better versus iteration 4) is **63.84%**, below 80%, so the experiment is
**not included**. The final accepted strategy remains iteration 4, ancestry
**1, 2, 4**. A modest historical economic improvement is not enough under the
declared evidence rule. No settings were changed after seeing this test result.
