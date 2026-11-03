# Current assumptions, exclusions and changes

## Revised Base-parent market anchor

The current retained research chain is **1, 4, 4.2, 4.3**. Iteration 4 restarts
from Base (gamma 10 in metadata) and excludes iteration 2. Its actual active-risk
equation continues to use pretest-calibrated gamma_TE 174.1, so fresh fitting
reproduces its earlier returns. P(test) changes to 99.63% against Base. The two
substages retain descriptive P(test), 66.88% and 53.17%, without an 80% gate.
Selected 50 characteristics, universes, dates and regional Base calculations
are unchanged. Trials 2, 3 and 5-10 retain original historical references;
they are not claimed to have been rerun on this new ancestry.

The new pre-4 diagnosis uses original Base execution over usable 2016-2019
validation months only. The 4.2/4.3 rationale images use 2015-2019 candidate
validation flows. No 2020+ outcome is used in these diagnoses. The Base return
gap is uncertain; costs explain much of its observed shortfall. Low market
beta motivates a benchmark-relative mandate, not proof of future gains.
One-SE selection remains a heuristic, not a multiple-testing correction or
equivalence test. All test results reuse already-inspected 2020-2025 data.
See [the detailed protocol and charts](ITERATION_4_SUBSTAGES.md).

## Earlier historical trials

Iteration 9 uses affine recursive benchmark-plus-active cost curves, including current and charged-exit orders, in coefficient learning. The benchmark-dependent linear term and recursive cost quadratic replace the earlier active-only finite-lag cost proxy. Original return/risk learning proxies, RFF/ridge grids, annual selection, actual execution, cost lambda and data remain unchanged. Fixed-coefficient polynomial identities and zero-ML recursion are verified independently. This is an active-extension refinement, not an exact reformulation of the full paper objective; actual annual refit-jump costs remain charged. See [the full derivation](ITERATION_9_BENCHMARK_NETTING.md).

Iteration 8 changes only annual candidate selection to a fixed historical one-SE rule, with pretest-only TE recalibration. Original coefficient, feature, risk, cost and execution formulas/grids are reused. Monthly candidate utility must independently match original December yearly sums. Every year's threshold uses only previously realised validation flows; fewer features, then stronger ridge, then smaller RFF variance win. This is an explicitly labelled methodological extension. Separately, the iteration-2 market-anchor diagnostic reads only iteration 2 and Market and never later iteration results; its regression decomposition is descriptive, not proof of future improvement.

Iteration 7 changes only stock-specific residual-volatility EWMA weighting. The 63/126/252-day grid is chosen on pretest utility at actual costs; each candidate gets pretest 1% TE calibration. Original saved factors/exposures/factor covariance are reused, and the reconstructed 126-day control must match saved risk before any portfolio fits. The chosen half-life is frozen before test. Base risks, stocks, 50 inputs, dates and costs remain unchanged; rejected iteration 6 is absent. See [its protocol](ITERATION_7_RISK_RESPONSIVENESS.md).

Iteration 6 adds an opt-in training-only cost preference (1, 1.5 or 2), selected by pretest actual-cost utility and frozen for test. The coefficient-learning denominator becomes risk plus the selected multiple of its cost term. Default 1 preserves the original portfolio mathematics exactly. Original execution lambda, cost accounting, stock membership, characteristics, dates and regional Base fits are unchanged. Each candidate uses pretest 1% TE calibration. See [ITERATION_6_SELECTIVE_TRADING.md](ITERATION_6_SELECTIVE_TRADING.md).

Performance reporting now uses additive cumulative net excess returns, matching the authors' `cumsum(r-tc)` charts. Drawdown is the additive path minus its running peak including initial zero, and cumulative performance versus Market is a difference of additive paths. Both are displayed in percentage points. This drawdown is a reporting extension, not a verified paper drawdown definition or a percent loss of investor capital. Geometric CAGR/Calmar reporting has been removed. Model fitting, saved monthly returns/costs/weights, annual means, Sharpe, utility, P and iteration acceptance are unchanged. Internal wealth and drift accounting retain their original compounding. See [ADDITIVE_REPORTING.md](ADDITIVE_REPORTING.md).

The active US high-cap experiment sequence is [the nine-step protocol](FIVE_ITERATION_PROTOCOL.md). It replaces the former three-iteration breadth/capacity sequence. Acceptance uses only the unchanged 2020–2025 test period's paired-bootstrap P(better net Sharpe) against the previous accepted version, threshold 80%. Annual refitting is already in Base. Rejected candidates remain reported; only accepted changes are inherited. Drawdown adds no gate. Test-selected final performance is not independent unseen confirmation. All regional base calculations and characteristics remain unchanged.

Base calibration selects the ridge penalty from the original 101-value validation grid. This applies to return models and regional Portfolio ML models. Market-specific validation, the equations, other hyperparameter selection and iteration settings are retained.

The latest authorised base adaptation is [market-specific RFF tuning](MARKET_SPECIFIC_RFF_TUNING.md): return-forecast hyperparameters are scored on historical members of each market separately, while fitting still uses its screened regional training panel. Portfolio ML already tunes its own RFF portfolio-weight model separately by market. No models or backtests were run for this change. The saved 100-candidate Static ML* results predate it.

The current methodology is documented in [US_HIGH_CAP_SELECTION.md](US_HIGH_CAP_SELECTION.md). It replaces the earlier three-market feature-selection method. Old characteristic archives and their fits were deleted at the user's request; active calculations use the new US high-cap-selected 50.

| Item | Current behavior |
|---|---|
| Characteristic selection | US high-cap held-stock observations only, 2005–2014 |
| Selection method | Observed correlation/covariance diagnostics with fixed association score, group quotas and redundancy penalties |
| ML training during selection | None; no single-feature regression R², return forecast or portfolio utility enters the score |
| Later/other-market outcomes | No 2015–2025 or US mid-cap/EU outcomes used for selection |
| Candidate/selected count | 153 verified published JKP candidates, 152 eligible, 50 selected |
| A/T/S | 13 published anchors, 13 minimum-theme selections, 24 score/redundancy selections |
| Groups | 13 published JKP groups; all 50 labels and directions published |
| Missing raw observations | Pairwise complete statistics; raw NaNs preserved; assumed-zero targets excluded |
| Matrices | Raw/ranked feature correlation and sample covariance; pair counts; separate return-augmented and per-stock diagnostics |
| Kati input update | New 50 installed in all three base-market and US-iteration configurations; raw source and ranked panels rebuilt |
| Existing ML implementation | Retained, including Fourier features and coefficient/portfolio-weight learning |
| Existing coefficients/results | Archived intact with their original feature set; not reused for the new 50 |
| New ML fits during this update | None |

Correlation or covariance is an in-period association, not causal evidence or guaranteed predictive performance. The selected set is preferred under the declared score and group rules, not proven globally optimal. Raw covariance magnitudes depend on units. Pairwise-complete covariance matrices may not be positive semidefinite and are not fitted portfolio risk matrices.

Published JKP directions orient every covariance theme exposure. There are no provisional groups. Empirical association signs are used only for ranking association strength. Future ML coefficients remain learned by the preserved implementation. Characteristic ranks themselves are not sign-flipped.

The selected 50 now include `dolvol_126d`. Its ranked value is a signal; trading costs continue to use the original unranked USD liquidity value separately. There are exactly 50 predictors.

Original memberships, return assumptions, liquidity infrastructure, annual expanding model schedule and monthly portfolio updates retain the Keats conventions. Missing held-stock forward returns retain the existing labelled zero-excess-return assumption when preparing model inputs; those assumed returns are not used in characteristic selection. Source-vintage, point-in-time and survivorship limitations remain disclosed; this change does not remove them.

Future model results require an explicit fresh fit. The retained base scope is US high-cap, US mid-cap and EU high-cap; iterations apply only to US high-cap Portfolio ML. Selecting names from US high-cap evidence does not establish their suitability for the other markets. Model evaluation remains a later 2020–2025 walk-forward exercise, with annual updates permitted to use already observed past test years.

See `outputs/us_high_cap_selection/selection_manifest.json`, `verification.json`, `METHODS_ASSUMPTIONS_AND_EXCLUSIONS.md` and `outputs/validation/us_high_cap_selection_migration.json` for the data/method/archival audit.
# Iteration 10: fixed-seed ensemble extension

Five RFF seeds (1-5) and equal 20% target-aim weights are fixed before fitting.
The original annual coefficient/hyperparameter learning, stock/feature/date
universes, risk estimator and actual cost formula are unchanged. Each member
has its own validation choices; all members share the same active-risk
coefficient within a fit. That coefficient is recalibrated from the ensemble's
pretest exposures using the original rule. Combined stock orders are executed
once and their quadratic costs reconstructed independently. Seed dispersion
does not imply alpha; savings versus separately traded members do not imply
savings versus iteration 4. This is an ensemble extension of the replication.
Rejected changes 5-9 are not inherited. See `ITERATION_10_SEED_ENSEMBLE.md`.

