# Current assumptions, exclusions and changes

Performance reporting now uses additive cumulative net excess returns, matching the authors' `cumsum(r-tc)` charts. Drawdown is the additive path minus its running peak including initial zero, and cumulative performance versus Market is a difference of additive paths. Both are displayed in percentage points. This drawdown is a reporting extension, not a verified paper drawdown definition or a percent loss of investor capital. Geometric CAGR/Calmar reporting has been removed. Model fitting, saved monthly returns/costs/weights, annual means, Sharpe, utility, P and iteration acceptance are unchanged. Internal wealth and drift accounting retain their original compounding. See [ADDITIVE_REPORTING.md](ADDITIVE_REPORTING.md).

The active US high-cap experiment sequence is [the five-step protocol](FIVE_ITERATION_PROTOCOL.md). It replaces the former three-iteration breadth/capacity sequence. Acceptance uses only the unchanged 2020–2025 test period's paired-bootstrap P(better net Sharpe) against the previous accepted version, threshold 80%. Annual refitting is already in Base. Rejected candidates remain reported; only accepted changes are inherited. Drawdown adds no gate. Test-selected final performance is not independent unseen confirmation. All regional base calculations and characteristics remain unchanged.

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
