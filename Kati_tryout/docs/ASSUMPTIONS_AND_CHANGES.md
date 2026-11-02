# Current assumptions, exclusions and changes

The current methodology is documented in [US_HIGH_CAP_SELECTION.md](US_HIGH_CAP_SELECTION.md). It replaces the earlier three-market feature-selection method. Old characteristic archives and their fits were deleted at the user's request; active calculations use the new US high-cap-selected 50.

| Item | Current behavior |
|---|---|
| Characteristic selection | US high-cap held-stock observations only, 2005–2014 |
| Selection method | Observed correlation/covariance diagnostics with fixed association score, group quotas and redundancy penalties |
| ML training during selection | None; no single-feature regression R², return forecast or portfolio utility enters the score |
| Later/other-market outcomes | No 2015–2025 or US mid-cap/EU outcomes used for selection |
| Candidate/selected count | 404 candidates, 397 eligible, 200 cap, 50 selected |
| A/T/S | 13 published anchors, 13 minimum-theme selections, 24 score/redundancy selections |
| Groups | 13 groups; 32 published labels and 18 provisional US-correlation assignments |
| Missing raw observations | Pairwise complete statistics; raw NaNs preserved; assumed-zero targets excluded |
| Matrices | Raw/ranked feature correlation and sample covariance; pair counts; separate return-augmented and per-stock diagnostics |
| Kati input update | New 50 installed in all three base-market and US-iteration configurations; raw source and ranked panels rebuilt |
| Existing ML implementation | Retained, including Fourier features and coefficient/portfolio-weight learning |
| Existing coefficients/results | Archived intact with their original feature set; not reused for the new 50 |
| New ML fits during this update | None |

Correlation or covariance is an in-period association, not causal evidence or guaranteed predictive performance. The selected set is preferred under the declared score and group rules, not proven globally optimal. Raw covariance magnitudes depend on units. Pairwise-complete covariance matrices may not be positive semidefinite and are not fitted portfolio risk matrices.

Published theme directions are retained for published members; provisional groups use their empirical 2005–2014 US high-cap sign. These signs orient covariance theme exposures; future ML coefficients remain learned by the preserved ML implementation. Characteristic ranks themselves are not sign-flipped.

The selected 50 now include `dolvol_126d`. Its ranked value is a signal; trading costs continue to use the original unranked USD liquidity value separately. There are exactly 50 predictors.

Original memberships, return assumptions, liquidity infrastructure, annual expanding model schedule and monthly portfolio updates retain the Keats conventions. Missing held-stock forward returns retain the existing labelled zero-excess-return assumption when preparing model inputs; those assumed returns are not used in characteristic selection. Source-vintage, point-in-time and survivorship limitations remain disclosed; this change does not remove them.

Future model results require an explicit fresh fit. The retained base scope is US high-cap, US mid-cap and EU high-cap; iterations apply only to US high-cap Portfolio ML. Selecting names from US high-cap evidence does not establish their suitability for the other markets. Model evaluation remains a later 2020–2025 walk-forward exercise, with annual updates permitted to use already observed past test years.

See `outputs/us_high_cap_selection/selection_manifest.json`, `verification.json`, `METHODS_ASSUMPTIONS_AND_EXCLUSIONS.md` and `outputs/validation/us_high_cap_selection_migration.json` for the data/method/archival audit.
