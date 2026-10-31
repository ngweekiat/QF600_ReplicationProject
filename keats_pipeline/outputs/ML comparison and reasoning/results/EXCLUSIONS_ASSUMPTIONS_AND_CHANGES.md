# Exclusions, assumptions and changes

Run timestamp: 2026-11-01T01:17:01.103262+08:00 (Asia/Singapore).

## Scope and changes made for this task

- Added only independent evaluation code under `analysis/ml_comparison/` and this new output directory.
- No existing ML implementation, configuration, prediction, portfolio, covariance estimate or source input was edited or fitted by the evaluation scripts.
- The previously authorised full-run process tree (PID 56932 and child 92028) was stopped before this analysis to prevent additional fitting. Its already saved partial outputs are retained. Status logs can still say running because stopping a process does not finalise its state file.
- Six pre-existing modified files were present before this task: Keats README, apply_history_policy.py, US-mid base.yaml, scripts/13_breadth.py, portfolio/recursion.py and test_execution_safety.py. Those are prior work, not comparison changes. Their current bytes are checked before/after this evaluation.
- The user's latest output location (`output/ML comparison and reasoning`) overrides the pasted brief's `outputs/ml_comparison` location. Every generated artifact, including development checks and final run validation, is under this single location.

## Sample and exclusions

- Primary sample is the declared test return window January 2020–December 2025. Models' saved validation returns are not included in headline test rankings.
- Within each dataset, every strategy uses identical observed return months; cross-dataset comparisons use the common overlap. No missing return month is filled. `tables/date_alignment_exclusions.csv` lists every date excluded by alignment.
- Only available saved strategies are compared. Grid/capacity and feature-permutation experiments are kept in separate diagnostics and are never counted as additional independently trained models.
- Unavailable strategies/variants and absent weights are marked SKIPPED, rather than inferred from other universes. UK-only and independently changed feature-set datasets are not available.
- Primary universe labels preserve base (about 80 names), US-large 250-stock, and any saved active-TE variants. No pooling of stocks, returns, currencies or incompatible mandates occurs.

## Existing upstream assumptions retained, not newly introduced

- Missing held-stock next-month excess returns were set to zero by prior user instruction. The previously reported full-history counts are 15 US-large, 21 US-mid, 20 European observations; selected 250-stock memberships can have different affected rows. The audit copies actual flags for every evaluated prepared panel, including original missing values, dates and IDs. These are assumptions, not observed zero returns; total returns equal the next-month USD risk-free rate.
- Unavailable lagged history caused specific stock-month exclusions in previously prepared 250-stock panels. Saved exclusion logs identify IDs, missing lag months and dates. Existing feature-missing values retain the pipeline's neutral 0.5 rank convention.
- The actual saved application ledgers contain two US-mid 250-stock exclusions. A third source history gap concerns the unprepared European 250-stock dataset; it is documented separately as pending, rather than counted as an applied exclusion. The audit preserves both ledgers and preparation-coverage status.
- Saved regional membership and training populations are used without rebuilding universes. US universes include NYSE and Nasdaq; European large-cap includes multiple countries, not UK only.
- Daily European returns are USD-adjusted under existing source construction; venue volume, listing selection and delisting treatment are not independently established by an accounting identity. Missing terminal returns and surviving data gaps limit survivorship claims.
- All portfolios use existing market-linked wealth paths with the configured $10B anchor at end-2019; covariance, feature ranks, lagged signals, liquidity coefficients and modelled impact are reused exactly.

## Metric, inference and scenario assumptions

- Paper Table 2: annual mean/cost/utility ×12, gross volatility ×sqrt(12), sample variance ddof=1, paper net Sharpe uses GROSS SD, monthly two-sided turnover (no half factor). Main evaluation gamma=10, even for active models with a distinct calibrated gamma_TE objective.
- Table 3 paper utility-flow probability uses the exact normal approximation from authors' R code. It is labelled a posterior-style probability, not a frequentist p-value. HAC tests and block bootstraps are separate evaluation extensions. The population- versus sample-variance finite-sample difference is explicit.
- Paired circular moving-block bootstrap: 2000 replications, fixed seed 20261101, primary block length 6 months, sensitivity lengths 3 and 12. Bootstrap resamples all paired values jointly and re-estimates means/variances.
- Bartlett Newey–West lag choice floor(4*(T/100)^(2/9)), capped T−1; normal-reference two-sided inference. Holm correction applies within each dataset separately for utility and Sharpe pair families. It does not correct undisclosed prior research trials or every exploratory scenario.
- Re-costing 0/0.5/1/2/3 times existing TC, fixed-weight AUM scaling, and evaluation gamma 1/5/10/20/100 do not re-optimise or retune models. Actual saved optimised grid points are labelled separately and incomplete grids remain incomplete.
- Raw dollar volume is used for liquidity diagnostics when available; rank-normalised features are not treated as dollar amounts. Formation months determine trade costs; realised next-month returns determine performance.
- Synthetic compounded excess-return indices are not total investor wealth. Any net month <=−100% invalidates full-sample geometric metrics; paths stop at that month. Paper arithmetic statistics retain all such observations with validity flags. No winsorisation or selective removal is applied.
- Regimes are descriptive. Local market up/down uses the contemporaneous saved Market return; VIX is prior-month mean with fixed threshold 20; USREC is current-vintage US context, including for Europe. Calendar stress windows and rolling statistics are not trading signals.
- Modelled quadratic impact can exceed investor wealth when leverage/turnover is unbounded or venue volume is tiny. Saved portfolio rankings in such cases are algebraic diagnostics, not feasible executed-investment claims. No external liquidity/participation/borrowing constraint is imposed after the fact.

## Unavailable tests

- Independent fresh holdout after 2025; multi-horizon 2–12 month predictions; independently refitted model random-seed return paths; unrun gamma/AUM grid points; alternative covariance estimators not already saved; externally observed execution costs and constraint-aware re-optimisation.
- Look-ahead checks inspect saved prediction/selection/covariance dates and source logic. No future-return perturbation refit is performed under the no-retraining instruction; historical point-in-time data vintages cannot be proven from final saved values alone.
- See `audit/`, `robustness/skipped_tests.csv`, and `tables/missing_saved_outputs.csv` for dataset-specific detail and all retained exceptions.

## Verification

- Each paper metric is independently recomputed from arrays and matched to reused project functions; accounting is independently reconstructed from available weights, panel returns, wealth, liquidity and inherited weights.
- Source SHA256 inventories before and after evaluate saved code/config/results; evaluated PF/weight files and data provenance are recorded. No model runner is invoked.
- Automated tests exercise posterior direction, paired bootstrap, HAC arithmetic, formula annualisation, turnover/exit semantics, geometric invalidity, common dates, missing values and preservation. Test output is in `validation/`.
