# Current update status — 2 November 2026

The new US high-cap-only correlation/covariance selection uses the 2005–2014 measurement window. It fits no ML model.

- 153 verified published JKP candidates; 152 meet coverage/month-count gates; 50 selected.
- 13 economic groups; 13 A, 13 T and 24 S selection reasons.
- All 50 have published JKP group labels and directions; no provisional groups.
- Interactive reports, correlation/covariance matrices, return associations and per-stock diagnostics are under `outputs/us_high_cap_selection/`.
- Kati configurations and input panels now use the new 50. Existing ML mathematics and weight-learning implementation are retained.
- Old characteristic archives, fits, weights, caches and reports have been removed at the user's request.
- No new stock-return or portfolio ML fit is run for this update. New-feature performance requires a fresh model run.

Current verification is recorded in `outputs/verification/build_verification.json`. The JKP-only membership guard has an independent test, including rejection of candidates absent from either reference. Raw correlation and covariance were independently checked against pandas. No model fitting is part of verification.

Current selection checks are in `outputs/us_high_cap_selection/verification.json`; input/notebook checks are in `outputs/verification/`; identical two-project outputs and independent matrix checks are recorded in `outputs/validation/jkp_only_double_check.json`. The current installation audit is `outputs/validation/us_high_cap_selection_migration.json`.
