# Current update status — 2 November 2026

The new US high-cap-only correlation/covariance selection uses the 2005–2014 measurement window. It fits no ML model.

- 404 candidates; 397 meet the coverage/month-count gates; 200 cap; 50 selected.
- 13 economic groups; 13 A, 13 T and 24 S selection reasons.
- 32 published group labels and 18 provisional US-correlation labels.
- Interactive reports, correlation/covariance matrices, return associations and per-stock diagnostics are under `outputs/us_high_cap_selection/`.
- Kati configurations and input panels now use the new 50. Existing ML mathematics and weight-learning implementation are retained.
- Old characteristic archives, fits, weights, caches and reports have been removed at the user's request.
- No new stock-return or portfolio ML fit is run for this update. New-feature performance requires a fresh model run.

Final verification: 71 combined tests passed; 251 input/source/notebook checks passed; all 97 notebook code cells executed successfully with model fitting disabled. Both projects reproduce the identical ordered 50 and all 24 matrix exports. The actual raw correlation/covariance matrices also match an independent pandas calculation. A/T/S filters, feature search and correlation/covariance chart/download toggles passed their logic checks.

Current machine-readable selection checks are in `outputs/us_high_cap_selection/verification.json`; input/notebook checks are in `outputs/verification/`; archive deletion is recorded in `outputs/validation/archive_removal.json`. Read the current reports for current check counts.
