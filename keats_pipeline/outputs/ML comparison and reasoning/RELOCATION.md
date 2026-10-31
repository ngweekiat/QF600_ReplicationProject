# Comparison code and analysis relocation

Moved at the user's request on 2026-11-01 (Asia/Singapore):

- `output/ML comparison and reasoning/` -> `keats_pipeline/outputs/ML comparison and reasoning/` (all reports, charts, tables and development/validation history).
- `analysis/ml_comparison/` -> `keats_pipeline/outputs/ML comparison and reasoning/analysis/ml_comparison/` (all evaluation code and tests).

Run `run_comparison.py` in this folder. Commands are in `analysis/ml_comparison/README.md`. The latest completed research report remains `run_20261101_011701_SGT/`.

Only path discovery, import bootstrapping, output destination, documentation and validation plumbing changed. The original ML implementations, configurations, saved predictions, weights, covariance estimates and portfolio results were not modified or retrained. Numeric comparison tables and chart data are retained.

Historical run settings, logs, source inventories and code hashes retain paths as recorded when those runs were made. This mapping explains those paths; they are not rewritten as though the older runs occurred here. Current validation results and code hashes are in `relocation_validation/`.

Archived mock Python files containing intentionally non-Python bytes were renamed from `.py` to `.py.fixture`, preserving their contents. The rename ledger is `relocation_validation/archived_fixture_renames.json`. The snapshot test now uses valid comment bytes so Keats' existing recursive Python syntax check remains meaningful. Comparison artifacts are excluded from the original-model source snapshot so new reports do not compare against themselves.

The completed run `run_20261101_011701_SGT/` is now named `results/` to keep report/chart paths below Windows legacy length limits. New timestamped runs use shorter `rYYYYMMDD_HHMMSS` folder names. All numerical result and chart files are preserved.
