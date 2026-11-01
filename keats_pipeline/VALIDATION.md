# Four-group organisation validation ? 1 November 2026

The current structure is base plus three preserved iterations, for each of the three markets. The current [availability CSV](outputs/availability_by_group_and_market.csv) and [result directory](outputs/index.html) supersede the earlier three-strategy cleanup records.

101 automated tests passed. The grouped organisation audit passed 205 checks; the base-only accounting/preservation audit passed 72 checks.

## Verified scope

- Base contains only original Static ML and default Portfolio ML: six series, 72 test months each (2020-2025).
- All base tables, regime reports, saved-output comparisons and charts were refreshed to show only those two methods.
- Original iteration mathematical implementations, 1/2/4% TE targets, 250-stock rules and 3/5/10-year/no-decay half-life options were restored. Script paths, group guards and reference-file routing were updated for separate folders.
- Every restored native iteration result/chart is checked against its saved source hash; numerical iteration modules are checked against the archived original code.
- Original prepared panels, covariance, forecasts, wealth, all 50 features and base portfolio return paths are preserved.
- No missing iteration was run. US high-cap iteration 1 has saved experiment results; iteration 2 has preparatory/reference outputs, and remaining missing experiments remain labelled unfinished/not run.
- Eight grouped CLI/plan entry-point smoke checks passed without fitting experiments. Dependency plans include preceding groups where required; --list-jobs does not execute them.

## Records

- [Organisation verification](outputs/organisation_audit/organisation_verification.json) and [detailed checks](outputs/organisation_audit/organisation_checks.csv).
- [Base independent accounting checks](outputs/base/results/independent_verification.csv) and [base completion](outputs/base/results/completion.json).
- [Latest test log](outputs/organisation_audit/pytest_results.txt), including the restored active/Portfolio-ML numerical tests and half-life regression tests.
- [Script map](scripts/SCRIPT_MAP.csv) and [script guide](scripts/README.md).

The tests use a unique workspace scratch directory on restricted Windows. Early attempts that could not use the system Temp directory are retained as diagnostic logs; the latest test log is authoritative.

Static ML's annual return-model refits and Portfolio ML's expanding annual fits are preserved; holdings update monthly. The nominal training/validation/test split and estimator-specific usable history are explained in outputs/base/TRAIN_TEST_AND_OOS.md. No separate unseen 2021-2025 holdout is claimed for Keats.

European base Static ML retains extreme modelled costs and invalid geometric growth months, with feasibility flags. Native iteration benchmark conventions are preserved as they were: the original half-life comparison uses a zero-cost paper-index benchmark, while other saved reports can use the traded Market benchmark. New descriptive summaries label their own benchmark convention and do not silently change original experiment decisions.
