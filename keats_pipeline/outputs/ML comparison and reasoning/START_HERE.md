# Keats ML comparison and reasoning

The completed evaluation is in **results**. Older `development_*`, `run_*` and short `t_*` directories are validation history and test support files retained for traceability.

- [Open the full report](results/index.html)
- [Read the research summary](results/RESEARCH_SUMMARY.md)
- [Exclusions, assumptions and changes](results/EXCLUSIONS_ASSUMPTIONS_AND_CHANGES.md)
- [Performance tables](results/tables/table2_consolidated.csv)
- [Statistical comparisons](results/tables/table3_pairwise_tests.csv)
- [Automated test results](results/validation/automated_tests.log)
- [Final delivery checks](results/validation/final_delivery_checks.json)
- [Commands and evaluation-code documentation](analysis/ml_comparison/README.md)

The evaluation covers 38 saved strategy/universe series, 166 paired comparisons, and 267 saved optimisation points. All 79 automated tests passed; the audit reports 333 passes and 8 explicit skips. SHA256 checks confirm 734 existing code/config/result files were unchanged. No model was retrained and no missing portfolio result was fabricated.

Some saved portfolios have extreme modelled costs and losses exceeding funded equity. The report retains and flags them; exact accounting reproduction does not establish economic feasibility. The original full Keats workflow remains incomplete, but all feasible saved-output comparisons requested here have been executed. Missing tests and unfinished upstream runs are explicitly documented.

Comparison code and analysis are now together inside Keats outputs. See [relocation notes](RELOCATION.md) and the [runner](run_comparison.py).
