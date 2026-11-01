# Keats results: four separate run groups

Open [index.html](index.html). The [availability CSV](availability_by_group_and_market.csv) is the current completion record.

```text
outputs/
  base/
    us_large/  us_mid/  eu_large/
    results/                 combined base-only statistics and charts
    analysis/                base comparison calculation code
  iteration_1/
    us_large/  us_mid/  eu_large/
  iteration_2/
    us_large/  us_mid/  eu_large/
  iteration_3/
    us_large/  us_mid/  eu_large/
  organisation_audit/        restoration hashes, tests and prior-state snapshots
```

- Base: Static ML and Portfolio ML only; two methods in each market's tables and charts.
- Iteration 1: original active benchmark-relative runs and capacity outputs. US high cap has saved experiment results; other markets have references but no active-run results.
- Iteration 2: original 250-stock prepared/reference outputs restored as they were. The active experiment was unfinished; preparation performance is labelled as reference/preparatory evidence.
- Iteration 3: code and group folders restored; no saved completed half-life experiment exists.

Native CSV, Parquet, PNG and PDF files remain in their original subfolders inside the matching run group. Each market's reports/ folder holds newly generated descriptive summaries and charts from those files. Missing output is never replaced by paper figures, zero results or another market's results.

No missing iteration was run in this organisation task. Keats evaluates 2020-2025 and has no separate unseen 2021-2025 holdout. Original experiment benchmark/cost conventions remain unchanged; new descriptive summaries identify their net-benchmark convention explicitly.

Scripts are grouped in [../scripts/](../scripts/README.md), with a complete [script CSV map](../scripts/SCRIPT_MAP.csv). The configs use the same four group names for every market.
