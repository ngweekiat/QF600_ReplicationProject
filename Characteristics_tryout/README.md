# Characteristics_tryout

## Current selection: US high-cap only, 2005–2014

Open [the current report and charts](outputs/us_high_cap_selection/index.html). The current `data/recommended_50.json` is selected from 153 verified JKP candidates, each present in WRDS Global Factor Data and both published JKP reference files. Only US high-cap observations from 2005–2014 are used. Of these, 152 pass eligibility checks; 50 are selected using association ranking, coverage, stability, persistence, quotas and redundancy penalties. There are 13 A, 13 T and 24 S selections across 13 groups. All 50 group labels and directions are published; none is provisional.

Run `python -X utf8 Characteristics_tryout/run.py select-us-high-cap` to reproduce it. `screen`, `analyse`, `finalize` and `explain-selection` use this US-only analysis. Legacy multi-market selection is disabled, and old selection archives were deleted at the user's request. `download_200.json` describes the source acquisition cache; the current US-only 200 cap is `data/us_high_cap_selection/candidate_200.csv`.

Reports include raw/ranked correlation and covariance matrices, return-augmented matrices, observed-pair counts, per-stock associations, charts, score/exclusion tables and source hashes. Current pointers are in `data/current_selection.json`. Kati uses this new 50; the ML implementation is preserved. The [JKP-only decision chart](outputs/us_high_cap_selection/charts/selection_decision_path_jkp_only.pdf) shows 153 verified candidates → 152 eligible → 50 selected and each inclusion reason. `candidate_200.csv` retains its historical filename but now contains 152 verified eligible candidates.

Independent experimental code copied from the characteristics study archived in Downloads/characteristcs. It lives outside keats_pipeline and does not activate a new feature set in its ML models.

## Historical study background (superseded for current calculations)

- Reads the existing US high-cap, US mid-cap and European high-cap Keats training panels and memberships.
- Screens source characteristics using return years 2005-2014, caps the full download at 200, and proposes 50 shared features using 2015-2019 validation.
- Calculates feature/return covariance, Pearson and rank correlation, cross-feature redundancy, coverage, persistence and annual one-feature walk-forward diagnostics.
- Keeps test returns from 2020 onward out of feature selection. Raw missing values remain missing; separate ranked exports use the stated 0.5 missing-signal convention.
- Requires a published JKP representative in each theme and keeps the original score, coverage rules and greedy selection method. The archived study screened 404 candidates; the live source schema is inspected during explicit acquisition.

The source study's proposed 50 are a research recommendation, not an activated replacement for Keats' original 50. Covariance/association is not proof of economic utility or a globally optimal predictor set.

## Folder layout

```text
Characteristics_tryout/
  run.py                 command entry point
  characteristics/       original calculation modules, relocated paths and tests
  data/                  200-characteristic dataset (Git LFS)
  outputs/               generated reports and validation logs (Git-ignored)
```

The dataset is included under data using Git LFS. After cloning, run `git lfs pull --include="Characteristics_tryout/data/**"`. See data/README.md for scope and checks. The original Downloads archive remains intact; generated study reports remain there.

## Run from the repository root

Use the project's existing Python 3.11-3.13 dependencies (including numpy, pandas, scipy, pyarrow, matplotlib and WRDS); pytest is required for tests. If needed, install the root project with `python -m pip install -e ".[dev]"`.

```powershell
$characteristicPython = 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe'
# Display paths without reading data or connecting to WRDS:
& $characteristicPython Characteristics_tryout/run.py paths

# Explicit network acquisition; this can download several GB:
& $characteristicPython Characteristics_tryout/run.py acquire

# Offline analysis after data is available:
& $characteristicPython Characteristics_tryout/run.py screen
& $characteristicPython Characteristics_tryout/run.py analyse
& $characteristicPython Characteristics_tryout/run.py finalize
& $characteristicPython -m pytest Characteristics_tryout/characteristics/tests -q
```

`acquire-lags` supplements the original Portfolio ML lag-history keys. `prepare` builds local metadata and requires source_schema.csv from acquisition or an existing cache. `--wait` lets analysis consume explicitly acquired partitions as they arrive. Acquisition resumes compatible cached files and uses the frozen 200-feature list when it already exists.

To reuse the downloaded dataset without copying it back into this project:

```powershell
& $characteristicPython Characteristics_tryout/run.py paths --data-dir 'C:\Users\katiravan\Downloads\characteristcs\keats_pipeline\data\characteristics'
& $characteristicPython Characteristics_tryout/run.py analyse --data-dir 'C:\Users\katiravan\Downloads\characteristcs\keats_pipeline\data\characteristics'
```

Pass the same --data-dir on every stage. --output-dir can select a separate report destination. Environment equivalents are CHARACTERISTICS_DATA_DIR and CHARACTERISTICS_OUTPUT_DIR. The default data and output folders are under Characteristics_tryout, independent of the terminal working directory. Analysis can create ranked exports inside the selected cache; the upload itself does not run that analysis or modify the Downloads archive.

## Selection explanation charts

Open `outputs/selection_explanation/index.html` for a searchable guide to the frozen suggested 50, or use `selection_overview.png` for presentations. `selected_50_evidence.png` shows each input's validation evidence across the three markets. Both charts also have PDF and SVG versions. The guide distinguishes published group labels from provisional correlation-based assignments.

Rebuild from the included source-table snapshots with:

```powershell
& $characteristicPython Characteristics_tryout/run.py explain-selection
```

Use `--analysis-dir <folder>` to read another completed analysis archive. This stage explains the existing recommendation without selecting new inputs or refitting Keats models.

## Input dependencies and credentials

The code reads the existing keats_pipeline/data/processed/<market>/with_zero_returns/panel.parquet files, Keats' original feature list, original supplemental lag keys, and the included data/reference/{cluster_labels,factor_details}.parquet metadata (with repository data/raw as fallback). These input datasets must exist locally; copying the code is not a substitute for acquiring/preparing them.

Explicit acquisition reuses keats_pipeline/acquire_missing.py's local credential loader. Credentials are read into memory, never copied into this folder or committed. The normal help, paths and offline stages do not connect to WRDS. Fresh acquisition fetches missing source column metadata before preparation.

## Relocation changes

The statistical/selection modules are retained from the Downloads source. Changes are limited to the standalone entry point, isolated default paths, optional cache/report path overrides, relocated report links, first-run source-schema bootstrap, and relocation tests. SOURCE_PROVENANCE.json records the copied source hashes. No changes are made inside keats_pipeline by this upload; no model is retrained or feature configuration switched.
