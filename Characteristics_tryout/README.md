# Characteristics_tryout

Independent experimental code copied from the characteristics study archived in Downloads/characteristcs. It lives outside keats_pipeline and does not activate a new feature set in its ML models.

## What it does

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
  data/                  optional local cache (Git-ignored)
  outputs/               generated reports and validation logs (Git-ignored)
```

This Git upload contains code, tests and documentation only. The downloaded data and existing study results remain in Downloads/characteristcs/keats_pipeline/data/characteristics and its corresponding outputs folder.

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

## Input dependencies and credentials

The code reads the existing keats_pipeline/data/processed/<market>/with_zero_returns/panel.parquet files, Keats' original feature list, original supplemental lag keys, and the repository's data/raw/{cluster_labels,factor_details}.parquet metadata. These input datasets must exist locally; copying the code is not a substitute for acquiring/preparing them.

Explicit acquisition reuses keats_pipeline/acquire_missing.py's local credential loader. Credentials are read into memory, never copied into this folder or committed. The normal help, paths and offline stages do not connect to WRDS. Fresh acquisition fetches missing source column metadata before preparation.

## Relocation changes

The statistical/selection modules are retained from the Downloads source. Changes are limited to the standalone entry point, isolated default paths, optional cache/report path overrides, relocated report links, first-run source-schema bootstrap, and relocation tests. SOURCE_PROVENANCE.json records the copied source hashes. No changes are made inside keats_pipeline by this upload; no model is retrained or feature configuration switched.
