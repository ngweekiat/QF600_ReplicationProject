# Supplementary source data

These inputs were acquired on 2026-10-31 for the existing Keats universes. Large binary files use Git LFS.
The original Keats files have not been replaced.

| File | Source / meaning |
|---|---|
| `daily_US.parquet` | CRSP CIZ (`crsp.dsf_v2.dlyret`), required US securities only; USD decimal total returns |
| `daily_EU.parquet` | Compustat Global prices, total-return factors, quotation units, split adjustments, FX and issue status; normalized to JKP security ids and USD total returns |
| `eu_security_mapping.parquet` | Explicit WRDS JKP id → Compustat gvkey/iid mapping; identifiers were not guessed |
| `monthly_returns_US.parquet`, `monthly_returns_EU.parquet` | Exact JKP observations for the missing stock/months and adjacent months; source missing values are preserved |
| `us_terminal_months.parquet` | CRSP monthly CIZ observations for the US securities with missing JKP forward returns |
| `fred_USREC.parquet` | FRED US recession indicator |
| `fred_VIXCLS.parquet` | FRED CBOE VIX observations |
| `fred_FEDFUNDS.parquet` | FRED effective federal funds rate |
| `fx_GBP_crosses.parquet` | Actual Compustat FX quotes used in the European USD conversion |
| `eu_delisting_metadata.parquet` | Source issue status and delisting reason used for European inactive issues |
| `eu_monthly_reconciliation.parquet` | All sampled monthly comparisons, including the five differences above 10 basis points |
| `unresolved_forward_returns.csv` | Exact remaining source gaps; no assumed zero returns |
| `lag_history_source_US.parquet`, `lag_history_source_EU.parquet` | 930 / 441 actual WRDS historical rows absent from the saved screened panels; no forward targets requested |
| `lag_history_US.parquet`, `lag_history_EU.parquet` | Ranked historical signal-only inputs; original ranks and membership are not modified |
| `unavailable_lag_history_US.csv`, `unavailable_lag_history_EU.csv` | Three unavailable December 2010 stock/months; user selected explicit affected-month exclusions |

Each source dataset has a JSON provenance file. Daily data uses `id,date,ret`; `ret` is total return,
not excess return or a price change. No return after December 2025 enters the normalized daily datasets.

European construction follows the price, FX and inactive-issue rules in the
[JKP source code](https://github.com/bkelly-lab/ReplicationCrisis/blob/master/GlobalFactors/project_macros.sas).
`eu_monthly_validation.json` records the comparison between compounded daily USD returns and sampled WRDS monthly total returns.

The JKP supplements repair known current-month gaps where the source reports a value. They do not replace existing observations.
Some terminal-security forward returns are absent from the source as well; these remain missing rather than being assumed zero.

For a fresh checkout, run `git lfs pull` before reading the Parquet files. The normal `run.py` workflow remains offline.
`acquire_missing.py` is an explicit acquisition utility and requires WRDS access for additional downloads; credentials are never saved here.
