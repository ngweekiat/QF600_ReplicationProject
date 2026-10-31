# Validation of the new workflow

Verified locally on 2026-10-31 using the installed Python 3.13 interpreter.

## Completed

- All five saved Keats Parquet schemas were inspected. All are monthly stock panels; none stores daily stock-return observations.
- The 7z archive was listed without extracting or replacing files. It contains the same monthly datasets.
- Existing monthly/daily risk-free rates and JKP characteristic metadata were reused. Additional source data is now included in `data/inputs/` following explicit acquisition.
- All three base universes were prepared, with all 50 characteristics.
- All 50 member feature ranks were compared against the saved Keats ranks, and every saved in-sample member row was retained.
- The original script/source hashes still match `original_manifest.json`.
- 37 workflow tests passed after full-run fixes. An additional 47 original portfolio/static/accounting/trading-speed tests passed against the isolated package.
- The small model checks fit 2015 forecasts twice, altering future realised returns in the second fit. Forecasts were identical for all three universes.
- 10 existing trading-speed/accounting tests also passed against the isolated `jkmp` snapshot.
- Stage 02 was invoked for every universe and stopped on missing daily inputs without network access.
- Stage 15 reports no separate unseen holdout; the original report's 2021–2025 result is not reused.

## Prepared data

The prepared panel includes the broad screened region for training, not just portfolio holdings.
The last formation month is November 2025, which earns the December 2025 return.

| Universe | Prepared regional rows | Saved member rows through November 2025 |
|---|---:|---:|
| US large-cap | 1,615,718 | 30,353 |
| US mid-cap | 1,615,718 | 29,215 |
| European large-cap | 1,270,311 | 30,530 |

## Acquired supplementary inputs

- US: 8,760,119 daily total returns for 2,028 required securities, from CRSP CIZ.
- Europe: 7,811,683 daily USD total returns for 1,769 required securities, from Compustat Global prices/factors, FX and issue status, with explicit JKP identifier mappings.
- European reconciliation: 3,647 sampled stock/months; median absolute difference approximately 2.08e-16; 99.8629% within 0.001 (10 basis points).
  Five differences exceed 10 basis points; the largest is 0.003704 (37.04 basis points). All comparison rows are supplied in `eu_monthly_reconciliation.parquet` for review.
- All three FRED series: USREC, VIXCLS and FEDFUNDS.
- Targeted JKP source observations repair all three previously missing current-month returns. Existing observed values are unchanged.
- CRSP terminal-month records, European issue status and FX observations are included as source evidence.
- New prepared datasets use `with_supplements/`; the first `base/` datasets remain intact.

## Still unavailable — no full backtest claimed

The user has now selected zero excess returns for missing held-stock next-month returns.
New `with_zero_returns/` prepared panels contain 56 flagged assumptions across the full saved history
(15 US large-cap, 21 US mid-cap, 20 Europe). The 14 gaps below were the later-history subset.
All originally observed returns are unchanged; the source inputs and earlier prepared datasets are retained.
`ret_ld1_original` and `ret_ld1_imputed_zero` preserve the distinction between observed and assumed values.
Total returns on these assumed observations equal the next-month risk-free rate. These assumptions affect
downstream fitting and performance and should be disclosed with any results.
The missing-forward-return stop is removed by this explicit imputation; a full covariance/backtest run has not been validated here.

Each script now reports all actual zero-imputed input observations by company label, stable stock ID,
formation date, next-month return date, original missing value and assumed value. A stage-specific CSV is
saved under the experiment output's `imputation_audit/` folder. Direct script execution and `run.py`
were both checked. Covariance's disclosure explicitly distinguishes these forward-return assumptions
from its historical daily-return risk inputs.

- Some held-stock forward returns: US large-cap has 3 missing values, US mid-cap 5, and Europe 6. WRDS also reports the JKP values as missing.
  Exact ids and months are in `data/inputs/unresolved_forward_returns.csv`. No zero or invented return has been inserted.
- A separate unseen holdout after the saved 2020–2025 test period.

Full covariance, full return-model fits, Portfolio-ML, wealth grids, and experiment results have not been generated.
The 250-stock selection is implemented and its mid-cap band/gap handling is tested; its full prepared datasets and performance runs are created on demand.
The ten original numerical tests and the small new model fits are validation checks, not research results.

## Full-run update (31 October 2026)

The older setup-only statements above describe the initial validation snapshot. The full workflow is now
being executed using `run_all.py --scope full`, with 114 dependency-ordered jobs and per-stage logs under
`outputs/run_logs/`. Current state JSON files are authoritative for completion/failure status.
All three base covariance and full return-model stages have run; European baseline Portfolio-ML, linear
Portfolio-ML, benchmarks, Static-ML and tables completed before the research grids began.

The first full pass exposed missing pre-entry signal history. WRDS supplied 930 US and 441 European
actual rows; original source panels/ranks are retained. Historical additions are lag-only observations,
not new targets or membership. Three unavailable December 2010 rows lead to user-selected exclusions
only at formation December 2011 for stock IDs 88818 and 91135 (250-stock US mid-cap) and 320230501
(250-stock Europe). Each output prints/saves those exclusions; base 80-stock membership is unchanged.

Covariance no longer requires unrelated current/forward monthly returns. Existing trained models can
provide forecasts for held stocks whose current realised return makes the row ineligible for fitting.
Static-ML uses equivalent batched monthly calculations, verified against the original with entries/exits
and both exit-cost settings. Rebuilt moment caches are archived, including metadata, rather than deleted.
Long grids, feature importance and active/capacity experiments are not described as finished until their
individual checkpoint reports completion. No old paper performance is reused as a Keats result.

## Rerun the checks

From the project root, with the installed Python 3.13 interpreter:

```powershell
& 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe' -m pytest keats_pipeline/tests -q
& 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe' keats_pipeline/run.py validate
```

See `README.md` for stage commands, input schemas and methodological differences from the original experiment.
