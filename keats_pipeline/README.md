# Keats data workflow

This is a separate, offline copy of the project's script workflow and `jkmp` calculation package.
The original `scripts/`, `src/`, configs, downloaded Keats files and existing results are unchanged.
Do not rerun `setup_workflow.py` or `adapt_scripts.py`: those are the one-time creation utilities.

Saved-output ML comparisons, evaluation code, research reports and assumption/exclusion records are together in
[outputs/ML comparison and reasoning](<outputs/ML comparison and reasoning/START_HERE.md>).
Run `outputs/ML comparison and reasoning/run_comparison.py` with Python 3.13 to evaluate existing results;
this comparison runner does not train models or optimise new portfolios.

Supplementary source inputs are stored in `data/inputs/` with provenance and checksums and use Git LFS.
See [the input catalogue](data/inputs/README.md) and `data/inputs/manifest.json` for the files actually included.
Run `git lfs pull` after cloning. `run.py` remains offline; `acquire_missing.py` is a separate explicit downloader.

## Selected missing-return assumption

The user selected **zero excess return** for missing next-month returns on held stocks.
This applies to 56 saved-history observations: 15 US large-cap, 21 US mid-cap and 20 European large-cap.
The 14 previously reported gaps were the subset since December 2013.
Missing returns outside the held universe are not filled. Observed returns and source inputs are unchanged.
`ret_ld1_original` preserves the original value, `ret_ld1_imputed_zero` flags each assumption, and
`zero_return_imputations.csv` logs the affected stock/months. These are assumed values, not observed returns,
and enter downstream return-based fitting and performance calculations. Results may therefore be biased by this choice.
Zero excess return means the corresponding total return equals the next-month risk-free rate, not zero total return.
The new datasets are in `data/processed/<universe>/with_zero_returns/`; earlier prepared datasets are retained.

## Historical signal coverage and exclusions

A full run identified historical characteristic rows absent from the saved screened panels. Only the missing
stock/months were queried: WRDS returned 930 US and 441 European actual observations. These supplementary
rows are used solely for lagged signals, never as training targets, new members or wealth-market observations.
Their characteristics are ranked against the saved regional reference population plus the queried observation
with average ties; existing saved ranks are not changed. Missing features retain the existing 0.5 rank convention.
No neutral growth values were needed for these acquired rows.

The user selected exclusion instead of inventing unavailable history. IDs 88818 and 91135 (250-stock US mid-cap)
and 320230501 (250-stock Europe) are excluded only at formation 2011-12-31 because 2010-12-31 history is unavailable.
They may enter a subsequent month once all required history exists. No base 80-stock month is excluded.
Every affected output prints the exclusion table and saves a stage-specific `_history_exclusions.csv`.
The prepared `history_exclusions.csv` records the exact security, formation/return month, missing history month and reason.

Covariance now checks its historical daily inputs rather than unrelated monthly return targets.
When a current realised return makes a row unsuitable for fitting, held stocks can still receive forecasts
from their current characteristics and an already fitted model; the runner checks and records that coverage.
Static-ML grid calculations batch common monthly data and Cholesky factors; equivalence tests cover entry,
exit costs and the original validation output. Existing moment caches are archived under `previous_runs/`, not deleted.
Monthly portfolio recursion also indexes historical returns once rather than repeatedly merging the regional panel;
direct equivalence tests cover stock entries, exits, zero positions and both exit-cost settings.
Mid-cap rank-IC reports use ranks 461-540, 376-460 and 541-625 so they describe the actual central and added bands.
Breadth reports identify rank-IC diagnostics as reference-model diagnostics: held-specific imputation means the
two fitted models are not assumed to have identical predictions.

Every Keats stage that reads a prepared dataset prints its zero-return assumptions automatically, including
company labels where available, stable stock IDs, formation dates, affected next-month return dates, and assumed values.
The full list includes warmup observations and is labelled as an input-data audit, not a claim that every row
affects every calculation. Covariance explicitly states that it uses historical daily returns rather than these forward values.
Each stage also saves its list to `outputs/<universe>/<variant-output>/imputation_audit/<stage>.csv`.
This works through `run.py` and when running files directly from `keats_pipeline/scripts/` in VS Code.

## What is reused

The workflow reads `../configs/Keats universe code/data/universes Keats/` directly:

- `screened_US.parquet` supplies the broad US model-training and ranking population.
- `screened_EU.parquet` supplies the broad European population.
- `panel_us_large.parquet`, `panel_us_mid.parquet`, and `panel_eu_large.parquet` supply the exact saved 80-stock membership and feature ranks.
- All 50 characteristics are used, assigned to themes using the existing JKP metadata in `../data/raw/`.
- Existing monthly/daily USD risk-free rates and factor metadata are reused. Only small reference files are copied.
- Large datasets are not downloaded or duplicated into a new raw-data directory. Prepared datasets are generated separately.

The archive `universes.7z` was listed without extracting it. It contains the same monthly datasets, not daily stock returns.

## Run from the repository root

To execute the dependency-ordered workflow with persistent logs and resume completed stages:

```powershell
& 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe' keats_pipeline/run_all.py --scope core
# Add grids, feature importance, regimes, active/breadth/half-life experiments and monitoring:
& 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe' keats_pipeline/run_all.py --scope full
```

Use `--universe us_large`, `us_mid`, or `eu_large` to select one; the default is all three.
Stage logs and a JSON completion/failure record are saved under `outputs/run_logs/`.
The full profile has 114 jobs. It retains completed files, stops dependent jobs when their upstream stage fails,
and continues independent work. Rerun the same command to resume after resolving a failure.
Grids retain both random-feature and linear results; a summary file alone does not mark both variants complete.
Neither command downloads, commits, pushes, or deletes original files.

Use Python 3.11–3.13 with the project's dependencies. The installed Python 3.13 was used for verification.
On this machine, `python` defaults to Python 3.12 without pandas. In VS Code select the Python 3.13 interpreter, or invoke it explicitly:

```powershell
$keatsPython = 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe'
& $keatsPython keats_pipeline/run.py audit
& $keatsPython keats_pipeline/run.py prepare --universe all
& $keatsPython keats_pipeline/run.py validate
& $keatsPython -m pytest keats_pipeline/tests -q
```

All paths in configs are relative to `keats_pipeline/`, independent of the terminal working directory.
`--universe` accepts `us_large`, `us_mid`, `eu_large`, or `all`.
`--variant` accepts `base`, `linear_pfml`, `active_pfml`, `universe_250`, or `halflife`.
An existing prepared dataset is retained on reruns; it is not silently overwritten.

## Stage mapping

| Stage | Purpose | Command after `python keats_pipeline/run.py` |
|---|---|---|
| 00 / audit | Inspect existing files and missing inputs; never download | `audit` |
| 01 / prepare | Convert saved Keats monthly panels to pipeline inputs | `prepare --universe all` |
| 02 | Original daily-return factor covariance estimator | `02 --universe us_large` |
| 03 | Annually refit return forecasts | `03 --universe us_large` |
| 04 | Market, equal-weight, minimum-variance, Markowitz-ML and Factor-ML benchmarks | `04 --universe us_large` |
| 05 | Static-ML and tuned Static-ML | `05 --universe us_large` |
| 06 | Portfolio-ML inputs | `06 --universe us_large` |
| 07 | Portfolio-ML estimation and performance | `07 --universe us_large` |
| 08 | Risk/wealth grids and efficient frontiers | `08 --universe us_large` |
| 09 | Feature-theme importance | `09 --universe us_large` |
| 10 | Regime analysis, using cached FRED only | `10 --universe us_large` |
| 11 | Tables and descriptive paper comparison | `11 --universe us_large` |
| monitor | Monitor the chosen 250-stock active test run | `monitor --universe us_large` |
| 12 | Active calibration, runs, capacity and report | `12 --universe us_large --variant active_pfml calibrate` |
| 13 | Compare 80-stock and 250-stock runs | `13 --universe us_large --te 0.02` |
| 14 | Compare equal-history versus half-life runs | `14 --universe us_large` |
| 15 | Report that a separate unseen holdout is unavailable | `15 --universe all` |

Stages 04–14 require the supplied daily inputs. Source-missing held-stock forward returns now follow the explicit
zero-excess-return assumption above and remain flagged for review.
Stage 03 does not require covariance and can run once preparation is complete. Full model/grids runs can be lengthy;
the setup validation does not claim to have produced full experimental results.

For the linear Portfolio-ML comparison repeat stages 06 and 07 with `--variant linear_pfml`.
The linear run shares its prepared data and output root with the base run, as in the original workflow; model outputs use separate `raw` and `rff` subfolders.

For the breadth experiment:

1. `prepare --universe all --variant universe_250`
2. Repeat 02–05 with `--variant universe_250` to build that universe's risk, forecasts and benchmarks.
3. Run stage 12 `calibrate`, then `run` and `capacity`, separately for `active_pfml` and `universe_250`.
4. Run stage 13. It reads both completed experiments.

For active runs/capacity, `--te-target 0.01`, `0.02`, or `0.04` chooses the corresponding freshly calibrated value.
The default is 0.02. A missing calibration is an error. Original values such as 12.82 or 49.09 are not reused.
Stage 12 `report` needs all three calibrated-budget runs and the baseline Portfolio-ML output.

For the recent-history experiment, run stage 12 `run --variant halflife` after the 250-stock baseline exists,
then stage 14. It reuses the 250-stock calibration and data but writes to a separate `u250_hl` output directory.
The comparison uses a 2% target; it does not automatically assert that either version is superior.

## Data and timing decisions

- Keats' training period is 2005–2014, validation is 2015–2019, and test is 2020–2025, in **return months**.
- The warmup rows before 2005 are retained for risk and lag history. Return-model fitting excludes returns before January 2005.
- Annual return-model tuning uses a trailing five-year internal validation window within the available past observations.
  For the first 2015 forecasts, internal training is 2005–2009 and internal validation is 2010–2014.
  It refits on all available training observations through 2014 before forecasting 2015. Subsequent annual fits use only observations available before that forecast year.
- Portfolio hyperparameters are selected each December using past realised results, following the original expanding-window approach.
- All 50 saved Keats member ranks are checked against the full screened region. Average percentile ranks with missing values set to 0.5 match Keats; the original pipeline's special zero-to-zero rank convention is not applied.
- Current-month excess returns are recovered from the previous row's forward return only for consecutive months of the same security.
  Gaps are left missing. Held stocks with missing current returns in the scored history stop risk/model execution rather than assume zero drift.
- Forward total returns add the USD risk-free rate of the **next** month. Dollar volume stays in USD, not USD millions; market cap is used for relative weighting only.
- The wealth path uses formation-cap-weighted returns of the full screened **same-region** population, anchored at end-2019, before the test starts. European wealth does not follow the US market.
  This is a screened-regional market proxy, not an external investable index. All figures should state that convention.
- Missing SIC on saved members is retained in the `Other` industry bucket; Keats allowed ff49/gics to satisfy its industry screen. Counts are recorded in `preparation.json`.
- Base membership comes directly from the saved Keats panel, including buffered counts above 80.
- The 250-stock extension uses the Keats buffering approach: large-cap ranks 1–250 (entry <=220, retention <=300).
  US mid-cap broadens around rank 500.5 to ranks 376–625 (entry 406–595, retention 326–675); it does not turn into a US large-cap portfolio.
- These changes in sample, characteristics, ranking and wealth benchmark mean the new results are not a like-for-like replication of the original report's numbers.
- 2021–2025 is inside the Keats test period. No fresh unseen holdout is invented, and none of the original experiment's conclusions or results are copied.
- FRED regime indicators remain US/global context variables for the European run, not European recession or policy indicators.
- The monitor leaves realised execution-cost observations missing; it does not mislabel simulated costs as observed costs.

## Input requirements and unavailable source observations

The original covariance model needs individual daily stock returns, not the saved daily-derived volatility characteristics.
Required files are described below; their current availability is reported in `outputs/input_audit.json`:

| Required local input | Contents |
|---|---|
| `data/inputs/daily_US.parquet` | `id`, `date`, `ret`: JKP security id, trading date, USD decimal total return; enough coverage for the regional top-500 risk pool and both US portfolios |
| `data/inputs/daily_EU.parquet` | Same schema, with **European JKP security ids**, not invented CRSP permnos; dividend-, split-, quotation-unit- and FX-adjusted USD returns |
| `data/inputs/daily_US.json` / `daily_EU.json` | Provenance declaring currency, return type, identifier and actual source |

Example provenance (replace source with the actual source):

```json
{"currency": "USD", "return_type": "total", "identifier": "jkp_id", "source": "Actual source and return construction"}
```

The ten-year daily covariance warmup is preserved. Provide daily returns covering the available 1994–2025 history;
exact usable dates depend on 2,520 factor-return observations and the additional Portfolio-ML lag requirements.
US daily data normally comes from CRSP; European returns require properly matched Compustat Global or equivalent data.
The workflow does not guess European security mappings or return adjustments.

For stages 10 and 13, `fred_USREC.parquet`, `fred_VIXCLS.parquet`, and `fred_FEDFUNDS.parquet` are provided in `data/inputs/`.
Each has `date,value` columns. Preparation copies them into the isolated raw folders.
The normal runner still stops on any cache miss instead of accessing the internet.

The three previously missing current-month returns were retrieved from WRDS and supplied without replacing existing observations.
The source still has 3/5/6 missing forward returns since December 2013 for US large-cap/US mid-cap/Europe.
WRDS also reports those JKP values as missing. Prepared datasets apply the user-selected zero assumption.
`run.py validate` writes their exact ids/months to
`outputs/<universe>/monthly_return_gaps.csv` reports any remaining unfilled gaps.
The original source-gap list is retained in `data/inputs/unresolved_forward_returns.csv`; no holdings are deleted.

The earlier supplemented prepared data remains in `data/processed/<universe>/with_supplements/`.
The initial prepared `base/` files are retained. CRSP terminal-month records are included as additional evidence;
the workflow does not silently substitute a different terminal-return convention.

## Review and validation

`original_manifest.json` records hashes of the original scripts and source modules. The runner checks that they are unchanged.
`data/processed/<universe>/base/preparation.json` records source fingerprints, counts, timing conventions and the resolved config.
`tests/test_keats.py` checks calendar gaps, security boundaries, no-download stops, path isolation, no fake holdout,
mid-cap selection, and prepared membership/ranks/return timing/cost units against the actual saved data.

Intermediate caches, prepared data and outputs are ignored by Git in this new folder. The supplementary `data/inputs/`
files are tracked, with Parquet files in Git LFS. Existing LFS-tracked Keats source datasets stay untouched.
