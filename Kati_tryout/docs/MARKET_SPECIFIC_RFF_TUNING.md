# Separate RFF tuning for each base market

RFF variance, feature count and ridge penalty are selected jointly using each market's historical validation members. Portfolio ML independently selects its own settings using preceding portfolio utility.

Status: implemented and checked without fitting models or running backtests. Existing fitted coefficients, returns and charts have not been refreshed.

| Base market | Return-forecast hyperparameter validation | Coefficient fitting panel | Portfolio ML RFF tuning |
|---|---|---|---|
| US high-cap (`us_large`) | Eligible historical US high-cap members only | Screened US panel | Already separate, using US high-cap portfolio utility |
| US mid-cap (`us_mid`) | Eligible historical US mid-cap members only | Screened US panel | Already separate, using US mid-cap portfolio utility |
| EU high-cap (`eu_large`) | Eligible historical EU high-cap members only | Screened EU panel | Already separate, using EU high-cap portfolio utility |

Previously these were already separate runs, but the return-forecast candidate validation error was measured on all screened regional stocks. This change sets `return_model.validation_scope: universe` in each base YAML. The best RFF variance, feature count and ridge penalty are selected using only that market's own historical member stock-months. Membership is measured at the historical formation month; today's constituents are never imposed on earlier dates. Candidate settings and learned coefficients are saved independently under each market's processed directory. Markets can legitimately choose the same settings without sharing a fitted model.

Coefficient fitting and final refitting continue to use screened regional stocks, following the paper's broad fitting convention. US high-cap and US mid-cap may therefore share eligible training observations, while their validation scores and selected models are independent. This is an adaptation of the validation population, not a claim to match the paper's exact validation sample. The 50 characteristics, stock universes, dates, five-year validation window, annual expanding refit, RFF transformation, ridge equation and validation MSE criterion are unchanged.

For return year 2020, candidate coefficients fit on realised returns from 2005–2014 and are scored on 2015–2019 market members. After selection, coefficients refit on all eligible 2005–2019 regional observations. Later return years advance the same boundaries annually. No return from the year being predicted enters its selection or coefficient fit.

The return predictor feeds Static ML* and prediction diagnostics. Portfolio ML has a separate RFF model that learns portfolio-weight coefficients directly from risk/cost-adjusted utility; it does not consume `pred_ld1`. Its existing separate market tuning is preserved. Consequently, changing return-forecast validation alone should not change the base Portfolio ML results. Its completed outputs can be reused.

The active nine-step iteration sequence uses the unchanged base data read-only. It does not refit an alternative return-prediction model or use the old 250-stock breadth configuration. See [the current protocol](FIVE_ITERATION_PROTOCOL.md). Historical three-step outputs are preserved in `outputs/legacy_iterations/`.

## Files and output audit

- Settings: `configs/us_large/base.yaml`, `configs/us_mid/base.yaml`, `configs/eu_large/base.yaml`.
- Scoring filter and RFF search: `scripts/jkmp/models/return_model.py`.
- Stage entry point: `scripts/shared/03_return_model.py`.
- Each market saves `chosen.csv`, `hp_search.parquet`, `diagnostics.csv`, `models.npz` and a tuning completion marker under `data/processed/<market>/with_zero_returns/return_model/`.
- `chosen.csv` and `hp_search.parquet` identify `market`, `validation_scope` and the scoring row count `n_val`. `chosen.csv` also records `n_val_regional` to distinguish the full regional validation-period population from the market scoring subset.
- Checkpoints prevent the runner from silently reusing old broad-validation fits. Changed predictions also invalidate the US Static ML* allocation checkpoint. Tables, comparisons and Figure 1 refresh when the base workflow reaches them.

## Run from the repository root

```powershell
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/run_all.py --group base --scope full --universe all
```

This resumes unaffected stages, refits the three return models, updates US high-cap Static ML* and refreshes the base reports and full Figure 1 frontier grid. It does not start iteration experiments. Figure 1 can take several hours. Existing model results remain the previous run until this command completes successfully.
