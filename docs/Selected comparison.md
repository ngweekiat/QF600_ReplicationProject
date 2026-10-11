# Selected comparison

The active comparison is Static-ML* versus Portfolio-ML (random Fourier features).
Both use the same universe, dates, covariance estimates, investor wealth and risk
aversion, and trading-cost assumptions. Markowitz-ML, Factor-ML, untuned Static-ML,
the linear Portfolio-ML extension and other benchmarks are excluded from reports.
Older result files are filtered when rebuilding the tables and figures.

After downloading the data, run these scripts using Python 3.13:

1. `scripts/01_prepare.py`
2. `scripts/02_covariance.py`
3. `scripts/03_return_model.py`
4. `scripts/05_static_ml.py`
5. `scripts/06_pfml_inputs.py`
6. `scripts/07_portfolio_ml.py`

Stage 04 is skipped. Return predictions remain necessary for Static-ML*.
Stage 07 rebuilds `outputs/table2_test.csv` with both selected methods. Stage 05
can produce a provisional table containing only Static-ML* before Stage 07 runs.
Optional stages 08 (frontiers), 09 (feature importance), and 10 (market regimes)
also use the selected comparison. Static-ML* feature importance holds its annual
hyperparameters fixed while re-predicting shuffled features and rerunning its
portfolio including trading costs; this is an extension to the original comparison.

The configured universe targets 80 eligible US stocks on NYSE and Nasdaq,
selected by market capitalisation each December. New members enter at rank 70
or better; existing members can remain through rank 100. Remaining slots are
filled to the target. Retained members are not trimmed to force exactly 80, and
ineligible stocks can leave during the year. Share codes 10, 11 and 12 and JKP
common/primary_sec/obs_main flags are required, with at least 50% of the 50
features available and sufficient consecutive history. The covariance model
uses a broader top-500 estimation universe. Test returns cover 2020–2025.

The 50% coverage screen now requires at least 25 of the 50 characteristics.
Other missing characteristic values are imputed after cross-sectional ranking.

The 50 characteristics cover all 13 JKP themes: value (6), momentum (5), quality
(6), investment (5), low risk (5), short-term reversal (3), profitability (4),
profit growth (4), low leverage (3), accruals (3), size (3), debt issuance (2),
and seasonality (1). These were selected in advance for economic coverage from
the original R feature set, rather than ranked by realised returns. Cluster
assignments follow JKP's labels, which sometimes differ from intuitive economic
categories. Representativeness does not establish predictive performance or
low correlations. Actual sample coverage must be checked after the WRDS pull.

Changing the characteristics also changes the covariance theme factors. Rebuild
data preparation, covariance, return models, Portfolio-ML inputs and results,
including any cached frontier grids (`scripts/08_grids.py --force`), before using
the expanded specification. An old licensed-data cache must be refreshed with
`scripts/00_download.py pull --force` to include the additional columns.
