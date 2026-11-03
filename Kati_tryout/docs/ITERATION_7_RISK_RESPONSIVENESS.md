# Iteration 7: stock-specific risk responsiveness

Hypothesis: the inherited portfolio may size active stock positions using
idiosyncratic risk forecasts that react too slowly, or react too noisily. Test
EWMA half-lives of **63, 126 (unchanged control), and 252 trading days**.

Only the idiosyncratic volatility weighting changes. Factor exposures, factor
returns, factor covariance, correlation and factor-volatility half-lives, the
2,520-day factor window, stocks, 50 characteristics, train/validation/test dates,
actual costs, learning cost preference, RFF and ridge grids remain unchanged.
This is risk-estimation parameter tuning within the existing model, not the
paper's exact parameter settings. The rejected iteration 6's 2x preference is
not inherited; the current reference is iteration 4 (accepted ancestry 1, 2, 4).

The implementation reconstructs daily residuals using the original saved daily
factor returns and prior-month exposures. The reconstructed 126-day variances
must match the original saved covariance before a candidate can be used. It then
copies the original control exactly and changes only the idiosyncratic diagonal
for the alternatives. Size-group imputation and data-availability rules are fixed.

For each candidate, recalibrate the existing 1% ex-ante TE target using only
pretest data. Choose maximum actual-cost utility at common evaluation gamma 10
over the same usable pretest returns, January 2016-December 2019 (48 months).
2015 supplies the first December model selection; upstream history is unchanged.
Ties prefer the unchanged 126-day control, then the smaller half-life.

Freeze the chosen half-life and gamma_TE before the 2020-2025 test comparison.
Annual expanding coefficient refits and historical hyperparameter choices still
run normally. Use paired bootstrap P(better paper-net Sharpe) against iteration 4,
10,000 draws, 6-month blocks, seed 20261102, acceptance >=80%. An identical control
is not an improvement. Drawdown adds no gate. Test results remain exploratory,
not independent unseen confirmation. All candidate pretest results are retained;
only the pretest-selected configuration is evaluated as this test experiment.

Entry point:

```powershell
python -X utf8 Kati_tryout/scripts/iteration_7/run.py
```

Code: `scripts/risk_responsiveness.py` and the existing experiment engine.
Risk data: `data/processed/us_large/iteration_7_covariance/` (original risk data
untouched). Results: `outputs/iteration_7/us_large/`, including control checks,
`risk_half_life_validation.csv`, `pretest_selection.json`, every pretest trial,
test weights/returns, charts and the decision/inheritance summary.

## Completed result

Pretest utility selected **252 days**, gamma_TE **162.6**. The independently
reconstructed 126-day variances matched the original to maximum absolute error
**1.6653e-16**, and alternative covariances retain identical stock IDs, exposures
and factor covariance across all **251** original formation months.

| Test metric, 2020-2025 | Iteration 4 | Iteration 7 |
|---|---:|---:|
| Annual net excess return | 15.1354% | 15.1322% |
| Paper net Sharpe | 0.870264 | 0.870353 |
| Annual actual trading cost | 0.05945% | 0.06444% |
| Monthly turnover | 4.0359% | 4.2065% |
| Worst additive drawdown | -27.4224 pp | -27.4374 pp |

P(better versus iteration 4) is **50.92%**, below 80%: **not included**.
The tiny Sharpe increase is not supported by the declared comparison; annual net
return falls slightly and costs rise. Final accepted ancestry stays **1, 2, 4**.
No settings were changed after observing this test result.
