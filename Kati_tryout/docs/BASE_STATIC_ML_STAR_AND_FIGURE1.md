# Base Static ML* and combined Figure 1

Base return models and Portfolio ML select ridge penalties from the original 101-value validation grid. Return-model validation is market-specific.

This update applies to Kati base. The characteristic set remains the current verified JKP-only 50.

The 100-candidate grid was subsequently run: the saved US high-cap Static ML* annual excess returns are approximately -1.90% gross and -2.19% net for 2020–2025. The latest [market-specific RFF validation change](MARKET_SPECIFIC_RFF_TUNING.md) is configured but has not been fitted. Existing results predate that change. Only US high-cap base Static ML* uses the 100-candidate grid.

| Market | Active base models | Comparison reference |
|---|---|---|
| US high-cap | Static ML*, Portfolio ML | US high-cap value-weighted Market |
| US mid-cap | Portfolio ML | US mid-cap value-weighted Market |
| EU high-cap | Portfolio ML | EU high-cap value-weighted Market |

## Static ML* calculation

The return forecaster retains random Fourier features and ridge regression, with the market-specific validation adaptation described above. The portfolio layer chooses three parameters from the existing YAML grid:

| Parameter | Role | Grid |
|---|---|---|
| k | Multiplies trading impact in the allocation decision | 1, 1/3, 1/5, 1/10, 1/20 |
| u | Shrinks the predicted return | 0, 0.1, 0.25, 0.5, 1 |
| g | Adds g times each covariance diagonal element to the risk matrix | 0, 1, 2, 3 |

The 100 fixed candidate paths start in January 2015. At each December, the rule selects the candidate with the highest cumulative past utility, following the repository's paper implementation. Its parameters apply to the next return year. Thus 2020 uses observations through December 2019; 2021 may use completed 2020 returns. It does not use future returns from the year being traded. Candidate validation may extend through the full saved history, but a future December choice is never applied to an earlier year. This expands the parameter choices as a small-universe adaptation; it retains the same formula and annual selection method but is not the paper's exact parameter grid.

For each candidate, cumulative variance at a month is `cummean(r²) - cummean(r)²`. The selection objective is `cummean(r - tc - gamma * cumulative_variance / 2)`, matching the paper port. This is distinct from the sample-variance convention used in headline performance tables. Ties select the first candidate in the documented Cartesian grid order.

Monthly weights solve `(gamma * Sigma + g * diag(gamma * Sigma) + W * k * Lambda) w = u * forecast + W * k * Lambda * inherited_weights`. Realised trading costs use the original impact coefficient, not k times that coefficient. The selected portfolio starts from value weights at the test start, then inherits drifted positions monthly. Shorting and leverage remain unconstrained as in the existing model.

The implementation is `scripts/jkmp/portfolio/static_ml_star.py`; the entry point is `scripts/base/05_static_ml.py`. Headline returns, stock weights, all candidate performance and annual choices are saved under `outputs/base/us_large/static_ml_star/`. The original US high-cap `static_ml/` files are retained only because existing iteration reports reference that fixed strategy; base discovery never reads them. The obsolete regional Static ML output folders are removed.

The last completed 45-candidate 2020–2025 US high-cap Static ML* run remains negative: approximately -1.80% gross excess return, -2.10% net excess return and -0.465 net Sharpe, compared with -5.13%, -5.44% and -0.671 for the historical fixed rule. These are not results for the expanded 100-candidate grid. Tuning does not guarantee positive returns or reproduce the paper's different sample.

## Combined Figure 1

Open `outputs/base/overall analysis/figure1/index.html`.

| File | Contents |
|---|---|
| `charts/figure1_combined.png` and `.pdf` | Combined two-panel base figure |
| `combined_frontier_summary.csv` | Every point behind the combined figure |
| `COMBINED_FIGURE_METHODS.md` | Curve definitions and limits |
| `<market>/charts/figure1_implementable_frontier.*` | Existing individual-market two-panel charts |
| `<market>/frontier_summary.csv` | That market's model/risk/fund-size results |
| `<market>/accounting_verification.csv` | Independent reconstruction from saved stock weights |

Panel A compares all three Portfolio ML markets and US high-cap Static ML* at their configured fund size. Dotted curves are independently fitted zero-impact limits. Market diamonds identify each region's own value-weighted reference; squares identify configured risk aversion. Panel B includes every retained model's investor-size curve, including US high-cap Static ML*, using color for fund size and line style for method/market. The grid has five risk-aversion values and six wealth levels, including the zero-impact limit: all 120 model points across the four retained market/model combinations are shown in Panel B.

Static ML* is retuned separately at every risk/fund-size point. Existing Portfolio ML frontier fits are reused only when all portfolio model settings and data/source hashes match, and their saved portfolio hash and calendar pass validation. Changing a model setting or source observation prevents reuse. No return rescaling substitutes for a fitted point.

The horizontal axis is realised annual gross volatility. The vertical axis is annual arithmetic excess return after modelled costs. All curves use 72 return months in 2020–2025, with wealth anchored at end-2019. These estimated out-of-sample curves may cross or fail to be monotone; the plotted lines are not a proven optimal envelope. Removed Factor ML and Markowitz ML models are not reintroduced. Mid-cap and EU Static ML curves are excluded. Existing iteration frontier experiments retain their original definitions.

## Run and verify

From the repository root, use the project Python environment:

```powershell
python -X utf8 Kati_tryout/scripts/run_all.py --group base --scope full --universe all
python -X utf8 -m pytest Kati_tryout/tests -q
python -X utf8 Kati_tryout/scripts/verify.py
```

The runner resumes completed stages and refreshes reports. Return-model checkpoints identify the market, validation scope, search settings, prepared-panel identity and forecasting source. Static ML* requires a completion marker specific to the k/u/g grid and forecast-file identity, so changed forecasts trigger a fresh allocation run. Derived tables and regimes require markers based on their model-return file hashes; changed returns trigger their refresh. Figure 1 checks configuration/source hashes. The notebook is rebuilt from the current sources. Running `--group base` preserves saved iteration outputs.
