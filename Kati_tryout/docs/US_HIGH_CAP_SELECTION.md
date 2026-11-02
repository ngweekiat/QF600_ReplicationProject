# Current characteristic selection: US high-cap only

The current 50 are selected by correlation/covariance association analysis using **US high-cap held-stock observations from January 2005 through December 2014 return months**. This is a measurement period, not ML training. No regression, return-forecast or portfolio model is fitted in this selection stage.

| Item | Current rule |
|---|---|
| Population | 9,682 US high-cap stock-months; 128 distinct stocks over the period |
| Candidate pool | All 404 source characteristics in the historical pilot |
| Eligibility | At least 60% observed characteristic coverage and 60 valid monthly IC observations |
| Eligible candidates | 397 |
| Intermediate cap | 200 selected using US evidence and group/redundancy rules |
| Final set | 50; 13 economic groups, each containing 2–6 inputs |
| Published/provisional labels | 32 published JKP labels; 18 provisional US-correlation assignments |
| A selections | 13 published group anchors |
| T selections | 13 selections to meet minimum theme representation |
| S selections | 24 selections using association score after redundancy penalty |
| Later observations used for selection | None from 2015 onward |
| Other markets used for selection | None |

## What is ranked

For each month, each characteristic is compared with the next month's observed USD excess stock return. The main measure is Spearman rank correlation, averaged equally across months. An empirical sign orients a negatively associated characteristic so that strong negative associations can qualify alongside strong positive ones. This sign is computed within 2005–2014, not fitted by an ML algorithm.

The transparent score combines percentile ranks:

- 55%: oriented average monthly rank correlation with next-month returns;
- 20%: observed feature coverage;
- 15%: consistency of the association's sign across years;
- 10%: persistence of the characteristic's cross-sectional rank.

The greedy choice adds 0.02 for published group membership and subtracts 0.20 times the largest absolute correlation with a previously selected characteristic. Group quotas retain a published anchor and at least two members per group, with at most six. A lower standalone score can therefore be retained for representation or reduced redundancy. The 200-candidate preliminary cap uses a 0.10 redundancy penalty. A/T/S explains inclusion; it is not a portfolio weight.

These rules define “best” for this analysis. They do not prove that the set is globally optimal or that it will outperform in a later backtest. In-period association is not unseen predictive evidence.

## Matrices and charts

Open `outputs/us_high_cap_selection/index.html`.

| File/folder | Meaning |
|---|---|
| `recommended_50.csv` | The 50, descriptions, groups, A/T/S and score components |
| `all_candidate_scores.csv` | All 404 ranks, eligibility and exclusion decisions |
| `matrices/selected50/raw_correlation.csv` | 50×50 Pearson correlation of raw characteristic values |
| `matrices/selected50/raw_covariance.csv` | 50×50 sample covariance in the source units |
| `matrices/selected50/rank_correlation.csv` | 50×50 correlation after ranking within each US high-cap month |
| `matrices/selected50/rank_covariance.csv` | 50×50 covariance of those monthly percentile ranks |
| `*_with_return_*.csv` | 51×51 versions with an explicit `NEXT_MONTH_EXCESS_RETURN` column/row |
| `*_pair_counts.csv` | Actual observed pairs for each matrix cell |
| `per_stock_selected50_*.csv` | Each stock versus each selected characteristic: temporal Pearson, Spearman or covariance |
| `per_stock_feature_return_associations.parquet` | Per-stock associations with observed-pair counts |
| `charts/` | Correlation/covariance heatmaps, group/reason chart and signed return-association chart |
| `selection_manifest.json`, `verification.json` | Period, population, method, source hashes and checks |

Raw missing values and missing observed returns remain missing. No assumed-zero return targets are admitted to selection. The original 9,682 membership rows include three unavailable observed targets; those pairs are excluded from return statistics. Matrix covariances use pairwise-complete observations, denominator n−1 and at least 120 pairs. Per-stock results require at least 24 pairs.

Covariance magnitudes depend on units; a large raw covariance does not by itself mean a stronger characteristic. Pairwise covariance matrices can fail positive-semidefiniteness and are descriptive diagnostics, not fitted portfolio risk matrices. Pooled ranked matrix correlations and equally weighted mean monthly IC are different summaries and are labelled separately.

## Kati inputs and preserved ML implementation

The selected 50 are installed in all Kati configurations and full-history input panels. Selection uses only US high-cap observations; future base models can still use the three existing market populations. The new choice has not been validated as optimal for US mid-cap or Europe.

Existing Static ML, Portfolio ML, Fourier features and coefficient/weight-learning code are retained. The user requested deletion of the old characteristic archives, including their coefficients, weights, caches, inputs and reports. Those archives have been removed. Active inputs and configurations use the new US high-cap-selected 50 across all three base markets and US high-cap iterations. New model performance requires a fresh fit.

The chart `outputs/us_high_cap_selection/charts/selection_decision_path.pdf` shows 404 candidates, 397 eligible, 200 shortlisted and the final 50. It gives the fixed scoring formula and every selected characteristic's score, group and A/T/S inclusion reason. Covariance matrices are descriptive exports; the selection rule uses return association and correlation redundancy, rather than ranking raw covariance magnitudes.

No new ML model is fitted by this task. Preparing ranks/theme columns only transforms inputs. A future explicit model run must fit new coefficients for the changed feature set.

The selected 50 now include `dolvol_126d`. Its monthly rank is one ML input; the trading-cost calculation separately retains its original unranked USD value. It is not an additional 51st predictor.

## Reproduce the association analysis only

From the repository root:

```powershell
& 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe' -X utf8 Kati_tryout/scripts/select_characteristics.py
```

Or run `Characteristics_tryout/run.py select-us-high-cap` with the same Python. Both copies contain the same calculation and reproduce the same ordered 50. Kati's analysis uses a self-contained US observation snapshot and requires no sibling-project data at runtime.
