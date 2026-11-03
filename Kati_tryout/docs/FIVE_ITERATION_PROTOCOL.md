# Ten US high-cap experiments

The active workflow is configured in `configs/us_high_cap_iterations.json` and run by `scripts/iteration_sequence.py`. Each `scripts/iteration_N/run.py` runs through that step in dependency order. Every experiment retains its code, stock weights, returns, annual choices, statistical decision, charts and a short `SUMMARY.md` in `outputs/iteration_N/us_large/`.

| Step | Change | Reference |
|---|---|---|
| 1 | Annual refitting, already in Base | Retained as design; P(better) N/A |
| 2 | Gamma 5 | Last accepted setup |
| 3 | Exit rank 120, impact costs x2 | Last accepted setup |
| 4 | Market plus 1% annual TE-target ML tilt | Last accepted setup |
| 5 | Choose 3/5/10-year half-life or no decay | Last accepted setup |
| 6 | Pretest-select a 1x/1.5x/2x learning-only trading-cost preference | Last accepted setup; currently iteration 4 |
| 7 | Pretest-select a 63/126/252-day idiosyncratic volatility half-life | Last accepted setup; currently iteration 4 |
| 8 | One-SE annual selection: fewer RFF features and stronger ridge among near-best historical candidates | Last accepted setup; currently iteration 4 |
| 9 | Recursive benchmark-plus-active cost representation in coefficient learning | Last accepted setup; currently iteration 4 |
| 10 | Equal target-aim weights across fixed RFF seeds 1-5; execute combined orders once | Last accepted setup; currently iteration 4 |

Iteration 10 is documented in [ITERATION_10_SEED_ENSEMBLE.md](ITERATION_10_SEED_ENSEMBLE.md).
The same annual original maximum-utility rule is applied separately to each seed,
with a common pretest-calibrated gamma and no performance-based seed selection.
Costs are charged on the combined orders, not averaged from separate books.

Iteration 9 is documented in [ITERATION_9_BENCHMARK_NETTING.md](ITERATION_9_BENCHMARK_NETTING.md). The cost curve is derived under the existing execution rule, verified against literal fixed-coefficient trades including exits, and calibrated on pretest TE only. Original return/risk proxies and all actual accounting remain unchanged.

Iteration 8 is described in [ITERATION_8_STABLE_SELECTION.md](ITERATION_8_STABLE_SELECTION.md). It is an annual model-selection extension, not a new portfolio objective. The rule is fixed before test; all annual choices use past utility only, with the same pretest TE-calibration discipline.

Iteration 7 is described in [ITERATION_7_RISK_RESPONSIVENESS.md](ITERATION_7_RISK_RESPONSIVENESS.md). Only the stock-specific risk weighting changes; the original 126-day control is independently reconstructed first. Factor risks, correlations, covariance window, stock/feature/date universes and costs stay fixed. It uses the same pretest selection and TE-calibration discipline.

Iteration 6 is described in [ITERATION_6_SELECTIVE_TRADING.md](ITERATION_6_SELECTIVE_TRADING.md). Its extra cost preference multiplies only the coefficient-learning cost matrix. It retains actual execution costs, original lambda and the execution rule. Every preference gets the same pretest TE-calibration rule. Actual-cost utility over the existing usable 2016-2019 pretest returns selects one scalar, frozen before the 2020-2025 comparison. All three trials remain visible. This is separate from iteration 3, which doubled both fitting and actual costs and also changed buffering.

P(better) is calculated only on Kati's existing January 2020–December 2025 test returns. It is the frequency of a positive candidate-minus-reference paper-net-Sharpe difference in 10,000 paired circular-block bootstrap samples, with six-month blocks and fixed seed 20261102. Ties receive half weight. This is bootstrap support conditional on the saved portfolios, not a conventional p-value or a probability of future success. The 80% threshold is the sole acceptance gate; drawdown is reported but adds no threshold. An identical candidate is not declared an improvement.

Each `decision.json` records its reference and ancestry. Under the current user instruction, iteration 4 explicitly restarts from Base / annual refitting (iteration 1), excluding gamma 5 from iteration 2. Later integer trials 5-10 currently remain historical results with their original ancestry until rerun. The current research branch is 1, 4, 4.2, 4.3; only 4.2/4.3 waive the 80% gate. See `ITERATION_4_SUBSTAGES.md`.

Benchmark-relative optimisation uses pretest-calibrated gamma_TE rather than the absolute-risk gamma. Gamma 5 is excluded from the current anchor ancestry and is not multiplied into gamma_TE. Removing its old ancestry label leaves the numerical active model unchanged. Historical step 5 reran TE calibration with its half-life search; that historical trial is not part of the current retained chain. Realised test TE need not equal 1%.

The base retains its original dates, annual expanding refits, monthly trading, 50 JKP characteristics, selected-ridge grid, US high-cap Static ML* and three regional Portfolio ML calculations. The US mid-cap and EU high-cap models are not iterated. Exit-120 buffering uses the original US screened population and December-to-November membership timing; it is not the former 250-stock breadth experiment. It may change the exact number of held stocks. Doubled costs enter both optimisation and realised trading costs. Original labelled missing-return and history-exclusion policies are retained.

The older benchmark-relative/breadth/decay iteration results and scripts are preserved under `outputs/legacy_iterations/` and `scripts/legacy_iterations/`, respectively, and excluded from active discovery. Their history is also in Git. They are not the new ten-step sequence.

## Evaluation limits

The 2020–2025 period is used to decide which experiments survive. Final performance on those same months is selected-sample evidence, not independent confirmation. Kati has no additional unseen holdout beyond that period; historical OOS prediction labels overlap the test. The current 50 were selected on US high-cap 2005–2014 outcomes. Every tried experiment is retained in the decision table, but bootstrap support does not correct all nested hyperparameter searches or earlier research choices. No deflated-Sharpe probability is fabricated from the nine-trial count.

All Sharpe comparisons use the paper convention: annual mean net excess return divided by annual gross-return volatility. A net-volatility Sharpe is supplied separately in performance CSVs. Cumulative performance uses additive monthly net excess returns, matching `cumsum(r-tc)` in the authors' replication code. Additive drawdown is the cumulative sum minus its running maximum, including the initial zero; values multiplied by 100 are **percentage points**, not percent losses of capital. This drawdown extends the chart convention; it is not a verified drawdown definition in the paper. Relative cumulative performance is a difference of additive paths, not a wealth-index ratio. CAGR and geometric Calmar are omitted. Portfolio wealth used internally for trading costs and position drift still follows the original economic accounting.

## Run

```powershell
# Existing Base plus all ten experiments, tables, charts and Figure 1:
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/run_all.py --group all --scope full --universe all

# Ten-step experiment sequence using existing completed Base inputs:
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/iteration_sequence.py
```

Open `outputs/iteration_summary/index.html` for the screenshot-style table and final-versus-Market three-panel chart. The corresponding CSV is `outputs/iteration_summary/statistical_tests_all_iterations.csv`. Every iteration summary links back to the previous accepted version and the common Base/Static ML*/Market references.
