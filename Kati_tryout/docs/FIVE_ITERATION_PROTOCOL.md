# Five US high-cap experiments

The active workflow is configured in `configs/us_high_cap_iterations.json` and run by `scripts/iteration_sequence.py`. Each `scripts/iteration_N/run.py` runs through that step in dependency order. Every experiment retains its code, stock weights, returns, annual choices, statistical decision, charts and a short `SUMMARY.md` in `outputs/iteration_N/us_large/`.

| Step | Change | Reference |
|---|---|---|
| 1 | Annual refitting, already in Base | Retained as design; P(better) N/A |
| 2 | Gamma 5 | Last accepted setup |
| 3 | Exit rank 120, impact costs x2 | Last accepted setup |
| 4 | Market plus 1% annual TE-target ML tilt | Last accepted setup |
| 5 | Choose 3/5/10-year half-life or no decay | Last accepted setup |

P(better) is calculated only on Kati's existing January 2020–December 2025 test returns. It is the frequency of a positive candidate-minus-reference paper-net-Sharpe difference in 10,000 paired circular-block bootstrap samples, with six-month blocks and fixed seed 20261102. Ties receive half weight. This is bootstrap support conditional on the saved portfolios, not a conventional p-value or a probability of future success. The 80% threshold is the sole acceptance gate; drawdown is reported but adds no threshold. An identical candidate is not declared an improvement.

Only accepted changes enter the next experiment. Each `decision.json` states the inherited iterations, candidate settings, previous reference, P(better), verdict, and accepted setup after the decision. For example, iteration 4 always inherits iteration 1 and may also inherit gamma 5 if step 2 passes. Its summary explains this. Rejected candidates remain visible in the explorer and master table but are disabled in the subsequent accepted setup.

Once benchmark-relative optimisation is introduced, its risk coefficient is the pretest-calibrated gamma_TE rather than the absolute-risk gamma. The inherited gamma-5 decision remains in the ancestry, but gamma=5 is not multiplied into gamma_TE: it is superseded in the tilt's active-risk equation. Reapplying both would change the existing calculation. Step 5 reruns the same pretest TE-calibration rule with the candidate half-life search, so the 1% target is held fixed rather than a particular numerical gamma_TE. Realised test TE need not equal 1%.

The base retains its original dates, annual expanding refits, monthly trading, 50 JKP characteristics, selected-ridge grid, US high-cap Static ML* and three regional Portfolio ML calculations. The US mid-cap and EU high-cap models are not iterated. Exit-120 buffering uses the original US screened population and December-to-November membership timing; it is not the former 250-stock breadth experiment. It may change the exact number of held stocks. Doubled costs enter both optimisation and realised trading costs. Original labelled missing-return and history-exclusion policies are retained.

The older benchmark-relative/breadth/decay iteration results and scripts are preserved under `outputs/legacy_iterations/` and `scripts/legacy_iterations/`, respectively, and excluded from active discovery. Their history is also in Git. They are not the new five-step sequence.

## Evaluation limits

The 2020–2025 period is used to decide which experiments survive. Final performance on those same months is selected-sample evidence, not independent confirmation. Kati has no additional unseen holdout beyond that period; historical OOS prediction labels overlap the test. The current 50 were selected on US high-cap 2005–2014 outcomes. Every tried experiment is retained in the decision table, but bootstrap support does not correct all nested hyperparameter searches or earlier research choices. No deflated-Sharpe probability is fabricated from the five-trial count.

All Sharpe comparisons use the paper convention: annual mean net excess return divided by annual gross-return volatility. A net-volatility Sharpe is supplied separately in performance CSVs. Cumulative performance uses additive monthly net excess returns, matching `cumsum(r-tc)` in the authors' replication code. Additive drawdown is the cumulative sum minus its running maximum, including the initial zero; values multiplied by 100 are **percentage points**, not percent losses of capital. This drawdown extends the chart convention; it is not a verified drawdown definition in the paper. Relative cumulative performance is a difference of additive paths, not a wealth-index ratio. CAGR and geometric Calmar are omitted. Portfolio wealth used internally for trading costs and position drift still follows the original economic accounting.

## Run

```powershell
# Existing Base plus all five experiments, tables, charts and Figure 1:
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/run_all.py --group all --scope full --universe all

# Five-step experiment sequence using existing completed Base inputs:
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/iteration_sequence.py
```

Open `outputs/iteration_summary/index.html` for the screenshot-style table and final-versus-Market three-panel chart. The corresponding CSV is `outputs/iteration_summary/statistical_tests_all_iterations.csv`. Every iteration summary links back to the previous accepted version and the common Base/Static ML*/Market references.
