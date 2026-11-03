# Iteration 4.2 and 4.3: a separate validation research branch

**Current revision:** iteration 4 starts directly from Base / annual refitting.
Iteration 2 (gamma 5) is excluded. The retained research chain is **1, 4, 4.2,
4.3**. Iteration 4 compares with Base (P(test) 99.63%); 4.2 compares with 4;
4.3 compares with 4.2. There is no separate 4.1. Historical trials 2, 3 and 5-10
remain available with their original references, and were not rebased or rerun.
The pre-revision results are under `outputs/revision_history/`.

These substages deepen the existing US high-cap market-anchor design. They are
stored beside `outputs/iteration_4/`, as `outputs/iteration_4.2/us_large/` and
`outputs/iteration_4.3/us_large/`. Their source entry points are under the
matching `scripts/iteration_4.2/` and `scripts/iteration_4.3/` folders, with shared
implementation in `scripts/active_validation.py`.

| Research stage | Question | Reference |
| --- | --- | --- |
| 4: existing market anchor | How does benchmark-relative Portfolio ML perform? | Existing saved iteration 4 |
| 4.2: executed active-utility validation | Do settings selected by the learning proxy also rank well after complete historical execution, risk and costs? | Iteration 4 |
| 4.3: uncertainty-aware selection | Does a fixed simplicity heuristic help when the highest validation utility is uncertain? | Research stage 4.2 |

The original learning score is calculated exactly for its own quadratic proxy.
This branch investigates a different validation implementation; it does not
assert a coding error in that score or guaranteed improvement. Candidate books
must use only information available before their forecast year, and realised
trading costs must be charged from their own inherited and target positions.
Read the model module and saved method/verification files for the final scoring,
candidate timing and calibration details.

For these **two substages only**, the user's instruction removes the 80%
P(test) retention gate. Paired P(test) remains reported and interpreted. Retaining
a research stage does not imply that it beats its predecessor, and P(test) is
bootstrap support rather than a conventional p-value or probability of future
outperformance. The already-inspected 2020-2025 period is not a fresh holdout.

The original trial history remains separate from the revised chain. US mid-cap and EU high-cap base models are
not altered by this branch. Both new stages are complete. Inspect each stage's
`SUMMARY.md`, `decision.json`, `performance_summary.csv` and verification files
for the exact saved evidence.

## Run from the repository root

```powershell
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/iteration_4/run.py --start-from 4
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/base_anchor_revision_reports.py --pretest
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/iteration_4.2/run.py
& "C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe" -X utf8 Kati_tryout/scripts/iteration_4.3/run.py
```

`run_all.py --group all` retains the original integer sequence. The complete
source notebook contains an additional opt-in control for this separate branch.
The dashboard offers dedicated 4.2 and 4.3 comparisons; existing integer selectors
do not automatically mix these substages into historical comparisons.

## Fixed design and timing

`configs/us_high_cap_substages.json` freezes both stages before their test runs.
Original RFF signals, 50 inputs, candidate grids, annual coefficient equations,
sample dates, risk estimates, wealth and actual costs are retained. Both stages
use iteration 4's existing calibrated gamma_TE **174.1**, without recalibration
or leverage rescaling. Their realised/ex-ante tracking errors need not equal the
original nominal 1% target; the changed selection is the only model-design
override in 4.2, followed by the fixed selection heuristic in 4.3.

Each of 1,414 fixed-(g,p,lambda) candidate books begins with VW holdings at the
December 2014 formation date, generating 2015 validation returns. Its coefficients
update each year using the original past-only fit. It carries its OWN inherited
weights and pays for annual coefficient changes, exits and actual trades. The
December score used for year Y contains returns only through December Y-1:

```
U = 12 * [mean((r - tc) - (r_B - tc_B))
          - 174.1/2 * sample_variance(r - r_B, ddof=1)]
```

4.2 maximises U. 4.3 bootstraps the best candidate's U with 2,000 paired circular
six-month-block samples, RECOMPUTING both mean and sample variance per resample.
Candidates within one bootstrap SE of the best score are eligible; fewer RFF
features, stronger ridge and smaller RFF variance break the selection. This is
a declared heuristic, not a hypothesis/equivalence test or a correction for
1,414-way selection. The rule is not tuned after seeing test returns.

The actual selected policy has its own 2016-2019 pretest and 2020-2025 test reset
starts. Its inherited holdings differ from each fixed-hyperparameter shadow book;
all actual policy-switch costs are charged, but validation does not exactly
anticipate every later switch transition. This remaining approximation is
explicit; executed validation is not claimed to eliminate all implementation gaps.

## Completed results: 2020-2025

| Measure | Iteration 4 | Iteration 4.2 | Iteration 4.3 |
| --- | ---: | ---: | ---: |
| Annual net excess return | 15.1354% | 15.1375% | 15.2004% |
| Net Sharpe (gross-return SD) | 0.870264 | 0.870544 | 0.870707 |
| Annual actual trading costs | 0.05945% | 0.06124% | 0.01894% |
| P(test), versus preceding stage | 99.63% vs Base | 66.88% | 53.17% |
| Mean test ex-ante TE | 0.5579% | 0.5697% | 0.2011% |

Both new stages are retained under the user's no-threshold instruction. Neither
P(test) provides clear evidence of superior Sharpe. 4.3 raises net excess return
by about 6.29 basis points/year versus 4.2 and reduces costs, while taking less
active risk. These are descriptive results on an already inspected test sample.
Common absolute-portfolio utility (gamma 10) and the new active-utility validation
score (gamma_TE 174.1) are different quantities and must not be conflated.

The 4.2 rationale honestly shows limited motivation: pretest rank correlation
0.973, different winners in 2 of 5 Decembers, and the same final 2019 winner.
For 4.3, the best validation winner stays bootstrap-best in only 3.6% of paired
resamples; 182 candidates meet the fixed one-SE heuristic. This motivates a
robustness experiment, not a promise of future improvement. Both rationale images
exclude all 2020+ strategy outcomes. Their CSVs, methods, input hashes and checks
are under each stage's `rationale/` folder.

The independent verifier reconstructs shadow orders, stock accounting, drift,
exits, annual choices, bootstrap utility, selected-policy execution and P(test),
including the revised Base parent. Exact current counts are saved in the
verification/completion JSON files. Unaffected fits remain preserved.

The removal of gamma-5 ancestry does not change returns: the active-risk equation
already uses gamma_TE 174.1 in place of the absolute gamma. The changed reference
does change iteration 4's P(test). This was verified by fresh fitting.

The new pre-iteration-4 diagnosis replays Base with original execution on
2016-2019 only. Base beta is 0.236 (95% block interval -0.052 to 0.458), and its
observed annual shortfall is 4.72 percentage points, but that gap's interval
includes zero. Saved costs account for 3.76 points of the gap. This motivates
testing a benchmark-relative mandate, not a claim that the anchor must improve
returns. Neither iteration 2 nor 2020+ outcomes enter this diagnosis.

## Evidence and charts

- [Pre-iteration-4 diagnosis](../outputs/iteration_4/us_large/charts/pretest_rationale.png)
- [Iteration-4 results vs Market](../outputs/iteration_4/us_large/charts/market_comparison_panel.png)
- [4.2 pretest statistical and economic rationale](../outputs/iteration_4.2/us_large/charts/rationale.png)
- [4.3 pretest statistical and economic rationale](../outputs/iteration_4.3/us_large/charts/rationale.png)
- [Joint test-results comparison](../outputs/iteration_4_substages_summary/index.html)
- [Independent numerical verification](../outputs/iteration_4.3/us_large/independent_verification.json)
- [All trials, with historical scope labelled](../outputs/iteration_summary/all_trials_with_descriptions.png)
- [Current retained chain only](../outputs/iteration_summary/kept_current_chain.png)

PNG and PDF versions sit together. Each stage's `rationale/` folder contains the
underlying statistics, methods and source checks. The notebook embeds all new
Python source and the frozen JSON protocol; `RUN_4_SUBSTAGES` defaults to false.
