# Keats: base plus three iterations

For US high cap (`us_large`), US mid cap (`us_mid`) and European high cap (`eu_large`), results are now separated by research run:

| Run group | Results | Scripts |
|---|---|---|
| Base | [Static ML and Portfolio ML only](outputs/base/index.html) | [base/](scripts/base/README.md) |
| Iteration 1 | [Benchmark-relative Portfolio ML](outputs/iteration_1/index.html) | [iteration_1/](scripts/iteration_1/README.md) |
| Iteration 2 | [250-stock breadth, capacity and cost sensitivity](outputs/iteration_2/index.html) | [iteration_2/](scripts/iteration_2/README.md) |
| Iteration 3 | [Recent-history/half-life weighting](outputs/iteration_3/index.html) | [iteration_3/](scripts/iteration_3/README.md) |

Open [the results directory](outputs/index.html), [availability table](outputs/availability_by_group_and_market.csv), or [script map](scripts/SCRIPT_MAP.csv). Every run group has separate market folders. Native CSV, Parquet, PNG and PDF outputs stay within their group; new descriptive summaries and charts are inside each market's `reports/` folder. Input and restoration audits are labelled separately.

The report links below reflect the completed organisation snapshot. A later all-iterations computation is still running; see [the publication status](outputs/RUN_STATUS_AT_COMMIT.json) and [snapshot notes](outputs/GIT_SNAPSHOT.md).

## What was restored

The original iteration mathematical implementations and available saved outputs were restored from Downloads/keats_removed_models_20261101. Iteration 1's active model, iteration 2's 250-stock/capacity machinery and iteration 3's half-life choices are retained. Changes to these scripts are routing, output paths and shared imports needed for the grouping; the original estimator formulas and experiment settings are preserved. The original tuned Static ML used by wider-universe references is isolated from the base implementation.

By user choice, unfinished experiments were **not completed**. US high-cap iteration 1 has saved active-run results. Iteration 2 has preparatory/reference outputs but no completed active 250-stock experiment. Other iteration/market combinations are explicitly marked unavailable or unfinished. Existing preparation/benchmark outputs are not relabelled as completed iteration performance. Earlier replication-paper figures are not inserted into Keats folders.

Market is an iteration benchmark, not a third base result. Base retains only original non-starred Static ML and default Portfolio ML. The original 50 characteristics, original base memberships and saved portfolio performance are retained. The separate broader-characteristics study remains in Downloads/characteristcs and is not activated.

## Commands and timing

The [script guide](scripts/README.md) explains common inputs and the group-specific entry points. Each market has `base.yaml`, `iteration_1.yaml`, `iteration_2.yaml` and `iteration_3.yaml`. Legacy aliases `active_pfml`, `universe_250` and `halflife` map to the corresponding iteration.

```powershell
$keatsPython = 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe'
# Existing-output reports only:
& $keatsPython -X utf8 keats_pipeline/run.py compare --group base
& $keatsPython -X utf8 keats_pipeline/run.py organise
# Plan inspection only:
& $keatsPython -X utf8 keats_pipeline/run_all.py --group all --scope full --list-jobs
& $keatsPython -m pytest keats_pipeline/tests -q
```

An actual `run_all.py --group ...` computation resumes missing jobs and can be lengthy; it was not used to complete missing iterations during this organisation task. Base-only computations use `--group base`. Data input/forecast/covariance stages are shared; `scripts/shared/active_engine.py` is the original engine used by all three iteration wrappers. No command commits or pushes automatically.

The nominal initial training split is 2005-2014, validation is 2015-2019 and test is 2020-2025, in return months. Models update annually on expanding past observations and weights update monthly. OOS is the broader forward-prediction history, not a separate unseen 2021-2025 holdout. Portfolio ML's usable moment returns start in February 2006 for the US and March 2012 for Europe due to covariance/lag requirements. Read [the full timing record](outputs/base/TRAIN_TEST_AND_OOS.md).

## Retained assumptions and provenance

Source data, original 50 features, base memberships, modelled costs and historical assumptions remain unchanged. Missing held-stock excess returns remain user-selected zero assumptions (15 US large, 21 US mid, 20 Europe over full history). The original lag-history supplements and exclusion policy remain available to the iterations. Base European Static ML retains extreme modelled costs and invalid geometric growth observations; these are flagged arithmetic diagnostics, not feasible investor wealth.

[Organisation records](outputs/organisation_audit/) contain the file restoration inventory, verification, saved logs and a snapshot of the prior code. Historical cleanup records under outputs/model_cleanup and prior log snapshots describe earlier states; the current four-group availability table is authoritative.
