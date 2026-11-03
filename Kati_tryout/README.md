# Kati_tryout

Open `outputs/dashboard.html` and choose **Open the interactive iteration explorer**,
or open `outputs/dashboard_plotly.html` directly. Plotly is bundled under
`outputs/dashboard_assets/`, so hover, zoom, legend filters and PNG export work
offline. Select an iteration for cumulative returns, additive drawdowns, market
differences, costs, turnover, rolling returns, annual returns and distributions.
Annual model choices use saved `chosen.csv` files. The source explorer plots
numeric CSV diagnostics as lines, bars, distributions, boxes or heatmaps; original
annotated figures are listed alongside them. Tables above 400 KB stay linked rather
than copied into the page. Archived/revision-history results are excluded.
Refresh both dashboards with `python -X utf8 Kati_tryout/scripts/dashboard.py`.
These views read saved results and do not refit any model. The overview Sharpe
uses net-return SD, explicitly labelled, whereas native reports may use gross SD.

**Current US high-cap research chain: Base / annual refitting → 4 → 4.2 → 4.3.**
Iteration 2 is excluded from that chain. Open the [updated result images and tables](outputs/iteration_summary/index.html).
Trials 2, 3 and 5-10 below retain their historical results and original references;
they were not rebased or rerun in this revision. See [the current protocol](docs/ITERATION_4_SUBSTAGES.md).

**Iteration 10: fixed five-seed RFF ensemble** averages stock-level aims from
seeds 1-5, then executes one combined portfolio using the original cost formula.
See [its protocol](docs/ITERATION_10_SEED_ENSEMBLE.md) and the pretest sensitivity
evidence under `outputs/iteration_10/us_large/pretest_seed_sensitivity/`.
Run `python -X utf8 Kati_tryout/scripts/iteration_10/run.py` to resume this
experiment using verified earlier results. Test results, verdict and ancestry
are recorded in `outputs/iteration_10/us_large/SUMMARY.md` when complete.

**Iteration 9: benchmark-aware recursive cost learning** is implemented in `scripts/benchmark_netting.py`, with [its derivation and limits](docs/ITERATION_9_BENCHMARK_NETTING.md). It starts from the last accepted setup and retains actual execution/accounting. Run `python -X utf8 Kati_tryout/scripts/iteration_9/run.py`.

An [iteration-4 diagnostic image](outputs/iteration_4/us_large/diagnostic_for_trade_netting/iteration_4_next_iteration_diagnostic.png) motivates [proposed iteration 9: benchmark-aware trade netting](docs/PROPOSED_ITERATION_9_TRADE_NETTING.md). The original diagnostic is the rationale for the now-authorised ninth experiment; the implemented derivation is documented separately.

**Iteration 8: stable one-SE model selection** keeps simpler RFF/ridge candidates when their historical utility is within one block-bootstrap SE of the best. See [its protocol and completed results](docs/ITERATION_8_STABLE_SELECTION.md). Run `python -X utf8 Kati_tryout/scripts/iteration_8/run.py`. Iteration 2 also has a [diagnostic image](outputs/iteration_2/us_large/diagnostic_for_market_anchor/iteration_2_gap_diagnostic.png) explaining the rationale for testing a market anchor, using only iteration 2 and Market data.

**Iteration 7: stock-specific risk responsiveness** tests 63/126/252-day idiosyncratic volatility half-lives on the last accepted setup, with pretest utility selection and the same 1% TE target. Run `python -X utf8 Kati_tryout/scripts/iteration_7/run.py`. See [its protocol](docs/ITERATION_7_RISK_RESPONSIVENESS.md). Original risk files and earlier fitted portfolios are retained.

**Iteration 6: more selective trading** adds a training-only cost preference selected on pretest actual-cost utility, with the 1% TE target retained. Run `python -X utf8 Kati_tryout/scripts/iteration_6/run.py` to run this experiment using verified earlier results. See [its protocol](docs/ITERATION_6_SELECTIVE_TRADING.md). The active sequence now has ten iterations; earlier rejected trials remain reported.

Performance reports use **additive cumulative returns**, matching the paper's replication charts. Additive drawdown and cumulative differences versus Market are in **percentage points**, rather than percent losses of wealth. See [Additive reporting](docs/ADDITIVE_REPORTING.md) for formulas and saved-output refresh commands.

The active US high-cap iteration sequence now has **ten separate experiment folders**: annual refitting already in Base; gamma 5; exit 120/costs x2; Market plus 1% TE tilt; recent-history weighting; more selective trading; stock-specific risk responsiveness; stable model selection; benchmark-aware trade netting; fixed RFF seed ensemble. Acceptance is test-period **P(better net Sharpe) >= 80%** against the last accepted version. Rejected experiments remain visible but are not inherited. Open `outputs/iteration_summary/index.html` and [the protocol](docs/FIVE_ITERATION_PROTOCOL.md). The three-market Base dates and calculations remain unchanged. Former iteration outputs/scripts are preserved under their respective `legacy_iterations/` folders.

Kati_tryout uses the **US high-cap-only association-selected 50**, measured over **2005–2014**. Open [the correlation/covariance report](outputs/us_high_cap_selection/index.html) and [the current selection guide](docs/US_HIGH_CAP_SELECTION.md). This selection fits no ML model. Open [Kati_tryout.ipynb](Kati_tryout.ipynb) for all Python source, explanations and execution controls.

The build is prepared and verified first. Fresh investment model fits are a separate step; old Keats model results and caches are not Kati results. Keats remains separate and unchanged by this copy.

Old characteristic archives, inputs and fits were removed at the user's request. All active calculations use the new US high-cap-selected JKP-only 50. The ML weight-learning implementation is retained; new model performance requires a fresh fit. Open [the JKP-only selection chart](outputs/us_high_cap_selection/charts/selection_decision_path_jkp_only.pdf) to see the candidate gates, score formula and all 50 inclusion reasons. Use `outputs/us_high_cap_selection/recommended_50_jkp_only.csv` for the current table.

| Location | Contents |
|---|---|
| `Kati_tryout.ipynb` | Complete documented source and guarded execution controls |
| `scripts/` | All pipeline Python, model libraries and reporting code |
| `configs/` | Base settings for three markets and US high-cap iteration settings |
| `data/characteristics/` | Frozen 50 names, raw observations and selection/group metadata |
| `data/source/` | Regional source panels and membership with the new characteristics |
| `data/inputs/`, `data/reference/` | Return, liquidity, lag-history and reference inputs |
| `data/processed/` | Prepared panels and newly fitted model caches |
| `outputs/` | Selection evidence, model results, charts, comparisons and verification records |
| `docs/` | [User guide](docs/USER_GUIDE.md) and [assumptions/changes](docs/ASSUMPTIONS_AND_CHANGES.md) |
| `tests/` | Verification tests |

Base uses **Static ML\*** for US high-cap only, and Portfolio ML for US high-cap, US mid-cap and EU high-cap. Market is a comparison reference. See [base changes and tuning](docs/BASE_STATIC_ML_STAR_AND_FIGURE1.md). The ten US high-cap experiments use the unchanged current Base, Static ML* and Market as references; their accepted-only ancestry is documented in each summary.

Base ridge calibration uses **validation selection from 101 penalties** (zero plus 100 positive log-spaced values). Return-model selection is market-specific. Portfolio ML selects its own penalties independently. Iterations retain their original ridge grid.

The [combined base Figure 1](outputs/base/overall%20analysis/figure1/index.html) includes all retained markets' frontiers, by portfolio method and investor size. The PDF is `outputs/base/overall analysis/figure1/charts/figure1_combined.pdf`.

Base return-forecast RFF tuning now scores each market's historical members separately: US high-cap, US mid-cap and EU high-cap. Coefficients still fit on the corresponding screened regional training panel; annual dates and RFF/ridge formulas are preserved. Portfolio ML already has its own separately tuned RFF portfolio-weight model in each market. See [market-specific RFF tuning](docs/MARKET_SPECIFIC_RFF_TUNING.md). This configuration update has not been fitted; saved results predate the market-specific validation change. The US high-cap Static ML* grid has 100 candidates.

From the repository root, using the project's Python 3.11–3.13 environment:

```powershell
# Check the new build without fitting models:
python -X utf8 Kati_tryout/scripts/verify.py

# Inspect all jobs without running them:
python -X utf8 Kati_tryout/scripts/run_all.py --group all --scope full --universe all --list-jobs

# Fresh/resumable full run, including Figure 1 (may take many hours):
python -X utf8 Kati_tryout/scripts/run_all.py --group all --scope full --universe all

# Refresh the interactive explorer from available results:
python -X utf8 Kati_tryout/scripts/dashboard.py
```

The notebook source cells write their named local files; they do not execute each file's model entry point. Its fitting flags default to false. Read the user guide before starting a full notebook run.

**Current retained research chain: Base / annual refitting → 4 → 4.2 → 4.3. Iteration 2 is excluded.** See [methods and run commands](docs/ITERATION_4_SUBSTAGES.md). Stage 4 starts directly from Base; 4.2 selects settings using executed historical active utility; 4.3 adds a fixed one-SE simplicity rule. Each has pretest-only diagnosis and a results panel. Only 4.2/4.3 waive the 80% P(test) gate. Open the [joint comparison](outputs/iteration_4_substages_summary/index.html), [all trials with historical scope labelled](outputs/iteration_summary/all_trials_with_descriptions.png), or [current retained chain only](outputs/iteration_summary/kept_current_chain.png). Other trials retain historical results and original references until explicitly rerun.

All 50 have published JKP group labels and directions. Candidates must match both published JKP Cluster Labels and Factor Details and exist in WRDS Global Factor Data. Selection uses US high-cap observations from 2005–2014 only: 153 verified candidates, 152 eligible, then 50 with 13 A, 13 T and 24 S inclusion reasons. Read [methods, exclusions and limitations](docs/US_HIGH_CAP_SELECTION.md).

If preparing a separate environment, install the listed packages with `python -m pip install -r Kati_tryout/configs/requirements.txt`. The existing project Python already has the core numerical packages. Select a Python 3.11?3.13 notebook kernel.
