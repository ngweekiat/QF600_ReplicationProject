# Kati_tryout

The active US high-cap iteration sequence now has **five separate experiment folders**: annual refitting already in Base; gamma 5; exit 120/costs x2; Market plus 1% TE tilt; recent-history weighting. Acceptance is test-period **P(better net Sharpe) >= 80%** against the last accepted version. Rejected experiments remain visible but are not inherited. Open `outputs/iteration_summary/index.html` and [the protocol](docs/FIVE_ITERATION_PROTOCOL.md). The three-market Base dates and calculations remain unchanged. Former iteration outputs/scripts are preserved under their respective `legacy_iterations/` folders.

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

Base uses **Static ML\*** for US high-cap only, and Portfolio ML for US high-cap, US mid-cap and EU high-cap. Market is a comparison reference. See [base changes and tuning](docs/BASE_STATIC_ML_STAR_AND_FIGURE1.md). The five US high-cap experiments use the unchanged current Base, Static ML* and Market as references; their accepted-only ancestry is documented in each summary.

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

All 50 have published JKP group labels and directions. Candidates must match both published JKP Cluster Labels and Factor Details and exist in WRDS Global Factor Data. Selection uses US high-cap observations from 2005–2014 only: 153 verified candidates, 152 eligible, then 50 with 13 A, 13 T and 24 S inclusion reasons. Read [methods, exclusions and limitations](docs/US_HIGH_CAP_SELECTION.md).

If preparing a separate environment, install the listed packages with `python -m pip install -r Kati_tryout/configs/requirements.txt`. The existing project Python already has the core numerical packages. Select a Python 3.11?3.13 notebook kernel.
