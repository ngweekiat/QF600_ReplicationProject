# Kati_tryout

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

Base retains Static ML and Portfolio ML for US high-cap, US mid-cap and EU high-cap. Iterations 1–3 apply only to US high-cap Portfolio ML. Market is a comparison reference. Every iteration comparison includes the earlier Portfolio ML versions and the unchanged base references.

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
