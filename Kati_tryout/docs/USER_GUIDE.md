# Using Kati_tryout

Open `Kati_tryout.ipynb` in VS Code or Jupyter using the project environment (Python 3.11–3.13). The notebook contains all Python source rather than only links to scripts. Use its outline to jump to a model or iteration and expand a source cell when needed. Every code cell has a Markdown explanation immediately before it.

The current feature selection is **US high-cap only, 2005–2014**, using correlation/covariance association statistics without ML training. Open `outputs/us_high_cap_selection/index.html` or run `scripts/select_characteristics.py` to reproduce rankings, matrices and the selection decision chart. Read [US_HIGH_CAP_SELECTION.md](US_HIGH_CAP_SELECTION.md) for the score and A/T/S rules. Leave `RUN_MODELS = False` for the association-only workflow. Old characteristic archives and their fitted ML weights/results were removed at the user's request.

## Notebook workflow

1. Run **Locate the pipeline** to select this folder and kernel.
2. Browse the documented source. Cells beginning `%%writefile` write exactly the displayed file under Kati; they do not run the script's main function. Run those cells only when you intend to materialize the notebook's source edition.
3. Run **Inspect the workflow before fitting** to print the dependency plan.
4. Run **Verify the local build** to check the data/configuration/implementation safeguards. Verification is not a fitted backtest.
5. In **Choose whether to run models**, change `RUN_MODELS` from false to true when ready. Choose the Figure 1 control separately. A full run may take many hours. Keep the kernel/session alive and inspect job logs for progress.
6. Run **Refresh the results explorer**, then use its available market/run/measure controls and report links. A browser can also open `outputs/dashboard.html` directly.

Running all cells with the default controls does not initiate the long model experiment. It does materialize the embedded source edition. If you have edited normal scripts since the notebook was generated, rebuild the notebook before running its source cells:

```powershell
python -X utf8 Kati_tryout/scripts/build_notebook.py
```

The notebook's metadata records a SHA256 hash for each embedded source/configuration file. Data remains in `data/`; the notebook is not a substitute for the accompanying dataset.

## Commands from the repository root

Use `python` from the same environment used for Keats. If the shell uses another interpreter, replace `python` with that environment's executable.

```powershell
# Build verification; no ML fits:
python -X utf8 Kati_tryout/scripts/verify.py

# Dependency plan; no model runs:
python -X utf8 Kati_tryout/scripts/run_all.py --group all --scope full --universe all --list-jobs

# Base only, all three markets:
python -X utf8 Kati_tryout/scripts/run_all.py --group base --scope core --universe all

# Base plus all US high-cap iterations and the full reporting/frontier scope:
python -X utf8 Kati_tryout/scripts/run_all.py --group all --scope full --universe all

# Refresh the explorer after completed or partially completed runs:
python -X utf8 Kati_tryout/scripts/dashboard.py
```

The runner uses dependency order and persistent logs. A rerun reuses compatible completed **Kati** outputs and resumes missing jobs. It does not download data, commit, or push. A failed prerequisite marks dependent jobs blocked; independent work can continue. Do not interpret a missing or blocked result as zero performance.

## Experiment map

| Group | What changes | What stays as the comparison |
|---|---|---|
| Base | US high-cap Static ML* with annual k/u/g tuning; Portfolio ML in all three markets, using the current JKP-only 50 | US high-cap Static ML*, three regional Portfolio ML models and Market references |
| Iteration 1 | US high-cap Portfolio ML becomes benchmark-relative; 1%, 2% and 4% pretest tracking-error budgets | Base Portfolio ML, base Static ML and original Market |
| Iteration 2 | Carries the active formulation into a 250-stock US high-cap universe; capacity and cost sensitivity | Iteration 1 and base references; separate 250-stock Market disclosed |
| Iteration 3 | Tests no decay against 3-, 5- and 10-year half-lives for recent observations | Iterations 1–2 and base references |
| Final | Compares base Portfolio ML and all three iterated Portfolio ML results | Wider report also includes Market and base Static ML |

Internal candidate grids are hyperparameter choices within a method. They are not additional headline ML families.

## Find data and outputs

| Need | Folder or file |
|---|---|
| Exact frozen 50 names | `data/characteristics/recommended_50.json` |
| Descriptions, groups, direction policy and inclusion reasons | `data/characteristics/selected_feature_metadata.csv` |
| Raw selected characteristics by region | `data/characteristics/` |
| Training/member source panels | `data/source/` |
| Return/liquidity/lag inputs | `data/inputs/` |
| Prepared ranked inputs and new caches | `data/processed/` |
| Snapshot provenance and integrity | `data/SNAPSHOT_MANIFEST.json` |
| Presentation and detailed selection evidence | Under `outputs/`, linked from the explorer |
| Three-market base comparisons | `outputs/base/overall analysis/` |
| A particular iteration comparison | `outputs/iteration_N/us_large/overall analysis/` |
| Four Portfolio ML versions | `outputs/base/overall analysis/us_high_cap_progression/` |
| All base models and iterations | `outputs/base/overall analysis/all_models_and_iterations/` |
| Figure 1 grids and charts | `outputs/base/overall analysis/figure1/` and iteration extensions |
| All-group job state/logs | `outputs/organisation_audit/run_logs/` |

CSV and Parquet outputs retain exact numbers; charts support explanation. Model reports appear only after their prerequisites finish. The first-run snapshot adapter `scripts/prepare_inputs.py` reads the sibling source projects only when creating the initial local snapshot. Once that snapshot is present, regular computations use Kati's own data.

Read [ASSUMPTIONS_AND_CHANGES.md](ASSUMPTIONS_AND_CHANGES.md) before describing performance or calling any interval an unseen test.
