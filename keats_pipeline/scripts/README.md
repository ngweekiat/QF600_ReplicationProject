# Script groups

| Folder | Used for | Main entry points |
|---|---|---|
| `base/` | Base results: Static ML and Portfolio ML only | `05_static_ml.py`, `07_portfolio_ml.py`, `10_regimes.py`, `11_tables.py`, `compare_results.py` |
| `iteration_1/` | Benchmark-relative model; 1%, 2%, 4% tracking-error targets | `run.py` delegates to the original shared active engine |
| `iteration_2/` | 250-stock universe, capacity and cost sensitivity | `run.py`, `compare_breadth.py`, `monitor.py`; `reference_*` retain the earlier wider-universe reference calculations |
| `iteration_3/` | Half-life search (3/5/10 years and no decay) | `run.py`, `compare_halflife.py` |
| `shared/` | Common inputs and numerical engines | `01_prepare.py`, `02_covariance.py`, `03_return_model.py`, `04_benchmarks.py`, `active_engine.py` |

`shared/06_pfml_inputs.py` is reused by base and iteration 2's absolute-return reference inputs. Its RFF transformation uses the same original mathematical implementation. Shared `src/jkmp/` modules are calculation libraries, not separate report groups. `portfolio/iteration_static_ml.py` preserves the older tuned Static ML used by iteration 2 references; `portfolio/static_ml.py` implements only the original non-starred base strategy.

[SCRIPT_MAP.csv](SCRIPT_MAP.csv) and [SCRIPT_MAP.json](SCRIPT_MAP.json) list every script. `../script_groups.py` is the authoritative dispatcher, including legacy variant aliases. It rejects an active iteration stage in the base group. Market benchmark output belongs to iteration reference folders.

Run from the repository root with Python 3.13:

```powershell
# Regenerate reports from existing saved data only; no missing experiment is fitted:
python -X utf8 keats_pipeline/run.py compare --group base
python -X utf8 keats_pipeline/run.py organise

# Inspect a plan without running it:
python -X utf8 keats_pipeline/run_all.py --group iteration_3 --scope full --list-jobs

# Examples for a later, explicitly requested computation:
python -X utf8 keats_pipeline/run.py 12 --group iteration_1 --universe us_large calibrate
python -X utf8 keats_pipeline/run.py 12 --group iteration_1 --universe us_large --te-target 0.02 run
python -X utf8 keats_pipeline/run.py 13 --group iteration_2 --universe us_large --te 0.02
python -X utf8 keats_pipeline/run.py 14 --group iteration_3 --universe us_large
```

`run_all.py --group iteration_2` or `iteration_3` includes required earlier groups. It may perform lengthy missing calculations; `--list-jobs` never does. This organisation task restored existing output only, as requested. The normal workflow remains offline.
