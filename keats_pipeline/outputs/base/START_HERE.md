# Base results: Static ML and Portfolio ML only

Open [index.html](index.html) for the readable result tables, economic interpretation and charts.

- [Performance and feasibility flags](results/performance_summary.csv)
- [Annual results](results/annual_performance.csv)
- [Training, test and OOS explanation](TRAIN_TEST_AND_OOS.md)
- [Assumptions, exclusions and changes](EXCLUSIONS_ASSUMPTIONS_AND_CHANGES.md)
- [Independent verification](results/independent_verification.csv)
- [Paired inference](results/paired_inference.csv)

Base contains only original Static ML and default Portfolio ML. Restored iteration results are separated into ../iteration_1, ../iteration_2 and ../iteration_3; unfinished runs are labelled explicitly.

Run `python keats_pipeline/scripts/base/compare_results.py` to refresh the saved-output comparison. This comparison does not fit any models. Use `python keats_pipeline/run_all.py --group base --scope full` for the dependency-ordered workflow.
