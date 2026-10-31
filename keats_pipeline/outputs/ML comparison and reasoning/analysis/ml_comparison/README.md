# Saved Keats ML comparison

All comparison code, reports, tables, charts and validation history are together under `keats_pipeline/outputs/ML comparison and reasoning/`. The scripts read saved Keats outputs and never train models, optimise portfolios or download inputs.

From the repository root:

```powershell
$comparisonPython = 'C:\Users\katiravan\AppData\Local\Programs\Python\Python313\python.exe'
$comparisonRunner = 'keats_pipeline/outputs/ML comparison and reasoning/run_comparison.py'
& $comparisonPython $comparisonRunner --run-tests
# Individual evaluation groups, each in a new timestamped directory:
& $comparisonPython $comparisonRunner --only performance
& $comparisonPython $comparisonRunner --only inference --bootstrap-reps 2000
& $comparisonPython $comparisonRunner --only robustness --universe us_mid
& $comparisonPython $comparisonRunner --only audit
```

The runner also accepts an absolute path from any working directory. It locates the repository from its own location. The default Python on this computer is 3.12 without project dependencies; use the explicit 3.13 executable. `--output` selects a fresh subdirectory of this comparison folder; existing directories are refused.

`--run-tests` executes both comparison and existing Keats tests. Logs are saved in the new report's `validation/` folder; temporary files stay inside this comparison folder. Extended-length Windows temporary paths avoid legacy path-length limits.

The completed report is in `results/`; see [Start here](../../START_HERE.md). Missing results are SKIPPED. Large saved losses remain in the arithmetic paper metrics; synthetic geometric paths stop when excess-index growth is nonpositive. Active strategies retain their gamma_TE objectives and are evaluated at common absolute gamma=10.

[Relocation notes](../../RELOCATION.md) map historical recorded paths to their current locations. Existing ML implementations and saved portfolio results were not modified.
