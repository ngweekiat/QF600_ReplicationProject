"""Build the complete documented source notebook without fitting an ML model.

Each Python file is embedded as an editable %%writefile cell. Running those
cells materializes the organised files; it does not execute their entry points.
The final cells inspect the plan, verify inputs, and optionally run models.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text, *, collapsed=False):
    metadata = {"tags": ["embedded-source"]} if collapsed else {}
    if collapsed:
        metadata.update(jupyter={"source_hidden": True}, collapsed=True)
    return {"cell_type": "code", "execution_count": None, "metadata": metadata,
            "outputs": [], "source": text.rstrip() + "\n"}


def caption(path, source):
    relative = path.relative_to(ROOT).as_posix()
    if path.suffix == ".py":
        try:
            doc = ast.get_docstring(ast.parse(source)) or ""
        except SyntaxError:
            doc = ""
        purpose = " ".join(doc.split())[:550] or "Supporting implementation for the Kati workflow."
    else:
        purpose = "Declarative settings for the local Kati workflow; paths resolve inside Kati_tryout."
    if "tests/" in relative:
        role = "Verification source. Tests run only when explicitly invoked; this cell writes the test file."
    elif "/jkmp/" in relative:
        role = "Calculation library. Other stages import its functions; this cell writes the module."
    elif "/analysis/" in relative or path.stem in {"dashboard", "comparison_analysis", "organise_results"}:
        role = "Reporting source. Reads available local results when called; this cell writes the module."
    else:
        role = "Pipeline implementation or configuration. This cell writes the file without running its main function."
    return f"### `{relative}`\n\n{purpose}\n\n**Role:** {role}\n\n**Input:** the embedded source below. **Output:** `{relative}`. Expand the code cell to read or edit it."


def source_groups():
    scripts = sorted((ROOT / "scripts").rglob("*.py"))
    tests = sorted((ROOT / "tests").rglob("*.py"))
    configs = sorted((ROOT / "configs").rglob("*.yaml"))
    settings = ROOT / "settings.json"
    grouped = [
        ("Workflow entry points and preparation", [p for p in scripts if p.parent == ROOT / "scripts"]),
        ("Shared data and estimation stages", [p for p in scripts if p.relative_to(ROOT / "scripts").parts[0] == "shared"]),
        ("Base: Static ML and Portfolio ML", [p for p in scripts if p.relative_to(ROOT / "scripts").parts[0] == "base"]),
        ("Iteration 1: benchmark-relative US high-cap Portfolio ML", [p for p in scripts if p.relative_to(ROOT / "scripts").parts[0] == "iteration_1"]),
        ("Iteration 2: broader US high-cap Portfolio ML and capacity", [p for p in scripts if p.relative_to(ROOT / "scripts").parts[0] == "iteration_2"]),
        ("Iteration 3: recent-history weighting", [p for p in scripts if p.relative_to(ROOT / "scripts").parts[0] == "iteration_3"]),
        ("Reporting and statistical comparisons", [p for p in scripts if p.relative_to(ROOT / "scripts").parts[0] == "analysis"]),
        ("Calculation libraries", [p for p in scripts if p.relative_to(ROOT / "scripts").parts[0] == "jkmp"]),
        ("Verification tests", tests),
        ("Experiment configurations", configs + ([settings] if settings.exists() else [])),
    ]
    included = {p for _, files in grouped for p in files}
    remaining = [p for p in scripts if p not in included]
    if remaining:
        grouped.append(("Additional supporting modules", remaining))
    return grouped


def build():
    cells = [markdown("""
# Kati_tryout — the complete documented pipeline

This notebook contains the Python source for an independent copy of Keats using the **frozen suggested 50 characteristics from Characteristics_tryout**. Data lives in `data/`, scripts in `scripts/`, configurations in `configs/`, and results in `outputs/`. Keats remains separate.

The notebook is a readable source edition and an execution hub. Every source file has its own explained code cell. Source cells use `%%writefile`: running them writes the organised file and **does not execute its model fit**. Calculation-library cells are collapsed initially where the notebook viewer supports it. Use the notebook outline and search to find a model, iteration, or function.

**Build status:** this edition is built and verified before the expensive experiment run. Existing Keats forecasts, coefficients, covariance caches and backtest results are not imported as Kati results. The dashboard reports which results are available.

1. Run **Locate the pipeline**.
2. Read or edit the embedded source sections as needed. Run source cells if you want to materialize that edition.
3. Inspect the dependency plan and run verification.
4. Set the model controls deliberately when ready to start fresh fits.
5. Refresh and explore the charts and tables.

The source notebook requires the accompanying local `data/` folder. It does not contain gigabytes of stock data inside its JSON.
"""), markdown("""
## Locate the pipeline

**Purpose:** select this experiment's folder and establish safe local paths. **Input:** the notebook's working directory. **Output:** `KATI_ROOT`, local folders, and the Python version. This cell does not read credentials, download data, or fit models.
"""), code('''from pathlib import Path
import os
import sys
import subprocess
import json

candidates = [Path.cwd(), Path.cwd() / "Kati_tryout"]
candidates += [p for p in Path.cwd().parents if p.name == "Kati_tryout"]
KATI_ROOT = next((p.resolve() for p in candidates
                  if p.name == "Kati_tryout" and (p / "data").is_dir()), None)
if KATI_ROOT is None:
    raise FileNotFoundError("Open this notebook in Kati_tryout or its parent repository.")
if not (3, 11) <= sys.version_info[:2] < (3, 14):
    raise RuntimeError("Choose a Python 3.11–3.13 kernel for the project's dependency range.")
for name in ("scripts", "configs", "data", "outputs", "docs", "tests"):
    (KATI_ROOT / name).mkdir(exist_ok=True)
os.chdir(KATI_ROOT)
sys.path.insert(0, str(KATI_ROOT / "scripts"))
print("Pipeline:", KATI_ROOT)
print("Python:", sys.executable)
'''), markdown("""
## Research scope and timing

| Run | Markets and methods | Comparison |
|---|---|---|
| Base | US high-cap, US mid-cap and EU high-cap; Static ML and Portfolio ML | All retained base ML models and the requested Market references |
| Iteration 1 | US high-cap Portfolio ML; benchmark-relative construction, 1%/2%/4% tracking-error targets | Base Portfolio ML, base Static ML and Market |
| Iteration 2 | US high-cap Portfolio ML; 250-stock breadth and capacity | Iteration 1 and unchanged base references |
| Iteration 3 | US high-cap Portfolio ML; recent-history weighting | Iterations 1–2 and unchanged base references |
| Final | US high-cap Portfolio ML base and iterations 1–3 | Four-version progression, with Market/Static references in the wider reports |

The current 50 are selected using **US high-cap observations from 2005–2014 only**. Selection is a correlation/covariance association analysis with no regression or ML training. No 2015–2025 return outcome or other-market observation enters this shortlist. The existing annual model-refit and monthly portfolio-weighting implementation is preserved for a later explicit model run. Old characteristic archives and their fitted results have been removed. See outputs/us_high_cap_selection/charts/selection_decision_path.pdf for the complete selection chart.

The 50 cover 13 groups: 32 published labels and 18 provisional US high-cap correlation-based labels. The 13 A, 13 T and 24 S reasons retain published anchors, theme representation and the association/redundancy rule. Covariance theme signs use published directions for published members and the 2005–2014 US high-cap association sign for provisional members. The selected 50 include dolvol_126d as a ranked signal; costs separately use its original unranked dollar-volume value. There are exactly 50 ML inputs.
"""), markdown("""
## Embedded source edition

The following sections contain the actual source, including all calculation libraries and verification tests. Each cell materializes only the named local file. The source is also available as normal files under `scripts/`; editing a script does not automatically update this notebook. Rebuild this edition with `python scripts/build_notebook.py` after reviewing script changes. Running an older notebook's source cells intentionally restores its embedded edition.
""")]
    hashes = {}
    count = 0
    for group, paths in source_groups():
        if not paths:
            continue
        cells.append(markdown(f"## Source — {group}"))
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_text(encoding="utf-8")
            hashes[relative] = hashlib.sha256((content.rstrip()+"\n").encode("utf-8")).hexdigest()
            cells.append(markdown(caption(path, content)))
            cells.append(code(f"%%writefile {relative}\n{content}", collapsed=True))
            count += 1
    cells.extend([
        markdown("""
## US high-cap correlation and covariance selection — no ML training

**Purpose:** reproduce the current 50 from 404 candidates using only US high-cap observations in 2005–2014. **Input:** the self-contained US high-cap observation snapshot. **Output:** rankings, A/T/S reasons, 50×50 feature matrices, return-augmented matrices, per-stock associations and interactive charts. This optional cell performs no ML fit and does not install a newly changed list automatically over fitted models.
"""), code('''RECOMPUTE_ASSOCIATIONS = False
if RECOMPUTE_ASSOCIATIONS:
    subprocess.run([sys.executable, "-X", "utf8", str(KATI_ROOT / "scripts/select_characteristics.py")],
                   cwd=KATI_ROOT, check=True)
else:
    print("Existing association report: outputs/us_high_cap_selection/index.html")
'''),
        markdown("""
## Inspect the workflow before fitting

**Purpose:** list the real job dependencies. **Input:** organised scripts/configurations. **Output:** printed execution plan. `--list-jobs` does not launch model fits. The first run needs fresh forecasts, covariance and portfolio coefficients for the new features.
"""), code('''command = [sys.executable, "-X", "utf8", str(KATI_ROOT / "scripts/run_all.py"),
           "--group", "all", "--scope", "full", "--universe", "all", "--list-jobs"]
subprocess.run(command, cwd=KATI_ROOT, check=True)
'''), markdown("""
## Verify the local build

**Purpose:** check the frozen inputs, local paths, source/notebook structure and implementation safeguards. **Input:** this local build. **Output:** the verification report and its printed status. This check does not establish portfolio performance; model results require the fresh fitting stages below.
"""), code('''subprocess.run([sys.executable, "-X", "utf8", str(KATI_ROOT / "scripts/verify.py")],
               cwd=KATI_ROOT, check=True)
'''), markdown("""
## Choose whether to run models

**Purpose:** make the long computation explicit. **Input:** the controls below. **Output:** an execution command, followed by model results only if `RUN_MODELS=True`.

Leave both flags false to browse and verify the build. Set `RUN_MODELS=True` for the full current experiment scope. `RUN_FIGURE1` controls the additional frontier fits: false adds `--skip-figure1`, while true includes the frontier stage. The full CLI command is in the user guide. The model run may take many hours and reuses only compatible completed Kati outputs on a later invocation.
"""), code('''RUN_MODELS = False
RUN_FIGURE1 = False
RUN_GROUP = "all"
MARKETS = "all"

runner = KATI_ROOT / "scripts/run_all.py"
model_command = [sys.executable, "-u", "-X", "utf8", str(runner),
                 "--group", RUN_GROUP, "--scope", "full", "--universe", MARKETS]
if not RUN_FIGURE1:
    model_command.append("--skip-figure1")
print("Command:", subprocess.list2cmdline(model_command))
if RUN_MODELS:
    subprocess.run(model_command, cwd=KATI_ROOT, check=True)
else:
    print("Model fits have not been started by this cell.")
'''), markdown("""
## Refresh the results explorer

**Purpose:** create a local dashboard from results that actually exist. **Input:** Kati result files. **Output:** charts, tables and report navigation. Before fitting, the explorer shows selection evidence and marks missing model results as unavailable. It does not substitute Keats performance. Use market, run and measure controls in the explorer where offered.
"""), code('''subprocess.run([sys.executable, "-X", "utf8", str(KATI_ROOT / "scripts/dashboard.py")],
               cwd=KATI_ROOT, check=True)
from IPython.display import HTML, display
dashboard = next((p for p in [KATI_ROOT / "outputs/dashboard.html", KATI_ROOT / "outputs/index.html"]
                  if p.exists()), None)
if dashboard is not None:
    relative = dashboard.relative_to(KATI_ROOT).as_posix()
    display(HTML(f'<p><a href="{relative}" target="_blank">Open the full results explorer</a></p>'
                 f'<iframe src="{relative}" style="width:100%;height:850px;border:1px solid #d5dee5"></iframe>'))
else:
    print("No dashboard has been generated yet; inspect the dashboard command's output.")
'''), markdown("""
## Explain the frozen 50 and their groups

**Purpose:** display the shortlist explanation, separately from fitted investment performance. **Input:** selection charts copied into Kati outputs. **Output:** presentation overview and detailed evidence chart, when available. The three selection codes mean **A** published group anchor, **T** minimum theme representation, and **S** balanced score with a redundancy penalty. They are reasons for inclusion, not ML coefficients or importance weights.
"""), code('''from IPython.display import Image, display
for filename in ("selection_groups_and_reasons.png", "selected50_rank_correlation.png", "selected50_rank_covariance.png"):
    image_path = KATI_ROOT / "outputs/us_high_cap_selection/charts" / filename
    if image_path.exists():
        display(Image(filename=str(image_path)))
    else:
        print(filename, "is not present; the frozen names and metadata are under data/characteristics/.")
'''), markdown("""
## Where to inspect completed results

| Question | Local result location |
|---|---|
| Which characteristic data and labels are used? | `data/characteristics/` and `data/SNAPSHOT_MANIFEST.json` |
| Which base model/market performs better? | `outputs/base/overall analysis/` |
| What changed in a specific iteration? | `outputs/iteration_N/us_large/overall analysis/` |
| How do the four Portfolio ML versions compare? | `outputs/base/overall analysis/us_high_cap_progression/` |
| How do all models and iterations compare? | `outputs/base/overall analysis/all_models_and_iterations/` |
| Where are Figure 1 charts? | `outputs/base/overall analysis/figure1/` and iteration extensions |
| What ran, failed, or was blocked? | `outputs/organisation_audit/run_logs/` for the all-group workflow |

CSV/Parquet tables and charts sit beside their corresponding report. Read the availability/status files before citing a result. A blocked downstream job means its prerequisite did not finish; it is not a zero-return observation. Read `docs/ASSUMPTIONS_AND_CHANGES.md` for timing, imputation, selection and interpretation limits.
""")])
    source_directories = sorted({str(Path(name).parent).replace("\\", "/") for name in hashes})
    cells[2]["source"] += "\n# Create the embedded source folders before any %%writefile cell runs.\n"
    cells[2]["source"] += f"for relative_directory in {source_directories!r}:\n    (KATI_ROOT / relative_directory).mkdir(parents=True, exist_ok=True)\n"
    notebook = {"cells": cells,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.13"},
                             "kati_source_manifest": hashes,
                             "kati_source_hash_convention": "UTF-8 text, LF newlines, trailing whitespace/newlines canonicalized to one final newline",
                             "kati_source_file_count": count,
                             "kati_model_results_embedded": False},
                "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"kati-{i:04d}"
    target = ROOT / "Kati_tryout.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Built {target}: {count} embedded files, {len(cells)} explained cells; no model fits run.")
    return target


if __name__ == "__main__":
    build()
