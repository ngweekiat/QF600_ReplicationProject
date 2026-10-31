"""Create isolated configurations and patch the initial script/package snapshot.

Developer utility: run once when creating this workflow, not before each experiment.
It never downloads data. Existing files are intentionally not recreated here.
"""
from pathlib import Path
import ast
import hashlib
import json
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))


def setup():
    # Read the declared feature list without importing or executing the downloader.
    tree = ast.parse((PROJECT / "configs/Keats universe code/build_universes.py").read_text())
    chars = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "CHARS" for t in n.targets))
    labels = pd.read_parquet(PROJECT / "data/raw/cluster_labels.parquet")
    details = pd.read_parquet(PROJECT / "data/raw/factor_details.parquet")
    themes = {}
    for char in chars:
        matches = labels.loc[labels.characteristic.eq(char), "cluster"]
        direction = details.loc[details.abr_jkp.eq(char), "direction"]
        if len(matches) != 1 or len(direction) != 1 or direction.iloc[0] not in (-1, 1):
            raise ValueError(f"Missing/ambiguous feature metadata: {char}")
        theme = matches.iloc[0].lower().replace(" ", "_").replace("-", "_")
        themes.setdefault(theme, []).append(char)
    (ROOT / "feature_names.json").write_text(json.dumps(chars, indent=2) + "\n")

    for universe in ("us_large", "us_mid", "eu_large"):
        folder = ROOT / "configs" / universe
        folder.mkdir(parents=True, exist_ok=True)
        base = {
            "base": "../original_base.yaml",
            "paths": {"raw": f"data/raw/{universe}", "processed": f"data/processed/{universe}/base",
                      "outputs": f"outputs/{universe}/base", "reference_r": "../reference/jkmp_r"},
            "dates": {"warmup_start": "1994-12-31", "validation_start": "2015-01-31",
                      "validation_end": "2019-12-31", "test_start": "2020-01-31", "test_end": "2025-12-31",
                      "holdout_start": None, "holdout_end": None, "data_end": "2025-12-31",
                      "holdout_locked": True, "evaluate": "test"},
            "features": {"themes": themes},
            "return_model": {"pred_years": [2015, 2025], "val_years": 5},
            "investor": {"wealth_end_date": "2019-12-31"},
            "grids": {"n_jobs": 1},
            "reporting": {
                "subperiods": [{"label": "2020-2022", "start": "2020-01-31", "end": "2022-12-31"},
                               {"label": "2023-2025", "start": "2023-01-31", "end": "2025-12-31"}],
                "cost_vix_start": "2020-01-31",
            },
        }
        # Feature themes replace the original six-theme mapping in this snapshot.
        full_base = yaml.safe_load((ROOT / "configs/original_base.yaml").read_text(encoding="utf-8"))
        full_base["features"]["themes"] = themes
        common = ROOT / "configs/keats_base.yaml"
        if not common.exists():
            common.write_text(yaml.safe_dump(full_base, sort_keys=False), encoding="utf-8")
        base["base"] = "../keats_base.yaml"
        variants = {
            "base": base,
            "linear_pfml": {"base": "base.yaml", "portfolio_ml": {"use_rff": False}},
            "active_pfml": {"base": "base.yaml", "portfolio_ml": {"add_constant": False, "benchmark_relative": True}},
            "universe_250": {"base": "active_pfml.yaml",
                             "paths": {"processed": f"data/processed/{universe}/universe_250", "outputs": f"outputs/{universe}/u250"},
                             "universe": {"target_n": 250, "entry_rank": 220, "exit_rank": 300}},
            "halflife": {"base": "universe_250.yaml", "paths": {"outputs": f"outputs/{universe}/u250_hl"},
                         "portfolio_ml": {"half_life_years": [3, 5, 10, None]}},
        }
        for name, content in variants.items():
            path = folder / f"{name}.yaml"
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    # Record the originals before patching; these hashes let reviewers verify the snapshot.
    manifest = {str(p.relative_to(PROJECT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in (PROJECT / "scripts", PROJECT / "src/jkmp") for p in folder.rglob("*.py")}
    (ROOT / "original_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for script in (ROOT / "scripts").glob("*.py"):
        if script.name.startswith("_"):
            continue
        text = script.read_text(encoding="utf-8")
        marker = "from __future__ import annotations\n"
        if marker in text:
            text = text.replace(marker, marker + "\nfrom _bootstrap import configure\nconfigure(__file__)\n", 1)
            script.write_text(text, encoding="utf-8")

    # Network access is disabled in the isolated snapshot, including accidental FRED cache misses.
    path = ROOT / "src/jkmp/data/wrds_pull.py"
    text = path.read_text(encoding="utf-8").replace("from dotenv import load_dotenv\n", "")
    text = text.replace("    import wrds\n\n    load_dotenv", "    raise RuntimeError('Downloads disabled: Keats workflow uses local inputs only.')\n    from dotenv import load_dotenv\n    import wrds\n\n    load_dotenv", 1)
    path.write_text(text, encoding="utf-8")
    path = ROOT / "src/jkmp/evaluation/regimes.py"
    text = path.read_text(encoding="utf-8").replace('    url = url_template.format(series=series)',
        '    raise FileNotFoundError(f"Missing local FRED cache for {series}; downloads are disabled.")\n    url = url_template.format(series=series)', 1)
    path.write_text(text, encoding="utf-8")

    # All model-fitting observations start in Keats' training period, not the risk warmup period.
    path = ROOT / "src/jkmp/models/return_model.py"
    text = path.read_text(encoding="utf-8").replace('rows = panel.loc[panel["valid_data"],',
        'rows = panel.loc[panel["valid_data"] & ((panel["eom"] + pd.offsets.MonthEnd(1)) >= pd.Timestamp("2005-01-31")),', 1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    setup()
