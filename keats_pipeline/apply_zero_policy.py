"""Create new prepared datasets with explicit zero excess-return imputations.

Retain the earlier with_supplements datasets and all original source files.
No network access or Git writes are performed.
"""
from pathlib import Path
import shutil
import json

import pandas as pd
import yaml

from workflow import ROOT, UNIVERSES, zero_missing_held_returns, assert_new_tree, write_json, originals_unchanged


def main():
    rf = pd.read_parquet(ROOT.parent / "data/raw/ff_factors_monthly.parquet").set_index("date")["rf"]
    for universe in UNIVERSES:
        config_file = ROOT / f"configs/{universe}/base.yaml"
        config = yaml.safe_load(config_file.read_text())
        source = ROOT / f"data/processed/{universe}/with_supplements"
        destination = ROOT / f"data/processed/{universe}/with_zero_returns"
        assert_new_tree(destination)
        if destination.exists():
            raise FileExistsError(f"Existing destination retained: {destination}")
        panel = pd.read_parquet(source / "panel.parquet")
        panel = zero_missing_held_returns(panel, rf)
        mask = panel.ret_ld1_imputed_zero
        panel.loc[mask, "valid_data"] = panel.loc[mask, "hist_months"].ge(12) & panel.loc[mask, "tr_ld0"].notna()
        destination.mkdir(parents=True)
        panel.to_parquet(destination / "panel.parquet", index=False)
        for name in ("wealth.parquet", "universe_membership.parquet", "universe_stats.csv", "screen_log.csv"):
            shutil.copy2(source / name, destination / name)
        panel.loc[mask, ["id", "eom", "ret_ld1_original", "ret_ld1", "tr_ld1"]].to_csv(
            destination / "zero_return_imputations.csv", index=False)
        report = json.loads((source / "preparation.json").read_text())
        report["missing_forward_return_policy"] = "zero_excess"
        report["zero_imputed_forward_returns"] = int(mask.sum())
        report["valid_training_rows"] = int(panel.valid_data.sum())
        report["previous_prepared_dataset"] = str(source)
        config["paths"]["processed"] = f"data/processed/{universe}/with_zero_returns"
        report["config"]["paths"]["processed"] = str(destination)
        write_json(destination / "preparation.json", report)
        # Switch only after the new dataset and audit log have been saved successfully.
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        print(f"{universe}: {int(mask.sum())} missing excess returns set to zero; earlier data retained", flush=True)
    originals_unchanged()


if __name__ == "__main__":
    main()
