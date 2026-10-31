"""Normalize the explicitly acquired pre-entry history without changing saved panels.

Only actual WRDS observations are normalized. Unavailable stock/months remain
unavailable and are handled by the selected membership-exclusion policy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow import ROOT, source_dir, numeric_ids, write_json


def main():
    features = json.loads((ROOT / "feature_names.json").read_text())
    inputs = ROOT / "data/inputs"
    for region, universe in (("US", "us_large"), ("EU", "eu_large")):
        output = inputs / f"lag_history_{region}.parquet"
        if output.exists():
            print(f"Existing normalized history retained: {output.name}", flush=True)
            continue
        source = numeric_ids(pd.read_parquet(inputs / f"lag_history_source_{region}.parquet"))
        reference = pd.read_parquet(source_dir() / f"screened_{region}.parquet", columns=["eom", *features])
        groups = reference.groupby("eom", sort=False).groups
        result = source[["id", "eom"]].copy()
        for feature in features:
            result[feature] = .5
        for date, group in source.groupby("eom", sort=False):
            for feature in features:
                baseline = np.sort(reference.loc[groups[date], feature].dropna().to_numpy(dtype=float))
                values = group[feature].to_numpy(dtype=float, na_value=np.nan)
                valid = np.isfinite(values)
                ranks = np.full(len(group), .5)
                if len(baseline):
                    lo = np.searchsorted(baseline, values[valid], side="left")
                    hi = np.searchsorted(baseline, values[valid], side="right")
                    # Average percentile rank of this observation inserted into the saved reference population.
                    ranks[valid] = (lo + hi + 2.0) / (2.0 * (len(baseline)+1))
                result.loc[group.index, feature] = ranks
        rf = pd.read_parquet(ROOT.parent / "data/raw/ff_factors_monthly.parquet").set_index("date").rf
        wealth = pd.read_parquet(ROOT / f"data/processed/{universe}/with_zero_returns/wealth.parquet")
        market = wealth.set_index("eom").mu_ld1
        current_total = source.ret_exc.astype(float) + source.eom.map(rf)
        current_market = (source.eom-pd.offsets.MonthEnd(1)).map(market)
        result["g"] = ((1+current_total)/(1+current_market)).fillna(1.0)
        result["growth_imputed_neutral"] = current_total.isna() | current_market.isna()
        result["ret_ld1"] = np.nan  # Lag-only rows are never current return targets.
        result["lambda"] = np.nan  # Lag-only rows are never current traded positions.
        result["lag_history_only"] = True
        result.to_parquet(output, index=False)
        missing = pd.read_csv(ROOT / f"outputs/run_logs/unavailable_lag_keys_{region}.csv")
        missing.to_csv(inputs / f"unavailable_lag_history_{region}.csv", index=False)
        write_json(output.with_suffix(".json"), {"source": f"lag_history_source_{region}.parquet",
            "purpose": "Actual pre-entry signal history only; not training targets, membership or wealth benchmark inputs",
            "rank_reference": "Saved screened regional population plus the queried observation; average ties; original ranks unchanged",
            "missing_feature_or_reference_rank": .5, "rows": len(result),
            "neutral_growth_rows": int(result.growth_imputed_neutral.sum()),
            "unavailable_stock_months": len(missing)})
        print(f"{region}: normalized {len(result)} actual historical rows; {len(missing)} unavailable rows retained as missing", flush=True)


if __name__ == "__main__":
    main()
