"""Offline lag-history completion and explicit per-stock/month exclusions."""
from pathlib import Path
import json

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]


def complete_rows(rows, need, features):
    missing = pd.MultiIndex.from_frame(need[["id","eom"]]).difference(
        pd.MultiIndex.from_frame(rows[["id","eom"]]))
    if not len(missing):
        return rows
    additions = []
    columns = ["id","eom","g","ret_ld1","lambda",*features]
    for region in ("US","EU"):
        path = ROOT / f"data/inputs/lag_history_{region}.parquet"
        if path.exists():
            if not set(columns).issubset(pq.ParquetFile(path).schema.names):
                continue  # A different feature specification must not read an incompatible supplement.
            frame = pd.read_parquet(path, columns=columns)
            keys = pd.MultiIndex.from_frame(frame[["id","eom"]])
            additions.append(frame[keys.isin(missing)])
    if additions:
        rows = pd.concat([rows,*additions], ignore_index=True)
    still_missing = pd.MultiIndex.from_frame(need[["id","eom"]]).difference(pd.MultiIndex.from_frame(rows[["id","eom"]]))
    if len(still_missing):
        raise ValueError(f"{len(still_missing)} stock/months have no row in actual lagged signal history (unavailable); "
                         "apply the explicit membership-exclusion policy rather than invent history")
    return rows


def exclude_unavailable(panel, universe, lookback):
    """Exclude only affected formation months, allowing a later month once history exists."""
    region = "EU" if universe.startswith("eu") else "US"
    path = ROOT / f"data/inputs/unavailable_lag_history_{region}.csv"
    result = panel.copy()
    result["excluded_missing_lag_history"] = False
    events = []
    if path.exists():
        unavailable = pd.read_csv(path, parse_dates=["eom"])
        for row in unavailable.itertuples():
            mask = result.in_universe & result.id.eq(row.id) & result.eom.between(
                row.eom, row.eom+pd.offsets.MonthEnd(lookback+1))
            for affected in result.loc[mask,["id","eom"]].itertuples():
                events.append({"stock_id":int(affected.id), "formation_month_end":affected.eom,
                    "return_month_end":affected.eom+pd.offsets.MonthEnd(1),
                    "missing_history_month_end":row.eom,
                    "reason":"Excluded for this stock/month: required lagged history not reported by WRDS"})
            result.loc[mask,"excluded_missing_lag_history"] = True
            result.loc[mask,"in_universe"] = False
    fields=["stock_id","formation_month_end","return_month_end","missing_history_month_end","reason"]
    return result,pd.DataFrame(events,columns=fields)
