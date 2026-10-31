"""Apply user-selected, logged stock/month exclusions without deleting old panels."""
import datetime as dt
import json
import shutil

import pandas as pd

from workflow import ROOT,UNIVERSES,config_path,write_json,originals_unchanged
from jkmp.config import load_config
from jkmp.data.lag_history import exclude_unavailable


def main():
    for universe in UNIVERSES:
        for variant in ("base","universe_250"):
            cfg=load_config(config_path(universe,variant))
            p=cfg.paths.processed
            if not (p/"panel.parquet").exists():
                continue
            if (p/"history_exclusions.csv").exists():
                events=pd.read_csv(p/"history_exclusions.csv")
                if len(events):
                    # A previous application may already have updated the panel;
                    # keep the monthly summary consistent without touching that panel.
                    summary=pd.read_parquet(p/"panel.parquet",columns=["eom","in_universe"])
                    counts=summary.loc[summary.in_universe].groupby("eom").size().rename("n_members")
                    archive=p/"previous_versions/before_history_summary"
                    archive.mkdir(parents=True,exist_ok=True)
                    if (p/"universe_stats.csv").exists() and not (archive/"universe_stats.csv").exists():
                        shutil.copy2(p/"universe_stats.csv",archive/"universe_stats.csv")
                    counts.to_csv(p/"universe_stats.csv")
                continue
            frame=pd.read_parquet(p/"panel.parquet")
            updated,events=exclude_unavailable(frame,universe,cfg.universe.lookback_months)
            if len(events):
                archive=p/"previous_versions"/("before_history_exclusions_"+dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
                archive.mkdir(parents=True)
                shutil.copy2(p/"panel.parquet",archive/"panel.parquet")
                shutil.copy2(p/"preparation.json",archive/"preparation.json")
                updated.to_parquet(p/"panel.parquet",index=False)
                if (p/"universe_stats.csv").exists():
                    shutil.copy2(p/"universe_stats.csv",archive/"universe_stats.csv")
                updated.loc[updated.in_universe].groupby("eom").size().rename("n_members").to_csv(p/"universe_stats.csv")
            events.to_csv(p/"history_exclusions.csv",index=False)
            report=json.loads((p/"preparation.json").read_text())
            report["history_excluded_stock_months"]=len(events)
            report["members"]=int(updated.in_universe.sum())
            write_json(p/"preparation.json",report)
            print(f"{universe}/{variant}: {len(events)} excluded stock/months",flush=True)
            if len(events): print(events.to_string(index=False),flush=True)
    originals_unchanged()


if __name__=="__main__":
    main()
