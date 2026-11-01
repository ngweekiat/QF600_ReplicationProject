"""Tables for precisely Static-ML and Portfolio-ML."""
from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import configure
configure(__file__)
import argparse
import pandas as pd
from jkmp.config import load_config
from jkmp.evaluation.metrics import table2,prob_outperformance,return_correlations,subperiod_table
from jkmp.strategies import STRATEGIES,require_strategies


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config");parser.add_argument("--override",action="append",default=[])
    args=parser.parse_args();cfg=load_config(args.config,args.override)
    if cfg.paths.outputs.parent.name != "base":raise ValueError("This script writes base results; choose the matching iteration script")
    out=cfg.paths.outputs
    paths=[out/"static_ml/pf.parquet",out/"portfolio_ml/rff/pf.parquet"]
    pfs=require_strategies(pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True),complete=True)
    folder=out/"tables";folder.mkdir(parents=True,exist_ok=True)
    t2=table2(pfs,cfg.investor.gamma,cfg.dates).reindex(STRATEGIES);t2.to_csv(out/"table2_test.csv");t2.to_csv(folder/"table2.csv")
    prob_outperformance(pfs,cfg.investor.gamma,cfg.dates,STRATEGIES).to_csv(folder/"table3.csv")
    return_correlations(pfs,cfg.dates,STRATEGIES).to_csv(folder/"table4.csv")
    periods={p.label:(p.start,p.end) for p in cfg.reporting.subperiods}
    subperiod_table(pfs,cfg.investor.gamma,cfg.dates,periods,["R_TC","SR_net","Utility","TC"]).to_csv(folder/"subperiods.csv")
    print(t2.to_string())

if __name__=="__main__":main()
