"""Run annually tuned Static-ML* for the US high-cap base only."""
from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import configure
configure(__file__)
import argparse
import pandas as pd
from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.holdout import evaluation_window
from jkmp.evaluation.metrics import table2
from jkmp.portfolio.static_ml_star import hyper_grid, run_static_ml_star, tuning_grid_marker


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config");parser.add_argument("--override",action="append",default=[])
    args=parser.parse_args();cfg=load_config(args.config,args.override)
    if cfg.paths.outputs.parent.name != "base":raise ValueError("This script writes base results; choose the matching iteration script")
    if cfg.paths.outputs.name != "us_large":raise ValueError("Base Static-ML* applies only to US high-cap")
    p=cfg.paths.processed
    predictions=pd.read_parquet(p/"predictions.parquet")[["id","eom","pred_ld1"]]
    panel=pd.read_parquet(p/"panel.parquet").merge(predictions,on=["id","eom"],how="left",validate="one_to_one")
    wealth=pd.read_parquet(p/"wealth.parquet");cov=load_covariance(p/"covariance")
    formation=pd.DatetimeIndex(sorted(cov));returns=formation+pd.offsets.MonthEnd(1);start,end=evaluation_window(cfg.dates)
    history=formation[(returns>=pd.Timestamp(cfg.return_model.pred_years[0],1,31))&(returns<=end)]
    evaluation=formation[(returns>=start)&(returns<=end)]
    grid=hyper_grid(cfg.static_ml.k_grid,cfg.static_ml.u_grid,cfg.static_ml.g_grid)
    result=run_static_ml_star(panel,wealth,cov,history,evaluation,cfg.investor.gamma,grid,cfg.portfolio.charge_exit_tc)
    out=cfg.paths.outputs/"static_ml_star";out.mkdir(parents=True,exist_ok=True)
    for name,frame in result.items():
        if name=="chosen":frame.to_csv(out/"chosen.csv",index=False)
        else:frame.to_parquet(out/f"{name}.parquet",index=False)
    print("Static-ML*: k/u/g selected each December on cumulative preceding utility; applied to the next return year.")
    print(result["chosen"].to_string(index=False))
    print(table2(result["pf"],cfg.investor.gamma,cfg.dates).to_string())
    import json
    tuning_grid_marker(cfg).write_text(json.dumps(dict(method="Static-ML*",candidates=len(grid),
        k_grid=cfg.static_ml.k_grid,u_grid=cfg.static_ml.u_grid,g_grid=cfg.static_ml.g_grid)),encoding="utf-8")

if __name__=="__main__":main()
