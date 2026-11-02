"""Run only the original fixed-parameter Static-ML strategy."""
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
from jkmp.portfolio.static_ml import run_static_ml


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config");parser.add_argument("--override",action="append",default=[])
    args=parser.parse_args();cfg=load_config(args.config,args.override)
    if cfg.paths.outputs.parent.name != "base":raise ValueError("This script writes base results; choose the matching iteration script")
    p=cfg.paths.processed
    predictions=pd.read_parquet(p/"predictions.parquet")[["id","eom","pred_ld1"]]
    panel=pd.read_parquet(p/"panel.parquet").merge(predictions,on=["id","eom"],how="left",validate="one_to_one")
    wealth=pd.read_parquet(p/"wealth.parquet");cov=load_covariance(p/"covariance")
    formation=pd.DatetimeIndex(sorted(cov));returns=formation+pd.offsets.MonthEnd(1);start,end=evaluation_window(cfg.dates)
    history=formation[(returns>=pd.Timestamp(cfg.return_model.pred_years[0],1,31))&(returns<=end)]
    evaluation=formation[(returns>=start)&(returns<=end)]
    result=run_static_ml(panel,wealth,cov,history,evaluation,cfg.investor.gamma,cfg.portfolio.charge_exit_tc)
    out=cfg.paths.outputs/"static_ml";out.mkdir(parents=True,exist_ok=True)
    for name,frame in result.items():frame.to_parquet(out/f"{name}.parquet",index=False)
    print("Static-ML uses fixed (k,u,g)=(1,1,0); annual return-forecast refits and monthly portfolio updates.")
    print(table2(result["pf"],cfg.investor.gamma,cfg.dates).to_string())

if __name__=="__main__":main()
