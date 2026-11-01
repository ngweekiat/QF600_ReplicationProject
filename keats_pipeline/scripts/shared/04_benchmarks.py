"""Market comparator for the test and broader forecast-history windows."""
from pathlib import Path as _ScriptPath
import sys as _script_sys
_script_sys.path.insert(0,str(_ScriptPath(__file__).resolve().parents[1]))
from _bootstrap import configure
configure(__file__)
import argparse
import pandas as pd
from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.evaluation.holdout import evaluation_window,last_open_month
from jkmp.evaluation.metrics import table2
from jkmp.portfolio.benchmarks import run_benchmarks


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config");parser.add_argument("--override",action="append",default=[])
    args=parser.parse_args();cfg=load_config(args.config,args.override)
    if cfg.paths.outputs.parent.name != "iteration_1":raise ValueError("Market references belong to iteration_1; base has only two ML methods")
    p=cfg.paths.processed
    panel=pd.read_parquet(p/"panel.parquet");wealth=pd.read_parquet(p/"wealth.parquet");cov=load_covariance(p/"covariance")
    formation=pd.DatetimeIndex(sorted(cov));returns=formation+pd.offsets.MonthEnd(1)
    first,last=cfg.return_model.pred_years;start,end=evaluation_window(cfg.dates)
    windows={"test":formation[(returns>=start)&(returns<=end)],
             "oos":formation[(returns>=pd.Timestamp(first,1,31))&(returns<=min(pd.Timestamp(last,12,31),last_open_month(cfg.dates)))]}
    out=cfg.paths.outputs/"benchmarks";out.mkdir(parents=True,exist_ok=True)
    for label,dates in windows.items():
        weights,pf=run_benchmarks(panel,wealth,cov,dates,cfg.portfolio.charge_exit_tc)
        weights=weights.loc[weights.type.eq("Market")];pf=pf.loc[pf.type.eq("Market")]
        weights.to_parquet(out/f"weights_{label}.parquet",index=False);pf.to_parquet(out/f"pf_{label}.parquet",index=False)
        table=table2(pf,cfg.investor.gamma,cfg.dates);table.to_csv(out/f"table2_{label}.csv");print(label,table.to_string())

if __name__=="__main__":main()
