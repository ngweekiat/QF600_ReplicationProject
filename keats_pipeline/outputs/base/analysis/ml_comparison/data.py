"""Only the three retained base strategies, with strict monthly alignment."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from jkmp.config import load_config
from jkmp.strategies import STRATEGIES,require_strategies

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def align(monthly: pd.DataFrame):
    """Common months per dataset, and then across datasets; never fill absent returns."""
    if monthly.empty:
        raise ValueError('No performance observations to align.')
    monthly=monthly.copy()
    monthly['eom_ret']=pd.to_datetime(monthly.eom_ret)
    if monthly[['dataset','strategy','eom_ret']].isna().any().any():
        raise ValueError('Missing dataset, strategy or return month.')
    if monthly.duplicated(['dataset','strategy','eom_ret']).any():
        raise ValueError('Duplicate dataset/strategy/return-month observations.')
    if not monthly.eom_ret.dt.is_month_end.all():
        raise ValueError('Saved monthly returns must be dated at month-end.')
    aligned, logs = [], []
    for dataset, group in monthly.groupby('dataset', sort=False):
        date_sets = [set(g.eom_ret) for _, g in group.groupby('strategy')]
        common = set.intersection(*date_sets)
        if len(common) < 12:
            raise ValueError(f'Fewer than 12 common observations: {dataset}')
        ordered=pd.DatetimeIndex(sorted(common)).to_period('M').asi8
        if not np.all(np.diff(ordered)==1):
            raise ValueError(f'Calendar gaps in common monthly sample: {dataset}; no missing return is fabricated.')
        for strategy, g in group.groupby('strategy'):
            drop = g.loc[~g.eom_ret.isin(common)]
            logs.append(dict(dataset=dataset, strategy=strategy, original_months=len(g),
                common_months=len(common), excluded_months=len(drop),
                excluded_dates=';'.join(drop.eom_ret.dt.strftime('%Y-%m-%d'))))
        aligned.append(group.loc[group.eom_ret.isin(common)])
    within = pd.concat(aligned, ignore_index=True).sort_values(['dataset','strategy','eom_ret'])
    common_all = set.intersection(*(set(g.eom_ret) for _, g in within.groupby('dataset')))
    across = within.loc[within.eom_ret.isin(common_all)].copy()
    return within, across, pd.DataFrame(logs)



def discover(root,universes=None):
    frames=[];inventory=[]
    for universe in universes or ("us_large","us_mid","eu_large"):
        cfg=load_config(root/"keats_pipeline/configs"/universe/"base.yaml")
        output=cfg.paths.outputs
        paths=(("Static-ML",output/"static_ml/pf.parquet",output/"static_ml/weights.parquet"),
               ("Portfolio-ML",output/"portfolio_ml/rff/pf.parquet",output/"portfolio_ml/rff/weights.parquet"))
        for strategy,pf_path,weight_path in paths:
            pf=pd.read_parquet(pf_path)
            if set(pf.type)!={strategy}:raise ValueError(f"Unexpected strategy in {pf_path}")
            pf.eom=pd.to_datetime(pf.eom);pf.eom_ret=pd.to_datetime(pf.eom_ret)
            expected=pd.date_range(cfg.dates.test_start,cfg.dates.test_end,freq="ME")
            if not pd.DatetimeIndex(sorted(pf.eom_ret)).equals(expected):raise ValueError("Incomplete test calendar")
            if not (pf.eom+pd.offsets.MonthEnd(1)).eq(pf.eom_ret).all():raise ValueError("Formation/return mismatch")
            require_strategies(pf)
            frames.append(pf.assign(dataset=f"{universe}/base",strategy=strategy,universe=universe))
            inventory.append(dict(universe=universe,strategy=strategy,pf_path=str(pf_path),weights_path=str(weight_path),
                                  processed_path=str(cfg.paths.processed),sha256=sha256(pf_path)))
        require_strategies(pd.concat([f for f in frames if f.universe.iloc[0]==universe]),complete=True)
    return pd.DataFrame(inventory),pd.concat(frames,ignore_index=True)
