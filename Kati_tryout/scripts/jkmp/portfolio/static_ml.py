"""Original Static-ML, with fixed cost/shrinkage/risk multipliers (1,1,0).

The return forecaster is annually refitted. Portfolio weights are updated monthly.
The portfolio recursion begins in 2015 and continues into the 2020-2025 test,
preserving the original non-starred strategy's inherited weights.
"""
from collections.abc import Mapping,Sequence
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from jkmp.covariance import CovParts
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.recursion import month_inputs,recursive_weights


def run_static_ml(panel: pd.DataFrame,wealth: pd.DataFrame,cov_parts: Mapping[pd.Timestamp,CovParts],
                  val_dates: Sequence[pd.Timestamp],eval_dates: Sequence[pd.Timestamp],gamma: float,
                  charge_exit_tc: bool) -> dict[str,pd.DataFrame]:
    history=pd.DatetimeIndex(sorted(val_dates));evaluation=pd.DatetimeIndex(sorted(eval_dates))
    if not set(evaluation).issubset(history):raise ValueError("Static evaluation must be within its continuous history")
    inputs=month_inputs(panel,wealth,cov_parts,history)
    ids=np.unique(np.concatenate([x.ids for x in inputs.values()]))
    panel=panel.loc[panel.id.isin(ids)&panel.eom.isin(history)].reset_index(drop=True)
    def target(date,x,held):
        cost=x.wealth*x.lam
        return cho_solve(cho_factor(gamma*x.sigma+np.diag(cost)),x.mu+cost*held)
    weights=recursive_weights(inputs,panel,wealth,target,charge_exit_tc).assign(type="Static-ML")
    oos=portfolio_returns(weights,panel,wealth).assign(type="Static-ML")
    return dict(pf=oos.loc[oos.eom.isin(evaluation)].reset_index(drop=True),
                weights=weights.loc[weights.eom.isin(evaluation)].reset_index(drop=True),
                pf_oos=oos,weights_oos=weights)
