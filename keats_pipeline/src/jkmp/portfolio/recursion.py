"""Month-by-month weights for methods whose target depends on the inherited weights (Static-ML, Portfolio-ML).

R ``static_val_fun`` and ``pfml_w`` share this loop: start from value weights in the first month, then
from last month's weights drifted by (1 + tr_ld1)/(1 + mu_ld1), new stocks at 0 (``accounting``); the
method maps (month, inherited weights of the universe) to new weights.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from jkmp.covariance import CovParts
from jkmp.portfolio.accounting import drift_weights, start_weights_for


@dataclass
class MonthInputs:
    ids: np.ndarray        # universe ids, sorted
    sigma: np.ndarray      # monthly Σ for ids
    lam: np.ndarray        # Λ diagonal for ids
    mu: np.ndarray | None  # pred_ld1 for ids (None if not requested)
    wealth: float
    panel_t: pd.DataFrame  # id, me at t (for the VW start and exit detection)


def month_inputs(panel: pd.DataFrame, wealth: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts],
                 dates: Sequence[pd.Timestamp], mu_col: str | None = "pred_ld1") -> dict[pd.Timestamp, MonthInputs]:
    """Per-month arrays for the universe, built once and shared by every run over the same months."""
    w_by = wealth.set_index("eom")["wealth"]
    p = panel[panel["eom"].isin(dates)]
    out = {}
    for eom, g in p.groupby("eom", sort=True):
        u = g[g["in_universe"]].sort_values("id")
        if mu_col is not None and u[mu_col].isna().any():
            raise ValueError(f"missing {mu_col} in the universe at {eom.date()}")
        ids = u["id"].to_numpy()
        out[eom] = MonthInputs(ids, cov_parts[eom].full(ids), u["lambda"].to_numpy(),
                               None if mu_col is None else u[mu_col].to_numpy(), float(w_by[eom]), g[["id", "me"]])
    return out


Target = Callable[[pd.Timestamp, MonthInputs, np.ndarray], np.ndarray]


def recursive_weights(inputs: Mapping[pd.Timestamp, MonthInputs], panel: pd.DataFrame, wealth: pd.DataFrame,
                      target: Target, charge_exit_tc: bool) -> pd.DataFrame:
    """Run ``target(eom, inputs, w_start of the universe) -> w`` over the months in ``inputs`` (consecutive).

    Returns ``id, eom, w, w_start, exit`` (exit rows only with ``charge_exit_tc``), ready for
    ``accounting.portfolio_returns``.
    """
    rows, previous_ids, previous_start = [], None, None
    # Index once: repeated counterfactual/grid runs must not rejoin the full panel
    # for every individual month. Initial weights, drift and exit rules match accounting.py.
    panel_returns = {date: group.set_index("id")["tr_ld1"] for date, group in
                     panel.loc[panel["eom"].isin(inputs), ["id","eom","tr_ld1"]].groupby("eom", sort=False)}
    market = wealth.set_index("eom")["mu_ld1"]
    for eom in sorted(inputs):
        x = inputs[eom]
        exit_ids, exit_start = np.array([],dtype=x.ids.dtype), np.array([],dtype=float)
        if previous_ids is None:
            capital=x.panel_t.set_index("id")["me"].reindex(x.ids).to_numpy()
            held=capital/capital.sum()
        else:
            positions=pd.Index(previous_ids).get_indexer(x.ids)
            held=np.zeros(len(x.ids))
            found=positions>=0
            held[found]=previous_start[positions[found]]
            if charge_exit_tc:
                leaving=(~pd.Index(previous_ids).isin(x.ids)) & (previous_start!=0.0)
                leaving &= pd.Index(previous_ids).isin(x.panel_t["id"])
                exit_ids,exit_start=previous_ids[leaving],previous_start[leaving]
        w = target(eom, x, held)
        if w.shape != held.shape or not np.isfinite(w).all():
            raise FloatingPointError(f"bad target weights at {eom.date()}")
        month=pd.DataFrame({"id":np.concatenate([x.ids,exit_ids]),"eom":eom,
            "w":np.concatenate([w,np.zeros(len(exit_ids))]),
            "w_start":np.concatenate([held,exit_start]),
            "exit":np.concatenate([np.zeros(len(x.ids),dtype=bool),np.ones(len(exit_ids),dtype=bool)])})
        rows.append(month)
        nonzero=w!=0.0
        previous_ids=x.ids[nonzero]
        if nonzero.any():
            returns=panel_returns[eom].reindex(previous_ids).to_numpy()
            growth=market.get(eom,np.nan)
            if pd.isna(returns).any() or pd.isna(growth):
                raise ValueError(f"Held positions lack tr_ld1/mu_ld1 at {eom.date()}")
            previous_start=w[nonzero]*(1.0+returns)/(1.0+growth)
        else:
            previous_start=np.array([],dtype=float)
    return pd.concat(rows, ignore_index=True)[["id", "eom", "w", "w_start", "exit"]]
