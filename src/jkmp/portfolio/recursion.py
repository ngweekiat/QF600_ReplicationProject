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
    rows, drifted = [], None
    for eom in sorted(inputs):
        x = inputs[eom]
        start = start_weights_for(eom, x.ids, drifted, x.panel_t, charge_exit_tc)
        held = start[~start["exit"]].set_index("id").loc[x.ids, "w_start"].to_numpy()
        w = target(eom, x, held)
        if w.shape != held.shape or not np.isfinite(w).all():
            raise FloatingPointError(f"bad target weights at {eom.date()}")
        month = start.assign(eom=eom, w=0.0)
        keep = ~month["exit"]
        month.loc[keep, "w"] = pd.Series(w, index=x.ids).loc[month.loc[keep, "id"]].to_numpy()
        rows.append(month)
        drifted = drift_weights(month[month["w"] != 0.0], panel, wealth)[["id", "w_start"]]
    return pd.concat(rows, ignore_index=True)[["id", "eom", "w", "w_start", "exit"]]
