"""Weight bookkeeping and trading costs. Port of ``w_fun`` (0 - Portfolio choice functions.R) and
``pf_ts_fun`` (0 - General functions.R).

Every portfolio method produces target weights ``w`` per formation month t. Trading happens from the
*inherited* weights ``w_start``: in the first month the value-weighted universe portfolio (R:
``initial_weights_new(w_type = "vw")``); afterwards last month's weights drifted by
``(1 + tr_ld1) / (1 + mu_ld1)`` (Eqs. 5–6). Stocks without a previous weight start at 0.

Exits: R drops a stock that leaves the universe, so selling it is free. With ``charge_exit_tc`` we add
a row ``w = 0`` for every stock held at t−1 that is not in the universe at t but still has a panel
row at t (it still trades, so its Λ_t is known). A stock with no row at t delisted or left the data:
its position was settled by the delisting return, so there is no sale and no cost.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MONTH = pd.offsets.MonthEnd(1)


def drift_weights(weights: pd.DataFrame, panel: pd.DataFrame, wealth: pd.DataFrame) -> pd.DataFrame:
    """Weights at t (``id, eom, w``) grown to the start of t+1: ``w·(1 + tr_ld1)/(1 + mu_ld1)``.

    Returns ``id, eom, w_start`` with ``eom`` = t+1.
    """
    d = (weights[["id", "eom", "w"]]
         .merge(panel[["id", "eom", "tr_ld1"]], on=["id", "eom"], how="left")
         .merge(wealth[["eom", "mu_ld1"]], on="eom", how="left"))
    if d[["tr_ld1", "mu_ld1"]].isna().any().any():
        bad = d[d[["tr_ld1", "mu_ld1"]].isna().any(axis=1)]
        raise ValueError(f"{len(bad)} held positions lack tr_ld1/mu_ld1, e.g. {bad.iloc[0][['id', 'eom']].tolist()}")
    return pd.DataFrame({"id": d["id"], "eom": d["eom"] + MONTH,
                         "w_start": d["w"] * (1.0 + d["tr_ld1"]) / (1.0 + d["mu_ld1"])})


def start_weights_for(eom: pd.Timestamp, universe_ids: np.ndarray, drifted: pd.DataFrame | None,
                      panel_t: pd.DataFrame, charge_exit_tc: bool) -> pd.DataFrame:
    """Inherited weights at one month for the universe (+ exit rows): ``id, w_start, exit``.

    ``drifted`` holds last month's drifted weights (``id, w_start``); ``None`` in the first month, when
    the start is the value-weighted universe. ``panel_t`` is the panel at ``eom`` (``id, me``).
    """
    uni = pd.DataFrame({"id": np.asarray(universe_ids)})
    if drifted is None:
        me = uni.merge(panel_t[["id", "me"]], on="id", how="left")["me"]
        return uni.assign(w_start=(me / me.sum()).to_numpy(), exit=False)
    out = uni.merge(drifted[["id", "w_start"]], on="id", how="left").fillna({"w_start": 0.0}).assign(exit=False)
    leaving = drifted[~drifted["id"].isin(uni["id"]) & (drifted["w_start"] != 0.0)]
    if charge_exit_tc and len(leaving):
        trades = leaving["id"].isin(panel_t["id"])
        out = pd.concat([out, leaving.loc[trades, ["id", "w_start"]].assign(exit=True)], ignore_index=True)
    return out


def add_start_weights(targets: pd.DataFrame, panel: pd.DataFrame, wealth: pd.DataFrame,
                      dates: Sequence[pd.Timestamp], charge_exit_tc: bool) -> pd.DataFrame:
    """R ``w_fun`` for methods whose targets do not depend on inherited weights.

    ``targets``: ``id, eom, w`` for the universe at each date (consecutive month-ends). Returns
    ``id, eom, w, w_start, exit``; exit rows (``w = 0``) are added only with ``charge_exit_tc``.
    """
    dates = pd.DatetimeIndex(sorted(dates))
    if len(dates) > 1 and not (dates[1:] == dates[:-1] + MONTH).all():
        raise ValueError("dates must be consecutive month-ends")
    tg = targets.loc[targets["eom"].isin(dates), ["id", "eom", "w"]]
    drifted = drift_weights(tg[tg["eom"] < dates[-1]], panel, wealth)
    drifted_by = dict(tuple(drifted.groupby("eom")))
    panel_by = dict(tuple(panel.loc[panel["eom"].isin(dates), ["id", "eom", "me"]].groupby("eom")))

    rows, n_charged, n_free = [], 0, 0
    for i, (eom, g) in enumerate(tg.groupby("eom", sort=True)):
        prev = None if i == 0 else drifted_by.get(eom, pd.DataFrame(columns=["id", "w_start"]))
        start = start_weights_for(eom, g["id"].to_numpy(), prev, panel_by[eom], charge_exit_tc)
        if prev is not None:
            leaving = prev[~prev["id"].isin(g["id"]) & (prev["w_start"] != 0.0)]
            charged = int(start["exit"].sum())
            n_charged += charged
            n_free += len(leaving) - charged
        rows.append(start.merge(g[["id", "w"]], on="id", how="left").fillna({"w": 0.0}).assign(eom=eom))
    out = pd.concat(rows, ignore_index=True)[["id", "eom", "w", "w_start", "exit"]]
    log.info("Exits: %d charged as sales, %d without a sale (delisted/left data or charge_exit_tc off)",
             n_charged, n_free)
    return out


def portfolio_returns(weights: pd.DataFrame, panel: pd.DataFrame, wealth: pd.DataFrame) -> pd.DataFrame:
    """R ``pf_ts_fun``: per formation month t, leverage, shorting, turnover, gross return over t+1 and
    trading cost ``wealth_t/2 · Σ λ_i (w_i − w_start_i)²`` (fraction of wealth). Row dated ``eom_ret = t+1``.
    """
    c = (weights.merge(panel[["id", "eom", "ret_ld1", "lambda"]], on=["id", "eom"], how="left")
         .merge(wealth[["eom", "wealth"]], on="eom", how="left"))
    held = ~c["exit"]
    if c.loc[held, ["ret_ld1", "lambda"]].isna().any().any() or c["lambda"].isna().any() or c["wealth"].isna().any():
        raise ValueError("missing ret_ld1, lambda or wealth for some weights")
    c["ret_ld1"] = c["ret_ld1"].where(held, 0.0)  # exit rows: w = 0, nothing earned
    dw = c["w"] - c["w_start"]
    c = c.assign(abs_w=c["w"].abs(), short=c["w"].clip(upper=0).abs(), abs_dw=dw.abs(),
                 r_i=c["w"] * c["ret_ld1"], tc_i=0.5 * c["wealth"] * c["lambda"] * dw**2)
    pf = c.groupby("eom").agg(inv=("abs_w", "sum"), shorting=("short", "sum"), turnover=("abs_dw", "sum"),
                              r=("r_i", "sum"), tc=("tc_i", "sum")).reset_index()
    pf.insert(1, "eom_ret", pf["eom"] + MONTH)
    return pf
