"""Static-ML and Static-ML* (Eq. 34 with the Eq. 41 tuning layer). Port of ``static_val_fun``,
``static_implement`` (0 - Portfolio choice functions.R) and ``sigma_gam_adj`` (0 - General functions.R).

Each month, given inherited weights w_start (VW in the first month, then drifted):

    w_t = (γΣ* + W·kΛ)⁻¹ (u·μ_t + W·kΛ·w_start),     γΣ* = γΣ + g·diag(γΣ)   (``cov_add``)

u shrinks the predicted returns ``pred_ld1``, g inflates the variances, k = 1/φ scales trading costs.
Static-ML is (k, u, g) = (1, 1, 0); Static-ML* picks (k, u, g) from the grid each December by the
highest cumulative realised utility since the validation start and uses it for the next return year.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from jkmp.covariance import CovParts
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.recursion import MonthInputs, month_inputs, recursive_weights

log = logging.getLogger(__name__)

MONTH = pd.offsets.MonthEnd(1)
Hyper = tuple[float, float, float]  # (k, u, g)


def sigma_gam_adj(sigma_gam: np.ndarray, g: float, cov_type: str = "cov_add") -> np.ndarray:
    """R ``sigma_gam_adj``: ``cov_add`` adds g × the diagonal; ``cov_mult`` scales by g."""
    if cov_type == "cov_add":
        return sigma_gam + np.diag(np.diag(sigma_gam) * g)
    if cov_type == "cov_mult":
        return sigma_gam * g
    raise ValueError(f"unsupported cov_type {cov_type!r}")


def static_weights(inputs: Mapping[pd.Timestamp, MonthInputs], panel: pd.DataFrame, wealth: pd.DataFrame,
                   gamma: float, hyper_for: Callable[[pd.Timestamp], Hyper], charge_exit_tc: bool) -> pd.DataFrame:
    """Recursive Static-ML weights over the months in ``inputs``: ``id, eom, w, w_start, exit``."""
    def target(eom: pd.Timestamp, x: MonthInputs, held: np.ndarray) -> np.ndarray:
        k, u, g = hyper_for(eom)
        lam_w = x.wealth * k * x.lam
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked in recursive_weights
            a = sigma_gam_adj(gamma * x.sigma, g) + np.diag(lam_w)
            return cho_solve(cho_factor(a), u * x.mu + lam_w * held)

    return recursive_weights(inputs, panel, wealth, target, charge_exit_tc)


def hyper_grid(k_grid: Sequence[float], u_grid: Sequence[float], g_grid: Sequence[float]) -> list[Hyper]:
    return [(float(k), float(u), float(g)) for k, u, g in itertools.product(k_grid, u_grid, g_grid)]


def validate(inputs, panel, wealth, gamma, grid: Sequence[Hyper], charge_exit_tc) -> pd.DataFrame:
    """Monthly performance for every fixed (k, u, g) over the validation months (one run each, VW start)."""
    out = []
    for i, hp in enumerate(grid):
        w = static_weights(inputs, panel, wealth, gamma, lambda _eom, hp=hp: hp, charge_exit_tc)
        out.append(portfolio_returns(w, panel, wealth).assign(hp_no=i, k=hp[0], u=hp[1], g=hp[2]))
    return pd.concat(out, ignore_index=True)


def select_hypers(validation: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Best (k, u, g) at each December by cumulative realised utility (R ``static_implement``).

    cum_var = cummean(r²) − cummean(r)²;  cum_obj = cummean(r − tc − ½γ·cum_var); rank by eom_ret.
    The December-Y choice is used for return year Y+1 (column ``year``).
    """
    v = validation.sort_values(["hp_no", "eom_ret"]).reset_index(drop=True)  # idxmax needs unique labels
    grp = v.groupby("hp_no")
    mean_r = grp["r"].transform(lambda s: s.expanding().mean())
    mean_r2 = (v["r"] ** 2).groupby(v["hp_no"]).transform(lambda s: s.expanding().mean())
    v["cum_var"] = mean_r2 - mean_r**2
    flow = v["r"] - v["tc"] - 0.5 * v["cum_var"] * gamma
    v["cum_obj"] = flow.groupby(v["hp_no"]).transform(lambda s: s.expanding().mean())
    dec = v[v["eom_ret"].dt.month == 12]
    best = dec.loc[dec.groupby("eom_ret")["cum_obj"].idxmax()]
    return best.assign(year=best["eom_ret"].dt.year + 1)[["year", "eom_ret", "hp_no", "k", "u", "g", "cum_obj"]].reset_index(drop=True)


def run_static_ml(panel: pd.DataFrame, wealth: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts],
                  val_dates: Sequence[pd.Timestamp], eval_dates: Sequence[pd.Timestamp], gamma: float,
                  grid: Sequence[Hyper], charge_exit_tc: bool) -> dict[str, pd.DataFrame]:
    """Validation over ``val_dates``; Static-ML (1, 1, 0) and Static-ML* (yearly-chosen) over ``eval_dates``.

    Static-ML is the (1, 1, 0) validation path restricted to the evaluation months (as R does);
    Static-ML* is re-run over ``eval_dates`` from VW initial weights with the chosen hyperparameters.
    """
    inputs = month_inputs(panel, wealth, cov_parts, sorted(set(val_dates) | set(eval_dates)))
    val_inputs = {d: inputs[d] for d in sorted(val_dates)}
    validation = validate(val_inputs, panel, wealth, gamma, grid, charge_exit_tc)
    chosen = select_hypers(validation, gamma)
    by_year = {int(r.year): (r.k, r.u, r.g) for r in chosen.itertuples()}

    def hyper_for(eom: pd.Timestamp) -> Hyper:
        year = (eom + MONTH).year  # return year of formation month eom
        if year not in by_year:
            raise ValueError(f"no Static-ML* hyperparameters chosen for return year {year}")
        return by_year[year]

    eval_dates = pd.DatetimeIndex(sorted(eval_dates))
    w_star = static_weights({d: inputs[d] for d in eval_dates}, panel, wealth, gamma, hyper_for, charge_exit_tc)
    pf_star = portfolio_returns(w_star, panel, wealth).assign(type="Static-ML*")
    base_no = grid.index((1.0, 1.0, 0.0))
    pf_base = validation[(validation["hp_no"] == base_no) & validation["eom"].isin(eval_dates)]
    pf_base = pf_base.drop(columns=["hp_no", "k", "u", "g"]).assign(type="Static-ML")
    log.info("Static-ML*: chose\n%s", chosen.to_string(index=False))
    return {"validation": validation, "chosen": chosen, "weights": w_star.assign(type="Static-ML*"),
            "pf": pd.concat([pf_base, pf_star], ignore_index=True)}
