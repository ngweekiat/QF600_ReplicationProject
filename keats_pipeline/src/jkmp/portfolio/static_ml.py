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
    # All fixed settings share the same monthly membership and observed returns.
    # Batch their inherited weights; this avoids thousands of identical pandas joins.
    # Each (k,g) Cholesky factor is shared by its u settings; formulas are unchanged.
    dates = sorted(inputs)
    count = len(grid)
    if not count:
        raise ValueError("Static-ML validation needs a nonempty hyperparameter grid")
    groups = {}
    for index, (k, u, g) in enumerate(grid):
        groups.setdefault((k, g), []).append(index)
    panels = {date: group.set_index("id") for date, group in panel.loc[
        panel["eom"].isin(dates), ["id", "eom", "me", "lambda", "ret_ld1", "tr_ld1"]].groupby("eom", sort=False)}
    market_growth = wealth.set_index("eom")["mu_ld1"]
    previous_ids, inherited, out = None, None, []
    for date in dates:
        x = inputs[date]
        monthly = panels[date]
        current = monthly.loc[x.ids]
        if previous_ids is None:
            capital = current["me"].to_numpy()
            held = np.broadcast_to((capital/capital.sum())[:,None], (len(x.ids), count)).copy()
            exit_turnover = np.zeros(count)
            exit_cost = np.zeros(count)
        else:
            positions = pd.Index(previous_ids).get_indexer(x.ids)
            held = np.zeros((len(x.ids), count))
            found = positions >= 0
            held[found] = inherited[positions[found]]
            exits = ~pd.Index(previous_ids).isin(x.ids)
            exits &= pd.Index(previous_ids).isin(monthly.index)
            exits &= np.any(inherited != 0.0, axis=1)
            if charge_exit_tc and exits.any():
                sold = inherited[exits]
                sale_lambda = monthly.loc[previous_ids[exits], "lambda"].to_numpy()
                if np.isnan(sale_lambda).any():
                    raise ValueError("Missing lambda on exiting positions")
                exit_turnover = np.abs(sold).sum(axis=0)
                exit_cost = (sale_lambda[:,None] * sold**2).sum(axis=0)
            else:
                exit_turnover = np.zeros(count)
                exit_cost = np.zeros(count)
        weights = np.empty_like(held)
        for (k,g), indices in groups.items():
            lam_w = x.wealth * k * x.lam
            matrix = sigma_gam_adj(gamma * x.sigma, g) + np.diag(lam_w)
            u_values = np.array([grid[i][1] for i in indices])
            rhs = x.mu[:,None] * u_values + lam_w[:,None] * held[:,indices]
            weights[:,indices] = cho_solve(cho_factor(matrix), rhs)
        target_returns = current["ret_ld1"].to_numpy()
        total_returns = current["tr_ld1"].to_numpy()
        growth = float(market_growth.loc[date])
        if (not np.isfinite(weights).all() or np.isnan(target_returns).any()
                or np.isnan(total_returns).any() or not np.isfinite(growth)):
            raise ValueError(f"Missing/nonfinite Static-ML validation inputs at {date.date()}")
        trades = weights-held
        month = pd.DataFrame({"eom": date, "eom_ret": date+MONTH,
            "inv": np.abs(weights).sum(axis=0), "shorting": -weights.clip(max=0).sum(axis=0),
            "turnover": np.abs(trades).sum(axis=0)+exit_turnover,
            "r": (weights*target_returns[:,None]).sum(axis=0),
            "tc": .5*x.wealth*((x.lam[:,None]*trades**2).sum(axis=0)+exit_cost),
            "hp_no": np.arange(count), "k": [hp[0] for hp in grid],
            "u": [hp[1] for hp in grid], "g": [hp[2] for hp in grid]})
        out.append(month)
        inherited = weights * ((1+total_returns)/(1+growth))[:,None]
        previous_ids = x.ids
        if date.month == 12:
            log.info("Static-ML validation %s: %d hyperparameter settings", date.date(), count)
    return pd.concat(out, ignore_index=True).sort_values(["hp_no", "eom"], ignore_index=True)


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
    # Every traded or exiting name must occur in one of these monthly universes.
    # Keep all rows for those names/dates (including non-members used for exits),
    # rather than repeatedly joining the entire screened regional training panel.
    held_ids = np.unique(np.concatenate([item.ids for item in inputs.values()]))
    panel = panel.loc[panel["id"].isin(held_ids) & panel["eom"].isin(inputs)].reset_index(drop=True)
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
