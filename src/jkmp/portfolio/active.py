"""Benchmark-relative (active) Portfolio-ML — our extension (docs/deviations.md D34, research log trial 2).

π_t = π^VW_t + a_t with a_t the Portfolio-ML aim/recursion on active weights. The objective is the
Portfolio-ML objective with the active risk aversion γ_TE in place of γ, so the inputs (r̃ = Ωᵀr, the active
return of each signal; Σ̃ = γ_TE·ΩᵀΣΩ + wΔΩᵀΛΔΩ) and m come from the unchanged functions. Runs use the
yearly sums of ``grids`` (same β and December selection as scripts 06–07, without monthly files).

γ_TE is calibrated to ex-ante tracking-error budgets **before** the test window: each grid value is run with hp
years up to the end of validation only, and the mean ex-ante TE of the implemented active weights over those
validation return months is interpolated (log TE against log γ_TE) at each budget.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from jkmp.config import Config
from jkmp.evaluation.metrics import active_weights, ex_ante_exposures
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.grids import GridInputs, _signal_kwargs
from jkmp.portfolio.portfolio_ml import (SearchResult, aim_portfolios, build_yearly_moments, pfml_weights,
                                         run_portfolio_ml, search_yearly, signal_set)
from jkmp.portfolio.recursion import month_inputs

log = logging.getLogger(__name__)

LABEL = "Portfolio-ML (active)"


def active_search(inp: GridInputs, cfg: Config, gamma_te: float, signal_dates: Sequence[pd.Timestamp],
                  hp_years: Sequence[int]) -> dict[str, SearchResult]:
    """The coefficient search of active Portfolio-ML at ``gamma_te`` for every signal set, with signals kept for
    ``signal_dates`` (the aim months). ``run_active`` selects from it; trial 6 executes every candidate in it."""
    if not cfg.portfolio_ml.benchmark_relative:
        raise ValueError("active Portfolio-ML needs portfolio_ml.benchmark_relative: true (configs/experiments/active_pfml.yaml)")
    return pfml_search(inp, cfg, gamma_te, signal_dates, hp_years)


def pfml_search(inp: GridInputs, cfg: Config, gamma: float, signal_dates: Sequence[pd.Timestamp],
                hp_years: Sequence[int]) -> dict[str, SearchResult]:
    """The yearly-sums coefficient search of Portfolio-ML (total-return or active, as ``cfg.portfolio_ml`` says) at
    risk aversion ``gamma``, with signals kept for ``signal_dates``."""
    pm = cfg.portfolio_ml
    _, tags, _ = signal_set(pm.use_rff, pm.rff_variance_grid)
    results = {}
    for tag, g in zip(tags, pm.rff_variance_grid if pm.use_rff else [None]):
        ym = build_yearly_moments(inp.panel, inp.wealth, inp.cov, inp.rf, inp.pfml_dates, cfg.features.names,
                                  signal_dates=pd.DatetimeIndex(sorted(signal_dates)), gamma=gamma,
                                  mu=cfg.investor.mu, lb=cfg.universe.lookback_months,
                                  iterations=cfg.investor.m_iterations, scale_by_vol=pm.scale_by_vol,
                                  add_constant=pm.add_constant, **_signal_kwargs(cfg, g))
        results[tag] = search_yearly(ym, tag, pm.p_grid, cfg.ridge_grid.values, hp_years, pm.half_life_years)
    return results


def active_with_choices(results: Mapping[str, SearchResult], chosen: pd.DataFrame, inp: GridInputs, cfg: Config,
                        gamma_te: float, eval_dates: Sequence[pd.Timestamp], label: str) -> dict[str, pd.DataFrame]:
    """Active Portfolio-ML over ``eval_dates`` (from value weights) with the yearly (tag, p, λ) of ``chosen`` (columns
    year, tag, p, l_no, l, h) instead of the original December selection; aims, recursion and accounting unchanged."""
    eval_dates = pd.DatetimeIndex(sorted(eval_dates))
    aims, coefs = aim_portfolios(results, chosen, eval_dates)
    inputs = month_inputs(inp.panel, inp.wealth, inp.cov, eval_dates, mu_col=None)
    weights = pfml_weights(inputs, aims, inp.panel, inp.wealth, inp.rf, gamma=gamma_te, mu=cfg.investor.mu,
                           iterations=cfg.investor.m_iterations, charge_exit_tc=cfg.portfolio.charge_exit_tc,
                           benchmark_relative=True)
    pf = portfolio_returns(weights, inp.panel, inp.wealth).assign(type=label)
    return {"aims": aims, "aim_coefs": coefs, "weights": weights.assign(type=label), "pf": pf}


def run_active(inp: GridInputs, cfg: Config, gamma_te: float, eval_dates: Sequence[pd.Timestamp],
               hp_years: Sequence[int]) -> dict[str, pd.DataFrame]:
    """Active Portfolio-ML at ``gamma_te``: search (g, p, λ) over ``hp_years``, build aims for ``eval_dates`` and
    run the active recursion over them from VW initial weights (``run_portfolio_ml`` outputs), as scripts/07 runs
    the baseline over the test window."""
    eval_dates = pd.DatetimeIndex(sorted(eval_dates))
    results = active_search(inp, cfg, gamma_te, eval_dates, hp_years)
    return run_portfolio_ml(results, inp.panel, inp.wealth, inp.cov, inp.rf, eval_dates, eval_dates, gamma=gamma_te,
                            mu=cfg.investor.mu, iterations=cfg.investor.m_iterations,
                            charge_exit_tc=cfg.portfolio.charge_exit_tc, label=LABEL, benchmark_relative=True)


def calibration_point(inp: GridInputs, cfg: Config, gamma_te: float) -> dict:
    """Mean ex-ante TE (annual) of the active weights over the validation return months, hp years ≤ validation end.
    Touches no return after ``dates.validation_end`` (the yearly sums of later years are never used)."""
    d = cfg.dates
    formation = pd.DatetimeIndex(sorted(inp.cov))
    ret = formation + pd.offsets.MonthEnd(1)
    hp_years = [y for y in inp.hp_years if y <= d.validation_end.year]
    aim_dates = formation[(ret.year > hp_years[0]) & (ret <= pd.Timestamp(d.validation_end))]
    res = run_active(inp, cfg, gamma_te, aim_dates, hp_years)
    ex = ex_ante_exposures(active_weights(res["weights"], inp.panel), inp.cov)
    return {"gamma_te": gamma_te, "months": len(aim_dates), "first_ret": (aim_dates[0] + pd.offsets.MonthEnd(1)).date(),
            "te_ex_ante": float(ex["te"].mean() * np.sqrt(12))}


def calibrate(points: pd.DataFrame, targets: Sequence[float]) -> pd.DataFrame:
    """γ_TE for each TE budget by linear interpolation of log γ_TE on log TE (``points``: gamma_te, te_ex_ante).
    Raises if a budget lies outside the TE range spanned by the grid (extend ``investor.gamma_te_grid``)."""
    p = points.sort_values("te_ex_ante")
    x, y = np.log(p["te_ex_ante"].to_numpy()), np.log(p["gamma_te"].to_numpy())
    if not (np.diff(y) < 0).all():
        log.warning("ex-ante TE is not monotone in gamma_te over the grid:\n%s", p.to_string(index=False))
    out = []
    for t in targets:
        if not x[0] <= np.log(t) <= x[-1]:
            raise ValueError(f"TE budget {t:.3f} outside the grid's range "
                             f"{p['te_ex_ante'].min():.4f} .. {p['te_ex_ante'].max():.4f}")
        # 4 significant figures: the value passed to the run (--override investor.gamma_te) and its folder name
        out.append({"te_target": t, "gamma_te": float(f"{np.exp(np.interp(np.log(t), x, y)):.4g}")})
    return pd.DataFrame(out)
