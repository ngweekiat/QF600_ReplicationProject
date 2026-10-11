"""Implementable efficient frontier grid: Static-ML* and Portfolio-ML re-tuned at every (γ, wealth).

R runs ``static_implement`` and ``pfml_implement`` from scratch for each (wealth, γ) job
(``slurm_build_portfolios.R`` with ``config_pfchoice_ief_*``; ``6 - Implementable efficient frontier.R``).
Both depend on (γ, w) through Σ̃, m and the cost terms, so every point re-tunes all hyperparameters.

Portfolio-ML inputs are cached per point as yearly sums (``build_yearly_moments``), which give exactly
the same β and December selection as the monthly matrices at ~1/12 the size. The wealth path is linear
in the end-of-sample wealth, so each level's path is the base path rescaled.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jkmp.config import Config
from jkmp.covariance import CovParts
from jkmp.evaluation.metrics import table2
from jkmp.portfolio.portfolio_ml import (build_yearly_moments, load_yearly_moments, run_portfolio_ml,
                                         save_yearly_moments, search_yearly, signal_set)
from jkmp.portfolio.static_ml import hyper_grid, run_static_ml

log = logging.getLogger(__name__)

MONTH = pd.offsets.MonthEnd(1)


@dataclass
class GridInputs:
    """Everything a grid point needs, small enough to ship to joblib workers."""
    panel: pd.DataFrame            # stocks ever in the universe, with pred_ld1
    wealth: pd.DataFrame           # base wealth path (cfg.investor.wealth at the end of the sample)
    cov: Mapping[pd.Timestamp, CovParts]
    rf: pd.Series                  # French monthly RF by month-end
    pfml_dates: pd.DatetimeIndex   # Portfolio-ML formation months (first Σ + lb + 1 .. test end − 1)
    val_dates: pd.DatetimeIndex    # Static-ML* validation formation months (returns from 2015)
    eval_dates: pd.DatetimeIndex   # test window formation months
    hp_years: list[int]


def grid_inputs(cfg: Config, panel: pd.DataFrame, wealth: pd.DataFrame, cov: Mapping[pd.Timestamp, CovParts],
                rf: pd.Series) -> GridInputs:
    """Date windows as in scripts 05 and 07; the panel cut to stocks that are ever in the universe
    (the only ones that are held, enter m and Σ̃, or are sold on exit)."""
    formation = pd.DatetimeIndex(sorted(cov))
    ret = formation + MONTH
    first_year = cfg.return_model.pred_years[0]
    test_start, test_end = pd.Timestamp(cfg.dates.test_start), pd.Timestamp(cfg.dates.test_end)
    lb = cfg.universe.lookback_months
    first = formation[0] + MONTH * (lb + 1)
    keep = panel["id"].isin(panel.loc[panel["in_universe"], "id"].unique())
    cols = ["id", "eom", "me", "in_universe", "lambda", "g", "tr_ld1", "ret_ld1", "pred_ld1", *cfg.features.names]
    return GridInputs(
        panel=panel.loc[keep & (panel["eom"] >= formation[0]), cols].reset_index(drop=True),
        wealth=wealth, cov=cov, rf=rf,
        pfml_dates=formation[(formation >= first) & (ret <= test_end)],
        val_dates=formation[(ret >= pd.Timestamp(first_year, 1, 31)) & (ret <= test_end)],
        eval_dates=formation[(ret >= test_start) & (ret <= test_end)],
        hp_years=list(range(first_year, test_end.year + 1)))


def point_key(gamma: float, wealth: float) -> str:
    return f"gamma{gamma:g}_wealth{wealth / 1e9:g}B"


def scaled_wealth(base: pd.DataFrame, level: float, base_level: float) -> pd.DataFrame:
    """Wealth path ending at ``level``: w_{t−1} = w_t/(1 + R_m) is linear in the end value."""
    return base.assign(wealth=base["wealth"] * (level / base_level))


def _signal_kwargs(cfg: Config, g: float | None) -> dict:
    """build_moments arguments for one signal set (RFF with variance g, or raw characteristics if g is None)."""
    pm = cfg.portfolio_ml
    if g is None:
        return dict(use_rff=False, rff_variance=None, p_max=None, seed=None, add_orig=False)
    return dict(use_rff=True, rff_variance=g, p_max=max(pm.p_grid), seed=cfg.seed, add_orig=pm.add_original_features)


def _pfml_moments(inp: GridInputs, cfg: Config, gamma: float, wealth: pd.DataFrame, level: float,
                  g: float | None, cache: Path):
    pm = cfg.portfolio_ml
    sig = _signal_kwargs(cfg, g)
    expected = {**sig, "features": list(cfg.features.names), "gamma": gamma, "mu": cfg.investor.mu,
                "lb": cfg.universe.lookback_months, "iterations": cfg.investor.m_iterations,
                "scale_by_vol": pm.scale_by_vol, "wealth_end": level,
                "first": f"{inp.pfml_dates[0]:%Y-%m-%d}", "last": f"{inp.pfml_dates[-1]:%Y-%m-%d}"}
    if cache.exists():
        ym = load_yearly_moments(cache)
        if ym.meta == expected and set(inp.eval_dates) <= set(ym.signals):
            return ym
        log.info("stale cache %s, rebuilding", cache)
    ym = build_yearly_moments(inp.panel, wealth, inp.cov, inp.rf, inp.pfml_dates, cfg.features.names,
                              signal_dates=inp.eval_dates, gamma=gamma, mu=cfg.investor.mu,
                              lb=cfg.universe.lookback_months, iterations=cfg.investor.m_iterations,
                              scale_by_vol=pm.scale_by_vol, **sig)
    ym.meta.update(wealth_end=level, first=expected["first"], last=expected["last"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    save_yearly_moments(ym, cache)
    return ym


def run_point(inp: GridInputs, cfg: Config, gamma: float, level: float, cache_root: Path, out_root: Path,
              force: bool = False) -> pd.DataFrame:
    """Static-ML, Static-ML* and Portfolio-ML (the signal set of ``portfolio_ml.use_rff``) at one (γ, wealth).

    Writes ``out_root/{key}/pf_static_ml.parquet`` and ``pf_portfolio_ml_{rff|raw}.parquet`` (+ chosen
    hyperparameters); Portfolio-ML moments go to ``cache_root/{key}/``. Existing files are reused unless
    ``force``, so a linear-signal run reuses the Static-ML results of the base run. Returns the pf rows
    of both methods, tagged with γ and wealth.
    """
    key = point_key(gamma, level)
    out = out_root / key
    out.mkdir(parents=True, exist_ok=True)
    wealth = scaled_wealth(inp.wealth, level, cfg.investor.wealth)

    static_path = out / "pf_static_ml.parquet"
    if force or not static_path.exists():
        s = cfg.static_ml
        static = run_static_ml(inp.panel, wealth, inp.cov, inp.val_dates, inp.eval_dates, gamma,
                               hyper_grid(s.k_grid, s.u_grid, s.g_grid), cfg.portfolio.charge_exit_tc)
        static["chosen"].to_csv(out / "chosen_static_ml.csv", index=False)
        static["pf"].loc[static["pf"]["type"] == "Static-ML*"].assign(gamma=gamma, wealth=level).to_parquet(static_path, index=False)

    pm = cfg.portfolio_ml
    sig, tags, label = signal_set(pm.use_rff, pm.rff_variance_grid)
    pfml_path = out / f"pf_portfolio_ml_{sig}.parquet"
    if force or not pfml_path.exists():
        results = {}
        for tag, g in zip(tags, pm.rff_variance_grid if pm.use_rff else [None]):
            ym = _pfml_moments(inp, cfg, gamma, wealth, level, g, cache_root / key / f"pfml_{tag}.npz")
            results[tag] = search_yearly(ym, tag, pm.p_grid, cfg.ridge_grid.values, inp.hp_years)
        pfml = run_portfolio_ml(results, inp.panel, wealth, inp.cov, inp.rf, inp.eval_dates, inp.eval_dates,
                                gamma=gamma, mu=cfg.investor.mu, iterations=cfg.investor.m_iterations,
                                charge_exit_tc=cfg.portfolio.charge_exit_tc, label=label)
        pfml["chosen"].to_csv(out / f"chosen_portfolio_ml_{sig}.csv", index=False)
        pfml["pf"].assign(gamma=gamma, wealth=level).to_parquet(pfml_path, index=False)
    return pd.concat([pd.read_parquet(static_path), pd.read_parquet(pfml_path)], ignore_index=True)


def load_point(out_root: Path, gamma: float, level: float) -> pd.DataFrame:
    """Every method stored for one point (Static-ML, Static-ML*, Portfolio-ML and/or its linear version)."""
    files = sorted((out_root / point_key(gamma, level)).glob("pf_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no results for {point_key(gamma, level)} in {out_root}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def summarise(pfs: pd.DataFrame) -> pd.DataFrame:
    """Table 2 statistics per (type, γ, wealth), each with utility at its own γ."""
    rows = []
    for (gamma, level), g in pfs.groupby(["gamma", "wealth"]):
        rows.append(table2(g, gamma).reset_index().assign(gamma=gamma, wealth=level))
    return pd.concat(rows, ignore_index=True)


def grid_points(gammas: Sequence[float], wealths: Sequence[float]) -> list[tuple[float, float]]:
    return [(float(g), float(w)) for w in wealths for g in gammas]
