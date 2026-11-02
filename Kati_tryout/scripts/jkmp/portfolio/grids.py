"""Shared date windows, wealth scaling and signal settings for Kati frontiers and iterations.

Only helper functions used by the retained Static ML and Portfolio ML are included.
Each frontier point is fitted by scripts/figure1.py.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jkmp.config import Config, DatesConfig
from jkmp.covariance import CovParts
from jkmp.evaluation.holdout import check_return_months, evaluation_window
from jkmp.evaluation.metrics import table2

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
    val_dates: pd.DatetimeIndex    # Forecast-history formation months (returns from the first prediction year)
    eval_dates: pd.DatetimeIndex   # test window formation months
    hp_years: list[int]


def grid_inputs(cfg: Config, panel: pd.DataFrame, wealth: pd.DataFrame, cov: Mapping[pd.Timestamp, CovParts],
                rf: pd.Series) -> GridInputs:
    """Date windows as in scripts 05 and 07; the panel cut to stocks that are ever in the universe
    (the only ones that are held, enter m and Σ̃, or are sold on exit)."""
    formation = pd.DatetimeIndex(sorted(cov))
    ret = formation + MONTH
    first_year = cfg.return_model.pred_years[0]
    test_start, test_end = evaluation_window(cfg.dates)
    lb = cfg.universe.lookback_months
    first = formation[0] + MONTH * (lb + 1)
    keep = panel["id"].isin(panel.loc[panel["in_universe"], "id"].unique())
    eval_dates = formation[(ret >= test_start) & (ret <= test_end)]
    val_dates = formation[(ret >= pd.Timestamp(first_year, 1, 31)) & (ret <= test_end)]
    check_return_months(eval_dates.union(val_dates) + MONTH, cfg.dates, "grid windows")
    cols = ["id", "eom", "me", "in_universe", "lambda", "g", "tr_ld1", "ret_ld1", "pred_ld1", *cfg.features.names]
    return GridInputs(
        panel=panel.loc[keep & (panel["eom"] >= formation[0]), cols].reset_index(drop=True),
        wealth=wealth, cov=cov, rf=rf,
        pfml_dates=formation[(formation >= first) & (ret <= test_end)],
        val_dates=val_dates, eval_dates=eval_dates,
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
