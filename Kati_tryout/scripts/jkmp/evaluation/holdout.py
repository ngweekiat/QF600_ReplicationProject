"""Holdout lock: no performance number may use a return month after ``dates.test_end`` while
``dates.holdout_locked`` is true (holdout ``dates.holdout_start`` .. ``dates.holdout_end``).

Every evaluation path calls ``check_return_months`` on the rows it summarises: Table 2
(``metrics.table2``, so also ``grids.summarise``), the grid windows (``grids.grid_inputs``), the scaled
frontier, feature importance, the regime statistics and the cumulative-utility plot, and the return-model
diagnostics. The frontier and Fig. 5 plots only draw frames those functions produced. Model fitting is not
checked: it may use data up to the end of each walk-forward year.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from jkmp.config import DatesConfig

MONTH = pd.offsets.MonthEnd(1)


class HoldoutLockedError(RuntimeError):
    """A performance computation reached a return month inside the locked holdout."""


def last_open_month(dates: DatesConfig) -> pd.Timestamp:
    """The last return month any performance number may use: test_end while locked, else holdout_end."""
    return pd.Timestamp(dates.test_end if dates.holdout_locked else dates.holdout_end)


def check_return_months(rows: pd.DataFrame | Sequence[pd.Timestamp], dates: DatesConfig, what: str) -> None:
    """Raise ``HoldoutLockedError`` if any return month is after ``last_open_month(dates)``.

    ``rows`` is a frame with ``eom_ret`` (the return month) or ``eom`` (the formation month, return month
    eom + 1), or a sequence of return months. ``what`` names the computation in the error message.
    """
    if isinstance(rows, pd.DataFrame):
        if "eom_ret" in rows:
            months = pd.DatetimeIndex(rows["eom_ret"])
        elif "eom" in rows:
            months = pd.DatetimeIndex(rows["eom"]) + MONTH
        else:
            raise ValueError(f"{what}: rows need an eom_ret or eom column to check the holdout lock")
    else:
        months = pd.DatetimeIndex(rows)
    last = last_open_month(dates)
    if len(months) and months.max() > last:
        raise HoldoutLockedError(
            f"{what}: return months up to {months.max().date()} but the holdout is locked after {last.date()} "
            f"(dates.holdout_locked = true)")


def evaluation_window(dates: DatesConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last return month evaluated by scripts 04–07 and 12: the test window, or — only with
    ``dates.evaluate: holdout`` and the lock off — the holdout (the walk-forward continues through it: models,
    β and hyperparameters keep updating each year on everything up to the year before)."""
    if dates.evaluate == "test":
        return pd.Timestamp(dates.test_start), pd.Timestamp(dates.test_end)
    if dates.holdout_locked:
        raise HoldoutLockedError("dates.evaluate: holdout while dates.holdout_locked is true")
    return pd.Timestamp(dates.holdout_start), pd.Timestamp(dates.holdout_end)
