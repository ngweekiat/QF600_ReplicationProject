"""Holdout lock (jkmp.evaluation.holdout): every evaluation path refuses return months after
dates.test_end while dates.holdout_locked is true, and runs once the lock is lifted."""

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.evaluation.feature_importance import BASE, importance
from jkmp.evaluation.holdout import HoldoutLockedError, check_return_months, last_open_month
from jkmp.evaluation.metrics import table2
from jkmp.evaluation.regimes import cumulative_utility, recost_vix, regime_table
from jkmp.portfolio.grids import summarise

LOCKED = load_config().dates
OPEN = load_config(overrides=["dates.holdout_locked=false"]).dates


def _pf(last_ret: str, n: int = 4) -> pd.DataFrame:
    ret = pd.date_range(end=last_ret, periods=n, freq="ME")
    return pd.DataFrame({"type": "A", "theme": BASE, "eom": ret - pd.offsets.MonthEnd(1), "eom_ret": ret,
                         "r": np.linspace(-0.01, 0.02, n), "tc": 0.001, "turnover": 0.1, "inv": 1.0,
                         "shorting": 0.0, "gamma": 10.0, "wealth": 1e10})


def test_last_open_month_follows_the_lock():
    assert last_open_month(LOCKED) == pd.Timestamp("2020-12-31")
    assert last_open_month(OPEN) == pd.Timestamp("2025-12-31")


def test_check_reads_return_months_from_eom_ret_eom_or_a_sequence():
    check_return_months(_pf("2020-12-31"), LOCKED, "ok")                 # last test month is allowed
    with pytest.raises(HoldoutLockedError, match="2021-01-31"):
        check_return_months(_pf("2021-01-31"), LOCKED, "eom_ret")
    with pytest.raises(HoldoutLockedError):                              # formation Dec 2020 earns Jan 2021
        check_return_months(pd.DataFrame({"eom": [pd.Timestamp("2020-12-31")]}), LOCKED, "eom")
    with pytest.raises(HoldoutLockedError):
        check_return_months(pd.DatetimeIndex(["2022-06-30"]), LOCKED, "sequence")
    with pytest.raises(ValueError, match="eom_ret or eom"):
        check_return_months(pd.DataFrame({"r": [0.0]}), LOCKED, "no dates")
    check_return_months(_pf("2025-12-31"), OPEN, "unlocked")
    with pytest.raises(HoldoutLockedError):                              # never past the holdout end
        check_return_months(_pf("2026-01-31"), OPEN, "after holdout")


@pytest.mark.parametrize("run", [
    lambda pf, d: table2(pf, 10, d),
    lambda pf, d: summarise(pf, d),
    lambda pf, d: importance(pf, 10, d),
    lambda pf, d: cumulative_utility(pf, 10, d),
    lambda pf, d: recost_vix(pf, pd.Series(1.0, index=pf["eom"]), 10, d),
    lambda pf, d: regime_table(pf, pd.DataFrame({"eom": pf["eom_ret"], "vix_regime": "low VIX",
                                                 "fed_regime": "on hold", "recession": "expansion"}), 10, d),
], ids=["table2", "grids.summarise", "importance", "cumulative_utility", "recost_vix", "regime_table"])
def test_evaluation_paths_raise_while_locked(run):
    pf = _pf("2021-03-31")
    with pytest.raises(HoldoutLockedError):
        run(pf, LOCKED)
    run(pf, OPEN)


def test_plot_frontier_and_diagnostics_raise_while_locked(tmp_path):
    from jkmp.evaluation.frontier import scaled_frontier
    from jkmp.evaluation.regimes import plot_cumulative_utility
    from jkmp.models.return_model import prediction_diagnostics

    pf = _pf("2021-03-31")
    cum = cumulative_utility(pf, 10, OPEN)
    with pytest.raises(HoldoutLockedError):
        plot_cumulative_utility(cum, [], ["A"], 10, tmp_path / "cum", LOCKED)
    assert all(p.exists() for p in plot_cumulative_utility(cum, [], ["A"], 10, tmp_path / "cum", OPEN))

    weights = pd.DataFrame({"type": "M", "id": 1, "eom": pf["eom"], "w": 1.0, "w_start": 1.0, "exit": False})
    with pytest.raises(HoldoutLockedError):
        scaled_frontier(weights, None, None, [0.1], 10, LOCKED)          # refuses before touching the data

    rows = pf[["eom", "eom_ret"]].merge(pd.DataFrame({"id": [1, 2, 3]}), how="cross")
    preds = rows.assign(year=rows["eom_ret"].dt.year, pred_ld1=np.tile([0.01, 0.0, -0.01], len(pf)))[
        ["id", "eom", "year", "pred_ld1"]]
    panel = rows.assign(ret_ld1=np.tile([0.02, 0.01, -0.03], len(pf)), in_universe=True)[
        ["id", "eom", "ret_ld1", "in_universe"]]
    with pytest.raises(HoldoutLockedError):
        prediction_diagnostics(preds, panel, LOCKED)
    assert len(prediction_diagnostics(preds, panel, OPEN)) == 2 * preds["year"].nunique()
