"""Cost re-costing (jkmp.evaluation.cost_sensitivity) and rank IC by size bucket (jkmp.models.return_model)."""

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.covariance import CovParts
from jkmp.evaluation.cost_sensitivity import capacity_aum, recost, vol_cost_scale
from jkmp.models.return_model import monthly_rank_ic, rank_ic_by_size
from jkmp.portfolio.accounting import portfolio_returns

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
MONTHS = pd.date_range("1990-01-31", periods=6, freq="ME")


def _setup():
    rng = np.random.default_rng(2)
    panel = pd.DataFrame([{"id": i, "eom": e, "ret_ld1": rng.normal(0, 0.05), "lambda": 1e-10 * (i + 1)}
                          for e in MONTHS for i in range(3)])
    weights = pd.DataFrame([{"id": i, "eom": e, "w": rng.normal(0, 0.3), "w_start": rng.normal(0, 0.3), "exit": False}
                            for e in MONTHS for i in range(3)])
    wealth = pd.DataFrame({"eom": MONTHS, "wealth": 1e10})
    return panel, weights, wealth


def test_recost_scales_tc_linearly_by_month_and_by_stock():
    panel, weights, wealth = _setup()
    base = portfolio_returns(weights, panel, wealth)
    by_month = recost(weights, panel, wealth, pd.Series(np.arange(1.0, 7.0), index=MONTHS))
    np.testing.assert_allclose(by_month["tc"], base["tc"] * np.arange(1.0, 7.0), rtol=1e-12)
    np.testing.assert_allclose(by_month["r"], base["r"])                         # returns untouched
    stock = pd.Series(2.0, index=pd.MultiIndex.from_frame(panel[["id", "eom"]]))
    np.testing.assert_allclose(recost(weights, panel, wealth, stock)["tc"], 2 * base["tc"], rtol=1e-12)
    partial = stock.drop(index=[(0, MONTHS[0])])                                 # missing scale -> Λ unchanged
    assert recost(weights, panel, wealth, partial)["tc"].iloc[0] < 2 * base["tc"].iloc[0]


def test_vol_cost_scale_uses_past_months_only():
    vols = [0.1, 0.2, 0.3, 0.4]
    cov = {m: CovParts(m, np.array([7]), ("f",), np.zeros((1, 1)), np.zeros((1, 1)), np.array([v**2]))
           for m, v in zip(MONTHS, vols)}
    s = vol_cost_scale(cov, window=2)
    np.testing.assert_allclose(s.loc[7].to_numpy(), [1.0, 0.2 / 0.15, 0.3 / 0.25, 0.4 / 0.35])


def test_capacity_aum_interpolates_in_log_wealth():
    w = [1e9, 1e10, 1e11]
    assert capacity_aum(w, [0.5, 0.4, 0.2], 0.3) == pytest.approx(np.exp(np.log(1e10) + 0.5 * np.log(10)))
    assert capacity_aum(w, [0.5, 0.4, 0.35], 0.3) == np.inf
    assert capacity_aum(w, [0.2, 0.1, 0.0], 0.3) == -np.inf


def test_rank_ic_by_size_buckets_by_market_cap():
    rng = np.random.default_rng(4)
    rows = []
    for e in pd.date_range("1990-01-31", periods=24, freq="ME"):
        for i in range(6):
            r = rng.normal()
            rows.append({"id": i, "eom": e, "me": 100 - i, "valid_data": True, "ret_ld1": r,
                         "pred_ld1": r if i < 3 else -r})                       # perfect in the top 3, inverted below
    d = pd.DataFrame(rows)
    out = rank_ic_by_size(d[["id", "eom", "pred_ld1"]], d, [(1, 3), (4, 6)], load_config().dates)
    allrow = out[out["decade"] == "all"].set_index("bucket")["rank_ic"]
    assert allrow["1-3"] == pytest.approx(1.0) and allrow["4-6"] == pytest.approx(-1.0)
    assert set(out["decade"]) == {1990, "all"}


def test_monthly_rank_ic_is_indexed_by_return_month():
    rows = [{"id": i, "eom": e, "in_universe": True, "ret_ld1": float(i), "pred_ld1": float(i)}
            for e in pd.date_range("1990-01-31", periods=3, freq="ME") for i in range(5)]
    d = pd.DataFrame(rows)
    ic = monthly_rank_ic(d[["id", "eom", "pred_ld1"]], d, load_config().dates)
    assert np.allclose(ic, 1.0) and ic.index[0] == pd.Timestamp("1990-02-28")
