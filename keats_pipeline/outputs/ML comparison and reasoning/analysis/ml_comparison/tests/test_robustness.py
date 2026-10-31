from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robustness import (cost_sensitivity, fixed_weight_aum, gamma_sensitivity,
                        saved_stats, _portfolio_diagnostics)


def example():
    return pd.DataFrame({"dataset": "test/base", "strategy": "ML",
                         "eom_ret": pd.date_range("2020-01-31", periods=4, freq="ME"),
                         "r": [.02, -.01, .04, .01], "tc": [.001, .002, .001, .003],
                         "turnover": [.2, .3, .1, .2], "inv": [1., 1., 1., 1.]})


def test_cost_scenarios_preserve_risk_and_turnover_and_reconcile_return():
    frame = example()
    result = cost_sensitivity(frame).set_index("cost_factor")
    assert len(result) == 5
    assert np.allclose(result.R - result.TC, result.R_TC)
    assert result.Vol.nunique() == 1
    assert result.Turnover.nunique() == 1
    assert result.loc[0., "TC"] == 0
    assert result.loc[3., "TC"] == pytest.approx(result.loc[1., "TC"] * 3)
    assert result.loc[3., "Utility"] < result.loc[0., "Utility"]


def test_gamma_changes_only_evaluation_penalty():
    frame = example()
    result = gamma_sensitivity(frame).set_index("evaluation_gamma")
    assert result.R_TC.nunique() == 1
    assert result.SR_net_paper.nunique() == 1
    expected = -0.5 * 12 * (100 - 1) * frame.r.var(ddof=1)
    assert result.loc[100., "Utility"] - result.loc[1., "Utility"] == pytest.approx(expected)


def test_aum_linear_cost_scaling_not_weight_scaling():
    frame = example()
    result = fixed_weight_aum(frame, {"test/base": 10e9}).set_index("aum_anchor_usd")
    assert result.R.nunique() == 1
    assert result.Lev.nunique() == 1
    assert result.loc[100e9, "TC"] == pytest.approx(result.loc[10e9, "TC"] * 10)


def test_paper_net_sharpe_uses_gross_volatility():
    frame = example()
    stats = saved_stats(frame)
    assert stats["SR_net_paper"] == pytest.approx(np.sqrt(12) * (frame.r - frame.tc).mean() / frame.r.std(ddof=1))


def test_subgroup_contributions_and_full_turnover_add_to_portfolio():
    date = pd.Timestamp("2020-01-31")
    weights = pd.DataFrame({"id": [1, 2, 3], "eom": date,
                            "w": [1.2, -.2, 0.], "w_start": [1., -.1, .1],
                            "exit": [False, False, True]})
    panel = pd.DataFrame({"id": [1, 2, 3], "eom": date, "me": [3., 2., 1.],
                          "dolvol": [300., 200., 100.], "ret_ld1": [.03, -.01, np.nan],
                          "lambda": [1e-10, 2e-10, 3e-10], "ff12": [1, 2, 3],
                          "size_bucket": ["high", "low", "outside_current_membership"],
                          "liquidity_bucket": ["high", "low", "outside_current_membership"]})
    wealth = pd.DataFrame({"eom": [date], "wealth": [1e9]})
    construction, groups, sectors = _portfolio_diagnostics(weights, panel, wealth, "ML", "test/base")
    assert construction.n_nonzero.iloc[0] == 2
    assert construction.gross_leverage.iloc[0] == pytest.approx(1.4)
    assert construction.abs_trade_turnover.iloc[0] == pytest.approx(.4)
    assert construction.abs_weight_hhi.iloc[0] == pytest.approx((1.2 / 1.4)**2 + (.2 / 1.4)**2)
    for _, group in groups.groupby("group_variable"):
        assert group.gross_return_contribution.sum() == pytest.approx(1.2 * .03 + -.2 * -.01)
        assert group.tc_contribution.sum() == pytest.approx(.5 * 1e9 * np.sum(np.array([1e-10, 2e-10, 3e-10]) * np.array([.2, -.1, -.1])**2))
    assert sectors.turnover.sum() == pytest.approx(.4)


def test_nonfinite_returns_are_rejected():
    frame = example()
    frame.loc[0, "r"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        saved_stats(frame)
