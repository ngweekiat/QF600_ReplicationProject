"""Hand-computed accounting checks for jkmp.portfolio.accounting (port of R w_fun / pf_ts_fun)."""

import pandas as pd
import pytest

from jkmp.portfolio.accounting import add_start_weights, portfolio_returns

T1, T2 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")
WEALTH = pd.DataFrame({"eom": [T1, T2], "wealth": [1.0e9, 1.02e9], "mu_ld1": [0.02, 0.01]})


def _panel(rows):
    cols = ["id", "eom", "me", "lambda", "tr_ld1", "ret_ld1"]
    return pd.DataFrame(rows, columns=cols)


TWO_STOCKS = _panel([
    (1, T1, 300.0, 1e-9, 0.10, 0.09), (2, T1, 100.0, 2e-9, -0.05, -0.06),
    (1, T2, 330.0, 1e-9, 0.02, 0.01), (2, T2, 95.0, 2e-9, 0.03, 0.02),
])
TARGETS = pd.DataFrame({"id": [1, 2, 1, 2], "eom": [T1, T1, T2, T2], "w": [0.6, 0.4, 0.5, 0.5]})


def _run(panel, targets, charge_exit_tc=True):
    w = add_start_weights(targets, panel, WEALTH, [T1, T2], charge_exit_tc)
    return w, portfolio_returns(w, panel, WEALTH).set_index("eom")


def test_two_stocks_two_months_by_hand():
    w, pf = _run(TWO_STOCKS, TARGETS)
    start = w.set_index(["eom", "id"])["w_start"]

    # Month 1: inherited = value weights 300/400, 100/400; trade to 0.6 / 0.4.
    assert start[(T1, 1)] == pytest.approx(0.75) and start[(T1, 2)] == pytest.approx(0.25)
    # tc = wealth/2 · Σ λ (Δw)² = 1e9/2 · (1e-9·0.15² + 2e-9·0.15²) = 0.03375
    assert pf.loc[T1, "tc"] == pytest.approx(0.5 * 1e9 * (1e-9 * 0.15**2 + 2e-9 * 0.15**2), abs=1e-12)
    assert pf.loc[T1, "tc"] == pytest.approx(0.03375, abs=1e-12)
    assert pf.loc[T1, "turnover"] == pytest.approx(0.30, abs=1e-12)
    assert pf.loc[T1, "r"] == pytest.approx(0.6 * 0.09 + 0.4 * -0.06, abs=1e-12)  # 0.03
    assert pf.loc[T1, "inv"] == pytest.approx(1.0) and pf.loc[T1, "eom_ret"] == T2

    # Month 2: inherited = drifted w·(1 + tr_ld1)/(1 + mu_ld1) = 0.6·1.10/1.02, 0.4·0.95/1.02.
    s1, s2 = 0.6 * 1.10 / 1.02, 0.4 * 0.95 / 1.02
    assert start[(T2, 1)] == pytest.approx(s1, abs=1e-12) and start[(T2, 2)] == pytest.approx(s2, abs=1e-12)
    # tc = 1.02e9/2 · (1e-9·(0.5 − 0.647059)² + 2e-9·(0.5 − 0.372549)²) = 0.0275980392
    assert pf.loc[T2, "tc"] == pytest.approx(0.5 * 1.02e9 * (1e-9 * (0.5 - s1) ** 2 + 2e-9 * (0.5 - s2) ** 2), abs=1e-12)
    assert pf.loc[T2, "tc"] == pytest.approx(0.0275980392, abs=1e-10)
    assert pf.loc[T2, "turnover"] == pytest.approx(abs(0.5 - s1) + abs(0.5 - s2), abs=1e-12)
    assert pf.loc[T2, "r"] == pytest.approx(0.5 * 0.01 + 0.5 * 0.02, abs=1e-12)


THREE_STOCKS_T1 = TWO_STOCKS[TWO_STOCKS["eom"] == T1].to_numpy().tolist() + [[3, T1, 100.0, 4e-9, 0.20, 0.19]]
TARGETS_EXIT = pd.DataFrame({"id": [1, 2, 3, 1, 2, 4], "eom": [T1, T1, T1, T2, T2, T2],
                             "w": [0.5, 0.3, 0.2, 0.5, 0.3, 0.2]})


def _exit_panel(stock3_trades_in_t2):
    rows = THREE_STOCKS_T1 + TWO_STOCKS[TWO_STOCKS["eom"] == T2].to_numpy().tolist() + [[4, T2, 50.0, 5e-9, 0.0, 0.01]]
    if stock3_trades_in_t2:
        rows.append([3, T2, 110.0, 4e-9, 0.0, 0.05])  # still trades, but left the universe
    return _panel(rows)


def test_exit_is_charged_when_the_stock_still_trades():
    w, pf = _run(_exit_panel(True), TARGETS_EXIT)
    exit_row = w[(w["eom"] == T2) & (w["id"] == 3)].iloc[0]
    s3 = 0.2 * 1.20 / 1.02
    assert exit_row["exit"] and exit_row["w"] == 0.0 and exit_row["w_start"] == pytest.approx(s3)
    # New stock 4 starts from 0.
    assert w[(w["eom"] == T2) & (w["id"] == 4)]["w_start"].item() == 0.0
    s1, s2 = 0.5 * 1.10 / 1.02, 0.3 * 0.95 / 1.02
    expected = 0.5 * 1.02e9 * (1e-9 * (0.5 - s1) ** 2 + 2e-9 * (0.3 - s2) ** 2 + 5e-9 * 0.2**2 + 4e-9 * s3**2)
    assert pf.loc[T2, "tc"] == pytest.approx(expected, abs=1e-12)
    assert pf.loc[T2, "r"] == pytest.approx(0.5 * 0.01 + 0.3 * 0.02 + 0.2 * 0.01, abs=1e-12)  # exit earns nothing


@pytest.mark.parametrize("stock3_trades, charge", [(False, True), (True, False)])
def test_exit_not_charged_for_delisting_or_when_flag_off(stock3_trades, charge):
    w, pf = _run(_exit_panel(stock3_trades), TARGETS_EXIT, charge_exit_tc=charge)
    assert not ((w["eom"] == T2) & (w["id"] == 3)).any()
    s1, s2 = 0.5 * 1.10 / 1.02, 0.3 * 0.95 / 1.02
    expected = 0.5 * 1.02e9 * (1e-9 * (0.5 - s1) ** 2 + 2e-9 * (0.3 - s2) ** 2 + 5e-9 * 0.2**2)
    assert pf.loc[T2, "tc"] == pytest.approx(expected, abs=1e-12)
