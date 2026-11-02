"""A short/entry/exit example checked against explicit economic accounting."""
import numpy as np
import pandas as pd
import pytest

from jkmp.portfolio.accounting import add_start_weights, portfolio_returns


def test_inherited_short_entry_and_exit_costs_by_hand():
    jan, feb = pd.to_datetime(["2020-01-31", "2020-02-29"])
    panel = pd.DataFrame({
        "id": [1, 2, 1, 2, 3], "eom": [jan, jan, feb, feb, feb],
        "me": [60., 40., 60., 40., 50.],
        "tr_ld1": [.10, -.05, .04, np.nan, -.01],
        "ret_ld1": [.09, -.06, .03, np.nan, -.02],
        "lambda": [1e-10, 2e-10, 1e-10, 2e-10, 3e-10],
    })
    wealth = pd.DataFrame({"eom": [jan, feb], "wealth": [1e9, 1e9], "mu_ld1": [.01, .02]})
    targets = pd.DataFrame({"id": [1, 2, 1, 3], "eom": [jan, jan, feb, feb], "w": [.6, -.2, .2, .5]})
    weights = add_start_weights(targets, panel, wealth, [jan, feb], charge_exit_tc=True)
    feb_weights = weights[weights.eom.eq(feb)].set_index("id")
    assert feb_weights.loc[1, "w_start"] == pytest.approx(.6 * 1.10 / 1.01)
    assert feb_weights.loc[2, "w_start"] == pytest.approx(-.2 * .95 / 1.01)
    assert feb_weights.loc[3, "w_start"] == 0
    assert feb_weights.loc[2, "exit"] and feb_weights.loc[2, "w"] == 0

    pf = portfolio_returns(weights, panel, wealth).set_index("eom")
    assert pf.loc[jan, "r"] == pytest.approx(.6 * .09 + (-.2) * (-.06))
    assert pf.loc[jan, "tc"] == pytest.approx(.5 * 1e9 * 2e-10 * (-.2 - .4) ** 2)
    delta1, delta2, delta3 = .2 - .6 * 1.10 / 1.01, .2 * .95 / 1.01, .5
    assert pf.loc[feb, "r"] == pytest.approx(.2 * .03 + .5 * (-.02))
    assert pf.loc[feb, "tc"] == pytest.approx(.5 * 1e9 * (1e-10 * delta1**2 + 2e-10 * delta2**2 + 3e-10 * delta3**2))
    assert pf.loc[feb, "turnover"] == pytest.approx(abs(delta1) + abs(delta2) + abs(delta3))
    assert pf.loc[feb, "inv"] == pytest.approx(.7)
    assert pf.loc[feb, "eom_ret"] == pd.Timestamp("2020-03-31")

    # A zero exit position can have no forward return; a held position cannot.
    missing = panel.copy()
    missing.loc[missing.eom.eq(feb) & missing.id.eq(3), "ret_ld1"] = np.nan
    with pytest.raises(ValueError, match="missing ret_ld1"):
        portfolio_returns(weights, missing, wealth)
