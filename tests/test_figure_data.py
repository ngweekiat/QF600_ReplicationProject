"""Report-figure series (jkmp.evaluation.figure_data): additive cumulative paths, rolling IR, IR standard error."""

import numpy as np
import pandas as pd
import pytest

from jkmp.evaluation.figure_data import cumulative_paths, ir_standard_error, rolling_ir
from jkmp.evaluation.metrics import utility_flow


def test_cumulative_paths_are_additive_per_type():
    months = pd.date_range("2000-01-31", periods=4, freq="ME")
    pfs = pd.DataFrame({"type": ["A"] * 4 + ["B"] * 4, "eom_ret": list(months) * 2,
                        "r": [0.01, -0.02, 0.03, 0.0, 0.1, 0.1, 0.1, 0.1], "tc": 0.001})
    out = cumulative_paths(pfs, gamma=10.0).set_index(["type", "eom_ret"])
    a = pfs[pfs["type"] == "A"]
    assert out.loc[("A", months[-1]), "cum_gross"] == pytest.approx(a["r"].sum())
    assert out.loc[("A", months[-1]), "cum_net"] == pytest.approx((a["r"] - a["tc"]).sum())
    assert out.loc[("A", months[-1]), "cum_util"] == pytest.approx(utility_flow(a, 10.0).sum())
    assert out.loc[("B", months[1]), "cum_net"] == pytest.approx(2 * 0.099)


def test_rolling_ir_and_standard_error():
    rng = np.random.default_rng(2)
    months = pd.date_range("2000-01-31", periods=60, freq="ME")
    act = pd.DataFrame({"eom_ret": months, "active": rng.normal(0.002, 0.01, 60)})
    act["active_net"] = act["active"] - 0.0005
    ir = rolling_ir(act, 36)
    last = act.iloc[-36:]
    assert ir.iloc[:35].isna().all()
    assert ir.iloc[-1] == pytest.approx(last["active_net"].mean() / last["active"].std(ddof=1) * np.sqrt(12))
    assert ir_standard_error(0.0, 480) == pytest.approx(1 / np.sqrt(40))
    assert ir_standard_error(0.5, 60) > ir_standard_error(0.5, 480)
