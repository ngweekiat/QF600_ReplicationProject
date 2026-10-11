"""Two-method reporting must exclude legacy results and preserve Static-ML* accounting."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from jkmp.covariance import CovParts
from jkmp.evaluation.comparison import load_comparison
from jkmp.evaluation.feature_importance import static_counterfactual
from jkmp.evaluation import frontier
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.static_ml import static_weights


def test_comparison_excludes_legacy_and_linear_results(tmp_path):
    for folder, methods in [("static_ml", ["Static-ML", "Static-ML*"]),
                            ("portfolio_ml/rff", ["Portfolio-ML"]),
                            ("portfolio_ml/raw", ["Portfolio-ML (linear)"]),
                            ("benchmarks", ["Markowitz-ML", "Factor-ML"])]:
        path = tmp_path / folder
        path.mkdir(parents=True)
        pd.DataFrame({"type": methods}).to_parquet(path / "pf.parquet")
    assert load_comparison(tmp_path)["type"].tolist() == ["Static-ML*", "Portfolio-ML"]


def test_static_feature_importance_baseline_matches_direct_portfolio():
    dates = pd.date_range("2019-12-31", periods=3, freq="ME")
    ids = np.array([1, 2, 3])
    panel = pd.DataFrame([dict(id=i, eom=d, me=100.0*i, in_universe=True,
                              **{"lambda": 2e-10}, signal=i*0.01, pred_ld1=i*0.01,
                              tr_ld1=i*0.005, ret_ld1=i*0.005)
                          for d in dates for i in ids])
    wealth = pd.DataFrame(dict(eom=dates, wealth=1e10, mu_ld1=0.01))
    cov = {d: CovParts(d, ids, ("f",), np.ones((3, 1))*0.05,
                       np.array([[0.002]]), np.full(3, 0.004)) for d in dates}
    model = SimpleNamespace(predict=lambda x, z: x[:, 0])
    chosen = pd.DataFrame(dict(year=[2020], k=[1/3], u=[0.5], g=[1.0]))
    perms = {d: np.array([2, 1, 0]) for d in dates}
    args = (panel, perms, ["signal"], {2020: model}, np.array([]), panel,
            wealth, cov, chosen)
    base = static_counterfactual(None, *args, gamma=10, charge_exit_tc=True)
    inputs = month_inputs(panel, wealth, cov, dates)
    weights = static_weights(inputs, panel, wealth, 10, lambda d: (1/3, 0.5, 1.0), True)
    expected = portfolio_returns(weights, panel, wealth)
    pd.testing.assert_frame_equal(base, expected)
    shuffled = static_counterfactual(["signal"], *args, gamma=10, charge_exit_tc=True)
    assert not np.allclose(shuffled["r"], base["r"])


def test_frontier_without_markowitz_reference(monkeypatch, tmp_path):
    summary = pd.DataFrame([dict(type=t, gamma=g, wealth=1e10,
                                 Vol=0.1/g, R_TC=0.05/g, Utility=0.02/g)
                            for t in ("Static-ML*", "Portfolio-ML") for g in (1, 10)])
    labels = []
    def capture(fig, path):
        labels.extend(fig.axes[0].get_legend_handles_labels()[1])
        frontier.plt.close(fig)
        return []
    monkeypatch.setattr(frontier, "_save", capture)
    frontier.plot_fig1a(summary, pd.DataFrame(columns=["type"]), 1e10, 10, None, tmp_path / "a")
    frontier.plot_fig1b(summary, None, tmp_path / "b")
    assert "Static-ML*" in labels and "Portfolio-ML" in labels
    assert not any("Markowitz" in label for label in labels)
