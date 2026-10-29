"""Hyperparameter edge shares and the Fig. D.1 plot (jkmp.evaluation.hyperparams)."""

import numpy as np
import pandas as pd

from jkmp.evaluation.hyperparams import edge_share, plot_hyperparameters


def test_edge_share_counts_picks_on_each_edge():
    out = edge_share({"p": (pd.Series([8, 8, 64, 512]), [8, 16, 64, 512]), "one": (pd.Series([1.0]), [1.0])})
    p = out.set_index("hyperparameter").loc["p"]
    assert p["n"] == 4 and p["share_at_min"] == 0.5 and p["share_at_max"] == 0.25
    assert np.isnan(out.set_index("hyperparameter").loc["one", "share_at_min"])


def test_plot_hyperparameters_writes_files(tmp_path):
    years = np.arange(1972, 1980)
    pf = pd.DataFrame({"year": years, "g": 0.05, "p": 64, "l": np.r_[0.0, np.exp(np.arange(7))]})
    st = pd.DataFrame({"year": years, "k": 1.0, "u": 0.5, "g": 1.0})
    out = plot_hyperparameters({"Portfolio-ML": pf, "Portfolio-ML (linear)": pf}, st, tmp_path / "figd1")
    assert all(p.exists() and p.stat().st_size > 0 for p in out)
