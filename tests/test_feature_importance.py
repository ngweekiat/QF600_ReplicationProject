"""Feature importance (jkmp.evaluation.feature_importance): the shuffle permutes only the theme's columns,
moving each donor stock's theme values together and preserving each month's cross-section; the same
permutation is used for every theme; coefficients on untouched signals see no change; importance is the
utility drop relative to the unshuffled run."""

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.covariance import CovParts
from jkmp.evaluation.feature_importance import (BASE, importance, permutations, pfml_counterfactual_aims, shuffled,
                                                summarise_seeds, theme_list)

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
RNG = np.random.default_rng(4)
FEATS = ["a1", "a2", "b1", "c1"]


def _universe(n=6, months=3):
    dates = pd.date_range("2020-01-31", periods=months, freq="ME")
    return pd.DataFrame([{"id": i, "eom": d, **{f: RNG.uniform() for f in FEATS}}
                         for d in dates for i in range(1, n + 1)])


def test_shuffle_moves_theme_columns_together_and_keeps_the_rest():
    u = _universe()
    perms = permutations(u, seed=1)
    rows = u[u["eom"] == u["eom"].iloc[0]].sort_values("id")
    perm = perms[rows["eom"].iloc[0]]
    x0 = rows[FEATS].to_numpy()
    x = shuffled(rows, FEATS, ["a1", "a2"], perm)
    np.testing.assert_array_equal(x[:, 2:], x0[:, 2:])                     # other themes untouched
    np.testing.assert_array_equal(x[:, :2], x0[perm][:, :2])                # donor's whole theme vector
    np.testing.assert_array_equal(np.sort(x[:, 0]), np.sort(x0[:, 0]))      # same cross-section
    np.testing.assert_array_equal(shuffled(rows, FEATS, None, perm), x0)    # unshuffled run


def test_same_permutations_for_every_theme_and_seed():
    u = _universe()
    a, b = permutations(u, seed=1), permutations(u, seed=1)
    assert all(np.array_equal(a[d], b[d]) for d in a) and len(a) == 3
    c = permutations(u, seed=2)
    assert any(not np.array_equal(a[d], c[d]) for d in a)


def test_linear_aims_ignore_shuffles_of_unused_signals():
    u = _universe()
    ids = np.arange(1, 7)
    cov = {d: CovParts(d, ids, ("f",), RNG.normal(size=(6, 1)) * 0.05, np.array([[0.002]]), np.full(6, 0.004))
           for d in u["eom"].unique()}
    columns = ["constant", *FEATS]
    meta = {"features": FEATS, "use_rff": False, "add_orig": False, "scale_by_vol": True}
    beta = {"constant": 0.3, "a1": 0.0, "a2": 0.0, "b1": 1.0, "c1": -0.5}   # theme a unused
    coefs = pd.concat([pd.DataFrame({"eom": d, "tag": "raw", "column": list(beta), "beta": list(beta.values())})
                       for d in u["eom"].unique()], ignore_index=True)
    perms = permutations(u, seed=3)
    specs = {"raw": (meta, columns, None)}
    base = pfml_counterfactual_aims(u, perms, None, coefs, specs, cov)
    np.testing.assert_allclose(pfml_counterfactual_aims(u, perms, ["a1", "a2"], coefs, specs, cov)["w_aim"],
                               base["w_aim"], rtol=1e-14)
    assert not np.allclose(pfml_counterfactual_aims(u, perms, ["b1"], coefs, specs, cov)["w_aim"], base["w_aim"])


def test_importance_is_drop_from_unshuffled():
    r_base, r_cf = np.array([0.02, 0.01, 0.03]), np.array([0.0, 0.01, -0.01])
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    pfs = pd.concat([pd.DataFrame({"type": "M", "theme": BASE, "eom_ret": months, "r": r_base, "tc": 0.001}),
                     pd.DataFrame({"type": "M", "theme": "value", "eom_ret": months, "r": r_cf, "tc": 0.002})])
    out = importance(pfs, gamma=10, dates=load_config().dates).set_index("theme")
    u = lambda r, tc: (r.mean() - 5 * r.var(ddof=1) - tc) * 12
    assert out.loc["value", "fi"] == pytest.approx(u(r_base, 0.001) - u(r_cf, 0.002))
    assert out.loc[BASE, "fi"] == 0.0
    assert [t for t, _ in theme_list({"value": ["x"], "momentum": ["y"]})] == [BASE, "value", "momentum"]


def test_importance_per_seed_and_standard_error():
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    r_base = np.array([0.02, 0.01, 0.03])
    cf = {1: np.array([0.0, 0.01, -0.01]), 2: np.array([0.01, 0.02, 0.0]), 3: np.array([0.02, 0.0, 0.01])}
    pfs = pd.concat([pd.DataFrame({"seed": s, "type": "M", "theme": th, "eom_ret": months, "r": r, "tc": 0.001})
                     for s, r_cf in cf.items() for th, r in ((BASE, r_base), ("value", r_cf))])
    by_seed = importance(pfs, gamma=10, dates=load_config().dates)
    u = lambda r: (r.mean() - 5 * r.var(ddof=1) - 0.001) * 12
    fis = np.array([u(r_base) - u(cf[s]) for s in (1, 2, 3)])
    got = by_seed[by_seed["theme"] == "value"].sort_values("seed")["fi"].to_numpy()
    np.testing.assert_allclose(got, fis, rtol=1e-12)
    out = summarise_seeds(by_seed).set_index("theme")
    assert out.loc["value", "fi"] == pytest.approx(fis.mean())
    assert out.loc["value", "fi_se"] == pytest.approx(fis.std(ddof=1) / np.sqrt(3))
    assert out.loc[BASE, "fi_se"] == 0.0 and out.loc["value", "n_seeds"] == 3
    assert load_config().feature_importance.seeds(1) == tuple(range(1, 21))
