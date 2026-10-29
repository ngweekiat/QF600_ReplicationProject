"""Random features, closed-form ridge and the yearly return model (jkmp.models)."""

import numpy as np
import pandas as pd
import pytest

# Test-side reference computations use plain matmuls, which trip Apple Accelerate's spurious
# divide/overflow flags; library code handles them itself (np.errstate + finite checks).
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

from jkmp.config import load_config
from jkmp.models.random_features import base_draws, column_subset, rff, rff_raw, weights_for
from jkmp.models.return_model import cutoffs_for_year, fit_return_model, formation_months_for_year, prefix_moments
from jkmp.models.ridge import Moments, mse_from_moments, ridge_path

RNG = np.random.default_rng(3)


# --------------------------------------------------------------------------- random features


def test_draws_are_seeded_shared_across_g_and_g_is_a_variance():
    z1, z2 = base_draws(18, 512, seed=1), base_draws(18, 512, seed=1)
    np.testing.assert_array_equal(z1, z2)
    w_small, w_big = weights_for(z1, np.exp(-3)), weights_for(z1, np.exp(-2))
    np.testing.assert_allclose(w_big / w_small, np.sqrt(np.exp(-2) / np.exp(-3)))   # same draws, scaled by √g
    big = weights_for(base_draws(4, 200_000, seed=2), 0.25)
    assert big.var() == pytest.approx(0.25, rel=0.01)                                # variance g, not sd


def test_rff_scaling_and_column_subset():
    x, z = RNG.uniform(size=(7, 18)), base_draws(18, 16, seed=0)
    full = rff_raw(x, z)                                   # 16 features: cos 1..8, sin 1..8
    np.testing.assert_allclose(rff(x, z, 6), full[:, column_subset(6, 16)] / np.sqrt(6))
    np.testing.assert_allclose(rff(x, z, 6)[:, :3], np.cos(x @ z[:, :3]) / np.sqrt(6))
    np.testing.assert_allclose(rff(x, z, 6)[:, 3:], np.sin(x @ z[:, :3]) / np.sqrt(6))


# --------------------------------------------------------------------------- ridge


def _data(n=400, p=6):
    x = RNG.normal(size=(n, p)) + 0.3
    y = x @ RNG.normal(size=p) * 0.1 + 0.05 + RNG.normal(size=n)
    return x, y


def test_ridge_path_matches_direct_solve_with_unpenalised_intercept():
    x, y = _data()
    lambdas = np.array([0.0, 1e-3, 0.1, 10.0])
    coef, b0 = ridge_path(Moments.of(x, y), lambdas)
    xc, yc = x - x.mean(0), y - y.mean()
    for i, lam in enumerate(lambdas):
        direct = np.linalg.solve(xc.T @ xc / len(y) + lam * np.eye(x.shape[1]), xc.T @ yc / len(y))
        np.testing.assert_allclose(coef[:, i], direct, rtol=1e-9, atol=1e-12)
        assert b0[i] == pytest.approx(y.mean() - x.mean(0) @ direct)
    ols = np.linalg.lstsq(np.column_stack([np.ones(len(y)), x]), y, rcond=None)[0]
    np.testing.assert_allclose(np.r_[b0[0], coef[:, 0]], ols, rtol=1e-9)   # λ = 0 is OLS


def test_moments_add_subtract_and_mse_from_moments():
    x, y = _data()
    a, b = Moments.of(x[:150], y[:150]), Moments.of(x[150:], y[150:])
    total = a + b
    np.testing.assert_allclose(total.sxx, Moments.of(x, y).sxx)
    np.testing.assert_allclose((total - a).sxy, b.sxy)
    coef, b0 = ridge_path(a, np.array([0.0, 0.5]))
    direct = ((y[150:, None] - b0 - x[150:] @ coef) ** 2).mean(axis=0)
    np.testing.assert_allclose(mse_from_moments(b, coef, b0), direct, rtol=1e-9)


# --------------------------------------------------------------------------- yearly splits


def test_year_boundaries_follow_r_data_split():
    # Return year 2015 with 5 validation years: train on returns to Dec 2009, validate 2010–2014.
    assert cutoffs_for_year(2015, 5) == (pd.Timestamp("2009-11-30"), pd.Timestamp("2014-11-30"))
    assert formation_months_for_year(2015) == (pd.Timestamp("2014-12-31"), pd.Timestamp("2015-11-30"))


CFG = load_config(overrides={
    "return_model.p_grid": [2, 4, 8], "return_model.pred_years": [2008, 2009], "return_model.val_years": 2,
    "ridge_grid.n": 5,
})
FEATS = list(CFG.features.names)


def _panel():
    eoms = pd.date_range("2002-01-31", "2009-11-30", freq="ME")
    n = 60
    x = RNG.uniform(size=(len(eoms) * n, len(FEATS)))
    p = pd.DataFrame(x, columns=FEATS).assign(id=np.tile(np.arange(n), len(eoms)), eom=np.repeat(eoms, n), valid_data=True)
    p["ret_ld1"] = 0.05 * (p[FEATS[0]] - 0.5) + RNG.normal(0, 0.05, len(p))   # signal in the first feature
    return p


def test_prefix_moments_equal_direct_sums():
    p = _panel()
    w = weights_for(base_draws(len(FEATS), 8, 0), 0.1)
    snaps = prefix_moments(p, FEATS, w, [pd.Timestamp("2004-11-30")])
    sub = p[p["eom"] <= "2004-11-30"]
    direct = Moments.of(rff_raw(sub[FEATS].to_numpy(), w), sub["ret_ld1"].to_numpy())
    np.testing.assert_allclose(snaps[pd.Timestamp("2004-11-30")].sxx, direct.sxx)
    assert snaps[pd.Timestamp("2004-11-30")].n == len(sub)


def test_return_model_predicts_signal_and_does_not_peek_at_the_future():
    p = _panel()
    res = fit_return_model(p, CFG)
    pr = res.predictions.merge(p, on=["id", "eom"])
    assert set(res.chosen["year"]) == {2008, 2009}
    assert pr["pred_ld1"].corr(pr[FEATS[0]]) > 0.3                       # recovers the planted signal
    lo, hi = formation_months_for_year(2008)
    assert pr.loc[pr["year"] == 2008, "eom"].between(lo, hi).all()
    # Scramble every return realised after the 2008 refit cutoff: 2008 predictions must not change.
    _, full_cut = cutoffs_for_year(2008, CFG.return_model.val_years)
    q = p.copy()
    future = q["eom"] > full_cut
    q.loc[future, "ret_ld1"] = RNG.normal(0, 1, future.sum())
    res2 = fit_return_model(q, CFG)
    a = res.predictions[res.predictions["year"] == 2008]["pred_ld1"].to_numpy()
    b = res2.predictions[res2.predictions["year"] == 2008]["pred_ld1"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_prediction_years_stop_at_the_last_open_year():
    from jkmp.config import load_config
    from jkmp.models.return_model import prediction_years

    assert prediction_years(load_config()) == list(range(1971, 2021))                 # locked: 1971 .. 2020
    assert prediction_years(load_config(overrides=["dates.holdout_locked=false"]))[-1] == 2025
