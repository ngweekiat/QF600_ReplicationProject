"""Offline tests for jkmp.covariance (synthetic data)."""

import math

import numpy as np
import pandas as pd
import pytest

from jkmp.covariance import (
    CovParts, daily_factor_regressions, ewma_factor_cov, ewma_vol, idiosyncratic_vol,
    load_covariance, save_covariance, sector_dummies, weighted_cov,
)

RNG = np.random.default_rng(0)
SECTORS = {"consumer": ["NoDur", "Durbl", "Shops"], "industrials": ["Manuf", "Chems", "Other"],
           "tech_telecom": ["BusEq", "Telcm"], "energy_utilities": ["Enrgy", "Utils"],
           "health": ["Hlth"], "finance": ["Money"]}


def ewma_c_reference(x, lam, start):
    """Line-by-line transcription of ewma_c in reference/jkmp_r/ewma.cpp."""
    n = len(x)
    var = [math.nan] * n
    if n <= start:
        return np.sqrt(var)
    na, init = 0, 0.0
    for i in range(start):
        if math.isnan(x[i]):
            na += 1
        else:
            init += x[i] ** 2
    var[start] = init / (start - 1 - na)
    for j in range(start + 1, n):
        var[j] = var[j - 1] if math.isnan(x[j - 1]) else lam * var[j - 1] + (1 - lam) * x[j - 1] ** 2
    return np.sqrt(var)


@pytest.mark.parametrize("with_nan", [False, True])
def test_ewma_vol_matches_cpp(with_nan):
    x = RNG.normal(0, 0.02, 400)
    if with_nan:
        x[[5, 100, 101, 250]] = np.nan
    lam = 0.5 ** (1 / 126)
    np.testing.assert_allclose(ewma_vol(x, lam, 63), ewma_c_reference(list(x), lam, 63), rtol=1e-12, equal_nan=True)


def test_ewma_vol_short_series_is_all_nan():
    assert np.isnan(ewma_vol(np.ones(63), 0.99, 63)).all()


def test_weighted_cov_matches_numpy_aweights():
    x, w = RNG.normal(size=(300, 4)), RNG.uniform(0.1, 1.0, 300)
    np.testing.assert_allclose(weighted_cov(x, w), np.cov(x, rowvar=False, aweights=w, ddof=1))


def test_ewma_factor_cov_combines_corr_and_var_half_lives():
    r = RNG.normal(size=(1000, 3))
    age = np.arange(1000, 0, -1)
    same = ewma_factor_cov(r, 126, 126)
    np.testing.assert_allclose(same, weighted_cov(r, 0.5 ** (age / 126)))
    mixed = ewma_factor_cov(r, 378, 126)
    np.testing.assert_allclose(np.diag(mixed), np.diag(weighted_cov(r, 0.5 ** (age / 126))))
    cor = weighted_cov(r, 0.5 ** (age / 378))
    sd = np.sqrt(np.diag(cor))
    np.testing.assert_allclose(mixed[0, 1] / np.sqrt(mixed[0, 0] * mixed[1, 1]), cor[0, 1] / (sd[0] * sd[1]))


def test_cov_parts_full_and_vol():
    X, F, ivar = RNG.normal(size=(3, 2)), np.array([[2.0, 0.5], [0.5, 1.0]]), np.array([0.1, 0.2, 0.3])
    p = CovParts(pd.Timestamp("2010-01-31"), np.array([10, 20, 30]), ("a", "b"), X, F, ivar)
    full = X @ F @ X.T + np.diag(ivar)
    np.testing.assert_allclose(p.full(), full)
    np.testing.assert_allclose(p.full([30, 10]), full[np.ix_([2, 0], [2, 0])])
    np.testing.assert_allclose(p.vol([20]), np.sqrt(full[1, 1]))
    with pytest.raises(KeyError):
        p.full([99])


def test_daily_regressions_recover_factor_returns_and_residuals():
    factors = ["sec_a", "sec_b", "theme_x"]
    n = 40
    loadings = pd.DataFrame({
        "eom": pd.Timestamp("2010-01-31"), "id": np.arange(n), "in_top": np.arange(n) < 30,
        "sec_a": (np.arange(n) % 2 == 0).astype(float), "sec_b": (np.arange(n) % 2 == 1).astype(float),
        "theme_x": RNG.normal(size=n),
    })
    days = pd.to_datetime(["2010-02-01", "2010-02-02"])
    f_true = np.array([[0.01, -0.02, 0.005], [0.0, 0.03, -0.01]])
    eps = RNG.normal(0, 0.001, size=(2, n))
    daily = pd.concat([
        pd.DataFrame({"permno": np.arange(n), "date": day,
                      "ret": loadings[factors].to_numpy() @ f_true[i] + eps[i] + 0.0001})
        for i, day in enumerate(days)])
    rf = pd.Series(0.0001, index=days)
    fct, res = daily_factor_regressions(loadings, daily, rf, factors)
    np.testing.assert_allclose(fct[factors].to_numpy(), f_true, atol=2e-3)
    assert (fct["n_obs"] == 30).all() and len(res) == 2 * n
    assert res["in_regression"].sum() == 60   # 10 non-top stocks still get residuals
    out = res[~res["in_regression"]]
    np.testing.assert_allclose(out["res"].abs().max(), np.abs(eps[:, 30:]).max(), atol=3e-3)


def test_idiosyncratic_vol_requires_min_obs_within_window():
    days = pd.bdate_range("2010-01-01", periods=30)
    full = pd.DataFrame({"id": 1, "date": days, "res": RNG.normal(0, 0.01, 30)})
    sparse = pd.DataFrame({"id": 2, "date": days[::3], "res": RNG.normal(0, 0.01, 10)})  # 1 obs every 3 days
    out = idiosyncratic_vol(pd.concat([full, sparse]), days, half_life=5, initial_obs=3, min_obs=4, min_window=5)
    assert set(out["id"]) == {1}  # stock 2 never has 4 obs within 5 trading days
    assert out["eom"].tolist() == [pd.Timestamp("2010-01-31"), pd.Timestamp("2010-02-28")]


def test_sector_dummies_requires_full_ff12_coverage():
    d = sector_dummies(pd.Series(["Hlth", "Money", "Utils"]), SECTORS)
    assert d.sum(axis=1).tolist() == [1.0, 1.0, 1.0] and d.loc[2, "sec_energy_utilities"] == 1.0
    with pytest.raises(ValueError):
        sector_dummies(pd.Series(["Hlth"]), {k: v for k, v in SECTORS.items() if k != "finance"})


def test_save_load_roundtrip(tmp_path):
    p = CovParts(pd.Timestamp("2010-01-31"), np.array([1, 2]), ("a", "b"), RNG.normal(size=(2, 2)),
                 np.array([[1.0, 0.2], [0.2, 2.0]]), np.array([0.1, 0.2]))
    save_covariance({p.eom: p}, tmp_path)
    q = load_covariance(tmp_path)[p.eom]
    assert q.factors == p.factors and (q.ids == p.ids).all()
    np.testing.assert_allclose(q.full(), p.full())


@pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
def test_vol_bias_is_one_when_returns_match_sigma_and_respects_the_lock():
    from jkmp.config import load_config
    from jkmp.covariance import CovParts, bias_by_decade, vol_bias
    from jkmp.evaluation.holdout import HoldoutLockedError

    rng = np.random.default_rng(5)
    n, k, months = 40, 3, pd.date_range("1999-01-31", periods=24, freq="ME")
    X, F, ivar = rng.normal(size=(n, k)), np.diag([0.002, 0.001, 0.0005]), np.full(n, 0.004)
    parts = {m: CovParts(eom=m, ids=np.arange(n), factors=("a", "b", "c"), X=X, F=F, ivar=ivar) for m in months}
    chol = np.linalg.cholesky(X @ F @ X.T + np.diag(ivar))
    panel = pd.concat([pd.DataFrame({"eom": m, "id": np.arange(n), "me": rng.uniform(1, 5, n), "in_universe": True,
                                     "ret_ld1": chol @ rng.normal(size=n)}) for m in months], ignore_index=True)
    dates = load_config().dates
    z = vol_bias(parts, panel, months, dates)
    assert set(z["scope"]) == {"stock", "ew", "vw"} and (z["eom_ret"] == z["eom"] + pd.offsets.MonthEnd(1)).all()
    one = z[z["eom"] == months[0]]
    r = panel.loc[panel["eom"] == months[0], "ret_ld1"].to_numpy()
    sd = np.sqrt(np.diag(X @ F @ X.T) + ivar)
    np.testing.assert_allclose(one.loc[one["scope"] == "stock", "z"], r / sd, rtol=1e-12)
    out = bias_by_decade(z)
    assert out["decade"].unique().tolist() == [1990, 2000]                # 1999-02 .. 2001-01 return months
    stock = out[out["scope"] == "stock"]
    np.testing.assert_allclose(stock["ratio"], 1.0, atol=0.15)            # 440 and 520 draws with the true Σ
    late = {m + pd.DateOffset(years=22) + pd.offsets.MonthEnd(0): v for m, v in parts.items()}
    with pytest.raises(HoldoutLockedError):
        vol_bias(late, panel.assign(eom=panel["eom"] + pd.DateOffset(years=22) + pd.offsets.MonthEnd(0)), list(late), dates)
