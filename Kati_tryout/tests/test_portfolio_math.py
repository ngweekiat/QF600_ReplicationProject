"""Portfolio-ML (jkmp.portfolio.portfolio_ml): inputs (signal scaling, the lag weighting against a literal
transcription of R's gtm loop, moments against the full Σ), the coefficient search against lstsq, the
December selection and aims, and the weights π_t = m·π_start + (I − m)·aim with their wealth limits."""

import numpy as np
import pandas as pd
import pytest

from jkmp.covariance import CovParts
from jkmp.models.random_features import column_subset
from jkmp.portfolio.portfolio_ml import (MonthMoments, PfmlMoments, aim_portfolios, build_moments, build_yearly_moments,
                                         december_utility, feature_index, load_yearly_moments, save_yearly_moments,
                                         search_yearly, yearly_sums,
                                         hp_reals, load_moments, month_moments, pfml_weights, ridge_coefs,
                                         save_moments,
                                         scale_signals, search, search_coef, select_hypers, signal_names,
                                         weighted_signals)
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.recursion import month_inputs
from jkmp.portfolio.trading_speed import trading_speed

pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

RNG = np.random.default_rng(11)


def _r_gtm_loop(signals, g, m, lb):
    """Literal transcription of the weighting block of R pfml_input_fun (lists named by lag)."""
    n = m.shape[0]
    gtm = [m @ np.diag(gi) for gi in g]
    gtm_agg, gtm_agg_l1 = [None] * (len(gtm) - 1), [None] * (len(gtm) - 1)
    gtm_agg[0] = np.eye(n)
    gtm_agg_l1[0] = np.eye(n)
    for i in range(1, lb + 1):                                   # R: for (i in 1:lb)
        gtm_agg[i] = gtm_agg[i - 1] @ gtm[i - 1]
        gtm_agg_l1[i] = gtm_agg_l1[i - 1] @ gtm[i]
    omega = const = omega_l1 = const_l1 = 0
    for i in range(lb + 1):                                      # R: for (i in 0:lb)
        omega = omega + gtm_agg[i] @ signals[i]
        const = const + gtm_agg[i]
        omega_l1 = omega_l1 + gtm_agg_l1[i] @ signals[i + 1]
        const_l1 = const_l1 + gtm_agg_l1[i]
    return np.linalg.solve(const, omega), np.linalg.solve(const_l1, omega_l1)


def _random_cov(eom, ids, k=3):
    return CovParts(eom, np.asarray(ids), tuple(f"f{j}" for j in range(k)), RNG.normal(size=(len(ids), k)) * 0.05,
                    np.diag(RNG.uniform(0.001, 0.003, k)), RNG.uniform(0.002, 0.006, len(ids)))


def test_scale_signals_demeans_and_normalises():
    raw = RNG.normal(size=(6, 4))
    s = scale_signals(raw, None)
    np.testing.assert_allclose(s[:, 0], 1 / np.sqrt(6))                  # constant → 1/√n
    np.testing.assert_allclose(s[:, 1:].sum(axis=0), 0, atol=1e-14)      # demeaned
    np.testing.assert_allclose((s**2).sum(axis=0), 1)                    # unit sum of squares
    vol = RNG.uniform(0.05, 0.1, 6)
    np.testing.assert_allclose(scale_signals(raw, vol), s / vol[:, None])  # rows × 1/vol
    flat = scale_signals(np.ones((6, 1)), None)
    assert np.all(flat[:, 1] == 0)                                       # 0/0 column left at zero


@pytest.mark.parametrize("lb", [0, 1, 3, 11])
def test_weighted_signals_matches_r_loop(lb):
    n, P = 5, 4
    signals = [RNG.normal(size=(n, P)) for _ in range(lb + 2)]
    g = [RNG.uniform(0.9, 1.1, n) for _ in range(lb + 2)]
    m = RNG.uniform(0, 0.2, (n, n)) + np.diag(RNG.uniform(0.3, 0.6, n))
    got = weighted_signals(signals, g, m, lb)
    want = _r_gtm_loop(signals, g, m, lb)
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


def test_weighted_signals_limits():
    n, P, lb = 4, 3, 2
    signals = [RNG.normal(size=(n, P)) for _ in range(lb + 2)]
    g = [np.ones(n)] * (lb + 2)
    omega, omega_l1 = weighted_signals(signals, g, np.zeros((n, n)), lb)    # m = 0: trade fully to today's aim
    np.testing.assert_allclose(omega, signals[0])
    np.testing.assert_allclose(omega_l1, signals[1])
    omega, omega_l1 = weighted_signals(signals, g, np.eye(n), lb)           # m = I, g = 1: equal-weighted lags
    np.testing.assert_allclose(omega, np.mean(signals[: lb + 1], axis=0))
    np.testing.assert_allclose(omega_l1, np.mean(signals[1:], axis=0))


def test_month_moments_match_full_sigma():
    n, P = 6, 5
    ids = np.arange(10, 10 + n)
    cov = _random_cov(pd.Timestamp("2020-01-31"), np.arange(8, 18))      # Σ covers more ids than the universe
    omega, omega_l1 = RNG.normal(size=(n, P)), RNG.normal(size=(n, P))
    g_t, r, lam = RNG.uniform(0.9, 1.1, n), RNG.normal(0, 0.05, n), RNG.uniform(1e-11, 1e-10, n)
    r_tilde, risk, tc = month_moments(omega, omega_l1, g_t, r, cov, ids, lam, 1e10, 10.0)
    sigma = cov.full(ids)
    d = omega - np.diag(g_t) @ omega_l1
    np.testing.assert_allclose(r_tilde, omega.T @ r)
    np.testing.assert_allclose(risk, 10.0 * omega.T @ sigma @ omega, rtol=1e-12)
    np.testing.assert_allclose(tc, 1e10 * d.T @ np.diag(lam) @ d, rtol=1e-12)
    with pytest.raises(KeyError):
        month_moments(omega, omega_l1, g_t, r, cov, ids + 100, lam, 1e10, 10.0)


def test_feature_index_matches_r_feature_names():
    names = signal_names([], 8)
    idx = feature_index(4, 8)
    assert [names[i] for i in idx] == ["constant", "rff1_cos", "rff2_cos", "rff1_sin", "rff2_sin"]
    np.testing.assert_array_equal(idx[1:] - 1, column_subset(4, 8))      # same layout as the return model


def _panel(n_ids=7, n_months=8, features=("a", "b", "c")):
    dates = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    rows = []
    for eom in dates:
        for i in range(n_ids):
            rows.append({"id": i + 1, "eom": eom, "in_universe": i < 5, "g": RNG.uniform(0.95, 1.05),
                         "ret_ld1": RNG.normal(0, 0.05), "lambda": RNG.uniform(1e-11, 1e-10),
                         **{f: RNG.uniform() for f in features}})
    panel = pd.DataFrame(rows)
    wealth = pd.DataFrame({"eom": dates, "wealth": 1e10, "mu_ld1": 0.01})
    cov = {eom: _random_cov(eom, np.arange(1, 6)) for eom in dates}
    rf = pd.Series(0.001, index=pd.date_range("2018-12-31", periods=n_months + 2, freq="ME"))
    return panel, wealth, cov, rf, dates, list(features)


@pytest.mark.parametrize("use_rff", [True, False])
def test_build_moments_end_to_end(use_rff, tmp_path):
    panel, wealth, cov, rf, dates, feats = _panel()
    lb = 2
    t = dates[-1:]
    cov[dates[-2]] = _random_cov(dates[-2], np.arange(1, 5))           # id 5 has no Σ at lag 1 → month median vol
    kw = dict(gamma=10.0, mu=0.007, lb=lb, iterations=10, use_rff=use_rff, rff_variance=0.1, p_max=8, seed=1)
    res = build_moments(panel, wealth, cov, rf, t, feats, **kw)
    P1 = 9 if use_rff else 4
    assert res.columns == signal_names(feats, 8 if use_rff else None)
    assert sorted(res.months) == list(t)

    # Recompute one month by hand from the panel
    d = t[-1]
    mm = res.months[d]
    ids = np.arange(1, 6)
    np.testing.assert_array_equal(mm.ids, ids)
    sigs, gs = [], []
    for i in range(lb + 2):
        s = panel[(panel["eom"] == d - pd.offsets.MonthEnd(i)) & panel["id"].isin(ids)].sort_values("id")
        x = s[feats].to_numpy()
        raw = np.hstack([np.cos(x @ res.rff_w), np.sin(x @ res.rff_w)]) if use_rff else x
        c = cov[s["eom"].iloc[0]]
        vol = pd.Series(c.vol(), index=c.ids)
        sigs.append(scale_signals(raw, vol.reindex(ids).fillna(vol.median()).to_numpy()))
        gs.append(s["g"].to_numpy())
    now = panel[(panel["eom"] == d) & panel["id"].isin(ids)].sort_values("id")
    m = trading_speed(cov[d].full(ids), now["lambda"].to_numpy(), 1e10, 10.0, 0.001, 0.007, 10)
    omega, omega_l1 = _r_gtm_loop(sigs, gs, m, lb)
    dlt = omega - np.diag(gs[0]) @ omega_l1
    assert mm.signal.shape == (5, P1)
    np.testing.assert_allclose(mm.signal, sigs[0], rtol=1e-12)
    np.testing.assert_allclose(mm.r_tilde, omega.T @ now["ret_ld1"].to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(mm.risk, 10.0 * omega.T @ cov[d].full(ids) @ omega, rtol=1e-10)
    np.testing.assert_allclose(mm.tc, 1e10 * dlt.T @ np.diag(now["lambda"].to_numpy()) @ dlt, rtol=1e-8, atol=1e-14)

    save_moments(res, tmp_path)
    back = load_moments(tmp_path)
    assert back.columns == res.columns and back.meta["use_rff"] == use_rff
    np.testing.assert_array_equal(back.months[d].denom, mm.denom)
    assert (back.rff_w is None) == (not use_rff)


def test_build_moments_requires_lag_rows():
    panel, wealth, cov, rf, dates, feats = _panel()
    panel = panel[~((panel["id"] == 3) & (panel["eom"] == dates[3]))]
    with pytest.raises(ValueError, match="no row"):
        build_moments(panel, wealth, cov, rf, dates[4:5], feats, gamma=10.0, mu=0.007, lb=2, iterations=10,
                      use_rff=False)


# --- coefficient search, validation, aims ---------------------------------------------------------

P_MAX = 6
LAMBDAS = [0.0, 0.05, 2.0]


def _toy_moments(seed=3, start="2011-12-31", end="2016-11-30"):
    """r̃_t = A_tᵀy_t and Σ̃_t = A_tᵀA_t, so the PF-ML ridge is a stacked least-squares problem."""
    rng = np.random.default_rng(seed)
    cols = signal_names([], P_MAX)
    months, data = {}, {}
    for d in pd.date_range(start, end, freq="ME"):
        a, y = rng.normal(size=(4, len(cols))), rng.normal(size=4)
        data[d] = (a, y)
        months[d] = MonthMoments(d, np.arange(1, 5), a.T @ y, a.T @ a, None, rng.normal(size=(4, len(cols))))
    meta = {"use_rff": True, "p_max": P_MAX, "add_orig": False, "features": ["x1", "x2"], "rff_variance": 0.1}
    return PfmlMoments(cols, meta, months), data


def _lstsq_beta(data, year, cols, lam):
    """Brute force: minimise Σ_t ‖A_t β − y_t‖²/N + λ‖β‖² over the months with returns realised by Dec year−1."""
    use = [d for d in data if (d + pd.offsets.MonthEnd(1)).year <= year - 1]
    n = len(use)
    A = np.vstack([data[d][0][:, cols] for d in use]) / np.sqrt(n)
    y = np.concatenate([data[d][1] for d in use]) / np.sqrt(n)
    A = np.vstack([A, np.sqrt(lam) * np.eye(len(cols))])
    y = np.concatenate([y, np.zeros(len(cols))])
    return np.linalg.lstsq(A, y, rcond=None)[0]


@pytest.mark.parametrize("p", [2, 4, 6])
def test_closed_form_beta_matches_lstsq(p):
    moments, data = _toy_moments()
    coefs = search_coef(moments, [2, 4, 6], LAMBDAS, [2014, 2015, 2016])
    cols = feature_index(p, P_MAX)
    for year in (2014, 2015, 2016):
        for j, lam in enumerate(LAMBDAS):
            np.testing.assert_allclose(coefs[year][p][:, j], _lstsq_beta(data, year, cols, lam), rtol=1e-9, atol=1e-12)


def test_ridge_coefs_equals_solve():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 7))
    d, r = a.T @ a / 30, rng.normal(size=7)
    for j, lam in enumerate(LAMBDAS):
        np.testing.assert_allclose(ridge_coefs(d, r, LAMBDAS)[:, j], np.linalg.solve(d + lam * np.eye(7), r), rtol=1e-10)


def test_coefficients_do_not_peek_at_the_hp_year():
    moments, _ = _toy_moments()
    before = search_coef(moments, [4], LAMBDAS, [2015])
    for d, mm in moments.months.items():
        if (d + pd.offsets.MonthEnd(1)).year >= 2015:                 # scramble everything realised in 2015+
            mm.r_tilde, mm.denom = mm.r_tilde * -7.0, mm.denom * 3.0
    after = search_coef(moments, [4], LAMBDAS, [2015])
    np.testing.assert_array_equal(before[2015][4], after[2015][4])


def test_hp_reals_objective_and_cumulative_mean():
    moments, _ = _toy_moments()
    coefs = search_coef(moments, [2, 6], LAMBDAS, [2015, 2016])
    v = hp_reals(moments, coefs, LAMBDAS)
    assert set(v["hp_end"]) == {2015, 2016} and len(v) == 2 * 12 * 2 * len(LAMBDAS)
    d = pd.Timestamp("2016-03-31")                                     # return month Apr 2016 → β of 2016
    c, b = feature_index(2, P_MAX), coefs[2016][2][:, 1]
    mm = moments.months[d]
    want = mm.r_tilde[c] @ b - 0.5 * b @ mm.denom[np.ix_(c, c)] @ b
    row = v[(v["eom"] == d) & (v["p"] == 2) & (v["l_no"] == 1)].iloc[0]
    assert row["obj"] == pytest.approx(want, rel=1e-12)
    path = v[(v["p"] == 2) & (v["l_no"] == 1)].sort_values("eom_ret")
    assert row["cum_obj"] == pytest.approx(path.loc[path["eom_ret"] <= row["eom_ret"], "obj"].mean(), rel=1e-12)
    assert (v.groupby("eom_ret")["rank"].min() == 1).all()


def test_select_hypers_across_signal_sets_for_next_year():
    months = pd.date_range("2015-01-31", "2016-12-31", freq="ME")
    rows = []
    for tag, p, r15, r16 in [("a", 2, 0.02, 0.0), ("b", 4, 0.0, 0.05)]:
        obj = np.where(months.year == 2015, r15, r16)
        cum = pd.Series(obj).expanding().mean().to_numpy()
        rows.append(pd.DataFrame({"eom_ret": months, "tag": tag, "g": 0.1, "p": p, "l_no": 0, "l": 0.0,
                                  "obj": obj, "cum_obj": cum}))
    best = select_hypers(pd.concat(rows)).set_index("year")
    assert best.loc[2016, "tag"] == "a" and best.loc[2017, "tag"] == "b"


def test_aim_portfolios_use_chosen_model_and_year_coefficients():
    moments, _ = _toy_moments()
    aim_dates = [pd.Timestamp("2015-12-31"), pd.Timestamp("2016-05-31")]     # return year 2016
    res = search(moments, "rff_g0.1000", [2, 4], LAMBDAS, [2015, 2016], aim_dates)
    chosen = pd.DataFrame({"year": [2016], "tag": ["rff_g0.1000"], "p": [4], "l_no": [2], "l": [LAMBDAS[2]]})
    aims, coefs = aim_portfolios({"rff_g0.1000": res}, chosen, aim_dates)
    c = feature_index(4, P_MAX)
    for d in aim_dates:
        want = moments.months[d].signal[:, c] @ res.coefs[2016][4][:, 2]
        np.testing.assert_allclose(aims.loc[aims["eom"] == d, "w_aim"].to_numpy(), want)
    assert list(coefs.loc[coefs["eom"] == aim_dates[0], "column"]) == ["constant", "rff1_cos", "rff2_cos",
                                                                        "rff1_sin", "rff2_sin"]
    with pytest.raises(ValueError, match="no Portfolio-ML hyperparameters"):
        aim_portfolios({"rff_g0.1000": res}, chosen, [pd.Timestamp("2016-12-31")])


# --- weights: π_t = m·π_start + (I − m)·aim ------------------------------------------------------

W_DATES = pd.date_range("2020-01-31", periods=5, freq="ME")
W_IDS = np.array([1, 2, 3])
W_GAMMA, W_MU, W_RF = 10.0, 0.007, 0.001


def _weights_setup(wealth_level):
    rng = np.random.default_rng(5)
    rows, aims = [], []
    base_aim = np.array([0.5, -0.3, 0.8])
    for t, eom in enumerate(W_DATES):
        for j, i in enumerate(W_IDS):
            rows.append({"id": i, "eom": eom, "me": 100.0 * i, "in_universe": True, "lambda": 2e-10 * i,
                         "tr_ld1": 0.01 * (i - 1), "ret_ld1": 0.005 * i})
        aims.append(pd.DataFrame({"id": W_IDS, "eom": eom, "w_aim": base_aim * (-1) ** t}))  # flips every month
    panel = pd.DataFrame(rows)
    wealth = pd.DataFrame({"eom": W_DATES, "wealth": wealth_level, "mu_ld1": 0.01})
    cov = {eom: CovParts(eom, W_IDS, ("f",), rng.normal(size=(3, 1)) * 0.05, np.array([[0.002]]),
                         np.array([0.004, 0.003, 0.005])) for eom in W_DATES}
    rf = pd.Series(W_RF, index=pd.date_range("2020-01-31", periods=7, freq="ME"))
    return panel, wealth, cov, rf, pd.concat(aims, ignore_index=True)


def _run_weights(wealth_level):
    panel, wealth, cov, rf, aims = _weights_setup(wealth_level)
    inputs = month_inputs(panel, wealth, cov, W_DATES, mu_col=None)
    w = pfml_weights(inputs, aims, panel, wealth, rf, gamma=W_GAMMA, mu=W_MU, iterations=10, charge_exit_tc=True)
    return w, portfolio_returns(w, panel, wealth), aims, cov, panel


def test_tiny_wealth_trades_fully_to_the_aim():
    w, pf, aims, _, _ = _run_weights(1e-6)                               # m ≈ 0
    merged = w.merge(aims, on=["id", "eom"])
    np.testing.assert_allclose(merged["w"], merged["w_aim"], atol=1e-10)
    assert (pf["turnover"] > 1.0).all()                                  # the aim flips sign every month


def test_huge_wealth_stops_trading():
    w, pf, _, _, _ = _run_weights(1e20)                                  # m → I
    np.testing.assert_allclose(w.loc[w["eom"] == W_DATES[0], "w"], [1 / 6, 2 / 6, 3 / 6], atol=1e-4)  # stays VW
    assert (pf["turnover"] < 1e-4).all()


def test_turnover_falls_with_wealth():
    turnover = [_run_weights(level)[1]["turnover"].mean() for level in (1e-6, 1e8, 1e10, 1e12, 1e20)]
    assert all(a > b for a, b in zip(turnover, turnover[1:]))


def test_one_recursion_step_by_hand():
    w, _, aims, cov, panel = _run_weights(1e10)
    lam = 2e-10 * W_IDS
    m = trading_speed(cov[W_DATES[0]].full(W_IDS), lam, 1e10, W_GAMMA, W_RF, W_MU, 10)
    vw = np.array([1, 2, 3]) / 6
    aim0 = aims.loc[aims["eom"] == W_DATES[0], "w_aim"].to_numpy()
    first = w[w["eom"] == W_DATES[0]].sort_values("id")
    np.testing.assert_allclose(first["w_start"], vw)
    np.testing.assert_allclose(first["w"], m @ vw + (np.eye(3) - m) @ aim0, rtol=1e-12)
    second = w[w["eom"] == W_DATES[1]].sort_values("id")
    np.testing.assert_allclose(second["w_start"], first["w"].to_numpy() * (1 + 0.01 * (W_IDS - 1)) / 1.01, rtol=1e-12)


def test_missing_aim_raises():
    panel, wealth, cov, rf, aims = _weights_setup(1e10)
    inputs = month_inputs(panel, wealth, cov, W_DATES, mu_col=None)
    with pytest.raises(ValueError, match="aim"):
        pfml_weights(inputs, aims[aims["id"] != 2], panel, wealth, rf, gamma=W_GAMMA, mu=W_MU, iterations=10,
                     charge_exit_tc=True)


# --- yearly sums (grid runs): same β and December selection without monthly matrices ---------------

def test_december_utility_equals_monthly_cumulative_mean():
    moments, _ = _toy_moments(start="2011-12-31", end="2017-11-30")
    coefs = search_coef(moments, [2, 6], LAMBDAS, [2015, 2016, 2017])
    monthly = hp_reals(moments, coefs, LAMBDAS)
    dec = december_utility(yearly_sums(moments.months), moments, coefs, LAMBDAS)
    m = monthly[monthly["eom_ret"].dt.month == 12].merge(dec, on=["eom_ret", "p", "l_no"], suffixes=("_m", "_y"))
    assert len(m) == len(dec) == 3 * 2 * len(LAMBDAS)
    np.testing.assert_allclose(m["cum_obj_m"], m["cum_obj_y"], rtol=1e-10)
    np.testing.assert_array_equal(m["rank_m"], m["rank_y"])


def test_yearly_moments_match_monthly_build_and_round_trip(tmp_path):
    panel, wealth, cov, rf, dates, feats = _panel(n_months=14)
    kw = dict(gamma=10.0, mu=0.007, lb=2, iterations=10, use_rff=True, rff_variance=0.1, p_max=8, seed=1)
    t = dates[3:]
    full = build_moments(panel, wealth, cov, rf, t, feats, **kw)
    ym = build_yearly_moments(panel, wealth, cov, rf, t, feats, signal_dates=t[-2:], **kw)
    want = yearly_sums(full.months)
    assert sorted(ym.years) == sorted(want)
    for y in want:
        assert ym.years[y].n == want[y].n
        np.testing.assert_allclose(ym.years[y].d_sum, want[y].d_sum, rtol=1e-12)
        np.testing.assert_allclose(ym.years[y].r_sum, want[y].r_sum, rtol=1e-12)
    np.testing.assert_array_equal(ym.signals[t[-1]][1], full.months[t[-1]].signal)
    save_yearly_moments(ym, tmp_path / "m.npz")
    back = load_yearly_moments(tmp_path / "m.npz")
    assert back.columns == ym.columns and back.meta == ym.meta and sorted(back.signals) == sorted(ym.signals)
    np.testing.assert_array_equal(back.years[2019].d_sum, ym.years[2019].d_sum)
    np.testing.assert_array_equal(back.signals[t[-1]][0], ym.signals[t[-1]][0])
    np.testing.assert_array_equal(back.rff_w, ym.rff_w)


def test_yearly_search_picks_the_same_hyperparameters():
    moments, _ = _toy_moments(start="2011-12-31", end="2017-11-30")
    aim_dates = [pd.Timestamp("2016-06-30")]
    monthly = search(moments, "t", [2, 4, 6], LAMBDAS, [2015, 2016, 2017], aim_dates)
    sigs = {d: (moments.months[d].ids, moments.months[d].signal) for d in aim_dates}
    from jkmp.portfolio.portfolio_ml import YearlyMoments
    ym = YearlyMoments(moments.columns, moments.meta, yearly_sums(moments.months), sigs)
    yearly = search_yearly(ym, "t", [2, 4, 6], LAMBDAS, [2015, 2016, 2017])
    a, b = select_hypers(monthly.validation), select_hypers(yearly.validation)
    pd.testing.assert_frame_equal(a[["year", "p", "l_no"]], b[["year", "p", "l_no"]])
    np.testing.assert_allclose(a["cum_obj"], b["cum_obj"], rtol=1e-10)


# --- benchmark-relative (active) Portfolio-ML, D34 -------------------------------------------------

def test_scale_signals_and_feature_index_without_constant():
    raw = np.random.default_rng(1).normal(size=(6, 4))
    with_c, without = scale_signals(raw, None), scale_signals(raw, None, add_constant=False)
    assert with_c.shape == (6, 5) and without.shape == (6, 4)
    np.testing.assert_allclose(without, with_c[:, 1:])
    np.testing.assert_allclose(without.sum(axis=0), 0.0, atol=1e-12)            # every column demeaned
    assert feature_index(4, 8, constant=False).tolist() == [0, 1, 4, 5]          # cos 1–2, sin 1–2
    assert feature_index(4, 8).tolist() == [0, 1, 2, 5, 6]


def test_active_recursion_tracks_vw_plus_active_book():
    panel, wealth, cov, rf, aims = _weights_setup(1e10)
    inputs = month_inputs(panel, wealth, cov, W_DATES, mu_col=None)
    kw = dict(gamma=W_GAMMA, mu=W_MU, iterations=10, charge_exit_tc=True, benchmark_relative=True)
    vw = np.array([1, 2, 3]) / 6
    zero = aims.assign(w_aim=0.0)
    w0 = pfml_weights(inputs, zero, panel, wealth, rf, **kw)    # no active view: VW, then trade drift back at speed m
    for k, (eom, g) in enumerate(w0.groupby("eom")):
        g = g.sort_values("id")
        mk = trading_speed(cov[eom].full(W_IDS), 2e-10 * W_IDS, 1e10, W_GAMMA, W_RF, W_MU, 10)
        expect = vw if k == 0 else vw + mk @ (g["w_start"].to_numpy() - vw)
        np.testing.assert_allclose(g["w"], expect, atol=1e-12)
    w = pfml_weights(inputs, aims, panel, wealth, rf, **kw)
    m = trading_speed(cov[W_DATES[0]].full(W_IDS), 2e-10 * W_IDS, 1e10, W_GAMMA, W_RF, W_MU, 10)
    aim0 = aims.loc[aims["eom"] == W_DATES[0], "w_aim"].to_numpy()
    first = w[w["eom"] == W_DATES[0]].sort_values("id")
    np.testing.assert_allclose(first["w"], vw + (np.eye(3) - m) @ aim0, rtol=1e-12)   # a_inh = 0 at the VW start
    second = w[w["eom"] == W_DATES[1]].sort_values("id")
    m1 = trading_speed(cov[W_DATES[1]].full(W_IDS), 2e-10 * W_IDS, 1e10, W_GAMMA, W_RF, W_MU, 10)
    aim1 = aims.loc[aims["eom"] == W_DATES[1], "w_aim"].to_numpy()
    a_inh = second["w_start"].to_numpy() - vw
    np.testing.assert_allclose(second["w"], vw + m1 @ a_inh + (np.eye(3) - m1) @ aim1, rtol=1e-12)
    # Δπ = Δa exactly: the trade is the change of the active book against today's value weights
    np.testing.assert_allclose(second["w"] - second["w_start"], (second["w"].to_numpy() - vw) - a_inh, atol=1e-15)


# --- half-life discounting of β, D36 -----------------------------------------------------------------

def _toy_yearly():
    from jkmp.portfolio.portfolio_ml import YearlyMoments
    moments, data = _toy_moments(start="2009-12-31", end="2017-11-30")
    d = pd.Timestamp("2016-06-30")
    return YearlyMoments(moments.columns, moments.meta, yearly_sums(moments.months),
                         {d: (moments.months[d].ids, moments.months[d].signal)}), data


def test_infinite_half_life_reproduces_the_undiscounted_betas_exactly():
    ym, _ = _toy_yearly()
    years = [2013, 2014, 2015, 2016, 2017]
    base = search_yearly(ym, "t", [2, 4, 6], LAMBDAS, years)
    tuned = search_yearly(ym, "t", [2, 4, 6], LAMBDAS, years, half_lives=(3, 5, 10, None))
    for y in years:
        for p in (2, 4, 6):
            np.testing.assert_array_equal(tuned.coefs_by_h[np.inf][y][p], base.coefs[y][p])   # bit for bit
    inf_rows = tuned.validation[tuned.validation["h"] == np.inf].drop(columns="h").reset_index(drop=True)
    pd.testing.assert_frame_equal(inf_rows, base.validation.reset_index(drop=True)[inf_rows.columns])
    assert set(tuned.validation["h"]) == {3.0, 5.0, 10.0, np.inf}
    assert "h" in select_hypers(tuned.validation)


def test_discounted_beta_by_hand():
    ym, _ = _toy_yearly()
    h, y = 3.0, 2016
    delta = 0.5 ** (1 / h)
    use = [s for s in sorted(ym.years) if s <= y - 1]
    wts = np.array([delta ** (y - s) for s in use])
    r_bar = sum(w * ym.years[s].r_sum for w, s in zip(wts, use)) / sum(w * ym.years[s].n for w, s in zip(wts, use))
    d_bar = sum(w * ym.years[s].d_sum for w, s in zip(wts, use)) / sum(w * ym.years[s].n for w, s in zip(wts, use))
    from jkmp.portfolio.portfolio_ml import coefs_from_sums
    got = coefs_from_sums(ym.years, ym, [4], LAMBDAS, [y], half_life=h)[y][4]
    c = feature_index(4, P_MAX)
    np.testing.assert_allclose(got, ridge_coefs(d_bar[np.ix_(c, c)], r_bar[c], LAMBDAS), rtol=1e-10)


def test_aims_use_the_chosen_half_life():
    ym, _ = _toy_yearly()
    res = search_yearly(ym, "t", [2, 4, 6], LAMBDAS, [2016], half_lives=(3, None))
    d = pd.Timestamp("2016-06-30")
    chosen = pd.DataFrame({"year": [2016], "tag": ["t"], "p": [4], "l_no": [1], "l": [LAMBDAS[1]], "h": [3.0]})
    aims, _ = aim_portfolios({"t": res}, chosen, [d])
    c = feature_index(4, P_MAX)
    np.testing.assert_allclose(aims["w_aim"], ym.signals[d][1][:, c] @ res.coefs_by_h[3.0][2016][4][:, 1])
    assert not np.allclose(res.coefs_by_h[3.0][2016][4], res.coefs_by_h[np.inf][2016][4])


def test_half_life_config_accepts_null():
    from jkmp.config import load_config
    from workflow import config_path
    assert load_config(config_path("us_large","iteration_3"),overrides=["portfolio_ml.half_life_years=[3, 5, 10, null]"]).portfolio_ml.half_life_years == (3.0, 5.0, 10.0, None)
