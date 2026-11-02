"""Table 2 statistics, Table 3 (probability of outperformance) and Table 4 (correlations). Ports of ``pf_summary``,
``utility_t``, ``prob_outperformance`` and the return correlations in reference/jkmp_r/6 - Base analysis.R.
Active (benchmark-relative) statistics against the value-weighted universe are our extension (D34).

Input: monthly performance rows (``jkmp.portfolio.accounting.portfolio_returns``) with columns
``type, r, tc, turnover, inv, shorting``. As in R, volatility, the net Sharpe ratio and utility all use
the variance of the **gross** return r (not r − tc); sample moments use n − 1 like R's ``sd``/``var``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from jkmp.config import DatesConfig
from jkmp.covariance import CovParts
from jkmp.evaluation.holdout import check_return_months

TABLE2_COLUMNS = ["n", "R", "Vol", "SR_gross", "TC", "R_TC", "SR_net", "Utility", "Turnover", "Lev", "Shorting"]


def utility_flow(pf: pd.DataFrame, gamma: float) -> pd.Series:
    """Monthly realised utility r − tc − ½γ(r − r̄)², r̄ the mean gross return of each ``type``."""
    rbar = pf.groupby("type")["r"].transform("mean")
    return pf["r"] - pf["tc"] - 0.5 * gamma * (pf["r"] - rbar) ** 2


def summary_stats(pf: pd.DataFrame, gamma: float) -> pd.Series:
    """One method's Table 2 row (annualised except Turnover, Lev and Shorting)."""
    r, tc = pf["r"], pf["tc"]
    sd = r.std(ddof=1)
    return pd.Series({
        "n": len(pf),
        "R": r.mean() * 12,
        "Vol": sd * np.sqrt(12),
        "SR_gross": r.mean() / sd * np.sqrt(12),
        "TC": tc.mean() * 12,
        "R_TC": (r - tc).mean() * 12,
        "SR_net": (r - tc).mean() / sd * np.sqrt(12),
        "Utility": (r.mean() - 0.5 * r.var(ddof=1) * gamma - tc.mean()) * 12,
        "Turnover": pf["turnover"].mean(),
        "Lev": pf["inv"].mean(),
        "Shorting": pf["shorting"].mean(),
    })


def table2(pfs: pd.DataFrame, gamma: float, dates: DatesConfig) -> pd.DataFrame:
    """Table 2 rows, one per ``type`` (in order of first appearance); refuses locked holdout months."""
    check_return_months(pfs, dates, "table2")
    rows = {t: summary_stats(g, gamma) for t, g in pfs.groupby("type", sort=False)}
    out = pd.DataFrame(rows).T[TABLE2_COLUMNS]
    out["n"] = out["n"].astype(int)
    return out.rename_axis("type")


def _wide(pfs: pd.DataFrame, value: pd.Series, types: Sequence[str]) -> pd.DataFrame:
    """Return month × type; every type must cover the same months."""
    wide = pfs.assign(value=value).pivot(index="eom_ret", columns="type", values="value")[list(types)]
    if wide.isna().any().any():
        raise ValueError("methods cover different return months")
    return wide


def prob_outperformance(pfs: pd.DataFrame, gamma: float, dates: DatesConfig, types: Sequence[str]) -> pd.DataFrame:
    """Table 3: posterior probability that the row method's mean utility flow beats the column method's.
    R ``prob_outperformance``: d_t = u_row,t − u_col,t, P = 1 − Φ(0; mean d, sd(d)/√T), with u_t = ``utility_flow``."""
    check_return_months(pfs, dates, "prob_outperformance")
    u = _wide(pfs, utility_flow(pfs, gamma), types)
    out = pd.DataFrame(np.nan, index=list(types), columns=list(types))
    for a in types:
        for b in types:
            if a != b:
                d = u[a] - u[b]
                out.loc[a, b] = 1.0 - norm.cdf(0.0, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
    return out.rename_axis("type")


def return_correlations(pfs: pd.DataFrame, dates: DatesConfig, types: Sequence[str]) -> pd.DataFrame:
    """Table 4: Pearson correlation of the monthly gross returns r (R: ``cor(method = "pearson")``)."""
    check_return_months(pfs, dates, "return_correlations")
    return _wide(pfs, pfs["r"], types).corr(method="pearson").rename_axis("type")


def subperiod_table(pfs: pd.DataFrame, gamma: float, dates: DatesConfig,
                    periods: Mapping[str, tuple[str, str]], columns: Sequence[str]) -> pd.DataFrame:
    """Table 2 ``columns`` within each (label: (first, last) return month) period; type × (column, period)."""
    parts = {label: table2(pfs[pfs["eom_ret"].between(pd.Timestamp(a), pd.Timestamp(b))], gamma, dates)[list(columns)]
             for label, (a, b) in periods.items()}
    out = pd.concat(parts, axis=1).swaplevel(axis=1)
    return out[[(c, label) for c in columns for label in periods]]


# --- Benchmark-relative statistics (extension, D34) ---------------------------------------------------

def active_returns(pf: pd.DataFrame, bench: pd.DataFrame, dates: DatesConfig) -> pd.DataFrame:
    """Monthly active return vs the benchmark: ``active`` = r − r_b (gross), ``active_net`` = (r − tc) − (r_b − tc_b),
    plus ``r`` and ``r_b``. ``pf`` and ``bench`` are single-method performance rows over the same months."""
    check_return_months(pf, dates, "active_returns")
    d = pf[["eom", "eom_ret", "r", "tc"]].merge(bench[["eom_ret", "r", "tc"]], on="eom_ret", suffixes=("", "_b"),
                                                validate="one_to_one")
    if len(d) != len(pf) or len(d) != len(bench):
        raise ValueError("strategy and benchmark cover different return months")
    return d.assign(active=d["r"] - d["r_b"], active_net=(d["r"] - d["tc"]) - (d["r_b"] - d["tc_b"]))


def active_utility_flow(active: pd.DataFrame, gamma_te: float) -> pd.Series:
    """Monthly active utility: active net return − (γ_TE/2)(active − mean active)² (gross active variance, as R4)."""
    a = active["active"]
    return active["active_net"] - 0.5 * gamma_te * (a - a.mean()) ** 2


def prob_greater(x: pd.Series, y: pd.Series | float = 0.0) -> float:
    """Posterior P(mean of x − y > 0) under R's ``prob_outperformance`` rule: 1 − Φ(0; mean d, sd(d)/√T)."""
    d = np.asarray(x - y, dtype=float)
    return float(1.0 - norm.cdf(0.0, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d))))


def active_weights(weights: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """a = w − π^VW per universe stock and formation month (exit rows dropped): ``id, eom, w, vw, a``."""
    w = weights.loc[~weights["exit"], ["id", "eom", "w"]].merge(panel[["id", "eom", "me"]], on=["id", "eom"], how="left")
    if w["me"].isna().any():
        raise ValueError("held stocks without market cap")
    vw = w["me"] / w.groupby("eom")["me"].transform("sum")
    return w.assign(vw=vw, a=w["w"] - vw)[["id", "eom", "w", "vw", "a"]]


def ex_ante_exposures(active: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts]) -> pd.DataFrame:
    """Per formation month: ex-ante active volatility √(aᵀΣa) (monthly; ``te``) and the factor exposures Xᵀa
    (one column per covariance factor: sectors and themes)."""
    rows = []
    for eom, g in active.groupby("eom", sort=True):
        c = cov_parts[eom]
        a = pd.Series(g["a"].to_numpy(), index=g["id"].to_numpy()).reindex(c.ids, fill_value=0.0)
        if not set(g["id"]) <= set(c.ids):
            raise KeyError(f"active positions without covariance at {eom.date()}")
        a = a.to_numpy()
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            b = c.X.T @ a
            var = float(b @ c.F @ b + (c.ivar * a**2).sum())
        if not (np.isfinite(var) and np.isfinite(b).all()):
            raise FloatingPointError(f"non-finite active risk at {eom.date()}")
        rows.append({"eom": eom, "te": np.sqrt(var), **dict(zip(c.factors, b))})
    return pd.DataFrame(rows)


def active_stats(pf: pd.DataFrame, bench: pd.DataFrame, weights: pd.DataFrame, panel: pd.DataFrame,
                 cov_parts: Mapping[pd.Timestamp, CovParts], gamma_te: float, dates: DatesConfig) -> pd.Series:
    """Annualised active statistics of one method against the benchmark (the VW universe, ``bench``)."""
    act = active_returns(pf, bench, dates)
    aw = active_weights(weights[weights["eom"].isin(pf["eom"])], panel)
    ex = ex_ante_exposures(aw, cov_parts)
    a, an, sd = act["active"], act["active_net"], act["active"].std(ddof=1)
    rb = act["r_b"]
    return pd.Series({
        "n": len(act),
        "active": a.mean() * 12, "active_net": an.mean() * 12,
        "te_realised": sd * np.sqrt(12), "te_ex_ante": ex["te"].mean() * np.sqrt(12),
        "ir_gross": a.mean() / sd * np.sqrt(12), "ir_net": an.mean() / sd * np.sqrt(12),
        "p_ir_net_pos": prob_greater(an),
        "beta": float(np.cov(act["r"], rb, ddof=1)[0, 1] / rb.var(ddof=1)),
        "neg_weight_share": float((aw["w"] < 0).groupby(aw["eom"]).mean().mean()),
        "active_utility": active_utility_flow(act, gamma_te).mean() * 12,
        "tc": act["tc"].mean() * 12, "tc_bench": act["tc_b"].mean() * 12,
    })


def deflated_sharpe(returns: pd.Series, trial_sharpes: Sequence[float], n_trials: int) -> dict:
    """Deflated Sharpe ratio (Bailey and López de Prado, 2014) of per-period ``returns`` after ``n_trials``
    trials whose per-period Sharpe ratios are ``trial_sharpes`` (their variance sets the expected maximum):
    SR₀ = √V[SR]·((1 − γₑ)Φ⁻¹(1 − 1/N) + γₑΦ⁻¹(1 − 1/(N e))),
    DSR = Φ((SR − SR₀)√(T − 1) / √(1 − γ₃SR + (γ₄ − 1)/4·SR²)), γ₃ skewness, γ₄ (non-excess) kurtosis."""
    r = np.asarray(returns, dtype=float)
    t, sr = len(r), r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=0)
    skew, kurt = float((z**3).mean()), float((z**4).mean())
    euler = 0.5772156649015329
    v = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1)) if len(trial_sharpes) > 1 else 0.0
    sr0 = (np.sqrt(v) * ((1 - euler) * norm.ppf(1 - 1 / n_trials) + euler * norm.ppf(1 - 1 / (n_trials * np.e)))
           if n_trials > 1 else 0.0)
    dsr = norm.cdf((sr - sr0) * np.sqrt(t - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))
    return {"T": t, "sr": float(sr), "sr_annual": float(sr * np.sqrt(12)), "sr0": float(sr0),
            "sr0_annual": float(sr0 * np.sqrt(12)), "skew": skew, "kurtosis": kurt, "n_trials": n_trials,
            "var_trial_sr": v, "dsr": float(dsr)}
