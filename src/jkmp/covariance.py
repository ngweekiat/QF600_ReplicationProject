"""Barra-style covariance matrix (Eqs. 36–37). Port of reference/jkmp_r/3 - Estimate Covariance Matrix.R.

    Σ_t = X_t F_t X_tᵀ + diag(ivar_t)          (monthly units: daily estimates × 21)

* Loadings X_t: 6 sector dummies (merged Fama-French 12) + 6 theme scores, standardised each month
  over the factor-estimation universe (the top 500 by market cap).
* Factor returns: each trading day, a no-intercept OLS of the top-500 stocks' excess returns on the
  previous month-end's loadings (R: ``lm(ret_exc ~ -1 + industries + clusters)``).
* F_t: exponentially weighted factor covariance over the last 10 years of daily factor returns, with
  correlations from a 378-day half-life and variances from a 126-day half-life, combined as sd·cor·sd.
* ivar_t: EWMA of each stock's squared residuals (126-day half-life; ``ewma_vol`` ports ewma.cpp),
  kept only with 200 observations in the last 252 trading days; otherwise the median of the stock's
  size group that month.

Residuals are also computed for stocks outside the regression sample (return minus loadings × factor
returns), so a universe stock that falls out of the top 500 keeps an idiosyncratic-risk history.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from jkmp.config import Config
from jkmp.data.prepare import FF12_RANGES
from jkmp.features import THEME_PREFIX, standardize_by_month

log = logging.getLogger(__name__)

SECTOR_PREFIX = "sec_"
FF12_NAMES = [name for name, _ in FF12_RANGES] + ["Other"]
COV_FILES = {"loadings": "cov_loadings.parquet", "factor": "cov_factor.parquet"}


# --------------------------------------------------------------------------- container


@dataclass(frozen=True, eq=False)
class CovParts:
    """Factor-model pieces of Σ at one month-end, for the universe stocks ``ids`` (sorted)."""

    eom: pd.Timestamp
    ids: np.ndarray            # (n,) stock ids
    factors: tuple[str, ...]   # (k,) factor names: sec_* then theme_*
    X: np.ndarray              # (n, k) loadings
    F: np.ndarray              # (k, k) monthly factor covariance
    ivar: np.ndarray           # (n,) monthly idiosyncratic variance

    def _rows(self, ids: Sequence[int] | None) -> np.ndarray | slice:
        if ids is None:
            return slice(None)
        pos = pd.Index(self.ids).get_indexer(np.asarray(ids))
        if (pos < 0).any():
            missing = np.asarray(ids)[pos < 0]
            raise KeyError(f"{len(missing)} ids have no covariance at {self.eom.date()}, e.g. {missing[:5]}")
        return pos

    def full(self, ids: Sequence[int] | None = None) -> np.ndarray:
        """Σ = X F Xᵀ + diag(ivar) for ``ids`` in the given order (default: all stocks)."""
        rows = self._rows(ids)
        X = self.X[rows]
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            sigma = X @ self.F @ X.T + np.diag(self.ivar[rows])
        return _finite(sigma, "Σ")

    def vol(self, ids: Sequence[int] | None = None) -> np.ndarray:
        """Monthly total volatility sqrt(diag Σ) without building Σ."""
        rows = self._rows(ids)
        X = self.X[rows]
        return _finite(np.sqrt(np.einsum("ij,jk,ik->i", X, self.F, X) + self.ivar[rows]), "vol")


def _finite(a: np.ndarray, what: str) -> np.ndarray:
    if not np.isfinite(a).all():
        raise FloatingPointError(f"non-finite {what}")
    return a


# --------------------------------------------------------------------------- estimators


def ewma_vol(x: np.ndarray, lam: float, start: int) -> np.ndarray:
    """EWMA volatility, a port of ``ewma_c`` (reference/jkmp_r/ewma.cpp).

    var[start] = sum(x[:start]²) / (start - 1 - #missing)       (mean assumed 0)
    var[j]     = lam·var[j-1] + (1 - lam)·x[j-1]²   for j > start  (uses data up to j-1 only)
                 var[j-1]                           if x[j-1] is missing
    Returns sqrt(var); NaN before ``start`` (and everywhere if len(x) <= start).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n <= start:
        return out
    head = x[:start]
    denom = start - 1 - int(np.isnan(head).sum())
    if denom <= 0:
        return out
    var = np.full(n, np.nan)
    var[start] = np.nansum(head**2) / denom
    sq = x**2
    if not np.isnan(sq[start:n - 1]).any():  # fast path: first-order recursive filter
        var[start + 1:] = lfilter([1.0 - lam], [1.0, -lam], sq[start:n - 1], zi=[lam * var[start]])[0]
    else:
        for j in range(start + 1, n):
            var[j] = var[j - 1] if np.isnan(sq[j - 1]) else lam * var[j - 1] + (1.0 - lam) * sq[j - 1]
    return np.sqrt(var)


def weighted_cov(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """R ``cov.wt(x, wt, center = TRUE, method = "unbiased")``: weights normalised to sum to one."""
    w = w / w.sum()
    # Apple Accelerate BLAS raises spurious divide/overflow flags in matmul with tiny weights (the
    # result matches einsum to 1e-16), so silence them and check the result explicitly instead.
    with np.errstate(all="ignore"):
        xc = x - w @ x
        cov = (xc * w[:, None]).T @ xc / (1.0 - np.sum(w**2))
    if not np.isfinite(cov).all():
        raise FloatingPointError("non-finite weighted covariance")
    return cov


def ewma_factor_cov(r: np.ndarray, half_life_corr: float, half_life_var: float) -> np.ndarray:
    """sd·cor·sd from two EWMA weightings of daily factor returns ``r`` (T × k, oldest first).

    R: weights (0.5^(1/hl))^(obs:1), tail(·, T) — the newest day has exponent 1.
    """
    age = np.arange(len(r), 0, -1, dtype=float)
    cov_c = weighted_cov(r, 0.5 ** (age / half_life_corr))
    sd_c = np.sqrt(np.diag(cov_c))
    cor = cov_c / np.outer(sd_c, sd_c)
    sd = np.sqrt(np.diag(weighted_cov(r, 0.5 ** (age / half_life_var))))
    return cor * np.outer(sd, sd)


# --------------------------------------------------------------------------- loadings


def sector_dummies(ff12: pd.Series, sectors: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """0/1 columns ``sec_<sector>``; every Fama-French 12 industry must map to exactly one sector."""
    lookup = {ind: sector for sector, inds in sectors.items() for ind in inds}
    unmapped = sorted(set(FF12_NAMES) - set(lookup))
    if unmapped:
        raise ValueError(f"covariance.sectors does not cover FF12 industries: {unmapped}")
    unknown = sorted(set(ff12.unique()) - set(lookup))
    if unknown:
        raise ValueError(f"unknown FF12 industries in data: {unknown}")
    sector = ff12.map(lookup)
    return pd.DataFrame({SECTOR_PREFIX + s: (sector == s).astype(float) for s in sectors}, index=ff12.index)


def monthly_loadings(panel: pd.DataFrame, top: pd.DataFrame, ids: Sequence[int],
                     sectors: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Loadings at each month-end for ``ids``: eom, id, size_grp, in_top, in_universe, sec_*, theme_*.

    Theme scores are standardised by month with the mean and sd of the top-N stocks (R standardises
    over its estimation universe) and applied to every stock.
    """
    themes = [c for c in panel.columns if c.startswith(THEME_PREFIX)]
    rows = panel.loc[panel["id"].isin(ids), ["eom", "id", "size_grp", "ff12", "in_universe", *themes]]
    rows = rows.merge(top[["eom", "id"]].assign(in_top=True), on=["eom", "id"], how="left")
    rows["in_top"] = rows["in_top"].notna()
    rows[themes] = standardize_by_month(rows[themes], rows["eom"], within=rows["in_top"])
    dummies = sector_dummies(rows["ff12"], sectors)
    return pd.concat([rows.drop(columns="ff12"), dummies], axis=1)


# --------------------------------------------------------------------------- daily regressions


def daily_factor_regressions(loadings: pd.DataFrame, daily: pd.DataFrame, rf_daily: pd.Series,
                             factors: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily cross-sectional OLS (no intercept) on the top-N stocks.

    Returns factor returns (index ``date``, one column per factor, plus ``n_obs`` and ``r2``) and
    residuals for every stock with loadings (``id, date, res, in_regression``). Returns in month t+1
    use loadings from month-end t.
    """
    d = daily.dropna(subset=["ret"])
    rf = d["date"].map(rf_daily)
    if rf.isna().any():
        log.warning("Dropping %d daily returns on dates without a Ken French RF", int(rf.isna().sum()))
    d = d.assign(ret_exc=d["ret"] - rf, eom=(d["date"].dt.to_period("M") - 1).dt.end_time.dt.normalize())
    d = d.dropna(subset=["ret_exc"]).rename(columns={"permno": "id"})
    d = d.merge(loadings[["eom", "id", "in_top", *factors]], on=["id", "eom"], how="inner")
    d = d.dropna(subset=list(factors)).sort_values(["date", "id"], ignore_index=True)

    X = d[list(factors)].to_numpy()
    y = d["ret_exc"].to_numpy()
    top = d["in_top"].to_numpy()
    dates, starts = np.unique(d["date"].to_numpy(), return_index=True)
    ends = np.append(starts[1:], len(d))
    k = len(factors)
    f = np.full((len(dates), k), np.nan)
    n_obs = np.zeros(len(dates), dtype=int)
    r2 = np.full(len(dates), np.nan)
    res = np.full(len(d), np.nan)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
        for i, (a, b) in enumerate(zip(starts, ends)):
            Xi, yi, ti = X[a:b], y[a:b], top[a:b]
            n_obs[i] = ti.sum()
            if n_obs[i] <= k:
                continue
            f[i] = np.linalg.lstsq(Xi[ti], yi[ti], rcond=None)[0]
            res[a:b] = yi - Xi @ f[i]
            rt, yt = res[a:b][ti], yi[ti]
            r2[i] = 1.0 - (rt @ rt) / ((yt - yt.mean()) @ (yt - yt.mean()))
    estimated = n_obs > k
    if not (np.isfinite(f[estimated]).all() and np.isfinite(res[np.repeat(estimated, ends - starts)]).all()):
        raise FloatingPointError("non-finite factor returns or residuals")
    fct = pd.DataFrame(f, index=pd.DatetimeIndex(dates, name="date"), columns=list(factors))
    fct = fct.assign(n_obs=n_obs, r2=r2).dropna(subset=list(factors))
    residuals = pd.DataFrame({"id": d["id"], "date": d["date"], "res": res, "in_regression": top}).dropna(subset=["res"])
    return fct, residuals


# --------------------------------------------------------------------------- idiosyncratic risk


def idiosyncratic_vol(residuals: pd.DataFrame, trading_days: pd.DatetimeIndex, half_life: float,
                      initial_obs: int, min_obs: int, min_window: int) -> pd.DataFrame:
    """Month-end EWMA residual volatility per stock (daily units): columns id, eom, res_vol.

    R: ``ewma_c(res, 0.5^(1/hl), start)`` by stock; keep a day only if the stock's observation
    ``min_obs`` rows back is within the last ``min_window`` trading days and the vol is defined;
    take the last remaining day of each month.
    """
    r = residuals.sort_values(["id", "date"], ignore_index=True)
    lam = 0.5 ** (1.0 / half_life)
    ids, starts = np.unique(r["id"].to_numpy(), return_index=True)
    ends = np.append(starts[1:], len(r))
    res = r["res"].to_numpy()
    vol = np.concatenate([ewma_vol(res[a:b], lam, initial_obs) for a, b in zip(starts, ends)])
    r["res_vol"] = vol

    days = pd.Series(trading_days.sort_values())
    window_start = pd.Series(days.shift(min_window).to_numpy(), index=days.to_numpy())
    date_back = r.groupby("id")["date"].shift(min_obs)
    keep = (date_back >= r["date"].map(window_start)) & r["res_vol"].notna()
    r = r[keep].assign(eom=lambda x: x["date"] + pd.offsets.MonthEnd(0))
    return r.groupby(["id", "eom"], as_index=False).tail(1)[["id", "eom", "res_vol"]].reset_index(drop=True)


# --------------------------------------------------------------------------- assembly


@dataclass
class CovarianceResult:
    parts: dict[pd.Timestamp, CovParts]
    factor_returns: pd.DataFrame   # daily, with n_obs and r2
    diagnostics: pd.DataFrame      # per month-end


def build_covariance(panel: pd.DataFrame, top: pd.DataFrame, daily: pd.DataFrame,
                     ff_daily: pd.DataFrame, cfg: Config) -> CovarianceResult:
    """CovParts for the universe stocks at every universe month-end with enough factor history."""
    c = cfg.covariance
    themes = [col for col in panel.columns if col.startswith(THEME_PREFIX)]
    factors = [SECTOR_PREFIX + s for s in c.sectors] + themes

    loadings = monthly_loadings(panel, top, daily["permno"].unique(), c.sectors)
    rf_daily = ff_daily.set_index("date")["rf"]
    fct, residuals = daily_factor_regressions(loadings, daily, rf_daily, factors)
    log.info("Daily regressions: %d days (%s to %s), median %d stocks, mean R² %.3f",
             len(fct), fct.index.min().date(), fct.index.max().date(), int(fct["n_obs"].median()), fct["r2"].mean())

    ivol = idiosyncratic_vol(residuals, fct.index, c.half_life_idio_days, c.idio_initial_obs,
                             c.idio_min_obs, c.idio_min_window)
    ivol_by_eom = dict(tuple(ivol.groupby("eom")))
    load_by_eom = dict(tuple(loadings.groupby("eom")))
    fr = fct[factors].to_numpy()
    fdates = fct.index.to_numpy()

    universe = panel.loc[panel["in_universe"], ["eom", "id"]]
    universe_size = universe.groupby("eom").size()
    parts, diag, skipped = {}, [], []
    for eom in pd.DatetimeIndex(universe_size.index).sort_values():
        end = int(np.searchsorted(fdates, np.datetime64(eom), side="right"))
        begin = max(0, end - c.window_days)
        if end - begin < c.min_window_days:
            skipped.append(eom)
            continue
        F = ewma_factor_cov(fr[begin:end], c.half_life_corr_days, c.half_life_var_days) * c.days_per_month

        L = load_by_eom[eom].set_index("id")
        pool = L[L["in_top"] | L["in_universe"]].join(
            ivol_by_eom.get(eom, pd.DataFrame(columns=["id", "res_vol"])).set_index("id")["res_vol"], how="left")
        # R: missing res_vol -> median of the stock's size group that month, then the overall median.
        fill = pool.groupby("size_grp")["res_vol"].transform("median").fillna(pool["res_vol"].median())
        res_vol = pool["res_vol"].fillna(fill)

        ids = np.sort(L.index[L["in_universe"]].to_numpy())
        if len(ids) != universe_size[eom]:
            raise ValueError(f"{universe_size[eom] - len(ids)} universe stocks at {eom.date()} have no loadings "
                             "(no CRSP daily data?)")
        parts[eom] = CovParts(
            eom=eom, ids=ids, factors=tuple(factors), X=L.loc[ids, factors].to_numpy(), F=F,
            ivar=res_vol.loc[ids].to_numpy() ** 2 * c.days_per_month,
        )
        diag.append({"eom": eom, "n_stocks": len(ids), "n_ivar_imputed": int(pool.loc[ids, "res_vol"].isna().sum()),
                     "n_factor_days": end - begin, "median_vol_annual": float(np.median(parts[eom].vol()) * np.sqrt(12))})
    if skipped:
        log.info("Skipped %d month-ends with fewer than %d factor-return days (%s to %s)",
                 len(skipped), c.min_window_days, skipped[0].date(), skipped[-1].date())
    if not parts:
        raise ValueError("no month-end has enough factor history")
    return CovarianceResult(parts, fct, pd.DataFrame(diag))


# --------------------------------------------------------------------------- storage


def save_covariance(parts: Mapping[pd.Timestamp, CovParts], out_dir: Path) -> None:
    """Two long tables: loadings + ivar per (eom, id), and F per (eom, factor)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    load, fac = [], []
    for eom, p in parts.items():
        load.append(pd.DataFrame(p.X, columns=p.factors).assign(eom=eom, id=p.ids, ivar=p.ivar))
        fac.append(pd.DataFrame(p.F, columns=p.factors).assign(eom=eom, factor=p.factors))
    pd.concat(load, ignore_index=True).to_parquet(out_dir / COV_FILES["loadings"], index=False)
    pd.concat(fac, ignore_index=True).to_parquet(out_dir / COV_FILES["factor"], index=False)


def load_covariance(in_dir: Path) -> dict[pd.Timestamp, CovParts]:
    load = pd.read_parquet(in_dir / COV_FILES["loadings"])
    fac = pd.read_parquet(in_dir / COV_FILES["factor"])
    factors = tuple(c for c in fac.columns if c not in ("eom", "factor"))
    parts = {}
    for eom, F in fac.groupby("eom", sort=True):
        L = load[load["eom"] == eom].sort_values("id")
        F = F.set_index("factor").loc[list(factors), list(factors)]
        parts[eom] = CovParts(eom=eom, ids=L["id"].to_numpy(), factors=factors,
                              X=L[list(factors)].to_numpy(), F=F.to_numpy(), ivar=L["ivar"].to_numpy())
    return parts
