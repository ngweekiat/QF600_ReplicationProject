"""Portfolio-ML (Eqs. 35–37 and Proposition 3). Ports of ``pfml_input_fun``, ``pfml_search_coef``,
``pfml_hp_reals_fun``, ``pfml_aims_fun``, ``pfml_w`` and ``pfml_implement`` (0 - Portfolio choice functions.R).

For each formation month t and the universe stocks at t (n of them), with P signals + a constant:

  signals  S_s (n × P+1), s = t, t−1, …, t−lb−1: features of the *current* ids at month s
           (random Fourier features of the ranked characteristics, or the characteristics themselves),
           demeaned across the n stocks, a constant added, every column scaled to unit sum of squares,
           then rows multiplied by 1/vol (vol = sqrt(diag Σ_s); median of the month if missing).
  weights  G_s = m·diag(g_s), with m the trading speed at t and g_s = (1+tr_ld0)/(1+mu_ld0) at s;
           A_0 = I, A_{i+1} = A_i G_{t−i}   and   A¹_0 = I, A¹_{i+1} = A¹_i G_{t−i−1}.
  Ω_t    = (Σ_i A_i)⁻¹ Σ_i A_i S_{t−i},             i = 0..lb
  Ω_{t−1} = (Σ_i A¹_i)⁻¹ Σ_i A¹_i S_{t−i−1}
  ΔΩ     = Ω_t − diag(g_t) Ω_{t−1}
  r̃ = Ωᵀ r_{t+1},   risk = γ ΩᵀΣΩ (factor form of Σ),   tc = w ΔΩᵀ Λ ΔΩ,   Σ̃ = risk + tc.

Columns are ordered as R's ``pfml_feat_fun(p_max)``: constant, rff1_cos … rff{p/2}_cos,
rff1_sin … rff{p/2}_sin (then the original features if ``add_orig``), so the p-feature model is
``feature_index(p, p_max)``. ``use_rff=False`` uses the ranked characteristics as signals (linear
Portfolio-ML extension). One call handles one RFF variance g, as R runs ``pfml_input_fun`` per g.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from jkmp.covariance import CovParts
from jkmp.models.random_features import base_draws, rff_raw, weights_for
from jkmp.portfolio.accounting import portfolio_returns
from jkmp.portfolio.recursion import MonthInputs, month_inputs, recursive_weights
from jkmp.portfolio.trading_speed import trading_speed

log = logging.getLogger(__name__)

MONTH = pd.offsets.MonthEnd(1)


@dataclass
class MonthMoments:
    eom: pd.Timestamp
    ids: np.ndarray        # (n,) universe ids at eom, sorted
    r_tilde: np.ndarray    # (P+1,)        Ωᵀ r
    denom: np.ndarray      # (P+1, P+1)    Σ̃ = risk + tc
    risk: np.ndarray | None  # (P+1, P+1)  γ ΩᵀΣΩ (None if loaded without it)
    signal: np.ndarray     # (n, P+1)      S_t, the signals that map coefficients to the aim portfolio

    @property
    def tc(self) -> np.ndarray:
        return self.denom - self.risk


@dataclass
class PfmlMoments:
    columns: list[str]
    meta: dict
    months: dict[pd.Timestamp, MonthMoments] = field(default_factory=dict)
    rff_w: np.ndarray | None = None   # (k, p_max/2) random weights W, None for raw characteristics


def signal_set(use_rff: bool, rff_variance_grid: Sequence[float]) -> tuple[str, list[str], str]:
    """(output folder, signal-set tags, method label) for ``portfolio_ml.use_rff``."""
    if use_rff:
        return "rff", [f"rff_g{g:.4f}" for g in rff_variance_grid], "Portfolio-ML"
    return "raw", ["raw"], "Portfolio-ML (linear)"


def signal_names(features: Sequence[str], p_max: int | None, add_orig: bool = False,
                 add_constant: bool = True) -> list[str]:
    """R ``pfml_feat_fun(p_max, orig_feat)``; ``p_max=None`` means the raw characteristics."""
    const = ["constant"] if add_constant else []
    if p_max is None:
        return [*const, *features]
    half = p_max // 2
    names = [*const, *[f"rff{j}_cos" for j in range(1, half + 1)], *[f"rff{j}_sin" for j in range(1, half + 1)]]
    return names + list(features) if add_orig else names


def feature_index(p: int, p_max: int, n_orig: int = 0, constant: bool = True) -> np.ndarray:
    """Columns of the p-feature model in the p_max layout: [constant], cos 1..p/2, sin 1..p/2 (+ originals)."""
    if p % 2 or p > p_max:
        raise ValueError(f"p must be even and <= {p_max}")
    half, half_max, c = p // 2, p_max // 2, int(constant)
    idx = [0] * c + [*range(c, c + half), *range(c + half_max, c + half_max + half)]
    return np.array(idx + list(range(c + p_max, c + p_max + n_orig)))


def scale_signals(raw: np.ndarray, vol: np.ndarray | None, add_constant: bool = True) -> np.ndarray:
    """Demean each column, prepend a constant, scale every column to unit sum of squares, divide rows by vol.

    R (``balanced = F``): ``x - mean(x)``, ``constant := 1``, ``x*sqrt(1/sum(x^2))``, ``diag(1/vol_scale) %*% s``.
    A column with no cross-sectional variation would be 0/0 in R; it is left at zero here. Without
    ``add_constant`` (``portfolio_ml.add_constant: false``) there is no constant column: after the 1/vol scaling it
    is an inverse-volatility long book, which a benchmark-relative active book should not hold (D34).
    """
    x = raw - raw.mean(axis=0)
    s = np.hstack([np.ones((len(raw), 1)), x]) if add_constant else x
    ss = (s**2).sum(axis=0)
    s = np.divide(s, np.sqrt(ss), out=np.zeros_like(s), where=ss > 0)
    return s if vol is None else s / vol[:, None]


def weighted_signals(signals: Sequence[np.ndarray], g: Sequence[np.ndarray], m: np.ndarray, lb: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Ω_t and Ω_{t−1} from ``signals[i]`` = S_{t−i} and ``g[i]`` = g_{t−i}, i = 0..lb+1 (R's gtm loop)."""
    if len(signals) != lb + 2 or len(g) != lb + 2:
        raise ValueError(f"need lb + 2 = {lb + 2} months of signals and growth factors")
    n = m.shape[0]
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
        gtm = [m * gi[None, :] for gi in g]                      # m %*% diag(gt)
        agg, agg_l1 = [np.eye(n)], [np.eye(n)]
        for i in range(lb):
            agg.append(agg[i] @ gtm[i])
            agg_l1.append(agg_l1[i] @ gtm[i + 1])
        omega = sum(a @ signals[i] for i, a in enumerate(agg))
        omega_l1 = sum(a @ signals[i + 1] for i, a in enumerate(agg_l1))
        omega = np.linalg.solve(sum(agg), omega)
        omega_l1 = np.linalg.solve(sum(agg_l1), omega_l1)
    if not (np.isfinite(omega).all() and np.isfinite(omega_l1).all()):
        raise FloatingPointError("non-finite weighted signals")
    return omega, omega_l1


def month_moments(omega: np.ndarray, omega_l1: np.ndarray, g_t: np.ndarray, r: np.ndarray, cov: CovParts,
                  ids: np.ndarray, lam: np.ndarray, wealth: float, gamma: float) -> tuple[np.ndarray, ...]:
    """r̃ = Ωᵀr, risk = γΩᵀΣΩ via Σ = XFXᵀ + diag(ivar), tc = wΔΩᵀΛΔΩ; returns (r̃, risk, tc)."""
    rows = np.searchsorted(cov.ids, ids)
    if (rows >= len(cov.ids)).any() or (cov.ids[np.minimum(rows, len(cov.ids) - 1)] != ids).any():
        raise KeyError(f"universe stocks without covariance loadings at {cov.eom.date()}")
    X, ivar = cov.X[rows], cov.ivar[rows]
    d_omega = omega - g_t[:, None] * omega_l1
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
        r_tilde = omega.T @ r
        b = X.T @ omega
        risk = gamma * (b.T @ cov.F @ b + omega.T @ (ivar[:, None] * omega))
        tc = wealth * (d_omega.T @ (lam[:, None] * d_omega))
    out = r_tilde, 0.5 * (risk + risk.T), 0.5 * (tc + tc.T)  # symmetric up to round-off
    if not all(np.isfinite(a).all() for a in out):
        raise FloatingPointError("non-finite Portfolio-ML moments")
    return out


def build_moments(panel: pd.DataFrame, wealth: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts],
                  rf: pd.Series, dates: Sequence[pd.Timestamp], features: Sequence[str], *, gamma: float,
                  mu: float, lb: int, iterations: int, scale_by_vol: bool = True, use_rff: bool = True,
                  rff_variance: float | None = None, p_max: int | None = None, seed: int | None = None,
                  add_orig: bool = False, add_constant: bool = True,
                  sink: Callable[[MonthMoments], None] | None = None) -> PfmlMoments:
    """R ``pfml_input_fun`` (``balanced = F``) for formation months ``dates``.

    With ``sink``, each month is passed to ``sink`` instead of being kept (``months`` stays empty), so
    callers can reduce months on the fly (``build_yearly_moments``).

    ``rf`` is Ken French's monthly RF indexed by month-end; m at t uses RF of the return month t+1
    (docs/deviations.md D20). Needs ``cov_parts`` for every month t−lb−1 .. t and panel rows for
    every universe stock at t in each of those months (guaranteed by ``valid_data``).
    """
    features = list(features)
    if use_rff:
        if rff_variance is None or p_max is None or seed is None:
            raise ValueError("use_rff needs rff_variance, p_max and seed")
        w_rff = weights_for(base_draws(len(features), p_max, seed), rff_variance)  # R: set.seed(seed); rff(p_max, g)
    else:
        w_rff, p_max, add_orig = None, None, False
    columns = signal_names(features, p_max, add_orig, add_constant)
    dates = pd.DatetimeIndex(sorted(dates))
    w_by = wealth.set_index("eom")["wealth"]
    lag_months = [pd.DatetimeIndex(dates - MONTH * i) for i in range(lb + 2)]

    # Raw signal inputs, once per (id, eom) needed: ids of the universe at t, months t−lb−1 .. t
    universe = panel.loc[panel["in_universe"] & panel["eom"].isin(dates), ["id", "eom"]]
    need = pd.concat([universe.assign(eom=universe["eom"] - MONTH * i) for i in range(lb + 2)]).drop_duplicates()
    rows = panel[["id", "eom", "g", "ret_ld1", "lambda", *features]].merge(need, on=["id", "eom"])
    x = rows[features].to_numpy(dtype=float)
    raw = rff_raw(x, w_rff) if use_rff else x
    if use_rff and add_orig:
        raw = np.hstack([raw, x])
    rows = rows[["id", "eom", "g", "ret_ld1", "lambda"]].reset_index(drop=True)
    by_eom = {eom: idx.to_numpy() for eom, idx in rows.groupby("eom").groups.items()}  # eom -> row positions
    row_id, row_g, row_ret, row_lam = (rows[c].to_numpy() for c in ("id", "g", "ret_ld1", "lambda"))
    vol_by = {}  # eom -> (vol Series over that month's Σ ids, median)

    def vol_for(eom: pd.Timestamp, ids: np.ndarray) -> np.ndarray | None:
        if not scale_by_vol:
            return None
        if eom not in vol_by:
            if eom not in cov_parts:
                raise KeyError(f"no covariance at {eom.date()} (needed for the vol scale)")
            c = cov_parts[eom]
            v = pd.Series(c.vol(), index=c.ids)
            vol_by[eom] = (v, float(v.median()))
        v, med = vol_by[eom]
        return v.reindex(ids).fillna(med).to_numpy()  # R: impute missing with the month's median

    out = PfmlMoments(columns=columns, rff_w=w_rff, meta={
        "use_rff": use_rff, "rff_variance": rff_variance, "p_max": p_max, "seed": seed, "add_orig": add_orig,
        "features": features, "gamma": gamma, "mu": mu, "lb": lb, "iterations": iterations,
        "scale_by_vol": scale_by_vol, "add_constant": add_constant})
    for j, t in enumerate(dates):
        ids = np.sort(universe.loc[universe["eom"] == t, "id"].to_numpy())
        signals, growth, now = [], [], None
        for i in range(lb + 2):
            s_month = lag_months[i][j]
            idx = by_eom.get(s_month, np.array([], dtype=int))
            pos = pd.Index(row_id[idx]).get_indexer(ids)
            if (pos < 0).any():
                raise ValueError(f"{(pos < 0).sum()} universe stocks at {t.date()} have no row at {s_month.date()}")
            r_idx = idx[pos]
            signals.append(scale_signals(raw[r_idx], vol_for(s_month, ids), add_constant))
            growth.append(row_g[r_idx])
            if i == 0:
                now = r_idx
        cov, w_t = cov_parts[t], float(w_by[t])
        lam = row_lam[now]
        m = trading_speed(cov.full(ids), lam, w_t, gamma, float(rf[t + MONTH]), mu, iterations)
        omega, omega_l1 = weighted_signals(signals, growth, m, lb)
        r_tilde, risk, tc = month_moments(omega, omega_l1, growth[0], row_ret[now], cov, ids, lam, w_t, gamma)
        mm = MonthMoments(t, ids, r_tilde, risk + tc, risk, signals[0])
        if sink is None:
            out.months[t] = mm
        else:
            sink(mm)
        if t.month == 12:
            log.info("PF-ML inputs %s: n=%d, P+1=%d", t.date(), len(ids), len(columns))
    return out


def save_moments(moments: PfmlMoments, directory: Path) -> None:
    """``directory/{meta.json, rff_w.npy, YYYY-MM-DD.npz}``; each npz holds ids, r_tilde, denom, risk, signal."""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.npz"):
        old.unlink()
    meta = {**moments.meta, "columns": moments.columns, "dates": [f"{d:%Y-%m-%d}" for d in sorted(moments.months)]}
    (directory / "meta.json").write_text(json.dumps(meta, indent=1))
    if moments.rff_w is not None:
        np.save(directory / "rff_w.npy", moments.rff_w)
    for eom, mm in moments.months.items():
        np.savez(directory / f"{eom:%Y-%m-%d}.npz", ids=mm.ids, r_tilde=mm.r_tilde, denom=mm.denom,
                 risk=mm.risk, signal=mm.signal)


def load_moments(directory: Path, with_risk: bool = True) -> PfmlMoments:
    """Inverse of ``save_moments``; ``with_risk=False`` skips the risk matrices (not needed for the search)."""
    meta = json.loads((directory / "meta.json").read_text())
    columns, dates = meta.pop("columns"), meta.pop("dates")
    w_path = directory / "rff_w.npy"
    out = PfmlMoments(columns=columns, meta=meta, rff_w=np.load(w_path) if w_path.exists() else None)
    for d in dates:
        eom = pd.Timestamp(d)
        with np.load(directory / f"{d}.npz") as z:
            risk = z["risk"] if with_risk else None
            out.months[eom] = MonthMoments(eom, z["ids"], z["r_tilde"], z["denom"], risk, z["signal"])
    return out


# --- Coefficients, validation and aim portfolios (pfml_search_coef, pfml_hp_reals_fun, pfml_aims_fun) ---

def return_year(eom: pd.Timestamp) -> int:
    """Year in which the return of formation month ``eom`` is realised (our year convention)."""
    return (eom + MONTH).year


def model_columns(moments, p: int) -> np.ndarray:
    """Columns of the p-feature model; raw characteristics always use every column."""
    meta = moments.meta
    if not meta["use_rff"]:
        return np.arange(len(moments.columns))
    return feature_index(p, meta["p_max"], len(meta["features"]) if meta["add_orig"] else 0,
                         meta.get("add_constant", True))


def p_values(moments, p_grid: Sequence[int]) -> list[int]:
    """The p grid for RFF signals; for raw characteristics the single model with all of them."""
    return [int(p) for p in p_grid] if moments.meta["use_rff"] else [len(moments.meta["features"])]


def ridge_coefs(denom_mean: np.ndarray, r_mean: np.ndarray, lambdas: Sequence[float]) -> np.ndarray:
    """β(λ) = (Σ̄ + λI)⁻¹ r̄ for every λ (columns) from one eigendecomposition: V diag(1/(d + λ)) Vᵀ r̄.

    R ``solve(denom_sub + l*diag(p+1)) %*% r_tilde_sub``: the constant is penalised like every signal.
    """
    lam = np.asarray(lambdas, dtype=float)
    d, V = np.linalg.eigh(0.5 * (denom_mean + denom_mean.T))
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
        coef = V @ ((V.T @ r_mean)[:, None] / (d[:, None] + lam[None, :]))
    if not np.isfinite(coef).all():
        raise FloatingPointError("non-finite Portfolio-ML coefficients (singular Σ̄ at λ = 0?)")
    return coef


@dataclass
class YearSums:
    """Sums over the months of one return year: count, Σ r̃_t and Σ Σ̃_t."""
    n: int
    r_sum: np.ndarray
    d_sum: np.ndarray


def yearly_sums(months: Mapping[pd.Timestamp, MonthMoments]) -> dict[int, YearSums]:
    out: dict[int, YearSums] = {}
    for d in sorted(months):
        mm, y = months[d], return_year(d)
        if y not in out:
            out[y] = YearSums(0, np.zeros_like(mm.r_tilde), np.zeros_like(mm.denom))
        out[y].n += 1
        out[y].r_sum += mm.r_tilde
        out[y].d_sum += mm.denom
    return out


def coefs_from_sums(sums: Mapping[int, YearSums], owner, p_grid: Sequence[int], lambdas: Sequence[float],
                    hp_years: Sequence[int], half_life: float | None = None) -> dict[int, dict[int, np.ndarray]]:
    """β for each hp year y from the running means of r̃ and Σ̃ over return years ≤ y−1.
    ``owner`` supplies ``meta`` and ``columns`` (a PfmlMoments or YearlyMoments).

    ``half_life`` h (years, D36): year s gets weight δ^{y−s}, δ = 0.5^{1/h}, so r̄_y = Σ δ^{y−s} r_sum_s / Σ δ^{y−s} n_s.
    The accumulators hold Σ δ^{−(s−s₀)} x_s (the common factor δ^{y−s₀} cancels in the ratio); ``None`` gives weight
    1.0, i.e. exactly the equally weighted running means of R ``pfml_search_coef``."""
    n_cols = len(owner.columns)
    r_sum, d_sum, n = np.zeros(n_cols), np.zeros((n_cols, n_cols)), 0.0
    years, k, out = sorted(sums), 0, {}
    for y in sorted(hp_years):
        while k < len(years) and years[k] <= y - 1:
            ys = sums[years[k]]
            wt = 1.0 if half_life is None else 0.5 ** (-(years[k] - years[0]) / half_life)
            if wt == 1.0:
                r_sum, d_sum, n = r_sum + ys.r_sum, d_sum + ys.d_sum, n + ys.n
            else:
                r_sum, d_sum, n = r_sum + wt * ys.r_sum, d_sum + wt * ys.d_sum, n + wt * ys.n
            k += 1
        if n == 0:
            raise ValueError(f"no Portfolio-ML training months before return year {y}")
        r_bar, d_bar = r_sum / n, d_sum / n
        out[y] = {}
        for p in p_values(owner, p_grid):
            c = model_columns(owner, p)
            out[y][p] = ridge_coefs(d_bar[np.ix_(c, c)], r_bar[c], lambdas)
        log.debug("PF-ML coefficients for %d: %d training months", y, n)
    return out


def search_coef(moments: PfmlMoments, p_grid: Sequence[int], lambdas: Sequence[float],
                hp_years: Sequence[int]) -> dict[int, dict[int, np.ndarray]]:
    """R ``pfml_search_coef``: ``{year: {p: β (P_p × L)}}``, β for hp year y from the means of r̃ and Σ̃
    over every month whose return is realised by December y−1 (running sums, updated year by year)."""
    return coefs_from_sums(yearly_sums(moments.months), moments, p_grid, lambdas, hp_years)


def hp_reals(moments: PfmlMoments, coefs: Mapping[int, Mapping[int, np.ndarray]], lambdas: Sequence[float]
             ) -> pd.DataFrame:
    """R ``pfml_hp_reals_fun``: realised utility r̃ᵀβ − ½βᵀΣ̃β of each (p, λ) in each month of hp year y
    (using the β of year y), its cumulative mean since the first hp year, and its rank by return month."""
    lam = np.asarray(lambdas, dtype=float)
    frames = []
    for y, by_p in coefs.items():
        months = [d for d in sorted(moments.months) if return_year(d) == y]
        for p, B in by_p.items():
            c = model_columns(moments, p)
            for d in months:
                mm = moments.months[d]
                with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
                    obj = mm.r_tilde[c] @ B - 0.5 * np.einsum("kl,kl->l", B, mm.denom[np.ix_(c, c)] @ B)
                if not np.isfinite(obj).all():
                    raise FloatingPointError(f"non-finite realised utility at {d.date()}")
                frames.append(pd.DataFrame({"hp_end": y, "eom": d, "eom_ret": d + MONTH, "p": p,
                                            "l_no": np.arange(len(lam)), "l": lam, "obj": obj}))
    v = pd.concat(frames, ignore_index=True).sort_values(["p", "l_no", "eom_ret"], ignore_index=True)
    v["cum_obj"] = v.groupby(["p", "l_no"])["obj"].transform(lambda s: s.expanding().mean())
    v["rank"] = v.groupby("eom_ret")["cum_obj"].rank(ascending=False, method="average")  # R frank(-cum_obj)
    return v


@dataclass
class SearchResult:
    """One signal set (one RFF variance g, or the raw characteristics) after the coefficient search."""
    tag: str
    g: float
    columns: list[str]
    meta: dict
    coefs: dict[int, dict[int, np.ndarray]]
    validation: pd.DataFrame
    signals: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]  # eom -> (ids, S_t) for the aim months
    coefs_by_h: dict[float, dict[int, dict[int, np.ndarray]]] | None = None  # half-life (inf = none) -> coefs (D36)


def search(moments: PfmlMoments, tag: str, p_grid: Sequence[int], lambdas: Sequence[float],
           hp_years: Sequence[int], aim_dates: Sequence[pd.Timestamp]) -> SearchResult:
    """Coefficients and validation for one signal set; keeps only the signals needed for the aims."""
    coefs = search_coef(moments, p_grid, lambdas, hp_years)
    g = moments.meta["rff_variance"] if moments.meta["use_rff"] else float("nan")
    validation = hp_reals(moments, coefs, lambdas).assign(tag=tag, g=g)
    signals = {d: (moments.months[d].ids, moments.months[d].signal) for d in aim_dates}
    return SearchResult(tag, g, moments.columns, moments.meta, coefs, validation, signals)


def december_utility(sums: Mapping[int, YearSums], owner, coefs: Mapping[int, Mapping[int, np.ndarray]],
                     lambdas: Sequence[float]) -> pd.DataFrame:
    """``hp_reals`` at the December return months only, from yearly sums. β is fixed within an hp year,
    so the year's total utility is Σr̃ᵀβ − ½βᵀ(ΣΣ̃)β and the December ``cum_obj`` (all R's selection
    uses) is exact: the running total over hp years divided by the running month count."""
    lam = np.asarray(lambdas, dtype=float)
    frames = []
    for y in sorted(coefs):
        ys = sums[y]
        for p, B in coefs[y].items():
            c = model_columns(owner, p)
            with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
                total = ys.r_sum[c] @ B - 0.5 * np.einsum("kl,kl->l", B, ys.d_sum[np.ix_(c, c)] @ B)
            if not np.isfinite(total).all():
                raise FloatingPointError(f"non-finite realised utility in {y}")
            frames.append(pd.DataFrame({"hp_end": y, "eom_ret": pd.Timestamp(y, 12, 31), "p": p, "n": ys.n,
                                        "l_no": np.arange(len(lam)), "l": lam, "total": total}))
    v = pd.concat(frames, ignore_index=True).sort_values(["p", "l_no", "eom_ret"], ignore_index=True)
    grp = v.groupby(["p", "l_no"])
    v["obj"] = v["total"] / v["n"]
    v["cum_obj"] = grp["total"].cumsum() / grp["n"].cumsum()
    v["rank"] = v.groupby("eom_ret")["cum_obj"].rank(ascending=False, method="average")
    return v.drop(columns="total")


@dataclass
class YearlyMoments:
    """Compact Portfolio-ML inputs: yearly sums of r̃ and Σ̃ plus the signals of selected months."""
    columns: list[str]
    meta: dict
    years: dict[int, YearSums]
    signals: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]]
    rff_w: np.ndarray | None = None


def build_yearly_moments(panel: pd.DataFrame, wealth: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts],
                         rf: pd.Series, dates: Sequence[pd.Timestamp], features: Sequence[str],
                         signal_dates: Sequence[pd.Timestamp], **kwargs) -> YearlyMoments:
    """``build_moments`` reduced on the fly to ``yearly_sums`` (one month in memory at a time)."""
    keep = set(pd.DatetimeIndex(signal_dates))
    years: dict[int, YearSums] = {}
    signals: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = {}

    def sink(mm: MonthMoments) -> None:
        y = return_year(mm.eom)
        if y not in years:
            years[y] = YearSums(0, np.zeros_like(mm.r_tilde), np.zeros_like(mm.denom))
        years[y].n += 1
        years[y].r_sum += mm.r_tilde
        years[y].d_sum += mm.denom
        if mm.eom in keep:
            signals[mm.eom] = (mm.ids, mm.signal)

    shell = build_moments(panel, wealth, cov_parts, rf, dates, features, sink=sink, **kwargs)
    missing = keep - set(signals)
    if missing:
        raise ValueError(f"{len(missing)} signal months are outside the formation months, e.g. {min(missing).date()}")
    return YearlyMoments(shell.columns, shell.meta, years, signals, shell.rff_w)


def save_yearly_moments(ym: YearlyMoments, path: Path) -> None:
    years = sorted(ym.years)
    dates = sorted(ym.signals)
    np.savez(path, meta=json.dumps({**ym.meta, "columns": ym.columns}), years=np.array(years),
             n=np.array([ym.years[y].n for y in years]), r_sum=np.stack([ym.years[y].r_sum for y in years]),
             d_sum=np.stack([ym.years[y].d_sum for y in years]),
             signal_dates=np.array([f"{d:%Y-%m-%d}" for d in dates]),
             counts=np.array([len(ym.signals[d][0]) for d in dates]),
             ids=np.concatenate([ym.signals[d][0] for d in dates]),
             signal=np.concatenate([ym.signals[d][1] for d in dates]),
             rff_w=ym.rff_w if ym.rff_w is not None else np.zeros((0, 0)))


def load_yearly_moments(path: Path) -> YearlyMoments:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        columns = meta.pop("columns")
        years = {int(y): YearSums(int(n), r, d) for y, n, r, d in zip(z["years"], z["n"], z["r_sum"], z["d_sum"])}
        bounds = np.concatenate([[0], np.cumsum(z["counts"])])
        ids, signal = z["ids"], z["signal"]  # each z[...] access re-reads the file
        signals = {pd.Timestamp(str(d)): (ids[a:b], signal[a:b])
                   for d, a, b in zip(z["signal_dates"], bounds[:-1], bounds[1:])}
        rff_w = z["rff_w"] if z["rff_w"].size else None
    return YearlyMoments(columns, meta, years, signals, rff_w)


def search_yearly(ym: YearlyMoments, tag: str, p_grid: Sequence[int], lambdas: Sequence[float],
                  hp_years: Sequence[int], half_lives: Sequence[float | None] = (None,)) -> SearchResult:
    """``search`` from yearly sums: same β and the same December selection, without monthly matrices.
    With several ``half_lives`` (D36) every h gets its own β; the validation rows carry ``h`` (inf = no decay) and
    are scored by the unchanged equally weighted cumulative utility, so h is picked each December like (g, p, λ)."""
    g = ym.meta["rff_variance"] if ym.meta["use_rff"] else float("nan")
    by_h, frames = {}, []
    for h in half_lives:
        key = np.inf if h is None else float(h)
        by_h[key] = coefs_from_sums(ym.years, ym, p_grid, lambdas, hp_years, half_life=h)
        frames.append(december_utility(ym.years, ym, by_h[key], lambdas).assign(h=key))
    validation = pd.concat(frames, ignore_index=True).assign(tag=tag, g=g)
    main = by_h.get(np.inf, next(iter(by_h.values())))
    return SearchResult(tag, g, ym.columns, ym.meta, main, validation, ym.signals, coefs_by_h=by_h)


def select_hypers(validation: pd.DataFrame) -> pd.DataFrame:
    """R ``pfml_implement``: across every (g, p, λ), the highest cumulative realised utility at each
    December return month; that (g, p, λ) is used in the next return year (column ``year``)."""
    v = validation.reset_index(drop=True)
    if "h" not in v:
        v = v.assign(h=np.inf)  # no half-life search: equal weights
    dec = v[v["eom_ret"].dt.month == 12]
    best = dec.loc[dec.groupby("eom_ret")["cum_obj"].idxmax()]
    best = best.assign(year=best["eom_ret"].dt.year + 1)
    return best[["year", "eom_ret", "tag", "g", "p", "l_no", "l", "h", "cum_obj"]].reset_index(drop=True)


def aim_portfolios(results: Mapping[str, SearchResult], chosen: pd.DataFrame, dates: Sequence[pd.Timestamp]
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R ``pfml_aims_fun``: aim weights S_t β for formation months ``dates``, with the (g, p, λ) chosen in
    December y−1 and the β of hp year y (trained on returns through y−1), y = return year of t.
    Returns ``aims`` (id, eom, w_aim) and ``coefs`` (eom, year, tag, p, l, column, beta)."""
    by_year = chosen.set_index("year")
    aims, coefs = [], []
    for d in sorted(dates):
        y = return_year(d)
        if y not in by_year.index:
            raise ValueError(f"no Portfolio-ML hyperparameters chosen for return year {y}")
        h = by_year.loc[y]
        res = results[h["tag"]]
        p, l_no = int(h["p"]), int(h["l_no"])
        c = model_columns(res, p)
        res_coefs = res.coefs if res.coefs_by_h is None else res.coefs_by_h[float(h.get("h", np.inf))]
        beta = res_coefs[y][p][:, l_no]
        ids, signal = res.signals[d]
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            w_aim = signal[:, c] @ beta
        if not np.isfinite(w_aim).all():
            raise FloatingPointError(f"non-finite aim portfolio at {d.date()}")
        aims.append(pd.DataFrame({"id": ids, "eom": d, "w_aim": w_aim}))
        coefs.append(pd.DataFrame({"eom": d, "year": y, "tag": h["tag"], "p": p, "l": h["l"],
                                   "column": [res.columns[i] for i in c], "beta": beta}))
    return pd.concat(aims, ignore_index=True), pd.concat(coefs, ignore_index=True)


# --- Weights and implementation (pfml_w, pfml_implement) -----------------------------------------

def pfml_weights(inputs: Mapping[pd.Timestamp, MonthInputs], aims: pd.DataFrame, panel: pd.DataFrame,
                 wealth: pd.DataFrame, rf: pd.Series, *, gamma: float, mu: float, iterations: int,
                 charge_exit_tc: bool, benchmark_relative: bool = False) -> pd.DataFrame:
    """R ``pfml_w``: π_t = m·π_start + (I − m)·aim_t, with m (Lemma 1) at t for the universe and π_start
    the VW universe in the first month, then the drifted π_{t−1} (new stocks at 0; ``accounting``).

    ``benchmark_relative`` (D34): the aim is an *active* book A_t and the recursion runs on active weights,
    a_t = m·a_inh + (I − m)·A_t with a_inh = π_start − π^VW_t, then π_t = π^VW_t + a_t (π^VW_t: value weights of
    the universe at t, ``benchmarks.market_targets``). Defining a_inh against today's value weights makes the
    traded Δπ = π_t − π_start equal Δa exactly, so the cost charged on the total portfolio is the cost of the
    active trades. The approximation is in the objective only: its ΔΩ = Ω_t − diag(g)Ω_{t−1} lets the active
    book drift with g and ignores the benchmark's own rebalancing (π^VW_t vs the drifted π^VW_{t−1}: universe
    changes and issuance), which is small — the Market row's turnover and cost measure it.

    m uses RF of the return month t+1 (docs/deviations.md D20). Returns ``id, eom, w, w_start, exit``.
    """
    aim_by = {eom: g.set_index("id")["w_aim"] for eom, g in aims.groupby("eom")}

    def target(eom: pd.Timestamp, x: MonthInputs, held: np.ndarray) -> np.ndarray:
        if eom not in aim_by:
            raise ValueError(f"no aim portfolio at {eom.date()}")
        aim = aim_by[eom].reindex(x.ids)
        if aim.isna().any():
            raise ValueError(f"{aim.isna().sum()} universe stocks lack an aim weight at {eom.date()}")
        aim = aim.to_numpy()
        m = trading_speed(x.sigma, x.lam, x.wealth, gamma, float(rf[eom + MONTH]), mu, iterations)
        vw = np.zeros_like(held)
        if benchmark_relative:
            me = x.panel_t.set_index("id")["me"].reindex(x.ids).to_numpy()
            vw = me / me.sum()
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked in recursive_weights
            return vw + m @ (held - vw) + aim - m @ aim

    return recursive_weights(inputs, panel, wealth, target, charge_exit_tc)


def run_portfolio_ml(results: Mapping[str, SearchResult], panel: pd.DataFrame, wealth: pd.DataFrame,
                     cov_parts: Mapping[pd.Timestamp, CovParts], rf: pd.Series, aim_dates: Sequence[pd.Timestamp],
                     eval_dates: Sequence[pd.Timestamp], *, gamma: float, mu: float, iterations: int,
                     charge_exit_tc: bool, label: str = "Portfolio-ML", benchmark_relative: bool = False
                     ) -> dict[str, pd.DataFrame]:
    """R ``pfml_implement`` after the per-g search: pick (g, p, λ) each December across all g, build the
    aim portfolios for ``aim_dates`` and run the weights over ``eval_dates`` from VW initial weights."""
    validation = pd.concat([r.validation for r in results.values()], ignore_index=True)
    chosen = select_hypers(validation)
    years = {return_year(d) for d in aim_dates}
    aims, coefs = aim_portfolios(results, chosen[chosen["year"].isin(years)], aim_dates)
    eval_dates = pd.DatetimeIndex(sorted(eval_dates))
    inputs = month_inputs(panel, wealth, cov_parts, eval_dates, mu_col=None)
    weights = pfml_weights(inputs, aims, panel, wealth, rf, gamma=gamma, mu=mu, iterations=iterations,
                           charge_exit_tc=charge_exit_tc, benchmark_relative=benchmark_relative)
    pf = portfolio_returns(weights, panel, wealth).assign(type=label)
    log.info("%s: chose\n%s", label, chosen.to_string(index=False))
    return {"validation": validation, "chosen": chosen, "aims": aims, "aim_coefs": coefs,
            "weights": weights.assign(type=label), "pf": pf}
