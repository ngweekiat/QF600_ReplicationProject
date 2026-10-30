"""Feature importance by theme (Figure 5). Port of ``pfml_cf_fun`` and ``tpf_cf_fun``
(0 - Portfolio choice functions.R), run by ``5 - Feature importance base.R`` and plotted in
``6 - Feature importance.R``.

For each theme, every universe stock's characteristics in that theme are replaced, month by month, by
those of another universe stock (one random permutation per month, the same permutation for every
theme and method: R reseeds before each theme). Nothing is re-estimated:

* Portfolio-ML: the random features are rebuilt with the **same** W, then the same scaling as the base
  signal S_t (demean, constant, unit sum of squares, 1/vol). The aim S_t β uses the **same** β and
  (g, p, λ), and the weights π_t = m·π_start + (I − m)·aim_t run as in the base case. The result is
  evaluated net of trading costs.
* Markowitz-ML: pred_ld1 is recomputed from the shuffled characteristics with each year's fitted
  return model (same W, p, λ, β), then w = Σ⁻¹μ/γ with R's γ = 100 and zero wealth (gross).

Importance = realised utility (mean r − ½γ var r − mean tc, ×12, γ = the investor's) of the
unshuffled run minus that of the shuffled run. The unshuffled run reproduces the base case.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from jkmp.config import DatesConfig  # noqa: E402
from jkmp.covariance import CovParts  # noqa: E402
from jkmp.evaluation.holdout import check_return_months  # noqa: E402
from jkmp.models.random_features import rff_raw  # noqa: E402
from jkmp.models.return_model import YearModel  # noqa: E402
from jkmp.portfolio.accounting import add_start_weights, portfolio_returns  # noqa: E402
from jkmp.portfolio.ml_benchmarks import markowitz_targets  # noqa: E402
from jkmp.portfolio.portfolio_ml import pfml_weights, return_year, scale_signals  # noqa: E402
from jkmp.portfolio.recursion import MonthInputs  # noqa: E402

BASE = "none"  # label of the unshuffled run (R: "bm")


def permutations(universe: pd.DataFrame, seed: int) -> dict[pd.Timestamp, np.ndarray]:
    """One permutation of the (id-sorted) universe per month; stock i takes the theme values of perm[i].
    R ``set.seed(seed); cf[, id_shuffle := sample(id, .N), by = eom]`` — one draw per month in date order."""
    rng = np.random.default_rng(seed)
    return {eom: rng.permutation(len(g)) for eom, g in universe.groupby("eom", sort=True)}


def shuffled(rows: pd.DataFrame, features: Sequence[str], theme_features: Sequence[str] | None,
             perm: np.ndarray) -> np.ndarray:
    """Characteristics of one month's universe (rows sorted by id) with ``theme_features`` permuted."""
    x = rows[list(features)].to_numpy(dtype=float).copy()
    if theme_features:
        cols = [list(features).index(f) for f in theme_features]
        x[:, cols] = x[perm][:, cols]
    return x


def _universe(panel: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    u = panel.loc[panel["in_universe"] & panel["eom"].isin(dates)]
    return u.sort_values(["eom", "id"]).reset_index(drop=True)


# --- Portfolio-ML ------------------------------------------------------------------------------------

def load_signal_spec(directory: Path) -> tuple[dict, list[str], np.ndarray | None]:
    """meta, column names and W of one Portfolio-ML signal set (scripts/06_pfml_inputs.py)."""
    meta = json.loads((directory / "meta.json").read_text())
    w_path = directory / "rff_w.npy"
    return meta, meta["columns"], (np.load(w_path) if w_path.exists() else None)


def pfml_counterfactual_aims(universe: pd.DataFrame, perms: Mapping[pd.Timestamp, np.ndarray],
                             theme_features: Sequence[str] | None, aim_coefs: pd.DataFrame,
                             specs: Mapping[str, tuple[dict, list[str], np.ndarray | None]],
                             cov_parts: Mapping[pd.Timestamp, CovParts]) -> pd.DataFrame:
    """R ``pfml_cf_fun``: aim_t = S_t(shuffled) β_t with the base β, (g, p) and W of each month."""
    coef_by = {eom: g for eom, g in aim_coefs.groupby("eom")}
    out = []
    for eom, rows in universe.groupby("eom", sort=True):
        c = coef_by[eom]
        meta, columns, w = specs[c["tag"].iloc[0]]
        x = shuffled(rows, meta["features"], theme_features, perms[eom])
        raw = rff_raw(x, w) if meta["use_rff"] else x
        if meta["use_rff"] and meta["add_orig"]:
            raw = np.hstack([raw, x])
        ids = rows["id"].to_numpy()
        vol = cov_parts[eom].vol(ids) if meta["scale_by_vol"] else None
        s = scale_signals(raw, vol, meta.get("add_constant", True))
        idx = [columns.index(name) for name in c["column"]]
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            w_aim = s[:, idx] @ c["beta"].to_numpy()
        if not np.isfinite(w_aim).all():
            raise FloatingPointError(f"non-finite counterfactual aim at {eom.date()}")
        out.append(pd.DataFrame({"id": ids, "eom": eom, "w_aim": w_aim}))
    return pd.concat(out, ignore_index=True)


def pfml_counterfactual(theme_features: Sequence[str] | None, universe: pd.DataFrame,
                        perms: Mapping[pd.Timestamp, np.ndarray], aim_coefs: pd.DataFrame, specs,
                        inputs: Mapping[pd.Timestamp, MonthInputs], panel: pd.DataFrame, wealth: pd.DataFrame,
                        cov_parts: Mapping[pd.Timestamp, CovParts], rf: pd.Series, *, gamma: float, mu: float,
                        iterations: int, charge_exit_tc: bool) -> pd.DataFrame:
    aims = pfml_counterfactual_aims(universe, perms, theme_features, aim_coefs, specs, cov_parts)
    w = pfml_weights(inputs, aims, panel, wealth, rf, gamma=gamma, mu=mu, iterations=iterations,
                     charge_exit_tc=charge_exit_tc)
    return portfolio_returns(w, panel, wealth)


# --- Markowitz-ML ------------------------------------------------------------------------------------

def markowitz_counterfactual(theme_features: Sequence[str] | None, universe: pd.DataFrame,
                             perms: Mapping[pd.Timestamp, np.ndarray], features: Sequence[str],
                             models: Mapping[int, YearModel], z: np.ndarray, panel: pd.DataFrame,
                             wealth: pd.DataFrame, cov_parts: Mapping[pd.Timestamp, CovParts], *, gamma: float,
                             charge_exit_tc: bool) -> pd.DataFrame:
    """R ``tpf_cf_fun``: pred_ld1 from the shuffled characteristics with the year's fitted model
    (``predict(newx = cf_new_x, s = opt lambda)``), then ``tpf_implement``. Pass a zero-wealth path
    and γ = 100 to follow R (gross, less extreme weights)."""
    parts = []
    for eom, rows in universe.groupby("eom", sort=True):
        x = shuffled(rows, features, theme_features, perms[eom])
        parts.append(rows[["id", "eom"]].assign(pred_ld1=models[return_year(eom)].predict(x, z)))
    targets = markowitz_targets(pd.concat(parts, ignore_index=True), cov_parts, gamma)
    dates = sorted(universe["eom"].unique())
    w = add_start_weights(targets, panel, wealth, dates, charge_exit_tc)
    return portfolio_returns(w, panel, wealth)


# --- Importance and Figure 5 --------------------------------------------------------------------------

def importance(pfs: pd.DataFrame, gamma: float, dates: DatesConfig) -> pd.DataFrame:
    """``pfs``: monthly rows with ``type, theme, r, tc`` (and ``seed`` if several permutation draws). Utility
    per (seed, type, theme) and the drop ``fi = U(none) − U(theme)`` (R: ``cf_obj[cluster == "bm"] - cf_obj``)."""
    check_return_months(pfs, dates, "feature importance")
    keys = ["seed", "type"] if "seed" in pfs else ["type"]
    u = pfs.groupby([*keys, "theme"], sort=False).apply(
        lambda g: (g["r"].mean() - 0.5 * gamma * g["r"].var(ddof=1) - g["tc"].mean()) * 12, include_groups=False)
    u = u.rename("utility").reset_index()
    base = u[u["theme"] == BASE].set_index(keys)["utility"]
    return u.assign(fi=base.reindex(pd.MultiIndex.from_frame(u[keys]) if len(keys) > 1 else u["type"]).to_numpy()
                    - u["utility"])


def summarise_seeds(imp: pd.DataFrame) -> pd.DataFrame:
    """Mean utility and importance over permutation seeds per (type, theme), with the standard error of the
    mean importance, sd(fi) / sqrt(n_seeds) (0 for the unshuffled run)."""
    g = imp.groupby(["type", "theme"], sort=False)
    out = g.agg(utility=("utility", "mean"), fi=("fi", "mean"), n_seeds=("seed", "nunique")).reset_index()
    se = g["fi"].std(ddof=1).to_numpy() / np.sqrt(out["n_seeds"].to_numpy())
    return out.assign(fi_se=np.nan_to_num(se))


def plot_importance(fi: pd.DataFrame, types: Sequence[str], path: Path) -> list[Path]:
    """Figure 5: horizontal bars of the utility drop by theme, one panel per method (sorted by the first)."""
    d = fi[fi["theme"] != BASE]
    order = d[d["type"] == types[0]].sort_values("fi")["theme"].tolist()
    fig, axes = plt.subplots(1, len(types), figsize=(3.4 * len(types), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t, colour in zip(axes, types, ["C0", "C4", "C2"]):
        s = d[d["type"] == t].set_index("theme").reindex(order)
        ax.barh([o.replace("_", " ").replace("short term", "short-term").title() for o in order], s["fi"],
                xerr=1.96 * s["fi_se"] if "fi_se" in s else None, color=colour, error_kw={"lw": 0.8})
        ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(f"{t} ({'gross' if t == 'Markowitz-ML' else 'net of TC'})", fontsize=9)
        ax.tick_params(labelsize=8)
    se = " (bars: ±1.96 s.e. across permutation seeds)" if "fi_se" in d else ""
    fig.supxlabel(f"Drop in realised utility from permuting the theme's features (annualised){se}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = [path.with_suffix(".png"), path.with_suffix(".pdf")]
    fig.savefig(out[0], dpi=200, bbox_inches="tight")
    fig.savefig(out[1], bbox_inches="tight")
    plt.close(fig)
    return out


def theme_list(themes: Mapping[str, Sequence[str]]) -> list[tuple[str, list[str] | None]]:
    """The unshuffled run first, then one entry per theme."""
    return [(BASE, None), *[(t, list(f)) for t, f in themes.items()]]
