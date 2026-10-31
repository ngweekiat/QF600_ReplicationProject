"""One-month return predictions ``pred_ld1``: random-feature ridge refit every year (expanding window).

Port of ``rff_hp_search`` (0 - Return prediction functions.R) with ``data_split`` (yearly updates),
with the validation window shortened to ``val_years`` (R: 10). For each return year y:

* train : formation months with returns realised up to Dec y−1−val_years  (eom ≤ Nov y−1−val_years)
* val   : the next ``val_years`` years of realised returns                (Dec y−1−val_years .. Nov y−1)
* pick (g, p, λ) by validation MSE; refit on train + val                  (eom ≤ Nov y−1)
* test  : predict for formation months Dec y−1 .. Nov y (returns realised in year y)

Training rows are all screened stocks with a 12-month history (``valid_data``), not just the portfolio
universe — R also fits its return models on all US stocks. The target is ``ret_ld1``; inputs are the
ranked characteristics in [0, 1].
"""

from __future__ import annotations

import logging
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from jkmp.config import Config, DatesConfig
from jkmp.evaluation.holdout import check_return_months, last_open_month
from jkmp.models.random_features import base_draws, column_subset, rff, rff_raw, weights_for
from jkmp.models.ridge import Moments, mse_from_moments, ridge_path

log = logging.getLogger(__name__)


@dataclass
class YearModel:
    """The fitted model that predicts return year ``year``: pred = RF_p(x; √g·Z) @ coef + b0."""
    year: int
    g: float
    p: int
    lam: float
    coef: np.ndarray   # (p,)
    b0: float

    def predict(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):  # spurious Accelerate matmul flags; checked below
            pred = rff(x, weights_for(z, self.g), self.p) @ self.coef + self.b0
        if not np.isfinite(pred).all():
            raise FloatingPointError("non-finite predictions")
        return pred


@dataclass
class ReturnModelResult:
    predictions: pd.DataFrame   # id, eom, year, pred_ld1
    hp_search: pd.DataFrame     # year, g, p, lam, val_mse
    chosen: pd.DataFrame        # one row per year: g, p, lam, val_mse, n_train, n_val, n_full
    models: dict[int, YearModel] = field(default_factory=dict)
    z: np.ndarray | None = None  # standard-normal draws shared by every g (n_features × p_max/2)


def cutoffs_for_year(year: int, val_years: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(last training formation month, last train+val formation month) for return year ``year``."""
    return pd.Timestamp(year - 1 - val_years, 11, 30), pd.Timestamp(year - 1, 11, 30)


def formation_months_for_year(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Formation months whose next-month return falls in ``year``: Dec y−1 .. Nov y."""
    return pd.Timestamp(year - 1, 12, 31), pd.Timestamp(year, 11, 30)


def prefix_moments(rows: pd.DataFrame, features: list[str], w: np.ndarray,
                   snapshots: list[pd.Timestamp]) -> dict[pd.Timestamp, Moments]:
    """Running sufficient statistics of [cos(sW), sin(sW)] and ret_ld1, recorded at each snapshot month."""
    wanted = set(snapshots)
    running = Moments.zeros(2 * w.shape[1])
    out = {}
    for eom, g in rows.groupby("eom", sort=True):
        if eom > max(snapshots):
            break
        running = running + Moments.of(rff_raw(g[features].to_numpy(), w), g["ret_ld1"].to_numpy())
        if eom in wanted:
            out[eom] = running
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"no data in snapshot months {sorted(missing)[:3]}")
    return out


def prediction_years(cfg: Config) -> list[int]:
    """Return years to fit and predict: ``return_model.pred_years``, cut at the last open year while the holdout is
    locked, so no validation MSE is ever computed on holdout return months."""
    first, last = cfg.return_model.pred_years
    return list(range(first, min(last, last_open_month(cfg.dates).year) + 1))


def fit_return_model(panel: pd.DataFrame, cfg: Config) -> ReturnModelResult:
    rm = cfg.return_model
    feats = list(cfg.features.names)
    lambdas = np.asarray(cfg.ridge_grid.values)
    p_grid, g_grid = list(rm.p_grid), list(rm.rff_variance_grid)
    p_max = max(p_grid)
    years = prediction_years(cfg)

    rows = panel.loc[panel["valid_data"] & ((panel["eom"] + pd.offsets.MonthEnd(1)) >= pd.Timestamp("2005-01-31")), ["id", "eom", *feats, "ret_ld1"]]
    # A held stock can be forecast from current characteristics even when its
    # current realised return makes that row ineligible for fitting.
    prediction_rows = panel.loc[panel["valid_data"] | panel["in_universe"], ["id", "eom", *feats, "ret_ld1"]] if "in_universe" in panel else rows
    cut = {y: cutoffs_for_year(y, rm.val_years) for y in years}
    snapshots = sorted({c for pair in cut.values() for c in pair})
    z = base_draws(len(feats), p_max, cfg.seed)  # same draws for every g (R: set.seed before each g)
    moments = {g: prefix_moments(rows, feats, weights_for(z, g), snapshots) for g in g_grid}
    log.info("Moments: %d rows, %d snapshot months, p_max %d, %d variances", len(rows), len(snapshots), p_max, len(g_grid))

    searches, chosen, preds, models = [], [], [], {}
    for y in years:
        tr_cut, full_cut = cut[y]
        records = []
        for g in g_grid:
            m_train, m_full = moments[g][tr_cut], moments[g][full_cut]
            m_val = m_full - m_train
            for p in p_grid:
                cols, scale = column_subset(p, p_max), p ** -0.5
                coef, b0 = ridge_path(m_train.select(cols, scale), lambdas)
                mse = mse_from_moments(m_val.select(cols, scale), coef, b0)
                records.append(pd.DataFrame({"year": y, "g": g, "p": p, "lam": lambdas, "val_mse": mse}))
        search = pd.concat(records, ignore_index=True)
        best = search.loc[search["val_mse"].idxmin()]
        g, p, lam = float(best["g"]), int(best["p"]), float(best["lam"])

        cols, scale = column_subset(p, p_max), p ** -0.5
        m_full = moments[g][full_cut]
        coef, b0 = ridge_path(m_full.select(cols, scale), np.array([lam]))
        lo, hi = formation_months_for_year(y)
        test = prediction_rows[prediction_rows["eom"].between(lo, hi)]
        models[y] = YearModel(y, g, p, lam, coef[:, 0].copy(), float(b0[0]))
        pred = models[y].predict(test[feats].to_numpy(), z)
        preds.append(test[["id", "eom"]].assign(year=y, pred_ld1=pred))
        n_train = moments[g][tr_cut].n
        chosen.append({"year": y, "g": g, "p": p, "lam": lam, "log_lam": np.log(lam) if lam > 0 else -np.inf,
                       "val_mse": float(best["val_mse"]), "n_train": n_train, "n_val": m_full.n - n_train,
                       "n_full": m_full.n, "n_test": len(test)})
        searches.append(search)
        log.info("Year %d: g=%.3f p=%d log(λ)=%s val MSE %.6f; %d train / %d val / %d test rows",
                 y, g, p, f"{np.log(lam):.1f}" if lam > 0 else "-inf (λ=0)", best["val_mse"],
                 n_train, m_full.n - n_train, len(test))
    return ReturnModelResult(pd.concat(preds, ignore_index=True), pd.concat(searches, ignore_index=True),
                             pd.DataFrame(chosen), models, z)


def save_models(result: ReturnModelResult, path: Path) -> None:
    """Fitted yearly models + the shared draws Z, so predictions can be recomputed on new inputs."""
    years = sorted(result.models)
    m = [result.models[y] for y in years]
    np.savez(path, z=result.z, years=np.array(years),
             meta=json.dumps([{"year": x.year, "g": x.g, "p": x.p, "lam": x.lam, "b0": x.b0} for x in m]),
             **{f"coef_{x.year}": x.coef for x in m})


def load_models(path: Path) -> tuple[dict[int, YearModel], np.ndarray]:
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        models = {d["year"]: YearModel(d["year"], d["g"], d["p"], d["lam"], f[f"coef_{d['year']}"], d["b0"])
                  for d in meta}
        return models, f["z"]


def rank_ic_by_size(predictions: pd.DataFrame, panel: pd.DataFrame, buckets: Sequence[Sequence[int]],
                    dates: DatesConfig) -> pd.DataFrame:
    """Mean monthly Spearman rank IC of pred_ld1 vs ret_ld1 within market-cap rank buckets (rank among the
    screened ``valid_data`` stocks each month, 1 = largest), per decade of the return month and overall."""
    check_return_months(predictions, dates, "rank_ic_by_size")
    d = predictions[["id", "eom", "pred_ld1"]].merge(panel.loc[panel["valid_data"], ["id", "eom", "me", "ret_ld1"]],
                                                     on=["id", "eom"])
    d["size_rank"] = d.groupby("eom")["me"].rank(ascending=False, method="first")
    out = []
    for lo, hi in buckets:
        b = d[d["size_rank"].between(lo, hi)]
        ic = b.groupby("eom")[["pred_ld1", "ret_ld1"]].apply(lambda x: x["pred_ld1"].rank().corr(x["ret_ld1"].rank()))
        decade = (ic.index + pd.offsets.MonthEnd(1)).year // 10 * 10
        label = f"{lo}-{hi}"
        out.append(ic.groupby(decade).agg(["mean", "size"]).assign(bucket=label).rename_axis("decade").reset_index())
        out.append(pd.DataFrame({"decade": ["all"], "mean": [ic.mean()], "size": [len(ic)], "bucket": [label]}))
    return pd.concat(out, ignore_index=True).rename(columns={"mean": "rank_ic", "size": "months"})


def monthly_rank_ic(predictions: pd.DataFrame, panel: pd.DataFrame, dates: DatesConfig) -> pd.Series:
    """Spearman rank IC of pred_ld1 vs ret_ld1 across the portfolio universe, one value per return month."""
    check_return_months(predictions, dates, "monthly_rank_ic")
    d = predictions[["id", "eom", "pred_ld1"]].merge(panel.loc[panel["in_universe"], ["id", "eom", "ret_ld1"]],
                                                     on=["id", "eom"])
    ic = d.groupby("eom")[["pred_ld1", "ret_ld1"]].apply(lambda x: x["pred_ld1"].rank().corr(x["ret_ld1"].rank()))
    return pd.Series(ic.to_numpy(), index=ic.index + pd.offsets.MonthEnd(1), name="rank_ic").rename_axis("eom_ret")


def prediction_diagnostics(predictions: pd.DataFrame, panel: pd.DataFrame, dates: DatesConfig) -> pd.DataFrame:
    """Per return year: out-of-sample R² against a zero forecast (Gu, Kelly and Xiu, 2020) and the mean
    monthly Spearman rank IC, for all predicted stocks and for the portfolio universe."""
    check_return_months(predictions, dates, "prediction_diagnostics")
    d = predictions.merge(panel[["id", "eom", "ret_ld1", "in_universe"]], on=["id", "eom"], how="left")
    out = []
    for (year, scope), g in pd.concat([d.assign(scope="all"), d[d["in_universe"]].assign(scope="universe")]).groupby(["year", "scope"]):
        ic = g.groupby("eom")[["pred_ld1", "ret_ld1"]].apply(lambda x: x["pred_ld1"].rank().corr(x["ret_ld1"].rank()))
        out.append({"year": year, "scope": scope, "n": len(g),
                    "r2_oos": 1 - ((g["ret_ld1"] - g["pred_ld1"]) ** 2).sum() / (g["ret_ld1"] ** 2).sum(),
                    "rank_ic": ic.mean(), "pred_sd": g["pred_ld1"].std()})
    return pd.DataFrame(out)
