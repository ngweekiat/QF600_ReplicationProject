"""Characteristic standardisation and theme scores.

* ``rank_features`` — cross-sectional rank in [0, 1] per month, exact zeros -> 0, missing -> 0.5
  (reference/jkmp_r/1 - Prepare Data.R, feature standardisation and imputation).
* ``theme_scores`` — per theme, the average rank of its characteristics after flipping those with
  direction -1 to ``1 - rank`` (reference/jkmp_r/3 - Estimate Covariance Matrix.R, ``cluster_ranks``).
  These are the theme loadings of the Barra-style covariance model; R standardises them by month
  within the estimation universe just before the factor regressions (``standardize_by_month``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from jkmp.data.wrds_pull import RAW_FILES

THEME_PREFIX = "theme_"


def rank_features(panel: pd.DataFrame, features: Sequence[str], percentile_rank: bool = True,
                  impute_value: float = 0.5) -> pd.DataFrame:
    """Rank each characteristic within each month to [0, 1]; return the standardised columns.

    ``rank(method="max", pct=True)`` equals R's ``ecdf(x)(x)`` (ties take the highest rank).
    R then sets characteristics whose raw value is exactly 0 to rank 0 ("ecdf always returns >0")
    and imputes missing values with 0.5. Without ranking, R imputes the monthly median instead.
    """
    x = panel[list(features)]
    if percentile_rank:
        ranks = x.groupby(panel["eom"]).rank(method="max", pct=True)
        return ranks.mask(x.eq(0), 0.0).fillna(impute_value)
    return x.fillna(x.groupby(panel["eom"]).transform("median"))


def normalise_theme_label(label: str) -> str:
    """JKP cluster label -> config theme name (R: lower-case, spaces and hyphens -> underscores)."""
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def load_theme_directions(raw_dir: Path, themes: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Feature, theme and sign direction for the configured features, checked against the JKP files.

    Themes come from the config but must agree with ``Cluster Labels.csv``; directions (+1 / -1)
    come from ``Factor Details.xlsx`` (column ``direction``, keyed by ``abr_jkp``). As in R,
    ``rvol_252d`` is assigned to low risk with direction -1.
    """
    labels = pd.read_parquet(raw_dir / RAW_FILES["cluster_labels"])
    jkp_theme = dict(zip(labels["characteristic"], labels["cluster"].map(normalise_theme_label)))
    jkp_theme["rvol_252d"] = "low_risk"
    details = pd.read_parquet(raw_dir / RAW_FILES["factor_details"]).dropna(subset=["abr_jkp"])
    direction = dict(zip(details["abr_jkp"], details["direction"]))
    direction["rvol_252d"] = -1.0

    rows = [{"feature": f, "theme": t, "direction": direction.get(f)} for t, feats in themes.items() for f in feats]
    out = pd.DataFrame(rows)
    wrong = out[[jkp_theme.get(f) != t for f, t in zip(out["feature"], out["theme"])]]
    if len(wrong):
        raise ValueError(f"config themes disagree with Cluster Labels.csv for: {wrong['feature'].tolist()}")
    missing = out[~out["direction"].isin([1.0, -1.0])]
    if len(missing):
        raise ValueError(f"no +1/-1 direction in Factor Details.xlsx for: {missing['feature'].tolist()}")
    return out


def theme_scores(ranks: pd.DataFrame, directions: pd.DataFrame) -> pd.DataFrame:
    """Average signed rank per theme: columns ``theme_<name>`` (R: ``rowMeans`` of flipped ranks)."""
    out = {}
    for theme, group in directions.groupby("theme", sort=False):
        signed = [ranks[f] if d == 1 else 1.0 - ranks[f] for f, d in zip(group["feature"], group["direction"])]
        out[THEME_PREFIX + theme] = pd.concat(signed, axis=1).mean(axis=1)
    return pd.DataFrame(out, index=ranks.index)


def standardize_by_month(x: pd.DataFrame, eom: pd.Series, within: pd.Series | None = None) -> pd.DataFrame:
    """Subtract the monthly cross-sectional mean and divide by the standard deviation (R: ``sd``, n-1).

    With ``within`` (a boolean mask), the mean and sd come from those rows only but are applied to
    every row — e.g. moments from the covariance estimation universe, applied to all stocks.
    """
    base = x if within is None else x[within]
    grouped = base.groupby(eom if within is None else eom[within])
    mean, sd = grouped.mean(), grouped.std()
    return (x - mean.reindex(eom).to_numpy()) / sd.reindex(eom).to_numpy()
