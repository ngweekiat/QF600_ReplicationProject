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
    """Load the frozen research groups and audited covariance direction policy.

    Published features retain published signs; provisional group members use
    their 2005–2014 training sign. These signs apply only to covariance theme
    exposures; return-model and Portfolio ML raw ranks remain unflipped.
    """
    # Kati's frozen metadata explicitly distinguishes provisional group members.
    # Keep the published reference files intact: do not mislabel proxies as JKP.
    frozen = Path(__file__).resolve().parents[2] / "data/characteristics/theme_directions.csv"
    if frozen.exists():
        spec = pd.read_csv(frozen)
        expected = {f: t for t, feats in themes.items() for f in feats}
        if spec.characteristic.duplicated().any() or set(spec.characteristic) != set(expected):
            raise ValueError("Frozen Kati theme metadata differs from configured 50 predictors")
        if any(expected[row.characteristic] != row.model_theme for row in spec.itertuples()):
            raise ValueError("Configured theme differs from the frozen research assignment")
        if not spec.model_theme_direction.isin([-1, 1]).all():
            raise ValueError("Frozen theme directions must be +1/-1")
        indexed = spec.set_index("characteristic")
        return pd.DataFrame([dict(feature=f, theme=t, direction=float(indexed.loc[f, "model_theme_direction"]),
                                 direction_source=indexed.loc[f, "model_direction_source"],
                                 published_group=bool(indexed.loc[f, "published_jkp_153"]))
                             for f, t in expected.items()])
    raise FileNotFoundError("Kati requires data/characteristics/theme_directions.csv; run scripts/prepare_inputs.py first")


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
