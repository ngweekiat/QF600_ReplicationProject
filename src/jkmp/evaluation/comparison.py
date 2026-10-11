"""The project's selected comparison: Static-ML* and nonlinear Portfolio-ML."""

from pathlib import Path

import pandas as pd

METHODS = ("Static-ML*", "Portfolio-ML")


def selected_methods(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude other methods, including results cached by previous runs."""
    return frame.loc[frame["type"].isin(METHODS)].copy()


def load_comparison(outputs: Path) -> pd.DataFrame:
    paths = [outputs / "static_ml" / "pf.parquet",
             outputs / "portfolio_ml" / "rff" / "pf.parquet"]
    frames = [selected_methods(pd.read_parquet(path)) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError("Run scripts/05_static_ml.py and scripts/07_portfolio_ml.py first.")
    return pd.concat(frames, ignore_index=True)
