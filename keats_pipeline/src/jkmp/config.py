"""Load ``configs/base.yaml`` (plus run-time overrides) into a frozen, typed ``Config``.

Usage::

    from jkmp.config import load_config

    cfg = load_config()                                         # configs/base.yaml
    cfg = load_config(overrides=["investor.gamma=5"])           # dotted-key strings
    cfg = load_config(overrides={"investor.wealth": 1.0e9})     # or a dotted-key mapping

The YAML file is the single source of truth: every dataclass field must appear in it,
unknown keys are rejected, and overrides may only replace keys that already exist.
An override replaces the whole value at its key (no deep merge).
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "us_large" / "base.yaml"


class ConfigError(ValueError):
    """Raised for missing, unknown, mistyped or inconsistent configuration values."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _require_even(values: tuple[int, ...], name: str) -> None:
    # Random Fourier features come in sin/cos pairs, so p must be even.
    _require(all(p > 0 and p % 2 == 0 for p in values), f"{name} must contain positive even integers")


# --------------------------------------------------------------------------- sections


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    processed: Path
    outputs: Path
    reference_r: Path


@dataclass(frozen=True)
class DataSourcesConfig:
    jkp_table: str
    crsp_daily_table: str
    factor_details_url: str
    cluster_labels_url: str
    ff_factors_url: str
    ff_factors_daily_url: str
    fred_csv_url: str


@dataclass(frozen=True)
class DatesConfig:
    warmup_start: dt.date
    validation_start: dt.date
    validation_end: dt.date
    test_start: dt.date
    test_end: dt.date
    holdout_start: dt.date | None
    holdout_end: dt.date | None
    data_end: dt.date
    holdout_locked: bool
    evaluate: str

    def __post_init__(self) -> None:
        order = ["warmup_start", "validation_start", "validation_end", "test_start", "test_end"]
        for earlier, later in zip(order, order[1:]):
            _require(
                getattr(self, earlier) < getattr(self, later),
                f"dates.{earlier} must be before dates.{later}",
            )
        _require(self.test_end <= self.data_end, "test_end must not exceed data_end")
        if self.holdout_start is None or self.holdout_end is None:
            _require(self.holdout_start is None and self.holdout_end is None,
                     "both holdout dates must be null when no separate holdout exists")
            _require(self.holdout_locked and self.evaluate == "test",
                     "No separate Keats holdout is available; evaluate the declared test period only")
        else:
            _require(self.test_end < self.holdout_start <= self.holdout_end <= self.data_end,
                     "holdout must follow test_end and not exceed data_end")
        _require(self.evaluate in ("test", "holdout"), "dates.evaluate must be 'test' or 'holdout'")
        _require(self.evaluate == "test" or not self.holdout_locked, "dates.evaluate: holdout needs holdout_locked: false")


@dataclass(frozen=True)
class UniverseConfig:
    exchanges: tuple[int, ...]
    share_codes: tuple[int, ...]
    jkp_flags: tuple[str, ...]
    target_n: int
    entry_rank: int
    exit_rank: int
    reconstitution_month: int
    min_feature_pct: float
    lookback_months: int

    def __post_init__(self) -> None:
        _require(
            0 < self.entry_rank <= self.target_n <= self.exit_rank,
            "universe ranks must satisfy 0 < entry_rank <= target_n <= exit_rank",
        )
        _require(1 <= self.reconstitution_month <= 12, "universe.reconstitution_month must be 1..12")
        _require(0.0 < self.min_feature_pct <= 1.0, "universe.min_feature_pct must be in (0, 1]")
        _require(self.lookback_months >= 0, "universe.lookback_months must be >= 0")


@dataclass(frozen=True)
class FeaturesConfig:
    percentile_rank: bool
    impute_value: float
    themes: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        _require(len(self.themes) > 0, "features.themes must not be empty")
        _require(all(len(f) > 0 for f in self.themes.values()), "every theme needs at least one feature")
        _require(len(self.names) == len(set(self.names)), "a feature appears in more than one theme")

    @property
    def names(self) -> tuple[str, ...]:
        """All features, in theme order."""
        return tuple(f for feats in self.themes.values() for f in feats)

    @property
    def theme_of(self) -> Mapping[str, str]:
        """Feature name -> theme name."""
        return MappingProxyType({f: theme for theme, feats in self.themes.items() for f in feats})


@dataclass(frozen=True)
class CovarianceConfig:
    factor_universe_top_n: int
    sectors: Mapping[str, tuple[str, ...]]
    window_days: int
    min_window_days: int
    half_life_corr_days: int
    half_life_var_days: int
    half_life_idio_days: int
    idio_min_obs: int
    idio_min_window: int
    idio_initial_obs: int
    days_per_month: int

    def __post_init__(self) -> None:
        _require(0 < self.min_window_days <= self.window_days, "covariance.min_window_days must be in (0, window_days]")
        _require(0 < self.idio_min_obs <= self.idio_min_window, "covariance.idio_min_obs must be in (0, idio_min_window]")
        industries = [i for group in self.sectors.values() for i in group]
        _require(len(industries) == len(set(industries)), "covariance.sectors: an industry is in two sectors")


@dataclass(frozen=True)
class TradingCostConfig:
    price_impact: float
    dolvol_column: str
    dolvol_units: str

    def __post_init__(self) -> None:
        _require(self.price_impact > 0, "trading_costs.price_impact must be > 0")
        _require(self.dolvol_units in {"dollars", "millions"}, "trading_costs.dolvol_units must be 'dollars' or 'millions'")


@dataclass(frozen=True)
class InvestorConfig:
    gamma: float
    wealth: float
    wealth_end_date: dt.date
    mu: float
    m_iterations: int
    gamma_te: float
    te_targets: tuple[float, ...]
    gamma_te_grid: tuple[float, ...]
    gamma_grid: tuple[float, ...]
    wealth_grid: tuple[float, ...]

    def __post_init__(self) -> None:
        _require(self.gamma > 0 and all(g > 0 for g in self.gamma_grid), "risk aversion must be > 0")
        _require(self.wealth > 0 and all(w > 0 for w in self.wealth_grid), "wealth must be > 0")
        _require(self.m_iterations > 0, "investor.m_iterations must be > 0")
        _require(self.gamma_te > 0 and all(g > 0 for g in self.gamma_te_grid), "gamma_te must be > 0")
        _require(all(0 < t < 1 for t in self.te_targets), "investor.te_targets are annual fractions in (0, 1)")


@dataclass(frozen=True)
class PortfolioConfig:
    charge_exit_tc: bool


@dataclass(frozen=True)
class RidgeGridConfig:
    include_zero: bool
    log_min: float
    log_max: float
    n: int

    def __post_init__(self) -> None:
        _require(self.n >= 2 and self.log_min < self.log_max, "ridge_grid needs n >= 2 and log_min < log_max")

    @property
    def values(self) -> tuple[float, ...]:
        """Penalties: optional 0, then exp of n evenly spaced points (R: c(0, exp(seq(...))))."""
        step = (self.log_max - self.log_min) / (self.n - 1)
        grid = tuple(math.exp(self.log_min + i * step) for i in range(self.n))
        return ((0.0,) + grid) if self.include_zero else grid


@dataclass(frozen=True)
class ReturnModelConfig:
    p_grid: tuple[int, ...]
    rff_variance_grid: tuple[float, ...]
    horizons: tuple[int, ...]
    val_years: int
    pred_years: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_even(self.p_grid, "return_model.p_grid")
        _require(all(h >= 1 for h in self.horizons), "return_model.horizons must be >= 1")
        _require(self.val_years >= 1, "return_model.val_years must be >= 1")
        _require(len(self.pred_years) == 2 and self.pred_years[0] <= self.pred_years[1],
                 "return_model.pred_years must be [first, last]")


@dataclass(frozen=True)
class PortfolioMLConfig:
    use_rff: bool
    p_grid: tuple[int, ...]
    rff_variance_grid: tuple[float, ...]
    scale_by_vol: bool
    add_original_features: bool
    add_constant: bool
    benchmark_relative: bool
    half_life_years: tuple[float | None, ...]

    def __post_init__(self) -> None:
        _require_even(self.p_grid, "portfolio_ml.p_grid")
        _require(len(self.half_life_years) >= 1 and all(h is None or h > 0 for h in self.half_life_years),
                 "portfolio_ml.half_life_years: positive years or null (= no decay)")


@dataclass(frozen=True)
class StaticMLConfig:
    k_grid: tuple[float, ...]
    u_grid: tuple[float, ...]
    g_grid: tuple[float, ...]


@dataclass(frozen=True)
class FactorMLConfig:
    n_portfolios: int


@dataclass(frozen=True)
class FeatureImportanceConfig:
    markowitz_gamma: float
    n_seeds: int

    def __post_init__(self) -> None:
        _require(self.markowitz_gamma > 0, "feature_importance.markowitz_gamma must be > 0")
        _require(self.n_seeds >= 1, "feature_importance.n_seeds must be >= 1")

    def seeds(self, seed: int) -> tuple[int, ...]:
        """Permutation seeds: the run's seed and the next n_seeds - 1."""
        return tuple(range(seed, seed + self.n_seeds))


@dataclass(frozen=True)
class EpisodeConfig:
    label: str
    start: dt.date
    end: dt.date

    def __post_init__(self) -> None:
        _require(self.start <= self.end, f"regimes episode {self.label!r}: start after end")


@dataclass(frozen=True)
class RegimesConfig:
    fred_series: tuple[str, ...]
    fed_change_months: int
    fed_threshold: float
    vix_norm: str
    plot_types: tuple[str, ...]
    episodes: tuple[EpisodeConfig, ...]

    def __post_init__(self) -> None:
        _require(self.fed_change_months >= 1, "regimes.fed_change_months must be >= 1")
        _require(self.fed_threshold >= 0, "regimes.fed_threshold must be >= 0")
        _require(self.vix_norm in ("window", "full"), "regimes.vix_norm must be 'window' or 'full'")


@dataclass(frozen=True)
class ReportingConfig:
    main_types: tuple[str, ...]
    subperiods: tuple[EpisodeConfig, ...]   # label, start, end (return months)
    ic_size_buckets: tuple[tuple[int, ...], ...]
    capacity_ir: float
    cost_vix_start: dt.date
    cost_vol_window: int

    def __post_init__(self) -> None:
        _require(len(self.main_types) >= 2, "reporting.main_types needs at least two methods")
        _require(all(len(b) == 2 and 1 <= b[0] <= b[1] for b in self.ic_size_buckets),
                 "reporting.ic_size_buckets must be [first, last] rank pairs")
        _require(self.cost_vol_window >= 1, "reporting.cost_vol_window must be >= 1")


@dataclass(frozen=True)
class MonitorConfig:
    ir_window: int
    drawdown_te_multiple: float
    te_budget: float
    beta_limits: tuple[float, ...]
    theme_limit: float
    ic_window: int
    ic_negative_months: int
    tc_months: int
    bias_window: int

    def __post_init__(self) -> None:
        _require(len(self.beta_limits) == 2 and self.beta_limits[0] < self.beta_limits[1], "monitor.beta_limits: [low, high]")
        _require(min(self.ir_window, self.ic_window, self.ic_negative_months, self.tc_months, self.bias_window) >= 1,
                 "monitor windows must be >= 1")


@dataclass(frozen=True)
class GridsConfig:
    n_jobs: int
    frontier_vol_max: float
    frontier_vol_step: float

    def __post_init__(self) -> None:
        _require(self.n_jobs != 0, "grids.n_jobs must be non-zero (joblib: -1 = all cores)")
        _require(0 < self.frontier_vol_step < self.frontier_vol_max, "grids needs 0 < frontier_vol_step < frontier_vol_max")


@dataclass(frozen=True)
class Config:
    seed: int
    paths: PathsConfig
    data_sources: DataSourcesConfig
    dates: DatesConfig
    universe: UniverseConfig
    features: FeaturesConfig
    covariance: CovarianceConfig
    trading_costs: TradingCostConfig
    investor: InvestorConfig
    portfolio: PortfolioConfig
    ridge_grid: RidgeGridConfig
    return_model: ReturnModelConfig
    portfolio_ml: PortfolioMLConfig
    static_ml: StaticMLConfig
    factor_ml: FactorMLConfig
    monitor: MonitorConfig
    grids: GridsConfig
    feature_importance: FeatureImportanceConfig
    reporting: ReportingConfig
    regimes: RegimesConfig

    def as_dict(self) -> dict[str, Any]:
        """Plain-Python copy (dicts, lists, str paths) for saving alongside run outputs."""
        return _to_plain(self)


# --------------------------------------------------------------------------- loading


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> Config:
    """Read a YAML config, apply dotted-key overrides, and build a validated ``Config``.

    A file may start with ``base: <path>`` (relative to that file): the base is read first and this
    file's keys are merged on top, so an experiment lists only what it changes.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _read_yaml(path, seen=())
    raw = _apply_overrides(raw, overrides)
    cfg = _build(Config, raw, "")
    if cfg.paths.outputs.parent.name == "base":
        _require(cfg.reporting.main_types == ("Static-ML","Portfolio-ML"), "Base reports contain exactly Static-ML and Portfolio-ML")
        _require(cfg.portfolio_ml.use_rff and not cfg.portfolio_ml.benchmark_relative
                 and cfg.portfolio_ml.add_constant and cfg.portfolio_ml.half_life_years == (None,),
                 "Base requires the original default Portfolio-ML; choose an iteration group for experimental options")
    from jkmp.imputation_audit import report_if_enabled
    report_if_enabled(cfg)
    return cfg


def _read_yaml(path: Path, seen: tuple[Path, ...]) -> dict:
    path = path.resolve()
    _require(path not in seen, f"circular 'base' chain at {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    _require(isinstance(raw, dict), f"{path} must contain a YAML mapping")
    base = raw.pop("base", None)
    if base is None:
        return raw
    return _merge(_read_yaml(path.parent / base, (*seen, path)), raw, "")


def _merge(base: dict, top: dict, where: str) -> dict:
    """Deep merge; every key in ``top`` must exist in ``base`` (typos fail loudly)."""
    out = copy.deepcopy(base)
    for key, value in top.items():
        name = f"{where}.{key}" if where else key
        _require(key in out, f"unknown key {name!r} in experiment config")
        out[key] = _merge(out[key], value, name) if isinstance(value, dict) and isinstance(out[key], dict) else value
    return out


def _apply_overrides(raw: dict, overrides: Mapping[str, Any] | Sequence[str] | None) -> dict:
    if not overrides:
        return raw
    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    elif isinstance(overrides, str):
        raise ConfigError("overrides must be a list of 'key=value' strings, not a single string")
    else:
        items = [_parse_assignment(s) for s in overrides]
    raw = copy.deepcopy(raw)
    for dotted, value in items:
        *parents, leaf = dotted.split(".")
        node = raw
        for key in parents:
            _require(isinstance(node.get(key), dict), f"unknown override key {dotted!r}")
            node = node[key]
        _require(leaf in node, f"unknown override key {dotted!r}")
        node[leaf] = value
    return raw


def _parse_assignment(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    _require(bool(sep) and bool(key.strip()), f"override {text!r} must look like 'section.key=value'")
    return key.strip(), yaml.safe_load(value)  # parse the value exactly as YAML would


def _build(cls: type, data: Any, where: str) -> Any:
    _require(isinstance(data, Mapping), f"{where or 'config'} must be a mapping")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown, missing = set(data) - names, names - set(data)
    _require(not unknown, f"unknown key(s) in {where or 'config'}: {sorted(unknown)}")
    _require(not missing, f"missing key(s) in {where or 'config'}: {sorted(missing)}")
    return cls(**{n: _coerce(hints[n], data[n], f"{where}.{n}" if where else n) for n in names})


def _coerce(tp: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, where)
    origin = get_origin(tp)
    if origin in (Union, UnionType) and NoneType in get_args(tp):  # X | None: YAML null is allowed
        (inner,) = [a for a in get_args(tp) if a is not NoneType]
        return None if value is None else _coerce(inner, value, where)
    if origin is tuple:
        _require(isinstance(value, Sequence) and not isinstance(value, str), f"{where} must be a list")
        (item_tp, _ellipsis) = get_args(tp)
        return tuple(_coerce(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin is Mapping:
        _require(isinstance(value, Mapping), f"{where} must be a mapping")
        key_tp, val_tp = get_args(tp)
        return MappingProxyType(
            {_coerce(key_tp, k, f"{where} key"): _coerce(val_tp, v, f"{where}.{k}") for k, v in value.items()}
        )
    if tp is bool:
        _require(isinstance(value, bool), f"{where} must be true/false, got {value!r}")
        return value
    if tp is int:
        _require(isinstance(value, int) and not isinstance(value, bool), f"{where} must be an integer, got {value!r}")
        return value
    if tp is float:
        # Catches the PyYAML pitfall where "1e9" is read as a string.
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{where} must be a number, got {value!r} (write 1.0e+9, not 1e9)",
        )
        return float(value)
    if tp is str:
        _require(isinstance(value, str), f"{where} must be a string, got {value!r}")
        return value
    if tp is dt.date:
        if isinstance(value, str):
            value = dt.date.fromisoformat(value)
        _require(isinstance(value, dt.date) and not isinstance(value, dt.datetime), f"{where} must be a YYYY-MM-DD date")
        return value
    if tp is Path:
        _require(isinstance(value, (str, Path)), f"{where} must be a path")
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p
    raise ConfigError(f"{where}: unsupported field type {tp!r}")


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
