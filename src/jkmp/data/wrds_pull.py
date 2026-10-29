"""Stage 0: locate and pull the raw data (WRDS, JKP reference files, Ken French).

Credentials never live in code or config: ``connect()`` reads them from environment variables, which may be
set in the shell or in the git-ignored project ``.env`` (template: ``.env.example``).

``explore()`` finds candidate WRDS tables and checks them against what the pipeline needs;
``pull_all()`` writes every raw input to ``data/raw/`` (skipping files that already exist).
"""

from __future__ import annotations

import io
import logging
import os
import re
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv

from jkmp.config import PROJECT_ROOT, Config

if TYPE_CHECKING:  # wrds (and its sqlalchemy) are imported only where a connection is used
    import wrds

log = logging.getLogger(__name__)

# Identifier and bookkeeping columns the JKP extract needs besides the characteristics in the config.
# size_grp: size-group median for missing idiosyncratic variance (R, 3 - Estimate Covariance Matrix.R).
# ret_exc: return over the month ending at eom, for the growth term g_t (tr_ld0 in R).
JKP_BASE_COLUMNS = (
    "id", "eom", "excntry", "sic", "size_grp", "me", "crsp_exchcd", "crsp_shrcd",
    "dolvol_126d", "ret_exc", "ret_exc_lead1m",
)

# crsp.dsf_v2 column names -> ours.
CRSP_V2_COLUMNS = {"permno": "permno", "dlycaldt": "date", "dlyret": "ret"}
CRSP_PERMNO_BATCH = 250

# Output files in data/raw/ (the top-N file name uses covariance.factor_universe_top_n).
RAW_FILES = {
    "jkp": "jkp_usa.parquet",
    "daily": "crsp_daily.parquet",
    "ff_monthly": "ff_factors_monthly.parquet",
    "ff_daily": "ff_factors_daily.parquet",
    "factor_details": "factor_details.parquet",
    "cluster_labels": "cluster_labels.parquet",
    "counts": "row_counts_by_month.csv",
}

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

# Table-name patterns (SQL ILIKE) for each dataset we need to locate.
CANDIDATE_PATTERNS = {
    "jkp": ["%global_factor%", "%jkp%"],
    "crsp_daily": ["dsf", "dsf_v2", "dsf62", "stkdlysecuritydata", "dsi", "wrds_dsfv2%"],
}

# Probe stock for cheap date-coverage checks: Apple, PERMNO 14593 (also its JKP id).
PROBE_PERMNO = 14593
DATE_COLUMN_CANDIDATES = ("eom", "date", "dlycaldt", "caldt")
QUERY_TIMEOUT_SECONDS = 180


# --------------------------------------------------------------------------- connection


def connect(env_file: Path = PROJECT_ROOT / ".env") -> wrds.Connection:
    """Open a WRDS connection: ``wrds.Connection(wrds_username=..., api_key=...)``.

    Reads WRDS_USERNAME and WRDS_API_KEY from the environment, after loading ``env_file`` if it exists;
    variables already set in the shell win over the file.
    wrds 3.5.0 accepts but ignores ``api_key``; the password is then found in ~/.pgpass.
    Set WRDS_PASSWORD only to override ~/.pgpass — never put the API key there.
    """
    import wrds

    load_dotenv(env_file, override=False)
    kwargs = {}
    if username := os.environ.get("WRDS_USERNAME"):
        kwargs["wrds_username"] = username
    if api_key := os.environ.get("WRDS_API_KEY"):
        kwargs["api_key"] = api_key
    if password := os.environ.get("WRDS_PASSWORD"):
        kwargs["wrds_password"] = password
    return wrds.Connection(**kwargs)


def _set_timeout(db: wrds.Connection, seconds: int) -> None:
    import sqlalchemy as sa

    db.connection.execute(sa.text(f"set statement_timeout = {int(seconds * 1000)}"))


# --------------------------------------------------------------------------- exploration


@dataclass
class TableReport:
    dataset: str
    schema: str
    table: str
    n_columns: int
    date_column: str | None
    missing_columns: list[str] = field(default_factory=list)
    coverage: str = ""


@dataclass
class ExploreReport:
    libraries: list[str]
    candidates: pd.DataFrame
    tables: list[TableReport]
    feature_check: pd.DataFrame


def find_candidate_tables(db: wrds.Connection) -> pd.DataFrame:
    """Tables/views the user can see whose names match ``CANDIDATE_PATTERNS``."""
    frames = []
    for dataset, patterns in CANDIDATE_PATTERNS.items():
        df = db.raw_sql(
            """
            select table_schema as schema, table_name as table, table_type
            from information_schema.tables
            where table_name ilike any(%(patterns)s)
            order by table_schema, table_name
            """,
            params={"patterns": patterns},
        )
        if dataset == "crsp_daily":
            df = df[df["schema"].str.lower().str.startswith("crsp")]
        frames.append(df.assign(dataset=dataset))
    return pd.concat(frames, ignore_index=True)


def table_columns(db: wrds.Connection, schema: str, table: str) -> list[str]:
    df = db.raw_sql(
        """
        select column_name from information_schema.columns
        where table_schema = %(schema)s and table_name = %(table)s
        order by ordinal_position
        """,
        params={"schema": schema, "table": table},
    )
    return df["column_name"].str.lower().tolist()


def probe_coverage(db: wrds.Connection, schema: str, table: str, columns: list[str]) -> tuple[str | None, str]:
    """Date range for the probe stock (index-friendly), as a printable string."""
    date_col = next((c for c in DATE_COLUMN_CANDIDATES if c in columns), None)
    id_col = next((c for c in ("permno", "id") if c in columns), None)
    if date_col is None:
        return None, "no recognised date column"
    if id_col is None:
        return date_col, "no permno/id column to probe"
    where = f"{id_col} = %(probe)s"
    if "excntry" in columns:
        where += " and excntry = 'USA'"
    try:
        _set_timeout(db, QUERY_TIMEOUT_SECONDS)
        df = db.raw_sql(
            f'select min({date_col}) as first, max({date_col}) as last, count(*) as n '
            f'from "{schema}"."{table}" where {where}',
            params={"probe": PROBE_PERMNO},
        )
    except Exception as err:  # timeouts and permission errors are findings, not failures
        return date_col, f"probe failed: {type(err).__name__}: {str(err).splitlines()[0][:120]}"
    finally:
        _set_timeout(db, 0)
    row = df.iloc[0]
    return date_col, f"{row['first']} to {row['last']} ({row['n']} rows for PERMNO {PROBE_PERMNO})"


def download_reference_file(url: str, dest_dir: Path) -> Path:
    """Download a small public file (JKP reference files) unless it already exists."""
    dest = dest_dir / urllib.request.url2pathname(url.rsplit("/", 1)[-1])
    if not dest.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s", url)
        dest.write_bytes(_http_get(url))
    return dest


def check_features(cfg: Config, factor_details: Path, cluster_labels: Path) -> pd.DataFrame:
    """Compare config features against JKP's Factor Details (names) and Cluster Labels (themes)."""
    details = pd.read_excel(factor_details)
    known = set(details["abr_jkp"].dropna().astype(str))
    direction = details.dropna(subset=["abr_jkp"]).set_index("abr_jkp")["direction"]
    labels = pd.read_csv(cluster_labels)
    # R (1 - Prepare Data.R): lower-case, spaces/hyphens -> underscores; rvol_252d added to low_risk.
    clusters = labels["cluster"].str.lower().str.replace(r"[\s-]", "_", regex=True)
    theme_of = dict(zip(labels["characteristic"], clusters)) | {"rvol_252d": "low_risk"}
    rows = []
    for feature in cfg.features.names:
        jkp_theme = theme_of.get(feature)
        rows.append({
            "feature": feature,
            "config_theme": cfg.features.theme_of[feature],
            "in_factor_details": feature in known,
            "jkp_theme": jkp_theme,
            "theme_matches": jkp_theme == cfg.features.theme_of[feature],
            "direction": direction.get(feature),
        })
    return pd.DataFrame(rows)


def explore(db: wrds.Connection, cfg: Config) -> ExploreReport:
    """Find JKP and CRSP daily candidates, check required columns and coverage, check features."""
    libraries = sorted(db.list_libraries())
    candidates = find_candidate_tables(db)
    required = {"jkp": list(JKP_BASE_COLUMNS) + list(cfg.features.names), "crsp_daily": ["permno"]}
    tables = []
    for row in candidates.itertuples(index=False):
        columns = table_columns(db, row.schema, row.table)
        date_col, coverage = probe_coverage(db, row.schema, row.table, columns)
        tables.append(TableReport(
            dataset=row.dataset, schema=row.schema, table=row.table, n_columns=len(columns),
            date_column=date_col,
            missing_columns=[c for c in required[row.dataset] if c not in columns],
            coverage=coverage,
        ))
    details = download_reference_file(cfg.data_sources.factor_details_url, cfg.paths.raw)
    clusters = download_reference_file(cfg.data_sources.cluster_labels_url, cfg.paths.raw)
    return ExploreReport(libraries, candidates, tables, check_features(cfg, details, clusters))


def format_report(report: ExploreReport) -> str:
    interesting = [lib for lib in report.libraries if any(k in lib.lower() for k in ("crsp", "contrib", "jkp", "global"))]
    lines = [
        f"WRDS libraries visible: {len(report.libraries)}",
        f"  relevant: {', '.join(interesting) or '(none)'}",
        "",
        "Candidate tables:",
    ]
    for t in report.tables:
        missing = ", ".join(t.missing_columns) if t.missing_columns else "none"
        lines += [
            f"  [{t.dataset}] {t.schema}.{t.table}  ({t.n_columns} columns, date column: {t.date_column})",
            f"      coverage: {t.coverage}",
            f"      missing required columns: {missing}",
        ]
    if not report.tables:
        lines.append("  (none found)")
    lines += ["", "Feature check against Factor Details.xlsx / Cluster Labels.csv:",
              report.feature_check.to_string(index=False)]
    return "\n".join(lines)


# --------------------------------------------------------------------------- SQL helpers


def _identifier(name: str) -> str:
    """Quote a column/schema/table name after checking it is a plain identifier (no injection)."""
    if not _IDENTIFIER.match(name):
        raise ValueError(f"not a plain SQL identifier: {name!r}")
    return f'"{name}"'


def _qualified(table: str) -> str:
    schema, sep, name = table.partition(".")
    if not sep:
        raise ValueError(f"table must be 'schema.table', got {table!r}")
    return f"{_identifier(schema)}.{_identifier(name)}"


# --------------------------------------------------------------------------- pulls


def pull_jkp(db: wrds.Connection, cfg: Config) -> pd.DataFrame:
    """JKP characteristics for US common stocks on the configured exchanges, warm-up to data end."""
    columns = [c for c in JKP_BASE_COLUMNS if c != "excntry"] + list(cfg.features.names)
    flags = "".join(f" and {_identifier(f)} = 1" for f in cfg.universe.jkp_flags)
    sql = f"""
        select {", ".join(_identifier(c) for c in columns)}
        from {_qualified(cfg.data_sources.jkp_table)}
        where excntry = 'USA'
          and eom between %(start)s and %(end)s
          and id <= 99999
          and crsp_exchcd = any(%(exchanges)s)
          and crsp_shrcd = any(%(share_codes)s){flags}
    """
    params = {
        "start": cfg.dates.warmup_start,
        "end": cfg.dates.data_end,
        "exchanges": list(cfg.universe.exchanges),
        "share_codes": list(cfg.universe.share_codes),
    }
    log.info("Pulling JKP from %s (%s to %s) ...", cfg.data_sources.jkp_table, params["start"], params["end"])
    df = db.raw_sql(sql, params=params, date_cols=["eom"])
    df = _tidy_numeric(df, int_cols=("id",))
    dupes = df.duplicated(["id", "eom"]).sum()
    if dupes:
        log.warning("JKP extract has %d duplicate (id, eom) rows", dupes)
    return df.sort_values(["eom", "id"], ignore_index=True)


def top_n_by_month(jkp: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n largest stocks by market cap each month: columns eom, id, me, me_rank (1 = largest)."""
    ranked = jkp.loc[jkp["me"].notna(), ["eom", "id", "me"]].sort_values(["eom", "id"])
    ranked["me_rank"] = ranked.groupby("eom")["me"].rank(method="first", ascending=False).astype("int64")
    return ranked[ranked["me_rank"] <= n].sort_values(["eom", "me_rank"], ignore_index=True)


def pull_crsp_daily(db: wrds.Connection, cfg: Config, permnos: Sequence[int]) -> pd.DataFrame:
    """Daily total returns from the CRSP v2 file for the given PERMNOs, over the full sample."""
    select = ", ".join(f"{_identifier(src)} as {_identifier(dst)}" for src, dst in CRSP_V2_COLUMNS.items())
    sql = f"""
        select {select}
        from {_qualified(cfg.data_sources.crsp_daily_table)}
        where permno = any(%(permnos)s) and dlycaldt between %(start)s and %(end)s
    """
    start, end = cfg.dates.warmup_start.replace(day=1), cfg.dates.data_end
    permnos = sorted(int(p) for p in permnos)
    frames = []
    for i in range(0, len(permnos), CRSP_PERMNO_BATCH):
        batch = permnos[i:i + CRSP_PERMNO_BATCH]
        frames.append(db.raw_sql(sql, params={"permnos": batch, "start": start, "end": end}, date_cols=["date"]))
        log.info("CRSP daily: %d/%d PERMNOs, %d rows so far",
                 i + len(batch), len(permnos), sum(len(f) for f in frames))
    df = _tidy_numeric(pd.concat(frames, ignore_index=True), int_cols=("permno",))
    return df.sort_values(["permno", "date"], ignore_index=True)


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (jkmp replication)"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        return resp.read()


def read_french_zip(content: bytes) -> pd.DataFrame:
    """Parse a Ken French factor CSV zip (monthly or daily section) into decimals.

    Returns columns ``date`` (month-end for monthly files) plus e.g. mkt_rf, smb, hml, rf.
    Rows with 4-digit keys (the annual section of monthly files) are dropped.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    header, rows = None, []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None and len(parts) > 1 and parts[0] == "" and parts[1]:
            header = [p.lower().replace("-", "_") for p in parts[1:]]
        elif header is not None and parts[0].isdigit() and len(parts[0]) in (6, 8):
            rows.append(parts[: len(header) + 1])
    if header is None or not rows:
        raise ValueError("unrecognised Ken French file layout")
    df = pd.DataFrame(rows, columns=["key", *header])
    key_len = df["key"].str.len()
    if key_len.nunique() != 1:
        raise ValueError("Ken French file mixes monthly and daily rows")
    if key_len.iloc[0] == 6:
        date = pd.to_datetime(df["key"], format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        date = pd.to_datetime(df["key"], format="%Y%m%d")
    values = df[header].astype("float64") / 100.0  # French files are in percent
    return pd.concat([date.rename("date"), values], axis=1)


def _tidy_numeric(df: pd.DataFrame, int_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Nullable WRDS dtypes -> plain int64 identifiers and float64 values (NaN for missing)."""
    df = df.copy()
    for col in df.columns:
        if col in int_cols:
            df[col] = df[col].astype("int64")
        elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("float64")
    return df


def _stringify_objects(df: pd.DataFrame) -> pd.DataFrame:
    """Mixed-type spreadsheet columns -> strings, so they can be written to parquet."""
    df = df.copy()
    for col in df.columns[df.dtypes == object]:
        df[col] = df[col].astype("string")
    return df


# --------------------------------------------------------------------------- bookkeeping


def row_counts_by_month(jkp: pd.DataFrame, top: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Per month: JKP rows, non-missing me / next-month return, top-N size, CRSP daily rows and stocks."""
    jkp_counts = jkp.groupby("eom").agg(
        jkp_rows=("id", "size"),
        me_nonnull=("me", "count"),
        ret_lead1m_nonnull=("ret_exc_lead1m", "count"),
    )
    top_counts = top.groupby("eom").size().rename("top_n")
    month = (daily["date"] + pd.offsets.MonthEnd(0)).rename("eom")
    daily_counts = daily.groupby(month).agg(daily_rows=("ret", "size"), daily_permnos=("permno", "nunique"))
    counts = jkp_counts.join(top_counts, how="outer").join(daily_counts, how="outer")
    return counts.fillna(0).astype("int64").rename_axis("eom").reset_index()


def log_row_counts(counts: pd.DataFrame) -> None:
    for row in counts.itertuples(index=False):
        log.info(
            "%s  jkp=%5d  me=%5d  ret_lead1m=%5d  top_n=%3d  daily_rows=%6d  daily_permnos=%4d",
            row.eom.strftime("%Y-%m"), row.jkp_rows, row.me_nonnull, row.ret_lead1m_nonnull,
            row.top_n, row.daily_rows, row.daily_permnos,
        )
    for col in ("jkp_rows", "top_n", "daily_permnos"):
        s = counts.loc[counts[col] > 0, col]
        log.info("%-14s per month: min %d, median %d, max %d", col, s.min(), int(s.median()), s.max())


def _cached(path: Path, build: Callable[[], pd.DataFrame], force: bool) -> pd.DataFrame:
    if path.exists() and not force:
        log.info("Using existing %s (pass --force to re-pull)", path.name)
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    log.info("Wrote %s: %d rows x %d columns", path.name, len(df), df.shape[1])
    return df


# --------------------------------------------------------------------------- orchestration


def pull_all(db: wrds.Connection, cfg: Config, force: bool = False) -> dict[str, Path]:
    """Pull every raw input into data/raw/. Existing files are reused unless ``force``."""
    raw = cfg.paths.raw
    raw.mkdir(parents=True, exist_ok=True)
    n = cfg.covariance.factor_universe_top_n
    paths = {key: raw / name for key, name in RAW_FILES.items()}
    paths["top"] = raw / f"top{n}_by_month.parquet"
    paths["csv"] = raw / f"jkp_usa_top{n}.csv"

    jkp = _cached(paths["jkp"], lambda: pull_jkp(db, cfg), force)
    top = _cached(paths["top"], lambda: top_n_by_month(jkp, n), force)

    # Deliverable: characteristics for the top-N stocks each month (the full extract stays in parquet).
    jkp.merge(top[["eom", "id", "me_rank"]], on=["eom", "id"]).to_csv(paths["csv"], index=False)
    log.info("Wrote %s", paths["csv"].name)

    daily = _cached(paths["daily"], lambda: pull_crsp_daily(db, cfg, top["id"].unique()), force)
    _cached(paths["ff_monthly"], lambda: read_french_zip(_http_get(cfg.data_sources.ff_factors_url)), force)
    _cached(paths["ff_daily"], lambda: read_french_zip(_http_get(cfg.data_sources.ff_factors_daily_url)), force)

    details = download_reference_file(cfg.data_sources.factor_details_url, raw)
    clusters = download_reference_file(cfg.data_sources.cluster_labels_url, raw)
    _cached(paths["factor_details"], lambda: _stringify_objects(pd.read_excel(details)), force)
    _cached(paths["cluster_labels"], lambda: pd.read_csv(clusters), force)

    counts = row_counts_by_month(jkp, top, daily)
    counts.to_csv(paths["counts"], index=False)
    log_row_counts(counts)
    return paths
