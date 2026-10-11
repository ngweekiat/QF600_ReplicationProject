"""Download the configured WRDS data and public JKP/French reference files.

This module is reconstructed from the repository's config, documentation, and tests.
It uses WRDS's normal PostgreSQL authentication, not an API key. No connection is
opened on import. SQL values are parameterised; identifiers are validated separately.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from jkmp.config import PROJECT_ROOT, Config

log = logging.getLogger(__name__)
RAW_FILES = {
    "jkp": "jkp_usa.parquet",
    "daily": "crsp_daily.parquet",
    "ff_monthly": "ff_factors_monthly.parquet",
    "ff_daily": "ff_factors_daily.parquet",
    "factor_details": "factor_details.parquet",
    "cluster_labels": "cluster_labels.parquet",
}
CHUNK_ROWS = 100_000
LOCAL_LOGIN_PATH = PROJECT_ROOT / "configs" / "wrds_login.local.json"


def _login_options() -> dict[str, str]:
    """Load local credentials without displaying or logging their contents."""
    credentials = {}
    if LOCAL_LOGIN_PATH.exists():
        credentials = json.loads(LOCAL_LOGIN_PATH.read_text(encoding="utf-8"))
        if not isinstance(credentials, dict):
            raise ValueError("local WRDS login file must contain a JSON object")
        for key in ("username", "password"):
            if not isinstance(credentials.get(key), str) or not credentials[key]:
                raise ValueError(f"local WRDS login file needs a non-empty {key}")
    username = os.environ.get("WRDS_USERNAME", "").strip() or credentials.get("username", "")
    options = {"wrds_username": username} if username else {}
    # A username override must not send the local password to a different account.
    if credentials and username == credentials["username"]:
        options["wrds_password"] = credentials["password"]
    return options


@contextmanager
def connect():
    """Connect once; always close. Let WRDS prompt securely if credentials are needed.

    Optional configs/wrds_login.local.json supplies automatic local login.
    WRDS_USERNAME can override the username. Otherwise WRDS can use PostgreSQL's
    password file or an interactive prompt. Credentials never enter the run YAML.
    On Windows the saved file is %APPDATA%/postgresql/pgpass.conf.
    """
    import wrds

    options = _login_options()
    if "wrds_password" in options:
        log.info("Connecting to WRDS with the saved local login.")
        db = wrds.Connection(autoconnect=False, **options)
        # WRDS's normal connect() hides the initial error and prompts again.
        # Its connection helper preserves the original error with raise_err=True.
        db._Connection__make_sa_engine_conn(raise_err=True)
        try:
            db.load_library_list()
        except Exception:
            db.close()
            raise
    else:
        db = wrds.Connection(**options)
    try:
        yield db
    finally:
        db.close()


def _identifier(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return '"' + name + '"'


def _qualified(table: str) -> str:
    parts = table.split(".")
    if len(parts) != 2:
        raise ValueError("WRDS table must have the form library.table")
    return ".".join(_identifier(p) for p in parts)


def _columns(db, table: str) -> set[str]:
    return set(db.raw_sql(f"SELECT * FROM {_qualified(table)} LIMIT 0").columns)


def _jkp_columns(cfg: Config) -> list[str]:
    return list(dict.fromkeys([
        "id", "eom", "sic", "size_grp", "me", "excntry",
        "crsp_exchcd", "crsp_shrcd", "ret_exc", "ret_exc_lead1m",
        cfg.trading_costs.dolvol_column, *cfg.universe.jkp_flags, *cfg.features.names,
    ]))


def _require_columns(available: set[str], required: list[str], table: str) -> None:
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}. Run the explore command to inspect it.")


def _daily_columns(available: set[str]) -> tuple[str, str]:
    date = next((c for c in ("dlycaldt", "date") if c in available), None)
    ret = next((c for c in ("dlyret", "ret") if c in available), None)
    if "permno" not in available or date is None or ret is None:
        raise ValueError("CRSP table needs permno, dlycaldt/date and dlyret/ret columns")
    return date, ret


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "QF600-Replication/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def read_french_zip(blob: bytes) -> pd.DataFrame:
    """Read the monthly OR daily section; omit headers and annual summaries."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError("expected one CSV inside the French ZIP")
        lines = z.read(names[0]).decode("utf-8-sig").splitlines()
    rows, width = [], None
    for line in lines:
        parts = [s.strip() for s in line.split(",")]
        date = parts[0]
        if not re.fullmatch(r"\d{6}|\d{8}", date):
            continue
        if len(parts) != 5:
            raise ValueError(f"unexpected French factor row: {line!r}")
        if width is not None and len(date) != width:
            raise ValueError("mixed monthly and daily rows in French data")
        width = len(date)
        rows.append(parts)
    if not rows:
        raise ValueError("French ZIP contains no monthly/daily factor observations")
    out = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    out["date"] = pd.to_datetime(out["date"], format="%Y%m" if width == 6 else "%Y%m%d")
    if width == 6:
        out["date"] += pd.offsets.MonthEnd(0)
    for name in ("mkt_rf", "smb", "hml", "rf"):
        values = pd.to_numeric(out[name]).astype("float64")
        # French missing-value sentinels are in percent units, before conversion.
        out[name] = values.mask(values.isin([-99.99, -999.0])) / 100.0
    if out["date"].duplicated().any():
        raise ValueError("duplicate dates in French factors")
    return out.sort_values("date").reset_index(drop=True)


def top_n_by_month(jkp: pd.DataFrame, n: int) -> pd.DataFrame:
    """Stable market-cap ranking, id as the tie breaker."""
    if n < 1:
        raise ValueError("n must be positive")
    rows = jkp[["eom", "id", "me"]].dropna(subset=["me"]).copy()
    rows = rows.sort_values(["eom", "me", "id"], ascending=[True, False, True])
    rows["me_rank"] = rows.groupby("eom").cumcount() + 1
    return rows[rows["me_rank"] <= n].reset_index(drop=True)


def _daily_counts(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.assign(eom=pd.to_datetime(daily["date"]) + pd.offsets.MonthEnd(0))
    return d.groupby("eom").agg(daily_rows=("permno", "size"), daily_permnos=("permno", "nunique")).reset_index()


def row_counts_by_month(jkp: pd.DataFrame, top: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    counts = jkp.groupby("eom").agg(
        jkp_rows=("id", "size"), me_nonnull=("me", "count"),
        ret_lead1m_nonnull=("ret_exc_lead1m", "count"),
    ).reset_index()
    top_counts = top.groupby("eom").size().rename("top_n").reset_index()
    return _combine_counts(counts, top_counts, _daily_counts(daily))


def _combine_counts(jkp: pd.DataFrame, top: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = jkp.merge(top, on="eom", how="outer").merge(daily, on="eom", how="outer").sort_values("eom")
    cols = [c for c in out if c != "eom"]
    out[cols] = out[cols].fillna(0).astype("int64")
    return out.reset_index(drop=True)


def _counts_from_parquet(path: Path) -> pd.DataFrame:
    """Scan daily rows in batches rather than materialising the full daily panel."""
    counts, stocks = {}, {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=CHUNK_ROWS, columns=["date", "permno"]):
        frame = batch.to_pandas()
        frame["eom"] = frame["date"] + pd.offsets.MonthEnd(0)
        for eom, group in frame.groupby("eom"):
            counts[eom] = counts.get(eom, 0) + len(group)
            stocks.setdefault(eom, set()).update(group["permno"])
    return pd.DataFrame(
        [(eom, counts[eom], len(stocks[eom])) for eom in sorted(counts)],
        columns=["eom", "daily_rows", "daily_permnos"],
    )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_jkp(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame[columns].copy()
    frame["eom"] = pd.to_datetime(frame["eom"]).dt.normalize()
    frame["id"] = pd.to_numeric(frame["id"], errors="raise").astype("int64")
    for name in columns:
        if name in ("id", "eom"):
            continue
        if name in ("size_grp", "excntry"):
            frame[name] = frame[name].astype("string")
        else:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
    if frame.duplicated(["id", "eom"]).any():
        raise ValueError("duplicate JKP stock-months after main-share filters")
    return frame


def _normalise_daily(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[["permno", "date", "ret"]].copy()
    frame["permno"] = pd.to_numeric(frame["permno"], errors="raise").astype("int64")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["ret"] = pd.to_numeric(frame["ret"], errors="coerce").astype("float64")
    if frame.duplicated(["permno", "date"]).any():
        raise ValueError("duplicate CRSP stock-days")
    return frame


def _stream_years(db, sql: str, params: dict, date_col: str, start, end,
                  path: Path, normalise) -> None:
    """Download year by year, stream to Parquet, and publish only a complete file."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    temporary = path.with_suffix(path.suffix + ".part")
    writer, n, last_key = None, 0, None
    try:
        for year in range(start.year, end.year + 1):
            lo, hi = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
            query_params = dict(params, start=lo.date(), end=hi.date())
            log.info("%s: downloading %d", path.name, year)
            chunks = db.raw_sql(sql, params=query_params, date_cols=[date_col],
                                chunksize=CHUNK_ROWS, return_iter=True)
            for chunk in chunks:
                if chunk.empty:
                    continue
                frame = normalise(chunk)
                id_col = "id" if "id" in frame else "permno"
                key_cols = [date_col, id_col]
                first_key = tuple(frame[key_cols].iloc[0])
                if first_key == last_key:
                    raise ValueError(f"duplicate observation at a download chunk boundary in {path.name}")
                last_key = tuple(frame[key_cols].iloc[-1])
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
                writer.write_table(table)
                n += len(frame)
        if writer is None:
            raise ValueError(f"no rows returned for {path.name}; check dates, filters and WRDS subscriptions")
        writer.close()
        writer = None
        temporary.replace(path)
        log.info("Saved %s (%s rows)", path.name, f"{n:,}")
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _pull_jkp(db, cfg: Config, path: Path) -> None:
    columns = _jkp_columns(cfg)
    _require_columns(_columns(db, cfg.data_sources.jkp_table), columns, cfg.data_sources.jkp_table)
    flags = " AND ".join(f"{_identifier(f)} = 1" for f in cfg.universe.jkp_flags)
    filters = [
        '"eom" BETWEEN %(start)s AND %(end)s', '"excntry" = %(country)s',
        '"id" BETWEEN 1 AND 99999',
        '"crsp_exchcd" IN %(exchanges)s', '"crsp_shrcd" IN %(shares)s',
    ]
    if flags:
        filters.append(flags)
    sql = (f"SELECT {', '.join(_identifier(c) for c in columns)} FROM {_qualified(cfg.data_sources.jkp_table)} "
           f"WHERE {' AND '.join(filters)} ORDER BY \"eom\", \"id\"")
    _stream_years(db, sql, {"country": "USA", "exchanges": tuple(cfg.universe.exchanges),
                           "shares": tuple(cfg.universe.share_codes)}, "eom",
                  cfg.dates.warmup_start, cfg.dates.test_end, path, lambda d: _normalise_jkp(d, columns))


def _pull_daily(db, cfg: Config, ids: list[int], path: Path) -> None:
    date, ret = _daily_columns(_columns(db, cfg.data_sources.crsp_daily_table))
    if not ids:
        raise ValueError("no stocks in the covariance factor universe")
    sql = (f"SELECT \"permno\", {_identifier(date)} AS \"date\", {_identifier(ret)} AS \"ret\" "
           f"FROM {_qualified(cfg.data_sources.crsp_daily_table)} "
           f"WHERE {_identifier(date)} BETWEEN %(start)s AND %(end)s AND \"permno\" IN %(ids)s "
           f"ORDER BY {_identifier(date)}, \"permno\"")
    first_day = pd.Timestamp(cfg.dates.warmup_start).replace(day=1)
    _stream_years(db, sql, {"ids": tuple(ids)}, "date", first_day, cfg.dates.test_end,
                  path, _normalise_daily)


def pull_public(cfg: Config, force: bool = False) -> dict[str, Path]:
    """Download public inputs; no WRDS login required."""
    raw, src = cfg.paths.raw, cfg.data_sources
    raw.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key, url in (("ff_monthly", src.ff_factors_url), ("ff_daily", src.ff_factors_daily_url)):
        path = raw / RAW_FILES[key]
        if force or not path.exists():
            log.info("Downloading %s", path.name)
            _atomic_parquet(read_french_zip(_fetch(url)), path)
        outputs[key] = path
    for key, url, original in (
        ("factor_details", src.factor_details_url, "Factor Details.xlsx"),
        ("cluster_labels", src.cluster_labels_url, "Cluster Labels.csv"),
    ):
        path = raw / RAW_FILES[key]
        if force or not path.exists():
            log.info("Downloading %s", original)
            blob = _fetch(url)
            frame = (pd.read_excel(io.BytesIO(blob)) if key == "factor_details"
                     else pd.read_csv(io.BytesIO(blob)))
            required = ["abr_jkp", "direction"] if key == "factor_details" else ["characteristic", "cluster"]
            _require_columns(set(frame), required, original)
            if key == "factor_details":
                frame["direction"] = pd.to_numeric(frame["direction"], errors="coerce").astype("float64")
            # Excel's descriptive columns can mix numbers and letters (e.g. time_to_create).
            # Keep their content as text so Arrow does not infer an incompatible integer column.
            for column in frame.select_dtypes(include="object"):
                frame[column] = frame[column].astype("string")
            _atomic_parquet(frame, path)
            (raw / original).write_bytes(blob)
        outputs[key] = path
    # A missing/wrong theme assignment should fail before an expensive WRDS pull.
    from jkmp.features import load_theme_directions
    load_theme_directions(raw, cfg.features.themes)
    return outputs


def _download_signature(cfg: Config) -> str:
    settings = {
        "version": 1, "start": str(cfg.dates.warmup_start), "end": str(cfg.dates.test_end),
        "jkp_table": cfg.data_sources.jkp_table, "daily_table": cfg.data_sources.crsp_daily_table,
        "columns": _jkp_columns(cfg), "exchanges": list(cfg.universe.exchanges),
        "shares": list(cfg.universe.share_codes), "flags": list(cfg.universe.jkp_flags),
        "top_n": cfg.covariance.factor_universe_top_n,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def pull_all(db, cfg: Config, force: bool = False) -> dict[str, Path]:
    raw = cfg.paths.raw
    raw.mkdir(parents=True, exist_ok=True)
    manifest = raw / "download_manifest.json"
    signature = _download_signature(cfg)
    state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    cached = [raw / RAW_FILES[k] for k in ("jkp", "daily")]
    if any(p.exists() for p in cached) and not force:
        if state.get("signature") != signature:
            raise ValueError("cached WRDS data has unknown/different settings; use pull --force or a different paths.raw")
    completed = set(state.get("completed", [])) if not force and state.get("signature") == signature else set()

    def save_state():
        temporary = manifest.with_suffix(".json.part")
        temporary.write_text(json.dumps({"signature": signature, "settings": cfg.as_dict(),
                                         "completed": sorted(completed)}, indent=2, default=str), encoding="utf-8")
        temporary.replace(manifest)

    # An interrupted forced refresh must not mark old files as belonging to new settings.
    save_state()
    outputs = pull_public(cfg, force=force)
    jkp_path = raw / RAW_FILES["jkp"]
    if force or not jkp_path.exists() or "jkp" not in completed:
        completed.discard("daily")
        save_state()
        _pull_jkp(db, cfg, jkp_path)
        completed.add("jkp")
        save_state()
    jkp = pd.read_parquet(jkp_path)
    if jkp.empty or jkp.duplicated(["id", "eom"]).any():
        raise ValueError("cached JKP data is empty or has duplicate stock-months")
    n = cfg.covariance.factor_universe_top_n
    top = top_n_by_month(jkp, n)
    top_path = raw / f"top{n}_by_month.parquet"
    _atomic_parquet(top, top_path)
    csv_path = raw / f"jkp_usa_top{n}.csv"
    jkp.merge(top[["eom", "id", "me_rank"]], on=["eom", "id"], validate="one_to_one").to_csv(csv_path, index=False)
    daily_path = raw / RAW_FILES["daily"]
    if force or not daily_path.exists() or "daily" not in completed:
        _pull_daily(db, cfg, sorted(int(i) for i in top["id"].unique()), daily_path)
        completed.add("daily")
        save_state()
    counts = _combine_counts(
        jkp.groupby("eom").agg(jkp_rows=("id", "size"), me_nonnull=("me", "count"),
                              ret_lead1m_nonnull=("ret_exc_lead1m", "count")).reset_index(),
        top.groupby("eom").size().rename("top_n").reset_index(),
        _counts_from_parquet(daily_path),
    )
    counts_path = raw / "row_counts_by_month.csv"
    counts.to_csv(counts_path, index=False)
    outputs.update(jkp=jkp_path, top=top_path, top_csv=csv_path, daily=daily_path, counts=counts_path)
    expected = pd.date_range(cfg.dates.warmup_start, cfg.dates.test_end, freq="ME")
    missing = expected.difference(pd.DatetimeIndex(jkp["eom"].unique()))
    if len(missing):
        raise ValueError(f"JKP data is missing {len(missing)} configured months, including {list(missing[:5])}")
    return outputs


def explore(db, cfg: Config) -> dict:
    """Inspect configured tables and compatible names, without downloading stock data."""
    report = {}
    for key, table in (("jkp", cfg.data_sources.jkp_table), ("daily", cfg.data_sources.crsp_daily_table)):
        try:
            available = _columns(db, table)
            if key == "jkp":
                missing = sorted(set(_jkp_columns(cfg)) - available)
                date = "eom"
            else:
                date, _ = _daily_columns(available)
                missing = []
            bounds = db.raw_sql(
                f"SELECT MIN({_identifier(date)}) AS first_date, MAX({_identifier(date)}) AS last_date "
                f"FROM {_qualified(table)}")
            report[key] = {"table": table, "column_count": len(available), "missing_columns": missing,
                           "date_range": bounds.iloc[0].astype(str).to_dict(),
                           "ok": not missing}
        except Exception as exc:
            # Do not echo connection errors/URIs, which can contain credentials.
            report[key] = {"table": table, "ok": False,
                           "error": f"{type(exc).__name__}: cannot inspect required table/columns; check subscriptions and table name"}
    candidates = {}
    for library, terms in (("contrib", ("factor", "global")), ("crsp", ("dsf",))):
        try:
            candidates[library] = [t for t in db.list_tables(library=library) if any(s in t for s in terms)]
        except Exception:
            candidates[library] = ["unable to list tables for this library"]
    report["candidates"] = candidates
    return report


def format_report(report: dict) -> str:
    return json.dumps(report, indent=2, default=str)
