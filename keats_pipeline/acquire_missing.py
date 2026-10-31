"""Explicit, one-connection acquisition of missing Keats inputs.

Credentials are read only into memory from environment variables or the existing
local WRDS login file. They are never written into the acquisition outputs.
The normal run.py workflow stays offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflow import broaden_membership, numeric_ids, source_dir

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "data/inputs"
CACHE = ROOT / "data/acquisition_cache"
START = "1994-11-01"
END = "2025-12-31"
JKP_CODE = "https://github.com/bkelly-lab/ReplicationCrisis/blob/master/GlobalFactors/project_macros.sas"


def cached_query(db, name, sql, params=None, date_cols=None):
    path = CACHE / f"{name}.parquet"
    if path.exists():
        print(f"Reuse {name}", flush=True)
        return pd.read_parquet(path)
    frame = db.raw_sql(sql, params=params or {}, date_cols=date_cols or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(f"Downloaded {name}: {len(frame):,} rows", flush=True)
    return frame


def required_ids():
    ids = {}
    for region, universe in (("US", "us_large"), ("EU", "eu_large")):
        top = pd.read_parquet(ROOT / f"data/raw/{universe}/top500_by_month.parquet")
        ids[region] = set(top.id.astype(int))
    for universe in ("us_large", "us_mid", "eu_large"):
        frame = pd.read_parquet(source_dir() / f"panel_{universe}.parquet", columns=["id"])
        ids["EU" if universe.startswith("eu") else "US"].update(frame.id.astype(int))
    # Include the 250-stock mid-cap extension, which can lie outside the top-500 risk pool.
    frame = numeric_ids(pd.read_parquet(source_dir() / "screened_US.parquet", columns=["id", "eom", "me"]))
    ids["US"].update(broaden_membership(frame, "us_mid").id.astype(int))
    return {key: sorted(values) for key, values in ids.items()}


def save_input(name, frame, provenance=None):
    INPUTS.mkdir(parents=True, exist_ok=True)
    path = INPUTS / f"{name}.parquet"
    if path.exists():
        raise FileExistsError(f"Existing input retained: {path}")
    frame.to_parquet(path, index=False)
    if provenance is not None:
        path.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved {path.name}: {len(frame):,} rows, {path.stat().st_size / 1e6:.1f} MB", flush=True)


def monthly_gaps(db):
    frames = []
    for universe in ("us_large", "us_mid", "eu_large"):
        gaps = pd.read_csv(ROOT / f"outputs/{universe}/monthly_return_gaps.csv", parse_dates=["eom"])
        frames.append(gaps.assign(region="EU" if universe.startswith("eu") else "US"))
    gaps = pd.concat(frames, ignore_index=True)
    for region, group in gaps.groupby("region"):
        path = INPUTS / f"monthly_returns_{region}.parquet"
        if path.exists():
            print(f"Reuse {path.name}", flush=True)
            continue
        keys = set()
        for row in group.itertuples():
            for offset in (-1, 0, 1):
                keys.add((int(row.id), (row.eom + pd.offsets.MonthEnd(offset)).date()))
        frame = cached_query(db, f"monthly_gaps_{region}", """
            select id, eom, gvkey, iid, ret, ret_exc, ret_exc_lead1m
            from contrib.global_factor where (id,eom) in %(keys)s
            order by id,eom
        """, {"keys": tuple(sorted(keys))}, ["eom"])
        frame["id"] = frame.id.astype("int64")
        save_input(f"monthly_returns_{region}", frame, {
            "source": "WRDS contrib.global_factor, only requested missing stock/months and adjacent months",
            "currency": "USD", "return_type": "excess", "identifier": "jkp_id",
            "no_existing_observation_replaced": True,
        })
        merged = group.merge(frame[["id", "eom", "ret_exc", "ret_exc_lead1m"]], on=["id", "eom"], how="left")
        print(merged[["id", "eom", "ret_exc", "ret_exc_lead1m"]].to_string(index=False), flush=True)


def terminal_events(db):
    """Keep authoritative CRSP terminal-month data separate from the original JKP series."""
    destination = INPUTS / "us_terminal_months.parquet"
    if destination.exists():
        print(f"Reuse {destination.name}", flush=True)
        return
    gaps = pd.concat([pd.read_csv(ROOT / f"outputs/{u}/monthly_return_gaps.csv", parse_dates=["eom"])
                      for u in ("us_large", "us_mid")], ignore_index=True)
    gaps = gaps[gaps.missing_next_month_return]
    columns = db.raw_sql("""select column_name from information_schema.columns
        where table_schema='crsp' and table_name='msf_v2' order by ordinal_position""", params={})
    names = set(columns.column_name)
    needed = {"permno", "mthcaldt", "mthret"}
    if not needed.issubset(names):
        raise ValueError("CRSP monthly CIZ table does not expose the expected total-return columns")
    optional = [c for c in ("mthdelflg", "mthretmissflg", "mthretx", "mthreti") if c in names]
    fields = ",".join(["permno as id", "mthcaldt as date", "mthret as ret", *optional])
    frame = cached_query(db, "us_terminal_months", "select " + fields + """
        from crsp.msf_v2 where permno in %(ids)s and mthcaldt between %(start)s and %(end)s
        order by permno,mthcaldt
    """, {"ids": tuple(int(value) for value in sorted(gaps.id.unique())), "start": str(gaps.eom.min().date()),
           "end": str((gaps.eom.max() + pd.offsets.MonthEnd(1)).date())}, ["date"])
    frame["id"] = frame.id.astype("int64")
    save_input("us_terminal_months", frame, {"source": "WRDS crsp.msf_v2.mthret and delisting flags",
        "currency": "USD", "return_type": "total", "identifier": "jkp_id",
        "note": "Actual source monthly observations for securities with missing JKP forward returns; not an assumed zero return"})


def lag_history(db):
    """Fetch only pre-entry stock/months absent from the saved screened panels."""
    features = json.loads((ROOT / "feature_names.json").read_text())
    for region in ("US", "EU"):
        output = INPUTS / f"lag_history_source_{region}.parquet"
        if output.exists():
            print(f"Reuse {output.name}", flush=True)
            continue
        keys = pd.read_csv(ROOT / f"outputs/run_logs/missing_lag_keys_{region}.csv", parse_dates=["eom"])
        parts = []
        for start in range(0, len(keys), 200):
            batch = keys.iloc[start:start+200]
            params = {"keys": tuple((int(r.id), r.eom.date()) for r in batch.itertuples())}
            fields = ",".join(["id", "eom", "ret_exc", *features])
            parts.append(cached_query(db, f"lag_history_{region}_{start:04d}",
                "select " + fields + " from contrib.global_factor where (id,eom) in %(keys)s order by id,eom",
                params, ["eom"]))
        frame = pd.concat(parts, ignore_index=True)
        frame["id"] = frame.id.astype("int64")
        save_input(f"lag_history_source_{region}", frame, {"source": "WRDS contrib.global_factor",
            "purpose": "Actual historical characteristics/current excess returns for lagged signals only; no forward targets downloaded",
            "currency": "USD", "requested_stock_months": len(keys), "returned_stock_months": len(frame),
            "original_saved_panels_unchanged": True})
        absent = pd.MultiIndex.from_frame(keys[["id", "eom"]]).difference(pd.MultiIndex.from_frame(frame[["id", "eom"]]))
        absent.to_frame(index=False).to_csv(ROOT / f"outputs/run_logs/unavailable_lag_keys_{region}.csv", index=False)
        print(f"{region}: requested {len(keys)}, found {len(frame)}, unavailable {len(absent)}", flush=True)


def daily_us(db, ids):
    if (INPUTS / "daily_US.parquet").exists():
        print("Reuse daily_US.parquet", flush=True)
        return
    parts = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start + 100]
        part = cached_query(db, f"daily_US_{start:04d}", """
            select permno as id, dlycaldt as date, dlyret as ret
            from crsp.dsf_v2
            where permno in %(ids)s and dlycaldt between %(start)s and %(end)s
            order by permno,dlycaldt
        """, {"ids": tuple(batch), "start": START, "end": END}, ["date"])
        parts.append(part)
    daily = pd.concat(parts, ignore_index=True).dropna(subset=["ret"])
    daily["id"] = daily.id.astype("int64")
    daily["ret"] = daily.ret.astype(float)
    if daily.duplicated(["id", "date"]).any() or (daily.ret < -1).any() or not np.isfinite(daily.ret).all():
        raise ValueError("Invalid or duplicate CRSP daily returns")
    absent = set(ids) - set(daily.id)
    if absent:
        raise ValueError(f"CRSP has no daily data for required ids: {sorted(absent)}")
    save_input("daily_US", daily, {"currency": "USD", "return_type": "total", "identifier": "jkp_id",
        "source": "WRDS crsp.dsf_v2.dlyret (CRSP CIZ total return, including its delisting treatment)",
        "first_date": str(daily.date.min().date()), "last_date": str(daily.date.max().date()),
        "required_security_count": len(ids), "rows": len(daily)})


def eu_mapping(db, ids):
    mapping = cached_query(db, "eu_security_mapping", """
        select distinct id,gvkey,iid from contrib.global_factor
        where id in %(ids)s and eom between %(start)s and %(end)s
        order by id,gvkey,iid
    """, {"ids": tuple(ids), "start": START, "end": END})
    if mapping[["id", "gvkey", "iid"]].isna().any().any() or mapping.id.duplicated().any():
        raise ValueError("Ambiguous or missing JKP to Compustat issue mapping; no mapping is guessed")
    mapping["id"] = mapping.id.astype("int64")
    if set(mapping.id) != set(ids):
        raise ValueError("Incomplete EU security mapping")
    return mapping


def daily_eu(db, ids):
    if (INPUTS / "daily_EU.parquet").exists():
        print("Reuse daily_EU.parquet", flush=True)
        return
    mapping = eu_mapping(db, ids)
    pieces = []
    remaining = []
    for start in range(0, len(mapping), 100):
        batch = mapping.iloc[start:start + 100]
        cached = CACHE / f"daily_EU_prices_{start:04d}.parquet"
        if cached.exists():
            print(f"Reuse {cached.name}", flush=True)
            pieces.append(pd.read_parquet(cached))
        else:
            remaining.append(batch)
    if remaining:
        wanted = pd.concat(remaining, ignore_index=True)
        key = hashlib.sha256(wanted.to_csv(index=False).encode()).hexdigest()[:12]
        completed = CACHE / f"EU_stream_{key}.json"
        if completed.exists():
            paths = json.loads(completed.read_text())
        else:
            # The table has a date/company index but no issue-pair index. A single indexed
            # scan plus a hash join avoids a full parallel table scan for each 100 issues.
            query = """
                with wanted as (
                    select * from unnest(%(gvkeys)s::text[], %(iids)s::text[]) as v(gvkey,iid)
                )
                select d.gvkey,d.iid,d.datadate as date,d.curcdd,d.prcstd,
                       d.prccd/nullif(d.qunit,0)/nullif(d.ajexdi,0)*d.trfd as ri_local
                from comp.g_secd d join wanted w on d.gvkey=w.gvkey and d.iid=w.iid
                where d.gvkey in %(companies)s and d.datadate between %(start)s and %(end)s
                      and d.prcstd in ('3','4','10')
            """
            params = {"gvkeys": wanted.gvkey.astype(str).tolist(), "iids": wanted.iid.astype(str).tolist(),
                      "companies": tuple(wanted.gvkey.astype(str).unique()), "start": START, "end": "2026-10-31"}
            # WRDS connections default to autocommit; PostgreSQL named cursors require a real transaction.
            db.connection.commit()
            db.connection = db.connection.execution_options(isolation_level="READ COMMITTED", stream_results=True)
            paths = []
            attempt = len(list(CACHE.glob(f"EU_stream_{key}_*.parquet")))
            with db.connection.begin():
                chunks = db.raw_sql(query, params=params, date_cols=["date"], return_iter=True, chunksize=250000)
                for index, part in enumerate(chunks):
                    name = f"EU_stream_{key}_{attempt + index:04d}.parquet"
                    part.to_parquet(CACHE / name, index=False)
                    paths.append(name)
                    print(f"Downloaded European stream chunk {index + 1}: {len(part):,} rows", flush=True)
            db.connection = db.connection.execution_options(isolation_level="AUTOCOMMIT", stream_results=False)
            completed.write_text(json.dumps(paths, indent=2))
        pieces.extend(pd.read_parquet(CACHE / name) for name in paths)
    prices = pd.concat(pieces, ignore_index=True).merge(mapping, on=["gvkey", "iid"], validate="many_to_one")
    prices["ri_local"] = pd.to_numeric(prices.ri_local)
    prices = prices[prices.ri_local.gt(0)].copy()
    if prices.duplicated(["id", "date"]).any():
        raise ValueError("Duplicate European prices require source-level resolution")
    currencies = sorted(set(prices.curcdd.dropna().astype(str)) | {"USD"})
    fx = cached_query(db, "fx_GBP_crosses", """
        select tocurd,datadate as date,exratd
        from comp.g_exrt_dly
        where fromcurd='GBP' and tocurd in %(currencies)s
              and datadate between %(start)s and %(end)s
        order by datadate,tocurd
    """, {"currencies": tuple(currencies), "start": "1994-10-01", "end": "2026-10-31"}, ["date"])
    if fx.duplicated(["tocurd", "date"]).any():
        raise ValueError("Duplicate FX quotes")
    quoted = fx.pivot(index="date", columns="tocurd", values="exratd").astype(float)
    # GBP is the base currency, so GBP/GBP is exactly one even if no self-quote is stored.
    quoted["GBP"] = 1.0
    calendar = pd.date_range("1994-10-01", "2026-10-31", freq="D")
    quoted = quoted.reindex(calendar).ffill()
    rates = quoted.rdiv(quoted["USD"], axis=0)  # USD per one unit of the local currency.
    rates["USD"] = 1.0
    prices["fx"] = np.nan
    for currency, index in prices.groupby("curcdd").groups.items():
        if currency not in rates:
            raise ValueError(f"No FX conversion for {currency}")
        prices.loc[index, "fx"] = prices.loc[index, "date"].map(rates[currency])
    if prices.fx.isna().any():
        raise ValueError(f"Missing FX on {int(prices.fx.isna().sum())} European stock-days; no backfill from future quotes")
    prices["ri"] = prices.ri_local * prices.fx
    prices = prices.sort_values(["id", "date"], ignore_index=True)
    prices["ret"] = prices.groupby("id").ri.pct_change(fill_method=None)
    prices["ret_local"] = prices.groupby("id").ri_local.pct_change(fill_method=None)
    prior_currency = prices.groupby("id").curcdd.shift()
    currency_change = prior_currency.notna() & prices.curcdd.ne(prior_currency)
    prices.loc[currency_change, "ret_local"] = prices.loc[currency_change, "ret"]
    # Match the JKP construction: inactive bankrupt/liquidated issues receive -30% on their last nonzero local-return day.
    security = cached_query(db, "eu_delisting_metadata", """
        select gvkey,iid,secstat,dlrsni from comp.g_security
        where (gvkey,iid) in %(pairs)s
    """, {"pairs": tuple(zip(mapping.gvkey.astype(str), mapping.iid.astype(str)))})
    if security.duplicated(["gvkey", "iid"]).any():
        raise ValueError("Ambiguous European delisting metadata")
    last_trade = prices[prices.ret_local.notna() & prices.ret_local.ne(0)].groupby("id").tail(1)
    delists = last_trade[["id", "date", "gvkey", "iid"]].merge(security, on=["gvkey", "iid"], how="left", validate="one_to_one")
    if delists.secstat.isna().any():
        raise ValueError("Missing security status for European issues")
    inactive = delists[delists.secstat.eq("I")]
    endings = inactive.set_index("id").date
    stop = prices.id.map(endings)
    prices = prices[stop.isna() | prices.date.le(stop)].copy()
    bad = inactive[inactive.dlrsni.isin(["02", "03"])][["id", "date"]]
    prices = prices.merge(bad.assign(delisting_adjustment=True), on=["id", "date"], how="left", validate="one_to_one")
    adjust = prices.delisting_adjustment.eq(True)
    prices.loc[adjust, "ret"] = (1.0 + prices.loc[adjust, "ret"]) * 0.7 - 1.0
    daily = prices.loc[prices.date.le(END) & prices.ret.notna(), ["id", "date", "ret"]].copy()
    daily["id"] = daily.id.astype("int64")
    if not np.isfinite(daily.ret).all() or (daily.ret < -1).any():
        raise ValueError("Invalid European daily total returns")
    if set(ids) - set(daily.id):
        raise ValueError("Some required European ids have no daily return observations")
    validate_eu_monthly(db, daily, ids)
    save_input("daily_EU", daily, {"currency": "USD", "return_type": "total", "identifier": "jkp_id",
        "source": "WRDS comp.g_secd prices/adjustment factors + comp.g_exrt_dly FX + comp.g_security delisting status",
        "construction_reference": JKP_CODE,
        "construction": "ri=(prccd/qunit)/ajexdi*trfd*USD_per_local; ret=ri/lag(ri)-1; JKP inactive-issue/delisting treatment",
        "FX": "GBP cross rates; past quotes carried forward, never backward",
        "required_security_count": len(ids), "rows": len(daily),
        "first_date": str(daily.date.min().date()), "last_date": str(daily.date.max().date()),
        "monthly_validation": "eu_monthly_validation.json"})
    if not (INPUTS / "eu_security_mapping.parquet").exists():
        save_input("eu_security_mapping", mapping)


def validate_eu_monthly(db, daily, ids):
    sample = cached_query(db, "eu_monthly_validation", """
        select id,eom,ret from contrib.global_factor
        where id in %(ids)s and eom in ('2019-01-31','2020-01-31','2020-03-31','2025-01-31')
    """, {"ids": tuple(ids)}, ["eom"])
    frame = daily.assign(eom=daily.date + pd.offsets.MonthEnd(0), gross=1 + daily.ret)
    actual = frame.groupby(["id", "eom"]).gross.prod().sub(1).rename("daily_compounded").reset_index()
    compare = sample.dropna(subset=["ret"]).merge(actual, on=["id", "eom"], validate="one_to_one")
    difference = abs(compare.ret.astype(float) - compare.daily_compounded)
    report = {"observations": len(compare), "median_absolute_error": float(difference.median()),
              "share_within_10bp": float(difference.le(0.001).mean()), "max_absolute_error": float(difference.max())}
    INPUTS.mkdir(parents=True, exist_ok=True)
    (INPUTS / "eu_monthly_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"EU daily/monthly validation: {report}", flush=True)
    if len(compare) < 100 or report["share_within_10bp"] < 0.95:
        # Preserve the candidate for diagnosis, but do not make it an accepted pipeline input.
        candidate = CACHE / "daily_EU_candidate.parquet"
        if not candidate.exists():
            daily.to_parquet(candidate, index=False)
        raise ValueError("European returns fail monthly reconciliation; candidate retained privately for diagnosis")


def fred():
    import io
    import urllib.request
    for series in ("USREC", "VIXCLS", "FEDFUNDS"):
        path = INPUTS / f"fred_{series}.parquet"
        if path.exists():
            print(f"Reuse {path.name}", flush=True)
            continue
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = pd.read_csv(io.BytesIO(response.read()))
        raw.columns = ["date", "value"]
        raw["date"] = pd.to_datetime(raw.date)
        raw["value"] = pd.to_numeric(raw.value, errors="coerce")
        raw = raw[raw.date.between(START, END)].dropna()
        if raw.empty:
            raise ValueError(f"No FRED observations: {series}")
        save_input(f"fred_{series}", raw, {"source": url, "series": series, "last_included_date": END})


def credentials():
    path = ROOT.parent / "configs/wrds_login.local.json"
    local = json.loads(path.read_text()) if path.exists() else {}
    username = os.environ.get("WRDS_USERNAME") or local.get("username")
    password = os.environ.get("WRDS_PASSWORD") or local.get("password")
    if not username or not password:
        raise RuntimeError("WRDS credentials unavailable. Set WRDS_USERNAME and WRDS_PASSWORD locally.")
    return username, password


def probe(db):
    columns = db.raw_sql("""
        select table_schema, table_name, column_name, data_type
        from information_schema.columns
        where (table_name ilike %(factor_pattern)s and table_schema = 'contrib')
           or (table_name in ('dsf_v2', 'g_secd', 'g_exrt_dly', 'exrt_dly', 'g_sec_history')
               and table_schema in ('crsp', 'comp', 'comp_global_daily', 'comp_global'))
        order by table_schema, table_name, ordinal_position
    """, params={"factor_pattern": "%global%factor%"})
    out = ROOT / "outputs/acquisition_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(columns.to_json(orient="records", indent=2))
    for (schema, table), group in columns.groupby(["table_schema", "table_name"]):
        names = list(group.column_name)
        print(f"{schema}.{table}: {len(names)} columns; " + ", ".join(names[:30]), flush=True)
    print(f"Saved table metadata: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["probe", "plan", "monthly", "terminal", "lags", "daily_US", "daily_EU", "fred", "all"])
    args = parser.parse_args()
    if args.step == "fred":
        fred()
        return 0
    import wrds
    username, password = credentials()
    os.environ.setdefault("PGCONNECT_TIMEOUT", "20")
    db = wrds.Connection(autoconnect=False, wrds_username=username, wrds_password=password)
    print("One WRDS connection attempt. Approve the Duo request if one appears.", flush=True)
    try:
        db._Connection__make_sa_engine_conn(raise_err=True)
        if args.step == "probe":
            probe(db)
        elif args.step == "plan":
            mapping = pd.read_parquet(CACHE / "eu_security_mapping.parquet").iloc[:100]
            pairs = tuple(zip(mapping.gvkey.astype(str), mapping.iid.astype(str)))
            for label, condition, values in (
                ("pair-filter", "(gvkey,iid) in %(pairs)s", {"pairs": pairs}),
                ("company-filter", "gvkey in %(keys)s", {"keys": tuple(mapping.gvkey.astype(str).unique())}),
            ):
                plan = db.raw_sql("EXPLAIN select gvkey,iid,datadate,prccd,qunit,ajexdi,trfd from comp.g_secd where "
                    + condition + " and datadate between '1994-11-01' and '2026-10-31'", params=values)
                print(label, flush=True)
                for value in plan.iloc[:, 0]:
                    print(str(value)[:200], flush=True)
            indexes = db.raw_sql("select indexdef from pg_indexes where schemaname='comp_global_daily' and tablename='g_secd'", params={})
            for definition in indexes.indexdef:
                print(definition, flush=True)
        elif args.step == "monthly":
            monthly_gaps(db)
        elif args.step == "terminal":
            terminal_events(db)
        elif args.step == "lags":
            lag_history(db)
        else:
            ids = required_ids()
            if args.step in ("all", "daily_US"):
                daily_us(db, ids["US"])
            if args.step in ("all", "daily_EU"):
                daily_eu(db, ids["EU"])
            if args.step == "all":
                monthly_gaps(db)
    except Exception as error:
        message = str(error).replace(password, "[redacted]")
        # Limit diagnostics; SQLAlchemy can include connection details in long tracebacks.
        print(f"WRDS acquisition stopped ({type(error).__name__}): {message[:600]}", flush=True)
        return 2
    finally:
        db.close()
    if args.step == "all":
        fred()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
