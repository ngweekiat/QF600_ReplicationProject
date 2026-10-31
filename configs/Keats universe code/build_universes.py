"""
Build the three 80-stock universes for the QF600 Portfolio-ML replication
from JKP Global Factor Data on WRDS (contrib.global_factor):

    us_large : US market-cap ranks 1-80      (NYSE + Nasdaq)
    us_mid   : US market-cap ranks 461-540   (centred on rank ~500)
    eu_large : developed-Europe ranks 1-80   (USD market cap, 15 countries)

Rules (same for every universe, so only size/region changes):
  * December reconstitution; membership applies to formation months Dec(y)..Nov(y+1),
    i.e. return months Jan(y+1)..Dec(y+1).
  * Buffer: enter if inside the band shrunk by 10 ranks, stay if inside the band
    widened by 20 ranks (us_large: entry <= 70, retention <= 100, as in the report;
    us_mid uses a wider 60-rank retention pad because mid-cap ranks move more, and
    admits new entrants only up to the 80 target).
    Remaining slots are filled with the stocks closest to the band centre.
    Retained members are not trimmed, so the count can exceed 80.
  * A member that fails the screens during the year drops out and cannot re-enter
    before the next December.
  * Screens: JKP common / primary_sec / obs_main / exch_main flags, positive market cap
    and dollar volume, >= 25 of the 50 characteristics, industry code present,
    >= 12 consecutive months of history. US only: CRSP share code 10/11/12 and
    exchange code 1 (NYSE) or 3 (Nasdaq).

Outputs (in --out):
  membership_<universe>.csv     who is in each year, with name, rank, size, liquidity
  panel_<universe>.parquet      monthly member rows: 50 characteristics (raw + ranked),
                                me, dolvol_126d, ret_exc_lead1m, period label
  screened_<region>.parquet     full screened regional panel (for the return model and
                                for ranking characteristics over the broad universe)
  summary_by_universe.csv       per-December medians of me / dolvol / Amihud, coverage,
                                churn, sector concentration (fills the essay placeholders)
  periods.csv                   train / validation / test calendar

Usage:
  Open this file in VS Code, select Python 3.13, then click Run Python File.
  Set WRDS_USERNAME and WRDS_PASSWORD in your environment; approve the WRDS Duo push.
  Login is attempted once without automatic retries.

  pip install wrds pandas pyarrow
  python build_universes.py --wrds-username YOUR_ID --out data/universes

Daily returns for the covariance model are NOT pulled here. US: CRSP crsp.dsf_v2 (already
in the repo pipeline). Europe: Compustat Global comp.g_secd + comp.g_exrt_dly (needs a
separate build).
"""
from __future__ import annotations

import argparse
import getpass
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- config
# Read credentials from the environment; never store passwords in source code.
WRDS_USERNAME = os.environ.get("WRDS_USERNAME", "")
WRDS_PASSWORD = os.environ.get("WRDS_PASSWORD", "")

CHARS = [
    # Value (6)
    "be_me", "sale_me", "ebitda_mev", "ni_me", "fcf_me", "div12m_me",
    # Momentum (5)
    "ret_12_1", "ret_6_1", "prc_highprc_252d", "ret_9_1", "ret_3_1",
    # Quality (6)
    "qmj_prof", "qmj_safety", "gp_at", "op_at", "cop_at", "at_turnover",
    # Investment (5)
    "at_gr1", "ppeinv_gr1a", "noa_gr1a", "capx_gr1", "emp_gr1",
    # Low risk (5)
    "ivol_capm_252d", "beta_60m", "rvol_21d", "betadown_252d", "zero_trades_126d",
    # Short-term reversal (3)
    "ret_1_0", "rskew_21d", "iskew_capm_21d",
    # Profitability (4)
    "ocf_at", "ni_be", "ebit_sale", "ope_be",
    # Profit growth (4)
    "ocf_at_chg1", "sale_emp_gr1", "tax_gr1a", "ret_12_7",
    # Low leverage (3)
    "cash_at", "netdebt_me", "tangibility",
    # Accruals (3)
    "oaccruals_at", "taccruals_at", "cowc_gr1a",
    # Size (3)
    "market_equity", "ami_126d", "dolvol_126d",
    # Debt issuance (2)
    "noa_at", "fnl_gr1a",
    # Seasonality (1)
    "seas_11_15an",
]
MIN_CHARS = 25
MIN_HISTORY = 12

ID_COLS = [
    "id", "eom", "excntry", "gvkey", "permno", "common", "primary_sec", "obs_main",
    "exch_main", "crsp_shrcd", "crsp_exchcd", "me", "ret_exc_lead1m", "size_grp",
    "ff49", "gics", "sic",
]

EUROPE = ["AUT", "BEL", "CHE", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR",
          "IRL", "ITA", "NLD", "NOR", "PRT", "SWE"]

UNIVERSES = {
    "us_large": dict(region="US", lo=1, hi=80),
    # Stocks near rank 500 are densely packed in market cap, so ranks move a lot each
    # year; a wider retention band (401-600) keeps churn closer to the large-cap level.
    "us_mid":   dict(region="US", lo=461, hi=540, keep_pad=60, cap_new=True),
    "eu_large": dict(region="EU", lo=1, hi=80),
}
TARGET = 80
BUFFER_IN, BUFFER_OUT = 10, 20

# Labelled by RETURN month (formation month + 1).
PERIODS = [
    ("warmup",     "1995-01", "2004-12"),  # covariance / lag history only
    ("train",      "2005-01", "2014-12"),
    ("validation", "2015-01", "2019-12"),
    ("test",       "2020-01", "2025-12"),
]
DATA_START, DATA_END = "1994-01-01", "2025-12-31"


# --------------------------------------------------------------------------- WRDS pull
def pull_jkp(db, countries: list[str], start: str, end: str) -> pd.DataFrame:
    avail = set(db.describe_table("contrib", "global_factor")["name"].str.lower())
    missing = [c for c in ID_COLS + CHARS if c not in avail]
    if missing:
        print(f"[warn] columns not in contrib.global_factor, skipped: {missing}")
    cols = [c for c in dict.fromkeys(ID_COLS + CHARS) if c in avail]
    clist = ", ".join(f"'{c}'" for c in countries)
    frames = []
    for y in range(int(start[:4]), int(end[:4]) + 1):
        q = (f"select {', '.join(cols)} from contrib.global_factor "
             f"where excntry in ({clist}) and eom between '{y}-01-01' and '{y}-12-31'")
        frames.append(db.raw_sql(q, date_cols=["eom"]))
        print(f"  pulled {countries[0] if len(countries) == 1 else 'EU'} {y}: {len(frames[-1]):,} rows")
    return pd.concat(frames, ignore_index=True)


def pull_names(db) -> pd.DataFrame:
    na = db.raw_sql("select gvkey, conm from comp.company")
    gl = db.raw_sql("select gvkey, conm from comp.g_company")
    out = pd.concat([na, gl]).drop_duplicates("gvkey")
    out["gvkey"] = out["gvkey"].astype(str).str.zfill(6)
    return out


# --------------------------------------------------------------------------- screens
def _consecutive_months(df: pd.DataFrame) -> pd.Series:
    m = df["eom"].dt.year * 12 + df["eom"].dt.month
    new_run = m.groupby(df["id"]).diff().ne(1)
    run_id = new_run.groupby(df["id"]).cumsum()
    return df.groupby([df["id"], run_id]).cumcount() + 1


def screen(df: pd.DataFrame, region: str) -> pd.DataFrame:
    df = df.sort_values(["id", "eom"]).copy()
    chars = [c for c in CHARS if c in df.columns]
    m = pd.Series(True, index=df.index)
    for flag in ["common", "primary_sec", "obs_main", "exch_main"]:
        if flag in df.columns:
            m &= df[flag].eq(1)
    m &= df["me"].gt(0) & df["dolvol_126d"].gt(0)
    m &= df[chars].notna().sum(axis=1).ge(MIN_CHARS)
    ind = [c for c in ["sic", "ff49", "gics"] if c in df.columns]
    m &= df[ind].notna().any(axis=1)
    if region == "US":
        for col, ok in [("crsp_shrcd", [10, 11, 12]), ("crsp_exchcd", [1, 3])]:
            if col in df.columns:
                m &= df[col].isin(ok)
            else:
                print(f"[warn] {col} missing: US screen on {col} not applied")
    df = df[m].copy()
    df["hist_months"] = _consecutive_months(df)
    df = df[df["hist_months"] >= MIN_HISTORY]
    if region == "EU" and "gvkey" in df.columns:  # one listing per company per month
        df = df.sort_values("me", ascending=False).drop_duplicates(["gvkey", "eom"])
    return df.sort_values(["eom", "id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- selection
def bands(lo: int, hi: int, keep_pad: int = BUFFER_OUT):
    entry = (1 if lo == 1 else lo + BUFFER_IN, hi - BUFFER_IN)
    keep = (max(1, lo - keep_pad), hi + keep_pad)
    return entry, keep


def select_members(scr: pd.DataFrame, lo: int, hi: int, keep_pad: int = BUFFER_OUT,
                   cap_new: bool = False) -> pd.DataFrame:
    """Annual December reconstitution with entry/retention buffers."""
    dec = scr[scr["eom"].dt.month == 12].copy()
    dec["mcap_rank"] = dec.groupby("eom")["me"].rank(ascending=False, method="first")
    (e_lo, e_hi), (k_lo, k_hi) = bands(lo, hi, keep_pad)
    centre = (lo + hi) / 2
    rows, prev = [], set()
    for eom, g in dec.groupby("eom"):
        r = g.set_index("id")["mcap_rank"]
        kept = {i for i in prev if i in r.index and k_lo <= r[i] <= k_hi}
        new_r = r[(r >= e_lo) & (r <= e_hi)].drop(list(kept), errors="ignore")
        if cap_new:  # only admit enough entrants (closest to centre) to reach the target
            new_r = new_r.iloc[np.argsort(np.abs(new_r.to_numpy() - centre), kind="stable")]
            new_r = new_r.iloc[: max(0, TARGET - len(kept))]
        new = set(new_r.index)
        status = {i: "retained" for i in kept} | {i: "new" for i in new}
        if len(status) < TARGET:
            cand = r.drop(list(status), errors="ignore")
            cand = cand.iloc[np.argsort(np.abs(cand.to_numpy() - centre), kind="stable")]
            for i in cand.index[: TARGET - len(status)]:
                status[i] = "fill"
        sel = g[g["id"].isin(status)].copy()
        sel["status"] = sel["id"].map(status)
        sel["form_year"] = eom.year
        rows.append(sel)
        prev = set(status)
    return pd.concat(rows, ignore_index=True)


def member_panel(scr: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Monthly rows for members, Dec(y)..Nov(y+1); drop-outs stay out until next Dec."""
    scr = scr.copy()
    scr["form_year"] = np.where(scr["eom"].dt.month == 12, scr["eom"].dt.year, scr["eom"].dt.year - 1)
    p = scr.merge(members[["id", "form_year", "status", "mcap_rank"]], on=["id", "form_year"])
    p = p.sort_values(["id", "form_year", "eom"])
    m = p["eom"].dt.year * 12 + p["eom"].dt.month
    gap = m.groupby([p["id"], p["form_year"]]).diff().fillna(1).ne(1)
    broken = gap.groupby([p["id"], p["form_year"]]).cummax()
    return p[~broken].reset_index(drop=True)


# --------------------------------------------------------------------------- extras
def add_ranks(panel: pd.DataFrame, scr: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks computed over the full screened regional universe, missing -> 0.5."""
    chars = [c for c in CHARS if c in scr.columns]
    rk = scr[["id", "eom"] + chars].copy()
    rk[chars] = rk.groupby("eom")[chars].rank(pct=True).fillna(0.5)
    rk = rk.rename(columns={c: f"{c}_rank" for c in chars})
    return panel.merge(rk, on=["id", "eom"], how="left")


def label_period(eom: pd.Series) -> pd.Series:
    ret_month = (eom + pd.offsets.MonthEnd(1)).dt.strftime("%Y-%m")
    out = pd.Series("pre", index=eom.index)
    for name, a, b in PERIODS:
        out[(ret_month >= a) & (ret_month <= b)] = name
    out[ret_month > PERIODS[-1][2]] = "post"
    return out


def summarise(name: str, members: pd.DataFrame) -> pd.DataFrame:
    out = []
    prev = set()
    chars = [c for c in CHARS if c in members.columns]
    for y, g in members.groupby("form_year"):
        ids = set(g["id"])
        sector = g["ff49"] if "ff49" in g else g.get("gics")
        shares = sector.value_counts(normalize=True) if sector is not None else pd.Series(dtype=float)
        out.append(dict(
            universe=name, form_year=y, n=len(ids),
            median_me_usd_m=g["me"].median(),
            median_dolvol_126d=g["dolvol_126d"].median(),
            median_ami_126d=g["ami_126d"].median() if "ami_126d" in g else np.nan,
            iqr_ami_126d=(g["ami_126d"].quantile(.75) - g["ami_126d"].quantile(.25)) if "ami_126d" in g else np.nan,
            char_coverage=g[chars].notna().mean(axis=1).mean(),
            churn=len(ids - prev) / len(ids) if prev else np.nan,
            sector_hhi=float((shares ** 2).sum()),
            n_countries=g["excntry"].nunique(),
        ))
        prev = ids
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- driver
def build(us: pd.DataFrame, eu: pd.DataFrame, names: pd.DataFrame | None, out: str, csv: bool):
    os.makedirs(out, exist_ok=True)
    screened = {"US": screen(us, "US"), "EU": screen(eu, "EU")}
    for reg, s in screened.items():
        s.to_parquet(f"{out}/screened_{reg}.parquet", index=False)
        print(f"screened {reg}: {s['id'].nunique():,} stocks, {len(s):,} stock-months")

    summaries = []
    for uname, cfg in UNIVERSES.items():
        scr = screened[cfg["region"]]
        mem = select_members(scr, cfg["lo"], cfg["hi"], cfg.get("keep_pad", BUFFER_OUT),
                             cfg.get("cap_new", False))
        if names is not None and "gvkey" in mem:
            mem["gvkey"] = mem["gvkey"].astype(str).str.zfill(6)
            mem = mem.merge(names, on="gvkey", how="left")
        keep = [c for c in ["form_year", "eom", "id", "gvkey", "permno", "conm", "excntry", "mcap_rank",
                            "status", "me", "dolvol_126d", "ami_126d", "ff49", "gics"] if c in mem]
        mem[keep].sort_values(["form_year", "mcap_rank"]).to_csv(f"{out}/membership_{uname}.csv", index=False)

        panel = add_ranks(member_panel(scr, mem), scr)
        panel["period"] = label_period(panel["eom"])
        panel.to_parquet(f"{out}/panel_{uname}.parquet", index=False)
        if csv:
            panel.to_csv(f"{out}/panel_{uname}.csv", index=False)
        summaries.append(summarise(uname, mem))
        n = mem.groupby("form_year")["id"].nunique()
        print(f"{uname}: {len(n)} reconstitutions, members/yr min {n.min()} median {int(n.median())} max {n.max()}")

    summ = pd.concat(summaries, ignore_index=True)
    summ.to_csv(f"{out}/summary_by_universe.csv", index=False)
    pd.DataFrame(PERIODS, columns=["period", "first_return_month", "last_return_month"]).to_csv(
        f"{out}/periods.csv", index=False)

    # units check: Apple-scale stocks trade ~$5-15B/day
    top = summ[(summ.universe == "us_large") & (summ.form_year == 2019)]
    if len(top):
        v = top["median_dolvol_126d"].iloc[0]
        unit = "USD millions" if v < 1e5 else "USD"
        print(f"[check] us_large Dec-2019 median dolvol_126d = {v:,.1f} -> looks like {unit}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrds-username", default=WRDS_USERNAME)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data" / "universes"))
    ap.add_argument("--csv", action="store_true", help="also write member panels as CSV")
    a = ap.parse_args()
    username = a.wrds_username.strip()
    if not username:
        ap.error("WRDS username must not be empty")
    password = WRDS_PASSWORD if WRDS_PASSWORD and username == WRDS_USERNAME else getpass.getpass(f"WRDS password for {username} (typing is hidden): ")
    if not password:
        ap.error("WRDS password must not be empty")

    import wrds
    print("One WRDS login attempt. Open Duo Mobile and approve the push.", flush=True)
    # Do not use Connection's default authentication retry/prompt flow.
    db = wrds.Connection(autoconnect=False, wrds_username=username, wrds_password=password)
    try:
        db._Connection__make_sa_engine_conn(raise_err=True)
    except Exception:
        print("WRDS login failed. No retry was made. Resolve the login/Duo issue before running again.", file=sys.stderr)
        raise

    # Reuse one authenticated connection for every query, then close it.
    try:
        # Initialise the metadata inspector on the existing connection; no new login.
        db.load_library_list()
        us = pull_jkp(db, ["USA"], DATA_START, DATA_END)
        eu = pull_jkp(db, EUROPE, DATA_START, DATA_END)
        names = pull_names(db)
    finally:
        db.close()
    build(us, eu, names, a.out, a.csv)
    print(f"Universe files saved to: {Path(a.out).resolve()}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EOFError:
        print("Keyboard input is unavailable. Use VS Code Run Python File in Terminal; do not pipe this code into Python.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Stopped. No automatic login retry will be made.", file=sys.stderr)
        sys.exit(130)
