"""ret_ld1 at month t must be the stock's return over month t+1 — checked against raw CRSP daily data.

Needs data/raw from `scripts/00_download.py pull`; skipped otherwise.

Tolerance: JKP builds ret_exc from CRSP monthly returns minus Ken French RF, which French publishes
rounded to 0.01%; compounded CRSP v2 daily returns match JKP to a median of 3e-5 and a 99th
percentile of 8e-4 across all top-500 stock-months, so we use 1e-3.
"""

import numpy as np
import pandas as pd
import pytest

from jkmp.config import load_config
from jkmp.costs import wealth_path
from jkmp.data.prepare import add_timing_columns, market_total_returns, risk_free_monthly
from jkmp.data.wrds_pull import RAW_FILES

cfg = load_config()
RAW = cfg.paths.raw
N_SAMPLES = 10
ATOL = 1e-3

pytestmark = pytest.mark.skipif(
    not all((RAW / RAW_FILES[k]).exists() for k in ("jkp", "daily", "ff_monthly")),
    reason="raw data not downloaded (run scripts/00_download.py pull)",
)


@pytest.fixture(scope="module")
def sample():
    """10 random stock-months (fixed seed) with complete CRSP daily data in month t+1."""
    jkp = pd.read_parquet(RAW / RAW_FILES["jkp"], columns=["id", "eom", "ret_exc", "ret_exc_lead1m"])
    daily = pd.read_parquet(RAW / RAW_FILES["daily"])
    ff = pd.read_parquet(RAW / RAW_FILES["ff_monthly"])

    daily["eom"] = daily["date"] + pd.offsets.MonthEnd(0)
    monthly = daily.groupby(["permno", "eom"])["ret"].agg(
        crsp_tr=lambda r: np.prod(1.0 + r) - 1.0, n_missing=lambda r: r.isna().sum()
    ).reset_index()
    complete = monthly[monthly["n_missing"] == 0]

    # Candidate rows: the stock has complete daily data in both month t and month t+1.
    cand = jkp.dropna(subset=["ret_exc_lead1m"]).assign(eom_next=lambda d: d["eom"] + pd.offsets.MonthEnd(1))
    cand = cand.merge(complete.rename(columns={"permno": "id", "eom": "eom_next", "crsp_tr": "crsp_tr_next"}),
                      on=["id", "eom_next"])
    cand = cand.merge(complete.rename(columns={"permno": "id", "crsp_tr": "crsp_tr_same"}), on=["id", "eom"])
    rows = cand.sample(N_SAMPLES, random_state=cfg.seed)

    wealth = wealth_path(cfg.investor.wealth, cfg.investor.wealth_end_date,
                         market_total_returns(ff, start=pd.Timestamp(cfg.dates.warmup_start)))
    rf = risk_free_monthly(ff)
    timed = add_timing_columns(rows[["id", "eom", "ret_exc", "ret_exc_lead1m"]], rf, wealth)
    rows = rows.assign(ret_ld1=timed["ret_ld1"].to_numpy(), tr_ld1=timed["tr_ld1"].to_numpy(),
                       rf_next=rows["eom_next"].map(rf).to_numpy())
    return rows


def test_ret_ld1_is_crsp_return_in_month_t_plus_1(sample):
    expected = sample["crsp_tr_next"] - sample["rf_next"]  # CRSP total return over t+1, made excess
    np.testing.assert_allclose(sample["ret_ld1"], expected, atol=ATOL, rtol=0)


def test_tr_ld1_is_crsp_total_return_in_month_t_plus_1(sample):
    np.testing.assert_allclose(sample["tr_ld1"], sample["crsp_tr_next"], atol=ATOL, rtol=0)


def test_ret_ld1_is_not_the_month_t_return(sample):
    """Guards against an off-by-one: ret_ld1 should generally differ from the month-t return."""
    differs = (sample["tr_ld1"] - sample["crsp_tr_same"]).abs() > ATOL
    assert differs.sum() >= N_SAMPLES - 2
