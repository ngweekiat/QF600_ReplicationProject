# CLAUDE.md

A Python replication of **Jensen, Kelly, Malamud & Pedersen (2026), "Machine Learning and the
Implementable Efficient Frontier", *Review of Financial Studies* 39: 3035–3078**, for the SMU QF600
(Asset Pricing) group project. The authors' R code is in `reference/jkmp_r/`.

The paper's core method, **Portfolio-ML**, learns the "aim" portfolio directly from characteristics
using random Fourier features and a utility objective that includes trading costs. It is compared
against Markowitz-ML, Factor-ML, Static-ML\* and simple benchmarks on returns *net of trading costs*.

## Status

- Stage 0 (raw-data pull) done: `src/jkmp/data/wrds_pull.py`, `scripts/00_download.py`.
- Stage 1 (screened panel) done: `src/jkmp/data/prepare.py`, `src/jkmp/data/universe.py`,
  `src/jkmp/features.py`, `src/jkmp/costs.py`, `scripts/01_prepare.py`; timing verified against CRSP
  daily in `tests/test_timing.py`.
- Stage 2 (covariance) done: `src/jkmp/covariance.py`, `scripts/02_covariance.py`. Σ for 2003-01 to 2025-11 (full 10-year window from 1993 daily data).
- Portfolio machinery done: `src/jkmp/portfolio/{accounting,trading_speed,benchmarks}.py`,
  `src/jkmp/evaluation/metrics.py`, `scripts/04_benchmarks.py` (Market, 1/N, Minimum variance →
  `outputs/benchmarks/`).
- Stage 3 (return model) done: `src/jkmp/models/{random_features,ridge,return_model}.py`,
  `scripts/03_return_model.py` → `pred_ld1` for return years 2015–2025. Markowitz-ML and Factor-ML in
  `src/jkmp/portfolio/ml_benchmarks.py`, run by `scripts/04_benchmarks.py`.
- Static-ML and Static-ML\* done: `src/jkmp/portfolio/static_ml.py`, `scripts/05_static_ml.py`
  (validation from return year 2015; (k, u, g) chosen each December for the next return year) →
  `outputs/static_ml/`, combined test-window Table 2 in `outputs/table2_test.csv`.
- Portfolio-ML inputs done: `jkmp.portfolio.portfolio_ml.build_moments` (port of `pfml_input_fun`),
  `scripts/06_pfml_inputs.py` → `data/processed/pfml/`. `portfolio_ml.use_rff: false` = raw characteristics
  (linear extension, `configs/experiments/linear_pfml.yaml`).
- Portfolio-ML done: `portfolio_ml.{search_coef, hp_reals, select_hypers, aim_portfolios, pfml_weights,
  run_portfolio_ml}` (ports of `pfml_search_coef`, `pfml_hp_reals_fun`, `pfml_aims_fun`, `pfml_w`, `pfml_implement`),
  `scripts/07_portfolio_ml.py` → `outputs/portfolio_ml/{rff,raw}/`. Recursive methods share
  `portfolio/recursion.py` (`month_inputs`, `recursive_weights`).
- Frontier grid done: `portfolio/grids.py`, `evaluation/frontier.py`, `scripts/08_grids.py` — Static-ML\* and
  Portfolio-ML re-tuned at every γ × wealth (joblib), Fig. 1A, Fig. 1B, capacity chart → `outputs/grids/`, `outputs/figures/`.
- Feature importance done: `evaluation/feature_importance.py` (ports of `pfml_cf_fun`, `tpf_cf_fun`),
  `scripts/09_feature_importance.py` → Fig. 5. Fitted return models saved by scripts/03 (`return_model/models.npz`).
- Regimes done (extension): `evaluation/regimes.py`, `scripts/10_regimes.py` — FRED USREC/VIX/FEDFUNDS labels,
  cumulative utility with shaded episodes, Table 2 stats by regime, VIX-scaled re-costing → `outputs/regimes/`.
- **Not yet:** Rank-ML, Table 3, Fig. 2.

New portfolio methods should produce target weights `id, eom, w` and reuse
`accounting.add_start_weights` (targets independent of inherited weights) or
`recursion.recursive_weights` (targets that depend on the inherited weights), then `accounting.portfolio_returns` and
`evaluation.metrics.table2`.

## Read these first

| File | What it gives you |
| --- | --- |
| `docs/QF600 Replication Plan.md` | Stage-by-stage pipeline, the formulas, the targets to replicate (Table 2, Figs 1A, 2, 5, Table 3), silent-bug list |
| `docs/JKMP R Code Map.md` | Where every paper equation lives in the R code; code-vs-paper details |
| `docs/deviations.md` | Every difference from the paper or R code, plus open decisions |
| `configs/base.yaml` | The design, as numbers. Authoritative over both docs above |

The plan was written for an earlier, larger design (NYSE above-median, 1952–2020). **Where it
conflicts with `configs/base.yaml` or the design below, the config wins.** The plan's method and
formulas still apply.

## Design (summary of `configs/base.yaml`)

- **Universe:** top 80 US common stocks (NYSE + Nasdaq) by market cap; 70/100 rank buffer,
  reconstituted each December.
- **Dates:** warm-up from 1993 (a full 10-year covariance window before 2005); train 2005–2014; validate 2015–2019; test 2020–2025.
- **Features:** 18 JKP characteristics in 6 themes (value, momentum, quality, investment, low risk,
  short-term reversal), percentile-ranked monthly, missing values set to 0.5.
- **Covariance:** Barra-style, Σ = X F Xᵀ + diag(idiosyncratic variance), with 12 industries + 6 theme
  factors; factor returns estimated on the **top 500** stocks; Σ is **monthly**.
- **Trading cost:** Λ = 0.2 / `dolvol_126d` (diagonal); dollar cost = (w/2)·Δπᵀ Λ Δπ.
- **Investor:** γ = 10 baseline; \$10B baseline wealth (end-2025); wealth grid \$1B–\$250B;
  γ grid {1, 5, 10, 20, 100}.
- **In scope:** Portfolio-ML, Static-ML(\*), Markowitz-ML, Factor-ML, Rank-ML, Market, 1/N, minimum
  variance. **Out of scope:** Multiperiod-ML, short-selling fees, size groups, simulations.

## Layout

```text
configs/base.yaml     all design parameters (no magic numbers in code)
src/jkmp/             all logic, importable and testable
scripts/              thin runners: parse args -> load_config -> call src/jkmp -> write outputs
tests/                pytest
docs/                 plan, R code map, deviations
reference/jkmp_r/     authors' R code (read-only)
notebooks/            exploration only; nothing a result depends on
data/raw, data/processed, outputs/   git-ignored
```

## Rules

1. **Logic lives in `src/jkmp/`.** Scripts in `scripts/` only wire config → library → files; no
   formulas in scripts or notebooks.
2. **All parameters come from `configs/base.yaml`** via `jkmp.config.load_config()`. Experiments live in
   `configs/experiments/*.yaml`, start with `base: ../base.yaml` and list only the keys they change. Vary runs with
   overrides (`load_config(overrides=["investor.gamma=5"])`), not by editing code. New parameters go
   into the YAML *and* the matching dataclass in `src/jkmp/config.py`.
3. **Never use `np.linalg.inv`.** Use `np.linalg.solve`, `scipy.linalg.solve` / `cho_solve`, or
   `lstsq`. For diagonal Λ, divide elementwise.
4. **Where the paper is ambiguous, follow the R code** in `reference/jkmp_r/`, and cite the R
   file and function in a comment where the port happens.
5. **Every deviation from the paper or the R code goes in `docs/deviations.md`**, in the same
   change that introduces it.
6. Do not edit `reference/jkmp_r/`.

## Conventions that prevent silent bugs

- **Timing:** `eom` = month-end when the portfolio is formed. `ret_ld1` = excess return over the
  next month (what the portfolio earns). `tr_ld1` = next month's total return (drifts weights).
  `tr_ld0` = total return over the month just ended (used in g_t). `mu_ld0`/`mu_ld1` = the same for
  the market. Never use information dated after `eom` to form a portfolio at `eom`.
- **Portfolio growth:** g_t = (1 + tr_ld0) / (1 + mu_ld0); inherited weight = π_{t−1}·(1 + tr)/(1 + mu).
- **Units:** returns and Σ are monthly; annualise only in reporting (mean × 12, sd × √12).
- **Tuning has no look-ahead:** hyperparameters and β used in year y are estimated on data ending
  before year y. `tests/test_models.py` checks this by scrambling future returns.
- **Years and windows refer to the month the return is realised.** "Year y" = formation months
  Dec y−1 .. Nov y (R `data_split`); the test window (returns 2020-01..2025-12) = formation
  2019-12..2025-11.
- Use `float64` throughout.
- **Cross-check** each ported function against its R counterpart (see the Code Map) on a small case.

## Commands

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"   # from the project root
.venv/bin/pytest                                   # run tests
.venv/bin/python scripts/00_download.py explore    # list/check WRDS tables
.venv/bin/python scripts/00_download.py pull       # raw data -> data/raw/ (existing files reused; --force to redo)
.venv/bin/python scripts/01_prepare.py             # panel, wealth, universe -> data/processed/
.venv/bin/python scripts/02_covariance.py          # Σ parts -> data/processed/covariance/
.venv/bin/python scripts/03_return_model.py        # pred_ld1 -> data/processed/predictions.parquet
.venv/bin/python scripts/04_benchmarks.py          # Market, 1/N, MinVar, Markowitz-ML, Factor-ML -> outputs/benchmarks/
.venv/bin/python scripts/05_static_ml.py           # Static-ML(*) -> outputs/static_ml/, outputs/table2_test.csv (needs 04)
.venv/bin/python scripts/06_pfml_inputs.py          # PF-ML r̃, Σ̃, signals -> data/processed/pfml/
.venv/bin/python scripts/07_portfolio_ml.py         # β, (g, p, λ), aims, weights -> outputs/portfolio_ml/, table2_test.csv
# Linear Portfolio-ML: same scripts (06, 07, 08) with --config configs/experiments/linear_pfml.yaml
.venv/bin/python scripts/08_grids.py [--force] [--plots-only]  # γ × wealth frontier, Fig 1A/1B, capacity (needs 01–04)
.venv/bin/python scripts/09_feature_importance.py  # theme permutations, Fig 5 (needs 03, 07)
.venv/bin/python scripts/10_regimes.py [--force]   # FRED regimes, cumulative utility, VIX re-costing (needs 04, 05, 07)
```

**Python 3.11–3.13 only.** `wrds` pins pandas < 2.3, which has no Python 3.14 wheels; a source
build on 3.14 segfaults against numpy 2.5. Versions are pinned in `pyproject.toml`.

**WRDS credentials:** optional `WRDS_USERNAME` sets the username. The installed WRDS PostgreSQL
client uses a saved password or a secure interactive prompt, not `WRDS_API_KEY`. On Windows its
password file is `%APPDATA%/postgresql/pgpass.conf`; on Unix it is `~/.pgpass`.
At the user's request, automatic login also supports the Git-ignored local file
`configs/wrds_login.local.json`; its contents must not appear in logs or tracked source.
Never write credentials into code, config, notebooks or logs. See `docs/Downloading data.md`.

## Raw data (`data/raw/`, from `scripts/00_download.py pull`)

| File | Contents |
| --- | --- |
| `jkp_usa.parquet` | JKP `contrib.global_factor`, USA, 1993-01 to 2025-12, exchcd 1/3, shrcd 10–12, `common = primary_sec = obs_main = 1`; ids, `me` (**\$ millions**), `dolvol_126d` (**\$**), `ret_exc`, `ret_exc_lead1m`, 18 features |
| `top500_by_month.parquet` | Top 500 by `me` each month (`eom, id, me, me_rank`) — the covariance factor universe |
| `jkp_usa_top500.csv` | Deliverable CSV: JKP rows for the top-500 stocks |
| `crsp_daily.parquet` | `crsp.dsf_v2` daily **total** returns (`permno, date, ret`), full 1993–2025 history of every stock ever in the top 500 |
| `ff_factors_monthly.parquet`, `ff_factors_daily.parquet` | Ken French Mkt-RF, SMB, HML, RF in decimals |
| `factor_details.parquet`, `cluster_labels.parquet` (+ originals) | JKP feature names, sign directions, themes |
| `row_counts_by_month.csv` | Per-month row counts across the files above |
| `fred_USREC.parquet`, `fred_VIXCLS.parquet`, `fred_FEDFUNDS.parquet` | FRED public CSVs (`date, value`), downloaded by `scripts/10_regimes.py` |

## Processed data (`data/processed/`, from `scripts/01_prepare.py`)

| File | Contents |
| --- | --- |
| `panel.parquet` | Screened stock-months 1993-01 to 2025-11: timing columns, `g` (drift), `dolvol`, `lambda`, `ff12`, ranked features (in [0, 1]), `theme_*` scores (signed average ranks, *not yet* standardised), `valid_data` (all screens + 12-month history; from 1994-01), `in_universe` (top-80 buffered universe; 1995-01 to 2025-11, 75–86 stocks a month), `recon_eom` |
| `wealth.parquet` | `eom, wealth, mu_ld1`: \$10B at end-2025, rolled back with the market total return |
| `universe_membership.parquet` | One row per December and member: `me_rank`, `status` (kept / added / filled), `last_eom`, `exit` |
| `universe_stats.csv` | Per membership year: size, kept / added / filled, dropped, turnover, early exits |
| `screen_log.csv` | Rows excluded by each screen |
| `covariance/cov_loadings.parquet`, `cov_factor.parquet` | Σ parts per month-end for universe stocks; load with `jkmp.covariance.load_covariance()` → `{eom: CovParts}`, then `.full(ids)` / `.vol(ids)`. **Monthly** units |
| `predictions.parquet` | `id, eom, year, pred_ld1` for all `valid_data` stocks, return years 2015–2025 |
| `return_model/models.npz` | Fitted model per return year (g, p, λ, β, intercept) + the shared draws Z; `return_model.load_models()` |
| `return_model/hp_search.parquet`, `chosen.csv`, `diagnostics.csv` | Validation MSE for every (year, g, p, λ); chosen hyperparameters; OOS R² and rank IC per year |
| `grids/{gammaG_wealthWB}/pfml_{rff_g*,raw}.npz` | Grid cache: Portfolio-ML yearly sums of r̃, Σ̃ + test-window signals per (γ, wealth, signal set), ~70 MB each (RFF); checked against the run's settings and rebuilt if stale |
| `covariance/factor_returns.parquet`, `cov_diagnostics.csv` | Daily factor returns (+ `n_obs`, `r2`); per-month stock counts, imputations, median vol |
| `pfml/rff_g{g:.4f}/`, `pfml/raw/` | Portfolio-ML inputs per formation month (first Σ + 12 months → test end − 1): `YYYY-MM-DD.npz` with `ids, r_tilde, denom (= risk + tc), risk, signal`; `meta.json` (column names: constant, rff1_cos…, rff1_sin…), `rff_w.npy`. Load with `portfolio_ml.load_moments()`; columns for p features via `feature_index(p, p_max)`. ~1 GB per g |

## Outputs (`outputs/`, from `scripts/04_benchmarks.py` and `05_static_ml.py`)

| File | Contents |
| --- | --- |
| `benchmarks/{weights,pf,table2}_{test,oos}.*` | Market, 1/N, Minimum variance, Markowitz-ML, Factor-ML; `test` = returns 2020–2025, `oos` = returns 2015–2025 |
| `static_ml/validation.parquet` | Monthly `pf_ts_fun` rows for every (k, u, g), returns 2015–2025 (`hp_no, k, u, g`) |
| `static_ml/chosen.csv` | Static-ML\* choice per return year (`year, eom_ret` of the December chosen, `k, u, g, cum_obj`) |
| `static_ml/weights.parquet`, `pf.parquet` | Static-ML\* weights; Static-ML and Static-ML\* monthly performance over the test window |
| `table2_test.csv` | Combined Table 2 for the test window (benchmarks, Static-ML(\*), Portfolio-ML, Portfolio-ML (linear)); rebuilt by 05 and 07 |
| `grids/{gammaG_wealthWB}/pf_static_ml.parquet`, `pf_portfolio_ml_{rff,raw}.parquet`, `chosen_*.csv` | Monthly performance and picks at one (γ, wealth) point, test window: Static-ML(\*), Portfolio-ML, Portfolio-ML (linear); `grids.load_point()` reads all |
| `grids/summary.csv`, `grids/scaled_benchmarks.csv` | Table 2 stats per (type, γ, wealth) (utility at each point's own γ); Markowitz-ML / Factor-ML scaled to target vols at \$10B |
| `figures/fig1a_frontier`, `fig1b_frontier_wealth`, `capacity` (.png, .pdf) | Fig. 1A (frontier by method at \$10B), Fig. 1B (Portfolio-ML by wealth), utility vs wealth at γ = 10 |
| `feature_importance/pf.parquet`, `importance.csv` | Monthly performance per (method, permuted theme); utility and drop `fi` vs the unshuffled run (`none`) |
| `figures/fig5_feature_importance` (.png, .pdf) | Fig. 5: utility drop by theme — Portfolio-ML (net), Portfolio-ML (linear, net), Markowitz-ML (gross) |
| `regimes/labels.csv`, `regime_table.csv` | Month labels (VIX, Fed stance, recession); Table 2 stats per (dimension, regime, method) |
| `regimes/vix_recost.csv`, `cumulative_utility.csv` | TC / R_TC / Utility with Λ ∝ VIX vs base; monthly utility flows and running sums |
| `figures/cumulative_utility` (.png, .pdf) | Cumulative utility 2020–2025 with the shaded episodes in `regimes.episodes` |
| `portfolio_ml/{rff,raw}/validation.parquet` | Realised utility `obj`, `cum_obj`, `rank` per (tag/g, p, λ) and return month, hp years 2015–2025 |
| `portfolio_ml/{rff,raw}/chosen.csv`, `aims.parquet`, `aim_coefs.parquet` | (g, p, λ) picked each December for the next return year; aim weights `id, eom, w_aim` (formation 2015-12 → 2025-11); the β used each month |
| `portfolio_ml/{rff,raw}/weights.parquet`, `pf.parquet` | Portfolio-ML weights π_t = m·π_start + (I − m)·aim and monthly performance over the test window (VW start) |

Covariance calibration (2005–2025): Σ always positive definite; realised/predicted vol ≈ 0.85 for
universe portfolios and 0.74 for single stocks — risk is overpredicted, mainly because daily → monthly
×21 scaling ignores the negative autocorrelation of daily returns (the R code does the same).

## Environment quirk: spurious matmul warnings

numpy on macOS uses Apple Accelerate, which raises spurious `divide by zero / overflow / invalid value
encountered in matmul` RuntimeWarnings (results verified identical to `einsum` to 1e-16). Wrap heavy
matmuls in `np.errstate(all="ignore")` **and** check `np.isfinite` on the result (see
`jkmp.covariance._finite`) — never silence warnings without the finite check.
