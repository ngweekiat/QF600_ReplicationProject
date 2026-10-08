---
title: JKMP R Code Map
tags:
  - QF600
  - replication
  - portfolio-ml
  - code-map
paper: "Jensen, Kelly, Malamud & Pedersen (2026), Machine Learning and the Implementable Efficient Frontier, RFS"
repo: https://github.com/theisij/ml-and-the-implementable-efficient-frontier
related: "[[QF600 Replication Plan]]"
---

# JKMP R Code Map

> [!summary] TL;DR
> The repo has four layers:
> 1. **One settings file** (`Main.R`)
> 2. **Three function libraries** that hold the paper's equations (`0 - *.R`)
> 3. **Numbered scripts** that build the data and run each method (`1`–`5`)
> 4. **Analysis scripts** that turn results into tables and figures (`6`, `7`)
>
> To port into Python, read these line by line: `0 - Portfolio choice functions.R`, `0 - General functions.R`, `1 - Prepare Data.R`, `3 - Estimate Covariance Matrix.R`. Everything else is runners, cluster plumbing or plotting.

See also: [[QF600 Replication Plan]]

---

## 1. How the files run

```text
Main.R  ← settings, feature list, libraries; sources the 3 function libraries + C++ helpers
  │
  ├─ slurm_fit_models.R  (cluster job; all US stocks, not just NYSE)
  │     ├─ 1 - Prepare Data.R
  │     └─ 2 - Fit Models.R              → Data/Generated/Models/model_h.RDS   (h = 1…12)
  │
  ├─ slurm_build_portfolios.R  (cluster job; one per γ / wealth / size config)
  │     ├─ 1 - Prepare Data.R
  │     ├─ 3 - Estimate Covariance Matrix.R
  │     ├─ 4 - Prepare Portfolio Data.R   (loads the saved predictions)
  │     ├─ 5 - Base case.R                → bms.csv, static-ml.RDS, portfolio-ml.RDS, multiperiod-ml.RDS
  │     └─ 5 - Feature importance *.R     → pfml_cf_*.csv, tpf_cf_base.csv, ret_cf.csv
  │
  └─ separate_analysis_run.R  (local PC)
        ├─ 1, 3, 4 again (rebuild inputs)
        ├─ 6 - *.R   ← compute every table and figure into an `output` list
        └─ 7 - Figures.R / 7 - Tables.R / 7 - Numbers.R   ← save PDFs, print LaTeX tables
```

---

## 2. File by file

### Settings and function libraries

| File | Lines | Role | What it does |
| --- | --- | --- | --- |
| `Main.R` | 192 | ⚙️ Settings | All settings in one list: dates, screens, `pi = 0.1` (price impact), hyperparameter grids, covariance half-lives, $\gamma$ and wealth grids, `pf_set` (wealth \$10B, $\gamma = 10$, `mu = 0.007`, `lb_hor = 11`), and the 115-feature list with exclusions. Loads libraries and `ewma.cpp` / `sqrtm_cpp.cpp` |
| `0 - General functions.R` | 257 | 📐 Equations + helpers | Builds the covariance, $\Lambda$ and returns; the shared accounting for gross return, trading cost, turnover and leverage (`pf_ts_fun`); the universe rules |
| `0 - Return prediction functions.R` | 188 | 📐 Equations (ML) | Random Fourier features, train/validation split, ridge hyperparameter search |
| `0 - Portfolio choice functions.R` | 839 | 📐 Equations (core) | Every portfolio method: $m$ (Lemma 1), all benchmarks, Static-ML, Portfolio-ML, Multiperiod-ML, counterfactual feature importance |

### Data and estimation scripts

| File | Lines | Role | What it does |
| --- | --- | --- | --- |
| `1 - Prepare Data.R` | 160 | 🧹 Data manipulation | Loads raw files, builds returns, wealth path and $\Lambda$, applies screens, ranks features, maps industries, flags the investable universe, reads daily returns |
| `2 - Fit Models.R` | 45 | 📈 Estimation | Fits the yearly-refit random-feature ridge model for each horizon and saves predictions |
| `3 - Estimate Covariance Matrix.R` | 182 | 🧹 Data + 📐 equations | Barra-style covariance (Eqs. 36–37); stores the factor loadings, factor covariance and idiosyncratic variances per month |
| `4 - Prepare Portfolio Data.R` | 30 | 🧹 Data manipulation | Merges predictions `pred_ld1…12` into `chars`; builds `lambda_list`; defines date vectors (`dates_oos`, `dates_hp`, `dates_m1`, `dates_m2`) |

### Runners

| File | Lines | Role | What it does |
| --- | --- | --- | --- |
| `5 - Base case.R` | 100 | ▶️ Runner | Calls each `*_implement` function at the base settings and saves results |
| `5 - Feature importance base.R` | 44 | ▶️ Runner | Theme-shuffling counterfactuals for Figure 5 |
| `5 - Feature importance IEF.R` | 28 | ▶️ Runner | The frontier version of feature importance |
| `5 - Feature importance ret.R` | 35 | ▶️ Runner | Return-model MSE after shuffling each theme |

### Analysis and output

| File | Lines | Role | What it does |
| --- | --- | --- | --- |
| `6 - Base analysis.R` | 357 | 📊 Analysis | Table 2 statistics, cumulative returns (Fig 2), outperformance probability (Table 3), correlations (Table 4), statistics over time |
| `6 - Implementable efficient frontier.R` | 219 | 📊 Analysis | Figures 1A and 1B: the frontiers, the mean-variance hyperbola, indifference curves |
| `6 - Feature importance.R` | 180 | 📊 Analysis | Figures 5 and 6 |
| `6 - Economic intuition.R` | 84 | 📊 Analysis | Figures 3 and 4 (weights by liquidity; the J&J vs Xerox example) |
| `6 - Performance across size distribution.R` | 60 | 📊 Analysis | Figure 8 |
| `6 - Short selling fees.R` | 138 | 📊 Analysis | Figure 7 (needs Markit data) |
| `6 - RF example.R` | 400 | 🧪 Illustration | Toy random-features example (appendix) |
| `7 - Figures.R` / `7 - Tables.R` / `7 - Numbers.R` | 36–101 | 💾 Output | Save PDF figures, print LaTeX tables and in-text numbers |

### Plumbing and helpers

| File | Lines | Role | What it does |
| --- | --- | --- | --- |
| `slurm_*.R`, `Joblists/`, `config_files/` | — | 🖥️ Cluster plumbing | Choose settings per job ($\gamma$, wealth, size screen); not needed on a laptop |
| `ewma.cpp` | 38 | C++ helper | EWMA volatility |
| `sqrtm_cpp.cpp` | 10 | C++ helper | Matrix square root (used for the starting value of $m$) |
| `simulations/` | ~630 | 🧪 Appendix E | Simulation study; skip |

---

## 3. Where each equation lives

### Costs, growth and utility

| Paper | Concept | File → function (line) | What the code does |
| --- | --- | --- | --- |
| Eqs. 4, 35 | Price impact $\Lambda$ | `1 - Prepare Data.R` → `chars[, lambda := 2/dolvol*settings$pi]` | $\Lambda = 2 \times 0.1 / \mathrm{dolvol\_126d} = 0.2/V$ |
| Eqs. 5–6 | Portfolio growth $g$ | `pfml_input_fun` (283), `w_fun` (26) | $g = (1 + \mathrm{tr\_ld0})/(1 + \mathrm{mu\_ld0})$; drifted weight $= w \times (1 + \mathrm{tr\_ld1})/(1 + \mathrm{mu\_ld1})$ |
| Eq. 7 | Trading cost | `pf_ts_fun` (General, 124) | `tc = wealth/2 * sum(lambda*(w - w_start)^2)` |
| Eq. 11, Table 2 | Realised utility and summary statistics | `6 - Base analysis.R` → `pf_summary` | utility $= (\overline{r} - \tfrac{1}{2}\gamma\,\mathrm{var}(r) - \overline{tc}) \times 12$ |
| Lemma 1 | Trading speed $m$ | `m_func` (2) | Closed-form starting value via `sqrtm_cpp`, then 10 fixed-point iterations; includes the portfolio-growth terms |

### Benchmarks

| Paper | Concept | File → function (line) | What the code does |
| --- | --- | --- | --- |
| Eq. 32 | Markowitz-ML | `tpf_implement` (42) | $\tfrac{1}{\gamma}\Sigma^{-1} \times$ prediction |
| Sec. 4.1 | Factor-ML | `factor_ml_implement` (121) | Value-weighted top/bottom decile |
| Sec. 4.1 | Rank-ML | `rw_implement` (168) | Rank weights scaled to $+1/-1$ |
| Sec. 6 | Market, 1/N, minimum variance | `mkt_implement` (156), `ew_implement` (144), `mv_implement` (188) | Standard weights |
| Eq. 34 | Static-ML | `m_static` (20), `static_val_fun` (210) | $w = (\gamma\Sigma + w\Lambda)^{-1}(u \cdot \mathrm{pred} + w\Lambda \times \text{drifted weights})$, monthly loop |
| Eq. 41 | Second tuning layer | `sigma_gam_adj` (General, 86), `static_implement` (240) | $u$ scales predictions; $k$ scales $\Lambda$; $v$ via `cov_add`: $\Sigma + g \cdot \mathrm{diag}(\Sigma)$ |
| Eq. 22 | Multiperiod-ML aim | `mp_aim_fun` (714), `mp_val_fun` (735) | Discounted sum of horizon-$\tau$ Markowitz portfolios |

### Return prediction

| Paper | Concept | File → function (line) | What the code does |
| --- | --- | --- | --- |
| Eqs. 38–39 | Random Fourier features | `rff` (Return pred., 21) | $W \sim N(0, g \cdot I)$; output $\cos(XW)$ and $\sin(XW)$ |
| Return models | Ridge search, splits | `rff_hp_search` (32), `data_split` (2) | `glmnet` ridge; choose $(g, p, \lambda)$ by validation MSE; refit on train + validation |

### Portfolio-ML

| Paper | Concept | File → function (line) | What the code does |
| --- | --- | --- | --- |
| Eqs. 24, 40 | Signals | `pfml_input_fun` (283) | Demean features, scale columns to unit sum of squares, add a constant, multiply by 1/vol, combine 12 lags weighted by $m \times \mathrm{diag}(g)$ |
| Eq. 25 | Per-month moments | `pfml_input_fun` (283) | $\tilde r = \Omega^\top r$; risk term $\gamma\,\Omega^\top\Sigma\,\Omega$ + cost term $w\,\Delta\Omega^\top\Lambda\,\Delta\Omega$ |
| Eq. 26 | Closed-form $\beta$ | `pfml_search_coef` (446) | Running sums of $\tilde r$ and $\tilde\Sigma$; $\beta = (\bar\Sigma + \lambda I)^{-1}\bar r$ for each $(p, \lambda)$ |
| Sec. 5.3 | Hyperparameter selection | `pfml_hp_reals_fun` (489) | Monthly utility $\tilde r^\top\beta - \tfrac{1}{2}\beta^\top\tilde\Sigma\beta$; pick the highest cumulative average each December |
| Eq. 21 | Aim and weights | `pfml_aims_fun` (514), `pfml_w` (536) | Aim = signal $\times\ \beta$; weights $= m \times$ drifted weights $+ (I - m) \times$ aim |
| — | Wrapper | `pfml_implement` (582) | Loops over $g$ → inputs → coefficients → validation → aims → weights |

### Covariance, testing, feature importance

| Paper | Concept | File → function (line) | What the code does |
| --- | --- | --- | --- |
| Eqs. 36–37 | Covariance | `3 - Estimate Covariance Matrix.R`, `create_cov` (General, 16) | Daily regressions on 12 industries + 13 themes; EWMA factor covariance; $\Sigma = XFX^\top + \mathrm{diag}(\text{idio. var.})$ |
| Table 3 | Outperformance probability | `6 - Base analysis.R` → `prob_outperformance` | $1 - \Phi(0;\ \overline{\mathrm{diff}},\ \mathrm{sd}/\sqrt{T})$ |
| Eq. 43 | Economic feature importance | `pfml_cf_fun` (654), `tpf_cf_fun` (62) | Shuffle one theme across stocks each month; rebuild features with the **same** $W$ and $\beta$ |

---

## 4. Key data objects passed between files

| Object | Built in | Contents |
| --- | --- | --- |
| `chars` | `1 - Prepare Data.R` | One row per stock-month: `id`, `eom`, `me`, ranked features, `lambda`, `ff12`, `valid`, plus timing columns (below) |
| `wealth` | `1 - Prepare Data.R` (`wealth_func`) | `eom`, `wealth`, `mu_ld1` (market total return next month), rolled backwards from \$10B at end-2020 |
| `risk_free` | `1 - Prepare Data.R` | `eom`, `rf` from `ff3_m.csv` |
| `cluster_labels` | `1 - Prepare Data.R` | feature → theme → sign direction (from the JKP files) |
| `barra_cov` | `3 - Estimate Covariance Matrix.R` | List by month of {loadings `fct_load`, factor covariance `fct_cov`, idiosyncratic variances `ivol_vec`} |
| `lambda_list` | `4 - Prepare Portfolio Data.R` | List by month of named $\Lambda$ vectors |
| `pred_ld1…pred_ld12` | `2 - Fit Models.R` → `4 - …` | Out-of-sample return predictions |

### Timing columns

> [!danger] The most common source of silent bugs
> Misaligning any of these makes results look far better than they are.

| Column | Meaning |
| --- | --- |
| `ret_ld1` | Excess return over the next month (what the portfolio earns) |
| `tr_ld1` | Total return over the next month (used to drift weights forward) |
| `tr_ld0` | Total return over the month just ended (used for $g$ in the Portfolio-ML signals) |
| `mu_ld1`, `mu_ld0` | The same two, for the market (wealth growth) |

---

## 5. Code details that matter for your replication

> [!important] These resolve ambiguities in the paper
> Where the paper and code disagree, follow the code.

1. **$m$ (Lemma 1) is the full version.** `m_func` includes the portfolio-growth terms. Port it directly rather than using a simplified approximation.
2. **Table 3 does divide by $\sqrt{T}$** (`sd(diff)/sqrt(length(diff))`).
3. **Table 2 uses gross-return variance.** Utility and net Sharpe both use `var(r)` / `sd(r)` of gross returns, not net returns.
4. **The variance add-on in Eq. 41 is a multiplier on the diagonal.** The code's `g ∈ {0, 1, 2}` gives $\Sigma + g \cdot \mathrm{diag}(\Sigma)$, which matches the paper's $v \in \{1, 2, 3\}$ ($v = 1 + g$).
5. **"Annualize" ×21 actually gives monthly $\Sigma$.**
6. **The random-feature weight scale $g$ is a variance,** not a standard deviation. Return models scale features by $1/\sqrt{p}$; Portfolio-ML instead scales each column to unit sum of squares.
7. **Return models are trained on all US stocks.** `slurm_fit_models.R` turns off the NYSE and size screens. Portfolios use only the NYSE above-median universe.
8. **Static-ML\* implements $k$ as a multiplier on $\Lambda$,** so $k = 1/\varphi$.
9. **Portfolio-ML lag weights are normalised by `solve(const)`,** their own sum, rather than $(I - m)$.
10. **The first month starts from value weights** (`initial_weights_new(w_type = "vw")`), not from zero.

> [!warning] Exiting stocks look uncharged
> As far as I can read `w_fun` and `pfml_w`, a stock that leaves the universe drops out of next month's data, so liquidating it isn't charged a trading cost. Decide whether to copy this. Charging the sale is more realistic; either way, state your choice in the report.

---

## Porting checklist

- [ ] `0 - General functions.R` → accounting (`pf_ts_fun`), covariance assembly (`create_cov`), universe rules (`addition_deletion_fun`)
- [ ] `1 - Prepare Data.R` → data pipeline and timing columns
- [ ] `3 - Estimate Covariance Matrix.R` → Barra-style covariance
- [ ] `0 - Portfolio choice functions.R` → `m_func`, `w_fun`, benchmarks, `static_val_fun`, `pfml_*`
- [ ] `0 - Return prediction functions.R` → `rff`, `rff_hp_search`
- [ ] `6 - Base analysis.R` → Table 2 and Table 3 formulas
