---
title: QF600 Replication Plan
tags:
  - QF600
  - asset-pricing
  - replication
  - portfolio-ml
paper: "Jensen, Kelly, Malamud & Pedersen (2026), Machine Learning and the Implementable Efficient Frontier, RFS"
repo: https://github.com/theisij/ml-and-the-implementable-efficient-frontier
---

# QF600 Replication Plan

> [!summary] TL;DR
> Write a scaled-down **Python** version in `QF600_Replication/` and use the authors' **R code as the answer key**. Replicate Table 2, Figures 1A, 2 and 5, and Table 3; skip Multiperiod-ML, short fees, size groups and simulations.

---

## 1. Why you can't just run their repo

- **It needs a computing cluster.** Fitting the 12 return models takes about 75GB of RAM per job. The base portfolio run takes 6 hours on 48 cores, and the full frontier is 20 more jobs.
- **One dataset isn't public.** `usa_dsf.csv` (daily stock returns, used for the covariance matrix) has to be built with their SAS code or requested by email. Everything else comes from WRDS (JKP `usa.csv`, which SMU's WRDS access should cover), French's website, or their GitHub/Dropbox.
- **The rubric only asks you to "replicate some portion"** and says "no need to beat benchmark." So a smaller, well-explained replication plus a real extension scores better than a partial copy of everything.

> [!tip] Recommendation
> Write a scaled-down Python version in `QF600_Replication/` and use their R code as the answer key. Your team can explain Python line by line, and the project rubric accepts R or Python.

---

## 2. What to replicate

| Paper output | Priority | Why |
| --- | --- | --- |
| **Table 2** (Portfolio-ML, Static-ML\*, Markowitz-ML, Factor-ML, Market, 1/N) | 🔴 Must | This is the core result |
| **Fig 2** (cumulative gross, net, and utility returns) | 🔴 Must | Cheap once you have Table 2 |
| **Fig 1A** (implementable efficient frontier) | 🔴 Must | Rerun Portfolio-ML and Static-ML\* at $\gamma \in \{1, 5, 10, 20, 100\}$ |
| **Table 3** (Bayesian probability of outperformance) | 🟢 Easy | About 10 lines of code |
| **Fig 5** (feature importance by theme) | 🟡 Should | Shuffle each theme's features and rerun with the same $\beta$ |
| **Fig 1B** (frontier by wealth level) | 🔵 Nice | Same as 1A, but varying $w$ |
| Multiperiod-ML, short fees, size groups, simulations | ⚪ Skip | High cost, adds little to the story |

---

## 3. Pipeline, stage by stage

Proposed layout:

```text
QF600_Replication/
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── notebooks/
└── outputs/
```

Each stage is a script that writes parquet or CSV, so a failure in one stage doesn't force you to rerun the earlier ones.

### Stage 0: Getting the data — `00_download.py`

Use the `wrds` Python package to pull:

- **JKP characteristics** (`excntry='USA'`) with only the needed columns: `id`, `eom`, `sic`, `size_grp`, `me`, `crsp_exchcd`, `dolvol_126d`, `rvol_252d`, plus about 115 features.
- **Monthly returns.**
- `ff3_m.csv` (risk-free rate), market returns, `Factor Details.xlsx` and `Cluster Labels.csv`.

Save the filtered extract as CSV, since the CSV itself is a required deliverable.

> [!question] The daily-return decision
> Either pull CRSP daily returns (`crsp.dsf`) for only the stocks in your universe, or build the covariance matrix from monthly returns (fallback described in Stage 2). **Try the daily pull first.**

### Stage 1: Data preparation — `01_prepare.py`

*Mirrors `1 - Prepare Data.R`.*

- **Screens:** NYSE only (`crsp_exchcd == 1`); non-missing `me`, returns at $t$ and $t+1$, `dolvol > 0` and SIC code; at least 50% of features present; 12 consecutive months of history (the lookback).
- **Size screen:** keep stocks above the NYSE median market cap, or just the top 300 to save compute. Then apply the 12-month addition/deletion rule so stocks don't flicker in and out of the universe.
- **Features:** percentile rank each feature within each month to $[0, 1]$, and set missing values to $0.5$.
- **Trading cost:** `lambda = 0.2 / dolvol_126d` (Eq. 35):
  $$\Lambda_{i,t} = \frac{0.2}{\mathrm{dolvol}_{i,t}}$$
- **Wealth path:** start from \$10B in Dec 2020 and roll it backwards with the market's total return, so the fund stays a constant share of the market. Store `mu_ld0` (the market return) and `tr_ld0` (the stock's own total return). These give the growth term from Eq. 6:
  $$g_t = \frac{1 + \mathrm{tr\_ld0}}{1 + \mathrm{mu\_ld0}}$$
- **Industries:** map SIC codes to the 12 Fama-French industries.

### Stage 2: Covariance matrix — `02_covariance.py`

*Mirrors `3 - Estimate Covariance Matrix.R`.*

- **13 theme scores:** average rank of each theme's characteristics (flipping signs using `direction`), then standardize cross-sectionally.
- **Daily factor returns:** each day, regress stock returns on the 12 industry dummies plus the 13 themes, with no intercept.
- **Factor covariance:** exponentially weighted over 10 years, with a 378-day half-life for correlations and 126 days for variances.
- **Stock-specific variance:** EWMA of squared residuals (126-day half-life), requiring 200 of the last 252 days. If that's missing, use the median for the stock's size group.
- **Store factors, not the full matrix:** keep `(loadings X, F, ivar)` per month and build on demand:
  $$\Sigma = X F X^\top + \operatorname{diag}(\text{ivar})$$
- **Units:** the code multiplies daily values by 21, so $\Sigma$ is **monthly**. The R comments call this "annualize," but it is monthly.

> [!note] Monthly-only fallback
> Run the same regression on monthly returns, with half-lives of about 18 months for correlations and 6 months for variances. Say so in the report as a documented deviation.

### Stage 3: Return prediction — `03_predict.py`

*Mirrors `0 - Return prediction functions.R` and `2 - Fit Models.R`.*

- **Random Fourier features:** draw $W \sim N(0, g \cdot I)$ with shape $K \times p/2$, and use features $[\cos(XW), \sin(XW)]$. *(See the warning below on what $g$ means.)*
- **Model:** ridge regression of `ret_ld1` on the random features, over the grid $p \in \{2, \dots, 512\}$ and $\lambda \in \{0, e^{-10}, \dots, e^{10}\}$. Use a closed-form solve with one eigendecomposition per $(p, g)$, so the whole $\lambda$ grid costs almost nothing.
- **Training schedule:** refit yearly from 1971 on an expanding window, using the last 10 years for validation and retraining on the full sample.
- **One horizon is enough:** with Multiperiod-ML dropped, you only need the 1-month horizon (`pred_ld1`), which cuts this stage by about 12×.

> [!warning] $g$ is a variance in the code
> In their code $g$ is the **variance** ($e^{-3}$ or $e^{-2}$), even though the paper calls it the standard deviation $\eta$.

### Stage 4: Benchmarks — `04_benchmarks.py`

*Mirrors `tpf_implement`, `factor_ml_implement`, `static_implement` and the others in `0 - Portfolio choice functions.R`.*

**Simple portfolios:**

| Method | Weights |
| --- | --- |
| Market | Value-weighted |
| 1/N | Equal-weighted |
| Minimum variance | $\dfrac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma^{-1}\mathbf{1}}$ |
| Markowitz-ML | $\Sigma^{-1}\mu / \gamma$ |
| Factor-ML | Value-weighted top and bottom 10% |

**Static-ML (Eq. 34)** — recursive, so it needs a loop over months:

$$\pi_t = \left(\gamma\Sigma + \tfrac{w}{\varphi}\Lambda\right)^{-1}\left(\mu + \tfrac{w}{\varphi}\Lambda\, g_t\, \pi_{t-1}\right)$$

**Static-ML\*:** a second tuning layer over $u \in \{0.25, 0.5, 1\}$, $v \in \{0, 1, 2\}$ and $k \in \{1, 1/3, 1/5\}$, picked yearly by highest realized utility since 1971.

> [!important] One shared weight-tracking function
> Every method needs the same inherited-weight bookkeeping (the paper's `w_fun`). Write it once and test it first:
> ```python
> w_start = w_prev * (1 + tr) / (1 + mu)
> ```

### Stage 5: Portfolio-ML — `05_portfolio_ml.py`

*The core of the paper. Mirrors `m_func` and `pfml_input_fun` → `pfml_search_coef` → `pfml_hp_reals_fun` → `pfml_aims_fun` → `pfml_w`.*

**Step 1: compute $m$ (Lemma 1).** This is a fixed-point iteration (`m_func`). Because $\Lambda$ is diagonal, $\Lambda^{-1/2}$ is cheap. Initialize with the closed-form guess, then iterate about 10 times.

**Step 2: build each month's signal matrix.** The aim portfolio is

$$A_t = \operatorname{diag}(1/\sigma_i)\cdot RF_t \cdot \beta$$

For each random feature: demean it cross-sectionally, scale it to unit sum of squares, and add a scaled constant column. Then combine the current and 11 lagged signal matrices, weighted by products of $m \cdot \operatorname{diag}(g)$ (Eq. 24, truncated at 12 lags).

> [!warning] Code-vs-paper difference
> The code normalizes these weights by the inverse of their own sum (`solve(const)`) instead of multiplying by $(I - m)$. This keeps the truncated sum well-scaled. **Copy the code here, and mention it in the report.**

**Step 3: per-month sufficient statistics.** Do this once per random-feature draw $g$:

$$\tilde r_t = \tilde s_t^\top r_{t+1} \quad \text{(length } p+1\text{)}$$

$$\tilde\Sigma_t = \gamma\, \tilde s_t^\top \Sigma\, \tilde s_t + w\,(\tilde s_t - g_t \tilde s_{t-1})^\top \Lambda\, (\tilde s_t - g_t \tilde s_{t-1}) \quad \text{(}(p+1)\times(p+1)\text{)}$$

Save these to disk. Nothing after this step touches individual stocks.

**Step 4: $\beta$ by closed form (Eq. 26).** For each year $y$ from 1971, take running averages of $\tilde r$ and $\tilde\Sigma$ up to $y-1$, then compute for every $(p, \lambda)$:

$$\beta = (\bar\Sigma + \lambda I)^{-1}\, \bar r$$

Smaller $p$ values just use the first $p$ columns of the $p = 512$ draw. This takes seconds.

**Step 5: hyperparameter selection.** Each month's realized utility is

$$\tilde r^\top \beta - \tfrac{1}{2}\beta^\top \tilde\Sigma \beta$$

Each December, choose the $(p, \lambda, g)$ with the highest cumulative average since 1971, and use it for the following year. The true out-of-sample period is 1981–2020.

**Step 6: final weights.** Run the recursion

$$\pi_t = m\, g_t\, \pi_{t-1} + (I - m)\, A_t, \qquad A_t = \text{signal}_t \cdot \beta$$

### Stage 6: Evaluation — `06_evaluate.py`

**Per-month metrics:**

| Metric | Formula |
| --- | --- |
| Gross return | $\pi^\top r$ |
| Trading cost | $TC = \tfrac{w}{2}(\pi_t - g_t\pi_{t-1})^\top \Lambda (\pi_t - g_t\pi_{t-1})$ |
| Net return | gross − TC |
| Turnover | $\sum \lvert \Delta w \rvert$ |
| Leverage | $\sum \lvert w \rvert$ |

**Annualized results:**

- Return $R = 12 \cdot \text{mean}$
- Volatility $= \sqrt{12} \cdot \text{sd}$
- Sharpe ratio, gross and net
- Utility $= 12 \cdot \text{mean(net)} - \tfrac{\gamma}{2} \cdot 12 \cdot \text{var(net)}$

**Outputs:**

- **Table 3:** pairwise differences in monthly utility, then $\Phi\!\left(\dfrac{\text{mean}}{\text{sd}/\sqrt{T}}\right)$.
- **Fig 1A:** rerun Stages 4–5 at the five $\gamma$ values.
- **Fig 5:** for each theme, shuffle that theme's features across stocks within each month, regenerate the random features using the **same** $W$, and rerun the weights with the **same** $\beta$. **Do not re-estimate $\beta$.** Importance is the drop in utility.

### Stage 7: Monitoring — `07_monitor.py`

*Required by the deliverables.*

Produce:

- Rolling 36-month charts of net Sharpe, utility, trading cost and turnover
- The chosen hyperparameters over time (the paper's Fig D.1)
- Feature-importance drift

One dashboard-style figure covers the "monitoring" requirement and gives you a slide-appendix screenshot.

---

## 4. Settings to shrink it to laptop size

| Setting | Paper | Suggested | Effect |
| --- | --- | --- | --- |
| Universe | ~646 stocks (above NYSE median) | Top 300 by market cap | Cost of $N^2$ terms drops about 4–5× |
| Maximum random features $p$ | 512 | 256 | $\tilde\Sigma$ storage drops 4× |
| $\lambda$ grid | 101 values | 101 (keep it, it's nearly free) | None |
| Horizons | 12 | 1 | 12× less in Stage 3 |
| Random-feature draws $g$ | 2 | 2 | None |

> [!info] Rough cost
> $600 \text{ months} \times 12 \text{ lags} \times N^2 \cdot p \approx 10^{12}$ floating-point operations at $N = 300$ and $p = 256$. That's minutes in NumPy.
> Storing $\tilde\Sigma$ takes about $600 \times 257^2 \times 8$ bytes $\approx$ 300MB.
> Use `float64`, and use `np.linalg.solve`, **never** `inv`.

---

## 5. Mistakes that silently ruin results

> [!danger] Timing
> `eom` is when you form the portfolio, `ret_ld1` is the next month's return, and `tr_ld0` is the return just realized (used for $g_t$). One misalignment here looks like amazing performance.

> [!danger] Units of `dolvol_126d`
> Check whether JKP reports it in dollars or millions. Trading cost scales with $w \cdot \Lambda$, so a factor of $10^6$ error makes trading cost either zero or enormous.
> **Sanity check:** Table 2 has Portfolio-ML trading cost ≈ 0.006 per year and Factor-ML ≈ 2.5.

> [!danger] Order of random-feature scaling
> Demean → scale to unit sum of squares → divide by volatility. Skip this and leverage grows with the number of stocks.

> [!danger] Universe changes
> When a stock leaves the universe, it is sold to zero (`w_start` is kept, the target is 0). When it enters, it starts from 0.

> [!danger] Look-ahead in tuning
> Hyperparameters chosen in December of $y-1$ apply to year $y$, and $\beta$ for year $y$ uses data only up to $y-1$.

---

## 6. Extensions for the "iterative improvement" requirement

Pick one or two:

- [ ] **Post-publication test (strongest):** JKP data now runs past 2020. Running 2021–2024 tests the paper on data it never saw.
- [ ] **Different cost model:** square-root price impact (closer to empirical estimates than the paper's quadratic cost), or a higher or lower Frazzini calibration (`pi`).
- [ ] **Different learner:** replace random features with a small MLP or with the 13 raw theme scores (a linear version, as in Brandt et al.). This asks whether the nonlinearity matters after costs.
- [ ] **Wealth sensitivity:** an "AUM capacity" curve, showing utility as a function of wealth.

---

## 7. Suggested order and team split

| Week | Work | Owner |
| --- | --- | --- |
| 1 | Stages 0–1, data checks against the paper's universe counts (min 184, median 646, max 805) | A |
| 1–2 | Stage 2 covariance, checked against plausible market volatility | B |
| 2 | Stage 3 predictions plus Stage 4 simple benchmarks, giving the first Table 2 rows | C |
| 2–3 | Stage 5 Portfolio-ML (the hardest part) | B + D |
| 3 | Static-ML\*, Table 2, Fig 2, Table 3 | C |
| 4 | Fig 1A, Fig 5, extension, monitoring | Everyone |
| 4–5 | Report (≤20 pages plus executive summary) and slides with speaker names | Everyone |

> [!success] Build Market and 1/N end to end first
> Data → weights → trading cost → Table 2 row. If your Market row comes out close to **R ≈ 0.08, Vol ≈ 0.15 and TC ≈ 0**, your timing and cost plumbing are right, and everything else plugs into the same framework.

---

## Next steps

- [ ] Confirm WRDS access
- [ ] Confirm team language (Python or R)
- [ ] Set up `QF600_Replication/` folder structure, config file, and the Stage 1 data-prep script
