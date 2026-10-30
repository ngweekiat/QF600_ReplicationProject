# Portfolio-ML for a $10B Large-Cap Mandate: Research Framework

Oct 29, 2026 · @wk

## The business question

The client is a $10B US large-cap equity mandate benchmarked to the cap-weighted index. The research question is whether Portfolio-ML can add value over that index after trading costs, and up to what fund size.

This is how a pension or sovereign fund actually hires a quant manager. It sets a benchmark, a tracking-error budget and a capacity requirement, then judges the manager on the information ratio net of costs. The paper's investor maximises total-return utility with γ = 10, a setup no institutional allocator uses. Recasting the work as a mandate gives every iteration a decision attached to it:

1. **Launch.** Does the strategy beat the index net of costs with enough statistical confidence to justify the fee?
2. **Mandate design.** Which universe and which tracking-error budget should the product offer?
3. **Capacity.** Above what AUM does the net alpha disappear, so the product should close to new money?
4. **Governance.** What should be monitored live, and what should trigger a review or a switch-off?

The replication's setup already suits this framing. The top 80 stocks are where a $10B fund can actually trade. The 2020–2025 test window comes after the paper's sample, so the exercise is the same due diligence an allocator runs on a published strategy before funding it. Post-publication decay is the norm in this literature: McLean and Pontiff (2016) find anomaly returns fall by about half after publication.

## Evaluation of the current replication

The pipeline in `QF600_Replication/` is a faithful, well-engineered port of the authors' R code, and the results that do not depend on signal strength reproduce. Its weak points are the backtest setup and the investment objective, and those weak points define the three iterations.

**What is solid.**

- **Faithful port.** Every paper equation is mapped to its R source (`docs/JKMP R Code Map.md`), and each of the 40-odd departures from the paper or the R code is logged with a reason (`docs/deviations.md`).
- **Tested.** 117 tests pass, including a look-ahead test that scrambles future returns (`tests/test_models.py`).
- **Sanity anchors reproduce.** Market turnover is 1.3% a month, cost-blind methods collapse after costs (Markowitz-ML pays 585% a year), and gross-return correlations are close to the paper's (0.54 against 0.52 for Portfolio-ML with Static-ML\*).
- **Built to iterate.** Every parameter lives in `configs/base.yaml`, and an experiment is a YAML override, as `configs/experiments/linear_pfml.yaml` already shows.

**Where utility is lost.** Realised utility splits into gross return, trading cost and the risk penalty:

```latex
U = \bar r \; - \; \overline{TC} \; - \; \tfrac{\gamma}{2}\,\sigma^2
```

| Method ($10B, γ = 10, 2020–25) | Gross return | Trading cost | Risk penalty | Utility |
| --- | --- | --- | --- | --- |
| Market | 15.16% | 0.02% | 15.35% | −0.21% |
| Portfolio-ML (linear) | 1.00% | 0.07% | 1.45% | −0.52% |
| Portfolio-ML | 0.35% | 0.17% | 2.03% | −1.85% |
| Static-ML\* | −1.35% | 0.17% | 1.37% | −2.89% |

Source: `outputs/table2_test.csv`; annualised. Trading cost is the smallest of the three for every method that uses the cost model.

**Issues, and where each one is dealt with.**

| # | Issue | Evidence | Dealt with in |
| --- | --- | --- | --- |
| 1 | Every method starts at value weights in Dec 2019, just before the COVID crash | Without 2020, Static-ML\* goes from −2.89% to +0.44% and overtakes Portfolio-ML (−1.64%); the fall in utility with wealth at γ = 100 disappears | Baseline fix 1 |
| 2 | Hyperparameters chosen at the grid edge | Portfolio-ML picks p = 64 (the minimum) from 2017 and λ = e¹⁰ (the maximum) in 2017; Static-ML\* sits at its most conservative corner in 8 of 11 years | Baseline fix 2 |
| 3 | Feature importance depends on the random draw | Only 2 of 6 themes keep their sign across 9 seeds | Baseline fix 3 |
| 4 | Total-return objective lets the model choose its own market exposure | β = 0.20 and α = −2.6% a year against the universe's value-weighted portfolio; correlation 0.82 with minimum variance, because the vol-scaled constant column is an inverse-volatility long book | Iteration 1 |
| 5 | 80 mega-caps leave almost no signal | Return-forecast rank IC 0.002 inside the universe against 0.07 across all screened stocks | Iteration 2 |
| 6 | Trading costs too small to matter at $10B | Portfolio-ML pays 0.17% a year; VIX re-costing changes its utility by −0.01% | Cost sensitivity inside Iteration 2 |
| 7 | β is estimated from equally weighted means since 2004, so the model is slow to unlearn | Low risk is the most harmful theme (−0.85%); minimum variance α = −3.8% in 2020–25 | Iteration 3 |
| 8 | Risk model over-predicts | Realised against predicted volatility 0.85 for portfolios, 0.74 for single stocks | Stated limitation; monitored with the bias statistic |
| 9 | Housekeeping | `wrds` imported when the library loads; CLAUDE.md out of date; report is 23 pages with no executive summary; `jkp_usa_top500.csv` is WRDS data | Baseline fix 4 |

Issue 8 is not made an iteration on purpose. Correcting over-prediction lowers Σ, which enlarges positions, and with negative alpha that lowers utility even though the risk model improves.

## Before iterating: harden the baseline

Four fixes turn the current replication into a baseline that iterations can be measured against. They correct the setup rather than improve the strategy, so they do not count as iterations.

| Fix | Code it touches | Done when |
| --- | --- | --- |
| 1. Run every method's weights from 2015-12 and report 2020–2025 only | `eval_dates` in `run_portfolio_ml` (`portfolio/portfolio_ml.py`); `scripts/04_benchmarks.py`, `05_static_ml.py`, `07_portfolio_ml.py` | Table 2 no longer depends on the December 2019 start, and 2016–2019 becomes a usable validation period |
| 2. Extend grids: Portfolio-ML p ∈ {8, 16, 32}; Static-ML\* u ∈ {0, 0.1} | `portfolio_ml.p_grid`, `static_ml.u_grid` in `configs/base.yaml` | No pick sits on a grid edge, or the edge is reported |
| 3. Feature importance over 20 seeds, with standard errors | `scripts/09_feature_importance.py`, `evaluation/feature_importance.py` | Figure 5 shows error bars |
| 4. Import `wrds` inside the pull functions; update CLAUDE.md; replace the WRDS CSV with a source reference | `data/wrds_pull.py`, `CLAUDE.md` | Tests run without WRDS installed |

Aims and Static-ML\* picks already exist from December 2015, so fix 1 needs no new estimation. Rerun Tables 2 and 3 afterwards. Whatever the new ranking is, it becomes the baseline, and the report says plainly if the paper's ordering of methods does not survive.

## Research protocol

Every iteration follows the same five rules, borrowed from how systematic funds guard against backtest overfitting. The point is that a reader can trust a kept change because the rule for keeping it was fixed before the result was seen.

1. **Write the hypothesis first.** Before running anything, record the hypothesis, the evidence behind it, the metric, the threshold for keeping the change and the kill criterion. Keep these in a research log (`docs/research_log.md`), the equivalent of an investment committee research memo.
2. **Change one thing.** Seeds, grids, universe and dates stay fixed unless the iteration is about them.
3. **Decide on 2016–2019, report on 2020–2025.** Once weights run from December 2015 (baseline fix 1), 2016–2019 is a 48-month walk-forward validation period that uses no test data. A change is kept or dropped there; its 2020–2025 result is reported either way, including for rejected changes.
4. **Keep a change only if it is likely real.** The rule is the paper's own Table 3 test: the posterior probability that the mean monthly utility difference is positive must be at least 80% on the validation period.

   ```latex
   \Pr\left(\mathbb{E}[d] > 0\right) = \Phi\left(\frac{\bar d}{s_d/\sqrt{T}}\right), \qquad d_t = u^{\text{new}}_t - u^{\text{old}}_t
   ```
5. **Count the trials.** Every variant run counts as a trial, kept or not. The final model's Sharpe ratio is reported with the deflated Sharpe ratio (Bailey and López de Prado, 2014), which discounts it for the number of trials. Harvey, Liu and Zhu (2016) argue new factors need a t-statistic above 3 for the same reason.

Each kept change then passes a robustness battery before it becomes the new baseline: 20 random-feature seeds, sub-periods, γ ∈ {5, 10, 20} and wealth from $1B to $250B. 80% is a deliberately loose bar for 48 months; the deflated Sharpe ratio at the end is the strict one.

## Roadmap

&#91;embedded content: iteration roadmap · baseline, fixes, 3 iterations, closure\]

The order follows the evaluation. Iteration 1 changes what the strategy is asked to do, Iteration 2 where it looks for signal, and Iteration 3 how fast it forgets. Each builds on the version kept before it.

Step-by-step tasks, files and git workflow: Implementation plan

## Iteration 1: benchmark-relative Portfolio-ML

Iteration 1 makes Portfolio-ML learn active bets around the index instead of a whole portfolio. It addresses issue 4, the largest source of lost utility, and it is the change that matters most to a real client.

**Hypothesis.** Under total-return utility with γ = 10, the model sets its own market exposure, about 35% net long with β = 0.20. Dividing the constant column by volatility adds an inverse-volatility tilt. Neither is a bet the client hired the manager to make. Learning only the deviation from the index removes both and isolates the real question: is there stock-selection alpha?

**Change.** Hold the benchmark and learn the active weights, penalising tracking error instead of total variance:

```latex
\pi_t = \pi^{\text{VW}}_t + a_t, \qquad a_t = \Omega_t\,\beta
```

```latex
\max_\beta \; \mathbb{E}\!\left[a_t^\top r_{t+1} \;-\; \tfrac{w_t}{2}\,\Delta a_t^\top \Lambda_t\, \Delta a_t \;-\; \tfrac{\gamma_{TE}}{2}\, a_t^\top \Sigma_t\, a_t\right]
```

Value weights drift with the market, so the benchmark itself barely trades (1.3% turnover a month, 0.02% a year in cost) and Δπ ≈ Δa. The paper's closed form for β therefore still applies.

**Where it lands in your code.**

| Change | File |
| --- | --- |
| Flag to drop the constant column (`portfolio_ml.add_constant: false`) | `scale_signals` in `portfolio/portfolio_ml.py` |
| γ\_TE in place of γ, set so ex-ante tracking error is 1%, 2% or 4% | `month_moments` (unchanged formula), new `investor.gamma_te` in `config.py` and `base.yaml` |
| Run the recursion on active weights, a\_t = m g a\_{t−1} + (I − m) A\_t, then add the value weights | `pfml_weights` in `portfolio/portfolio_ml.py`; value weights from `portfolio/benchmarks.py` |
| Active return, tracking error and information ratio | `evaluation/metrics.py` |
| The experiment | `configs/experiments/active_pfml.yaml` |

**Real-world analogue.** Institutional large-cap mandates work this way: a benchmark, a tracking-error budget (about 1–2% for enhanced index, 4–6% for active) and an information-ratio target (Grinold and Kahn, 2000).

**Metrics.** Active return, realised against ex-ante tracking error, gross and net information ratio, β to the benchmark, active exposure to each theme, and the share of stocks with negative total weight. A long-only mandate would need a constraint that breaks the closed form; report that as a limitation rather than solving it.

**Kill criterion.** The net information ratio is not positive with at least 80% probability on 2016–2019 at any tracking-error budget. That outcome is still informative: it says the 18 characteristics carry no stock-selection alpha among mega-caps, which Iteration 2 then tests in a wider universe.

## Iteration 2: breadth versus capacity

Iteration 2 widens the universe from 80 to 250 stocks, and then to 500 if run time allows. It tests your report's own leading explanation (§9.2, cause 1) and addresses issues 5 and 6.

**Hypothesis.** The signal exists, but not among mega-caps. The fundamental law of active management (Grinold and Kahn, 2000; Clarke, de Silva and Thorley, 2002) links the information ratio to forecast skill and the number of independent bets:

```latex
IR \;\approx\; TC \times IC \times \sqrt{BR}
```

Here TC is the transfer coefficient (how much of the forecast survives portfolio constraints), not trading cost. With a rank IC of 0.002 inside the top 80, no portfolio construction can produce alpha. Across all screened stocks the IC is 0.07, so the question is how far down the size distribution the strategy must go to find it.

**Change.** Run the Iteration 1 model on the wider universe, measured against that universe's own cap-weighted index.

| Change | File |
| --- | --- |
| `universe.target_n` 80 → 250, with `entry_rank` and `exit_rank` scaled (e.g. 220 / 300) | `configs/experiments/universe_250.yaml`; logic in `data/universe.py` |
| Factor returns are already estimated on the top 500; only loadings and idiosyncratic variances for the new members need rebuilding | `covariance.factor_universe_top_n` |
| Capacity curve over the existing wealth grid ($1B to $250B) | `scripts/08_grids.py`, `evaluation/frontier.py` |

Run time is the constraint: the trading-speed matrix m is N × N and is rebuilt each month for 13 signal months. Time a single year at N = 250 before committing to 500.

**Cost sensitivity, inside this iteration.** A wider universe brings in less liquid stocks, so costs start to bind. Re-cost the final weights under two stress versions of Λ without retraining: the VIX-scaled Λ already in `evaluation/regimes.py`, and a stock-level version, Λ × (σ\_{i,t} / σ̄\_{i,t}), using past data only. Report the capacity estimate under all three. This is evaluation only, so it is not a separate iteration.

**Real-world analogue.** Choosing between an S&P 100-style, S&P 500 or Russell 1000 mandate is exactly this trade-off: more breadth means more bets but lower capacity. Pre-trade cost models that scale with volatility (Almgren, Thum, Hauptmann and Li, 2005) are what desks use to check that capacity holds in a stress month.

**Metrics.** Rank IC by size bucket, net information ratio, trading cost, and the AUM at which the net information ratio falls below 0.3, under each cost model.

**Kill criterion.** The net information ratio is no better than in Iteration 1, with less than 80% probability of improvement on 2016–2019. The finding would then be that the 18 characteristics, not the universe, are the bottleneck.

## Iteration 3: time-decayed estimation for signal decay

Iteration 3 lets the model forget old evidence faster, so it can unlearn a premium that has stopped paying. It addresses issue 7: the low-risk tilt learned on 2004–2019 data reversed after 2020.

**Hypothesis.** β is estimated from equally weighted means of r̃ and Σ̃ over every month since 2004, so a 2005 month counts as much as a 2019 month. When a premium decays or reverses, the model reacts slowly. Giving recent years more weight should cut exposure to decaying themes sooner.

**Change.** Weight each past year's sufficient statistics by an exponential half-life h:

```latex
\bar{\tilde r}_y = \frac{\sum_{s<y} \delta^{\,y-s}\, \tilde r_s}{\sum_{s<y} \delta^{\,y-s}}, \qquad \bar{\tilde \Sigma}_y \text{ likewise}, \qquad \delta = 0.5^{1/h}
```

The half-life h ∈ {3, 5, 10, ∞} years becomes one more hyperparameter, picked each December with (g, p, λ) on cumulative validation utility. The choice is therefore out of sample, and h = ∞ reproduces the current model exactly, so the comparison is clean.

**Where it lands in your code.**

| Change | File |
| --- | --- |
| Discounted running sums in place of plain sums | `coefs_from_sums` and `december_utility` in `portfolio/portfolio_ml.py` (the yearly sums in `YearSums` already make this a few lines) |
| h added to the hyperparameter search and selection | `search`, `select_hypers`; new `portfolio_ml.half_life_years` in `config.py` and `base.yaml` |
| Report the chosen h each year | `outputs/portfolio_ml/*/chosen.csv` |

The hyperparameter search grows fourfold, but it runs on yearly sums, so the cost is small.

**Why not economic feature selection.** Dropping themes by their feature importance was the earlier plan. It is weaker here: permutation importance measures how much the fitted model relies on a theme, not the gain from refitting without it; only two themes are stable across seeds; and combining characteristics lowers trading costs through netting (DeMiguel et al., 2020), so dropping one can raise turnover. Time decay deals with the same problem, a signal that stopped working, without any of these issues.

**Real-world analogue.** Alpha decay is one of a quant desk's standing concerns. Signal-health reports track each signal's rolling contribution, and a model that adapts its estimation window is the standard response. Low volatility was widely described as crowded in the late 2010s and then lagged the 2020–2025 mega-cap rally.

**Metrics.** Chosen half-life by year, active exposure to the low-risk and other themes through time, and utility and information ratio against Iteration 2.

**Kill criterion.** h = ∞ is chosen most years, or the gain is below the 80% bar on 2016–2019. That would mean the reversal was too abrupt for any backward-looking estimator, a finding worth reporting in its own right.

## Closure: go/no-go and monitoring

The project ends the way a quant research cycle does: with a recommendation to an investment committee and a plan for monitoring the strategy live. "Do not launch" is a legitimate outcome; the guideline does not require beating the benchmark.

**Go/no-go criteria.** The thresholds are illustrative; agree them before the baseline fixes and record them in the research log.

| Criterion | Threshold | Measured on |
| --- | --- | --- |
| Net information ratio | ≥ 0.3 | 2020–2025 |
| Probability of beating the index (Table 3 test) | ≥ 80% | 2020–2025 |
| Deflated Sharpe ratio | Probability > 0.95 | All trials in the research log |
| Realised tracking error | Within ±20% of budget | 2020–2025 |
| Capacity (AUM where net IR falls below 0.3) | ≥ $10B under every cost model | Iteration 2 wealth grid |

**Monitoring script** (deliverable 3, `scripts/11_monitor.py`). It runs monthly on the latest data and writes a one-page dashboard, which doubles as the slide-appendix screenshot. Each check traces back to an iteration or a stated limitation:

| What it checks | Source | Flag when |
| --- | --- | --- |
| Active return and information ratio, rolling 36 months | Iteration 1 | Rolling IR below 0, or drawdown beyond 2 × the tracking-error budget |
| β and theme exposures of the active book | Iterations 1, 3 | Any exposure outside its limit |
| Rank IC of the signals, rolling 12 months | Iteration 2 | IC below 0 for 6 consecutive months |
| Realised against predicted trading cost | Iteration 2 | Realised above predicted for 3 consecutive months |
| Chosen half-life and the low-risk exposure | Iteration 3 | Half-life jumps by more than one grid step in a year |
| Bias statistic of the tracking-error forecast | Issue 8 | Outside 1 ± √(2/T) |

The Fed's SR 11-7 guidance asks for this kind of ongoing monitoring of a model in production. It also answers the question every allocator asks before funding: what would make you switch this off?

## References

- Jensen, T. I., Kelly, B., Malamud, S. and Pedersen, L. H. (2026). Machine Learning and the Implementable Efficient Frontier. *Review of Financial Studies* 39(10), 3035–3078. [doi:10.1093/rfs/hhag022](https://doi.org/10.1093/rfs/hhag022)
- Almgren, R., Thum, C., Hauptmann, E. and Li, H. (2005). Direct Estimation of Equity Market Impact. *Risk* 18(7).
- Bailey, D. H. and López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40(5).
- Clarke, R., de Silva, H. and Thorley, S. (2002). Portfolio Constraints and the Fundamental Law of Active Management. *Financial Analysts Journal* 58(5).
- DeMiguel, V., Martín-Utrera, A., Nogales, F. J. and Uppal, R. (2020). A Transaction-Cost Perspective on the Multitude of Firm Characteristics. *Review of Financial Studies* 33(5).
- Grinold, R. C. and Kahn, R. N. (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
- Harvey, C. R., Liu, Y. and Zhu, H. (2016). …and the Cross-Section of Expected Returns. *Review of Financial Studies* 29(1).
- McLean, R. D. and Pontiff, J. (2016). Does Academic Research Destroy Stock Return Predictability? *Journal of Finance* 71(1).
- Menchero, J., Orr, D. J. and Wang, J. (2011). *The Barra US Equity Model (USE4): Methodology Notes*. MSCI.
- Board of Governors of the Federal Reserve System and OCC (2011). *SR 11-7: Supervisory Guidance on Model Risk Management*.

Replication figures are from the repo's `outputs/` folder as of 8 October 2026, plus the attribution and 2021–2025 numbers computed from those files during the review. References other than the paper are cited from memory; check volume and issue details before submission.
