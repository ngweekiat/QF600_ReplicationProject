# Macro stress tests: Fed policy scenarios (evaluation, not a trial)

**Status: pre-registered 2026-11-03, before any scenario number was computed.** Results are appended below
the line at the end. Nothing here changes a model: every strategy is the frozen version from
`docs/research_log.md` (baseline methods and the final kept version, trial 3). These tests are **not trials**:
they do not enter the trial count or the deflated Sharpe ratio, and no result may feed back into a model choice.

## Question

The paper's cost-aware methods assume trading costs and risk are stable (JKMP Sec. 5.1.3: "the investor behaves
as if trading costs and risks are constant over time"). Monetary-policy turns are when that assumption is weakest.
Recent events: the FOMC raised the target range by 25 bp to 3.75–4.00% on 2026-09-17 (the first hike after the
2024–25 cuts), held on 2026-10-28 with two dissents for a hike, and futures price a December 2026 hike at
roughly 80% (CME FedWatch, 2026-10-29). How did, and how might, our strategies behave in each situation?

## Data

- **Policy decisions:** FRED `DFEDTAR` (target, 1982-09-27 .. 2008-12-15) spliced with `DFEDTARU` (upper bound of
  the target range, 2008-12-16 onward), cached in `data/raw/`. Month-end target T_m; monthly change
  ΔT_m = T_m − T_{m−1}; a month is a **hike** if ΔT_m > `move_threshold`, a **cut** if ΔT_m < −`move_threshold`,
  else a **hold**. Events start 1982-10 (first full month of target data).
- **Strategy returns:** the saved monthly returns of each frozen method, 1981–2020 (test) followed by 2021–2025
  (holdout, already run once). Return months only; nothing is re-fitted.
- **Final version's current book:** its last formation month, 2025-11 (active weights a vs the top-250 VW
  universe), with Σ's factor loadings X and idiosyncratic variances at that month.
- **Factor returns:** the daily sector and theme factor returns of the risk model (top-500 regressions), summed to
  calendar months.

## Event definitions (fixed before running)

prev_move(m) = the most recent non-hold month strictly before m.

| Scenario | Today's event | Historical analogue month t |
|---|---|---|
| **S1 First hike after cuts** | 2026-09 | hike in t and prev_move(t) = cut |
| **S2 Hold right after a hike** | 2026-10 | hold in t and hike in t−1 |
| **S3 The next decision** (Dec 2026, P(hike) = 0.8) | 2026-12 | each S2 month t, split by its outcome: **hike** if any hike in t+1 .. t+2, else **no hike** |

Windows (return months): S1 and S2 start at the event month t (the decision month's own return) for horizons
h ∈ {1, 3, 6}. S3 starts at t + 2 (the December analogue of an October hold) and lasts 3 months (Dec–Feb).
Only complete windows inside the saved return history are used. Overlapping windows are kept and counted.

## Two lenses

1. **Historical lens (actual strategy returns).** For each analogue window: cumulative net return Σ(r − tc) and
   cumulative trading cost Σ tc for Market, Portfolio-ML, Static-ML\*, Markowitz-ML, Factor-ML; for the final
   version, cumulative net active return vs the paper index (top-250 VW, no benchmark cost) and the paper index's
   own excess return. Paper link: the ratio of each method's monthly tc inside the windows to its average tc.
2. **Current-book lens (forward-looking).** The 2025-11 book's factor exposures b = Xᵀa are applied to the factor
   returns of each analogue window: gross active return = Σ_m bᵀf_m, plus idiosyncratic noise
   N(0, h·Σ ivar·a²). Net = gross − the final version's mean monthly tc in the same windows × h.
   Distribution: 10,000 draws, an analogue window drawn with replacement each time (seed = config seed).
   **S3 forecast** = a mixture: each draw takes the "hike" outcome set with probability `p_hike` = 0.8, else the
   "no hike" set. Reported: mean, sd, 5th / 50th / 95th percentile, P(net active < 0), expected shortfall at 5%.

## Caveats stated in advance

- Few analogues (expected roughly 5–15 per set); results are descriptive ranges, not significance tests.
- The current book is from 2025-11, ~11 months before the 2026 events: the lens measures how exposed the strategy's
  typical book is, not today's fund. Our return data end 2025-12, so no actual 2026 strategy return is shown.
- The 0.8 is the market-implied probability, an input; nothing here forecasts the Fed.
- The factor model covers priced factor risk only; stock-specific risk enters as noise.
- 1982–1987 `DFEDTAR` values are the Fed staff's reconstructed targets.

## Addendum A: rate sensitivity and liquidity stress (pre-registered 2026-11-03, before any number below was computed)

Same status: evaluation of the frozen final version, not a trial. Motivation: markets react to the *surprise* in a
decision, not the decision. With a December hike priced at 80%, the shock is mostly a no-hike.

**A1. Rate sensitivity.** Data: FRED `DGS2` (2-year Treasury yield, daily) → month-end yield; Δy_m = change on the
previous month-end, in percentage points. For each factor k of the risk model (6 sectors, 6 themes), OLS of the
monthly factor return on Δy with an intercept gives β_k (return per 1 pp). The current book's active rate
sensitivity is s = Σ_k b_k β_k, with b = Xᵀa of the 2025-11 book. Estimation-error variance bᵀCov(β̂)b, with
Cov(β̂) = residual factor covariance / Σ(Δy − mean Δy)². The paper index's own sensitivity comes from the same
regression on its monthly excess return. Windows: **1982-10 .. 2025-12** (headline, same span as the policy events)
and **2000-01 .. 2025-12** (robustness).

**Shocks.** December surprise from the market-implied probability p = 0.8 and a 25 bp move: a hike surprises by
+(1 − p)·25 = **+5 bp**, no hike by −p·25 = **−20 bp** (the probability-weighted surprise is zero by construction).
Plus a **+100 bp** rate shock. Each shock moves the 2-year yield one-for-one (stated assumption; empirical
pass-through below one would scale every result down proportionally). For each shock: expected one-month active
return s × shock, and a 90% range adding the factor residual risk bᵀΣ_res b, the idiosyncratic active variance and
the estimation-error variance; the index's expected return; total ≈ index + active.

**A2. Liquidity stress: cost to exit.** For the 2025-11 book, at the fund's wealth that month W and the book's own
Λ_i (= 0.2 / dollar volume, as in every cost in this project):
- *exit the active book* (trade back to the top-250 VW index): cost / W = ½ W Σ_i k Λ_i a_i²;
- *liquidate everything to cash*: the same with a_i replaced by the full weight w_i.

Cost multipliers k ∈ {1, 2, 3} (stressed liquidity), executed at once or split evenly over 3 months (quadratic
cost: 1/3 of the one-shot cost if volume is unchanged). Also at the frontier wealth levels (\$1B–\$250B by end-2020,
rescaled by the same wealth path). Compared with the final version's annual net active return (1981–2020: 2.02%;
2021–2025: 0.63%).

**Caveats.** Linear rate betas from monthly data (FOMC and non-FOMC moves mixed); the 2-year yield proxies the
policy path; dollar volume is from 2025-11 and falls in real liquidity crises (the multipliers stand in for that).

---

## Results (run 2026-11-03; `scripts/16_macro_stress.py` → `outputs/macro_stress/`, `outputs/figures/macro_stress.png`)

**Classification check.** The rules label the actual 2026 path correctly: 2026-09 = S1 (hike, previous move a cut),
2026-10 = S2 (hold after the hike), and the December outcome is unknown (not yet observed). Analogue months
1982–2025: S1 14, S2 44, S3 "hike" 33, S3 "no hike" 11 (ignoring 2026). S1 includes the 1980s fine-tuning
reversals (1983-03 .. 1988-03, 8 of 14), then 1994-02, 1997-03, 1999-06, 2004-06, 2015-12, 2022-03.

**Historical lens, mean 3-month cumulative return (percentage points).** Final = net active vs the paper index.

| Scenario | n | Final (net active) | Paper index | Portfolio-ML (net) | Static-ML\* (net) | Market top-80 (net) |
|---|---|---|---|---|---|---|
| S1 first hike after cuts | 14 | **+1.52** (median +0.75; 4 of 14 < 0) | −0.67 | +1.35 | +0.53 | −0.43 |
| S2 hold after a hike | 44 | +0.57 | +2.07 | +1.04 | +0.99 | +2.11 |
| S3 next decision: hike | 33 | +0.50 | +1.69 | +0.85 | +0.80 | +1.83 |
| S3 next decision: no hike | 11 | −0.23 | +7.08 | +0.78 | +1.78 | +6.37 |
| **S3 mixed, P(hike) = 0.8** | — | **+0.36** | +2.77 | +0.84 | +1.00 | +2.74 |
| *Unconditional: mean monthly × 3, 1981–2025 (540 months)* | — | *+0.47* | *+2.10* | | | |

Factor-ML (−81 pp) and Markowitz-ML (−768 pp) lose heavily in every window, as in Table 2: their costs dominate.

*Post-hoc check, not pre-registered:* S1 restricted to the modern cycle turns since 1994 (n = 6): final +0.56
(median +0.15, range −0.50 .. +2.11), paper index −2.31, Portfolio-ML +1.69, Static-ML\* +0.45. The full-sample
S1 mean is lifted by two 1980s windows (1985-08 +6.6, 1986-05 +6.1).

**Paper link: trading cost inside the windows vs each method's average month (`tc_ratio.csv`, 3 months).** After a
first hike, cost-blind Factor-ML trades at 1.33× its usual cost and the Market reference at 1.80×, while the
cost-aware Portfolio-ML (0.92×) and Static-ML\* (0.72×) do **not** trade more; the final version's cost rises to 1.79×
of a small base (its monthly tc is ~0.02%). In S2 / S3 the ratios are ~0.7–1.1 for every method.

**Current-book lens (book of 2025-11; `scenarios_current_book.csv`).** Factor exposures Xᵀa: investment +0.20,
quality +0.12, momentum +0.11, value −0.04, low risk +0.03, energy & utilities −0.03; idiosyncratic active vol
0.23% a month. Net active return over 3 months:

| Scenario | windows | mean | 5th pct | 95th pct | P(< 0) | ES 5% |
|---|---|---|---|---|---|---|
| S1 first hike after cuts | 14 | +0.13 | −0.72 | +0.99 | 39% | −0.94 |
| S2 hold after a hike | 44 | +0.09 | −0.80 | +0.94 | 42% | −1.06 |
| S3 next decision: hike | 33 | +0.03 | −0.85 | +0.90 | 48% | −1.09 |
| S3 next decision: no hike | 11 | −0.01 | −0.71 | +0.70 | 50% | −0.90 |
| **S3 mixed, P(hike) = 0.8** | — | **+0.02** | **−0.83** | **+0.87** | **48%** | **−1.06** |

Gross factor P&L in S3-hike windows (+0.09 pp) comes mainly from quality (+0.08) and momentum (+0.06), offset by
the energy & utilities underweight (−0.10); the low-risk tilt in this book is too small to matter (−0.03).

**Reading.**
1. After a first hike after cuts, the final version has historically **added** active return while the index fell
   (+1.52 vs −0.67 pp over 3 months; +0.56 vs −2.31 on the modern turns only). Its edge is largest when the market is
   weak, as a benchmark-relative quality/momentum tilt would suggest.
2. Pauses are where it lags in total-return terms: when the Fed stops for good ("no hike", e.g. 2000-06, 2006-07,
   2019-01, 2023-08), the index rallied +7 pp in 3 months and the final version's active return was slightly negative.
   With β ≈ 1 it still captures the rally; only the active part lags.
3. For December 2026, the current book's mixed scenario is close to flat (+0.02 pp, 90% band −0.83 .. +0.87 pp over
   3 months): a 2%-TE book has small active risk, and the scenario changes its expected active return very little.
4. Paper link: the cost-aware methods did not raise their trading in policy turns, while the cost-blind methods did;
   Λ's stability assumption (Sec. 5.1.3) did not visibly hurt the cost-aware methods in these windows.

**Limits.** Overlapping windows and few analogues (n = 6 modern S1 turns); descriptive only. The current-book lens
covers factor P&L, not stock-specific alpha, so it is conservative relative to the historical lens. The book is from
2025-11. Nothing here fed back into any model choice.

## Results, Addendum A (run 2026-11-03; `rate_betas.csv`, `rate_scenarios.csv`, `exit_cost.csv`)

**A1. Rate sensitivity.** Factor betas on monthly 2-year-yield changes (return per +1 pp):

| Factor | 1982–2025 (n = 519) | 2000–2025 (n = 312) |
|---|---|---|
| Low risk theme | **−1.38%** (se 0.28) | **−2.23%** (se 0.51) |
| Value theme | +0.29% (se 0.16) | +0.40% (se 0.29) |
| Investment / quality / momentum / short-term reversal | within ±0.13% (not significant) | within ±0.07% (not significant) |
| Paper index (excess return) | −0.07% (se 0.63; R² 0.00) | **+3.77%** (se 1.00; R² 0.04) |

Only the low-risk theme has a clear, stable rate sensitivity: bond-like, low-volatility stocks fall when yields rise.
The index's own sensitivity changes sign across windows: since 2000, yields have tended to rise on good growth news
(positive stock–bond correlation), so all yield moves, not only policy surprises, drive this estimate.

The current book (2025-11) is close to rate-neutral: active sensitivity **−0.02% per +1 pp** (1982–2025; −0.03% on
2000–2025), with an estimation standard error larger than the estimate. Its low-risk exposure (+0.03) is too small
to carry the low-risk theme's rate risk.

One-month scenarios, current book (percentage points; 90% range includes residual factor, idiosyncratic and
estimation risk):

| Scenario (Δ 2-year yield) | Active expected | Active 90% range | Index expected: 1982–2025 / 2000–2025 |
|---|---|---|---|
| December hike, as expected (+5 bp) | −0.00 | −0.52 .. +0.52 | −0.00 / +0.19 |
| December no hike, dovish surprise (−20 bp) | +0.00 | −0.51 .. +0.52 | +0.01 / −0.75 |
| +100 bp rate shock | −0.02 | −0.54 .. +0.50 | −0.07 / +3.77 |

The December surprise moves the active book by well under 0.1 bp in expectation; its one-month risk (about
±0.5 pp) comes from everything other than rates. For the total portfolio, the sign of the rate effect depends on the
stock–bond regime, so no robust direction can be claimed.

**A2. Cost to exit.** The fund held \$18.47B in 2025-11 (\$10B at end-2020 rolled forward with the market). Cost as a
share of the fund, trading at once (spreading over 3 months divides each figure by 3):

| Position traded | Normal costs (k = 1) | k = 2 | k = 3 |
|---|---|---|---|
| Active book → index | **0.36%** (\$66M) | 0.72% (\$133M) | 1.08% (\$199M) |
| Everything → cash | 1.30% (\$239M) | 2.59% (\$479M) | 3.89% (\$718M) |

Against the strategy's net active return (2.02% a year in 1981–2020, 0.63% in 2021–2025), exiting the active book at
once costs 0.18 years of test-period active return, or 0.57 years at the holdout rate; under tripled costs, 0.53 and
1.7 years. Spreading the exit over 3 months brings the normal-cost figure to 0.12% of the fund.

By fund size (end-2020 wealth; same book, active exit at once, normal costs): \$1B 0.04%, \$10B 0.36%, \$50B 1.80%,
\$100B 3.60%, \$250B 8.99%. Cost per unit of wealth rises linearly with size (quadratic impact), the mechanism behind
JKMP's Fig. 1B. These larger-size figures hold the \$10B book fixed; the model at a larger size would hold a smaller,
slower active book (as the capacity analysis of trial 3 does), so they are upper bounds.

**Reading.**
1. The final version's active book is effectively rate-neutral: neither the December decision nor a +100 bp shock
   moves its expected active return by more than a few hundredths of a percentage point. What the December decision
   changes is the *index*, which a benchmark-relative investor holds anyway.
2. The one rate-sensitive building block is the low-risk theme. A manager who wanted to limit rate risk in the
   active book would monitor its low-risk exposure, which is small in the current book.
3. De-risking is affordable at \$10B: exiting the active book at once costs about two months of 1981–2020 active
   return at normal costs (0.18 years; 0.57 years at the weaker holdout rate). Tripled costs spread over 3 months
   cost the same 0.36% of the fund. It becomes expensive at large sizes (\$100B+), the same size effect that drives
   the paper's implementable frontier.
