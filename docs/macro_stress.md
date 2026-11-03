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
