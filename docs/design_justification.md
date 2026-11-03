# Replication design: what we changed and why

This note justifies every place our replication departs from Jensen, Kelly, Malamud and Pedersen (2026) ("JKMP")
or from the authors' R code. The full technical log is `docs/deviations.md` (entries D1–D38, R1–R13); this
document groups those entries and explains each one in terms of the decision a real investment team would face.

Each decision is judged on three questions:

1. **Constraint.** Was it forced by data, licensing or compute?
2. **Realism.** Does it match how a large institutional investor would actually run the strategy?
3. **Evidence.** Does the replication still reproduce the paper where the change should not matter?

The short answer to question 3 comes first, because it underpins everything else.

## The changes did not break the replication

Over the paper's own test window (1981–2020, 480 months, \$10B investor, γ = 10), the parts of the pipeline that do
not depend on our scale-down reproduce JKMP's Table 2 closely:

| Check | JKMP | Ours |
|---|---|---|
| Market: return / volatility / net Sharpe | 0.08 / 0.15 / 0.51 | 0.078 / 0.149 / 0.51 |
| Factor-ML utility (cost-blind long-short, wiped out by costs) | −2.49 | −2.64 |
| Markowitz-ML: highest gross Sharpe, catastrophic net | gross SR 2.06, net "−" | gross SR 1.00, net R −41 |
| Static-ML (one tuning layer): negative utility | −0.165 | −0.011 |
| Static-ML\* and Portfolio-ML: positive utility | 0.030 / 0.086 | 0.018 / 0.019 |
| Most important theme for Portfolio-ML after costs | Quality | Quality |
| Short-term reversal: important before costs, not after | Yes | Yes |

The same data, cost model and accounting feed every method, so the close match on Market and on the cost-blind
methods shows the plumbing is right. The remaining gaps (Portfolio-ML's magnitude, and its margin over
Static-ML\*) are in the model inputs, which is where our scale-down acts.

---

## 1. Scale-down decisions

### 1.1 Investment universe: top 80 (baseline) and top 250 (final) US stocks, not ~650 NYSE stocks (D2, D3, D35)

| | JKMP | Ours |
|---|---|---|
| Universe | NYSE stocks above median market cap: min 184, median 646, max 805 stocks | Top 80 by market cap (baseline); top 250 (final version, trial 3) |
| Stability rule | Enter after 12 valid months, leave after 12 invalid months | Rank buffer, reconstituted each December: top 80 with entry at rank 70 and exit at rank 100; top 250 with entry 220, exit 300 |

**Constraint.** Every month the method builds an N × N risk matrix, solves a fixed-point problem for the trading-speed
matrix m, and computes Portfolio-ML moments for each candidate hyperparameter setting. Cost grows faster than
N, and the walk-forward runs every year from 1971 to 2020. A top-80 universe made the full paper timeline feasible
on a laptop.

**Realism.**
- JKMP's investor has \$10B. At that size the realistic investable set is large caps: a \$10B allocation to a
  US equity strategy is usually run as a large-cap mandate (an S&P 500-style sleeve) because the smaller half
  of the market cannot absorb the positions. JKMP say the same thing: larger investors "must trade more slowly and
  focus more on liquid stocks" (Sec. 6.2), and the methods are "most relevant for investors large enough to have a
  staff" (footnote 3).
- A rank buffer with an annual December reconstitution is how index providers and index-tracking funds manage
  membership: a stock is not dropped the moment it slips a few places, which avoids paying to sell it and buy it
  back. Our buffer cut name turnover to about 6% a year.

**Evidence.**
- JKMP show their result holds among large stocks: in the largest size group (80th–100th percentile of NYSE stocks,
  Sec. 6.5, Fig. 8) Portfolio-ML remains the best method. Our universes sit inside that group, so we test the claim
  where the paper says it should hold.
- Breadth was widened on evidence, not by assumption. Trial 3 was pre-registered on the fundamental law of active
  management (IR ≈ skill × √breadth): rank IC is higher among stocks ranked 81–250 than in the top 80 (0.043 vs 0.031,
  1971–2020), and widening to 250 stocks raised the net information ratio from 0.42 to 0.67 (P = 98.4%) and capacity
  from \$31B to \$107B.
- We stopped at 250 deliberately. A 500-stock run was feasible (about 13 minutes) but would have been a new,
  un-pre-registered trial before the holdout; the trial count was capped instead. The 300 in "entry 220 / exit 300" is
  the exit buffer of the 250-stock universe, not a third universe size.

### 1.2 Exchanges: NYSE and Nasdaq, not NYSE only (D1)

**Why JKMP exclude Nasdaq.** Nasdaq stocks enter CRSP in 1972 (AMEX in 1962), which creates artificial jumps in the
stock count and therefore in turnover and trading cost.

**Why we keep it.**
- *Realism.* Today's largest US companies, including Apple, Microsoft, Nvidia, Amazon, Alphabet and Meta, are listed
  on Nasdaq. A NYSE-only large-cap universe would exclude them; no real large-cap mandate would.
- *Evidence that the paper's concern does not apply to us.* Checked on our 1952–2025 data: Nasdaq stocks have no
  dollar volume before 1983, so our liquidity screen keeps them out until then, and no Nasdaq stock enters the
  top-80 universe before December 1989. Name turnover is 8.5%, 8.3% and 5.5% a year in 1966–70, 1971–75 and 1976–80,
  with no jump in 1972–73. NYSE-only throughout would have removed 42 names (about 15 of the 80 each year in
  2001–2020, including the large technology firms) from the test window.

### 1.3 Characteristics: 18 in 6 themes, not 115 in 13 (D5, O5)

**Constraint.** Data volume (the raw pull, the ranked panel and the Fourier-feature inputs all scale with the number
of characteristics) and laptop compute.

**Realism.**
- Production quant teams run deliberately parsimonious signal sets. Every signal must be explained to an investment
  committee and monitored, and model-risk governance (for example the US supervisory guidance on model risk
  management, SR 11-7) favours models whose inputs can be validated individually. Three signals per theme keeps the
  theme-level feature importance readable.
- The 18 were chosen **in advance**, not by back-testing returns, which avoids the selection bias that comes from
  picking characteristics on the same data used to evaluate them.

**Evidence.**
- All 18 are among JKMP's 115; each theme assignment and sign was checked against the published JKP files
  (`Cluster Labels.csv`, `Factor Details.xlsx`; deviations O5).
- The set spans both fast signals (short-term reversal) and slow ones (value, quality), so the paper's central
  mechanism can still be tested: trading costs push a large investor away from fast signals and towards persistent
  ones. It replicates (quality first after costs; short-term reversal matters only before costs).
- Portfolio-ML is not starved of inputs: it chose 128 or 256 random features in 33 of the 40 test years.

**Cost, stated openly.** Magnitudes are about a quarter of JKMP's (Portfolio-ML utility 0.019 vs 0.086); only 6 of 13
themes can be analysed; and with few inputs the linear Portfolio-ML matches or beats the random-feature version.

### 1.4 Risk model: 6 sector groups estimated on the top 500 stocks (D6, D7, D18)

| | JKMP | Ours |
|---|---|---|
| Industry factors | 12 Fama-French industries | 6 sectors merged from the same 12 (consumer, industrials, tech & telecom, energy & utilities, health, finance) |
| Theme factors | 13 | 6 (follows from D5) |
| Factor estimation universe | The portfolio universe | The top 500 stocks by market cap |

**Constraint.** With 80 stocks, several of the 12 industries hold only one to three stocks in a given month, so their
daily factor returns would be estimated from almost no data, and a 25-regressor daily cross-section is unstable.

**Realism.** Commercial risk models work this way. JKMP model theirs on MSCI Barra, which estimates factor returns
on a broad estimation universe and applies them to whatever portfolio is being risk-managed. Grouping industries
into a handful of sectors mirrors how risk is reported and limited on most desks (sector-level limits, GICS
sectors).

**Evidence.** The risk model's calibration is in the normal range for this kind of model: realised over predicted
volatility of 0.80–0.87 for universe portfolios in 2000–2019, in line with the same daily-to-monthly scaling the
authors' R code uses. When an idiosyncratic variance cannot be estimated, the size-group median is pooled over the
top 500; this was never triggered (0 of 20,489 stock-months).

### 1.5 Factor-ML: quintiles, not deciles (D25)

With about 80 stocks, decile portfolios hold 8 stocks per leg, which is too concentrated to measure a signal (a
real long-short desk would not run 8-name books against a 10% sort). Quintiles hold about 16 per leg. Factor-ML's
utility still matches JKMP's (−2.64 vs −2.49), so the change does not affect the paper's conclusion that cost-blind
sorts are not implementable.

### 1.6 Hyperparameter grids extended at the edges (D33)

Portfolio-ML's number of random features was widened from {64 … 512} to {8 … 512}, and Static-ML\*'s
return-shrinkage parameter from {0.25, 0.5, 1} to {0, 0.1, 0.25, 0.5, 1}. With fewer characteristics and stocks, the
best setting could sit below the paper's grid; standard ML practice is to extend a grid until the chosen values are
not stuck at its edge. The added values use the same random draws (no new inputs), and edge frequencies are
reported with the baseline (`outputs/tables/grid_edges.csv`): Portfolio-ML picked the smallest value in only 8% of
years.

---

## 2. Data-source substitutions (forced by data availability)

| # | JKMP / R | Ours | Real-world reason |
|---|---|---|---|
| D11 | Legacy CRSP daily file built by JKP's SAS code | CRSP version 2 daily file (`crsp.dsf_v2`) | WRDS has moved to CRSP v2; the legacy daily file ends 2024-12-31 and the authors' intermediate file is not public. Any team replicating today faces the same switch |
| D17 | Share codes 10/11/12 from legacy CRSP | Same screen, a no-op in CRSP v2 (all JKP rows map to code 10) | CRSP v2 dropped legacy share codes; keeping code 12 (foreign-incorporated firms) matches the paper anyway |
| D15 | JKP market returns file | Kenneth French's market excess return + risk-free rate | Same concept (value-weighted CRSP market), the industry and academic standard, already in our data |
| D16 | Monthly returns rebuilt from CRSP; missing returns inside a stock's life set to 0 | JKP's own monthly excess returns; rows with a missing return are screened out | Fewer moving parts; avoids treating a missing return as a zero return |
| D12 | No explicit filter (relies on JKP's input file) | Require JKP's common, primary-security and main-observation flags | Guarantees one main share class per company, as a portfolio system would hold |
| D13 | Risk-free rate joined one month stale | Risk-free rate of the month the return is earned | Matches how JKP compute excess returns; verified against CRSP daily (`tests/test_timing.py`) |

---

## 3. Corrections and realism improvements

| # | JKMP / R | Ours | Why it matters in practice |
|---|---|---|---|
| D14 | Wealth rolled back with a first-order approximation | Exact recursion w(t−1) = w(t)/(1 + R(t)), as the paper's text states | R's approximation puts the investor at 0.37× the correct wealth at the 1981 test start (\$0.046B vs \$0.125B), so it understates trading costs for most of the sample. A fund's cost budget must be computed at its true size |
| D19 | Selling a stock that leaves the universe is free | Charged at the normal impact cost if the stock still trades (a delisted stock is settled at its delisting return) | Index deletions are real trades: funds pay market impact when they sell a removed stock, often alongside every other index fund. Effect is small (e.g. +0.0035 a year for minimum variance in the old test) and can be switched off to match R |
| D20 | Risk-free rate of the formation month in the trading-speed matrix | Risk-free rate of the return month | Consistent with D13 |
| D9 | \$10B by end-2020, rolled back only | Same anchor, also rolled forward for 2021–2025 | Needed so the holdout investor keeps the same relative size |
| D21 | Iterative ridge solver (glmnet) | Exact closed form for the same objective | Same answer, faster and deterministic |
| D22 | R's random-number generator | NumPy's generator, same structure | The draws cannot match across languages; results depend on draws only through noise |
| D26, D27, D29 | Hyperparameters applied to formation-month years | Applied to return-month years; same validation start (1971) and schedule | One consistent year convention across the return model and every method, with no look-ahead (tested by scrambling future returns) |

Where the paper's text and the R code disagree, we follow the R code (deviations R1–R13), because the code is what
produced the published tables. Examples: realised utility and net Sharpe use gross-return variance (R4, R11); the
Static-ML\* cost parameter multiplies Λ by k rather than dividing by φ (R6); the random-feature scale is a variance,
not a standard deviation (R1).

---

## 4. Scope: what we did not replicate

| Not replicated | Why | Effect on conclusions |
|---|---|---|
| Multiperiod-ML (12 return horizons) | Needs 12 return models; compute | Portfolio-ML, the paper's preferred method, and Static-ML\*, its strongest benchmark, are both replicated |
| Rank-ML | Low priority: like Factor-ML, a cost-blind sort | Factor-ML already shows cost-blind sorts fail |
| Short-selling fees (Sec. 6.4) | Markit lending data not available to us | JKMP find the effect modest (−0.5% a year for Portfolio-ML, almost fully offset by lending income) |
| Size groups (Sec. 6.5) | Our universe is one size group (the largest) | We cite JKMP's own size results to justify our universe |
| Simulations (Internet Appendix E) | Out of scope for an empirical replication | None for the empirical results |

---

## 5. What we added beyond the paper

These are not deviations from the replication; they are additions a real allocator would require before trusting
the strategy:

- **A locked holdout, 2021–2025, run once** (D4, D37). JKMP's sample ends in 2020; we evaluate on five years the
  authors never saw. In practice this is the "paper-trading" period before capital is committed.
- **Pre-registered iterations with a trial count and a deflated Sharpe ratio** (`docs/research_log.md`). Each
  change was written down with its keep and kill rule before it ran, much as an investment committee signs off
  a model change before it goes live, and the final result is discounted for the number of things tried (DSR = 0.90).
- **Feature importance over 20 permutation seeds, not one** (D31). With a single seed, the sign of most themes'
  importance depended on the draw.
- **A benchmark-relative version with a tracking-error budget** (D34). A \$10B large-cap mandate is usually judged
  against an index with a tracking-error limit, not on total-return utility.
- **Monitoring and macro stress tests** (D37, D38): monthly risk flags for a live fund, and Fed-policy scenarios
  linked to the 2026 policy turn (`docs/macro_stress.md`).

---

## 6. Limitations we accept

- **Magnitudes.** Portfolio-ML's utility is about a quarter of JKMP's (0.019 vs 0.086), and its net Sharpe is 0.61 vs
  1.33. Fewer characteristics and fewer stocks both reduce the signal available; with one run we cannot say how much
  each contributes.
- **Portfolio-ML vs Static-ML\*.** JKMP find Portfolio-ML better with 99.9% probability; we find 55%. Our scale-down
  leaves too little signal to separate the two methods.
- **Theme coverage.** Feature-importance conclusions cover 6 of the 13 themes.

None of these change the paper's qualitative conclusions that we do reproduce: trading costs decide which method
wins, cost-blind ML portfolios are not implementable for a large investor, and persistent characteristics such as
quality matter most after costs.
