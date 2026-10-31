# Exclusions, assumptions, changes and limits

This comparison reads saved outputs. It does not fit, tune, retrain, change or
repair a model. Its new audit files are separate from every source result.
The working-tree file lists the state at audit time, including the newly added
comparison code. Existing tracked modifications in keats_pipeline/README.md,
apply_history_policy.py, configs/us_mid/base.yaml, scripts/13_breadth.py,
src/jkmp/portfolio/recursion.py and tests/test_execution_safety.py predate this
comparison and must not be represented as comparison changes. Source checksums
record the exact saved predictions, covariance pieces, configs and weights read.

## Existing input assumptions and exclusions

- Missing next-month held-stock excess returns were explicitly set to zero in
  the pre-existing workflow. Their total return equals the next-month USD
  risk-free return. They are assumptions, not observations. Every available
  occurrence is copied to `all_zero_return_assumptions.csv`; the prepared-panel
  summary separates complete history from the evaluated test months. Dataset
  variants can share the same economic event; rows must not be summed as unique
  source events across variants. Unfilled broad-training missing values are not
  fabricated by this evaluation.
- Unavailable required historical signal stock/months were excluded in the
  pre-existing workflow rather than assigned invented history. Exact IDs,
  missing history and formation/return dates are copied to
  `all_history_exclusions.csv`. No further security exclusion is imposed here.
  Applied-ledger coverage is separately saved in `history_exclusion_coverage.csv`.
  The raw source history files document three missing security/months: US IDs
  88818 and 91135, and European ID 320230501, all December 2010. At the audited
  snapshot, the saved US-mid250 ledger contains two applied December 2011
  exclusions; EU250 has not been prepared and has no applied exclusion ledger.
  Therefore the applied count is two, not three. The documented European source
  gap is retained in `all_unavailable_lag_history_inputs.csv`, with no invented
  applied portfolio event. Earlier README policy descriptions are not evidence
  that an unrun preparation stage has completed.
- Features were previously average-percentile-ranked over the broad screened
  same-region population; missing feature ranks were assigned 0.5. All 50
  characteristics and the saved memberships are retained. Supplementary lagged
  history is used only as lagged information, not additional fitted targets.
- US common-stock screens include NYSE and Nasdaq (CRSP 1 and 3), share codes
  10/11/12, primary/common/observed/main-exchange flags, positive market cap and
  dollar volume, industry/history and characteristic coverage. US mid-cap is
  centred on market-cap ranks 461-540. European large-cap spans developed
  Europe, not UK-only. Buffers and exits make monthly counts vary.
- Equity returns and investor wealth are USD. European daily risk inputs use
  Compustat Global price/dividend/FX reconstruction and JKP security matching;
  US risk inputs use CRSP CIZ total returns with its delisting treatment.
  Original provenance and reconciliation files remain authoritative. Daily risk
  coverage/reconciliation does not prove every monthly delisting payoff is
  correct. Missing held forward returns remain the explicit zero assumption.
- A stock disappearing from the panel is treated by original accounting as
  settled through its reported terminal return, with no additional sale. This
  assumption is not independently verified against terminal cash proceeds.
  Exit cost rows are retained only where an exiting security still has a panel
  row and saved Lambda. Shorting fees, borrow constraints, fees, taxes and
  unmodelled execution costs are absent.

## Actual timing and estimation

- All date labels refer to earned returns: formation December 2019 earns
  January 2020. Declared portfolio validation is 2015-2019; reported test/OOS
  is 2020-2025. Historical signals/risk need earlier warmup.
- Return-model fitting targets begin January 2005. It refits yearly. For
  predicted return year y, its five-year tuning validation is y-5 through y-1;
  the earlier expanding rows train the candidate coefficients, and selected
  coefficients refit through December y-1. Thus the 2015 forecasts initially
  train on 2005-2009, validate 2010-2014 and refit on 2005-2014. For 2020 this
  becomes candidate training 2005-2014, validation 2015-2019, refit 2005-2019.
  A simplistic fixed training 2005-2014 label does not describe all annual fits.
- Static-ML* and Portfolio-ML select each following year's hyperparameters
  from cumulative saved realised validation utility ending in the previous
  December. Earlier test observations can enter later walk-forward fits and
  selections as they become available. This is sequential OOS, not a frozen
  untouched holdout. Saved date/year separation is audited; exact historical
  model execution timestamps and vintage vendor data are not retained.
- Covariance uses prior/current formation-date information, factor daily
  return rows dated no later than eom, a 2,520-trading-day window, EWMA
  correlation half-life 378 days and variance/idiosyncratic half-life 126 days.
  Daily estimates are multiplied by 21 for monthly units. Missing idiosyncratic
  volatility is filled by existing size-group/overall medians. Exact saved
  positive definiteness and conditioning are checked on each evaluated month;
  this does not establish that the forecast is well calibrated.
- No additional unseen holdout follows December 2025. No post-2025 result or
  independently verified point-in-time publication vintage is available.

## Accounting conventions and independent checks

- r is saved monthly gross EXCESS return; tc is a positive monthly fraction
  of investor wealth. Net excess is r-tc. Turnover is sum(abs(w-w_start)),
  the full L1 trade amount, not conventional half-turnover. Gross leverage is
  sum(abs(w)); shorting is sum(max(-w,0)). Trading cost is
  0.5 * wealth_t * sum(Lambda_i,t * (w_i,t-w_start_i,t)^2).
- Funded-growth diagnostics add the saved next-month risk-free return:
  gross_total=r+RF and net_total=r-tc+RF. The audit flags growth <=0 using those
  total returns. A separate synthetic excess-index flag r-tc<=-1 does not by
  itself establish funded bankruptcy and must not be counted interchangeably.
- Independent reconstruction checks gross return, cost, turnover, leverage and
  shorting against saved weights and exact saved inputs to tolerance 1e-10.
  Inherited weights are checked as prior target*(1+stock total return)/
  (1+screened regional market total return). The first saved month is excluded
  from that drift comparison because its prior state may predate saved weights.
- Fixed Static-ML returns are saved but its fixed-parameter weights are not
  retained; Static-ML* has separate saved weights. Fixed Static-ML accounting,
  concentration and weight comparisons are marked unavailable, not inferred
  from Static-ML* weights.
- The screened regional value-weighted market is the wealth growth proxy,
  anchored to USD10bn at end-2019. It is not an external investable index.
- Paper Table 2 volatility, net Sharpe denominator and variance penalty use
  GROSS return standard deviation/variance. Annual means/costs/utility use 12;
  volatility and Sharpe use sqrt(12). Sample variance uses n-1. The Table 3
  paper utility-flow centring has a finite-sample n versus n-1 distinction,
  which the statistical module must identify separately.

## Differences and inference limits relative to Jensen et al.

These small buffered 80/250-stock US-size and European universes, 50 features,
2005-start fitting, five-year return-model tuning window, 2020-2025 test and
USD10bn end-2019 wealth path differ from the paper sample and investment
universe. Raw-characteristic linear Portfolio-ML, benchmark-relative active
Portfolio-ML, half-life, regime and breadth experiments are extensions. RFF
complexity grids and computational choices may differ; effective configs
and hyperparameter dictionary record their actual meanings. Findings are
descriptive tests of this saved implementation, not a numerical replication
of the paper's levels. Across-region/size gaps confound universe, liquidity,
signals, risk estimates and return paths; they cannot identify causal model
quality. Multiple variants sharing inputs are not independent replications.

Missing outputs, unavailable seed return paths, multi-horizon forecasts,
alternative covariance fits, original fixed Static-ML weights and fresh
holdout tests must be SKIPPED rather than simulated or filled. No result
selection using this report should be represented as new untouched OOS.
