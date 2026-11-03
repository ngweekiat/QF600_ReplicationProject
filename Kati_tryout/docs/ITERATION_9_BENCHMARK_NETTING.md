# Iteration 9: benchmark-aware recursive cost learning

Starts from the last accepted setup: currently iteration 4, ancestry 1, 2, 4.
Rejected changes 5-8 are absent. Only the cost representation in coefficient
learning changes. Stocks, 50 inputs, data dates, risk estimates, the original
return/risk learning proxies, RFF/ridge grids, annual maximum-utility selection,
wealth, actual impact lambda and monthly execution/accounting remain unchanged.
The 1% ex-ante TE coefficient is recalibrated with pretest data only.

## Derivation and scope

For a FIXED coefficient vector beta, the unchanged execution rule is affine:

```
inherited weights = h_t + H_t beta
target weights    = b_t + Q_t beta
b_t = market_t + m_t (h_t - market_t)
Q_t = m_t H_t + S_t - m_t S_t
```

The next h/H are drifted using the same stock-return/wealth-growth ratio as
the existing accounting. New stocks inherit zero. Charged exits have target
zero and negative inherited base/response orders; no-sale delistings retain
the original rule. Initial inherited base weights are VW and response is zero.

For each current or charged-exit order, define dB=b-h and T=Q-H. Then:

```
TC(beta) = .5 W sum(lambda * (dB + T beta)^2)
         = c0 + c' beta + .5 beta' C beta
c0 = .5 W sum(lambda * dB^2)
c  = W T' diag(lambda) dB
C  = W T' diag(lambda) T
```

This uses the recursive zero-ML baseline, not the fully rebalanced Market
reference's observed trades. To be consistent, the cost quadratic also uses
the exact recursive response T, replacing the earlier finite-lag Delta-Omega
cost proxy. It is not sufficient to append an observed historical cost saving.

Learning retains the original r_tilde and risk proxy but uses:

```
linear reward = original_r_tilde - c
denominator   = original_risk + C
beta(lambda)  = solve(denominator_mean + lambda I, linear_reward_mean)
```

The constant c0 is independent of beta, RFF variance, feature count and ridge
within each fixed gamma run; omitting it from candidate ranking cannot change
that ranking. It is nevertheless retained in all cost-identity audits.

This is an explicit cost-only refinement of our active extension. It does NOT
claim an exact reformulation of the entire paper objective: original finite-lag
return/risk proxies stay unchanged. Exact affine costs apply to constant beta;
annual coefficient jumps and test-start reset costs are charged by the unchanged
actual execution/accounting, not claimed as fully predicted by this learning
proxy. The historical learning baseline starts at the first training formation
month; implemented validation/test books retain their original start/reset dates.
Original RF timing, terminal-return and missing-data conventions are retained.

## Checks and experiment rule

- Synthetic literal-execution tests compare base plus response with the original
  trading function for nonzero/zero beta, changing membership, drift and exits.
- Central finite differences verify the linear cost gradient/sign and 1/2 factor.
- Every historical month checks three coefficient probes against literal orders.
- The zero-ML baseline is independently implemented with the original execution
  function and all its costs are reconciled; both RFF variances must share it.
- Coefficients for year Y use returns only through Y-1. Calibration uses only
  the original usable pretest portfolio returns (2016-2019); no dates change.
- Net performance is independently reconstructed from actual saved stock orders.
- Compare 2020-2025 with iteration 4 using the existing 10,000 paired six-month
  bootstrap samples and 80% P(better paper-net Sharpe) gate. Drawdown adds no gate.
  Already inspected test outcomes are not independent unseen confirmation.

Run `python -X utf8 Kati_tryout/scripts/iteration_9/run.py`.
Implementation: `scripts/benchmark_netting.py`. Outputs:
`outputs/iteration_9/us_large/`, including `cost_audit.csv`, historical baseline
weights/returns for verification, pretest calibration trials, implemented model
weights/returns, inherited settings, decision, charts and summary.

## Completed result: 2020-2025

| Measure | Previous accepted iteration 4 | Iteration 9 |
| --- | ---: | ---: |
| Annual net excess return | 15.13536% | 15.13535% |
| Paper-convention net Sharpe | 0.870264 | 0.870595 |
| Annual actual trading costs | 0.059453% | 0.061866% |
| Worst additive drawdown (percentage points) | -27.422 | -27.378 |

P(better versus iteration 4) is **59.25%**, below 80%; iteration 9 is **not
included**. The paired bootstrap 95% interval for the Sharpe difference is
[-0.001282, +0.002255]. The Sharpe improvement is very small and net returns
are essentially unchanged. Actual costs increase slightly; a more faithful
cost-learning representation does not guarantee lower realised costs or returns
that improve materially. No further settings were tried after seeing this result.

The pretest-only calibration brackets the 1% target with gamma 100 and 250;
the original interpolation rule selects gamma_TE 167.6 (iteration 4: 174.1).
The final accepted ancestry remains **1, 2, 4**, with Base and all prior
experiment portfolios preserved. Code tests, independent order accounting,
cost identities and bootstrap checks accompany the saved report. This is
selected-test evidence, not independent unseen confirmation.
