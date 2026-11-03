# Original proposal: iteration 9, benchmark-aware trade netting

This records the original pre-implementation rationale. The user subsequently
authorised implementation; see [the implemented derivation and scope](ITERATION_9_BENCHMARK_NETTING.md).

The iteration-4 diagnostic is under
`outputs/iteration_4/us_large/diagnostic_for_trade_netting/`.
Its exact stock-level accounting separates benchmark trades, differences in
orders, and their cost interaction. It uses only iteration 4 and Market data.

Economic hypothesis: improve which ML orders are attractive by learning about
their interaction with unavoidable benchmark rebalancing, rather than merely
increasing a global turnover penalty. The actual reported costs already use net
trades and are correct; the hypothesis concerns the benchmark-relative learning
approximation, which does not explicitly include the benchmark cost cross-term.

The proposed refinement must derive the full trading baseline under the existing
partial-adjustment recursion, wealth drift and initialisation. Then incorporate
its interaction with the signal-driven order basis in the coefficient-learning
objective. Do not simply append the empirical Market interaction observed in the
test period, alter assumed lambda, subtract historical costs, or claim the saved
netting benefit a second time. Preserve the existing quadratic trading-cost form.

Start from iteration 4 (ancestry 1, 2, 4), without rejected changes 5-8. Keep
stocks, 50 characteristics, train/validation/test duration, risk estimator,
actual impact parameter, investor wealth, RFF/ridge grids and annual refit timing.
Recalibrate the existing 1% ex-ante TE target using pretest data. Use the declared
80% P(better net Sharpe) comparison against iteration 4 on the same test history.
Count this as a new research trial; it is not independent unseen confirmation.

This is a refinement of our active extension, not an assertion of exact paper
replication. Its potential benefit is a hypothesis. The observed active gross
edge is small and uncertain, so correct cost learning may still fail to improve
net returns. This historical proposal is superseded by the implemented protocol
linked above; the diagnostic itself contains no iteration-9 outcomes.
