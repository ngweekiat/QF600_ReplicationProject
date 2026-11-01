# Assumptions, exclusions and changes

The base report contains only original non-starred Static ML and default Fourier-signal Portfolio ML for US large cap, US mid cap and European large cap. The original 50 characteristics, portfolio memberships, return assumptions, risk inputs, wealth paths and expanding annual timing are unchanged.

Prior backup material remains in `C:/Users/katiravan/Downloads/keats_removed_models_20261101`. Only the base is restricted to these two strategies. The three iterations have been restored separately with their original existing results and completion status. Historical preparation provenance remains a record of the earlier preparation, not active settings. Each group has its own resolved YAML files and output root.

Existing base return forecasts, covariance, moment caches and portfolio paths were reused. This organisation refreshes saved-output tables and charts without fitting missing experiments. Retained input SHA256 hashes and the six base performance paths are independently checked against their pre-cleanup values. The preceding cleanup had regenerated portfolios; this grouping preserves those same model definitions and returns.

The user-selected zero-excess-return assumption remains for 15 US-large, 21 US-mid and 20 European saved held-stock observations over full history. They are flagged assumptions, not observed zeros. No original 80-stock membership is removed by the separate unavailable-lag-history rule. Missing characteristics retain the original neutral rank 0.5 convention. Existing daily-data currency, source-vintage and terminal-return limitations remain; the cleanup does not establish point-in-time or survivorship-free data.

All headline methods use the same 72 return months, January 2020 through December 2025, separately within each market. No returns or stocks are pooled across regions. OOS overlaps test; no additional unseen holdout is claimed. Static ML continues its inherited 2015 portfolio state into test; Market and Portfolio ML retain their original test-start initialization.

Paper-convention annual mean return, cost and utility multiply monthly figures by 12. Volatility multiplies gross-return SD by sqrt(12); net Sharpe uses gross-return SD. Utility uses gamma=10 and sample gross variance. Monthly turnover is two-sided with no half factor. Other geometric path measures are undefined if a monthly net excess return is at or below -100%; no return is silently removed.

HAC tests use Bartlett/Newey-West covariances with normal-reference inference. Holm correction is within each market's pair family. Paired circular block bootstrap uses 2,000 draws, seed 20261101 and 3/6/12-month blocks. These intervals do not correct all prior research/model selection. The paper-style utility probability is an approximation, not a HAC-robust posterior or frequentist p-value.

Cost sensitivities scale modelled costs at fixed weights and do not re-optimise. Unconstrained weights and quadratic impact can imply infeasible turnover or losses; arithmetic results in those cases are diagnostics, not executable investor wealth or proven profits. Net performance and economic utility must be read together with cost, leverage and feasibility flags. Regimes remain descriptive and use cached current-vintage US/global context even for Europe.
