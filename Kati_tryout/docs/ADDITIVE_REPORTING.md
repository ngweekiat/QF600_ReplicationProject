# Additive performance reporting

Kati's active performance reports now use the additive convention in the authors'
replication script `6 - Base analysis.R`, lines 44-46:

- Gross cumulative return: `cumsum(r)`.
- Net cumulative return: `cumsum(r - tc)`.
- Cumulative utility: the sum of monthly utility flows.

Source: https://github.com/theisij/ml-and-the-implementable-efficient-frontier/blob/main/6%20-%20Base%20analysis.R

For presentation consistency, Kati defines **additive drawdown** as the cumulative
net return minus its running maximum, including the initial zero. Relative
performance is the cumulative net return minus Market's cumulative net return.
Both are in return units; multiplying by 100 gives **percentage points (pp)**.
Additive drawdown is our reporting extension of that chart convention, not a
verified drawdown definition in the research paper. It is not a percent loss of
an investor's capital. Two successive -10% returns give -20 pp additive drawdown.

The shared `max_drawdown_excess`, `drawdown_excess`, and `cumulative_net_excess`
CSV columns now use this convention. Reports carry `path_convention=additive`;
the summary CSV specifies the drawdown unit. The explicitly arithmetic cumulative
column remains as an equal-valued compatibility alias. CAGR, geometric Calmar,
compounded excess indices and their validity flags are removed from active
comparison reports, rather than mixing conventions. Additive model diagnostics
can exceed 100 pp; they must not be described as feasible wealth paths.

This is a reporting change. Saved returns, costs, weights, model coefficients,
selected characteristics, stocks and dates are unchanged. Annual arithmetic
returns, Sharpe, utility, P(better), acceptance decisions and final ancestry are
unchanged. The original wealth evolution used for trading costs and drift is
economic accounting and retains its compounding. Figure 1 uses annual mean
returns and volatility and is unaffected.

Base results in all three markets, all nine active iterations, the final report,
the overall comparisons, the compact screenshot, the iteration 2/4 presentation
panels and the embedded notebook are refreshed. Legacy iteration folders remain
historical reference material and are not regenerated as current results.

Refresh saved-output reports without fitting models:

```powershell
python -X utf8 Kati_tryout/scripts/base/compare_results.py
python -X utf8 Kati_tryout/scripts/comparison_analysis.py --only-group all
python -X utf8 Kati_tryout/scripts/organise_results.py
python -X utf8 Kati_tryout/scripts/dashboard.py
python -X utf8 Kati_tryout/scripts/build_notebook.py
```
