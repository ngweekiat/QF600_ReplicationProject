"""Report extras computed from saved pipeline outputs (no re-estimation):
Table 3 (Bayesian probability of outperformance), Table 4 (gross-return correlations), Fig. 2 (cumulative
gross / net / utility). Usage: python make_report_extras.py <QF600_Replication/outputs> <out_dir>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

out_root, dest = Path(sys.argv[1]), Path(sys.argv[2]); dest.mkdir(parents=True, exist_ok=True)
GAMMA = 10.0
pf = pd.concat([pd.read_parquet(out_root / f) for f in
    ["benchmarks/pf_test.parquet", "static_ml/pf.parquet", "portfolio_ml/rff/pf.parquet", "portfolio_ml/raw/pf.parquet"]])
pf["eom_ret"] = pd.to_datetime(pf["eom_ret"])
pf["net"] = pf["r"] - pf["tc"]
# Utility flow as in the repo's Table 2 (R pf_summary): variance of GROSS returns (deviation R4/R11)
pf["u"] = pf["net"] - GAMMA / 2 * (pf["r"] - pf.groupby("type")["r"].transform("mean")) ** 2

order = ["Portfolio-ML", "Portfolio-ML (linear)", "Static-ML*", "Static-ML", "Market", "1/N",
         "Minimum variance", "Factor-ML", "Markowitz-ML"]
U = pf.pivot(index="eom_ret", columns="type", values="u")[order]
T = len(U)
prob = pd.DataFrame(index=order, columns=order, dtype=float)
for a in order:
    for b in order:
        if a != b:
            d = U[a] - U[b]
            prob.loc[a, b] = norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(T)))
prob.to_csv(dest / "table3_prob_outperform.csv")
G = pf.pivot(index="eom_ret", columns="type", values="r")[order]
G.corr().to_csv(dest / "table4_corr_gross.csv")
print("T =", T); print((prob * 100).round(1).to_string()); print(G.corr().round(2).to_string())

plot = ["Portfolio-ML", "Portfolio-ML (linear)", "Static-ML*", "Market", "1/N"]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
for k, (col, title) in enumerate([("r", "Gross return"), ("net", "Net of trading cost"), ("u", "Realised utility (γ = 10)")]):
    W = pf.pivot(index="eom_ret", columns="type", values=col)
    for t in plot:
        ax[k].plot(W.index, W[t].cumsum(), label=t, lw=1.8 if "Portfolio" in t else 1.2)
    ax[k].axhline(0, color="grey", lw=0.6); ax[k].set_title(title)
ax[0].set_ylabel("Cumulative sum of monthly returns"); ax[0].legend(fontsize=8)
fig.suptitle("Figure 2 replication: performance over time, test window 2020–2025 ($10B, γ = 10)")
fig.tight_layout(); fig.savefig(dest / "fig2_performance_over_time.png", dpi=170)
