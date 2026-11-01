"""Paired, saved-output inference; this module never fits an investment model.

Paper formulas are from ``reference/jkmp_r/6 - Base analysis.R``, lines 25-40
and 102-115. Its Table 3 normal probability is labelled a *paper posterior
approximation*, not a frequentist p-value or an autocorrelation-robust posterior.

For Table 2 utility U=12[mean(r-tc)-gamma*var(r,ddof=1)/2], the finite-sample
HAC flow uses n/(n-1)*(r-mean(r))**2. Paper Table 3 instead uses the uncorrected
centered square. Both differences are retained so this small mismatch is visible.

HAC uses Bartlett/Newey-West covariances and a normal approximation. Net Sharpe
means sqrt(12)*mean(r-tc)/sd(r,ddof=1), using gross-return volatility as in R.
Its paired HAC test uses the delta-method influence function. Confidence
intervals use paired circular moving-block bootstrap samples and recompute
sample means, variances, and Sharpe ratios in every draw. These are percentile
intervals, not studentized intervals. They are research-sample uncertainty,
not protection against prior model selection, survivorship, or missing data.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm family-wise adjustment, preserving NaNs and input ordering."""
    pvalues = np.asarray(pvalues, dtype=float)
    result = np.full(pvalues.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if not len(valid):
        return result
    ordered = valid[np.argsort(pvalues[valid], kind="stable")]
    adjusted = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * pvalues[ordered])
    result[ordered] = np.minimum(adjusted, 1.0)
    return result


def newey_west_mean(values: np.ndarray, lags: int | None = None) -> dict[str, float]:
    """Mean and HAC SE, using centered lag covariance sums divided by n.

    Default lag is floor(4*(n/100)**(2/9)), capped at n-1. No prewhitening or
    additional degrees-of-freedom multiplier is used. Constant differences
    produce SE zero: p=1 at zero, p=0 for a deterministic nonzero difference.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("HAC requires at least two finite observations")
    n = len(x)
    lag = min(n - 1, max(0, int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else int(lags)))
    mean = float(x.mean())
    residual = x - mean
    variance = float(residual @ residual / n)
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * float(residual[k:] @ residual[:-k] / n)
    se = float(np.sqrt(max(variance, 0.0) / n))
    if se <= np.finfo(float).eps * max(1.0, abs(mean)):
        z = 0.0 if abs(mean) <= np.finfo(float).eps else np.sign(mean) * np.inf
        se = 0.0
    else:
        z = mean / se
    return {"mean": mean, "se": se, "z": float(z), "p": float(2 * norm.sf(abs(z))), "lags": lag}


def _normal_probability(mean: float, se: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(se):
        return float("nan")
    if se <= np.finfo(float).eps * max(1.0, abs(mean)):
        return 0.5 if abs(mean) <= np.finfo(float).eps else float(mean > 0)
    return float(norm.cdf(mean / se))


def utility_flows(r: np.ndarray, tc: np.ndarray, gamma: float, sample_corrected: bool = False) -> np.ndarray:
    """Monthly centered-gross-variance flow; corrected flow averages to Table 2 U/12."""
    r, tc = np.asarray(r, dtype=float), np.asarray(tc, dtype=float)
    if len(r) < 2 or r.shape != tc.shape or not np.isfinite(r).all() or not np.isfinite(tc).all():
        raise ValueError("utility requires matching finite arrays with at least two months")
    factor = len(r) / (len(r) - 1) if sample_corrected else 1.0
    return r - tc - 0.5 * gamma * factor * (r - r.mean()) ** 2


def net_sharpe(r: np.ndarray, tc: np.ndarray) -> float:
    """Paper annualized SR: mean net divided by sample SD of gross monthly return."""
    r, tc = np.asarray(r, dtype=float), np.asarray(tc, dtype=float)
    sd = float(np.std(r, ddof=1))
    threshold = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(r))))
    return float(np.sqrt(12) * np.mean(r - tc) / sd) if sd > threshold else float("nan")


def sharpe_influence(r: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """Plug-in influence values of annualized net/gross-SD Sharpe (mean zero).

    The n/(n-1) correction matches sample variance in the plug-in denominator.
    Valid for positive gross-return variance; returns NaN for constant gross r.
    """
    r, tc = np.asarray(r, dtype=float), np.asarray(tc, dtype=float)
    n = len(r)
    variance = float(r.var(ddof=1))
    threshold = 64 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(r))))
    if n < 2 or variance <= threshold**2:
        return np.full(r.shape, np.nan)
    sd, mean_net = np.sqrt(variance), (r - tc).mean()
    squares = (r - r.mean()) ** 2
    variance_if = n / (n - 1) * (squares - squares.mean())
    return np.sqrt(12) * ((r - tc - mean_net) / sd - mean_net * variance_if / (2 * sd**3))


def circular_block_indices(n: int, replicates: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Paired circular blocks with random starts, concatenated and truncated to n."""
    if n < 2 or replicates < 1 or not 1 <= block_length <= n:
        raise ValueError("require n>=2, replicates>=1 and 1<=block_length<=n")
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=(replicates, blocks))
    return ((starts[..., None] + np.arange(block_length)) % n).reshape(replicates, -1)[:, :n]


def _bootstrap_differences(ra: np.ndarray, ta: np.ndarray, rb: np.ndarray, tb: np.ndarray,
                           gamma: float, reps: int, block: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    indices = circular_block_indices(len(ra), reps, block, rng)
    a, ac, b, bc = ra[indices], ta[indices], rb[indices], tb[indices]
    utility = 12 * ((a - ac).mean(axis=1) - (b - bc).mean(axis=1)
                    - 0.5 * gamma * (a.var(axis=1, ddof=1) - b.var(axis=1, ddof=1)))
    sd_a, sd_b = a.std(axis=1, ddof=1), b.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.sqrt(12) * ((a - ac).mean(axis=1) / sd_a - (b - bc).mean(axis=1) / sd_b)
    threshold_a = 64 * np.finfo(float).eps * np.maximum(1.0, np.max(np.abs(a), axis=1))
    threshold_b = 64 * np.finfo(float).eps * np.maximum(1.0, np.max(np.abs(b), axis=1))
    sharpe[(sd_a <= threshold_a) | (sd_b <= threshold_b)] = np.nan
    return utility, sharpe


def _interval(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return float("nan"), float("nan")
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high)


def compare_pairs(pf: pd.DataFrame, gamma: float = 10.0, bootstrap_reps: int = 2000,
                  block_length: int = 6, seed: int = 20261101) -> pd.DataFrame:
    """Compare unordered strategy pairs separately within each dataset.

    Each row is A minus B, with P(A>B), paired two-sided frequentist HAC tests,
    and paired block-bootstrap percentile CIs. Finite joint monthly observations
    are used; excluded counts are explicit. Nonconsecutive overlapping months
    raise ValueError because row-adjacency blocks would misrepresent time.
    Duplicate dataset/strategy/month keys also raise. Holm adjustment families
    comprise all unordered pairs in a dataset, separately for utility and SR.
    """
    required = {"dataset", "strategy", "eom_ret", "r", "tc"}
    if not required <= set(pf):
        raise ValueError(f"missing comparison columns: {sorted(required - set(pf))}")
    if bootstrap_reps < 1 or block_length < 1 or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("require positive bootstrap settings and finite nonnegative gamma")
    data = pf.copy()
    data["eom_ret"] = pd.to_datetime(data["eom_ret"])
    if data[["dataset", "strategy", "eom_ret"]].isna().any().any():
        raise ValueError("missing dataset, strategy, or return-month identifier")
    if not data.eom_ret.dt.is_month_end.all():
        raise ValueError("return observations must be dated at month-end")
    if data.duplicated(["dataset", "strategy", "eom_ret"]).any():
        raise ValueError("duplicate dataset/strategy/return-month observations")
    rng = np.random.default_rng(seed)
    rows = []
    for dataset, sample in data.groupby("dataset", sort=True):
        strategies = sorted(sample.strategy.unique())
        for a, b in combinations(strategies, 2):
            left = sample.loc[sample.strategy.eq(a), ["eom_ret", "r", "tc"]].set_index("eom_ret")
            right = sample.loc[sample.strategy.eq(b), ["eom_ret", "r", "tc"]].set_index("eom_ret")
            joined = left.join(right, how="inner", lsuffix="_a", rsuffix="_b").sort_index()
            valid = np.isfinite(joined.to_numpy()).all(axis=1)
            paired = joined.loc[valid]
            n = len(paired)
            row = {"dataset": dataset, "strategy_a": a, "strategy_b": b, "n": n,
                   "n_a_available": len(left), "n_b_available": len(right),
                   "n_a_excluded": len(left) - n, "n_b_excluded": len(right) - n,
                   "n_joint_nonfinite_excluded": int((~valid).sum()), "gamma_evaluation": gamma,
                   "bootstrap_reps": bootstrap_reps, "bootstrap_block_months": min(block_length, n),
                   "seed": seed, "ci_level": 0.95,
                   "start": paired.index.min() if n else pd.NaT,
                   "end": paired.index.max() if n else pd.NaT,
                   "posterior_method": "paper iid normal approximation; not HAC; not a p-value",
                   "bootstrap_method": "paired circular moving blocks; percentile intervals; re-estimated means/variances",
                   "multiple_testing_family": "all unordered strategy pairs within dataset, separately utility and Sharpe"}
            if n < 3:
                row.update(status="SKIPPED", reason="fewer than three finite paired months")
                rows.append(row)
                continue
            periods = paired.index.to_period("M").asi8
            if not np.all(np.diff(periods) == 1):
                raise ValueError(f"nonconsecutive paired return months for {dataset}: {a} vs {b}")
            ra, ta, rb, tb = (paired[c].to_numpy(dtype=float) for c in ["r_a", "tc_a", "r_b", "tc_b"])
            paper_diff = utility_flows(ra, ta, gamma) - utility_flows(rb, tb, gamma)
            sample_diff = utility_flows(ra, ta, gamma, True) - utility_flows(rb, tb, gamma, True)
            hac = newey_west_mean(sample_diff)
            posterior_se = float(paper_diff.std(ddof=1) / np.sqrt(n))
            bootstrap_u, bootstrap_sr = _bootstrap_differences(ra, ta, rb, tb, gamma, bootstrap_reps,
                                                              min(block_length, n), rng)
            ulo, uhi = _interval(bootstrap_u)
            slo, shi = _interval(bootstrap_sr)
            sr_diff = net_sharpe(ra, ta) - net_sharpe(rb, tb)
            influence = sharpe_influence(ra, ta) - sharpe_influence(rb, tb)
            row.update(status="PASS", reason="", posterior_probability_a_gt_b=_normal_probability(float(paper_diff.mean()), posterior_se),
                       paper_flow_mean_diff_annual=12 * paper_diff.mean(),
                       utility_diff_annual=12 * sample_diff.mean(), utility_hac_se_annual=12 * hac["se"],
                       utility_hac_p=hac["p"], utility_hac_z=hac["z"], hac_lags=int(hac["lags"]),
                       utility_hac_ci_low=12 * (hac["mean"] - norm.ppf(.975) * hac["se"]),
                       utility_hac_ci_high=12 * (hac["mean"] + norm.ppf(.975) * hac["se"]),
                       utility_boot_ci_low=ulo, utility_boot_ci_high=uhi,
                       sharpe_diff=sr_diff, sharpe_boot_ci_low=slo, sharpe_boot_ci_high=shi,
                       utility_boot_valid_reps=int(np.isfinite(bootstrap_u).sum()),
                       sharpe_boot_valid_reps=int(np.isfinite(bootstrap_sr).sum()))
            if np.isfinite(influence).all() and np.isfinite(sr_diff):
                ihac = newey_west_mean(influence)
                se = ihac["se"]
                z = (0.0 if abs(sr_diff) <= np.finfo(float).eps else np.sign(sr_diff) * np.inf) if se == 0 else sr_diff / se
                row.update(sharpe_hac_se=se, sharpe_hac_z=z, sharpe_hac_p=float(2 * norm.sf(abs(z))),
                           sharpe_hac_ci_low=sr_diff - norm.ppf(.975) * se,
                           sharpe_hac_ci_high=sr_diff + norm.ppf(.975) * se,
                           sharpe_status="PASS")
            else:
                row.update(sharpe_hac_se=np.nan, sharpe_hac_z=np.nan, sharpe_hac_p=np.nan,
                           sharpe_hac_ci_low=np.nan, sharpe_hac_ci_high=np.nan,
                           sharpe_status="SKIPPED: zero gross volatility")
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for metric in ["utility", "sharpe"]:
        raw = f"{metric}_hac_p"
        if raw not in result:
            result[raw] = np.nan
        result[f"{metric}_hac_p_holm"] = result.groupby("dataset", sort=False)[raw].transform(
            lambda x: holm_adjust(x.to_numpy()))
    return result


def bootstrap_rank_stability(pf: pd.DataFrame, gamma: float = 10.0,
                             bootstrap_reps: int = 2000, block_length: int = 6,
                             seed: int = 20261101) -> pd.DataFrame:
    """Joint-block uncertainty of rankings conditional on saved portfolios.

    Every strategy in a dataset receives the SAME sampled months in each draw.
    Rank 1 is best. The 95% rank interval uses empirical inverse-CDF quantiles
    (integer ranks). Ties receive their minimum rank; the probability of best
    splits tied first place equally. These are bootstrap stability frequencies,
    not Bayesian posterior probabilities, corrected selection inference, or
    probabilities a strategy will win in a future sample. No model is re-fitted.
    """
    required={"dataset","strategy","eom_ret","r","tc"}
    if not required <= set(pf):
        raise ValueError(f"missing ranking columns: {sorted(required-set(pf))}")
    if bootstrap_reps<1 or block_length<1 or not np.isfinite(gamma) or gamma<0:
        raise ValueError("require positive bootstrap settings and finite nonnegative gamma")
    data=pf.copy()
    data['eom_ret']=pd.to_datetime(data.eom_ret)
    if data[['dataset','strategy','eom_ret']].isna().any().any():
        raise ValueError('missing dataset, strategy or return-month identifier')
    if data.duplicated(['dataset','strategy','eom_ret']).any():
        raise ValueError('duplicate dataset/strategy/return-month observations')
    rng=np.random.default_rng(seed)
    rows=[]
    for dataset,g in data.groupby('dataset',sort=True):
        gross=g.pivot(index='eom_ret',columns='strategy',values='r').sort_index()
        costs=g.pivot(index='eom_ret',columns='strategy',values='tc').reindex_like(gross)
        if len(gross)<3 or not np.isfinite(gross.to_numpy()).all() or not np.isfinite(costs.to_numpy()).all():
            raise ValueError(f'ranking requires at least 3 identical finite saved months for {dataset}')
        if not gross.index.is_month_end.all() or not np.all(np.diff(gross.index.to_period('M').asi8)==1):
            raise ValueError(f'ranking requires consecutive month-end observations for {dataset}')
        r,tc=gross.to_numpy(),costs.to_numpy()
        indices=circular_block_indices(len(gross),bootstrap_reps,min(block_length,len(gross)),rng)
        boot_r,boot_tc=r[indices],tc[indices]
        boot_net_mean=(boot_r-boot_tc).mean(axis=1)
        boot_variance=boot_r.var(axis=1,ddof=1)
        utility=12*(boot_net_mean-.5*gamma*boot_variance)
        with np.errstate(divide='ignore',invalid='ignore'):
            sharpe=np.sqrt(12)*boot_net_mean/np.sqrt(boot_variance)
        threshold=64*np.finfo(float).eps*np.maximum(1.,np.max(np.abs(boot_r),axis=1))
        sharpe[np.sqrt(boot_variance)<=threshold]=np.nan
        obs_net=(r-tc).mean(axis=0)
        obs_var=r.var(axis=0,ddof=1)
        obs_u=12*(obs_net-.5*gamma*obs_var)
        with np.errstate(divide='ignore',invalid='ignore'):
            obs_sr=np.sqrt(12)*obs_net/np.sqrt(obs_var)
        obs_threshold=64*np.finfo(float).eps*np.maximum(1.,np.max(np.abs(r),axis=0))
        obs_sr[np.sqrt(obs_var)<=obs_threshold]=np.nan
        for metric,scores,observed in [('utility',utility,obs_u),('paper_net_sharpe',sharpe,obs_sr)]:
            ordering=rankdata(-scores,method='min',axis=1,nan_policy='omit')
            observed_rank=rankdata(-observed,method='min',nan_policy='omit')
            tie_count=(ordering==1).sum(axis=1)
            for i,strategy in enumerate(gross.columns):
                valid=np.isfinite(ordering[:,i])
                sample_ranks=ordering[valid,i]
                sample_scores=scores[valid,i]
                if len(sample_ranks):
                    rank_ci=np.quantile(sample_ranks,[.025,.975],method='inverted_cdf')
                    score_ci=np.quantile(sample_scores,[.025,.975])
                    p_best=float(np.mean((sample_ranks==1)/tie_count[valid]))
                    p_top3=float(np.mean(sample_ranks<=min(3,len(gross.columns))))
                else:
                    rank_ci=score_ci=[np.nan,np.nan]
                    p_best=p_top3=np.nan
                rows.append(dict(dataset=dataset,strategy=strategy,ranking_metric=metric,
                    n=len(gross),sample_start=gross.index.min(),sample_end=gross.index.max(),
                    n_strategies=len(gross.columns),observed_metric=observed[i],observed_rank=observed_rank[i],
                    rank_ci_low=rank_ci[0],rank_ci_high=rank_ci[1],
                    score_boot_ci_low=score_ci[0],score_boot_ci_high=score_ci[1],
                    bootstrap_probability_best_ties_split=p_best,bootstrap_probability_top3=p_top3,
                    bootstrap_valid_reps=int(valid.sum()),bootstrap_reps=bootstrap_reps,
                    block_length=min(block_length,len(gross)),seed=seed,gamma_evaluation=gamma,
                    interpretation='Joint circular-block bootstrap rank stability, conditional on saved portfolios; not a posterior or future-win probability.'))
    return pd.DataFrame(rows)
