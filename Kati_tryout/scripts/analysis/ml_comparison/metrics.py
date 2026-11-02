"""Paper-convention metrics plus explicitly labelled evaluation extensions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def additive_path(net):
    """Paper cumulative return and signed drop from its peak, including initial 0.

    Drawdown extends the paper's additive chart convention; multiply by 100 for
    percentage points. It is not a percentage loss of investor wealth.
    """
    values = np.asarray(net, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Additive paths require a nonempty finite return vector.')
    cumulative = np.cumsum(values)
    peak = np.maximum.accumulate(np.r_[0., cumulative])[1:]
    return cumulative, cumulative - peak


def statistics(group: pd.DataFrame, gamma=10.0) -> dict:
    """Use project Table 2 formulas; independently checked in the validation suite."""
    from jkmp.evaluation.metrics import summary_stats
    g = group.sort_values('eom_ret')
    if len(g) < 2:
        raise ValueError('Performance statistics require at least two months.')
    if g.eom_ret.duplicated().any():
        raise ValueError('Duplicate return months in performance statistics.')
    if not np.isfinite(g[['r','tc','turnover','inv','shorting']].to_numpy(float)).all():
        raise ValueError('Nonfinite saved performance; no missing returns are filled.')
    out = summary_stats(g, gamma).to_dict()
    net = (g.r - g.tc).to_numpy(float)
    cumulative, dd = additive_path(net)
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2))
    sd_net = np.std(net, ddof=1)
    out.update(sample_start=str(g.eom_ret.min().date()), sample_end=str(g.eom_ret.max().date()),
        evaluation_gamma=gamma, max_drawdown_excess=float(dd.min()),
        path_convention='additive', drawdown_unit='decimal return difference (x100 = percentage points)',
        sortino_zero_excess=float(np.mean(net) / downside * np.sqrt(12)) if downside > 0 else np.nan,
        cumulative_net_excess=float(cumulative[-1]),
        cumulative_net_excess_arithmetic=float(net.sum()),
        net_return_volatility=sd_net * np.sqrt(12),
        net_sharpe_net_sd=float(np.mean(net) / sd_net * np.sqrt(12)) if sd_net > 0 else np.nan)
    return out


def table(monthly, gamma=10.0):
    return pd.DataFrame([dict(dataset=dataset,strategy=strategy,**statistics(g,gamma))
        for (dataset,strategy),g in monthly.groupby(['dataset','strategy'],sort=False)])


def annual(monthly,gamma=10.0):
    rows=[]
    for (dataset,strategy,year),g in monthly.groupby(['dataset','strategy',monthly.eom_ret.dt.year]):
        rows.append(dict(dataset=dataset,strategy=strategy,year=year,**statistics(g,gamma)))
    return pd.DataFrame(rows)


def paths_and_rolling(monthly,gamma=10.0,window=24):
    rows=[]
    for (dataset,strategy),g in monthly.groupby(['dataset','strategy'],sort=False):
        g=g.sort_values('eom_ret')
        net=g.r-g.tc
        cumulative, drawdown = additive_path(net)
        mean=g.r.rolling(window,min_periods=window).mean()
        sd=g.r.rolling(window,min_periods=window).std(ddof=1)
        costs=g.tc.rolling(window,min_periods=window).mean()
        p=pd.DataFrame(dict(dataset=dataset,strategy=strategy,eom_ret=g.eom_ret,
            net_excess=net,drawdown_excess=drawdown,path_convention='additive',
            cumulative_net_excess=cumulative,cumulative_net_excess_arithmetic=cumulative,
            cumulative_utility_paper=(net-.5*gamma*(g.r-g.r.mean())**2).cumsum(),
            rolling_net_sharpe_paper=(mean-costs)/sd*np.sqrt(12),
            rolling_utility=(mean-costs-.5*gamma*sd**2)*12,
            rolling_window=window))
        rows.append(p)
    return pd.concat(rows,ignore_index=True)


def ranks(t):
    out=t.copy()
    for column,ascending in [('Utility',False),('SR_net',False),('R_TC',False),('Vol',True),('TC',True),('Turnover',True)]:
        out[f'rank_{column}']=out.groupby('dataset')[column].rank(ascending=ascending,method='min')
    return out.sort_values(['dataset','rank_Utility'])


def metric_dictionary():
    descriptions=[
        ('R','12*mean(r)','decimal/year','Annual arithmetic gross excess return.'),
        ('Vol','sqrt(12)*sd(r, ddof=1)','decimal/year^0.5','Gross-return volatility; paper risk convention.'),
        ('SR_gross','sqrt(12)*mean(r)/sd(r, ddof=1)','ratio','Gross excess Sharpe.'),
        ('TC','12*mean(tc)','fraction/year','Modelled quadratic trading costs, not observed execution costs.'),
        ('R_TC','12*mean(r-tc)','decimal/year','Annual arithmetic net excess return.'),
        ('SR_net','sqrt(12)*mean(r-tc)/sd(r, ddof=1)','ratio','Paper net Sharpe uses GROSS volatility.'),
        ('Utility','12*(mean(r-tc)-gamma*var(r, ddof=1)/2)','decimal/year','Sample mean-variance utility; common evaluation gamma=10.'),
        ('Turnover','mean(sum_i(abs(w_i-w_start_i)))','fraction/month','Two-sided absolute traded weight; NOT divided by two.'),
        ('Lev','mean(sum_i(abs(w_i)))','gross weight','Average monthly gross leverage.'),
        ('Shorting','mean(sum_i(max(-w_i,0)))','gross weight','Average short exposure.'),
        ('max_drawdown_excess','min(C_t-max(0,C_1,...,C_t)); C=cumsum(r-tc)','decimal return difference','Additive net excess-return drawdown; x100 gives percentage points, not percentage loss of wealth. Reporting extension of the paper additive chart.'),
        ('sortino_zero_excess','sqrt(12)*mean(r-tc)/sqrt(mean(min(r-tc,0)^2))','ratio','Zero excess-return hurdle, includes all months in downside denominator.'),
        ('path_convention','additive','label','Paper reporting convention; no compounded growth or CAGR in active performance reports.'),
        ('cumulative_net_excess','sum(r-tc)','decimal return sum','Additive cumulative net excess return; x100 gives percentage points, not investor wealth.'),
        ('cumulative_net_excess_arithmetic','sum(r-tc)','fraction','Paper cumulative net-excess path; arithmetic sum, not investor wealth.'),
        ('cumulative_utility_paper','cumsum(r-tc-gamma*(r-mean(r))^2/2)','fraction','Paper cumulative utility flow using full evaluated sample gross mean; descriptive ex-post path.'),
        ('net_sharpe_net_sd','sqrt(12)*mean(r-tc)/sd(r-tc, ddof=1)','ratio','Conventional net-return SD denominator; provided separately from paper metric.'),
        ('posterior_probability_a_gt_b','Phi(mean(u_A-u_B)/(sd(u_A-u_B)/sqrt(T)))','probability','Paper normal posterior-style probability; not a frequentist p-value; no HAC adjustment.'),
        ('utility_flow','r-tc-gamma*(r-mean(r))^2/2','fraction/month','Exact R flow; its mean uses population variance and differs from Table 2 finite-sample utility.'),
        ('utility_hac_p','2*Phi(-abs(mean(d)/SE_NW(d)))','p-value','Paired sample-variance-adjusted utility-flow mean difference; Bartlett Newey-West.'),
        ('sharpe_hac_p','Normal test of paired paper-SR difference using delta-method influences','p-value','Asymptotic HAC inference; bootstrap supplies non-normal uncertainty.'),
        ('bootstrap_CI','Paired circular moving-block resamples, re-estimate each statistic','same as statistic','Percentile 95% confidence intervals; serial dependence preserved within blocks.'),
        ('Holm_p','Step-down Holm on all unordered pairs per dataset and test family','adjusted p-value','Controls family-wise error within predeclared family; does not remove prior research selection.'),
        ('weight_cosine','dot(wA,wB)/(norm(wA)*norm(wB))','[-1,1]','Average monthly similarity on union of saved IDs, absent weights zero.'),
        ('cost_multiplier','r - m*tc; w and gross r unchanged','scenario','Counterfactual re-costing, not re-optimised portfolio.'),
        ('fixed_weight_AUM','tc_new=tc_saved*AUM_anchor_new/AUM_anchor_saved','scenario','Quadratic impact at unchanged weights/relative wealth path; no participation constraint.'),
        ('forecast_volatility','sqrt(12*w^T*Sigma_t*w)','decimal/year^0.5','Saved contemporaneous covariance forecast; compare with realised gross risk descriptively.')]
    return pd.DataFrame(descriptions,columns=['metric','formula','unit','interpretation_and_method'])
