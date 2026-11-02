"""Paper-convention metrics plus explicitly labelled evaluation extensions."""
from __future__ import annotations

import numpy as np
import pandas as pd


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
    wealth = np.cumprod(1 + net)
    # Include initial wealth=1: a loss in the first return month is a drawdown.
    peak = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    dd = wealth / peak - 1
    cagr_valid = bool((1 + net > 0).all())
    cagr = wealth[-1] ** (12 / len(net)) - 1 if cagr_valid else np.nan
    downside = np.sqrt(np.mean(np.minimum(net, 0.0) ** 2))
    sd_net = np.std(net, ddof=1)
    out.update(sample_start=str(g.eom_ret.min().date()), sample_end=str(g.eom_ret.max().date()),
        evaluation_gamma=gamma, max_drawdown_excess=float(dd.min()) if cagr_valid else np.nan,
        sortino_zero_excess=float(np.mean(net) / downside * np.sqrt(12)) if downside > 0 else np.nan,
        cagr_excess=cagr, cumulative_net_excess=float(wealth[-1] - 1) if cagr_valid else np.nan,
        cumulative_net_excess_arithmetic=float(net.sum()),
        calmar_excess=float(cagr / abs(dd.min())) if cagr_valid and dd.min() < 0 else np.nan,
        net_return_volatility=sd_net * np.sqrt(12),
        net_sharpe_net_sd=float(np.mean(net) / sd_net * np.sqrt(12)) if sd_net > 0 else np.nan,
        geometric_excess_path_valid=cagr_valid)
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
        wealth=(1+net).cumprod()
        peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
        # A synthetic excess-index growth factor <=0 invalidates all subsequent
        # geometric wealth/drawdown observations. Arithmetic paper paths remain
        # meaningful even for unconstrained portfolios with enormous modelled TC.
        geometric_valid=(net.gt(-1)).cummin()
        drawdown=(wealth/peak-1).where(geometric_valid)
        wealth=wealth.where(geometric_valid)
        mean=g.r.rolling(window,min_periods=window).mean()
        sd=g.r.rolling(window,min_periods=window).std(ddof=1)
        costs=g.tc.rolling(window,min_periods=window).mean()
        p=pd.DataFrame(dict(dataset=dataset,strategy=strategy,eom_ret=g.eom_ret,
            net_excess=net,wealth_index_excess=wealth,drawdown_excess=drawdown,
            geometric_excess_path_valid=geometric_valid,
            cumulative_net_excess_arithmetic=net.cumsum(),
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
        ('max_drawdown_excess','min(W_t/max(1,W_1,...,W_t)-1); W=cumprod(1+r-tc)','fraction','Synthetic compounded EXCESS-return drawdown; not investor total-wealth drawdown.'),
        ('sortino_zero_excess','sqrt(12)*mean(r-tc)/sqrt(mean(min(r-tc,0)^2))','ratio','Zero excess-return hurdle, includes all months in downside denominator.'),
        ('calmar_excess','cagr_excess/abs(max_drawdown_excess)','ratio','Excess-index geometric return over absolute drawdown.'),
        ('cagr_excess','prod(1+r-tc)^(12/T)-1','decimal/year','Synthetic excess-index CAGR; not arithmetic return minus annual RF.'),
        ('cumulative_net_excess','prod(1+r-tc)-1','fraction','Synthetic cumulative excess-return index; no assumption of fully funded investor wealth.'),
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
