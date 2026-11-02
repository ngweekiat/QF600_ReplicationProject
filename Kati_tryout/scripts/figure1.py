"""Figure 1 adapted to the retained Keats models, with independently fitted grid points.

Base: risk aversion x fund size, US high-cap Static ML* and regional Portfolio ML. No retired ML
families are run. Iteration extensions use active TE settings on a separate
axis; a TE sweep is never labelled as the paper's absolute-risk frontier.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass, replace
import hashlib
import html
import json
import logging
import os
from pathlib import Path
import shutil
import time
from collections.abc import Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.linalg import cho_factor, cho_solve

from workflow import ROOT, UNIVERSES, config_path
from jkmp.config import load_config
from jkmp.covariance import load_covariance
from jkmp.data.prepare import risk_free_monthly
from jkmp.portfolio.grids import grid_inputs, scaled_wealth, point_key, _signal_kwargs
from jkmp.portfolio.portfolio_ml import (build_yearly_moments, save_yearly_moments,
    load_yearly_moments, search_yearly, run_portfolio_ml, signal_set)
from jkmp.portfolio.static_ml_star import hyper_grid, run_static_ml_star
from jkmp.evaluation.metrics import summary_stats
from comparison_analysis import BASE_OUT, NAMES, STYLE, relative_metrics, market_dir, build_market_reference

OUT = BASE_OUT / "figure1"
ENGINE_FILES = ("scripts/jkmp/portfolio/portfolio_ml.py", "scripts/jkmp/portfolio/static_ml.py",
    "scripts/jkmp/portfolio/trading_speed.py", "scripts/jkmp/portfolio/accounting.py",
    "scripts/jkmp/portfolio/recursion.py", "scripts/jkmp/models/random_features.py")
log = logging.getLogger(__name__)


def frontier_methods(universe):
    return ([("Static-ML*","pf_static.parquet")] if universe=="us_large" else [])+[("Portfolio-ML","pf_portfolio.parquet")]


def portfolio_compatible(previous,current):
    """Reuse unchanged Portfolio ML fits only after checking every model/data input."""
    def inputs(record):
        config=dict(record.get("config",{}))
        for key in ("reporting","static_ml","regimes"):config.pop(key,None)
        sources={p:h for p,h in record.get("sources",{}).items() if not p.replace(chr(92),"/").endswith(("/static_ml.py","/static_ml_star.py"))}
        return config,sources
    return bool(previous) and inputs(previous)==inputs(current)


def moments_compatible(previous,current):
    """Ridge and return forecasts do not enter Portfolio ML sufficient statistics."""
    def inputs(record):
        config=dict(record.get('config',{}))
        for key in ('reporting','regimes','static_ml','ridge_grid','return_model'):config.pop(key,None)
        sources={p:h for p,h in record.get('sources',{}).items()
                 if not p.replace(chr(92),'/').endswith(('/predictions.parquet','/static_ml.py','/static_ml_star.py'))}
        return config,sources
    return bool(previous) and inputs(previous)==inputs(current)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""): h.update(block)
    return h.hexdigest()


def primitive(value):
    if is_dataclass(value):return {f.name:primitive(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):return {str(k):primitive(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [primitive(x) for x in value]
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    return str(value)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(".writing.json")
    temp.write_text(json.dumps(value,indent=2,default=str)+"\n",encoding="utf-8")
    temp.replace(path)


def provenance(universe,variant="base"):
    cfg=load_config(config_path(universe,variant))
    sources=[cfg.paths.processed/name for name in ("panel.parquet","predictions.parquet","wealth.parquet",
        "covariance/cov_factor.parquet","covariance/cov_loadings.parquet","covariance/provenance.json")]
    engine=list(ENGINE_FILES)
    if variant=="base":engine[1]="scripts/jkmp/portfolio/static_ml_star.py"
    sources += [cfg.paths.raw/"ff_factors_monthly.parquet"]+[ROOT/p for p in engine]
    sources += [p for region in ("US","EU") for p in (ROOT/f"data/inputs/lag_history_{region}.parquet",) if p.exists()]
    if variant!="base":sources.append(ROOT/"scripts/jkmp/portfolio/active.py")
    record=dict(config=primitive(cfg),sources={str(p):digest(p) for p in sources},
        estimator=("Annually tuned US high-cap Static ML* and regional RFF Portfolio ML" if variant=="base" else "Keats fixed Static ML and expanding utility-trained RFF Portfolio ML"),format_version=2)
    record["signature"]=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return record


def load_inputs(cfg):
    p=cfg.paths.processed
    # grid_inputs uses only stocks ever held. Push that identical filter into
    # the parquet reader rather than merging a million-row regional panel.
    members=pd.read_parquet(p/'panel.parquet',columns=['id'],filters=[('in_universe','=',True)])
    ids=sorted(members.id.unique().tolist())
    columns=list(dict.fromkeys(['id','eom','me','in_universe','lambda','g','tr_ld1','ret_ld1',*cfg.features.names]))
    panel=pd.read_parquet(p/'panel.parquet',columns=columns,filters=[('id','in',ids)])
    preds=pd.read_parquet(p/'predictions.parquet',columns=['id','eom','pred_ld1'],filters=[('id','in',ids)])
    panel=panel.merge(preds,on=["id","eom"],how="left",validate="one_to_one")
    wealth=pd.read_parquet(p/"wealth.parquet")
    cov=load_covariance(p/"covariance")
    rf=risk_free_monthly(pd.read_parquet(cfg.paths.raw/"ff_factors_monthly.parquet"))
    return grid_inputs(cfg,panel,wealth,cov,rf)


def validate_pf(pf, cfg, method):
    expected=pd.date_range(cfg.dates.test_start,cfg.dates.test_end,freq="ME")
    d=pf.sort_values("eom_ret")
    if set(d.type)!={method} or not pd.DatetimeIndex(d.eom_ret).equals(expected):
        raise ValueError(f"Wrong strategy or incomplete calendar: {method}")
    if not np.isfinite(d[["r","tc","turnover","inv","shorting"]].to_numpy(float)).all():
        raise ValueError("Nonfinite frontier performance")
    return d


def valid_point(folder, signature, cfg):
    marker=folder/"completion.json"
    if not marker.exists():return False
    record=json.loads(marker.read_text())
    if record.get("signature")!=signature:return False
    for method,filename in frontier_methods(cfg.paths.outputs.name):
        path=folder/filename
        if not path.exists() or digest(path)!=record.get("hashes",{}).get(filename):return False
        validate_pf(pd.read_parquet(path),cfg,method)
    return True


def point_signature(source_signature,gamma,level):
    return hashlib.sha256(f"{source_signature}|{gamma:.17g}|{level:.17g}".encode()).hexdigest()


def fit_point(universe,gamma,level,source_signature,force=False,previous_signature=None,previous_moments_signature=None):
    """New fit at each (gamma,wealth); source headline portfolios are read-only."""
    cfg=load_config(config_path(universe))
    folder=OUT/universe/"points"/point_key(gamma,level)
    signature=point_signature(source_signature,gamma,level)
    if not force and valid_point(folder,signature,cfg):return dict(point=folder.name,status="reused")
    old=json.loads((folder/"completion.json").read_text()) if (folder/"completion.json").exists() else {}
    portfolio_reuse=(not force and previous_signature is not None and
        old.get("signature")==point_signature(previous_signature,gamma,level) and
        (folder/"pf_portfolio.parquet").exists() and
        digest(folder/"pf_portfolio.parquet")==old.get("hashes",{}).get("pf_portfolio.parquet"))
    folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    write_json(folder/"status.json",dict(status="running",gamma=gamma,wealth=level,pid=os.getpid()))
    try:
        native=gamma==cfg.investor.gamma and level==cfg.investor.wealth
        if native:
            # Exact anchor point is the already-fitted retained strategy, not a
            # rescaled approximation. It also provides a preservation check.
            copies=[
                            ("portfolio_ml/rff/pf.parquet","pf_portfolio.parquet"),
                            ("portfolio_ml/rff/chosen.csv","chosen_portfolio.csv")]
            if universe=="us_large":copies += [("static_ml_star/pf.parquet","pf_static.parquet"),("static_ml_star/weights.parquet","weights_static.parquet"),("static_ml_star/chosen.csv","chosen_static.csv")]
            for src,dst in copies:
                shutil.copy2(cfg.paths.outputs/src,folder/dst)
            origin="existing exact baseline gamma and wealth"
        else:
            if universe=="us_large" or not portfolio_reuse:
                inp=load_inputs(cfg)
                wealth=scaled_wealth(inp.wealth,level,cfg.investor.wealth)
            if universe=="us_large":
                grid=hyper_grid(cfg.static_ml.k_grid,cfg.static_ml.u_grid,cfg.static_ml.g_grid)
                static=run_static_ml_star(inp.panel,wealth,inp.cov,inp.val_dates,inp.eval_dates,gamma,grid,cfg.portfolio.charge_exit_tc)
                static["pf"].to_parquet(folder/"pf_static.parquet",index=False)
                static["weights"].to_parquet(folder/"weights_static.parquet",index=False)
                static["chosen"].to_csv(folder/"chosen_static.csv",index=False)
                static["validation"].to_csv(folder/"validation_static.csv",index=False)
            if portfolio_reuse:
                validate_pf(pd.read_parquet(folder/"pf_portfolio.parquet"),cfg,"Portfolio-ML")
            else:
                old_moments=point_signature(previous_moments_signature,gamma,level) if previous_moments_signature else None
                fit_portfolio_point(inp,cfg,wealth,gamma,signature,force,folder,old_moments)
            origin="Static ML* retuned where enabled; Portfolio ML reused only after matching source hashes and model config" if portfolio_reuse else "fresh coefficients and past-utility hyperparameter selection at this gamma and wealth"
        for method,filename in frontier_methods(universe):
            validate_pf(pd.read_parquet(folder/filename),cfg,method)
        record=dict(status="complete",signature=signature,gamma=gamma,wealth=level,origin=origin,
            seconds=round(time.monotonic()-started,2),
            hashes={n:digest(folder/n) for _,n in frontier_methods(universe)})
        write_json(folder/"completion.json",record);write_json(folder/"status.json",record)
        print(f"COMPLETE {universe} gamma={gamma:g} wealth=${level/1e9:g}B ({record['seconds']:.1f}s)",flush=True)
        return dict(point=folder.name,status="complete")
    except Exception as error:
        write_json(folder/"status.json",dict(status="failed",gamma=gamma,wealth=level,error=str(error)))
        raise


def fit_portfolio_point(inp,cfg,wealth,gamma,signature,force,folder,previous_signature=None):
    universe=cfg.paths.outputs.name
    pm=cfg.portfolio_ml;results={}
    _,tags,label=signal_set(pm.use_rff,pm.rff_variance_grid)
    # Compact root avoids Windows MAX_PATH in this deeply nested repo.
    # The full signature is retained in each cache and checked below.
    cache=ROOT/"data/figure1"/universe/signature[:16]
    def read_cache(path):
        import zipfile
        if force or not path.exists():return None
        try:return load_yearly_moments(path)
        except (zipfile.BadZipFile,EOFError,ValueError,OSError) as error:
            log.warning('Incomplete/corrupt derived cache %s; rebuilding from observed inputs (%s)',path,error)
            return None
    for tag,g in zip(tags,pm.rff_variance_grid):
        path=cache/f"{tag}.npz"
        ym=read_cache(path)
        if ym is None and not force and previous_signature:
            old=ROOT/'data/figure1'/universe/previous_signature[:16]/f'{tag}.npz'
            if old.exists():
                cached=read_cache(old)
                if cached is not None and cached.meta.get('figure1_signature')==previous_signature:
                    ym=cached
                    # In-memory provenance only; keep the original cache intact.
                    ym.meta['figure1_signature']=signature
        if ym is None or ym.meta.get("figure1_signature")!=signature:
            ym=build_yearly_moments(inp.panel,wealth,inp.cov,inp.rf,inp.pfml_dates,cfg.features.names,
                signal_dates=inp.eval_dates,gamma=gamma,mu=cfg.investor.mu,lb=cfg.universe.lookback_months,
                iterations=cfg.investor.m_iterations,scale_by_vol=pm.scale_by_vol,
                add_constant=pm.add_constant,**_signal_kwargs(cfg,g))
            ym.meta["figure1_signature"]=signature
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_name(path.stem+'.writing.npz')
            save_yearly_moments(ym,temporary);temporary.replace(path)
        results[tag]=search_yearly(ym,tag,pm.p_grid,cfg.ridge_grid.values,inp.hp_years,pm.half_life_years)
    result=run_portfolio_ml(results,inp.panel,wealth,inp.cov,inp.rf,inp.eval_dates,inp.eval_dates,
        gamma=gamma,mu=cfg.investor.mu,iterations=cfg.investor.m_iterations,
        charge_exit_tc=cfg.portfolio.charge_exit_tc,label=label)
    for name in ("pf","weights","aim_coefs"):
        result[name].to_parquet(folder/f"{name}_portfolio.parquet",index=False)
    result["chosen"].to_csv(folder/"chosen_portfolio.csv",index=False)
    result["validation"].to_csv(folder/"validation_portfolio.csv",index=False)


def frictionless_reference(inp,cfg):
    """Paper Eq 42, evaluated on forward returns; a diagnostic, not another ML family."""
    targets=np.r_[np.arange(-.5,.50001,.05),.6,.75,1.,2.]/12
    per_target={float(k):[] for k in targets};constraints=[]
    for date in inp.eval_dates:
        d=inp.panel.loc[inp.panel.eom.eq(date)&inp.panel.in_universe].sort_values("id")
        ids=d.id.to_numpy();mu=d.pred_ld1.to_numpy();one=np.ones(len(ids))
        sigma=inp.cov[date].full(ids)
        # Center before solving: the textbook a*c-b*b form suffers severe
        # cancellation when stock forecasts are almost identical.
        if np.ptp(mu)<=1e-10*max(1.,np.max(np.abs(mu))):
            raise ValueError(f"Degenerate frictionless reference: nearly constant forecasts at {date.date()}")
        z=mu-mu.mean()
        solved=cho_solve(cho_factor(sigma),np.column_stack([z,one]))
        x,y=solved[:,0],solved[:,1];c=float(one@y)
        w0=y/c;v=x-y*float(one@x)/c
        mu0=float(mu@w0);delta=float(z@v)
        if delta<=0:raise ValueError(f"Degenerate frictionless reference at {date.date()}")
        for k in targets:
            w=w0+(k-mu0)*v/delta
            if abs(w.sum()-1)>1e-8 or abs(w@mu-k)>1e-8:
                raise ValueError(f"Degenerate frictionless reference: unstable constraints at {date.date()}")
            per_target[float(k)].append(dict(eom=date,eom_ret=date+pd.offsets.MonthEnd(1),r=float(w@d.ret_ld1),
                tc=0.,turnover=np.nan,inv=float(abs(w).sum()),shorting=float(-w.clip(max=0).sum())))
            constraints.append(dict(eom=date,target_monthly=k,weight_sum=float(w.sum()),predicted_return=float(w@mu)))
    monthly=pd.concat([pd.DataFrame(v).assign(target_monthly=k) for k,v in per_target.items()],ignore_index=True)
    summary=pd.DataFrame([dict(target_monthly=k,**summary_stats(pd.DataFrame(v),cfg.investor.gamma).to_dict()) for k,v in per_target.items()])
    return monthly,summary,pd.DataFrame(constraints)


def summarize_market(universe,gammas,levels,signature):
    cfg=load_config(config_path(universe));folder=OUT/universe
    rows=[];monthly=[];checks=[]
    for level in levels:
        for gamma in gammas:
            point=folder/"points"/point_key(gamma,level)
            if not valid_point(point,point_signature(signature,gamma,level),cfg):raise ValueError(f"Incomplete frontier point: {point}")
            for method,name in frontier_methods(universe):
                d=validate_pf(pd.read_parquet(point/name),cfg,method)
                row=dict(universe=universe,type=method,gamma=gamma,wealth=level,**summary_stats(d,gamma).to_dict())
                rows.append(row);monthly.append(d.assign(gamma=gamma,wealth=level))
                # Independent scalar arithmetic check against the metric library.
                r=d.r.to_numpy();tc=d.tc.to_numpy();mean=sum(r)/len(r)
                variance=sum((x-mean)**2 for x in r)/(len(r)-1)
                passed=np.isclose(row["R_TC"],12*(sum(r-tc)/len(r)),rtol=1e-11,atol=1e-12) and np.isclose(row["Vol"],np.sqrt(12*variance))
                if level==0:passed=passed and bool(d.tc.eq(0).all())
                checks.append(dict(type=method,gamma=gamma,wealth=level,status="PASS" if passed else "FAIL"))
    summary=pd.DataFrame(rows);summary.to_csv(folder/"frontier_summary.csv",index=False)
    pd.concat(monthly,ignore_index=True).to_csv(folder/"monthly_returns.csv",index=False)
    pd.DataFrame(checks).to_csv(folder/"independent_verification.csv",index=False)
    if any(c["status"]!="PASS" for c in checks):raise AssertionError("Frontier arithmetic verification failed")
    anchor=summary.loc[summary.gamma.eq(cfg.investor.gamma)&summary.wealth.eq(cfg.investor.wealth)]
    for row in anchor.itertuples():
        rel="static_ml_star/pf.parquet" if row.type=="Static-ML*" else "portfolio_ml/rff/pf.parquet"
        native=pd.read_parquet(cfg.paths.outputs/rel)
        d=pd.concat(monthly).loc[lambda x:x.type.eq(row.type)&x.gamma.eq(row.gamma)&x.wealth.eq(row.wealth)]
        np.testing.assert_allclose(d[["r","tc"]],native.sort_values("eom_ret")[["r","tc"]],rtol=0,atol=0)
    ref_path=folder/"frictionless_reference.csv";ref_status=folder/"frictionless_reference_status.json"
    cached=json.loads(ref_status.read_text()) if ref_status.exists() else {}
    if cached.get('source_signature')!=signature or cached.get('method_version')!=3:
        inp=load_inputs(cfg)
        try:
            m,s,c=frictionless_reference(inp,cfg)
            status=dict(status='available',method_version=3,source_signature=signature,excluded_months=0,
                turnover='not calculated for the diagnostic reference; reported as missing')
        except ValueError as error:
            if not str(error).startswith('Degenerate frictionless reference'):raise
            # Do not draw a shortened-sample or numerically infeasible curve.
            m=pd.DataFrame(columns=['eom','eom_ret','target_monthly','r','tc'])
            s=pd.DataFrame(columns=['target_monthly','n','R','Vol','R_TC'])
            c=pd.DataFrame(columns=['eom','target_monthly','weight_sum','predicted_return'])
            status=dict(status='unavailable',reason=str(error),method_version=3,source_signature=signature,
                excluded_months=0,entire_reference_omitted=True,retained_model_frontiers_affected=False)
        m.to_csv(folder/"frictionless_monthly.csv",index=False);s.to_csv(ref_path,index=False)
        c.to_csv(folder/"frictionless_constraint_checks.csv",index=False);write_json(ref_status,status)
    constraints=pd.read_csv(folder/"frictionless_constraint_checks.csv")
    if not constraints.empty:
        np.testing.assert_allclose(constraints.weight_sum,1,atol=1e-8)
        np.testing.assert_allclose(constraints.predicted_return,constraints.target_monthly,atol=1e-8)
    return summary,pd.read_csv(ref_path)


def verify_grid_accounting(universe,summary):
    """Independently reconcile every fitted point to its saved stock weights."""
    cfg=load_config(config_path(universe));folder=OUT/universe
    panel=pd.read_parquet(cfg.paths.processed/'panel.parquet',columns=['id','eom','ret_ld1','lambda','in_universe'])
    ids=panel.loc[panel.in_universe,'id'].unique()
    dates=pd.date_range(cfg.dates.test_start,cfg.dates.test_end,freq='ME')-pd.offsets.MonthEnd(1)
    panel=panel.loc[panel.id.isin(ids)&panel.eom.isin(dates),['id','eom','ret_ld1','lambda']]
    wealth=pd.read_parquet(cfg.paths.processed/'wealth.parquet').set_index('eom').wealth
    rows=[];names=['r','tc','turnover','inv','shorting']
    for row in summary.itertuples():
        point=folder/'points'/point_key(row.gamma,row.wealth)
        method='static' if row.type=='Static-ML*' else 'portfolio'
        weights_path=point/f'weights_{method}.parquet'
        if not weights_path.exists():
            rel='static_ml_star/weights.parquet' if method=='static' else 'portfolio_ml/rff/weights.parquet'
            weights_path=cfg.paths.outputs/rel
        weights=pd.read_parquet(weights_path)
        d=weights.merge(panel,on=['id','eom'],how='left',validate='one_to_one')
        held=~d['exit']
        if d['lambda'].isna().any() or d.loc[held,'ret_ld1'].isna().any():raise ValueError('Incomplete frontier weight accounting sources')
        d['r_i']=d.w*d.ret_ld1.where(held,0.)
        d['tc_i']=.5*d.eom.map(wealth)*(row.wealth/cfg.investor.wealth)*d['lambda']*(d.w-d.w_start)**2
        d['turnover_i']=(d.w-d.w_start).abs();d['inv_i']=d.w.abs();d['short_i']=-d.w.clip(upper=0)
        calc=d.groupby('eom').agg(r=('r_i','sum'),tc=('tc_i','sum'),turnover=('turnover_i','sum'),inv=('inv_i','sum'),shorting=('short_i','sum'))
        pf=pd.read_parquet(point/f'pf_{method}.parquet').sort_values('eom')
        observed=calc.reindex(pf.eom)[names].to_numpy()
        expected=pf[names].to_numpy()
        passed=np.allclose(observed,expected,rtol=1e-10,atol=1e-12)
        rows.append(dict(type=row.type,gamma=row.gamma,wealth=row.wealth,months=len(pf),
            status='PASS' if passed else 'FAIL',max_return_error=float(abs(observed[:,0]-expected[:,0]).max()),
            max_cost_error=float(abs(observed[:,1]-expected[:,1]).max()),weights_path=str(weights_path.relative_to(ROOT))))
    result=pd.DataFrame(rows);result.to_csv(folder/'accounting_verification.csv',index=False)
    if result.status.ne('PASS').any():raise AssertionError('Frontier weight accounting failed')
    return result


def save_figure(fig,path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf"):
        target=path.with_suffix('.'+suffix)
        try:fig.savefig(target,dpi=180,bbox_inches='tight')
        except PermissionError:
            target=path.with_name(path.name+'_current').with_suffix('.'+suffix)
            fig.savefig(target,dpi=180,bbox_inches='tight')
            log.warning('Existing chart is open/locked; current export saved to %s',target)
    plt.close(fig)


def plot_market(universe,summary,reference):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg=load_config(config_path(universe));folder=OUT/universe
    fig,axes=plt.subplots(1,2,figsize=(14,5.3))
    markers={1:"o",5:"^",10:"s",20:"+",100:"x"}
    s=summary.loc[summary.wealth.eq(cfg.investor.wealth)]
    for method,color in (("Portfolio-ML","#126c9a"),("Static-ML*","#d67b29")):
        d=s.loc[s.type.eq(method)].sort_values("Vol")
        if d.empty:continue
        axes[0].plot(100*d.Vol,100*d.R_TC,"-",color=color,label=method)
        for r in d.itertuples():
            axes[0].scatter([100*r.Vol],[100*r.R_TC],marker=markers.get(r.gamma,"o"),s=35,color=color)
        base=d.loc[d.gamma.eq(cfg.investor.gamma)]
        if not base.empty:
            r=base.iloc[0];x=np.linspace(0,max(s.Vol)*1.1,150)
            axes[0].plot(100*x,100*(r.Utility+.5*cfg.investor.gamma*x*x),ls="--",color=color,alpha=.5,lw=.8)
    if not (market_dir(universe)/"pf.parquet").exists():build_market_reference(universe)
    market=summary_stats(pd.read_parquet(market_dir(universe)/"pf.parquet"),cfg.investor.gamma)
    axes[0].scatter([100*market.Vol],[100*market.R_TC],marker="D",s=35,color="#222",label="Market reference",zorder=5)
    levels=sorted(summary.wealth.unique());cmap=plt.get_cmap("viridis")
    for i,level in enumerate(levels):
        d=summary.loc[summary.type.eq("Portfolio-ML")&summary.wealth.eq(level)].sort_values("Vol")
        color=cmap(i/max(len(levels)-1,1))
        axes[1].plot(100*d.Vol,100*d.R_TC,"-",color=color,label="Zero-cost limit" if level==0 else f"${level/1e9:g}B")
        for r in d.itertuples():axes[1].scatter([100*r.Vol],[100*r.R_TC],marker=markers.get(r.gamma,"o"),s=25,color=color)
    for ax in axes:
        if not reference.empty:ax.plot(100*reference.Vol,100*reference.R_TC,ls=":",lw=1,color="#777",label="Frictionless risky-assets reference")
        ax.axhline(0,color="#777",lw=.6);ax.grid(alpha=.2)
        ax.set(xlabel="Realised gross volatility (% per year)",ylabel="Net excess return (% per year)")
        ax.legend(fontsize=7)
        # Reference targets include very high-risk portfolios; keep the view on
        # actual retained-model points, while exporting every reference point.
    for ax,points,extra_r,extra_v in ((axes[0],s,[market.R_TC],[market.Vol]),
                                    (axes[1],summary.loc[summary.type.eq("Portfolio-ML")],[],[])):
        y=np.r_[points.R_TC,extra_r];x=np.r_[points.Vol,extra_v]
        pad=max(.01,float(np.ptp(y))*.15)
        ax.set(xlim=(0,100*max(x)*1.15),ylim=(100*(min(0,y.min())-pad),100*(max(0,y.max())+pad)))
        if not reference.empty and reference.Vol.min()>max(x)*1.15:
            for line in ax.lines:
                if line.get_label()=="Frictionless risky-assets reference":line.set_label("Frictionless ref. (outside displayed risk range)")
        ax.legend(fontsize=7)
    axes[1].legend(fontsize=7,loc="upper left",bbox_to_anchor=(1.02,1))
    axes[0].set_title(f"A  Retained models at ${cfg.investor.wealth/1e9:g}B\nSymbols identify risk aversion")
    axes[1].set_title("B  Portfolio ML by investor size\nEach point is independently fitted")
    fig.suptitle(f"{NAMES[universe]}   Estimated implementable frontier   2020–2025")
    fig.text(.5,.015,"Risk aversion gamma: 1 circle; 5 triangle; 10 square; 20 plus; 100 cross",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));save_figure(fig,folder/"charts/figure1_implementable_frontier")


def write_market_page(universe,summary):
    cfg=load_config(config_path(universe));folder=OUT/universe
    text=("Figure 1 includes US high-cap Static ML* and regional Portfolio ML. Every risk-aversion/fund-size point refits the portfolio rule; "
        "Portfolio ML reselects its RFF scale, model size and ridge using cumulative past utility. US high-cap Static ML* reselects "
        "k/u/g each December using cumulative preceding utility and existing annual return forecasts. Existing baseline model outputs are reused only at their exact original gamma and wealth. "
        "Zero wealth denotes the exact zero-impact mathematical limit, not a real fund. All wealth figures are anchored at end-2019, "
        "unlike the paper's end-2020 anchor. Realised 2020–2025 gross risk and net excess returns are evaluated on identical 72 months. "
        "Lines join fitted grid points and need not be efficient or monotone out of sample. Dashed curves show gamma-10 indifference curves. "
        "Crossing fund-size curves are possible because these are finite-sample fitted strategies, not the known true optimum. "
        "The dotted risky-assets reference applies the paper's constrained mean-variance formula to saved forecasts and contemporaneous covariance; "
        "it is a zero-cost diagnostic, not an additional headline ML method. Fixed Static ML, Markowitz ML and Factor ML are omitted. US mid-cap and EU high-cap have no Static ML strategy. "
        "No separate unseen holdout is claimed. Original missing-return and initialization conventions remain in force. "
        "Returns and costs are arithmetic model diagnostics, not compounded investor wealth. Unconstrained allocations can imply extreme costs and infeasible losses, in unconstrained rules. "
        "Where the frictionless reference is outside the displayed risk range, the legend says so and its full values remain in the CSV. "
        "No optimality claim follows from choosing the best-looking point on this test plot.")
    reference_status=json.loads((folder/'frictionless_reference_status.json').read_text())
    if reference_status['status']=='unavailable':
        text+=' The optional frictionless reference is unavailable for this market: '+reference_status['reason']+'. No months were dropped or returns fabricated; the retained model frontiers remain complete.'
    (folder/"METHODS_AND_EXCLUSIONS.md").write_text("# Figure 1 methods\n\n"+text+"\n",encoding="utf-8")
    shown=summary[["type","gamma","wealth","R_TC","Vol","SR_net","Utility","TC"]].copy()
    shown["wealth"]/=1e9
    for col in ("R_TC","Vol","Utility","TC"):shown[col]*=100
    shown=shown.rename(columns={"wealth":"Wealth $B","R_TC":"Net return %","Vol":"Risk %","Utility":"Utility %","TC":"Cost %"})
    links=''.join(f'<li><a href="{p.name}">{p.stem.replace("_"," ")}</a></li>' for p in sorted(folder.glob("*.csv")))
    page=f'<h1>{NAMES[universe]} Figure 1</h1><p>{html.escape(text)}</p><img src="charts/figure1_implementable_frontier.png" alt="Two-panel implementable frontier"><p><a href="charts/figure1_implementable_frontier.pdf">Download PDF chart</a></p><div class="scroll">'+shown.to_html(index=False,float_format=lambda x:f"{x:,.4g}")+f'</div><ul>{links}</ul><p><a href="../index.html">All Figure 1 results</a></p>'
    (folder/"index.html").write_text('<!doctype html><meta charset="utf-8">'+STYLE+page,encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-universe",choices=("all",*UNIVERSES),default="all")
    parser.add_argument("--jobs",type=int,default=3)
    parser.add_argument("--gammas",type=float,nargs="+")
    parser.add_argument("--wealth-billions",type=float,nargs="+")
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--plots-only",action="store_true")
    parser.add_argument("--include-iterations",action="store_true")
    args=parser.parse_args(argv)
    if args.jobs<1:parser.error("jobs must be positive")
    universes=UNIVERSES if args.selected_universe=="all" else (args.selected_universe,)
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s")
    for universe in universes:
        cfg=load_config(config_path(universe));gammas=tuple(args.gammas or cfg.investor.gamma_grid)
        levels=tuple(w*1e9 for w in args.wealth_billions) if args.wealth_billions is not None else (0.,*cfg.investor.wealth_grid)
        if any(not np.isfinite(g) or g<=0 for g in gammas) or any(not np.isfinite(w) or w<0 for w in levels):parser.error("Require gamma>0 and wealth>=0")
        if len(set(gammas))!=len(gammas) or len(set(levels))!=len(levels):parser.error("Duplicate grid points")
        if cfg.investor.gamma not in gammas or cfg.investor.wealth not in levels:
            parser.error("Figure 1 must include the configured baseline gamma and wealth anchor")
        record=provenance(universe);folder=OUT/universe;folder.mkdir(parents=True,exist_ok=True)
        old_path=folder/"source_provenance.json"
        previous=json.loads(old_path.read_text()) if old_path.exists() else {}
        moment_signature=previous.get('signature') if moments_compatible(previous,record) else None
        if moment_signature==record['signature']:moment_signature=None
        moment_record=folder/'moment_reuse_provenance.json'
        if moment_record.exists() and moment_signature is None:
            saved=json.loads(moment_record.read_text())
            if moments_compatible(saved,record):moment_signature=saved['signature']
        elif moment_signature and moment_signature!=record['signature']:
            write_json(moment_record,previous)
        reuse_signature=previous.get("signature") if portfolio_compatible(previous,record) else None
        # Keep the previous manifest until point migration finishes so interrupted runs resume.
        migration=folder/"portfolio_reuse_provenance.json"
        if migration.exists() and reuse_signature is None:
            previous=json.loads(migration.read_text())
            reuse_signature=previous.get("signature") if portfolio_compatible(previous,record) else None
        if reuse_signature and reuse_signature!=record["signature"]:write_json(migration,previous)
        write_json(folder/"source_provenance.json",record)
        points=[(g,w) for w in levels for g in gammas]
        print(f"FIGURE 1 {universe}: {len(points)} points, {args.jobs} workers",flush=True)
        if not args.plots_only:
            with parallel_config(backend="loky",inner_max_num_threads=1):
                Parallel(n_jobs=args.jobs,verbose=10)(delayed(fit_point)(universe,g,w,record["signature"],args.force,reuse_signature,moment_signature) for g,w in points)
        summary,reference=summarize_market(universe,gammas,levels,record["signature"])
        verify_grid_accounting(universe,summary)
        plot_market(universe,summary,reference);write_market_page(universe,summary)
        # Data/model sources must be unchanged after fitting new output-only grids.
        after=provenance(universe)
        if after["signature"]!=record["signature"]:raise AssertionError("Source inputs changed during frontier run")
        write_json(folder/"completion.json",dict(status="complete",grid_points=len(points),methods=[m for m,_ in frontier_methods(universe)],
            monthly_rows=len(pd.date_range(cfg.dates.test_start,cfg.dates.test_end,freq="ME"))*len(points)*len(frontier_methods(universe)),signature=record["signature"],source_inputs_unchanged=True))
    if args.include_iterations and "us_large" in universes:
        from iteration_sequence import report_saved
        report_saved()
    plot_combined()
    index_page()
    return 0


def index_page():
    rows=[]
    for universe in UNIVERSES:
        p=OUT/universe/"completion.json"
        if p.exists():rows.append(f'<li><a href="{universe}/index.html">{NAMES[universe]} Figure 1</a></li>')
    text='<h1>Figure 1 Implementable frontiers</h1><p>Independently fitted retained models across risk aversion and investor size.</p><ul>'+''.join(rows)+'</ul>'
    if (OUT/"charts/figure1_combined.png").exists():
        pdf='figure1_combined_current.pdf' if (OUT/'charts/figure1_combined_current.pdf').exists() and (OUT/'charts/figure1_combined_current.pdf').stat().st_mtime_ns>(OUT/'charts/figure1_combined.pdf').stat().st_mtime_ns else 'figure1_combined.pdf'
        text+=f'<h2>Combined base Figure 1: all retained frontiers</h2><img src="charts/figure1_combined.png" alt="Combined method and investor-size frontiers for all three markets"><p><a href="charts/{pdf}">Combined PDF</a> | <a href="combined_frontier_summary.csv">All underlying points</a></p>'
    recommendations=("# Recommended next figures for your iterations\n\n"
        "| Priority | Paper figure | Why it helps | Required work |\n|---|---|---|---|\n"
        "| 1 | Positions by liquidity | Diagnose the exit-120/costs-x2 experiment against its previous accepted setup | Aggregate saved long/short weights into contemporaneous liquidity deciles; no model refit |\n"
        "| 2 | Stock-weight paths | Show iteration 4's benchmark-relative tilts and iteration 5's weighting changes | Plot actual and active weights for consistently available stocks; no model refit |\n"
        "| 3 | Complete Figure 2 gross return net return and utility panels | Makes each iteration's gain or loss relative to its predecessor visible after costs and risk | Saved monthly results suffice; clarify arithmetic accumulation and utility convention |\n"
        "| Later | Figure 5 economic feature importance | Explains which economic themes matter under the fitted strategy, rather than ranking raw correlations | Permute themes, keep fitted coefficients fixed, rebuild portfolio decisions and measure utility loss; repeated counterfactual runs |\n\n"
        "The current five-step sequence has no 250-stock breadth experiment. Borrowing/lending-fee figures require data not currently present. The experiment decision table, returns, drawdowns, relative index and costs/turnover plots are available in outputs/iteration_summary and each iteration folder.\n")
    (OUT/"RECOMMENDED_NEXT_FIGURES.md").write_text(recommendations,encoding="utf-8")
    text+='<p><a href="RECOMMENDED_NEXT_FIGURES.md">Recommended next figures for the iterations</a></p>'
    text+='<p><a href="../../../iteration_summary/index.html">Five US high-cap experiments and final-strategy charts</a></p>'
    (OUT/"index.html").write_text('<!doctype html><meta charset="utf-8">'+STYLE+text,encoding="utf-8")
    from comparison_analysis import add_navigation
    add_navigation(include_iterations=False)


def plot_combined():
    """Combined paper-style A/B figure using every verified retained base curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    frames=[]
    for universe in UNIVERSES:
        folder=OUT/universe;path=folder/"frontier_summary.csv";marker=folder/"completion.json"
        if not path.exists() or not marker.exists():return False
        complete=json.loads(marker.read_text())
        if complete.get("signature")!=provenance(universe)["signature"]:return False
        d=pd.read_csv(path)
        if set(d.type)!={m for m,_ in frontier_methods(universe)}:
            raise ValueError(f"Unexpected methods in combined frontier: {universe}")
        if d.duplicated(["type","gamma","wealth"]).any():raise ValueError("Duplicate combined frontier points")
        frames.append(d.assign(universe=universe,market=NAMES[universe]))
    all_points=pd.concat(frames,ignore_index=True)
    if not np.isfinite(all_points[["Vol","R_TC"]]).all().all():raise ValueError("Nonfinite combined frontier")
    all_points.to_csv(OUT/"combined_frontier_summary.csv",index=False)
    fig,axes=plt.subplots(2,1,figsize=(12,12))
    colors={"us_large":"#146c9a","us_mid":"#ae4362","eu_large":"#159472"}
    anchors=[]
    for universe in UNIVERSES:
        cfg=load_config(config_path(universe));d=all_points.loc[all_points.universe.eq(universe)]
        for method,_ in frontier_methods(universe):
            color="#d87920" if method=="Static-ML*" else colors[universe]
            label=f"{NAMES[universe]} {method}"
            for level,style in ((cfg.investor.wealth,"-"),(0.,":")):
                curve=d.loc[d.type.eq(method)&d.wealth.eq(level)].sort_values("Vol")
                if curve.empty:continue
                axes[0].plot(100*curve.Vol,100*curve.R_TC,style,color=color,lw=1.8 if level else 1.1,
                    marker="o" if level else None,ms=3,label=label if level else "_nolegend_")
                if level:
                    anchor=curve.loc[curve.gamma.eq(cfg.investor.gamma)]
                    axes[0].scatter(100*anchor.Vol,100*anchor.R_TC,color=color,marker="s",s=58,zorder=5)
                    anchors.append(curve)
        pf=pd.read_parquet(market_dir(universe)/"pf.parquet")
        st=summary_stats(pf,cfg.investor.gamma)
        axes[0].scatter(100*st.Vol,100*st.R_TC,marker="D",color=colors[universe],s=40,label=f"{NAMES[universe]} Market")
    axes[0].plot([],[],":",color="#666",label="Dotted: zero-impact refit")
    levels=sorted(all_points.wealth.unique());cmap=plt.get_cmap("viridis")
    styles={"us_large":"-","us_mid":"--","eu_large":":"}
    for universe in UNIVERSES:
        for method,_ in frontier_methods(universe):
            for i,level in enumerate(levels):
                d=all_points.loc[all_points.universe.eq(universe)&all_points.type.eq(method)&all_points.wealth.eq(level)].sort_values("Vol")
                if d.empty:continue
                axes[1].plot(100*d.Vol,100*d.R_TC,ls="-." if method=="Static-ML*" else styles[universe],
                    color=cmap(i/max(1,len(levels)-1)),lw=1.5,marker="o",ms=2.5,label="_nolegend_")
    handles=[Line2D([0],[0],color=cmap(i/max(1,len(levels)-1)),label="Zero-impact limit" if level==0 else f"${level/1e9:g}B") for i,level in enumerate(levels)]
    handles += [Line2D([0],[0],color="#333",ls=styles[u],label=NAMES[u]+" Portfolio ML") for u in UNIVERSES]
    handles += [Line2D([0],[0],color="#333",ls="-.",label="US high-cap Static ML*")]
    axes[1].legend(handles=handles,fontsize=8,ncol=3,loc="best")
    axes[0].legend(fontsize=8,ncol=2,loc="best")
    axes[0].set_title("A   Implementable frontiers by portfolio method and market",loc="left",fontsize=13)
    axes[1].set_title("B   All retained frontiers by investor size, method and market",loc="left",fontsize=13)
    for ax in axes:
        ax.set(xlabel="Realised gross volatility (% per year)",ylabel="Excess return net of trading costs (% per year)")
        ax.axhline(0,color="#777",lw=.6);ax.grid(alpha=.18)
    fig.suptitle("Combined base Figure 1 | US high-cap, US mid-cap and EU high-cap",fontsize=15)
    fig.text(.5,.016,"2020–2025; A: solid lines at each market's configured fund size, squares at configured risk aversion.\n"
        "B: every fitted investor-size curve. Lines join separate risk-aversion fits; test curves need not be monotone or efficient.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.97))
    save_figure(fig,OUT/"charts/figure1_combined")
    (OUT/"COMBINED_FIGURE_METHODS.md").write_text(
        "# Combined base Figure 1\n\nPanel A compares US high-cap Static ML* and Portfolio ML in all three markets at their configured fund size. "
        "Dotted curves use separately fitted zero-impact rules. Diamonds are same-market value-weighted references. "
        "Panel B shows every retained model's fund-size curve, including US high-cap Static ML*; color identifies size and line style identifies method/market. "
        "Each point varies risk aversion, with model parameters chosen from preceding observations. "
        "Static ML* is retuned separately at every risk/fund-size point. US mid-cap and EU high-cap have no Static ML model. "
        "Returns are annual arithmetic excess returns after modelled costs; risk is annualised gross-return SD. "
        "All curves use 72 months in 2020–2025; fund size is anchored at end-2019. "
        "Cross-market curves represent different stock universes. These are realised fitted strategies, not a known optimal efficient envelope. "
        "No iteration fit or result is part of this base figure.\n",encoding="utf-8")
    return True




if __name__=="__main__":raise SystemExit(main())
