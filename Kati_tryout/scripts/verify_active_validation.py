"""Independent checks of the iteration-4.2/4.3 executed-validation branch.

No fitting/selection implementation is imported. Literal stock accounting,
candidate drift, annual choices, utility bootstrap, selected-book transitions
and descriptive P(test) are reconstructed from saved inputs and outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from workflow import ROOT,config_path
from jkmp.config import load_config

MONTH=pd.offsets.MonthEnd(1)
GAMMA=174.1
BRANCH42=ROOT/'outputs/iteration_4.2/us_large'
BRANCH43=ROOT/'outputs/iteration_4.3/us_large'
KEYS=['r','tc','turnover','inv','shorting']


def cid_key(value):
    if isinstance(value,bytes):value=value.decode()
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and value.is_integer():value=int(value)
    return str(value)


class Checks:
    def __init__(self):self.rows=[]
    def add(self,name,ok,detail=''):
        self.rows.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL',detail=str(detail)))
        return bool(ok)
    def require(self,name,ok,detail=''):
        if not self.add(name,ok,detail):raise ValueError(name+': '+str(detail))
    def close(self,name,actual,expected,rtol=1e-9,atol=1e-11):
        a=np.asarray(actual);b=np.asarray(expected)
        ok=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=rtol,atol=atol)
        error=float(np.max(np.abs(a-b))) if a.shape==b.shape and a.size else None
        return self.add(name,ok,f'max_absolute_error={error}; rtol={rtol}; atol={atol}')


def active_utility(gross,net):
    """Independent sample-variance statistic; annual units throughout."""
    x=np.asarray(gross,dtype=float);z=np.asarray(net,dtype=float)
    return 12*(z.mean(axis=0)-GAMMA*.5*x.var(axis=0,ddof=1))


def bootstrap_utility_se(gross,net,date):
    n=len(gross);rng=np.random.default_rng(20261107+pd.Timestamp(date).year)
    starts=rng.integers(0,n,size=(2000,int(np.ceil(n/6))))
    indices=((starts[:,:,None]+np.arange(6))%n).reshape(2000,-1)[:,:n]
    x=np.asarray(gross)[indices];z=np.asarray(net)[indices]
    values=12*(z.mean(axis=1)-GAMMA*.5*x.var(axis=1,ddof=1))
    # Independent literal-index/variance checks on the same resamples.
    for i in range(10):
        literal=[(int(s)+j)%n for s in starts[i] for j in range(6)][:n]
        a=[float(gross[j]) for j in literal];b=[float(net[j]) for j in literal]
        mean=sum(a)/n;variance=sum((v-mean)**2 for v in a)/(n-1)
        expected=12*(sum(b)/n-GAMMA*.5*variance)
        if not np.isclose(values[i],expected,rtol=1e-10,atol=1e-12):
            raise AssertionError('Literal utility-bootstrap variance does not match')
    return float(np.std(values,ddof=1))


def probability(candidate,reference):
    a=candidate.sort_values('eom_ret').set_index('eom_ret');b=reference.sort_values('eom_ret').set_index('eom_ret')
    if not a.index.equals(b.index):raise ValueError('P(test) calendars differ')
    if np.array_equal(a[['r','tc']].to_numpy(),b[['r','tc']].to_numpy()):return None,(0.,0.)
    n=len(a);rng=np.random.default_rng(20261102)
    starts=rng.integers(0,n,size=(10000,int(np.ceil(n/6))))
    ix=((starts[:,:,None]+np.arange(6))%n).reshape(10000,-1)[:,:n]
    ar=a.r.to_numpy()[ix];ac=a.tc.to_numpy()[ix];br=b.r.to_numpy()[ix];bc=b.tc.to_numpy()[ix]
    delta=np.sqrt(12)*((ar-ac).mean(axis=1)/ar.std(axis=1,ddof=1)-(br-bc).mean(axis=1)/br.std(axis=1,ddof=1))
    if not np.isfinite(delta).all():raise ValueError('Undefined paired Sharpe replicates')
    p=float(np.mean((delta>1e-12).astype(float)+.5*np.isclose(delta,0,rtol=0,atol=1e-12)))
    return p,tuple(float(v) for v in np.quantile(delta,[.025,.975]))


class Inputs:
    def __init__(self):
        self.cfg=load_config(config_path('us_large'));p=self.cfg.paths.processed
        self.panel=pd.read_parquet(p/'panel.parquet',columns=['id','eom','me','ret_ld1','tr_ld1','lambda','in_universe'],
            filters=[('eom','>=',pd.Timestamp('2014-12-31')),('eom','<=',pd.Timestamp('2025-11-30'))])
        if self.panel.duplicated(['id','eom']).any():raise ValueError('Duplicate source panel keys')
        self.by_date={d:g.set_index('id') for d,g in self.panel.groupby('eom',sort=True)}
        self.wealth=pd.read_parquet(p/'wealth.parquet').set_index('eom')
        load=pd.read_parquet(p/'covariance/cov_loadings.parquet')
        factor=pd.read_parquet(p/'covariance/cov_factor.parquet')
        self.factors=[c for c in factor.columns if c not in ('eom','factor')]
        self.loadings={d:g.set_index('id') for d,g in load.groupby('eom')}
        self.factor_cov={d:g.set_index('factor').loc[self.factors,self.factors].to_numpy() for d,g in factor.groupby('eom')}
        rf=pd.read_parquet(self.cfg.paths.raw/'ff_factors_monthly.parquet')
        date='date' if 'date' in rf else 'eom'
        self.rf=pd.Series(rf.rf.to_numpy(),index=pd.to_datetime(rf[date]))
        self.speed_cache={}

    def speed(self,date,ids):
        """Independent eigenvalue start + general solves for the same fixed point."""
        key=(date,tuple(ids))
        if key in self.speed_cache:return self.speed_cache[key]
        rows=self.loadings[date].reindex(ids);X=rows[self.factors].to_numpy()
        sigma=X@self.factor_cov[date]@X.T+np.diag(rows.ivar.to_numpy())
        lam=self.by_date[date]['lambda'].reindex(ids).to_numpy();wealth=float(self.wealth.loc[date,'wealth'])
        root_lam=np.sqrt(lam);n=len(ids);I=np.eye(n)
        x=(GAMMA/wealth)*sigma/root_lam[:,None]/root_lam[None,:]
        eigen,V=np.linalg.eigh(x+2*I)
        start=2/(eigen+np.sqrt(np.maximum(eigen*eigen-4,0.)))
        tilde=(V*start)@V.T
        G=np.ones((n,n))+sigma/(1+float(self.rf.loc[date+MONTH])+self.cfg.investor.mu)**2
        y=np.diag(1+np.diag(G))
        for _ in range(self.cfg.investor.m_iterations):
            tilde=np.linalg.solve(x+y-tilde*G,I)
            tilde=(tilde+tilde.T)*.5
        speed=tilde*root_lam[None,:]/root_lam[:,None]
        self.speed_cache[key]=speed
        return speed


def load_cache(folder,label,inputs,checks):
    catalog=pd.read_csv(folder/'catalog.csv');catalog['_cid']=catalog.cid.map(cid_key)
    checks.require(label+':unique_catalog_cids',not catalog._cid.duplicated().any())
    checks.require(label+':original_1414_candidate_grid',len(catalog)==1414
        and set(catalog.p)=={8,16,32,64,128,256,512} and catalog.groupby('tag').size().eq(707).all())
    checks.require(label+':stable_catalog_order',catalog.cid.is_monotonic_increasing)
    checks.close(label+':original_RFF_variance_grid',sorted(catalog.g.unique()),sorted(inputs.cfg.portfolio_ml.rff_variance_grid),rtol=1e-12,atol=1e-14)
    for (tag,p),rows in catalog.groupby(['tag','p']):
        rows=rows.sort_values('l_no')
        checks.require(f'{label}:{tag}:p{p}:ridge_indices',np.array_equal(rows.l_no,np.arange(len(inputs.cfg.ridge_grid.values))))
        checks.close(f'{label}:{tag}:p{p}:original_ridge_grid',rows.l,inputs.cfg.ridge_grid.values,rtol=1e-12,atol=1e-14)
    flows=pd.read_parquet(folder/'candidate_flows.parquet');flows['eom_ret']=pd.to_datetime(flows.eom_ret)
    flows['_cid']=flows.cid.map(cid_key)
    checks.require(label+':unique_candidate_months',not flows.duplicated(['eom_ret','_cid']).any())
    dates=pd.date_range('2015-01-31','2019-12-31' if label=='pretest' else '2025-12-31',freq='ME')
    checks.require(label+':complete_expected_calendar',pd.DatetimeIndex(sorted(flows.eom_ret.unique())).equals(dates))
    checks.require(label+':same_all_candidate_calendar',flows.groupby('_cid').size().eq(len(dates)).all()
        and set(flows._cid)==set(catalog._cid))
    checks.require(label+':finite_candidate_cashflows',np.isfinite(flows[KEYS+['active_gross','active_net']]).all().all())
    market=pd.read_parquet(folder/'market_pf.parquet').sort_values('eom_ret').set_index('eom_ret')
    checks.require(label+':market_identical_calendar',market.index.equals(dates))
    joined=flows.merge(market[['r','tc']],left_on='eom_ret',right_index=True,suffixes=('','_market'),validate='many_to_one')
    checks.close(label+':active_gross_identity',joined.active_gross,joined.r-joined.r_market)
    checks.close(label+':active_net_identity',joined.active_net,(joined.r-joined.tc)-(joined.r_market-joined.tc_market))
    indexed=flows.set_index(['eom_ret','_cid']);order_refs={};seen=set()
    for tag,tag_catalog in catalog.groupby('tag',sort=False):
        previous=None
        paths=sorted((folder/'orders'/str(tag)).glob('*.npz'))
        checks.require(label+':'+tag+':one_order_file_per_formation',len(paths)==len(dates))
        for path,date_ret in zip(paths,dates):
            date=pd.Timestamp(path.stem);prefix=f'{label}:{tag}:{date.date()}'
            checks.require(prefix+':formation_return_timing',date+MONTH==date_ret)
            with np.load(path,allow_pickle=False) as z:
                ids=z['ids'];W=np.asarray(z['W'],float);H=np.asarray(z['H'],float);ex=z['exit'].astype(bool)
                ret=np.asarray(z['ret'],float);lam=np.asarray(z['lam'],float);wealth=float(np.asarray(z['wealth']).item())
                cids=[cid_key(v) for v in z['cids']]
            checks.require(prefix+':valid_orders_dimensions',W.shape==H.shape==(len(ids),len(cids))
                and len(set(ids))==len(ids) and len(set(cids))==len(cids)
                and len(ret)==len(lam)==len(ex)==len(ids) and np.isfinite(np.r_[W.ravel(),H.ravel(),ret,lam,wealth]).all())
            checks.require(prefix+':tag_catalog_alignment',set(cids)==set(tag_catalog._cid))
            checks.close(prefix+':exit_targets_zero',W[ex],np.zeros_like(W[ex]))
            source=inputs.by_date[date].reindex(ids)
            checks.require(prefix+':all_order_ids_observed',source.index.is_unique and source['lambda'].notna().all())
            checks.close(prefix+':original_impact_inputs',lam,source['lambda'])
            checks.close(prefix+':original_wealth_input',[wealth],[inputs.wealth.loc[date,'wealth']])
            checks.close(prefix+':original_return_inputs',ret,np.where(ex,0.,source.ret_ld1))
            held_ids=ids[~ex]
            universe=inputs.by_date[date].query('in_universe').index
            checks.require(prefix+':original_held_universe',set(held_ids)==set(universe))
            expected_H=np.zeros_like(H)
            if previous is None:
                cap=source.loc[held_ids,'me'].to_numpy();expected_H[~ex]=(cap/cap.sum())[:,None]
                checks.require(prefix+':no_initial_exits',not ex.any())
            else:
                prev_ids,prev_W,prev_date,prev_cids=previous
                checks.require(prefix+':candidate_column_order_stable',cids==prev_cids)
                growth=(1+inputs.by_date[prev_date].tr_ld1.reindex(prev_ids).to_numpy())/(1+float(inputs.wealth.loc[prev_date,'mu_ld1']))
                drift=pd.DataFrame(prev_W*growth[:,None],index=prev_ids)
                expected_H=drift.reindex(ids,fill_value=0.).to_numpy()
                leaving=drift.loc[~drift.index.isin(held_ids)&drift.index.isin(inputs.by_date[date].index)]
                required=set(leaving.index[np.any(leaving.to_numpy()!=0.,axis=1)]) if inputs.cfg.portfolio.charge_exit_tc else set()
                checks.require(prefix+':every_nonzero_charged_exit_present',required.issubset(set(ids[ex])))
                extra=[i for i in ids[ex] if i not in required]
                checks.require(prefix+':extra_union_exits_are_zero',not extra or np.array_equal(
                    H[pd.Index(ids).get_indexer(extra)],np.zeros((len(extra),len(cids)))))
            checks.close(prefix+':literal_prior_weight_drift_and_annual_continuity',H,expected_H)
            values=np.column_stack([np.sum(W*ret[:,None],axis=0),
                .5*wealth*np.sum(lam[:,None]*(W-H)**2,axis=0),np.sum(abs(W-H),axis=0),
                np.sum(abs(W),axis=0),np.sum(np.maximum(-W,0),axis=0)])
            observed=indexed.loc[[(date_ret,c) for c in cids],KEYS].to_numpy()
            checks.close(prefix+':all_literal_stock_cashflow_metrics',values,observed)
            for col,c in enumerate(cids):order_refs[(date_ret,c)]=(path,col);seen.add((date_ret,c))
            previous=(held_ids.copy(),W[~ex].copy(),date,cids)
    checks.require(label+':every_cached_flow_has_raw_orders',seen==set(indexed.index))
    # Direct benchmark calculation, including its own drift, membership exits and costs.
    previous=None
    for date_ret in dates:
        date=date_ret-MONTH;p=inputs.by_date[date];u=p.query('in_universe')
        ids=u.index.to_numpy();b=u.me.to_numpy()/u.me.sum();held=np.zeros(len(ids));exit_cost=exit_turnover=0.
        if previous is None:held=b.copy()
        else:
            old_ids,old_b,old_date=previous
            growth=(1+inputs.by_date[old_date].tr_ld1.reindex(old_ids).to_numpy())/(1+float(inputs.wealth.loc[old_date,'mu_ld1']))
            drift=pd.Series(old_b*growth,index=old_ids);held=drift.reindex(ids,fill_value=0.).to_numpy()
            leaving=drift.loc[~drift.index.isin(ids)&drift.index.isin(p.index)]
            if inputs.cfg.portfolio.charge_exit_tc:
                exit_cost=float(np.sum(p['lambda'].reindex(leaving.index).to_numpy()*leaving.to_numpy()**2))
                exit_turnover=float(np.abs(leaving).sum())
        r=float(np.sum(b*u.ret_ld1.to_numpy()))
        tc=.5*float(inputs.wealth.loc[date,'wealth'])*(float(np.sum(u['lambda'].to_numpy()*(b-held)**2))+exit_cost)
        checks.close(f'{label}:{date.date()}:literal_market_reference',[r,tc],[market.loc[date_ret,'r'],market.loc[date_ret,'tc']])
        if 'turnover' in market:checks.close(f'{label}:{date.date()}:literal_market_turnover',
            [np.abs(b-held).sum()+exit_turnover],[market.loc[date_ret,'turnover']])
        previous=(ids,b,date)
    gross=flows.pivot(index='eom_ret',columns='_cid',values='active_gross').reindex(index=dates,columns=catalog._cid).to_numpy()
    net=flows.pivot(index='eom_ret',columns='_cid',values='active_net').reindex(index=dates,columns=catalog._cid).to_numpy()
    return dict(folder=folder,catalog=catalog,flows=flows,dates=dates,gross=gross,net=net,order_refs=order_refs)


def verify_choices(stage,folder,cache,checks):
    chosen=pd.read_csv(folder/'chosen.csv',parse_dates=['eom_ret']);chosen['_cid']=chosen.cid.map(cid_key)
    audit=pd.read_csv(folder/'selection_audit.csv')
    checks.require(stage+':one_choice_per_year',not chosen.year.duplicated().any())
    checks.require(stage+':all2016_to2025_choices_present',set(range(2016,2026)).issubset(set(chosen.year)))
    catalog=cache['catalog'];dates=cache['dates'];g=cache['gross'];n=cache['net'];by_cid=catalog.set_index('_cid')
    for row in chosen.itertuples():
        date=pd.Timestamp(int(row.year)-1,12,31);prefix=f'{stage}:{row.year}'
        checks.require(prefix+':prior_december_selection_date',row.eom_ret==date)
        keep=dates<=date
        checks.require(prefix+':only_completed_historical_validation_months',int(keep.sum())==12*(row.year-2015))
        scores=active_utility(g[keep],n[keep]);best_index=int(np.argmax(scores));best=catalog.iloc[best_index]
        if stage=='4.2':selected=best;se=0.;threshold=float(scores[best_index]);eligible_count=1
        else:
            se=bootstrap_utility_se(g[keep,best_index],n[keep,best_index],date)
            threshold=float(scores[best_index]-se)
            eligible=catalog.loc[scores>=threshold-1e-15].copy();eligible_count=len(eligible)
            selected=eligible.sort_values(['p','l','g','cid'],ascending=[True,False,True,True],kind='stable').iloc[0]
        chosen_cid=cid_key(row.cid)
        checks.require(prefix+':literal_executed_utility_choice',chosen_cid==selected['_cid'])
        index=int(np.flatnonzero(catalog._cid.to_numpy()==chosen_cid)[0])
        checks.close(prefix+':annual_score_sample_variance_and_units',[row.cum_obj],[scores[index]],rtol=1e-10,atol=1e-12)
        meta=by_cid.loc[chosen_cid]
        checks.require(prefix+':selected_metadata_matches_catalog',row.tag==meta.tag and int(row.p)==int(meta.p)
            and int(row.l_no)==int(meta.l_no) and np.isclose(row.g,meta.g,rtol=1e-12,atol=1e-14)
            and np.isclose(row.l,meta.l,rtol=1e-12,atol=1e-14))
        saved=audit.loc[audit.year.eq(row.year)]
        checks.require(prefix+':single_audit_row',len(saved)==1);a=saved.iloc[0]
        checks.require(prefix+':audit_history_and_cutoff',int(a.history_months)==int(keep.sum())
            and int(a.candidate_count)==len(catalog) and pd.Timestamp(a.selection_month)==date)
        checks.require(prefix+':best_and_selected_audit',cid_key(a.best_cid)==best['_cid'] and cid_key(a.selected_cid)==chosen_cid)
        checks.close(prefix+':best_utility_audit',[a.best_utility],[scores[best_index]],rtol=1e-10,atol=1e-12)
        if stage=='4.3':
            checks.close(prefix+':literal_recomputed_utility_bootstrap_SE',[a.bootstrap_se],[se],rtol=1e-10,atol=1e-12)
            checks.close(prefix+':one_SE_threshold',[a.utility_threshold],[threshold],rtol=1e-10,atol=1e-12)
            checks.require(prefix+':one_SE_eligible_count',int(a.eligible_count)==eligible_count)
    return chosen


def verify_policy(stage,folder,label,chosen,cache,inputs,checks):
    pf=pd.read_parquet(folder/'pf.parquet').sort_values('eom_ret');w=pd.read_parquet(folder/'weights.parquet')
    dates=pd.date_range('2016-01-31','2019-12-31',freq='ME') if label=='pretest' else pd.date_range('2020-01-31','2025-12-31',freq='ME')
    prefix=stage+':'+label
    checks.require(prefix+':headline_calendar',pd.DatetimeIndex(pf.eom_ret).equals(dates))
    checks.require(prefix+':unique_policy_stock_month',not w.duplicated(['id','eom']).any())
    literal=[];previous=None;annual=chosen.set_index('year')
    for date_ret in dates:
        date=date_ret-MONTH;month=w.loc[w.eom.eq(date)].copy();p=inputs.by_date[date]
        checks.require(f'{prefix}:{date.date()}:nonempty_month',not month.empty)
        source=p.reindex(month.id);ids=month.id.to_numpy();ex=month.exit.to_numpy(bool)
        current=ids[~ex];cap=p.me.reindex(current).to_numpy()
        checks.require(f'{prefix}:{date.date()}:original_current_members',set(current)==set(p.query('in_universe').index))
        expected=np.zeros(len(ids))
        if previous is None:
            expected[~ex]=cap/cap.sum()
            checks.require(f'{prefix}:{date.date()}:no_initial_policy_exits',not ex.any())
        else:
            old_ids,old_weights,old_date=previous
            growth=(1+inputs.by_date[old_date].tr_ld1.reindex(old_ids).to_numpy())/(1+float(inputs.wealth.loc[old_date,'mu_ld1']))
            drift=pd.Series(old_weights*growth,index=old_ids)
            expected=drift.reindex(ids,fill_value=0.).to_numpy()
            leaving=drift.loc[~drift.index.isin(current)&drift.index.isin(p.index)&drift.ne(0.)]
            required=set(leaving.index) if inputs.cfg.portfolio.charge_exit_tc else set()
            checks.require(f'{prefix}:{date.date()}:all_charged_policy_exits',set(ids[ex])==required)
        checks.close(f'{prefix}:{date.date()}:own_inherited_weights',month.w_start,expected)
        checks.close(f'{prefix}:{date.date()}:exit_targets_zero',month.loc[month.exit,'w'],np.zeros(int(ex.sum())))
        ret=np.where(ex,0.,source.ret_ld1.to_numpy());impact=source['lambda'].to_numpy();wealth=float(inputs.wealth.loc[date,'wealth'])
        values=[np.sum(month.w.to_numpy()*ret),.5*wealth*np.sum(impact*(month.w-month.w_start).to_numpy()**2),
            np.abs(month.w-month.w_start).sum(),month.w.abs().sum(),np.maximum(-month.w.to_numpy(),0).sum()]
        literal.append(values)
        # Link the chosen shadow coefficients to actual executed weights, rather
        # than incorrectly comparing whole shadow books with different holdings.
        cid=cid_key(annual.loc[date_ret.year,'cid']);path,col=cache['order_refs'][(date_ret,cid)]
        with np.load(path,allow_pickle=False) as z:
            shadow=pd.DataFrame(dict(id=z['ids'],W=z['W'][:,col],H=z['H'][:,col],exit=z['exit'].astype(bool))).set_index('id')
        aligned=shadow.reindex(current)
        checks.require(f'{prefix}:{date.date()}:chosen_candidate_all_current_stocks',aligned[['W','H']].notna().all().all())
        expected_w=aligned.W.to_numpy()+inputs.speed(date,current)@(month.loc[~month.exit,'w_start'].to_numpy()-aligned.H.to_numpy())
        checks.close(f'{prefix}:{date.date()}:independent_chosen_candidate_execution',month.loc[~month.exit,'w'],expected_w,rtol=2e-9,atol=2e-10)
        previous=(current.copy(),month.loc[~month.exit,'w'].to_numpy(),date)
    checks.close(prefix+':literal_stock_return_cost_turnover_leverage_shorting',literal,pf[KEYS].to_numpy())
    return pf


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    checks=Checks();OUT=BRANCH43;OUT.mkdir(parents=True,exist_ok=True);error=None
    try:
        inputs=Inputs()
        caches={name:load_cache(BRANCH42/'candidate_cache'/name,name,inputs,checks) for name in ('pretest','full')}
        pre=caches['pretest']['flows'].set_index(['eom_ret','_cid']).sort_index()
        full=caches['full']['flows'].set_index(['eom_ret','_cid']).sort_index()
        checks.require('candidate_cache:pretest_full_catalog_identical',caches['pretest']['catalog'].equals(caches['full']['catalog']))
        checks.close('candidate_cache:pretest_full_prefix_equal',pre[KEYS+['active_gross','active_net']],
            full.loc[pre.index,KEYS+['active_gross','active_net']])
        previous=pd.read_parquet(ROOT/'outputs/iteration_4/us_large/pf.parquet')
        anchor=json.loads((ROOT/'outputs/iteration_4/us_large/decision.json').read_text())
        checks.require('4:direct_Base_parent_without_gamma5',anchor['inherited_iterations']==[1]
            and anchor['included_after']==[1,4] and anchor['candidate_state']['gamma']==inputs.cfg.investor.gamma)
        base=pd.read_parquet(ROOT/'outputs/base/us_large/portfolio_ml/rff/pf.parquet')
        anchor_p,_=probability(previous,base)
        checks.close('4:literal_P_test_vs_Base',[anchor['p_better']],[anchor_p],rtol=0,atol=1e-12)
        w=pd.read_parquet(ROOT/'outputs/iteration_4/us_large/weights.parquet')
        stock=w.merge(inputs.panel[['id','eom','ret_ld1','lambda']],on=['id','eom'],how='left',validate='many_to_one').merge(inputs.wealth.reset_index(),on='eom',validate='many_to_one')
        checks.require('4:complete_stock_inputs',len(stock)==len(w) and stock['lambda'].notna().all() and stock.loc[~stock.exit,'ret_ld1'].notna().all())
        literal=pd.DataFrame(dict(eom=stock.eom,r=stock.w*stock.ret_ld1.where(~stock.exit,0),tc=.5*stock.wealth*stock['lambda']*(stock.w-stock.w_start)**2)).groupby('eom').sum()
        literal.index+=MONTH
        checks.close('4:literal_stock_accounting',literal.loc[previous.eom_ret,['r','tc']],previous[['r','tc']])
        for stage,folder in [('4.2',BRANCH42),('4.3',BRANCH43)]:
            protocol=json.loads((folder/'protocol.json').read_text());decision=json.loads((folder/'decision.json').read_text())
            checks.require(stage+':correct_Base_anchored_ancestry',decision['inherited_iterations']==([1,4] if stage=='4.2' else [1,4,'4.2']) and 2 not in decision['included_after'])
            checks.require(stage+':same_frozen_active_risk_coefficient',float(protocol['gamma_te'])==GAMMA)
            checks.require(stage+':sample_gross_active_variance_declared',
                protocol.get('mean_variance_risk')=='gross_active_sample_variance'
                or 'sample_var(active_gross,ddof=1)' in protocol.get('objective',''))
            checks.require(stage+':historical_selection_starts2015',
                int(protocol.get('selection_prefixstarts',pd.Timestamp(protocol['candidate_return_start']).year))==2015)
            checks.require(stage+':P_test_bootstrap_protocol',protocol['p_test_reps']==10000
                and protocol['p_test_block_months']==6 and protocol['p_test_seed']==20261102
                and protocol['p_test_is_retention_gate'] is False)
            if stage=='4.3':checks.require('4.3:fixed_one_SE_bootstrap_rule',protocol['bootstrap_reps']==2000
                and protocol['bootstrap_block_months']==6 and protocol['bootstrap_seed']==20261107)
            chosen=verify_choices(stage,folder,caches['full'],checks)
            verify_policy(stage,folder/'pretest','pretest',chosen,caches['pretest'],inputs,checks)
            current=verify_policy(stage,folder,'test',chosen,caches['full'],inputs,checks)
            p,interval=probability(current,previous)
            checks.require(stage+':retained_without_80percent_gate',decision['accepted'] is True
                and 'threshold' in decision and decision['threshold'] is None)
            if p is None:checks.require(stage+':identical_P_is_NA',decision['p_better'] is None)
            else:checks.close(stage+':literal_P_test_vs_previous',[decision['p_better']],[p],rtol=0,atol=1e-12)
            if isinstance(decision.get('bootstrap'),dict):
                record=decision['bootstrap']
                if 'ci_low' in record and 'ci_high' in record:checks.close(stage+':literal_P_test_Sharpe_difference_interval',
                    [record['ci_low'],record['ci_high']],interval,rtol=1e-10,atol=1e-12)
            previous=current
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';checks.add('verification_completed',False,error)
    frame=pd.DataFrame(checks.rows);frame.to_csv(OUT/'independent_verification.csv',index=False)
    passed=int(frame.status.eq('PASS').sum());report=dict(status='PASS' if passed==len(frame) else 'FAIL',
        checks=len(frame),passed=passed,models_fitted=False,error=error,
        actual_costs_and_execution_reconstructed=True,choices_reconstructed_without_selector_import=True)
    (OUT/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    if passed<len(frame):print(frame.loc[frame.status.eq('FAIL')].to_string(index=False))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
