"""Independent saved-result checks for the ten-step sequence; no investment models fitted."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from workflow import ROOT,config_path
from jkmp.config import load_config


def main():
    protocol=json.loads((ROOT/'configs/us_high_cap_iterations.json').read_text())
    cfg=load_config(config_path('us_large'))
    panel=pd.read_parquet(cfg.paths.processed/'panel.parquet',columns=['id','eom','ret_ld1','lambda'])
    wealth=pd.read_parquet(cfg.paths.processed/'wealth.parquet',columns=['eom','wealth'])
    base=ROOT/'outputs/base/us_large/portfolio_ml/rff'
    previous=pd.read_parquet(base/'pf.parquet').sort_values('eom_ret')
    accepted=[1];checks=[];sensitivity=[]
    def check(name,ok):
        checks.append(dict(check=name,status='PASS' if bool(ok) else 'FAIL'))
    for n in range(1,11):
        folder=ROOT/f'outputs/iteration_{n}/us_large'
        decision=json.loads((folder/'decision.json').read_text())
        if n==4 and next(s for s in protocol['iterations'] if s['number']==4).get('parent_iteration')==1:
            accepted=[1];previous=pd.read_parquet(base/'pf.parquet').sort_values('eom_ret')
        if n>=5 and decision['inherited_iterations']!=accepted and 2 in decision['inherited_iterations']:
            # Retained historical trial: validate against its archived parent,
            # rather than pretending it has been rerun on the revised branch.
            accepted=list(decision['inherited_iterations'])
            previous=pd.read_parquet(ROOT/'outputs/revision_history/before_iteration4_base_parent/iteration_4/pf.parquet').sort_values('eom_ret')
        pf=pd.read_parquet(folder/'pf.parquet').sort_values('eom_ret')
        w=pd.read_parquet(folder/'weights.parquet')
        state=decision['candidate_state']
        check(f'{n}:complete_identical_test_calendar',pd.DatetimeIndex(pf.eom_ret).equals(pd.date_range('2020-01-31','2025-12-31',freq='ME')))
        check(f'{n}:inherits_only_accepted',decision['inherited_iterations']==accepted)
        d=w.merge(panel,on=['id','eom'],how='left',validate='many_to_one').merge(wealth,on='eom',validate='many_to_one')
        check(f'{n}:all_held_targets_observed_or_original_explicit_zero',d.loc[~d.exit,'ret_ld1'].notna().all())
        # Iteration 3 can introduce members previously unheld: recover only its
        # explicitly saved zero-return assumptions, never invent a target here.
        if d.loc[~d.exit,'ret_ld1'].isna().any():raise ValueError('Accounting requires an explicit new-member missing-return audit')
        cost=.5*d.wealth*d['lambda']*state['cost_multiple']*(d.w-d.w_start)**2
        gross=d.w*d.ret_ld1.where(~d.exit,0.)
        recomputed=pd.DataFrame(dict(eom=d.eom,r=gross,tc=cost,turnover=(d.w-d.w_start).abs())).groupby('eom').sum()
        recomputed.index+=pd.offsets.MonthEnd(1)
        observed=pf.set_index('eom_ret')
        check(f'{n}:independent_stock_return_cost_turnover_sums',np.allclose(recomputed.loc[observed.index],observed[recomputed.columns],rtol=1e-10,atol=1e-12))
        # Literal additive path, separate from the shared reporting helper.
        total=peak=0.;worst=0.
        for value in (pf.r-pf.tc):
            total+=value;peak=max(peak,total);worst=min(worst,total-peak)
        reported=pd.read_csv(folder/'performance_summary.csv').set_index('strategy').loc['Candidate']
        check(f'{n}:additive_cumulative_and_drawdown',reported.path_convention=='additive'
              and np.isclose(reported.cumulative_net_excess,total,rtol=1e-12,atol=1e-12)
              and np.isclose(reported.max_drawdown_excess,worst,rtol=1e-12,atol=1e-12))
        if n==1:
            check('1:bitwise_base_returns',np.array_equal(pf[['r','tc']].to_numpy(),previous[['r','tc']].to_numpy()))
            check('1:p_is_not_applicable',decision['p_better'] is None)
        else:
            a=pf.r.to_numpy();ac=pf.tc.to_numpy();b=previous.r.to_numpy();bc=previous.tc.to_numpy()
            for block in (3,6,12):
                rng=np.random.default_rng(protocol['bootstrap_seed'])
                starts=rng.integers(0,len(a),size=(protocol['bootstrap_reps'],int(np.ceil(len(a)/block))))
                index=((starts[:,:,None]+np.arange(block))%len(a)).reshape(protocol['bootstrap_reps'],-1)[:,:len(a)]
                delta=np.sqrt(12)*((a[index]-ac[index]).mean(axis=1)/a[index].std(axis=1,ddof=1)
                                   -(b[index]-bc[index]).mean(axis=1)/b[index].std(axis=1,ddof=1))
                p=float(np.mean((delta>1e-12)+.5*np.isclose(delta,0,rtol=0,atol=1e-12)))
                sensitivity.append(dict(iteration=n,block_months=block,P_better=p,passes_80=p>=.8))
                if block==6:
                    if decision['p_better'] is None:check(f'{n}:identical_candidate',np.allclose(a-ac,b-bc,rtol=0,atol=1e-12))
                    else:check(f'{n}:literal_paired_bootstrap_probability',abs(p-decision['p_better'])<1e-12)
                    check(f'{n}:acceptance_rule',decision['accepted']==(decision['p_better'] is not None and p>=.8))
        if decision['accepted'] and n not in accepted:accepted.append(n)
        check(f'{n}:included_after_matches_decision',accepted==decision['included_after'])
        if decision['accepted']:previous=pf
        if state['benchmark_relative']:
            calibration=pd.read_csv(folder/'calibration.csv')
            check(f'{n}:calibration_uses_only_pretest_returns',pd.to_datetime(calibration.last_return).le(pd.Timestamp(cfg.dates.validation_end)).all())
        chosen=pd.read_csv(folder/'chosen.csv',parse_dates=['eom_ret']) if n>1 else None
        if chosen is not None:check(f'{n}:settings_chosen_before_forecast_year',(chosen.eom_ret.dt.year+1).eq(chosen.year).all())
        if n==6:
            selection=json.loads((folder/'pretest_selection.json').read_text())
            trials=pd.read_csv(folder/'cost_preference_validation.csv')
            best=trials.sort_values(['validation_utility','training_cost_multiplier'],ascending=[False,True]).iloc[0]
            check('6:preference_selected_without_test_returns',not selection['test_returns_used']
                  and pd.to_datetime(trials.last_return).le(pd.Timestamp(cfg.dates.validation_end)).all())
            check('6:preference_maximises_pretest_actual_cost_utility',selection['selected_multiplier']==best.training_cost_multiplier)
            check('6:actual_costs_unchanged',state['cost_multiple']==1. and trials.actual_cost_multiple.eq(1.).all())
            check('6:selected_preference_frozen_for_test',selection['frozen_for_all_test_years']
                  and state['training_cost_multiplier']==selection['selected_multiplier'])
            check('6:inherits_accepted_not_rejected_trials',decision['inherited_iterations']==[1,2,4])
            for trial in trials.itertuples():
                path=folder/'pretest_trials'/f'cost_{trial.training_cost_multiplier:g}'
                tw=pd.read_parquet(path/'weights.parquet')
                tp=pd.read_parquet(path/'pf.parquet').sort_values('eom_ret').set_index('eom_ret')
                stock=tw.merge(panel,on=['id','eom'],how='left',validate='many_to_one').merge(wealth,on='eom',validate='many_to_one')
                cash=pd.DataFrame(dict(eom=stock.eom,r=stock.w*stock.ret_ld1.where(~stock.exit,0.),
                     tc=.5*stock.wealth*stock['lambda']*(stock.w-stock.w_start)**2)).groupby('eom').sum()
                cash.index+=pd.offsets.MonthEnd(1)
                check(f'6:pretest_{trial.training_cost_multiplier:g}x_actual_stock_cost_accounting',
                      np.allclose(cash.loc[tp.index],tp[['r','tc']],rtol=1e-10,atol=1e-12))
                utility=12*((tp.r-tp.tc).mean()-5*tp.r.var(ddof=1))
                check(f'6:pretest_{trial.training_cost_multiplier:g}x_independent_utility',
                      abs(utility-trial.validation_utility)<1e-12)
        if n==7:
            selection=json.loads((folder/'pretest_selection.json').read_text())
            trials=pd.read_csv(folder/'risk_half_life_validation.csv')
            ranked=trials.assign(change=trials.half_life_days.ne(126)).sort_values(['validation_utility','change','half_life_days'],ascending=[False,True,True])
            control=json.loads((folder/'risk_control_verification.json').read_text())
            check('7:original_126_day_control_reconstructed',control['status']=='PASS' and any(c['original_control_matched'] for c in control['checks']))
            check('7:pretest_only_frozen_choice',not selection['test_returns_used'] and selection['frozen_for_all_test_years']
                  and pd.to_datetime(trials.last_return).le(pd.Timestamp(cfg.dates.validation_end)).all())
            check('7:choice_maximises_pretest_utility',selection['selected_half_life_days']==int(ranked.iloc[0].half_life_days))
            check('7:inherits_no_rejected_changes',decision['inherited_iterations']==[1,2,4] and state.get('training_cost_multiplier',1.)==1. and state['cost_multiple']==1.)
            check('7:chosen_half_life_matches_frozen_state',state['idio_half_life_days']==selection['selected_half_life_days'])
            original_load=pd.read_parquet(cfg.paths.processed/'covariance/cov_loadings.parquet').sort_values(['eom','id']).reset_index(drop=True)
            original_factor=pd.read_parquet(cfg.paths.processed/'covariance/cov_factor.parquet').sort_values(['eom','factor']).reset_index(drop=True)
            for trial in trials.itertuples():
                risk_path=ROOT/f'data/processed/us_large/iteration_7_covariance/hl_{trial.half_life_days}'
                candidate_load=pd.read_parquet(risk_path/'cov_loadings.parquet').sort_values(['eom','id']).reset_index(drop=True)
                candidate_factor=pd.read_parquet(risk_path/'cov_factor.parquet').sort_values(['eom','factor']).reset_index(drop=True)
                check(f'7:{trial.half_life_days}d_only_idiosyncratic_variance_changes',
                      original_load.drop(columns='ivar').equals(candidate_load.drop(columns='ivar')) and original_factor.equals(candidate_factor))
                if trial.half_life_days==126:check('7:126d_control_exact',original_load.equals(candidate_load))
                path=folder/'pretest_trials'/f'hl_{trial.half_life_days}'
                tw=pd.read_parquet(path/'weights.parquet')
                tp=pd.read_parquet(path/'pf.parquet').sort_values('eom_ret').set_index('eom_ret')
                stock=tw.merge(panel,on=['id','eom'],how='left',validate='many_to_one').merge(wealth,on='eom',validate='many_to_one')
                cash=pd.DataFrame(dict(eom=stock.eom,r=stock.w*stock.ret_ld1.where(~stock.exit,0.),
                     tc=.5*stock.wealth*stock['lambda']*(stock.w-stock.w_start)**2)).groupby('eom').sum()
                cash.index+=pd.offsets.MonthEnd(1)
                check(f'7:{trial.half_life_days}d_pretest_stock_accounting',np.allclose(cash.loc[tp.index],tp[['r','tc']],rtol=1e-10,atol=1e-12))
                utility=12*((tp.r-tp.tc).mean()-5*tp.r.var(ddof=1))
                check(f'7:{trial.half_life_days}d_independent_pretest_utility',abs(utility-trial.validation_utility)<1e-12)
        if n==8:
            frozen=json.loads((folder/'pretest_selection.json').read_text())
            audit=pd.read_csv(folder/'selection_audit.csv',parse_dates=['selection_month','validation_last_return'])
            flows=pd.read_parquet(folder/'validation.parquet')
            check('8:fixed_rule_and_pretest_TE',frozen['rule_frozen'] and not frozen['test_returns_used_for_rule_or_TE']
                  and pd.Timestamp(frozen['calibration_last_return'])<=pd.Timestamp(cfg.dates.validation_end))
            check('8:inherits_no_rejected_changes',decision['inherited_iterations']==[1,2,4]
                  and state.get('training_cost_multiplier',1.)==1. and 'idio_half_life_days' not in state and state['half_lives']==[None])
            for selected in audit.itertuples():
                history=flows.loc[flows.eom_ret.le(selected.selection_month)].sort_values('eom_ret')
                scores=history.groupby(['tag','g','p','l_no','l','h'],sort=False).agg(mean=('obj','mean'),months=('obj','size')).reset_index()
                best=scores.loc[scores['mean'].idxmax()]
                sample=history.loc[history.tag.eq(best.tag)&history.p.eq(best.p)&history.l_no.eq(best.l_no)&history.h.eq(best.h),'obj'].to_numpy()
                rng=np.random.default_rng(int(selected.bootstrap_seed));means=[];block=int(selected.block_months)
                for _ in range(int(selected.bootstrap_reps)):
                    starts=rng.integers(0,len(sample),size=int(np.ceil(len(sample)/block)))
                    index=[(int(start)+j)%len(sample) for start in starts for j in range(block)][:len(sample)]
                    means.append(float(np.mean(sample[index])))
                se=float(np.std(means,ddof=1));threshold=float(best['mean']-se)
                eligible=scores.loc[scores['mean'].ge(threshold-1e-15)]
                actual=eligible.sort_values(['p','l','g','tag'],ascending=[True,False,True,True],kind='stable').iloc[0]
                check(f'8:{selected.year}:literal_block_SE_and_eligibility',
                      abs(se-selected.bootstrap_se)<1e-12 and abs(threshold-selected.utility_threshold)<1e-12
                      and len(eligible)==selected.eligible_count and len(scores)==1414)
                check(f'8:{selected.year}:simplest_eligible_selected_without_future',
                      actual.p==selected.selected_p and np.isclose(actual.l,selected.selected_l)
                      and np.isclose(actual.g,selected.selected_g) and selected.selection_month.year+1==selected.year)
        if n==9:
            frozen=json.loads((folder/'pretest_selection.json').read_text())
            check('9:pretest_only_TE_and_unchanged_return_risk_execution',
                  not frozen['test_returns_used'] and frozen['return_risk_proxies_unchanged']
                  and frozen['actual_cost_formula_unchanged']
                  and pd.Timestamp(frozen['calibration_last_return'])<=pd.Timestamp(cfg.dates.validation_end))
            check('9:inherits_no_rejected_changes',decision['inherited_iterations']==[1,2,4]
                  and state.get('training_cost_multiplier',1.)==1. and state['cost_multiple']==1.
                  and 'idio_half_life_days' not in state and state.get('selection_rule')!='one_se'
                  and state['half_lives']==[None] and state['cost_learning']=='recursive_benchmark_netting')
            for source in [folder,*sorted((folder/'calibration_trials').glob('gamma_*'))]:
                prefix='9:'+source.name
                audit=pd.read_csv(source/'cost_audit.csv',parse_dates=['eom'])
                zero_w=pd.read_parquet(source/'baseline_weights.parquet')
                zero_pf=pd.read_parquet(source/'baseline_pf.parquet').set_index('eom_ret')
                stock=zero_w.merge(panel,on=['id','eom'],how='left',validate='many_to_one').merge(wealth,on='eom',validate='many_to_one')
                cash=pd.DataFrame(dict(eom=stock.eom,r=stock.w*stock.ret_ld1.where(~stock.exit,0.),
                     tc=.5*stock.wealth*stock['lambda']*(stock.w-stock.w_start)**2)).groupby('eom').sum()
                cash.index+=pd.offsets.MonthEnd(1)
                check(prefix+':independent_zero_signal_stock_accounting',
                      np.allclose(cash.loc[zero_pf.index],zero_pf[['r','tc']],rtol=1e-10,atol=1e-12))
                check(prefix+':every_month_has_two_RFF_audits_and_three_literal_probes',
                      audit.groupby('eom').size().eq(2).all() and audit.polynomial_probes.eq(3).all()
                      and np.isfinite(audit.cost_polynomial_error).all() and audit.cost_polynomial_error.max()<1e-10)
                check(prefix+':baseline_cost_constant_independent_of_RFF',
                      audit.groupby('eom').baseline_cost.nunique().eq(1).all()
                      and np.allclose(audit.groupby('eom').baseline_cost.first().to_numpy(),zero_pf.tc.to_numpy(),rtol=1e-10,atol=1e-12))
                check(prefix+':cost_curvature_nonnegative',audit.cost_quadratic_trace.ge(-1e-12).all())
                dates=audit.eom.drop_duplicates().sort_values()
                check(prefix+':consecutive_historical_months',pd.DatetimeIndex(dates).equals(pd.date_range(dates.min(),dates.max(),freq='ME')))
                for tag,rows in audit.groupby('tag'):
                    rows=rows.sort_values('eom')
                    prior=pd.to_datetime(rows.prior_formation_input)
                    check(prefix+':'+tag+':prior_formation_timing',pd.isna(prior.iloc[0])
                          and np.array_equal(prior.iloc[1:].to_numpy(),(rows.eom.iloc[1:]-pd.offsets.MonthEnd(1)).to_numpy()))
                if source!=folder:
                    trial_pf=pd.read_parquet(source/'pf.parquet').set_index('eom_ret')
                    tw=pd.read_parquet(source/'weights.parquet')
                    ts=tw.merge(panel,on=['id','eom'],how='left',validate='many_to_one').merge(wealth,on='eom',validate='many_to_one')
                    actual=pd.DataFrame(dict(eom=ts.eom,r=ts.w*ts.ret_ld1.where(~ts.exit,0.),
                         tc=.5*ts.wealth*ts['lambda']*(ts.w-ts.w_start)**2)).groupby('eom').sum()
                    actual.index+=pd.offsets.MonthEnd(1)
                    check(prefix+':pretest_calendar_and_actual_stock_accounting',
                          trial_pf.index.max()<=pd.Timestamp(cfg.dates.validation_end)
                          and np.allclose(actual.loc[trial_pf.index],trial_pf[['r','tc']],rtol=1e-10,atol=1e-12))
        if n==10:
            frozen=json.loads((folder/'pretest_selection.json').read_text())
            check('10:five_fixed_equal_weight_seeds_not_selected_by_returns',frozen['seeds']==[1,2,3,4,5]
                  and frozen['seed_weights']==[.2]*5 and not frozen['seeds_selected_by_performance']
                  and not frozen['seed_count_selected'] and not frozen['test_returns_used'])
            check('10:pretest_TE_and_unchanged_learning_execution',frozen['same_gamma_for_all_members']
                  and frozen['learning_and_actual_cost_equations_unchanged']
                  and pd.Timestamp(frozen['calibration_last_return'])<=pd.Timestamp(cfg.dates.validation_end))
            check('10:inherits_no_rejected_changes',decision['inherited_iterations']==[1,2,4]
                  and state.get('training_cost_multiplier',1.)==1. and state['cost_multiple']==1.
                  and 'idio_half_life_days' not in state and 'cost_learning' not in state
                  and state.get('selection_rule')!='one_se' and state['half_lives']==[None]
                  and state['ensemble_rule']=='equal_seed_aims')
            sources=[folder,folder/'pretest_seed_sensitivity',*sorted((folder/'calibration_trials').glob('gamma_*'))]
            for source in sources:
                prefix='10:'+source.name
                combined=pd.read_parquet(source/'weights.parquet').set_index(['eom','id','exit'])
                cp=pd.read_parquet(source/'pf.parquet').set_index('eom_ret')
                aims=[];weights=[];cashflows=[];gammas=[]
                for seed in frozen['seeds']:
                    path=source/'members'/f'seed_{seed}'
                    tw=pd.read_parquet(path/'weights.parquet')
                    tp=pd.read_parquet(path/'pf.parquet').set_index('eom_ret')
                    stock=tw.merge(panel,on=['id','eom'],how='left',validate='many_to_one').merge(wealth,on='eom',validate='many_to_one')
                    cash=pd.DataFrame(dict(eom=stock.eom,r=stock.w*stock.ret_ld1.where(~stock.exit,0.),
                         tc=.5*stock.wealth*stock['lambda']*(stock.w-stock.w_start)**2)).groupby('eom').sum()
                    cash.index+=pd.offsets.MonthEnd(1)
                    check(prefix+f':seed_{seed}:literal_stock_accounting',np.allclose(cash.loc[tp.index],tp[['r','tc']],rtol=1e-10,atol=1e-12))
                    chosen_member=pd.read_csv(path/'chosen.csv',parse_dates=['eom_ret'])
                    flows=pd.read_parquet(path/'validation.parquet')
                    dec=flows.loc[flows.eom_ret.dt.month.eq(12)&flows.cum_obj.notna()]
                    best=dec.loc[dec.groupby('eom_ret').cum_obj.idxmax()].sort_values('eom_ret')
                    selected=chosen_member.sort_values('eom_ret')
                    check(prefix+f':seed_{seed}:original_annual_max_utility_and_past_only',
                          len(selected)==len(best) and selected.eom_ret.dt.year.add(1).eq(selected.year).all()
                          and np.array_equal(selected[['tag','p','l_no']].to_numpy(),best[['tag','p','l_no']].to_numpy())
                          and np.allclose(selected.cum_obj,best.cum_obj,rtol=1e-11,atol=1e-12))
                    aims.append(pd.read_parquet(path/'aims.parquet').assign(seed=seed))
                    weights.append(tw.set_index(['eom','id','exit'])[['w','w_start']].add_suffix(f'_{seed}'))
                    cashflows.append(tp[['r','tc']])
                    gammas.append(json.loads((path/'completion.json').read_text())['gamma_te'])
                check(prefix+':all_members_share_same_gamma',len(set(gammas))==1
                      and (source!=folder or gammas[0]==frozen['gamma_te']))
                targets=pd.concat(aims).groupby(['eom','id']).agg(w_aim=('w_aim','mean'),members=('seed','nunique'))
                actual_aim=pd.read_parquet(source/'aims.parquet').set_index(['eom','id']).sort_index()
                check(prefix+':equal_stock_aims_every_seed_every_month',targets.members.eq(5).all()
                      and targets.index.equals(actual_aim.index)
                      and np.allclose(targets.w_aim,actual_aim.w_aim,rtol=1e-11,atol=1e-12))
                joined=pd.concat(weights,axis=1).fillna(0.).sort_index()
                ordered=combined.reindex(joined.index).fillna(0.)
                starts=np.column_stack([joined[f'w_start_{s}'] for s in frozen['seeds']])
                ends=np.column_stack([joined[f'w_{s}'] for s in frozen['seeds']]);orders=ends-starts
                check(prefix+':literal_mean_executed_and_inherited_weights',
                      np.allclose(ordered.w,ends.mean(axis=1),rtol=1e-10,atol=1e-12)
                      and np.allclose(ordered.w_start,starts.mean(axis=1),rtol=1e-10,atol=1e-12))
                stock=ordered.reset_index().merge(panel,on=['id','eom'],validate='many_to_one').merge(wealth,on='eom',validate='many_to_one')
                scale=.5*stock.wealth.to_numpy()*stock['lambda'].to_numpy()
                mean_cost=scale*np.mean(orders**2,axis=1)
                cost=scale*(stock.w-stock.w_start).to_numpy()**2
                variance=scale*np.mean((orders-orders.mean(axis=1)[:,None])**2,axis=1)
                literal=pd.DataFrame(dict(eom=stock.eom,mean_cost=mean_cost,cost=cost,variance=variance)).groupby('eom').sum()
                saved=pd.read_csv(source/'cost_netting_audit.csv',parse_dates=['eom']).set_index('eom')
                check(prefix+':literal_combined_cost_and_Jensen_variance_identity',
                      np.allclose(literal[['mean_cost','cost','variance']],saved[['mean_member_cost','ensemble_cost','variance_cost_saving']],rtol=1e-10,atol=1e-12)
                      and np.allclose(literal.mean_cost-literal.cost,literal.variance,rtol=1e-10,atol=1e-12)
                      and literal.variance.ge(-1e-12).all())
                literal.index+=pd.offsets.MonthEnd(1)
                mean_cash=np.mean([c.to_numpy() for c in cashflows],axis=0)
                check(prefix+':gross_return_average_costs_from_combined_orders',
                      np.allclose(cp.r,mean_cash[:,0],rtol=1e-10,atol=1e-12)
                      and np.allclose(cp.tc,literal.cost,rtol=1e-10,atol=1e-12)
                      and np.allclose(mean_cash[:,1]-cp.tc,literal.variance,rtol=1e-10,atol=1e-12))
                if source!=folder:check(prefix+':pretest_only',cp.index.max()<=pd.Timestamp(cfg.dates.validation_end))
            diagnostic=json.loads((folder/'pretest_seed_sensitivity/diagnostic.json').read_text())
            check('10:original_seed_pretest_control_and_no_seed_cherry_picking',diagnostic['seed_1_matches_original_control']
                  and not diagnostic['seeds_selected_by_performance'] and diagnostic['pretest_only'])
            diag=folder/'pretest_seed_sensitivity'
            one=pd.read_parquet(diag/'members/seed_1/pf.parquet').sort_values('eom_ret')
            control=pd.read_parquet(ROOT/'outputs/iteration_6/us_large/pretest_trials/cost_1/pf.parquet').sort_values('eom_ret')
            check('10:independent_original_seed_control_reproduction',one.eom_ret.equals(control.eom_ret)
                  and np.allclose(one[['r','tc']],control[['r','tc']],rtol=1e-10,atol=1e-12))
            market=pd.read_parquet(diag/'market_reference.parquet').sort_values('eom_ret')
            pairs=pd.read_csv(diag/'pairwise_seed_sensitivity.csv')
            for pair in pairs.itertuples():
                a=pd.read_parquet(diag/f'members/seed_{pair.seed_a}/pf.parquet').sort_values('eom_ret')
                b=pd.read_parquet(diag/f'members/seed_{pair.seed_b}/pf.parquet').sort_values('eom_ret')
                active_a=(a.r-a.tc).to_numpy()-(market.r-market.tc).to_numpy()
                active_b=(b.r-b.tc).to_numpy()-(market.r-market.tc).to_numpy()
                check(f'10:pretest_pair_{pair.seed_a}_{pair.seed_b}:independent_active_correlation',
                      np.isclose(np.corrcoef(active_a,active_b)[0,1],pair.net_active_return_correlation,rtol=1e-10,atol=1e-12))
    summary=json.loads((ROOT/'outputs/iteration_summary/final/summary.json').read_text())
    if summary.get('research_branch'):
        accepted=[1,4,'4.2','4.3']
        previous=pd.read_parquet(ROOT/'outputs/iteration_4.3/us_large/pf.parquet').sort_values('eom_ret')
    check('final:accepted_ancestry_matches_declared_branch',summary['included_iterations']==accepted)
    final=pd.read_parquet(ROOT/'outputs/iteration_summary/final/pf.parquet').sort_values('eom_ret')
    check('final:equals_last_accepted_not_last_tried',np.array_equal(final[['r','tc']],previous[['r','tc']]))
    result=pd.DataFrame(checks)
    result.to_csv(ROOT/'outputs/iteration_summary/independent_verification.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT/'outputs/iteration_summary/bootstrap_block_sensitivity.csv',index=False)
    print(result.to_string(index=False))
    print('Accepted:',accepted,'Checks:',len(result),'Passed:',int(result.status.eq('PASS').sum()))
    return 0 if result.status.eq('PASS').all() else 2


if __name__=='__main__':raise SystemExit(main())
