"""Independent saved-result checks for the five-step sequence; no investment models fitted."""
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
    for n in range(1,6):
        folder=ROOT/f'outputs/iteration_{n}/us_large'
        decision=json.loads((folder/'decision.json').read_text())
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
    summary=json.loads((ROOT/'outputs/iteration_summary/final/summary.json').read_text())
    check('final:accepted_ancestry_matches_all_five_decisions',summary['included_iterations']==accepted)
    final=pd.read_parquet(ROOT/'outputs/iteration_summary/final/pf.parquet').sort_values('eom_ret')
    check('final:equals_last_accepted_not_last_tried',np.array_equal(final[['r','tc']],previous[['r','tc']]))
    result=pd.DataFrame(checks)
    result.to_csv(ROOT/'outputs/iteration_summary/independent_verification.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(ROOT/'outputs/iteration_summary/bootstrap_block_sensitivity.csv',index=False)
    print(result.to_string(index=False))
    print('Accepted:',accepted,'Checks:',len(result),'Passed:',int(result.status.eq('PASS').sum()))
    return 0 if result.status.eq('PASS').all() else 2


if __name__=='__main__':raise SystemExit(main())
