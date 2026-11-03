"""Iteration 6: choose training-only cost aversion on pretest actual-cost utility.

The scalar multiplies only the cost term in coefficient learning. It does not
scale observed lambda, realised TC, execution rules or reported returns. Each
candidate's active risk coefficient is calibrated using pretest data. One
multiplier is frozen before the 2020-2025 comparison; all trials are retained.
"""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.ml_comparison.metrics import statistics
from jkmp.portfolio.active import calibrate
from workflow import ROOT

log=logging.getLogger(__name__)


def choose_trial(rows):
    """Predeclared selection: actual-cost utility, then smaller cost preference."""
    table=pd.DataFrame(rows)
    if not np.isfinite(table[['training_cost_multiplier','validation_utility']]).all().all():
        raise ValueError('Nonfinite pretest selection scores')
    return table.sort_values(['validation_utility','training_cost_multiplier'],ascending=[False,True],kind='stable').iloc[0]


def save_result(result,folder):
    folder.mkdir(parents=True,exist_ok=True)
    for name,frame in result.items():
        if name=='chosen':frame.to_csv(folder/'chosen.csv',index=False)
        else:frame.to_parquet(folder/f'{name}.parquet',index=False)


def run_selective(engine,state,folder,spec):
    from iteration_sequence import digest,write_json
    if not state['benchmark_relative'] or state['te_target']!=.01 or state['cost_multiple']!=1.:
        raise ValueError('Iteration 6 requires inherited 1% active setup and unchanged actual costs')
    previous=ROOT/'outputs/iteration_5/us_large/decision.json'
    previous_number=json.loads(previous.read_text())['last_accepted_after'] if previous.exists() else 4
    reference=ROOT/f'outputs/iteration_{previous_number}/us_large'
    source_hashes={name:digest(reference/name) for name in ('pf.parquet','weights.parquet') if (reference/name).exists()}
    initial=dict(state)
    key=hashlib.sha256(json.dumps(dict(source=engine.signature,state=initial,spec=spec,
        reference=source_hashes,
        implementation=digest(Path(__file__)),math=digest(ROOT/'scripts/jkmp/portfolio/portfolio_ml.py')),
        sort_keys=True).encode()).hexdigest()
    marker=folder/'selective_manifest.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['signature']==key and all(digest(folder/n)==h for n,h in old['hashes'].items()):
            state.update(old['selected_state'])
            log.info('Reuse verified iteration 6')
            return pd.read_parquet(folder/'pf.parquet')
    rows=[]
    baseline_gamma=float(pd.read_csv(reference/'calibrated.csv').gamma_te.iloc[0])
    for multiplier in spec['training_cost_candidates']:
        trial=folder/'pretest_trials'/f'cost_{multiplier:g}'
        trial.mkdir(parents=True,exist_ok=True)
        trial_state={**initial,'training_cost_multiplier':float(multiplier)}
        trial_marker=trial/'completion.json'
        stamp=dict(signature=key,multiplier=multiplier)
        if trial_marker.exists() and json.loads(trial_marker.read_text()).get('stamp')==stamp:
            saved=json.loads(trial_marker.read_text())
            if all(digest(trial/n)==h for n,h in saved['hashes'].items()):
                rows.append(saved['row']);log.info('Reuse pretest trial %s',multiplier);continue
        log.info('PRETEST cost preference %s; true execution costs unchanged',multiplier)
        if multiplier==1.:
            gamma_te=baseline_gamma
        else:
            points=[]
            for gamma in (100.,250.,50.,500.,25.,1000.,10.,2500.):
                point=engine.fit(trial_state,trial,calibration=True,gamma_te=gamma)
                points.append(point)
                pd.DataFrame(points).to_csv(trial/'calibration.csv',index=False)
                if len(points)>1 and min(p['te_ex_ante'] for p in points)<=state['te_target']<=max(p['te_ex_ante'] for p in points):break
            calibrated=calibrate(pd.DataFrame(points),[state['te_target']])
            calibrated.to_csv(trial/'calibrated.csv',index=False)
            gamma_te=float(calibrated.gamma_te.iloc[0])
        result=engine.fit(trial_state,trial,calibration=True,gamma_te=gamma_te,validation_result=True)
        pf=result['pf']
        dates=pd.DatetimeIndex(pf.eom_ret)
        if dates.max()>pd.Timestamp(engine.cfg.dates.validation_end):raise ValueError('Test returns entered multiplier selection')
        save_result(result,trial)
        score=statistics(pf)
        row=dict(training_cost_multiplier=multiplier,gamma_te=gamma_te,
                 validation_utility=score['Utility'],validation_net_return=score['R_TC'],
                 validation_net_sharpe=score['SR_net'],validation_cost=score['TC'],
                 validation_turnover=score['Turnover'],validation_months=len(pf),
                 first_return=str(dates.min().date()),last_return=str(dates.max().date()),
                 actual_cost_multiple=state['cost_multiple'])
        rows.append(row)
        names=['pf.parquet','weights.parquet','chosen.csv']
        write_json(trial_marker,dict(stamp=stamp,row=row,hashes={n:digest(trial/n) for n in names}))
        pd.DataFrame(rows).to_csv(folder/'cost_preference_validation.csv',index=False)
    chosen=choose_trial(rows)
    plot_trials(pd.DataFrame(rows),folder)
    multiplier=float(chosen.training_cost_multiplier)
    state['training_cost_multiplier']=multiplier
    write_json(folder/'pretest_selection.json',dict(selected_multiplier=multiplier,
        gamma_te=float(chosen.gamma_te),selection='maximum pretest actual-cost utility; ties smaller multiplier',
        candidates=spec['training_cost_candidates'],validation_last_return=str(chosen.last_return),
        test_returns_used=False,frozen_for_all_test_years=True,actual_cost_multiple=state['cost_multiple']))
    pd.DataFrame([dict(te_target=state['te_target'],gamma_te=float(chosen.gamma_te))]).to_csv(folder/'calibrated.csv',index=False)
    selected_trial=folder/'pretest_trials'/f'cost_{multiplier:g}'
    if (selected_trial/'calibration.csv').exists():
        pd.read_csv(selected_trial/'calibration.csv').to_csv(folder/'calibration.csv',index=False)
    else:
        # The 1x control inherits the original pretest calibration exactly.
        pd.read_csv(reference/'calibration.csv').to_csv(folder/'calibration.csv',index=False)
    if multiplier==1.:
        import shutil
        for name in ('pf.parquet','weights.parquet','chosen.csv','aims.parquet','aim_coefs.parquet','validation.parquet'):
            shutil.copy2(reference/name,folder/name)
    else:
        log.info('Frozen pretest multiplier %s, gamma_TE %s; now fit test portfolio',multiplier,chosen.gamma_te)
        engine.fit(state,folder,gamma_te=float(chosen.gamma_te))
    names=['pf.parquet','weights.parquet','chosen.csv']
    write_json(marker,dict(signature=key,selected_state=state,hashes={n:digest(folder/n) for n in names}))
    return pd.read_parquet(folder/'pf.parquet')


def plot_trials(table,folder):
    """Expose every pretest trial; the 2020-2025 returns do not enter these bars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    x=np.arange(len(table));labels=[f'{c:g}x' for c in table.training_cost_multiplier]
    for ax,column,title in zip(axes,['validation_utility','validation_cost','validation_turnover'],
                              ['Annual utility at gamma 10','Annual actual trading cost','Monthly turnover']):
        ax.bar(x,100*table[column],color='#257a9e')
        ax.set_xticks(x,labels);ax.set_title(title);ax.set_ylabel('Percent');ax.grid(axis='y',alpha=.2)
        ax.set_xlabel('Training-only cost preference')
    fig.suptitle('Iteration 6 pretest selection | 2016-2019 | Actual costs remain 1x')
    fig.tight_layout();out=folder/'charts';out.mkdir(exist_ok=True)
    fig.savefig(out/'pretest_cost_preference.png',dpi=170);plt.close(fig)
