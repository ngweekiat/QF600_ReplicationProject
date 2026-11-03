"""Create presentation panels from saved iteration returns; no model fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflow import ROOT
from analysis.ml_comparison.metrics import additive_path

BASE = ROOT / 'outputs/base/us_large/portfolio_ml/rff'
PROTOCOL = ROOT / 'configs/us_high_cap_iterations.json'


def build(number):
    from iteration_sequence import paired_probability
    folder = ROOT / f'outputs/iteration_{number}/us_large'
    frames = {
        'Base Portfolio ML': pd.read_parquet(BASE / 'pf.parquet'),
        f'Iteration {number}': pd.read_parquet(folder / 'pf.parquet'),
        'Market': pd.read_parquet(ROOT / 'outputs/base/overall analysis/market_references/us_large/pf.parquet'),
    }
    frames = {k: v.sort_values('eom_ret').reset_index(drop=True) for k, v in frames.items()}
    if number in (6,7,8,9,10):
        frames['Iteration 4']=pd.read_parquet(ROOT/'outputs/iteration_4/us_large/pf.parquet').sort_values('eom_ret').reset_index(drop=True)
    candidate, market = frames[f'Iteration {number}'], frames['Market']
    assert all(v.eom_ret.equals(candidate.eom_ret) and len(v) == 72 for v in frames.values())
    protocol = json.loads(PROTOCOL.read_text())
    probability = paired_probability(candidate, market, protocol['bootstrap_reps'],
                                     protocol['bootstrap_block_months'], protocol['bootstrap_seed'])['probability']
    stats = pd.read_csv(folder / 'performance_summary.csv').set_index('strategy')
    net = candidate.r.to_numpy() - candidate.tc.to_numpy()
    market_net = market.r.to_numpy() - market.tc.to_numpy()
    active = net - market_net
    annual_active = 12 * active.mean()
    te = active.std(ddof=1) * np.sqrt(12)
    ir = annual_active / te
    decision = json.loads((folder / 'decision.json').read_text())
    title = {2:'Gamma 5',4:'Market + 1% TE-target tilt','4.2':'Executed active-utility validation','4.3':'One-SE simpler selection',6:'More selective trading',7:'Stock-specific risk responsiveness',8:'Stable one-SE model selection',9:'Benchmark-aware trade netting',10:'Equal-weight RFF seed ensemble'}[number]
    if number==6:
        selection=json.loads((folder/'pretest_selection.json').read_text())
        title+=f' ({selection["selected_multiplier"]:g}x training penalty)'
    if number==7:
        selection=json.loads((folder/'pretest_selection.json').read_text())
        title+=f' ({selection["selected_half_life_days"]}-day half-life)'
    fig = plt.figure(figsize=(13.5, 7.2), dpi=180, facecolor='white')
    navy = '#1c304d'
    fig.text(.035, .95, '3 | RESULTS', color='#00868b', fontsize=11, weight='bold')
    fig.text(.035, .89, f'Iteration {number} vs the market, 2020 to 2025',
             color=navy, fontsize=23, weight='bold')
    inherited = ', '.join(map(str, decision['included_after']))
    fig.text(.035, .845, f'US high-cap | {title} | Included iterations: {inherited}', fontsize=11, color='#535e6b')
    axes = [fig.add_axes([.07 + j * .31, .445, .275, .32]) for j in range(3)]
    paths = {}
    ordered=[(k,v) for k,v in frames.items() if k!=f'Iteration {number}']+[(f'Iteration {number}',candidate)]
    for label, frame in ordered:
        values = frame.r.to_numpy() - frame.tc.to_numpy()
        path, drawdown = additive_path(values)
        paths[label] = path
        color = {'Base Portfolio ML': '#9cbed3', f'Iteration {number}': '#dd842b', 'Market': '#172333',
                 **({'Iteration 4':'#8862a5'} if number in (6,7,8,9,10) else {})}[label]
        for ax, series in zip(axes[:2], [100 * path, 100 * drawdown]):
            ax.plot(frame.eom_ret, series, color=color, lw=1.6 if label != 'Base Portfolio ML' else 1.1,
                    ls='--' if number in (6,7,8,9,10) and label=='Iteration 4' else '-',label=label)
    relative = 100 * (paths[f'Iteration {number}'] - paths['Market'])
    axes[2].plot(candidate.eom_ret, relative, color='#179d53', lw=1.6)
    axes[2].fill_between(candidate.eom_ret, relative, 0, color='#179d53', alpha=.14)
    titles = ['Cumulative net return (additive)', 'Additive drawdown from peak', 'Cumulative return minus Market']
    for ax, name in zip(axes, titles):
        ax.set_title(name, fontsize=10, color=navy)
        ax.set_ylabel('Percentage points', fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.15)
        ax.axhline(0, color='#777777', lw=.6)
    axes[0].legend(fontsize=7, frameon=False, loc='upper left')
    table_rows = [
        ['Net excess return / year', f"{stats.loc['Candidate', 'R_TC']:.2%}", f"{stats.loc['Market', 'R_TC']:.2%}", 'Active return / year', f'{annual_active:.3%}'],
        ['Net Sharpe', f"{stats.loc['Candidate', 'SR_net']:.3f}", f"{stats.loc['Market', 'SR_net']:.3f}", 'Information ratio (net-active SD)', f'{ir:.3f}'],
        ['Worst additive DD (pp)', f"{100*stats.loc['Candidate', 'max_drawdown_excess']:.2f}", f"{100*stats.loc['Market', 'max_drawdown_excess']:.2f}", 'P(better Sharpe vs market)', f'{probability:.1%}'],
    ]
    ax = fig.add_axes([.035, .205, .93, .175]); ax.axis('off')
    table = ax.table(cellText=table_rows, colLabels=['Measure', f'Iteration {number}', 'Market', 'Relative to market', 'Value'],
                     colWidths=[.25, .15, .15, .30, .15], cellLoc='left', colLoc='left', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(.6); cell.set_edgecolor('#68717d'); cell.PAD=.025
        cell.set_facecolor(navy if r == 0 else ('#eef2f7' if r % 2 else 'white'))
        cell.set_text_props(color='white' if r == 0 else '#253140', weight='bold' if r == 0 else 'normal')
    conclusion = ('Gamma 5 improves on Base, but substantially underperforms the market.' if number == 2 else
                  'The tilt closely tracks the market; evidence of a higher Sharpe versus market is below 80%.')
    if number in ('4.2','4.3'):
        conclusion=f'P(test) versus {decision["reference"]}: {decision["p_better"]:.2%}. Retained for research; no 80% gate.'
    if number in (6,7,8,9,10):
        probability_previous=decision['p_better']
        support='N/A (identical)' if probability_previous is None else f'{probability_previous:.1%}'
        conclusion=f'{decision["verdict"]}. P(better vs iteration 4): {support}.'
    fig.text(.035, .15, conclusion, fontsize=11, color=navy, weight='bold')
    fig.text(.035, .105, 'Net returns are annualised arithmetic excess returns after costs. Curves and drawdowns add monthly returns (percentage points).', fontsize=9, color='#555f6b')
    fig.text(.035, .075, 'P is paired block-bootstrap support, not a conventional p-value. The table compares with Market; the acceptance verdict compares with the prior accepted iteration.', fontsize=8.5, color='#555f6b')
    final_note = 'Test-period selection results; no independent unseen confirmation. Additive drawdown is not a percentage loss of capital.'
    fig.text(.035, .045, final_note, fontsize=9, color='#555f6b')
    output = folder / 'charts'; output.mkdir(exist_ok=True)
    fig.savefig(output / 'market_comparison_panel.png', dpi=180, facecolor='white')
    fig.savefig(output / 'market_comparison_panel.pdf', facecolor='white')
    plt.close(fig)
    pd.DataFrame(table_rows, columns=['Measure', f'Iteration {number}', 'Market', 'Relative to market', 'Value']).to_csv(output / 'market_comparison_panel.csv', index=False)
    print(output / 'market_comparison_panel.png')


def build_summary():
    """Refresh the compact screenshot from the current experiment CSV."""
    folder = ROOT / 'outputs/iteration_summary'
    data = pd.read_csv(folder / 'statistical_tests_all_iterations.csv')
    if data.iteration.astype(str).isin(['4.2','4.3']).any():
        from base_anchor_revision_reports import results
        results()
        return
    names = {1:'Refit yearly', 2:'Gamma 5', 3:'Exit 120, costs x2',
             4:'Market + 1% TE tilt', 5:'Recent weighting',6:'More selective trading',7:'Risk responsiveness',8:'Stable model selection',9:'Benchmark trade netting',10:'RFF seed ensemble'}
    rows = []
    for r in data.itertuples():
        verdict = 'Retained as design' if r.iteration == 1 else ('Kept' if r.included else 'Not included')
        rows.append([f'{r.iteration}  {names[r.iteration]}', f'{r.Sharpe:.3f}',
                     'N/A' if pd.isna(r.P_better) else f'{r.P_better:.1%}',
                     f'{100*r.worst_drawdown:.1f}', verdict])
    fig = plt.figure(figsize=(12,5.1), dpi=180, facecolor='white')
    count={6:'six',7:'seven',8:'eight',9:'nine',10:'ten'}.get(len(data),str(len(data)))
    fig.text(.025,.925,f'Statistical tests across all {count} iterations',fontsize=19,weight='bold',color='#242750')
    fig.text(.025,.865,'Kati_tryout | US high-cap | Test period: 2020-2025 | Additive returns',fontsize=11,color='#555b68')
    ax=fig.add_axes([.025,.27,.95,.52]); ax.axis('off')
    t=ax.table(cellText=rows,colLabels=['Iteration','Net Sharpe','P(better)','Additive DD (pp)','Verdict'],
               colWidths=[.30,.12,.13,.18,.27],cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    t.auto_set_font_size(False); t.set_fontsize(11)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#626777');cell.set_linewidth(.65);cell.PAD=.025
        if r==0:
            cell.set_facecolor('#242750');cell.set_text_props(color='white',weight='bold')
        else:
            cell.set_facecolor('#f0f3f7' if r%2 else 'white');cell.get_text().set_color('#252a35')
            if c==4:
                cell.set_text_props(weight='bold',color='#242750' if r==1 else ('#008b85' if rows[r-1][4]=='Kept' else '#bd351e'))
    kept = data.loc[data.included, 'iteration'].astype(str).tolist()
    fig.text(.025,.195,'Accepted sequence: '+', '.join(kept)+'. Annual refitting is already in Base.',fontsize=11,weight='bold',color='#242750')
    fig.text(.025,.137,'P(better) compares net Sharpe with the previous accepted version using a paired block bootstrap.',fontsize=10,color='#454b56')
    fig.text(.025,.091,'DD: drop from the additive return peak, in percentage points (pp). Drawdown is not an acceptance gate.',fontsize=10,color='#454b56')
    fig.text(.025,.045,'Test-period selection results; these are not independent unseen-data confirmation.',fontsize=9,color='#666b75')
    fig.savefig(folder/'statistical_tests_screenshot.png', dpi=180, facecolor='white');plt.close(fig)
    build_described_summary(data,names,folder)


def build_described_summary(data,names,folder):
    """Presentation table with plain-language descriptions and saved statistics."""
    from textwrap import fill
    descriptions={
        1:'Refit coefficients each year using expanding historical data. Already included in Base.',
        2:'Lower risk aversion from gamma 10 to 5, allowing the ML portfolio to take more risk.',
        3:'Retain stocks until size rank 120 and double assumed trading costs in learning and execution.',
        4:'Start from the value-weighted Market and learn active tilts with a pretest-calibrated 1% annual tracking-error target.',
        5:'Let annual validation choose recent-history half-lives of 3, 5 or 10 years, or equal historical weights.',
        6:'Penalise trading more during learning: pretest validation selected 2x. Actual trading-cost assumptions stay unchanged.',
        7:'Tune how quickly stock-specific risk estimates react: pretest validation selected a 252-day half-life.',
        8:'Prefer fewer RFF features and stronger ridge when historical utility is within one bootstrap standard error of the best.',
        9:'Learn the combined cost of benchmark and active trades using their recursive order responses. Actual execution stays unchanged.',
        10:'Average stock target tilts from five fixed RFF seeds, then execute one combined book using the original cost formula.',
    }
    rows=[]
    for r in data.itertuples():
        verdict='Retained as design' if r.iteration==1 else ('Kept' if r.included else 'Not included')
        rows.append([fill(f'{r.iteration}  {names[r.iteration]}',27),fill(descriptions[r.iteration],66),
                     f'{r.Sharpe:.3f}','N/A' if pd.isna(r.P_better) else f'{r.P_better:.2%}',
                     f'{100*r.worst_drawdown:.1f}',verdict])
    fig=plt.figure(figsize=(18,9),dpi=180,facecolor='white')
    navy='#242750'
    count={9:'nine',10:'ten'}.get(len(data),str(len(data)))
    fig.text(.025,.945,f'Statistical tests across all {count} iterations',fontsize=24,weight='bold',color=navy)
    fig.text(.025,.902,'Kati_tryout | US high-cap | Test period: 2020-2025 | Additive returns',fontsize=12,color='#555b68')
    ax=fig.add_axes([.025,.235,.95,.615]);ax.axis('off')
    table=ax.table(cellText=rows,
        colLabels=['Iteration','What changed','Net Sharpe','P(better)','Additive DD (pp)','Verdict'],
        colWidths=[.19,.385,.08,.08,.105,.16],cellLoc='left',colLoc='left',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#72798a');cell.set_linewidth(.6);cell.PAD=.025
        if row==0:
            cell.set_facecolor(navy);cell.set_text_props(color='white',weight='bold')
        else:
            cell.set_facecolor('#f0f3f7' if row%2 else 'white');cell.set_text_props(color='#252a35')
            if col==5:
                cell.set_text_props(weight='bold',color=navy if rows[row-1][5]=='Retained as design' else
                    ('#008b85' if rows[row-1][5]=='Kept' else '#bd351e'))
    kept=', '.join(data.loc[data.included,'iteration'].astype(str))
    fig.text(.025,.182,f'Accepted sequence: {kept}. Only trials passing the 80% rule are carried forward; annual refitting is already in Base.',fontsize=12,weight='bold',color=navy)
    fig.text(.025,.14,'Each trial inherits only accepted earlier changes; rejected changes are not carried forward.',fontsize=11,color='#454b56')
    fig.text(.025,.106,'P(better): paired block-bootstrap support for a higher net Sharpe than the previous accepted version; it is not a conventional p-value.',fontsize=11,color='#454b56')
    fig.text(.025,.072,'DD: drop from the additive return peak, in percentage points (pp). Drawdown is reported but is not an acceptance gate.',fontsize=11,color='#454b56')
    fig.text(.025,.038,'All decisions use the same inspected test period; these results are not independent unseen-data confirmation.',fontsize=10,color='#666b75')
    for suffix in ('png','pdf'):
        fig.savefig(folder/f'statistical_tests_with_descriptions.{suffix}',dpi=180,facecolor='white')
    plt.close(fig)
    data.assign(description=data.iteration.map(descriptions)).to_csv(folder/'statistical_tests_with_descriptions.csv',index=False)


if __name__ == '__main__':
    for iteration in (2, 4):
        build(iteration)
    build_summary()
