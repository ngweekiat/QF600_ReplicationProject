"""Index saved base/iteration outputs without fitting or completing experiments."""
from pathlib import Path
import hashlib
import html
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
GROUPS=('base','iteration_1','iteration_2','iteration_3')
MARKETS=('us_large','us_mid','eu_large')
DESCRIPTIONS={
    'base':'Original Static ML and default Portfolio ML only.',
    'iteration_1':'Benchmark-relative Portfolio ML, with 1%, 2% and 4% tracking-error targets.',
    'iteration_2':'Wider 250-stock universe, capacity and cost sensitivity; references iteration 1.',
    'iteration_3':'Time-decayed estimation with 3/5/10-year half-lives and no decay; references iteration 2.'}


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def active_label(path,root):
    calibration=root/'active/calibrated.csv'
    if calibration.exists():
        for row in pd.read_csv(calibration).itertuples():
            if f'gte{row.gamma_te:g}'==path.parent.name:return f'Portfolio-ML, TE {row.te_target:.0%}'
    return f'Portfolio-ML, {path.parent.name}'


def discover(group,market):
    root=ROOT/'outputs'/group/market
    files=[]
    if group=='base':
        files=[(root/'static_ml/pf.parquet','experiment',None),(root/'portfolio_ml/rff/pf.parquet','experiment',None)]
    else:
        files=[(p,'experiment',active_label(p,root)) for p in sorted((root/'active').glob('gte*/pf.parquet'))]
        for rel in ('benchmarks/pf_test.parquet','static_ml/pf.parquet','portfolio_ml/rff/pf.parquet','portfolio_ml/raw/pf.parquet'):
            path=root/rel
            if path.exists():files.append((path,'reference_or_preparatory',None))
    frames=[];inventory=[]
    for path,role,label in files:
        if not path.exists():continue
        data=pd.read_parquet(path)
        for name,d in data.groupby('type',sort=False):
            if group=='base' and name not in ('Static-ML','Portfolio-ML'):raise ValueError('Other method in base')
            d=d.copy();d['strategy']=label or name;d['role']=role;d['group']=group;d['market']=market
            frames.append(d)
            inventory.append(dict(group=group,market=market,strategy=label or name,role=role,
                file=str(path.relative_to(ROOT/'outputs')),sha256=sha256(path),months=len(d)))
    columns=['type','eom','eom_ret','r','tc','turnover','inv','shorting','strategy','role','group','market']
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)),inventory


def stats(data,market_root):
    rows=[];benchmark=None
    path=market_root/'benchmarks/pf_test.parquet'
    if path.exists():benchmark=pd.read_parquet(path).query("type == 'Market'").set_index('eom_ret')
    for (strategy,role),d in data.groupby(['strategy','role'],sort=False):
        r=d.r.to_numpy(float);tc=d.tc.to_numpy(float);sd=np.std(r,ddof=1)
        row=dict(strategy=strategy,role=role,months=len(d),first_return=str(pd.Timestamp(d.eom_ret.min()).date()),
            last_return=str(pd.Timestamp(d.eom_ret.max()).date()),annual_gross_return=12*r.mean(),annual_cost=12*tc.mean(),
            annual_net_return=12*(r-tc).mean(),annual_gross_volatility=np.sqrt(12)*sd,
            net_sharpe_paper=np.sqrt(12)*(r-tc).mean()/sd if sd>0 else np.nan,
            mean_leverage=d.inv.mean(),mean_monthly_turnover=d.turnover.mean())
        if benchmark is not None:
            b=benchmark.reindex(d.eom_ret)
            if b[['r','tc']].notna().all().all():
                active=r-b.r.to_numpy();active_net=(r-tc)-(b.r-b.tc).to_numpy();te=active.std(ddof=1)*np.sqrt(12)
                row.update(annual_active_net=12*active_net.mean(),tracking_error_realised=te,
                    information_ratio_net=12*active_net.mean()/te if te>0 else np.nan,
                    benchmark_cost_convention='net strategy minus net same-universe Market')
        rows.append(row)
    return pd.DataFrame(rows)


def plot(data,destination,title):
    if data.empty:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for label,d in data.groupby('strategy'):
        d=d.sort_values('eom_ret')
        axes[0].plot(d.eom_ret,100*d.r.cumsum(),label=label)
        axes[1].plot(d.eom_ret,100*d.tc.cumsum(),label=label)
    axes[0].set(title='Cumulative gross excess return (arithmetic)',ylabel='Percentage points')
    axes[1].set(title='Cumulative modelled trading cost',ylabel='Percentage points')
    if data.tc.max()>1:axes[1].set_yscale('symlog',linthresh=1);axes[1].set_title('Cumulative cost (symlog)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle(title);fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(destination.with_suffix('.'+suffix),dpi=170)
    plt.close(fig)


def main():
    availability=[];inventory=[]
    style='<style>body{max-width:1300px;margin:36px auto;padding:0 20px;font:15px/1.5 system-ui;color:#21323e}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ced8df;padding:8px}th{background:#e8eff4}.scroll{overflow-x:auto}img{width:100%}</style>'
    for group in GROUPS:
        group_root=ROOT/'outputs'/group;group_root.mkdir(exist_ok=True)
        for market in MARKETS:
            root=group_root/market;reports=root/'reports';reports.mkdir(parents=True,exist_ok=True)
            data,files=discover(group,market);inventory.extend(files)
            experiment=data.loc[data.role.eq('experiment')];reference=data.loc[data.role.ne('experiment')]
            if group=='base':status='available' if experiment.strategy.nunique()==2 else 'partial'
            elif not experiment.empty:status='saved experiment results available'
            elif not reference.empty:status='preparatory/reference outputs only; experiment unfinished'
            elif any(p.is_file() for p in root.rglob('*') if 'reports' not in p.relative_to(root).parts and p.name not in ('status.json','README.md','index.html')):status='input audits/references only; experiment unfinished'
            else:status='not run; no saved experiment performance'
            record=dict(group=group,market=market,status=status,experiment_series=int(experiment.strategy.nunique()),
                reference_series=int(reference.strategy.nunique()),new_experiments_run=False)
            availability.append(record)
            (root/'status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            table=stats(data,root)
            if not data.empty:
                data.to_csv(reports/'available_monthly_returns.csv',index=False)
                table.to_csv(reports/'available_performance.csv',index=False)
            plot(experiment,reports/'charts/experiment_returns_and_costs',f'{group} | {market} | saved experiment results')
            plot(reference,reports/'charts/reference_returns_and_costs',f'{group} | {market} | reference/preparatory results')
            intro=f'<h1>{group}: {market}</h1><p>{html.escape(DESCRIPTIONS[group])}</p><p><strong>Status: {html.escape(status)}</strong></p>'
            text='Existing Keats outputs restored and organised only. No missing experiment was run. Keats test returns are 2020-2025; no separate unseen 2021-2025 holdout is claimed. Reference/preparatory statistics are not a completed iteration result.'
            body=intro+'<p>'+text+'</p>'
            if not table.empty:body+='<h2>Available saved performance</h2><p>Returns and costs are annual decimal fractions; leverage and turnover are ratios.</p><div class="scroll">'+table.to_html(index=False,float_format=lambda x:f'{x:.5g}')+'</div>'
            if (root/'active/summary.csv').exists():body+='<p><a href="active/summary.csv">Original iteration summary (unchanged)</a></p>'
            for role in ('experiment','reference'):
                image=f'reports/charts/{role}_returns_and_costs.png'
                if (root/image).exists():body+=f'<img src="{image}" alt="{role} results">'
            (root/'index.html').write_text('<!doctype html><meta charset="utf-8">'+style+body,encoding='utf-8')
            (root/'README.md').write_text(f'# {group}: {market}\n\n{DESCRIPTIONS[group]}\n\n**Status: {status}.**\n\n{text}\n\nOpen index.html. Original CSV, Parquet and chart files remain in their restored subfolders. Newly generated descriptive summaries/charts are under reports/.\n',encoding='utf-8')
        rows=pd.DataFrame(availability).query('group == @group')
        rows.to_csv(group_root/'availability.csv',index=False)
        links=''.join(f'<li><a href="{m}/index.html">{m}</a></li>' for m in MARKETS)
        # Base keeps its richer validated two-method comparison as index.html.
        if group!='base':
            (group_root/'index.html').write_text('<!doctype html><meta charset="utf-8">'+style+f'<h1>{group}</h1><p>{DESCRIPTIONS[group]}</p>'+rows.to_html(index=False)+f'<ul>{links}</ul><p>Only existing saved Keats results are shown; unfinished experiments remain unfinished.</p>',encoding='utf-8')
    out=ROOT/'outputs'
    pd.DataFrame(availability).to_csv(out/'availability_by_group_and_market.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'organisation_audit/result_source_inventory.csv',index=False)
    links=''.join(f'<li><a href="{g}/index.html">{g}</a>: {DESCRIPTIONS[g]}</li>' for g in GROUPS)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8">'+style+'<h1>Keats results by run group</h1><ul>'+links+'</ul>'+pd.DataFrame(availability).to_html(index=False)+'<p>Base contains only Static ML and Portfolio ML. Iteration results retain their original strategies and completion status. No missing experiments were run.</p>',encoding='utf-8')
    print(pd.DataFrame(availability).to_string(index=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
