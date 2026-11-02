"""US high-cap-only association selection; no forecast or portfolio model fit.

Candidates must appear in both published JKP Cluster Labels and Factor Details
and in the WRDS source schema. The measurement window is US high-cap 2005–2014.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/characteristics/us_high_cap_inputs'
OUTPUT=ROOT/'outputs'
from selection_rules import assign_themes, diverse_pick
from association_statistics import panel_metrics, summarize, pairwise_matrices, paired_columns

REASONS = {"published JKP representative for economic theme": "A",
           "minimum economic-theme representation": "T",
           "balanced predictive score with redundancy penalty": "S"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def verified_jkp_registry(registry, data=DATA):
    """Whitelist published JKP names; never infer membership from column position."""
    reference = Path(data)/'reference'
    if not reference.exists(): reference = Path(data).parents[1]/'reference'
    cluster_path=reference/'cluster_labels.parquet'
    details_path=reference/'factor_details.parquet'
    clusters=pd.read_parquet(cluster_path).rename(columns={'cluster':'published_theme'})
    details=pd.read_parquet(details_path).dropna(subset=['abr_jkp'])
    if clusters.characteristic.duplicated().any() or details.abr_jkp.duplicated().any():
        raise ValueError('Published JKP metadata has duplicate identifiers')
    allowed=set(clusters.characteristic)&set(details.abr_jkp)
    verified=registry.loc[registry.characteristic.isin(allowed)].copy()
    # Independently replace cached labels/signs with published reference values.
    verified=verified.drop(columns=['published_theme','direction','name_new'],errors='ignore')
    verified=verified.merge(clusters,on='characteristic',validate='one_to_one')
    verified=verified.merge(details[['abr_jkp','direction','name_new']],left_on='characteristic',right_on='abr_jkp',validate='one_to_one').drop(columns='abr_jkp')
    verified['published_jkp_153']=True
    verified['jkp_membership_verified']=True
    if not verified.direction.isin([-1,1]).all():raise ValueError('Missing published JKP direction')
    return verified.reindex(sorted(verified.columns),axis=1),[cluster_path,details_path]


def load_sample(mode='training', data=DATA):
    """Filter membership and return dates before computing any statistics."""
    data = Path(data)
    local = data/'us_high_cap_sample.parquet'
    if local.exists():
        if mode != 'training':raise ValueError('The self-contained sample contains 2005–2014 observations only')
        frame = pd.read_parquet(local)
        registry = pd.read_csv(data/'candidate_registry.csv')
        names = registry.characteristic.tolist()
        if not frame.member_us_large.all() or not frame.return_month.between('2005-01-31','2014-12-31').all():
            raise ValueError('Standalone sample has out-of-scope rows')
        return frame,names,registry,[local,data/'candidate_registry.csv']
    meta_path = data/'metadata_US.parquet'
    meta = pd.read_parquet(meta_path)
    end = '2014-12-31' if mode == 'training' else '2019-12-31'
    meta = meta.loc[meta.member_us_large & meta.return_month.between('2005-01-31', end)].copy()
    if meta.duplicated(['id','eom']).any(): raise ValueError('Duplicate US high-cap membership keys')
    registry = pd.read_csv(data/'candidate_registry.csv')
    if mode == 'training':
        names = registry.characteristic.tolist()
        paths = sorted((data/'pilot').glob('US_*.parquet'))
    else:
        names = json.loads((data/'download_200.json').read_text())
        paths = [p for p in sorted((data/'panels').glob('US_*.parquet')) if int(p.stem.split('_')[1]) <= 2019]
    parts = []
    sources = [meta_path, data/'candidate_registry.csv']
    for path in paths:
        frame = pd.read_parquet(path)
        frame = frame.merge(meta[['id','eom']], on=['id','eom'], how='inner', validate='one_to_one')
        parts.append(frame)
        sources.append(path)
    raw = pd.concat(parts, ignore_index=True)
    price = data/'pilot/prc_US.parquet'
    if 'prc' in names and price.exists():
        supplement = pd.read_parquet(price).rename(columns={'prc':'price_supplement'})
        raw = raw.merge(supplement, on=['id','eom'], how='left', validate='one_to_one')
        raw['prc'] = raw.get('prc', pd.Series(np.nan, index=raw.index)).combine_first(raw.price_supplement)
        raw = raw.drop(columns='price_supplement')
        sources.append(price)
    if raw.duplicated(['id','eom']).any() or not set(names).issubset(raw):
        raise ValueError('Incomplete or duplicate US characteristic observations')
    frame = meta.merge(raw[['id','eom',*names]], on=['id','eom'], how='left', validate='one_to_one', indicator=True)
    if not frame._merge.eq('both').all(): raise ValueError('US high-cap keys absent from raw source')
    frame = frame.drop(columns='_merge').sort_values(['eom','id']).reset_index(drop=True)
    frame[names] = frame[names].apply(pd.to_numeric, errors='raise').astype(float).replace([np.inf,-np.inf],np.nan)
    # Missing observed targets stay missing; do not load model-imputed returns.
    assert frame.member_us_large.all() and frame.return_month.max() <= pd.Timestamp(end)
    return frame, names, registry.loc[registry.characteristic.isin(names)].copy(), sources


def association_scores(monthly, mode='training'):
    if set(monthly.universe) != {'us_large'}:raise ValueError('Selection statistics must contain only US high-cap stocks')
    if not monthly.return_month.between('2005-01-31','2019-12-31').all():raise ValueError('Out-of-scope return dates')
    train = summarize(monthly.loc[monthly.return_month.dt.year.le(2014)]).set_index('characteristic')
    fit = train.copy()
    direction = np.where(train.mean_rank_ic.ge(0), 1, -1)
    fit['training_direction'] = direction
    fit['training_mean_rank_ic'] = train.mean_rank_ic
    fit['minimum_training_coverage'] = train.coverage
    fit['score_period_mean_rank_ic'] = train.mean_rank_ic
    fit['score_coverage'] = train.coverage
    fit['minimum_validation_coverage'] = np.nan
    fit['validation_oriented_ic'] = np.nan
    scoring = monthly.loc[monthly.return_month.dt.year.le(2014)].copy()
    if mode == 'validation':
        val = summarize(monthly.loc[monthly.return_month.dt.year.ge(2015)]).set_index('characteristic').reindex(train.index)
        fit['score_period_mean_rank_ic'] = val.mean_rank_ic
        fit['score_coverage'] = val.coverage
        fit['minimum_validation_coverage'] = val.coverage
        fit['validation_oriented_ic'] = val.mean_rank_ic*direction
        fit['eligible'] = (train.coverage.ge(.6)&val.coverage.ge(.6)&train.valid_ic_months.ge(60)&val.valid_ic_months.ge(30))
        scoring = monthly.loc[monthly.return_month.dt.year.ge(2015)].copy()
    else:
        fit['eligible'] = train.coverage.ge(.6)&train.valid_ic_months.ge(60)
    fit['oriented_score_ic'] = fit.score_period_mean_rank_ic*direction
    yearly = scoring.assign(year=scoring.return_month.dt.year).groupby(['characteristic','year']).rank_ic.mean()
    fit['directional_year_stability'] = [float((yearly.loc[c].dropna()*d>0).mean()) for c,d in zip(fit.index,direction)]
    fit['mean_persistence'] = scoring.groupby('characteristic').rank_persistence.mean().reindex(fit.index)
    # All components are observed associations; no regression, ML or portfolio fit.
    fit['final_score'] = (.55*fit.oriented_score_ic.rank(pct=True)+.20*fit.score_coverage.rank(pct=True)
        +.15*fit.directional_year_stability.rank(pct=True)+.10*fit.mean_persistence.fillna(0).rank(pct=True))
    return fit.reset_index()


def export_matrices(frame, names, folder):
    folder.mkdir(parents=True, exist_ok=True)
    raw = frame[names].to_numpy(float)
    ranked = frame.groupby('eom')[names].rank(method='average',pct=True)
    for scale, values in [('raw',raw),('rank',ranked.to_numpy(float))]:
        corr,cov,n = pairwise_matrices(values, min_pairs=120)
        for label, array in [('correlation',corr),('covariance',cov),('pair_counts',n)]:
            pd.DataFrame(array,index=names,columns=names).to_csv(folder/f'{scale}_{label}.csv')
    # A labelled return column makes feature-to-return associations explicit.
    target = frame.observed_ret_ld1.to_numpy(float)
    target_rank = frame.groupby('eom').observed_ret_ld1.rank(method='average',pct=True).to_numpy(float)
    for scale, values, y in [('raw',raw,target),('rank',ranked.to_numpy(float),target_rank)]:
        corr,cov,n = pairwise_matrices(np.column_stack([values,y]),min_pairs=120)
        labels = names+['NEXT_MONTH_EXCESS_RETURN']
        for label,array in [('correlation',corr),('covariance',cov),('pair_counts',n)]:
            pd.DataFrame(array,index=labels,columns=labels).to_csv(folder/f'{scale}_with_return_{label}.csv')


def stock_associations(frame, names):
    results = []
    for stock, g in frame.groupby('id',sort=True):
        x = g[names]
        y = g.observed_ret_ld1.to_numpy(float)
        n,pearson,cov = paired_columns(x.to_numpy(float),y,min_pairs=24)
        valid = np.isfinite(x.to_numpy(float))&np.isfinite(y)[:,None]
        xr = x.where(valid).rank(method='average',pct=True)
        yr = pd.DataFrame(np.broadcast_to(y[:,None],x.shape),index=x.index,columns=x.columns).where(valid).rank(method='average',pct=True)
        _,spearman,_ = paired_columns(xr.to_numpy(),yr.to_numpy(),min_pairs=24)
        results.append(pd.DataFrame(dict(id=stock,characteristic=names,n_observed_pairs=n,
            temporal_pearson=pearson,temporal_spearman=spearman,raw_return_covariance=cov)))
    return pd.concat(results,ignore_index=True)


def charts(selected, scores, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    folder = out/'charts';folder.mkdir(exist_ok=True)
    # Actual counts and scores from this run, rather than an illustrative funnel.
    eligible = int(scores.eligible.sum())
    fig = plt.figure(figsize=(16, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 2.5])
    ax = fig.add_subplot(grid[0, :]); ax.axis('off')
    steps = [(f'{len(scores)} verified JKP candidates', 'Published metadata + WRDS\nUS high-cap; 2005–2014'),
             (f'{eligible} eligible candidates', 'Coverage ≥60%\n≥60 valid IC months'),
             (f'{int(scores.selected_200.sum())} shortlisted', 'Group coverage + score\nRedundancy penalty 0.10'),
             ('50 selected', '13 groups; 2–6 per group\nRedundancy penalty 0.20')]
    for i, (title, detail) in enumerate(steps):
        x = .125 + i*.25
        ax.text(x, .60, title+'\n\n'+detail, ha='center', va='center',
                fontsize=9, bbox=dict(boxstyle='round,pad=.8', facecolor='#edf4f7', edgecolor='#247b9b'),
                transform=ax.transAxes, wrap=True)
        if i<3: ax.annotate('', xy=(x+.16,.60), xytext=(x+.10,.60),
                            xycoords='axes fraction', arrowprops=dict(arrowstyle='->',color='#247b9b'))
    ax.set_title('How the 50 characteristics were chosen', loc='left', fontsize=21, weight='bold', pad=25)
    ax = fig.add_subplot(grid[1, 0])
    ax.barh(['Return association (absolute mean monthly rank IC)', 'Observed data coverage',
             'Yearly association-sign stability', 'Rank persistence'], [55,20,15,10], color='#247b9b')
    ax.invert_yaxis(); ax.set_xlim(0,65); ax.set_xlabel('Fixed score weight (%)')
    for i,v in enumerate([55,20,15,10]): ax.text(v+1,i,str(v)+'%',va='center')
    ax = fig.add_subplot(grid[1, 1]); ax.axis('off')
    ax.text(0,1,'Selection rules',weight='bold',fontsize=14,va='top')
    ax.text(0,.80,'Score uses percentile ranks of the four measures.\n'
            'Greedy priority = score + 0.02 for published label\n'
            '− penalty × largest absolute correlation with chosen signals.\n\n'
            'A: 13 published anchors (one per group)\n'
            'T: 13 additions to reach two members per group\n'
            'S: 24 further choices by score and redundancy',va='top',fontsize=11,linespacing=1.6)
    # Show all 50, their measured score, group and actual inclusion reason.
    palette={'A':'#247b9b','T':'#8c79a2','S':'#5c956d'}
    for column in range(2):
        ax=fig.add_subplot(grid[2,column])
        rows=selected.iloc[column*25:(column+1)*25]
        ax.barh(range(len(rows)),rows.final_score,color=[palette[c] for c in rows.reason_code])
        ax.set_yticks(range(len(rows)),[f'{r.characteristic} [{r.reason_code}] · {r.theme}' for r in rows.itertuples()],fontsize=8)
        ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Association score before redundancy adjustment')
        for i,v in enumerate(rows.final_score):ax.text(v+.01,i,f'{v:.3f}',va='center',fontsize=7)
    fig.text(.02,.012,'No ML fit. A/T/S explains inclusion, not model weights. Group quotas and correlation penalties mean the final 50 are not simply the 50 highest individual correlations.',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.98),h_pad=2.5,w_pad=3)
    for ext in ('png','svg','pdf'):
        name='selection_decision_path_jkp_only.pdf' if ext=='pdf' else f'selection_decision_path.{ext}'
        fig.savefig(folder/name,dpi=180)
    plt.close(fig)
    ordered = selected.sort_values(['theme','final_score'],ascending=[True,False])
    names = ordered.characteristic.tolist()
    for kind, title in [('correlation','Correlation'),('covariance','Covariance')]:
        a = pd.read_csv(out/f'matrices/selected50/rank_{kind}.csv',index_col=0).loc[names,names]
        fig,ax = plt.subplots(figsize=(14,12))
        limit = 1 if kind=='correlation' else float(np.nanmax(abs(a.to_numpy())))
        plot = ax.imshow(a,cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_xticks(range(50), names,rotation=90,fontsize=7)
        ax.set_yticks(range(50), names,fontsize=7)
        ax.set_title(f'US high-cap: selected 50 — ranked-feature {kind}\nSelection sample only; pairwise observed values, no ML model',loc='left',pad=20,weight='bold')
        fig.colorbar(plot,ax=ax,shrink=.75,label=f'{title} of monthly within-US-high-cap percentile ranks')
        fig.tight_layout();fig.savefig(folder/f'selected50_rank_{kind}.png',dpi=180);fig.savefig(folder/f'selected50_rank_{kind}.svg');plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1.1,1]})
    groups = selected.groupby('theme').agg(total=('characteristic','size'),published=('published_jkp_153','sum')).sort_index()
    y = np.arange(len(groups))
    axes[0].barh(y,groups.published,color='#247b9b',label='Published labels')
    if (groups.total-groups.published).sum():
        axes[0].barh(y,groups.total-groups.published,left=groups.published,color='#df9d36',label='Provisional US 2005–2014 correlation labels')
    axes[0].set_yticks(y,groups.index);axes[0].invert_yaxis();axes[0].set_xlim(0,7)
    for i,n in enumerate(groups.total):axes[0].text(n+.08,i,str(n),va='center',weight='bold')
    axes[0].set_title('13 groups; 2–6 inputs per group',loc='left',weight='bold');axes[0].set_xlabel('Selected characteristics');axes[0].legend(loc='lower left',bbox_to_anchor=(0,-.2),fontsize=8)
    reasons = selected.reason_code.value_counts().reindex(['A','T','S'],fill_value=0)
    axes[1].barh(['A: published group anchor','T: minimum group quota','S: association score after redundancy'],reasons.values,color=['#247b9b','#8c79a2','#5c956d'])
    axes[1].invert_yaxis()
    for i,n in enumerate(reasons):axes[1].text(n+.3,i,str(n),va='center',weight='bold')
    axes[1].set_xlim(0,max(reasons)+5);axes[1].set_xlabel('Selected characteristics');axes[1].set_title('Why each characteristic was included',loc='left',weight='bold')
    fig.suptitle('Why these 50? US high-cap selection only',fontsize=19,weight='bold',x=.02,ha='left')
    fig.text(.02,.015,'Fixed score: 55% oriented monthly rank IC + 20% coverage + 15% yearly direction stability + 10% persistence.\nAll selected names, groups and directions are verified against published JKP metadata. A/T/S are inclusion reasons, not ML coefficients.',fontsize=10)
    fig.tight_layout(rect=(0,.1,1,.94));fig.savefig(folder/'selection_groups_and_reasons.png',dpi=180);fig.savefig(folder/'selection_groups_and_reasons.svg');plt.close(fig)
    fig,ax = plt.subplots(figsize=(14,15))
    colors = plt.get_cmap('tab20')(np.linspace(0,1,13));palette = dict(zip(sorted(ordered.theme.unique()),colors))
    ax.barh(range(50),ordered.score_period_mean_rank_ic.to_numpy()*100,color=[palette[t] for t in ordered.theme])
    labels = [f'{r.characteristic}  [{r.reason_code}]' for r in ordered.itertuples()]
    ax.set_yticks(range(50),labels,fontsize=9);ax.invert_yaxis();ax.axvline(0,color='#243644',lw=.8)
    ax.set_xlabel('Mean monthly Spearman correlation × 100 (IC +2 means correlation +0.02)')
    ax.set_title('US high-cap return association for each selected characteristic\nSigned association with next-month observed excess return; colours identify groups',loc='left',weight='bold',pad=20)
    ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(folder/'selected50_return_association.png',dpi=180);fig.savefig(folder/'selected50_return_association.svg');plt.close(fig)


def report(selected, scores, monthly, frame, manifest, out):
    title = 'US high-cap-only characteristic selection'
    columns = ['characteristic','description','theme','published_jkp_153','reason_code','training_mean_rank_ic','score_period_mean_rank_ic','score_coverage','directional_year_stability','mean_persistence','final_score']
    payload = json.dumps(selected[columns].replace({np.nan:None}).to_dict('records'),allow_nan=False).replace('<','\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>US high-cap characteristic selection</title>
<style>body{max-width:1250px;margin:35px auto;padding:0 22px;font:15px/1.55 system-ui;color:#19384b}h1{font-size:32px}h2{margin-top:32px}.note{padding:16px;background:#eef4f7;border-radius:9px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:8px;border-bottom:1px solid #dbe4eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef4f7}img{max-width:100%}select,input{font:inherit;padding:7px;margin:6px}a{color:#126083}summary{cursor:pointer;font-weight:bold}</style>
<h1>US high-cap-only: 50 characteristics</h1><p>Selection uses US high-cap membership only. US mid-cap and European stocks do not contribute to scores, groups, matrices or selection.</p>
<div class="note">SAMPLE<br>Only names appearing in both published JKP Cluster Labels and Factor Details are eligible. All selected inputs use published group labels and directions. These 50 follow the declared association/diversity rule; no regression, forecast or ML portfolio was fitted.</div>
<h2>How were these 50 selected?</h2><img src="charts/selection_decision_path.png" alt="Verified JKP candidates, eligibility, final 50 scores and inclusion reasons"><p><a href="charts/selection_decision_path_jkp_only.pdf">Download JKP-only selection chart as PDF</a></p>
<h2>Why these 50 and their groups?</h2><img src="charts/selection_groups_and_reasons.png" alt="Group coverage and selection reasons">
<p>A = published group anchor; T = minimum theme representation; S = score after redundancy penalty. Every group assignment is taken from published JKP metadata.</p>
<label>Group<select id="theme"><option value="">All groups</option></select></label><label>Reason<select id="reason"><option value="">All reasons</option><option>A</option><option>T</option><option>S</option></select></label><label>Find<input id="query" placeholder="Characteristic or description"></label><p id="count"></p><div class="scroll"><table id="features"></table></div>
<h2>Correlation and covariance matrices</h2><p>The chart below switches between ranked-feature correlation and covariance. Raw-value and return-augmented matrices are downloadable too. Raw covariance magnitudes depend on units and should not be interpreted as comparable predictive importance.</p><select id="matrix"><option value="correlation">Rank correlation</option><option value="covariance">Rank covariance</option></select><p><a id="matrixcsv" href="matrices/selected50/rank_correlation.csv">Download shown 50×50 matrix</a></p><img id="matriximage" src="charts/selected50_rank_correlation.png" alt="Selected feature matrix">
<details><summary>Each characteristic’s association with next-month return</summary><img src="charts/selected50_return_association.png" alt="Return correlations"></details>
<h2>Data and audit files</h2>LINKS
<h2>Methods, assumptions and exclusions</h2><p>METHODS</p>
<script>const DATA=PAYLOAD;const $=x=>document.getElementById(x);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const num=x=>Number(x).toFixed(4);$('theme').innerHTML+=[...new Set(DATA.map(x=>x.theme))].sort().map(x=>'<option>'+esc(x)+'</option>').join('');function draw(){let rows=DATA.filter(x=>(!$('theme').value||x.theme===$('theme').value)&&(!$('reason').value||x.reason_code===$('reason').value)&&(`${x.characteristic} ${x.description}`.toLowerCase().includes($('query').value.toLowerCase())));$('count').textContent=rows.length+' of 50 characteristics';$('features').innerHTML='<tr><th>Characteristic</th><th>Description</th><th>Group</th><th>Group label</th><th>Reason</th><th>Mean monthly rank IC</th><th>Coverage</th><th>Score</th></tr>'+rows.map(x=>`<tr><td>${esc(x.characteristic)}</td><td>${esc(x.description)}</td><td>${esc(x.theme)}</td><td>${x.published_jkp_153?'Published':'Provisional proxy'}</td><td>${x.reason_code}</td><td>${num(x.score_period_mean_rank_ic)}</td><td>${(100*x.score_coverage).toFixed(1)}%</td><td>${num(x.final_score)}</td></tr>`).join('');}['theme','reason','query'].forEach(id=>$(id).oninput=draw);$('matrix').onchange=()=>{const kind=$('matrix').value;$('matriximage').src='charts/selected50_rank_'+kind+'.png';$('matrixcsv').href='matrices/selected50/rank_'+kind+'.csv';};draw();</script></html>'''
    methods = ('Monthly cross-sectional Pearson/Spearman correlations compare contemporaneous characteristics with next-month observed excess returns. '
        'Selection score = 55% oriented rank-IC percentile + 20% coverage percentile + 15% yearly sign-stability percentile + 10% persistence percentile. '
        'Candidate membership requires a WRDS source column and exact matches in published JKP Cluster Labels and Factor Details. Published groups and directions are used. Empirical association orientation is measured on 2005–2014 US high-cap data. Eligibility requires at least 60% observed feature coverage and 60 valid monthly ICs. '
        'The greedy rule adds 0.02 for published membership and subtracts 0.20 times maximum absolute training rank-correlation with already selected inputs. Each group has 2–6 members and a published anchor. '
        'Matrices use pairwise-complete observations, at least 120 pairs, and sample covariance (n−1); pair counts are supplied. Raw NaNs and unavailable returns stay missing. '
        'Rank matrices pool within-month percentile ranks; their return column is a pooled diagnostic and differs from the equal-month average IC used for selection. '
        'Per-stock correlations are time-series associations, requiring 24 observed pairs, and are reported separately. No cumulative return target is used. '
        'Pairwise covariance matrices may not be positive semidefinite and are descriptive diagnostics, not fitted portfolio risk matrices. '
        'The dataset retains its supplied source vintage and original membership assumptions; these checks do not establish point-in-time or survivorship-free data. '
        'No outcomes from 2015 onward, other-market return outcomes, regression R², fitted forecasts or portfolio utility enter the new selection. Raw covariance is a diagnostic; return rank correlation and correlation redundancy drive selection.')
    links = ['recommended_50_jkp_only.csv','recommended_50.json','all_candidate_scores.csv','excluded_nonpublished_candidates.csv','candidate_200.csv','group_summary.csv',
             'feature_return_summary.csv','per_stock_feature_return_associations.parquet','per_stock_selected50_pearson.csv',
             'per_stock_selected50_spearman.csv','per_stock_selected50_covariance.csv','selection_manifest.json',
             'verification.json','METHODS_ASSUMPTIONS_AND_EXCLUSIONS.md']
    links += [str(p.relative_to(out)).replace('\\','/') for p in sorted((out/'matrices/selected50').glob('*.csv'))]
    page = page.replace('SAMPLE',html.escape(manifest['sample_description'])).replace('PAYLOAD',payload).replace('METHODS',html.escape(methods))
    page = page.replace('LINKS','<ul>'+''.join(f'<li><a href="{html.escape(p)}">{html.escape(p)}</a></li>' for p in links)+'</ul>')
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'METHODS_ASSUMPTIONS_AND_EXCLUSIONS.md').write_text('# US high-cap association selection\n\n'+manifest['sample_description']+'\n\n'+methods+'\n',encoding='utf-8')


def select_us_high_cap(mode='training', out=None):
    if mode!='training':raise ValueError('Current selection is restricted to the 2005–2014 measurement window; no later validation outcomes')
    out = Path(out) if out else OUTPUT/'us_high_cap_selection'
    out.mkdir(parents=True,exist_ok=True)
    frame,names,registry,sources = load_sample(mode)
    registry,reference_sources=verified_jkp_registry(registry)
    excluded=sorted(set(names)-set(registry.characteristic))
    exclusion_path=out/'excluded_nonpublished_candidates.csv'
    if not excluded and exclusion_path.exists():
        excluded=sorted(set(pd.read_csv(exclusion_path).characteristic)-set(registry.characteristic))
    names=registry.characteristic.tolist()
    sources+=reference_sources
    frame=frame.drop(columns=excluded,errors='ignore')
    pd.DataFrame({'characteristic':excluded,'decision':'Excluded: not in both published JKP references'}).to_csv(out/'excluded_nonpublished_candidates.csv',index=False)
    if mode == 'training':
        snapshot = DATA/'us_high_cap_selection' if not (DATA/'us_high_cap_sample.parquet').exists() else DATA
        snapshot.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(snapshot/'us_high_cap_sample.parquet',index=False)
        registry.to_csv(snapshot/'candidate_registry.csv',index=False)
    print(f'US high-cap only: {len(frame):,} rows; {len(names)} candidates; mode={mode}',flush=True)
    monthly = panel_metrics(frame,names,'us_large')
    monthly.to_parquet(out/'monthly_feature_return_metrics.parquet',index=False)
    scores = association_scores(monthly,mode)
    training = frame.loc[frame.return_month.dt.year.le(2014)]
    ranks = training.groupby('eom')[names].rank(pct=True)
    correlation,_,_ = pairwise_matrices(ranks.to_numpy(float),min_pairs=120)
    corr = pd.DataFrame(correlation,index=names,columns=names)
    registry = assign_themes(registry,corr)
    scores = scores.merge(registry,on='characteristic',validate='one_to_one')
    cap = diverse_pick(scores,corr,min(200,int(scores.eligible.sum())),min_theme=2,score_col='final_score',redundancy_penalty=.10,published_anchor=True)
    subset = scores.loc[scores.characteristic.isin(cap.characteristic)]
    chosen = diverse_pick(subset,corr,50,min_theme=2,max_theme=6,score_col='final_score',redundancy_penalty=.20,published_anchor=True)
    chosen['reason_code'] = chosen.selection_reason.map(REASONS)
    chosen.to_csv(out/'recommended_50_jkp_only.csv',index=False)
    try:chosen.to_csv(out/'recommended_50.csv',index=False)
    except PermissionError:pass  # A user-open legacy CSV is never used as an active input.
    (out/'recommended_50.json').write_text(json.dumps(chosen.characteristic.tolist(),indent=2)+'\n',encoding='utf-8')
    scores['selected_200'] = scores.characteristic.isin(cap.characteristic)
    scores['selected_50'] = scores.characteristic.isin(chosen.characteristic)
    scores['decision'] = np.select([~scores.eligible,scores.selected_50,~scores.selected_200],['Fails coverage/valid-month gate','Selected 50','Outside US-only 200 cap'],default='Inside 200; not chosen under group/score/redundancy rule')
    scores.sort_values('final_score',ascending=False).to_csv(out/'all_candidate_scores.csv',index=False)
    cap.to_csv(out/'candidate_200.csv',index=False)
    group = chosen.groupby('theme').agg(features=('characteristic','size'),published_representatives=('published_jkp_153','sum'))
    group.to_csv(out/'group_summary.csv')
    summarize(monthly).to_csv(out/'feature_return_summary.csv',index=False)
    export_matrices(training,names,out/'matrices/all_candidates_training')
    export_matrices(training,chosen.characteristic.tolist(),out/'matrices/selected50')
    per_stock = stock_associations(frame,chosen.characteristic.tolist())
    per_stock.to_parquet(out/'per_stock_feature_return_associations.parquet',index=False)
    for metric,label in [('temporal_pearson','pearson'),('temporal_spearman','spearman'),('raw_return_covariance','covariance')]:
        per_stock.pivot(index='id',columns='characteristic',values=metric).to_csv(out/f'per_stock_selected50_{label}.csv')
    manifests = dict(scope='us_large only',mode=mode,candidates=len(names),eligible=int(scores.eligible.sum()),cap=len(cap),selected=50,
        sample_description=f'{len(frame):,} US high-cap held-stock months; {frame.id.nunique()} distinct stocks; measurement return months 2005–2014. No ML training.',
        selection_return_start=str(frame.return_month.min().date()),selection_return_end=str(frame.return_month.max().date()),
        training_only=mode=='training',other_markets_used=False,test_outcomes_used=False,models_fitted=False,regression_r2_used=False,
        selection_weights=dict(oriented_rank_ic=.55,coverage=.20,yearly_direction_stability=.15,rank_persistence=.10),
        reason_counts=chosen.reason_code.value_counts().to_dict(),published_group_count=int(chosen.published_jkp_153.sum()),
        proxy_group_count=int((~chosen.published_jkp_153).sum()),
        candidate_policy='Exact WRDS source names intersect published JKP Cluster Labels and Factor Details',
        jkp_only=True,published_groups_only=True,excluded_unverified_count=len(excluded),
        source_files=[dict(path=str(p.resolve()),sha256=sha(p)) for p in sources],
        caveat='Best under the declared association/diversity rule; no global optimality or joint-ML performance claim.')
    (out/'selection_manifest.json').write_text(json.dumps(manifests,indent=2)+'\n',encoding='utf-8')
    checks = dict(exactly_50_unique=len(chosen)==chosen.characteristic.nunique()==50,
        only_us_high_cap=bool(frame.member_us_large.all()),no_test_outcomes=bool(frame.return_month.max()<pd.Timestamp('2020-01-01')),
        all_eligible=bool(chosen.eligible.all()),thirteen_groups_with_2_to_6=bool(len(group)==13 and group.features.between(2,6).all()),
        published_anchor_each_group=bool(group.published_representatives.ge(1).all()),
        no_models_fitted=True,reproduced_order=diverse_pick(subset,corr,50,min_theme=2,max_theme=6,score_col='final_score',redundancy_penalty=.20,published_anchor=True).characteristic.tolist()==chosen.characteristic.tolist())
    checks['all_selected_verified_jkp']=bool(chosen.jkp_membership_verified.all() and chosen.published_jkp_153.all())
    for path in (out/'matrices').rglob('*.csv'):
        a=pd.read_csv(path,index_col=0).to_numpy(float)
        checks['symmetric_'+str(path.relative_to(out))]=bool(np.allclose(a,a.T,equal_nan=True,rtol=1e-10,atol=1e-12))
    if not all(checks.values()):raise AssertionError({k:v for k,v in checks.items() if not v})
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n',encoding='utf-8')
    charts(chosen,scores,out)
    report(chosen,scores,monthly,frame,manifests,out)
    print(json.dumps({k:v for k,v in manifests.items() if k!='source_files'},indent=2),flush=True)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-mode',choices=['training','validation'],default='training')
    args=parser.parse_args(argv)
    select_us_high_cap(args.selection_mode)
    return 0
