"""Explain the frozen 50-feature recommendation without reselecting or fitting ML.

Read completed diagnostic tables, verify them against the frozen names, and
produce a presentation overview, a 50-row evidence chart and a standalone
interactive HTML guide. All writes remain under the tryout output directory.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .paths import DATA, OUTPUT

MARKETS = ("us_large", "us_mid", "eu_large")
MARKET_LABELS = {"us_large":"US high-cap", "us_mid":"US mid-cap", "eu_large":"EU high-cap"}
REASONS = {"published JKP representative for economic theme":"A",
    "minimum economic-theme representation":"T",
    "balanced predictive score with redundancy penalty":"S"}
SOURCE_FILES = ("recommended_50.csv", "recommended_50_explained.csv", "selected_theme_coverage.csv",
    "selected_50_validation_by_universe.csv", "all_capped_characteristic_scores.csv",
    "selected_feature_redundancy.csv", "screening/screening_decision.json")
WEIGHTS = [("Balanced validation rank IC",35), ("Weakest-market validation rank IC",15),
    ("Worst-market coverage",15), ("Direction stability",10), ("Signal persistence",10),
    ("Single-feature walk-forward R squared",15)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_evidence(source):
    source=Path(source).resolve()
    missing=[name for name in SOURCE_FILES if not (source/name).is_file()]
    if missing:raise FileNotFoundError("Completed selection results are required; missing: "+", ".join(missing))
    selected=pd.read_csv(source/"recommended_50.csv")
    validation=pd.read_csv(source/"selected_50_validation_by_universe.csv")
    candidates=pd.read_csv(source/"all_capped_characteristic_scores.csv")
    decision=json.loads((source/"screening/screening_decision.json").read_text())
    frozen=json.loads((DATA/"recommended_50.json").read_text())
    if len(selected)!=50 or selected.characteristic.duplicated().any() or set(selected.characteristic)!=set(frozen):
        raise ValueError("Saved recommendation does not match the frozen 50 names")
    if len(candidates)!=200 or not selected.eligible.all():raise ValueError("Unexpected candidate eligibility")
    if validation.duplicated(["universe","characteristic"]).any() or len(validation)!=150:
        raise ValueError("Expected exactly 50 validation rows per market")
    if set(validation.universe)!=set(MARKETS):raise ValueError("Unexpected markets")
    if not pd.to_datetime(validation.sample_start).eq(pd.Timestamp('2015-01-31')).all() or not pd.to_datetime(validation.sample_end).eq(pd.Timestamp('2019-12-31')).all():
        raise ValueError("Validation chart must use only 2015-2019 returns")
    validation=validation.assign(oriented_ic=validation.mean_rank_ic*validation.training_direction)
    wide=validation.pivot(index='characteristic',columns='universe',values='oriented_ic')
    cover=validation.pivot(index='characteristic',columns='universe',values='coverage')
    if not np.isfinite(wide.to_numpy()).all() or not np.isfinite(cover.to_numpy()).all():raise ValueError("Missing selected-feature validation evidence")
    s=selected.set_index('characteristic')
    np.testing.assert_allclose(wide.loc[s.index].mean(axis=1),s.validation_oriented_ic,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(cover.loc[s.index].min(axis=1),s.minimum_validation_coverage,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(candidates.set_index('characteristic').loc[s.index,'final_score'],s.final_score)
    counts=selected.groupby('theme').size()
    if len(counts)!=13 or not counts.between(2,6).all():raise ValueError("Theme representation differs from the declared rule")
    if not selected.groupby('theme').published_jkp_153.sum().ge(1).all():raise ValueError("Missing published theme anchor")
    if not selected.minimum_training_coverage.ge(.6).all() or not selected.minimum_validation_coverage.ge(.6).all():
        raise ValueError("Coverage threshold failed")
    if set(selected.selection_reason)!=set(REASONS):raise ValueError("Unknown selection reason")
    selected=selected.sort_values(['theme','final_score'],ascending=[True,False]).reset_index(drop=True)
    for market in MARKETS:
        selected[f'{market}_oriented_ic']=selected.characteristic.map(wide[market])
        selected[f'{market}_coverage']=selected.characteristic.map(cover[market])
    selected['theme_status']=np.where(selected.published_jkp_153,'Published JKP group','Inferred training-correlation proxy')
    selected['reason_code']=selected.selection_reason.map(REASONS)
    return selected,validation,candidates,decision


def overview(selected,candidates,decision,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig=plt.figure(figsize=(16,9),facecolor='white')
    grid=fig.add_gridspec(3,2,height_ratios=[.95,2.4,1.75],width_ratios=[1,1.2],hspace=.55,wspace=.4)
    flow=fig.add_subplot(grid[0,:]);flow.axis('off')
    stages=[(decision['candidates'],'Source candidates'),(decision['eligible'],'Pass training screen'),
        (decision['retained'],'Full-history download'),(int(candidates.eligible.sum()),'Pass final coverage'),(50,'Recommended inputs')]
    for i,(n,label) in enumerate(stages):
        x=.015+i*.2
        flow.add_patch(FancyBboxPatch((x,.12),.165,.7,boxstyle='round,pad=.01,rounding_size=.035',
            transform=flow.transAxes,facecolor='#143952' if i==4 else '#edf3f7',edgecolor='none'))
        color='white' if i==4 else '#17394e'
        flow.text(x+.0825,.6,f'{n}',ha='center',va='center',fontsize=22,weight='bold',color=color,transform=flow.transAxes)
        flow.text(x+.0825,.29,label,ha='center',va='center',fontsize=9,color=color,transform=flow.transAxes)
        if i<4:flow.annotate('',xy=(x+.187,.47),xytext=(x+.17,.47),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#718493'))
    flow.set_title('A  The shortlist was built in stages',loc='left',fontsize=12,weight='bold',pad=9)
    groups=selected.groupby('theme').agg(total=('characteristic','size'),published=('published_jkp_153','sum')).sort_index()
    groups['proxy']=groups.total-groups.published
    ax=fig.add_subplot(grid[1:,0]);y=np.arange(len(groups))
    ax.barh(y,groups.published,color='#247a9e',label=f'Published group labels ({int(groups.published.sum())})')
    ax.barh(y,groups.proxy,left=groups.published,color='#df9b36',label=f'Provisional proxy labels ({int(groups.proxy.sum())})')
    ax.set_yticks(y,groups.index);ax.invert_yaxis();ax.set_xlim(0,7.1);ax.set_xticks(range(0,8));ax.set_xlabel('Number of selected characteristics')
    for i,n in enumerate(groups.total):ax.text(n+.12,i,str(n),va='center',fontsize=10,weight='bold')
    ax.axvline(2,color='#738391',ls=':',lw=1);ax.axvline(6,color='#738391',ls=':',lw=1)
    ax.set_title('B  All 13 groups are represented',loc='left',fontsize=12,weight='bold',pad=12)
    ax.legend(loc='lower left',bbox_to_anchor=(0,-.19),frameon=False,fontsize=9)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax=fig.add_subplot(grid[1,1]);labels,values=zip(*WEIGHTS)
    ax.barh(np.arange(6),values,color=['#247a9e','#247a9e','#5b9567','#8b79a3','#df9b36','#247a9e'])
    ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,42)
    for i,v in enumerate(values):ax.text(v+1,i,f'{v}%',va='center',fontsize=10)
    ax.set_title('C  The final score balances six criteria',loc='left',fontsize=12,weight='bold',pad=12)
    ax.set_xlabel('Fixed score weight, applied to percentile ranks');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax=fig.add_subplot(grid[2,1]);counts=selected.reason_code.value_counts()
    labels=['Published group anchor','Minimum group quota','Score after redundancy penalty']
    nums=[int(counts.get(c,0)) for c in ('A','T','S')]
    ax.barh(range(3),nums,color=['#247a9e','#8b79a3','#5b9567']);ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlim(0,29)
    for i,n in enumerate(nums):ax.text(n+.5,i,str(n),va='center')
    ax.set_title('D  Why each place in the shortlist was allocated',loc='left',fontsize=12,weight='bold',pad=11)
    ax.set_xlabel('Number of characteristics');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for axis in fig.axes:
        for spine in ('top','right'):axis.spines[spine].set_visible(False)
    fig.suptitle('Why the proposed 50 characteristics were selected',x=.04,ha='left',fontsize=21,weight='bold',color='#17394e')
    fig.text(.04,.925,'Coverage across US high-cap, US mid-cap and EU high-cap, with predictive evidence and economic-theme balance',fontsize=11,color='#425a6a')
    fig.text(.04,.08,'Training screen: 2005-2014. Final selection evidence: 2015-2019. Returns from 2020-2025 did not select the features.',fontsize=9)
    fig.text(.04,.055,'Each group has 2-6 inputs and at least one published representative. Greedy choice adds 0.02 for published membership and subtracts 0.20 x maximum redundancy.',fontsize=9)
    fig.text(.04,.03,f'{int((~selected.published_jkp_153).sum())} group labels are inferred proxies, not official economic classifications. {int(selected.validation_oriented_ic.lt(0).sum())} selected feature has negative validation IC; {int(selected.mean_walk_forward_r2.lt(0).sum())} have negative single-feature validation R squared.',fontsize=9)
    fig.subplots_adjust(left=.15,right=.97,top=.87,bottom=.2)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'selection_overview.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)


def detail_chart(selected,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    n=len(selected);fig=plt.figure(figsize=(17,19),facecolor='white')
    gs=fig.add_gridspec(1,5,width_ratios=[1.8,2.4,1,1,.8],wspace=.1)
    group=fig.add_subplot(gs[0,0]);group.set_xlim(0,1);group.set_ylim(n-.5,-.5);group.axis('off')
    themes=list(selected.theme.unique());colors=plt.get_cmap('tab20').colors
    starts=[]
    for j,theme in enumerate(themes):
        ix=np.flatnonzero(selected.theme.eq(theme));lo,hi=ix.min(),ix.max();starts.append(lo)
        group.axhspan(lo-.45,hi+.45,xmin=.02,xmax=.49,color=colors[j%len(colors)],alpha=.22)
        group.text(.06,(lo+hi)/2,f'{theme}\n{len(ix)} inputs',va='center',fontsize=10,weight='bold')
    matrix=selected[[f'{u}_oriented_ic' for u in MARKETS]].to_numpy()*100
    ax=fig.add_subplot(gs[0,1]);bound=max(3,float(np.nanmax(abs(matrix))))
    image=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-bound,vmax=bound)
    labels=[r.characteristic+(' [J]' if r.published_jkp_153 else ' [P]') for r in selected.itertuples()]
    ax.set_yticks(range(n),labels,fontsize=8.5);ax.set_xticks(range(3),['US high-cap','US mid-cap','EU high-cap'],fontsize=9)
    ax.xaxis.tick_top();ax.tick_params(axis='y',length=0)
    ax.set_title('Validation rank IC x 100',pad=30,fontsize=12,weight='bold')
    for y in range(n):
        for x in range(3):ax.text(x,y,f'{matrix[y,x]:+.1f}',ha='center',va='center',fontsize=8,color='white' if abs(matrix[y,x])>bound*.62 else '#172e3e')
    coverage=fig.add_subplot(gs[0,2]);vals=selected.minimum_validation_coverage.to_numpy()
    coverage.imshow(vals[:,None],aspect='auto',cmap='Greens',vmin=.6,vmax=1)
    coverage.set_title('Minimum\ncoverage',pad=15,fontsize=11,weight='bold');coverage.set_xticks([]);coverage.set_yticks([])
    for y,v in enumerate(vals):coverage.text(0,y,f'{v:.0%}',ha='center',va='center',fontsize=8,color='white' if v>.88 else '#17394e')
    score=fig.add_subplot(gs[0,3]);v=selected.final_score.to_numpy();score.imshow(v[:,None],aspect='auto',cmap='Blues',vmin=0,vmax=1)
    score.set_title('Final score\n0 to 1',pad=15,fontsize=11,weight='bold');score.set_xticks([]);score.set_yticks([])
    for y,value in enumerate(v):score.text(0,y,f'{value:.2f}',ha='center',va='center',fontsize=8,color='white' if value>.65 else '#17394e')
    reason=fig.add_subplot(gs[0,4]);codes=selected.reason_code.map({'A':0,'T':1,'S':2}).to_numpy()
    reason.imshow(codes[:,None],aspect='auto',cmap=ListedColormap(['#c6e0ed','#e0d6eb','#d5e5d8']),vmin=0,vmax=2)
    reason.set_title('Selection\nreason',pad=15,fontsize=11,weight='bold');reason.set_xticks([]);reason.set_yticks([])
    for y,c in enumerate(selected.reason_code):reason.text(0,y,c,ha='center',va='center',fontsize=9,weight='bold')
    for axis in (ax,coverage,score,reason):
        for start in starts[1:]:axis.axhline(start-.5,color='white',lw=2)
        for spine in axis.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.50,.073,.35,.012])
    cb=fig.colorbar(image,cax=cax,orientation='horizontal')
    cb.set_label('Training-oriented validation correlation x 100: positive supports the training direction; negative reverses it',fontsize=9)
    fig.suptitle('The 50 proposed inputs and their selection evidence',fontsize=21,weight='bold',x=.055,ha='left',y=.985,color='#17394e')
    fig.text(.055,.958,'Monthly cross-sectional correlations with next-month observed stock excess returns, 2015-2019. Sign learned before 2015.',fontsize=10)
    fig.text(.055,.04,'[J] Published JKP group label.  [P] Provisional group assigned by training correlations; review before treating it as an economic label.',fontsize=10)
    fig.text(.055,.025,'A = published group anchor; T = minimum theme representation; S = balanced score with redundancy penalty. Coverage is the worst of the three markets.',fontsize=10)
    fig.text(.055,.01,'An IC value of +2.0 here means correlation +0.020, not a 2% return. Final score is a fixed weighted percentile score, not a probability or ML coefficient.',fontsize=10)
    fig.subplots_adjust(left=.055,right=.97,top=.9,bottom=.12)
    for ext in ('png','svg','pdf'):fig.savefig(out/f'selected_50_evidence.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)


def interactive_page(selected,candidates,decision,out):
    records=json.loads(selected.replace({np.inf:None,-np.inf:None}).to_json(orient='records'))
    payload=json.dumps(records).replace('<','\\u003c')
    options=''.join(f'<option>{html.escape(t)}</option>' for t in selected.theme.unique())
    page='''<!doctype html><meta charset="utf-8"><title>Why these 50 characteristics</title>
<style>body{max-width:1450px;margin:36px auto;padding:0 24px;font:16px/1.5 system-ui;color:#203748}h1{font-size:34px;margin-bottom:5px}h2{margin-top:30px}.intro{font-size:18px;max-width:1000px}img{max-width:100%}.controls{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}select,input{padding:8px;font:inherit;border:1px solid #aebec8;border-radius:5px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dce4e9;text-align:left}th{background:#edf3f7;position:sticky;top:0}td.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto;max-height:780px}.proxy{color:#9b600b}.negative{color:#a52828}.small{font-size:13px;color:#526574}a{color:#126388}</style>
<h1>Why the proposed 50 characteristics were selected</h1>
<p class="intro">The shortlist balances predictive evidence, data coverage, persistence, economic-group representation and redundancy. It is a research recommendation for a later model refit.</p>
<p><b>50 inputs, 13 groups, three markets.</b> Training screening uses 2005-2014; final selection evidence uses 2015-2019. Returns from 2020-2025 did not select the features.</p>
<img src="selection_overview.png" alt="Selection stages, theme distribution, scoring weights and reasons">
<p><a href="selection_overview.png">Download overview PNG</a> | <a href="selection_overview.pdf">PDF</a> | <a href="selection_overview.svg">Editable SVG</a></p>
<h2>Explain an individual feature or group</h2>
<p>Choose a group or search a name. Correlations below have been multiplied by the direction learned in training; positive values support that direction. Each market is weighted equally.</p>
<div class="controls"><label>Group <select id="theme"><option value="">All groups</option>OPTIONS</select></label><label>Search <input id="search" placeholder="Name or description"></label><label>Group source <select id="status"><option value="">All labels</option><option value="published">Published JKP only</option><option value="proxy">Inferred proxies only</option></select></label></div>
<p id="count"></p><div class="scroll"><table><thead><tr><th>Characteristic and meaning</th><th>Group and source</th><th>US high-cap IC</th><th>US mid-cap IC</th><th>EU high-cap IC</th><th>Worst coverage</th><th>Final score</th><th>Why selected</th></tr></thead><tbody id="body"></tbody></table></div>
<h2>All 50 inputs at a glance</h2><img src="selected_50_evidence.png" alt="Grouped 50-row scorecard of market correlations, coverage, score and selection reasons">
<p><a href="selected_50_evidence.pdf">Detailed PDF chart</a> | <a href="selected_50_evidence.svg">Editable SVG</a> | <a href="selection_evidence.csv">Underlying evidence CSV</a> | <a href="verification.json">Verification and source hashes</a></p>
<h2>How to explain the choice accurately</h2>
<ul><li>The final score uses fixed weights: 35% balanced validation IC, 15% weakest-market IC, 15% coverage, 10% directional stability, 10% persistence and 15% single-feature walk-forward R squared. These weights are not learned by an ML model.</li>
<li>The greedy selector requires at least one eligible published representative in each group, 2-6 total inputs per group, and penalizes the strongest absolute correlation with inputs already selected. It adds a small 0.02 preference for published membership and subtracts 0.20 times the redundancy measure.</li>
<li>PUBCOUNT chosen inputs have published JKP group labels; PROXYCOUNT use empirical training-correlation proxies. A proxy is provisional and can differ from the feature's ordinary economic meaning.</li>
<li>NEGICCOUNT selected input(s) have negative average validation IC; NEGR2COUNT have negative single-feature walk-forward R squared. Some places are reserved for group representation. A selection chart is not evidence that every input is independently strong.</li>
<li>Redundancy is penalized, not eliminated: some selected pairs remain highly correlated. Coverage means observed data availability, not predictive accuracy. Persistence is a turnover-related proxy, not measured trading cost.</li>
<li>The shortlist has not been evaluated as a newly fitted Static ML or Portfolio ML portfolio. It is not the utility-based permutation importance in the paper. Keats still uses its original 50 inputs.</li></ul>
<script>const rows=PAYLOAD;const names={A:'Published representative for group',T:'Minimum group representation',S:'Balanced predictive score with redundancy penalty'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ic=x=>`<td class="num ${x<0?'negative':''}">${Number(x).toFixed(3)}</td>`;
function update(){let theme=document.getElementById('theme').value,q=document.getElementById('search').value.toLowerCase(),status=document.getElementById('status').value;let show=rows.filter(r=>(!theme||r.theme===theme)&&(!q||(r.characteristic+' '+r.description).toLowerCase().includes(q))&&(!status||(status==='published'?r.published_jkp_153:!r.published_jkp_153)));
document.getElementById('count').textContent=show.length+' of 50 characteristics shown';document.getElementById('body').innerHTML=show.map(r=>`<tr><td><b>${esc(r.characteristic)}</b><br>${esc(r.name_new||r.description)}</td><td>${esc(r.theme)}<br><span class="${r.published_jkp_153?'small':'proxy'}">${r.published_jkp_153?'Published JKP label':'Provisional correlation proxy'}</span></td>${ic(r.us_large_oriented_ic)}${ic(r.us_mid_oriented_ic)}${ic(r.eu_large_oriented_ic)}<td class="num">${(r.minimum_validation_coverage*100).toFixed(1)}%</td><td class="num">${r.final_score.toFixed(3)}</td><td>${esc(names[r.reason_code])}</td></tr>`).join('');}
for(let id of ['theme','search','status'])document.getElementById(id).addEventListener('input',update);update();</script>'''
    page=page.replace('OPTIONS',options).replace('PAYLOAD',payload)
    for key,value in {'PUBCOUNT':int(selected.published_jkp_153.sum()),'PROXYCOUNT':int((~selected.published_jkp_153).sum()),'NEGICCOUNT':int(selected.validation_oriented_ic.lt(0).sum()),'NEGR2COUNT':int(selected.mean_walk_forward_r2.lt(0).sum())}.items():page=page.replace(key,str(value))
    (out/'index.html').write_text(page,encoding='utf-8')


def explain_selection(source=None):
    if source is None:source=OUTPUT if (OUTPUT/'recommended_50.csv').exists() else OUTPUT/'selection_explanation/source_tables'
    selected,validation,candidates,decision=load_evidence(source)
    out=OUTPUT/'selection_explanation';out.mkdir(parents=True,exist_ok=True)
    source_dir=out/'source_tables';source_dir.mkdir(exist_ok=True)
    hashes={}
    for name in SOURCE_FILES:
        original=Path(source)/name;dest=source_dir/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if original.resolve()!=dest.resolve():shutil.copy2(original,dest)
        assert sha(original)==sha(dest)
        hashes[name]=dict(source=str(original.resolve()),sha256=sha(dest))
    selected.to_csv(out/'selection_evidence.csv',index=False)
    selected.groupby('theme').agg(selected=('characteristic','size'),published=('published_jkp_153','sum'),
        minimum_validation_coverage=('minimum_validation_coverage','min')).to_csv(out/'group_summary.csv')
    overview(selected,candidates,decision,out);detail_chart(selected,out);interactive_page(selected,candidates,decision,out)
    verification=dict(status='PASS',frozen_50_names_unchanged=True,selected=50,groups=13,
        published_labels=int(selected.published_jkp_153.sum()),proxy_labels=int((~selected.published_jkp_153).sum()),
        candidate_count=len(candidates),final_eligible=int(candidates.eligible.sum()),
        validation_rows=len(validation),validation_period='2015-2019',test_outcomes_used=False,
        ml_models_refitted=False,per_stock_time_series_correlations_used_for_selection=False,
        negative_validation_ic=int(selected.validation_oriented_ic.lt(0).sum()),
        negative_single_feature_r2=int(selected.mean_walk_forward_r2.lt(0).sum()),source_files=hashes)
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    (out/'README.md').write_text('''# Explaining the proposed 50 characteristics

Open index.html for the interactive guide. selection_overview.png is the presentation overview; selected_50_evidence.png shows all 50 inputs grouped by theme with market-specific validation evidence. PDF and SVG versions are supplied. selection_evidence.csv contains the underlying values and exact selection reasons.

These graphics explain the existing frozen rule-based recommendation. They do not reselect features or fit an ML portfolio. They use monthly cross-sectional characteristic/next-month-return rank correlations, not each stock's time-series correlation with accumulated returns. Twenty-two group labels are provisional training-correlation proxies; they are identified explicitly.

Source CSV/JSON snapshots and their SHA256 hashes are retained in source_tables and verification.json. Keats original inputs and the existing recommended_50.json remain unchanged.
''',encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k!='source_files'},indent=2))
    print('Open',out/'index.html')
    return 0
