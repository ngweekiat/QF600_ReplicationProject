"""Build an offline Plotly explorer from saved results, without fitting models."""
from pathlib import Path
import json
import os
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'test_tmp', 'legacy_iterations', 'revision_history', 'pre_revision',
            'verification', 'organisation_audit', 'members', 'pretest_trials'}


def build_plotly(root=ROOT, series=None):
    from plotly.offline import get_plotlyjs
    if series is None:
        from dashboard import collect_series
        series = collect_series(root)
    root = Path(root)
    out = root / 'outputs'
    assets = out / 'dashboard_assets'
    assets.mkdir(exist_ok=True)
    (assets / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    tables, figures, skipped = [], [], []
    paths = []
    for folder, directories, files in os.walk(out, followlinks=False):
        directories[:] = [d for d in directories if d not in EXCLUDED and not (Path(folder) / d).is_symlink()]
        paths.extend(Path(folder) / name for name in files if name.endswith(('.csv', '.png')))
    for path in sorted(paths):
        rel = path.relative_to(out)
        if any(p in EXCLUDED for p in rel.parts) or not path.is_file():
            continue
        group = next((p for p in rel.parts if p.startswith('iteration_')), 'base')
        if path.suffix.lower() == '.png':
            figures.append(dict(group=group, path=rel.as_posix(), title=path.stem.replace('_', ' ')))
        if path.suffix.lower() != '.csv':
            continue
        # Avoid copying stock-level input/history or huge candidate books into a browser.
        if path.stat().st_size > 400_000:
            skipped.append(dict(path=rel.as_posix(), reason='Large table: open source CSV directly'))
            continue
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, UnicodeError, pd.errors.ParserError):
            continue
        numeric = [c for c in frame.select_dtypes(include='number') if frame[c].notna().any()]
        if frame.empty or not numeric:
            continue
        clean = frame.replace([np.inf, -np.inf], np.nan).astype(object).where(pd.notna(frame.replace([np.inf, -np.inf], np.nan)), None)
        tables.append(dict(group=group, path=rel.as_posix(), columns=list(frame), numeric=numeric,
                           rows=clean.to_dict('records')))
    payload = json.dumps(dict(series=series, tables=tables, figures=figures, skipped=skipped), allow_nan=False, default=str).replace('<', '\\u003c')
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kati interactive iteration explorer</title><script src="dashboard_assets/plotly.min.js"></script>
<style>body{font:15px/1.5 system-ui;color:#18364a;background:#f5f8fb;max-width:1450px;margin:24px auto;padding:0 20px}a{color:#136b8c}.card{background:white;border:1px solid #dbe4ec;border-radius:12px;padding:20px;margin:20px 0}select{padding:8px;max-width:100%;margin:5px}label{display:inline-block;margin-right:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px}.chart{min-height:360px}.muted{color:#607083}img{max-width:100%}summary{cursor:pointer}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto;max-height:400px}@media(max-width:600px){.grid{display:block}}</style>
<h1>Interactive iteration explorer</h1><p><a href="dashboard.html">Main dashboard</a> · Saved Kati results; hover, zoom, toggle legends and export each Plotly chart.</p>
<p class="card">Current research chain: Base → 4 → 4.2 → 4.3. Stage 4.2 selects executed historical active utility; 4.3 adds the one-SE simplicity rule. Other trials retain their historical ancestry. P(better) is bootstrap support; 4.2/4.3 have no 80% retention gate.</p>
<div class="card"><label>Market <select id="market"></select></label><label>Iteration <select id="stage"></select></label><label>Comparison <select id="scope"><option value="stage">Selected iteration + Base + Market</option><option value="chain">Current retained chain</option><option value="all">All saved trials</option></select></label><p id="status"></p><div id="summary" class="scroll"></div></div>
<div class="grid" id="charts"></div>
<section class="card"><h2>Annual model choices</h2><p>Selected feature counts and regularisation are read from each saved chosen.csv. Year denotes the year the choice applies to.</p><div id="choices" class="grid"></div></section>
<section class="card"><h2>Interactive source plots and diagnostics</h2><p>Every eligible numeric result CSV can be plotted. Select a table, x-axis, metric, grouping and chart type. Units and definitions follow the source CSV; these exploratory plots do not replace the source methodology.</p><label>Source <select id="dataset"></select></label><p id="source"></p><label>X <select id="x"></select></label><label>Y <select id="y"></select></label><label>Series <select id="split"></select></label><label>Chart <select id="kind"><option>scatter</option><option>bar</option><option>histogram</option><option>heatmap</option><option>box</option></select></label><div id="diagnostic" class="chart"></div><details><summary>Source data</summary><div id="data" class="scroll"></div></details></section>
<section class="card"><h2>Original figure catalogue</h2><p>All existing PNG figures for this iteration are available below. The interactive source explorer above covers numeric result tables, including rationale, validation, annual performance, statistical comparisons and frontier data. Multi-panel originals remain available for their exact annotations.</p><div id="gallery"></div><details><summary>Large source tables</summary><div id="large"></div></details></section>
<script>const D=PAYLOAD, $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}}, layout={template:'plotly_white',margin:{t:55,r:25,b:65,l:70},hovermode:'x unified',legend:{orientation:'h',y:-.25}};
function plot(id,traces,title,yaxis={}){Plotly.react(id,traces,{...layout,title:{text:title,font:{size:16}},yaxis},config)}
function opts(id,values,value){$(id).innerHTML=values.map(v=>`<option value="${esc(v)}">${esc(v.replaceAll('_',' '))}</option>`).join('');if(values.includes(value))$(id).value=value;}
function table(rows,cols){return '<table><tr>'+cols.map(c=>'<th>'+esc(c)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+esc(r[c])+'</td>').join('')+'</tr>').join('')+'</table>'}
const groups=[...new Set([...D.series.map(s=>s.group),...D.tables.map(s=>s.group)])].sort((a,b)=>a==='base'?-1:b==='base'?1:a.localeCompare(b,undefined,{numeric:true}));
opts('market',[...new Set(D.series.map(s=>s.market))],'us_large');opts('stage',groups,'iteration_4.3');
const titles=['Cumulative net excess return (additive)','Additive drawdown from peak','Cumulative net return minus Market','Cumulative trading costs','Monthly turnover','Rolling 12-month net excess return','Annual net excess return','Monthly net return distribution'];
$('charts').innerHTML=titles.map((t,i)=>`<div class="card"><div class="chart" id="chart${i}"></div></div>`).join('');
function refresh(){const market=$('market').value,stage=$('stage').value,scope=$('scope').value;
let ss=D.series.filter(s=>s.market===market&&(s.group==='base'||(scope==='all')||(scope==='chain'?['iteration_4','iteration_4.2','iteration_4.3'].includes(s.group):s.group===stage)));
let dates=ss.length?ss[0].points.map(p=>p.date).filter(d=>ss.every(s=>s.points.some(p=>p.date===d))):[];
$('status').textContent=dates.length?`${dates.length} common months: ${dates[0]} to ${dates.at(-1)}. Returns and drawdowns are additive, not compounded wealth.`:'No common saved months for this selection.';
const rows=ss.map(s=>({...s,p:dates.map(d=>s.points.find(p=>p.date===d))})), marketRow=rows.find(s=>s.label.startsWith('Market'));
const traces=titles.map(()=>[]), stats=[];
for(const s of rows){let cum=0,peak=0,cost=0;const c=[],dd=[],cc=[],roll=[],annual={};s.p.forEach((p,i)=>{cum+=p.net;peak=Math.max(peak,cum);cost+=p.cost;c.push(cum);dd.push(cum-peak);cc.push(cost);roll.push(i<11?null:s.p.slice(i-11,i+1).reduce((a,b)=>a+b.net,0));annual[p.date.slice(0,4)]=(annual[p.date.slice(0,4)]||0)+p.net});
const base={name:s.label,x:dates,type:'scatter',mode:'lines'};[c,dd,marketRow?s.p.map((p,i)=>p.net-marketRow.p[i].net).map((v,i,a)=>a.slice(0,i+1).reduce((x,y)=>x+y,0)):[],cc,s.p.map(p=>p.turnover),roll].forEach((v,i)=>traces[i].push({...base,y:v}));
traces[6].push({type:'bar',name:s.label,x:Object.keys(annual),y:Object.values(annual)});traces[7].push({type:'histogram',name:s.label,x:s.p.map(p=>p.net),opacity:.55});
let n=s.p.length,mean=n?s.p.reduce((a,p)=>a+p.net,0)/n:0,sd=n>1?Math.sqrt(s.p.reduce((a,p)=>a+(p.net-mean)**2,0)/(n-1)):0;
stats.push({Strategy:s.label,'Annual net return':(1200*mean).toFixed(2)+'%','Annual cost':n?(1200*cost/n).toFixed(2)+'%':'','Net volatility':(100*Math.sqrt(12)*sd).toFixed(2)+'%','Sharpe (net SD)':sd?(Math.sqrt(12)*mean/sd).toFixed(3):'N/A'});}
titles.forEach((t,i)=>plot('chart'+i,traces[i],t,i===7?{title:'Months'}:{tickformat:'.1%',title:i===1?'Percentage-point drop (fraction units)':undefined}));$('summary').innerHTML=table(stats,['Strategy','Annual net return','Annual cost','Net volatility','Sharpe (net SD)']);
const selected=D.tables.filter(t=>t.group===stage&&t.path.endsWith('/chosen.csv')&&!/pretest|members|trials/.test(t.path)&&t.columns.includes('p'));
$('choices').innerHTML=selected.length?selected.map((t,i)=>`<div><p><a href="${esc(t.path)}">${esc(t.path)}</a></p><div id="choice${i}" class="chart"></div></div>`).join(''):'<p>No saved annual p choices in this selection.</p>';
selected.forEach((t,i)=>plot('choice'+i,['p','l','g'].filter(c=>t.numeric.includes(c)).map(c=>({type:'scatter',mode:'lines+markers',line:{shape:'hv'},name:c,x:t.rows.map(r=>r.year),y:t.rows.map(r=>r[c]),visible:c==='p'?true:'legendonly'})),'Annual hyperparameter choices'));
const ts=D.tables.filter(t=>t.group===stage);$('dataset').innerHTML=ts.map(t=>`<option value="${esc(t.path)}">${esc(t.path)}</option>`).join('');dataset();
$('gallery').innerHTML=D.figures.filter(f=>f.group===stage).map(f=>`<details><summary>${esc(f.path)}</summary><a href="${esc(f.path)}"><img loading="lazy" src="${esc(f.path)}" alt="${esc(f.title)}"></a></details>`).join('')||'<p>No original figures for this selection.</p>';
$('large').innerHTML=D.skipped.filter(f=>f.path.startsWith(stage+'/')).map(f=>`<p><a href="${esc(f.path)}">${esc(f.path)}</a>: ${esc(f.reason)}</p>`).join('');}
function current(){return D.tables.find(t=>t.path===$('dataset').value)}
function dataset(){const t=current();if(!t){Plotly.purge('diagnostic');$('source').textContent='No numeric source tables.';$('data').innerHTML='';return}opts('x',t.columns,t.columns.find(c=>['eom_ret','date','year','iteration','eom'].includes(c))||t.columns[0]);opts('y',t.numeric,t.numeric.find(c=>c!==$('x').value)||t.numeric[0]);opts('split',['(none)',...t.columns],t.columns.find(c=>['strategy','type','market','theme'].includes(c))||'(none)');diagnostic();}
function diagnostic(){const t=current();if(!t)return;const x=$('x').value,y=$('y').value,split=$('split').value,kind=$('kind').value;const keys=split==='(none)'?['All']:[...new Set(t.rows.map(r=>String(r[split])))];
let traces=keys.map(k=>{const rows=t.rows.filter(r=>r[y]!==null&&(split==='(none)'||String(r[split])===k));return {name:k,type:kind==='heatmap'?'scatter':kind,x:rows.map(r=>r[x]),y:rows.map(r=>r[y]),mode:'lines+markers'}});
if(kind==='histogram')traces=traces.map(t=>({...t,x:t.y,y:undefined,opacity:.65}));
if(kind==='heatmap'){const cats=[...new Set(t.rows.map(r=>String(r[x])))];traces=[{type:'heatmap',x:cats,y:keys,z:keys.map(k=>cats.map(c=>{const rs=t.rows.filter(r=>String(r[x])===c&&(split==='(none)'||String(r[split])===k)&&r[y]!==null);return rs.length?rs.reduce((a,r)=>a+Number(r[y]),0)/rs.length:null})),colorbar:{title:y}}];}
plot('diagnostic',traces,y+' by '+x,{title:kind==='histogram'?'Count':y});$('source').innerHTML=`<a href="${esc(t.path)}">Download source CSV</a> · ${t.rows.length} rows. Heatmaps show cell means for repeated keys. Preview: first 200 rows.`;$('data').innerHTML=table(t.rows.slice(0,200),t.columns);}
['market','stage','scope'].forEach(id=>$(id).onchange=refresh);$('dataset').onchange=dataset;['x','y','split','kind'].forEach(id=>$(id).onchange=diagnostic);refresh();
</script></html>'''
    (out / 'dashboard_plotly.html').write_text(page.replace('PAYLOAD', payload), encoding='utf-8')
    manifest = dict(series=len(series), numeric_tables=len(tables), original_figures=len(figures), large_tables=skipped,
                    offline=True, models_fitted=False, table_sources=[t['path'] for t in tables])
    (out / 'dashboard_plotly_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    print(build_plotly())
