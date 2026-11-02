"""Build a local interactive viewer of actual Kati results; never fit a model.

Selectors switch market, experiment, metric and visible strategies. Missing
results remain explicitly unavailable. Arithmetic cumulative returns are shown
because unconstrained strategies can have monthly net returns below -100%.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MARKETS = {"us_large": "US high-cap", "us_mid": "US mid-cap", "eu_large": "EU high-cap"}


def collect_series(root=ROOT):
    """Read only local completed portfolio files; keep reference identities clear."""
    root = Path(root)
    records = []
    for market in MARKETS:
        base = root / "outputs/base" / market
        for folder, label in (("static_ml", "Static ML"), ("portfolio_ml/rff", "Portfolio ML base")):
            records.append((market, "base", label, base / folder / "pf.parquet"))
        records.append((market, "base", "Market (base basket)", root / "outputs/base/overall analysis/market_references" / market / "pf.parquet"))
    for number in (1, 2, 3):
        base = root / f"outputs/iteration_{number}/us_large"
        owner = number if number < 3 else 2
        cal = root / f"outputs/iteration_{owner}/us_large/active/calibrated.csv"
        if not cal.exists():
            continue
        for row in pd.read_csv(cal).itertuples():
            if number == 3 and not np.isclose(row.te_target, .02):
                continue
            records.append(("us_large", f"iteration_{number}", f"Portfolio ML iteration {number} (TE {row.te_target:.0%})", base / f"active/gte{row.gamma_te:g}/pf.parquet"))
        if number == 2:
            records.append(("us_large", "iteration_2", "Market (250-stock basket)", base / "benchmarks/pf_test.parquet"))
    series = []
    for market, group, label, path in records:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "type" in frame and label.startswith("Market"):
            frame = frame[frame.type.eq("Market")]
        elif "type" in frame:
            expected = "Static-ML" if label == "Static ML" else ("Portfolio-ML" if group == "base" else "Portfolio-ML (active)")
            if group == "base" and set(frame.type) != {expected}:
                raise ValueError(f"Unexpected strategy identity in {path}")
        if frame.empty:
            continue
        required = {"eom_ret", "r", "tc", "turnover"}
        if not required.issubset(frame):
            raise ValueError(f"Incomplete portfolio result: {path}")
        frame = frame.sort_values("eom_ret")
        if frame.eom_ret.duplicated().any():
            raise ValueError(f"Duplicate result month: {path}")
        values = frame[["r", "tc", "turnover"]].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite result: {path}")
        net = values[:, 0] - values[:, 1]
        sd = float(np.std(net, ddof=1)) if len(net) > 1 else 0.
        points = [dict(date=pd.Timestamp(row.eom_ret).strftime("%Y-%m-%d"), gross=float(row.r), cost=float(row.tc), net=float(row.r-row.tc), turnover=float(row.turnover)) for row in frame.itertuples()]
        series.append(dict(market=market, group=group, label=label, source=str(path.relative_to(root)).replace("\\", "/"), points=points,
            summary=dict(months=len(frame), first=points[0]["date"], last=points[-1]["date"], annual_net=float(12*net.mean()), annual_cost=float(12*values[:, 1].mean()), net_vol=float(np.sqrt(12)*sd), net_sharpe=float(np.sqrt(12)*net.mean()/sd) if sd else None,
                         nonpositive_wealth_months=int((1+net <= 0).sum()))))
    return series


def build_dashboard(root=ROOT):
    root = Path(root)
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    series = collect_series(root)
    feature_path = root / "data/characteristics/selected_feature_metadata.csv"
    features = pd.read_csv(feature_path).replace({np.nan: None}).to_dict("records") if feature_path.exists() else []
    payload = json.dumps(dict(series=series, features=features), allow_nan=False, default=str).replace("<", "\\u003c")
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Kati_tryout results</title>
<style>body{max-width:1250px;margin:30px auto;padding:0 24px;font:15px/1.55 system-ui;color:#18364a;background:#f9fbfc}h1{font-size:34px}h2{margin-top:32px}select,input,button{font:inherit}select{padding:7px}label{margin-right:18px;display:inline-block}a{color:#126084}.card{padding:20px;background:white;border:1px solid #dce5eb;border-radius:12px;margin:18px 0}.note{background:#edf3f8;padding:15px;border-radius:8px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #dce5eb;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf3f8}img{max-width:100%}svg{width:100%;height:auto}.muted{color:#597080}.warn{color:#9b4a18}#legend label{font-size:13px;margin:5px 15px 5px 0}nav a{margin-right:18px}summary{cursor:pointer;font-weight:650}</style>
<h1>Kati_tryout</h1><p>The frozen research 50, using the Keats model mathematics in an independent pipeline.</p>
<nav><a href="us_high_cap_selection/index.html">US high-cap selection and matrices</a><a href="#models">Model results</a><a href="#features">Why these 50?</a><a href="../docs/USER_GUIDE.md">User guide</a><a href="../docs/ASSUMPTIONS_AND_CHANGES.md">Assumptions and changes</a></nav>
<div class="note"><b>Current selection: US high-cap only, 2005–2014.</b> These 50 were ranked using observed correlations, coverage, stability and persistence, with group quotas and redundancy penalties. No ML model was fitted for this selection. Old characteristic archives and their fitted results have been removed. Performance for the new 50 requires a later explicit refit.</div>
<section id="models"><h2>Model comparisons</h2><div class="card"><label>Market <select id="market"><option value="us_large">US high-cap</option><option value="us_mid">US mid-cap</option><option value="eu_large">EU high-cap</option></select></label><label>Comparison <select id="group"><option value="base">Base</option><option value="iteration_1">Iteration 1 + base</option><option value="iteration_2">Iteration 2 + earlier versions</option><option value="iteration_3">All iterations + base</option></select></label><label>Chart <select id="metric"><option value="net">Cumulative arithmetic net excess return</option><option value="gross">Cumulative arithmetic gross excess return</option><option value="cost">Cumulative modelled cost</option><option value="turnover">Monthly turnover</option></select></label><div id="legend"></div><p id="status" class="muted"></p><div id="plot"></div><div class="scroll"><table id="metrics"></table></div><p class="muted">Charts compare common available months. Cumulative return lines add monthly excess returns; they do not represent compounded investor wealth. The table uses the same common months. Modelled costs and unconstrained weights follow Keats.</p></div>
<details class="card"><summary>Native reports, CSV tables and frontier charts</summary><ul id="reports"></ul><p>After running all models, the native overall-analysis reports include statistical comparisons, uncertainty, annual/subperiod tables and static charts. Figure 1 requires its separate fresh grid fits.</p></details></section>
<section id="features"><h2>The selected 50 and their groups</h2><div class="note">50 published JKP group labels; no provisional groups. The candidate whitelist requires both JKP reference files and WRDS source availability. Selection uses US high-cap observations from 2005–2014. A = published anchor, T = group quota, S = association score after redundancy penalty. <a href="us_high_cap_selection/index.html">Open the decision chart, correlations, covariances and matrix downloads</a>.</div>
<div class="card"><label>Group <select id="theme"><option value="">All groups</option></select></label><label>Find <input id="search" placeholder="Characteristic or description"></label><p id="feature_count"></p><div class="scroll"><table id="featuretable"></table></div></div>
<details class="card" open><summary>Selection overview</summary>OVERVIEW</details><details class="card"><summary>Evidence for each of the 50</summary>DETAIL</details></section>
<script>const DATA=PAYLOAD;
const $=x=>document.getElementById(x), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const colors=['#16739b','#b35a27','#4b885e','#845aa0','#ca9b2d','#c45065','#35636a','#535e96'];
const pct=x=>Number.isFinite(x)?(100*x).toFixed(2)+'%':'—', num=x=>Number.isFinite(x)?x.toFixed(3):'—'; let visible=new Set(DATA.series.map(s=>s.label));
function candidates(){let level=Number($('group').value.split('_')[1]||0),market=$('market').value;return DATA.series.filter(s=>s.market===market&&(s.group==='base'||Number(s.group.split('_')[1])<=level));}
function draw(){let all=candidates();$('legend').innerHTML=all.map((s,i)=>`<label style="color:${colors[i%colors.length]}"><input type="checkbox" data-label="${esc(s.label)}" ${visible.has(s.label)?'checked':''}> ${esc(s.label)}</label>`).join('');$('legend').querySelectorAll('input').forEach(el=>el.onchange=()=>{el.checked?visible.add(el.dataset.label):visible.delete(el.dataset.label);draw()});
let ss=all.filter(s=>visible.has(s.label));if(!ss.length){$('status').textContent=all.length?'Select at least one strategy.':'No completed Kati model results yet. Run the models using the notebook controls or scripts/run_all.py.';$('plot').innerHTML='';$('metrics').innerHTML='';return;}
let dates=ss[0].points.map(p=>p.date).filter(d=>ss.every(s=>s.points.some(p=>p.date===d)));if(!dates.length){$('status').textContent='These strategies have no common result months.';$('plot').innerHTML='';$('metrics').innerHTML='';return;}
let key=$('metric').value; let rows=ss.map(s=>({...s,p:dates.map(d=>s.points.find(p=>p.date===d))}));let curves=rows.map(s=>{let c=0;return s.p.map(p=>key==='turnover'?p[key]:(c+=p[key]));});let vals=curves.flat(),lo=Math.min(0,...vals),hi=Math.max(0,...vals);if(hi===lo)hi=lo+1;let pad=(hi-lo)*.08;lo-=pad;hi+=pad;
const W=1080,H=390,L=78,R=22,T=25,B=50,x=i=>L+(W-L-R)*i/Math.max(1,dates.length-1),y=v=>T+(H-T-B)*(hi-v)/(hi-lo);
let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Selected strategy comparison">`;
for(let i=0;i<5;i++){let v=lo+(hi-lo)*i/4;svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#dde5eb"/><text x="${L-9}" y="${y(v)+4}" text-anchor="end" font-size="12">${key==='turnover'?v.toFixed(1):pct(v)}</text>`;}
curves.forEach((v,j)=>{let index=all.findIndex(s=>s.label===rows[j].label);svg+=`<polyline fill="none" stroke="${colors[index%colors.length]}" stroke-width="2.5" points="${v.map((z,i)=>`${x(i)},${y(z)}`).join(' ')}"/>`;});
[0,Math.floor((dates.length-1)/2),dates.length-1].forEach(i=>svg+=`<text x="${x(i)}" y="${H-15}" text-anchor="middle" font-size="12">${dates[i]}</text>`);$('plot').innerHTML=svg+'</svg>';
$('status').textContent=`${dates.length} common monthly observations: ${dates[0]} to ${dates.at(-1)}. Source files shown below.`;
$('metrics').innerHTML='<tr><th>Strategy</th><th>Annual net return</th><th>Annual cost</th><th>Annual net volatility</th><th>Net Sharpe</th><th>Months net ≤ −100%</th><th>Data</th></tr>'+rows.map(s=>{let n=s.p.length,mean=s.p.reduce((a,p)=>a+p.net,0)/n,sd=n>1?Math.sqrt(s.p.reduce((a,p)=>a+(p.net-mean)**2,0)/(n-1)):0,cost=s.p.reduce((a,p)=>a+p.cost,0)/n;return `<tr><td>${esc(s.label)}</td><td>${pct(12*mean)}</td><td>${pct(12*cost)}</td><td>${pct(Math.sqrt(12)*sd)}</td><td>${sd?num(Math.sqrt(12)*mean/sd):'—'}</td><td>${s.p.filter(p=>p.net<=-1).length}</td><td><a href="${esc(s.source.replace(/^outputs\//,''))}">Parquet</a></td></tr>`}).join('');}
['market','group','metric'].forEach(id=>$(id).onchange=()=>{if($('market').value!=='us_large')$('group').value='base';draw()});
let themes=[...new Set(DATA.features.map(f=>f.theme))].sort();$('theme').innerHTML+=themes.map(t=>`<option>${esc(t)}</option>`).join('');function features(){let query=$('search').value.toLowerCase(),theme=$('theme').value,ff=DATA.features.filter(f=>(!theme||f.theme===theme)&&`${f.characteristic} ${f.description}`.toLowerCase().includes(query));$('feature_count').textContent=`${ff.length} of ${DATA.features.length} characteristics`;$('featuretable').innerHTML='<tr><th>Characteristic</th><th>Description</th><th>Group</th><th>Group evidence</th><th>US high-cap 2005–2014 rank IC</th><th>US high-cap coverage</th><th>Selection reason</th></tr>'+ff.map(f=>`<tr><td>${esc(f.characteristic)}</td><td style="text-align:left">${esc(f.description)}</td><td>${esc(f.theme)}</td><td>${f.published_jkp_153===true?'Published':'Provisional proxy'}</td><td>${num(Number(f.training_mean_rank_ic))}</td><td>${pct(Number(f.score_coverage))}</td><td>${esc(f.reason_code||f.selection_reason)}</td></tr>`).join('');}['theme','search'].forEach(id=>$(id).oninput=features);
const reports=REPORTS;$('reports').innerHTML=reports.map(r=>`<li><a href="${esc(r.path)}">${esc(r.title)}</a></li>`).join('')||'<li>No fitted-model reports yet.</li>';features();draw();
</script></html>'''
    links = []
    for rel, title in (("base/overall analysis/index.html", "Base: all markets"), ("base/overall analysis/all_models_and_iterations/index.html", "All base models and iterations"), ("base/overall analysis/us_high_cap_progression/index.html", "Four US high-cap Portfolio ML versions"), ("base/overall analysis/figure1/index.html", "Figure 1 frontiers")):
        if (out / rel).exists():
            links.append(dict(path=rel, title=title))
    for i in (1, 2, 3):
        rel = f"iteration_{i}/us_large/overall analysis/index.html"
        if (out / rel).exists():
            links.append(dict(path=rel, title=f"Iteration {i} comparison"))
    for token, filename in (("OVERVIEW", "selection_groups_and_reasons.png"), ("DETAIL", "selected50_return_association.png")):
        rel = "us_high_cap_selection/charts/" + filename
        content = f'<img src="{rel}" alt="Frozen characteristic selection evidence">' if (out / rel).exists() else '<p>Selection chart not bundled; the table above retains the frozen evidence.</p>'
        template = template.replace(token, content)
    template = template.replace("PAYLOAD", payload).replace("REPORTS", json.dumps(links))
    (out / "dashboard.html").write_text(template, encoding="utf-8")
    (out / "dashboard_status.json").write_text(json.dumps(dict(completed_series=len(series), feature_count=len(features), models_fitted_by_dashboard=False, reported_results="Kati local outputs only"), indent=2)+"\n", encoding="utf-8")
    print(f"Dashboard: {out/'dashboard.html'} ({len(series)} completed model/reference series)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build_dashboard())
