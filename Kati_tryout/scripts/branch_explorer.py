"""Build a local explorer for the Wee Kiat Attempt scripts branch."""
from pathlib import Path
import json
import html

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'wee-kiat-attempt'
REMOTE = 'https://github.com/ngweekiat/QF600_ReplicationProject'
SCRIPTS = [
    ('00_download.py', 'Download source data', 'Acquire WRDS/Fama-French inputs and record the data pull.','download'),
    ('01_prepare.py', 'Prepare data', 'Build point-in-time panels, wealth paths, membership and timing-safe inputs.','prepare'),
    ('02_covariance.py', 'Estimate covariance', 'Construct rolling covariance and risk inputs used by portfolio methods.','risk'),
    ('03_return_model.py', 'Fit return model', 'Tune and fit the return-forecast model over random features and ridge penalties.','model'),
    ('04_benchmarks.py', 'Run benchmarks', 'Create market, minimum-variance, Factor-ML and Markowitz-ML references.','benchmark'),
    ('05_static_ml.py', 'Run Static-ML*', 'Estimate the predict-then-optimise strategy and save annual choices and portfolios.','portfolio'),
    ('06_pfml_inputs.py', 'Build Portfolio-ML moments', 'Precompute sufficient statistics for nonlinear and linear portfolio signals.','portfolio'),
    ('07_portfolio_ml.py', 'Run Portfolio-ML', 'Tune yearly signal persistence, feature count and regularisation, then implement weights.','portfolio'),
    ('08_grids.py', 'Efficient frontier grid', 'Rerun and retune methods across risk aversion and investor wealth.','frontier'),
    ('09_feature_importance.py', 'Feature importance', 'Permute characteristic themes and measure the change in realised utility.','diagnostic'),
    ('10_regimes.py', 'Regime analysis', 'Summarise recession, volatility and rate regimes and their portfolio outcomes.','diagnostic'),
    ('11_monitor.py', 'Monitor results', 'Check saved portfolios for changes, risk flags and anomalous annual choices.','diagnostic'),
    ('11_tables.py', 'Write tables', 'Generate comparison tables and reporting summaries from saved results.','report'),
    ('12_active_pfml.py', 'Active Portfolio-ML', 'Calibrate tracking-error active portfolios and compare against benchmark-relative references.','active'),
    ('13_breadth.py', 'Universe breadth', 'Compare portfolio capacity and performance as the investable universe expands.','capacity'),
    ('14_halflife.py', 'Half-life test', 'Test whether changing the trading or signal half-life improves realised outcomes.','iteration'),
    ('15_holdout.py', 'Holdout evaluation', 'Run the frozen strategy on the post-publication holdout period.','holdout'),
    ('16_macro_stress.py', 'Macro stress tests', 'Re-cost and classify portfolios under rates, VIX and recession scenarios.','stress'),
    ('17_report_figures.py', 'Report figures', 'Assemble paper replication, frontier, iteration, stress and appendix figures.','report'),
]


def build(root=ROOT):
    root = Path(root)
    out = root / 'outputs' / 'wee_kiat_attempt_explorer.html'
    payload = json.dumps(dict(branch=BRANCH, remote=REMOTE, scripts=[
        dict(file=f, title=t, description=d, category=c,
             url=f'{REMOTE}/blob/{BRANCH}/scripts/{f}') for f,t,d,c in SCRIPTS
    ]), ensure_ascii=False).replace('<', '\\u003c')
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Wee Kiat Attempt — script explorer</title>
<style>body{font:15px/1.55 system-ui;color:#18364a;background:#f6f9fb;max-width:1250px;margin:25px auto;padding:0 22px}h1{font-size:32px;margin-bottom:5px}a{color:#0d698d}.card{background:white;border:1px solid #d9e4eb;border-radius:12px;padding:18px;margin:18px 0}.bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.bar input,.bar select{font:inherit;padding:8px;border:1px solid #c6d4de;border-radius:7px}.pipeline{display:flex;flex-wrap:wrap;gap:8px}.step{background:#e8f3f5;border:1px solid #a8cdd1;border-radius:8px;padding:8px 11px}.step b{display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.script{border:1px solid #d9e4eb;border-radius:10px;padding:14px;background:#fff}.script h3{margin:0 0 4px}.tag{display:inline-block;background:#edf1f5;border-radius:12px;padding:2px 8px;font-size:12px;color:#526b78}.muted{color:#607686}.count{font-weight:650}</style>
<h1>Wee Kiat Attempt — script explorer</h1><p class="muted">Branch <code>BRANCH</code> · <a href="REMOTE/tree/BRANCH/scripts">Open the source branch on GitHub</a></p>
<div class="card"><h2>Pipeline order</h2><div class="pipeline" id="pipeline"></div><p class="muted">The branch contains runnable source scripts. This explorer is a map of their roles and links to the exact branch files; it does not run data downloads or model fits.</p></div>
<div class="card"><div class="bar"><label>Search <input id="search" placeholder="portfolio, frontier, stress…"></label><label>Category <select id="category"><option value="">All categories</option></select></label><span class="count" id="count"></span></div></div>
<div class="grid" id="scripts"></div>
<script>const D=PAYLOAD,$=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cats=[...new Set(D.scripts.map(s=>s.category))].sort();$('category').innerHTML+=[...cats].map(c=>`<option>${esc(c)}</option>`).join('');
$('pipeline').innerHTML=D.scripts.map((s,i)=>`<div class="step"><b>${i.toString().padStart(2,'0')}</b>${esc(s.title)}</div>`).join('');
function draw(){const q=$('search').value.toLowerCase(),c=$('category').value;const ss=D.scripts.filter(s=>(!c||s.category===c)&&(`${s.file} ${s.title} ${s.description}`.toLowerCase().includes(q)));$('count').textContent=`${ss.length} of ${D.scripts.length} scripts`;$('scripts').innerHTML=ss.map(s=>`<article class="script"><span class="tag">${esc(s.category)}</span><h3>${esc(s.file)}</h3><p><b>${esc(s.title)}</b></p><p>${esc(s.description)}</p><a href="${esc(s.url)}" target="_blank" rel="noopener">View source on GitHub ↗</a></article>`).join('')||'<p>No scripts match the filter.</p>'}['search','category'].forEach(id=>$(id).oninput=draw);draw();</script></html>'''
    page = page.replace('BRANCH', BRANCH).replace('REMOTE', REMOTE).replace('PAYLOAD', payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding='utf-8')
    return out


if __name__ == '__main__':
    print(build())
