// Exercise embedded dashboard code and selectors without a browser or model fits.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const path = require('path');
const page = fs.readFileSync(path.join(__dirname, '../outputs/dashboard_plotly.html'), 'utf8');
const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {
    value: '', textContent: '', _html: '',
    set innerHTML(v) {
      this._html = v;
      const match = v.match(/<option(?: value="([^"]*)")?>([^<]*)/);
      if (match) this.value = match[1] ?? match[2];
    },
    get innerHTML() { return this._html; },
  });
  return nodes.get(id);
}
let calls = 0;
const context = vm.createContext({document: {getElementById: node}, Plotly: {
  react(id, traces) {
    assert(Array.isArray(traces), id);
    for (const trace of traces) {
      if (trace.x && trace.y && trace.type !== 'heatmap') assert.equal(trace.x.length, trace.y.length, id);
    }
    calls++;
  }, purge() {},
}});
vm.runInContext(page.match(/<script>\s*([\s\S]*?)<\/script>/)[1], context);
for (const stage of ['base', 'iteration_1', 'iteration_4', 'iteration_4.2', 'iteration_4.3', 'iteration_10']) {
  node('stage').value = stage;
  vm.runInContext('refresh()', context);
  for (const kind of ['scatter', 'bar', 'histogram', 'heatmap', 'box']) {
    node('kind').value = kind;
    vm.runInContext('diagnostic()', context);
  }
}
for (const market of ['us_mid', 'eu_large']) {
  node('market').value = market;
  node('stage').value = 'base';
  vm.runInContext('refresh()', context);
}
assert(calls > 60);
console.log(`Dashboard JavaScript verified: ${calls} Plotly render calls across stages, markets and chart types.`);
