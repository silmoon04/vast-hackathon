// node parity_check.js input.json -> prints JSON: for every query the top-k [{i, cost}] computed by mimic_search.js
const fs = require('fs');
const S = require('./mimic_search.js');
const inp = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const D = S.prepare(inp.data);
const out = inp.queries.map(q => {
  q.speed = q.speed ? Float64Array.from(q.speed) : null;
  const r = S.search(D, q, i => inp.allow ? inp.allow[i] : true, 8);
  return r.map(x => ({ i: x.i, cost: x.cost, rot: x.rot }));
});
console.log(JSON.stringify(out));
