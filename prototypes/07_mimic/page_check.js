// node page_check.js out/draw.html -> runs the page's own scripts against a stub DOM and prints JSON about what it rendered.
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
class El {
  constructor(tag) { this.tag = tag; this.children = []; this._html = ''; this.style = {}; this.dataset = {}; this.classList = { toggle() {}, add() {} }; this.value = ''; this.checked = false; this.listeners = {}; this.width = 640; this.height = 360; this.textContent = ''; this.complete = true; this.naturalWidth = 10; }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  set innerHTML(v) { this._html = v; if (v === '') this.children = []; }
  get innerHTML() { return this._html; }
  getContext() { return new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => { t[k] = v; return true; } }); }
  getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 360 }; }
  setPointerCapture() {}
}
const els = {};
const doc = { getElementById: id => els[id] || (els[id] = new El('div')), createElement: t => new El(t) };
els['clip'] = new El('select'); els['clip'].value = '-1';
class Img extends El { constructor() { super('img'); this.complete = true; this.naturalWidth = 10; } set src(v) { this._s = v; } get src() { return this._s; } }
const ctx = { document: doc, performance: { now: () => Date.now() }, Image: Img, console, Math, Float64Array, Set, Object, Array, JSON, window: {} };
vm.createContext(ctx);
scripts.forEach(s => vm.runInContext(s, ctx));
const grid = els['grid'];
const out = { cards: grid.children.length, first: grid.children[0] ? grid.children[0].children[1]._html.replace(/<[^>]+>/g, ' ').slice(0, 160) : null,
  status: els['status'].textContent };
// switch to every preset and a freehand path
const names = vm.runInContext('Object.keys(RAW.presets)', ctx), per = {};
for (const p of names) { vm.runInContext(`usePreset(${JSON.stringify(p)})`, ctx); per[p] = grid.children.length; }
vm.runInContext('pts=[[0.1,0.5],[0.3,0.5],[0.5,0.52],[0.7,0.5],[0.9,0.5]]; times=[]; presetSpeed=null; run()', ctx);
out.perPreset = per; out.freehand = grid.children.length;
console.log(JSON.stringify(out));
