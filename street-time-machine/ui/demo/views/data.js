/* Data: a query console over the people index. Rows are the real tracks of both scenes (lab, caviar). The browser
   runs the filter itself over the local index; the SQL block is that same filter written out. Every timing on the
   query plan is measured here with performance.now(). Rows open on the timeline, in Similar or in the Catalogue. */
(function () {
  "use strict";
  const { el, esc } = STM;

  const CSS = `
.v-data{container-type:inline-size}
.v-data .vd-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(128px,1fr));gap:12px;margin-bottom:16px}
.v-data .vd-stat{padding:12px 14px;min-width:0}
.v-data .vd-stat > span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-data .card-h{min-width:0;flex-wrap:wrap}
.v-data .vd-meta{font-size:12px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-data .vd-count{font:600 12px var(--f-mono);color:var(--ink-2)}
.v-data .vd-stat b{font-size:20px}
.v-data .vd-layout{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;align-items:start}
.v-data .vd-main,.v-data .vd-side{display:grid;grid-template-columns:minmax(0,1fr);gap:16px;min-width:0}
.v-data .vd-ask{display:flex;align-items:center;gap:8px;border:1px solid var(--line-2);border-radius:12px;padding:6px 6px 6px 12px;background:var(--card);transition:border-color .2s,box-shadow .2s}
.v-data .vd-ask:focus-within{border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}
.v-data .vd-ask input{flex:1;min-width:0;border:0;outline:0;background:transparent;font-size:14px;padding:6px 0}
.v-data .vd-ask svg{color:var(--ink-3);flex:none}
.v-data .vd-chips{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0 16px;align-items:center}
.v-data .vd-sub{display:flex;align-items:center;gap:8px;margin-bottom:8px}
.v-data .vd-sub .grow{flex:1}
.v-data .vd-sql{min-height:150px;margin:0}
.v-data .vd-caret{display:inline-block;width:7px;height:14px;background:var(--accent);vertical-align:-2px;margin-left:1px;animation:vd-blink 1s steps(1) infinite}
@keyframes vd-blink{50%{opacity:0}}
.v-data .vd-plan{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}
.v-data .vd-step{border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--sunk);display:grid;gap:2px;min-width:0;opacity:.55;transition:opacity .25s,border-color .25s,background .25s,transform .25s var(--ease)}
.v-data .vd-step.run{opacity:1;border-color:var(--accent-line);background:var(--accent-soft);transform:translateY(-2px)}
.v-data .vd-step.done{opacity:1;background:var(--card)}
.v-data .vd-step .row{gap:6px;font-weight:600;font-size:12.5px}
.v-data .vd-step .st{width:14px;height:14px;border-radius:50%;border:1.5px solid var(--line-2);display:grid;place-items:center;flex:none}
.v-data .vd-step.run .st{border-color:var(--accent-line);border-top-color:var(--accent);animation:spin .8s linear infinite}
.v-data .vd-step.done .st{background:var(--ok);border-color:var(--ok);color:var(--card)}
.v-data .vd-step small{font:11px/1.35 var(--f-mono);color:var(--ink-3);min-height:30px;overflow-wrap:anywhere}
.v-data .vd-step .ms{font:11.5px var(--f-mono);color:var(--ink-2)}
.v-data .vd-planfoot{display:flex;align-items:center;gap:10px;margin-top:10px;font-size:12px;color:var(--ink-3)}
.v-data .vd-planfoot .prog{flex:1}
.v-data .vd-lanes{display:grid;gap:7px;padding:14px 16px;border-bottom:1px solid var(--line)}
.v-data .vd-lane{display:grid;grid-template-columns:150px minmax(0,1fr) 44px;gap:10px;align-items:center;font-size:11.5px;color:var(--ink-2)}
.v-data .vd-lane > span:first-child{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-data .vd-lane .ax{font:11px var(--f-mono);color:var(--ink-3);text-align:right}
.v-data .vd-bar{position:relative;height:14px;border-radius:5px;background:var(--sunk);border:1px solid var(--line)}
.v-data .vd-bar i{position:absolute;top:2px;bottom:2px;border-radius:3px;background:var(--line-2);min-width:3px}
.v-data .vd-bar i.hit{background:var(--accent);cursor:pointer;transform:scaleX(0);transform-origin:left;transition:transform .6s var(--ease)}
.v-data .vd-bar i.hit.on{transform:scaleX(1)}
.v-data .vd-bar i.hit:hover{box-shadow:0 0 0 2px var(--accent-line)}
.v-data .vd-table td{padding:8px 10px}
.v-data .vd-table th{padding:8px 10px}
.v-data .vd-table tbody tr{cursor:pointer;animation:vd-rise .45s var(--ease) both}
@keyframes vd-rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.v-data .vd-table .crop{width:24px;height:32px;border-radius:5px}
.v-data .vd-id b{font:600 12.5px var(--f-mono);display:block}
.v-data .vd-id small,.v-data .vd-cam small{display:block;font-size:11px;color:var(--ink-3);white-space:nowrap}
.v-data .vd-cam b{font-weight:600;font-size:12.5px}
.v-data .vd-span{position:relative;height:6px;width:84px;border-radius:99px;background:var(--sunk);border:1px solid var(--line);overflow:hidden}
.v-data .vd-span i{position:absolute;top:0;bottom:0;background:var(--accent);border-radius:99px;transform:scaleX(0);transform-origin:left;transition:transform .7s var(--ease)}
.v-data .vd-span.on i{transform:scaleX(1)}
.v-data .vd-spanc{display:flex;align-items:center;gap:8px;white-space:nowrap}
.v-data .vd-look{display:flex;align-items:center;gap:6px;color:var(--ink-2);min-width:120px}
.v-data .vd-look .swatch{flex:none}
.v-data .vd-acts{display:flex;gap:2px;opacity:.5;transition:opacity .2s}
.v-data .vd-table tbody tr:hover .vd-acts{opacity:1}
.v-data .vd-ib{width:28px;height:28px;border-radius:8px;border:1px solid transparent;background:transparent;display:grid;place-items:center;color:var(--ink-2);transition:background .2s,color .2s,border-color .2s}
.v-data .vd-ib:hover{background:var(--card);border-color:var(--line);color:var(--accent)}
.v-data .vd-ib .ico{width:15px;height:15px}
.v-data .vd-tbl + .vd-tbl{border-top:1px solid var(--line)}
.v-data .vd-tbl > button{display:flex;align-items:center;gap:8px;width:100%;border:0;background:none;padding:10px 16px;text-align:left;transition:background .2s}
.v-data .vd-tbl > button:hover{background:var(--sunk)}
.v-data .vd-tbl > button b{font:500 12.5px var(--f-mono);flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-data .vd-tbl > button svg{color:var(--ink-3);transition:transform .25s var(--ease)}
.v-data .vd-tbl.open > button svg{transform:rotate(180deg)}
.v-data .vd-cols{display:none;padding:2px 16px 12px 16px}
.v-data .vd-tbl.open .vd-cols{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px 12px;animation:paneIn .3s var(--ease)}
.v-data .vd-cols span{font:11.5px var(--f-mono);color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-data .vd-cols em{font:11px var(--f-mono);color:var(--ink-3);font-style:normal;text-align:right}
.v-data .vd-note{padding:10px 16px;border-top:1px solid var(--line);font-size:12px;color:var(--ink-3);display:flex;gap:8px;align-items:flex-start}
.v-data .vd-hist{display:grid;padding:6px}
.v-data .vd-hist button{display:grid;gap:2px;text-align:left;border:0;background:none;padding:8px 10px;border-radius:9px;transition:background .2s;animation:vd-rise .35s var(--ease) both}
.v-data .vd-hist button:hover{background:var(--sunk)}
.v-data .vd-hist button span{font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-data .vd-hist button{min-width:0}
.v-data .vd-hist button small{font:11px var(--f-mono);color:var(--ink-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-data .vd-planfoot .mono{white-space:nowrap}
@container (max-width:980px){.v-data .vd-layout{grid-template-columns:minmax(0,1fr)}.v-data .vd-side{grid-template-columns:1fr 1fr;align-items:start}}
@container (max-width:720px){.v-data .vd-plan{grid-template-columns:repeat(3,minmax(0,1fr))}.v-data .vd-side{grid-template-columns:minmax(0,1fr)}.v-data .vd-lane{grid-template-columns:96px minmax(0,1fr) 40px}}
@media (max-width:820px){.v-data .vd-layout{grid-template-columns:minmax(0,1fr)}}
`;

  const PRESETS = [
    "orange or red top on any camera",
    "people seen for more than 20 s",
    "who was on camera B between 14:00:30 and 14:00:50",
    "grey top and grey trousers in the corridor",
    "same clothes as A-8 in the lab",
  ];
  const COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey"];
  const TOP_W = ["top", "shirt", "tee", "hoodie", "jumper", "sweater", "jacket", "coat", "blouse", "vest", "polo"];
  const BOT_W = ["trousers", "pants", "jeans", "shorts", "skirt", "leggings", "lower", "bottom", "bottoms", "legs"];
  const KW = /^(SELECT|FROM|WHERE|AND|OR|IN|NOT|ORDER|BY|LIMIT|ASC|DESC|TIME|BETWEEN|LIKE|TRUE)$/;

  const baseSec = s => { const [h, m, x] = String(s.clock0 || "T00:00:00").slice(11, 19).split(":").map(Number); return (h || 0) * 3600 + (m || 0) * 60 + (x || 0); };
  const pad = n => String(n).padStart(2, "0");
  const hms = sec => { sec = Math.floor(sec) % 86400; return `${pad(Math.floor(sec / 3600))}:${pad(Math.floor(sec / 60) % 60)}:${pad(sec % 60)}`; };
  const toSec = str => { const p = str.split(":").map(Number); return p[0] * 3600 + p[1] * 60 + (p[2] || 0); };
  const fmtMs = v => v < 0.1 ? "<0.1 ms" : v < 10 ? v.toFixed(1) + " ms" : Math.round(v) + " ms";
  const lit = v => `'${String(v).replace(/'/g, "''")}'`;
  const inList = (col, set) => set.size > 1 ? `${col} IN (${[...set].map(lit).join(", ")})` : `${col} = ${lit([...set][0])}`;
  /* count-up that never shows a negative first frame (rAF time can precede the call time) */
  function countUp(node, to, { ms = 900, dec = 0, suffix = "" } = {}, alive = () => true) {
    const t0 = performance.now(), fin = () => { node.textContent = to.toFixed(dec) + suffix; };
    if (STM.reduced) return fin();
    const step = now => { if (!alive()) return; const f = Math.min(1, Math.max(0, (now - t0) / ms)), e = 1 - Math.pow(1 - f, 3);
      node.textContent = (to * e).toFixed(dec) + suffix; if (f < 1) requestAnimationFrame(step); else fin(); };
    node.textContent = (0).toFixed(dec) + suffix; requestAnimationFrame(step);
  }
  const ico = (name, size = 15) => (STM.icons[name] || "").replace('class="ico"', `class="ico" width="${size}" height="${size}"`);

  /* ---------- the index as rows ---------- */
  function flatten(scenes) {
    const rows = [];
    for (const s of scenes) {
      const b0 = baseSec(s), idOf = {};
      (s.identities || []).forEach(i => i.tracks.forEach(t => { idOf[t] = i.id; }));
      for (const t of s.tracks) {
        const cam = s.cam(t.cam) || {}, look = t.look || {};
        rows.push({
          track_id: t.id, scene: s.name, cam: t.cam, cam_name: cam.name || t.cam, t0: t.t0, t1: t.t1, dur: t.t1 - t.t0,
          ts0: b0 + t.t0, ts1: b0 + t.t1, top: look.top || "unknown", bottom: look.bottom || "unknown", desc: t.desc || "",
          samples: (t.boxes || []).length, identity: idOf[t.id] || null, crop: s.cropUrl(t), duration: s.duration,
        });
      }
    }
    return rows;
  }

  /* ---------- plain words to a filter (colour words, camera letters and names, durations, clock times) ---------- */
  function parse(text, rows, scenes) {
    const raw = String(text || "").replace(/\s+/g, " ").trim(), q = raw.toLowerCase().replace(/gray/g, "grey");
    const words = q.split(/[^a-z0-9]+/).filter(Boolean), F = [], add = (sql, fn) => F.push({ sql, fn });
    let order = { sql: "scene DESC, t0 ASC", fn: (a, b) => a.scene === b.scene ? a.t0 - b.t0 : a.scene < b.scene ? 1 : -1 };
    const sceneWant = /\blab\b/.test(q) ? "lab" : /\b(caviar|mall|shopping)\b/.test(q) ? "caviar" : null;
    if (sceneWant) add(`scene = ${lit(sceneWant)}`, r => r.scene === sceneWant);
    const names = new Set();
    for (const s of scenes) for (const c of s.cameras) {
      const keys = c.name.toLowerCase().split(/\s+/).filter(w => w.length >= 4 && w !== "front" && w !== "aisle");
      if (keys.some(k => words.includes(k))) names.add(c.name);
    }
    if (names.size) add(inList("cam_name", names), r => names.has(r.cam_name));
    const cm = q.match(/\b(?:camera|cam)\s*([ab])\b/);
    if (cm) { const c = cm[1].toUpperCase(); add(`cam = ${lit(c)}`, r => r.cam === c); }
    const idm = q.match(/\b([ab])-(\d+)\b/), sim = /\b(like|similar|same)\b/.test(q);
    let ref = null;
    if (idm) {
      const id = idm[1].toUpperCase() + "-" + idm[2];
      ref = rows.find(r => r.track_id === id && (!sceneWant || r.scene === sceneWant)) || rows.find(r => r.track_id === id);
      if (ref && sim) {
        add(`top = ${lit(ref.top)}`, r => r.top === ref.top);
        add(`bottom = ${lit(ref.bottom)}`, r => r.bottom === ref.bottom);
        add(`NOT (scene = ${lit(ref.scene)} AND track_id = ${lit(id)})`, r => !(r.scene === ref.scene && r.track_id === id));
      } else add(`track_id = ${lit(id)}`, r => r.track_id === id);
    }
    const more = q.match(/(?:more than|longer than|over|at least|above|>)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b/);
    const less = q.match(/(?:less than|shorter than|under|below|<)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b/);
    if (more) { const v = +more[1]; add(`dur > ${v}`, r => r.dur > v); }
    if (less) { const v = +less[1]; add(`dur < ${v}`, r => r.dur < v); }
    if (more) order = { sql: "dur DESC", fn: (a, b) => b.dur - a.dur };
    else if (less) order = { sql: "dur ASC", fn: (a, b) => a.dur - b.dur };
    const tm = q.match(/\b\d{1,2}:\d{2}(?::\d{2})?\b/g);
    if (tm && tm.length >= 2) {
      let a = toSec(tm[0]), b = toSec(tm[1]); if (a > b) [a, b] = [b, a];
      add(`ts1 >= TIME ${lit(hms(a))} AND ts0 <= TIME ${lit(hms(b))}`, r => r.ts1 >= a && r.ts0 <= b);
    } else if (tm) { const a = toSec(tm[0]); add(`TIME ${lit(hms(a))} BETWEEN ts0 AND ts1`, r => r.ts0 <= a && r.ts1 >= a); }
    if (!(ref && sim)) {
      const top = new Set(), bot = new Set(), any = new Set();
      words.forEach((w, i) => {
        if (!COLOURS.includes(w)) return;
        const ahead = words.slice(i + 1, i + 5), part = ahead.find(x => TOP_W.includes(x) || BOT_W.includes(x));
        (part ? (TOP_W.includes(part) ? top : bot) : any).add(w);
      });
      if (top.size) add(inList("top", top), r => top.has(r.top));
      if (bot.size) add(inList("bottom", bot), r => bot.has(r.bottom));
      if (any.size) add(`(${inList("top", any)} OR ${inList("bottom", any)})`, r => any.has(r.top) || any.has(r.bottom));
    }
    if (!F.length) {
      const vocab = new Set(rows.flatMap(r => r.desc.split(/[^a-z]+/))), kw = words.filter(w => w.length >= 4 && vocab.has(w));
      if (kw.length) add(kw.length > 1 ? `(${kw.map(w => `desc LIKE ${lit("%" + w + "%")}`).join(" OR ")})` : `desc LIKE ${lit("%" + kw[0] + "%")}`, r => kw.some(w => r.desc.includes(w)));
    }
    const where = F.length ? F.map(f => f.sql).join("\n  AND ") : "TRUE";
    const sql = `-- ${raw.slice(0, 72) || "every track"}\nSELECT track_id, scene, cam, t0, t1, top, bottom\nFROM stm.tracks\nWHERE ${where}\nORDER BY ${order.sql}\nLIMIT 50;`;
    return { raw, filters: F, order, sql };
  }

  function tokens(sql) {
    const out = [], re = /(--[^\n]*)|('(?:[^']|'')*')|(\b\d+(?:\.\d+)?\b)|([A-Za-z_][A-Za-z_0-9.]*)|(\s+|.)/g; let m;
    while ((m = re.exec(sql))) out.push([m[1] ? "c" : m[2] ? "s" : m[3] ? "n" : m[4] && KW.test(m[4]) ? "k" : "", m[0]]);
    return out;
  }
  function paint(node, toks, n, total) {
    let html = "", left = n;
    for (const [c, t] of toks) { if (left <= 0) break; const s = t.slice(0, left); left -= s.length; html += c ? `<span class="${c}">${esc(s)}</span>` : esc(s); }
    node.innerHTML = html + (n < total ? '<span class="vd-caret"></span>' : "");
  }

  STM.view("data", {
    label: "Data",
    async mount(root, ctx) {
      let dead = false, raf = 0, runId = 0, ready = false, ROWS = [], scenes = [], byScene = {};
      const last = { rows: [] };
      const timers = new Set();
      const later = (fn, ms) => { const t = setTimeout(() => { timers.delete(t); if (!dead) fn(); }, ms); timers.add(t); };
      const wait = ms => new Promise(r => later(r, ms));
      const wrap = el("div", { class: "v-data" });
      root.append(el("style", {}, CSS), wrap);

      STM.pageHead(wrap, "Data", "Query the people index in plain words.");

      /* stats strip, skeleton first */
      const stats = el("div", { class: "vd-stats stagger" });
      const STAT = [["Person tracks", "tracks"], ["Box samples", "samples"], ["Cameras", "cams"], ["Footage indexed", "footage"], ["Index build", "build"]];
      const statNodes = {};
      STAT.forEach(([label, k]) => { const b = el("b", {}, STM.skeleton(22, "60%", 6)); statNodes[k] = b; stats.append(el("div", { class: "card stat vd-stat" }, b, el("span", {}, label))); });
      wrap.append(stats);

      const layout = el("div", { class: "vd-layout" }), main = el("div", { class: "vd-main" }), side = el("div", { class: "vd-side" });
      layout.append(main, side); wrap.append(layout);

      /* ask card */
      const input = el("input", { id: "data-query", type: "text", value: PRESETS[0], "aria-label": "Ask the index", placeholder: "e.g. blue top on camera A", autocomplete: "off" });
      const runBtn = el("button", { class: "btn btn-primary btn-sm", id: "data-run", type: "button", onclick: () => run(input.value) }, el("span", { html: STM.icons.play }), "Run");
      input.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); run(input.value); } });
      input.addEventListener("input", () => chips.forEach(c => c.classList.remove("on")));
      const chips = PRESETS.map((p, i) => el("button", { class: "chip", type: "button", id: "data-preset-" + i, onclick: () => { input.value = p; run(p); } }, p));
      const sqlBox = el("pre", { class: "code vd-sql", id: "data-sql", "aria-label": "Generated SQL" });
      const copyBtn = el("button", { class: "btn btn-ghost btn-sm", type: "button", id: "data-copy-sql", onclick: () => {
        const txt = sqlBox.textContent; if (!txt) return;
        Promise.resolve(navigator.clipboard && navigator.clipboard.writeText(txt)).then(() => STM.toast("SQL copied")).catch(() => STM.toast("Copy is blocked in this window"));
      } }, "Copy SQL");
      const STEPS = [["parse", "Parse"], ["plan", "Plan"], ["scan", "Scan"], ["filter", "Filter"], ["sort", "Sort"], ["ret", "Return"]];
      const plan = el("div", { class: "vd-plan", id: "data-plan" });
      const stepNodes = STEPS.map(([k, label]) => {
        const n = el("div", { class: "vd-step" }, el("div", { class: "row" }, el("span", { class: "st" }), label), el("small", {}, "waiting"), el("span", { class: "ms" }, "--"));
        plan.append(n); return n;
      });
      const planProg = el("i"), planNote = el("span", { class: "mono" }, "Not run yet");
      main.append(el("div", { class: "card pop" },
        el("div", { class: "card-h" }, el("h3", {}, "Ask the index")),
        el("div", { class: "card-b" },
          el("div", { class: "vd-ask" }, el("span", { html: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>' }), input, runBtn),
          el("div", { class: "vd-chips" }, el("span", { class: "eyebrow" }, "Try"), chips),
          el("div", { class: "vd-sub" }, el("span", { class: "eyebrow" }, "Generated SQL"), el("div", { class: "grow" }), copyBtn), sqlBox,
          el("div", { class: "vd-sub", style: { marginTop: "16px" } }, el("span", { class: "eyebrow" }, "Query plan")),
          plan, el("div", { class: "vd-planfoot" }, el("div", { class: "prog" }, planProg), planNote))));

      /* results card */
      const resCount = el("span", { class: "vd-count" }, "0 rows"), resMeta = el("span", { class: "muted mono vd-meta" }, "");
      const lanes = el("div", { class: "vd-lanes", id: "data-coverage" });
      const resBody = el("div", { class: "scroll-x" });
      const catBtn = el("button", { class: "btn btn-sm", type: "button", id: "data-open-catalogue", onclick: () => {
        const rows = last.rows; if (!rows.length) return;
        STM.state.dataSelection = rows.map(r => ({ scene: r.scene, track: r.track_id }));
        focusScene(rows[0]); STM.go("catalogue", { tracks: rows.filter(r => r.scene === rows[0].scene).map(r => r.track_id), from: "data" });
      } }, el("span", { html: ico("catalogue", 14) }), "Open in Catalogue");
      main.append(el("div", { class: "card pop", id: "data-results", style: { animationDelay: ".08s" } },
        el("div", { class: "card-h" }, el("h3", {}, "Results"), resCount, resMeta, el("div", { class: "grow" }), catBtn), lanes, resBody));

      /* side: tables + history */
      const tablesBody = el("div", {}, STM.skeleton(14, "70%"), STM.skeleton(14, "50%"));
      tablesBody.style.cssText = "display:grid;gap:10px;padding:14px 16px";
      side.append(el("div", { class: "card pop", id: "data-tables", style: { animationDelay: ".12s" } },
        el("div", { class: "card-h" }, el("h3", {}, "Tables")),
        tablesBody));
      const histList = el("div", { class: "vd-hist", id: "data-history" });
      side.append(el("div", { class: "card pop", style: { animationDelay: ".16s" } },
        el("div", { class: "card-h" }, el("h3", {}, "Query history")), histList));

      /* load both real scenes */
      scenes = (await Promise.all(["lab", "caviar"].map(n => STM.loadScene(n).catch(() => null)))).filter(Boolean);
      if (dead) return () => {};
      ROWS = flatten(scenes); byScene = Object.fromEntries(scenes.map(s => [s.name, s])); ready = true;
      const nIdent = scenes.reduce((a, s) => a + (s.identities || []).length, 0), nCams = scenes.reduce((a, s) => a + s.cameras.length, 0);
      const nItems = scenes.reduce((a, s) => a + (s.items || []).length, 0), nSamples = ROWS.reduce((a, r) => a + r.samples, 0);
      const footage = scenes.reduce((a, s) => a + s.duration, 0), build = scenes.reduce((a, s) => a + ((s.build || {}).seconds || 0), 0);

      later(() => {
        const put = (k, v, o) => { statNodes[k].innerHTML = ""; countUp(statNodes[k], v, o, () => !dead); };
        put("tracks", ROWS.length); put("samples", nSamples); put("cams", nCams);
        put("footage", footage, { dec: 1, suffix: " s" }); put("build", build, { dec: 1, suffix: " s" });
      }, 180);

      /* schema browser */
      const SCHEMA = [
        { name: "stm.tracks", count: ROWS.length, cols: [["track_id", "text"], ["scene", "text"], ["cam", "text"], ["cam_name", "text"], ["t0", "float"], ["t1", "float"], ["dur", "float"], ["ts0", "time"], ["ts1", "time"], ["top", "text"], ["bottom", "text"], ["desc", "text"], ["samples", "int"], ["boxes", "list<t,x,y,w,h>"], ["identity", "text"]] },
        { name: "stm.identities", count: nIdent, cols: [["identity", "text"], ["scene", "text"], ["tracks", "list<text>"], ["hops", "int"], ["hop_score", "float"]] },
        { name: "stm.cameras", count: nCams, cols: [["cam", "text"], ["scene", "text"], ["name", "text"], ["links", "list<text>"], ["width", "int"], ["height", "int"], ["footage_s", "float"], ["samples", "int"]] },
        { name: "stm.items", count: nItems, cols: [["item_id", "text"], ["scene", "text"], ["cam", "text"], ["t", "float"], ["kind", "text"]] },
      ];
      later(() => {
        tablesBody.innerHTML = ""; tablesBody.style.cssText = "";
        SCHEMA.forEach((t, i) => {
          const cnt = el("span", { class: "muted mono", style: { fontSize: "11.5px" } }, "0");
          const box = el("div", { class: "vd-tbl" + (i === 0 ? " open" : ""), id: "data-table-" + t.name.split(".")[1] },
            el("button", { type: "button", onclick: () => box.classList.toggle("open") }, el("b", {}, t.name), cnt, el("span", { class: "muted", style: { fontSize: "11px" } }, "rows"), el("span", { html: STM.icons.chev })),
            el("div", { class: "vd-cols" }, t.cols.flatMap(([c, ty]) => [el("span", {}, c), el("em", {}, ty)])));
          tablesBody.append(box);
          later(() => countUp(cnt, t.count, { ms: 700 }, () => !dead), 60 * i);
        });
      }, 320);

      /* history */
      const H = STM.state.dataHistory = STM.state.dataHistory || [];
      const renderHist = () => {
        histList.innerHTML = "";
        if (!H.length) { histList.append(el("div", { class: "empty" }, "Runs appear here.")); return; }
        H.slice(0, 7).forEach((h, i) => histList.append(el("button", { type: "button", id: "data-history-" + i, title: h.q, onclick: () => { input.value = h.q; run(h.q); } },
          el("span", {}, h.q), el("small", {}, `${h.rows} rows · ${fmtMs(h.ms)} · ${h.at}`))));
      };
      renderHist();

      /* navigation out to the other views */
      function focusScene(r) {
        if (r.scene !== STM.sceneName && byScene[r.scene]) {
          STM.sceneName = r.scene;
          if (STM.desk && STM.desk.reset) try { STM.desk.reset(); } catch (e) { /* desk not mounted */ }
          STM.toast(`Switched to the ${esc(byScene[r.scene].title)} scene`);
        }
        STM.state.focus = { scene: r.scene, track: r.track_id, t: r.t0 };
      }
      const openIn = (view, r) => { focusScene(r); STM.go(view, { track: r.track_id, t: r.t0, from: "data" }); };

      /* coverage lanes: every track faint, matches in accent */
      function renderLanes(hits) {
        lanes.innerHTML = "";
        const hitKey = new Set(hits.map(r => r.scene + "/" + r.track_id));
        let k = 0;
        for (const s of scenes) for (const c of s.cameras) {
          const bar = el("div", { class: "vd-bar" });
          ROWS.filter(r => r.scene === s.name && r.cam === c.id).forEach(r => {
            const hit = hitKey.has(r.scene + "/" + r.track_id);
            const i = el("i", { class: hit ? "hit" : null, title: `${r.track_id} · ${hms(r.ts0)} to ${hms(r.ts1)}`, style: { left: (r.t0 / s.duration * 100) + "%", width: Math.max(0.6, r.dur / s.duration * 100) + "%" } });
            if (hit) { i.addEventListener("click", () => openIn("timeline", r)); later(() => i.classList.add("on"), 80 + 50 * k++); }
            bar.append(i);
          });
          lanes.append(el("div", { class: "vd-lane" }, el("span", {}, el("b", {}, `${s.name} · ${c.id}`), ` ${c.name}`), bar, el("span", { class: "ax" }, Math.round(s.duration) + " s")));
        }
      }
      function skeletonTable() {
        resBody.innerHTML = "";
        const box = el("div", { style: { display: "grid", gap: "12px", padding: "16px" } });
        for (let i = 0; i < 5; i++) box.append(el("div", { class: "row" }, STM.skeleton(32, "24px", 5), STM.skeleton(12, "14%"), STM.skeleton(12, "18%"), STM.skeleton(12, "22%"), STM.skeleton(6, "16%", 99), STM.skeleton(12, "20%")));
        resBody.append(box);
      }
      function buildTable(out) {
        if (!out.length) return el("div", { class: "empty" }, "No rows match. Try a colour word, a camera letter or a clock time.");
        const tb = el("tbody");
        out.forEach((r, i) => {
          const span = el("div", { class: "vd-span" }, el("i", { style: { left: (r.t0 / r.duration * 100) + "%", width: Math.max(2, r.dur / r.duration * 100) + "%" } }));
          const ib = (view, icon, label) => el("button", { class: "vd-ib", type: "button", title: label, "aria-label": label, onclick: e => { e.stopPropagation(); openIn(view, r); } }, el("span", { html: ico(icon) }));
          const tr = el("tr", { id: "data-row-" + i, title: "Open on the timeline", style: { animationDelay: (40 * Math.min(i, 14)) + "ms" }, onclick: () => openIn("timeline", r) },
            el("td", {}, el("img", { class: "crop", src: r.crop, alt: "Crop of " + r.track_id, loading: "lazy", width: 24, height: 32 })),
            el("td", { class: "vd-id" }, el("b", {}, r.track_id), el("small", {}, `${r.scene}${r.identity ? " · " + r.identity : ""} · ${r.samples} samples`)),
            el("td", { class: "vd-cam" }, el("b", {}, "Cam " + r.cam), el("small", {}, r.cam_name)),
            el("td", { class: "mono", style: { whiteSpace: "nowrap", fontSize: "12px" } }, `${hms(r.ts0)} – ${hms(r.ts1)}`),
            el("td", {}, el("div", { class: "vd-spanc" }, span, el("span", { class: "mono", style: { fontSize: "12px" } }, r.dur.toFixed(1) + " s"))),
            el("td", {}, el("div", { class: "vd-look" }, el("span", { class: "swatch", title: "top " + r.top, style: { background: STM.colourHex[r.top] || "transparent" } }),
              el("span", { class: "swatch", title: "lower " + r.bottom, style: { background: STM.colourHex[r.bottom] || "transparent" } }), el("span", {}, r.desc))),
            el("td", {}, el("div", { class: "vd-acts" }, ib("timeline", "timeline", "Open on the timeline"), ib("similar", "similar", "Find similar people"))));
          tb.append(tr);
          later(() => span.classList.add("on"), 200 + 40 * Math.min(i, 14));
        });
        return el("table", { class: "table vd-table" },
          el("thead", {}, el("tr", {}, ["", "Track", "Camera", "Seen", "On screen", "Clothing", ""].map(h => el("th", {}, h)))), tb);
      }

      /* the run: type the SQL, walk the plan, reveal the rows */
      async function run(text) {
        if (!ready) return;
        const my = ++runId, alive = () => !dead && my === runId;
        text = String(text || "").trim() || PRESETS[0];
        chips.forEach((c, i) => c.classList.toggle("on", PRESETS[i] === text));
        runBtn.innerHTML = '<span class="spin" style="width:14px;height:14px;border-color:rgba(255,255,255,.35);border-top-color:#fff"></span>Running';
        stepNodes.forEach(n => { n.className = "vd-step"; n.querySelector("small").textContent = "waiting"; n.querySelector(".ms").textContent = "--"; });
        planProg.style.width = "0%"; planNote.textContent = "Writing SQL";
        resCount.textContent = "running"; resMeta.textContent = ""; skeletonTable(); renderLanes([]);
        const T = {}; let t = performance.now();
        const parsed = parse(text, ROWS, scenes); T.parse = performance.now() - t;
        t = performance.now(); const toks = tokens(parsed.sql), total = parsed.sql.length; T.plan = performance.now() - t;
        if (STM.reduced) paint(sqlBox, toks, total, total);
        else await new Promise(res => {
          let n = 0;
          const step = () => { if (!alive()) return res(); n = Math.min(total, n + 4); paint(sqlBox, toks, n, total); if (n < total) raf = requestAnimationFrame(step); else res(); };
          raf = requestAnimationFrame(step);
        });
        if (!alive()) return;
        /* execute and measure each stage for real */
        t = performance.now(); const scanned = ROWS.map(r => Object.assign({}, r)); T.scan = performance.now() - t;
        t = performance.now(); const kept = scanned.filter(r => parsed.filters.every(f => f.fn(r))); T.filter = performance.now() - t;
        t = performance.now(); kept.sort(parsed.order.fn); T.sort = performance.now() - t;
        t = performance.now(); const out = kept.slice(0, 50), table = buildTable(out); T.ret = performance.now() - t;
        const engine = Object.values(T).reduce((a, b) => a + b, 0);
        const detail = { parse: `${parsed.raw.split(" ").length} words`, plan: `${parsed.filters.length} ${parsed.filters.length === 1 ? "predicate" : "predicates"}`, scan: `stm.tracks · ${ROWS.length} rows`,
          filter: `${kept.length} of ${ROWS.length} kept`, sort: parsed.order.sql.split(",")[0], ret: `${out.length} rows` };
        planNote.textContent = "Walking the plan";
        for (let i = 0; i < STEPS.length; i++) {
          const n = stepNodes[i], k = STEPS[i][0];
          n.className = "vd-step run"; n.querySelector("small").textContent = detail[k];
          await wait(STM.reduced ? 0 : 115); if (!alive()) return;
          n.className = "vd-step done"; n.querySelector(".st").innerHTML = STM.icons.check; n.querySelector(".ms").textContent = fmtMs(T[k]);
          planProg.style.width = ((i + 1) / STEPS.length * 100) + "%";
        }
        planNote.textContent = `${fmtMs(engine)} engine time`;
        resBody.innerHTML = ""; resBody.append(table); renderLanes(out);
        resCount.textContent = `${out.length} ${out.length === 1 ? "row" : "rows"}`; resMeta.textContent = `${fmtMs(engine)} · ${parsed.filters.length ? "filtered" : "no filter words found, every row"}`;
        catBtn.disabled = !out.length;
        last.rows = out;
        const now = new Date();
        H.unshift({ q: text, rows: out.length, ms: engine, at: `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}` });
        if (H.length > 20) H.length = 20;
        renderHist();
        runBtn.innerHTML = STM.icons.play + "Run";
        STM.emit("data", { kind: "results", query: text, rows: out.map(r => ({ scene: r.scene, track: r.track_id })) });
      }

      const first = (ctx.opts && ctx.opts.query) || PRESETS[0];
      input.value = first;
      later(() => run(first), 260);

      return () => { dead = true; runId++; cancelAnimationFrame(raf); timers.forEach(clearTimeout); timers.clear(); };
    },
  });
})();
