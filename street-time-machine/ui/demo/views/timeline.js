/* Timeline view: multi-camera swimlanes of the active scene. Bars, hops, events and density are all computed
   from the real index tracks in scene.json; nothing here is simulated. */
(function () {
  "use strict";
  const S = window.STM;
  const PAD = 14, BAR = 20, GAP = 6, LP = 10, STEP = 0.5;
  const pos = f => `calc(${PAD}px + (100% - ${2 * PAD}px) * ${f})`;
  const wpos = f => `calc((100% - ${2 * PAD}px) * ${f})`;
  const hex = c => S.colourHex[c] || S.colourHex.grey;
  const lum = h => { const n = parseInt(h.slice(1), 16); return (0.299 * (n >> 16 & 255) + 0.587 * (n >> 8 & 255) + 0.114 * (n & 255)) / 255; };
  const textOn = h => lum(h) > 0.6 ? S.colourHex.black : S.colourHex.white;
  const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : "";
  const fmtS = v => (v >= 10 ? Math.round(v) : Math.round(v * 10) / 10) + " s";
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const CSS = `
.v-timeline{container-type:inline-size;display:grid;gap:16px}
.v-timeline .view-title{margin-bottom:0}
.v-timeline .tl-scenes{gap:6px;flex-wrap:wrap}
.v-timeline .tl-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.v-timeline .tl-stat{padding:12px 16px;display:grid;gap:4px;min-width:0}
.v-timeline .tl-stat b{font-size:22px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums;min-height:28px;display:flex;align-items:center}
.v-timeline .tl-stat small{font-size:12px;color:var(--ink-3);overflow-wrap:anywhere}
.v-timeline .tl-main{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;align-items:start}
.v-timeline .tl-bottom{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:16px;align-items:start}
.v-timeline .card-h{flex-wrap:wrap}
.v-timeline .tl-clock{font-size:13px;padding:5px 9px;border-radius:8px;background:var(--sunk);border:1px solid var(--line);color:var(--ink)}
.v-timeline .tl-tools{margin-bottom:12px;font-size:12.5px;flex-wrap:wrap}
.v-timeline .tl-tools input[type=range]{width:140px;accent-color:var(--accent)}
.v-timeline .tl-key{display:inline-block;width:14px;height:10px;border-radius:3px;background:color-mix(in srgb,var(--accent) 14%,transparent);border:1px dashed var(--accent-line);vertical-align:-1px}
.v-timeline .tl{position:relative;display:grid;grid-template-columns:150px minmax(0,1fr);border:1px solid var(--line);border-radius:var(--r);background:var(--card)}
.v-timeline .tl-labels{border-right:1px solid var(--line);background:var(--sunk);border-radius:var(--r) 0 0 var(--r);min-width:0}
.v-timeline .tl-axis-sp{height:30px;border-bottom:1px solid var(--line);padding:8px 12px}
.v-timeline .tl-lab{display:flex;flex-direction:column;justify-content:center;align-items:flex-start;gap:2px;width:100%;border:0;border-bottom:1px solid var(--line);background:transparent;padding:0 12px;text-align:left;transition:background .2s;position:relative}
.v-timeline .tl-lab:last-child{border-bottom:0}
.v-timeline .tl-lab:hover{background:var(--card)}
.v-timeline .tl-lab.on::before{content:"";position:absolute;left:0;top:10px;bottom:10px;width:3px;border-radius:0 3px 3px 0;background:var(--accent)}
.v-timeline .tl-lab{min-width:0;overflow:hidden}
.v-timeline .tl-lab b,.v-timeline .tl-lab span{max-width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-timeline .tl-lab b{font-size:13px;font-weight:600}
.v-timeline .tl-lab span{font-size:11.5px}
.v-timeline .tl-scroll{overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;border-radius:0 var(--r) var(--r) 0}
.v-timeline .tl-inner{position:relative;min-width:100%;cursor:crosshair;user-select:none;-webkit-user-select:none;touch-action:pan-y}
.v-timeline .tl-axis{position:relative;height:30px;border-bottom:1px solid var(--line);z-index:1}
.v-timeline .tl-tick{position:absolute;top:5px;transform:translateX(-50%);font:11px var(--f-mono);color:var(--ink-3);white-space:nowrap}
@keyframes tlFade{from{opacity:0}to{opacity:1}}
.v-timeline .tl-tick.first{transform:translateX(-2px)}
.v-timeline .tl-tick.last{transform:translateX(calc(-100% + 2px))}
.v-timeline .tl-gridl{position:absolute;inset:30px 0 0 0;pointer-events:none}
.v-timeline .tl-gridl i{position:absolute;top:0;bottom:0;width:1px;background:var(--line)}
.v-timeline .tl-lane{position:relative;border-bottom:1px solid var(--line);z-index:1}
.v-timeline .tl-lane.last{border-bottom:0}
.v-timeline .tl-sk{position:absolute;height:20px}
.v-timeline .tl-bar{position:absolute;display:block;height:20px;min-width:6px;border:0;border-radius:6px;padding:0 7px 3px;font:500 10.5px/17px var(--f-mono);text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transform-origin:left center;cursor:pointer;
  box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--ink) 14%,transparent),inset 0 -4px 0 var(--bb);transition:transform .2s var(--ease),box-shadow .2s,opacity .25s,filter .25s;animation:tlBarIn .55s var(--ease) backwards}
.v-timeline .tl-bar:hover{transform:translateY(-1px);z-index:3;box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--ink) 14%,transparent),inset 0 -4px 0 var(--bb),0 6px 14px color-mix(in srgb,var(--ink) 18%,transparent)}
.v-timeline .tl-bar.tiny{color:transparent!important}
.v-timeline .tl.has-sel .tl-bar:not(.on){opacity:.28;filter:saturate(.4)}
.v-timeline .tl.has-sel .tl-bar:not(.on):hover{opacity:.7}
.v-timeline .tl-bar.on{outline:2px solid var(--accent);outline-offset:2px;z-index:2}
@keyframes tlBarIn{from{transform:scaleX(0);opacity:0}to{transform:scaleX(1);opacity:1}}
.v-timeline .tl-band{position:absolute;top:30px;bottom:0;background:color-mix(in srgb,var(--accent) 5%,transparent);border-left:1px dashed var(--accent-line);border-right:1px dashed var(--accent-line);pointer-events:none;transition:background .4s}
.v-timeline .tl-band.hot{background:color-mix(in srgb,var(--accent) 16%,transparent)}
.v-timeline .tl-links{position:absolute;left:0;top:0;pointer-events:none;z-index:2;overflow:visible}
.v-timeline .tl-link{fill:none;stroke:var(--accent);stroke-width:1.5;stroke-dasharray:5 4;opacity:.45;animation:tlDash 1s linear infinite}
.v-timeline .tl-link.on{stroke-width:2.2;opacity:1}
.v-timeline .tl-dot{fill:var(--card);stroke:var(--accent);stroke-width:2}
@keyframes tlDash{to{stroke-dashoffset:-18}}
.v-timeline .tl-hopl{position:absolute;inset:0;pointer-events:none;z-index:3}
.v-timeline .tl-hop{position:absolute;transform:translate(-50%,-50%);font-size:10.5px;font-weight:600;color:var(--accent);background:var(--card);border:1px solid var(--accent-line);border-radius:99px;padding:1px 7px;white-space:nowrap;opacity:.75;transition:opacity .2s}
.v-timeline .tl-hop.on{opacity:1;box-shadow:var(--shadow)}
.v-timeline .tl-ph{position:absolute;top:0;bottom:0;width:0;z-index:4;pointer-events:none}
.v-timeline .tl-ph::before{content:"";position:absolute;top:30px;bottom:0;left:-1px;width:2px;background:var(--accent);border-radius:2px}
.v-timeline .tl-ph i{position:absolute;top:25px;left:-5px;width:10px;height:10px;border-radius:50%;background:var(--accent);transition:transform .15s}
.v-timeline .tl.scrub .tl-ph i{transform:scale(1.3)}
.v-timeline .tl-tip{position:absolute;left:0;top:0;z-index:10;display:grid;grid-template-columns:64px 1fr;gap:10px;width:268px;padding:10px;background:var(--card);border:1px solid var(--line);border-radius:var(--r);box-shadow:var(--shadow-lg);pointer-events:none;opacity:0;transform:translateY(4px) scale(.98);transition:opacity .15s,transform .15s var(--ease);font-size:12px;color:var(--ink-2)}
.v-timeline .tl-tip.show{opacity:1;transform:none}
.v-timeline .tl-tip img{width:64px;height:86px}
.v-timeline .tl-tip b{display:block;font-size:13px;color:var(--ink);margin-bottom:2px}
.v-timeline .tl-tip .row{gap:6px;margin-top:6px}
.v-timeline .tl-sel:empty{display:none}
.v-timeline .tl-sel{display:flex;align-items:center;gap:12px;margin-top:12px;padding:10px 12px;border:1px solid var(--line);border-radius:var(--r);background:var(--sunk);min-height:66px;flex-wrap:wrap;font-size:12.5px;transition:background .3s,border-color .3s}
.v-timeline .tl-sel.has{border-color:var(--accent-line);background:var(--accent-soft)}
.v-timeline .tl-sel img{width:40px;height:54px;flex:none}
.v-timeline .tl-sel-m{flex:1 1 220px;min-width:0;display:grid;gap:3px}
.v-timeline .tl-sel-m .row{gap:8px;flex-wrap:wrap}
.v-timeline .tl-sel-m > div{overflow-wrap:anywhere}
.v-timeline .tl-frame{position:relative;border-radius:var(--r);overflow:hidden;background:var(--video-bg);width:100%}
.v-timeline .tl-frame video,.v-timeline .tl-frame canvas{position:absolute;inset:0;width:100%;height:100%;display:block;object-fit:fill}
.v-timeline .tl-vstate{position:absolute;inset:0;display:grid;place-items:center;font-size:12px;color:var(--ink-3);text-align:center;padding:12px}
.v-timeline .tl-vlabel{position:absolute;left:8px;top:8px;font-size:10.5px;padding:2px 7px;border-radius:6px;background:color-mix(in srgb,var(--video-bg) 72%,transparent)}
.v-timeline .tl-pv-meta{display:grid;gap:10px;margin-top:12px}
.v-timeline .tl-pv-meta .row b{font-size:13px}
.v-timeline .tl-inview{display:flex;flex-wrap:wrap;gap:6px;min-height:26px}
.v-timeline .tl-evs{position:relative;max-height:392px;overflow:auto;padding:6px;display:grid;gap:2px}
.v-timeline .tl-ev{display:grid;grid-template-columns:64px 24px minmax(0,1fr);align-items:center;gap:10px;width:100%;padding:7px 8px;border:0;border-radius:10px;background:transparent;text-align:left;transition:background .15s;position:relative}
.v-timeline .tl-ev:hover{background:var(--sunk)}
.v-timeline .tl-ev.now{background:var(--accent-soft)}
.v-timeline .tl-ev.now::before{content:"";position:absolute;left:0;top:8px;bottom:8px;width:3px;border-radius:3px;background:var(--accent)}
.v-timeline .tl-ev.off{display:none}
.v-timeline .tl-ev-t{font:12px var(--f-mono);color:var(--accent)}
.v-timeline .tl-ev img{width:24px;height:32px}
.v-timeline .tl-ev-ico{width:24px;height:32px;border-radius:5px;background:var(--sunk);border:1px dashed var(--accent-line)}
.v-timeline .tl-ev-m{display:grid;min-width:0;font-size:12.5px}
.v-timeline .tl-ev-m b,.v-timeline .tl-ev-m span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-timeline .tl-ev-m b{font-weight:500}
.v-timeline .tl-ev-m span{font-size:11.5px}
.v-timeline .d-read{font-size:12px;font-variant-numeric:tabular-nums;color:var(--ink-2);min-height:18px;margin-bottom:6px}
.v-timeline .d-box{position:relative;height:150px;cursor:crosshair}
.v-timeline .d-clip{position:absolute;inset:0;clip-path:inset(0 100% 0 0);transition:clip-path 1.1s var(--ease)}
.v-timeline .d-clip.in{clip-path:inset(0 0 0 0)}
.v-timeline .d-box svg{width:100%;height:100%;display:block}
.v-timeline .d-a0{fill:color-mix(in srgb,var(--accent) 55%,var(--card))}
.v-timeline .d-a1{fill:color-mix(in srgb,var(--accent) 22%,var(--card))}
.v-timeline .d-ln{fill:none;stroke:var(--accent);stroke-width:1.5}
.v-timeline .d-gl{stroke:var(--line-2);stroke-width:1;stroke-dasharray:3 3}
.v-timeline .d-y{position:absolute;left:4px;font:10.5px var(--f-mono);color:var(--ink-3);transform:translateY(-110%)}
.v-timeline .d-x{display:flex;justify-content:space-between;font:11px var(--f-mono);color:var(--ink-3);margin-top:6px}
.v-timeline .d-ph,.v-timeline .d-hv{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:var(--accent);pointer-events:none}
.v-timeline .d-hv{width:1px;background:var(--ink-3);opacity:0;transition:opacity .15s}
.v-timeline .d-box:hover .d-hv{opacity:.7}
.v-timeline .d-band{position:absolute;top:0;bottom:0;background:color-mix(in srgb,var(--accent) 7%,transparent);pointer-events:none}
.v-timeline .d-leg{display:flex;gap:12px;font-size:12px;color:var(--ink-2);flex-wrap:wrap}
.v-timeline .d-leg i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.v-timeline .d-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
@container (max-width:980px){.v-timeline .tl-main{grid-template-columns:minmax(0,1fr)}}
@container (max-width:780px){.v-timeline .tl-bottom{grid-template-columns:minmax(0,1fr)}.v-timeline .tl{grid-template-columns:140px minmax(0,1fr)}}
@media (max-width:820px){.v-timeline .tl-main,.v-timeline .tl-bottom{grid-template-columns:minmax(0,1fr)}.v-timeline .tl{grid-template-columns:140px minmax(0,1fr)}}
`;

  function analyse(s) {
    const dur = s.duration, people = s.tracks.filter(t => (t.kind || "person") === "person");
    const row = new Map(), byCam = new Map();
    const lanes = s.cameras.map(c => {
      const ts = people.filter(t => t.cam === c.id).sort((a, b) => a.t0 - b.t0 || a.t1 - b.t1), ends = [];
      ts.forEach(t => { let r = ends.findIndex(e => t.t0 >= e + 0.35); if (r < 0) { r = ends.length; ends.push(0); } ends[r] = t.t1; row.set(t.id, r); });
      byCam.set(c.id, ts);
      return { cam: c, tracks: ts, rows: Math.max(1, ends.length) };
    });
    const idOf = new Map(), hops = [];
    (s.identities || []).forEach(i => i.tracks.forEach(id => idOf.set(id, i)));
    (s.identities || []).forEach(i => {
      if (i.hops && i.hops.length) i.hops.forEach(h => hops.push({ idn: i.id, from: h.from, to: h.to, score: h.score, why: h.why }));
      else if (i.tracks.length > 1) {
        const ts = i.tracks.map(s.track).filter(Boolean).sort((a, b) => a.t0 - b.t0);
        for (let k = 1; k < ts.length; k++) hops.push({ idn: i.id, from: ts[k - 1].id, to: ts[k].id });
      }
    });
    const okHops = hops.filter(h => (h.a = s.track(h.from)) && (h.b = s.track(h.to)));
    okHops.forEach(h => { h.t = h.b.t0 < h.a.t1 ? (h.b.t0 + h.a.t1) / 2 : h.a.t1; });
    const dens = [];
    for (let t = 0; t <= dur + 1e-6; t += STEP) {
      const by = {}, who = new Set(); s.cameras.forEach(c => { by[c.id] = 0; });
      people.forEach(p => { if (t >= p.t0 && t <= p.t1) { by[p.cam]++; who.add((idOf.get(p.id) || p).id); } });
      dens.push({ t: Math.round(t * 100) / 100, by, n: who.size, stack: Object.values(by).reduce((a, b) => a + b, 0) });
    }
    const peak = dens.reduce((m, d) => d.n > m.n ? d : m, dens[0]);
    const avg = dens.reduce((a, d) => a + d.n, 0) / dens.length;
    const empty = dens.filter(d => d.n === 0).length * STEP;
    const maxStack = Math.max(1, ...dens.map(d => d.stack));
    const win = Math.min(10, dur); let busy = { t: 0, n: -1, ps: 0, win };
    for (let w = 0; w <= dur - win + 1e-6; w += STEP) {
      let n = 0, ps = 0;
      people.forEach(p => { const o = Math.min(p.t1, w + win) - Math.max(p.t0, w); if (o >= 0) { n++; ps += o; } });
      if (n > busy.n || (n === busy.n && ps > busy.ps)) busy = { t: w, n, ps, win };
    }
    const longest = people.reduce((m, p) => (p.t1 - p.t0) > (m.t1 - m.t0) ? p : m, people[0]);
    return { lanes, row, byCam, idOf, hops: okHops, dens, peak, avg, empty, maxStack, busy, longest, people };
  }

  S.view("timeline", {
    label: "Timeline",
    async mount(root, ctx) {
      const el = S.el, s = ctx.scene || await S.loadScene(S.sceneName), dur = s.duration;
      const offs = [], timers = [];
      const on = (tg, ev, fn, opt) => { tg.addEventListener(ev, fn, opt); offs.push(() => tg.removeEventListener(ev, fn, opt)); };
      const later = (fn, ms) => timers.push(setTimeout(fn, ms));
      root.append(el("style", { html: CSS }));
      const wrap = el("div", { class: "v-timeline" });
      root.append(wrap);
      const A = analyse(s);
      const st = { ph: 0, playing: false, speed: 1, zoom: 1, sel: null, cam: s.cameras[0].id, raf: 0, zraf: 0, last: 0, ready: false, failed: false, followed: null, nowIdx: -1, inKey: "", filter: "all", built: false };
      const firstOnCam = A.byCam.get(st.cam)[0] || A.people[0];
      st.ph = firstOnCam ? Math.min(dur, firstOnCam.t0 + 0.5) : 0;
      const clockFine = t => `${s.clock(t)}.${Math.floor(((t % 1) + 1e-9) * 10) % 10}`;

      /* ---------- header + stats ---------- */
      const scenes = [["meva", "Campus walkway"], ["lab", "Lab floor"], ["caviar", "Shopping corridor"]];
      const pick = el("div", { class: "row tl-scenes" }, el("span", { class: "eyebrow" }, "Scene"), scenes.map(([n, label]) =>
        el("button", { type: "button", class: "chip" + (s.name === n ? " on" : ""), id: "timeline-scene-" + n, onclick: async e => {
          if (n === s.name) return;
          e.currentTarget.prepend(el("span", { class: "spin", style: { width: "11px", height: "11px" } }));
          const ns = await S.loadScene(n).catch(() => null);
          if (!ns) { S.toast("That scene could not be loaded."); return; }
          S.sceneName = n; S.emit("scene", n); S.go("timeline");
        } }, label)));
      S.pageHead(wrap, "Timeline", `${s.cameras.length} cameras · ${s.clock(0)}–${s.clock(dur)}`, pick);
      const statDefs = [
        ["People tracks", A.people.length, 0, "", A.lanes.map(l => `${l.tracks.length} on Cam ${l.cam.id}`).join(" · ")],
        ["Cross-camera hops", A.hops.length, 0, "", A.hops.length ? A.hops.map(h => `${h.idn}: Cam ${h.a.cam} → ${h.b.cam}`).join(" · ") : "none in this clip"],
        ["Most in view at once", A.peak.n, 0, "", `at ${s.clock(A.peak.t)}`],
      ];
      const statNums = [];
      wrap.append(el("div", { class: "tl-stats stagger" }, statDefs.map(d => {
        const b = el("b", {}, S.skeleton(24, "64px")); statNums.push(b);
        return el("div", { class: "card tl-stat" }, el("span", { class: "eyebrow" }, d[0]), b, el("small", { title: d[4] }, d[4]));
      })));

      /* ---------- card 1: swimlanes ---------- */
      const playBtn = el("button", { type: "button", class: "btn btn-primary btn-sm", id: "timeline-play", onclick: () => st.playing ? pause() : play() });
      const speedBtn = el("button", { type: "button", class: "btn btn-sm", id: "timeline-speed", title: "Playback speed", onclick: () => setSpeed(st.speed === 1 ? 2 : st.speed === 2 ? 4 : 1) }, "1x");
      const readout = el("span", { class: "mono tl-clock", id: "timeline-clock" });
      const zoom = el("input", { type: "range", min: 1, max: 4, step: 0.1, value: 1, id: "timeline-zoom", "aria-label": "Zoom" });
      const zoomVal = el("span", { class: "mono muted" }, "1.0x");
      const labels = el("div", { class: "tl-labels" }, el("div", { class: "tl-axis-sp eyebrow" }, "Camera"));
      const axis = el("div", { class: "tl-axis" }), gridL = el("div", { class: "tl-gridl" });
      const band = el("div", { class: "tl-band", style: { left: pos(A.busy.t / dur), width: wpos(A.busy.win / dur) } });
      const NS = "http://www.w3.org/2000/svg", links = document.createElementNS(NS, "svg");
      links.setAttribute("class", "tl-links"); links.setAttribute("aria-hidden", "true");
      const hopLayer = el("div", { class: "tl-hopl" });
      const phEl = el("div", { class: "tl-ph", id: "timeline-playhead" }, el("i"));
      const inner = el("div", { class: "tl-inner", id: "timeline-lanes" }, gridL, band, axis);
      A.lanes.forEach((L, k) => {
        L.h = L.rows * (BAR + GAP) - GAP + LP * 2 + 4;
        L.el = el("div", { class: "tl-lane" + (k === A.lanes.length - 1 ? " last" : ""), style: { height: L.h + "px" }, "data-cam": L.cam.id });
        for (let r = 0; r < Math.min(L.rows, 3); r++) L.el.append(el("div", { class: "skel tl-sk", style: { top: LP + r * (BAR + GAP) + "px", left: pos(0.04 + r * 0.22), width: wpos(0.3 + (r % 2) * 0.25) } }));
        L.lab = el("button", { type: "button", class: "tl-lab", style: { height: L.h + "px" }, id: "timeline-lane-" + L.cam.id, title: `${L.cam.name}: show in the preview`, onclick: () => setCam(L.cam.id) },
          el("b", {}, L.cam.name), el("span", { class: "muted" }, `Cam ${L.cam.id} · ${L.tracks.length} track${L.tracks.length === 1 ? "" : "s"}`));
        labels.append(L.lab); inner.append(L.el);
      });
      inner.append(links, hopLayer, phEl);
      const scroller = el("div", { class: "tl-scroll" }, inner);
      const tip = el("div", { class: "tl-tip", "aria-hidden": "true" });
      const tl = el("div", { class: "tl" }, labels, scroller, tip);
      const selBox = el("div", { class: "tl-sel", id: "timeline-selection" });
      const tlCard = el("div", { class: "card", id: "timeline-card" },
        el("div", { class: "card-h" }, el("h3", {}, "Swimlanes"), el("div", { class: "grow" }), readout, speedBtn, playBtn),
        el("div", { class: "card-b" },
          el("div", { class: "row tl-tools" }, el("span", { class: "muted" }, "Click a bar to follow one person. Drag to scrub."),
            el("span", { class: "muted" }, el("span", { class: "tl-key" }), ` busiest ${fmtS(A.busy.win)}`), el("div", { class: "grow" }), el("span", { class: "eyebrow" }, "Zoom"), zoom, zoomVal),
          tl, selBox));

      /* ---------- preview card ---------- */
      const camChips = s.cameras.map(c => el("button", { type: "button", class: "chip", id: "timeline-cam-" + c.id, onclick: () => setCam(c.id) }, "Cam " + c.id));
      const video = el("video", { muted: true, playsinline: true, preload: "auto", id: "timeline-video" }); video.muted = true;
      const cv = el("canvas"), vstate = el("div", { class: "tl-vstate" }, el("span", { class: "spin" }));
      const vlabel = el("span", { class: "tl-vlabel mono", style: { color: S.colourHex.white } });
      const frame = el("div", { class: "tl-frame" }, video, cv, vstate, vlabel);
      const camName = el("b"), sceneDd = el("dd", { class: "mono" }), videoDd = el("dd", { class: "mono" }), inView = el("div", { class: "tl-inview" });
      const pvCard = el("div", { class: "card", id: "timeline-preview" },
        el("div", { class: "card-h" }, el("h3", {}, "Preview"), el("div", { class: "grow" }), camChips),
        el("div", { class: "card-b" }, frame,
          el("div", { class: "tl-pv-meta" }, el("div", { class: "row", html: S.icons.cam }, camName),
            el("dl", { class: "kv", style: { margin: 0 } }, el("dt", {}, "Scene clock"), sceneDd, el("dt", {}, "Video time"), videoDd),
            el("div", { class: "eyebrow" }, "In view now"), inView)));
      wrap.append(el("div", { class: "tl-main" }, tlCard, pvCard));

      /* ---------- card 2: events + density ---------- */
      const evList = el("div", { class: "tl-evs", id: "timeline-events" }, [0, 1, 2, 3, 4, 5].map(() => el("div", { class: "row", style: { padding: "8px" } }, S.skeleton(14, "60px"), S.skeleton(28, "22px", 6), S.skeleton(14, "70%"))));
      const filters = [["all", "All"], ["hl", "Highlights"], ...s.cameras.map(c => ["cam-" + c.id, "Cam " + c.id])];
      const filterBtns = filters.map(([k, label]) => el("button", { type: "button", class: "chip" + (k === "all" ? " on" : ""), id: "timeline-ev-" + k, onclick: () => setFilter(k) }, label));
      const dRead = el("div", { class: "d-read" }, "Hover the chart to read a moment. Click to jump there.");
      const dClip = el("div", { class: "d-clip" }), dPh = el("div", { class: "d-ph" }), dHv = el("div", { class: "d-hv" });
      const dBox = el("div", { class: "d-box", id: "timeline-density" }, S.skeleton("100%", "100%", 10));
      const dStats = el("div", { class: "d-stats" });
      wrap.append(el("div", { class: "tl-bottom" },
        el("div", { class: "card" }, el("div", { class: "card-h" }, el("h3", {}, "Events"), el("div", { class: "grow" }), el("div", { class: "row wrap", style: { gap: "6px" } }, filterBtns)), evList),
        el("div", { class: "card" },
          el("div", { class: "card-h" }, el("h3", {}, "People in view"), el("div", { class: "grow" }),
            el("div", { class: "d-leg" }, s.cameras.map((c, k) => el("span", {}, el("i", { class: "d-a" + (k % 2), style: { background: `color-mix(in srgb,var(--accent) ${k % 2 ? 22 : 55}%,var(--card))` } }), c.name)))),
          el("div", { class: "card-b" }, dRead, dBox, el("div", { class: "d-x" }, el("span", {}, s.clock(0)), el("span", {}, s.clock(dur / 2)), el("span", {}, s.clock(dur))), dStats))));

      /* ---------- geometry ---------- */
      const innerW = () => inner.offsetWidth;
      const xOf = t => PAD + t / dur * (innerW() - 2 * PAD);
      const barTop = t => { const L = A.lanes.find(l => l.cam.id === t.cam); return L.el.offsetTop + LP + A.row.get(t.id) * (BAR + GAP); };
      const localXY = (node, e) => { const r = node.getBoundingClientRect(), k = r.width / (node.offsetWidth || 1) || 1; return [(e.clientX - r.left) / k, (e.clientY - r.top) / k, k]; };
      const selGroup = () => { if (!st.sel) return new Set(); const i = A.idOf.get(st.sel); return new Set(i ? i.tracks : [st.sel]); };

      let tickKey = "";
      function drawTicks() {
        const bw = innerW() - 2 * PAD; if (bw <= 0) return;
        const pps = bw / dur;
        const step = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find(v => v * pps >= 92) || 600;
        const key = step + ":" + Math.round(bw / 8); if (key === tickKey) return; tickKey = key;
        axis.innerHTML = ""; gridL.innerHTML = "";
        const ticks = [];
        for (let t = 0; t <= dur + 1e-6; t += step) {
          const f = t / dur, txt = step < 1 ? s.clock(t).slice(3) + "." + Math.round((t % 1) * 10) : s.clock(t);
          const w = txt.length * 6.8, x = PAD + f * bw, kind = t === 0 ? "first" : f > 0.97 ? "last" : "";
          const l = kind === "first" ? x - 2 : kind === "last" ? x - w + 2 : x - w / 2;
          ticks.push({ f, txt, kind, l, r: l + w });
        }
        const last = ticks[ticks.length - 1];
        let edge = -Infinity;
        ticks.forEach(k => {
          gridL.append(el("i", { style: { left: pos(k.f) } }));
          if (k.l < edge + 12 || (k !== last && last.kind === "last" && k.r > last.l - 12)) return;
          axis.append(el("span", { class: "tl-tick" + (k.kind ? " " + k.kind : ""), style: { left: pos(k.f) } }, k.txt));
          edge = k.r;
        });
      }
      function drawLinks() {
        const W = innerW(), H = inner.offsetHeight;
        links.setAttribute("width", W); links.setAttribute("height", H); links.setAttribute("viewBox", `0 0 ${W} ${H}`);
        links.innerHTML = ""; hopLayer.innerHTML = "";
        if (!st.built) return;
        const bw = W - 2 * PAD;
        A.lanes.forEach(L => L.tracks.forEach(t => { const b = L.el.querySelector(`[data-id="${t.id}"]`); if (b) b.classList.toggle("tiny", (t.t1 - t.t0) / dur * bw < String(t.id).length * 6.5 + 14); }));
        const selI = st.sel && A.idOf.get(st.sel);
        A.hops.forEach((h, i) => {
          const a = h.a, b = h.b, onSel = !!(selI && selI.id === h.idn), down = barTop(a) < barTop(b);
          const xa = h.b.t0 < h.a.t1 ? xOf(h.t) : xOf(a.t1), xb = h.b.t0 < h.a.t1 ? xOf(h.t) : xOf(b.t0);
          const ya = barTop(a) + (down ? BAR : 0), yb = barTop(b) + (down ? 0 : BAR), ym = (ya + yb) / 2;
          links.insertAdjacentHTML("beforeend", `<path class="tl-link${onSel ? " on" : ""}" d="M${xa},${ya} C${xa},${ym} ${xb},${ym} ${xb},${yb}"/><circle class="tl-dot" cx="${xa}" cy="${ya}" r="3.5"/><circle class="tl-dot" cx="${xb}" cy="${yb}" r="3.5"/>`);
          if (onSel) hopLayer.append(el("span", { class: "tl-hop on", id: "timeline-hop-" + i, style: { left: (xa + xb) / 2 + "px", top: ym + "px" } },
            `Cam ${a.cam} → ${b.cam}` + (h.score != null ? ` · ${h.score.toFixed(2)}` : "")));
        });
      }
      function drawBars() {
        A.lanes.forEach(L => {
          L.el.innerHTML = "";
          L.tracks.forEach((t, i) => {
            const c = hex(t.look && t.look.top), idn = A.idOf.get(t.id);
            const b = el("button", { type: "button", class: "tl-bar", id: "timeline-bar-" + t.id, "data-id": t.id, "aria-label": `${t.desc}, ${s.clock(t.t0)} to ${s.clock(t.t1)}`,
              style: { left: pos(t.t0 / dur), width: wpos((t.t1 - t.t0) / dur), top: LP + A.row.get(t.id) * (BAR + GAP) + "px", background: c, color: textOn(c), animationDelay: Math.min(650, 40 + t.t0 / dur * 420 + i * 24) + "ms" },
              onclick: () => select(t.id, true) }, t.id);
            b.style.setProperty("--bb", hex(t.look && t.look.bottom));
            b.addEventListener("pointerenter", e => showTip(t, idn, e));
            b.addEventListener("pointermove", moveTip);
            b.addEventListener("pointerleave", hideTip);
            L.el.append(b);
          });
        });
      }

      /* ---------- tooltip ---------- */
      function showTip(t, idn, e) {
        const c = s.cam(t.cam);
        tip.innerHTML = `<img class="crop" src="${S.esc(s.cropUrl(t))}" alt="Crop of track ${S.esc(t.id)}"><div><b>${S.esc(cap(t.desc))}</b>`
          + `<div>${S.esc(c.name)} · Cam ${S.esc(c.id)}</div><div class="mono">${s.clock(t.t0)} – ${s.clock(t.t1)}</div>`
          + `<div class="row mono muted">${fmtS(t.t1 - t.t0)} · ${S.esc(t.id)}${idn ? " · " + S.esc(idn.id) : ""}</div></div>`;
        tip.classList.add("show"); moveTip(e);
      }
      function hideTip() { tip.classList.remove("show"); tip.style.left = "0px"; tip.style.top = "0px"; }
      function moveTip(e) {
        const [x, y] = localXY(tl, e), w = tl.offsetWidth, tw = 268;
        let left = x + 16; if (left + tw > w - 4) left = Math.max(4, x - tw - 16);
        tip.style.transform = ""; tip.style.left = left + "px"; tip.style.top = Math.max(4, y + 14) + "px";
      }

      /* ---------- playhead + sync ---------- */
      function setPH(t, fromLoop) {
        st.ph = clamp(t, 0, dur);
        const f = st.ph / dur;
        phEl.style.left = pos(f); dPh.style.left = f * 100 + "%";
        readout.textContent = clockFine(st.ph); sceneDd.textContent = clockFine(st.ph);
        if (!fromLoop) syncVideo();
        paintBoxes(); markNow();
      }
      function syncVideo(force) {
        if (!st.ready || st.failed) return;
        const off = s.cam(st.cam).offset || 0, vt = clamp(st.ph - off, 0, Math.max(0, (video.duration || dur) - 0.05));
        if (force || Math.abs(video.currentTime - vt) > 0.06) { try { video.currentTime = vt; } catch (e) {} }
      }
      function seekTo(t, center) {
        setPH(t);
        if (st.zoom > 1.01) {
          const x = xOf(st.ph), sl = scroller.scrollLeft, cw = scroller.clientWidth;
          if (center || x < sl + 20 || x > sl + cw - 20) scroller.scrollTo({ left: x - cw * 0.3, behavior: S.reduced ? "auto" : "smooth" });
        }
      }
      function play() {
        if (st.playing) return;
        if (st.ph >= dur - 0.05) setPH(0);
        st.playing = true; st.last = performance.now(); paintPlay();
        if (st.ready && !st.failed) { syncVideo(true); video.playbackRate = st.speed; video.play().catch(() => {}); }
        st.raf = requestAnimationFrame(tick);
      }
      function pause() {
        st.playing = false; cancelAnimationFrame(st.raf); try { video.pause(); } catch (e) {} paintPlay();
      }
      function paintPlay() { playBtn.innerHTML = (st.playing ? S.icons.pause : S.icons.play) + (st.playing ? "Pause" : "Play"); }
      function setSpeed(v) { st.speed = v; speedBtn.textContent = v + "x"; try { video.playbackRate = v; } catch (e) {} }
      function tick(now) {
        if (!st.playing) return;
        const dt = Math.min(0.1, (now - st.last) / 1000) * st.speed; st.last = now;
        let next = st.ph + dt;
        const off = s.cam(st.cam).offset || 0;
        if (st.ready && !st.failed && !video.paused && !video.seeking && video.readyState >= 3) {
          const vt = video.currentTime + off;
          if (Math.abs(vt - next) < 0.5) next = vt; else { try { video.currentTime = Math.max(0, next - off); } catch (e) {} }
        }
        if (next >= dur) { setPH(dur, true); pause(); return; }
        setPH(next, true); follow();
        if (st.zoom > 1.01) {
          const x = xOf(st.ph), sl = scroller.scrollLeft, cw = scroller.clientWidth;
          if (x > sl + cw * 0.85 || x < sl) scroller.scrollLeft = x - cw * 0.3;
        }
        st.raf = requestAnimationFrame(tick);
      }
      function follow() {
        if (!st.sel) return;
        const idn = A.idOf.get(st.sel); if (!idn || idn.tracks.length < 2) return;
        const cur = idn.tracks.map(s.track).filter(t => t && st.ph >= t.t0 && st.ph <= t.t1).sort((a, b) => b.t0 - a.t0)[0];
        if (cur && cur.cam !== st.cam && cur.id !== st.followed) { st.followed = cur.id; setCam(cur.cam); S.toast(`Following ${S.esc(idn.id)} onto ${S.esc(s.cam(cur.cam).name)}`); }
      }

      /* ---------- zoom ---------- */
      function setZoom(z) {
        const to = clamp(+z || 1, 1, 4), from = st.zoom, t0 = performance.now(), ms = S.reduced ? 0 : 380;
        zoom.value = to; zoomVal.textContent = to.toFixed(1) + "x";
        const anchor = xOf(st.ph) - scroller.scrollLeft;
        cancelAnimationFrame(st.zraf);
        const step = now => {
          const k = ms ? Math.min(1, (now - t0) / ms) : 1, e = 1 - Math.pow(1 - k, 3);
          st.zoom = from + (to - from) * e;
          inner.style.width = st.zoom * 100 + "%";
          scroller.scrollLeft = xOf(st.ph) - anchor;
          drawTicks(); drawLinks();
          if (k < 1) st.zraf = requestAnimationFrame(step);
        };
        st.zraf = requestAnimationFrame(step);
      }

      /* ---------- selection ---------- */
      function select(id, seek) {
        const t = id && s.track(id);
        st.sel = t ? id : null; st.followed = null;
        const g = selGroup();
        tl.classList.toggle("has-sel", !!st.sel);
        inner.querySelectorAll(".tl-bar").forEach(b => b.classList.toggle("on", g.has(b.dataset.id)));
        drawLinks(); paintSel();
        if (t) { if (st.cam !== t.cam) setCam(t.cam); if (seek && (st.ph < t.t0 || st.ph > t.t1)) seekTo(t.t0); }
        paintBoxes();
        S.emit("timeline:select", { scene: s.name, track: st.sel });
      }
      function paintSel() {
        selBox.innerHTML = ""; selBox.classList.toggle("has", !!st.sel);
        if (!st.sel) return;
        const t = s.track(st.sel), idn = A.idOf.get(t.id);
        const ts = (idn ? idn.tracks : [t.id]).map(s.track).filter(Boolean).sort((a, b) => a.t0 - b.t0);
        const hops = A.hops.filter(h => idn && h.idn === idn.id);
        const first = Math.min(...ts.map(x => x.t0)), last = Math.max(...ts.map(x => x.t1));
        selBox.append(el("img", { class: "crop", src: s.cropUrl(t), alt: "Crop of track " + t.id }),
          el("div", { class: "tl-sel-m" },
            el("div", { class: "row" }, el("b", {}, cap(t.desc)), el("span", { class: "mono muted" }, idn ? idn.id : t.id),
              el("span", { class: "muted" }, el("span", { class: "swatch", style: { background: hex(t.look && t.look.top) } }), " ", el("span", { class: "swatch", style: { background: hex(t.look && t.look.bottom) } }))),
            el("div", { class: "mono dim" }, ts.map(x => `Cam ${x.cam} ${s.clock(x.t0)}–${s.clock(x.t1)}`).join("  →  ") + `  ·  ${fmtS(last - first)} in view`),
            el("div", { class: "muted" }, hops.length ? hops.map(h => `Hop Cam ${h.a.cam} → Cam ${h.b.cam}: ${String(h.why || "same identity").replace(/->/g, "→")}`).join("; ") : "Seen on one camera only in this clip.")),
          el("div", { class: "row" }, el("button", { type: "button", class: "btn btn-sm", id: "timeline-jump", onclick: () => seekTo(first, true) }, "Jump to first seen"),
            el("button", { type: "button", class: "btn btn-ghost btn-sm", id: "timeline-clear", onclick: () => select(null) }, "Clear")));
        selBox.classList.remove("pop"); void selBox.offsetWidth; selBox.classList.add("pop");
      }

      /* ---------- preview ---------- */
      function setCam(id) {
        const c = s.cam(id); if (!c) return;
        const changed = st.cam !== id || !video.getAttribute("src"); st.cam = id;
        camChips.forEach(b => b.classList.toggle("on", b.id === "timeline-cam-" + id));
        A.lanes.forEach(L => L.lab.classList.toggle("on", L.cam.id === id));
        camName.textContent = `${c.name} · Cam ${c.id}`;
        const sz = ((s.build.per_camera || []).find(p => p.camera === id) || {}).size || [16, 9];
        frame.style.aspectRatio = `${sz[0]} / ${sz[1]}`;
        st.inKey = "";
        if (changed) {
          st.ready = false; st.failed = false; vstate.hidden = false; vstate.innerHTML = '<span class="spin"></span>';
          video.src = s.videoUrl(c); video.load();
        }
        paintBoxes();
      }
      function paintBoxes() {
        const w = cv.clientWidth, h = cv.clientHeight, cam = s.cam(st.cam);
        vlabel.textContent = `CAM ${cam.id} · ${clockFine(st.ph)}`;
        const vt = Math.max(0, st.ph - (cam.offset || 0));
        videoDd.textContent = `${vt.toFixed(1)} s / ${fmtS(st.ready && video.duration ? video.duration : dur)}`;
        const g = selGroup(), vis = [];
        (A.byCam.get(st.cam) || []).forEach(t => { if (st.ph >= t.t0 - 0.25 && st.ph <= t.t1 + 0.25) { const b = s.boxAt(t, st.ph); if (b) vis.push([t, b]); } });
        if (w && h) {
          const dpr = window.devicePixelRatio || 1;
          if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
          const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
          const accent = getComputedStyle(wrap).getPropertyValue("--accent").trim() || S.colourHex.blue;
          c.font = '600 10px "JetBrains Mono", ui-monospace, monospace'; c.textBaseline = "middle";
          vis.forEach(([t, b]) => {
            const on = g.has(t.id), col = on ? accent : hex(t.look && t.look.top), x = b[0] * w, y = b[1] * h, bw = b[2] * w, bh = b[3] * h;
            c.globalAlpha = st.sel && !on ? 0.5 : 1; c.lineWidth = on ? 2.5 : 1.5; c.strokeStyle = col; c.strokeRect(x, y, bw, bh);
            const label = on && A.idOf.get(t.id) ? A.idOf.get(t.id).id : t.id, tw = c.measureText(label).width + 8, ly = y > 15 ? y - 14 : y;
            c.fillStyle = col; c.fillRect(x - (on ? 1.25 : 0.75), ly, tw, 14); c.fillStyle = textOn(on ? S.colourHex.blue : col); c.fillText(label, x + 3, ly + 7);
          });
          c.globalAlpha = 1;
        }
        const key = vis.map(v => v[0].id).join(",") + "|" + (st.sel || "");
        if (key !== st.inKey) {
          st.inKey = key; inView.innerHTML = "";
          if (!vis.length) inView.append(el("span", { class: "muted", style: { fontSize: "12px" } }, "Nobody on this camera right now."));
          vis.forEach(([t]) => inView.append(el("button", { type: "button", class: "chip" + (g.has(t.id) ? " on" : ""), onclick: () => select(t.id) },
            el("span", { class: "swatch", style: { background: hex(t.look && t.look.top) } }), `${t.id} · ${t.look ? t.look.top + " top" : ""}`)));
        }
      }
      on(video, "loadeddata", () => { st.ready = true; vstate.hidden = true; video.playbackRate = st.speed; syncVideo(true); if (st.playing) video.play().catch(() => {}); paintBoxes(); });
      on(video, "seeked", () => paintBoxes());
      on(video, "error", () => { st.failed = true; vstate.hidden = false; vstate.textContent = "Video file not available for this camera. Boxes still follow the index."; });

      /* ---------- events ---------- */
      const events = [];
      A.people.forEach(t => {
        const c = s.cam(t.cam);
        events.push(t.t0 <= 0.5 ? { t: t.t0, kind: "enter", track: t, b: `In view at start · ${c.name}`, sub: `${t.desc} · ${t.id}`, pill: ["plain", "Start"] }
          : { t: t.t0, kind: "enter", track: t, b: `Enters ${c.name}`, sub: `${t.desc} · ${t.id}`, pill: ["ok", "Enters"] });
        if (t.t1 < dur - 0.6) events.push({ t: t.t1, kind: "leave", track: t, b: `Leaves ${c.name}`, sub: `${t.desc} · stayed ${fmtS(t.t1 - t.t0)}`, pill: ["plain", "Leaves"] });
      });
      A.hops.forEach((h, i) => events.push({ t: h.t, kind: "hop", id: "timeline-event-hop-" + i, track: h.b, cams: [h.a.cam, h.b.cam], b: `${h.idn} moves Cam ${h.a.cam} to Cam ${h.b.cam}`, sub: String(h.why || "same identity").replace(/->/g, "→"), pill: ["info", "Hop"] }));
      if (A.longest) events.push({ t: A.longest.t0, kind: "hl", id: "timeline-event-longest", track: A.longest, b: `Longest dwell · ${fmtS(A.longest.t1 - A.longest.t0)} on ${s.cam(A.longest.cam).name}`, sub: `${A.longest.desc} · ${A.longest.id}`, pill: ["info", "Highlight"] });
      events.push({ t: A.busy.t, kind: "hl", id: "timeline-event-busiest", band: true, b: `Busiest ${fmtS(A.busy.win)} · ${A.busy.n} people tracks`, sub: `${s.clock(A.busy.t)}–${s.clock(A.busy.t + A.busy.win)} · ${Math.round(A.busy.ps)} person-seconds`, pill: ["info", "Highlight"] });
      events.push({ t: A.peak.t, kind: "hl", id: "timeline-event-peak", b: `Most in view at once · ${A.peak.n} ${A.peak.n === 1 ? "person" : "people"}`, sub: s.cameras.map(c => `${A.peak.by[c.id]} on Cam ${c.id}`).join(" · "), pill: ["info", "Highlight"] });
      const order = { hl: 0, hop: 1, enter: 2, leave: 3 };
      events.sort((a, b) => a.t - b.t || order[a.kind] - order[b.kind]);
      function onEvent(e) {
        seekTo(e.t, true);
        if (e.track) select(e.track.id, false);
        if (e.band) { band.classList.add("hot"); later(() => band.classList.remove("hot"), 2600); }
        if (!e.track && e.kind === "hl" && !e.band) { const busiest = A.dens.find(d => d.t === e.t); if (busiest) dRead.textContent = readText(busiest); }
      }
      function buildEvents() {
        evList.innerHTML = ""; evList.classList.add("stagger");
        events.forEach((e, i) => {
          e.row = el("button", { type: "button", class: "tl-ev", id: e.id || "timeline-ev-row-" + i, title: `${e.b}. ${e.sub}`, onclick: () => onEvent(e) },
            el("span", { class: "tl-ev-t" }, s.clock(e.t)),
            e.track ? el("img", { class: "crop", src: s.cropUrl(e.track), alt: "", loading: "lazy" }) : el("span", { class: "tl-ev-ico" }),
            el("span", { class: "tl-ev-m" }, el("b", {}, e.b), el("span", { class: "muted" }, e.sub)));
          evList.append(e.row);
        });
        setFilter(st.filter); markNow();
      }
      function setFilter(k) {
        st.filter = k;
        filterBtns.forEach(b => b.classList.toggle("on", b.id === "timeline-ev-" + k));
        events.forEach(e => {
          if (!e.row) return;
          const show = k === "all" || (k === "hl" && (e.kind === "hl" || e.kind === "hop")) || (k.startsWith("cam-") && ((e.track && e.track.cam === k.slice(4)) || (e.cams && e.cams.includes(k.slice(4)))));
          e.row.classList.toggle("off", !show);
        });
      }
      function markNow() {
        if (!events.length || !events[0].row) return;
        let idx = -1; for (let i = 0; i < events.length; i++) if (events[i].t <= st.ph + 0.05) idx = i;
        if (idx === st.nowIdx) return;
        if (st.nowIdx >= 0 && events[st.nowIdx]) events[st.nowIdx].row.classList.remove("now");
        st.nowIdx = idx;
        if (idx < 0) return;
        const r = events[idx].row; r.classList.add("now");
        if (st.playing && !r.classList.contains("off")) evList.scrollTo({ top: r.offsetTop - evList.clientHeight / 2, behavior: S.reduced ? "auto" : "smooth" });
      }

      /* ---------- density ---------- */
      const readText = d => `${s.clock(d.t)} · ${d.n} ${d.n === 1 ? "person" : "people"} in view (${s.cameras.map(c => `Cam ${c.id} ${d.by[c.id]}`).join(" · ")})`;
      function buildDensity() {
        const W = 1000, H = 150, top = 14, bot = 2, max = A.maxStack;
        const X = t => (t / dur * W).toFixed(1), Y = v => (H - bot - v / max * (H - top - bot)).toFixed(1);
        let lower = A.dens.map(() => 0), svg = "";
        const grid = [...new Set([max, Math.round(max / 2)])].filter(v => v > 0);
        grid.forEach(v => { svg += `<line class="d-gl" x1="0" x2="${W}" y1="${Y(v)}" y2="${Y(v)}" vector-effect="non-scaling-stroke"/>`; });
        s.cameras.forEach((c, k) => {
          const upper = A.dens.map((d, i) => lower[i] + d.by[c.id]);
          const up = A.dens.map((d, i) => `${X(d.t)},${Y(upper[i])}`), dn = A.dens.map((d, i) => `${X(d.t)},${Y(lower[i])}`).reverse();
          svg += `<path class="d-a${k % 2}" d="M${up.join("L")}L${dn.join("L")}Z"/>`;
          lower = upper;
        });
        svg += `<path class="d-ln" vector-effect="non-scaling-stroke" d="M${A.dens.map((d, i) => `${X(d.t)},${Y(lower[i])}`).join("L")}"/>`;
        dClip.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${svg}</svg>`;
        dBox.innerHTML = "";
        dBox.append(el("div", { class: "d-band", style: { left: A.busy.t / dur * 100 + "%", width: A.busy.win / dur * 100 + "%" } }), dClip, ...grid.map(v => el("span", { class: "d-y", style: { top: Y(v) / H * 100 + "%" } }, String(v))), dHv, dPh);
        requestAnimationFrame(() => requestAnimationFrame(() => dClip.classList.add("in")));
        const pickD = e => { const [x] = localXY(dBox, e), f = clamp(x / (dBox.offsetWidth || 1), 0, 1); return A.dens[Math.round(f * dur / STEP)] || A.dens[A.dens.length - 1]; };
        on(dBox, "pointermove", e => { const d = pickD(e); dHv.style.left = d.t / dur * 100 + "%"; dRead.textContent = readText(d); });
        on(dBox, "click", e => seekTo(pickD(e).t, true));
        const nums = [["Peak", A.peak.n, 0, "", `at ${s.clock(A.peak.t)}`], ["Average in view", A.avg, 1, "", "people per moment"], ["Nobody in view", A.empty, 0, " s", `of ${fmtS(dur)}`]];
        dStats.innerHTML = "";
        nums.forEach(([label, v, dec, suf, sub]) => { const b = el("b"); dStats.append(el("div", { class: "stat" }, b, el("span", {}, `${label} · ${sub}`))); S.countUp(b, v, { dec, suffix: suf, ms: 1000 }); });
      }

      /* ---------- scrub ---------- */
      let dragging = false;
      const scrubAt = e => { const [x] = localXY(inner, e); setPH((x - PAD) / (innerW() - 2 * PAD) * dur); };
      on(inner, "pointerdown", e => {
        if (e.button !== 0 || e.target.closest(".tl-bar")) return;
        dragging = true; tl.classList.add("scrub"); hideTip();
        try { inner.setPointerCapture(e.pointerId); } catch (err) {}
        scrubAt(e);
      });
      on(inner, "pointermove", e => { if (dragging) scrubAt(e); });
      const endDrag = () => { dragging = false; tl.classList.remove("scrub"); };
      on(inner, "pointerup", endDrag); on(inner, "pointercancel", endDrag);
      on(zoom, "input", () => setZoom(zoom.value));
      on(document, "keydown", e => {
        const tg = e.target; if (tg && (/INPUT|TEXTAREA|SELECT|BUTTON/.test(tg.tagName) || tg.isContentEditable)) return;
        if (e.key === " ") { e.preventDefault(); st.playing ? pause() : play(); }
        else if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); seekTo(st.ph + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 5 : 1)); }
      });
      const onSeek = ev => { const d = ev.detail || {}; if (typeof d.t === "number") { if (d.cam) setCam(d.cam); seekTo(d.t, true); } };
      on(S.bus, "seek", onSeek);
      const ro = new ResizeObserver(() => { drawTicks(); drawLinks(); paintBoxes(); });
      ro.observe(scroller); ro.observe(frame);

      /* ---------- go ---------- */
      paintPlay(); paintSel(); setCam(st.cam); setPH(st.ph); drawTicks();
      S.timeline = { select: id => select(id, true), seek: t => seekTo(t, true), play, pause, zoom: setZoom, cam: setCam, speed: setSpeed, state: () => ({ ph: st.ph, sel: st.sel, cam: st.cam, playing: st.playing }) };
      const cleanup = () => {
        st.playing = false; cancelAnimationFrame(st.raf); cancelAnimationFrame(st.zraf);
        timers.forEach(clearTimeout); offs.forEach(f => f()); ro.disconnect();
        try { video.pause(); video.removeAttribute("src"); video.load(); } catch (e) {}
        delete S.timeline;
      };
      await S.sleep(600);
      if (!wrap.isConnected) { cleanup(); return; }
      st.built = true;
      drawBars(); drawLinks();
      statDefs.forEach((d, i) => S.countUp(statNums[i], d[1], { dec: d[2], suffix: d[3], ms: 900 }));
      buildEvents(); buildDensity(); setPH(st.ph);

      return cleanup;
    },
  });
})();
