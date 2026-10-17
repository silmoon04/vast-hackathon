/* Catalogue: every person sighting the fast index catalogued, across all scenes. Sticky filters, FLIP-animated
   grid, and a detail drawer with the box-centre path, the identity's other sightings and hand-offs to Similar / Timeline. */
(function () {
  "use strict";
  const { el, esc } = STM;
  const EASE = "cubic-bezier(.2,.8,.2,1)";
  const ICO_SEARCH = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>';
  const ICO_X = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>';
  const hex = c => STM.colourHex[c] || STM.colourHex.grey;
  const cap = s => s ? s[0].toUpperCase() + s.slice(1) : "Unknown";
  const short = s => String(s.title || s.name).split(",")[0];
  const sec = v => v.toFixed(1) + " s";
  const norm = s => String(s || "").toLowerCase().replace(/\s+/g, " ").trim();
  const camSeconds = s => { const pc = (s.build && s.build.per_camera) || []; return pc.length ? pc.reduce((a, p) => a + (p.duration || 0), 0) : s.duration * s.cameras.length; };
  const frameSize = (s, cam) => { const p = ((s.build && s.build.per_camera) || []).find(p => p.camera === cam); return p && p.size ? p.size : [16, 9]; };
  const mark = (txt, q) => {
    const i = q ? txt.toLowerCase().indexOf(q) : -1;
    return i < 0 ? esc(txt) : esc(txt.slice(0, i)) + "<mark>" + esc(txt.slice(i, i + q.length)) + "</mark>" + esc(txt.slice(i + q.length));
  };
  const sw = (c, part) => el("span", { title: `${part}: ${c || "unknown"}` }, el("i", { class: "swatch", style: { background: hex(c) } }), cap(c));

  const colourWords = d => { const cs = Object.keys(STM.colourHex || {}); return norm(d).split(/[^a-z]+/).filter(x => cs.includes(x)); };
  const median = a => { const b = a.slice().sort((x, y) => x - y); return b.length ? b[b.length >> 1] : 0; };
  /* person-shaped: median box taller than wide in pixels, big enough, and seen long enough */
  function isPerson(s, t) {
    const bs = (t.boxes || []).filter(b => b && b.length >= 5);
    if (!bs.length || t.t1 - t.t0 < 1.5) return false;
    const [fw, fh] = frameSize(s, t.cam);
    const w = median(bs.map(b => b[3])), h = median(bs.map(b => b[4]));
    if (!w || h < 0.06) return false;
    return (h * fh) / (w * fw) >= 1.25;
  }
  const CSS = `
.v-catalogue .tf input{border:1px solid var(--line-2);border-radius:8px;background:var(--card);padding:4px 6px;font:12px var(--f-mono);color:var(--ink)}
.v-catalogue .hidline{font-size:12px;flex-basis:100%}
.v-catalogue .hd-r{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.v-catalogue .ico{width:16px;height:16px}
.v-catalogue .stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:14px}
.v-catalogue .st{padding:12px 16px;display:grid;gap:4px;align-content:start;min-width:0}
.v-catalogue .st-v{font-size:24px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.15}
.v-catalogue .st-v small{font-size:13px;font-weight:500;color:var(--ink-3);margin-left:4px;letter-spacing:0}
.v-catalogue .st-s{font-size:12px;color:var(--ink-3);overflow-wrap:anywhere}
.v-catalogue .sentinel{height:1px;margin-bottom:-1px}
.v-catalogue .fbar{position:sticky;top:8px;z-index:6;padding:10px 12px;border-radius:var(--r-lg) var(--r-lg) 0 0;background:color-mix(in srgb,var(--card) 90%,transparent);backdrop-filter:saturate(1.4) blur(10px);transition:box-shadow .25s,border-color .25s,border-radius .25s}
.v-catalogue .fbar.stuck{box-shadow:var(--shadow-lg);border-color:var(--line-2);border-radius:var(--r-lg)}
.v-catalogue .facets{padding:10px 12px 12px;margin-bottom:16px;border-top:0;border-radius:0 0 var(--r-lg) var(--r-lg);display:flex;align-items:center;gap:10px 14px;flex-wrap:wrap}
.v-catalogue .facets:empty{display:none}
.v-catalogue .frow{display:flex;align-items:center;gap:10px 14px;flex-wrap:wrap}
.v-catalogue .search{flex:1 1 260px;display:flex;align-items:center;gap:8px;border:1px solid var(--line-2);border-radius:10px;padding:0 6px 0 10px;background:var(--card);color:var(--ink-3);transition:border-color .2s,box-shadow .2s;min-width:0}
.v-catalogue .search:focus-within{border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}
.v-catalogue .search input{flex:1;border:0;outline:0;background:transparent;padding:8px 0;min-width:0;color:var(--ink)}
.v-catalogue .search .x{border:0;background:var(--sunk);border-radius:6px;padding:2px 8px;font-size:11.5px;color:var(--ink-3);opacity:0;pointer-events:none;transition:opacity .2s}
.v-catalogue .search.has .x{opacity:1;pointer-events:auto}
.v-catalogue .count{font-size:12.5px;color:var(--ink-3);white-space:nowrap}
.v-catalogue .count b{color:var(--ink);font-variant-numeric:tabular-nums;display:inline-block;min-width:2ch;text-align:right}
.v-catalogue .sort{display:flex;align-items:center;gap:8px;font-size:12.5px;color:var(--ink-3)}
.v-catalogue .sort select{border:1px solid var(--line-2);border-radius:9px;background:var(--card);padding:7px 10px;font-size:13px;color:var(--ink)}
.v-catalogue .grp{display:flex;align-items:center;gap:6px;flex-wrap:wrap;min-width:0}
.v-catalogue .grp > .eyebrow{margin-right:2px}
.v-catalogue .chip{padding:3px 9px;transition:background .2s,border-color .2s,color .2s,opacity .25s,transform .15s}
.v-catalogue .chip:active{transform:scale(.95)}
.v-catalogue .chip .n{font:11px var(--f-mono);color:var(--ink-3)}
.v-catalogue .chip.on .n{color:var(--accent)}
.v-catalogue .chip .tag{font:600 11px var(--f-mono)}
.v-catalogue .chip.zero{opacity:.42}
.v-catalogue .clear{transition:opacity .2s,width .2s}
.v-catalogue .clear.hide{display:none}
.v-catalogue .cgrid{position:relative;display:grid;grid-template-columns:repeat(auto-fill,minmax(176px,1fr));gap:14px}
.v-catalogue .cc{display:flex;flex-direction:column;text-align:left;padding:0;border:1px solid var(--line);border-radius:var(--r);background:var(--card);box-shadow:var(--shadow);overflow:hidden;color:inherit;transition:border-color .2s,box-shadow .25s,transform .25s var(--ease)}
.v-catalogue .cc[hidden]{display:none}
.v-catalogue button.cc:hover{border-color:var(--accent-line);box-shadow:var(--shadow-lg);transform:translateY(-3px)}
.v-catalogue .cc.sel{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft),var(--shadow)}
.v-catalogue .cc-img{position:relative;display:block;aspect-ratio:3/4;background:var(--sunk);overflow:hidden}
.v-catalogue .cc-img img,.v-catalogue .big img.fg{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transform:scale(1.06);transition:opacity .5s var(--ease),transform .7s var(--ease)}
.v-catalogue .cc-img img.in,.v-catalogue .big img.in{opacity:1;transform:none}
.v-catalogue button.cc:hover .cc-img img.in{transform:scale(1.05)}
.v-catalogue .ph{position:absolute;inset:0;border-radius:0;transition:opacity .45s}
.v-catalogue .ph.gone{opacity:0}
.v-catalogue .noimg::after{content:"No crop";position:absolute;inset:0;display:grid;place-items:center;font-size:12px;color:var(--ink-3)}
.v-catalogue .cc-dur{position:absolute;right:8px;bottom:8px;z-index:2;font:500 11px var(--f-mono);padding:2px 6px;border-radius:6px;background:color-mix(in srgb,var(--ink) 74%,transparent);color:var(--card)}
.v-catalogue .cc-b{display:grid;gap:6px;padding:10px 12px 12px;min-width:0}
.v-catalogue .cc-t{display:flex;align-items:baseline;gap:8px;justify-content:space-between;min-width:0}
.v-catalogue .cc-t b{font:600 13px var(--f-mono);white-space:nowrap;flex:none}
.v-catalogue .cc-t span{font-size:11.5px;color:var(--ink-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.v-catalogue .cc-d{font-size:12.5px;color:var(--ink-2);line-height:1.35;min-height:2.7em;overflow-wrap:anywhere}
.v-catalogue .cc-f{display:flex;align-items:center;gap:6px;min-width:0}
.v-catalogue .cc-f .cc-clk{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-catalogue .cc-f .swatch{margin:0}
.v-catalogue mark{background:var(--accent-soft);color:var(--accent);border-radius:3px;padding:0 1px}
.v-catalogue .cc-sw{display:flex;gap:10px;flex-wrap:wrap;font-size:11.5px;color:var(--ink-3)}
.v-catalogue .cc-sw span{display:inline-flex;align-items:center;gap:5px}
.v-catalogue .cc-clk{font:11px var(--f-mono);color:var(--ink-2)}
.v-catalogue .none{grid-column:1/-1}
.v-catalogue .scrim{position:fixed;inset:0;z-index:49;background:color-mix(in srgb,var(--ink) 22%,transparent);opacity:0;pointer-events:none;transition:opacity .35s var(--ease)}
.v-catalogue .scrim.open{opacity:1;pointer-events:auto}
.v-catalogue .drawer{position:fixed;top:0;right:0;bottom:0;z-index:50;width:min(460px,100%);background:var(--card);border-left:1px solid var(--line);box-shadow:var(--shadow-lg);display:flex;flex-direction:column;transform:translateX(104%);visibility:hidden;transition:transform .5s var(--ease),visibility 0s linear .5s}
.v-catalogue .drawer.open{transform:none;visibility:visible;transition:transform .5s var(--ease),visibility 0s}
.v-catalogue .dh{display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid var(--line)}
.v-catalogue .dh .grow{flex:1;min-width:0}
.v-catalogue .dh h3{font:600 18px var(--f-mono)}
.v-catalogue .db{flex:1;overflow:auto;padding:18px;display:grid;gap:20px;align-content:start}
.v-catalogue .hero{display:grid;grid-template-columns:150px minmax(0,1fr);gap:16px;align-items:start}
.v-catalogue .dh .muted{font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:45%}
.v-catalogue .big{position:relative;aspect-ratio:3/4;border-radius:var(--r);overflow:hidden;background:var(--sunk);border:1px solid var(--line)}
.v-catalogue .big img.bg{position:absolute;inset:-10%;width:120%;height:120%;object-fit:cover;filter:blur(14px);opacity:.35}
.v-catalogue .big img.fg{object-fit:contain}
.v-catalogue .lead{margin:0 0 10px;font-size:14px;font-weight:500}
.v-catalogue .sec{display:grid;gap:10px}
.v-catalogue .sec-h{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600}
.v-catalogue .sec-h span{margin-left:auto;font-weight:400;font-size:11.5px;color:var(--ink-3)}
.v-catalogue .pp{width:100%;height:auto;display:block}
.v-catalogue .pp-frame{fill:var(--sunk);stroke:var(--line-2)}
.v-catalogue .pp-grid{fill:none;stroke:var(--line);stroke-dasharray:3 4}
.v-catalogue .pp-box{fill:none;stroke:var(--accent-line)}
.v-catalogue .pp-line{fill:none;stroke:var(--accent);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1}
.v-catalogue .pp-dot{stroke:var(--ink-3);stroke-width:.6;opacity:0;animation:pop .3s var(--ease) both}
.v-catalogue .pp-in{fill:var(--ok);stroke:var(--card);stroke-width:2}
.v-catalogue .pp-out{fill:var(--bad);stroke:var(--card);stroke-width:2}
.v-catalogue .pp-mover{fill:var(--accent);stroke:var(--card);stroke-width:2}
.v-catalogue .pp text{font:10px var(--f-mono);fill:var(--ink-3)}
.v-catalogue .tstrip{position:relative;height:8px;border-radius:99px;background:var(--sunk);border:1px solid var(--line)}
.v-catalogue .tstrip i{position:absolute;top:-1px;bottom:-1px;width:0;border-radius:99px;background:var(--accent);transition:width .9s var(--ease)}
.v-catalogue .tlab{display:flex;justify-content:space-between;gap:8px;font:11px var(--f-mono);color:var(--ink-3)}
.v-catalogue .os{display:flex;align-items:center;gap:10px;padding:8px 10px 8px 8px;border:1px solid var(--line);border-radius:10px;background:var(--card);text-align:left;width:100%;transition:border-color .2s,background .2s}
.v-catalogue .os:hover{border-color:var(--accent-line);background:var(--sunk)}
.v-catalogue .os img{width:36px;height:48px;border-radius:6px;object-fit:cover;background:var(--sunk);flex:none}
.v-catalogue .os .grow{flex:1;display:grid;gap:1px;font-size:12px;color:var(--ink-3);min-width:0}
.v-catalogue .os .grow b{font:600 13px var(--f-mono);color:var(--ink)}
.v-catalogue .hop{font-size:12px;color:var(--ink-2);padding:10px 12px;border-radius:10px;background:var(--sunk);border:1px dashed var(--line-2);display:grid;gap:3px}
.v-catalogue .df{display:flex;gap:8px;padding:14px 18px;border-top:1px solid var(--line);background:var(--card)}
.v-catalogue .df .btn{flex:1}
@media (max-width:820px){
  .v-catalogue .fbar{top:0}
  .v-catalogue .drawer{bottom:auto;height:100vh}
}
@media (max-width:520px){.v-catalogue .hero{grid-template-columns:110px minmax(0,1fr)}}`;

  STM.view("catalogue", {
    label: "Catalogue",
    badge: () => String(Object.values(STM.scenes).reduce((a, s) => a + s.tracks.length, 0)),
    mount(root, ctx) {
      let dead = false, raf = 0, io = null, current = null, shown = 0, qT = 0, input = null, countEl = null, countBox = null, clearBtn = null, lastFocus = null;
      let items = [], chips = [], scenes = [];
      const timers = new Set();
      const later = (fn, ms) => { const id = setTimeout(() => { timers.delete(id); if (!dead) fn(); }, ms); timers.add(id); return id; };
      const opts = (ctx && ctx.opts) || {};
      const f = { q: "", scene: new Set(), cam: new Set(), top: new Set(), low: new Set(), from: "", to: "", pin: null, sort: "longest" };
      let showAll = false, hiddenN = 0;

      /* ---------- shell + skeletons ---------- */
      const wrap = el("div", { class: "v-catalogue" }, el("style", { html: CSS }));
      root.append(wrap);
      STM.pageHead(wrap, "Catalogue", "Every person sighting, across all scenes.");
      const stats = el("div", { class: "stats" });
      const sentinel = el("div", { class: "sentinel" });
      const bar = el("div", { class: "card fbar", id: "catalogue-filters" });
      const facets = el("div", { class: "card facets", id: "catalogue-facets" });
      const grid = el("div", { class: "cgrid", id: "catalogue-grid" });
      wrap.append(stats, sentinel, bar, facets, grid);
      for (let i = 0; i < 4; i++) stats.append(el("div", { class: "card st" }, STM.skeleton(12, "45%"), STM.skeleton(28, "35%", 6), STM.skeleton(10, "75%")));
      bar.append(el("div", { class: "frow" }, STM.skeleton(36, "100%", 10)));
      facets.append(STM.skeleton(24, "68%", 99));
      for (let i = 0; i < 12; i++) grid.append(el("div", { class: "cc" }, el("span", { class: "cc-img skel", style: { borderRadius: 0 } }),
        el("span", { class: "cc-b" }, STM.skeleton(12, "40%"), STM.skeleton(10, "92%"), STM.skeleton(10, "60%"))));

      /* ---------- drawer (built once so its hook ids are stable) ---------- */
      const dTitle = el("h3", {}, "");
      const dScene = el("span", { class: "muted" }, "");
      const dBody = el("div", { class: "db" });
      const simBtn = el("button", { class: "btn btn-primary", type: "button", id: "catalogue-similar", html: STM.icons.similar + "<span>Find similar</span>", onclick: () => {
        if (!current) return;
        STM.state.similarSeed = current.key;          // similar.js reads the "scene:track" key
        STM.go("similar", {});
      } });
      const tlBtn = el("button", { class: "btn", type: "button", id: "catalogue-timeline", html: STM.icons.timeline + "<span>Show on timeline</span>", onclick: () => {
        if (!current) return;
        const { s, t } = current;
        if (STM.sceneName !== s.name) {                // the timeline draws ctx.scene, so switch like the Data view does
          STM.sceneName = s.name;
          if (STM.desk && STM.desk.reset) try { STM.desk.reset(); } catch (e) { /* desk not mounted */ }
          STM.toast(`Switched to the ${esc(s.title)} scene`);
        }
        STM.state.focus = STM.state.timelineSeed = { scene: s.name, track: t.id, t: t.t0 };
        STM.go("timeline", { track: t.id, t: t.t0, from: "catalogue" });
      } });
      const deskBtn = el("button", { class: "btn btn-primary", type: "button", id: "catalogue-desk", html: "<span>Open on desk</span>", onclick: () => {
        if (!current) return;
        const { s, t } = current;
        if (STM.sceneName !== s.name) {
          STM.sceneName = s.name;
          if (STM.desk && STM.desk.reset) try { STM.desk.reset(); } catch (e) { /* desk not mounted */ }
        }
        STM.go("desk");
        setTimeout(() => { if (STM.sceneName === s.name && STM.desk && STM.desk.seek) try { STM.desk.seek(t.t0 + 0.5, t.cam); } catch (e) { /* ignore */ } }, 400);
      } });
      const closeBtn = el("button", { class: "icon-btn", type: "button", id: "catalogue-drawer-close", "aria-label": "Close detail", html: ICO_X, onclick: () => closeDrawer() });
      const drawer = el("aside", { class: "drawer", id: "catalogue-drawer", "aria-label": "Sighting detail", "aria-hidden": "true" },
        el("div", { class: "dh" }, el("div", { class: "grow" }, el("div", { class: "eyebrow" }, "Sighting"), dTitle), dScene, closeBtn),
        dBody, el("div", { class: "df" }, deskBtn, simBtn, tlBtn));
      const scrim = el("div", { class: "scrim", onclick: () => closeDrawer() });
      wrap.append(scrim, drawer);
      const onKey = e => { if (e.key === "Escape" && current) closeDrawer(); };
      document.addEventListener("keydown", onKey);

      /* ---------- filtering ---------- */
      const val = (it, g) => g === "scene" ? it.s.name : g === "cam" ? it.s.name + ":" + it.t.cam : g === "low" ? it.low : it.top;
      const hm = (s, t) => String(s.clock(t)).slice(0, 5);
      const inTime = it => (!f.from || hm(it.s, it.t.t1) >= f.from) && (!f.to || hm(it.s, it.t.t0) <= f.to);
      const pass = (it, ex) => (showAll || it.person) && (!f.q || it.hay.includes(f.q)) && (!f.pin || f.pin.has(it.key)) && inTime(it) &&
        ["scene", "cam", "top", "low"].every(g => g === ex || !f[g].size || f[g].has(val(it, g)));
      const sceneIdx = {};
      const SORT = {
        longest: (a, b) => b.dur - a.dur || a.t.t0 - b.t.t0 || a.key.localeCompare(b.key),
        first: (a, b) => a.t.t0 - b.t.t0 || b.dur - a.dur || a.key.localeCompare(b.key),
        camera: (a, b) => sceneIdx[a.s.name] - sceneIdx[b.s.name] || a.t.cam.localeCompare(b.t.cam) || a.t.t0 - b.t.t0,
      };
      const emptyEl = el("div", { class: "card empty none", hidden: true }, "No sightings match these filters. ",
        el("button", { class: "btn btn-sm", type: "button", onclick: () => clearAll() }, "Clear filters"));

      function endLeave(it) {
        it.leaving = false;
        it.el.getAnimations().forEach(a => a.cancel());
        Object.assign(it.el.style, { position: "", left: "", top: "", width: "", pointerEvents: "", zIndex: "" });
        it.el.hidden = true;
      }
      function apply(anim) {
        const list = items.filter(it => pass(it)).sort(SORT[f.sort] || SORT.longest);
        const show = new Set(list);
        for (const it of items) { it.el.getAnimations().forEach(a => a.cancel()); if (it.leaving) endLeave(it); }
        const first = new Map();
        for (const it of items) if (!it.el.hidden) first.set(it, [it.el.offsetLeft, it.el.offsetTop, it.el.offsetWidth]);
        const entering = [], leaving = [];
        for (const it of items) {
          const was = !it.el.hidden, now = show.has(it);
          if (was && !now) leaving.push(it);
          else if (!was && now) { it.el.hidden = false; entering.push(it); }
        }
        for (const it of list) grid.append(it.el);
        grid.append(emptyEl);
        for (const it of leaving) {
          const [x, y, w] = first.get(it); it.leaving = true;
          Object.assign(it.el.style, { position: "absolute", left: x + "px", top: y + "px", width: w + "px", pointerEvents: "none", zIndex: "0" });
        }
        if (!anim || STM.reduced) leaving.forEach(endLeave);
        else {
          for (const it of list) {
            const p = first.get(it); if (!p || entering.includes(it)) continue;
            const dx = p[0] - it.el.offsetLeft, dy = p[1] - it.el.offsetTop;
            if (dx || dy) it.el.animate([{ transform: `translate(${dx}px,${dy}px)` }, { transform: "none" }], { duration: 520, easing: EASE });
          }
          entering.forEach((it, i) => it.el.animate([{ opacity: 0, transform: "scale(.9)" }, { opacity: 1, transform: "none" }],
            { duration: 420, delay: 90 + Math.min(i * 28, 280), easing: EASE, fill: "backwards" }));
          leaving.forEach(it => {
            const a = it.el.animate([{ opacity: 1, transform: "none" }, { opacity: 0, transform: "scale(.88)" }], { duration: 230, easing: "ease-in", fill: "forwards" });
            a.onfinish = () => { if (it.leaving) endLeave(it); };
          });
        }
        const prev = grid.querySelector("#catalogue-first-card"); if (prev) prev.removeAttribute("id");
        if (list[0]) list[0].el.id = "catalogue-first-card";
        for (const it of list) it.desc.innerHTML = mark(it.t.desc || "no description", f.q);
        for (const c of chips) {
          const n = c.g === "pin" ? items.filter(it => pass(it)).length : items.filter(it => val(it, c.g) === c.v && pass(it, c.g)).length;
          c.n.textContent = n; c.el.classList.toggle("zero", !n && !c.el.classList.contains("on"));
        }
        const nOn = (f.q ? 1 : 0) + (f.pin ? 1 : 0) + (f.from ? 1 : 0) + (f.to ? 1 : 0) + f.scene.size + f.cam.size + f.top.size + f.low.size;
        clearBtn.classList.toggle("hide", !nOn); clearBtn.textContent = `Clear ${nOn} filter${nOn === 1 ? "" : "s"}`;
        emptyEl.hidden = list.length > 0;
        tweenCount(list.length);
        return list;
      }
      function tweenCount(to) {
        cancelAnimationFrame(raf);
        if (to !== shown && countBox) { countBox.classList.remove("flash"); void countBox.offsetWidth; countBox.classList.add("flash"); }
        if (STM.reduced) { shown = to; countEl.textContent = to; return; }
        const from = shown, t0 = performance.now();
        const step = now => {
          if (dead) return;
          const k = Math.min(1, (now - t0) / 520), e = 1 - Math.pow(1 - k, 3);
          shown = Math.round(from + (to - from) * e); countEl.textContent = shown;
          if (k < 1) raf = requestAnimationFrame(step);
        };
        raf = requestAnimationFrame(step);
      }
      function clearAll() {
        f.q = ""; f.pin = null; f.from = f.to = ""; f.scene.clear(); f.cam.clear(); f.top.clear(); f.low.clear();
        wrap.querySelectorAll(".tf input").forEach(x => { x.value = ""; });
        if (input) { input.value = ""; input.parentNode.classList.remove("has"); }
        chips.forEach(c => { c.el.classList.remove("on"); c.el.setAttribute("aria-pressed", "false"); if (c.g === "pin") c.el.remove(); });
        chips = chips.filter(c => c.g !== "pin");
        apply(true);
      }

      /* ---------- cards ---------- */
      function makeCard(it) {
        const { s, t, cam, look } = it;
        const ph = el("span", { class: "ph skel" });
        const img = el("img", { alt: `Crop of sighting ${t.id}`, decoding: "async", draggable: "false" });
        const box = el("span", { class: "cc-img" }, ph, img, el("span", { class: "cc-dur" }, sec(it.dur)));
        img.addEventListener("load", () => { img.classList.add("in"); ph.classList.add("gone"); });
        img.addEventListener("error", () => { ph.classList.add("gone"); box.classList.add("noimg"); });
        img.src = s.cropUrl(t);
        it.desc = el("span", { class: "cc-d" }, t.desc || "no description");
        const dot = (c, part) => el("i", { class: "swatch", title: `${part}: ${c || "unknown"}`, style: { background: hex(c) } });
        return el("button", { class: "cc", type: "button", "data-key": it.key, "aria-label": `${t.id}, ${t.desc || "no description"}, camera ${cam.name}`, onclick: e => { lastFocus = e.currentTarget; openDrawer(it); } },
          box,
          el("span", { class: "cc-b" },
            el("span", { class: "cc-t" }, el("b", {}, t.id), el("span", { title: `${short(s)}, camera ${cam.id}` }, cam.name)),
            it.desc,
            el("span", { class: "cc-f" }, el("span", { class: "cc-clk" }, `${s.clock(t.t0)}–${s.clock(t.t1)}`), dot(look.top, "Top"), dot(look.bottom, "Lower body"))));
      }

      /* ---------- stats ---------- */
      function renderStats() {
        const idents = scenes.reduce((a, s) => a + (s.identities || []).length, 0);
        const linked = scenes.reduce((a, s) => a + (s.identities || []).filter(i => i.tracks.length > 1).length, 0);
        const cams = scenes.reduce((a, s) => a + s.cameras.length, 0);
        const camS = scenes.reduce((a, s) => a + camSeconds(s), 0);
        const per = scenes.map(s => ({ s, n: items.filter(i => i.s === s).length }));
        const num = (n, o, unit) => { const b = el("span", {}, "0"); STM.countUp(b, n, Object.assign({ ms: 800 }, o)); return el("div", { class: "st-v" }, b, unit ? el("small", {}, unit) : null); };
        const tile = (label, v, sub) => el("div", { class: "card st" }, el("span", { class: "eyebrow" }, label), v, el("div", { class: "st-s", title: sub }, sub));
        const mins = camS >= 90;
        stats.replaceChildren(
          tile("Sightings", num(items.length), per.map(p => `${short(p.s)} ${p.n}`).join(" · ")),
          tile("People", num(idents), `${linked} seen on more than one camera`),
          tile("Cameras", num(cams), `in ${scenes.length} scene${scenes.length === 1 ? "" : "s"}`),
          tile("Footage", num(mins ? camS / 60 : camS, { dec: 1 }, mins ? "min" : "s"), "summed over all cameras"));
      }

      /* ---------- filter bar ---------- */
      function chip(g, v, kids, title) {
        const n = el("span", { class: "n" }, "0");
        const b = el("button", { class: "chip", type: "button", id: `catalogue-chip-${g}-${v.replace(/[^a-z0-9]+/gi, "-")}`.toLowerCase(), "aria-pressed": "false", title: title || null }, ...kids, n);
        const c = { g, v, el: b, n };
        b.addEventListener("click", () => {
          if (g === "pin") { f.pin = null; b.remove(); chips = chips.filter(x => x !== c); apply(true); return; }
          const set = f[g], on = !set.has(v);
          on ? set.add(v) : set.delete(v);
          b.classList.toggle("on", on); b.setAttribute("aria-pressed", String(on));
          apply(true);
        });
        chips.push(c); return b;
      }
      function renderBar() {
        input = el("input", { id: "catalogue-search", type: "text", placeholder: "Search descriptions, e.g. red top", autocomplete: "off", spellcheck: "false", "aria-label": "Search sightings" });
        const sbox = el("label", { class: "search" }, el("span", { html: ICO_SEARCH, style: { display: "grid" } }), input,
          el("button", { class: "x", type: "button", onclick: e => { e.preventDefault(); input.value = ""; input.dispatchEvent(new Event("input")); input.focus(); } }, "Clear"));
        input.addEventListener("input", () => {
          sbox.classList.toggle("has", !!input.value);
          clearTimeout(qT); timers.delete(qT);
          qT = later(() => { f.q = norm(input.value); apply(true); }, 140);
        });
        input.addEventListener("keydown", e => { if (e.key === "Enter") { const it = items.find(x => x.el.id === "catalogue-first-card"); if (it && !it.el.hidden) openDrawer(it); } });
        const sel = el("select", { id: "catalogue-sort", "aria-label": "Sort sightings" },
          ...[["longest", "Longest seen"], ["first", "First seen"], ["camera", "Camera"]].map(([v, l]) => el("option", { value: v }, l)));
        sel.addEventListener("change", () => { f.sort = sel.value; apply(true); });
        countEl = el("b", { id: "catalogue-count" }, "0");
        countBox = el("div", { class: "count", "aria-live": "polite" }, countEl, ` of ${items.filter(it => showAll || it.person).length} ${showAll ? "detections" : "people"}`);
        clearBtn = el("button", { class: "btn btn-ghost btn-sm clear hide", type: "button", id: "catalogue-clear", onclick: () => clearAll() }, "Clear filters");
        chips = [];
        const tops = {}, lows = {}; items.forEach(it => { if (it.top) tops[it.top] = (tops[it.top] || 0) + 1; if (it.low) lows[it.low] = (lows[it.low] || 0) + 1; });
        const colGrp = (g, label, m) => el("div", { class: "grp" }, el("span", { class: "eyebrow" }, label),
          ...Object.keys(m).sort((a, b) => m[b] - m[a] || a.localeCompare(b)).map(c => chip(g, c, [el("i", { class: "swatch", style: { background: hex(c) } }), cap(c)])));
        const tIn = k => { const x = el("input", { type: "time", "aria-label": "Time " + k }); x.value = f[k]; x.addEventListener("change", () => { f[k] = x.value.slice(0, 5); apply(true); }); return x; };
        const timeGrp = el("div", { class: "grp tf" }, el("span", { class: "eyebrow" }, "Time"), tIn("from"), el("span", { class: "muted" }, "to"), tIn("to"));
        const hidLine = el("div", { class: "grp muted hidline" }, hiddenN ? `${hiddenN} non-person detection${hiddenN === 1 ? "" : "s"} ${showAll ? "shown" : "hidden"}` : "",
          hiddenN ? el("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => { showAll = !showAll; renderBar(); apply(true); } }, showAll ? "Hide them" : "Show them") : null);
        const sceneGrp = el("div", { class: "grp" }, el("span", { class: "eyebrow" }, "Scene"), ...scenes.map(s => chip("scene", s.name, [short(s)], s.title)));
        const camGrp = el("div", { class: "grp" }, el("span", { class: "eyebrow" }, "Camera"),
          ...scenes.flatMap(s => s.cameras.map(c => chip("cam", s.name + ":" + c.id, [el("span", { class: "tag" }, c.id), c.name], `${short(s)}, camera ${c.id}`))));
        const topGrp = colGrp("top", "Top colour", tops), lowGrp = colGrp("low", "Lower colour", lows);
        for (const c of chips) if (f[c.g] && f[c.g].has && f[c.g].has(c.v)) { c.el.classList.add("on"); c.el.setAttribute("aria-pressed", "true"); }
        const pinWrap = el("div", { class: "grp" });
        if (opts.tracks && opts.tracks.length && ctx.scene) {
          f.pin = new Set(opts.tracks.map(id => ctx.scene.name + ":" + id));
          const pc = chip("pin", "data", [`From ${opts.from === "data" ? "Data" : "query"}: ${opts.tracks.length} in ${short(ctx.scene)}`, el("span", { html: ICO_X, style: { display: "grid" } })], "Remove this filter");
          pc.classList.add("on"); pinWrap.append(pc);
        }
        bar.replaceChildren(el("div", { class: "frow" }, sbox, clearBtn, el("label", { class: "sort" }, "Sort", sel), countBox));
        facets.replaceChildren(...(pinWrap.childNodes.length ? [pinWrap] : []), sceneGrp, camGrp, topGrp, lowGrp, timeGrp, hidLine);
        bar.classList.add("pop"); facets.classList.add("pop");
      }

      /* ---------- drawer ---------- */
      function pathSvg(it) {
        const { s, t, cam, look } = it;
        const [fw, fh] = frameSize(s, t.cam), W = 360, H = Math.round(W * fh / fw);
        const pts = t.boxes.map(b => [(b[1] + b[3] / 2) * W, (b[2] + b[4] / 2) * H]);
        if (!pts.length) return '<div class="empty">No boxes recorded.</div>';
        const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
        const step = Math.max(1, Math.ceil(t.boxes.length / 8));
        const ghost = t.boxes.filter((b, i) => i % step === 0).map(b => `<rect class="pp-box" x="${(b[1] * W).toFixed(1)}" y="${(b[2] * H).toFixed(1)}" width="${(b[3] * W).toFixed(1)}" height="${(b[4] * H).toFixed(1)}" rx="2"/>`).join("");
        const dots = pts.map((p, i) => `<circle class="pp-dot" cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.2" fill="${hex(look.top)}" style="animation-delay:${(0.25 + i / pts.length * 1.1).toFixed(2)}s"/>`).join("");
        const a = pts[0], z = pts[pts.length - 1], mv = Math.max(2.5, Math.min(8, it.dur / 4)).toFixed(1);
        return `<svg class="pp" viewBox="-6 -6 ${W + 12} ${H + 12}" role="img" aria-label="Path of ${esc(t.id)} in camera ${esc(cam.id)}">
          <rect class="pp-frame" x="0" y="0" width="${W}" height="${H}" rx="8"/>
          <path class="pp-grid" d="M${W / 3} 0V${H}M${2 * W / 3} 0V${H}M0 ${H / 3}H${W}M0 ${2 * H / 3}H${W}"/>${ghost}
          <path class="pp-line" pathLength="1" d="${d}"/>${dots}
          <circle class="pp-in" cx="${a[0].toFixed(1)}" cy="${a[1].toFixed(1)}" r="5"/><circle class="pp-out" cx="${z[0].toFixed(1)}" cy="${z[1].toFixed(1)}" r="5"/>
          <circle class="pp-mover" r="4.5" opacity="0"><set attributeName="opacity" to="1" begin="1.4s"/><animateMotion dur="${mv}s" begin="1.4s" repeatCount="indefinite" path="${d}"/></circle>
          <text x="10" y="18">CAM ${esc(cam.id)} · ${esc(cam.name)}</text><text x="${W - 10}" y="${H - 10}" text-anchor="end">${fw}×${fh}</text></svg>`;
      }
      function openDrawer(it) {
        if (dead) return;
        if (current && current.el) current.el.classList.remove("sel");
        current = it; it.el.classList.add("sel");
        const { s, t, cam, look, ident } = it;
        dTitle.textContent = t.id; dScene.textContent = short(s);
        const ph = el("span", { class: "ph skel" }), bg = el("img", { class: "bg", alt: "" }), fg = el("img", { class: "fg", alt: `Crop of sighting ${t.id}` });
        fg.addEventListener("load", () => { fg.classList.add("in"); ph.classList.add("gone"); });
        fg.addEventListener("error", () => ph.classList.add("gone"));
        bg.src = fg.src = s.cropUrl(t);
        const kv = el("dl", { class: "kv" });
        const row = (k, v) => kv.append(el("dt", {}, k), el("dd", {}, v));
        row("Scene", short(s));
        row("Camera", `${cam.id} · ${cam.name}`);
        row("Identity", ident ? `${ident.id}${ident.tracks.length > 1 ? ` · ${ident.tracks.length} sightings` : ""}` : "not linked");
        row("First seen", el("span", { class: "mono" }, s.clock(t.t0)));
        row("Last seen", el("span", { class: "mono" }, s.clock(t.t1)));
        row("Duration", sec(it.dur));
        row("Look", el("span", { class: "cc-sw" }, sw(look.top, "top"), sw(look.bottom, "lower body")));
        const seg = el("i", { style: { left: (t.t0 / s.duration * 100).toFixed(2) + "%" } });
        const others = ident ? ident.tracks.filter(id => id !== t.id).map(id => items.find(x => x.s === s && x.t.id === id)).filter(Boolean) : [];
        const alike = items.filter(x => x !== it && x.look.top === look.top && x.look.bottom === look.bottom).length;
        const otherSec = el("div", { class: "sec" }, el("div", { class: "sec-h" }, "Same person, other sightings", el("span", {}, ident ? `Identity ${ident.id}` : "")));
        if (others.length) {
          for (const o of others) otherSec.append(el("button", { class: "os", type: "button", onclick: () => openDrawer(o) },
            el("img", { src: o.s.cropUrl(o.t), alt: `Crop of sighting ${o.t.id}` }),
            el("span", { class: "grow" }, el("b", {}, o.t.id), `Camera ${o.cam.id} · ${o.cam.name}`, el("span", { class: "mono" }, `${o.s.clock(o.t.t0)}–${o.s.clock(o.t.t1)}`)),
            el("span", { class: "muted", html: STM.icons.arrow })));
          for (const h of ident.hops || []) otherSec.append(el("div", { class: "hop" },
            el("b", { class: "mono" }, `${h.from} → ${h.to}`), `Follow score ${Number(h.score).toFixed(2)}, gap ${sec(Number(h.gap) || 0)}`, el("span", { class: "muted" }, h.why || "")));
        } else otherSec.append(el("div", { class: "hop" }, "No other sightings. The follow step linked none to this one."));
        otherSec.append(el("div", { class: "muted", style: { fontSize: "12px" } },
          alike ? `${alike} other sighting${alike > 1 ? "s" : ""} in the catalogue share ${look.top} top and ${look.bottom} lower body. Find similar ranks them by crop colour.` : "No other sighting in the catalogue shares this exact look."));
        dBody.replaceChildren(
          el("div", { class: "hero" }, el("div", { class: "big" }, bg, ph, fg), el("div", {}, el("p", { class: "lead" }, cap(t.desc || "no description")), kv)),
          el("div", { class: "sec" }, el("div", { class: "sec-h" }, "Path in frame"),
            el("div", { html: pathSvg(it) }),
            el("div", { class: "tstrip", title: "When in the clip" }, seg),
            el("div", { class: "tlab" }, el("span", {}, s.clock(0)), el("span", {}, `${sec(t.t0)} to ${sec(t.t1)} of ${sec(s.duration)}`), el("span", {}, s.clock(s.duration)))),
          otherSec);
        dBody.scrollTop = 0;
        dBody.classList.remove("pop"); void dBody.offsetWidth; dBody.classList.add("pop");
        const line = dBody.querySelector(".pp-line");
        if (line) STM.reduced ? (line.style.strokeDashoffset = "0") : line.animate([{ strokeDashoffset: "1" }, { strokeDashoffset: "0" }], { duration: 1300, delay: 250, easing: EASE, fill: "forwards" });
        later(() => { seg.style.width = (it.dur / s.duration * 100).toFixed(2) + "%"; }, 120);
        if (matchMedia("(max-width:820px)").matches) {
          const y = (document.scrollingElement || document.documentElement).scrollTop + "px"; drawer.style.top = y;
        } else drawer.style.top = "";
        drawer.classList.add("open"); scrim.classList.add("open"); drawer.setAttribute("aria-hidden", "false");
        STM.state.catalogueFocus = { scene: s.name, track: t.id };
        try { closeBtn.focus({ preventScroll: true }); } catch (e) { /* focus is a nicety */ }
      }
      function closeDrawer() {
        if (current && current.el) current.el.classList.remove("sel");
        current = null;
        drawer.classList.remove("open"); scrim.classList.remove("open"); drawer.setAttribute("aria-hidden", "true");
        if (lastFocus && lastFocus.isConnected) try { lastFocus.focus({ preventScroll: true }); } catch (e) { /* ignore */ }
      }

      /* ---------- load every scene, then reveal ---------- */
      const api = {
        open: (scene, track) => { const it = items.find(x => x.s.name === scene && x.t.id === track); if (it) openDrawer(it); return !!it; },
        search: q => { if (input) { input.value = q; input.dispatchEvent(new Event("input")); } },
        close: () => closeDrawer(),
      };
      STM.catalogue = api;
      const names = ["meva", "lab", "caviar", ...((STM.sceneList || []).map(x => Array.isArray(x) ? x[0] : x && x.name).filter(Boolean))];
      Promise.all([...new Set(names)].map(n => Promise.resolve().then(() => STM.loadScene(n)).catch(() => null)).concat(STM.sleep(300))).then(() => {
        if (dead) return;
        scenes = Object.values(STM.scenes).filter(s => s && s.tracks);
        if (ctx.scene && scenes.includes(ctx.scene)) scenes = [ctx.scene, ...scenes.filter(s => s !== ctx.scene)];
        if (!scenes.length) throw new Error("no scenes");
        scenes.forEach((s, i) => { sceneIdx[s.name] = i; });
        items = [];
        for (const s of scenes) {
          const idOf = {};
          for (const i of s.identities || []) for (const tid of i.tracks) idOf[tid] = i;
          for (const t of s.tracks) {
            if (t.kind && t.kind !== "person") continue;
            const cam = s.cam(t.cam) || { id: t.cam, name: "Camera " + t.cam };
            const look = t.look || {}, cw = colourWords(t.desc);
            items.push({ key: s.name + ":" + t.id, s, t, cam, look, dur: Math.max(0, t.t1 - t.t0), ident: idOf[t.id] || null,
              top: cw[0] || look.top || "", low: cw[1] || look.bottom || "", person: isPerson(s, t),
              hay: norm([t.desc, t.id, cam.name, short(s), s.name].join(" ")) });
          }
        }
        hiddenN = items.filter(it => !it.person).length;
        renderStats();
        renderBar();
        grid.replaceChildren();
        items.forEach(it => { it.el = makeCard(it); grid.append(it.el); });
        const list = apply(false);
        list.forEach((it, i) => {
          const delay = Math.min(i * 30, 480);
          if (!STM.reduced) it.el.animate([{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }], { duration: 420, delay, easing: EASE, fill: "backwards" });
        });
        const vw = document.getElementById("viewWrap");
        if ("IntersectionObserver" in window) {
          io = new IntersectionObserver(([e]) => bar.classList.toggle("stuck", !e.isIntersecting), { root: vw && vw.scrollHeight > vw.clientHeight ? vw : null });
          io.observe(sentinel);
        }
        if (opts.q) api.search(String(opts.q));
        const want = opts.open || (opts.track && ctx.scene ? { scene: ctx.scene.name, track: opts.track } : null);
        if (want) later(() => api.open(want.scene, want.track), 750);
      }).catch(e => {
        if (dead) return;
        grid.replaceChildren(el("div", { class: "card empty none" }, `Could not load the index (${e.message}).`));
      });

      return () => {
        dead = true;
        timers.forEach(clearTimeout); timers.clear();
        cancelAnimationFrame(raf);
        document.removeEventListener("keydown", onKey);
        if (io) io.disconnect();
        if (STM.catalogue === api) delete STM.catalogue;
      };
    },
  });
})();
