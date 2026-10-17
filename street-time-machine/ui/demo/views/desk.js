/* Desk: the Find & Follow hero. Camera player with live boxes, search steps, matches / route / moments.
   STM.desk holds the actions the agent's tools call (search, pick, reject, route, moment, near, follow, reset). */
(function () {
  "use strict";
  const { el, esc, sleep, icons } = STM;

  /* ---------- search over the index (clothing words only) ---------- */
  const TOP_W = /\b(top|shirt|t-?shirt|tee|hoodie|jumper|sweater|jacket|coat|blouse|vest|hi-?vis|polo|cardigan)\b/;
  const BOT_W = /\b(trousers|pants|jeans|shorts|skirt|leggings|lower|bottoms?|joggers|chinos)\b/;
  const COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "grey", "gray", "navy", "beige", "dark", "light"];
  const CSS_COL = { red: "#d33", orange: "#f08a24", yellow: "#f2d02b", green: "#2e9e4f", blue: "#2f5bea", purple: "#8a4fd0", pink: "#ee7fb3", brown: "#8b5a2b", black: "#111", white: "#f4f4f4", grey: "#8a8a8a", gray: "#8a8a8a" };
  const ALIAS = { gray: "grey", navy: "blue", beige: "brown", dark: "black", light: "white", maroon: "red", tan: "brown" };
  const NEAR = { red: ["orange", "pink", "brown"], orange: ["red", "yellow", "brown"], yellow: ["orange"], pink: ["red", "purple"], purple: ["pink", "blue"], blue: ["purple", "black"], brown: ["orange", "red", "black"], black: ["grey", "blue", "brown"], grey: ["white", "black"], white: ["grey"], green: ["yellow"] };
  function parseLook(text) {
    const t = " " + text.toLowerCase().replace(/[^a-z\- ]/g, " ") + " ";
    const words = t.split(/\s+/).filter(Boolean), look = { top: null, bottom: null, ignored: [], assumed: [] };
    let pending = null;
    for (let i = 0; i < words.length; i++) {
      const w = words[i];
      if (COLOURS.includes(w)) { pending = ALIAS[w] || w; continue; }
      if (TOP_W.test(w) && pending) { look.top = pending; pending = null; continue; }
      if (BOT_W.test(w) && pending) { look.bottom = pending; pending = null; continue; }
      if (w === "jeans" && !look.bottom) { look.bottom = "blue"; look.assumed.push("jeans read as blue"); continue; }
      if (/^hi-?vis$/.test(w) && !look.top) { look.top = "yellow"; look.assumed.push("hi-vis read as yellow"); continue; }
      if (/^(hat|cap|helmet|hard-?hat|glasses|beard|face|tall|short|bald|bag|backpack|phone)$/.test(w)) look.ignored.push(w);
    }
    if (pending && !look.top) look.top = pending;      // "someone in red" -> top colour
    else if (pending && !look.bottom) look.bottom = pending;
    return look;
  }
  const part = (want, got) => !want ? null : got === want ? 1 : (NEAR[want] || []).includes(got) ? 0.55 : 0;
  const tier = sc => sc >= 0.85 ? "strong" : sc >= 0.6 ? "possible" : "weak";

  /* time window from tool args: {a, b} scene seconds and a camera letter, each optional */
  function windowFrom(args = {}) {
    const s = STM.scene(), w = {};
    for (const [k, key] of [["after", "a"], ["before", "b"]]) {
      if (!args[k]) continue;
      const t = s.toT(args[k]);
      if (t == null) (w.bad = w.bad || []).push(args[k]); else w[key] = t;
    }
    if (args.camera) { const c = String(args.camera).trim().toUpperCase().replace(/^CAM(ERA)?\s*/, ""); if (s.cam(c)) w.cam = c; else (w.bad = w.bad || []).push("camera " + args.camera); }
    if (w.a != null && w.b != null && w.a > w.b) [w.a, w.b] = [w.b, w.a];
    w.outside = (w.a != null && w.a > s.duration) || (w.b != null && w.b < 0);
    return w;
  }
  const hasWin = w => !!w && (w.a != null || w.b != null || !!w.cam);
  const inWin = (t, w) => !hasWin(w) || ((w.a == null || t.t1 >= w.a) && (w.b == null || t.t0 <= w.b) && (!w.cam || t.cam === w.cam));
  function winText(w) {
    if (!hasWin(w)) return "";
    const s = STM.scene(), parts = [];
    if (w.cam) parts.push(`Camera ${w.cam}`);
    if (w.a != null && w.b != null) parts.push(`${s.clock(Math.max(0, w.a))}–${s.clock(Math.min(s.duration, w.b))}`);
    else if (w.a != null) parts.push(`after ${s.clock(w.a)}`);
    else if (w.b != null) parts.push(`before ${s.clock(w.b)}`);
    return parts.join(" · ");
  }
  function winPhrase(w) {                        // the same window as words: "on Camera B between 16:51:00 and 16:52:00"
    if (!hasWin(w)) return "";
    const s = STM.scene(), parts = [];
    if (w.cam) parts.push(`on Camera ${w.cam}`);
    if (w.a != null && w.b != null) parts.push(`between ${s.clock(Math.max(0, w.a))} and ${s.clock(Math.min(s.duration, w.b))}`);
    else if (w.a != null) parts.push(`after ${s.clock(w.a)}`);
    else if (w.b != null) parts.push(`before ${s.clock(w.b)}`);
    return parts.join(" ");
  }
  function rank(look) {
    const s = STM.scene();
    return s.tracks.map(t => {
      const a = part(look.top, t.look.top), b = part(look.bottom, t.look.bottom);
      const parts = [a, b].filter(v => v != null);
      if (!parts.length) return null;
      const dur = t.t1 - t.t0, score = parts.reduce((x, y) => x + y, 0) / parts.length * (0.85 + 0.15 * Math.min(1, dur / 30));
      return { t, score };
    }).filter(r => r && r.score >= 0.35).sort((x, y) => y.score - x.score);
  }
  /* text (or a look read from a photo) -> ranked matches inside the time window; relaxes one colour before giving up */
  function search(text, refine, args = {}) {
    const s = STM.scene(), prev = STM.state.look || {};
    const look = typeof text === "object" ? { ignored: [], assumed: [], ...text } : parseLook(text);
    if (refine) { look.top = look.top || prev.top; look.bottom = look.bottom || prev.bottom; }
    const t0 = performance.now(), w = args.window || windowFrom(args);
    let all = rank(look), inside = all.filter(r => inWin(r.t, w)), relaxed = null;
    if (!inside.length && look.top && look.bottom) {
      for (const drop of ["bottom", "top"]) {
        const r2 = rank({ ...look, [drop]: null }).filter(r => inWin(r.t, w));
        if (r2.length) { inside = r2.map(r => ({ ...r, score: r.score * 0.8 })); relaxed = drop === "bottom" ? `Nobody had both colours, so these match the top (${look.top}) only` : `Nobody had both colours, so these match the lower (${look.bottom}) only`; break; }
      }
    }
    const results = inside.slice(0, 6);
    const outside = all.filter(r => !inWin(r.t, w));
    Object.assign(STM.state, { query: typeof text === "string" ? text : (look.photo ? "Photo reference" : STM.state.query), look, window: w, winArgs: args.window ? STM.state.winArgs : args,
      candidates: results.map(r => r.t.id), scores: Object.fromEntries(results.map(r => [r.t.id, r.score])), subject: null, lead: null, followed: false, phase: results.length ? "confirm" : "describe" });
    STM.emit("desk", { kind: "search" });
    return { look, results, relaxed, outside, window: w, ms: performance.now() - t0, scanned: s.tracks.length, cams: s.cameras.length };
  }
  /* narrow the matches on screen: keep the same look, merge the new window into the current one */
  function narrow(args = {}) {
    if (!STM.state.look) return null;
    const merged = { ...(STM.state.winArgs || {}) };
    for (const k of ["after", "before", "camera"]) if (args[k] !== undefined) merged[k] = args[k] || undefined;
    return search(STM.state.look, false, merged);
  }
  /* time buckets and cameras for a set of tracks: what the chat offers as narrowing chips */
  function buckets(ids) {
    const s = STM.scene(), tr = ids.map(id => s.track(id)).filter(Boolean);
    const step = [10, 15, 30, 60, 120, 300, 600, 1800, 3600].find(x => s.duration / x <= 6) || 3600;
    const out = [];
    for (let a = 0; a < s.duration; a += step) {
      const b = Math.min(s.duration, a + step), n = tr.filter(t => t.t1 >= a && t.t0 <= b).length;
      if (n && b - a >= step / 3) out.push({ a, b, n, label: `${s.clock(a).slice(0, step >= 60 ? 5 : 8)}–${s.clock(b).slice(0, step >= 60 ? 5 : 8)}` });
    }
    const cams = s.cameras.map(c => ({ cam: c.id, n: tr.filter(t => t.cam === c.id).length })).filter(c => c.n);
    return { times: out, cams, step };
  }
  function count(args = {}) {
    const s = STM.scene(), w = windowFrom(args), tr = s.tracks.filter(t => inWin(t, w));
    const people = new Set(tr.map(t => (s.identities.find(i => i.tracks.includes(t.id)) || { id: t.id }).id)).size;
    const perCam = s.cameras.map(c => ({ cam: c.id, name: c.name, n: tr.filter(t => t.cam === c.id).length }));
    let busiest = null;
    for (let t = Math.max(0, w.a || 0); t <= Math.min(s.duration, w.b ?? s.duration); t += 2) {
      const n = tr.filter(x => x.t0 <= t && x.t1 >= t).length;
      if (!busiest || n > busiest.n) busiest = { t, n };
    }
    return { w, tracks: tr, people, perCam, busiest };
  }
  /* sightings of the same person on other cameras: the follow step's identity links, then same-colour sightings */
  function route(id) {
    const s = STM.scene(), t = s.track(id); if (!t) return [];
    const ident = s.identities.find(i => i.tracks.includes(id));
    const linked = new Set(ident ? ident.tracks : [id]);
    const legs = [...linked].map(x => ({ t: s.track(x), how: x === id ? "confirmed" : "follow link" }));
    for (const o of s.tracks) {
      if (linked.has(o.id) || o.cam === t.cam) continue;
      if (o.look.top === t.look.top && o.look.bottom === t.look.bottom && (o.t1 - o.t0) > 4) legs.push({ t: o, how: "same colours" });
    }
    return legs.filter(l => l.t).sort((a, b) => a.t.t0 - b.t.t0);
  }
  function lastSighting(id) {
    const legs = route(id); if (!legs.length) return null;
    const last = legs.reduce((a, b) => (b.t.t1 > a.t.t1 ? b : a));
    return { track: last.t, t: last.t.t1, box: last.t.boxes[last.t.boxes.length - 1] };
  }
  function near(id) {        // who shared a camera with the subject in the 10 s after they were last seen there
    const s = STM.scene(), last = lastSighting(id); if (!last) return [];
    return s.tracks.filter(o => o.cam === last.track.cam && o.id !== last.track.id && o.t1 >= last.t - 15 && o.t0 <= last.t + 10)
      .map(o => ({ t: o, gap: +(o.t0 - last.t).toFixed(1) })).slice(0, 4);
  }

  STM.desk = {
    parseLook, search, narrow, buckets, count, windowFrom, winText, winPhrase, hasWin, tier, route, lastSighting, near,
    pick(n) {
      const id = STM.state.candidates[(n || 1) - 1]; if (!id) return null;
      STM.state.subject = id; STM.state.phase = "route";
      const l = lastSighting(id); STM.state.lead = l ? l.track.id : null;
      STM.emit("desk", { kind: "pick", id }); return STM.scene().track(id);
    },
    reject(n) {
      const i = (n || 1) - 1, id = STM.state.candidates[i]; if (!id) return null;
      STM.state.candidates.splice(i, 1); STM.emit("desk", { kind: "reject", id }); return id;
    },
    reset() { Object.assign(STM.state, { query: "", look: null, window: null, winArgs: {}, candidates: [], scores: {}, subject: null, lead: null, followed: false, phase: "describe" }); STM.emit("desk", { kind: "reset" }); },
    seek(t, cam) { STM.emit("seek", { t, cam }); },
  };

  STM.sceneList = STM.sceneList || [["meva", "Campus walkway", "real"], ["caviar", "Shopping corridor", "real"], ["lab", "Lab floor", "synthetic"]];

  /* ---------- the view ---------- */
  STM.view("desk", {
    label: "Find & Follow", live: true,
    async mount(root) {
      let s = STM.scene();
      const style = el("style", {}, `
.v-desk .cols{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(300px,1fr);gap:20px;align-items:start}
@media (max-width:1000px){.v-desk .cols{grid-template-columns:1fr}}
.v-desk .player{position:relative;border-radius:var(--r-lg);overflow:hidden;background:var(--video-bg);aspect-ratio:16/9;box-shadow:var(--shadow)}
.v-desk .player video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:var(--video-bg)}
.v-desk .player canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.v-desk .pip{position:absolute;right:14px;top:14px;width:26%;aspect-ratio:16/9;border-radius:10px;overflow:hidden;border:2px solid rgba(255,255,255,.7);box-shadow:0 8px 24px rgba(0,0,0,.4);cursor:pointer;transition:transform .3s var(--ease);background:#000;z-index:3}
.v-desk .pip:hover{transform:scale(1.04)}
.v-desk .pip video,.v-desk .pip canvas{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
.v-desk .pip span{position:absolute;left:6px;bottom:5px;color:#fff;font:600 10.5px var(--f-body);background:rgba(0,0,0,.55);padding:1px 6px;border-radius:5px}
.v-desk .hud{position:absolute;left:0;right:0;bottom:0;padding:40px 16px 12px;background:linear-gradient(transparent,rgba(5,8,16,.85));color:#fff;z-index:2}
.v-desk .hud .meta{display:flex;justify-content:space-between;align-items:center;font-size:12.5px;margin-bottom:8px;gap:10px}
.v-desk .hud .meta b{font-weight:600}
.v-desk .bar{position:relative;height:5px;border-radius:99px;background:rgba(255,255,255,.22);cursor:pointer}
.v-desk .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:inherit;background:#fff}
.v-desk .bar .mk{position:absolute;top:-3px;height:11px;border-radius:3px;background:var(--accent);opacity:.85}
.v-desk .bar .mk.subj{background:var(--ok)}
.v-desk.lean .cols{grid-template-columns:minmax(0,1.3fr) minmax(320px,1fr)}
.v-desk.lean .match{grid-template-columns:52px minmax(0,1fr)}
.v-desk.lean .match > .row:last-child{grid-column:2;justify-content:flex-start;margin-top:-2px}
.v-desk.lean .match img{width:52px;height:68px}
.v-desk .bar .win{position:absolute;top:-6px;bottom:-6px;border-radius:5px;background:rgba(255,255,255,.16);box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.75);animation:pop .35s var(--ease)}
.v-desk .winbar{display:flex;align-items:center;gap:8px;padding:8px 10px;margin-bottom:8px;border-radius:10px;background:var(--accent-soft);color:var(--accent);font-size:12.5px}
.v-desk .winbar b{font-weight:600}
.v-desk .winbar button{margin-left:auto}
.v-desk .bigplay{position:absolute;left:50%;top:50%;width:58px;height:58px;margin:-29px 0 0 -29px;border-radius:50%;background:rgba(255,255,255,.95);color:#111;display:grid;place-items:center;border:0;z-index:2;transition:transform .2s,opacity .3s;box-shadow:0 6px 20px rgba(0,0,0,.35)}
.v-desk .bigplay:hover{transform:scale(1.06)}
.v-desk .playing .bigplay{opacity:0;pointer-events:none}
.v-desk .ctrls{display:flex;gap:6px;align-items:center}
.v-desk .ctrls button{background:rgba(255,255,255,.14);border:0;color:#fff;border-radius:7px;padding:3px 8px;font-size:11.5px}
.v-desk .ctrls button.on{background:#fff;color:#111}
.v-desk .camtag{position:absolute;left:14px;top:14px;z-index:2;display:flex;gap:6px}
.v-desk .camtag span{background:rgba(0,0,0,.55);color:#fff;font-size:11.5px;padding:3px 8px;border-radius:6px;display:inline-flex;gap:6px;align-items:center}
.v-desk .camtag .rec{width:7px;height:7px;border-radius:50%;background:#ff4d4f;animation:pulse 1.6s infinite}
.v-desk .under{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}
.v-desk .steps-card .step{border:1px solid var(--line);border-radius:12px;margin:0 14px 10px;overflow:hidden;transition:border-color .3s,background .3s}
.v-desk .steps-card .step.active{border-color:var(--accent-line);box-shadow:0 0 0 3px var(--accent-soft)}
.v-desk .steps-card .step.locked{opacity:.55}
.v-desk .step-h{display:flex;gap:10px;align-items:center;padding:11px 12px;width:100%;background:none;border:0;text-align:left}
.v-desk .step-h .st{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;flex:none;border:1.5px solid var(--line-2);color:var(--ink-3)}
.v-desk .step.done .st{background:var(--ok);border-color:var(--ok);color:#fff}
.v-desk .step.active .st{border-color:var(--accent);color:var(--accent)}
.v-desk .step.active .st::after{content:"";width:8px;height:8px;border-radius:50%;background:var(--accent)}
.v-desk .step-h b{display:block;font-size:13px;font-weight:600}
.v-desk .step-h small{font-size:11.5px;color:var(--ink-3)}
.v-desk .step.done .step-h small{color:var(--ok)}
.v-desk .step.active .step-h small{color:var(--accent)}
.v-desk .step-h .chev{margin-left:auto;color:var(--ink-3);transition:transform .3s var(--ease)}
.v-desk .step.open .chev{transform:rotate(180deg)}
.v-desk .step-b{display:grid;grid-template-rows:0fr;transition:grid-template-rows .4s var(--ease)}
.v-desk .step.open .step-b{grid-template-rows:1fr}
.v-desk .step-b > div{overflow:hidden}
.v-desk .sub{display:flex;gap:10px;align-items:center;padding:8px 12px 8px 44px;font-size:12.5px;border-top:1px solid var(--line)}
.v-desk .sub .sst{width:16px;height:16px;border-radius:50%;display:grid;place-items:center;flex:none;border:1.5px solid var(--line-2)}
.v-desk .sub.done .sst{background:var(--ok);border-color:var(--ok);color:#fff}
.v-desk .sub .r{margin-left:auto;color:var(--ink-3);font-size:11.5px}
.v-desk .sub .prog{flex:1;max-width:120px}
.v-desk .match{display:grid;grid-template-columns:56px minmax(0,1fr) auto;gap:12px;align-items:center;padding:10px 4px;border-bottom:1px solid var(--line)}
.v-desk .match:last-child{border-bottom:0}
.v-desk .match img{width:56px;height:72px}
.v-desk .match .sc{display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--ink-3);margin-top:5px}
.v-desk .match .sc .prog{width:110px}
.v-desk .match.subj{background:var(--ok-soft);border-radius:10px}
.v-desk .legs{display:flex;gap:0;align-items:stretch;overflow-x:auto;padding:6px 2px 4px}
.v-desk .leg{flex:1;min-width:150px;border:1px solid var(--line);border-radius:12px;padding:10px;display:grid;gap:6px;background:var(--card)}
.v-desk .leg img{width:100%;height:120px;object-fit:cover}
.v-desk .hop{display:grid;place-items:center;min-width:56px;color:var(--ink-3);font-size:11px;text-align:center}
.v-desk .hop svg{color:var(--accent)}
.v-desk .moment{display:flex;gap:12px;align-items:center;padding:9px 6px;border-radius:9px;cursor:pointer}
.v-desk .moment:hover{background:var(--sunk)}
.v-desk .moment .tm{font:12px var(--f-mono);color:var(--accent);min-width:64px}
.v-desk .camlist .cam{display:grid;grid-template-columns:76px 1fr auto;gap:10px;align-items:center;padding:8px 14px;border-top:1px solid var(--line)}
.v-desk .camlist video{width:76px;height:44px;object-fit:cover;border-radius:7px;background:var(--video-bg)}
.v-desk .camlist .cam{width:100%;border:0;border-top:1px solid var(--line);background:none;text-align:left;font:inherit;color:inherit;grid-template-columns:76px minmax(0,1fr);cursor:pointer;transition:background .15s}
.v-desk .camlist .cam:hover{background:var(--sunk)}
.v-desk .camlist .cam.on{background:var(--accent-soft)}
.v-desk .camlist .cam.on video{box-shadow:0 0 0 2px var(--accent)}
.v-desk .camlist .cam b{display:block;font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-desk .camlist .cam .muted{font-size:12px}
.v-desk .camlist .credit{padding:10px 14px;border-top:1px solid var(--line);font-size:11.5px;color:var(--ink-3)}
.v-desk .camtag{display:none}
.v-desk .player{container-type:inline-size}
.v-desk .pip span{white-space:nowrap;max-width:calc(100% - 12px);overflow:hidden;text-overflow:ellipsis}
.v-desk .hud .meta > span:first-child{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-desk .hud .meta .mono{flex:none}
.v-desk .ctrls{flex:none}
.v-desk .ctrls button{height:26px;min-width:28px;display:inline-grid;place-items:center;font-variant-numeric:tabular-nums}
.v-desk .ctrls button:hover{background:rgba(255,255,255,.26)}
.v-desk .ctrls button.on{background:#fff;color:#111}
.v-desk .ctrls .sep{width:1px;height:14px;background:rgba(255,255,255,.25);margin:0 2px}
.v-desk .bar{height:6px}
.v-desk .bar::before{content:"";position:absolute;left:0;right:0;top:-8px;bottom:-8px}
.v-desk .player:fullscreen{border-radius:0;aspect-ratio:auto}
@container (max-width:560px){.v-desk .ctrls .spd,.v-desk .ctrls .sep{display:none}.v-desk .hud{padding:28px 10px 9px}.v-desk .pip{width:32%}}
@container (max-width:380px){.v-desk .hud .meta > span:first-child{display:none}.v-desk .ctrls .fs{display:none}}
@media (max-width:1180px){.v-desk.lean .cols{grid-template-columns:1fr}}
`);
      root.classList.add("v-desk"); root.append(style);

      const fileIn = el("input", { type: "file", accept: "video/*", hidden: true, id: "desk-video-file", onchange: () => { const f = fileIn.files[0]; fileIn.value = ""; if (f && STM.chat) STM.chat.upload(f); } });
      const sceneChips = STM.libraryPicker ? STM.libraryPicker(fileIn) : el("div", { class: "row" },
        ...STM.sceneList.filter(([n]) => STM.scenes[n] || n === STM.sceneName).map(([n, label, kind]) => el("button", {
          class: "chip" + (n === STM.sceneName ? " on" : ""), type: "button", id: "desk-scene-" + n,
          onclick: async () => { if (n === STM.sceneName) return; await STM.loadScene(n); STM.sceneName = n; STM.desk.reset(); STM.go("desk"); if (STM.chat) STM.chat.note(`Switched to ${label}. ${STM.scene().cameras.length} camera${STM.scene().cameras.length > 1 ? "s" : ""}, ${STM.scene().span()}.`); }
        }, label)),
        el("button", { class: "btn btn-sm", type: "button", id: "desk-add-video", onclick: () => fileIn.click() }, el("span", { html: icons.cam }), "Add a video"), fileIn);
      const short = (STM.sceneList.find(x => x[0] === STM.sceneName) || [0, s.title])[1];
      STM.pageHead(root, el("span", {}, short), `${s.cameras.length} camera${s.cameras.length > 1 ? "s" : ""} · ${s.span()} · ${s.name.startsWith("library-") ? "recording time" : new Date(s.clock0).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}`, sceneChips);
      if (STM.libraryNotice) { const notice = STM.libraryNotice(); if (notice) root.append(notice); }

      /* skeleton first, like a real app waiting on its cameras (staged for the video; the app skips it) */
      if (STM.showcase) {
        const skel = el("div", { class: "cols" },
          el("div", { class: "grid" }, STM.skeleton("auto", "100%", 16), STM.skeleton(30, "60%"), STM.skeleton(64), STM.skeleton(220, "100%", 16)),
          el("div", { class: "grid" }, STM.skeleton(380, "100%", 16), STM.skeleton(44, "100%", 10)));
        skel.firstChild.firstChild.style.aspectRatio = "16/9";
        root.append(skel);
        await sleep(STM.reduced ? 0 : 650);
        skel.remove();
      }

      const camRows = s.cameras.map(c => {
        const v = el("video", { muted: true, src: s.videoUrl(c) + "#t=8", preload: "metadata", playsinline: true }); v.muted = true;
        const n = s.tracks.filter(t => t.cam === c.id).length;
        return el("button", { class: "cam", type: "button", "data-cam": c.id, title: `Show Camera ${c.id} in the player`,
          onclick: () => { if (c.id === pipCam) pip.onclick(); else if (c.id !== mainCam) { mainCam = c.id; setCams(); renderMarkers(); } STM.$("#desk-player").scrollIntoView({ block: "nearest", behavior: "smooth" }); } },
          v, el("div", { style: { minWidth: 0 } }, el("b", {}, `Camera ${c.id} · ${c.name}`), el("div", { class: "muted" }, `${s.name.startsWith("library-") && s.index_status !== "ready" ? "Search index pending" : n + " people tracks"} · starts ${s.clock(c.offset || 0).slice(0, 5)}`)));
      });
      const camIds = s.cameras.map(c => c.id);
      let mainCam = camIds[0], pipCam = camIds[1] || null;
      const vMain = el("video", { muted: true, playsinline: true, preload: "auto" }); vMain.muted = true;
      const cMain = el("canvas");
      const vPip = el("video", { muted: true, playsinline: true, preload: "auto" }); vPip.muted = true;
      const cPip = el("canvas");
      const pip = el("div", { class: "pip", title: "Swap cameras", id: "desk-pip" }, vPip, cPip, el("span"));
      const fill = el("i"), bar = el("div", { class: "bar", id: "desk-bar" }, fill);
      const clockEl = el("span", { class: "mono" }), camName = el("b"), ctrls = el("div", { class: "ctrls" });
      const bigplay = el("button", { class: "bigplay", type: "button", "aria-label": "Play", html: '<svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>', id: "desk-play" });
      const camtag = el("div", { class: "camtag" }, el("span", { class: "camtag-n" }));
      const player = el("div", { class: "player", id: "desk-player" }, vMain, cMain, camtag, pipCam ? pip : null, bigplay,
        el("div", { class: "hud" }, el("div", { class: "meta" }, el("span", {}, camName), ctrls, clockEl), bar));

      let speed = 1;
      const playBtn = el("button", { type: "button", html: icons.play, "aria-label": "Play or pause" });
      const nowT = () => vMain.currentTime + (s.cam(mainCam).offset || 0);
      const back = el("button", { type: "button", title: "Back 5 s", "aria-label": "Back 5 seconds", onclick: () => seek(Math.max(0, nowT() - 5)) }, "−5s");
      const fwd = el("button", { type: "button", title: "Forward 5 s", "aria-label": "Forward 5 seconds", onclick: () => seek(Math.min(s.duration, nowT() + 5)) }, "+5s");
      const full = el("button", { type: "button", class: "fs", title: "Full screen", "aria-label": "Full screen", html: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>',
        onclick: () => (document.fullscreenElement ? document.exitFullscreen() : player.requestFullscreen()).catch(() => {}) });
      let analysis = !!STM.showcase;
      const anBtn = el("button", { type: "button", class: "an" + (analysis ? " on" : ""), title: "Show every tracked person", "aria-pressed": String(analysis),
        onclick: e => { analysis = !analysis; e.currentTarget.classList.toggle("on", analysis); e.currentTarget.setAttribute("aria-pressed", String(analysis)); } }, "Analysis");
      ctrls.append(playBtn, back, fwd, el("i", { class: "sep" }), ...[1, 2, 4].map(x => el("button", { type: "button", class: "spd" + (x === 1 ? " on" : ""), onclick: e => { speed = x; vMain.playbackRate = vPip.playbackRate = x; ctrls.querySelectorAll(".spd").forEach(b => b.classList.toggle("on", b === e.currentTarget)); } }, x + "×")), el("i", { class: "sep" }), anBtn, full);
      const setCams = () => {
        vMain.src = s.videoUrl(s.cam(mainCam)); if (pipCam) vPip.src = s.videoUrl(s.cam(pipCam));
        camName.textContent = `Camera ${mainCam} · ${s.cam(mainCam).name}`;

        if (pipCam) { pip.querySelector("span").textContent = `Camera ${pipCam}`; pip.title = `Swap to Camera ${pipCam} · ${s.cam(pipCam).name}`; }
        camRows.forEach(r => r.classList.toggle("on", r.dataset.cam === mainCam));
      };
      setCams();
      const play = () => { vMain.play().catch(() => {}); if (pipCam) vPip.play().catch(() => {}); };
      const pause = () => { vMain.pause(); vPip.pause(); };
      const toggle = () => (vMain.paused ? play() : pause());
      bigplay.onclick = toggle; playBtn.onclick = toggle;
      vMain.onplay = () => { player.classList.add("playing"); playBtn.innerHTML = icons.pause; };
      vMain.onpause = () => { player.classList.remove("playing"); playBtn.innerHTML = icons.play; };
      pip.onclick = () => { const t = vMain.currentTime, was = !vMain.paused; [mainCam, pipCam] = [pipCam, mainCam]; setCams(); vMain.currentTime = vPip.currentTime = t; if (was) play(); renderMarkers(); };
      bar.onclick = e => { const r = bar.getBoundingClientRect(); seek((e.clientX - r.left) / r.width * s.duration); };
      const seek = (t, cam) => {
        if (cam && cam !== mainCam && cam === pipCam) pip.onclick();
        const off = s.cam(mainCam).offset || 0;
        vMain.currentTime = Math.max(0, t - off); if (pipCam) vPip.currentTime = Math.max(0, t - (s.cam(pipCam).offset || 0));
      };

      /* boxes on both videos */
      /* per-frame work stays small: theme colours and the subject's route are cached, not recomputed per box */
      const colours = { ok: "#1F9D55", accent: "#2F5BEA", at: 0 };
      let legsKey = null, legsSet = new Set();
      const legsFor = () => { const k = STM.state.subject || ""; if (k !== legsKey) { legsKey = k; legsSet = k ? new Set(route(k).map(l => l.t.id)) : new Set(); } return legsSet; };
      const readColours = () => { const cs = getComputedStyle(document.documentElement); colours.ok = cs.getPropertyValue("--ok").trim() || colours.ok; colours.accent = cs.getPropertyValue("--accent").trim() || colours.accent; };
      readColours(); const themeObs = new MutationObserver(readColours); themeObs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      const draw = (cv, video, cam, small) => {
        const w = cv.clientWidth, h = cv.clientHeight; if (!w) return;
        const dpr = devicePixelRatio || 1; if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
        const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
        const vw = video.videoWidth || 16, vh = video.videoHeight || 9, k = Math.min(w / vw, h / vh), ox = (w - vw * k) / 2, oy = (h - vh * k) / 2, W = vw * k, H = vh * k;
        const t = video.currentTime + (s.cam(cam).offset || 0);
        const cands = new Set(STM.state.candidates), subjLegs = legsFor();
        let inView = 0;
        for (const tr of s.tracks) {
          if (tr.cam !== cam) continue;
          const b = s.boxAt(tr, t); if (!b) continue;
          inView++;
          const isS = subjLegs.has(tr.id), isC = cands.has(tr.id);
          const col = isS ? colours.ok : isC ? colours.accent : analysis ? "rgba(255,255,255,.85)" : "rgba(255,255,255,.55)";
          const x = ox + b[0] * W, y = oy + b[1] * H, bw = b[2] * W, bh = b[3] * H;
          if (analysis && !small) {
            /* motion trail: box-bottom-centre over the last 4 s */
            let px = null, py = null;
            g.setLineDash([]); g.lineWidth = 2; g.lineCap = "round";
            for (let k2 = 16; k2 >= 0; k2--) {
              const tb = s.boxAt(tr, t - k2 * 0.25); if (!tb) { px = null; continue; }
              const qx = ox + (tb[0] + tb[2] / 2) * W, qy = oy + (tb[1] + tb[3]) * H;
              if (px !== null) { g.globalAlpha = 0.85 * (1 - k2 / 17); g.strokeStyle = col; g.beginPath(); g.moveTo(px, py); g.lineTo(qx, qy); g.stroke(); }
              px = qx; py = qy;
            }
            g.globalAlpha = 1;
          }
          g.lineWidth = isS || isC ? 2.2 : analysis && !small ? 1.6 : 1; g.strokeStyle = col.trim() || "#2F5BEA";
          if (!isS && !isC && !analysis) g.setLineDash([4, 3]); else g.setLineDash([]);
          if (analysis && !small && !isS && !isC) {
            g.beginPath(); g.roundRect ? g.roundRect(x, y, bw, bh, 4) : g.rect(x, y, bw, bh); g.stroke();
            const label = tr.id; g.font = "500 11px 'JetBrains Mono', ui-monospace, monospace";
            const tw = g.measureText(label).width + 10 + 24;
            g.fillStyle = "rgba(0,0,0,.6)"; g.beginPath(); g.roundRect ? g.roundRect(x, Math.max(0, y - 18), tw, 16, 4) : g.rect(x, Math.max(0, y - 18), tw, 16); g.fill();
            g.fillStyle = "#fff"; g.fillText(label, x + 5, Math.max(12, y - 6));
            const lk = tr.look || (tr._lk ||= parseLook(tr.desc || "")), sy = Math.max(4, y - 14), sx = x + tw - 24;
            [lk.top, lk.bottom].forEach((c, i) => { g.fillStyle = CSS_COL[c] || "rgba(255,255,255,.15)"; g.fillRect(sx + i * 10, sy, 8, 8); g.strokeStyle = "rgba(255,255,255,.5)"; g.lineWidth = 1; g.strokeRect(sx + i * 10 + .5, sy + .5, 7, 7); });
            continue;
          }
          g.beginPath(); g.roundRect ? g.roundRect(x, y, bw, bh, 4) : g.rect(x, y, bw, bh); g.stroke();
          if (!small && (isS || isC || STM.state.candidates.length === 0)) {
            const label = isS ? `${tr.id} · them` : isC ? `#${STM.state.candidates.indexOf(tr.id) + 1} ${tr.desc.split(",")[0]}` : tr.id;
            g.font = "500 11px Outfit, system-ui, sans-serif"; const tw = g.measureText(label).width + 10;
            g.fillStyle = isS || isC ? (col.trim() || "#2F5BEA") : "rgba(0,0,0,.55)"; g.setLineDash([]);
            g.beginPath(); g.roundRect ? g.roundRect(x, Math.max(0, y - 18), tw, 16, 4) : g.rect(x, y - 18, tw, 16); g.fill();
            g.fillStyle = "#fff"; g.fillText(label, x + 5, Math.max(12, y - 6));
          }
        }
        if (analysis && !small) {
          const txt = `${inView} ${inView === 1 ? "person" : "people"} in view · Camera ${cam}`;
          g.font = "500 12px Outfit, system-ui, sans-serif"; g.setLineDash([]);
          const tw = g.measureText(txt).width + 16;
          g.fillStyle = "rgba(0,0,0,.6)"; g.beginPath(); g.roundRect ? g.roundRect(12, 12, tw, 22, 6) : g.rect(12, 12, tw, 22); g.fill();
          g.fillStyle = "#fff"; g.fillText(txt, 20, 27);
        }
      };
      let raf;
      const tick = () => {
        const t = vMain.currentTime + (s.cam(mainCam).offset || 0);
        fill.style.width = (t / s.duration * 100) + "%";
        clockEl.textContent = s.clock(t);
        if (pipCam && !vMain.paused && Math.abs(vPip.currentTime - vMain.currentTime) > 0.3) vPip.currentTime = vMain.currentTime;
        draw(cMain, vMain, mainCam, false); if (pipCam) draw(cPip, vPip, pipCam, true);
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
      vMain.loop = true; vPip.loop = true;

      const renderMarkers = () => {
        bar.querySelectorAll(".mk,.win").forEach(m => m.remove());
        const w = STM.state.window;
        if (w && (w.a != null || w.b != null) && (!w.cam || w.cam === mainCam)) {
          const a = Math.max(0, w.a ?? 0), b = Math.min(s.duration, w.b ?? s.duration);
          if (b > a) bar.append(el("span", { class: "win", id: "desk-window-band", title: winText(w), style: { left: (a / s.duration * 100) + "%", width: ((b - a) / s.duration * 100) + "%" } }));
        }
        const ids = STM.state.subject ? route(STM.state.subject).map(l => l.t.id) : STM.state.candidates;
        for (const id of ids) {
          const t = s.track(id); if (!t || t.cam !== mainCam) continue;
          bar.append(el("span", { class: "mk" + (STM.state.subject ? " subj" : ""), style: { left: (t.t0 / s.duration * 100) + "%", width: Math.max(.8, (t.t1 - t.t0) / s.duration * 100) + "%" } }));
        }
      };

      /* left column */
      const chipsRow = el("div", { class: "under demo-only" },
        el("span", { class: "chip" }, el("span", { html: icons.cam }), `${s.cameras.length} cameras`),
        el("span", { class: "chip" }, `${Math.round(s.duration)} s of footage`),
        el("span", { class: "chip" }, s.name.startsWith("library-") && s.index_status !== "ready" ? "Index pending" : `Indexed in ${(+s.build.seconds).toFixed(1)} s`),
        el("span", { class: "chip" }, `${s.tracks.length} people tracks`),
        el("span", { class: "chip" }, "No faces or names used"));
      const banner = el("div", { class: "ai-banner demo-only", id: "desk-banner" }, el("div", { class: "spark", html: icons.spark }), el("div", {}, el("b", {}, "Desk agent"), el("p", {})));
      const setBanner = (title, text) => { banner.querySelector("b").textContent = title; banner.querySelector("p").textContent = text; banner.classList.remove("pop"); void banner.offsetWidth; banner.classList.add("pop"); };

      const tabsHost = el("div"), pane = el("div", { class: "card-b", style: { minHeight: "220px" } });
      const tabCard = el("div", { class: "card" }, tabsHost, pane);
      let tab = "matches";
      const tabs = STM.tabs(tabsHost, [{ id: "matches", label: "Matches" }, { id: "route", label: "Route" }, { id: "moments", label: "Moments" }], id => { tab = id; renderPane(); }, "matches");
      tabsHost.querySelectorAll("button").forEach(b => b.id = "desk-tab-" + b.dataset.tab);

      function renderPane() {
        pane.innerHTML = ""; const box = el("div", { class: "tab-pane" }); pane.append(box);
        if (tab === "matches") {
          const w = STM.state.window;
          if (hasWin(w)) box.append(el("div", { class: "winbar", id: "desk-window" }, el("span", { html: icons.clock || "" }), el("span", {}, "Narrowed to ", el("b", {}, winText(w))),
            el("button", { class: "btn btn-sm btn-ghost", type: "button", id: "desk-window-clear", onclick: () => { STM.desk.search(STM.state.look, false, {}); if (STM.chat) STM.chat.note("Showing matches from the whole recording again."); } }, "Clear")));
          if (!STM.state.candidates.length && STM.state.query) {
            box.append(el("div", { class: "empty" }, el("div", { style: { fontWeight: 600, color: "var(--ink)", marginBottom: "4px" } }, "No match in this window"),
              "Try a wider time window, another camera or one colour you're sure of. Colours are read from the video pixels, so lighting can shift them."));
            return;
          }
          if (!STM.state.candidates.length) {
            box.append(el("div", { class: "empty" }, el("div", { style: { fontWeight: 600, color: "var(--ink)", marginBottom: "4px" } }, "No search yet"),
              "Tell the desk agent what they wore, and when or where you last saw them if you know. For example “black top and jeans, after 16:52”."));
            return;
          }
          const list = el("div", { class: "stagger" });
          STM.state.candidates.forEach((id, i) => {
            const t = s.track(id), sc = (STM.state.scores || {})[id] || 0, isS = STM.state.subject === id;
            const p = el("div", { class: "prog" + (isS ? " ok" : "") }, el("i"));
            list.append(el("div", { class: "match" + (isS ? " subj" : "") },
              el("img", { class: "crop", src: s.cropUrl(t), alt: t.desc, loading: "lazy" }),
              el("div", { style: { minWidth: 0 } },
                el("div", { class: "row" }, el("b", {}, `#${i + 1} · ${t.desc}`), isS ? el("span", { class: "pill ok" }, "Confirmed") : null),
                el("div", { class: "muted", style: { fontSize: "12px" } }, `Camera ${t.cam} · ${s.cam(t.cam).name} · ${s.clock(t.t0)}–${s.clock(t.t1)} · ${Math.round(t.t1 - t.t0)} s`),
                el("div", { class: "sc" }, p, `${Math.round(sc * 100)}% colour match`)),
              el("div", { class: "row" },
                el("button", { class: "btn btn-sm btn-ghost", type: "button", onclick: () => seek(t.t0 + 0.5, t.cam) }, "View"),
                isS ? null : el("button", { class: "btn btn-sm btn-primary", type: "button", id: `desk-confirm-${i + 1}`, onclick: () => { STM.desk.pick(i + 1); if (STM.chat) STM.chat.note(`You confirmed match #${i + 1}.`); } }, "That's them"))));
            requestAnimationFrame(() => setTimeout(() => p.firstChild.style.width = Math.round(sc * 100) + "%", 120 + i * 60));
          });
          box.append(list);
        } else if (tab === "route") {
          const id = STM.state.subject;
          if (!id) { box.append(el("div", { class: "empty" }, "Confirm a match first. The route shows every camera that saw them, in order.")); return; }
          const legs = route(id), row = el("div", { class: "legs stagger" });
          legs.forEach((l, i) => {
            if (i) {
              const gap = l.t.t0 - legs[i - 1].t.t1;
              row.append(el("div", { class: "hop", html: `<svg width="40" height="14" viewBox="0 0 40 14"><path d="M2 7h34M30 2l6 5-6 5" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"><animate attributeName="stroke-dashoffset" from="14" to="0" dur="1s" repeatCount="indefinite"/></path></svg><span>${gap < 0 ? "overlap " + Math.round(-gap) + " s" : Math.round(gap) + " s later"}</span>` }));
            }
            row.append(el("div", { class: "leg" },
              el("img", { class: "crop", src: s.cropUrl(l.t), alt: l.t.desc }),
              el("div", { class: "row" }, el("b", { style: { fontSize: "12.5px" } }, `Camera ${l.t.cam}`), el("span", { class: "pill " + (l.how === "confirmed" ? "ok" : "info") }, l.how)),
              el("div", { class: "muted mono", style: { fontSize: "11.5px" } }, `${s.clock(l.t.t0)} → ${s.clock(l.t.t1)}`),
              el("button", { class: "btn btn-sm", type: "button", onclick: () => seek(l.t.t0 + .5, l.t.cam) }, "Play from here")));
          });
          box.append(row);
          const last = lastSighting(id);
          if (last) box.append(el("div", { class: "ai-banner", style: { marginTop: "12px" } }, el("div", { class: "spark", html: icons.spark }),
            el("div", {}, el("b", {}, `Last seen on Camera ${last.track.cam} at ${s.clock(last.t)}`), el("p", {}, `${s.cam(last.track.cam).name}. Same-colour sightings on other cameras are a lead to check, not proof it is the same person.`))));
        } else {
          const id = STM.state.subject || STM.state.candidates[0];
          const evs = [];
          if (id) {
            for (const l of route(id)) { evs.push({ t: l.t.t0, cam: l.t.cam, text: `Enters Camera ${l.t.cam} (${s.cam(l.t.cam).name})`, crop: l.t }); evs.push({ t: l.t.t1, cam: l.t.cam, text: `Leaves Camera ${l.t.cam}`, crop: l.t }); }
            for (const n of near(id)) evs.push({ t: Math.max(n.t.t0, 0), cam: n.t.cam, text: `${n.t.desc} on the same camera (${n.gap >= 0 ? n.gap + " s after" : -n.gap + " s before"} they left)`, crop: n.t, near: true });
          } else for (const t of [...s.tracks].sort((a, b) => a.t0 - b.t0).slice(0, 8)) evs.push({ t: t.t0, cam: t.cam, text: `${t.desc} enters Camera ${t.cam}`, crop: t });
          evs.sort((a, b) => a.t - b.t);
          const list = el("div", { class: "stagger" });
          for (const e of evs) list.append(el("div", { class: "moment", onclick: () => seek(e.t + .2, e.cam) },
            el("span", { class: "tm" }, s.clock(e.t)), el("img", { class: "crop", src: s.cropUrl(e.crop), style: { width: "26px", height: "34px" }, alt: "" }),
            el("span", {}, e.text), e.near ? el("span", { class: "pill warn", style: { marginLeft: "auto" } }, "nearby") : null));
          box.append(list.childNodes.length ? list : el("div", { class: "empty" }, "No moments yet."));
        }
      }

      /* right column: search steps (accordion), CTA, cameras */
      const stepsCard = el("div", { class: "card steps-card demo-only", id: "desk-steps" });
      const cta = el("button", { class: "btn btn-primary btn-block", type: "button", id: "desk-continue" }, el("span", {}, "Continue search"), el("span", { html: icons.arrow }));
      const robotBtn = el("button", { class: "btn btn-block", type: "button", id: "desk-robot", onclick: () => STM.go("robot") }, el("span", { html: STM.icons.robot.replace('class="ico"', 'width="16" height="16"') }), "Send Rover 1 to last sighting");
      const ctaNote = el("div", { class: "muted", style: { fontSize: "11.5px", textAlign: "center" } }, "Steps unlock as you confirm each one");
      const open = new Set(["describe"]);
      function renderSteps() {
        const st = STM.state, s2 = STM.scene();
        const phase = st.subject ? (st.followed ? "follow" : "route") : st.candidates.length ? "confirm" : "describe";
        const order = ["describe", "confirm", "route", "follow"], pi = order.indexOf(phase);
        open.clear(); open.add(phase);
        const lookChips = st.look ? [st.look.top && `top: ${st.look.top}`, st.look.bottom && `lower: ${st.look.bottom}`].filter(Boolean) : [];
        const last = st.subject ? lastSighting(st.subject) : null;
        const defs = [
          ["describe", "01. Describe the person", st.query ? "Complete" : "In progress · type or speak", [
            [st.query ? `“${st.query.slice(0, 40)}”` : "What were they wearing?", !!st.query, lookChips.join(" · ")],
            ["Faces and names are never used", true, "policy"]]],
          ["confirm", "02. Confirm it's them", st.subject ? "Complete" : st.candidates.length ? `In progress · ${st.candidates.length} matches` : "Locked · needs a description", [
            [`Searched ${s2.cameras.length} cameras, ${s2.tracks.length} tracks`, !!st.query, ""],
            ["Pick the right person", !!st.subject, st.subject ? st.subject : "", st.candidates.length && !st.subject ? 0.5 : st.subject ? 1 : 0]]],
          ["route", "03. Retrace their route", st.subject ? (st.followed ? "Complete" : "In progress") : "Locked · confirm a match", [
            [st.subject ? `${route(st.subject).length} sightings across cameras` : "Every camera that saw them", !!st.subject, ""],
            [last ? `Last seen ${s2.clock(last.t)} on Camera ${last.track.cam}` : "Last sighting", !!last, ""]]],
          ["follow", "04. Follow up", st.subject ? "Ready" : "Locked", [
            ["Who came near after they left", !!st.followed, ""], ["Send Rover 1 to check the spot", false, "robot"]]],
        ];
        stepsCard.innerHTML = "";
        stepsCard.append(el("div", { class: "card-h" }, el("h3", {}, "Search steps"), el("div", { class: "grow" }), el("span", { class: "muted", style: { fontSize: "12px" } }, `4 steps · ${s2.cameras.length} cameras`)));
        const wrap = el("div", { style: { paddingTop: "12px" } });
        defs.forEach(([id, title, sub, subs], i) => {
          const state = i < pi ? "done" : i === pi ? "active" : "locked";
          const stepEl = el("div", { class: `step ${state}` + (open.has(id) ? " open" : ""), id: "desk-step-" + id },
            el("button", { class: "step-h", type: "button", onclick: () => stepEl.classList.toggle("open") },
              el("span", { class: "st", html: state === "done" ? icons.check : state === "locked" ? icons.lock : "" }),
              el("span", {}, el("b", {}, title), el("small", {}, sub)), el("span", { class: "chev", html: icons.chev })),
            el("div", { class: "step-b" }, el("div", {}, ...subs.map(([txt, ok, r, prog]) => el("div", { class: "sub" + (ok ? " done" : "") },
              el("span", { class: "sst", html: ok ? icons.check : "" }), el("span", {}, txt),
              prog != null && !ok ? el("div", { class: "prog" }, el("i", { style: { width: prog * 100 + "%" } })) : null,
              r ? el("span", { class: "r" }, r) : null)))));
          wrap.append(stepEl);
        });
        stepsCard.append(wrap);
        cta.disabled = false;
        cta.firstChild.textContent = phase === "describe" ? "Describe who you lost" : phase === "confirm" ? "Confirm the best match" : phase === "route" ? "Who came near after?" : "Search again";
        robotBtn.disabled = !st.subject;
      }
      cta.onclick = () => {
        const st = STM.state;
        if (!st.query) { $chatFocus(); return; }
        if (!st.subject) { STM.desk.pick(1); if (STM.chat) STM.chat.note("You confirmed match #1."); return; }
        if (!st.followed) { STM.state.followed = true; tabs.pick("moments"); renderSteps(); if (STM.chat) STM.chat.run("who_came_near", {}, "Who came near after?"); return; }
        STM.desk.reset();
      };
      const $chatFocus = () => { STM.$("#chat").classList.remove("closed"); STM.$("#q").focus(); };

      const camList = el("div", { class: "card camlist" }, el("div", { class: "card-h" }, el("h3", {}, "Cameras")),
        ...camRows, s.credit ? el("div", { class: "credit" }, s.credit) : null);

      /* the video layout keeps every panel; the app puts the evidence (video) beside the answer (matches) */
      if (STM.showcase) {
        tabCard.style.marginTop = "16px";
        const left = el("div", { class: "grid stagger", style: { gap: "0" } }, player, chipsRow, banner, tabCard);
        const right = el("div", { class: "grid stagger" }, stepsCard, el("div", { class: "grid demo-only", style: { gap: "8px" } }, cta, robotBtn, ctaNote), camList);
        root.append(el("div", { class: "cols" }, left, right));
      } else {
        root.classList.add("lean");
        root.append(el("div", { class: "cols" }, el("div", { class: "grid stagger" }, player, camList), el("div", { class: "grid stagger" }, tabCard)));
      }

      const syncBanner = () => {
        const st = STM.state;
        if (st.subject) {
          const last = lastSighting(st.subject), t = s.track(st.subject);
          setBanner("Route ready", `${t.desc}: ${route(st.subject).length} sightings. Last seen on Camera ${last.track.cam} at ${s.clock(last.t)}. Ask “who came near?” or send Rover 1 to check.`);
        } else if (st.candidates.length) setBanner(`${st.candidates.length} people match`, `Best match is ${s.track(st.candidates[0]).desc} on Camera ${s.track(st.candidates[0]).cam}. Say “that's them” or pick a number.`);
        else if (st.query) setBanner("No match yet", "Nobody on these cameras matches that. Try another colour or fewer details.");
        else setBanner("Desk agent", "Tell the agent what the person wore. It searches every camera at once and never uses faces or names.");
      };
      const onDesk = d => {
        renderSteps(); syncBanner(); renderMarkers();
        if (d && d.kind === "search") { tabs.pick("matches"); const first = STM.state.candidates[0]; if (first) { const t = s.track(first); seek(t.t0 + 1, t.cam); } }
        if (d && d.kind === "pick") { tabs.pick("route"); const last = lastSighting(d.id); if (last) seek(Math.max(0, last.t - 3), last.track.cam); play(); }
        if (d && (d.kind === "reject" || d.kind === "reset")) renderPane();
      };
      const offDesk = e => onDesk(e.detail), offSeek = e => seek(e.detail.t, e.detail.cam);
      STM.bus.addEventListener("desk", offDesk); STM.bus.addEventListener("seek", offSeek);
      renderSteps(); syncBanner();
      vMain.addEventListener("loadedmetadata", () => { renderMarkers(); seek(STM.state.candidates.length ? s.track(STM.state.candidates[0]).t0 + 1 : 12); }, { once: true });

      return () => { cancelAnimationFrame(raf); themeObs.disconnect(); pause(); STM.bus.removeEventListener("desk", offDesk); STM.bus.removeEventListener("seek", offSeek); root.classList.remove("v-desk", "lean"); };
    },
  });
})();
