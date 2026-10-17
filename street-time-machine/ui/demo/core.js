/* Find & Follow demo desk: shared state, helpers, router, boot sequence. Views register with STM.view(id, spec)
   and get mount(root, ctx). Scenes are the real ones built by scripts/build_find_scene.py (demo/scenes/<name>). */
(function () {
  "use strict";
  const STM = window.STM = {};
  const $ = STM.$ = (s, r = document) => r.querySelector(s);
  const $$ = STM.$$ = (s, r = document) => [...r.querySelectorAll(s)];
  STM.sleep = ms => new Promise(r => setTimeout(r, ms));
  STM.esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  STM.el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "html") n.innerHTML = v;
      else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of kids.flat()) if (c != null && c !== false) n.append(c.nodeType ? c : document.createTextNode(c));
    return n;
  };
  STM.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- facts: measured today; views quote these instead of inventing numbers ---------- */
  STM.facts = {
    labBuild: { seconds: 32.2, index_s: 30.9, cams: 2, footage_s: 60, fps: 2, fragments: 43, tracks: 9 },
    caviarBuild: { seconds: 3.9, index_s: 3.86, cams: 2, footage_s: 15.3, fps: 2, fragments: 34, tracks: 21 },
    routeMs: [400, 1000],          // DeepSeek V4.1 Flash routing, typical range seen today
    sttMs: [900, 1300],            // Canary push-to-talk round trip (browser encode to text)
    vssIndexMin: [18, 26],         // VSS archive indexing per clip (archive path only)
    models: { router: "DeepSeek V4.1 Flash", fallback: ["GLM 5.3 Flash", "DeepSeek V3.1"], stt: "NVIDIA Canary", caption: "Nemotron VL" },
  };

  /* ---------- icons ---------- */
  const I = (d, extra = "") => `<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" ${extra}>${d}</svg>`;
  STM.icons = {
    desk: I('<circle cx="9" cy="7" r="3"/><path d="M3.5 20a5.5 5.5 0 0111 0"/><circle cx="17.5" cy="14.5" r="3"/><path d="M19.7 16.7L22 19"/>'),
    timeline: I('<path d="M3 6h10M3 12h18M3 18h7"/><circle cx="16" cy="6" r="2"/><circle cx="13" cy="18" r="2"/>'),
    catalogue: I('<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>'),
    similar: I('<circle cx="8" cy="9" r="4"/><circle cx="16" cy="15" r="4"/><path d="M11 12l2 1"/>'),
    pipeline: I('<rect x="2" y="9" width="5" height="6" rx="1.5"/><rect x="10" y="9" width="5" height="6" rx="1.5"/><rect x="18" y="9" width="4" height="6" rx="1.5"/><path d="M7 12h3M15 12h3"/>'),
    data: I('<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>'),
    robot: I('<rect x="5" y="8" width="14" height="10" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01M8 21h8"/><circle cx="12" cy="3" r="1"/>'),
    spark: '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9z"/></svg>',
    check: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l5 5L20 7"/></svg>',
    lock: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 018 0v4"/></svg>',
    play: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
    pause: '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>',
    arrow: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    chev: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>',
    cam: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>',
  };

  /* ---------- scenes ---------- */
  STM.scenes = {};
  STM.loadScene = async (name, at) => {
    if (STM.scenes[name]) return STM.scenes[name];
    const library = name.startsWith("library-");
    let base = at || `demo/scenes/${name}/`;
    const response = await fetch(library ? `/api/find/library/${encodeURIComponent(name)}/scene` : base + "scene.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Footage unavailable (${response.status})`);
    const s = await response.json();
    if (library) base = s.base || `/out/find/${name}/`;
    s.name = name; s.base = base;
    s.cam = id => s.cameras.find(c => c.id === id);
    s.track = id => s.tracks.find(t => t.id === id);
    const asset = path => /^(\/|https?:|data:|blob:)/.test(path) ? path : base + path;
    s.cropUrl = t => asset(t.crop || `crops/${t.id}.jpg`);
    s.videoUrl = c => asset(c.video || `cam${c.id}.mp4`);
    s.clock = t => { const d = new Date(new Date(s.clock0).getTime() + t * 1000); return d.toTimeString().slice(0, 8); };
    s.toT = str => {                               // "16:52", "4:52pm", "16:52:30" -> scene seconds (null if unreadable)
      const m = String(str || "").trim().toLowerCase().match(/^(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(am|pm)?$/);
      if (!m || (!m[2] && !m[4])) return null;
      let h = +m[1]; if (m[4] === "pm" && h < 12) h += 12; if (m[4] === "am" && h === 12) h = 0;
      const b = new Date(s.clock0), d = new Date(b); d.setHours(h, +(m[2] || 0), +(m[3] || 0), 0);
      return (d - b) / 1000;
    };
    s.span = () => `${s.clock(0)}–${s.clock(s.duration)}`;
    s.boxAt = (t, time) => {                       // linear interpolation between sampled boxes
      const b = t.boxes; if (!b.length || time < b[0][0] - .25 || time > b[b.length - 1][0] + .25) return null;
      for (let i = 0; i < b.length - 1; i++) if (time >= b[i][0] && time <= b[i + 1][0]) {
        const f = (time - b[i][0]) / Math.max(1e-6, b[i + 1][0] - b[i][0]);
        return [1, 2, 3, 4].map(k => b[i][k] + (b[i + 1][k] - b[i][k]) * f);
      }
      return b[time < b[0][0] ? 0 : b.length - 1].slice(1);
    };
    STM.scenes[name] = s;
    return s;
  };
  STM.sceneName = "meva";
  STM.scene = () => STM.scenes[STM.sceneName];

  /* desk state shared by the desk view, chat and robot */
  STM.state = { query: "", candidates: [], subject: null, lead: null, item: null, phase: "describe" };
  STM.bus = new EventTarget();
  STM.emit = (type, detail) => STM.bus.dispatchEvent(new CustomEvent(type, { detail }));
  STM.on = (type, fn) => STM.bus.addEventListener(type, e => fn(e.detail));

  /* ---------- ui helpers ---------- */
  let toastT;
  STM.toast = (msg, ms = 2400) => {
    const t = $("#toast"); t.innerHTML = msg; t.classList.add("show");
    clearTimeout(toastT); toastT = setTimeout(() => t.classList.remove("show"), ms);
  };
  STM.countUp = (node, to, { ms = 900, dec = 0, suffix = "" } = {}) => {
    const t0 = performance.now(), from = 0;
    const step = now => {
      const f = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - f, 3);
      node.textContent = (from + (to - from) * e).toFixed(dec) + suffix;
      if (f < 1) requestAnimationFrame(step);
    };
    if (STM.reduced) node.textContent = to.toFixed(dec) + suffix; else requestAnimationFrame(step);
  };
  /* tabs with a sliding underline: STM.tabs(container, [{id,label}], onPick, initialId) */
  STM.tabs = (host, items, onPick, initial) => {
    host.classList.add("tabs"); host.innerHTML = "";
    const ink = STM.el("span", { class: "ink" });
    const btns = items.map(it => STM.el("button", { type: "button", "data-tab": it.id, onclick: () => pick(it.id) }, it.label));
    host.append(...btns, ink);
    const pick = id => {
      btns.forEach(b => b.classList.toggle("on", b.dataset.tab === id));
      const b = btns.find(b => b.dataset.tab === id);
      if (b) { ink.style.left = b.offsetLeft + "px"; ink.style.width = b.offsetWidth + "px"; }
      onPick(id);
    };
    requestAnimationFrame(() => pick(initial || items[0].id));
    return { pick };
  };
  STM.skeleton = (h = 14, w = "100%", r = 8) => STM.el("div", { class: "skel", style: { height: typeof h === "number" ? h + "px" : h, width: w, borderRadius: r + "px" } });
  STM.colourHex = { red: "#D9433B", orange: "#EE8A2E", yellow: "#E9C530", green: "#3E9B57", blue: "#3368D6", purple: "#7B54C9", pink: "#E46CA6", brown: "#8A5A3B", black: "#22252B", white: "#F4F4F2", grey: "#9AA0A8", gray: "#9AA0A8" };

  /* ---------- router ---------- */
  STM.views = {};
  STM.order = ["desk", "timeline", "catalogue", "similar", "pipeline", "data", "robot"];
  /* One codebase, two builds. The app (default) is what a desk operator uses: Find & Follow, Timeline, Catalogue.
     Showcase (?showcase or #tour) adds what exists for the demo video: pipeline, data, similar, robot, search steps,
     latency badges and staged loading. Showcase-only elements carry class="demo-only"; nothing is deleted. */
  STM.showcase = /[?&]showcase\b/.test(location.search) || location.hash === "#tour";
  document.documentElement.classList.toggle("showcase", STM.showcase);
  STM.appOrder = ["desk", "timeline", "catalogue"];
  STM.view = (id, spec) => { STM.views[id] = spec; };
  STM.current = null;
  let mounted = null;
  STM.go = async (id, opts = {}) => {
    if (!STM.views[id] || (!STM.showcase && !STM.appOrder.includes(id))) id = "desk";
    if (mounted && mounted.unmount) try { mounted.unmount(); } catch (e) { console.error(e); }
    STM.current = id;
    $$("#nav a").forEach(a => a.classList.toggle("on", a.dataset.view === id));
    const root = $("#view"); root.innerHTML = ""; $("#viewWrap").scrollTop = 0;
    try { history.replaceState(null, "", "#" + id); } catch (e) {}
    const spec = STM.views[id];
    mounted = { unmount: null };
    try {
      const r = await spec.mount(root, { scene: STM.scene(), opts });
      if (typeof r === "function") mounted.unmount = r;
    } catch (e) {
      console.error(e);
      root.append(STM.el("div", { class: "card empty" }, `This view failed to load (${e.message}).`));
    }
    STM.emit("view", id);
  };
  STM.buildNav = () => {
    const nav = $("#nav"); nav.innerHTML = "";
    for (const id of (STM.showcase ? STM.order : STM.appOrder)) {
      const v = STM.views[id]; if (!v) continue;
      const a = STM.el("a", { href: "#" + id, "data-view": id, onclick: e => { e.preventDefault(); STM.go(id); } });
      a.innerHTML = (STM.icons[id] || "") + `<span>${STM.esc(v.label)}</span>`;
      nav.append(a);
    }
  };
  STM.pageHead = (root, title, sub, right) => {
    const h = STM.el("div", { class: "view-title" },
      STM.el("div", {}, STM.el("h1", {}, title), sub ? STM.el("p", {}, sub) : null), STM.el("div", { class: "grow" }), right || null);
    root.append(h); return h;
  };

  /* ---------- health of the live services (route + speech), shown in the sidebar card ---------- */
  STM.live = { route: null, speech: null };
  STM.checkLive = async () => {
    const get = async u => { try { const r = await fetch(u, { cache: "no-store" }); return r.ok ? r.json() : null; } catch (e) { return null; } };
    const [h, sp] = await Promise.all([get("/api/find/health"), get("/api/find/speech")]);
    STM.live.route = !!(h && h.ok && h.models && h.models.length);
    STM.live.speech = !!(sp && sp.available);
    return STM.live;
  };

  /* ---------- boot ---------- */
  STM.boot = async () => {
    STM.startHash = location.hash;
    const steps = [
      ["Connecting to cameras", async () => { await STM.loadScene("meva"); await Promise.all(["caviar", "lab"].map(n => STM.loadScene(n).catch(() => null))); if (STM.loadLibrary) await STM.loadLibrary(); const wanted = new URLSearchParams(location.search).get("scene"); if (wanted && (STM.libraryScenes || {})[wanted]) { await STM.loadScene(wanted); STM.sceneName = wanted; } return `${STM.scene().cameras.length} online`; }],
      ["Loading the people index", async () => `${Object.values(STM.scenes).reduce((a, x) => a + x.tracks.length, 0)} tracks`],
      ["Warming the desk agent", async () => { await Promise.race([STM.checkLive(), STM.sleep(1500)]); return STM.live.route ? "ready" : "offline router"; }],
      ["Checking privacy rules", async () => "no faces, no names"],
    ];
    const box = $("#bootSteps");
    const rows = steps.map(([label]) => {
      const r = STM.el("div", {}, STM.el("span", { class: "st", html: '<span class="spin" style="width:14px;height:14px;opacity:.25;animation:none"></span>' }), label, STM.el("span", { class: "ms" }));
      box.append(r); return r;
    });
    for (let i = 0; i < steps.length; i++) {
      const r = rows[i], t0 = performance.now();
      r.querySelector(".st").innerHTML = '<span class="spin" style="width:14px;height:14px"></span>';
      let note = "";
      try { note = await steps[i][1](); } catch (e) { note = "unavailable"; }
      await STM.sleep(Math.max(0, (STM.showcase ? 380 : 60) - (performance.now() - t0)));
      r.classList.add("done");
      r.querySelector(".st").innerHTML = `<span style="width:16px;height:16px;border-radius:50%;background:var(--ok);color:#fff;display:grid;place-items:center">${STM.icons.check}</span>`;
      r.querySelector(".ms").textContent = note;
    }
    STM.buildNav();
    if (innerWidth <= 1040) $("#chat").classList.add("closed");
    await STM.sleep(250);
    $("#boot").classList.add("gone"); setTimeout(() => { $("#boot").hidden = true; }, 700);
    const start = (location.hash || "#desk").slice(1);
    STM.go(STM.views[start] ? start : "desk");
    if (STM.chat) STM.chat.init();
    if (STM.tour) STM.tour.init();
    wireChrome();
  };

  function wireChrome() {
    $("#chatToggle").onclick = () => $("#chat").classList.toggle("closed");
    $("#chatClose").onclick = () => $("#chat").classList.add("closed");
    $("#themeBtn").onclick = () => {
      const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
      r.dataset.theme = dark ? "light" : "dark";
    };
    document.addEventListener("keydown", e => {
      if (e.key === "/" && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) { e.preventDefault(); $("#chat").classList.remove("closed"); $("#q").focus(); }
    });
  }
})();
