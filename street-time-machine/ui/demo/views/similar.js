/* Similar: appearance search computed in the browser. Every crop from both scenes is read into a small canvas and turned
   into a colour signature (upper body + lower body histograms). Cosine similarity ranks sightings; a 2-D PCA of all
   signatures (power iteration on the covariance) draws the appearance map. Nothing here uses faces. */
(function () {
  "use strict";
  const { el, esc, sleep } = STM;

  /* ---------- colour signature ---------- */
  const HUES = [["red", 345, 15], ["orange", 15, 40], ["yellow", 40, 70], ["green", 70, 165], ["blue", 165, 255], ["purple", 255, 290], ["pink", 290, 345]];
  const LEVELS = ["dark", "mid", "bright"];
  const BINS = [];
  HUES.forEach(([h]) => LEVELS.forEach(l => BINS.push({ hue: h, level: l, name: l === "dark" && h === "orange" ? "brown" : l === "mid" ? h : `${l} ${h}` })));
  ["black", "grey", "white"].forEach(h => BINS.push({ hue: h, level: "mid", name: h }));
  const NB = BINS.length;                         // 24 bins per half, 48 values per signature
  const hueBin = h => HUES.findIndex(([, a, b]) => (a > b ? h >= a || h < b : h >= a && h < b));

  function readPixels(img, cv) {
    const h = Math.min(64, img.naturalHeight), w = Math.max(4, Math.round(img.naturalWidth * h / img.naturalHeight));
    cv.width = w; cv.height = h;
    const c = cv.getContext("2d", { willReadFrequently: true });
    c.clearRect(0, 0, w, h); c.drawImage(img, 0, 0, w, h);
    return { data: c.getImageData(0, 0, w, h).data, w, h };
  }
  function histOf({ data, w, h }) {                // centre-weighted, head and feet rows skipped
    const top = new Float64Array(NB), bot = new Float64Array(NB);
    for (let y = 0; y < h; y++) {
      const fy = (y + .5) / h; if (fy < .12 || fy >= .92) continue;
      const H = fy < .5 ? top : bot;
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4; if (data[i + 3] < 128) continue;
        const fx = (x + .5) / w, wt = Math.exp(-Math.pow((fx - .5) / .26, 2) / 2);
        const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
        const mx = Math.max(r, g, b), d = mx - Math.min(r, g, b), s = mx ? d / mx : 0;
        let k;
        if (mx < .18 || (s < .22 && mx < .3)) k = NB - 3;
        else if (s < .22) k = mx > .75 ? NB - 1 : NB - 2;
        else {
          let hh = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
          hh = (hh * 60 + 360) % 360;
          k = Math.max(0, hueBin(hh)) * 3 + (mx < .45 ? 0 : mx < .75 ? 1 : 2);
        }
        H[k] += wt;
      }
    }
    const norm = a => { const t = a.reduce((x, y) => x + y, 0) || 1; return Array.from(a, v => v / t); };
    return [norm(top), norm(bot)];
  }
  function wordsHist(word) {                        // fallback when pixels cannot be read: the index's colour word
    const hgt = new Array(NB).fill(0), w = word === "gray" ? "grey" : word;
    const k = BINS.findIndex(b => b.name === w); hgt[k >= 0 ? k : NB - 2] = 1; return hgt;
  }
  const sig = (top, bot) => top.map(Math.sqrt).concat(bot.map(Math.sqrt));   // each half has unit length
  const dot = (a, b, from = 0, to = a.length) => { let s = 0; for (let i = from; i < to; i++) s += a[i] * b[i]; return s; };
  const cosine = (a, b) => dot(a, b) / (Math.sqrt(dot(a, a) * dot(b, b)) || 1);

  function pca(X) {
    const n = X.length, d = X[0].length, mean = new Array(d).fill(0);
    X.forEach(v => v.forEach((x, j) => { mean[j] += x / n; }));
    const Z = X.map(v => v.map((x, j) => x - mean[j]));
    const C = Array.from({ length: d }, () => new Float64Array(d));
    for (const z of Z) for (let i = 0; i < d; i++) { const zi = z[i]; if (zi) for (let j = 0; j < d; j++) C[i][j] += zi * z[j] / Math.max(1, n - 1); }
    let total = 0; for (let i = 0; i < d; i++) total += C[i][i];
    const comps = [], vars = [];
    for (let k = 0; k < 2; k++) {
      let v = Array.from({ length: d }, (_, i) => Math.sin(i * 12.9898 + k * 78.233) + .01), lam = 0;
      for (let it = 0; it < 150; it++) {
        const w = new Array(d).fill(0);
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += C[i][j] * v[j]; w[i] = s; }
        lam = Math.sqrt(dot(w, w)) || 1; v = w.map(x => x / lam);
      }
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] -= lam * v[i] * v[j];
      comps.push(v); vars.push(total ? lam / total : 0);
    }
    return { pts: Z.map(z => comps.map(c => dot(z, c))), vars };
  }

  const CSS = `
.v-similar{display:grid;gap:16px;container-type:inline-size}
.v-similar .vs-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))}
.v-similar .vs-stats .stat{padding:14px 18px;border-right:1px solid var(--line)}
.v-similar .vs-stats .stat:last-child{border-right:0}
.v-similar .vs-top{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);gap:16px;align-items:stretch}
.v-similar .vs-top > .card{display:flex;flex-direction:column;min-width:0}
.v-similar .vs-q{display:grid;grid-template-columns:132px minmax(0,1fr);gap:16px;align-items:start}
.v-similar .vs-ph{position:relative;border-radius:12px;background:var(--sunk);border:1px solid var(--line);aspect-ratio:3/4;overflow:hidden}
.v-similar .vs-ph img{width:100%;height:100%;object-fit:contain;border-radius:0;background:transparent}
.v-similar .vs-qi{display:grid;gap:10px;min-width:0}
.v-similar .vs-desc{font-size:13px;font-weight:500;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.v-similar .vs-sig{display:grid;gap:3px}
.v-similar .vs-sig .hd{display:flex;justify-content:space-between;gap:8px;font-size:11.5px;color:var(--ink-3);min-width:0}
.v-similar .vs-sig .hd b{color:var(--ink-2);font-weight:500;flex:none}
.v-similar .vs-sig .hd span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-similar .vs-sig svg{width:100%;height:40px;display:block;background:var(--sunk);border-radius:6px}
.v-similar .vs-sig g.b{transform-box:fill-box;transform-origin:50% 100%;transform:scaleY(0);transition:transform .7s var(--ease)}
.v-similar .vs-sig.go g.b{transform:scaleY(1)}
.v-similar .vs-pick{border-top:1px solid var(--line);padding:12px 16px 4px}
.v-similar .vs-strip{display:flex;gap:16px;overflow-x:auto;padding:8px 2px 12px}
.v-similar .vs-grp{display:grid;gap:6px;flex:none}
.v-similar .vs-grp .eyebrow{position:sticky;left:0;justify-self:start;white-space:nowrap}
.v-similar .vs-row{display:flex;gap:6px}
.v-similar .vs-thumb{flex:none;width:44px;height:60px;border-radius:9px;border:1.5px solid var(--line);background:var(--sunk);padding:0;overflow:hidden;transition:transform .2s var(--ease),border-color .2s,box-shadow .2s}
.v-similar .vs-thumb:hover{transform:translateY(-2px);border-color:var(--accent-line)}
.v-similar .vs-thumb.on{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.v-similar .vs-thumb img{width:100%;height:100%;object-fit:cover;display:block}
.v-similar .vs-act{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 16px 16px;margin-top:auto}
.v-similar .vs-run{padding:12px 22px;font-size:14.5px;font-weight:600;border-radius:12px}
.v-similar .vs-run .spin{border-color:color-mix(in srgb,var(--card) 35%,transparent);border-top-color:var(--card)}
.v-similar .vs-map{position:relative;border-radius:12px;background:var(--sunk);border:1px solid var(--line);overflow:hidden;min-height:440px}
.v-similar .vs-map canvas{display:block;width:100%;height:440px;cursor:default}
.v-similar .vs-ax{position:absolute;font-size:10.5px;color:var(--ink-3);pointer-events:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:calc(100% - 40px)}
.v-similar .vs-ax.x{left:50%;bottom:5px;transform:translateX(-50%)}
.v-similar .vs-ax.y{left:12px;top:50%;transform:translate(-50%,-50%) rotate(-90deg);max-width:380px}
.v-similar .vs-tip{position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);box-shadow:var(--shadow-lg);border-radius:10px;padding:8px 10px;font-size:12px;opacity:0;transform:translateY(4px);transition:opacity .15s,transform .15s;max-width:220px;z-index:2}
.v-similar .vs-tip.show{opacity:1;transform:none}
.v-similar .vs-lg{display:flex;gap:14px;flex-wrap:wrap;font-size:11.5px;color:var(--ink-3);margin-top:10px}
.v-similar .vs-lg span{display:inline-flex;gap:6px;align-items:center}
.v-similar .vs-lg i{width:12px;height:12px;border-radius:50%;border:2px solid var(--ink-3);display:inline-block}
.v-similar .vs-lg i.dash{border-style:dashed}
.v-similar .vs-lg i.ln{width:16px;height:0;border:0;border-top:2px solid var(--accent);border-radius:0}
.v-similar .vs-steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:12px}
.v-similar .vs-step{display:flex;gap:10px;align-items:center;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:var(--sunk);transition:border-color .3s,background .3s,box-shadow .3s;min-width:0}
.v-similar .vs-step .bub{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;flex:none;border:1.5px solid var(--line-2);color:var(--ink-3);font-size:11px;font-weight:600}
.v-similar .vs-step b{display:block;font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-similar .vs-step small{display:block;font:11px var(--f-mono);color:var(--ink-3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.v-similar .vs-step.run{border-color:var(--accent-line);background:var(--accent-soft);box-shadow:0 0 0 3px var(--accent-soft)}
.v-similar .vs-step.run .bub{border-color:var(--accent);color:var(--accent)}
.v-similar .vs-step.done .bub{background:var(--ok);border-color:var(--ok);color:var(--card)}
.v-similar .vs-step.done small{color:var(--ok)}
.v-similar .vs-sum{font-size:12px;color:var(--ink-3);margin:10px 0 14px;min-height:18px}
.v-similar .vs-res{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:12px}
.v-similar .vs-tile{border:1px solid var(--line);border-radius:12px;background:var(--card);overflow:hidden;cursor:pointer;transition:transform .2s var(--ease),border-color .2s,box-shadow .2s;min-width:0}
.v-similar .vs-tile:hover,.v-similar .vs-tile:focus-visible{transform:translateY(-3px);border-color:var(--accent);box-shadow:var(--shadow-lg)}
.v-similar .vs-tile .vs-ph{border:0;border-bottom:1px solid var(--line);border-radius:0;aspect-ratio:4/5}
.v-similar .vs-rk{position:absolute;left:8px;top:8px;background:var(--card);border:1px solid var(--line);border-radius:99px;font:600 11px var(--f-mono);padding:1px 7px;color:var(--ink-2)}
.v-similar .vs-rk.one{background:var(--accent);border-color:var(--accent);color:var(--card)}
.v-similar .vs-tb{padding:9px 10px 10px;display:grid;gap:6px}
.v-similar .vs-tb .pct{font-size:16px;font-weight:600;letter-spacing:-.02em;font-family:var(--f-mono)}
.v-similar .vs-tb .vs-desc{font-size:11.5px;font-weight:500}
.v-similar .vs-meta{font-size:11px;line-height:1.35;overflow-wrap:anywhere}
.v-similar .card-h .vs-sub{font-size:12.5px;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-similar .vs-badges{gap:4px}
.v-similar .vs-badges .pill{font-size:10.5px;padding:1px 7px}
@container (max-width:820px){
  .v-similar .vs-top{grid-template-columns:minmax(0,1fr)}
  .v-similar .vs-stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .v-similar .vs-stats .stat:nth-child(2){border-right:0}
  .v-similar .vs-stats .stat:nth-child(-n+2){border-bottom:1px solid var(--line)}
  .v-similar .vs-steps{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@container (max-width:1100px){.v-similar .vs-res{grid-template-columns:repeat(4,minmax(0,1fr))}}
@container (max-width:560px){.v-similar .vs-res{grid-template-columns:repeat(2,minmax(0,1fr))}}
@container (max-width:480px){.v-similar .vs-q{grid-template-columns:96px minmax(0,1fr)}.v-similar .vs-steps{grid-template-columns:minmax(0,1fr)}}`;

  const loadImg = url => new Promise(res => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = url; });
  const short = s => String(s.title || s.name).split(",")[0];
  const fmt = ms => ms < 0.01 ? "<0.01 ms" : ms < 10 ? ms.toFixed(2) + " ms" : Math.round(ms) + " ms";
  const sw = c => `<span class="swatch" style="background:${STM.colourHex[c] || "var(--sunk)"}"></span>`;
  const lookHtml = t => `${sw(t.look.top)}${sw(t.look.bottom)}<span>${esc(t.desc)}</span>`;
  const searchIco = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><circle cx="8" cy="9" r="4"/><circle cx="16" cy="15" r="4"/><path d="M11 12l2 1"/></svg>';

  function sigBlock(label, hist) {
    const W = 240, H = 40, bw = W / NB;
    const mx = Math.max(...hist, 1e-4);
    let s = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">`;
    hist.forEach((v, k) => {
      const b = BINS[k], h = Math.max(1, v / mx * (H - 5)), x = k * bw + 1.2, w = bw - 2.4, y = H - h;
      const hex = STM.colourHex[b.hue];
      const over = b.level === "dark" ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${STM.colourHex.black}" opacity=".42"/>`
        : b.level === "bright" && !["black", "grey", "white"].includes(b.hue) ? `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${STM.colourHex.white}" opacity=".3"/>` : "";
      s += `<g class="b" style="transition-delay:${k * 16}ms"><rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${hex}" ${b.hue === "white" ? 'stroke="var(--line-2)" vector-effect="non-scaling-stroke"' : ""}/>${over}</g>`;
    });
    s += "</svg>";
    const top2 = hist.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]).filter(a => a[0] >= .05).slice(0, 3)
      .map(([v, k]) => `${Math.round(v * 100)}% ${BINS[k].name}`).join(" · ");
    return el("div", { class: "vs-sig", html: `<div class="hd"><b>${label}</b><span>${esc(top2)}</span></div>${s}` });
  }

  STM.view("similar", {
    label: "Similar",
    async mount(root, ctx) {
      let dead = false, raf = 0, ro = null, runId = 0, scope = "scene", hoverKey = null, query = null, last = null;
      const timers = [], later = (fn, ms) => timers.push(setTimeout(() => { if (!dead) fn(); }, ms));
      const wrap = el("div", { class: "v-similar" });
      root.append(el("style", {}, CSS), wrap);
      STM.pageHead(wrap, "Find similar", "Pick a sighting to rank the others by clothing colour.");

      /* skeleton first */
      const statsCard = el("div", { class: "card vs-stats" });
      for (let i = 0; i < 4; i++) statsCard.append(el("div", { class: "stat" }, STM.skeleton(24, "50%"), el("div", { style: { height: "6px" } }), STM.skeleton(11, "70%")));
      const qCard = el("div", { class: "card" }, el("div", { class: "card-h" }, el("h3", {}, "Query")),
        el("div", { class: "card-b vs-q" }, STM.skeleton(176, "100%", 12), el("div", { class: "vs-qi" }, STM.skeleton(16, "70%"), STM.skeleton(12, "90%"), STM.skeleton(12, "60%"), STM.skeleton(40), STM.skeleton(40))));
      const mCard = el("div", { class: "card" }, el("div", { class: "card-h" }, el("h3", {}, "Appearance map")), el("div", { class: "card-b" }, STM.skeleton(360, "100%", 12)));
      const rCard = el("div", { class: "card" });
      wrap.append(statsCard, el("div", { class: "vs-top" }, qCard, mCard), rCard);
      const t0 = performance.now();

      /* load both scenes and every crop */
      const names = [ctx.scene ? ctx.scene.name : "lab", "lab", "caviar"].filter((n, i, a) => a.indexOf(n) === i);
      const scenes = (await Promise.all(names.map(n => STM.loadScene(n).catch(() => null)))).filter(Boolean);
      const items = [];
      for (const s of scenes) for (const t of s.tracks) items.push({ key: `${s.name}:${t.id}`, s, t, url: s.cropUrl(t), ident: (s.identities || []).find(i => i.tracks.includes(t.id)) || null });
      await Promise.all(items.map(async it => { it.img = await loadImg(it.url); }));
      if (!wrap.isConnected) return () => {};
      const scratch = document.createElement("canvas");
      const tb = performance.now();
      let wordsUsed = 0;
      for (const it of items) {
        try { if (!it.img) throw 0; [it.top, it.bot] = histOf(readPixels(it.img, scratch)); it.mode = "pixels"; }
        catch (e) { it.top = wordsHist(it.t.look.top); it.bot = wordsHist(it.t.look.bottom); it.mode = "words"; wordsUsed++; }
        it.vec = sig(it.top, it.bot);
      }
      const buildMs = performance.now() - tb;
      const P = items.length > 2 ? pca(items.map(it => it.vec)) : { pts: items.map(() => [0, 0]), vars: [0, 0] };
      await sleep(Math.max(0, 420 - (performance.now() - t0)));
      if (!wrap.isConnected || !items.length) { if (wrap.isConnected) rCard.replaceWith(el("div", { class: "card empty" }, "No crops found in the scenes.")); return () => {}; }

      /* stats */
      statsCard.innerHTML = "";
      const camCount = scenes.reduce((a, s) => a + s.cameras.length, 0);
      [[items.length, 0, "", `sightings from ${scenes.length} scenes`], [camCount, 0, "", "cameras"], [NB * 2, 0, "", "values per signature"], [buildMs, 1, " ms", "to build every signature"]].forEach(([v, dec, suf, lab]) => {
        const b = el("b", {}, "0"); statsCard.append(el("div", { class: "stat pop" }, b, el("span", {}, lab))); STM.countUp(b, v, { dec, suffix: suf });
      });

      /* query card */
      let seedKey = STM.state.similarSeed;
      if (seedKey && !String(seedKey).includes(":")) seedKey = `${names[0]}:${seedKey}`;
      const home = items.filter(it => it.s === scenes[0]);
      query = items.find(it => it.key === seedKey) || home.slice().sort((a, b) => (b.t.t1 - b.t.t0) - (a.t.t1 - a.t.t0))[0] || items[0];
      const qBody = el("div", { class: "card-b vs-q", id: "similar-query" });
      const strip = el("div", { class: "vs-strip", id: "similar-strip" });
      const thumbs = new Map();
      for (const s of scenes) {
        const row = el("div", { class: "vs-row" });
        items.filter(it => it.s === s).forEach(it => {
          const b = el("button", { type: "button", class: "vs-thumb", id: `similar-pick-${s.name}-${it.t.id}`, title: `${it.t.id}: ${it.t.desc}`, "aria-label": `Use ${it.t.id} as the query`, onclick: () => setQuery(it, false) },
            el("img", { src: it.url, alt: "", loading: "lazy" }));
          thumbs.set(it.key, b); row.append(b);
        });
        strip.append(el("div", { class: "vs-grp" }, el("span", { class: "eyebrow" }, `${short(s)} · ${s.tracks.length}`), row));
      }
      const runBtn = el("button", { type: "button", class: "btn btn-primary vs-run", id: "similar-run", onclick: () => run() });
      runBtn.innerHTML = `${searchIco}Find similar`;
      const chipOne = el("button", { type: "button", class: "chip on", id: "similar-scope-scene", title: "Compare with sightings from the same venue", onclick: () => setScope("scene") }, "Same scene");
      const chipBoth = el("button", { type: "button", class: "chip", id: "similar-scope-both", title: "Also compare with the other venue", onclick: () => setScope("both") }, "Both scenes");
      qCard.innerHTML = "";
      qCard.append(
        el("div", { class: "card-h" }, el("h3", {}, "Query"), el("div", { class: "grow" }),
          wordsUsed ? el("span", { class: "pill warn" }, `${wordsUsed} crops fell back to index words`) : null),
        qBody,
        el("div", { class: "vs-pick" }, el("span", { class: "eyebrow" }, "Pick another"), strip),
        el("div", { class: "vs-act" }, runBtn, el("div", { class: "row" }, chipOne, chipBoth)));

      function renderQuery() {
        const it = query, t = it.t, s = it.s, cam = s.cam(t.cam);
        qBody.innerHTML = "";
        qBody.append(el("div", { class: "vs-ph pop" }, el("img", { class: "crop", src: it.url, alt: `Crop of ${t.id}`, id: "similar-query-img" })),
          el("div", { class: "vs-qi stagger" },
            el("div", { class: "row wrap", style: { gap: "6px" } }, el("b", { class: "mono", style: { fontSize: "15px" } }, t.id), el("span", { class: "muted", style: { fontSize: "12.5px" } }, short(s)), it.mode === "words" ? el("span", { class: "pill warn" }, "From index words") : null),
            el("div", { class: "vs-desc", html: lookHtml(t) }),
            el("dl", { class: "kv", style: { margin: 0 }, html: `<dt>Camera</dt><dd>${esc(t.cam)} · ${esc(cam ? cam.name : "")}</dd><dt>Seen</dt><dd class="mono">${s.clock(t.t0)} to ${s.clock(t.t1)}</dd><dt>On screen</dt><dd>${(t.t1 - t.t0).toFixed(1)} s</dd>` }),
            sigBlock("Upper body", it.top), sigBlock("Lower body", it.bot)));
        thumbs.forEach((b, k) => b.classList.toggle("on", k === it.key));
        const th = thumbs.get(it.key);
        if (th) strip.scrollTo({ left: Math.max(0, th.offsetLeft - strip.offsetLeft - strip.clientWidth / 2 + 22), behavior: "smooth" });
        requestAnimationFrame(() => requestAnimationFrame(() => qBody.querySelectorAll(".vs-sig").forEach(n => n.classList.add("go"))));
      }

      /* appearance map */
      const cv = el("canvas", { id: "similar-map", role: "img", "aria-label": `Appearance map of ${items.length} sightings. Close dots have similar clothing colours.` });
      const tip = el("div", { class: "vs-tip" });
      const pct = v => Math.round(v * 100);
      const mapBox = el("div", { class: "vs-map" }, cv, tip,
        el("span", { class: "vs-ax x" }, `Component 1 · ${pct(P.vars[0])}% of variance`),
        el("span", { class: "vs-ax y" }, `Component 2 · ${pct(P.vars[1])}%`));
      mCard.innerHTML = "";
      mCard.append(el("div", { class: "card-h" }, el("h3", {}, "Appearance map")),
        el("div", { class: "card-b" }, mapBox,
          el("div", { class: "vs-lg", html: `<span><i></i>Ring: top colour</span><span><i class="dash"></i>Dashed: ${esc(scenes[1] ? short(scenes[1]) : "other scene")}</span><span><i class="ln"></i>Query to top matches</span>` })));
      const ctx2 = cv.getContext("2d"), dpr = Math.min(2, window.devicePixelRatio || 1);
      let W = 0, H = 440, col = {}, frameN = 0, matches = [];
      const pts = items.map((it, i) => ({ it, i, x: 0, y: 0, fx: 0, fy: 0, tx: 0, ty: 0, ts: 0, td: 1, first: true }));
      const readCols = () => { const cs = getComputedStyle(document.documentElement), g = n => cs.getPropertyValue(n).trim(); col = { accent: g("--accent"), line: g("--line"), line2: g("--line-2"), ink: g("--ink"), ink3: g("--ink-3"), sunk: g("--sunk"), card: g("--card"), ok: g("--ok") }; };
      function layout() {
        const r = cv.getBoundingClientRect(); if (!r.width || Math.abs(r.width - W) < 1) return;
        W = r.width; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
        const pad = 30, xs = P.pts.map(p => p[0]), ys = P.pts.map(p => p[1]);
        const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
        pts.forEach((p, i) => { p.tx = pad + (xs[i] - x0) / ((x1 - x0) || 1) * (W - 2 * pad); p.ty = pad + (1 - (ys[i] - y0) / ((y1 - y0) || 1)) * (H - 2 * pad - 30); });
        for (let k = 0; k < 60; k++) for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {   // nudge overlapping dots apart
          const a = pts[i], b = pts[j], dx = b.tx - a.tx, dy = b.ty - a.ty, d = Math.hypot(dx, dy) || .01, md = 32;
          if (d < md) { const m = (md - d) / 2, ux = dx / d, uy = dy / d; a.tx -= ux * m; a.ty -= uy * m; b.tx += ux * m; b.ty += uy * m; }
        }
        const now = performance.now();
        pts.forEach(p => {
          p.tx = Math.min(W - 26, Math.max(30, p.tx)); p.ty = Math.min(H - 52, Math.max(26, p.ty));
          if (p.first) { p.x = p.fx = Math.random() * W; p.y = p.fy = Math.random() * H; p.ts = now + p.i * 24; p.td = 1100; p.first = false; }
          else if (now > p.ts + p.td) { p.fx = p.x; p.fy = p.y; p.ts = now; p.td = 450; }
        });
      }
      function drawDot(p, r, ring, alpha) {
        const c = ctx2; c.save(); c.globalAlpha = alpha;
        c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2); c.fillStyle = col.card; c.fill();
        if (p.it.img) {
          c.save(); c.clip(); const im = p.it.img, sc = (2 * r) / im.naturalWidth, ih = im.naturalHeight * sc;
          c.drawImage(im, p.x - r, p.y - r - Math.max(0, ih - 2 * r) * .3, 2 * r, ih); c.restore();
        }
        c.beginPath(); c.arc(p.x, p.y, r, 0, Math.PI * 2); c.lineWidth = 2.2; c.strokeStyle = ring;
        if (p.it.s !== scenes[0]) c.setLineDash([3, 2.5]); c.stroke(); c.restore();
      }
      function frame(now) {
        raf = requestAnimationFrame(frame);
        if (!W) return;
        if (frameN++ % 30 === 0) readCols();
        const c = ctx2; c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
        c.strokeStyle = col.line; c.lineWidth = 1; c.beginPath();
        for (let gx = 1; gx < 6; gx++) { c.moveTo(W * gx / 6, 0); c.lineTo(W * gx / 6, H); }
        for (let gy = 1; gy < 4; gy++) { c.moveTo(0, H * gy / 4); c.lineTo(W, H * gy / 4); }
        c.stroke();
        for (const p of pts) {
          const f = Math.min(1, Math.max(0, (now - p.ts) / p.td)), e = 1 - Math.pow(1 - f, 3);
          p.x = p.fx + (p.tx - p.fx) * e; p.y = p.fy + (p.ty - p.fy) * e;
        }
        const qp = pts.find(p => p.it === query), mk = new Map(matches.map(m => [m.key, m]));
        if (qp) for (const m of matches) {                              // lines grow from the query out to each match
          const p = pts[m.idx], f = Math.min(1, Math.max(0, (now - m.t) / 550)); if (!f) continue;
          c.save(); c.globalAlpha = .25 + .6 * m.sim * (m.rank < 3 ? 1 : .7); c.strokeStyle = col.accent; c.lineWidth = m.rank === 0 ? 2.6 : 1.6;
          c.beginPath(); c.moveTo(qp.x, qp.y); c.lineTo(qp.x + (p.x - qp.x) * f, qp.y + (p.y - qp.y) * f); c.stroke(); c.restore();
        }
        const dim = matches.length ? .45 : 1;
        for (const p of pts) if (p !== qp && !mk.has(p.it.key)) drawDot(p, p.it.key === hoverKey ? 15 : 11, STM.colourHex[p.it.t.look.top] || col.ink3, p.it.key === hoverKey ? 1 : dim);
        for (const m of matches) {
          const p = pts[m.idx], age = now - m.t - 550; if (age < -550) continue;
          if (age > 0 && age < 3600) {                                  // pulse a few times, then hold a steady ring
            const ph = (age % 1200) / 1200; c.save(); c.globalAlpha = (1 - ph) * .8; c.strokeStyle = col.accent; c.lineWidth = 2;
            c.beginPath(); c.arc(p.x, p.y, 15 + ph * 14, 0, Math.PI * 2); c.stroke(); c.restore();
          }
          c.save(); c.beginPath(); c.arc(p.x, p.y, 17, 0, Math.PI * 2); c.strokeStyle = col.accent; c.globalAlpha = age > 0 ? .9 : .3; c.lineWidth = 2; c.stroke(); c.restore();
          drawDot(p, p.it.key === hoverKey ? 16 : 13.5, STM.colourHex[p.it.t.look.top] || col.ink3, 1);
          if (age > 0) {
            c.save(); c.font = "600 10px 'JetBrains Mono', ui-monospace, monospace"; const lab = `#${m.rank + 1}`, tw = c.measureText(lab).width + 8;
            c.fillStyle = m.rank === 0 ? col.accent : col.card; c.strokeStyle = col.accent; c.lineWidth = 1;
            c.beginPath(); c.roundRect ? c.roundRect(p.x + 9, p.y - 22, tw, 14, 7) : c.rect(p.x + 9, p.y - 22, tw, 14); c.fill(); c.stroke();
            c.fillStyle = m.rank === 0 ? col.card : col.accent; c.fillText(lab, p.x + 13, p.y - 11.5); c.restore();
          }
        }
        if (qp) {
          const br = 1 + .06 * Math.sin(now / 380);
          c.save(); c.beginPath(); c.arc(qp.x, qp.y, 23 * br, 0, Math.PI * 2); c.fillStyle = col.accent; c.globalAlpha = .14; c.fill(); c.restore();
          c.save(); c.beginPath(); c.arc(qp.x, qp.y, 20, 0, Math.PI * 2); c.strokeStyle = col.accent; c.lineWidth = 3; c.stroke(); c.restore();
          drawDot(qp, 16, STM.colourHex[qp.it.t.look.top] || col.ink3, 1);
          c.save(); c.font = "600 10.5px Outfit, system-ui, sans-serif"; c.fillStyle = col.accent; c.textAlign = "center"; c.fillText("Query", qp.x, qp.y + 34); c.restore();
        }
      }
      const hit = e => {
        const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
        let best = null, bd = 18; for (const p of pts) { const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = p; } }
        return best;
      };
      cv.addEventListener("pointermove", e => {
        const p = hit(e); hoverKey = p ? p.it.key : null; cv.style.cursor = p ? "pointer" : "default";
        if (!p) { tip.classList.remove("show"); return; }
        const m = last && last.scored.find(r => r.it === p.it);
        tip.innerHTML = `<b class="mono">${esc(p.it.t.id)}</b> <span class="muted">· ${esc(short(p.it.s))}</span><div class="vs-desc" style="font-size:12px;margin-top:3px">${lookHtml(p.it.t)}</div>` +
          (p.it === query ? `<div class="muted" style="margin-top:3px">Current query</div>` : m ? `<div class="muted" style="margin-top:3px">Similarity <b class="mono" style="color:var(--ink)">${(m.sim * 100).toFixed(1)}%</b></div>` : `<div class="muted" style="margin-top:3px">Click to search from here</div>`);
        tip.style.left = Math.min(W - 230, p.x + 18) + "px"; tip.style.top = Math.max(6, p.y - 20) + "px"; tip.classList.add("show");
      });
      cv.addEventListener("pointerleave", () => { hoverKey = null; tip.classList.remove("show"); });
      cv.addEventListener("click", e => { const p = hit(e); if (p && p.it !== query) setQuery(p.it, true); });

      /* results card */
      const stepEls = [0, 1, 2, 3].map(i => el("div", { class: "vs-step" }, el("span", { class: "bub" }, String(i + 1)), el("div", { style: { minWidth: 0 } }, el("b", {}, ""), el("small", {}, "waiting"))));
      const prog = el("i"), sum = el("div", { class: "vs-sum", id: "similar-summary" }), res = el("div", { class: "vs-res", id: "similar-results" });
      const skelTiles = () => { res.innerHTML = ""; for (let i = 0; i < 8; i++) res.append(el("div", { class: "vs-tile", style: { cursor: "default" } }, STM.skeleton("auto", "100%", 0), el("div", { class: "vs-tb" }, STM.skeleton(16, "50%"), STM.skeleton(6), STM.skeleton(11, "85%")))); res.querySelectorAll(".vs-tile > .skel").forEach(n => { n.style.aspectRatio = "4/5"; }); };
      rCard.append(el("div", { class: "card-h" }, el("h3", {}, "Ranked matches"), el("span", { class: "muted vs-sub" }, "Top 8 by colour similarity")),
        el("div", { class: "card-b" }, el("div", { class: "vs-steps", id: "similar-steps" }, stepEls), el("div", { class: "prog" }, prog), sum, res));
      skelTiles();

      function setStep(i, state, note) { const s = stepEls[i]; s.className = "vs-step " + state; s.querySelector(".bub").innerHTML = state === "done" ? STM.icons.check : state === "run" ? '<span class="spin" style="width:12px;height:12px"></span>' : String(i + 1); if (note != null) s.querySelector("small").textContent = note; }
      function setScope(v) { if (v === scope) return; scope = v; chipBoth.classList.toggle("on", v === "both"); chipOne.classList.toggle("on", v === "scene"); run(); }
      function setQuery(it, andRun) {
        query = it; STM.state.similarSeed = it.key; matches = []; last = null; renderQuery(); STM.emit("similar", { seed: it.key });
        if (andRun) run();
        else { runId++; resetSteps(); skelTiles(); sum.textContent = "Query changed. Press Find similar to rank the other sightings."; runBtn.classList.remove("flash"); void runBtn.offsetWidth; runBtn.classList.add("flash"); runBtn.disabled = false; runBtn.innerHTML = `${searchIco}Find similar`; }
      }
      function resetSteps() { const n = items.filter(it => it !== query && (scope === "both" || it.s === query.s)).length; ["Reading pixels", "Building colour signature", `Comparing with ${n} sightings`, "Ranking"].forEach((l, i) => { stepEls[i].querySelector("b").textContent = l; setStep(i, "", "waiting"); }); prog.style.width = "0%"; }

      async function run() {
        const my = ++runId, q = query, alive = () => !dead && my === runId;
        const pool = items.filter(it => it !== q && (scope === "both" || it.s === q.s));
        resetSteps(); matches = []; skelTiles(); sum.innerHTML = '<span class="dots"><i></i><i></i><i></i></span>';
        runBtn.innerHTML = '<span class="spin"></span>Searching'; runBtn.setAttribute("aria-busy", "true");
        const times = [], pace = async i => { setStep(i, "done", fmt(times[i])); prog.style.width = ((i + 1) * 25) + "%"; await sleep(260); return alive(); };
        let px = null, tt;
        setStep(0, "run", "reading"); await sleep(160); if (!alive()) return;
        tt = performance.now(); try { if (q.img && q.mode === "pixels") px = readPixels(q.img, scratch); } catch (e) { px = null; } times[0] = performance.now() - tt;
        if (!await pace(0)) return;
        setStep(1, "run", "building"); await sleep(120); if (!alive()) return;
        tt = performance.now(); const h = px ? histOf(px) : [q.top, q.bot], qv = sig(h[0], h[1]); times[1] = performance.now() - tt;
        if (!await pace(1)) return;
        setStep(2, "run", "comparing"); await sleep(120); if (!alive()) return;
        tt = performance.now(); const scored = pool.map(it => ({ it, sim: cosine(qv, it.vec), up: dot(qv, it.vec, 0, NB), low: dot(qv, it.vec, NB, 2 * NB) })); times[2] = performance.now() - tt;
        if (!await pace(2)) return;
        setStep(3, "run", "sorting"); await sleep(120); if (!alive()) return;
        tt = performance.now(); scored.sort((a, b) => b.sim - a.sim); const top = scored.slice(0, 8); times[3] = performance.now() - tt;
        setStep(3, "done", fmt(times[3])); prog.style.width = "100%";
        runBtn.innerHTML = `${searchIco}Find similar`; runBtn.removeAttribute("aria-busy");
        last = { scored, top };
        const total = times.reduce((a, b) => a + b, 0);
        sum.innerHTML = `Ranked <b>${pool.length}</b> sightings in <b class="mono">${fmt(total)}</b>.`;
        const now = performance.now();
        matches = top.map((r, rank) => ({ key: r.it.key, idx: items.indexOf(r.it), sim: r.sim, rank, t: now + rank * 110 }));
        renderResults(top);
      }

      function renderResults(top) {
        res.innerHTML = ""; res.classList.remove("stagger"); void res.offsetWidth; res.classList.add("stagger");
        const q = query;
        top.forEach((r, i) => {
          const it = r.it, t = it.t, s = it.s, cam = s.cam(t.cam);
          const same = it.s === q.s && q.ident && it.ident === q.ident;
          const hop = same ? (q.ident.hops || []).find(hp => [hp.from, hp.to].includes(t.id)) : null;
          const pctEl = el("b", { class: "mono pct" }, "0%"), bar = el("i");
          const tile = el("div", {
            class: "vs-tile", id: `similar-res-${i + 1}`, role: "button", tabindex: "0", title: "Search from this sighting",
            onclick: () => setQuery(it, true), onkeydown: e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setQuery(it, true); } },
            onmouseenter: () => { hoverKey = it.key; }, onmouseleave: () => { hoverKey = null; },
          },
            el("div", { class: "vs-ph" }, el("img", { class: "crop", src: it.url, alt: `Crop of ${t.id}` }), el("span", { class: "vs-rk" + (i === 0 ? " one" : "") }, `#${i + 1}`)),
            el("div", { class: "vs-tb" },
              el("div", { class: "row", style: { justifyContent: "space-between" } }, pctEl, el("span", { class: "muted", style: { fontSize: "11px" } }, "match")),
              el("div", { class: "prog" + (same ? " ok" : "") }, bar),
              el("div", { class: "vs-desc", html: lookHtml(t) }),
              el("div", { class: "vs-meta muted" }, `${t.id} · Cam ${t.cam} ${cam ? cam.name : ""} · ${s.clock(t.t0)}${it.s !== q.s ? " · " + short(s) : ""}`),
              el("div", { class: "vs-meta muted mono" }, `upper ${pct(r.up)} · lower ${pct(r.low)}`),
              el("div", { class: "row wrap vs-badges" },
                same ? el("span", { class: "pill ok", title: hop ? `Follow link: ${hop.why}` : "Same identity in the index" }, "Same person") : null)));
          res.append(tile);
          later(() => { STM.countUp(pctEl, r.sim * 100, { ms: 900, dec: 1, suffix: "%" }); bar.style.width = (r.sim * 100).toFixed(1) + "%"; }, 140 + i * 70);
        });
        if (!top.length) res.append(el("div", { class: "empty" }, "No other sightings in this scope."));
      }

      renderQuery(); resetSteps();
      sum.textContent = "Preparing the first search…";
      readCols();
      ro = new ResizeObserver(() => layout()); ro.observe(cv);
      layout(); raf = requestAnimationFrame(frame);
      later(() => run(), 650);

      const api = { run: () => run(), pick: key => { const it = items.find(x => x.key === key || x.t.id === key); if (it) setQuery(it, false); return !!it; } };
      STM.similar = api;
      return () => { dead = true; runId++; cancelAnimationFrame(raf); timers.forEach(clearTimeout); if (ro) ro.disconnect(); if (STM.similar === api) delete STM.similar; };
    },
  });
})();
