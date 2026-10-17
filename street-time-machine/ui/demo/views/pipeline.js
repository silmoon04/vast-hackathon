/* Pipeline view: how a camera clip becomes a searchable people index. Every number comes from the real scene builds
   (scene.build) or STM.facts. The "Run on a new clip" button replays today's caviar build log at about real speed. */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const S = (tag, attrs = {}, ...kids) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
    for (const c of kids) if (c != null) n.append(c.nodeType ? c : document.createTextNode(c));
    return n;
  };
  const fmt = v => String(Math.round(v * 10) / 10);
  const ICON = {
    cameras: '<path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>',
    decode: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    detect: '<path d="M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3"/><rect x="9" y="8" width="6" height="8" rx="1"/>',
    track: '<circle cx="5" cy="17" r="2"/><circle cx="12" cy="10" r="2"/><circle cx="19" cy="6" r="2"/><path d="M6.5 15.5l4-4M13.7 9l3.6-2"/>',
    follow: '<rect x="2" y="7" width="7" height="10" rx="2"/><rect x="15" y="7" width="7" height="10" rx="2"/><path d="M9 12h6M13 10l2 2-2 2"/>',
    describe: '<path d="M20.6 13.4l-7.2 7.2a2 2 0 01-2.8 0L3 13V3h10l7.6 7.6a2 2 0 010 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
    index: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    agent: '<path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/><path d="M8 9h8M8 13h5"/>',
    desk: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
  };
  /* hide axis labels that would collide with the previous visible one; the last label always stays */
  function thinLabels(box) {
    if (!box) return;
    const sp = [...box.children]; sp.forEach(n => { n.style.visibility = ""; });
    const r = sp.map(n => n.getBoundingClientRect()); if (!r.length || !r[0].width) return;
    let prev = -1;
    sp.forEach((n, i) => {
      if (prev >= 0 && r[i].left < r[prev].right + 8) {
        if (i === sp.length - 1) { sp[prev].style.visibility = "hidden"; prev = i; } else n.style.visibility = "hidden";
      } else prev = i;
    });
  }
  const svgIcon = d => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;

  const CSS = `
.v-pipeline{container-type:inline-size;display:grid;gap:16px;min-width:0}
.v-pipeline .pl-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}
.v-pipeline .pl-tile{padding:14px 16px;display:grid;gap:4px;position:relative;overflow:hidden;min-width:0;transition:transform .25s var(--ease),box-shadow .25s}
.v-pipeline .pl-tile .eyebrow,.v-pipeline .pl-tile .pl-sub{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;display:block}
.v-pipeline .pl-tile:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg)}
.v-pipeline .pl-tile b{font-size:26px;margin-top:4px}
.v-pipeline .pl-tile .pl-sub{font-size:12px;color:var(--ink-3)}
.v-pipeline .card-h{min-width:0;flex-wrap:wrap}
.v-pipeline .card-h .pl-cap{font-size:12px;color:var(--ink-3);min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-pipeline .pl-ghost{padding:18px 16px 10px;min-height:150px}
.v-pipeline .pl-graph{display:block;overflow:visible;font-family:var(--f-body)}
.v-pipeline .pl-wire{fill:none;stroke:var(--line-2);stroke-width:1.5;stroke-dasharray:3 4}
.v-pipeline .pl-arrow{fill:none;stroke:var(--ink-4);stroke-width:1.5;stroke-linecap:round;stroke-linejoin:round}
.v-pipeline .pl-dot{fill:var(--accent)}
.v-pipeline .pl-node{cursor:pointer;outline:none}
.v-pipeline .pl-in.first{animation:plIn .55s var(--ease) both}
@keyframes plIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
.v-pipeline .pl-node .bg{fill:var(--card);stroke:var(--line);stroke-width:1;transition:stroke .3s,fill .3s}
.v-pipeline .pl-node:hover .bg,.v-pipeline .pl-node:focus-visible .bg,.v-pipeline .pl-node.sel .bg{stroke:var(--accent-line);fill:var(--sunk)}
.v-pipeline .pl-node .halo{fill:none;stroke:var(--accent);stroke-width:2;opacity:0;transform-box:fill-box;transform-origin:center}
.v-pipeline .pl-node.is-run .bg{stroke:var(--accent);stroke-width:1.5}
.v-pipeline .pl-node.is-run .halo{animation:plHalo 1.1s var(--ease) infinite}
@keyframes plHalo{from{opacity:.55;transform:scale(1)}to{opacity:0;transform:scale(1.09)}}
.v-pipeline .pl-node .badge{fill:var(--accent-soft);transition:fill .3s}
.v-pipeline .pl-node .ico-g{fill:none;stroke:var(--accent);stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round;transition:stroke .3s}
.v-pipeline .pl-node.is-run .badge{fill:var(--accent)} .v-pipeline .pl-node.is-run .ico-g{stroke:var(--card)}
.v-pipeline .pl-node.is-wait .badge{fill:var(--sunk)} .v-pipeline .pl-node.is-wait .ico-g{stroke:var(--ink-3)}
.v-pipeline .pl-node .nm{fill:var(--ink);font-size:13px;font-weight:600}
.v-pipeline .pl-node.is-wait .nm{fill:var(--ink-3)}
.v-pipeline .pl-node .sb{fill:var(--ink-3);font-size:10.5px}
.v-pipeline .pl-node .mt{fill:var(--ink-2);font-family:var(--f-mono);font-size:10px}
.v-pipeline .pl-node.is-run .mt{fill:var(--accent)}
.v-pipeline .pl-node .pbg{fill:var(--sunk)} .v-pipeline .pl-node .pfill{fill:var(--accent)}
.v-pipeline .pl-node.is-done .pfill{fill:var(--ok)}
.v-pipeline .pl-node .r-wait{fill:var(--ink-4)} .v-pipeline .pl-node .r-track{fill:none;stroke:var(--accent-line);stroke-width:2.2}
.v-pipeline .pl-node .r-spin{fill:none;stroke:var(--accent);stroke-width:2.2;stroke-linecap:round;transform-box:fill-box;transform-origin:center;animation:spin .8s linear infinite}
.v-pipeline .pl-node .r-done circle{fill:var(--ok)}
.v-pipeline .pl-node .r-done path{fill:none;stroke:var(--card);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.v-pipeline .pl-node .r-wait,.v-pipeline .pl-node .r-track,.v-pipeline .pl-node .r-spin,.v-pipeline .pl-node .r-done{display:none}
.v-pipeline .pl-node.is-wait .r-wait,.v-pipeline .pl-node.is-run .r-track,.v-pipeline .pl-node.is-run .r-spin,.v-pipeline .pl-node.is-done .r-done{display:inline}
.v-pipeline .pl-node.is-done .r-done{animation:plPop .4s var(--ease) both;transform-box:fill-box;transform-origin:center}
@keyframes plPop{from{transform:scale(.4);opacity:0}to{transform:none;opacity:1}}
.v-pipeline .pl-gfoot{display:flex;align-items:center;gap:12px;flex-wrap:wrap;padding:12px 16px;border-top:1px solid var(--line);background:var(--sunk);border-radius:0 0 var(--r-lg) var(--r-lg);font-size:12.5px}
.v-pipeline .pl-detail{flex:1 1 320px;min-width:0;color:var(--ink-2)}
.v-pipeline .pl-detail b{color:var(--ink);margin-right:6px}
.v-pipeline .pl-clock{display:flex;align-items:center;gap:8px;color:var(--ink-3);flex:0 1 300px}
.v-pipeline .pl-clock .prog{flex:1;min-width:80px}
.v-pipeline .pl-clock .prog>i{transition:none}
.v-pipeline .pl-clock .mono{color:var(--ink);min-width:72px;text-align:right}
.v-pipeline .pl-bspin{width:13px;height:13px;border-radius:50%;border:2px solid color-mix(in srgb,currentColor 35%,transparent);border-top-color:currentColor;animation:spin .8s linear infinite}
.v-pipeline .pl-two{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:16px;align-items:start}
.v-pipeline .pl-two.rev{grid-template-columns:minmax(0,1fr) minmax(0,1.2fr)}
.v-pipeline .pl-three{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.v-pipeline .pl-keeps td:nth-child(2){min-width:170px}
.v-pipeline .pl-keeps td{white-space:normal}
.v-pipeline .pl-log{height:244px;overflow:auto;margin:0;border:0;border-radius:0;background:var(--sunk);padding:12px 16px}
.v-pipeline .pl-log>div{animation:rise .3s var(--ease) both}
.v-pipeline .pl-caret{display:inline-block;width:7px;height:13px;background:var(--accent);vertical-align:-2px;animation:plBlink 1s steps(2) infinite}
@keyframes plBlink{50%{opacity:0}}
.v-pipeline .pl-note{font-size:11.5px;color:var(--ink-3);padding:10px 16px;border-top:1px solid var(--line)}
.v-pipeline .pl-gantt{display:grid;grid-template-columns:minmax(104px,150px) minmax(0,1fr);gap:0 14px;font-size:12px}
.v-pipeline .pl-glab div{min-width:0}
.v-pipeline .pl-glab span>span,.v-pipeline .pl-glab small{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-pipeline .pl-glab em{flex:none}
.v-pipeline .pl-glab,.v-pipeline .pl-gtr{display:grid;gap:10px;align-content:start}
.v-pipeline .pl-glab div{height:26px;display:flex;flex-direction:column;justify-content:center;line-height:1.15}
.v-pipeline .pl-glab small{color:var(--ink-3);font-size:10.5px}
.v-pipeline .pl-glab span{display:flex;justify-content:space-between;gap:6px}
.v-pipeline .pl-glab em{font:11px var(--f-mono);font-style:normal;color:var(--ink-2)}
.v-pipeline .pl-gtr{position:relative;padding-bottom:22px}
.v-pipeline .pl-lane{height:26px;background:var(--sunk);border:1px solid var(--line);border-radius:7px;position:relative}
.v-pipeline .pl-lane i{position:absolute;top:3px;bottom:3px;width:0;min-width:3px;border-radius:4px;background:color-mix(in srgb,var(--accent) 20%,transparent);border-left:3px solid var(--accent);transition:width 1s var(--ease)}
.v-pipeline .pl-lane i.f{background:color-mix(in srgb,var(--violet) 22%,transparent);border-left-color:var(--violet)}
.v-pipeline .pl-lane i.t{background:color-mix(in srgb,var(--ok) 18%,transparent);border-left-color:var(--ok)}
.v-pipeline .pl-axis{position:absolute;left:0;right:0;bottom:0;height:16px;font:10.5px var(--f-mono);color:var(--ink-3)}
.v-pipeline .pl-axis span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.v-pipeline .pl-axis span:last-child{transform:translateX(-100%)}
.v-pipeline .pl-axis span:first-child{transform:none}
.v-pipeline .pl-head{position:absolute;top:-4px;bottom:18px;width:2px;margin-left:-1px;background:var(--accent);border-radius:2px;left:0;opacity:0;transition:opacity .3s}
.v-pipeline .pl-head.on{opacity:1}
.v-pipeline .pl-crops{display:grid;grid-template-columns:repeat(auto-fill,minmax(70px,1fr));gap:10px;min-height:118px}
.v-pipeline .pl-crop{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--card);padding:0;text-align:left;transition:transform .2s var(--ease),border-color .2s}
.v-pipeline .pl-crop:hover{transform:translateY(-2px);border-color:var(--accent)}
.v-pipeline .pl-crop img{width:100%;aspect-ratio:3/4;object-fit:cover;display:block;background:var(--sunk)}
.v-pipeline .pl-crop div{padding:4px 6px;font:10.5px var(--f-mono);display:flex;gap:4px;align-items:center;color:var(--ink-2)}
.v-pipeline .pl-cmp{display:grid;gap:10px}
.v-pipeline .pl-crow{display:grid;grid-template-columns:minmax(96px,150px) minmax(0,1fr) 78px;gap:12px;align-items:center;font-size:12.5px}
.v-pipeline .pl-crow>div{min-width:0}
.v-pipeline .pl-crow>div:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-pipeline .pl-crow small{display:block;color:var(--ink-3);font-size:11px;overflow:hidden;text-overflow:ellipsis}
.v-pipeline .pl-keeps td.k-no{color:var(--ink-3)}
.v-pipeline .pl-crow .mono{text-align:right;font-size:12px}
.v-pipeline .pl-cbar{height:12px;border-radius:99px;background:var(--sunk);border:1px solid var(--line);position:relative;overflow:hidden}
.v-pipeline .pl-cbar i{position:absolute;left:0;top:0;bottom:0;width:0;border-radius:99px;transition:width 1.2s var(--ease)}
.v-pipeline .pl-cbar i.lo{background:var(--accent)} .v-pipeline .pl-cbar i.hi{background:color-mix(in srgb,var(--accent) 30%,transparent)}
.v-pipeline .pl-cbar.w i.lo{background:var(--warn)} .v-pipeline .pl-cbar.w i.hi{background:color-mix(in srgb,var(--warn) 30%,transparent)}
.v-pipeline .pl-cgrp{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);margin-top:4px}
.v-pipeline .pl-cax{position:relative;height:14px;font:10.5px var(--f-mono);color:var(--ink-3)}
.v-pipeline .pl-cax span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.v-pipeline .pl-keeps tr.pl-hl td{animation:plHl 1.4s var(--ease)}
@keyframes plHl{from{background:var(--accent-soft)}to{background:transparent}}
.v-pipeline .pl-pow{padding:16px;display:grid;gap:10px;align-content:start;min-width:0;overflow:hidden;transition:transform .25s var(--ease),box-shadow .25s,border-color .25s}
.v-pipeline .pl-pow:hover{transform:translateY(-2px);box-shadow:var(--shadow-lg);border-color:var(--accent-line)}
.v-pipeline .pl-pow p{margin:0;font-size:12.5px;color:var(--ink-2)}
.v-pipeline .pl-mini{height:64px;position:relative;background:var(--sunk);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.v-pipeline .pl-mini i{position:absolute;height:4px;border-radius:2px;width:0;transition:width .9s var(--ease)}
.v-pipeline .pl-thumbs{display:flex;gap:6px;height:64px;overflow:hidden;min-width:0}
.v-pipeline .pl-pow .row{min-width:0;flex-wrap:nowrap}
.v-pipeline .pl-pow h3{flex:none}
.v-pipeline .pl-pow .pl-pn{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;color:var(--ink-3)}
.v-pipeline .pl-pow .btn{justify-self:start}
.v-pipeline .pl-pow small{color:var(--ink-3);font-size:11.5px;margin-top:-4px}
.v-pipeline .pl-thumbs img{height:64px;width:48px;flex:none}
@container (max-width:960px){
  .v-pipeline .pl-two,.v-pipeline .pl-two.rev{grid-template-columns:minmax(0,1fr)}
}
@container (max-width:820px){
  .v-pipeline .pl-stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .v-pipeline .pl-three{grid-template-columns:minmax(0,1fr)}
  .v-pipeline .pl-crow{grid-template-columns:110px minmax(0,1fr) 70px}
}
@media (max-width:820px){
  .v-pipeline .pl-stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .v-pipeline .pl-two,.v-pipeline .pl-two.rev,.v-pipeline .pl-three{grid-template-columns:minmax(0,1fr)}
}`;

  /* one build summary per scene, from scene.build with STM.facts as the fallback */
  function buildOf(s, key) {
    const f = STM.facts[key + "Build"] || {}, b = (s && s.build) || {}, per = b.per_camera || [];
    return {
      key, scene: s, title: s ? s.title : key, cams: s ? s.cameras.length : f.cams, dur: s ? s.duration : f.footage_s,
      seconds: b.seconds ?? f.seconds, index_s: b.index_s ?? f.index_s, follow_s: b.follow_s || 0, fps: b.fps ?? f.fps,
      fragments: b.fragments ?? f.fragments, tracks: b.person_tracks ?? f.tracks, hops: b.hops || 0, per,
      size: per[0] ? per[0].size : null, samples: per.reduce((a, p) => a + (p.samples || 0), 0),
      describe: b.describe || "colour words", videos: b.videos || {}, people: s && s.identities ? s.identities.length : null,
    };
  }
  const stagesOf = b => [
    { id: "cameras", name: "Cameras", sub: `${b.cams} fixed views`, metric: `${b.cams} × ${fmt(b.dur)} s`, tip: "Two fixed camera views of the same space, read as video files." },
    { id: "decode", name: "Decode", sub: `ffmpeg · ${b.fps} fps`, metric: b.size ? `${b.size[0]}×${b.size[1]}` : "h264", tip: `ffmpeg decodes the h264 video and samples ${b.fps} frames per second.` },
    { id: "detect", name: "Detect", sub: "MOG2 blobs", metric: `${b.samples} frames`, tip: "MOG2 background subtraction finds moving blobs. It looks for motion, not faces." },
    { id: "track", name: "Track", sub: "fragments", metric: `${b.fragments} fragments`, tip: "Blobs are linked frame to frame into fragments, then joined into person tracks." },
    { id: "follow", name: "Follow", sub: "cross-camera", metric: `${b.hops} hop${b.hops === 1 ? "" : "s"}`, tip: "A track that leaves one camera and enters the next is linked, with a score and a reason." },
    { id: "describe", name: "Describe", sub: "colour words", metric: `${b.tracks} looks`, tip: "Each track gets colour words for top and lower body. Nemotron VL captions are the slower, richer option." },
    { id: "index", name: "Index", sub: "local index · VAST VSS archive", metric: `${b.tracks} tracks`, tip: "Tracks, boxes and one crop per track go into the local fast index that the live desk searches. The archive path reads the team's VAST VSS caption rows and VastDB detection sidecars; it is not used by the live desk." },
    { id: "agent", name: "Agent", sub: "DeepSeek routes", metric: `${fmt(STM.facts.routeMs[0] / 1000)}–${fmt(STM.facts.routeMs[1] / 1000)} s`, tip: `${STM.facts.models.router} reads the question and picks one tool. It never sees video. Spoken questions are transcribed by NVIDIA Canary on the team's VAST cluster GPU.` },
    { id: "desk", name: "Desk", sub: "operator", metric: "live search", tip: "The operator gets the answer: which track, which camera, what time, and the clip." },
  ];

  STM.view("pipeline", {
    label: "Pipeline",
    async mount(root, ctx) {
      const el = STM.el, esc = STM.esc;
      let dead = false, raf = 0, ro = null, run = null, sel = "lab", gW = 0, firstDraw = true, last = performance.now();
      const timers = [];
      root.append(el("style", {}, CSS));

      const chipLab = el("button", { type: "button", class: "chip on", id: "pipeline-scene-lab", onclick: () => setSel("lab") }, "Lab · 2 × 60 s");
      const chipCav = el("button", { type: "button", class: "chip", id: "pipeline-scene-caviar", onclick: () => setSel("caviar") }, "Caviar · 2 × 15.3 s");
      STM.pageHead(root, "Pipeline", "From camera clip to searchable index, timed on today's builds.", el("div", { class: "row wrap" }, chipLab, chipCav));
      const wrap = el("div", { class: "v-pipeline" });
      root.append(wrap);

      /* skeleton first, while the second scene loads */
      const sk = el("div", { class: "pl-stats" }, [0, 1, 2, 3].map(() => el("div", { class: "card pl-tile" }, STM.skeleton(10, "45%"), STM.skeleton(26, "60%"), STM.skeleton(10, "80%"))));
      wrap.append(sk, el("div", { class: "card card-b" }, STM.skeleton(18, "30%"), el("div", { style: { height: "14px" } }), STM.skeleton(120, "100%", 12)));
      const [lab, cav] = await Promise.all([STM.loadScene("lab").catch(() => ctx.scene), STM.loadScene("caviar").catch(() => null), STM.sleep(420)]);
      if (dead || !root.contains(wrap)) return () => {};
      const B = { lab: buildOf(lab || ctx.scene, "lab"), caviar: buildOf(cav, "caviar") };
      if (ctx.scene && ctx.scene.name === "caviar") sel = "caviar";
      wrap.innerHTML = "";

      /* ---------- stat tiles ---------- */
      const stats = el("div", { class: "pl-stats stagger", id: "pipeline-stats" });
      const tiles = [0, 1, 2, 3].map(() => {
        const t = { lab: el("div", { class: "eyebrow" }), num: el("span"), pre: el("span"), sub: el("div", { class: "pl-sub" }) };
        stats.append(el("div", { class: "card pl-tile stat" }, t.lab, el("b", {}, t.pre, t.num), t.sub));
        return t;
      });
      function renderStats() {
        const b = B[sel], foot = b.cams * b.dur, F = STM.facts;
        const rows = [
          ["Footage indexed", "", foot, foot % 1 ? 1 : 0, " s", `${b.cams} cameras × ${fmt(b.dur)} s`],
          ["Index time", "", b.seconds, 1, " s", `${b.fragments} fragments → ${b.tracks} tracks at ${b.fps} fps`],
          ["Speed vs real time", "", foot / b.seconds, 1, "×", `${fmt(foot)} s of footage per ${fmt(b.seconds)} s of work`],
          ["Agent round trip", `${fmt(F.routeMs[0] / 1000)}–`, F.routeMs[1] / 1000, 1, " s", `${F.models.router} picks one tool`],
        ];
        rows.forEach(([l, pre, v, dec, suf, sub], i) => {
          const t = tiles[i]; t.lab.textContent = l; t.pre.textContent = pre; t.sub.textContent = sub;
          STM.countUp(t.num, v, { dec, suffix: suf, ms: 1000 });
        });
      }

      /* ---------- pipeline graph ---------- */
      const runBtn = el("button", { type: "button", class: "btn btn-primary btn-sm", id: "pipeline-run", onclick: startRun, html: STM.icons.play + "Run on a new clip" });
      const status = el("span", { class: "pill ok", id: "pipeline-status" }, "Index ready");
      const gHost = el("div", { class: "pl-ghost", id: "pipeline-graph" });
      const detail = el("div", { class: "pl-detail" });
      const clockEl = el("span", { class: "mono" }, "0.00 s"), clockBar = el("i");
      const graphCard = el("div", { class: "card" },
        el("div", { class: "card-h" }, el("h3", {}, "From camera to desk"), status, el("div", { class: "grow" }), runBtn),
        gHost,
        el("div", { class: "pl-gfoot" }, detail, el("div", { class: "pl-clock" }, el("span", {}, "Build clock"), el("div", { class: "prog" }, clockBar), clockEl)));
      const st = stagesOf(B[sel]).map(() => ({ s: "done", p: 1 }));
      let nodes = [], wires = [], stages = stagesOf(B[sel]);
      const setDetail = i => { const d = stages[i]; detail.innerHTML = `<b>${esc(d.name)}</b>${esc(d.tip)}`; };

      function fit(t, max) {                                   // truncate with an ellipsis instead of squashing glyphs
        const full = t.__full != null && t.textContent === t.__shown ? t.__full : t.textContent;
        t.__full = full; t.textContent = full;
        try {
          if (t.getComputedTextLength() > max) {
            let n = full.length;
            while (n > 1 && (t.textContent = full.slice(0, n - 1).trimEnd() + "\u2026", t.getComputedTextLength() > max)) n--;
          }
        } catch (e) {}
        t.__shown = t.textContent;
      }
      function drawGraph() {
        const W = Math.floor(gHost.clientWidth - 32);
        if (W < 200) return;
        gW = W;
        const cols = W >= 1000 ? 9 : W >= 520 ? 5 : 3, gap = cols === 9 ? 14 : 22, nw = (W - gap * (cols - 1)) / cols, nh = 112, rg = 46, pad = 8;
        const rows = Math.ceil(9 / cols), H = pad * 2 + rows * nh + (rows - 1) * rg;
        const P = i => { const r = Math.floor(i / cols); let c = i % cols; if (r % 2) c = cols - 1 - c; return { x: c * (nw + gap), y: pad + r * (nh + rg) }; };
        const svg = S("svg", { class: "pl-graph", viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Pipeline from cameras to desk" });
        const wireG = S("g"), dotG = S("g"), nodeG = S("g");
        svg.append(wireG, dotG, nodeG);
        gHost.innerHTML = ""; gHost.append(svg);
        wires = [];
        for (let i = 0; i < 8; i++) {
          const a = P(i), b = P(i + 1); let d, ar;
          if (a.y === b.y) {
            const cy = a.y + nh / 2, fw = b.x > a.x, x1 = fw ? a.x + nw : a.x, x2 = fw ? b.x : b.x + nw, k = fw ? -5 : 5;
            d = `M${x1} ${cy}L${x2} ${cy}`; ar = `M${x2 + k} ${cy - 4}L${x2} ${cy}L${x2 + k} ${cy + 4}`;
          } else {
            const cx = a.x + nw / 2; d = `M${cx} ${a.y + nh}L${cx} ${b.y}`; ar = `M${cx - 4} ${b.y - 5}L${cx} ${b.y}L${cx + 4} ${b.y - 5}`;
          }
          const path = S("path", { d, class: "pl-wire" });
          wireG.append(path, S("path", { d: ar, class: "pl-arrow" }));
          const len = path.getTotalLength();
          const dots = [0, 1, 2].map(k => { const c = S("circle", { r: 2.6, class: "pl-dot", opacity: 0 }); dotG.append(c); return { c, f: k / 3 }; });
          wires.push({ path, len, dots, to: i + 1 });
        }
        nodes = stages.map((sd, i) => {
          const p = P(i);
          const g = S("g", { transform: `translate(${p.x} ${p.y})`, class: "pl-node", tabindex: 0, "data-stage": sd.id, role: "button", "aria-label": `${sd.name}: ${sd.tip}` });
          const inn = S("g", { class: "pl-in" + (firstDraw ? " first" : ""), style: `animation-delay:${i * 60}ms` });
          const ico = S("g", { class: "ico-g", transform: "translate(18 18) scale(.75)" }); ico.innerHTML = ICON[sd.id];
          const ring = S("g", { transform: `translate(${nw - 20} 27)` });
          ring.append(S("circle", { class: "r-wait", r: 3.5 }), S("circle", { class: "r-track", r: 8 }), S("circle", { class: "r-spin", r: 8, "stroke-dasharray": "15 60" }),
            S("g", { class: "r-done" }, S("circle", { r: 8.5 }), S("path", { d: "M-3.8 0.2l2.5 2.5L3.9-2.8" })));
          const nm = S("text", { class: "nm", x: 12, y: 62 }, sd.name), sb = S("text", { class: "sb", x: 12, y: 78 }, sd.sub), mt = S("text", { class: "mt", x: 12, y: 95 }, sd.metric);
          const pfill = S("rect", { class: "pfill", x: 12, y: nh - 11, width: 0, height: 3, rx: 1.5 });
          inn.append(S("rect", { class: "halo", width: nw, height: nh, rx: 12 }), S("rect", { class: "bg", width: nw, height: nh, rx: 12 }),
            S("rect", { class: "badge", x: 12, y: 12, width: 30, height: 30, rx: 8 }), ico, ring, nm, sb, mt,
            S("rect", { class: "pbg", x: 12, y: nh - 11, width: nw - 24, height: 3, rx: 1.5 }), pfill);
          g.append(inn); nodeG.append(g);
          const pick = () => { nodes.forEach(n => n.g.classList.toggle("sel", n.g === g)); setDetail(i); flashRow(sd.id); };
          g.addEventListener("mouseenter", () => setDetail(i));
          g.addEventListener("click", pick);
          g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
          fit(nm, nw - 24); fit(sb, nw - 24); fit(mt, nw - 24);
          return { g, mt, pfill, w: nw - 24, txt: sd.metric, s: null, p: -1 };
        });
        firstDraw = false;
        nodes.forEach((_, i) => paint(i, true));
      }
      function paint(i, force) {
        const n = nodes[i], x = st[i]; if (!n) return;
        if (force || n.s !== x.s) {
          n.s = x.s; n.g.classList.toggle("is-wait", x.s === "wait"); n.g.classList.toggle("is-run", x.s === "run"); n.g.classList.toggle("is-done", x.s === "done");
          const txt = !run || x.s === "done" ? stages[i].metric : x.s === "run" ? "running" : "waiting";
          if (txt !== n.txt || force) { n.mt.textContent = txt; n.txt = txt; fit(n.mt, n.w); }
        }
        if (force || Math.abs(n.p - x.p) > .004) { n.p = x.p; n.pfill.setAttribute("width", Math.max(0, n.w * x.p)); }
      }

      /* ---------- build log + build timeline ---------- */
      const logEl = el("div", { class: "code pl-log", id: "pipeline-log", role: "log" });
      const idleLog = () => { logEl.innerHTML = `<div><span class="c">Waiting. Press Run on a new clip to replay today's caviar build.</span></div><div><span class="pl-caret"></span></div>`; };
      idleLog();
      const logCard = el("div", { class: "card" },
        el("div", { class: "card-h" }, el("h3", {}, "Build log"), el("div", { class: "grow" }), el("span", { class: "pl-cap mono" }, "caviar · 2 cams")),
        logEl);
      const gLab = el("div", { class: "pl-glab" }), gTr = el("div", { class: "pl-gtr" }), gTitle = el("span", { class: "pl-cap" });
      const head = el("div", { class: "pl-head" });
      let gMax = 4;
      const ganttCard = el("div", { class: "card", id: "pipeline-timeline" },
        el("div", { class: "card-h" }, el("h3", {}, "Build timeline"), el("div", { class: "grow" }), gTitle),
        el("div", { class: "card-b" }, el("div", { class: "pl-gantt" }, gLab, gTr)),
        el("div", { class: "pl-note" }, "Cameras are indexed in parallel."));
      function renderGantt() {
        const b = B[sel], step = b.seconds <= 5 ? 1 : b.seconds <= 10 ? 2 : b.seconds <= 50 ? 10 : 30;
        gMax = Math.ceil(b.seconds * 1.04 / step) * step;
        gTitle.textContent = b.title;
        const lanes = b.per.map(p => ({ l: `Camera ${p.camera}`, s: "decode · detect · track", a: 0, z: p.seconds, v: `${p.seconds.toFixed(2)} s`, c: "" }));
        lanes.push({ l: "Follow", s: "cross-camera links", a: b.index_s, z: b.index_s + b.follow_s, v: `${b.follow_s.toFixed(2)} s`, c: "f" });
        lanes.push({ l: "Whole build", s: "start to index ready", a: 0, z: b.seconds, v: `${b.seconds.toFixed(2)} s`, c: "t" });
        gLab.innerHTML = ""; gTr.innerHTML = "";
        const bars = [];
        lanes.forEach(L => {
          gLab.append(el("div", {}, el("span", {}, el("span", {}, L.l), el("em", {}, L.v)), el("small", {}, L.s)));
          const i = el("i", { class: L.c, style: { left: (L.a / gMax * 100) + "%" } });
          gTr.append(el("div", { class: "pl-lane" }, i)); bars.push([i, (L.z - L.a) / gMax * 100]);
        });
        const ax = el("div", { class: "pl-axis" });
        for (let v = 0; v <= gMax + 1e-9; v += step) ax.append(el("span", { style: { left: (v / gMax * 100) + "%" } }, v + (v + step > gMax + 1e-9 ? " s" : "")));
        gTr.append(ax, head);
        requestAnimationFrame(() => { thinLabels(ax); requestAnimationFrame(() => bars.forEach(([i, w]) => { i.style.width = w + "%"; })); });
      }

      /* ---------- tracks in the index ---------- */
      const cropsEl = el("div", { class: "pl-crops stagger" }), cropCount = el("span", { class: "muted", style: { fontSize: "12.5px" } });
      const sw = c => el("span", { class: "swatch", style: { background: STM.colourHex[c] || "var(--sunk)" } });
      const crop = (s, t, pop) => el("button", { type: "button", class: "pl-crop" + (pop ? " pop" : ""), title: `${t.id} · ${t.desc}`, onclick: () => STM.go("catalogue") },
        el("img", { src: s.cropUrl(t), alt: `Track ${t.id}, ${t.desc}`, loading: "lazy" }),
        el("div", {}, sw(t.look && t.look.top), sw(t.look && t.look.bottom), el("span", {}, t.id)));
      const tracksCard = el("div", { class: "card", id: "pipeline-tracks" },
        el("div", { class: "card-h" }, el("h3", {}, "Tracks in the index"), cropCount, el("div", { class: "grow" }),
          el("button", { type: "button", class: "btn btn-sm", onclick: () => STM.go("catalogue") }, "Open catalogue")),
        el("div", { class: "card-b" }, cropsEl));
      function renderCrops() {
        const s = B[sel].scene; cropsEl.innerHTML = "";
        if (!s) { cropsEl.append(el("div", { class: "empty" }, "This scene is not loaded.")); return; }
        s.tracks.forEach(t => cropsEl.append(crop(s, t)));
        cropCount.textContent = `${s.tracks.length} person tracks · ${s.cameras.length} cameras`;
      }

      /* ---------- live path vs archive path ---------- */
      const F = STM.facts, lg = v => Math.max(0, Math.min(1, Math.log10(v / 0.1) / Math.log10(36000)));
      const cmpRows = [
        ["Live path"],
        ["Agent routing", "per question", F.routeMs[0] / 1000, F.routeMs[1] / 1000, `${fmt(F.routeMs[0] / 1000)}–${fmt(F.routeMs[1] / 1000)} s`],
        ["Canary on VAST GPU", "per spoken question", F.sttMs[0] / 1000, F.sttMs[1] / 1000, `${fmt(F.sttMs[0] / 1000)}–${fmt(F.sttMs[1] / 1000)} s`],
        ["Fast index, caviar", `${B.caviar.cams} × ${fmt(B.caviar.dur)} s footage`, B.caviar.seconds, null, `${fmt(B.caviar.seconds)} s`],
        ["Fast index, lab", `${B.lab.cams} × ${fmt(B.lab.dur)} s footage`, B.lab.seconds, null, `${fmt(B.lab.seconds)} s`],
        ["Archive path"],
        ["VSS archive index", "per clip", F.vssIndexMin[0] * 60, F.vssIndexMin[1] * 60, `${F.vssIndexMin[0]}–${F.vssIndexMin[1]} min`, true],
      ];
      const cmpBars = [];
      const cmp = el("div", { class: "pl-cmp" }, cmpRows.map(r => {
        if (r.length === 1) return el("div", { class: "pl-cgrp" }, r[0]);
        const lo = el("i", { class: "lo" }), hi = el("i", { class: "hi" });
        cmpBars.push([lo, lg(r[2])], [hi, r[3] ? lg(r[3]) : 0]);
        return el("div", { class: "pl-crow" }, el("div", {}, r[0], el("small", {}, r[1])), el("div", { class: "pl-cbar" + (r[5] ? " w" : "") }, hi, lo), el("span", { class: "mono" }, r[4]));
      }));
      const ax = el("div", { class: "pl-cax" });
      [[1, "1 s"], [10, "10 s"], [60, "1 min"], [3600, "1 h"]].forEach(([v, t]) => ax.append(el("span", { style: { left: (lg(v) * 100) + "%" } }, t)));
      cmp.append(el("div", { class: "pl-crow" }, el("div"), ax, el("div")));
      const cmpCard = el("div", { class: "card", id: "pipeline-compare" },
        el("div", { class: "card-h" }, el("h3", {}, "Live path vs archive path")),
        el("div", { class: "card-b" }, cmp),
        el("div", { class: "pl-note" }, "Log scale. Light bar ends mark the top of each measured range."));

      /* ---------- what each stage keeps ---------- */
      const keepRows = [
        ["decode", "Decode", `Sampled frames at ${B.lab.fps} fps`, "ok", "Yes"],
        ["detect", "Detect", "Moving-blob boxes (MOG2)", "ok", "Yes"],
        ["track", "Track", "Fragments with boxes [t, x, y, w, h]", "ok", "Yes"],
        ["follow", "Follow", "Cross-camera links, each with a score and a reason", "ok", "Yes"],
        ["describe", "Describe", "Colour words for top and lower body", "ok", "Yes"],
        ["index", "Index", "Track table and one crop per track", "ok", "Yes"],
        ["agent", "Agent", "One tool name and its arguments", "info", "Question text only"],
        ["desk", "Desk", "Answer: track, camera, time, clip", "ok", "Yes"],
      ];
      const keepsCard = el("div", { class: "card", id: "pipeline-keeps" },
        el("div", { class: "card-h" }, el("h3", {}, "What each stage keeps")),
        el("div", { class: "scroll-x" }, el("table", { class: "table pl-keeps" },
          el("thead", {}, el("tr", {}, el("th", {}, "Stage"), el("th", {}, "Output"), el("th", {}, "Kept on device"), el("th", {}, "Faces used"))),
          el("tbody", {}, keepRows.map(([id, n, o, k, v]) => el("tr", { "data-stage": id },
            el("td", {}, el("b", { style: { fontWeight: 600 } }, n)), el("td", { class: "dim" }, o),
            el("td", {}, v), el("td", { class: "k-no" }, "Never")))))));
      function flashRow(id) {
        const r = keepsCard.querySelector(`tr[data-stage="${id}"]`); if (!r) return;
        r.classList.remove("pl-hl"); void r.offsetWidth; r.classList.add("pl-hl");
      }

      /* ---------- what the index powers: timeline, catalogue, similar ---------- */
      const miniTl = el("div", { class: "pl-mini" }), thumbs = el("div", { class: "pl-thumbs" }), simBox = el("div", { class: "pl-thumbs" });
      const tlNote = el("span", { class: "pl-pn" }), catNote = el("span", { class: "pl-pn" }), simNote = el("span", { class: "pl-pn" }), simWhy = el("small");
      const pow = (id, view, title, text, mini, note) => el("div", { class: "card pl-pow" },
        el("div", { class: "row" }, el("h3", { style: { fontSize: "14px", fontWeight: 600 } }, title), el("div", { class: "grow" }), note),
        el("p", {}, text), mini,
        el("button", { type: "button", class: "btn btn-sm", id, onclick: () => STM.go(view), html: `Open ${title.toLowerCase()} ${STM.icons.arrow}` }));
      const powers = el("div", { class: "pl-three stagger" },
        pow("pipeline-open-timeline", "timeline", "Timeline", "Every track on one time axis, per camera. Scrub to any second.", miniTl, tlNote),
        pow("pipeline-open-catalogue", "catalogue", "Catalogue", "One card per person track, with its crop and colour words.", thumbs, catNote),
        pow("pipeline-open-similar", "similar", "Similar", "Tracks with the same colour words, ranked.", el("div", { style: { display: "grid", gap: "8px", minWidth: 0 } }, simBox, simWhy), simNote));
      function renderPowers() {
        const s = B[sel].scene; miniTl.innerHTML = ""; thumbs.innerHTML = ""; simBox.innerHTML = ""; simWhy.textContent = ""; simNote.textContent = "";
        if (!s) return;
        const cams = s.cameras.map(c => c.id), bars = [];
        s.tracks.forEach((t, k) => {
          const lane = cams.indexOf(t.cam), y = 8 + lane * 26 + (k % 4) * 5;
          const i = el("i", { style: { left: (t.t0 / s.duration * 100) + "%", top: y + "px", background: STM.colourHex[t.look && t.look.top] || "var(--accent)" }, title: `${t.id} ${s.clock(t.t0)}–${s.clock(t.t1)}` });
          miniTl.append(i); bars.push([i, Math.max(1, (t.t1 - t.t0) / s.duration * 100)]);
        });
        requestAnimationFrame(() => requestAnimationFrame(() => bars.forEach(([i, w]) => { i.style.width = w + "%"; })));
        tlNote.textContent = `${fmt(s.duration)} s · ${cams.length} cams`;
        s.tracks.slice(0, 8).forEach(t => thumbs.append(el("img", { class: "crop", src: s.cropUrl(t), alt: t.desc, title: `${t.id} · ${t.desc}` })));
        catNote.textContent = `${s.tracks.length} cards`;
        const groups = {};
        s.tracks.forEach(t => { (groups[t.desc] = groups[t.desc] || []).push(t); });
        const best = Object.entries(groups).sort((a, b) => b[1].length - a[1].length)[0];
        if (best) {
          best[1].slice(0, 6).forEach(t => simBox.append(el("img", { class: "crop", src: s.cropUrl(t), alt: t.desc, title: t.id })));
          simNote.textContent = `${best[1].length} alike`; simWhy.textContent = `Most common look: ${best[0]}`;
        }
      }

      wrap.append(stats, graphCard, el("div", { class: "pl-two" }, logCard, ganttCard), tracksCard, el("div", { class: "pl-two rev" }, cmpCard, keepsCard), powers);

      function setSel(k) {
        if (run || !B[k] || !B[k].scene) return;
        sel = k; chipLab.classList.toggle("on", k === "lab"); chipCav.classList.toggle("on", k === "caviar");
        stages = stagesOf(B[sel]);
        renderStats(); renderGantt(); renderCrops(); renderPowers();
        nodes.forEach((_, i) => paint(i, true));
        clockEl.textContent = B[sel].seconds.toFixed(2) + " s"; clockBar.style.width = "100%";
        setDetail(0);
      }

      /* ---------- staged run: replays the caviar build at about real speed ---------- */
      function startRun() {
        if (run) return;
        const b = B.caviar, s = b.scene;
        if (!s || !b.per.length) { STM.toast("The caviar clip is not loaded."); return; }
        setSel("caviar");
        const secs = b.per.map(p => p.seconds), tmax = Math.max(...secs), tmin = Math.min(...secs), ix = b.index_s + b.follow_s, mid = (ix + b.seconds) / 2;
        const C = [[0, 0], [0, tmin * .38], [tmin * .38, tmin * .77], [tmin * .77, tmax], [tmax, ix], [ix, mid], [mid, b.seconds], [b.seconds, b.seconds], [b.seconds, b.seconds]];
        const MS = [[0, 350], [350, 1500], [1500, 2650], [2650, 3700], [3700, 4150], [4150, 4550], [4550, 5000], [5000, 5350], [5350, 5700]];
        const msAt = c => { for (let k = 0; k < 9; k++) { const [a, z] = C[k]; if (z > a && c >= a && c <= z) return MS[k][0] + (c - a) / (z - a) * (MS[k][1] - MS[k][0]); } return MS[8][0]; };
        const cnt = c => s.tracks.filter(t => t.cam === c).length;
        const idn = (s.identities || []).find(i => i.hops && i.hops.length), hop = idn && idn.hops[0];
        const L = [], add = (at, stamp, html) => L.push({ at, stamp, html });
        add(60, 0, `job <span class="k">${esc(s.name)}</span> · ${s.cameras.length} cameras · ${fmt(b.dur)} s each`);
        b.per.forEach((p, k) => add(380 + k * 160, 0, `decode cam${esc(p.camera)}.mp4 ${p.size[0]}x${p.size[1]} ${esc(b.videos[p.camera] || "h264")} @ ${b.fps} fps`));
        add(1540, 0, `detect: MOG2 background subtraction, moving blobs only`);
        add(1760, null, `<span class="c">no face or identity features are computed</span>`);
        add(2690, 0, `track: link blobs frame to frame into fragments`);
        b.per.forEach(p => add(msAt(p.seconds), p.seconds, `cam${esc(p.camera)}: <span class="n">${p.samples}</span> samples · ${p.skipped} skipped · ${cnt(p.camera)} person tracks`));
        add(msAt(tmax) + 30, b.index_s, `track: <span class="n">${b.fragments}</span> fragments → <span class="n">${b.tracks}</span> person tracks`);
        add(MS[4][0] + 180, ix, hop ? `follow: ${b.hops} cross-camera hop ${esc(hop.from)} → ${esc(hop.to)} · score ${hop.score}` : `follow: ${b.hops} cross-camera hops`);
        if (hop) add(MS[4][0] + 320, null, `<span class="c">why: ${esc(hop.why)}</span>`);
        add(MS[5][0] + 60, ix, `describe: ${esc(b.describe)} for ${b.tracks} tracks`);
        if (s.tracks[0]) add(MS[5][0] + 220, null, `<span class="c">${esc(s.tracks[0].id)}: ${esc(s.tracks[0].desc)}</span>`);
        add(MS[6][0] + 320, b.seconds, `<span class="s">index ready: ${b.tracks} tracks · ${b.people} people · ${b.seconds.toFixed(2)} s</span>`);
        add(MS[7][0] + 80, null, `agent: ${esc(F.models.router)} routes each question to one tool`);
        add(MS[8][0] + 80, null, `desk: search is live on ${s.cameras.length} cameras`);
        L.sort((x, y) => x.at - y.at);
        const order = s.tracks.slice().sort((x, y) => x.t0 - y.t0), cropAt = order.map((_, k) => MS[3][0] + 200 + k * (MS[6][1] - MS[3][0] - 300) / order.length);
        run = { t0: performance.now(), b, s, C, MS, L, li: 0, ci: 0, order, cropAt, total: MS[8][1] };
        st.forEach(x => { x.s = "wait"; x.p = 0; }); nodes.forEach((_, i) => paint(i, true));
        logEl.innerHTML = ""; cropsEl.innerHTML = ""; cropCount.textContent = "0 person tracks so far";
        runBtn.innerHTML = '<span class="pl-bspin"></span>Indexing clip'; runBtn.setAttribute("aria-busy", "true");
        status.className = "pill info"; status.textContent = "Indexing caviar";
        head.classList.add("on"); chipLab.disabled = chipCav.disabled = true;
      }
      function tickRun(now) {
        const R = run, t = now - R.t0;
        R.MS.forEach(([a, z], k) => {
          const s = t < a ? "wait" : t < z ? "run" : "done", p = t < a ? 0 : t < z ? (t - a) / (z - a) : 1;
          st[k].s = s; st[k].p = p; paint(k);
        });
        let c = R.b.seconds;
        for (let k = 0; k < 9; k++) if (t < R.MS[k][1]) { const f = Math.max(0, (t - R.MS[k][0]) / (R.MS[k][1] - R.MS[k][0])); c = R.C[k][0] + (R.C[k][1] - R.C[k][0]) * f; break; }
        clockEl.textContent = c.toFixed(2) + " s"; clockBar.style.width = (c / R.b.seconds * 100) + "%";
        head.style.left = (c / gMax * 100) + "%";
        while (R.li < R.L.length && R.L[R.li].at <= t) {
          const ln = R.L[R.li++], stamp = ln.stamp == null ? "       " : `[${ln.stamp.toFixed(1).padStart(4, "0")}s]`;
          logEl.append(el("div", { html: `<span class="n">${stamp}</span> ${ln.html}` })); logEl.scrollTop = logEl.scrollHeight;
        }
        while (R.ci < R.order.length && R.cropAt[R.ci] <= t) {
          cropsEl.append(crop(R.s, R.order[R.ci++], true)); cropCount.textContent = `${R.ci} person tracks so far`;
        }
        if (t >= R.total + 150) finishRun();
      }
      function finishRun() {
        const R = run; run = null;
        while (R.ci < R.order.length) cropsEl.append(crop(R.s, R.order[R.ci++], true));
        cropCount.textContent = `${R.s.tracks.length} person tracks · ${R.s.cameras.length} cameras`;
        st.forEach(x => { x.s = "done"; x.p = 1; }); nodes.forEach((_, i) => paint(i, true));
        logEl.append(el("div", { html: '<span class="pl-caret"></span>' })); logEl.scrollTop = logEl.scrollHeight;
        runBtn.innerHTML = STM.icons.play + "Run again"; runBtn.removeAttribute("aria-busy");
        status.className = "pill ok"; status.textContent = "Index ready";
        head.classList.remove("on"); chipLab.disabled = chipCav.disabled = false;
        clockEl.textContent = R.b.seconds.toFixed(2) + " s";
        STM.toast(`${STM.icons.check}&nbsp;Index ready in ${R.b.seconds.toFixed(1)} s`);
        STM.emit("pipeline:done", { tracks: R.b.tracks, seconds: R.b.seconds });
      }

      /* ---------- frame loop: particles along the wires, plus the staged run ---------- */
      function frame(now) {
        if (dead) return;
        raf = requestAnimationFrame(frame);
        const dt = Math.min(.05, (now - last) / 1000); last = now;
        if (run) tickRun(now);
        for (const w of wires) {
          const target = st[w.to], live = !run || target.s !== "wait", hot = run && target.s === "run", v = hot ? 170 : 55;
          for (const d of w.dots) {
            d.f = (d.f + dt * v / Math.max(20, w.len)) % 1;
            if (!live) { d.c.setAttribute("opacity", 0); continue; }
            const pt = w.path.getPointAtLength(d.f * w.len);
            d.c.setAttribute("cx", pt.x); d.c.setAttribute("cy", pt.y);
            d.c.setAttribute("opacity", (Math.sin(Math.PI * d.f) * (hot ? 1 : .75)).toFixed(3));
          }
        }
      }

      setSel(sel);
      drawGraph();
      setDetail(0);
      requestAnimationFrame(() => requestAnimationFrame(() => cmpBars.forEach(([i, w], k) => { i.style.transitionDelay = (k * 40) + "ms"; i.style.width = (w * 100) + "%"; })));
      requestAnimationFrame(() => thinLabels(ax));
      if ("ResizeObserver" in window) {
        ro = new ResizeObserver(() => { if (Math.abs(Math.floor(gHost.clientWidth - 32) - gW) > 2) drawGraph(); wrap.querySelectorAll(".pl-axis,.pl-cax").forEach(thinLabels); });
        ro.observe(gHost); ro.observe(ganttCard); ro.observe(cmpCard);
      }
      raf = requestAnimationFrame(frame);
      if (ctx.opts && ctx.opts.autorun) timers.push(setTimeout(() => { if (!dead) startRun(); }, 900));

      return () => {
        dead = true; run = null;
        cancelAnimationFrame(raf);
        timers.forEach(clearTimeout);
        if (ro) ro.disconnect();
      };
    },
  });
})();
