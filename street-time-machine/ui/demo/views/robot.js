/* Robot view: a simulated mobile robot ("Rover 1") the desk can find on the cameras and send to the last sighting.
   Camera layout, the sighting and track trails come from the real scene index. The robot, its motion, battery,
   camera detection and lidar scans are simulated in the browser and labelled as such. */
(function () {
  "use strict";
  const PXM = 30, VMAX = 1.2, FAST = 6, PI = Math.PI, C = 2 * PI * 26;
  const ROOM = [30, 30, 900, 480], AISLE = 300, DOCK = { x: 140, y: 404 };
  const BENCH = [[50, 46, 110, 26], [560, 118, 110, 34], [740, 118, 130, 34], [660, 372, 200, 30]];
  const CAFE = [396, 410, 214, 90], TABLES = [[450, 446], [510, 474], [560, 438]];
  const OBST = BENCH.concat(TABLES.map(([x, y]) => [x - 14, y - 14, 28, 28]));
  const CMDS = ["Go to camera B", "Scan for a grey backpack", "Hold position and guard", "Return to dock"];
  const PILL = { Docked: "ok", Moving: "info", Scanning: "warn", Holding: "plain" };
  const f1 = v => Math.round(v * 10) / 10;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const ease = f => (1 - Math.cos(PI * f)) / 2;
  const angN = a => Math.atan2(Math.sin(a), Math.cos(a));
  const pt = (p, a, r) => ({ x: p.x + r * Math.cos(a), y: p.y + r * Math.sin(a) });
  const inRect = (p, r) => p.x >= r[0] && p.x <= r[0] + r[2] && p.y >= r[1] && p.y <= r[1] + r[3];
  const sector = (p, a0, a1, R) => {
    const q0 = pt(p, a0, R), q1 = pt(p, a1, R);
    return `M${f1(p.x)} ${f1(p.y)}L${f1(q0.x)} ${f1(q0.y)}A${R} ${R} 0 ${a1 - a0 > PI ? 1 : 0} 1 ${f1(q1.x)} ${f1(q1.y)}Z`;
  };
  function ray(p, a, max) {                       // distance to the first wall or obstacle along a ray, or null
    const dx = Math.cos(a) || 1e-9, dy = Math.sin(a) || 1e-9, [rx, ry, rw, rh] = ROOM;
    let t = Math.min(max, dx > 0 ? (rx + rw - p.x) / dx : (rx - p.x) / dx, dy > 0 ? (ry + rh - p.y) / dy : (ry - p.y) / dy);
    for (const [x, y, w, h] of OBST) {
      const tx1 = (x - p.x) / dx, tx2 = (x + w - p.x) / dx, ty1 = (y - p.y) / dy, ty2 = (y + h - p.y) / dy;
      const tn = Math.max(Math.min(tx1, tx2), Math.min(ty1, ty2)), tf = Math.min(Math.max(tx1, tx2), Math.max(ty1, ty2));
      if (tf >= tn && tn > 0) t = Math.min(t, tn);
    }
    return t < max ? t : null;
  }
  function pickTrack(s) {                         // the desk's lead if it is in this scene, else the longest camera-B track
    for (const v of [STM.state.lead, STM.state.subject]) {
      if (!v) continue;
      const id = typeof v === "string" ? v : (v.track && v.track.id) || v.track || v.trackId || v.id;
      const t = typeof id === "string" && s.track(id);
      if (t && t.boxes && t.boxes.length) return { t, from: "desk" };
    }
    const cam = (s.cameras[1] || s.cameras[0]).id;
    const pool = s.tracks.filter(t => t.cam === cam && t.boxes.length).sort((a, b) => (b.t1 - b.t0) - (a.t1 - a.t0) || b.t1 - a.t1);
    return { t: pool[0] || s.tracks.find(t => t.boxes.length), from: "index" };
  }

  const CSS = `
.v-robot{container-type:inline-size}
.v-robot .grow{flex:1}
.v-robot .vr-banner{margin-bottom:16px}
.v-robot .vr-grid{display:grid;grid-template-columns:minmax(0,1.75fr) minmax(290px,1fr);gap:16px;margin-bottom:16px;align-items:start}
@container (max-width:820px){.v-robot .vr-grid{grid-template-columns:minmax(0,1fr)}}
@media (max-width:820px){.v-robot .vr-grid{grid-template-columns:minmax(0,1fr)}}
.v-robot .vr-mapwrap{border-radius:12px;overflow:hidden;border:1px solid var(--line);background:var(--sunk)}
.v-robot svg.vr-map{display:block;width:100%;height:auto}
.v-robot .m-floor{fill:var(--card)} .v-robot .m-gl{fill:none;stroke:var(--line);stroke-width:1}
.v-robot .m-aisle{fill:var(--sunk)} .v-robot .m-aisle-c{fill:none;stroke:var(--line-2);stroke-width:2;stroke-dasharray:14 12}
.v-robot .m-wall{fill:none;stroke:var(--ink-2);stroke-width:5;stroke-linecap:round;stroke-linejoin:round}
.v-robot .m-bench{fill:var(--sunk);stroke:var(--line-2);stroke-width:1.5}
.v-robot .m-cafe{fill:color-mix(in srgb,var(--warn) 9%,transparent);stroke:color-mix(in srgb,var(--warn) 40%,transparent);stroke-dasharray:5 5}
.v-robot .m-table{fill:var(--card);stroke:var(--line-2);stroke-width:1.5}
.v-robot .m-dock{fill:color-mix(in srgb,var(--ok) 13%,transparent);stroke:var(--ok);stroke-width:1.5}
.v-robot .m-pin{stroke:var(--ok);stroke-width:3;stroke-linecap:round}
.v-robot .m-txt{font:500 15px var(--f-body);fill:var(--ink-3)} .v-robot .m-txt.mid{text-anchor:middle} .v-robot .m-txt.end{text-anchor:end}
.v-robot .m-cone{fill:color-mix(in srgb,var(--accent) 8%,transparent);stroke:var(--accent-line);stroke-width:1.5;transition:fill .4s}
.v-robot .m-cone.on{fill:color-mix(in srgb,var(--accent) 20%,transparent)}
.v-robot .m-beam{stroke:var(--accent);stroke-width:2.5;opacity:0;transition:opacity .2s} .v-robot .m-beam.on{opacity:.95}
.v-robot .m-beamw{fill:color-mix(in srgb,var(--accent) 14%,transparent)}
.v-robot .m-cambody{fill:var(--ink)} .v-robot .m-lens{fill:var(--accent);stroke:var(--card);stroke-width:2}
.v-robot .m-camlab rect{fill:var(--card);stroke:var(--line-2)} .v-robot .m-camlab text{font:600 13px var(--f-body);fill:var(--ink-2);text-anchor:middle}
.v-robot .m-trail{fill:none;stroke-width:2.5;stroke-dasharray:2 5;stroke-linecap:round;opacity:.75}
.v-robot .m-other{opacity:.45;stroke:var(--ink-3);stroke-width:1}
.v-robot .m-seen{stroke:var(--ink-3);stroke-width:1.2;stroke-dasharray:3 5}
.v-robot .m-planset{transition:opacity .6s} .v-robot .m-planset.gone{opacity:0}
.v-robot .m-plan-bg{fill:none;stroke:color-mix(in srgb,var(--accent) 16%,transparent);stroke-width:12;stroke-linecap:round;stroke-linejoin:round;transition:stroke-dashoffset .7s var(--ease)}
.v-robot .m-plan{fill:none;stroke:var(--accent);stroke-width:2.5;stroke-dasharray:8 8;animation:vrAnts .8s linear infinite;opacity:0;transition:opacity .4s .3s}
.v-robot .m-plan.on{opacity:.9}
.v-robot .m-done{fill:none;stroke:var(--accent);stroke-width:4;stroke-linecap:round;stroke-linejoin:round}
@keyframes vrAnts{to{stroke-dashoffset:-16}}
.v-robot .m-pulse{transform-box:fill-box;transform-origin:center;animation:vrPulse 2s var(--ease) infinite}
.v-robot .m-pulse.b{animation-delay:1s}
@keyframes vrPulse{from{transform:scale(1);opacity:.55}to{transform:scale(3.4);opacity:0}}
.v-robot .m-sdot{stroke:var(--card);stroke-width:2.5}
.v-robot .m-tag rect,.v-robot .m-call rect{fill:var(--ink)} .v-robot .m-tag text,.v-robot .m-call text{font:600 13px var(--f-mono);fill:var(--card);text-anchor:middle}
.v-robot .m-call{transition:opacity .35s}
.v-robot .m-radar{transition:opacity .3s} .v-robot .m-radar.off{opacity:0}
.v-robot .m-rbody{fill:var(--accent);stroke:var(--card);stroke-width:2.5} .v-robot .m-head{fill:var(--accent)} .v-robot .m-eye{fill:var(--card)}
.v-robot .m-rlab{font:600 14px var(--f-body);fill:var(--accent);text-anchor:middle}
.v-robot .m-ring{fill:none;stroke:var(--accent);stroke-width:2.5;opacity:0;transform-box:fill-box;transform-origin:center}
.v-robot .m-ring.on{animation:vrRing 1.3s var(--ease) 3}
@keyframes vrRing{from{opacity:.9;transform:scale(.7)}to{opacity:0;transform:scale(2.4)}}
.v-robot .m-guard{fill:color-mix(in srgb,var(--warn) 6%,transparent);stroke:var(--warn);stroke-width:2;stroke-dasharray:6 6;opacity:0;transition:opacity .3s;transform-box:fill-box;transform-origin:center}
.v-robot .m-guard.on{opacity:1;animation:vrSpin 8s linear infinite}
@keyframes vrSpin{to{transform:rotate(360deg)}}
.v-robot .m-lidar{transition:opacity 1s} .v-robot .m-lidar.fade{opacity:0}
.v-robot .m-lid{fill:color-mix(in srgb,var(--warn) 24%,transparent)}
.v-robot .m-arc{fill:color-mix(in srgb,var(--warn) 5%,transparent);stroke:color-mix(in srgb,var(--warn) 45%,transparent);stroke-dasharray:3 5}
.v-robot .m-hit{fill:var(--warn);transform-box:fill-box;transform-origin:center;animation:pop .3s var(--ease) both}
.v-robot .m-scale path{stroke:var(--ink-3);stroke-width:2;fill:none} .v-robot .m-scale text{font:500 14px var(--f-mono);fill:var(--ink-3);text-anchor:middle}
.v-robot .vr-sight{display:flex;align-items:center;gap:12px;margin-top:12px;flex-wrap:wrap}
.v-robot .vr-sight img{width:38px;height:50px;flex:none}
.v-robot .vr-who{display:grid;gap:1px;font-size:12.5px;min-width:0;flex:1 1 200px}
.v-robot .vr-who > span{min-width:0;overflow-wrap:anywhere}
.v-robot .card-h{min-width:0}
.v-robot .kv dd{min-width:0;overflow-wrap:anywhere}
.v-robot .vr-links{display:flex;gap:6px;margin-left:auto;flex-wrap:wrap}
.v-robot .vr-legend{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:12px;padding-top:10px;border-top:1px solid var(--line);font-size:11.5px;color:var(--ink-3)}
.v-robot .lg{display:inline-block;width:14px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.v-robot .lg.cone{background:color-mix(in srgb,var(--accent) 18%,transparent);border:1px solid var(--accent-line)}
.v-robot .lg.dot{width:10px;border-radius:50%} .v-robot .lg.dash{height:0;border-top:2px dashed var(--accent);border-radius:0}
.v-robot .lg.bot{background:var(--accent);border-radius:4px}
.v-robot .vr-ava{width:28px;height:28px;border-radius:9px;background:var(--accent);color:var(--card);display:grid;place-items:center;flex:none}
.v-robot .vr-ava svg{width:16px;height:16px}
.v-robot .vr-gauges{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-bottom:14px}
.v-robot .vr-g{border:1px solid var(--line);border-radius:12px;padding:10px 6px 8px;display:grid;justify-items:center;gap:4px;background:var(--sunk);font-size:11.5px;color:var(--ink-3)}
.v-robot .vr-ring{position:relative;width:64px;height:64px}
.v-robot .vr-ring svg{width:64px;height:64px;display:block}
.v-robot .vr-ring b{position:absolute;inset:0;display:grid;place-items:center;font:600 13px var(--f-mono);color:var(--ink)}
.v-robot .vr-ring .trk{fill:none;stroke:var(--line);stroke-width:6}
.v-robot .vr-ring .val{fill:none;stroke:var(--accent);stroke-width:6;stroke-linecap:round;transition:stroke-dashoffset .5s var(--ease),stroke .3s}
.v-robot .vr-ring .val.ok{stroke:var(--ok)} .v-robot .vr-ring .val.warn{stroke:var(--warn)}
.v-robot .vr-route{display:grid;gap:6px;margin-top:14px;font-size:12px}
.v-robot .vr-route .prog>i{transition:width .15s linear}
.v-robot .vr-found{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px;padding:9px 11px;border-radius:10px;background:var(--accent-soft);border:1px solid var(--accent-line);font-size:12.5px;animation:pop .4s var(--ease) both}
.v-robot .vr-found[hidden]{display:none}
.v-robot .vr-ck{width:16px;height:16px;border-radius:50%;background:var(--ok);color:var(--card);display:grid;place-items:center;flex:none}
.v-robot .vr-acts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-top:14px}
.v-robot .vr-mission{margin-top:16px}
.v-robot .vr-mission ol{list-style:none;margin:10px 0 0;padding:0;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));position:relative}
.v-robot .vr-mission ol::before{content:"";position:absolute;left:10%;right:10%;top:7px;height:2px;background:var(--line)}
.v-robot .vr-mission li{display:grid;justify-items:center;gap:5px;font-size:11.5px;color:var(--ink-3);position:relative;min-width:0}
.v-robot .vr-mission li span{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-robot .vr-g span{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.v-robot .vr-mission li i{width:16px;height:16px;border-radius:50%;background:var(--card);border:2px solid var(--line-2);transition:background .3s,border-color .3s,box-shadow .3s}
.v-robot .vr-mission li.on i{border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)} .v-robot .vr-mission li.on span{color:var(--accent);font-weight:500}
.v-robot .vr-mission li.done i{background:var(--ok);border-color:var(--ok)} .v-robot .vr-mission li.done span{color:var(--ink)}
.v-robot .vr-bump{animation:pop .45s var(--ease)}
.v-robot .vr-chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.v-robot .vr-chatlog{height:270px;overflow:auto;display:flex;flex-direction:column;gap:10px;padding:12px;background:var(--sunk);border:1px solid var(--line);border-radius:12px;scroll-behavior:smooth}
.v-robot .vr-m{display:flex;gap:8px;align-items:flex-end;max-width:88%;animation:rise .35s var(--ease) both}
.v-robot .vr-m.me{align-self:flex-end;flex-direction:row-reverse}
.v-robot .vr-bub{background:var(--card);border:1px solid var(--line);padding:8px 11px;border-radius:12px 12px 12px 4px;font-size:13px;min-width:0}
.v-robot .vr-m.me .vr-bub{background:var(--accent);border-color:var(--accent);color:var(--card);border-radius:12px 12px 4px 12px}
.v-robot .vr-from{display:block;font-size:10.5px;color:var(--ink-3);font-weight:500;margin-bottom:2px}
.v-robot .vr-in{display:flex;gap:8px;margin-top:12px}
.v-robot .vr-in input{flex:1;min-width:0;border:1px solid var(--line-2);border-radius:10px;padding:9px 12px;background:var(--card);outline:0;transition:border-color .2s,box-shadow .2s}
.v-robot .vr-in input:focus{border-color:var(--accent);box-shadow:0 0 0 4px var(--accent-soft)}
.v-robot .vr-log{height:362px;overflow:auto;display:grid;align-content:start}
.v-robot .vr-ev{display:grid;grid-template-columns:auto 8px minmax(0,1fr);gap:10px;align-items:baseline;padding:8px 2px;border-bottom:1px solid var(--line);font-size:12.5px;animation:rise .35s var(--ease) both}
.v-robot .vr-ev time{font:11.5px var(--f-mono);color:var(--ink-3)}
.v-robot .vr-ev i{width:8px;height:8px;border-radius:50%;background:var(--accent);align-self:center}
.v-robot .vr-ev i.ok{background:var(--ok)} .v-robot .vr-ev i.warn{background:var(--warn)} .v-robot .vr-ev i.sim{background:var(--violet)} .v-robot .vr-ev i.plain{background:var(--ink-4)}
`;

  async function mount(root, ctx) {
    const { el, esc } = STM;
    let alive = true, raf = 0, rafB = 0, rafL = 0, rafC = 0, iv = 0, job = 0, finding = false, callT = 0, nEv = 0;
    const timers = new Set();
    const later = (fn, ms) => { const id = setTimeout(() => { timers.delete(id); if (alive) fn(); }, ms); timers.add(id); return id; };
    const wait = ms => new Promise(r => later(r, ms));
    root.append(el("style", {}, CSS));
    const wrap = el("div", { class: "v-robot" });
    root.append(wrap);
    STM.pageHead(wrap, "Robot", "Find Rover 1 on the cameras and send it to the last sighting.");
    const sk = el("div", { class: "vr-grid" },
      el("div", { class: "card card-b" }, el("div", { class: "skel", style: { aspectRatio: "16 / 9" } })),
      el("div", { class: "card card-b grid" }, STM.skeleton(22, "55%"), STM.skeleton(86), STM.skeleton(14), STM.skeleton(14, "80%"), STM.skeleton(38)));
    wrap.append(sk);
    const s = ctx.scene || await STM.loadScene("lab");
    await wait(STM.reduced ? 0 : 320);
    if (!wrap.isConnected) { alive = false; timers.forEach(clearTimeout); return () => {}; }
    sk.remove();

    /* ---------- geometry from the real scene ---------- */
    const cams = s.cameras;
    const xs = cams.map((c, i) => (c.pos ? c.pos[0] : i)), mn = Math.min(...xs), mx = Math.max(...xs);
    const geo = cams.map((c, i) => ({ id: c.id, name: c.name || "Camera " + c.id, x: cams.length < 2 ? 480 : 250 + (xs[i] - mn) / (mx - mn || 1) * 455, y: 36, dir: PI / 2, half: PI / 6, range: 400 }));
    const gOf = id => geo.find(g => g.id === id) || geo[0];
    const inCone = (g, p) => { const dx = p.x - g.x, dy = p.y - g.y; return Math.hypot(dx, dy) <= g.range && Math.abs(angN(Math.atan2(dy, dx) - g.dir)) <= g.half; };
    const proj = (g, b) => { const cx = b[0] + b[2] / 2, foot = Math.min(1, b[1] + b[3]); return pt(g, g.dir - (cx - .5) * 2 * g.half * .9, g.range * (.92 - .72 * foot)); };
    const pick = pickTrack(s), T = pick.t, last = T.boxes[T.boxes.length - 1], sg = gOf(T.cam), sp = proj(sg, last.slice(1));
    const sightClock = s.clock(last[0]), hex = STM.colourHex[(T.look || {}).top] || STM.colourHex.grey;
    const sightTarget = { x: clamp(sp.x, 110, 890), y: clamp(sp.y + (sp.y < AISLE ? 40 : -40), 190, 340) };
    const trail = T.boxes.filter((b, i) => i % 2 === 0).concat([last]).map(b => proj(sg, b.slice(1)));
    const zoneOf = p => {
      if (Math.hypot(p.x - DOCK.x, p.y - DOCK.y) < 20) return "Charger dock";
      if (inRect(p, CAFE)) return "Cafe";
      const c = geo.filter(g => inCone(g, p)).sort((a, b) => Math.abs(a.x - p.x) - Math.abs(b.x - p.x))[0];
      if (Math.abs(p.y - AISLE) < 34) return c ? `Main aisle, camera ${c.id}` : "Main aisle";
      return c ? `Camera ${c.id} area (${c.name})` : "Lab floor";
    };

    /* ---------- simulated robot state ---------- */
    const R = { x: DOCK.x, y: DOCK.y, h: -PI / 2 };
    let battery = 86, speed = 0, eta = 0, prog = 0, odo = 1284.3, state = "Docked", task = "Charging at the dock";

    /* ---------- map ---------- */
    const camSvg = geo.map(g => {
      const lab = `Cam ${g.id} · ${g.name}`, w = lab.length * 7.4 + 18;
      return `<rect class="m-cambody" x="${f1(g.x - 11)}" y="24" width="22" height="12" rx="3"/><circle class="m-lens" cx="${f1(g.x)}" cy="38" r="5"/>
        <g class="m-camlab" transform="translate(${f1(g.x)} 72)"><rect x="${f1(-w / 2)}" y="-12" width="${f1(w)}" height="22" rx="7"/><text y="4">${esc(lab)}</text></g>`;
    }).join("");
    const others = s.tracks.filter(t => t !== T && t.boxes.length).map(t => {
      const b = t.boxes[t.boxes.length - 1], p = proj(gOf(t.cam), b.slice(1)), c = STM.colourHex[(t.look || {}).top] || STM.colourHex.grey;
      return `<circle class="m-other" cx="${f1(p.x)}" cy="${f1(p.y)}" r="4" style="fill:${c}"><title>${esc(t.id)} · ${esc(t.desc)} · last seen ${s.clock(b[0])}</title></circle>`;
    }).join("");
    const tag = `${T.id} · ${sightClock}`, tw = tag.length * 8 + 16;
    const svg = `<svg class="vr-map" viewBox="0 0 960 540" role="img" aria-label="Floor map with cameras, the last sighting and Rover 1">
<defs><pattern id="vr-grid" width="30" height="30" patternUnits="userSpaceOnUse"><path d="M30 0H0V30" class="m-gl"/></pattern>
<radialGradient id="vr-rad" cx="0" cy="0" r="84" gradientUnits="userSpaceOnUse"><stop offset="0" style="stop-color:var(--accent);stop-opacity:.34"/><stop offset="1" style="stop-color:var(--accent);stop-opacity:0"/></radialGradient>
<clipPath id="vr-clip"><rect x="30" y="30" width="900" height="480" rx="6"/></clipPath></defs>
<rect class="m-floor" x="30" y="30" width="900" height="480" rx="6"/><rect x="30" y="30" width="900" height="480" fill="url(#vr-grid)"/>
<rect class="m-aisle" x="30" y="${AISLE - 30}" width="900" height="60"/><path class="m-aisle-c" d="M44 ${AISLE}H916"/>
<rect class="m-cafe" x="${CAFE[0]}" y="${CAFE[1]}" width="${CAFE[2]}" height="${CAFE[3]}" rx="10"/>
${TABLES.map(([x, y]) => `<circle class="m-table" cx="${x}" cy="${y}" r="14"/>`).join("")}
${BENCH.map(([x, y, w, h]) => `<rect class="m-bench" x="${x}" y="${y}" width="${w}" height="${h}" rx="5"/>`).join("")}
<rect class="m-dock" x="114" y="384" width="52" height="40" rx="8"/><path class="m-pin" d="M130 420h20"/>
<g clip-path="url(#vr-clip)">${geo.map(g => `<path class="m-cone" data-cam="${esc(g.id)}" d="${sector(g, g.dir - g.half, g.dir + g.half, g.range)}"/>`).join("")}<path class="m-beamw"/><line class="m-beam"/></g>
<path class="m-wall" d="M30 270V30H930V270M930 330V510H30V330"/>
<text class="m-txt" x="44" y="262">Entrance</text><text class="m-txt end" x="916" y="262">East exit</text><text class="m-txt mid" x="105" y="64">Racks</text>
<text class="m-txt mid" x="715" y="108">Workbenches</text><text class="m-txt mid" x="760" y="420">Benches</text><text class="m-txt" x="408" y="428">Cafe</text>
<text class="m-txt" x="174" y="430">Charger</text><text class="m-txt" x="330" y="${AISLE - 8}">Main aisle</text>
${camSvg}
<polyline class="m-trail" points="${trail.map(p => f1(p.x) + "," + f1(p.y)).join(" ")}" style="stroke:${hex}"/>${others}
<line class="m-seen" x1="${f1(sg.x)}" y1="42" x2="${f1(sp.x)}" y2="${f1(sp.y)}"/>
<g class="m-planset gone"><path class="m-plan-bg"/><path class="m-plan"/><path class="m-done"/></g>
<g id="robot-sighting" transform="translate(${f1(sp.x)} ${f1(sp.y)})"><circle class="m-pulse" r="8" style="fill:${hex}"/><circle class="m-pulse b" r="8" style="fill:${hex}"/><circle class="m-sdot" r="7.5" style="fill:${hex}"/>
<g class="m-tag" transform="translate(0 -22)"><rect x="${f1(-tw / 2)}" y="-11" width="${f1(tw)}" height="20" rx="6"/><text y="3.5">${esc(tag)}</text></g></g>
<g class="m-lidar"></g>
<g class="m-robot"><g class="m-radar"><path d="${sector({ x: 0, y: 0 }, 0, .9, 84)}" fill="url(#vr-rad)"/><animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="3.2s" repeatCount="indefinite"/></g>
<circle class="m-guard" r="45"/><circle class="m-ring" r="20"/>
<g class="m-body"><rect class="m-rbody" x="-15" y="-12" width="30" height="24" rx="8"/><path class="m-head" d="M18 -6L26 0L18 6Z"/><circle class="m-eye" cx="7" cy="-5" r="2.2"/><circle class="m-eye" cx="7" cy="5" r="2.2"/></g>
<text class="m-rlab" y="34">Rover 1</text></g>
<g class="m-call" opacity="0"><rect y="-12" height="24" rx="7"/><text y="4"></text></g>
<g class="m-scale"><path d="M770 490h150M770 484v12M920 484v12"/><text x="845" y="478">5 m</text></g></svg>`;

    const crop = el("img", { class: "crop", src: s.cropUrl(T), alt: `Crop of ${T.id}`, loading: "lazy" });
    const sw = k => (T.look && T.look[k] && STM.colourHex[T.look[k]]) ? `<span class="swatch" style="background:${STM.colourHex[T.look[k]]}"></span>` : "";
    const mapCard = el("div", { class: "card", id: "robot-map" });
    mapCard.innerHTML = `<div class="card-h"><h3>Floor map</h3><div class="grow"></div><span class="muted mono" style="font-size:12px;white-space:nowrap">${FAST}× speed</span></div>
      <div class="card-b"><div class="vr-mapwrap">${svg}</div>
      <div class="vr-sight"><span class="vr-cropbox"></span><div class="vr-who"><span class="eyebrow">Last sighting${pick.from === "desk" ? " · desk lead" : ""}</span>
        <span><b class="mono">${esc(T.id)}</b> ${sw("top")} ${sw("bottom")} ${esc(T.desc || "")}</span>
        <span class="muted">Camera ${esc(sg.id)} · ${esc(sg.name)} · ${sightClock}</span></div>
        <div class="vr-links"><button class="btn btn-sm" type="button" id="robot-open-timeline">Timeline</button><button class="btn btn-sm" type="button" id="robot-open-catalogue">Catalogue</button><button class="btn btn-sm" type="button" id="robot-open-similar">Similar looks</button></div></div>
      <div class="vr-legend"><span><i class="lg cone"></i>Camera view</span><span><i class="lg dot" style="background:${hex}"></i>Last sighting</span><span><i class="lg dash"></i>Planned path</span><span><i class="lg bot"></i>Rover 1</span></div></div>`;
    mapCard.querySelector(".vr-cropbox").replaceWith(crop);
    const q = sel => mapCard.querySelector(sel);
    const robotG = q(".m-robot"), bodyG = q(".m-body"), radar = q(".m-radar"), ring = q(".m-ring"), guard = q(".m-guard");
    const lid = q(".m-lidar"), beam = q(".m-beam"), beamW = q(".m-beamw"), call = q(".m-call");
    const planSet = q(".m-planset"), planBg = q(".m-plan-bg"), plan = q(".m-plan"), done = q(".m-done");
    const coneEl = id => mapCard.querySelector(`.m-cone[data-cam="${id}"]`);

    /* ---------- status card ---------- */
    const gauge = (id, label) => `<div class="vr-g"><div class="vr-ring"><svg viewBox="0 0 64 64"><circle class="trk" cx="32" cy="32" r="26"/><circle class="val" data-g="${id}" cx="32" cy="32" r="26" stroke-dasharray="${C.toFixed(2)}" stroke-dashoffset="${C.toFixed(2)}" transform="rotate(-90 32 32)"/></svg><b data-n="${id}">0</b></div><span>${label}</span></div>`;
    const stat = el("div", { class: "card", id: "robot-status" });
    stat.innerHTML = `<div class="card-h"><h3>Rover 1</h3><div class="grow"></div><span class="pill ok" id="robot-state">Docked</span></div>
      <div class="card-b"><div class="vr-gauges">${gauge("bat", "Battery")}${gauge("spd", "Speed m/s")}${gauge("eta", "ETA")}</div>
      <dl class="kv"><dt>Location</dt><dd data-k="loc"></dd><dt>Task</dt><dd data-k="task"></dd><dt>Odometer</dt><dd class="mono" data-k="odo">0</dd><dt>Link</dt><dd>Simulated, no hardware</dd></dl>
      <div class="vr-route"><div class="row"><span class="muted">Route progress</span><div class="grow"></div><span class="mono" data-k="pct">0%</span></div><div class="prog"><i data-k="bar"></i></div></div>
      <div class="vr-mission" id="robot-mission"><span class="eyebrow">Mission</span><ol>${["Find", "Plan", "Drive", "Scan", "Hold"].map(n => `<li><i></i><span>${n}</span></li>`).join("")}</ol></div>
      <div class="vr-found" id="robot-found" hidden></div>
      <div class="vr-acts"><button class="btn" type="button" id="robot-find"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>Find the robot</button>
      <button class="btn btn-primary" type="button" id="robot-send">Send to last sighting ${STM.icons.arrow}</button></div></div>`;
    const sq = sel => stat.querySelector(sel);
    const pill = sq("#robot-state"), found = sq("#robot-found"), btnFind = sq("#robot-find");
    const gv = id => sq(`[data-g="${id}"]`), gn = id => sq(`[data-n="${id}"]`), kv = k => sq(`[data-k="${k}"]`);

    /* ---------- console + log ---------- */
    const con = el("div", { class: "card", id: "robot-console" });
    con.innerHTML = `<div class="card-h"><h3>Talk to Rover 1</h3></div>
      <div class="card-b"><div class="vr-chips">${CMDS.map((c, i) => `<button type="button" class="chip" id="robot-cmd-${i}">${esc(c)}</button>`).join("")}</div>
      <div class="vr-chatlog" id="robot-chat"></div>
      <form class="vr-in"><input id="robot-input" type="text" autocomplete="off" placeholder="Tell Rover 1 what to do, e.g. go to camera A"><button class="btn btn-primary" type="submit" id="robot-say">Send</button></form></div>`;
    const chat = con.querySelector("#robot-chat"), input = con.querySelector("#robot-input");
    const logCard = el("div", { class: "card" });
    logCard.innerHTML = `<div class="card-h"><h3>Event log</h3><div class="grow"></div><span class="muted mono" style="font-size:12px" data-k="n">0 events</span></div><div class="card-b"><div class="vr-log" id="robot-log"></div></div>`;
    const logEl = logCard.querySelector("#robot-log");

    wrap.append(
      el("div", { class: "vr-grid stagger" }, mapCard, stat),
      el("div", { class: "vr-grid stagger" }, con, logCard));

    /* ---------- ui ---------- */
    const setRing = (id, f) => gv(id).setAttribute("stroke-dashoffset", (C * (1 - clamp(f, 0, 1))).toFixed(2));
    function ui(skipNums) {
      setRing("bat", battery / 100); gv("bat").setAttribute("class", "val " + (battery > 30 ? "ok" : "warn"));
      setRing("spd", speed / VMAX); setRing("eta", prog);
      if (!skipNums) { gn("bat").textContent = battery.toFixed(1) + "%"; kv("odo").textContent = odo.toFixed(1) + " m"; }
      gn("spd").textContent = speed.toFixed(2); gn("eta").textContent = eta > 0 ? Math.ceil(eta) + " s" : "--";
      kv("loc").textContent = zoneOf(R); kv("task").textContent = task;
      kv("pct").textContent = Math.round(prog * 100) + "%"; kv("bar").style.width = (prog * 100).toFixed(1) + "%";
    }
    function countUp(node, to, dec, suffix, ms) {          // like STM.countUp, but never shows a negative first frame
      const t0 = performance.now();
      const step = () => { if (!alive) return; const f = clamp((performance.now() - t0) / ms, 0, 1); node.textContent = (to * (1 - Math.pow(1 - f, 3))).toFixed(dec) + suffix; if (f < 1) rafC = requestAnimationFrame(step); };
      if (STM.reduced) node.textContent = to.toFixed(dec) + suffix; else step();
    }
    function mission(i, fin) { stat.querySelectorAll(".vr-mission li").forEach((li, k) => { li.className = k < i || (fin && k === i) ? "done" : k === i ? "on" : ""; }); }
    function setState(st) { state = st; pill.className = "pill " + PILL[st]; pill.textContent = st; void pill.offsetWidth; pill.classList.add("vr-bump"); }
    function place() { robotG.setAttribute("transform", `translate(${f1(R.x)} ${f1(R.y)})`); bodyG.setAttribute("transform", `rotate(${f1(R.h * 180 / PI)})`); }
    function log(text, kind = "info") {
      logEl.prepend(el("div", { class: "vr-ev" }, el("time", {}, new Date().toTimeString().slice(0, 8)), el("i", { class: kind }), el("span", {}, text)));
      nEv++; logCard.querySelector('[data-k="n"]').textContent = `${nEv} event${nEv === 1 ? "" : "s"}`;
    }
    function addMsg(who, html) {
      const m = el("div", { class: "vr-m " + who }, who === "bot" ? el("div", { class: "vr-ava", html: STM.icons.robot }) : null);
      const b = el("div", { class: "vr-bub", html }); m.append(b); chat.append(m); chat.scrollTop = chat.scrollHeight; return b;
    }
    let chain = Promise.resolve();
    function say(html, ms = 700) {
      chain = chain.then(async () => {
        const b = addMsg("bot", '<span class="dots"><i></i><i></i><i></i></span>');
        await wait(STM.reduced ? 0 : ms);
        b.innerHTML = `<span class="vr-from">Rover 1</span>${html}`; chat.scrollTop = chat.scrollHeight;
      });
      return chain;
    }
    function callout(text) {
      const t = call.querySelector("text"), w = text.length * 8 + 18; t.textContent = text;
      const r = call.querySelector("rect"); r.setAttribute("x", f1(-w / 2)); r.setAttribute("width", f1(w));
      call.setAttribute("transform", `translate(${f1(clamp(R.x, 40 + w / 2, 920 - w / 2))} ${f1(R.y - 48)})`); call.setAttribute("opacity", 1);
      clearTimeout(callT); callT = later(() => call.setAttribute("opacity", 0), 4500);
    }

    /* ---------- motion ---------- */
    function planPath(from, to) {
      const pts = [{ x: from.x, y: from.y }];
      if (Math.abs(from.x - to.x) > 2) { if (Math.abs(from.y - AISLE) > 2) pts.push({ x: from.x, y: AISLE }); pts.push({ x: to.x, y: AISLE }); }
      pts.push({ x: to.x, y: to.y });
      return pts.filter((p, i) => !i || Math.hypot(p.x - pts[i - 1].x, p.y - pts[i - 1].y) > 1);
    }
    function at(pts, d) {
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i], l = Math.hypot(b.x - a.x, b.y - a.y);
        if (d <= l || i === pts.length - 1) { const k = l ? Math.min(1, d / l) : 1; return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k, a: Math.atan2(b.y - a.y, b.x - a.x) }; }
        d -= l;
      }
      return { x: R.x, y: R.y, a: R.h };
    }
    function drive(pts, L, Tsim, my) {
      return new Promise(res => {
        const dur = Math.max(600, Tsim / FAST * 1000) * (STM.reduced ? .15 : 1), t0 = performance.now();
        let lastD = 0;
        const step = now => {
          if (!alive || my !== job) return;
          const f = Math.min(1, (now - t0) / dur), d = L * ease(f), p = at(pts, d), dm = (d - lastD) / PXM;
          lastD = d; R.x = p.x; R.y = p.y; R.h += angN(p.a - R.h) * .22;
          battery = Math.max(5, battery - dm * .06); odo += dm;
          speed = VMAX * Math.sin(PI * f); eta = Tsim * (1 - f); prog = ease(f);
          place(); done.setAttribute("stroke-dasharray", `${f1(d)} ${f1(L + 20)}`); ui();
          if (f < 1) raf = requestAnimationFrame(step); else { speed = 0; eta = 0; ui(); res(true); }
        };
        raf = requestAnimationFrame(step);
      });
    }
    function halt() { job++; cancelAnimationFrame(raf); speed = 0; eta = 0; guard.classList.remove("on"); }
    async function travel(target, label) {
      halt(); const my = job;
      task = `Planning a path to ${label}`; mission(1); ui();
      await wait(STM.reduced ? 0 : 480); if (my !== job) return false;
      const pts = planPath(R, target);
      let L = 0; for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
      if (L < 3) { prog = 1; ui(); return true; }
      const d = "M" + pts.map(p => `${f1(p.x)} ${f1(p.y)}`).join("L"), Lm = L / PXM, Tsim = Lm / (VMAX / (PI / 2));
      planSet.classList.remove("gone"); [planBg, plan, done].forEach(n => n.setAttribute("d", d));
      planBg.style.transition = "none"; planBg.style.strokeDasharray = `${f1(L)}px ${f1(L)}px`; planBg.style.strokeDashoffset = f1(L) + "px";
      plan.classList.remove("on"); done.setAttribute("stroke-dasharray", `0 ${f1(L + 20)}`);
      void planBg.getBoundingClientRect(); planBg.style.transition = ""; planBg.style.strokeDashoffset = "0px"; plan.classList.add("on");
      const turns = pts.length - 2;
      log(`Path planned to ${label}: ${Lm.toFixed(1)} m, ${turns} turn${turns === 1 ? "" : "s"}`, "info");
      say(`Route planned to ${esc(label)}: ${Lm.toFixed(1)} m along the aisles, about ${Math.round(Tsim)} s at up to ${VMAX} m/s.`, 450);
      await wait(STM.reduced ? 0 : 520); if (my !== job) return false;
      prog = 0; mission(2); setState("Moving"); task = `Driving to ${label}`; call.setAttribute("opacity", 0);
      const ok = await drive(pts, L, Tsim, my); if (!ok || my !== job) return false;
      later(() => { if (my === job) planSet.classList.add("gone"); }, 1400);
      log(`Arrived at ${label}`, "ok");
      return true;
    }
    function lidar(ms, my) {
      return new Promise(res => {
        const p = { x: R.x, y: R.y }, h = R.h, RR = 170, a0 = h - PI / 2, t0 = performance.now();
        lid.classList.remove("fade"); radar.classList.add("off");
        lid.innerHTML = `<path class="m-arc" d="${sector(p, a0, h + PI / 2, RR)}"/><path class="m-lid"/>`;
        const wedge = lid.lastChild; let next = a0, hits = 0;
        const step = now => {
          if (!alive) return;
          if (my !== job) { lid.innerHTML = ""; radar.classList.remove("off"); return res(-1); }
          const f = Math.min(1, (now - t0) / ms), a = a0 + PI * f;
          wedge.setAttribute("d", sector(p, Math.max(a0, a - .35), a, RR));
          let dots = "";
          for (; next <= a + 1e-9; next += .045) { const r = ray(p, next, RR); if (r != null) { const c = pt(p, next, r); dots += `<circle class="m-hit" cx="${f1(c.x)}" cy="${f1(c.y)}" r="2.6"/>`; hits++; } }
          if (dots) lid.insertAdjacentHTML("beforeend", dots);
          if (f < 1) rafL = requestAnimationFrame(step);
          else { wedge.remove(); radar.classList.remove("off"); later(() => lid.classList.add("fade"), 2400); res(hits); }
        };
        rafL = requestAnimationFrame(step);
      });
    }
    async function scanHere(label) {
      const my = job; mission(3); setState("Scanning"); task = label; ui();
      const hits = await lidar(STM.reduced ? 300 : 2400, my);
      return my === job ? hits : -1;
    }

    /* ---------- actions ---------- */
    function sweep(g, ms) {
      return new Promise(res => {
        const t0 = performance.now(), a0 = g.dir - g.half;
        const step = now => {
          if (!alive) return;
          const f = Math.min(1, (now - t0) / ms), a = a0 + 2 * g.half * ease(f), c = pt(g, a, g.range);
          beam.setAttribute("x1", f1(g.x)); beam.setAttribute("y1", g.y); beam.setAttribute("x2", f1(c.x)); beam.setAttribute("y2", f1(c.y));
          beamW.setAttribute("d", sector(g, a0, Math.max(a0 + .001, a), g.range));
          if (f < 1) rafB = requestAnimationFrame(step); else res();
        };
        rafB = requestAnimationFrame(step);
      });
    }
    async function find() {
      if (finding) return; finding = true; btnFind.disabled = true;
      mission(0); found.hidden = false; found.innerHTML = `<span class="spin"></span>Checking ${geo.length} cameras for Rover 1`;
      log(`Looking for Rover 1 on ${geo.length} cameras`, "info");
      for (const g of geo) {
        coneEl(g.id).classList.add("on"); beam.classList.add("on");
        await sweep(g, STM.reduced ? 60 : 720);
        coneEl(g.id).classList.remove("on"); beamW.setAttribute("d", "");
      }
      beam.classList.remove("on");
      const seen = geo.filter(g => inCone(g, R)).sort((a, b) => Math.abs(a.x - R.x) - Math.abs(b.x - R.x))[0];
      ring.classList.remove("on"); void ring.getBoundingClientRect(); ring.classList.add("on");
      const where = state === "Docked" ? "docked at the charger" : `${state.toLowerCase()} at ${zoneOf(R)}`;
      const msg = seen ? `Rover 1 seen on camera ${seen.id}, ${where}` : `Rover 1 is outside both camera views; position from its odometry: ${zoneOf(R)}`;
      callout(seen ? `Camera ${seen.id} · Rover 1` : "Odometry only");
      found.innerHTML = `<span class="vr-ck">${STM.icons.check}</span><span>${esc(msg)}.</span>`;
      if (state === "Docked" || state === "Holding") mission(0, true);
      log(msg, "ok"); STM.toast(esc(msg));
      finding = false; btnFind.disabled = false;
      return msg;
    }
    async function goSighting() {
      if (!(await travel(sightTarget, "the last sighting"))) return;
      say(`Arrived at the last sighting, ${esc(zoneOf(R))}. Scanning 180 degrees.`, 450);
      const hits = await scanHere("Lidar scan, 180 degrees"); if (hits < 0) return;
      mission(4); setState("Holding"); task = "Holding at the last sighting"; guard.classList.add("on"); ui();
      log(`Scan at the last sighting: ${hits} lidar returns`, "sim");
      say(`Scan done: ${hits} lidar returns, aisle clear. I can not identify people, so I am holding here. The desk last saw ${esc(T.id)} (${esc(T.desc || "no description")}) on camera ${esc(sg.id)} at ${sightClock}.`);
    }
    async function goCam(id) {
      const g = geo.find(k => k.id.toUpperCase() === id);
      if (!g) return say(`There is no camera ${esc(id)} in this scene. Cameras: ${geo.map(k => k.id).join(", ")}.`);
      if (!(await travel({ x: clamp(g.x, 110, 890), y: 240 }, `camera ${g.id} aisle`))) return;
      say(`Arrived at camera ${esc(g.id)} aisle. Scanning 180 degrees.`, 450);
      const hits = await scanHere("Lidar scan, 180 degrees"); if (hits < 0) return;
      mission(4, true); setState("Holding"); task = `Waiting at camera ${g.id} aisle`; ui();
      log(`Scan at camera ${g.id} aisle: ${hits} lidar returns`, "sim");
      say(`Scan done: ${hits} lidar returns, nothing blocking the aisle. Waiting here for the next task.`);
    }
    async function goDock() {
      if (!(await travel({ x: DOCK.x, y: DOCK.y }, "the dock"))) return;
      mission(-1); setState("Docked"); task = "Charging at the dock"; R.h = -PI / 2; place(); ui();
      say(`Docked at the charger. Battery ${battery.toFixed(0)}%, charging.`);
    }
    async function scanFor(what) {
      what = (what || "").replace(/[.?!]+$/, "").trim() || "anything unusual";
      const wasDocked = state === "Docked"; halt();
      say(`Scanning 180 degrees for ${esc(what)} at ${esc(zoneOf(R))}.`, 450);
      const hits = await scanHere(`Scanning for ${what}`); if (hits < 0) return;
      mission(3, true); setState(wasDocked ? "Docked" : "Holding"); task = wasDocked ? "Charging at the dock" : `Holding at ${zoneOf(R)}`; ui();
      log(`Scan for ${what}: nothing matched`, "sim");
      const it = STM.state.item, it_t = it && (it.t != null ? it.t : it.time);
      const tail = /bag|backpack|case|item|phone|wallet|umbrella/.test(what)
        ? (typeof it_t === "number" ? `the desk still shows it at ${s.clock(it_t)}${it.cam ? " on camera " + esc(it.cam) : ""}.` : "the index has no bag logged for this scene, so the desk has nothing to compare against.")
        : "I check spots and items only; ask the desk to search people on the cameras.";
      say(`No ${esc(what)} in view; ${tail}`);
    }
    function hold() {
      halt(); mission(4); setState("Holding"); task = "Guarding a 1.5 m radius"; guard.classList.add("on"); ui();
      log(`Holding and guarding at ${zoneOf(R)}`, "warn");
      say(`Holding at ${esc(zoneOf(R))}. Guarding a 1.5 m radius. I will report anyone who stops at the spot.`);
    }
    function command(text) {
      const raw = String(text || "").trim(); if (!raw) return;
      addMsg("me", esc(raw)); log(`Desk command: ${raw}`, "plain");
      const l = raw.toLowerCase(); let m;
      if (/\b(stop|cancel|halt|freeze)\b/.test(l)) { const was = state; halt(); setState(was === "Docked" ? "Docked" : "Holding"); task = "Stopped"; ui(); return say(`Stopped at ${esc(zoneOf(R))}.`); }
      if (/(where are you|find (the )?(robot|rover)|locate)/.test(l)) return find().then(m => m && say(`Cameras report: ${esc(m)}.`, 450));
      if (/\b(dock|charg\w*|home|return)\b/.test(l)) return goDock();
      if ((m = l.match(/\bcam(?:era)?\s*([a-z])\b/))) return goCam(m[1].toUpperCase());
      if (/(sighting|last seen|go there|send|follow)/.test(l)) return goSighting();
      if (/\b(hold|guard|stay|wait|watch)\b/.test(l)) return hold();
      if ((m = l.match(/\b(?:scan|look|search|check)\b(?:\s+(?:for|around))?\s*(?:(?:an?|the)\s+)?(.*)$/))) return scanFor(m[1]);
      if (/(status|battery|how are you|report)/.test(l)) return say(`Battery ${battery.toFixed(0)}%, ${state.toLowerCase()} at ${esc(zoneOf(R))}. Odometer ${odo.toFixed(1)} m.`);
      return say("I understand: go to camera A or B, go to the last sighting, scan for an item, hold and guard, stop, return to dock, status.");
    }

    /* ---------- wire up ---------- */
    btnFind.onclick = find;
    sq("#robot-send").onclick = () => command("Go to the last sighting");
    CMDS.forEach((c, i) => { con.querySelector(`#robot-cmd-${i}`).onclick = () => command(c); });
    con.querySelector("form").onsubmit = e => { e.preventDefault(); command(input.value); input.value = ""; };
    [["timeline", "timeline"], ["catalogue", "catalogue"], ["similar", "similar"]].forEach(([k, v]) => { q(`#robot-open-${k}`).onclick = () => STM.go(v); });
    const api = { find, send: goSighting, command };
    STM.robot = api;

    place();
    requestAnimationFrame(() => { if (!alive) return; ui(true); countUp(gn("bat"), battery, 1, "%", 900); countUp(kv("odo"), odo, 1, " m", 1100); });
    log("Rover 1 connected. Simulated robot, no hardware linked", "sim");
    log(`Cameras on the map: ${geo.map(g => `${g.id} (${g.name})`).join(", ")}`, "info");
    later(() => log(`Last sighting from the ${pick.from === "desk" ? "desk lead" : "index"}: ${T.id}, ${T.desc || "no description"}, camera ${sg.id} at ${sightClock}`, "ok"), 250);
    later(() => log(`Docked at the charger, battery ${battery.toFixed(0)}%`, "ok"), 500);
    later(() => say(`Rover 1 here, docked at the charger with ${battery.toFixed(0)}% battery. Send me to the last sighting or pick a command.`, 650), 450);
    later(() => { iv = setInterval(() => { if (alive && state === "Docked" && battery < 100) { battery = Math.min(100, battery + .1); ui(); } }, 1000); }, 1600);

    return () => {
      alive = false; job++;
      timers.forEach(clearTimeout); timers.clear(); clearInterval(iv);
      cancelAnimationFrame(raf); cancelAnimationFrame(rafB); cancelAnimationFrame(rafL); cancelAnimationFrame(rafC);
      if (STM.robot === api) delete STM.robot;
    };
  }

  STM.view("robot", { label: "Robot", mount });
})();
