"""Build out/draw.html: a self-contained interactive MIMIC page. Draw a path on the canvas; DTW runs in the page against the
embedded 32-point descriptors of every tracklet; top matches show with thumbnails (a few hundred plausible tracklets) or
trajectory glyphs over a clip backdrop (the rest). Page budget: < 15 MB."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mimic_core as mc  # noqa: E402

pk = mc.pk
MAX_THUMBS = 520
THUMB_W = 224


def jpg_b64(img: np.ndarray, q: int = 58) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def pick_thumbs(d: dict, df, arr) -> list[str]:
    """Tracklets worth a real thumbnail: the top results of many probe queries (presets x mirrors/rotations, whole table and
    per clip) plus the longest-moving ones to fill the budget."""
    chosen: dict[str, None] = {}
    probes = []
    for p in mc.PRESETS:
        q0 = mc.preset_query(p)
        for flip in (False, True):
            for ang in (0, 90, 180, 270):
                pts = (q0.pts - 0.5) @ mc.rot2(np.deg2rad(ang)).T + 0.5
                if flip:
                    pts = pts * [-1, 1] + [1, 0]
                probes.append(mc.Query(pts, q0.aspect, q0.pos_w, q0.size_w, q0.any_rot, q0.speed, q0.speed_w, p, q0.shape_w, q0.turn_w, q0.straight_w))
    for q in probes:
        for k in mc.match(q, df, arr, k=14).key:
            chosen[k] = None
    for v in d["clips"]:
        m = (df.video == v).values
        if m.sum() < 3:
            continue
        sub = df[m].reset_index(drop=True)
        a = {k: x[m] for k, x in arr.items()}
        for p in mc.PRESETS:
            for k in mc.match(mc.preset_query(p), sub, a, k=6).key:
                chosen[k] = None
    rest = df.assign(score=df["len"] * np.minimum(df["dur"], 10)).sort_values("score", ascending=False)
    for k in rest.key:
        if len(chosen) >= MAX_THUMBS:
            break
        chosen.setdefault(k, None)
    return list(chosen)[:MAX_THUMBS]


def _pauses(d: dict, key: str) -> list:
    """Up to 3 longest halts of a tracklet as [x, y, seconds] in frame-normalised coordinates (same space as the descriptor's pos)."""
    tl = d["tls"][key]
    runs = sorted(mc.find_pauses(tl, d["meta"][tl.video]["fps"]), key=lambda r: -r[2])[:3]
    return [[round(float(tl.xy[(a + b) // 2][0]), 4), round(float(tl.xy[(a + b) // 2][1]), 4), round(float(sec), 1)] for a, b, sec in runs]


def build(d: dict, df, cal: dict, gres: dict | None = None) -> Path:
    out = HERE / "out"
    arr = mc.tl_arrays(df)
    thumbs = pick_thumbs(d, df, arr)
    tmap: dict[str, int] = {}
    imgs: list[str] = []
    for k in thumbs:
        tl = d["tls"][k]
        try:
            fr = pk.frame_at(tl.video, float(tl.t[len(tl.t) // 2]))
        except Exception:
            continue
        s = THUMB_W / fr.shape[1]
        fr = cv2.resize(fr, (THUMB_W, int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        tmap[k] = len(imgs)
        imgs.append(jpg_b64(fr))
    clip_info = []
    for c in d["clips"]:
        m = d["meta"][c]
        mid = pk.frame_at(c, m["duration"] * 0.4)
        s = 320 / mid.shape[1]
        mid = cv2.resize(mid, (320, int(mid.shape[0] * s)), interpolation=cv2.INTER_AREA)
        clip_info.append({"name": c, "w": m["w"], "h": m["h"], "aspect": m["w"] / m["h"], "bg": jpg_b64(mid, 62),
                          "title": pk.videos()[c]["title"], "n": int((df.video == c).sum())})
    cidx = {c: i for i, c in enumerate(d["clips"])}
    gclips = {}
    if gres:
        for p, r in gres.items():
            cp = out / "global" / f"clip_{p}.mp4"
            if cp.exists() and len(r):
                gclips[r.key.iloc[0]] = f"global/clip_{p}.mp4"
    for c in d["clips"]:
        for p in mc.PRESETS:
            cp = out / c / f"clip_{p}.mp4"
            if cp.exists():
                try:
                    # the best match of that preset inside the clip is the first row of its top8 table; the report names the clip
                    pass
                except Exception:
                    pass
    data = {
        "n": int(len(df)),
        "pos": [round(float(v), 5) for v in np.array(df["pos"].tolist()).ravel()],
        "speed": [round(float(v), 3) for v in np.array(df["speed"].tolist()).ravel()],
        "ext": [round(float(v), 5) for v in df["ext"]], "rel": [round(float(v), 4) for v in df["rel_ext"]], "straight": [round(float(v), 4) for v in df["straight"]], "len": [round(float(v), 5) for v in df["len"]],
        "aspect": [clip_info[cidx[v]]["aspect"] for v in df["video"]],
        "v": [cidx[v] for v in df["video"]], "cls": list(df["cls"]), "t0": [round(float(v), 1) for v in df["t0"]],
        "t1": [round(float(v), 1) for v in df["t1"]], "mid": [round(float(v), 3) for v in df["midf"]],
        "img": [tmap.get(k, -1) for k in df["key"]], "key": list(df["key"]),
        "clip": [gclips.get(k) for k in df["key"]],
        "pz": [_pauses(d, k) for k in df["key"]],
        "clips": clip_info, "imgs": imgs,
        "presets": {p: {"pts": [[round(float(x), 4), round(float(y), 4)] for x, y in mc.preset_points(p)], **{k: v for k, v in mc.PRESETS[p].items() if k != "desc"},
                        "speed": (None if mc.preset_speed(p) is None else [round(float(v), 3) for v in mc.preset_speed(p)]), "desc": mc.PRESETS[p]["desc"],
                        "marks": [[round(x, 4), round(y, 4), lab] for x, y, lab in mc.preset_marks(p)]}
                    for p in mc.PRESETS},
        "cal": cal,
    }
    js = (HERE / "mimic_search.js").read_text(encoding="utf-8")
    html = TEMPLATE.replace("__SEARCH_JS__", js).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    p = out / "draw.html"
    p.write_text(html, encoding="utf-8")
    mb = p.stat().st_size / 1e6
    print(f"[mimic] draw.html: {mb:.1f} MB, {len(imgs)} thumbnails, {len(df)} tracklets")
    assert mb < 15, "page too big"
    return p


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MIMIC draw a path</title>
<style>
:root{--bg:#f4f5f7;--fg:#171b20;--muted:#5b6672;--card:#ffffff;--line:#d9dee4;--acc:#2f6fde;--ok:#1f9d55;--warn:#d98e04;--bad:#d64545;--canvas:#101418}
@media (prefers-color-scheme: dark){:root{--bg:#14171b;--fg:#e8ebee;--muted:#98a3ae;--card:#1c2127;--line:#2b323a;--acc:#6ea0ff;--ok:#4cc77d;--warn:#f0b429;--bad:#ff6b6b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:16px 16px 4px;max-width:1280px;margin:0 auto}h1{margin:0;font-size:26px;letter-spacing:.5px}h1 b{color:var(--acc)}
.sub{color:var(--muted);margin:2px 0 0}
main{max-width:1280px;margin:0 auto;padding:10px 16px 40px;display:grid;grid-template-columns:minmax(300px,640px) 1fr;gap:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
canvas#draw{width:100%;display:block;background:var(--canvas);border-radius:8px;touch-action:none;cursor:crosshair}
.row{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 0;align-items:center}
button,select{font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:5px 10px;cursor:pointer}
button:hover{border-color:var(--acc)}button.on{background:var(--acc);color:#fff;border-color:var(--acc)}
label{font-size:13px;color:var(--muted);display:flex;gap:6px;align-items:center}
input[type=range]{width:120px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.res{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.res canvas{width:100%;display:block;background:#000}.res div{padding:6px 8px;font-size:12.5px;color:var(--muted)}
.res b{color:var(--fg)}.chip{display:inline-block;border-radius:9px;padding:0 7px;color:#fff;font-weight:600;font-size:11.5px}
.note{font-size:12.5px;color:var(--muted);margin-top:10px}
#status{font-size:13px;color:var(--muted);margin:6px 0 0;min-height:1.3em}
a{color:var(--acc)}
.banner{grid-column:1/-1;background:var(--card);border:2px solid var(--bad);border-radius:10px;padding:10px 14px;font-weight:600;color:var(--bad)}
.banner small{display:block;font-weight:400;color:var(--muted)}
.res.weak{opacity:.5}.more{grid-column:1/-1;font-size:13px;color:var(--muted)}.more button{padding:2px 8px;font-size:12.5px}
</style></head><body>
<header><h1><b>MIMIC</b> draw how it moved</h1>
<p class="sub">Draw a path on the canvas, or pick a sketch. The page matches it (DTW on 32-point motion descriptors) against every tracked object in the test clips.</p></header>
<main>
<section class="panel">
 <canvas id="draw" width="640" height="360"></canvas>
 <div class="row" id="presets"></div>
 <div class="row">
  <label>Backdrop / search in <select id="clip"></select></label>
  <label>Class <select id="cls"><option value="">any</option><option value="person">people</option><option value="veh">vehicles</option></select></label>
  <button id="clear">Clear</button>
 </div>
 <div class="row">
  <label><input type="checkbox" id="rot"> any orientation</label>
  <label>where it happened <input type="range" id="posw" min="0" max="1" step="0.05" value="0"></label>
  <label>size matters <input type="range" id="sizew" min="0" max="1" step="0.05" value="0"></label>
  <label><input type="checkbox" id="timing"> my pauses count</label>
 </div>
 <p id="status">Draw with the mouse or finger.</p>
 <p class="note">Matches are by motion only. Costs: green = strong, amber = plausible, red = weak (thresholds calibrated on hand-checked results). Footage is local test footage; thumbnails show the frame at the middle of the track with its camera-compensated trajectory.</p>
</section>
<section><div id="grid"></div></section>
</main>
<script>
__SEARCH_JS__
</script>
<script>
const RAW = __DATA__;
const D = prepare({n: RAW.n, pos: RAW.pos, speed: RAW.speed, ext: RAW.ext, len: RAW.len, aspect: RAW.aspect, rel: RAW.rel, straight: RAW.straight});
const cv = document.getElementById('draw'), cx = cv.getContext('2d');
const grid = document.getElementById('grid');
const VEH = new Set(['car', 'truck', 'bus', 'motorcycle', 'bicycle']);
let curStr = 0, curShape = 1, curTurn = 0.8, pts = [], times = [], drawing = false, current = null, bgImg = null, presetSpeed = null, curPreset = null, showWeak = false;
const imgCache = {};
function img(src) { if (!imgCache[src]) { const i = new Image(); i.src = src; i.onload = () => { paint(); }; imgCache[src] = i; } return imgCache[src]; }
const sel = document.getElementById('clip');
sel.innerHTML = '<option value="-1">all clips (no backdrop)</option>' + RAW.clips.map((c, i) => `<option value="${i}">${c.name} (${c.n} tracklets)</option>`).join('');
function aspect() { const v = +sel.value; return v >= 0 ? RAW.clips[v].aspect : 16 / 9; }
function sizeCanvas() { const a = aspect(); cv.height = Math.round(Math.min(480, 640 / a)); cv.width = Math.round(cv.height * a); if (cv.width > 640) { cv.width = 640; cv.height = Math.round(640 / a); } paint(); }
function paint() {
  cx.fillStyle = '#101418'; cx.fillRect(0, 0, cv.width, cv.height);
  const v = +sel.value;
  if (v >= 0) { const i = img(RAW.clips[v].bg); if (i.complete) { cx.globalAlpha = .55; cx.drawImage(i, 0, 0, cv.width, cv.height); cx.globalAlpha = 1; } }
  else { cx.strokeStyle = '#242c34'; cx.lineWidth = 1; for (let k = 1; k < 8; k++) { cx.beginPath(); cx.moveTo(cv.width * k / 8, 0); cx.lineTo(cv.width * k / 8, cv.height); cx.stroke(); } for (let k = 1; k < 5; k++) { cx.beginPath(); cx.moveTo(0, cv.height * k / 5); cx.lineTo(cv.width, cv.height * k / 5); cx.stroke(); } }
  if (pts.length > 1) { const P = pts.map(p => [p[0] * cv.width, p[1] * cv.height]); stroke(cx, P, 4); qMarks().forEach(m => mark(cx, m[0] * cv.width, m[1] * cv.height, m[2], 16, P)); }
}
function mark(c, x, y, label, r, P, fs) {   // pause ring / dwell halo: the temporal cue of the sketch
  c.save(); c.lineWidth = 3;
  if (label.startsWith('stays') && P) { let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9; P.forEach(p => { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); });
    c.strokeStyle = '#5ad7ff'; c.beginPath(); c.ellipse((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2 + 22, (y1 - y0) / 2 + 22, 0, 0, 7); c.stroke();
    c.fillStyle = '#5ad7ff'; c.font = 'bold 13px system-ui'; c.textAlign = 'center'; c.fillText(label, (x0 + x1) / 2, y1 + 40); }
  else { c.fillStyle = 'rgba(90,215,255,.35)'; c.strokeStyle = '#5ad7ff'; c.beginPath(); c.arc(x, y, r, 0, 7); c.fill(); c.stroke();
    c.fillStyle = '#5ad7ff'; c.font = 'bold ' + (fs || 13) + 'px system-ui'; c.textAlign = 'center'; c.strokeStyle = 'rgba(0,0,0,.8)'; c.lineWidth = 3; c.strokeText(label, x, y - r - 5); c.fillText(label, x, y - r - 5); }
  c.restore();
}
function qMarks() {   // [[x, y, label]] in canvas coordinates: a preset's own pauses, or the pauses made while drawing (if 'my pauses count')
  if (curPreset && RAW.presets[curPreset].marks.length) return RAW.presets[curPreset].marks;
  const out = []; if (!curPreset && times.length && document.getElementById('timing').checked) for (let i = 1; i < pts.length; i++) { const g = times[i] - times[i - 1]; if (g > 500) out.push([pts[i - 1][0], pts[i - 1][1], (g / 1000).toFixed(1) + ' s']); }
  return out;
}
function stroke(c, P, w) {
  c.lineJoin = 'round'; c.lineCap = 'round';
  c.strokeStyle = 'rgba(0,0,0,.75)'; c.lineWidth = w + 3; c.beginPath(); P.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke();
  c.strokeStyle = '#ffc83c'; c.lineWidth = w; c.beginPath(); P.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); c.stroke();
  const a = P[0], b = P[P.length - 1], k = P[Math.max(0, P.length - 6)];
  c.fillStyle = '#3cd070'; c.beginPath(); c.arc(a[0], a[1], w + 2, 0, 7); c.fill();
  const ang = Math.atan2(b[1] - k[1], b[0] - k[0]), hl = w * 4;
  c.fillStyle = '#ff4040'; c.beginPath(); c.moveTo(b[0] + Math.cos(ang) * hl * .6, b[1] + Math.sin(ang) * hl * .6);
  c.lineTo(b[0] + Math.cos(ang + 2.5) * hl, b[1] + Math.sin(ang + 2.5) * hl); c.lineTo(b[0] + Math.cos(ang - 2.5) * hl, b[1] + Math.sin(ang - 2.5) * hl); c.fill();
}
function xy(e) { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; }
cv.addEventListener('pointerdown', e => { cv.setPointerCapture(e.pointerId); drawing = true; pts = [xy(e)]; times = [performance.now()]; presetSpeed = null; curPreset = null; curShape = 1; curTurn = 0.8; curStr = 0; setPreset(null); paint(); });
cv.addEventListener('pointermove', e => { if (!drawing) return; const p = xy(e), l = pts[pts.length - 1]; if (Math.hypot((p[0] - l[0]) * cv.width, (p[1] - l[1]) * cv.height) > 2) { pts.push(p); times.push(performance.now()); paint(); } });
cv.addEventListener('pointerup', () => { drawing = false; if (pts.length > 3) run(); });
document.getElementById('clear').onclick = () => { pts = []; grid.innerHTML = ''; current = null; curPreset = null; setPreset(null); paint(); status('Draw with the mouse or finger.'); };
const pbox = document.getElementById('presets');
Object.keys(RAW.presets).forEach(p => { const b = document.createElement('button'); b.textContent = p.replace(/_/g, ' '); b.title = RAW.presets[p].desc; b.dataset.p = p; b.onclick = () => usePreset(p); pbox.appendChild(b); });
function setPreset(p) { [...pbox.children].forEach(b => b.classList.toggle('on', b.dataset.p === p)); }
function usePreset(p) {
  const P = RAW.presets[p]; pts = P.pts.map(a => a.slice()); times = []; presetSpeed = P.speed; curPreset = p; curShape = P.shape_w == null ? 1 : P.shape_w; curTurn = P.turn_w == null ? 0.8 : P.turn_w; curStr = P.straight_w || 0;
  document.getElementById('rot').checked = P.any_rot; document.getElementById('posw').value = P.pos_w; document.getElementById('sizew').value = P.size_w;
  setPreset(p); sel.value = -1; sizeCanvas(); run(p);
}
function status(s) { document.getElementById('status').textContent = s; }
['rot', 'posw', 'sizew', 'timing', 'cls'].forEach(id => document.getElementById(id).addEventListener('input', () => { if (pts.length > 3) { paint(); run(curPreset); } }));
sel.addEventListener('change', () => { sizeCanvas(); if (pts.length > 3) run(); });
function timeSpeed() {
  if (!times.length || pts.length < 6) return null; const n = pts.length, sp = new Float64Array(32);
  const T = times.map(t => (t - times[0]) / 1000), tot = T[n - 1] || 1; let j = 0;
  for (let k = 0; k < 32; k++) { const t = tot * k / 31; while (j < n - 2 && T[j + 1] < t) j++; const dt = Math.max(T[j + 1] - T[j], 1e-3); sp[k] = Math.hypot((pts[j + 1][0] - pts[j][0]) * aspect(), pts[j + 1][1] - pts[j][1]) / dt; }
  const sm = sp.map((v, i) => { let s = 0, c = 0; for (let d = -2; d <= 2; d++) { const q = sp[i + d]; if (q !== undefined) { s += q; c++; } } return s / c; });
  const mean = sm.reduce((a, b) => a + b, 0) / 32 || 1; return Float64Array.from(sm, v => v / mean);
}
function costChip(c) { const cal = RAW.cal || {}; if (cal.strong == null) return ['', 'var(--muted)']; return c <= cal.strong ? ['strong', 'var(--ok)'] : c <= cal.plausible ? ['plausible', 'var(--warn)'] : ['weak', 'var(--bad)']; }
function run(presetName) {
  if (pts.length < 4) return; const t0 = performance.now();
  const v = +sel.value, cls = document.getElementById('cls').value, tm = document.getElementById('timing').checked;
  let speed = presetSpeed, sw = presetName ? RAW.presets[presetName].speed_w : 0;
  if (!presetSpeed && tm) { speed = timeSpeed(); sw = speed ? 3 : 0; }   // same weight as the stop_and_go preset
  if (presetSpeed && !presetName) sw = 1;
  const q = {shape_w: curShape, turn_w: curTurn, straight_w: curStr, pts, aspect: aspect(), pos_w: +document.getElementById('posw').value, size_w: +document.getElementById('sizew').value,
    any_rot: document.getElementById('rot').checked, speed, speed_w: sw};
  const allow = i => (v < 0 || RAW.v[i] === v) && (cls === '' || (cls === 'veh' ? VEH.has(RAW.cls[i]) : RAW.cls[i] === cls));
  const res = search(D, q, allow, 8);
  const ms = performance.now() - t0;
  status(`${res.length} candidates ranked in ${ms.toFixed(0)} ms over ${RAW.n} tracklets${v >= 0 ? ' (one clip)' : ''}. Best cost ${res.length ? res[0].cost.toFixed(2) : '-'}`);
  grid.innerHTML = '';
  const cal = RAW.cal || {}, plaus = cal.plausible == null ? 1e9 : cal.plausible;
  const good = res.filter(r => r.cost <= plaus), weakR = res.filter(r => r.cost > plaus);
  if (res.length && !good.length) {
    const b = document.createElement('div'); b.className = 'banner';
    b.innerHTML = `NO MATCH: nothing ${v >= 0 ? 'in this clip' : 'in the footage'} moved like that (best cost ${res[0].cost.toFixed(2)}, match threshold ${plaus.toFixed(2)}).<small>The greyed-out cards are the closest candidates, not matches. Try another clip, a simpler sketch, or tick "any orientation".</small>`;
    grid.appendChild(b);
  }
  const order = good.length ? good.concat(showWeak ? weakR : []) : res.slice(0, 3);
  order.forEach((r, rank) => {
    const i = r.i, cl = RAW.clips[RAW.v[i]], el = document.createElement('div'); el.className = 'res' + (r.cost > plaus ? ' weak' : '');
    const c = document.createElement('canvas'); const hasImg = RAW.img[i] >= 0; const im = img(hasImg ? RAW.imgs[RAW.img[i]] : cl.bg);
    c.width = 336; c.height = Math.round(336 / cl.aspect); el.appendChild(c);
    const draw = () => { const k = c.getContext('2d'); if (im.complete && im.naturalWidth) { k.globalAlpha = hasImg ? 1 : .45; k.drawImage(im, 0, 0, c.width, c.height); k.globalAlpha = 1; }
      const P = []; for (let m = 0; m < 32; m++) P.push([D.pos[i * 64 + 2 * m] * c.width, D.pos[i * 64 + 2 * m + 1] * c.height]); stroke(k, P, 3);
      const mi = Math.min(31, Math.round(RAW.mid[i] * 31)); k.strokeStyle = '#fff'; k.lineWidth = 2; k.beginPath(); k.arc(P[mi][0], P[mi][1], 7, 0, 7); k.stroke();
      (RAW.pz[i] || []).slice(0, 2).forEach(z => mark(k, z[0] * c.width, z[1] * c.height, z[2].toFixed(z[2] >= 2 ? 0 : 1) + ' s', 9, null, 11));
      if (q.speed && q.speed_w > 0) spark(k, 6, c.height - 38, 100, 32, D.speed.subarray(i * 32, i * 32 + 32), q.speed); };
    if (im.complete) draw(); else im.addEventListener('load', draw);
    const [lab, col] = costChip(r.cost); const d2 = document.createElement('div');
    const clip = RAW.clip[i] ? ` &middot; <a href="${RAW.clip[i]}" target="_blank">clip</a>` : '';
    const rot = (q.any_rot && r.rot) ? ` <span class="chip" style="background:var(--muted)" title="The sketch matched this track after turning the sketch by ${r.rot} degrees (any orientation is on)">sketch turned ${r.rot}&deg;</span>` : '';
    d2.innerHTML = `<b>#${rank + 1} ${RAW.cls[i]}</b> ${cl.name} ${fmt(RAW.t0[i])}-${fmt(RAW.t1[i])} ${lab ? `<span class="chip" style="background:${col}">${lab}</span>` : ''}${rot}<br>cost ${r.cost.toFixed(2)} (shape ${r.cs.toFixed(3)}, turn ${r.ct.toFixed(2)})${hasImg ? '' : ' &middot; glyph only'}${clip}`;
    el.appendChild(d2); grid.appendChild(el);
  });
  if (good.length && weakR.length) {
    const m = document.createElement('div'); m.className = 'more';
    m.innerHTML = `${weakR.length} weaker candidate${weakR.length > 1 ? 's' : ''} ${showWeak ? 'shown greyed out' : 'hidden'} (cost above ${plaus.toFixed(2)}, not a match) `;
    const bt = document.createElement('button'); bt.textContent = showWeak ? 'hide them' : 'show them'; bt.onclick = () => { showWeak = !showWeak; run(curPreset); }; m.appendChild(bt); grid.appendChild(m);
  }
}
function spark(k, x, y, w, h, vals, ref) {   // speed over time (speed / mean speed): a stop is a dip to zero; grey = what you asked for
  k.save(); k.fillStyle = 'rgba(10,12,15,.72)'; k.fillRect(x, y, w, h); k.strokeStyle = '#5b6672'; k.strokeRect(x + .5, y + .5, w - 1, h - 1);
  const line = (v, col, lw) => { k.strokeStyle = col; k.lineWidth = lw; k.beginPath(); for (let m = 0; m < 32; m++) { const yy = y + h - 4 - (h - 12) * Math.min(3, Math.max(0, v[m])) / 3, xx = x + 3 + (w - 6) * m / 31; m ? k.lineTo(xx, yy) : k.moveTo(xx, yy); } k.stroke(); };
  line(ref, '#9aa5b1', 1); line(vals, '#ffc83c', 2); k.fillStyle = '#9aa5b1'; k.font = '9px system-ui'; k.fillText('speed vs sketch', x + 3, y + 9); k.restore();
}
function fmt(t) { const m = Math.floor(t / 60), s = (t % 60).toFixed(0).padStart(2, '0'); return m + ':' + s; }
sizeCanvas(); usePreset('u_turn');
</script></body></html>
"""
