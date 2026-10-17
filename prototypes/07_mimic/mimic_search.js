// MIMIC search: a line-for-line port of mimic_core.match() (descriptor kNN shortlist + DTW re-rank). Pure functions, no DOM.
// Data layout (built by build_page.py): D.n tracklets; D.pos Float64Array(n*64), D.speed (n*32), D.ext (n), D.len (n),
// D.rel (n) extent in body heights, D.aspect (n) isotropic aspect of the tracklet's clip; derived at load: D.shape (n*64), D.turn (n*32).
const MN = 32;
const W_TURN = 0.35, W_SHAPE_D = 0.10, W_TURN_D = 0.35, MIN_REL = 0.5, MIN_EXT = 0.03;

function resampleArc(p, n) {
  n = n || MN;
  const m = p.length, d = new Float64Array(m);
  for (let i = 1; i < m; i++) d[i] = d[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]);
  const out = [];
  if (d[m - 1] < 1e-9) { for (let i = 0; i < n; i++) out.push([p[0][0], p[0][1]]); return out; }
  let j = 0;
  for (let k = 0; k < n; k++) {
    const u = d[m - 1] * k / (n - 1);
    while (j < m - 2 && d[j + 1] < u) j++;
    const a = (u - d[j]) / Math.max(d[j + 1] - d[j], 1e-12);
    const t = Math.min(1, Math.max(0, a));
    out.push([p[j][0] + (p[j + 1][0] - p[j][0]) * t, p[j][1] + (p[j + 1][1] - p[j][1]) * t]);
  }
  return out;
}
function shapeNorm(pos, aspect) {           // pos: [[x,y]*32] frame-normalised -> start at origin, unit length (isotropic)
  const q = pos.map(p => [p[0] * aspect, p[1]]);
  let L = 0;
  for (let i = 1; i < q.length; i++) L += Math.hypot(q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1]);
  L = Math.max(L, 1e-9);
  return q.map(p => [(p[0] - q[0][0]) / L, (p[1] - q[0][1]) / L]);
}
function turnCurve(shp) {
  const v = [];
  let L = 0;
  for (let i = 1; i < shp.length; i++) { const dx = shp[i][0] - shp[i - 1][0], dy = shp[i][1] - shp[i - 1][1]; v.push([dx, dy]); L += Math.hypot(dx, dy); }
  if (L < 1e-6) return new Float64Array(MN);
  const h = new Float64Array(v.length);
  let prev = 0;
  for (let i = 0; i < v.length; i++) {
    let a = Math.atan2(v[i][1], v[i][0]);
    if (i > 0) { let d = a - prev; while (d > Math.PI) { a -= 2 * Math.PI; d = a - prev; } while (d < -Math.PI) { a += 2 * Math.PI; d = a - prev; } }
    h[i] = a; prev = a;
  }
  const h0 = h[0], out = new Float64Array(MN);
  for (let k = 0; k < MN; k++) {
    const x = k * (h.length - 1) / (MN - 1), i0 = Math.floor(x), i1 = Math.min(h.length - 1, i0 + 1), f = x - i0;
    out[k] = ((h[i0] - h0) * (1 - f) + (h[i1] - h0) * f) / Math.PI;
  }
  return out;
}
function rotPts(shp, a) { const c = Math.cos(a), s = Math.sin(a); return shp.map(p => [p[0] * c - p[1] * s, p[0] * s + p[1] * c]); }

// DTW with Sakoe-Chiba band; a, b: flat Float64Array with dim d; returns mean per-step cost along the optimal path.
function dtwFlat(a, an, b, bn, d, band) {
  const W = bn + 1, D = new Float64Array((an + 1) * W).fill(Infinity), Ln = new Float64Array((an + 1) * W);
  D[0] = 0;
  for (let i = 1; i <= an; i++) {
    const jlo = Math.max(1, i - band), jhi = Math.min(bn, i + band);
    for (let j = jlo; j <= jhi; j++) {
      let c = 0;
      for (let k = 0; k < d; k++) { const t = a[(i - 1) * d + k] - b[(j - 1) * d + k]; c += t * t; }
      c = Math.sqrt(c);
      const d0 = D[(i - 1) * W + j - 1], d1 = D[(i - 1) * W + j], d2 = D[i * W + j - 1];
      let best = d0, bl = Ln[(i - 1) * W + j - 1];
      if (d1 < best) { best = d1; bl = Ln[(i - 1) * W + j]; }
      if (d2 < best) { best = d2; bl = Ln[i * W + j - 1]; }
      D[i * W + j] = c + best; Ln[i * W + j] = bl + 1;
    }
  }
  return D[an * W + bn] / Math.max(Ln[an * W + bn], 1);
}
const flat2 = pts => { const o = new Float64Array(pts.length * 2); pts.forEach((p, i) => { o[2 * i] = p[0]; o[2 * i + 1] = p[1]; }); return o; };

function prepare(raw) {                      // raw: {n, pos:[n*64], speed:[n*32], ext, len, aspect}
  const n = raw.n, D = { n, straight: Float64Array.from(raw.straight), rel: Float64Array.from(raw.rel), pos: Float64Array.from(raw.pos), speed: Float64Array.from(raw.speed), ext: Float64Array.from(raw.ext),
    len: Float64Array.from(raw.len), aspect: Float64Array.from(raw.aspect), shape: new Float64Array(n * 64), turn: new Float64Array(n * 32) };
  for (let i = 0; i < n; i++) {
    const pos = []; for (let k = 0; k < MN; k++) pos.push([D.pos[i * 64 + 2 * k], D.pos[i * 64 + 2 * k + 1]]);
    const sh = shapeNorm(pos, D.aspect[i]);
    for (let k = 0; k < MN; k++) { D.shape[i * 64 + 2 * k] = sh[k][0]; D.shape[i * 64 + 2 * k + 1] = sh[k][1]; }
    D.turn.set(turnCurve(sh), i * 32);
  }
  return D;
}

// q: {pts:[[x,y]..] canvas-normalised, aspect, pos_w, size_w, any_rot, speed (Float64Array(32)|null), speed_w, shape_w, turn_w}
function search(D, q, allow, k, shortlist) {
  k = k || 8; shortlist = shortlist || 250;
  const posQ = resampleArc(q.pts), shpQ = shapeNorm(posQ, q.aspect), tcQ = turnCurve(shpQ);
  let lo = [1e9, 1e9], hi = [-1e9, -1e9];
  q.pts.forEach(p => { const x = p[0] * q.aspect, y = p[1]; lo = [Math.min(lo[0], x), Math.min(lo[1], y)]; hi = [Math.max(hi[0], x), Math.max(hi[1], y)]; });
  const extQ = Math.hypot(hi[0] - lo[0], hi[1] - lo[1]);
  let qlen = 0; for (let i = 1; i < q.pts.length; i++) qlen += Math.hypot((q.pts[i][0] - q.pts[i - 1][0]) * q.aspect, q.pts[i][1] - q.pts[i - 1][1]);
  const strQ = Math.hypot((q.pts[q.pts.length - 1][0] - q.pts[0][0]) * q.aspect, q.pts[q.pts.length - 1][1] - q.pts[0][1]) / Math.max(qlen, 1e-9);
  const stw = q.straight_w || 0;
  const rots = []; if (q.any_rot) for (let a = 0; a < 360; a += 30) rots.push(a * Math.PI / 180); else rots.push(0);
  const qs = rots.map(a => rotPts(shpQ, a));
  const centred = pts => { let mx = 0, my = 0; pts.forEach(p => { mx += p[0]; my += p[1]; }); mx /= pts.length; my /= pts.length; return pts.map(p => [p[0] - mx, p[1] - my]); };
  const posQc = centred(posQ);
  const qvs = qs.map(sq => {
    const v = []; centred(sq).forEach(p => v.push(p[0], p[1]));
    posQc.forEach(p => v.push(p[0] * q.pos_w * 2, p[1] * q.pos_w * 2));
    for (let k2 = 0; k2 < MN; k2++) v.push(tcQ[k2] * W_TURN);
    let nn = 0; v.forEach(t => nn += t * t); nn = Math.max(Math.sqrt(nn), 1e-9);
    return v.map(t => t / nn);
  });
  const s1 = new Float64Array(D.n).fill(-9);
  const tv = new Float64Array(64 + 64 + 32);
  for (let i = 0; i < D.n; i++) {
    if (!allow(i) || D.rel[i] < MIN_REL || D.ext[i] < MIN_EXT) continue;
    let mx = 0, my = 0, px = 0, py = 0;
    for (let k2 = 0; k2 < MN; k2++) { mx += D.shape[i * 64 + 2 * k2]; my += D.shape[i * 64 + 2 * k2 + 1]; px += D.pos[i * 64 + 2 * k2]; py += D.pos[i * 64 + 2 * k2 + 1]; }
    mx /= MN; my /= MN; px /= MN; py /= MN;
    let nn = 0;
    for (let k2 = 0; k2 < MN; k2++) {
      tv[2 * k2] = D.shape[i * 64 + 2 * k2] - mx; tv[2 * k2 + 1] = D.shape[i * 64 + 2 * k2 + 1] - my;
      tv[64 + 2 * k2] = (D.pos[i * 64 + 2 * k2] - px) * q.pos_w * 2; tv[64 + 2 * k2 + 1] = (D.pos[i * 64 + 2 * k2 + 1] - py) * q.pos_w * 2;
      tv[128 + k2] = D.turn[i * 32 + k2] * W_TURN;
    }
    for (let t = 0; t < 160; t++) nn += tv[t] * tv[t];
    nn = Math.max(Math.sqrt(nn), 1e-9);
    let best = -9;
    for (const qv of qvs) { let s = 0; for (let t = 0; t < 160; t++) s += tv[t] * qv[t]; best = Math.max(best, s / nn); }
    let sc = best;
    if (q.size_w > 0) sc -= q.size_w * 0.5 * Math.abs(Math.log(Math.max(D.ext[i], 1e-3) / Math.max(extQ, 1e-3)));
    if (q.speed && q.speed_w > 0) {
      let a = 0; for (let k2 = 0; k2 < MN; k2++) a += Math.abs(Math.min(3, Math.max(0, D.speed[i * 32 + k2])) - Math.min(3, Math.max(0, q.speed[k2])));
      sc -= q.speed_w * 0.5 * a / MN;
    }
    if (stw > 0) sc -= stw * 0.5 * Math.abs(D.straight[i] - strQ);
    s1[i] = sc;
  }
  const idx = Array.from({ length: D.n }, (_, i) => i).filter(i => s1[i] > -8).sort((a, b) => s1[b] - s1[a]).slice(0, shortlist);
  const qflat = qs.map(flat2), tcFlat = Float64Array.from(tcQ), posFlat = flat2(posQ);
  const spQ = q.speed ? Float64Array.from(q.speed, v => Math.min(3, Math.max(0, v))) : null;
  const shw = q.shape_w == null ? 1 : q.shape_w, tuw = q.turn_w == null ? 0.8 : q.turn_w;
  const wsum = shw + tuw + q.pos_w * 2 + q.speed_w + q.size_w + stw;
  const res = idx.map(i => {
    const sh = D.shape.subarray(i * 64, i * 64 + 64);
    let cs = Infinity, rot = 0; for (let ri = 0; ri < qflat.length; ri++) { const c0 = dtwFlat(qflat[ri], MN, sh, MN, 2, 6); if (c0 < cs) { cs = c0; rot = Math.round(rots[ri] * 180 / Math.PI); } }
    const ct = dtwFlat(tcFlat, MN, D.turn.subarray(i * 32, i * 32 + 32), MN, 1, 4);
    const cp = q.pos_w > 0 ? dtwFlat(posFlat, MN, D.pos.subarray(i * 64, i * 64 + 64), MN, 2, 6) : 0;
    const csp = (spQ && q.speed_w > 0) ? dtwFlat(spQ, MN, Float64Array.from(D.speed.subarray(i * 32, i * 32 + 32), v => Math.min(3, Math.max(0, v))), MN, 1, 4) : 0;
    const csz = q.size_w > 0 ? Math.abs(Math.log(Math.max(D.ext[i], 1e-3) / Math.max(extQ, 1e-3))) : 0;
    const cstr = stw > 0 ? Math.abs(D.straight[i] - strQ) : 0;
    const cost = (shw * cs / W_SHAPE_D + tuw * ct / W_TURN_D + stw * cstr / 0.5 + q.pos_w * 2 * cp / 0.15 + q.speed_w * csp / 0.8 + q.size_w * csz / 0.7) / wsum;
    return { i, cost, cs, ct, cp, csp, csz, cstr, rot };
  });
  res.sort((a, b) => a.cost - b.cost);
  return res.slice(0, k);
}

if (typeof module !== 'undefined') module.exports = { prepare, search, resampleArc, shapeNorm, turnCurve, MN };
