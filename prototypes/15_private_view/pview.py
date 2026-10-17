"""PRIVATE VIEW core: privacy-mask drift monitor.

Static privacy masks (set up on a reference frame) vs where the windows REALLY are now.
  H(t)  = camera motion reference -> current frame (feature matching + RANSAC homography, pview_motion)
  true window position = H(t) . zone polygon        static mask = reference polygon (+ safety margin)
  exposed % = share of the (visible) window area that lies outside the static mask.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402
from pview_motion import MotionTracker  # noqa: E402

HERE = Path(__file__).resolve().parent
RS = 0.5  # raster scale for exposure maths (half resolution is plenty: sub-pixel error < 1 px)

# ------------------------------------------------------------------------------------------- zones


@dataclass
class Zone:
    id: str
    name: str
    poly: np.ndarray            # Nx2 float, reference-frame pixels
    stand_in: bool = False


def load_zones(video: str) -> tuple[dict[str, Any], list[Zone]]:
    d = json.loads((HERE / "zones" / f"{video}.json").read_text(encoding="utf-8"))
    zs = [Zone(z["id"], z.get("name", z["id"]), np.float32(z["poly"]), bool(z.get("stand_in", False))) for z in d["zones"]]
    return d, zs


def poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _fill(poly: np.ndarray, shape: tuple[int, int], rs: float = RS) -> np.ndarray:
    h, w = shape
    m = np.zeros((int(h * rs), int(w * rs)), np.uint8)
    cv2.fillPoly(m, [np.round(poly * rs * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m


def static_masks(zones: list[Zone], shape: tuple[int, int], margin_px: float) -> list[np.ndarray]:
    """Raster of each static privacy mask: reference polygon dilated by the safety margin."""
    out = []
    r = max(0, int(round(margin_px * RS)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) if r > 0 else None
    for z in zones:
        m = _fill(z.poly, shape)
        out.append(cv2.dilate(m, k) if k is not None else m)
    return out


def warp_poly(poly: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float32), H.astype(np.float64)).reshape(-1, 2)


def evaluate(zones: list[Zone], masks: list[np.ndarray], H: np.ndarray | None, shape: tuple[int, int],
             min_visible: float = 0.25) -> list[dict[str, Any]]:
    """Per zone: where the window is now, how much of it is exposed. H maps reference -> current (None = lost)."""
    h, w = shape
    out = []
    for z, m in zip(zones, masks):
        if H is None:
            out.append({"id": z.id, "state": "unverified", "exposed": float("nan"), "vis": float("nan"),
                        "mis": float("nan"), "poly": None})
            continue
        p = warp_poly(z.poly, H)
        full = poly_area(p) * RS * RS
        if full < 4:
            out.append({"id": z.id, "state": "offframe", "exposed": float("nan"), "vis": 0.0, "mis": float("nan"), "poly": p})
            continue
        win = _fill(p, shape)            # raster is implicitly clipped to the frame
        vis = int(win.sum())
        mis = float(np.mean(np.linalg.norm(p - z.poly, axis=1)))
        if vis < min_visible * full or vis < 6:
            out.append({"id": z.id, "state": "offframe", "exposed": float("nan"), "vis": vis / full, "mis": mis, "poly": p})
            continue
        exp = int((win & (1 - m)).sum())
        out.append({"id": z.id, "state": "ok", "exposed": 100.0 * exp / vis, "vis": vis / full, "mis": mis, "poly": p})
    return out


# ------------------------------------------------------------------------------ simulated injections


@dataclass
class Injection:
    """SIMULATED camera knock / shake, applied to the real frames. The matrix is the ground truth."""
    kind: str                    # 'step' | 'shake'
    t0: float
    t1: float
    dx: float = 0.0
    dy: float = 0.0
    rot: float = 0.0             # degrees, about the frame centre
    amp: float = 0.0             # shake amplitude px
    hz: float = 2.0
    label: str = "SIMULATED"

    def matrix(self, t: float, w: int, h: int) -> np.ndarray:
        if not (self.t0 <= t < self.t1):
            return np.eye(3)
        dx, dy, rot = self.dx, self.dy, self.rot
        if self.kind == "shake":
            u = t - self.t0
            env = math.exp(-1.2 * u)                                 # decaying wobble
            dx = self.amp * env * math.sin(2 * math.pi * self.hz * u)
            dy = 0.6 * self.amp * env * math.sin(2 * math.pi * self.hz * u + 1.1)
            rot = 0.0
        c = (w / 2, h / 2)
        R = cv2.getRotationMatrix2D(c, rot, 1.0)
        M = np.vstack([R, [0, 0, 1]])
        M[0, 2] += dx
        M[1, 2] += dy
        return M


def inject(frame: np.ndarray, M: np.ndarray) -> np.ndarray:
    if np.allclose(M, np.eye(3)):
        return frame
    return cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)


# ------------------------------------------------------------------------------------------- events


def runs(flags: list[bool]) -> list[tuple[int, int]]:
    out, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        if not f and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(flags) - 1))
    return out


def make_events(ts: list[float], dt: float, series: dict[str, list[float]], lost: list[bool], thr: float,
                min_dur: float, merge_gap: float, mis: dict[str, list[float]]) -> list[dict[str, Any]]:
    """Exposure log. Per zone: runs with exposed% >= thr. Whole camera: UNVERIFIED runs (tracker lost)."""
    evs: list[dict[str, Any]] = []
    for zid, vals in series.items():
        flags = [(not math.isnan(v)) and v >= thr for v in vals]
        rr = runs(flags)
        merged: list[list[int]] = []
        for a, b in rr:
            if merged and ts[a] - (ts[merged[-1][1]] + dt) <= merge_gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        for a, b in merged:
            dur = ts[b] + dt - ts[a]
            if dur < min_dur:
                continue
            seg = [v for v in vals[a:b + 1] if not math.isnan(v)]
            ms = [v for v in mis[zid][a:b + 1] if not math.isnan(v)]
            evs.append({"zone": zid, "kind": "EXPOSED", "start_s": round(ts[a], 2), "end_s": round(ts[b] + dt, 2),
                        "duration_s": round(dur, 2), "max_exposed_pct": round(max(seg), 1),
                        "mean_exposed_pct": round(float(np.mean(seg)), 1), "max_misalign_px": round(max(ms), 1) if ms else None,
                        "frames": b - a + 1})
    for a, b in runs(lost):
        dur = ts[b] + dt - ts[a]
        if dur < min_dur:
            continue
        evs.append({"zone": "ALL", "kind": "UNVERIFIED", "start_s": round(ts[a], 2), "end_s": round(ts[b] + dt, 2),
                    "duration_s": round(dur, 2), "max_exposed_pct": None, "mean_exposed_pct": None,
                    "max_misalign_px": None, "frames": b - a + 1})
    evs.sort(key=lambda e: (e["start_s"], e["zone"]))
    return evs


# ------------------------------------------------------------------------------------------ drawing

_FONTS: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    k = ("b" if bold else "r", size)
    if k not in _FONTS:
        for p in (("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
                  ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")):
            try:
                _FONTS[k] = ImageFont.truetype(p, size)
                break
            except OSError:
                continue
        else:
            _FONTS[k] = ImageFont.load_default()
    return _FONTS[k]


def ptext(img: np.ndarray, items: list[tuple[str, tuple[int, int], int, tuple[int, int, int], bool]]) -> np.ndarray:
    """Draw several texts with PIL (nicer than Hershey). items: (text, (x, y), size, rgb, bold)."""
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    for text, xy, size, rgb, bold in items:
        d.text(xy, text, font=font(size, bold), fill=rgb)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def tw(text: str, size: int, bold: bool = False) -> int:
    return int(font(size, bold).getlength(text))


GREEN = (80, 220, 90)     # BGR
RED = (60, 60, 235)
AMBER = (0, 190, 255)
YELLOW = (40, 235, 255)
GREY = (120, 120, 120)


def blend_poly(img: np.ndarray, poly: np.ndarray, color, alpha: float) -> None:
    ov = img.copy()
    cv2.fillPoly(ov, [np.round(poly).astype(np.int32)], color, lineType=cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def dilate_poly_pts(poly: np.ndarray, margin: float) -> np.ndarray:
    """Approximate dilated polygon outline for drawing (convex quads: push vertices out from the centroid)."""
    c = poly.mean(axis=0)
    v = poly - c
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
    return poly + v / n * margin * 1.1


def _mask_poly(z: Zone, margin: float) -> np.ndarray:
    """Axis-aligned dilation of the static polygon for drawing (matches the raster dilation closely)."""
    p = z.poly
    x0, y0 = p.min(axis=0) - margin
    x1, y1 = p.max(axis=0) + margin
    if len(p) == 4 and np.allclose(p[0, 1], p[1, 1]) and np.allclose(p[1, 0], p[2, 0]):
        return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return dilate_poly_pts(p, margin)


def strip_color(v: float, state: str, thr: float):
    """BGR colour for an exposure-strip cell."""
    if state == "unverified":
        return (150, 150, 150)
    if state == "offframe":
        return (95, 95, 95)
    if v < thr:
        return (90, 190, 70)
    # 5 .. 100 % : amber -> deep red
    f = min(1.0, (v - thr) / (100 - thr))
    return (int(60 - 20 * f), int(190 - 150 * f), int(250 - 25 * f))


def draw_strip(width: int, row_h: int, ids: list[str], ts: list[float], series: dict[str, list[float]], states: dict[str, list[str]],
               duration: float, t_cursor: float | None, thr: float, t0: float = 0.0, label_w: int = 52,
               tick_every: float | None = None, bg=(26, 26, 26), events: list[dict] | None = None) -> np.ndarray:
    """Exposure strip: one row per zone (green = covered, amber..red = exposed %, light grey = tracker lost)."""
    n = len(ids)
    gh = 18
    img = np.full((n * row_h + gh + 6, width, 3), bg, np.uint8)
    x0, x1 = label_w, width - 8
    span = max(1e-6, duration - t0)
    for r, zid in enumerate(ids):
        y = r * row_h
        cv2.rectangle(img, (x0, y + 2), (x1, y + row_h - 2), (45, 45, 45), -1)
        vals, sts = series[zid], states[zid]
        for k, t in enumerate(ts):
            xa = x0 + int((t - t0) / span * (x1 - x0))
            xb = x0 + int((t + (ts[1] - ts[0] if len(ts) > 1 else 0.1) - t0) / span * (x1 - x0)) + 1
            cv2.rectangle(img, (xa, y + 2), (max(xa + 1, xb), y + row_h - 2), strip_color(vals[k], sts[k], thr), -1)
    te = tick_every or max(1.0, round(span / 8))
    k = 0.0
    while t0 + k <= duration + 1e-6:
        x = x0 + int(k / span * (x1 - x0))
        cv2.line(img, (x, n * row_h), (x, n * row_h + 4), (180, 180, 180), 1)
        cv2.putText(img, pk.mmss(t0 + k)[:5] if te >= 1 else pk.mmss(t0 + k), (max(0, x - 14), n * row_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
        k += te
    if t_cursor is not None:
        x = x0 + int((t_cursor - t0) / span * (x1 - x0))
        cv2.line(img, (x, 0), (x, n * row_h), (255, 255, 255), 2)
    items = [(zid, (6, r * row_h + max(0, (row_h - 16) // 2) - 2), 13, (230, 230, 230), True) for r, zid in enumerate(ids)]
    return ptext(img, items)


# ------------------------------------------------------------------------------------------ pipeline


def default_cfg(video: str) -> dict[str, Any]:
    return {"video": video}


def process(video: str, *, t_start: float, t_end: float, ref_t: float, zones_file: str | None = None,
            injection: Injection | None = None, sample_hz: float = 10.0, margin_frac: float = 0.006,
            thr: float = 5.0, min_dur: float = 0.3, merge_gap: float = 0.5, feat: str = "sift",
            out_dir: Path | None = None, overlay: bool = True, overlay_speed: float = 1.0, overlay_w: int = 960,
            progress: bool = False, stabilise: bool = False) -> dict[str, Any]:
    """Run the monitor on one clip segment. Returns everything needed for report/hero/results."""
    meta = pk.info(video)
    fps, W, Hh = meta["fps"], meta["w"], meta["h"]
    t_end = min(t_end, meta["duration"])
    stride = max(1, int(round(fps / sample_hz)))
    dt = stride / fps
    zinfo, zones = load_zones(zones_file or video)
    margin = max(3.0, margin_frac * W)
    masks = static_masks(zones, (Hh, W), margin)
    ref_frame = pk.frame_at(video, ref_t)
    tracker = MotionTracker(ref_frame, feat)
    ids = [z.id for z in zones]
    ts: list[float] = []
    series = {z: [] for z in ids}
    truth = {z: [] for z in ids}
    mis = {z: [] for z in ids}
    states = {z: [] for z in ids}
    lost: list[bool] = []
    track_rows: list[dict[str, Any]] = []
    Hs: list[Any] = []
    peak = {"score": -1.0, "t": None, "frame": None, "res": None}
    keep_frames: dict[int, Any] = {}
    writer = None
    spd = overlay_speed
    odir = Path(out_dir) if out_dir else None
    ov_tmp = None
    if overlay and odir:
        odir.mkdir(parents=True, exist_ok=True)
        ov_tmp = odir / "overlay.mp4"
        writer = pk.Mp4Writer(ov_tmp, sample_hz * spd)
    # optional pass 1: estimate the REAL (hand-held) motion; the main pass then stabilises the clip to the reference so
    # that the SIMULATED injection is the only motion and the ground truth is exactly the injected matrix
    stabH: list[np.ndarray] = []
    real_pass = None
    if stabilise:
        tr1 = MotionTracker(ref_frame, feat)
        ts1, lost1 = [], []
        ser1 = {z: [] for z in ids}
        mis1 = {z: [] for z in ids}
        for i, t, fr0 in pk.iter_frames(video, stride=stride, start=t_start, end=t_end):
            r1 = tr1.update(fr0)
            lost1.append(r1["H"] is None)
            stabH.append(r1["H"] if r1["H"] is not None else (stabH[-1] if stabH else np.eye(3)))
            ts1.append(round(t, 3))
            for z, rr in zip(ids, evaluate(zones, masks, r1["H"], (Hh, W))):
                ser1[z].append(rr["exposed"]); mis1[z].append(rr["mis"])
            if progress and len(ts1) % 100 == 0:
                print(f"  .. pass1 {pk.mmss(t)}", flush=True)
        real_pass = {"events": make_events(ts1, dt, ser1, lost1, thr, min_dur, merge_gap, mis1),
                     "max_exposed_pct": max([v for z in ids for v in ser1[z] if not math.isnan(v)] or [0.0]),
                     "max_misalign_px": max([v for z in ids for v in mis1[z] if not math.isnan(v)] or [0.0]),
                     "lost_samples": int(sum(lost1)), "ts": ts1, "series": ser1}
    cur = {"ts": ts, "series": series, "states": states, "lost": lost, "ids": ids, "thr": thr, "dt": dt,
           "frame_size": (W, Hh), "zones": zones, "t_total": (t_start, t_end)}
    ow = overlay_w if W > overlay_w else W
    sc = ow / W
    n = 0
    snaps: list[tuple[float, np.ndarray, list[dict[str, Any]]]] = []
    snap_dt = min(3.0, (t_end - t_start) / 12.0)
    next_snap = t_start
    for i, t, fr0 in pk.iter_frames(video, stride=stride, start=t_start, end=t_end):
        if stabilise and n < len(stabH):
            fr0 = cv2.warpPerspective(fr0, stabH[n], (W, Hh), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REFLECT101)
        M = injection.matrix(t, W, Hh) if injection else np.eye(3)
        fr = inject(fr0, M) if injection else fr0
        r = tracker.update(fr)
        H = r["H"]
        res = evaluate(zones, masks, H, (Hh, W))
        ts.append(round(t, 3))
        lost.append(H is None)
        Hs.append(H)
        for z, rr in zip(ids, res):
            series[z].append(rr["exposed"])
            states[z].append(rr["state"])
            mis[z].append(rr["mis"])
        if injection:
            tres = evaluate(zones, masks, M, (Hh, W))
            for z, rr in zip(ids, tres):
                truth[z].append(rr["exposed"])
        track_rows.append({"t": round(t, 2), "inliers": r["inliers"], "src": r["src"]})
        score = sum(v for v in (rr["exposed"] for rr in res) if not math.isnan(v))
        if score > peak["score"]:
            peak = {"score": score, "t": t, "frame": fr.copy(), "res": res}
        if t >= next_snap - 1e-6:
            snaps.append((t, cv2.resize(fr, (640, int(Hh * 640 / W)), interpolation=cv2.INTER_AREA), res))
            next_snap += snap_dt
        if writer is not None:
            smf = cv2.resize(fr, (ow, int(Hh * ow / W)), interpolation=cv2.INTER_AREA) if W > ow else fr
            writer.write(render_overlay_frame(cur, smf, res, t, len(ts) - 1, sc, margin))
        n += 1
        if progress and n % 100 == 0:
            print(f"  .. {pk.mmss(t)} ({n} samples)", flush=True)
    if writer is not None:
        writer.__exit__()
    events = make_events(ts, dt, series, lost, thr, min_dur, merge_gap, mis)
    out: dict[str, Any] = {
        "video": video, "t_start": t_start, "t_end": t_end, "ref_t": ref_t, "sample_hz": sample_hz, "stride": stride, "dt": dt,
        "margin_px": margin, "thr": thr, "ids": ids, "ts": ts, "series": series, "truth": truth if injection else None,
        "mis": mis, "states": states, "lost": lost, "events": events, "tracker": track_rows, "zones": zones, "zinfo": zinfo,
        "frame_size": (W, Hh), "injection": injection, "ref_frame": ref_frame, "peak": peak, "Hs": Hs,
        "real_pass": real_pass, "snaps": snaps, "stabilised": stabilise, "feat": feat, "tracker_stats": tracker.stats,
    }
    if writer is not None:
        out["overlay_path"] = str(ov_tmp)
    return out


def banner_state(out: dict[str, Any], k: int) -> tuple[str, tuple[int, int, int]]:
    """Headline for sample k: worst zone right now."""
    thr = out["thr"]
    if out["lost"][k]:
        return "VIEW NOT RECOGNISED  masks cannot be verified", (110, 110, 110)
    worst, wz = 0.0, None
    nex = 0
    for z in out["ids"]:
        v = out["series"][z][k]
        if not math.isnan(v) and v >= thr:
            nex += 1
        if not math.isnan(v) and v > worst:
            worst, wz = v, z
    if nex == 0:
        return "ALL MASKS ALIGNED", (60, 150, 50)
    return f"BREACH  {nex} window{'s' if nex > 1 else ''} exposed  (worst {wz} {worst:.0f}%)", (40, 40, 215)


def render_overlay_frame(out: dict[str, Any], frame: np.ndarray, res: list[dict[str, Any]], t: float, k: int, sc: float,
                         margin: float) -> np.ndarray:
    W, Hh = out["frame_size"]
    zones = out["zones"]
    # draw at reduced size: scale polygons
    scene = _draw_scaled(frame, zones, res, margin, sc, (Hh, W), out["thr"])
    text, col = banner_state(out, k)
    ow = scene.shape[1]
    bar = np.full((34, ow, 3), col, np.uint8)
    top = ptext(bar, [(text, (10, 3), 20, (255, 255, 255), True), (pk.mmss(t), (ow - 80, 5), 16, (255, 255, 255), False)])
    # legend + progressive strip
    kk = k + 1
    ts = out["ts"][:kk]
    series = {z: out["series"][z][:kk] for z in out["ids"]}
    states = {z: out["states"][z][:kk] for z in out["ids"]}
    t0_, t1_ = out["t_total"]
    strip = draw_strip(ow, 14 if len(out["ids"]) > 5 else 18, out["ids"], ts, series, states, t1_, t,
                       out["thr"], t0=t0_)
    leg = np.full((22, ow, 3), (26, 26, 26), np.uint8)
    legtxt = ("red = static mask    green = true window position    yellow = exposed    grey = view not recognised" if ow < 800 else
              "red = static privacy mask      green = where the window really is      yellow = exposed      grey = view not recognised")
    leg = ptext(leg, [(legtxt, (6, 2), 12 if ow < 800 else 13, (200, 200, 200), False)])
    return np.vstack([top, scene, leg, strip])


def _draw_scaled(frame_small: np.ndarray, zones: list[Zone], res: list[dict[str, Any]], margin: float, sc: float,
                 full_shape: tuple[int, int], thr: float) -> np.ndarray:
    """draw_scene on an already-downscaled frame: compute exposure raster at full-res geometry, draw at scale."""
    h, w = full_shape
    # reuse draw_scene by passing a small frame and treating polygons in full-res coords scaled by sc
    img = frame_small.copy()
    masks = static_masks(zones, (h, w), margin)
    th = 2
    for z in zones:
        mp = _mask_poly(z, margin) * sc
        blend_poly(img, mp, (30, 30, 200), 0.45)
        cv2.polylines(img, [np.round(mp).astype(np.int32)], True, RED, th, cv2.LINE_AA)
    for z, r, m in zip(zones, res, masks):
        if r["poly"] is None or r["state"] != "ok":
            continue
        win = _fill(r["poly"], (h, w))
        ex = (win & (1 - m)).astype(np.uint8)
        if ex.sum() > 3:
            exf = cv2.resize(ex, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            ov = img.copy()
            ov[exf] = (0, 235, 255)
            cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    for z, r in zip(zones, res):
        if r["poly"] is None:
            continue
        p = r["poly"] * sc
        bad = r["state"] == "ok" and r["exposed"] >= thr
        cv2.polylines(img, [np.round(p).astype(np.int32)], True, (0, 235, 255) if bad else GREEN, th, cv2.LINE_AA)
    fs = 0.5 * max(0.7, min(1.0, img.shape[1] / 1000))
    for z, r in zip(zones, res):
        if r["poly"] is None or r["state"] == "offframe":
            continue
        p = r["poly"] * sc
        tl = p.min(axis=0)
        okc = r["state"] == "ok" and r["exposed"] < thr
        txt = z.id if (r["state"] != "ok" or okc) else f"{z.id} {r['exposed']:.0f}%"
        bg = (30, 140, 30) if okc else (20, 20, 215)
        (tw_, th_), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        x, y = int(max(0, min(tl[0], img.shape[1] - tw_ - 8))), int(max(th_ + 8, tl[1] - 4))
        cv2.rectangle(img, (x, y - th_ - 6), (x + tw_ + 8, y + 3), bg, -1)
        cv2.putText(img, txt, (x + 4, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# -------------------------------------------------------------------------------------- truth scoring


def match_events(truth_ev: list[dict], meas_ev: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for te in truth_ev:
        cands = [m for m in meas_ev if m["zone"] == te["zone"] and m["kind"] == "EXPOSED"
                 and m["start_s"] < te["end_s"] and m["end_s"] > te["start_s"]]
        if not cands:
            rows.append({"zone": te["zone"], "truth": te, "meas": None})
            continue
        m = max(cands, key=lambda c: min(c["end_s"], te["end_s"]) - max(c["start_s"], te["start_s"]))
        rows.append({"zone": te["zone"], "truth": te, "meas": m})
    return rows


def write_csv(path: Path, events: list[dict[str, Any]]) -> None:
    cols = ["zone", "kind", "start_s", "end_s", "duration_s", "max_exposed_pct", "mean_exposed_pct", "max_misalign_px", "frames"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in events:
            w.writerow({c: e.get(c) for c in cols})


def write_samples_csv(path: Path, out: dict[str, Any]) -> None:
    ids = out["ids"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "inliers", "tracker_src", "lost"] + [f"{z}_exposed_pct" for z in ids] + [f"{z}_misalign_px" for z in ids]
                   + ([f"{z}_truth_pct" for z in ids] if out["truth"] else []))
        for k, t in enumerate(out["ts"]):
            tr = out["tracker"][k]
            row = [t, tr["inliers"], tr["src"], int(out["lost"][k])]
            row += ["" if math.isnan(out["series"][z][k]) else round(out["series"][z][k], 1) for z in ids]
            row += ["" if math.isnan(out["mis"][z][k]) else round(out["mis"][z][k], 1) for z in ids]
            if out["truth"]:
                row += ["" if math.isnan(out["truth"][z][k]) else round(out["truth"][z][k], 1) for z in ids]
            w.writerow(row)
