"""items: when a placed object (phone on a table, bike at a rack, bag on a bench) was in view, and when it went.

`item_span(video, box)` samples the video at `fps` and, per sample, cuts the box and a ring of its surroundings:
  dark      too dark to see: unknown
  present   the box matches the object's look (also with someone close by: nobody looks exactly like the phone)
  occluded  the ring differs from its usual look (per-pixel median over the clip) and the box does not match the
            object: someone is in front of it. Never counted as gone.
  gone      ring clear and the box no longer matches it (it matches the empty look, or is plainly changed)
  unsure    ring clear, box between the two looks
The object's look comes from `ref_t` (a second it is known to be there) or, by default, automatically: the clear
samples split into two looks (2-medians), and the object is the look its surroundings explain worst (the box is
inpainted from the ring's median look: an empty table continues the table around it, a phone on it does not). With
one look all clip long the box is assumed to show the object throughout.
Two looks differ by max(share of pixels changed, shift of the box's mean colour / MEAN_SCALE). Pixels allow a 1 px
shift (a pixel only counts as changed when it is outside the 3x3 min..max of the other image), so small camera shake
is not a change; the mean colour catches a low-contrast object (dark bag on a dark bench) that few pixels flag.
It went = the last run of clear "gone" samples lasting >= 3 s with no present sample after it. t1 = the last present
sample before that run, so the seconds someone stood in front of it while it went are not counted as visible.
Brightness drift (auto-exposure, clouds) is removed per sample with the ring's median offset.
All times out (and `ref_t` in) are video seconds (frame timestamps) minus offset_s, 2 dp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BOX_PX = 40            # the box is scaled so its longer side is at most this many px
RING = 0.75            # ring width around the box, x the box's longer side
GAP = 0.15             # margin between the box and the ring, x the longer side (shadows, loose boxes)
PIX = 18               # a pixel changed when a blurred BGR channel is this far outside the other's 3x3 min..max
MEAN_SCALE = 40.0      # a box mean-colour shift of this many levels counts like a fully changed box
RING_CLEAR = 0.08      # ring share changed below this = nobody in front of the box
MIN_RING_PX = 20       # fewer ring pixels than this (box fills the frame) = occlusion can't be checked
DARK_HARD, DARK_LUMA, DARK_STD = 15.0, 35.0, 8.0   # nearly black, or dim and flat
MIN_SEP = 0.15         # two looks closer than this (see _dist) are one look
MIN_GONE_S = 3.0       # a departure needs this long of clear "gone" samples
MIN_PRESENT = 2        # samples in a run before "present" counts (one odd frame is not the object)
REF_SAMPLES = 400      # at most this many samples (evenly spread) build the ring's median look
STATES = ("present", "gone", "occluded", "unsure", "dark")


@dataclass
class _Scan:
    t: np.ndarray        # (n,) output seconds (video seconds - offset_s)
    patch: np.ndarray    # (n, bh, bw, 3) float32 box patches, exposure-corrected against the ring's median look
    ring: np.ndarray     # (n,) share of ring pixels changed
    dark: np.ndarray     # (n,) bool
    pred: np.ndarray     # (bh, bw, 3) the box inpainted from the median look of its surroundings
    step: float          # seconds between samples
    has_ring: bool


def _geometry(W: int, H: int, box: list[float]) -> tuple[tuple[int, int, int, int], tuple[int, int],
                                                           tuple[slice, slice], np.ndarray]:
    """ROI (x0, y0, x1, y1) in frame px, scaled ROI size (w, h), box slices and ring mask in scaled ROI px."""
    x, y, w, h = (float(v) for v in box)
    if not (w > 0 and h > 0):
        raise ValueError(f"box needs w and h > 0, got {box}")
    bx0, by0 = max(0.0, x * W), max(0.0, y * H)
    bx1, by1 = min(float(W), (x + w) * W), min(float(H), (y + h) * H)
    if bx1 - bx0 < 2 or by1 - by0 < 2:
        raise ValueError(f"box {box} is outside the {W}x{H} frame or under 2 px")
    L = max(bx1 - bx0, by1 - by0)
    pad = (RING + GAP) * L
    X0, Y0 = max(0, int(bx0 - pad)), max(0, int(by0 - pad))
    X1, Y1 = min(W, int(math.ceil(bx1 + pad))), min(H, int(math.ceil(by1 + pad)))
    s = min(1.0, BOX_PX / L)
    rw, rh = max(1, round((X1 - X0) * s)), max(1, round((Y1 - Y0) * s))

    def rc(a: float, b: float, lo: int, n: int) -> tuple[int, int]:
        i0 = min(n - 1, max(0, int(round((a - lo) * s))))
        return i0, min(n, max(i0 + 1, int(round((b - lo) * s))))

    xa, xb = rc(bx0, bx1, X0, rw)
    ya, yb = rc(by0, by1, Y0, rh)
    g = GAP * L
    ga, gb = rc(bx0 - g, bx1 + g, X0, rw)
    gc, gd = rc(by0 - g, by1 + g, Y0, rh)
    ring = np.ones((rh, rw), bool)
    ring[gc:gd, ga:gb] = False
    return (X0, Y0, X1, Y1), (rw, rh), (slice(ya, yb), slice(xa, xb)), ring


_K3 = np.ones((3, 3), np.uint8)


def _range(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3x3 min and max of an (h, w, 3) float32 image."""
    return cv2.erode(q, _K3), cv2.dilate(q, _K3)


def _scan(video_path: str, box: list[float], fps: float, offset_s: float) -> _Scan:
    """One sequential pass: scaled, blurred ROI per sample; then ring changes against the clip's median ROI."""
    if not fps > 0:
        raise ValueError("fps must be > 0")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"can't open video {video_path}")
    try:
        vfps = cap.get(cv2.CAP_PROP_FPS)
        vfps = vfps if vfps and math.isfinite(vfps) and vfps > 0 else 25.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        (X0, Y0, X1, Y1), size, bsl, ring = _geometry(W, H, box)
        step = 1.0 / fps
        ts: list[float] = []
        rois: list[np.ndarray] = []
        k, nxt = 0, 0.0
        while cap.grab():
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)      # frame timestamp: right for variable frame rate video
            tv = ms / 1000.0 if ms > 0 else k / vfps
            k += 1
            if tv + 1e-6 < nxt:
                continue
            ok, fr = cap.retrieve()
            if not ok:
                break
            while nxt <= tv + 1e-6:
                nxt += step
            roi = cv2.resize(fr[Y0:Y1, X0:X1], size, interpolation=cv2.INTER_AREA)
            rois.append(cv2.GaussianBlur(roi, (3, 3), 0))
            ts.append(tv - offset_s)
    finally:
        cap.release()
    n = len(rois)
    bh, bw = bsl[0].stop - bsl[0].start, bsl[1].stop - bsl[1].start
    if n == 0:
        return _Scan(np.zeros(0), np.zeros((0, bh, bw, 3), np.float32), np.zeros(0), np.zeros(0, bool),
                     np.zeros((bh, bw, 3), np.float32), step, False)
    stack = np.stack(rois)
    pick = np.unique(np.linspace(0, n - 1, min(n, REF_SAMPLES)).round().astype(int))
    ref = np.median(stack[pick], axis=0).astype(np.float32)
    lo, hi = _range(ref)
    lo, hi = lo[ring], hi[ring]
    has_ring = int(ring.sum()) >= MIN_RING_PX
    mask = np.zeros(ring.shape, np.uint8)
    mask[bsl] = 255
    pred = cv2.inpaint(np.clip(ref, 0, 255).astype(np.uint8), mask, 3, cv2.INPAINT_TELEA)[bsl].astype(np.float32)
    patch = np.zeros((n, bh, bw, 3), np.float32)
    ring_fg = np.zeros(n)
    dark = np.zeros(n, bool)
    for i in range(n):
        roi = stack[i]
        f = roi.astype(np.float32)
        off = np.zeros(3, np.float32)
        if has_ring:
            r = f[ring]
            off = np.median(r - ref[ring], axis=0)
            r = r - off
            ring_fg[i] = float(((r < lo - PIX) | (r > hi + PIX)).any(axis=1).mean())
        patch[i] = f[bsl] - off
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        m, sd = float(g.mean()), float(g.std())
        dark[i] = m < DARK_HARD or (m < DARK_LUMA and sd < DARK_STD)
    return _Scan(np.asarray(ts), patch, ring_fg, dark, pred, step, has_ring)


def _dist(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """How different box patches p (..., h, w, 3) are from look q (h, w, 3), 0..1: the larger of the share of pixels
    outside q's 3x3 min..max by > PIX and the box's mean-colour shift / MEAN_SCALE."""
    lo, hi = _range(np.ascontiguousarray(q, np.float32))
    frac = ((p < lo - PIX) | (p > hi + PIX)).any(axis=-1).mean(axis=(-1, -2))
    shift = np.abs(p.mean(axis=(-3, -2)) - q.mean(axis=(0, 1))).max(axis=-1) / MEAN_SCALE
    return np.maximum(frac, np.minimum(shift, 1.0))


def _looks(sc: _Scan, clear: np.ndarray, ref_t: float | None) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Object look O, empty look E (or None) and how they were found."""
    idx = np.flatnonzero(clear)
    if len(idx) == 0:
        return None, None, "no clear view of the box"
    P = sc.patch[idx]
    if ref_t is not None:
        j = idx[int(np.argmin(np.abs(sc.t[idx] - ref_t)))]
        if abs(sc.t[j] - ref_t) > 5.0:
            return None, None, f"no clear view of the box within 5 s of ref_t {ref_t:.2f}"
        near = _dist(P, sc.patch[j]) < MIN_SEP
        O = np.median(P[near], axis=0)
        far = _dist(P, O) >= 2 * MIN_SEP
        E = np.median(P[far], axis=0) if far.sum() >= 3 else None
        return O, E, f"object look from the clear sample at {sc.t[j]:.2f} s (ref_t {ref_t:.2f})"
    c1 = np.median(P, axis=0)                     # dominant look
    d1 = _dist(P, c1)
    nxt = np.r_[d1[1:], 0.0]
    pair = np.array([float(_dist(P[i + 1], P[i])) for i in range(len(P) - 1)] + [1.0])
    cand = np.flatnonzero((d1 >= MIN_SEP) & (nxt >= MIN_SEP) & (pair < MIN_SEP))   # a sustained other look
    if len(cand) == 0:
        return c1, None, "one look all clip long: taken to be the object"
    c2 = P[cand[np.argmax(d1[cand])]]
    lab = np.zeros(len(P), bool)
    for _ in range(6):
        lab = _dist(P, c2) < _dist(P, c1)
        if lab.sum() < MIN_PRESENT or (~lab).sum() < MIN_PRESENT:
            break
        c1, c2 = np.median(P[~lab], axis=0), np.median(P[lab], axis=0)
    if lab.sum() < MIN_PRESENT or (~lab).sum() < MIN_PRESENT or float(_dist(c1, c2)) < MIN_SEP:
        return np.median(P, axis=0), None, "one look all clip long: taken to be the object"
    if not sc.has_ring:                           # no surroundings to compare: the more common look is the object
        O, E = (c2, c1) if lab.sum() > (~lab).sum() else (c1, c2)
        return O, E, "two looks; no surroundings to tell them apart, the more common one taken to be the object"
    u1, u2 = (float(np.abs(c - sc.pred).mean()) for c in (c1, c2))
    O, E = (c2, c1) if u2 > u1 else (c1, c2)
    return O, E, "two looks; the object is the one its surroundings explain worst"


def _classify(sc: _Scan, ref_t: float | None) -> tuple[list[str], np.ndarray, str, bool]:
    """Per-sample state, distance to the object look, how the look was found, and whether an empty look exists."""
    n = len(sc.t)
    occ = sc.ring >= RING_CLEAR
    clear = ~occ & ~sc.dark
    states = ["dark" if sc.dark[i] else "occluded" if occ[i] else "unsure" for i in range(n)]
    O, E, how = _looks(sc, clear, ref_t)
    dO = np.full(n, np.nan)
    if O is None:
        return states, dO, how, False
    dO = _dist(sc.patch, O)
    dE = _dist(sc.patch, E) if E is not None else np.full(n, np.inf)
    busy = occ & ~sc.dark & (dO < MIN_SEP) & (dO <= dE)      # someone close by, the object plainly in view
    for i in np.flatnonzero(busy):
        states[i] = "present"
    if E is None:
        for i in np.flatnonzero(clear):
            states[i] = "present" if dO[i] < MIN_SEP else "gone" if dO[i] >= 2 * MIN_SEP else "unsure"
        return states, dO, how, False
    G = max(MIN_SEP, 0.5 * float(_dist(O, E)))
    for i in np.flatnonzero(clear):
        if dO[i] < G and dO[i] <= dE[i]:
            states[i] = "present"
        elif dO[i] >= G and (dE[i] < dO[i] or dO[i] >= 2 * G):
            states[i] = "gone"
    return states, dO, how, True


def _runs(t: np.ndarray, states: list[str], step: float) -> list[dict[str, Any]]:
    """Runs of present / gone samples (other states are gaps), blips dropped, neighbours of one state merged."""
    raw: list[dict[str, Any]] = []
    for i, s in enumerate(states):
        if s not in ("present", "gone"):
            continue
        if raw and raw[-1]["state"] == s:
            raw[-1]["last"] = i
            raw[-1]["n"] += 1
        else:
            raw.append({"state": s, "first": i, "last": i, "n": 1})

    def valid(r: dict[str, Any]) -> bool:
        if r["state"] == "present":
            return r["n"] >= MIN_PRESENT
        return t[r["last"]] - t[r["first"]] + step >= MIN_GONE_S - 1e-6

    out: list[dict[str, Any]] = []
    for r in (r for r in raw if valid(r)):
        if out and out[-1]["state"] == r["state"]:
            out[-1]["last"] = r["last"]
            out[-1]["n"] += r["n"]
        else:
            out.append(dict(r))
    return out


def _spans(t: np.ndarray, states: list[str], want: str, step: float) -> list[list[float]]:
    out: list[list[float]] = []
    for i, s in enumerate(states):
        if s != want:
            continue
        if out and t[i] - out[-1][1] <= step * 1.5:
            out[-1][1] = float(t[i])
        else:
            out.append([float(t[i]), float(t[i])])
    return [[round(a, 2), round(b, 2)] for a, b in out]


def _fmt_spans(sp: list[list[float]], k: int = 4) -> str:
    s = ", ".join(f"{a:.1f}-{b:.1f}" for a, b in sp[:k])
    return s + (f" (+{len(sp) - k} more)" if len(sp) > k else "")


def item_frames(video_path: str, box: list[float], fps: float = 2, offset_s: float = 0.0, *,
                ref_t: float | None = None, per_second: bool = True) -> list[dict[str, Any]]:
    """Presence per second (or per sample with per_second=False), for debugging:
    [{"t", "state": present|gone|occluded|unsure|dark, "ring": share of ring changed, "d_obj": box vs object look}].
    "gone" here is the raw per-sample call; item_span only believes it after MIN_GONE_S of it.
    A second takes the strongest call among its samples: present > gone > occluded > unsure > dark."""
    sc = _scan(video_path, box, fps, offset_s)
    states, dO, _, _ = _classify(sc, ref_t)
    rows = [{"t": round(float(sc.t[i]), 2), "state": states[i], "ring": round(float(sc.ring[i]), 3),
             "d_obj": None if np.isnan(dO[i]) else round(float(dO[i]), 3)} for i in range(len(states))]
    if not per_second:
        return rows
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        sec = math.floor(r["t"] + 1e-6)
        cur = out.get(sec)
        if cur is None:
            out[sec] = dict(r, t=sec)
            continue
        if STATES.index(r["state"]) < STATES.index(cur["state"]):
            cur["state"] = r["state"]
        cur["ring"] = max(cur["ring"], r["ring"])
        if r["d_obj"] is not None:
            cur["d_obj"] = r["d_obj"] if cur["d_obj"] is None else min(cur["d_obj"], r["d_obj"])
    return [out[k] for k in sorted(out)]


def item_span(video_path: str, box: list[float], fps: float = 2, offset_s: float = 0.0, *,
              ref_t: float | None = None) -> dict[str, Any]:
    """When the object in `box` (normalised [x, y, w, h]) was first visible and the last second it was visible
    before it went. {"t0": float|None, "t1": float|None (None = never went, or can't tell), "evidence": str,
    "gone_t": first clear sample of the departure|None, "occluded": [[start, end], ...]}.
    ref_t (optional, same time base as the output) = a second the object is known to be in the box."""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"no video at {video_path}")
    sc = _scan(video_path, box, fps, offset_s)
    res: dict[str, Any] = {"t0": None, "t1": None, "evidence": "", "gone_t": None, "occluded": []}
    if len(sc.t) == 0:
        res["evidence"] = "no frames could be read"
        return res
    states, _, how, has_empty = _classify(sc, ref_t)
    t, step = sc.t, sc.step
    occ = _spans(t, states, "occluded", step)
    res["occluded"] = occ
    runs = _runs(t, states, step)
    parts = [how]
    if not sc.has_ring:
        parts.append("box fills the view, no surroundings to check for people in front: relying on the 3 s rule")
    n_dark, n_unsure = states.count("dark"), states.count("unsure")
    first_p = next((r for r in runs if r["state"] == "present"), None)
    if first_p is None:
        parts.append("never clearly seen in the box")
        if occ:
            parts.append(f"occluded {_fmt_spans(occ)} s")
        res["evidence"] = "; ".join(parts)
        return res
    res["t0"] = round(float(t[first_p["first"]]), 2)
    parts.append(f"first seen {res['t0']:.2f} s")
    if occ:
        parts.append(f"occluded {_fmt_spans(occ)} s (not counted as gone)")
    away = [r for r in runs[:-1] if r["state"] == "gone" and r["first"] > first_p["first"]]
    if away:
        parts.append("away and back: " + ", ".join(f"{t[r['first']]:.1f}-{t[r['last']]:.1f}" for r in away) + " s")
    last = runs[-1]
    if last["state"] == "gone" and last["first"] > first_p["first"]:
        prev = next(r for r in reversed(runs[:-1]) if r["state"] == "present")
        res["t1"] = round(float(t[prev["last"]]), 2)
        res["gone_t"] = round(float(t[last["first"]]), 2)
        dur = float(t[last["last"]] - t[last["first"]] + step)
        look = "matches the empty look" if has_empty else "box plainly changed"
        between = [s for s in occ if s[1] >= res["t1"] and s[0] <= res["gone_t"]]
        parts.append(f"last visible {res['t1']:.2f} s; gone from {res['gone_t']:.2f} s ({dur:.1f} s with a clear "
                     f"view, {look})" + (f"; someone in front {_fmt_spans(between)} s while it went" if between else ""))
    else:
        tail = [s for s in states[last["last"] + 1:] if s != "present"]
        lastseen = float(t[last["last"]]) if last["state"] == "present" else None
        if lastseen is not None and not tail:
            parts.append(f"still there at the end ({lastseen:.2f} s)")
        elif lastseen is not None:
            g = states[last["last"] + 1:].count("gone")
            why = (f"box changed for only {g * step:.1f} s at the end, too short to call gone" if g else
                   "can't see the box after that")
            parts.append(f"last clearly seen {lastseen:.2f} s; {why}")
    if n_unsure:
        parts.append(f"{n_unsure} unsure samples")
    if n_dark:
        parts.append(f"{n_dark} dark samples")
    res["evidence"] = "; ".join(parts)
    return res
