"""Day A vs Day B: which parked vehicles stayed, left, arrived, moved or were swapped.

The ptz_parking clip is a PTZ camera: between the two simulated days it pans and zooms away
and comes back to a DIFFERENT framing. Comparing raw pixel boxes across days would be fiction,
so this module:
  1. finds the stable view of each day (cuts = big frame-to-frame jumps; segments with the same
     framing are merged when a homography between them is near identity),
  2. aligns Day B's view onto Day A's with a SIFT + RANSAC homography (exact for a pan/zoom camera),
  3. compares parked vehicles only inside the part of the lot both views can see; everything
     else goes to `not_compared` with a reason instead of being called "left" or "arrived".
All boxes in `changes` are in Day A pixel coordinates (Day B's own box is kept as `box_b`).
"""
from __future__ import annotations

import sys
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm.data import CAMS, VEHICLES, days, info, out_dir, save_crop, tracks, write_json

CUT_DIFF = 12.0        # mean abs grey difference (160x90) between samples 0.5 s apart that marks a cut
SAMPLE_S = 0.5
MIN_SEG_S = 10.0       # shorter stable stretches are transitions, not views
SAME_VIEW_PX = 12.0    # homography corner shift below this = same framing
IOU_MATCH = 0.5
SIM_SAME = 0.6         # appearance similarity at/above this = same vehicle
MOVE_WIDTHS = 2.0      # moved = similar look, centre within 2 box widths
MIN_PARK_S = 20.0
BG_FRAMES = 31
KEEP_S = 4.0          # keep one full-res frame this often during the scan (saves ~150 slow seeks)
LAST: dict[str, Any] = {}  # last build's working state, for tests and debugging


# ---------------------------------------------------------------- video helpers
def _read(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, img = cap.read()
    return img if ok else None


def scan_views(cam: str, keep: list[tuple[float, np.ndarray]] | None = None) -> list[dict[str, float]]:
    """Stable segments [{start, end}] separated by cuts, sampled every SAMPLE_S seconds.
    One sequential decode; if `keep` is given, a full-res frame every KEEP_S seconds is appended to it."""
    i = info(cam)
    step = max(1, int(round(i["fps"] * SAMPLE_S)))
    keep_step = step * max(1, int(round(KEEP_S / SAMPLE_S)))
    cap = cv2.VideoCapture(str(CAMS[cam]["video"]))
    prev, k, cuts, last_t = None, 0, [], 0.0
    while cap.grab():
        if k % step == 0:
            ok, img = cap.retrieve()
            if ok:
                g = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                g = g.astype(np.float32)
                t = k / i["fps"]
                if keep is not None and k % keep_step == 0:
                    keep.append((t, img))
                if prev is not None and float(np.abs(g - prev).mean()) > CUT_DIFF:
                    cuts.append(t)
                prev, last_t = g, t
        k += 1
    cap.release()
    segs, s0 = [], 0.0
    for c in cuts + [last_t + SAMPLE_S]:
        a, b = s0, c - SAMPLE_S  # the sample before the cut is the last one of the old view
        if b - a >= MIN_SEG_S:
            segs.append({"start": round(a, 2), "end": round(b, 2)})
        s0 = c
    return segs


def kept_in(keep: list[tuple[float, np.ndarray]], spans: list[tuple[float, float]]) -> list[tuple[float, np.ndarray]]:
    return [(t, f) for t, f in keep if any(a <= t <= b for a, b in spans)]


def median_of(cam: str, keep: list[tuple[float, np.ndarray]], spans: list[tuple[float, float]],
              cap: int = 15) -> np.ndarray:
    fr = [f for _, f in kept_in(keep, spans)]
    if len(fr) > cap:  # full-res medians are slow; 15 spread frames are plenty to drop passers-by
        fr = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, cap)]
    if len(fr) >= 5:
        return np.median(np.stack(fr), axis=0).astype(np.uint8)
    return np.median(np.stack([median_bg(cam, a, b, 9) for a, b in spans]), axis=0).astype(np.uint8)


def median_bg(cam: str, a: float, b: float, n: int = BG_FRAMES) -> np.ndarray:
    cap = cv2.VideoCapture(str(CAMS[cam]["video"]))
    frames = [f for f in (_read(cap, t) for t in np.linspace(a + 0.2, b - 0.2, n)) if f is not None]
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def homography(src: np.ndarray, dst: np.ndarray) -> dict[str, Any]:
    """SIFT + ratio test + RANSAC. H maps src pixels to dst pixels."""
    sift = cv2.SIFT_create(nfeatures=4000)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (src, dst))
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    res = {"H": None, "matches": 0, "inliers": 0, "reproj_px": None}
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return res
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < 0.75 * rest[0].distance]
    res["matches"] = len(good)
    if len(good) < 12:
        return res
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    if H is None:
        return res
    inl = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(pa[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    res.update(H=H, inliers=int(inl.sum()), reproj_px=round(float(np.linalg.norm(proj - pb[inl], axis=1).mean()), 2))
    return res


def fixed_view(src: np.ndarray, dst: np.ndarray, max_px: float = 6.0, margin: float = 0.04) -> dict[str, Any]:
    """Fallback for a fixed camera when SIFT finds too few matches (fog, dusk): phase correlation on CLAHE gradient
    maps must put the shift within max_px, and gradient NCC at that shift must beat every shift of 12-40 px by margin.
    Returns {"ok", "dx", "dy", "ncc", "ncc_off"}; H maps src pixels to dst pixels (a pure translation)."""
    cl = cv2.createCLAHE(3.0, (8, 8))

    def grad(x: np.ndarray) -> np.ndarray:
        g = cl.apply(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
        m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        return cv2.GaussianBlur(m, (0, 0), 2)

    ms, md = grad(src), grad(dst)
    (dx, dy), _ = cv2.phaseCorrelate(ms, md, cv2.createHanningWindow(ms.shape[::-1], cv2.CV_32F))

    def ncc_at(sx: float, sy: float) -> float:
        w = cv2.warpAffine(ms, np.float32([[1, 0, sx], [0, 1, sy]]), ms.shape[::-1])[40:-40, 40:-40]
        p, q = md[40:-40, 40:-40] - md[40:-40, 40:-40].mean(), w - w.mean()
        return float((p * q).sum() / np.sqrt((p * p).sum() * (q * q).sum()))

    here = ncc_at(dx, dy)
    off = max(ncc_at(dx + a, dy + b) for a, b in [(12, 0), (-12, 0), (0, 12), (0, -12), (25, 0), (0, 25), (40, 0),
                                                   (-25, -25)])
    ok = bool(np.hypot(dx, dy) <= max_px and here - off >= margin)
    return {"ok": ok, "dx": round(float(dx), 2), "dy": round(float(dy), 2), "ncc": round(here, 3),
            "ncc_off": round(off, 3), "H": np.float64([[1, 0, dx], [0, 1, dy], [0, 0, 1]])}


def corner_shift(H: np.ndarray, w: int, h: int) -> float:
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return float(np.linalg.norm(cv2.perspectiveTransform(c, H) - c, axis=2).max())


def map_box(H: np.ndarray, box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    p = cv2.perspectiveTransform(np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2), H)
    p = p.reshape(-1, 2)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ---------------------------------------------------------------- views per day
def day_views(cam: str, keep: list[tuple[float, np.ndarray]] | None = None) -> dict[str, dict[str, Any]]:
    """For each day: its main stable view {spans, start, end, seconds, other_views}."""
    keep = [] if keep is None else keep
    segs = scan_views(cam, keep)
    i = info(cam)
    out = {}
    for d in days(cam):
        mine = [{**s, "start": max(s["start"], d["start"]), "end": min(s["end"], d["end"])} for s in segs]
        mine = [s for s in mine if s["end"] - s["start"] >= MIN_SEG_S]
        if not mine:
            continue
        for s in mine:
            s["bg"] = median_of(cam, keep, [(s["start"], s["end"])], cap=7)
        # merge neighbouring segments that show the same framing (e.g. an exposure jump, not a pan)
        groups = [[mine[0]]]
        for s in mine[1:]:
            hm = homography(s["bg"], groups[-1][-1]["bg"])
            if hm["H"] is not None and hm["inliers"] >= 40 and corner_shift(hm["H"], i["w"], i["h"]) <= SAME_VIEW_PX:
                groups[-1].append(s)
            else:
                groups.append([s])
        best = max(groups, key=lambda g: sum(s["end"] - s["start"] for s in g))
        spans = [(s["start"], s["end"]) for s in best]
        out[d["id"]] = {"spans": spans, "start": spans[0][0], "end": spans[-1][1],
                        "seconds": round(sum(b - a for a, b in spans), 2),
                        "other_views": len(groups) - 1}
    return out


def is_infrared(img: np.ndarray) -> bool:
    """Night infrared plates carry no colour: mean HSV saturation stays near zero."""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()) < 12.0


def vehicle_seconds(cam: str, spans: list[tuple[float, float]], box: list[float], min_iou: float = 0.3) -> int:
    """Longest unbroken run of seconds (inside spans) in which any vehicle box overlaps box by IoU >= min_iou."""
    df = tracks(cam)
    df = df[df.name.isin(VEHICLES) & in_spans(df.t, spans)]
    if df.empty:
        return 0
    ix = (np.minimum(df.x2, box[2]) - np.maximum(df.x1, box[0])).clip(lower=0)
    iy = (np.minimum(df.y2, box[3]) - np.maximum(df.y1, box[1])).clip(lower=0)
    inter = ix * iy
    u = (df.x2 - df.x1) * (df.y2 - df.y1) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    secs = sorted(set(df[(inter / u.where(u > 0, 1)) >= min_iou].t.astype(int)))
    run = best = 0
    for i, sec in enumerate(secs):  # longest unbroken run: a standing vehicle, not cars driving past
        run = run + 1 if i and sec == secs[i - 1] + 1 else 1
        best = max(best, run)
    return best


def centre_in(a: list[float], b: list[float]) -> bool:
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


def in_spans(t: pd.Series, spans: list[tuple[float, float]]) -> pd.Series:
    m = pd.Series(False, index=t.index)
    for a, b in spans:
        m |= (t >= a) & (t <= b)
    return m


def parked_in(cam: str, spans: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Parked vehicles seen in the given view spans, one per physical slot (IoU >= 0.5 merged)."""
    df = tracks(cam)
    df = df[(df.tid >= 0) & df.name.isin(VEHICLES) & in_spans(df.t, spans)]
    cands = []
    for tid, g in df.groupby("tid"):
        if len(g) < 8:
            continue
        med = g[["x1", "y1", "x2", "y2"]].median()
        w = float(med.x2 - med.x1)
        dev = np.hypot(g.cx - g.cx.median(), g.cy - g.cy.median()).quantile(0.9)
        if g.t.iat[-1] - g.t.iat[0] < MIN_PARK_S or dev > 0.5 * w:
            continue
        mid = g.iloc[len(g) // 2]
        cands.append({"tid": int(tid), "name": g.name.mode().iat[0], "n": len(g),
                      "box": [float(med.x1), float(med.y1), float(med.x2), float(med.y2)],
                      "t": float(mid.t), "first": float(g.t.iat[0]), "last": float(g.t.iat[-1])})
    cands.sort(key=lambda c: -c["n"])
    slots: list[dict[str, Any]] = []
    for c in cands:
        hit = next((s for s in slots if iou(s["box"], c["box"]) >= IOU_MATCH), None)
        if hit:
            hit["also"].append(c["tid"])
        else:
            slots.append({**c, "also": []})
    return slots


# ---------------------------------------------------------------- appearance
def _patch(img: np.ndarray, box: list[float]) -> np.ndarray | None:
    H, W = img.shape[:2]
    x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
    x2, y2 = int(min(W, box[2])), int(min(H, box[3]))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    return img[y1:y2, x1:x2]


def clip_to(valid: np.ndarray, box: list[float]) -> list[float] | None:
    """Tight box of the part of `box` inside the shared view mask (None if under 16 px either way)."""
    H, W = valid.shape
    x1, y1 = int(max(0, np.floor(box[0]))), int(max(0, np.floor(box[1])))
    x2, y2 = int(min(W, np.ceil(box[2]))), int(min(H, np.ceil(box[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    m = valid[y1:y2, x1:x2]
    cols, rows = np.flatnonzero(m.any(axis=0)), np.flatnonzero(m.any(axis=1))
    if not len(cols):
        return None
    c = [float(x1 + cols[0]), float(y1 + rows[0]), float(x1 + cols[-1] + 1), float(y1 + rows[-1] + 1)]
    if c[2] - c[0] < 16 or c[3] - c[1] < 16:
        return None
    return c


def touches_edge(box: list[float], w: int, h: int, px: float = 3.0) -> bool:
    return box[0] <= px or box[1] <= px or box[2] >= w - px or box[3] >= h - px


def exposure_match(src: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gain/offset that maps src's colours onto ref's over the shared mask."""
    s, r = src[mask].astype(np.float32), ref[mask].astype(np.float32)
    gain = r.std(axis=0) / np.maximum(s.std(axis=0), 1e-3)
    return gain, r.mean(axis=0) - gain * s.mean(axis=0)


def sample_frames(cam: str, spans: list[tuple[float, float]], n: int, H: np.ndarray | None,
                  size: tuple[int, int], gain_off: tuple[np.ndarray, np.ndarray] | None = None,
                  keep: list[tuple[float, np.ndarray]] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """~n frames spread over the spans (optionally warped by H and exposure-matched). Returns (times, frames).
    Uses frames kept during the scan when there are at least 12 of them, else seeks."""
    have = kept_in(keep or [], spans)
    if len(have) >= 12:
        src = have
        cap = None
    else:
        total = sum(b - a for a, b in spans)
        ts = []
        for a, b in spans:
            k = max(2, int(round(n * (b - a) / total)))
            ts += list(np.linspace(a + 0.3, b - 0.3, k))
        cap = cv2.VideoCapture(str(CAMS[cam]["video"]))
        src = [(float(t), _read(cap, float(t))) for t in ts]
        cap.release()
    keep_t, keep_f = [], []
    for t, img in src:
        if img is None:
            continue
        if H is not None:
            img = cv2.warpPerspective(img, H, size)
        if gain_off is not None:
            img = np.clip(img.astype(np.float32) * gain_off[0] + gain_off[1], 0, 255).astype(np.uint8)
        keep_t.append(float(t))
        keep_f.append(img)
    return np.array(keep_t), np.stack(keep_f)


def obj_patch(times: np.ndarray, frames: np.ndarray, o: dict[str, Any], box: list[float]) -> np.ndarray | None:
    """Median patch over the sample frames inside the object's own lifetime (removes passers-by)."""
    idx = np.flatnonzero((times >= o["first"]) & (times <= o["last"]))
    if not len(idx):
        idx = [int(np.abs(times - o["t"]).argmin())]
    ps = [_patch(frames[i], box) for i in idx]
    ps = [p for p in ps if p is not None]
    return np.median(np.stack(ps), axis=0).astype(np.uint8) if ps else None


def similarity(pa: np.ndarray | None, pb: np.ndarray | None) -> float | None:
    """Look similarity in [-1, 1]: mean of a coarse HSV (hue, sat) histogram correlation and a grey
    normalised cross-correlation that tolerates ~8% misalignment (template slid over the other patch)."""
    if pa is None or pb is None or min(pa.shape[:2] + pb.shape[:2]) < 6:
        return None
    pb = cv2.resize(pb, (pa.shape[1], pa.shape[0]), interpolation=cv2.INTER_AREA)
    ha, hb = (cv2.calcHist([cv2.cvtColor(cv2.GaussianBlur(p, (3, 3), 0), cv2.COLOR_BGR2HSV)], [0, 1], None,
                           [18, 16], [0, 180, 0, 256]) for p in (pa, pb))
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    ga, gb = (cv2.resize(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), (48, 48), interpolation=cv2.INTER_AREA)
              .astype(np.float32) for p in (pa, pb))
    if ga.std() < 1 or gb.std() < 1:
        ncc = 0.0
    else:
        ncc = max(float(cv2.matchTemplate(gb, ga[4:44, 4:44], cv2.TM_CCOEFF_NORMED).max()),
                  float(cv2.matchTemplate(ga, gb[4:44, 4:44], cv2.TM_CCOEFF_NORMED).max()))
    return round(0.5 * hist + 0.5 * ncc, 3)


def inner(valid: np.ndarray, box: list[float]) -> list[float] | None:
    """Shrink a box until (almost) every pixel in it is inside the shared view (the view edge is slanted)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = valid.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    while x2 - x1 >= 16 and y2 - y1 >= 16:
        m = valid[y1:y2, x1:x2]
        if m.mean() >= 0.99:
            return [float(x1), float(y1), float(x2), float(y2)]
        side = int(np.argmin([m[:, 0].mean(), m[0, :].mean(), m[:, -1].mean(), m[-1, :].mean()]))
        x1, y1, x2, y2 = x1 + (side == 0), y1 + (side == 1), x2 - (side == 2), y2 - (side == 3)
    return None


# ---------------------------------------------------------------- crops
def crop_ref(cam: str, o: dict[str, Any], frames: list[tuple[float, np.ndarray]]) -> dict[str, Any]:
    """Same crop as stm.data.save_crop(cam, tid, t) (15% pad, crops/<tid>_<t>.jpg) but cut from a frame
    already in memory: the kept frame inside the object's lifetime nearest its middle. Falls back to
    save_crop (which re-opens the video) when no kept frame fits."""
    inside = [(t, f) for t, f in frames if o["first"] <= t <= o["last"]]
    if not inside:
        return {"object_id": o["tid"], "crop": save_crop(cam, o["tid"], o["t"]), "t": round(o["t"], 2)}
    t, img = min(inside, key=lambda x: abs(x[0] - o["t"]))
    g = tracks(cam)
    g = g[g.tid == o["tid"]]
    r = g.iloc[int((g.t - t).abs().argmin())]
    H, W = img.shape[:2]
    px, py = r.bw * 0.15, r.bh * 0.15
    x1, y1 = int(max(0, r.x1 - px)), int(max(0, r.y1 - py))
    x2, y2 = int(min(W, r.x2 + px)), int(min(H, r.y2 + py))
    name = f"{o['tid']}_{t:.1f}.jpg"
    p = out_dir(cam) / "crops" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"object_id": o["tid"], "crop": f"crops/{name}", "t": round(t, 2)}


# ---------------------------------------------------------------- main
def _r(box: list[float]) -> list[float]:
    return [round(float(v), 1) for v in box]


def build(cam: str) -> dict[str, Any] | None:
    t0 = time.time()
    ds = days(cam)
    if len(ds) < 2:
        print(f"{cam}: only one day, nothing to diff")
        return None
    da, db = ds[0]["id"], ds[1]["id"]
    od = out_dir(cam)
    i = info(cam)
    W, H_ = i["w"], i["h"]

    keep: list[tuple[float, np.ndarray]] = []
    views = day_views(cam, keep)
    if da not in views or db not in views:
        missing = [d for d in (da, db) if d not in views]
        res = {"camera": cam, "day_a": da, "day_b": db, "changes": [],
               "counts": dict.fromkeys(["same", "arrived", "left", "moved", "swapped"], 0),
               "background_a": f"background_{da}.jpg", "background_b": f"background_{db}.jpg",
               "status": "UNKNOWN", "reason": f"no stable view of {missing} lasting {MIN_SEG_S:.0f}s", "not_compared": []}
        write_json(od / "diff.json", res)
        return res

    va, vb = views[da], views[db]
    bga = median_of(cam, keep, va["spans"])
    bgb = median_of(cam, keep, vb["spans"])
    cv2.imwrite(str(od / f"background_{da}.jpg"), bga, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(od / f"background_{db}.jpg"), bgb, [cv2.IMWRITE_JPEG_QUALITY, 88])

    hm = homography(bgb, bga)  # Day B pixels -> Day A pixels
    shift = corner_shift(hm["H"], W, H_) if hm["H"] is not None else None
    aligned_ok = hm["H"] is not None and hm["inliers"] >= 40 and (hm["reproj_px"] or 99) <= 3.0
    Hb = hm["H"] if aligned_ok else np.eye(3)
    method = "SIFT + RANSAC homography on per-day median backgrounds"
    fv = None
    if not aligned_ok:  # fog or dusk starves SIFT; a fixed camera still lines up by a few pixels at most
        fv = fixed_view(bgb, bga)
        if fv["ok"]:
            aligned_ok, Hb, shift = True, fv["H"], float(np.hypot(fv["dx"], fv["dy"]))
            method = (f"fixed camera: SIFT found only {hm['inliers']} inliers, so the days were lined up by phase "
                      f"correlation of edge maps (shift {fv['dx']}, {fv['dy']} px; edge match {fv['ncc']} vs "
                      f"{fv['ncc_off']} when shifted)")
    bgb_in_a = cv2.warpPerspective(bgb, Hb, (W, H_))
    cv2.imwrite(str(od / f"background_{db}_aligned.jpg"), bgb_in_a, [cv2.IMWRITE_JPEG_QUALITY, 88])
    valid = cv2.warpPerspective(np.full((H_, W), 255, np.uint8), Hb, (W, H_)) > 0
    overlap = float(valid.mean())

    alignment = {"method": method, "fixed_view": {k: v for k, v in fv.items() if k != "H"} if fv else None,
                 "ok": bool(aligned_ok), "matches": hm["matches"], "inliers": hm["inliers"],
                 "reproj_px": hm["reproj_px"], "corner_shift_px": round(shift, 1) if shift is not None else None,
                 "overlap_frac": round(overlap, 3),
                 "view_a": {"start": round(va["start"], 2), "end": round(va["end"], 2), "seconds": va["seconds"]},
                 "view_b": {"start": round(vb["start"], 2), "end": round(vb["end"], 2), "seconds": vb["seconds"]},
                 "H_b_to_a": np.round(Hb, 6).tolist()}

    base = {"camera": cam, "day_a": da, "day_b": db,
            "background_a": f"background_{da}.jpg", "background_b": f"background_{db}.jpg",
            "background_b_aligned": f"background_{db}_aligned.jpg", "boxes_in": f"day {da} pixels",
            "alignment": alignment}
    if not aligned_ok:
        res = {**base, "changes": [], "counts": dict.fromkeys(["same", "arrived", "left", "moved", "swapped"], 0),
               "status": "UNKNOWN", "not_compared": [],
               "reason": f"camera framing differs between days and could not be aligned "
                         f"({hm['inliers']} inliers, reprojection {hm['reproj_px']} px)"}
        write_json(od / "diff.json", res)
        return res

    A = parked_in(cam, va["spans"])
    B = parked_in(cam, vb["spans"])
    for b in B:
        b["box_a"] = map_box(Hb, b["box"])
        b["cut"] = touches_edge(b["box"], W, H_)  # partly outside day B's own frame
    for a in A:
        a["cut"] = touches_edge(a["box"], W, H_)

    # appearance: day A frames as-is; day B frames warped into day A's view and exposure-matched,
    # so both crops of a spot are pixel-aligned. Median over frames inside each object's lifetime.
    gain_off = exposure_match(bgb_in_a, bga, valid)
    ta, FA = sample_frames(cam, va["spans"], 24, None, (W, H_), keep=keep)
    tb, FB = sample_frames(cam, vb["spans"], 24, Hb, (W, H_), gain_off, keep=keep)
    raw = {"A": kept_in(keep, va["spans"]), "B": kept_in(keep, vb["spans"])}  # unwarped frames for crops
    raw = {da: raw["A"], db: raw["B"]}
    bgb_eq = np.clip(bgb_in_a.astype(np.float32) * gain_off[0] + gain_off[1], 0, 255).astype(np.uint8)

    LAST.clear()
    LAST.update(A=A, B=B, ta=ta, tb=tb, valid=valid, H=Hb)  # small state only (frames are ~150 MB)
    for a in A:
        a["cmp"] = clip_to(valid, a["box"])
    for b in B:
        b["cmp"] = clip_to(valid, b["box_a"])

    def sim_ab(a: dict[str, Any], b: dict[str, Any], box: list[float]) -> float | None:
        return similarity(obj_patch(ta, FA, a, box), obj_patch(tb, FB, b, box))

    def area(x: list[float]) -> float:
        return max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1])

    # 1) pair by IoU of the parts both views can see (day A pixels), greedy, highest first
    pairs = sorted(((iou(a["cmp"], b["cmp"]), ia, ib) for ia, a in enumerate(A) for ib, b in enumerate(B)
                    if a["cmp"] and b["cmp"]), reverse=True)
    used_a, used_b, matched = set(), set(), []
    for v, ia, ib in pairs:
        if v < IOU_MATCH:
            break
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matched.append((ia, ib, v))
    # infrared on one day only: looks can't be compared, and night boxes often merge two cars into one,
    # so a spot also counts as shared when one box's centre sits inside the other
    ir = {da: is_infrared(bga), db: is_infrared(bgb)}
    mixed = ir[da] != ir[db]
    if mixed:
        for ia, a in enumerate(A):
            for ib, b in enumerate(B):
                if ia in used_a or ib in used_b or not a["cmp"] or not b["cmp"]:
                    continue
                if centre_in(a["cmp"], b["cmp"]) or centre_in(b["cmp"], a["cmp"]):
                    used_a.add(ia)
                    used_b.add(ib)
                    matched.append((ia, ib, iou(a["cmp"], b["cmp"])))
    ir_day = next((d for d in (da, db) if ir[d]), None)

    changes: list[dict[str, Any]] = []
    not_compared: list[dict[str, Any]] = []
    crops: dict[tuple[str, int], dict[str, Any]] = {}

    def ref(day: str, o: dict[str, Any]) -> dict[str, Any]:
        k = (day, o["tid"])
        if k not in crops:
            crops[k] = crop_ref(cam, o, raw[day])
        return crops[k]

    def merged(o: dict[str, Any]) -> str:
        return f" Tracker also used id {', '.join(map(str, o['also']))} for this spot." if o["also"] else ""

    def skip(day: str, o: dict[str, Any], box: list[float], reason: str, **kw: Any) -> None:
        not_compared.append({"day": day, "object_id": o["tid"], "box": _r(box), **kw, "reason": reason})

    for ia, ib, v in matched:
        a, b = A[ia], B[ib]
        if mixed:
            changes.append({"kind": "same", "unconfirmed": True, "box": _r(a["box"]), "box_b": _r(b["box"]),
                            "a": ref(da, a), "b": ref(db, b), "similarity": None, "iou": round(v, 3),
                            "note": f"A vehicle is parked in this spot on both days. Day {ir_day} is night infrared "
                                    f"(no colour), so it can't confirm it is the same vehicle." + merged(a)})
            continue
        box = inner(valid,[max(a["cmp"][0], b["cmp"][0]), max(a["cmp"][1], b["cmp"][1]),
                            min(a["cmp"][2], b["cmp"][2]), min(a["cmp"][3], b["cmp"][3])])
        sim = sim_ab(a, b, box) if box else None
        if sim is None:
            skip(da, a, a["box"], "boxes overlap but the shared part is too small to compare looks",
                 box_b=_r(b["box"]), b_object_id=b["tid"])
            continue
        kind = "same" if sim >= SIM_SAME else "swapped"
        is_partial = a["cut"] or b["cut"] or area(a["cmp"]) < 0.9 * area(a["box"])
        if kind == "swapped" and is_partial:  # never claim a swap from a sliver at a frame edge
            skip(da, a, a["box"], f"same spot on both days but the vehicle is only partly visible, so its look "
                                  f"can't be compared reliably (similarity {sim})",
                 box_b=_r(b["box"]), b_object_id=b["tid"], similarity=sim)
            continue
        partial = " Only partly visible on one day." if is_partial else ""
        if kind == "same":
            note = f"Same {a['name']} in this spot on both days (look similarity {sim:.2f})."
        else:
            note = (f"Spot taken on both days, but by a different-looking vehicle "
                    f"(look similarity {sim:.2f} < {SIM_SAME}).")
        changes.append({"kind": kind, "box": _r(a["box"]), "box_b": _r(b["box"]),
                        "a": ref(da, a), "b": ref(db, b), "similarity": sim, "iou": round(v, 3),
                        "note": note + partial + merged(a)})

    free_a = [ia for ia in range(len(A)) if ia not in used_a]
    free_b = [ib for ib in range(len(B)) if ib not in used_b]

    # 2) moved: similar look, nearby, not overlapping enough (compare each at its own spot)
    cand = []
    for ia in (free_a if not mixed else []):  # no look comparison across colour vs infrared
        a = A[ia]
        if not a["cmp"] or a["cut"]:
            continue
        aw = a["box"][2] - a["box"][0]
        acx, acy = (a["box"][0] + a["box"][2]) / 2, (a["box"][1] + a["box"][3]) / 2
        pa = obj_patch(ta, FA, a, a["cmp"])
        for ib in free_b:
            b = B[ib]
            if not b["cmp"] or b["cut"] or pa is None:
                continue
            bx = b["box_a"]
            d = float(np.hypot((bx[0] + bx[2]) / 2 - acx, (bx[1] + bx[3]) / 2 - acy))
            if d > MOVE_WIDTHS * aw:
                continue
            pb = obj_patch(tb, FB, b, b["cmp"])
            if pb is None:
                continue
            sim = similarity(pa, cv2.resize(pb, (pa.shape[1], pa.shape[0])))
            if sim is not None and sim >= SIM_SAME:
                cand.append((sim, ia, ib, d))
    for sim, ia, ib, d in sorted(cand, reverse=True):
        if ia not in free_a or ib not in free_b:
            continue
        a, b = A[ia], B[ib]
        free_a.remove(ia)
        free_b.remove(ib)
        changes.append({"kind": "moved", "box": _r(a["box"]), "box_b": _r(b["box"]), "box_b_in_a": _r(b["box_a"]),
                        "a": ref(da, a), "b": ref(db, b), "similarity": sim,
                        "iou": round(iou(a["cmp"], b["cmp"]), 3),
                        "note": f"Similar-looking {a['name']} (look similarity {sim:.2f}) parked {d:.0f} px away "
                                f"on day {db}."})

    # 3) left / arrived, only where the other day's view sees the whole spot and its pixels disagree
    for ia in free_a:
        a = A[ia]
        if not a["cmp"] or area(a["cmp"]) < 0.9 * area(a["box"]):
            skip(da, a, a["box"], f"spot is (mostly) outside day {db}'s camera view")
            continue
        if mixed:  # decide from detections, not looks: any vehicle box on the spot on the other day?
            n = vehicle_seconds(cam, vb["spans"], a["cmp"])
            if n >= 3:  # 1-2 s = traffic driving past the spot
                skip(da, a, a["box"], f"a vehicle stood on this spot for {n} s in a row on day {db} (night infrared) "
                                      f"but not long enough to count as parked: can't tell", seconds=n)
                continue
            changes.append({"kind": "left", "box": _r(a["box"]), "a": ref(da, a), "b": None, "similarity": None,
                            "note": f"{a['name'].capitalize()} parked on day {da} (seen {a['first']:.0f}-{a['last']:.0f} s); "
                                    f"no vehicle stood on this spot in day {db}'s {vb['seconds'] / 60:.0f} min view"
                                    f"{' (only passing traffic crossed it)' if n else ''}." + merged(a)})
            continue
        still = similarity(obj_patch(ta, FA, a, a["cmp"]), _patch(bgb_eq, a["cmp"]))
        if still is None or still >= SIM_SAME:
            skip(da, a, a["box"], f"no parked track on day {db}, but day {db}'s pixels there still look like this "
                                  f"vehicle (similarity {still}): likely a detector miss, not counted",
                 similarity=still)
            continue
        changes.append({"kind": "left", "box": _r(a["box"]), "a": ref(da, a), "b": None, "similarity": still,
                        "note": f"{a['name'].capitalize()} parked on day {da} (seen {a['first']:.0f}-{a['last']:.0f} s) "
                                f"is gone on day {db}; day {db}'s view of the spot matches it only {still:.2f}."
                                + merged(a)})
    for ib in free_b:
        b = B[ib]
        if not b["cmp"] or b["cut"] or area(b["cmp"]) < 0.9 * area(b["box_a"]):
            skip(db, b, b["box_a"], f"vehicle is cut off by the edge of day {db}'s frame or outside day {da}'s view",
                 box_b=_r(b["box"]))
            continue
        if mixed:
            n = vehicle_seconds(cam, va["spans"], b["cmp"])
            if n >= 3:
                skip(db, b, b["box_a"], f"a vehicle stood on this spot for {n} s in a row on day {da} but not long "
                                        f"enough to count as parked: can't tell", box_b=_r(b["box"]), seconds=n)
                continue
            changes.append({"kind": "arrived", "box": _r(b["box_a"]), "box_b": _r(b["box"]), "a": None,
                            "b": ref(db, b), "similarity": None,
                            "note": f"{b['name'].capitalize()} parked on day {db} (seen {b['first']:.0f}-{b['last']:.0f} s); "
                                    f"no vehicle stood on this spot in day {da}'s {va['seconds'] / 60:.0f} min view"
                                    f"{' (only passing traffic crossed it)' if n else ''}." + merged(b)})
            continue
        before = similarity(_patch(bga, b["cmp"]), obj_patch(tb, FB, b, b["cmp"]))
        if before is None or before >= SIM_SAME:
            skip(db, b, b["box_a"], f"no parked track on day {da}, but day {da}'s pixels there already look like "
                                    f"this vehicle (similarity {before}): likely a detector miss, not counted",
                 box_b=_r(b["box"]), similarity=before)
            continue
        changes.append({"kind": "arrived", "box": _r(b["box_a"]), "box_b": _r(b["box"]), "a": None, "b": ref(db, b),
                        "similarity": before,
                        "note": f"{b['name'].capitalize()} parked on day {db} (seen {b['first']:.0f}-{b['last']:.0f} s) "
                                f"was not there on day {da}; day {da}'s view of the spot matches it only {before:.2f}."
                                + merged(b)})

    order = {"left": 0, "arrived": 1, "swapped": 2, "moved": 3, "same": 4}
    changes.sort(key=lambda c: (order[c["kind"]], c["box"][1], c["box"][0]))
    counts = {k: sum(c["kind"] == k for c in changes) for k in ("same", "arrived", "left", "moved", "swapped")}
    res = {**base, "changes": changes, "counts": counts, "not_compared": not_compared,
           "infrared": {da: ir[da], db: ir[db]}, "unconfirmed": sum(bool(c.get("unconfirmed")) for c in changes),
           "parked": {da: len(A), db: len(B)}, "seconds_to_build": round(time.time() - t0, 1),
           "note": (f"Camera framing differs between day {da} and day {db}; day {db} was aligned onto day {da} with a "
                    f"homography ({hm['inliers']} inliers, {hm['reproj_px']} px error). Only the {overlap:.0%} of "
                    f"day {da}'s frame that day {db} also sees is compared." if (shift or 0) > SAME_VIEW_PX else
                    f"Same camera framing on both days ({hm['inliers']} inliers).")
           + (" " + "; ".join(CAMS[cam]["simulated"]) + "." if CAMS[cam]["simulated"] else "")}
    write_json(od / "diff.json", res)
    lines = [f"diff for {cam}: day {da} view {va['start']:.1f}-{va['end']:.1f}s ({va['seconds']:.0f}s), "
             f"day {db} view {vb['start']:.1f}-{vb['end']:.1f}s ({vb['seconds']:.0f}s)",
             f"alignment: {hm['inliers']}/{hm['matches']} inliers, reprojection {hm['reproj_px']} px, "
             f"corner shift {alignment['corner_shift_px']} px, overlap {overlap:.1%}",
             f"parked vehicles: day {da} {len(A)}, day {db} {len(B)}",
             "counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()),
             f"not compared: {len(not_compared)} "
             f"({sum('outside' in n['reason'] for n in not_compared)} outside the shared view, "
             f"{sum('detector miss' in n['reason'] for n in not_compared)} detector misses, "
             f"{sum('partly visible' in n['reason'] for n in not_compared)} only partly visible)",
             f"built in {res['seconds_to_build']} s"]
    (od / "diff_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


def main(argv: list[str]) -> None:
    cams = argv or [c for c in CAMS if len(CAMS[c]["days"]) >= 2]
    for cam in cams:
        r = build(cam)
        if r is not None:
            print(cam, r["counts"], "not_compared", len(r.get("not_compared", [])), r.get("status", "OK"))
            p = out_dir(cam) / "diff_summary.txt"
            if p.exists():
                print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main(sys.argv[1:])
