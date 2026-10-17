#!/usr/bin/env python
"""RELEASE DESK: subject-access-request CCTV release, mask-by-default.

Core loop (see README.md):
  1 candidates   real SQL over the 5 s segment table + a track table (duckdb stands in for VastDB)
  2 pick         officer gives --tid; for tests the desk auto-picks (and says so)
  3 trim         first..last seen of the requester, +/- 1 s
  4 export       everything outside the requester's dilated region is pixelated + blurred + dimmed
  5 leak check   independent det_l + pose on sampled frames of the EXPORT FILE: anything person/vehicle outside = LEAK
  6 risky        frames where another person/vehicle overlaps the released region -> forced human review + crops
  7 record       release_record.json / report.html with hashes, leak result and a stopwatch

    python run.py --all
    python run.py --video retail_cctv [--tid 608] [--window 0:111] [--cls person]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

LEAK_CLASSES = ("person", "car", "truck", "bus", "motorcycle")  # people + vehicles (plates identify people)
HO_MIN_PER_MIN = 9.0        # Home Office ACE case study: ~9 h to redact ~60 min of footage = 9 min per minute
PAD_S = 0.4                 # +/- trim padding (was 1.0 s: the extra seconds were extrapolated boxes that released only slivers)
MAX_GAP_S = 0.9             # requester gap that is bridged by interpolation; longer gaps => frame fully masked
EXTRAP_S = 0.4              # extrapolated boxes beyond the tracked span (tracker confirmation lag) never go further than this
MIN_EXTRAP_VISIBLE = 0.25   # an extrapolated box must have >= 25% of its area inside the frame, else the frame is not released
OPEN_K = 5                  # morphological opening of the released region: removes unmasked slivers <= 4 px wide
DUP_IOU = 0.6               # another detection overlapping the requester box this much is the same object
FACE_KP = (0, 1, 2, 3, 4)   # COCO nose, eyes, ears
MIN_STRANGER_PX = 24        # hero/source panel: a stranger box shorter than this is not drawn nor counted
LEAK_CONF = 0.35            # gating confidence of the leak check (detectors run at 0.20; weaker hits are reported, not gating)

# the simulated requests (the window/class are the REQUEST; the desk finds the requester)
DEFAULTS: dict[str, dict[str, Any]] = {
    "retail_cctv": dict(cls="person", window=(0.0, 110.9), subject="a shopper in the store",
                        request="Any footage of me in the shop on the day (I was the shopper in the shop)"),
    "lisbon_parking": dict(cls="car", window=(15.0, 19.0), subject="my parked car", min_dur=1.5, stranger_note="a face on a poster",
                           request="Footage of my black Mini parked on the kerb (a political poster with a face hangs behind it)"),
    "person_bicycle_car_detection": dict(cls="person", window=(38.0, 53.9), subject="a person walking through the car park",
                                         request="Footage of me crossing the car park"),
}
# extra "hard case" requests, run with --hard (stored under results key <video>__<tag>): the earlier lisbon demo, a stranger walks in front of the van
HARD = {"van": dict(video="lisbon_parking", cls="car", window=(529.0, 535.5), subject="my parked van", min_dur=3.0,
                    request="Footage of my red van parked on the pavement (an elderly woman walked in front of it)", ref="SAR-DEMO-0004",
                    stranger_note="an elderly woman")}
DEFAULT_SET = list(DEFAULTS)

# ---------------------------------------------------------------------------------------------- timing helpers

GPU_WAIT = [0.0]
_orig_lock = pk.gpu_lock


@contextlib.contextmanager
def _timed_lock(timeout: float = 3600.0):
    t0 = time.time()
    with _orig_lock(timeout):
        GPU_WAIT[0] += time.time() - t0
        yield


pk.gpu_lock = _timed_lock  # protokit functions look up gpu_lock at call time: we learn how long the shared GPU made us wait


class Timer:
    def __init__(self) -> None:
        self.stages: dict[str, dict[str, float]] = {}

    @contextlib.contextmanager
    def stage(self, name: str):
        t0, g0 = time.time(), GPU_WAIT[0]
        yield
        wall, wait = time.time() - t0, GPU_WAIT[0] - g0
        self.stages[name] = {"wall_s": round(wall, 2), "gpu_lock_wait_s": round(wait, 2), "net_s": round(wall - wait, 2)}

    def total(self) -> dict[str, float]:
        w = sum(s["wall_s"] for s in self.stages.values())
        g = sum(s["gpu_lock_wait_s"] for s in self.stages.values())
        return {"wall_s": round(w, 2), "gpu_lock_wait_s": round(g, 2), "net_s": round(w - g, 2)}


# ---------------------------------------------------------------------------------------------- geometry helpers


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def inter_area(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def box_area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def mask_of(rects, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in rects:
        m[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = 1
    return m.astype(bool)


def out_frac(box, region: np.ndarray) -> float:
    """Fraction of a detection box that lies OUTSIDE the released region."""
    h, w = region.shape
    x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return 1.0 - float(region[y1:y2, x1:x2].mean())


# ------------------------------------------------------------------------------------------ 1. candidates (SQL)


def detect_cuts(src: Path, f_lo: int, f_hi: int, thr: float = 25.0) -> list[int]:
    """Shot starts in [f_lo, f_hi] (first entry = f_lo). A track id must never be followed across a cut."""
    cuts, prev = [f_lo], None
    for f, fr in read_frames(src, list(range(f_lo, f_hi + 1))):
        g = cv2.resize(fr, (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
        if prev is not None and float(np.abs(g - prev).mean()) > thr:
            cuts.append(f)
        prev = g
    return cuts


def build_tables(name: str, tr: pk.Tracks, window: tuple[float, float], cls: str, cuts: list[int] | None = None):
    cuts = cuts or [0]
    ends = cuts[1:] + [10 ** 9]
    seg = pk.segments(name, tr=tr)
    for c in LEAK_CLASSES:  # make sure the columns exist for any class
        for p in ("n_", "peak_"):
            if p + c not in seg:
                seg[p + c] = 0
    lo, hi = window
    rows = tr.rows[tr.rows["tid"] >= 0]
    rw = rows[(rows["t"] >= lo) & (rows["t"] <= hi)].copy()
    rw["shot"] = np.searchsorted(np.asarray(cuts), rw["f"].to_numpy(), side="right") - 1
    lk = tr.rows[tr.rows["name"].isin(LEAK_CLASSES) & (tr.rows["t"] >= lo - 1.0) & (tr.rows["t"] <= hi + 1.0)]
    by_f = {f: g[["x1", "y1", "x2", "y2", "tid"]].to_numpy() for f, g in lk.groupby("f")}
    recs = []
    for (tid, shot), g in rw.groupby(["tid", "shot"]):
        nm = g["name"].value_counts().index[0]
        if g["name"].nunique() > 1:
            g = g[g["name"] == nm]
        ov = 0
        for r in g.itertuples():
            box = (r.x1, r.y1, r.x2, r.y2)
            for o in by_f.get(r.f, ()):
                if o[4] != tid and iou(box, o[:4]) <= DUP_IOU and inter_area(box, o[:4]) >= 0.02 * box_area(o[:4]):
                    ov += 1
                    break
        em = max(2.0, 0.03 * min(tr.w, tr.h))  # box within 3% of a frame border = partly cut off / entering / leaving
        edge = ((g["x1"] < em) | (g["y1"] < em) | (g["x2"] > tr.w - em) | (g["y2"] > tr.h - em)).mean()
        dur = float(g["t"].max() - g["t"].min())
        gaps = np.diff(np.sort(g["t"].to_numpy()))
        bridged = float(gaps[gaps <= MAX_GAP_S].sum() / dur) if dur > 0 else 0.0  # share of the track the export can follow
        recs.append(dict(tid=int(tid), shot=int(max(shot, 0)), shot_from=int(cuts[max(shot, 0)]), shot_to=int(min(ends[max(shot, 0)] - 1, 10 ** 8)),
                         name=nm, t_first=round(float(g["t"].min()), 2), t_last=round(float(g["t"].max()), 2),
                         dur=round(dur, 2), n=len(g), cov=round(bridged, 3),
                         conf=round(float(g["conf"].mean()), 3), bh=round(float(g["bh"].median()), 1),
                         bw=round(float(g["bw"].median()), 1), overlap=round(ov / len(g), 3), edge=round(float(edge), 3)))
    trk = pd.DataFrame(recs, columns=["tid", "shot", "shot_from", "shot_to", "name", "t_first", "t_last", "dur", "n", "cov", "conf", "bh", "bw", "overlap", "edge"])
    return seg, trk


def candidate_sql(seg: pd.DataFrame, trk: pd.DataFrame, window: tuple[float, float], cls: str,
                  max_dur: float = 45.0, min_h: float = 60.0, min_dur: float = 3.0) -> dict[str, Any]:
    """Real SQL (duckdb standing in for VastDB). Simple dialect on purpose: no JOIN / GROUP BY / IN / window."""
    con = pk.vastdb({"segments": seg, "tracks": trk})
    lo, hi = window
    q_seg = (f'SELECT seg, start, "end", n_{cls}, peak_{cls} FROM segments '
             f'WHERE start < {hi:.2f} AND "end" > {lo:.2f} AND n_{cls} > 0 ORDER BY start')
    q_trk = ("SELECT tid, shot, shot_from, shot_to, name, t_first, t_last, dur, bh, conf, cov, overlap, edge, "
             "LEAST(dur, 30) * conf * (1 - overlap) * (1 - edge) * LEAST(bh, 250) / 250 AS score FROM tracks "
             f"WHERE name = '{cls}' AND bh >= {min_h:.0f} AND cov >= 0.8 AND dur >= {min_dur:g} AND dur <= {max_dur:.0f} "
             "ORDER BY score DESC LIMIT 10")
    q_all = ("SELECT tid, shot, shot_from, shot_to, name, t_first, t_last, dur, bh, conf, cov, overlap, edge FROM tracks "
             f"WHERE name = '{cls}' ORDER BY dur DESC LIMIT 5")
    out = {"q_seg": q_seg, "q_trk": q_trk, "segments": con.execute(q_seg).df(), "tracks": con.execute(q_trk).df()}
    if out["tracks"].empty:  # relax: nothing passed the visibility filters, show what exists
        out["tracks"] = con.execute(q_all).df().assign(score=0.0)
        out["relaxed"] = True
    return out


def candidate_sheet(name: str, tr: pk.Tracks, cand: pd.DataFrame, picked: list[int], window, path: Path) -> None:
    tiles, labels = [], []
    for r in cand.head(8).itertuples():
        g = tr.track(r.tid)
        g = g[(g["t"] >= window[0]) & (g["t"] <= window[1])]
        b = g.loc[(g["bw"] * g["bh"] * g["conf"]).idxmax()]
        fr = pk.frame_at(name, float(b["t"]))
        m = 0.15 * max(b.bw, b.bh)
        x1, y1 = int(max(0, b.x1 - m)), int(max(0, b.y1 - m))
        x2, y2 = int(min(fr.shape[1], b.x2 + m)), int(min(fr.shape[0], b.y2 + m))
        crop = fr[y1:y2, x1:x2]
        s = min(150 / crop.shape[1], 190 / crop.shape[0])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
        tile = np.full((190, 150, 3), 28, np.uint8)
        oy, ox = (190 - crop.shape[0]) // 2, (150 - crop.shape[1]) // 2
        tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        if r.tid in picked:
            cv2.rectangle(tile, (0, 0), (149, 189), (80, 200, 60), 4)
        tiles.append(tile)
        labels.append(f"#{r.tid} {r.dur:.0f}s {r.bh:.0f}px")
    if not tiles:
        return
    pk.save_jpg(pk.sheet(tiles, labels, cols=min(8, len(tiles)), tile_w=150), path)


# ------------------------------------------------------------------------------------------ 2/3. plan the release


class Plan:
    """Per exported frame: the released rectangles (+ optional exact mask), strangers that overlap, mode."""

    def __init__(self, f0: int, f1: int, h: int, w: int):
        self.f0, self.f1, self.h, self.w = f0, f1, h, w
        self.n = f1 - f0 + 1
        self.rects: list[list[tuple[float, float, float, float]]] = [[] for _ in range(self.n)]
        self.modes: list[str] = ["gap"] * self.n
        self.track_boxes: list[list[tuple[float, float, float, float]]] = [[] for _ in range(self.n)]
        self.overlaps: list[list[dict[str, Any]]] = [[] for _ in range(self.n)]
        self.masks: dict[int, np.ndarray] = {}
        self.seg_state: dict[int, str] = {}
        self.holes: dict[int, list[tuple[float, float, float, float]]] = {}  # boxes the leak check made us cut out

    def region(self, i: int) -> np.ndarray:
        if i in self.masks:
            m = np.unpackbits(self.masks[i])[: self.h * self.w].reshape(self.h, self.w).astype(bool)
        else:
            m = mask_of(self.rects[i], self.h, self.w)
        for x1, y1, x2, y2 in self.holes.get(i, ()):
            m[max(0, int(y1)):min(self.h, int(y2)), max(0, int(x1)):min(self.w, int(x2))] = False
        if m.any():  # opening: a hole cut at the box edge must not leave a 1-2 px unmasked sliver that belongs to nobody
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_K, OPEN_K))) > 0
        return m

    def set_mask(self, i: int, m: np.ndarray) -> None:
        self.masks[i] = np.packbits(m.astype(np.uint8), axis=None)


def _track_arrays(tr: pk.Tracks, tids: list[int], frange: tuple[int, int]):
    arr = {}
    for t in tids:
        g = tr.track(t)
        g = g[(g["f"] >= frange[0]) & (g["f"] <= frange[1])].sort_values("f")
        g = g.drop_duplicates("f")
        arr[t] = (g["f"].to_numpy(), g[["x1", "y1", "x2", "y2"]].to_numpy(float))
    return arr


def _box_at(fs: np.ndarray, B: np.ndarray, f: int, fps: float, stride: int):
    """-> (box, mode, dist_frames, speed_px_per_frame) or None (gap)."""
    k = min(6, len(fs) - 1)

    def vel(a: int, b: int):
        if fs[b] == fs[a]:
            return np.zeros(4)
        return (B[b] - B[a]) / (fs[b] - fs[a])

    if f < fs[0]:
        d = fs[0] - f
        if d / fps > EXTRAP_S:
            return None
        v = vel(0, k)
        return B[0] + v * (f - fs[0]), "extrap", d, float(np.hypot(*(0.5 * (v[:2] + v[2:]))))
    if f > fs[-1]:
        d = f - fs[-1]
        if d / fps > EXTRAP_S:
            return None
        v = vel(len(fs) - 1 - k, len(fs) - 1)
        return B[-1] + v * (f - fs[-1]), "extrap", d, float(np.hypot(*(0.5 * (v[:2] + v[2:]))))
    i = int(np.searchsorted(fs, f))
    if fs[i] == f:
        a, b = max(0, i - 1), min(len(fs) - 1, i + 1)
        v = vel(a, b)
        return B[i].copy(), "tracked", 0, float(np.hypot(*(0.5 * (v[:2] + v[2:]))))
    a, b = i - 1, i
    if (fs[b] - fs[a]) / fps > MAX_GAP_S:
        return None
    u = (f - fs[a]) / (fs[b] - fs[a])
    v = vel(a, b)
    return B[a] * (1 - u) + B[b] * u, "interp", int(min(f - fs[a], fs[b] - f)), float(np.hypot(*(0.5 * (v[:2] + v[2:]))))


def make_plan(tr: pk.Tracks, tids: list[int], window: tuple[float, float], fps: float, nframes: int,
              frange: tuple[int, int]) -> Plan:
    """frange = the shot (frame range without a cut) the requester's track lives in: the release never leaves it."""
    arr = _track_arrays(tr, tids, frange)
    ts = tr.rows[tr.rows["tid"].isin(tids) & (tr.rows["t"] >= window[0]) & (tr.rows["t"] <= window[1])
                 & (tr.rows["f"] >= frange[0]) & (tr.rows["f"] <= frange[1])]
    t_first, t_last = float(ts["t"].min()), float(ts["t"].max())
    f0 = max(0, frange[0], int(round((max(window[0], t_first - PAD_S)) * fps)))
    f1 = min(nframes - 1, frange[1], int(round((min(window[1], t_last + PAD_S)) * fps)))
    plan = Plan(f0, f1, tr.h, tr.w)
    stride = tr.stride
    rows = tr.rows[tr.rows["name"].isin(LEAK_CLASSES) & (tr.rows["f"] >= f0 - 2 * stride) & (tr.rows["f"] <= f1 + 2 * stride)]
    by_f = {f: g[["x1", "y1", "x2", "y2", "tid", "name"]].to_numpy(object) for f, g in rows.groupby("f")}
    for i in range(plan.n):
        f = f0 + i
        modes = []
        for t in tids:
            fs, B = arr[t]
            r = _box_at(fs, B, f, fps, stride)
            if r is None:
                continue
            box, mode, dist, speed = r
            if mode == "extrap":  # the requester is (mostly) not in the frame: nothing useful to release
                vis = box_area((max(0, box[0]), max(0, box[1]), min(plan.w, box[2]), min(plan.h, box[3]))) / max(1.0, box_area(box))
                if vis < MIN_EXTRAP_VISIBLE:
                    continue
            size = max(box[2] - box[0], box[3] - box[1])
            m = 0.10 * size + 4 + min(0.5 * size, speed * (stride + dist * (3 if mode == "extrap" else 1)))
            rect = (max(0, box[0] - m), max(0, box[1] - m), min(plan.w, box[2] + m), min(plan.h, box[3] + m))
            plan.rects[i].append(rect)
            plan.track_boxes[i].append(tuple(box))
            modes.append(mode)
        if modes:
            plan.modes[i] = "tracked" if "tracked" in modes else ("interp" if "interp" in modes else "extrap")
        # strangers from the nearest tracked sample (stride-aware)
        fn = int(round(f / stride)) * stride
        if fn < frange[0]:
            fn += stride
        elif fn > frange[1]:
            fn -= stride
        for o in by_f.get(fn, ()):
            if int(o[4]) in tids:
                continue
            ob = tuple(float(v) for v in o[:4])
            if any(iou(ob, tb) > DUP_IOU for tb in plan.track_boxes[i]):
                continue  # duplicate detection of the requester's own object
            for rect in plan.rects[i]:
                ia = inter_area(ob, rect)
                if ia >= 100 and ia >= 0.02 * box_area(ob):
                    plan.overlaps[i].append({"tid": int(o[4]), "name": str(o[5]), "box": [round(v, 1) for v in ob],
                                             "overlap_px": int(ia)})
                    break
    live = [i for i in range(plan.n) if plan.rects[i]]  # trim leading/trailing frames with nothing to release
    if live and (live[0] > 0 or live[-1] < plan.n - 1):
        lo, hi = live[0], live[-1] + 1
        plan.rects, plan.modes = plan.rects[lo:hi], plan.modes[lo:hi]
        plan.track_boxes, plan.overlaps = plan.track_boxes[lo:hi], plan.overlaps[lo:hi]
        plan.f0, plan.n = plan.f0 + lo, hi - lo
        plan.f1 = plan.f0 + plan.n - 1
    return plan


# ------------------------------------------------------------------------------------- frame access / masking


def read_frames(path: str | Path, indices: list[int]):
    """Yield (index, frame) for sorted source frame indices (one decode pass, seeking only when it pays)."""
    cap = cv2.VideoCapture(str(path))
    want = sorted(set(indices))
    if not want:
        return
    cur = want[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
    j = 0
    try:
        while j < len(want):
            if want[j] - cur > 120:  # big jump: seek
                cur = want[j]
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            while cur < want[j]:
                if not cap.grab():
                    return
                cur += 1
            ok, fr = cap.read()
            if not ok:
                return
            yield want[j], fr
            cur += 1
            j += 1
    finally:
        cap.release()


class Gpu:
    """All three models loaded ONCE inside ONE gpu_lock hold: the shared GPU is contended, so every extra lock acquisition
    cost minutes of queueing in testing. (Uses protokit's private loader; pk.detect_many/pose_many would re-queue each call.)"""

    def __init__(self) -> None:
        self.seg, self.det, self.pose = pk._yolo("seg"), pk._yolo("det_l"), pk._yolo("pose")

    def detect(self, fr: np.ndarray, conf: float = 0.20) -> list[dict[str, Any]]:
        r = self.det.predict(fr, conf=conf, imgsz=640, verbose=False)[0]
        return [{"name": self.det.names[c], "conf": cf, "box": xy} for c, cf, xy in
                zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]

    def poses(self, fr: np.ndarray, conf: float = 0.20) -> list[dict[str, Any]]:
        r = self.pose.predict(fr, conf=conf, imgsz=640, verbose=False)[0]
        kps = r.keypoints.data.tolist() if r.keypoints is not None else []
        return [{"conf": cf, "box": xy, "kp": kp} for cf, xy, kp in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist(), kps)]


def seg_refine(gpu: Gpu, src: Path, plan: Plan, cls: str, clip_conf: float = 0.25) -> dict[str, Any]:
    """Frames where a stranger overlaps the released rectangle: replace the rectangle with the requester's instance mask
    (YOLO26s-seg, dilated) minus strangers' masks. Everything stays masked by default if the mask cannot be matched."""
    idx = [i for i in range(plan.n) if plan.overlaps[i] and plan.rects[i]]
    stats = {"seg_frames": len(idx), "matched": 0, "unmatched_kept_rect": 0, "stranger_box_px_removed": 0}
    if not idx:
        return stats
    h, w = plan.h, plan.w
    need = [plan.f0 + i for i in idx]
    if True:
        m = gpu.seg  # local workaround: pk.seg_masks_many keeps every full-res mask in RAM; we consume per frame
        for f, fr in read_frames(src, need):
            i = f - plan.f0
            res = m.predict(fr, conf=clip_conf, imgsz=640, verbose=False, retina_masks=True)[0]
            dets = []
            if res.masks is not None:
                mk = res.masks.data.cpu().numpy() > 0.5
                for k, (c, xy) in enumerate(zip(res.boxes.cls.int().tolist(), res.boxes.xyxy.tolist())):
                    if m.names[c] in LEAK_CLASSES:
                        mm = mk[k]
                        if mm.shape != (h, w):
                            mm = cv2.resize(mm.astype(np.uint8), (w, h)) > 0
                        dets.append({"box": xy, "mask": mm, "name": m.names[c]})
            rect_mask = mask_of(plan.rects[i], h, w)
            req = np.zeros((h, w), bool)
            used: set[int] = set()
            keep_rect = np.zeros((h, w), bool)
            for tb, rect in zip(plan.track_boxes[i], plan.rects[i]):
                best, bj = 0.0, -1
                for j, d in enumerate(dets):
                    if j not in used and iou(d["box"], tb) > best:
                        best, bj = iou(d["box"], tb), j
                if bj >= 0 and best >= 0.4:
                    used.add(bj)
                    req |= dets[bj]["mask"]
                else:
                    keep_rect |= mask_of([rect], h, w)  # cannot match: keep the whole (dilated) rectangle
            # strangers. Person request: hide the stranger's whole (dilated) BOX, not the silhouette (measured: det_l still fired on a
            # blurred person-shaped hole, and did not on a box hole). Vehicle request with a PERSON stranger: hide only the person's
            # silhouette (dilated) so the requester's vehicle is not cut in half by a grey rectangle; the leak check still verifies it.
            sboxes = [tuple(o["box"]) for o in plan.overlaps[i]]
            for j, d in enumerate(dets):
                if j not in used and not any(iou(d["box"], tb) > DUP_IOU for tb in plan.track_boxes[i]):
                    sboxes.append(tuple(d["box"]))
            hole = np.zeros((h, w), bool)
            for x1, y1, x2, y2 in sboxes:
                sil = None
                if cls != "person":
                    cand_j = [(iou(d["box"], (x1, y1, x2, y2)), j) for j, d in enumerate(dets) if j not in used and d["name"] == "person"]
                    if cand_j and max(cand_j)[0] >= 0.4:
                        sil = dets[max(cand_j)[1]]["mask"]
                if sil is not None:
                    ks = int(max(9, 0.05 * (y2 - y1))) | 1
                    hole |= cv2.dilate(sil.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))) > 0
                    stats["silhouette_holes"] = stats.get("silhouette_holes", 0) + 1
                else:
                    hole[max(0, int(y1) - 8):min(h, int(y2) + 8), max(0, int(x1) - 8):min(w, int(x2) + 8)] = True
            size = max(tb[3] - tb[1] for tb in plan.track_boxes[i])
            k1 = int(max(5, 0.06 * size)) | 1
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
            region = (cv2.dilate(req.astype(np.uint8), kern) > 0) & rect_mask
            region |= keep_rect
            before = int((hole & region).sum())
            region &= ~hole
            plan.set_mask(i, region)
            stats["stranger_box_px_removed"] += before
            if keep_rect.any():
                stats["unmatched_kept_rect"] += 1
                plan.seg_state[i] = "rect_kept"
            else:
                stats["matched"] += 1
                plan.seg_state[i] = "seg"
    return stats


def pixelate(frame: np.ndarray) -> np.ndarray:
    """Heavy, cheap, irreversible: downscale by ~H/16 (a standing 200 px person becomes ~4 cells), blur, dim."""
    h, w = frame.shape[:2]
    b = max(16, int(round(h / 16)))
    small = cv2.resize(frame, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 0.8)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.convertScaleAbs(pix, alpha=0.55)


def compose(frame: np.ndarray, region: np.ndarray, tag: str | None, outline: bool = False) -> np.ndarray:
    out = pixelate(frame)
    cv2.copyTo(frame, region.astype(np.uint8), out)
    if outline and region.any():  # preview only: the released file carries no outline (it would draw the stranger's silhouette)
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (210, 190, 40), 1, cv2.LINE_AA)
    if tag:
        pk.label(out, tag, (6, out.shape[0] - 24), scale=0.42, bg=(20, 20, 20))
    return out


def export(src: Path, plan: Plan, fps: float, dst: Path, ref: str) -> dict[str, Any]:
    n_written = 0
    with pk.Mp4Writer(dst, fps) as wr:
        for f, fr in read_frames(src, list(range(plan.f0, plan.f1 + 1))):
            i = f - plan.f0
            tag = f"{ref} | REDACTED COPY | source t={pk.mmss(f / fps)}"
            wr.write(compose(fr, plan.region(i), tag))
            n_written += 1
    return {"frames_written": n_written}


# ---------------------------------------------------------------------------------------------- 5. leak check


def _read_export(dst: Path, indices: list[int]):
    yield from read_frames(dst, indices)


def judge(dets: list[dict[str, Any]], poses: list[dict[str, Any]], req_boxes: list, rects: list, cls: str,
          region: np.ndarray | None = None) -> dict[str, Any]:
    """Leak judgement on ONE frame. A person/vehicle detection is the requester only if it is of the requester's class family and
    overlaps the requester's tracked box (or its released rectangle) with IoU >= 0.4. EVERY other person/vehicle detection, wherever it
    sits (outside the region, or inside it, e.g. a stranger behind the requester, or a passenger), is a LEAK. A visible face
    keypoint on a non-requester person is a face leak. `region` (only passed for segmentation-refined frames, where the released
    pixels are the requester's own instance mask): a same-family detection whose box is >= 50% released pixels is a visible FRAGMENT
    of the requester (e.g. the part of a van left visible beside a hidden stranger), not a leak."""
    fam = ("person",) if cls == "person" else tuple(c for c in LEAK_CLASSES if c != "person")

    def is_req(box, name) -> bool:
        return name in fam and any(iou(box, b) >= 0.4 for b in list(req_boxes) + list(rects))

    leaks, weak, inside, frag = [], [], 0, 0
    for d in dets:
        if d["name"] not in LEAK_CLASSES:
            continue
        item = {"kind": "det_l:" + d["name"], "conf": round(d["conf"], 2), "box": [round(v) for v in d["box"]]}
        if is_req(d["box"], d["name"]):
            inside += d["conf"] >= LEAK_CONF
        elif region is not None and d["name"] in fam and out_frac(d["box"], region) <= 0.5:
            inside += d["conf"] >= LEAK_CONF
            frag += d["conf"] >= LEAK_CONF
        else:
            (leaks if d["conf"] >= LEAK_CONF else weak).append(item)
    face = 0
    for p in poses:
        if is_req(p["box"], "person"):
            continue
        item = {"kind": "pose:person", "conf": round(p["conf"], 2), "box": [round(v) for v in p["box"]]}
        if p["conf"] < LEAK_CONF:
            weak.append(item)
            continue
        if any(p["kp"][k][2] > 0.5 for k in FACE_KP):
            face += 1
            item["face_keypoint_visible"] = True
        leaks.append(item)
    return {"leaks": leaks, "weak": weak, "inside": int(inside), "face": face, "fragments": int(frag)}


def leak_check(gpu: Gpu, src: Path, dst: Path, plan: Plan, sample: list[int], cls: str) -> dict[int, dict]:
    """Independent models on the decoded EXPORT FILE and, as a control, on the same source frames."""
    per: dict[int, dict] = {}
    for (_, fo), (i, fe) in zip(read_frames(src, [plan.f0 + k for k in sample]), _read_export(dst, sample)):
        rb, rr = plan.track_boxes[i], plan.rects[i]
        reg = plan.region(i) if i in plan.masks else None
        do_, po_ = gpu.detect(fo), gpu.poses(fo)
        sj, ej = judge(do_, po_, rb, rr, cls, reg), judge(gpu.detect(fe), gpu.poses(fe), rb, rr, cls, reg)
        # corroboration: the export can only reveal what the same detectors could see in the unredacted frame. A detection that
        # matches no stranger in the source (e.g. a pose model hallucinating one giant person on a blur) is reported, not gating.
        sb = [l["box"] for l in sj["leaks"]]
        ej["unmatched"] = [l for l in ej["leaks"] if not any(iou(l["box"], b) >= 0.3 for b in sb)]
        ej["leaks"] = [l for l in ej["leaks"] if any(iou(l["box"], b) >= 0.3 for b in sb)]
        ej["face"] = sum(1 for l in ej["leaks"] if l.get("face_keypoint_visible"))
        per[i] = {"src": sj, "exp": ej, "src_raw": do_}
    return per


def patch_plan(plan: Plan, per: dict[int, dict], step: int) -> set[int]:
    """Remediation: cut every box the checker found out of the released region, in the frame it was found in and +/- one sampling
    step around it (the stranger is probably there in the unsampled neighbours too). Those frames join the human-review list."""
    patched: set[int] = set()
    for i, r in per.items():
        for l in r["exp"]["leaks"]:
            x1, y1, x2, y2 = l["box"]
            m = 0.12 * max(x2 - x1, y2 - y1) + 6
            hb = (x1 - m, y1 - m, x2 + m, y2 + m)
            for j in range(max(0, i - step), min(plan.n, i + step + 1)):
                plan.holes.setdefault(j, []).append(hb)
                ia = max((inter_area(hb, rc) for rc in plan.rects[j]), default=0.0)
                plan.overlaps[j].append({"tid": -2, "name": "found by leak check", "box": [round(v, 1) for v in (x1, y1, x2, y2)],
                                         "overlap_px": int(ia)})
                patched.add(j)
    return patched


def fault_injection(gpu: Gpu, src: Path, dst: Path, plan: Plan, per: dict[int, dict], cls: str, k: int = 12,
                    out_jpg: Path | None = None) -> dict[str, Any]:
    """SIMULATED fault: paste the SOURCE pixels of one stranger back into the export (as if the redaction had missed them),
    then ask the same checker. A checker that cannot catch this is worthless."""
    by_i: dict[int, dict] = {}
    for i, r in sorted(per.items()):
        for d in sorted(r["src_raw"], key=lambda d: -(d["box"][3] - d["box"][1])):
            if (d["name"] in LEAK_CLASSES and d["conf"] >= 0.5 and (d["box"][3] - d["box"][1]) >= 45
                    and out_frac(d["box"], plan.region(i)) > 0.9):
                by_i.setdefault(i, d)  # a stranger that the export currently hides almost completely
    if not by_i:
        return {"injected": 0, "caught": 0, "note": "no stranger visible in the sampled source frames: nothing to inject"}
    ids = sorted(by_i)
    pick = ids[:: max(1, len(ids) // k)][:k]
    caught, example, n_inj = 0, None, 0
    for (_, fo), (i, fe) in zip(read_frames(src, [plan.f0 + j for j in pick]), _read_export(dst, pick)):
        fr = fe.copy()
        x1, y1, x2, y2 = (int(v) for v in by_i[i]["box"])
        x1, y1, x2, y2 = max(0, x1 - 6), max(0, y1 - 6), min(fr.shape[1], x2 + 6), min(fr.shape[0], y2 + 6)
        fr[y1:y2, x1:x2] = fo[y1:y2, x1:x2]
        v = judge(gpu.detect(fr), gpu.poses(fr), plan.track_boxes[i], plan.rects[i], cls, plan.region(i) if i in plan.masks else None)
        hit = any(iou(l["box"], (x1, y1, x2, y2)) > 0.3 or inter_area(l["box"], (x1, y1, x2, y2)) > 0.3 * max(1.0, box_area(l["box"]))
                  for l in v["leaks"])
        n_inj += 1
        caught += int(hit)
        if hit and example is None:
            example = (fr.copy(), v["leaks"])
    if example is not None and out_jpg is not None:
        im, leaks = example
        for l in leaks:
            cv2.rectangle(im, tuple(l["box"][:2]), tuple(l["box"][2:]), (0, 0, 255), 3)
        pk.label(im, "SIMULATED FAULT: checker flags the un-redacted stranger -> LEAK", (8, 8), scale=0.6, bg=(0, 0, 200))
        pk.save_jpg(pk.fit(im, 960), out_jpg)
    return {"injected": n_inj, "caught": caught, "label": "SIMULATED (stranger pixels pasted back into the export)"}


# ---------------------------------------------------------------------------------------------- 6. risky frames


def risky_intervals(plan: Plan, fps: float) -> list[dict[str, Any]]:
    idx = [i for i in range(plan.n) if plan.overlaps[i]]
    out = []
    for i in idx:
        if out and i - out[-1]["_last"] <= 2:
            out[-1]["_last"] = i
            out[-1]["frames"] += 1
            out[-1]["peak_overlap_px"] = max(out[-1]["peak_overlap_px"], max(o["overlap_px"] for o in plan.overlaps[i]))
            out[-1]["tids"] |= {o["tid"] for o in plan.overlaps[i]}
            if max(o["overlap_px"] for o in plan.overlaps[i]) >= out[-1]["peak_overlap_px"]:
                out[-1]["rep"] = i
        else:
            out.append({"_first": i, "_last": i, "frames": 1, "rep": i, "tids": {o["tid"] for o in plan.overlaps[i]},
                        "peak_overlap_px": max(o["overlap_px"] for o in plan.overlaps[i])})
    for o in out:
        o["t_start"] = round((plan.f0 + o.pop("_first")) / fps, 2)
        o["t_end"] = round((plan.f0 + o.pop("_last")) / fps, 2)
        o["tids"] = sorted(o["tids"])
    return out


def risky_crops(src: Path, dst: Path, plan: Plan, intervals: list[dict], fps: float, outdir: Path, k: int = 8) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.jpg"):
        old.unlink()
    top = sorted(intervals, key=lambda o: (-o["frames"], -o["peak_overlap_px"]))[:k]
    top = sorted(top, key=lambda o: o["t_start"])
    reps = [o["rep"] for o in top]
    if not reps:
        return []
    orig = dict(read_frames(src, [plan.f0 + i for i in reps]))
    exp = dict(_read_export(dst, reps))
    files = []
    for n, o in enumerate(top):
        i = o["rep"]
        if plan.f0 + i not in orig or i not in exp:
            continue
        xs = [r[0] for r in plan.rects[i]] + [r[2] for r in plan.rects[i]] + [b for ov in plan.overlaps[i] for b in (ov["box"][0], ov["box"][2])]
        ys = [r[1] for r in plan.rects[i]] + [r[3] for r in plan.rects[i]] + [b for ov in plan.overlaps[i] for b in (ov["box"][1], ov["box"][3])]
        x1, y1, x2, y2 = max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20), min(plan.w, int(max(xs)) + 20), min(plan.h, int(max(ys)) + 20)
        so = orig[plan.f0 + i].copy()
        for ov in plan.overlaps[i]:
            cv2.rectangle(so, tuple(int(v) for v in ov["box"][:2]), tuple(int(v) for v in ov["box"][2:]), (60, 60, 255), 2)
        for r in plan.rects[i]:
            cv2.rectangle(so, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (80, 200, 60), 2)
        a, b = so[y1:y2, x1:x2], exp[i][y1:y2, x1:x2]
        s = max(1.0, 260 / max(a.shape[0], 1))
        s = min(s, 3.0)
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        pk.label(a, "SOURCE", (4, 4), scale=0.42)
        pk.label(b, f"RELEASED {pk.mmss((plan.f0 + i) / fps)}", (4, 4), scale=0.42)
        pad = np.full((a.shape[0], 6, 3), 24, np.uint8)
        p = OUT / outdir.parent.name / "risky" / f"r{n + 1:02d}_{(plan.f0 + i) / fps:06.2f}s.jpg"
        pk.save_jpg(np.hstack([a, pad, b]), p)
        files.append(str(p))
    return files


# --------------------------------------------------------------------------------------------------- hero image

FONT_B, FONT_R = "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/segoeui.ttf"
FONT_M = "C:/Windows/Fonts/consola.ttf"


def _font(size: int, bold: bool = False, mono: bool = False):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(FONT_M if mono else (FONT_B if bold else FONT_R), size)
    except Exception:
        return ImageFont.load_default()


def visible_strangers(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The strangers the source panel draws AND counts: det_l detections, de-duplicated (a car also seen as a truck is one box),
    at least MIN_STRANGER_PX tall. The label counts exactly these boxes."""
    keep: list[dict[str, Any]] = []
    for l in sorted((l for l in items if l["kind"].startswith("det_l")), key=lambda l: -l["conf"]):
        b = l["box"]
        if b[3] - b[1] >= MIN_STRANGER_PX and not any(iou(b, k["box"]) > 0.5 for k in keep):
            keep.append(l)
    return keep


def hero(path: Path, *, orig_bgr: np.ndarray, exp_bgr: np.ndarray, n_strangers: int, rec: dict, plan: Plan,
         intervals: list[dict], fps: float, duration: float, window: tuple[float, float], video: str) -> None:
    from PIL import Image, ImageDraw
    W, H = 1280, 800
    BG, PANEL, FG, MUTE = (11, 18, 32), (20, 30, 50), (240, 244, 250), (150, 165, 185)
    GREEN, AMBER, RED = (52, 211, 120), (251, 191, 36), (248, 90, 90)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 14), "RELEASE DESK", font=_font(30, True), fill=FG)
    d.text((24, 50), "CCTV subject access request: mask-by-default redaction with an independent leak check", font=_font(14), fill=MUTE)
    right = f"{rec['request']['ref']}  ·  {video}  ·  asked for: {rec['request']['subject']}"
    d.text((W - 24 - d.textlength(right, font=_font(15)), 22), right, font=_font(15), fill=MUTE)
    pw, ph, top = 620, 350, 84

    def place(bgr: np.ndarray, x: int):
        im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        s = min(pw / im.width, ph / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        d.rectangle([x, top, x + pw, top + ph], fill=(0, 0, 0))
        img.paste(im, (x + (pw - im.width) // 2, top + (ph - im.height) // 2))

    place(orig_bgr, 14)
    place(exp_bgr, 14 + pw + 32)
    note = rec["request"].get("stranger_note")
    src_txt = (f"SOURCE: {n_strangers} other {'person/vehicle' if n_strangers == 1 else 'people / vehicles'} in this frame"
               + (f" ({note})" if note and n_strangers == 1 else "") if n_strangers
               else "SOURCE: as filmed (no other person or vehicle in this frame)")
    for x, txt, col in ((14, src_txt, RED if n_strangers else MUTE), (14 + pw + 32, "RELEASED: only the data subject", GREEN)):
        d.rounded_rectangle([x + 8, top + 8, x + 8 + d.textlength(txt, font=_font(19, True)) + 22, top + 40], 8, fill=(10, 14, 24))
        d.rounded_rectangle([x + 8, top + 8, x + 8 + d.textlength(txt, font=_font(19, True)) + 22, top + 40], 8, outline=col, width=2)
        d.text((x + 19, top + 11), txt, font=_font(19, True), fill=col)
    ax = 14 + pw + 4  # arrow
    d.polygon([(ax, top + ph // 2 - 14), (ax + 24, top + ph // 2), (ax, top + ph // 2 + 14)], fill=FG)

    # verdict banner UNDER the panels (never on top of the released subject)
    lk = rec["leak_check"]
    leaks = lk["leaks_in_export"]
    n_chk, n_rel = lk["frames_checked"], rec["trim"]["frames_released"]
    every = "every frame" if n_chk >= n_rel else f"every {lk['sample_step']}th frame + all overlap frames"
    by, bh_ = top + ph + 12, 66
    sc = GREEN if leaks == 0 else RED
    d.rounded_rectangle([14, by, 14 + 760, by + bh_], 12, fill=(8, 22, 16) if leaks == 0 else (30, 10, 10))
    d.rounded_rectangle([14, by, 14 + 760, by + bh_], 12, outline=sc, width=4)
    big = f"{leaks} strangers detected in {n_chk} / {n_rel} export frames" if leaks == 0 else f"BLOCKED: {leaks} leaks in the export"
    d.text((32, by + 8), big, font=_font(28, True), fill=sc)
    d.text((32, by + 42), f"checked: {every} of the exported file · 2 independent detectors (yolo26l + pose) · detector evidence, not a proof",
           font=_font(13), fill=(205, 220, 212))
    if len(lk["rounds"]) > 1:
        r1, r2 = lk["rounds"][0], lk["rounds"][1]
        badge = (f"CAUGHT AND FIXED: round 1 found {r1['leaks']} leak{'s' if r1['leaks'] != 1 else ''}, "
                 f"cut out and re-exported; round 2 found {r2['leaks']}")
        bc = AMBER
    else:
        badge = "No leak found in round 1: nothing to patch"
        bc = MUTE
    bx = 14 + 760 + 14
    d.rounded_rectangle([bx, by, W - 14, by + bh_], 12, fill=PANEL, outline=bc, width=2)
    words, lines_, cur = badge.split(), [], ""
    for wd in words:  # wrap the badge text
        if d.textlength(cur + " " + wd, font=_font(14, True)) > (W - 14 - bx - 28):
            lines_.append(cur.strip())
            cur = ""
        cur += " " + wd
    lines_.append(cur.strip())
    for j, ln in enumerate(lines_[:3]):
        d.text((bx + 14, by + 7 + 20 * j), ln, font=_font(14, True), fill=bc)

    # tiles
    ty, th, tw, gap = by + bh_ + 16, 118, 300, 16
    rel, clip = rec["trim"]["seconds_released"], rec["source"]["duration_s"]
    manual_rel = rel * HO_MIN_PER_MIN  # seconds of officer time if done by hand at 9 min per min of footage
    cp = rec["compute"]
    net = cp["total"]["net_s"]
    nint = len(intervals)
    faster = net <= manual_rel
    tiles = [
        ("LEAK CHECK", f"{leaks} leaks", [f"{n_chk} of {n_rel} export frames, 2 models",
                                          (f"strangers in {lk['strangers_in_source_frames']} source frames" + (", none in the frame shown" if not n_strangers else ""))
                                          if lk["strangers_in_source_frames"] else "no stranger visible in these source frames"],
         GREEN if leaks == 0 else RED),
        ("RELEASED", f"{rel:.1f} s", [f"of {clip:.1f} s of footage", f"subject = track #{','.join(map(str, rec['requester']['tids']))}, {rec['requester']['median_height_px']:.0f} px tall"], FG),
        ("HUMAN REVIEW", f"{rec['risky']['frames']} frames", [f"{nint} overlap interval{'s' if nint != 1 else ''} forced to the officer", "crops in the report"],
         AMBER if rec["risky"]["frames"] else GREEN),
        ("STOPWATCH (END TO END)", f"{net:.0f} s machine" if net < 600 else f"{net / 60:.1f} min machine",
         [f"by hand ≈ {manual_rel:.0f} s ({manual_rel / 60:.1f} min) for these {rel:.1f} s*",
          f"export {cp['export_x_realtime']:.1f}× real time · review not counted" if faster else
          f"SLOWER than scaled manual · export {cp['export_x_realtime']:.1f}× real time"],
         FG if faster else AMBER),
    ]
    for k, (cap, big_, subs, col) in enumerate(tiles):
        x = 14 + k * (tw + gap)
        d.rounded_rectangle([x, ty, x + tw, ty + th], 12, fill=PANEL)
        d.text((x + 16, ty + 10), cap, font=_font(13, True), fill=MUTE)
        d.text((x + 16, ty + 26), big_, font=_font(34 if k == 3 else 38, True), fill=col)
        for j, sub in enumerate(subs):
            d.text((x + 16, ty + 78 + 17 * j), sub, font=_font(12 if k in (0, 3) else 13), fill=MUTE)
    # timeline over the requested window
    tl_y = ty + th + 36
    wlo, whi = window
    d.text((16, tl_y - 24), f"TIMELINE OF THE REQUEST WINDOW  {pk.mmss(wlo)[:5]} to {pk.mmss(whi)[:5]}", font=_font(13, True), fill=MUTE)
    x0, x1 = 16, W - 16
    sx = (x1 - x0) / max(whi - wlo, 1e-6)
    tx = lambda t: x0 + (min(max(t, wlo), whi) - wlo) * sx  # noqa: E731
    d.rounded_rectangle([x0, tl_y, x1, tl_y + 26], 6, fill=(62, 78, 108))
    ra, rb = plan.f0 / fps, (plan.f1 + 1) / fps
    d.rectangle([tx(ra), tl_y, max(tx(ra) + 2, tx(rb)), tl_y + 26], fill=GREEN)
    for o in intervals:
        d.rectangle([tx(o["t_start"]), tl_y, max(tx(o["t_start"]) + 2, tx(o["t_end"] + 1 / fps)), tl_y + 26], fill=AMBER)
    d.text((x0, tl_y + 30), pk.mmss(wlo)[:6], font=_font(12), fill=MUTE)
    d.text((x1 - 44, tl_y + 30), pk.mmss(whi)[:6], font=_font(12), fill=MUTE)
    lx = x0 + 90
    for col, txt in (((62, 78, 108), "requested window: masked, not released"), (GREEN, "released seconds"), (AMBER, "forced human review")):
        d.rectangle([lx, tl_y + 33, lx + 12, tl_y + 45], fill=col)
        d.text((lx + 18, tl_y + 30), txt, font=_font(12), fill=MUTE)
        lx += 18 + d.textlength(txt, font=_font(12)) + 34
    foot = (f"* machine time net of shared-GPU queueing; Home Office ACE figure (~9 h per 60 min) scaled to this release, not measured here. "
            f"src sha256 {rec['source']['sha256'][:16]}…  export sha256 {rec['export']['sha256'][:16]}…")
    d.text((16, H - 24), foot, font=_font(12), fill=MUTE)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)


# -------------------------------------------------------------------------------------------------- the pipeline


def run_one(name: str, *, cls: str | None = None, window: tuple[float, float] | None = None, tids: list[int] | None = None,
            max_sample: int = 400, tag: str | None = None) -> dict[str, Any]:
    cfg = HARD[tag] if tag else DEFAULTS.get(name, {})
    cls = cls or cfg.get("cls", "person")
    meta = pk.info(name)
    window = window or cfg.get("window") or (0.0, meta["duration"])
    window = (max(0.0, window[0]), min(meta["duration"], window[1]))
    subject = cfg.get("subject", f"a {cls}")
    ref = f"SAR-DEMO-{abs(hash(name)) % 9000 + 1000}"
    ref = {"retail_cctv": "SAR-DEMO-0001", "lisbon_parking": "SAR-DEMO-0002", "person_bicycle_car_detection": "SAR-DEMO-0003"}.get(name, ref)
    ref = cfg.get("ref", ref)
    key = f"{name}__{tag}" if tag else name  # results key and output folder
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    src = pk.video_path(name)
    fps = meta["fps"]
    T = Timer()
    print(f"[{name}] request: {subject} ({cls}) window {window[0]:.1f}-{window[1]:.1f}s", flush=True)

    with T.stage("0_shot_cuts"):
        f_lo, f_hi = int(round(window[0] * fps)), min(meta["frames"] - 1, int(round(window[1] * fps)))
        cuts = detect_cuts(src, f_lo, f_hi)
    with T.stage("1_candidates_sql"):
        tr = pk.tracks(name, verbose=False)
        seg, trk = build_tables(name, tr, window, cls, cuts)
        cand = candidate_sql(seg, trk, window, cls, min_dur=cfg.get("min_dur", 3.0))
    auto = not tids
    if auto:
        if cand["tracks"].empty:
            raise SystemExit(f"[{name}] no candidate {cls} track in the window")
        tids = [int(cand["tracks"].iloc[0]["tid"])]
        pick = cand["tracks"].iloc[0]
    else:
        sub = trk[trk["tid"].isin(tids)].sort_values("dur", ascending=False)
        if sub.empty:
            raise SystemExit(f"[{name}] tid {tids} not in the window")
        pick = sub.iloc[0]
    frange = (int(pick["shot_from"]), int(pick["shot_to"]))
    cand_row = trk[trk["tid"].isin(tids) & (trk["shot"] == int(pick["shot"]))].to_dict("records")
    with T.stage("1b_candidate_thumbnails"):
        candidate_sheet(name, tr, cand["tracks"], tids, window, out / "candidates.jpg")
    bh = float(tr.rows[tr.rows["tid"].isin(tids) & (tr.rows["f"] >= frange[0]) & (tr.rows["f"] <= frange[1])
                       & (tr.rows["t"] >= window[0]) & (tr.rows["t"] <= window[1])]["bh"].median())
    print(f"[{name}] candidates: {len(cand['tracks'])}; requester tid {tids} (auto={auto}) median height {bh:.0f}px", flush=True)

    with T.stage("2_trim_and_plan"):
        plan = make_plan(tr, tids, window, fps, meta["frames"], frange)
    dst = out / "export.mp4"
    # sample for the leak check: regular stride + a share of risky frames (decided before anything is exported)
    step = max(1, math.ceil(plan.n / max_sample))
    sample = set(range(0, plan.n, step))
    risky_idx = [i for i in range(plan.n) if plan.overlaps[i]]
    sample |= set(risky_idx[:: max(1, len(risky_idx) // 50)]) if risky_idx else set()
    sample = sorted(sample)

    # one GPU-lock hold for segmentation, export, leak check and fault injection (the shared GPU queue is the slow part)
    t_q = time.time()
    with pk.gpu_lock():
        waited = time.time() - t_q
        T.stages["gpu_lock_wait"] = {"wall_s": round(waited, 2), "gpu_lock_wait_s": round(waited, 2), "net_s": 0.0}
        with T.stage("3_load_models"):
            gpu = Gpu()
        with T.stage("3_seg_refine"):
            seg_stats = seg_refine(gpu, src, plan, cls)
        with T.stage("4_export"):
            export(src, plan, fps, dst, ref)
        rel_s = plan.n / fps
        exp_s = T.stages["4_export"]["wall_s"]
        print(f"[{name}] exported {plan.n} frames ({rel_s:.1f}s) in {exp_s:.1f}s = {rel_s / max(exp_s, 1e-6):.1f}x real time", flush=True)
        cap = cv2.VideoCapture(str(dst))  # verify the file we are about to certify
        n_exp = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        with T.stage("5_leak_check"):
            per = leak_check(gpu, src, dst, plan, sample, cls)
        rounds = [{"round": 1, "frames_checked": len(per), "leaks": sum(len(r["exp"]["leaks"]) for r in per.values()),
                   "leak_frames": sorted(i for i, r in per.items() if r["exp"]["leaks"]),
                   "example": [r["exp"]["leaks"][0] for i, r in sorted(per.items()) if r["exp"]["leaks"]][:2]}]
        if rounds[0]["leaks"]:
            with T.stage("5c_patch_reexport_recheck"):
                patched = patch_plan(plan, per, step)
                export(src, plan, fps, dst, ref)
                sample = sorted(set(sample) | patched)  # round 2: everything round 1 saw, plus every patched frame
                per = leak_check(gpu, src, dst, plan, sample, cls)
            rounds.append({"round": 2, "frames_checked": len(per), "patched_frames": len(patched),
                           "leaks": sum(len(r["exp"]["leaks"]) for r in per.values()),
                           "leak_frames": sorted(i for i, r in per.items() if r["exp"]["leaks"])})
            risky_idx = [i for i in range(plan.n) if plan.overlaps[i]]
            print(f"[{name}] round 1 found {rounds[0]['leaks']} leaks -> patched {len(patched)} frames -> round 2: {rounds[1]['leaks']} leaks", flush=True)
        with T.stage("5b_fault_injection"):
            fi = fault_injection(gpu, src, dst, plan, per, cls, out_jpg=out / "fault_injection.jpg")
        del gpu
        pk._unload()

    leak_frames = {i: r["exp"]["leaks"] for i, r in per.items() if r["exp"]["leaks"]}
    src_strangers = {i: r["src"]["leaks"] for i, r in per.items() if r["src"]["leaks"]}
    req_seen = sum(1 for r in per.values() if r["exp"]["inside"] > 0)
    req_seen_src = sum(1 for r in per.values() if r["src"]["inside"] > 0)
    face_leaks = sum(r["exp"]["face"] for r in per.values())
    lk = {"detectors": ["yolo26l (det_l)", "yolo26s-pose (face keypoints > 0.5)"], "gating_confidence": LEAK_CONF, "run_confidence": 0.20,
          "rule": f"LEAK = any person/vehicle detection or pose in the export with conf >= {LEAK_CONF} that is not the requester (requester = same class family and IoU >= 0.4 with the requester's tracked box or released rectangle); weaker hits (0.20-{LEAK_CONF}) are listed as weak_detections, not gating",
          "frames_checked": len(per), "sample_step": step, "risky_frames_included": len(set(risky_idx) & set(sample)),
          "leaks_in_export": int(sum(len(v) for v in leak_frames.values())), "leak_frames": sorted(leak_frames),
          "face_keypoint_leaks": int(face_leaks), "rounds": rounds,
          "unmatched_export_detections_not_gating": int(sum(len(r["exp"]["unmatched"]) for r in per.values())),
          "weak_detections_in_export_below_gate": int(sum(len(r["exp"]["weak"]) for r in per.values())),
          "weak_detections_in_source_below_gate": int(sum(len(r["src"]["weak"]) for r in per.values())),
          "requester_fragments_not_counted_as_leaks": int(sum(r["exp"]["fragments"] for r in per.values())),
          "strangers_in_source_total": int(sum(len(v) for v in src_strangers.values())),
          "strangers_in_source_frames": len(src_strangers),
          "control_requester_redetected_in_export_pct": round(100 * req_seen / max(1, len(per)), 1),
          "control_requester_detected_in_source_pct": round(100 * req_seen_src / max(1, len(per)), 1),
          "example_leaks": [leak_frames[i][:2] for i in sorted(leak_frames)[:3]]}

    (out / "leaks.json").write_text(json.dumps({str(i): {"export_leaks": r["exp"]["leaks"], "export_weak": r["exp"]["weak"], "export_unmatched": r["exp"]["unmatched"],
                                                        "source_strangers": len(r["src"]["leaks"])} for i, r in sorted(per.items())}),
                                    encoding="utf-8")
    ivs = risky_intervals(plan, fps)
    with T.stage("6_risky_crops"):
        crops = risky_crops(src, dst, plan, ivs, fps, out / "risky", k=8)

    # hero frame: a clean frame with the most strangers in the source
    def hero_score(i: int, r: dict) -> float:
        ns = len(visible_strangers(r["src"]["leaks"]))
        tb = plan.track_boxes[i][0]
        em = 0.03 * min(plan.w, plan.h)
        edge = tb[0] < em or tb[1] < em or tb[2] > plan.w - em or tb[3] > plan.h - em
        area = math.sqrt(box_area(tb) / (plan.w * plan.h))
        return min(ns, 6) + 9 * area - 0.5 * len(plan.overlaps[i]) - 3 * edge
    cands = [(hero_score(i, r), i) for i, r in per.items()
             if plan.modes[i] == "tracked" and plan.track_boxes[i] and not r["exp"]["leaks"] and r["exp"]["inside"] > 0]
    best_i = max(cands)[1] if cands else sample[len(sample) // 2]
    fo = dict(read_frames(src, [plan.f0 + best_i]))[plan.f0 + best_i].copy()
    fe = dict(_read_export(dst, [best_i]))[best_i]
    shown = visible_strangers(per[best_i]["src"]["leaks"])  # the label counts exactly the boxes drawn
    n_str = len(shown)
    for l in shown:
        cv2.rectangle(fo, tuple(l["box"][:2]), tuple(l["box"][2:]), (60, 60, 255), 2)
    for rr in plan.rects[best_i]:
        cv2.rectangle(fo, (int(rr[0]), int(rr[1])), (int(rr[2]), int(rr[3])), (80, 220, 60), 3)
    pk.save_jpg(fe, out / "export_frame.jpg")
    fe_vis = fe.copy()  # preview only: outline of the released region (the exported file carries none)
    cs, _ = cv2.findContours(plan.region(best_i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fe_vis, cs, -1, (210, 190, 40), 2, cv2.LINE_AA)

    modes = {m: plan.modes.count(m) for m in ("tracked", "interp", "extrap", "gap")}
    exp_total_n = max(1, plan.n)
    comp = T.total()
    manual_rel_s = rel_s * HO_MIN_PER_MIN
    manual_clip_s = meta["duration"] * HO_MIN_PER_MIN
    req_text = cfg.get("request", f"Footage of {subject}")
    due = dt.date(2026, 11, 1)
    rec = {
        "ref": ref, "tool": "Release Desk prototype",
        "request": {"ref": ref, "subject": subject, "text": req_text, "stranger_note": cfg.get("stranger_note", ""), "class": cls, "window_s": list(window),
                    "window": f"{pk.mmss(window[0])}-{pk.mmss(window[1])}", "received": "2026-10-01 (SIMULATED)",
                    "due": f"{due.isoformat()} (ICO: one calendar month, SIMULATED date)"},
        "source": {"video": name, "file": str(src).replace("\\", "/"), "sha256": pk.sha256(src), "duration_s": round(meta["duration"], 2),
                   "fps": round(fps, 3), "resolution": f"{meta['w']}x{meta['h']}", "credit": pk.credit(name)},
        "candidates": {"segments_in_window": int(len(cand["segments"])), "tracks_listed": int(len(cand["tracks"])),
                       "sql_segments": cand["q_seg"], "sql_tracks": cand["q_trk"],
                       "relaxed_filters": bool(cand.get("relaxed", False))},
        "requester": {"tids": tids, "auto_picked_for_test": auto, "median_height_px": round(bh, 1),
                      "track_stats": cand_row,
                      "pick_rule": ("auto: highest score = min(dur,30)*conf*(1-overlap)*(1-edge)*min(h,250)/250 among tracks with height>=60px, "
                                    "coverage>=0.8, 3<=dur<=45 s" if auto else "officer supplied --tid")},
        "trim": {"first_seen_s": round(float(pick["t_first"]), 2), "last_seen_s": round(float(pick["t_last"]), 2),
                 "shot_cuts_in_window": len(cuts) - 1, "shot_s": [round(frange[0] / fps, 2), round(min(frange[1], meta["frames"] - 1) / fps, 2)],
                 "pad_s": PAD_S, "released_from_s": round(plan.f0 / fps, 2), "released_to_s": round((plan.f1 + 1) / fps, 2),
                 "frames_released": plan.n, "seconds_released": round(rel_s, 2),
                 "fraction_of_clip": round(rel_s / meta["duration"], 3)},
        "masking": {"default": "mask-by-default: everything outside the released region is pixelated (cells = frame height/16), blurred and dimmed",
                    "region": "requester box dilated by 10% + 4 px + motion margin; YOLO26s-seg instance mask (dilated, minus strangers) on overlap frames",
                    "frame_modes": modes, "frames_fully_masked_no_requester": modes["gap"],
                    "seg_refinement": seg_stats, "audio": "dropped"},
        "leak_check": lk,
        "fault_injection": fi,
        "risky": {"frames": len(risky_idx), "fraction_of_released": round(len(risky_idx) / exp_total_n, 3), "intervals": len(ivs),
                  "interval_list": [{k: v for k, v in o.items() if k != "rep"} for o in ivs],
                  "rule": "another person/vehicle box overlaps the released region (>= 100 px^2 and >= 2% of its box)",
                  "crops": [str(Path(c).relative_to(out)).replace("\\", "/") for c in crops],
                  "human_review": "REQUIRED before release for every listed interval"},
        "export": {"file": "export.mp4", "sha256": pk.sha256(dst), "frames": n_exp, "frames_expected": plan.n,
                   "bytes": dst.stat().st_size},
        "compute": {"stages": T.stages, "total": comp, "export_s": exp_s,
                    "export_x_realtime": round(rel_s / max(exp_s, 1e-6), 2),
                    "pipeline_net_x_realtime": round(rel_s / max(comp["net_s"], 1e-6), 2),
                    "machine": "shared laptop: ~15 other jobs on the same GPU/CPU; gpu_lock_wait excluded in 'net'",
                    "home_office_baseline": {"figure": "~9 h to redact ~60 min of footage (Home Office ACE case study) = 9 min per min",
                                             "scaled_to_released_seconds_min": round(manual_rel_s / 60, 1),
                                             "scaled_to_whole_clip_min": round(manual_clip_s / 60, 1),
                                             "note": "scaled from a published figure, NOT measured here; human review time not measured"}},
        "sign_off": {"officer": "PENDING", "status": "NOT RELEASED until the risky intervals are reviewed"},
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }
    # verdict + kill test
    verdict = ("BLOCKED" if lk["leaks_in_export"] else ("REVIEW" if risky_idx else "CLEAR"))
    rec["verdict"] = verdict
    rec["kill_test"] = {"criterion": "requester >= ~60 px tall AND export at >= real time",
                        "requester_height_px": round(bh, 1), "export_x_realtime": rec["compute"]["export_x_realtime"],
                        "passed": bool(bh >= 60 and rec["compute"]["export_x_realtime"] >= 1.0)}
    (out / "release_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    # per-frame regions for the smoke test (rects only)
    (out / "regions.json").write_text(json.dumps({"f0": plan.f0, "fps": fps, "modes": plan.modes,
                                                  "rects": [[[round(v, 1) for v in r] for r in rs] for rs in plan.rects],
                                                  "holes": {str(k): [[round(v, 1) for v in b] for b in bs] for k, bs in plan.holes.items()},
                                                  "sample": sample}), encoding="utf-8")

    hero(out / "hero.jpg", orig_bgr=fo, exp_bgr=fe_vis, n_strangers=n_str, rec=rec, plan=plan, intervals=ivs, fps=fps,
         duration=meta["duration"], window=window, video=name)
    write_report(out, name, rec, cand, ivs, tr, seg)
    print(f"[{name}] verdict {verdict}: leaks {lk['leaks_in_export']}, strangers in source (sampled frames) {lk['strangers_in_source_total']}, "
          f"risky {len(risky_idx)} frames, fault-inj {fi.get('caught')}/{fi.get('injected')}, total net {comp['net_s']}s", flush=True)
    return rec


def write_report(out: Path, name: str, rec: dict, cand: dict, ivs: list[dict], tr: pk.Tracks, seg: pd.DataFrame) -> None:
    lk, tm, cp = rec["leak_check"], rec["trim"], rec["compute"]
    verdict_txt = {"CLEAR": "CLEAR: no leaks, no overlap frames", "REVIEW": "LEAK CHECK PASSED, HUMAN REVIEW REQUIRED",
                   "BLOCKED": "BLOCKED: leaks found"}[rec["verdict"]]
    col = {"CLEAR": "#2e9e5b", "REVIEW": "#e0a100", "BLOCKED": "#d64545"}[rec["verdict"]]
    R = pk.Report(out / "report.html", f"Release record {rec['ref']}", f"{rec['request']['text']}. Window {rec['request']['window']}.")
    R.credit(name)
    R.html(f'<p><span class="tag" style="background:{col};font-size:15px;padding:4px 14px">{verdict_txt}</span></p>')
    R.image(out / "hero.jpg", "Original frame (left) vs the frame that would be released (right). Same frame number.")
    R.kv({"Released": f"{rec['trim']['seconds_released']} s of {rec['source']['duration_s']} s",
          "Leaks in export": f"{lk['leaks_in_export']} / {lk['frames_checked']} frames",
          "Strangers in source": f"{lk['strangers_in_source_frames']} / {lk['frames_checked']} frames",
          "Human review": f"{rec['risky']['frames']} frames, {rec['risky']['intervals']} intervals",
          "Export speed": f"{cp['export_x_realtime']}x real time", "Requester height": f"{rec['requester']['median_height_px']} px"})
    R.h2("1. Candidates (SQL over the 5 s segment table and the track table)")
    R.text("Segments in the requested window that contain the requested class:")
    R.html(f"<pre>{cand['q_seg']}</pre>")
    R.table(cand["segments"].head(12))
    R.text("Candidate tracks, ranked (the officer would pick from the thumbnails; tests auto-pick the top row and say so):")
    R.html(f"<pre>{cand['q_trk']}</pre>")
    R.table(cand["tracks"])
    if (out / "candidates.jpg").exists():
        R.image(out / "candidates.jpg", f"Candidate thumbnails; green outline = picked ({'auto-picked for the test' if rec['requester']['auto_picked_for_test'] else 'officer'}).")
    R.h2("2. Export (mask-by-default)")
    R.video(out / "export.mp4", f"{rec['export']['frames']} frames, audio dropped. Thin yellow outline = released region.")
    R.kv({"Trim": f"{pk.mmss(tm['released_from_s'])} to {pk.mmss(tm['released_to_s'])}", "Frames tracked / interp / extrap / gap":
          "/".join(str(rec['masking']['frame_modes'][k]) for k in ('tracked', 'interp', 'extrap', 'gap')),
          "Seg-mask frames": rec['masking']['seg_refinement']['matched']})
    R.h2("3. Leak check (independent models on the exported file)")
    R.json({k: v for k, v in lk.items() if k != "example_leaks"}, "Detectors run on the decoded export.mp4, not on the pipeline's own frames.")
    if (out / "fault_injection.jpg").exists():
        R.image(out / "fault_injection.jpg", f"SIMULATED fault: {rec['fault_injection'].get('caught')}/{rec['fault_injection'].get('injected')} injected leaks caught by the checker.")
    R.h2("4. Forced human review")
    R.text(f"{rec['risky']['frames']} frames in {rec['risky']['intervals']} intervals have another person or vehicle overlapping the released "
           "region (tid -2 = a stranger the leak check itself found). Crops: left = source (green = released rectangle, red = stranger), "
           "right = what is released.")
    R.table([{"from": pk.mmss(o["t_start"]), "to": pk.mmss(o["t_end"]), "frames": o["frames"], "strangers (tid)": ",".join(map(str, o["tids"])),
              "peak overlap px": o["peak_overlap_px"]} for o in ivs])
    for c in rec["risky"]["crops"]:
        R.image(out / c, c)
    R.h2("5. Release record")
    R.text(f"Source sha256 {rec['source']['sha256']}")
    R.text(f"Export sha256 {rec['export']['sha256']}")
    hb = cp["home_office_baseline"]
    R.text(f"Machine time {cp['total']['net_s']} s net of shared-GPU waits ({cp['total']['wall_s']} s wall); export alone {cp['export_s']} s. "
           f"Home Office baseline (about 9 h per 60 min, scaled; not measured here): {hb['scaled_to_released_seconds_min']} min by hand for the "
           f"released seconds, {hb['scaled_to_whole_clip_min']} min for the whole clip. Review time for the flagged frames is not included.")
    R.json({k: rec[k] for k in ("request", "requester", "trim", "masking", "sign_off")}, "Release record (JSON on disk: release_record.json)")
    R.save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run the default test set")
    ap.add_argument("--video", help="catalogue name of one clip")
    ap.add_argument("--tid", help="requester track id(s), comma separated (officer pick); default: auto-pick for tests")
    ap.add_argument("--window", help="request window 'start:end' seconds (default per clip)")
    ap.add_argument("--cls", help="class of the requester (person, car, ...)")
    ap.add_argument("--max-sample", type=int, default=400, help="max export frames for the leak check (default: every frame up to 400)")
    ap.add_argument("--hard", action="store_true", help="also run the hard-case appendix requests (lisbon van with a stranger in front)")
    ap.add_argument("--list", action="store_true", help="only print the candidate tracks (step 1)")
    a = ap.parse_args()
    names = DEFAULT_SET if a.all else ([a.video] if a.video else [])
    if not names and not a.hard:
        ap.error("give --all, --video NAME or --hard")
    win = tuple(float(x) for x in a.window.split(":")) if a.window else None
    tids = [int(x) for x in a.tid.split(",")] if a.tid else None
    results: dict[str, Any] = {}
    t0 = time.time()
    for n in names:
        if a.list:
            cfg = DEFAULTS.get(n, {})
            tr = pk.tracks(n, verbose=False)
            w = win or cfg.get("window") or (0, tr.duration)
            cls = a.cls or cfg.get("cls", "person")
            m_ = pk.info(n)
            cuts = detect_cuts(pk.video_path(n), int(w[0] * m_["fps"]), min(m_["frames"] - 1, int(w[1] * m_["fps"])))
            seg, trk = build_tables(n, tr, w, cls, cuts)
            c = candidate_sql(seg, trk, w, cls)
            print(c["q_seg"], "\n", c["segments"].head(8).to_string(), "\n", c["q_trk"], "\n", c["tracks"].to_string())
            continue
        results[n] = run_one(n, cls=a.cls, window=win, tids=tids, max_sample=a.max_sample)
    if a.hard and not a.list:
        for tg, cfg in HARD.items():
            results[f"{cfg['video']}__{tg}"] = run_one(cfg["video"], max_sample=a.max_sample, tag=tg)
    if results:
        write_results_json(results, time.time() - t0, merge=True)


def write_results_json(results: dict[str, Any], wall: float, merge: bool) -> None:
    p = HERE / "results.json"
    old = {}
    if merge and p.exists():
        with contextlib.suppress(Exception):
            old = json.loads(p.read_text(encoding="utf-8")).get("videos", {})
    vids = {**old}
    for n, r in results.items():
        vids[n] = {
            "video": r["source"]["video"], "request": r["request"]["text"], "class": r["request"]["class"], "window_s": r["request"]["window_s"],
            "stamp": (f"{r['leak_check']['leaks_in_export']} strangers detected in {r['leak_check']['frames_checked']}/{r['trim']['frames_released']} "
                      "export frames (detector evidence on the checked frames, not a proof)"),
            "every_frame_checked": r["leak_check"]["frames_checked"] >= r["trim"]["frames_released"],
            "requester_tids": r["requester"]["tids"], "auto_picked_for_test": r["requester"]["auto_picked_for_test"],
            "requester_height_px": r["requester"]["median_height_px"],
            "clip_s": r["source"]["duration_s"], "seconds_released": r["trim"]["seconds_released"],
            "fraction_of_clip": r["trim"]["fraction_of_clip"], "frames_released": r["trim"]["frames_released"],
            "frame_modes": r["masking"]["frame_modes"], "seg_refinement": r["masking"]["seg_refinement"],
            "leak_check": {k: v for k, v in r["leak_check"].items() if k not in ("detectors", "rule", "example_leaks")},
            "fault_injection_SIMULATED": r["fault_injection"],
            "risky_frames": r["risky"]["frames"], "risky_intervals": r["risky"]["intervals"],
            "risky_fraction_of_released": r["risky"]["fraction_of_released"],
            "export_s": r["compute"]["export_s"], "export_x_realtime": r["compute"]["export_x_realtime"],
            "pipeline_net_s": r["compute"]["total"]["net_s"], "pipeline_wall_s": r["compute"]["total"]["wall_s"],
            "pipeline_net_x_realtime": r["compute"]["pipeline_net_x_realtime"],
            "stages": r["compute"]["stages"],
            "home_office_scaled_min_released": r["compute"]["home_office_baseline"]["scaled_to_released_seconds_min"],
            "home_office_scaled_min_clip": r["compute"]["home_office_baseline"]["scaled_to_whole_clip_min"],
            "source_sha256": r["source"]["sha256"], "export_sha256": r["export"]["sha256"],
            "verdict": r["verdict"], "kill_test": r["kill_test"],
        }
    pk.write_results(p, {"idea": "RELEASE DESK", "run_wall_s": round(wall, 1), "videos": vids})


if __name__ == "__main__":
    main()
