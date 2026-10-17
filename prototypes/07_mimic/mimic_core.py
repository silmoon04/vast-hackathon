"""MIMIC core: camera-motion windows, tracklets, motion descriptors, the VastDB-style table, kNN + DTW matching.

Everything here is CPU/numpy. The only inputs are the cached YOLO26s+ByteTrack tracks from protokit (the local
stand-in for per-frame YOLO11s detections on the day) and the video frames (for camera-motion estimation).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import protokit as pk  # noqa: E402

CACHE = HERE / "_cache"
N = 32                       # points per resampled trajectory
MOVERS = ("person", "bicycle", "car", "motorcycle", "bus", "truck", "skateboard", "dog", "cat", "horse",
          "boat", "airplane", "bird")
DEFAULT_CLIPS = ["retail_cctv", "lisbon_parking", "ptz_parking", "rochester_night", "yonge_dundas", "motorway_a40",
                 "dashcam_urban", "elephants_dream", "skate_fall", "floor_lie", "assist_elderly", "help_sit", "sit_down",
                 "car_detection", "classroom", "face_demographics_walking_and_pause", "one_by_one_person_detection",
                 "people_detection", "person_bicycle_car_detection", "store_aisle_detection", "worker_zone_detection"]

# ------------------------------------------------------------------------------------------ camera windows
CAM_W = 256                  # analysis width for camera-motion estimation


def _est(prev: np.ndarray, g: np.ndarray, mask: np.ndarray | None = None, strict: bool = False) -> tuple[np.ndarray, bool, float]:
    """Partial-affine camera motion prev->cur: sparse LK first (small motion), ORB matching as the fallback (big jumps
    of a low-frame-rate handheld clip). Returns (3x3, ok, inlier_fraction)."""
    pts = cv2.goodFeaturesToTrack(prev, 250, 0.01, 6, mask=mask)
    if pts is not None and len(pts) >= 12:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=4)
        st = st.ravel() == 1
        if st.sum() >= 12:
            flow = np.linalg.norm((nxt - pts).reshape(-1, 2)[st], axis=1)
            if float((flow < 0.4).mean()) >= 0.45:      # the background did not move: locked-off camera, whatever the foreground does
                return np.eye(3), True, 1.0
            A, inl = cv2.estimateAffinePartial2D(pts[st], nxt[st], method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if A is not None and inl.sum() >= (40 if strict else 15) and inl.sum() / st.sum() >= (0.75 if strict else 0.5):
                return np.vstack([A, [0, 0, 1]]), True, float(inl.sum() / st.sum())
    if strict:                       # a locked camera needs strong evidence to be called moving
        return np.eye(3), False, 0.0
    orb = cv2.ORB_create(700, fastThreshold=10)
    k1, d1 = orb.detectAndCompute(prev, mask)
    k2, d2 = orb.detectAndCompute(g, None)
    if d1 is not None and d2 is not None and len(k1) >= 15 and len(k2) >= 15:
        ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
        if mask is not None:
            ms = [m for m in ms if mask[int(k2[m.trainIdx].pt[1]) % mask.shape[0], int(k2[m.trainIdx].pt[0]) % mask.shape[1]] > 0]
        if len(ms) >= 15:
            p1 = np.float32([k1[m.queryIdx].pt for m in ms])
            p2 = np.float32([k2[m.trainIdx].pt for m in ms])
            A, inl = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if A is not None and inl.sum() >= 18 and inl.sum() / len(ms) >= 0.4:
                return np.vstack([A, [0, 0, 1]]), True, float(inl.sum() / len(ms))
    return np.eye(3), False, 0.0


def camera_chain(name: str) -> dict:
    """Per sampled frame: affine camera motion prev->cur (partial affine, in CAM_W-wide pixels), a cut flag and a
    reliability flag. Cached as npz. This is what lets MIMIC separate 'the object moved' from 'the camera moved'.
    Duplicate frames (low-frame-rate sources repeated up to 25 fps) are recognised and carry no camera motion."""
    tr = pk.tracks(name, verbose=False)
    f = CACHE / f"cam_{name}_v4.npz"
    if f.exists():
        d = np.load(f)
        return {k: d[k] for k in d.files}
    for attempt in range(4):                                # decoding can fail under memory pressure: never cache a truncated chain
        out = _chain(name, tr)
        if out["f"][-1] >= tr.duration * tr.src_fps - 3 * tr.stride - 2:
            break
        print(f"[mimic] {name}: truncated camera chain (attempt {attempt + 1}), retrying", file=sys.stderr)
    else:
        raise RuntimeError(f"could not decode all of {name}")
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(f, **out)
    return out


def _chain(name: str, tr) -> dict:
    stride = tr.stride
    frames_idx, T, cut, rel, mad = [], [], [], [], []
    prev = None
    sc = None
    boxes = {int(f): g[["x1", "y1", "x2", "y2"]].values for f, g in tr.rows[tr.rows["name"].isin(MOVERS)].groupby("f")}
    prev_i = None
    static_run = 0
    for i, t, fr in pk.iter_frames(name, stride=stride):
        if sc is None:
            sc = CAM_W / fr.shape[1]
        g = cv2.cvtColor(cv2.resize(fr, (CAM_W, int(round(fr.shape[0] * sc))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        frames_idx.append(i)
        if prev is None:
            T.append(np.eye(3)); cut.append(True); rel.append(False); mad.append(0.0)
        else:
            m = float(np.mean(cv2.absdiff(g, prev)))
            if m < 0.25:                                    # same picture again (static camera or repeated frame)
                M, ok, inl = np.eye(3), True, 1.0
            else:
                mask = np.full(g.shape, 255, np.uint8)       # background only: no features on detected movers
                for bi in (prev_i, i):
                    for x1, y1, x2, y2 in boxes.get(bi, []):
                        w_, h_ = (x2 - x1) * 0.08, (y2 - y1) * 0.08
                        mask[max(0, int((y1 - h_) * sc)):int((y2 + h_) * sc) + 1, max(0, int((x1 - w_) * sc)):int((x2 + w_) * sc) + 1] = 0
                if (mask > 0).mean() < 0.15:
                    mask = None
                M, ok, inl = _est(prev, g, mask, static_run >= 5)
                if not ok and static_run >= 5 and m < 35:   # foreground fills the view of a camera that was locked: stay locked
                    M, ok, inl = np.eye(3), True, 0.5
                s = math.hypot(M[0, 0], M[1, 0])
                if ok and math.hypot(M[0, 2], M[1, 2]) < 0.16 and abs(s - 1) < 0.0015 and abs(math.atan2(M[1, 0], M[0, 0])) < 0.0015:
                    M = np.eye(3)                           # deadband: a locked-off camera stays exactly locked
            static_run = static_run + 1 if ((ok and np.allclose(M, np.eye(3))) or (not ok and m < 6)) else 0
            is_cut = ((not ok) and m > 35) or float(g.mean()) < 6
            T.append(M); cut.append(bool(is_cut)); rel.append(bool(ok) or m < 6); mad.append(m)
        if prev is None or m >= 0.25:
            prev, prev_i = g, i
    return {"f": np.array(frames_idx), "T": np.array(T), "cut": np.array(cut), "rel": np.array(rel), "mad": np.array(mad),
            "scale": np.array([sc or 1.0])}


def camera_windows(name: str, max_shift: float = 0.10, max_scale: float = 0.10, max_rot: float = 0.10, min_len: float = 1.2
                   ) -> pd.DataFrame:
    """Split the clip into windows of (near-)constant camera: new window at every cut, unreliable estimate or when the
    accumulated camera shift/zoom/rotation since the window start exceeds the limits. Inside a window every point is
    mapped back to the window's first-frame coordinates (camera-motion compensation), so a panning/shaking camera does not
    fake object motion. Returns rows: w, f0, f1, t0, t1, moving (camera moved at all), and the per-frame inverse
    cumulative matrices as `inv` (dict f -> 3x3)."""
    tr = pk.tracks(name, verbose=False)
    ch = camera_chain(name)
    fi, T, cut, rel = ch["f"], ch["T"].copy(), ch["cut"], ch["rel"]
    sc = float(ch["scale"][0])
    # Mostly locked-off camera: short, small 'camera motions' are foreground (a big person crossing the frame drags the
    # estimate), not the camera. Reset them to identity; sustained or large motion is kept.
    nonid = ~np.all(np.isclose(T, np.eye(3), atol=1e-9), axis=(1, 2))
    if nonid.mean() < 0.3:
        k = 0
        while k < len(T):
            if nonid[k]:
                e = k
                while e + 1 < len(T) and nonid[e + 1]:
                    e += 1
                cum = np.eye(3)
                for j in range(k, e + 1):
                    cum = T[j] @ cum
                if (e - k + 1) < 10 and math.hypot(cum[0, 2], cum[1, 2]) < 0.04 * CAM_W:
                    for j in range(k, e + 1):
                        T[j] = np.eye(3)
                k = e + 1
            else:
                k += 1
    lim = max_shift * CAM_W
    wins, invs = [], {}
    w, cum, w_start = -1, np.eye(3), 0
    moved = False
    for k in range(len(fi)):
        new = k == 0 or cut[k] or not rel[k]
        if not new:
            c2 = T[k] @ cum
            sh = math.hypot(c2[0, 2], c2[1, 2])
            s = math.hypot(c2[0, 0], c2[1, 0])
            rot = abs(math.atan2(c2[1, 0], c2[0, 0]))
            if sh > lim or abs(s - 1) > max_scale or rot > max_rot:
                new = True
        if new:
            if w >= 0:
                wins[-1]["f1"] = int(fi[k - 1])
                wins[-1]["moving"] = moved
            w += 1
            cum = np.eye(3)
            moved = False
            wins.append({"w": w, "f0": int(fi[k]), "f1": int(fi[k]), "moving": False})
            if not (k == 0 or cut[k]) and rel[k]:     # camera drifted out of tolerance: restart with this frame's motion kept
                pass
        else:
            cum = T[k] @ cum
            if not np.allclose(T[k], np.eye(3)):
                moved = True
        invs[int(fi[k])] = (w, np.linalg.inv(cum))
    wins[-1]["f1"] = int(fi[-1])
    wins[-1]["moving"] = moved
    df = pd.DataFrame(wins)
    fps = tr.src_fps
    df["t0"], df["t1"] = df["f0"] / fps, df["f1"] / fps
    df.attrs["inv"] = invs
    df.attrs["scale"] = sc
    return df


# ------------------------------------------------------------------------------------------------ tracklets

@dataclass
class Tracklet:
    key: str
    video: str
    tid: int
    cls: str
    win: int
    t: np.ndarray            # seconds, uniform at the track fps
    xy: np.ndarray           # (n,2) normalised to the window-start frame size, camera-compensated (box centre)
    wh: np.ndarray           # (n,2) normalised box size
    W: int
    H: int


def _smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < k:
        return a
    pad = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(a.shape[1])], 1)


def build_tracklets(name: str, min_dur: float = 1.5, max_gap: float = 1.0, classes=MOVERS) -> tuple[list[Tracklet], pd.DataFrame]:
    tr = pk.tracks(name, verbose=False)
    win = camera_windows(name)
    invs = win.attrs["inv"]
    sc = win.attrs["scale"]
    r = tr.rows[(tr.rows["tid"] >= 0) & tr.rows["name"].isin(classes)]
    out: list[Tracklet] = []
    step = 1.0 / tr.fps
    for tid, g in r.groupby("tid"):
        g = g.sort_values("f")
        cls = g["name"].value_counts().index[0]
        g = g[g["name"] == cls]
        ws = np.array([invs.get(int(f), (-1, None))[0] for f in g["f"]])
        # split at window changes and big temporal gaps
        brk = np.where((np.diff(ws) != 0) | (np.diff(g["t"].values) > max_gap + 1e-6))[0] + 1
        for seg_i, idx in enumerate(np.split(np.arange(len(g)), brk)):
            if len(idx) < 4:
                continue
            s = g.iloc[idx]
            if s["t"].iloc[-1] - s["t"].iloc[0] < min_dur or ws[idx[0]] < 0:
                continue
            xy_px = np.stack([s["cx"].values, s["cy"].values], 1)
            hom = np.hstack([xy_px * sc, np.ones((len(s), 1))])           # to CAM_W scale
            comp = np.stack([(invs[int(f)][1] @ h)[:2] for f, h in zip(s["f"], hom)], 0) / sc
            comp = comp / np.array([tr.w, tr.h])
            # fill gaps by interpolation on the uniform time grid
            ts = s["t"].values
            tu = np.arange(ts[0], ts[-1] + 1e-6, step)
            xyu = np.stack([np.interp(tu, ts, comp[:, 0]), np.interp(tu, ts, comp[:, 1])], 1)
            whu = np.stack([np.interp(tu, ts, s["bw"].values / tr.w), np.interp(tu, ts, s["bh"].values / tr.h)], 1)
            out.append(Tracklet(f"{name}:{tid}:{ws[idx[0]]}", name, int(tid), cls, int(ws[idx[0]]), tu, _smooth(xyu, 5), whu, tr.w, tr.h))
    return out, win


# ---------------------------------------------------------------------------------------------- descriptors

def resample_arc(xy: np.ndarray, n: int = N) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if d[-1] < 1e-9:
        return np.repeat(xy[:1], n, 0)
    u = np.linspace(0, d[-1], n)
    return np.stack([np.interp(u, d, xy[:, 0]), np.interp(u, d, xy[:, 1])], 1)


def resample_time(a: np.ndarray, n: int = N) -> np.ndarray:
    u = np.linspace(0, len(a) - 1, n)
    return np.stack([np.interp(u, np.arange(len(a)), a[:, j]) for j in range(a.shape[1])], 1)


def shape_norm(pos: np.ndarray, aspect: float) -> np.ndarray:
    """Translation- and scale-invariant path (isotropic pixel geometry via aspect = W/H): start at the origin, length 1."""
    p = pos * np.array([aspect, 1.0])
    L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    return (p - p[0]) / max(L, 1e-9)


def turning_curve(shp: np.ndarray) -> np.ndarray:
    """Cumulative heading change along the path (rotation invariant), 32 points, in units of pi."""
    v = np.diff(shp, axis=0)
    if np.linalg.norm(v, axis=1).sum() < 1e-6:
        return np.zeros(N)
    h = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    h = h - h[0]
    return np.interp(np.linspace(0, len(h) - 1, N), np.arange(len(h)), h) / np.pi


def descriptor(xy: np.ndarray, aspect: float, bh: np.ndarray | None = None, fps: float = 15.0) -> dict:
    """Motion descriptor of a camera-compensated path (frame-normalised coordinates).
    pos    32x2 arc-length resampled path in frame-normalised coordinates (where it happened)
    shape  32x2 same path, start at origin, unit length (what it looked like, any place / size)
    turn   32   cumulative heading curve of the shape in units of pi (rotation invariant)
    speed  32   time-resampled speed profile / mean speed (stop-and-go, accelerating, ...)
    scalars: len (path length, frame heights), ext (bbox diagonal), straight (net / length), turn_total (pi),
             stop_frac and stops (episodes slower than 0.25 box-heights per second for >= 0.6 s)"""
    pos = resample_arc(xy)
    shp = shape_norm(pos, aspect)
    tc = turning_curve(shp)
    raw = xy * np.array([aspect, 1.0])
    step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    Lraw = float(step.sum())
    sp = resample_time(np.r_[step[:1], step][:, None], N)[:, 0]
    sp = sp / max(float(sp.mean()), 1e-9)
    net = float(np.linalg.norm(raw[-1] - raw[0]))
    ext = float(np.linalg.norm(raw.max(0) - raw.min(0)))
    stop_frac, stops = 0.0, 0
    if bh is not None and len(step) >= 3:
        v = np.convolve(np.pad(step, 2, mode="edge"), np.ones(5) / 5, mode="valid")[: len(step)] * fps / np.maximum(bh[1:], 1e-3)
        slow = v < 0.25
        stop_frac = float(slow.mean())
        run = 0
        for sflag in np.r_[slow, False]:
            if sflag:
                run += 1
            else:
                if run / fps >= 0.6:
                    stops += 1
                run = 0
    return {"pos": pos, "shape": shp, "turn": tc, "speed": sp, "len": Lraw, "ext": ext, "straight": net / max(Lraw, 1e-9),
            "turn_total": float(tc[-1]), "stop_frac": stop_frac, "stops": stops}


def table_from(tls: list[Tracklet], aspect_of: dict[str, float], fps_of: dict[str, float]) -> pd.DataFrame:
    rows = []
    for t in tls:
        d = descriptor(t.xy, aspect_of[t.video], t.wh[:, 1], fps_of[t.video])
        rows.append({"key": t.key, "video": t.video, "tid": t.tid, "cls": t.cls, "win": t.win, "t0": float(t.t[0]),
                     "t1": float(t.t[-1]), "dur": float(t.t[-1] - t.t[0]), "n": len(t.t),
                     "len": d["len"], "ext": d["ext"], "rel_ext": d["ext"] / max(float(t.wh[:, 1].mean()), 1e-3), "straight": d["straight"], "turn_total": d["turn_total"],
                     "stop_frac": d["stop_frac"], "stops": d["stops"], "mean_h": float(t.wh[:, 1].mean()),
                     "pos": d["pos"].ravel().tolist(), "shape": d["shape"].ravel().tolist(),
                     "speed": d["speed"].tolist(), "turn_curve": d["turn"].tolist()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- DTW

def dtw_batch(q: np.ndarray, cands: np.ndarray, band: int = 6) -> np.ndarray:
    """DTW (Sakoe-Chiba band) of one query (n,d) against K candidates (K,m,d), vectorised over K.
    Returns the mean per-step cost along the optimal warp path, shape (K,)."""
    K, m = cands.shape[0], cands.shape[1]
    n = len(q)
    C = np.linalg.norm(q[None, :, None, :] - cands[:, None, :, :], axis=3)
    D = np.full((K, n + 1, m + 1), np.inf)
    Ln = np.zeros((K, n + 1, m + 1))
    D[:, 0, 0] = 0
    ar = np.arange(K)
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            opts = np.stack([D[:, i - 1, j - 1], D[:, i - 1, j], D[:, i, j - 1]], 1)
            lens = np.stack([Ln[:, i - 1, j - 1], Ln[:, i - 1, j], Ln[:, i, j - 1]], 1)
            k = opts.argmin(1)
            D[:, i, j] = C[:, i - 1, j - 1] + opts[ar, k]
            Ln[:, i, j] = lens[ar, k] + 1
    return D[:, n, m] / np.maximum(Ln[:, n, m], 1)


# ------------------------------------------------------------------------------------------------- queries

def preset_points(name: str) -> np.ndarray:
    """Preset sketches as 'drawn' polylines in canvas coordinates (0..1, canvas aspect 16:9)."""
    if name == "u_turn":       # come in from the left, hairpin at the right, go back
        th = np.linspace(-np.pi / 2, np.pi / 2, 24)
        arc = np.stack([0.62 + 0.10 * np.cos(th) / (16 / 9), 0.5 + 0.10 * np.sin(th)], 1)
        return np.vstack([np.stack([np.linspace(0.15, 0.62, 16), np.full(16, 0.40)], 1), arc,
                          np.stack([np.linspace(0.62, 0.15, 16), np.full(16, 0.60)], 1)])
    if name == "left_to_right":
        return np.stack([np.linspace(0.08, 0.92, 40), np.full(40, 0.55)], 1)
    if name == "diagonal":
        return np.stack([np.linspace(0.12, 0.88, 40), np.linspace(0.18, 0.82, 40)], 1)
    if name == "loiter":       # hang around: a tight scribble in one place
        t = np.linspace(0, 4 * np.pi, 80)
        return np.stack([0.5 + 0.045 * np.cos(t) / (16 / 9) * 3 + 0.02 * np.cos(2.3 * t) / (16 / 9) * 3,
                         0.5 + 0.07 * np.sin(t) + 0.02 * np.sin(1.7 * t)], 1)
    if name == "stop_and_go":  # a path along a line, drawn with pauses (speed profile below)
        return np.stack([np.linspace(0.1, 0.9, 60), np.full(60, 0.5)], 1)
    raise KeyError(name)


# per-preset matching options (position weight, size weight, rotation-invariant?, speed weight)
PRESETS = {
    "u_turn":        dict(pos_w=0.0, size_w=0.0, any_rot=True,  speed_w=0.0, desc="comes in, turns round and goes back"),
    "left_to_right": dict(pos_w=0.0, size_w=0.0, any_rot=False, speed_w=0.0, desc="crosses the frame from the left to the right"),
    "stop_and_go":   dict(pos_w=0.0, size_w=0.0, any_rot=True,  speed_w=3.0, shape_w=0.25, turn_w=0.2, desc="moves, halts, moves again (two halts)"),
    "loiter":        dict(pos_w=0.0, size_w=1.5, any_rot=True,  speed_w=0.0, shape_w=0.3, turn_w=0.2, straight_w=1.5, desc="hangs around inside a small area"),
    "diagonal":      dict(pos_w=0.0, size_w=0.0, any_rot=False, speed_w=0.0, desc="crosses the frame top-left to bottom-right"),
}


def preset_marks(name: str) -> list[tuple[float, float, str]]:
    """Temporal / dwell cues drawn on a preset sketch (canvas coordinates 0..1): where the sketch pauses or stays."""
    if name == "stop_and_go":
        sp = preset_speed(name)
        pts = preset_points(name)
        marks, i = [], 0
        while i < len(sp):
            if sp[i] < 0.1:
                j = i
                while j < len(sp) and sp[j] < 0.1:
                    j += 1
                f = ((i + j - 1) / 2) / (len(sp) - 1)
                k = int(round(f * (len(pts) - 1)))
                marks.append((float(pts[k, 0]), float(pts[k, 1]), "pause"))
                i = j
            else:
                i += 1
        return marks
    if name == "loiter":
        return [(0.5, 0.5, "stays in a small area")]
    return []


def find_pauses(t: Tracklet, fps: float | None = None) -> list[tuple[int, int, float]]:
    """Episodes where the object is slower than 0.25 box-heights per second for >= 0.6 s (same rule as the descriptor's stop count).
    Returns (first index, last index, seconds) into the tracklet's points."""
    n = len(t.t)
    if n < 4:
        return []
    fps = fps or (n - 1) / max(float(t.t[-1] - t.t[0]), 1e-6)
    raw = t.xy * np.array([t.W / t.H, 1.0])
    step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    v = np.convolve(np.pad(step, 2, mode="edge"), np.ones(5) / 5, mode="valid")[: len(step)] * fps / np.maximum(t.wh[1:, 1], 1e-3)
    slow = v < 0.25
    runs, start = [], None
    for i, f in enumerate(np.r_[slow, False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if (i - start) / fps >= 0.6:
                runs.append((start, min(i, n - 1), (i - start) / fps))
            start = None
    return runs


def preset_speed(name: str) -> np.ndarray | None:
    if name != "stop_and_go":
        return None
    sp = np.zeros(N)
    sp[0:7] = 1.8; sp[7:13] = 0.0; sp[13:21] = 1.8; sp[21:27] = 0.0; sp[27:] = 1.8
    return sp / sp.mean()


@dataclass
class Query:
    pts: np.ndarray                  # (k,2) drawn polyline in canvas coordinates 0..1
    aspect: float = 16 / 9           # canvas aspect (w/h)
    pos_w: float = 0.0
    size_w: float = 0.0
    any_rot: bool = False
    speed: np.ndarray | None = None
    speed_w: float = 0.0
    name: str = "drawn"
    shape_w: float = 1.0             # how much the drawn geometry counts
    turn_w: float = 0.8              # how much the turning profile counts (rotation invariant)
    straight_w: float = 0.0          # how much 'net displacement / path length' counts (loitering = low)


def preset_query(name: str) -> Query:
    o = PRESETS[name]
    return Query(preset_points(name), 16 / 9, o["pos_w"], o["size_w"], o["any_rot"], preset_speed(name), o["speed_w"], name,
                 o.get("shape_w", 1.0), o.get("turn_w", 0.8), o.get("straight_w", 0.0))


def rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def tl_arrays(df: pd.DataFrame) -> dict:
    return {"pos": np.array(df["pos"].tolist()).reshape(-1, N, 2), "shape": np.array(df["shape"].tolist()).reshape(-1, N, 2),
            "speed": np.array(df["speed"].tolist()), "turn": np.array(df["turn_curve"].tolist()),
            "ext": df["ext"].values, "len": df["len"].values, "rel": df["rel_ext"].values, "straight": df["straight"].values}


ROTS = np.deg2rad(np.arange(0, 360, 30))
W_TURN, W_SHAPE_D, W_TURN_D = 0.35, 0.10, 0.35
MIN_REL, MIN_EXT = 0.5, 0.03


def match(q: Query, df: pd.DataFrame, arr: dict | None = None, k: int = 8, shortlist: int = 250) -> pd.DataFrame:
    """Stage 1: cosine kNN on the 32-point descriptor (shape + turning curve, + position when pos_w > 0) with cheap size /
    speed terms. Stage 2: DTW re-rank (shape under the allowed rotations, turning curve, position, speed).
    Returns the top-k rows of df with cost columns (lower = better)."""
    arr = arr or tl_arrays(df)
    pos_q = resample_arc(q.pts)
    shp_q = shape_norm(pos_q, q.aspect)
    tc_q = turning_curve(shp_q)
    qraw = q.pts * [q.aspect, 1]
    straight_q = float(np.linalg.norm(qraw[-1] - qraw[0]) / max(np.linalg.norm(np.diff(qraw, axis=0), axis=1).sum(), 1e-9))
    ext_q = float(np.linalg.norm(qraw.max(0) - qraw.min(0)))
    rots = ROTS if q.any_rot else np.array([0.0])
    qs = [shp_q @ rot2(a).T for a in rots]

    def v1(shp, pos):
        sh = shp - shp.mean(-2, keepdims=True)
        po = pos - pos.mean(-2, keepdims=True)
        return np.concatenate([sh.reshape(*sh.shape[:-2], -1), po.reshape(*po.shape[:-2], -1) * (q.pos_w * 2.0)], -1)

    tv = np.concatenate([v1(arr["shape"], arr["pos"]), arr["turn"] * W_TURN], -1)
    tv = tv / np.maximum(np.linalg.norm(tv, axis=1, keepdims=True), 1e-9)
    sims = np.full(len(df), -9.0)
    for sq in qs:
        qv = np.concatenate([v1(sq, pos_q), tc_q * W_TURN])
        sims = np.maximum(sims, tv @ (qv / max(np.linalg.norm(qv), 1e-9)))
    s1 = sims.copy()
    if q.size_w > 0:
        s1 -= q.size_w * 0.5 * np.abs(np.log(np.maximum(arr["ext"], 1e-3) / max(ext_q, 1e-3)))
    if q.speed is not None and q.speed_w > 0:
        s1 -= q.speed_w * 0.5 * np.abs(np.clip(arr["speed"], 0, 3) - np.clip(q.speed, 0, 3)).mean(1)
    if q.straight_w > 0:
        s1 -= q.straight_w * 0.5 * np.abs(arr["straight"] - straight_q)
    s1 = np.where((arr["rel"] < MIN_REL) | (arr["ext"] < MIN_EXT), -9, s1)   # skip parked / jittering objects: must move >= half a body height
    cand = np.argsort(-s1)[:shortlist]
    cand = cand[s1[cand] > -8]
    c_shape = np.full(len(cand), np.inf)
    rot_idx = np.zeros(len(cand), int)                       # which sketch rotation matched best (0 deg when rotation is off)
    for ri, sq in enumerate(qs):
        cs_r = dtw_batch(sq, arr["shape"][cand])
        better = cs_r < c_shape
        rot_idx[better] = ri
        c_shape = np.where(better, cs_r, c_shape)
    c_turn = dtw_batch(tc_q[:, None], arr["turn"][cand][:, :, None], band=4)
    c_pos = dtw_batch(pos_q, arr["pos"][cand]) if q.pos_w > 0 else np.zeros(len(cand))
    c_spd = (dtw_batch(np.clip(q.speed, 0, 3)[:, None], np.clip(arr["speed"][cand], 0, 3)[:, :, None], band=4)
             if (q.speed is not None and q.speed_w > 0) else np.zeros(len(cand)))
    c_size = (np.abs(np.log(np.maximum(arr["ext"][cand], 1e-3) / max(ext_q, 1e-3))) if q.size_w > 0 else np.zeros(len(cand)))
    c_str = np.abs(arr["straight"][cand] - straight_q) if q.straight_w > 0 else np.zeros(len(cand))
    wsum = q.shape_w + q.turn_w + q.pos_w * 2 + q.speed_w + q.size_w + q.straight_w
    cost = (q.shape_w * c_shape / W_SHAPE_D + q.turn_w * c_turn / W_TURN_D + q.pos_w * 2 * c_pos / 0.15 + q.speed_w * c_spd / 0.8
            + q.size_w * c_size / 0.7 + q.straight_w * c_str / 0.5) / wsum
    out = df.iloc[cand].copy()
    out["cost"] = cost
    out["c_shape"], out["c_turn"], out["c_pos"], out["c_spd"], out["c_size"], out["c_str"], out["s1"] = c_shape, c_turn, c_pos, c_spd, c_size, c_str, s1[cand]
    out["rot_deg"] = np.rad2deg(rots[rot_idx]).round().astype(int)     # the sketch was matched after turning it by this angle
    return out.sort_values("cost").head(k).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ build all

def mid_fraction(t: Tracklet) -> float:
    """Fraction of the path (by arc length) reached at the middle of the tracklet's time span."""
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(t.xy, axis=0), axis=1))]
    return float(d[len(d) // 2] / d[-1]) if d[-1] > 1e-9 else 0.5


def build_all(clips: list[str] | None = None, rebuild: bool = False) -> dict:
    """Camera windows + tracklets + descriptor table for every clip (cached in _cache/tracklets.pkl)."""
    import pickle
    clips = clips or DEFAULT_CLIPS
    f = CACHE / "tracklets.pkl"
    if f.exists() and not rebuild:
        d = pickle.loads(f.read_bytes())
        if d["clips"] == clips:
            return d
    tls: dict[str, Tracklet] = {}
    wins: dict[str, pd.DataFrame] = {}
    meta: dict[str, dict] = {}
    for c in clips:
        tr = pk.tracks(c, verbose=False)
        ts, w = build_tracklets(c, min_dur=1.5)
        wins[c] = w
        meta[c] = {"w": tr.w, "h": tr.h, "fps": tr.fps, "src_fps": tr.src_fps, "duration": tr.duration, "n_tracks": int(tr.summary().shape[0]),
                   "cam_cover": float((w.t1 - w.t0).sum()), "windows": int(len(w))}
        for t in ts:
            tls[t.key] = t
    df = table_from(list(tls.values()), {c: meta[c]["w"] / meta[c]["h"] for c in clips}, {c: meta[c]["fps"] for c in clips})
    df["midf"] = [mid_fraction(tls[k]) for k in df["key"]]
    df["tmid"] = (df["t0"] + df["t1"]) / 2
    d = {"clips": clips, "tls": tls, "wins": wins, "meta": meta, "df": df}
    CACHE.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(d))
    return d
