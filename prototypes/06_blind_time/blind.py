"""BLIND TIME core: per-second camera state vs a reference view, plus three-state query answers.

States (per second of footage):
  OK        the camera delivers the reference view, sharp, lit, live
  DARK      (almost) no light reaches the sensor / black frames
  FROZEN    the feed repeats the same frame (stalled stream)
  BLUR      defocused / smeared: detail is gone
  MOVED     the reference scene is not (fully) in view any more: ORB+RANSAC similarity says the field of view
            has shifted/zoomed, or the reference scene can no longer be registered at all
  OCCLUDED  a big connected region of the (registered) view differs from the reference background (near object)
Queries answer SEEN / SEEN IN OTHER VIEW / NOTHING FOUND / COULD NOT SEE.  NOTHING FOUND is only allowed when the
camera was OK for (nearly) the whole window.  SEEN IN OTHER VIEW = the class was detected, but only while the camera
looked somewhere else than the reference view (the asked-about place itself was not watched).
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

STATES = ["OK", "DARK", "FROZEN", "BLUR", "MOVED", "OCCLUDED"]
STATE_HEX = {"OK": "#2e9e5b", "DARK": "#20242c", "FROZEN": "#3b9ad9", "BLUR": "#9b6bd1", "MOVED": "#e8a200",
             "OCCLUDED": "#d64545"}
STATE_WHY = {"OK": "reference view, sharp, live", "DARK": "no light / black frames", "FROZEN": "same frame repeated (stalled feed)",
             "BLUR": "out of focus / smeared", "MOVED": "camera looks elsewhere (not the reference view)",
             "OCCLUDED": "large near object covers the view"}

CFG: dict[str, Any] = {
    "work_w": 640,             # width used for ORB / blur / occlusion analysis
    "tiny_w": 160,             # width for the consecutive-frame difference (frozen detector)
    "ref_seconds": 7,          # reference = temporal median of this many frames (1 per second) from ref_t
    "frozen_core_eps": 0.004,  # median consecutive-frame mean abs diff (0-255, 160 px gray) of a second with "identical frames"
    "frozen_ext_eps": 0.006,   # a freeze run also swallows its neighbours up to this (x264 needs ~1 s to settle on a still frame)
    "frozen_min_run": 2,       # shortest freeze reported (s)
    "dark_abs": 14.0,          # mean luminance below this = DARK
    "dark_rel": 0.30,          # ... or below this fraction of the reference luminance (and < 60)
    "reg_min_inliers": 20,     # ORB+RANSAC: fewer inliers = reference scene not registrable
    "reg_min_ratio": 0.30,     # inlier / good-match ratio
    "moved_min_vis": 0.93,     # fraction of the reference scene still in view below which we say MOVED
    "moved_scale": (0.85, 1.18),
    "moved_rot_deg": 6.0,
    "blur_excess": 0.17,       # Crete blur metric above (reference blur + this) = BLUR (reference scene still registrable)
    "blur_unreg_excess": 0.40, # ... when the reference scene can NOT be registered, only extreme blur is called BLUR,
                               #     otherwise the camera is simply looking elsewhere (MOVED), motion blur included
    "occ_diff": 38,            # per-pixel abs difference (aligned to the reference background) counted as changed
    "occ_frac": 0.25,          # largest changed connected region as a fraction of the frame
    "lost_frac": 0.75,         # or: biggest blob of reference landmarks that vanished, as a fraction of the view
    "flat_cover": 0.85,        # unregistrable and this share of 16 px blocks is flat = lens covered (OCCLUDED)
    "despeckle": True,         # 1 s glitches between identical neighbours are absorbed
    "coverage_for_nothing": 0.95,
}


# ------------------------------------------------------------------------------------------------ small helpers


def _gray_w(frame: np.ndarray, w: int) -> np.ndarray:
    if frame.shape[1] != w:  # resize first (3 channels, area) then gray: much cheaper than gray at full res
        frame = cv2.resize(frame, (w, int(round(frame.shape[0] * w / frame.shape[1]))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def blur_metric(g: np.ndarray) -> float:
    """Crete et al. no-reference blur metric on a gray image: ~0.3-0.55 for sharp, > 0.7 for blurred (0..1)."""
    g = g.astype(np.float32)
    out = []
    for axis in (0, 1):
        k = (1, 9) if axis == 0 else (9, 1)
        b = cv2.blur(g, k)
        d = np.abs(np.diff(g, axis=axis))
        db = np.abs(np.diff(b, axis=axis))
        v = np.maximum(0, d - db)
        s = float(d.sum()) + 1e-6
        out.append((s - float(v.sum())) / s)
    return float(max(out))


def _clahe(g: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)


class Reference:
    """The home view: temporal median of a few frames (removes moving objects), with ORB features."""

    def __init__(self, src: str | Path, ref_t: float = 0.0):
        self.ref_t = ref_t
        w = CFG["work_w"]
        grays = [_gray_w(pk.frame_at(src, ref_t + k), w) for k in range(CFG["ref_seconds"])]
        self.bg = np.median(np.stack(grays), axis=0).astype(np.uint8)
        self.color = cv2.resize(pk.frame_at(src, ref_t + CFG["ref_seconds"] // 2), (w, self.bg.shape[0]),
                                interpolation=cv2.INTER_AREA)
        self.h, self.w = self.bg.shape
        self.lum = float(self.bg.mean())
        self.blur = blur_metric(self.bg)
        self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=7)
        self.kp, self.des = self.orb.detectAndCompute(_clahe(self.bg), None)
        self.pts = np.float32([k.pt for k in self.kp])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.bg_blur = cv2.GaussianBlur(self.bg, (7, 7), 0)
        # landmark grid: how many reference keypoints live in each of 16 x 9 cells
        self.gx, self.gy = 16, 9
        if len(self.pts):
            self.cell = (np.minimum((self.pts[:, 1] / self.h * self.gy).astype(int), self.gy - 1) * self.gx
                         + np.minimum((self.pts[:, 0] / self.w * self.gx).astype(int), self.gx - 1))
        else:
            self.cell = np.zeros(0, int)
        self.cell_n = np.bincount(self.cell, minlength=self.gx * self.gy)

    def lost_fraction(self, inlier_train_idx: np.ndarray) -> float:
        """Landmark loss: cells that hold reference keypoints but none of them is matched any more; the biggest connected
        blob of such cells (closed over keypoint-free cells) as a share of the view = how much of it something hides."""
        n_cells = self.gx * self.gy
        got = np.bincount(self.cell[inlier_train_idx], minlength=n_cells) if len(inlier_train_idx) else np.zeros(n_cells, int)
        lost = ((self.cell_n >= 3) & (got == 0)).reshape(self.gy, self.gx).astype(np.uint8)
        lost = cv2.morphologyEx(lost, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(lost)
        if n <= 1:
            return 0.0
        return float(stats[1:, cv2.CC_STAT_AREA].max() / n_cells)

    def register(self, g: np.ndarray) -> dict[str, Any]:
        """Similarity transform current -> reference with ORB + RANSAC. Returns metrics (n_inl 0 on failure)."""
        res: dict[str, Any] = {"n_good": 0, "n_inl": 0, "ratio": 0.0, "scale": None, "rot": None, "vis": 0.0, "M": None, "lost": None}
        kp, des = self.orb.detectAndCompute(_clahe(g), None)
        if des is None or len(kp) < 12 or self.des is None:
            return res
        good = []
        for m in self.matcher.knnMatch(des, self.des, k=2):
            if len(m) == 2 and m[0].distance < 0.8 * m[1].distance:
                good.append(m[0])
        res["n_good"] = len(good)
        if len(good) < 12:
            return res
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = self.pts[[m.trainIdx for m in good]]
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000,
                                             confidence=0.995)
        if M is None or inl is None:
            return res
        n = int(inl.sum())
        s = float(np.hypot(M[0, 0], M[1, 0]))
        if not (0.2 < s < 5.0):  # degenerate fit
            return res
        res.update(n_inl=n, ratio=n / len(good), scale=s, rot=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
                   M=[round(float(v), 5) for v in M.ravel()])
        res["vis"] = self.visible_fraction(M, g.shape)
        res["lost"] = self.lost_fraction(np.array([m.trainIdx for m, k in zip(good, inl.ravel()) if k], dtype=int))
        return res

    def visible_fraction(self, M: np.ndarray, cur_shape: tuple[int, int],
                         zone: tuple[float, float, float, float] | None = None) -> float:
        """Share of the reference view (or of a normalised zone of it) that lies inside the current frame."""
        x1, y1, x2, y2 = (0, 0, 1, 1) if zone is None else zone
        ref_poly = np.float32([[x1 * self.w, y1 * self.h], [x2 * self.w, y1 * self.h], [x2 * self.w, y2 * self.h],
                               [x1 * self.w, y2 * self.h]])
        A = np.vstack([np.asarray(M, np.float64).reshape(2, 3), [0, 0, 1]])
        inv = np.linalg.inv(A)  # reference -> current coordinates
        cur_poly = cv2.perspectiveTransform(ref_poly[None], inv)[0]
        frame_poly = np.float32([[0, 0], [cur_shape[1], 0], [cur_shape[1], cur_shape[0]], [0, cur_shape[0]]])
        a_ref = abs(cv2.contourArea(cur_poly))
        if a_ref < 1:
            return 0.0
        inter, _ = cv2.intersectConvexConvex(cur_poly.astype(np.float32), frame_poly)
        return float(max(0.0, min(1.0, inter / a_ref)))

    def occluded_fraction(self, g: np.ndarray, M: np.ndarray) -> float:
        """Largest connected region of the registered frame that differs from the reference background."""
        warped = cv2.warpAffine(g, np.asarray(M, np.float64).reshape(2, 3), (self.w, self.h), flags=cv2.INTER_LINEAR,
                                borderValue=0)
        valid = cv2.warpAffine(np.full(g.shape, 255, np.uint8), np.asarray(M, np.float64).reshape(2, 3), (self.w, self.h),
                               flags=cv2.INTER_NEAREST, borderValue=0) > 0
        d = np.abs(cv2.GaussianBlur(warped, (7, 7), 0).astype(np.int16) - self.bg_blur.astype(np.int16))
        if valid.any():
            d = np.abs(d - int(np.median(d[valid])))  # global exposure change is not occlusion
        m = ((d > CFG["occ_diff"]) & valid).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            return 0.0
        return float(stats[1:, cv2.CC_STAT_AREA].max() / (self.w * self.h))


# ------------------------------------------------------------------------------------------------ extraction


def _frames_resumable(src: str | Path, duration: float, retries: int = 6):
    """pk.iter_frames, but resume after a transient decoder failure (the laptop runs many jobs, memory spikes)."""
    pos, last_i, tries = 0.0, -1, 0
    while True:
        last_t = None
        for i, t, fr in pk.iter_frames(src, stride=1, start=pos):
            if i <= last_i:
                continue
            last_i, last_t = i, t
            yield i, t, fr
        if last_t is None or last_t >= duration - 1.5 or tries >= retries:
            return
        tries += 1
        pos = max(0.0, last_t - 0.2)


def _jpg(fr: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tobytes()


def unjpg(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


PAIRS_PER_SEC = 5  # consecutive-frame pairs examined per second (at 10/30/50/70/90% of the second)


def _sample_plan(fps: float, duration: float) -> tuple[dict[int, tuple[int, int, int]], int]:
    """frame index -> (second, pair j, 0|1) for the 2 x 5 frames we really need; only these are retrieved
    (the others are just grabbed): decoding is the cost, colour conversion + resize is skipped for ~80% of frames."""
    n = max(2, int(round(fps)))
    plan: dict[int, tuple[int, int, int]] = {}
    for s in range(int(np.ceil(duration))):
        base = int(np.ceil(s * fps - 1e-9))
        for j in range(PAIRS_PER_SEC):
            i0 = base + max(0, int(round(n * (0.1 + 0.2 * j))) - (1 if j == PAIRS_PER_SEC - 1 and n < 4 else 0))
            plan[i0] = (s, j, 0)
            plan[i0 + 1] = (s, j, 1)
    return plan, n


def extract(src: str | Path, ref_t: float = 0.0, *, keep_frames: bool = True) -> dict[str, Any]:
    """One pass over the footage -> per-second feature rows (+ the middle frame of each second as JPEG, native res)."""
    meta = pk.info(src)
    fps, nfr = meta["fps"], meta["frames"]
    ref = Reference(src, ref_t)
    tiny_w, work_w = CFG["tiny_w"], CFG["work_w"]
    plan, _ = _sample_plan(fps, meta["duration"])
    rows: list[dict[str, Any]] = []
    mids: list[bytes] = []
    cur: dict[str, Any] = {"s": -1}

    def flush() -> None:
        c = cur
        if c["s"] < 0 or "mid" not in c or len(c["pairs"]) < 3:
            return
        g = _gray_w(c["mid"], work_w)
        row: dict[str, Any] = {"s": c["s"], "src_w": int(c["mid"].shape[1]),
                               "lum": float(g.mean()), "p99": float(np.percentile(g, 99)), "blur": blur_metric(g),
                               "pairs": [round(p, 4) for p in c["pairs"]],
                               "flat": float((cv2.resize(g, (g.shape[1] // 16 * 16, g.shape[0] // 16 * 16))
                                              .reshape(g.shape[0] // 16, 16, g.shape[1] // 16, 16)
                                              .std(axis=(1, 3)) < 4.0).mean())}
        row.update(ref.register(g))
        ok = row["M"] and row["n_inl"] >= CFG["reg_min_inliers"]
        row["occ"] = ref.occluded_fraction(g, np.array(row["M"]).reshape(2, 3)) if ok else None
        rows.append(row)
        mids.append(_jpg(c["mid"]))

    i, retries = 0, 0
    cap = cv2.VideoCapture(str(pk.video_path(src)))
    last_tiny: dict[int, np.ndarray] = {}
    while True:
        if not cap.grab():
            if i < nfr - 3 and retries < 8:  # transient decoder failure: reopen and resume
                retries += 1
                cap.release()
                cap = cv2.VideoCapture(str(pk.video_path(src)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                continue
            break
        role = plan.get(i)
        if role is not None:
            try:
                ok, fr = cap.retrieve()
                if ok:
                    s, j, w = role
                    if s != cur["s"]:
                        flush()
                        cur = {"s": s, "pairs": []}
                    tiny = _gray_w(fr, tiny_w).astype(np.float32)
                    if w == 0:
                        last_tiny[j] = tiny
                        if j == PAIRS_PER_SEC // 2:
                            cur["mid"] = fr
                    elif j in last_tiny:
                        cur["pairs"].append(float(np.abs(tiny - last_tiny.pop(j)).mean()))
            except (MemoryError, cv2.error):  # the shared laptop is out of memory for a moment: lose this sample, go on
                time.sleep(1.0)
        i += 1
    cap.release()
    flush()
    return {"rows": rows, "mids": mids, "ref": ref, "fps": fps, "w": meta["w"], "h": meta["h"], "duration": meta["duration"]}


# ------------------------------------------------------------------------------------------------ classify


def frozen_flags(rows: list[dict[str, Any]]) -> list[bool]:
    """Freeze runs: >= frozen_min_run seconds whose median frame-pair difference is ~0, extended into settling neighbours."""
    med = [float(np.median(r["pairs"])) if r["pairs"] else 1.0 for r in rows]
    core = [m < CFG["frozen_core_eps"] for m in med]
    flags = [False] * len(rows)
    i = 0
    while i < len(rows):
        if not core[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and core[j + 1]:
            j += 1
        if j - i + 1 >= CFG["frozen_min_run"]:
            a, b = i, j
            while a - 1 >= 0 and med[a - 1] < CFG["frozen_ext_eps"] and not core[a - 1]:
                a -= 1
            while b + 1 < len(rows) and med[b + 1] < CFG["frozen_ext_eps"] and not core[b + 1]:
                b += 1
            for k in range(a, b + 1):
                flags[k] = True
        i = j + 1
    return flags


def classify_row(r: dict[str, Any], ref_lum: float, ref_blur: float, frozen: bool = False) -> tuple[str, dict[str, Any]]:
    c = CFG
    dark = r["lum"] < c["dark_abs"] or (r["lum"] < c["dark_rel"] * ref_lum and r["lum"] < 60)
    if dark:
        return "DARK", {"why": f"mean luminance {r['lum']:.0f} (reference {ref_lum:.0f})"}
    if frozen:
        return "FROZEN", {"why": f"identical frames (median pair difference {np.median(r['pairs']):.4f})"}
    reg_ok = r["n_inl"] >= c["reg_min_inliers"] and r["ratio"] >= c["reg_min_ratio"] and r["M"] is not None
    blurry = r["blur"] > ref_blur + c["blur_excess"]
    if reg_ok:
        lo, hi = c["moved_scale"]
        if r["vis"] < c["moved_min_vis"] or not (lo <= r["scale"] <= hi) or abs(r["rot"]) > c["moved_rot_deg"]:
            return "MOVED", {"why": f"{r['vis'] * 100:.0f}% of the reference view in frame, zoom x{r['scale']:.2f}, "
                                    f"{r['n_inl']} inliers"}
        if blurry:
            return "BLUR", {"why": f"blur {r['blur']:.2f} vs reference {ref_blur:.2f}"}
        occ, lost = r.get("occ") or 0.0, r.get("lost") or 0.0
        if occ >= c["occ_frac"] or lost >= c["lost_frac"]:
            return "OCCLUDED", {"why": f"{max(occ, lost) * 100:.0f}% of the view covered (changed {occ * 100:.0f}%, landmarks lost {lost * 100:.0f}%)"}
        return "OK", {"why": f"{r['n_inl']} inliers, {r['vis'] * 100:.0f}% in view"}
    if r["blur"] > ref_blur + c["blur_unreg_excess"]:
        return "BLUR", {"why": f"blur {r['blur']:.2f} vs reference {ref_blur:.2f}, scene unregistrable"}
    if r["flat"] > c["flat_cover"]:
        return "OCCLUDED", {"why": f"{r['flat'] * 100:.0f}% of the frame is flat (covered)"}
    return "MOVED", {"why": f"reference scene not found ({r['n_inl']} inliers)"}


def despeckle(states: list[str]) -> list[str]:
    out = list(states)
    for i in range(1, len(states) - 1):
        a, b, c = states[i - 1], states[i], states[i + 1]
        if b != a and a == c:
            out[i] = a  # 1 s glitch between identical neighbours
    return out


def classify(an: dict[str, Any]) -> list[dict[str, Any]]:
    ref = an["ref"]
    rows = an["rows"]
    fz = frozen_flags(rows)
    raw = [classify_row(r, ref.lum, ref.blur, f) for r, f in zip(rows, fz)]
    states = [x[0] for x in raw]
    sm = despeckle(states) if CFG["despeckle"] else states
    for r, (s0, info), s1 in zip(rows, raw, sm):
        r["raw_state"], r["state"], r["why"] = s0, s1, info["why"]
    return rows


def spans(states: list[str], t0: int = 0) -> list[tuple[int, int, str]]:
    """Run-length encode: (start_s, end_s_exclusive, state)."""
    out: list[tuple[int, int, str]] = []
    for i, s in enumerate(states):
        if out and out[-1][2] == s:
            out[-1] = (out[-1][0], i + 1 + t0, s)
        else:
            out.append((i + t0, i + 1 + t0, s))
    return out


# ------------------------------------------------------------------------------------------------ detections


def detect_seconds(mids: list[bytes], conf: float = 0.35, chunk: int = 600) -> list[list[dict[str, Any]]]:
    """YOLO26s (DataEngine YOLO stand-in) on the middle frame of every second. Batched, GPU-locked."""
    out: list[list[dict[str, Any]]] = []
    for i in range(0, len(mids), chunk):
        res = pk.detect_many([unjpg(b) for b in mids[i:i + chunk]], conf=conf)
        out += [[{"name": d["name"], "conf": round(d["conf"], 3), "box": [round(v, 1) for v in d["box"]]} for d in fr] for fr in res]
    return out


# ------------------------------------------------------------------------------------------------ queries


@dataclass
class Query:
    cls: str
    t0: float
    t1: float
    zone: tuple[float, float, float, float] | None = None  # normalised x1,y1,x2,y2 in the reference view

    @staticmethod
    def parse(text: str) -> "Query":
        """'person 02:00-04:00' or 'car 0:30-1:10 zone=0,0.5,1,1'."""
        parts = text.split()
        cls = parts[0]
        a, b = parts[1].split("-")
        zone = None
        for p in parts[2:]:
            if p.startswith("zone="):
                zone = tuple(float(v) for v in p[5:].split(","))  # type: ignore[assignment]

        def tt(x: str) -> float:
            sec = 0.0
            for v in x.split(":"):
                sec = sec * 60 + float(v)
            return sec
        return Query(cls, tt(a), tt(b), zone)

    def label(self) -> str:
        z = "" if self.zone is None else " [zone]"
        return f"{self.cls} {pk.mmss(self.t0)[:5]}-{pk.mmss(self.t1)[:5]}{z}"


BLIND = {"DARK", "FROZEN", "MOVED"}


def answer(rows: list[dict[str, Any]], dets: list[list[dict[str, Any]]], q: Query, ref: Reference | None = None) -> dict[str, Any]:
    """SEEN / SEEN IN OTHER VIEW / NOTHING FOUND / COULD NOT SEE for a class (and optional zone) in a window."""
    n = len(rows)
    a, b = max(0, int(np.floor(q.t0))), min(n, int(np.ceil(q.t1)))
    secs = list(range(a, b))
    if not secs:
        return {"query": q.label(), "verdict": "COULD NOT SEE", "naive": "NOTHING FOUND", "coverage": 0.0,
                "blind_s": 0, "window_s": 0, "blind_spans": [], "evidence": [], "note": "window outside the footage"}

    def zone_cov(i: int) -> float:
        """How much of the query zone the camera actually covers in second i (1.0 for whole-frame queries)."""
        r = rows[i]
        st = r["state"]
        if st in ("DARK", "FROZEN"):
            return 0.0
        if q.zone is None:
            return 1.0 if st == "OK" else 0.0
        if r["M"] is None:
            return 0.0
        v = ref.visible_fraction(np.array(r["M"]).reshape(2, 3), (ref.h, ref.w), q.zone) if ref else r["vis"]
        if st == "MOVED":
            return v if v >= 0.5 else 0.0
        return v if st == "OK" else 0.0

    def in_zone(d: dict[str, Any], i: int) -> bool:
        if q.zone is None:
            return True
        r = rows[i]
        if r["M"] is None or ref is None:
            return False
        sx = ref.w / r["src_w"]
        cx, cy = (d["box"][0] + d["box"][2]) / 2 * sx, (d["box"][1] + d["box"][3]) / 2 * sx
        M = np.array(r["M"]).reshape(2, 3)
        px, py = M[0, 0] * cx + M[0, 1] * cy + M[0, 2], M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        x1, y1, x2, y2 = q.zone
        return x1 * ref.w <= px <= x2 * ref.w and y1 * ref.h <= py <= y2 * ref.h

    cov = [zone_cov(i) for i in secs]
    coverage = float(np.mean(cov))
    blind_idx = [i for i, c in zip(secs, cov) if c < 0.5]
    sp = []
    for i in blind_idx:
        st = rows[i]["state"] if rows[i]["state"] != "OK" else "MOVED"
        if sp and sp[-1][1] == i and sp[-1][2] == st:
            sp[-1] = (sp[-1][0], i + 1, st)
        else:
            sp.append((i, i + 1, st))
    ev, off = [], []
    for i in secs:
        st = rows[i]["state"]
        if st in ("DARK", "FROZEN"):
            continue  # black or stale frames are not new evidence
        hits = [d for d in dets[i] if d["name"] == q.cls and in_zone(d, i)]
        if not hits:
            continue
        item = {"s": i, "state": st, "n": len(hits), "conf": max(h["conf"] for h in hits),
                "box": max(hits, key=lambda h: h["conf"])["box"]}
        # evidence counts if the camera covers the zone, or shows it degraded (BLUR / OCCLUDED); a hit in some other
        # field of view (MOVED away from the zone) is real but is not about the place that was asked about
        if cov[i - a] >= 0.5 or st in ("BLUR", "OCCLUDED"):
            ev.append(item)
        else:
            off.append(item)
    def naive_in_zone(d: dict[str, Any], i: int) -> bool:  # a search without camera state: zone in raw frame coordinates
        if q.zone is None:
            return True
        sw = rows[i]["src_w"]
        sh = sw * ref.h / ref.w
        cx, cy = (d["box"][0] + d["box"][2]) / 2 / sw, (d["box"][1] + d["box"][3]) / 2 / sh
        return q.zone[0] <= cx <= q.zone[2] and q.zone[1] <= cy <= q.zone[3]

    naive_hits = [i for i in secs if any(d["name"] == q.cls and naive_in_zone(d, i) for d in dets[i])]
    naive = "SEEN" if naive_hits else "NOTHING FOUND"
    blind_s = len(blind_idx)
    if ev:
        verdict = "SEEN"
    elif off:
        verdict = "SEEN IN OTHER VIEW"  # real detections, but not of the asked-about view: never rendered as a contradiction
    elif coverage >= CFG["coverage_for_nothing"]:
        verdict = "NOTHING FOUND"
    else:
        verdict = "COULD NOT SEE"
    return {"query": q.label(), "cls": q.cls, "t0": q.t0, "t1": q.t1, "verdict": verdict, "naive": naive,
            "coverage": coverage, "window_s": len(secs), "blind_s": blind_s,
            "blind_spans": [{"t0": x, "t1": y, "state": s} for x, y, s in sp],
            "evidence": ev, "evidence_off_view": off,
            "naive_misleading": naive == "NOTHING FOUND" and verdict == "COULD NOT SEE",
            "naive_wrong": naive != verdict,
            "naive_stale": naive == "SEEN" and verdict == "COULD NOT SEE"}


def fmt_dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 60} m {s % 60:02d} s" if s >= 60 else f"{s} s"


# ------------------------------------------------------------------------------------------------ io


def save_rows(path: Path, rows: list[dict[str, Any]], dets: list[list[dict[str, Any]]], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows, "dets": dets, **extra}, default=float), encoding="utf-8")


def load_rows(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_for(src: str | Path, rows: list[dict[str, Any]], mids: list[bytes], conf: float = 0.35
               ) -> tuple[list[list[dict[str, Any]]], str]:
    """Per-second detections. Catalogue clips reuse the cached YOLO26s+ByteTrack table (the DataEngine YOLO stand-in, no GPU
    wait); anything else (our SIMULATED-fault files, new footage) is run through YOLO26s on the middle frame of each second."""
    name = str(src)
    if name in pk.videos():
        try:
            tr = pk.tracks(name)  # cached: instant; (never recomputed here)
            out = []
            for r in rows:
                d = tr.at(r["s"] + 0.5, tol=0.5)
                out.append([{"name": x.name, "conf": round(float(x.conf), 3),
                             "box": [round(float(v), 1) for v in (x.x1, x.y1, x.x2, x.y2)]}
                            for x in d.itertuples() if x.conf >= conf])
            return out, "cached YOLO26s+ByteTrack tracks (15 fps), frame nearest the middle of each second"
        except Exception as e:  # noqa: BLE001
            print(f"  [detect] track cache unavailable ({type(e).__name__}); running YOLO26s")
    return detect_seconds(mids, conf), "YOLO26s (TensorRT) on the middle frame of each second"


def analyze(src: str | Path, workdir: Path, ref_t: float = 0.0, force: bool = False, log=print, with_dets: bool = True) -> dict[str, Any]:
    """extract + detect (cached on disk, keyed by file size/mtime/ref_t) + classify. Returns the analysis dict."""
    import os
    p = pk.video_path(src)
    st = os.stat(p)
    key = {"v": 4, "path": str(p), "size": st.st_size, "mtime": int(st.st_mtime), "ref_t": ref_t}
    if not with_dets:
        key["dets"] = False
    f = workdir / "features.json"
    if f.exists() and not force:
        d = load_rows(f)
        if d.get("key") == key and not d.get("dets_pending"):
            an = {"rows": d["rows"], "dets": d["dets"], "ref": Reference(src, ref_t), "fps": d["fps"], "w": d["w"], "h": d["h"],
                  "duration": d["duration"], "seconds_to_compute": d.get("seconds_to_compute")}
            classify(an)
            log(f"  [{p.stem}] features from cache")
            return an
    t0 = time.time()
    an = extract(src, ref_t)
    t1 = time.time()
    mids = an.pop("mids")
    meta = {"key": key, "fps": an["fps"], "w": an["w"], "h": an["h"], "duration": an["duration"]}
    save_rows(f, an["rows"], [[] for _ in an["rows"]], {**meta, "dets_pending": True})  # features survive a failed detection
    if with_dets:
        an["dets"], src_of_dets = detect_for(src, an["rows"], mids)
    else:
        an["dets"], src_of_dets = [[] for _ in an["rows"]], "not run"
    t2 = time.time()
    an["seconds_to_compute"] = {"extract": round(t1 - t0, 1), "detect": round(t2 - t1, 1), "detections_from": src_of_dets}
    save_rows(f, an["rows"], an["dets"], {**meta, "seconds_to_compute": an["seconds_to_compute"]})
    classify(an)
    log(f"  [{p.stem}] analysed {len(an['rows'])} s: extract {t1 - t0:.0f}s, YOLO {t2 - t1:.0f}s")
    return an


# ------------------------------------------------------------------------------------------------ VastDB-style SQL


def health_tables(rows: list[dict[str, Any]], dets: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """The two tables the query needs, shaped like rows we would write to VastDB next to the DataEngine segment table:
    health(sec, state, ok, usable) and hits(sec, cls, n, conf, usable): one row per second per class."""
    import pandas as pd
    health = pd.DataFrame({"sec": [r["s"] for r in rows], "state": [r["state"] for r in rows],
                           "ok": [int(r["state"] == "OK") for r in rows],
                           "usable": [int(r["state"] in ("OK", "BLUR", "OCCLUDED")) for r in rows]})
    recs = []
    for r, fr in zip(rows, dets):
        by: dict[str, list[float]] = {}
        for d in fr:
            by.setdefault(d["name"], []).append(d["conf"])
        for cls, cf in by.items():
            recs.append({"sec": r["s"], "cls": cls, "n": len(cf), "conf": max(cf),
                         "usable": int(r["state"] in ("OK", "BLUR", "OCCLUDED")), "other_view": int(r["state"] == "MOVED")})
    hits = pd.DataFrame(recs, columns=["sec", "cls", "n", "conf", "usable", "other_view"])
    return {"health": health, "hits": hits}


SQL_COVERAGE = "SELECT COUNT(*), SUM(ok) FROM health WHERE sec >= {a} AND sec < {b}"
SQL_HITS = "SELECT COUNT(*) FROM hits WHERE cls = '{cls}' AND usable = 1 AND sec >= {a} AND sec < {b}"


SQL_OTHER = "SELECT COUNT(*) FROM hits WHERE cls = '{cls}' AND other_view = 1 AND sec >= {a} AND sec < {b}"


def answer_sql(con: Any, q: Query, n_rows: int) -> dict[str, Any]:
    """The same three-state answer from plain SQL (WHERE + COUNT/SUM only: no GROUP BY / JOIN / IN, which VastDB lacks).
    Whole-frame queries only (no zone)."""
    a, b = max(0, int(np.floor(q.t0))), min(n_rows, int(np.ceil(q.t1)))
    cnt, ok = con.execute(SQL_COVERAGE.format(a=a, b=b)).fetchone()
    n_hit = con.execute(SQL_HITS.format(cls=q.cls.replace("'", ""), a=a, b=b)).fetchone()[0]
    n_oth = con.execute(SQL_OTHER.format(cls=q.cls.replace("'", ""), a=a, b=b)).fetchone()[0]
    cov = (ok or 0) / cnt if cnt else 0.0
    verdict = ("SEEN" if n_hit else "SEEN IN OTHER VIEW" if n_oth else
               "NOTHING FOUND" if cov >= CFG["coverage_for_nothing"] else "COULD NOT SEE")
    return {"verdict": verdict, "coverage": cov, "hit_seconds": int(n_hit), "other_view_seconds": int(n_oth), "window_s": int(cnt),
            "sql": [SQL_COVERAGE.format(a=a, b=b), SQL_HITS.format(cls=q.cls, a=a, b=b), SQL_OTHER.format(cls=q.cls, a=a, b=b)]}
