"""Overnight Ledger core: per-second PRESENT / ABSENT / CAN'T-SEE ledger for one parked car, from a fixed camera.

Everything here is local: cached YOLO26s+ByteTrack boxes (stand-in for the DataEngine YOLO11s step), OpenCV for
frame health + view alignment + crop comparison. No VLM is needed for correctness.

States per second (all in the REFERENCE frame coordinates, so a PTZ nudge or a re-aim does not break matching):
  P  present       a vehicle box overlaps the car's spot (IoU), or the spot still looks like the reference car
  A  absent        spot visible, no vehicle there, spot looks different from the reference car
  D  different     a vehicle is there but the crop no longer looks like the reference car
  G  can't see     with a cause: covered (dark/flat), frozen, blurred, view (view changed / spot out of frame),
                   occluded (another vehicle/person covers the spot)
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

HERE = Path(__file__).resolve().parent
VEH = ("car", "truck", "bus")
MAX_FILL_S = 2         # alignment dropouts up to this long, between two identical decisive seconds, take the neighbours' state
MIN_BLIND_S = 3        # shorter can't-see gaps are dropouts, not blind stretches: no before/after verdict

# ------------------------------------------------------------------ thresholds (calibrated, see README "Calibration")
P = dict(
    coarse_same=0.80,      # 64x36 layout correlation with the reference above this => same camera view
    cut_d=18.0,            # mean abs diff between 5 fps samples that counts as a hard cut when splitting runs
    dark_mean=22.0,        # frame mean luma below this => covered / dark
    flat_std=12.0,         # frame luma std below this => lens covered by something flat
    frozen_d=0.03,         # 5 fps sample-to-sample mean abs diff below this => identical frame
    frozen_min_samples=10,  # >= 2 s of identical frames
    blur_ratio=0.30,       # Laplacian variance vs reference below this (view aligned) => blurred
    iou_present=0.40,      # vehicle box IoU with the spot in reference coordinates
    occl_cover=0.40,       # fraction of the spot covered by another detection => occluded
    app_present=0.55,      # crop-vs-reference SSIM at/above this => still the car even if the detector missed it
    app_absent=0.48,       # crop SSIM below this with no vehicle box => spot is really empty/different; in between => unsure
    app_diff=0.45,         # crop SSIM below this while a vehicle box is there => different vehicle
    orb_min_inliers=22, orb_min_ratio=0.25, scale_lo=0.85, scale_hi=1.18, max_rot_deg=7.0,
    # before/after comparison across a blind stretch (calibrated on clean footage, see calibration.json)
    cmp_ssim=0.50, cmp_block=0.20, cmp_hue=-1.0,   # hue overlaps between similar-coloured cars: reported, not used
    contact_pad=0.30,      # dilation of the car footprint (fraction of box width) for the contact zone
)


# ------------------------------------------------------------------------------------------------------ features
def vkey(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    return hashlib.sha1(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:12]


def features(path: str | Path, cache_dir: Path, *, refresh: bool = False, sample_fps: float = 5.0) -> dict[str, Any]:
    """5 fps scalars per sampled frame: t, mean luma, luma std, Laplacian var, diff to previous sample, hist distance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"feat_{Path(path).stem}_{vkey(path)}.npz"
    if f.exists() and not refresh:
        z = np.load(f)
        return {k: z[k] for k in z.files}
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / sample_fps)))
    rows, prev, prev_h, i = [], None, None, 0
    while True:
        if not cap.grab():
            break
        if i % stride == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            s = cv2.resize(g, (192, max(1, int(192 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
            m = cv2.resize(g, (320, max(1, int(320 * g.shape[0] / g.shape[1]))), interpolation=cv2.INTER_AREA)
            h = cv2.calcHist([s], [0], None, [32], [0, 256]).ravel()
            h /= max(h.sum(), 1)
            d = 0.0 if prev is None else float(np.abs(s.astype(np.int16) - prev).mean())
            hd = 0.0 if prev_h is None else float(np.abs(np.sqrt(h) - np.sqrt(prev_h)).sum())
            rows.append((i / fps, float(m.mean()), float(m.std()), cv2.Laplacian(m, cv2.CV_64F).var(), d, hd))
            prev, prev_h = s.astype(np.int16), h
        i += 1
    cap.release()
    a = np.array(rows, dtype=np.float64).reshape(-1, 6)
    out = {"t": a[:, 0], "mean": a[:, 1], "std": a[:, 2], "lap": a[:, 3], "d": a[:, 4], "hd": a[:, 5],
           "fps": np.array(fps), "n_src_frames": np.array(i)}
    np.savez_compressed(f, **out)
    return out


def covered_frame(mean: float, std: float, ref_mean: float, ref_std: float) -> bool:
    """Lens covered: very dark or flat, AND much darker/flatter than the reference scene (a night shot or an empty
    asphalt lot is dark/flat all the time and must not read as a covered lens)."""
    return bool((mean < P["dark_mean"] and mean < 0.5 * ref_mean) or (std < P["flat_std"] and std < 0.3 * ref_std))


def frozen_mask(F: dict[str, Any]) -> np.ndarray:
    """Samples inside a run of identical frames lasting >= 2 s. One-sample blips (an encoder keyframe refresh inside a
    frozen stretch reads d ~ 0.5) do not break a run; real scenes read d >= 0.08 even when nothing moves."""
    d = F["d"]
    fz = d < P["frozen_d"]
    if len(fz):
        fz[0] = False
    fz = fz.copy()
    for i in range(1, len(d) - 1):
        if not fz[i] and fz[i - 1] and fz[i + 1] and d[i] < 1.0:
            fz[i] = True
    run = np.zeros(len(d), bool)
    i = 0
    while i < len(d):
        if fz[i]:
            j = i
            while j + 1 < len(d) and fz[j + 1]:
                j += 1
            if j - i + 1 >= P["frozen_min_samples"]:
                run[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return run


def split_runs(F: dict[str, Any], cut_d: float = P["cut_d"]) -> list[tuple[float, float]]:
    """Segments of continuous footage between hard cuts (a slideshow gives many tiny runs, a CCTV one long run)."""
    t, d = F["t"], F["d"]
    if len(t) == 0:
        return []
    step = float(np.median(np.diff(t))) if len(t) > 1 else 0.2
    cuts = [0] + [i for i in range(1, len(t)) if d[i] > cut_d] + [len(t)]
    return [(float(t[a]), float(t[b - 1] + step)) for a, b in zip(cuts[:-1], cuts[1:])]


# --------------------------------------------------------------------------------------------------- spot selection
@dataclass
class Spot:
    box: tuple[float, float, float, float]   # reference-frame pixel box of the parked car
    t_ref: float                             # reference frame time
    run: tuple[float, float]                 # footage run it was selected in
    tids: list[int] = field(default_factory=list)
    name: str = "car"
    stay: float = 0.0                        # seconds the car sat at this spot inside the run
    stationary_px: float = 0.0               # p90 centre drift (px)
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"box": [round(v, 1) for v in self.box], "t_ref": round(self.t_ref, 2), "run": [round(v, 1) for v in self.run],
                "tids": self.tids, "name": self.name, "stay_s": round(self.stay, 1), "drift_px": round(self.stationary_px, 1)}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _fill_gaps(secs: set[int], gap: int = 3) -> set[int]:
    s = sorted(secs)
    out = set(s)
    for a, b in zip(s[:-1], s[1:]):
        if 1 < b - a <= gap + 1:
            out.update(range(a + 1, b))
    return out


def candidate_spots(tr: pk.Tracks, runs: list[tuple[float, float]], *, min_stay: float = 20.0, min_run: float = 20.0,
                    t_lo: float = 0.0, t_hi: float | None = None) -> list[Spot]:
    """Parked-car candidates, best first. A car is 'parked' when its box centre drifts < 25 % of its size over the run."""
    out: list[Spot] = []
    W, H = tr.w, tr.h
    vr = tr.rows[(tr.rows["name"].isin(VEH)) & (tr.rows["tid"] >= 0) & (tr.rows["conf"] >= 0.25)]
    for (r0, r1) in runs:
        r0, r1 = max(r0, t_lo), min(r1, t_hi if t_hi is not None else r1)
        if r1 - r0 < min_run:
            continue
        r = vr[(vr["t"] >= r0 + 0.5) & (vr["t"] < r1)]
        if r.empty:
            continue
        per: list[dict[str, Any]] = []
        for tid, g in r.groupby("tid"):
            if len(g) < 8:
                continue
            med = g[["x1", "y1", "x2", "y2"]].median().to_numpy()
            size = max(med[2] - med[0], med[3] - med[1])
            drift = float(np.percentile(np.hypot(g["cx"] - (med[0] + med[2]) / 2, g["cy"] - (med[1] + med[3]) / 2), 90))
            if drift > 0.25 * size:
                continue
            if med[0] < 6 or med[1] < 6 or med[2] > W - 6 or med[3] > H - 6:
                continue            # clipped by the frame edge
            if (med[2] - med[0]) * (med[3] - med[1]) < 0.0015 * W * H:
                continue            # too small to compare crops
            secs = set(int(x) for x in g["t"].to_numpy())
            per.append({"tid": int(tid), "box": med, "drift": drift, "secs": secs, "name": g["name"].mode().iloc[0],
                        "t0": float(g["t"].min())})
        per.sort(key=lambda p: p["t0"])
        used = [False] * len(per)
        for i, p in enumerate(per):                     # merge broken tracks by box overlap (IDs break, boxes do not)
            if used[i]:
                continue
            members, secs, boxes = [p], set(p["secs"]), [p["box"]]
            used[i] = True
            for j in range(i + 1, len(per)):
                if not used[j] and iou(p["box"], per[j]["box"]) >= 0.5:
                    used[j] = True
                    members.append(per[j])
                    secs |= per[j]["secs"]
                    boxes.append(per[j]["box"])
            secs = _fill_gaps(secs)
            stay = float(len(secs))
            if stay < min_stay:
                continue
            box = np.median(np.array(boxes), axis=0)
            first = min(secs)
            out.append(Spot(tuple(float(v) for v in box), t_ref=max(r0 + 1.0, float(first)) + 0.5, run=(r0, r1),
                            tids=[m["tid"] for m in members], name=members[0]["name"], stay=stay,
                            stationary_px=float(max(m["drift"] for m in members))))
    # rank: stay (5 s bins, so ties go to the more useful car) then box area
    out.sort(key=lambda s: (-round(s.stay / 5.0), -(s.box[2] - s.box[0]) * (s.box[3] - s.box[1])))
    return out


# --------------------------------------------------------------------------------------------- appearance compare
def _ssim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (7, 7)
    mu_a, mu_b = cv2.GaussianBlur(a, k, 1.5), cv2.GaussianBlur(b, k, 1.5)
    saa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    sbb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))


def _norm(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    return np.clip((g - g.mean()) / (g.std() + 1e-3) * 40 + 128, 0, 255)


def compare(a: np.ndarray, b: np.ndarray, margin: int) -> dict[str, float]:
    """Compare two crops of the same spot (BGR, each with `margin` px of context around the spot box).
    The inner spot of `a` is template-matched inside `b` (+-margin px) to absorb alignment error, both are
    brightness/contrast-normalised, then: global SSIM, worst 4x4 block SSIM, hue/saturation histogram correlation."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h, w = ga.shape
    m = max(1, min(margin, h // 4, w // 4))
    tpl = ga[m:h - m, m:w - m]
    res = cv2.matchTemplate(gb, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    x, y = loc
    pa, pb = tpl, gb[y:y + tpl.shape[0], x:x + tpl.shape[1]]
    sz = (96, max(24, int(96 * pa.shape[0] / max(pa.shape[1], 1))))
    na, nb = _norm(cv2.resize(pa, sz, interpolation=cv2.INTER_AREA)), _norm(cv2.resize(pb, sz, interpolation=cv2.INTER_AREA))
    na, nb = cv2.GaussianBlur(na, (0, 0), 1.2), cv2.GaussianBlur(nb, (0, 0), 1.2)   # tolerate compression / 1 px error
    s = _ssim(na, nb)[3:-3, 3:-3]
    bh, bw = s.shape[0] // 4, s.shape[1] // 4
    blocks = [float(s[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean()) for i in range(4) for j in range(4)]
    ca = a[m:h - m, m:w - m]
    cb = b[y:y + tpl.shape[0], x:x + tpl.shape[1]]
    ha = cv2.calcHist([cv2.cvtColor(ca, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(cb, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(ha, ha, 1, 0, cv2.NORM_L1)
    cv2.normalize(hb, hb, 1, 0, cv2.NORM_L1)
    hue = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return {"ssim": float(s.mean()), "block_min": float(min(blocks)), "hue": hue,
            "shift_px": float(np.hypot(x - m, y - m)), "score": float(s.mean())}


def looks_changed(c: dict[str, float]) -> bool:
    return c["ssim"] < P["cmp_ssim"] or c["block_min"] < P["cmp_block"] or c["hue"] < P["cmp_hue"]


# ---------------------------------------------------------------------------------------------------- the ledger
def _clahe_gray(img: np.ndarray, width: int = 800) -> tuple[np.ndarray, float]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = min(1.0, width / g.shape[1])
    if s < 1:
        g = cv2.resize(g, (int(g.shape[1] * s), int(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(2.0, (8, 8)).apply(g), s


class Aligner:
    """Estimates the cur->ref homography (PTZ pan/tilt/zoom is a homography) from ORB matches; None if the view changed."""

    def __init__(self, ref_bgr: np.ndarray):
        self.orb = cv2.ORB_create(2500, fastThreshold=8)
        self.ref, self.s = _clahe_gray(ref_bgr)
        self.kp, self.des = self.orb.detectAndCompute(self.ref, None)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def __call__(self, cur_bgr: np.ndarray) -> tuple[np.ndarray | None, int]:
        g, s = _clahe_gray(cur_bgr)
        kp, des = self.orb.detectAndCompute(g, None)
        if des is None or self.des is None or len(kp) < 30:
            return None, 0
        keep = np.ones(len(kp), bool)
        # A camera on-screen display ("ID: 001", clocks) is identical in every frame and wins RANSAC with an identity
        # homography. If the consensus set is a compact blob, drop it and fit again on the scene.
        for _ in range(3):
            good = [m for m, n in (p for p in self.bf.knnMatch(des, self.des, k=2) if len(p) == 2)
                    if m.distance < 0.75 * n.distance and keep[m.queryIdx]]
            if len(good) < P["orb_min_inliers"]:
                return None, len(good)
            src = np.float32([kp[m.queryIdx].pt for m in good]) / s
            dst = np.float32([self.kp[m.trainIdx].pt for m in good]) / self.s
            M, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if M is None or inl is None:
                return None, 0
            ins = inl.ravel().astype(bool)
            pts = src[ins]
            span = (np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])) / (g.shape[1] / s * g.shape[0] / s)
            if span >= 0.12 or ins.sum() < 8:
                break
            x0, y0 = pts.min(0) - 25
            x1, y1 = pts.max(0) + 25
            for i, k_ in enumerate(kp):
                if x0 <= k_.pt[0] / s <= x1 and y0 <= k_.pt[1] / s <= y1:
                    keep[i] = False
        n_in = int(ins.sum())
        # local scale / rotation at the frame centre, and perspective strength: a PTZ pan/tilt/zoom is a homography
        cx, cy = g.shape[1] / 2 / s, g.shape[0] / 2 / s
        p0 = cv2.perspectiveTransform(np.float32([[[cx, cy]], [[cx + 100, cy]], [[cx, cy + 100]]]), M)[:, 0]
        v1, v2 = (p0[1] - p0[0]) / 100.0, (p0[2] - p0[0]) / 100.0
        scale = float(np.sqrt(abs(v1[0] * v2[1] - v1[1] * v2[0])))
        rot = float(np.degrees(np.arctan2(v1[1], v1[0])))
        if (span < 0.12 or n_in < P["orb_min_inliers"] or n_in / len(good) < P["orb_min_ratio"]
                or not P["scale_lo"] <= scale <= P["scale_hi"] or abs(rot) > P["max_rot_deg"]
                or abs(M[2, 0]) > 5e-4 or abs(M[2, 1]) > 5e-4):
            return None, n_in
        return M, n_in


def apply_M(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), M.astype(np.float64))[:, 0]


def map_box(M: np.ndarray, box) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    p = apply_M(M, np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64))
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


def inv_M(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)


@dataclass
class Ledger:
    video: str
    duration: float
    spot: Spot
    state: list[str]                 # per second: 'P','A','D','G'
    cause: list[str]                 # per second: '' or covered/frozen/blurred/view/occluded
    app: list[float]                 # per second appearance score vs reference (nan if no view)
    Ms: list[Any]                    # per second cur->ref transform or None
    crops: dict[int, np.ndarray]     # clear seconds -> BGR crop (spot + margin) in reference coordinates
    margin: int
    ref_frame: np.ndarray
    ref_crop: np.ndarray
    window: tuple[int, int]          # [first, last+1) seconds where the spot was ever aligned
    health: dict[str, np.ndarray]
    params: dict[str, Any] = field(default_factory=dict)


def _spot_crop(img_ref_coords: np.ndarray, box, margin: int) -> np.ndarray:
    H, W = img_ref_coords.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    pad = cv2.copyMakeBorder(img_ref_coords, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
    return pad[y1:y2 + 2 * margin, x1:x2 + 2 * margin].copy()


def second_frames(path: str | Path, nsec: int):
    """Yield (k, frame) for one frame per second (first sampled frame at/after k+0.4 s)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    k, i = 0, 0
    nxt = 0.4
    while k < nsec:
        if not cap.grab():
            break
        if i / fps >= nxt:
            ok, fr = cap.retrieve()
            if not ok:
                break
            yield k, fr
            k += 1
            nxt = k + 0.4
        i += 1
    cap.release()


@dataclass
class Align:
    """Video-wide, spot-independent alignment: per second, the transform cur->ref and the frame warped into reference
    coordinates (JPEG bytes). Computed once per video, shared by every car ledger."""
    video: str
    t_ref: float
    ref_frame: np.ndarray
    nsec: int
    Ms: list[Any]
    jpgs: dict[int, bytes]
    W: int
    H: int
    coarse: list[float] = field(default_factory=list)   # 64x36 zero-mean correlation with the reference, every second


def _tiny(img: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    g -= g.mean()
    return g / (np.linalg.norm(g) + 1e-6)


def align_video(path: str | Path, F: dict[str, Any], t_ref: float, *, video: str | None = None,
                cache_dir: Path | None = None) -> Align:
    path = Path(path)
    video = video or pk.name_of(path)
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ck = hashlib.sha1(("v3" + vkey(path) + repr(round(t_ref, 2)) + repr([P[k] for k in sorted(P) if k.startswith(("orb", "scale", "max_rot"))])).encode()).hexdigest()[:12]
        cache = cache_dir / f"align_{path.stem}_{ck}.pkl"
        if cache.exists():
            return pickle.loads(cache.read_bytes())
    nsec = int(np.ceil(float(F["t"][-1]) + 0.2))
    ref = pk.frame_at(path, t_ref)
    H, W = ref.shape[:2]
    al = Aligner(ref)
    Ms: list[Any] = [None] * nsec
    jpgs: dict[int, bytes] = {}
    coarse = [float("nan")] * nsec
    ref_tiny = _tiny(ref)
    for k, fr in second_frames(path, nsec):
        coarse[k] = float((ref_tiny * _tiny(fr)).sum())
        M, _ = al(fr)
        if M is None:
            continue
        Ms[k] = M
        warped = cv2.warpPerspective(fr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        jpgs[k] = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tobytes()
    out = Align(video, t_ref, ref, nsec, Ms, jpgs, W, H, coarse)
    if cache is not None:
        cache.write_bytes(pickle.dumps(out))
    return out


def visible_seconds(al: Align, box) -> list[int]:
    """Seconds in which the spot is inside the (aligned) frame."""
    out = []
    for k, M in enumerate(al.Ms):
        if M is None:
            continue
        bx = map_box(inv_M(M), box)
        area = max(0.0, min(bx[2], al.W) - max(bx[0], 0)) * max(0.0, min(bx[3], al.H) - max(bx[1], 0))
        if area >= 0.9 * max(1.0, (bx[2] - bx[0]) * (bx[3] - bx[1])):
            out.append(k)
    return out


def margin_for_box(box) -> int:
    """Context (px) kept around the spot box in every crop: the search range for the +-px re-registration in compare()."""
    return int(max(10, round(0.25 * max(box[2] - box[0], box[3] - box[1]))))


def spot_margin(spot: Spot) -> int:
    return margin_for_box(spot.box)


def _ctx(img: np.ndarray, box) -> np.ndarray:
    """Grey context window (the spot box grown by one box size on every side), float32, for local registration."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb, ya, yb = int(max(0, x1 - w)), int(min(W, x2 + w)), int(max(0, y1 - h)), int(min(H, y2 + h))
    g = cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def local_shift(ref_ctx: np.ndarray, cur_ctx: np.ndarray, max_px: float = 30.0) -> tuple[float, float, float]:
    """Residual (dx, dy) of the spot's surroundings after the global homography: cur ~ ref shifted by (dx, dy).
    A homography fitted mostly to distant trees leaves a ground-plane error of ~10 px; the car's own ground
    context (paint lines, neighbours) removes it. Returns (0, 0, response) if the correlation is weak."""
    if ref_ctx.shape != cur_ctx.shape or min(ref_ctx.shape) < 16:
        return 0.0, 0.0, 0.0
    win = cv2.createHanningWindow((ref_ctx.shape[1], ref_ctx.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref_ctx, cur_ctx, win)
    if resp < 0.10 or np.hypot(dx, dy) > max_px:
        return 0.0, 0.0, float(resp)
    return float(dx), float(dy), float(resp)


def ref_image(al: Align, spot: Spot) -> np.ndarray:
    """The frame the spot was chosen in, in reference coordinates (a car seen before a cut is not in the alignment frame)."""
    k = int(spot.t_ref)
    if abs(spot.t_ref - al.t_ref) > 1.5 and k in al.jpgs:
        return cv2.imdecode(np.frombuffer(al.jpgs[k], np.uint8), cv2.IMREAD_COLOR)
    return al.ref_frame


def precrop(al: Align, spots: list[Spot]) -> list[tuple[dict[int, np.ndarray], dict[int, tuple[float, float]]]]:
    """Per spot: crops for every second it is visible (taken at the locally re-registered position) and the local shifts."""
    vis = [set(visible_seconds(al, s.box)) for s in spots]
    out: list[tuple[dict[int, np.ndarray], dict[int, tuple[float, float]]]] = [({}, {}) for _ in spots]
    ref_ctx = [_ctx(ref_image(al, s), s.box) for s in spots]
    for k in sorted(al.jpgs):
        need = [i for i, v in enumerate(vis) if k in v]
        if not need:
            continue
        warped = cv2.imdecode(np.frombuffer(al.jpgs[k], np.uint8), cv2.IMREAD_COLOR)
        for i in need:
            sp = spots[i]
            dx, dy, _ = local_shift(ref_ctx[i], _ctx(warped, sp.box))
            m = spot_margin(sp)
            dx = float(np.clip(dx, -sp.box[0], al.W - sp.box[2]))
            dy = float(np.clip(dy, -sp.box[1], al.H - sp.box[3]))
            bx = (sp.box[0] + dx, sp.box[1] + dy, sp.box[2] + dx, sp.box[3] + dy)
            out[i][0][k] = _spot_crop(warped, bx, m)
            out[i][1][k] = (dx, dy)
    return out


def build(al: Align, tr: pk.Tracks, F: dict[str, Any], spot: Spot,
          pre: tuple[dict[int, np.ndarray], dict[int, tuple[float, float]]] | None = None) -> Ledger:
    video, nsec, W, H, Ms_all = al.video, al.nsec, al.W, al.H, al.Ms
    margin = spot_margin(spot)
    ref_crop = _spot_crop(ref_image(al, spot), spot.box, margin)
    if pre is None:
        pre = precrop(al, [spot])[0]
    cur_crop, shifts = pre
    vis = set(cur_crop)
    Ms: list[Any] = [Ms_all[k] if k in vis else None for k in range(nsec)]
    app = [float("nan")] * nsec
    for k in vis:
        app[k] = compare(ref_crop, cur_crop[k], margin)["score"]
    crops: dict[int, np.ndarray] = {}
    # --- health per second from the 5 fps scalars
    t = F["t"]
    sec = t.astype(int)
    run = frozen_mask(F)
    ref_i = int(np.argmin(np.abs(t - al.t_ref)))
    lo, hi = max(0, ref_i - 8), min(len(t), ref_i + 9)
    ref_lap = float(np.median(F["lap"][lo:hi]))
    ref_mean = float(np.median(F["mean"][lo:hi]))
    ref_std = float(np.median(F["std"][lo:hi]))
    hm, hs, hl, hf = (np.full(nsec, np.nan) for _ in range(4))
    for k in range(nsec):
        idx = np.where(sec == k)[0]
        if len(idx) == 0:
            continue
        hm[k], hs[k], hl[k] = np.median(F["mean"][idx]), np.median(F["std"][idx]), np.median(F["lap"][idx])
        hf[k] = run[idx].mean()
    # --- detections per second, mapped into reference coordinates with that second's transform
    rows = tr.rows[tr.rows["name"].isin(VEH + ("person",)) & (tr.rows["conf"] >= 0.30)]
    rows = rows.assign(sec=rows["t"].astype(int))
    by_sec = {k: g for k, g in rows.groupby("sec")}
    state, cause = ["G"] * nsec, [""] * nsec
    sbox = spot.box
    sarea = (sbox[2] - sbox[0]) * (sbox[3] - sbox[1])
    n_frames = max(1, int(round(tr.fps)))
    for k in range(nsec):
        if np.isnan(hm[k]):
            cause[k] = "view"
            continue
        if covered_frame(hm[k], hs[k], ref_mean, ref_std):
            cause[k] = "covered"
            continue
        if hf[k] >= 0.6:
            cause[k] = "frozen"
            continue
        low_lap = ref_lap > 0 and hl[k] < P["blur_ratio"] * ref_lap
        if Ms[k] is None:
            # features lost but the coarse layout still equals the reference => same view gone soft, not a moved camera
            same_layout = k < len(al.coarse) and al.coarse[k] >= P["coarse_same"] and al.Ms[k] is None
            cause[k] = "blurred" if (low_lap and same_layout) else "view"
            continue
        if low_lap:
            cause[k] = "blurred"
            continue
        g = by_sec.get(k)
        match_f: set[int] = set()
        occl = 0.0
        if g is not None and len(g):
            bxs = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float64)
            corners = np.stack([bxs[:, [0, 1]], bxs[:, [2, 1]], bxs[:, [2, 3]], bxs[:, [0, 3]]], axis=1).reshape(-1, 2)
            mc = apply_M(Ms[k], corners).reshape(-1, 4, 2) - np.array(shifts.get(k, (0.0, 0.0)))
            mx1, my1, mx2, my2 = mc[:, :, 0].min(1), mc[:, :, 1].min(1), mc[:, :, 0].max(1), mc[:, :, 1].max(1)
            ix = np.clip(np.minimum(mx2, sbox[2]) - np.maximum(mx1, sbox[0]), 0, None)
            iy = np.clip(np.minimum(my2, sbox[3]) - np.maximum(my1, sbox[1]), 0, None)
            inter = ix * iy
            iou_v = inter / np.maximum((mx2 - mx1) * (my2 - my1) + sarea - inter, 1e-6)
            isveh = g["name"].isin(VEH).to_numpy()
            hit = isveh & (iou_v >= P["iou_present"])
            match_f = set(g["f"].to_numpy()[hit].astype(int).tolist())
            rest = ~hit
            if rest.any():
                occl = float((inter[rest] / sarea).max())
        det_present = len(match_f) >= max(2, int(0.25 * n_frames))
        a = app[k]
        if det_present and a < P["app_diff"]:
            state[k], cause[k] = "D", ""
        elif det_present:
            state[k] = "P"
        elif occl >= P["occl_cover"]:
            cause[k] = "occluded"
        elif a >= P["app_present"]:
            state[k] = "P"
        elif a >= P["app_absent"]:
            cause[k] = "unsure"
        else:
            state[k] = "A"
    # --- flicker filter: a single odd second between two identical neighbours takes the neighbours' state
    for k in range(1, nsec - 1):
        if state[k] != state[k - 1] and state[k - 1] == state[k + 1] and state[k - 1] in "PA" and state[k] in "PAD":
            state[k] = state[k - 1]
    # an 'unsure' second whose nearest decisive neighbours (within 3 s, both sides) agree takes their state
    for k in range(nsec):
        if cause[k] == "unsure":
            prev = next((q for q in range(k - 1, max(-1, k - 4), -1) if state[q] in "PA"), None)
            nxt = next((q for q in range(k + 1, min(nsec, k + 4)) if state[q] in "PA"), None)
            if prev is not None and nxt is not None and state[prev] == state[nxt]:
                state[k], cause[k] = state[prev], ""
    # a 1-2 s alignment dropout ('view'/'unsure') with the SAME decisive state on both sides is not a blind stretch: the car
    # cannot leave and come back in 2 s. It takes the neighbours' state and is counted in params['filled_gap_s'] (no hiding).
    filled: list[int] = []
    k = 0
    while k < nsec:
        if state[k] == "G" and cause[k] in ("view", "unsure"):
            j = k
            while j + 1 < nsec and state[j + 1] == "G" and cause[j + 1] in ("view", "unsure"):
                j += 1
            if (j - k + 1) <= MAX_FILL_S and k > 0 and j + 1 < nsec and state[k - 1] in "PA" and state[k - 1] == state[j + 1]:
                for q in range(k, j + 1):
                    state[q], cause[q] = state[k - 1], ""
                    filled.append(q)
            k = j + 1
        else:
            k += 1
    for k in range(nsec):
        if state[k] != "G" and k in cur_crop:
            crops[k] = cur_crop[k]
    ok_secs = [k for k in range(nsec) if Ms[k] is not None]
    window = (min(ok_secs), max(ok_secs) + 1) if ok_secs else (0, 0)
    return Ledger(video, float(F["t"][-1] + 0.2), spot, state, cause, app, Ms, crops, margin, al.ref_frame, ref_crop, window,
                  {"mean": hm, "std": hs, "lap": hl, "frozen": hf, "ref_lap": np.array(ref_lap), "ref_mean": np.array(ref_mean)},
                  {"fps": float(F["fps"]), "filled_gap_s": sorted(filled)})


# ---------------------------------------------------------------------------------------- stretches, contacts
@dataclass
class Stretch:
    s: int
    e: int                      # [s, e)
    causes: dict[str, int]
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    cmp: dict[str, float] | None = None
    verdict: str = ""

    @property
    def dur(self) -> int:
        return self.e - self.s

    @property
    def cause(self) -> str:
        return max(self.causes, key=self.causes.get) if self.causes else "unknown"


def stretches(led: Ledger, min_len: int = 1) -> list[Stretch]:
    """Can't-see stretches inside the ledger window, each with before/after comparison and verdict."""
    w0, w1 = led.window
    out: list[Stretch] = []
    k = w0
    while k < w1:
        if led.state[k] == "G":
            j = k
            while j + 1 < w1 and led.state[j + 1] == "G":
                j += 1
            c: dict[str, int] = {}
            for q in range(k, j + 1):
                c[led.cause[q]] = c.get(led.cause[q], 0) + 1
            st = Stretch(k, j + 1, c)
            if st.dur >= min_len:
                out.append(st)
            k = j + 1
        else:
            k += 1
    for st in out:
        if st.dur < MIN_BLIND_S:
            st.verdict = "short gap (alignment/occlusion dropout)"
            continue
        bs = [q for q in range(max(w0, st.s - 6), st.s) if led.state[q] != "G" and q in led.crops][-3:]
        as_ = [q for q in range(st.e, min(w1, st.e + 6)) if led.state[q] != "G" and q in led.crops][:3]
        if not bs or not as_:
            st.verdict = "UNKNOWN: no clear view " + ("before" if not bs else "after")
            continue

        def pick(qs):
            return max(qs, key=lambda q: cv2.Laplacian(cv2.cvtColor(led.crops[q], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        qb, qa = pick(bs), pick(as_)
        sb, sa = led.state[qb], led.state[qa]
        st.before = {"sec": qb, "state": sb}
        st.after = {"sec": qa, "state": sa}
        if sb == "P" and sa == "A":
            st.verdict = "CAR GONE: left during the blind stretch"
        elif sb == "A" and sa == "P":
            st.verdict = "CAR BACK: arrived during the blind stretch"
        elif sb == "A" and sa == "A":
            st.verdict = "spot empty before and after"
        else:
            c = compare(led.crops[qb], led.crops[qa], led.margin)
            st.cmp = c
            st.verdict = "CHANGED: check" if (looks_changed(c) or "D" in (sb, sa)) else "same car"
    return out


def contacts(led: Ledger, tr: pk.Tracks, *, pad: float | None = None, min_dur: float = 0.4, merge_gap: float = 1.5
             ) -> list[dict[str, Any]]:
    """People whose FEET (bottom-centre of the person box) enter a dilated footprint zone around the car."""
    pad = P["contact_pad"] if pad is None else pad
    x1, y1, x2, y2 = led.spot.box
    bw, bh = x2 - x1, y2 - y1
    px = max(12.0, pad * bw)
    zone = (x1 - px, y1 + 0.45 * bh - 0.5 * px, x2 + px, y2 + px)    # lower half of the car (ground contact) + pad
    pr = tr.rows[(tr.rows["name"] == "person") & (tr.rows["conf"] >= 0.35)]
    hits = []
    for f, t, tid, a1, b1, a2, b2, cf in pr[["f", "t", "tid", "x1", "y1", "x2", "y2", "conf"]].itertuples(index=False):
        k = int(t)
        if k >= len(led.Ms) or led.Ms[k] is None or not (led.window[0] <= k < led.window[1]):
            continue
        fx, fy = apply_M(led.Ms[k], [[(a1 + a2) / 2, b2]])[0]
        if zone[0] <= fx <= zone[2] and zone[1] <= fy <= zone[3]:
            inside = x1 <= fx <= x2 and y1 + 0.45 * bh <= fy <= y2
            dist = 0.0 if inside else float(np.hypot(max(x1 - fx, 0, fx - x2), max(y1 + 0.45 * bh - fy, 0, fy - y2)))
            hits.append((float(t), int(tid), dist, (a1, b1, a2, b2), k, float(cf)))
    hits.sort()
    events: list[dict[str, Any]] = []
    for h in hits:
        for ev in events:
            if h[0] - ev["t1"] <= merge_gap:        # merge by time proximity (ids break; people near a car are rare)
                ev["t1"] = h[0]
                ev["hits"].append(h)
                break
        else:
            events.append({"t0": h[0], "t1": h[0], "hits": [h]})
    out = []
    for ev in events:
        best = min(ev["hits"], key=lambda h: h[2])
        if ev["t1"] - ev["t0"] < min_dur:
            continue
        out.append({"t0": round(ev["t0"], 1), "t1": round(ev["t1"], 1), "dur": round(ev["t1"] - ev["t0"], 1),
                    "min_dist_px": round(best[2], 1), "touch_zone": bool(best[2] == 0.0), "t_best": round(best[0], 2),
                    "box_best": [round(v, 1) for v in best[3]], "sec_best": best[4], "n_frames": len(ev["hits"]),
                    "tids": sorted({h[1] for h in ev["hits"]})})
    return out


def contact_check(led: Ledger, c: dict[str, Any], *, pad_s: int = 1, max_gap_s: int = 40) -> dict[str, Any]:
    """TOUCH? beat: compare the car crop just before a person's visit with the crop just after it (same SSIM test as the
    blind stretches). Only says 'same car or not'; fine damage (scratches) is outside what this test sees."""
    t0, t1 = int(c["t0"]), int(np.ceil(c["t1"]))
    before = [q for q in range(t0 - pad_s, max(-1, t0 - max_gap_s), -1) if q in led.crops and led.state[q] in "PD"]
    after = [q for q in range(t1 + pad_s, min(len(led.state), t1 + max_gap_s)) if q in led.crops and led.state[q] in "PD"]
    if not before or not after:
        return {"verdict": "n/a: no clear view of the car " + ("before" if not before else "after") + " the visit"}
    qb, qa = before[0], after[0]
    cm = compare(led.crops[qb], led.crops[qa], led.margin)
    ch = looks_changed(cm) or led.state[qb] == "D" or led.state[qa] == "D"
    return {"before_sec": qb, "after_sec": qa, "ssim": round(cm["ssim"], 3), "block_min": round(cm["block_min"], 3),
            "verdict": "CHANGED: check" if ch else "same car"}


def summarize(led: Ledger) -> dict[str, Any]:
    w0, w1 = led.window
    st = led.state[w0:w1]
    n = max(1, len(st))
    cnt = {k: st.count(k) for k in "PADG"}
    causes: dict[str, int] = {}
    for q in range(w0, w1):
        if led.state[q] == "G":
            causes[led.cause[q]] = causes.get(led.cause[q], 0) + 1
    return {"window_s": [w0, w1], "seconds": n, "present_s": cnt["P"], "absent_s": cnt["A"], "different_s": cnt["D"],
            "cant_see_s": cnt["G"], "cant_see_causes": causes, "present_pct": round(100 * cnt["P"] / n, 1),
            "cant_see_pct": round(100 * cnt["G"] / n, 1),
            "filled_gap_s": len([q for q in led.params.get("filled_gap_s", []) if w0 <= q < w1])}


def spans(led: Ledger, window_only: bool = True) -> list[tuple[int, int, str, str]]:
    """Run-length spans (s, e, code, cause) over the window; code in P/A/D/G."""
    w0, w1 = led.window if window_only else (0, len(led.state))
    out, k = [], w0
    while k < w1:
        j = k
        while j + 1 < w1 and led.state[j + 1] == led.state[k] and led.cause[j + 1] == led.cause[k]:
            j += 1
        out.append((k, j + 1, led.state[k], led.cause[k]))
        k = j + 1
    return out
