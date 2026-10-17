"""fastindex: make camera videos followable in seconds, CPU only, no models.

index_video(path, cam) decodes with cv2 at 2 fps (grab every frame, retrieve only the samples; times are frame
timestamps, so variable frame rate phone video stays in step with the browser's playback), runs MOG2 on a
640-wide copy primed with a median background plate, drops foreground that has not changed for 3 s (plate ghosts,
people standing still) and remembers what those pixels showed (later foreground that looks the same there is dropped
too: a ghost that MOG2 keeps re-flagging, but also a look-alike standing on that exact spot), turns foreground blobs
into boxes and links them with
stm.tracker.track (greedy IoU + centre distance). Each fragment track gets normalised boxes [t, x, y, w, h] in VIDEO
seconds, a 12-bin colour signature of its upper and lower body halves (foreground pixels only), colour words and the
best crop (largest sharp box) as JPEG bytes in `_crop`. index_cameras runs several videos in parallel threads.

Measured earlier on the dev clips: MOG2 at 2 fps gave recall 0.68 / precision 0.70 against YOLO; 5 fps was worse and
HOG was useless, so there is no person classifier: a track is a "person" when its blob is mostly taller than wide.
People who stand still for more than ~10 s fade into the background model. One visit often splits into several
fragments; stm.follow.merge_fragments joins them.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm import tracker

COLORS = ("red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown", "black", "white", "grey")
WORDS = {"cyan": "blue"}        # find.html's colour list has no cyan
FPS = 2.0
WORK_W = 640                    # MOG2 working width
LR = 1 / 240                    # MOG2 learning rate per sample (~2 min memory at 2 fps)
BG_RATIO = 0.7                  # MOG2 background ratio: lower = someone standing still takes longer to fade
WARMUP = 2                      # samples after priming before blobs count
MIN_AREA = 0.0003               # blob area as a share of the frame
MIN_H = 0.04                    # blob height as a share of the frame
MAX_FG = 0.35                   # a sample with more foreground than this is a global change (light, shake): skipped
TALL = 1.2                      # h >= TALL * w counts as person-shaped
CROP_H = 240                    # stored crops are at most this tall
TOP_K = 8                       # signature = mean of the TOP_K largest observations
PLATE_N = 9                     # frames in the median background plate
HOLD = 6                        # foreground pixels unchanged (grey diff < STILL_DIFF) for this many samples (3 s) are
STILL_DIFF = 8                  # dropped: background ghosts from the plate, or someone standing still. What such a
ALT_DIFF = 14                   # pixel showed is kept; later foreground within ALT_DIFF of it is dropped too
STATIC = 0.25                   # a track whose centre never moves this many box heights is static (a background
                                # ghost or someone who never moved): kind "static", never a person

# hue bands (OpenCV 0..179) for chromatic pixels
_HUES = [(0, 8, "red"), (8, 20, "orange"), (20, 34, "yellow"), (34, 82, "green"), (82, 100, "cyan"),
         (100, 130, "blue"), (130, 150, "purple"), (150, 170, "pink"), (170, 180, "red")]
_IDX = {c: i for i, c in enumerate(COLORS)}
# neighbouring colours share a little mass before comparing, so lighting shifts cost less
_NEAR = {"red": ("orange", "pink", "brown"), "orange": ("red", "yellow", "brown"), "yellow": ("orange",),
         "green": ("cyan",), "cyan": ("green", "blue"), "blue": ("cyan", "purple"), "purple": ("blue", "pink"),
         "pink": ("red", "purple"), "brown": ("orange", "red", "black"), "black": ("grey", "brown"),
         "white": ("grey",), "grey": ("black", "white")}


def _smooth_matrix(keep: float = 0.7) -> np.ndarray:
    m = np.zeros((len(COLORS), len(COLORS)))
    for c, nb in _NEAR.items():
        i = _IDX[c]
        m[i, i] = keep
        for n in nb:
            m[i, _IDX[n]] = (1 - keep) / len(nb)
    return m


SMOOTH = _smooth_matrix()


def colour_hist(hsv: np.ndarray) -> np.ndarray:
    """12-bin colour histogram (sums to 1) of HSV pixels, shape (n, 3)."""
    v = np.zeros(len(COLORS))
    if len(hsv) == 0:
        return v
    H, S, V = hsv[:, 0].astype(np.int32), hsv[:, 1].astype(np.int32), hsv[:, 2].astype(np.int32)
    chroma = (S >= 60) & (V >= 50)
    brown = chroma & (H >= 5) & (H < 25) & (V < 140)
    pinkish = chroma & ((H < 8) | (H >= 170)) & (S < 110) & (V >= 170)
    for lo, hi, name in _HUES:
        v[_IDX[name]] += np.count_nonzero(chroma & ~brown & ~pinkish & (H >= lo) & (H < hi))
    v[_IDX["brown"]] += np.count_nonzero(brown)
    v[_IDX["pink"]] += np.count_nonzero(pinkish)
    ach = ~chroma
    white = ach & (V >= 180) & (S < 50)
    black = ach & (V < 60)
    v[_IDX["black"]] += np.count_nonzero(black)
    v[_IDX["white"]] += np.count_nonzero(white)
    v[_IDX["grey"]] += np.count_nonzero(ach & ~white & ~black)
    return v / max(1.0, v.sum())


def colour_word(hist: list[float] | np.ndarray) -> str | None:
    """The dominant colour as a word from find.html's list, or None for an empty histogram."""
    h = np.asarray(hist, float)
    if h.sum() <= 0:
        return None
    c = COLORS[int(h.argmax())]
    return WORDS.get(c, c)


def hsv_distance(a: dict[str, Any], b: dict[str, Any], w_upper: float = 0.6) -> float:
    """0 (same colours) .. 1 (nothing shared): 1 - histogram intersection after neighbour smoothing, upper and lower
    halves weighted w_upper : 1 - w_upper."""
    d = []
    for k in ("upper", "lower"):
        x, y = np.asarray(a[k], float) @ SMOOTH, np.asarray(b[k], float) @ SMOOTH
        d.append(1.0 - float(np.minimum(x, y).sum()) if x.sum() > 0 and y.sum() > 0 else 1.0)
    return round(w_upper * d[0] + (1 - w_upper) * d[1], 4)


def look_of(sig: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Colour words for a signature: (look, desc)."""
    top, bottom = colour_word(sig["upper"]), colour_word(sig["lower"])
    look = {"top": top, "bottom": bottom, "items": [], "source": "hsv"}
    parts = [f"{top} top" if top else None, f"{bottom} lower body" if bottom else None]
    return look, ", ".join(p for p in parts if p) or "colours unclear"


def _join(boxes: list[list[int]]) -> list[list[int]]:
    """Join blobs of one body split by the background (head / torso / legs): horizontal overlap >= 0.6 x the
    narrower box and a vertical gap <= 0.08 x the joined height (two people one above the other stay apart)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                gy = max(a[1], b[1]) - min(a[3], b[3])
                if ox >= 0.6 * min(a[2] - a[0], b[2] - b[0]) and gy <= 0.08 * (max(a[3], b[3]) - min(a[1], b[1])):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _plate(cap: cv2.VideoCapture, n_frames: int, size: tuple[int, int], max_frame: int) -> np.ndarray | None:
    """Median of PLATE_N frames spread over the video (jittered, so a periodic motion is not sampled in step):
    people walking through are not in it."""
    last = max(1, min(n_frames, max_frame) - 1)
    step = last / PLATE_N
    jitter = np.random.default_rng(7).uniform(0.1, 0.9, PLATE_N)
    imgs = []
    for k in ((np.arange(PLATE_N) + jitter) * step).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, img = cap.read()
        if ok:
            imgs.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(imgs), axis=0).astype(np.uint8) if imgs else None


def frame_time(cap: cv2.VideoCapture, i: int, vfps: float) -> float:
    """Seconds of the frame just grabbed: its timestamp (right for variable frame rate phone video, and what a browser
    plays), else frame index / fps when the backend gives none."""
    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    return ms / 1000.0 if ms > 0 else i / vfps


def _halves(hsv: np.ndarray, mask: np.ndarray, b: list[int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = b
    h, w = y2 - y1, x2 - x1
    out = []
    for a, z in ((0.12, 0.5), (0.5, 0.92)):
        ya, yb = y1 + int(a * h), max(y1 + int(z * h), y1 + int(a * h) + 1)
        xa, xb = x1 + int(0.15 * w), max(x1 + int(0.85 * w), x1 + int(0.15 * w) + 1)
        px, m = hsv[ya:yb, xa:xb].reshape(-1, 3), mask[ya:yb, xa:xb].reshape(-1) > 0
        out.append(colour_hist(px[m] if m.sum() >= 12 else px))
    return out[0], out[1]


def index_video(path: str | Path, cam: str = "A", fps: float = FPS, max_s: float | None = None,
                work_w: int = WORK_W) -> dict[str, Any]:
    """Index one video. Returns {camera, path, fps, size, duration, samples, skipped, seconds, tracks}; tracks are
    fragments in video seconds (see module docstring) with `kind` "person" (moving, mostly taller than wide),
    "object" (moving, not person-shaped) or "static" (never moved: a background ghost or someone standing still)."""
    t_start = time.perf_counter()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ww = min(work_w, W)
    wh = max(2, int(round(H * ww / W)))
    max_frame = n_frames if max_s is None else min(n_frames or 10 ** 9, int(max_s * vfps) + 1)
    mog = cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=25, detectShadows=True)
    mog.setBackgroundRatio(BG_RATIO)
    plate = _plate(cap, n_frames, (ww, wh), max_frame) if n_frames > PLATE_N else None
    if plate is not None:
        for k in range(3):
            mog.apply(plate, learningRate=1.0 if k == 0 else 0.5)
    k_open = np.ones((3, 3), np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11))
    min_area, min_h = MIN_AREA * ww * wh, MIN_H * wh
    sx, sy = W / ww, H / wh
    rows: list[dict[str, Any]] = []
    extra: list[tuple[np.ndarray, np.ndarray, bool, float, float, bytes | None]] = []
    step, next_t, i, k, skipped, last_t = 1.0 / fps, 0.0, 0, 0, 0, None
    prev, held = None, np.zeros((wh, ww), np.uint8)
    alt, alt_ok = np.zeros((wh, ww, 3), np.uint8), np.zeros((wh, ww), np.uint8)    # what ghost pixels really show
    alt_box = None
    while cap.grab():
        t = frame_time(cap, i, vfps)
        if max_s is not None and t > max_s + 1e-6:
            break
        last_t = t
        if t + 1e-6 >= next_t:
            while next_t <= t + 1e-6:
                next_t += step
            ok, img = cap.retrieve()
            if ok:
                small = cv2.resize(img, (ww, wh), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                fg = mog.apply(small, learningRate=LR if plate is not None or k > 4 else -1)
                k += 1
                mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
                if prev is not None:        # foreground that has not changed for HOLD samples is a ghost or still:
                    same = cv2.threshold(cv2.absdiff(gray, prev), STILL_DIFF - 1, 255, cv2.THRESH_BINARY_INV)[1]
                    still = cv2.bitwise_and(mask, same)
                    held = cv2.bitwise_and(cv2.add(held, 1), still)       # samples unchanged (0/1 masks are 0/255)
                    ghost = cv2.threshold(held, HOLD - 1, 255, cv2.THRESH_BINARY)[1]
                    if cv2.countNonZero(ghost):                           # remember what it shows, drop it from now on
                        cv2.copyTo(small, ghost, alt)
                        alt_ok = cv2.bitwise_or(alt_ok, ghost)
                        x, y, w, h = cv2.boundingRect(alt_ok)
                        alt_box = (slice(y, y + h), slice(x, x + w))
                        mask = cv2.bitwise_and(mask, cv2.bitwise_not(ghost))
                    if alt_box is not None:
                        d = cv2.split(cv2.absdiff(small[alt_box], alt[alt_box]))
                        d = cv2.max(cv2.max(d[0], d[1]), d[2])
                        same = cv2.threshold(d, ALT_DIFF - 1, 255, cv2.THRESH_BINARY_INV)[1]
                        mb = mask[alt_box]
                        mb[cv2.bitwise_and(alt_ok[alt_box], same) > 0] = 0
                prev = gray
                if k > WARMUP:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
                    if cv2.countNonZero(mask) > MAX_FG * mask.size:
                        skipped += 1
                    else:
                        _detect(t, k, img, small, gray, mask, min_area, min_h, sx, sy, rows, extra)
        i += 1
    cap.release()
    duration = last_t + 1.0 / vfps if last_t is not None else 0.0
    tracks = _assemble(cam, rows, extra, ww, wh)
    return {"camera": cam, "path": str(path), "fps": fps, "size": [W, H], "duration": round(duration, 2),
            "samples": k, "skipped": skipped, "seconds": round(time.perf_counter() - t_start, 2), "tracks": tracks}


def _detect(t: float, k: int, img: np.ndarray, small: np.ndarray, gray: np.ndarray, mask: np.ndarray,
            min_area: float, min_h: float, sx: float, sy: float, rows: list, extra: list) -> None:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = [[int(x), int(y), int(x + w), int(y + h)] for x, y, w, h, a in stats[1:n] if a >= min_area / 3]
    boxes = [b for b in _join(boxes) if (b[2] - b[0]) * (b[3] - b[1]) >= min_area and b[3] - b[1] >= min_h]
    if not boxes:
        return
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    Hf, Wf = img.shape[:2]
    for b in boxes:
        x1, y1, x2, y2 = b
        up, lo = _halves(hsv, mask, b)
        tall = (y2 - y1) >= TALL * (x2 - x1)
        sharp = float(cv2.Laplacian(gray[y1:y2, x1:x2], cv2.CV_64F).var())
        jpg = None
        if tall:
            px, py = 0.1 * (x2 - x1) * sx, 0.08 * (y2 - y1) * sy
            X1, Y1 = int(max(0, x1 * sx - px)), int(max(0, y1 * sy - py))
            X2, Y2 = int(min(Wf, x2 * sx + px)), int(min(Hf, y2 * sy + py))
            crop = img[Y1:Y2, X1:X2]
            if crop.size:
                if crop.shape[0] > CROP_H:
                    f = CROP_H / crop.shape[0]
                    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * f)), CROP_H), interpolation=cv2.INTER_AREA)
                ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpg = enc.tobytes() if ok else None
        rows.append({"f": k, "t": t, "cls": 0, "name": "blob", "conf": 1.0, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        extra.append((up, lo, tall, float((x2 - x1) * (y2 - y1)), sharp, jpg))


def _assemble(cam: str, rows: list, extra: list, ww: int, wh: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = tracker.track(pd.DataFrame(rows), max_age=1.0, iou_min=0.1, dist_max=0.9, min_hits=3)
    out = []
    for tid, g in df[df.tid >= 0].groupby("tid"):
        g = g.sort_values("t")
        ex = [extra[j] for j in g.index]
        area = np.array([e[3] for e in ex])
        top = np.argsort(-area)[:TOP_K]
        sig = {"upper": [round(float(v), 4) for v in np.mean([ex[j][0] for j in top], axis=0)],
               "lower": [round(float(v), 4) for v in np.mean([ex[j][1] for j in top], axis=0)]}
        sharp = np.array([e[4] for e in ex])
        med = float(np.median(sharp)) or 1.0
        score = area * np.minimum(1.0, sharp / med)
        score[[e[5] is None for e in ex]] = -1
        best = int(score.argmax())
        look, desc = look_of(sig)
        tall = float(np.mean([e[2] for e in ex]))
        boxes = [[round(float(r.t), 2), round(r.x1 / ww, 4), round(r.y1 / wh, 4), round((r.x2 - r.x1) / ww, 4),
                  round((r.y2 - r.y1) / wh, 4)] for r in g.itertuples()]
        c = np.array([[b[1] + b[3] / 2, (b[2] + b[4] / 2) * wh / ww] for b in boxes])    # width units
        moved = float(np.hypot(*(c - c[0]).T).max()) / max(1e-6, float(np.median([b[4] for b in boxes])) * wh / ww)
        kind = "static" if moved < STATIC else "person" if tall >= 0.5 else "object"
        out.append({"id": f"{cam}-{int(tid)}", "cam": cam, "kind": kind, "moved": round(moved, 2),
                    "t0": boxes[0][0], "t1": boxes[-1][0], "n": len(boxes), "tall": round(tall, 2),
                    "boxes": boxes, "sig": sig, "look": look, "desc": desc,
                    "crop_t": boxes[best][0] if score[best] >= 0 else None,
                    "_crop": ex[best][5] if score[best] >= 0 else None,
                    "_crop_score": float(score[best]), "fragments": [f"{cam}-{int(tid)}"]})
    return out


def index_cameras(jobs: list[dict[str, Any]], fps: float = FPS, max_s: float | None = None,
                  workers: int | None = None) -> list[dict[str, Any]]:
    """jobs: [{"cam": id, "video": path}] -> index_video results in the same order, run in parallel threads
    (cv2 decode and MOG2 release the GIL)."""
    with ThreadPoolExecutor(max_workers=workers or max(1, len(jobs))) as ex:
        futs = [ex.submit(index_video, j["video"], j["cam"], fps, max_s) for j in jobs]
        return [f.result() for f in futs]


def save_crop(track: dict[str, Any], path: str | Path) -> str | None:
    """Write the track's best crop (JPEG bytes from indexing) to path; returns the path or None."""
    if not track.get("_crop"):
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(track["_crop"])
    return str(p)
