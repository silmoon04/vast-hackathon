"""UNCUT core: archive index (5 s Cosmos-Embed1 segments + per-frame dHash variants) and two-stage clip alignment.

Stage 1: Cosmos-Embed1 video embeddings of 5 s windows, kNN over the archive segments -> candidate windows.
Stage 2: dHash frame-sequence alignment (offset, speed, mirror, crop, jumps = cuts, unmatched spans = inserted
foreign frames) restricted to the candidate videos.
"""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE / "_work"
HFPS = 5.0                       # hash sampling rate (frames per second of the *clip being hashed*)
GW = 256                         # grey thumbnail width kept for hashing
HW, HH = 17, 16                  # dHash grid -> 16 x 16 = 256 bits
NB = (HW - 1) * HH


class Var(NamedTuple):
    """One way the viral clip may have been derived from an archive frame: crop kind/size/position + mirror."""
    kind: str        # "sq" = uniform crop of `crop` x `crop` of the frame; "p916" = full-height 9:16 portrait strip
    crop: float      # sq: fraction kept per axis; p916: unused (1.0)
    ox: float        # horizontal position of the crop window: 0 = left edge, 0.5 = centre, 1 = right edge
    mirror: bool


def _variants():
    base = [Var("sq", c, 0.5, False) for c in (1.0, 0.9, 0.8, 0.7, 0.6)]
    base += [Var("sq", c, ox, False) for c in (0.8, 0.7) for ox in (0.0, 1.0)]       # off-centre (left / right) crops
    base += [Var("p916", 1.0, ox, False) for ox in (0.0, 0.5, 1.0)]                    # 16:9 -> 9:16 portrait re-crop
    return base + [v._replace(mirror=True) for v in base]


VARIANTS = _variants()     # 12 crops x {normal, mirrored} = 24 archive-side variants per frame


def vname(v: Var) -> str:
    if v.kind == "p916":
        return "p916" + {0.0: "L", 0.5: "C", 1.0: "R"}[v.ox] + ("M" if v.mirror else "")
    return f"{v.crop:.1f}" + {0.0: "L", 0.5: "", 1.0: "R"}[v.ox] + ("M" if v.mirror else "")


def variant_by_name(n: str) -> Var:
    return next(v for v in VARIANTS if vname(v) == n)


def crop_box(v: Var, h: int, w: int, unmirror=False) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) as fractions of the archive frame kept by this variant. unmirror=True gives the box on the
    UNMIRRORED archive frame (for drawing); otherwise it is the box in the (possibly mirrored) frame that is hashed."""
    if v.kind == "p916":
        cw, ch = min(1.0, 0.5625 * h / w), 1.0
    else:
        cw = ch = v.crop
    x0, y0 = (1 - cw) * v.ox, (1 - ch) / 2
    box = (x0, y0, x0 + cw, y0 + ch)
    if unmirror and v.mirror:
        box = (1 - box[2], box[1], 1 - box[0], box[3])
    return box


def crop_desc(v: Var) -> str:
    where = {0.0: "left", 0.5: "centre", 1.0: "right"}[v.ox]
    if v.mirror and v.ox != 0.5:
        where = {"left": "right", "right": "left"}[where]       # described on the unmirrored archive frame
    if v.kind == "p916":
        return f"9:16 portrait re-crop ({where})"
    if v.crop >= 1.0:
        return "none"
    return f"{v.crop * 100:.0f}%" + ("" if v.ox == 0.5 else f", {where}-aligned")


SEG = 5.0

ARCHIVE = ["retail_cctv", "lisbon_parking", "ptz_parking", "rochester_night", "yonge_dundas", "motorway_a40",
           "dashcam_urban", "elephants_dream", "floor_lie", "assist_elderly", "help_sit", "car_detection", "classroom",
           "face_demographics_walking_and_pause", "one_by_one_person_detection", "people_detection",
           "person_bicycle_car_detection", "store_aisle_detection", "worker_zone_detection"]
# same-camera hold-out: lisbon_parking [295, 335) is NOT in the archive (hard negative test)
HOLDOUT = {"lisbon_parking": (295.0, 335.0)}


# ---------------------------------------------------------------------------------------------- frames + hashes
def grey_frames(path: str | Path, fps: float = HFPS) -> np.ndarray:
    """Decode a video with ffmpeg at `fps` into grey GW-wide thumbnails. Returns (n, h, GW) uint8."""
    path = str(path)
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    th = max(8, int(round(GW * h / w / 2)) * 2)
    cmd = [pk.FFMPEG, "-v", "error", "-i", path, "-vf", f"fps={fps},scale={GW}:{th}:flags=area", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, th, GW)


def dhash_bits(g: np.ndarray, v: Var) -> np.ndarray:
    """256-bit dHash (uint8 0/1, length NB) of one grey thumbnail under variant v (mirror, then crop window)."""
    if v.mirror:
        g = g[:, ::-1]
    h, w = g.shape
    if not (v.kind == "sq" and v.crop >= 1.0):
        x0, y0, x1, y1 = crop_box(v, h, w)
        g = g[int(round(y0 * h)):int(round(y1 * h)), int(round(x0 * w)):int(round(x1 * w))]
    s = cv2.resize(np.ascontiguousarray(g), (HW, HH), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (s[:, 1:] > s[:, :-1]).astype(np.uint8).ravel()


def hash_variants(frames: np.ndarray, variants=VARIANTS) -> dict[str, np.ndarray]:
    return {vname(v): np.stack([dhash_bits(g, v) for g in frames]) if len(frames) else np.zeros((0, NB), np.uint8)
            for v in variants}


def query_hashes(frames: np.ndarray) -> np.ndarray:
    v0 = Var("sq", 1.0, 0.5, False)
    return np.stack([dhash_bits(g, v0) for g in frames])


def strip_bars(fr: np.ndarray, thr=24, min_frac=0.04, dark=0.97, bridge=0.08):
    """Candidate removal of black letterbox/pillarbox bars. A row/column is 'bar' if >= `dark` of its pixels are <= thr in
    EVERY frame; gaps of up to `bridge` of the height/width are bridged so captions burned into the bars do not stop the
    detection. Returns (frames, info) with the fraction removed per side, or (frames, None) when there is no bar.
    This is only a HYPOTHESIS: verify() aligns both the raw and the stripped clip and keeps whichever explains more."""
    if len(fr) == 0:
        return fr, None
    isdark = fr <= thr
    rows = isdark.mean(axis=2).min(axis=0) >= dark          # (h,) dark in every frame
    cols = isdark.mean(axis=1).min(axis=0) >= dark          # (w,)
    h, w = fr.shape[1:]

    def lead(a, gap):
        last, i, gap = 0, 0, max(2, int(gap * len(a)))
        while i < len(a):
            if a[i]:
                last, i = i + 1, i + 1
                continue
            j = i
            while j < len(a) and not a[j]:
                j += 1
            if j < len(a) and j - i <= gap:
                i = j
            else:
                break
        return last
    t, b, l, r = lead(rows, bridge), lead(rows[::-1], bridge), lead(cols, bridge), lead(cols[::-1], bridge)
    t, b, l, r = (x if x >= min_frac * (h if i < 2 else w) else 0 for i, x in enumerate((t, b, l, r)))
    if not (t or b or l or r) or t + b >= h * 0.85 or l + r >= w * 0.85:
        return fr, None
    return fr[:, t:h - b, l:w - r], {"top": t / h, "bottom": b / h, "left": l / w, "right": r / w}


# ------------------------------------------------------------------------------------------------ archive index
def _hash_one(name: str):
    """Per-video hash arrays for every variant; reuses cached variants and computes only the missing ones."""
    f = WORK / "hash" / f"{name}.npz"
    have = {}
    if f.exists():
        z = np.load(f)
        have = {k: z[k] for k in z.files}
    need = [v for v in VARIANTS if vname(v) not in have]
    if need:
        fr = grey_frames(pk.video_path(name))
        have.update(hash_variants(fr, need))
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **have)
    return name, {vname(v): have[vname(v)] for v in VARIANTS}


def _cut_seg(name: str, s: float, e: float) -> Path:
    dst = WORK / "seg" / f"{name}_{int(round(s * 10)):06d}.mp4"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([pk.FFMPEG, "-y", "-v", "error", "-ss", f"{s:.3f}", "-i", str(pk.video_path(name)), "-t", f"{e - s:.3f}",
                    "-vf", "scale='trunc(min(320,iw)/2)*2':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                    "-pix_fmt", "yuv420p", "-an", "-g", "25", str(dst)], check=True)
    return dst


def embed_clip_windows(path: str | Path, windows: list[tuple[float, float]], tag: str) -> np.ndarray:
    outs = []
    for k, (s, e) in enumerate(windows):
        dst = WORK / "q" / f"{tag}_{k}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([pk.FFMPEG, "-y", "-v", "error", "-ss", f"{s:.3f}", "-i", str(path), "-t", f"{e - s:.3f}", "-vf",
                        "scale='trunc(min(320,iw)/2)*2':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                        "-pix_fmt", "yuv420p", "-an", "-g", "25", str(dst)], check=True)
        outs.append(dst)
    return pk.embed_video(outs)


class Archive:
    """segments table (video, start, end, vec) + per-video hash variants."""

    def __init__(self, names=ARCHIVE):
        self.names = list(names)
        self.hash: dict[str, dict[str, np.ndarray]] = {}
        self.dur: dict[str, float] = {}
        self.seg = []      # list of (video, start, end)
        self.vec = None

    @staticmethod
    def held_out(name: str, s: float, e: float) -> bool:
        h = HOLDOUT.get(name)
        return bool(h and s < h[1] and e > h[0])

    def build(self, verbose=True):
        import time
        t0 = time.time()
        with ThreadPoolExecutor(10) as ex:
            for name, hv in ex.map(_hash_one, self.names):
                self.hash[name] = hv
        for n in self.names:
            self.dur[n] = pk.info(n)["duration"]
        if verbose:
            print(f"[uncut] hashed {len(self.names)} videos in {time.time() - t0:.0f}s", flush=True)
        jobs = []
        for n in self.names:
            d = self.dur[n]
            k = 0
            while k * SEG < d - 0.5:
                s, e = k * SEG, min((k + 1) * SEG, d)
                if e - s >= 2.0 and not self.held_out(n, s, e):
                    jobs.append((n, s, e))
                k += 1
        t1 = time.time()
        with ThreadPoolExecutor(10) as ex:
            paths = list(ex.map(lambda j: _cut_seg(*j), jobs))
        if verbose:
            print(f"[uncut] cut {len(jobs)} segments in {time.time() - t1:.0f}s", flush=True)
        t1 = time.time()
        vecs = []
        for i in range(0, len(paths), 16):
            vecs.append(pk.embed_video(paths[i:i + 16]))
        self.vec = np.concatenate(vecs)
        self.vec /= np.linalg.norm(self.vec, axis=1, keepdims=True) + 1e-9
        self.seg = jobs
        if verbose:
            print(f"[uncut] embedded {len(jobs)} segments in {time.time() - t1:.0f}s", flush=True)
        self._finish()
        np.save(WORK / "index_vec.npy", self.vec)
        (WORK / "index_seg.json").write_text(json.dumps(self.seg))
        return self

    def _finish(self):
        """validity mask per video: archive hash frames inside a held-out window do not exist for the matcher."""
        self.valid = {}
        for n in self.names:
            k = len(self.hash[n]["1.0"])
            t = np.arange(k) / HFPS
            ok = np.ones(k, bool)
            h = HOLDOUT.get(n)
            if h:
                ok &= ~((t >= h[0]) & (t < h[1]))
            self.valid[n] = ok

    @classmethod
    def open(cls, verbose=True):
        """Load the cached index (build it first time: hashes ~1 min, 5 s cuts ~3 min, Cosmos-Embed1 ~8 min)."""
        a = cls()
        if not ((WORK / "index_vec.npy").exists() and (WORK / "index_seg.json").exists()):
            return a.build(verbose)
        with ThreadPoolExecutor(10) as ex:                       # fills in any variant missing from the hash cache
            for n, hv in ex.map(_hash_one, a.names):
                a.hash[n] = hv
        for n in a.names:
            a.dur[n] = pk.info(n)["duration"]
        a.vec = np.load(WORK / "index_vec.npy")
        a.seg = [tuple(x) for x in json.loads((WORK / "index_seg.json").read_text())]
        a._finish()
        return a


# ------------------------------------------------------------------------------------------------- stage 1
def stage1(arc: Archive, qvec: np.ndarray, qwins: list[tuple[float, float]], topk=10):
    """Per query window kNN, then aggregate to top-k (video, seg) candidates by summed similarity."""
    q = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-9)
    sim = q @ arc.vec.T                              # (W, S)
    agg = np.zeros(sim.shape[1])
    for w in range(sim.shape[0]):
        top = np.argsort(-sim[w])[:topk]
        agg[top] += sim[w, top]
    order = np.argsort(-agg)[:topk]
    return [{"video": arc.seg[i][0], "start": arc.seg[i][1], "end": arc.seg[i][2], "score": float(agg[i]),
             "best_sim": float(sim[:, i].max())} for i in order if agg[i] > 0], sim

