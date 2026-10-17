"""leaklib: redactors (test-export makers) and the LEAK TEST auditor.

REDACTOR side (makes the exports under test; uses the DataEngine-style detector, YOLO26s):
  a  keyframe : blur person boxes only on every 3rd frame of the working grid
  b  conf     : blur only persons with conf >= 0.60 (misses small / partial people)
  c  good     : all persons (cached tracks + a second low-confidence YOLO26s pass), every frame,
                +-1 frame hold, boxes dilated by 30 %

AUDITOR side (motivated intruder; sees ONLY the export + the original for the sharpness reference;
different models from the redactor: YOLO26l person detector + YOLO26s-pose face landmarks run on the full
frame AND on up-scaled head crops of every YOLO26l person):
  face evidence  = nose+eye or both eyes visible (landmark visibility >= 0.5)
  identifiable   = face width >= MIN_FACE_W px and enough texture in the ORIGINAL (Laplacian variance)
  UNREDACTED     = sharpness(export region) / sharpness(original region) >= SHARP_RATIO
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

MIN_FACE_W = 16        # px: policy, the auditor flags faces at least this wide (conservative lower bound)
GT_CLEAR_W = 20        # px: scoring only counts faces this wide as ground truth (margin vs the flag threshold)
MIN_LV_ORIG = 30.0     # Laplacian variance of the ORIGINAL face core: needs real detail to be a leak (raised from 12 after a blurred, low-contrast face was flagged on C and D)
SHARP_RATIO = 0.50     # export/original sharpness ratio at or above which the face counts as still sharp
KP_VIS = 0.5
DET_L_CONF = 0.15
CHUNK = 64             # frames per GPU batch (keeps every locked block short)

# ------------------------------------------------------------------------------------------------ redactors

REDACTORS = {
    "a_keyframe": {"label": "A  keyframe-only", "blurb": "blurs person boxes on every 3rd frame only"},
    "b_conf": {"label": "B  confidence-gated", "blurb": "blurs only persons with detector conf >= 0.60"},
    "c_good": {"label": "C  full redaction", "blurb": "all persons, every frame, dilated boxes"},
    "d_tight": {"label": "D  tight redaction", "blurb": "all persons, every frame, UNdilated boxes"},
}


def dilate(b, frac, w, h, top_extra=0.0):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - frac * bw)), max(0, int(y1 - (frac + top_extra) * bh)),
            min(w, int(x2 + frac * bw)), min(h, int(y2 + frac * bh)))


def blur_rects(img: np.ndarray, rects) -> np.ndarray:
    out = img.copy()
    for x1, y1, x2, y2 in rects:
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        k = int(max(9, min(x2 - x1, y2 - y1) / 2.5)) | 1
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out


def redaction_boxes(kind: str, tr_rows_by_f: dict, fi: int, k: int, extra_by_k: dict, w: int, h: int):
    """Rects the redactor `kind` will blur for grid frame k (source frame fi). Returns list of int rects."""
    rows = tr_rows_by_f.get(fi, [])
    if kind == "a_keyframe":
        if k % 3 != 0:
            return []
        return [dilate(r[:4], 0.10, w, h) for r in rows]
    if kind == "b_conf":
        return [dilate(r[:4], 0.10, w, h) for r in rows if r[4] >= 0.60]
    if kind == "d_tight":  # every person (low-conf pass), every frame, the detector box as is: no dilation, no hold
        return [dilate(r, 0.0, w, h) for r in extra_by_k.get(k, [])]
    out = []
    for kk in (k - 1, k, k + 1):  # c_good: current frame plus a one-frame hold either side
        out += [dilate(r, 0.30, w, h, 0.10) for r in extra_by_k.get(kk, [])]
    return out


# ------------------------------------------------------------------------------------------ sharpness / faces


def lap_var(img: np.ndarray, rect) -> float:
    """Variance of the Laplacian of the region resized to a fixed 40x48 grid (scale-robust)."""
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    H, W = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if g.shape[1] > 40 else cv2.INTER_CUBIC
    g = cv2.resize(g, (40, 48), interpolation=interp)
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())


METRIC = "core_v3"     # sharpness is measured on the eyes+nose CORE of the face rect (see core_rect); bump to invalidate caches
CORE_W, CORE_H = 0.60, 0.50
SHARP_FRAC = 0.50      # at least this share of the textured core pixels must be as sharp as in the original


def core_rect(rect):
    """Eyes+nose core of a face rectangle (face_from_kp centres the rect between eyes and nose). Sharpness is measured
    here, not on the whole rectangle: with a tight blur box a blurred face can still have sharp background (shelves,
    hair, a neighbour) inside the landmark rectangle, which made the first version flag blurred faces (found with D)."""
    x1, y1, x2, y2 = rect
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return [cx - CORE_W * w / 2, cy - CORE_H * h / 2, cx + CORE_W * w / 2, cy + CORE_H * h / 2]


def sharp_frac(exp_img: np.ndarray, orig_img: np.ndarray, rect) -> float:
    """Share of the textured pixels of the face core (texture measured in the ORIGINAL) that are still at least half as
    sharp in the export. A face edge poking out of a blur box gives a small share; an untouched face gives ~1."""
    x1, y1, x2, y2 = (int(round(v)) for v in rect)
    H, W = exp_img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    en = []
    for im in (exp_img, orig_img):
        g = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA if g.shape[1] > 32 else cv2.INTER_CUBIC).astype(np.float32)
        en.append(cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), -1, (3, 3)))
    ee, eo = en
    tex = eo >= 4.0
    if tex.sum() < 24:
        return 0.0
    return float(((ee >= 0.5 * eo) & tex).sum() / tex.sum())


def face_from_kp(kp) -> dict | None:
    """kp: 17x3. Returns {rect, w, evid} for a face with real landmark evidence, else None."""
    nose, le, re_, lear, rear = kp[0], kp[1], kp[2], kp[3], kp[4]
    v = lambda p: p[2] >= KP_VIS  # noqa: E731
    both_eyes = v(le) and v(re_)
    one_eye = v(le) or v(re_)
    if not (both_eyes or (v(nose) and one_eye)):
        return None
    ests = []
    if both_eyes:
        ests.append(2.4 * float(np.hypot(le[0] - re_[0], le[1] - re_[1])))
    if v(lear) and v(rear):
        ests.append(1.1 * float(np.hypot(lear[0] - rear[0], lear[1] - rear[1])))
    if v(nose) and one_eye and not both_eyes:
        e = le if v(le) else re_
        ests.append(3.4 * float(np.hypot(nose[0] - e[0], nose[1] - e[1])))
    W = float(np.median(ests))
    H = 1.35 * W
    pts = [p for p in (nose, le, re_) if v(p)]
    cx = float(np.mean([p[0] for p in pts]))
    eye_y = float(np.mean([p[1] for p in (le, re_) if v(p)])) if one_eye else float(nose[1]) - 0.2 * H
    cy = eye_y + 0.12 * H
    return {"rect": [cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2], "w": W,
            "evid": "eyes+nose" if (both_eyes and v(nose)) else "eyes" if both_eyes else "nose+eye"}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def head_crop(box, W, H):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1, cx2 = max(0, x1 - 0.15 * bw), min(W, x2 + 0.15 * bw)
    cy1, cy2 = max(0, y1 - 0.10 * bh), min(H, y1 + 0.52 * bh)
    return int(cx1), int(cy1), int(cx2), int(cy2)


class Gpu:
    """ONE locked session that can span many chunks (and several streams): 14 other spikes share this GPU and every
    protokit detect_many/pose_many call re-queues on gpu.lock (measured here: 760 s wait + 152 s hold for a single
    120-frame chunk), so the auditor takes the lock once per video, batches inference and refreshes the lock file's
    mtime so peers never treat it as stale."""

    wait_s = 0.0   # cumulative seconds spent waiting for the shared GPU lock
    held_s = 0.0   # cumulative seconds the lock was held (model load + inference)

    def __enter__(self):
        t = time.time()
        self._cm = pk.gpu_lock()
        self._cm.__enter__()
        self._t = time.time()
        Gpu.wait_s += self._t - t
        return self

    def __exit__(self, *a):
        pk._unload()
        r = self._cm.__exit__(*a)
        Gpu.held_s += time.time() - self._t
        return r

    def touch(self):
        try:
            import os
            os.utime(pk.CACHE / "gpu.lock", None)
        except OSError:
            pass

    def predict(self, kind, frames, conf, pose=False, batch=16):
        m = pk._yolo(kind)
        out = []
        engine = kind == "det"          # TensorRT engine: static batch 1, precision baked in
        B = 1 if engine else batch
        kw = {} if engine else {"half": True}
        for i in range(0, len(frames), B):
            for res in m.predict(frames[i:i + B], conf=conf, imgsz=640, verbose=False, batch=B, **kw):
                if pose:
                    kps = res.keypoints.data.tolist() if res.keypoints is not None else []
                    out.append([{"conf": cf, "box": xy, "kp": kp} for cf, xy, kp in
                                zip(res.boxes.conf.tolist(), res.boxes.xyxy.tolist(), kps)])
                else:
                    out.append([{"name": m.names[c], "conf": cf, "box": xy} for c, cf, xy in
                                zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())])
        self.touch()
        return out


def find_faces(g: Gpu, frames: list[np.ndarray]) -> list[dict]:
    """One auditor pass over a batch of frames -> per frame {'persons': [...], 'faces': [...]} (no sharpness yet)."""
    H, W = frames[0].shape[:2]
    dets = g.predict("det_l", frames, DET_L_CONF)
    persons = [[{"box": d["box"], "conf": d["conf"]} for d in fr if d["name"] == "person"
                and d["box"][3] - d["box"][1] >= 36] for fr in dets]
    full = g.predict("pose", frames, 0.25, pose=True)
    crops, meta = [], []
    for i, ps in enumerate(persons):
        for p in ps:
            cx1, cy1, cx2, cy2 = head_crop(p["box"], W, H)
            if cx2 - cx1 < 12 or cy2 - cy1 < 12:
                continue
            c = frames[i][cy1:cy2, cx1:cx2]
            s = float(np.clip(640 / max(c.shape[:2]), 1.0, 8.0))
            if s != 1.0:
                c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            crops.append(c)
            meta.append((i, cx1, cy1, s))
    cposes = g.predict("pose", crops, 0.25, pose=True, batch=32) if crops else []
    cand: list[list[dict]] = [[] for _ in frames]
    for i, fr in enumerate(full):
        for p in fr:
            fc = face_from_kp(p["kp"])
            if fc:
                cand[i].append({**fc, "pconf": p["conf"], "src": "frame"})
    for (i, ox, oy, s), poses in zip(meta, cposes):
        for p in poses:
            kp = [[ox + a / s, oy + b / s, c] for a, b, c in p["kp"]]
            fc = face_from_kp(kp)
            if fc:
                cand[i].append({**fc, "pconf": p["conf"], "src": "crop"})
    out = []
    for i in range(len(frames)):
        keep: list[dict] = []
        for c in sorted(cand[i], key=lambda c: -c["pconf"]):
            if c["w"] < 8 or any(iou(c["rect"], k["rect"]) > 0.3 for k in keep):
                continue
            keep.append(c)
        out.append({"persons": persons[i], "faces": keep})
    return out


def audit_stream(g: Gpu, frames_iter, orig_iter=None, ckpt: Path | None = None):
    """frames_iter yields (k, fi, t, frame) of the thing under audit; orig_iter the aligned originals (None when
    auditing the original itself). Returns list of per-frame dicts with faces + sharpness numbers.
    ckpt: optional json path; progress is saved after every chunk so a killed run resumes (shared machine)."""
    import json
    res: list[dict] = []
    skip = 0
    if ckpt is not None and Path(ckpt).exists():
        try:
            res = json.loads(Path(ckpt).read_text())
            skip = len(res)
        except Exception:
            res, skip = [], 0
    buf, obuf = [], []

    def flush():
        if not buf:
            return
        found = find_faces(g, [b[3] for b in buf])
        for (k, fi, t, fr), fo, o in zip(buf, found, obuf):
            faces = []
            for c in fo["faces"]:
                ref = o if o is not None else fr
                cr = core_rect(c["rect"])
                lv_e = lap_var(fr, cr)
                lv_o = lv_e if o is None else lap_var(ref, cr)
                fr_sharp = 1.0 if o is None else sharp_frac(fr, ref, cr)
                faces.append({"rect": [round(v, 1) for v in c["rect"]], "w": round(c["w"], 1), "evid": c["evid"],
                              "pconf": round(c["pconf"], 2), "src": c["src"], "lv_e": round(lv_e, 1),
                              "lv_o": round(lv_o, 1), "ratio": round(lv_e / (lv_o + 1e-6), 3),
                              "frac": round(fr_sharp, 3)})
            res.append({"k": k, "f": fi, "t": round(t, 3), "n_persons": len(fo["persons"]),
                        "persons": [[round(v, 1) for v in p["box"]] for p in fo["persons"]], "faces": faces})
        buf.clear()
        obuf.clear()
        if ckpt is not None:
            Path(ckpt).write_text(json.dumps(res))

    for idx, item in enumerate(frames_iter):
        o = next(orig_iter)[3] if orig_iter is not None else None
        if idx < skip:
            continue
        buf.append(item)
        obuf.append(o)
        if len(buf) >= CHUNK:
            flush()
    flush()
    return res


def tag_in_person(res: list[dict], ref: list[dict], grow: float = 0.10) -> None:
    """A face must belong to a person: its centre lies inside a slightly grown YOLO26l person box of the SAME audited
    frame of the ORIGINAL audit `ref` (so a blurred export cannot hide the person). Removes pose-model 'faces' on product art and posters (found on redactor D, retail shelf)."""
    for r, o in zip(res, ref):
        for f in r["faces"]:
            cx, cy = (f["rect"][0] + f["rect"][2]) / 2, (f["rect"][1] + f["rect"][3]) / 2
            f["in_person"] = any(b[0] - grow * (b[2] - b[0]) <= cx <= b[2] + grow * (b[2] - b[0]) and
                                 b[1] - grow * (b[3] - b[1]) <= cy <= b[3] + grow * (b[3] - b[1]) for b in o["persons"])


def is_identifiable(face: dict, min_w: float | None = None) -> bool:
    return (face["w"] >= (MIN_FACE_W if min_w is None else min_w) and face["lv_o"] >= MIN_LV_ORIG
            and face.get("in_person", True))


def is_flagged(face: dict) -> bool:
    return is_identifiable(face) and face["ratio"] >= SHARP_RATIO and face.get("frac", 1.0) >= SHARP_FRAC


# ------------------------------------------------------------------------------------------------ coverage


def coverage(rect, rects, n: int = 8) -> float:
    """Fraction of the face rect inside the union of the redactor's blur rects (grid sampling)."""
    if not rects:
        return 0.0
    x1, y1, x2, y2 = rect
    xs = np.linspace(x1, x2, n + 2)[1:-1]
    ys = np.linspace(y1, y2, n + 2)[1:-1]
    gx, gy = np.meshgrid(xs, ys)
    inside = np.zeros(gx.shape, bool)
    for a, b, c, d in rects:
        inside |= (gx >= a) & (gx <= c) & (gy >= b) & (gy <= d)
    return float(inside.mean())


# ---------------------------------------------------------------------------------------- person-level ledger


def link_faces(frames: list[dict], fps: float, only_flagged: bool = True):
    """Simple centroid linker over auditor faces -> persons P1.. with leaked frames / seconds."""
    people: list[dict] = []
    for fr in frames:
        for fc in fr["faces"]:
            if only_flagged and not is_flagged(fc):
                continue
            cx, cy = (fc["rect"][0] + fc["rect"][2]) / 2, (fc["rect"][1] + fc["rect"][3]) / 2
            best, bd = None, 1e9
            for p in people:
                if fr["k"] - p["last_k"] > 6:
                    continue
                d = np.hypot(cx - p["cx"], cy - p["cy"])
                if d < max(1.2 * fc["w"], 30) * (1 + 0.15 * (fr["k"] - p["last_k"])) and d < bd:
                    best, bd = p, d
            if best is None:
                best = {"id": len(people) + 1, "frames": [], "cx": cx, "cy": cy, "last_k": fr["k"]}
                people.append(best)
            best["cx"], best["cy"], best["last_k"] = cx, cy, fr["k"]
            best["frames"].append((fr["k"], fr["t"], fc))
    out = []
    for p in people:
        ts = [f[1] for f in p["frames"]]
        best = max(p["frames"], key=lambda f: f[2]["w"] * min(1.0, f[2]["ratio"]))
        out.append({"person": f"P{p['id']}", "frames": len(p["frames"]), "seconds": round(len(p["frames"]) / fps, 2),
                    "first_t": min(ts), "last_t": max(ts), "best_k": best[0], "best_t": best[1],
                    "best_rect": best[2]["rect"], "ks": [f[0] for f in p["frames"]],
                    "rects": [f[2]["rect"] for f in p["frames"]]})
    return out


def merge_people(people: list[dict], fps: float, max_gap_s: float = 3.0) -> list[list[dict]]:
    """Re-link fragmented face tracks into an ESTIMATE of distinct people: a track continues an earlier one if it starts
    after that one ended, within `max_gap_s`, near where it ended (allowing ~60 px/s of walking). Greedy, position-only:
    it can still split one person (long gap) or fuse two (crossing paths). Returns the list of chains (lists of tracks)."""
    tr = sorted(people, key=lambda p: p["ks"][0])
    chains: list[list[dict]] = []
    for p in tr:
        k0 = p["ks"][0]
        r0 = p["rects"][0]
        c0 = ((r0[0] + r0[2]) / 2, (r0[1] + r0[3]) / 2)
        w0 = r0[2] - r0[0]
        best, bd = None, 1e9
        for ch in chains:
            last = ch[-1]
            gap = (k0 - last["ks"][-1]) / fps
            if gap <= 0 or gap > max_gap_s:
                continue
            rl = last["rects"][-1]
            cl = ((rl[0] + rl[2]) / 2, (rl[1] + rl[3]) / 2)
            d = float(np.hypot(c0[0] - cl[0], c0[1] - cl[1]))
            lim = 1.5 * max(w0, rl[2] - rl[0]) + 60.0 * gap
            if d <= lim and d < bd:
                best, bd = ch, d
        if best is None:
            chains.append([p])
        else:
            best.append(p)
    return chains


def summarize_people(chains: list[list[dict]], fps: float) -> list[dict]:
    """One row per estimated person (chain of face tracks): leaked frames / seconds, first-last time, best crop."""
    out = []
    for i, ch in enumerate(chains, 1):
        ks = sorted({k for p in ch for k in p["ks"]})
        cand = [p for p in ch if p["frames"] >= 3] or ch
        b = max(cand, key=lambda p: (p["best_rect"][2] - p["best_rect"][0]) * min(1.0, p["frames"] / 6))
        out.append({"person": f"P{i}", "tracks": len(ch), "frames": len(ks), "seconds": round(len(ks) / fps, 2),
                    "first_t": min(p["first_t"] for p in ch), "last_t": max(p["last_t"] for p in ch),
                    "best_k": b["best_k"], "best_t": b["best_t"], "best_rect": b["best_rect"], "ks": ks,
                    "rects": b["rects"], "rect_ks": b["ks"]})
    return out


def refresh_sharpness(res: list[dict], frames_iter, orig_iter=None) -> list[dict]:
    """CPU-only: recompute lv_e / lv_o / ratio of every audited face with the current sharpness metric from the frames
    (face rectangles and landmarks come from the cached GPU pass). frames_iter yields (k, fi, t, frame)."""
    for r, item in zip(res, frames_iter):
        fr = item[3]
        o = next(orig_iter)[3] if orig_iter is not None else None
        for f in r["faces"]:
            cr = core_rect(f["rect"])
            lv_e = lap_var(fr, cr)
            lv_o = lv_e if o is None else lap_var(o, cr)
            f["lv_e"], f["lv_o"], f["ratio"] = round(lv_e, 1), round(lv_o, 1), round(lv_e / (lv_o + 1e-6), 3)
            f["frac"] = 1.0 if o is None else round(sharp_frac(fr, o, cr), 3)
    return res
