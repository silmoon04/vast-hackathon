"""SIMULATED camera faults injected into real, healthy footage = ground truth for the state detector.

Everything produced here is labelled SIMULATED. The faulted clip is re-encoded to H.264 (as a recorder would) and then
analysed through exactly the same path as any other video (no side channel from the injector to the detector).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import blind as bl
import protokit as pk

# the spec'd fault set (+ one extra: occlusion). Starts are deliberately NOT on whole seconds.
BASE = "retail_cctv"
DEFAULT_PLAN = [
    {"state": "DARK", "t0": 14.3, "t1": 24.3, "what": "black frames, 10 s"},
    {"state": "FROZEN", "t0": 36.7, "t1": 44.7, "what": "last frame repeated, 8 s"},
    {"state": "BLUR", "t0": 58.2, "t1": 64.2, "what": "Gaussian blur sigma 10 px (at 1270 px), 6 s", "sigma": 10},
    {"state": "MOVED", "t0": 76.5, "t1": 86.5, "what": "view shifted 30% right + 8% down, rotated 4 deg, 10 s", "dx": 0.30, "dy": 0.08, "rot": 4.0},
    {"state": "OCCLUDED", "t0": 88.3, "t1": 95.3, "what": "dark textured blob over ~40% of the frame, 7 s (extra fault)"},
]


# HELD-OUT injections: same five fault types, NEW times / strengths / blob position, injected into two OTHER healthy clips.
# Thresholds in blind.CFG were frozen before these were looked at; they are scored once, as-is (no re-tuning).
HELDOUT: dict[str, list[dict[str, Any]]] = {
    "store_aisle_detection": [
        {"state": "DARK", "t0": 9.6, "t1": 15.6, "what": "black frames, 6 s"},
        {"state": "FROZEN", "t0": 21.4, "t1": 28.4, "what": "last frame repeated, 7 s"},
        {"state": "BLUR", "t0": 33.1, "t1": 38.1, "what": "Gaussian blur sigma 7 px, 5 s", "sigma": 7},
        {"state": "MOVED", "t0": 42.2, "t1": 48.2, "what": "view shifted 20% right + 5% down, rotated 2 deg, 6 s", "dx": 0.20, "dy": 0.05, "rot": 2.0},
        {"state": "OCCLUDED", "t0": 52.4, "t1": 57.4, "what": "dark textured blob over ~30% of the frame (other seed / position), 5 s",
         "area": 0.30, "seed": 7, "cx": 0.62, "cy": 0.45},
    ],
    "yonge_dundas": [
        {"state": "DARK", "t0": 9.4, "t1": 14.4, "what": "dimmed to 15% brightness, 5 s", "gain": 0.15},
        {"state": "FROZEN", "t0": 19.2, "t1": 26.2, "what": "last frame repeated, 7 s"},
        {"state": "BLUR", "t0": 31.6, "t1": 36.6, "what": "Gaussian blur sigma 7 px, 5 s", "sigma": 7},
        {"state": "MOVED", "t0": 40.3, "t1": 45.3, "what": "view shifted 22% right, 5% up, rotated -3 deg, 5 s", "dx": 0.22, "dy": -0.05, "rot": -3.0},
        {"state": "OCCLUDED", "t0": 49.1, "t1": 54.1, "what": "dark textured blob over ~30% of the frame (other seed / position), 5 s",
         "area": 0.30, "seed": 11, "cx": 0.30, "cy": 0.40},
    ],
}


def sweep_plan(start: float = 4.3, dur: float = 4.0, gap: float = 2.5) -> list[dict[str, Any]]:
    """Sensitivity clip: one fault per variant with different strength, so we learn where detection stops working."""
    variants: list[dict[str, Any]] = []
    for sg in (2, 4, 6, 10):
        variants.append({"state": "BLUR", "sigma": sg, "what": f"blur sigma {sg} px"})
    for sh in (0.05, 0.10, 0.20, 0.30):
        variants.append({"state": "MOVED", "dx": sh, "dy": 0.0, "rot": 0.0, "what": f"shift {int(sh * 100)}% of width"})
    for k in (0.5, 0.25, 0.10):
        variants.append({"state": "DARK", "gain": k, "what": f"dimmed to {int(k * 100)}% brightness"})
    for fr in (0.15, 0.25, 0.40):
        variants.append({"state": "OCCLUDED", "area": fr, "what": f"blob covering ~{int(fr * 100)}% of frame"})
    t = start
    for v in variants:
        v["t0"], v["t1"] = round(t, 2), round(t + dur, 2)
        t += dur + gap
    return variants


def _blob(shape: tuple[int, int], area: float, seed: int = 3, cx: float = 0.38, cy: float = 0.55) -> tuple[np.ndarray, np.ndarray]:
    """A near-object stand-in: dark brownish textured ellipse. Returns (bgr texture, mask)."""
    h, w = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), np.uint8)
    # ellipse axes so that pi*a*b = area*w*h with a/b ~ 0.9*w/h
    b = np.sqrt(area * w * h / (np.pi * 1.1 * w / h))
    a = b * 1.1 * w / h
    cv2.ellipse(mask, (int(w * cx), int(h * cy)), (int(a), int(b)), -12, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (38, 44, 58)
    noise = cv2.GaussianBlur(rng.normal(0, 22, (h, w)).astype(np.float32), (0, 0), 3)
    grad = np.linspace(-18, 18, w, dtype=np.float32)[None, :]
    tex = np.clip(base + noise[..., None] + grad[..., None], 0, 255).astype(np.uint8)
    return tex, mask


def apply_fault(frame: np.ndarray, f: dict[str, Any], held: np.ndarray | None, cache: dict[str, Any]) -> np.ndarray:
    h, w = frame.shape[:2]
    st = f["state"]
    if st == "DARK":
        if "gain" in f:
            return np.clip(frame.astype(np.float32) * f["gain"], 0, 255).astype(np.uint8)
        return np.zeros_like(frame)
    if st == "FROZEN":
        return held if held is not None else frame
    if st == "BLUR":
        return cv2.GaussianBlur(frame, (0, 0), f["sigma"])
    if st == "MOVED":
        ang = np.radians(f.get("rot", 0.0))
        c, s = np.cos(ang), np.sin(ang)
        cx, cy = w / 2, h / 2
        M = np.array([[c, -s, cx - c * cx + s * cy + f["dx"] * w], [s, c, cy - s * cx - c * cy + f["dy"] * h]], np.float64)
        return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if st == "OCCLUDED":
        area = f.get("area", 0.40)
        bp = (f.get("seed", 3), f.get("cx", 0.38), f.get("cy", 0.55))
        key = f"blob{area}{bp}"
        if key not in cache:
            cache[key] = _blob((h, w), area, *bp)
        tex, mask = cache[key]
        a = (mask.astype(np.float32) / 255.0)[..., None]
        return (frame * (1 - a) + tex * a).astype(np.uint8)
    return frame


def make_faulty(base: str, plan: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """Write the faulted copy of `base` to `out` (H.264) and return {fps, plan}. SIMULATED."""
    meta = pk.info(base)
    fps = meta["fps"]
    held, cache = None, {}
    prev = None
    with pk.Mp4Writer(out, fps) as wr:
        for i, t, fr in bl._frames_resumable(base, meta["duration"]):
            f = next((p for p in plan if p["t0"] <= t < p["t1"]), None)
            if f is None:
                held = None
                wr.write(fr)
            else:
                if f["state"] == "FROZEN" and held is None:
                    held = prev if prev is not None else fr
                wr.write(apply_fault(fr, f, held, cache))
            prev = fr
    return {"fps": fps, "plan": plan, "base": base, "file": str(out)}


def natural_stall_seconds(base: str, thresh: float = 0.7, tol: int = 12) -> list[int]:
    """Independent ground truth for stalls already present in the untouched clip: seconds in which >= 70% of consecutive
    NATIVE-resolution frame pairs differ by at most `tol` grey levels at every pixel (codec noise only). A different
    statistic from the detector's (which uses the median of 160 px mean differences)."""
    meta = pk.info(base)
    prev, cnt, tot = None, {}, {}
    for i, t, fr in bl._frames_resumable(base, meta["duration"]):
        s = int(t)
        if prev is not None:
            tot[s] = tot.get(s, 0) + 1
            if int(np.abs(fr.astype(np.int16) - prev.astype(np.int16)).max()) <= tol:
                cnt[s] = cnt.get(s, 0) + 1
        prev = fr
    return sorted(s for s in tot if cnt.get(s, 0) / tot[s] >= thresh)


def gt_labels(n: int, plan: list[dict[str, Any]], natural_frozen: list[int] | None = None) -> list[str]:
    """Per-second ground truth: the injected fault covering >= 50% of that second, else natural freezes, else OK."""
    out = []
    nat = set(natural_frozen or [])
    for s in range(n):
        lab = "OK"
        best = 0.0
        for p in plan:
            ov = max(0.0, min(s + 1, p["t1"]) - max(s, p["t0"]))
            if ov >= 0.5 and ov > best:
                lab, best = p["state"], ov
        if lab == "OK" and s in nat:
            lab = "FROZEN"
        out.append(lab)
    return out


def score(pred: list[str], gt: list[str], tol: int = 1, states: list[str] | None = None) -> dict[str, Any]:
    """Per-state precision / recall. 'strict' = exact per-second agreement; 'tolerant' = a second also counts as correct
    if the same state appears in the other labelling within +-tol seconds (boundary jitter)."""
    n = min(len(pred), len(gt))
    states = states or bl.STATES
    res: dict[str, Any] = {}
    for st in states:
        tp = sum(1 for i in range(n) if pred[i] == st and gt[i] == st)
        fp = sum(1 for i in range(n) if pred[i] == st and gt[i] != st)
        fn = sum(1 for i in range(n) if pred[i] != st and gt[i] == st)
        tp_t = sum(1 for i in range(n) if gt[i] == st and st in pred[max(0, i - tol):i + tol + 1])
        fp_t = sum(1 for i in range(n) if pred[i] == st and st not in gt[max(0, i - tol):i + tol + 1])
        res[st] = {"gt_s": tp + fn, "pred_s": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                   "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                   "recall_tol": tp_t / (tp + fn) if tp + fn else None,
                   "precision_tol": (tp + fp - fp_t) / (tp + fp) if tp + fp else None}
    return res


def confusion(pred: list[str], gt: list[str]) -> dict[str, dict[str, int]]:
    n = min(len(pred), len(gt))
    m: dict[str, dict[str, int]] = {g: {} for g in bl.STATES}
    for i in range(n):
        m[gt[i]][pred[i]] = m[gt[i]].get(pred[i], 0) + 1
    return m
