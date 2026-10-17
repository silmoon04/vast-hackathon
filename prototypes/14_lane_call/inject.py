"""SIMULATED stops injected into the real motorway_a40 clip (the headline use case has no real stop in any footage we have).

What is real: the A40 footage, the YOLO detector run on the injected pixels (after H.264 re-encode), the S = 5 s stopped-vehicle
rule, the registration, the lane geometry, the call, the card.
What is simulated: three vehicles that are pasted (static) into the scene from t = 0 to the end of the clip. Each pasted
vehicle is a real car cut out of the same clip (YOLO26s-seg silhouette, feathered edge, contact shadow, scaled to the lane
perspective), so nothing is drawn by hand except WHERE it is pasted and how large. Real vehicles that are nearer the camera than
a pasted one are pasted back over it (original pixels, inside the silhouette) so traffic drives in front of the stopped car.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import protokit as pk  # noqa: E402
import lanecall as L  # noqa: E402

BASE = "motorway_a40"
ASSETS = HERE / "inject_assets"

# src_t / src_box: the real car that is cut out (seconds, pixel box on the 1280x720 frame, found on a viewed grid frame)
# at: (centre x, ground y, width px) where it is pasted; expect: truth by construction (checked by eye on the hero/eyeball sheet)
SPEC = [
    {"id": "inj_shoulder", "desc": "car stopped on the RIGHT hard shoulder (facing traffic)", "src_t": 8.0,
     "src_box": [588, 608, 680, 684], "at": (948, 652, 62), "expect": "AT KERB"},
    {"id": "inj_lane_left", "desc": "car stopped in the LEFT carriageway running lane (rear view)", "src_t": 8.0,
     "src_box": [364, 578, 456, 644], "at": (345, 612, 90), "expect": "IN LANE"},
    {"id": "inj_lane_right", "desc": "car stopped in the RIGHT carriageway running lane (front view)", "src_t": 8.0,
     "src_box": [740, 636, 820, 706], "at": (810, 602, 68), "expect": "IN LANE"},
]


def _sig() -> str:
    return hashlib.sha1(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:10]


def extract_assets() -> dict[str, dict]:
    """Cut each source car out with its YOLO26s-seg silhouette; cached as RGBA PNG (no GPU after the first run)."""
    ASSETS.mkdir(exist_ok=True)
    need = [s for s in SPEC if not (ASSETS / f"{s['id']}.png").exists()]
    if need:
        times = sorted({s["src_t"] for s in need})
        frs = {t: pk.frame_at(BASE, t) for t in times}
        dets = dict(zip(times, pk.seg_masks_many([frs[t] for t in times], classes=["car", "truck", "bus"])))
        for s in need:
            fr, ds = frs[s["src_t"]], dets[s["src_t"]]
            b = np.array([s["src_box"]], float)
            best, bi = None, 0.3
            for d in ds:
                i = L._iou(b, np.array([d["box"]], float))
                if i > bi:
                    best, bi = d, i
            if best is None:
                raise RuntimeError(f"{s['id']}: no seg detection near {s['src_box']}")
            m = best["mask"].astype(np.uint8) * 255
            x1, y1, x2, y2 = (int(v) for v in best["box"])
            pad = 3
            x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(fr.shape[1], x2 + pad), min(fr.shape[0], y2 + pad)
            rgba = np.dstack([fr[y1:y2, x1:x2], m[y1:y2, x1:x2]])
            cv2.imwrite(str(ASSETS / f"{s['id']}.png"), rgba)
            print(f"[inject] cut out {s['id']}: seg box iou {bi:.2f}, {rgba.shape[1]}x{rgba.shape[0]} px")
    return {s["id"]: {"rgba": cv2.imread(str(ASSETS / f"{s['id']}.png"), cv2.IMREAD_UNCHANGED)} for s in SPEC}


def prepare_sprites(assets: dict[str, dict]) -> list[dict]:
    """Scale each cut-out to its target width, build the feathered alpha and the contact shadow once."""
    out = []
    for s in SPEC:
        rgba = assets[s["id"]]["rgba"]
        cx, gy, wpx = s["at"]
        sc = wpx / rgba.shape[1]
        sp = cv2.resize(rgba, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        a = cv2.GaussianBlur(sp[..., 3].astype(np.float32) / 255.0, (0, 0), 0.8)
        a = np.clip((a - 0.15) / 0.85, 0, 1)
        h, w = sp.shape[:2]
        x0, y0 = int(round(cx - w / 2)), int(round(gy - h))
        # contact shadow: dark soft ellipse under the car's lower edge
        sh = np.zeros((h + 40, w + 40), np.float32)
        cv2.ellipse(sh, (w // 2 + 20, h + 6), (int(w * 0.5), max(3, int(h * 0.09))), 0, 0, 360, 0.55, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), max(2.0, h * 0.04))
        out.append({"spec": s, "bgr": sp[..., :3].astype(np.float32), "alpha": a, "x0": x0, "y0": y0, "w": w, "h": h, "shadow": sh,
                    "box": [x0, y0, x0 + w, y0 + h]})
    return out


def paste(frame: np.ndarray, spr: dict, real_boxes: list[list[float]], real_frame: np.ndarray) -> np.ndarray:
    H, W = frame.shape[:2]
    x0, y0, w, h = spr["x0"], spr["y0"], spr["w"], spr["h"]
    # shadow
    sx0, sy0 = x0 - 20, y0
    sh = spr["shadow"]
    ya, yb, xa, xb = max(0, sy0), min(H, sy0 + sh.shape[0]), max(0, sx0), min(W, sx0 + sh.shape[1])
    if yb > ya and xb > xa:
        s_ = sh[ya - sy0:yb - sy0, xa - sx0:xb - sx0, None]
        frame[ya:yb, xa:xb] = (frame[ya:yb, xa:xb].astype(np.float32) * (1 - s_)).astype(np.uint8)
    ya, yb, xa, xb = max(0, y0), min(H, y0 + h), max(0, x0), min(W, x0 + w)
    if yb <= ya or xb <= xa:
        return frame
    a = spr["alpha"][ya - y0:yb - y0, xa - x0:xb - x0].copy()
    # nearer real vehicles stay in front: bring back the original pixels inside their boxes (their ground edge is lower)
    gy = y0 + h
    for b in real_boxes:
        if b[3] > gy + 3:
            bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            ix1, iy1, ix2, iy2 = max(bx1, xa), max(by1, ya), min(bx2, xb), min(by2, yb)
            if ix2 > ix1 and iy2 > iy1:
                a[iy1 - ya:iy2 - ya, ix1 - xa:ix2 - xa] = 0.0
    src = spr["bgr"][ya - y0:yb - y0, xa - x0:xb - x0]
    frame[ya:yb, xa:xb] = (frame[ya:yb, xa:xb].astype(np.float32) * (1 - a[..., None]) + src * a[..., None]).astype(np.uint8)
    return frame


def build(out: Path) -> dict:
    """Render the injected clip (H.264), run the SAME detector on its frames (stride as pk.tracks), return a Tracks object."""
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / "injected.mp4"
    cache = out / "_inject_dets.json.gz"
    sig = _sig()
    meta = pk.info(BASE)
    base_tr = pk.tracks(BASE)
    stride = base_tr.stride
    if not mp4.exists() or not cache.exists() or json.loads(gzip.decompress(cache.read_bytes()))["sig"] != sig:
        assets = extract_assets()
        sprites = prepare_sprites(assets)
        real = base_tr.rows[base_tr.rows["name"].isin(["car", "truck", "bus", "motorcycle"])]
        with pk.Mp4Writer(mp4, meta["fps"]) as w:
            for i, t, fr in pk.iter_frames(BASE, stride=1):
                near = real[real["t"] == real["t"].iloc[(real["t"] - t).abs().argmin()]] if len(real) else real
                rb = [[r.x1, r.y1, r.x2, r.y2] for r in near.itertuples()]
                f = fr.copy()
                for spr in sprites:
                    f = paste(f, spr, rb, fr)
                w.write(f)
        # detect on the RE-ENCODED frames, so compression noise is part of the test
        frames, meta_rows = [], []
        for i, t, fr in pk.iter_frames(mp4, stride=stride):
            frames.append(fr)
            meta_rows.append((i, t))
        assert len(frames) <= 600, len(frames)
        dets = pk.detect_many(frames, model="det", conf=0.25)
        rows = []
        for (i, t), ds in zip(meta_rows, dets):
            for d in ds:
                rows.append([i, round(t, 3), -1, 0, d["name"], round(d["conf"], 3), *[round(v, 1) for v in d["box"]]])
        cache.write_bytes(gzip.compress(json.dumps({"sig": sig, "rows": rows, "boxes": [s["box"] for s in sprites]}).encode()))
        for stale in (out / "_seg_cache.json",):
            if stale.exists():
                stale.unlink()
    d = json.loads(gzip.decompress(cache.read_bytes()))
    df = pd.DataFrame(d["rows"], columns=pk.COLS)
    df = pk._with_geom(df)
    inf = pk.info(mp4)
    tr = pk.Tracks("motorway_a40_inject", meta["fps"], stride, inf["w"], inf["h"], inf["duration"], {}, df,
                   "yolo26s (re-detected on injected frames, no tracker)", {"injected": [s["id"] for s in SPEC]})
    return {"tracks": tr, "mp4": mp4, "boxes": d["boxes"], "spec": SPEC}
