"""Shared data layer: one camera-day of tracks, frames, background plates and object summaries.

Dev mode reads local stand-in clips and cached YOLO+ByteTrack tracks. On event day the same
functions get a Pack D adapter (VastDB detection sidecars + S3 segments) behind CAMS.
Every module imports only from here, so swapping the source never touches feature code.
"""
from __future__ import annotations

import gzip
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("STM_OUT", str(ROOT / "out"))).expanduser()
TRACK_CACHE = Path(os.environ.get("STM_TRACK_CACHE", str(ROOT / "_cache" / "tracks"))).expanduser()
VIDEO_ROOT = Path(os.environ.get("STM_VIDEO_ROOT", str(ROOT / "videos"))).expanduser()
ONLINE = VIDEO_ROOT / "online"
INTEL = VIDEO_ROOT / "intel"

# day spans are seconds into the video. A split of one clip into two "days" is SIMULATED and labelled so.
CAMS: dict[str, dict[str, Any]] = {
    "ptz_parking": {"video": ONLINE / "ptz_parking.mp4", "title": "Car park, fixed camera",
                    "days": [("A", "Day A", 0.0, 158.0), ("B", "Day B", 158.0, None)],
                    "simulated": ["Day A / Day B are two halves of one 5-minute clip"],
                    "credit": "PTZ parking camera, Wikimedia Commons, public domain"},
    "car_detection": {"video": INTEL / "car-detection.mp4", "title": "Car park entrance",
                      "days": [("A", "Clip", 0.0, None)], "simulated": [],
                      "credit": "Intel IoT sample videos, CC BY 4.0"},
    "person_bicycle_car_detection": {"video": INTEL / "person-bicycle-car-detection.mp4", "title": "Street with people, bikes, cars",
                                     "days": [("A", "Clip", 0.0, None)], "simulated": [],
                                     "credit": "Intel IoT sample videos, CC BY 4.0"},
    "rochester_night": {"video": ONLINE / "rochester_night.mp4", "title": "Night street of flats",
                        "days": [("A", "Night", 0.0, None)], "simulated": [],
                        "credit": "Wikimedia Commons, CC BY-SA 4.0"},
    "yonge_dundas": {"video": ONLINE / "yonge_dundas.mp4", "title": "City crossing",
                     "days": [("A", "Clip", 0.0, None)], "simulated": [],
                     "credit": "Wikimedia Commons, CC0"},
}
# Extra cams (e.g. Pack D built on the lab VM) live in cams.local.json so every module process sees them.
# A cam whose video is not on this machine (built on the VM, outputs synced here) goes to REMOTE_CAMS instead, so a
# module run here can never overwrite its synced outputs with an empty result.
_LOCAL = Path(os.environ.get("STM_CAMERAS", str(ROOT / "cams.local.json"))).expanduser()
REMOTE_CAMS: dict[str, dict[str, Any]] = {}
if _LOCAL.exists():
    for _k, _v in json.loads(_LOCAL.read_text(encoding="utf-8")).items():
        _video = Path(_v["video"]).expanduser()
        if not _video.is_absolute():
            _video = _LOCAL.parent / _video
        _e = {**_v, "video": _video, "days": [tuple(d) for d in _v["days"]]}
        (CAMS if _e["video"].exists() else REMOTE_CAMS)[_k] = _e
COLS = ["f", "t", "tid", "cls", "name", "conf", "x1", "y1", "x2", "y2"]
VEHICLES = ("car", "truck", "bus", "motorcycle")
MOVERS = ("person", "bicycle", "car", "truck", "bus", "motorcycle")


def out_dir(cam: str) -> Path:
    d = OUT / cam
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=_np), encoding="utf-8")
    return p


def _np(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return round(float(o), 3)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


@lru_cache(maxsize=None)
def info(cam: str) -> dict[str, float]:
    cap = cv2.VideoCapture(str(CAMS[cam]["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "w": w, "h": h, "duration": n / fps}


def days(cam: str) -> list[dict[str, Any]]:
    dur = info(cam)["duration"]
    return [{"id": i, "label": lab, "start": a, "end": dur if b is None else b} for i, lab, a, b in CAMS[cam]["days"]]


def day_of(cam: str, t: float) -> str:
    for d in days(cam):
        if d["start"] <= t < d["end"] or t == d["end"]:
            return d["id"]
    return days(cam)[-1]["id"]


@lru_cache(maxsize=None)
def tracks(cam: str) -> pd.DataFrame:
    """All detections with track ids: f, t, tid, cls, name, conf, x1..y2, cx, cy, bw, bh, day. tid<0 = untracked."""
    hits = sorted(TRACK_CACHE.glob(f"{cam}__det__*.json.gz"))
    if not hits:
        raise FileNotFoundError(f"no cached tracks for {cam} in {TRACK_CACHE}")
    d = json.load(gzip.open(hits[0], "rt", encoding="utf-8"))
    df = pd.DataFrame(d["rows"], columns=COLS)
    df["cx"], df["cy"] = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2
    df["bw"], df["bh"] = df.x2 - df.x1, df.y2 - df.y1
    bounds = days(cam)
    df["day"] = bounds[-1]["id"]
    for b in reversed(bounds):
        df.loc[(df.t >= b["start"]) & (df.t < b["end"]), "day"] = b["id"]
    return df.sort_values(["t", "tid"]).reset_index(drop=True)


def objects(cam: str, min_frames: int = 8) -> pd.DataFrame:
    """One row per track: id, name, day, first_t, last_t, n, median box, travel_px, parked.

    parked = on screen 20 s or more and its centre never strays more than half a box width.
    """
    df = tracks(cam)
    df = df[(df.tid >= 0) & df.name.isin(MOVERS)]
    rows = []
    for tid, g in df.groupby("tid"):
        if len(g) < min_frames:
            continue
        name = g.name.mode().iat[0]
        med = g[["x1", "y1", "x2", "y2"]].median()
        span = g.t.iat[-1] - g.t.iat[0]
        dev = np.hypot(g.cx - g.cx.median(), g.cy - g.cy.median()).quantile(0.9)
        travel = float(np.hypot(np.diff(g.cx), np.diff(g.cy)).sum())
        rows.append({"id": int(tid), "name": name, "day": g.day.mode().iat[0], "first_t": float(g.t.iat[0]),
                     "last_t": float(g.t.iat[-1]), "n": len(g), "conf": float(g.conf.median()),
                     "x1": float(med.x1), "y1": float(med.y1), "x2": float(med.x2), "y2": float(med.y2),
                     "travel_px": travel,
                     "parked": bool(name in VEHICLES and span >= 20 and dev <= 0.5 * (med.x2 - med.x1))})
    return pd.DataFrame(rows)


def path(cam: str, tid: int, step: float = 0.5) -> list[list[float]]:
    """[[t, cx, cy], ...] for one track, thinned to one point per `step` seconds."""
    g = tracks(cam)
    g = g[g.tid == tid]
    if g.empty:
        return []
    g = g.assign(b=(g.t // step)).groupby("b").first()
    return [[round(r.t, 2), round(r.cx, 1), round(r.cy, 1)] for r in g.itertuples()]


def frame_at(cam: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(CAMS[cam]["video"]))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"no frame at {t:.2f}s in {cam}")
    return img


def background(cam: str, day: str | None = None, n: int = 48) -> np.ndarray:
    """Median of n evenly spaced frames: the street with moving things removed. Cached as a jpg."""
    p = out_dir(cam) / f"background{'_' + day if day else ''}.jpg"
    if p.exists():
        return cv2.imread(str(p))
    span = next(d for d in days(cam) if d["id"] == day) if day else {"start": 0.0, "end": info(cam)["duration"]}
    ts = np.linspace(span["start"] + 0.2, span["end"] - 0.5, n)
    cap = cv2.VideoCapture(str(CAMS[cam]["video"]))
    frames = []
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(p), bg, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return bg


def crop(cam: str, tid: int, t: float | None = None, pad: float = 0.15) -> tuple[np.ndarray, float]:
    """Crop of a track at time t (default: the frame where its box is biggest). Returns (img, t_used)."""
    g = tracks(cam)
    g = g[g.tid == tid]
    if g.empty:
        raise KeyError(f"track {tid} not in {cam}")
    r = g.iloc[(g.t - t).abs().argmin()] if t is not None else g.loc[(g.bw * g.bh).idxmax()]
    img = frame_at(cam, float(r.t))
    H, W = img.shape[:2]
    px, py = r.bw * pad, r.bh * pad
    x1, y1 = int(max(0, r.x1 - px)), int(max(0, r.y1 - py))
    x2, y2 = int(min(W, r.x2 + px)), int(min(H, r.y2 + py))
    return img[y1:y2, x1:x2].copy(), float(r.t)


def save_crop(cam: str, tid: int, t: float | None = None, sub: str = "crops") -> str:
    """Write crops/<tid>.jpg (or <tid>_<t>.jpg when t is given); return the path relative to out/<cam>/."""
    img, tu = crop(cam, tid, t)
    name = f"{tid}.jpg" if t is None else f"{tid}_{tu:.1f}.jpg"
    p = out_dir(cam) / sub / name
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"{sub}/{name}"


def write_meta(cam: str) -> dict[str, Any]:
    i = info(cam)
    meta = {"camera": cam, "title": CAMS[cam]["title"], "credit": CAMS[cam]["credit"], "fps": i["fps"],
            "w": i["w"], "h": i["h"], "duration": round(i["duration"], 2), "days": days(cam),
            "simulated": CAMS[cam]["simulated"], "background": "background.jpg", "video": str(CAMS[cam]["video"])}
    background(cam)
    write_json(out_dir(cam) / "meta.json", meta)
    return meta


if __name__ == "__main__":
    import sys
    for cam in sys.argv[1:] or list(CAMS):
        m = write_meta(cam)
        o = objects(cam)
        print(cam, f"{m['duration']:.0f}s", len(tracks(cam)), "dets", len(o), "objects",
              int(o.parked.sum()) if len(o) else 0, "parked")
