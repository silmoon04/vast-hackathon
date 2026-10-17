"""Pack D adapter: neighbourhood-camera detection sidecars (VastDB rows) -> a Street Time Machine camera.

Sidecar shape (vast-hackathon/docs/API_NOTES.md, GET /videos/detections), one per 5-10 s segment:
  {source, segment_source, video_shape:[h,w], fps, frame_count, detection_count, object_classes, object_counts,
   max_detection_conf, frames:[{frame_index, time_sec, shape:[h,w], detections:[{label, confidence, bbox:[x1,y1,x2,y2]}]}]}
`time_sec` is relative to the segment, so each segment is shifted by its start (`segment_start_sec` from the VastDB
row, or from the sidecar when a fixture carries it) and the segments are merged in time order.

`register(cam, video_path, det_rows, ...)` writes `<cam>__det__s1__c25.json.gz` (track-cache row shape
[f, t, tid, cls, name, conf, x1, y1, x2, y2]) into data.TRACK_CACHE, running stm.tracker when ids are missing,
and adds the cam to data.CAMS for this process. No network: sidecars must be dicts or local files.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import pandas as pd

from stm import data, tracker

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]
CLS = {n: i for i, n in enumerate(COCO)}
_SEG = re.compile(r"_segment_(\d+)_of_(\d+)")


def cls_id(label: str) -> int:
    """COCO class id for a label name (case/space tolerant); -1 when not a COCO name."""
    k = str(label).strip().lower().replace("_", " ")
    return CLS.get(k, CLS.get({"motorbike": "motorcycle", "aeroplane": "airplane", "tvmonitor": "tv"}.get(k, ""), -1))


def load_sidecar(path: str | Path) -> dict[str, Any]:
    """A local sidecar file (.json or gzipped .json.gz). s3:// URIs are refused: no network here."""
    if str(path).startswith("s3://"):
        raise ValueError("packd reads local sidecars only; fetch s3:// sidecars with vastkit first")
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _segments(det_rows: Iterable[Any], segment_sec: float | None) -> list[tuple[float, dict[str, Any], str]]:
    """Normalise inputs to [(segment_start_sec, sidecar, how_start_known)], sorted by start."""
    segs = []
    for i, r in enumerate(det_rows):
        if isinstance(r, (str, Path)):
            row, side = {}, load_sidecar(r)
        elif isinstance(r, dict) and "frames" in r:
            row, side = r, r
        elif isinstance(r, dict):
            row = r
            side = r.get("sidecar") or r.get("detections")
            if side is None and r.get("detection_sidecar_uri"):
                side = load_sidecar(r["detection_sidecar_uri"])
            if side is None:
                raise ValueError(f"row {i} has no sidecar, detections or detection_sidecar_uri")
        else:
            raise TypeError(f"row {i}: unsupported type {type(r).__name__}")
        start = row.get("segment_start_sec")
        start, how = (side.get("segment_start_sec") if start is None else start), "row"
        if start is None:
            m = _SEG.search(str(row.get("source") or side.get("segment_source") or ""))
            num = row.get("segment_number") or (int(m.group(1)) if m else None)
            if num is not None and segment_sec:
                start, how = (int(num) - 1) * float(segment_sec), "segment_number"
        segs.append((None if start is None else float(start), side, how, i))
    # segments without a known start are chained after the previous one by their own length (frame_count / fps)
    known = sorted([s for s in segs if s[0] is not None], key=lambda s: (s[0], s[3]))
    unknown = [s for s in segs if s[0] is None]
    out = [(s[0], s[1], s[2]) for s in known]
    cursor = max((s + _seg_len(sd) for s, sd, _ in out), default=0.0)
    for _, side, _, _ in unknown:
        out.append((cursor, side, "chained"))
        cursor += _seg_len(side)
    return out


def _seg_len(side: dict[str, Any]) -> float:
    fps = float(side.get("fps") or 30.0)
    if side.get("frame_count"):
        return float(side["frame_count"]) / fps
    ts = [float(f.get("time_sec", 0.0)) for f in side.get("frames", [])]
    return (max(ts) + 1.0 / fps) if ts else 0.0


def sidecars_to_rows(det_rows: Iterable[Any], fps: float | None = None, segment_sec: float | None = None,
                     size: tuple[int, int] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge segment sidecars into one time-ordered detections frame (columns data.COLS, tid -1 if absent).

    size: (w, h) of the video the cam will play; boxes from a sidecar whose video_shape / frame shape differs
    are scaled to it (bbox pixels are in the segment's own resolution).
    """
    segs = _segments(det_rows, segment_sec)
    recs = []
    starts = []
    for start, side, how in segs:
        sfps = float(side.get("fps") or fps or 30.0)
        seg_meta = {"start": round(start, 3), "how": how, "frames": len(side.get("frames", []))}
        starts.append(seg_meta)
        for fr in side.get("frames", []):
            ts = fr.get("time_sec")
            rel = float(ts) if ts is not None else float(fr.get("frame_index", 0)) / sfps
            t = start + rel
            sx = sy = 1.0
            shp = fr.get("shape") or side.get("video_shape")
            if size and size[0] and size[1] and shp and len(shp) >= 2 and shp[0] and shp[1]:
                sx, sy = size[0] / float(shp[1]), size[1] / float(shp[0])
                if abs(sx - 1) > 1e-6 or abs(sy - 1) > 1e-6:
                    seg_meta["scaled_from"] = [int(shp[1]), int(shp[0])]
            for d in fr.get("detections", []):
                b = d.get("bbox") or d.get("xyxy")
                if not b or len(b) != 4:
                    continue
                name = str(d.get("label", d.get("name", ""))).strip().lower()
                tid = d.get("track_id", d.get("tid"))
                recs.append({"t": t, "tid": -1 if tid is None else int(tid),
                             "cls": int(d["cls"]) if "cls" in d else cls_id(name), "name": name,
                             "conf": float(d.get("confidence", d.get("conf", 0.0))),
                             "x1": float(b[0]) * sx, "y1": float(b[1]) * sy,
                             "x2": float(b[2]) * sx, "y2": float(b[3]) * sy})
    df = pd.DataFrame(recs, columns=["t", "tid", "cls", "name", "conf", "x1", "y1", "x2", "y2"])
    vfps = float(fps or (segs[0][1].get("fps") if segs else None) or 30.0)
    if len(df):
        df = df.sort_values("t", kind="stable").reset_index(drop=True)
        # global frame index from time; distinct sidecar frames must never share an f
        ut = sorted(df.t.unique())
        fmap, last = {}, -1
        for t in ut:
            f = max(int(round(t * vfps)), last + 1)
            fmap[t], last = f, f
        df.insert(0, "f", df.t.map(fmap).astype(int))
    else:
        df.insert(0, "f", pd.Series(dtype=int))
    meta = {"segments": starts, "fps_used": vfps, "detections": int(len(df))}
    return df[data.COLS], meta


def register(cam: str, video_path: str | Path, det_rows: Iterable[Any], days: list[tuple] | None = None,
             title: str | None = None, credit: str = "Event neighbourhood-camera pack (Pack D)",
             segment_sec: float | None = None, simulated: list[str] | None = None) -> Path:
    """Convert detection rows to a track cache for `cam` and add it to data.CAMS. Returns the cache path."""
    video_path = Path(video_path)
    for p in sorted(data.TRACK_CACHE.glob(f"{cam}__det__*.json.gz")):
        if _cache_source(p) != "stm.packd":
            raise FileExistsError(f"{p.name} already exists and was not written by packd; pick another cam name")
    fps = w = h = None
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    rows_in = list(det_rows)
    df, meta = sidecars_to_rows(rows_in, fps=fps, segment_sec=segment_sec, size=(w, h) if w and h else None)
    retracked = bool(len(df)) and bool((df.tid < 0).all())
    if retracked:
        df = tracker.track(df)[data.COLS]
    if (h is None or w is None) and rows_in:
        first = rows_in[0] if isinstance(rows_in[0], dict) else {}
        shp = first.get("video_shape") or (first.get("sidecar") or {}).get("video_shape")
        if shp:
            h, w = int(shp[0]), int(shp[1])
    df = df.sort_values(["t", "tid"], kind="stable")
    rows = [[int(r.f), round(float(r.t), 3), int(r.tid), int(r.cls), str(r.name), round(float(r.conf), 3),
             round(float(r.x1), 1), round(float(r.y1), 1), round(float(r.x2), 1), round(float(r.y2), 1)]
            for r in df.itertuples(index=False)]
    payload = {"video": cam, "src_fps": meta["fps_used"], "stride": 1, "w": w, "h": h,
               "duration": round(float(df.t.max()), 3) if len(df) else 0.0,
               "names": {str(i): n for i, n in enumerate(COCO)}, "rows": rows, "model": "det",
               "extra": {"source": "stm.packd", "segments": meta["segments"], "retracked": retracked,
                         "video_path": str(video_path)}}
    data.TRACK_CACHE.mkdir(parents=True, exist_ok=True)
    out = data.TRACK_CACHE / f"{cam}__det__s1__c25.json.gz"
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    data.CAMS[cam] = {"video": video_path, "title": title or cam.replace("_", " "),
                      "days": days or [("A", "Clip", 0.0, None)], "simulated": simulated or [], "credit": credit}
    data.tracks.cache_clear()
    data.info.cache_clear()
    return out


def _cache_source(p: Path) -> str | None:
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return (json.load(fh).get("extra") or {}).get("source")
    except (OSError, ValueError, AttributeError):
        return None


def unregister(cam: str, delete_cache: bool = True) -> None:
    """Remove a runtime cam (and, by default, the cache file packd wrote for it)."""
    p = data.TRACK_CACHE / f"{cam}__det__s1__c25.json.gz"
    if delete_cache and p.exists() and _cache_source(p) == "stm.packd":  # never delete a cache packd did not write
        p.unlink()
    data.CAMS.pop(cam, None)
    data.tracks.cache_clear()
    data.info.cache_clear()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Register local Pack D detection sidecars as a camera")
    ap.add_argument("cam")
    ap.add_argument("video")
    ap.add_argument("sidecars", nargs="+", help="local sidecar .json/.json.gz files or directories of them")
    ap.add_argument("--segment-sec", type=float, default=None, help="segment length when starts are not given")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    files: list[Path] = []
    for s in a.sidecars:
        p = Path(s)
        files += sorted(p.glob("*.json*")) if p.is_dir() else [p]
    out = register(a.cam, a.video, files, title=a.title, segment_sec=a.segment_sec)
    df = data.tracks(a.cam)
    obj = data.objects(a.cam)
    raw = json.load(gzip.open(out, "rt", encoding="utf-8"))
    segs = ", ".join(f"{s['start']:.2f}s ({s['how']}, {s['frames']} frames)" for s in raw["extra"]["segments"])
    line = (f"{a.cam}: {len(files)} sidecars [{segs}], {len(df)} detections, retracked={raw['extra']['retracked']}, "
            f"{df.tid[df.tid >= 0].nunique()} tracks, {len(obj)} objects (>=8 frames), "
            f"{int(obj.parked.sum()) if len(obj) else 0} parked -> {out.name}")
    print(line)
    (data.OUT / "packd_summary.txt").write_text(
        "packd: local detection sidecars merged by segment start, re-tracked when ids are missing.\n" + line + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
