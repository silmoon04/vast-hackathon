"""Frame reading, raw YOLO detections and an FPS benchmark (webcam index or local file).

    python -m vastkit.frames --source clip.mp4 --weights yolo26s.engine --imgsz 640
    python -m vastkit.frames --source 0 --weights yolo11s.pt --jsonl runs/dets.jsonl --max-frames 300

Prints a JSON summary {frames, dropped, wall_fps, latency_ms:{p50,p95,max}}. With --jsonl, also
writes one line per processed frame: {t, frame, dropped, latency_ms, detections:[{label, conf, bbox}]}.
Deciding what any detection means is left to the caller.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def open_source(source: str | int):
    """cv2.VideoCapture for a webcam index or a local file (buffer of 1 so live reads stay fresh)."""
    import cv2
    src: Any = int(source) if isinstance(source, int) or str(source).isdigit() else str(source)
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source {source!r}")
    return cap


class _Latest:
    """Background reader keeping only the newest frame; `paced` replays a file at its own fps.

    The reader owns the capture and releases it on exit, so release() never races an in-flight read().
    """

    def __init__(self, cap, paced: bool):
        import cv2
        self.cap, self.lock = cap, threading.Lock()
        self.frame: np.ndarray | None = None
        self.idx, self.ok, self.stopped = -1, True, False
        self.period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30) if paced else 0.0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        nxt = time.perf_counter()
        try:
            while not self.stopped:
                ok, f = self.cap.read()
                if not ok:
                    break
                with self.lock:
                    self.frame, self.idx = f, self.idx + 1
                if self.period:
                    nxt += self.period
                    time.sleep(max(0.0, nxt - time.perf_counter()))
        finally:
            self.ok = False
            self.cap.release()


def iter_frames(source: str | int, *, paced: bool = True, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray, int]]:
    """Yield (frame_index, t_seconds, bgr_frame, dropped_since_last).

    Webcams always read latest-frame-only. Files with paced=True are replayed at source fps and frames
    the consumer is too slow for are skipped (like a live camera); paced=False yields every frame.
    dropped_since_last counts source frames never read since the previous yield; stride skips are not drops.
    """
    cap = open_source(source)
    live = isinstance(source, int) or str(source).isdigit()
    reader = _Latest(cap, paced=not live) if (live or paced) else None
    t0, last, dropped = time.perf_counter(), -1, 0
    try:
        while True:
            if reader:
                with reader.lock:
                    idx, frame = reader.idx, reader.frame
                if frame is None or idx == last:
                    if not reader.ok:
                        break
                    time.sleep(0.001)
                    continue
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                idx = last + 1
            dropped, last = dropped + idx - last - 1, idx  # carried across stride skips
            if stride > 1 and idx % stride:
                continue
            yield idx, time.perf_counter() - t0, frame, dropped
            dropped = 0
    finally:
        if reader:
            reader.stopped = True
            reader.th.join(timeout=2.0)  # a reader stuck in read() still releases cap once it returns
        else:
            cap.release()


class Detector:
    """Ultralytics YOLO (.pt / .engine / .onnx) returning plain detection dicts."""

    def __init__(self, weights: str | Path, *, imgsz: int = 640, conf: float = 0.3, device: Any = 0):
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="detect")
        self.kw = dict(imgsz=imgsz, conf=conf, device=device, verbose=False)
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), **self.kw)  # warm up (TRT/CUDA init)

    def __call__(self, frame: np.ndarray) -> list[dict[str, Any]]:
        r = self.model.predict(frame, **self.kw)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        return [{"label": r.names[int(c)], "conf": round(float(s), 3), "bbox": [round(float(v), 1) for v in xyxy]}
                for c, s, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]


def bench(source: str | int, detector: Detector, *, paced: bool = True, max_frames: int | None = None,
          jsonl: str | Path | None = None) -> dict[str, Any]:
    if jsonl:
        Path(jsonl).parent.mkdir(parents=True, exist_ok=True)
    out = open(jsonl, "w", encoding="utf-8") if jsonl else None
    lat, dropped, n, t_first, t_last = [], 0, 0, None, 0.0
    try:
        for idx, t, frame, d in iter_frames(source, paced=paced):
            s = time.perf_counter()
            dets = detector(frame)
            ms = (time.perf_counter() - s) * 1000
            lat.append(ms)
            dropped += d
            n += 1
            t_first = t if t_first is None else t_first
            t_last = t
            if out:
                out.write(json.dumps({"t": round(t, 3), "frame": idx, "dropped": d, "latency_ms": round(ms, 2),
                                      "detections": dets}) + "\n")
            if max_frames and n >= max_frames:
                break
    finally:
        if out:
            out.close()
    a = np.asarray(lat or [0.0])
    span = max(t_last - (t_first or 0.0), 1e-9)
    return {"frames": n, "dropped": dropped, "paced": paced, "wall_fps": round((n - 1) / span, 1) if n > 1 else None,
            "latency_ms": {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
                           "max": round(float(a.max()), 2)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, help="webcam index or local video file")
    ap.add_argument("--weights", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--every-frame", action="store_true", help="files: process every frame (no real-time pacing)")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--jsonl", help="also write per-frame raw detections here")
    a = ap.parse_args(argv)
    det = Detector(a.weights, imgsz=a.imgsz, conf=a.conf)
    print(json.dumps(bench(a.source, det, paced=not a.every_frame, max_frames=a.max_frames, jsonl=a.jsonl)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
