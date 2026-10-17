"""Local mirror of the VAST GPU YOLO endpoint (same /healthz and /v1/infer contract), on this laptop's GPU.

    .venv/Scripts/python.exe -m vastkit.local.yolo_server --port 8002 [--weights <models>/yolo11s.pt]
    then  YOLO_URL=http://127.0.0.1:8002

Default weights are yolo11s.pt, the model the hosted endpoint serves, so classes and counts match.
Point --weights at yolo26s.engine for speed. perception_json.source is PERCEPTION_SOURCE (default
yolo11_coco, as hosted), whatever the weights. Optional extensions, ignored by the real service:
`stride` (infer every Nth frame) and `conf` (default 0.4, as hosted).
"""
from __future__ import annotations

import argparse
import base64
import binascii
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MAX_B64_CHARS = 280 * 1024 * 1024
MAX_BYTES = 200 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

WEIGHTS = os.environ.get("LOCAL_YOLO_WEIGHTS", "yolo11s.pt")
BEARER = os.environ.get("LOCAL_BEARER", "")
DEVICE = os.environ.get("LOCAL_YOLO_DEVICE", "0")

app = FastAPI(title="local yolo mirror")
_model: Any = None
_lock = threading.Lock()


class InferRequest(BaseModel):
    url: str | None = None
    video_base64: str | None = None
    filename: str | None = None
    include_frames: bool = False
    stride: int = 1
    conf: float = 0.4


def model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(WEIGHTS, task="detect")
    return _model


@app.middleware("http")
async def auth(request: Request, call_next):
    if BEARER and request.url.path != "/healthz":
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if got != BEARER.removeprefix("Bearer ").strip():
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/healthz")
def healthz():
    import torch
    return {"ok": True, "model": Path(WEIGHTS).name, "device": DEVICE, "cuda_available": torch.cuda.is_available(),
            "model_loaded": _model is not None, "inputs": ["url", "video_base64"]}


# HTTP error details here and in infer() mirror the hosted server, blueprint
# scripts/vss-blueprint-models/yolo-infer/main.py. The "too large" 413s and the infer-base64 422 are this
# mirror's own wording.
def _materialize(req: InferRequest) -> Path:
    if bool(req.url) == bool(req.video_base64):
        raise HTTPException(422, "Provide exactly one of: url, video_base64")
    suffix = Path(req.filename or "clip.mp4").suffix.lower()
    suffix = suffix if suffix in VIDEO_SUFFIXES else ".mp4"
    if req.video_base64:
        b64 = req.video_base64
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        if len(b64) > MAX_B64_CHARS:
            raise HTTPException(413, "video_base64 too large")
        try:
            raw = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError) as e:
            raise HTTPException(400, f"Invalid base64 in video_base64: {e}")
    else:
        if not req.url.startswith(("http://", "https://")):
            raise HTTPException(400, f"Unsupported URL scheme: {req.url.split(':', 1)[0]}")
        import httpx
        try:
            r = httpx.get(req.url, timeout=120, follow_redirects=True)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Presigned URL fetch failed: {e}")
        if r.status_code != 200:
            raise HTTPException(502, f"Presigned URL fetch failed: HTTP {r.status_code}")
        raw = r.content
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "video too large")
    if len(raw) < 12:
        raise HTTPException(400, "Decoded video is too small to be a valid MP4")
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return Path(path)


def run_infer(path: Path, conf: float, stride: int, include_frames: bool) -> dict[str, Any]:
    frames, counts, classes, maxc, ndet = [], {}, set(), 0.0, 0
    with _lock:
        m = model()
        for i, r in enumerate(m.predict(str(path), stream=True, conf=conf, device=DEVICE, verbose=False,
                                        vid_stride=max(1, stride))):
            dets = []
            if r.boxes is not None:
                for c, s, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    label = r.names[int(c)]
                    dets.append({"label": label, "confidence": round(float(s), 4), "bbox": [int(v) for v in xyxy]})
                    counts[label] = counts.get(label, 0) + 1  # raw totals across frames, as hosted
                    classes.add(label)
                    maxc = max(maxc, float(s))
            ndet += len(dets)
            frames.append({"frame_index": i * max(1, stride), "shape": list(r.orig_shape), "detections": dets})
    pj = {"source": os.environ.get("PERCEPTION_SOURCE", "yolo11_coco"), "frames": frames if include_frames else None,
          "object_classes": sorted(classes), "object_counts": counts, "max_detection_conf": round(maxc, 4),
          "frame_count": len(frames), "detection_count": ndet}
    return {"ok": True, "perception_ok": len(frames) > 0, "perception_json": pj, "object_classes": sorted(classes),
            "object_counts": counts, "max_detection_conf": round(maxc, 4),
            "frames": frames if include_frames else None}


@app.post("/v1/infer")
def infer(req: InferRequest):
    path = _materialize(req)
    try:
        return run_infer(path, req.conf, req.stride, req.include_frames)
    except HTTPException:
        raise
    except Exception as e:  # hosted error shape (yolo-infer/main.py, see _materialize)
        raise HTTPException(502, f"Inference failed: {e}")
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/infer-base64")
def infer_b64(req: InferRequest):
    if not req.video_base64 or req.url:
        raise HTTPException(422, "Provide video_base64 (and no url)")
    return infer(req)


def main() -> None:
    global WEIGHTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--no-preload", action="store_true")
    a = ap.parse_args()
    WEIGHTS = a.weights
    if not a.no_preload:
        model()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
