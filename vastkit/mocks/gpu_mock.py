"""Deterministic stubs for the four event GPU endpoints, on one port (default 8766).

  Cosmos3-Reason  GET /v1/models, /v1/health/ready|live, POST /v1/chat/completions
  Cosmos-Embed1   POST /v1/embeddings      (hash-seeded unit vectors, 256-d)
  YOLO            GET /healthz, POST /v1/infer
  Canary-1B       POST /v1/audio/transcriptions

Reproduces the NIM quirks clients must survive: fps + num_frames together -> 400,
fps above MOCK_SOURCE_FPS (30) -> 400, unknown model id -> 404, think/answer format when
the reasoning suffix is present. For realistic outputs, point clients at the vastkit/local/ servers
(``python -m vastkit local``) instead.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import struct
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

MODEL_ID = os.environ.get("MOCK_COSMOS_MODEL", "nvidia/cosmos3-nano-reasoner")
SOURCE_FPS = float(os.environ.get("MOCK_SOURCE_FPS", "30"))
app = FastAPI(title="gpu-mock")


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "mock"}]}


@app.get("/v1/health/ready")
def ready() -> Dict[str, str]:
    return {"status": "ready"}


@app.get("/v1/health/live")
def live() -> Dict[str, str]:
    return {"status": "live"}


def _text_of(content: Union[str, List[Dict[str, Any]]]) -> tuple:
    if isinstance(content, str):
        return content, 0
    text, vids = "", 0
    for part in content:
        if part.get("type") == "text":
            text += part.get("text", "")
        elif part.get("type") in ("video_url", "video"):
            vids += 1
    return text, vids


@app.post("/v1/chat/completions")
def chat(body: Dict[str, Any]) -> Dict[str, Any]:
    # Not from the blueprint: the 404 follows vLLM's OpenAI-server wording; the two 400s are the mock's own
    # wording for the NIM-documented fps rules (vastkit.gpu.cosmos keys on "exclusive" / "exceeds ... N").
    if body.get("model") not in (MODEL_ID, "local"):
        raise HTTPException(404, f"The model `{body.get('model')}` does not exist.")
    vid = (body.get("media_io_kwargs") or {}).get("video") or {}
    if "fps" in vid and "num_frames" in vid:
        raise HTTPException(400, "media_io_kwargs: fps and num_frames are mutually exclusive")
    if vid.get("fps", 0) > SOURCE_FPS:
        raise HTTPException(400, f"Requested fps {vid['fps']} exceeds video fps {SOURCE_FPS}")
    last = body["messages"][-1]
    text, vids = _text_of(last.get("content", ""))
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        props = (rf.get("json_schema", {}).get("schema", {}).get("properties") or {})
        if "events" in props:
            content = json.dumps({"events": [
                {"start": "00:00.50", "end": "00:02.00", "caption": "A person walks across the frame."},
                {"start": "00:02.00", "end": "00:04.50", "caption": "An object stays still."}]})
        else:
            content = json.dumps({k: (0 if v.get("type") in ("number", "integer") else
                                      False if v.get("type") == "boolean" else
                                      [] if v.get("type") == "array" else "mock")
                                  for k, v in props.items()})
    elif "<think>" in text:
        content = ("<think>\nThe clip shows motion consistent with the question; checking timing.\n</think>\n\n"
                   f"<answer>\nMock answer ({vids} video part(s)).\n</answer>")
    else:
        content = f"Mock answer to: {text[:80]} ({vids} video part(s))."
    return {"id": "chatcmpl-mock", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1950 * max(1, vids), "completion_tokens": 40, "total_tokens": 1990}}


def _vec(seed: str, dim: int = 256) -> List[float]:
    out: List[float] = []
    i = 0
    while len(out) < dim:
        h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        out += [(x / 2**31) - 1.0 for x in struct.unpack("<8I", h)]
        i += 1
    out = out[:dim]
    n = math.sqrt(sum(x * x for x in out))
    return [x / n for x in out]


class EmbedReq(BaseModel):
    input: Union[str, List[str]]
    model: str = "nvidia/cosmos-embed1"
    request_type: Optional[str] = "query"
    encoding_format: Optional[str] = "float"


@app.post("/v1/embeddings")
def embeddings(b: EmbedReq) -> Dict[str, Any]:
    items = b.input if isinstance(b.input, list) else [b.input]
    if not items:
        raise HTTPException(400, "input must not be empty")
    data = []
    for i, s in enumerate(items):
        key = hashlib.sha256(s.encode()).hexdigest() if s.startswith("data:video") else s.lower().strip()
        data.append({"object": "embedding", "index": i, "embedding": _vec(key)})
    return {"object": "list", "data": data, "model": b.model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "model": "yolo11s.pt", "device": "mock", "cuda_available": False,
            "model_loaded": True, "inputs": ["url", "video_base64"]}


class InferReq(BaseModel):
    url: Optional[str] = None
    video_base64: Optional[str] = None
    filename: Optional[str] = None
    include_frames: bool = False


@app.post("/v1/infer")
def infer(b: InferReq) -> Dict[str, Any]:
    # error details mirror blueprint scripts/vss-blueprint-models/yolo-infer/main.py
    if bool(b.url) == bool(b.video_base64):
        raise HTTPException(422, "Provide exactly one of: url, video_base64")
    raw = b.video_base64 or ""
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[-1]
    try:
        data = base64.b64decode(raw, validate=True) if raw else b""
    except Exception as exc:
        raise HTTPException(400, f"Invalid base64 in video_base64: {exc}")
    if b.video_base64 and len(data) < 12:
        raise HTTPException(400, "Decoded video is too small to be a valid MP4")
    frames = [{"frame_index": i, "shape": [720, 1280], "detections": [
        {"label": "person", "confidence": 0.8812, "bbox": [400 + 4 * i, 200, 480 + 4 * i, 520]},
        {"label": "truck", "confidence": 0.7121, "bbox": [700, 260, 1100, 600]}]} for i in range(10)]
    counts = {"person": 10, "truck": 10}  # raw totals across frames (as the real server does)
    # source: the PERCEPTION_SOURCE default in blueprint scripts/vss-blueprint-models/yolo-infer/main.py
    pj = {"source": "yolo11_coco", "frames": frames if b.include_frames else None,
          "object_classes": sorted(counts), "object_counts": counts, "max_detection_conf": 0.8812,
          "frame_count": len(frames), "detection_count": 20}
    return {"ok": True, "perception_ok": True, "perception_json": pj, "object_classes": sorted(counts),
            "object_counts": counts, "max_detection_conf": 0.8812,
            "frames": frames if b.include_frames else None}


@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model: Optional[str] = Form(None),
                     language: Optional[str] = Form(None), response_format: Optional[str] = Form(None)):
    data = await file.read()
    return {"text": f"mock transcript of {file.filename} ({len(data)} bytes)"}


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="GPU endpoint stubs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    print(f"[gpu-mock] http://{a.host}:{a.port}  (cosmos model id {MODEL_ID})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
