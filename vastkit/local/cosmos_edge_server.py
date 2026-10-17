"""Local OpenAI-compatible mirror of the hosted Cosmos3 reasoning endpoint, backed by nvidia/Cosmos3-Edge.

    .venv/Scripts/python.exe -m vastkit.local.cosmos_edge_server --port 8001
    then  COSMOS3_REASON_URL=http://127.0.0.1:8001   (model id is discovered from /v1/models)

Measured on an RTX 4080 Laptop (bf16, 448px long side, 2 fps): ~11 s load, 5.8 GB peak VRAM,
~1.9k prompt tokens for an 8 s clip, 7-9 output tok/s. The hosted NIM is much faster, so use
this for offline development and as a fallback, not for throughput numbers.

Accepted (as on the NIM): content parts text | video_url | image_url (data: URIs or http(s) URLs),
max_tokens, temperature, top_p, media_io_kwargs {fps | num_frames} (both -> 400; fps above the
source -> 400), chat_template_kwargs {enable_thinking}. response_format json_schema is best-effort:
the schema is appended to the prompt and the output is not constrained.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


MODEL_ID = os.environ.get("LOCAL_COSMOS_MODEL", "nvidia/Cosmos3-Edge")
ALIASES = {a.strip() for a in os.environ.get("LOCAL_COSMOS_ALIASES", "").split(",") if a.strip()}
MAX_SIDE = int(os.environ.get("LOCAL_COSMOS_MAX_SIDE", "448"))
DEFAULT_FPS = float(os.environ.get("LOCAL_COSMOS_FPS", "2"))
MAX_FRAMES = int(os.environ.get("LOCAL_COSMOS_MAX_FRAMES", "32"))
THINK_DEFAULT = os.environ.get("LOCAL_COSMOS_THINK_DEFAULT", "0") == "1"
BEARER = os.environ.get("LOCAL_BEARER", "")

app = FastAPI(title="local cosmos mirror")
_state: dict[str, Any] = {"model": None, "processor": None, "error": None}
_lock = threading.Lock()


def load() -> None:
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        t0 = time.time()
        _state["processor"] = AutoProcessor.from_pretrained(MODEL_ID)
        _state["model"] = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
        print(f"[cosmos] loaded {MODEL_ID} in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:  # surfaced via /v1/health/ready
        _state["error"] = repr(e)
        print(f"[cosmos] load failed: {e!r}", flush=True)


@app.middleware("http")
async def auth(request: Request, call_next):
    if BEARER and not request.url.path.startswith("/v1/health"):
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if got != BEARER.removeprefix("Bearer ").strip():
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "local"}
                                       for m in [MODEL_ID, *sorted(ALIASES)]]}


@app.get("/v1/health/live")
def live():
    return {"status": "live"}


@app.get("/v1/health/ready")
def ready():
    if _state["model"] is None:
        return JSONResponse({"status": "not ready", "error": _state["error"]}, status_code=503)
    return {"status": "ready"}


def _fetch(url: str, suffix: str) -> Path:
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[-1])
    elif url.startswith(("http://", "https://")):
        import httpx
        r = httpx.get(url, timeout=120, follow_redirects=True)
        if r.status_code != 200:
            raise HTTPException(400, f"could not fetch media: HTTP {r.status_code}")
        raw = r.content
    else:
        raise HTTPException(400, "media url must be a data: URI or http(s) URL")
    fd, p = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return Path(p)


def _resize(img):
    import cv2
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
    return img


def decode_video(path: Path, fps: float | None, num_frames: int | None):
    import av
    import numpy as np
    from transformers.video_utils import VideoMetadata
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        src_fps = float(s.average_rate or 30)
        total = s.frames or 0
        if fps is not None and fps > src_fps + 1e-6:
            raise HTTPException(400, f"Requested fps {fps} exceeds video fps {src_fps:.2f}")
        frames, idx = [], []
        decoded = [f for f in c.decode(s)] if num_frames else None
        if decoded is not None:
            total = len(decoded)
            picks = sorted({min(total - 1, int(round(i))) for i in np.linspace(0, total - 1, num_frames)})
            for i in picks:
                frames.append(_resize(decoded[i].to_ndarray(format="rgb24"))); idx.append(i)
        else:
            step = max(1, round(src_fps / (fps or DEFAULT_FPS)))
            for i, f in enumerate(c.decode(s)):
                if i % step == 0:
                    frames.append(_resize(f.to_ndarray(format="rgb24"))); idx.append(i)
    if not frames:
        raise HTTPException(400, "no frames decoded")
    if len(frames) > MAX_FRAMES:  # keep the prompt bounded on a laptop GPU
        keep = np.linspace(0, len(frames) - 1, MAX_FRAMES).round().astype(int)
        frames, idx = [frames[k] for k in keep], [idx[k] for k in keep]
    if len(frames) % 2:  # the processor's temporal patch needs an even count
        frames.append(frames[-1]); idx.append(idx[-1])
    meta = VideoMetadata(total_num_frames=max(total, idx[-1] + 1), fps=src_fps, frames_indices=idx)
    return np.stack(frames), meta


def decode_image(path: Path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return _resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def build(body: dict[str, Any]):
    mio = body.get("media_io_kwargs") or {}
    mio = mio.get("video", mio)  # vLLM nests per modality: {"video": {"fps": 2}}
    fps, nf = mio.get("fps"), mio.get("num_frames")
    if fps is not None and nf is not None:
        raise HTTPException(400, "fps and num_frames are mutually exclusive")
    msgs, videos, metas, images, tmp = [], [], [], [], []
    schema = ((body.get("response_format") or {}).get("json_schema") or {}).get("schema")
    try:
        for m in body.get("messages", []):
            content = m.get("content")
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
            out = []
            for p in parts:
                t = p.get("type")
                if t == "text":
                    out.append({"type": "text", "text": p.get("text", "")})
                elif t == "video_url":
                    path = _fetch(p["video_url"]["url"], ".mp4"); tmp.append(path)
                    v, meta = decode_video(path, fps, nf)
                    videos.append(v); metas.append(meta); out.append({"type": "video"})
                elif t == "image_url":
                    path = _fetch(p["image_url"]["url"], ".jpg"); tmp.append(path)
                    images.append(decode_image(path)); out.append({"type": "image"})
                else:
                    raise HTTPException(400, f"unsupported content part type {t!r}")
            msgs.append({"role": m.get("role", "user"), "content": out})
    finally:
        for p in tmp:
            p.unlink(missing_ok=True)
    if schema and msgs:  # small models echo a raw schema back, so show a filled-in example instead
        msgs[-1]["content"].append({"type": "text", "text": "Respond with only JSON shaped exactly like this "
                                    "example (replace the placeholder values):\n" + json.dumps(_example(schema))})
    return msgs, videos, metas, images


def _example(s: dict[str, Any], key: str = "") -> Any:
    if "enum" in s:
        return s["enum"][0]
    t = s.get("type")
    t = t[0] if isinstance(t, list) else t
    if t == "object" or "properties" in s:
        return {k: _example(v, k) for k, v in (s.get("properties") or {}).items()}
    if t == "array":
        return [_example(s.get("items") or {"type": "string"}, key)]
    if t in ("number", "integer"):
        return 0
    if t == "boolean":
        return False
    return f"<{key or 'text'}>"


@app.post("/v1/chat/completions")
def chat(body: dict[str, Any]):
    import torch
    if body.get("model") not in {MODEL_ID, *ALIASES}:
        raise HTTPException(404, f"The model `{body.get('model')}` does not exist.")
    if body.get("stream"):
        raise HTTPException(400, "stream=true is not supported by the local mirror")
    if _state["model"] is None:
        raise HTTPException(503, "model loading" if not _state["error"] else _state["error"])
    msgs, videos, metas, images = build(body)
    think = (body.get("chat_template_kwargs") or {}).get("enable_thinking", THINK_DEFAULT)
    proc, model = _state["processor"], _state["model"]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    kw: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
    if videos:
        kw.update(videos=videos, video_metadata=metas, do_sample_frames=False)
    if images:
        kw["images"] = images
    max_new = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
    temp = float(body.get("temperature") or 0)
    gen: dict[str, Any] = {"max_new_tokens": max_new, "do_sample": temp > 0}
    if temp > 0:
        gen.update(temperature=temp, top_p=float(body.get("top_p") or 1.0))
        if body.get("top_k"):
            gen["top_k"] = int(body["top_k"])
    with _lock:
        inputs = proc(**kw).to(model.device)
        for k, v in inputs.items():
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                inputs[k] = v.to(torch.bfloat16)
        n_in = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, **gen)
    n_out = out.shape[1] - n_in
    content = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "length" if n_out >= max_new else "stop"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    threading.Thread(target=load, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
