"""Local mirror of the hosted Cosmos-Embed1 /v1/embeddings endpoint (256-d, nvidia/Cosmos-Embed1-224p).

Runs in its own venv because the model's remote code needs transformers 4.x:

    .venv-embed/Scripts/python.exe -m vastkit.local.embed_server --port 8003
    then  COSMOS_EMBED1_URL=http://127.0.0.1:8003

Text and video share one 256-d space. Vectors from this mirror are NOT comparable with vectors the
hosted embedder wrote into VastDB unless both run the same checkpoint and preprocessing; use the
hosted endpoint for anything that queries the team's collection.
"""
from __future__ import annotations

import argparse
import base64
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


REPO = "nvidia/Cosmos-Embed1-224p"
REVISION = os.environ.get("LOCAL_EMBED_REVISION", "787e0b996f5260a71ad474a283c90539a2e12986")  # pinned remote code
SERVED_ID = os.environ.get("LOCAL_EMBED_MODEL", "nvidia/cosmos-embed1")
N_FRAMES = 8
BEARER = os.environ.get("LOCAL_BEARER", "")

app = FastAPI(title="local cosmos-embed1 mirror")
_state: dict[str, Any] = {"model": None, "proc": None, "error": None}
_lock = threading.Lock()


def load() -> None:
    try:
        import torch
        from transformers import AutoModel, AutoProcessor
        t0 = time.time()
        _state["proc"] = AutoProcessor.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
        _state["model"] = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).to(
            "cuda", dtype=torch.bfloat16).eval()
        print(f"[embed] loaded {REPO}@{REVISION[:8]} in {time.time() - t0:.1f}s", flush=True)
    except Exception as e:
        _state["error"] = repr(e)
        print(f"[embed] load failed: {e!r}", flush=True)


@app.middleware("http")
async def auth(request: Request, call_next):
    if BEARER and not request.url.path.startswith("/v1/health"):
        got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if got != BEARER.removeprefix("Bearer ").strip():
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_ID, "object": "model", "owned_by": "local"}]}


@app.get("/v1/health/live")
def live():
    return {"status": "live"}


@app.get("/v1/health/ready")
def ready():
    if _state["model"] is None:
        return JSONResponse({"status": "not ready", "error": _state["error"]}, status_code=503)
    return {"status": "ready"}


def frames8(path: Path):
    """8 frames spread uniformly over the clip, resized to 224x224 RGB."""
    import av
    import cv2
    import numpy as np
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        total = float(s.duration * s.time_base) if s.duration else (c.duration or 0) / 1e6
        targets = np.linspace(0, total, N_FRAMES + 2)[1:-1]
        out, k = [], 0
        for f in c.decode(s):
            if f.time is None:
                continue
            while k < N_FRAMES and f.time >= targets[k]:
                out.append(f.to_ndarray(format="rgb24")); k += 1
            if k == N_FRAMES:
                break
    if not out:
        raise HTTPException(400, "no frames decoded from video input")
    while len(out) < N_FRAMES:
        out.append(out[-1])
    return np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in out])


def _video_from_uri(uri: str) -> Path:
    raw = base64.b64decode(uri.split(",", 1)[-1])
    fd, p = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return Path(p)


@app.post("/v1/embeddings")
def embeddings(body: dict[str, Any]):
    import numpy as np
    import torch
    if _state["model"] is None:
        raise HTTPException(503, "model loading" if not _state["error"] else _state["error"])
    inp = body.get("input")
    items = [inp] if isinstance(inp, str) else list(inp or [])
    if not items or not all(isinstance(x, str) for x in items):
        raise HTTPException(422, "input must be a string or a list of strings")
    is_vid = [x.startswith("data:video") for x in items]
    proc, model = _state["proc"], _state["model"]
    vecs: list[Any] = [None] * len(items)
    with _lock, torch.no_grad():
        texts = [(i, x) for i, x in enumerate(items) if not is_vid[i]]
        if texts:
            ti = proc(text=[x for _, x in texts]).to("cuda", dtype=torch.bfloat16)
            T = model.get_text_embeddings(**ti).text_proj.float().cpu().numpy()
            for (i, _), v in zip(texts, T):
                vecs[i] = v
        for i, x in enumerate(items):
            if not is_vid[i]:
                continue
            p = _video_from_uri(x)
            try:
                clip = frames8(p)
            finally:
                p.unlink(missing_ok=True)
            vi = proc(videos=np.stack([clip.transpose(0, 3, 1, 2)])).to("cuda", dtype=torch.bfloat16)
            vecs[i] = model.get_video_embeddings(**vi).visual_proj.float().cpu().numpy()[0]
    data = [{"object": "embedding", "index": i, "embedding": [float(f) for f in v]} for i, v in enumerate(vecs)]
    return {"object": "list", "data": data, "model": body.get("model") or SERVED_ID,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    a = ap.parse_args()
    threading.Thread(target=load, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
