"""Client for the pipeline's YOLO server (POST /v1/infer, GET /healthz).

Response differences from what is stored in VastDB:
- every frame is inferred (no stride) so include_frames on long clips is huge
- frames carry no time_sec; ``with_times()`` adds it from fps
- object_counts are raw totals across frames; ``unique_counts()`` gives the
  max-per-frame estimate the ingest sidecar stores (blueprint
  source-code/ingest/video-detector/common/clients.py, object_counts.py)
"""
from __future__ import annotations

import base64
from collections import Counter
from pathlib import Path
from typing import Any

from .. import http as _http
from ..config import Settings, settings as _settings


def unique_counts(frames: list[dict[str, Any]]) -> dict[str, int]:
    best: Counter[str] = Counter()
    for f in frames or []:
        c = Counter(d["label"] for d in f.get("detections", []))
        for k, v in c.items():
            best[k] = max(best[k], v)
    return dict(best)


def with_times(frames: list[dict[str, Any]], fps: float) -> list[dict[str, Any]]:
    return [dict(f, time_sec=round(f["frame_index"] / fps, 3)) for f in frames or []]


class YoloClient:
    def __init__(self, base_url: str | None = None, *, token: str | None = None, proxy: str | None = None,
                 timeout: float = 180.0, cfg: Settings | None = None):
        cfg = cfg or _settings()
        self.base_url = (base_url or cfg.yolo_url).rstrip("/")
        if not self.base_url:
            raise ValueError("YOLO_URL unset (or VSS_MOCK=1 for the GPU mock)")
        self.client = _http.make_client(self.base_url, proxy=proxy if proxy is not None else cfg.socks_proxy,
                                        token=token if token is not None else cfg.gpu_bearer_token,
                                        timeout=timeout)

    def health(self) -> dict[str, Any]:
        r = _http.request(self.client, "GET", "/healthz")
        if r.status_code != 200:
            raise _http.APIError(r, "healthz")
        return r.json()

    def ready(self) -> bool:
        try:
            h = self.health()
            return bool(h.get("ok") and h.get("model_loaded", True))
        except Exception:
            return False

    def infer(self, clip: str | Path | bytes, *, include_frames: bool = True,
              filename: str | None = None) -> dict[str, Any]:
        data = clip if isinstance(clip, bytes) else Path(clip).read_bytes()
        if len(data) < 12:
            raise ValueError("clip too small to be a valid MP4")
        name = filename or (Path(clip).name if not isinstance(clip, bytes) else "clip.mp4")
        body = {"video_base64": base64.b64encode(data).decode(), "filename": name,
                "include_frames": include_frames}
        r = _http.request(self.client, "POST", "/v1/infer", json=body)
        if r.status_code != 200:
            raise _http.APIError(r, "infer")
        return r.json()

    def close(self) -> None:
        self.client.close()
