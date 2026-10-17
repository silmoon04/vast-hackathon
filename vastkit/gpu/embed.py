"""Cosmos-Embed1 client (POST /v1/embeddings). Always 256-d for the 224p model the pipeline uses.

Text queries and video clips land in the same space as vss-collection.vectors /
vectors_visual only when you use the SAME embed server as ingest.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

from .. import http as _http
from ..config import Settings, settings as _settings
from .cosmos import video_data_uri

DIM = 256


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


class CosmosEmbed:
    def __init__(self, base_url: str | None = None, *, model: str | None = None, token: str | None = None,
                 proxy: str | None = None, timeout: float = 180.0, cfg: Settings | None = None,
                 dim: int = DIM):
        cfg = cfg or _settings()
        self.base_url = (base_url or cfg.cosmos_embed_url).rstrip("/")
        if not self.base_url:
            raise ValueError("COSMOS_EMBED1_URL unset (or VSS_MOCK=1 for the GPU mock)")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.model = model or cfg.cosmos_embed_model or "nvidia/cosmos-embed1"
        self.dim = dim
        self.client = _http.make_client(self.base_url, proxy=proxy if proxy is not None else cfg.socks_proxy,
                                        token=token if token is not None else cfg.gpu_bearer_token,
                                        timeout=timeout)

    def _embed(self, inp: str | list[str], request_type: str) -> list[list[float]]:
        body: dict[str, Any] = {"input": inp, "model": self.model, "request_type": request_type,
                                "encoding_format": "float"}
        r = _http.request(self.client, "POST", "/v1/embeddings", json=body)
        if r.status_code != 200:
            raise _http.APIError(r, "embeddings")
        data = r.json().get("data") or []
        if not data:
            raise RuntimeError("embed server returned no data")
        vecs = [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
        for v in vecs:
            if len(v) != self.dim:
                raise RuntimeError(f"expected {self.dim}-d embedding, got {len(v)} (wrong model/variant?)")
        return vecs

    def text(self, query: str) -> list[float]:
        return self._embed(query, "query")[0]

    def texts(self, items: list[str]) -> list[list[float]]:
        return self._embed(items, "bulk_text") if items else []

    def video(self, clip: str | Path | bytes) -> list[float]:
        """Embed one short clip (8 frames are sampled server-side; keep it <= ~15 s)."""
        return self._embed(video_data_uri(clip), "query")[0]

    def ready(self) -> bool:
        """/v1/health/ready decides; /health or /v1/models only when that route 404/405s."""
        try:
            code = self.client.get("/v1/health/ready").status_code
            if code not in (404, 405):
                return code == 200
            return any(self.client.get(p).status_code == 200 for p in ("/health", "/v1/models"))
        except Exception:
            return False

    def close(self) -> None:
        self.client.close()
