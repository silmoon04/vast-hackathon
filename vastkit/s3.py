"""Fetch segment/chunk objects from VAST S3 (data VIP) by their s3:// URI, with a content cache.

The data VIP is private: run on the lab VM, or fall back to VSSClient.download (which proxies
/videos/stream). Presigned URLs point at the same private VIP, so browsers cannot use them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import Settings, settings as _settings

CACHE = Path.home() / ".vastkit" / "cache" / "s3"


def split_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://") or "/" not in uri[5:]:
        raise ValueError(f"not an s3://bucket/key URI: {uri!r}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def client(cfg: Settings | None = None):
    import boto3
    from botocore.config import Config
    cfg = cfg or _settings()
    if not (cfg.s3_endpoint and cfg.access_key and cfg.secret_key):
        raise RuntimeError("S3_ENDPOINT / ACCESS_KEY / SECRET_KEY unset")
    ep = cfg.s3_endpoint if cfg.s3_endpoint.startswith("http") else f"http://{cfg.s3_endpoint}"
    return boto3.client("s3", endpoint_url=ep, aws_access_key_id=cfg.access_key,
                        aws_secret_access_key=cfg.secret_key, verify=False,
                        config=Config(signature_version="s3v4", retries={"max_attempts": 3},
                                      s3={"addressing_style": "path"}))


def fetch(uri: str, *, cfg: Settings | None = None, cache: bool = True, s3: Any = None) -> Path:
    """Download (once) and return a local path. Cache key = sha256(uri)."""
    bucket, key = split_uri(uri)
    dst = CACHE / hashlib.sha256(uri.encode()).hexdigest()[:24] / Path(key).name
    if cache and dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    (s3 or client(cfg)).download_file(bucket, key, str(dst))
    return dst


def head(uri: str, *, cfg: Settings | None = None, s3: Any = None) -> dict[str, Any]:
    """Object metadata. Chunk objects carry ingest metadata (scenario, custom-prompt, camera-id...)."""
    bucket, key = split_uri(uri)
    r = (s3 or client(cfg)).head_object(Bucket=bucket, Key=key)
    return {"size": r.get("ContentLength"), "content_type": r.get("ContentType"), "metadata": r.get("Metadata", {})}


def load_sidecar(uri_or_path: str | Path, *, cfg: Settings | None = None) -> dict[str, Any]:
    """Detection sidecar JSON (s3://.../detections/<stem>.json.gz or a local file)."""
    p = Path(uri_or_path) if not str(uri_or_path).startswith("s3://") else fetch(str(uri_or_path), cfg=cfg)
    raw = p.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def list_keys(bucket: str, prefix: str = "", *, cfg: Settings | None = None, s3: Any = None,
              suffix: str | None = None, limit: int | None = None) -> list[str]:
    cl = s3 or client(cfg)
    out: list[str] = []
    for page in cl.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if suffix and not obj["Key"].endswith(suffix):
                continue
            out.append(obj["Key"])
            if limit and len(out) >= limit:
                return out
    return out
