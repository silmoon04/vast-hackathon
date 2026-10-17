"""Shared httpx plumbing: per-client SOCKS proxy, optional bearer, bounded retries.

Never set HTTPS_PROXY globally for the lab tunnel: it would also route W&B and PyPI.
Pass ``proxy=settings().socks_proxy`` (socks5h://127.0.0.1:1080) only to clients that
talk to private lab hosts. Loopback targets (local mirrors, mocks) always bypass it.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterable

import httpx

log = logging.getLogger("vastkit.http")

# httpx/openai log full request URLs at INFO, and VSS stream URLs carry ?token=. Quiet them
# unless the application already chose a level.
for _name in ("httpx", "httpcore", "httpx2", "httpcore2", "openai"):
    if logging.getLogger(_name).level == logging.NOTSET:
        logging.getLogger(_name).setLevel(logging.WARNING)

TRANSIENT = frozenset({408, 429, 500, 502, 503, 504})


PUBLIC_HOSTS = (".wandb.ai", "wandb.ai")
# Never tunnelled: local GPU mirrors and the mocks listen here, and socks5h would resolve
# 127.0.0.1 on the far side of the tunnel.
LOOPBACK_MOUNTS = ("all://127.0.0.1", "all://localhost", "all://[::1]")


def ensure_no_proxy(hosts: Iterable[str] = PUBLIC_HOSTS) -> None:
    """Add hosts to NO_PROXY/no_proxy so an HTTPS_PROXY/ALL_PROXY pointed at the lab tunnel
    can never carry W&B (or other public) traffic."""
    for var in ("NO_PROXY", "no_proxy"):
        cur = [x.strip() for x in os.environ.get(var, "").split(",") if x.strip()]
        os.environ[var] = ",".join(cur + [h for h in hosts if h not in cur])


def bearer(token: str | None) -> dict[str, str]:
    """Authorization header from a raw token or a 'Bearer x' string; {} when unset."""
    if not token:
        return {}
    token = token.strip()
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"
    return {"Authorization": token}


def make_client(base_url: str = "", *, proxy: str | None = None, token: str | None = None,
                timeout: float = 60.0, headers: dict[str, str] | None = None) -> httpx.Client:
    """httpx.Client; ``proxy`` applies to every host except loopback (LOOPBACK_MOUNTS)."""
    h = {"User-Agent": "vastkit/0.1"}
    h.update(bearer(token))
    h.update(headers or {})
    mounts = dict.fromkeys(LOOPBACK_MOUNTS) if proxy else None  # None = direct transport
    return httpx.Client(base_url=base_url.rstrip("/"), proxy=proxy or None, mounts=mounts, headers=h,
                        timeout=httpx.Timeout(timeout, connect=10.0), follow_redirects=False)


def request(client: httpx.Client, method: str, url: str, *, retries: int = 2,
            backoff: float = 1.5, retry_on: Iterable[int] = TRANSIENT, **kw: Any) -> httpx.Response:
    """client.request with retries on transient status codes and connection errors.

    Returns the last response (callers decide whether to raise). Bodies are never logged.
    """
    retry_on = frozenset(retry_on)
    attempt = 0
    while True:
        try:
            resp = client.request(method, url, **kw)
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
            if attempt >= retries:
                raise
            log.warning("%s %s failed (%s); retry %d/%d", method, url, type(exc).__name__,
                        attempt + 1, retries)
        else:
            if resp.status_code not in retry_on or attempt >= retries:
                return resp
            log.warning("%s %s -> %d; retry %d/%d", method, url, resp.status_code,
                        attempt + 1, retries)
        attempt += 1
        time.sleep(backoff * attempt)


def error_detail(resp: httpx.Response) -> str:
    """FastAPI {'detail': ...} when JSON, else the first 200 chars (some 500s are plain text)."""
    try:
        body = resp.json()
        if isinstance(body, dict) and "detail" in body:
            return str(body["detail"])
        return str(body)[:200]
    except ValueError:
        return resp.text[:200]


class APIError(RuntimeError):
    def __init__(self, resp: httpx.Response, what: str = ""):
        self.status_code = resp.status_code
        self.detail = error_detail(resp)
        path = resp.request.url.path if resp.request else ""
        super().__init__(f"{what or path}: HTTP {resp.status_code}: {self.detail}")
