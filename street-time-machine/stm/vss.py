"""Stdlib client for the team VSS backend ($INGRESS_URL/api/v1): browse cameras, pull sidecars, search captions.

Read-only: the login POST, GETs and the search POST. No uploads, no re-ingest, no third-party packages, so it runs on
the lab VM (python3 stdlib) as well as on the laptop. Shapes follow vast-hackathon/docs/API_NOTES.md.

Routes (all under $INGRESS_URL/api/v1):
  POST auth/login {username, password} -> access_token     GET videos/explore?scope&limit&offset -> chunks[]
  GET  tools/segments?original_video=   -> segments[]        GET videos/detections?source= -> YOLO sidecar
  GET  videos/metadata?source=          -> one segment row   GET videos/stream?source=&token= -> mp4 bytes
  POST search {query, top_k, min_similarity, metadata_filters, llm_top_n} -> results[], chunk_results[]
  GET  dashboard/stats?scope=

Config, first complete set wins: env VSS_URL/VSS_USERNAME/VSS_PASSWORD; then env INGRESS_URL/USERNAME/PASSWORD (Linux
only: on Windows USERNAME is the login name); then the single /config/*.config file (KEY=VALUE, parsed in memory).
The JWT lives only in memory. repr/str/log lines/error messages never carry the password, the username, the token or
a stream URL. 401 -> one re-login and retry; 404 -> None; transport errors, 429 and 5xx -> 3 retries with backoff.

Module-level functions (login, explore, segments, detections, segment_meta, stream, search, stats, cameras, pull)
use a lazily built default Client; tests install a fake one with set_client(Client(cfg, transport=fake)).

CLI:  python3 -m stm.vss cams [--scope mine]
      python3 -m stm.vss pull <cam> --dest DIR [--no-video] [--scope mine]
      python3 -m stm.vss search <cam> '<query>' [--top-k 10] [--min-sim 0.25]
      python3 -m stm.vss stats [--scope mine]
`pull` writes DIR/segments.json ({"chunks": [...], "segments": [...]}, object_counts kept as the row's JSON string,
the shape stm.captions reads), DIR/sidecars/<segment>.json (with segment_start_sec/segment_end_sec/chunk added, the
shape stm.packd reads) and DIR/chunks/<chunk>.mp4. Reruns skip sidecars and videos already on disk.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import http.client
import io
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

log = logging.getLogger("stm.vss")

CONFIG_GLOB = "/config/*.config"
SCOPES = ("all", "mine", "public")
RETRIES = 3
RETRY_STATUS = {429, 500, 502, 503, 504}
RETRYABLE = (OSError, http.client.HTTPException, json.JSONDecodeError)   # URLError and timeouts are OSErrors
DROP_KEYS = ("embedding", "embeddings", "vector", "vectors", "vectors_visual")
MAX_PAGES = 500


class VSSError(RuntimeError):
    """A VSS call that failed for good. The message names the route and status, never a URL or a secret."""

    def __init__(self, method: str, route: str, status: int | None, detail: str = ""):
        self.method, self.route, self.status, self.detail = method, route, status, detail
        super().__init__(f"VSS {method} {route} -> {status if status is not None else 'no response'}"
                         + (f": {detail}" if detail else ""))


class VSSConfigError(RuntimeError):
    """No usable VSS config was found (the message names where it looked, never a value)."""


class VSSAuthError(VSSError):
    """The login was rejected or returned no token."""


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class Config:
    url: str
    username: str
    password: str
    origin: str = ""

    @property
    def host(self) -> str:
        return urllib.parse.urlsplit(self.url).netloc or "?"

    def __repr__(self) -> str:
        return _blank(f"Config(origin={self.origin!r}, host={self.host!r}, username=<redacted>, password=<redacted>)",
                      self.username, self.password)

    __str__ = __repr__


def _blank(text: str, *secrets: str | None) -> str:
    """Replace each secret (raw, JSON-escaped and URL-quoted) with ***. A team username can be part of the host."""
    for secret in secrets:
        if secret:
            for form in (secret, json.dumps(secret)[1:-1], urllib.parse.quote(secret, safe="")):
                text = text.replace(form, "***")
    return text


def parse_config_text(text: str) -> dict[str, str]:
    """KEY=VALUE lines (optional `export `, quotes, # comments) -> dict. Values stay in memory only."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export "):].strip()
        out[k] = v.strip().strip('"').strip("'")
    return out


def _complete(d: dict[str, str], keys: tuple[str, str, str], origin: str) -> Config | None:
    vals = [str(d.get(k) or "").strip() for k in keys]
    return Config(*vals, origin=origin) if all(vals) else None


def load_config(env: dict[str, str] | None = None, platform: str | None = None,
                config_glob: str = CONFIG_GLOB) -> Config:
    """First complete source: env VSS_*, env INGRESS_URL/USERNAME/PASSWORD (Linux only), the one config file."""
    env = dict(os.environ) if env is None else env
    platform = sys.platform if platform is None else platform
    cfg = _complete(env, ("VSS_URL", "VSS_USERNAME", "VSS_PASSWORD"), "env:VSS_*")
    if cfg is None and platform.startswith("linux"):
        cfg = _complete(env, ("INGRESS_URL", "USERNAME", "PASSWORD"), "env:INGRESS_URL")
    if cfg is not None:
        return cfg
    files = sorted(glob.glob(config_glob))
    if len(files) != 1:
        raise VSSConfigError(f"no VSS config: set VSS_URL/VSS_USERNAME/VSS_PASSWORD, or provide exactly one "
                             f"{config_glob} file (found {len(files)})")
    vals = parse_config_text(Path(files[0]).read_text(encoding="utf-8"))
    cfg = (_complete(vals, ("INGRESS_URL", "USERNAME", "PASSWORD"), "file:" + Path(files[0]).name)
           or _complete(vals, ("VSS_URL", "VSS_USERNAME", "VSS_PASSWORD"), "file:" + Path(files[0]).name))
    if cfg is None:
        raise VSSConfigError(f"{Path(files[0]).name} lacks INGRESS_URL/USERNAME/PASSWORD")
    return cfg


# ---------------------------------------------------------------- transport

@dataclass
class Response:
    status: int
    headers: dict[str, str]          # lower-cased names
    body: BinaryIO

    def read(self, n: int = -1) -> bytes:
        return self.body.read(n)

    def json(self) -> Any:
        raw = self.body.read()
        return json.loads(raw) if raw else None

    def close(self) -> None:
        try:
            self.body.close()
        except Exception:  # noqa: BLE001 - closing a dead socket must not mask the real result
            pass


Transport = Callable[[str, str, dict[str, str], "bytes | None", float], Response]


def urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None,
                     timeout: float) -> Response:
    """Default transport. HTTP error statuses come back as a Response; network failures raise (OSError etc.)."""
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raw = e.read() if e.fp is not None else b""
        return Response(e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, io.BytesIO(raw))
    return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, r)


def _detail(resp: Response, limit: int = 200) -> str:
    try:
        raw = resp.read(4096)
    except RETRYABLE:
        return ""
    try:
        d = json.loads(raw)
        txt = d.get("detail") if isinstance(d, dict) else d
        txt = json.dumps(txt) if not isinstance(txt, str) else txt
    except (ValueError, TypeError):
        txt = raw.decode("utf-8", "replace")
    return " ".join(str(txt).split())[:limit]


def _content_range(value: str | None) -> tuple[int | None, int | None]:
    """'bytes a-b/N' or 'bytes */N' -> (a, N); unknown parts are None."""
    if not value or not value.startswith("bytes "):
        return None, None
    span, _, total = value[6:].partition("/")
    start = span.split("-", 1)[0]
    return (int(start) if start.isdigit() else None), (int(total) if total.isdigit() else None)


# ---------------------------------------------------------------- shaping

def parse_counts(v: Any) -> dict[str, int]:
    """object_counts as stored (JSON string, dict or empty) -> {label: int}; unparseable -> {}."""
    if isinstance(v, str):
        try:
            v = json.loads(v) if v.strip() else {}
        except ValueError:
            return {}
    if not isinstance(v, dict):
        return {}
    out = {}
    for k, n in v.items():
        try:
            out[str(k)] = int(n)
        except (TypeError, ValueError):
            continue
    return out


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k not in DROP_KEYS}
    if "object_counts" in out:
        out["object_counts"] = parse_counts(out["object_counts"])
    return out


def _hit(r: dict[str, Any]) -> dict[str, Any]:
    return {"source": r.get("source"), "similarity_score": float(r.get("similarity_score") or 0.0),
            "reasoning_content": r.get("reasoning_content") or "", "original_video": r.get("original_video"),
            "start_sec": r.get("segment_start_sec"), "end_sec": r.get("segment_end_sec"),
            "segment_number": r.get("segment_number"), "camera_id": r.get("camera_id"),
            "filename": r.get("filename")}


def hits_from(raw: dict[str, Any], camera_id: str | None = None, top_k: int = 100) -> list[dict[str, Any]]:
    """/search response -> segment hits, best first; a hit tagged with another camera is dropped."""
    hits = [_hit(r) for r in raw.get("results") or [] if isinstance(r, dict)]
    if camera_id is not None:
        hits = [h for h in hits if h["camera_id"] in (None, camera_id)]
    hits.sort(key=lambda h: -h["similarity_score"])
    return hits[:int(top_k)]


def stem(uri: str) -> str:
    """s3://bucket/a/b/name.mp4 -> name"""
    return str(uri).rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _check_scope(scope: str) -> None:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")


def _write_json(path: Path, obj: Any, **kw: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, default=str, **kw), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- client

class Client:
    """Thread-safe VSS client. Token in memory only; a 401 re-logs in once per stale token."""

    def __init__(self, config: Config | None = None, transport: Transport | None = None,
                 retries: int = RETRIES, timeout: float = 60.0, stream_timeout: float = 300.0,
                 sleep: Callable[[float], None] = time.sleep):
        self.config = config or load_config()
        self.transport = transport or urllib_transport
        self.retries, self.timeout, self.stream_timeout, self._sleep = retries, timeout, stream_timeout, sleep
        base = self.config.url.rstrip("/")
        self._base = base if base.endswith("/api/v1") else base + "/api/v1"
        self._token: str | None = None
        self._lock = threading.RLock()
        self.logins = 0

    def __repr__(self) -> str:
        return self._scrub(f"VSSClient(host={self.config.host!r}, origin={self.config.origin!r}, user=<redacted>, "
                           f"token={'<set>' if self._token else None})")

    __str__ = __repr__

    # -- plumbing
    def _url(self, route: str, params: dict[str, Any] | None = None) -> str:
        q = {k: v for k, v in (params or {}).items() if v is not None}
        return f"{self._base}/{route}" + ("?" + urllib.parse.urlencode(q) if q else "")

    def _scrub(self, text: str) -> str:
        """Defence in depth for reprs and server error details: blank the token, password and username."""
        return _blank(text, self._token, self.config.password, self.config.username)

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(4.0, 0.5 * 2 ** attempt))

    def login(self) -> None:
        """POST auth/login; keeps the access_token in memory. Bad credentials raise VSSAuthError at once."""
        body = json.dumps({"username": self.config.username, "password": self.config.password}).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        with self._lock:
            last = ""
            for attempt in range(self.retries + 1):
                try:
                    resp = self.transport("POST", self._url("auth/login"), headers, body, self.timeout)
                    try:
                        status = resp.status
                        data = resp.json() if 200 <= status < 300 else None
                        detail = "" if data is not None or status == 422 else self._scrub(_detail(resp))
                    finally:
                        resp.close()
                except RETRYABLE as e:
                    last = type(e).__name__
                    self._backoff(attempt)
                    continue
                if 200 <= status < 300:
                    tok = (data or {}).get("access_token") if isinstance(data, dict) else None
                    if not tok:
                        raise VSSAuthError("POST", "auth/login", status, "response had no access_token")
                    self._token, self.logins = str(tok), self.logins + 1
                    log.debug("login ok (%s)", self._scrub(self.config.origin))
                    return
                if status in RETRY_STATUS:
                    last = str(status)
                    self._backoff(attempt)
                    continue
                raise VSSAuthError("POST", "auth/login", status, detail)
            raise VSSAuthError("POST", "auth/login", None, f"gave up after {self.retries + 1} attempts ({last})")

    def _token_now(self) -> str:
        with self._lock:
            if not self._token:
                self.login()
            return self._token  # type: ignore[return-value]

    def _relogin(self, stale: str) -> None:
        with self._lock:
            if self._token == stale:        # another thread may already have refreshed it
                self._token = None
                self.login()

    def call(self, method: str, route: str, params: dict[str, Any] | None = None,
             payload: Any = None) -> Any:
        """One JSON call with auth, retries and 401 re-login. 404 -> None. Other 4xx raise VSSError."""
        last = ""
        for attempt in range(self.retries + 1):
            tok = self._token_now()
            headers = {"Accept": "application/json", "Authorization": "Bearer " + tok}
            body = None
            if payload is not None:
                body, headers["Content-Type"] = json.dumps(payload).encode(), "application/json"
            t0 = time.perf_counter()
            try:
                resp = self.transport(method, self._url(route, params), headers, body, self.timeout)
                try:
                    status = resp.status
                    data = resp.json() if 200 <= status < 300 else None
                    detail = "" if 200 <= status < 300 or status in (401, 404) else self._scrub(_detail(resp))
                finally:
                    resp.close()
            except RETRYABLE as e:
                last = type(e).__name__
                log.debug("%s %s failed (%s), attempt %d", method, route, last, attempt + 1)
                self._backoff(attempt)
                continue
            log.debug("%s %s -> %d in %.0f ms", method, route, status, (time.perf_counter() - t0) * 1000)
            if 200 <= status < 300:
                return data
            if status == 401:
                last = "401"
                self._relogin(tok)
                continue
            if status == 404:
                return None
            if status in RETRY_STATUS:
                last = f"{status} {detail}".strip()
                self._backoff(attempt)
                continue
            raise VSSError(method, route, status, detail)
        raise VSSError(method, route, None, f"gave up after {self.retries + 1} attempts ({last})")

    # -- API
    def explore(self, camera_id: str | None = None, location: str | None = None, scope: str = "all",
                indexed: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Every chunk visible in scope (paged by `limit`), deduped by original_video, sorted by it.

        The server ignores camera_id, so it is filtered here; location (and indexed) are sent to the server.
        """
        _check_scope(scope)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        off = 0
        for _ in range(MAX_PAGES):
            d = self.call("GET", "videos/explore", {"scope": scope, "location": location, "indexed": indexed,
                                                    "limit": limit, "offset": off}) or {}
            got = d.get("chunks") or []
            for c in got:
                key = str(c.get("original_video") or c.get("filename") or "")
                if key not in seen:
                    seen.add(key)
                    out.append(c)
            off += len(got)
            total = d.get("total")
            if not got or (isinstance(total, int) and off >= total):
                break
        if camera_id is not None:
            out = [c for c in out if c.get("camera_id") == camera_id]
        return sorted(out, key=lambda c: str(c.get("original_video") or c.get("filename") or ""))

    def cameras(self, scope: str = "all", location: str | None = None) -> list[dict[str, Any]]:
        """Per camera_id: chunk and segment (sum of total_segments) counts, locations, upload span."""
        cams: dict[str, dict[str, Any]] = {}
        for c in self.explore(scope=scope, location=location):
            cam = str(c.get("camera_id") or "(none)")
            r = cams.setdefault(cam, {"camera_id": cam, "chunks": 0, "segments": 0, "public_chunks": 0,
                                      "locations": set(), "first_upload": None, "last_upload": None})
            r["chunks"] += 1
            r["segments"] += int(c.get("total_segments") or 0)
            r["public_chunks"] += bool(c.get("is_public"))
            if c.get("location"):
                r["locations"].add(str(c["location"]))
            up = c.get("upload_timestamp")
            if up:
                r["first_upload"] = min(filter(None, [r["first_upload"], str(up)]))
                r["last_upload"] = max(filter(None, [r["last_upload"], str(up)]))
        return [dict(r, locations=sorted(r["locations"])) for _, r in sorted(cams.items())]

    def segments(self, original_video: str) -> list[dict[str, Any]]:
        """Segment rows of one chunk, sorted by segment_number, object_counts parsed to a dict ([] when none)."""
        d = self.call("GET", "tools/segments", {"original_video": original_video}) or {}
        rows = d.get("segments") if isinstance(d, dict) else d
        rows = [_clean_row(r) for r in (rows or []) if isinstance(r, dict)]
        return sorted(rows, key=lambda r: (int(r.get("segment_number") or 0), str(r.get("source") or "")))

    def detections(self, source: str) -> dict[str, Any] | None:
        """The YOLO sidecar of one segment (frames[].time_sec is relative to the segment), None if absent."""
        return self.call("GET", "videos/detections", {"source": source})

    def segment_meta(self, source: str) -> dict[str, Any] | None:
        """The full row of one segment (object_counts parsed), None if absent or not visible."""
        d = self.call("GET", "videos/metadata", {"source": source})
        return _clean_row(d) if isinstance(d, dict) else d

    def stats(self, scope: str = "all") -> dict[str, Any] | None:
        _check_scope(scope)
        return self.call("GET", "dashboard/stats", {"scope": scope})

    def search_raw(self, q: str, camera_id: str | None = None, top_k: int = 10, min_similarity: float = 0.25,
                   location: str | None = None, capture_type: str | None = None,
                   llm_top_n: int | None = 1) -> dict[str, Any]:
        """POST search; the whole response minus sql_query (it embeds the query vector)."""
        if not str(q).strip():
            raise ValueError("empty query")
        if not 1 <= int(top_k) <= 100:
            raise ValueError("top_k must be 1..100")
        if not 0.0 <= float(min_similarity) <= 1.0:
            raise ValueError("min_similarity must be 0..1")
        if llm_top_n is not None and int(llm_top_n) < 1:
            raise ValueError("llm_top_n must be >= 1 (0 is a 422)")
        filters = {k: v for k, v in (("camera_id", camera_id), ("location", location),
                                     ("capture_type", capture_type)) if v is not None}
        body: dict[str, Any] = {"query": str(q), "top_k": int(top_k), "min_similarity": float(min_similarity),
                                "metadata_filters": filters}
        if llm_top_n is not None:
            body["llm_top_n"] = int(llm_top_n)
        d = self.call("POST", "search", payload=body) or {}
        d.pop("sql_query", None)
        return d

    def search(self, q: str, camera_id: str | None = None, top_k: int = 10, min_similarity: float = 0.25,
               **kw: Any) -> list[dict[str, Any]]:
        """Segment hits, best first: {source, similarity_score, reasoning_content, original_video, start_sec,
        end_sec, segment_number, camera_id, filename}. A hit tagged with another camera is dropped."""
        d = self.search_raw(q, camera_id=camera_id, top_k=top_k, min_similarity=min_similarity, **kw)
        return hits_from(d, camera_id, top_k)

    def stream(self, source: str, dest: str | Path, force: bool = False) -> int | None:
        """Download one mp4 to dest (resumable via dest.part + Range). Returns bytes, None on 404.

        An existing dest is complete (it is only renamed into place when done) and is kept unless force.
        """
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0 and not force:
            return dest.stat().st_size
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        name, last = stem(source), ""
        for attempt in range(self.retries + 1):
            tok = self._token_now()
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                resp = self.transport("GET", self._url("videos/stream", {"source": source, "token": tok}),
                                      headers, None, self.stream_timeout)
            except RETRYABLE as e:
                last = type(e).__name__
                self._backoff(attempt)
                continue
            try:
                status = resp.status
                if status == 401:
                    last = "401"
                    self._relogin(tok)
                    continue
                if status == 404:
                    return None
                if status == 416:
                    _, total = _content_range(resp.headers.get("content-range"))
                    if total is not None and have == total:
                        os.replace(part, dest)
                        return total
                    part.unlink(missing_ok=True)
                    last = "416"
                    continue
                if status in (200, 206):
                    start, total = _content_range(resp.headers.get("content-range"))
                    if status == 206 and start != have:
                        part.unlink(missing_ok=True)
                        last = "206 at wrong offset"
                        continue
                    clen = resp.headers.get("content-length")
                    if total is None and clen and clen.isdigit():
                        total = int(clen) + (have if status == 206 else 0)
                    try:
                        with open(part, "ab" if status == 206 else "wb") as fh:
                            while True:
                                buf = resp.read(1 << 20)
                                if not buf:
                                    break
                                fh.write(buf)
                    except RETRYABLE as e:
                        last = f"{type(e).__name__} after {part.stat().st_size if part.exists() else 0} bytes"
                        self._backoff(attempt)
                        continue
                    size = part.stat().st_size
                    if total is not None and size != total:
                        last = f"short read {size}/{total}"
                        self._backoff(attempt)
                        continue
                    os.replace(part, dest)
                    log.debug("stream %s -> %d bytes", name, size)
                    return size
                if status in RETRY_STATUS:
                    last = str(status)
                    self._backoff(attempt)
                    continue
                raise VSSError("GET", "videos/stream", status, self._scrub(_detail(resp)))
            finally:
                resp.close()
        raise VSSError("GET", "videos/stream", None, f"{name}: gave up after {self.retries + 1} attempts ({last})")

    def pull(self, camera_id: str, dest: str | Path, video: bool = True, scope: str = "all",
             location: str | None = None, workers: int = 6,
             progress: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Mirror one camera to dest: segments.json, sidecars/<segment>.json, chunks/<chunk>.mp4 (if video).

        A camera with no visible chunks writes nothing (a typo must not overwrite a good mirror).
        """
        t0 = time.perf_counter()
        dest = Path(dest)
        chunks = [c for c in self.explore(camera_id=camera_id, location=location, scope=scope)
                  if c.get("original_video")]
        if chunks:
            (dest / "sidecars").mkdir(parents=True, exist_ok=True)

        def one(chunk: dict[str, Any]) -> tuple[str, list[dict[str, Any]], int, int | None]:
            src = str(chunk["original_video"])
            name, rows, fetched = stem(src), [], 0
            for s in self.segments(src):
                sid = str(s.get("source") or "")
                path = dest / "sidecars" / (stem(sid) + ".json")
                have = path.exists() and path.stat().st_size > 0
                if not have and sid:
                    side = self.detections(sid)
                    if side is not None:
                        side.update(segment_start_sec=s.get("segment_start_sec"),
                                    segment_end_sec=s.get("segment_end_sec"), chunk=name)
                        _write_json(path, side)
                        have, fetched = True, fetched + 1
                row = dict(s, object_counts=json.dumps(s.get("object_counts") or {}), chunk=name,
                           has_sidecar=have)
                rows.append(row)
            size = self.stream(src, dest / "chunks" / (name + ".mp4")) if video else None
            if progress:
                progress(f"{name} {len(rows)} segments {sum(r['has_sidecar'] for r in rows)} sidecars"
                         + (f" {size / 1e6:.1f} MB" if size else ""))
            return name, rows, fetched, size

        rows: list[dict[str, Any]] = []
        fetched, sizes = 0, []
        with cf.ThreadPoolExecutor(max(1, workers)) as ex:
            for _, got, n, size in ex.map(one, chunks):
                rows += got
                fetched += n
                sizes.append(size)
        if chunks:
            _write_json(dest / "segments.json", {"camera_id": camera_id, "scope": scope,
                                                 "chunks": [c["original_video"] for c in chunks], "segments": rows})
        with_side = sum(r["has_sidecar"] for r in rows)
        return {"camera_id": camera_id, "scope": scope, "chunks": len(chunks), "segments": len(rows),
                "expected_segments": sum(int(c.get("total_segments") or 0) for c in chunks),
                "sidecars": with_side, "sidecars_fetched": fetched, "missing_sidecars": len(rows) - with_side,
                "videos": sum(1 for s in sizes if s), "missing_videos": sum(1 for s in sizes if video and not s),
                "video_mb": round(sum(s or 0 for s in sizes) / 1e6, 1),
                "seconds": round(time.perf_counter() - t0, 1), "dest": str(dest)}


# ---------------------------------------------------------------- module-level API (default client)

_default: Client | None = None
_default_lock = threading.Lock()


def client() -> Client:
    """The default client, built from load_config() on first use."""
    global _default
    with _default_lock:
        if _default is None:
            _default = Client()
        return _default


def set_client(c: Client | None) -> Client | None:
    """Install a client (a fake one in tests); returns the previous one."""
    global _default
    with _default_lock:
        old, _default = _default, c
        return old


def login() -> None:
    client().login()


def explore(camera_id: str | None = None, location: str | None = None, scope: str = "all",
            **kw: Any) -> list[dict[str, Any]]:
    return client().explore(camera_id=camera_id, location=location, scope=scope, **kw)


def cameras(scope: str = "all", location: str | None = None) -> list[dict[str, Any]]:
    return client().cameras(scope=scope, location=location)


def segments(original_video: str) -> list[dict[str, Any]]:
    return client().segments(original_video)


def detections(source: str) -> dict[str, Any] | None:
    return client().detections(source)


def segment_meta(source: str) -> dict[str, Any] | None:
    return client().segment_meta(source)


def stream(source: str, dest: str | Path, force: bool = False) -> int | None:
    return client().stream(source, dest, force=force)


def search(q: str, camera_id: str | None = None, top_k: int = 10, min_similarity: float = 0.25,
           **kw: Any) -> list[dict[str, Any]]:
    return client().search(q, camera_id=camera_id, top_k=top_k, min_similarity=min_similarity, **kw)


def stats(scope: str = "all") -> dict[str, Any] | None:
    return client().stats(scope)


def pull(camera_id: str, dest: str | Path, video: bool = True, scope: str = "all", **kw: Any) -> dict[str, Any]:
    return client().pull(camera_id, dest, video=video, scope=scope, **kw)


# ---------------------------------------------------------------- CLI

def _short(text: str, n: int = 100) -> str:
    t = " ".join(str(text or "").split())
    return t if len(t) <= n else t[:n - 3] + "..."


def _cmd_cams(c: Client, a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    cams = c.cameras(scope=a.scope, location=a.location)
    ms = (time.perf_counter() - t0) * 1000
    if not a.json:
        print(f"{'camera_id':32} {'chunks':>6} {'segments':>8} {'public':>6}  location")
        for r in cams:
            print(f"{r['camera_id']:32} {r['chunks']:6d} {r['segments']:8d} {r['public_chunks']:6d}  "
                  f"{','.join(r['locations'])}")
        print(f"{len(cams)} cameras, {sum(r['chunks'] for r in cams)} chunks, "
              f"{sum(r['segments'] for r in cams)} segments (scope={a.scope}, {ms:.0f} ms)")
    return {"cameras": cams, "ms": round(ms, 1)}


def _cmd_pull(c: Client, a: argparse.Namespace) -> dict[str, Any]:
    out = c.pull(a.cam, a.dest, video=not a.no_video, scope=a.scope, location=a.location, workers=a.workers,
                 progress=None if a.json else (lambda m: print(m, flush=True)))
    if not a.json:
        print(f"pull {out['camera_id']} (scope={out['scope']}): {out['chunks']} chunks, {out['segments']} segments "
              f"(expected {out['expected_segments']}), {out['sidecars']} sidecars ({out['sidecars_fetched']} "
              f"fetched, {out['missing_sidecars']} missing), {out['videos']} videos {out['video_mb']} MB, "
              f"{out['seconds']} s -> {out['dest']}")
    if not out["chunks"]:
        raise ValueError(f"no chunks for camera {a.cam!r} (scope={a.scope}); see `python3 -m stm.vss cams`")
    return out


def _num(v: Any, nd: int = 0) -> str:
    return "?" if v is None else f"{float(v):.{nd}f}"


def _cmd_search(c: Client, a: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    c._token_now()                                   # log in first so the latency below is the search alone
    login_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    raw = c.search_raw(a.query, camera_id=a.cam, top_k=a.top_k, min_similarity=a.min_sim)
    ms = (time.perf_counter() - t1) * 1000
    syn = raw.get("llm_synthesis") if isinstance(raw.get("llm_synthesis"), dict) else {}
    out = {"query": a.query, "camera_id": a.cam, "latency_ms": round(ms, 1), "login_ms": round(login_ms, 1),
           "hits": hits_from(raw, a.cam, a.top_k),
           "server": {"embedding_time_ms": raw.get("embedding_time_ms"), "search_time_ms": raw.get("search_time_ms"),
                      "synthesis_s": syn.get("processing_time"), "synthesis_error": syn.get("error"),
                      "results": len(raw.get("results") or []), "chunk_results": len(raw.get("chunk_results") or [])}}
    if not a.json:
        s = out["server"]
        print(f"search {a.cam} {a.query!r}: {len(out['hits'])} hits in {ms:.0f} ms (login before it {login_ms:.0f} ms;"
              f" server embed {_num(s['embedding_time_ms'])} ms, search {_num(s['search_time_ms'])} ms, synthesis "
              f"{_num(s['synthesis_s'], 2)} s)")
        for i, h in enumerate(out["hits"], 1):
            print(f"{i:2d}. {h['similarity_score']:.4f}  {stem(h['original_video'] or '')} seg {h['segment_number']} "
                  f"({h['start_sec']}-{h['end_sec']} s)  {_short(h['reasoning_content'])}")
    return out


def _cmd_stats(c: Client, a: argparse.Namespace) -> dict[str, Any]:
    d = c.stats(a.scope) or {}
    keep = {k: d.get(k) for k in ("scope", "generated_at", "query_time_ms", "table_available", "overview", "quality")}
    if not a.json:
        print(json.dumps(keep, indent=1, default=str))
    return keep


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="print one JSON result instead of text")
    common.add_argument("-v", "--verbose", action="store_true", help="debug log of route/status/latency to stderr")
    ap = argparse.ArgumentParser(prog="python3 -m stm.vss", description="Read-only team VSS client")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cams", parents=[common], help="camera ids with chunk/segment counts")
    p.add_argument("--scope", default="all", choices=SCOPES)
    p.add_argument("--location", default=None)
    p = sub.add_parser("pull", parents=[common], help="mirror one camera's segment rows, sidecars and chunk videos")
    p.add_argument("cam")
    p.add_argument("--dest", required=True)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--scope", default="all", choices=SCOPES)
    p.add_argument("--location", default=None)
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("search", parents=[common], help="semantic search within one camera")
    p.add_argument("cam")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-sim", type=float, default=0.25)
    p = sub.add_parser("stats", parents=[common], help="dashboard overview counts")
    p.add_argument("--scope", default="all", choices=SCOPES)
    a = ap.parse_args(argv)
    if a.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    try:
        out = {"cams": _cmd_cams, "pull": _cmd_pull, "search": _cmd_search, "stats": _cmd_stats}[a.cmd](client(), a)
    except (VSSError, VSSConfigError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
