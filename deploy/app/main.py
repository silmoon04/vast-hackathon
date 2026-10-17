"""Stdlib-only /app server for the team namespace (python:3.12-slim, no pip install needed).

Routes (the Ingress strips the /app prefix, so everything is served at /):
  GET  /health                 200 immediately, no upstream call (readiness probe)
  GET  / , /<static file>      index.html, app.js (+ style.css if added) from this directory (flat ConfigMap)
  GET|POST /api/v1/<route>     proxied to $VSS_URL/api/v1/<route> with a server-side JWT (login cached,
                               re-login on 401), only for the (method, route) pairs in READ_ROUTES.
                               WRITE_ROUTES also need ALLOW_WRITES=1 and an X-App-Token header equal to
                               APP_TOKEN. Everything else is 403: DELETE/PUT/PATCH, config, auth/login,
                               streaming/*, batch-sync/*, any route with a 'prefill' segment and any route
                               not listed.
  GET  /video?source=s3://...  segment bytes via /videos/stream, Range/206 passed through; token stays here.
                               Only buckets in VIDEO_BUCKETS; unset = 403 (/videos/stream has no ACL: it
                               streams any key the backend's S3 credentials can read).
  POST /jobs/api/v1/<route>    same guard; runs the proxied call in a thread -> {job_id}; GET /jobs/<id> polls.
                               Use for calls that can exceed the nginx 60 s upstream timeout. Only the newest
                               MAX_JOBS (256) jobs are kept in memory; older ids poll as 404.
Paths are percent-decoded and normalised before the guard, and the canonical path is what is forwarded.
A failed VSS login or an unreachable VSS is a 502 (for a job: status "error", http_status 502).

Env: PORT (8080), VSS_URL, VSS_USERNAME, VSS_PASSWORD, ALLOW_WRITES (0), APP_TOKEN (unset = writes stay off),
VIDEO_BUCKETS (comma-separated buckets /video may stream; unset = /video is off), UPSTREAM_TIMEOUT (300).
Logs one line per request: method, path without the query string (stream URLs carry the token), status.
"""
from __future__ import annotations

import base64
import hmac
import http.client
import json
import mimetypes
import os
import posixpath
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
VSS_URL = os.environ.get("VSS_URL", "").rstrip("/")
VSS_USERNAME = os.environ.get("VSS_USERNAME", "")
VSS_PASSWORD = os.environ.get("VSS_PASSWORD", "")
ALLOW_WRITES = os.environ.get("ALLOW_WRITES", "0") == "1"
APP_TOKEN = os.environ.get("APP_TOKEN", "")
VIDEO_BUCKETS = {b.strip() for b in os.environ.get("VIDEO_BUCKETS", "").split(",") if b.strip()}
TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "300"))
STATIC = {"index.html", "app.js", "style.css", "favicon.ico"}  # whichever exist
# (method, route under /api/v1/); "x/*" matches every route below x/. HEAD is checked as GET.
READ_ROUTES = frozenset({
    ("GET", "auth/me"), ("POST", "search"), ("GET", "tools/*"), ("POST", "tools/*"),
    ("POST", "agent/ask"), ("POST", "agent/search-and-answer"),
    ("GET", "videos/explore"), ("GET", "videos/chunk"), ("GET", "videos/metadata"), ("GET", "videos/detections"),
    ("POST", "videos/synthesize"), ("GET", "videos/stream"),
    ("GET", "dashboard/stats"), ("GET", "metadata/*"), ("GET", "suggestions"),
})
WRITE_ROUTES = frozenset({("POST", "videos/upload"), ("POST", "dashboard/reingest"), ("GET", "dashboard/reingest/*")})
# Never forwarded, whatever the lists say: /config and */prefill return connection details and raw S3 keys,
# streaming/batch-sync start captures into the team bucket, and logins stay server-side.
DENY_ROUTES = ("config", "auth/login", "streaming", "batch-sync")
DENY_SEGMENTS = {"prefill"}  # denied at any depth, e.g. tools/prefill even though tools/* is listed
MAX_JOBS = 256  # /jobs keeps only the newest results in memory; older ids then poll as 404
NO_JWT = {"videos/stream"}  # authenticates by the caller's ?token=; the server JWT is not attached
DRAIN_MAX = 16 << 20  # unread request bodies up to this size are discarded; larger ones close the connection
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
       "proxy-authenticate", "host", "content-length", "authorization", "cookie", "x-app-token"}
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._~/-]*")

_tok = {"jwt": "", "exp": 0.0}
_tok_lock = threading.Lock()
_jobs: dict[str, dict] = {}  # insertion order = age
_jobs_lock = threading.Lock()


def _login() -> str:
    body = json.dumps({"username": VSS_USERNAME, "password": VSS_PASSWORD}).encode()
    req = urllib.request.Request(f"{VSS_URL}/api/v1/auth/login", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        jwt = json.loads(r.read())["access_token"]
    try:
        payload = jwt.split(".")[1]
        exp = float(json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))["exp"])
    except Exception:
        exp = time.time() + 23 * 3600
    _tok.update(jwt=jwt, exp=exp)
    return jwt


def token(force: bool = False) -> str:
    with _tok_lock:
        if force or not _tok["jwt"] or time.time() > _tok["exp"] - 300:
            return _login()
        return _tok["jwt"]


def bad_gateway(why: str, e: BaseException) -> tuple[int, list, bytes]:
    """The 502 triple upstream() returns; names the exception type only."""
    return 502, [("Content-Type", "application/json")], json.dumps({"detail": f"{why}: {type(e).__name__}"}).encode()


def upstream(method: str, path_qs: str, body: bytes | None, headers: dict[str, str], *, auth: bool = True):
    """-> (status, headers list, body bytes). Retries once with a fresh JWT on 401; a failed login or
    transport error is a 502."""
    for attempt in (0, 1):
        h = dict(headers)
        if auth:
            try:
                h["Authorization"] = f"Bearer {token(force=attempt == 1)}"
            except Exception as e:  # login refused, unreachable or malformed
                return bad_gateway("upstream login failed", e)
        req = urllib.request.Request(f"{VSS_URL}{path_qs}", data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status, list(r.headers.items()), r.read()
        except urllib.error.HTTPError as e:
            if e.code == 401 and auth and attempt == 0:
                continue
            return e.code, list(e.headers.items()), e.read()
        except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, resets, truncated replies
            return bad_gateway("upstream unreachable", e)
    raise AssertionError("unreachable")


def canonical(path: str) -> str | None:
    """Percent-decoded path with '//' collapsed and dot segments resolved; None if anything ambiguous is left
    (double encoding, characters a route never uses). Upstream decodes once too, so guard and route agree."""
    p = posixpath.normpath(re.sub(r"/{2,}", "/", urllib.parse.unquote(path)))
    if not _SAFE_PATH.fullmatch(p) or any(s in p for s in ("%", "//", "/./", "/../")):
        return None
    return p


def _listed(method: str, route: str, rules: frozenset) -> bool:
    m = "GET" if method == "HEAD" else method
    parts = route.split("/")
    return (m, route) in rules or any((m, "/".join(parts[:i]) + "/*") in rules for i in range(1, len(parts)))


def guard(method: str, route: str, app_token: str) -> str | None:
    """None if `method /api/v1/<route>` (canonical) may be forwarded, else the 403 detail."""
    if any(route == d or route.startswith(d + "/") for d in DENY_ROUTES) or DENY_SEGMENTS & set(route.split("/")):
        return "not exposed through /app"
    if _listed(method, route, READ_ROUTES):
        return None
    if not _listed(method, route, WRITE_ROUTES):
        return "not exposed through /app"
    if not (ALLOW_WRITES and APP_TOKEN):
        return "writes disabled (set ALLOW_WRITES=1 and APP_TOKEN)"
    if not hmac.compare_digest(app_token.encode(), APP_TOKEN.encode()):
        return "missing or wrong X-App-Token"
    return None


def new_job() -> dict:
    """Register a running job, evicting the oldest beyond MAX_JOBS. Workers update the returned dict."""
    jid = uuid.uuid4().hex
    job = {"job_id": jid, "status": "running", "started": time.time()}
    with _jobs_lock:
        _jobs[jid] = job
        while len(_jobs) > MAX_JOBS:
            del _jobs[next(iter(_jobs))]
    return job


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "app/1"
    _consumed = False  # request body read (or forwarded) by this request

    def log_request(self, code="-", size="-") -> None:
        """Method, path without the query string (it can carry the token), status. Nothing else."""
        method, path = (self.command, self.path.split("?", 1)[0]) if self.command else ("-", "-")
        print(f"{method} {path} -> {int(code) if isinstance(code, int) else code}", flush=True)

    def log_message(self, fmt, *args) -> None:  # log_error & co: their args can echo raw request bytes
        pass

    def _send(self, status: int, body: bytes, ctype: str = "application/json", extra: list | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra or []:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj).encode())

    def _length(self) -> int | None:
        """Declared body size; None if Content-Length is malformed or negative."""
        v = (self.headers.get("Content-Length") or "0").strip()
        return int(v) if re.fullmatch(r"[0-9]+", v) else None

    def _body(self) -> bytes | None:
        self._consumed = True
        n = self._length() or 0
        return self.rfile.read(n) if n else None

    def _drain(self) -> None:
        """Discard an unread body so keep-alive never parses (or logs) it as the next request."""
        if self._consumed or self.close_connection:
            return
        n = self._length() or 0
        if n > DRAIN_MAX:
            self.close_connection = True
            return
        try:
            if n:
                self.rfile.read(n)
        except OSError:
            self.close_connection = True

    def _denied(self, route: str) -> bool:
        """Send the 403 and return True unless guard() lets this request through."""
        why = guard(self.command, route, self.headers.get("X-App-Token", ""))
        if why:
            self._json(403, {"detail": why})
        return why is not None

    def _fwd_headers(self) -> dict[str, str]:
        return {k: v for k, v in self.headers.items() if k.lower() not in HOP}

    def _relay(self, status: int, headers: list, body: bytes) -> None:
        self.send_response(status)
        for k, v in headers:
            if k.lower() not in HOP and k.lower() != "set-cookie":
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _route(self) -> None:
        raw, _, query = self.path.partition("?")
        path = canonical(raw)
        if path is None:
            return self._json(400, {"detail": "ambiguous path (double-encoded or unexpected characters)"})
        if path == "/health":
            return self._json(200, {"ok": True})
        if path.startswith("/api/v1/"):
            if not VSS_URL:
                return self._json(503, {"detail": "VSS_URL not configured"})
            route = path.removeprefix("/api/v1/")
            if self._denied(route):
                return
            return self._relay(*upstream(self.command, path + (f"?{query}" if query else ""), self._body(),
                                         self._fwd_headers(), auth=route not in NO_JWT))
        if path == "/video" and self.command in ("GET", "HEAD"):
            src = urllib.parse.parse_qs(query).get("source", [""])[0]
            bucket = src.removeprefix("s3://").split("/", 1)[0] if src.startswith("s3://") else ""
            if not bucket:
                return self._json(400, {"detail": "source must be s3://bucket/key"})
            if bucket not in VIDEO_BUCKETS:  # default-deny: anonymous callers must not read any S3 key
                return self._json(403, {"detail": "bucket not in VIDEO_BUCKETS" if VIDEO_BUCKETS
                                        else "VIDEO_BUCKETS unset: /video is off"})
            h = {k: v for k, v in self.headers.items() if k.lower() in ("range", "if-none-match")}
            try:
                qs = urllib.parse.urlencode({"source": src, "token": token()})
                st, hd, body = upstream("GET", f"/api/v1/videos/stream?{qs}", None, h, auth=False)
                if st == 401:  # token rotated: retry once
                    qs = urllib.parse.urlencode({"source": src, "token": token(force=True)})
                    st, hd, body = upstream("GET", f"/api/v1/videos/stream?{qs}", None, h, auth=False)
            except Exception as e:  # the login behind token() failed
                st, hd, body = bad_gateway("upstream login failed", e)
            return self._relay(st, hd, body)
        if path.startswith("/jobs/"):
            rest = path[len("/jobs/"):]
            if self.command == "GET" and "/" not in rest:
                job = _jobs.get(rest)
                return self._json(200, job) if job else self._json(404, {"detail": "no such job"})
            if not rest.startswith("api/v1/"):
                return self._json(400, {"detail": "jobs wrap /api/v1/* calls only"})
            route = rest.removeprefix("api/v1/")
            if self._denied(route):
                return
            target = "/" + rest + (f"?{query}" if query else "")
            body, hdrs, method = self._body(), self._fwd_headers(), self.command
            job = new_job()
            snapshot = dict(job)  # the 202 body, taken before the worker can update the job

            def work():  # always ends the job, so GET /jobs/<id> never polls "running" forever
                try:
                    st, _, out = upstream(method, target, body, hdrs, auth=route not in NO_JWT)
                    try:
                        res = json.loads(out)
                    except ValueError:
                        res = out.decode("utf-8", "replace")[:4000]
                except Exception as e:
                    print(f"job error on {method} {path}: {type(e).__name__}", flush=True)
                    st, res = 502, {"detail": f"upstream error: {type(e).__name__}"}
                job.update(status="done" if st < 400 else "error", http_status=st, result=res,
                           seconds=round(time.time() - job["started"], 2))  # harmless if already evicted
            threading.Thread(target=work, daemon=True).start()
            return self._json(202, snapshot)
        name = path.lstrip("/") or "index.html"
        if self.command in ("GET", "HEAD") and name in STATIC and (HERE / name).is_file():
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            return self._send(200, (HERE / name).read_bytes(), ctype, [("Cache-Control", "no-cache")])
        return self._json(404, {"detail": "not found"})

    def _safe(self):
        self._consumed = False
        try:
            if self.headers.get("Transfer-Encoding") or self._length() is None:
                self.close_connection = True  # body extent unknown: never parse it as the next request
                return self._json(400, {"detail": "body needs a valid Content-Length (chunked is not supported)"})
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as e:
            self.close_connection = True
            print(f"error on {self.command} {self.path.split('?', 1)[0]}: {type(e).__name__}", flush=True)
            try:
                self._json(500, {"detail": type(e).__name__})
            except Exception:
                pass
        finally:
            self._drain()

    do_GET = do_POST = do_DELETE = do_PUT = do_PATCH = do_HEAD = _safe


def main() -> None:
    port = int(os.environ.get("PORT", "8080"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    writes = ("on, X-App-Token required" if ALLOW_WRITES and APP_TOKEN
              else "off: ALLOW_WRITES=1 needs APP_TOKEN" if ALLOW_WRITES else "off")
    video = f"{len(VIDEO_BUCKETS)} bucket(s)" if VIDEO_BUCKETS else "off: VIDEO_BUCKETS unset"
    print(f"listening on :{port} (vss {'configured' if VSS_URL else 'NOT configured'}, writes {writes}, "
          f"video {video})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
