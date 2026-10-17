"""Small shared helpers: secret scrubbing, a content-hash disk cache, a JSONL logger, timing."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

CACHE_ROOT = Path.home() / ".vastkit" / "cache"

# Key names to redact, with any prefix (s3_, vdb_, AWS_, x-) and an optional _key/_id suffix:
# SECRET_KEY, s3_access_key, gpu_bearer_token, X-Api-Key, X-Amz-Signature. Plurals (max_tokens) don't match.
_SECRET_KEY = (r"(?<![\w-])[\w-]*(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key"
               r"|signature|x-amz-[a-z-]+)(?:[_-]?(?:key|id))?")
# key (optionally quoted, also \"escaped\" as inside a JSON string) then = or :, then the value:
# an escaped, double- or single-quoted string, or a bare run (JSON null/true/false and objects are left alone)
_SECRET_KV = re.compile(
    r"(?i)(?P<k>" + _SECRET_KEY + r"(?P<kq>\\?[\"'])?[ \t]*(?P<sep>[:=])[ \t]*)"
    r"(?P<v>\\\"[^\"]*?\\\"|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"
    r"|(?!(?:null|true|false|none)\b)(?:\\\\|[^\s\"'&,;{}\[\]\\])+)")
_SECRET_PATTERNS = [
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), "Bearer <redacted>"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "<jwt>"),
    (re.compile(r"\b(?:sk|wandb|hf)_[A-Za-z0-9]{16,}\b"), "<key>"),
]
_URL = re.compile(r"\b(?:https?|s3)://[^\s\"'<>]+")


def _redact_kv(m: re.Match) -> str:
    v = m["v"]
    if v.startswith('\\"'):
        q = '\\"'
    elif v[0] in "\"'":
        q = v[0]
    else:  # bare value: quote it like a quoted key so JSON / reprs stay parseable
        q = (m["kq"] or "") if m["sep"] == ":" else ""
    return f"{m['k']}{q}<redacted>{q}"


def scrub(text: Any, *, urls: bool = False) -> str:
    """Redact secret-named values (key=v, "key": "v", 'key': 'v'), bearer tokens, JWTs and API keys
    (and optionally every URL) from text before printing or logging."""
    s = _SECRET_KV.sub(_redact_kv, str(text))
    for pat, rep in _SECRET_PATTERNS:
        s = pat.sub(rep, s)
    if urls:
        s = _URL.sub("<url>", s)
    return s


def content_key(*parts: Any) -> str:
    """sha256 over args; bytes are hashed raw and existing file paths by their content (other paths by name)."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        elif (isinstance(p, Path) or (isinstance(p, str) and len(p) < 400)) and Path(p).is_file():
            h.update(Path(p).read_bytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
        h.update(b"\x00")
    return h.hexdigest()


def disk_cache(namespace: str, *, ttl: float | None = None) -> Callable:
    """Decorator: memoise a pure-ish function on disk by content hash of its arguments.

        @disk_cache("captions")
        def caption(clip_path, prompt): ...
    Call with ``_refresh=True`` to bypass and overwrite the entry.
    """
    root = CACHE_ROOT / namespace

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, _refresh: bool = False, **kw: Any) -> Any:
            # each arg separately, so file-path args are keyed by their content
            key = content_key(f"{fn.__module__}.{fn.__qualname__}", *args,
                              *[x for kv in sorted(kw.items()) for x in kv])
            f = root / f"{key[:32]}.pkl"
            if not _refresh and f.is_file() and (ttl is None or time.time() - f.stat().st_mtime < ttl):
                try:
                    return pickle.loads(f.read_bytes())
                except Exception:
                    pass
            out = fn(*args, **kw)
            data = pickle.dumps(out)
            root.mkdir(parents=True, exist_ok=True)
            # unique temp file per write: concurrent identical calls never share (or tear) one
            fd, tmp = tempfile.mkstemp(dir=root, prefix=f.stem + ".", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, f)
            except OSError:
                Path(tmp).unlink(missing_ok=True)
                if not f.is_file():  # no entry at all: a real I/O error
                    raise
                # else another writer won the race (or f is open elsewhere on Windows): keep it, out is still good
            return out
        wrapper.cache_dir = root  # type: ignore[attr-defined]
        return wrapper
    return deco


class JsonlLog:
    """Thread-safe append-only JSONL log; every record is scrubbed and timestamped."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        line = scrub(json.dumps(rec, default=str))
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return rec

    def read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]


@contextmanager
def timed(label: str = "", sink: dict[str, float] | None = None) -> Iterator[dict[str, float]]:
    """with timed("search") as t: ... ; t["s"] holds the elapsed seconds."""
    t = {"s": 0.0}
    t0 = time.perf_counter()
    try:
        yield t
    finally:
        t["s"] = time.perf_counter() - t0
        if sink is not None and label:
            sink[label] = sink.get(label, 0.0) + t["s"]
