"""Client for the VSS (video search & summarization) backend REST API at $INGRESS_URL/api/v1.

Backend behaviour (read from the blueprint backend source as API reference) that this client accounts for:
- JWT lasts 1 day with no refresh route; a 401 triggers one re-login and a retry.
- /search returns 200 even when Cosmos synthesis fails: check ``synthesis_error()``.
- /agent/ask global mode can raise a plain-text 500 on the blueprint build; prefer
  ``search_and_answer``.
- Upload, reingest and delete are not idempotent: retried only on connect errors, never on a
  5xx or timeout (the server may already have acted). Reads and searches retry on 502/503/504.
- Never log /config output, /search ``sql_query`` (it embeds a 256-float vector), or
  stream URLs (they carry the token).
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

import httpx

from . import http as _http
from .config import Settings, settings as _settings

log = logging.getLogger("vastkit.vss")

# Client-side copies of backend checks (BE = source-code/retrieval/video-backend/src): BE/config.py
# allowed_video_extensions and BE/utils/stream_upload.py _STREAM_ID.
ALLOWED_EXT = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
STREAM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
TIME_FILTERS = {"all", "5m", "15m", "1h", "24h", "7d", "custom"}
SCOPES = {"all", "mine", "public"}
CUSTOM_PROMPT_MAX = 800
# Substrings of folder/file names of downloaded public sample footage; never push those into the shared VSS.
INTERNET_SAMPLE_HINTS = ("sample-videos", "sample_videos", "intel", "pexels", "pixabay", "youtube", "yt-dlp",
                         "vimeo")
# Short hints that are also parts of ordinary words (intelligence, IntelliJ): matched as a whole word only.
WORD_HINTS = {"intel"}
# data/ next to the vastkit package: in a source checkout that is the gitignored folder holding the mock corpus
# segments cut from public sample clips. With a wheel install this is site-packages/data, so a clone's data/
# (where scripts/build_mock_corpus.py writes) is NOT covered: set VSS_MOCK_CORPUS or install the clone editable.
REPO_DATA = Path(__file__).resolve().parents[1] / "data"


def jwt_claims(token: str) -> dict[str, Any]:
    """Decode a JWT payload without verifying it (to read exp/sub)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError):
        return {}


def synthesis_error(resp: dict[str, Any]) -> str | None:
    """Error string from a /search or /agent/* response whose Cosmos synthesis failed."""
    syn = resp.get("llm_synthesis") or (resp.get("evidence") or {}).get("llm_synthesis")
    if isinstance(syn, dict) and syn.get("error"):
        return str(syn["error"])
    answer = resp.get("answer") or ""
    # prefix of llm_synthesis "response" on a Cosmos failure in the blueprint backend (BE =
    # source-code/retrieval/video-backend/src: services/llm_service.py synthesize_search_results, api/v1/search.py);
    # the /agent/* routes copy that response into "answer" (BE/api/v1/agent.py)
    if isinstance(answer, str) and answer.startswith("Failed to generate AI synthesis"):
        return answer
    return None


def _under(path: Path, root: Path) -> bool:
    p, r = os.path.normcase(str(path)), os.path.normcase(str(root))
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:  # different drives
        return False


def _mock_corpus_dirs() -> list[Path]:
    """$VSS_MOCK_CORPUS if it is a folder, else the folders of the segments its JSON lists."""
    env = os.environ.get("VSS_MOCK_CORPUS", "").strip()
    corpus = Path(env).resolve() if env else None
    if corpus is None or not corpus.exists():
        return []
    if corpus.is_dir():
        return [corpus]
    try:
        rows = json.loads(corpus.read_text(encoding="utf-8"))
        return sorted({Path(r["_local_path"]).resolve().parent for r in rows
                       if isinstance(r, dict) and r.get("_local_path")})
    except (OSError, ValueError, TypeError):
        return [corpus.parent]  # unreadable corpus: refuse its whole folder


def _hint_in(part: str, hint: str) -> bool:
    low = part.lower()
    if hint in WORD_HINTS:
        return re.search(rf"(?<![a-z]){re.escape(hint)}(?![a-z])", low) is not None
    return hint in low


def _check_upload_path(path: Path) -> None:
    """PermissionError for sample-footage hints in the path and for repo data / mock corpus files."""
    p = path.resolve()
    hit = next((h for part in p.parts for h in INTERNET_SAMPLE_HINTS if _hint_in(part, h)), None)
    if hit:
        raise PermissionError(f"refusing to upload {p.name}: its path contains '{hit}' "
                              "(internet footage must not go into VSS)")
    for root in (REPO_DATA, *_mock_corpus_dirs()):
        if _under(p, root.resolve()):
            raise PermissionError(f"refusing to upload {p.name}: it is under {root} (repo data / mock corpus)")


class VSSClient:
    def __init__(self, cfg: Settings | None = None, *, base_url: str | None = None,
                 username: str | None = None, password: str | None = None,
                 proxy: str | None = None, timeout: float = 130.0):
        cfg = cfg or _settings()
        self.base_url = (base_url or cfg.vss_url).rstrip("/")
        if not self.base_url:
            raise ValueError("VSS base URL unset: set INGRESS_URL / VSS_BASE_URL, or VSS_MOCK=1")
        self.username = username or cfg.vss_username
        self._password = password or cfg.vss_password
        self._token: str | None = None
        self._exp: float = 0.0
        self.client = _http.make_client(self.base_url, proxy=proxy if proxy is not None else cfg.socks_proxy,
                                        timeout=timeout)

    # ---------------- auth ----------------
    def login(self) -> str:
        r = _http.request(self.client, "POST", "/api/v1/auth/login",
                          json={"username": self.username, "password": self._password})
        if r.status_code != 200:
            raise _http.APIError(r, "login")
        self._token = r.json()["access_token"]
        exp = jwt_claims(self._token).get("exp")
        self._exp = float(exp) if exp else time.time() + 23 * 3600
        return self._token

    @property
    def token(self) -> str:
        if not self._token or time.time() > self._exp - 300:
            self.login()
        assert self._token
        return self._token

    def _send(self, method: str, url: str, idempotent: bool, **kw: Any) -> httpx.Response:
        if idempotent:
            return _http.request(self.client, method, url, retry_on={502, 503, 504}, **kw)
        # After a 5xx or timeout the server may already have acted: retry only if nothing was sent.
        for attempt in (1, 2):
            try:
                return _http.request(self.client, method, url, retries=0, **kw)
            except httpx.ConnectError:
                log.warning("%s %s failed (ConnectError); retry %d/2", method, url, attempt)
                time.sleep(1.5 * attempt)
        return _http.request(self.client, method, url, retries=0, **kw)

    def _call(self, method: str, path: str, *, auth: bool = True, ok: tuple[int, ...] = (200,),
              idempotent: bool = True, **kw: Any) -> httpx.Response:
        headers = kw.pop("headers", {}) or {}
        for attempt in (0, 1):
            if auth:
                headers["Authorization"] = f"Bearer {self.token}"
            r = self._send(method, "/api/v1" + path, idempotent, headers=headers, **kw)
            if r.status_code == 401 and auth and attempt == 0:
                self._token = None  # expired or invalidated: log in once more
                continue
            if r.status_code not in ok:
                raise _http.APIError(r, f"{method} {path}")
            return r
        raise AssertionError("unreachable")

    def _json(self, method: str, path: str, **kw: Any) -> Any:
        return self._call(method, path, **kw).json()

    def me(self) -> dict[str, Any]:
        """Cheapest authenticated health check (/health is not routed via the ingress)."""
        return self._json("GET", "/auth/me")

    def config(self) -> dict[str, Any]:
        """Backend config. Contains internal hostnames: do not print or log it."""
        return self._json("GET", "/config")

    # ---------------- search & agent ----------------
    @staticmethod
    def _search_body(query: str, top_k: int = 15, *, tags: list[str] | None = None,
                     include_public: bool = True, public_only: bool = False,
                     system_prompt: str | None = None, time_filter: str = "all",
                     custom_start_date: str | None = None, custom_end_date: str | None = None,
                     metadata_filters: dict[str, Any] | None = None, min_similarity: float = 0.1,
                     llm_top_n: int = 3, hybrid_text_weight: float | None = None) -> dict[str, Any]:
        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be 1..100")
        if time_filter not in TIME_FILTERS:
            raise ValueError(f"time_filter must be one of {sorted(TIME_FILTERS)}")
        for t in tags or []:
            if "'" in t:
                raise ValueError("tags are interpolated into SQL unescaped; a quote causes a 500")
        for k, v in (metadata_filters or {}).items():
            if isinstance(v, (list, tuple, dict)):
                raise ValueError(f"metadata_filters[{k}] must be a scalar")
        body: dict[str, Any] = {
            "query": query, "top_k": top_k, "tags": tags or [], "include_public": include_public,
            "public_only": public_only, "time_filter": time_filter,
            "metadata_filters": metadata_filters or {}, "min_similarity": min_similarity,
            "llm_top_n": max(1, llm_top_n),
        }
        if system_prompt is not None:
            body["system_prompt"] = system_prompt
        if time_filter == "custom":
            body["custom_start_date"], body["custom_end_date"] = custom_start_date, custom_end_date
        if hybrid_text_weight is not None:
            body["hybrid_text_weight"] = min(1.0, max(0.0, hybrid_text_weight))
        return body

    def search(self, query: str, top_k: int = 15, **kw: Any) -> dict[str, Any]:
        """Hybrid text+visual search. Returns VideoSearchResponse; drop ``sql_query`` before logging."""
        return self._json("POST", "/search", json=self._search_body(query, top_k, **kw))

    def search_and_answer(self, query: str, top_k: int = 15, **kw: Any) -> dict[str, Any]:
        return self._json("POST", "/agent/search-and-answer", json=self._search_body(query, top_k, **kw))

    def ask(self, question: str, original_video: str | None = None, top_k: int = 10) -> dict[str, Any]:
        """/agent/ask. Global mode falls back to search-and-answer on the known plain-text 500."""
        body: dict[str, Any] = {"question": question, "top_k": top_k}
        if original_video:
            body["original_video"] = original_video
        try:
            return self._json("POST", "/agent/ask", json=body)
        except _http.APIError as e:
            if e.status_code == 500 and not original_video:
                log.info("agent/ask 500 in global mode; using search-and-answer")
                return self.search_and_answer(question, top_k=min(top_k, 100))
            raise

    # ---------------- per-video tools ----------------
    def segments(self, original_video: str) -> dict[str, Any]:
        """All captions of one parent video, sorted by segment_number."""
        return self._json("GET", "/tools/segments", params={"original_video": original_video})

    def segment(self, source: str) -> dict[str, Any] | None:
        try:
            return self._json("GET", "/tools/segment", params={"source": source})["segment"]
        except _http.APIError as e:
            if e.status_code == 404:  # also what a Query Engine outage looks like
                return None
            raise

    def metadata(self, source: str) -> dict[str, Any] | None:
        try:
            return self._json("GET", "/videos/metadata", params={"source": source})
        except _http.APIError as e:
            if e.status_code == 404:
                return None
            raise

    def detections(self, source: str) -> dict[str, Any] | None:
        try:
            return self._json("GET", "/videos/detections", params={"source": source})
        except _http.APIError as e:
            if e.status_code == 404:
                return None
            raise

    def synthesize(self, original_video: str, question: str | None = None, max_segments: int = 50,
                   system_prompt: str | None = None) -> dict[str, Any]:
        """Text-only summary over stored captions (Cosmos does not see the video here)."""
        body: dict[str, Any] = {"original_video": original_video, "max_segments": max_segments}
        if question:
            body["question"] = question
        if system_prompt:
            body["system_prompt"] = system_prompt
        return self._json("POST", "/videos/synthesize", json=body)

    # ---------------- browse ----------------
    def explore(self, scope: str = "all", *, date: str | None = None, location: str | None = None,
                indexed: str = "complete", limit: int = 100, offset: int = 0) -> dict[str, Any]:
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {sorted(SCOPES)}")
        params: dict[str, Any] = {"scope": scope, "indexed": indexed, "limit": limit, "offset": offset}
        if date:
            params["date"] = date
        if location:
            params["location"] = location
        return self._json("GET", "/videos/explore", params=params)

    def explore_all(self, scope: str = "all", indexed: str = "complete", **kw: Any) -> Iterator[dict[str, Any]]:
        """Yield every chunk, paging with limit=100. Rows are cached ~45 s per worker."""
        offset = 0
        while True:
            page = self.explore(scope, indexed=indexed, limit=100, offset=offset, **kw)
            chunks = page.get("chunks") or []
            yield from chunks
            offset += len(chunks)
            if not chunks or offset >= int(page.get("total") or 0):
                return

    def chunk(self, stream_id: str, chunk_index: int, scope: str = "all") -> dict[str, Any] | None:
        """Stored chunk_index is 0-based (upload chunk_number is 1-based)."""
        try:
            return self._json("GET", "/videos/chunk",
                              params={"stream_id": stream_id, "chunk_index": chunk_index, "scope": scope})
        except _http.APIError as e:
            if e.status_code == 404:
                return None
            raise

    def stream(self, source: str, byte_range: str | None = None) -> httpx.Response:
        """Fetch segment bytes server-side (token goes in the query string, so never log the URL)."""
        headers = {"Range": byte_range} if byte_range else {}
        return self._call("GET", "/videos/stream", auth=False, ok=(200, 206, 304),
                          params={"source": source, "token": self.token}, headers=headers)

    def download(self, source: str, dest: str | Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.stream(source).content)
        return dest

    # ---------------- ingest ----------------
    def upload(self, path: str | Path, *, is_public: bool = False, tags: list[str] | None = None,
               allowed_users: list[str] | None = None, scenario: str = "", custom_prompt: str = "",
               camera_id: str = "", capture_type: str = "", location: str = "",
               stream_id: str = "", chunk_number: int | None = None,
               max_upload_mb: float | None = None, i_own_this_video: bool = False) -> dict[str, Any]:
        """Multipart upload. Only footage you recorded yourself or that the organisers provided.

        Ownership is the caller's responsibility: pass ``i_own_this_video=True`` to confirm it. Even
        then PermissionError is raised for a resolved path where any folder or the file name contains
        an INTERNET_SAMPLE_HINTS entry (case-insensitive substring, except the WORD_HINTS entry 'intel',
        matched as a whole word only), for anything under ``REPO_DATA`` (the data/ folder of the source
        checkout vastkit runs from; with a wheel install a clone's data/, such as the corpus
        scripts/build_mock_corpus.py writes, is not covered), and for the mock corpus in $VSS_MOCK_CORPUS (that
        folder, or the folders of the segments its JSON lists; set it in this shell to guard a clone's corpus).
        Not retried on 5xx/timeouts (it would ingest the video twice).
        """
        path = Path(path)
        if not i_own_this_video:
            raise PermissionError("upload() needs i_own_this_video=True (own or organiser-provided footage only)")
        _check_upload_path(path)
        if path.suffix.lower() not in ALLOWED_EXT:
            raise ValueError(f"extension {path.suffix} not in {sorted(ALLOWED_EXT)}")
        if max_upload_mb is not None and path.stat().st_size > max_upload_mb * 1024 * 1024:
            raise ValueError(f"{path.name} exceeds {max_upload_mb} MB")
        if bool(stream_id) != (chunk_number is not None):
            raise ValueError("stream_id and chunk_number must be set together")
        if stream_id and not STREAM_ID_RE.match(stream_id):
            raise ValueError("stream_id must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
        if chunk_number is not None and chunk_number < 1:
            raise ValueError("chunk_number is 1-based")
        if len(custom_prompt.strip()) > CUSTOM_PROMPT_MAX:
            log.warning("custom_prompt is %d chars; the server truncates to %d",
                        len(custom_prompt.strip()), CUSTOM_PROMPT_MAX)
        data: dict[str, Any] = {
            "is_public": str(is_public).lower(), "tags": ",".join(tags or []),
            "allowed_users": ",".join(allowed_users or []), "scenario": scenario,
            "custom_prompt": custom_prompt, "camera_id": camera_id, "capture_type": capture_type,
            "location": location, "stream_id": stream_id,
        }
        if chunk_number is not None:
            data["chunk_number"] = str(chunk_number)
        mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
        with path.open("rb") as fh:
            return self._json("POST", "/videos/upload", idempotent=False, data=data,
                              files={"file": (path.name, fh, mime)})

    def reingest(self, *, original_video: str | None = None, stream_id: str | None = None,
                 custom_prompt: str, chunk_count: int = 1, scenario: str = "",
                 camera_id: str = "", capture_type: str = "", location: str = "") -> dict[str, Any]:
        """Re-run DataEngine on an indexed parent with your own prompt. Takes minutes; not retried on 5xx.

        The route shape comes from the kit docs (not in the blueprint snapshot): verify it live first.
        """
        if bool(original_video) == bool(stream_id):
            raise ValueError("pass exactly one of original_video / stream_id")
        if not custom_prompt.strip() and not scenario:
            raise ValueError("pass custom_prompt (or a server scenario value)")
        if len(custom_prompt) > CUSTOM_PROMPT_MAX:
            raise ValueError(f"custom_prompt must be <= {CUSTOM_PROMPT_MAX} chars")
        body: dict[str, Any] = {"chunk_count": chunk_count}
        body["original_video" if original_video else "stream_id"] = original_video or stream_id
        for k, v in {"scenario": scenario, "custom_prompt": custom_prompt, "camera_id": camera_id,
                     "capture_type": capture_type, "location": location}.items():
            if v:
                body[k] = v
        return self._json("POST", "/dashboard/reingest", idempotent=False, json=body)

    def reingest_status(self, job_id: str) -> dict[str, Any] | None:
        """None on 404: the job record may live in another worker or was lost on restart."""
        try:
            return self._json("GET", f"/dashboard/reingest/{job_id}")
        except _http.APIError as e:
            if e.status_code == 404:
                return None
            raise

    def wait_reingest(self, job_id: str, *, timeout: float = 900, poll: float = 4.0,
                      on_update: Any = None) -> dict[str, Any] | None:
        """Poll until status == 'completed' (or failed/error). 404s count as 'unknown', not done."""
        deadline = time.time() + timeout
        last: dict[str, Any] | None = None
        while time.time() < deadline:
            st = self.reingest_status(job_id)
            if st is not None:
                last = st
                if on_update:
                    on_update(st)
                if str(st.get("status", "")).lower() in {"completed", "failed", "error", "cancelled"}:
                    return st
            time.sleep(poll)
        return last

    def delete_video(self, original_video: str, *, confirm: bool = False) -> dict[str, Any]:
        """IRREVERSIBLE (S3 objects, segments, rows). Never expose as an agent tool. Not retried on 5xx."""
        if confirm is not True:
            raise PermissionError("delete_video requires confirm=True")
        return self._json("DELETE", "/videos", idempotent=False, params={"original_video": original_video})

    # ---------------- dashboard / metadata ----------------
    def dashboard_stats(self, scope: str = "all") -> dict[str, Any]:
        return self._json("GET", "/dashboard/stats", params={"scope": scope})

    def metadata_schema(self) -> dict[str, Any]:
        return self._json("GET", "/metadata/schema")

    def metadata_values(self, field: str, prefix: str = "", limit: int = 50) -> list[str]:
        return self._json("GET", "/metadata/values",
                          params={"field": field, "prefix": prefix, "limit": limit})["values"]

    def ingest_config(self) -> dict[str, Any]:
        """Public. Read scenario presets from here rather than hardcoding them."""
        return self._json("GET", "/metadata/ingest-config", auth=False)

    def suggestions(self, chips_only: bool = False) -> dict[str, Any]:
        return self._json("GET", "/suggestions", params={"chips_only": str(chips_only).lower()})

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "VSSClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def strip_for_log(resp: dict[str, Any]) -> dict[str, Any]:
    """Copy of a search response safe to log or put in a Weave trace (no SQL / vectors)."""
    out = {k: v for k, v in resp.items() if k != "sql_query"}
    return out
