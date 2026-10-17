"""Offline stand-in for the VSS REST backend, for building clients without the lab.

SHAPE-ONLY: response fields, status codes and auth match the public vss-blueprint source
(d0e58da) and kit docs, but the search scoring is a naive word-overlap placeholder.
Ranking, similarity values and synthesis text are NOT representative of the real
backend; never tune anything against this mock.

Run:  python -m vastkit.mocks.vss_mock --port 8765        (then VSS_MOCK=1 for vastkit clients)

What matches the real API (see docs/API_NOTES.md):
- login returns an HS256 JWT {sub, exp=+1 day, iat, role:"user"}; bad password -> 401
- every /api/v1 route except login and metadata/ingest-config requires Bearer (or raw) JWT
- 401 bodies: "Could not validate credentials" / "Invalid token" / "Token has expired"
- /search: request validation ranges, response keys, chunk grouping, synthesis errors as 200
- /videos/stream: ?token= auth, Range/206/416, ETag/304
- upload: extension + size + stream_id checks, object_key keeps the extension's case
- status and error strings (HTTPException details, the upload message, the synthesis-error prefix, the
  app name) are copied from the blueprint files they mirror (BE = source-code/retrieval/video-backend/src:
  main.py, config.py, services/{auth,llm}_service.py, api/v1/{auth,agent,tools,videos,metadata,search}.py,
  utils/stream_upload.py), each cited next to its string; the reingest/job and "in mock" messages are
  the mock's own
- dashboard/reingest: GUESSED from kit docs (not in the blueprint snapshot); logs a warning

Fault injection (env): MOCK_LATENCY_MS, MOCK_SYNTH_FAIL=1, MOCK_ASK_BUG=1 (reproduce the
blueprint /agent/ask global-mode plain-text 500), MOCK_REINGEST_SECONDS (30),
MOCK_REINGEST_404 (0.25), MOCK_LLM_URL + MOCK_LLM_MODEL (OpenAI-compatible text model for
synthesis, e.g. the local Cosmos3-Edge mirror; default is a deterministic template).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

try:
    from .corpus import load_corpus, make_rows
except ImportError:  # run as a script
    from corpus import load_corpus, make_rows  # type: ignore

TEAM = os.environ.get("VSS_MOCK_TEAM", "team-mock")
PASSWORD = os.environ.get("VSS_MOCK_PASSWORD", "mock-password")
JWT_SECRET = os.environ.get("VSS_MOCK_JWT_SECRET", "local-mock-secret").encode()
MAX_UPLOAD_MB = float(os.environ.get("VSS_MOCK_MAX_UPLOAD_MB", "25"))
ALLOWED = [".mp4", ".mov", ".webm", ".avi", ".mkv"]
MAX_PROMPT = 800
STREAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
UPLOAD_DIR = Path(os.environ.get("VSS_MOCK_UPLOAD_DIR", Path.home() / ".vastkit" / "mock_uploads"))

app = FastAPI(title="vss-mock", version="1.0.0")
ROWS: List[Dict[str, Any]] = load_corpus(os.environ.get("VSS_MOCK_CORPUS"), TEAM)
JOBS: Dict[str, Dict[str, Any]] = {}
_counter = {"n": 0}


# ---------------------------------------------------------------- JWT (HS256, stdlib)
def _utcnow() -> datetime:
    """Naive UTC, like the backend's upload_timestamp."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def jwt_encode(claims: Dict[str, Any]) -> str:
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64(json.dumps(claims, separators=(",", ":")).encode())
    sig = _b64(hmac.new(JWT_SECRET, f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{sig}"


def jwt_decode(token: str) -> Dict[str, Any]:
    """401 details here and in current_user() mirror BE/services/auth_service.py."""
    try:
        head, body, sig = token.split(".")
        good = _b64(hmac.new(JWT_SECRET, f"{head}.{body}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            raise ValueError
        claims = json.loads(_unb64(body))
    except Exception:
        raise HTTPException(401, "Invalid token", headers={"WWW-Authenticate": "Bearer"})
    if claims.get("exp", 0) < time.time():
        raise HTTPException(401, "Token has expired", headers={"WWW-Authenticate": "Bearer"})
    return claims


def current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization:
        raise HTTPException(401, "Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    token = authorization[7:] if authorization.lower().startswith("bearer ") else authorization
    return jwt_decode(token.strip())


def _latency() -> None:
    ms = float(os.environ.get("MOCK_LATENCY_MS", "0") or 0)
    if ms:
        time.sleep(ms / 1000)


# ---------------------------------------------------------------- helpers
def _public(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _visible(row: Dict[str, Any], user: str, include_public: bool = True, public_only: bool = False) -> bool:
    if public_only:
        return bool(row.get("is_public"))
    mine = user in (row.get("allowed_users") or [])
    return mine or (include_public and bool(row.get("is_public")))


def _scope_ok(row: Dict[str, Any], user: str, scope: str) -> bool:
    if scope == "mine":
        return user in (row.get("allowed_users") or [])
    if scope == "public":
        return bool(row.get("is_public"))
    return _visible(row, user)


_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to and or is are was were with from for by into this that it its".split())


def _tokens(s: str) -> List[str]:
    return [t for t in _TOK.findall(s.lower()) if t not in _STOP]


def _score(query: str, text: str) -> float:
    """Placeholder: fraction of query words present in the text. Not the real embedding score."""
    q = set(_tokens(query))
    return round(len(q & set(_tokens(text))) / len(q), 4) if q else 0.0


def _by_video(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r["original_video"], r["segment_number"]))


def _time_ok(row: Dict[str, Any], tf: str, start: Optional[str], end: Optional[str]) -> bool:
    ts = datetime.fromisoformat(str(row["upload_timestamp"]).replace("Z", "").split("+")[0])
    now = _utcnow()
    presets = {"5m": 5, "15m": 15, "1h": 60, "24h": 1440, "7d": 10080}
    if tf in presets:
        return ts >= now - timedelta(minutes=presets[tf])
    if tf == "custom":
        def p(x: Optional[str]) -> Optional[datetime]:
            if not x:
                return None
            return datetime.fromisoformat(x.replace("Z", "").split("+")[0].split(".")[0])
        s, e = p(start), p(end)
        return (s is None or ts >= s) and (e is None or ts <= e)
    return True  # "all" and unknown values silently apply no filter (as in the source)


def _meta_ok(row: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    for k, v in filters.items():
        k = k.replace('"', "").replace("'", "").replace(";", "")
        if k == "object_classes":
            if str(v).lower() not in (row.get("object_classes") or "").lower():
                return False
        elif k not in row:  # "Search failed: " is how BE/api/v1/search.py wraps errors; the rest is the mock's
            raise HTTPException(500, f"Search failed: column {k} not found")
        elif row.get(k) != v:
            return False
    return True


def _chunk(original: str, segs: List[Dict[str, Any]], query: str, all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    best = max(segs, key=lambda r: r["similarity_score"])
    timeline_rows = _by_video([r for r in all_rows if r["original_video"] == original])
    matched = {r["segment_number"] for r in segs}
    timeline = [{
        "segment_number": r["segment_number"], "segment_start_sec": r["segment_start_sec"],
        "segment_end_sec": r["segment_end_sec"], "source": r["source"],
        "reasoning_content": r.get("reasoning_content"), "object_classes": r.get("object_classes"),
        "object_counts": r.get("object_counts"), "perception_ok": r.get("perception_ok"),
        "similarity_score": next((s["similarity_score"] for s in segs if s["segment_number"] == r["segment_number"]), 0),
        "is_search_match": r["segment_number"] in matched, "query_highlight": r["segment_number"] in matched,
        "is_best_match": r["segment_number"] == best["segment_number"],
    } for r in timeline_rows]
    em = json.loads(best.get("extra_metadata") or "{}")
    return {
        "original_video": original, "filename": best["filename"],
        "chunk_duration_sec": max((r["segment_end_sec"] for r in timeline_rows), default=0.0),
        "total_segments": best.get("total_segments"), "similarity_score": best["similarity_score"],
        "best_segment_number": best["segment_number"], "best_match_start_sec": best["segment_start_sec"],
        "best_match_end_sec": best["segment_end_sec"], "preview_source": best["source"],
        "reasoning_content": best.get("reasoning_content"), "is_public": best.get("is_public"),
        "upload_timestamp": best.get("upload_timestamp"), "tags": best.get("tags", []),
        "matched_segment_count": len(segs), "query": query, "timeline": timeline,
        "camera_id": best.get("camera_id"), "capture_type": best.get("capture_type"),
        "location": best.get("location"), "cosmos_model": best.get("cosmos_model"),
        "tokens_used": best.get("tokens_used"), "cached_prompt_tokens": best.get("cached_prompt_tokens"),
        "stream_id": em.get("stream_id"), "chunk_index": em.get("chunk_index"),
        "stream_chunk_total": None, "prev_chunk_index": None, "next_chunk_index": None,
    }


def _fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def _synthesize(question: str, segs: List[Dict[str, Any]], system_prompt: Optional[str]) -> Dict[str, Any]:
    t0 = time.time()
    used = [f"{s['filename']} (best moment {_fmt(s['segment_start_sec'])}–{_fmt(s['segment_end_sec'])})" for s in segs]
    if os.environ.get("MOCK_SYNTH_FAIL") == "1":
        err = "Cosmos request failed: 503 Service Unavailable (mock)"
        # response prefix mirrors BE/services/llm_service.py and BE/api/v1/search.py (vastkit.vss checks it)
        return {"response": f"Failed to generate AI synthesis: {err}", "segments_used": 0,
                "segments_analyzed": [], "model": "nvidia/cosmos3-reason", "tokens_used": 0,
                "processing_time": round(time.time() - t0, 3), "error": err}
    context = "\n".join(f"[{u}] {s.get('reasoning_content')}" for u, s in zip(used, segs))
    llm_url = os.environ.get("MOCK_LLM_URL")
    text, tokens = None, 0
    if llm_url:
        try:
            import httpx
            r = httpx.post(llm_url.rstrip("/") + "/v1/chat/completions", timeout=120, json={
                "model": os.environ.get("MOCK_LLM_MODEL", "local"), "temperature": 0.2, "max_tokens": 600,
                "messages": [{"role": "system", "content": system_prompt or "Answer from the video segment descriptions only. Cite timestamps."},
                             {"role": "user", "content": f"Question: {question}\n\nSegments:\n{context}"}]})
            r.raise_for_status()
            j = r.json()
            text, tokens = j["choices"][0]["message"]["content"], j.get("usage", {}).get("total_tokens", 0)
        except Exception as exc:  # fall back to the template, like a degraded backend
            text = None
            print(f"[vss-mock] MOCK_LLM_URL failed: {type(exc).__name__}")
    if text is None:
        bullets = "\n".join(f"- **{u}**: {s.get('reasoning_content')}" for u, s in zip(used, segs))
        text = f"**Answer (mock synthesis)** to: _{question}_\n\n{bullets}"
        tokens = 50 + 20 * len(segs)
    return {"response": text, "segments_used": len(segs), "segments_analyzed": used,
            "model": "nvidia/cosmos3-reason", "tokens_used": tokens,
            "processing_time": round(time.time() - t0, 3), "error": None}


# ---------------------------------------------------------------- root / health / auth
# "/" and /health bodies mirror BE/main.py; app_name/version are the BE/config.py defaults
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/")
def root() -> Dict[str, str]:
    return {"name": "Video Reasoning API", "version": "1.0.0", "status": "running"}


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/v1/auth/login")
def login(b: LoginRequest) -> Dict[str, Any]:
    if b.password != PASSWORD and os.environ.get("VSS_MOCK_ANY_PASSWORD") != "1":
        raise HTTPException(401, "Invalid username or password")  # BE/api/v1/auth.py
    now = int(time.time())
    ttl = int(os.environ.get("VSS_MOCK_TOKEN_TTL", str(24 * 3600)))
    tok = jwt_encode({"sub": b.username, "exp": now + ttl, "iat": now, "role": "user"})
    return {"access_token": tok, "token_type": "bearer", "username": b.username}


@app.get("/api/v1/auth/me")
def me(u: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    return {"username": u["sub"], "email": None, "auth_type": u.get("role", "user")}


@app.get("/api/v1/config")
def config(u: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    return {
        "s3": {"endpoint": "http://mock-data-vip", "access_key": "***", "secret_key": "***",
               "chunks_bucket": f"{TEAM}-vss-chunks", "segments_bucket": f"{TEAM}-vss-chunks-segments"},
        "vastdb": {"endpoint": "http://mock-qe-vip", "bucket": f"{TEAM}-vss-db", "schema": "vss-schema",
                   "collection": "vss-collection", "access_key": "***", "secret_key": "***"},
        "embedding": {"host": "mock-embed", "port": 8003, "model": "nvidia/cosmos-embed1", "dimensions": 256},
        "synthesis": {"cosmos_host": "mock-cosmos", "cosmos_port": 8001, "cosmoshttpscheme": "http",
                      "cosmos_model": "nvidia/cosmos3-reason", "cosmos_temperature": 0.2,
                      "synthesis_max_tokens": 2000, "synthesis_timeout_seconds": 120,
                      "default_system_prompt": "You are a video analysis assistant."},
        "app": {"app_name": "Video Reasoning API", "max_upload_size_mb": MAX_UPLOAD_MB,  # BE/config.py default
                "allowed_video_extensions": ALLOWED, "cors_origins": ["*"], "display_timezone": "UTC"},
    }


# ---------------------------------------------------------------- search
class VideoSearchRequest(BaseModel):
    query: str
    top_k: int = Field(15, ge=1, le=100)
    tags: List[str] = []
    include_public: bool = True
    public_only: bool = False
    system_prompt: Optional[str] = None
    time_filter: str = "all"
    custom_start_date: Optional[str] = None
    custom_end_date: Optional[str] = None
    metadata_filters: Dict[str, Any] = {}
    min_similarity: float = Field(0.1, ge=0.0, le=1.0)
    llm_top_n: int = Field(3, ge=1, le=100)  # BE/schemas/search.py: ge=1 (llm_top_n=0 -> 422)
    hybrid_text_weight: Optional[float] = None


def _run_search(b: VideoSearchRequest, user: str) -> Dict[str, Any]:
    _latency()
    t0 = time.time()
    for t in b.tags:
        if "'" in t:  # "Search failed: " is the detail prefix of BE/api/v1/search.py; the rest is simulated
            raise HTTPException(500, "Search failed: syntax error at or near \"'\" (unescaped tag)")
    pool, filtered = [], 0
    for r in ROWS:
        if not _visible(r, user, b.include_public, b.public_only):
            filtered += 1
            continue
        if b.tags and not set(b.tags) & set(r.get("tags") or []):
            continue
        if not _time_ok(r, b.time_filter, b.custom_start_date, b.custom_end_date):
            continue
        if not _meta_ok(r, b.metadata_filters):
            continue
        pool.append(r)
    emb_ms = 35.0
    scored = []
    for r in pool:  # placeholder scoring; hybrid_text_weight is accepted but ignored here
        sc = _score(b.query, f"{r.get('reasoning_content', '')} {r.get('object_classes', '')}")
        if sc >= b.min_similarity:
            scored.append(dict(_public(r), similarity_score=sc))
    scored.sort(key=lambda r: -r["similarity_score"])
    results = scored[:b.top_k]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in scored:
        groups[r["original_video"]].append(r)
    chunks = sorted((_chunk(ov, segs, b.query, ROWS) for ov, segs in groups.items()),
                    key=lambda c: -c["similarity_score"])[:b.top_k]
    synth = None
    if chunks:
        top = [next(r for r in scored if r["source"] == c["preview_source"]) for c in chunks[:b.llm_top_n]]
        synth = _synthesize(b.query, top, b.system_prompt)
    vec = ",".join(["0.0"] * 256)
    return {
        "results": results, "chunk_results": chunks, "total": len(results), "chunk_total": len(chunks),
        "query": b.query, "embedding_time_ms": emb_ms, "search_time_ms": round((time.time() - t0) * 1000, 2),
        "permission_filtered": filtered, "llm_synthesis": synth,
        "sql_query": f'SELECT ... array_cosine_distance(vectors::FLOAT[256], ARRAY[{vec}]::FLOAT[256]) ...',
    }


@app.post("/api/v1/search")
def search(b: VideoSearchRequest, u: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    return _run_search(b, u["sub"])


@app.post("/api/v1/tools/search")
def tools_search(b: VideoSearchRequest, u: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    return _run_search(b, u["sub"])


# tool_used values ("search_hybrid", "video_segments") as in BE/api/v1/agent.py agent_ask() / agent_search_and_answer()
@app.post("/api/v1/agent/search-and-answer")
def search_and_answer(b: VideoSearchRequest, u: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    s = _run_search(b, u["sub"])
    syn = s["llm_synthesis"]
    answer = (syn or {}).get("response") or \
        (s["chunk_results"][0]["reasoning_content"] if s["chunk_results"] else None) or \
        (s["results"][0]["reasoning_content"] if s["results"] else None) or "No matching segments found."
    return {"answer": answer, "tool_used": "search_hybrid", "evidence": {
        "query": b.query, "segment_hit_count": s["total"], "chunk_count": s["chunk_total"],
        "permission_filtered": s["permission_filtered"], "embedding_time_ms": s["embedding_time_ms"],
        "search_time_ms": s["search_time_ms"],
        "chunks": [{k: c[k] for k in ("original_video", "filename", "similarity_score", "best_match_start_sec",
                                       "best_match_end_sec", "matched_segment_count", "preview_source")}
                   for c in s["chunk_results"][:10]],
        "top_segment_sources": [r["source"] for r in s["results"][:5]], "llm_synthesis": syn}}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    original_video: Optional[str] = None
    top_k: int = Field(10, ge=1, le=50)


@app.post("/api/v1/agent/ask")
def ask(b: AskRequest, u: Dict[str, Any] = Depends(current_user)):
    user = u["sub"]
    if b.original_video:
        segs = [r for r in _by_video(ROWS) if r["original_video"] == b.original_video and _visible(r, user)]
        if not segs:  # detail mirrors BE/api/v1/agent.py agent_ask()
            raise HTTPException(404, f"No accessible segments found for video: {b.original_video}")
        used = [dict(_public(r), similarity_score=1.0) for r in segs[:b.top_k]]  # first top_k, not ranked
        syn = _synthesize(b.question, used, None)
        return {"answer": syn["response"], "tool_used": "video_segments",
                "evidence": {"segment_count": len(segs), "segments_used": len(used)}}
    s = _run_search(VideoSearchRequest(query=b.question, top_k=b.top_k), user)
    if s["results"] and os.environ.get("MOCK_ASK_BUG") == "1":
        return PlainTextResponse("Internal Server Error", status_code=500)
    answer = (s["llm_synthesis"] or {}).get("response") or "No matching segments found."
    return {"answer": answer, "tool_used": "search_hybrid",
            "evidence": {"result_count": s["total"], "top_sources": [r["source"] for r in s["results"][:5]]}}


# ---------------------------------------------------------------- tools / videos
def _rows_for(original: str, user: str) -> List[Dict[str, Any]]:
    return [r for r in _by_video(ROWS) if r["original_video"] == original and _visible(r, user)]


def _row_by_source(source: str, user: str) -> Optional[Dict[str, Any]]:
    return next((r for r in ROWS if r["source"] == source and _visible(r, user)), None)


@app.get("/api/v1/tools/segments")
def tools_segments(original_video: str = Query(...), u: Dict[str, Any] = Depends(current_user)):
    segs = [_public(r) for r in _rows_for(original_video, u["sub"])]
    return {"original_video": original_video, "total": len(segs), "segments": segs}


@app.get("/api/v1/tools/segment")
def tools_segment(source: str = Query(...), u: Dict[str, Any] = Depends(current_user)):
    r = _row_by_source(source, u["sub"])
    if not r:
        raise HTTPException(404, "Segment not found")  # BE/api/v1/tools.py
    return {"segment": _public(r)}


@app.get("/api/v1/videos/metadata")
def video_metadata(source: str = Query(...), u: Dict[str, Any] = Depends(current_user)):
    r = _row_by_source(source, u["sub"])
    if not r:  # detail mirrors BE/api/v1/videos.py get_video_metadata()
        raise HTTPException(404, "Video not found or you don't have permission to access it")
    return dict(_public(r), similarity_score=0.0)


def _sidecar(r: Dict[str, Any]) -> Dict[str, Any]:
    counts = json.loads(r.get("object_counts") or "{}")
    labels = [c for c in counts]
    frames = []
    n = 10
    for i in range(n):
        dets = [{"label": lab, "confidence": round(0.55 + 0.04 * ((i + j) % 9), 4),
                 "bbox": [100 + 40 * j + 5 * i, 200 + 3 * i, 180 + 40 * j + 5 * i, 420 + 3 * i]}
                for j, lab in enumerate(labels)]
        for d in dets:
            d["xyxy"] = list(d["bbox"])
        frames.append({"frame_index": i * 15, "time_sec": round(i * 15 / 30.0, 3), "shape": [720, 1280],
                       "detections": dets})
    # "yolo11_coco": the default source in source-code/ingest/video-detector/common/clients.py (sidecar payload)
    return {"source": "yolo11_coco", "segment_source": r["source"], "video_shape": [720, 1280], "fps": 30.0,
            "frame_count": r.get("detection_frame_count", 150), "detection_count": sum(len(f["detections"]) for f in frames),
            "object_classes": sorted(labels), "object_counts": counts,
            "max_detection_conf": r.get("max_detection_conf"), "frames": frames}


@app.get("/api/v1/videos/detections")
def video_detections(source: str = Query(...), u: Dict[str, Any] = Depends(current_user)):
    r = _row_by_source(source, u["sub"])
    if not r:  # details mirror BE/api/v1/videos.py, as do the chunk, stream, playback and delete routes below
        raise HTTPException(404, "Segment not found")
    if not r.get("detection_sidecar_uri"):
        raise HTTPException(404, "No detection sidecar for segment")
    return _sidecar(r)


@app.get("/api/v1/tools/detections")
def tools_detections(source: str = Query(...), u: Dict[str, Any] = Depends(current_user)):
    return video_detections(source, u)


class VideoSynthesizeRequest(BaseModel):
    original_video: str = Field(..., min_length=1)
    # mirrors the blueprint BE/schemas/explore.py VideoSynthesizeRequest default question (reworded, not copied)
    question: str = Field("Give a chronological summary of the video.", min_length=1)
    max_segments: int = Field(50, ge=1, le=100)
    system_prompt: Optional[str] = None


def _do_synth(b: VideoSynthesizeRequest, user: str) -> Dict[str, Any]:
    segs = _rows_for(b.original_video, user)
    if not segs:  # detail mirrors BE/api/v1/videos.py synthesize_video_chunk()
        raise HTTPException(404, f"No accessible segments found for video: {b.original_video}")
    used = segs[:b.max_segments]
    syn = _synthesize(b.question, used, b.system_prompt)
    return {"original_video": b.original_video, "segment_count": len(segs), "segments_used": len(used),
            "answer": syn["response"], "llm_synthesis": syn,
            "generated_at": datetime.now(timezone.utc).isoformat()}


@app.post("/api/v1/videos/synthesize")
def videos_synthesize(b: VideoSynthesizeRequest, u: Dict[str, Any] = Depends(current_user)):
    return _do_synth(b, u["sub"])


@app.post("/api/v1/tools/synthesize")
def tools_synthesize(b: VideoSynthesizeRequest, u: Dict[str, Any] = Depends(current_user)):
    return _do_synth(b, u["sub"])


def _complete(original: str) -> bool:
    rows = [r for r in ROWS if r["original_video"] == original]
    if not rows:
        return False
    total = max(int(r.get("total_segments") or 0) for r in rows)
    have = {r["segment_number"] for r in rows if r.get("reasoning_content")}
    return all(i in have for i in range(1, total + 1))


def _explore(user: str, scope: str, date: Optional[str], location: Optional[str], indexed: str,
             limit: int, offset: int) -> Dict[str, Any]:
    rows = [r for r in _by_video(ROWS) if _scope_ok(r, user, scope)]
    parents: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        parents[r["original_video"]].append(r)
    by_day: Dict[str, int] = defaultdict(int)
    locs: Dict[str, int] = defaultdict(int)
    chunks = []
    for ov, segs in parents.items():
        day = str(segs[0]["upload_timestamp"])[:10]
        loc = segs[0].get("location") or "(empty)"
        if (indexed == "complete") != _complete(ov):
            continue
        by_day[day] += 1
        locs[loc] += 1
        if date and day != date:
            continue
        if location and loc != location:
            continue
        c = _chunk(ov, [dict(s, similarity_score=0.0) for s in segs[:1]], "", ROWS)
        chunks.append(c)
    chunks.sort(key=lambda c: str(c["upload_timestamp"]), reverse=True)
    return {"chunks": chunks[offset:offset + limit], "total": len(chunks),
            "uploads_by_day": [{"date": d, "chunk_count": n} for d, n in sorted(by_day.items(), reverse=True)],
            "locations": [{"label": k, "chunk_count": n} for k, n in sorted(locs.items())],
            "scope": scope, "selected_date": date, "selected_location": location, "indexed": indexed,
            "limit": limit, "offset": offset, "table_available": True, "table_message": None}


@app.get("/api/v1/videos/explore")
def explore(scope: str = Query("all", pattern="^(all|mine|public)$"), date: Optional[str] = None,
            location: Optional[str] = None, indexed: str = Query("complete", pattern="^(complete|partial)$"),
            limit: int = Query(48, ge=1, le=100), offset: int = Query(0, ge=0),
            u: Dict[str, Any] = Depends(current_user)):
    _latency()
    return _explore(u["sub"], scope, date, location, indexed, limit, offset)


@app.get("/api/v1/tools/explore")
def tools_explore(scope: str = Query("all", pattern="^(all|mine|public)$"), date: Optional[str] = None,
                  location: Optional[str] = None, limit: int = Query(48, ge=1, le=100),
                  offset: int = Query(0, ge=0), u: Dict[str, Any] = Depends(current_user)):
    return _explore(u["sub"], scope, date, location, "complete", limit, offset)


@app.get("/api/v1/videos/chunk")
def video_chunk(stream_id: str = Query(..., min_length=1), chunk_index: int = Query(..., ge=0),
                scope: str = Query("all", pattern="^(all|mine|public)$"), u: Dict[str, Any] = Depends(current_user)):
    for r in _by_video(ROWS):
        em = json.loads(r.get("extra_metadata") or "{}")
        if em.get("stream_id") == stream_id and em.get("chunk_index") == chunk_index and _scope_ok(r, u["sub"], scope):
            return _chunk(r["original_video"], [dict(r, similarity_score=0.0)], "", ROWS)
    raise HTTPException(404, "Stream chunk not found")


def _stream_auth(token: str) -> Dict[str, Any]:
    try:
        return jwt_decode(token)
    except HTTPException:
        raise HTTPException(401, "Invalid or expired token")


def _s3_parts(source: str) -> tuple:
    if not source.startswith("s3://"):
        raise HTTPException(400, "Invalid S3 source URL format")
    rest = source[5:]
    if "/" not in rest or rest.endswith("/"):
        raise HTTPException(400, "Invalid S3 URL format. Expected: s3://bucket/key")
    return tuple(rest.split("/", 1))


@app.get("/api/v1/videos/stream")
def stream(request: Request, source: str = Query(...), token: str = Query(...)):
    _stream_auth(token)  # NB: the real route has no ACL check on source
    _s3_parts(source)
    row = next((r for r in ROWS if r["source"] == source or r["original_video"] == source), None)
    path = Path(row["_local_path"]) if row and row.get("_local_path") else None
    if not path or not path.is_file():
        raise HTTPException(404, "Object not found in mock (no _local_path for this source)")
    data_len = path.stat().st_size
    etag = '"' + hashlib.md5(f"{path}:{data_len}:{path.stat().st_mtime}".encode()).hexdigest() + '"'
    base = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=3600",
            "Content-Disposition": "inline", "ETag": etag, "Content-Type": "video/mp4"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=base)
    rng = request.headers.get("range")
    if rng:
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if not m or (not m.group(1) and not m.group(2)):
            return Response(status_code=416, headers={**base, "Content-Range": f"bytes */{data_len}"})
        if m.group(1):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else data_len - 1
        else:
            start, end = max(0, data_len - int(m.group(2))), data_len - 1
        end = min(end, data_len - 1)
        if start > end or start >= data_len:
            return Response(status_code=416, headers={**base, "Content-Range": f"bytes */{data_len}"})
        with path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(end - start + 1)
        return Response(chunk, status_code=206, headers={**base, "Content-Range": f"bytes {start}-{end}/{data_len}",
                                                          "Content-Length": str(len(chunk))})
    return Response(path.read_bytes(), status_code=200, headers={**base, "Content-Length": str(data_len)})


@app.get("/api/v1/videos/playback-url")
def playback_url(source: str = Query(...), token: str = Query(...), expires_in: int = Query(3600, ge=60, le=86400)):
    _stream_auth(token)
    b, k = _s3_parts(source)
    return {"url": f"http://mock-data-vip/{b}/{k}?X-Amz-Expires={expires_in}&X-Amz-Signature=mock",
            "expires_in": expires_in}


@app.post("/api/v1/videos/upload")
async def upload(file: UploadFile = File(...), is_public: bool = Form(False), tags: str = Form(""),
                 allowed_users: str = Form(""), scenario: str = Form(""), custom_prompt: str = Form(""),
                 camera_id: str = Form(""), capture_type: str = Form(""), location: str = Form(""),
                 stream_id: str = Form(""), chunk_number: Optional[int] = Form(None),
                 u: Dict[str, Any] = Depends(current_user)):
    name = file.filename or "upload.mp4"
    ext = Path(name).suffix
    # details and the success message mirror BE/api/v1/videos.py upload_video(); the chunk/stream_id
    # checks mirror BE/utils/stream_upload.py stream_s3_metadata()
    if ext.lower() not in ALLOWED:
        raise HTTPException(400, "Invalid file extension. Allowed: " + ", ".join(ALLOWED))
    if chunk_number is not None and not stream_id:
        raise HTTPException(400, "stream_id is required when chunk_number is set")
    if stream_id and chunk_number is None:
        raise HTTPException(400, "chunk_number is required when stream_id is set")
    if chunk_number is not None and chunk_number < 1:
        raise HTTPException(400, "chunk_number starts at 1")
    if stream_id and not STREAM_RE.match(stream_id):
        raise HTTPException(400, "stream_id must start with a letter or number and contain only letters, "
                                 "numbers, dots, underscores, or hyphens")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max size: {int(MAX_UPLOAD_MB)}MB")
    user = u["sub"]
    custom_prompt = custom_prompt.strip()[:MAX_PROMPT]
    now = _utcnow()
    key = f"{user}/{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}{ext}"  # extension case preserved
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    local = UPLOAD_DIR / key.replace("/", "__")
    local.write_bytes(data)
    n_seg = 4
    users = sorted({user, *[x.strip() for x in allowed_users.split(",") if x.strip()]})
    # segments are keyed by the stored object's basename (like the real segmenter), so re-uploads
    # of the same filename get their own segment sources / pks
    rows = make_rows(Path(key).name, [f"(mock) segment {i} of uploaded clip {name}" + (f" | prompt: {custom_prompt[:60]}" if custom_prompt else "")
                            for i in range(1, n_seg + 1)],
                     team=user, capture_type=capture_type or "general", location=location, camera_id=camera_id,
                     classes=["person"], local_path=str(local), uploaded=now, is_public=is_public)
    for r in rows:
        r["original_video"] = f"s3://{TEAM}-vss-chunks/{key}"
        r["allowed_users"] = users
        r["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        r["_ready_at"] = time.time() + 20 + 5 * r["segment_number"]  # appears progressively (indexed=partial)
        if stream_id:
            em = json.loads(r["extra_metadata"])
            em.update({"stream_id": stream_id, "chunk_index": chunk_number - 1, "ingest_kind": "stream_chunk"})
            r["extra_metadata"] = json.dumps(em)
    PENDING.extend(rows)
    return {"success": True, "object_key": key, "stream_id": stream_id or None,
            "chunk_number": chunk_number if stream_id else None,
            "message": "Video uploaded successfully and will be processed shortly"}  # BE videos.py upload_video()


PENDING: List[Dict[str, Any]] = []


@app.middleware("http")
async def _promote_pending(request: Request, call_next):
    now = time.time()
    ready = [r for r in PENDING if r["_ready_at"] <= now]
    for r in ready:
        PENDING.remove(r)
        ROWS.append(r)
    return await call_next(request)


@app.delete("/api/v1/videos")
def delete_video(original_video: str = Query(..., min_length=1), u: Dict[str, Any] = Depends(current_user)):
    rows = [r for r in ROWS if r["original_video"] == original_video]
    if not rows:
        raise HTTPException(404, "Video not found")
    if u["sub"] not in (rows[0].get("allowed_users") or []) and f"/{u['sub']}/" not in original_video:
        raise HTTPException(403, "You can only delete videos you uploaded")
    for r in rows:
        ROWS.remove(r)
    return {"original_video": original_video, "segments_deleted": len(rows), "prompts_deleted": 0,
            "objects_deleted": len(rows) * 2 + 1, "object_errors": []}


# ---------------------------------------------------------------- dashboard / reingest
@app.get("/api/v1/dashboard/stats")
def stats(scope: str = Query("all", pattern="^(all|mine|public)$"), u: Dict[str, Any] = Depends(current_user)):
    t0 = time.time()
    user = u["sub"]
    rows = [r for r in ROWS if _scope_ok(r, user, scope)]
    dd = rows  # the real table can hold re-ingest duplicates; the mock never creates them
    parents = {r["original_video"] for r in rows}
    full = {p for p in parents if _complete(p)}
    objs: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in dd:
        for lab, n in json.loads(r.get("object_counts") or "{}").items():
            objs[lab][0] += 1
            objs[lab][1] += n
    meta: Dict[str, Dict[str, int]] = {f: defaultdict(int) for f in ("camera_id", "capture_type", "location")}
    for r in dd:
        for f in meta:
            meta[f][r.get(f) or "(empty)"] += 1
    n = len(rows) or 1
    reason_ok = sum(1 for r in rows if r.get("reasoning_content"))
    perc_ok = sum(1 for r in rows if r.get("perception_ok"))
    with_cls = sum(1 for r in rows if r.get("object_classes"))
    recent = []
    for p in sorted(parents):
        segs = [r for r in rows if r["original_video"] == p]
        uniq = len({r["segment_number"] for r in segs})
        recent.append({"original_video": p, "filename": segs[0]["filename"], "segment_rows": len(segs),
                       "indexed_clips": uniq, "chunk_count": 1, "re_ingest_rows": len(segs) - uniq,
                       "unique_segments": uniq, "expected_segments": segs[0].get("total_segments"),
                       "duplicate_rows": len(segs) - uniq, "upload_timestamp": segs[0]["upload_timestamp"],
                       "camera_id": segs[0].get("camera_id"), "capture_type": segs[0].get("capture_type"),
                       "location": segs[0].get("location"), "is_public": segs[0].get("is_public")})
    by_day: Dict[str, int] = defaultdict(int)
    for r in rows:
        by_day[str(r["upload_timestamp"])[:10]] += 1
    dup_rows = len(rows) - len(dd)
    return {
        "table": f"{TEAM}-vss-db/vss-schema/vss-collection", "table_available": True, "table_message": None,
        "generated_at": datetime.now(timezone.utc).isoformat(), "query_time_ms": round((time.time() - t0) * 1000, 2),
        "scope": scope,
        "overview": {"total_rows": len(rows), "segment_rows": len(rows), "other_rows": 0,
                     "unique_videos": len(parents), "fully_indexed_videos": len(full), "indexed_clips": len(dd),
                     "re_ingest_rows": dup_rows, "re_ingest_clips": len({(r["original_video"]) for r in rows}) - len(parents),
                     "stream_sessions": 0, "duplicate_segment_slots": 0,
                     "duplicate_segment_rows": dup_rows,
                     "public_segment_rows": sum(1 for r in rows if r.get("is_public")),
                     "private_segment_rows": sum(1 for r in rows if not r.get("is_public"))},
        "quality": {"reasoning_ok": reason_ok, "reasoning_ok_pct": round(100 * reason_ok / n, 1),
                    "perception_ok": perc_ok, "perception_ok_pct": round(100 * perc_ok / n, 1),
                    "with_object_classes": with_cls, "with_object_classes_pct": round(100 * with_cls / n, 1)},
        "objects": [{"label": k, "segment_count": v[0], "instance_count": v[1]} for k, v in sorted(objs.items())],
        "metadata": {f: [{"label": k, "count": v} for k, v in sorted(m.items())] for f, m in meta.items()},
        "uploads_by_day": [{"date": d, "segment_rows": c} for d, c in sorted(by_day.items(), reverse=True)],
        "recent_videos": recent,
        "s3_inventory": {"chunks_bucket": f"{TEAM}-vss-chunks", "chunks_mp4": len(parents),
                         "segments_bucket": f"{TEAM}-vss-chunks-segments", "segments_mp4": len(rows),
                         "segmenter_output_bucket": f"{TEAM}-vss-chunks-segments", "segmenter_output_mp4": len(rows),
                         "errors": None},
        "pipeline_alignment": {"segments_s3_mp4": len(rows), "indexed_clips": len(dd), "segment_rows": len(rows),
                               "pending_index": len(PENDING), "re_ingest_excess": dup_rows,
                               "indexed_matches_segments_s3": True, "rows_match_segments_s3": True,
                               "segments_bucket_matches_segmenter": True, "healthy": not PENDING},
    }


class ReingestRequest(BaseModel):  # GUESSED shape from kit SKILL.md; route absent from the blueprint snapshot
    original_video: Optional[str] = None
    stream_id: Optional[str] = None
    chunk_count: int = Field(1, ge=1, le=100)
    camera_id: Optional[str] = None
    capture_type: Optional[str] = None
    location: Optional[str] = None
    scenario: Optional[str] = None
    custom_prompt: Optional[str] = Field(None, max_length=MAX_PROMPT)


@app.post("/api/v1/dashboard/reingest")
def reingest(b: ReingestRequest, u: Dict[str, Any] = Depends(current_user)):
    print("[vss-mock] WARNING: dashboard/reingest is a guessed route; verify against the live backend")
    if bool(b.original_video) == bool(b.stream_id):
        raise HTTPException(422, "Provide exactly one of original_video or stream_id")
    target = b.original_video
    if b.stream_id:
        target = next((r["original_video"] for r in ROWS
                       if json.loads(r.get("extra_metadata") or "{}").get("stream_id") == b.stream_id), None)
    segs = _rows_for(target or "", u["sub"]) if target else []
    if not segs:
        raise HTTPException(404, "Video is not indexed")
    jid = uuid.uuid4().hex
    JOBS[jid] = {"t0": time.time(), "target": target, "segs": len(segs), "chunks": b.chunk_count,
                 "prompt": b.custom_prompt or "", "scenario": b.scenario or "", "done": False,
                 "meta": {k: v for k, v in {"camera_id": b.camera_id, "capture_type": b.capture_type,
                                            "location": b.location}.items() if v}}
    return {"job_id": jid, "status": "running", "selected_chunks": b.chunk_count, "copied_segments": len(segs)}


def _finish_job(j: Dict[str, Any]) -> None:
    """Update the target's rows in place (the real backend's row handling is not modelled)."""
    if j["done"]:
        return
    j["done"] = True
    now = _utcnow()
    for r in ROWS:
        if r["original_video"] == j["target"]:
            r["upload_timestamp"] = now.isoformat()
            r["reasoning_content"] = "[re-ingested] " + r["reasoning_content"].removeprefix("[re-ingested] ")
            r.update(j["meta"])


@app.get("/api/v1/dashboard/reingest/{job_id}")
def reingest_status(job_id: str, u: Dict[str, Any] = Depends(current_user)):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    _counter["n"] += 1
    rate = float(os.environ.get("MOCK_REINGEST_404", "0.25"))
    if rate and (_counter["n"] * 7919 % 100) < rate * 100:  # deterministic "other worker" 404s
        raise HTTPException(404, "Job not found")
    dur = float(os.environ.get("MOCK_REINGEST_SECONDS", "30"))
    frac = min(1.0, (time.time() - j["t0"]) / dur)
    if frac >= 1:
        _finish_job(j)
    return {"job_id": job_id, "status": "completed" if frac >= 1 else "running",
            "completed_chunks": int(frac * j["chunks"]), "total_chunks": j["chunks"],
            "indexed_segments": int(frac * j["segs"]), "total_segments": j["segs"]}


# ---------------------------------------------------------------- metadata / suggestions
FILTERABLE = ["camera_id", "capture_type", "location", "object_classes"]
# Field labels and the default labels here and in ingest_config() mirror the blueprint (BE/api/v1/metadata.py,
# source-code/shared/ingest_metadata.py); the use_custom_prompt / metadata_section_* labels and the placeholders
# are the mock's own neutral wording.
LABELS = {"camera_id": "Camera ID", "capture_type": "Capture Type", "location": "Location", "object_classes": "Object class"}
# The real backend serves its own scenario list; the mock exposes one neutral entry, whose label is the
# blueprint's "general" one (source-code/shared/ingest_metadata.py ANALYSIS_SCENARIO_LABELS).
SCENARIOS = [("general", "General Video Analysis")]


def _distinct(field: str, user: str) -> List[str]:
    vals = set()
    for r in ROWS:
        if not _visible(r, user):
            continue
        if field == "object_classes":
            vals.update(x for x in (r.get(field) or "").split(",") if x)
        elif r.get(field):
            vals.add(r[field])
    return sorted(vals)


@app.get("/api/v1/metadata/schema")
def metadata_schema(u: Dict[str, Any] = Depends(current_user)):
    out = []
    for f in FILTERABLE:
        vals = _distinct(f, u["sub"])
        item = {"name": f, "type": "string", "ui_type": "select" if 1 <= len(vals) <= 100 else "text", "label": LABELS[f]}
        if item["ui_type"] == "select":
            item["options"] = vals
        out.append(item)
    return {"schema": out, "table": "vss-collection"}


@app.get("/api/v1/metadata/values")
def metadata_values(field: str = Query(...), prefix: str = "", limit: int = 50, u: Dict[str, Any] = Depends(current_user)):
    if field not in FILTERABLE:  # BE/api/v1/metadata.py
        raise HTTPException(400, f"Field '{field}' is not available for value lookup")
    vals = [v for v in _distinct(field, u["sub"]) if v.lower().startswith(prefix.lower())][:limit]
    return {"field": field, "values": vals, "count": len(vals)}


@app.get("/api/v1/metadata/ingest-config")
def ingest_config():
    # field and default labels, and the one capture-type entry (its "general" item in CAPTURE_TYPES): blueprint
    # source-code/shared/ingest_metadata.py; the rest is the mock's own wording (see the note above LABELS)
    return {
        "custom_prompt_max_length": MAX_PROMPT,
        "filterable_fields": [{"key": k, "label": LABELS[k]} for k in FILTERABLE],
        "capture_types": [{"value": "general", "label": "General"}],
        "capture_type_default_label": "-- Select Type --",
        "analysis_scenarios": [{"value": v, "label": lab} for v, lab in SCENARIOS],
        "analysis_scenario_default_label": "-- Use Default (from settings) --",
        "labels": {"use_custom_prompt": "Use custom prompt", "metadata_section_title": "Metadata",
                   "metadata_section_description": "Optional labels used for filtering",
                   "camera_id": "Camera ID", "capture_type": "Capture Type", "location": "Location",
                   "object_classes": "Object class"},
        "placeholders": {"camera_id": "e.g. cam-01", "location": "e.g. site-a", "tags": "comma,separated",
                         "allowed_users": "comma,separated", "custom_prompt": "Describe what to look for (max 800 chars)"},
    }


@app.get("/api/v1/suggestions")
def suggestions(chips_only: bool = False, u: Dict[str, Any] = Depends(current_user)):
    now = _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    prompts: List[str] = []  # generated server-side by the real backend; not modelled
    events = []
    if not chips_only:
        for r in [r for r in _by_video(ROWS) if _visible(r, u["sub"])][:5]:
            events.append({"query_text": r["reasoning_content"][:60], "label": "", "original_video": r["original_video"],
                           "filename": r["filename"], "segment_start_sec": r["segment_start_sec"],
                           "segment_end_sec": r["segment_end_sec"], "generated_at": now, "batch_id": "mock-batch",
                           "upload_timestamp": str(r["upload_timestamp"]) + "Z"})
    return {"batch_id": "mock-batch", "generated_at": now, "search_prompts": prompts, "key_events": events,
            "key_events_count": len(events), "table": f"{TEAM}-vss-db/vss-schema/vss-prompts-events",
            "prompts_table_available": True, "table_message": None}


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="Shape-only local VSS backend mock")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"[vss-mock] team={TEAM} rows={len(ROWS)} -> http://{a.host}:{a.port}/api/v1 (password via VSS_MOCK_PASSWORD)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
