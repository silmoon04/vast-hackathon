"""Find & Follow request router: a fast LLM turns what the operator said (typed or spoken, filler and mis-hearings
included) into exactly ONE tool call. The browser runs the tool over the scene and writes every fact itself, so the
model never states a time, a camera or a match.

POST /api/find/route  {"text": "...", "state": {...}}  ->  {"tool", "args", "model", "ms"}   (503 when no model answers)
GET  /api/find/health                                     ->  {"ok", "relay", "models"}
GET  /api/find/speech                                     ->  {"available", "provider"}
POST /api/find/transcribe  raw audio/wav (mono 16 kHz PCM16, <= 20 s)  ->  {"text", "ms", "provider"}
POST /api/find/scene?name=&start=HH:MM:SS  raw video  ->  {"scene", "base", "seconds"}  (indexed with build_find_scene)

Speech goes to the team's Find Desk (Canary, push-to-talk) at STM_FIND_STT, default its local tunnel
http://127.0.0.1:18782/find/api/transcribe . That server holds the GPU token; this one only forwards the recording.

The model is reached through STM_FIND_LLM (default http://127.0.0.1:8782/v1/chat/completions): a local-only relay on
the lab VM, tunnelled with  ssh -N -L 8782:127.0.0.1:8782 <vm> . The relay adds the W&B key on the VM, so no key is
on this machine. Models are tried in order (STM_FIND_MODELS, comma separated).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

LLM_URL = os.environ.get("STM_FIND_LLM", "http://127.0.0.1:8782/v1/chat/completions")
MODELS = [m.strip() for m in os.environ.get(
    "STM_FIND_MODELS", "deepseek-ai/DeepSeek-V4.1-Flash,zai-org/GLM-5.3-Flash,deepseek-ai/DeepSeek-V3.1").split(",") if m.strip()]
TIMEOUT_S = float(os.environ.get("STM_FIND_TIMEOUT", "6"))
STT_URL = os.environ.get("STM_FIND_STT", "http://127.0.0.1:18782/find/api/transcribe")
MAX_AUDIO_BYTES = 1 << 20

SYSTEM = """You route a venue operator's request in a "find and follow" camera desk to exactly ONE tool call.
Never answer in text; always call one tool. The desk finds a person by clothing and carried items (never by face or
name), lets the user confirm which candidate is them, then watches the spot they left their item, flags who was there
when the item disappeared (a lead, not a culprit), and follows people camera to camera.
Rules:
- Input may be speech-to-text: ignore filler, fix obvious mis-hearings of clothing words ("read hoodie" -> "red hoodie",
  "great backpack" -> "grey backpack"). Keep colours and garments exactly as meant.
- A new description of the user (or of anyone) -> search_people. Extra details about the same person while candidates
  are on screen ("with a backpack", "I was at the table") -> search_people with refine=true.
- "yes", "that's me", "the first one", "number two", "the left one" -> pick_candidate (1-based; "yes"/"that's me" = 1).
- "no", "not me", "not that one" -> reject_candidate.
- Who came near / who took / who touched / what happened to my phone or bike -> who_came_near.
- Times: the desk state gives the footage clock range. Turn any time the user gives ("after ten to five", "between
  16:51 and 16:52", "in the last minute", "around 4:52pm") into clock times HH:MM:SS inside that range.
- Narrowing the matches on screen to a time window or a camera ("only after 16:52", "just camera B", "the ones around
  16:51", "earlier than that") -> narrow_search. A brand-new description with a time -> search_people with after/before.
  narrow_search keeps the current window and changes only the fields you pass; pass "" to widen a field again
  ("any camera" -> camera ""; "any time" / "the whole recording" -> after "" and before "").
- When was someone seen / what time did they arrive or leave / how long were they there -> when_seen.
- How many people / how busy / anyone at all on a camera or in a time window -> count_people.
- Where did they go / follow them / where is it now -> follow_lead. Where did I go / my route -> where_did_i_go.
- Show / play the moment it went -> show_moment. Ghost / trails / everyone who passed -> ghost_trails.
- Start over / reset -> start_over.
- Asking who someone IS (name, identity, face, "is that John"), or anything the desk cannot do -> cannot_do.
"""


def _fn(name: str, desc: str, props: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props or {}, "required": required or []}}}


TOOLS = [
    _fn("search_people", "Search every camera for people matching a description of clothing colours and carried items.",
        {"description": {"type": "string", "description": "clothing and items only, e.g. 'red hoodie, grey backpack'"},
         "refine": {"type": "boolean", "description": "true to add to the description already on screen"},
         "after": {"type": "string", "description": "optional clock time HH:MM:SS"},
         "before": {"type": "string", "description": "optional clock time HH:MM:SS"},
         "camera": {"type": "string", "description": "optional camera letter"}}, ["description"]),
    _fn("narrow_search", "Keep only the matches on screen seen inside a time window and/or on one camera.",
        {"after": {"type": "string", "description": "clock time HH:MM:SS"},
         "before": {"type": "string", "description": "clock time HH:MM:SS"},
         "camera": {"type": "string", "description": "camera letter"}}),
    _fn("when_seen", "Report when the confirmed person (or the best match) was first and last seen on each camera."),
    _fn("count_people", "Count the people seen on a camera and/or in a time window.",
        {"after": {"type": "string"}, "before": {"type": "string"}, "camera": {"type": "string"}}),
    _fn("pick_candidate", "The user says which candidate on screen is them.",
        {"number": {"type": "integer", "minimum": 1, "maximum": 6}}, ["number"]),
    _fn("reject_candidate", "The user says a candidate on screen is not them.",
        {"number": {"type": "integer", "minimum": 1, "maximum": 6}}, ["number"]),
    _fn("who_came_near", "Who came near the user's item or spot after the user left it."),
    _fn("follow_lead", "Follow the flagged lead (or the item) camera to camera to the last sighting."),
    _fn("where_did_i_go", "Show the confirmed user's own route across cameras."),
    _fn("show_moment", "Play the moment the item stopped being visible."),
    _fn("ghost_trails", "Draw the paths of everyone who passed through each view after the user left.",
        {"on": {"type": "boolean"}}),
    _fn("start_over", "Clear the search and start again."),
    _fn("cannot_do", "The request is outside what the desk does (e.g. naming or identifying a person).",
        {"reason": {"type": "string"}}, ["reason"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
TEXT_CALL_RE = re.compile(r'"?name"?\s*[:=]\s*"([a-z_]+)"[\s\S]*?"?arguments"?\s*[:=]\s*(\{[\s\S]*?\})', re.I)

router = APIRouter(prefix="/api/find", tags=["find"])


class RouteIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=400)
    state: dict[str, Any] = Field(default_factory=dict)


def _post(body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(LLM_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _state_line(st: dict[str, Any]) -> str:
    parts = []
    if st.get("clock"):
        parts.append(f"Footage clock: {st['clock']}")
    if st.get("cameras"):
        parts.append(f"Cameras: {st['cameras']}")
    if st.get("window"):
        parts.append(f"Current time window: {st['window']}")
    cands = st.get("candidates") or []
    if cands:
        parts.append("Candidates on screen: " + "; ".join(f"{i + 1}. {c}" for i, c in enumerate(cands[:6])))
    parts.append("User confirmed: " + ("yes" if st.get("subject") else "not yet"))
    if st.get("item"):
        parts.append(f"User's item: {st['item']}")
    parts.append("Lead flagged: " + ("yes" if st.get("lead") else "no"))
    return "Desk state. " + " | ".join(parts)


def parse_call(msg: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The tool call from an OpenAI-style message; also accepts a call written as text (some models do that)."""
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = fn.get("name")
        if name in TOOL_NAMES:
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            return name, args if isinstance(args, dict) else {}
    content = msg.get("content") or ""
    m = TEXT_CALL_RE.search(content)
    if m and m.group(1) in TOOL_NAMES:
        try:
            return m.group(1), json.loads(m.group(2))
        except ValueError:
            return m.group(1), {}
    m = re.search(r"\b(" + "|".join(sorted(TOOL_NAMES)) + r")\b", content)
    if m and m.group(1) not in ("search_people", "pick_candidate", "reject_candidate", "cannot_do"):
        return m.group(1), {}
    return None


def route(text: str, state: dict[str, Any]) -> dict[str, Any]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "system", "content": _state_line(state)},
            {"role": "user", "content": text}]
    tried = []
    for model in MODELS:
        t0 = time.perf_counter()
        body = {"model": model, "messages": msgs, "tools": TOOLS, "tool_choice": "auto", "temperature": 0,
                "max_tokens": 300, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            out = _post(body, TIMEOUT_S)
        except urllib.error.HTTPError as e:
            if e.code == 400:                     # a model that rejects chat_template_kwargs: once more without
                body.pop("chat_template_kwargs", None)
                try:
                    out = _post(body, TIMEOUT_S)
                except Exception as e2:
                    tried.append(f"{model}: {type(e2).__name__}")
                    continue
            else:
                tried.append(f"{model}: HTTP {e.code}")
                continue
        except Exception as e:
            tried.append(f"{model}: {type(e).__name__}")
            continue
        ms = round((time.perf_counter() - t0) * 1000)
        msg = ((out.get("choices") or [{}])[0].get("message")) or {}
        call = parse_call(msg)
        if call:
            return {"tool": call[0], "args": call[1], "model": model, "ms": ms, "tried": tried}
        tried.append(f"{model}: no tool call")
    raise HTTPException(503, {"error": "no model routed the request", "tried": tried})


@router.post("/route")
def api_route(body: RouteIn) -> dict[str, Any]:
    return route(body.text, body.state)


@router.get("/health")
def api_health() -> dict[str, Any]:
    url = LLM_URL.replace("/chat/completions", "/models")
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
        return {"ok": True, "relay": True, "models": [m for m in MODELS if m in ids]}
    except Exception as e:
        return {"ok": False, "relay": False, "error": type(e).__name__, "models": []}


def _stt_detail(e: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(e.read()).get("detail") or f"HTTP {e.code}")[:200]
    except Exception:
        return f"HTTP {e.code}"


def transcribe(audio: bytes) -> dict[str, Any]:
    """Forward one push-to-talk recording to the Find Desk's Canary route; the provider's time is reported apart."""
    if not 44 <= len(audio) <= MAX_AUDIO_BYTES or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        raise HTTPException(400, {"error": "send mono 16 kHz PCM16 WAV of up to 20 s"})
    req = urllib.request.Request(STT_URL, data=audio, headers={"Content-Type": "audio/wav"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=35) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise HTTPException(502 if e.code >= 500 else e.code, {"error": _stt_detail(e)})
    except Exception as e:
        raise HTTPException(503, {"error": f"speech service unreachable ({type(e).__name__})"})
    return {"text": str(out.get("text") or "").strip(), "ms": round((time.perf_counter() - t0) * 1000),
            "provider": out.get("provider") or "Canary", "provider_ms": out.get("latency_ms")}


@router.get("/speech")
def api_speech() -> dict[str, Any]:
    url = STT_URL.replace("/transcribe", "/speech")
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            out = json.loads(r.read())
        return {"available": bool(out.get("available")), "provider": out.get("provider") or "Canary"}
    except Exception as e:
        return {"available": False, "provider": None, "error": type(e).__name__}


@router.post("/transcribe")
async def api_transcribe(request: Request) -> dict[str, Any]:
    audio = await request.body()
    return await run_in_threadpool(transcribe, audio)


ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "out" / "find"
MAX_VIDEO_BYTES = 400 << 20
CLOCK_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def build_upload(video: bytes, name: str, start: str | None) -> dict[str, Any]:
    """Index one uploaded camera video with the same fast pipeline as the demo scenes; served at /out/find/<scene>/."""
    if not 1000 < len(video) <= MAX_VIDEO_BYTES:
        raise HTTPException(400, {"error": "send one video file of up to 400 MB"})
    slug = (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32] or "upload") + time.strftime("-%H%M%S")
    work = SCENES / "_in" / slug
    work.mkdir(parents=True, exist_ok=True)
    (work / "video.mp4").write_bytes(video)
    clock = start if start and CLOCK_RE.match(start) else "00:00:00"      # unknown wall clock: show video time
    cams = {"title": name[:60] or "Uploaded video", "clock0": clock, "date": time.strftime("%Y-%m-%d"),
            "credit": "Uploaded at the desk" + ("" if start else " · times are video time"),
            "cameras": [{"id": "A", "name": (name[:40] or "Uploaded camera"), "video": "video.mp4", "offset_s": 0.0,
                         "pos": [0.5, 0.5], "exits": {}, "links": []}]}
    (work / "cams.json").write_text(json.dumps(cams), encoding="utf-8")
    t0 = time.perf_counter()
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_find_scene.py"), "--cams", str(work / "cams.json"),
                            "--scene", slug, "--out", str(SCENES)], capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        raise HTTPException(504, {"error": "indexing took longer than 15 minutes; try a shorter clip"})
    if r.returncode or not (SCENES / slug / "scene.json").exists():
        raise HTTPException(500, {"error": "indexing failed", "log": (r.stdout + r.stderr)[-600:]})
    return {"scene": slug, "base": f"/out/find/{slug}/", "seconds": round(time.perf_counter() - t0, 1),
            "summary": [ln for ln in r.stdout.strip().splitlines() if ln.startswith(("scene", "build"))][-2:]}


@router.post("/scene")
async def api_scene(request: Request, name: str = "Uploaded video", start: str | None = None) -> dict[str, Any]:
    video = await request.body()
    return await run_in_threadpool(build_upload, video, name, start)
