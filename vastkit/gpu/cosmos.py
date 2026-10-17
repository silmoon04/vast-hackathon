"""One client for Cosmos reasoning VLMs behind an OpenAI-compatible API.

The same code talks to the event's COSMOS3_REASON_URL (NIM or vLLM), a local vLLM/WSL
server, or ``vastkit/local/cosmos_edge_server.py`` (``python -m vastkit local``) on this laptop.

Known failure modes this handles:
- the documented model id 404s, so the id is discovered via GET /v1/models
- media_io_kwargs fps above the clip's real fps -> 400 (retried once at the clip fps the error names);
  fps and num_frames together -> 400
- reasoning needs the official <think>/<answer> suffix and max_tokens >= 4096
- Edge thinks by default: send chat_template_kwargs enable_thinking=False for direct answers
- finish_reason == "length": continue up to 4 rounds (as the blueprint backend does:
  source-code/retrieval/video-backend/src/services/llm_service.py, synthesis_max_continuations in config.py)
- endpoints that reject response_format / chat_template_kwargs: retried without them (remembered
  per client only when the server says the field itself is unknown)
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .. import http as _http
from ..config import Settings, settings as _settings

log = logging.getLogger("vastkit.cosmos")

# reasoning output format from the Cosmos Reason model card
REASON_SUFFIX = ("\nAnswer the question in the following format: <think>\nyour reasoning\n</think>\n\n"
                 "<answer>\nyour answer\n</answer>.")
# continuation turn; same idea as the blueprint's source-code/retrieval/video-backend/src/services/llm_service.py
# (reworded, not copied)
CONTINUE_MSG = ("Your last reply was cut off. Carry on from the point where it stopped, without starting "
                "over or repeating anything already written.")
FALLBACK_MODELS = ("nvidia/cosmos3-nano-reasoner", "nvidia/cosmos3-reason")
# optional request keys -> words in a 400/422 detail that point at that key
_OPTIONAL_KEYS = {
    "response_format": ("response_format", "json_schema", "guided"),
    "chat_template_kwargs": ("chat_template_kwargs", "enable_thinking"),
    "media_io_kwargs": ("media_io_kwargs", "fps", "num_frames"),
    "mm_processor_kwargs": ("mm_processor_kwargs", "pixels"),
    "top_k": ("top_k",),
}
# detail wording that means the server does not know the field at all (not that it rejected a value)
_UNKNOWN_FIELD = ("not permitted", "extra_forbidden", "unexpected keyword", "unrecognized", "unknown field",
                  "unknown parameter", "not supported", "unsupported")
CACHE_DIR = Path.home() / ".vastkit" / "cache" / "cosmos"

# Official prompt wording only; write your own on the day.
PROMPTS = {
    # https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/prompts/caption.yaml
    "caption": "Caption the video in detail.",
    # https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/prompts/temporal_localization.yaml
    "events": ("Describe the notable events in the provided video. Provide the result in json format with "
               "'mm:ss.ff' format for time depiction for each event. Use keywords 'start', 'end' and "
               "'caption' in the json output."),
}

EVENTS_SCHEMA = {
    "type": "object",
    "properties": {"events": {"type": "array", "items": {
        "type": "object",
        "properties": {"start": {"type": "string"}, "end": {"type": "string"},
                       "caption": {"type": "string"}},
        "required": ["start", "end", "caption"]}}},
    "required": ["events"],
}


def split_reasoning(text: str, reasoning_content: str | None = None) -> tuple[str, str]:
    """Return (think, answer) from '<think>..</think><answer>..</answer>' or bare text."""
    text = text or ""
    think = (reasoning_content or "").strip()
    m = re.search(r"<think>(.*?)</think>", text, re.S)
    if m and not think:
        think = m.group(1).strip()
    elif "</think>" in text and not think:  # template already opened <think>, so only the close tag is emitted
        think = text.split("</think>")[0].replace("<think>", "").strip()
    a = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)
    answer = a.group(1).strip() if a else text.split("</think>")[-1].strip()
    return think, answer


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in model output (handles ```json fences)."""
    if text is None:
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except ValueError:
        pass
    pairs = sorted((("{", "}"), ("[", "]")), key=lambda p: s.find(p[0]) % (len(s) + 1))  # earliest opener first
    for open_c, close_c in pairs:
        i, j = s.find(open_c), s.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                continue
    return None


_EVENT_RE = re.compile(r'"start"\s*:\s*"?([\d:.]+)"?\s*,\s*"end"\s*:\s*"?([\d:.]+)"?\s*,\s*'
                       r'"caption"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def scan_events(text: str) -> list[dict[str, str]]:
    """Regex fallback for broken JSON: every "start": .., "end": .., "caption": .. run in the text."""
    return [{"start": a, "end": b, "caption": c} for a, b, c in _EVENT_RE.findall(text)]


def mmss_to_sec(t: str | float | int) -> float:
    """'mm:ss.ff' / 'hh:mm:ss' / '12.5' / '<12.5 seconds>' -> seconds (first time in the text; 0.0 if none)."""
    if isinstance(t, (int, float)):
        return float(t)
    m = re.search(r"\d+(?:\.\d+)?(?::\d+(?:\.\d+)?)*", str(t))
    try:
        parts = [float(p) for p in m.group(0).split(":")] if m else []
    except ValueError:
        return 0.0
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def video_data_uri(path_or_bytes: str | Path | bytes, mime: str = "video/mp4") -> str:
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _is_loopback(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _clamp_fps(mio: Any, detail: str) -> dict[str, Any] | None:
    """media_io_kwargs with fps lowered to the clip fps an 'exceeds video fps N' 400 names, else None."""
    m = re.search(r"exceeds\D*?(\d+(?:\.(\d+))?)", detail)
    vid = mio.get("video", mio) if isinstance(mio, dict) else None
    if not m or not isinstance(vid, dict) or not isinstance(vid.get("fps"), (int, float)):
        return None
    fps, n = float(m.group(1)), len(m.group(2) or "")
    fps = round(fps - 10 ** -n, n) if n else fps  # the detail rounds the clip fps: stay one printed digit below
    if not 0 < fps < vid["fps"]:
        return None
    vid = {**vid, "fps": fps}
    return {**mio, "video": vid} if "video" in mio else vid


@dataclass
class CosmosResult:
    answer: str
    think: str = ""
    raw: str = ""
    model: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    cached: bool = False
    dropped: list[str] = field(default_factory=list)  # requested keys the server rejected (answer made without)

    @property
    def json(self) -> Any:
        return extract_json(self.answer) if extract_json(self.answer) is not None else extract_json(self.raw)


class CosmosReason:
    def __init__(self, base_url: str | None = None, *, model: str | None = None, token: str | None = None,
                 proxy: str | None = None, timeout: float = 180.0, cfg: Settings | None = None,
                 cache: bool | None = None):
        cfg = cfg or _settings()
        self.base_url = (base_url or cfg.cosmos_reason_url).rstrip("/")
        if not self.base_url:
            raise ValueError("COSMOS3_REASON_URL unset (or VSS_MOCK=1 for the GPU mock)")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self._model = model or None
        self._env_model = cfg.cosmos_reason_model
        # disk cache of ask_video answers; off by default for the mock and loopback servers (mock / local mirror)
        self.cache = cache if cache is not None else not (cfg.mock or _is_loopback(self.base_url))
        self.unsupported: set[str] = set()  # request keys this server rejected
        self.client = _http.make_client(self.base_url, proxy=proxy if proxy is not None else cfg.socks_proxy,
                                        token=token if token is not None else cfg.gpu_bearer_token,
                                        timeout=timeout)

    # ---------- discovery / health ----------
    def models(self) -> list[str]:
        r = _http.request(self.client, "GET", "/v1/models")
        if r.status_code != 200:
            raise _http.APIError(r, "GET /v1/models")
        return [m["id"] for m in r.json().get("data", [])]

    @property
    def model(self) -> str:
        if not self._model:
            try:
                ids = self.models()
                self._model = ids[0] if ids else None
            except (httpx.HTTPError, _http.APIError, KeyError) as exc:
                log.warning("model discovery failed (%s); falling back", exc)
            self._model = self._model or self._env_model or FALLBACK_MODELS[0]
        return self._model

    def ready(self) -> bool:
        """/v1/health/ready decides; /v1/models or /health only when that route 404/405s."""
        try:
            code = self.client.get("/v1/health/ready").status_code
            if code not in (404, 405):
                return code == 200
            return any(self.client.get(p).status_code == 200 for p in ("/v1/models", "/health"))
        except httpx.HTTPError:
            return False

    # ---------- core chat ----------
    def chat(self, messages: list[dict[str, Any]], *, max_tokens: int = 1024, temperature: float = 0.2,
             top_p: float | None = None, json_schema: dict[str, Any] | None = None,
             extra: dict[str, Any] | None = None, continue_rounds: int = 4) -> CosmosResult:
        t0 = time.time()
        body: dict[str, Any] = {"model": self.model, "messages": list(messages), "max_tokens": max_tokens,
                                "temperature": temperature, "stream": False}
        if top_p is not None:
            body["top_p"] = top_p
        if json_schema and "response_format" not in self.unsupported:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "result", "schema": json_schema}}
        for k, v in (extra or {}).items():
            if k not in self.unsupported:
                body[k] = v
        pieces: list[str] = []
        reasoning, finish, usage = "", "", {}
        for _round in range(continue_rounds + 1):
            r = self._post(body)
            j = r.json()
            ch = j["choices"][0]
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            reasoning = reasoning or msg.get("reasoning_content") or msg.get("reasoning") or ""
            pieces.append(text)
            finish = ch.get("finish_reason") or ""
            u = j.get("usage") or {}
            for k2, v2 in u.items():
                if isinstance(v2, (int, float)):
                    usage[k2] = usage.get(k2, 0) + v2
            if finish not in ("length", "max_tokens") or json_schema:
                break
            body["messages"] = body["messages"] + [{"role": "assistant", "content": text},
                                                   {"role": "user", "content": CONTINUE_MSG}]
        raw = "".join(pieces)
        think, answer = split_reasoning(raw, reasoning)
        requested = [*(["response_format"] if json_schema else []), *(extra or {})]
        return CosmosResult(answer=answer, think=think, raw=raw, model=body["model"], finish_reason=finish,
                            usage=usage, latency_s=round(time.time() - t0, 2),
                            dropped=[k for k in requested if k not in body])

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        """POST chat/completions, retrying around optional keys the server rejects (400/422).

        A key goes into self.unsupported only when the server says the field itself is unknown; other
        rejections drop it for this request only. media_io_kwargs value errors are never dropped: fps
        above the clip fps is retried once at the clip fps the error names, anything else raises.
        """
        clamped = False
        for _ in range(4):
            r = _http.request(self.client, "POST", "/v1/chat/completions", json=body)
            if r.status_code == 200:
                return r
            if r.status_code == 404 and "model" in _http.error_detail(r).lower() and body["model"] != FALLBACK_MODELS[0]:
                log.warning("model %s not found; trying discovery again", body["model"])
                self._model = None
                body["model"] = self.model
                continue
            if r.status_code in (400, 422):
                detail = _http.error_detail(r).lower()
                key = next((k for k, hints in _OPTIONAL_KEYS.items()
                            if k in body and any(h in detail for h in hints)), None)
                if key == "media_io_kwargs" and ("exceeds" in detail or "exclusive" in detail):
                    mio = None if clamped else _clamp_fps(body[key], detail)
                    if mio is not None:
                        log.warning("fps above the clip fps; retrying with %s", mio)
                        body[key], clamped = mio, True
                        continue
                elif key and any(w in detail for w in _UNKNOWN_FIELD):
                    log.warning("server does not accept %s; leaving it out for this client", key)
                    self.unsupported.add(key)
                    body.pop(key)
                    continue
                elif key and key != "media_io_kwargs":
                    log.warning("server rejected %s; retrying this request without it", key)
                    body.pop(key)
                    continue
            raise _http.APIError(r, "chat/completions")
        raise _http.APIError(r, "chat/completions")

    # ---------- video helpers ----------
    def ask_video(self, video: str | Path | bytes, question: str, *, system: str | None = None,
                  fps: float | None = None, num_frames: int | None = None, source_fps: float | None = None,
                  reasoning: bool = False, thinking_flag: bool | None = None,
                  json_schema: dict[str, Any] | None = None, max_tokens: int | None = None,
                  temperature: float = 0.2, mime: str = "video/mp4") -> CosmosResult:
        """Ask about one short clip (prep it with vastkit.video.prep_clip: ~640px, <=20 s).

        reasoning=True appends the official <think>/<answer> suffix (max_tokens >= 4096).
        thinking_flag sets chat_template_kwargs.enable_thinking (Edge thinks by default).
        """
        if fps is not None and num_frames is not None:
            raise ValueError("fps and num_frames are mutually exclusive (server returns 400)")
        if fps is not None and source_fps is not None and fps > source_fps:
            log.info("clamping fps %.2f to source fps %.2f", fps, source_fps)
            fps = source_fps
        data = video if isinstance(video, bytes) else Path(video).read_bytes()
        prompt = question + (REASON_SUFFIX if reasoning else "")
        extra: dict[str, Any] = {}
        if fps is not None:
            extra["media_io_kwargs"] = {"video": {"fps": fps}}
        elif num_frames is not None:
            extra["media_io_kwargs"] = {"video": {"num_frames": num_frames}}
        if thinking_flag is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": thinking_flag}
        mx = max_tokens or (4096 if reasoning else 1024)
        key = None
        if self.cache:
            key = hashlib.sha256(data + json.dumps([self.base_url, self.model, prompt, system, extra, json_schema,
                                                    mx, temperature], sort_keys=True).encode()).hexdigest()
            hit = CACHE_DIR / f"{key}.json"
            if hit.is_file():
                d = json.loads(hit.read_text(encoding="utf-8"))
                return CosmosResult(**{**d, "cached": True})
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": video_data_uri(data, mime)}},
        ]})
        res = self.chat(msgs, max_tokens=mx, temperature=temperature, json_schema=json_schema, extra=extra)
        if key and res.answer and not res.dropped:  # never cache an answer made without the requested extras
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            d = {k: getattr(res, k) for k in ("answer", "think", "raw", "model", "finish_reason", "usage", "latency_s")}
            (CACHE_DIR / f"{key}.json").write_text(json.dumps(d), encoding="utf-8")
        return res

    def events(self, video: str | Path | bytes, *, prompt: str | None = None, **kw: Any) -> list[dict[str, Any]]:
        """Thin wrapper over the official temporal-localisation prompt (PROMPTS["events"]): the model's
        start/end/caption items parsed into [{start_sec, end_sec, caption}]."""
        res = self.ask_video(video, prompt or PROMPTS["events"], json_schema=EVENTS_SCHEMA,
                             thinking_flag=kw.pop("thinking_flag", False), **kw)
        parsed = res.json
        items = parsed.get("events", parsed) if isinstance(parsed, dict) else parsed
        if isinstance(items, dict):  # a single event object
            items = [items]
        rows = [e for e in items if isinstance(e, dict) and "caption" in e] if isinstance(items, list) else []
        if not rows:  # malformed JSON from small models: salvage start/end/caption triples
            rows = scan_events(res.answer or res.raw or "")
        return [{"start_sec": mmss_to_sec(e.get("start", 0)), "end_sec": mmss_to_sec(e.get("end", 0)),
                 "caption": str(e["caption"])} for e in rows]

    def ask_text(self, prompt: str, *, system: str | None = None, **kw: Any) -> CosmosResult:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)

    def close(self) -> None:
        self.client.close()
