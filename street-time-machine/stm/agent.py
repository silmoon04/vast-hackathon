"""Visibility-first Ask agent: an LLM on W&B Inference picks the tools, the code keeps the answer honest.

`answer(cam, question, t=None) -> dict` returns the stm.ask keys (type, answer, sentence, latency_ms, evidence,
honesty, blind, last_seen, refused) plus trace[]: every guard, LLM call and tool call with its args, latency_ms
and the model that asked for it. The server can call it as a drop-in next to stm.ask.answer.

Tools the LLM may call (all read files under out/<cam>/, nothing is written):
  stm_visibility(start, end)  seen / blind seconds and blind spans with reasons, from ledger.json
  stm_ask(question)           the rule engine (stm.ask), else the matching stored answer in qa.json
  stm_near(object_id | t)     who came near a parked vehicle, from near.json
  stm_diff()                  day-to-day changes, from diff.json
  vss_search(query)           caption search over the stored VSS rows (segments.json, keyword + IDF); the live team
                              VSS /search (a POST, through stm.vss) only with STM_VSS_LIVE=1, falling back to the
                              stored rows when it fails. Every hit is put on the camera clock (the chunk_offsets.json
                              whose total matches this video) and checked against the ledger: a hit inside blind
                              time is tagged 'not evidence: <reason>'.
  vss_segment(source)         one caption segment by its source URI, tagged the same way

Guards live in code, not in the prompt:
* 'who did it / who is this / identify' questions get a fixed refusal before any LLM call.
* If >= 50% of the asked time range is blind (or not recorded), the answer is CAN'T TELL with the reason, no LLM.
* At most 3 tool rounds inside an 8 s budget. Past that, or when W&B is unreachable, the rule engine answers.
* The LLM writes only the sentence. Type, answer and evidence come from tool outputs; a sentence carrying a number no
  tool returned, or contradicting the tool's YES / NO, is replaced by the tool's own sentence.

Config: WANDB_API_KEY, WANDB_TEAM, WANDB_PROJECT from the environment, else read from /etc/environment in memory.
The key is never printed, logged or put in a repr. Optional Weave trace to trace.wandb.ai behind STM_WEAVE=1 (off).
Stdlib only (urllib). CLI: python -m stm.agent <cam> "<question>" [--t SECONDS]
"""
from __future__ import annotations

import argparse
import base64
import importlib
import inspect
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
API_URL = "https://api.inference.wandb.ai/v1/chat/completions"
WEAVE_URL = "https://trace.wandb.ai"
ENV_FILE = "/etc/environment"
WANDB_KEYS = ("WANDB_API_KEY", "WANDB_TEAM", "WANDB_PROJECT")
MODELS = ("deepseek-ai/DeepSeek-V3.1", "zai-org/GLM-5.3", "meta-llama/Llama-3.3-70B-Instruct")
MAX_ROUNDS = 3          # tool rounds; one more call (tool_choice none) may then write the sentence
BUDGET_S = 8.0          # wall clock for the whole answer, tools included
MAX_TOKENS = 1500
BLIND_FORCE = 0.5       # this share of the asked range blind -> CAN'T TELL
POINT_HALF = 5.0        # a question about one moment looks at +-5 s around it
TOOL_CHARS = 4000       # cap on one tool result as the LLM sees it
MAX_EVIDENCE = 12
LIVE_VSS_ENV = "STM_VSS_LIVE"   # =1 lets vss_search POST to the team VSS /search; off: stored captions only
# Pack D files besides out/<cam>/ and $STM_CHUNK_OFFSETS / $STM_SEGMENTS: the seam-safe build (scripts/build_packd_v2.py),
# the laptop corpus stm.captions reads, the VM's raw mirror. chunk_offsets.json is picked by its total (see _offsets).
PACKD_DIRS = (ROOT / "packd", ROOT / "data" / "packd", Path("~/packd"))
OFFSETS_TOL_S = 15.0    # a chunk_offsets.json total this far (or 1 %) from the video length is another layout

REFUSAL = "I can show who came near and when, never who did it."
REFUSAL_HONESTY = "by design: being near is not doing it; no person is identified or described"
CANT_TELL = "CAN'T TELL"

REASON_WORDS = {"turned_away": "camera turned away", "moved": "camera turned away", "cut": "camera turned away",
                "camera_moving": "camera turned away", "view_changed": "camera turned away", "blocked": "view blocked",
                "dark": "too dark", "blur": "blurred", "frozen": "picture frozen", "no_signal": "no signal",
                "blind": "camera blind", "not_recorded": "not recorded"}

IDENTITY_RE = re.compile(
    r"\b(identif\w*|recogni[sz]\w*|facial|face\s*(?:id|match)|who\s+(?:is|was|are|were)\s+(?:this|that|these|those|"
    r"he|she|they|him|her|it|the\s+(?:person|people|man|men|woman|women|guy|driver|cyclist|kid|owner|walker))|"
    r"who'?s\s+(?:this|that|the)|whose\s+(?:car|van|truck|bike|bicycle|vehicle|motorcycle)|"
    r"name\s+of\s+the|what\s+is\s+(?:his|her|their)\s+name|(?:licen[cs]e|number)\s+plates?|plate\s+number)\b")
BLAME_RE = re.compile(
    r"\b(?:who|which\s+\w+|what|anyone|anybody|someone|somebody)\b[^?.!]*?\b(dent(?:ed|s)?|scratch(?:ed|es)?|hit|hits|"
    r"stole|stolen|steal\w*|damag\w*|keyed|broke|broken|smash\w*|vandali[sz]\w*|crash\w*|scrap(?:e|ed)|bump\w*|"
    r"rob(?:bed)?|nicked|did\s+(?:it|this|that))\b|\b(blame|fault|responsible|culprit|suspect|guilty|perpetrator)\b")
TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")
SECS_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b")
LAST_RE = re.compile(r"\b(?:last|past)\s+(\d+(?:\.\d+)?)\s*(h|hrs?|hours?|m|mins?|minutes?|s|secs?|seconds?)\b")
NOW_RE = re.compile(r"\b(still|now|currently|at the moment|right now|at the end)\b")
NUM_RE = re.compile(r"\d+(?::\d{2})+|\d+(?:\.\d+)?")
# DeepSeek sometimes writes its tool call into the content as template text, V3.1:
#   <|tool_call_begin|>NAME<|tool_sep|>{json}<|tool_call_end|>   (the bars are U+FF5C, the underscores U+2581)
# and V3: <|tool_call_begin|>function<|tool_sep|>NAME ```json {json} ```<|tool_call_end|>
_B, _U = "[|\uff5c]", "[_\u2581]"
DS_CALL_RE = re.compile(rf"<{_B}tool{_U}call{_U}begin{_B}>(?:function<{_B}tool{_U}sep{_B}>)?\s*([\w.-]+)\s*"
                        rf"(?:<{_B}tool{_U}sep{_B}>)?\s*(?:```(?:json)?)?\s*(\{{.*?\}})\s*(?:```)?\s*"
                        rf"<{_B}tool{_U}call{_U}end{_B}>", re.S)
MARKUP_RE = re.compile(rf"<{_B}[^<>]{{0,40}}{_B}>")
FINAL_NUDGE = "No more tools. Write the final answer now: one or two plain sentences from the tool results above."
ORDINALS = ("first", "second", "third", "fourth")

SYSTEM = (
    "You are the Ask agent of Street Time Machine. You answer questions about the recording of one fixed street "
    "camera, using only the tools; never answer from general knowledge.\n"
    "- Times are seconds into the recording; write them as mm:ss like the tools do.\n"
    "- For a question about a time or a period, check stm_visibility for it; if the camera was blind, say what it "
    "could not see.\n"
    "- stm_ask answers: is X still there, when did X leave, which way did X go, who came near X, how many X since a "
    "time, what changed between the days. stm_near lists who came close to a parked vehicle. stm_diff compares the "
    "days. vss_search finds AI captions for a description; a hit tagged 'not evidence' must not be used as fact.\n"
    "- Never say who did something or who someone is, and never describe people's looks.\n"
    "- Use as few tool calls as you need (usually one or two), then answer.\n"
    "- Final reply: one or two plain sentences, no markdown, no lists, using only numbers and times that appear in "
    "the tool results.")


def _fn(name: str, desc: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


TOOLS = [
    _fn("stm_visibility", "What the camera could see between two times (seconds into the recording): seen and "
        "blind seconds, blind spans with reasons. Check this for any question about a time or period.",
        {"start": {"type": "number", "description": "seconds"}, "end": {"type": "number", "description": "seconds"}},
        ["start", "end"]),
    _fn("stm_ask", "Rule engine over the detector tracks: still there, when left, which way, who came near, how many "
        "since a time, what changed between the days. Pass the question in plain words (object ids as #12).",
        {"question": {"type": "string"}}, ["question"]),
    _fn("stm_near", "Who came within half a car-height of a parked vehicle while it was parked, with times. Give "
        "object_id (a parked vehicle or a visitor), or t (seconds) for vehicles parked then, or nothing for the "
        "busiest ones. Being near is not doing anything.",
        {"object_id": {"type": "integer"}, "t": {"type": "number", "description": "seconds"}}, []),
    _fn("stm_diff", "Day-to-day comparison of the parked vehicles: same / arrived / left / moved / swapped, with "
        "notes on what could not be compared.", {}, []),
    _fn("vss_search", "Search the AI captions of this camera's recording for a description, e.g. 'a delivery van'. "
        "Hits come with camera times; hits in blind time or outside the asked time are tagged 'not evidence'. "
        "start / end (seconds) narrow it to a period; by default it is the period the question is about.",
        {"query": {"type": "string"}, "top_k": {"type": "integer", "description": "1-10, default 5"},
         "start": {"type": "number", "description": "seconds"}, "end": {"type": "number", "description": "seconds"}},
        ["query"]),
    _fn("vss_segment", "One caption segment (full caption, camera time, visibility tag) by its source URI from "
        "vss_search.", {"source": {"type": "string"}}, ["source"]),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
TOOL_PARAMS = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in TOOLS}
Chat = Callable[..., dict[str, Any]]


# ----------------------------------------------------------------------------- small helpers

def mmss(t: float | None) -> str | None:
    """mm:ss, floored like the UI's clock."""
    if t is None:
        return None
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def r2(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def reason_word(reason: str | None) -> str:
    first = str(reason or "blind").split("+")[0].strip().replace(" ", "_")
    return REASON_WORDS.get(first, first.replace("_", " "))


_SECRETS: set[str] = set()
_SCRUB_RE = re.compile(r"(Bearer\s+|Basic\s+|token=|api_key=|key=)[^\s&\"']+|eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}",
                       re.I)


def scrub(text: Any) -> str:
    """Error text without any secret: registered keys, bearer / basic values, token query params, JWTs."""
    s = str(text)
    for sec in _SECRETS:
        if sec:
            s = s.replace(sec, "<redacted>")
    return _SCRUB_RE.sub(lambda m: (m.group(1) or "") + "<redacted>", s)[:300]


@lru_cache(maxsize=64)
def _json_cached(path: str, mtime: float) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_json(p: Path) -> Any:
    try:
        return _json_cached(str(p), p.stat().st_mtime)
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- W&B config + HTTP

@dataclass(frozen=True)
class WandbConfig:
    key: str
    team: str
    project: str

    def __repr__(self) -> str:
        return f"WandbConfig(team={self.team!r}, project={self.project!r}, key=<redacted>)"

    __str__ = __repr__

    @property
    def project_id(self) -> str:
        return f"{self.team}/{self.project}"


def _env_file(path: str) -> dict[str, str]:
    """KEY=VALUE lines of an env file, only the W&B keys, parsed in memory."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.replace("export ", "").strip()
        if k in WANDB_KEYS:
            out[k] = v.strip().strip('"').strip("'")
    return out


def load_config(env_file: str = ENV_FILE) -> WandbConfig | None:
    """W&B Inference settings from the environment, else /etc/environment. None when any is missing."""
    vals = {k: os.environ.get(k, "") for k in WANDB_KEYS}
    if not all(vals.values()):
        vals = {**_env_file(env_file), **{k: v for k, v in vals.items() if v}}
    if not all(vals.get(k) for k in WANDB_KEYS):
        return None
    _SECRETS.add(vals["WANDB_API_KEY"])
    return WandbConfig(vals["WANDB_API_KEY"], vals["WANDB_TEAM"], vals["WANDB_PROJECT"])


class LLMError(RuntimeError):
    """W&B Inference unreachable, refused the call or sent something unusable."""

    def __init__(self, msg: str, status: int | None = None):
        super().__init__(scrub(msg))
        self.status = status


def post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
    req = urllib.request.Request(url, json.dumps(body, default=str).encode("utf-8"),
                                 {"Content-Type": "application/json", "Accept": "application/json",
                                  "User-Agent": "stm-agent/1", **headers})
    try:
        with urllib.request.urlopen(req, timeout=max(0.2, timeout)) as r:
            raw = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read(300).decode("utf-8", "replace") if e.fp else ""
        raise LLMError(f"HTTP {e.code}: {detail}", e.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"{type(e).__name__}: {getattr(e, 'reason', e)}") from None
    try:
        return json.loads(raw) if raw.strip() else {}
    except ValueError:
        raise LLMError("reply is not JSON") from None


def _template_kwargs(model: str) -> dict[str, Any] | None:
    """Switch thinking off where the chat template supports it (reasoning models spend max_tokens on it)."""
    m = model.lower()
    return {"enable_thinking": False} if ("glm" in m or "qwen" in m) else None


def wandb_chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
               tool_choice: str = "auto", timeout: float = BUDGET_S,
               cfg: WandbConfig | None = None) -> dict[str, Any]:
    """One chat completion on W&B Inference. Returns {content, tool_calls, finish_reason, usage}."""
    cfg = cfg or load_config()
    if cfg is None:
        raise LLMError("no W&B config (WANDB_API_KEY / WANDB_TEAM / WANDB_PROJECT)")
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": MAX_TOKENS}
    if tools:
        body["tools"], body["tool_choice"] = tools, tool_choice
    extra = _template_kwargs(model)
    if extra:
        body["chat_template_kwargs"] = extra
    headers = {"Authorization": f"Bearer {cfg.key}", "OpenAI-Project": cfg.project_id}
    t0 = time.perf_counter()
    try:
        resp = post_json(API_URL, body, headers, timeout)
    except LLMError as e:
        if not extra or e.status != 400:
            raise
        body.pop("chat_template_kwargs")        # template does not take it: ask again without
        resp = post_json(API_URL, body, headers, timeout - (time.perf_counter() - t0))
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if not choices:
        raise LLMError("no choices in the reply")
    msg = choices[0].get("message") or {}
    return {"content": msg.get("content"), "tool_calls": msg.get("tool_calls") or [],
            "finish_reason": choices[0].get("finish_reason"), "usage": resp.get("usage") or {}}


# ----------------------------------------------------------------------------- question guards

def refusal(question: str) -> str | None:
    """'identity' or 'blame' when the question asks who someone is or who did something, else None."""
    q = question.lower().replace("\u2019", "'")
    if IDENTITY_RE.search(q):
        return "identity"
    if BLAME_RE.search(q):
        return "blame"
    return None


def _clock(m: re.Match) -> float:
    a, b, c = m.group(1), m.group(2), m.group(3)
    return int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)


def _unit_s(u: str) -> float:
    return 3600.0 if u.startswith("h") else 60.0 if u.startswith("m") else 1.0


def _days_named(ql: str, days: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Days the question names: their labels ('1 Sep', 'Day A'), long ids, 'first day', 'yesterday', 'today'."""
    hit = []
    for i, d in enumerate(days):
        names = {str(d.get("label") or "").lower()}
        if len(str(d.get("id") or "")) >= 3:
            names.add(str(d["id"]).lower())
        if len(days) >= 2:
            names |= {f"day {i + 1}", f"day {chr(97 + i)}"} | ({f"{ORDINALS[i]} day"} if i < len(ORDINALS) else set())
            if i == len(days) - 1:
                names |= {"today", "last day"}
            if i == len(days) - 2:
                names.add("yesterday")
        if any(n and re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", ql) for n in names):
            hit.append(d)
    return hit


def asked_range(question: str, t: float | None, dur: float, days: list[dict[str, Any]]) -> tuple[float, float, str]:
    """The stretch of recording the question is about: (start, end, why). Plain rules, never an LLM."""
    ql = question.lower()
    now = dur if t is None else min(max(0.0, float(t)), dur)
    clip = lambda a, b: (r2(max(0.0, min(a, dur))), r2(max(0.0, min(b, dur))))  # noqa: E731
    m = LAST_RE.search(ql)
    if m:
        a, b = clip(now - float(m.group(1)) * _unit_s(m.group(2)), now)
        return a, b, "last " + m.group(1) + " " + m.group(2)
    hits = sorted([(x.start(), _clock(x)) for x in TIME_RE.finditer(ql)]
                  + [(x.start(), float(x.group(1))) for x in SECS_RE.finditer(ql)])
    if len(hits) >= 2:
        a, b = sorted(v for _, v in hits[:2])
        return (*clip(a, b), "between two times")
    if len(hits) == 1:
        pos, x = hits[0]
        before = re.findall(r"\b(since|after|from|before|until|till|by|at|around)\b", ql[:pos])
        word = before[-1] if before else "at"
        if word in ("since", "after", "from"):
            return (*clip(x, now if now > x else dur), f"{word} {mmss(x)}")
        if word in ("before", "until", "till", "by"):
            return (*clip(0.0, x), f"{word} {mmss(x)}")
        return (*clip(x - POINT_HALF, x + POINT_HALF), f"around {mmss(x)}")
    named = _days_named(ql, days)
    if named:
        a, b = min(float(d["start"]) for d in named), max(float(d["end"]) for d in named)
        return (*clip(a, b), "day " + " + ".join(str(d.get("label") or d.get("id")) for d in named))
    if NOW_RE.search(ql):
        return (*clip(now - 2 * POINT_HALF, now), f"now ({mmss(now)})")
    return (*clip(0.0, now), f"up to {mmss(now)}")


# ----------------------------------------------------------------------------- visibility (ledger.json)

def second_map(ledger: dict[str, Any]) -> dict[int, str | None]:
    """Second -> None (visible) or the blind reason. per_second when present, else the blind spans."""
    per = ledger.get("per_second")
    if per:
        return {int(p["t"]): (None if p.get("visible") else str(p.get("reason") or "blind")) for p in per}
    m: dict[int, str | None] = {s: None for s in range(int(ledger.get("seconds") or 0))}
    for b in ledger.get("blind") or []:
        for s in range(int(b["start"]), int(math.ceil(float(b["end"])))):
            if s in m:
                m[s] = str(b.get("reason") or "blind")
    return m


def visibility(secmap: dict[int, str | None] | None, start: float, end: float) -> dict[str, Any]:
    """Seen / blind seconds in [start, end) by whole ledger seconds; seconds outside the recording count as blind."""
    a, b = sorted((float(start), float(end)))
    base = {"start": r2(a), "end": r2(b), "range": f"{mmss(a)}-{mmss(b)}"}
    if secmap is None:
        return {**base, "known": False, "note": "no ledger.json for this camera: what it could see is unknown"}
    s0 = int(math.floor(a))
    secs = range(s0, max(s0 + 1, int(math.ceil(b))))
    seen = 0
    reasons: dict[str, int] = {}
    spans: list[dict[str, Any]] = []
    for s in secs:
        r = secmap[s] if s in secmap else "not_recorded"
        if r is None:
            seen += 1
            continue
        reasons[r] = reasons.get(r, 0) + 1
        if spans and spans[-1]["reason"] == r and spans[-1]["end"] == s:
            spans[-1]["end"] = s + 1
        else:
            spans.append({"start": s, "end": s + 1, "reason": r})
    n = len(secs)
    blind = n - seen
    for sp in spans:
        sp.update(start=float(sp["start"]), end=float(sp["end"]), why=reason_word(sp["reason"]),
                  span=f"{mmss(sp['start'])}-{mmss(sp['end'])}")
    return {**base, "known": True, "seconds": n, "seen_s": seen, "blind_s": blind,
            "blind_pct": round(100.0 * blind / n) if n else 0, "blind_frac": round(blind / n, 3) if n else 0.0,
            "reasons": {reason_word(k): v for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])},
            "spans": spans[:12], "more_spans": max(0, len(spans) - 12)}


def describe_blind(vis: dict[str, Any], n: int = 3) -> str:
    """'picture frozen 05:16-07:04; blurred 19:21-21:11 and 2 more'."""
    sp = vis.get("spans") or []
    s = "; ".join(f"{x['why']} {x['span']}" for x in sp[:n])
    more = len(sp) - n + int(vis.get("more_spans") or 0)
    return s + (f" and {more} more" if more > 0 else "")


# ----------------------------------------------------------------------------- rule engine fallback

def _norm(s: str | None) -> str:
    """Same normalisation as the server's qa.json match: lower case, letters/digits/spaces only."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (s or "").lower())).strip()


def qa_lookup(cam_dir: Path, question: str, t: float | None) -> dict[str, Any] | None:
    """Stored answer from qa.json: exact question first, else the closest wording with the same ids and times."""
    qa = read_json(cam_dir / "qa.json")
    items = (qa.get("items") if isinstance(qa, dict) else qa) or []
    want = _norm(question)
    wtoks, wnums = set(want.split()), set(NUM_RE.findall(question))
    best, best_s = None, 0.0
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("answer"), dict):
            continue
        q = it.get("question") or it.get("q") or ""
        it_t = it.get("t")
        if t is not None and it_t is not None and abs(float(it_t) - float(t)) > 0.5:
            continue
        if _norm(q) == want:
            return {**it["answer"], "matched_question": q}
        toks = set(_norm(q).split())
        score = len(wtoks & toks) / max(1, len(wtoks | toks))
        if set(NUM_RE.findall(q)) == wnums and score > best_s:
            best, best_s = it, score
    if best is not None and best_s >= 0.6:
        q = best.get("question") or best.get("q")
        return {**best["answer"], "matched_question": q, "match_score": round(best_s, 2)}
    return None


def _import_ask() -> Callable[..., dict[str, Any]] | None:
    try:
        return importlib.import_module("stm.ask").answer
    except Exception:
        return None


def warm_ask() -> None:
    """Import stm.ask in the background (cv2 + pandas take ~2 s cold) while the first LLM call is in flight."""
    if "stm.ask" not in sys.modules:
        threading.Thread(target=_import_ask, daemon=True).start()


def rule_answer(cam: str, question: str, t: float | None, cam_dir: Path,
                ask_fn: Callable[..., dict[str, Any]] | None = None) -> tuple[dict[str, Any], str]:
    """The rule engine's answer: stm.ask when it can run here, else qa.json, else an honest UNKNOWN."""
    fn = ask_fn or _import_ask()
    err = None
    if fn is not None:
        try:
            return dict(fn(cam, question, t)), "stm.ask"
        except Exception as e:      # camera not on this machine, engine mid-edit, ...
            err = f"{type(e).__name__}: {scrub(e)}"
    hit = qa_lookup(cam_dir, question, t)
    if hit is not None:
        return hit, "qa.json"
    return ({"type": "unknown", "answer": "UNKNOWN", "evidence": [], "last_seen": None, "blind": [],
             "refused": None, "honesty": "no rule engine on this machine" + (f" ({err})" if err else ""),
             "sentence": "I can't answer that from this camera's files. Try one of the sample questions."}, "none")


# ----------------------------------------------------------------------------- tools

def _coerce(name: str, args: Any) -> dict[str, Any]:
    """Tool args as a dict of the declared params, numbers coerced (Llama sends '120' for 120)."""
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        return {}
    props = TOOL_PARAMS.get(name, {})
    out: dict[str, Any] = {}
    for k, spec in props.items():
        v = args.get(k)
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none")):
            continue
        typ = spec.get("type")
        try:
            if typ == "integer":
                v = int(float(str(v).strip().lstrip("#")))
            elif typ == "number":
                v = float(str(v).strip())
            elif typ == "string":
                v = str(v)
        except ValueError:
            continue
        out[k] = v
    return out


def _flex_call(fn: Callable[..., Any], first: Any, **opt: Any) -> Any:
    """fn(first, **only the options its signature names): aliases such as camera_id / cam can be offered together,
    and nothing unknown is pushed into a **kw (stm.vss.search forwards **kw to its request builder)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(first)
    return fn(first, **{k: v for k, v in opt.items() if k in params and params[k].kind != params[k].VAR_KEYWORD})


def _hits(resp: Any) -> list[dict[str, Any]]:
    if isinstance(resp, list):
        return [h for h in resp if isinstance(h, dict)]
    if isinstance(resp, dict):
        for k in ("hits", "results", "segments", "items"):
            if isinstance(resp.get(k), list):
                return [h for h in resp[k] if isinstance(h, dict)]
    return []


def _num(x: Any) -> float | None:
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


STOP = {"a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "is", "was", "were", "are", "any", "there",
        "did", "do", "does", "see", "seen", "show", "me", "when", "what", "which", "with", "by", "for", "it", "this",
        "that", "camera", "video", "time", "times", "ever", "come", "came", "go", "went"}


def _words(text: str) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", text.lower())} - STOP


def idf_rank(rows: list[dict[str, Any]], query: str, k: int, need: float = 0.6) -> list[dict[str, Any]]:
    """Caption rows ranked by the IDF-weighted share of query words they contain: a rare word ('van') must be there,
    common ones ('street', 'parked') barely count. Keeps rows scoring >= need."""
    docs = [_words(str(r.get("reasoning_content") or "")) for r in rows]
    qw = _words(query)
    if not qw or not docs:
        return []
    df = {w: sum(1 for d in docs if w in d) for w in qw}
    wt = {w: math.log((len(docs) + 1) / (df[w] + 0.5)) for w in qw}
    tot = sum(wt.values()) or 1.0
    scored = sorted(((sum(wt[w] for w in qw & d) / tot, i) for i, d in enumerate(docs)), key=lambda x: (-x[0], x[1]))
    return [{**rows[i], "similarity_score": round(sc, 2)} for sc, i in scored if sc >= need][:k]


class Toolbox:
    """The tools of one answer, bound to one camera's out/ folder. Keeps every call's full result."""

    def __init__(self, cam: str, out: Path, t: float | None, ask_fn: Callable[..., dict[str, Any]] | None = None,
                 vss: Any = None):
        self.cam, self.dir, self.t, self.ask_fn, self._vss_mod = cam, out / cam, t, ask_fn, vss
        self.meta = read_json(self.dir / "meta.json") or {}
        led = read_json(self.dir / "ledger.json")
        self.ledger = led if isinstance(led, dict) else None
        self.secmap = second_map(self.ledger) if self.ledger else None
        self.dur = float(self.meta.get("duration") or (self.ledger or {}).get("seconds") or 0.0)
        self.days = list(self.meta.get("days") or [])
        self.calls: list[dict[str, Any]] = []
        self._vss_given = vss is not None
        self._offs: dict[str, float] | None = None
        self._offs_dir: Path | None = None
        self.deadline: float | None = None     # perf_counter time the answer must be done by (network tools)
        self.window: tuple[float, float] | None = None     # the asked range: caption hits outside it are not evidence

    # -- dispatch
    def run(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        if name not in TOOL_NAMES:
            res: dict[str, Any] = {"error": f"unknown tool {name}"}
        elif name.startswith("vss_") and self.deadline is not None:
            res = self._bounded(name, args, self.deadline - t0)
        else:
            res = self._safe(name, args)
        self.calls.append({"tool": name, "args": args, "result": res})
        return res, _ms(t0)

    def _safe(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            return getattr(self, name)(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {scrub(e)}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {scrub(e)}"}

    def _bounded(self, name: str, args: dict[str, Any], left: float) -> dict[str, Any]:
        """A network tool inside what is left of the budget (stm.vss itself waits up to 60 s and retries)."""
        box: dict[str, Any] = {}
        th = threading.Thread(target=lambda: box.update(res=self._safe(name, args)), daemon=True)
        th.start()
        th.join(max(0.05, left))
        return box.get("res") or {"error": f"{name} did not answer within the time budget"}

    # -- ledger
    def stm_visibility(self, start: float, end: float) -> dict[str, Any]:
        return visibility(self.secmap, start, end)

    # -- rule engine
    def stm_ask(self, question: str) -> dict[str, Any]:
        res, src = rule_answer(self.cam, question, self.t, self.dir, self.ask_fn)
        return {**res, "source": src}

    # -- near.json
    def _target(self, x: dict[str, Any]) -> dict[str, Any]:
        vis = [{"object_id": v.get("object_id"), "name": v.get("name"), "closest_t": v.get("closest_t"),
                "closest_mmss": mmss(_num(v.get("closest_t"))), "gap_rel": v.get("gap_rel"),
                "enter_t": v.get("enter_t"), "exit_t": v.get("exit_t"), "crop": v.get("crop")}
               for v in (x.get("visitors") or [])]
        return {"object_id": x.get("object_id"), "name": x.get("name"), "box": x.get("box"),
                "start": x.get("start"), "end": x.get("end"), "start_mmss": mmss(_num(x.get("start"))),
                "end_mmss": mmss(_num(x.get("end"))), "crop": x.get("crop"), "summary": x.get("summary"),
                "n_visitors": len(vis), "visitors": vis[:8], "blind": (x.get("blind") or [])[:6]}

    def stm_near(self, object_id: int | None = None, t: float | None = None) -> dict[str, Any]:
        near = read_json(self.dir / "near.json")
        if not isinstance(near, dict):
            return {"error": "no near.json for this camera"}
        targets = [x for x in near.get("targets") or [] if isinstance(x, dict)]
        if object_id is not None:
            hit = [x for x in targets if int(x.get("object_id", -1)) == object_id] or \
                  [x for x in targets if any(int(v.get("object_id", -1)) == object_id for v in x.get("visitors") or [])]
            how = f"object #{object_id}"
        elif t is not None:
            hit = [x for x in targets if float(x.get("start", 0)) <= t <= float(x.get("end", 0))]
            how = f"parked at {mmss(t)}"
        else:
            hit = sorted(targets, key=lambda x: -len(x.get("visitors") or []))
            how = "most visited"
        return {"matched": len(hit), "of_targets": len(targets), "selected": how,
                "targets": [self._target(x) for x in hit[:3]],
                "note": "near is not doing: this shows who was close, never who did anything"}

    # -- diff.json
    def stm_diff(self) -> dict[str, Any]:
        d = read_json(self.dir / "diff.json")
        if not isinstance(d, dict):
            return {"error": "no diff.json for this camera (one day only, or not built)"}

        def side(s: Any) -> Any:
            return {k: s.get(k) for k in ("object_id", "t", "crop")} | {"t_mmss": mmss(_num(s.get("t")))} \
                if isinstance(s, dict) else None
        ch = [{"kind": c.get("kind"), "box": c.get("box"), "a": side(c.get("a")), "b": side(c.get("b")),
               "unconfirmed": bool(c.get("unconfirmed")), "similarity": c.get("similarity"),
               "note": str(c.get("note") or "")[:220]} for c in d.get("changes") or [] if isinstance(c, dict)]
        return {"day_a": d.get("day_a"), "day_b": d.get("day_b"), "counts": d.get("counts"),
                "infrared": d.get("infrared"), "unconfirmed": d.get("unconfirmed"), "parked": d.get("parked"),
                "alignment_ok": (d.get("alignment") or {}).get("ok"), "note": d.get("note"),
                "simulated": self.meta.get("simulated") or [], "changes": ch[:10],
                "not_compared": [str(x.get("reason") or "")[:160] for x in d.get("not_compared") or []
                                 if isinstance(x, dict)][:5]}

    # -- VSS captions
    def _vss(self) -> Any:
        if self._vss_mod is None:
            try:
                self._vss_mod = importlib.import_module("stm.vss")
            except Exception:
                self._vss_mod = False
        return self._vss_mod or None

    def _live_search(self, mod: Any) -> bool:
        """The team VSS /search is a POST: used only for an injected module (tests) or with STM_VSS_LIVE=1."""
        return mod is not None and hasattr(mod, "search") and (self._vss_given or os.environ.get(LIVE_VSS_ENV) == "1")

    def _candidates(self, name: str, env: str) -> list[Path]:
        """Existing copies of a Pack D file, in order: beside the camera's outputs, $env, then PACKD_DIRS."""
        ps = [self.dir / name] + ([Path(os.environ[env])] if os.environ.get(env) else []) + \
             [d.expanduser() / name for d in PACKD_DIRS]
        return [p for p in dict.fromkeys(ps) if p.is_file()]

    def _offsets(self) -> dict[str, float]:
        """Chunk name -> its start on the camera clock, from the chunk_offsets.json of the layout this camera's video
        uses. A file whose `total` is within the tolerance of the video length wins (closest first); a file without a
        total is used only when none matches; a file whose total is off belongs to another build and is never used
        (whole-file offsets on the seam-safe video would put captions up to 7 min off)."""
        if self._offs is None:
            tol = max(OFFSETS_TOL_S, 0.01 * self.dur)
            match: list[tuple[float, int, Path, dict]] = []
            untotalled: list[tuple[int, Path, dict]] = []
            for i, p in enumerate(self._candidates("chunk_offsets.json", "STM_CHUNK_OFFSETS")):
                d = read_json(p)
                offs = d.get("offsets") if isinstance(d, dict) else None
                if not isinstance(offs, dict) or not offs:
                    continue
                tot = _num(d.get("total"))
                if tot is None or self.dur <= 0:
                    untotalled.append((i, p, offs))
                elif abs(tot - self.dur) <= tol:
                    match.append((abs(tot - self.dur), i, p, offs))
            best = min(match, key=lambda x: (x[0], x[1]))[1:] if match else (untotalled[0] if untotalled else None)
            self._offs_dir = best[1].parent if best else None
            self._offs = {str(k): float(v) for k, v in (best[2] if best else {}).items()}
        return self._offs

    def _local_rows(self) -> list[dict[str, Any]]:
        self._offsets()
        ps = sorted(self._candidates("segments.json", "STM_SEGMENTS"), key=lambda p: p.parent != self._offs_dir)
        d = read_json(ps[0]) if ps else None
        rows = _hits(d)
        return [r for r in rows if str(r.get("camera_id") or self.cam) == self.cam]

    def cam_time(self, hit: dict[str, Any]) -> tuple[float | None, float | None]:
        """A hit's start / end on this camera's clock: its own camera-time fields, else chunk offset + segment time."""
        for ks, ke in (("cam_start", "cam_end"), ("camera_start", "camera_end"), ("t_start", "t_end"),
                       ("cam_t", "cam_t_end"), ("camera_t", "camera_t_end")):
            s = _num(hit.get(ks))
            if s is not None:
                e = _num(hit.get(ke))
                return s, (e if e is not None else s)
        offs = self._offsets()
        ss = _num(hit.get("segment_start_sec", hit.get("start_sec")))      # stm.vss hits: start_sec / end_sec
        se = _num(hit.get("segment_end_sec", hit.get("end_sec")))
        if not offs or ss is None:
            return None, None
        names = [str(hit.get("filename") or ""), str(hit.get("source") or ""), str(hit.get("original_video") or "")]
        for n in names:
            stem = n.rsplit("/", 1)[-1].split("_segment_")[0].rsplit(".", 1)[0]
            if stem in offs:
                return offs[stem] + ss, offs[stem] + (se if se is not None else ss)
        return None, None

    def place(self, hit: dict[str, Any], full: bool = False,
              window: tuple[float, float] | None = None) -> dict[str, Any]:
        """One caption hit on the camera clock, tagged 'evidence' or 'not evidence: <reason>' by the ledger and the
        asked time window."""
        start, end = self.cam_time(hit)
        cap = str(hit.get("reasoning_content") or hit.get("caption") or hit.get("text") or "")
        row = {"source": hit.get("source") or hit.get("segment_source"), "t": r2(start), "end": r2(end),
               "t_mmss": mmss(start), "similarity": r2(_num(hit.get("similarity_score") or hit.get("score"))),
               "caption": cap if full else cap[:260]}
        other = hit.get("camera_id")
        if other and str(other) != self.cam:
            row["tag"] = f"not evidence: another camera ({other})"
        elif start is None:
            row["tag"] = "not evidence: can't place this hit on the camera clock"
        elif window and ((end if end is not None else start) < window[0] or start > window[1]):
            row["tag"] = f"not evidence: outside the asked time {mmss(window[0])}-{mmss(window[1])}"
        else:
            vis = visibility(self.secmap, start, end if end and end > start else start)
            if not vis["known"]:
                row["tag"] = "evidence (no ledger: visibility unknown)"
            elif vis["blind_frac"] >= BLIND_FORCE:
                row["tag"] = f"not evidence: {describe_blind(vis, 2)}"
            elif vis["blind_s"]:
                row["tag"] = f"evidence (partly blind: {vis['blind_s']} of {vis['seconds']} s)"
            else:
                row["tag"] = "evidence"
        return row

    def _window(self, start: float | None, end: float | None) -> tuple[float, float] | None:
        """The time window caption hits must fall in: the LLM's start / end, else the asked range; None = all."""
        lo, hi = (start, end) if (start is not None or end is not None) else (self.window or (None, None))
        lo, hi = (0.0 if lo is None else float(lo)), (self.dur if hi is None else float(hi))
        return (lo, hi) if (lo > 0.5 or hi < self.dur - 0.5) else None

    def vss_search(self, query: str, top_k: int = 5, start: float | None = None,
                   end: float | None = None) -> dict[str, Any]:
        k = max(1, min(10, int(top_k or 5)))
        win = self._window(start, end)
        mod = self._vss()
        raw: list[dict[str, Any]] | None = None
        live_err = None
        if self._live_search(mod):
            try:
                raw = _hits(_flex_call(mod.search, query, camera_id=self.cam, cam=self.cam,
                                       top_k=min(10, 3 * k) if win else k))     # no time filter server-side: ask more
                backend = "vss"
            except Exception as e:      # no VSS config on this machine, network down: the stored captions answer
                live_err = f"{type(e).__name__}: {scrub(e)}"
        if raw is None:
            rows = self._local_rows()
            if not rows:
                return {"error": "caption search is not available on this machine (no live VSS search, no "
                                 "segments.json)" + (f"; live VSS failed: {live_err}" if live_err else "")}
            if win:
                rows = [r for r in rows if (lambda a, b: a is not None and b is not None and b >= win[0]
                                            and a <= win[1])(*self.cam_time(r))]
            raw = idf_rank(rows, query, k)
            backend = "local captions (keyword match on stored VSS rows, no embedding)" + (
                f"; live VSS failed: {live_err}" if live_err else "")
        placed = [self.place(h, window=win) for h in raw]
        inside = [h for h in placed if "outside the asked time" not in h["tag"]]
        hits = inside[:k] + [h for h in placed if h not in inside][:max(0, k - len(inside))]
        ok = [h for h in hits if h["tag"].startswith("evidence")]
        return {"query": query, "backend": backend, "window": f"{mmss(win[0])}-{mmss(win[1])}" if win else "all",
                "n_hits": len(hits), "n_evidence": len(ok), "n_not_evidence": len(hits) - len(ok), "hits": hits}

    def vss_segment(self, source: str) -> dict[str, Any]:
        mod = self._vss()
        seg = None
        fn = (getattr(mod, "segment_meta", None) or getattr(mod, "segment", None)) if mod is not None else None
        if fn is not None:
            try:
                seg = _flex_call(fn, source)
            except Exception:           # no VSS config here or the backend is down: try the stored rows
                seg = None
            if isinstance(seg, dict) and isinstance(seg.get("segment"), dict):
                seg = seg["segment"]
        if not isinstance(seg, dict):
            seg = next((r for r in self._local_rows() if r.get("source") == source), None)
        if not isinstance(seg, dict):
            return {"error": f"segment not found: {source}"}
        return {"segment": self.place(seg, full=True)}


# ----------------------------------------------------------------------------- the LLM loop

def _tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Calls from a reply: the standard tool_calls, or one JSON call written into the content (Llama does that)."""
    out = []
    for i, c in enumerate(msg.get("tool_calls") or []):
        fn = c.get("function") or {}
        out.append({"id": str(c.get("id") or f"call_{i}"), "name": str(fn.get("name") or ""),
                    "raw": fn.get("arguments")})
    if out:
        return out
    text = str(msg.get("content") or "").strip()
    ds = [{"id": f"call_{i}", "name": m.group(1), "raw": m.group(2)} for i, m in enumerate(DS_CALL_RE.finditer(text))
          if m.group(1) in TOOL_NAMES]
    if ds:
        return ds
    text = re.sub(r"^<\|python_tag\|>|^```(?:json)?|```$", "", text).strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            j = json.loads(text)
        except ValueError:
            return []
        if isinstance(j, dict) and j.get("name") in TOOL_NAMES:
            return [{"id": "call_0", "name": j["name"], "raw": j.get("parameters") or j.get("arguments") or {}}]
    return []


def clean_sentence(text: Any) -> str:
    """Plain sentence(s): no thinking block, no markdown, at most ~500 characters cut at a sentence end."""
    s = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    if DS_CALL_RE.search(s) or re.search(r"tool[_\u2581]call", s):
        return ""                               # a tool call written as text is not an answer
    s = MARKUP_RE.sub("", s)                    # stray template tokens (<|eot_id|> and the like)
    s = re.sub(r"^\s*(?:#+|>+|[-*\u2022]|\d+[.)])\s+", "", s, flags=re.M)     # headings, quotes, list bullets
    s = re.sub(r"[*`]+", "", s)
    s = " ".join(s.split())
    if len(s) > 500:
        cut = s[:500]
        s = cut[:cut.rfind(". ") + 1] if ". " in cut else cut
    return s


def _numnorm(n: str) -> str:
    if ":" in n:
        return ":".join([str(int(n.split(":")[0]))] + n.split(":")[1:])
    x = float(n)
    return str(int(x)) if x == int(x) else f"{x:.2f}".rstrip("0").rstrip(".")


def allowed_numbers(texts: list[str]) -> set[str]:
    """Every number the tools (and the question context) gave, with the roundings and mm:ss a writer may use."""
    out: set[str] = set()
    for s in texts:
        for n in NUM_RE.findall(s):
            out.add(_numnorm(n))
            if ":" not in n:
                x = float(n)
                out |= {_numnorm(str(int(x))), _numnorm(str(round(x))), _numnorm(f"{x:.1f}"), _numnorm(mmss(x))}
    return out


def unsupported_numbers(sentence: str, allowed: set[str]) -> list[str]:
    return [n for n in NUM_RE.findall(sentence) if _numnorm(n) not in allowed]


def contradicts(sentence: str, answer: str) -> bool:
    """The sentence opens with yes / no against the tool's YES / NO / UNKNOWN, or with yes against a zero count."""
    first = (re.findall(r"[a-z']+", sentence.lower()) or [""])[0]
    a = str(answer).upper()
    zero = (a.split() or [""])[0] == "0"
    return (first == "yes" and (a in ("NO", "UNKNOWN", CANT_TELL) or zero)) or (first == "no" and a == "YES")


def _context(cam: str, box: Toolbox, q: str, t: float | None, rng: tuple[float, float, str],
             vis: dict[str, Any]) -> str:
    days = "; ".join(f"{d.get('label') or d.get('id')} {mmss(_num(d.get('start')))}-{mmss(_num(d.get('end')))}"
                     for d in box.days)
    lines = [f"Camera {cam} ({box.meta.get('title') or 'street camera'}). Recording 00:00-{mmss(box.dur)} "
             f"({box.dur:.0f} s). Days: {days or 'one'}."]
    if box.meta.get("simulated"):
        lines.append("SIMULATED: " + "; ".join(map(str, box.meta["simulated"])))
    lines.append(f"Question asked at: {'end of recording' if t is None else mmss(t)} "
                 f"({box.dur if t is None else t:.0f} s).")
    if vis.get("known"):
        lines.append(f"Time range the question is about: {vis['range']} ({rng[2]}); camera blind {vis['blind_s']} of "
                     f"{vis['seconds']} s there ({vis['blind_pct']}%){': ' + describe_blind(vis) if vis['spans'] else ''}.")
    else:
        lines.append(f"Time range the question is about: {vis['range']} ({rng[2]}); visibility unknown (no ledger).")
    lines.append(f"Question: {q}")
    return "\n".join(lines)


def llm_loop(box: Toolbox, messages: list[dict[str, Any]], chat: Chat, models: tuple[str, ...], deadline: float,
             trace: list[dict[str, Any]], t_start: float) -> tuple[str | None, str | None, str | None]:
    """Up to MAX_ROUNDS tool rounds. Returns (sentence, model, why it stopped without one)."""
    mi, rounds, model = 0, 0, None
    while True:
        left = deadline - time.perf_counter()
        if left <= 0.25:
            return None, model, "time budget used up"
        final = rounds >= MAX_ROUNDS
        model = models[mi]
        t0, at = time.perf_counter(), _ms(t_start)
        try:
            msg = chat(model=model, messages=messages, tools=TOOLS, tool_choice="none" if final else "auto",
                       timeout=left)
        except Exception as e:      # LLMError, or anything a fake / proxy throws
            trace.append({"tool": "llm", "args": {"round": rounds + 1}, "latency_ms": _ms(t0), "model": model,
                          "at_ms": at, "ok": False, "error": scrub(e)})
            mi += 1
            if mi >= len(models):
                return None, None, f"W&B Inference unreachable ({scrub(e)})"
            continue
        calls = [] if final else _tool_calls(msg)
        usage = msg.get("usage") or {}
        trace.append({"tool": "llm", "args": {"round": rounds + 1, "tool_choice": "none" if final else "auto"},
                      "latency_ms": _ms(t0), "model": model, "at_ms": at, "ok": True,
                      "tool_calls": [c["name"] for c in calls], "tokens": usage.get("total_tokens")})
        if not calls:
            text = clean_sentence(msg.get("content"))
            if text and not (final and _tool_calls(msg)):
                return text, model, None
            return None, model, (f"{MAX_ROUNDS} tool rounds used up" if final else "model returned no text")
        rounds += 1
        norm = [{"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(_coerce(c["name"], c["raw"]))}}
                for c in calls]
        messages.append({"role": "assistant", "content": "", "tool_calls": norm})
        for c in calls:
            args, at = _coerce(c["name"], c["raw"]), _ms(t_start)
            res, ms = box.run(c["name"], args)
            trace.append({"tool": c["name"], "args": args, "latency_ms": ms, "model": model, "round": rounds,
                          "at_ms": at, "ok": "error" not in res,
                          **{k: res[k] for k in ("source", "backend", "error") if isinstance(res.get(k), str)}})
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                             "content": json.dumps(res, default=str, separators=(",", ":"))[:TOOL_CHARS]})
        if rounds >= MAX_ROUNDS:
            messages.append({"role": "user", "content": FINAL_NUDGE})


# ----------------------------------------------------------------------------- composing the answer

def _last(box: Toolbox, name: str, ok: Callable[[dict[str, Any]], bool] = lambda r: "error" not in r) -> Any:
    for c in reversed(box.calls):
        if c["tool"] == name and ok(c["result"]):
            return c
    return None


def primary(box: Toolbox) -> tuple[str, dict[str, Any]] | None:
    """The tool result the answer stands on: rule engine > near > diff > captions > visibility."""
    c = _last(box, "stm_ask", lambda r: "error" not in r and r.get("type") not in (None, "unknown"))
    if c is None:
        c = _last(box, "stm_near", lambda r: bool(r.get("targets")))
    for name in ("stm_diff", "vss_search", "stm_visibility"):
        c = c or _last(box, name)
    return (c["tool"], c) if c else None


def _from_tool(name: str, call: dict[str, Any]) -> dict[str, Any]:
    """type, answer, own sentence, honesty, last_seen, blind and refused, all read off one tool result."""
    r = call["result"]
    if name == "stm_ask":
        return {"type": r.get("type"), "answer": str(r.get("answer")), "sentence": r.get("sentence"),
                "honesty": r.get("honesty"), "last_seen": r.get("last_seen"), "blind": r.get("blind") or [],
                "refused": r.get("refused")}
    if name == "stm_near":
        tg = r["targets"][0]
        return {"type": "who_near", "answer": str(tg["n_visitors"]), "sentence": tg.get("summary") or
                f"{tg['n_visitors']} came near #{tg['object_id']} between {tg['start_mmss']} and {tg['end_mmss']}.",
                "honesty": "being near is not doing it", "blind": tg.get("blind") or []}
    if name == "stm_diff":
        n = r.get("counts") or {}
        ans = f"{n.get('arrived', 0)} arrived, {n.get('left', 0)} left, {n.get('same', 0)} stayed"
        hon = [f"{r['unconfirmed']} 'same' unconfirmed"] if r.get("unconfirmed") else []
        hon += [f"SIMULATED: {s}" for s in r.get("simulated") or []]
        return {"type": "what_changed", "answer": ans, "sentence": f"{r.get('day_a')} vs {r.get('day_b')}: {ans}.",
                "honesty": "; ".join(hon) or None}
    if name == "vss_search":
        ok = [h for h in r["hits"] if h["tag"].startswith("evidence")]
        at = ", ".join(h["t_mmss"] for h in ok[:4])
        s = (f"{len(ok)} caption hit(s) for '{r['query']}'" + (f" at {at}" if at else "") + "."
             + (f" {r['n_not_evidence']} more fell in blind time or off the camera clock and are not evidence."
                if r["n_not_evidence"] else ""))
        # the count is of caption hits, not of objects: a bare "10" under "was there a white van?" reads as 10 vans
        return {"type": "search", "answer": f"{len(ok)} caption hit{'' if len(ok) == 1 else 's'}", "sentence": s,
                "honesty": "captions are AI descriptions, not detections" + (
                    f"; {r['n_not_evidence']} hit(s) tagged not evidence" if r["n_not_evidence"] else "")}
    v = r
    if not v.get("known"):
        return {"type": "visibility", "answer": "UNKNOWN", "sentence": v.get("note"), "honesty": v.get("note")}
    ans = "YES" if v["blind_s"] == 0 else "NO" if v["seen_s"] == 0 else f"{v['seen_s']} of {v['seconds']} s seen"
    s = f"The camera saw {v['seen_s']} of {v['seconds']} s in {v['range']}" + (
        f"; blind {v['blind_s']} s ({describe_blind(v)})." if v["blind_s"] else ".")
    return {"type": "visibility", "answer": ans, "sentence": s, "honesty": None,
            "blind": [{k: x[k] for k in ("start", "end", "reason")} for x in v["spans"]]}


def evidence_of(box: Toolbox) -> list[dict[str, Any]]:
    """Evidence from every tool result, deduped: rule-engine sightings, near visitors, caption hits (tagged)."""
    ev: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(item: dict[str, Any]) -> None:
        key = (item.get("object_id"), item.get("t"), item.get("source"))
        if key not in seen and len(ev) < MAX_EVIDENCE:
            seen.add(key)
            ev.append(item)
    for c in box.calls:
        r = c["result"]
        if "error" in r:
            continue
        if c["tool"] == "stm_ask":
            for e in r.get("evidence") or []:
                add({**e, "from": "stm_ask"})
        elif c["tool"] == "stm_near":
            for tg in r.get("targets") or []:
                for v in tg["visitors"]:
                    add({"object_id": v["object_id"], "t": v["closest_t"], "crop": v["crop"], "box": None,
                         "from": "stm_near", "near": tg["object_id"]})
        elif c["tool"] == "stm_diff":
            for ch in r.get("changes") or []:
                for s in (ch.get("a"), ch.get("b")):
                    if s:
                        add({"object_id": s.get("object_id"), "t": s.get("t"), "crop": s.get("crop"),
                             "box": ch.get("box"), "from": "stm_diff", "kind": ch.get("kind")})
        elif c["tool"] in ("vss_search", "vss_segment"):
            for h in r.get("hits") or ([r["segment"]] if r.get("segment") else []):
                add({"object_id": None, "t": h["t"], "end": h["end"], "crop": None, "box": None,
                     "from": c["tool"], "source": h["source"], "caption": h["caption"][:200],
                     "similarity": h["similarity"], "tag": h["tag"]})
    return ev


def _join(*parts: Any) -> str | None:
    out = [str(p) for p in parts if p]
    return "; ".join(dict.fromkeys(out)) or None


def _blind_list(*lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out, seen = [], set()
    for lst in lists:
        for b in lst or []:
            key = (r2(_num(b.get("start"))), r2(_num(b.get("end"))))
            if key not in seen:
                seen.add(key)
                out.append({"start": key[0], "end": key[1], "reason": b.get("reason")})
    return sorted(out, key=lambda b: b["start"] or 0.0)


def _range_note(vis: dict[str, Any]) -> str | None:
    if not vis.get("known"):
        return vis.get("note")
    if not vis["blind_s"]:
        return None
    n = len(vis["spans"]) + int(vis.get("more_spans") or 0)
    why = describe_blind(vis) if n <= 3 else \
        ", ".join(f"{k} {v} s" for k, v in list(vis["reasons"].items())[:3]) + f", in {n} spans"
    return f"{vis['blind_s']} of {vis['seconds']} s of the asked range {vis['range']} were blind ({why})"


def _rules_result(box: Toolbox, cam: str, q: str, t: float | None, ask_fn: Any, why: str,
                  vis: dict[str, Any], trace: list[dict[str, Any]], t_start: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    res, src = rule_answer(cam, q, t, box.dir, ask_fn)
    trace.append({"tool": "rules", "args": {"question": q, "t": t, "why": why}, "latency_ms": _ms(t0),
                  "model": None, "at_ms": _ms(t_start), "source": src})
    return {"type": res.get("type") or "unknown", "answer": str(res.get("answer")), "sentence": res.get("sentence"),
            "evidence": res.get("evidence") or [], "honesty": _join(res.get("honesty"), _range_note(vis)),
            "blind": _blind_list(res.get("blind") or [], vis.get("spans") or []), "last_seen": res.get("last_seen"),
            "refused": res.get("refused"), "model": None, "engine": f"rules ({src})", "fallback": why}


def _compose(box: Toolbox, sentence: str | None, model: str | None, allowed: set[str], vis: dict[str, Any],
             trace: list[dict[str, Any]], t_start: float) -> dict[str, Any] | None:
    """The final answer from tool outputs; the LLM's sentence only if it passes the number and yes/no checks."""
    p = primary(box)
    if p is None:
        return None
    base = _from_tool(*p)
    note = None
    if base.get("refused"):
        sentence, note = None, "rule engine refused: its own sentence is used"
    elif sentence:
        bad = unsupported_numbers(sentence, allowed)
        if bad:
            sentence, note = None, f"LLM sentence had numbers no tool gave ({', '.join(bad[:4])})"
        elif contradicts(sentence, base["answer"]):
            sentence, note = None, f"LLM sentence contradicted the tool answer {base['answer']}"
    if note:
        trace.append({"tool": "guard:sentence", "args": {"primary": p[0]}, "latency_ms": 0.0, "model": model,
                      "at_ms": _ms(t_start), "note": note})
    return {"type": base["type"], "answer": base["answer"], "sentence": sentence or base["sentence"],
            "evidence": evidence_of(box), "honesty": _join(base.get("honesty"), _range_note(vis)),
            "blind": _blind_list(base.get("blind") or [], vis.get("spans") or []),
            "last_seen": base.get("last_seen"), "refused": base.get("refused"), "model": model,
            "engine": f"llm ({p[0]})", "fallback": None, "sentence_from": "llm" if sentence else p[0]}


# ----------------------------------------------------------------------------- answer()

def answer(cam: str, question: str, t: float | None = None, *, chat: Chat | None = None,
           models: tuple[str, ...] = MODELS, budget_s: float = BUDGET_S, out: Path | None = None,
           ask_fn: Callable[..., dict[str, Any]] | None = None, vss: Any = None) -> dict[str, Any]:
    """Answer one question about one camera. Same keys as stm.ask.answer plus trace[], model, engine, fallback."""
    t_start = time.perf_counter()
    root = out or OUT
    if not (root / cam).is_dir():
        raise KeyError(f"unknown camera {cam}")
    if ask_fn is None:
        warm_ask()
    q = " ".join(str(question or "").split())[:400]
    box = Toolbox(cam, root, t, ask_fn, vss)
    box.deadline = t_start + budget_s
    trace: list[dict[str, Any]] = []

    kind = refusal(q)
    if kind:
        trace.append({"tool": "guard:refuse", "args": {"kind": kind}, "latency_ms": 0.0, "model": None, "at_ms": 0.0})
        res = {"type": "refusal", "answer": "UNKNOWN", "sentence": REFUSAL, "evidence": [],
               "honesty": REFUSAL_HONESTY, "blind": [], "last_seen": None, "refused": kind, "model": None,
               "engine": "guard", "fallback": None}
        return _finish(cam, q, t, res, trace, t_start, None)

    t0 = time.perf_counter()
    rng = asked_range(q, t, box.dur, box.days)
    vis = visibility(box.secmap, rng[0], rng[1])
    trace.append({"tool": "guard:visibility", "args": {"start": rng[0], "end": rng[1], "why": rng[2]},
                  "latency_ms": _ms(t0), "model": None, "at_ms": _ms(t_start),
                  "blind_pct": vis.get("blind_pct"), "known": vis.get("known")})
    asked = {"start": rng[0], "end": rng[1], "why": rng[2], "blind_pct": vis.get("blind_pct")}
    box.window = (rng[0], rng[1])
    if vis.get("known") and vis["blind_frac"] >= BLIND_FORCE:
        why = describe_blind(vis)
        res = {"type": "visibility", "answer": CANT_TELL,
               "sentence": f"Can't tell: the camera could not see {vis['blind_pct']}% of {vis['range']} ({why}).",
               "evidence": [], "honesty": f"forced by the visibility guard: {vis['blind_s']} of {vis['seconds']} s "
                                         f"blind ({why}); any answer would be a guess",
               "blind": _blind_list(vis["spans"]), "last_seen": None, "refused": None, "model": None,
               "engine": "guard", "fallback": None}
        return _finish(cam, q, t, res, trace, t_start, asked)

    ctx = _context(cam, box, q, t, rng, vis)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx}]
    deadline = t_start + budget_s
    sentence, model, why = llm_loop(box, messages, chat or wandb_chat, models, deadline, trace, t_start)
    res = None
    if why is None or why == "model returned no text":
        allowed = allowed_numbers([ctx] + [m["content"] for m in messages if m.get("role") == "tool"])
        res = _compose(box, sentence, model, allowed, vis, trace, t_start)
        if res is None:
            why = why or "LLM answered without calling a tool"
    if res is None:
        res = _rules_result(box, cam, q, t, ask_fn, why or "no answer", vis, trace, t_start)
    return _finish(cam, q, t, res, trace, t_start, asked)


def _finish(cam: str, q: str, t: float | None, res: dict[str, Any], trace: list[dict[str, Any]], t_start: float,
            asked: dict[str, Any] | None) -> dict[str, Any]:
    out = {"type": res["type"], "answer": res["answer"], "sentence": res["sentence"],
           "latency_ms": _ms(t_start), "evidence": res.get("evidence") or [], "honesty": res.get("honesty"),
           "blind": res.get("blind") or [], "last_seen": res.get("last_seen"), "refused": res.get("refused"),
           "trace": trace, "model": res.get("model"), "engine": res.get("engine"), "fallback": res.get("fallback"),
           "asked": asked}
    if res.get("sentence_from"):
        out["sentence_from"] = res["sentence_from"]
    out["weave"] = weave_log(cam, q, t, out)
    return out


# ----------------------------------------------------------------------------- optional Weave trace

def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def weave_log(cam: str, question: str, t: float | None, res: dict[str, Any],
              cfg: WandbConfig | None = None) -> str | None:
    """Post the answer and its trace as Weave calls (POST /call/start + /call/end). Only with STM_WEAVE=1."""
    if os.environ.get("STM_WEAVE") != "1":
        return None
    cfg = cfg or load_config()
    if cfg is None:
        return "failed: no W&B config"
    auth = base64.b64encode(f"api:{cfg.key}".encode()).decode()
    hdr = {"Authorization": f"Basic {auth}"}
    end = time.time()
    start = end - float(res["latency_ms"]) / 1000.0
    trace_id, root = str(uuid.uuid4()), str(uuid.uuid4())
    out = {k: v for k, v in res.items() if k not in ("trace", "weave")}
    try:
        post_json(f"{WEAVE_URL}/call/start", {"start": {
            "project_id": cfg.project_id, "id": root, "trace_id": trace_id, "op_name": "stm.agent.answer",
            "started_at": _iso(start), "attributes": {"cam": cam},
            "inputs": {"cam": cam, "question": question, "t": t}}}, hdr, 3.0)
        for step in res["trace"]:
            cid, s0 = str(uuid.uuid4()), start + float(step.get("at_ms") or 0.0) / 1000.0
            post_json(f"{WEAVE_URL}/call/start", {"start": {
                "project_id": cfg.project_id, "id": cid, "trace_id": trace_id, "parent_id": root,
                "op_name": f"stm.agent.{step['tool']}", "started_at": _iso(s0), "attributes": {},
                "inputs": {"args": step.get("args"), "model": step.get("model")}}}, hdr, 3.0)
            post_json(f"{WEAVE_URL}/call/end", {"end": {
                "project_id": cfg.project_id, "id": cid, "ended_at": _iso(s0 + float(step["latency_ms"]) / 1000.0),
                "output": {k: v for k, v in step.items() if k not in ("args",)}, "summary": {}}}, hdr, 3.0)
        post_json(f"{WEAVE_URL}/call/end", {"end": {
            "project_id": cfg.project_id, "id": root, "ended_at": _iso(end), "output": out, "summary": {}}}, hdr, 3.0)
        return f"logged trace {trace_id}"
    except LLMError as e:
        return f"failed: {scrub(e)}"


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stm.agent", description="Ask one camera a question (W&B agent).")
    ap.add_argument("cam")
    ap.add_argument("question")
    ap.add_argument("--t", type=float, default=None, help="seconds into the recording the question is asked at")
    a = ap.parse_args(argv)
    res = answer(a.cam, a.question, a.t)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
