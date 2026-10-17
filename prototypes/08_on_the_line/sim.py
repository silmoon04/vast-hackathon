"""Live simulation: advance a clock through the video, answer scripted police questions only from data before it."""
from __future__ import annotations

# make the module importable on its own (protokit lives in prototypes/)
import sys as _sys  # noqa: E402
from pathlib import Path as _P  # noqa: E402
_HERE_PATHS = [str(_P(__file__).resolve().parents[1]), str(_P(__file__).resolve().parent)]
_sys.path[:0] = [x for x in _HERE_PATHS if x not in _sys.path]

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

import protokit as pk
from engine import GROUP_WORD, Entity, LiveFeed, SceneState
from qa import Card, answer

HERE = Path(__file__).resolve().parent
# SIMULATED availability delay of a description (hosted Cosmos3-Reason speed). None = use the measured local latency.
VLM_LATENCY: float | None = 3.0

PROMPTS = {
    "person": "This crop comes from a security camera. Describe this person for a police report in one short "
              "sentence: colours of top and trousers, hat, bag, anything carried. Do not guess age or identity.",
    "vehicle": "This crop comes from a security camera. Describe this vehicle for a police report in one short "
               "sentence: colour, body type (hatchback, saloon, van, truck, SUV) and any visible markings.",
}


def _prep(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(1.0, 224.0 / min(h, w))
    s = min(s, 512.0 / max(h, w)) if max(h, w) * s > 512 else s
    if s > 1.01:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return img


def vlm_describe(e: Entity) -> dict[str, Any] | None:
    """One Cosmos call for the entity's best crop. Returns text + model latency (seconds, read from the cache file so
    a cache hit still carries the original latency)."""
    img = _prep(e.best["img"])
    prompt = PROMPTS[e.group]
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + pk._jpg_b64(img)}},
                                         {"type": "text", "text": prompt}]}]
    f = pk.CACHE / "vlm" / f"{pk._key('vlm', msgs, 70)}.json"
    text = pk.vlm(prompt, [img], max_tokens=70)
    if not text or not f.exists():
        return None
    measured = float(json.loads(f.read_text(encoding="utf-8")).get("seconds", 1.0))
    return {"text": " ".join(text.split()), "seconds": VLM_LATENCY if VLM_LATENCY is not None else measured,
            "measured_seconds": measured}


# -------------------------------------------------------------------------------------------------- scoring


def score_check(chk: dict[str, Any], facts: dict[str, Any]) -> tuple[bool, Any]:
    ty = chk["type"]
    if ty == "count_now":
        got = facts.get("count_now", {}).get(chk["group"])
        return (got is not None and chk["range"][0] <= got <= chk["range"][1]), got
    if ty == "total":
        got = facts.get("total", {}).get(chk["group"])
        return (got is not None and chk["range"][0] <= got <= chk["range"][1]), got
    if ty == "exit_side":
        got = (facts.get("direction") or facts.get("still") or {}).get("exit_side")
        return got in chk["in"], got
    if ty == "heading":
        got = (facts.get("direction") or {}).get("heading")
        return got in chk["in"], got
    if ty == "colour":
        got = ((facts.get("appearance") or {}).get("colours") or {}).get(chk["part"])
        return got in chk["in"], got
    if ty == "present":
        got = (facts.get("still") or {}).get("present")
        return got == chk["value"], got
    if ty == "arrival":
        got = (facts.get("arrival") or {}).get("first_t")
        return (got is not None and abs(got - chk["t"]) <= chk["tol"]), None if got is None else round(got, 2)
    raise ValueError(ty)


@dataclass
class SimResult:
    name: str
    script: dict[str, Any]
    state: SceneState
    cards: list[Card]
    scored: list[dict[str, Any]] = field(default_factory=list)
    wall_s: float = 0.0
    frame_t: list[float] = field(default_factory=list)
    feed_ms: list[float] = field(default_factory=list)


def simulate(name: str, script: dict[str, Any], *, use_vlm: bool = True, only: int | None = None,
             on_frame: Callable[[SceneState, float, np.ndarray, list[Card], bool], None] | None = None,
             end: float | None = None) -> SimResult:
    """Run one clip live. `only`: answer questions 0..only (dialogue context) and stop at question `only`'s time (causality check)."""
    tr = pk.tracks(name)
    alarm = script["alarm"]
    qs = [(i, q) for i, q in enumerate(script["questions"]) if only is None or i <= only]   # earlier questions give the dialogue focus
    stop = end if end is not None else (qs[-1][1]["t"] + 1e-3 if only is not None else script["hangup"])
    st = SceneState(tr.w, tr.h, tr.fps, alarm, use_vlm=use_vlm, vlm_budget=script.get("vlm_budget", 28))
    st.describe_cb = vlm_describe
    feed = LiveFeed(name, tr, alarm, stop)
    cards: list[Card] = []
    scored: list[dict[str, Any]] = []
    pending = list(qs)
    focus: Entity | None = None
    t0 = time.time()

    def ask_due(upto: float) -> bool:
        nonlocal focus
        fired = False
        while pending and pending[0][1]["t"] < upto - 1e-9:
            i, q = pending.pop(0)
            card, focus = answer(st, q["text"], focus)
            if card is None:
                continue
            card.t = q["t"]
            cards.append(card)
            fired = True
            checks = []
            for c in q.get("checks", []):
                ok, got = score_check(c, card.facts)
                checks.append({**c, "ok": bool(ok), "got": got})
            scored.append({"i": i, "id": q["id"], "t": q["t"], "text": q["text"], "headline": card.headline,
                           "latency_ms": round(card.latency_ms, 3), "intents": card.intents, "checks": checks,
                           "correct": all(c["ok"] for c in checks) if checks else None, "facts": card.facts,
                           "caveats": card.caveats})
        return fired

    frame_t, feed_ms = [], []
    for t, i, frame, rows in feed:
        ask_due(t)               # questions up to the previous frame answered BEFORE this frame is ingested
        v0, tf = st.vlm_calls, time.perf_counter()
        st.feed(t, frame, rows)
        if st.vlm_calls == v0:
            feed_ms.append((time.perf_counter() - tf) * 1000)
        frame_t.append(t)
        if on_frame:
            on_frame(st, t, frame, cards, bool(cards and cards[-1].t > t - 1.5 / max(tr.fps, 1)))
    # flush questions that fall after the last frame fed (state as of the last frame)
    while pending:
        i, q = pending[0]
        if q["t"] < stop + 1.0:
            ask_due(max(q["t"] + 1.0, stop + 1.0))
        else:
            pending.pop(0)
    return SimResult(name, script, st, cards, scored, time.time() - t0, frame_t, feed_ms)
