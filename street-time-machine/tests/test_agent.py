"""Tests for stm.agent, fully offline: fake camera files, a scripted fake LLM, a fake stm.vss and a local HTTP
server standing in for W&B Inference / Weave. Nothing here touches the network or the real out/ folders."""
from __future__ import annotations

import base64
import io
import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from stm import agent

CAM = "fakecam"
KEY = "test-key-not-a-real-one-1234567890"
REQUIRED = {"type", "answer", "sentence", "latency_ms", "evidence", "honesty", "blind", "last_seen", "refused", "trace"}


# ----------------------------------------------------------------------------- fake camera

def _write(d: Path, name: str, obj) -> None:
    (d / name).write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture()
def out(tmp_path: Path) -> Path:
    """A 300 s two-day camera: frozen 120-240 s (02:00-04:00), no signal 60-62 s, one parked car #7."""
    d = tmp_path / CAM
    d.mkdir()
    per, blind = [], [{"start": 60.0, "end": 62.0, "reason": "no_signal", "score": 1.0},
                      {"start": 120.0, "end": 240.0, "reason": "frozen", "score": 0.9}]
    for s in range(300):
        r = "no_signal" if 60 <= s < 62 else "frozen" if 120 <= s < 240 else None
        per.append({"t": s, "visible": r is None, "reason": r})
    _write(d, "ledger.json", {"camera": CAM, "seconds": 300, "blind": blind, "spots": [], "per_second": per})
    _write(d, "meta.json", {"camera": CAM, "title": "Test street", "duration": 300.0, "w": 1280, "h": 720,
                            "days": [{"id": "A", "label": "Day A", "start": 0.0, "end": 150.0},
                                     {"id": "B", "label": "Day B", "start": 150.0, "end": 300.0}],
                            "simulated": ["Day A / Day B are two halves of one clip"]})
    _write(d, "near.json", {"camera": CAM, "targets": [{
        "object_id": 7, "name": "car", "box": [100, 200, 300, 320], "start": 10.0, "end": 110.0, "crop": "near/7.jpg",
        "visitors": [{"object_id": 21, "name": "person", "closest_t": 30.5, "gap_px": 3.0, "gap_rel": 0.1,
                      "enter_t": 29.0, "exit_t": 32.0, "crop": "near/21_30.5.jpg"},
                     {"object_id": 22, "name": "bicycle", "closest_t": 80.25, "gap_px": 9.0, "gap_rel": 0.3,
                      "enter_t": 79.0, "exit_t": 81.0, "crop": "near/22_80.2.jpg"}],
        "blind": [{"start": 60.0, "end": 62.0, "reason": "no_signal"}],
        "summary": "1 person and 1 bicycle came within half a car-height of it between 00:10 and 01:50; "
                   "camera was blind for 2.0 s."}]})
    _write(d, "diff.json", {"camera": CAM, "day_a": "A", "day_b": "B", "counts": {"same": 1, "arrived": 1, "left": 0,
                                                                                 "moved": 0, "swapped": 0},
                            "changes": [{"kind": "same", "box": [100, 200, 300, 320],
                                         "a": {"object_id": 7, "crop": "crops/7.jpg", "t": 50.0},
                                         "b": {"object_id": 30, "crop": "crops/30.jpg", "t": 260.0},
                                         "similarity": 0.91, "note": "Same car on both days."},
                                        {"kind": "arrived", "box": [400, 200, 600, 320], "a": None,
                                         "b": {"object_id": 31, "crop": "crops/31.jpg", "t": 270.0},
                                         "similarity": None, "note": "New car on day B."}]})
    _write(d, "qa.json", {"camera": CAM, "items": [{"question": "Is car #7 still there?", "t": None,
                                                    "type": "still_there", "answer": {
        "type": "still_there", "answer": "YES", "sentence": "Yes. Car #7 is still in its spot at 04:59.",
        "latency_ms": 1.0, "evidence": [{"object_id": 7, "t": 299.5, "crop": "ask_crops/7.jpg", "box": None}],
        "honesty": None, "last_seen": 299.5, "blind": [], "refused": None}}]})
    _write(d, "chunk_offsets.json", {"offsets": {"cam_chunk_0000": 0.0, "cam_chunk_0001": 150.0}})
    return tmp_path


def fake_ask(cam: str, q: str, t):
    """Stand-in for stm.ask.answer: knows car #7, raises like stm.ask does for a camera it has no tracks for."""
    if "still there" in q.lower() and "#7" in q:
        return {"type": "still_there", "answer": "YES", "latency_ms": 1.0,
                "sentence": "Yes. Car #7 is still in its spot at 04:59 (detected there at 299.5 s).",
                "evidence": [{"object_id": 7, "t": 299.5, "crop": "ask_crops/7.jpg", "box": [100, 200, 300, 320]}],
                "honesty": None, "last_seen": 299.5, "blind": [], "refused": None}
    raise KeyError(f"unknown camera {cam}")


def no_ask(cam: str, q: str, t):
    raise KeyError(f"unknown camera {cam}")


def _hit(fn: str, a: float, b: float, cap: str, **kw):
    return {"filename": fn, "source": f"s3://segs/{fn}", "segment_start_sec": a, "segment_end_sec": b,
            "reasoning_content": cap, "similarity_score": 0.8, **kw}


FAKE_VSS = types.SimpleNamespace(
    search=lambda query, cam=None, top_k=5: [
        _hit("cam_chunk_0000_segment_009_of_030.mp4", 40.0, 45.0, "A white delivery van stops at the kerb.",
             camera_id=CAM),
        _hit("cam_chunk_0001_segment_002_of_030.mp4", 5.0, 10.0, "A delivery van passes.", camera_id=CAM),
        _hit("other_chunk_0000_segment_001_of_006.mp4", 0.0, 5.0, "A van.", camera_id="other_cam"),
        _hit("nochunk_segment_001_of_006.mp4", 0.0, 5.0, "A van parks.")][:top_k],
    segment=lambda source: {"segment": _hit(source.rsplit("/", 1)[-1], 40.0, 45.0, "A white delivery van.")})


# ----------------------------------------------------------------------------- scripted fake LLM

def call(name: str, args, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class ScriptedChat:
    """Replies by question: a list of {'tool_calls': [...]} / {'content': str}. Records every request."""

    def __init__(self, script: dict[str, list[dict]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.requests: list[dict] = []

    def __call__(self, model, messages, tools, tool_choice="auto", timeout=8.0):
        q = messages[1]["content"].rsplit("Question: ", 1)[-1]
        self.requests.append({"model": model, "q": q, "tool_choice": tool_choice, "n": len(messages),
                              "tool_msgs": [m for m in messages if m["role"] == "tool"]})
        steps = self.script.get(q) or [{"content": ""}]
        r = steps.pop(0) if len(steps) > 1 else steps[0]
        return {"content": r.get("content"), "tool_calls": r.get("tool_calls") or [], "usage": {"total_tokens": 42}}


Q_STILL = "Is car #7 still there?"
Q_BLIND = "Did anyone walk past between 02:00 and 04:00?"
Q_DENT = "Who dented car #7?"
Q_WHO = "Who is this person at 00:30?"
Q_VAN = "Was there a delivery van?"
Q_NEAR = "Who came near car #7?"
Q_DAYS = "What changed between Day A and Day B?"
Q_SEE = "Could the camera see at 01:00?"

SCRIPT = {
    Q_STILL: [{"tool_calls": [call("stm_ask", {"question": Q_STILL})]},
              {"content": "Yes. Car #7 is still in its spot at 04:59."}],
    Q_VAN: [{"tool_calls": [call("vss_search", {"query": "delivery van"}, "c1"),
                            call("stm_visibility", {"start": 0, "end": 300}, "c2")]},
            {"content": "One caption puts a delivery van at 00:40; a second hit at 02:35 fell in frozen time and "
                        "is not evidence."}],
    Q_NEAR: [{"tool_calls": [call("stm_near", '{"object_id": "7"}')]},          # Llama style: number as string
             {"content": "2 came near car #7 between 00:10 and 01:50; near is not doing."}],
    Q_DAYS: [{"tool_calls": [call("stm_diff", {})]},
             {"content": "Day A vs Day B: 1 arrived, 0 left, 1 stayed (the days are simulated)."}],
    Q_SEE: [{"tool_calls": [call("stm_visibility", {"start": "55", "end": "65"})]},
            {"content": "Mostly: it saw 8 of 10 s around 01:00, with no signal 01:00-01:02."}],
}


@pytest.fixture()
def chat() -> ScriptedChat:
    return ScriptedChat(SCRIPT)


def ask(out: Path, chat, q: str, t=None, **kw) -> dict:
    kw.setdefault("ask_fn", fake_ask)
    kw.setdefault("vss", FAKE_VSS)
    return agent.answer(CAM, q, t, chat=chat, out=out, **kw)


# ----------------------------------------------------------------------------- the 8 scripted questions

def test_eight_scripted_questions(out, chat):
    res = {q: ask(out, chat, q) for q in (Q_STILL, Q_BLIND, Q_DENT, Q_WHO, Q_VAN, Q_NEAR, Q_DAYS, Q_SEE)}
    for q, r in res.items():
        assert REQUIRED <= set(r), q
        assert isinstance(r["trace"], list) and r["trace"], q
        for step in r["trace"]:
            assert {"tool", "args", "latency_ms", "model"} <= set(step), (q, step)
        assert r["latency_ms"] < 2000, q
    # the LLM was never asked the guarded questions
    asked = {x["q"] for x in chat.requests}
    assert not asked & {Q_BLIND, Q_DENT, Q_WHO}

    r = res[Q_STILL]
    assert (r["type"], r["answer"], r["engine"], r["sentence_from"]) == ("still_there", "YES", "llm (stm_ask)", "llm")
    assert r["evidence"][0]["object_id"] == 7 and r["last_seen"] == 299.5
    assert any(s["tool"] == "stm_ask" and s["model"] == agent.MODELS[0] for s in r["trace"])


def test_blind_time_is_cant_tell(out, chat):
    r = ask(out, chat, Q_BLIND)
    assert r["answer"] == agent.CANT_TELL and r["engine"] == "guard"
    assert "picture frozen 02:00-04:00" in r["sentence"] and "100%" in r["sentence"]
    assert r["blind"] == [{"start": 120.0, "end": 240.0, "reason": "frozen"}]
    assert r["asked"]["start"] == 120.0 and r["asked"]["end"] == 240.0
    assert chat.requests == []


def test_blame_and_identity_are_refused_before_the_llm(out, chat):
    for q, kind in ((Q_DENT, "blame"), (Q_WHO, "identity"), ("who did it?", "blame"),
                    ("Identify the driver of the van", "identity"), ("Whose car is parked there?", "identity")):
        r = ask(out, chat, q)
        assert (r["sentence"], r["refused"], r["type"]) == (agent.REFUSAL, kind, "refusal"), q
        assert r["trace"][0]["tool"] == "guard:refuse"
    assert chat.requests == []


def test_descriptive_query_cites_vss_hits_and_flags_blind_ones(out, chat):
    r = ask(out, chat, Q_VAN)
    assert (r["type"], r["answer"], r["engine"]) == ("search", "1 caption hit", "llm (vss_search)")
    ev = {e["t"]: e for e in r["evidence"] if e["from"] == "vss_search"}
    assert ev[40.0]["tag"] == "evidence" and "delivery van" in ev[40.0]["caption"]
    assert ev[155.0]["tag"].startswith("not evidence: picture frozen")      # 150 s chunk offset + 5 s
    tags = [e["tag"] for e in r["evidence"] if e["from"] == "vss_search"]
    assert "not evidence: another camera (other_cam)" in tags
    assert "not evidence: can't place this hit on the camera clock" in tags
    assert r["sentence_from"] == "llm" and "02:35" in r["sentence"]
    tools = [s["tool"] for s in r["trace"] if s["tool"] not in ("llm",) and not s["tool"].startswith("guard")]
    assert tools == ["vss_search", "stm_visibility"]                         # both parallel calls ran in round 1
    assert all(s.get("round") == 1 for s in r["trace"] if s["tool"] in tools)


def test_near_with_string_args_from_llama(out, chat):
    r = ask(out, chat, Q_NEAR, models=("meta-llama/Llama-3.3-70B-Instruct",))
    assert (r["type"], r["answer"]) == ("who_near", "2")
    step = next(s for s in r["trace"] if s["tool"] == "stm_near")
    assert step["args"] == {"object_id": 7} and step["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert [e["object_id"] for e in r["evidence"]] == [21, 22]
    assert "near is not doing" in r["honesty"] or "being near is not doing it" in r["honesty"]


def test_days_diff_keeps_simulated_note(out, chat):
    r = ask(out, chat, Q_DAYS)
    assert (r["type"], r["answer"]) == ("what_changed", "1 arrived, 0 left, 1 stayed")
    assert "SIMULATED" in r["honesty"]
    assert r["asked"]["why"] == "day Day A + Day B"
    assert {e["object_id"] for e in r["evidence"]} == {7, 30, 31}


def test_visibility_question(out, chat):
    r = ask(out, chat, Q_SEE)
    assert (r["type"], r["answer"]) == ("visibility", "8 of 10 s seen")
    assert r["asked"] == {"start": 55.0, "end": 65.0, "why": "around 01:00", "blind_pct": 20}
    step = next(s for s in r["trace"] if s["tool"] == "stm_visibility")
    assert step["args"] == {"start": 55.0, "end": 65.0}


# ----------------------------------------------------------------------------- fallbacks and guards

def test_wandb_unreachable_falls_back_to_rules(out):
    def down(**kw):
        raise agent.LLMError("URLError: [Errno 11001] getaddrinfo failed")
    r = ask(out, down, Q_STILL, ask_fn=no_ask)
    assert r["engine"] == "rules (qa.json)" and r["answer"] == "YES"
    assert r["fallback"].startswith("W&B Inference unreachable")
    assert [s["model"] for s in r["trace"] if s["tool"] == "llm"] == list(agent.MODELS)   # every model tried
    assert r["trace"][-1]["tool"] == "rules"


def test_rule_fallback_prefers_stm_ask(out):
    def down(**kw):
        raise agent.LLMError("down")
    r = ask(out, down, Q_STILL)
    assert r["engine"] == "rules (stm.ask)" and "299.5 s" in r["sentence"]


def test_tool_rounds_are_capped_then_rules(out):
    always = ScriptedChat({Q_STILL: [{"tool_calls": [call("stm_visibility", {"start": 0, "end": 10})]}]})
    r = ask(out, always, Q_STILL)
    assert sum(1 for s in r["trace"] if s["tool"] == "stm_visibility") == agent.MAX_ROUNDS
    assert [x["tool_choice"] for x in always.requests] == ["auto"] * agent.MAX_ROUNDS + ["none"]
    assert r["engine"].startswith("rules") and "tool rounds used up" in r["fallback"]


def test_time_budget_then_rules(out):
    def slow(**kw):
        time.sleep(0.25)
        return {"content": None, "tool_calls": [call("stm_visibility", {"start": 0, "end": 10})]}
    r = ask(out, slow, Q_STILL, budget_s=0.4)
    assert r["engine"].startswith("rules") and "time budget" in r["fallback"]
    assert r["latency_ms"] < 2000


def test_slow_vss_cannot_break_the_budget(out, chat):
    def slow_search(q, camera_id=None, top_k=10):
        time.sleep(1.5)
        return []
    t0 = time.perf_counter()
    r = ask(out, chat, Q_VAN, vss=types.SimpleNamespace(search=slow_search), budget_s=0.6)
    assert time.perf_counter() - t0 < 1.2
    step = next(s for s in r["trace"] if s["tool"] == "vss_search")
    assert step["ok"] is False and step["latency_ms"] < 900
    assert r["engine"].startswith("rules") and "time budget" in r["fallback"]


def test_sentence_with_invented_number_is_replaced(out):
    liar = ScriptedChat({Q_NEAR: [{"tool_calls": [call("stm_near", {"object_id": 7})]},
                                  {"content": "5 people came near car #7."}]})
    r = ask(out, liar, Q_NEAR)
    assert r["answer"] == "2" and r["sentence"].startswith("1 person and 1 bicycle")
    assert r["sentence_from"] == "stm_near"
    assert any(s["tool"] == "guard:sentence" and "5" in s["note"] for s in r["trace"])


def test_sentence_contradicting_the_tool_is_replaced(out):
    bad = ScriptedChat({Q_STILL: [{"tool_calls": [call("stm_ask", {"question": Q_STILL})]},
                                  {"content": "No, car #7 left."}]})
    r = ask(out, bad, Q_STILL)
    assert r["answer"] == "YES" and r["sentence"].startswith("Yes. Car #7")


def test_inline_json_tool_call_and_empty_content(out):
    inline = ScriptedChat({Q_NEAR: [{"content": '<|python_tag|>{"name": "stm_near", "parameters": {"object_id": "7"}}'},
                                    {"content": "   "}]})                   # thinking ate the tokens: no text
    r = ask(out, inline, Q_NEAR)
    assert r["answer"] == "2" and r["sentence_from"] == "stm_near" and r["engine"] == "llm (stm_near)"


def test_answer_without_tools_is_not_trusted(out):
    chatty = ScriptedChat({Q_STILL: [{"content": "Yes, it is still there."}]})
    r = ask(out, chatty, Q_STILL)
    assert r["engine"] == "rules (stm.ask)" and "without calling a tool" in r["fallback"]


# ----------------------------------------------------------------------------- pieces

def test_refusal_rules():
    assert agent.refusal("who dented it") == "blame"
    assert agent.refusal("Which car hit the red car?") == "blame"
    assert agent.refusal("Who is responsible for the scratch?") == "blame"
    assert agent.refusal("Can you recognise the man?") == "identity"
    assert agent.refusal("what's the plate number of the van") == "identity"
    for ok in (Q_NEAR, Q_STILL, "How many people walked by?", "Who came close to the van at 05:00?",
               "Which way did the person go?"):
        assert agent.refusal(ok) is None, ok


def test_asked_range_rules():
    days = [{"id": "A", "label": "Day A", "start": 0.0, "end": 150.0},
            {"id": "B", "label": "Day B", "start": 150.0, "end": 300.0}]
    rng = lambda q, t=None: agent.asked_range(q, t, 300.0, days)[:2]  # noqa: E731
    assert rng("between 01:00 and 02:30") == (60.0, 150.0)
    assert rng("how many cars since 03:00?") == (180.0, 300.0)
    assert rng("how many cars since 03:00?", 250.0) == (180.0, 250.0)
    assert rng("When did car #7 leave, by 02:00?") == (0.0, 120.0)
    assert rng("Is car #7 there at 01:00?") == (55.0, 65.0)
    assert rng("what happened on day B") == (150.0, 300.0)
    assert rng("anyone in the last 2 minutes?", 200.0) == (80.0, 200.0)
    assert rng("is it still there?", 100.0) == (90.0, 100.0)
    assert rng("how many people walked by?") == (0.0, 300.0)
    assert rng("at 1:00:00") == (300.0, 300.0)                              # clipped to the recording


def test_visibility_counts_and_not_recorded(out):
    box = agent.Toolbox(CAM, out, None)
    v = box.stm_visibility(100, 130)
    assert (v["seconds"], v["seen_s"], v["blind_s"], v["blind_pct"]) == (30, 20, 10, 33)
    assert v["spans"][0]["span"] == "02:00-02:10" and v["reasons"] == {"picture frozen": 10}
    v = box.stm_visibility(290, 310)
    assert v["blind_s"] == 10 and v["reasons"] == {"not recorded": 10}
    whole = box.stm_visibility(0, 300)
    assert whole["blind_s"] == 122 and whole["seen_s"] == 178


def test_no_ledger_means_unknown_not_guessed(out):
    (out / CAM / "ledger.json").unlink()
    v = agent.Toolbox(CAM, out, None).stm_visibility(120, 240)
    assert v["known"] is False and "unknown" in v["note"]


def test_lazy_stm_vss_import_and_local_caption_fallback(out, monkeypatch):
    monkeypatch.setitem(sys.modules, "stm.vss", FAKE_VSS)
    monkeypatch.setenv("STM_VSS_LIVE", "1")                                  # the live /search POST is opt-in
    r = agent.Toolbox(CAM, out, None).vss_search("delivery van", top_k=2)
    assert r["backend"] == "vss" and r["n_hits"] == 2 and r["n_evidence"] == 1
    seg = agent.Toolbox(CAM, out, None).vss_segment("s3://segs/cam_chunk_0000_segment_009_of_030.mp4")
    assert seg["segment"]["t"] == 40.0 and seg["segment"]["tag"] == "evidence"

    monkeypatch.setitem(sys.modules, "stm.vss", None)                       # W3's module missing: import fails
    _write(out / CAM, "segments.json", {"segments": [
        _hit("cam_chunk_0001_segment_020_of_030.mp4", 100.0, 105.0, "A grey delivery van unloads boxes.",
             camera_id=CAM),
        _hit("cam_chunk_0000_segment_001_of_030.mp4", 0.0, 5.0, "An empty street.", camera_id=CAM)]})
    r = agent.Toolbox(CAM, out, None).vss_search("delivery vans")
    assert r["backend"].startswith("local captions") and r["n_hits"] == 1
    assert r["hits"][0]["t"] == 250.0 and r["hits"][0]["tag"] == "evidence"


def test_with_real_stm_vss_module_and_fake_transport(out, chat, monkeypatch):
    """W3's stm.vss as it is, with a fake transport: the search POST it builds and the hits it returns."""
    vss = pytest.importorskip("stm.vss")
    monkeypatch.setenv("STM_VSS_LIVE", "1")
    seen = []

    def transport(method, url, headers, body, timeout):
        route = url.split("/api/v1/", 1)[1].split("?", 1)[0]
        seen.append({"method": method, "route": route, "body": json.loads(body) if body else None})
        if route == "auth/login":
            obj = {"access_token": "jwt.fake-token", "token_type": "bearer"}
        elif route == "search":
            obj = {"results": [
                {"source": "s3://segs/cam_chunk_0001_segment_002_of_030.mp4", "similarity_score": 0.7,
                 "reasoning_content": "A delivery van passes.", "segment_start_sec": "5.0", "segment_end_sec": "10.0",
                 "original_video": "s3://chunks/cam_chunk_0001.mp4", "camera_id": CAM,
                 "filename": "cam_chunk_0001_segment_002_of_030.mp4"},
                {"source": "s3://segs/cam_chunk_0000_segment_009_of_030.mp4", "similarity_score": 0.9,
                 "reasoning_content": "A white delivery van stops.", "segment_start_sec": "40.0",
                 "segment_end_sec": "45.0", "original_video": "s3://chunks/cam_chunk_0000.mp4", "camera_id": CAM,
                 "filename": "cam_chunk_0000_segment_009_of_030.mp4"}], "chunk_results": [], "total": 2}
        else:
            obj = {"detail": "not found"}
        return vss.Response(200 if route in ("auth/login", "search") else 404, {}, io.BytesIO(json.dumps(obj).encode()))

    old = vss.set_client(vss.Client(vss.Config("https://vss.example.test", "u", "p", "test"), transport=transport,
                                    sleep=lambda s: None))
    try:
        r = agent.answer(CAM, Q_VAN, out=out, chat=chat, ask_fn=fake_ask)
    finally:
        vss.set_client(old)
    body = next(x["body"] for x in seen if x["route"] == "search")
    assert body["query"] == "delivery van" and body["metadata_filters"] == {"camera_id": CAM}
    assert body["top_k"] == 5
    ev = {e["t"]: e["tag"] for e in r["evidence"] if e["from"] == "vss_search"}
    assert ev == {40.0: "evidence", 155.0: ev[155.0]} and ev[155.0].startswith("not evidence: picture frozen")
    assert r["answer"] == "1 caption hit" and "jwt.fake-token" not in json.dumps(r)


def test_camera_time_fields_from_vss_win_over_offsets(out):
    box = agent.Toolbox(CAM, out, None)
    assert box.cam_time({"cam_start": 12.5, "cam_end": 17.5, "filename": "cam_chunk_0001_segment_001.mp4",
                         "segment_start_sec": 0}) == (12.5, 17.5)
    assert box.cam_time({"filename": "cam_chunk_0001_segment_001_of_030.mp4", "segment_start_sec": "5.0",
                         "segment_end_sec": "10.0"}) == (155.0, 160.0)


def test_qa_lookup_needs_same_ids(out):
    d = out / CAM
    assert agent.qa_lookup(d, "is car #7 still there", None)["answer"] == "YES"
    assert agent.qa_lookup(d, "Is car #7 still there now?", None)["match_score"] >= 0.6
    assert agent.qa_lookup(d, "Is car #8 still there?", None) is None


def test_allowed_numbers_and_clean_sentence():
    allowed = agent.allowed_numbers(['{"t": 155.0, "n": 2, "blind_pct": 46, "t_mmss": "02:35"}'])
    assert agent.unsupported_numbers("2 cars at 02:35 and 2:35, 46% blind, 155 s", allowed) == []
    assert agent.unsupported_numbers("3 cars at 02:36", allowed) == ["3", "02:36"]
    assert agent.clean_sentence("<think>hm</think>**2 cars** came near car #7.") == "2 cars came near car #7."
    assert agent.clean_sentence("- 2 cars came.") == "2 cars came."


def test_config_from_env_file_and_no_key_leak(tmp_path, monkeypatch):
    for k in agent.WANDB_KEYS:
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / "environment"
    env.write_text(f'PATH="/usr/bin"\nWANDB_API_KEY="{KEY}"\nWANDB_TEAM=team\nexport WANDB_PROJECT=proj\n')
    cfg = agent.load_config(str(env))
    assert (cfg.team, cfg.project, cfg.project_id) == ("team", "proj", "team/proj")
    assert KEY not in repr(cfg) and KEY not in str(cfg)
    assert KEY not in agent.scrub(f"HTTP 401: bad key {KEY}; Bearer abc.def; token=xyz")
    assert "xyz" not in agent.scrub("GET /stream?source=a&token=xyz")
    assert agent.load_config(str(tmp_path / "missing")) is None


# ----------------------------------------------------------------------------- real urllib path, local server

@pytest.fixture()
def server(monkeypatch):
    """Local stand-in for W&B Inference and Weave: scripted JSON replies, records every request."""
    state = {"replies": [], "seen": []}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            state["seen"].append({"path": self.path, "headers": dict(self.headers),
                                  "body": json.loads(self.rfile.read(n) or b"{}")})
            status, payload = state["replies"].pop(0) if state["replies"] else (503, {"error": "busy"})
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("WANDB_API_KEY", KEY)
    monkeypatch.setenv("WANDB_TEAM", "team")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    monkeypatch.setattr(agent, "API_URL", base + "/v1/chat/completions")
    monkeypatch.setattr(agent, "WEAVE_URL", base + "/weave")
    state["base"] = base
    yield state
    srv.shutdown()


def _choice(msg: dict, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"role": "assistant", **msg}, "finish_reason": finish}],
            "usage": {"total_tokens": 99}}


def test_wandb_chat_over_http(out, server):
    server["replies"] += [(200, _choice({"content": None, "tool_calls": [call("stm_ask", {"question": Q_STILL})]},
                                        "tool_calls")),
                          (200, _choice({"content": "Yes. Car #7 is still in its spot at 04:59."}))]
    r = agent.answer(CAM, Q_STILL, out=out, ask_fn=fake_ask, vss=FAKE_VSS)
    assert (r["engine"], r["model"], r["answer"]) == ("llm (stm_ask)", agent.MODELS[0], "YES")
    first, second = server["seen"]
    h = {k.lower(): v for k, v in first["headers"].items()}
    assert h["authorization"] == f"Bearer {KEY}" and h["openai-project"] == "team/proj"
    b = first["body"]
    assert (b["model"], b["temperature"], b["max_tokens"], b["tool_choice"]) == (agent.MODELS[0], 0, 1500, "auto")
    assert len(b["tools"]) == 6 and "chat_template_kwargs" not in b
    assert second["body"]["messages"][-1]["role"] == "tool"
    assert KEY not in json.dumps(r)


def test_glm_gets_thinking_off_and_retries_without_it(out, server):
    server["replies"] += [(400, {"detail": "unknown chat_template_kwargs"}),
                          (200, _choice({"content": None, "tool_calls": [call("stm_near", {"object_id": 7})]})),
                          (200, _choice({"content": "2 came near car #7 between 00:10 and 01:50."}))]
    r = agent.answer(CAM, Q_NEAR, out=out, ask_fn=fake_ask, models=("zai-org/GLM-5.3",))
    assert r["answer"] == "2" and r["model"] == "zai-org/GLM-5.3"
    assert server["seen"][0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "chat_template_kwargs" not in server["seen"][1]["body"]


def test_http_errors_fall_back_to_rules_without_leaking(out, server):
    r = agent.answer(CAM, Q_STILL, out=out, ask_fn=no_ask)                  # server answers 503 to everything
    assert r["engine"] == "rules (qa.json)" and r["answer"] == "YES"
    assert len(server["seen"]) == len(agent.MODELS)
    assert all("HTTP 503" in s["error"] for s in r["trace"] if s["tool"] == "llm")
    assert KEY not in json.dumps(r)


def test_no_config_falls_back(out, monkeypatch):
    for k in agent.WANDB_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(agent, "ENV_FILE", str(out / "no-such-env"))
    monkeypatch.setattr(agent.load_config, "__defaults__", (str(out / "no-such-env"),))
    r = agent.answer(CAM, Q_STILL, out=out, ask_fn=no_ask)
    assert r["engine"] == "rules (qa.json)" and "no W&B config" in r["fallback"]


def test_weave_off_by_default_and_posts_calls_when_on(out, server, monkeypatch):
    chat = ScriptedChat(SCRIPT)
    monkeypatch.delenv("STM_WEAVE", raising=False)
    assert ask(out, chat, Q_STILL)["weave"] is None and server["seen"] == []
    monkeypatch.setenv("STM_WEAVE", "1")
    server["replies"] += [(200, {"id": "x"})] * 20
    r = ask(out, chat, Q_DENT)                                              # refusal: one guard step
    assert r["weave"].startswith("logged trace")
    paths = [s["path"] for s in server["seen"]]
    assert paths == ["/weave/call/start", "/weave/call/start", "/weave/call/end", "/weave/call/end"]
    h = {k.lower(): v for k, v in server["seen"][0]["headers"].items()}
    assert base64.b64decode(h["authorization"].split()[1]).decode() == f"api:{KEY}"
    start = server["seen"][0]["body"]["start"]
    assert start["project_id"] == "team/proj" and start["op_name"] == "stm.agent.answer"
    assert server["seen"][1]["body"]["start"]["parent_id"] == start["id"]


def test_unknown_camera_raises(out, chat):
    with pytest.raises(KeyError):
        agent.answer("nope", Q_STILL, chat=chat, out=out, ask_fn=fake_ask)


def test_real_ledger_if_present():
    """On the real Pack D ledger (read only): visibility of the whole recording matches per_second."""
    p = agent.OUT / "neighborhood_cam-1" / "ledger.json"
    led = agent.read_json(p)
    if not isinstance(led, dict) or not led.get("per_second"):
        pytest.skip("no Pack D ledger on this machine")
    box = agent.Toolbox("neighborhood_cam-1", agent.OUT, None)
    v = box.stm_visibility(0, len(led["per_second"]))
    assert v["blind_s"] == sum(1 for x in led["per_second"] if not x["visible"])
    assert v["seen_s"] + v["blind_s"] == v["seconds"]


# ----------------------------------------------------------------------------- after the first live smoke

def _ds(name: str, args: dict) -> str:
    """A tool call the way DeepSeek-V3.1 sometimes writes it into the content (fullwidth bars, U+2581)."""
    b, u = "\uff5c", "\u2581"
    return (f"<{b}tool{u}calls{u}begin{b}><{b}tool{u}call{u}begin{b}>{name}<{b}tool{u}sep{b}>{json.dumps(args)}"
            f"<{b}tool{u}call{u}end{b}><{b}tool{u}calls{u}end{b}>")


def test_deepseek_text_tool_calls_are_run_and_never_shown(out):
    ds = ScriptedChat({Q_NEAR: [{"content": _ds("stm_near", {"object_id": 7})},
                                {"content": _ds("stm_nope", {"x": 1})}]})          # markup again, unknown tool
    r = ask(out, ds, Q_NEAR)
    assert [s["tool"] for s in r["trace"] if s["tool"] == "stm_near"] == ["stm_near"]
    assert r["answer"] == "2" and "tool" not in r["sentence"] and "\uff5c" not in r["sentence"]
    assert r["sentence_from"] == "stm_near"
    v3 = "<|tool_call_begin|>function<|tool_sep|>stm_diff\n```json\n{}\n```<|tool_call_end|>"
    assert [c["name"] for c in agent._tool_calls({"content": v3})] == ["stm_diff"]
    assert agent.clean_sentence(_ds("stm_diff", {})) == ""
    assert agent.clean_sentence("Yes, car #7 is there.<|eot_id|>") == "Yes, car #7 is there."


def test_final_call_after_the_rounds_cap_is_nudged(out):
    seen = []

    def always(model, messages, tools, tool_choice="auto", timeout=8.0):
        seen.append((tool_choice, messages[-1]))
        if tool_choice == "none":
            return {"content": "Yes. Car #7 is still in its spot at 04:59.", "tool_calls": []}
        return {"content": None, "tool_calls": [call("stm_ask", {"question": Q_STILL})]}
    r = ask(out, always, Q_STILL)
    assert seen[-1][0] == "none" and seen[-1][1] == {"role": "user", "content": agent.FINAL_NUDGE}
    assert r["answer"] == "YES" and r["engine"] == "llm (stm_ask)"


def test_caption_hits_outside_the_asked_time_are_not_evidence(out):
    q = "Was there a delivery van between 00:30 and 01:00?"
    chat = ScriptedChat({q: [{"tool_calls": [call("vss_search", {"query": "delivery van"})]},
                             {"content": "One caption puts a delivery van at 00:40."}]})
    r = ask(out, chat, q)
    tags = {e["t"]: e["tag"] for e in r["evidence"] if e["from"] == "vss_search"}
    assert tags[40.0] == "evidence" and tags[155.0] == "not evidence: outside the asked time 00:30-01:00"
    assert r["answer"] == "1 caption hit"
    box = agent.Toolbox(CAM, out, None, vss=FAKE_VSS)
    assert box.vss_search("delivery van", start=150, end=200)["n_evidence"] == 0      # 02:35 hit is frozen time
    assert box.vss_search("delivery van")["window"] == "all"


def test_local_caption_match_weights_rare_words():
    rows = [{"reasoning_content": f"Cars parked along the street, scene {i}."} for i in range(8)]
    rows.append({"reasoning_content": "A white van parked on the street by the kerb."})
    rows.append({"reasoning_content": "A white car parked on the street."})
    hits = agent.idf_rank(rows, "Was there a white van parked on the street?", 5)
    assert [h["reasoning_content"] for h in hits] == [rows[8]["reasoning_content"]]
    assert agent.idf_rank(rows, "a bus", 5) == []


def test_range_note_summarises_many_spans():
    spans = [{"why": "picture frozen", "span": f"0{i}:00-0{i}:05"} for i in range(5)]
    vis = {"known": True, "blind_s": 25, "seconds": 600, "range": "00:00-10:00", "spans": spans, "more_spans": 2,
           "reasons": {"picture frozen": 20, "no signal": 5}}
    assert agent._range_note(vis) == ("25 of 600 s of the asked range 00:00-10:00 were blind "
                                      "(picture frozen 20 s, no signal 5 s, in 7 spans)")


# ----------------------------------------------------------------------------- review fixes

def _local_segments(out: Path) -> None:
    _write(out / CAM, "segments.json", {"segments": [
        _hit("cam_chunk_0001_segment_020_of_030.mp4", 100.0, 105.0, "A grey delivery van unloads boxes.",
             camera_id=CAM)]})


def test_live_vss_search_is_opt_in_and_falls_back_to_stored_captions(out, monkeypatch):
    """The team VSS /search is a POST: without STM_VSS_LIVE=1 the stored rows answer even when stm.vss imports."""
    _local_segments(out)
    monkeypatch.setitem(sys.modules, "stm.vss", FAKE_VSS)
    monkeypatch.delenv("STM_VSS_LIVE", raising=False)
    r = agent.Toolbox(CAM, out, None).vss_search("delivery van")
    assert r["backend"].startswith("local captions") and r["hits"][0]["t"] == 250.0

    def broken(query, camera_id=None, top_k=10):
        raise RuntimeError("no VSS config: set VSS_URL/VSS_USERNAME/VSS_PASSWORD")
    monkeypatch.setitem(sys.modules, "stm.vss", types.SimpleNamespace(search=broken))
    monkeypatch.setenv("STM_VSS_LIVE", "1")
    r = agent.Toolbox(CAM, out, None).vss_search("delivery van")
    assert r["backend"].startswith("local captions") and "live VSS failed: RuntimeError" in r["backend"]
    assert r["n_evidence"] == 1


def test_search_answer_names_caption_hits_and_zero_hits_cannot_say_yes():
    assert agent.contradicts("Yes, a van was there.", "0 caption hits")
    assert not agent.contradicts("Yes, one caption shows a van.", "1 caption hit")
    assert agent.contradicts("Yes, it is.", "NO") and not agent.contradicts("No.", "NO")


def test_chunk_offsets_follow_the_layout_of_this_video(out, tmp_path, monkeypatch):
    """A whole-file chunk_offsets.json (total off from the video length) is never used on a seam-safe video."""
    monkeypatch.setattr(agent, "PACKD_DIRS", ())
    hit = {"filename": "cam_chunk_0001_segment_002_of_030.mp4", "segment_start_sec": 5.0, "segment_end_sec": 10.0}
    _write(out / CAM, "chunk_offsets.json", {"offsets": {"cam_chunk_0000": 0.0, "cam_chunk_0001": 150.0},
                                             "total": 420.0})            # old layout: 420 s, the video is 300 s
    assert agent.Toolbox(CAM, out, None).cam_time(hit) == (None, None)   # better unplaced than 2 min off
    v2 = tmp_path / "v2_offsets.json"
    v2.write_text(json.dumps({"offsets": {"cam_chunk_0000": 0.0, "cam_chunk_0001": 120.0}, "total": 300.0}))
    monkeypatch.setenv("STM_CHUNK_OFFSETS", str(v2))
    assert agent.Toolbox(CAM, out, None).cam_time(hit) == (125.0, 130.0)
    _write(out / CAM, "chunk_offsets.json", {"offsets": {"cam_chunk_0000": 0.0, "cam_chunk_0001": 110.0}})
    assert agent.Toolbox(CAM, out, None).cam_time(hit) == (125.0, 130.0)   # a matching total beats no total
    monkeypatch.delenv("STM_CHUNK_OFFSETS")
    assert agent.Toolbox(CAM, out, None).cam_time(hit) == (115.0, 120.0)   # no total anywhere: the first file
