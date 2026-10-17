"""stm.ask: parsing, contract shape and honesty rules on real cached tracks.

Uses out/<cam>/ask_index.json when the CLI has run (fast); otherwise builds it (small cams take seconds)."""
from __future__ import annotations

import json
import re
import statistics

import pytest

from stm import ask, data

KEYS = {"type", "answer", "sentence", "latency_ms", "evidence", "honesty", "last_seen", "blind", "refused"}
EV_KEYS = {"object_id", "t", "crop", "box"}


BASE_KEYS = {"type", "answer", "sentence", "latency_ms", "evidence", "honesty"}


def check_contract(cam: str, a: dict, server: bool = False) -> None:
    if server:      # stm/server.py passes on a subset; the extra keys appear once it forwards them
        assert BASE_KEYS <= set(a) <= KEYS
        a = {"last_seen": None, "blind": [], "refused": None, **a}
    assert set(a) == KEYS
    assert a["type"] in ask.TYPES or a["type"] == "unknown"
    assert isinstance(a["answer"], str) and a["answer"]
    assert isinstance(a["sentence"], str) and a["sentence"]
    assert isinstance(a["latency_ms"], float) and a["latency_ms"] >= 0
    assert a["honesty"] is None or isinstance(a["honesty"], str)
    assert a["last_seen"] is None or isinstance(a["last_seen"], float)
    assert a["refused"] in (None, "appearance", "blame")
    for b in a["blind"]:
        assert set(b) == {"start", "end", "reason"} and b["end"] >= b["start"] >= 0
    # times are mm:ss, never raw seconds or '-0.00'
    assert not re.search(r"\d+\.\d\d s\b", a["sentence"]), a["sentence"]
    assert "-0.00" not in a["sentence"] + (a["honesty"] or "")
    # people are never described by how they look
    assert not re.search(r"person #\d+ \(in ", a["sentence"]), a["sentence"]
    for e in a["evidence"]:
        assert set(e) == EV_KEYS
        assert isinstance(e["object_id"], int)
        assert 0 <= e["t"] <= ask.duration(cam) + 0.01
        if e["box"] is not None:
            x1, y1, x2, y2 = e["box"]
            assert x2 > x1 and y2 > y1
        if e["crop"] is not None:
            assert (data.out_dir(cam) / e["crop"]).exists(), e["crop"]
    json.dumps(a)   # serialisable


@pytest.mark.parametrize("q,typ", [
    ("Is the red car still there?", "still_there"),
    ("When did the white car leave?", "when_left"),
    ("Which way did the person go?", "which_way"),
    ("What direction did the bike take?", "which_way"),
    ("Who came near the silver car?", "who_near"),
    ("Did anyone touch the black car?", "who_near"),
    ("Who dented the red car?", "who_near"),
    ("How many people walked by since 1:00?", "count_since"),
    ("What changed between Day A and Day B?", "what_changed"),
    ("Anything different since 0:20?", "what_changed"),
])
def test_parse_types(q, typ):
    assert ask.parse(q)["type"] == typ


def test_parse_details():
    p = ask.parse("Who came near the silver car?")
    assert p["colour"] == "grey" and p["names"] == ("car", "truck")
    p = ask.parse("How many people walked by since 1:30?")
    assert p["since"] == 90.0 and p["names"] == ("person",)
    p = ask.parse("How many cars in the last 2 minutes?")
    assert p["last"] == 120.0
    assert ask.parse("Is car #12 still there at 2:00?")["object_id"] == 12
    assert ask.parse("Is car #12 still there at 2:00?")["at"] == 120.0
    assert ask.parse("what is the weather")["type"] is None
    p = ask.parse("Who dented the red car?")
    assert p["blame"] and p["colour"] == "red" and p["names"] == ("car", "truck")
    assert not ask.parse("Who came near the red car?")["blame"]


def test_mmss():
    assert ask.mmss(-0.001) == "00:00" and ask.mmss(155.0) == "02:35" and ask.mmss(61.9) == "01:01"


def test_colour_classifier():
    import numpy as np
    red = np.zeros((40, 60, 3), np.uint8); red[:] = (30, 30, 200)
    white = np.full((40, 60, 3), 235, np.uint8)
    black = np.full((40, 60, 3), 20, np.uint8)
    assert ask.dominant(ask.colour_of(red, "car")) == "red"
    assert ask.dominant(ask.colour_of(white, "car")) == "white"
    assert ask.dominant(ask.colour_of(black, "car")) == "black"


@pytest.mark.parametrize("cam", ["car_detection", "person_bicycle_car_detection"])
def test_small_cams_all_types(cam):
    qs = ["Is the car still there?", "When did the car leave?", "Which way did the car go?",
          "Who came near the car?", "How many cars came by?", "What changed?", "blah"]
    for q in qs:
        check_contract(cam, ask.answer(cam, q))


def test_person_exit_edge_is_real():
    cam = "person_bicycle_car_detection"
    a = ask.answer(cam, "Which way did the person go?")
    check_contract(cam, a)
    if a["answer"] in ("left", "right", "top", "bottom") and "edge" in a["sentence"]:
        oid = a["evidence"][-1]["object_id"]
        tr = data.tracks(cam)
        last = tr[tr.tid == oid].iloc[-1]
        W, H = data.info(cam)["w"], data.info(cam)["h"]
        near = {"left": last.x1 <= 0.03 * W, "right": last.x2 >= 0.97 * W,
                "top": last.y1 <= 0.03 * H, "bottom": last.y2 >= 0.97 * H}
        assert near[a["answer"]]


def test_counts_match_objects():
    cam = "person_bicycle_car_detection"
    a = ask.answer(cam, "How many people walked by?")
    n = int(a["answer"])
    o = ask.objs(cam)
    assert 0 < n <= int(((o.name == "person") & o.moved).sum())
    assert len(a["evidence"]) == min(n, 8)


def test_ptz_honesty_and_truth():
    """ptz_parking: the camera turns away after Day A, so a Day A car asked about at the end is UNKNOWN, never NO;
    car #7 is seen in its spot at 60 s and the spot is empty at 100 s (checked by eye), across a video cut at 68 s."""
    cam = "ptz_parking"
    a = ask.answer(cam, "Is car #1 still there?")
    check_contract(cam, a)
    assert a["answer"] == "UNKNOWN" and a["honesty"]
    a = ask.answer(cam, "Is car #1 still there?", 150.0)
    assert a["answer"] == "YES"
    a = ask.answer(cam, "Is car #7 still there?", 60.0)
    assert a["answer"] == "YES"
    a = ask.answer(cam, "Is car #7 still there?", 120.0)
    assert a["answer"] == "NO"
    a = ask.answer(cam, "When did car #7 leave?", 120.0)
    check_contract(cam, a)
    assert a["answer"] == "UNKNOWN" and "01:08" in a["sentence"]   # left across the cut: bounded, not invented
    blind = ask.index(cam)["blind"]
    assert any(b["start"] <= 68.1 <= b["end"] for b in blind)


def test_ptz_red_car_and_blame():
    cam = "ptz_parking"
    a = ask.answer(cam, "Is the red car still there?")
    check_contract(cam, a)
    assert a["answer"] == "UNKNOWN" and a["last_seen"] is not None and a["blind"]
    assert "camera turned away" in a["sentence"]
    near = ask.answer(cam, "Who came near the red car?")
    b = ask.answer(cam, "Who dented the red car?")
    check_contract(cam, b)
    assert b["refused"] == "blame" and b["type"] == "who_near" and "can't say who did it" in b["sentence"]
    assert [e["object_id"] for e in b["evidence"]] == [e["object_id"] for e in near["evidence"]]
    for q in ("Who scratched the red car?", "Who hit car #7?", "Who stole the red car?"):
        assert ask.answer(cam, q)["refused"] == "blame", q


def test_one_blind_source():
    """With ledger.json present, blind time comes only from it, and totals match the ledger."""
    cam = "ptz_parking"
    p = data.out_dir(cam) / "ledger.json"
    if not p.exists():
        pytest.skip("no ledger.json")
    L = json.loads(p.read_text(encoding="utf-8"))
    idx = ask.index(cam)
    assert idx["ledger"] and all(b["source"] == "ledger" for b in idx["blind"])
    ours = ask.blind_seconds(cam, 0.0, ask.duration(cam))
    tot = L.get("blind_total_s")
    if tot is None:
        tot = sum(b["end"] - b["start"] for b in L["blind"])    # ledger spans do not overlap
    assert abs(float(tot) - ours) <= 1.0
    a = ask.answer(cam, "How many people walked by?")
    assert f"{float(tot):.0f} s" in a["honesty"]
    assert "frozen" not in (a["honesty"] or "")


def test_day_compare_from_diff():
    cam = "ptz_parking"
    p = data.out_dir(cam) / "diff.json"
    D = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    ds = ask.days_of(cam)
    a = ask.answer(cam, f"What changed between {ds[0]['label']} and {ds[-1]['label']}?")
    check_contract(cam, a)
    if D is None or not (D.get("alignment") or {}).get("ok", True):
        assert a["answer"] == "UNKNOWN"
        return
    c = D["counts"]
    assert f"{c['same']} the same car" in a["sentence"]
    if c["left"]:
        assert f"{c['left']} left" in a["answer"]
    if D.get("not_compared"):
        assert f"{len(D['not_compared'])} not compared" in a["sentence"]
    for q2 in ("What changed since yesterday?", "What changed between the first day and the second day?"):
        assert ask.answer(cam, q2)["answer"] == a["answer"], q2


def test_day_words(monkeypatch):
    days = ({"id": "20260901", "label": "1 Sep", "start": 0.0, "end": 900.0},
            {"id": "20260902", "label": "2 Sep", "start": 900.0, "end": 1860.0})
    monkeypatch.setattr(ask, "days_of", lambda cam: days)

    def res(q):
        return ask.resolve_days("x", ask.parse(q))
    assert res("What changed between 1 Sep and 2 Sep?") == [0, 1]
    assert res("How many cars came by yesterday?") == [0]
    assert res("How many cars came by today?") == [1]
    assert res("Was the red car there on the first day?") == [0]
    assert res("How many people on the second day") == [1]
    assert res("what changed on September 2nd") == [1]
    assert res("cars on 20260901") == [0]
    assert res("How many cars came by since 1:19?") == []
    assert ask.parse("What changed since yesterday?")["since_day"]
    ab = ({"id": "A", "label": "Day A", "start": 0.0, "end": 10.0}, {"id": "B", "label": "Day B", "start": 10.0, "end": 20.0})
    monkeypatch.setattr(ask, "days_of", lambda cam: ab)
    assert res("What changed between Day A and Day B?") == [0, 1]
    assert res("What changed yesterday?") == [0]


def test_day_window_ptz():
    cam = "ptz_parking"
    ds = ask.days_of(cam)
    a = ask.answer(cam, "How many cars came by yesterday?")
    check_contract(cam, a)
    assert ds[0]["label"] in a["sentence"]
    assert all(ds[0]["start"] <= e["t"] <= ds[0]["end"] for e in a["evidence"])


@pytest.mark.parametrize("q,refuse", [
    ("Find the man in the black shirt", True),
    ("Which way did the person in red go?", True),
    ("Where did the woman with blonde hair go?", True),
    ("Show me the white man", True),
    ("Which way did the red bike go?", True),
    ("Who came near the red car?", False),
    ("Did someone come near the red car?", False),
    ("How many people walked by?", False),
    ("Which way did the person go?", False),
])
def test_appearance_refusal(q, refuse):
    a = ask.answer("ptz_parking", q)
    check_contract("ptz_parking", a)
    assert (a["refused"] == "appearance") == refuse, (q, a["sentence"])
    if refuse:
        assert a["answer"] == "UNKNOWN" and a["honesty"] == "by design" and "never describes" in a["sentence"]


def test_warm_latency():
    cam = "ptz_parking"
    qs = [q for q, _ in ask.demo_questions(cam)]
    for q in qs:
        ask.answer(cam, q)
    lat = [ask.answer(cam, q)["latency_ms"] for q in qs]
    assert statistics.median(lat) < 200, lat


def test_qa_json_files():
    if not (data.out_dir("ptz_parking") / "qa.json").exists():
        pytest.skip("run python -m stm.ask first")
    for cam in [c for c in data.CAMS if (data.OUT / c / "qa.json").exists()]:
        d = json.loads((data.OUT / cam / "qa.json").read_text(encoding="utf-8"))
        assert d["camera"] == cam and len(d["items"]) == 8
        for it in d["items"]:
            assert isinstance(it["question"], str) and it["type"] == it["answer"]["type"]
            check_contract(cam, it["answer"])
    d = json.loads((data.OUT / "ptz_parking" / "qa.json").read_text(encoding="utf-8"))
    by = {it["question"]: it["answer"] for it in d["items"]}
    assert by["Is the red car still there?"]["answer"] == "UNKNOWN"
    assert by["Who dented the red car?"]["refused"] == "blame"
    assert by["Find the man in the black shirt"]["refused"] == "appearance"
    assert not any("since 00:20" in q for q in by)


def test_server_routes():
    from fastapi.testclient import TestClient
    from stm.server import app
    c = TestClient(app)
    cams = c.get("/api/cams", params={"all": 1}).json()
    assert any(x["camera"] == "car_detection" for x in cams)
    r = c.get("/api/ask", params={"cam": "car_detection", "q": "Which way did the car go?"})
    assert r.status_code == 200
    check_contract("car_detection", r.json(), server=True)
    assert c.get("/api/ask", params={"cam": "nope", "q": "x"}).status_code == 404
    assert c.get("/out/car_detection/meta.json").status_code == 200
    assert c.get("/").status_code == 200
