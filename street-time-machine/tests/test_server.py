"""stm.server: camera order, live and precomputed answers, clean 404s, no-store headers, background warm-up."""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("STM_WARM", "0")      # a plain TestClient never runs the lifespan; this is belt and braces

import pytest
from fastapi.testclient import TestClient

from stm import ask, data, server

PACKD = "neighborhood_cam-1"
ANS_KEYS = {"type", "answer", "sentence", "latency_ms", "evidence", "honesty"}


@pytest.fixture()
def client():
    return TestClient(server.app)


def _meta(root, cam, days):
    d = root / cam
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"camera": cam, "title": f"{cam} title", "duration": 100.0, "w": 1280,
                                             "h": 720, "days": days, "simulated": []}), encoding="utf-8")
    return d


@pytest.fixture()
def fake_out(tmp_path, monkeypatch):
    """A fixture out/ with the event camera (real day ids), the two demo cams and one dev cam."""
    real = [{"id": "20260901", "label": "1 Sep", "start": 0.0, "end": 50.0},
            {"id": "20260902", "label": "2 Sep", "start": 50.0, "end": 100.0}]
    d = _meta(tmp_path, PACKD, real)
    (d / "qa.json").write_text(json.dumps({"camera": PACKD, "items": [
        {"question": "How many cars came by?", "t": None, "type": "count_since",
         "answer": {"type": "count_since", "answer": "41", "sentence": "41 cars came by.", "latency_ms": 3942.0,
                    "evidence": [], "honesty": None}},
        {"question": "Is car #7 still there at 00:30?", "t": 30.0, "type": "still_there",
         "answer": {"type": "still_there", "answer": "YES", "sentence": "Yes.", "latency_ms": 12.0,
                    "evidence": [{"object_id": 7, "t": 30.0, "crop": None, "box": [1, 2, 3, 4]}], "honesty": None}}]}),
        encoding="utf-8")
    for c in ("ptz_parking", "yonge_dundas", "car_detection"):
        _meta(tmp_path, c, [{"id": "A", "label": "Clip", "start": 0.0, "end": 100.0}])
    monkeypatch.setattr(server, "OUT", tmp_path)
    monkeypatch.delitem(data.CAMS, PACKD, raising=False)     # this laptop has no Pack D tracks
    server._LIVE.discard(PACKD)
    return tmp_path


def test_cam_order_event_camera_first(client, fake_out):
    cams = client.get("/api/cams").json()
    assert [c["camera"] for c in cams] == [PACKD, "ptz_parking", "yonge_dundas"]
    assert cams[0]["default"] is True and not any(c["default"] for c in cams[1:])
    assert [d["id"] for d in cams[0]["days"]] == ["20260901", "20260902"]     # read from meta, never assumed A/B
    assert [d["label"] for d in cams[0]["days"]] == ["1 Sep", "2 Sep"]
    assert cams[0]["ask"] == "precomputed" and cams[0]["has_qa"] is True
    every = [c["camera"] for c in client.get("/api/cams", params={"all": 1}).json()]
    assert every[:3] == [PACKD, "ptz_parking", "yonge_dundas"] and "car_detection" in every


def test_cam_order_without_event_camera(client, fake_out):
    import shutil
    shutil.rmtree(fake_out / PACKD)
    cams = client.get("/api/cams").json()
    assert [c["camera"] for c in cams] == ["ptz_parking", "yonge_dundas"] and cams[0]["default"]


def test_real_cams_curated(client):
    ids = [c["camera"] for c in client.get("/api/cams").json()]
    assert "ptz_parking" in ids and "car_detection" not in ids and "rochester_night" not in ids
    if PACKD in ids:
        assert ids[0] == PACKD
    assert ids.index("ptz_parking") < ids.index("yonge_dundas") if "yonge_dundas" in ids else True


def test_precomputed_exact_match_and_404(client, fake_out):
    r = client.get("/api/ask", params={"cam": PACKD, "q": "how many cars came by"})   # case/punctuation-insensitive
    assert r.status_code == 200 and r.headers["x-stm-source"] == "precomputed"
    a = r.json()
    assert set(a) == ANS_KEYS and a["answer"] == "41"
    assert a["latency_ms"] is None              # never re-serve qa.json's 3942 ms as if measured now
    r = client.get("/api/ask", params={"cam": PACKD, "q": "Is car #7 still there at 00:30?", "t": 30})
    assert r.status_code == 200 and r.json()["answer"] == "YES"
    r = client.get("/api/ask", params={"cam": PACKD, "q": "Is car #7 still there at 00:30?", "t": 90})
    assert r.status_code == 404                 # same words at another moment is not the stored answer
    r = client.get("/api/ask", params={"cam": PACKD, "q": "Who came near the blue van?"})
    assert r.status_code == 404 and "sample questions" in r.json()["detail"]


def test_no_tracks_no_qa_is_clean_404(client, fake_out):
    (fake_out / PACKD / "qa.json").unlink()
    r = client.get("/api/ask", params={"cam": PACKD, "q": "How many cars came by?"})
    assert r.status_code == 404 and isinstance(r.json()["detail"], str)
    assert str(data.TRACK_CACHE) not in r.text   # no local paths / env values in the message


def test_live_needs_tracks_video_and_index(fake_out, monkeypatch, tmp_path_factory):
    """Pack D synced with tracks but no fresh ask_index.json: never trigger a 31-minute scan inside a request."""
    cache = tmp_path_factory.mktemp("tracks")
    vid = cache / "v.mp4"
    vid.write_bytes(b"x")
    monkeypatch.setattr(data, "TRACK_CACHE", cache)
    monkeypatch.setattr(data, "OUT", fake_out)
    monkeypatch.setitem(data.CAMS, PACKD, {"video": vid, "days": [], "title": "t", "simulated": [], "credit": ""})
    assert server.ask_mode(PACKD) == "precomputed"                       # no tracks here
    (cache / f"{PACKD}__det__s1__c25.json.gz").write_bytes(b"")
    meta = json.loads((fake_out / PACKD / "meta.json").read_text(encoding="utf-8"))
    (fake_out / PACKD / "meta.json").write_text(json.dumps(dict(meta, duration=1860.0)), encoding="utf-8")
    assert server.ask_mode(PACKD) == "precomputed"                       # long clip, no index: would rescan
    (fake_out / PACKD / "ask_index.json").write_text(json.dumps({"version": ask.INDEX_VERSION}), encoding="utf-8")
    assert server.ask_mode(PACKD) == "live"
    server._LIVE.discard(PACKD)


def test_unknown_and_bad_cam(client):
    assert client.get("/api/ask", params={"cam": "nope", "q": "x"}).status_code == 404
    assert client.get("/api/ask", params={"cam": "../stm", "q": "x"}).status_code == 404
    assert client.get("/api/ask", params={"cam": "ptz_parking"}).status_code == 422


def test_live_answer_matches_qa_and_measures_latency(client):
    qa = json.loads((data.OUT / "ptz_parking" / "qa.json").read_text(encoding="utf-8"))
    it = next(i for i in qa["items"] if i["question"] == "How many people walked by?")
    r = client.get("/api/ask", params={"cam": "ptz_parking", "q": it["question"]})
    assert r.status_code == 200 and r.headers["x-stm-source"] == "live"
    a = r.json()
    assert set(a) == ANS_KEYS and a["answer"] == it["answer"]["answer"]
    assert isinstance(a["latency_ms"], float) and a["latency_ms"] >= float(r.headers["x-stm-engine-ms"])
    t0 = time.perf_counter()
    r = client.get("/api/ask", params={"cam": "ptz_parking", "q": "Is the black car still there?"})
    assert r.status_code == 200 and (time.perf_counter() - t0) < 2.0     # warm: no cold reload


def test_no_store_headers(client):
    for url in ("/", "/out/ptz_parking/meta.json", "/api/cams"):
        r = client.get(url)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-store", url


def test_video_range_request(client):
    if not (data.OUT / "ptz_parking" / "synopsis.mp4").exists():
        pytest.skip("no synopsis.mp4")
    r = client.get("/out/ptz_parking/synopsis.mp4", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100      # the browser can seek the video


def test_warm_cam_stats(fake_out, monkeypatch):
    calls = []

    def fake_answer(cam, q, t=None):
        calls.append(q)
        return {"answer": "40" if q.startswith("How many") else "YES"}
    monkeypatch.setattr(ask, "answer", fake_answer)
    monkeypatch.setattr(server, "ask_mode", lambda cam: "live")
    st = server.warm_cam(PACKD)
    assert st["state"] == "done" and st["n"] == 2 and st["p50_ms"] is not None and st["max_ms"] >= st["p50_ms"]
    assert len(calls) == 3                                    # one load call + every qa.json question
    assert [d["question"] for d in st["drift"]] == ["How many cars came by?"]    # 40 live vs 41 stored


def test_startup_serves_before_warm_finishes(monkeypatch, fake_out):
    monkeypatch.setenv("STM_WARM", "1")
    monkeypatch.setattr(server, "_WARM", {"enabled": False, "cams": {}, "total": None})
    monkeypatch.setattr(server, "warm_cam", lambda cam: time.sleep(1.5) or {"state": "done"})
    t0 = time.perf_counter()
    with TestClient(server.app) as c:
        assert c.get("/api/cams").status_code == 200
        assert time.perf_counter() - t0 < 1.0               # the warm-up runs in the background
        assert set(c.get("/api/health").json()["warm"]) == {PACKD, "ptz_parking", "yonge_dundas"}
