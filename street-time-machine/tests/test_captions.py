"""Caption lie detector: claim rules on fixed sentences, judging rules, and the real corpus run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stm import captions as C

CAM = "person_bicycle_car_detection"


def counts(text: str) -> dict:
    return C.claimed_counts(C.extract_claims(text))


@pytest.mark.parametrize("text, want", [
    ("Two people, a man and a woman, are walking through a room.", {"person": 2}),
    ("A man and a woman are walking down a long aisle.", {"person": 2}),
    ("Three people enter. The person on the left wears a hat, the person on the right a cap.", {"person": 3}),
    ("A woman walks in. Another woman follows. A man waits. The man sits.", {"person": 3}),
    ("A white car enters the frame and drives into a parking lot.", {"car": 1}),
    ("A person in a black outfit is riding a bicycle in a parking lot.", {"person": 1, "bicycle": 1}),
    ("Shelves are filled with white plates and bowls.", {"bowl": None}),
    ("A black and white cat sleeps.", {"cat": 1}),
    ("2 dogs and three cars.", {"dog": 2, "car": 3}),
    ("A bolt is placed on a black textured surface.", {}),
    ("An orange wall next to a parking lot.", {}),
    # event-pack phrasing: negation, hyphenated adjectives, "vehicles", vague quantities, re-descriptions
    ("There are no pedestrians or moving vehicles present in the scene.", {"person": 0}),
    ("Two vehicles are parked along the curb: a dark-colored SUV on the left and a dark sedan on the right.",
     {"car": 2}),
    ("Two vehicles are parked along the curb.", {"vehicle": 2}),
    ("A car drives by. No other vehicles or pedestrians are visible.", {"car": 1, "person": 0}),
    ("A few pedestrians walk on the sidewalk.", {"person": None}),
    ("A man walks a dog. There are no other people.", {"dog": 1, "person": 1}),
    ("A white pickup truck drives past. The truck is a standard pickup.", {"truck": 1}),
    ("A dog runs. A dog runs. A dog runs.", {"dog": 1}),
    ("The donut appears to be moved by an unseen person.", {"donut": None, "person": None}),
    ("The street appears empty of moving vehicles or pedestrians.", {"person": 0}),
])
def test_claim_rules(text, want):
    assert counts(text) == want


def test_judge_rules():
    c = C.extract_claims
    assert C.judge(c("Two cars park."), {"car": 2}, None, "Two cars park.")["verdict"] == "consistent"
    assert C.judge(c("Two cars park."), {"car": 3}, None, "Two cars park.")["verdict"] == "consistent"
    # "a car" only says at least one: a detector that counts more is not a contradiction, just a count difference
    more = C.judge(c("A car parks."), {"car": 3}, None, "A car parks.")
    assert more["verdict"] == "consistent" and more["count_differs"]
    off = C.judge(c("One car parks."), {"car": 3}, None, "One car parks.")
    assert off["verdict"] == "contradiction" and "off by 2" in off["why"]
    over = C.judge(c("A man, a woman and a boy walk."), {"person": 1}, None, "A man, a woman and a boy walk.")
    assert over["verdict"] == "contradiction"
    # "no people" against a detector that counted people
    none = "There are no people on the street."
    assert C.judge(c(none), {"person": 0}, None, none)["verdict"] == "consistent"
    assert C.judge(c(none), {"person": 1}, None, none)["count_differs"]
    assert C.judge(c(none), {"person": 2}, None, none)["verdict"] == "contradiction"
    # car / van / truck split between the two AIs is amber, not red
    van = "A white truck drives past."
    vj = C.judge(c(van), {"car": 2}, None, van)
    assert vj["verdict"] == "consistent" and vj["class_differs"]
    unseen = C.judge(c("A man walks a dog."), {"person": 1}, {"person": 1}, "A man walks a dog.")
    assert unseen["verdict"] == "contradiction" and "dog" in unseen["why"]
    # a detector that misses a static table is weak evidence, never a contradiction on its own
    table = C.judge(c("A man stands at a table."), {"person": 1}, None, "A man stands at a table.")
    assert table["verdict"] == "consistent" and "Unconfirmed" in table["why"]
    assert C.judge([], {"toothbrush": 1}, None, "A bolt on a surface.")["verdict"] == "unverifiable"
    drone = "A drone flies over a parking lot."
    assert C.judge(c(drone), {}, {}, drone)["verdict"] == "contradiction"
    assert C.judge(c(drone), {"bird": 1}, {}, drone)["verdict"] == "unverifiable"


def test_detector_suspect_rules():
    bolt = C.judge([], {"toothbrush": 2}, None, "A bolt falls on a black surface.")
    assert bolt["detector_suspect"] and "toothbrush" in bolt["detector_reason"]
    lot = "A white car drives out of a parking lot."
    j = C.judge(C.extract_claims(lot), {"car": 1, "boat": 1}, {"car": 1}, lot)
    assert j["verdict"] == "consistent" and j["detector_suspect"] and "boat" in j["detector_reason"]
    room = "A woman walks across a room."
    plain = C.judge(C.extract_claims(room), {"person": 1, "dog": 1}, None, room)
    assert not plain["detector_suspect"]          # no second signal: only "not mentioned"


def test_our_counts_real_tracks():
    # 25-30 s: one cyclist; ByteTrack id 13 flips bicycle/person/tennis racket, majority class must win
    assert C.our_counts(CAM, 25.0, 30.0) == {"bicycle": 1, "person": 1}
    assert C.our_counts(CAM, 20.0, 25.0) == {}
    assert C.our_counts(CAM, 15.0, 20.0) == {"car": 1}


def test_full_run_matches_contract(tmp_path: Path):
    out = C.run(out_root=tmp_path)
    corpus = json.loads(C.CORPUS.read_text(encoding="utf-8"))
    rows = out["rows"]
    assert len(rows) == len(corpus) == 24
    assert set(out["totals"]) == {"consistent", "contradiction", "unverifiable"}
    assert sum(out["totals"].values()) == 24
    keys = {"id", "clip", "start", "end", "caption", "claims", "pipeline_counts", "our_counts", "verdict", "why",
            "frame", "detector_suspect"}
    for r, src in zip(rows, corpus):
        assert keys <= set(r)
        assert r["verdict"] in out["totals"] and r["why"]
        assert r["pipeline_counts"] == json.loads(src["object_counts"])
        assert r["caption"] == src["reasoning_content"].strip()
        assert 0 <= r["start"] < r["end"]
        for cl in r["claims"]:
            assert set(cl) == {"text", "object", "count", "negated"}
            assert cl["object"] in C.COCO or cl["object"] == "vehicle"
            assert cl["count"] is None or cl["count"] > 0 or (cl["count"] == 0 and cl["negated"])
        assert (r["our_counts"] is not None) == (r["clip"] == "person-bicycle-car-detection")
        if r["detector_suspect"]:
            assert r["detector_reason"]
        assert r["frame"] == f"frames/{r['id']}.jpg" and (tmp_path / r["frame"]).stat().st_size > 5000
        if r["frame_ours"]:
            assert (tmp_path / r["frame_ours"]).exists() and r["start"] <= r["frame_ours_t"] < r["end"]
    by = {r["id"]: r for r in rows}
    # the drone caption over an empty car park is the one caption contradicted by both detectors
    drone = by["person-bicycle-car-detection_005"]
    assert drone["verdict"] == "contradiction" and drone["model"] == "nvidia/Cosmos3-Edge"
    assert drone["strip"] == "frames/person-bicycle-car-detection_005_strip.jpg" and len(drone["strip_t"]) == 6
    assert (tmp_path / drone["strip"]).stat().st_size > 5000
    assert all(drone["start"] <= t < drone["end"] for t in drone["strip_t"])
    assert by["store-aisle-detection_005"]["count_differs"] and by["people-detection_005"]["count_differs"]
    assert out["practice"] is True and out["camera"] is None
    assert by["bolt-detection_001"]["verdict"] == "unverifiable" and by["bolt-detection_001"]["detector_suspect"]
    assert by["people-detection_006"]["claims"][0]["count"] == 3
    saved = json.loads((tmp_path / "captions.json").read_text(encoding="utf-8"))
    assert saved["totals"] == out["totals"]
    assert (tmp_path / "captions_summary.txt").read_text(encoding="utf-8").startswith("captions: 24 rows")


def _pack(tmp_path: Path) -> Path:
    """A two-day event pack in the segments.json shape, with chunk offsets next to it."""
    def row(chunk, num, cap, counts):
        return {"filename": f"{chunk}_segment_{num:03d}_of_006.mp4", "source": f"s3://x/{chunk}_{num}.mp4",
                "reasoning_content": cap, "segment_number": str(num), "segment_start_sec": str(5.0 * (num - 1)),
                "segment_end_sec": str(5.0 * num), "object_counts": json.dumps(counts), "camera_id": "cam-x",
                "chunk": chunk, "cosmos_model": "m", "perception_source": "yolo11_coco"}
    a, b = "20260101_000000_hood_20250901_chunk_0000", "20260101_000500_hood_20250902_chunk_0000"
    pack = {"chunks": [f"s3://x/{a}.mp4", f"s3://x/{b}.mp4"], "segments": [
        row(a, 1, "A person walks a dog past two parked cars.", {"person": 1, "car": 2}),
        row(a, 2, "There are no people. A car is parked.", {"car": 3}),
        row(b, 1, "A person is seen walking along the street.", {"car": 3, "donut": 1}),
    ]}
    (tmp_path / "segments.json").write_text(json.dumps(pack), encoding="utf-8")
    offs = {"offsets": {a: 0.0, b: 100.0}, "days": [["20250901", "1 Sep", 0.0, 100.0], ["20250902", "2 Sep", 100.0, None]]}
    (tmp_path / "chunk_offsets.json").write_text(json.dumps(offs), encoding="utf-8")
    return tmp_path / "segments.json"


def test_event_pack_shape_without_video(tmp_path: Path):
    out = C.run(corpus=_pack(tmp_path), frames=False, out_root=tmp_path / "o")
    assert out["camera"] == "cam-x" and out["practice"] is False
    assert (tmp_path / "o" / "cam-x.json").exists() and not (tmp_path / "o" / "captions.json").exists()
    assert out["days"] == [{"id": "20250901", "label": "1 Sep"}, {"id": "20250902", "label": "2 Sep"}]
    r1, r2, r3 = out["rows"]
    assert (r1["day"], r1["start"], r1["end"], r1["day_label"]) == ("20250901", 0.0, 5.0, "1 Sep")
    assert (r3["day"], r3["start"], r3["seg_start"]) == ("20250902", 100.0, 0.0)
    assert all(r["our_counts"] is None and r["frame"] is None and r["strip"] is None for r in out["rows"])
    assert r1["verdict"] == "contradiction" and "dog" in r1["why"]          # the detector saw no dog
    assert r2["verdict"] == "consistent" and r2["count_differs"]            # "a car" vs 3: count differs only
    assert r3["verdict"] == "contradiction" and r3["detector_suspect"]       # person unseen; donut on a street?
    assert "same YOLO" in out["witness_note"]
    assert out["totals"] == {"consistent": 1, "contradiction": 2, "unverifiable": 0}


@pytest.mark.skipif(not C.PACKD.exists(), reason="event pack not synced")
def test_real_event_pack(tmp_path: Path):
    src = json.loads(C.PACKD.read_text(encoding="utf-8"))
    out = C.run(corpus=C.PACKD, frames=False, out_root=tmp_path)
    assert len(out["rows"]) == len(src["segments"]) and sum(out["totals"].values()) == len(src["segments"])
    assert all(r["our_counts"] is None for r in out["rows"])               # one witness, never presented as two
    assert {d["id"] for d in out["days"]} == {r["day"] for r in out["rows"]}
    assert all(r["start"] < r["end"] for r in out["rows"])
    assert out["totals"]["contradiction"] < len(src["segments"]) // 10     # contradictions are the exception
