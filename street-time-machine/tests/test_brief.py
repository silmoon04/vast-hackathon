"""Tests for stm.brief: rules on small fixtures shaped like each upstream contract, plus real data."""
from __future__ import annotations

import json
import re

import pytest

from stm import brief, data

KEYS = {"camera", "headline", "bullets", "honesty", "generated_from", "missing"}


def fx_ledger(blind=((10.0, 30.0, "dark"), (100.0, 120.0, "blocked")), seconds=316):
    return {"camera": "fx", "seconds": seconds,
            "blind": [{"start": s, "end": e, "reason": r, "score": 0.9} for s, e, r in blind],
            "spots": [{"spot_id": "S1", "box": [0, 0, 10, 10],
                       "spans": [{"start": 0.0, "end": 50.0, "state": "present", "object_id": 7, "reason": None}],
                       "changes": [{"t": 50.0, "kind": "left", "before": "crops/7.jpg", "after": None,
                                    "same_vehicle": None, "similarity": None}]}],
            "per_second": [{"t": i, "visible": True, "reason": None} for i in range(seconds)]}


def fx_diff():
    def ch(kind, a, b):
        return {"kind": kind, "box": [1, 2, 3, 4], "similarity": 0.5, "note": "",
                "a": {"object_id": a, "crop": f"crops/{a}.jpg", "t": 12.5} if a else None,
                "b": {"object_id": b, "crop": f"crops/{b}.jpg", "t": 200.25} if b else None}
    changes = [ch("left", 1, None), ch("left", 2, None), ch("left", 3, None),
               ch("arrived", None, 4), ch("arrived", None, 5), ch("same", 6, 8)]
    return {"camera": "fx", "day_a": "A", "day_b": "B", "changes": changes,
            "counts": {"same": 1, "arrived": 2, "left": 3, "moved": 0, "swapped": 0},
            "background_a": "background_A.jpg", "background_b": "background_B.jpg"}


def fx_near():
    v = lambda i, n, g, t: {"object_id": i, "name": n, "closest_t": t, "gap_px": 4.0, "gap_rel": g,
                            "enter_t": t - 1, "exit_t": t + 1, "crop": f"near/{i}.jpg"}
    return {"camera": "fx", "targets": [
        {"object_id": 6, "name": "car", "box": [1, 2, 3, 4], "start": 0.0, "end": 155.0, "crop": "near/6.jpg",
         "visitors": [v(20, "person", 0.2, 33.0), v(21, "person", 0.1, 44.0), v(22, "truck", 0.4, 55.0)],
         "blind": [], "summary": "x"},
        {"object_id": 1, "name": "car", "box": [1, 2, 3, 4], "start": 0.0, "end": 90.0, "crop": "near/1.jpg",
         "visitors": [], "blind": [], "summary": "y"}]}


def fx_moments():
    return {"camera": "fx", "moments": [
        {"t": 61.0, "end": 64.0, "kind": "crowd", "title": "9 people at once", "detail": "", "object_ids": [20],
         "thumb": "moments/1.jpg", "box": None, "score": 0.9},
        {"t": 70.0, "end": 72.0, "kind": "close_pass", "title": "A person passed close to a moving car",
         "detail": "", "object_ids": [21], "thumb": "moments/2.jpg", "box": None, "score": 0.8},
        {"t": 10.0, "end": 30.0, "kind": "blind", "title": "Camera went dark", "detail": "", "object_ids": [],
         "thumb": "moments/3.jpg", "box": None, "score": 1.0}]}


def fx_flow():
    return {"camera": "fx", "image": "flow.jpg", "w": 1280, "h": 720,
            "lanes": [{"lane_id": "L1", "name": "left to right along the bottom", "group": "vehicle", "count": 17,
                       "from": [0, 600], "to": [1200, 600], "path": [[0, 600]] * 20, "median_speed_px_s": 80.5,
                       "example_ids": [30]},
                      {"lane_id": "L2", "name": "up the middle", "group": "person", "count": 5,
                       "from": [600, 700], "to": [600, 100], "path": [[600, 700]] * 20, "median_speed_px_s": 20.0,
                       "example_ids": [20]}],
            "per_minute": [{"t": 0, "person": 3, "vehicle": 4}],
            "busiest": {"t": 61.0, "count": 11, "frame": "frames/busiest.jpg"}}


def fx_synopsis():
    return {"camera": "fx", "src_duration": 80.0, "out_duration": 30.0, "fps": 12.0, "w": 960, "h": 540,
            "ghosts": [{"object_id": i, "name": "car", "day": "A", "src_start": 0.0, "src_end": 1.0,
                        "out_start": 0.0, "out_end": 1.0, "label": "00:00", "boxes": []} for i in range(13)]}


def fx_index():
    return {"camera": "fx", "objects": [{"object_id": i} for i in range(25)],
            "colours": {"white": 12, "black": 8, "red": 5}}


def fx_meta():
    return {"camera": "fx", "duration": 316.13, "simulated": ["Day A / Day B are two halves of one clip"],
            "days": [{"id": "A", "label": "Day A", "start": 0.0, "end": 158.0},
                     {"id": "B", "label": "Day B", "start": 158.0, "end": 316.13}]}


def fx_counts():
    return {"total": 14, "moving": {"person": 4, "car": 2}, "parked": 8, "parked_names": {"car": 8},
            "names": {i: "car" for i in range(1, 9)} | {20: "person", 21: "person", 22: "truck"}, "last_t": 300.0}


def all_inputs():
    return {"ledger": fx_ledger(), "diff": fx_diff(), "near": fx_near(), "moments": fx_moments(),
            "flow": fx_flow(), "synopsis": fx_synopsis(), "index": fx_index(), "meta": fx_meta()}


def numbers_in(obj, acc=None):
    """Every number in a JSON-like object, plus list lengths: the numbers a brief may copy."""
    acc = set() if acc is None else acc
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.add(round(float(obj), 2))
    elif isinstance(obj, str):
        acc.update(float(x) for x in re.findall(r"\d+(?:\.\d+)?", obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            numbers_in(k, acc) if isinstance(k, str) else None
            numbers_in(v, acc)
    elif isinstance(obj, (list, tuple)):
        acc.add(float(len(obj)))
        for v in obj:
            numbers_in(v, acc)
    return acc


def text_numbers(s):
    s = re.sub(r"\b\d\d:\d\d\b", " ", s)  # mm:ss times come from t values
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]


def test_full_fixture_headline_and_shape():
    b = brief.build("fx", all_inputs(), fx_counts(), fx_meta()["simulated"], [])
    assert set(b) == KEYS
    assert b["headline"] == "3 cars left and 2 arrived between Day A and Day B; the camera could not see for 40 s."
    assert len(b["bullets"]) == brief.MAX_BULLETS
    for x in b["bullets"]:
        assert set(x) == {"text", "t", "kind", "crop"}
    kinds = [x["kind"] for x in b["bullets"]]
    assert kinds[:2] == ["left", "arrived"] and "blind" not in kinds
    assert b["bullets"][0]["crop"] == "crops/1.jpg" and b["bullets"][0]["t"] == 12.5
    assert any("SIMULATED" in h for h in b["honesty"])
    assert b["honesty"][0].startswith("The camera could not see for 40 s in total, in 2 stretches")
    assert set(b["generated_from"]) >= {f"{k}.json" for k in brief.INPUTS}
    json.dumps(b)  # serialisable


def test_every_number_is_copied_from_inputs():
    ins, counts = all_inputs(), fx_counts()
    b = brief.build("fx", ins, counts, fx_meta()["simulated"], [])
    allowed = numbers_in(ins) | numbers_in(counts) | {40.0}  # 40 = summed blind time (20 + 20)
    for s in [b["headline"]] + [x["text"] for x in b["bullets"]]:
        for n in text_numbers(s):
            assert n in allowed, (n, s)


def test_near_bullet_counts_visitors():
    b = brief.build("fx", {"near": fx_near()}, None, [], [])
    nb = next(x for x in b["bullets"] if x["kind"] == "near")
    assert "3 came close (2 people and 1 truck)" in nb["text"]
    assert nb["t"] == 44.0 and nb["crop"] == "near/6.jpg"
    assert any("1 of the 2 parked vehicles" in h for h in b["honesty"])


def test_flow_moments_index_bullets():
    b = brief.build("fx", {"moments": fx_moments(), "flow": fx_flow(), "index": fx_index(),
                           "synopsis": fx_synopsis()}, None, [], [])
    texts = " | ".join(x["text"] for x in b["bullets"])
    assert "9 people at once at 01:01." in texts
    assert "Busiest route: 17 vehicles went left to right along the bottom." in texts
    assert "A 30 s replay shows 13 cars from the 80 s recording at once." in texts
    assert "white (12)" in texts
    assert "Camera went dark" not in texts  # blind moments belong in honesty, not bullets
    assert "Busiest moment: 11 people and vehicles on screen at 01:01." in texts
    f = fx_flow()
    f["busiest"].update(person=7, vehicle=4)
    b2 = brief.build("fx", {"flow": f}, None, [], [])
    assert any("Busiest moment: 7 people and 4 vehicles on screen at 01:01." == x["text"] for x in b2["bullets"])


def test_moments_one_per_kind():
    m = fx_moments()
    m["moments"] += [dict(m["moments"][1], t=99.0, score=0.95), dict(m["moments"][1], t=98.0, score=0.1)]
    b = brief.build("fx", {"moments": m}, None, [], [])
    kinds = [x["kind"] for x in b["bullets"]]
    assert kinds.count("close_pass") == 1 and kinds[0] == "crowd"
    assert next(x for x in b["bullets"] if x["kind"] == "close_pass")["t"] == 99.0


def test_only_object_counts():
    missing = [f"{k}.json" for k in brief.INPUTS]
    b = brief.build("fx", {}, fx_counts(), [], missing)
    assert b["headline"] == "4 people and 2 cars went by and 8 vehicles stayed parked."
    assert b["missing"] == missing
    assert any("Not checked yet whether the camera could see" in h for h in b["honesty"])
    assert [x["kind"] for x in b["bullets"]] == ["count"]


def test_nothing_at_all():
    b = brief.build("fx", {}, None, [], ["ledger.json"])
    assert b["headline"] == "No results yet for this camera."
    assert b["bullets"] == [] and b["generated_from"] == []


def test_ledger_spots_without_diff():
    led = fx_ledger(blind=())
    led["spots"][0]["changes"] += [{"t": 80.0, "kind": "arrived", "before": None, "after": "crops/9.jpg",
                                    "same_vehicle": None, "similarity": None},
                                   {"t": 90.0, "kind": "arrived", "before": None, "after": "crops/10.jpg",
                                    "same_vehicle": None, "similarity": None}]
    b = brief.build("fx", {"ledger": led}, None, [], [])
    assert b["headline"] == "2 vehicles arrived and 1 left in the parking spots; the camera could see the whole time."
    arr = next(x for x in b["bullets"] if x["kind"] == "arrived")
    assert arr["t"] == 80.0 and arr["crop"] == "crops/9.jpg"


def test_mostly_blind_is_flagged():
    led = fx_ledger(blind=((0.0, 29.0, "blocked"),), seconds=31)
    led["spots"] = []
    b = brief.build("fx", {"ledger": led, "meta": {"duration": 30.16}}, fx_counts(), [], [])
    assert b["headline"].endswith("the camera could not see for 29 s of the 30 s.")
    assert any("one of the two is wrong" in h for h in b["honesty"])


def test_overlapping_blind_spans_merge():
    spans = brief.merged_blind({"blind": [{"start": 0, "end": 10, "reason": "dark"},
                                          {"start": 5, "end": 12, "reason": "blur"},
                                          {"start": 20, "end": 21, "reason": "dark"}]})
    assert brief.blind_seconds(spans) == pytest.approx(13.0)


def test_load_inputs_skips_unreadable(tmp_path):
    (tmp_path / "ledger.json").write_text(json.dumps(fx_ledger()), encoding="utf-8")
    (tmp_path / "diff.json").write_text('{"camera": "fx", "chan', encoding="utf-8")  # half-written
    found, missing = brief.load_inputs("fx", tmp_path)
    assert set(found) == {"ledger"}
    assert "diff.json (unreadable)" in missing and "near.json" in missing


def test_missing_crop_files_become_null(tmp_path):
    (tmp_path / "crops").mkdir()
    (tmp_path / "crops" / "1.jpg").write_bytes(b"x")
    b = brief.build("fx", {"diff": fx_diff()}, None, [], [], root=tmp_path)
    crops = {x["kind"]: x["crop"] for x in b["bullets"]}
    assert crops["left"] == "crops/1.jpg" and crops["arrived"] is None


def _have_tracks(cam):
    return any(data.TRACK_CACHE.glob(f"{cam}__det__*.json.gz"))


@pytest.mark.skipif(not _have_tracks("person_bicycle_car_detection"), reason="no cached tracks")
def test_real_objects_headline():
    objs = data.objects("person_bicycle_car_detection")
    c = brief.object_counts(objs)
    assert c["total"] == len(objs) and c["parked"] == int(objs.parked.sum())
    b = brief.build("person_bicycle_car_detection", {}, c, [], [])
    moving = objs[~objs.parked]
    top = moving.name.value_counts()
    for name, n in list(top.items())[:3]:
        assert f"{n} {brief.noun(name, n)}" in b["headline"]


def test_real_outputs_ptz_parking():
    """Read whatever real upstream outputs exist (read-only) and check the brief copies their counts."""
    ins, missing = brief.load_inputs("ptz_parking")
    b = brief.build("ptz_parking", ins, None, data.CAMS["ptz_parking"]["simulated"], missing)
    assert any(h.startswith("SIMULATED") for h in b["honesty"])
    assert len(b["bullets"]) <= brief.MAX_BULLETS
    if "diff" in ins:
        left = int(ins["diff"]["counts"].get("left", 0))
        if left:
            assert b["headline"].startswith(f"{left} ")
    if "ledger" in ins:
        tot = brief.ledger_blind_total(ins["ledger"])
        if tot >= 0.5:
            assert brief.dur(tot) in b["headline"]
        if ins["ledger"].get("blind_total_s") is not None:
            assert tot == float(ins["ledger"]["blind_total_s"])
    assert not any(x["kind"] == "long_stay" or x["text"].startswith("Person stays") for x in b["bullets"])


def test_bullet_inside_blind_time_is_flagged():
    f = fx_flow()
    f["busiest"]["t"] = 105.0  # inside the 100-120 s blocked stretch of fx_ledger()
    b = brief.build("fx", {"flow": f, "ledger": fx_ledger()}, None, [], [])
    assert any(h.startswith('"Busiest moment" at 01:45 falls in a stretch') and "view blocked" in h
               for h in b["honesty"])


def test_diff_none_arrived_and_all_zero():
    d = fx_diff()
    d["counts"].update(arrived=0)
    b = brief.build("fx", {"diff": d}, fx_counts(), [], [])
    assert b["headline"] == "3 cars left and none arrived between Day A and Day B."
    d["counts"] = {"same": 1, "arrived": 0, "left": 0, "moved": 0, "swapped": 0}
    b = brief.build("fx", {"diff": d}, fx_counts(), [], [])
    assert b["headline"] == "No parked car left or arrived between Day A and Day B."


def test_touching_blind_spans_count_as_one_stretch():
    led = fx_ledger(blind=((156.0, 157.0, "blocked"), (157.0, 158.0, "blur"), (158.0, 237.0, "blocked"),
                           (69.0, 70.0, "no_signal")))
    b = brief.build("fx", {"ledger": led}, None, [], [])
    assert b["honesty"][0].startswith("The camera could not see for 82 s in total, in 2 stretches")
    assert "the longest was 02:36 to 03:57" in b["honesty"][0]


def test_still_tracks_are_not_counted_as_going_by():
    import pandas as pd
    objs = pd.DataFrame([
        {"id": 1, "name": "car", "parked": False, "travel_px": 500.0, "x1": 0, "x2": 100, "last_t": 5.0},
        {"id": 2, "name": "car", "parked": False, "travel_px": 12.0, "x1": 0, "x2": 100, "last_t": 6.0},
        {"id": 3, "name": "car", "parked": True, "travel_px": 3.0, "x1": 0, "x2": 100, "last_t": 9.0}])
    c = brief.object_counts(objs)
    assert c["moving"] == {"car": 1} and c["still"] == 1 and c["parked"] == 1
    b = brief.build("fx", {}, c, [], [])
    assert b["headline"].startswith("1 car went by")
    assert any(h.startswith("1 more track hardly moved") for h in b["honesty"])


def test_ledger_totals_and_camera_motion_words():
    """The ledger's own blind_total_s / blind_by_reason / blind_by_day win over summing spans."""
    led = fx_ledger(blind=((68.0, 70.0, "no_signal"), (156.0, 168.0, "moved"), (168.0, 206.0, "turned_away")))
    led.update(blind_total_s=97.0, blind_by_reason={"turned_away": 50.0, "moved": 45.0, "no_signal": 2.0},
               blind_by_day=[{"day": "A", "label": "Day A", "start": 0.0, "end": 158.0, "blind_s": 4.0,
                              "by_reason": {"no_signal": 2.0, "moved": 2.0}},
                             {"day": "B", "label": "Day B", "start": 158.0, "end": 316.13, "blind_s": 93.0,
                              "by_reason": {"moved": 43.0, "turned_away": 50.0}}])
    b = brief.build("fx", {"ledger": led}, None, [], [])
    assert b["headline"].endswith("the camera could not see for 1 min 37 s.")
    h = b["honesty"][0]
    assert h.startswith("The camera could not see for 1 min 37 s in total (4 s on Day A and 1 min 33 s on Day B), in 2 stretches")
    assert "50 s camera turned away, 45 s camera moving and 2 s no picture" in h
    assert brief.REASON_WORDS["turned_away"] == "camera turned away" and brief.REASON_WORDS["moved"] == "camera moving"
    del led["blind_by_reason"]  # by_day only: reasons summed across days
    assert brief.ledger_blind_by_reason(led) == {"turned_away": 50.0, "moved": 45.0, "no_signal": 2.0}
    del led["blind_total_s"], led["blind_by_day"]  # old ledger: spans summed
    assert brief.ledger_blind_total(led) == pytest.approx(52.0)


def test_colour_bullet_counts_coloured_vehicles_only():
    ix = {"camera": "fx", "colours": {"black": 40, "white": 20, "null": 3},
          "n_coloured": 60, "n_vehicles": 84,
          "objects": [{"object_id": i, "name": "car", "colour": None} for i in range(24)]}
    b = brief.build("fx", {"index": ix}, None, [], [])
    c = next(x for x in b["bullets"] if x["kind"] == "colour")
    assert c["text"] == "Most common colour: black (40) and white (20) of 60 vehicles."
    assert "null" not in c["text"] and "people" not in c["text"]
    assert any(h.startswith("24 of 84 vehicles have no colour") for h in b["honesty"])
    # an older index (no n_coloured) also coloured people: only its coloured vehicle rows count
    old = {"camera": "fx", "colours": {"grey": 3, "black": 1},
           "objects": [{"object_id": 1, "name": "car", "colour": "grey"}, {"object_id": 2, "name": "car", "colour": "grey"},
                       {"object_id": 3, "name": "person", "colour": "grey"}, {"object_id": 4, "name": "truck", "colour": "black"},
                       {"object_id": 5, "name": "car", "colour": None}]}
    assert brief.colour_counts(old) == ([("grey", 2), ("black", 1)], 3)


def test_replay_uses_synopsis_window():
    syn = fx_synopsis() | {"src_duration": 316.13, "src_window": [0.0, 155.0], "view_label": "Day A view"}
    b = brief.build("fx", {"synopsis": syn}, None, [], [])
    r = next(x for x in b["bullets"] if x["kind"] == "replay")
    assert r["text"] == "A 30 s replay shows 13 cars from 2 min 35 s of the 5 min 16 s recording (Day A view) at once."
    syn["src_window"] = [0.0, 316.13]
    r = next(x for x in brief.build("fx", {"synopsis": syn}, None, [], [])["bullets"] if x["kind"] == "replay")
    assert r["text"] == "A 30 s replay shows 13 cars from the 5 min 16 s recording at once."


def test_long_stay_is_never_a_bullet():
    m = fx_moments()
    m["moments"].append({"t": 106.47, "end": 125.0, "kind": "long_stay", "title": "Person stays 19 s", "detail": "",
                         "object_ids": [20], "thumb": "moments/4.jpg", "box": None, "score": 0.99})
    b = brief.build("fx", {"moments": m}, None, [], [])
    assert not any(x["kind"] == "long_stay" or x["text"].startswith("Person stays") for x in b["bullets"])
    assert [x["kind"] for x in b["bullets"]] == ["crowd", "close_pass"]
