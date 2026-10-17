import json

import cv2
import numpy as np
import pandas as pd
import pytest

from stm import data as D
from stm import synopsis as S

CAM = "person_bicycle_car_detection"  # small real clip: builds in a few seconds


@pytest.fixture(scope="module")
def built():
    res = S.build(CAM, verbose=False)
    return res, json.loads((D.out_dir(CAM) / "synopsis.json").read_text(encoding="utf-8"))


def test_contract_keys_and_sane_values(built):
    res, j = built
    assert set(j) >= {"camera", "src_duration", "out_duration", "fps", "w", "h", "ghosts"}
    assert j["camera"] == CAM and j["w"] == 960 and j["fps"] == 12.0 and j["h"] % 2 == 0
    assert 0 < j["out_duration"] <= 60.0 and j["out_duration"] < j["src_duration"]
    assert abs(j["src_duration"] - D.info(CAM)["duration"]) < 0.01
    ids = {int(i) for i in D.objects(CAM).id}
    assert j["ghosts"]
    for g in j["ghosts"]:
        assert set(g) >= {"object_id", "name", "day", "src_start", "src_end", "out_start", "out_end", "label", "boxes"}
        assert g["object_id"] in ids
        assert g["src_end"] - g["src_start"] >= S.MIN_SPAN
        assert 0 <= g["out_start"] < g["out_end"] <= j["out_duration"] + 1e-6
        assert len(g["label"]) == 5 and g["label"][2] == ":"
        assert g["label"] == S.mmss(g["src_start"])
        ts = [b[0] for b in g["boxes"]]
        assert ts == sorted(ts) and all(g["out_start"] - 1e-6 <= t <= g["out_end"] + 1e-6 for t in ts)
        for _, x1, y1, x2, y2 in g["boxes"]:
            assert 0 <= x1 < x2 <= j["w"] and 0 <= y1 < y2 <= j["h"]
    vid = D.out_dir(CAM) / "synopsis.mp4"
    cap = cv2.VideoCapture(str(vid))
    n, w, h = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert (w, h) == (j["w"], j["h"]) and abs(n / 12.0 - j["out_duration"]) <= 0.2
    assert (D.out_dir(CAM) / "synopsis_summary.txt").exists()


def test_ghost_pixels_match_source(built):
    """The rendered ghost at its box must look like the source frame at the ghost's source time."""
    _, j = built
    g = max((g for g in j["ghosts"] if not g.get("parked")), key=lambda g: (g["boxes"][0][3] - g["boxes"][0][1]) *
            (g["boxes"][0][4] - g["boxes"][0][2]))
    b = g["boxes"][len(g["boxes"]) // 2]
    t_out, x1, y1, x2, y2 = b
    cap = cv2.VideoCapture(str(D.out_dir(CAM) / "synopsis.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_out * j["fps"])))
    ok, frame = cap.read()
    cap.release()
    assert ok
    sp = (g.get("speed") or 1.0)
    src_t = g["src_start"] + (t_out - g["out_start"]) * sp
    tr = D.tracks(CAM)
    r = tr[tr.tid == g["object_id"]].iloc[lambda d: (d.t - src_t).abs().argsort()[:1]].iloc[0]
    src = D.frame_at(CAM, float(r.t))
    s = j["w"] / D.info(CAM)["w"]
    src_small = cv2.resize(src, (j["w"], j["h"]), interpolation=cv2.INTER_AREA)
    a = frame[y1:y2, x1:x2].astype(np.float32).ravel()
    want = src_small[y1:y2, x1:x2].astype(np.float32).ravel()
    bg = cv2.resize(D.background(CAM), (j["w"], j["h"]))[y1:y2, x1:x2].astype(np.float32).ravel()
    corr_src = np.corrcoef(a, want)[0, 1]
    assert corr_src > 0.6, corr_src
    assert np.abs(a - want).mean() < np.abs(a - bg).mean() + 1.0  # closer to the source object than to the plate
    assert abs(r.x1 * s - x1) <= 2 and abs(r.y2 * s - y2) <= 2  # box is the track's box, scaled


def test_schedule_avoids_overlap():
    # two identical 2 s ghosts in the same place: the second must start after the first ends
    boxes = [(k * S.STEP, (2, 2, 6, 6)) for k in range(8)]
    starts = S.schedule([boxes, boxes], gw=20, gh=20, max_out=10.0)
    assert starts[0] == 0.0 and starts[1] >= 2.0
    # a ghost elsewhere can share the start time
    other = [(k * S.STEP, (12, 12, 16, 16)) for k in range(8)]
    assert S.schedule([boxes, other], gw=20, gh=20, max_out=10.0) == [0.0, 0.0]


def test_ptz_output_if_built():
    p = D.OUT / "ptz_parking" / "synopsis.json"
    if not p.exists():
        pytest.skip("run python -m stm.synopsis ptz_parking first")
    j = json.loads(p.read_text(encoding="utf-8"))
    parked = [g for g in j["ghosts"] if g.get("parked")]
    moving = [g for g in j["ghosts"] if not g.get("parked")]
    assert j["out_duration"] <= 60.0 and moving and parked
    for g in parked:  # one static box: never moves
        assert len({tuple(b[1:]) for b in g["boxes"]}) == 1
    days = {g["day"] for g in j["ghosts"]}
    assert days <= {"A", "B"}
    # one camera view per render: the Day A home view only, on its own plate, no parked labels
    assert j["view"] == "A" and j["view_label"] == "Day A view" and days == {"A"}
    w0, w1 = j["src_window"]
    assert w0 == 0.0 and 150.0 <= w1 <= 156.0
    assert all(w0 - 1e-6 <= g["src_start"] < g["src_end"] <= w1 + 1e-6 for g in j["ghosts"])
    assert j["skipped"]["other_view_tracks"] > 0
    assert [s["plate"] for s in j["segments"]] == ["background_A.jpg"]
    assert j["out_duration"] <= 30.0 + 1e-6
    assert all("parked_label" not in g and isinstance(g["parked"], bool) for g in j["ghosts"])
    # a parked car is one ghost: no two parked ghosts share a spot at the same output time
    for i, a in enumerate(parked):
        for b in parked[i + 1:]:
            if S._iou(a["boxes"][0][1:], b["boxes"][0][1:]) >= 0.4:
                assert a["out_end"] <= b["out_start"] + S.STEP or b["out_end"] <= a["out_start"] + S.STEP


def test_mask_is_feathered_no_rectangle_edge():
    crop = np.full((40, 30, 3), 220, np.uint8)
    plate = np.zeros((40, 30, 3), np.uint8)
    a = S._alpha(crop, plate)[..., 0]
    assert a.max() <= S.ALPHA + 1e-6 and a[20, 15] > 0.8 * S.ALPHA
    assert max(a[0].max(), a[-1].max(), a[:, 0].max(), a[:, -1].max()) < 0.25 * S.ALPHA
    # a crop identical to the plate pastes nothing: no faint rectangle
    assert S._alpha(plate, plate).max() == 0.0


def test_labels_sit_above_ghost_and_skip_overlaps():
    r1 = S.label_rect("01:23", (100, 100, 140, 180), 960, 540)
    assert r1[3] <= 100 and r1[0] >= 0 and r1[1] >= 0
    assert 8 <= r1[3] - r1[1] <= 16  # ~11 px text
    r2 = S.label_rect("01:24", (104, 102, 144, 182), 960, 540)
    assert S._overlaps(r2, [r1])
    r3 = S.label_rect("01:25", (400, 100, 440, 180), 960, 540)
    assert not S._overlaps(r3, [r1])
    top = S.label_rect("00:05", (10, 0, 50, 40), 960, 540)  # no room above: goes inside, still on screen
    assert top[1] >= 0


def test_view_windows_split_at_long_turn_away():
    blind = [{"start": 68, "end": 70, "reason": "no_signal"}, {"start": 156, "end": 157, "reason": "blocked"},
             {"start": 157, "end": 159, "reason": "blur"}, {"start": 159, "end": 237, "reason": "blocked"},
             {"start": 237, "end": 251, "reason": "blur"}, {"start": 290, "end": 295, "reason": "blocked"}]
    assert S.view_windows(blind, 316.13) == [(0.0, 155.0), (252.0, 316.13)]
    assert S.view_windows([], 40.0) == [(0.0, 40.0)]


def _row(i, day, t0, t1, box, name="car", parked=False, travel=2.0):
    x1, y1, x2, y2 = box
    return {"id": i, "name": name, "day": day, "first_t": t0, "last_t": t1, "n": 50, "conf": 0.8,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "travel_px": travel, "parked": parked}


def test_two_real_days_one_ghost_per_parked_car(monkeypatch):
    """Pack D shape: fixed camera, two real days with date ids (never A/B), a parked car fragmented by the
    tracker all day, a second car later in the same spot, movers and stationary fragments, and a cap."""
    d1, d2 = "20260901", "20260902"
    spot = (100, 100, 200, 160)
    rows = [_row(1, d1, 0, 300, spot, parked=True), _row(2, d1, 305, 600, spot, parked=True),
            _row(3, d1, 610, 900, (102, 101, 201, 161), parked=False),          # ID-switch fragment, still
            _row(4, d2, 960, 1800, spot, parked=True),                          # day 2: its own ghost
            _row(5, d1, 50, 58, (400, 300, 480, 360), travel=900.0),            # movers
            _row(6, d1, 70, 75, (500, 300, 520, 360), name="person", travel=200.0),
            _row(7, d2, 1000, 1010, (400, 300, 480, 360), travel=800.0),
            _row(8, d2, 1100, 1120, (600, 100, 700, 160), travel=3.0),         # stationary, no parked car
            _row(9, d2, 1200, 1200.4, (10, 10, 30, 30), travel=50.0)]          # too short
    monkeypatch.setattr(S.D, "objects", lambda cam: pd.DataFrame(rows))
    blocks = [{"day": d1, "label": "1 Sep", "src_start": 0.0, "src_end": 930.0, "plate": f"background_{d1}.jpg"},
              {"day": d2, "label": "2 Sep", "src_start": 930.0, "src_end": 1860.0, "plate": f"background_{d2}.jpg"}]
    moving, parked, drops = S.select("packd_fake", blocks, cap=3)
    assert len(parked) == 2
    p1 = next(p for p in parked if p["day"] == d1)
    assert p1["object_ids"] == [1, 2, 3] and p1["first_t"] == 0 and p1["last_t"] == 900
    assert next(p for p in parked if p["day"] == d2)["object_ids"] == [4]
    assert drops["short"] == 1 and drops["cap"] == 1 and drops["other_view"] == 0
    assert sorted(moving.id) == [5, 6, 7]  # the cap drops the stationary fragment first, never a mover
    assert set(moving.day) | {p["day"] for p in parked} == {d1, d2}
    # output budget: a 31 min two-day clip gets the full 60 s, split between the days, each a real share
    total = S.budget_for("packd_fake", 1860.0, [30.0] * 200)
    assert total == S.MAX_OUT
    lens = S.split_budget(total, [400.0, 1200.0])
    assert sum(lens) <= total + 1e-6 and min(lens) >= 0.25 * total


def test_head_blur_only_touches_the_head():
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 255, (120, 40, 3), dtype=np.uint8)
    out = S.blur_head(crop)
    hh = int(np.ceil(120 * S.HEAD_FRAC))
    assert out.shape == crop.shape and out is not crop
    assert out[:hh].astype(float).std() < 0.35 * crop[:hh].astype(float).std()
    assert np.array_equal(out[hh:], crop[hh:])
