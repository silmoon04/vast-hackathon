"""Tests for stm.flow: synthetic checks of the geometry helpers plus real-data checks against stm.data."""
import json
import math

import cv2
import numpy as np
import pytest

from stm import data as D
from stm import flow as F

SMALL = "person_bicycle_car_detection"  # small real clip: computes in a few seconds


def has_tracks(cam: str) -> bool:
    try:
        D.tracks(cam)
        return D.CAMS[cam]["video"].exists()
    except FileNotFoundError:
        return False


# ----------------------------------------------------------------------------- synthetic
def test_resample_is_even_and_keeps_ends():
    r = F.resample(np.array([[0, 0], [10, 0], [10, 30]], float), 20)
    assert r.shape == (20, 2)
    assert np.allclose(r[0], [0, 0]) and np.allclose(r[-1], [10, 30])
    # unevenly spaced input on a straight line comes out evenly spaced
    r = F.resample(np.array([[0, 0], [1, 0], [2, 0], [40, 0]], float), 20)
    assert np.allclose(np.diff(r[:, 0]), 40 / 19) and np.allclose(r[:, 1], 0)
    assert F.resample(np.array([[5.0, 6.0]]), 20).shape == (20, 2)


def test_lane_names_follow_direction_and_place():
    W, H = 1000, 600
    assert F.lane_name(np.array([[50, 550], [950, 560]]), W, H) == "left to right along the bottom"
    assert F.lane_name(np.array([[950, 40], [50, 40]]), W, H) == "right to left along the top"
    assert F.lane_name(np.array([[500, 50], [500, 550]]), W, H) == "top to bottom through the centre"
    assert F.lane_name(np.array([[100, 100], [600, 500]]), W, H) == "top left to bottom right"
    # direction is judged in pixels: 45 px right and 14 px down is a rightward walk, not a diagonal
    assert F.lane_name(np.array([[1119.5, 422.5], [1164.0, 436.7]]), 1280, 720).startswith("left to right")


def test_dedupe_frame_merges_one_object_held_by_two_ids():
    import pandas as pd
    rows = pd.DataFrame([
        {"tid": 79, "name": "person", "conf": 0.54, "x1": 574, "y1": 447, "x2": 618, "y2": 527, "t": 18.33},
        {"tid": 134, "name": "person", "conf": 0.46, "x1": 587, "y1": 449, "x2": 624, "y2": 526, "t": 18.33},
        {"tid": 5, "name": "bicycle", "conf": 0.6, "x1": 574, "y1": 470, "x2": 618, "y2": 530, "t": 18.33},
        {"tid": 133, "name": "car", "conf": 0.57, "x1": 1199, "y1": 460, "x2": 1268, "y2": 526, "t": 18.33}])
    kept = F.dedupe_frame(rows)
    assert sorted(kept.tid) == [5, 79, 133]  # the rider's bicycle is a different group, so it stays


def _fake_track(tid, a, z, group="vehicle"):
    pts = np.array([[i * 0.25, a[0] + (z[0] - a[0]) * i / 9, a[1] + (z[1] - a[1]) * i / 9] for i in range(10)])
    return {"tid": tid, "group": group, "entry": np.array(a, float), "exit": np.array(z, float), "pts": pts,
            "speed": 100.0}


def test_clustering_splits_opposite_directions_and_merges_parallel_ones():
    W, H = 1000, 600
    rng = np.random.default_rng(0)
    tr = [_fake_track(i, (50 + rng.normal(0, 10), 300 + rng.normal(0, 10)), (950 + rng.normal(0, 10), 300)) for i in range(6)]
    tr += [_fake_track(100 + i, (950, 310 + rng.normal(0, 10)), (50, 300 + rng.normal(0, 10))) for i in range(4)]
    cl = F.cluster_tracks(tr, W, H)
    assert [len(c) for c in cl] == [6, 4]
    assert {t["tid"] for t in cl[1]} == {100, 101, 102, 103}


def test_view_score_tells_moved_camera_from_passing_object():
    rng = np.random.default_rng(1)
    ref = cv2.GaussianBlur(rng.uniform(0, 255, (F.SH, F.SW)).astype(np.float32), (0, 0), 2.0)
    m = np.ones_like(ref, bool)
    assert F.view_score(ref, ref, m) == 1.0
    # brightness/contrast change: same view
    assert F.view_score(ref * 0.6 + 20, ref, m) == 1.0
    # a big object covering a third of the picture, masked out as a detection: same view
    obj = ref.copy()
    obj[20:70, 40:100] = 255
    m2 = m.copy()
    m2[18:72, 38:102] = False
    assert F.view_score(obj, ref, m2) >= F.VIEW_THR
    # whole picture shifted by 12 thumbnail px (a pan): different view
    moved = np.roll(ref, 12, axis=1)
    assert F.view_score(moved, ref, m) < F.VIEW_THR
    # almost everything masked: cannot judge
    assert math.isnan(F.view_score(ref, ref, np.zeros_like(m)))


# ----------------------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def small():
    if not has_tracks(SMALL):
        pytest.skip("no cached tracks/video")
    return F.compute(SMALL)


def test_compute_matches_contract(small):
    js = small["json"]
    i = D.info(SMALL)
    assert js["camera"] == SMALL and js["image"] == "flow.jpg" and (js["w"], js["h"]) == (i["w"], i["h"])
    assert js["lanes"], "this clip has people and cars crossing"
    ids = [l["lane_id"] for l in js["lanes"]]
    assert ids == [f"L{k + 1}" for k in range(len(ids))]
    counts = [l["count"] for l in js["lanes"]]
    assert counts == sorted(counts, reverse=True)
    moving = {t["tid"]: t for t in small["tt"]["moving"]}
    for l in js["lanes"]:
        assert set(l) >= {"lane_id", "name", "group", "count", "from", "to", "path", "median_speed_px_s", "example_ids"}
        assert l["group"] in ("person", "vehicle") and l["count"] >= 1
        assert len(l["path"]) == F.PATH_POINTS and l["path"][0] == l["from"] and l["path"][-1] == l["to"]
        assert all(-1 <= x <= js["w"] + 1 and -1 <= y <= js["h"] + 1 for x, y in l["path"])
        assert l["example_ids"] and all(e in moving and moving[e]["group"] == l["group"] for e in l["example_ids"])
        assert l["median_speed_px_s"] > 0
    assert sum(counts) + sum(js["totals"]["not_in_main_lanes"].values()) == js["totals"]["moving"]


def test_per_minute_counts_recomputed_from_tracks(small):
    js, tt = small["json"], small["tt"]
    moving = {t["tid"]: t["group"] for t in tt["moving"]}
    df = D.tracks(SMALL)
    df = df[df.tid.isin(moving)]
    df = df[F.visible_mask(tt["views"], df.t.to_numpy())]
    assert len(js["per_minute"]) == math.ceil(D.info(SMALL)["duration"] / 60)
    for row in js["per_minute"]:
        seg = df[(df.t >= row["t"]) & (df.t < row["t"] + 60)]
        ids = set(int(x) for x in seg.tid.unique())
        assert row["person"] == sum(moving[i] == "person" for i in ids)
        assert row["vehicle"] == sum(moving[i] == "vehicle" for i in ids)
    # every moving track is counted in at least one minute
    assert sum(r["person"] + r["vehicle"] for r in js["per_minute"]) >= js["totals"]["moving"]


def test_busiest_is_the_max_per_frame_count(small):
    js, tt = small["json"], small["tt"]
    moving = {t["tid"] for t in tt["moving"]}
    df = tt["df"][tt["df"].tid.isin(moving)]
    per_f = {f: len(F.dedupe_frame(g)) for f, g in df.groupby("f")}
    best = max(per_f.values())
    first = min(f for f, n in per_f.items() if n == best)
    b = js["busiest"]
    assert b["count"] == best and b["person"] + b["vehicle"] == b["count"] <= b["track_ids"]
    assert b["frame"] == "frames/busiest.jpg"
    assert abs(b["t"] - float(df[df.f == first].t.iloc[0])) < 0.01


def test_moving_excludes_parked_and_short_tracks(small):
    obj = D.objects(SMALL, min_frames=F.MIN_FRAMES)
    parked = set(obj.id[obj.parked]) if len(obj) else set()
    df = D.tracks(SMALL)
    for t in small["tt"]["moving"]:
        assert t["tid"] not in parked
        assert (df.tid == t["tid"]).sum() >= F.MIN_FRAMES
        assert t["net"] >= 0.02 * D.info(SMALL)["w"]


# ----------------------------------------------------------------------------- written outputs
@pytest.mark.parametrize("cam", list(D.CAMS))
def test_written_outputs(cam):
    p = D.OUT / cam / "flow.json"
    if not p.exists():
        pytest.skip("run python -m stm.flow first")
    js = json.loads(p.read_text(encoding="utf-8"))
    img = cv2.imread(str(D.OUT / cam / js["image"]))
    assert img is not None and img.shape[1] >= 1280 and (img.shape[1], img.shape[0]) == (js["image_w"], js["image_h"])
    assert abs(img.shape[1] / img.shape[0] - js["w"] / js["h"]) < 0.01
    if js["busiest"] is not None:
        assert (D.OUT / cam / js["busiest"]["frame"]).exists()
    assert (D.OUT / cam / "flow_summary.txt").exists()
    v = js["view"]
    assert 0 <= v["main_view_s"] <= v["duration_s"] + 0.01
    if not v["excluded"]:
        assert abs(v["main_view_s"] - v["duration_s"]) < 0.02
    assert abs(sum(r.get("visible_s", 0) for r in js["per_minute"]) - v["main_view_s"]) < 0.05
    assert all(r.get("visible_s", 0) <= r["seconds"] + 0.01 for r in js["per_minute"])
    # minutes the main view never shows are UNKNOWN (null), never zero
    for row in js["per_minute"]:
        if row.get("visible_s") == 0:
            assert row["person"] is None and row["vehicle"] is None
        else:
            assert isinstance(row["person"], int) and isinstance(row["vehicle"], int)
    # no lane example is a parked vehicle
    obj = D.objects(cam, min_frames=F.MIN_FRAMES)
    parked = set(int(x) for x in obj.id[obj.parked]) if len(obj) else set()
    for l in js["lanes"]:
        assert not parked & set(l["example_ids"])


def test_ptz_parking_drops_the_other_camera_views():
    """ptz_parking pans/zooms away after ~155 s (checked by eye on sampled frames); those seconds are excluded."""
    p = D.OUT / "ptz_parking" / "flow.json"
    if not p.exists():
        pytest.skip("run python -m stm.flow ptz_parking first")
    js = json.loads(p.read_text(encoding="utf-8"))
    v = js["view"]
    assert v["excluded"] and v["main_view_s"] < v["duration_s"]
    assert any(e["start"] <= 200 <= e["end"] for e in v["excluded"])  # the zoomed-in pole view
    if js["busiest"] is not None:
        assert not any(e["start"] <= js["busiest"]["t"] <= e["end"] for e in v["excluded"])
