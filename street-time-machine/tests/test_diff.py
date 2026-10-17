"""stm.diff: unit checks on real ptz_parking data + contract checks on out/ptz_parking/diff.json.

The full build takes 1-2 minutes, so the JSON checks read the output of `python -m stm.diff`.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from stm import diff as D
from stm.data import OUT, info

CAM = "ptz_parking"
KINDS = ("same", "arrived", "left", "moved", "swapped")


@pytest.fixture(scope="module")
def res() -> dict:
    p = OUT / CAM / "diff.json"
    if not p.exists():
        pytest.skip("run `python -m stm.diff` first")
    return json.loads(p.read_text(encoding="utf-8"))


def test_geometry_helpers():
    assert D.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert D.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(D.iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert D.map_box(np.eye(3), [1, 2, 3, 4]) == pytest.approx([1, 2, 3, 4])
    valid = np.zeros((100, 100), bool)
    valid[:, :50] = True
    assert D.clip_to(valid, [30, 10, 80, 60]) == [30, 10, 50, 60]
    assert D.clip_to(valid, [60, 10, 90, 60]) is None
    slanted = np.tri(100, 100, 0, bool)[::-1]  # valid above the anti-diagonal
    b = D.inner(slanted, [0, 0, 60, 60])
    assert b is not None and slanted[int(b[1]):int(b[3]), int(b[0]):int(b[2])].mean() >= 0.99
    assert D.touches_edge([0, 5, 10, 10], 100, 100) and not D.touches_edge([5, 5, 10, 10], 100, 100)


def test_similarity_is_shift_tolerant_and_discriminates():
    bg = cv2.imread(str(OUT / CAM / "background_A.jpg"))
    if bg is None:
        pytest.skip("run `python -m stm.diff` first")
    white, dark = [234, 372, 286, 420], [341, 455, 433, 530]  # white car (id 25) vs dark car (id 12), day A
    pa = D._patch(bg, white)
    assert D.similarity(pa, pa) == pytest.approx(1.0, abs=1e-3)
    shifted = D._patch(bg, [white[0] + 3, white[1] + 2, white[2] + 3, white[3] + 2])
    assert D.similarity(pa, shifted) >= D.SIM_SAME
    assert D.similarity(pa, D._patch(bg, dark)) < D.SIM_SAME
    assert D.similarity(None, pa) is None


def test_backgrounds_align_with_homography():
    a = cv2.imread(str(OUT / CAM / "background_A.jpg"))
    b = cv2.imread(str(OUT / CAM / "background_B.jpg"))
    if a is None or b is None:
        pytest.skip("run `python -m stm.diff` first")
    i = info(CAM)
    assert a.shape == b.shape == (i["h"], i["w"], 3)
    hm = D.homography(b, a)
    assert hm["H"] is not None and hm["inliers"] >= 40 and hm["reproj_px"] <= 3.0
    # the camera really did re-frame between the halves: raw pixel boxes are NOT comparable
    assert D.corner_shift(hm["H"], i["w"], i["h"]) > 100


def test_single_day_camera_is_skipped():
    assert D.build("car_detection") is None


def test_contract_shape(res):
    assert res["camera"] == CAM and res["day_a"] == "A" and res["day_b"] == "B"
    assert set(res["counts"]) == set(KINDS)
    assert res["background_a"] == "background_A.jpg" and res["background_b"] == "background_B.jpg"
    for k in ("background_a", "background_b", "background_b_aligned"):
        assert (OUT / CAM / res[k]).exists()
    assert {c["kind"] for c in res["changes"]} <= set(KINDS)
    for k in KINDS:
        assert res["counts"][k] == sum(c["kind"] == k for c in res["changes"])
    for c in res["changes"]:
        assert set(c) >= {"kind", "box", "a", "b", "similarity", "note"}
        assert len(c["box"]) == 4 and c["box"][0] < c["box"][2] and c["box"][1] < c["box"][3]
        assert isinstance(c["note"], str) and c["note"]
        assert (c["a"] is None) == (c["kind"] == "arrived")
        assert (c["b"] is None) == (c["kind"] == "left")
        assert c["similarity"] is None or -1.0 <= c["similarity"] <= 1.0


def test_claims_are_true_to_the_data(res):
    al = res["alignment"]
    assert al["ok"] and al["inliers"] >= 40 and 0 < al["overlap_frac"] <= 1
    va, vb = al["view_a"], al["view_b"]
    i = info(CAM)
    for c in res["changes"]:
        # crops exist, are real images and come from inside that day's stable view
        for side, v in (("a", va), ("b", vb)):
            ref = c[side]
            if ref is None:
                continue
            assert v["start"] <= ref["t"] <= v["end"]
            img = cv2.imread(str(OUT / CAM / ref["crop"]))
            assert img is not None and img.shape[0] > 8 and img.shape[1] > 8 and img.std() > 3
        # every claim sits inside day A's frame
        x1, y1, x2, y2 = c["box"]
        assert x1 >= -1 and y1 >= -1 and x2 <= i["w"] + 1 and y2 <= i["h"] + 1
        if c["kind"] in ("same", "moved"):
            assert c["similarity"] >= D.SIM_SAME
        if c["kind"] in ("left", "arrived", "swapped"):
            assert c["similarity"] is None or c["similarity"] < D.SIM_SAME
    # spots one day can't see are never reported as left/arrived
    for n in res["not_compared"]:
        assert n["reason"]
    ids_nc = {(n["day"], n["object_id"]) for n in res["not_compared"]}
    ids_ch = {("A", c["a"]["object_id"]) for c in res["changes"] if c["a"]} | \
             {("B", c["b"]["object_id"]) for c in res["changes"] if c["b"]}
    assert not ids_nc & ids_ch
    assert len(ids_ch & {k for k in ids_ch if k[0] == "A"}) + len([n for n in res["not_compared"] if n["day"] == "A"]) \
        == res["parked"]["A"]
    # a skipped A/B pair (same spot, too partial to judge) carries its day-B vehicle as b_object_id
    assert len({k for k in ids_ch if k[0] == "B"}) + len([n for n in res["not_compared"] if n["day"] == "B"]) \
        + len([n for n in res["not_compared"] if "b_object_id" in n]) == res["parked"]["B"]


def test_summary_written(res):
    s = (OUT / CAM / "diff_summary.txt").read_text(encoding="utf-8")
    for k, v in res["counts"].items():
        assert f"{k} {v}" in s
