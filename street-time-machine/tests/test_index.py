"""Tests for stm.index: colour namer on synthetic patches, people-by-appearance refusal, search parsing, and a real
build on car_detection. Built cams on disk (any cam, any day ids) are checked for person colours."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from stm import data, index

SMALL = "car_detection"          # 5 objects, builds in a few seconds


def patch(bgr: tuple[int, int, int], h: int = 60, w: int = 100, noise: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = bgr
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def car(body_bgr: tuple[int, int, int]) -> np.ndarray:
    """A crude car: painted body with a dark window band on top and dark tyres at the bottom."""
    img = patch(body_bgr)
    img[:18, 15:85] = (30, 30, 32)       # windscreen
    img[50:, :25] = img[50:, 75:] = (20, 20, 20)
    return img


@pytest.mark.parametrize("bgr,want", [
    ((245, 245, 245), "white"), ((20, 20, 20), "black"), ((110, 110, 110), "grey"), ((155, 155, 155), "silver"),
    ((30, 30, 200), "red"), ((200, 60, 30), "blue"), ((40, 170, 40), "green"), ((30, 220, 230), "yellow"),
    ((20, 120, 240), "orange"), ((30, 60, 100), "brown"), ((60, 30, 120), "red"),
])
def test_colour_of_plain_patches(bgr, want):
    c = index.colour_of(patch(bgr), "car")
    assert c["colour"] == want, c
    assert len(c["rgb"]) == 3 and all(0 <= v <= 255 for v in c["rgb"])
    assert 0 < c["frac"] <= 1


@pytest.mark.parametrize("bgr,want", [((235, 235, 235), "white"), ((30, 30, 190), "red"), ((15, 15, 15), "black")])
def test_colour_of_ignores_windows_and_tyres(bgr, want):
    assert index.colour_of(car(bgr), "car")["colour"] == want


def test_achromatic_swatch_matches_name():
    # a white roof over a dark windscreen/shadow half: named by the bright surface, so the swatch must be light too
    img = patch((240, 240, 240))
    img[:30] = patch((50, 50, 50), h=30)
    c = index.colour_of(img, "car")
    assert c["colour"] == "white", c
    assert min(c["rgb"]) >= 172, c


def test_faint_blue_tint_is_not_blue():
    # a black car reflecting the sky: a third of its pixels slightly blue, the rest near black
    img = patch((25, 25, 28))
    img[::3] = patch((120, 80, 60))[::3]
    assert index.colour_of(img, "car")["colour"] != "blue"


@pytest.mark.parametrize("name", ["person", "bicycle", "motorcycle"])
def test_people_and_riders_never_coloured(name):
    img = patch((200, 200, 200), h=200, w=60)       # light legs / background
    img[40:110, 15:45] = patch((30, 30, 200), h=70, w=30)   # a red shirt: must never be reported
    c = index.colour_of(img, name)
    assert c["colour"] is None and c["rgb"] is None and c["reliable"] is False


def test_judge_withholds_unconfident_vehicle_colours():
    red = {"colour": "red", "reliable": True}
    assert index.judge("car", red, 5000, night=False) is None
    assert index.judge("car", red, 5000, night=True) == "night"          # a dark car under street lights
    assert index.judge("car", {"colour": "red", "reliable": False}, 5000, night=False) == "dark box"
    assert index.judge("truck", red, 300, night=False) == "too small"
    assert index.judge("person", red, 5000, night=False) == "person"
    assert index.judge("bicycle", red, 5000, night=False) == "rider"


def test_frame_light_night_vs_day():
    day = patch((120, 120, 120), h=360, w=640)
    sodium_night = patch((10, 25, 45), h=360, w=640)      # dark and orange-tinted by street lights
    infrared = cv2.cvtColor(cv2.cvtColor(patch((90, 100, 110), h=360, w=640, noise=30), cv2.COLOR_BGR2GRAY),
                            cv2.COLOR_GRAY2BGR)            # greyscale IR: not even dark, but colours are meaningless
    asphalt_day = patch((38, 40, 44), h=360, w=640)        # sunny car park, dark tarmac: dark but not night
    assert not index.is_night(index.frame_light(day, (0, 0, 10, 10)))
    assert index.is_night(index.frame_light(sodium_night, (0, 0, 10, 10)))
    assert index.is_night(index.frame_light(infrared, (0, 0, 10, 10)))
    assert not index.is_night(index.frame_light(asphalt_day, (0, 0, 10, 10)))
    pale_day = patch((128, 130, 133), h=360, w=640, noise=3)  # washed-out daytime clip: low but non-zero S
    assert not index.is_night(index.frame_light(pale_day, (0, 0, 10, 10)))
    # a big black lorry filling half a sunny frame is masked out: still day
    day[:, :320] = 10
    assert not index.is_night(index.frame_light(day, (0, 0, 320, 360)))


def test_dark_box_is_marked_unreliable():
    assert index.colour_of(patch((15, 15, 15), noise=2), "car")["reliable"] is False
    assert index.colour_of(patch((200, 200, 200)), "car")["reliable"] is True


def test_parse_synonyms():
    assert index.parse("white van") == {"colours": ["white"], "names": ["car"], "parked": None}
    assert index.parse("people")["names"] == ["person"]
    assert index.parse("pedestrians")["names"] == ["person"]
    assert index.parse("bikes")["names"] == ["bicycle"]
    assert index.parse("gray SUV") == {"colours": ["grey"], "names": ["car"], "parked": None}
    assert index.parse("parked silver cars") == {"colours": ["silver"], "names": ["car"], "parked": True}
    assert index.parse("hello there") == {"colours": [], "names": [], "parked": None}


ROWS = [
    {"object_id": 1, "name": "car", "colour": "white", "parked": True, "n": 50, "first_t": 0.0},
    {"object_id": 2, "name": "car", "colour": "silver", "parked": False, "n": 90, "first_t": 3.0},
    {"object_id": 3, "name": "person", "colour": None, "parked": False, "n": 20, "first_t": 1.0},
    {"object_id": 4, "name": "truck", "colour": "white", "parked": True, "n": 10, "first_t": 2.0},
    {"object_id": 5, "name": "bicycle", "colour": None, "parked": False, "n": 15, "first_t": 4.0},
    {"object_id": 6, "name": "car", "colour": None, "colour_why": "night", "parked": False, "n": 5, "first_t": 9.0},
]


@pytest.mark.parametrize("q", ["man in black", "person in white", "white people", "black people", "red hoodie",
                               "a person in a black shirt", "Find the man in the black shirt", "woman", "kids",
                               "black guy", "anyone with a beard", "pedestrian wearing a cap", "dark-skinned man",
                               "girl in a blue dress", "someone in red"])
def test_refuses_people_by_appearance(q):
    assert index.refusal(q) == index.REFUSAL
    assert index._match_why(ROWS, q) == ([], index.REFUSAL)


@pytest.mark.parametrize("q", ["white car", "people", "pedestrians", "parked silver cars", "dark car", "bikes",
                               "someone near my car", "who came near the white van", "person near the street light"])
def test_allows_vehicle_and_plain_people_queries(q):
    assert index.refusal(q) is None


def test_people_never_matched_by_colour_even_if_data_is_stale():
    stale = ROWS + [{"object_id": 7, "name": "person", "colour": "red", "parked": False, "n": 99, "first_t": 0.5}]
    assert 7 not in index._match(stale, "red")
    assert 7 not in index._match(stale, "orange")                    # NEAR fallback can't reach people either
    assert 7 in index._match(stale, "people")                        # class alone is fine: who was there


def test_match_reasons():
    assert index._match_why(ROWS, "grey car")[1].startswith("no grey found")
    ids, why = index._match_why(ROWS, "red bike")
    assert ids == [] and "rider" in why
    ids, why = index._match_why(ROWS, "blue car")
    assert ids == [] and why.startswith("can't tell: 1 matching vehicle")
    assert index._match_why(ROWS, "white car") == ([1], None)


def test_match_rules():
    assert index._match(ROWS, "white car") == [1]
    assert index._match(ROWS, "cars") == [2, 1, 6]                  # most frames first; uncoloured cars still cars
    assert index._match(ROWS, "white vehicle") == [1, 4]
    assert index._match(ROWS, "people") == [3]
    assert index._match(ROWS, "parked") == [1, 4]
    assert index._match(ROWS, "grey car") == [2]                     # no grey: honest neighbour (silver)
    assert index._match(ROWS, "grey car", near=False) == []
    assert index._match(ROWS, "green bus") == []
    assert index._match(ROWS, "what happened") == []                 # nothing understood: not everyone


@pytest.fixture(scope="module")
def built():
    return index.build(SMALL)


def test_build_real_camera(built):
    objs = data.objects(SMALL)
    rows = built["objects"]
    assert built["camera"] == SMALL
    assert len(rows) == min(index.CAP, len(objs)) > 0
    assert {r["object_id"] for r in rows} <= set(int(i) for i in objs.id)
    assert sum(built["colours"].values()) == built["n_coloured"] == sum(r["colour"] is not None for r in rows)
    for r in rows:
        if r["name"] in index.COLOURED and r["colour_why"] is None:
            assert r["colour"] in index.COLOURS and r["colour_reliable"] is True
        else:
            assert r["colour"] is None and r["rgb"] is None and r["colour_reliable"] is False
        assert r["first_t"] <= r["last_t"]
        assert r["crop"] == f"crops/{r['object_id']}.jpg"
        assert (data.out_dir(SMALL) / r["crop"]).exists()
        o = objs[objs.id == r["object_id"]].iloc[0]
        assert r["name"] == o["name"] and r["n"] == o["n"] and r["parked"] == bool(o["parked"])
        band = {"black": (0, 100), "grey": (100, 130), "silver": (130, 172), "white": (172, 256)}.get(r["colour"])
        if band:                                   # the swatch shows the shade that was named
            assert band[0] - 25 <= max(r["rgb"]) < band[1] + 25, r
    on_disk = json.loads((data.out_dir(SMALL) / "index.json").read_text(encoding="utf-8"))
    assert on_disk["colours"] == built["colours"]
    summary = (data.out_dir(SMALL) / "index_summary.txt").read_text(encoding="utf-8")
    assert f"objects indexed: {len(rows)}" in summary


def test_crop_geometry_matches_save_crop(built):
    """Our sequential-pass crop has the same rectangle as data.crop (what data.save_crop writes)."""
    r = built["objects"][0]
    ref, t_ref = data.crop(SMALL, r["object_id"])
    ours = cv2.imread(str(data.out_dir(SMALL) / r["crop"]))
    assert ours.shape == ref.shape
    assert abs(t_ref - r["crop_t"]) < 0.01
    # same frame (jpeg + seek decode differences only)
    assert float(np.abs(ours.astype(int) - ref.astype(int)).mean()) < 12


def test_search_real_camera(built):
    rows = built["objects"]
    first = next(r for r in rows if r["colour"])
    got = index.search(SMALL, f"{first['colour']} {first['name']}")
    assert index.last_reason is None
    assert first["object_id"] in got
    assert all(r["colour"] == first["colour"] and r["name"] == first["name"]
               for r in rows if r["object_id"] in got)
    cars = index.search(SMALL, "cars")
    assert sorted(cars) == sorted(r["object_id"] for r in rows if r["name"] == "car")
    assert index.search(SMALL, "zebra") == []
    assert index.search(SMALL, "man in a white shirt") == [] and index.last_reason == index.REFUSAL
    assert index.search(SMALL, "cars") and index.last_reason is None


def test_ptz_parking_cap_if_built():
    p = data.OUT / "ptz_parking" / "index.json"
    if not p.exists():
        pytest.skip("run python -m stm.index ptz_parking first")
    idx = json.loads(p.read_text(encoding="utf-8"))
    objs = data.objects("ptz_parking")
    assert len(idx["objects"]) == min(index.CAP, len(objs))
    kept_n = sorted((r["n"] for r in idx["objects"]), reverse=True)
    assert kept_n == sorted(objs.n, reverse=True)[:len(kept_n)]   # the most-seen objects were kept
    assert sum(idx["colours"].values()) == idx["n_coloured"]


def _built_indexes():
    return sorted(p for p in data.OUT.glob("*/index.json"))


@pytest.mark.parametrize("p", _built_indexes(), ids=lambda p: p.parent.name)
def test_no_built_camera_describes_people(p):
    """Every index on disk (dev cams and the real pack, whatever its day ids): no colour or swatch on a person or
    rider, the colour tally counts vehicles only, and the summary never offers a 'person in <colour>' search."""
    idx = json.loads(p.read_text(encoding="utf-8"))
    for r in idx["objects"]:
        if r["name"] not in index.COLOURED:
            assert r.get("colour") is None and r.get("rgb") is None, (p.parent.name, r["object_id"])
    assert sum(idx["colours"].values()) == sum(r.get("colour") is not None for r in idx["objects"])
    s = (p.parent / "index_summary.txt")
    if s.exists():
        assert "person in" not in s.read_text(encoding="utf-8")


def test_night_camera_has_no_vehicle_colours_if_built():
    p = data.OUT / "rochester_night" / "index.json"
    if not p.exists():
        pytest.skip("rochester_night not indexed")
    idx = json.loads(p.read_text(encoding="utf-8"))
    assert idx["colours"] == {} and all(r["colour"] is None for r in idx["objects"])
