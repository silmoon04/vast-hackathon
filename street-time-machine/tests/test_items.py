"""Tests for stm.items on a synthetic clip made here with cv2: a phone appears on a textured table, one person stands
in front of it for ~2 s and leaves (still there), another stands right beside it for 4 s (still in view), a third
stands in front of it and takes it (gone), a fourth walks past after. No network, no real footage."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from stm import items

W, H, FPS, DUR = 640, 360, 10, 40.0
BOX = [0.45, 0.62, 0.06, 0.04]      # the phone, normalised x, y, w, h
PLACED = 3.0                        # phone appears
COVER2 = 24.73                      # taker's body first overlaps the phone: the last moment it was visible
TAKEN = 26.0                        # phone lifted while hidden


def _background(rng: np.random.Generator) -> np.ndarray:
    low = rng.integers(40, 200, (24, 40, 3), dtype=np.uint8)
    bg = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.int16)
    tx0, tx1, ty0, ty1 = int(0.25 * W), int(0.75 * W), int(0.55 * H), int(0.80 * H)
    wood = np.zeros((ty1 - ty0, tx1 - tx0, 3), np.int16) + np.array([40, 85, 140], np.int16)
    wood += ((np.arange(ty1 - ty0) // 6) % 2 * 18)[:, None, None].astype(np.int16)
    bg[ty0:ty1, tx0:tx1] = wood
    bg += rng.integers(-10, 11, (H, W, 3), dtype=np.int16)
    return bg


def _person_x(t: float, arrive: float, leave: float, exit_dir: int, stop: float = 307.0, v: float = 200.0) -> float | None:
    """Centre x of a person walking in from the left, standing at `stop` from arrive to leave, walking off."""
    start = arrive - (stop + 40) / v
    if t < start:
        return None
    if t < arrive:
        return stop - (arrive - t) * v
    if t <= leave:
        return stop
    x = stop + exit_dir * (t - leave) * v
    return x if -40 <= x <= W + 40 else None


def _draw_person(img: np.ndarray, cx: float, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    x0, x1 = int(cx - 35), int(cx + 35)
    y0, ym, y1 = 120, 220, 330
    img[y0:ym, max(0, x0):max(0, x1)] = top
    img[ym:y1, max(0, x0):max(0, x1)] = bottom


def _make_clip(path, take: bool) -> str:
    rng = np.random.default_rng(7)
    bg = _background(rng)
    px0, py0 = int(BOX[0] * W), int(BOX[1] * H)
    px1, py1 = int((BOX[0] + BOX[2]) * W), int((BOX[1] + BOX[3]) * H)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not w.isOpened():
        path = path.with_suffix(".avi")
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for k in range(int(DUR * FPS)):
        t = k / FPS
        img = bg + int(15 * t / DUR)                       # slow exposure drift
        img = np.clip(img, 0, 255).astype(np.uint8)
        if t >= PLACED and not (take and t >= TAKEN):
            img[py0 + 1:py1 - 1, px0 + 1:px1 - 1] = (30, 30, 35)
            img[py0 + 3:py1 - 3, px0 + 4:px1 - 4] = (170, 150, 90)
        for arrive, leave, exit_dir, stop, top, bottom in ((12.0, 14.0, 1, 307.0, (40, 40, 200), (25, 25, 25)),
                                                            (17.0, 21.0, -1, 248.0, (40, 200, 200), (90, 60, 30)),
                                                            (25.0, 26.5, -1, 307.0, (200, 90, 40), (120, 120, 120)),
                                                            (32.0, 32.0, 1, 307.0, (60, 170, 60), (40, 40, 90))):
            cx = _person_x(t, arrive, leave, exit_dir, stop)
            if cx is not None:
                _draw_person(img, cx, top, bottom)
        w.write(img)
    w.release()
    return str(path)


@pytest.fixture(scope="module")
def taken(tmp_path_factory):
    return _make_clip(tmp_path_factory.mktemp("items") / "taken.mp4", take=True)


@pytest.fixture(scope="module")
def stays(tmp_path_factory):
    return _make_clip(tmp_path_factory.mktemp("items") / "stays.mp4", take=False)


def test_taken_t0_t1_and_occlusion_not_gone(taken):
    r = items.item_span(taken, BOX)
    assert set(r) >= {"t0", "t1", "evidence"}
    assert r["t0"] is not None and abs(r["t0"] - PLACED) <= 1.0
    assert r["t1"] is not None and abs(r["t1"] - COVER2) <= 1.5, r
    assert r["gone_t"] is not None and TAKEN <= r["gone_t"] <= TAKEN + 2.5
    assert any(a <= 13.0 <= b for a, b in r["occluded"]), r["occluded"]
    assert "gone from" in r["evidence"] and "last visible" in r["evidence"]


def test_frames_occlusion_is_not_gone(taken):
    fr = {row["t"]: row for row in items.item_frames(taken, BOX)}
    assert all(fr[s]["state"] != "gone" for s in range(4, 24)), [(s, fr[s]["state"]) for s in range(4, 24)]
    assert all(fr[s]["state"] in ("occluded", "present") for s in (12, 13))
    assert any(fr[s]["state"] == "occluded" for s in (12, 13))
    assert all(fr[s]["state"] == "present" for s in range(15, 24))   # incl. 17-21 s, someone right beside it
    rows = items.item_frames(taken, BOX, per_second=False)
    assert max(r["ring"] for r in rows if 17.5 <= r["t"] <= 20.5) >= items.RING_CLEAR   # the ring did see them
    assert all(fr[s]["state"] == "gone" for s in range(28, 31))
    assert all(fr[s]["state"] in ("gone", "occluded") for s in range(28, 40))
    assert all(fr[s]["state"] == "gone" for s in range(0, 3))      # before it was put down: the empty look


def test_never_taken_has_no_t1(stays):
    r = items.item_span(stays, BOX)
    assert r["t0"] is not None and abs(r["t0"] - PLACED) <= 1.0
    assert r["t1"] is None and r["gone_t"] is None
    assert "still there at the end" in r["evidence"]
    assert any(a <= 13.0 <= b for a, b in r["occluded"]) and any(a <= 25.5 <= b for a, b in r["occluded"])


def test_ref_t_and_offset(taken):
    auto = items.item_span(taken, BOX)
    ref = items.item_span(taken, BOX, ref_t=8.0)
    assert ref["t1"] == auto["t1"] and ref["t0"] == auto["t0"]
    assert "ref_t" in ref["evidence"]
    shifted = items.item_span(taken, BOX, offset_s=10.0)
    assert shifted["t1"] == pytest.approx(auto["t1"] - 10.0, abs=0.01)
    assert shifted["t0"] == pytest.approx(auto["t0"] - 10.0, abs=0.01)


def test_fps_one_still_finds_it(taken):
    r = items.item_span(taken, BOX, fps=1)
    assert r["t1"] is not None and abs(r["t1"] - COVER2) <= 1.5


def test_bad_input(taken, tmp_path):
    with pytest.raises(FileNotFoundError):
        items.item_span(str(tmp_path / "missing.mp4"), BOX)
    with pytest.raises(ValueError):
        items.item_span(taken, [0.4, 0.4, 0.0, 0.1])
    with pytest.raises(ValueError):
        items.item_span(taken, [1.2, 0.4, 0.1, 0.1])
