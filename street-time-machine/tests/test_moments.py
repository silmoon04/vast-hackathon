"""Tests for stm.moments: helpers on synthetic boxes, candidates on real cached tracks, and the written JSON."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stm import data, moments as M

KINDS = {"arrival", "departure", "crowd", "close_pass", "long_stay", "blind"}


def has_tracks(cam: str) -> bool:
    return any(data.TRACK_CACHE.glob(f"{cam}__det__*.json.gz"))


# ---------------------------------------------------------------- helpers
def test_edge_gap_and_iou():
    assert M.edge_gap(0, 0, 10, 10, 5, 5, 20, 20) == 0            # overlap
    assert M.edge_gap(0, 0, 10, 10, 13, 0, 20, 10) == 3           # side by side
    assert M.edge_gap(0, 0, 10, 10, 13, 14, 20, 20) == pytest.approx(5.0)  # diagonal 3-4-5
    df = pd.DataFrame([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], columns=["x1", "y1", "x2", "y2"])
    iou = M.iou_many([0, 0, 10, 10], df)
    assert iou[0] == pytest.approx(1.0) and iou[1] == pytest.approx(50 / 150) and iou[2] == 0


def test_merge_blind_joins_close_spans():
    spans = [{"start": 10, "end": 11, "reason": "blur"}, {"start": 11, "end": 40, "reason": "blocked"},
             {"start": 41.5, "end": 42, "reason": "blur"}, {"start": 60, "end": 65, "reason": "dark"}]
    m = M.merge_blind(spans)
    assert [(b["start"], b["end"]) for b in m] == [(10, 42), (60, 65)]
    assert m[0]["by_reason"] == {"blur": 1.5, "blocked": 29}


def test_select_caps_order_and_diversity():
    c = [{"t": float(i), "kind": "close_pass", "score": 0.9 - i * 0.001, "object_ids": [i], "box": [i * 50, 0, i * 50 + 10, 10]}
         for i in range(30)]
    c.append({"t": 99.0, "kind": "long_stay", "score": 0.1, "object_ids": [999], "box": [0, 0, 1, 1]})
    out = M.select(c)
    assert len(out) <= M.MAX_MOMENTS
    assert [o["t"] for o in out] == sorted(o["t"] for o in out)
    assert any(o["kind"] == "long_stay" for o in out), "best of every kind is kept even with a low score"
    assert sum(o["kind"] == "close_pass" for o in out) <= 2 * M.CAPS["close_pass"]


def test_titles_have_no_colour_words():
    colours = {"white", "black", "grey", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown"}
    for cam in data.CAMS:
        p = data.OUT / cam / "moments.json"
        if p.exists():
            for m in json.loads(p.read_text(encoding="utf-8"))["moments"]:
                assert not colours & set(m["title"].lower().split()), m["title"]


# ---------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def yonge():
    if not has_tracks("yonge_dundas"):
        pytest.skip("no cached tracks for yonge_dundas")
    return data.tracks("yonge_dundas"), data.objects("yonge_dundas")


def test_crowd_peak_matches_frame_counts(yonge):
    df, objs = yonge
    st: dict = {}
    cr = M.crowd_moments(df, objs, st)
    ids = set(objs[objs.name == "person"].id)
    raw = df[df.tid.isin(ids) & (df.name == "person")].groupby("f").tid.nunique()
    assert cr, "yonge_dundas has a crowd"
    assert st["crowd_peak"] <= int(raw.max())
    top = max(cr, key=lambda m: m["score"])
    assert top["title"] == f"{st['crowd_peak']} people at once"
    # the thumbnail frame really shows that many tracked people
    assert len(top["object_ids"]) == st["crowd_peak"]


def test_long_stay_is_longest_person(yonge):
    df, objs = yonge
    ls = M.long_stay_moments(df, objs, {})
    ppl = objs[objs.name == "person"]
    longest = ppl.loc[(ppl.last_t - ppl.first_t).idxmax()]
    assert len(ls) == 1 and ls[0]["object_ids"] == [int(longest.id)]
    assert ls[0]["end"] - ls[0]["t"] == pytest.approx(longest.last_t - longest.first_t, abs=0.02)


def test_close_pass_rule_holds(yonge):
    df, objs = yonge
    cps = M.close_pass_moments(df, objs, {})
    assert cps
    moving = set(objs[objs.name.isin(data.VEHICLES) & ~objs.parked].id)
    for c in cps:
        pid, vid = c["object_ids"]
        assert vid in moving and pid in set(objs[objs.name == "person"].id)
        rows = []
        for tid in (pid, vid):
            g = df[df.tid == tid]
            rows.append(g.iloc[(g.t - c["t"]).abs().argmin()])
        p, v = rows
        assert abs(p.t - c["t"]) < 0.01 and p.f == v.f, "both boxes come from the same frame"
        gap = M.edge_gap(p.x1, p.y1, p.x2, p.y2, v.x1, v.y1, v.x2, v.y2)
        assert gap < M.CLOSE_REL * v.bh


def test_spot_change_sees_a_car_leave(monkeypatch):
    """Synthetic frames: a dark car on grey asphalt that stays (no change) or leaves (big change)."""
    empty = np.full((100, 200, 3), 128, np.uint8)
    car = empty.copy()
    car[40:80, 50:150] = 20
    frames = {1.0: car, 2.0: car, 3.0: empty}

    def fake_frame_at(cam, t):  # same contract as data.frame_at: ValueError when there is no frame
        if round(t, 2) not in frames:
            raise ValueError(f"no frame at {t}")
        return frames[round(t, 2)]

    monkeypatch.setattr(M.data, "frame_at", fake_frame_at)
    box = [50, 40, 150, 80]
    assert M.spot_change("x", box, 1.0, [2.0], {}) < M.SPOT_CHANGE_MIN          # still there
    assert M.spot_change("x", box, 1.0, [3.0], {}) >= M.SPOT_CHANGE_MIN         # gone
    assert M.spot_change("x", box, 1.0, [3.0, 2.0], {}) < M.SPOT_CHANGE_MIN     # gone in one frame only
    assert M.spot_change("x", box, 9.0, [3.0], {}) is None                      # unreadable frame


def test_parked_dropout_is_not_a_departure():
    """ptz_parking car #28 loses its detections at ~24.9 s but never leaves (seen when checking frames)."""
    cam = "ptz_parking"
    if not has_tracks(cam):
        pytest.skip("no cached tracks")
    objs = data.objects(cam).set_index("id")
    if 28 not in objs.index or not objs.loc[28].parked:
        pytest.skip("track cache changed")
    o = objs.loc[28]
    box = [o.x1, o.y1, o.x2, o.y2]
    assert M.spot_change(cam, box, o.last_t - 0.5, [o.last_t + 3.0], {}) < M.SPOT_CHANGE_MIN


def test_written_parked_moments_hold():
    cam = "ptz_parking"
    p = data.OUT / cam / "moments.json"
    if not has_tracks(cam) or not p.exists():
        pytest.skip("no cached tracks / moments.json")
    df, objs = data.tracks(cam), data.objects(cam)
    blind = M.ledger_blind(cam) or []
    parked = set(objs[objs.parked].id)
    veh = df[df.name.isin(data.VEHICLES)]
    res = [m for m in json.loads(p.read_text(encoding="utf-8"))["moments"] if m["kind"] in ("arrival", "departure")]
    for m in res:
        assert "mean grey change" in m["detail"]
        (oid,) = m["object_ids"]
        assert oid in parked
        o = objs.set_index("id").loc[oid]
        te = o.first_t if m["kind"] == "arrival" else o.last_t
        assert m["t"] == pytest.approx(te, abs=0.01)
        assert not any(b["start"] - M.BLIND_PAD <= te <= b["end"] + M.BLIND_PAD for b in blind)
        lo, hi = (te - M.EMPTY_WIN, te - 0.5) if m["kind"] == "arrival" else (te + 0.5, te + M.EMPTY_WIN)
        w = veh[(veh.t >= lo) & (veh.t <= hi) & (veh.tid != oid)]
        assert int((M.iou_many([o.x1, o.y1, o.x2, o.y2], w) >= 0.4).sum()) < 5


def test_build_small_cam_schema():
    cam = "person_bicycle_car_detection"
    if not has_tracks(cam):
        pytest.skip("no cached tracks")
    res, st = M.build(cam, thumbs=False)
    assert res["camera"] == cam and len(res["moments"]) <= 12
    assert ("blind" in st["candidates"]) and (st["candidates"]["blind"] == 0 or st["ledger"])


# ---------------------------------------------------------------- written outputs (from the CLI run)
@pytest.mark.parametrize("cam", list(data.CAMS))
def test_written_moments_json(cam):
    p = data.OUT / cam / "moments.json"
    if not p.exists():
        pytest.skip(f"run python -m stm.moments {cam} first")
    d = json.loads(p.read_text(encoding="utf-8"))
    ms = d["moments"]
    assert d["camera"] == cam and len(ms) <= 12
    assert [m["t"] for m in ms] == sorted(m["t"] for m in ms)
    dur = data.info(cam)["duration"]
    has_ledger = (data.OUT / cam / "ledger.json").exists()
    for m in ms:
        assert set(m) == {"t", "end", "kind", "title", "detail", "object_ids", "thumb", "box", "score"}
        assert m["kind"] in KINDS and m["title"] and m["detail"]
        assert 0 <= m["t"] <= m["end"] <= dur + 0.5
        assert m["box"] is None or (len(m["box"]) == 4 and m["box"][0] < m["box"][2] and m["box"][1] < m["box"][3])
        if m["kind"] == "blind":
            assert has_ledger
        if m["thumb"]:
            assert m["thumb"].startswith("moments/") and (data.OUT / cam / m["thumb"]).exists()
            import cv2
            img = cv2.imread(str(data.OUT / cam / m["thumb"]))
            assert img is not None and img.shape[1] == M.THUMB_W
