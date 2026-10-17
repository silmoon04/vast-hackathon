"""Tests for stm.near on real dev data. Runs in well under 60 s (no crops, no file writes in scans)."""
from __future__ import annotations

import json
import re

import cv2
import numpy as np
import pytest

from stm import data, near

CAM = "ptz_parking"
TARGET_KEYS = {"object_id", "name", "box", "start", "end", "crop", "visitors", "blind", "summary"}
VISITOR_KEYS = {"object_id", "name", "closest_t", "gap_px", "gap_rel", "enter_t", "exit_t", "crop"}


@pytest.fixture(scope="module")
def own():
    """Our own fallback check, kept fast: 60-260 s covers the 68 s cut and the 155-252 s PTZ swing."""
    return near.own_blind(CAM, 60.0, 260.0)


@pytest.fixture(scope="module")
def blind(own):
    """What scan uses: the ledger's spans alone when the ledger module has run, else our own check."""
    if (data.OUT / CAM / "ledger.json").exists():
        return near.blind_spans(CAM)
    return own, "own 60-260 s"


@pytest.fixture(scope="module")
def result(blind):
    mp = pytest.MonkeyPatch()
    mp.setattr(near, "blind_spans", lambda cam: blind)
    try:
        yield near.scan(CAM, crops=False, write=False)
    finally:
        mp.undo()


def test_edge_gap_and_iou():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    g = near.edge_gap(box, np.array([5.0, 13.0, 13.0]), np.array([5.0, 0.0, 14.0]),
                      np.array([20.0, 20.0, 20.0]), np.array([20.0, 10.0, 20.0]))
    assert g.tolist() == [0.0, 3.0, 5.0]          # overlap, side gap, corner gap (3-4-5)
    assert near.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert near.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_merge_spans_joins_overlaps():
    m = near.merge_spans([{"start": 5, "end": 6, "reason": "dark"}, {"start": 0, "end": 2, "reason": "frozen"},
                          {"start": 1.5, "end": 3, "reason": "dark"}])
    assert [(s["start"], s["end"]) for s in m] == [(0, 3), (5, 6)]
    assert m[0]["reason"] == "frozen+dark"


def test_view_change_found_on_ptz(own, blind):
    spans, src = blind
    # the PTZ camera swings away between ~155 s and ~252 s; that must be blind, the stable start must not be
    merged = near.merge_spans(spans)
    away = sum(max(0.0, min(252, s["end"]) - max(158, s["start"])) for s in merged)
    assert away >= 0.5 * (252 - 158), merged
    for sp in (spans, own):
        assert not any(s["start"] < 50 and s["end"] > 1 for s in near.merge_spans(sp)), sp
    # the small re-aim at ~68 s is caught by our own (fallback) check
    assert any(s["reason"] == "view_changed" and s["start"] <= 69 and s["end"] >= 68 for s in own), own
    assert src


def _ledger_dir(tmp_path, monkeypatch, blind):
    (tmp_path / "ledger.json").write_text(json.dumps({"camera": "x", "blind": blind}), encoding="utf-8")
    monkeypatch.setattr(near.data, "out_dir", lambda cam: tmp_path)


def test_ledger_spans_only_when_ledger_exists(tmp_path, monkeypatch):
    """The ledger records camera motion itself (moved / turned_away): no own view_changed spans on top."""
    _ledger_dir(tmp_path, monkeypatch, [{"start": 156.0, "end": 168.0, "reason": "moved", "score": 0.75},
                                        {"start": 168.0, "end": 206.0, "reason": "turned_away", "score": 1.0}])

    def boom(*a, **k):
        raise AssertionError("own check must not run when the ledger exists")
    monkeypatch.setattr(near, "own_blind", boom)
    spans, src = near.blind_spans("x")
    assert src == "ledger.json"
    assert [(s["start"], s["end"], s["reason"]) for s in spans] == [(156.0, 168.0, "moved"),
                                                                     (168.0, 206.0, "turned_away")]


def test_own_check_when_ledger_missing_or_unreadable(tmp_path, monkeypatch):
    own = [{"start": 1.0, "end": 2.0, "reason": "view_changed", "score": 0.3}]
    monkeypatch.setattr(near, "own_blind", lambda cam: own)
    monkeypatch.setattr(near.data, "out_dir", lambda cam: tmp_path)
    spans, src = near.blind_spans("x")
    assert spans == own and "not found" in src
    (tmp_path / "ledger.json").write_text('{"camera": "x", "bli', encoding="utf-8")  # half-written
    assert near.blind_spans("x")[0] == own


def test_contract_and_truth(result, blind):
    spans, _ = blind
    objs = near.objects(CAM).set_index("id")
    assert result["camera"] == CAM
    tg = result["targets"]
    assert near.MAX_TARGETS is None and len(tg) == len(near.pick_targets(near.objects(CAM))[0]) > 0  # all written
    counts = [len(t["visitors"]) for t in tg]
    assert counts == sorted(counts, reverse=True)
    for t in tg:
        assert set(t) == TARGET_KEYS
        assert bool(objs.loc[t["object_id"], "parked"])
        x1, y1, x2, y2 = t["box"]
        assert 0 <= x1 < x2 and 0 <= y1 < y2
        assert t["start"] < t["end"]
        assert "did it" not in t["summary"].lower() and "culprit" not in t["summary"].lower()
        for b in t["blind"]:
            assert t["start"] <= b["start"] <= b["end"] <= t["end"] and b["reason"]
        for v in t["visitors"]:
            assert set(v) == VISITOR_KEYS
            assert v["object_id"] != t["object_id"]
            assert not bool(objs.loc[v["object_id"], "parked"])
            assert 0 <= v["gap_rel"] <= near.NEAR_REL
            assert abs(v["gap_rel"] - v["gap_px"] / (y2 - y1)) < 0.01
            assert t["start"] <= v["enter_t"] <= v["closest_t"] <= v["exit_t"] <= t["end"]
            # never claim a visitor at a moment the camera could not see
            assert not any(s["start"] <= v["closest_t"] <= s["end"] for s in spans)
            # the gap is recomputable from the raw detection at closest_t
            g = data.tracks(CAM)
            row = g[(g.tid == v["object_id"]) & ((g.t - v["closest_t"]).abs() < 0.01)].iloc[0]
            gap = near.edge_gap(np.array(t["box"]), row.x1, row.y1, row.x2, row.y2)
            assert abs(float(gap) - v["gap_px"]) < 1.0


def test_simulated_blackout_hides_visitors(blind):
    """SIMULATED: declare 0-158 s blind; no Day A visitor may survive and targets must report the blind time."""
    spans, src = blind
    fake = near.merge_spans(spans + [{"start": 0.0, "end": 158.0, "reason": "dark", "score": 1.0}])
    mp = pytest.MonkeyPatch()
    mp.setattr(near, "blind_spans", lambda cam: (fake, "simulated"))
    try:
        r = near.scan(CAM, crops=False, write=False)
    finally:
        mp.undo()
    for t in r["targets"]:
        assert all(v["closest_t"] > 158.0 for v in t["visitors"])
        if t["start"] < 150:
            assert t["blind"] and "UNKNOWN" in t["summary"]


def test_cameras_without_targets_are_honest():
    r = near.scan("person_bicycle_car_detection", crops=False, write=False)
    assert r["camera"] == "person_bicycle_car_detection" and r["targets"] == []


def test_written_output_if_present():
    """If the CLI has run, its file must match the contract and every crop must be a real image."""
    p = data.OUT / CAM / "near.json"
    if not p.exists():
        pytest.skip("run python -m stm.near first")
    r = json.loads(p.read_text(encoding="utf-8"))
    for t in r["targets"]:
        assert set(t) == TARGET_KEYS
        for rel in [t["crop"]] + [v["crop"] for v in t["visitors"]]:
            img = cv2.imread(str(data.OUT / CAM / rel))
            assert img is not None and img.shape[0] > 4 and img.shape[1] > 4, rel
    summ = (data.OUT / CAM / "near_summary.txt").read_text(encoding="utf-8")
    if "capped at 40" in summ:
        pytest.skip("near.json predates the uncapped writer: rerun python -m stm.near " + CAM)
    found, written = map(int, re.search(r"targets found: (\d+); written: (\d+)", summ).groups())
    assert found == written == len(r["targets"])
    assert "own view-change" not in summ  # one blind record: the ledger's, when it exists
