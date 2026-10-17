"""Fast smoke checks (< 60 s) of the key invariants. Needs `python run.py --all` to have been run once.

    python test_smoke.py        (or: pytest -q test_smoke.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import blind as bl  # noqa: E402

RES = HERE / "results.json"
NAMES = ["ptz_parking", "rochester_night", "retail_cctv_SIM"]


def _res() -> dict:
    assert RES.exists(), "run `python run.py --all` first"
    return json.loads(RES.read_text(encoding="utf-8"))["videos"]


# ----------------------------------------------------------------------------- pure-logic checks (no footage needed)


def _rows(states):
    return [{"s": i, "state": s, "M": None, "vis": 1.0, "src_w": 640} for i, s in enumerate(states)]


def _dets(n, hits):
    return [[{"name": "person", "conf": 0.9, "box": [10, 10, 50, 90]}] if i in hits else [] for i in range(n)]


def test_three_states():
    ok = _rows(["OK"] * 60)
    q = bl.Query.parse("person 0:10-0:40")
    assert bl.answer(ok, _dets(60, set()), q)["verdict"] == "NOTHING FOUND"
    assert bl.answer(ok, _dets(60, {20}), q)["verdict"] == "SEEN"
    blind = _rows(["OK"] * 20 + ["MOVED"] * 30 + ["OK"] * 10)
    a = bl.answer(blind, _dets(60, set()), q)
    assert a["verdict"] == "COULD NOT SEE" and a["naive"] == "NOTHING FOUND" and a["naive_misleading"]
    assert a["blind_spans"] == [{"t0": 20, "t1": 40, "state": "MOVED"}] and abs(a["coverage"] - 10 / 30) < 1e-9


def test_other_view_hits_are_not_a_contradiction():
    rows = _rows(["OK"] * 10 + ["MOVED"] * 20)
    q = bl.Query.parse("person 0:10-0:30")
    a = bl.answer(rows, _dets(30, {15}), q)
    assert a["naive"] == "SEEN" and a["verdict"] == "SEEN IN OTHER VIEW" and not a["evidence"] and a["evidence_off_view"]
    assert a["coverage"] == 0.0 and not a["naive_stale"]
    # a hit on the reference view wins over the other-view hit
    assert bl.answer(rows, _dets(30, {5, 15}), bl.Query.parse("person 0:00-0:30"))["verdict"] == "SEEN"


def test_stale_and_dark_hits_are_not_evidence():
    rows = _rows(["OK"] * 10 + ["FROZEN"] * 10 + ["DARK"] * 10)
    a = bl.answer(rows, _dets(30, {12, 25}), bl.Query.parse("person 0:10-0:30"))
    assert a["naive"] == "SEEN" and a["verdict"] == "COULD NOT SEE" and not a["evidence"] and a["naive_stale"]


def test_nothing_found_requires_coverage():
    rows = _rows(["OK"] * 58 + ["BLUR"] * 2)  # 96.7% coverage: still a valid NOTHING FOUND
    assert bl.answer(rows, _dets(60, set()), bl.Query.parse("person 0:00-1:00"))["verdict"] == "NOTHING FOUND"
    rows = _rows(["OK"] * 50 + ["BLUR"] * 10)
    assert bl.answer(rows, _dets(60, set()), bl.Query.parse("person 0:00-1:00"))["verdict"] == "COULD NOT SEE"


def test_blur_metric_orders_sharp_over_blurred():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (240, 320), dtype=np.uint8), (0, 0), 0.8)
    assert bl.blur_metric(cv2.GaussianBlur(img, (0, 0), 6)) > bl.blur_metric(img) + 0.15


def test_despeckle_and_spans():
    assert bl.despeckle(["OK", "MOVED", "OK", "OK"]) == ["OK", "OK", "OK", "OK"]
    assert bl.spans(["OK", "OK", "DARK", "DARK", "OK"]) == [(0, 2, "OK"), (2, 4, "DARK"), (4, 5, "OK")]


# ----------------------------------------------------------------------------- invariants on the produced outputs


def test_outputs_exist_and_hero_size():
    for n in NAMES:
        d = HERE / "out" / n
        assert (d / "report.html").stat().st_size > 5000, n
        h = cv2.imread(str(d / "hero.jpg"))
        assert h is not None and h.shape[1] <= 1280 and h.shape[0] >= 600, n
        assert "Local tests only" in (d / "report.html").read_text(encoding="utf-8")  # credit footer from pk.Report


def test_no_nothing_found_without_coverage():
    for n, r in _res().items():
        for q in r.get("queries", []):
            if q["verdict"] == "NOTHING FOUND":
                assert q["coverage"] >= bl.CFG["coverage_for_nothing"], (n, q["query"])
            if q["verdict"] == "SEEN":
                assert q["evidence"], (n, q["query"])


def test_sql_answers_match_python():
    n = 0
    for name, r in _res().items():
        for q in r.get("queries", []):
            if "sql_check" in q:
                n += 1
                assert q["sql_check"]["agrees"], (name, q["query"])
    assert n >= 12


def test_ptz_demo_moment():
    r = _res()["ptz_parking"]
    q = next(q for q in r["queries"] if q["query"].startswith("person 02:40"))
    assert q["verdict"] == "COULD NOT SEE" and q["naive"] == "NOTHING FOUND" and q["naive_misleading"]
    assert q["coverage"] == 0.0 and q["blind_s"] >= 78
    # the literal 02:00-04:00 window: a pedestrian really is visible before the PTZ leaves, so it is SEEN with a partial view
    q2 = next(q for q in r["queries"] if q["query"].startswith("person 02:00"))
    assert q2["verdict"] == "SEEN" and q2["coverage"] < 0.5
    assert r["state_seconds"]["MOVED"] > 150 and r["state_seconds"]["OK"] > 150
    ey = r["eyeball"]
    assert ey is not None and ey["agreement"] >= 0.97 and ey["away_recall"] >= 0.95, ey


def test_ptz_other_view_and_partial_view_rows():
    r = _res()["ptz_parking"]
    q = next(q for q in r["queries"] if q["query"].startswith("car 04:20"))
    assert q["verdict"] == "SEEN IN OTHER VIEW" and q["naive"] == "SEEN" and q["coverage"] == 0.0
    q = next(q for q in r["queries"] if q["query"].startswith("person 02:00"))  # literal brief window: partial view
    assert q["verdict"] == "SEEN" and 0.2 < q["coverage"] < 0.4 and q["blind_s"] >= 80


def test_sim_hero_leads_with_stale_frame():
    r = _res()["retail_cctv_SIM"]
    q = r["queries"][0]
    assert q["query"].startswith("person 00:37") and q["naive"] == "SEEN" and q["verdict"] == "COULD NOT SEE" and q["naive_stale"]


def test_heldout_simulated_faults():
    """Thresholds were frozen before these injections (other clips, other times / strengths): 4 spec faults must hold up."""
    allr = json.loads(RES.read_text(encoding="utf-8"))["videos"]
    ho = [v for k, v in allr.items() if k.startswith("heldout_")]
    assert len(ho) >= 2
    for v in ho:
        assert v["SIMULATED"] is True and v["false_alarm_seconds"] == 0, v["video"]
        for st in ("DARK", "FROZEN", "BLUR", "MOVED"):
            assert v["recall"][st] >= 0.90, (v["video"], st)


def test_rochester_is_night_not_dark():
    r = _res()["rochester_night"]
    assert r["state_seconds"]["DARK"] == 0 and r["state_seconds"]["MOVED"] > 50
    assert r["eyeball"]["agreement"] >= 0.95


def test_simulated_fault_recall_kill_test():
    r = _res()["retail_cctv_SIM"]
    assert r["SIMULATED"] is True
    assert r["recall_min_4_spec_faults"] >= 0.90, r["scores_smoothed"]
    for st in ("DARK", "FROZEN", "BLUR", "MOVED"):
        assert r["scores_smoothed"][st]["recall"] >= 0.90, st
    assert r["scores_smoothed"]["OK"]["precision"] >= 0.90


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
