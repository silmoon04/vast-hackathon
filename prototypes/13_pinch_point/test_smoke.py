"""Fast invariants (< 60 s) for the Pinch Point Register outputs. Run:  python test_smoke.py  (or pytest)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import protokit as pk  # noqa: E402
import run  # noqa: E402

RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
VIDS = RES["videos"]


def test_all_default_videos_have_outputs():
    for v in run.DEFAULT_SET:
        assert v in VIDS, v
        d = HERE / "out" / v
        for f in ("hero.jpg", "report.html", "plan_entry.md", "plan_entry.json", "zone_reference.jpg"):
            assert (d / f).exists(), (v, f)
        img = cv2.imread(str(d / "hero.jpg"))
        assert img is not None and img.shape[1] <= 1280 and img.shape[0] >= 600
        assert "Footage:" in (d / "report.html").read_text(encoding="utf-8")
        for e in VIDS[v]["evidence"]:
            im = cv2.imread(str(d / e["file"]))
            assert im is not None and im.shape[1] <= 1280


def test_backtest_semantics():
    sec = np.array([0, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 5, 5, 0])
    # literal rule (default, no dips allowed): runs 1-3 (3 s) and 5-8 (4 s) fire, the 2 s blip at 11-12 does not
    b = run.backtest(sec, 5, 3.0)
    assert b["gap_s"] == 0 and b["n_fires"] == 2
    assert [(f["start"], f["end"], f["fire_at"]) for f in b["fires"]] == [(1, 4, 4), (5, 9, 8)]
    assert b["time_above_s"] == 9
    # opt-in: forgiving a 1 s dip merges seconds 1..8 into one 8 s episode
    g = run.backtest(sec, 5, 3.0, gap=1.0)
    assert g["n_fires"] == 1 and g["fires"][0]["start"] == 1 and g["fires"][0]["end"] == 9
    assert run.backtest(sec, 6, 3.0)["n_fires"] == 0
    assert run.backtest(sec, 5, 1.0)["n_fires"] == 3
    assert run.backtest(np.zeros(10, int), 1, 1.0)["n_fires"] == 0


def test_trigger_is_literal_no_hidden_dip_merge():
    """Regression for the review finding: the series 12, 11, 12 must NOT count as '>= 12 for >= 3 s'."""
    sec = np.array([0, 12, 11, 12, 0])
    assert run.backtest(sec, 12, 3.0)["n_fires"] == 0
    assert run.backtest(sec, 11, 3.0)["n_fires"] == 1
    assert run.backtest(sec, 12, 3.0, gap=1.0)["n_fires"] == 1   # only when explicitly forgiven
    assert run.max_firing_n(run.sweep(sec, 3.0)) == 11
    # every reported fire in the produced results really has >= N on every second of its episode
    for v, r in VIDS.items():
        occ = np.array(r["occupancy_1hz"])
        b = r["backtest"]
        assert b["gap_s"] == 0, v
        for f in b["fires"]:
            seg = occ[int(f["start"]):int(f["end"])]
            assert len(seg) >= b["s"] and seg.min() >= b["n"], (v, f)
            assert f["fire_at"] <= f["end"]


def test_zone_feet_rule():
    rows = pd.DataFrame({"cx": [50.0, 50.0, 150.0], "y2": [80.0, 20.0, 80.0]})
    poly = [(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]  # lower half of a 100x100 frame... w=100 -> x 0..100
    m = run.in_zone_mask(rows, poly, 100, 100)
    assert m.tolist() == [True, False, False]
    assert run.in_zone_mask(pd.DataFrame({"cx": [10.0], "y2": [500.0]}), poly, 100, 100).tolist() == [True]  # cut-off feet


def test_sweep_monotone_and_matches_sql():
    for v, r in VIDS.items():
        sw = r["sweep"]
        t = [x["time_above_s"] for x in sw]
        assert all(a >= b for a, b in zip(t, t[1:])), v
        assert sw[-1]["time_above_s"] == 0 and sw[0]["time_above_s"] > 0
        assert r["peak_1hz"] == max(r["occupancy_1hz"])
        assert r["vastdb_sql_check"]["matches_python"], v
        assert r["recommend"]["n"] == int(np.ceil(np.percentile(r["occupancy_1hz"], 95)))
        b = r["backtest"]
        assert b == run.backtest(np.array(r["occupancy_1hz"]), b["n"], b["s"])


def test_recount_from_cached_tracks_matches_results():
    for v, r in VIDS.items():
        tr = pk.tracks(v)
        sz = run.zone_series(tr, [tuple(p) for p in r["zone"]["poly"]])
        assert sz["sec"].tolist() == r["occupancy_1hz"], v
        for e in r["evidence"]:
            gi = int(np.argmin(np.abs(sz["t"] - e["t"])))
            assert int(sz["frame_count"][gi]) == e["count"], (v, e)
            assert 0 <= e["count"] <= r["peak_frame_max"]


def test_plan_entry_json_complete():
    for v in run.DEFAULT_SET:
        j = json.loads((HERE / "out" / v / "plan_entry.json").read_text(encoding="utf-8"))
        assert j["trigger"]["min_people"] >= 1 and j["trigger"]["min_seconds"] > 0
        assert len(j["zone"]["poly"]) >= 3 and j["actions"] and j["evidence"]
        md = (HERE / "out" / v / "plan_entry.md").read_text(encoding="utf-8")
        for h in ("## Zone", "## Trigger", "## Back-test", "## Evidence frames", "## Actions"):
            assert h in md, (v, h)


def test_kill_test_recorded_honestly():
    k = RES["kill_test"]
    assert k["n_frames"] >= 3
    assert k["n_within"] == sum(r["within_20pct"] for r in k["rows"])
    # the headline claim only holds if most hand-counted peak frames agree; fail loudly if it regresses
    assert k["n_peak_within"] == k["n_peak_frames"] >= 3 and k["n_within"] >= 6, k["summary"]


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
