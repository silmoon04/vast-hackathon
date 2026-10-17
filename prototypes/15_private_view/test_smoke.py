"""Fast invariants of PRIVATE VIEW on the produced outputs (run `python run.py --all` first). < 60 s, no GPU, no network."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import pview as pv  # noqa: E402
from pview_motion import MotionTracker  # noqa: E402

VIDEOS = ["ptz_parking", "rochester_night", "rochester_real", "lisbon_parking"]


def results() -> dict:
    return json.loads((HERE / "results.json").read_text(encoding="utf-8"))


def test_outputs_exist_and_hero_is_small_enough():
    for v in VIDEOS:
        d = HERE / "out" / v
        for f in ("report.html", "hero.jpg", "overlay.mp4", "exposure_log.csv", "samples.csv", "strip.png", "zones_setup.jpg"):
            assert (d / f).exists() and (d / f).stat().st_size > 100, f"{v}/{f} missing"
        im = cv2.imread(str(d / "hero.jpg"))
        assert im is not None and im.shape[1] <= 1280 and im.shape[0] >= 600, f"{v} hero size {None if im is None else im.shape}"
        cap = cv2.VideoCapture(str(d / "overlay.mp4"))
        ok, fr = cap.read()
        assert ok and fr.shape[1] >= 600, f"{v} overlay unreadable"
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 20
        cap.release()
        html = (d / "report.html").read_text(encoding="utf-8")
        assert "hero.jpg" in html and "Footage:" in html     # credit line from pk.Report.credit()


def test_exposure_maths():
    zone = pv.Zone("T", "t", np.float32([[100, 100], [200, 100], [200, 150], [100, 150]]))
    shape = (400, 600)
    masks = pv.static_masks([zone], shape, margin_px=5)
    ident = pv.evaluate([zone], masks, np.eye(3), shape)[0]
    assert ident["state"] == "ok" and ident["exposed"] < 0.5 and ident["mis"] < 1e-3
    # window slides 25 px right: 20 px stick out of the (5 px margin) mask -> 20 % of a 100 px wide window
    T = np.float64([[1, 0, 25], [0, 1, 0], [0, 0, 1]])
    r = pv.evaluate([zone], masks, T, shape)[0]
    assert abs(r["exposed"] - 20.0) < 2.5, r
    # window moves completely out of its mask: 100 % exposed
    T2 = np.float64([[1, 0, 160], [0, 1, 60], [0, 0, 1]])
    assert pv.evaluate([zone], masks, T2, shape)[0]["exposed"] > 99
    # lost view is "unverified", never a silent 0 %
    lost = pv.evaluate([zone], masks, None, shape)[0]
    assert lost["state"] == "unverified" and math.isnan(lost["exposed"])
    # window pushed out of the frame: not counted as exposed
    T3 = np.float64([[1, 0, 900], [0, 1, 0], [0, 0, 1]])
    assert pv.evaluate([zone], masks, T3, shape)[0]["state"] == "offframe"


def test_event_log_logic():
    ts = [round(i * 0.1, 1) for i in range(100)]
    vals = [0.0] * 20 + [50.0] * 30 + [0.0] * 50
    ev = pv.make_events(ts, 0.1, {"W": vals}, [False] * 100, 5.0, 0.3, 0.5, {"W": [1.0] * 100})
    assert len(ev) == 1 and ev[0]["start_s"] == 2.0 and ev[0]["end_s"] == 5.0 and ev[0]["max_exposed_pct"] == 50.0
    lost = [False] * 60 + [True] * 20 + [False] * 20
    ev2 = pv.make_events(ts, 0.1, {"W": [0.0] * 100}, lost, 5.0, 0.3, 0.5, {"W": [1.0] * 100})
    assert [e["kind"] for e in ev2] == ["UNVERIFIED"] and abs(ev2[0]["duration_s"] - 2.0) < 1e-6


def test_injection_matrix_is_identity_outside_window():
    inj = pv.Injection("step", 8.0, 18.0, dx=28, dy=12, rot=1.0)
    assert np.allclose(inj.matrix(7.9, 1280, 720), np.eye(3)) and np.allclose(inj.matrix(18.0, 1280, 720), np.eye(3))
    M = inj.matrix(10.0, 1280, 720)
    assert not np.allclose(M, np.eye(3)) and abs(np.linalg.det(M[:2, :2]) - 1.0) < 1e-6


def test_tracker_recovers_a_known_shift_on_a_real_frame():
    import protokit as pk
    ref = pk.frame_at("rochester_night", 10.0)
    h, w = ref.shape[:2]
    M = pv.Injection("step", 0, 1, dx=-31, dy=17, rot=1.4).matrix(0.5, w, h)
    r = MotionTracker(ref).update(pv.inject(ref, M))
    assert r["H"] is not None and r["inliers"] > 150
    pc = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    err = np.linalg.norm(pv.warp_poly(pc, r["H"]) - pv.warp_poly(pc, M), axis=1).mean()
    assert err < 1.0, err


def test_rochester_kill_test_numbers():
    v = results()["videos"]["rochester_night"]
    t = v["truth"]
    assert v["simulated"] and t["n_truth_zones"] >= 5
    assert t["false_alarm_events"] == [] and t["all_pass"]
    for z in t["per_zone"]:
        assert z["detected"] and abs(z["d_start_s"]) <= 0.5 and abs(z["d_end_s"]) <= 0.5
        assert abs(z["d_max_pp"]) <= 10 and abs(z["d_mean_pp"]) <= 10
        assert abs(z["truth_start"] - 8.0) <= 0.25 and abs(z["truth_end"] - 18.0) <= 0.25   # truth is the injected 8-18 s window


def test_ptz_story():
    v = results()["videos"]["ptz_parking"]
    ex = [e for e in v["events"] if e["kind"] == "EXPOSED"]
    un = [e for e in v["events"] if e["kind"] == "UNVERIFIED"]
    assert ex and max(e["max_exposed_pct"] for e in ex) >= 90          # after the PTZ move the masks cover nothing
    assert any(e["duration_s"] >= 30 for e in ex)
    assert un and max(e["duration_s"] for e in un) >= 30                # zoomed-in view cannot be registered
    assert min(e["start_s"] for e in ex + un) >= 100                    # first 2 min: camera on its preset, no alarm
    lm = v["landmark_check"]
    assert lm["max_err_px"] <= 12, lm


def test_lisbon_shake_is_tracked_per_frame():
    v = results()["videos"]["lisbon_parking"]
    t = v["truth"]
    assert v["simulated"] and t["mean_abs_err_pp_all_samples"] <= 3.0 and t["max_abs_err_pp_all_samples"] <= 15
    assert t["false_alarm_events"] == []


def test_headlines_list_every_window_and_do_not_merge_different_events():
    import run
    ev = [{"zone": z, "kind": "EXPOSED", "start_s": 8.0, "end_s": 18.0, "duration_s": 10.0, "max_exposed_pct": p}
          for z, p in (("W1", 9.0), ("W7", 37.4), ("W6", 33.9))]
    g = run.group_events(ev)
    assert len(g) == 1
    txt = run.group_sentence(g[0])
    assert txt.index("W7 37%") < txt.index("W6 34%") < txt.index("W1 9%") and "for 10 s" in txt     # each window's own peak, sorted
    lis = [{"zone": "V2", "kind": "EXPOSED", "start_s": 532.2, "end_s": 532.56, "duration_s": 0.36, "max_exposed_pct": 9.5},
           {"zone": "V3", "kind": "EXPOSED", "start_s": 532.2, "end_s": 533.12, "duration_s": 0.92, "max_exposed_pct": 25.4}]
    assert len(run.group_events(lis)) == 2                                                          # different spans stay separate
    r = results()["videos"]
    for name in ("rochester_night", "rochester_real", "lisbon_parking"):
        gs = run.group_events(r[name]["events"])
        assert gs and all(len(x["zone_pct"]) == len(x["zones"]) for x in gs)


def test_hero_stamp_hides_no_window_and_cosmos_is_labelled_unusable():
    r = results()["videos"]
    for name in ("rochester_night", "rochester_real", "lisbon_parking"):
        assert r[name]["hero"]["stamp_covered_window_px"] <= 5, (name, r[name]["hero"])
    v = r["rochester_night"]["setup"]
    assert "UNUSABLE" in v["cosmos_verdict"] and v["cosmos_recall_iou0.3"] < 0.5


def test_logged_series_hides_sub_minimum_blips():
    import run
    o = {"ids": ["A"], "ts": [0.0, 0.1, 0.2, 0.3, 0.4], "series": {"A": [0.0, 30.0, 0.0, 40.0, 40.0]},
         "events": [{"zone": "A", "kind": "EXPOSED", "start_s": 0.3, "end_s": 0.5}]}
    assert run.logged_series(o)["A"] == [0.0, 0.0, 0.0, 40.0, 40.0]


def test_real_footage_cross_check_two_independent_estimators_agree():
    xc = results()["videos"]["rochester_real"]["ecc_crosscheck"]
    assert xc["samples"] >= 80 and xc["corner_err_px_mean"] <= 1.0 and xc["corner_err_px_max"] <= 4.0, xc
    assert xc["exposure_err_pp_mean"] <= 1.5 and xc["exposure_err_pp_max"] <= 6.0, xc
    assert xc["events_matched"] == xc["events_sift"] == xc["events_ecc"], xc
    m = results()["videos"]["rochester_real"]
    assert not m["simulated"] and m["n_exposed_events"] >= 8                                        # real drift breaches, no injection
    assert 5 <= max(e["max_exposed_pct"] for e in m["events"] if e["kind"] == "EXPOSED") < 60       # a partial exposure, not 100 %


def test_non_circular_knock_on_the_raw_clip():
    k = results()["checks"]["raw_knock"]
    assert "RAW" in k["design"] and k["n_knock_events"] >= 7 and k["n_knock_pass"] == k["n_knock_events"], k
    assert k["all_pass"] and k["false_alarm_events"] == [] and k["tracker_lost_samples"] == 0
    assert k["mean_abs_err_pp_all_samples"] <= 2.0 and k["max_abs_err_pp_all_samples"] <= 10.0
    for p in k["per_event"]:
        if p["knock"]:
            assert abs(p["d_start_s"]) <= 0.5 and abs(p["d_end_s"]) <= 0.5 and abs(p["d_max_pp"]) <= 10
            assert abs(p["truth_start"] - 8.0) <= 0.25 and abs(p["truth_end"] - 14.0) <= 0.25
    assert max(k["truth_peak_pct_by_zone"].values()) < 60                                           # partial exposure, not a 100 % case


def test_results_json_has_checks_and_no_nans():
    r = results()
    assert r["checks"]["shift_sweep"]
    s = json.dumps(r)
    assert "NaN" not in s


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, str(e)[:200])
    print("OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
