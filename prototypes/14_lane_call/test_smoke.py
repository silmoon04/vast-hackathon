"""Fast invariants of LANE CALL on the produced outputs (run `python run.py --all --bonus` first). < 60 s.

    python test_smoke.py        (or: python -m pytest test_smoke.py -q)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import lanecall as L  # noqa: E402
import lanecall_core as lc  # noqa: E402
import protokit as pk  # noqa: E402

CALLS = {L.IN_LANE, L.AT_KERB, L.CANT}
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
V = R["videos"]


def _rows(boxes_by_t: dict[float, list[tuple]]) -> pd.DataFrame:
    rows = []
    for k, (t, bs) in enumerate(sorted(boxes_by_t.items())):
        for tid, (x1, y1, x2, y2) in bs:
            rows.append({"f": k, "t": t, "tid": tid, "cls": 2, "name": "car", "conf": 0.8, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return pd.DataFrame(rows)


def test_detector_survives_id_switches_and_ignores_movers():
    fps = 10.0
    still = {k / fps: [(1 + (k // 15), (100 + (k % 3), 100, 220 + (k % 2), 180))] for k in range(0, 90)}  # id changes every 1.5 s
    assert len(lc.stopped_alerts(_rows(still), fps, S=5.0)) == 1
    moving = {k / fps: [(1, (100 + 6 * k, 100, 220 + 6 * k, 180))] for k in range(0, 90)}
    assert lc.stopped_alerts(_rows(moving), fps, S=5.0) == []
    short = {k / fps: [(1, (100, 100, 220, 180))] for k in range(0, 30)}  # 3 s only
    assert lc.stopped_alerts(_rows(short), fps, S=5.0) == []


def test_zone_geometry_and_registration_on_reference_frame():
    cams = L.load_cams("ptz_parking")
    assert {c.id for c in cams} == {"ptz_A", "ptz_B"}
    b = next(c for c in cams if c.id == "ptz_B")
    reg = L.register(b.ref, b)
    assert reg and reg["inliers"] > 200 and reg["shift_px"] < 2.0          # a frame registers onto itself
    assert b.zone_at((0.5 * b.w, 0.625 * b.h))[0] == "lane"                # the aisle
    assert b.zone_at((0.5 * b.w, 0.85 * b.h))[0] == "kerb"                 # a bay
    a = next(c for c in cams if c.id == "ptz_A")
    assert L.register(a.ref, b) is None or L.register(a.ref, b)["shift_px"] > 20  # preset A is not preset B


def test_detector_survives_a_3s_occlusion_of_a_stopped_car():
    fps = 10.0
    seen = {k / fps: [(1, (100, 100, 220, 180))] for k in range(0, 160) if not (60 <= k < 90)}   # hidden 6.0 to 9.0 s
    assert len(lc.stopped_alerts(_rows(seen), fps, S=5.0)) == 1


def test_results_cover_default_set():
    for v in ("lisbon_parking", "motorway_a40", "motorway_a40_inject", "ptz_parking"):
        assert v in V, v
        assert (HERE / "out" / v / "hero.jpg").exists() and (HERE / "out" / v / "report.html").exists()


def test_a40_real_clip_expects_zero_alerts():
    assert V["motorway_a40"]["n_alerts"] == 0 and V["motorway_a40"]["cards"] == []


def test_a40_injected_stops_run_through_the_real_detector_and_are_called_right():
    v = V["motorway_a40_inject"]
    assert v["control_clean_alerts"] == 0                                   # same clip, nothing pasted: quiet
    assert v["n_alerts"] == 3 and v["by_call"][L.IN_LANE] == 2 and v["by_call"][L.AT_KERB] == 1
    assert [c["call"] for c in v["cards"]][:2] == [L.IN_LANE, L.IN_LANE]    # IN LANE first
    assert all(c["foot_src"] == "seg-mask" for c in v["alerts"])
    assert v["accuracy"]["right"] == v["accuracy"]["judged"] == 3
    assert "SIMULATED" in (HERE / "out" / "motorway_a40_inject" / "report.html").read_text(encoding="utf-8")
    assert all(c["t_alert"] >= 5.0 for c in v["alerts"])                    # fired only after the S = 5 s rule


def test_every_card_is_well_formed_and_queue_is_severity_ordered():
    for name, v in V.items():
        order = [L.SEV[c["call"]] for c in v["cards"]]
        assert order == sorted(order), f"{name}: queue not severity ordered"
        assert v["n_cards"] == len(v["cards"]) == sum(v["by_card"].values())
        assert v["n_alerts"] == len(v["alerts"]) == sum(v["by_call"].values())
        assert v["n_cards"] <= v["n_alerts"]
        for c in v["alerts"]:
            assert c["call"] in CALLS and 0.0 <= c["conf"] <= 1.0 and c["reason"]
            assert len(c["box"]) == 4 and c["dur"] >= 0
            if c["call"] == L.IN_LANE:
                assert c["lane_frac"] is not None and c["lane_frac"] >= L.LANE_MIN
        for c in v["cards"]:
            assert c["call"] in CALLS | {L.QUEUE}


def test_lisbon_has_in_lane_first_and_is_flagged_as_a_slideshow():
    lis = V["lisbon_parking"]
    assert lis["mode"] == "slides" and lis["n_slides"] > 1000 and lis["native_alerts_at_S"] <= 2
    assert lis["by_call"][L.IN_LANE] >= 3 and lis["cards"][0]["call"] == L.IN_LANE
    assert "SIMULATED" in (HERE / "out" / "lisbon_parking" / "report.html").read_text(encoding="utf-8")


def test_red_light_queue_collapses_to_one_low_severity_card():
    y = V["yonge_dundas"]
    q = [c for c in y["cards"] if c["call"] == L.QUEUE]
    assert len(q) == 1 and q[0]["n_members"] >= 3 and y["n_cards"] < y["n_alerts"]
    assert y["cards"][-1]["call"] == L.QUEUE or all(c["call"] != L.AT_KERB for c in y["cards"])   # below lone IN LANE and CAN'T SEE


def test_queue_grouping_rule_on_synthetic_alerts():
    def a(i, x, t_stop, zone="carriageway", cam="c"):
        return {"call": L.IN_LANE, "cam": cam, "zone": zone, "t_stop": t_stop, "box": [x, 100, x + 100, 180]}
    import run
    assert len(run.group_queues([a(0, 100, 0), a(1, 300, 2), a(2, 520, 4)])) == 1          # 3 stacked, together: a queue
    assert run.group_queues([a(0, 100, 0), a(1, 300, 2)]) == []                              # 2 is not a queue
    assert run.group_queues([a(0, 100, 0), a(1, 300, 40), a(2, 520, 80)]) == []              # not together in time
    assert run.group_queues([a(0, 100, 0), a(1, 300, 2, cam="d"), a(2, 520, 4)]) == []       # other camera
    assert run.group_queues([a(0, 100, 0), a(1, 3000, 2), a(2, 6000, 4)]) == []              # far apart


def test_ptz_repeat_alerts_are_merged_across_presets():
    p = V["ptz_parking"]
    assert len(p["merged"]) >= 5 and p["n_alerts"] < 87
    assert {m["dropped"]["cam"] for m in p["merged"]} | {m["kept"]["cam"] for m in p["merged"]}


def test_ground_contact_line_follows_an_angled_silhouette_and_confidence_scales_with_lane_fraction():
    prof = [[100 + 10 * k, 200 - 6 * k] for k in range(12)]                  # contact line rises to the right: an angled car
    fp = L.footprint([100, 100, 220, 200], prof=prof)
    assert fp[0, 1] - fp[-1, 1] > 20                                          # diagonal, not a flat box edge
    flat = L.footprint([100, 100, 220, 200])
    assert np.allclose(flat[:, 1], flat[0, 1])
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    cam = L.Cam({"id": "t", "video": "x", "zones": [{"name": "lane", "kind": "lane", "pts": [[0, 0.6], [1, 0.6], [1, 1], [0, 1]]},
                                                    {"name": "verge", "kind": "kerb", "pts": [[0, 0], [1, 0], [1, 0.6], [0, 0.6]]}]}, img)
    reg = {"H": np.eye(3), "inliers": 100, "ratio": 1.0, "area": 1.0, "shift_px": 0.0}
    confs = []
    for ya, yb in ((400, 400), (250, 330), (200, 330)):                       # contact line fully in the lane, ~70 %, ~40 % in the lane
        prof = [[200 + 24 * k, ya + (yb - ya) * k / 5] for k in range(6)]
        c = L.make_call(img, [200, max(ya, yb) - 80, 320, max(ya, yb)], cam, reg, [], 0.8, prof=prof)
        confs.append((c["lane_frac"], c["conf"], c["call"]))
    assert all(c[2] == L.IN_LANE for c in confs)
    assert confs[0][0] > confs[1][0] > confs[2][0] and confs[0][1] > confs[1][1] > confs[2][1], confs   # confidence follows the fraction
    assert confs[1][1] < 0.9                                                  # a 70 % overlap no longer shows as 97 %
    cc = L.make_call(img, [200, 170, 320, 250], cam, reg, [], 0.8, prof=[[200, 250], [230, 250], [260, 250], [290, 250], [320, 250]])
    assert cc["call"] == L.AT_KERB


def test_queue_order_matches_the_vastdb_style_sql():
    for name, v in V.items():
        assert v["sql_queue_matches_python"], name
        assert v["sql_counts"] == v["by_card"], name


def test_alert_to_card_latency_budget():
    assert R["overall"]["card_ms_max"] < 2000, R["overall"]
    assert R["overall"]["under_2s"]


def test_single_video_runs_merge_into_results_instead_of_overwriting():
    import run
    agg = run.aggregate(V)
    for k in ("alerts", "cards", "eyeball_right", "eyeball_judged", "in_lane_precision", "in_lane_recall"):
        assert agg[k] == R["overall"][k], k
    assert set(R["set"]) == set(V)


def test_hero_images_are_product_shots():
    for v in V:
        img = cv2.imread(str(HERE / "out" / v / "hero.jpg"))
        assert img is not None and img.shape[1] <= 1280 and img.shape[0] >= 600
        assert float(img.std()) > 25, f"{v}: hero looks blank"


def test_eyeball_accuracy_is_reproducible_from_ground_truth():
    gt = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
    right = judged = 0
    for name, v in V.items():
        if name not in gt:
            continue
        for c in v["alerts"]:
            cx, cy = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
            m = [g for g in gt[name]["alerts"] if abs(g["t_alert"] - c["t_alert"]) < 1 and abs(g["cx"] - cx) < 12 and abs(g["cy"] - cy) < 12]
            if m:
                judged += 1
                t = m[0]["truth"]
                right += int(t == c["call"] or (t == "UNJUDGEABLE" and c["call"] == L.CANT))
    assert judged >= 100
    assert R["overall"]["eyeball_judged"] == judged and R["overall"]["eyeball_right"] == right
    print(f"eyeball: {right}/{judged} = {right / judged:.1%}")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-s"]))
