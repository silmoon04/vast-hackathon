"""Fast invariants (< 60 s, no GPU) on the outputs produced by `python run.py --all`.

    python test_smoke.py            (or: python -m pytest -q test_smoke.py)
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
import protokit as pk  # noqa: E402
import run as rd  # noqa: E402

RES = json.loads((HERE / "results.json").read_text(encoding="utf-8")) if (HERE / "results.json").exists() else {"videos": {}}
VIDS = sorted(RES["videos"])


def _lap_energy(img: np.ndarray, mask: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    return float(lap[mask].mean()) if mask.any() else 0.0


def _frame(path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    assert ok, f"cannot read frame {idx} of {path.name}"
    return fr


# ------------------------------------------------------------------------------------------------ unit level


def test_results_exist():
    assert VIDS, "run `python run.py --all` first"
    assert set(rd.DEFAULT_SET) <= set(VIDS), f"missing default clips: {set(rd.DEFAULT_SET) - set(VIDS)}"


def test_judge_flags_every_non_requester():
    req = [(100, 100, 200, 300)]
    rect = [(90, 90, 210, 310)]
    same = {"name": "person", "conf": 0.9, "box": [102, 98, 198, 302]}
    stranger_far = {"name": "person", "conf": 0.5, "box": [400, 100, 480, 300]}
    stranger_inside = {"name": "person", "conf": 0.5, "box": [150, 120, 205, 230]}
    car = {"name": "car", "conf": 0.5, "box": [300, 300, 380, 360]}
    v = rd.judge([same], [], req, rect, "person")
    assert v["leaks"] == [] and v["inside"] == 1
    assert len(rd.judge([same, stranger_far], [], req, rect, "person")["leaks"]) == 1
    assert len(rd.judge([same, stranger_inside], [], req, rect, "person")["leaks"]) == 1, "a stranger INSIDE the rectangle is still a leak"
    assert len(rd.judge([car], [], req, rect, "person")["leaks"]) == 1, "other vehicles (plates) leak for a person request"
    pose_face = {"conf": 0.8, "box": [400, 100, 480, 300], "kp": [[420, 120, 0.9]] * 17}
    v = rd.judge([], [pose_face], req, rect, "person")
    assert v["face"] == 1 and any(l["kind"] == "pose:person" for l in v["leaks"])
    # a car request: a person detection is never "the requester"
    assert len(rd.judge([{"name": "person", "conf": 0.9, "box": [102, 98, 198, 302]}], [], req, rect, "car")["leaks"]) == 1


def test_gaps_stay_masked():
    fs = np.array([0, 2, 4, 6, 40])
    B = np.tile(np.array([[10.0, 10.0, 50.0, 90.0]]), (5, 1))
    assert rd._box_at(fs, B, 3, 25.0, 2)[1] == "interp"
    assert rd._box_at(fs, B, 4, 25.0, 2)[1] == "tracked"
    assert rd._box_at(fs, B, 20, 25.0, 2) is None, "a gap > MAX_GAP_S must not be bridged: that frame is fully masked"
    assert rd._box_at(fs, B, -3, 25.0, 2)[1] == "extrap"
    assert rd._box_at(fs, B, 200, 25.0, 2) is None


def test_region_opening_removes_slivers():
    """Review fix: a hole cut at a box edge must not leave a 1-2 px unmasked strip."""
    plan = rd.Plan(0, 0, 200, 300)
    plan.rects[0] = [(50, 50, 150, 150)]
    plan.holes[0] = [(40, 40, 148, 160)]          # leaves a 2 px strip at x = 148..150
    m = plan.region(0)
    assert not m[:, 148:150].any(), "2 px sliver survived"
    plan.holes[0] = []
    assert plan.region(0)[60:140, 60:140].all(), "opening must keep the real region"


def test_plan_drops_invisible_extrapolated_pad():
    """Review fix: pad frames whose extrapolated box is (almost) outside the frame are not released at all."""
    import pandas as pd
    rows = [[f, f / 10, 7, 0, "person", 0.9, 10.0 + 4 * (f - 20), 20.0, 50.0 + 4 * (f - 20), 100.0] for f in range(20, 60)]
    df = pd.DataFrame(rows, columns=pk.COLS)
    df["cx"], df["cy"], df["bw"], df["bh"] = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2, df.x2 - df.x1, df.y2 - df.y1
    tr = pk.Tracks("fake", 10.0, 1, 640, 480, 10.0, {0: "person"}, df)
    plan = rd.make_plan(tr, [7], (0.0, 10.0), 10.0, 100, (0, 99))
    assert plan.f0 >= 20 - int(rd.PAD_S * 10) and plan.f1 <= 59 + int(rd.PAD_S * 10)
    assert all(plan.rects[i] for i in range(plan.n)), "every released frame must have a region (no empty leading/trailing frames)"


def test_pixelate_destroys_detail():
    rng = np.random.default_rng(1)
    img = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    out = rd.pixelate(img)
    m = np.ones(img.shape[:2], bool)
    assert _lap_energy(out, m) < 0.02 * _lap_energy(img, m)


# ------------------------------------------------------------------------------------------ output level


def _records():
    for v in VIDS:
        out = HERE / "out" / v
        yield v, out, json.loads((out / "release_record.json").read_text(encoding="utf-8")), RES["videos"][v]


def test_files_and_hero():
    for v, out, rec, _ in _records():
        for f in ("hero.jpg", "report.html", "export.mp4", "release_record.json", "regions.json", "leaks.json"):
            assert (out / f).exists(), f"{v}: missing {f}"
        h = cv2.imread(str(out / "hero.jpg"))
        assert h is not None and h.shape[1] <= 1280, f"{v}: hero too wide"
        html = (out / "report.html").read_text(encoding="utf-8")
        assert rec["export"]["sha256"] in html and "Forced human review" in html, f"{v}: report lacks the record"


def test_hashes_and_frame_counts():
    for v, out, rec, res in _records():
        assert pk.sha256(out / "export.mp4") == rec["export"]["sha256"] == res["export_sha256"], f"{v}: export hash"
        assert pk.sha256(pk.video_path(rec["source"]["video"])) == rec["source"]["sha256"] == res["source_sha256"], f"{v}: source hash"
        cap = cv2.VideoCapture(str(out / "export.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == rec["export"]["frames_expected"] == rec["trim"]["frames_released"], f"{v}: frames {n}"
        assert abs(n / rec["source"]["fps"] - rec["trim"]["seconds_released"]) < 0.1


def test_trim_inside_window_and_shot():
    for v, out, rec, _ in _records():
        lo, hi = rec["request"]["window_s"]
        t = rec["trim"]
        assert t["released_from_s"] >= lo - 0.05 and t["released_to_s"] <= hi + 0.1, f"{v}: release leaves the requested window"
        assert t["released_from_s"] >= t["shot_s"][0] - 0.05 and t["released_to_s"] <= t["shot_s"][1] + 0.1, f"{v}: release crosses a cut"
        assert t["seconds_released"] <= rec["source"]["duration_s"]
        assert t["released_from_s"] <= t["first_seen_s"] + 0.02 and t["released_to_s"] >= t["last_seen_s"] - 0.02, f"{v}: requester trimmed off"


def test_mask_by_default_pixels():
    """Outside the released rectangles the exported pixels carry (almost) no detail; inside they keep it."""
    for v, out, rec, _ in _records():
        reg = json.loads((out / "regions.json").read_text(encoding="utf-8"))
        n = len(reg["rects"])
        src = pk.video_path(rec["source"]["video"])
        for i in sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1}):
            fe = _frame(out / "export.mp4", i)
            fo = _frame(src, reg["f0"] + i)
            h, w = fe.shape[:2]
            rm = rd.mask_of([tuple(r) for r in reg["rects"][i]], h, w)
            outside = ~cv2.dilate(rm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)  # ignore the codec-smeared edge
            eo, es = _lap_energy(fe, outside), _lap_energy(fo, outside)
            # the burned-in caption sits on the bottom 30 px: exclude it
            outside[h - 30:, :] = False
            eo, es = _lap_energy(fe, outside), _lap_energy(fo, outside)
            if es > 1.0:
                assert eo < 0.25 * es + 0.3, f"{v} frame {i}: outside-region detail {eo:.2f} vs source {es:.2f}: NOT masked"
            if reg["modes"][i] == "gap":
                assert _lap_energy(fe, np.ones((h, w), bool)) < 0.5 * _lap_energy(fo, np.ones((h, w), bool)) + 0.3


def test_leak_check_clean_and_checker_works():
    for v, out, rec, res in _records():
        lk = rec["leak_check"]
        assert lk["frames_checked"] >= min(20, rec["trim"]["frames_released"]), f"{v}: too few frames checked"
        if v in rd.DEFAULT_SET:  # review fix: the stamp claims every export frame was checked
            assert lk["frames_checked"] >= rec["trim"]["frames_released"], f"{v}: not every export frame was leak-checked"
            assert res["every_frame_checked"] and "export frames" in res["stamp"]
        assert lk["leaks_in_export"] == 0 and res["verdict"] != "BLOCKED", f"{v}: LEAKS {lk['leaks_in_export']} in {lk['leak_frames'][:5]}"
        assert lk["face_keypoint_leaks"] == 0
        assert lk["rounds"][-1]["leaks"] == lk["leaks_in_export"], "final round must be the certified file"
        if len(lk["rounds"]) > 1:
            assert lk["rounds"][1]["frames_checked"] >= lk["rounds"][0]["frames_checked"]
        fi = rec["fault_injection"]
        if fi.get("injected", 0):
            assert fi["caught"] >= 0.8 * fi["injected"], f"{v}: the checker missed injected leaks ({fi['caught']}/{fi['injected']})"
        assert lk["control_requester_redetected_in_export_pct"] >= 40, f"{v}: detectors cannot even see the requester in the export"


def test_risky_frames_have_review_list():
    for v, out, rec, _ in _records():
        r = rec["risky"]
        assert r["frames"] == sum(i["frames"] for i in r["interval_list"])
        if r["frames"]:
            assert r["crops"], f"{v}: risky frames but no crops"
            assert all((out / c).exists() for c in r["crops"])
            assert rec["sign_off"]["officer"] == "PENDING"


def test_kill_test_and_record_fields():
    for v, out, rec, res in _records():
        kt = rec["kill_test"]
        assert kt["passed"] and kt["requester_height_px"] >= 60 and kt["export_x_realtime"] >= 1.0, f"{v}: {kt}"
        for k in ("request", "source", "requester", "trim", "leak_check", "risky", "export", "compute", "sign_off"):
            assert k in rec, f"{v}: record lacks {k}"
        assert rec["compute"]["home_office_baseline"]["note"]
        assert res["requester_tids"] == rec["requester"]["tids"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-q", str(Path(__file__)), "-p", "no:cacheprovider"]))
