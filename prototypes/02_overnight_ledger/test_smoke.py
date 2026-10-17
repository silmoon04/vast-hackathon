"""Fast smoke checks (< 60 s) of the Overnight Ledger invariants, on the outputs produced by `python run.py --all`.
Run:  ../../.venv/Scripts/python.exe test_smoke.py      (or pytest -q test_smoke.py)"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import ledger as L  # noqa: E402
import protokit as pk  # noqa: E402

RES = json.loads((HERE / "results.json").read_text(encoding="utf-8")) if (HERE / "results.json").exists() else {}
GT = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
VIDEOS = ["ptz_parking", "lisbon_parking", "car_detection", "sim_a", "sim_b"]


def states(csv_path: Path) -> dict[int, str]:
    with open(csv_path, encoding="utf-8") as f:
        return {int(r["sec"]): r["state"] for r in csv.DictReader(f)}


# ------------------------------------------------------------------------------------------------ outputs exist
def test_outputs_exist_and_hero_size():
    for v in VIDEOS:
        assert v in RES, f"{v} missing from results.json (run python run.py --all)"
        hero = HERE / "out" / v / "hero.jpg"
        assert hero.exists() and hero.stat().st_size > 20_000, hero
        im = cv2.imread(str(hero))
        assert im is not None and im.shape[1] <= 1280 and im.shape[0] >= 600, (v, None if im is None else im.shape)
        rep = (HERE / "out" / v / "report.html").read_text(encoding="utf-8")
        assert "<img" in rep and "hero.jpg" in rep


def test_no_ledger_when_no_parked_car():
    for v in ("lisbon_parking", "car_detection"):
        assert RES[v]["ledger"] is False and "No parked car" in RES[v]["reason"]
    assert RES["lisbon_parking"]["hard_cuts"] > 1000          # photo slideshow, not a fixed camera
    assert RES["car_detection"]["longest_still_vehicle_s"] < 20


# ------------------------------------------------------------------------------------------------------ ptz ledger
def test_ptz_ledger_adds_up_and_finds_real_blind_window():
    r = RES["ptz_parking"]
    s = r["summary"]
    assert s["present_s"] + s["absent_s"] + s["different_s"] + s["cant_see_s"] == s["seconds"]
    big = [x for x in r["stretches"] if x["dur_s"] >= 60]
    assert len(big) == 1, r["stretches"]
    b = big[0]
    assert b["cause"] == "view" and 150 <= b["start"] <= 165 and 230 <= b["end"] <= 245        # eyeballed 155-157 .. 237-240 s
    assert b["before_state"] == "P" and b["after_state"] == "P" and b["verdict"] == "same car"
    assert r["verdict"] == "THERE, SAME CAR"


def test_ptz_contact_matches_eyeball():
    cts = RES["ptz_parking"]["contacts"]
    gt = GT["ptz_parking"]["car_A"]["person_close"]
    assert any(c["t0"] <= gt["to_s"] and c["t1"] >= gt["from_s"] for c in cts), cts


def test_strip_matches_eyeball_for_two_cars():
    gt = GT["ptz_parking"]
    a = states(HERE / "out" / "ptz_parking" / "ledger.csv")
    b = states(HERE / "out" / "ptz_parking" / "ledger_second_car.csv")
    sa = RES["ptz_parking"]["spot"]["box"]
    sb = RES["ptz_parking"]["second_car"]["spot"]["box"]
    assert max(abs(x - y) for x, y in zip(sa, gt["car_A"]["box"])) < 6, "main car is not the car that was eyeballed"
    assert max(abs(x - y) for x, y in zip(sb, gt["car_B"]["box"])) < 6, "second car is not the car that was eyeballed"
    bad, grey, ok = [], [], 0

    def run_len(led: dict[int, str], k: int) -> int:
        i = j = k
        while led.get(i - 1) == "G":
            i -= 1
        while led.get(j + 1) == "G":
            j += 1
        return j - i + 1
    for car, led in (("car_A", a), ("car_B", b)):
        for key, want in (("present", "P"), ("absent", "A"), ("blind", "G")):
            for k in gt[car].get(key, []):
                if led[k] == want:
                    ok += 1
                elif want != "G" and led[k] == "G" and run_len(led, k) <= 2:
                    grey.append((car, k))               # honest 1-2 s alignment dropout: 'can't see', never a wrong P/A
                else:
                    bad.append((car, k, "want " + want, led[k]))
    assert not bad, bad
    assert ok + len(grey) == 33 and len(grey) <= 2, (ok, grey)


def test_headline_numbers_agree_and_dropouts_are_accounted():
    r = RES["ptz_parking"]
    s = r["summary"]
    assert s["cant_see_s"] == sum(x["dur_s"] for x in r["stretches"]), (s, r["stretches"])   # no stray grey slivers
    assert s["filled_gap_s"] <= 4                                      # 1-2 s alignment dropouts between identical states, reported
    assert r["verdict"] == "THERE, SAME CAR" and "damage" in (HERE / "out" / "ptz_parking" / "report.html").read_text(encoding="utf-8").lower()


def test_touch_check_and_known_miss_are_reported():
    c = RES["ptz_parking"]["contacts"][0]["touch_check"]
    assert c["verdict"] == "same car" and 0.5 < c["ssim"] <= 1.0 and c["before_sec"] < 137 < c["after_sec"]
    d = RES["sim_b"]["sim_eval"]["damage_flagged"]
    assert not any(x["verdict"].startswith("CHANGED") for x in d)    # the scratch is a known miss: the hero must say so, not claim it
    assert "NOT detected" in str(RES["sim_b"].get("hero_note"))
    assert RES["ptz_parking"]["hero_badge"].startswith("SIMULATED camera faults: 36/36")


# ------------------------------------------------------------------------------- simulated faults (ground truth)
def test_simulated_faults_are_found_at_the_right_seconds():
    for v in ("sim_a", "sim_b"):
        e = RES[v]["sim_eval"]
        assert e["precision"] >= 0.9 and e["recall"] >= 0.9, (v, e["precision"], e["recall"])
        assert e["cause_accuracy"] >= 0.9, (v, e["cause_accuracy"])
        for f in e["per_fault"]:
            assert f["pred"] is not None, (v, f)
            assert abs(f["onset_error_s"]) <= 1 and abs(f["offset_error_s"]) <= 1, (v, f)
    kinds = {f["kind"] for v in ("sim_a", "sim_b") for f in RES[v]["sim_eval"]["per_fault"]}
    assert {"cover", "freeze", "blur", "cloth"} <= kinds


def test_clean_footage_has_no_false_blind_seconds():
    # real ptz window 0..157 s (before the tour): no blind second may be flagged
    led = states(HERE / "out" / "ptz_parking" / "ledger.csv")
    assert all(led[k] != "G" for k in range(0, 156)), [k for k in range(0, 156) if led[k] == "G"]


# ------------------------------------------------------------------------------------------------------- unit checks
def test_compare_identical_vs_different():
    rng = np.random.default_rng(0)
    base = (rng.random((90, 120, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2)
    c = L.compare(base, base.copy(), 10)
    assert c["ssim"] > 0.95 and not L.looks_changed(c)
    other = cv2.GaussianBlur((rng.random((90, 120, 3)) * 255).astype(np.uint8), (0, 0), 2)
    assert L.looks_changed(L.compare(base, other, 10))


def test_frozen_mask_needs_two_seconds_and_survives_keyframe_blip():
    d = np.full(100, 0.3)
    d[20:23] = 0.0                  # 3 identical samples = 0.6 s: not a freeze
    d[40:60] = 0.0
    d[50] = 0.5                     # encoder keyframe blip inside the freeze
    m = L.frozen_mask({"d": d})
    assert not m[20:23].any() and m[41:59].all() and m[50]


def test_iou_and_candidate_spots_prefer_the_parked_car():
    assert abs(L.iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1) < 1e-9 and L.iou((0, 0, 1, 1), (5, 5, 6, 6)) == 0
    rows = []
    for i in range(0, 600):                       # 40 s at 15 fps: tid 1 parked, tid 2 drives through
        t = i / 15
        rows.append([i * 2, t, 1, 2, "car", 0.9, 300, 300, 420, 380])
        rows.append([i * 2, t, 2, 2, "car", 0.9, 10 + i, 100, 130 + i, 180])
    df = pk._with_geom(pd.DataFrame(rows, columns=pk.COLS))
    tr = pk.Tracks("fake", 30.0, 2, 1280, 720, 40.0, {2: "car"}, df)
    c = L.candidate_spots(tr, [(0.0, 40.0)])
    assert len(c) == 1 and c[0].tids == [1] and c[0].stay >= 35


def test_aligner_recovers_a_known_pan():
    fr = pk.frame_at("ptz_parking", 100.4)
    shifted = cv2.warpAffine(fr, np.float32([[1, 0, 37], [0, 1, 14]]), (fr.shape[1], fr.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    M, n = L.Aligner(fr)(shifted)
    assert M is not None and n > 50
    p = L.apply_M(M, [[100.0, 100.0]])[0]
    assert abs(p[0] - 63) < 2.5 and abs(p[1] - 86) < 2.5, p       # shifted frame maps back onto the reference


if __name__ == "__main__":
    import time
    t0 = time.time()
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, str(e)[:300])
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'} in {time.time() - t0:.1f}s")
    sys.exit(1 if fails else 0)
