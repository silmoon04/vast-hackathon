"""Fast invariants for LAST UPRIGHT (< 60 s, no GPU, no network): synthetic posture logic + the produced outputs.

    ../../.venv/Scripts/python.exe test_smoke.py        (or: pytest test_smoke.py)
Run `python run.py --all` first; the output checks need results.json and out/<video>/hero.jpg."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import posture as P  # noqa: E402

DEFAULT_SET = ["skate_fall", "floor_lie", "assist_elderly", "help_sit", "sit_down", "classroom"]


# ----------------------------------------------------------------------------------- synthetic skeletons
def skeleton(kind: str, tilt_deg: float = 0.0, cx: float = 300.0) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """17x3 COCO keypoints for a stand / sit / lie figure (image coords, y down), torso rotated by tilt_deg."""
    kp = np.zeros((17, 3))
    kp[:, 2] = 0.9
    th = np.radians(tilt_deg)
    ux, uy = np.sin(th), -np.cos(th)           # unit vector hips -> shoulders (up when tilt = 0)
    H = np.array([cx, 300.0])
    S = H + 100 * np.array([ux, uy])
    head = H + 160 * np.array([ux, uy])
    for i in (5, 6):
        kp[i, :2] = S + (-12 if i == 5 else 12) * np.array([-uy, ux])
    for i in (11, 12):
        kp[i, :2] = H + (-10 if i == 11 else 10) * np.array([-uy, ux])
    for i in range(5):
        kp[i, :2] = head
    for i in (7, 8, 9, 10):
        kp[i, :2] = S + np.array([0, 40.0])
    if kind == "stand":
        for i in (13, 14):
            kp[i, :2] = H + np.array([0, 95.0])
        for i in (15, 16):
            kp[i, :2] = H + np.array([0, 190.0])
    elif kind == "sit":
        for i in (13, 14):
            kp[i, :2] = H + np.array([90.0, 5.0])
        for i in (15, 16):
            kp[i, :2] = H + np.array([95.0, 100.0])
    else:  # lie: legs continue the torso line downwards
        d = -np.array([ux, uy])
        for i in (13, 14):
            kp[i, :2] = H + 95 * d
        for i in (15, 16):
            kp[i, :2] = H + 190 * d
    pts = kp[:, :2]
    box = (pts[:, 0].min() - 10, pts[:, 1].min() - 10, pts[:, 0].max() + 10, pts[:, 1].max() + 10)
    return kp, box


def posture_of(kind: str, tilt: float) -> str:
    kp, box = skeleton(kind, tilt)
    return P.raw_posture(P.features_row(kp, box), False)


def series(spec: list[tuple[float, float, str, float]], fps: float = 15.0) -> pd.DataFrame:
    """spec: (t0, t1, kind, tilt) pieces, tilt linearly interpolated inside a piece -> a poses DataFrame (one tid)."""
    rows, f = [], 0
    for t0, t1, kind, tilt in spec:
        for t in np.arange(t0, t1, 1 / fps):
            kp, box = skeleton(kind, tilt)
            rows.append({"f": f, "t": round(float(t), 3), "tid": 1, "conf": 0.9, "x1": box[0], "y1": box[1], "x2": box[2],
                         "y2": box[3], "kp": kp.tolist()})
            f += 1
    return pd.DataFrame(rows)


def ramp(t0: float, t1: float, kind: str, a0: float, a1: float, fps: float = 15.0) -> list[tuple[float, float, str, float]]:
    ts = np.arange(t0, t1, 1 / fps)
    return [(float(t), float(t) + 1 / fps, kind, float(a0 + (a1 - a0) * (t - t0) / max(t1 - t0, 1e-6))) for t in ts]


def events_of(spec) -> list[dict]:
    df = series(spec)
    res, _ = P.analyse(df)
    return res[0].events if res else []


# ------------------------------------------------------------------------------------------------- tests
def test_static_postures():
    assert posture_of("stand", 0) == P.UP
    assert posture_of("sit", 0) == P.SIT
    assert posture_of("lie", 90) == P.DOWN
    assert posture_of("stand", 20) == P.UP          # a lean is not a fall
    assert posture_of("stand", 45) != P.DOWN        # bending 45 deg is not lying


def test_schmitt_hysteresis():
    kp, box = skeleton("lie", 50)
    f = P.features_row(kp, box)
    assert P.raw_posture(f, was_down=False) != P.DOWN   # 50 deg does not enter DOWN (needs 60)
    assert P.raw_posture(f, was_down=True) == P.DOWN    # but once down it stays down until < 42 deg


def test_flicker_removed():
    t = np.arange(0, 3, 1 / 15)
    raw = [P.UP] * len(t)
    for i in (20, 21):                                   # 2 stray DOWN frames
        raw[i] = P.DOWN
    assert P.DOWN not in P.smooth_states(t, raw)


def test_fast_fall_is_fall():
    spec = [(0, 3, "stand", 0)] + ramp(3, 3.4, "lie", 0, 90) + [(3.4, 8, "lie", 90)]
    down = [e for e in events_of(spec) if e["kind"] == "down"]
    assert len(down) == 1 and down[0]["speed"] == "FAST"
    assert 3.1 <= down[0]["last_upright"] <= 3.4 and 3.3 <= down[0]["first_down"] <= 3.9
    assert down[0]["down_s"] > 3


def test_slow_lie_down_is_not_fall():
    spec = [(0, 3, "stand", 0)] + ramp(3, 9, "lie", 0, 90) + [(9, 14, "lie", 90)]
    down = [e for e in events_of(spec) if e["kind"] == "down"]
    assert len(down) == 1 and down[0]["speed"] in ("MEDIUM", "SLOW")
    res, _ = P.analyse(series(spec))
    assert P.verdict(res, 14)["label"] == "LIE"


def test_last_upright_is_an_upright_frame_not_a_sprawl():
    """stand -> crumple into a near-vertical-torso SITTING sprawl on the ground -> flat: 'last upright' must be a standing
    frame (before the crumple), not the sprawl frame, and ground contact comes after it and before/at flat."""
    spec = [(0, 3, "stand", 0)] + ramp(3, 3.4, "stand", 0, 10) + [(3.4, 4.0, "sit", 0)] + ramp(4.0, 4.3, "lie", 0, 90) + [(4.3, 9, "lie", 90)]
    res, _ = P.analyse(series(spec))
    e = [x for x in res[0].events if x["kind"] == "down"][0]
    rows = res[0].rows.set_index("t")
    assert str(rows.loc[e["last_upright"], "state"]) == P.UP
    assert e["last_upright"] < 3.5 and e["last_upright"] < e["contact_t"] <= e["flat_t"] + 1e-6
    assert abs(e["interval"] - (e["flat_t"] - e["last_upright"])) < 1e-6


def test_clock_is_labelled_simulated():
    import PIL.Image
    p = HERE / "out" / "skate_fall" / "clock_sim.jpg"
    assert p.exists() and PIL.Image.open(p).width <= 1280


def test_already_down_has_no_transition():
    down = [e for e in events_of([(0, 5, "lie", 90)]) if e["kind"] == "down"]
    assert len(down) == 1 and down[0]["last_upright"] is None and down[0]["speed"] == "NOT_OBSERVED"


def test_sit_is_not_alarm():
    spec = [(0, 2, "stand", 0), (2, 6, "sit", 0)]
    res, _ = P.analyse(series(spec))
    v = P.verdict(res, 6)
    assert v["label"] == "NO_ALARM"


def test_long_lie_clock_monotonic():
    last, ranks = -1, {"log": 0, "check": 1, "urgent": 2, "long-lie": 3}
    for s in (0, 5, 59, 60, 299, 300, 3599, 3600, 7200):
        c = P.clock_state(s)
        assert ranks[c["level"]] >= last
        last = ranks[c["level"]]
    assert P.clock_state(4000)["level"] == "long-lie" and P.clock_state(0)["next_rung"] == "CHECK NOW"


def test_outputs_exist_and_labels():
    rp = HERE / "results.json"
    assert rp.exists(), "run `python run.py --all` first"
    res = json.loads(rp.read_text(encoding="utf-8"))
    gt = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
    vids = res["videos"]
    for n in DEFAULT_SET:
        assert n in vids, f"{n} missing from results.json"
        v = vids[n]
        for fn in (v["files"]["hero"], v["files"]["report"]):
            assert (HERE / fn).exists(), fn
        assert v["label"] in ("FALL", "LIE", "NO_ALARM")
        assert v["correct"] == (v["label"] == gt[n]["label"])       # recorded verdict matches the recorded ground truth
    assert vids["classroom"]["falls_flagged"] == 0, "classroom (negative) must produce zero falls"
    sf = vids["skate_fall"]
    assert sf["label"] == "FALL" and abs(sf["event"]["first_down"] - gt["skate_fall"]["first_down_s"]) <= gt["skate_fall"]["tolerance_s"]
    ev = sf["event"]                                  # the three times are ordered and match the viewed ground truth
    assert ev["pre_state"] in ("UPRIGHT", "PARTIAL")  # 'last upright' is an on-feet frame, never a sprawled SITTING one
    assert ev["last_upright"] < ev["contact_t"] <= ev["first_down"] + 1e-6
    assert abs(ev["interval"] - (ev["flat_t"] - ev["last_upright"])) < 0.01
    assert abs(ev["last_upright"] - gt["skate_fall"]["last_upright_s"]) <= gt["skate_fall"]["last_upright_tolerance_s"]
    assert abs(ev["contact_t"] - gt["skate_fall"]["ground_contact_s"]) <= gt["skate_fall"]["ground_contact_tolerance_s"]
    assert 0.5 <= ev["interval"] <= 1.5 and ev["speed"] == "FAST"
    assert sf["headline"].count("00:") == 2 and f"{int(ev['contact_t']):02d}" in sf["headline"]
    k = res["kill_test"]
    assert k["pexels_total"] == 5 and k["passed"] == (k["pexels_correct"] >= 4 and k["classroom_falls"] == 0)


def test_hero_images():
    from PIL import Image
    for n in DEFAULT_SET:
        p = HERE / "out" / n / "hero.jpg"
        assert p.exists(), p
        im = Image.open(p)
        assert im.width <= 1280 and im.height >= 600
        a = np.asarray(im.convert("L"), float)
        assert a.std() > 20                                          # not a blank card


def test_vastdb_sql_subset():
    res = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    sql = next(iter(res["videos"].values()))["vastdb_sql"].upper()
    for banned in (" IN ", " ILIKE ", "GROUP BY", " JOIN ", "OVER ("):
        assert banned not in sql


if __name__ == "__main__":
    import time
    t0 = time.time()
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    bad = 0
    for k, fn in fns:
        try:
            fn()
            print(f"PASS {k}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL {k}: {type(e).__name__}: {e}")
    print(f"{len(fns) - bad}/{len(fns)} passed in {time.time() - t0:.1f}s")
    sys.exit(1 if bad else 0)
