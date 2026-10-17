"""Fast invariants on the produced outputs (run `python run.py --all` first). < 60 s."""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import core  # noqa: E402
import run  # noqa: E402

R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
V = R["videos"]
DASH = ["dashcam_urban", "sim_cutin"]
FIXED = ["motorway_a40", "person_bicycle_car_detection"]


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))


def test_all_videos_present_with_hero_and_report():
    for v in DASH + FIXED + ["ctrl_highway"]:
        assert v in V, v
        h = HERE / "out" / v / "hero.jpg"
        assert h.exists() and Image.open(h).size[0] <= 1280
        assert (HERE / "out" / v / "report.html").exists()


def test_dashcam_packets_complete():
    for v in DASH:
        r = V[v]
        assert r["packets_generated"] and r["camera"]["mode"] == "dashcam"
        assert 1 <= len(r["candidates"]) <= 5
        for n, c in enumerate(r["candidates"], 1):
            d = HERE / "out" / v / f"cand_{n}"
            for f in ("clip.mp4", "key.jpg", "crop.jpg", "storyboard.jpg", "packet.json"):
                assert (d / f).exists(), (v, n, f)
            assert abs(dur(d / "clip.mp4") - 10.0) < 0.3
            p = json.loads((d / "packet.json").read_text(encoding="utf-8"))
            for k in ("clock", "incident", "vehicle", "statement_draft", "evidence"):
                assert k in p
            assert "NOT extracted" in p["vehicle"]["registration"]       # no plate-OCR claim
            assert len(p["evidence"]["source_sha256"]) == 64
            a, b = c["window"]
            assert 0 <= a < b <= r["duration_s"] + 0.1 and abs((b - a) - 10) < 0.2
        sev = [c["sev_final"] for c in r["candidates"]]
        assert sev == sorted(sev, reverse=True)
        ts = sorted(c["t"] for c in r["candidates"])
        assert all(y - x >= 8 - 1e-6 for x, y in zip(ts, ts[1:]))        # clips barely overlap


def test_fixed_cameras_are_refused():
    for v in FIXED:
        assert V[v]["camera"]["mode"] == "fixed" and not V[v]["packets_generated"]
        assert not list((HERE / "out" / v).glob("cand_*/packet.json"))


def test_clock_arithmetic():
    c = run.clock(dt.date(2026, 10, 10), dt.date(2026, 10, 17))
    assert c["deadline"] == "2026-10-24" and c["days_left"] == 7 and c["state"] == "soon"
    c = run.clock(dt.date(2026, 9, 28), dt.date(2026, 10, 17))
    assert c["days_left"] == -5 and c["state"] == "passed" and "PASSED" in c["text"]
    for v in DASH:
        ck = V[v]["clock"]
        assert dt.date.fromisoformat(ck["deadline"]) - dt.date.fromisoformat(ck["recorded"]) == dt.timedelta(days=14)


def test_geometry_prefers_close_in_path_approach():
    def row(tid, Z, X, g, bh):
        return {"tid": tid, "name": "car", "t": 1.0, "f": 1, "s": bh, "g": g, "Z": Z, "X": X,
                "clear": abs(X) - 0.9 - 0.9, "lat": 0.0, "toward": 0.0, "ratio": np.nan, "ve": 0.0, "rg": 0.0,
                "ratio2": np.nan, "bh": bh}
    f = core.score(pd.DataFrame([row(1, 3.5, 0.2, 1.0, 190), row(2, 25, 0.2, 1.0, 40), row(3, 3.5, 6.0, 1.0, 190)]))
    s = dict(zip(f["tid"], f["sev"]))
    assert s[1] > 0.3 and s[1] > 5 * s[2] and s[1] > 5 * s[3]


def test_simulated_cutin_found_and_control_quiet():
    c = V["sim_cutin"]["candidates"][0]
    assert c["kind"] == "cut-in / crossing" and 8.0 <= c["t"] <= 11.5          # injected 8.0 s, SIMULATED
    assert V["ctrl_highway"]["candidates"] == []


def test_kill_test_dashcam_urban():
    e = R["eyeball"]["dashcam_urban"]
    assert e["reportable_in_top3"] >= 1 and e["near_field_precision_at_5"] == 1.0


def test_verdict_never_overclaims():
    v = lambda p, c: run.verdict(p, c)["code"]      # noqa: E731
    assert v(False, [{"sev_final": 0.99}]) == "not_dashcam"
    assert v(True, []) == "none"
    assert v(True, [{"sev_final": 0.99, "moving_ev": 1.0, "cosmos_parked": False}]) == "review"   # no heuristic earns SNAP-READY
    for name, code in (("ctrl_highway", "none"), ("dashcam_urban", "review"), ("sim_cutin", "review"),
                       ("motorway_a40", "not_dashcam"), ("person_bicycle_car_detection", "not_dashcam")):
        assert V[name]["verdict"]["code"] == code, name


def test_hero_clock_bar_grey_unless_packet_and_red_when_passed():
    def px(name):
        im = Image.open(HERE / "out" / name / "hero.jpg").convert("RGB")
        return np.array(im.getpixel((60, 652)), dtype=int)
    for n in FIXED + ["ctrl_highway"]:
        p = px(n)
        assert p.max() - p.min() < 40, (n, p)                          # clock not running: grey, not green or red
    for n in DASH:
        if V[n]["clock"]["state"] == "passed":
            p = px(n)
            assert p[0] > 180 and p[1] < 110, (n, p)                  # whole window gone: red


def test_overlay_date_parsing_and_grounding():
    today = dt.date(2026, 10, 17)
    assert run.parse_overlay("2013/06/18 19:41:50", today) == dt.datetime(2013, 6, 18, 19, 41, 50)
    assert run.parse_overlay("2013/06/18", today).date() == dt.date(2013, 6, 18)
    assert run.parse_overlay("UNREADABLE", today) is None
    assert run.parse_overlay("2031/01/02 10:00:00", today) is None      # future date rejected
    assert run.parse_overlay("2013/13/45", today) is None               # impossible date rejected
    ck = V["dashcam_urban"]["clock"]
    if ck["overlay_read"]:                                              # VLM was up: clock must be grounded in the footage
        assert ck["recorded"] == ck["overlay_read"]["date"] == "2013-06-18" and ck["state"] == "passed"
    else:
        assert ck["recorded_source"].startswith("assumed")
