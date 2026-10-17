"""Fast smoke checks (< 60 s, no GPU, no network) of the NIGHT EXIT invariants.

Part 1: unit checks on synthetic tracks (crossing direction, debounce, stitching, query parsing, colour naming, ranking).
Part 2: invariants on the outputs produced by `python run.py --all` (results.json, hero.jpg, report.html, eyeball match).
Part 3 (default on, --no-recompute to skip): re-runs run.py on one_by_one_person_detection (cached tracks/poses, no Cosmos) into
a temp folder inside this folder and checks the recomputed crossings, so the pipeline code is exercised, not only the saved files.
Run: .venv/Scripts/python.exe prototypes/04_night_exit/test_smoke.py [--no-recompute]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nightexit as ne  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def track(tid: int, pts: list[tuple[float, float, float]], bh: float = 100.0) -> pd.DataFrame:
    """rows for one person walking through (t, cx, y2) samples."""
    rows = []
    for k, (t, cx, y2) in enumerate(pts):
        rows.append({"f": int(t * 10), "t": t, "tid": tid, "pid": tid, "cls": 0, "name": "person", "conf": 0.9,
                     "x1": cx - 20, "y1": y2 - bh, "x2": cx + 20, "y2": y2, "cx": cx, "cy": y2 - bh / 2, "bw": 40.0, "bh": bh})
    return pd.DataFrame(rows)


LINE = ne.Line("Door", (100, 0), (100, 400), (0, 200), "foot")      # inside = left of x=100


def unit_checks() -> None:
    out = track(1, [(t, 20 + 20 * t, 300) for t in range(0, 10)])               # walks left -> right: OUT
    ev = ne.find_crossings(out, [LINE])
    check(len(ev) == 1 and ev.dir.iloc[0] == "OUT" and 3.5 < ev.t.iloc[0] < 4.5, "walking out across the line is one OUT at ~4 s")
    back = track(2, [(t, 220 - 20 * t, 300) for t in range(0, 10)])             # right -> left: IN
    ev = ne.find_crossings(back, [LINE])
    check(len(ev) == 1 and ev.dir.iloc[0] == "IN", "walking in is one IN")
    # flicker: steps over by 8 px for 0.2 s and back (must be ignored), then really leaves
    fl = track(3, [(0, 60, 300), (1, 90, 300), (1.1, 112, 300), (1.2, 112, 300), (1.3, 90, 300), (2, 60, 300), (3, 90, 300),
                   (4, 130, 300), (5, 170, 300)])
    ev = ne.find_crossings(fl, [LINE])
    check(len(ev) == 1 and ev.dir.iloc[0] == "OUT", "0.2 s flicker over the line is debounced, the real exit is kept")
    # segment extent: walking past the end of a short line is not a crossing
    short = ne.Line("short", (100, 0), (100, 50), (0, 25), "foot")
    check(ne.find_crossings(track(4, [(t, 20 + 20 * t, 300) for t in range(0, 10)]), [short]).empty, "crossing outside the segment is ignored")
    # blind span suppresses the crossing
    ev = ne.find_crossings(out, [LINE], blind=[(3.0, 5.0)])
    check(ev.empty, "crossings inside a camera-blind span are suppressed")
    # first/last stretch shorter than dwell is kept (person leaves frame right after the line)
    ex = track(5, [(0, 60, 300), (1, 80, 300), (2, 125, 300), (2.1, 150, 300)])
    check(len(ne.find_crossings(ex, [LINE])) == 1, "an exit right before the track ends is kept")
    # stitching: A ends, B starts 1.5 s later where A would be
    a = track(10, [(t, 20 + 20 * t, 300) for t in np.arange(0, 3, 0.2)])
    b = track(11, [(t, 20 + 20 * t, 300) for t in np.arange(4.5, 7, 0.2)])
    r = ne.stitch(pd.concat([a, b]).drop(columns="pid"), 5.0, 3.0)
    check(r.pid.nunique() == 1, "constant-velocity fragments with a 1.5 s gap are stitched into one person")
    far = track(12, [(t, 600 - 10 * t, 100) for t in np.arange(4.5, 7, 0.2)])
    r = ne.stitch(pd.concat([a, far]).drop(columns="pid"), 5.0, 3.0)
    check(r.pid.nunique() == 2, "a fragment that is nowhere near the prediction is not stitched")
    ev = ne.find_crossings(ne.stitch(pd.concat([a, b]).drop(columns="pid"), 5.0, 3.0), [LINE])
    check(len(ev) == 1 and bool(ev.inferred.iloc[0]), "a crossing inside a detection gap is reported as interpolated")


def appearance_checks() -> None:
    def hist_of(bgr: tuple[int, int, int]) -> dict[str, float]:
        return ne.colour_hist(np.tile(np.array(bgr, np.uint8), (400, 1)))
    check(next(iter(hist_of((128, 128, 128)))) == "grey", "grey pixels are named grey")
    check(next(iter(hist_of((20, 20, 22)))) == "black", "near-black pixels are named black")
    check(next(iter(hist_of((235, 238, 240)))) == "white", "bright desaturated pixels are named white")
    check(next(iter(hist_of((200, 60, 40)))) == "blue", "BGR (200,60,40) is blue")
    check(next(iter(hist_of((40, 40, 200)))) == "red", "BGR (40,40,200) is red")
    q = ne.parse_query("light grey shirt and blue jeans")
    check(q["top"] == ["grey"] and q["legs"] == ["blue"], "query 'light grey shirt and blue jeans' -> top grey, legs blue")
    check(ne.parse_query("grey top")["top"] == ["grey"], "query 'grey top' -> top grey")
    check(ne.parse_query("blonde woman in a black cardigan")["top"] == ["black"], "query with a cardigan binds black to the top")
    check(ne.parse_query("dark trousers")["legs"] == ["black"], "'dark trousers' -> legs black")
    check(ne.parse_clock("00:12") == 12 and ne.parse_clock("01:00:30") == 3630 and ne.parse_clock("45") == 45, "clock strings parse")
    people = {1: {"top": {"black": 0.9}, "legs": {"black": 0.8}, "desc": ""},
              2: {"top": {"grey": 0.8, "black": 0.1}, "legs": {"blue": 0.7}, "desc": ""},
              3: {"top": {"red": 0.9}, "legs": {"black": 0.9}, "desc": ""}}
    rk = ne.rank_people(people, "grey top", mode="colour")
    check([x["pid"] for x in rk][0] == 2 and rk[0]["score"] > rk[1]["score"], "colour ranking puts the grey top first")
    rk = ne.rank_people(people, "grey shirt and blue jeans", mode="colour")
    check(rk[0]["pid"] == 2 and rk[0]["score"] > 0.9, "grey top + blue legs scores ~1 for the matching person")


def output_checks() -> None:
    rj = HERE / "results.json"
    if not rj.exists():
        check(False, "results.json exists (run python run.py --all first)")
        return
    res = json.loads(rj.read_text(encoding="utf-8"))
    check(res.get("clock_note") == "video time is the clock", "results.json states that video time is the clock")
    for v in ne.DEFAULT_SET:
        vr = res["videos"].get(v)
        check(vr is not None, f"{v}: present in results.json")
        if vr is None:
            continue
        hero = HERE / vr["hero"]
        im = cv2.imread(str(hero))
        check(im is not None and im.shape[1] <= 1280 and im.shape[0] > 400 and float(im.std()) > 20, f"{v}: hero.jpg readable, <= 1280 px wide, not blank")
        check((HERE / vr["report"]).exists() and "Video prototype" in (HERE / vr["report"]).read_text(encoding="utf-8"),
              f"{v}: report.html written with credits footer")
        ts = [c["t"] for c in vr["crossings"]]
        check(ts == sorted(ts), f"{v}: crossings are in time order")
        check(all(0 <= t <= vr["duration_s"] for t in ts), f"{v}: crossing times lie inside the clip")
        check(all(c["dir"] in ("IN", "OUT") for c in vr["crossings"]), f"{v}: every crossing has a direction")
        for c in vr["demo_query"]["cards"]:
            check(c["primary"]["t"] >= ne.parse_clock(vr["demo_query"]["since"]) - 1e-6, f"{v}: card person {c['person']} crossed after T")
        scores = [c["score"] for c in vr["demo_query"]["cards"] if c["score"] is not None]
        check(scores == sorted(scores, reverse=True), f"{v}: cards are sorted best match first")
        if "vs_eyeball" in vr:
            e = vr["vs_eyeball"]
            check(e["direction_accuracy"] in (None, 1.0) or e["direction_accuracy"] >= 0.9, f"{v}: direction accuracy on matched eyeballed crossings >= 0.9 ({e['direction_accuracy']})")
    for v in ne.DEFAULT_SET:                      # the default hero must tell the exit story: the best match WENT OUT
        vr = res["videos"].get(v)
        if vr and vr["demo_query"]["cards"]:
            top = vr["demo_query"]["cards"][0]
            check(top["primary"]["dir"] == "OUT", f"{v}: default hero headlines an OUT crossing (best match {top['primary']['dir']} {top['primary']['clock']})")
            check(top["abs_score"] is not None and top["close_matches"] is not None, f"{v}: top card carries an absolute evidence score and a close-match count")
    rq = [q for v in res["videos"].values() for q in v.get("ranking_queries", [])]
    gt_ = [q for q in rq if q["query"] == "grey top"]
    check(bool(gt_) and gt_[0]["combined"]["close_matches_within_10pts"] >= 3,
          "'grey top' (3 real grey tops) is flagged as >= 3 close matches, not as a decisive winner")
    ob = res["videos"].get("one_by_one_person_detection")
    if ob:
        outs = [c for c in ob["crossings"] if c["dir"] == "OUT"]
        ins = [c for c in ob["crossings"] if c["dir"] == "IN"]
        check(len(outs) == 6 and len(ins) == 6, "one_by_one: 6 people in, 6 people out (eyeballed truth)")
        check(ob["vs_eyeball"]["recall"] == 1.0 and ob["vs_eyeball"]["precision"] == 1.0, "one_by_one: all 12 eyeballed crossings found, none extra")
    ro = res["videos"].get("rochester_night")
    if ro:
        check(ro["camera_blind_spans"], "rochester: camera-motion guard found the pan")
        check(all(not any(a <= c["t"] <= b for a, b in ro["camera_blind_spans"]) for c in ro["crossings"]), "rochester: no crossing counted while the camera moves")
    kt = res.get("kill_test")
    check(kt is not None and kt["ranking_queries_preregistered"] >= 3, "kill test block present with >= 3 pre-registered ranking queries")
    if kt:
        check(kt["ranking_passed_now"] == kt["ranking_queries_preregistered"], "all pre-registered description queries rank the right person first (post-hoc fix noted in results.json)")
        check(kt["events_matched"] == kt["events_truth"] and kt["direction_correct"] == kt["events_matched"], "every eyeballed crossing found with the right direction")
        check("NOT clean evidence" in kt["post_hoc_fix"], "results.json discloses the post-hoc garment fix")
    for v in ne.DEFAULT_SET:
        vr = res["videos"].get(v)
        if vr:
            check(vr["vastdb_shortlist"]["all_crossings_inside"], f"{v}: every crossing after T lies in a VastDB-shortlisted 5 s segment")
            html_t = (HERE / vr["report"]).read_text(encoding="utf-8")
            import re as _re
            refs = _re.findall(r'src="([^"]+\.jpg)"', html_t)
            check(refs and all((HERE / "out" / v / r_).exists() for r_ in refs), f"{v}: every image referenced by report.html exists ({len(refs)})")


def recompute_checks() -> None:
    """Re-run the pipeline on one clip (cached tracks, no VLM) into a temp folder and compare with the eyeballed truth."""
    tmp = HERE / "_smoke_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    try:
        pr = subprocess.run([sys.executable, str(HERE / "run.py"), "--video", "one_by_one_person_detection", "--no-vlm",
                             "--out-dir", str(tmp), "--results", str(tmp / "results.json")],
                            capture_output=True, text=True, timeout=150)
        check(pr.returncode == 0, f"recompute: run.py exits 0 ({time.time() - t0:.0f} s)")
        if pr.returncode != 0:
            print(pr.stderr[-800:])
            return
        vr = json.loads((tmp / "results.json").read_text(encoding="utf-8"))["videos"]["one_by_one_person_detection"]
        cr = vr["crossings"]
        check(len(cr) == 12 and sum(c["dir"] == "OUT" for c in cr) == 6 and sum(c["dir"] == "IN" for c in cr) == 6,
              f"recompute: 12 crossings, 6 IN and 6 OUT ({len(cr)} found)")
        e = vr["vs_eyeball"]
        check(e["matched"] == 12 and e["direction_correct"] == 12 and e["extra"] == 0, f"recompute: all 12 eyeballed crossings matched with the right direction, {e['extra']} extra")
        saved = json.loads((HERE / "results.json").read_text(encoding="utf-8"))["videos"]["one_by_one_person_detection"]["crossings"]
        check([(c["dir"], round(c["t"], 1)) for c in cr] == [(c["dir"], round(c["t"], 1)) for c in saved],
              "recompute: crossings identical to the saved results.json")
        im = cv2.imread(str(tmp / "one_by_one_person_detection" / "hero.jpg"))
        check(im is not None and im.shape[1] <= 1280 and float(im.std()) > 20, "recompute: hero.jpg rendered")
        check(len(vr["demo_query"]["cards"]) == 4, "recompute: 4 people listed since 00:21")
    except subprocess.TimeoutExpired:
        check(False, "recompute: run.py finished within 150 s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unit_checks()
    appearance_checks()
    output_checks()
    if "--no-recompute" not in sys.argv:
        recompute_checks()
    print(f"\n{len(FAILS)} failure(s)")
    sys.exit(1 if FAILS else 0)
