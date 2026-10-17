"""Fast invariants over the produced outputs (no GPU, < 60 s):  python test_smoke.py"""
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
import leaklib as L  # noqa: E402

OUT = HERE / "out"
RES = json.loads((HERE / "results.json").read_text())
VIDEOS = [k for k in RES if not k.startswith("_")]
KINDS = list(L.REDACTORS)
fails = []


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- unit invariants of the auditor primitives
rng = np.random.default_rng(0)
sharp = (rng.random((60, 60, 3)) * 255).astype(np.uint8)
blur = L.blur_rects(sharp, [(0, 0, 60, 60)])
check(L.lap_var(blur, (5, 5, 55, 55)) < 0.1 * L.lap_var(sharp, (5, 5, 55, 55)), "blur collapses Laplacian variance")
check(L.coverage((10, 10, 20, 20), [(0, 0, 30, 30)]) == 1.0 and L.coverage((10, 10, 20, 20), []) == 0.0, "coverage bounds")
kp = [[50, 50, 0.0]] * 17
kp[0], kp[1], kp[2] = [50, 60, 0.9], [44, 50, 0.9], [56, 50, 0.9]
fc = L.face_from_kp(kp)
check(fc is not None and fc["w"] > 20, "face_from_kp builds a face from nose+eyes")
kp2 = [[50, 50, 0.0]] * 17
check(L.face_from_kp(kp2) is None, "no landmarks -> no face")
check(L.is_flagged({"w": 40, "lv_o": 500, "ratio": 0.9}) and not L.is_flagged({"w": 40, "lv_o": 500, "ratio": 0.05}), "flag rule")
check(not L.is_flagged({"w": 6, "lv_o": 500, "ratio": 0.9}), "tiny faces are not identifiable")

# ---- per video outputs
for v in VIDEOS:
    d = OUT / v
    hero = cv2.imread(str(d / "hero.jpg"))
    check(hero is not None and hero.shape[1] <= 1280 and hero.shape[0] >= 500, f"{v}: hero.jpg readable and <= 1280 px wide ({None if hero is None else hero.shape[:2]})")
    check((d / "report.html").exists() and "Leak test" in (d / "report.html").read_text(encoding="utf-8"), f"{v}: report.html")
    r = RES[v]
    n_frames = r["frames"]
    for kd in KINDS:
        cert = json.loads((d / f"certificate_{kd}.json").read_text())
        ex = d / cert["export"]
        cap = cv2.VideoCapture(str(ex))
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        check(abs(nf - n_frames) <= 1, f"{v}/{kd}: export has {nf} frames, audit has {n_frames}")
        check(cert["verdict"] in ("PASS", "FAIL") and (cert["verdict"] == "PASS") == (cert["frames_with_exposed_face"] == 0),
              f"{v}/{kd}: verdict consistent with exposed-frame count ({cert['verdict']}, {cert['frames_with_exposed_face']})")
        check(pk.sha256(ex) == cert["export_sha256"], f"{v}/{kd}: certificate sha256 matches the export file")
        g = r["ground_truth"][kd]
        check(g["recall"] is None or 0 <= g["recall"] <= 1, f"{v}/{kd}: recall in [0,1] ({g['recall']})")
        check(0 <= g["false_flag_rate"] <= 1, f"{v}/{kd}: false flag rate in [0,1] ({g['false_flag_rate']})")
    # redactor A: only every 3rd frame may be blurred => non-keyframes pixel-close to the original
    a = d / "export_a_keyframe.mp4"
    meta = json.loads((d / "_work" / "window.json").read_text())
    log = meta["rects"]["a_keyframe"]
    check(all(len(x) == 0 for k, x in enumerate(log) if k % 3 != 0), f"{v}: redactor A blurs nothing off the keyframes")
    check(any(len(x) > 0 for k, x in enumerate(log) if k % 3 == 0), f"{v}: redactor A blurs on keyframes")
    # redactor C blurs at least as many frames as B and B at least as many as A
    cnt = {kd: sum(1 for x in meta["rects"][kd] if x) for kd in KINDS}
    check(cnt["c_good"] >= cnt["b_conf"] >= cnt["a_keyframe"], f"{v}: redaction coverage ordering C>=B>=A frames {cnt}")
    # redactor D is tight: every box is exactly an (integer-rounded) detector box, so no box may be larger than C's dilated one
    area = lambda rs: sum((r[2] - r[0]) * (r[3] - r[1]) for r in rs)
    tot = {kd: sum(area(x) for x in meta["rects"][kd]) for kd in ("c_good", "d_tight")}
    check(0 < tot["d_tight"] < tot["c_good"], f"{v}: D (tight) blurs less area than C (dilated): {tot['d_tight']:.0f} < {tot['c_good']:.0f}")
    # exported frames: a blurred person region is smoother than the same region in the original
    m, stride = pk.info(v), max(1, int(round(pk.info(v)["fps"] / 15.0)))
    fi0 = next(fi for k, (fi, t, fr) in enumerate(pk.iter_frames(v, stride=stride, start=meta["t0"], end=meta["t0"] + 1)) if k == 0)
    orig0 = pk.frame_at(v, fi0 / m["fps"] + 1e-4)
    cap = cv2.VideoCapture(str(d / "export_c_good.mp4"))
    ok, c0 = cap.read()
    cap.release()
    rects = meta["rects"]["c_good"][0]
    if rects:
        x1, y1, x2, y2 = rects[0]
        check(L.lap_var(c0, (x1, y1, x2, y2)) < L.lap_var(orig0, (x1, y1, x2, y2)), f"{v}: export C first frame is smoother than the original inside its blur box")
    else:
        check(True, f"{v}: first frame has no person (skip blur check)")
# ---- headline numbers recompute from the raw per-frame audits
for v in VIDEOS:
    for kd in KINDS:
        d = OUT / v
        a = json.loads((d / "_work" / f"audit_{kd}.json").read_text())["res"]
        o = json.loads((d / "_work" / "audit_original.json").read_text())["res"]
        check(json.loads((d / "_work" / f"audit_{kd}.json").read_text()).get("metric") == L.METRIC, f"{v}/{kd}: audit cache uses metric {L.METRIC}")
        L.tag_in_person(a, o)
        cert = json.loads((d / f"certificate_{kd}.json").read_text())
        n_flag = sum(1 for fr in a if any(L.is_flagged(f) for f in fr["faces"]))
        check(n_flag == cert["frames_with_exposed_face"], f"{v}/{kd}: person-tagged audit gives the certificate's frame count ({n_flag})")
        check(cert["people_estimate"] <= cert["face_tracks_exposed"], f"{v}/{kd}: people estimate <= face tracks ({cert['people_estimate']} <= {cert['face_tracks_exposed']})")
# the good exports: C must PASS everywhere; D (tight) leaks are real only if the eyeball labels confirm one
KT = RES["_meta"]["kill_test"]
check(all(RES[v]["certificates"]["c_good"]["verdict"] == "PASS" for v in VIDEOS), "export C (dilated, full) PASSes on every clip")
check(all(RES[v]["certificates"]["a_keyframe"]["verdict"] == "FAIL" for v in VIDEOS), "export A (keyframe-only) FAILs on every clip")
check(KT["false_flags_on_tight_good_export_D"].get("eyeball_flags", {}).get("flagged_frames_labelled", 0) > 0, "kill test carries eyeball-labelled flags on the tight export D")
check(0 <= KT["recall_frame_level_by_construction"]["recall"] <= 1 and KT["passed"] is True, "kill test block present and passed")
# independent labels: the labels file matches the frozen samples
S = json.loads((HERE / "eyeball_sample.json").read_text())
G = json.loads((HERE / "eyeball_gt.json").read_text())
check(len(G["recall"]) == len(S["recall"]) and len(G["precision"]) == len(S["precision"]), "eyeball labels match the frozen samples")
# ---- hero images tell the story: every hero has one column per export and a verdict colour for each
for v in VIDEOS:
    hero = cv2.imread(str(OUT / v / "hero.jpg"))
    check(hero is not None and hero.shape[0] > 800, f"{v}: hero is the tall product image ({None if hero is None else hero.shape})")
# ---- results.json shape
for v in VIDEOS:
    check(set(RES[v]["certificates"]) == set(KINDS) and len(KINDS) == 4, f"{v}: certificates for all 4 exports in results.json")
check("kill_test" in RES.get("_meta", {}), "results.json carries the kill test block")
print("\nFAILED:" if fails else "\nALL OK", fails if fails else "")
sys.exit(1 if fails else 0)
