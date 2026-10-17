"""Fast invariants (<60 s) on the produced outputs plus a no-server alignment check. Run: python test_smoke.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align as al  # noqa: E402
import make_tests as mt  # noqa: E402
import uncut as U  # noqa: E402
from uncut import pk  # noqa: E402

RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
CLIPS = RES["clips"]
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


t0 = time.time()
# 1. archive hygiene
check("sit_down and skate_fall are NOT in the archive", not ({"sit_down", "skate_fall"} & set(U.ARCHIVE)))
check("lisbon hold-out declared", "lisbon_parking" in U.HOLDOUT)
check("archive size recorded", RES["summary"]["archive"]["segments"] > 400, str(RES["summary"]["archive"]["segments"]))
# 2. verdicts + offsets on the 5 kill clips
for k in mt.KILL:
    c = CLIPS[k]
    check(f"{k}: verdict {c['verdict']} == truth {c['ground_truth']['verdict']}", c["verdict"] == c["ground_truth"]["verdict"])
    if c["ground_truth"].get("source"):
        e = c["eval"]
        check(f"{k}: source right and offsets within 1 s", bool(e.get("offset_ok")),
              f"start {e.get('start_err', -1):.2f}s end {e.get('end_err', -1):.2f}s")
check("kill test passed flag", RES["summary"]["kill_test"]["passed"])
# 3. specific measured invariants
t1, t2, t3, t4 = (CLIPS[k]["best"] for k in mt.KILL[:4])
check("t1 detected as 80% crop, not mirrored", abs(t1["crop"] - 0.8) < 0.01 and not t1["mirrored"])
check("t2 detected as mirrored, uncropped", t2["mirrored"] and t2["crop"] == 1.0)
check("t3 speed ~1.25 and one removed gap of ~3 s at clip 6 s",
      abs(t3["speed"] - 1.25) < 0.08 and len(t3["removed"]) == 1 and abs(t3["removed"][0]["seconds"] - 3.0) < 0.5
      and abs(t3["removed"][0]["at_q"] - 6.0) < 0.5 and abs(t3["removed"][0]["at_src"] - 17.5) < 0.5,
      f"speed {t3['speed']:.2f} removed {[(round(x['seconds'], 2), round(x['at_q'], 1), round(x['at_src'], 1)) for x in t3['removed']]}")
check("removed seconds are shown to the nearest 0.5 s (3.0, not 3.1)", t3["removed"][0]["seconds_r"] == 3.0)
check("t4 has one ~5 s foreign span at clip time 5-10 s",
      len(t4["foreign"]) == 1 and abs(t4["foreign"][0]["q_start"] - 5.0) < 0.6 and abs(t4["foreign"][0]["q_end"] - 10.0) < 0.6)
check("t4 splice is PARTIAL, never a green MATCH", CLIPS["t4_splice_ptz_sitdown"]["verdict"] == "PARTIAL" and
      "NOT from our archive" in CLIPS["t4_splice_ptz_sitdown"]["headline"])
check("t5 (skate_fall) has no best alignment", CLIPS["t5_skate_fall"]["best"] is None)
check("same-camera hold-out is not matched", CLIPS["x6_lisbon_holdout"]["verdict"] == "NOT IN ARCHIVE")
t6 = CLIPS[mt.HERO]["best"]
check("t6 hero clip: MATCH with crop 80% AND one ~3 s cut at clip 7 s",
      CLIPS[mt.HERO]["verdict"] == "MATCH" and abs(t6["crop"] - 0.8) < 0.01 and len(t6["removed"]) == 1 and
      abs(t6["removed"][0]["seconds"] - 3.0) < 0.5 and abs(t6["removed"][0]["at_q"] - 7.0) < 0.6,
      f"{[(round(x['seconds'], 2), round(x['at_q'], 1)) for x in t6['removed']]}")
check("headline uses clip time first, archive time in parentheses", "at clip 00:07 (archive" in CLIPS[mt.HERO]["headline"],
      CLIPS[mt.HERO]["headline"])
# off-centre / portrait / letterbox probes: the outcomes are recorded as measured (x15 is a documented miss)
check("x9 off-centre 70% crop found with the right variant", CLIPS["x9_dashcam_offcentre"]["best"]["variant"] == "0.7R" and
      CLIPS["x9_dashcam_offcentre"]["verdict"] == "MATCH")
check("x13 9:16 re-crop with caption found as p916R", CLIPS["x13_dashcam_916_caption"]["best"]["variant"] == "p916R")
check("x14 letterboxed clip found with the bars ignored", CLIPS["x14_retail_tiktok_bars"]["verdict"] == "MATCH" and
      bool(CLIPS["x14_retail_tiktok_bars"]["bars_stripped"]))
check("x15 off-grid 9:16 crop is still a documented miss (NOT IN ARCHIVE)", CLIPS["x15_yonge_916_offgrid"]["verdict"] == "NOT IN ARCHIVE")
# 4. artefacts
for k in CLIPS:
    h = HERE / "out" / k / "hero.jpg"
    r = HERE / "out" / k / "report.html"
    ok = h.exists() and r.exists()
    if ok:
        import cv2
        w = cv2.imread(str(h)).shape[1]
        ok = w <= 1280 and "Footage:" in r.read_text(encoding="utf-8")
    check(f"{k}: hero.jpg <=1280 px and report.html with credits", ok)
# 4b. the stamp colour on the t4 hero is amber/orange, not green (pixel check of the stamp area)
import cv2  # noqa: E402
for k, want in (("t4_splice_ptz_sitdown", "orange"), ("t1_lisbon_crop80", "green")):
    im = cv2.imread(str(HERE / "out" / k / "hero.jpg"))[0:140, 880:1280].reshape(-1, 3).astype(int)    # BGR stamp area
    near = lambda rgb: int((np.abs(im - np.array(rgb[::-1])).sum(1) < 60).sum())  # noqa: E731
    o, g = near((240, 120, 40)), near((46, 170, 98))
    check(f"{k} hero stamp is {want}", (o > 300 and o > 4 * g) if want == "orange" else (g > 300 and g > 4 * o), f"orange px {o}, green px {g}")
# 4c. verdict rules (pure function) incl. the AMBIGUOUS paths that no test clip triggers
import verify as V  # noqa: E402
check("classify: full coverage -> MATCH", V.classify(1.0, 15, 0.0, 0.1, 0.2, "x", 0.1, 0.9)[0] == "MATCH")
check("classify: 50% + 5 s foreign -> PARTIAL", V.classify(0.5, 10, 5.0, 0.1, 0.2, "x", 0.1, 0.9)[0] == "PARTIAL")
check("classify: 85% explained -> PARTIAL", V.classify(0.85, 20, 0.0, 0.1, 0.2, "x", 0.1, 0.9)[0] == "PARTIAL")
check("classify: strong alternative offset -> AMBIGUOUS", V.classify(1.0, 15, 0.0, 0.8, 0.2, "x", 0.1, 0.9)[0] == "AMBIGUOUS")
check("classify: near-equal runner-up camera -> AMBIGUOUS", V.classify(1.0, 15, 0.0, 0.1, 0.9, "retail_cctv", 0.8, 0.9)[0] == "AMBIGUOUS")
check("classify: 20% aligned -> AMBIGUOUS (UNPROVEN on real footage)", V.classify(0.2, 15, 12.0, 0.1, 0.2, "x", 0.1, 0.9)[0] == "AMBIGUOUS")
check("classify: 5% aligned -> NOT IN ARCHIVE", V.classify(0.05, 15, 14.0, 0.1, 0.2, "x", 0.1, 0.9)[0] == "NOT IN ARCHIVE")
# 4d. bars: a black letterbox with a caption in it is detected; a plain dark noisy scene is not
rng0 = np.random.default_rng(1)
lb = rng0.integers(60, 200, (10, 200, 100)).astype(np.uint8)
lb[:, :60], lb[:, 140:] = 0, 0
lb[:, 20:30, 30:70] = 255                                    # caption inside the top bar
lb2, info = U.strip_bars(lb)
check("strip_bars removes a captioned letterbox (60 px top + bottom)", info is not None and lb2.shape[1] == 80,
      f"{lb2.shape}")
night = rng0.integers(0, 60, (10, 200, 100)).astype(np.uint8)
check("strip_bars leaves a dark noisy scene alone", U.strip_bars(night)[1] is None)
# 5. no-server alignment check on cached archive hashes (cropped + noisy copy of retail_cctv 80-95 s)
z = np.load(U.WORK / "hash" / "retail_cctv.npz")
A = z["1.0"]
src = A[400:475].copy()                               # 15 s at 5 fps
rng = np.random.default_rng(0)
noisy = np.where(rng.random(src.shape) < 0.03, 1 - src, src).astype(np.uint8)
tq = np.arange(len(noisy)) / U.HFPS
res = al.align_one(noisy, tq, A, np.ones(len(A), bool), np.ones(len(A), bool))
d = al.describe(res, tq, 15.0)
check("synthetic exact-offset alignment (3% bit noise)", d["segments"] and abs(d["segments"][0]["src_start"] - 80.0) < 0.3 and
      d["coverage"] > 0.95, f"src_start {d['segments'][0]['src_start']:.2f}" if d["segments"] else "none")
# synthetic cut: drop frames 30..45 (3 s) and re-run
cut = np.concatenate([src[:30], src[45:]])
tq2 = np.arange(len(cut)) / U.HFPS
res2 = al.align_one(cut, tq2, A, np.ones(len(A), bool), np.ones(len(A), bool))
d2 = al.describe(res2, tq2, len(cut) / U.HFPS)
check("synthetic 3 s cut recovered", len(d2["removed"]) == 1 and abs(d2["removed"][0]["seconds"] - 3.0) < 0.5,
      str([round(x["seconds"], 2) for x in d2["removed"]]))
# random noise must not align
rnd = (rng.random((50, U.NB)) < 0.5).astype(np.uint8)
res3 = al.align_one(rnd, np.arange(50) / U.HFPS, A, np.ones(len(A), bool), np.ones(len(A), bool))
check("random hashes align to nothing", not res3["segments"])
dt = time.time() - t0
check(f"runtime {dt:.0f}s < 60 s", dt < 60)
print(f"\n{len(fails)} failed" if fails else "\nall smoke checks passed")
sys.exit(1 if fails else 0)
