"""Fast invariants for MIMIC (< 60 s; ~30 s of it is the two real-browser page checks): python test_smoke.py   (needs `python run.py --all` to have produced out/ and results.json)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import mimic_core as mc  # noqa: E402

pk = mc.pk
OUT = HERE / "out"
RES = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def jpg_width(p: Path) -> int:
    import cv2
    im = cv2.imread(str(p))
    return 0 if im is None else im.shape[1]


# ---------------------------------------------------------------------------------------------- outputs exist
for c in mc.DEFAULT_CLIPS:
    h, r = OUT / c / "hero.jpg", OUT / c / "report.html"
    check(f"outputs for {c}", h.exists() and r.exists() and 0 < jpg_width(h) <= 1280, f"hero width {jpg_width(h) if h.exists() else 0}")
check("global hero", (OUT / "hero.jpg").exists() and jpg_width(OUT / "hero.jpg") <= 1280)
page = OUT / "draw.html"
check("interactive page < 15 MB", page.exists() and page.stat().st_size < 15e6, f"{page.stat().st_size / 1e6:.1f} MB" if page.exists() else "missing")
check("page embeds the search code", page.exists() and "function dtwFlat" in page.read_text(encoding="utf-8")[:400000])

# ----------------------------------------------------------------------------- results.json is self-consistent
vids = RES["videos"]
check("results cover every default clip", set(vids) == set(mc.DEFAULT_CLIPS))
check("tracklet totals add up", sum(v["tracklets"] for v in vids.values()) == RES["totals"]["tracklets"])
ev = RES["eval"]
check("kill test recorded from hand labels", ev["kill_test"]["labelled"] and ev["kill_test"]["passed"] is True,
      f"{ev['kill_test']['presets_passing']}/4 presets, mean P@3 {ev['mean_p_at_3_4presets']}")
gt = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8"))
drift = []
for p, g in RES["global"].items():
    for i, row in enumerate(g["top8"][:3]):
        lab = gt["labels"].get(p, {}).get(row["key"])
        if lab is None:
            drift.append(f"{p}#{i + 1} unlabelled")
check("every top-3 result was hand-labelled", not drift, "; ".join(drift))
_d0 = mc.build_all()
_lis = _d0["df"][_d0["df"].video == "lisbon_parking"]
_search = int(((_lis["rel_ext"] >= mc.MIN_REL) & (_lis["ext"] >= mc.MIN_EXT)).sum())
check("lisbon_parking is a photo slideshow: thousands of cuts, no searchable trajectory", vids["lisbon_parking"]["camera_windows"] > 1000 and _search == 0,
      f"{vids['lisbon_parking']['camera_windows']} windows, {len(_lis)} jitter tracklets, {_search} searchable")
check("the retail/worker clips each keep >= 5 searchable tracklets", all(int((_d0["df"][_d0["df"].video == v]["rel_ext"] >= mc.MIN_REL).sum()) >= 5 for v in ("retail_cctv", "worker_zone_detection")))

# ----------------------------------------------------------------------------------- descriptor / matching maths
t = np.linspace(0, 1, 50)
line = np.stack([0.1 + 0.8 * t, np.full(50, 0.5)], 1)
d1 = mc.descriptor(line, 16 / 9)
check("descriptor shapes", d1["pos"].shape == (32, 2) and d1["shape"].shape == (32, 2) and d1["turn"].shape == (32,) and d1["speed"].shape == (32,))
check("shape is normalised (starts at origin, unit length)", np.allclose(d1["shape"][0], 0) and abs(np.linalg.norm(np.diff(d1["shape"], axis=0), axis=1).sum() - 1) < 1e-6)
a = d1["shape"][None]
check("DTW(x, x) = 0", abs(mc.dtw_batch(d1["shape"], a)[0]) < 1e-9)
check("DTW is symmetric in cost scale", abs(mc.dtw_batch(d1["shape"], np.stack([d1["shape"][::-1]]))[0] - mc.dtw_batch(d1["shape"][::-1], a)[0]) < 1e-9)

# a synthetic table of five motions: the right preset must find the right motion
def tl(xy, name):
    n = len(xy)
    return mc.Tracklet(name, "synthetic", 1, "car", 0, np.arange(n) / 15.0, np.asarray(xy, float), np.full((n, 2), 0.1), 1280, 720)

th = np.linspace(-np.pi / 2, np.pi / 2, 30)
uturn = np.vstack([np.stack([np.linspace(0.2, 0.6, 30), np.full(30, 0.4)], 1), np.stack([0.6 + 0.1 * np.cos(th) * 0.56, 0.5 + 0.1 * np.sin(th)], 1),
                   np.stack([np.linspace(0.6, 0.2, 30), np.full(30, 0.6)], 1)])
rl = line[::-1]
circle = np.stack([0.5 + 0.1 * np.cos(np.linspace(0, 2 * np.pi, 60)) * 0.56, 0.5 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, 60))], 1)
scribble = np.stack([0.5 + 0.03 * np.cos(np.linspace(0, 12, 80)), 0.5 + 0.04 * np.sin(np.linspace(0, 9, 80))], 1)
diag = np.stack([np.linspace(0.1, 0.9, 50), np.linspace(0.15, 0.85, 50)], 1)
tls = [tl(line, "L2R"), tl(rl, "R2L"), tl(uturn, "UTURN"), tl(circle, "CIRCLE"), tl(scribble, "SCRIBBLE"), tl(diag, "DIAG")]
df = mc.table_from(tls, {"synthetic": 1280 / 720}, {"synthetic": 15.0})
arr = mc.tl_arrays(df)
best = {p: mc.match(mc.preset_query(p), df, arr, k=2).key.iloc[0] for p in ("u_turn", "left_to_right", "diagonal", "loiter")}
check("synthetic: u_turn finds the U-turn", best["u_turn"] == "UTURN", best["u_turn"])
check("synthetic: left_to_right finds L2R, not R2L (direction aware)", best["left_to_right"] == "L2R", best["left_to_right"])
check("synthetic: diagonal finds the diagonal", best["diagonal"] == "DIAG", best["diagonal"])
check("synthetic: loiter finds the small scribble", best["loiter"] in ("SCRIBBLE", "CIRCLE"), best["loiter"])
q = mc.preset_query("left_to_right")
q2 = mc.Query(q.pts + [0.0, -0.2], q.aspect, 0, 0, False, None, 0, "moved")
check("position-free query is translation invariant", mc.match(q, df, arr, k=1).key.iloc[0] == mc.match(q2, df, arr, k=1).key.iloc[0])

# --------------------------------------------------------------- the page runs (stub DOM) and finds the worker U-turn
rp = subprocess.run(["node", str(HERE / "page_check.js"), str(page)], capture_output=True, text=True, timeout=50)
try:
    pc = json.loads(rp.stdout)
    check("page: 8 cards for the u-turn preset, best is a worker_zone hairpin", pc["cards"] == 8 and "worker_zone" in (pc["first"] or ""), pc.get("first", "")[:60])
    check("page: every preset and a freehand path return cards (weak ones are hidden behind a banner / toggle)", all(v >= 3 for v in pc["perPreset"].values()) and pc["freehand"] >= 3, str(pc["perPreset"]))
except Exception as ex:  # noqa: BLE001
    check("page runs under node", False, (rp.stderr or str(ex))[-200:])

# ------------------------------------------------------------------- JS search == python search (node), on the real table
d = mc.build_all()
dfr = d["df"]
arr_r = mc.tl_arrays(dfr)
queries, py = [], {}
for p in ("u_turn", "left_to_right", "stop_and_go", "loiter", "diagonal"):
    qq = mc.preset_query(p)
    queries.append({"name": p, "pts": qq.pts.tolist(), "aspect": qq.aspect, "pos_w": qq.pos_w, "size_w": qq.size_w, "any_rot": qq.any_rot,
                    "speed": None if qq.speed is None else qq.speed.tolist(), "speed_w": qq.speed_w, "shape_w": qq.shape_w, "turn_w": qq.turn_w, "straight_w": qq.straight_w})
    py[p] = mc.match(qq, dfr, arr_r, k=8)
data = {"n": int(len(dfr)), "pos": np.array(dfr["pos"].tolist()).ravel().round(5).tolist(), "speed": np.array(dfr["speed"].tolist()).ravel().round(3).tolist(),
        "ext": dfr["ext"].round(5).tolist(), "rel": dfr["rel_ext"].round(4).tolist(), "straight": dfr["straight"].round(4).tolist(), "len": dfr["len"].round(5).tolist(), "aspect": [d["meta"][v]["w"] / d["meta"][v]["h"] for v in dfr["video"]]}
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "in.json"
    f.write_text(json.dumps({"data": data, "queries": queries}), encoding="utf-8")
    r = subprocess.run(["node", str(HERE / "parity_check.js"), str(f)], capture_output=True, text=True, timeout=50)
if r.returncode != 0:
    check("node parity run", False, r.stderr[-200:])
else:
    js = json.loads(r.stdout)
    for qd, jr in zip(queries, js):
        pk_keys = list(py[qd["name"]].key[:5])
        js_keys = [dfr.key.iloc[x["i"]] for x in jr[:5]]
        same = pk_keys[:3] == js_keys[:3]
        dc = max(abs(a - b) for a, b in zip(py[qd["name"]].cost[:3], [x["cost"] for x in jr[:3]]))
        check(f"JS == python top-3 for {qd['name']}", same and dc < 0.02, f"max cost diff {dc:.4f}")
        if qd["any_rot"]:
            check(f"JS == python matched rotation of the top hit for {qd['name']}", int(py[qd["name"]].rot_deg.iloc[0]) == int(jr[0]["rot"]),
                  f"python {int(py[qd['name']].rot_deg.iloc[0])} deg, js {jr[0]['rot']} deg")
        else:
            check(f"{qd['name']}: no rotation allowed -> every hit keeps the drawn direction", (py[qd["name"]].rot_deg == 0).all() and all(x["rot"] == 0 for x in jr[:8]))

# ------------------------------------------------------------- direction semantics, halts, negatives, weak-result handling
ut = mc.match(mc.preset_query("u_turn"), dfr, arr_r, k=8)
check("u_turn top hit reports how far the sketch was turned (multiple of 30 deg)", int(ut.rot_deg.iloc[0]) % 30 == 0 and ut.rot_deg.between(0, 330).all(), f"rot {list(ut.rot_deg[:3])}")
# a path with a 3 s halt in the middle: exactly one pause of ~3 s; presets expose their temporal cues
n = 150
x = np.r_[np.linspace(0.1, 0.4, 50), np.full(50, 0.4), np.linspace(0.4, 0.7, 50)]
stop_tl = mc.Tracklet("s", "synthetic", 2, "person", 0, np.arange(n) / 15.0, np.stack([x, np.full(n, 0.5)], 1), np.full((n, 2), [0.05, 0.2]), 1280, 720)
pz = mc.find_pauses(stop_tl, 15.0)
check("find_pauses: one halt of about 3 s in walk-stop-walk", len(pz) == 1 and 2.4 <= pz[0][2] <= 3.4, str([(a, b, round(c, 1)) for a, b, c in pz]))
check("stop_and_go preset carries two pause marks, loiter a dwell mark, left_to_right none",
      len(mc.preset_marks("stop_and_go")) == 2 and len(mc.preset_marks("loiter")) == 1 and mc.preset_marks("left_to_right") == [])
neg = RES.get("negative_test", {})
check("hard negative recorded: u_turn drawn on yonge_dundas -> NO MATCH, hero_negative.jpg written",
      neg.get("verdict") == "NO MATCH" and (OUT / "hero_negative.jpg").exists() and 0 < jpg_width(OUT / "hero_negative.jpg") <= 1280, f"cost {neg.get('top1_cost')}")
page_txt = page.read_text(encoding="utf-8")
check("page has the weak-result handling (banner, hide-weak toggle) and the turned-sketch chip", all(t in page_txt for t in ("className = 'banner'", "weaker candidate", "sketch turned")))

# --------------------------------------------------- the page in a REAL browser (headless Chrome/Edge, file://, synthetic pointer events)
try:
    sys.path.insert(0, str(HERE / "tools"))
    import page_drive as pdv
    pdv.find_browser()
    have_browser = True
except Exception:  # noqa: BLE001
    have_browser = False
if have_browser:
    scs = {s["name"]: s for s in pdv.SCENARIOS}
    r1 = pdv.run(scs["draw_u_turn"])
    check("browser: drawing a U-turn with the mouse returns 8 rendered cards, no JS errors, best is a worker hairpin",
          r1.get("cards", 0) >= 5 and r1.get("nonblank_canvases") == r1.get("cards") and not r1.get("errors") and "worker_zone" in (r1.get("first") or ""), (r1.get("first") or r1.get("fatal", ""))[:70])
    r2 = pdv.run(scs["yonge_u_turn_negative"])
    check("browser: U-turn drawn on yonge_dundas shows the NO MATCH banner and greyed cards", bool(r2.get("banner")) and r2.get("weak_cards", 0) >= 1 and not r2.get("errors"), str({k: r2.get(k) for k in ("banner", "weak_cards")}))
else:
    print("SKIP browser checks (no Chrome/Edge found)")

print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
