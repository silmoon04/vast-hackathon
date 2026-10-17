"""Overnight Ledger: was my parked car there all night, was the camera blind, did anyone come near it?

  python run.py --all                 ptz_parking, lisbon_parking, car_detection + two SIMULATED fault copies of a ptz window
  python run.py --video ptz_parking   one catalogue clip (add --tid N to pick the car by tracker id, --cars K for the table)
  python run.py --video sim_a|sim_b   the SIMULATED fault copies (built from the real ptz window, ground truth known)
Outputs: out/<video>/{hero.jpg, report.html, strip.png, check.jpg, ledger.csv, thumbs/}, results.json
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protokit as pk  # noqa: E402
import ledger as L  # noqa: E402
import render as R  # noqa: E402
import sim as S  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
CACHE = OUT / "_cache"
DEFAULT = ["ptz_parking", "lisbon_parking", "car_detection"]
# SIMULATED fault copies of the real ptz_parking footage [69 s, 154 s) (steady view, after the one real camera re-aim at 68.2 s)
SIM_BASE = ("ptz_parking", 69.0, 154.0)
SIMS = {
    "sim_a": {"faults": [{"kind": "cover", "start": 10, "end": 20}, {"kind": "freeze", "start": 55, "end": 63}], "damage": None,
              "title": "SIMULATED: 10 s black lens cover + 8 s frozen feed"},
    "sim_b": {"faults": [{"kind": "cloth", "start": 20, "end": 27}, {"kind": "blur", "start": 40, "end": 46},
                         {"kind": "freeze", "start": 62, "end": 67}], "damage": {"t": 24.0},
              "title": "SIMULATED: cloth over lens 7 s + defocus 6 s + freeze 5 s, scratch painted on the car at 00:24"},
}
CAUSE_OF = {"cover": "covered", "cloth": "covered", "freeze": "frozen", "blur": "blurred"}
CAUSE_TXT = {"covered": "lens covered", "frozen": "feed frozen", "blurred": "out of focus",
             "view": "camera looking elsewhere (view changed)", "occluded": "spot blocked by another vehicle or person",
             "unsure": "spot unclear (car partly hidden or lighting changed)"}
COLORS = {"P": "#2e9e5b", "A": "#d64545", "D": "#e0a100", "G": "#8a94a0"}


HERO_CTX: dict[str, dict[str, Any]] = {}


def save_hero(name: str, outdir: Path, badge: str | None = None) -> None:
    ctx = dict(HERO_CTX[name])
    ctx["badge"] = badge
    pk.save_jpg(R.hero_ledger(ctx), outdir / "hero.jpg")


def kill_badge(results: dict[str, Any]) -> str | None:
    """One-line proof of the strongest metric, from the SIMULATED fault copies (None until they have been run)."""
    ev = [results[k]["sim_eval"] for k in SIMS if k in results and "sim_eval" in results[k]]
    if len(ev) < len(SIMS):
        return None
    gt_s, tp, fp = (sum(e[k] for e in ev) for k in ("gt_blind_s", "tp", "fp"))
    n_f = sum(len(e["per_fault"]) for e in ev)
    return f"SIMULATED camera faults: {tp}/{gt_s} blind seconds caught ({n_f} faults), {fp} false alarms"


def log(*a: Any) -> None:
    print(f"[ledger {time.strftime('%H:%M:%S')}]", *a, flush=True)


def tag(sim_meta: dict[str, Any] | None, s: int, e: int, cause: str) -> str:
    """Cause text, with SIMULATED appended when the stretch overlaps an injected fault."""
    txt = CAUSE_TXT.get(cause, cause)
    if sim_meta and any(a < e and b > s and CAUSE_OF.get(k) == cause for a, b, k in sim_meta["gt_blind"]):
        txt += " (SIMULATED)"
    return txt


def unity(x: float) -> str:
    return f"{x:.2f}"


# ------------------------------------------------------------------------------------------------- no-ledger videos
def health_only(F: dict[str, Any]) -> dict[str, Any]:
    """Whole-frame health flags per second without any car: dark/flat, frozen (>= 2 s of identical frames)."""
    t = F["t"]
    sec = t.astype(int)
    n = int(sec.max()) + 1 if len(sec) else 0
    dark = np.zeros(n, bool)
    froz = np.zeros(n, bool)
    run = L.frozen_mask(F)
    for k in range(n):
        idx = np.where(sec == k)[0]
        if len(idx):
            dark[k] = L.covered_frame(np.median(F["mean"][idx]), np.median(F["std"][idx]), float(np.median(F["mean"])),
                                      float(np.median(F["std"])))
            froz[k] = run[idx].mean() >= 0.6
    return {"seconds": n, "dark_or_flat_s": int(dark.sum()), "frozen_s": int(froz.sum())}


def no_car(name: str, path: Path, tr: pk.Tracks, F: dict[str, Any], runs, outdir: Path, why: str) -> dict[str, Any]:
    inf = pk.info(path)
    longest = max(((b - a), a) for a, b in runs) if runs else (0.0, 0.0)
    cuts = int(sum(1 for x in F["d"] if x > L.P["cut_d"]))
    veh = tr.rows[tr.rows["name"].isin(L.VEH) & (tr.rows["tid"] >= 0)]
    dwell = 0.0
    if not veh.empty:
        g = veh.groupby("tid")
        dwell = float(max(((x["t"].max() - x["t"].min()) if (np.hypot(x["cx"].max() - x["cx"].min(), x["cy"].max() - x["cy"].min())
                                                              < 0.25 * max(x["bw"].median(), x["bh"].median())) else 0.0)
                          for _, x in g))
    ts = np.linspace(0.04, 0.96, 12) * inf["duration"]
    sheet = pk.sheet([pk.frame_at(path, t) for t in ts], [pk.mmss(t) for t in ts], cols=4, tile_w=300)
    hc = health_only(F)
    lines = [f"!{why}",
             f"{cuts} hard cuts in {inf['duration']:.0f} s; longest continuous shot {longest[0]:.1f} s (at {R.mmss(longest[1])})",
             f"longest a vehicle sat still: {dwell:.1f} s (a parked car needs >= 20 s in one shot)",
             f"health flags: {hc['dark_or_flat_s']} s dark/flat, {hc['frozen_s']} s frozen (static slate / still images)"]
    hero = R.hero_nocar({"subtitle": f"{name}  |  {inf['w']}x{inf['h']}  |  {inf['duration']:.0f} s", "sheet": sheet,
                         "stamp": "NO LEDGER POSSIBLE", "lines": lines,
                         "footer": "Overnight Ledger refuses to invent a verdict: no parked car in one continuous view."})
    pk.save_jpg(hero, outdir / "hero.jpg")
    rep = pk.Report(outdir / "report.html", f"Overnight Ledger: {name}", "No parked car in a continuous view: the ledger declines.")
    rep.credit(path)
    rep.image(outdir / "hero.jpg", "hero")
    rep.kv({"hard cuts": cuts, "longest shot (s)": round(longest[0], 1), "longest still vehicle (s)": round(dwell, 1),
            "frozen seconds (static slate)": hc["frozen_s"]})
    rep.text(why)
    rep.save()
    return {"video": name, "ledger": False, "reason": why, "hard_cuts": cuts, "duration_s": round(inf["duration"], 1),
            "longest_shot_s": round(longest[0], 1), "longest_still_vehicle_s": round(dwell, 1), "health": hc}


# ------------------------------------------------------------------------------------------------------ main analysis
def verdict(led: L.Ledger, strs: list[L.Stretch], cts: list[dict[str, Any]], sim_meta=None) -> tuple[tuple[str, tuple], list[str], dict]:
    s = L.summarize(led)
    w0, w1 = led.window
    big = [x for x in strs if x.dur >= 3]
    absent_runs = [(a, b) for a, b, c, _ in L.spans(led) if c == "A" and b - a >= 2]
    changed = [x for x in big if x.verdict.startswith(("CHANGED", "CAR GONE"))]
    unknown = s["cant_see_s"]
    if absent_runs:
        a, b = absent_runs[0]
        stamp = ("WAS NOT THERE", R.RED)
        lead = f"!Spot empty {R.mmss(a)} to {R.mmss(b)}" + (f" and {len(absent_runs) - 1} more time(s)" if len(absent_runs) > 1 else "")
    elif any(x.verdict.startswith("CHANGED") for x in big):
        stamp = ("CHANGED: CHECK", R.AMBER)
        lead = "!The car looks different after a blind stretch"
    elif unknown:
        stamp = ("THERE, SAME CAR", R.GREEN)
        lead = "!Present whenever we could see it"
    else:
        stamp = ("THERE THE WHOLE TIME", R.GREEN)
        lead = "!Present in every second"
    parts = []
    if unknown:
        by = {}
        for q in range(w0, w1):
            if led.state[q] == "G":
                by[led.cause[q]] = by.get(led.cause[q], 0) + 1
        if len(by) == 1:
            cs = CAUSE_TXT.get(next(iter(by)), "").split(" (")[0]
        else:
            cs = ", ".join(f"{n} s {CAUSE_TXT.get(c, c).split(' (')[0]}" for c, n in sorted(by.items(), key=lambda kv: -kv[1]))
        sim_flag = " (SIMULATED)" if sim_meta else ""
        parts.append(f"!UNKNOWN {unknown} s: {cs}{sim_flag}")
    else:
        parts.append(f"!Seen in all {s['seconds']} s analysed")
    if big:
        ok = [x for x in big if x.verdict == "same car"]
        if ok and len(ok) == len(big):
            parts.append("!Car present before and after every blind stretch: same car")
        elif big:
            parts.append("!" + "; ".join(f"{R.mmss(x.s)}: {x.verdict}" for x in big[:2]))
    parts.insert(0, lead)
    if cts:
        times = ", ".join(R.mmss(c["t_best"]) for c in cts[:4])
        near = sum(1 for c in cts if c["touch_zone"])
        parts.append(f"!{len(cts)} {'person' if len(cts) == 1 else 'people'} came close ({times})" + (f", {near} stepped into the car footprint" if near else ""))
    else:
        parts.append("!Nobody came near the car")
    parts.append(f"{s['present_pct']:.0f}% present, {s['cant_see_pct']:.0f}% can't see, {s['absent_s']} s absent over {R.mmss(w0)} to {R.mmss(w1)}")
    return stamp, parts, s


def contact_thumb(al: L.Align, led: L.Ledger, c: dict[str, Any], label_pre: str = "") -> np.ndarray | None:
    k = c["sec_best"]
    if k not in al.jpgs or led.Ms[k] is None:
        return None
    im = cv2.imdecode(np.frombuffer(al.jpgs[k], np.uint8), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = led.spot.box
    bw, bh = x2 - x1, y2 - y1
    pb = L.map_box(led.Ms[k], c["box_best"])
    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (60, 200, 90), 2)
    cv2.rectangle(im, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (0, 140, 255), 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(1.2 * bw, 90)
    half_h = max(1.0 * bh, 70)
    xa, xb = int(max(0, cx - half_w)), int(min(al.W, cx + half_w))
    ya, yb = int(max(0, min(pb[1], y1) - 20)), int(min(al.H, max(pb[3], y2) + 20))
    ya = max(0, min(ya, int(cy - half_h)))
    return im[ya:yb, xa:xb].copy()


def tick(rep_rows: list[str], *cells: str) -> None:
    rep_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")


def analyze(name: str, outdir: Path, *, path: Path | None = None, tr: pk.Tracks | None = None, sim_meta: dict[str, Any] | None = None,
            spot: L.Spot | None = None, tid: int | None = None, n_cars: int = 0, title: str | None = None) -> tuple[dict, Any]:
    t_start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    path = path or pk.video_path(name)
    tr = tr or pk.tracks(name)
    F = L.features(path, CACHE)
    runs = L.split_runs(F)
    inf = pk.info(path)
    log(name, f"features ok, {len(runs)} runs, longest {max((b - a) for a, b in runs):.1f} s")
    if spot is None:
        cands = L.candidate_spots(tr, runs)
        if not cands:
            why = "No parked car found: no vehicle sat still for 20 s inside one continuous shot"
            return no_car(name, path, tr, F, runs, outdir, why), None
    else:
        cands = [spot]
    al = L.align_video(path, F, cands[0].t_ref, video=name, cache_dir=CACHE)
    log(name, f"aligned {sum(m is not None for m in al.Ms)}/{al.nsec} seconds to the reference view")
    # candidate boxes are raw pixels of their own shot: keep the shots that sit in the reference view (identity transform)
    def in_ref_view(c: L.Spot) -> bool:
        M = al.Ms[min(int(c.t_ref), al.nsec - 1)]
        return M is not None and abs(M[0, 2]) < 8 and abs(M[1, 2]) < 8 and 0.97 < M[0, 0] < 1.03
    cands = [c for c in cands if in_ref_view(c)] or cands[:1]
    for c in cands:
        c.vis = len(L.visible_seconds(al, c.box))
    if tid is not None:
        pick = [c for c in cands if tid in c.tids]
        if not pick:
            raise SystemExit(f"tid {tid} is not a parked-car candidate (candidates: {[c.tids[:3] for c in cands[:12]]})")
        cands = pick + [c for c in cands if c not in pick]
    elif spot is None:
        # park-longest first, then the car we can verify the longest (still in frame after the camera comes back), then size
        cands.sort(key=lambda c: (-round(c.stay / 5.0), -round(c.vis / 10.0), -(c.box[2] - c.box[0]) * (c.box[3] - c.box[1])))
    allc = (cands if n_cars <= 0 else cands[:n_cars])[:40]
    # every candidate gets a ledger (20 at a time so crops never pile up); the table shows them all and the most interesting
    # one (a car that is gone for >= 5 s) becomes the "second car" of the hero
    cars: list[dict[str, Any]] = []
    ledgers: list[L.Ledger] = []
    for b0 in range(0, len(allc), 20):
        batch = allc[b0:b0 + 20]
        for j, (c, pre) in enumerate(zip(batch, L.precrop(al, batch))):
            led = L.build(al, tr, F, c, pre)
            sm, st_i, ct_i = L.summarize(led), L.stretches(led), L.contacts(led, tr)
            cars.append({"rank": b0 + j + 1, "tids": led.spot.tids[:4], "box": [round(v) for v in led.spot.box],
                         "name": led.spot.name, "shot": [round(v) for v in led.spot.run], "stay_s": round(led.spot.stay),
                         "visible_s": getattr(led.spot, "vis", None),
                         **{k: sm[k] for k in ("present_s", "absent_s", "different_s", "cant_see_s")},
                         "stretch_verdicts": sorted({x.verdict for x in st_i}), "contacts": len(ct_i)})
            if b0 + j > 0:
                led.crops = {}
            ledgers.append(led)
    main = ledgers[0]
    log(name, f"{len(ledgers)} candidate cars; main car tids {main.spot.tids[:3]} box {[round(v) for v in main.spot.box]}, "
              f"summary {L.summarize(main)}")
    strs = L.stretches(main)
    cts = L.contacts(main, tr)
    stamp, lines, summ = verdict(main, strs, cts, sim_meta)
    w0, w1 = main.window
    second = max((i for i, c in enumerate(cars) if i > 0 and c["absent_s"] >= 5), key=lambda i: cars[i]["absent_s"], default=None)
    # ---- thumbnails
    th = outdir / "thumbs"
    th.mkdir(exist_ok=True)
    focus = None
    stretch_out = []
    for i, st in enumerate(strs):
        d = {"start": st.s, "end": st.e, "dur_s": st.dur, "cause": st.cause, "causes": st.causes, "verdict": st.verdict}
        if st.cmp:
            d["compare"] = {k: round(v, 3) for k, v in st.cmp.items()}
        if st.before and st.after:
            d["before_sec"], d["after_sec"] = st.before["sec"], st.after["sec"]
            d["before_state"], d["after_state"] = st.before["state"], st.after["state"]
            b, a = main.crops[st.before["sec"]], main.crops[st.after["sec"]]
            pk.save_jpg(np.hstack([cv2.resize(b, (a.shape[1], a.shape[0])), np.full((a.shape[0], 6, 3), 255, np.uint8), a]),
                        th / f"stretch_{i}_before_after.jpg")
            d["thumb"] = f"thumbs/stretch_{i}_before_after.jpg"
        stretch_out.append(d)
    big = [s for s in strs if s.dur >= 3 and s.before and s.after]
    if big:
        fs = max(big, key=lambda s: s.dur)
        focus = {"before": main.crops[fs.before["sec"]], "after": main.crops[fs.after["sec"]], "verdict": fs.verdict,
                 "tb": fs.before["sec"], "ta": fs.after["sec"],
                 "metric": (f"SSIM {fs.cmp['ssim']:.2f}, worst block {fs.cmp['block_min']:.2f}" if fs.cmp else "")}
    contacts_ctx, contacts_out = [], []
    for j, c in enumerate(cts):
        tn = contact_thumb(al, main, c)
        d = dict(c)
        d["touch_check"] = L.contact_check(main, c)
        tc = d["touch_check"]
        touch_txt = ("car same before/after (SSIM %.2f)" % tc["ssim"]) if tc["verdict"] == "same car" and "ssim" in tc else (
            "car CHANGED: check (SSIM %.2f)" % tc["ssim"] if "ssim" in tc else "no clear before/after view")
        if tn is not None:
            pk.save_jpg(tn, th / f"contact_{j}.jpg")
            d["thumb"] = f"thumbs/contact_{j}.jpg"
            lab = f"{R.mmss(c['t0'])}-{R.mmss(c['t1'])}  {'in footprint' if c['touch_zone'] else 'closest ' + str(int(c['min_dist_px'])) + ' px'}"
            contacts_ctx.append({"thumb": tn, "label": lab, "touch": touch_txt})
        contacts_out.append(d)
    # ---- hero
    gt = None
    if sim_meta:
        gt = [(a, b, k) for a, b, k in sim_meta["gt_blind"]]
    sp = L.spans(main)
    sec_ctx = None
    if second is not None:
        led2 = ledgers[second]
        first_abs = next((a for a, b, c, _ in L.spans(led2) if c == "A"), None)
        sec_ctx = {"title": f"SECOND CAR (tracker {led2.spot.tids[:2]}): present until {R.mmss(first_abs)}, then the spot is empty "
                            f"({L.summarize(led2)['absent_s']} s absent)",
                   "spans": L.spans(led2)}
    # honest scope note on the stamp, and (for the SIMULATED scratch) the known miss next to the before/after pair
    dmg_note = None
    if sim_meta and sim_meta.get("damage"):
        t_dm = sim_meta["damage"]["t"]
        flagged = any(x.verdict.startswith("CHANGED") for x in strs) or any(c["touch_check"]["verdict"].startswith("CHANGED") for c in contacts_out)
        dmg_note = (f"SIMULATED scratch painted on the car at {R.mmss(t_dm)}: " + ("flagged." if flagged else "NOT detected (known miss). The appearance check cannot see scratches."), not flagged)
    elif stamp[0] in ("THERE, SAME CAR", "THERE THE WHOLE TIME"):
        dmg_note = ("Appearance check only: scratches and dents are not checked.", False)
    credit_txt = pk.credit(sim_meta["base"] if sim_meta else path)
    short_credit = credit_txt.split(", via")[0]
    HERO_CTX[name] = {
        "video": name, "subtitle": f"{name}  |  {inf['w']}x{inf['h']}  |  window {R.mmss(w0)} to {R.mmss(w1)}",
        "ref_frame": L.ref_image(al, main.spot), "box": main.spot.box, "t0": w0, "t1": w1, "spans": sp, "gt": gt, "stamp": stamp, "lines": lines,
        "stamp_note": "same car = appearance match; damage is NOT checked" if stamp[0].startswith("THERE") else None,
        "damage_note": dmg_note,
        "stretches": [{"s": s.s, "e": s.e, "label": f"UNKNOWN {s.dur} s: {R.CAUSE.get(s.cause, s.cause)}" + (" (SIMULATED)" if "SIMULATED" in tag(sim_meta, s.s, s.e, s.cause) else "")} for s in strs], "contacts": contacts_ctx, "second": sec_ctx, "focus": focus,
        "sim": bool(sim_meta), "strip_title": "SECOND BY SECOND, THE WHOLE WINDOW",
        "footer": ("Stand-ins: YOLO26s+ByteTrack, OpenCV. " + ("Faults on this clip are SIMULATED. " if sim_meta else "") + "Footage: " + short_credit)}
    save_hero(name, outdir)
    pk.save_jpg(R.strip_png(sp, w0, w1, gt=gt, title=f"{name}: car at box {[round(v) for v in main.spot.box]}"), outdir / "strip.png")
    # ---- eyeball check sheet: raw frames with the spot mapped back into each frame, labelled with the state
    secs = sorted(set(np.linspace(w0, w1 - 1, 14).astype(int).tolist()
                      + [q for st in strs[:3] for q in (st.s - 1, st.s, st.e - 1, st.e) if w0 <= q < w1]))
    tiles, labs = [], []
    for k in secs[:20]:
        fr = pk.frame_at(path, min(k + 0.4, inf["duration"] - 0.1))
        if main.Ms[k] is not None:
            bx = L.map_box(L.inv_M(main.Ms[k]), main.spot.box)
            col = {"P": (90, 200, 60), "A": (60, 60, 230), "D": (0, 170, 230), "G": (150, 150, 150)}[main.state[k]]
            cv2.rectangle(fr, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), col, 4)
        lab = f"{R.mmss(k)} {main.state[k]}" + (f" ({main.cause[k]})" if main.cause[k] else "")
        tiles.append(fr)
        labs.append(lab)
    pk.save_jpg(pk.sheet(tiles, labs, cols=5, tile_w=300), outdir / "check.jpg")
    with open(outdir / "ledger.csv", "w", encoding="utf-8") as f:
        f.write("sec,time,state,cause,appearance_ssim\n")
        for k in range(len(main.state)):
            a = main.app[k]
            f.write(f"{k},{R.mmss(k)},{main.state[k]},{main.cause[k]},{'' if a != a else round(a, 3)}\n")
    second_out = None
    if second is not None:
        led2 = ledgers[second]
        with open(outdir / "ledger_second_car.csv", "w", encoding="utf-8") as f:
            f.write("sec,time,state,cause,appearance_ssim\n")
            for k in range(len(led2.state)):
                a = led2.app[k]
                f.write(f"{k},{R.mmss(k)},{led2.state[k]},{led2.cause[k]},{'' if a != a else round(a, 3)}\n")
        second_out = {"spot": led2.spot.to_json(), "summary": L.summarize(led2), "spans": [list(x) for x in L.spans(led2)]}
    # ---- the ledger as a VastDB table (duckdb stand-in): real SQL, restricted to the dialect VastDB accepts
    #      (WHERE / ORDER BY / LIMIT / plain aggregates; no IN, ILIKE, GROUP BY, JOIN, window functions)
    tbl = pd.DataFrame({"video": name, "car": "car_1", "sec": list(range(len(main.state))), "state": main.state, "cause": main.cause})
    con = pk.vastdb({"ledger": tbl})
    sql_demo = []
    for q, text in (("seconds we can't see", "SELECT COUNT(*) AS n FROM ledger WHERE state = 'G'"),
                    ("seconds the car is confirmed present", "SELECT COUNT(*) AS n FROM ledger WHERE state = 'P'"),
                    ("first 3 seconds the spot was empty", "SELECT sec FROM ledger WHERE state = 'A' ORDER BY sec LIMIT 3"),
                    ("first covered-lens second", "SELECT MIN(sec) AS first_sec FROM ledger WHERE state = 'G' AND cause = 'covered'")):
        sql_demo.append({"question": q, "sql": text, "rows": con.execute(text).fetchall()})
    # ---- html report
    rep = pk.Report(outdir / "report.html", f"Overnight Ledger: {name}", title or "Per-second ledger of one parked car")
    rep.credit(path)
    rep.image(outdir / "hero.jpg", "The 5-second story")
    rep.kv({"verdict": stamp[0], "present": f"{summ['present_pct']}%", "can't see": f"{summ['cant_see_s']} s",
            "absent": f"{summ['absent_s']} s", "people close": len(cts), "window": f"{R.mmss(w0)}-{R.mmss(w1)}"})
    rep.text("Scope of the verdict: 'same car' is an appearance match (SSIM / block SSIM / colour) of the car crop before and after "
             "each blind stretch. Damage (scratches, dents) is NOT checked. " + (dmg_note[0] if dmg_note and sim_meta else ""))
    rep.h2("Strip (hover for causes)")
    def tip(a: int, b: int, c: str, cz: str) -> str:
        what = {"P": "present", "A": "ABSENT", "D": "different car"}.get(c) or ("CAN'T SEE: " + tag(sim_meta, a, b, cz))
        return f"{R.mmss(a)}-{R.mmss(b)} {what}"
    tl = [(a - w0, b - w0, {"P": "present", "A": "absent", "D": "#e0a100", "G": "#8a94a0"}[c], tip(a, b, c, cz)) for a, b, c, cz in sp]
    rep.html(f'<figure>{pk.timeline_svg(tl, w1 - w0, width=1000, ticks=8)}'
             f"<figcaption>Axis is relative: 00:00 = {R.mmss(w0)} of the clip.</figcaption></figure>")
    rep.image(outdir / "strip.png", "Strip with injected faults (blue brackets) where simulated" if sim_meta else "Strip")
    if strs:
        rep.h2("Can't-see stretches: before / after")
        rows = ["<table><tr><th>when</th><th>length</th><th>cause</th><th>verdict</th><th>before vs after</th></tr>"]
        for d, st in zip(stretch_out, strs):
            img = f'<img src="{d["thumb"]}" style="max-height:110px">' if "thumb" in d else ""
            m = d.get("compare")
            ms = f"<br><small>SSIM {m['ssim']}, worst block {m['block_min']}, hue {m['hue']}</small>" if m else ""
            tick(rows, f"{R.mmss(st.s)}-{R.mmss(st.e)}", f"{st.dur} s", html.escape(tag(sim_meta, st.s, st.e, st.cause)),
                 f"<b>{html.escape(st.verdict)}</b>{ms}", img)
        rows.append("</table>")
        rep.html("".join(rows))
    rep.h2("People near the car")
    if contacts_out:
        rows = ["<table><tr><th>when</th><th>closest</th><th>frames</th><th>TOUCH? car before vs after the visit</th><th>thumbnail</th></tr>"]
        for d in contacts_out:
            img = f'<img src="{d["thumb"]}" style="max-height:130px">' if "thumb" in d else ""
            tick(rows, f"{R.mmss(d['t0'])}-{R.mmss(d['t1'])} ({d['dur']} s)",
                 "inside the car footprint" if d["touch_zone"] else f"{d['min_dist_px']} px from the footprint", str(d["n_frames"]),
                 (f"<b>{html.escape(d['touch_check']['verdict'])}</b><br><small>SSIM {d['touch_check']['ssim']} "
                  f"({R.mmss(d['touch_check']['before_sec'])} vs {R.mmss(d['touch_check']['after_sec'])})</small>")
                 if "ssim" in d["touch_check"] else html.escape(d["touch_check"]["verdict"]), img)
        rows.append("</table>")
        rep.html("".join(rows))
    else:
        rep.text("No person's feet entered the dilated zone (30% of the car width around its lower half).")
    rep.text("'Came close' is geometry only: feet inside a zone around the car. It does not say anyone touched it. The TOUCH? score "
             "compares the car crop just before and just after the visit with the same SSIM test as the blind stretches: it answers "
             "'same car?', not 'any scratch?'.")
    rep.h2("Other parked cars in the same view")
    rep.table(cars, ["rank", "tids", "box", "shot", "stay_s", "visible_s", "present_s", "absent_s", "cant_see_s", "contacts", "stretch_verdicts"], max_rows=60)
    rep.h2("Check frames (the spot box is mapped back into each raw frame)")
    rep.image(outdir / "check.jpg", "box colour: green present, red absent, grey can't see")
    rep.h2("Ledger as a VastDB table (duckdb stand-in, VastDB-safe SQL)")
    rep.table([{"question": d["question"], "sql": d["sql"], "answer": str(d["rows"])} for d in sql_demo])
    rep.h2("Thresholds used")
    rep.json({k: v for k, v in L.P.items()})
    rep.save()
    res = {"video": name, "ledger": True, "duration_s": round(inf["duration"], 1), "spot": main.spot.to_json(),
           "summary": summ, "verdict": stamp[0], "stretches": stretch_out, "contacts": contacts_out, "cars": cars,
           "second_car_index": second, "second_car": second_out, "sql_demo": sql_demo, "main_spans": [list(x) for x in sp], "aligned_seconds": int(sum(m is not None for m in al.Ms)),
           "hero_note": dmg_note[0] if dmg_note else None, "runtime_s": round(time.time() - t_start, 1), "hero": str((outdir / "hero.jpg").relative_to(HERE)).replace("\\", "/")}
    if sim_meta:
        res["sim"] = {"title": sim_meta.get("title"), "faults": sim_meta["faults"], "damage": sim_meta.get("damage")}
        res["sim_eval"] = eval_sim(main, sim_meta, strs)
    return res, (main, al, ledgers)


# --------------------------------------------------------------------------------------------------- sim evaluation
def eval_sim(led: L.Ledger, meta: dict[str, Any], strs: list[L.Stretch]) -> dict[str, Any]:
    n = len(led.state)
    gt = np.zeros(n, bool)
    cause_gt = [""] * n
    for a, b, k in meta["gt_blind"]:
        for q in range(int(a), min(int(b), n)):
            gt[q] = True
            cause_gt[q] = CAUSE_OF[k]
    pred = np.array([s == "G" for s in led.state], bool)

    def near(m: np.ndarray, k: int, tol: int) -> bool:
        return bool(m[max(0, k - tol):k + tol + 1].any())
    tp = int((gt & pred).sum())
    prec, rec = tp / max(1, int(pred.sum())), tp / max(1, int(gt.sum()))
    tp_t = sum(1 for k in range(n) if pred[k] and near(gt, k, 1))
    rec_t = sum(1 for k in range(n) if gt[k] and near(pred, k, 1))
    cause_ok = sum(1 for k in range(n) if gt[k] and pred[k] and led.cause[k] == cause_gt[k])
    per = []
    for a, b, kind in meta["gt_blind"]:
        ks = [k for k in range(max(0, int(a) - 2), min(n, int(b) + 2)) if pred[k]]
        # contiguous predicted block touching the gt interval
        blk = [k for k in ks if int(a) - 1 <= k <= int(b)]
        per.append({"kind": kind, "gt": [a, b], "pred": [min(blk), max(blk) + 1] if blk else None,
                    "onset_error_s": (min(blk) - int(a)) if blk else None, "offset_error_s": (max(blk) + 1 - int(b)) if blk else None,
                    "cause_correct_pct": round(100 * np.mean([led.cause[k] == CAUSE_OF[kind] for k in range(int(a), int(b)) if k < n and pred[k]]
                                                              or [0]), 1)})
    fp = [k for k in range(n) if pred[k] and not gt[k]]
    out = {"seconds": n, "gt_blind_s": int(gt.sum()), "pred_blind_s": int(pred.sum()), "tp": tp, "fp": len(fp), "fn": int(gt.sum()) - tp,
           "precision": round(prec, 3), "recall": round(rec, 3), "precision_tol1s": round(tp_t / max(1, int(pred.sum())), 3),
           "recall_tol1s": round(rec_t / max(1, int(gt.sum())), 3), "cause_accuracy": round(cause_ok / max(1, tp), 3),
           "false_positive_seconds": fp[:20], "per_fault": per,
           "stretch_verdicts": [{"start": s.s, "end": s.e, "cause": s.cause, "verdict": s.verdict} for s in strs]}
    if meta.get("damage"):
        t_dm = meta["damage"]["t"]
        out["damage_flagged"] = [{"stretch": [s.s, s.e], "verdict": s.verdict, "damage_before": s.e <= t_dm or s.s < t_dm <= s.e,
                                  "expected": "CHANGED" if (s.s < t_dm <= s.e) else "same car"} for s in strs]
    return out


# ------------------------------------------------------------------------------------------------------------ main
def build_sim(key: str, ptz_spot: L.Spot) -> tuple[Path, pk.Tracks, dict[str, Any], L.Spot]:
    cfg = SIMS[key]
    base, t0, t1 = SIM_BASE
    outdir = OUT / key
    outdir.mkdir(parents=True, exist_ok=True)
    mp4 = outdir / "sim.mp4"
    base_tr = pk.tracks(base)
    meta_f = outdir / "sim_meta.json"
    dmg = {"t": cfg["damage"]["t"], "box": ptz_spot.box} if cfg["damage"] else None
    want = {"faults": cfg["faults"], "damage": dmg, "t0": t0, "t1": t1}
    if mp4.exists() and meta_f.exists() and json.loads(meta_f.read_text()).get("want") == json.loads(json.dumps(want)):
        meta = json.loads(meta_f.read_text())["meta"]
        meta["gt_blind"] = [tuple(x) for x in meta["gt_blind"]]
    else:
        log(key, "writing SIMULATED fault copy", cfg["title"])
        meta = S.make(base, t0, t1, cfg["faults"], mp4, damage=dmg)
        meta["title"] = cfg["title"]
        meta_f.write_text(json.dumps({"want": want, "meta": meta}, default=str))
    tr = S.sim_tracks(base_tr, meta, key)
    spot = L.Spot(ptz_spot.box, t_ref=ptz_spot.t_ref - t0, run=(0.0, t1 - t0), tids=ptz_spot.tids, name=ptz_spot.name,
                  stay=ptz_spot.stay, note="spot chosen on the clean footage")
    return mp4, tr, meta, spot


HEALTH_SET = ["retail_cctv", "yonge_dundas", "motorway_a40", "dashcam_urban", "rochester_night", "classroom",
              "people_detection", "store_aisle_detection", "worker_zone_detection", "one_by_one_person_detection"]


def health_check() -> dict[str, Any]:
    """False-alarm check of the blind-window detector on real, unmodified clips: any second flagged covered/frozen here
    is a false alarm (unless the clip really contains a still image)."""
    out: dict[str, Any] = {}
    for v in HEALTH_SET:
        if v not in pk.videos():
            continue
        F = L.features(pk.video_path(v), CACHE)
        h = health_only(F)
        out[v] = {"seconds": h["seconds"], "covered_or_flat_s": h["dark_or_flat_s"], "frozen_s": h["frozen_s"]}
        log("health", v, out[v])
    tot = sum(x["seconds"] for x in out.values())
    out["_total"] = {"clips": len(out), "seconds": tot, "false_alarm_s": sum(x["covered_or_flat_s"] + x["frozen_s"] for x in out.values()
                                                                           if "seconds" in x)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="default test set + simulated fault copies")
    ap.add_argument("--video", help="catalogue name, or sim_a / sim_b")
    ap.add_argument("--tid", type=int, help="watch the parked car that has this tracker id (default: auto)")
    ap.add_argument("--cars", type=int, default=0, help="cap on parked cars in the comparison table (0 = all candidates, max 40)")
    ap.add_argument("--refresh", action="store_true", help="recompute cached features/alignment")
    a = ap.parse_args()
    if a.refresh and CACHE.exists():
        for f in CACHE.glob("*"):
            f.unlink()
    names = DEFAULT if a.all else ([a.video] if a.video else [])
    if a.video == "health":
        names = []
    if not names and a.video != "health":
        ap.error("give --all or --video NAME")
    results: dict[str, Any] = {}
    rf = HERE / "results.json"
    if rf.exists() and not a.all:
        results = json.loads(rf.read_text(encoding="utf-8"))
    t0 = time.time()
    ptz_spot: L.Spot | None = None
    want_sims = [k for k in SIMS if a.all or a.video == k]
    order = [n for n in names if n in pk.videos()]
    if want_sims and "ptz_parking" not in order:
        order = ["ptz_parking"] + order
    for name in order:
        res, ctx = analyze(name, OUT / name, tid=a.tid if name == "ptz_parking" else None, n_cars=a.cars,
                           title=f"{pk.videos()[name]['title']}")
        results[name] = res
        if name == "ptz_parking" and ctx is not None:
            ptz_spot = ctx[0].spot
        log(name, "done", res.get("verdict", res.get("reason")))
    for key in want_sims:
        if ptz_spot is None:
            raise SystemExit("the simulated copies need the ptz_parking car selection")
        mp4, tr, meta, spot = build_sim(key, ptz_spot)
        res, _ = analyze(key, OUT / key, path=mp4, tr=tr, sim_meta=meta, spot=spot, n_cars=1, title=SIMS[key]["title"])
        results[key] = res
        log(key, "done", res["sim_eval"]["precision"], res["sim_eval"]["recall"])
    # the strongest metric (SIMULATED-fault kill test) goes on the main hero as a badge, once the sim results exist
    badge = kill_badge(results)
    if "ptz_parking" in HERO_CTX and badge:
        save_hero("ptz_parking", OUT / "ptz_parking", badge)
        results["ptz_parking"]["hero_badge"] = badge
    if a.all or a.video == "health":
        results["health_check"] = health_check()
    results["_meta"] = {"generated": time.strftime("%Y-%m-%d %H:%M"), "runtime_s": round(time.time() - t0, 1),
                        "thresholds": L.P, "note": "sim_* are SIMULATED fault copies of a real ptz_parking window"}
    pk.write_results(rf, results)
    log("all done in", round(time.time() - t0), "s ->", rf)


if __name__ == "__main__":
    main()
