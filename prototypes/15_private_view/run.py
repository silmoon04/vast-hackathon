"""PRIVATE VIEW: privacy-mask drift monitor for CCTV.

  python run.py --all                 default test set (ptz_parking real PTZ, rochester_night SIMULATED knock,
                                      lisbon_parking SIMULATED small shake)
  python run.py --video NAME          one catalogue clip (needs zones/NAME.json, else classical auto-zones)
  options: --shift dx,dy,rot  --shift-at t0,t1  --no-sim  --thr 5  --margin 0.006  --feat sift|orb  --no-overlay

Outputs per clip in out/<video>/: report.html, hero.jpg, overlay.mp4, exposure_log.csv, samples.csv, strip.png,
zones_setup.jpg;  plus results.json next to this file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402
import pview as pv  # noqa: E402
from pview_zones import classical_windows, cosmos_windows, poly_iou  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

CONFIGS: dict[str, dict[str, Any]] = {
    # real PTZ camera: no injection, no frame-exact ground truth (landmark eyeball check instead)
    "ptz_parking": dict(t_start=0.0, t_end=316.0, ref_t=5.0, sample_hz=1.5, overlay_speed=4.0, min_dur=0.6, merge_gap=2.0,
                        injection=None, camera="PTZ security camera, parking lot"),
    # calm 17 s of a hand-held night clip + SIMULATED camera knock 8.0-18.0 s (ground truth = the injected matrix)
    "rochester_night": dict(t_start=4.0, t_end=21.0, ref_t=10.0, sample_hz=5.0, overlay_speed=1.0, min_dur=0.3, merge_gap=0.5,
                            injection=dict(kind="step", t0=8.0, t1=18.0, dx=28.0, dy=12.0, rot=1.0), stabilise=True,
                            camera="Corporation Street, Rochester (night, flats)"),
    # held photo inside a photo slideshow + SIMULATED small decaying shake 532.2-533.4 s
    "lisbon_parking": dict(t_start=531.2, t_end=533.8, ref_t=531.2, sample_hz=25.0, overlay_speed=0.5, min_dur=0.08, merge_gap=0.3,
                           injection=dict(kind="shake", t0=532.2, t1=533.4, amp=14.0, hz=2.0),
                           camera="Lisbon street photo (slideshow hold, NOT a fixed camera)"),
}
CONFIGS["rochester_real"] = dict(
    # same clip, same zones, NO injection, NO stabilisation: the real hand-held drift as the camera really moved
    video="rochester_night", real=True, t_start=4.0, t_end=21.0, ref_t=10.0, sample_hz=5.0, overlay_speed=1.0, min_dur=0.3,
    merge_gap=0.5, injection=None, camera="Corporation Street, Rochester (night): REAL hand-held drift")
DEFAULT_SET = ["ptz_parking", "rochester_night", "rochester_real", "lisbon_parking"]


RAW_KNOCK: dict[str, Any] | None = None      # set by main(): the non-circular raw-clip knock test, shown on the Rochester hero


def tfmt(t: float, fine: bool = False) -> str:
    return pk.mmss(t) if fine else pk.mmss(t)[:5]


def sentence(e: dict[str, Any], fine: bool = False) -> str:
    d = e["duration_s"]
    ds = f"{d:.0f} s" if d >= 10 else f"{d:.1f} s"
    return f"{e['zone']} exposed {e['max_exposed_pct']:.0f}% for {ds} ({tfmt(e['start_s'], fine)}-{tfmt(e['end_s'], fine)})"


def group_events(evs: list[dict[str, Any]], tol: float = 0.45) -> list[dict[str, Any]]:
    """Merge per-zone EXPOSED events with (nearly) the same time span into one headline; keeps each window's own peak %."""
    groups: list[dict[str, Any]] = []
    for e in sorted((e for e in evs if e["kind"] == "EXPOSED"), key=lambda e: (e["start_s"], e["zone"])):
        for g in groups:
            if abs(g["start_s"] - e["start_s"]) <= tol and abs(g["end_s"] - e["end_s"]) <= tol:
                g["zones"].append(e["zone"])
                g["zone_pct"][e["zone"]] = e["max_exposed_pct"]
                g["max_exposed_pct"] = max(g["max_exposed_pct"], e["max_exposed_pct"])
                break
        else:
            groups.append({"zones": [e["zone"]], "zone_pct": {e["zone"]: e["max_exposed_pct"]}, "start_s": e["start_s"],
                           "end_s": e["end_s"], "duration_s": e["duration_s"], "max_exposed_pct": e["max_exposed_pct"]})
    for g in groups:
        z = g["zones"]
        g["zone"] = ", ".join(z) if len(z) <= 7 else f"{len(z)} windows"
    return groups


def group_sentence(g: dict[str, Any], fine: bool = False, max_items: int = 8) -> str:
    """'W7 37%, W6 34%, W1 9% exposed for 10 s (00:08.0-00:18.0)': every window's own peak, never one number for all."""
    items = sorted(g["zone_pct"].items(), key=lambda kv: -kv[1])
    if len({round(p) for _, p in items}) == 1 and len(items) > 1:
        lst = ", ".join(z for z, _ in items[:max_items]) + (f" +{len(items) - max_items} more" if len(items) > max_items else "") + f" {items[0][1]:.0f}%"
    else:
        lst = ", ".join(f"{z} {p:.0f}%" for z, p in items[:max_items]) + (f" +{len(items) - max_items} more" if len(items) > max_items else "")
    d = g["duration_s"]
    ds = f"{d:.0f} s" if d >= 10 else f"{d:.1f} s"
    return f"{lst} exposed for {ds} ({tfmt(g['start_s'], fine)}-{tfmt(g['end_s'], fine)})"


def logged_series(o: dict[str, Any]) -> dict[str, list[float]]:
    """Exposure series for the strips: a sample counts as exposed only inside a logged EXPOSED event, so a one-frame blip below the
    log's minimum duration is shown covered (as the log says) instead of as an unexplained amber cell."""
    out: dict[str, list[float]] = {}
    for z in o["ids"]:
        evs = [e for e in o["events"] if e["kind"] == "EXPOSED" and e["zone"] == z]
        vals = []
        for t, v in zip(o["ts"], o["series"][z]):
            inside = any(e["start_s"] - 1e-6 <= t < e["end_s"] - 1e-6 for e in evs)
            vals.append(v if (inside or math.isnan(v)) else 0.0)
        out[z] = vals
    return out


# ------------------------------------------------------------------------------------------ setup stage


def setup_stage(video: str, o: dict[str, Any], odir: Path, try_cosmos: bool) -> dict[str, Any]:  # noqa: C901
    """Zone proposals on the reference frame: Cosmos grounding -> classical fallback -> operator polygons (the zones used)."""
    ref = o["ref_frame"]
    h, w = ref.shape[:2]
    zones = o["zones"]
    info: dict[str, Any] = {"manual_zones": len(zones), "stand_in": all(z.stand_in for z in zones)}
    props_cl = classical_windows(ref)
    info["classical_proposals"] = len(props_cl)
    cos, raw, reason = ([], "", "skipped (stand-in zones are not windows)" if info["stand_in"] else "not re-run here (see rochester_night)")
    if try_cosmos and not info["stand_in"]:
        cos, raw, reason = cosmos_windows(ref)
    info["cosmos_status"] = reason
    info["cosmos_boxes"] = len(cos)
    info["cosmos_raw_head"] = raw[:160]

    def box_poly(b):
        return np.float32([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])

    def recall(props):
        if not props:
            return 0.0, 0
        hit = 0
        for z in zones:
            if max(poly_iou(z.poly, box_poly(p["box"]), (h, w)) for p in props) >= 0.3:
                hit += 1
        good = sum(1 for p in props if max(poly_iou(z.poly, box_poly(p["box"]), (h, w)) for z in zones) >= 0.3)
        return hit / len(zones), good

    info["classical_recall_iou0.3"], info["classical_precise"] = recall(props_cl)
    info["cosmos_recall_iou0.3"], info["cosmos_precise"] = recall(cos)
    vis = cv2.convertScaleAbs(ref, alpha=1.5, beta=8) if ref.mean() < 70 else ref.copy()
    for p in props_cl:
        cv2.rectangle(vis, (p["box"][0], p["box"][1]), (p["box"][2], p["box"][3]), (255, 200, 0), 2)
    for p in cos:
        cv2.rectangle(vis, (p["box"][0], p["box"][1]), (p["box"][2], p["box"][3]), (255, 0, 255), 2)
    for z in zones:
        cv2.polylines(vis, [np.int32(z.poly)], True, (60, 235, 70), 3, cv2.LINE_AA)
        x, y = z.poly.min(axis=0)
        cv2.putText(vis, z.id, (int(x) + 3, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 235, 255), 2, cv2.LINE_AA)
    n_hit = round(info["cosmos_recall_iou0.3"] * len(zones))
    if cos:
        cverdict = f"{len(cos)} box{'es' if len(cos) != 1 else ''}, {n_hit}/{len(zones)} on a window: " + ("usable" if n_hit >= 0.5 * len(zones) else "UNUSABLE")
    else:
        cverdict = reason
    info["cosmos_verdict"] = cverdict
    leg = (f"green = operator polygons used ({len(zones)})   cyan = classical proposals ({len(props_cl)})   "
           f"magenta = Cosmos grounding ({cverdict})")
    vis = pv.ptext(vis, [(leg, (10, h - 30), max(14, h // 40), (255, 255, 255), True)])
    pk.save_jpg(pk.fit(vis, 1000), odir / "zones_setup.jpg")
    return info


# ------------------------------------------------------------------------------------------ truth scoring


def score_truth(o: dict[str, Any], tol_t: float = 0.5, tol_pct: float = 10.0) -> dict[str, Any]:
    inj = o["injection"]
    ids, ts, dt = o["ids"], o["ts"], o["dt"]
    truth_ev = pv.make_events(ts, dt, o["truth"], [False] * len(ts), o["thr"], o["_min_dur"], o["_merge_gap"],
                              {z: [float("nan")] * len(ts) for z in ids})
    rows = pv.match_events(truth_ev, o["events"])
    per = []
    for r in rows:
        te, me = r["truth"], r["meas"]
        row: dict[str, Any] = {"zone": r["zone"], "truth_start": te["start_s"], "truth_end": te["end_s"],
                               "truth_max_pct": te["max_exposed_pct"], "truth_mean_pct": te["mean_exposed_pct"]}
        if me is None:
            row.update(detected=False, passed=False)
        else:
            ds, de = me["start_s"] - te["start_s"], me["end_s"] - te["end_s"]
            dmax = me["max_exposed_pct"] - te["max_exposed_pct"]
            dmean = me["mean_exposed_pct"] - te["mean_exposed_pct"]
            row.update(detected=True, meas_start=me["start_s"], meas_end=me["end_s"], meas_max_pct=me["max_exposed_pct"],
                       meas_mean_pct=me["mean_exposed_pct"], d_start_s=round(ds, 2), d_end_s=round(de, 2),
                       d_max_pp=round(dmax, 1), d_mean_pp=round(dmean, 1),
                       passed=bool(abs(ds) <= tol_t and abs(de) <= tol_t and abs(dmax) <= tol_pct and abs(dmean) <= tol_pct))
        per.append(row)
    truth_zones = {r["zone"] for r in rows}
    false_alarms = [e for e in o["events"] if e["kind"] == "EXPOSED" and e["zone"] not in truth_zones]
    # per-sample accuracy over every zone/sample
    errs = []
    for z in ids:
        for m, t in zip(o["series"][z], o["truth"][z]):
            if not (math.isnan(m) or math.isnan(t)):
                errs.append(abs(m - t))
    # injected-interval check against the exact injection times (not just the sampled truth series)
    out = {"truth_events": truth_ev, "per_zone": per, "false_alarm_events": false_alarms,
           "n_truth_zones": len(per), "n_pass": sum(1 for p in per if p["passed"]),
           "mean_abs_err_pp_all_samples": round(float(np.mean(errs)), 2) if errs else None,
           "max_abs_err_pp_all_samples": round(float(np.max(errs)), 1) if errs else None,
           "tol_s": tol_t, "tol_pp": tol_pct,
           "all_pass": bool(per) and all(p["passed"] for p in per) and not false_alarms}
    return out


# ------------------------------------------------------------------------------------------ hero image


def stamp_layer(size: tuple[int, int], title: str, sub: str, color=(214, 40, 40), angle: float = 5.0) -> Image.Image:
    f1, f2 = pv.font(34, True), pv.font(15, True)
    w1, w2 = f1.getlength(title), f2.getlength(sub)
    bw = int(max(w1, w2) + 28)
    bh = 34 + 15 + 30
    lay = Image.new("RGBA", (bw + 8, bh + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.rounded_rectangle((4, 4, bw + 4, bh + 4), 10, fill=(255, 255, 255, 215), outline=color + (255,), width=4)
    d.text((4 + (bw - w1) / 2, 4), title, font=f1, fill=color + (255,))
    d.text((4 + (bw - w2) / 2, 4 + 46), sub, font=f2, fill=color + (255,))
    return lay.rotate(angle, expand=True, resample=Image.BICUBIC)


def place_stamp(lay: Image.Image, panel_xy: tuple[int, int], panel_wh: tuple[int, int],
                zone_boxes: list[tuple[float, float, float, float]]) -> tuple[tuple[int, int], float]:
    """Pick the spot in the NOW panel where the stamp covers the least window area (exact: counts opaque stamp pixels inside each
    true window box, panel pixels). Returns (top-left in canvas px, covered window pixels)."""
    alpha = np.asarray(lay.split()[-1]) > 10
    lh, lw = alpha.shape
    pw_, ph_ = panel_wh
    cands = [(pw_ - lw - 6, ph_ - lh - 4), (6, ph_ - lh - 4), ((pw_ - lw) // 2, ph_ - lh - 4), (pw_ - lw - 6, 6), (6, 6)]
    best = None
    for cx, cy in cands:
        cov = 0.0
        for x0, y0, x1, y1 in zone_boxes:
            xa, xb = int(max(cx, x0)), int(min(cx + lw, x1))
            ya, yb = int(max(cy, y0)), int(min(cy + lh, y1))
            if xb > xa and yb > ya:
                cov += float(alpha[ya - cy:yb - cy, xa - cx:xb - cx].sum())
        if best is None or cov < best[1] - 1e-6:
            best = ((cx, cy), cov)
    (cx, cy), cov = best
    return (panel_xy[0] + cx, panel_xy[1] + cy), cov


def make_hero(o: dict[str, Any], cfg: dict[str, Any], path: Path, truth: dict[str, Any] | None) -> dict[str, Any]:
    W_, H_ = o["frame_size"]
    zones, thr, margin = o["zones"], o["thr"], o["margin_px"]
    pw = 620
    sc = pw / W_
    ph = int(round(H_ * sc))
    shape = (H_, W_)
    masks = pv.static_masks(zones, shape, margin)
    res_ref = pv.evaluate(zones, masks, np.eye(3), shape)
    ref_small = cv2.resize(o["ref_frame"], (pw, ph), interpolation=cv2.INTER_AREA)
    left = pv._draw_scaled(ref_small, zones, res_ref, margin, sc, shape, thr)
    ex_events = [e for e in o["events"] if e["kind"] == "EXPOSED"]
    un_events = [e for e in o["events"] if e["kind"] == "UNVERIFIED"]
    have_breach = bool(ex_events)
    fine = (o["t_end"] - o["t_start"]) < 30
    top = sorted(group_events(ex_events), key=lambda g: -(g["duration_s"] * g["max_exposed_pct"]))[:3]
    t_peak = o["ref_t"]
    right = left.copy()
    right_res = res_ref
    if have_breach:
        # NOW frame: the stored snapshot nearest to the middle of the most important breach (a settled moment, not a transition)
        g0 = top[0]
        tmid = 0.5 * (g0["start_s"] + g0["end_s"])
        snaps = o["snaps"]
        if snaps:
            inside = [sn for sn in snaps if g0["start_s"] - 1e-6 <= sn[0] < g0["end_s"]] or snaps
            score = lambda sn: sum(r["exposed"] for r in sn[2] if r["state"] == "ok" and r["id"] in g0["zones"])  # noqa: E731
            best = max(score(sn) for sn in inside)
            t_peak, sf, right_res = min((sn for sn in inside if score(sn) >= 0.9 * best), key=lambda sn: abs(sn[0] - tmid))
            right_src = cv2.resize(sf, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            t_peak, right_res = o["peak"]["t"], o["peak"]["res"]
            right_src = cv2.resize(o["peak"]["frame"], (pw, ph), interpolation=cv2.INTER_AREA)
        right = pv._draw_scaled(right_src, zones, right_res, margin, sc, shape, thr)
    # strip
    ids = o["ids"]
    n = len(ids)
    row_h = 22 if n <= 6 else 18
    t0, t1 = o["t_start"], o["t_end"]
    strip = pv.draw_strip(1180, row_h, ids, o["ts"], logged_series(o), o["states"], t1, None, thr, t0=t0, label_w=48)
    # headline events
    # assemble canvas
    top_h, lab_h = 92, 40
    gap = 12
    y_panels = top_h + lab_h
    y_strip_title = y_panels + ph + 36
    y_strip = y_strip_title + 46
    y_foot = y_strip + strip.shape[0] + 12
    rk = RAW_KNOCK if (truth and o["video"] == "rochester_night" and o.get("stabilised")) else None
    n_lines = max(1, len(top)) + (1 if un_events else 0) + (1 if (truth or o.get("xcheck")) else 0) + (1 if rk else 0)
    foot_h = 22 + 34 * n_lines + 34
    H = y_foot + foot_h
    cv = np.full((H, 1280, 3), (23, 17, 15), np.uint8)       # BGR of #0f1117-ish
    cv[:top_h] = (58, 36, 22)
    cv[y_panels:y_panels + ph, 14:14 + pw] = left
    cv[y_panels:y_panels + ph, 14 + pw + gap:14 + 2 * pw + gap] = right
    cv2.rectangle(cv, (14, y_panels), (14 + pw, y_panels + ph), (90, 200, 90), 3)
    cv2.rectangle(cv, (14 + pw + gap, y_panels), (14 + 2 * pw + gap, y_panels + ph), (60, 60, 235) if have_breach else (90, 200, 90), 3)
    cv[y_strip:y_strip + strip.shape[0], 50:50 + strip.shape[1]] = strip
    items: list = []
    items.append(("PRIVATE VIEW", (22, 12), 40, (255, 255, 255), True))
    items.append(("privacy-mask drift monitor for CCTV: are the masks still on the windows?", (24, 60), 17, (190, 205, 225), False))
    cam = cfg.get("camera", o["video"])
    items.append((cam, (1260 - pv.tw(cam, 17, True), 16), 17, (255, 255, 255), True))
    tag = ("SIMULATED camera knock: " + (f"{o['injection'].t0:.0f}-{o['injection'].t1:.0f} s" if o["injection"] else "")) if o["injection"] \
        else "REAL camera motion, no injection"
    items.append((tag, (1260 - pv.tw(tag, 15), 44), 15, (255, 205, 110) if o["injection"] else (150, 235, 150), False))
    if all(z.stand_in for z in zones):
        sn = "stand-in zones, not residential windows (none in this clip)"
        items.append((sn, (1260 - pv.tw(sn, 13), 68), 13, (200, 200, 200), False))
    if o["injection"] is not None:
        i0, i1 = o["injection"].t0, o["injection"].t1
        span = max(1e-6, t1 - t0)
        xa = 50 + 48 + int((i0 - t0) / span * (1172 - 48))
        xb = 50 + 48 + int((i1 - t0) / span * (1172 - 48))
        yb = y_strip - 6
        cv2.line(cv, (xa, yb), (xb, yb), (255, 205, 110), 2)
        cv2.line(cv, (xa, yb - 4), (xa, yb + 2), (255, 205, 110), 2)
        cv2.line(cv, (xb, yb - 4), (xb, yb + 2), (255, 205, 110), 2)
        lab = "injected knock (ground truth)"
        items.append((lab, (min(xb - pv.tw(lab, 12), xa + 4) if xb - xa > pv.tw(lab, 12) + 8 else xb + 6, yb - 17), 12, (255, 205, 110), False))
    out_img = cv
    # panel labels (pills)
    def pill(x, y, text, bg):
        nonlocal out_img
        wtxt = pv.tw(text, 18, True) + 22
        cv2.rectangle(out_img, (x, y), (x + wtxt, y + 30), bg, -1)
        items.append((text, (x + 11, y + 2), 18, (255, 255, 255), True))
    pill(14, top_h + 4, f"SETUP  {tfmt(o['ref_t'], fine)}  masks on windows", (60, 140, 50))
    if have_breach:
        pill(14 + pw + gap, top_h + 4, f"NOW  {tfmt(t_peak, fine)}  camera moved, masks did not", (40, 40, 205))
    else:
        pill(14 + pw + gap, top_h + 4, f"NOW  {tfmt(t_peak, fine)}  masks still aligned", (60, 140, 50))
    items.append(("red = static privacy mask      green = where the window really is      yellow = window pixels left exposed",
                  (14, y_panels + ph + 8), 15, (200, 200, 200), False))
    items.append(("EXPOSURE LOG  (one row per privacy zone; green = covered, amber to red = % of window exposed in a logged event, grey = view not recognised)",
                  (14, y_strip_title + 6), 16, (255, 255, 255), True))
    y = y_foot + 8
    if top:
        for e in top:
            txt = "■ " + group_sentence(e, fine)
            fs_ = next((f for f in (26, 24, 22, 20, 18) if pv.tw(txt, f, True) <= 1244), 18)
            items.append((txt, (18, y + (26 - fs_) // 2), fs_, (255, 120, 110), True))
            y += 34
    else:
        items.append(("■ No window exposed above %.0f%%" % thr, (18, y), 26, (140, 235, 140), True)); y += 34
    for e in un_events[:1]:
        items.append((f"■ Camera view not recognised for {e['duration_s']:.0f} s ({pk.mmss(e['start_s'])[:5]}-{pk.mmss(e['end_s'])[:5]}): "
                      "masks cannot be verified, treat as exposed", (18, y), 20, (190, 190, 190), True))
        y += 34
    if truth:
        inj = o["injection"]
        tp = truth["per_zone"]
        if tp:
            ds = max(abs(p.get("d_start_s", 9)) for p in tp)
            de = max(abs(p.get("d_end_s", 9)) for p in tp)
            dm = max(abs(p.get("d_max_pp", 99)) for p in tp)
            msg = (f"GROUND TRUTH (simulated knock {inj.t0:.1f}-{inj.t1:.1f} s{', stabilised frames' if o.get('stabilised') else ''}): "
                   f"start err <= {ds:.2f} s, end err <= {de:.2f} s, peak err <= {dm:.1f} pp, {len(tp)} windows  ->  "
                   f"{'PASS' if truth['all_pass'] else 'FAIL'}")
        else:
            msg = "GROUND TRUTH: no window exposed by the injected shift"
        fs2 = next((f for f in (18, 17, 16, 15, 14) if pv.tw(msg, f, True) <= 1244), 14)
        items.append((msg, (18, y), fs2, (150, 235, 150) if truth["all_pass"] else (110, 110, 255), True))
        y += 34
        if rk:
            kp = [p for p in rk["per_event"] if p["knock"]]
            m2 = (f"NON-CIRCULAR TEST on the raw clip (knock {rk['injection']['t0']:.0f}-{rk['injection']['t1']:.0f} s, truth from an independent estimator): "
                  f"{rk['n_knock_pass']}/{rk['n_knock_events']} windows pass, start/end err <= "
                  f"{max(max(abs(p['d_start_s']), abs(p['d_end_s'])) for p in kp):.2f} s, peak err <= {max(abs(p['d_max_pp']) for p in kp):.1f} pp")
            items.append((m2, (18, y), 16, (150, 235, 150) if rk["all_pass"] else (110, 110, 255), True))
            y += 34
    elif o.get("xcheck"):
        xc = o["xcheck"]
        msg = (f"NO GROUND TRUTH (real motion). Independent ECC estimator agrees: windows within {xc['corner_err_px_mean']:.1f} px on average "
               f"(max {xc['corner_err_px_max']:.1f}), exposure within {xc['exposure_err_pp_mean']:.1f} pp (max {xc['exposure_err_pp_max']:.1f})")
        items.append((msg, (18, y), 17, (150, 235, 150), True))
        y += 34
    out_img = pv.ptext(out_img, items)
    # stamp on the right panel, placed where it hides the least window area
    stamp_info: dict[str, Any] = {}
    if have_breach:
        e0 = top[0]
        nz = len(e0["zones"])
        wz = max(e0["zone_pct"], key=e0["zone_pct"].get)
        sub = (f"{nz} windows exposed, worst {wz} {e0['zone_pct'][wz]:.0f}%" if nz > 1 else f"{wz} exposed {e0['zone_pct'][wz]:.0f}%")
        lay = stamp_layer((pw, ph), "BREACH", sub)
    else:
        lay = stamp_layer((pw, ph), "ALL CLEAR", "masks aligned", color=(36, 140, 56))
    wboxes = []
    for r in right_res:
        if r["poly"] is not None and r["state"] != "offframe":
            p_ = r["poly"] * sc
            wboxes.append((float(p_[:, 0].min()), float(p_[:, 1].min()), float(p_[:, 0].max()), float(p_[:, 1].max())))
    (px, py), cov = place_stamp(lay, (14 + pw + gap, y_panels), (pw, ph), wboxes)
    stamp_info = {"stamp_covered_window_px": round(cov, 1), "stamp_xy": [px, py]}
    im = Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    im.alpha_composite(lay, (px, py))
    out_img = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    assert out_img.shape[1] <= 1280
    pk.save_jpg(out_img, path, q=90)
    return stamp_info


# -------------------------------------------------------------------------------------------- one clip


def make_injection(cfg: dict[str, Any], args: argparse.Namespace) -> pv.Injection | None:
    if args.no_sim or cfg.get("real"):
        return None
    d = dict(cfg.get("injection") or {})
    if args.shift or args.shift_at:
        d = d or dict(kind="step", t0=cfg["t_start"] + 4, t1=cfg["t_start"] + 14)
        d["kind"] = "step"
        if args.shift:
            dx, dy, rot = (float(v) for v in args.shift.split(","))
            d.update(dx=dx, dy=dy, rot=rot)
        if args.shift_at:
            d["t0"], d["t1"] = (float(v) for v in args.shift_at.split(","))
    return pv.Injection(**d) if d else None


def auto_zones(video: str) -> None:
    """No zones file for this clip: classical window proposals on a frame at 1 s become zones (stand-ins)."""
    f = pk.frame_at(video, 1.0)
    props = classical_windows(f)[:6] or [{"box": [f.shape[1] // 3, f.shape[0] // 3, f.shape[1] // 2, f.shape[0] // 2]}]
    zs = [{"id": f"A{i + 1}", "name": f"auto zone {i + 1}", "stand_in": True,
           "poly": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]} for i, p in enumerate(props) for b in [p["box"]]]
    (HERE / "zones").mkdir(exist_ok=True)
    (HERE / "zones" / f"{video}.json").write_text(json.dumps({"video": video, "ref_t": 1.0, "note": "auto zones from the classical detector",
                                                              "zones": zs}), encoding="utf-8")


def ecc_crosscheck(vid: str, o: dict[str, Any]) -> dict[str, Any]:
    """Real footage has no ground truth, so compare the monitor (SIFT+RANSAC) with an independent intensity-based estimator (ECC)
    on the same sampled frames: where do the true window outlines land, and how much exposure does each say."""
    from pview_motion import EccTracker
    ecc = EccTracker(o["ref_frame"])
    W, Hh = o["frame_size"]
    masks = pv.static_masks(o["zones"], (Hh, W), o["margin_px"])
    corner, dexp, n = [], [], 0
    series_e = {z: [] for z in o["ids"]}
    for k, (i, t, fr) in enumerate(pk.iter_frames(vid, stride=o["stride"], start=o["t_start"], end=o["t_end"])):
        if k >= len(o["Hs"]):
            break
        He = ecc.update(fr)
        Hs = o["Hs"][k]
        res_e = pv.evaluate(o["zones"], masks, He, (Hh, W)) if He is not None else None
        for j, z in enumerate(o["ids"]):
            series_e[z].append(res_e[j]["exposed"] if res_e else float("nan"))
        if He is None or Hs is None:
            continue
        for zz in o["zones"]:
            corner.append(float(np.linalg.norm(pv.warp_poly(zz.poly, Hs) - pv.warp_poly(zz.poly, He), axis=1).mean()))
        res_s = pv.evaluate(o["zones"], masks, Hs, (Hh, W))
        for a_, b_ in zip(res_s, res_e):
            if a_["state"] == "ok" and b_["state"] == "ok":
                dexp.append(abs(a_["exposed"] - b_["exposed"]))
        n += 1
    ev_e = pv.make_events(o["ts"], o["dt"], series_e, [False] * len(o["ts"]), o["thr"], o["_min_dur"], o["_merge_gap"],
                          {z: [float("nan")] * len(o["ts"]) for z in o["ids"]})
    ev_s = [e for e in o["events"] if e["kind"] == "EXPOSED"]
    rows = pv.match_events(ev_e, ev_s)
    ds = [abs(r["meas"]["start_s"] - r["truth"]["start_s"]) for r in rows if r["meas"]]
    de = [abs(r["meas"]["end_s"] - r["truth"]["end_s"]) for r in rows if r["meas"]]
    dm = [abs(r["meas"]["max_exposed_pct"] - r["truth"]["max_exposed_pct"]) for r in rows if r["meas"]]
    return {"estimator_b": "ECC homography (intensity based, top 62% of the frame, no feature matching)", "samples": n,
            "corner_err_px_mean": round(float(np.mean(corner)), 2) if corner else None,
            "corner_err_px_max": round(float(np.max(corner)), 2) if corner else None,
            "exposure_err_pp_mean": round(float(np.mean(dexp)), 2) if dexp else None,
            "exposure_err_pp_max": round(float(np.max(dexp)), 1) if dexp else None,
            "events_ecc": len(ev_e), "events_sift": len(ev_s), "events_matched": sum(1 for r in rows if r["meas"]),
            "event_start_err_s_max": round(max(ds), 2) if ds else None, "event_end_err_s_max": round(max(de), 2) if de else None,
            "event_peak_err_pp_max": round(max(dm), 1) if dm else None,
            "note": "two independent estimators agreeing is evidence, not ground truth: both could be wrong the same way "
                    "(e.g. if the building itself moved), which is why the hero frames were also checked by eye"}


def run_video(name: str, args: argparse.Namespace) -> dict[str, Any]:
    t_run = time.time()
    cfg0 = CONFIGS.get(name) or {}
    vid = cfg0.get("video", name)               # catalogue clip (several outputs can use the same clip)
    if not (HERE / "zones" / f"{vid}.json").exists():
        auto_zones(vid)
    cfg = dict(cfg0 or dict(t_start=0.0, t_end=min(60.0, pk.info(vid)["duration"]), ref_t=1.0, sample_hz=8.0,
                            overlay_speed=1.0, min_dur=0.4, merge_gap=0.6, injection=None, camera=name))
    inj = make_injection(cfg, args)
    odir = OUT / name
    odir.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] running {vid} {cfg['t_start']:.0f}-{cfg['t_end']:.0f} s, sample {cfg['sample_hz']:g} Hz, "
          f"{'SIMULATED ' + inj.kind if inj else 'no injection'}", flush=True)
    o = pv.process(vid, t_start=cfg["t_start"], t_end=cfg["t_end"], ref_t=cfg["ref_t"], injection=inj,
                   sample_hz=cfg["sample_hz"], margin_frac=args.margin, thr=args.thr, min_dur=cfg["min_dur"],
                   merge_gap=cfg["merge_gap"], feat=args.feat, out_dir=odir, overlay=not args.no_overlay,
                   stabilise=bool(cfg.get("stabilise")) and inj is not None,
                   overlay_speed=cfg["overlay_speed"], progress=True)
    o["_min_dur"], o["_merge_gap"] = cfg["min_dur"], cfg["merge_gap"]
    t_proc = time.time() - t_run
    setup = setup_stage(vid, o, odir, try_cosmos=(name == "rochester_night"))
    truth = score_truth(o) if inj else None
    xcheck = None
    if cfg.get("real") and vid == "rochester_night":
        xcheck = ecc_crosscheck(vid, o)
        o["xcheck"] = xcheck
    pv.write_csv(odir / "exposure_log.csv", o["events"])
    pv.write_samples_csv(odir / "samples.csv", o)
    stamp = make_hero(o, cfg, odir / "hero.jpg", truth)
    # strip image for the report
    strip = pv.draw_strip(1200, 24 if len(o["ids"]) <= 6 else 18, o["ids"], o["ts"], logged_series(o), o["states"], o["t_end"], None,
                          o["thr"], t0=o["t_start"], label_w=52)
    pk.save_png(strip, odir / "strip.png")
    metrics = collect_metrics(name, cfg, o, truth, setup, t_proc)
    metrics["hero"] = stamp
    if xcheck:
        metrics["ecc_crosscheck"] = xcheck
    if name == "ptz_parking":
        metrics["landmark_check"] = ptz_landmark_check(o)
    metrics["vastdb_demo"] = vastdb_demo(o)
    build_report(name, vid, cfg, o, truth, setup, metrics, odir)
    metrics["runtime_s"] = round(time.time() - t_run, 1)
    print(f"[{name}] done in {metrics['runtime_s']} s: {len(o['events'])} events", flush=True)
    return metrics


def collect_metrics(name: str, cfg: dict[str, Any], o: dict[str, Any], truth: dict[str, Any] | None, setup: dict[str, Any],
                    t_proc: float) -> dict[str, Any]:
    ts, dt = o["ts"], o["dt"]
    inl = [r["inliers"] for r in o["tracker"] if r["src"] != "lost"]
    lost_s = sum(dt for x in o["lost"] if x)
    ex_ev = [e for e in o["events"] if e["kind"] == "EXPOSED"]
    per_zone = {}
    for z in o["ids"]:
        v = o["series"][z]
        flagged = [(not math.isnan(a)) and a >= o["thr"] for a in v]
        okv = [a for a in v if not math.isnan(a)]
        per_zone[z] = {"exposed_s": round(sum(dt for f in flagged if f), 2), "max_exposed_pct": round(max(okv), 1) if okv else None,
                       "max_misalign_px": round(float(np.nanmax(o["mis"][z])), 1) if not all(math.isnan(x) for x in o["mis"][z]) else None}
    m: dict[str, Any] = {
        "video": cfg.get("video", name), "camera": cfg.get("camera"), "segment_s": [o["t_start"], o["t_end"]], "ref_t": o["ref_t"],
        "samples": len(ts), "sample_hz": o["sample_hz"], "mask_margin_px": round(o["margin_px"], 1), "exposed_threshold_pct": o["thr"],
        "features": o.get("feat", "sift"),
        "tracker_lost_s": round(lost_s, 2), "tracker_lost_pct": round(100 * sum(o["lost"]) / max(1, len(ts)), 1),
        "median_inliers": int(np.median(inl)) if inl else 0, "per_zone": per_zone,
        "events": o["events"], "n_exposed_events": len(ex_ev), "total_exposed_zone_seconds": round(sum(e["duration_s"] for e in ex_ev), 1),
        "worst_event": sentence(max(ex_ev, key=lambda e: e["duration_s"] * (e["max_exposed_pct"] or 0)), (o["t_end"] - o["t_start"]) < 30) if ex_ev else None,
        "setup": setup, "process_s": round(t_proc, 1),
        "simulated": bool(o["injection"]),
        "stabilised_before_injection": bool(o.get("stabilised")), "tracker_stats": o.get("tracker_stats"),
    }
    rp_ = o.get("real_pass")
    if rp_:
        m["real_drift_no_injection"] = {"events": rp_["events"], "max_exposed_pct": round(rp_["max_exposed_pct"], 1),
                                        "max_misalign_px": round(rp_["max_misalign_px"], 1), "lost_samples": rp_["lost_samples"]}
    if o["injection"]:
        i = o["injection"]
        m["injection"] = {k: getattr(i, k) for k in ("kind", "t0", "t1", "dx", "dy", "rot", "amp", "hz")}
        tp = truth["per_zone"] if truth else []
        m["truth"] = {k: v for k, v in truth.items() if k != "truth_events"} if truth else None
        m["truth"]["truth_events"] = truth["truth_events"] if truth else None
    return m


# ------------------------------------------------------------------------------------------- report html


def build_report(name: str, vid: str, cfg: dict[str, Any], o: dict[str, Any], truth: dict[str, Any] | None, setup: dict[str, Any],
                 m: dict[str, Any], odir: Path) -> None:
    rp = pk.Report(odir / "report.html", f"Private View: {name}",
                   "Do the privacy masks still sit on the windows? Static masks (red) vs the true window position (green) as the camera moves.")
    rp.credit(vid)
    rp.image(odir / "hero.jpg", "Hero: setup frame vs worst moment, exposure log and verdict.")
    ex = [e for e in o["events"] if e["kind"] == "EXPOSED"]
    rp.kv({"windows watched": len(o["ids"]), "breach events": len(ex),
           "worst": (sentence(max(ex, key=lambda e: e["duration_s"] * e["max_exposed_pct"]), (o["t_end"] - o["t_start"]) < 30) if ex else "none"),
           "view lost": f"{m['tracker_lost_s']:.0f} s ({m['tracker_lost_pct']}%)", "median inliers": m["median_inliers"],
           "mask margin": f"{m['mask_margin_px']} px"})
    if (odir / "overlay.mp4").exists():
        rp.h2("Overlay video")
        rp.video(odir / "overlay.mp4", "red = static mask, green = true window position, yellow = exposed pixels; exposure strip below.")
    rp.h2("Exposure log")
    if o["events"]:
        rp.table(o["events"], ["zone", "kind", "start_s", "end_s", "duration_s", "max_exposed_pct", "mean_exposed_pct",
                               "max_misalign_px", "frames"])
    else:
        rp.text("No exposure events.")
    rp.image(odir / "strip.png", "Exposure strip per zone (green covered, amber to red exposed %, grey view not recognised).")
    if truth:
        inj = o["injection"]
        rp.h2("Ground truth check (SIMULATED injection)")
        rp.text(f"Injected {inj.kind} {inj.t0:.1f}-{inj.t1:.1f} s on real frames; truth = the injected matrix, same raster maths. "
                f"Tolerance: start/end {truth['tol_s']} s, exposure {truth['tol_pp']} percentage points. "
                f"Result: {truth['n_pass']}/{truth['n_truth_zones']} windows pass, {len(truth['false_alarm_events'])} false alarms, "
                f"mean per-sample error {truth['mean_abs_err_pp_all_samples']} pp.")
        rp.table(truth["per_zone"], ["zone", "truth_start", "truth_end", "truth_max_pct", "meas_start", "meas_end", "meas_max_pct",
                                     "d_start_s", "d_end_s", "d_max_pp", "d_mean_pp", "passed"])
    if m.get("real_drift_no_injection"):
        rd = m["real_drift_no_injection"]
        rp.h2("Real hand-held drift in the same segment (pass 1, no injection)")
        rp.text(f"The Rochester clip is hand-held. Measured on the raw frames: windows drift up to {rd['max_misalign_px']} px from their "
                f"reference position and up to {rd['max_exposed_pct']}% of a window is exposed (REAL, no ground truth). "
                "The simulated test above is run on frames stabilised with this estimate, so the injected knock is the only motion.")
        if rd["events"]:
            rp.table(rd["events"], ["zone", "kind", "start_s", "end_s", "duration_s", "max_exposed_pct", "max_misalign_px"])
    if m.get("ecc_crosscheck"):
        xc = m["ecc_crosscheck"]
        rp.h2("Real motion, no ground truth: cross-check with an independent estimator")
        rp.text(f"Same frames, second estimator ({xc['estimator_b']}). True window outlines agree within {xc['corner_err_px_mean']} px on average "
                f"(max {xc['corner_err_px_max']} px); exposure agrees within {xc['exposure_err_pp_mean']} pp on average (max {xc['exposure_err_pp_max']} pp); "
                f"{xc['events_matched']}/{xc['events_sift']} logged events match an ECC event, start/end within {xc['event_start_err_s_max']}/"
                f"{xc['event_end_err_s_max']} s, peak within {xc['event_peak_err_pp_max']} pp. {xc['note']}.")
    if m.get("landmark_check"):
        lc = m["landmark_check"]
        rp.h2("Eyeball landmark check (real PTZ move)")
        rp.text(f"4 roof landmarks read by eye at t={lc.get('t')} s vs the warped reference positions: errors {lc.get('err_px')} px "
                f"(max {lc.get('max_err_px')} px). {lc.get('note', '')}")
    rp.h2("Zone setup")
    rp.text(f"Operator polygons: {setup['manual_zones']}. Classical proposals: {setup['classical_proposals']} "
            f"(recall at IoU 0.3: {setup['classical_recall_iou0.3']:.0%}). Cosmos grounding: {setup.get('cosmos_verdict', setup['cosmos_status'])} "
            f"(recall {setup['cosmos_recall_iou0.3']:.0%}).")
    rp.image(odir / "zones_setup.jpg", "Reference frame: operator polygons (green), classical proposals (cyan), Cosmos boxes (magenta).")
    rp.text(o["zinfo"].get("note", ""))
    rp.save()


# ------------------------------------------------------------------------------------------------ VAST stand-ins


def seg_exposure_table(o: dict[str, Any], seg: float = 5.0):
    """5 s segment table (like the DataEngine segments) with the worst exposure per zone and an 'unverified' flag."""
    import pandas as pd
    ts = np.array(o["ts"])
    rows = []
    for k in range(int(np.ceil((o["t_end"] - o["t_start"]) / seg))):
        a = o["t_start"] + k * seg
        idx = np.where((ts >= a) & (ts < a + seg))[0]
        row: dict[str, Any] = {"seg": k, "seg_start": a, "seg_end": a + seg, "unverified": int(any(o["lost"][i] for i in idx))}
        for z in o["ids"]:
            v = [o["series"][z][i] for i in idx if not math.isnan(o["series"][z][i])]
            row[f"max_exposed_{z}"] = round(max(v), 1) if v else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def vastdb_demo(o: dict[str, Any]) -> dict[str, Any]:
    """The exposure log as a VastDB table; SQL restricted to the subset VastDB accepts (no GROUP BY / IN / JOIN)."""
    df = seg_exposure_table(o)
    con = pk.vastdb({"seg_exposure": df})
    z0 = o["ids"][0]
    cond = " OR ".join(f"max_exposed_{z} >= 50" for z in o["ids"])
    sql = f"SELECT seg, seg_start, seg_end FROM seg_exposure WHERE {cond} OR unverified = 1 ORDER BY seg"
    rows = con.execute(sql).fetchall()
    return {"sql": sql, "rows": len(rows), "segments_total": len(df), "first": rows[0] if rows else None}


def embed_view_check(video: str = "ptz_parking", seg: float = 5.0, max_segments: int = 70) -> dict[str, Any]:
    """Cross-check with Cosmos-Embed1 (local stand-in): cosine similarity of every 5 s segment to the setup segment.
    A moved PTZ should show up as a similarity drop without any geometry."""
    from concurrent.futures import ThreadPoolExecutor
    res = json.loads((HERE / "results.json").read_text(encoding="utf-8"))["videos"][video]
    d = pk.info(video)
    n = min(max_segments, int(d["duration"] // seg))
    tmp = OUT / "_scratch" / "emb"
    tmp.mkdir(parents=True, exist_ok=True)
    clips = []
    for k in range(n):
        p = tmp / f"{video}_{k:03d}.mp4"
        if not p.exists():
            pk.cut(video, k * seg, (k + 1) * seg, p, width=480)
        clips.append(p)
    embs = []
    t0 = time.time()
    for i in range(0, n, 6):
        embs.append(pk.embed_video(clips[i:i + 6]))
    E = np.vstack(embs)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = E @ E[1]          # reference = the 5-10 s segment (setup frame at 5 s)
    ev = res["events"]
    state = []
    for k in range(n):
        a, b = k * seg, (k + 1) * seg
        un = any(e["kind"] == "UNVERIFIED" and e["start_s"] < b and e["end_s"] > a for e in ev)
        ex = any(e["kind"] == "EXPOSED" and e["start_s"] < b and e["end_s"] > a for e in ev)
        state.append("unverified" if un else ("exposed" if ex else "aligned"))
    al = [float(s) for s, st in zip(sim, state) if st == "aligned"]
    mv = [float(s) for s, st in zip(sim, state) if st != "aligned"]
    thr = (min(al) + max(mv)) / 2 if al and mv else None
    return {"segments": n, "seconds": round(time.time() - t0, 1), "aligned_sim_min": round(min(al), 3) if al else None,
            "aligned_sim_mean": round(float(np.mean(al)), 3) if al else None, "moved_sim_max": round(max(mv), 3) if mv else None,
            "moved_sim_mean": round(float(np.mean(mv)), 3) if mv else None,
            "separable": bool(al and mv and min(al) > max(mv)),
            "sim_by_segment": [round(float(s), 3) for s in sim], "state_by_segment": state,
            "note": "similarity to the 5-10 s segment; 'moved' = segments overlapping an EXPOSED/UNVERIFIED event of the geometric pipeline"}


# ------------------------------------------------------------------------------------------------ checks

# EYEBALL landmarks (read by eye from 4x crops of the raw frames): reference frame t=5.0 s -> frame at t=280 s (after the PTZ move).
# chimney tops of the hotel roof, right end of the roof, right gable apex.
PTZ_LM_REF = np.float32([[48.7, 139.5], [222.5, 136.0], [286.0, 146.0], [241.0, 156.0]])
PTZ_LM_280 = np.float32([[584.5, 11.0], [756.0, 9.0], [819.0, 20.0], [774.0, 27.5]])


def ptz_landmark_check(o: dict[str, Any], t_check: float = 280.0) -> dict[str, Any]:
    k = int(np.argmin(np.abs(np.array(o["ts"]) - t_check)))
    H = o["Hs"][k]
    if H is None:
        return {"t": o["ts"][k], "error": "view lost at the check time"}
    p = pv.warp_poly(PTZ_LM_REF, H)
    e = np.linalg.norm(p - PTZ_LM_280, axis=1)
    return {"t": o["ts"][k], "n_landmarks": 4, "err_px": [round(float(x), 1) for x in e], "max_err_px": round(float(e.max()), 1),
            "mean_err_px": round(float(e.mean()), 1), "camera_shift_px": [round(float(v), 0) for v in (H[0, 2], H[1, 2])],
            "note": "landmarks read by eye (+-2 px); the shift is a real PTZ move, there is no frame-exact truth"}


def run_checks() -> dict[str, Any]:
    """Cheap accuracy checks on single frames: SIMULATED shifts of growing size on the Rochester reference frame, SIFT vs ORB."""
    from pview_motion import MotionTracker
    video = "rochester_night"
    d, zones = pv.load_zones(video)
    ref = pk.frame_at(video, d["ref_t"])
    h, w = ref.shape[:2]
    margin = max(3.0, 0.006 * w)
    masks = pv.static_masks(zones, (h, w), margin)
    out: dict[str, Any] = {"shift_sweep": [], "note": "SIMULATED shifts applied to the reference frame itself (no real motion, no moving objects): "
                           "an idealised upper bound on accuracy"}
    sweep = [(6, 3, 0.2), (14, 6, 0.5), (28, 12, 1.0), (45, 20, 1.5), (70, 30, 2.0), (-35, -15, -1.0), (-90, 40, -2.5)]
    for dx, dy, rot in sweep:
        M = pv.Injection("step", 0, 1, dx=dx, dy=dy, rot=rot).matrix(0.5, w, h)
        cur = pv.inject(ref, M)
        truth = pv.evaluate(zones, masks, M, (h, w))
        row: dict[str, Any] = {"dx": dx, "dy": dy, "rot": rot, "truth_max_pct": round(max(t["exposed"] for t in truth if t["state"] == "ok"), 1)}
        for kind in ("sift", "orb"):
            r = MotionTracker(ref, kind, min_inliers=30).update(cur)
            if r["H"] is None:
                row[kind] = {"lost": True}
                continue
            res = pv.evaluate(zones, masks, r["H"], (h, w))
            errs = [abs(a["exposed"] - b["exposed"]) for a, b in zip(res, truth) if a["state"] == "ok" and b["state"] == "ok"]
            pc = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            ce = np.linalg.norm(pv.warp_poly(pc, r["H"]) - pv.warp_poly(pc, M), axis=1).mean()
            row[kind] = {"corner_err_px": round(float(ce), 2), "mean_abs_err_pp": round(float(np.mean(errs)), 2),
                         "max_abs_err_pp": round(float(np.max(errs)), 1), "inliers": r["inliers"]}
        out["shift_sweep"].append(row)
    out["raw_knock"] = RAW_KNOCK or raw_knock_check()
    return out


def raw_knock_check(video: str = "rochester_night", tol_t: float = 0.5, tol_pp: float = 10.0) -> dict[str, Any]:
    """NON-CIRCULAR kill test. The knock is injected into the RAW hand-held clip (no stabilisation by the monitor's own estimate).
    The raw clip already drifts for real, so truth for the knocked frame is  M_knock @ H_real  where H_real comes from a different
    algorithm (ECC, no feature matching) than the monitor (SIFT+RANSAC). The monitor only sees the knocked pixels."""
    from pview_motion import EccTracker, MotionTracker
    cfg = CONFIGS["rochester_night"]
    # a different knock from the stabilised test (other direction, 8-14 s): the real hand-held drift is calm there (< 5 % exposed
    # before 6.4 s and after 15.8 s), so the logged event ends where the knock ends instead of merging into drift
    inj = pv.Injection(kind="step", t0=8.0, t1=14.0, dx=-26.0, dy=14.0, rot=-1.2)
    _, zones = pv.load_zones(video)
    meta = pk.info(video)
    W, Hh = meta["w"], meta["h"]
    margin = max(3.0, 0.006 * W)
    masks = pv.static_masks(zones, (Hh, W), margin)
    ids = [z.id for z in zones]
    ref = pk.frame_at(video, cfg["ref_t"])
    tracker, ecc = MotionTracker(ref), EccTracker(ref)
    stride = max(1, int(round(meta["fps"] / cfg["sample_hz"])))
    dt = stride / meta["fps"]
    ts: list[float] = []
    ser = {z: [] for z in ids}      # measured (monitor sees the knocked pixels only)
    tru = {z: [] for z in ids}      # truth = M_knock @ H_real(ECC)
    nai = {z: [] for z in ids}      # naive truth = M_knock only (ignores the real drift)
    mis = {z: [] for z in ids}
    lost: list[bool] = []
    t0 = time.time()
    for i, t, fr0 in pk.iter_frames(video, stride=stride, start=cfg["t_start"], end=cfg["t_end"]):
        He = ecc.update(fr0)
        M = inj.matrix(t, W, Hh)
        r = tracker.update(pv.inject(fr0, M))
        ts.append(round(t, 3))
        lost.append(r["H"] is None)
        rm = pv.evaluate(zones, masks, r["H"], (Hh, W))
        rt = pv.evaluate(zones, masks, M @ He if He is not None else None, (Hh, W))
        rn = pv.evaluate(zones, masks, M, (Hh, W))
        for z, a_, b_, c_ in zip(ids, rm, rt, rn):
            ser[z].append(a_["exposed"]); tru[z].append(b_["exposed"]); nai[z].append(c_["exposed"]); mis[z].append(a_["mis"])
    kw = dict(thr=5.0, min_dur=cfg["min_dur"], merge_gap=cfg["merge_gap"])
    nanm = {z: [float("nan")] * len(ts) for z in ids}
    ev_m = pv.make_events(ts, dt, ser, lost, kw["thr"], kw["min_dur"], kw["merge_gap"], mis)
    ev_t = pv.make_events(ts, dt, tru, [False] * len(ts), kw["thr"], kw["min_dur"], kw["merge_gap"], nanm)
    rows = pv.match_events(ev_t, ev_m)
    per = []
    for r in rows:
        te, me = r["truth"], r["meas"]
        row: dict[str, Any] = {"zone": r["zone"], "truth_start": te["start_s"], "truth_end": te["end_s"], "truth_max_pct": te["max_exposed_pct"],
                               "knock": bool(te["start_s"] < inj.t1 and te["end_s"] > inj.t0)}
        if me is None:
            row.update(detected=False, passed=False)
        else:
            ds, de, dm = me["start_s"] - te["start_s"], me["end_s"] - te["end_s"], me["max_exposed_pct"] - te["max_exposed_pct"]
            row.update(detected=True, meas_start=me["start_s"], meas_end=me["end_s"], meas_max_pct=me["max_exposed_pct"],
                       d_start_s=round(ds, 2), d_end_s=round(de, 2), d_max_pp=round(dm, 1),
                       passed=bool(abs(ds) <= tol_t and abs(de) <= tol_t and abs(dm) <= tol_pp))
        per.append(row)
    kper = [p for p in per if p["knock"]]
    false_alarms = [e for e in ev_m if e["kind"] == "EXPOSED" and not any(
        t_["zone"] == e["zone"] and t_["start_s"] < e["end_s"] and t_["end_s"] > e["start_s"] for t_ in ev_t)]
    errs = [abs(m - t) for z in ids for m, t in zip(ser[z], tru[z]) if not (math.isnan(m) or math.isnan(t))]
    errs_naive = [abs(m - t) for z in ids for m, t in zip(ser[z], nai[z]) if not (math.isnan(m) or math.isnan(t))]
    # the knock itself: samples inside the injected window only
    k_in = [k for k, t in enumerate(ts) if inj.t0 <= t < inj.t1]
    errs_in = [abs(ser[z][k] - tru[z][k]) for z in ids for k in k_in if not (math.isnan(ser[z][k]) or math.isnan(tru[z][k]))]
    # a partial exposure that is NOT 100%: the real-drift + knock peaks per window
    peaks = {z: round(max([v for v in tru[z] if not math.isnan(v)] or [0.0]), 1) for z in ids}
    return {"design": "knock injected into the RAW unstabilised clip; truth = M_knock @ H_real, H_real from ECC (independent algorithm)",
            "t_start_end_s": [cfg["t_start"], cfg["t_end"]], "injection": {"t0": inj.t0, "t1": inj.t1, "dx": inj.dx, "dy": inj.dy, "rot": inj.rot},
            "n_truth_events": len(ev_t), "n_pass": sum(1 for p in per if p["passed"]), "false_alarm_events": false_alarms,
            "n_knock_events": len(kper), "n_knock_pass": sum(1 for p in kper if p["passed"]),
            "n_drift_only_events": len(per) - len(kper), "n_drift_only_pass": sum(1 for p in per if not p["knock"] and p["passed"]),
            "per_event": per, "truth_peak_pct_by_zone": peaks,
            "mean_abs_err_pp_all_samples": round(float(np.mean(errs)), 2) if errs else None,
            "max_abs_err_pp_all_samples": round(float(np.max(errs)), 1) if errs else None,
            "mean_abs_err_pp_knock_samples": round(float(np.mean(errs_in)), 2) if errs_in else None,
            "max_abs_err_pp_knock_samples": round(float(np.max(errs_in)), 1) if errs_in else None,
            "mean_abs_err_pp_vs_injected_only_truth": round(float(np.mean(errs_naive)), 2) if errs_naive else None,
            "tracker_lost_samples": int(sum(lost)), "tol_s": tol_t, "tol_pp": tol_pp,
            "all_pass": bool(per) and all(p["passed"] for p in per) and not false_alarms, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="default test set")
    ap.add_argument("--video", help="one catalogue clip")
    ap.add_argument("--shift", help="dx,dy,rot_deg for a SIMULATED step knock")
    ap.add_argument("--shift-at", help="t0,t1 (s) of the simulated knock")
    ap.add_argument("--no-sim", action="store_true", help="disable the simulated injection")
    ap.add_argument("--thr", type=float, default=5.0, help="exposed %% that counts as a breach")
    ap.add_argument("--margin", type=float, default=0.006, help="mask safety margin as fraction of frame width")
    ap.add_argument("--feat", default="sift", choices=["sift", "orb"])
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--embed-check", action="store_true", help="Cosmos-Embed1 view-change cross-check on ptz_parking (needs :8003)")
    args = ap.parse_args()
    names = DEFAULT_SET if args.all else ([args.video] if args.video else [])
    if not names and not args.embed_check:
        ap.error("use --all, --video NAME or --embed-check")
    t0 = time.time()
    results = {}
    rpath = HERE / "results.json"
    global RAW_KNOCK
    if args.all:
        RAW_KNOCK = raw_knock_check()
    elif rpath.exists():
        try:
            RAW_KNOCK = json.loads(rpath.read_text(encoding="utf-8")).get("checks", {}).get("raw_knock")
        except Exception:
            RAW_KNOCK = None
    if rpath.exists() and not args.all:
        try:
            results = json.loads(rpath.read_text(encoding="utf-8")).get("videos", {})
        except Exception:
            results = {}
    for n in names:
        results[n] = run_video(n, args)
    summary = {"generated_s": round(time.time() - t0, 1), "videos": results}
    if args.all:
        summary["checks"] = run_checks()
        summary["generated_s"] = round(time.time() - t0, 1)
    if "rochester_night" in results and results["rochester_night"].get("truth"):
        tr = results["rochester_night"]["truth"]
        summary["kill_test"] = {"criterion": "simulated shift: exposure start/end within 0.5 s and % exposed within 10% of truth",
                                "windows_pass": f"{tr['n_pass']}/{tr['n_truth_zones']}", "false_alarms": len(tr["false_alarm_events"]),
                                "passed": bool(tr["all_pass"])}
    old = {}
    if rpath.exists():
        try:
            old = json.loads(rpath.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    for k in ("checks", "protokit_issues", "eyeball"):
        if k in old and k not in summary:
            summary[k] = old[k]
    if args.all and "checks" in summary and "embed_view_change" in old.get("checks", {}):
        summary["checks"]["embed_view_change"] = old["checks"]["embed_view_change"]
    if args.embed_check:
        summary.setdefault("checks", {})["embed_view_change"] = embed_view_check()
    pk.write_results(rpath, summary)
    print(f"total {summary['generated_s']} s; results -> {rpath}")


if __name__ == "__main__":
    main()
