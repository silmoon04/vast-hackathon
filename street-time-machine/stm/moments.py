"""Highlight reel: up to 12 notable moments per camera, sorted by t -> out/<cam>/moments.json.

Kinds (CONTRACTS.md, Wave 2 "moments"):
  arrival / departure  a parked vehicle (data.objects parked=True) appears / leaves. Kept only when the
                       spot is seen empty for a few seconds on the other side AND its picture changes (the
                       detector often loses a parked car that never moved), the camera is not blind or
                       resetting (many parked tracks starting/ending together) and the event is not at the
                       clip edge. Otherwise it is a re-detection, an id switch or a view change, not an event.
  crowd                peak count of people on screen at once (per-frame count, 0.5 s rolling median).
  close_pass           a person and a MOVING vehicle with edge gap < 0.3 x vehicle height, feet and wheels
                       at a similar image depth, person not inside the vehicle box, not riding along.
  long_stay            the longest-staying person.
  blind                merged spans from out/<cam>/ledger.json; skipped (and said so) when it does not exist.
Every number comes from the cached tracks or the ledger. Titles carry no colours (none are computed here).

CLI: python -m stm.moments [cams...]   (default: every cam in data.CAMS with cached tracks)
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm import data

MAX_MOMENTS = 12
THUMB_W = 480
CLOSE_REL = 0.3          # close_pass: edge gap < 0.3 x vehicle height
DEPTH_REL = 0.5          # |person bottom - vehicle bottom| <= 0.5 x vehicle height (similar ground depth)
INSIDE_FRAC = 0.6        # person box this much inside the vehicle box = driver/occluded, not a pass
MOVING_REL = 0.3         # vehicle moving if centre speed >= 0.3 vehicle heights per second ...
MOVING_MIN_PX = 10.0     # ... and >= 10 px/s
EDGE_S = 2.0             # arrival/departure this close to the first/last detection = clip edge
EMPTY_WIN = 10.0         # look this far before an arrival / after a departure for the spot being empty
BLIND_PAD = 5.0          # arrivals/departures within 5 s of a blind span are camera effects, not events
RESET_N = 4              # >= 4 parked tracks starting/ending within +-2 s = view or tracker reset
CROWD_MIN = 3            # fewer than 3 people at once is not a crowd
LONG_MIN_S = 5.0
BLIND_MIN_S = 3.0        # shorter blind glitches are listed in the ledger, not the highlight reel
CLOSE_MIN_FRAMES = 3
SPOT_CHANGE_MIN = 15.0   # arrival/departure: the spot's picture must change by >= 15 grey levels (mean abs, 0-255)
SPOT_SAMPLES = (3.0, 6.0, 9.0)  # seconds on the "empty" side compared against a frame with the vehicle there
CAPS = {"arrival": 5, "departure": 5, "crowd": 2, "close_pass": 4, "long_stay": 1, "blind": 3}
PERSON_BGR = (59, 169, 242)     # #F2A93B
VEHICLE_BGR = (255, 182, 127)   # #7FB6FF


def mmss(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"


def r2(x: float) -> float:
    return round(float(x), 2)


def _box(r: Any) -> list[float]:
    return [round(float(r.x1), 1), round(float(r.y1), 1), round(float(r.x2), 1), round(float(r.y2), 1)]


def _union(boxes: list[list[float]]) -> list[float]:
    b = np.array(boxes, dtype=float)
    return [round(float(b[:, 0].min()), 1), round(float(b[:, 1].min()), 1),
            round(float(b[:, 2].max()), 1), round(float(b[:, 3].max()), 1)]


def iou_many(box: list[float], df: pd.DataFrame) -> np.ndarray:
    x1, y1, x2, y2 = box
    ix = np.clip(np.minimum(df.x2.values, x2) - np.maximum(df.x1.values, x1), 0, None)
    iy = np.clip(np.minimum(df.y2.values, y2) - np.maximum(df.y1.values, y1), 0, None)
    inter = ix * iy
    area = (df.x2.values - df.x1.values) * (df.y2.values - df.y1.values) + (x2 - x1) * (y2 - y1) - inter
    return np.where(area > 0, inter / np.maximum(area, 1e-9), 0.0)


def edge_gap(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
    """Edge-to-edge distance between boxes (0 when they overlap). Works on scalars or arrays."""
    dx = np.maximum(0, np.maximum(bx1 - ax2, ax1 - bx2))
    dy = np.maximum(0, np.maximum(by1 - ay2, ay1 - by2))
    return np.hypot(dx, dy)


# ------------------------------------------------------------------ blind spans
def ledger_blind(cam: str) -> list[dict] | None:
    """Blind spans from ledger.json, or None when the ledger does not exist / cannot be read."""
    p = data.OUT / cam / "ledger.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return [{"start": float(b["start"]), "end": float(b["end"]), "reason": str(b.get("reason") or "unknown")}
                for b in d.get("blind", [])]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def merge_blind(spans: list[dict], gap: float = 2.0) -> list[dict]:
    """Merge blind spans separated by <= gap s. Keeps seconds per reason."""
    out: list[dict] = []
    for b in sorted(spans, key=lambda s: s["start"]):
        dur = max(0.0, b["end"] - b["start"])
        if out and b["start"] - out[-1]["end"] <= gap:
            o = out[-1]
            o["end"] = max(o["end"], b["end"])
            o["by_reason"][b["reason"]] = o["by_reason"].get(b["reason"], 0.0) + dur
        else:
            out.append({"start": b["start"], "end": b["end"], "by_reason": {b["reason"]: dur}})
    return out


def _near_blind(t: float, spans: list[dict], pad: float) -> bool:
    return any(b["start"] - pad <= t <= b["end"] + pad for b in spans)


def spot_change(cam: str, box: list[float], t_with: float, t_other: list[float],
                cache: dict | None = None, stop_below: float = SPOT_CHANGE_MIN) -> float | None:
    """Smallest mean abs grey difference (0-255) inside `box` between the frame at t_with (vehicle there) and
    each frame at t_other. Small = the picture did not change: the vehicle is most likely still there and the
    detector/tracker just lost it. Stops at the first difference below `stop_below` (frame seeks are slow).
    None when no frame pair can be read."""
    cache = {} if cache is None else cache

    def grey(t: float) -> np.ndarray | None:
        k = round(float(t), 2)
        if k not in cache:
            try:
                cache[k] = cv2.cvtColor(data.frame_at(cam, k), cv2.COLOR_BGR2GRAY)
            except ValueError:
                cache[k] = None
        g = cache[k]
        if g is None:
            return None
        H, W = g.shape[:2]
        x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(W, box[2])), int(min(H, box[3]))
        return g[y1:y2, x1:x2].astype(np.float32) if x2 > x1 and y2 > y1 else None

    ref = grey(t_with)
    if ref is None:
        return None
    diffs: list[float] = []
    for t in t_other:
        o = grey(t)
        if o is None or o.shape != ref.shape:
            continue
        diffs.append(float(np.abs(o - ref).mean()))
        if diffs[-1] < stop_below:
            break
    return min(diffs) if diffs else None


BLIND_TITLE = {"dark": "Picture too dark", "blur": "Picture blurred", "frozen": "Picture frozen",
               "blocked": "View blocked", "no_signal": "No signal"}


def blind_moments(cam: str, spans: list[dict], df: pd.DataFrame, dur: float) -> list[dict]:
    out = []
    for b in merge_blind(spans):
        length = b["end"] - b["start"]
        if length < BLIND_MIN_S:
            continue
        main = max(b["by_reason"], key=b["by_reason"].get)
        parts = ", ".join(f"{k} {v:.0f} s" for k, v in sorted(b["by_reason"].items(), key=lambda kv: -kv[1]))
        inside = (df.t >= b["start"]) & (df.t < b["end"])
        n_det = int(inside.sum())
        rate_in = n_det / length
        rate_out = int((~inside).sum()) / max(1e-6, dur - length)
        detail = (f"The health ledger marks {mmss(b['start'])} to {mmss(b['end'])} ({length:.0f} s) as not visible "
                  f"({parts}). What happened then is UNKNOWN.")
        if n_det and rate_out > 0 and rate_in >= 0.5 * rate_out:
            detail += (f" The detector still returned {n_det} detections in this span ({rate_in:.0f}/s vs {rate_out:.0f}/s "
                       "elsewhere), so the ledger and the detector disagree here.")
        out.append({"t": r2(b["start"]), "end": r2(b["end"]), "kind": "blind",
                    "title": f"{BLIND_TITLE.get(main, 'Camera blind')} for {length:.0f} s", "detail": detail,
                    "object_ids": [], "box": None, "score": round(0.55 + 0.4 * min(1.0, length / 60.0), 3),
                    "_thumb_t": (b["start"] + b["end"]) / 2, "_draw": []})
    return out


# ------------------------------------------------------------------ arrivals / departures
def parked_moments(cam: str, df: pd.DataFrame, objs: pd.DataFrame, blind: list[dict], stats: dict) -> list[dict]:
    if objs.empty:
        return []
    parked = objs[objs.parked]
    if parked.empty:
        return []
    t0, t1 = float(df.t.min()), float(df.t.max())
    dur = data.info(cam)["duration"]
    veh = df[df.name.isin(data.VEHICLES)]
    out = []
    rej = stats.setdefault("parked_rejected", {})
    cache: dict = {}

    def reject(why: str) -> None:
        rej[why] = rej.get(why, 0) + 1

    for r in parked.itertuples():
        box = [r.x1, r.y1, r.x2, r.y2]
        stay = r.last_t - r.first_t
        g = df[df.tid == r.id]
        bw = max(1.0, r.x2 - r.x1)
        mx, my = g.cx.median(), g.cy.median()
        for kind, te in (("arrival", r.first_t), ("departure", r.last_t)):
            if (kind == "arrival" and te - t0 < EDGE_S) or (kind == "departure" and t1 - te < EDGE_S):
                reject(f"{kind}: at clip edge")
                continue
            if _near_blind(te, blind, BLIND_PAD):
                reject(f"{kind}: next to a blind span")
                continue
            col = "first_t" if kind == "arrival" else "last_t"
            if int((parked[col] - te).abs().le(2.0).sum()) >= RESET_N:
                reject(f"{kind}: view/tracker reset")
                continue
            lo, hi = (te - EMPTY_WIN, te - 0.5) if kind == "arrival" else (te + 0.5, te + EMPTY_WIN)
            lo, hi = max(lo, t0), min(hi, t1)
            if hi - lo < 2.0:
                reject(f"{kind}: too little video on the other side")
                continue
            w = veh[(veh.t >= lo) & (veh.t <= hi) & (veh.tid != r.id)]
            occupied = int((iou_many(box, w) >= 0.4).sum()) if len(w) else 0
            if occupied >= 5:
                reject(f"{kind}: spot not empty on the other side (re-detection / id switch)")
                continue
            # the detector can lose a parked car for many seconds: require the spot's picture to change
            sgn = -1.0 if kind == "arrival" else 1.0
            t_with = float(g.iloc[(g.t - (te - sgn * 0.5)).abs().argmin()].t)
            others = [te + sgn * s for s in SPOT_SAMPLES if lo <= te + sgn * s <= hi and not _near_blind(te + sgn * s, blind, 0.5)]
            change = spot_change(cam, box, t_with, others, cache)
            if change is None:
                reject(f"{kind}: spot not visible on the other side")
                continue
            if change < SPOT_CHANGE_MIN:
                reject(f"{kind}: picture unchanged (vehicle still there, detector lost it)")
                continue
            # did we see it drive in / out? (first or last second away from its parked position)
            edge = g[g.t <= r.first_t + 1.0].iloc[0] if kind == "arrival" else g[g.t >= r.last_t - 1.0].iloc[-1]
            moved = float(np.hypot(edge.cx - mx, edge.cy - my) / bw)
            seen_move = moved >= 0.5
            nm = str(r.name)
            label = nm.capitalize()
            same = parked[(parked.id != r.id) & (iou_many(box, parked) >= 0.4)]
            if kind == "arrival":
                prev = same[same.last_t <= te].sort_values("last_t")
                nb = (f" Same spot as {prev.name.iat[-1]} #{int(prev.id.iat[-1])}, gone since {mmss(prev.last_t.iat[-1])}; "
                      "whether it is the same vehicle is not checked.") if len(prev) else ""
            else:
                nxt_ = same[same.first_t >= te].sort_values("first_t")
                nb = (f" {nxt_.name.iat[0].capitalize()} #{int(nxt_.id.iat[0])} appears in the same spot at "
                      f"{mmss(nxt_.first_t.iat[0])}; whether it is the same vehicle is not checked.") if len(nxt_) else ""
            if kind == "arrival":
                how = "drives in" if seen_move else "appears (the approach was not seen)"
                detail = (f"{label} #{r.id} {how} at {mmss(te)} and stays parked {stay:.0f} s, until {mmss(r.last_t)}. "
                          f"No vehicle was detected in the spot in the {hi - lo:.0f} s before, and the spot looked different then "
                          f"(mean grey change {change:.0f} of 255).{nb}")
                title = f"{label} arrives"
                thumb_t = min(r.last_t, te + 0.5)
                t, end = te, min(dur, te + 3.0)
            else:
                how = "drives out" if seen_move else "disappears (the exit was not seen)"
                detail = (f"{label} #{r.id}, parked since {mmss(r.first_t)} ({stay:.0f} s), {how} at {mmss(te)}. "
                          f"No vehicle was detected in the spot in the {hi - lo:.0f} s after, and the spot looked different then "
                          f"(mean grey change {change:.0f} of 255).{nb}")
                title = f"{label} leaves"
                thumb_t = max(r.first_t, te - 0.5)
                t, end = te, min(dur, te + 3.0)
            gb = g.iloc[(g.t - thumb_t).abs().argmin()]
            score = 0.55 + 0.25 * min(1.0, stay / 120.0) + (0.12 if seen_move else 0.0)
            out.append({"t": r2(t), "end": r2(max(end, t)), "kind": kind, "title": title, "detail": detail,
                        "object_ids": [int(r.id)], "box": _box(gb), "score": round(score, 3),
                        "_thumb_t": float(gb.t), "_draw": [(_box(gb), VEHICLE_BGR)]})
    return out


# ------------------------------------------------------------------ crowd
def person_frames(df: pd.DataFrame, objs: pd.DataFrame) -> pd.DataFrame:
    ids = set(objs[objs.name == "person"].id) if len(objs) else set()
    return df[df.tid.isin(ids) & (df.name == "person")]


def crowd_moments(df: pd.DataFrame, objs: pd.DataFrame, stats: dict) -> list[dict]:
    p = person_frames(df, objs)
    frames = df[["f", "t"]].drop_duplicates("f").sort_values("f").reset_index(drop=True)
    cnt = p.groupby("f").tid.nunique()
    frames["n"] = frames.f.map(cnt).fillna(0).astype(int)
    if frames.empty:
        stats["crowd_peak"] = 0
        return []
    frames["s"] = frames.n.rolling(7, center=True, min_periods=1).median().round().astype(int)
    peak = int(frames.s.max())
    stats["crowd_peak"] = peak
    if peak < CROWD_MIN:
        return []
    out, used = [], []
    for i in frames.sort_values(["s", "t"], ascending=[False, True]).index:
        s = int(frames.s[i])
        if s < max(CROWD_MIN, int(np.ceil(0.75 * peak))) or len(out) >= 3:
            break
        t = float(frames.t[i])
        if any(abs(t - u) < 10.0 for u in used):
            continue
        used.append(t)
        # run of frames around i with smoothed count >= s
        j, k = i, i
        while j > 0 and frames.s[j - 1] >= s:
            j -= 1
        while k < len(frames) - 1 and frames.s[k + 1] >= s:
            k += 1
        # frame in the run whose raw count equals s, nearest the run's middle -> thumb + box
        run = frames.loc[j:k]
        hit = run[run.n == s]
        fr = hit.iloc[(hit.t - run.t.median()).abs().argmin()] if len(hit) else frames.loc[i]
        pp = p[p.f == fr.f]
        boxes = [_box(b) for b in pp.itertuples()]
        out.append({"t": r2(run.t.iat[0]), "end": r2(run.t.iat[-1]), "kind": "crowd", "title": f"{s} people at once",
                    "detail": (f"{s} people on screen together for {run.t.iat[-1] - run.t.iat[0]:.1f} s from "
                               f"{run.t.iat[0]:.1f} s ({mmss(run.t.iat[0])}); the most at once in this clip is {peak}."),
                    "object_ids": sorted(int(x) for x in pp.tid.unique()), "box": _union(boxes) if boxes else None,
                    "score": round(0.5 + 0.4 * s / peak, 3), "_thumb_t": float(fr.t),
                    "_draw": [(b, PERSON_BGR) for b in boxes]})
    return out


# ------------------------------------------------------------------ close pass
def vehicle_speeds(v: pd.DataFrame, half: float = 0.5) -> pd.Series:
    """Centre speed in px/s from the track's own positions +-half s around each detection."""
    sp = pd.Series(0.0, index=v.index)
    for _, g in v.groupby("tid"):
        tt = g.t.values
        if len(tt) < 2:
            continue
        a, b = np.clip(tt - half, tt[0], tt[-1]), np.clip(tt + half, tt[0], tt[-1])
        dx = np.interp(b, tt, g.cx.values) - np.interp(a, tt, g.cx.values)
        dy = np.interp(b, tt, g.cy.values) - np.interp(a, tt, g.cy.values)
        sp.loc[g.index] = np.hypot(dx, dy) / np.maximum(b - a, 1e-6)
    return sp


def close_pass_moments(df: pd.DataFrame, objs: pd.DataFrame, stats: dict) -> list[dict]:
    if objs.empty:
        return []
    p = person_frames(df, objs)
    vid = set(objs[objs.name.isin(data.VEHICLES) & ~objs.parked].id)
    v = df[df.tid.isin(vid) & df.name.isin(data.VEHICLES)].copy()
    stats["close_pass_pairs_checked"] = 0
    if p.empty or v.empty:
        return []
    v["speed"] = vehicle_speeds(v)
    v = v[(v.speed >= MOVING_MIN_PX) & (v.speed >= MOVING_REL * v.bh)]
    if v.empty:
        return []
    m = p.merge(v, on="f", suffixes=("_p", "_v"))
    if m.empty:
        return []
    m["gap"] = edge_gap(m.x1_p, m.y1_p, m.x2_p, m.y2_p, m.x1_v, m.y1_v, m.x2_v, m.y2_v)
    m["gap_rel"] = m.gap / m.bh_v.clip(lower=1)
    ix = (np.minimum(m.x2_p, m.x2_v) - np.maximum(m.x1_p, m.x1_v)).clip(lower=0)
    iy = (np.minimum(m.y2_p, m.y2_v) - np.maximum(m.y1_p, m.y1_v)).clip(lower=0)
    m["inside"] = ix * iy / (m.bw_p * m.bh_p).clip(lower=1)
    m = m[(m.gap_rel < CLOSE_REL) & ((m.y2_p - m.y2_v).abs() <= DEPTH_REL * m.bh_v) & (m.inside < INSIDE_FRAC)]
    stats["close_pass_frames"] = int(len(m))
    if m.empty:
        return []
    pn = p.groupby("tid").size()
    vname = objs.set_index("id").name
    out = []
    for (pid, vid_), g in m.groupby(["tid_p", "tid_v"]):
        stats["close_pass_pairs_checked"] += 1
        g = g.sort_values("t_p")
        span = g.t_p.iat[-1] - g.t_p.iat[0]
        if span > 3.0 and len(g) > 0.5 * pn.get(pid, len(g)):
            stats["close_pass_riding_skipped"] = stats.get("close_pass_riding_skipped", 0) + 1
            continue
        runs = (g.t_p.diff() > 1.0).cumsum()
        for _, rg in g.groupby(runs):
            if len(rg) < CLOSE_MIN_FRAMES:
                stats["close_pass_short_skipped"] = stats.get("close_pass_short_skipped", 0) + 1
                continue
            b = rg.loc[rg.gap_rel.idxmin()]
            vn = str(vname.get(vid_, b.name_v))
            closeness = 1.0 - float(b.gap_rel) / CLOSE_REL
            speedf = min(1.0, float(b.speed) / max(1.0, 3.0 * float(b.bh_v)))
            depthf = 1.0 - min(1.0, abs(float(b.y2_p - b.y2_v)) / (DEPTH_REL * max(1.0, float(b.bh_v))))
            lastf = min(1.0, float(rg.t_p.iat[-1] - rg.t_p.iat[0]) / 2.0)
            pb = [b.x1_p, b.y1_p, b.x2_p, b.y2_p]
            vb = [b.x1_v, b.y1_v, b.x2_v, b.y2_v]
            pb, vb = [round(float(x), 1) for x in pb], [round(float(x), 1) for x in vb]
            gap_txt = "touching boxes" if b.gap < 0.5 else f"{b.gap:.0f} px apart"
            out.append({"t": r2(b.t_p), "end": r2(max(rg.t_p.iat[-1], b.t_p)), "kind": "close_pass",
                        "title": f"Person passes close to moving {vn}",
                        "detail": (f"Person #{int(pid)} and moving {vn} #{int(vid_)} ({b.speed:.0f} px/s) were {gap_txt} "
                                   f"at {mmss(b.t_p)}, {b.gap_rel:.2f} x the {vn}'s height, close for "
                                   f"{rg.t_p.iat[-1] - rg.t_p.iat[0]:.1f} s. Image distance only; true distance is not known."),
                        "object_ids": [int(pid), int(vid_)], "box": _union([pb, vb]),
                        "score": round(0.45 + 0.15 * closeness + 0.12 * speedf + 0.1 * depthf + 0.08 * lastf, 3), "_thumb_t": float(b.t_p),
                        "_draw": [(pb, PERSON_BGR), (vb, VEHICLE_BGR)], "_person": int(pid)})
    # one event per person within 3 s: keep the best
    out.sort(key=lambda x: -x["score"])
    kept: list[dict] = []
    for c in out:
        if not any(k["_person"] == c["_person"] and abs(k["t"] - c["t"]) < 3.0 for k in kept):
            kept.append(c)
    return kept


# ------------------------------------------------------------------ long stay
def long_stay_moments(df: pd.DataFrame, objs: pd.DataFrame, stats: dict) -> list[dict]:
    if objs.empty:
        return []
    ppl = objs[objs.name == "person"].copy()
    if ppl.empty:
        return []
    ppl["stay"] = ppl.last_t - ppl.first_t
    r = ppl.sort_values(["stay", "n"], ascending=False).iloc[0]
    stats["long_stay_s"] = round(float(r.stay), 2)
    if r.stay < LONG_MIN_S:
        return []
    g = df[df.tid == r.id]
    mid = (r.first_t + r.last_t) / 2
    gb = g.iloc[(g.t - mid).abs().argmin()]
    others = ppl[ppl.id != r.id].stay
    nxt = f" The next longest person stayed {others.max():.0f} s." if len(others) else ""
    return [{"t": r2(r.first_t), "end": r2(r.last_t), "kind": "long_stay", "title": f"Person stays {r.stay:.0f} s",
             "detail": (f"Person #{int(r.id)} is on screen from {mmss(r.first_t)} to {mmss(r.last_t)} ({r.stay:.0f} s), "
                        f"the longest of {len(ppl)} people tracked; they moved {r.travel_px:.0f} px in total.{nxt}"),
             "object_ids": [int(r.id)], "box": _box(gb), "score": round(0.5 + 0.4 * min(1.0, r.stay / 60.0), 3),
             "_thumb_t": float(gb.t), "_draw": [(_box(gb), PERSON_BGR)]}]


# ------------------------------------------------------------------ selection + thumbs
def select(cands: list[dict], k: int = MAX_MOMENTS) -> list[dict]:
    """Best of each kind first, then highest score, with per-kind caps and no near-duplicates."""
    by = sorted(cands, key=lambda c: -c["score"])
    chosen: list[dict] = []

    def dup(c: dict) -> bool:
        def overlap(a, b):
            return a is not None and b is not None and iou_many(a, pd.DataFrame([b], columns=["x1", "y1", "x2", "y2"]))[0] > 0.3
        return any(o["kind"] == c["kind"] and ((abs(o["t"] - c["t"]) < 2.0 and (overlap(o["box"], c["box"]) or o["box"] is None))
                   or (c["object_ids"] and o["object_ids"] == c["object_ids"])) for o in chosen)

    for kind in CAPS:
        best = next((c for c in by if c["kind"] == kind), None)
        if best is not None and len(chosen) < k:
            chosen.append(best)
    for mult in (1, 2):  # second pass relaxes the per-kind caps when there is still room
        for c in by:
            if len(chosen) >= k:
                break
            if c in chosen or dup(c) or sum(o["kind"] == c["kind"] for o in chosen) >= mult * CAPS.get(c["kind"], k):
                continue
            chosen.append(c)
    return sorted(chosen, key=lambda c: (c["t"], c["kind"]))


def write_thumb(cam: str, m: dict, path: Path) -> bool:
    try:
        img = data.frame_at(cam, m["_thumb_t"])
    except ValueError:
        return False
    for box, col in m["_draw"]:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, img.shape[1] // 320))
    h, w = img.shape[:2]
    img = cv2.resize(img, (THUMB_W, int(round(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 82]))


def build(cam: str, thumbs: bool = True) -> tuple[dict, dict]:
    t_start = time.time()
    df = data.tracks(cam)
    objs = data.objects(cam)
    dur = data.info(cam)["duration"]
    raw_blind = ledger_blind(cam)
    blind = raw_blind or []
    stats: dict[str, Any] = {"ledger": raw_blind is not None, "objects": int(len(objs)),
                             "parked": int(objs.parked.sum()) if len(objs) else 0,
                             "people": int((objs.name == "person").sum()) if len(objs) else 0}
    cands: list[dict] = []
    cands += parked_moments(cam, df, objs, blind, stats)
    cands += crowd_moments(df, objs, stats)
    cands += close_pass_moments(df, objs, stats)
    cands += long_stay_moments(df, objs, stats)
    if raw_blind is not None:
        cands += blind_moments(cam, raw_blind, df, dur)
    stats["candidates"] = {k: sum(c["kind"] == k for c in cands) for k in CAPS}
    chosen = select(cands)
    mdir = data.out_dir(cam) / "moments"
    if thumbs and mdir.exists():
        shutil.rmtree(mdir)
    moments = []
    for n, m in enumerate(chosen, 1):
        rel = f"moments/{n}.jpg"
        ok = write_thumb(cam, m, data.out_dir(cam) / rel) if thumbs else False
        moments.append({"t": m["t"], "end": m["end"], "kind": m["kind"], "title": m["title"], "detail": m["detail"],
                        "object_ids": m["object_ids"], "thumb": rel if (ok or not thumbs) else None,
                        "box": m["box"], "score": m["score"]})
    stats["kept"] = {k: sum(m["kind"] == k for m in moments) for k in CAPS}
    stats["runtime_s"] = round(time.time() - t_start, 1)
    return {"camera": cam, "moments": moments}, stats


def summary_text(cam: str, res: dict, st: dict) -> str:
    lines = [f"moments for {cam}: {len(res['moments'])} kept (max {MAX_MOMENTS})",
             f"objects: {st['objects']} tracks ({st['parked']} parked vehicles, {st['people']} people)",
             "candidates by kind: " + ", ".join(f"{k} {v}" for k, v in st["candidates"].items()),
             "kept by kind: " + ", ".join(f"{k} {v}" for k, v in st["kept"].items()),
             f"crowd peak (people at once, 0.5 s median): {st.get('crowd_peak', 0)}"
             + ("" if st.get("crowd_peak", 0) >= CROWD_MIN else f" (< {CROWD_MIN}, no crowd moment)"),
             f"longest-staying person: {st['long_stay_s']} s" if "long_stay_s" in st else "longest-staying person: none tracked",
             f"close_pass: {st.get('close_pass_frames', 0)} close frames, {st.get('close_pass_pairs_checked', 0)} person-vehicle pairs, "
             f"{st.get('close_pass_riding_skipped', 0)} skipped as riding/inside, {st.get('close_pass_short_skipped', 0)} shorter than {CLOSE_MIN_FRAMES} frames"]
    if st.get("parked_rejected"):
        lines.append("arrival/departure candidates rejected: " +
                     "; ".join(f"{k}: {v}" for k, v in sorted(st["parked_rejected"].items())))
    lines.append("blind: from ledger.json" if st["ledger"] else
                 "blind: SKIPPED, out/<cam>/ledger.json not found, so blind time is not known here")
    for m in res["moments"]:
        lines.append(f"  {mmss(m['t'])}  {m['kind']:<10} {m['score']:.2f}  {m['title']}")
    lines.append(f"runtime: {st['runtime_s']} s")
    return "\n".join(lines) + "\n"


def cams_with_tracks() -> list[str]:
    return [c for c in data.CAMS if any(data.TRACK_CACHE.glob(f"{c}__det__*.json.gz"))]


def main(argv: list[str]) -> int:
    cams = argv or cams_with_tracks()
    overall = []
    for cam in cams:
        if cam not in data.CAMS:
            print(f"unknown cam {cam}")
            overall.append(f"{cam}: unknown camera, skipped")
            continue
        try:
            res, st = build(cam)
        except FileNotFoundError as e:
            print(f"{cam}: skipped ({e})")
            overall.append(f"{cam}: skipped, no cached tracks")
            continue
        data.write_json(data.out_dir(cam) / "moments.json", res)
        txt = summary_text(cam, res, st)
        (data.out_dir(cam) / "moments_summary.txt").write_text(txt, encoding="utf-8")
        print(txt)
        overall.append(f"{cam}: {len(res['moments'])} moments " +
                       str({k: v for k, v in st["kept"].items() if v}) +
                       ("" if st["ledger"] else " | blind skipped: no ledger.json") + f" | {st['runtime_s']} s")
    if not argv:
        (data.OUT / "moments_summary.txt").write_text("\n".join(overall) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
