"""Hit-and-run detective: for each parked vehicle, who/what came close while it was there.

Targets  = parked vehicles (objects(cam).parked); on cameras with none, vehicles on screen >= 10 s.
Visitors = other *moving* tracks whose edge-to-edge box gap to the target, divided by the target's
           box height, drops to <= 0.5 at a moment the camera could actually see (not blind).
Blind    = ledger.json spans when the ledger module has run, plus our own cheap check
           (dark / frozen / camera moving or cutting), so a camera pan never manufactures visitors.
The summary says who was near. It never says who did anything.

CLI: python -m stm.near [cams...]
"""
from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm import data

NEAR_REL = 0.5          # visitor if gap / target box height <= this
MAX_TARGETS = 40        # per camera, sorted by number of visitors
LONG_LIVED_S = 10.0     # fallback target rule on cameras with no parked vehicle
DUP_IOU = 0.5           # a track overlapping the target this much is the target re-detected
MOVE_REL = 0.5          # a vehicle visitor must travel >= this x its own box width (outside blind time)
CHAIN_GAP_S = 3.0       # tracks of one class that hand over within 3 s, close by = one visitor
CHAIN_OVERLAP_S = 1.0   # ...or overlap by up to 1 s (ByteTrack sometimes doubles an id)
SAMPLE_HZ = 2.0         # own blind check sampling rate
DARK_LUMA = 35.0
FROZEN_DIFF = 0.05      # mean abs grey diff between samples; codec noise on a still scene is ~0.2
FROZEN_MIN_S = 3.0
VIEW_PIX = 15.0         # a pixel "changed" if its grey level moved more than this
VIEW_FRAC = 0.25        # view changed if this share of pixels changed between samples
MERGE_GAP_S = 1.0       # blind spans closer than this are joined
DEFAULT_CAMS = ["ptz_parking", "car_detection", "person_bicycle_car_detection"]
SUB = "near"            # crops go to out/<cam>/near/


# ---------------------------------------------------------------- blind time
def own_blind(cam: str, start: float = 0.0, end: float | None = None) -> list[dict[str, Any]]:
    """Cheap visibility check at SAMPLE_HZ on 160x90 grey frames.

    dark          mean luma < 35
    frozen        mean abs difference to the previous sample < 0.05 for >= 3 s (a static car park
                  still gives ~0.2 from sensor/codec noise, a stuck feed gives ~0)
    view_changed  >= 25 % of pixels changed by > 15 grey levels between samples: the PTZ camera
                  panned/zoomed/cut, or something filled the view. Detections then are not trusted.
    """
    cap = cv2.VideoCapture(str(data.CAMS[cam]["video"]))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / SAMPLE_HZ)))
    ts, luma, diff, frac = [], [], [], []
    prev = None
    i = 0
    if start > 0:                       # optional window (tests); the CLI always scans everything
        i = int(start * fps) // step * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    last = float("inf") if end is None else end * fps
    while i <= last and cap.grab():
        if i % step == 0:
            ok, img = cap.retrieve()
            if ok:
                g = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2GRAY).astype(np.float32)
                ts.append(i / fps)
                luma.append(float(g.mean()))
                if prev is None:
                    diff.append(np.nan)
                    frac.append(0.0)
                else:
                    d = np.abs(g - prev)
                    diff.append(float(d.mean()))
                    frac.append(float((d > VIEW_PIX).mean()))
                prev = g
        i += 1
    cap.release()
    if not ts:
        return []
    ts_a, luma_a, diff_a, frac_a = np.array(ts), np.array(luma), np.array(diff), np.array(frac)
    half = 0.5 / SAMPLE_HZ
    flags: list[tuple[float, float, str, float]] = []
    for k, t in enumerate(ts_a):
        if luma_a[k] < DARK_LUMA:
            flags.append((t - half, t + half, "dark", round(float(luma_a[k]), 2)))
        if frac_a[k] >= VIEW_FRAC:
            # the change happened between the previous sample and this one; pad half a step each side
            flags.append((t - 3 * half, t + half, "view_changed", round(float(frac_a[k]), 2)))
    run: list[int] = []
    for k in list(range(1, len(ts_a))) + [None]:
        if k is not None and np.isfinite(diff_a[k]) and diff_a[k] < FROZEN_DIFF:
            run.append(k)
            continue
        if run and ts_a[run[-1]] - ts_a[run[0] - 1] >= FROZEN_MIN_S:
            flags.append((ts_a[run[0] - 1], ts_a[run[-1]] + half, "frozen", 0.0))
        run = []
    dur = data.info(cam)["duration"]
    spans = [{"start": max(0.0, a), "end": min(dur, b), "reason": r, "score": s} for a, b, r, s in flags]
    # join same-reason spans less than MERGE_GAP_S apart; different reasons stay separate entries
    return [m for r in sorted({x["reason"] for x in spans})
            for m in merge_spans([x for x in spans if x["reason"] == r], gap=MERGE_GAP_S)]


def merge_spans(spans: list[dict[str, Any]], gap: float = 0.0) -> list[dict[str, Any]]:
    """Union spans that overlap (or are within `gap` s); reasons are joined with '+', score is the max."""
    out: list[dict[str, Any]] = []
    for s in sorted(spans, key=lambda s: (s["start"], s["end"])):
        if out and s["start"] <= out[-1]["end"] + gap:
            o = out[-1]
            o["end"] = max(o["end"], s["end"])
            if s["reason"] not in o["reason"].split("+"):
                o["reason"] = o["reason"] + "+" + s["reason"]
            o["score"] = max(o.get("score") or 0.0, s.get("score") or 0.0)
        else:
            out.append({"start": float(s["start"]), "end": float(s["end"]), "reason": str(s["reason"]),
                        "score": float(s.get("score") or 0.0)})
    return out


def blind_spans(cam: str) -> tuple[list[dict[str, Any]], str]:
    """Ledger blind spans if the ledger ran, plus our own view-change check (else our full check).

    Returned unmerged, one reason per entry, so a target only reports reasons inside its own window."""
    own = own_blind(cam)
    lp = data.out_dir(cam) / "ledger.json"
    if lp.exists():
        try:
            led = json.loads(lp.read_text(encoding="utf-8")).get("blind", [])
            led = [{"start": float(b["start"]), "end": float(b["end"]), "reason": str(b.get("reason") or "blind"),
                    "score": float(b.get("score") or 0.0)} for b in led]
            # the ledger has no camera-motion reason; keep ours for that
            return led + [b for b in own if b["reason"] == "view_changed"], "ledger.json + own view-change check"
        except (ValueError, KeyError, TypeError):
            pass
    return own, "own check (dark / frozen / view changed), ledger.json not found"


def in_spans(t: np.ndarray, spans: list[dict[str, Any]]) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for s in spans:
        m |= (t >= s["start"]) & (t <= s["end"])
    return m


def clip_spans(spans: list[dict[str, Any]], a: float, b: float) -> list[dict[str, Any]]:
    """Blind time inside [a, b]: clip each raw span, then union, so reasons are only those that apply."""
    cut = [{"start": max(a, s["start"]), "end": min(b, s["end"]), "reason": s["reason"]}
           for s in spans if s["end"] > a and s["start"] < b]
    return [{"start": round(m["start"], 2), "end": round(m["end"], 2), "reason": m["reason"]}
            for m in merge_spans(cut) if m["end"] - m["start"] > 0.01]


# ---------------------------------------------------------------- geometry
def edge_gap(box: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    """Edge-to-edge distance between one box and many boxes (0 when they overlap)."""
    dx = np.maximum(0.0, np.maximum(box[0] - x2, x1 - box[2]))
    dy = np.maximum(0.0, np.maximum(box[1] - y2, y1 - box[3]))
    return np.hypot(dx, dy)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


# ---------------------------------------------------------------- core
@lru_cache(maxsize=None)
def _objects(cam: str) -> pd.DataFrame:
    return data.objects(cam)


def objects(cam: str) -> pd.DataFrame:
    """data.objects is ~6 s on ptz_parking; compute it once per process."""
    return _objects(cam).copy()


def pick_targets(objs: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if objs.empty:
        return objs, "no tracked objects"
    parked = objs[objs.parked]
    if len(parked):
        return parked, "parked vehicles (on screen >= 20 s, centre within half a box width)"
    span = objs.last_t - objs.first_t
    long = objs[objs.name.isin(data.VEHICLES) & (span >= LONG_LIVED_S)]
    longest = float(span[objs.name.isin(data.VEHICLES)].max()) if objs.name.isin(data.VEHICLES).any() else 0.0
    return long, (f"no parked vehicles; vehicles on screen >= {LONG_LIVED_S:.0f} s "
                  f"(longest vehicle here: {longest:.1f} s)")


def moving_candidates(df: pd.DataFrame, objs: pd.DataFrame, blind: list[dict[str, Any]]) -> dict[int, str]:
    """Tracks that can be visitors: people/bicycles always; vehicles only if they really moved while visible."""
    keep: dict[int, str] = {}
    vis = df[~in_spans(df.t.to_numpy(), blind)]
    for r in objs.itertuples():
        if r.parked:
            continue
        if r.name in ("person", "bicycle"):
            keep[int(r.id)] = r.name
            continue
        g = vis[vis.tid == r.id]
        if len(g) < 3:
            continue
        ext = float(np.hypot(g.cx.max() - g.cx.min(), g.cy.max() - g.cy.min()))
        if ext >= MOVE_REL * max(1.0, float(r.x2 - r.x1)):
            keep[int(r.id)] = r.name
    return keep


def chain_ids(df: pd.DataFrame, ids: list[int]) -> dict[int, int]:
    """Map each track to a chain root, so one person split into several track ids counts once.

    b joins a's chain when they share a class, b starts between 1 s before a ends (duplicate ids
    overlap briefly) and 3 s after, and b's first position is within (1.5 + gap s) box widths of
    where a was at that moment. Visitor rows keep their own track ids; only the summary counts chains.
    """
    info = {}
    for tid in ids:
        g = df[df.tid == tid]
        info[tid] = (g.name.mode().iat[0], g.t.to_numpy(), g.cx.to_numpy(), g.cy.to_numpy(), float(g.bw.median()))
    root = {tid: tid for tid in ids}
    order = sorted(ids, key=lambda k: info[k][1][0])
    for j, b in enumerate(order):
        nb, tb, xb, yb, wb = info[b]
        for a in order[:j]:
            na, ta, xa, ya, wa = info[a]
            gap = tb[0] - ta[-1]
            if na != nb or not (-CHAIN_OVERLAP_S <= gap <= CHAIN_GAP_S):
                continue
            k = int(np.abs(ta - tb[0]).argmin())
            if np.hypot(xb[0] - xa[k], yb[0] - ya[k]) <= (1.5 + max(0.0, gap)) * max(wa, wb):
                root[b] = root[a]
                break
    return root


def plural(n: int, name: str) -> str:
    word = {"person": ("person", "people"), "bus": ("bus", "buses")}.get(name, (name, name + "s"))
    return f"{n} {word[0] if n == 1 else word[1]}"


def scan(cam: str, crops: bool = True, max_targets: int = MAX_TARGETS, write: bool = True) -> dict[str, Any]:
    t0 = time.time()
    df = data.tracks(cam)
    df = df[df.tid >= 0]
    objs = objects(cam)
    targets, rule = pick_targets(objs)
    blind, blind_src = blind_spans(cam)
    cands = moving_candidates(df, objs, blind) if len(objs) else {}
    cdf = df[df.tid.isin(list(cands))]
    cdf = cdf[~in_spans(cdf.t.to_numpy(), blind)]
    roots = chain_ids(df, list(cands)) if cands else {}

    found = []
    for r in targets.itertuples():
        box = np.array([r.x1, r.y1, r.x2, r.y2], dtype=float)
        h = max(1.0, box[3] - box[1])
        w = cdf[(cdf.t >= r.first_t) & (cdf.t <= r.last_t) & (cdf.tid != r.id)]
        visitors = []
        if len(w):
            gap = edge_gap(box, w.x1.to_numpy(), w.y1.to_numpy(), w.x2.to_numpy(), w.y2.to_numpy())
            w = w.assign(gap=gap, rel=gap / h)
            for tid, g in w[w.rel <= NEAR_REL].groupby("tid"):
                # the target itself, re-detected under another id, is not a visitor
                med = g[["x1", "y1", "x2", "y2"]].median().to_numpy()
                if iou(med, box) >= DUP_IOU:
                    continue
                k = g.gap.idxmin()
                visitors.append({"object_id": int(tid), "name": cands[int(tid)],
                                 "closest_t": round(float(g.t[k]), 2), "gap_px": round(float(g.gap[k]), 1),
                                 "gap_rel": round(float(g.rel[k]), 3), "enter_t": round(float(g.t.min()), 2),
                                 "exit_t": round(float(g.t.max()), 2), "crop": None})
        visitors.sort(key=lambda v: v["closest_t"])
        tb = clip_spans(blind, r.first_t, r.last_t)
        blind_s = sum(b["end"] - b["start"] for b in tb)
        found.append({"object_id": int(r.id), "name": r.name, "box": [round(float(x), 1) for x in box],
                      "start": round(float(r.first_t), 2), "end": round(float(r.last_t), 2), "crop": None,
                      "visitors": visitors, "blind": tb, "_blind_s": blind_s})

    found.sort(key=lambda x: (-len(x["visitors"]), -(x["end"] - x["start"]), x["object_id"]))
    total = len(found)
    found = found[:max_targets]
    for tg in found:
        groups: dict[str, set[int]] = {}
        for v in tg["visitors"]:
            groups.setdefault(v["name"], set()).add(roots.get(v["object_id"], v["object_id"]))
        noun = "car" if tg["name"] == "car" else tg["name"]
        if groups:
            parts = [plural(len(s), n) for n, s in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]
            who = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
            ntr = len(tg["visitors"])
            nun = sum(len(s) for s in groups.values())
            extra = f" ({ntr} tracks)" if ntr != nun else ""
            s = f"{who}{extra} came within half a {noun}-height of it"
        else:
            s = f"Nobody came within half a {noun}-height of it"
        s += f" between {fmt(tg['start'])} and {fmt(tg['end'])}; camera was blind for {tg['_blind_s']:.1f} s"
        if tg["blind"]:
            reasons = sorted({x for b in tg["blind"] for x in b["reason"].split("+")})
            s += f" ({', '.join(reasons)}), so who was near it then is UNKNOWN"
        tg["summary"] = s + "."
        del tg["_blind_s"]
    if crops and found:
        reqs = [(tg["object_id"], None) for tg in found] + \
               [(v["object_id"], v["closest_t"]) for tg in found for v in tg["visitors"]]
        paths = save_crops(cam, reqs)
        for tg in found:
            tg["crop"] = paths[(tg["object_id"], None)]
            for v in tg["visitors"]:
                v["crop"] = paths[(v["object_id"], v["closest_t"])]

    res = {"camera": cam, "targets": found}
    mb = merge_spans(blind)
    mb_txt = ", ".join("{:.1f}-{:.1f} s {}".format(b["start"], b["end"], b["reason"]) for b in mb)
    nvis = sum(len(t["visitors"]) for t in found)
    by = pd.Series([v["name"] for t in found for v in t["visitors"]]).value_counts().to_dict() if nvis else {}
    lines = [
        f"near: {cam}",
        f"target rule: {rule}",
        f"targets found: {total}; written: {len(found)} (capped at {max_targets}, sorted by number of visitors)",
        f"targets with >= 1 visitor: {sum(1 for t in found if t['visitors'])}",
        f"visitor entries (gap <= {NEAR_REL} x target box height, seen while camera visible): {nvis} {by}",
        f"moving candidate tracks: {len(cands)}",
        f"blind source: {blind_src}; blind spans: {len(mb)}, {sum(b['end'] - b['start'] for b in mb):.1f} s total "
        f"({mb_txt or 'none'})",
        "visitors are 'who was near', never 'who did it'.",
        f"runtime: {time.time() - t0:.1f} s",
    ]
    if write:
        data.write_json(data.out_dir(cam) / "near.json", res)
        (data.out_dir(cam) / "near_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


def save_crops(cam: str, reqs: list[tuple[int, float | None]], pad: float = 0.15) -> dict:
    """Same crops as data.save_crop (biggest box, or the detection nearest t), but read in ONE
    sequential decode pass instead of one seek per crop. Returns {(tid, t): path rel. to out/<cam>/}."""
    df = data.tracks(cam)
    fps = data.info(cam)["fps"]
    want: dict[int, list[tuple[tuple[int, float | None], Any, str]]] = {}
    for tid, t in dict.fromkeys(reqs):
        g = df[df.tid == tid]
        r = g.iloc[(g.t - t).abs().argmin()] if t is not None else g.loc[(g.bw * g.bh).idxmax()]
        name = f"{tid}.jpg" if t is None else f"{tid}_{float(r.t):.1f}.jpg"
        want.setdefault(int(round(float(r.t) * fps)), []).append(((tid, t), r, name))
    folder = data.out_dir(cam) / SUB
    folder.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    cap = cv2.VideoCapture(str(data.CAMS[cam]["video"]))
    last = max(want) if want else -1
    i = 0
    while i <= last and cap.grab():
        if i in want:
            ok, img = cap.retrieve()
            if ok:
                H, W = img.shape[:2]
                for key, r, name in want[i]:
                    px, py = r.bw * pad, r.bh * pad
                    x1, y1 = int(max(0, r.x1 - px)), int(max(0, r.y1 - py))
                    x2, y2 = int(min(W, r.x2 + px)), int(min(H, r.y2 + py))
                    cv2.imwrite(str(folder / name), img[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    out[key] = f"{SUB}/{name}"
        i += 1
    cap.release()
    for k in dict.fromkeys(reqs):           # anything the pass could not reach: fall back to a seek
        if k not in out:
            out[k] = data.save_crop(cam, k[0], t=k[1], sub=SUB)
    return out


def fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


if __name__ == "__main__":
    for cam in sys.argv[1:] or DEFAULT_CAMS:
        t0 = time.time()
        r = scan(cam)
        n = len(r["targets"])
        print(cam, n, "targets", sum(len(t["visitors"]) for t in r["targets"]), "visitors",
              f"{time.time() - t0:.1f}s")
        print((data.out_dir(cam) / "near_summary.txt").read_text(encoding="utf-8"))
