"""Video synopsis: every object from one camera view on its own background plate, time-compressed to <= 60 s.

One camera view per render. A PTZ camera that turns away mid-clip (ptz_parking) is rendered on its first home
view only, found from the ledger's long "blocked"/"blur" stretches; tracks from other views are skipped and
counted. A fixed camera with several real days gets one output segment per day, each on that day's plate, in
day order, so ghosts from different days never mix.

Moving objects are re-timed inside their day segment by a greedy scheduler (fewest box overlaps, earliest start
wins ties) and played at source speed (long tracks are sped up so they fit). Parked vehicles are one static,
unlabelled, half-transparent ghost per stay: tracker fragments of the same parked car (same spot, same day, no
real gap) are merged into one ghost. Movers are preferred over stationary fragments when a cap applies.
People's heads (top 25 % of the box) are Gaussian-blurred before pasting. Ghost pixels are cut from the source
frames at the exact detection frame in one sequential decode pass. The mp4 and json are written atomically.

CLI: python -m stm.synopsis [cams...]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm import data as D

FFMPEG = Path(os.environ.get("STM_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg")
OUT_W = 960
FPS = 12.0
STEP = 0.25            # box / crop sample step, seconds
MAX_OUT = 60.0         # synopsis length cap, seconds
MIN_SPAN = 1.0         # skip tracks shorter than this
ALPHA = 0.85           # moving ghost opacity
PARKED_ALPHA = 0.5     # static parked ghost opacity
MAX_GHOSTS = {"yonge_dundas": 120}
DEFAULT_CAP = 150      # most moving ghosts in one render (movers first, then longest)
OUT_TARGET = {"ptz_parking": 30.0}   # fixed output length for these cams
CONC = 6.0             # otherwise aim for about this many moving ghosts on screen at once
VIEW_FIRST = ("ptz_parking",)        # cams whose camera turns away: render the first home view only
VIEW_BREAK_S = 20.0    # a blocked/blur run this long = the camera pointed elsewhere
VIEW_MARGIN = 1.0      # seconds trimmed either side of a view break (the ledger is per second)
MIN_OUT = 15.0         # shortest auto output (few ghosts), unless half the window is shorter
MERGE_GAP = 30.0       # fragments at one spot closer than this in time are the same stay
HEAD_FRAC = 0.25       # top share of a person box that is blurred
CUT_HEAD_FRAC = 1.0    # a person cut off by the bottom of the frame is right at the lens: blur all of it
CLOSE_FRAC = 0.45      # people taller than this share of the frame (median box) walk right past the lens,
CLOSE_MAX_FRAC = 0.55  # ... or at their biggest: not replayed (giant heads, real faces), counted instead
DEFAULT_CAMS = ["ptz_parking", "person_bicycle_car_detection", "yonge_dundas"]
GRID = 8               # occupancy grid cell size in output pixels


def mmss(t: float) -> str:
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


def _iou(a: Any, b: Any) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def _box(r: dict[str, Any]) -> tuple[float, float, float, float]:
    return (r["x1"], r["y1"], r["x2"], r["y2"])


# ---------------------------------------------------------------- which view, which days

def view_windows(blind: list[dict[str, Any]], dur: float, break_s: float = VIEW_BREAK_S,
                 margin: float = VIEW_MARGIN) -> list[tuple[float, float]]:
    """Split [0, dur] at long stretches where the camera looked elsewhere (merged blocked/blur runs)."""
    spans = sorted((float(b["start"]), float(b["end"])) for b in blind if b.get("reason") in ("blocked", "blur"))
    runs: list[list[float]] = []
    for a, b in spans:
        if runs and a <= runs[-1][1] + 1.0:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    wins, cur = [], 0.0
    for a, b in (r for r in runs if r[1] - r[0] >= break_s):
        if a - margin > cur:
            wins.append((round(cur, 2), round(a - margin, 2)))
        cur = b + margin
    if dur > cur:
        wins.append((round(cur, 2), round(dur, 2)))
    return wins or [(0.0, round(dur, 2))]


def choose_view(cam: str) -> dict[str, Any]:
    """{'window': (w0, w1), 'view': id, 'label': str} for the one view this render shows."""
    dur = D.info(cam)["duration"]
    days = D.days(cam)
    if cam not in VIEW_FIRST:
        return {"window": (0.0, round(dur, 2)), "view": "fixed", "label": ""}
    lp = D.OUT / cam / "ledger.json"
    blind = json.loads(lp.read_text(encoding="utf-8")).get("blind", []) if lp.exists() else []
    wins = view_windows(blind, dur) if blind else [(days[0]["start"], days[0]["end"])]
    w0, w1 = next((w for w in wins if w[1] - w[0] >= 30.0), wins[0])
    d = next((d for d in days if d["start"] <= w0 < d["end"]), days[0])
    w1 = min(w1, d["end"])
    return {"window": (round(w0, 2), round(w1, 2)), "view": d["id"], "label": f"{d['label']} view"}


def blocks_for(cam: str, window: tuple[float, float]) -> list[dict[str, Any]]:
    """One output segment per day that overlaps the window, in day order, each with its own plate."""
    days = D.days(cam)
    out = []
    for d in days:
        a, b = max(d["start"], window[0]), min(d["end"], window[1])
        if b - a > 0.5:
            plate = f"background_{d['id']}.jpg" if len(days) > 1 else "background.jpg"
            out.append({"day": d["id"], "label": d["label"], "src_start": round(a, 2), "src_end": round(b, 2),
                        "plate": plate})
    return out


# ---------------------------------------------------------------- what to show

def _still(r: dict[str, Any]) -> bool:
    return r["name"] in D.VEHICLES and r["travel_px"] < max(2.0 * (r["x2"] - r["x1"]), 40.0)


def stays(rows: list[dict[str, Any]], iou: float = 0.4, gap: float = MERGE_GAP) -> list[list[dict[str, Any]]]:
    """Group stationary rows into stays: same spot (box IoU with the spot's longest row) and no time gap
    longer than `gap` seconds inside a stay."""
    spots: list[list[dict[str, Any]]] = []
    for r in sorted(rows, key=lambda r: -(r["last_t"] - r["first_t"])):
        for s in spots:
            if _iou(_box(r), _box(s[0])) >= iou:
                s.append(r)
                break
        else:
            spots.append([r])
    out = []
    for s in spots:
        cur: list[dict[str, Any]] = []
        end = -1e9
        for r in sorted(s, key=lambda r: r["first_t"]):
            if cur and r["first_t"] - end > gap:
                out.append(cur)
                cur, end = [], -1e9
            cur.append(r)
            end = max(end, r["last_t"])
        out.append(cur)
    return out


def select(cam: str, blocks: list[dict[str, Any]] | None = None, cap: int | None = None,
           frame_h: float | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, int]]:
    """Return (moving objects, parked stays, drop counts). Objects are clipped to their day's block; objects
    outside every block (another camera view) are skipped and counted. With frame_h, people whose box is
    taller than CLOSE_FRAC of the frame (walking right past the lens) are skipped and counted."""
    blocks = blocks or blocks_for(cam, (0.0, D.info(cam)["duration"]))
    o = D.objects(cam)
    drops = {"other_view": 0, "short": 0, "too_close": 0, "parked_fragment": 0, "cap": 0}
    cols = ["id", "name", "day", "first_t", "last_t", "x1", "y1", "x2", "y2", "travel_px", "parked", "span", "moves"]
    if o.empty:
        return pd.DataFrame(columns=cols), [], drops
    by_day = {b["day"]: b for b in blocks}
    if frame_h:
        tr = D.tracks(cam)
        bh_max = tr[tr.tid >= 0].groupby("tid").bh.max().to_dict()
    rows = []
    for r in o.to_dict("records"):
        r = {**r, "id": int(r["id"]), "parked": bool(r["parked"]), "day": str(r["day"])}
        b = by_day.get(r["day"])
        a, e = (max(r["first_t"], b["src_start"]), min(r["last_t"], b["src_end"])) if b else (0.0, -1.0)
        if e <= a:
            drops["other_view"] += 1
            continue
        if e - a < MIN_SPAN:
            drops["short"] += 1
            continue
        if frame_h and r["name"] == "person" and (r["y2"] - r["y1"] > CLOSE_FRAC * frame_h
                                                   or bh_max.get(r["id"], 0.0) > CLOSE_MAX_FRAC * frame_h):
            drops["too_close"] += 1
            continue
        rows.append({**r, "first_t": float(a), "last_t": float(e), "span": float(e - a)})
    parked: list[dict[str, Any]] = []
    movers = []
    for b in blocks:
        day_rows = [r for r in rows if r["day"] == b["day"]]
        still = [r for r in day_rows if r["parked"] or _still(r)]
        used: set[int] = set()
        day_parked = []
        for st in stays(still):
            # only a stay with a real parked track is a parked car; stationary fragments alone (a queue at a
            # red light) stay movers
            if not any(r["parked"] for r in st):
                continue
            main = max(st, key=lambda r: (r["parked"], r["span"]))
            day_parked.append({**main, "first_t": min(r["first_t"] for r in st),
                               "last_t": max(r["last_t"] for r in st),
                               "object_ids": sorted(r["id"] for r in st), "main": main})
            used.update(r["id"] for r in st)
            drops["parked_fragment"] += len(st) - 1
        pboxes = [_box(p) for p in day_parked]
        for r in day_rows:
            if r["id"] in used:
                continue
            # a stationary fragment on a parked car's box is a tracker ID switch of that car: already shown
            if _still(r) and any(_iou(_box(r), pb) >= 0.3 for pb in pboxes):
                drops["parked_fragment"] += 1
                continue
            movers.append({**r, "moves": not _still(r)})
        parked += day_parked
    moving = pd.DataFrame([{k: m[k] for k in cols} for m in movers], columns=cols)
    cap = cap if cap is not None else MAX_GHOSTS.get(cam, DEFAULT_CAP)
    if cap and len(moving) > cap:
        drops["cap"] = int(len(moving) - cap)
        moving = moving.sort_values(["moves", "span"], ascending=[False, False]).head(cap)
    return moving.sort_values("first_t").reset_index(drop=True), sorted(parked, key=lambda p: p["first_t"]), drops


def blur_head(crop: np.ndarray, frac: float = HEAD_FRAC) -> np.ndarray:
    """Gaussian-blur the top `frac` of a person crop (where the face is). Returns a new array."""
    h, w = crop.shape[:2]
    hh = int(np.ceil(h * frac))
    if hh < 1 or w < 1:
        return crop
    out = crop.copy()
    k = max(5, int(round(max(hh, w * 0.5))) | 1)
    top = min(h, hh + k // 2)  # blur a little past the cut, then keep only the head band
    blurred = cv2.GaussianBlur(cv2.GaussianBlur(crop[:top], (k, k), 0), (k, k), 0)
    out[:hh] = blurred[:hh]
    return out


# ---------------------------------------------------------------- timing

def samples(g: pd.DataFrame, t0: float, t1: float, speed: float) -> list[tuple[float, Any]]:
    """[(k * STEP out-offset, nearest detection row)] for one track; gaps > 0.2 s get no sample."""
    ts = g.t.to_numpy()
    out = []
    n = int(np.floor((t1 - t0) / speed / STEP + 1e-6)) + 1
    for k in range(n):
        st = t0 + k * STEP * speed
        i = int(np.searchsorted(ts, st))
        cand = [j for j in (i - 1, i) if 0 <= j < len(ts)]
        if not cand:
            continue
        j = min(cand, key=lambda j: abs(ts[j] - st))
        if abs(ts[j] - st) <= 0.2 * max(1.0, speed):
            out.append((k * STEP, g.iloc[j]))
    return out


def schedule(box_lists: list[list[tuple[float, np.ndarray]]], gw: int, gh: int,
             max_out: float = MAX_OUT) -> list[float]:
    """Greedy: in the given order, start each ghost at the out time where its boxes overlap the
    already-placed boxes the least (ties: earliest). box_lists hold (out-offset, grid box) pairs."""
    T = int(round(max_out / STEP))
    occ = np.zeros((T + 1, gh, gw), np.float32)
    starts = []
    for boxes in box_lists:
        if not boxes:
            starts.append(0.0)
            continue
        L = int(round(boxes[-1][0] / STEP)) + 1
        n_start = max(1, T + 1 - L + 1)
        cost = np.zeros(n_start, np.float64)
        slices = []
        for off, b in boxes:
            k = int(round(off / STEP))
            x1, y1, x2, y2 = b
            v = occ[:, y1:y2, x1:x2].sum(axis=(1, 2))
            seg = v[k:k + n_start]
            cost[:len(seg)] += seg
            slices.append((k, x1, y1, x2, y2))
        s = int(np.argmin(cost + np.arange(n_start) * 1e-6))
        for k, x1, y1, x2, y2 in slices:
            if s + k <= T:
                occ[s + k, y1:y2, x1:x2] += 1.0
        starts.append(s * STEP)
    return starts


def budget_for(cam: str, win_len: float, mover_spans: list[float]) -> float:
    """Output length: a fixed target for some cams, else room for ~CONC moving ghosts at once, at most half
    the window (always a real compression), at least the longest ghost, at most MAX_OUT."""
    if cam in OUT_TARGET:
        return float(min(MAX_OUT, OUT_TARGET[cam]))
    longest = min(max(mover_spans, default=STEP), MAX_OUT)
    b = max(longest, min(0.5 * win_len, max(MIN_OUT, sum(mover_spans) / CONC)))
    return float(np.clip(np.ceil(b / STEP) * STEP, 1.0, MAX_OUT))


def split_budget(total: float, loads: list[float]) -> list[float]:
    """Share the output between day segments by their moving load, each at least half an equal share."""
    n = len(loads)
    if n == 0:
        return []
    tot = sum(loads)
    sh = [max((x / tot) if tot > 0 else 1.0 / n, 0.5 / n) for x in loads]
    s = sum(sh)
    return [max(2.0, float(np.floor(total * x / s / STEP) * STEP)) for x in sh]


# ---------------------------------------------------------------- pixels

def redact_people(img: np.ndarray, boxes: Any) -> np.ndarray:
    """Blur every detected person's head in a source frame, in place, before any crop is cut from it, so a
    face never rides along inside another ghost (a pedestrian in front of a car). A person cut off by the
    bottom edge is right at the lens: all of the box is blurred."""
    H, W = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        x1, x2 = int(max(0, np.floor(x1))), int(min(W, np.ceil(x2)))
        y1, y2 = int(max(0, np.floor(y1))), int(min(H, np.ceil(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        frac = CUT_HEAD_FRAC if y2 >= H - 0.015 * H else HEAD_FRAC
        img[y1:y2, x1:x2] = blur_head(img[y1:y2, x1:x2], frac)
    return img


def _extract(cam: str, need: dict[int, list[tuple[Any, ...]]], scale: float, out_h: int,
             people: dict[int, Any] | None = None) -> dict[Any, np.ndarray]:
    """One sequential decode. need: frame index -> [(key, x1, y1, x2, y2) in source px]. people: frame index ->
    person boxes to redact first. Returns key -> crop already resized to synopsis pixels."""
    crops: dict[Any, np.ndarray] = {}
    if not need:
        return crops
    cap = cv2.VideoCapture(str(D.CAMS[cam]["video"]))
    last = max(need)
    for idx in range(last + 1):
        if not cap.grab():
            break
        if idx not in need:
            continue
        ok, img = cap.retrieve()
        if not ok:
            continue
        H, W = img.shape[:2]
        if people and idx in people:
            redact_people(img, people[idx])
        for key, x1, y1, x2, y2 in need[idx]:
            ox1, oy1, ox2, oy2 = _obox((x1, y1, x2, y2), scale, OUT_W, out_h)
            sx1, sy1 = int(max(0, round(ox1 / scale))), int(max(0, round(oy1 / scale)))
            sx2, sy2 = int(min(W, round(ox2 / scale))), int(min(H, round(oy2 / scale)))
            if ox2 - ox1 < 2 or oy2 - oy1 < 2 or sx2 - sx1 < 1 or sy2 - sy1 < 1:
                continue
            crops[key] = cv2.resize(img[sy1:sy2, sx1:sx2], (ox2 - ox1, oy2 - oy1), interpolation=cv2.INTER_AREA)
    cap.release()
    return crops


def _obox(b: tuple[float, float, float, float], scale: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = b
    return (int(np.clip(round(x1 * scale), 0, w)), int(np.clip(round(y1 * scale), 0, h)),
            int(np.clip(round(x2 * scale), 0, w)), int(np.clip(round(y2 * scale), 0, h)))


def _alpha(crop: np.ndarray, bgp: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Soft foreground mask (where the crop differs from the plate), so ghosts are shapes, not rectangles."""
    d = cv2.absdiff(crop, bgp).max(axis=2).astype(np.float32)
    m = np.clip((d - 14.0) / 26.0, 0.0, 1.0)
    h, w = m.shape
    k = max(3, (min(h, w) // 8) | 1)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    m = cv2.GaussianBlur(m, (k, k), 0)
    # Feather the crop border to zero so no rectangle edge ever shows, only the object's shape.
    r = max(2.0, min(h, w) / 6.0)
    ex = np.minimum(np.arange(w) + 0.5, w - 0.5 - np.arange(w)) / r
    ey = np.minimum(np.arange(h) + 0.5, h - 0.5 - np.arange(h)) / r
    m *= np.clip(ey[:, None], 0, 1) * np.clip(ex[None, :], 0, 1)
    return (m * alpha)[..., None]


def _paste(frame: np.ndarray, crop: np.ndarray, a: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    if crop.shape[0] != y2 - y1 or crop.shape[1] != x2 - x1:
        return
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (roi * (1 - a) + crop.astype(np.float32) * a).astype(np.uint8)


LABEL_FONT, LABEL_SCALE = cv2.FONT_HERSHEY_SIMPLEX, 0.42   # ~11 px line height


def label_rect(text: str, box: tuple[int, int, int, int], fw: int, fh: int) -> tuple[int, int, int, int]:
    """Where a time label goes: centred just above the ghost (inside its top edge if there is no room)."""
    x1, y1, x2, _ = box
    (tw, th), base = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, 1)
    tw, th = tw + 2, th + base + 2           # include the 1 px outline
    lx = int(np.clip((x1 + x2) / 2 - tw / 2, 0, fw - tw))
    ly = y1 - th - 2 if y1 - th - 2 >= 0 else min(y1 + 2, fh - th)
    return lx, ly, lx + tw, ly + th


def _overlaps(a: tuple[int, int, int, int], rects: list[tuple[int, int, int, int]]) -> bool:
    return any(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3] for b in rects)


def _label(frame: np.ndarray, text: str, rect: tuple[int, int, int, int]) -> None:
    """Plain light text with a 1 px dark outline: no box, no backing plate."""
    (_, th), _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, 1)
    org = (rect[0] + 1, rect[1] + 1 + th)
    cv2.putText(frame, text, org, LABEL_FONT, LABEL_SCALE, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, LABEL_FONT, LABEL_SCALE, (245, 245, 245), 1, cv2.LINE_AA)


def _corner(frame: np.ndarray, text: str) -> None:
    """The view / day name, bottom-left, in the same plain outlined style as the time labels."""
    (_, _), base = cv2.getTextSize(text, LABEL_FONT, 0.5, 1)
    org = (10, frame.shape[0] - 10 - base)
    cv2.putText(frame, text, org, LABEL_FONT, 0.5, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, LABEL_FONT, 0.5, (245, 245, 245), 1, cv2.LINE_AA)


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=D._np), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- build

def build(cam: str, verbose: bool = True) -> dict[str, Any]:
    t_start = time.time()
    info = D.info(cam)
    W, H = info["w"], info["h"]
    src_dur = info["duration"]
    scale = OUT_W / W
    out_h = int(round(H * scale / 2)) * 2
    view = choose_view(cam)
    w0, w1 = view["window"]
    blocks = blocks_for(cam, view["window"])
    multi = len(D.days(cam)) > 1
    for b in blocks:
        img = D.background(cam, b["day"]) if multi else D.background(cam)
        b["bg"] = cv2.resize(img, (OUT_W, out_h), interpolation=cv2.INTER_AREA)
    moving, parked, drops = select(cam, blocks, frame_h=H)
    tr = D.tracks(cam)
    tr = tr[tr.tid >= 0]
    groups = {int(k): g for k, g in tr.groupby("tid")}
    gw, gh = int(np.ceil(OUT_W / GRID)), int(np.ceil(out_h / GRID))

    mv = moving.to_dict("records")
    total = budget_for(cam, w1 - w0, [r["span"] for r in mv])
    lens = split_budget(total, [sum(r["span"] for r in mv if r["day"] == b["day"]) for b in blocks])

    need: dict[int, list[tuple[Any, ...]]] = {}
    ghosts: list[dict[str, Any]] = []
    o0 = 0.0
    for si, (b, L) in enumerate(zip(blocks, lens)):
        # Moving ghosts: sample the track every STEP of output time; speed up tracks that would not fit.
        plans = []
        for r in (r for r in mv if r["day"] == b["day"]):
            g = groups[int(r["id"])]
            g = g[(g.t >= r["first_t"] - 1e-6) & (g.t <= r["last_t"] + 1e-6)]
            speed = max(1.0, r["span"] / max(STEP, L - 2 * STEP))
            sm = samples(g, r["first_t"], r["last_t"], speed)
            if not sm:
                continue
            grid_boxes = []
            for off, row in sm:
                ob = _obox((row.x1, row.y1, row.x2, row.y2), scale, OUT_W, out_h)
                gb = (ob[0] // GRID, ob[1] // GRID, max(ob[0] // GRID + 1, -(-ob[2] // GRID)),
                      max(ob[1] // GRID + 1, -(-ob[3] // GRID)))
                grid_boxes.append((off, gb))
            plans.append({"r": r, "speed": speed, "samples": sm, "grid": grid_boxes})
        starts = schedule([p["grid"] for p in plans], gw, gh, max(STEP, L - STEP))
        end = 0.0
        for p, s0 in zip(plans, starts):
            r = p["r"]
            s0 = o0 + s0
            boxes = []
            for k, (off, row) in enumerate(p["samples"]):  # k = crop key, matches the box index
                ob = _obox((row.x1, row.y1, row.x2, row.y2), scale, OUT_W, out_h)
                boxes.append([round(s0 + off, 2), *ob])
                need.setdefault(int(row.f), []).append(((int(r["id"]), k), row.x1, row.y1, row.x2, row.y2))
            dur = r["span"] / p["speed"]
            ghosts.append({"object_id": int(r["id"]), "name": r["name"], "day": r["day"],
                           "src_start": round(r["first_t"], 2), "src_end": round(r["last_t"], 2),
                           "out_start": round(s0, 2), "out_end": round(s0 + dur, 2), "label": mmss(r["first_t"]),
                           "boxes": boxes, "parked": False, "speed": round(p["speed"], 2), "_seg": si,
                           "_src_t": [round(float(row.t), 3) for _, row in p["samples"]],
                           "_step": {int(round(off / STEP)): i for i, (off, _) in enumerate(p["samples"])}})
            end = max(end, s0 + dur - o0)
        bp = [p for p in parked if p["day"] == b["day"]]
        seg_len = round(max(end, 1.0), 2)
        b.update(out_start=round(o0, 2), out_end=round(o0 + seg_len, 2))
        # Parked ghosts: one still (the detection frame where the box is biggest) per stay. A stay that covers
        # most of the day holds its spot for the whole segment; a shorter one for its share of the day.
        bsrc = max(1e-6, b["src_end"] - b["src_start"])
        for p in bp:
            m = p["main"]
            g = groups[int(m["id"])]
            g = g[(g.t >= m["first_t"] - 1e-6) & (g.t <= m["last_t"] + 1e-6)]
            row = g.loc[(g.bw * g.bh).idxmax()]
            if (p["last_t"] - p["first_t"]) >= 0.8 * bsrc:
                a, e = 0.0, seg_len
            else:
                a = (p["first_t"] - b["src_start"]) / bsrc * seg_len
                e = min(seg_len, max(a + STEP, (p["last_t"] - b["src_start"]) / bsrc * seg_len))
                a = max(0.0, min(a, e - STEP))
            a, e = o0 + a, o0 + e
            ob = _obox(_box(m), scale, OUT_W, out_h)
            n = int(np.floor((e - a) / STEP + 1e-6)) + 1
            need.setdefault(int(row.f), []).append(((int(m["id"]), "p"), m["x1"], m["y1"], m["x2"], m["y2"]))
            ghosts.append({"object_id": int(m["id"]), "object_ids": p["object_ids"], "name": m["name"],
                           "day": p["day"], "src_start": round(p["first_t"], 2), "src_end": round(p["last_t"], 2),
                           "out_start": round(a, 2), "out_end": round(e, 2), "label": mmss(p["first_t"]),
                           "boxes": [[round(a + k * STEP, 2), *ob] for k in range(n)], "parked": True,
                           "speed": None, "_seg": si})
        o0 += seg_len
    out_dur = round(min(MAX_OUT, max(o0, 1.0)), 2)
    for g in ghosts:  # never past the end of the video
        g["out_end"] = min(g["out_end"], out_dur)
        g["boxes"] = [bx for bx in g["boxes"] if bx[0] <= out_dur + 1e-6]

    t_dec = time.time()
    allp = D.tracks(cam)
    allp = allp[(allp.name == "person") & allp.f.isin(list(need))]
    people = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy(float) for f, g in allp.groupby("f")}
    crops = _extract(cam, need, scale, out_h, people)
    t_dec = time.time() - t_dec
    n_blur = sum(len(v) for v in people.values())
    gone: dict[int, set[int]] = {}
    for g in ghosts:
        if g["name"] != "person" or g["parked"]:
            continue
        for k, bx in enumerate(g["boxes"]):
            key = (g["object_id"], k)
            if key not in crops:
                continue
            if bx[4] >= out_h - 8:
                # cut off by the bottom edge = right at the lens: not shown at all at this step
                crops.pop(key)
                gone.setdefault(id(g), set()).add(k)
            else:
                crops[key] = blur_head(crops[key])

    # Render to a temp file, then swap it in, so the UI never loads a half-written mp4.
    n_frames = max(1, int(round(out_dur * FPS)))
    od = D.out_dir(cam)
    mp4, tmp = od / "synopsis.mp4", od / "synopsis.tmp.mp4"
    ff = subprocess.Popen([str(FFMPEG), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{OUT_W}x{out_h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                           str(tmp)], stdin=subprocess.PIPE)
    parked_g = [g for g in ghosts if g["parked"]]
    moving_g = [g for g in ghosts if not g["parked"]]
    alphas: dict[Any, np.ndarray] = {}
    corner = [(b["label"] if len(blocks) > 1 else view["label"]) for b in blocks]
    fallback_bg = None if blocks else cv2.resize(D.background(cam), (OUT_W, out_h))

    def paste(frame: np.ndarray, key: Any, box: tuple[int, int, int, int], bg: np.ndarray, al: float) -> bool:
        c = crops.get(key)
        if c is None:
            return False
        if key not in alphas:
            alphas[key] = _alpha(c, bg[box[1]:box[3], box[0]:box[2]], al)
        _paste(frame, c, alphas[key], box)
        return True
    seg = 0
    for fi in range(n_frames):
        t = fi / FPS
        while seg < len(blocks) - 1 and t >= blocks[seg]["out_end"] - 1e-6:
            seg += 1
        bg = blocks[seg]["bg"] if blocks else fallback_bg
        frame = bg.copy()
        live: list[tuple[tuple[int, int, int, int], str]] = []
        for g in parked_g:
            if g["_seg"] == seg and g["out_start"] <= t <= g["out_end"] + 1e-6:
                # parked ghosts: half transparent, no box and no label (the UI shows details on hover)
                paste(frame, (g["object_id"], "p"), tuple(g["boxes"][0][1:]), bg, PARKED_ALPHA)
        for g in moving_g:
            if g["_seg"] != seg or not (g["out_start"] - 1e-6 <= t <= g["out_end"] + STEP):
                continue
            k = int(np.floor((t - g["out_start"]) / STEP + 1e-6))
            # boxes are sparse when the track has gaps: show the ghost only at steps that have a detection.
            bi = g["_step"].get(k)
            if bi is None or bi >= len(g["boxes"]):
                continue
            box = tuple(g["boxes"][bi][1:])
            if paste(frame, (g["object_id"], bi), box, bg, ALPHA):
                live.append((box, mmss(g["_src_t"][bi])))
        # moving ghosts: a small mm:ss source-time label just above each one, skipped when it would
        # overlap a label already drawn (earliest-scheduled ghosts win, so labels stay stable).
        taken: list[tuple[int, int, int, int]] = []
        for box, text in live:
            rect = label_rect(text, box, OUT_W, out_h)
            if _overlaps(rect, taken):
                continue
            taken.append(rect)
            _label(frame, text, rect)
        if blocks and corner[seg]:
            _corner(frame, corner[seg])
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    if ff.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"ffmpeg failed for {cam} (code {ff.returncode})")
    os.replace(tmp, mp4)

    n_gone = sum(len(v) for v in gone.values())
    kept = []
    for g in ghosts:
        for k in ("_src_t", "_step", "_seg"):
            g.pop(k, None)
        if id(g) in gone:  # steps not drawn (person at the lens) are not hoverable either
            g["boxes"] = [bx for k, bx in enumerate(g["boxes"]) if k not in gone[id(g)]]
            if len(g["boxes"]) < 2:  # out_start/out_end stay as scheduled: they map out time to source time
                continue
        kept.append(g)
    ghosts = kept
    moving_g = [g for g in ghosts if not g["parked"]]
    segments = [{k: b[k] for k in ("day", "label", "out_start", "out_end", "src_start", "src_end", "plate")}
                for b in blocks]
    res = {"camera": cam, "src_duration": round(src_dur, 2), "out_duration": out_dur, "fps": FPS,
           "w": OUT_W, "h": out_h, "view": view["view"], "view_label": view["label"],
           "src_window": [w0, w1], "plate": segments[0]["plate"] if segments else "background.jpg",
           "segments": segments,
           "skipped": {"other_view_tracks": drops["other_view"], "short": drops["short"],
                       "too_close_people": drops["too_close"],
                       "parked_fragments_merged": drops["parked_fragment"], "cap": drops["cap"]},
           "redacted": {"person_detections_blurred": n_blur, "lens_close_person_steps_hidden": n_gone},
           "ghosts": ghosts}
    _write_json_atomic(od / "synopsis.json", res)
    secs = time.time() - t_start
    n_boxes = sum(len(g["boxes"]) for g in moving_g)
    overlap = _overlap_stats(moving_g)
    sped = sum(1 for g in moving_g if g["speed"] and g["speed"] > 1)
    summary = [f"synopsis for {cam}: {src_dur:.1f} s of source, view {view['view']} "
               f"({w0:.0f}-{w1:.0f} s) -> {out_dur:.1f} s at {FPS:.0f} fps, {OUT_W}x{out_h}",
               "segments: " + "; ".join(f"{s['label']} src {s['src_start']:.0f}-{s['src_end']:.0f} s -> out "
                                        f"{s['out_start']:.1f}-{s['out_end']:.1f} s on {s['plate']}" for s in segments),
               f"ghosts: {len(ghosts)} ({len(moving_g)} moving, {len(parked_g)} parked static)",
               f"by class: {pd.Series([g['name'] for g in ghosts]).value_counts().to_dict() if ghosts else {}}",
               f"skipped: {drops['other_view']} tracks from another camera view, {drops['short']} shorter than "
               f"{MIN_SPAN:.0f} s, {drops['too_close']} people walking right past the lens, "
               f"{drops['parked_fragment']} parked-car fragments merged or dropped (ID switches), "
               f"{drops['cap']} by the cap of {MAX_GHOSTS.get(cam, DEFAULT_CAP)} (movers kept first)",
               f"moving ghost samples: {n_boxes} boxes, {len(crops)} crops cut from source frames; "
               f"{sped} long tracks sped up to fit; {n_blur} person detections head-blurred in source frames, "
               f"{n_gone} steps of people at the lens hidden",
               f"overlap: {overlap['pairs']} overlapping ghost-box pairs across {overlap['frames']} of "
               f"{overlap['steps']} 0.25 s steps",
               f"runtime: {secs:.1f} s (decode {t_dec:.1f} s)"]
    (od / "synopsis_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if verbose:
        print("\n".join(summary))
    return res


def _overlap_stats(moving_g: list[dict[str, Any]]) -> dict[str, int]:
    by_t: dict[float, list[list[float]]] = {}
    for g in moving_g:
        for b in g["boxes"]:
            by_t.setdefault(b[0], []).append(b[1:])
    pairs = frames = 0
    for bs in by_t.values():
        a = np.array(bs, float)
        if len(a) < 2:
            continue
        ix = np.clip(np.minimum(a[:, None, 2], a[None, :, 2]) - np.maximum(a[:, None, 0], a[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(a[:, None, 3], a[None, :, 3]) - np.maximum(a[:, None, 1], a[None, :, 1]), 0, None)
        p = int((np.triu((ix * iy) > 0, 1)).sum())
        pairs += p
        frames += p > 0
    return {"pairs": pairs, "frames": frames, "steps": len(by_t)}


def main(argv: list[str]) -> None:
    for cam in argv or DEFAULT_CAMS:
        build(cam)


if __name__ == "__main__":
    main(sys.argv[1:])
