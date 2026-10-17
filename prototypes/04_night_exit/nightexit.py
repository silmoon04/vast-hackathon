"""NIGHT EXIT core: exit-line crossings of stitched person tracks, best crop, clothing colours, ranking.

Pure functions; run.py wires them to the CLI and renders. Video time is the clock."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

# ----------------------------------------------------------------------------------------- config per clip


@dataclass
class Line:
    name: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    inside: tuple[float, float]          # a point on the "inside" (room) side; IN = towards it, OUT = away
    anchor: str = "foot"                 # 'foot' (bottom-centre of box) or 'centre'
    group: str = ""                      # logical door name when a door is drawn as several segments (default: name)

    def signed(self, x: float, y: float) -> float:
        """Signed distance in px, positive on the inside."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        dx, dy = x2 - x1, y2 - y1
        n = float(np.hypot(dx, dy))
        s = (dx * (y - y1) - dy * (x - x1)) / n
        si = (dx * (self.inside[1] - y1) - dy * (self.inside[0] - x1)) / n
        return s if si > 0 else -s

    def u(self, x: float, y: float) -> float:
        """Position along the segment, 0..1 between p1 and p2."""
        (x1, y1), (x2, y2) = self.p1, self.p2
        dx, dy = x2 - x1, y2 - y1
        return ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "p1": list(self.p1), "p2": list(self.p2), "inside": list(self.inside),
                "anchor": self.anchor, "group": self.group or self.name}


# Lines were drawn by eye on reference frames (viewed contact sheets / track starts and ends), see README.
CLIPS: dict[str, dict[str, Any]] = {
    "one_by_one_person_detection": {   # 768x432, top-down room; people walk in at the bottom, leave at the right edge
        "lines": [Line("E door", (665, 0), (665, 432), (300, 200), "centre"),
                  Line("S door", (0, 300), (560, 300), (300, 150), "centre")],
        "since": "00:21", "query": "light grey shirt and blue jeans", "stitch_gap": 1.5,
    },
    "rochester_night": {               # 1280x720, night street; one pedestrian walks left->right along the pavement
        "lines": [Line("Pavement gate", (1000, 430), (1000, 720), (500, 600), "foot")],
        "since": "00:00", "query": "dark clothes", "stitch_gap": 3.2, "motion_guard": True,
    },
    "retail_cctv": {                   # 1270x720, shop door is out of frame at the bottom right; mat leads to it
        "lines": [Line("Door", (940, 719), (1270, 440), (800, 400), "foot")],
        "since": "00:30", "query": "blue coat and pink top", "stitch_gap": 2.0,   # demo story: OUT, then back in 1 s later
    },
}
DEFAULT_SET = ["one_by_one_person_detection", "rochester_night", "retail_cctv"]


# --------------------------------------------------------------------------------------------- utilities


def parse_clock(s: str | float) -> float:
    """'00:12' -> 12, '01:00:30' -> 3630, '45' or 45.5 -> seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    parts = [float(p) for p in str(s).strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


def clock(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def read_frames(src: str, idxs: Sequence[int]) -> dict[int, np.ndarray]:
    """Decode only the wanted source frame indices in one sequential pass."""
    want = sorted(set(int(i) for i in idxs))
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(pk.video_path(src)))
    i, k = 0, 0
    while k < len(want):
        if not cap.grab():
            break
        if i == want[k]:
            ok, fr = cap.retrieve()
            if ok:
                out[i] = fr
            k += 1
        i += 1
    cap.release()
    return out


# ------------------------------------------------------------------------------------- track preparation


def person_rows(name: str) -> tuple[pd.DataFrame, pk.Tracks]:
    """Rows of every person track (majority class person, >= 4 person rows) + short mislabelled pre-roll fragments.

    A person entering at the frame edge is sometimes first boxed as 'dog'/'cat' for a few frames; such a short
    non-person fragment ending right where a person track starts at the frame edge is absorbed into it."""
    tr = pk.tracks(name)
    rows = tr.rows[tr.rows["tid"] >= 0]
    maj = rows.groupby("tid")["name"].agg(lambda x: x.value_counts().index[0])
    npers = rows[rows["name"] == "person"].groupby("tid").size()
    ptids = [t for t, m in maj.items() if m == "person" and npers.get(t, 0) >= 4]
    g = {t: d.sort_values("t") for t, d in rows.groupby("tid")}
    W, H = tr.w, tr.h
    remap: dict[int, int] = {}
    for t in g:
        if t in ptids or len(g[t]) > int(2.5 * tr.fps):
            continue
        last = g[t].iloc[-1]
        for p in ptids:
            f0 = g[p].iloc[0]
            edge = f0.x1 <= 6 or f0.y1 <= 6 or f0.x2 >= W - 6 or f0.y2 >= H - 6
            if edge and -0.4 <= f0.t - last.t <= 0.8 and iou((last.x1, last.y1, last.x2, last.y2),
                                                            (f0.x1, f0.y1, f0.x2, f0.y2)) >= 0.35:
                remap[t] = p
                break
    r = rows[rows["tid"].isin(ptids) | rows["tid"].isin(list(remap))].copy()
    r["tid"] = r["tid"].map(lambda t: remap.get(t, t))
    r = r.sort_values(["tid", "f", "conf"], ascending=[True, True, False]).drop_duplicates(["tid", "f"])
    r = r.sort_values(["tid", "t"]).reset_index(drop=True)
    return r, tr


def stitch(r: pd.DataFrame, fps: float, max_gap: float = 2.5, pos_tol: float = 0.9) -> pd.DataFrame:
    """Greedy constant-velocity stitching of track fragments: adds column pid (person identity).

    A (ending) is linked to B (starting later, gap in (0, max_gap] s) when B's first box is close to where A would be
    by then and box heights are similar. Conservative on purpose: a wrong merge can hide or invent a crossing."""
    g = {tid: d for tid, d in r.groupby("tid")}
    info = {}
    for tid, d in g.items():
        k = min(len(d) - 1, max(2, int(0.6 * fps)))
        a, b = d.iloc[-1], d.iloc[-1 - k]
        dt = max(a.t - b.t, 1e-3)
        vx, vy = (a.cx - b.cx) / dt, (a.y2 - b.y2) / dt
        sp = float(np.hypot(vx, vy))
        cap_sp = 0.9 * d.bh.median()      # px/s cap: a person moves < ~1 body height per second
        if sp > cap_sp:
            vx, vy = vx * cap_sp / sp, vy * cap_sp / sp
        info[tid] = dict(t0=d.t.iloc[0], t1=a.t, end=(a.cx, a.y2), start=(d.cx.iloc[0], d.y2.iloc[0]), v=(vx, vy),
                         bh0=d.bh.iloc[:3].median(), bh1=d.bh.iloc[-3:].median(), n=len(d))
    cands = []
    for ta, ia in info.items():
        for tb, ib in info.items():
            gap = ib["t0"] - ia["t1"]
            if ta == tb or gap <= 0 or gap > max_gap:
                continue
            px = ia["end"][0] + ia["v"][0] * gap
            py = ia["end"][1] + ia["v"][1] * gap
            err = float(np.hypot(ib["start"][0] - px, ib["start"][1] - py))
            h = max(ia["bh1"], ib["bh0"])
            ratio = ib["bh0"] / max(ia["bh1"], 1e-3)
            if err <= pos_tol * h * (0.5 + 0.5 * gap / max_gap) + 12 and 0.55 <= ratio <= 1.8:
                cands.append((err / h + 0.05 * gap, ta, tb))
    cands.sort()
    nxt, prv = {}, {}
    for _, ta, tb in cands:
        if ta in nxt or tb in prv:
            continue
        nxt[ta], prv[tb] = tb, ta
    pid = {}
    for tid in sorted(info, key=lambda k: info[k]["t0"]):
        pid[tid] = pid[prv[tid]] if tid in prv and prv[tid] in pid else tid
    r = r.copy()
    r["pid"] = r["tid"].map(pid)
    return r


def camera_blind_spans(name: str, *, step: float = 0.5, max_shift: float = 0.035
                       ) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Spans where the camera view no longer matches the reference (first) frame: exit lines are only valid
    while the view is the one the line was drawn on. Phase correlation of downscaled grey frames (upper 65%, away
    from passing traffic) vs frame 0, median-filtered over 3 samples."""
    meta = pk.info(name)
    stride = max(1, int(round(meta["fps"] * step)))
    ref, win = None, None
    ts: list[float] = []
    shifts: list[float] = []
    resps: list[float] = []
    for _, t, fr in pk.iter_frames(name, stride=stride):
        fr = fr[: int(fr.shape[0] * 0.65)]
        g = cv2.cvtColor(cv2.resize(fr, (320, int(320 * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        if ref is None:
            ref, win = g, cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ref, g, win)
        ts.append(t)
        shifts.append(float(np.hypot(dx, dy)) / 320.0)
        resps.append(float(resp))
    sh = np.asarray(shifts)
    # a different view has no strong correlation peak at all: peak < 20% of the typical reference-view peak
    weak = np.asarray(resps) < 0.2 * float(np.median(resps[:10]))
    sh = np.where(weak, 1.0, sh)
    if len(sh) >= 3:
        sh = np.median(np.stack([np.r_[sh[0], sh[:-1]], sh, np.r_[sh[1:], sh[-1]]]), axis=0)
    spans, start = [], None
    for t, s_ in zip(ts, sh):
        if s_ > max_shift and start is None:
            start = t
        if s_ <= max_shift and start is not None:
            spans.append((start, t))
            start = None
    if start is not None:
        spans.append((start, meta["duration"]))
    merged: list[tuple[float, float]] = []
    for a_, b_ in spans:                    # a view that is "back" for < 3 s is phase-correlation noise, not a valid view
        if merged and a_ - merged[-1][1] < 3.0:
            merged[-1] = (merged[-1][0], b_)
        else:
            merged.append((a_, b_))
    return merged, {"max_shift_frac": round(float(sh.max()), 3), "samples": len(ts)}


# ------------------------------------------------------------------------------------------- crossings


def anchor_xy(row: Any, anchor: str) -> tuple[float, float]:
    return (row.cx, row.y2) if anchor == "foot" else (row.cx, row.cy)


def find_crossings(r: pd.DataFrame, lines: Sequence[Line], *, min_dwell: float = 0.4,
                   blind: Sequence[tuple[float, float]] = ()) -> pd.DataFrame:
    """Every crossing of every door by every stitched identity.

    Columns: pid, line (door group), seg, dir (IN/OUT), t, x, y, f (nearest frame), inferred (True when the
    crossing happened inside a >0.6 s detection gap and was interpolated). Debounce: samples within a small
    band of the line are ignored (4-12 px, 4% of box height) and a side must be held >= min_dwell s unless it is
    the first or last stretch of the identity. The crossing point must lie on the segment."""
    ev = []
    for pid, d in r.groupby("pid"):
        d = d.sort_values("t")
        for ln in lines:
            conf = []
            for row in d.itertuples():
                x, y = anchor_xy(row, ln.anchor)
                sd = ln.signed(x, y)
                if abs(sd) >= float(np.clip(0.04 * row.bh, 4, 12)):
                    conf.append((row.t, x, y, row.f, sd))
            runs: list[list[Any]] = []          # runs of constant side; short interior runs are flicker
            for c in conf:
                if runs and (runs[-1][-1][4] > 0) == (c[4] > 0):
                    runs[-1].append(c)
                else:
                    runs.append([c])
            runs = [run for k, run in enumerate(runs) if k in (0, len(runs) - 1) or run[-1][0] - run[0][0] >= min_dwell]
            merged: list[Any] = []
            for run in runs:                    # re-join runs split by a dropped flicker run
                if merged and (merged[-1][-1][4] > 0) == (run[0][4] > 0):
                    merged[-1] = merged[-1] + run
                else:
                    merged.append(run)
            for ra, rb in zip(merged[:-1], merged[1:]):
                a, b = ra[-1], rb[0]
                w = a[4] / (a[4] - b[4])
                tx, ty = a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w
                if not -0.03 <= ln.u(tx, ty) <= 1.03:
                    continue
                t = a[0] + (b[0] - a[0]) * w
                if any(s_ <= t <= e_ for s_, e_ in blind):
                    continue
                ev.append({"pid": int(pid), "line": ln.group or ln.name, "seg": ln.name,
                           "dir": "IN" if b[4] > 0 else "OUT", "t": round(t, 2), "x": round(tx), "y": round(ty),
                           "f": int(a[3] if w < 0.5 else b[3]), "inferred": bool(b[0] - a[0] > 0.6)})
    out = pd.DataFrame(ev, columns=["pid", "line", "seg", "dir", "t", "x", "y", "f", "inferred"])
    out = out.sort_values("t").reset_index(drop=True)
    keep_rows, last = [], {}                    # same person, door, direction twice within 2 s (L corner) = one event
    for k, e in out.iterrows():
        key = (e.pid, e.line, e.dir)
        if key in last and e.t - last[key] < 2.0:
            continue
        last[key] = e.t
        keep_rows.append(k)
    return out.loc[keep_rows].reset_index(drop=True)


# ------------------------------------------------------------------------------------------ colour naming

COLOURS = ["black", "grey", "white", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"]
SYN = {"gray": "grey", "silver": "grey", "charcoal": "grey", "navy": "blue", "cyan": "blue", "teal": "blue",
       "turquoise": "blue", "dark": "black", "cream": "beige", "tan": "beige", "khaki": "beige", "ivory": "white",
       "maroon": "red", "burgundy": "red", "magenta": "pink", "violet": "purple", "denim": "blue"}
# partial credit when the person colour is a near neighbour of the queried one: (person colour, wanted colour)
NEAR = {("black", "grey"): 0.4, ("grey", "black"): 0.4, ("grey", "white"): 0.4, ("white", "grey"): 0.4,
        ("black", "blue"): 0.25, ("blue", "black"): 0.25, ("beige", "white"): 0.35, ("white", "beige"): 0.35,
        ("brown", "beige"): 0.35, ("beige", "brown"): 0.35, ("pink", "red"): 0.3, ("red", "pink"): 0.3,
        ("orange", "yellow"): 0.25, ("yellow", "orange"): 0.25, ("brown", "orange"): 0.25, ("orange", "brown"): 0.25,
        ("pink", "purple"): 0.3, ("purple", "pink"): 0.3, ("brown", "black"): 0.25, ("black", "brown"): 0.25}


def classify_pixels(bgr: np.ndarray) -> np.ndarray:
    """(N,3) BGR uint8 -> (N,) colour index into COLOURS."""
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    h, s, v = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    idx = {c: i for i, c in enumerate(COLOURS)}
    out = np.full(len(h), idx["grey"], np.int32)
    chroma = s >= 0.20
    out[(~chroma) & (v >= 0.78)] = idx["white"]
    out[(~chroma) & (v < 0.30)] = idx["black"]
    out[chroma & (v < 0.24)] = idx["black"]
    c = chroma & (v >= 0.24)
    hue = np.select([(h < 12) | (h >= 345), h < 42, h < 70, h < 170, h < 262, h < 300, h < 345],
                    [idx["red"], idx["orange"], idx["yellow"], idx["green"], idx["blue"], idx["purple"], idx["pink"]],
                    idx["red"])
    out[c] = hue[c]
    org = c & (out == idx["orange"])
    out[org & (v < 0.62)] = idx["brown"]
    out[org & (s < 0.45) & (v >= 0.62)] = idx["beige"]
    out[c & (out == idx["yellow"]) & (s < 0.40)] = idx["beige"]
    out[c & (out == idx["red"]) & (s < 0.45) & (v >= 0.7)] = idx["pink"]
    return out


def colour_hist(bgr_pixels: np.ndarray) -> dict[str, float]:
    """Share of region pixels per named colour (shares >= 4% kept, sorted)."""
    if len(bgr_pixels) < 12:
        return {}
    cls = classify_pixels(bgr_pixels)
    cnt = np.bincount(cls, minlength=len(COLOURS)).astype(float)
    cnt /= cnt.sum()
    return {COLOURS[i]: round(float(cnt[i]), 3) for i in np.argsort(-cnt) if cnt[i] >= 0.04}


def region_pixels(img: np.ndarray, box: Sequence[float], kp: Sequence[Sequence[float]] | None, part: str) -> np.ndarray:
    """Pixels of the torso ('top') or legs ('legs'). Uses pose keypoints when confident, else a box heuristic."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    rect = None
    if kp is not None:
        k = np.asarray(kp, float)
        if part == "top" and min(k[5][2], k[6][2], k[11][2], k[12][2]) > 0.5:
            xs, ys = [k[5][0], k[6][0], k[11][0], k[12][0]], [k[5][1], k[6][1], k[11][1], k[12][1]]
            cx0, cx1, cy0, cy1 = min(xs), max(xs), min(ys), max(ys)
            sw, sh = cx1 - cx0, cy1 - cy0
            if sw > 6 and sh > 8:
                rect = (cx0 + 0.15 * sw, cy0 + 0.08 * sh, cx1 - 0.15 * sw, cy1 - 0.15 * sh)
        if part == "legs" and min(k[11][2], k[12][2], k[13][2], k[14][2]) > 0.5:
            xs = [k[11][0], k[12][0], k[13][0], k[14][0]]
            ys = [k[11][1], k[12][1], k[13][1], k[14][1]]
            rect = (min(xs) - 0.05 * bw, min(ys) + 0.15 * (max(ys) - min(ys)), max(xs) + 0.05 * bw, max(ys))
    if rect is None:
        rect = ((x1 + 0.25 * bw, y1 + 0.20 * bh, x2 - 0.25 * bw, y1 + 0.50 * bh) if part == "top"
                else (x1 + 0.28 * bw, y1 + 0.60 * bh, x2 - 0.28 * bw, y1 + 0.92 * bh))
    a, b, c, d = (int(round(v)) for v in rect)
    a, c, b, d = max(0, min(a, c)), min(W, max(a, c)), max(0, min(b, d)), min(H, max(b, d))
    if c - a < 3 or d - b < 3:
        return np.zeros((0, 3), np.uint8)
    return img[b:d, a:c].reshape(-1, 3)


# --------------------------------------------------------------------------------------- query / ranking

TOP_WORDS = {"top", "shirt", "tshirt", "t-shirt", "jacket", "coat", "hoodie", "jumper", "sweater", "cardigan", "blouse",
             "fleece", "vest", "jersey", "sweatshirt", "clothes", "clothing"}
LEG_WORDS = {"trousers", "jeans", "pants", "skirt", "shorts", "leggings", "bottoms", "joggers"}


def parse_query(q: str) -> dict[str, Any]:
    """Free text -> colour constraints on top/legs, e.g. 'grey top, blue jeans' -> top: grey, legs: blue.
    A colour followed (within 3 words) by a garment word binds to that garment; a bare colour binds to the top;
    'jeans' alone means blue legs."""
    low = q.lower()
    toks = re.findall(r"[a-z\-]+", low)
    top: list[str] = []
    legs: list[str] = []
    bare: list[str] = []
    for i, tk in enumerate(toks):
        c = SYN.get(tk, tk)
        if c not in COLOURS:
            continue
        tgt = None
        for nx in toks[i + 1:i + 4]:
            if nx in TOP_WORDS:
                tgt = "top"
                break
            if nx in LEG_WORDS:
                tgt = "legs"
                break
            if SYN.get(nx, nx) in COLOURS:
                break
        (top if tgt == "top" else legs if tgt == "legs" else bare).append(c)
    if re.search(r"\bjeans\b", low) and not legs:
        legs.append("blue")
    top = top or bare
    return {"top": list(dict.fromkeys(top)), "legs": list(dict.fromkeys(legs)), "text": q.strip()}


def colour_score(hist: dict[str, float], wanted: Sequence[str]) -> float | None:
    """Mean over wanted colours of (share of that colour + partial credit for near colours), capped at 1."""
    if not wanted:
        return None
    if not hist:
        return 0.0
    sc = []
    for w in wanted:
        s = hist.get(w, 0.0) + sum(hist.get(c, 0.0) * v for (c, w2), v in NEAR.items() if w2 == w)
        sc.append(min(1.0, s / 0.6))        # 60% of the region in that colour counts as a full match
    return float(np.mean(sc))


# ------------------------------------------------------------------------------------- crops + appearance


def candidate_frames(r: pd.DataFrame, pid: int, t_ref: float, tr: pk.Tracks, k: int = 4, window: float = 5.0
                     ) -> list[Any]:
    """Up to k rows of identity `pid` near time t_ref: big, not clipped by the frame edge, not overlapped by others."""
    d = r[(r["pid"] == pid) & ((r["t"] - t_ref).abs() <= window)]
    if len(d) < 3:
        d = r[r["pid"] == pid]
    allr = r.groupby("f")
    scored = []
    for row in d.itertuples():
        clipped = row.x1 <= 3 or row.y1 <= 3 or row.x2 >= tr.w - 3 or row.y2 >= tr.h - 3
        others = allr.get_group(row.f)
        mx = 0.0
        for o in others.itertuples():
            if o.pid != pid:
                mx = max(mx, iou((row.x1, row.y1, row.x2, row.y2), (o.x1, o.y1, o.x2, o.y2)))
        scored.append((row.bw * row.bh * row.conf * (0.45 if clipped else 1.0) * (1.0 - 0.8 * min(mx * 2, 1.0)), row))
    scored.sort(key=lambda z: -z[0])
    picked: list[Any] = []
    for sc, row in scored:
        if all(abs(row.t - p.t) >= 0.4 for _, p in picked):
            picked.append((sc, row))
        if len(picked) >= k:
            break
    return picked


def sharpness(crop: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(crop, (96, max(8, int(96 * crop.shape[0] / max(crop.shape[1], 1)))),
                                interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


_POSE_CPU: Any = None


def cpu_pose(frames: Sequence[np.ndarray], conf: float = 0.25) -> list[list[dict[str, Any]]]:
    """YOLO26s-pose on the CPU (2 threads), same output shape as pk.pose_many. A few dozen frames only; it avoids
    waiting for the shared GPU lock (protokit's lock is unfair: one run waited 11 min for it)."""
    global _POSE_CPU
    if _POSE_CPU is None:
        import torch
        torch.set_num_threads(2)
        from ultralytics import YOLO
        _POSE_CPU = YOLO(str(pk.MODELS / "yolo26s-pose.pt"), task="pose")
    out = []
    for fr in frames:
        res = _POSE_CPU.predict(fr, conf=conf, imgsz=640, verbose=False, device="cpu")[0]
        kps = res.keypoints.data.tolist() if res.keypoints is not None else []
        out.append([{"conf": cf, "box": xy, "kp": kp} for cf, xy, kp in
                    zip(res.boxes.conf.tolist(), res.boxes.xyxy.tolist(), kps)])
    return out


def cached_poses(name: str, frames: dict[int, np.ndarray], want: Sequence[int], budget: float = 150.0
                 ) -> dict[str, Any]:
    """Pose results per source frame index, cached in ./cache. New frames are computed on the CPU in a worker thread
    with a time budget: when the machine is saturated the caller continues without keypoints (box heuristic for the
    colour regions) and the worker still fills the cache for the next run."""
    import threading
    cache_f = Path(__file__).resolve().parent / "cache" / f"pose_{name}.json"
    pcache: dict[str, Any] = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    need = sorted({f for f in want if str(f) not in pcache and f in frames})
    if need:
        def work() -> None:
            for f_, res in zip(need, cpu_pose([frames[f_] for f_ in need])):
                pcache[str(f_)] = res
            cache_f.parent.mkdir(exist_ok=True)
            cache_f.write_text(json.dumps(pcache))
        th = threading.Thread(target=work, daemon=True, name="pose")
        th.start()
        th.join(timeout=budget)
        if th.is_alive():
            print(f"[night_exit] pose budget {budget:.0f}s exceeded: using box heuristics (worker keeps filling cache)", file=sys.stderr)
    return pcache


def appearance(name: str, r: pd.DataFrame, tr: pk.Tracks, ev: pd.DataFrame, out_dir: Path, *, use_vlm: bool = True,
               max_vlm: int = 30) -> dict[int, dict[str, Any]]:
    """Per identity that crossed a door: best crop (largest x sharpest of 4 candidates), torso/legs colours (pose
    keypoints, median over candidates), one-line Cosmos description (cached, local stand-in)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    people: dict[int, dict[str, Any]] = {}
    cand: dict[int, list[Any]] = {}
    for pid, e in ev.groupby("pid"):
        cand[int(pid)] = candidate_frames(r, int(pid), float(e["t"].iloc[0]), tr)
    frames = read_frames(name, [row.f for c in cand.values() for _, row in c])
    keys = [(pid, i) for pid, c in cand.items() for i in range(len(c)) if c[i][1].f in frames]
    pcache = cached_poses(name, frames, [cand[p][i][1].f for p, i in keys])
    pose_of = {(p, i): pcache.get(str(cand[p][i][1].f), []) for p, i in keys}
    for pid, c in cand.items():
        best, hist_top, hist_legs, used = None, [], [], 0
        for i, (sc, row) in enumerate(c):
            if row.f not in frames:
                continue
            img = frames[row.f]
            box = (row.x1, row.y1, row.x2, row.y2)
            kp, bi = None, 0.0
            for p in pose_of.get((pid, i), []):
                v = iou(box, p["box"])
                if v > bi:
                    bi, kp = v, p["kp"]
            if bi < 0.4:
                kp = None
            top = colour_hist(region_pixels(img, box, kp, "top"))
            legs = colour_hist(region_pixels(img, box, kp, "legs"))
            hist_top.append(top)
            hist_legs.append(legs)
            x1, y1, x2, y2 = (int(v) for v in box)
            pad = int(0.06 * (y2 - y1))
            crop = img[max(0, y1 - pad):min(img.shape[0], y2 + pad), max(0, x1 - pad):min(img.shape[1], x2 + pad)]
            if crop.size == 0:
                continue
            q = sc * (0.5 + min(sharpness(crop), 400.0) / 400.0)
            if best is None or q > best[0]:
                best = (q, crop.copy(), row, kp is not None)
            used += 1
        if best is None:
            continue

        def merge(hs: list[dict[str, float]]) -> dict[str, float]:
            acc: dict[str, float] = {}
            for h in hs:
                for k_, v in h.items():
                    acc[k_] = acc.get(k_, 0.0) + v / max(len(hs), 1)
            return {k_: round(v, 3) for k_, v in sorted(acc.items(), key=lambda z: -z[1]) if v >= 0.05}

        crop = best[1]
        s = 360.0 / max(crop.shape[0], 1)
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 360), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        path = out_dir / f"pid{pid}.jpg"
        pk.save_jpg(crop, path)
        people[pid] = {"crop": path, "crop_bgr": crop, "t_crop": float(best[2].t), "box": [best[2].x1, best[2].y1, best[2].x2, best[2].y2],
                       "top": merge(hist_top), "legs": merge(hist_legs), "pose_ok": best[3], "n_cand": used}
    prompt = ("Describe the person in this image in one short line: the colour and type of the top, the colour "
              "of the trousers or skirt, and the hair. Plain words, no guessing about identity.")
    order = sorted(people)[:max_vlm]
    if use_vlm and order:                  # the shared Cosmos server queues requests (30-120 s each under load): overlap them
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as ex:
            texts = list(ex.map(lambda k: pk.vlm(prompt, [people[k]["crop_bgr"]], max_tokens=60), order))
    else:
        texts = []
    for pid, p in people.items():
        p["desc"] = ""
    for pid, txt in zip(order, texts):
        people[pid]["desc"] = " ".join(txt.split())
    return people


def embed_scores(query: str, descs: dict[int, str]) -> dict[int, float]:
    """cosine(Cosmos-Embed1 text(query), Cosmos-Embed1 text(description)); {} when the embed server is unavailable."""
    ids = [k for k, v in descs.items() if v]
    if not ids or not query.strip():
        return {}
    try:
        q = pk.embed_text([query])[0]
        d = pk.embed_text([descs[k] for k in ids])
    except Exception as e:  # server down: degrade, colour match still ranks
        print(f"[night_exit] embed unavailable: {type(e).__name__}", file=sys.stderr)
        return {}
    q = q / (np.linalg.norm(q) + 1e-9)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return {k: float(d[i] @ q) for i, k in enumerate(ids)}


def desc_mentions(desc: str, wanted: Sequence[str]) -> float | None:
    if not wanted or not desc:
        return None
    low = desc.lower().replace("gray", "grey")
    words = set(re.findall(r"[a-z]+", low))
    hit = 0
    for w in wanted:
        alts = {w} | {k for k, v in SYN.items() if v == w}
        hit += 1 if alts & words else 0
    return hit / len(wanted)


GARMENTS = [{"coat", "jacket", "parka", "overcoat", "blazer", "anorak", "puffer"}, {"hoodie", "hooded", "hoody", "sweatshirt"},
            {"cardigan", "jumper", "sweater", "knit", "knitted", "pullover", "fleece"}, {"t-shirt", "tshirt", "tee"},
            {"shirt", "blouse"}, {"dress"}, {"skirt"}, {"jeans"}, {"trousers", "pants", "leggings", "joggers"}, {"shorts"}]


def garment_mention(query: str, desc: str) -> float | None:
    """Specific garment words in the query (coat, hoodie, cardigan...) vs the Cosmos description: 1 when the description
    uses a word of the same family, else 0. None when the query names no specific garment ('top' / 'clothes' are generic)."""
    if not desc:
        return None
    q = set(re.findall(r"[a-z\-]+", query.lower()))
    d = set(re.findall(r"[a-z\-]+", desc.lower()))
    fams = [f for f in GARMENTS if f & q]
    if not fams:
        return None
    return sum(1.0 for f in fams if f & d) / len(fams)


W_PIX, W_VLM, W_EMB = 0.55, 0.20, 0.25


def rank_people(people: dict[int, dict[str, Any]], query: str, mode: str = "combined") -> list[dict[str, Any]]:
    """Score every person against the description. mode: combined | colour | embed | text. Returns best first."""
    pq = parse_query(query)
    emb = embed_scores(query, {k: p["desc"] for k, p in people.items()}) if mode in ("combined", "embed") else {}
    if emb:
        lo, hi = min(emb.values()), max(emb.values())
    rows = []
    for pid, p in people.items():
        parts = {}
        cs = [colour_score(p["top"], pq["top"]), colour_score(p["legs"], pq["legs"])]
        cs = [c for c in cs if c is not None]
        parts["pix"] = float(np.mean(cs)) if cs else None
        vparts = [v for v in (desc_mentions(p["desc"], pq["top"] + pq["legs"]), garment_mention(query, p["desc"])) if v is not None]
        parts["vlm"] = float(np.mean(vparts)) if vparts else None
        if pid in emb:
            parts["emb"] = 0.5 if hi - lo < 1e-6 else (emb[pid] - lo) / (hi - lo)
            parts["emb_raw"] = emb[pid]
        else:
            parts["emb"] = None
        # absolute (not min-maxed) evidence: pixel-colour match + Cosmos-description match only
        ab = [(W_PIX, parts["pix"]), (W_VLM, parts["vlm"])]
        ab = [(w, v) for w, v in ab if v is not None]
        parts["abs"] = (sum(w * v for w, v in ab) / sum(w for w, _ in ab)) if ab else None
        use = {"combined": {"pix": W_PIX, "vlm": W_VLM, "emb": W_EMB}, "colour": {"pix": 1.0}, "embed": {"emb": 1.0},
               "text": {"vlm": 1.0}}[mode]
        num = sum(w * parts[k] for k, w in use.items() if parts.get(k) is not None)
        den = sum(w for k, w in use.items() if parts.get(k) is not None)
        rows.append({"pid": pid, "score": (num / den) if den else 0.0, **{f"s_{k}": v for k, v in parts.items()}})
    rows.sort(key=lambda z: -z["score"])
    return rows
