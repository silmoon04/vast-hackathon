"""Where things move: heat trails of every moving track over the background plate, plus the main lanes.

Outputs (per camera, under out/<cam>/):
  flow.json          lanes, per-minute counts and the busiest moment (contract "flow" in CONTRACTS.md)
  flow.jpg           heat trail (people amber, vehicles blue) with lane arrows and counts, >= 1280 px wide
  frames/busiest.jpg the frame with the most moving tracks on screen at once, boxes drawn
  flow_summary.txt   what was computed, with counts

Run: python -m stm.flow [cams...]
"""
from __future__ import annotations

import math
import sys
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from stm import data

# BGR. Contract: people amber #F2A93B, vehicles blue #7FB6FF.
COLOURS = {"person": (59, 169, 242), "vehicle": (255, 182, 127)}
MIN_FRAMES = 5          # a track needs this many detections to count at all
THIN_S = 0.25           # trails and paths use one smoothed point per THIN_S seconds
PATH_POINTS = 20        # lane path = mean trajectory resampled to this many points
MAX_LANES = 8
LANE_CUT = 0.25        # lane clustering distance cut in normalised (entry, exit) space
OUT_W = 1280            # flow.jpg is at least this wide
SAMPLE_S = 0.5          # camera-view check: one frame per SAMPLE_S seconds
VIEW_THR = 0.4          # under 40% of the textured tiles still matching the main view = different view
TILE_NCC = 0.6          # a tile "matches" when its masked correlation with the main view is at least this
TILES = (6, 4)          # tile grid (columns, rows) on the thumbnail
SW, SH = 160, 90        # size of the view-check thumbnails
DUP_IOU = 0.5           # busiest frame: same-group boxes overlapping this much are one object


def group_of(name: str) -> str:
    """person -> person; bicycle, car, truck, bus, motorcycle -> vehicle."""
    return "person" if name == "person" else "vehicle"


# ----------------------------------------------------------------------------- camera view
def _small(im: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(im, (SW, SH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def view_score(g: np.ndarray, ref: np.ndarray, m: np.ndarray) -> float:
    """Share of textured tiles of thumbnail g that still match the main view `ref`.

    Pixels outside mask m (detection boxes) are ignored, so passing cars and crowds do not count. A tile is
    usable when at least half its pixels are free and the reference has some texture there (std >= 4);
    it matches when the zero-mean correlation over the free pixels is >= TILE_NCC (this ignores brightness
    and contrast changes). A pan, zoom or a different scene fails most tiles; a passing object only a few.
    nan when fewer than 3 tiles are usable.
    """
    H, W = g.shape
    nx, ny = TILES
    ok = n = 0
    for i in range(ny):
        for j in range(nx):
            sl = (slice(i * H // ny, (i + 1) * H // ny), slice(j * W // nx, (j + 1) * W // nx))
            mm = m[sl]
            if mm.mean() < 0.5:
                continue
            x, y = g[sl][mm], ref[sl][mm]
            if y.std() < 4.0:
                continue
            x, y = x - x.mean(), y - y.mean()
            d = math.sqrt(float((x * x).sum() * (y * y).sum()))
            n += 1
            ok += d > 1e-3 and float((x * y).sum()) / d >= TILE_NCC
    return ok / n if n >= 3 else float("nan")


def camera_views(cam: str) -> dict[str, Any]:
    """Which seconds show the camera's main view.

    Reads one frame per SAMPLE_S s, masks out every detection box near that time (so passing cars and
    crowds do not count), and scores the rest against the main view (median of the better-matching half
    of all samples) with view_score (tile agreement). Samples under VIEW_THR, plus one sample either side, are excluded as "view differs"
    (PTZ move, zoom, glare or blockage). Also returns a full-resolution plate for the main view.
    """
    i = data.info(cam)
    W, H, fps = i["w"], i["h"], i["fps"]
    step = max(1, int(round(fps * SAMPLE_S)))
    cap = cv2.VideoCapture(str(data.CAMS[cam]["video"]))
    n_est = max(1, i["frames"] // step)
    keep_every = max(1, math.ceil(n_est / 48))
    ts, small, full = [], [], {}
    f = 0
    while cap.grab():
        if f % step == 0:
            ok, im = cap.retrieve()
            if ok:
                if len(ts) % keep_every == 0:
                    full[len(ts)] = im
                ts.append(f / fps)
                small.append(_small(im))
        f += 1
    cap.release()
    ts_a = np.array(ts)
    dt = step / fps
    if not len(ts_a):
        return {"ts": ts_a, "dt": dt, "main": np.zeros(0, bool), "score": np.zeros(0), "plate": None,
                "excluded": [], "main_s": 0.0}
    G = np.stack(small)
    M = np.ones(G.shape, bool)
    df = data.tracks(cam)
    tt, bx = df.t.to_numpy(), df[["x1", "y1", "x2", "y2"]].to_numpy()
    order = np.argsort(tt, kind="stable")
    tt, bx = tt[order], bx[order]
    for k, t in enumerate(ts_a):
        lo, hi = np.searchsorted(tt, t - 0.3), np.searchsorted(tt, t + 0.3)
        for x1, y1, x2, y2 in bx[lo:hi]:
            pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            M[k, max(0, int((y1 - ph) * SH / H)):int(math.ceil((y2 + ph) * SH / H)),
              max(0, int((x1 - pw) * SW / W)):int(math.ceil((x2 + pw) * SW / W))] = False
    ref = np.median(G, axis=0)
    sc = np.array([view_score(G[k], ref, M[k]) for k in range(len(G))])
    top = sc >= np.nanmedian(sc) if np.isfinite(sc).any() else np.ones(len(sc), bool)
    ref = np.median(G[top], axis=0)
    sc = np.array([view_score(G[k], ref, M[k]) for k in range(len(G))])
    bad = np.nan_to_num(sc, nan=1.0) < VIEW_THR   # nan = too covered to judge: keep
    bad = bad | np.r_[False, bad[:-1]] | np.r_[bad[1:], False]
    main = ~bad
    dur = i["duration"]
    excluded = []
    k = 0
    while k < len(main):
        if not main[k]:
            j = k
            while j + 1 < len(main) and not main[j + 1]:
                j += 1
            seg = sc[k:j + 1]
            excluded.append({"start": round(max(0.0, ts_a[k] - dt / 2), 2), "end": round(min(dur, ts_a[j] + dt / 2), 2),
                             "reason": "frame differs from the main camera view (camera moved, zoomed, glare or blocked)",
                             "min_score": round(float(np.nanmin(seg)), 3) if np.isfinite(seg).any() else None})
            k = j + 1
        else:
            k += 1
    plate = None
    if bad.mean() > 0.05:  # the shared background mixes views: build one from the longest main-view run
        runs, a = [], None
        for k2, okk in enumerate(list(main) + [False]):
            if okk and a is None:
                a = k2
            elif not okk and a is not None:
                runs.append((a, k2))
                a = None
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
            good = [im for idx, im in full.items() if a <= idx < b]
            if len(good) < 3:
                good = [im for idx, im in full.items() if main[idx]]
            if good:
                plate = np.median(np.stack(good), axis=0).astype(np.uint8)
    return {"ts": ts_a, "dt": dt, "main": main, "score": sc, "plate": plate, "excluded": excluded,
            "main_s": round(_covered(ts_a, dt, dur, main, 0.0, dur), 2), "duration": dur}


def _covered(ts: np.ndarray, dt: float, dur: float, sel: np.ndarray, a: float, b: float) -> float:
    """Seconds of [a, b) covered by the selected samples. Sample k stands for [t_k - dt/2, t_k + dt/2),
    clipped to [0, dur); the last sample also stands for the tail of the clip after it."""
    if not len(ts):
        return 0.0
    lo = np.clip(ts - dt / 2, 0.0, dur)
    hi = np.clip(ts + dt / 2, 0.0, dur)
    hi[-1] = dur
    lo, hi = np.maximum(lo[sel], a), np.minimum(hi[sel], b)
    return float(np.clip(hi - lo, 0, None).sum())


def visible_mask(views: dict[str, Any], t: np.ndarray) -> np.ndarray:
    """True for times whose nearest view sample shows the main view."""
    if not len(views["ts"]):
        return np.zeros(len(t), bool)
    idx = np.clip(np.round(np.asarray(t) / views["dt"]).astype(int), 0, len(views["main"]) - 1)
    return views["main"][idx]


# ----------------------------------------------------------------------------- tracks
def _thin(g: pd.DataFrame) -> np.ndarray:
    """[[t, cx, cy], ...] with one mean point per THIN_S seconds, lightly smoothed."""
    b = (g.t.to_numpy() // THIN_S).astype(np.int64)
    a = g.assign(_b=b).groupby("_b")[["t", "cx", "cy"]].mean().to_numpy()
    if len(a) >= 5:  # 3-point moving average on x, y, keep end points
        sm = a.copy()
        sm[1:-1, 1:] = (a[:-2, 1:] + a[1:-1, 1:] + a[2:, 1:]) / 3.0
        a = sm
    return a


def track_table(cam: str, views: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify every tracked mover as parked, stationary or moving; return the moving ones with their paths.

    With `views`, detections outside the main camera view are dropped first.
    """
    i = data.info(cam)
    W, H = i["w"], i["h"]
    df = data.tracks(cam)
    df = df[(df.tid >= 0) & df.name.isin(data.MOVERS)]
    n_all = len(df)
    if views is not None:
        df = df[visible_mask(views, df.t.to_numpy())]
    obj = data.objects(cam, min_frames=MIN_FRAMES)
    parked_ids = set(int(x) for x in obj.id[obj.parked]) if len(obj) else set()
    moving, n_parked, n_still, n_short = [], 0, 0, 0
    for tid, g in df.groupby("tid", sort=False):
        tid = int(tid)
        if len(g) < MIN_FRAMES:
            n_short += 1
            continue
        if tid in parked_ids:
            n_parked += 1
            continue
        g = g.sort_values("t")
        name = g.name.mode().iat[0]
        k = max(1, min(5, len(g) // 5))
        a = np.array([g.cx.iloc[:k].median(), g.cy.iloc[:k].median()])
        z = np.array([g.cx.iloc[-k:].median(), g.cy.iloc[-k:].median()])
        net = float(np.hypot(*(z - a)))
        size = float(np.sqrt((g.bw * g.bh).clip(lower=1)).median())
        if net < max(0.02 * W, 0.5 * size):
            n_still += 1
            continue
        pts = _thin(g)
        span = float(g.t.iat[-1] - g.t.iat[0])
        length = float(np.hypot(np.diff(pts[:, 1]), np.diff(pts[:, 2])).sum()) if len(pts) > 1 else net
        moving.append({"tid": tid, "name": name, "group": group_of(name), "n": len(g),
                       "t0": float(g.t.iat[0]), "t1": float(g.t.iat[-1]),
                       "entry": a, "exit": z, "net": net, "pts": pts,
                       "speed": length / span if span > 0 else 0.0})
    return {"W": W, "H": H, "duration": i["duration"], "moving": moving, "parked": n_parked,
            "stationary": n_still, "short": n_short, "df": df, "views": views,
            "dropped_dets": n_all - len(df)}


# ----------------------------------------------------------------------------- lanes
def resample(xy: np.ndarray, n: int = PATH_POINTS) -> np.ndarray:
    """Resample a polyline [[x, y], ...] to n points evenly spaced by arc length."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 1:
        return np.repeat(xy, n, axis=0)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(xy[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, xy[:, 0]), np.interp(u, s, xy[:, 1])], axis=1)


def cluster_tracks(tr: list[dict], W: int, H: int, cut: float = LANE_CUT) -> list[list[dict]]:
    """Cluster tracks of one group by (entry point, exit point), normalised to the frame size.

    Average-linkage hierarchical clustering on [entry_x/W, entry_y/H, exit_x/W, exit_y/H], cut at `cut`
    (so the number of lanes follows from the data; opposite directions never share a lane).
    Returns clusters, biggest first.
    """
    if not tr:
        return []
    X = np.array([[t["entry"][0] / W, t["entry"][1] / H, t["exit"][0] / W, t["exit"][1] / H] for t in tr])
    if len(tr) == 1:
        lab = np.array([1])
    else:
        lab = fcluster(linkage(X, method="average"), t=cut, criterion="distance")
    groups: dict[int, list[dict]] = {}
    for t, l in zip(tr, lab):
        groups.setdefault(int(l), []).append(t)
    return sorted(groups.values(), key=len, reverse=True)


def lane_name(path: np.ndarray, W: int, H: int) -> str:
    """Plain words for a path in screen terms, e.g. 'left to right along the bottom'."""
    p = np.asarray(path, dtype=float)
    dx, dy = p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]  # direction judged in pixels, as seen on screen
    mx, my = p[:, 0].mean() / W, p[:, 1].mean() / H
    third = lambda v, words: words[min(len(words) - 1, int(v * len(words)))] if v >= 0 else words[0]
    h = "left to right" if dx > 0 else "right to left"
    v = "top to bottom" if dy > 0 else "bottom to top"
    if abs(dx) >= 2 * abs(dy):
        return f"{h} {third(my, ['along the top', 'across the upper middle', 'across the middle', 'across the lower middle', 'along the bottom'])}"
    if abs(dy) >= 2 * abs(dx):
        return f"{v} {third(mx, ['on the far left', 'left of centre', 'through the centre', 'right of centre', 'on the far right'])}"
    a = ("top" if dy > 0 else "bottom") + (" left" if dx > 0 else " right")
    b = ("bottom" if dy > 0 else "top") + (" right" if dx > 0 else " left")
    return f"{a} to {b}"


def build_lanes(tt: dict[str, Any]) -> tuple[list[dict], dict[str, int]]:
    W, H = tt["W"], tt["H"]
    cand = []
    other = {"person": 0, "vehicle": 0}
    for grp in ("vehicle", "person"):
        tr = [t for t in tt["moving"] if t["group"] == grp]
        min_count = 1 if len(tr) < 15 else max(2, math.ceil(0.03 * len(tr)))
        for members in cluster_tracks(tr, W, H):
            if len(members) < min_count:
                other[grp] += len(members)
                continue
            path = np.mean([resample(m["pts"][:, 1:]) for m in members], axis=0)
            if np.hypot(*(path[-1] - path[0])) < 0.03 * W:  # mixed directions cancel out: not a lane
                other[grp] += len(members)
                continue
            c = np.mean([np.r_[m["entry"], m["exit"]] for m in members], axis=0)
            d = [np.linalg.norm(np.r_[m["entry"], m["exit"]] - c) for m in members]
            ex = [members[i]["tid"] for i in np.argsort(d)[:5]]
            cand.append({"group": grp, "count": len(members), "path": path,
                         "median_speed_px_s": float(np.median([m["speed"] for m in members])),
                         "example_ids": ex, "members": [m["tid"] for m in members]})
    cand.sort(key=lambda l: l["count"], reverse=True)
    for l in cand[MAX_LANES:]:
        other[l["group"]] += l["count"]
    lanes, seen = [], {}
    for i, l in enumerate(cand[:MAX_LANES]):
        nm = lane_name(l["path"], W, H)
        key = (l["group"], nm)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            nm = f"{nm} ({seen[key]})"
        p = l["path"]
        lanes.append({"lane_id": f"L{i + 1}", "name": nm, "group": l["group"], "count": int(l["count"]),
                      "from": [round(float(p[0, 0]), 1), round(float(p[0, 1]), 1)],
                      "to": [round(float(p[-1, 0]), 1), round(float(p[-1, 1]), 1)],
                      "path": [[round(float(x), 1), round(float(y), 1)] for x, y in p],
                      "median_speed_px_s": round(l["median_speed_px_s"], 2),
                      "example_ids": [int(x) for x in l["example_ids"]],
                      "_members": l["members"]})
    return lanes, other


# ----------------------------------------------------------------------------- counts
def per_minute(tt: dict[str, Any]) -> list[dict]:
    """Unique moving track ids per group in each minute [t, t+60)."""
    grp = {t["tid"]: t["group"] for t in tt["moving"]}
    df = tt["df"][tt["df"].tid.isin(grp)]
    n_min = max(1, math.ceil(tt["duration"] / 60.0))
    rows = []
    for m in range(n_min):
        seg = df[(df.t >= 60 * m) & (df.t < 60 * (m + 1))]
        ids = seg.tid.unique()
        c = {"person": 0, "vehicle": 0}
        for i in ids:
            c[grp[int(i)]] += 1
        row = {"t": 60 * m, "person": c["person"], "vehicle": c["vehicle"],
               "seconds": round(min(60.0, tt["duration"] - 60 * m), 2)}
        v = tt.get("views")
        if v is not None and len(v["ts"]):
            row["visible_s"] = round(_covered(v["ts"], v["dt"], tt["duration"], v["main"], 60.0 * m,
                                              min(60.0 * (m + 1), tt["duration"])), 2)
            if row["visible_s"] == 0:
                row["person"] = row["vehicle"] = None
                row["unknown"] = "main camera view not visible in this minute"
        rows.append(row)
    return rows


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def dedupe_frame(rows: pd.DataFrame, iou: float = DUP_IOU) -> pd.DataFrame:
    """One detection per object in a frame: drop a box overlapping a more confident box of the same group
    (person / vehicle, by that frame's class) with IoU >= `iou`. The tracker sometimes hands one object
    to a new id while the old id still has a box in the same frame; those must not count twice."""
    rows = rows.sort_values("conf", ascending=False)
    keep, kept = [], []
    for idx, r in zip(rows.index, rows.itertuples()):
        b = np.array([r.x1, r.y1, r.x2, r.y2], float)
        g = group_of(r.name)
        if any(g == kg and _iou(b, kb) >= iou for kg, kb in kept):
            continue
        kept.append((g, b))
        keep.append(idx)
    return rows.loc[keep]


def busiest(tt: dict[str, Any]) -> dict[str, Any] | None:
    """The detection frame with the most moving objects on screen at once (earliest on ties).

    Counted per frame after dedupe_frame, split by each detection's class in that frame."""
    grp = {t["tid"]: t["group"] for t in tt["moving"]}
    df = tt["df"][tt["df"].tid.isin(grp)]
    if df.empty:
        return None
    raw = df.groupby("f").tid.nunique().reset_index(name="n").sort_values(["n", "f"], ascending=[False, True])
    best = None  # (count, f, rows)
    for f, n in zip(raw.f.to_numpy(), raw.n.to_numpy()):
        if best is not None and n < best[0]:
            break  # deduped count <= raw count, so no later frame can beat the best
        rows = dedupe_frame(df[df.f == f])
        if best is None or len(rows) > best[0] or (len(rows) == best[0] and f < best[1]):
            best = (len(rows), int(f), rows)
    count, f, rows = best
    n_p = int(sum(group_of(nm) == "person" for nm in rows.name))
    return {"t": round(float(rows.t.iloc[0]), 2), "count": int(count), "person": n_p,
            "vehicle": int(count) - n_p, "f": f, "rows": rows,
            "raw_ids": int(df[df.f == f].tid.nunique())}


# ----------------------------------------------------------------------------- drawing
def _text(img, s, org, scale, colour=(255, 255, 255), th=1, bg=None, pad=6, border=None):
    """putText with an optional filled box behind; returns the box (x1, y1, x2, y2)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, tht), base = cv2.getTextSize(s, font, scale, th)
    x, y = int(org[0]), int(org[1])
    box = (x - pad, y - tht - pad, x + tw + pad, y + base + pad)
    if bg is not None:
        ov = img.copy()
        cv2.rectangle(ov, box[:2], box[2:], bg, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    if border is not None:
        cv2.rectangle(img, box[:2], box[2:], border, 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, colour, th, cv2.LINE_AA)
    return box


def _overlap(a, b, m=4):
    return not (a[2] + m < b[0] or b[2] + m < a[0] or a[3] + m < b[1] or b[3] + m < a[1])


def render(cam: str, tt: dict[str, Any], lanes: list[dict]) -> np.ndarray:
    W, H = tt["W"], tt["H"]
    vw = tt.get("views")
    bg = vw["plate"] if vw is not None and vw.get("plate") is not None else data.background(cam)
    if bg.shape[1] != W or bg.shape[0] != H:
        bg = cv2.resize(bg, (W, H))
    s = max(1.0, OUT_W / W)
    OW, OH = int(round(W * s)), int(round(H * s))
    base = cv2.resize(bg, (OW, OH), interpolation=cv2.INTER_CUBIC) if s != 1 else bg.copy()
    # calm the plate so trails read: desaturate, then scale brightness toward a mean of ~70
    grey = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    base = cv2.addWeighted(base, 0.45, grey, 0.55, 0)
    gain = float(np.clip(70.0 / max(1.0, base.mean()), 0.4, 1.8))
    img = np.clip(base.astype(np.float32) * gain, 0, 255)

    thick = max(2, int(round(0.005 * OW)))
    sigma = 0.006 * OW
    for grp in ("vehicle", "person"):  # people drawn last so they sit on top
        tr = [t for t in tt["moving"] if t["group"] == grp]
        if not tr:
            continue
        heat = np.zeros((OH, OW), np.float32)
        mask = np.zeros((OH, OW), np.uint8)
        for t in tr:
            p = np.round(t["pts"][:, 1:] * s).astype(np.int32)
            if len(p) < 2:
                continue
            mask[:] = 0
            cv2.polylines(mask, [p], False, 255, thick, cv2.LINE_AA)
            heat += mask.astype(np.float32) / 255.0
        heat = cv2.GaussianBlur(heat, (0, 0), sigma)
        nz = heat[heat > 1e-3]
        if nz.size == 0:
            continue
        v = np.clip(heat / max(1e-6, float(np.percentile(nz, 99))), 0, 1) ** (0.6 if len(tr) < 40 else 0.85)
        alpha = (v * 0.88)[..., None]
        col = np.array(COLOURS[grp], np.float32)
        hot = col + (255 - col) * 0.55  # cores glow paler
        colour = col[None, None, :] * (1 - v[..., None]) + hot[None, None, :] * v[..., None]
        img = img * (1 - alpha) + colour * alpha
    img = img.astype(np.uint8)

    # lane arrows: dark outline, coloured body, arrowhead, badge with id and count
    fs = 0.62 * s if s > 1 else 0.62
    fs = max(0.55, min(0.9, fs))
    max_c = max([l["count"] for l in lanes], default=1)
    boxes: list[tuple] = []
    legend_box = (0, 0, int(470 * OW / 1280), int(118 * OW / 1280))
    boxes.append(legend_box)
    notes = [f"Trails = moving track ids; {tt['parked']} parked and {tt['stationary']} stationary excluded. "
             f"Arrows = lanes (count = tracks)."]
    if vw is not None and vw["excluded"]:
        hidden = tt["duration"] - vw["main_s"]
        notes.insert(0, f"Main camera view only: {vw['main_s']:.0f} of {tt['duration']:.0f} s used; "
                        f"{hidden:.0f} s where the view differs are left out.")
    fsn = max(0.42, min(0.6, 0.48 * OW / 1280))
    sizes = [cv2.getTextSize(n_, cv2.FONT_HERSHEY_DUPLEX, fsn, 1)[0] for n_ in notes]
    notes_h = sum(h_ + 16 for _, h_ in sizes)
    boxes.append((OW - max(w_ for w_, _ in sizes) - 30, OH - 14 - notes_h, OW, OH))  # footnotes, bottom right
    drawn = []
    for l in reversed(lanes):  # biggest lane drawn last, on top
        p = np.array(l["path"], float) * s
        if np.hypot(*(p[-1] - p[0])) < 4:
            continue
        drawn.append((l, p))
        w = max(3, int(round((3 + 5 * math.sqrt(l["count"] / max_c)) * OW / 1280)))
        col = COLOURS[l["group"]]
        # arrowhead direction from the last ~3 points
        tip, back = p[-1], p[max(0, len(p) - 4)]
        d = tip - back
        dn = d / max(1e-6, np.hypot(*d))
        L = max(16, 4.5 * w)
        nrm = np.array([-dn[1], dn[0]])
        body = p[:-1][np.hypot(*(p[:-1] - tip).T) > L * 0.7]
        pi = np.round(np.vstack([body, tip - dn * L * 0.7])).astype(np.int32)
        cv2.polylines(img, [pi], False, (20, 20, 20), w + 4, cv2.LINE_AA)
        cv2.polylines(img, [pi], False, col, w, cv2.LINE_AA)
        tri = np.array([tip, tip - dn * L + nrm * L * 0.55, tip - dn * L - nrm * L * 0.55])
        cv2.fillConvexPoly(img, np.round(tri).astype(np.int32), (20, 20, 20), cv2.LINE_AA)
        tri2 = np.array([tip - dn * 3, tip - dn * (L - 3) + nrm * (L - 4) * 0.5, tip - dn * (L - 3) - nrm * (L - 4) * 0.5])
        cv2.fillConvexPoly(img, np.round(tri2).astype(np.int32), col, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in p[0]), w + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in p[0]), w, col, -1, cv2.LINE_AA)
        boxes.append((tip[0] - L, tip[1] - L, tip[0] + L, tip[1] + L))  # keep badges off arrowheads
    # badges after all arrows so no arrow covers a badge; biggest lane gets the first pick of spots
    for l, p in reversed(drawn):
        col = COLOURS[l["group"]]
        label = f"{l['lane_id']}  {l['count']}"
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
        placed = None
        for frac in (0.5, 0.35, 0.65, 0.2, 0.8, 0.1, 0.9):
            q = p[int(frac * (len(p) - 1))]
            for off in ((10, -10), (10, tht + 20), (-tw - 22, -10), (-tw - 22, tht + 20)):
                x, y = q[0] + off[0], q[1] + off[1]
                x = float(np.clip(x, 8, OW - tw - 14))
                y = float(np.clip(y, tht + 10, OH - 12))
                b = (x - 6, y - tht - 6, x + tw + 6, y + 8)
                if not any(_overlap(b, o) for o in boxes):
                    placed = (x, y)
                    break
            if placed:
                break
        if placed is None:  # crowded: step upward from the path midpoint until the badge is clear
            q = p[len(p) // 2]
            x = float(np.clip(q[0] + 10, 8, OW - tw - 14))
            for k in range(12):
                y = float(np.clip(q[1] - 10 - k * (tht + 14), tht + 10, OH - 12))
                if not any(_overlap((x - 6, y - tht - 6, x + tw + 6, y + 8), o) for o in boxes):
                    break
            placed = (x, y)
        bx = _text(img, label, placed, fs, (255, 255, 255), 1, bg=(24, 24, 24), border=col)
        boxes.append(bx)

    # legend + honest footnote
    np_ = sum(t["group"] == "person" for t in tt["moving"])
    nv = len(tt["moving"]) - np_
    ls = max(0.55, min(0.85, 0.72 * OW / 1280))
    _text(img, data.CAMS[cam]["title"] + ": where things move", (18, int(34 * ls / 0.72)), ls * 1.05,
          (255, 255, 255), 1, bg=(18, 18, 18), pad=8)
    y = int(70 * ls / 0.72)
    for grp, n in (("person", np_), ("vehicle", nv)):
        word = "people" if grp == "person" else "vehicles (incl. bikes)"
        cv2.rectangle(img, (18, y - 14), (40, y + 2), COLOURS[grp], -1, cv2.LINE_AA)
        _text(img, f"{n} moving {word}", (50, y), ls * 0.85, (240, 240, 240), 1, bg=(18, 18, 18), pad=5)
        y += int(30 * ls / 0.72)
    yb = OH - 14
    for note in reversed(notes):
        (tw, tht), _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_DUPLEX, fsn, 1)
        _text(img, note, (max(8, OW - tw - 16), yb), fsn, (220, 220, 220), 1, bg=(18, 18, 18), pad=5)
        yb -= tht + 16
    return img


def render_busiest(cam: str, tt: dict[str, Any], b: dict[str, Any]) -> np.ndarray:
    img = data.frame_at(cam, b["t"])
    W = img.shape[1]
    s = max(1.0, OUT_W / W) if W < OUT_W else 1.0
    th = max(2, int(round(W / 400)))
    for r in b["rows"].itertuples():
        c = COLOURS[group_of(r.name)]
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (20, 20, 20), th + 2, cv2.LINE_AA)
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, th, cv2.LINE_AA)
    if s != 1.0:
        img = cv2.resize(img, (int(W * s), int(img.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    mm, ss = divmod(int(b["t"]), 60)
    _text(img, f"Busiest moment {mm:02d}:{ss:02d}  -  {b['count']} moving on screen "
               f"({b['person']} people, {b['vehicle']} vehicles)", (16, 34), 0.75, (255, 255, 255), 1,
          bg=(18, 18, 18), pad=8)
    return img


# ----------------------------------------------------------------------------- main
def compute(cam: str) -> dict[str, Any]:
    """Everything except image writing. Returns {'json': flow.json dict, 'tt': table, 'lanes': lanes, 'busiest': ...}."""
    views = camera_views(cam)
    tt = track_table(cam, views)
    lanes, other = build_lanes(tt)
    pm = per_minute(tt)
    b = busiest(tt)
    js = {"camera": cam, "image": "flow.jpg", "w": int(tt["W"]), "h": int(tt["H"]),
          "lanes": [{k: v for k, v in l.items() if not k.startswith("_")} for l in lanes],
          "per_minute": pm,
          "busiest": None if b is None else {"t": b["t"], "count": b["count"], "frame": "frames/busiest.jpg",
                                             "person": b["person"], "vehicle": b["vehicle"],
                                             "track_ids": b["raw_ids"]},
          "totals": {"moving": len(tt["moving"]),
                     "moving_person": sum(t["group"] == "person" for t in tt["moving"]),
                     "moving_vehicle": sum(t["group"] == "vehicle" for t in tt["moving"]),
                     "parked_excluded": tt["parked"], "stationary_excluded": tt["stationary"],
                     "too_short_excluded": tt["short"], "in_lanes": int(sum(l["count"] for l in lanes)),
                     "not_in_main_lanes": other, "dropped_detections_other_view": int(tt["dropped_dets"])},
          "view": {"main_view_s": views["main_s"], "duration_s": round(float(tt["duration"]), 2),
                   "excluded": views["excluded"],
                   "plate": "built from main-view frames" if views["plate"] is not None else "background.jpg"},
          "notes": ["Counts are tracker ids, not verified individuals: one object can split into several ids "
                    "and two can merge.",
                    "Coordinates are source-video pixels (w x h); flow.jpg may be upscaled, see image_w/image_h.",
                    "Group 'vehicle' includes bicycles and motorcycles.",
                    "Busiest count = boxes on screen in one frame after merging same-group boxes with IoU >= "
                    f"{DUP_IOU} (one object held by two tracker ids); person/vehicle split by that frame's class; "
                    "track_ids = tracker ids before merging."] + list(data.CAMS[cam].get("simulated", []))}
    if views["excluded"]:
        js["notes"].append(f"The camera view changes in this clip: only {views['main_s']:.1f} of {tt['duration']:.1f} s "
                           f"match the main view; tracks and counts cover those seconds only, the rest is UNKNOWN.")
    return {"json": js, "tt": tt, "lanes": lanes, "busiest": b}


def run(cam: str) -> dict[str, Any]:
    t0 = time.time()
    r = compute(cam)
    js, tt = r["json"], r["tt"]
    od = data.out_dir(cam)
    img = render(cam, tt, r["lanes"])
    cv2.imwrite(str(od / "flow.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    js["image_w"], js["image_h"] = int(img.shape[1]), int(img.shape[0])
    if r["busiest"] is not None:
        (od / "frames").mkdir(exist_ok=True)
        cv2.imwrite(str(od / "frames" / "busiest.jpg"), render_busiest(cam, tt, r["busiest"]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    data.write_json(od / "flow.json", js)
    write_summary(cam, js, time.time() - t0)
    return js


def write_summary(cam: str, js: dict[str, Any], secs: float) -> None:
    T = js["totals"]
    L = [f"flow summary: {cam}",
         f"moving tracks: {T['moving']} ({T['moving_person']} people, {T['moving_vehicle']} vehicles incl. bikes)",
         f"excluded: {T['parked_excluded']} parked, {T['stationary_excluded']} stationary, "
         f"{T['too_short_excluded']} shorter than {MIN_FRAMES} detections",
         f"lanes: {len(js['lanes'])} covering {T['in_lanes']} tracks; outside main lanes: "
         f"{T['not_in_main_lanes']['person']} people, {T['not_in_main_lanes']['vehicle']} vehicles"]
    for l in js["lanes"]:
        L.append(f"  {l['lane_id']} {l['group']:7s} {l['count']:4d}  {l['name']}  "
                 f"median {l['median_speed_px_s']:.1f} px/s  e.g. ids {l['example_ids'][:3]}")
    V = js["view"]
    L.append(f"main camera view: {V['main_view_s']:.1f} of {V['duration_s']:.1f} s; plate: {V['plate']}; "
             f"{T['dropped_detections_other_view']} detections from other views dropped")
    for e in V["excluded"]:
        L.append(f"  excluded {e['start']:.1f}-{e['end']:.1f} s: {e['reason']} (min score {e['min_score']})")
    L.append("per minute (unique moving track ids, main view only):")
    for m in js["per_minute"]:
        vis = f", visible {m['visible_s']:.0f} s" if "visible_s" in m else ""
        if m["person"] is None:
            L.append(f"  {m['t'] // 60:02d}:00  UNKNOWN ({m.get('unknown')}){vis}")
        else:
            L.append(f"  {m['t'] // 60:02d}:00  people {m['person']:4d}  vehicles {m['vehicle']:4d}  "
                     f"({m['seconds']:.0f} s{vis})")
    b = js["busiest"]
    L.append("busiest moment: " + ("UNKNOWN (no moving tracks)" if b is None else
                                   f"t={b['t']:.2f}s, {b['count']} moving on screen "
                                   f"({b['person']} people, {b['vehicle']} vehicles) -> {b['frame']}"))
    L += [f"note: {n}" for n in js["notes"]]
    L.append(f"computed in {secs:.1f} s")
    (data.out_dir(cam) / "flow_summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str]) -> None:
    for cam in argv or list(data.CAMS):
        t0 = time.time()
        js = run(cam)
        T = js["totals"]
        b = js["busiest"]
        print(f"{cam}: {T['moving']} moving ({T['moving_person']} p / {T['moving_vehicle']} v), "
              f"{len(js['lanes'])} lanes, busiest {b['count'] if b else 'UNKNOWN'} at "
              f"{b['t'] if b else '-'}s, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
