"""Health ledger: per-second visibility, parking spots and what changed in each spot.

1 frame per second at 320 px wide is enough to tell when the camera could not see:
  no_signal    no frame, or a cut in the recording (an isolated jump between two steady stretches)
  dark         nearly black (mean luma < 15), or dim (< 35) AND flat (grey std < 10). A dim street with
               lamps and visible structure (dusk, foggy dawn) is NOT dark.
  frozen       near-zero change vs the previous sample for >= 3 s, and no detector box moved to a new place
  moved        the camera itself is moving (PTZ swing, zoom, shake): the view jumps between samples
  turned_away  the camera holds still on a scene that is not its home view (it matches another steady
               view, or it got there by moving): "camera pointed elsewhere"
  blocked      a large area differs from the home view with no camera motion and no detection to explain it
  blur         contrast-normalised sharpness (Laplacian var / grey var) far below this view's own median.
               Fog and low light scale contrast but keep edges, so they do NOT count as blur; real smear does.
A switch between colour and night infrared (the picture loses all colour in one step) is a lighting change, not
a cut and not a new view: the infrared stretch keeps the view it switched from, with its own infrared plate, and
blur is judged per view, day and lighting. Night infrared jumps often (headlights, IR gain), so a jump is a cut
only when it stands out from the change around it.
Spots are clusters of parked-vehicle boxes (IoU >= 0.5) inside one home view. A spot is never
"gone" while the camera is blind or looking elsewhere: those seconds are "unknown".
ledger.json also carries one blind total (`blind_total_s`) and per-day totals (`blind_by_day`) so every
view prints the same number.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm import data

SAMPLE_W = 320
DARK_LUMA = 35.0        # dim: below this mean luma the frame is dark only if it is also flat
DARK_HARD = 15.0        # nearly black: dark whatever the structure
DARK_STD = 10.0         # grey std below this = no visible structure (a lit street at dusk is ~20-25)
FROZEN_DIFF = 0.05      # mean abs grey diff (0..255) = "no change at all". A real freeze repeats the frame (0.0);
                        # a quiet street under fog can sit near 0.1 (real cams: >= 0.35), so keep this low
FROZEN_RUN = 3          # seconds
FROZEN_BLIP = 0.6       # a step this small for <= BLIP_RUN samples inside a freeze = the encoder refreshing the frame
BLIP_RUN = 2
BLUR_FRAC = 0.35        # sharpness below this fraction of the same view's (and day's) median = blur
MIN_AWAY_S = 3          # a "turned_away" run shorter than this, next to camera motion, is still the move
CELLS = (8, 6)          # grid for the per-cell "is the home scene still there" check
CELL_TEX = 12.0         # a plate cell needs this much grey std to be judged
CELL_NCC = 0.7          # a cell correlating this well with the home plate still shows the home scene
HOME_CELLS = 0.6        # >= this share of matching cells = still the home view (fog / dusk drift, not view lost)
STILL_HOME = 0.1        # >= this share still matching = the camera has not moved: something blocks part of it
REASONS = ("no_signal", "dark", "frozen", "moved", "turned_away", "blocked", "blur")
REASON_WORDS = {"no_signal": "no picture", "dark": "too dark", "frozen": "picture froze",
                "moved": "camera moving", "turned_away": "camera pointed elsewhere",
                "blocked": "view blocked", "blur": "picture blurred"}
JUMP_DIFF = 12.0        # mean abs grey diff between 1 s samples = the view jumped
CUT_ALONE = 2.0         # a cut's jump is >= this many times the biggest change within 3 s of it. A cut stands
                        # out (Pack D day cut: 28 vs <= 3); night IR headlight sweeps jump in a row (13 vs 10)
DET_MOVE_PX = 3.0       # a detector box this far (full-res px, any side) from every box of the still stretch =
                        # something moved: the picture is live. A frozen frame re-encoded still jitters boxes by
                        # <= 0.6 px and flickers weak ones on and off in place (Pack D), so those do not count
DET_LAG_S = 8           # box motion only within this many seconds of the start of a pixel-still run...
DET_SETTLE_S = 20       # ...followed by at least this long with no box motion = detector lag, not a live picture
MONO_SAT = 2.0          # mean HSV saturation below this = a black-and-white night infrared picture (Pack D IR: 0.0;
                        # the greyest colour cam, car_detection, never drops below 4.4)
FLIP_SAT = 10.0         # ...and a switch needs real colour on the other side of the jump (Pack D dusk: ~72)
FLIP_GAP = 5            # steady stretches this close on either side of a colour<->IR switch are the same view
CHANGE_PX = 30          # grey level change that counts a pixel as "different from the plate"
BLOCK_FRAC = 0.35       # unexplained changed area that counts as blocked / view lost
NCC_MIN = 0.6           # normalised correlation with the home plate below this = the camera looks elsewhere
NCC_TEX = 10.0          # ...only used when the plate has texture (grey std); bare tarmac correlates as noise
MIN_VIEW_S = 10         # a stable run this long becomes a candidate view (shorter clips: 15 % of the clip)
OCCLUDED = 0.25         # another vehicle's box covering this share of an empty-looking spot = can't tell
MIN_GONE_S = 3          # a "gone" this short right next to a blind stretch is too little evidence: unknown
PATCH_SAME = 14.0       # mean abs grey diff of a spot patch vs its last confirmed-occupied look = still there
SPOT_IOU = 0.5
SAME_SIM = 0.55         # 0.5 * colour-histogram correlation + 0.5 * small-image correlation = same vehicle
BRIDGE_S = 60           # a detector gap up to this long, with the same vehicle on both sides, is not a departure
DEFAULT_CAMS = ["ptz_parking", "car_detection", "rochester_night"]


# ---------------------------------------------------------------- sampling
@dataclass
class Samples:
    cam: str
    seconds: int
    w: int
    h: int
    grey: list[np.ndarray | None] = field(default_factory=list)   # one 320-wide grey frame per second, None = unreadable
    sat: list[float | None] = field(default_factory=list)         # mean HSV saturation per second (empty = not measured)


def _mono(smp: Samples) -> tuple[np.ndarray, np.ndarray]:
    """(mono, sat) per second: mono = a black-and-white (night infrared) picture. All False without saturation."""
    n = smp.seconds
    sat = np.full(n, np.nan)
    if len(smp.sat) == n:
        sat = np.array([np.nan if v is None else float(v) for v in smp.sat])
    return np.nan_to_num(sat, nan=255.0) < MONO_SAT, sat


def sample(cam: str, max_s: int | None = None, crops: "CropStore | None" = None) -> Samples:
    """Read one frame per whole second (sequential grab, so it is fast). If a CropStore is given, its
    requested crops are cut in the same pass, so the video is decoded only once."""
    inf = data.info(cam)
    fps, n = inf["fps"], int(inf["frames"])
    secs = int(np.floor(inf["duration"]))
    if max_s is not None:
        secs = min(secs, max_s)
    want = {int(round(s * fps)): s for s in range(secs + 1) if int(round(s * fps)) < n}
    sh = int(round(SAMPLE_W * inf["h"] / inf["w"]))
    grey: list[np.ndarray | None] = [None] * (secs + 1)
    sat: list[float | None] = [None] * (secs + 1)
    cap = cv2.VideoCapture(str(data.CAMS[cam]["video"]))
    by_f = crops.frames() if crops is not None else {}
    last = max([*want, *by_f]) if (want or by_f) else -1
    i = 0
    while i <= last:
        if not cap.grab():
            break
        if i in want or i in by_f:
            ok, img = cap.retrieve()
            if ok and img is not None:
                if i in want:
                    small = cv2.resize(img, (SAMPLE_W, sh), interpolation=cv2.INTER_AREA)
                    grey[want[i]] = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    sat[want[i]] = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1].mean())
                if i in by_f:
                    crops.cut(img, by_f[i])
        i += 1
    cap.release()
    return Samples(cam, secs + 1, inf["w"], inf["h"], grey, sat)


# ---------------------------------------------------------------- per-second health
def _det_boxes_by_second(cam: str, n: int) -> dict[int, np.ndarray]:
    tr = data.tracks(cam)
    tr = tr[tr.name.isin(data.MOVERS)]
    sec = tr.t.round().astype(int)
    out: dict[int, np.ndarray] = {}
    for s, g in tr.groupby(sec):
        if 0 <= s < n:
            out[int(s)] = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
    return out


def _mask(shape: tuple[int, int], boxes: np.ndarray | None, scale: float) -> np.ndarray:
    """Union of boxes (full-res coords) rasterised on the small frame, vectorised with a difference array."""
    H, W = shape
    if boxes is None or not len(boxes):
        return np.zeros(shape, bool)
    b = boxes * scale
    x1 = np.clip(np.floor(b[:, 0]), 0, W).astype(int)
    y1 = np.clip(np.floor(b[:, 1]), 0, H).astype(int)
    x2 = np.clip(np.ceil(b[:, 2]), 0, W).astype(int)
    y2 = np.clip(np.ceil(b[:, 3]), 0, H).astype(int)
    d = np.zeros((H + 1, W + 1), np.int32)
    np.add.at(d, (y1, x1), 1)
    np.add.at(d, (y1, x2), -1)
    np.add.at(d, (y2, x1), -1)
    np.add.at(d, (y2, x2), 1)
    return d.cumsum(0).cumsum(1)[:H, :W] > 0


def _absdiff(a: np.ndarray, b: np.ndarray, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """|a - b| after removing a global brightness offset (auto-exposure), measured on the `keep` pixels."""
    if keep is None or keep.mean() < 0.2:
        keep = np.ones(a.shape, bool)
    a16, b16 = a.astype(np.int16), b.astype(np.int16)
    off = int(np.median(a16[keep])) - int(np.median(b16[keep]))
    return np.abs(a16 - b16 - off), keep


def _changed_frac(g: np.ndarray, plate: np.ndarray, mask: np.ndarray | None) -> float:
    d, keep = _absdiff(g, plate, None if mask is None else ~mask)
    return float(((d > CHANGE_PX) & keep).mean())


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Zero-mean correlation of two grey images at 80 px wide, over the pixels NOT under `mask` only
    (so detected objects and auto-exposure do not count). 1.0 when too little is left to judge."""
    size = (80, max(1, int(80 * a.shape[0] / a.shape[1])))
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(b, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    keep = np.ones(a.shape, bool)
    if mask is not None:
        keep = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_AREA) == 0
        if keep.mean() < 0.3:
            return 1.0
    a, b = a[keep], b[keep]
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _cell_match(x: np.ndarray, plate: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Share of the plate's textured grid cells (8x6) that still correlate with the frame (NCC >= CELL_NCC).
    Gain/offset-free per cell, so fog, dusk and auto-exposure keep it high (~0.85 on ptz under heavy fog);
    a moved or zoomed camera drops it to ~0 (ptz 156-250 s: <= 0.04); a block over part of the lens leaves
    the uncovered cells matching (0.13-0.6 in SIMULATED blocks). 1.0 when no cell is textured enough to judge."""
    H, W = x.shape
    keep = np.ones(x.shape, bool) if mask is None else ~mask
    tot = ok = 0
    for i in range(CELLS[1]):
        for j in range(CELLS[0]):
            ys, xs = slice(i * H // CELLS[1], (i + 1) * H // CELLS[1]), slice(j * W // CELLS[0], (j + 1) * W // CELLS[0])
            k = keep[ys, xs]
            if k.mean() < 0.5:
                continue
            b = plate[ys, xs][k].astype(np.float32)
            if b.std() < CELL_TEX:
                continue
            a = x[ys, xs][k].astype(np.float32)
            a, b = a - a.mean(), b - b.mean()
            den = float(np.sqrt((a * a).sum() * (b * b).sum()))
            tot += 1
            ok += bool(den > 0 and float((a * b).sum()) / den >= CELL_NCC)
    return ok / tot if tot else 1.0


def health(cam: str, smp: Samples, det_boxes: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    """Classify every sampled second. Pure on the samples, so tests can tamper with frames in memory."""
    n = smp.seconds
    det_boxes = _det_boxes_by_second(cam, n) if det_boxes is None else det_boxes
    scale = SAMPLE_W / smp.w
    g = smp.grey
    shape = next((x.shape for x in g if x is not None), (1, 1))
    masks = [_mask(shape, det_boxes.get(s), scale) for s in range(n)]
    luma = np.array([x.mean() if x is not None else np.nan for x in g])
    gstd = np.array([x.std() if x is not None else np.nan for x in g])
    lap = np.full(n, np.nan)      # Laplacian variance of the scene (pixels under detections left out)
    svar = np.full(n, np.nan)     # grey variance of the same pixels
    for s, x in enumerate(g):
        if x is None:
            continue
        L = cv2.Laplacian(x, cv2.CV_64F)
        keep = ~masks[s]
        if keep.mean() >= 0.3:
            lap[s], svar[s] = float(L[keep].var()), float(x[keep].var())
        else:
            lap[s], svar[s] = float(L.var()), float(x.var())
    # contrast-normalised sharpness: fog (I*t + A*(1-t)) and exposure gain scale lap var and grey var alike
    sharp = lap / np.maximum(np.nan_to_num(svar, nan=0.0), 1.0)
    diff = np.full(n, np.nan)      # change vs previous sample, ignoring pixels under detections
    diff_all = np.full(n, np.nan)  # the same over every pixel: a car driving by proves the feed is live
    for s in range(1, n):
        if g[s] is not None and g[s - 1] is not None:
            keep = ~(masks[s] | masks[s - 1])
            d, keep = _absdiff(g[s], g[s - 1], keep)
            diff[s] = float(d[keep].mean())
            diff_all[s] = float(np.abs(g[s].astype(np.int16) - g[s - 1].astype(np.int16)).mean())
    dark = (luma < DARK_HARD) | ((luma < DARK_LUMA) & (gstd < DARK_STD))
    nosig = np.array([x is None for x in g])

    # a detector box at a place no box held during this still stretch = something moved: the feed is live (a quiet
    # street whose pixels barely change). A frozen frame gives the same boxes: sub-pixel jitter from re-encoding and
    # weak boxes flickering on and off in place are not motion (Pack D: 456 s of real freezes, zero pixel change).
    def _uniq(b: np.ndarray | None) -> np.ndarray:
        return np.zeros((0, 4)) if b is None or not len(b) else np.unique(np.round(b, 1), axis=0)
    px_still = np.nan_to_num(diff_all, nan=99.0) < FROZEN_DIFF
    # a frozen picture steps a little when the encoder refreshes it (a keyframe, Pack D: 0.09-0.49 grey levels, then
    # identical again): <= BLIP_RUN such samples between still samples are still the same frozen picture
    small = np.nan_to_num(diff_all, nan=99.0) < FROZEN_BLIP
    s = 1
    while s < n:
        if px_still[s] or not small[s]:
            s += 1
            continue
        e = s
        while e + 1 < n and small[e + 1] and not px_still[e + 1]:
            e += 1
        if e - s + 1 <= BLIP_RUN and px_still[s - 1] and e + 1 < n and px_still[e + 1]:
            px_still[s:e + 1] = True
        s = e + 1
    det_live = np.zeros(n, bool)
    seen = _uniq(det_boxes.get(0))     # every box seen since the picture last changed
    for s in range(1, n):
        cur = _uniq(det_boxes.get(s))
        if not px_still[s]:
            seen = cur
            continue
        if len(cur) and len(seen):
            gap = np.abs(cur[:, None, :] - seen[None, :, :]).max(axis=2).min(axis=1)
            det_live[s] = bool(gap.max() > DET_MOVE_PX)
        seen = cur if det_live[s] else (np.unique(np.vstack([seen, cur]), axis=0) if len(cur) else seen)
    # the detector can lag the picture after a cut: Pack D boxes keep moving for the first 4-7 s of a 40 s freeze and
    # then sit still. Box motion only in the first DET_LAG_S of a long still run, then none for DET_SETTLE_S, is lag.
    # (one changed sample between two still runs does not end the run: a freeze includes the sample before it)
    s = 1
    while s < n:
        if not px_still[s]:
            s += 1
            continue
        e = s
        while e + 1 < n and (px_still[e + 1] or (e + 2 < n and px_still[e + 2])):
            e += 1
        lv = det_live[s:e + 1]
        # motion = two live samples in a row; one alone mid-freeze is a weak box popping up in the re-encoding noise
        live = np.flatnonzero(lv & (np.r_[False, lv[:-1]] | np.r_[lv[1:], False]))
        if len(live) and live[-1] < DET_LAG_S and (e - s) - live[-1] >= DET_SETTLE_S:
            det_live[s:e + 1] = False
        s = e + 1

    # frozen: a run of >= FROZEN_RUN consecutive near-identical samples (the first of the run included)
    still = px_still & ~det_live
    frozen = np.zeros(n, bool)
    s = 1
    while s < n:
        if still[s]:
            e = s
            while e + 1 < n and still[e + 1]:
                e += 1
            if e - s + 2 >= FROZEN_RUN:
                frozen[s - 1:e + 1] = True
            s = e + 1
        else:
            s += 1

    # colour <-> night infrared switch between two samples (the IR cut filter): a lighting change, not a cut or a move
    mono, sat = _mono(smp)
    flip = np.zeros(n, bool)
    for s in range(1, n):
        flip[s] = bool(mono[s] != mono[s - 1] and np.nanmax([sat[s], sat[s - 1], 0.0]) >= FLIP_SAT)

    # stable segments -> views (merged when their plates match) -> one home view per day (and per lighting)
    bad = dark | nosig | frozen
    segs: list[tuple[int, int]] = []
    a = None
    for s in range(n):
        brk = bad[s] or flip[s] or (s > 0 and not np.isnan(diff[s]) and diff[s] > JUMP_DIFF)
        if brk or a is None:
            if a is not None:
                segs.append((a, s))
            a = None if bad[s] else s
    if a is not None:
        segs.append((a, n))
    min_view = max(3, min(MIN_VIEW_S, int(round(0.15 * n))))
    segs = [(a, b) for a, b in segs if b - a >= min_view]
    stable = np.zeros(n, bool)
    for a, b in segs:
        stable[a:b] = True
    jump = (np.nan_to_num(diff, nan=0.0) > JUMP_DIFF) & ~flip

    def _tex(p: np.ndarray) -> float:
        return float(cv2.resize(p, (80, 45), interpolation=cv2.INTER_AREA).std())

    # a view keeps one plate per lighting: a stretch that matches none of them but follows a colour<->IR switch with
    # the camera holding still is the same view under new light (IR vs colour plates do not correlate: NCC < 0)
    views: list[dict[str, Any]] = []
    last_seg: tuple[int, int] | None = None   # (end, view) of the last steady stretch
    for a, b in segs:
        idx = np.linspace(a, b - 1, min(b - a, 40)).round().astype(int)
        plate = np.median(np.stack([g[i] for i in idx]), axis=0).astype(np.uint8)
        vi = next((i for i, v in enumerate(views)
                   if any(_changed_frac(plate, p, None) < BLOCK_FRAC and _ncc(plate, p) >= NCC_MIN for p in v["plates"])),
                  None)
        if vi is None and last_seg is not None and a - last_seg[0] <= FLIP_GAP and flip[max(0, last_seg[0] - 1):a + 1].any():
            vi = last_seg[1]
            views[vi]["plates"].append(plate)
        if vi is None:
            views.append({"plate": plate, "plates": [plate], "segs": []})
            vi = len(views) - 1
        views[vi]["segs"].append((a, b))
        last_seg = (b, vi)
    for v in views:
        v["tex"] = _tex(v["plate"])
    home: list[int] = []
    for d in data.days(cam):
        best, best_n = None, 0
        for vi, v in enumerate(views):
            k = sum(max(0, min(b, d["end"]) - max(a, d["start"])) for a, b in v["segs"])
            if k > best_n:
                best, best_n = vi, k
        if best is not None and best not in home:
            home.append(best)
        # a day that switches to infrared: the colour and the infrared stretch each keep their own home view
        lo, hi = int(np.ceil(d["start"])), min(n, int(np.ceil(d["end"])))
        if mono[lo:hi].any():
            seg_view = np.full(n, -1)
            for vi, v in enumerate(views):
                for a, b in v["segs"]:
                    seg_view[a:b] = vi
            for md in (False, True):
                sel = seg_view[lo:hi][(mono[lo:hi] == md) & (seg_view[lo:hi] >= 0)]
                if len(sel) >= min_view:
                    bv = int(np.bincount(sel).argmax())
                    if bv not in home:
                        home.append(bv)
    if not home:   # no stable stretch at all: fall back to the median background plate of the whole clip
        bg = data.background(cam)
        bg = cv2.cvtColor(cv2.resize(bg, (SAMPLE_W, g[0].shape[0] if g[0] is not None else bg.shape[0])), cv2.COLOR_BGR2GRAY)
        views.append({"plate": bg, "plates": [bg], "segs": [], "tex": float(cv2.resize(bg, (80, 45)).std())})
        home = [len(views) - 1]
    home_pl = [(v, p, _tex(p)) for v in home for p in views[v]["plates"]]     # every lighting plate of every home view

    view_of = np.full(n, -1)
    block = np.ones(n)
    home_cells = np.ones(n)     # share of home-plate cells still matching (only computed for not-home seconds)
    for s in range(n):
        if g[s] is None:
            continue
        sc = []
        for v, pl, tex in home_pl:
            m = masks[s]
            fr = _changed_frac(g[s], pl, m)
            # detected objects do not count as "view lost"; bare low-texture plates make NCC meaningless
            nc = _ncc(g[s], pl, m if m.any() else None) if tex >= NCC_TEX else 1.0
            sc.append((max(fr / BLOCK_FRAC, (1 - nc) / (1 - NCC_MIN)), fr, nc))
        k = int(np.argmin([x[0] for x in sc]))
        block[s] = max(sc[k][1], 1 - sc[k][2])
        if sc[k][0] < 1.0:
            view_of[s] = home_pl[k][0]
            continue
        # pixel-level check failed: is the home scene still there under fog / low light / a partial block?
        cm = [_cell_match(g[s], pl, masks[s]) for _, pl, _ in home_pl]
        home_cells[s] = max(cm)
        if home_cells[s] >= HOME_CELLS:
            view_of[s] = home_pl[int(np.argmax(cm))][0]
    not_home = (view_of < 0) & ~nosig
    # an isolated jump between two stable stretches = a cut in the recording (time we never saw). Isolated = at least
    # CUT_ALONE x the biggest change within 3 s around it: at night headlights and IR gain make jumps in a row
    near = np.where(flip, np.nan, diff)
    alone = np.array([bool(diff[s] >= CUT_ALONE * np.nanmax(np.r_[near[max(0, s - 3):s], near[s + 1:s + 4], 0.0]))
                      for s in range(n)])
    # a frozen stretch is steady too: Pack D jumps into a frozen recording at a chunk start (1 s, then the freeze)
    steady = stable | frozen
    cut = np.array([bool(jump[s] and s > 0 and alone[s] and steady[s - 1]
                         and (steady[s] or (s + 1 < n and steady[s + 1]))) for s in range(n)])
    cut[:-1] |= cut[1:]     # the cut lies somewhere between samples s-1 and s: neither side is trustworthy
    # unsteady: not inside any stable stretch and jumping = the camera itself is moving (PTZ / handheld)
    moving = np.array([bool(not stable[s] and (jump[s] or (s + 1 < n and jump[s + 1]))) for s in range(n)])
    # a not-home sample that matches one of the camera's OTHER steady views = the camera looks elsewhere
    others = [v for vi, v in enumerate(views) if vi not in home]
    other = np.zeros(n, bool)
    for s in np.flatnonzero(not_home):
        if g[s] is None:
            continue
        m = masks[s]
        for v in others:
            for p in v["plates"]:
                nc = _ncc(g[s], p, m if m.any() else None) if _tex(p) >= NCC_TEX else 1.0
                if _changed_frac(g[s], p, m) < BLOCK_FRAC and nc >= NCC_MIN:
                    other[s] = True
                    break
            if other[s]:
                break
    # a not-home run that the camera reached by moving (or that matches another view) is "turned away";
    # one with no camera motion around it is something in front of the lens: "blocked"
    away_cand = not_home & ~dark & ~frozen & ~cut
    away_run = np.zeros(n, bool)
    s = 0
    while s < n:
        if not away_cand[s]:
            s += 1
            continue
        e = s
        while e + 1 < n and away_cand[e + 1]:
            e += 1
        lo, hi = max(0, s - 1), min(n, e + 2)
        if moving[lo:hi].any() or other[s:e + 1].any():
            away_run[s:e + 1] = True
        s = e + 1

    # blur: sharpness vs the median of the same home view on the same day in the same lighting (colour / infrared)
    # (fallback: view and day, then view, then camera)
    day_ix = np.zeros(n, int)
    for k, d in enumerate(data.days(cam)):
        day_ix[int(np.ceil(d["start"])):] = k
    usable = ~dark & ~nosig & ~frozen & ~cut & ~moving & (view_of >= 0)
    def refs(x: np.ndarray) -> np.ndarray:
        out = np.full(n, float(np.median(x[usable])) if usable.any() else 0.0)
        for v in set(int(k) for k in view_of[usable]):
            in_v = usable & (view_of == v)
            v_ref = float(np.median(x[in_v]))
            for k in set(int(d) for d in day_ix[in_v]):
                in_vd = in_v & (day_ix == k)
                vd_ref = float(np.median(x[in_vd])) if in_vd.sum() >= 10 else v_ref
                out[(view_of == v) & (day_ix == k)] = vd_ref
                for md in set(bool(b) for b in mono[in_vd]):
                    in_vdm = in_vd & (mono == md)
                    if in_vdm.sum() >= 10:
                        out[(view_of == v) & (day_ix == k) & (mono == md)] = float(np.median(x[in_vdm]))
        return out

    ref, lap_ref = refs(sharp), refs(lap)
    # both must drop: fog/dusk lower raw edge energy but keep the normalised sharpness; a big object
    # entering lowers the normalised value but not the edge energy. Real smear lowers both (~100x).
    blur = (sharp < BLUR_FRAC * ref) & (lap < BLUR_FRAC * lap_ref)

    reason: list[str | None] = []
    score: list[float] = []
    for s in range(n):
        if nosig[s]:
            r, sc = "no_signal", 1.0
        elif dark[s]:
            r, sc = "dark", 1.0 - luma[s] / DARK_LUMA
        elif frozen[s]:
            r, sc = "frozen", 1.0 - min(1.0, np.nan_to_num(diff_all[s], nan=0.0) / FROZEN_DIFF)
        elif cut[s]:
            r, sc = "no_signal", min(1.0, diff[s] / (4 * JUMP_DIFF))
        elif moving[s]:
            dd = np.nanmax([diff[s], diff[s + 1] if s + 1 < n else np.nan, 0.0])
            r, sc = "moved", min(1.0, dd / (4 * JUMP_DIFF))
        elif not_home[s]:
            # part of the home scene still sits where it was = the camera did not move: something blocks it
            r, sc = ("turned_away" if away_run[s] and home_cells[s] < STILL_HOME else "blocked"), block[s]
        elif blur[s]:
            r, sc = "blur", 1.0 - sharp[s] / (BLUR_FRAC * ref[s])
        else:
            r, sc = None, 0.0
        reason.append(r)
        score.append(float(np.clip(sc, 0.0, 1.0)))
    # a turned_away flicker of < MIN_AWAY_S between camera moves is still the move (the camera had not settled)
    s = 0
    while s < n:
        if reason[s] != "turned_away":
            s += 1
            continue
        e = s
        while e + 1 < n and reason[e + 1] == "turned_away":
            e += 1
        if e - s + 1 < MIN_AWAY_S and ((s > 0 and reason[s - 1] == "moved") or (e + 1 < n and reason[e + 1] == "moved")):
            for k in range(s, e + 1):
                reason[k] = "moved"
        s = e + 1

    dur = data.info(cam)["duration"]
    spans: list[dict[str, Any]] = []
    s = 0
    while s < n:
        if reason[s] is None:
            s += 1
            continue
        e = s
        while e + 1 < n and reason[e + 1] == reason[s]:
            e += 1
        spans.append({"start": round(float(s), 2), "end": round(float(min(e + 1, dur)), 2), "reason": reason[s],
                      "score": round(float(np.mean(score[s:e + 1])), 2)})
        s = e + 1
    per_second = [{"t": s, "visible": reason[s] is None, "reason": reason[s]} for s in range(n)]
    # when the picture switched between colour and night infrared (seen, not blind: the camera kept looking)
    lighting = [{"t": float(s), "to": "infrared" if mono[s] else "colour"} for s in np.flatnonzero(flip)]
    return {"blind": spans, "per_second": per_second, "reason": reason, "view_of": view_of, "home": home,
            "views": views, "stable": stable, "lighting_changes": lighting,
            **totals(cam, spans, np.where([r is None for r in reason], view_of, -1)),
            "stats": {"luma": luma, "std": gstd, "lap": lap, "sharp": sharp, "sharp_ref": ref, "diff": diff,
                      "block": block, "sat": sat, "px_still": px_still, "det_live": det_live}}


def totals(cam: str, spans: list[dict[str, Any]], view_of: np.ndarray) -> dict[str, Any]:
    """The ONE blind number every view prints, per-day totals, and when each day's home view was on screen
    (`view_of` = home view per second, -1 while blind). Day ids/labels come from data.days, never assumed A/B."""
    by_reason: dict[str, float] = {}
    for b in spans:
        by_reason[b["reason"]] = by_reason.get(b["reason"], 0.0) + b["end"] - b["start"]
    by_day, home_views = [], []
    for d in data.days(cam):
        a, z = float(d["start"]), float(d["end"])
        br: dict[str, float] = {}
        for b in spans:
            ov = min(b["end"], z) - max(b["start"], a)
            if ov > 0:
                br[b["reason"]] = br.get(b["reason"], 0.0) + ov
        by_day.append({"day": d["id"], "label": d["label"], "start": round(a, 2), "end": round(z, 2),
                       "blind_s": round(float(sum(br.values())), 1),
                       "by_reason": {k: round(float(v), 1) for k, v in br.items()}})
        lo, hi = int(np.ceil(a)), min(len(view_of), int(np.ceil(z)))
        vs = view_of[lo:hi]
        vs = vs[vs >= 0]
        if not len(vs):
            continue
        v = int(np.bincount(vs).argmax())
        on = lo + np.flatnonzero(view_of[lo:hi] == v)
        home_views.append({"id": d["id"], "day": d["id"], "label": d["label"], "view": v,
                           "start": float(on[0]), "end": round(float(min(on[-1] + 1, z)), 2), "seen_s": int(len(on))})
    return {"blind_total_s": round(float(sum(by_reason.values())), 1),
            "blind_by_reason": {k: round(float(v), 1) for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1])},
            "blind_by_day": by_day, "home_views": home_views}


# ---------------------------------------------------------------- spots
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _inner(img: np.ndarray, f: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * f):h - int(h * f), int(w * f):w - int(w * f)]   # drop the crop padding (mostly tarmac)


def _hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Same-vehicle score of two crops: mean of H-S colour histogram correlation and 32x24 grey correlation."""
    a, b = _inner(a), _inner(b)
    hc = float(cv2.compareHist(_hist(a), _hist(b), cv2.HISTCMP_CORREL))
    ga = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (32, 24), interpolation=cv2.INTER_AREA)
    gb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (32, 24), interpolation=cv2.INTER_AREA)
    return 0.5 * hc + 0.5 * _ncc(ga, gb)


class CropStore:
    """Crops of a track at its first steady look ("start": ~1 s after it appears) and its last look ("end").

    All requested crops are cut in ONE sequential pass over the video (seeking per crop costs ~1 s each)."""

    def __init__(self, cam: str):
        self.cam = cam
        self.tr = data.tracks(cam)
        self.by_tid = {int(t): g for t, g in self.tr[self.tr.tid >= 0].groupby("tid")}
        self.rows: dict[tuple[int, str], Any] = {}
        self.img: dict[tuple[int, str], np.ndarray] = {}
        self.saved: dict[tuple[int, str], str] = {}

    def want(self, tid: int, sides: tuple[str, ...] = ("start", "end")) -> None:
        if all((tid, sd) in self.rows for sd in sides) or tid not in self.by_tid:
            return
        g = self.by_tid[tid]
        t0, t1 = float(g.t.iat[0]), float(g.t.iat[-1])
        med = g[["x1", "y1", "x2", "y2"]].median().to_numpy()

        def typical(w: pd.DataFrame) -> Any:     # the look whose box is closest to the track's usual box
            w = w if len(w) else g
            return w.iloc[int(np.abs(w[["x1", "y1", "x2", "y2"]].to_numpy() - med).sum(axis=1).argmin())]

        if "end" in sides:
            self.rows[(tid, "end")] = typical(g[g.t >= t1 - 5.0])
        if "start" in sides:
            self.rows[(tid, "start")] = typical(g[(g.t >= min(t0 + 0.5, t1)) & (g.t <= t0 + 5.0)])

    def want_vehicles(self) -> None:
        """Every tracked vehicle can end up as a spot occupant, so ask for all of them up front."""
        for tid, g in self.by_tid.items():
            if g.name.isin(data.VEHICLES).mean() >= 0.5 and len(g) >= 8:
                self.want(tid)

    def frames(self) -> dict[int, list[tuple[int, str]]]:
        fps = data.info(self.cam)["fps"]
        by_f: dict[int, list[tuple[int, str]]] = {}
        for key, r in self.rows.items():
            if key not in self.img:
                by_f.setdefault(int(round(float(r.t) * fps)), []).append(key)
        return by_f

    def cut(self, img: np.ndarray, keys: list[tuple[int, str]]) -> None:
        H, W = img.shape[:2]
        for key in keys:
            r = self.rows[key]
            px, py = r.bw * 0.15, r.bh * 0.15      # same padding as data.crop
            x1, y1 = int(max(0, r.x1 - px)), int(max(0, r.y1 - py))
            x2, y2 = int(min(W, r.x2 + px)), int(min(H, r.y2 + py))
            c = img[y1:y2, x1:x2]
            if c.size:
                self.img[key] = c.copy()

    def fetch(self) -> None:
        """Cut any crops still missing, in one sequential pass."""
        by_f = self.frames()
        if not by_f:
            return
        cap = cv2.VideoCapture(str(data.CAMS[self.cam]["video"]))
        i, last = 0, max(by_f)
        while i <= last and cap.grab():
            if i in by_f:
                ok, img = cap.retrieve()
                if ok:
                    self.cut(img, by_f[i])
            i += 1
        cap.release()

    def get(self, tid: int | None, side: str) -> np.ndarray | None:
        return None if tid is None else self.img.get((tid, side))

    def path(self, tid: int | None, side: str) -> str | None:
        """Write the crop (save_crop naming: crops/<tid>_<t>.jpg) and return its out/<cam>-relative path."""
        if tid is None or (tid, side) not in self.img:
            return None
        if (tid, side) not in self.saved:
            rel = f"crops/{tid}_{float(self.rows[(tid, side)].t):.1f}.jpg"
            p = data.out_dir(self.cam) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), self.img[(tid, side)], [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.saved[(tid, side)] = rel
        return self.saved[(tid, side)]

    def sim(self, a: int | None, b: int | None) -> float | None:
        ia, ib = self.get(a, "end"), self.get(b, "start")
        return None if ia is None or ib is None else round(similarity(ia, ib), 2)


def build_spots(cam: str, hl: dict[str, Any], smp: Samples, obj: pd.DataFrame | None = None,
                crops: "CropStore | None" = None) -> list[dict[str, Any]]:
    obj = data.objects(cam) if obj is None else obj
    if obj.empty or not obj.parked.any():
        return []
    park = obj[obj.parked].sort_values("first_t")
    view_of = hl["view_of"]
    n = len(view_of)

    def obj_view(r) -> int:
        a, b = int(np.floor(r.first_t)), min(n - 1, int(np.ceil(r.last_t)))
        v = view_of[a:b + 1]
        v = v[v >= 0]
        return int(np.bincount(v).argmax()) if len(v) else -1

    clusters: list[dict[str, Any]] = []
    for r in park.itertuples():
        box = [r.x1, r.y1, r.x2, r.y2]
        v = obj_view(r)
        best, bi = None, SPOT_IOU
        for c in clusters:
            if c["view"] != v:
                continue
            i = _iou(box, c["box"])
            if i >= bi:
                best, bi = c, i
        if best is None:
            clusters.append({"view": v, "box": box, "members": [r]})
        else:
            best["members"].append(r)
            best["box"] = list(np.median([[m.x1, m.y1, m.x2, m.y2] for m in best["members"]], axis=0))
    clusters.sort(key=lambda c: (c["view"], c["box"][1], c["box"][0]))

    tr = data.tracks(cam)
    veh = tr[tr.name.isin(data.VEHICLES)]
    vsec = veh.t.round().astype(int)
    vb = veh[["x1", "y1", "x2", "y2"]].to_numpy(float)
    vtid = veh.tid.to_numpy()
    vs = vsec.to_numpy()
    reason = hl["reason"]
    dur = data.info(cam)["duration"]
    crops = CropStore(cam) if crops is None else crops

    scale = SAMPLE_W / smp.w
    spots: list[dict[str, Any]] = []
    spot_cells = []
    for k, c in enumerate(clusters, 1):
        box = [round(float(x), 1) for x in c["box"]]
        bx = [int(c["box"][0] * scale), int(c["box"][1] * scale), int(np.ceil(c["box"][2] * scale)), int(np.ceil(c["box"][3] * scale))]

        # vehicle detections overlapping this spot (IoU >= SPOT_IOU), first tid per second, vectorised
        ix = np.clip(np.minimum(vb[:, 2], box[2]) - np.maximum(vb[:, 0], box[0]), 0, None)
        iy = np.clip(np.minimum(vb[:, 3], box[3]) - np.maximum(vb[:, 1], box[1]), 0, None)
        inter = ix * iy
        union = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
        ok = inter >= SPOT_IOU * np.maximum(union, 1e-9)
        hits: dict[int, int] = {}
        for sec, tid in zip(vs[ok], vtid[ok]):
            hits.setdefault(int(sec), int(tid))
        cover: dict[int, float] = {}      # biggest share of the spot hidden by some other vehicle's box
        area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
        part = (~ok) & (inter > OCCLUDED * area)
        for sec, fr in zip(vs[part], inter[part] / area):
            cover[int(sec)] = max(cover.get(int(sec), 0.0), float(fr))

        def patch(s: int) -> np.ndarray | None:
            im = smp.grey[s]
            return None if im is None else im[bx[1]:bx[3], bx[0]:bx[2]].astype(np.int16)

        ref: tuple[np.ndarray, int | None] | None = None    # last confirmed-occupied look of the spot
        cells: list[tuple[str, int | None, str | None]] = []
        for s in range(n):
            if reason[s] is not None:
                cells.append(("unknown", None, f"camera could not see ({REASON_WORDS.get(reason[s], reason[s])})"))
                continue
            if view_of[s] != c["view"]:
                cells.append(("unknown", None, "camera pointed at another view"))
                continue
            m = next((m for m in c["members"] if m.first_t - 0.5 <= s <= m.last_t + 0.5), None)
            if m is not None:
                cells.append(("present", int(m.id), None))
                pt = patch(s)
                if pt is not None and pt.size:
                    ref = (pt, int(m.id))
                continue
            hit = hits.get(s)
            if hit is not None:
                oid = hit if hit >= 0 else None
                cells.append(("present", oid, "vehicle detected, not a parked track"))
                pt = patch(s)
                if pt is not None and pt.size:
                    ref = (pt, oid)
                continue
            pt = patch(s)
            if ref is not None and pt is not None and pt.size and float(np.abs(pt - ref[0]).mean()) < PATCH_SAME:
                # no detection, but the pixels still look like the parked car: the detector missed it
                cells.append(("present", ref[1], "detector missed it; spot pixels unchanged"))
            elif cover.get(s, 0.0) >= OCCLUDED:
                cells.append(("unknown", None, "spot hidden behind another vehicle"))
            else:
                ref = None
                cells.append(("gone", None, None))
        # backward pass: seconds before the first detection that already look like the car = it was there
        refb: tuple[np.ndarray, int | None] | None = None
        for s in range(n - 1, -1, -1):
            st, oid, _ = cells[s]
            pt = patch(s)
            if st == "present":
                if pt is not None and pt.size:
                    refb = (pt, oid)
            elif st == "gone":
                if refb is not None and pt is not None and pt.size and float(np.abs(pt - refb[0]).mean()) < PATCH_SAME:
                    cells[s] = ("present", refb[1], "detector missed it; spot pixels unchanged")
                else:
                    refb = None
        for oid in {oid for st, oid, _ in cells if st == "present" and oid is not None}:
            crops.want(oid)
        spot_cells.append((k, c, box, cells))
    crops.fetch()      # normally a no-op: run() cut every vehicle crop during the sampling pass
    for k, c, box, cells in spot_cells:
        spans = _bridge(_merge(cells, dur), crops)
        changes = _changes(spans, crops)
        spots.append({"spot_id": f"S{k}", "box": box, "view": int(c["view"]), "spans": spans, "changes": changes})
    return spots


def _merge(cells: list[tuple[str, int | None, str | None]], dur: float) -> list[dict[str, Any]]:
    spans, s, n = [], 0, len(cells)
    while s < n:
        e = s
        while e + 1 < n and cells[e + 1] == cells[s]:
            e += 1
        st, oid, rs = cells[s]
        spans.append({"start": round(float(s), 2), "end": round(float(min(e + 1, dur)), 2), "state": st,
                      "object_id": oid, "reason": rs})
        s = e + 1
    for i, sp in enumerate(spans):
        near_blind = (i > 0 and spans[i - 1]["state"] == "unknown") or (i + 1 < len(spans) and spans[i + 1]["state"] == "unknown")
        if sp["state"] == "gone" and near_blind and sp["end"] - sp["start"] < MIN_GONE_S:
            sp.update(state="unknown", reason="too little clear view next to a blind stretch to call it empty")
    return spans


def _bridge(spans: list[dict[str, Any]], crops: CropStore) -> list[dict[str, Any]]:
    """'gone' between two sightings of the same vehicle (<= BRIDGE_S of gone, unknown gaps skipped) is a
    detector gap, not a departure. Unknown spans stay unknown."""
    known = [i for i, sp in enumerate(spans) if sp["state"] != "unknown"]
    for j, i in enumerate(known):
        a = spans[i]
        if a["state"] != "present" or a["object_id"] is None:
            continue
        gone, k = [], j + 1
        while k < len(known) and spans[known[k]]["state"] == "gone":
            gone.append(known[k])
            k += 1
        if not gone or k >= len(known):
            continue
        b = spans[known[k]]
        if b["object_id"] is None or sum(spans[x]["end"] - spans[x]["start"] for x in gone) > BRIDGE_S:
            continue
        sim = crops.sim(a["object_id"], b["object_id"]) if a["object_id"] != b["object_id"] else 1.0
        if sim is not None and sim >= SAME_SIM:
            for x in gone:
                spans[x].update(state="present", object_id=a["object_id"],
                                reason="no detection, but the same vehicle is seen before and after")
    out: list[dict[str, Any]] = []
    for sp in spans:      # re-merge runs that now read the same
        if out and out[-1]["state"] == sp["state"] == "present" and out[-1]["object_id"] == sp["object_id"]:
            out[-1]["end"] = sp["end"]
            out[-1]["reason"] = out[-1]["reason"] or sp["reason"]
        else:
            out.append(dict(sp))
    return out


def _changes(spans: list[dict[str, Any]], crops: CropStore) -> list[dict[str, Any]]:
    """Walk the known (present/gone) spans; unknown gaps are skipped, never read as 'gone'."""
    out = []
    prev = None          # last known span
    gap = False
    for sp in spans:
        if sp["state"] == "unknown":
            gap = gap or prev is not None
            continue
        if prev is not None:
            a, b = prev, sp
            t = b["start"]
            ch = None
            if a["state"] == "gone" and b["state"] == "present":
                ch = {"t": t, "kind": "arrived", "before": None, "after": crops.path(b["object_id"], "start"),
                      "same_vehicle": None, "similarity": None}
            elif a["state"] == "present" and b["state"] == "gone":
                ch = {"t": t, "kind": "left", "before": crops.path(a["object_id"], "end"), "after": None,
                      "same_vehicle": None, "similarity": None}
            elif a["state"] == "present" and b["state"] == "present" and a["object_id"] != b["object_id"]:
                sim = crops.sim(a["object_id"], b["object_id"])
                same = None if sim is None else bool(sim >= SAME_SIM)
                if same is not True:        # a tracker id change on the same car is not a change
                    ch = {"t": t, "kind": "swapped", "before": crops.path(a["object_id"], "end"),
                          "after": crops.path(b["object_id"], "start"), "same_vehicle": same, "similarity": sim}
            if ch is not None:
                if gap:
                    ch["note"] = f"happened while the camera could not see (between {a['end']:.0f}s and {t:.0f}s)"
                out.append(ch)
        prev = sp
        gap = False
    return out


# ---------------------------------------------------------------- run
def run(cam: str) -> dict[str, Any]:
    t0 = time.time()
    crops = CropStore(cam)
    crops.want_vehicles()
    smp = sample(cam, crops=crops)       # one decode pass: 1 fps health samples + every vehicle crop
    hl = health(cam, smp)
    t_health = time.time() - t0
    t1 = time.time()
    obj = data.objects(cam)
    spots = build_spots(cam, hl, smp, obj, crops)
    t2 = time.time()
    led = {"camera": cam, "seconds": smp.seconds, "blind": hl["blind"],
           "blind_total_s": hl["blind_total_s"], "blind_by_reason": hl["blind_by_reason"],
           "blind_by_day": hl["blind_by_day"], "home_views": hl["home_views"], "reason_words": REASON_WORDS,
           "spots": spots, "per_second": hl["per_second"]}
    od = data.out_dir(cam)
    data.write_json(od / "ledger.json", led)
    blind_s = sum(1 for p in hl["per_second"] if not p["visible"])
    kinds: dict[str, int] = {}
    for sp in spots:
        for ch in sp["changes"]:
            kinds[ch["kind"]] = kinds.get(ch["kind"], 0) + 1
    n_parked = int(obj.parked.sum()) if len(obj) else 0
    per_day = "; ".join(f"{d['label']} ({d['day']}): {d['blind_s']} s {d['by_reason'] or ''}".rstrip()
                        for d in hl["blind_by_day"])
    hv = "; ".join(f"{h['label']}: view {h['view']} on screen {h['start']:.0f}-{h['end']:.0f}s ({h['seen_s']} s)"
                   for h in hl["home_views"])
    lines = [f"ledger for {cam}: {smp.seconds} seconds sampled at 1 fps, {SAMPLE_W} px wide",
             f"blind_total_s: {hl['blind_total_s']} ({blind_s} sampled seconds) in {len(hl['blind'])} spans;"
             f" by reason: {hl['blind_by_reason'] or 'none'}",
             f"blind by day: {per_day}",
             f"home views: {len(hl['home'])} (of {len(hl['views'])} stable views found): {hv or 'none'}",
             "lighting changes (not blind): " + (", ".join(f"to {x['to']} at {x['t']:.0f}s" for x in hl["lighting_changes"])
                                                 or "none"),
             "reasons: " + ", ".join(f"{k} = {v}" for k, v in REASON_WORDS.items()),
             f"parked tracks: {n_parked}; spots: {len(spots)}; changes: {kinds or 'none'}",
             f"runtime: {time.time() - t0:.1f}s (sample+health {t_health:.1f}s, spots+crops {t2 - t1:.1f}s)"]
    (od / "ledger_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(" | ".join(lines))
    return led


if __name__ == "__main__":
    for c in sys.argv[1:] or DEFAULT_CAMS:
        run(c)
