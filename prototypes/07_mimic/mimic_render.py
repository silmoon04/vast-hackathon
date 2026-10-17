"""MIMIC rendering: result cards with trajectory overlays, query sketch panels, hero images, match clips."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mimic_core as mc  # noqa: E402

pk = mc.pk
BG = (27, 23, 20)            # BGR of #14171b
PANEL = (39, 34, 29)
FG = (236, 233, 230)
MUTED = (178, 165, 154)
ACC = (255, 160, 110)        # light blue (BGR)
GREEN, AMBER, RED = (91, 200, 70), (40, 175, 240), (70, 70, 235)
FONT = cv2.FONT_HERSHEY_DUPLEX
_ROWS: dict[str, dict] = {}
_FRAMES: dict = {}


def frame(video: str, t: float, max_w: int = 640) -> np.ndarray:
    """Frame at t, cached (downscaled to <= max_w): the same tracklet appears in several presets and in the hero."""
    k = (video, round(t, 2))
    if k not in _FRAMES:
        fr = pk.frame_at(video, t)
        if fr.shape[1] > max_w:
            fr = cv2.resize(fr, (max_w, int(fr.shape[0] * max_w / fr.shape[1])), interpolation=cv2.INTER_AREA)
        if len(_FRAMES) > 160:
            _FRAMES.pop(next(iter(_FRAMES)))
        _FRAMES[k] = fr
    return _FRAMES[k]


def text(img, s, xy, scale=0.6, col=FG, thick=1, font=FONT, halo=BG):
    cv2.putText(img, s, (int(xy[0]), int(xy[1])), font, scale, halo, thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, (int(xy[0]), int(xy[1])), font, scale, col, thick, cv2.LINE_AA)
    return img


def tsize(s, scale=0.6, thick=1):
    (w, h), b = cv2.getTextSize(s, FONT, scale, thick)
    return w, h


def _rows(video: str) -> dict:
    if video not in _ROWS:
        r = pk.tracks(video, verbose=False).rows
        _ROWS[video] = {tid: g.sort_values("f") for tid, g in r[r["tid"] >= 0].groupby("tid")}
    return _ROWS[video]


def path_px(d: dict, tl, f: int | None = None) -> np.ndarray:
    """The tracklet's (camera-compensated) path in pixel coordinates of frame f (reference frame when f is None)."""
    xy = tl.xy * np.array([tl.W, tl.H])
    if f is None:
        return xy
    w = d["wins"][tl.video]
    invs, sc = w.attrs["inv"], w.attrs["scale"]
    fs = min(invs, key=lambda k: abs(k - f))
    cum = np.linalg.inv(invs[fs][1])
    hom = np.hstack([xy * sc, np.ones((len(xy), 1))])
    return (cum @ hom.T).T[:, :2] / sc


def box_at(d: dict, tl, t: float):
    g = _rows(tl.video).get(tl.tid)
    if g is None:
        return None
    i = int(np.argmin(np.abs(g["t"].values - t)))
    r = g.iloc[i]
    return (r["x1"], r["y1"], r["x2"], r["y2"]), int(r["f"])


def grad(i: int, n: int):
    """blue -> orange along the path (BGR)."""
    a = i / max(n - 1, 1)
    return (int(255 * (1 - a) + 40 * a), int(140 * (1 - a) + 160 * a), int(40 * (1 - a) + 255 * a))


def draw_path(img, pts, thick=3, dots=True):
    n = len(pts)
    p = np.round(pts).astype(np.int32)
    if n > 1:
        cv2.polylines(img, [p], False, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.polylines(img, [p], False, (60, 200, 255), thick, cv2.LINE_AA)
    if dots and n > 1:
        cv2.circle(img, tuple(p[0]), thick + 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p[0]), thick + 2, GREEN, -1, cv2.LINE_AA)
        k = max(0, n - 8)
        v = (p[-1] - p[k]).astype(float)
        if np.linalg.norm(v) > 1e-6:
            v /= np.linalg.norm(v)
            L, Wd = 3.2 * thick + 6, 1.8 * thick + 3                      # arrow head size is fixed in pixels
            tip = p[-1] + v * L * 0.5
            base = p[-1] - v * L * 0.5
            nrm = np.array([-v[1], v[0]])
            tri = np.array([tip, base + nrm * Wd, base - nrm * Wd]).astype(np.int32)
            cv2.fillPoly(img, [tri], (0, 0, 0), cv2.LINE_AA)
            cv2.fillPoly(img, [(tri - tip).astype(float).__mul__(0.78).__add__(tip).astype(np.int32)], (50, 60, 255), cv2.LINE_AA)
    return img


PAUSE_COL = (255, 220, 90)       # light cyan-blue (BGR): a halt on the path


def draw_pause_ring(img, xy, label: str, r: int = 13, scale: float = 0.5):
    x, y = int(xy[0]), int(xy[1])
    ov = img.copy()
    cv2.circle(ov, (x, y), r, PAUSE_COL, -1, cv2.LINE_AA)
    img[:] = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    cv2.circle(img, (x, y), r, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, PAUSE_COL, 2, cv2.LINE_AA)
    if label:
        tw, th = tsize(label, scale)
        bx, by = x - tw // 2, y - r - 8
        by = by if by - th > 2 else y + r + th + 8
        text(img, label, (max(2, min(bx, img.shape[1] - tw - 2)), by), scale, PAUSE_COL)


def draw_pauses(img, pts_px, runs, w: int = 400, max_n: int = 3):
    """Rings with 'x s' where the object halts (walk, stop, walk is then visible in a single frame)."""
    for a, b, secs in sorted(runs, key=lambda r: -r[2])[:max_n]:
        k = (a + b) // 2
        if 0 <= k < len(pts_px):
            draw_pause_ring(img, pts_px[k], f"{secs:.0f} s" if secs >= 2 else f"{secs:.1f} s", r=max(8, w // 32), scale=max(0.4, w / 900))


def spark(img, x: int, y: int, w: int, h: int, vals, col, label: str = "", ref=None, ref_col=(150, 150, 150), vmax: float = 3.0):
    """Tiny speed-over-time plot (speed / mean speed): a stop shows up as a dip to zero."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (20, 17, 14), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 80, 70), 1)
    cv2.line(img, (x + 1, y + h - 6), (x + w - 1, y + h - 6), (60, 54, 48), 1)

    def pl(v, c, t):
        v = np.clip(np.asarray(v, float), 0, vmax)
        P = np.stack([x + 4 + (w - 8) * np.arange(len(v)) / max(len(v) - 1, 1), y + h - 6 - (h - 18) * v / vmax], 1).astype(np.int32)
        cv2.polylines(img, [P], False, c, t, cv2.LINE_AA)
    if ref is not None:
        pl(ref, ref_col, 1)
    pl(vals, col, 2)
    if label:
        text(img, label, (x + 4, y + 11), 0.36, MUTED, halo=(20, 17, 14))
    return img


def card(d: dict, key: str, w: int = 400, tag: str = "", sub: str = "", t: float | None = None, pauses: bool = False, dim: bool = False) -> np.ndarray:
    """Frame at the middle of the tracklet with its trajectory (projected into that frame) and the object box."""
    tl = d["tls"][key]
    t = float(tl.t[len(tl.t) // 2]) if t is None else t
    fr0 = frame(tl.video, t)
    full_w = d["meta"][tl.video]["w"]
    fr = cv2.resize(fr0, (w, int(fr0.shape[0] * w / fr0.shape[1])), interpolation=cv2.INTER_AREA)
    s = w / full_w                                           # path / box coordinates are in source pixels
    bx = box_at(d, tl, t)
    f = bx[1] if bx else None
    pts = path_px(d, tl, f) * s
    if dim:                                                  # weak result: greyed out so it does not read as a match
        fr = (fr.astype(np.float32) * 0.42 + 18).astype(np.uint8)
    draw_path(fr, pts, thick=max(2, w // 160))
    if pauses:
        draw_pauses(fr, pts, mc.find_pauses(tl, d["meta"][tl.video]["fps"]), w)
    if bx:
        x1, y1, x2, y2 = (np.array(bx[0]) * s).astype(int)
        cv2.rectangle(fr, (x1, y1), (x2, y2), (255, 255, 255), 1 if dim else 2, cv2.LINE_AA)
    if tag:
        tw, th = tsize(tag, 0.55)
        cv2.rectangle(fr, (0, 0), (tw + 14, th + 12), (0, 0, 0), -1)
        text(fr, tag, (7, th + 5), 0.55)
    if sub:
        tw, th = tsize(sub, 0.45)
        cv2.rectangle(fr, (0, fr.shape[0] - th - 12), (tw + 12, fr.shape[0]), (0, 0, 0), -1)
        text(fr, sub, (6, fr.shape[0] - 6), 0.45, MUTED)
    return fr


def sketch_panel(pts: np.ndarray, w: int, h: int, title: str = "YOU DRAW", sub: str = "", aspect: float = 16 / 9,
                 marks: list | None = None, speed=None, rot: int = 0) -> np.ndarray:
    """The query sketch. marks = [(x, y, label)] in canvas coordinates (pauses, dwell); speed = the sketch's speed profile (sparkline);
    rot = how far the best match was turned relative to this sketch (a small 'as matched' copy is drawn when not 0)."""
    img = np.full((h, w, 3), PANEL, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (90, 80, 70), 1)
    for g in range(1, 8):
        cv2.line(img, (int(w * g / 8), 0), (int(w * g / 8), h), (47, 42, 36), 1)
    for g in range(1, 5):
        cv2.line(img, (0, int(h * g / 5)), (w, int(h * g / 5)), (47, 42, 36), 1)
    # fit the sketch into the panel keeping its isotropic shape
    q = pts * np.array([aspect, 1.0])
    lo, hi = q.min(0), q.max(0)
    sc = min((w * 0.78) / max(hi[0] - lo[0], 1e-6), (h * 0.62) / max(hi[1] - lo[1], 1e-6))
    c = (lo + hi) / 2
    cen = np.array([w / 2, h * 0.52])

    def to_panel(p):
        return (np.asarray(p, float) * np.array([aspect, 1.0]) - c) * sc + cen
    P = to_panel(pts)
    draw_path(img, P, thick=5)
    for (mx, my, lab) in (marks or []):
        mp = to_panel([mx, my])
        if lab.startswith("stays"):                          # dwell: a halo around the whole scribble
            ctr = (int(P[:, 0].mean()), int(P[:, 1].mean()))
            rx, ry = int((P[:, 0].max() - P[:, 0].min()) / 2 + 22), int((P[:, 1].max() - P[:, 1].min()) / 2 + 22)
            cv2.ellipse(img, ctr, (rx, ry), 0, 0, 360, PAUSE_COL, 2, cv2.LINE_AA)
            text(img, lab, (ctr[0] - tsize(lab, 0.5)[0] // 2, ctr[1] + ry + 20), 0.5, PAUSE_COL)
        else:
            draw_pause_ring(img, mp, lab, r=15, scale=0.55)
    text(img, title, (14, 30), 0.8, ACC, 1)
    if sub:
        text(img, sub, (14, h - 12), 0.5, MUTED)
    if speed is not None:
        spark(img, w - 142, h - 54, 130, 44, speed, (60, 200, 255), "speed (sketch)")
    if rot:
        mw, mh = 164, 76
        mini = np.full((mh, mw, 3), (30, 26, 22), np.uint8)
        cv2.rectangle(mini, (0, 0), (mw - 1, mh - 1), ACC, 1)
        qq = (pts * np.array([aspect, 1.0])) - (pts * np.array([aspect, 1.0])).mean(0)
        a = np.deg2rad(rot)
        qq = qq @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).T
        lo2, hi2 = qq.min(0), qq.max(0)
        sc2 = min((mw * 0.72) / max(hi2[0] - lo2[0], 1e-6), (mh * 0.46) / max(hi2[1] - lo2[1], 1e-6))
        P2 = (qq - (lo2 + hi2) / 2) * sc2 + np.array([mw / 2, mh * 0.60])
        draw_path(mini, P2, thick=2)
        text(mini, f"as matched: turned {rot} deg", (4, 13), 0.36, ACC, halo=(30, 26, 22))
        img[8:8 + mh, w - mw - 8:w - 8] = mini
    return img


def stamp(img, label: str, col, pos: tuple[int, int], scale=1.4, angle=-7):
    """Rubber-stamp style verdict, alpha-blended and rotated."""
    tw, th = tsize(label, scale, 3)
    pad = 16
    sw, sh = tw + 2 * pad + 8, th + 2 * pad + 8
    layer = np.zeros((sh + 40, sw + 40, 3), np.uint8)
    mask = np.zeros(layer.shape[:2], np.uint8)
    for im, c in ((layer, col), (mask, 255)):
        cv2.rectangle(im, (20, 20), (20 + sw, 20 + sh), c, 5, cv2.LINE_AA)
        cv2.putText(im, label, (20 + pad + 4, 20 + pad + 4 + th), FONT, scale, c, 3, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((layer.shape[1] / 2, layer.shape[0] / 2), -angle, 1.0)
    layer = cv2.warpAffine(layer, M, (layer.shape[1], layer.shape[0]))
    mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    x, y = pos
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + layer.shape[1]), min(H, y + layer.shape[0])
    if x1 <= x0 or y1 <= y0:
        return img
    sl = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    m = (mask[sl] > 0)[..., None]
    reg = img[y0:y1, x0:x1]
    reg[:] = np.where(m, (0.12 * reg + 0.88 * layer[sl]).astype(np.uint8), reg)
    return img


def timeline_strip(img, x, y, w, h, duration, cover_spans, marks):
    """Usable-camera coverage (blue) with the match spans (rank colours) on top."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 54, 48), -1)
    for a, b in cover_spans:
        cv2.rectangle(img, (x + int(w * a / duration), y), (x + max(int(w * b / duration), int(w * a / duration) + 1), y + h), (120, 85, 55), -1)
    for a, b, col in marks:
        xa = x + int(w * a / duration)
        xb = max(x + int(w * b / duration), xa + 4)
        cv2.rectangle(img, (xa, y - 5), (xb, y + h + 5), col, -1)
        cv2.rectangle(img, (xa, y - 5), (xb, y + h + 5), (0, 0, 0), 1)
    text(img, "0:00", (x, y + h + 24), 0.45, MUTED)
    lab = pk.mmss(duration)[:-2]
    text(img, lab, (x + w - tsize(lab, 0.45)[0], y + h + 24), 0.45, MUTED)
    return img


def apex_time(tl) -> float:
    """The moment worth showing: where the path turns hardest (heading change over ~1 s), else the middle."""
    n = len(tl.t)
    fps = (n - 1) / max(tl.t[-1] - tl.t[0], 1e-6)
    h = max(2, int(round(fps * 0.5)))
    if n < 2 * h + 3:
        return float(tl.t[n // 2])
    v = np.diff(tl.xy * np.array([tl.W / tl.H, 1.0]), axis=0)
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    best, bi = 0.0, n // 2
    for i in range(h, len(ang) - h):
        dev = abs(ang[min(len(ang) - 1, i + h)] - ang[i - h])
        if dev > best and np.linalg.norm(v[i - h:i + h].sum(0)) > 0:
            best, bi = dev, i
    return float(tl.t[bi]) if best > 0.5 else float(tl.t[n // 2])


def write_clip(d: dict, key: str, dst: Path, tag: str = "", width: int = 640, pad: float = 0.5, max_len: float = 18.0) -> Path:
    """Short mp4 around the match: box + trail of the object, camera-compensated path projected into every frame."""
    tl = d["tls"][key]
    t0, t1 = max(0.0, float(tl.t[0]) - pad), float(tl.t[-1]) + pad
    if t1 - t0 > max_len:
        tm = apex_time(tl)
        t0, t1 = max(0.0, tm - max_len / 2), tm + max_len / 2
    info = pk.info(tl.video)
    full = path_px(d, tl)
    g = _rows(tl.video).get(tl.tid)
    with pk.Mp4Writer(dst, info["fps"], width) as wr:
        for i, t, fr in pk.iter_frames(tl.video, start=t0, end=t1):
            s = width / fr.shape[1]
            fr = cv2.resize(fr, (width, int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
            f = int(round(t * d["meta"][tl.video]["src_fps"]))
            pts = path_px(d, tl, f) * s
            overlay = fr.copy()
            cv2.polylines(overlay, [np.round(pts).astype(np.int32)], False, (60, 200, 255), 2, cv2.LINE_AA)
            fr = cv2.addWeighted(overlay, 0.45, fr, 0.55, 0)
            k = int(np.searchsorted(tl.t, t))
            if tl.t[0] - 1e-6 <= t <= tl.t[-1] + 1e-6 and k > 1:
                draw_path(fr, pts[:min(k, len(pts))], thick=3, dots=True)
            if g is not None and tl.t[0] - 0.2 <= t <= tl.t[-1] + 0.2:
                j = int(np.argmin(np.abs(g["t"].values - t)))
                r = g.iloc[j]
                cv2.rectangle(fr, (int(r["x1"] * s), int(r["y1"] * s)), (int(r["x2"] * s), int(r["y2"] * s)), (255, 255, 255), 2, cv2.LINE_AA)
            pk.label(fr, f"{tag}  {pk.mmss(t)}", (6, 6), scale=0.5)
            wr.write(fr)
    return dst


def filmstrip(d: dict, key: str, n: int = 8, w: int = 240) -> np.ndarray:
    """n frames spread over the tracklet, each with the object box, trail so far and elapsed time: how a human checks motion."""
    tl = d["tls"][key]
    ts = np.linspace(tl.t[0], tl.t[-1], n)
    tiles = []
    for t in ts:
        fr0 = frame(tl.video, float(t))
        fr = cv2.resize(fr0, (w, int(fr0.shape[0] * w / fr0.shape[1])), interpolation=cv2.INTER_AREA)
        s = w / d["meta"][tl.video]["w"]
        bx = box_at(d, tl, float(t))
        f = bx[1] if bx else None
        pts = path_px(d, tl, f) * s
        k = int(np.searchsorted(tl.t, t))
        draw_path(fr, pts[:max(2, min(k, len(pts)))], thick=2, dots=False)
        if bx:
            x1, y1, x2, y2 = (np.array(bx[0]) * s).astype(int)
            cv2.rectangle(fr, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        pk.label(fr, f"+{t - tl.t[0]:.1f}s", (3, 3), scale=0.4)
        tiles.append(fr)
    return pk.sheet(tiles, cols=n, tile_w=w, pad=2)
