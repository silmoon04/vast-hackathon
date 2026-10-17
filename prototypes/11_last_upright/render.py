"""Drawing for LAST UPRIGHT: skeleton overlays, posture strip, the hero card (PIL) and the annotated mp4."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402
import posture as P  # noqa: E402

# state colours (RGB) chosen to stay distinguishable without red/green: teal / blue / red / grey
COL = {P.UP: (31, 181, 160), P.SIT: (79, 140, 255), P.DOWN: (255, 77, 79), P.PART: (170, 140, 230), P.UNK: (130, 140, 150), "GONE": (52, 60, 70)}
LEVEL_COL = {"FALL": (226, 52, 60), "LIE": (240, 160, 30), "LIE_BED": (79, 140, 255), "NO_ALARM": (31, 181, 160)}
BG, CARD, INK, MUTED = (16, 21, 28), (27, 34, 44), (236, 240, 244), (150, 162, 176)
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4)]
_FONTS: dict[tuple[int, bool], Any] = {}


def font(size: int, bold: bool = False):
    k = (size, bold)
    if k not in _FONTS:
        names = (["bahnschrift.ttf", "segoeuib.ttf", "arialbd.ttf"] if bold else ["bahnschrift.ttf", "segoeui.ttf", "arial.ttf"]) + ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
        f = None
        for n in names:
            try:
                f = ImageFont.truetype(n, size)
                if bold and n == "bahnschrift.ttf":
                    try:
                        f.set_variation_by_name("Bold")
                    except Exception:
                        pass
                break
            except Exception:
                continue
        _FONTS[k] = f or ImageFont.load_default()
    return _FONTS[k]


def bgr(c):
    return (int(c[2]), int(c[1]), int(c[0]))


# --------------------------------------------------------------------------------------- frame annotation
def draw_person(img: np.ndarray, kp: np.ndarray, box, state: str, label: str | None = None, thick: int = 2) -> np.ndarray:
    c = bgr(COL.get(state, COL[P.UNK]))
    for a, b in EDGES:
        if kp[a, 2] >= P.KP_VIS and kp[b, 2] >= P.KP_VIS:
            cv2.line(img, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), c, thick + 1, cv2.LINE_AA)
    for i in range(17):
        if kp[i, 2] >= P.KP_VIS:
            cv2.circle(img, (int(kp[i, 0]), int(kp[i, 1])), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), c, thick, cv2.LINE_AA)
    if label:
        s = max(0.5, min(img.shape[1] / 1400, 0.9))
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, s, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), c, -1)
        cv2.putText(img, label, (x1 + 4, ty - 3), cv2.FONT_HERSHEY_DUPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def annotate_frame(img: np.ndarray, rows: pd.DataFrame, kps: np.ndarray, only_pid=None,
                   label: bool = True) -> np.ndarray:
    """only_pid: None (everyone), one person id or a set of ids (the hero panels draw only the person under review)."""
    out = img.copy()
    keep = None if only_pid is None else ({only_pid} if isinstance(only_pid, (int, np.integer)) else set(only_pid))
    for i, r in enumerate(rows.itertuples()):
        if keep is not None and r.person not in keep:
            continue
        lab = f"{r.state} {r.tilt:.0f}°" if label and r.tilt == r.tilt else (r.state if label else None)
        draw_person(out, kps[i], (r.x1, r.y1, r.x2, r.y2), r.state, lab, thick=max(2, img.shape[1] // 640))
    return out


def crop_panel(img: np.ndarray, boxes: list[tuple[float, float, float, float]], pw: int, ph: int) -> np.ndarray:
    """Fit img into a pw x ph panel: landscape frames are cropped around the person, portrait ones are letterboxed."""
    H, W = img.shape[:2]
    if W < H * 1.1 or not boxes:
        s = min(pw / W, ph / H)
        im = cv2.resize(img, (max(1, int(W * s)), max(1, int(H * s))), interpolation=cv2.INTER_AREA)
        canvas = np.full((ph, pw, 3), (28, 22, 18), np.uint8)
        y0, x0 = (ph - im.shape[0]) // 2, (pw - im.shape[1]) // 2
        canvas[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        return canvas
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ch = max((y2 - y1) * 1.9, (x2 - x1) * 1.9 * ph / pw, H * 0.45)
    ch = min(ch, H, W * ph / pw)
    cw = ch * pw / ph
    ox = float(np.clip(cx - cw / 2, 0, W - cw))
    oy = float(np.clip(cy - ch / 2, 0, H - ch))
    crop = img[int(oy):int(oy + ch), int(ox):int(ox + cw)]
    return cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------------- strip
def spans_of(rows: pd.DataFrame, duration: float) -> list[tuple[float, float, str]]:
    """(start, end, state|GONE) over [0, duration] for one person."""
    r = rows.sort_values("t")
    t, st = r["t"].to_numpy(), r["state"].to_numpy()
    if len(t) == 0:
        return [(0.0, duration, "GONE")]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.07
    out: list[tuple[float, float, str]] = []
    if t[0] > dt * 1.5:
        out.append((0.0, float(t[0]), "GONE"))
    for i in range(len(t)):
        end = float(t[i + 1]) if i + 1 < len(t) and t[i + 1] - t[i] <= 2.5 * dt else float(t[i]) + dt
        s = str(st[i])
        if out and out[-1][2] == s and abs(out[-1][1] - t[i]) < 1e-6:
            out[-1] = (out[-1][0], end, s)
        else:
            out.append((float(t[i]), end, s))
        if i + 1 < len(t) and t[i + 1] - end > 1e-6:
            out.append((end, float(t[i + 1]), "GONE"))
    if out[-1][1] < duration - dt:
        out.append((out[-1][1], duration, "GONE"))
    return out


def draw_strip(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, spans, duration: float,
               marks: list[tuple[float, str, tuple[int, int, int]]] = (), ticks: int = 8, t0: float = 0.0) -> None:
    """Strip over the clip window [t0, duration]."""
    span_t = max(duration - t0, 1e-6)
    d.rounded_rectangle((x, y, x + w, y + h), 6, fill=COL["GONE"])
    for s, e, st in spans:
        s, e = max(s, t0), max(e, t0)
        if e <= s:
            continue
        xs, xe = x + w * (min(s, duration) - t0) / span_t, x + w * (min(e, duration) - t0) / span_t
        d.rectangle((xs, y, max(xe, xs + 1), y + h), fill=COL.get(st, COL[P.UNK]))
    for k in range(ticks + 1):
        tx = x + w * k / ticks
        d.line((tx, y + h, tx, y + h + 5), fill=MUTED, width=1)
        lab = f"{t0 + k * span_t / ticks:.1f}s"
        d.text((min(tx, x + w - 30) if k == ticks else max(tx - 12, x), y + h + 6), lab, font=font(13), fill=MUTED)
    prev_end = -1e9
    for t, lab, col in sorted(marks, key=lambda m: m[0]):
        mx = x + w * (t - t0) / span_t
        d.polygon([(mx - 8, y - 12), (mx + 8, y - 12), (mx, y - 1)], fill=col)
        d.line((mx, y - 1, mx, y + h), fill=(255, 255, 255), width=2)
        label = f"{lab} {t:.1f}s"
        tw = d.textlength(label, font=font(14, True))
        lx = min(max(mx - tw / 2, x), x + w - tw)
        if lx < prev_end + 8:   # neighbour label in the way: start this one right of the marker
            lx = max(mx + 4, prev_end + 8)
        d.text((lx, y - 32), label, font=font(14, True), fill=col)
        prev_end = lx + tw


def legend(d: ImageDraw.ImageDraw, x: int, y: int) -> None:
    for name, st in (("UPRIGHT", P.UP), ("SITTING", P.SIT), ("DOWN", P.DOWN), ("partial view", P.PART), ("unclear", P.UNK), ("not in view", "GONE")):
        d.rectangle((x, y + 3, x + 12, y + 15), fill=COL[st])
        d.text((x + 17, y), name, font=font(13), fill=MUTED)
        x += 28 + d.textlength(name, font=font(13))


# --------------------------------------------------------------------------------------------------- hero
def pil_from_bgr(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def clip_text(d: ImageDraw.ImageDraw, text: str, f, maxw: int) -> str:
    if d.textlength(text, font=f) <= maxw:
        return text
    while text and d.textlength(text + "...", font=f) > maxw:
        text = text[:-1]
    return text + "..."


def wrap(d: ImageDraw.ImageDraw, text: str, f, maxw: int, max_lines: int = 2) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = clip_text(d, lines[-1] + "...", f, maxw)
    return lines


def chip(d: ImageDraw.ImageDraw, x: int, y: int, text: str, fill, size: int = 17, fg=(255, 255, 255)) -> int:
    f = font(size, True)
    w = int(d.textlength(text, font=f)) + 20
    d.rounded_rectangle((x, y, x + w, y + size + 14), 8, fill=fill)
    d.text((x + 10, y + 6), text, font=f, fill=fg)
    return w


RUNG_COL = {"log": (240, 160, 30), "check": (245, 120, 40), "urgent": (226, 52, 60), "long-lie": (170, 40, 120)}


def card(d: ImageDraw.ImageDraw, box, title: str, tfont: int = 13) -> None:
    d.rounded_rectangle(box, 12, fill=CARD)
    d.text((box[0] + 14, box[1] + 10), title, font=font(tfont, True), fill=MUTED)


def hero_card(c: dict[str, Any], path: Path) -> Path:
    """c: stamp, level, headline, sub, panels (list of 2 or 3: img BGR + cap + state + sub), big_txt, speed_txt, arrow_sub,
    strip (+ strip_t0), marks, clock, ladder, ladder_sub, vlm, boxonly, title, credit."""
    W, H = 1280, 720
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    panels = c["panels"]
    n = len(panels)
    # header
    d.text((30, 16), "LAST UPRIGHT", font=font(30, True), fill=INK)
    d.text((236, 26), "when did they go down, and for how long?", font=font(17), fill=MUTED)
    d.text((W - 30, 24), c["title"], font=font(18, True), fill=MUTED, anchor="ra")
    # verdict stamp + headline (+ the one transition time on the right when there are three panels)
    col = LEVEL_COL[c["level"]]
    sw = int(d.textlength(c["stamp"], font=font(46, True))) + 44
    d.rounded_rectangle((30, 66, 30 + sw, 134), 12, fill=col)
    d.rounded_rectangle((36, 72, 24 + sw, 128), 8, outline=(255, 255, 255), width=2)
    d.text((30 + sw / 2, 100), c["stamp"], font=font(46, True), fill=(255, 255, 255), anchor="mm")
    right_w = 190 if n == 3 else 0
    d.text((52 + sw, 68), clip_text(d, c["headline"], font(27, True), W - 82 - sw - right_w), font=font(27, True), fill=INK)
    for i, ln in enumerate(wrap(d, c["sub"], font(17), W - 82 - sw - right_w)):
        d.text((52 + sw, 104 + i * 21), ln, font=font(17), fill=MUTED)
    if n == 3:
        d.text((W - 30, 98), c["big_txt"], font=font(46, True), fill=col, anchor="rm")
        for i, ln in enumerate(wrap(d, c["speed_txt"], font(15, True), 190, 2)):
            d.text((W - 30, 130 + i * 17), ln, font=font(15, True), fill=INK, anchor="rm")
    # panels
    py, ph = 150, 304
    pw, gap = (540, 140) if n == 2 else (390, 25)
    xs = [30 + k * (pw + gap) for k in range(n)]
    for p, x in zip(panels, xs):
        panel = pil_from_bgr(cv2.resize(p["img"], (pw, ph), interpolation=cv2.INTER_AREA))
        im.paste(panel, (x, py))
        d.rounded_rectangle((x - 1, py - 1, x + pw, py + ph), 4, outline=COL.get(p["state"], MUTED), width=3)
        chip(d, x + 8, py + 8, p["cap"], COL.get(p["state"], (60, 70, 80)), size=17 if n == 2 else 15)
        if p.get("gap_txt"):
            f = font(15, True)
            tw = int(d.textlength(p["gap_txt"], font=f)) + 16
            d.rounded_rectangle((x + pw - tw - 8, py + 8, x + pw - 8, py + 34), 7, fill=(20, 26, 34))
            d.text((x + pw - tw, py + 13), p["gap_txt"], font=f, fill=INK)
        if p.get("sub"):
            d.text((x + 10, py + ph - 24), clip_text(d, p["sub"], font(14, True), pw - 20), font=font(14, True), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    if n == 2:   # arrow + interval between the two panels
        mx = 570 + 70
        d.text((mx, py + 62), c["big_txt"], font=font(36, True), fill=col, anchor="mm")
        for i, ln in enumerate(wrap(d, c["speed_txt"], font(16, True), 128)):
            d.text((mx, py + 98 + i * 20), ln, font=font(16, True), fill=INK, anchor="mm")
        d.line((582, py + 150, 698, py + 150), fill=col, width=6)
        d.polygon([(698, py + 138), (698, py + 162), (712, py + 150)], fill=col)
        for i, ln in enumerate(wrap(d, c["arrow_sub"], font(13), 128)):
            d.text((mx, py + 182 + i * 17), ln, font=font(13), fill=MUTED, anchor="mm")
    else:
        for k in range(2):
            ax = xs[k] + pw + 2
            d.line((ax, py + ph // 2, ax + 17, py + ph // 2), fill=col, width=4)
            d.polygon([(ax + 17, py + ph // 2 - 9), (ax + 17, py + ph // 2 + 9), (ax + 24, py + ph // 2)], fill=col)
    # posture strip
    sy = 512
    d.text((30, sy - 52), "POSTURE OVER TIME (this person)", font=font(14, True), fill=MUTED)
    legend(d, 330, sy - 54)
    draw_strip(d, 30, sy, W - 60, 34, c["strip"], c["duration"], c["marks"], t0=c.get("strip_t0", 0.0))
    # bottom cards: clock | ladder | Cosmos | box-only baseline
    by, bh = 588, 98
    card(d, (30, by, 292, by + bh), c["clock"]["title"])
    d.text((44, by + 28), c["clock"]["big"], font=font(40, True), fill=col)
    d.text((44, by + 76), clip_text(d, c["clock"]["sub"], font(12), 236), font=font(12), fill=MUTED)
    card(d, (304, by, 706, by + bh), "LONG-LIE LADDER (configurable)")
    x = 318
    for lab, active, rung_level in c["ladder"]:
        f = font(12, True)
        rc = RUNG_COL.get(rung_level, col)
        w = int(d.textlength(lab, font=f)) + 14
        if active:
            d.rounded_rectangle((x, by + 34, x + w, by + 58), 7, fill=rc)
            d.text((x + 7, by + 39), lab, font=f, fill=(255, 255, 255))
        else:
            d.rounded_rectangle((x, by + 34, x + w, by + 58), 7, fill=(34, 42, 54), outline=(80, 92, 108), width=1)
            d.text((x + 7, by + 39), lab, font=f, fill=MUTED)
        x += w + 5
    for i, ln in enumerate(wrap(d, c["ladder_sub"], font(12), 372, 2)):
        d.text((318, by + 64 + i * 15), ln, font=font(12), fill=MUTED)
    card(d, (718, by, 994, by + bh), "COSMOS CHECK (local stand-in)")
    v = c["vlm"]
    d.text((732, by + 30), clip_text(d, v["line1"], font(17, True), 250), font=font(17, True), fill=v["col"])
    for i, ln in enumerate(v["lines"][:2]):
        d.text((732, by + 58 + i * 16), clip_text(d, ln, font(12), 250), font=font(12), fill=MUTED)
    card(d, (1006, by, W - 30, by + bh), "BOX-ONLY BASELINE")
    b = c["boxonly"]
    d.text((1020, by + 30), clip_text(d, b["line1"], font(17, True), 214), font=font(17, True), fill=b["col"])
    for i, ln in enumerate(wrap(d, b["text"], font(12), 214, 3)):
        d.text((1020, by + 56 + i * 15), ln, font=font(12), fill=MUTED)
    d.text((W - 30, H - 8), c["credit"], font=font(11), fill=(100, 112, 126), anchor="rd")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    return path


# ------------------------------------------------------------------------------------------- SIMULATED clock
def clock_txt(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def clock_filmstrip(base: np.ndarray, observed_s: float, path: Path, title: str, steps=(60, 300, 3600)) -> Path:
    """The long-lie clock as staff would see it. Card 1 is the observed time (real); the others advance the clock with
    posture.clock_state and are SIMULATED (the footage is a few seconds long)."""
    W, Hh = 1280, 400
    im = Image.new("RGB", (W, Hh), BG)
    d = ImageDraw.Draw(im)
    d.text((30, 14), "LONG-LIE CLOCK", font=font(26, True), fill=INK)
    d.text((250, 22), title, font=font(16), fill=MUTED)
    d.text((W - 30, 22), "only card 1 is measured; cards 2-4 are SIMULATED (clock logic, footage is too short)", font=font(14, True), fill=(240, 160, 30), anchor="ra")
    times = [observed_s, *steps]
    cw, gap = 295, 13
    for k, tsec in enumerate(times):
        x = 30 + k * (cw + gap)
        st = P.clock_state(tsec)
        rc = RUNG_COL[st["level"]]
        d.rounded_rectangle((x, 56, x + cw, 380), 14, fill=CARD)
        th = cv2.resize(base, (cw - 20, int((cw - 20) * 9 / 16)), interpolation=cv2.INTER_AREA)
        if k > 0:
            th = (th.astype(np.float32) * 0.55).astype(np.uint8)
        im.paste(pil_from_bgr(th), (x + 10, 66))
        ty = 66 + th.shape[0]
        real = k == 0
        d.rounded_rectangle((x + 16, 72, x + 16 + (112 if real else 108), 96), 6, fill=(31, 181, 160) if real else (240, 160, 30))
        d.text((x + 24, 76), "OBSERVED" if real else "SIMULATED", font=font(14, True), fill=(10, 14, 18))
        d.text((x + cw / 2, ty + 48), (f"{tsec:.1f} s" if real else clock_txt(tsec)), font=font(46, True), fill=rc, anchor="mm")
        d.text((x + cw / 2, ty + 86), "on the floor", font=font(14), fill=MUTED, anchor="mm")
        lab = st["rung"]
        lw = int(d.textlength(lab, font=font(20, True))) + 28
        d.rounded_rectangle((x + cw / 2 - lw / 2, ty + 102, x + cw / 2 + lw / 2, ty + 134), 9, fill=rc)
        d.text((x + cw / 2, ty + 118), lab, font=font(20, True), fill=(255, 255, 255), anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    return path


def clock_video(base: np.ndarray, observed_s: float, path: Path, seconds: float = 8.0, fps: float = 15.0, end_s: float = 3600.0) -> Path:
    """SIMULATED animation: the clock accelerates (log scale) from the observed time to `end_s`, rung banner changes."""
    W = 960
    img = cv2.resize(base, (W, int(base.shape[0] * W / base.shape[1])), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) * 0.6).astype(np.uint8)
    nfr = int(seconds * fps)
    with pk.Mp4Writer(path, fps) as w:
        for f in range(nfr):
            frac = min(1.0, f / max(nfr - 1, 1))
            s = observed_s * (end_s / max(observed_s, 1.0)) ** frac if observed_s < end_s else observed_s
            st = P.clock_state(s)
            rc = RUNG_COL[st["level"]]
            fr = img.copy()
            h, w_ = fr.shape[:2]
            cv2.rectangle(fr, (0, 0), (w_, 54), bgr(rc), -1)
            cv2.putText(fr, f"{st['rung']}", (16, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(fr, "SIMULATED CLOCK (accelerated)", (w_ - 400, 34), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(fr, f"ON FLOOR  {clock_txt(s)}", (16, h - 24), cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            w.write(fr)
    return path
