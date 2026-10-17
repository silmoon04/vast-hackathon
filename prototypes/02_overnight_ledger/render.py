"""Hero image + strip rendering for the Overnight Ledger (PIL for real typography, OpenCV arrays in/out)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path("C:/Windows/Fonts")
INK, MUTED, BG, CARD, LINE = (24, 32, 44), (112, 124, 138), (244, 246, 249), (255, 255, 255), (222, 227, 234)
GREEN, RED, GREY, AMBER, BLUE = (46, 158, 91), (214, 69, 69), (138, 148, 160), (224, 161, 0), (47, 111, 222)
CAUSE = {"covered": "lens covered", "frozen": "feed frozen", "blurred": "out of focus", "view": "camera looking elsewhere",
         "occluded": "spot blocked", "unsure": "unclear"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for n in (("arialbd.ttf" if bold else "arial.ttf"), ("calibrib.ttf" if bold else "calibri.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(str(FONTS / n) if (FONTS / n).exists() else n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def paste_fit(im: Image.Image, bgr: np.ndarray, x: int, y: int, w: int, h: int | None = None, border=LINE) -> tuple[int, int]:
    s = w / bgr.shape[1]
    if h is not None:
        s = min(s, h / bgr.shape[0])
    t = to_pil(cv2.resize(bgr, (max(1, int(bgr.shape[1] * s)), max(1, int(bgr.shape[0] * s))), interpolation=cv2.INTER_AREA))
    im.paste(t, (x, y))
    ImageDraw.Draw(im).rectangle([x - 1, y - 1, x + t.width, y + t.height], outline=border, width=1)
    return t.width, t.height


def text_fit(d: ImageDraw.ImageDraw, xy, s: str, size: int, fill, bold=False, maxw: int | None = None):
    f = font(size, bold)
    while maxw and d.textlength(s, font=f) > maxw and size > 11:
        size -= 1
        f = font(size, bold)
    d.text(xy, s, font=f, fill=fill)
    return f


def ellipsize(d: ImageDraw.ImageDraw, s: str, size: int, maxw: int, bold=False) -> str:
    f = font(size, bold)
    if d.textlength(s, font=f) <= maxw:
        return s
    while s and d.textlength(s + "...", font=f) > maxw:
        s = s[:-1]
    return s.rstrip() + "..."


def hatch(d: ImageDraw.ImageDraw, x0, y0, x1, y1, col, step=7):
    for k in range(-(y1 - y0), x1 - x0, step):
        a = (x0 + max(k, 0), y0 + max(-k, 0))
        b = (min(x0 + k + (y1 - y0), x1), y0 + min(y1 - y0, x1 - x0 - k))
        d.line([a, b], fill=col, width=1)


def draw_strip(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, spans: list[tuple[int, int, str, str]], t0: float, t1: float,
               *, gt: list[tuple[float, float, str]] | None = None, axis: bool = True, labels: bool = True) -> None:
    """spans: (s, e, code, cause) -> green/red/amber/grey(hatched) blocks. gt: SIMULATED faults drawn as brackets below."""
    sx = w / max(t1 - t0, 1e-6)
    d.rectangle([x, y, x + w, y + h], fill=(232, 236, 241))
    for s, e, code, cause in spans:
        xa, xb = x + (s - t0) * sx, x + (e - t0) * sx
        col = {"P": GREEN, "A": RED, "D": AMBER, "G": GREY}[code]
        d.rectangle([xa, y, max(xb - 0.5, xa + 1), y + h], fill=col)
        if code == "G" and xb - xa > 6:
            hatch(d, int(xa), y, int(xb), y + h, (210, 216, 223))
    if axis:
        n = 8
        for k in range(n + 1):
            xx = x + k * w / n
            d.line([xx, y + h, xx, y + h + 5], fill=MUTED)
            tt = mmss(t0 + k * (t1 - t0) / n)
            tw = d.textlength(tt, font=font(13))
            d.text((min(max(xx - tw / 2, x), x + w - tw), y + h + 6), tt, font=font(13), fill=MUTED)
    if gt:
        for a, b, kind in gt:
            xa, xb = x + (a - t0) * sx, x + (b - t0) * sx
            yy = y + h + 26
            d.line([xa, yy, xb, yy], fill=BLUE, width=3)
            d.line([xa, yy - 4, xa, yy + 4], fill=BLUE, width=2)
            d.line([xb, yy - 4, xb, yy + 4], fill=BLUE, width=2)
            text_fit(d, (xa, yy + 5), f"SIMULATED {kind} {int(round(b - a))} s", 13, BLUE, True, maxw=max(120, int(xb - xa)))


def rounded(d: ImageDraw.ImageDraw, box, r, fill, outline=None, width=1):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def hero_ledger(ctx: dict[str, Any]) -> np.ndarray:
    """ctx keys: video, subtitle, ref_frame, box, t0, t1, spans, gt, stamp(text,color), lines[list[str]],
    stretches[list of dict(s,e,label,verdict,before,after,cmp)], contacts[list of dict(label,thumb)], second(dict or None),
    sim(bool)."""
    W = 1280
    gt = ctx.get("gt")
    sy = 408
    nxt = sy + 56 + (74 if gt else 46)                     # first y below the strip (axis labels, or SIMULATED brackets)
    sec = ctx.get("second")
    yy2 = nxt + 24
    by = (yy2 + 28 + 34) if sec else nxt + 4                # top of the bottom row
    H = by + 24 + 130 + 48
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 76], fill=(15, 23, 42))
    d.text((28, 16), "OVERNIGHT LEDGER", font=font(34, True), fill=(255, 255, 255))
    sub = ctx["subtitle"]
    f = font(18)
    badge, sim_txt = ctx.get("badge"), "SIMULATED faults injected" if ctx.get("sim") else None
    d.text((W - 28 - d.textlength(sub, font=f), 8 if badge else 30), sub, font=f, fill=(176, 190, 208))
    if badge:
        bf = font(16, True)
        btxt = ellipsize(d, badge, 16, W - 420, True)
        bw_ = d.textlength(btxt, font=bf)
        rounded(d, (W - 28 - bw_ - 24, 36, W - 28, 66), 15, (22, 101, 52))
        d.text((W - 28 - bw_ - 12, 41), btxt, font=bf, fill=(220, 252, 231))
    elif sim_txt:
        d.text((W - 28 - d.textlength(sim_txt, font=font(15, True)), 54), sim_txt, font=font(15, True), fill=(250, 204, 21))
    # reference frame + spot
    ref = ctx["ref_frame"].copy()
    x1, y1, x2, y2 = (int(v) for v in ctx["box"])
    cv2.rectangle(ref, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 0), 8)
    cv2.rectangle(ref, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (60, 200, 90), 5)
    pw, ph = paste_fit(im, ref, 28, 128, 400)
    d.text((28, 98), "THE CAR WE WATCH", font=font(17, True), fill=MUTED)
    # verdict stamp (with the honest scope note inside the card)
    stamp, scol = ctx["stamp"]
    sx0 = 500
    rounded(d, (sx0, 100, W - 28, 214), 14, scol)
    text_fit(d, (sx0 + 26, 106), stamp, 54, (255, 255, 255), True, maxw=W - 28 - sx0 - 52)
    if ctx.get("stamp_note"):
        text_fit(d, (sx0 + 28, 176), ctx["stamp_note"], 18, (255, 255, 255), False, maxw=W - 28 - sx0 - 56)
    yy = 228
    for i, ln in enumerate(ctx["lines"][:4]):
        key = ln.startswith("!")
        text = ln.lstrip("!")
        text_fit(d, (sx0, yy), text, 24 if i < 2 else 20, INK if key else MUTED, key, maxw=W - 28 - sx0)
        yy += 33 if i < 2 else 28
    # strips
    d.text((28, sy - 30), ctx.get("strip_title", "SECOND BY SECOND:  PRESENT   ABSENT   CAN'T SEE"), font=font(17, True), fill=MUTED)
    lx = 540
    for col, txt, off in ((GREEN, "present", 0), (RED, "absent", 100), (GREY, "can't see", 190), (AMBER, "different car", 290)):
        d.rectangle([lx + off, sy - 26, lx + off + 16, sy - 10], fill=col)
        d.text((lx + off + 22, sy - 29), txt, font=font(15), fill=MUTED)
    draw_strip(d, 28, sy, W - 56, 56, ctx["spans"], ctx["t0"], ctx["t1"], gt=gt)
    # in-strip labels for the longest can't-see stretches
    sxs = (W - 56) / max(ctx["t1"] - ctx["t0"], 1e-6)
    for st in sorted(ctx["stretches"], key=lambda q: -(q["e"] - q["s"]))[:3]:
        xa = 28 + (st["s"] - ctx["t0"]) * sxs
        xb = 28 + (st["e"] - ctx["t0"]) * sxs
        wpx = xb - xa - 10
        if wpx < 60 or not st.get("label"):
            continue
        size = 20
        while size > 11 and d.textlength(st["label"], font=font(size, True)) > wpx:
            size -= 1
        if d.textlength(st["label"], font=font(size, True)) > wpx:
            continue
        tw = d.textlength(st["label"], font=font(size, True))
        cx = (xa + xb) / 2
        rounded(d, (cx - tw / 2 - 8, sy + 14, cx + tw / 2 + 8, sy + 42), 6, (40, 48, 60))
        d.text((cx - tw / 2, sy + 15), st["label"], font=font(size, True), fill=(255, 255, 255))
    # second car
    if sec:
        d.text((28, yy2 - 24), sec["title"], font=font(16, True), fill=MUTED)
        draw_strip(d, 28, yy2, W - 56, 28, sec["spans"], ctx["t0"], ctx["t1"], axis=False)
    # bottom row: before/after and contacts
    bx = 28
    st = ctx.get("focus")
    if st:
        d.text((bx, by - 2), "BLIND STRETCH: BEFORE vs AFTER", font=font(16, True), fill=MUTED)
        wc = 150
        w1, h1 = paste_fit(im, st["before"], bx, by + 24, wc, 130)
        w2, h2 = paste_fit(im, st["after"], bx + wc + 40, by + 24, wc, 130)
        d.text((bx, by + 24 + max(h1, h2) + 4), f"before {mmss(st['tb'])}", font=font(14), fill=MUTED)
        d.text((bx + wc + 40, by + 24 + max(h1, h2) + 4), f"after {mmss(st['ta'])}", font=font(14), fill=MUTED)
        d.text((bx + wc - 6 + 10, by + 70), "→", font=font(30, True), fill=MUTED)
        col = GREEN if st["verdict"] == "same car" else (RED if st["verdict"].startswith(("CHANGED", "CAR GONE")) else AMBER)
        vtxt = st["verdict"].split(":")[0] if st["verdict"].startswith("CAR GONE") else st["verdict"]
        rounded(d, (bx + 2 * wc + 70, by + 28, bx + 2 * wc + 70 + 200, by + 66), 8, col)
        text_fit(d, (bx + 2 * wc + 82, by + 34), vtxt.upper(), 22, (255, 255, 255), True, maxw=176)
        ny = by + 76
        if st.get("metric"):
            d.text((bx + 2 * wc + 70, ny), st["metric"], font=font(15), fill=MUTED)
            ny += 24
        note, hot = ctx.get("damage_note") or (None, False)
        if note:
            fnt = font(14, hot)
            words, cur, rows = note.split(), "", []
            for wd in words:
                if d.textlength((cur + " " + wd).strip(), font=fnt) > 232:
                    rows.append(cur)
                    cur = wd
                else:
                    cur = (cur + " " + wd).strip()
            rows.append(cur)
            hh = 8 + 18 * len(rows)
            rounded(d, (bx + 2 * wc + 66, ny, bx + 2 * wc + 66 + 244, ny + hh), 6, (255, 247, 224) if hot else (240, 243, 247),
                    outline=AMBER if hot else LINE, width=2 if hot else 1)
            for i, r_ in enumerate(rows):
                d.text((bx + 2 * wc + 74, ny + 4 + 18 * i), r_, font=fnt, fill=(150, 90, 0) if hot else MUTED)
    cx0 = 660
    d.text((cx0, by - 2), "PEOPLE WHO CAME CLOSE (feet in the zone)", font=font(16, True), fill=MUTED)
    if ctx["contacts"]:
        x = cx0
        for c in ctx["contacts"][:3]:
            w1, h1 = paste_fit(im, c["thumb"], x, by + 24, 190, 110)
            text_fit(d, (x, by + 24 + h1 + 4), c["label"], 14, MUTED, maxw=190)
            if c.get("touch"):
                tcol = GREEN if c["touch"].startswith("car same") else (RED if "CHANGED" in c["touch"] else MUTED)
                text_fit(d, (x, by + 24 + h1 + 22), "TOUCH? " + c["touch"], 14, tcol, True, maxw=190)
            x += 200
    else:
        d.text((cx0, by + 40), "nobody came near the car", font=font(22, True), fill=GREEN)
    foot = ellipsize(d, ctx.get("footer", ""), 13, W - 56)
    d.text((28, H - 22), foot, font=font(13), fill=MUTED)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def hero_nocar(ctx: dict[str, Any]) -> np.ndarray:
    """Honest 'no ledger possible' hero: contact sheet + reason."""
    W, H = 1280, 720
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 76], fill=(15, 23, 42))
    d.text((28, 16), "OVERNIGHT LEDGER", font=font(34, True), fill=(255, 255, 255))
    sub = ctx["subtitle"]
    d.text((W - 28 - d.textlength(sub, font=font(18)), 30), sub, font=font(18), fill=(176, 190, 208))
    paste_fit(im, ctx["sheet"], 28, 100, 760, 520)
    rounded(d, (820, 100, W - 28, 190), 14, GREY)
    text_fit(d, (844, 112), ctx["stamp"], 40, (255, 255, 255), True, maxw=W - 28 - 820 - 48)
    yy = 214
    for ln in ctx["lines"][:8]:
        key = ln.startswith("!")
        f = font(22 if key else 19, key)
        words, cur, rows = ln.lstrip("!").split(), "", []
        for wd in words:
            if d.textlength((cur + " " + wd).strip(), font=f) > W - 28 - 820:
                rows.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        rows.append(cur)
        for r_ in rows:
            d.text((820, yy), r_, font=f, fill=INK if key else MUTED)
            yy += 30 if key else 26
        yy += 12
    d.text((28, H - 30), ctx.get("footer", ""), font=font(14), fill=MUTED)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def strip_png(spans, t0, t1, gt=None, w=1240, title: str = "") -> np.ndarray:
    im = Image.new("RGB", (w + 40, 120 if gt else 96), CARD)
    d = ImageDraw.Draw(im)
    if title:
        d.text((20, 4), title, font=font(14, True), fill=INK)
    draw_strip(d, 20, 28, w, 34, spans, t0, t1, gt=gt)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
