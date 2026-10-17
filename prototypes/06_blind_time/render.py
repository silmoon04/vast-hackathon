"""Hero image + report pieces for BLIND TIME (PIL for type, OpenCV for frames)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import blind as bl
import protokit as pk

W, H = 1280, 720
BG, CARD, FG, MUTED, LINE = (14, 19, 25), (23, 30, 39), (234, 238, 242), (148, 160, 173), (46, 56, 68)
VERDICT_RGB = {"SEEN": (52, 178, 104), "NOTHING FOUND": (122, 138, 156), "COULD NOT SEE": (240, 120, 52),
               "SEEN IN OTHER VIEW": (58, 150, 214)}
STATE_RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in bl.STATE_HEX.items()}
FONT_DIR = Path("C:/Windows/Fonts")


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for f in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        try:
            return ImageFont.truetype(str(FONT_DIR / f), size)
        except OSError:
            continue
    return ImageFont.load_default()


class Cv:
    def __init__(self, w: int = W, h: int = H, bg=BG):
        self.im = Image.new("RGB", (w, h), bg)
        self.d = ImageDraw.Draw(self.im, "RGBA")

    def text(self, xy, s, size=18, color=FG, bold=False, anchor="la") -> int:
        self.d.text(xy, s, font=font(size, bold), fill=color, anchor=anchor)
        return int(self.d.textlength(s, font=font(size, bold)))

    def rr(self, box, fill=None, outline=None, r=10, width=2) -> None:
        self.d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

    def chip(self, xy, s, fill, size=15, fg=(255, 255, 255), bold=True, pad=(10, 4), outline=None) -> tuple[int, int]:
        f = font(size, bold)
        tw = int(self.d.textlength(s, font=f))
        x, y = xy
        h = size + 2 * pad[1] + 4
        self.rr((x, y, x + tw + 2 * pad[0], y + h), fill=fill, outline=outline, r=h // 2, width=2)
        self.d.text((x + pad[0], y + pad[1] + 1), s, font=f, fill=fg)
        return tw + 2 * pad[0], h

    def paste_bgr(self, img: np.ndarray, xy, w: int, border=None, bw=5) -> tuple[int, int]:
        h = int(img.shape[0] * w / img.shape[1])
        im = Image.fromarray(cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        self.im.paste(im, xy)
        if border:
            self.d.rectangle((xy[0], xy[1], xy[0] + w, xy[1] + h), outline=border, width=bw)
        return w, h

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.im.save(path, quality=92)
        return path


def _span_text(sp: dict[str, Any]) -> str:
    return f"{pk.mmss(sp['t0'])[:5]}-{pk.mmss(sp['t1'])[:5]}"


def draw_timeline(c: Cv, x: int, y: int, w: int, h: int, states: Sequence[str], duration: float,
                  window: tuple[float, float] | None = None, labels: bool = True) -> None:
    n = len(states)
    sx = w / max(duration, 1.0)
    c.d.rectangle((x - 1, y - 1, x + w + 1, y + h + 1), outline=(80, 92, 106), width=1)
    for a, b, st in bl.spans(list(states)):
        x1, x2 = x + a * sx, x + min(b, duration) * sx
        c.d.rectangle((x1, y, max(x2, x1 + 1), y + h), fill=STATE_RGB[st])
        if labels and st != "OK" and (x2 - x1) > 62:
            c.d.text(((x1 + x2) / 2, y + h / 2), st, font=font(14, True), fill=(255, 255, 255), anchor="mm")
    step = 10 if duration > 90 else 5
    if duration > 400:
        step = 60
    elif duration > 200:
        step = 30
    k = 0
    while k <= duration:
        xx = x + k * sx
        c.d.line((xx, y + h, xx, y + h + 5), fill=MUTED, width=1)
        c.text((xx, y + h + 6), pk.mmss(k)[:5].lstrip("0") if k >= 60 else f"0:{int(k):02d}" if k < 60 else "", 12, MUTED, anchor="ma")
        k += step
    if window:
        x1, x2 = x + window[0] * sx, x + min(window[1], duration) * sx
        c.d.rectangle((x1, y - 10, x2, y - 6), fill=(255, 255, 255))
        c.d.rectangle((x1 - 1, y - 12, x1 + 1, y + h + 1), fill=(255, 255, 255))
        c.d.rectangle((x2 - 1, y - 12, x2 + 1, y + h + 1), fill=(255, 255, 255))


def legend(c: Cv, x: int, y: int, present: Sequence[str]) -> None:
    for st in bl.STATES:
        if st not in present and st != "OK":
            continue
        c.d.rectangle((x, y + 3, x + 14, y + 17), fill=STATE_RGB[st], outline=(80, 92, 106))
        x += 22 + c.text((x + 22, y), st, 14, FG, True) + 22


def stamp(c: Cv, xy, text: str, rgb, size=56) -> tuple[int, int]:
    f = font(size, True)
    tw = int(c.d.textlength(text, font=f))
    x, y = xy
    pad = 20
    h = size + 28
    c.rr((x, y, x + tw + 2 * pad, y + h), fill=rgb + (38,), outline=rgb, r=14, width=5)
    c.d.text((x + pad, y + 8), text, font=f, fill=rgb)
    return tw + 2 * pad, h


def annotate(img: np.ndarray, dets: list[dict[str, Any]] | None = None, cls: str | None = None, state: str | None = None,
             caption: str | None = None) -> np.ndarray:
    im = img.copy()
    for d in dets or []:
        if cls and d["name"] != cls:
            continue
        pk.box(im, d["box"], (60, 220, 120), f"{d['name']} {d['conf']:.2f}", 3)
    if state:
        rgb = STATE_RGB[state][::-1]
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), rgb, max(6, im.shape[1] // 120))
    if caption:
        pk.label(im, caption, (14, 14), scale=max(0.6, im.shape[1] / 1100), bg=STATE_RGB[state][::-1] if state else (0, 0, 0))
    return im


def hero(path: Path, *, video: str, duration: float, states: Sequence[str], head: dict[str, Any], also: list[dict[str, Any]],
         thumbs: list[tuple[np.ndarray, str, tuple[int, int, int]]], tiles: list[tuple[str, str]], subtitle: str,
         badge: str = "", footnote: str = "") -> Path:
    c = Cv()
    # header
    c.rr((0, 0, W, 62), fill=CARD, r=0)
    bw = c.text((24, 14), "BLIND TIME", 30, FG, True)
    c.text((24 + bw + 14, 24), "camera health vs reference view", 17, MUTED)
    c.text((W - 24, 12), video, 22, FG, True, "ra")
    c.text((W - 24, 38), subtitle, 14, MUTED, False, "ra")
    # query card
    c.rr((20, 78, 716, 424), fill=CARD, outline=LINE, r=14, width=1)
    c.text((44, 92), "YOU ASKED", 14, MUTED, True)
    qx = c.text((44, 110), head["query"].replace("-", " - "), 36, FG, True)
    if badge:
        c.chip((44 + qx + 16, 118), badge, (60, 72, 88), 14)
    verdict = head["verdict"]
    rgb = VERDICT_RGB[verdict]
    c.text((44, 160), "BLIND TIME SAYS", 13, MUTED, True)
    sw, _ = stamp(c, (44, 178), verdict, rgb, 46 if len(verdict) < 14 else 30)
    nx = 44 + sw + 28
    c.text((nx, 160), "TODAY'S SEARCH SAYS", 13, MUTED, True)
    mis = head["naive"] != verdict and verdict != "SEEN IN OTHER VIEW"
    tw = c.text((nx, 184), head["naive"], 21, (170, 180, 192) if mis else FG, True)
    if mis:
        c.d.line((nx, 184 + 15, nx + tw, 184 + 15), fill=(220, 90, 90), width=3)
        c.chip((nx, 214), "STALE FRAME" if head.get("naive_stale") else "MISLEADING", (190, 55, 55), 13)
    if verdict == "COULD NOT SEE":
        lines = [f"Blind for {bl.fmt_dur(head['blind_s'])} of {bl.fmt_dur(head['window_s'])}  |  coverage {head['coverage'] * 100:.0f}%"]
        off = head.get("evidence_off_view") or []
        if off:
            lines.append(f"{len(off)} s of hits in a different field of view (not the reference view)")
    elif verdict == "SEEN IN OTHER VIEW":
        off = head.get("evidence_off_view") or []
        lines = [f"Hits in {len(off)} s, first at {pk.mmss(off[0]['s'])[:5]}, but only while the camera looked elsewhere",
                 f"The reference view itself was not watched: coverage {head['coverage'] * 100:.0f}%"]
    elif verdict == "NOTHING FOUND":
        lines = [f"Camera OK for the whole window ({head['coverage'] * 100:.0f}% coverage):", "the absence of hits is real"]
    else:
        ev = head["evidence"]
        seen_s = max(0, head["window_s"] - head["blind_s"])
        lines = [f"Hit in {len(ev)} s, first at {pk.mmss(ev[0]['s'])[:5]}  |  coverage {head['coverage'] * 100:.0f}%"]
        if head["blind_s"]:
            lines.append(f"Blind for {bl.fmt_dur(head['blind_s'])}: more hits may have been missed (lower bound)")
    y = 258
    for ln in lines:
        c.text((44, y), ln, 18, FG, True)
        y += 26
    y += 6
    for sp in head["blind_spans"][:4]:
        st = sp["state"]
        c.d.rectangle((44, y + 4, 60, y + 20), fill=STATE_RGB[st])
        wtxt = c.text((70, y), f"{st}  {_span_text(sp)}  ({bl.fmt_dur(sp['t1'] - sp['t0'])})", 16, FG, True)
        c.text((70 + wtxt + 12, y + 1), bl.STATE_WHY[st], 14, MUTED)
        y += 26
    if not head["blind_spans"]:
        c.text((44, y), "No blind sub-spans in this window.", 16, MUTED)
    c.text((44, 398), "A missing hit only proves absence while the camera could see.", 14, MUTED)
    # thumbs
    tx = 740
    tw_ = 252
    for k, (img, cap, col) in enumerate(thumbs[:2]):
        x = tx + k * (tw_ + 12)
        w_, h_ = c.paste_bgr(img, (x, 78), tw_, border=col, bw=5)
        c.chip((x + 8, 78 + h_ - 34), cap, (0, 0, 0, 190), 13, pad=(8, 3))
    # tiles
    for k, (big, small) in enumerate(tiles[:3]):
        x = tx + k * 174
        c.rr((x, 262, x + 166, 424), fill=CARD, outline=LINE, r=12, width=1)
        c.text((x + 83, 296), big, 34 if len(big) < 7 else 28, FG, True, "ma")
        for j, ln in enumerate(small.split("\n")):
            c.text((x + 83, 346 + j * 20), ln, 14, MUTED, False, "ma")
    # also asked
    c.text((24, 438), "ALSO ASKED", 13, MUTED, True)
    y = 458
    for a in also[:3]:
        c.rr((20, y, W - 20, y + 36), fill=CARD, outline=LINE, r=10, width=1)
        c.text((36, y + 5), a["query"].replace("-", " - "), 19, FG, True)
        x = 330
        mis_ = a["naive"] != a["verdict"]
        c.chip((x, y + 4), a["naive"], (70, 82, 96), 14, fg=(190, 200, 210))
        c.text((x + 150, y + 5), ">", 20, MUTED, True)
        c.chip((x + 176, y + 4), a["verdict"], VERDICT_RGB[a["verdict"]], 14)
        x += 70 if a["verdict"] == "SEEN IN OTHER VIEW" else 0
        c.text((x + 340, y + 8), f"coverage {a['coverage'] * 100:.0f}%", 16, FG, True)
        if a["blind_spans"]:
            top = max(a["blind_spans"], key=lambda s: s["t1"] - s["t0"])
            c.text((x + 470, y + 8), f"{top['state']} {_span_text(top)}", 16, STATE_RGB[top["state"]] if top["state"] != "DARK" else (200, 205, 212), True)
        if mis_ and a["naive"] == "NOTHING FOUND":
            c.chip((W - 150, y + 4), "MISLEADING", (190, 55, 55), 13)
        elif a.get("naive_stale"):
            c.chip((W - 190, y + 4), "STALE / BLACK FRAMES", (190, 55, 55), 12)
        elif a["verdict"] == "SEEN" and a["coverage"] < 0.95:
            c.chip((W - 300, y + 4), f"BLIND {bl.fmt_dur(a['blind_s'])}: HITS = LOWER BOUND", (96, 80, 30), 12, fg=(255, 220, 120))
        y += 42
    # timeline
    c.text((24, 590), "CAMERA HEALTH, PER SECOND", 13, MUTED, True)
    draw_timeline(c, 24, 628, W - 48, 36, states, duration, window=(head["t0"], head["t1"]))
    c.text((W - 24, 590), "white bracket = the window you asked about", 13, MUTED, False, "ra")
    legend(c, 24, 690, set(states))
    if footnote:
        c.text((W - 24, 692), footnote, 13, MUTED, False, "ra")
    return c.save(path)
