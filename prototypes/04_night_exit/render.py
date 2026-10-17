"""Rendering for NIGHT EXIT: hero image (card stack + timeline), evidence frames."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import nightexit as ne

FONT_DIR = Path("C:/Windows/Fonts")
BG, PANEL, PANEL2 = (15, 22, 32), (26, 35, 50), (34, 46, 66)
FG, MUTED = (236, 241, 247), (150, 165, 184)
GREEN, AMBER, TEAL, RED = (53, 196, 124), (255, 176, 32), (63, 208, 201), (240, 84, 84)
SWATCH = {"black": (25, 25, 28), "grey": (128, 128, 132), "white": (240, 240, 240), "red": (214, 48, 49),
          "orange": (240, 140, 40), "yellow": (240, 210, 50), "green": (60, 160, 80), "blue": (50, 100, 210),
          "purple": (140, 70, 180), "pink": (240, 130, 190), "brown": (120, 80, 50), "beige": (214, 190, 150)}


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeuii.ttf" if italic else "segoeui.ttf"
    for n in (name, "arialbd.ttf" if bold else "arial.ttf"):
        try:
            return ImageFont.truetype(str(FONT_DIR / n), size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit_text(d: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> str:
    if d.textlength(text, font=fnt) <= width:
        return text
    while len(text) > 3 and d.textlength(text + "...", font=fnt) > width:
        text = text[:-1]
    return text.rstrip() + "..."


def wrap2(d: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int, lines: int = 2) -> list[str]:
    """Greedy word wrap into at most `lines` lines (the last one is shortened with '...')."""
    out: list[str] = []
    cur = ""
    words = text.split()
    for k, w in enumerate(words):
        t = (cur + " " + w).strip()
        if d.textlength(t, font=fnt) <= width or not cur:
            cur = t
        else:
            out.append(cur)
            cur = w
            if len(out) == lines - 1:
                cur = " ".join(words[k:])
                break
    out.append(cur)
    out[-1] = fit_text(d, out[-1], fnt, width)
    return out[:lines]


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def contain(img: np.ndarray, w: int, h: int) -> Image.Image:
    s = min(w / img.shape[1], h / img.shape[0])
    return to_pil(cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA))


def draw_lines(img: np.ndarray, lines: Sequence[ne.Line], th: int = 3, label: bool = True) -> None:
    for ln in lines:
        cv2.line(img, tuple(int(v) for v in ln.p1), tuple(int(v) for v in ln.p2), (0, 230, 255), th, cv2.LINE_AA)


def evidence_frame(img: np.ndarray, lines: Sequence[ne.Line], box: Sequence[float] | None, ev: Any, title: str) -> np.ndarray:
    im = img.copy()
    draw_lines(im, lines, 3)
    col = (60, 176, 255) if ev.dir == "OUT" else (201, 208, 63)
    if box is not None:
        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, 3)
    cv2.circle(im, (int(ev.x), int(ev.y)), 9, (0, 0, 255), -1)
    cv2.circle(im, (int(ev.x), int(ev.y)), 11, (255, 255, 255), 2)
    s = max(0.6, im.shape[1] / 1600)
    (tw, th_), b = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
    cv2.rectangle(im, (0, 0), (tw + 16, th_ + b + 14), (0, 0, 0), -1)
    cv2.putText(im, title, (8, th_ + 6), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def evidence_thumb(img: np.ndarray, lines: Sequence[ne.Line], box: Sequence[float] | None, ev: Any) -> np.ndarray:
    """Zoomed evidence for a card: the frame at the crossing, window centred on the crossing point (about 42 % of the frame width),
    exit line yellow, person boxed (amber = OUT, teal = IN), red dot = the point where the line was crossed."""
    im = img.copy()
    H, W = im.shape[:2]
    th = max(3, W // 260)
    draw_lines(im, lines, th)
    col = (60, 176, 255) if ev["dir"] == "OUT" else (201, 208, 63)
    if box is not None:
        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), col, th)
    cv2.circle(im, (int(ev["x"]), int(ev["y"])), th * 3, (0, 0, 255), -1)
    cv2.circle(im, (int(ev["x"]), int(ev["y"])), th * 3 + 2, (255, 255, 255), 2)
    ww = int(W * 0.42)
    wh = int(ww * 9 / 16)
    if box is not None:                       # make sure the whole person fits when he/she is tall
        wh = max(wh, int(min(H, (box[3] - box[1]) * 1.25)))
        ww = max(ww, int(wh * 16 / 9))
    ww, wh = min(ww, W), min(wh, H)
    x0 = int(min(max(ev["x"] - ww / 2, 0), W - ww))
    y0 = int(min(max(ev["y"] - wh / 2, 0), H - wh))
    return im[y0:y0 + wh, x0:x0 + ww]


def swatches(d: ImageDraw.ImageDraw, x: int, y: int, label: str, hist: dict[str, float], fnt: ImageFont.FreeTypeFont,
             n: int = 2) -> int:
    d.text((x, y), label, font=fnt, fill=MUTED)
    x += int(d.textlength(label, font=fnt)) + 6
    for name, share in list(hist.items())[:n]:
        d.rounded_rectangle((x, y + 3, x + 16, y + 19), 4, fill=SWATCH.get(name, (90, 90, 90)), outline=(200, 205, 215))
        t = f"{name} {int(round(share * 100))}%"
        d.text((x + 21, y), t, font=fnt, fill=FG)
        x += 21 + int(d.textlength(t, font=fnt)) + 12
    return x


def hero(path: Path, *, name: str, title: str, since: float, query: str, cards: list[dict[str, Any]],
         all_events: Any, duration: float, blind: Sequence[tuple[float, float]], lines: Sequence[ne.Line],
         ref_frame: np.ndarray | None, focus_box: Sequence[float] | None, n_before: int) -> Path:
    W, H = 1280, 780
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # header
    d.rectangle((0, 0, W, 84), fill=(10, 15, 23))
    d.ellipse((22, 22, 62, 62), fill=(245, 214, 120))
    d.ellipse((34, 18, 70, 56), fill=(10, 15, 23))
    d.text((84, 14), "NIGHT EXIT", font=font(38, True), fill=FG)
    d.text((86, 56), "who went through the exit since...", font=font(15), fill=MUTED)
    q = f"since {ne.clock(since)}" + (f'   |   "{query}"' if query.strip() else "")
    qf = font(26, True)
    qw = int(d.textlength(q, font=qf))
    d.rounded_rectangle((W - qw - 70, 18, W - 22, 66), 24, fill=PANEL2, outline=(70, 90, 120))
    d.text((W - qw - 46, 24), q, font=qf, fill=FG)
    d.text((W - qw - 70 - int(d.textlength(title, font=font(14))) - 14, 34), title, font=font(14), fill=MUTED)

    # left: reference frame at the best match
    x0, y0, pw = 16, 100, 500
    fw = pw - 20
    fh = int(fw * (ref_frame.shape[0] / ref_frame.shape[1])) if ref_frame is not None else 340
    ph = 10 + fh + 14 + 76 + 12 + 112
    d.rounded_rectangle((x0, y0, x0 + pw, y0 + ph), 14, fill=PANEL)
    if ref_frame is not None:
        fr = ref_frame.copy()
        draw_lines(fr, lines, max(3, fr.shape[1] // 300))
        if focus_box is not None:
            bc = (60, 176, 255) if cards and cards[0]["primary"]["dir"] == "OUT" else (201, 208, 63)
            cv2.rectangle(fr, (int(focus_box[0]), int(focus_box[1])), (int(focus_box[2]), int(focus_box[3])), bc,
                          max(3, fr.shape[1] // 280))
        pil = contain(fr, fw, fh)
        img.paste(pil, (x0 + 10, y0 + 10))
        d.rounded_rectangle((x0 + 20, y0 + 20, x0 + 140, y0 + 46), 8, fill=(0, 0, 0))
        d.text((x0 + 30, y0 + 21), "EXIT LINE", font=font(15, True), fill=(0, 230, 255))
        if cards:
            pp = cards[0]["primary"]
            lab = f"{pp['dir']} {ne.clock(pp['t'])}"
            d.rounded_rectangle((x0 + fw - 150, y0 + 20, x0 + fw, y0 + 46), 8, fill=(0, 0, 0))
            d.text((x0 + fw - 140, y0 + 21), "frame at " + ne.clock(pp["t"]), font=font(15, True), fill=FG)
    ly = y0 + 10 + fh + 14
    if cards:
        top = cards[0]
        p = top["primary"]
        verdict = f"WENT {p['dir']} at {ne.clock(p['t'])}" if query.strip() else f"{len(cards)} person(s) crossed since {ne.clock(since)}"
        if query.strip() and top.get("returned") is not None:
            verdict = f"WENT OUT {ne.clock(p['t'])}, BACK IN {top['returned']:.0f}s later"
        vc = AMBER if p["dir"] == "OUT" else TEAL
        d.rounded_rectangle((x0 + 12, ly, x0 + pw - 12, ly + 76), 12, fill=(18, 40, 32) if query.strip() else PANEL2,
                            outline=GREEN if query.strip() else (70, 90, 120), width=3)
        vsz = 34
        while vsz > 18 and d.textlength(verdict, font=font(vsz, True)) > pw - 70:
            vsz -= 2
        d.text((x0 + 26, ly + 6), verdict, font=font(vsz, True), fill=vc if query.strip() else FG)
        n_close = cards[0].get("n_close", 1) if query.strip() else 1
        sub = (("best match" if n_close <= 1 and len(cards) > 1 else "only candidate in this window" if len(cards) == 1 else f"best of {n_close} close matches, not a sure thing")
               if query.strip() else "time order")
        d.text((x0 + 26, ly + 52), sub, font=font(15, True if n_close > 1 else False), fill=AMBER if n_close > 1 else MUTED)
    else:
        d.rounded_rectangle((x0 + 12, ly, x0 + pw - 12, ly + 76), 12, fill=PANEL2, outline=(70, 90, 120), width=3)
        d.text((x0 + 30, ly + 10), f"NO CROSSING since {ne.clock(since)}", font=font(32, True), fill=MUTED)
    ny = ly + 76 + 12
    notes = ["Clock = video time (00:00 = start of the clip). Yellow = exit line, box = the person crossing.",
             "Rank = torso/leg colour + Cosmos one-liner + Cosmos-Embed1 text match."
             if query.strip() else "No description given: crossings listed in time order.",
             f"{len(all_events)} crossing(s) in the clip, {n_before} before {ne.clock(since)} (ignored)."]
    yy = ny
    for ln_ in notes:
        for w_ in wrap2(d, ln_, font(13), pw - 36, 2):
            d.text((x0 + 18, yy), w_, font=font(13), fill=MUTED)
            yy += 18
        yy += 4

    # right: card stack
    cx, cy, cw, ch, gap = 532, 100, 732, 124, 8
    for k in range(4):
        y = cy + k * (ch + gap)
        if k >= len(cards):
            if k == len(cards):
                d.rounded_rectangle((cx, y, cx + cw, y + ch), 14, fill=PANEL)
                if k == 0:
                    msg = ["Nobody went through the exit in this window."]
                else:
                    msg = [f"Only {k} person(s) crossed since {ne.clock(since)}."]
                if blind:
                    msg.append(f"The camera moves from {ne.clock(blind[0][0])}: the exit line is switched off")
                    msg.append("there (brown on the timeline), so later crossings cannot be seen.")
                else:
                    msg.append("That is everyone the line saw in this window.")
                for j, m in enumerate(msg):
                    d.text((cx + 24, y + 28 + j * 26), fit_text(d, m, font(18 if j else 20, bold=(j == 0)), cw - 48), font=font(18 if j else 20, bold=(j == 0)), fill=MUTED)
            continue
        c = cards[k]
        best = k == 0 and bool(query.strip())
        d.rounded_rectangle((cx, y, cx + cw, y + ch), 14, fill=PANEL2 if best else PANEL, outline=GREEN if best else (45, 58, 80),
                            width=3 if best else 1)
        thumb = contain(c["crop_bgr"], 78, ch - 16) if c.get("crop_bgr") is not None else None
        if thumb is not None:
            img.paste(thumb, (cx + 14, y + 8 + (ch - 16 - thumb.height) // 2))
        evt = contain(c["ev_bgr"], 160, ch - 16) if c.get("ev_bgr") is not None else None
        if evt is not None:           # the frame where the line was crossed: line yellow, person boxed, red dot = crossing
            ex, ey = cx + 98, y + 8 + (ch - 16 - evt.height) // 2
            img.paste(evt, (ex, ey))
            d.rectangle((ex - 1, ey - 1, ex + evt.width, ey + evt.height), outline=(90, 110, 140))
            d.rounded_rectangle((ex + 3, ey + 3, ex + 74, ey + 19), 5, fill=(0, 0, 0))
            d.text((ex + 8, ey + 3), "crossing", font=font(12, True), fill=(0, 230, 255))
        d.ellipse((cx + 8, y + 6, cx + 36, y + 34), fill=GREEN if best else (70, 90, 120))
        d.text((cx + 22, y + 8), str(k + 1), font=font(17, True), fill=BG if best else FG, anchor="ma")
        p = c["primary"]
        tx = cx + 270
        col = AMBER if p["dir"] == "OUT" else TEAL
        d.text((tx, y + 6), f"{p['dir']}", font=font(30, True), fill=col)
        d.text((tx + 78, y + 6), ne.clock(p["t"]), font=font(30, True), fill=FG)
        extra = f"via {p['line']}"
        if p.get("inferred"):
            extra += " (interpolated)"
        if c.get("returned") is not None:
            extra += f" | back in {c['returned']:.0f}s later"
        others = [e for e in c["events"] if e is not p]
        if others:
            extra += "  | also " + ", ".join(f"{e['dir']} {ne.clock(e['t'])}" for e in others[:2])
        d.text((tx, y + 42), fit_text(d, extra, font(14), cw - 270 - 120), font=font(14), fill=MUTED)
        swatches(d, tx, y + 62, "top", c["top"], font(14))
        swatches(d, tx + 215, y + 62, "legs", c["legs"], font(14), n=1)
        desc = c["desc"] or "(no description)"
        for j, ln_ in enumerate(wrap2(d, f'"{desc}"', font(13, italic=True), cw - 270 - 112)):
            d.text((tx, y + 84 + j * 17), ln_, font=font(13, italic=True), fill=(190, 202, 220))
        if query.strip() and c.get("score") is not None:
            bw = 84
            bx = cx + cw - bw - 14
            close = c.get("n_close", 1) > 1 and c.get("gap", 1) <= 0.10
            d.text((bx + bw, y + 16), f"{int(round(c['score'] * 100))}%", font=font(26, True), fill=GREEN if best else FG, anchor="ra")
            d.rounded_rectangle((bx, y + 50, bx + bw, y + 58), 4, fill=(50, 62, 84))
            d.rounded_rectangle((bx, y + 50, bx + max(6, int(bw * c["score"])), y + 58), 4, fill=GREEN if best else (110, 135, 175))
            d.text((bx + bw, y + 62), "rank score", font=font(11), fill=MUTED, anchor="ra")
            if c.get("abs") is not None:
                d.text((bx + bw, y + 78), f"evidence {int(round(c['abs'] * 100))}%", font=font(13, True), fill=FG, anchor="ra")
                d.text((bx + bw, y + 94), "colour + text", font=font(11), fill=MUTED, anchor="ra")
            if close:
                d.text((bx + bw, y + 106), "close match", font=font(12, True), fill=AMBER, anchor="ra")
        if best:
            n_close = c.get("n_close", 1)
            lab, bcol, bw_ = ("BEST MATCH", GREEN, 122) if n_close <= 1 else (f"1 OF {n_close} CLOSE", AMBER, 130)
            if len(cards) == 1:
                lab, bcol, bw_ = "ONLY CANDIDATE", (110, 135, 175), 140
            d.rounded_rectangle((cx + cw - bw_ - 18, y - 1, cx + cw - 18, y + 16), 6, fill=bcol)
            d.text((cx + cw - 18 - bw_ / 2, y - 2), lab, font=font(13, True), fill=BG, anchor="ma")

    # timeline
    ty0, ty1 = 668, 770
    d.rounded_rectangle((16, ty0 - 26, W - 16, ty1 + 4), 14, fill=PANEL)
    d.text((32, ty0 - 22), "TIMELINE (video time = the clock)   up = OUT, down = IN", font=font(13, True), fill=MUTED)
    lx0, lx1 = 40, W - 40
    sx = (lx1 - lx0) / max(duration, 1e-6)
    ax = ty0 + 46
    d.rectangle((lx0, ax - 4, lx1, ax + 4), fill=(60, 74, 98))
    tx_since = lx0 + since * sx
    if since > 0:
        d.rectangle((lx0, ax - 14, tx_since, ax + 14), fill=(44, 52, 68))
        if tx_since - lx0 > 100:
            d.text((lx0 + 6, ax - 32), "before: ignored", font=font(12), fill=MUTED)
    for s_, e_ in blind:
        d.rectangle((lx0 + s_ * sx, ax - 14, lx0 + e_ * sx, ax + 14), fill=(86, 70, 40))
    if blind:
        d.text((lx1 - 6, ax + 16), "camera moving: exit line not valid", font=font(12), fill=(200, 170, 110), anchor="ra")
    nt = 8
    for k in range(nt + 1):
        x = lx0 + k * (lx1 - lx0) / nt
        d.line((x, ax + 4, x, ax + 10), fill=MUTED)
        d.text((x, ax + 30 if False else ax + 32), ne.clock(k * duration / nt), font=font(12), fill=MUTED, anchor="ma")
    rank_of = {}
    for k, c in enumerate(cards):
        for e in c["events"]:
            rank_of[(e["pid"], e["t"])] = k + 1
    for e in all_events.itertuples():
        x = lx0 + e.t * sx
        inwin = e.t >= since
        col = (AMBER if e.dir == "OUT" else TEAL) if inwin else (100, 108, 122)
        up = e.dir == "IN"
        pts = [(x, ax - 8 - 12), (x - 7, ax - 8), (x + 7, ax - 8)] if not up else [(x, ax - 8), (x - 7, ax - 8 - 12), (x + 7, ax - 8 - 12)]
        d.polygon(pts, fill=col)
        r_ = rank_of.get((e.pid, e.t))
        if r_ == 1 and query.strip():
            d.ellipse((x - 11, ax - 52, x + 11, ax - 30), fill=GREEN)
            d.text((x, ax - 52), "1", font=font(15, True), fill=BG, anchor="ma")
    d.line((tx_since, ax - 36, tx_since, ax + 28), fill=RED, width=3)
    d.text((tx_since + 6, ax + 12), f"since {ne.clock(since)}", font=font(13, True), fill=RED)
    d.text((W - 32, ty0 - 22), f"{n_before} crossing(s) before, {len(cards)} person(s) since", font=font(13), fill=MUTED, anchor="ra")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)
    return path
