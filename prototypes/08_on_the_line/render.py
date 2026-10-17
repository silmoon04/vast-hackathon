"""Dashboard rendering (PIL): live frame with boxes, presence timeline, answer cards, incident log, verdict stamp.
One compose() serves the demo mp4 (live, causal) and the hero.jpg (call summary)."""
from __future__ import annotations

# make the module importable on its own (protokit lives in prototypes/)
import sys as _sys  # noqa: E402
from pathlib import Path as _P  # noqa: E402
_HERE_PATHS = [str(_P(__file__).resolve().parents[1]), str(_P(__file__).resolve().parent)]
_sys.path[:0] = [x for x in _HERE_PATHS if x not in _sys.path]

import textwrap
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import protokit as pk
from engine import GROUP_PLURAL, GROUP_WORD, SceneState, display_colour
from qa import Card

W, H = 1280, 720
BG, PANEL, LINE = (13, 20, 32), (22, 32, 49), (45, 60, 85)
INK, MUTED, ACCENT = (232, 238, 246), (140, 156, 178), (74, 163, 255)
GOOD, BAD, WARN = (52, 199, 120), (240, 84, 84), (245, 190, 60)
GROUP_RGB = {"person": (88, 166, 255), "vehicle": (255, 160, 64), "bicycle": (110, 220, 150)}
_F: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_CARD_CACHE: dict[str, Any] = {}


def font(size: int, bold: bool = False, light: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else ("segoeuil.ttf" if light else "segoeui.ttf")
    k = (name, size)
    if k not in _F:
        try:
            _F[k] = ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            _F[k] = ImageFont.load_default()
    return _F[k]


def _wrap(d: ImageDraw.ImageDraw, text: str, f, width: int, max_lines: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=f) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ,") + "..."
    return lines


def _paste_fit(canvas: Image.Image, img_bgr: np.ndarray, box: tuple[int, int, int, int], bg=(8, 12, 20)) -> None:
    x, y, w, h = box
    tile = Image.new("RGB", (w, h), bg)
    ih, iw = img_bgr.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    im = Image.fromarray(cv2.cvtColor(cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC),
                                      cv2.COLOR_BGR2RGB))
    tile.paste(im, ((w - nw) // 2, (h - nh) // 2))
    canvas.paste(tile, (x, y))


def _chip(d: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fg, bg, f, pad: int = 6, anchor_right: bool = False) -> int:
    tw = int(d.textlength(text, font=f))
    x, y = xy
    if anchor_right:
        x -= tw + 2 * pad
    d.rounded_rectangle((x, y, x + tw + 2 * pad, y + f.size + 6), 6, fill=bg)
    d.text((x + pad, y + 2), text, font=f, fill=fg)
    return x + tw + 2 * pad


def _verdict_of(i: int, scored: list[dict[str, Any]] | None) -> bool | None:
    if not scored:
        return None
    for s in scored:
        if s["i"] == i:
            return s["correct"]
    return None


def _tag(d: ImageDraw.ImageDraw, text: str, x: int, y: int, small: bool = False) -> None:
    t = text.replace("best crop ", "").replace("last good view ", "").replace("first good view ", "").replace("who ", "")
    t = t.replace("path, last seen ", "path ")[: (11 if small else 18)]
    d.text((x, y), t, font=font(10 if small else 11), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))


def draw_cards(d: ImageDraw.ImageDraw, canvas: Image.Image, cards: list[Card], x: int, y: int, w: int, *,
               scored: list[dict[str, Any]] | None, slots: int, card_h: int, new_idx: int | None, q_idx: list[int]) -> None:
    for slot in range(slots):
        cy = y + slot * (card_h + 6)
        if slot >= len(cards):
            d.rounded_rectangle((x, cy, x + w, cy + card_h), 10, outline=LINE, width=1, fill=(16, 25, 40))
            d.text((x + 14, cy + card_h // 2 - 10), f"waiting for question {slot + 1}", font=font(15), fill=(85, 100, 122))
            continue
        c = cards[slot]
        v = _verdict_of(q_idx[slot], scored)
        border = ACCENT if new_idx == slot else LINE
        d.rounded_rectangle((x, cy, x + w, cy + card_h), 10, fill=PANEL, outline=border, width=3 if new_idx == slot else 1)
        # evidence thumbnails: 1 tile, 2 side by side, 3-4 in a 2x2 grid
        ths = c.evidence
        tx, ty, tw, th = x + 8, cy + 8, 128, card_h - 16
        if len(ths) == 1:
            _paste_fit(canvas, ths[0].img, (tx, ty, tw, th))
            _tag(d, ths[0].caption, tx + 3, ty + th - 16)
        elif len(ths) == 2:
            hw = tw // 2 - 1
            for k in range(2):
                _paste_fit(canvas, ths[k].img, (tx + k * (hw + 2), ty, hw, th))
                _tag(d, ths[k].caption, tx + k * (hw + 2) + 2, ty + th - 16, small=True)
        elif ths:
            cell = (tw // 2 - 1, th // 2 - 1)
            for k in range(min(len(ths), 4)):
                _paste_fit(canvas, ths[k].img, (tx + (k % 2) * (cell[0] + 2), ty + (k // 2) * (cell[1] + 2), cell[0], cell[1]))
                _tag(d, ths[k].caption, tx + (k % 2) * (cell[0] + 2) + 2, ty + (k // 2) * (cell[1] + 2) + cell[1] - 14, small=True)
        else:
            d.rectangle((tx, ty, tx + tw, ty + th), fill=(8, 12, 20))
        lx = x + 146
        lw = w - 156
        d.text((lx, cy + 6), f"Q{slot + 1}  {pk.mmss(c.t)}", font=font(12, True), fill=MUTED)
        qtxt = _wrap(d, f"\u201c{c.question}\u201d", font(13), lw - 80, 1)[0]
        d.text((lx + 78, cy + 5), qtxt, font=font(13), fill=(170, 184, 204))
        hl = _wrap(d, c.headline, font(19, True), lw, 2)
        for k, line in enumerate(hl):
            d.text((lx, cy + 26 + k * 23), line, font=font(19, True), fill=INK)
        yy = cy + 26 + len(hl) * 23 + 1
        det = c.lines[0] if c.lines else ""
        room = max(0, int((cy + card_h - 26 - yy) // 15))
        if det and room >= 1:
            for k, ln in enumerate(_wrap(d, det, font(12), lw, min(2, room))):
                d.text((lx, yy + k * 15), ln, font=font(12), fill=MUTED)
        # footer chips
        fy = cy + card_h - 24
        _chip(d, (x + w - 8, fy), f"{c.latency_ms:.0f} ms" if c.latency_ms >= 1 else f"{c.latency_ms:.1f} ms",
              (170, 184, 204), (31, 45, 68), font(12), anchor_right=True)
        nx = lx
        if v is not None:
            nx = _chip(d, (lx, fy), "matches eyeball" if v else "WRONG vs eyeball", (6, 24, 14) if v else (255, 255, 255),
                       GOOD if v else BAD, font(12, True)) + 6
        if c.caveats:
            short = c.caveats[0]
            _chip(d, (nx, fy), short[:30], (40, 28, 0), WARN, font(11))


def _timeline(d: ImageDraw.ImageDraw, st: SceneState, x: int, y: int, w: int, t_now: float, t0: float, t1: float,
              qs: list[tuple[float, str]], groups: list[str], live: bool = True, focus: set[int] | None = None) -> int:
    """Presence-count lanes (one per class) with question markers. Returns bottom y."""
    lane_h = 28
    sx = w / max(1e-6, t1 - t0)
    for gi, g in enumerate(groups):
        ly = y + gi * (lane_h + 3)
        d.rounded_rectangle((x, ly, x + w, ly + lane_h), 5, fill=(16, 25, 40))
        log = [(t, c.get(g, 0)) for t, c in st.count_log if t <= t_now + 1e-6]
        mx = max([n for _, n in log] + [1]) if log else 1
        col = GROUP_RGB[g]
        if len(log) > 1:
            pts = [(x, ly + lane_h)]
            for t, n in log:
                pts.append((x + (t - t0) * sx, ly + lane_h - 2 - (lane_h - 6) * n / mx))
            pts.append((x + (log[-1][0] - t0) * sx, ly + lane_h))
            d.polygon(pts, fill=tuple(int(c * 0.55) for c in col))
            d.line(pts[1:-1], fill=col, width=2)
        # arrivals
        for e in st.confirmed(g):
            if e.first_t >= t0 and not e.at_alarm and e.first_t <= t_now and (not focus or e.eid in focus):
                ax = x + (e.first_t - t0) * sx
                d.polygon([(ax, ly + 2), (ax - 4, ly + 9), (ax + 4, ly + 9)], fill=(255, 255, 255))
        lab = f"{GROUP_PLURAL[g]} in view: {log[-1][1] if log else 0}" if live else f"{GROUP_PLURAL[g]} in view over the call"
        d.text((x + 6, ly + 1), lab, font=font(12, True), fill=INK)
        d.text((x + w - 56, ly + 1), f"peak {mx}", font=font(11), fill=MUTED)
    bot = y + len(groups) * (lane_h + 3)
    for k, (tq, lab) in enumerate(qs):
        if tq <= t_now + 1e-6:
            qx = x + (tq - t0) * sx
            d.line((qx, y - 2, qx, bot), fill=ACCENT, width=2)
            d.text((qx + 3, bot - 2), lab, font=font(12, True), fill=ACCENT)
    cx = x + (t_now - t0) * sx
    d.line((cx, y - 4, cx, bot + 2), fill=(255, 255, 255), width=1)
    for k in range(5):
        tt = t0 + (t1 - t0) * k / 4
        d.text((x + w * k / 4 - (22 if k == 4 else 0), bot + 12), pk.mmss(tt), font=font(10), fill=MUTED)
    return bot + 28


def _boxes(d: ImageDraw.ImageDraw, st: SceneState, t: float, rect: tuple[int, int, int, int], fw: int, fh: int,
           trails: bool = True, all_trails: bool = False, dark: bool = False, focus: set[int] | None = None,
           highlight: set[int] | None = None) -> None:
    rx, ry, rw, rh = rect
    s = min(rw / fw, rh / fh)
    ox, oy = rx + (rw - fw * s) / 2, ry + (rh - fh * s) / 2
    tol = 0.5 / max(st.fps, 1) + 1e-6
    ents = [e for e in st.confirmed() if not (focus and e.eid not in focus)]
    cur_of = {e.eid: [p for p in e.pts if abs(p[0] - t) <= tol] for e in ents}
    crowd = sum(1 for v in cur_of.values() if v) > 8 and not focus      # crowded: label only the entity police asked about
    for e in ents:
        rgb = GROUP_RGB[e.group]
        hl = bool(highlight and e.eid in highlight)
        if crowd and not hl:
            trails_here = False
        else:
            trails_here = trails
        pts = e.pts if all_trails else [p for p in e.pts if t - 2.0 <= p[0] <= t + 1e-6]
        if trails_here and len(pts) > 1:
            xy = [(ox + (p[1] + p[3]) / 2 * s, oy + (p[2] + p[4]) / 2 * s) for p in pts]
            d.line(xy, fill=rgb if not all_trails else tuple(int(c * 0.8) for c in rgb), width=2 if not all_trails else 3)
            if all_trails and not (st.now - e.last_t <= st.hold):
                (x0, y0), (x1_, y1_) = xy[max(0, len(xy) - 6)], xy[-1]
                ang = np.arctan2(y1_ - y0, x1_ - x0)
                for da in (2.7, -2.7):
                    d.line((x1_, y1_, x1_ + 10 * np.cos(ang + da), y1_ + 10 * np.sin(ang + da)), fill=rgb, width=3)
        cur = cur_of[e.eid]
        if cur and not e.carried:
            p = cur[-1]
            b = (ox + p[1] * s, oy + p[2] * s, ox + p[3] * s, oy + p[4] * s)
            if crowd and not hl:
                d.rectangle(b, outline=tuple(int(c * 0.75) for c in rgb), width=1)
                continue
            d.rectangle(b, outline=(255, 255, 255) if hl and crowd else rgb, width=3)
            c = st.colours(e)
            nm = "" if dark else (display_colour(e.group, c.get("body") or c.get("top")) or "")
            lab = f"{GROUP_WORD[e.group].upper() if e.cls_name not in ('car', 'truck', 'bus') else e.cls_name.upper()} #{e.eid} {nm}".strip()
            f = font(13, True)
            tw = d.textlength(lab, font=f)
            ly = max(ry, b[1] - 20)
            d.rectangle((b[0], ly, b[0] + tw + 8, ly + 19), fill=rgb)
            d.text((b[0] + 4, ly + 1), lab, font=f, fill=(8, 14, 24))


def compose(st: SceneState, t: float, frame: np.ndarray, cards: list[Card], *, title: str, hangup_t: float,
            questions: list[dict[str, Any]], scored: list[dict[str, Any]] | None = None, q_idx: list[int] | None = None,
            new_idx: int | None = None, caption: str | None = None, summary: bool = False,
            stamp: tuple[str, str] | None = None, log_lines: list[str] | None = None, groups: list[str] | None = None,
            slots: int = 5, focus: set[int] | None = None) -> np.ndarray:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    groups = groups or [g for g in ("person", "vehicle", "bicycle") if any(e.group == g for e in st.confirmed())] or ["person"]
    q_idx = q_idx or list(range(len(cards)))
    # header
    d.rectangle((0, 0, W, 62), fill=(10, 16, 27))
    d.text((20, 8), "ON THE LINE", font=font(30, True), fill=INK)
    if stamp:
        _header_badge(d, stamp, 300, 9)
    else:
        d.text((228, 17), "scene answers for the RVRC operator on the phone", font=font(16), fill=MUTED)
    clock = f"{'CALL SUMMARY, frame at' if summary else 'LIVE REPLAY'}  {pk.mmss(t)}"
    d.text((20, 40), f"{title}", font=font(13), fill=MUTED)
    _chip(d, (W - 20, 10), f"ALARM {pk.mmss(st.alarm_t)}   {clock}   (+{max(0.0, t - st.alarm_t):.1f} s)", INK, (31, 45, 68), font(15, True), anchor_right=True)
    # video panel
    vx, vy, vw, vh = 20, 74, 720, 405
    d.rectangle((vx - 1, vy - 1, vx + vw, vy + vh), fill=(0, 0, 0), outline=LINE)
    _paste_fit(img, frame, (vx, vy, vw, vh), bg=(0, 0, 0))
    d = ImageDraw.Draw(img)
    fh, fw = frame.shape[:2]
    _boxes(d, st, t, (vx, vy, vw, vh), fw, fh, trails=True, all_trails=summary, dark=float(frame.mean()) < 70, focus=focus,
           highlight={cards[-1].target_eid} if cards and cards[-1].target_eid else None)
    if summary:
        d.rounded_rectangle((vx + 8, vy + vh - 30, vx + 330, vy + vh - 8), 6, fill=(0, 0, 0))
        d.text((vx + 14, vy + vh - 28), ("paths of the people the police asked about" if focus else "paths of everyone seen during the call"), font=font(13), fill=INK)
    # timeline
    qs = [(q["t"], f"Q{i + 1}") for i, q in enumerate(questions)]
    bot = _timeline(d, st, vx, vy + vh + 12, vw, t if not summary else hangup_t, st.alarm_t, hangup_t, qs, groups, live=not summary, focus=focus)
    # caption / log
    if caption:
        d.rounded_rectangle((vx, bot + 2, vx + vw, bot + 46), 8, fill=(26, 40, 62))
        d.text((vx + 10, bot + 5), "POLICE (Canary ASR on the day)", font=font(11, True), fill=ACCENT)
        d.text((vx + 10, bot + 21), _wrap(d, f"\u201c{caption}\u201d", font(17), vw - 20, 1)[0], font=font(17), fill=INK)
    if log_lines:
        ly = bot + 2
        d.text((vx, ly), "INCIDENT LOG, auto-drafted at hang-up", font=font(13, True), fill=ACCENT)
        for k, ln in enumerate(log_lines[:5]):
            d.text((vx, ly + 18 + k * 16), _wrap(d, ln, font(12), vw, 1)[0], font=font(12), fill=(205, 214, 228))
    # cards
    cx, cy, cw = 760, 74, 500
    ch = (H - cy - 14 - 6 * (slots - 1)) // slots
    key = (tuple(id(c) for c in cards), new_idx, bool(scored), slots)
    hit = _CARD_CACHE.get("k") == key
    if hit:
        img.paste(_CARD_CACHE["img"], (cx, cy))
    else:
        draw_cards(d, img, cards, cx, cy, cw, scored=scored, slots=slots, card_h=ch, new_idx=new_idx, q_idx=q_idx)
        _CARD_CACHE["k"], _CARD_CACHE["img"] = key, img.crop((cx, cy, cx + cw, H - 8))
    d = ImageDraw.Draw(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _header_badge(d: ImageDraw.ImageDraw, stamp: tuple[str, str], x: int, y: int) -> None:
    """Verdict stamp in the header strip (it used to cover the live frame)."""
    big, small = stamp
    col = GOOD if "/" not in big or int(big.split("/")[0]) >= 4 else BAD
    bw = 500
    d.rounded_rectangle((x, y, x + bw, y + 44), 10, fill=(8, 30, 20), outline=col, width=3)
    d.text((x + 12, y + 1), big, font=font(30, True), fill=col)
    tx = x + 14 + int(d.textlength(big, font=font(30, True)))
    d.text((tx + 4, y + 4), "vs my eyeball truth", font=font(14, True), fill=(230, 240, 235))
    d.text((tx + 4, y + 22), small, font=font(13), fill=(170, 200, 185))


def log_screen(st: SceneState, frame: np.ndarray, cards: list[Card], log: dict[str, Any], *, title: str, hangup_t: float,
               questions: list[dict[str, Any]], scored=None, q_idx=None) -> np.ndarray:
    """Hang-up screen: the video panel is replaced by the full auto-drafted incident log."""
    base = compose(st, hangup_t, frame, cards, title=title, hangup_t=hangup_t, questions=questions, scored=scored,
                   q_idx=q_idx, summary=False, slots=max(5, len(questions)))
    img = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    x, y, w, h = 20, 74, 720, 636
    d.rounded_rectangle((x, y, x + w, y + h), 10, fill=(18, 27, 43), outline=ACCENT, width=2)
    d.text((x + 16, y + 10), "INCIDENT LOG", font=font(24, True), fill=INK)
    d.text((x + 190, y + 17), "auto-drafted at hang-up, operator reviews before filing", font=font(13), fill=MUTED)
    yy = y + 50
    for ln in log["lines"][:20]:
        col = {"qa": ACCENT, "alarm": WARN, "arrived": (150, 220, 170), "departed": (240, 170, 120)}.get(ln.kind, INK)
        d.text((x + 16, yy), pk.mmss(ln.t), font=font(13, True), fill=col)
        wl = _wrap(d, ln.text, font(14), w - 100, 2)
        for k, s in enumerate(wl):
            d.text((x + 80, yy + k * 17), s, font=font(14), fill=(220, 228, 240))
        yy += 17 * len(wl) + 5
        if yy > y + h - 90:
            break
    yy = y + h - 70
    d.line((x + 16, yy - 6, x + w - 16, yy - 6), fill=LINE)
    for k, s in enumerate(log["summary"][:3]):
        d.text((x + 16, yy + k * 20), _wrap(d, s, font(14, True), w - 32, 1)[0], font=font(14, True), fill=INK)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
