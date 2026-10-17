"""Image rendering for Hide & Seek (PIL): corpus hero and per-video heroes."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG, PANEL, LINE = (14, 17, 22), (23, 28, 36), (42, 50, 61)
FG, MUT, AMBER, GREEN, RED, SKY = (238, 241, 245), (147, 160, 176), (255, 176, 32), (47, 191, 113), (229, 72, 77), (74, 163, 255)
FONTS = ["C:/Windows/Fonts/bahnschrift.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
FONTS_B = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for f in (FONTS_B if bold else FONTS) + FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def thumb(path: Path, w: int, h: int, cover: bool = False) -> Image.Image:
    im = Image.open(path).convert("RGB")
    s = max(w / im.width, h / im.height) if cover else min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def text_w(d: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont) -> int:
    return int(d.textlength(s, font=f))


def fit_text(d: ImageDraw.ImageDraw, s: str, f: ImageFont.FreeTypeFont, maxw: int) -> str:
    if text_w(d, s, f) <= maxw:
        return s
    while s and text_w(d, s + "...", f) > maxw:
        s = s[:-1]
    return s + "..."


def pill(d: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, f, fill, fg=(255, 255, 255), pad: int = 8) -> int:
    w = text_w(d, s, f) + 2 * pad
    h = f.size + 12
    d.rounded_rectangle([xy[0], xy[1], xy[0] + w, xy[1] + h], radius=h // 2, fill=fill)
    d.text((xy[0] + pad, xy[1] + 3), s, font=f, fill=fg)
    return w


def dashed_rect(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], col, width: int = 2, dash: int = 6) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, dash * 2):
        d.line([x, y0, min(x + dash, x1), y0], fill=col, width=width)
        d.line([x, y1, min(x + dash, x1), y1], fill=col, width=width)
    for y in range(y0, y1, dash * 2):
        d.line([x0, y, x0, min(y + dash, y1)], fill=col, width=width)
        d.line([x1, y, x1, min(y + dash, y1)], fill=col, width=width)


CYAN = (64, 224, 255)


def label_pill(d: ImageDraw.ImageDraw, xy: tuple[int, int], s: str, f) -> None:
    w = text_w(d, s, f) + 14
    d.rounded_rectangle([xy[0], xy[1], xy[0] + w, xy[1] + f.size + 8], radius=7, fill=(8, 10, 14), outline=(70, 80, 92))
    d.text((xy[0] + 7, xy[1] + 3), s, font=f, fill=(255, 255, 255))


def stamp(img: Image.Image, center: tuple[int, int], lines: list[str], col, angle: float = -8, size: int = 40) -> None:
    f1, f2 = font(size, True), font(int(size * 0.5), True)
    tmp = Image.new("RGBA", (900, 260), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    w1 = text_w(d, lines[0], f1)
    w2 = max([text_w(d, s, f2) for s in lines[1:]] + [0])
    w = max(w1, w2) + 40
    h = size + 20 + (len(lines) - 1) * (f2.size + 6)
    d.rounded_rectangle([10, 10, 10 + w, 10 + h], radius=12, outline=col + (255,), width=6, fill=(14, 17, 22, 215))
    d.text((30, 14), lines[0], font=f1, fill=col + (255,))
    for k, s in enumerate(lines[1:]):
        d.text((30, 14 + size + 6 + k * (f2.size + 6)), s, font=f2, fill=col + (255,))
    tmp = tmp.crop((0, 0, w + 20, h + 20)).rotate(angle, expand=True, resample=Image.BICUBIC)
    img.paste(tmp, (center[0] - tmp.width // 2, center[1] - tmp.height // 2), tmp)


def card(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, big: str, sub: str, col=AMBER) -> None:
    d.rounded_rectangle(box, radius=12, fill=PANEL, outline=LINE)
    d.text((box[0] + 16, box[1] + 10), title, font=font(15, True), fill=MUT)
    d.text((box[0] + 16, box[1] + 30), big, font=font(44, True), fill=col)
    d.text((box[0] + 16, box[1] + 84), fit_text(d, sub, font(15), box[2] - box[0] - 28), font=font(15), fill=MUT)


def hero_corpus(path: Path, clips: list[dict[str, Any]], thumbs: Path, trace: list[dict[str, Any]], secret: int,
                stats: dict[str, Any], twin_ids: set[str], blind_ids: set[str], kill: dict[str, Any]) -> None:
    W, H = 1280, 790
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((30, 14), "HIDE", font=font(56, True), fill=FG)
    x = 30 + text_w(d, "HIDE ", font(56, True))
    d.text((x, 14), "&", font=font(56, True), fill=AMBER)
    d.text((x + text_w(d, "& ", font(56, True)), 14), "SEEK", font=font(56, True), fill=FG)
    d.text((30, 90), "Can our video index actually tell our clips apart?", font=font(24), fill=MUT)
    d.text((W - 30 - text_w(d, f"{len(clips)} clips from {len({c['video'] for c in clips})} videos", font(20)), 30),
           f"{len(clips)} clips from {len({c['video'] for c in clips})} videos", font=font(20), fill=MUT)
    d.text((W - 30 - text_w(d, f"{stats['n_questions_available']} candidate yes/no questions", font(20)), 58),
           f"{stats['n_questions_available']} candidate yes/no questions", font=font(20), fill=MUT)
    # left: a game
    c = clips[secret]
    d.rounded_rectangle([24, 130, 616, 625], radius=14, fill=PANEL, outline=LINE)
    th = thumb(thumbs / f"{c['id']}.jpg", 196, 110, cover=True)
    img.paste(th, (40, 146))
    d.rectangle([40, 146, 236, 256], outline=AMBER, width=3)
    d.text((252, 144), "SECRET CLIP FOUND", font=font(16, True), fill=MUT)
    d.text((252, 164), f"{len(trace)} questions", font=font(44, True), fill=AMBER)
    d.text((252, 218), fit_text(d, f"{c['id']}  {c['video'].replace('_', ' ')}", font(16), 350), font=font(16), fill=MUT)
    d.text((252, 238), f"{c['t0']:.0f}-{c['t1']:.0f} s", font=font(16), fill=MUT)
    y = 280
    rows = trace[:8]
    for k, t in enumerate(rows):
        yes = t["a"]
        pill(d, (40, y + 2), "YES" if yes else "NO", font(15, True), GREEN if yes else RED, pad=9)
        d.text((100, y), fit_text(d, t["text"].replace("At any moment, ", "Any moment: "), font(18), 410), font=font(18), fill=FG)
        right = f"{t['before']} > {t['after']}"
        d.text((600 - text_w(d, right, font(19, True)), y), right, font=font(19, True), fill=AMBER)
        y += 40
    if len(trace) > 8:
        d.text((100, y), f"+ {len(trace) - 8} more", font=font(16), fill=MUT)
        y += 30
    pill(d, (40, y + 2), "YES", font(15, True), GREEN, pad=9)
    d.text((100, y), f"Is it this one?  ({c['id']}, confirmation guess)", font=font(18), fill=FG)
    d.text((600 - text_w(d, "found", font(19, True)), y), "found", font=font(19, True), fill=GREEN)
    # right: grid of all clips
    gx, gy, tw, tht, gap = 650, 130, 70, 39, 5
    cols = 8
    for i, cl in enumerate(clips):
        r, k = divmod(i, cols)
        x0, y0 = gx + k * (tw + gap), gy + r * (tht + gap)
        t = thumb(thumbs / f"{cl['id']}.jpg", tw, tht, cover=True)
        if cl["id"] in blind_ids:
            t = Image.blend(t.convert("L").convert("RGB"), Image.new("RGB", (tw, tht), (0, 0, 0)), 0.55)
        img.paste(t, (x0, y0))
        if cl["id"] in twin_ids:
            d.rectangle([x0 - 1, y0 - 1, x0 + tw, y0 + tht], outline=RED, width=2)
        if cl["id"] in blind_ids:
            dashed_rect(d, (x0 - 2, y0 - 2, x0 + tw + 1, y0 + tht + 1), CYAN, 3, 5)
        if i == secret:
            d.rectangle([x0 - 2, y0 - 2, x0 + tw + 1, y0 + tht + 1], outline=AMBER, width=3)
    ly = gy + math.ceil(len(clips) / cols) * (tht + gap) + 6
    xx = gx
    for col, lab in ((AMBER, "secret"), (RED, "has an attribute twin"), (CYAN, "detectors see nothing (dashed)")):
        if col == CYAN:
            dashed_rect(d, (xx, ly + 4, xx + 16, ly + 18), col, 3, 4)
        else:
            d.rectangle([xx, ly + 4, xx + 16, ly + 18], outline=col, width=3)
        d.text((xx + 24, ly), lab, font=font(15), fill=MUT)
        xx += 24 + text_w(d, lab, font(15)) + 22
    # stats cards
    cw = 300
    y = 650
    card(d, (24, y, 24 + cw, y + 120), "FOUND UNIQUELY", f"{stats['n_found']}/{stats['n_clips']}",
         f"by questions alone ({100 * stats['success_rate']:.0f}%)", GREEN if stats["success_rate"] >= 0.9 else AMBER)
    card(d, (24 + cw + 14, y, 24 + 2 * cw + 14, y + 120), "MEAN QUESTIONS", f"{stats['mean_questions_all']:.2f}",
         f"perfect halving would be {math.log2(stats['n_clips']):.1f}; worst clip {stats['max_questions']}",
         GREEN if stats["mean_questions_all"] <= 7 else AMBER)
    card(d, (24 + 2 * (cw + 14), y, 24 + 3 * cw + 28, y + 120), "ATTRIBUTE TWINS", f"{stats['clips_in_twin_groups']} clips",
         f"in {stats['twin_groups']} groups the index cannot separate", RED if stats["clips_in_twin_groups"] else GREEN)
    card(d, (24 + 3 * (cw + 14), y, 24 + 4 * cw + 42, y + 120), "INVISIBLE TO SEARCH", f"{stats['blind']} clips",
         "no detection of any COCO class", (170, 170, 170))
    ok = kill["passed"]
    lines = [("KILL TEST PASSED" if ok else "KILL TEST FAILED"), kill["short"]]
    if kill.get("v1_found") is not None:
        lines.append(f"was {100 * kill['v1_found']:.0f}% before caption keywords were added")
    stamp(img, (1000, 580), lines, GREEN if ok else RED, -3, 30)
    d.text((24, 773), "Local stand-ins: YOLO26s tracks, local Cosmos3-Edge, duckdb. Footage: Intel IoT, Wikimedia Commons, Pexels (see report).",
           font=font(13), fill=(100, 110, 122))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)


def hero_video(path: Path, video: str, items: list[dict[str, Any]], thumbs: Path, summary: dict[str, Any], credit: str) -> None:
    """items: per clip {id, t0, t1, chips[list], n_q, found, twins[list ids], blind(bool), why}."""
    W = 1280
    n = len(items)
    ncol = 4
    nrow = math.ceil(n / ncol)
    cw = (W - 48 - (ncol - 1) * 16) // ncol
    th_h = int(cw * 9 / 16)
    card_h = th_h + 128
    H = 160 + nrow * (card_h + 16) + 40
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 16), video.replace("_", " ").upper(), font=font(42, True), fill=FG)
    d.text((24, 74), f"{n} clips in the archive - can the index find each one, and tell it from its neighbours?",
           font=font(20), fill=MUT)
    d.text((24, 104), fit_text(d, credit, font(14), 840), font=font(14), fill=(100, 110, 122))
    for k, it in enumerate(items):
        r, c = divmod(k, ncol)
        x0, y0 = 24 + c * (cw + 16), 146 + r * (card_h + 16)
        d.rounded_rectangle([x0, y0, x0 + cw, y0 + card_h], radius=12, fill=PANEL, outline=LINE)
        t = thumb(thumbs / f"{it['id']}.jpg", cw - 16, th_h - 8, cover=True)
        if it["blind"]:
            t = Image.blend(t.convert("L").convert("RGB"), Image.new("RGB", t.size, (0, 0, 0)), 0.45)
        img.paste(t, (x0 + 8, y0 + 8))
        label_pill(d, (x0 + 14, y0 + 14), f"{it['id']}  {it['t0']:.0f}-{it['t1']:.0f}s", font(15, True))
        if it["blind"]:
            dashed_rect(d, (x0 + 8, y0 + 8, x0 + cw - 8, y0 + 8 + th_h - 8), CYAN, 3, 8)
        xy = [x0 + 14, y0 + th_h + 12]
        for chip in it["chips"][:4]:
            w = text_w(d, chip, font(14)) + 16
            if xy[0] + w > x0 + cw - 8:
                xy = [x0 + 14, xy[1] + 26]
            pill(d, tuple(xy), chip, font(14), (44, 54, 68), (214, 222, 232), pad=8)
            xy[0] += w + 6
        by = y0 + card_h - 44
        if it["blind"]:
            pill(d, (x0 + 14, by), f"INVISIBLE ({it['why']})", font(16, True), (96, 96, 96))
        elif it["found"]:
            pill(d, (x0 + 14, by), f"FOUND IN {it['n_q']} Q", font(18, True), GREEN)
        else:
            pill(d, (x0 + 14, by), "TWIN: " + ", ".join(it["twins"][:3]), font(18, True), RED)
    ok = summary["n_found"] == summary["n"]
    col = GREEN if ok else (RED if summary["n_found"] < summary["n"] / 2 else AMBER)
    lab = "ALL FINDABLE" if ok else f"{summary['n_found']} OF {summary['n']} FINDABLE"
    stamp(img, (W - 250, 62), [lab, f"mean {summary['mean_q']:.1f} questions" if summary["mean_q"] is not None else ""], col, -4, 30)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)
