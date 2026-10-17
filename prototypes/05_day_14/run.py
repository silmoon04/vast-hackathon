"""DAY 14: long dashcam recording -> top-5 candidate incident moments -> Operation Snap packet + 14-day clock.

    python run.py --all                         # dashcam_urban, motorway_a40, person_bicycle_car_detection
    python run.py --video dashcam_urban --recorded 2026-10-10 --start-time 18:40:00 --location "Frankfurt, Hanauer Landstr."

Pipeline (each stage names the VAST service that replaces the local stand-in on the day, see README):
  tracks (DataEngine YOLO11s boxes + tracker) -> ego-motion flow (CPU) -> ground-plane geometry + severity ->
  top-5 peaks -> clip / key frame / zoom crop / Cosmos text -> Snap packet (+ 14-day clock) -> hero + report.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import html
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protokit as pk  # noqa: E402
import core  # noqa: E402
import ego as egomod  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PARKED_RE = re.compile(r"\b(parked|stationary|not moving)\b", re.I)
POOL = 8
MIN_SEV = 0.15
DEFAULT_SET = ["dashcam_urban", "motorway_a40", "person_bicycle_car_detection"]
WINDOW_DAYS = 14
CLIP_PRE, CLIP_POST = 5.0, 5.0

KIND_LABEL = {"close pass": "CLOSE PASS", "closing": "CLOSING FAST", "cut-in / crossing": "CUT-IN / CROSSING"}
OFFENCE = {  # wording suggestions only; the user and the force decide
    "close pass": "Close pass / driving without due care and attention",
    "closing": "Vehicle closing fast in my lane (following too close / sudden braking)",
    "cut-in / crossing": "Cutting in or crossing in front of me (without due consideration)",
}


# ------------------------------------------------------------------------------------------------------ fonts / draw
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for f in (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int, max_lines: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=fnt) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;") + "..."
    return lines


# ------------------------------------------------------------------------------------------------------ analysis
def clock(recorded: dt.date, today: dt.date) -> dict:
    deadline = recorded + dt.timedelta(days=WINDOW_DAYS)
    left = (deadline - today).days
    state = "passed" if left < 0 else "urgent" if left <= 3 else "soon" if left <= 7 else "ok"
    ago = f"{-left:,} days ago" + (f" ({-left / 365.25:.1f} years)" if -left > 365 else "")
    text = (f"recorded {recorded:%d %b %Y} -> deadline {deadline:%d %b %Y}, {left} days left" if left >= 0 else
            f"recorded {recorded:%d %b %Y} -> deadline {deadline:%d %b %Y} PASSED {ago}")
    return {"recorded": recorded.isoformat(), "deadline": deadline.isoformat(), "days_left": left, "state": state,
            "text": text, "today": today.isoformat(), "window_days": WINDOW_DAYS, "recorded_source": "given"}


DATE_RE = re.compile(r"(\d{4})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{1,2})(?:\D+(\d{1,2}):(\d{2}):(\d{2}))?")


def parse_overlay(text: str, today: dt.date) -> dt.datetime | None:
    """First plausible YYYY/MM/DD [HH:MM:SS] in a model answer; None if absent, impossible, or in the future."""
    m = DATE_RE.search(text or "")
    if not m:
        return None
    y, mo, d_ = (int(m.group(i)) for i in (1, 2, 3))
    hh, mm, ss = (int(m.group(i)) if m.group(i) else 0 for i in (4, 5, 6))
    try:
        v = dt.datetime(y, mo, d_, hh, mm, ss)
    except ValueError:
        return None
    return v if 2000 <= y and v.date() <= today else None


def read_overlay_date(src: str, meta: dict, today: dt.date, timeout: float = 60.0) -> dict | None:
    """Read a burned-in recording date from the top strip of 3 frames with Cosmos (local stand-in). Accepted only when at
    least 2 of 3 reads give the same plausible date; otherwise None and the caller falls back to today."""
    ts = [meta["duration"] * q for q in (0.1, 0.5, 0.9)]

    def one(t: float) -> dt.datetime | None:
        fr = pk.frame_at(src, min(t, meta["duration"] - 0.1))
        h, w = fr.shape[:2]
        best = None
        for y0, y1 in ((0, 0.075), (0.925, 1.0)):                       # top strip first (this camera), then bottom strip
            strip = cv2.resize(fr[int(h * y0):int(h * y1), int(w * 0.2):int(w * 0.8)], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            txt = pk.vlm("Read the burned-in date and time text in this image. Reply exactly in the form YYYY/MM/DD HH:MM:SS, "
                         "or UNREADABLE if there is no date text.", [strip], max_tokens=30, timeout=timeout)
            best = parse_overlay(txt, today)
            if best:
                return best
        return best

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        reads = list(ex.map(one, ts))
    days = [r.date() for r in reads if r]
    if len(days) < 2:
        return None
    day, n = max(((d_, days.count(d_)) for d_ in set(days)), key=lambda x: x[1])
    if n < 2:
        return None
    starts = [r - dt.timedelta(seconds=t) for r, t in zip(reads, ts) if r and r.date() == day and (r.hour or r.minute or r.second)]
    start = None
    if len(starts) >= 2 and max(starts) - min(starts) <= dt.timedelta(seconds=4):
        start = starts[0].strftime("%H:%M:%S")
    return {"date": day.isoformat(), "start_time": start, "reads": [None if r is None else r.isoformat() for r in reads],
            "agree": f"{n}/3", "source": "burned-in overlay read by Cosmos3-Edge (local stand-in)"}


def verdict(packets: bool, cands: list[dict]) -> dict:
    """What the hero stamp may claim. The tool finds moments worth a human look; it never says an offence happened."""
    if not packets:
        return {"code": "not_dashcam", "lines": ["NOT A DASHCAM"], "rgb": (235, 70, 70),
                "sub": "fixed camera: no ego vehicle, no Snap packet"}
    if not cands:
        return {"code": "none", "lines": ["NO INCIDENTS FOUND"], "rgb": (90, 190, 220),
                "sub": "nothing above the severity floor: nothing to report"}
    return {"code": "review", "lines": ["REVIEW", "CLOSE INTERACTION"], "rgb": (245, 176, 45),
            "sub": "a person decides whether it is an offence"}  # never SNAP-READY: no signal here is reliable enough


def moving_label(ev: float, ratio2: float) -> str:
    if np.isnan(ratio2):
        return "unknown"
    return "moving-like" if ev >= 0.7 else "static-like" if ev <= 0.3 else "unclear"


def mine(name: str, tr: pk.Tracks, ego: dict, k: int) -> dict:
    """Stage 2: severity per track-frame, risk curve, top-k separated peaks, culprit per peak."""
    feat = core.per_track_features(tr, ego)
    sc = core.score(feat, tr.h)
    grid, curve = core.risk_curve(sc, tr.duration)
    peaks = core.pick_peaks(grid, curve, k=k)
    cands = []
    for i in peaks:
        t = float(grid[i])
        w = sc[(sc["t"] - t).abs() < 0.15].sort_values("sev", ascending=False).iloc[0]
        cands.append({"t": round(t, 2), "sev": round(float(curve[i]), 3), "tid": int(w["tid"]), "cls": str(w["name"]),
                      "kind": str(w["kind"]), "ttc_s": None if not np.isfinite(w["ttc"]) else round(float(w["ttc"]), 2),
                      "range_m": round(float(w["Z"]), 1), "lateral_m": round(float(w["X"]), 1),
                      "clearance_m": round(float(w["clear"]), 1), "growth_per_s": round(float(w["g"]), 2),
                      "ratio2": None if np.isnan(w["ratio2"]) else round(float(w["ratio2"]), 2),
                      "moving_ev": round(float(w["moving_ev"]), 2),
                      "moving": moving_label(float(w["moving_ev"]), float(w["ratio2"])),
                      "t_frame": float(w["t"])})
    return {"sc": sc, "grid": grid, "curve": curve, "cands": cands}


def box_at(tr: pk.Tracks, tid: int, t: float) -> pd.Series | None:
    g = tr.rows[tr.rows["tid"] == tid]
    if g.empty:
        return None
    j = (g["t"] - t).abs().idxmin()
    return g.loc[j] if abs(g.loc[j, "t"] - t) < 0.4 else None


def annotate(frame: np.ndarray, tr: pk.Tracks, t: float, tid: int, tag: str, color: tuple = (40, 40, 235)) -> np.ndarray:
    f = frame.copy()
    for _, b in tr.at(t, tol=0.2).iterrows():
        if b["name"] in core.VEHICLES and int(b["tid"]) != tid:
            pk.box(f, [b.x1, b.y1, b.x2, b.y2], (150, 150, 150), None, 1)
    b = box_at(tr, tid, t)
    if b is not None:
        pk.box(f, [b.x1, b.y1, b.x2, b.y2], color, None, 4)
        pk.label(f, tag, (int(b.x1), max(0, int(b.y1) - 30)), scale=0.7, bg=color)
    return f


def best_crop(video: str, tr: pk.Tracks, tid: int, t: float) -> tuple[np.ndarray, float, dict] | None:
    """Sharpest, largest, un-truncated view of the culprit within +-1.2 s."""
    g = tr.rows[(tr.rows["tid"] == tid) & ((tr.rows["t"] - t).abs() <= 1.2)]
    if g.empty:
        return None
    best, score = None, -1.0
    want = {int(f) for f in g["f"]}
    for i, tt, fr in pk.iter_frames(video, stride=1, start=max(0.0, g["t"].min() - 0.05), end=g["t"].max() + 0.1):
        if i not in want:
            continue
        b = g[g["f"] == i].iloc[0]
        x1, y1, x2, y2 = (int(max(0, v)) for v in (b.x1, b.y1, b.x2, b.y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = fr[y1:y2, x1:x2]
        sharp = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        trunc = b.x1 < 3 or b.x2 > tr.w - 3 or b.y2 > tr.h - 3
        s = (x2 - x1) * (y2 - y1) * (1 + min(sharp, 400) / 200.0) * (0.35 if trunc else 1.0)
        if s > score:
            score, best = s, (fr, tt, {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "sharp": sharp, "f": int(i)})
    if best is None:
        return None
    fr, tt, bb = best
    bw, bh = bb["x2"] - bb["x1"], bb["y2"] - bb["y1"]
    mx, my = 0.12 * bw, 0.12 * bh
    x1, y1 = int(max(0, bb["x1"] - mx)), int(max(0, bb["y1"] - my))
    x2, y2 = int(min(tr.w, bb["x2"] + mx)), int(min(tr.h, bb["y2"] + my))
    crop = fr[y1:y2, x1:x2]
    return crop, tt, {**bb, "crop_w": x2 - x1, "crop_h": y2 - y1, "frame": fr}


def rule_text(c: dict) -> str:
    side = "left" if c["lateral_m"] < -0.3 else "right" if c["lateral_m"] > 0.3 else "ahead"
    return (f"{c['cls']} {side} of the camera car at about {c['range_m']} m (range proxy); "
            f"{'closing at TTC proxy %.1f s' % c['ttc_s'] if c['ttc_s'] else 'no clear closing'}.")


def describe(frame_boxed: np.ndarray, c: dict, use_vlm: bool, timeout: float) -> dict:
    if not use_vlm:
        return {"text": "", "source": "off"}
    prompt = ("This is a frame from a forward-facing car dashcam. The vehicle inside the RED box is the one being reported. "
              "In one sentence describe that vehicle by colour and body type only (for example 'silver saloon car') and which side "
              "of the camera car it is on. Do not say what it is doing. Do not guess the make or model and do not guess any number plate.")
    t0 = time.time()
    text = pk.vlm(prompt, [frame_boxed], max_tokens=90, timeout=timeout)
    return {"text": " ".join(text.split()), "source": "cosmos3-edge (local stand-in)" if text else "unavailable",
            "seconds": round(time.time() - t0, 1)}


def snap_packet(video: str, meta: dict, sha: str, c: dict, idx: int, win: tuple[float, float], desc: dict,
                ck: dict, args: argparse.Namespace, crop_info: dict | None) -> dict:
    t = c["t"]
    wall = None
    known = not ck["recorded_source"].startswith("assumed")
    start_time = args.start_time or ck.get("start_time")
    if start_time:
        h, m, s = (int(x) for x in start_time.split(":"))
        base = dt.datetime.combine(dt.date.fromisoformat(ck["recorded"]), dt.time(h, m, s))
        wall = (base + dt.timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")
    vehicle = desc["text"] or rule_text(c)
    narrative = (
        f"On {ck['recorded'] if known else '[date]'} at about {wall.split(' ')[1] if wall else '[time]'} I was [driving / cycling] on "
        f"[road name / junction{', ' + args.location if args.location else ''}], heading [direction]. "
        f"At {pk.mmss(t)} into my camera recording a vehicle ({c['cls']}) "
        f"{'came very close alongside me' if c['kind'] == 'close pass' else 'moved into my path' if c['kind'].startswith('cut') else 'closed on me quickly in my lane'}. "
        "[Say what you did: braked / swerved / nothing - only what is true.] "
        "[Describe the other driver's behaviour in your own words.] "
        "The footage is the unedited original; its SHA-256 is given below.")
    return {
        "packet_id": f"{video}-{idx}",
        "status": "DRAFT - the person submitting must check every word is true to what they saw",
        "clock": ck,
        "incident": {"stream_time": pk.mmss(t), "stream_seconds": t, "wall_clock": wall or "unknown (give --start-time HH:MM:SS)",
                     "date": f"{ck['recorded']} ({ck['recorded_source']})",
                     "location": args.location or "[add street / junction / GPS from your camera app]",
                     "clip": f"cand_{idx}/clip.mp4 ({pk.mmss(win[0])} to {pk.mmss(win[1])} of the recording, unedited re-encode)",
                     "category_suggestion": OFFENCE[c["kind"]] + " (suggestion only; choose on the portal)"},
        "vehicle": {"detector_class": c["cls"], "ai_description": vehicle, "ai_description_source": desc["source"],
                    "registration": "[read it from the clip: NOT extracted by this tool]",
                    "crop_px": None if not crop_info else f"{crop_info['crop_w']}x{crop_info['crop_h']}"},
        "statement_draft": narrative,
        "analysis_notes_not_for_statement": {
            "kind": c["kind"], "severity_0_1": c["sev"], "ttc_proxy_s": c["ttc_s"], "range_proxy_m": c["range_m"],
            "lateral_offset_proxy_m": c["lateral_m"], "relative_motion_test": c["moving"],
            "note": "Proxies from a monocular ground-plane model with assumed camera height and focal length; they rank "
                    "moments, they are not measurements and must not be quoted as distances or speeds."},
        "evidence": {"source_file": Path(meta["path"]).name, "source_sha256": sha, "source_duration_s": round(meta["duration"], 1),
                     "resolution": f"{meta['w']}x{meta['h']}", "keep": "Keep the original file; the portal may ask for it."},
    }


# ------------------------------------------------------------------------------------------------------ hero image
def context_zoom(frame: np.ndarray, box: tuple[float, float, float, float], out_w: int, out_h: int) -> tuple[Image.Image, tuple]:
    """Crop around the box (>= 3x its size, >= 40% of the frame width, same aspect as the panel) so the vehicle is big
    but the road around it stays visible. Returns the resized crop and the crop rectangle in source pixels."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    cw = min(W, max(3.0 * (x2 - x1), 0.40 * W, 3.0 * (y2 - y1) * out_w / out_h))
    ch = cw * out_h / out_w
    if ch > H:
        ch, cw = H, H * out_w / out_h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1 = int(min(max(0, cx - cw / 2), W - cw))
    Y1 = int(min(max(0, cy - ch / 2), H - ch))
    crop = frame[Y1:Y1 + int(ch), X1:X1 + int(cw)]
    return pil(cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)), (X1, Y1, int(cw), int(ch))


def render_hero(path: Path, name: str, meta: dict, res: dict, key_info: dict | None, crop_img: np.ndarray | None,
                ck: dict, mode: dict, timings: dict, packets: bool, thumbs: list | None = None) -> None:
    Wd, Hd = 1280, 720
    bg, fg, mut = (15, 19, 24), (238, 242, 246), (140, 152, 164)
    grey = (96, 106, 118)
    img = Image.new("RGB", (Wd, Hd), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, Wd, 62], fill=(24, 30, 38))
    d.text((24, 12), "DAY 14", font=font(30, True), fill=(255, 196, 61))
    d.text((142, 18), "dashcam in  ->  Operation Snap packet out", font=font(22), fill=fg)
    dur = meta["duration"]
    d.text((Wd - 24, 20), f"{name}.mp4   {pk.mmss(dur)[:5]} of footage", font=font(20), fill=mut, anchor="ra")
    cands = res["cands"]
    top = cands[0] if cands else None
    vd = verdict(packets, cands)
    res["verdict"] = {k: vd[k] for k in ("code", "lines", "sub")}
    analysed = packets and bool(cands)                       # only a real dashcam with candidates gets red/amber ticks + clock
    kx, ky, kw, kh = 24, 76, 770, 433
    full_w = vd["code"] == "none"
    if full_w:
        # nothing found: no empty frame panel, a plain statement instead
        d.rounded_rectangle([kx, ky, kx + 1232, ky + kh], 12, fill=(22, 28, 36), outline=(60, 70, 82), width=2)
        d.text((kx + 616, ky + 120), "0", font=font(120, True), fill=vd["rgb"], anchor="mm")
        d.text((kx + 616, ky + 215), "moments worth a Snap report", font=font(34, True), fill=fg, anchor="mm")
        mx = max([c["sev"] for c in res.get("pool", [])] or [0])
        d.text((kx + 616, ky + 262), f"scanned {pk.mmss(dur)[:5]} of dashcam footage; the highest risk score was {mx:.2f}, below the {MIN_SEV:.2f} floor",
               font=font(21), fill=mut, anchor="mm")
        d.text((kx + 616, ky + 292), "no close pass, cut-in or fast closing in the lane: no packet, no clock", font=font(21), fill=mut, anchor="mm")
    elif key_info is not None:
        box = (key_info["x1"], key_info["y1"], key_info["x2"], key_info["y2"])
        zoom, rect = context_zoom(key_info["annotated"], box, kw, kh)
        img.paste(zoom, (kx, ky))
        d.rectangle([kx, ky, kx + kw, ky + kh], outline=(60, 70, 82), width=2)
        # full-frame inset with the zoom window outlined, for context
        iw = 214
        ih = int(iw * key_info["annotated"].shape[0] / key_info["annotated"].shape[1])
        inset = pil(cv2.resize(key_info["annotated"], (iw, ih), interpolation=cv2.INTER_AREA))
        ix, iy = kx + kw - iw - 12, ky + kh - ih - 12
        img.paste(inset, (ix, iy))
        sx_ = iw / key_info["annotated"].shape[1]
        d.rectangle([ix + rect[0] * sx_, iy + rect[1] * sx_, ix + (rect[0] + rect[2]) * sx_, iy + (rect[1] + rect[3]) * sx_],
                    outline=(255, 255, 255), width=2)
        d.rectangle([ix, iy, ix + iw, iy + ih], outline=(200, 208, 216), width=2)
    else:
        d.rectangle([kx, ky, kx + kw, ky + kh], fill=(30, 36, 44))
    if top and not full_w:
        label = f"#1  {KIND_LABEL[top['kind']]}" if packets else "look-alike only: not analysed for Snap"
        col = (232, 62, 62) if packets else grey
        fw = font(30, True) if packets else font(22, True)
        tw = d.textlength(label, font=fw)
        by = ky + 12 if packets else ky + kh - 58
        d.rounded_rectangle([kx + 12, by, kx + 32 + tw, by + 46], 8, fill=col)
        d.text((kx + 22, by + 3 if packets else by + 10), label, font=fw, fill=(255, 255, 255))
        if packets:
            sub = f"{pk.mmss(top['t'])[:5]} in   .   TTC proxy {top['ttc_s']:.1f} s   .   {top['range_m']:.0f} m (proxy)" if top["ttc_s"] \
                else f"{pk.mmss(top['t'])[:5]} in   .   range proxy {top['range_m']:.0f} m"
            d.rounded_rectangle([kx + 12, ky + kh - 52, kx + 24 + d.textlength(sub, font=font(22)), ky + kh - 14], 8, fill=(0, 0, 0))
            d.text((kx + 20, ky + kh - 48), sub, font=font(22), fill=fg)
    # right panel: zoom + cosmos
    rx, rw = 816, 440
    if not full_w:
        d.text((rx, 80), "ZOOM ON THE VEHICLE" if packets else "WHAT THE DETECTOR FLAGGED", font=font(15, True), fill=mut)
        ch = 0
        if crop_img is not None:
            sc = min(rw / crop_img.shape[1], 200 / crop_img.shape[0])
            cw, ch = int(crop_img.shape[1] * sc), int(crop_img.shape[0] * sc)
            img.paste(pil(cv2.resize(crop_img, (cw, ch), interpolation=cv2.INTER_CUBIC)), (rx, 102))
            d.rectangle([rx, 102, rx + cw, 102 + ch], outline=(232, 62, 62) if packets else grey, width=3)
        y = 102 + max(ch, 40) + 14
        d.text((rx, y), "COSMOS SAYS (local stand-in, can be wrong)" if packets else "COSMOS", font=font(15, True), fill=mut)
        txt = (res.get("top_text") or "").strip() or ("(Cosmos description unavailable in this run)" if packets
                                                      else "(not run: static camera, no packet)")
        for ln in wrap(d, txt, font(19), rw, 2):
            y += 22
            d.text((rx, y), ln, font=font(19), fill=fg)
        y += 40
        d.text((rx, y), "PLATE: not read by this tool. Check the clip." if packets else 'No ego vehicle, so no "my lane".',
               font=font(15), fill=mut)
    if thumbs and packets and not full_w:
        d.text((rx, ky + kh - 90), f"{len(thumbs)} MOMENT{'S' if len(thumbs) != 1 else ''} FOUND", font=font(15, True), fill=mut)
        tx = rx
        for n_, th_ in enumerate(thumbs, 1):
            tw_, thh = 82, 58
            img.paste(pil(cv2.resize(th_, (tw_, thh), interpolation=cv2.INTER_AREA)), (tx, ky + kh - 66))
            col_ = (232, 62, 62) if n_ == 1 else (255, 196, 61)
            d.rectangle([tx, ky + kh - 66, tx + tw_, ky + kh - 66 + thh], outline=col_, width=2)
            d.ellipse([tx + 3, ky + kh - 63, tx + 23, ky + kh - 43], fill=col_)
            d.text((tx + 13, ky + kh - 53), str(n_), font=font(14, True), fill=(0, 0, 0), anchor="mm")
            tx += tw_ + 7
    # verdict stamp
    stamp = Image.new("RGBA", (Wd, Hd), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    scol = (*vd["rgb"], 238)
    f1 = font(46, True) if len(vd["lines"]) == 1 else font(44, True)
    f2 = font(23, True)
    w1 = sd.textlength(vd["lines"][0], font=f1)
    w2 = sd.textlength(vd["lines"][1], font=f2) if len(vd["lines"]) > 1 else 0
    tw = max(w1, w2)
    hh = 68 if len(vd["lines"]) == 1 else 104
    if full_w:
        sx, sy = kx + 616 - tw / 2, ky + 318
    else:
        sx, sy = kx + kw - tw - 56, ky + 22
    sd.rounded_rectangle([sx - 16, sy - 6, sx + tw + 16, sy - 6 + hh], 10, outline=scol, width=6, fill=(0, 0, 0, 165))
    sd.text((sx + tw / 2, sy), vd["lines"][0], font=f1, fill=scol, anchor="ma")
    if len(vd["lines"]) > 1:
        sd.text((sx + tw / 2, sy + 54), vd["lines"][1], font=f2, fill=scol, anchor="ma")
    stamp = stamp.rotate(0 if full_w else -5, resample=Image.BICUBIC, center=(sx + tw / 2, sy + hh / 2))
    img.paste(stamp, (0, 0), stamp)
    d = ImageDraw.Draw(img)
    if name.startswith("sim_"):
        msg = "SIMULATED: cut-in injected into real footage"
        d.rounded_rectangle([kx + 12, ky + 64, kx + 32 + d.textlength(msg, font=font(20, True)), ky + 96], 6, fill=(255, 196, 61))
        d.text((kx + 22, ky + 67), msg, font=font(20, True), fill=(0, 0, 0))
    # risk timeline
    ty, th = ky + kh + 14, 60
    d.text((24, ty), "RISK ACROSS THE WHOLE RECORDING (no scrubbing)" if packets else
           "RISK-LIKE SIGNAL (fixed camera: not used, shown for the record)", font=font(15, True), fill=mut)
    ty += 30
    d.rectangle([24, ty, 24 + 1232, ty + th], fill=(24, 30, 38))
    grid, curve = res["grid"], res["curve"]
    if len(grid) > 2 and curve.max() > 0:
        xs = 24 + grid / max(grid[-1], 1e-6) * 1232
        ys = ty + th - 4 - (curve / max(curve.max(), 0.3)) * (th - 14)
        pts = [(24, ty + th)] + list(zip(xs, ys)) + [(24 + 1232, ty + th)]
        d.polygon(pts, fill=(52, 86, 140) if packets else (52, 60, 70))
        d.line(list(zip(xs, ys)), fill=(110, 160, 235) if packets else (110, 118, 128), width=2)
    for n_, c in enumerate(cands, 1):
        x = 24 + c["t"] / max(dur, 1e-6) * 1232
        tc = ((232, 62, 62) if n_ == 1 else (255, 196, 61)) if packets else grey
        d.line([x, ty, x, ty + th], fill=tc, width=2)
        d.ellipse([x - 12, ty - 8, x + 12, ty + 16], fill=tc)
        d.text((x, ty + 4), str(n_), font=font(15, True), fill=(0, 0, 0) if packets else (230, 234, 238), anchor="mm")
    # clock + chips
    cy = ty + th + 10
    d.text((24, cy), "14-DAY CLOCK", font=font(15, True), fill=mut)
    cy += 22
    barw = 1232
    seg = barw / WINDOW_DAYS
    elapsed = WINDOW_DAYS - max(0, min(WINDOW_DAYS, ck["days_left"]))
    ccol = {"ok": (40, 200, 110), "soon": (240, 170, 40), "urgent": (235, 70, 70), "passed": (235, 70, 70)}[ck["state"]]
    for i in range(WINDOW_DAYS):
        if not analysed:
            fillc = (46, 54, 64)                                  # no packet: the clock is not running
        elif ck["state"] == "passed":
            fillc = ccol                                          # whole window gone
        else:
            fillc = (70, 78, 90) if i < elapsed else ccol
        d.rounded_rectangle([24 + i * seg + 2, cy, 24 + (i + 1) * seg - 2, cy + 14], 4, fill=fillc)
    if analysed:
        rec, dl = dt.date.fromisoformat(ck["recorded"]), dt.date.fromisoformat(ck["deadline"])
        tail = f"{ck['days_left']} DAYS LEFT" if ck["days_left"] >= 0 else "PASSED " + ck["text"].split("PASSED ")[1].upper()
        d.text((24, cy + 18), f"RECORDED {rec:%d %b %Y} -> DEADLINE {dl:%d %b %Y}: {tail}".upper().replace("PASSED", "PASSED"),
               font=font(18, True), fill=ccol)
        d.text((24, cy + 42), "recording date: " + ck["recorded_source"], font=font(14), fill=mut)
    else:
        why = "no packet: not a dashcam, so no clock" if not packets else "no packet: nothing to report, so no clock"
        d.text((24, cy + 18), why.upper(), font=font(18, True), fill=grey)
    chips = f"analysed {pk.mmss(dur)[:5]} in {timings['total']:.0f} s  .  {len(cands)} candidates  .  camera: {mode['mode']}"
    d.text((Wd - 24, cy + 20), chips, font=font(17), fill=mut, anchor="ra")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)


# ------------------------------------------------------------------------------------------------------ one video
def run_video(src: str, args: argparse.Namespace) -> dict:
    name = pk.name_of(src)
    T = {}
    t_all = time.time()
    meta = pk.info(src)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()

    t0 = time.time()
    tr = pk.tracks(src)
    T["tracks_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    ego = egomod.ego_signals(src, refresh=args.refresh_ego)
    T["ego_flow_s"] = round(time.time() - t0, 1)
    mode = core.camera_mode(ego)
    packets = mode["mode"] == "dashcam"
    # recording date: --recorded > burned-in overlay read by Cosmos (dashcams only) > assumed today
    t0 = time.time()
    ov = None
    if args.recorded:
        recorded, rsrc = dt.date.fromisoformat(args.recorded), "from --recorded"
    else:
        ov = read_overlay_date(src, meta, today) if (packets and not args.no_vlm) else None
        recorded, rsrc = ((dt.date.fromisoformat(ov["date"]),
                           f"read from the burned-in overlay, {ov['agree']} reads agree, Cosmos stand-in") if ov else
                          (today, "assumed = today; no readable overlay, pass --recorded YYYY-MM-DD"))
    ck = clock(recorded, today)
    ck["recorded_source"] = rsrc
    ck["overlay_read"] = ov
    if ov and ov.get("start_time"):
        ck["start_time"] = ov["start_time"]
    T["date_read_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    res = mine(name, tr, ego, POOL)
    T["mine_s"] = round(time.time() - t0, 1)
    sha = pk.sha256(meta["path"])
    pool = res["cands"]                                     # top-POOL moments by geometry alone
    for i, c in enumerate(pool):
        c["pool_rank"] = i + 1
    print(f"[{name}] mode={mode['mode']} pool={[(c['t'], c['kind'], c['sev']) for c in pool]}", flush=True)

    # ---- stage 3 (optional): Cosmos describes every pooled moment; 'parked' answers are demoted, never required
    t0 = time.time()
    use_vlm = not args.no_vlm and packets
    ex = cf.ThreadPoolExecutor(max_workers=max(1, len(pool)))
    futures = {}
    for c in pool:
        fr = pk.frame_at(src, min(c["t"], meta["duration"] - 0.05))
        c["_keyplain"] = annotate(fr, tr, c["t_frame"], c["tid"], "reported vehicle")
        futures[c["pool_rank"]] = ex.submit(describe, c["_keyplain"], c, use_vlm, args.vlm_timeout)
    for c in pool:
        try:
            c["cosmos"] = futures[c["pool_rank"]].result(timeout=args.vlm_timeout + 30)
        except Exception as e:  # noqa: BLE001
            c["cosmos"] = {"text": "", "source": f"unavailable ({type(e).__name__})"}
        c["cosmos_parked"] = bool(PARKED_RE.search(c["cosmos"]["text"]))
        # The local Cosmos stand-in's parked/moving answers measured at chance on my labels (README), so by default they are
        # displayed but NOT used for ranking. --cosmos-triage re-enables the old demotion (x0.35) to test the hosted model.
        c["sev_final"] = round(c["sev"] * (0.35 if (c["cosmos_parked"] and args.cosmos_triage) else 1.0), 3)
    ex.shutdown(wait=False, cancel_futures=True)
    T["cosmos_s"] = round(time.time() - t0, 1)
    ordered = sorted(pool, key=lambda c: -c["sev_final"])
    cands = [c for c in ordered if c["sev_final"] >= MIN_SEV][:args.top]
    res["cands"] = cands
    res["pool"] = pool
    print(f"[{name}] final={[(c['t'], c['kind'], c['sev_final'], 'parked' if c['cosmos_parked'] else '') for c in cands]}", flush=True)

    # ---- stage 4: per-candidate evidence (parallel)
    t0 = time.time()

    def build(n_: int, c: dict) -> dict:
        d_ = out / f"cand_{n_}"
        d_.mkdir(exist_ok=True)
        t = c["t"]
        fr = pk.frame_at(src, min(t, meta["duration"] - 0.05))
        key = annotate(fr, tr, c["t_frame"], c["tid"], f"#{n_} {c['cls']}" if packets else c["cls"],
                       (40, 40, 235) if packets else (118, 126, 138))
        pk.save_jpg(key, d_ / "key.jpg")
        bc = best_crop(src, tr, c["tid"], c["t_frame"])
        crop_info = None
        if bc is not None:
            crop, tt, crop_info = bc
            pk.save_jpg(crop if crop.shape[1] >= 360 else cv2.resize(crop, None, fx=360 / crop.shape[1], fy=360 / crop.shape[1],
                                                                      interpolation=cv2.INTER_CUBIC), d_ / "crop.jpg")
        ts = [min(max(t + dtt, 0), meta["duration"] - 0.1) for dtt in (-2.0, -1.0, 0.0, 1.0, 2.0)]
        sb = [annotate(pk.frame_at(src, x), tr, x, c["tid"], "") for x in ts]
        pk.save_jpg(pk.sheet(sb, [("t%+.0f s" % (x - t)) for x in ts], cols=5, tile_w=380), d_ / "storyboard.jpg")
        a = max(0.0, t - CLIP_PRE)
        b = min(meta["duration"], a + CLIP_PRE + CLIP_POST)
        a = max(0.0, b - (CLIP_PRE + CLIP_POST))
        if packets and not args.no_clips:
            pk.cut(src, a, b, d_ / "clip.mp4", width=None)
        key_info = None
        if bc is not None:      # hero key frame: the sharpest un-clipped moment of this vehicle within +-1.2 s of the peak
            key_info = {k: crop_info[k] for k in ("x1", "y1", "x2", "y2")}
            key_info["annotated"] = annotate(crop_info["frame"], tr, bc[1], c["tid"], f"#{n_} {c['cls']}" if packets else c["cls"],
                                           (40, 40, 235) if packets else (118, 126, 138))
            key_info["t"] = bc[1]
            crop_info = {k: v for k, v in crop_info.items() if k != "frame"}
        return {"n": n_, "dir": d_, "win": (a, b), "crop_info": crop_info, "key": key, "key_info": key_info,
                "crop": None if bc is None else bc[0]}

    with cf.ThreadPoolExecutor(max_workers=5) as ex2:
        assets = list(ex2.map(lambda p: build(*p), list(enumerate(cands, 1))))
    T["assets_s"] = round(time.time() - t0, 1)
    descs = {a["n"]: cands[a["n"] - 1]["cosmos"] for a in assets}

    packets_out = []
    for a in assets:
        c = cands[a["n"] - 1]
        if packets:
            p = snap_packet(name, meta, sha, c, a["n"], a["win"], descs[a["n"]], ck, args, a["crop_info"])
            (a["dir"] / "packet.json").write_text(json.dumps(p, indent=1, ensure_ascii=False), encoding="utf-8")
            packets_out.append(p)
        c["clip"] = f"cand_{a['n']}/clip.mp4" if packets and not args.no_clips else None
        c["window"] = [round(a["win"][0], 1), round(a["win"][1], 1)]
        c["crop_px"] = None if not a["crop_info"] else f"{a['crop_info']['crop_w']}x{a['crop_info']['crop_h']}"
    T["total"] = round(time.time() - t_all, 1)
    res["top_text"] = descs.get(1, {}).get("text", "") if cands else ""

    top_ki = assets[0]["key_info"] if assets else None
    if assets and top_ki is None:                                 # no usable crop: fall back to the peak frame, whole-frame box
        top_ki = {"x1": 0, "y1": 0, "x2": meta["w"], "y2": meta["h"], "annotated": assets[0]["key"], "t": cands[0]["t"]}
    render_hero(out / "hero.jpg", name, meta, res, top_ki, assets[0]["crop"] if assets else None, ck, mode, T, packets,
                thumbs=[a_["crop"] if a_["crop"] is not None else a_["key"] for a_ in assets])
    write_report(out, name, meta, res, assets, packets_out, ck, mode, T, packets, tr)
    for c in pool:
        c.pop("_keyplain", None)
    summary = {"video": name, "duration_s": round(meta["duration"], 1), "resolution": f"{meta['w']}x{meta['h']}",
               "camera": mode, "packets_generated": packets, "verdict": res["verdict"], "clock": ck, "timings_s": T, "candidates": cands, "pool": pool,
               "source_sha256": sha}
    return summary


def write_report(out: Path, name: str, meta: dict, res: dict, assets: list, packets: list, ck: dict, mode: dict, T: dict,
                 packets_ok: bool, tr: pk.Tracks) -> None:
    rep = pk.Report(out / "report.html", f"Day 14: {name}", "Dashcam recording -> candidate incidents -> Operation Snap packet")
    rep.credit(name)
    rep.html(f'<figure><img src="hero.jpg"><figcaption>hero.jpg</figcaption></figure>')
    if not packets_ok:
        rep.html('<div class="card" style="border-color:#d64545"><b>Static camera detected.</b> No ego vehicle, so there is no '
                 '"my lane", and no Snap packet is generated. The rows below are what the same detector would flag at a '
                 "fixed viewpoint, shown only to document the failure mode.</div>")
    rep.kv({"verdict": " ".join(res["verdict"]["lines"]) + " - " + res["verdict"]["sub"],
            "footage": pk.mmss(meta["duration"])[:5], "analysis wall time": f"{T['total']:.0f} s",
            "camera": mode["mode"], "14-day clock": ck["text"] if packets_ok and res["cands"] else "not running (no packet)",
            "recording date source": ck["recorded_source"], "candidates": len(res["cands"])})
    spans = []
    rep.h2("Where the risk is")
    dur = meta["duration"]
    for n_, c in enumerate(res["cands"], 1):
        spans.append((c["t"] - 0.5, c["t"] + 0.5, "#e03a3a", f"#{n_} {c['kind']} at {pk.mmss(c['t'])}"))
    rep.timeline(spans, dur, "Red ticks = ranked candidates (hover for time).")
    rep.h2("Candidates")
    for a in assets:
        c = res["cands"][a["n"] - 1]
        d_ = a["dir"].name
        parts = [f'<div class="card"><h3 style="margin:0 0 6px">#{a["n"]}  {KIND_LABEL[c["kind"]]}  at {pk.mmss(c["t"])}'
                 f'  (geometry severity {c["sev"]:.2f}{", Cosmos text says parked (stand-in unreliable; %s)" % ("demoted to %.2f" % c["sev_final"] if c["sev_final"] != c["sev"] else "not used for ranking") if c.get("cosmos_parked") else ""}; relative-motion test, unreliable: {c["moving"]})</h3>']
        if c.get("clip"):
            parts.append(f'<video src="{d_}/clip.mp4" controls muted playsinline preload="metadata"></video>')
        parts.append(f'<img src="{d_}/storyboard.jpg" loading="lazy"><div style="display:flex;gap:12px;margin-top:8px;flex-wrap:wrap">'
                     f'<img src="{d_}/key.jpg" style="max-width:520px" loading="lazy">')
        if a["crop"] is not None:
            parts.append(f'<img src="{d_}/crop.jpg" style="max-height:260px;width:auto" loading="lazy">')
        parts.append("</div>")
        txt = c["cosmos"]["text"] or "(Cosmos description unavailable) " + rule_text(c)
        parts.append(f'<p><b>Cosmos:</b> {html.escape(txt)}<br><span class="sub">source: {html.escape(c["cosmos"]["source"])}'
                     f'. Plate: not read by this tool (crop {c["crop_px"]} px).</span></p>')
        if packets:
            p = packets[a["n"] - 1]
            parts.append("<details><summary>Snap packet (draft)</summary><pre>" + html.escape(
                json.dumps(p, indent=1, ensure_ascii=False)) + "</pre></details>")
        parts.append("</div>")
        rep.html("".join(parts))
    rep.h2("Timings")
    rep.json(T)
    rep.save()


# ------------------------------------------------------------------------------------------------------ main
def label_truth(results: list[dict]) -> dict:
    """Join candidates with the hand-made ground truth (ground_truth.json, from frames I viewed).
    near_field = a real vehicle really was within ~1-2 m (real / close_static / unclear); reportable = label 'real' only."""
    p = HERE / "ground_truth.json"
    if not p.exists():
        return {}
    gt = json.loads(p.read_text(encoding="utf-8"))
    stats = {}
    for r in results:
        g = gt.get(r["video"])
        if not g:
            continue
        if not g["events"] and "not a dashcam" not in g.get("note", ""):
            stats[r["video"]] = {"expected": "no incident: zero candidates above the severity floor",
                                 "final_candidates": len(r["candidates"]), "max_pool_severity": max([c["sev"] for c in r.get("pool", [])] or [0]),
                                 "correct": len(r["candidates"]) == 0}
            continue
        if "not a dashcam" in g.get("note", ""):
            stats[r["video"]] = {"expected": "refuse: no Snap packet (not a dashcam)", "camera_mode": r["camera"]["mode"],
                                 "packets_generated": r["packets_generated"], "correct": not r["packets_generated"]}
            continue
        for c in list(r["candidates"]) + list(r.get("pool", [])):
            lab = "no_event_found_at_this_time"
            for e in g["events"]:
                if e["t0"] <= c["t"] <= e["t1"]:
                    lab, c["eyeball"] = e["label"], e["note"]
            c["eyeball_label"] = lab
        lab = [c["eyeball_label"] for c in r["candidates"]]
        near = lambda L: sum(1 for x in L if x in ("real", "close_static", "unclear", "real_simulated"))  # noqa: E731
        rep = lambda L: sum(1 for x in L if x in ("real", "real_simulated"))  # noqa: E731
        geo = sorted(r.get("pool", []), key=lambda c: -c["sev"])[:5]     # ranking by geometry alone (no Cosmos)
        geo_lab = [c["eyeball_label"] for c in geo]
        stats[r["video"]] = {
            "labels_final_order": lab, "labels_geometry_only_order": geo_lab,
            "reportable_in_top3": rep(lab[:3]), "reportable_in_top5": rep(lab[:5]),
            "near_field_precision_at_3": round(near(lab[:3]) / max(1, len(lab[:3])), 2),
            "near_field_precision_at_5": round(near(lab[:5]) / max(1, len(lab[:5])), 2),
            "reportable_precision_at_5": round(rep(lab[:5]) / max(1, len(lab[:5])), 2),
            "geometry_only_reportable_in_top3": rep(geo_lab[:3]), "geometry_only_reportable_in_top5": rep(geo_lab[:5]),
            "n_candidates": len(lab),
            "top1_label": lab[0] if lab else None}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run the default test set")
    ap.add_argument("--video", help="catalogue name or path")
    ap.add_argument("--recorded", help="date of the recording YYYY-MM-DD (default: today)")
    ap.add_argument("--today", help="override today's date (for demos of the clock)")
    ap.add_argument("--start-time", help="wall-clock time the recording started, HH:MM:SS")
    ap.add_argument("--location", help="free text for the packet")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--cosmos-triage", action="store_true", help="demote candidates Cosmos calls parked (off: stand-in is at chance)")
    ap.add_argument("--refresh-ego", action="store_true", help="recompute the optical-flow cache (cold timing)")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--vlm-timeout", type=float, default=150.0)
    args = ap.parse_args()
    names = DEFAULT_SET if args.all else [args.video] if args.video else []
    if not names:
        ap.error("give --all or --video NAME")
    t0 = time.time()
    results = [run_video(n, args) for n in names]
    stats = label_truth(results)
    if len(names) == len(DEFAULT_SET) or (HERE / "results.json").exists() is False:
        pass
    allres = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "wall_s": round(time.time() - t0, 1),
              "videos": {r["video"]: r for r in results}, "eyeball": stats,
              "clock_examples": [clock(dt.date(2026, 10, 10), dt.date(2026, 10, 17)), clock(dt.date(2026, 10, 15), dt.date(2026, 10, 17)),
                                 clock(dt.date(2026, 9, 28), dt.date(2026, 10, 17))]}
    old = HERE / "results.json"
    if old.exists():                       # always merge: sim_cutin / ctrl_highway are run separately
        try:
            prev = json.loads(old.read_text(encoding="utf-8"))
            prev["videos"].update(allres["videos"])
            prev["eyeball"].update(stats)
            allres = {**prev, "generated": allres["generated"]}
        except Exception:  # noqa: BLE001
            pass
    pk.write_results(old, allres)
    print(f"done in {time.time() - t0:.0f} s -> {old}")


if __name__ == "__main__":
    main()
