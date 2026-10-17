"""PINCH POINT REGISTER: zone occupancy back-test, trigger sweep, plan-entry export.

Core loop (no GPU, no VLM needed): cached YOLO26s+ByteTrack person detections -> feet (box bottom-centre) inside a
zone polygon -> people-in-zone per second -> back-test the trigger ">= N people for >= S s" over the whole recording
-> sweep N, recommend a threshold (95th percentile of occupancy) -> export a plan entry (Markdown + JSON) with
evidence frames at the peaks.  Cosmos (local stand-in) only *suggests* labels for the zone; nothing depends on it.

    run.py --all                       default test set (yonge_dundas people_detection store_aisle_detection retail_cctv)
    run.py --video yonge_dundas        one clip
    run.py --video X --zone "0,.55;1,.55;1,1.03;0,1.03" --n 12 --s 3 --area-m2 400 --density-limit 2
    run.py --all --xcheck              also re-count the peak frames with the independent larger detector (GPU)
    run.py --video X --preview         only draw the zone on a reference frame
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
DEFAULT_SET = ["retail_cctv", "yonge_dundas", "people_detection", "store_aisle_detection"]   # retail_cctv leads: fixed camera
GAP_MERGE_S = 0.0       # LITERAL rule by default: every second of an episode must be >= N (use --gap 1 to forgive 1 s dips)
DEFAULT_S = 3.0         # sustain time (s). Real deployments use 30-120 s; clips are only 50-111 s long
DEFAULT_DENSITY_LIMIT = 2.0   # people per m2 used for the capacity-based alternative (assumption, see README)

# Zone polygons are NORMALISED (x/width, y/height), chosen by looking at a reference frame of each clip. Values may
# exceed 1.0 so that people whose feet are cut off by the bottom edge of the frame still count.  area_m2 is an
# ESTIMATE eyeballed from the scene (shop-door / lane-width scale), not a survey: density numbers inherit that error.
ZONES: dict[str, dict[str, Any]] = {
    "yonge_dundas": {
        "name": "Scramble crossing (roadway inside the kerb)",
        "poly": [(0.0, 0.62), (0.35, 0.62), (0.70, 0.625), (1.0, 0.63), (1.0, 1.03), (0.0, 1.03)],
        "area_m2": 450.0, "site": "Toronto, Yonge-Dundas diagonal crossing (pedestrian-eye view)",
        "fixed_camera": False},
    "people_detection": {
        "name": "Office corridor floor",
        "poly": [(0.10, 0.52), (0.88, 0.52), (0.93, 1.03), (0.05, 1.03)],
        "area_m2": 30.0, "site": "Office hallway, overhead wide-angle camera"},
    "store_aisle_detection": {
        "name": "Main shop aisle",
        "poly": [(0.30, 0.20), (0.62, 0.20), (0.67, 0.29), (0.78, 0.69), (0.86, 1.03), (0.0, 1.03), (0.0, 0.75), (0.06, 0.60), (0.15, 0.42)],
        "area_m2": 40.0, "site": "Crockery store, aisle camera"},
    "retail_cctv": {
        "name": "Till and queue area",
        "poly": [(0.0, 0.28), (0.46, 0.20), (0.56, 0.62), (0.45, 1.03), (0.0, 1.03)],
        "area_m2": 25.0, "site": "Convenience store, ceiling CCTV"},
}

# ------------------------------------------------------------------------------------------------ zone counting


def parse_zone(s: str) -> list[tuple[float, float]]:
    return [tuple(float(v) for v in p.split(",")) for p in s.replace(" ", "").split(";") if p]  # type: ignore[misc]


def poly_px(poly: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    return np.array([(x * w, y * h) for x, y in poly], dtype=np.float32)


def in_zone_mask(rows: pd.DataFrame, poly: list[tuple[float, float]], w: int, h: int) -> np.ndarray:
    """Feet = box bottom-centre (y clipped to the frame so cut-off people still have feet 'at the edge')."""
    if rows.empty:
        return np.zeros(0, bool)
    feet = np.c_[rows["cx"].to_numpy(), np.minimum(rows["y2"].to_numpy(), h - 1.0)]
    return MPath(poly_px(poly, w, h)).contains_points(feet)


def poly_area_px(poly: list[tuple[float, float]], w: int, h: int) -> float:
    p = poly_px(poly, w, h)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def zone_series(tr: pk.Tracks, poly: list[tuple[float, float]], conf: float = 0.0) -> dict[str, Any]:
    """Per-sampled-frame and per-second (median of the second's frames) people-in-zone counts."""
    p = tr.of("person")
    p = p[p["conf"] >= conf]
    inz = in_zone_mask(p, poly, tr.w, tr.h)
    nframes = int(round(tr.duration * tr.src_fps))
    grid = np.arange(0, nframes, tr.stride)
    cnt = pd.Series(inz.astype(int), index=p["f"].to_numpy()).groupby(level=0).sum() if len(p) else pd.Series(dtype=int)
    allcnt = p.groupby("f").size() if len(p) else pd.Series(dtype=int)
    frame_cnt = cnt.reindex(grid, fill_value=0).astype(int)
    frame_all = allcnt.reindex(grid, fill_value=0).astype(int)
    t = grid / tr.src_fps
    sec = np.floor(t).astype(int)
    nsec = int(math.ceil(tr.duration))
    per_sec = pd.Series(frame_cnt.to_numpy()).groupby(sec).median().reindex(range(nsec), fill_value=0)
    per_sec = np.floor(per_sec.to_numpy() + 0.5).astype(int)  # median of ints can be x.5 -> round half up
    return {"f": grid, "t": t, "frame_count": frame_cnt.to_numpy(), "frame_all": frame_all.to_numpy(),
            "sec": per_sec, "inz_rows": p.assign(in_zone=inz)}


# ---------------------------------------------------------------------------------------------- trigger back-test


def backtest(sec: np.ndarray, n: int, s: float, gap: float | None = None) -> dict[str, Any]:
    """Trigger '>= n people for >= s s' on a 1 Hz series. Episodes = runs of seconds with count >= n; runs separated
    by <= gap seconds are merged (default gap = 0: literal, no dips allowed); an episode FIRES once it has lasted
    >= s s (fire time = start + s, i.e. when the sustain condition is first satisfied)."""
    gap = GAP_MERGE_S if gap is None else gap
    above = sec >= n
    runs: list[list[int]] = []
    i, L = 0, len(sec)
    while i < L:
        if above[i]:
            j = i
            while j + 1 < L and above[j + 1]:
                j += 1
            if runs and i - runs[-1][1] - 1 <= gap:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j + 1
        else:
            i += 1
    fires = []
    for a, b in runs:
        dur = b - a + 1
        if dur >= s:
            seg = sec[a:b + 1]
            pk_i = a + int(np.argmax(seg))
            fires.append({"start": float(a), "fire_at": float(a + s), "end": float(b + 1), "duration_s": float(dur),
                          "peak": int(seg.max()), "peak_t": float(pk_i), "mean": round(float(seg.mean()), 2)})
    over = float(above.sum())
    return {"n": int(n), "s": float(s), "gap_s": float(gap), "fires": fires, "n_fires": len(fires), "time_above_s": over,
            "time_above_pct": round(100 * over / max(L, 1), 1),
            "longest_s": max([f["duration_s"] for f in fires], default=0.0)}


def sweep(sec: np.ndarray, s: float, gap: float | None = None) -> list[dict[str, Any]]:
    top = int(sec.max()) if len(sec) else 0
    return [{k: v for k, v in backtest(sec, n, s, gap).items() if k != "fires"} for n in range(1, top + 2)]


def max_firing_n(sw: list[dict[str, Any]]) -> int | None:
    """Highest N for which the trigger still fires at least once (the recording's 'sustained peak')."""
    ns = [r["n"] for r in sw if r["n_fires"] > 0]
    return max(ns) if ns else None


def recommend_n(sec: np.ndarray) -> dict[str, Any]:
    """Data-driven starting point: the 95th percentile of the 1 Hz occupancy over the whole recording (rounded up)."""
    p95 = float(np.percentile(sec, 95)) if len(sec) else 0.0
    return {"rule": "ceil(P95 of per-second people-in-zone over the recording)", "p95": round(p95, 2),
            "n": max(1, int(math.ceil(p95))), "p50": float(np.percentile(sec, 50)) if len(sec) else 0.0,
            "mean": round(float(sec.mean()), 2) if len(sec) else 0.0}


def top_peaks(sec: np.ndarray, k: int = 3, sep: float = 8.0) -> list[int]:
    """Seconds of the k highest distinct peaks (>= sep s apart)."""
    order = np.argsort(-sec, kind="stable")
    out: list[int] = []
    for i in order:
        if sec[i] <= 0:
            break
        if all(abs(int(i) - o) >= sep for o in out):
            out.append(int(i))
        if len(out) >= k:
            break
    return sorted(out)


# ---------------------------------------------------------------------------------------------------- drawing

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for f in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


C_BG, C_CARD, C_LINE = (13, 17, 23), (22, 29, 38), (45, 56, 70)
C_FG, C_MUTED = (236, 240, 244), (140, 154, 168)
C_RED, C_AMBER, C_GREEN, C_BLUE = (239, 68, 68), (245, 166, 35), (52, 199, 120), (88, 166, 255)


def pick_frame_for_second(sz: dict[str, Any], sec_i: int) -> int:
    """Index (into the sampled grid) of the frame inside second `sec_i` whose count equals that second's value."""
    idx = np.where(np.floor(sz["t"]).astype(int) == sec_i)[0]
    if len(idx) == 0:
        return 0
    target = sz["sec"][sec_i]
    best = idx[np.argmin(np.abs(sz["frame_count"][idx] - target) * 1000 + np.abs(idx - idx.mean()))]
    return int(best)


def render_overlay(frame: np.ndarray, tr: pk.Tracks, sz: dict[str, Any], poly, gi: int, *, width: int = 960,
                   boxes: bool = True, badge: str | None = None, caption: str | None = None) -> np.ndarray:
    """Frame with the zone filled, counted people boxed green (feet dots), others thin grey; big count badge."""
    h0, w0 = frame.shape[:2]
    s = width / w0
    img = cv2.resize(frame, (width, int(h0 * s)), interpolation=cv2.INTER_AREA)
    pts = (poly_px(poly, w0, h0) * s).astype(np.int32)
    # everything OUTSIDE the zone is dimmed and desaturated, the zone itself gets an amber tint: what is counted is obvious
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    grey = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    dim = cv2.addWeighted(grey, 0.40, np.zeros_like(img), 0, 0)
    tint = cv2.addWeighted(img, 0.74, np.full_like(img, (40, 160, 245)), 0.26, 0)
    img = np.where(mask[..., None] > 0, tint, dim)
    cv2.polylines(img, [pts], True, (60, 190, 255), 3, cv2.LINE_AA)
    f = int(sz["f"][gi])
    if boxes:
        d = sz["inz_rows"]
        d = d[d["f"] == f]
        for r in d.itertuples():
            b = [r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s]
            foot = (int(r.cx * s), int(min(r.y2, h0 - 1) * s))
            if r.in_zone:
                # feet within ~12 px (frame scale) of the line: counted, but flagged amber as boundary-sensitive
                edge = abs(cv2.pointPolygonTest(pts.reshape(-1, 1, 2), foot, True)) < 12 * width / 960
                col = (40, 170, 255) if edge else (90, 230, 120)
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                cv2.circle(img, foot, 5, (60, 60, 255), -1)
            else:
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                for xx in range(x1, x2, 10):   # dashed grey box = detected but NOT counted
                    cv2.line(img, (xx, y1), (min(xx + 5, x2), y1), (170, 170, 170), 1)
                    cv2.line(img, (xx, y2), (min(xx + 5, x2), y2), (170, 170, 170), 1)
                for yy in range(y1, y2, 10):
                    cv2.line(img, (x1, yy), (x1, min(yy + 5, y2)), (170, 170, 170), 1)
                    cv2.line(img, (x2, yy), (x2, min(yy + 5, y2)), (170, 170, 170), 1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    n = int(sz["frame_count"][gi])
    txt = badge or f"{n} IN ZONE"
    fnt = _font(max(22, width // 22), True)
    tw = dr.textlength(txt, font=fnt)
    dr.rounded_rectangle((12, 12, 12 + tw + 28, 12 + fnt.size + 18), 10, fill=(10, 14, 20), outline=(60, 190, 255), width=3)
    dr.text((26, 18), txt, font=fnt, fill=(255, 255, 255))
    if boxes:
        lf = _font(max(12, width // 70), True)
        items = [("counted", (90, 230, 120)), ("counted, foot on the line", (255, 170, 40)), ("outside: not counted", (190, 190, 190))]
        lw = max(dr.textlength(t, font=lf) for t, _ in items) + 34
        x0, y0 = pil.width - lw - 12, 12
        dr.rounded_rectangle((x0, y0, x0 + lw, y0 + len(items) * (lf.size + 7) + 8), 8, fill=(10, 14, 20))
        for k, (t, c) in enumerate(items):
            yy = y0 + 8 + k * (lf.size + 7)
            dr.rectangle((x0 + 8, yy + 3, x0 + 20, yy + lf.size - 1), outline=c, width=2)
            dr.text((x0 + 26, yy), t, font=lf, fill=(225, 230, 236))
    if caption:
        f2 = _font(max(14, width // 50))
        cw = dr.textlength(caption, font=f2)
        dr.rounded_rectangle((12, pil.height - f2.size - 26, 12 + cw + 20, pil.height - 10), 6, fill=(10, 14, 20))
        dr.text((22, pil.height - f2.size - 21), caption, font=f2, fill=(220, 226, 232))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def rule_gap(bt: dict[str, Any]) -> str:
    g = bt.get("gap_s", 0.0)
    return "every second" if g == 0 else f"{g:g} s dips allowed"


def timeline_chart(sec: np.ndarray, bt: dict[str, Any], peaks: list[int], width_px: int = 1232, height_px: int = 230):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    bg, card, fg, mut = "#0d1117", "#161d26", "#ecf0f4", "#8c9aa8"
    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.045, 0.19, 0.945, 0.63], facecolor=card)
    x = np.arange(len(sec) + 1)
    y = np.r_[sec, sec[-1:]] if len(sec) else np.zeros(1)
    ax.fill_between(x, 0, y, step="post", color="#58a6ff", alpha=0.35, lw=0)
    ax.step(x, y, where="post", color="#58a6ff", lw=1.8)
    for k, f in enumerate(bt["fires"], 1):
        ax.axvspan(f["start"], f["end"], color="#ef4444", alpha=0.30, lw=0)
        ax.text((f["start"] + f["end"]) / 2, max(sec.max(), bt["n"]) * 1.18 + 0.4, f"#{k}", color="#ff8a8a", ha="center",
                va="bottom", fontsize=10.5, fontweight="bold")
    ax.axhline(bt["n"] - 0.5, color="#f5a623", ls="--", lw=1.6)
    for p in peaks:
        ax.plot([p + 0.5], [sec[p]], marker="v", color="#ffffff", ms=7, zorder=5)
        ax.text(p + 0.5, sec[p] + max(sec.max(), 1) * 0.07, f"{sec[p]}", color="#ffffff", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    top = max(sec.max(), bt["n"]) * 1.18 + 0.4
    ax.set_xlim(0, max(len(sec), 1)); ax.set_ylim(0, top + top * 0.18)
    ax.tick_params(colors=mut, labelsize=9.5)
    for sp in ax.spines.values():
        sp.set_color("#2d3846")
    ax.grid(axis="y", color="#2d3846", lw=0.6)
    ax.set_xticks(np.linspace(0, len(sec), 9))
    ax.set_xticklabels([pk.mmss(v).split(".")[0] for v in np.linspace(0, len(sec), 9)])
    fig.text(0.045, 0.9, "PEOPLE IN ZONE PER SECOND", color=fg, fontsize=11.5, fontweight="bold", va="center")
    fig.text(0.36, 0.9, f"- - trigger: ≥ {bt['n']} people for ≥ {bt['s']:g} s, {rule_gap(bt)}", color="#f5a623",
             fontsize=10.5, fontweight="bold", va="center")
    fig.text(0.99, 0.9, "red = trigger would have fired   \u25BC = peak evidence frames", color=mut, fontsize=9.5, ha="right",
             va="center")
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def make_hero(path: Path, res: dict[str, Any], peak_img: np.ndarray, chart: np.ndarray) -> None:
    W, M = 1280, 24
    fw = 792
    pf = cv2.resize(peak_img, (fw, int(peak_img.shape[0] * fw / peak_img.shape[1])), interpolation=cv2.INTER_AREA)
    fh = pf.shape[0]
    top = 108
    H = top + fh + 14 + chart.shape[0] + 22
    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)
    # header
    d.text((M, 14), "PINCH POINT REGISTER", font=_font(32, True), fill=C_FG)
    sub = f"{res['zone']['name']}  \u00b7  {res['site']}  \u00b7  {pk.mmss(res['duration_s']).split('.')[0]} recorded"
    sub_font = _font(17)
    while d.textlength(sub, font=sub_font) > 800 and len(sub) > 20:
        sub = sub[:-4].rstrip() + "..."
    d.text((M, 56), sub, font=sub_font, fill=C_MUTED)
    bt = res["backtest"]
    sparse = res["peak_1hz"] < 6
    if bt["n_fires"] > 0:
        stamp, col = f"WOULD HAVE FIRED {bt['n_fires']}\u00d7", (C_AMBER if sparse else C_RED)
    else:
        stamp, col = (f"NO FIRE AT N≥{bt['n']}", C_BLUE) if res.get("max_firing_n") else ("NEVER FIRED", C_GREEN)
    sf = _font(30, True)
    tw = d.textlength(stamp, font=sf)
    lay = Image.new("RGBA", (int(tw) + 60, 64), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    ld.rounded_rectangle((0, 0, lay.width - 1, lay.height - 1), 12, outline=col + (255,), width=5, fill=col + (40,))
    ld.text((30, 8), stamp, font=sf, fill=col + (255,))
    lay = lay.rotate(3, expand=True, resample=Image.BICUBIC)
    img.paste(lay, (W - M - lay.width, 8), lay)
    rule = f"rule: ≥ {bt['n']} people for ≥ {bt['s']:g} s, {rule_gap(bt)}"
    d.text((W - M - d.textlength(rule, font=_font(13, True)), 77), rule, font=_font(13, True), fill=C_MUTED)
    notes = []
    if sparse:
        notes.append(f"sparse zone (peak {res['peak_1hz']}): not a pinch point")
    if not res["zone"].get("fixed_camera", True):
        notes.append("MOVING CAMERA: not a fixed pinch point (limitation demo)")
    if notes:
        note = "  |  ".join(notes)
        d.text((M, 82), note, font=_font(13, True), fill=C_AMBER)
    # frame
    img.paste(Image.fromarray(cv2.cvtColor(pf, cv2.COLOR_BGR2RGB)), (M, top))
    d.rectangle((M - 1, top - 1, M + fw, top + fh), outline=C_LINE, width=1)
    # KPI cards
    px = M + fw + 20
    pw = W - M - px
    cw, ch = (pw - 12) // 2, 96
    mf = res.get("max_firing_n")
    sw0 = {r["n"]: r for r in res["sweep"]}
    kp = [("PEAK IN ZONE", f"{res['peak_1hz']}", f"at {pk.mmss(res['peak_t']).split('.')[0]}  (area is a guess: no density)", C_BLUE),
          ("HIGHEST N THAT STILL FIRES", f"N≥{mf}" if mf else "none",
           (f"N≥{mf + 1}: {sw0[mf + 1]['n_fires']} fires" if mf and (mf + 1) in sw0 else "no sustained crowd"), C_RED if mf else C_MUTED),
          ("TIME OVER TRIGGER", f"{bt['time_above_s']:.0f} s", f"{bt['time_above_pct']:.0f}% of recording", C_AMBER),
          ("LONGEST EPISODE", f"{bt['longest_s']:.0f} s", f"{bt['n_fires']} fire(s) at N≥{bt['n']}", col)]
    for k, (a, b, c, cc) in enumerate(kp):
        x, y = px + (k % 2) * (cw + 12), top + (k // 2) * (ch + 12)
        d.rounded_rectangle((x, y, x + cw, y + ch), 10, fill=C_CARD, outline=C_LINE)
        d.rectangle((x, y + 10, x + 4, y + ch - 10), fill=cc)
        d.text((x + 16, y + 8), a, font=_font(12, True), fill=C_MUTED)
        d.text((x + 16, y + 26), b, font=_font(40, True), fill=C_FG)
        d.text((x + 16, y + 74), c, font=_font(12), fill=C_MUTED)
    # sweep table
    y0 = top + 2 * (ch + 12) + 2
    d.text((px, y0), "THRESHOLD SWEEP  (N \u2192 fires, time over)", font=_font(12, True), fill=C_MUTED)
    d.text((px, y0 + 15), "P95 rule fires on the busiest ~5% by design; the sweep is the insight", font=_font(11), fill=C_AMBER)
    rec = res["recommend"]["n"]
    sw = {r["n"]: r for r in res["sweep"]}
    pick = sorted({n for n in range(rec - 2, rec + 3) if n in sw} | {bt["n"]} | ({mf, mf + 1} if mf else set()))
    pick = [n for n in pick if n in sw][:7]
    rh = max(22, int((fh + top - y0 - 52) / max(len(pick), 1)))
    rh = min(rh, 28)
    for k, n in enumerate(pick):
        r = sw[n]
        yy = y0 + 34 + k * rh
        sel = n == bt["n"]
        if sel:
            d.rounded_rectangle((px - 4, yy - 2, px + pw, yy + rh - 4), 6, fill=(40, 52, 68), outline=C_AMBER)
        tag = "  \u2190 recommended (P95)" if n == rec else ""
        d.text((px + 6, yy), f"N \u2265 {n}", font=_font(15, True), fill=C_FG)
        d.text((px + 78, yy), f"{r['n_fires']} fire(s)", font=_font(15), fill=C_RED if r["n_fires"] else C_MUTED)
        d.text((px + 168, yy), f"{r['time_above_s']:.0f} s over", font=_font(15), fill=C_MUTED)
        if tag:
            d.text((px + 258, yy + 1), "\u2190 P95", font=_font(13, True), fill=C_AMBER)
    # chart
    cy = top + fh + 14
    img.paste(Image.fromarray(chart), (M, cy))
    v = pk.videos().get(res["video"], {})
    d.text((M, H - 20), "Prototype. Local stand-ins: YOLO26s+ByteTrack, duckdb, Cosmos3-Edge. Footage: " +
           f"{str(v.get('author', ''))[:60]}, {v.get('license', '')}", font=_font(11), fill=(100, 112, 126))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=92)


# ---------------------------------------------------------------------------------------------------- cosmos


def cosmos_label(frame: np.ndarray, poly, w: int, h: int, site: str) -> dict[str, Any]:
    """Cosmos3 (local stand-in) suggestion for the zone. Advisory only; returns {} if the VLM is down."""
    img = frame.copy()
    cv2.polylines(img, [poly_px(poly, w, h).astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA)
    prompt = ("This is a CCTV frame. The yellow polygon marks a zone a venue crowd manager wants to monitor. "
              "Return JSON with keys: zone_name (max 5 words, what the place is), zone_type (one of: crossing, queue, "
              "corridor, doorway, aisle, platform, stairs, open floor, other), crowd_risks (list of up to 3 short phrases), "
              "suggested_controls (list of up to 3 short phrases). Describe only what is visible.")
    t0 = time.time()
    j = pk.vlm_json(prompt, [img], max_tokens=260)
    if isinstance(j, dict):
        j["_seconds"] = round(time.time() - t0, 1)
        return j
    return {}


# ---------------------------------------------------------------------------------------------------- plan entry


def plan_markdown(res: dict[str, Any]) -> str:
    z, bt, rec = res["zone"], res["backtest"], res["recommend"]
    cs = res.get("cosmos") or {}
    L = [f"# Pinch-point plan entry {res['plan_id']}", "",
         "Status: DRAFT generated from recorded footage; needs sign-off by the venue's crowd-safety owner.", "",
         "## Zone", f"- Name: {z['name']} (site: {res['site']})",
         *(["- **WARNING: moving (handheld) camera; the polygon is image-plane only, so this is not a fixed pinch point. Limitation demo only.**"]
           if not z.get("fixed_camera", True) else []),
         f"- Camera clip: `{res['video']}` ({res['frame_w']}x{res['frame_h']}, {res['duration_s']:.0f} s)",
         f"- Polygon (normalised x,y): `{z['poly_text']}`",
         f"- Area: ~{z['area_m2']:g} m2 ({z['area_basis']})",
         f"- Counting rule: person detected, box bottom-centre (feet) inside the polygon; 1 Hz median.", "",
         "## Trigger", f"- **>= {bt['n']} people in zone for >= {bt['s']:g} s, {rule_gap(bt)}** ({res['trigger_basis']}); the alert is raised once the "
         f"condition has held for S seconds (start + S).",
         f"- Highest N that still fires at least once on this recording: {res['max_firing_n']}." if res.get("max_firing_n") else
         "- No N fires on this recording with these S / gap settings.",
         "- Note: the P95 reference fires on the busiest ~5% of any recording by construction; read the sweep, not the stamp.",
         f"- Data-driven reference: P95 occupancy = {rec['p95']} -> N = {rec['n']}; median {rec['p50']:g}, mean {rec['mean']}.",
         f"- Capacity-based alternative: N = {res['capacity_n']} ({res['density_limit']:g} people/m2 x {z['area_m2']:g} m2, "
         f"area is an estimate); the recording peaks at {100 * res['peak_1hz'] / max(res['capacity_n'], 1):.0f}% of it, so a "
         "capacity rule would never have fired here.", "",
         "## Back-test over the whole recording",
         f"- Fired {bt['n_fires']} time(s); time at or above N: {bt['time_above_s']:.0f} s ({bt['time_above_pct']:.0f}%); "
         f"longest episode {bt['longest_s']:.0f} s; peak {res['peak_1hz']} people at {pk.mmss(res['peak_t'])} "
         f"(~{(res.get('peak_density_per_m2') or 0):.2f} people/m2, estimated area)."]
    for k, f in enumerate(bt["fires"], 1):
        L.append(f"  - #{k}: {pk.mmss(f['start'])} to {pk.mmss(f['end'])} ({f['duration_s']:.0f} s), peak {f['peak']}, "
                 f"alert at {pk.mmss(f['fire_at'])}")
    L += ["", "| N | fires | time at/above N (s) | % of recording |", "|---|---|---|---|"]
    for r in res["sweep"]:
        mark = " **<- chosen**" if r["n"] == bt["n"] else (" (P95)" if r["n"] == rec["n"] else "")
        L.append(f"| {r['n']}{mark} | {r['n_fires']} | {r['time_above_s']:.0f} | {r['time_above_pct']:.0f} |")
    L += ["", "## Evidence frames (peaks)"]
    for e in res["evidence"]:
        L.append(f"- {pk.mmss(e['t'])}: {e['count']} people in zone -> `{e['file']}`")
        L.append(f"  ![peak {pk.mmss(e['t'])}]({e['file']})")
    L += ["", "## Cosmos suggestions (local stand-in model; unverified, a human must confirm)"]
    if cs:
        L += [f"- Suggested name: {cs.get('zone_name', '?')} (type: {cs.get('zone_type', '?')})",
              f"- Risks: {'; '.join(map(str, cs.get('crowd_risks', []) or []))}",
              f"- Controls: {'; '.join(map(str, cs.get('suggested_controls', []) or []))}"]
    else:
        L.append("- (no suggestion: VLM unavailable)")
    L += ["", "## Actions (to be completed by the operator)",
          "- [ ] Staff / steward response when trigger fires: _TBD_",
          "- [ ] Who is notified and how: _TBD_",
          "- [ ] Physical control (barrier, one-way, hold at entry): _TBD_",
          "- [ ] Escalation if still above N after _TBD_ minutes: _TBD_",
          "- [ ] Review date and owner: _TBD_", "",
          "## Assumptions and limits"]
    L += [f"- {x}" for x in res["limits"]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- pipeline


def vastdb_check(video: str, sec: np.ndarray, bt: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the headline numbers with real SQL against the duckdb VastDB stand-in (guards the python maths).
    VastDB dialect subset: no IN / ILIKE / GROUP BY / JOIN / window functions, so only filters and aggregates."""
    df = pd.DataFrame({"video": video, "sec": np.arange(len(sec)), "n_in_zone": sec})
    con = pk.vastdb({"zone_occupancy": df})
    n = bt["n"]
    r = con.execute(f"SELECT COUNT(*), MAX(n_in_zone), SUM(n_in_zone) FROM zone_occupancy WHERE n_in_zone >= {n}").fetchone()
    mx = con.execute("SELECT MAX(n_in_zone) FROM zone_occupancy").fetchone()[0]
    return {"sql": f"SELECT COUNT(*) FROM zone_occupancy WHERE n_in_zone >= {n}", "seconds_over": int(r[0]),
            "peak": int(mx), "matches_python": int(r[0]) == int(bt["time_above_s"]) and int(mx) == int(sec.max())}


def run_video(name: str, args: argparse.Namespace) -> dict[str, Any]:
    t_start = time.time()
    z = ZONES.get(name, {"name": "Zone", "poly": [(0, 0), (1, 0), (1, 1), (0, 1)], "area_m2": 50.0, "site": name})
    poly = parse_zone(args.zone) if args.zone and not args.all else z["poly"]
    area = args.area_m2 if args.area_m2 and not args.all else z["area_m2"]
    tr = pk.tracks(name)
    d_out = OUT / name
    d_out.mkdir(parents=True, exist_ok=True)
    for stale in list(d_out.glob("evidence_*.jpg")) + [d_out / "zone_preview.jpg"]:
        stale.unlink(missing_ok=True)
    sz = zone_series(tr, poly, conf=args.conf)
    sec = sz["sec"]
    s = args.s if args.s is not None else DEFAULT_S
    rec = recommend_n(sec)
    n = args.n if (args.n and not args.all) else rec["n"]
    gap = args.gap if args.gap is not None else GAP_MERGE_S
    bt = backtest(sec, n, s, gap)
    sw = sweep(sec, s, gap)
    mf = max_firing_n(sw)
    peaks = top_peaks(sec, 3)
    pk_sec = int(np.argmax(sec)) if len(sec) else 0
    meta = pk.info(name)
    lim = args.density_limit
    cap_n = max(1, int(math.floor(area * lim)))
    peak_dens = float(sec.max()) / area if area else None

    # evidence frames: peaks of fires first (distinct), then the top overall peaks
    ev_secs: list[int] = []
    for f in bt["fires"]:
        ps = int(f["peak_t"])
        if all(abs(ps - e) >= 6 for e in ev_secs):
            ev_secs.append(ps)
    for ps in peaks:
        if len(ev_secs) >= 3:
            break
        if all(abs(ps - e) >= 6 for e in ev_secs):
            ev_secs.append(ps)
    ev_secs = sorted(ev_secs[:4])
    evidence = []
    for ps in ev_secs:
        gi = pick_frame_for_second(sz, ps)
        t = float(sz["t"][gi])
        fr = pk.frame_at(name, t)
        img = render_overlay(fr, tr, sz, poly, gi, width=960,
                             caption=f"{name}  {pk.mmss(t)}  frame {int(sz['f'][gi])}  1 Hz median {int(sec[ps])}")
        fn = f"evidence_{int(round(t * 10)):04d}.jpg"
        pk.save_jpg(img, d_out / fn)
        evidence.append({"t": round(t, 2), "f": int(sz["f"][gi]), "second": ps, "count": int(sz["frame_count"][gi]),
                         "count_1hz": int(sec[ps]), "file": fn})
    # hero frame: the overall peak
    gi = pick_frame_for_second(sz, pk_sec)
    t_pk = float(sz["t"][gi])
    fr_pk = pk.frame_at(name, t_pk)
    peak_img = render_overlay(fr_pk, tr, sz, poly, gi, width=960,
                              caption=f"peak  {pk.mmss(t_pk)}  \u00b7  zone people = box bottom-centre inside the polygon")
    pk.save_jpg(render_overlay(pk.frame_at(name, 0.0 if tr.duration < 1 else min(t_pk, tr.duration - 0.1)), tr, sz, poly, gi,
                               width=960, boxes=False, badge="ZONE"), d_out / "zone_reference.jpg")
    cosmos = {} if args.no_vlm else cosmos_label(fr_pk, poly, tr.w, tr.h, z["site"])
    hero_res = {
        "video": name, "site": z["site"], "plan_id": f"PP-{name.upper()[:12]}-01", "duration_s": round(tr.duration, 1),
        "frame_w": tr.w, "frame_h": tr.h, "fps": round(tr.src_fps, 2), "sample_fps": round(tr.fps, 2),
        "credit": pk.credit(name), "peak_1hz": int(sec.max()), "peak_t": float(pk_sec),
        "peak_frame_max": int(sz["frame_count"].max()), "peak_density_per_m2": round(peak_dens, 3) if peak_dens else None,
        "zone": {"name": z["name"], "poly": [list(map(float, p)) for p in poly], "poly_text": ";".join(f"{x:g},{y:g}" for x, y in poly),
                 "area_m2": area, "area_basis": "ESTIMATE eyeballed from the scene, +/-50%",
                 "fixed_camera": bool(z.get("fixed_camera", True))},
        "zone_area_share_of_frame": round(poly_area_px(poly, tr.w, tr.h) / (tr.w * tr.h), 3),
        "recommend": rec, "backtest": bt, "sweep": sw, "max_firing_n": mf, "capacity_n": cap_n, "density_limit": lim,
        "trigger_basis": "chosen N = recommended P95 value" if n == rec["n"] else "N set on the command line",
        "evidence": evidence, "cosmos": cosmos,
        "person_detections_total": int(len(tr.of("person"))), "frames_sampled": int(len(sz["f"])),
        "zone_person_seconds": int(sec.sum()), "occupancy_p50": rec["p50"], "occupancy_mean": rec["mean"],
        "vastdb_sql_check": vastdb_check(name, sec, bt),
        "occupancy_1hz": [int(v) for v in sec],
        "limits": [
            "Counts come from YOLO26s+ByteTrack detections at ~12-15 fps and 640 px input: small / distant / heavily occluded "
            "people are missed, so counts are a lower bound in dense crowds.",
            "Zone is a fixed polygon on the image plane; it is only valid for a fixed camera (the Yonge-Dundas clip is "
            "handheld and walks with the crowd: its zone counts mostly people near the camera operator, so it is NOT a pinch point).",
            "Feet are approximated by box bottom-centre; cut-off people at the frame edge are counted as standing at the edge.",
            "Area (m2) and therefore density is an eyeballed estimate, not a survey; use measured floor area before relying on it.",
            f"Clips are short (<= 2 min); the sustain time S={s:g} s is scaled down from realistic 30-120 s.",
            "P95 rule picks N from the recording's own busiest 5%, so it is a starting point for the planner, not a safety limit; "
            "with the literal sustain rule it may not fire at all (sustained crowds are rarer than busy seconds). A capacity-based "
            "N needs a surveyed floor area (ours are guesses).",
        ],
    }
    # artifacts
    chart = timeline_chart(sec, bt, ev_secs)
    make_hero(d_out / "hero.jpg", hero_res, peak_img, chart)
    md = plan_markdown(hero_res)
    (d_out / "plan_entry.md").write_text(md, encoding="utf-8")
    plan_json = {k: hero_res[k] for k in ("plan_id", "video", "site", "zone", "recommend", "backtest", "capacity_n",
                                          "density_limit", "evidence", "cosmos", "limits", "trigger_basis")}
    plan_json["trigger"] = {"min_people": bt["n"], "min_seconds": bt["s"], "dip_allowed_s": bt["gap_s"], "max_firing_n": hero_res["max_firing_n"]}
    plan_json["actions"] = [{"action": "TBD", "owner": "TBD", "notify": "TBD"} for _ in range(3)]
    plan_json["sweep"] = sw
    pk.write_results(d_out / "plan_entry.json", plan_json)
    # report
    rp = pk.Report(d_out / "report.html", f"Pinch point register: {name}", f"{z['name']} - {z['site']}")
    rp.credit(name)
    rp.image(d_out / "hero.jpg", "Peak frame, back-test timeline and threshold sweep")
    rp.kv({"peak people in zone (1 Hz)": hero_res["peak_1hz"], "trigger": f">= {bt['n']} for >= {bt['s']:g} s",
           "fires": bt["n_fires"], "time over": f"{bt['time_above_s']:.0f} s ({bt['time_above_pct']:.0f}%)",
           "recommended N (P95)": rec["n"], "highest N that fires": hero_res["max_firing_n"], "peak density (est.)": f"{peak_dens:.2f} /m2" if peak_dens else "n/a"})
    rp.h2("Episodes")
    rp.table([{"#": k, "start": pk.mmss(f["start"]), "alert at": pk.mmss(f["fire_at"]), "end": pk.mmss(f["end"]),
               "duration s": f["duration_s"], "peak": f["peak"], "mean": f["mean"]} for k, f in enumerate(bt["fires"], 1)]
             or [{"#": "-", "note": "trigger never fired"}])
    rp.h2("Threshold sweep")
    rp.table(sw, ["n", "n_fires", "time_above_s", "time_above_pct", "longest_s"])
    rp.h2("Evidence frames")
    for e in evidence:
        rp.image(d_out / e["file"], f"{pk.mmss(e['t'])}: {e['count']} people in zone (1 Hz median {e['count_1hz']})")
    rp.h2("Zone as drawn")
    rp.image(d_out / "zone_reference.jpg", "Reference frame with the zone polygon")
    rp.h2("Cosmos label suggestions (advisory, local stand-in)")
    rp.json(cosmos or {"note": "VLM unavailable"})
    rp.h2("VastDB SQL check (duckdb stand-in)")
    rp.json(hero_res["vastdb_sql_check"])
    rp.h2("Plan entry (Markdown)")
    rp.html(f"<pre>{__import__('html').escape(md)}</pre>")
    rp.save()
    hero_res["seconds"] = round(time.time() - t_start, 1)
    return hero_res


def xcheck(name: str, res: dict[str, Any]) -> dict[str, Any]:
    """Independent recount of the evidence frames with the larger detector (yolo26l) inside the same polygon."""
    poly = [tuple(p) for p in res["zone"]["poly"]]
    w, h = res["frame_w"], res["frame_h"]
    frames = [pk.frame_at(name, e["t"]) for e in res["evidence"]]
    dets = pk.detect_many(frames, model="det_l", conf=0.25)
    out = []
    for e, ds in zip(res["evidence"], dets):
        rows = pd.DataFrame([{"cx": (d["box"][0] + d["box"][2]) / 2, "y2": d["box"][3]} for d in ds if d["name"] == "person"])
        c = int(in_zone_mask(rows, poly, w, h).sum()) if len(rows) else 0
        out.append({"t": e["t"], "yolo26s_track_count": e["count"], "yolo26l_count": c})
    return {"frames": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="default test set")
    ap.add_argument("--video", help="catalogue name (or path)")
    ap.add_argument("--zone", help='normalised polygon "x,y;x,y;..." (single video only)')
    ap.add_argument("--n", type=int, help="trigger people threshold (default: recommended P95)")
    ap.add_argument("--s", type=float, help=f"trigger sustain seconds (default {DEFAULT_S:g})")
    ap.add_argument("--area-m2", type=float, help="zone area estimate in m2 (single video only)")
    ap.add_argument("--density-limit", type=float, default=DEFAULT_DENSITY_LIMIT, help="people/m2 for the capacity-based N")
    ap.add_argument("--gap", type=float, help="seconds of dip below N forgiven inside an episode (default 0 = literal rule)")
    ap.add_argument("--conf", type=float, default=0.0, help="extra person confidence filter (cached tracks use 0.25)")
    ap.add_argument("--no-vlm", action="store_true", help="skip Cosmos label suggestions")
    ap.add_argument("--xcheck", action="store_true", help="recount evidence frames with yolo26l (needs the GPU)")
    ap.add_argument("--preview", action="store_true", help="only draw the zone on a reference frame")
    a = ap.parse_args()
    names = DEFAULT_SET if a.all else [a.video] if a.video else []
    if not names:
        ap.error("give --all or --video NAME")
    results: dict[str, Any] = {}
    prev = {}
    rj = HERE / "results.json"
    if rj.exists():
        with open(rj, encoding="utf-8") as fh:
            prev = json.load(fh)
    t0 = time.time()
    for nm in names:
        if a.preview:
            tr = pk.tracks(nm)
            z = ZONES[nm]
            poly = parse_zone(a.zone) if a.zone else z["poly"]
            sz = zone_series(tr, poly)
            sec = sz["sec"]
            gi = pick_frame_for_second(sz, int(np.argmax(sec)))
            fr = pk.frame_at(nm, float(sz["t"][gi]))
            pk.save_jpg(render_overlay(fr, tr, sz, poly, gi, width=1280), OUT / nm / "zone_preview.jpg")
            print(nm, "peak 1Hz", int(sec.max()), "p95", np.percentile(sec, 95))
            continue
        r = run_video(nm, a)
        old = prev.get("videos", {}).get(nm, {})
        if a.xcheck:
            r["xcheck"] = xcheck(nm, r)
        elif old.get("xcheck") and [e["t"] for e in old["xcheck"]["frames"]] == [e["t"] for e in r["evidence"]]:
            r["xcheck"] = old["xcheck"]
        results[nm] = r
        b = r["backtest"]
        print(f"{nm:<24} peak {r['peak_1hz']:>3}  P95 {r['recommend']['p95']:>5}  N>={b['n']} S={b['s']:g}: "
              f"{b['n_fires']} fires, {b['time_above_s']:.0f}s over, longest {b['longest_s']:.0f}s  ({r['seconds']}s)")
    if a.preview:
        return
    merged = dict(prev.get("videos", {}))
    merged.update(results)
    gt = HERE / "ground_truth.json"
    kill = kill_test(merged, gt) if gt.exists() else None
    pk.write_results(rj, {"idea": "13 PINCH POINT REGISTER", "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "defaults": {"sustain_s": DEFAULT_S, "gap_merge_s": GAP_MERGE_S, "trigger_semantics": "every second of the sustain window must be >= N (no dips)",
                                       "density_limit_per_m2": DEFAULT_DENSITY_LIMIT}, "videos": merged, "kill_test": kill})
    print(f"done in {time.time() - t0:.0f}s -> {rj}")
    if kill:
        print("kill test:", kill["summary"])


def kill_test(videos: dict[str, Any], gt_path: Path) -> dict[str, Any]:
    """Compare zone counts at hand-counted frames with the eyeballed ground truth (+/-20%)."""
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    rows = []
    for g in gt["frames"]:
        v = videos.get(g["video"])
        if not v:
            continue
        poly = [tuple(p) for p in v["zone"]["poly"]]
        tr = pk.tracks(g["video"])
        sz = zone_series(tr, poly)
        gi = int(np.argmin(np.abs(sz["t"] - g["t"])))
        pred = int(sz["frame_count"][gi])
        eye = float(g["eyeball"])
        err = (pred - eye) / eye if eye else float("nan")
        rows.append({"video": g["video"], "t": g["t"], "frame": int(sz["f"][gi]), "eyeball": eye, "eyeball_range": g.get("range"),
                     "zone_count": pred, "error_pct": round(100 * err, 1), "within_20pct": bool(abs(err) <= 0.20),
                     "note": g.get("note", "")})
    ok = sum(r["within_20pct"] for r in rows)
    peak = [r for r, g in zip(rows, [g for g in gt["frames"] if videos.get(g["video"])]) if g.get("peak_frame")]
    pok = sum(r["within_20pct"] for r in peak)
    return {"criterion": "zone counts within +/-20% of eyeballed counts on 3 peak frames (the 3 fire/peak frames of the lead clip "
                         "retail_cctv); all eyeballed frames reported too", "rows": rows,
            "n_frames": len(rows), "n_within": ok, "n_peak_frames": len(peak), "n_peak_within": pok,
            "mean_abs_error_pct": round(float(np.mean([abs(r["error_pct"]) for r in rows])), 1) if rows else None,
            "passed": bool(len(peak) >= 3 and pok == len(peak)),
            "summary": f"peak frames {pok}/{len(peak)} within +/-20%; all eyeballed frames {ok}/{len(rows)}"}


if __name__ == "__main__":
    main()
