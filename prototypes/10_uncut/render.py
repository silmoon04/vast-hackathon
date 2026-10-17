"""hero.jpg, alignment plot, filmstrip and report.html for one verified clip."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import uncut as U
import verify as V
from uncut import pk

FD = "C:/Windows/Fonts/"
GREEN, AMBER, RED = (46, 170, 98), (232, 163, 23), (214, 69, 69)
BG, PANEL, INK, MUTED = (15, 20, 25), (27, 34, 42), (236, 240, 244), (140, 152, 165)
CHIP_EDIT, CHIP_FOREIGN = (232, 163, 23), (214, 69, 69)


def font(sz: int, bold=False):
    try:
        return ImageFont.truetype(FD + ("segoeuib.ttf" if bold else "segoeui.ttf"), sz)
    except OSError:
        return ImageFont.load_default()


def mm(t: float) -> str:
    s = int(round(t))
    return f"{s // 60:02d}:{s % 60:02d}"


def mmx(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


ORANGE = (240, 120, 40)


def verdict_colour(v: str):
    return {"MATCH": GREEN, "AMBIGUOUS": AMBER, "PARTIAL": ORANGE}.get(v, RED)


def chips(r: dict) -> list[tuple[str, tuple]]:
    b = r.get("best")
    out = []
    if r["verdict"] == "NOT IN ARCHIVE" or not b:
        return out
    if b["crop"] < 1.0 or b.get("crop_kind") == "p916":
        out.append((("RE-CROPPED " if b.get("crop_kind") == "p916" else "CROPPED ") +
                    b["crop_desc"].upper().replace("9:16 PORTRAIT RE-CROP ", "9:16 ").strip(), CHIP_EDIT))
    if b["mirrored"]:
        out.append(("MIRRORED", CHIP_EDIT))
    if abs(b["speed"] - 1) > 0.04:
        out.append((f"SPEED {b['speed']:.2f}x", CHIP_EDIT))
    for rm in b["removed"]:
        sec = abs(rm.get("seconds_r", rm["seconds"]))
        out.append((f"{sec:.1f} s {'REMOVED' if rm['seconds'] > 0 else 'REPEATED'} @ clip {mm(rm['at_q'])} (archive {mmx(rm['at_src'])})",
                    CHIP_FOREIGN))
    for f in b["foreign"]:
        out.append((f"{f['q_end'] - f['q_start']:.0f} s FOREIGN FOOTAGE @ clip {mm(f['q_start'])}", CHIP_FOREIGN))
    if r.get("bars_stripped"):
        out.append(("BLACK BARS IGNORED", (110, 120, 132)))
    if not out:
        out.append(("NO EDITS DETECTED", GREEN))
    return out


def _paste(img: Image.Image, bgr: np.ndarray, box, caption=None, crop_rect=None, dim=False):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    ih, iw = bgr.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    if dim:
        im = Image.blend(im, Image.new("RGB", im.size, (20, 25, 30)), 0.65)
    ox, oy = x0 + (w - nw) // 2, y0 + (h - nh) // 2
    img.paste(im, (ox, oy))
    d = ImageDraw.Draw(img)
    if crop_rect:
        bx0, by0, bx1, by1 = crop_rect
        r = (ox + bx0 * nw, oy + by0 * nh, ox + bx1 * nw, oy + by1 * nh)
        for k in range(0, int(r[2] - r[0]), 16):
            d.line([(r[0] + k, r[1]), (min(r[0] + k + 8, r[2]), r[1])], fill=(80, 220, 255), width=3)
            d.line([(r[0] + k, r[3]), (min(r[0] + k + 8, r[2]), r[3])], fill=(80, 220, 255), width=3)
        for k in range(0, int(r[3] - r[1]), 16):
            d.line([(r[0], r[1] + k), (r[0], min(r[1] + k + 8, r[3]))], fill=(80, 220, 255), width=3)
            d.line([(r[2], r[1] + k), (r[2], min(r[1] + k + 8, r[3]))], fill=(80, 220, 255), width=3)
    return (ox, oy, nw, nh)


def _label(d, xy, text, f, fg=INK, bg=(0, 0, 0, 170), pad=6):
    l, t, r, b = d.textbbox(xy, text, font=f)
    d.rounded_rectangle((l - pad, t - pad + 2, r + pad, b + pad), 6, fill=bg[:3])
    d.text(xy, text, font=f, fill=fg)


def _wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + ([cur] if cur else [])


def hero(r: dict, clip: Path, dst: Path, n_seg: int, n_cam: int) -> Path:
    W, H = 1280, 720
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    col = verdict_colour(r["verdict"])
    b = r.get("best")
    # header
    d.text((40, 22), "UNCUT", font=font(34, True), fill=INK)
    d.text((d.textlength("UNCUT", font=font(34, True)) + 58, 32), "Is this clip really from our camera, and what was cut?", font=font(18), fill=MUTED)
    # headline
    located = bool(r["verdict"] in ("MATCH", "PARTIAL") and b)
    if located:
        s0, s1 = b["segments"][0]["src_start"], b["segments"][-1]["src_end"]
        if r["verdict"] == "PARTIAL":
            fs = sum(f["q_end"] - f["q_start"] for f in b["foreign"]) or (1 - b["coverage"]) * r["duration"]
            head = f"Only {b['coverage'] * 100:.0f}% of this clip is ours"
            sub = (f"{V.pretty(b['video'])} {mm(s0)}–{mm(s1)} is real. About {fs:.0f} s has no source in the archive: "
                   f"treat the clip as spliced")
        else:
            head = f"{V.pretty(b['video'])}  ·  {mm(s0)}–{mm(s1)}"
            sub = "Frame-by-frame alignment against the archive: source, window, and every edit measured"
    elif r["verdict"] == "AMBIGUOUS" and b:
        head = f"Possibly {V.pretty(b['video'])}, but not certain"
        sub = r["reason"]
    else:
        head = "Not in our archive"
        sub = r["reason"]
    d.text((40, 84), head, font=font(44, True), fill=INK)
    d.text((42, 142), sub, font=font(17), fill=MUTED)
    # chips (wrap to a second row if needed; frames start below)
    x, y = 40, 172
    cf = font(17, True)
    for text, c in chips(r):
        tw = d.textlength(text, font=cf)
        if x + tw + 24 > 1240 and x > 40:
            x, y = 40, y + 34
        d.rounded_rectangle((x, y, x + tw + 24, y + 30), 15, fill=c)
        d.text((x + 12, y + 2), text, font=cf, fill=(15, 20, 25))
        x += tw + 34
    ftop = 232 if y < 190 else 246
    fh = 300 if y < 190 else 286
    # frames
    q_t, s_t = 0.0, 0.0
    if b:
        seg = max(b["segments"], key=lambda s: s["n"])
        q_t, s_t = (seg["q_start"] + seg["q_end"]) / 2, (seg["src_start"] + seg["src_end"]) / 2
    qf = pk.frame_at(clip, min(q_t, max(0.0, r["duration"] - 0.1)))
    pa = _paste(img, qf, (40, ftop, 620, ftop + fh))
    d = ImageDraw.Draw(img)
    _label(d, (pa[0] + 10, pa[1] + 10), "VIRAL CLIP", font(16, True))
    _label(d, (pa[0] + 10, pa[1] + pa[3] - 36), f"clip time {mmx(q_t)}", font(15), fg=MUTED)
    if b and r["verdict"] != "NOT IN ARCHIVE":
        sf = pk.frame_at(b["video"], min(s_t, U.pk.info(b["video"])["duration"] - 0.1))
        var = U.variant_by_name(b["variant"])
        box = None if (var.kind == "sq" and var.crop >= 1.0) else U.crop_box(var, *sf.shape[:2], unmirror=True)
        pb = _paste(img, sf, (660, ftop, 1240, ftop + fh), crop_rect=box)
        d = ImageDraw.Draw(img)
        _label(d, (pb[0] + 10, pb[1] + 10), f"OUR ARCHIVE · {V.pretty(b['video'])}", font(16, True))
        _label(d, (pb[0] + 10, pb[1] + pb[3] - 36), f"archive time {mmx(s_t)}" + ("  (shown unmirrored)" if b["mirrored"] else ""),
               font(15), fg=MUTED)
        d.text((628, ftop + 140), "→", font=font(34, True), fill=col)
    else:
        d.rounded_rectangle((660, ftop, 1240, ftop + fh), 10, fill=PANEL)
        d.text((690, ftop + 70), "No archive footage aligns", font=font(30, True), fill=INK)
        d.text((690, ftop + 118), f"{n_seg} five-second segments from {n_cam} cameras searched", font=font(17), fill=MUTED)
        c0 = r["stage1"][0] if r.get("stage1") else None
        if c0:
            txt = f"Closest Cosmos-Embed1 window: {V.pretty(c0['video'])} {mm(c0['start'])}. No frame aligns with it."
            for k, line in enumerate(_wrap(d, txt, font(17), 520)):
                d.text((690, ftop + 150 + 26 * k), line, font=font(17), fill=MUTED)
        note = ("Not found is not proof it is not ours: crops at unindexed positions, rotation and overlays are not searched.")
        for k, line in enumerate(_wrap(d, note, font(14), 520)):
            d.text((690, ftop + 228 + 20 * k), line, font=font(14), fill=(110, 122, 135))
    # alignment ribbons: both drawn to the same scale, in ARCHIVE seconds (clip seconds x speed)
    d.text((40, 548), "CLIP vs ARCHIVE, TO SCALE", font=font(13, True), fill=MUTED)
    tl = f"clip {mmx(r['duration'])} as received"
    d.text((1240 - d.textlength(tl, font=font(13)), 548), tl, font=font(13), fill=MUTED)
    bx0, bx1 = 40, 1240
    yq0, yq1 = 566, 592
    if located:
        sp = b["speed"]
        first = b["segments"][0]
        origin = first["src_start"] - first["q_start"] * sp                 # archive second that clip t=0 would map to
        srcs = [(s["src_start"], s["src_end"]) for s in b["segments"]]
        lo, hi = min(a for a, _ in srcs), max(e for _, e in srcs)
        rng0, rng1 = min(origin, lo), max(origin + r["duration"] * sp, hi)
        scale = (bx1 - bx0) / max(rng1 - rng0, 1e-6)
        ax = lambda t: bx0 + (t - rng0) * scale  # noqa: E731                # archive second -> x
        qx = lambda t: ax(origin + t * sp)  # noqa: E731                      # clip second -> x
        d.rounded_rectangle((qx(0), yq0, qx(r["duration"]), yq1), 5, fill=(60, 66, 74))
        for sg in b["segments"]:
            d.rectangle((qx(sg["q_start"]), yq0, qx(sg["q_end"]), yq1), fill=GREEN)
        for f in b["foreign"]:
            d.rectangle((qx(f["q_start"]), yq0, qx(f["q_end"]), yq1), fill=RED)
            d.text((qx(f["q_start"]) + 8, yq0 + 3), "no source in archive", font=font(15, True), fill=(255, 255, 255))
        yb0, yb1 = 636, 662
        d.rounded_rectangle((bx0, yb0, bx1, yb1), 5, fill=(44, 50, 58))
        for sg in b["segments"]:
            d.polygon([(qx(sg["q_start"]), yq1), (qx(sg["q_end"]), yq1), (ax(sg["src_end"]), yb0), (ax(sg["src_start"]), yb0)],
                      fill=(36, 92, 64))
            d.rectangle((ax(sg["src_start"]), yb0, ax(sg["src_end"]), yb1), fill=GREEN)
        d.text((40, 612), f"ARCHIVE {V.pretty(b['video']).upper()}  {mm(rng0)}–{mm(rng1)}  ·  "
                          f"{sum(e - a for a, e in srcs):.0f} s of it used", font=font(13, True), fill=INK)
        for p_, n_ in zip(b["segments"][:-1], b["segments"][1:]):
            g0, g1 = ax(p_["src_end"]), ax(n_["src_start"])
            if g1 - g0 > 6:
                d.rectangle((g0, yb0, g1, yb1), fill=(150, 48, 48))
                sec = (n_["src_start"] - p_["src_end"])
                lab = f"{round(sec * 2) / 2:.1f} s cut out"
                d.text(((g0 + g1) / 2 - d.textlength(lab, font=font(15, True)) / 2, yb0 + 3), lab, font=font(15, True), fill=(255, 255, 255))
    else:
        d.rounded_rectangle((bx0, yq0, bx1, yq1), 5, fill=(60, 66, 74))
        d.text((bx0 + 12, yq0 + 3), "no frames align with anything in the archive", font=font(15, True), fill=(255, 255, 255))
    # stamp
    st = Image.new("RGBA", (380, 130), (0, 0, 0, 0))
    sd = ImageDraw.Draw(st)
    sd.rounded_rectangle((4, 4, 376, 126), 16, outline=col + (255,), width=7)
    word = r["verdict"]
    f = font(46 if len(word) < 8 else 34, True)
    tw = sd.textlength(word, font=f)
    if word == "PARTIAL":
        sd.text(((380 - tw) / 2, 16), word, font=f, fill=col + (255,))
        sub2 = "SPLICED · NOT ALL OURS"
        sd.text(((380 - sd.textlength(sub2, font=font(22, True))) / 2, 78), sub2, font=font(22, True), fill=col + (255,))
    else:
        sd.text(((380 - tw) / 2, 30 if len(word) < 8 else 38), word, font=f, fill=col + (255,))
    st = st.rotate(5, expand=True, resample=Image.BICUBIC)
    img.paste(st, (W - st.width - 24, 4), st)
    # footer
    d = ImageDraw.Draw(img)
    ft = f"hash match {r['n_frames']} frames @ 5 fps  ·  stage 1 Cosmos-Embed1 kNN, stage 2 dHash alignment  ·  {r['seconds']:.0f} s"
    d.text((40, 684), ft, font=font(14), fill=MUTED)
    img.save(dst, quality=90)
    return dst


def alignment_plot(r: dict, dst: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = r.get("best")
    al = r.get("_aligned")
    if not b or al is None:
        return None
    tq, m, j = r["_tq"], al["match"], al["j"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    ax.scatter(tq[m], j[m] / U.HFPS, s=14, c="#2eaa62", label="matched frame")
    ax.scatter(tq[~m], np.full((~m).sum(), np.nan), s=1)
    for s in b["segments"]:
        ax.plot([s["q_start"], s["q_end"]], [s["src_start"], s["src_end"]], c="#1b6e43", lw=2)
    for f in b["foreign"]:
        ax.axvspan(f["q_start"], f["q_end"], color="#d64545", alpha=0.18, label="no source")
    for rm in b["removed"]:
        ax.axvline(rm["at_q"], c="#d64545", ls="--", lw=1)
        ax.annotate(f"{rm['seconds']:+.1f} s", (rm["at_q"], rm["at_src"]), textcoords="offset points", xytext=(6, -14), color="#d64545")
    ax.set_xlabel("viral clip time (s)")
    ax.set_ylabel(f"archive time in {b['video']} (s)")
    ax.set_title(f"frame alignment, speed {b['speed']:.2f}x" + (", mirrored" if b["mirrored"] else "") +
                 (f", crop {b['crop_desc']}" if b["crop"] < 1 or b.get("crop_kind") == "p916" else ""))
    h, l = ax.get_legend_handles_labels()
    ax.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys(), loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)
    return dst


def filmstrip(r: dict, clip: Path, dst: Path, n=6) -> Path | None:
    b = r.get("best")
    if not b or r["verdict"] == "NOT IN ARCHIVE":
        return None
    a = r["_aligned"]
    dur = r["duration"]
    ts = np.linspace(0.3, dur - 0.5, n)
    top, bot = [], []
    for t in ts:
        i = int(round(t * U.HFPS))
        i = min(i, len(a["match"]) - 1)
        qf = pk.frame_at(clip, float(t))
        top.append(qf)
        if a["match"][i]:
            st = a["j"][i] / U.HFPS
            sf = pk.frame_at(b["video"], float(min(st, pk.info(b["video"])["duration"] - 0.1)))
            if b["mirrored"]:
                sf = sf[:, ::-1].copy()
            var = U.variant_by_name(b["variant"])
            if not (var.kind == "sq" and var.crop >= 1.0):
                h, w = sf.shape[:2]
                fx0, fy0, fx1, fy1 = U.crop_box(var, h, w)      # sf is already mirrored here, so the hashed box applies directly
                sf = sf[int(round(fy0 * h)):int(round(fy1 * h)), int(round(fx0 * w)):int(round(fx1 * w))].copy()
            pk.label(sf, f"archive {mmx(st)}", (4, 4), scale=0.9)
            bot.append(sf)
        else:
            blank = np.full_like(cv2.resize(qf, (qf.shape[1], qf.shape[0])), 40)
            cv2.putText(blank, "no source", (20, blank.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 230), 2)
            bot.append(blank)
        pk.label(top[-1], f"clip {mmx(t)}", (4, 4), scale=0.9)
    tw = 416
    h = int(tw * top[0].shape[0] / top[0].shape[1])
    fix = lambda x: cv2.resize(x, (tw, h), interpolation=cv2.INTER_AREA)  # noqa: E731
    rows = []
    for k in range(0, n, 3):
        rows += [np.hstack([fix(x) for x in top[k:k + 3]]), np.hstack([fix(x) for x in bot[k:k + 3]]),
                 np.full((6, tw * len(top[k:k + 3]), 3), 24, np.uint8)]
    img = np.vstack(rows)
    pk.save_jpg(img, dst)
    return dst


def report(r: dict, gt: dict | None, clip: Path, outdir: Path, hero_path: Path, plot, strip) -> Path:
    name = clip.stem
    rep = pk.Report(outdir / "report.html", f"UNCUT: {name}", V.headline(r))
    rep.credit(clip)
    b = r.get("best")
    for k in ([b["video"]] if b else []):
        rep.credit(k)
    rep.image(hero_path, "The answer in one image")
    kv = {"verdict": r["verdict"], "frames hashed": r["n_frames"], "verified in": f"{r['seconds']:.0f} s"}
    if b:
        kv.update({"clip explained": f"{b['coverage'] * 100:.0f}%", "speed": f"{b['speed']:.2f}x",
                   "mirrored": str(b["mirrored"]), "crop": f"{b['crop'] * 100:.0f}%", "stage 1 hit": str(b["stage1_hit"])})
    rep.kv(kv)
    if gt:
        rep.h2("Ground truth vs measured")
        rows = [{"item": "verdict", "truth": gt["verdict"], "measured": r["verdict"]}]
        if gt.get("source"):
            rows += [{"item": "source", "truth": gt["source"], "measured": b["video"] if b else "-"},
                     {"item": "source window", "truth": f"{gt['src_start']}-{gt['src_end']} s",
                      "measured": (f"{b['segments'][0]['src_start']:.1f}-{b['segments'][-1]['src_end']:.1f} s" if b else "-")},
                     {"item": "speed", "truth": gt["speed"], "measured": f"{b['speed']:.2f}" if b else "-"},
                     {"item": "mirrored", "truth": gt["mirrored"], "measured": b["mirrored"] if b else "-"},
                     {"item": "crop", "truth": gt["crop"], "measured": b["crop_desc"] if b else "-"},
                     {"item": "removed", "truth": gt["removed"], "measured": [(round(x['at_src'], 1), round(x['seconds'], 1)) for x in b["removed"]] if b else "-"},
                     {"item": "foreign (clip time)", "truth": gt["foreign"], "measured": [(round(x['q_start'], 1), round(x['q_end'], 1)) for x in b["foreign"]] if b else "-"}]
        rep.table(rows)
        rep.text("Edit applied: " + gt["edit"])
    if plot:
        rep.h2("Alignment")
        rep.image(plot, "Each green dot: one clip frame matched to its archive frame. Jumps = cuts; gaps = foreign footage.")
    if strip:
        rep.h2("Same moments, side by side")
        rep.image(strip, "Top: viral clip. Bottom: the archive frame the alignment picked (mirror/crop undone for comparison).")
    rep.h2("Stage 1: Cosmos-Embed1 shortlist (top 10 windows)")
    rep.table([{"video": c["video"], "window": f"{c['start']:.0f}-{c['end']:.0f} s", "score": round(c["score"], 2),
                "best cosine": round(c["best_sim"], 3)} for c in r["stage1"]])
    rep.h2("Verdict reasoning")
    rep.text(r["reason"])
    fs = r.get("fullscan")
    rep.text(f"Diagnostic (hash-only scan of the whole archive, no stage 1): {fs if fs else 'nothing aligns'}")
    rep.html("<p class='sub'>Query clip is local and was never uploaded.</p>")
    return rep.save()
