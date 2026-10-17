"""LEAK TEST: prove a redacted CCTV export did not leak a face.

    python run.py --all                     # default test set (3 clips x 3 exports)
    python run.py --video retail_cctv       # one clip
    options: --secs 30 --start S --refresh (recompute everything) --rescore (re-evaluate cached audits only)

Pipeline per clip: pick an audit window -> 3 redactors write 3 real mp4 exports -> auditor (YOLO26l + YOLO26s-pose
on full frames and up-scaled head crops, sharpness vs the original) -> ground truth from the redactor logs ->
per-export PASS/FAIL certificate, leak reel, hero.jpg, report.html, results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import protokit as pk  # noqa: E402
import leaklib as L  # noqa: E402

OUT = HERE / "out"
DEFAULT_SET = ["face_demographics_walking_and_pause", "retail_cctv", "yonge_dundas"]
KINDS = list(L.REDACTORS)
SHORT = {"a_keyframe": "A", "b_conf": "B", "c_good": "C", "d_tight": "D"}
RED, GREEN, AMBER, INK, PANEL = (214, 69, 69), (46, 158, 91), (224, 161, 0), (24, 28, 34), (36, 41, 49)
FONT_R, FONT_B = "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT_R, sz)
    except Exception:
        return ImageFont.load_default()


# ------------------------------------------------------------------------------------------------ window / grid


def grid_info(name):
    m = pk.info(name)
    stride = max(1, int(round(m["fps"] / 15.0)))
    return m, stride, m["fps"] / stride


def pick_window(name, secs, start=None):
    """Audit window of `secs` seconds with the most / largest people (sliding window over cached tracks)."""
    m = pk.info(name)
    secs = min(secs, m["duration"])
    if start is not None:
        return float(start), float(min(m["duration"], start + secs))
    p = pk.tracks(name).of("person")
    p = p[(p.conf >= 0.3) & (p.bh >= 50)]
    per = np.zeros(int(m["duration"]) + 2)
    for t, bh in zip(p.t.values, p.bh.values):
        per[int(t)] += min(bh, 300) / 300.0
    csum = np.concatenate([[0], np.cumsum(per)])
    best, bs = 0, -1.0
    for s in range(0, max(1, int(m["duration"] - secs) + 1)):
        v = csum[min(len(csum) - 1, s + int(secs))] - csum[s]
        if v > bs:
            best, bs = s, v
    return float(best), float(min(m["duration"], best + secs))


def src_frames(name, t0, t1, stride):
    for k, (fi, t, fr) in enumerate(pk.iter_frames(name, stride=stride, start=t0, end=t1)):
        yield k, fi, t, fr


def export_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield k, k, k / fps, fr
        k += 1
    cap.release()


def chunks(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# --------------------------------------------------------------------------------------------------- redactors


def build_exports(name, t0, t1, wd, log, kinds=None):
    """Write the three exports; return redaction log {kind: [rects per k]} + grid meta. Uses the redactor-side model
    (YOLO26s): cached tracks for A/B, plus a low-confidence YOLO26s pass for C."""
    m, stride, fps = grid_info(name)
    W, H = m["w"], m["h"]
    tr = pk.tracks(name).of("person")
    by_f: dict[int, list] = {}
    for f, x1, y1, x2, y2, c in zip(tr.f, tr.x1, tr.y1, tr.x2, tr.y2, tr.conf):
        by_f.setdefault(int(f), []).append((float(x1), float(y1), float(x2), float(y2), float(c)))
    cfile = wd / "redactor_c_boxes.json"
    extra = None
    if cfile.exists():  # cache is only valid for this exact window
        d = json.loads(cfile.read_text())
        if isinstance(d, dict) and d.get("t0") == t0 and d.get("t1") == t1:
            extra = {int(k): v for k, v in d["boxes"].items()}
    if extra is None:  # low-confidence YOLO26s person boxes for the good redactor (one locked block per chunk)
        extra = {}
        with L.Gpu() as g:  # one lock acquisition for the whole low-confidence pass (YOLO26s TensorRT, fast)
            for ch in chunks(src_frames(name, t0, t1, stride), L.CHUNK):
                dets = g.predict("det", [c[3] for c in ch], 0.10)
                for (k, fi, t, fr), d in zip(ch, dets):
                    extra[k] = [[round(v, 1) for v in x["box"]] for x in d if x["name"] == "person"]
        log(f"    redactor C low-conf pass {len(extra)} frames")
        cfile.write_text(json.dumps({"t0": t0, "t1": t1, "boxes": extra}))
    kinds = list(kinds or KINDS)  # only the listed redactors are (re)written; the others keep their exports
    paths = {kd: wd.parent / f"export_{kd}.mp4" for kd in kinds}
    writers = {kd: pk.Mp4Writer(paths[kd], fps) for kd in kinds}
    rect_log = {kd: [] for kd in kinds}
    n = 0
    for k, fi, t, fr in src_frames(name, t0, t1, stride):
        for kd in kinds:
            rects = L.redaction_boxes(kd, by_f, fi, k, extra, W, H)
            rect_log[kd].append([list(map(int, r)) for r in rects])
            writers[kd].write(L.blur_rects(fr, rects))
        n += 1
    for w in writers.values():
        w.__exit__()
    return {"fps": fps, "stride": stride, "n": n, "paths": {k: str(v) for k, v in paths.items()}, "rects": rect_log}


# ----------------------------------------------------------------------------------------------------- audits


def run_audit(name, t0, t1, wd, meta, refresh, log):
    audits = {}
    m, stride, fps = grid_info(name)
    targets = {"original": None, **{kd: Path(meta["paths"][kd]) for kd in KINDS}}

    def key_of(path):  # an audit is only valid for the exact window + export file it was computed on
        return f"{t0:.2f}-{t1:.2f}-" + (str(Path(path).stat().st_size) if path is not None else "src")

    def load(key):
        f = wd / f"audit_{key}.json"
        if f.exists() and not refresh:
            d = json.loads(f.read_text())
            if isinstance(d, dict) and d.get("key") == key_of(targets[key]):
                if d.get("metric") != L.METRIC:  # same faces, new sharpness metric: recompute on CPU, no GPU needed
                    path = targets[key]
                    res = L.refresh_sharpness(d["res"], export_frames(path) if path is not None else src_frames(name, t0, t1, stride),
                                              src_frames(name, t0, t1, stride) if path is not None else None)
                    f.write_text(json.dumps({"key": d["key"], "metric": L.METRIC, "res": res}))
                    log(f"    {key}: sharpness re-measured with {L.METRIC} (CPU)")
                    return res
                return d["res"]
            if isinstance(d, list) and len(d) == meta["n"]:  # legacy format (same window, frame counts agree)
                return d
        return None

    for key in targets:
        r_ = load(key)
        if r_ is not None:
            audits[key] = r_
    todo = [k for k in targets if k not in audits]
    if todo:
        with L.Gpu() as g:  # ONE lock acquisition per clip for the original + 3 exports
            log(f"    GPU lock acquired after {L.Gpu.wait_s:.0f}s total wait")
            for key in todo:
                path = targets[key]
                t = time.time()
                if path is None:
                    res = L.audit_stream(g, src_frames(name, t0, t1, stride), ckpt=wd / f"ckpt_{key}_{L.METRIC}_{key_of(path)}.json")
                else:
                    res = L.audit_stream(g, export_frames(path), src_frames(name, t0, t1, stride),
                                         ckpt=wd / f"ckpt_{key}_{L.METRIC}_{key_of(path)}.json")
                (wd / f"audit_{key}.json").write_text(json.dumps({"key": key_of(path), "metric": L.METRIC, "res": res}))
                audits[key] = res
                log(f"    audited {key}: {len(res)} frames, {sum(len(r['faces']) for r in res)} faces, {time.time() - t:.0f}s")
    return audits


# --------------------------------------------------------------------------------------------------- scoring


def centre(r):
    return (r[0] + r[2]) / 2, (r[1] + r[3]) / 2


def same_face(a, b):
    (ax, ay), (bx, by) = centre(a), centre(b)
    return np.hypot(ax - bx, ay - by) <= 0.6 * max(a[2] - a[0], b[2] - b[0])


def score_export(orig, exp, rects, fps):
    """Ground truth from the redactor log vs auditor verdicts (the auditor never sees `rects`).
    GT universe = faces the auditor found in the ORIGINAL (identifiable: width >= MIN_FACE_W, texture). A face is
    'clearly identifiable' (counted in recall) from GT_CLEAR_W px up, so frame-to-frame flicker of faces right at the
    flag threshold does not decide the score. Exposure truth = coverage by the redactor's own blur rectangles."""
    gt_exp, gt_red, gt_part, borderline = [], [], [], []
    tp, over, partial_f, unmatched = 0, [], [], []
    gt_frames = []
    for k, (o, e) in enumerate(zip(orig, exp)):
        og_all = [dict(f, k=k, t=o["t"]) for f in o["faces"] if L.is_identifiable(f)]
        for g in og_all:
            g["cov"] = L.coverage(g["rect"], rects[k])
        flagged = [f for f in e["faces"] if L.is_flagged(f)]
        core = [g for g in og_all if g["w"] >= L.GT_CLEAR_W]
        for g in core:
            g["hit"] = any(same_face(g["rect"], f["rect"]) for f in flagged)
            (gt_exp if g["cov"] < 0.5 else gt_red if g["cov"] >= 0.9 else gt_part).append(g)
        borderline += [g for g in og_all if g["w"] < L.GT_CLEAR_W and g["cov"] < 0.5]
        gt_frames.append({"k": k, "t": o["t"], "faces": [g for g in core if g["cov"] < 0.5]})
        for f in flagged:
            cf = L.coverage(f["rect"], rects[k])  # objective: how much of the flagged face did the redactor actually blur?
            m = [g for g in og_all if same_face(g["rect"], f["rect"])]
            if cf >= 0.9:
                over.append({"k": k, "t": e["t"], "cov": cf, **f})        # flagged although fully inside a blur box
            elif cf < 0.5:
                tp += 1                                                    # unblurred region: a face-like sharp region
                if not m:
                    unmatched.append({"k": k, "t": e["t"], "cov": cf, **f})  # (no counterpart in the original pass)
            else:
                partial_f.append({"k": k, "t": e["t"], "cov": cf, **f})
    n_exp = len(gt_exp)
    n_hit = sum(g["hit"] for g in gt_exp)
    # person level: link GT-exposed faces into people, a person counts as caught if ANY of their frames was flagged
    gt_people = L.link_faces(gt_frames, fps, only_flagged=False)
    hit_k = {(g["k"], round(g["rect"][0])) for g in gt_exp if g["hit"]}
    hit_by_k = {k for k, _ in hit_k}
    caught = [p for p in gt_people if any(k in hit_by_k for k in p["ks"])]
    n_core = len(gt_exp) + len(gt_red) + len(gt_part)
    return {"gt_faces_total": n_core, "gt_exposed": n_exp, "gt_redacted": len(gt_red), "gt_partial": len(gt_part),
            "exposed_caught": n_hit, "recall": round(n_hit / n_exp, 4) if n_exp else None,
            "gt_people_exposed": len(gt_people), "gt_people_caught": len(caught),
            "person_recall": round(len(caught) / len(gt_people), 4) if gt_people else None,
            "borderline_faces_exposed_not_scored": len(borderline),
            "flags_on_unblurred_faces": tp, "flags_without_original_counterpart": len(unmatched),
            "false_flags_over_redacted": len(over), "false_flags_no_face": 0,
            "flags_on_partially_covered": len(partial_f),
            "false_flag_rate": round(len(over) / n_core, 4) if n_core else 0.0,
            "false_flag_rate_denominator": n_core,
            "false_flag_examples": over[:12], "unmatched_examples": unmatched[:12], "partial_examples": partial_f[:6]}


def certificate(name, kind, exp, fps, meta, ledger_people, people, orig_path, window):
    flagged_frames = [r for r in exp if any(L.is_flagged(f) for f in r["faces"])]
    verdict = "PASS" if not flagged_frames else "FAIL"
    path = Path(meta["paths"][kind])
    return {"video": name, "export": path.name, "redactor": L.REDACTORS[kind]["label"], "verdict": verdict,
            "frames_audited": len(exp), "seconds_audited": round(len(exp) / fps, 2),
            "frames_with_exposed_face": len(flagged_frames), "seconds_with_exposed_face": round(len(flagged_frames) / fps, 2),
            "face_tracks_exposed": len(ledger_people), "people_estimate": len(people),
            "people_estimate_note": "face tracks re-linked across gaps <= 3 s by position; an estimate, not a count of verified identities",
            "first_exposure": pk.mmss(min(r["t"] for r in flagged_frames)) if flagged_frames else None,
            "audit_window_s": [round(window[0], 2), round(window[1], 2)],
            "method": {"auditor_models": "YOLO26l person detector + YOLO26s-pose face landmarks (frame + 8x head crops)",
                       "redactor_models": "YOLO26s (+ConfByteTrack cache)",
                       "min_face_px": L.MIN_FACE_W, "min_original_texture_lapvar": L.MIN_LV_ORIG,
                       "sharp_ratio_threshold": L.SHARP_RATIO, "sharp_pixel_fraction_threshold": L.SHARP_FRAC,
                       "metric": f"{L.METRIC}: measured on the eyes+nose core ({L.CORE_W:.0%} x {L.CORE_H:.0%} of the face rectangle); "
                                 "a face must lie inside a YOLO26l person box of the original",
                       "rule": "FAIL if ANY identifiable face is still sharp in ANY frame"},
            "export_sha256": pk.sha256(path), "original_sha256": pk.sha256(pk.video_path(name)),
            "persons": [{k: v for k, v in p.items() if k not in ("ks", "best_rect", "rects", "rect_ks")} for p in people[:40]]}


def head_advisory(name, t0, t1, wd, paths, exps, stride, refresh=False):
    """Landmark-free second look (CPU only): for every YOLO26l person box in an export, is the HEAD region (top 22 %,
    central 50 % width) still as sharp as in the original?  Catches sharp heads the pose model gave no landmarks for.
    Advisory: reported next to the certificate, it does not change the PASS/FAIL rule."""
    f = wd / "head_advisory.json"
    sig = {kd: Path(paths[kd]).stat().st_size for kd in KINDS}
    if f.exists() and not refresh:
        d = json.loads(f.read_text())
        if d.get("sig") == sig and d.get("t0") == t0 and d.get("min_w") == L.MIN_FACE_W:
            return d["res"]
    res = {kd: {"frames": 0, "heads": 0, "sharp_heads": 0, "persons_checked": 0} for kd in KINDS}
    its = {kd: export_frames(paths[kd]) for kd in KINDS}
    for k, fi, t, fr in src_frames(name, t0, t1, stride):
        efr = {kd: next(its[kd])[3] for kd in KINDS}
        for kd in KINDS:
            n = 0
            for b in exps[kd][k]["persons"]:
                bw, bh = b[2] - b[0], b[3] - b[1]
                if bh < 48:
                    continue
                res[kd]["persons_checked"] += 1
                hr = (b[0] + 0.25 * bw, b[1], b[2] - 0.25 * bw, b[1] + 0.22 * bh)
                if hr[2] - hr[0] < L.MIN_FACE_W:
                    continue
                lo = L.lap_var(fr, hr)
                if lo < L.MIN_LV_ORIG:
                    continue
                res[kd]["heads"] += 1
                if L.lap_var(efr[kd], hr) / (lo + 1e-6) >= L.SHARP_RATIO:
                    res[kd]["sharp_heads"] += 1
                    n += 1
            res[kd]["frames"] += 1 if n else 0
    f.write_text(json.dumps({"sig": sig, "t0": t0, "min_w": L.MIN_FACE_W, "res": res}))
    return res


# ------------------------------------------------------------------------------------------------------ images


def crop_face(frame, rect, grow=0.5, size=150):
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    a = max(w, h) * (1 + grow) / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = frame.shape[:2]
    X1, Y1, X2, Y2 = int(max(0, cx - a)), int(max(0, cy - a)), int(min(W, cx + a)), int(min(H, cy + a))
    c = frame[Y1:Y2, X1:X2]
    if c.size == 0:
        c = np.zeros((size, size, 3), np.uint8)
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC)


def read_frame(path, k):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def tile_with_strip(crop, text, tile, col=(60, 60, 214)):
    """Face crop with a coloured frame and the time / person label on a solid strip BELOW it (nothing over the face)."""
    t = crop.copy()
    cv2.rectangle(t, (0, 0), (tile - 1, tile - 1), col, 3)
    strip = np.full((22, tile, 3), (36, 30, 30), np.uint8)
    cv2.putText(strip, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([t, strip])


def ranked_people(people):
    return sorted(people, key=lambda p: -(min(p["frames"], 30) * ((p["best_rect"][2] - p["best_rect"][0]) ** 0.5)))


def leak_reel(kind, exp, paths, fps, people, name=None, eye=None, n=18, tile=150):
    """Crops of exposed faces with their times: per person the best frame plus two spread out (same track)."""
    picks = []
    for p in ranked_people(people)[: max(1, n // 3)]:
        ks = p["rect_ks"]
        sel = {p["best_k"]} | {ks[j] for j in np.linspace(0, len(ks) - 1, min(3, len(ks))).astype(int)}
        picks += [(p, k) for k in sorted(sel)[:3]]
    tiles = []
    for p, k in picks:
        rect = p["rects"][p["rect_ks"].index(k)]
        fr = read_frame(paths[kind], k)
        if fr is None:
            continue
        ev = (eye or {}).get((name, kind, k, round(rect[0], 1), round(rect[1], 1)))
        if ev is False:
            tiles.append(tile_with_strip(crop_face(fr, rect, 0.7, tile), f"{pk.mmss(exp[k]['t'])}  false flag", tile, (0, 170, 240)))
        else:
            tiles.append(tile_with_strip(crop_face(fr, rect, 0.7, tile), f"{pk.mmss(exp[k]['t'])}  {p['person']}  ~{p['frames']}f", tile))
    if not tiles:
        return None
    return pk.sheet(tiles, cols=min(6, len(tiles)), tile_w=tile, pad=4)


def pil_text(d, xy, text, sz, fill, bold=False, anchor="la"):
    d.text(xy, text, font=font(sz, bold), fill=fill, anchor=anchor)


def stamp(d, x, y, verdict, sz=40):
    col = GREEN if verdict == "PASS" else RED
    f = font(sz, True)
    w = int(d.textlength(verdict, font=f)) + 24
    d.rounded_rectangle([x - w, y, x, y + sz + 14], radius=8, fill=col)
    d.text((x - w // 2, y + (sz + 14) // 2), verdict, font=f, fill=(255, 255, 255), anchor="mm")


def eye_flag_labels():
    """{(video, export, k, rect0, rect1): True/False} from the eyeball labels of FLAGGED faces (labels written by me)."""
    try:
        s = json.loads((HERE / "eyeball_sample.json").read_text())
        lab = json.loads((HERE / "eyeball_gt.json").read_text()).get("precision", "")
        if len(lab) != len(s["precision"]):
            return {}
        return {(it["video"], it["export"], it["k"], round(it["rect"][0], 1), round(it["rect"][1], 1)): y == "Y"
                for it, y in zip(s["precision"], lab)}
    except Exception:
        return {}


def face_key(name, kd, k, f):
    return (name, kd, k, round(f["rect"][0], 1), round(f["rect"][1], 1))


def pick_k(kind, exps, orig, name=None, eye=None):
    """Per-column hero frame. FAIL: the frame where THIS export shows most readable faces (biggest total width as
    tie-break). PASS: the worst case for the redactor, the frame with most identifiable faces in the original."""
    best, bs = 0, -1.0
    flagged_any = any(L.is_flagged(f) for r in exps[kind] for f in r["faces"])
    for r in exps[kind]:
        k = r["k"]
        if flagged_any:
            fl = [f for f in r["faces"] if L.is_flagged(f)]
            s = len(fl) * 1000 + sum(f["w"] for f in fl) + 100000 * sum(1 for f in fl if (eye or {}).get(face_key(name, kind, k, f)) is True)
        else:
            idf = [f for f in orig[k]["faces"] if L.is_identifiable(f)]
            s = len(idf) * 1000 + sum(f["w"] for f in idf)
        if s > bs:
            best, bs = k, s
    return best


def make_hero(name, window, certs, exps, orig, paths, fps, metrics, people, out):
    t0w = window[0]
    W = 1280
    n_col = len(KINDS)
    mx, gap = 20, 12
    cw = (W - 2 * mx - gap * (n_col - 1)) // n_col
    eye = eye_flag_labels()
    ks = {kd: pick_k(kd, exps, orig, name, eye) for kd in KINDS}
    panels = []
    for kd in KINDS:
        k = ks[kd]
        fr = read_frame(paths[kd], k)
        if fr is None:
            fr = np.zeros((360, 640, 3), np.uint8)
        sc = cw / fr.shape[1]
        small = cv2.resize(fr, (cw, int(fr.shape[0] * sc)), interpolation=cv2.INTER_AREA)
        fl = [f for f in exps[kd][k]["faces"] if L.is_flagged(f)]
        for f in fl:  # red = readable face; amber = flagged, but my eyeball label says it is not a readable face
            x1, y1, x2, y2 = [v * sc for v in f["rect"]]
            col = (0, 170, 240) if eye.get(face_key(name, kd, k, f)) is False else (60, 60, 235)
            cv2.rectangle(small, (int(x1 - 5), int(y1 - 5)), (int(x2 + 5), int(y2 + 5)), col, 2)
        if fl:  # evidence inset: the largest eyeball-confirmed (else largest) exposed face, cropped from the export at full resolution
            f = max(fl, key=lambda f: (eye.get(face_key(name, kd, k, f)) is True, f["w"]))
            ins = crop_face(fr, f["rect"], 0.8, 92)
            cv2.rectangle(ins, (0, 0), (91, 91), (60, 60, 235), 3)
            small[small.shape[0] - 92 - 34: small.shape[0] - 34, 4:96] = ins
        panels.append((small, exps[kd][k], len(fl), len([f for f in orig[k]["faces"] if L.is_identifiable(f)])))
    ph = max(p[0].shape[0] for p in panels)
    head, card_h = 124, 52 + ph + 166
    fail = [kd for kd in KINDS if certs[kd]["verdict"] == "FAIL"]
    reel_kinds = sorted(fail, key=lambda kd: -certs[kd]["frames_with_exposed_face"])[:3]
    tile = 118
    H = head + card_h + 16 + 22 + len(KINDS) * 26 + 30 + max(1, len(reel_kinds)) * (tile + 22 + 30) + 40
    img = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(img)
    pil_text(d, (mx + 8, 12), "LEAK TEST", 44, (255, 255, 255), True)
    pil_text(d, (mx + 8, 62), f"Did the redaction leak a face?  Motivated-intruder audit of {n_col} redacted CCTV exports  |  clip: {name}  |  "
             f"{certs[KINDS[0]]['seconds_audited']:.0f} s audited", 14, (170, 180, 192))
    pil_text(d, (W - 28, 18), "auditor: YOLO26l + pose landmarks", 14, (140, 150, 162), anchor="ra")
    pil_text(d, (W - 28, 38), "redactors: YOLO26s", 14, (140, 150, 162), anchor="ra")
    # one-line story under the title (numbers are measured, from the certificates)
    a = certs["a_keyframe"]
    nfa = a["frames_audited"]
    if a["verdict"] == "FAIL":
        story = (f"A looks blurred at playback speed, but {a['frames_with_exposed_face']} of {nfa} frames "
                 f"({100 * a['frames_with_exposed_face'] / nfa:.0f} %) show readable faces: ~{a['people_estimate']} people (est.), {a['seconds_with_exposed_face']:.1f} s.")
    else:
        story = "Every export passed on this clip."
    pil_text(d, (mx + 8, 86), story, 20, (255, 190, 120), True)
    for i, kd in enumerate(KINDS):
        x0 = mx + i * (cw + gap)
        y0 = head
        c = certs[kd]
        small, r, n_here, n_ident = panels[i]
        d.rounded_rectangle([x0, y0, x0 + cw, y0 + card_h], radius=10, fill=PANEL)
        pil_text(d, (x0 + 10, y0 + 6), L.REDACTORS[kd]["label"], 17, (255, 255, 255), True)
        pil_text(d, (x0 + 10, y0 + 30), L.REDACTORS[kd]["blurb"], 11, (150, 160, 172))
        img.paste(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), (x0, y0 + 52))
        stamp(d, x0 + cw - 8, y0 + 58, c["verdict"], 28)
        base = y0 + 52 + ph
        txt = (f"{pk.mmss(r['t'])}: {n_here} readable face(s)" if n_here else f"{pk.mmss(r['t'])}: 0 of {n_ident} readable")
        d.rectangle([x0, base - 30, x0 + cw, base], fill=(18, 21, 26))
        pil_text(d, (x0 + 8, base - 25), txt, 14, (255, 150, 150) if n_here else (140, 225, 170), True)
        yy = base + 8
        if c["verdict"] == "FAIL":
            pil_text(d, (x0 + 10, yy), f"{c['frames_with_exposed_face']} frames / {c['seconds_with_exposed_face']:.1f} s", 24, (255, 120, 120), True)
            pil_text(d, (x0 + 10, yy + 34), f"~{c['people_estimate']} people ({c['face_tracks_exposed']} face tracks)", 14, (225, 230, 236))
            pil_text(d, (x0 + 10, yy + 54), f"first leak at {c['first_exposure']}", 13, (225, 230, 236))
        else:
            pil_text(d, (x0 + 10, yy), "0 exposed faces", 24, (120, 225, 160), True)
            pil_text(d, (x0 + 10, yy + 34), f"{c['frames_audited']} frames audited", 14, (225, 230, 236))
            pil_text(d, (x0 + 10, yy + 54), "frame shown = most faces in original", 12, (150, 160, 172))
        m = metrics[kd]
        pil_text(d, (x0 + 10, yy + 78), f"truth: {m['gt_exposed']} faces left exposed, auditor caught {m['exposed_caught']}", 11, (150, 160, 172))
        pil_text(d, (x0 + 10, yy + 94), f"false flags: {m['false_flags_over_redacted']} of {m['false_flag_rate_denominator']} faces", 11, (150, 160, 172))
        hs = c["head_region_advisory"]["person_frames_with_sharp_head"]
        pil_text(d, (x0 + 10, yy + 110), f"head-region check (advisory): {hs}", 11, (150, 160, 172))
        lab = [v for key, v in eye.items() if key[0] == name and key[1] == kd]
        if lab:
            pil_text(d, (x0 + 10, yy + 126), f"eyeball check of flags: {sum(lab)} of {len(lab)} are readable faces"
                     + (" (sample)" if kd in ("a_keyframe", "b_conf") else ""), 11, (255, 200, 120))
    # timeline rows
    ty = head + card_h + 16
    dur = certs[KINDS[0]]["seconds_audited"]
    tx0, tx1 = mx + 190, W - mx
    pil_text(d, (mx + 8, ty), "frames with a readable face, over the audited window (red = leak):", 13, (150, 160, 172))
    ty += 22
    for kd in KINDS:
        pil_text(d, (mx + 8, ty + 2), f"{SHORT[kd]}  {L.REDACTORS[kd]['label'][3:]}", 14, (225, 230, 236))
        d.rectangle([tx0, ty, tx1, ty + 20], fill=(52, 58, 67))
        for r in exps[kd]:
            if any(L.is_flagged(f) for f in r["faces"]):
                x = tx0 + (tx1 - tx0) * (r["t"] - t0w) / max(dur, 1e-6)
                d.rectangle([x, ty, x + max(1.5, (tx1 - tx0) / len(exps[kd])), ty + 20], fill=RED)
        ty += 26
    pil_text(d, (tx0, ty), "0 s", 12, (140, 150, 162))
    pil_text(d, (tx1, ty), f"{dur:.0f} s", 12, (140, 150, 162), anchor="ra")
    reel_y = ty + 24
    for kd in reel_kinds:
        pil_text(d, (mx + 8, reel_y), f"leak reel, export {SHORT[kd]} (crops from the export; time, person, leaked frames; amber = flag my eyeball check rejects)",
                 14, (225, 230, 236), True)
        x = mx + 8
        for p in ranked_people(people[kd])[: (W - 2 * mx) // (tile + 12)]:
            fr = read_frame(paths[kd], p["best_k"])
            if fr is None:
                continue
            ev = eye.get((name, kd, p["best_k"], round(p["best_rect"][0], 1), round(p["best_rect"][1], 1)))
            if ev is False:  # flagged by the auditor, but my eyeball label says it is not a readable face: show it as such
                t = tile_with_strip(crop_face(fr, p["best_rect"], 0.7, tile), f"{pk.mmss(exps[kd][p['best_k']]['t'])} false flag", tile, (0, 170, 240))
            else:
                t = tile_with_strip(crop_face(fr, p["best_rect"], 0.7, tile), f"{pk.mmss(exps[kd][p['best_k']]['t'])} {p['person']} ~{p['frames']}f", tile)
            img.paste(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)), (x, reel_y + 24))
            x += tile + 12
        reel_y += tile + 22 + 30
    if not reel_kinds:
        pil_text(d, (mx + 8, reel_y), "no exposed faces in any export", 15, (150, 160, 172))
    pil_text(d, (mx + 8, H - 22), "Video prototype: local stand-ins for DataEngine / VastDB / Cosmos. "
             "Footage credit in report.html. Redactors A and B are SIMULATED faulty redactors.", 11, (110, 120, 132))
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    pk.save_jpg(arr, out / "hero.jpg", 90)
    return ks


# ----------------------------------------------------------------------------------------------- per-video run


def run_video(name, args, log=print):
    t_start = time.time()
    out = OUT / name
    wd = OUT / name / "_work"
    wd.mkdir(parents=True, exist_ok=True)
    t0, t1 = pick_window(name, args.secs, args.start)
    m, stride, fps = grid_info(name)
    log(f"[{name}] window {pk.mmss(t0)}-{pk.mmss(t1)} at {fps:.1f} fps grid (stride {stride})")
    wmeta = wd / "window.json"
    refresh = args.refresh
    if wmeta.exists() and not refresh:
        meta = json.loads(wmeta.read_text())
        if meta.get("t0") != t0 or meta.get("t1") != t1 or not all(Path(p).exists() for p in meta["paths"].values()):
            refresh = True
    if not wmeta.exists():
        refresh = True
    if refresh:
        meta = build_exports(name, t0, t1, wd, log)
        meta.update({"t0": t0, "t1": t1})
        wmeta.write_text(json.dumps(meta))
        log(f"    exports written ({meta['n']} frames each)")
    else:  # a redactor added after the first build (e.g. D): write only the missing export(s)
        missing = [kd for kd in KINDS if kd not in meta["paths"]]
        if missing:
            part = build_exports(name, t0, t1, wd, log, kinds=missing)
            assert part["n"] == meta["n"], "frame grid changed"
            meta["paths"].update(part["paths"])
            meta["rects"].update(part["rects"])
            wmeta.write_text(json.dumps(meta))
            log(f"    exports added: {missing}")
    paths = {k: Path(v) for k, v in meta["paths"].items()}
    audits = run_audit(name, t0, t1, wd, meta, refresh, log)
    orig = audits["original"]
    exps = {kd: audits[kd] for kd in KINDS}
    for res_ in [orig, *exps.values()]:
        L.tag_in_person(res_, orig)
    assert all(len(v) == len(orig) for v in exps.values()), "export/original frame count mismatch"
    for v in exps.values():  # exports carry export-relative time; use source time everywhere
        for e, o in zip(v, orig):
            e["t"] = o["t"]
    metrics = {kd: score_export(orig, exps[kd], meta["rects"][kd], fps) for kd in KINDS}
    ledgers = {kd: L.link_faces(exps[kd], fps) for kd in KINDS}
    people = {kd: L.summarize_people(L.merge_people(ledgers[kd], fps), fps) for kd in KINDS}
    certs = {kd: certificate(name, kd, exps[kd], fps, meta, ledgers[kd], people[kd], None, (t0, t1)) for kd in KINDS}
    adv = head_advisory(name, t0, t1, wd, paths, exps, stride)
    for kd in KINDS:
        certs[kd]["head_region_advisory"] = {"person_frames_with_sharp_head": adv[kd]["sharp_heads"],
                                             "frames_with_sharp_head": adv[kd]["frames"], "heads_checked": adv[kd]["heads"],
                                             "note": "landmark-free check, does not change the verdict"}
    for kd in KINDS:
        pk.write_results(out / f"certificate_{kd}.json", certs[kd])
    # leak reels
    reels = {}
    for kd in KINDS:
        (out / f"leak_reel_{SHORT[kd].lower()}.jpg").unlink(missing_ok=True)  # no stale reel from an earlier run
        img = leak_reel(kd, exps[kd], paths, fps, people[kd], name, eye_flag_labels())
        if img is not None:
            reels[kd] = pk.save_jpg(img, out / f"leak_reel_{SHORT[kd].lower()}.jpg", 86)
    hk = make_hero(name, (t0, t1), certs, exps, orig, paths, fps, metrics, people, out)
    # VastDB-stand-in ledger + DataEngine segment table (restricted SQL: no GROUP BY / JOIN / IN)
    seg = pk.segments(name)
    rows = []
    for kd in KINDS:
        for r in exps[kd]:
            for f in r["faces"]:
                if L.is_flagged(f):
                    rows.append({"video": name, "export": kd, "k": r["k"], "t": r["t"], "seg": int(r["t"] // 5),
                                 "w": f["w"], "ratio": f["ratio"]})
    import pandas as pd
    leaks = pd.DataFrame(rows, columns=["video", "export", "k", "t", "seg", "w", "ratio"])
    con = pk.vastdb({"leaks": leaks, "segments": seg})
    sql_log = []
    seg_rows = []
    for _, s in seg.iterrows():
        if s["end"] <= t0 or s["start"] >= t1:
            continue
        row = {"segment": f"{pk.mmss(s['start'])}-{pk.mmss(s['end'])}", "peak persons (DataEngine)": int(s.get("peak_person", 0))}
        for kd in KINDS:
            q = f"SELECT COUNT(*) FROM leaks WHERE export = '{kd}' AND seg = {int(s['seg'])}"
            row[f"exposed face detections {SHORT[kd]}"] = int(con.execute(q).fetchone()[0])
        seg_rows.append(row)
    sql_log.append("SELECT COUNT(*) FROM leaks WHERE export = '<kind>' AND seg = <n>   -- per DataEngine 5 s segment (no GROUP BY)")
    for kd in KINDS:
        n_sql = int(con.execute(f"SELECT COUNT(*) FROM leaks WHERE export = '{kd}'").fetchone()[0])
        assert n_sql == sum(1 for r in exps[kd] for f in r["faces"] if L.is_flagged(f)), "SQL ledger != audit"
    sql_log.append("SELECT COUNT(*) FROM leaks WHERE export = '<kind>'   -- checked equal to the audit's flagged-face count")
    # report
    write_report(name, (t0, t1), certs, metrics, people, reels, paths, seg_rows, sql_log, out, hk, fps, args)
    EYE_AUDITS[name] = {kd: exps[kd] for kd in KINDS}
    res = {"video": name, "window_s": [t0, t1], "fps_grid": round(fps, 3), "frames": len(orig),
           "faces_identifiable_in_original": sum(1 for r in orig for f in r["faces"] if L.is_identifiable(f)),
           "certificates": {kd: {k: v for k, v in certs[kd].items() if k != "persons"} for kd in KINDS},
           "ground_truth": {kd: {k: v for k, v in metrics[kd].items() if not k.endswith("examples")} for kd in KINDS},
           "false_flag_examples": {kd: metrics[kd]["false_flag_examples"] for kd in KINDS},
           "ledger_people": {kd: [{k: v for k, v in p.items() if k not in ("ks", "rects", "rect_ks")} for p in people[kd]][:30] for kd in KINDS},
           "gpu_lock_wait_s": round(L.Gpu.wait_s), "gpu_held_s": round(L.Gpu.held_s),
           "seconds": round(time.time() - t_start, 1), "hero_frame_k": hk}
    return res


def write_report(name, window, certs, metrics, people, reels, paths, seg_rows, sql_log, out, hk, fps, args):
    rp = pk.Report(out / "report.html", f"Leak test: {name}",
                   "Motivated-intruder audit of three redacted exports. Auditor models differ from the redactor's.")
    rp.credit(name)
    rp.image(out / "hero.jpg", "Each column shows the frame where THAT export leaks most (PASS columns: the frame with most faces in the original). Red boxes = faces the auditor can still read.")
    rp.h2("Certificates")
    rows = [{"export": certs[kd]["export"], "redactor": certs[kd]["redactor"], "verdict": certs[kd]["verdict"],
             "frames exposed": certs[kd]["frames_with_exposed_face"], "seconds exposed": certs[kd]["seconds_with_exposed_face"],
             "face tracks exposed": certs[kd]["face_tracks_exposed"], "people (estimate)": certs[kd]["people_estimate"],
             "sharp heads, landmark-free (advisory)": certs[kd]["head_region_advisory"]["person_frames_with_sharp_head"], "export sha256": certs[kd]["export_sha256"][:16] + "..."}
            for kd in KINDS]
    rp.table(rows)
    rp.html(f'<p class="sub">Rule: an export FAILS if ANY identifiable face (width >= {L.MIN_FACE_W}px, texture in the original) is '
            f"still sharp in ANY single frame: on the eyes+nose core of the face, Laplacian-variance ratio export/original >= {L.SHARP_RATIO} AND "
            f">= {L.SHARP_FRAC:.0%} of the textured core pixels as sharp as in the original; the original core needs Laplacian variance >= {L.MIN_LV_ORIG:.0f}. "
            "Full certificate JSON per export in this folder.</p>")
    rp.h2("Playback: they all look fine at speed")
    for kd in KINDS:
        rp.video(paths[kd], f"{L.REDACTORS[kd]['label']}: {certs[kd]['verdict']}")
    rp.h2("Leak reels (crops of exposed faces, taken from the export)")
    for kd in KINDS:
        if kd in reels:
            rp.image(reels[kd], f"export {SHORT[kd]}: {certs[kd]['frames_with_exposed_face']} frames, "
                                f"~{certs[kd]['people_estimate']} people / {certs[kd]['face_tracks_exposed']} face tracks (tile label = time, person, leaked frames)")
        else:
            rp.text(f"Export {SHORT[kd]}: no exposed faces.")
    rp.h2("Ground truth vs auditor (from the redactor logs)")
    gt = [{"export": SHORT[kd], "faces readable in original": metrics[kd]["gt_faces_total"],
           "left exposed by redactor": metrics[kd]["gt_exposed"], "caught by auditor": metrics[kd]["exposed_caught"],
           "recall": metrics[kd]["recall"] if metrics[kd]["recall"] is not None else "n/a",
           "people exposed / caught": f"{metrics[kd]['gt_people_exposed']} / {metrics[kd]['gt_people_caught']}",
           "false flags": metrics[kd]["false_flags_over_redacted"] + metrics[kd]["false_flags_no_face"],
           "false flag rate": metrics[kd]["false_flag_rate"]} for kd in KINDS]
    rp.table(gt)
    rp.h2("Per-person ledger (estimated people), export A")
    rp.table([{k: (pk.mmss(v) if k.endswith("_t") else v) for k, v in p.items() if k in ("person", "tracks", "frames", "seconds", "first_t", "last_t")}
              for p in people["a_keyframe"][:25]])
    rp.h2("DataEngine 5 s segments x exposure (VastDB SQL stand-in)")
    rp.table(seg_rows)
    rp.html("<pre>" + "\n".join(sql_log) + "</pre>")
    rp.save()


# ------------------------------------------------------------------------------------------------------- main


def kill_test(res):
    """Kill criterion: auditor recall >= 90 % on leaks injected by redactor A, <= 5 % false flags on the good export."""
    import eyeball
    gt = [r["ground_truth"] for r in res.values()]
    exp = sum(g["a_keyframe"]["gt_exposed"] for g in gt)
    hit = sum(g["a_keyframe"]["exposed_caught"] for g in gt)
    ppl = sum(g["a_keyframe"]["gt_people_exposed"] for g in gt)
    ppl_hit = sum(g["a_keyframe"]["gt_people_caught"] for g in gt)
    core_c = sum(g["c_good"]["false_flag_rate_denominator"] for g in gt)
    ff_c = sum(g["c_good"]["false_flags_over_redacted"] for g in gt)
    flagged_c = sum(r["certificates"]["c_good"]["frames_with_exposed_face"] for r in res.values())
    core_d = sum(g["d_tight"]["false_flag_rate_denominator"] for g in gt)
    ff_d = sum(g["d_tight"]["false_flags_over_redacted"] for g in gt)
    part_d = sum(g["d_tight"]["flags_on_partially_covered"] for g in gt)
    expd_d = sum(g["d_tight"]["gt_exposed"] for g in gt)
    hit_d = sum(g["d_tight"]["exposed_caught"] for g in gt)
    flagged_d = sum(r["certificates"]["d_tight"]["frames_with_exposed_face"] for r in res.values())
    ffd = ff_d / core_d if core_d else 0.0
    eye = eyeball.score({v: EYE_AUDITS[v] for v in res if v in EYE_AUDITS}) if EYE_AUDITS else None
    rec = hit / exp if exp else None
    ffr = ff_c / core_c if core_c else 0.0
    out = {"criterion": "auditor recall >= 90% on leaks injected by redactor A AND <= 5% false flags on the good exports (C dilated and D tight)",
           "recall_frame_level_by_construction": {"exposed_faces": exp, "caught": hit, "recall": round(rec, 4) if rec is not None else None},
           "recall_person_level": {"people": ppl, "caught": ppl_hit, "recall": round(ppl_hit / ppl, 4) if ppl else None},
           "false_flags_on_good_export": {"over_redacted_flags": ff_c, "faces_audited": core_c, "rate": round(ffr, 4),
                                          "frames_flagged_in_good_export": flagged_c},
           "false_flags_on_tight_good_export_D": {
               "over_redacted_flags": ff_d, "faces_audited": core_d, "rate": round(ffd, 4), "frames_flagged": flagged_d,
               "flags_on_partially_covered_faces_not_counted_as_false": part_d,
               "faces_left_exposed_by_D_per_its_own_log": expd_d, "of_which_caught": hit_d,
               "per_video": {v: {"verdict": r["certificates"]["d_tight"]["verdict"], "frames_flagged": r["certificates"]["d_tight"]["frames_with_exposed_face"]}
                             for v, r in res.items()}},
           "eyeball": eye}
    pe = (eye or {}).get("precision") or {}
    de = (pe.get("by_export") or {}).get("d_tight")
    if de and de["n"]:  # eyeball-labelled flags on D that are not readable faces
        out["false_flags_on_tight_good_export_D"]["eyeball_flags"] = {
            "flagged_frames_labelled": de["n"], "real_readable_faces": de["real"], "not_readable": de["n"] - de["real"],
            "not_readable_per_face_audited": round((de["n"] - de["real"]) / core_d, 4) if core_d else None,
            "not_readable_per_frame_audited": round((de["n"] - de["real"]) / max(1, sum(r["frames"] for r in res.values())), 4)}
        ffd = max(ffd, (de["n"] - de["real"]) / core_d if core_d else 0.0)
    cond = [rec is not None and rec >= 0.90, ffr <= 0.05, ffd <= 0.05]
    if eye and eye.get("recall") and eye["recall"].get("recall") is not None:
        cond.append(eye["recall"]["recall"] >= 0.90)
    out["passed"] = bool(all(cond))
    return out



_STACK = None
EYE_AUDITS: dict = {}


def main():
    import faulthandler
    OUT.mkdir(exist_ok=True)
    (OUT / "_work").mkdir(exist_ok=True)
    global _STACK
    _STACK = open(OUT / "_work" / "stack.log", "w")
    faulthandler.dump_traceback_later(900, repeat=True, file=_STACK)  # hang diagnosis
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video", action="append")
    ap.add_argument("--secs", type=float, default=20.0)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--refresh", action="store_true", help="rebuild exports and re-run the auditor")
    args = ap.parse_args()
    names = DEFAULT_SET if args.all else (args.video or [])
    if not names:
        ap.error("give --all or --video NAME")
    OUT.mkdir(exist_ok=True)
    allres = {}
    t0 = time.time()
    for n in names:
        allres[n] = run_video(n, args)
    allres["_meta"] = {"total_seconds": round(time.time() - t0, 1), "auditor": {
        "flag_min_face_px": L.MIN_FACE_W, "gt_clear_face_px": L.GT_CLEAR_W, "min_lv_orig": L.MIN_LV_ORIG,
        "sharp_ratio": L.SHARP_RATIO}, "kill_test": kill_test({k: v for k, v in allres.items()})}
    ex = {}
    rp = HERE / "results.json"
    if rp.exists() and not args.all:
        ex = json.loads(rp.read_text())
    ex.update(allres)
    pk.write_results(rp, ex)
    print(f"done in {time.time() - t0:.0f}s -> {rp}")


if __name__ == "__main__":
    main()
