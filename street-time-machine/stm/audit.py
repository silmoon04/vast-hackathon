"""Known-truth pipeline audit ("Check their pipeline"): push a clip whose every second we know through the team
VSS pipeline and score what comes back, one 5 s VSS segment at a time.

Inputs (data/stm_test/): the two uploaded clips, ground_truth.json (spans with state, source range and the
burned-in clock at each span start, plus the expected label of every 5 s segment) and upload_results.json (per
upload: timing, the VSS segments with Cosmos reasoning_content, YOLO object_counts and the per-frame detection
sidecar).

OUR checks, CPU only, no model, no network:
  * per second (one frame at s + 0.5 s, 320 px wide, the clock box blanked):
      black   mean luma < 15 (stm.ledger DARK_HARD)
      frozen  mean |grey difference| to the previous second < 0.13 (= ffmpeg freezedetect n=0.0005 x 255) for
              >= 3 samples in a row, the first sample of the run included (stm.ledger's run rule)
      ir      mean HSV saturation < 12: a black-and-white (infrared) picture
      live    anything else. Precedence black > frozen > ir > live.
  * per frame: hard cuts (mean |grey difference| to the previous frame > 12, stm.ledger JUMP_DIFF). ffmpeg
    freezedetect (n=0.0005, d=2) and scdet (threshold 10) on the same clock-blanked picture are kept as a second
    witness when ffmpeg is on PATH.
  * the burned-in clock (top-left, x 0-420 / y 0-80 at 1280x720) is read at 1 fps by template matching: glyph
    templates are cut from the first span (0-20 s), whose clock text the ground truth gives, and every other
    second (and the whole 15 s clip) is read blind. Optional cross-check: google/gemma-4-31B-it on W&B Inference
    reads a few crops (`wandb` sub-command, run where WANDB_* is set; its result is merged from wandb_clock.json).
  * CLOCK vs PICTURE (the footage-notary core):
      clock_runs_on_frozen  the clock advances while the picture is one held frame
      clock_runs_on_black   the clock advances over no picture (the recorder ran, the camera did not)
      splice                after a blind stretch the picture resumes so close to the last picture that at most
                            X s can have passed (X = the smallest lag at which 90 % of this clip's own live frame
                            pairs differ more), while the clock says >= X + 3 s passed: footage is missing
      resume_consistent     (info) the picture moved on about as much as the clock says: no contradiction
      date_change_at_cut    a hard cut where the clock's date changes
      clock_jump            the clock moves by more than 1 s against 1 s of video (with or without a cut)
Per 5 s VSS segment: truth label, our label (majority of its seconds), caption excerpt (<= 160 chars, people's
looks and sentences about people removed), caption claims (keyword rules below), YOLO counts and how many sidecar
frame pairs had identical boxes, caption_honest, ours_correct, notes.
  caption_honest: a frozen segment needs a freeze word, a black one a black/no-picture word, an ir one a
  black-and-white/IR word; a live segment is honest unless it claims frozen or black. Any clock value the
  caption quotes must lie within 1 s of the true clock of that segment.
  events: computed from the ground-truth spans (frozen / black spans, splices, date changes) and matched to our
  flags (same kind, same time) and to captions (a segment touching the event says the matching thing).
Scores are counted from those rows; nothing is typed in.

Output: out/stm-test-honesty/audit.json (+ audit.js: the same object as window.__STM_AUDIT__ so ui/audit.html also
works from file://), frames/*.jpg (one per segment, plus the splice evidence pair), audit_summary.txt.

CLI: python -m stm.audit                    full audit from data/stm_test -> out/stm-test-honesty
     python3 audit.py wandb --out F --clip V --times 2.5,7.5 [--clip V2 --times ...]
                                            W&B clock cross-check (run where WANDB_* is set, e.g. the lab VM;
                                            this file runs there on its own: stdlib + cv2 + numpy only)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "stm_test"
OUT = ROOT / "out" / "stm-test-honesty"
MAIN_CLIP = "stm_test_honesty_60s.mp4"
BASE_CLIP = "stm_test_short_15s.mp4"

SAMPLE_W = 320
REF_W, REF_H = 1280, 720
CLOCK_BOX = (0, 0, 420, 80)   # x0, y0, x1, y1 of the burned-in clock at 1280x720
BLACK_LUMA = 15.0             # stm.ledger DARK_HARD: nearly black whatever the structure
FROZEN_DIFF = 0.13            # mean |grey diff| (0..255); ffmpeg freezedetect n=0.0005 is 0.0005 x 255 = 0.13
FROZEN_RUN = 3                # samples (seconds) of the same picture before it counts as frozen
IR_SAT = 12.0                 # mean HSV saturation (0..255) below this = black-and-white picture
CUT_DIFF = 12.0               # stm.ledger JUMP_DIFF: frame-to-frame jump = a cut
LAG_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0)
LAG_MIN_PAIRS = 5             # a lag needs this many live frame pairs before it may bound a resume
SPLICE_MIN_S = 3.0            # clock elapsed minus the most the picture can have moved on
SEG_S = 5.0                   # VSS segment length
CLOCK_INK = 170               # white overlay text; the black@0.55 box keeps the scene under it below ~115
CLOCK_PAD = 4
CLOCK_SHIFT = 2               # +-px search around each character cell
CLOCK_MIN = 0.6               # best template score below this at any position = unreadable
CLOCK_FMT = "%Y-%m-%d %H:%M:%S"
CLOCK_LAYOUT = "dddd-dd-dd dd:dd:dd"
CLOCK_TOL_S = 1               # the clock floors seconds; a quote within 1 s of the truth is right
YOLO_STATIC = 0.95            # share of sidecar frame pairs with identical boxes that means a held picture
EXCERPT_MAX = 160
THUMB_W = 480
POLL_S = 5                    # the upload test polled the VSS API every 5 s (upload_test.py)
WANDB_URL = "https://api.inference.wandb.ai/v1/chat/completions"
WANDB_MODEL = "google/gemma-4-31B-it"
WANDB_PROMPT = ("Each of the {n} images is a crop of a security-camera clock overlay. Read the exact date and time "
                "shown in each image, in order. Answer with only a JSON array of {n} strings formatted "
                "YYYY-MM-DD HH:MM:SS, using null for a crop you cannot read.")
STATE_WORDS = {"live": "live picture", "frozen": "frozen picture", "black": "black, no picture",
               "ir": "night infrared (black and white)"}


# ---------------------------------------------------------------- small helpers
def _np(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return round(float(o), 3)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_np), encoding="utf-8")
    return path


def parse_clock(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), CLOCK_FMT)
    except ValueError:
        return None


def fmt_clock(d: datetime | None) -> str | None:
    return d.strftime(CLOCK_FMT) if d else None


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def sample_frame(s: int, fps: float) -> int:
    """Frame index of the 1 fps sample of second s (mid-second, so span edges on whole seconds never blur it)."""
    return int(round((s + 0.5) * fps))


def held_runs(still: np.ndarray, min_len: int) -> np.ndarray:
    """still[i] = item i repeats item i-1. A run of k such items is k+1 identical items; it counts when
    k + 1 >= min_len and then the first item of the run is marked too (stm.ledger's frozen rule)."""
    n = len(still)
    out = np.zeros(n, bool)
    i = 1
    while i < n:
        if still[i]:
            e = i
            while e + 1 < n and still[e + 1]:
                e += 1
            if e - i + 2 >= min_len:
                out[i - 1:e + 1] = True
            i = e + 1
        else:
            i += 1
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    out, a = [], None
    for i, v in enumerate(mask):
        if v and a is None:
            a = i
        elif not v and a is not None:
            out.append((a, i))
            a = None
    if a is not None:
        out.append((a, len(mask)))
    return out


# ---------------------------------------------------------------- ground truth
def truth_seconds(spans: list[dict[str, Any]], n: int) -> tuple[list[str], list[str | None]]:
    """Per whole second (sampled at s + 0.5): truth state (live|frozen|black|ir) and the true clock text."""
    labels, clocks = [], []
    for s in range(n):
        t = s + 0.5
        sp = next((x for x in spans if x["start"] <= t < x["end"]), spans[-1])
        st = sp["state"]
        if st == "live" and "infrared" in sp.get("light", ""):
            st = "ir"
        labels.append(st)
        c0 = parse_clock(sp.get("clock_start"))
        clocks.append(fmt_clock(c0 + timedelta(seconds=int(np.floor(t - sp["start"])))) if c0 else None)
    return labels, clocks


def truth_events(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What a careful auditor must find, computed from the spans: every frozen / black span (the clock keeps
    running over it), every splice (a picture span that starts on a source moment already shown earlier while its
    clock is >= SPLICE_MIN_S later than when that moment was first shown) and every date change at a cut."""
    ev: list[dict[str, Any]] = []
    for k, sp in enumerate(spans):
        if sp["state"] in ("frozen", "black"):
            ev.append({"kind": f"clock_runs_on_{sp['state']}", "start": float(sp["start"]), "end": float(sp["end"])})
        if not k or "src" not in sp or sp["state"] != "live":
            continue
        c_now, c_prev = parse_clock(sp.get("clock_start")), parse_clock(spans[k - 1].get("clock_start"))
        if c_now and c_prev and c_now.date() != c_prev.date():
            ev.append({"kind": "date_change_at_cut", "start": float(sp["start"]), "end": float(sp["start"]),
                       "clock_from": fmt_clock(c_prev + timedelta(seconds=spans[k - 1]["end"] - spans[k - 1]["start"] - 1)),
                       "clock_to": fmt_clock(c_now)})
        for j in range(k - 1, -1, -1):
            pj = spans[j]
            if "src" not in pj:
                continue
            a, b = pj["src"]
            if not (a - 0.1 <= sp["src"][0] <= b + 0.1):
                continue
            rate = (b - a) / max(pj["end"] - pj["start"], 1e-9)
            into = (sp["src"][0] - a) / rate if rate > 0 else 0.0
            shown = parse_clock(pj["clock_start"]) + timedelta(seconds=into)
            missing = (c_now - shown).total_seconds()
            if missing >= SPLICE_MIN_S:
                ev.append({"kind": "splice", "start": float(sp["start"]), "end": float(sp["start"]),
                           "ref_t": round(pj["start"] + into, 2), "missing_s": round(missing, 1)})
            break
    return ev


# ---------------------------------------------------------------- decoding
@dataclass
class Clip:
    name: str
    fps: float
    w: int
    h: int
    grey: np.ndarray          # (n, sh, 320) uint8, clock box blanked
    keep: np.ndarray          # (sh, 320) bool: pixels outside the clock box
    luma: np.ndarray          # (n,) mean grey per frame outside the clock box
    sat: np.ndarray           # (S,) mean HSV saturation per 1 fps sample, outside the clock box
    clocks: list[np.ndarray]  # S grey crops of the clock box at 420x80

    @property
    def n(self) -> int:
        return len(self.grey)

    @property
    def seconds(self) -> int:
        return len(self.clocks)

    @property
    def duration(self) -> float:
        return self.n / self.fps


def _clock_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = CLOCK_BOX
    sx, sy = w / REF_W, h / REF_H
    c = img[int(y0 * sy):int(round(y1 * sy)), int(x0 * sx):int(round(x1 * sx))]
    c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
    return c if c.shape == (y1 - y0, x1 - x0) else cv2.resize(c, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)


def load_clip(path: Path) -> Clip:
    """Decode every frame once: 320-wide grey with the clock blanked, plus colour and clock crops at 1 fps."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sh = int(round(SAMPLE_W * h / w))
    bx = int(np.ceil(CLOCK_BOX[2] / REF_W * SAMPLE_W)) + 1
    by = int(np.ceil(CLOCK_BOX[3] / REF_H * sh)) + 1
    keep = np.ones((sh, SAMPLE_W), bool)
    keep[:by, :bx] = False
    grey, luma, sat, clocks = [], [], [], []
    s_next, i = 0, 0
    while True:
        ok, img = cap.read()
        if not ok or img is None:
            break
        small = cv2.resize(img, (SAMPLE_W, sh), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        g[~keep] = 0
        grey.append(g)
        luma.append(float(g[keep].mean()))
        if i == sample_frame(s_next, fps):
            sat.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1][keep].mean()))
            clocks.append(_clock_crop(img))
            s_next += 1
        i += 1
    cap.release()
    return Clip(path.name, fps, w, h, np.stack(grey), keep, np.array(luma), np.array(sat), clocks)


def grab(path: Path, frames: list[int]) -> dict[int, np.ndarray]:
    """BGR frames by index (sequential decode, so exact)."""
    want, out = set(frames), {}
    if not want:
        return out
    cap = cv2.VideoCapture(str(path))
    i = 0
    while i <= max(want):
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = img
        i += 1
    cap.release()
    return out


def frame_diff(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16))[keep].mean())


def f2f_series(grey: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Change of every frame vs the previous one (nan for frame 0)."""
    out = np.full(len(grey), np.nan)
    for i in range(1, len(grey)):
        out[i] = frame_diff(grey[i], grey[i - 1], keep)
    return out


# ---------------------------------------------------------------- our per-second ledger
def label_seconds(luma: np.ndarray, sat: np.ndarray, diff: np.ndarray) -> list[str]:
    """Pure on the per-second arrays (diff[s] = change vs sample s-1, nan for s=0)."""
    black = np.asarray(luma) < BLACK_LUMA
    frozen = held_runs(np.nan_to_num(np.asarray(diff, float), nan=99.0) < FROZEN_DIFF, FROZEN_RUN)
    ir = np.asarray(sat) < IR_SAT
    return ["black" if black[s] else "frozen" if frozen[s] else "ir" if ir[s] else "live" for s in range(len(black))]


def per_second(clip: Clip) -> dict[str, Any]:
    S = clip.seconds
    sf = [sample_frame(s, clip.fps) for s in range(S)]
    luma = np.array([clip.luma[f] for f in sf])
    diff = np.full(S, np.nan)
    for s in range(1, S):
        diff[s] = frame_diff(clip.grey[sf[s]], clip.grey[sf[s - 1]], clip.keep)
    return {"labels": label_seconds(luma, clip.sat, diff), "luma": luma, "sat": clip.sat, "diff": diff}


# ---------------------------------------------------------------- frame-level picture states, cuts, lag curve
def frame_states(f2f: np.ndarray, luma: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """black / frozen (held >= FROZEN_RUN s) / live per frame. f2f[i] = change vs frame i-1 (nan at 0)."""
    black = luma < BLACK_LUMA
    held = np.nan_to_num(f2f, nan=99.0) < FROZEN_DIFF
    frozen = held_runs(held, int(round(FROZEN_RUN * fps))) & ~black
    return {"black": black, "frozen": frozen, "live": ~(black | frozen)}


def find_cuts(f2f: np.ndarray, black: np.ndarray, fps: float) -> list[dict[str, Any]]:
    out = []
    for i in np.flatnonzero(np.nan_to_num(f2f, nan=0.0) > CUT_DIFF):
        kind = "picture_lost" if black[i] and not black[i - 1] else \
               "picture_returns" if black[i - 1] and not black[i] else "hard_cut"
        out.append({"frame": int(i), "t": _r(i / fps), "diff": _r(f2f[i]), "kind": kind})
    return out


def lag_curve(grey: np.ndarray, live: np.ndarray, cut_frames: list[int], keep: np.ndarray,
              fps: float, step: int = 2) -> list[dict[str, Any]]:
    """Median and 10th percentile of |grey diff| between two live frames `lag` apart inside one unbroken live
    run: how fast this scene changes on its own."""
    run_id = np.cumsum(np.isin(np.arange(len(grey)), cut_frames) | ~live)
    out = []
    for lag_s in LAG_S:
        L = max(1, int(round(lag_s * fps)))
        vals = [frame_diff(grey[i + L], grey[i], keep) for i in range(0, len(grey) - L, step)
                if live[i] and live[i + L] and run_id[i] == run_id[i + L]]
        out.append({"lag_s": lag_s, "n": len(vals), "median": _r(np.median(vals), 3) if vals else None,
                    "p10": _r(np.percentile(vals, 10), 3) if vals else None})
    return out


def elapsed_bound(d: float, curve: list[dict[str, Any]]) -> float | None:
    """Smallest lag at which 90 % of live frame pairs differ more than d: the picture cannot have moved on by more
    than that. None when no lag separates (the scene is too still to tell)."""
    for c in curve:
        if c["n"] >= LAG_MIN_PAIRS and c["p10"] is not None and c["p10"] > d:
            return c["lag_s"]
    return None


# ---------------------------------------------------------------- clock reading (template match, no model)
@dataclass
class Glyphs:
    y0: int
    y1: int
    c0: float      # centre of character i = c0 + i * pitch
    pitch: float
    width: int
    tmpl: dict[str, np.ndarray]
    n_calib: int


def _ink_runs(cols: np.ndarray, offset: int = 0, min_gap: int = 2) -> list[tuple[int, int]]:
    rr = runs(cols)
    merged: list[list[int]] = []
    for a, b in rr:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a + offset, b + offset) for a, b in merged]


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    rr = runs(mask)
    return max(rr, key=lambda r: r[1] - r[0]) if rr else None


def _cell(crop: np.ndarray, g: Glyphs | tuple[int, int, int], cx: float, pad: int) -> np.ndarray:
    y0, y1, width = (g.y0, g.y1, g.width) if isinstance(g, Glyphs) else g
    P = CLOCK_PAD
    big = cv2.copyMakeBorder(crop, P, P, P, P, cv2.BORDER_REPLICATE)
    x = int(round(cx - width / 2)) + P
    return big[y0 + P - pad:y1 + P + pad, x - pad:x + width + pad].astype(np.float32)


def fit_glyphs(crops: list[np.ndarray], texts: list[str]) -> Glyphs:
    """Character grid and one mean template per character from crops whose clock text is known."""
    stack = np.stack(crops)
    dark = np.median(stack, axis=0) < 128          # the overlay's black@0.55 box; a bright sky around it is not
    by = _longest_run(dark.mean(axis=1) >= 0.3)
    bx = _longest_run(dark[by[0]:by[1]].mean(axis=0) >= 0.3) if by else None
    if not by or not bx:
        raise ValueError("no clock box in the calibration crops")
    steady = np.zeros(dark.shape, bool)
    steady[by[0]:by[1], bx[0]:bx[1]] = ((stack > CLOCK_INK).mean(axis=0) >= 0.5)[by[0]:by[1], bx[0]:bx[1]]
    rows, cols = np.flatnonzero(steady.any(axis=1)), np.flatnonzero(steady.any(axis=0))
    if not len(rows):
        raise ValueError("no clock text in the calibration crops")
    y0, y1 = max(by[0], int(rows[0]) - 2), min(by[1], int(rows[-1]) + 3)
    x_lo = max(bx[0], int(cols[0]) - 4)
    xs, idx = [], []
    for c, t in zip(crops, texts):
        rr = _ink_runs((c[y0:y1, x_lo:] > CLOCK_INK).any(axis=0), x_lo)
        pos = [i for i, ch in enumerate(t) if ch != " "]
        if len(rr) != len(pos):
            continue
        xs += [(a + b - 1) / 2 for a, b in rr]
        idx += pos
    if len(set(idx)) < 3:
        raise ValueError("could not segment the calibration clock text")
    pitch, c0 = np.polyfit(idx, xs, 1)
    width = int(round(pitch))
    cells: dict[str, list[np.ndarray]] = {}
    for c, t in zip(crops, texts):
        for i, ch in enumerate(t):
            if ch != " ":
                cells.setdefault(ch, []).append(_cell(c, (y0, y1, width), c0 + i * pitch, 0))
    tmpl = {ch: np.mean(v, axis=0).astype(np.float32) for ch, v in cells.items()}
    return Glyphs(y0, y1, float(c0), float(pitch), width, tmpl, len(crops))


def read_clock(crop: np.ndarray, g: Glyphs) -> dict[str, Any]:
    """Read one clock crop on the fitted grid. text is None when any position scores below CLOCK_MIN, a separator
    does not match, or the result is not a valid date-time."""
    chars, scores, margins = [], [], []
    for i, kind in enumerate(CLOCK_LAYOUT):
        if kind == " ":
            chars.append(" ")
            continue
        win = _cell(crop, g, g.c0 + i * g.pitch, CLOCK_SHIFT)
        cands = "0123456789" if kind == "d" else kind
        sc = []
        for ch in cands:
            if ch in g.tmpl:
                r = cv2.matchTemplate(win, g.tmpl[ch], cv2.TM_CCOEFF_NORMED)
                sc.append((float(np.nan_to_num(r.max(), nan=0.0)), ch))
        if not sc:
            return {"text": None, "raw": None, "score": 0.0, "margin": 0.0}
        sc.sort(reverse=True)
        chars.append(sc[0][1])
        scores.append(sc[0][0])
        if kind == "d" and len(sc) > 1:
            margins.append(sc[0][0] - sc[1][0])
    raw = "".join(chars)
    score, margin = min(scores), (min(margins) if margins else 0.0)
    ok = score >= CLOCK_MIN and parse_clock(raw) is not None
    return {"text": raw if ok else None, "raw": raw, "score": round(score, 3), "margin": round(margin, 3)}


def calibrate(clip: Clip, spans: list[dict[str, Any]]) -> tuple[Glyphs, list[int]]:
    """Templates from the seconds of the FIRST span only (its clock text is known); the rest is read blind."""
    sp = spans[0]
    c0 = parse_clock(sp["clock_start"])
    secs = [s for s in range(clip.seconds) if sp["start"] <= s + 0.5 < sp["end"]]
    texts = [fmt_clock(c0 + timedelta(seconds=int(np.floor(s + 0.5 - sp["start"])))) for s in secs]
    return fit_glyphs([clip.clocks[s] for s in secs], texts), secs


# ---------------------------------------------------------------- clock vs picture
def _clock_at(reads: list[datetime | None], i: int, fps: float) -> datetime | None:
    s = min(len(reads) - 1, max(0, int(np.floor(i / fps))))
    return reads[s]


def clock_picture_flags(grey: np.ndarray, keep: np.ndarray, f2f: np.ndarray, luma: np.ndarray, fps: float,
                        reads: list[datetime | None]) -> dict[str, Any]:
    """Compare the burned-in clock with what the picture does. Pure on arrays so tests can build a clip in memory."""
    st = frame_states(f2f, luma, fps)
    cuts = find_cuts(f2f, st["black"], fps)
    curve = lag_curve(grey, st["live"], [c["frame"] for c in cuts], keep, fps)
    S = len(reads)
    sf = [sample_frame(s, fps) for s in range(S)]
    flags: list[dict[str, Any]] = []

    for kind, mask in (("frozen", st["frozen"]), ("black", st["black"])):
        for a, b in runs(mask):
            ins = [reads[s] for s in range(S) if a <= sf[s] < b and reads[s] is not None]
            if len(ins) < 2 or (b - a) / fps < 1.0:
                continue
            adv = (ins[-1] - ins[0]).total_seconds()
            if adv > 0:
                what = "the picture is one held frame" if kind == "frozen" else "there is no picture"
                flags.append({"kind": f"clock_runs_on_{kind}", "start": _r(a / fps), "end": _r(b / fps),
                              "clock_from": fmt_clock(ins[0]), "clock_to": fmt_clock(ins[-1]), "advanced_s": adv,
                              "severity": "alarm" if kind == "frozen" else "info",
                              "text": f"clock ran {adv:.0f} s ({ins[0]:%H:%M:%S} -> {ins[-1]:%H:%M:%S}) while {what}"})

    blind = ~st["live"]
    for a, b in runs(blind):
        if b >= len(grey) or a == 0:
            continue
        ref = a if not st["black"][a] else a - 1          # a held picture was first shown at a
        d = frame_diff(grey[b], grey[ref], keep)
        bound = elapsed_bound(d, curve)
        c_ref, c_res = _clock_at(reads, ref, fps), _clock_at(reads, b, fps)
        if c_ref is None or c_res is None:
            flags.append({"kind": "resume_unchecked", "start": _r(b / fps), "end": _r(b / fps), "severity": "info",
                          "text": "picture resumed but the clock was unreadable on one side"})
            continue
        clk = (c_res - c_ref).total_seconds()
        base = {"start": _r(b / fps), "end": _r(b / fps), "ref_t": _r(ref / fps), "ref_frame": int(ref),
                "resume_frame": int(b), "picture_diff": _r(d, 3), "picture_elapsed_max_s": bound,
                "clock_elapsed_s": clk, "clock_ref": fmt_clock(c_ref), "clock_resume": fmt_clock(c_res)}
        usable = [c["lag_s"] for c in curve if c["n"] >= LAG_MIN_PAIRS and c["p10"] is not None]
        if bound is None and not usable:
            flags.append({**base, "kind": "resume_unchecked", "severity": "info",
                          "text": "picture resumed; too little live footage to tell how much time passed"})
        elif bound is None or clk - bound < SPLICE_MIN_S:
            moved = f"more than {max(usable):g} s" if bound is None else f"at most {bound:g} s"
            flags.append({**base, "kind": "resume_consistent", "severity": "info",
                          "text": (f"picture resumed at {b / fps:.1f} s having moved on {moved} of normal motion; "
                                   f"the clock says +{clk:.0f} s: no contradiction")})
        else:
            flags.append({**base, "kind": "splice", "missing_min_s": round(clk - bound, 1), "severity": "alarm",
                          "text": (f"picture at {b / fps:.1f} s continues from the frame shown at {ref / fps:.1f} s "
                                   f"(at most {bound:g} s of motion apart) while the clock ran on {clk:.0f} s "
                                   f"({c_ref:%H:%M:%S} -> {c_res:%H:%M:%S}): >= {clk - bound:.0f} s of footage missing")})

    for c in cuts:
        i = c["frame"]
        before = [s for s in range(S) if sf[s] < i and reads[s] is not None]
        after = [s for s in range(S) if sf[s] >= i and reads[s] is not None]
        if not before or not after:
            continue
        sb, sa = before[-1], after[0]
        jump = (reads[sa] - reads[sb]).total_seconds() - (sf[sa] - sf[sb]) / fps
        if reads[sa].date() != reads[sb].date() and c["kind"] == "hard_cut":
            flags.append({"kind": "date_change_at_cut", "start": c["t"], "end": c["t"], "severity": "alarm",
                          "clock_from": fmt_clock(reads[sb]), "clock_to": fmt_clock(reads[sa]), "jump_s": jump,
                          "text": (f"hard cut at {c['t']:.2f} s: the clock date changes {reads[sb]:%Y-%m-%d %H:%M:%S} -> "
                                   f"{reads[sa]:%Y-%m-%d %H:%M:%S} ({_span_words(jump)} later in 1 s of video)")})
        elif abs(jump) > CLOCK_TOL_S:
            flags.append({"kind": "clock_jump", "start": c["t"], "end": c["t"], "severity": "alarm", "jump_s": jump,
                          "text": f"{c['kind'].replace('_', ' ')} at {c['t']:.2f} s and the clock jumps {jump:+.0f} s"})
    cut_secs = {int(np.floor(c["frame"] / fps)) for c in cuts}
    for s in range(1, S):
        if reads[s] is None or reads[s - 1] is None or s in cut_secs or (s - 1) in cut_secs:
            continue
        jump = (reads[s] - reads[s - 1]).total_seconds() - 1.0
        if abs(jump) > CLOCK_TOL_S:
            flags.append({"kind": "clock_jump", "start": float(s), "end": float(s), "severity": "alarm",
                          "jump_s": jump, "text": f"the clock jumps {jump:+.0f} s at {s} s with no cut in the picture"})
    flags.sort(key=lambda f: (f["start"], f["kind"]))
    return {"flags": flags, "cuts": cuts, "lag_curve": curve,
            "frozen_frames": runs(st["frozen"]), "black_frames": runs(st["black"])}


def _span_words(sec: float) -> str:
    sec = abs(int(round(sec)))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    parts = [f"{d} d" if d else "", f"{h} h" if h else "", f"{m} min" if m else "", f"{s} s" if s and not d else ""]
    return " ".join(p for p in parts if p) or "0 s"


def ffmpeg_checks(path: Path, w: int, h: int, duration: float) -> dict[str, Any] | None:
    """ffmpeg freezedetect + scdet on the clock-blanked picture: an independent second witness."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    bw, bh = int(round(CLOCK_BOX[2] * w / REF_W)), int(round(CLOCK_BOX[3] * h / REF_H))
    vf = f"drawbox=x=0:y=0:w={bw}:h={bh}:color=black:t=fill,freezedetect=n=0.0005:d=2,scdet=threshold=10"
    try:
        p = subprocess.run([exe, "-hide_banner", "-nostats", "-i", str(path), "-vf", vf, "-an", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": type(e).__name__}
    log = p.stderr
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", log)]
    freezes = [[_r(a), _r(ends[k]) if k < len(ends) else _r(duration)] for k, a in enumerate(starts)]
    scenes = [{"t": _r(float(t)), "score": _r(float(sc))}
              for sc, t in re.findall(r"lavfi\.scd\.score: ([\d.]+), lavfi\.scd\.time: ([\d.]+)", log)]
    return {"freezedetect": freezes, "scdet": scenes, "filter": vf}


# ---------------------------------------------------------------- captions
_PERSON = re.compile(r"\b(person|people|pedestrians?|man|men|woman|women|boys?|girls?|child|children|kids?|"
                     r"cyclists?|joggers?|walkers?|someone|somebody|individuals?)\b", re.I)
_NEG = re.compile(r"\b(no|not|without|nor|none|never)\b", re.I)
_FROZEN = re.compile(r"\b(frozen|freez\w*|still (?:image|frame|photo|picture|shot)|static (?:image|frame|picture)|"
                     r"same (?:frame|image)|paused|not updating|unchanging|(?:no|without) (?:any )?(?:visible )?"
                     r"changes?)\b", re.I)
_BLACK = re.compile(r"\b(black screen|completely black|entirely black|fully black|all black|pitch black|"
                    r"blank (?:screen|frame|image|picture)|no (?:signal|picture|image|video)|black frame|"
                    r"screen (?:is|goes|turns|remains) black)\b", re.I)
_DARK = re.compile(r"\b(darkness|dark (?:scene|street|frame|image|picture|screen)|night-?time|at night|night scene|"
                   r"low[- ]light|dimly lit|poorly lit)\b", re.I)
_IR = re.compile(r"(black[- ]and[- ]white|monochrome|gr[ae]yscale|infra-?red|night[- ]vision|\bIR\b)", re.I)
_CLOCKW = re.compile(r"\b(time-?stamp|time stamp|clock|date and time|time display|time overlay)\b", re.I)
_CLOCKV = re.compile(r"(\d{4}-\d{2}-\d{2})?[ T]?\b(\d{1,2}:\d{2}:\d{2})\b")
_MOVE = re.compile(r"\b(driv(?:es|ing|en)|moves|moving|in motion|travels|travell?ing|walk(?:s|ing)|passes|passing|"
                   r"exits|exiting|enters|entering|crosses|crossing|turns|turning|pulls (?:in|out|away))\b", re.I)
_STILL = re.compile(r"\b(no (?:visible )?(?:movement|motion|activity)|no moving|motionless|nothing moves|"
                    r"completely still)\b", re.I)
_CALM = re.compile(r"\b(calm|quiet|still|peaceful|tranquil|serene)\b", re.I)
_CUT = re.compile(r"\b(cuts? (?:to|away)|abrupt\w*|jumps?|jumped|skips?|skipped|gap|discontinu\w*|"
                  r"transitions? (?:to|from)|switch(?:es|ed)? to|changes? to (?:a )?(?:different|night|new)|"
                  r"different day|next day|date chang\w*|scene change|resumes?)\b", re.I)
_OTHER = re.compile(r"\b(other|another|additional|further)\b", re.I)
_CLAUSE = re.compile(r"[,;]|\band\b|\bwhile\b|\bbut\b", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def scrub_people(text: str) -> str:
    """People's looks out (stm.captions.scrub_appearance when importable), then every sentence that describes a
    person (a negated mention like 'no pedestrians' stays): nothing here checks people and they are not the point."""
    try:
        from stm.captions import scrub_appearance
        text = scrub_appearance(text)[0]
    except Exception:  # captions.py busy or not on this machine: the sentence filter below still applies
        pass
    return " ".join(s for s in sentences(text) if not (_PERSON.search(s) and not _NEG.search(s)))


def clock_values(text: str) -> list[str]:
    out: list[str] = []
    for d, t in _CLOCKV.findall(text or ""):
        v = f"{d} {t}" if d else t
        if v not in out:
            out.append(v)
    return out


def caption_claims(text: str) -> dict[str, Any]:
    """What the caption says about the picture itself, by plain keyword rules."""
    moving = still = False
    for s in sentences(text):
        for m in _MOVE.finditer(s):
            clause = _CLAUSE.split(s[:m.start()])[-1]
            if _NEG.search(clause) and not _OTHER.search(clause):   # "no OTHER moving vehicles" is not "nothing moves"
                still = True
            else:
                moving = True
    vals = clock_values(text)
    return {"frozen": bool(_FROZEN.search(text)), "black": bool(_BLACK.search(text)), "dark": bool(_DARK.search(text)),
            "ir": bool(_IR.search(text)), "clock": bool(_CLOCKW.search(text)) or bool(vals), "clock_values": vals,
            "movement": moving, "still": still or bool(_STILL.search(text)),
            "calm": sorted({w.lower() for w in _CALM.findall(text)}), "cut": bool(_CUT.search(text))}


_KEY = re.compile("|".join(p.pattern for p in (_FROZEN, _BLACK, _DARK, _IR, _CLOCKW, _STILL, _CUT)), re.I)


def _clip_text(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    cut = s[:limit - 1]
    return (cut[:cut.rfind(" ")] if " " in cut else cut).rstrip(",;: ") + "…"


def excerpt(text: str, limit: int = EXCERPT_MAX) -> str:
    """<= limit chars: the first sentence plus the sentences that carry a checked claim, in caption order."""
    ss = sentences(scrub_people(text))
    if not ss:
        return ""
    key = [i for i, s in enumerate(ss) if _KEY.search(s)]
    order = [0] + [i for i in key if i] + [i for i in range(1, len(ss)) if i not in key]
    picked: list[int] = []

    def join(ix: list[int]) -> str:
        out = ""
        for k, i in enumerate(ix):
            out += ss[i] if not k else (" " if i == ix[k - 1] + 1 else " … ") + ss[i]
        return out
    for i in order:
        cand = sorted(picked + [i])
        if len(join(cand)) <= limit:
            picked = cand
    return join(picked) if picked else _clip_text(ss[0], limit)


def clock_quotes_ok(values: list[str], first: datetime | None, last: datetime | None) -> bool | None:
    """Every clock value quoted in the caption within CLOCK_TOL_S of the segment's true clock range."""
    if not values or first is None or last is None:
        return None
    lo, hi = first - timedelta(seconds=CLOCK_TOL_S), last + timedelta(seconds=CLOCK_TOL_S)
    for v in values:
        dt = parse_clock(v) if len(v) > 8 else parse_clock(f"{first:%Y-%m-%d} {int(v[:-6]):02d}{v[-6:]}")
        if dt is None or not lo <= dt <= hi:
            return False
    return True


# ---------------------------------------------------------------- YOLO sidecar
def yolo_summary(seg: dict[str, Any]) -> dict[str, Any]:
    try:
        counts = json.loads(seg.get("object_counts") or "{}")
    except (TypeError, json.JSONDecodeError):
        counts = {}
    frames = (seg.get("detections") or {}).get("frames") or []
    sets = [sorted(tuple(d["bbox"]) for d in f.get("detections", [])) for f in frames]
    pairs = max(0, len(sets) - 1)
    same = sum(a == b for a, b in zip(sets, sets[1:]))
    boxes = sum(len(x) for x in sets)
    return {"counts": counts, "detections": int(seg.get("detection_count") or boxes), "frames": len(frames),
            "static_pairs": same, "pairs": pairs,
            "boxes_frozen": bool(pairs and boxes and same / pairs >= YOLO_STATIC)}


# ---------------------------------------------------------------- judging
def judge(truth: str, ours: str, claims: dict[str, Any], clock_ok: bool | None) -> tuple[bool, bool, list[str]]:
    """caption_honest, ours_correct, notes for one segment."""
    notes: list[str] = []
    if truth == "frozen":
        honest = claims["frozen"]
        if not honest:
            notes.append("the picture is one frozen frame; the caption describes it as a live street"
                         + (" ('no moving vehicles' is true of a still image, but it never says the picture stopped)"
                            if claims["still"] else ""))
    elif truth == "black":
        honest = claims["black"]
        notes.append("says the screen is black" if honest else "the screen is black; the caption does not say so")
    elif truth == "ir":
        honest = claims["ir"] and not (claims["frozen"] or claims["black"])
        notes.append("says black and white (infrared)" if claims["ir"] else
                     "the picture is black-and-white infrared; the caption does not say so")
    else:
        honest = not (claims["frozen"] or claims["black"])
        if not honest:
            notes.append("claims a frozen or black picture on a live segment")
    if clock_ok is False:
        honest = False
        notes.append("quotes a clock time that is not on screen in this segment")
    elif clock_ok:
        notes.append("read the burned-in clock correctly: " + ", ".join(claims["clock_values"][:5]))
    if ours != truth:
        notes.append(f"our ledger says {ours}, truth is {truth}")
    return bool(honest), ours == truth, notes


def cosmos_noticed(ev: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    """Did a caption of a segment touching the event say the matching thing?"""
    if ev["end"] > ev["start"]:      # a span: the segments overlapping it
        touch = [r for r in rows if min(r["end"], ev["end"]) - max(r["start"], ev["start"]) > 0]
    else:                            # a moment: the segments on both sides of it
        touch = [r for r in rows if r["start"] <= ev["start"] <= r["end"]]
    k = ev["kind"]
    for r in touch:
        c = r["claims"]
        if k == "clock_runs_on_frozen" and c["frozen"]:
            return True
        if k == "clock_runs_on_black" and c["black"] and len(c["clock_values"]) >= 2:
            return True
        if k in ("splice", "date_change_at_cut", "clock_jump") and c["cut"]:
            return True
        if k == "date_change_at_cut" and len({v[:10] for v in c["clock_values"] if len(v) > 8}) > 1:
            return True
    return False


def ours_caught(ev: dict[str, Any], flags: list[dict[str, Any]]) -> dict[str, Any] | None:
    for f in flags:
        if f["kind"] != ev["kind"]:
            continue
        if ev["end"] > ev["start"]:
            ov = min(f["end"], ev["end"]) - max(f["start"], ev["start"])
            if ov >= 0.5 * (ev["end"] - ev["start"]):
                return f
        elif abs(f["start"] - ev["start"]) <= 1.0:
            return f
    return None


EVENT_WORDS = {"clock_runs_on_frozen": "clock runs on a frozen picture", "clock_runs_on_black": "clock runs over black",
               "splice": "splice: picture resumes where it froze, clock ran on", "date_change_at_cut": "date changes at a hard cut",
               "clock_jump": "clock jump"}


def build_rows(segments: list[dict[str, Any]], truth_seg: list[str], ours_sec: list[str],
               truth_clock: list[str | None], flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per VSS segment. Pure on its inputs (the 12-row fixture test calls this)."""
    rows = []
    S = len(ours_sec)
    for k, seg in enumerate(segments):
        a, b = float(seg["segment_start_sec"]), float(seg["segment_end_sec"])
        secs = [s for s in range(S) if a <= s + 0.5 < b]
        tally: dict[str, int] = {}
        for s in secs:
            tally[ours_sec[s]] = tally.get(ours_sec[s], 0) + 1
        ours = max(tally, key=lambda x: tally[x]) if tally else "unknown"
        truth = truth_seg[k] if k < len(truth_seg) else "unknown"
        cap = seg.get("reasoning_content") or ""
        claims = caption_claims(cap)
        tc = [parse_clock(truth_clock[s]) for s in secs if truth_clock[s]]
        ok = clock_quotes_ok(claims["clock_values"], tc[0] if tc else None, tc[-1] if tc else None)
        honest, correct, notes = judge(truth, ours, claims, ok)
        yolo = yolo_summary(seg)
        if yolo["boxes_frozen"] and truth == "frozen":
            notes.append(f"the YOLO sidecar has identical boxes in {yolo['static_pairs']}/{yolo['pairs']} frame pairs: "
                         "the detector data shows the freeze too")
        mine = [f for f in flags if f["end"] <= f["start"] and a <= f["start"] < b] + \
               [f for f in flags if f["end"] > f["start"] and min(b, f["end"]) - max(a, f["start"]) > 0]
        rows.append({"seg": k + 1, "start": a, "end": b, "truth": truth, "ours": ours, "ours_seconds": tally,
                     "caption": excerpt(cap), "caption_chars": len(cap), "claims": claims, "yolo": yolo,
                     "caption_honest": honest, "ours_correct": correct, "flags": [f["kind"] for f in mine],
                     "notes": notes, "truth_clock": [fmt_clock(tc[0]) if tc else None, fmt_clock(tc[-1]) if tc else None]})
    return rows


def add_event_notes(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    for ev in events:
        if ev["kind"] == "clock_runs_on_black" or ev["cosmos"]:
            continue
        for r in rows:
            point = ev["end"] <= ev["start"]
            hit = (r["start"] <= ev["start"] < r["end"]) if point else \
                (min(r["end"], ev["end"]) - max(r["start"], ev["start"]) > 0)
            if hit:
                r["notes"].append(f"caption missed: {ev['text']}")


def score(rows: list[dict[str, Any]], events: list[dict[str, Any]], flags: list[dict[str, Any]],
          ours_sec: list[str], truth_sec: list[str]) -> dict[str, Any]:
    def ev(kind: str) -> dict[str, Any] | None:
        e = next((x for x in events if x["kind"] == kind), None)
        return None if e is None else {"t": e["start"], "ours": e["ours"], "cosmos": e["cosmos"]}
    matched = {id(e["ours_flag"]) for e in events if e.get("ours_flag") is not None}
    false_flags = [f for f in flags if f.get("severity") == "alarm" and id(f) not in matched]
    return {"segments": len(rows), "cosmos_honest": sum(r["caption_honest"] for r in rows),
            "ours_correct": sum(r["ours_correct"] for r in rows),
            "ours_seconds_correct": sum(a == b for a, b in zip(ours_sec, truth_sec)), "seconds": len(truth_sec),
            "events": len(events), "events_ours": sum(e["ours"] for e in events),
            "events_cosmos": sum(e["cosmos"] for e in events), "false_flags_ours": len(false_flags),
            "splice": ev("splice"), "date_change": ev("date_change_at_cut")}


# ---------------------------------------------------------------- timing
def timing(results: dict[str, Any], name: str, video_s: float) -> dict[str, Any]:
    u = results["uploads"][name]
    t_up = next((e["t"] for e in results.get("events", []) if e.get("msg") == f"uploaded {name}"), 0.0)
    ch = next(iter(u["chunks"].values()))
    others = [next(iter(v["chunks"].values())).get("seen_any") for k, v in results["uploads"].items() if k != name]
    idx = ch["seen_complete"] - t_up
    restarts = [e["t"] for e in results.get("events", []) if "restarted" in e.get("msg", "")]
    return {"upload_s": u["upload_s"], "bytes": u["bytes"], "first_seen_after_s": round(ch["seen_any"] - t_up, 1),
            "indexed_after_s": round(idx, 1), "indexed_after_min": round(idx / 60, 1),
            "captions_fetched_after_s": round(ch["segments_done"] - t_up, 1), "video_s": round(video_s, 2),
            "x_realtime": round(idx / video_s, 1), "segments": ch["total_segments"], "poll_s": POLL_S,
            "batch": any(o is not None and abs(o - ch["seen_any"]) <= POLL_S for o in others),
            "poller_restarts_s": restarts}


# ---------------------------------------------------------------- one clip
def _thumb(img: np.ndarray, path: Path) -> None:
    h, w = img.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.resize(img, (THUMB_W, int(round(THUMB_W * h / w))), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 82])


def audit_clip(path: Path, truth: dict[str, Any], upload: dict[str, Any], results: dict[str, Any],
               glyphs: Glyphs | None, prefix: str, out: Path) -> tuple[dict[str, Any], Glyphs]:
    t0 = time.time()
    clip = load_clip(path)
    S = clip.seconds
    spans = truth["spans"]
    truth_sec, truth_clock = truth_seconds(spans, S)
    ps = per_second(clip)
    calib_secs: list[int] = []
    if glyphs is None:
        glyphs, calib_secs = calibrate(clip, spans)
    reads = [read_clock(c, glyphs) for c in clip.clocks]
    dts = [parse_clock(r["text"]) for r in reads]
    cp = clock_picture_flags(clip.grey, clip.keep, f2f_series(clip.grey, clip.keep), clip.luma, clip.fps, dts)
    flags = cp["flags"]
    segs = sorted(next(iter(upload["chunks"].values()))["segments"], key=lambda x: x["segment_number"])
    rows = build_rows(segs, truth.get("expected_by_5s_segment", []), ps["labels"], truth_clock, flags)
    events = []
    for e in truth_events(spans):
        f = ours_caught(e, flags)
        e = {**e, "ours": f is not None, "ours_flag": f}
        e["cosmos"] = cosmos_noticed(e, rows)
        e["text"] = f["text"] if f else EVENT_WORDS.get(e["kind"], e["kind"])
        events.append(e)
    add_event_notes(rows, events)
    sc = score(rows, events, flags, ps["labels"], truth_sec)
    for e in events:
        e.pop("ours_flag", None)

    held_out = [s for s in range(S) if s not in calib_secs]
    exact = [s for s in range(S) if reads[s]["text"] and reads[s]["text"] == truth_clock[s]]
    sc["clock_read"] = {"exact": len(exact), "n": S, "held_out_exact": sum(s in exact for s in held_out),
                        "held_out_n": len(held_out), "unreadable": sum(r["text"] is None for r in reads),
                        "calibration_seconds": [calib_secs[0], calib_secs[-1]] if calib_secs else None}

    # thumbnails: one per segment (its second +2.5 s) and the splice evidence pair
    want = {r["seg"]: sample_frame(int(r["start"]) + 2, clip.fps) for r in rows}
    pair = [(f["ref_frame"], f["resume_frame"]) for f in flags if f["kind"] == "splice"]
    imgs = grab(path, [*want.values(), *[x for p in pair for x in p]])
    for r in rows:
        if want[r["seg"]] in imgs:
            r["thumb"] = f"frames/{prefix}seg{r['seg']:02d}.jpg"
            _thumb(imgs[want[r["seg"]]], out / r["thumb"])
    for f in flags:
        if f["kind"] == "splice" and f["ref_frame"] in imgs and f["resume_frame"] in imgs:
            f["thumbs"] = [f"frames/{prefix}splice_ref.jpg", f"frames/{prefix}splice_resume.jpg"]
            _thumb(imgs[f["ref_frame"]], out / f["thumbs"][0])
            _thumb(imgs[f["resume_frame"]], out / f["thumbs"][1])

    ff = ffmpeg_checks(path, clip.w, clip.h, clip.duration)
    mood = {st: [bool(r["claims"]["calm"]) for r in rows if r["truth"] == st] for st in ("live", "frozen")}
    res = {"file": path.name, "object_key": upload["resp"].get("object_key"), "video_s": round(clip.duration, 2),
           "fps": clip.fps, "w": clip.w, "h": clip.h, "seconds": S, "segments_n": len(rows),
           "truth": {"spans": spans, "per_second": truth_sec, "clock": truth_clock},
           "ours": {"per_second": ps["labels"], "luma": [_r(x, 1) for x in ps["luma"]],
                    "sat": [_r(x, 1) for x in ps["sat"]], "diff": [_r(x, 3) for x in ps["diff"]],
                    "cuts": cp["cuts"], "ffmpeg": ff},
           "clock": {"per_second": [{"t": s, "text": r["text"], "score": r["score"], "margin": r["margin"]}
                                    for s, r in enumerate(reads)],
                     "method": "template match on a fixed character grid; glyphs cut from the first span (known clock)",
                     "grid": {"y0": glyphs.y0, "y1": glyphs.y1, "c0": round(glyphs.c0, 2),
                              "pitch": round(glyphs.pitch, 3), "chars": "".join(sorted(glyphs.tmpl))}},
           "flags": flags, "events": events, "lag_curve": cp["lag_curve"], "segments": rows, "scores": sc,
           "mood_words": {k: {"with": sum(v), "of": len(v)} for k, v in mood.items()},
           "timing": timing(results, path.name, clip.duration), "runtime_s": round(time.time() - t0, 1)}
    return res, glyphs


def merge_wandb(res: dict[str, Any], wb: dict[str, Any] | None) -> None:
    """Compare W&B vision-model reads with our template reads and the truth at the same seconds."""
    if not wb:
        res["clock"]["wandb"] = None
        return
    part = (wb.get("clips") or {}).get(res["file"])
    if not part:
        res["clock"]["wandb"] = None
        return
    rows, agree, right = [], 0, 0
    for x in part.get("reads", []):
        s = int(np.floor(x["t"]))
        ours = res["clock"]["per_second"][s]["text"] if s < res["seconds"] else None
        tru = res["truth"]["clock"][s] if s < res["seconds"] else None
        txt = x.get("text")
        agree += bool(txt and txt == ours)
        right += bool(txt and txt == tru)
        rows.append({"t": x["t"], "wandb": txt, "ours": ours, "truth": tru})
    res["clock"]["wandb"] = {"model": wb.get("model"), "n": len(rows), "agree_with_ours": agree,
                             "match_truth": right, "calls": part.get("calls"), "tokens": part.get("tokens"),
                             "errors": part.get("errors", []), "reads": rows}
    res["scores"]["wandb_clock"] = {"n": len(rows), "agree_with_ours": agree, "match_truth": right}


def summary_text(doc: dict[str, Any]) -> str:
    lines = ["Known-truth pipeline audit (out/stm-test-honesty/audit.json)"]
    for key in ("clip", "baseline"):
        r = doc.get(key)
        if not r:
            continue
        sc, tm = r["scores"], r["timing"]
        lines.append(f"\n{r['file']}  {r['video_s']} s, {sc['segments']} VSS segments")
        lines.append(f"  Cosmos captions honest: {sc['cosmos_honest']}/{sc['segments']}   "
                     f"our ledger correct: {sc['ours_correct']}/{sc['segments']} segments, "
                     f"{sc['ours_seconds_correct']}/{sc['seconds']} seconds")
        lines.append(f"  truth events: {sc['events']}  caught by us: {sc['events_ours']}  noticed by captions: "
                     f"{sc['events_cosmos']}  our false alarms: {sc['false_flags_ours']}")
        cr = sc["clock_read"]
        lines.append(f"  clock read (template match): {cr['exact']}/{cr['n']} exact, held-out "
                     f"{cr['held_out_exact']}/{cr['held_out_n']}, unreadable {cr['unreadable']}")
        if sc.get("wandb_clock"):
            w = sc["wandb_clock"]
            lines.append(f"  W&B {doc['method']['wandb_model']} cross-check: {w['agree_with_ours']}/{w['n']} agree with "
                         f"ours, {w['match_truth']}/{w['n']} match truth")
        lines.append(f"  timing: upload {tm['upload_s']} s, indexed after {tm['indexed_after_s']} s "
                     f"({tm['indexed_after_min']} min, {tm['x_realtime']}x the video), batch={tm['batch']}")
        for f in r["flags"]:
            lines.append(f"  flag {f['start']:>6.2f}s {f['kind']:<22} {f['text']}")
        for row in r["segments"]:
            lines.append(f"  seg {row['seg']:>2} {row['start']:>5.1f}-{row['end']:<5.1f} truth {row['truth']:<6} "
                         f"ours {row['ours']:<6} honest={str(row['caption_honest']):<5} | {row['caption'][:70]}")
    return "\n".join(lines) + "\n"


def run(data_dir: Path = DATA, out: Path = OUT) -> dict[str, Any]:
    gt = json.loads((data_dir / "ground_truth.json").read_text(encoding="utf-8"))
    results = json.loads((data_dir / "upload_results.json").read_text(encoding="utf-8"))
    wb_path = out / "wandb_clock.json"
    wb = json.loads(wb_path.read_text(encoding="utf-8")) if wb_path.exists() else None
    main, glyphs = audit_clip(data_dir / MAIN_CLIP, gt[MAIN_CLIP], results["uploads"][MAIN_CLIP], results,
                              None, "", out)
    base, _ = audit_clip(data_dir / BASE_CLIP, gt[BASE_CLIP], results["uploads"][BASE_CLIP], results,
                         glyphs, "base_", out)
    for r in (main, base):
        merge_wandb(r, wb)
    m = results["uploads"][MAIN_CLIP]["chunks"]
    seg0 = next(iter(m.values()))["segments"][0]
    doc = {"title": "Known-truth pipeline audit", "generated": datetime.now().isoformat(timespec="seconds"),
           "camera_id": results.get("form", {}).get("camera_id"), "is_public": results.get("form", {}).get("is_public"),
           "pipeline": {"captioner": seg0.get("cosmos_model"), "detector": seg0.get("perception_source"),
                        "segment_s": SEG_S},
           "clip": main, "baseline": base,
           "method": {"sample": "1 frame per second at s + 0.5 s, 320 px wide, clock box blanked",
                      "black_luma": BLACK_LUMA, "frozen_diff": FROZEN_DIFF, "frozen_run_s": FROZEN_RUN,
                      "ir_saturation": IR_SAT, "cut_diff": CUT_DIFF, "splice_min_s": SPLICE_MIN_S,
                      "clock_box": list(CLOCK_BOX), "clock_min_score": CLOCK_MIN, "yolo_static": YOLO_STATIC,
                      "wandb_model": WANDB_MODEL, "excerpt_max": EXCERPT_MAX}}
    write_json(out / "audit.json", doc)
    (out / "audit.js").write_text("window.__STM_AUDIT__ = " + json.dumps(doc, default=_np) + ";\n", encoding="utf-8")
    (out / "audit_summary.txt").write_text(summary_text(doc), encoding="utf-8")
    return doc


# ---------------------------------------------------------------- W&B cross-check (runs on its own, e.g. the lab VM)
def _env(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    try:
        for line in Path("/etc/environment").read_text().splitlines():
            k, _, val = line.partition("=")
            if k.strip() == name:
                return val.strip().strip('"').strip("'") or None
    except OSError:
        pass
    return None


def strip_fences(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


def _redact(s: str, *secrets: str | None) -> str:
    for x in secrets:
        if x:
            s = s.replace(x, "***")
    return re.sub(r"(?i)(bearer\s+)\S+", r"\1***", s)[:300]


def clock_crops_at(video: Path, times: list[float]) -> dict[float, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    cap.release()
    idx = {t: int(round(t * fps)) for t in times}
    imgs = grab(video, list(idx.values()))
    return {t: _clock_crop(imgs[i]) for t, i in idx.items() if i in imgs}


def wandb_read_clocks(video: Path, times: list[float], batch: int = 6, model: str = WANDB_MODEL) -> dict[str, Any]:
    key, team, project = _env("WANDB_API_KEY"), _env("WANDB_TEAM"), _env("WANDB_PROJECT")
    if not (key and team and project):
        raise RuntimeError("WANDB_API_KEY / WANDB_TEAM / WANDB_PROJECT not set")
    crops = clock_crops_at(video, times)
    reads, errors, calls, tokens, t0 = [], [], 0, 0, time.time()
    for k in range(0, len(times), batch):
        part = [t for t in times[k:k + batch] if t in crops]
        content: list[dict[str, Any]] = [{"type": "text", "text": WANDB_PROMPT.format(n=len(part))}]
        for t in part:
            ok, buf = cv2.imencode(".png", crops[t])
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()}})
        body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0,
                "max_tokens": 600}
        req = urllib.request.Request(WANDB_URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {key}", "OpenAI-Project": f"{team}/{project}",
                                              "Content-Type": "application/json"})
        arr: list[Any] = []
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read().decode())
            calls += 1
            tokens += int((resp.get("usage") or {}).get("total_tokens") or 0)
            arr = json.loads(strip_fences(resp["choices"][0]["message"].get("content") or "[]"))
        except urllib.error.HTTPError as e:
            errors.append(_redact(f"HTTP {e.code}: {e.read().decode(errors='replace')}", key))
        except Exception as e:  # network / parse: recorded, never raised with the request attached
            errors.append(_redact(f"{type(e).__name__}: {e}", key))
        for j, t in enumerate(part):
            v = arr[j] if isinstance(arr, list) and j < len(arr) else None
            reads.append({"t": t, "text": v.strip() if isinstance(v, str) else None})
    return {"reads": reads, "calls": calls, "tokens": tokens, "seconds": round(time.time() - t0, 1), "errors": errors}


def main(argv: list[str]) -> int:
    if argv[:1] == ["wandb"]:
        ap = argparse.ArgumentParser(prog="audit.py wandb")
        ap.add_argument("--out", required=True)
        ap.add_argument("--clip", action="append", required=True)
        ap.add_argument("--times", action="append", required=True)
        ap.add_argument("--model", default=WANDB_MODEL)
        a = ap.parse_args(argv[1:])
        doc: dict[str, Any] = {"model": a.model, "clips": {}}
        for v, ts in zip(a.clip, a.times):
            res = wandb_read_clocks(Path(v).expanduser(), [float(x) for x in ts.split(",")], model=a.model)
            doc["clips"][Path(v).name] = res
            print(Path(v).name, f"{len(res['reads'])} crops, {res['calls']} calls, {res['tokens']} tokens, "
                  f"{len(res['errors'])} errors")
        write_json(Path(a.out).expanduser(), doc)
        return 0
    doc = run()
    print(summary_text(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
