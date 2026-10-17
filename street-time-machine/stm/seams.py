"""Seam-safe joins for Pack D chunk files: how much of each chunk to keep and where it lands on the camera timeline.

Why: the team VSS cut each neighborhood_cam-1 chunk file into 5 s segments for its first 30 s only (max
segment_end_sec 30.0) while the files run 6.4-42.3 s, and consecutive files of one day overlap (file k+1 starts
about 25 s into file k). Concatenating whole files leaves tails with no detection sidecars and repeats the overlap,
so a parked car is cut at every seam. Per chunk k (seconds from the chunk's first frame):

    keep_end[k] = min(duration[k], covered_end[k], stride[k])     stride only when it was measured cleanly
    offset[k+1] = offset[k] + keep_end[k]                         chunks back to back on the new timeline

covered_end = the largest segment_end_sec of the chunk's VSS segments. stride[k] = where in chunk k the first frame
of chunk k+1 (same day) was recorded, found by matching early frames of k+1 against the end of k (`best_match`,
`stride_from_matches`). A seam whose stride could not be measured (static scene, no overlap, ambiguous match) keeps
the covered end and is listed in `unmeasured`. The last chunk of a day has no seam to measure.

Pure functions: no video, no network. scripts/build_packd_v2.py decodes frames on the lab VM and calls these.
CLI: python -m stm.seams <seams.json> re-plans from the inputs stored in a build's seams.json and checks it.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

RX = re.compile(r"_(\d{8})_chunk_(\d+)")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MATCH_RATIO = 0.2        # a distinct match costs at most this fraction of the best cost elsewhere (runner-up)
MIN_DISTINCT = 0.25      # and that runner-up is at least this mean squared grey difference (0-255 scale, 160x90)
MAX_COST = 0.05          # and the match itself is near 0: overlapping chunk files hold the same encoded frames
MAX_SPREAD = 0.3         # s: counting matches must agree on the stride within this
MIN_STRIDE = 5.0         # s: a stride shorter than this is a false match, not an overlap
MIN_KEPT_SEGMENT = 1.0   # s: a caption segment cut shorter than this is dropped from the corpus
EPS = 1e-6


def chunk_key(name: str) -> tuple[str, int]:
    """(day YYYYMMDD, chunk number) from a chunk name or path such as ..._neighborhood_20260901_chunk_0003.mp4."""
    m = RX.search(Path(str(name)).name)
    if not m:
        raise ValueError(f"not a chunk name: {name}")
    return m.group(1), int(m.group(2))


def order_chunks(names: Iterable[str]) -> list[str]:
    """Chunk names in recording order (day, then chunk number), duplicates removed."""
    return sorted(dict.fromkeys(names), key=chunk_key)


def same_day_seams(order: list[str]) -> list[tuple[str, str]]:
    """Consecutive (a, b) pairs recorded on the same day; a day change is not a seam."""
    return [(a, b) for a, b in zip(order, order[1:]) if chunk_key(a)[0] == chunk_key(b)[0]]


def coverage(segments: Iterable[dict[str, Any]]) -> dict[str, list[list[float]]]:
    """Per chunk, the merged [start, end] spans its segments cover (chunk-relative seconds)."""
    spans: dict[str, list[list[float]]] = {}
    for s in segments:
        ch = str(s.get("chunk") or "")
        if not ch:
            continue
        a, b = float(s.get("segment_start_sec") or 0.0), float(s.get("segment_end_sec") or 0.0)
        if b > a:
            spans.setdefault(ch, []).append([a, b])
    out = {}
    for ch, xs in spans.items():
        merged: list[list[float]] = []
        for a, b in sorted(xs):
            if merged and a <= merged[-1][1] + 0.05:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[ch] = merged
    return out


def covered_ends(cov: dict[str, list[list[float]]]) -> dict[str, float]:
    """Largest covered second per chunk."""
    return {ch: max(b for _, b in xs) for ch, xs in cov.items() if xs}


def mse_to(frames: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Mean squared difference of one small grey frame q (h, w) to each of frames (n, h, w)."""
    d = frames.astype(np.float32) - q.astype(np.float32)[None]
    return (d * d).mean(axis=(1, 2))


def best_match(q: np.ndarray, frames: np.ndarray, times: list[float] | np.ndarray,
               guard: float = 0.5) -> tuple[float, float, float]:
    """(time of the best match, its cost, best cost further than `guard` s from it) of q among frames."""
    ts = np.asarray(times, dtype=float)
    m = mse_to(frames, q)
    i = int(np.argmin(m))
    far = m[np.abs(ts - ts[i]) > guard]
    return float(ts[i]), float(m[i]), float(far.min()) if len(far) else float("inf")


def stride_from_matches(matches: list[tuple[float, float, float, float]], duration_a: float,
                        ratio: float = MATCH_RATIO, distinct: float = MIN_DISTINCT, max_cost: float = MAX_COST,
                        spread: float = MAX_SPREAD) -> tuple[float | None, str]:
    """Stride of a seam a -> b from matches [(t_in_a, t_in_b, cost, runner_up), ...] of b's early frames searched
    in a (runner_up = the best cost further than 0.5 s from the match, see best_match).

    A match counts only when it is exact and distinct: cost <= max_cost (the overlap of two chunk files is the same
    encoded frames, so it decodes identically; a look-alike moment elsewhere costs more), cost <= ratio * runner_up
    and runner_up >= distinct (a frozen or empty picture matches everywhere about equally and says nothing). At least
    two counting matches, and at least half of them, must agree on t_in_a - t_in_b within `spread`; the stride is
    their median and must lie inside chunk a. A query whose content lies past the end of a just finds no distinct
    match. Returns (stride, "clean") or (None, reason)."""
    if not matches:
        return None, "no frames"
    good = [ta - tb for ta, tb, cost, run in matches if cost <= max_cost and run >= distinct and cost <= ratio * run]
    if len(good) < 2:
        return None, f"static or frozen picture ({len(good)} of {len(matches)} frames match distinctly)"
    med = statistics.median(good)
    inl = [o for o in good if abs(o - med) <= spread]
    if len(inl) < 2 or 2 * len(inl) < len(good):
        return None, f"distinct matches disagree ({min(good):.2f}-{max(good):.2f} s)"
    s = float(statistics.median(inl))
    if not MIN_STRIDE <= s <= duration_a + EPS:
        return None, f"stride {s:.2f} s outside chunk"
    return round(s, 3), "clean"


def snap(t: float, fps: float | None) -> float:
    """t rounded down to the frame grid of fps (unchanged when fps is None)."""
    if not fps:
        return round(t, 3)
    return round(int(t * fps + EPS) / fps, 6)


def cumulative(keeps: list[float]) -> list[float]:
    """Start of each piece when pieces of these lengths are laid back to back from 0."""
    out, t = [], 0.0
    for k in keeps:
        out.append(round(t, 6))
        t += k
    return out


def day_label(day: str) -> str:
    """'20260901' -> '1 Sep'."""
    return f"{int(day[6:])} {MONTHS[int(day[4:6]) - 1]}"


def days_table(rows: list[dict[str, Any]]) -> list[list[Any]]:
    """[[day, label, start, end|None], ...] on the new timeline (the last day is open-ended, like build_packd)."""
    starts: dict[str, float] = {}
    for r in rows:
        starts.setdefault(r["day"], r["offset"])
    ds = sorted(starts)
    return [[d, day_label(d), round(starts[d], 3), round(starts[ds[i + 1]], 3) if i + 1 < len(ds) else None]
            for i, d in enumerate(ds)]


def _overlap(spans: list[list[float]], a: float, b: float) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


def keep_end(duration: float, covered_end: float, stride: float | None,
             fps: float | None = None) -> tuple[float, str]:
    """(keep_end, which limit won: "duration" | "covered" | "stride") for one chunk."""
    cands = [(duration, "duration"), (covered_end, "covered")] + ([(float(stride), "stride")] if stride is not None
                                                                   else [])
    keep, by = min(cands, key=lambda x: x[0])
    return max(0.0, snap(keep, fps)), by


def plan(chunks: list[dict[str, Any]], cov: dict[str, list[list[float]]], strides: dict[str, float | None],
         reasons: dict[str, str] | None = None, fps: float | None = None,
         actual: dict[str, float] | None = None) -> dict[str, Any]:
    """The seam-safe layout of the chunks.

    chunks: [{"chunk", "duration"}] (any order); cov: coverage(); strides: {chunk a: stride of the seam a -> next
    chunk, or None when unmeasured}; reasons: why a seam is unmeasured; fps: snap keep_end down to that frame grid;
    actual: {chunk: length of the piece really encoded}, which then replaces keep_end (the plan's value stays in
    planned_keep_end) so offsets follow the video that exists.
    Returns {"rows": [...per chunk...], "seams": [...], "unmeasured": [...], "days": [...], "totals": {...}}."""
    reasons = reasons or {}
    dur = {str(c["chunk"]): float(c["duration"]) for c in chunks}
    order = order_chunks(dur)
    nxt = dict(same_day_seams(order))
    ends = covered_ends(cov)
    rows: list[dict[str, Any]] = []
    for ch in order:
        d, ce = dur[ch], ends.get(ch, 0.0)
        st = strides.get(ch) if ch in nxt else None
        planned, by = keep_end(d, ce, st, fps)
        keep = round(float(actual[ch]), 6) if actual and ch in actual else planned
        top = min(ce, d)
        rows.append({"chunk": ch, "day": chunk_key(ch)[0], "n": chunk_key(ch)[1], "duration": round(d, 3),
                     "covered_end": round(ce, 3), "stride": st, "keep_end": keep, "planned_keep_end": planned,
                     "keep_by": by, "removed_s": round(d - keep, 3),
                     "removed_uncovered_s": round(max(0.0, d - max(keep, top)), 3),
                     "removed_overlap_s": round(max(0.0, top - keep), 3),
                     "indexed_s": round(_overlap(cov.get(ch, []), 0.0, keep), 3)})
    for r, off in zip(rows, cumulative([r["keep_end"] for r in rows])):
        r["offset"] = off
    seams = [{"a": a, "b": b, "stride": strides.get(a), "measured": strides.get(a) is not None,
              "reason": "clean" if strides.get(a) is not None else reasons.get(a, "not measured")}
             for a, b in same_day_seams(order)]
    keys = ("removed_s", "removed_uncovered_s", "removed_overlap_s", "indexed_s")
    totals = {"chunks": len(rows), "raw_s": round(sum(r["duration"] for r in rows), 3),
              "kept_s": round(sum(r["keep_end"] for r in rows), 3),
              **{k: round(sum(r[k] for r in rows), 3) for k in keys},
              "seams": len(seams), "measured_seams": sum(s["measured"] for s in seams)}
    return {"rows": rows, "seams": seams, "unmeasured": [s for s in seams if not s["measured"]],
            "days": days_table(rows), "totals": totals}


def offsets(p: dict[str, Any]) -> dict[str, float]:
    """{chunk: offset} of a plan, the chunk_offsets.json shape."""
    return {r["chunk"]: r["offset"] for r in p["rows"]}


def uncovered(p: dict[str, Any], cov: dict[str, list[list[float]]], min_s: float = 0.05) -> list[list[float]]:
    """Spans of the NEW timeline with no segment coverage (gaps inside a chunk's kept part)."""
    out = []
    for r in p["rows"]:
        t = 0.0
        for a, b in cov.get(r["chunk"], []) + [[r["keep_end"], r["keep_end"]]]:
            a = min(a, r["keep_end"])
            if a - t > min_s:
                out.append([round(r["offset"] + t, 3), round(r["offset"] + a, 3)])
            t = max(t, min(b, r["keep_end"]))
    return out


def removed_spans(p: dict[str, Any], old_offsets: dict[str, float]) -> list[list[float]]:
    """Spans of the OLD whole-file timeline that the plan cuts: [old_offset + keep_end, old_offset + duration]."""
    return [[round(old_offsets[r["chunk"]] + r["keep_end"], 3), round(old_offsets[r["chunk"]] + r["duration"], 3)]
            for r in p["rows"] if r["chunk"] in old_offsets and r["duration"] - r["keep_end"] > EPS]


def _frame_t(fr: dict[str, Any], fps: float) -> float:
    ts = fr.get("time_sec")
    return float(ts) if ts is not None else float(fr.get("frame_index", 0)) / fps


def trim_sidecar(side: dict[str, Any], keep_end: float) -> dict[str, Any] | None:
    """The sidecar without frames at or after keep_end (chunk time = segment_start_sec + time_sec).

    None when nothing is left. A trimmed sidecar gets frame_count / detection_count / segment_end_sec updated and
    `trimmed_at`; an untouched one is returned as is."""
    s0 = float(side.get("segment_start_sec") or 0.0)
    fps = float(side.get("fps") or 20.0)
    frames = side.get("frames", [])
    kept = [f for f in frames if s0 + _frame_t(f, fps) < keep_end - EPS]
    if not kept:
        return None
    if len(kept) == len(frames):
        return side
    out = dict(side, frames=kept, frame_count=len(kept),
               detection_count=sum(len(f.get("detections", [])) for f in kept), trimmed_at=round(keep_end, 3))
    if side.get("segment_end_sec") is not None:
        out["segment_end_sec"] = round(min(float(side["segment_end_sec"]), keep_end), 3)
    return out


def trim_segments(segments: list[dict[str, Any]], keep: dict[str, float],
                  min_kept: float = MIN_KEPT_SEGMENT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, dropped) caption rows: a row keeps its part before its chunk's keep_end (end clipped, original in
    segment_end_sec_orig); a row left with less than min_kept seconds is dropped. Rows of unknown chunks pass."""
    kept, dropped = [], []
    for s in segments:
        k = keep.get(str(s.get("chunk") or ""))
        a, b = float(s.get("segment_start_sec") or 0.0), float(s.get("segment_end_sec") or 0.0)
        if k is None or b <= k + EPS:
            kept.append(s)
        elif min(b, k) - a < min_kept:
            dropped.append(s)
        else:
            kept.append(dict(s, segment_end_sec=round(k, 3), segment_end_sec_orig=b,
                             duration=round(k - a, 3)))
    return kept, dropped


def summary(p: dict[str, Any]) -> str:
    """Human-readable lines for a plan."""
    t = p["totals"]
    lines = [f"seams: {t['chunks']} chunks, raw {t['raw_s']:.2f} s -> kept {t['kept_s']:.2f} s "
             f"(removed {t['removed_s']:.2f} s: {t['removed_uncovered_s']:.2f} s without sidecars, "
             f"{t['removed_overlap_s']:.2f} s repeated in the next chunk); indexed {t['indexed_s']:.2f} s",
             f"same-day seams {t['seams']}, stride measured {t['measured_seams']}, unmeasured {len(p['unmeasured'])}",
             "days: " + ", ".join(f"{d[1]} {d[2]:.1f}-{'end' if d[3] is None else f'{d[3]:.1f}'}" for d in p["days"])]
    lines += [f"  unmeasured {s['a']} -> {s['b']}: {s['reason']}" for s in p["unmeasured"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-plan a Pack D seam layout from a build's seams.json and check it")
    ap.add_argument("seams_json")
    a = ap.parse_args(argv)
    rec = json.loads(Path(a.seams_json).read_text(encoding="utf-8"))
    inp = rec["inputs"]
    cov = {k: [list(x) for x in v] for k, v in inp["coverage"].items()}
    again = plan(inp["chunks"], cov, inp["strides"], inp.get("reasons"), fps=inp.get("fps"), actual=inp.get("actual"))
    print(summary(again))
    stored = {r["chunk"]: (r["planned_keep_end"], r["offset"]) for r in rec["plan"]["rows"]}
    bad = [r["chunk"] for r in again["rows"] if r["chunk"] not in stored
           or max(abs(stored[r["chunk"]][0] - r["planned_keep_end"]), abs(stored[r["chunk"]][1] - r["offset"])) > 1e-3]
    print("stored plan matches" if not bad else f"stored plan differs for {len(bad)} chunks: {bad[:5]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
