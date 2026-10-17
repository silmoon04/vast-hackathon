"""verify(): the whole UNCUT loop for one query clip -> verdict dict."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import align as al
import uncut as U
from uncut import HFPS, pk

PRETTY = {"retail_cctv": "Retail CCTV", "lisbon_parking": "Lisbon cam", "ptz_parking": "PTZ car park",
          "rochester_night": "Rochester night cam", "yonge_dundas": "Yonge-Dundas cam", "motorway_a40": "A40 motorway",
          "dashcam_urban": "Urban dashcam", "elephants_dream": "Elephants Dream"}


def pretty(v: str) -> str:
    return PRETTY.get(v, v.replace("_", " "))


def windows(dur: float, w=5.0, stride=2.5):
    out, s = [], 0.0
    while s + 2.0 <= dur:
        out.append((s, min(s + w, dur)))
        s += stride
    return out or [(0.0, dur)]


def region_mask(arc, cands, pad):
    reg = {}
    for c in cands:
        n = arc.names and c["video"]
        k = len(arc.hash[n]["1.0"])
        m = reg.setdefault(n, np.zeros(k, bool))
        lo, hi = int(max(0, c["start"] - pad) * HFPS), int((c["end"] + pad) * HFPS) + 1
        m[lo:hi] = True
    return reg


def best_alignment(arc, q, tq, videos, regions, topn=4):
    """cov0 for every (video, variant); full alignment on the best few; returns (best, all aligned list, cov0 table)."""
    cov = []
    for n in videos:
        for vn, A in arc.hash[n].items():
            D = al.ham(q, A)
            D[:, ~(arc.valid[n] & regions[n])] = 1.0
            cov.append((float((D.min(1) <= al.TAU).mean()), n, vn))
    cov.sort(reverse=True)
    aligned = []
    for c0, n, vn in cov[:topn]:
        if c0 < 0.08:
            continue
        r = al.align_one(q, tq, arc.hash[n][vn], arc.valid[n], regions[n])
        r.update(video=n, variant=vn, cov0=c0)
        aligned.append(r)
    best = None
    if aligned:
        top = max(aligned, key=lambda r: r["score"])
        if top["score"] > 0:     # least-modified explanation (same video) within 3% of the best score
            elig = [r for r in aligned if r["video"] == top["video"] and r["score"] >= 0.97 * top["score"]]
            best = min(elig, key=lambda r: al.OCCAM.index(r["variant"]))
    return best, aligned, cov


PARTIAL_COVERAGE = 0.90      # below this share of the clip explained (or any foreign span >= MIN_FOREIGN) => PARTIAL


def classify(coverage: float, dur: float, foreign_s: float, twin: float, rel: float, runner_name: str | None,
             runner_cov: float, best_cov: float) -> tuple[str, str]:
    """Verdict rules (pure function, unit-tested in test_smoke.py).
    MATCH = (almost) the whole clip is archive footage; PARTIAL = a real part of it is, the rest has no source
    (spliced / inserted / heavily altered); AMBIGUOUS = cannot say which archive moment/camera; else NOT IN ARCHIVE."""
    msec = coverage * dur
    if coverage >= 0.30 and msec >= 2.0:
        if twin >= 0.75:
            return "AMBIGUOUS", f"an alternative alignment at another time scores {twin * 100:.0f}% as well"
        if rel >= 0.8:
            return "AMBIGUOUS", f"{pretty(runner_name)} matches almost as well ({runner_cov:.2f} vs {best_cov:.2f})"
        if foreign_s >= al.MIN_FOREIGN or coverage < PARTIAL_COVERAGE:
            return "PARTIAL", (f"only {coverage * 100:.0f}% of the clip is archive footage; "
                               f"{max(foreign_s, (1 - coverage) * dur):.0f} s has no source")
        return "MATCH", f"{coverage * 100:.0f}% of the clip aligns frame-by-frame"
    if coverage >= 0.12:
        return "AMBIGUOUS", f"only {coverage * 100:.0f}% of the clip aligns"
    return "NOT IN ARCHIVE", f"best candidate covers only {coverage * 100:.0f}% of the clip"


def verify(arc: "U.Archive", clip: str | Path, tag: str, log=print) -> dict:
    t0 = time.time()
    clip = Path(clip)
    meta = pk.info(clip)
    dur = meta["duration"]
    fr0 = U.grey_frames(clip)
    fr_s, bars = U.strip_bars(fr0)         # black letterbox / pillarbox bars: tried as a second hypothesis below
    q = U.query_hashes(fr0)
    tq = np.arange(len(q)) / HFPS
    t_hash = time.time() - t0
    # ---- stage 1: Cosmos-Embed1 kNN
    t1 = time.time()
    wins = windows(dur)
    qvec = U.embed_clip_windows(clip, wins, tag)
    cands, sim = U.stage1(arc, qvec, wins)
    t_s1 = time.time() - t1
    # ---- stage 2: hash alignment restricted to the stage-1 candidate windows (+ margin)
    t2 = time.time()
    pad = max(10.0, 1.6 * dur)
    regions = region_mask(arc, cands, pad)
    vids = sorted(regions)
    full_regions = {n: np.ones(len(arc.hash[n]["1.0"]), bool) for n in arc.names}
    regions_all = {n: regions.get(n, np.zeros(len(arc.hash[n]["1.0"]), bool)) for n in arc.names}
    best, aligned, cov = best_alignment(arc, q, tq, vids, {**regions_all})
    used_bars = None
    if bars is not None:                    # hypothesis 2: the clip is the content inside black bars
        q_s = U.query_hashes(fr_s)
        b2, a2, c2 = best_alignment(arc, q_s, tq, vids, {**regions_all})
        if b2 is not None and (best is None or b2["score"] > 1.05 * best["score"]):
            best, aligned, cov, q, used_bars = b2, a2, c2, q_s, bars
    t_s2 = time.time() - t2
    out = {"clip": clip.name, "duration": dur, "n_frames": int(len(q)), "stage1": cands, "stage1_windows": len(wins),
           "bars_detected": bars, "bars_stripped": used_bars, "t_hash": t_hash, "t_stage1": t_s1, "t_stage2": t_s2}
    # ---- diagnostic: hash-only scan over the WHOLE archive (no stage 1)
    t3 = time.time()
    best_full, _, cov_full = best_alignment(arc, q, tq, arc.names, full_regions)
    out["t_fullscan"] = time.time() - t3
    out["fullscan"] = ({"video": best_full["video"], "variant": best_full["variant"],
                        "coverage": float(best_full["match"].mean())} if best_full else None)
    verdict, why = "NOT IN ARCHIVE", "no archive footage aligns with this clip"
    out.update(verdict=verdict, reason=why, best=None)
    if best is not None:
        d = al.describe(best, tq, dur)
        twin = best.get('alt_ratio', 0.0)
        others = [c for c in cov if c[1] != best["video"]]
        runner = max(others, default=(0.0, None, None))
        rel = runner[0] / max(best["cov0"], 1e-6)
        var = U.variant_by_name(best["variant"])
        msec = d["coverage"] * dur
        verdict, why = classify(d["coverage"], dur, d["foreign_seconds"], twin, rel, runner[1], runner[0], best["cov0"])
        hit = any(c["video"] == best["video"] and d["segments"] and
                  c["start"] <= d["segments"][0]["src_end"] and c["end"] >= d["segments"][0]["src_start"] for c in cands)
        out.update(best={"video": best["video"], "variant": best["variant"],
                         "crop": 1.0 if var.kind == "p916" else var.crop, "crop_kind": var.kind,
                         "crop_desc": U.crop_desc(var), "crop_offset": var.ox, "mirrored": var.mirror,
                         "alt_ratio": twin, "runner_up": {"video": runner[1], "cov0": runner[0]},
                         "stage1_hit": bool(hit), **d},
                   verdict=verdict, reason=why)
        out["_aligned"] = best
    out["_q"], out["_tq"] = q, tq
    out["seconds"] = time.time() - t0
    return out


def mmr(t: float) -> str:
    s = int(round(t))
    return f"{s // 60:02d}:{s % 60:02d}"


def mmx1(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def edit_phrases(b: dict, r: dict | None = None) -> list[str]:
    """Edits in words. Positions are in VIRAL-CLIP time first, archive time in parentheses."""
    bits = []
    if b.get("crop_kind") == "p916":
        bits.append("re-cropped to " + b["crop_desc"].replace("9:16 portrait re-crop ", "9:16 portrait "))
    elif b["crop"] < 1.0:
        bits.append(f"cropped to {b['crop'] * 100:.0f}%" + (f" ({b['crop_desc'].split(', ')[1]})" if ", " in b["crop_desc"] else ""))
    if b["mirrored"]:
        bits.append("mirrored")
    if r is not None and r.get("bars_stripped"):
        bits.append("letterboxed (black bars ignored)")
    if abs(b["speed"] - 1) > 0.04:
        bits.append(f"speed {b['speed']:.2f}x")
    for rm in b["removed"]:
        sec = rm.get("seconds_r", rm["seconds"])
        unc = f", position +/-{rm['pos_uncertain_s']:.1f} s" if rm.get("pos_uncertain_s", 0) >= 0.5 else ""
        word = "removed" if rm["seconds"] > 0 else "repeated"
        bits.append(f"{abs(sec):.1f} s {word} at clip {mmr(rm['at_q'])} (archive {mmx1(rm['at_src'])}{unc})")
    return bits


def headline(r: dict) -> str:
    """One line a journalist can paste: 'MATCH: Lisbon cam, 01:00-01:18, cropped to 80%, 3.0 s removed at clip 00:07 (archive 01:06.8)'."""
    b = r.get("best")
    if r["verdict"] not in ("MATCH", "PARTIAL") or not b:
        return f"{r['verdict']}: {r['reason']}"
    s0, s1 = b["segments"][0]["src_start"], b["segments"][-1]["src_end"]
    bits = [f"{mmr(s0)}-{mmr(s1)}"] + edit_phrases(b, r)
    if r["verdict"] == "PARTIAL":
        fo = "; ".join(f"{f['q_end'] - f['q_start']:.0f} s at clip {mmr(f['q_start'])}" for f in b["foreign"]) or "part of it"
        return (f"PARTIAL: {b['coverage'] * 100:.0f}% of the clip is {pretty(b['video'])}, " + ", ".join(bits) +
                f"; NOT from our archive: {fo}")
    return f"MATCH: {pretty(b['video'])}, " + ", ".join(bits)
