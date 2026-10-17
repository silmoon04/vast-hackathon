"""Seam-safe Pack D rebuild (lab VM): neighborhood_cam-1 with every chunk file cut at its keep_end (stm.seams).

    cd <repo root> && STM_TRACK_CACHE=$PWD/_cache/tracks python3 scripts/build_packd_v2.py [--packd ~/packd]
        [--jobs 3] [--remeasure] [--reencode] [--measure-only]

Inputs, read only: <packd>/chunks/*.mp4, <packd>/sidecars/*.json, <packd>/segments.json (+ <packd>/chunk_offsets.json
of the whole-file build, only to report which old spans were cut). Everything written stays under the repo root:
 1. ffprobe every chunk; covered_end = largest segment_end_sec of its VSS segments.
 2. stride of every same-day seam: early frames of chunk k+1 (0.3-14 s) are matched against chunk k from 12 s on
    (160x90 grey MSE; stm.seams.best_match / stride_from_matches). Cached in packd/strides.json.
 3. keep_end per chunk (stm.seams.plan, snapped to the 20 fps grid).
 4. each chunk trimmed to [0, keep_end), 1280x720, 20 fps CFR (day 1 is 19.92 fps, day 2 19.83 fps; the sidecars
    count 20), libx264 like build_packd.py -> packd/parts/; the parts are joined by the concat demuxer with stream
    copy -> packd/neighborhood_2day_v2_720p.mp4. Offsets use the encoded part lengths, then are checked by frame.
 5. sidecar frames at/after keep_end are dropped and segment starts shifted onto the new timeline; caption rows
    are trimmed the same way -> packd/segments.json, beside packd/chunk_offsets.json where stm.captions looks.
 6. cams.local.json, packd/chunk_offsets.json, packd/seams.json; then packd.register and data.write_meta exactly
    like build_packd.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stm import seams  # noqa: E402  (pure; stm.data is imported only after cams.local.json is written)

CAM = "neighborhood_cam-1"
FPS = 20.0
SIZE = (160, 90)
HEAD = [0.3, 0.8, 1.3, 1.8, 2.5, 3.5, 4.5, 5.5, 6.5, 8.0, 10.0, 12.0, 14.0]  # s into chunk k+1: queries
A_FROM = 12.0                            # s into chunk k: where the search starts (strides are ~25 s)
WORK = ROOT / "packd"
VIDEO = WORK / "neighborhood_2day_v2_720p.mp4"
TITLE = "Neighbourhood street camera (Pack D)"
CREDIT = "Event Pack D, neighborhood_cam-1, via team VSS (YOLO11 sidecars, Cosmos captions)"


def dur(p: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                          "default=nw=1:nk=1", str(p)]).decode().strip())


def nframes(p: Path) -> int:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
                                   "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(p)])
    return int(out.decode().strip().split(",")[0])


def read_small(path: Path, t0: float, t1: float) -> tuple[list[float], np.ndarray]:
    """Frame times (s from the first frame) and 160x90 grey frames in [t0, t1], read sequentially (no seeking)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    ts, fs = [], []
    while cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > t1:
            break
        if t >= t0 - 1e-3:
            ok, img = cap.retrieve()
            if ok:
                ts.append(round(t, 4))
                fs.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), SIZE, interpolation=cv2.INTER_AREA))
    cap.release()
    return ts, (np.stack(fs).astype(np.float32) if fs else np.zeros((0, SIZE[1], SIZE[0]), np.float32))


def measure_seam(job: tuple[str, str, str, str, float]) -> dict[str, Any]:
    """Stride of the seam a -> b by matching b's HEAD frames inside a (see stm.seams.stride_from_matches)."""
    import cv2
    cv2.setNumThreads(2)
    a, b, pa, pb, dur_a = job
    tb, fb = read_small(Path(pb), 0.0, max(HEAD) + 0.2)
    ta, fa = read_small(Path(pa), A_FROM, dur_a + 1.0)
    if not len(tb) or not len(ta):
        return {"a": a, "b": b, "stride": None, "reason": "no frames", "motion": 0.0, "matches": []}
    qi = sorted({int(np.argmin(np.abs(np.asarray(tb) - h))) for h in HEAD})
    q = fb[qi]
    motion = float(max(seams.mse_to(q, x).max() for x in q))
    matches = []
    for i in qi:
        t_a, cost, runner = seams.best_match(fb[i], fa, ta)
        matches.append([t_a, tb[i], round(cost, 3), round(runner, 3)])
    stride, reason = seams.stride_from_matches([tuple(m) for m in matches], dur_a)
    return {"a": a, "b": b, "stride": stride, "reason": reason, "motion": round(motion, 2), "matches": matches}


def measure(chunks: dict[str, Path], dur_: dict[str, float], cache: Path, remeasure: bool, jobs: int) -> dict:
    old = json.loads(cache.read_text()) if cache.exists() and not remeasure else {}
    todo = [(a, b, str(chunks[a]), str(chunks[b]), dur_[a]) for a, b in seams.same_day_seams(seams.order_chunks(chunks))
            if a not in old]
    if todo:
        t0 = time.time()
        with Pool(max(1, jobs)) as pool:
            for r in pool.imap_unordered(measure_seam, todo):
                old[r["a"]] = r
        print(f"measured {len(todo)} seams in {time.time() - t0:.0f} s")
    cache.write_text(json.dumps(old, indent=1))
    for r in old.values():  # the decision is re-made from the cached matches, so a rule change needs no decoding
        r["stride"], r["reason"] = seams.stride_from_matches([tuple(m) for m in r["matches"]], dur_[r["a"]])
    return old


def encode_part(c: Path, keep: float, out: Path) -> int:
    """chunk [0, keep) -> 1280x720 20 fps h264 part of exactly round(keep * FPS) frames; returns its frame count.

    trim cuts on the source timestamps; -frames:v stops the fps filter from adding a frame at the end of stream."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    n = int(round(keep * FPS))
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", str(c), "-an",
                           "-vf", f"trim=end={keep:.6f},setpts=PTS-STARTPTS,fps={FPS:g},scale=1280:720",
                           "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-g", "20",
                           "-pix_fmt", "yuv420p", str(tmp)])
    tmp.replace(out)
    got = nframes(out)
    if got != n:
        print(f"warning: {c.name} encoded {got} frames, planned {n}; offsets follow the encoded length")
    return got


def encode(p: dict[str, Any], chunks: dict[str, Path], jobs: int, reencode: bool) -> dict[str, float]:
    """Encode every part (reusing parts whose keep_end has not changed); {chunk: encoded length in s}."""
    man_p = WORK / "parts" / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() and not reencode else {}
    todo = [r for r in p["rows"] if man.get(r["chunk"], {}).get("keep_end") != r["planned_keep_end"]
            or not (WORK / "parts" / f"{r['chunk']}.mp4").exists()]
    t0 = time.time()
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        futs = {r["chunk"]: (r["planned_keep_end"], ex.submit(encode_part, chunks[r["chunk"]], r["planned_keep_end"],
                                                              WORK / "parts" / f"{r['chunk']}.mp4")) for r in todo}
        for ch, (keep, f) in futs.items():
            man[ch] = {"keep_end": keep, "frames": f.result()}
    man_p.parent.mkdir(parents=True, exist_ok=True)
    man_p.write_text(json.dumps(man, indent=1))
    if todo:
        print(f"encoded {len(todo)} parts in {time.time() - t0:.0f} s")
    return {r["chunk"]: round(man[r["chunk"]]["frames"] / FPS, 6) for r in p["rows"]}


def join_needed(reencode: bool) -> bool:
    """True when the joined video is missing or was joined from other parts than the current manifest lists."""
    man, joined = WORK / "parts" / "manifest.json", WORK / "parts" / "joined.json"
    if reencode or not VIDEO.exists() or not joined.exists():
        return True
    return json.loads(man.read_text()) != json.loads(joined.read_text())


def join(p: dict[str, Any]) -> None:
    """Parts back to back by the concat demuxer (stream copy); records which manifest it joined."""
    lst = WORK / "concat_v2.txt"
    lst.write_text("".join(f"file '{WORK / 'parts' / (r['chunk'] + '.mp4')}'\n" for r in p["rows"]))
    tmp = VIDEO.with_suffix(".tmp.mp4")
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                           "-movflags", "+faststart", str(tmp)])
    tmp.replace(VIDEO)
    (WORK / "parts" / "joined.json").write_text((WORK / "parts" / "manifest.json").read_text())


def verify(p: dict[str, Any]) -> dict[str, Any]:
    """Frame count of the joined video vs the plan, and each part's frame at +1 s vs the joined video at offset+1 s
    (stream copy: an aligned seam gives an identical frame)."""
    import cv2
    n = nframes(VIDEO)
    want = int(round(sum(r["keep_end"] for r in p["rows"]) * FPS))
    cap = cv2.VideoCapture(str(VIDEO))
    worst, shifted = 0.0, []
    for r in p["rows"]:
        t = min(1.0, r["keep_end"] / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((r["offset"] + t) * FPS)))
        ok, img = cap.read()
        pc = cv2.VideoCapture(str(WORK / "parts" / f"{r['chunk']}.mp4"))
        pc.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * FPS)))
        ok2, ref = pc.read()
        pc.release()
        if not (ok and ok2):
            shifted.append(r["chunk"])
            continue
        m = float(np.mean((img.astype(np.float32) - ref.astype(np.float32)) ** 2))
        worst = max(worst, m)
        if m > 1.0:
            shifted.append(r["chunk"])
    cap.release()
    return {"frames": n, "frames_planned": want, "seam_frame_mse_max": round(worst, 3), "seams_misaligned": shifted}


def sidecar_rows(sides: list[dict[str, Any]], keep: dict[str, float], offs: dict[str, float],
                 starts: dict[str, float] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Trim each sidecar at its chunk's keep_end and shift it onto the new timeline (segment_start_sec absolute)."""
    rows, st = [], {"in": len(sides), "kept": 0, "trimmed": 0, "dropped": 0, "frames_dropped": 0}
    for side in sides:
        chunk = side.get("chunk") or str(side.get("_file", "")).split("_segment_")[0]
        if side.get("segment_start_sec") is None and starts:
            side = dict(side, segment_start_sec=starts.get(side.get("_file", ""), 0.0))
        t = seams.trim_sidecar(side, keep[chunk])
        st["frames_dropped"] += len(side.get("frames", [])) - (len(t["frames"]) if t else 0)
        if t is None:
            st["dropped"] += 1
            continue
        st["trimmed" if t is not side else "kept"] += 1
        rows.append(dict(t, segment_start_sec=round(offs[chunk] + float(side.get("segment_start_sec") or 0.0), 6)))
    return rows, st


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packd", default=os.path.expanduser("~/packd"), help="read-only Pack D inputs")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--remeasure", action="store_true", help="ignore packd/strides.json")
    ap.add_argument("--reencode", action="store_true", help="ignore encoded parts")
    ap.add_argument("--measure-only", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("STM_TRACK_CACHE", str(ROOT / "_cache" / "tracks"))  # read when stm.data is imported
    pk = Path(a.packd)
    WORK.mkdir(parents=True, exist_ok=True)
    chunks = {c.stem: c for c in pk.glob("chunks/*.mp4")}
    dur_ = {k: dur(c) for k, c in chunks.items()}
    seg_obj = json.loads((pk / "segments.json").read_text())
    segs = seg_obj["segments"] if isinstance(seg_obj, dict) else seg_obj
    cov = seams.coverage(segs)
    meas = measure(chunks, dur_, WORK / "strides.json", a.remeasure, a.jobs)
    strides = {k: v["stride"] for k, v in meas.items()}
    reasons = {k: v["reason"] for k, v in meas.items()}
    inputs = {"chunks": [{"chunk": k, "duration": round(dur_[k], 6)} for k in seams.order_chunks(chunks)],
              "coverage": cov, "strides": strides, "reasons": reasons, "fps": FPS}
    p = seams.plan(inputs["chunks"], cov, strides, reasons, fps=FPS)
    print(seams.summary(p))
    if a.measure_only:
        for k in seams.order_chunks(meas):
            m = meas[k]
            print(k[-24:], m["stride"], m["reason"], m["motion"],
                  [(round(x[0] - x[1], 2), x[2], x[3]) for x in m["matches"]])
        return 0

    actual = encode(p, chunks, a.jobs, a.reencode)
    inputs["actual"] = actual
    p = seams.plan(inputs["chunks"], cov, strides, reasons, fps=FPS, actual=actual)
    off = seams.offsets(p)
    keep = {r["chunk"]: r["keep_end"] for r in p["rows"]}
    rejoined = join_needed(a.reencode)
    if rejoined:
        t0 = time.time()
        join(p)
        print("video", VIDEO, round(VIDEO.stat().st_size / 1e6, 1), "MB", round(time.time() - t0), "s")
    chk = verify(p)
    print("verify", chk)

    files = sorted(pk.glob("sidecars/*.json"))
    sides = [dict(json.loads(f.read_text()), _file=f.stem) for f in files]
    starts = {Path(str(s.get("filename") or "")).stem: float(s.get("segment_start_sec") or 0.0) for s in segs}
    rows, side_st = sidecar_rows(sides, keep, off, starts)
    for r in rows:
        r.pop("_file", None)
    print(len(rows), "sidecars", side_st)
    kept_segs, dropped_segs = seams.trim_segments(segs, keep)
    seg_out = dict(seg_obj, segments=kept_segs) if isinstance(seg_obj, dict) else kept_segs
    (WORK / "segments.json").write_text(json.dumps(seg_out, indent=1))

    days = p["days"]
    cam_entry = {"video": str(VIDEO), "title": TITLE, "days": days, "simulated": [], "credit": CREDIT,
                 "day_note": "Day labels come from the recording file names"}
    (ROOT / "cams.local.json").write_text(json.dumps({CAM: cam_entry}, indent=1))
    (WORK / "chunk_offsets.json").write_text(json.dumps({"offsets": off, "days": days, "total": p["totals"]["kept_s"],
                                                         "keep_end": keep, "layout": "seam-safe v2 (stm.seams)"},
                                                        indent=1))
    old_p = pk / "chunk_offsets.json"
    old_off = json.loads(old_p.read_text())["offsets"] if old_p.exists() else {}
    rec = {"camera": CAM, "video": str(VIDEO), "fps": FPS, "rule": seams.__doc__.split("Pure functions")[0].strip(),
           "inputs": inputs, "plan": p, "measurements": meas, "verify": chk, "sidecars": side_st,
           "corpus": {"segments_in": len(segs), "kept": len(kept_segs),
                      "clipped": sum("segment_end_sec_orig" in s for s in kept_segs), "dropped": len(dropped_segs),
                      "dropped_ids": [s.get("filename") for s in dropped_segs]},
           "uncovered_new": seams.uncovered(p, cov), "removed_old_timeline": seams.removed_spans(p, old_off)}
    (WORK / "seams.json").write_text(json.dumps(rec, indent=1))

    from stm import data, packd  # noqa: E402  (after cams.local.json exists)

    if rejoined:  # the median plates of a previous layout no longer match the video; modules redraw the day plates
        for bg in data.OUT.joinpath(CAM).glob("background*.jpg"):
            bg.unlink()
    t0 = time.time()
    out = packd.register(CAM, VIDEO, rows, days=[tuple(d) for d in days], title=TITLE, credit=CREDIT)
    df = data.tracks(CAM)
    obj = data.objects(CAM)
    print("registered", out, len(df), "dets", df.tid[df.tid >= 0].nunique(), "tracks", len(obj), "objects",
          int(obj.parked.sum()) if len(obj) else 0, "parked", round(time.time() - t0), "s")
    print(obj.groupby(["day", "name"]).size().to_string())
    m = data.write_meta(CAM)
    print("meta", m["duration"], m["w"], m["h"], m["fps"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
