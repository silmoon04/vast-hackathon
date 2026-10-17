"""Build a Find & Follow scene (ui/find.html?scene=<name>) from a camera topology file.

    <py> scripts/build_find_scene.py --cams examples/follow/cams.follow.json --scene cafe [--out out/find]
                                     [--caption] [--seconds N]

1. stm.fastindex indexes every camera at 2 fps (MOG2 blobs, one thread per camera) while ffmpeg writes each video
   as a browser-friendly H.264 mp4: an 8-bit 4:2:0 H.264 stream is copied without re-encoding, anything else is
   re-encoded (the file is copied as is when ffmpeg is not on PATH).
2. stm.follow merges fragments per camera, links cameras by the topology and builds identities.
3. Item spans come from stm.items.item_span and descriptions from stm.caption.describe_crops (--caption) when those
   modules exist; otherwise an item is shown for the whole footage and people get the HSV colour words.
4. Writes <out>/<scene>/scene.json (v1), cam<ID>.mp4 and crops/<track>.jpg, checks the scene and prints the build time.

Where find.html reads a field differently from the v1 sketch, the scene follows find.html:
- clock0 is an ISO date-time ("2026-10-17T13:00:00"); find.html does new Date(clock0). A topology "HH:MM:SS" gets the
  topology "date" (default today).
- camera pos is in the map's 500 x 150 SVG units; a topology pos in 0..1 is scaled.
- offset is the scene second at which the video starts: find.html plays video t = scene t - offset. The topology's
  offset_s has the same meaning. (stm.items counts the other way, video t - offset_s, so it is called with -offset.)
- exits is a text label of where the camera's way out of the network leads (find.html prints "heading for the <exits>")
  or null; the zone map goes to exitZones.
- blind link targets are also listed in links (the map only draws a blind link that is in links).
- item t1 is never null (find.html compares and plots it): unknown or "never went" = end of footage, with `gone`.
  A topology item may give `ref_t` (a scene second it is known to be in its box); it is passed to stm.items.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stm import fastindex, follow  # noqa: E402

OUT = ROOT / "out" / "find"
NAME_RE = re.compile(r"^[\w.-]{1,64}$")       # the scene names find.html accepts
MAP_W, MAP_H = 500, 150                       # find.html #map viewBox
TRACK_KINDS = {"person", "bike", "bag"}
ITEM_KINDS = {"phone", "bike", "bag"}


# ---------------------------------------------------------------- topology
def load_topology(path: str | Path) -> dict[str, Any]:
    """cams.follow.json with every camera video resolved against the file's folder."""
    p = Path(path)
    topo = json.loads(p.read_text(encoding="utf-8"))
    for c in topo["cameras"]:
        v = Path(c["video"])
        c["video"] = str(v if v.is_absolute() else (p.parent / v).resolve())
        c.setdefault("offset_s", 0.0)
    return topo


def iso_clock(clock0: str | None, date: str | None = None) -> str:
    """ISO date-time for find.html from "HH:MM[:SS]" (+ date, default today) or an ISO string."""
    day = date or dt.date.today().isoformat()
    if not clock0:
        return f"{day}T00:00:00"
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", clock0):
        hh, mm, *ss = clock0.split(":")
        return f"{day}T{int(hh):02d}:{int(mm):02d}:{int(ss[0]) if ss else 0:02d}"
    return dt.datetime.fromisoformat(clock0).replace(tzinfo=None).isoformat(timespec="seconds")


def map_pos(pos: list[float] | None, k: int, n: int) -> list[float]:
    """Topology pos 0..1 -> map SVG units (pixel values > 1 pass through); missing pos -> spread along a row."""
    if not pos:
        return [round(40 + 420 * (k / max(1, n - 1))), 75]
    x, y = float(pos[0]), float(pos[1])
    if x <= 1 and y <= 1:
        return [round(30 + x * (MAP_W - 60)), round(22 + y * (MAP_H - 60))]
    return [x, y]


def video_name(cam_id: str) -> str:
    return "cam" + re.sub(r"[^\w.-]", "_", cam_id) + ".mp4"


# ---------------------------------------------------------------- video
def ffmpeg_path() -> str | None:
    p = shutil.which("ffmpeg")
    if p:
        return p
    c = Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe")
    return str(c) if c.exists() else None


def browser_ready(ff: str, src: str) -> bool:
    """True when the video stream is already 8-bit 4:2:0 H.264 of at most 1920 x 1088 (every browser plays it), so it
    can be copied into the mp4 instead of re-encoded. False when ffprobe is missing or says otherwise."""
    probe = shutil.which("ffprobe") or str(Path(ff).with_name("ffprobe" + Path(ff).suffix))
    try:
        out = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name,pix_fmt,width,height", "-of", "json", src],
                             capture_output=True, text=True, timeout=30)
        st = json.loads(out.stdout)["streams"][0]
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError):
        return False
    return (st.get("codec_name") == "h264" and st.get("pix_fmt") in ("yuv420p", "yuvj420p")
            and int(st.get("width") or 1e9) <= 1920 and int(st.get("height") or 1e9) <= 1088)


def start_transcode(src: str, dst: Path, seconds: float | None) -> tuple[subprocess.Popen, str] | None:
    """Start ffmpeg: copy an H.264 stream that browsers already play into a faststart mp4 (no re-encode, timestamps
    kept), else re-encode to H.264 yuv420p <= 960 px wide with 2 threads, so indexing keeps most cores. No audio.
    (process, what it does), None when ffmpeg is missing."""
    ff = ffmpeg_path()
    if not ff:
        return None
    lim = ["-t", f"{seconds:g}"] if seconds else []
    if browser_ready(ff, src):
        cmd = [ff, "-nostdin", "-y", "-v", "error", "-i", src] + lim + [
            "-map", "0:v:0", "-an", "-c:v", "copy", "-movflags", "+faststart", str(dst)]
    else:
        cmd = [ff, "-nostdin", "-y", "-v", "error", "-threads", "2", "-i", src] + lim + [
            "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
            "-threads", "2", "-vf", "scale='trunc(min(960,iw)/2)*2':-2", "-movflags", "+faststart", str(dst)]
    return (subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE),
            "h264 copied as is" if "copy" in cmd else "ffmpeg h264")


def finish_transcode(job: tuple[subprocess.Popen, str] | None, src: str, dst: Path, log: Callable[[str], None]) -> str:
    if job is not None:
        proc, how = job
        _, err = proc.communicate()
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            return how
        log(f"ffmpeg failed for {src} (code {proc.returncode}): {err.decode(errors='replace')[-300:]}; copying")
    shutil.copyfile(src, dst)
    return "copied (not transcoded)"


# ---------------------------------------------------------------- optional modules
def describe(tracks: list[dict[str, Any]], folder: Path, log: Callable[[str], None]) -> str:
    """--caption: replace desc / look with stm.caption.describe_crops when it exists. Returns what was used."""
    try:
        from stm import caption
    except ImportError:
        return "hsv colour words (stm.caption not available)"
    todo = [t for t in tracks if t.get("crop")]
    try:
        res = caption.describe_crops([str(folder / t["crop"]) for t in todo])
    except Exception as e:     # a captioner failure must not lose the scene
        log(f"stm.caption.describe_crops failed: {e!r}; keeping hsv colour words")
        return "hsv colour words (captioner failed)"
    if not isinstance(res, list) or len(res) != len(todo):
        log("stm.caption.describe_crops returned the wrong number of results; keeping hsv colour words")
        return "hsv colour words (captioner result unusable)"
    for t, r in zip(todo, res):
        if isinstance(r, dict) and r.get("desc"):
            t["desc"] = str(r["desc"])
            if isinstance(r.get("look"), dict):
                t["look"] = {"items": [], **r["look"]}
    return "stm.caption.describe_crops"


def item_span(video: str, box: list[float], offset: float, log: Callable[[str], None],
              ref_t: float | None = None) -> dict[str, Any] | None:
    """stm.items.item_span in scene seconds (its times are video seconds - offset_s, so offset_s = -camera offset;
    ref_t is in the same scene seconds). None when the module is missing or fails."""
    try:
        from stm import items
    except ImportError:
        return None
    try:
        extra = {"ref_t": ref_t} if ref_t is not None else {}
        return items.item_span(video, box, fps=2, offset_s=-offset, **extra)
    except Exception as e:
        log(f"stm.items.item_span failed for {video}: {e!r}")
        return None


def build_items(topo: dict[str, Any], sizes: dict[str, list[int]], tracks: list[dict[str, Any]],
                idents: list[dict[str, Any]], duration: float, log: Callable[[str], None],
                ends: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Scene items. ends: scene second each camera's (possibly --seconds cut) footage ends; an item still visible then,
    or whose span is unknown, gets t1 = that end and gone false."""
    ident_of = {t: i["id"] for i in idents for t in i["tracks"]}
    out = []
    for c in topo["cameras"]:
        W, H = sizes[c["id"]]
        start = max(0.0, float(c["offset_s"]))
        end = min(duration, (ends or {}).get(c["id"], duration))
        for it in c.get("items") or []:
            box = [float(v) for v in it["box"]]
            if max(box) > 1:                       # pixels -> 0..1
                box = [box[0] / W, box[1] / H, box[2] / W, box[3] / H]
            ref_t = float(it["ref_t"]) if it.get("ref_t") is not None else None
            span = item_span(c["video"], box, float(c["offset_s"]), log, ref_t)
            t0 = span.get("t0") if span else None
            t1 = span.get("t1") if span else None
            if span is None:
                ev = "stm.items not available: item span unknown, shown for the whole footage"
            else:
                ev = str(span.get("evidence") or "")
            rec = {"id": it["id"], "kind": it.get("kind", "bag"), "label": it.get("label", it["id"]), "cam": c["id"],
                   "owner": None, "box": [round(v, 4) for v in box],
                   "t0": round(min(max(0.0, float(t0)), end), 2) if t0 is not None else round(min(start, end), 2),
                   "t1": round(min(max(0.0, float(t1)), end), 2) if t1 is not None else round(end, 2),
                   "gone": t1 is not None and float(t1) <= end, "evidence": ev}
            rec["t1"] = max(rec["t1"], rec["t0"])
            rec["owner"] = follow.item_owner(rec, tracks, ident_of)
            out.append(rec)
    return out


# ---------------------------------------------------------------- scene
def exit_label(cam: dict[str, Any], ids: set[str]) -> str | None:
    outs = [str(v) for v in (cam.get("exits") or {}).values() if v not in ids]
    return " or ".join(dict.fromkeys(outs)) or None


def build(cams: str | Path, scene: str, out: str | Path = OUT, caption: bool = False, seconds: float | None = None,
          fps: float = fastindex.FPS, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Index, follow and write <out>/<scene>/scene.json. Returns the scene dict (plus a `build` block of timings)."""
    t_start, cpu_start = time.perf_counter(), time.process_time()
    if not NAME_RE.match(scene):
        raise ValueError(f"scene name {scene!r} must match {NAME_RE.pattern} (find.html rejects others)")
    topo = load_topology(cams)
    folder = Path(out) / scene
    (folder / "crops").mkdir(parents=True, exist_ok=True)
    for old in (folder / "crops").glob("*.jpg"):
        old.unlink()
    cameras = topo["cameras"]
    ids = {c["id"] for c in cameras}
    procs = {c["id"]: start_transcode(c["video"], folder / video_name(c["id"]), seconds) for c in cameras}
    res = fastindex.index_cameras([{"cam": c["id"], "video": c["video"]} for c in cameras], fps=fps, max_s=seconds)
    t_index, cpu_index = time.perf_counter() - t_start, time.process_time() - cpu_start
    raw = [follow.shift(t, float(c["offset_s"])) for c, r in zip(cameras, res) for t in r["tracks"]]
    merged, hops, idents = follow.follow(raw, cameras)
    t_follow = time.perf_counter() - t_start - t_index
    duration = max([float(c["offset_s"]) + r["duration"] for c, r in zip(cameras, res)] + [1.0])
    persons = []
    for t in merged:
        if t["kind"] != "person":
            continue
        t["boxes"] = [b for b in t["boxes"] if 0 <= b[0] <= duration]
        if not t["boxes"]:
            continue
        t["t0"], t["t1"] = t["boxes"][0][0], t["boxes"][-1][0]
        crop = "crops/" + re.sub(r"[^\w.-]", "_", t["id"]) + ".jpg"
        t["crop"] = crop if fastindex.save_crop(t, folder / crop) else None
        persons.append(t)
    keep = {t["id"] for t in persons}      # tracks wholly outside 0..duration are gone: rebuild the chains without them
    hops = [h for h in hops if h["from"] in keep and h["to"] in keep]
    idents = follow.identities(persons, hops)
    described = describe(persons, folder, log) if caption else "hsv colour words"
    items = build_items(topo, {c["id"]: r["size"] for c, r in zip(cameras, res)}, persons, idents, duration, log,
                        {c["id"]: float(c["offset_s"]) + r["duration"] for c, r in zip(cameras, res)})
    vids = {c["id"]: finish_transcode(procs[c["id"]], c["video"], folder / video_name(c["id"]), log) for c in cameras}
    cams_out = []
    for k, c in enumerate(cameras):
        blind = [b["to"] for b in c.get("blind") or [] if b["to"] in ids]
        links = [ln["to"] for ln in c.get("links") or [] if ln["to"] in ids]
        cams_out.append({"id": c["id"], "name": c.get("name", c["id"]), "pos": map_pos(c.get("pos"), k, len(cameras)),
                         "links": list(dict.fromkeys(links + blind)), "blindLinks": blind,
                         "props": c.get("props") or [], "video": video_name(c["id"]),
                         "offset": float(c["offset_s"]), "exits": exit_label(c, ids), "exitZones": c.get("exits") or {}})
    tracks_out = [{"id": t["id"], "cam": t["cam"], "kind": "person", "t0": t["t0"], "t1": t["t1"], "boxes": t["boxes"],
                   "desc": t["desc"], "look": t["look"], "crop": t["crop"], "fragments": t["fragments"]}
                  for t in sorted(persons, key=lambda t: (t["t0"], t["cam"]))]
    secs = time.perf_counter() - t_start
    sc = {"version": 1, "simulated": False, "title": topo.get("title", scene),
          "clock0": iso_clock(topo.get("clock0"), topo.get("date")), "credit": topo.get("credit", ""),
          "duration": round(duration, 2), "cameras": cams_out, "tracks": tracks_out, "identities": idents,
          "items": items,
          "build": {"seconds": round(secs, 2), "index_s": round(t_index, 2),
                    "index_cpu_s": round(cpu_index, 2), "follow_s": round(t_follow, 2),
                    "fps": fps, "seconds_limit": seconds, "describe": described, "videos": vids,
                    "fragments": len(raw), "person_tracks": len(persons), "hops": len(hops),
                    "per_camera": [{k: r[k] for k in ("camera", "size", "duration", "samples", "skipped", "seconds")}
                                   for r in res]}}
    problems = check_scene(sc, folder)
    sc["build"]["problems"] = problems
    (folder / "scene.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
    log(f"scene {scene}: {len(cameras)} cameras, {len(raw)} fragments -> {len(persons)} person tracks, "
        f"{len(idents)} identities, {len(hops)} hops, {len(items)} items")
    log(f"build time {secs:.1f} s (index {t_index:.1f} s using {cpu_index:.0f} CPU-s in this process, "
        f"follow {t_follow:.2f} s) -> {folder / 'scene.json'}")
    for p in problems:
        log("PROBLEM " + p)
    return sc


# ---------------------------------------------------------------- check
def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_scene(sc: dict[str, Any], folder: str | Path | None = None) -> list[str]:
    """Problems with a scene against v1 as find.html reads it (empty list = fine). With folder, files must exist."""
    p: list[str] = []
    folder = Path(folder) if folder else None
    if sc.get("version") != 1:
        p.append("version must be 1")
    if not isinstance(sc.get("simulated"), bool):
        p.append("simulated must be a bool")
    for k in ("title", "credit", "clock0"):
        if not isinstance(sc.get(k), str):
            p.append(f"{k} must be a string")
    try:
        dt.datetime.fromisoformat(sc.get("clock0", ""))
    except (TypeError, ValueError):
        p.append("clock0 must be an ISO date-time (find.html does new Date(clock0))")
    if not _num(sc.get("duration")) or sc["duration"] <= 0:
        p.append("duration must be a positive number")
    cams = sc.get("cameras") or []
    cids = [c.get("id") for c in cams]
    if not cams or len(set(cids)) != len(cids):
        p.append("cameras missing or ids not unique")
    for c in cams:
        w = f"camera {c.get('id')}"
        if not isinstance(c.get("name"), str):
            p.append(f"{w}: name")
        pos = c.get("pos")
        if not (isinstance(pos, list) and len(pos) == 2 and all(_num(v) for v in pos)
                and 0 <= pos[0] <= MAP_W and 0 <= pos[1] <= MAP_H):
            p.append(f"{w}: pos must be [x, y] inside the {MAP_W}x{MAP_H} map")
        if not set(c.get("links") or []) <= set(cids) or not set(c.get("blindLinks") or []) <= set(c.get("links") or []):
            p.append(f"{w}: links / blindLinks must name cameras (blindLinks also in links)")
        if not _num(c.get("offset")):
            p.append(f"{w}: offset must be a number")
        if c.get("exits") is not None and not isinstance(c.get("exits"), str):
            p.append(f"{w}: exits must be a string or null (find.html prints it)")
        if not isinstance(c.get("video"), str) or (folder and not (folder / c["video"]).exists()):
            p.append(f"{w}: video file missing")
    tracks = sc.get("tracks") or []
    tids = {t.get("id") for t in tracks}
    if len(tids) != len(tracks):
        p.append("track ids not unique")
    for t in tracks:
        w = f"track {t.get('id')}"
        if t.get("cam") not in cids or t.get("kind") not in TRACK_KINDS:
            p.append(f"{w}: cam / kind")
        b = t.get("boxes") or []
        if not b or not _num(t.get("t0")) or not _num(t.get("t1")) or t["t0"] > t["t1"]:
            p.append(f"{w}: t0 / t1 / boxes")
            continue
        if any(len(x) != 5 or not all(_num(v) for v in x) for x in b):
            p.append(f"{w}: boxes must be [t, x, y, w, h]")
            continue
        if [x[0] for x in b] != sorted(x[0] for x in b) or b[0][0] != t["t0"] or b[-1][0] != t["t1"]:
            p.append(f"{w}: box times must be sorted and span t0..t1")
        if any(x[1] < -1e-3 or x[2] < -1e-3 or x[3] <= 0 or x[4] <= 0 or x[1] + x[3] > 1.001 or x[2] + x[4] > 1.001
               for x in b):
            p.append(f"{w}: boxes must be normalised 0..1")
        if not isinstance(t.get("desc"), str) or not isinstance(t.get("look"), dict):
            p.append(f"{w}: desc / look")
        if t.get("crop") is not None and (not isinstance(t["crop"], str) or (folder and not (folder / t["crop"]).exists())):
            p.append(f"{w}: crop file missing")
    seen: dict[str, str] = {}
    by_id = {t["id"]: t for t in tracks if "id" in t}
    for i in sc.get("identities") or []:
        w = f"identity {i.get('id')}"
        ts = i.get("tracks") or []
        if not ts or any(x not in tids for x in ts):
            p.append(f"{w}: tracks must be known track ids")
            continue
        for x in ts:
            if x in seen:
                p.append(f"{w}: track {x} also in {seen[x]}")
            seen[x] = i["id"]
        if [by_id[x]["t0"] for x in ts] != sorted(by_id[x]["t0"] for x in ts):
            p.append(f"{w}: tracks not in time order")
        pairs = list(zip(ts, ts[1:]))
        for h in i.get("hops") or []:
            if (h.get("from"), h.get("to")) not in pairs:
                p.append(f"{w}: hop {h.get('from')}->{h.get('to')} is not between consecutive tracks")
            if not _num(h.get("gap")) or h["gap"] < 0 or not _num(h.get("score")) or not 0 <= h["score"] <= 1 \
                    or not isinstance(h.get("why"), str):
                p.append(f"{w}: hop gap / score / why")
    if any(t.get("kind") == "person" and t["id"] not in seen for t in tracks):
        p.append("some person tracks are in no identity")
    iids = {i.get("id") for i in sc.get("identities") or []}
    for it in sc.get("items") or []:
        w = f"item {it.get('id')}"
        if it.get("cam") not in cids or not isinstance(it.get("label"), str):
            p.append(f"{w}: cam / label")
        if it.get("kind") not in ITEM_KINDS:
            p.append(f"{w}: kind should be one of {sorted(ITEM_KINDS)}")
        if it.get("owner") is not None and it["owner"] not in iids:
            p.append(f"{w}: owner must be an identity id or null")
        bx = it.get("box")
        if not (isinstance(bx, list) and len(bx) == 4 and all(_num(v) and 0 <= v <= 1 for v in bx)):
            p.append(f"{w}: box must be [x, y, w, h] in 0..1")
        if not _num(it.get("t0")) or not _num(it.get("t1")) or it["t1"] < it["t0"]:
            p.append(f"{w}: t0 / t1 must be numbers with t1 >= t0")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cams", required=True, help="cams.follow.json topology")
    ap.add_argument("--scene", required=True, help="scene name: <out>/<scene>/scene.json, find.html?scene=<scene>")
    ap.add_argument("--out", default=str(OUT), help="output root (default out/find)")
    ap.add_argument("--caption", action="store_true", help="describe crops with stm.caption when it exists")
    ap.add_argument("--seconds", type=float, default=None, help="use only the first N seconds of each video")
    a = ap.parse_args(argv)
    sc = build(a.cams, a.scene, a.out, caption=a.caption, seconds=a.seconds)
    return 1 if sc["build"]["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
