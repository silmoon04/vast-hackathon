"""Build the Pack D camera (neighborhood_cam-1) on the lab VM: one 2-day video + a track cache + cams.local.json.

Chunks are 30 s pieces of each day's recording; segment_start_sec in each sidecar is relative to its chunk,
so absolute time = (sum of earlier chunk durations in the combined video) + segment_start_sec.
Run from the street-time-machine root: python scripts/build_packd.py --pack data/packd
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("STM_TRACK_CACHE", str(ROOT / "_cache" / "tracks"))
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pack", type=Path, default=ROOT / "data" / "packd")
parser.add_argument("--camera", default="neighborhood_cam-1")
args = parser.parse_args()
PK = args.pack.expanduser().resolve()
CAM = args.camera
VIDEO = PK / "neighborhood_2day_720p.mp4"
RX = re.compile(r"neighborhood_(\d{8})_chunk_(\d{4})")


def dur(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                          "default=nw=1:nk=1", str(p)]).decode().strip())


chunks = sorted(PK.glob("chunks/*.mp4"), key=lambda p: RX.search(p.name).groups())
if not chunks:
    raise SystemExit("No Pack D chunks found; run fetch_packd.py or supply --pack.")
offs, t, day_start = {}, 0.0, {}
for c in chunks:
    day, idx = RX.search(c.name).groups()
    day_start.setdefault(day, t)
    offs[c.stem] = t
    t += dur(c)
total = t
print(len(chunks), "chunks", round(total, 1), "s", {d: round(s, 1) for d, s in day_start.items()})

if not VIDEO.exists():
    lst = PK / "concat.txt"
    lst.write_text("".join("file '" + c.as_posix().replace("'", "'\\''") + "'\n" for c in chunks))
    t0 = time.time()
    subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-an",
                           "-vf", "scale=1280:720", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                           "-g", "20", "-pix_fmt", "yuv420p", str(VIDEO)])
    print("video", VIDEO, round(VIDEO.stat().st_size / 1e6, 1), "MB", round(time.time() - t0), "s")

rows = []
for f in sorted(PK.glob("sidecars/*.json")):
    side = json.loads(f.read_text())
    chunk = side.get("chunk") or f.name.split("_segment_")[0]
    side["segment_start_sec"] = offs[chunk] + float(side.get("segment_start_sec") or 0.0)
    rows.append(side)
print(len(rows), "sidecars")

days_sorted = sorted(day_start)
labels = {d: f"{int(d[6:])} {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(d[4:6]) - 1]}"
          for d in days_sorted}
days = []
for i, d in enumerate(days_sorted):
    end = day_start[days_sorted[i + 1]] if i + 1 < len(days_sorted) else None
    days.append([d, labels[d], round(day_start[d], 3), None if end is None else round(end, 3)])
cam_entry = {"video": str(VIDEO), "title": "Neighbourhood street camera (Pack D)", "days": days,
             "simulated": [],
             "credit": "Event Pack D, neighborhood_cam-1, via team VSS (YOLO11 sidecars, Cosmos captions)",
             "day_note": "Day labels come from the recording file names"}
local_config = ROOT / "cams.local.json"
existing = json.loads(local_config.read_text(encoding="utf-8")) if local_config.exists() else {}
existing[CAM] = cam_entry
local_config.write_text(json.dumps(existing, indent=1), encoding="utf-8")
(PK / "chunk_offsets.json").write_text(json.dumps({"offsets": offs, "days": days, "total": total}, indent=1))

from stm import data, packd  # noqa: E402  (after cams.local.json exists)

t0 = time.time()
out = packd.register(CAM, VIDEO, rows, days=[tuple(d) for d in days], title=cam_entry["title"],
                     credit=cam_entry["credit"])
df = data.tracks(CAM)
obj = data.objects(CAM)
print("registered", out, len(df), "dets", df.tid[df.tid >= 0].nunique(), "tracks", len(obj), "objects",
      int(obj.parked.sum()) if len(obj) else 0, "parked", round(time.time() - t0), "s")
print(obj.groupby(["day", "name"]).size().to_string())
m = data.write_meta(CAM)
print("meta", m["duration"], m["w"], m["h"], m["fps"])
