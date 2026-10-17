"""Download the openly licensed Commons test clips listed in a manifest, verify SHA-1, transcode to browser H.264 mp4.

Local tests only: never commit these clips, never ingest them into VSS, never write them to team S3 buckets.
usage: python -I fetch_videos.py MANIFEST_RAW.json OUT_DIR VIDEOS_JSON
"""
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
SLUGS = {"BLK-HDPTZ12": "ptz_parking", "Corporation Street": "rochester_night", "Dashcam Recording": "dashcam_urban",
         "DiagonalCrosswalk": "yonge_dundas", "Elephants Dream": "elephants_dream", "HD CCTV Camera": "retail_cctv",
         "Illegal parking": "lisbon_parking", "Motorway A40": "motorway_a40"}
raw, out_dir, videos_json = json.load(open(sys.argv[1], encoding="utf-8")), Path(sys.argv[2]), Path(sys.argv[3])
entries = json.loads(videos_json.read_text(encoding="utf-8")) if videos_json.exists() else {}
for item in raw:
    slug = next(v for k, v in SLUGS.items() if item["title"].startswith(k))
    d = out_dir / "raw" / slug  # one new directory per downloaded file
    d.mkdir(parents=True, exist_ok=True)
    src = d / ("source" + Path(item["url"]).suffix)
    if not src.exists() or hashlib.sha1(src.read_bytes()).hexdigest() != item["sha1"]:
        print("downloading", slug, f"{item['bytes'] / 1e6:.1f}MB", flush=True)
        req = urllib.request.Request(item["url"], headers={"User-Agent": "vast-hack-proto-test/0.1 (local research)"})
        with urllib.request.urlopen(req, timeout=120) as r, open(src, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if hashlib.sha1(src.read_bytes()).hexdigest() != item["sha1"]:
            print("SHA-1 MISMATCH, skipping", slug); continue
    mp4 = out_dir / f"{slug}.mp4"
    if not mp4.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", "scale='trunc(min(1280,iw)/2)*2':-2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-b:a", "128k", "-movflags", "+faststart", str(mp4)], check=True)
    entries[slug] = {"path": str(mp4).replace("\\", "/"), "source": item["page"], "title": item["title"],
                     "license": item["license"], "license_url": item.get("license_url"), "author": item["author"],
                     "origin": "wikimedia_commons"}
    videos_json.write_text(json.dumps(entries, indent=1, ensure_ascii=False), encoding="utf-8")
    print("ready", slug, flush=True)
print("done", len(entries))
