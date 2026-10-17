"""Pose source for LAST UPRIGHT.

1. protokit's shared cache (prototypes/_cache/poses) when the file exists: pk.poses(name) returns it instantly.
2. Otherwise, because the shared GPU lock can be held for many minutes by other spikes, the SAME model and tracker
   (yolo26s-pose + ByteTrack, imgsz 640, conf 0.25, ~15 fps) run on CPU and are cached under this folder's _cache/.
   Never touches protokit's cache or the GPU lock."""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import protokit as pk

HERE = Path(__file__).resolve().parent


def _stride(name: str, fps_target: float = 15.0) -> int:
    return max(1, int(round(pk.info(name)["fps"] / fps_target)))


def load(name: str, allow_cpu: bool = True) -> tuple[pd.DataFrame, str]:
    """-> (poses DataFrame, source) with source in {'protokit-cache', 'cpu-local'}."""
    stride = _stride(name)
    shared = pk.CACHE / "poses" / f"{pk.name_of(name)}__s{stride}.json.gz"
    if shared.exists():
        return pk.poses(name), "protokit-cache"
    f = HERE / "_cache" / f"{name}__s{stride}.json.gz"
    if f.exists():
        return pk._with_geom(pd.DataFrame(json.loads(gzip.decompress(f.read_bytes())))), "cpu-local"
    if not allow_cpu:
        raise FileNotFoundError(f"no cached poses for {name}")
    from ultralytics import YOLO
    m = YOLO(str(pk.MODELS / "yolo26s-pose.pt"), task="pose")
    rows, t0 = [], time.time()
    for i, t, frame in pk.iter_frames(name, stride=stride):
        res = m.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, imgsz=640, verbose=False, device="cpu")[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [-1] * len(res.boxes)
        for tid, cf, xy, kp in zip(ids, res.boxes.conf.tolist(), res.boxes.xyxy.tolist(), res.keypoints.data.tolist()):
            rows.append({"f": i, "t": round(t, 3), "tid": tid, "conf": round(cf, 3), "x1": xy[0], "y1": xy[1], "x2": xy[2],
                         "y2": xy[3], "kp": [[round(a, 1), round(b, 1), round(c, 2)] for a, b, c in kp]})
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(gzip.compress(json.dumps(rows).encode()))
    print(f"[poses_src] {name}: {len(rows)} detections on CPU in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return pk._with_geom(pd.DataFrame(rows)), "cpu-local"


if __name__ == "__main__":
    for n in sys.argv[1:]:
        d, src = load(n)
        print(n, src, len(d), d.tid.nunique())
