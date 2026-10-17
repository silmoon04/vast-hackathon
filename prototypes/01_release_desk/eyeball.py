"""Contact sheet for human eyeballing: source frames (green = released rectangle) over the exported frames, same indices.

    python eyeball.py [video ...]      -> out/<video>/eyeball.jpg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import protokit as pk  # noqa: E402


def frame(path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    return fr


def main() -> None:
    names = sys.argv[1:] or sorted(p.name for p in (HERE / "out").iterdir() if (p / "regions.json").exists())
    for name in names:
        out = HERE / "out" / name
        vid = json.loads((out / "release_record.json").read_text(encoding="utf-8"))["source"]["video"]
        reg = json.loads((out / "regions.json").read_text(encoding="utf-8"))
        n = len(reg["rects"])
        idx = sorted({int(round(k * (n - 1) / 5)) for k in range(6)})
        top, bot = [], []
        for i in idx:
            so = frame(pk.video_path(vid), reg["f0"] + i)
            for r in reg["rects"][i]:
                cv2.rectangle(so, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (60, 220, 60), 3)
            top.append(so)
            bot.append(frame(out / "export.mp4", i))
        labels = [f"source t={pk.mmss((reg['f0'] + i) / reg['fps'])} [{reg['modes'][i]}]" for i in idx]
        a = pk.sheet(top, labels, cols=len(idx), tile_w=300)
        b = pk.sheet(bot, ["EXPORT"] * len(idx), cols=len(idx), tile_w=300)
        pk.save_jpg(np.vstack([a, b]), out / "eyeball.jpg")
        print(out / "eyeball.jpg")


if __name__ == "__main__":
    main()
