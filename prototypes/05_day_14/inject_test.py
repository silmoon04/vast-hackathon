"""SIMULATED incident test. The real footage (dashcam_urban) contains no clear reportable incident, so this builds a
20 s excerpt of the same drive (96-116 s, a straight road) twice:

  ctrl_highway.mp4   unmodified excerpt (control)
  sim_cutin.mp4      the same excerpt with a SIMULATED cut-in: a car cut out of the real footage (YOLO26s-seg mask, the
                     grey hatchback seen from behind at 119.7 s) is composited from the adjacent lane into the ego lane
                     (range 14 m -> 4 m in 1.6 s, then it pulls away). The geometry uses the SAME ground-plane model as
                     core.py, so this tests ranking and plumbing, NOT real-world realism.

Both go through the identical pipeline (run.py); the question is whether the simulated cut-in is ranked #1 and the control
is not flagged above it. Everything produced here is labelled SIMULATED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protokit as pk  # noqa: E402
import core  # noqa: E402
import ego as egomod  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = "dashcam_urban"
T0, T1 = 96.0, 116.0
T_INJ = 8.0          # seconds into the excerpt when the cut-in starts
APPROACH, PULL = 1.6, 2.6


def cutout() -> tuple[np.ndarray, np.ndarray]:
    """RGBA-like (bgr, alpha) of the grey hatchback from behind at 119.7 s."""
    fr = pk.frame_at(SRC, 119.7)
    dets = pk.seg_masks_many([fr], classes=["car"], conf=0.3)[0]
    best = max(dets, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    x1, y1, x2, y2 = (int(v) for v in best["box"])
    m = best["mask"][y1:y2, x1:x2].astype(np.uint8) * 255
    m = cv2.GaussianBlur(m, (5, 5), 0)
    return fr[y1:y2, x1:x2].copy(), m


def place(frame: np.ndarray, car: np.ndarray, mask: np.ndarray, Z: float, X: float, W: int, H: int) -> None:
    f = core.FOCAL * W
    h_px = int(f * 1.45 / Z)
    w_px = max(8, int(h_px * car.shape[1] / car.shape[0]))
    y2 = int(core.HORIZON * H + f * core.CAM_H / Z)
    cx = int(W / 2 + f * X / Z)
    c = cv2.resize(car, (w_px, h_px), interpolation=cv2.INTER_CUBIC)
    a = cv2.resize(mask, (w_px, h_px), interpolation=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255.0
    x0, y0 = cx - w_px // 2, y2 - h_px
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w_px), min(H, y2)
    if xb <= xa or yb <= ya:
        return
    c, a = c[ya - y0:yb - y0, xa - x0:xb - x0], a[ya - y0:yb - y0, xa - x0:xb - x0]
    roi = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (a * c + (1 - a) * roi).astype(np.uint8)
    # soft contact shadow on the road
    sh = frame[max(0, y2 - 3):min(H, y2 + 3), xa:xb].astype(np.float32) * 0.7
    frame[max(0, y2 - 3):min(H, y2 + 3), xa:xb] = sh.astype(np.uint8)


def trajectory(s: float) -> tuple[float, float] | None:
    """(Z, X) of the simulated car at excerpt time s, or None when it is not on screen."""
    u = s - T_INJ
    if u < 0 or u > APPROACH + PULL:
        return None
    if u <= APPROACH:
        k = u / APPROACH
        e = k * k * (3 - 2 * k)
        return 14.0 - 10.0 * e, -3.4 + 3.2 * e
    k = (u - APPROACH) / PULL
    return 4.0 + 5.0 * k * k, -0.2 + 0.1 * k


def build() -> dict:
    meta = pk.info(SRC)
    W, H = meta["w"], meta["h"]
    car, mask = cutout()
    d = HERE / "out" / "sim"
    d.mkdir(parents=True, exist_ok=True)
    with pk.Mp4Writer(d / "ctrl_highway.mp4", meta["fps"]) as wc, pk.Mp4Writer(d / "sim_cutin.mp4", meta["fps"]) as ws:
        for i, t, fr in pk.iter_frames(SRC, start=T0, end=T1):
            s = t - T0
            wc.write(fr)
            g = fr.copy()
            tz = trajectory(s)
            if tz:
                place(g, car, mask, tz[0], tz[1], W, H)
            ws.write(g)
    return {"inject_at_s": T_INJ, "excerpt": [T0, T1], "approach_s": APPROACH, "car_from": f"{SRC} @119.7 s (seg mask)"}


if __name__ == "__main__":
    info = build()
    print(json.dumps(info))
    print(HERE / "out" / "sim")
