"""Ego-motion proxies from dense optical flow on a downscaled road band (no model, CPU only)."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

HERE = Path(__file__).resolve().parent
W, H = 160, 90
HORIZON = 0.49  # fraction of frame height; fitted on dashcam_urban so static objects have equal ground-plane flow at all depths


def ego_signals(name: str, *, rate: float = 7.5, refresh: bool = False) -> dict[str, np.ndarray]:
    """Per sampled frame pair (ffmpeg decodes straight to 160x90 gray at `rate` fps, ~10x faster than OpenCV decode):
       t     time (s)
       speed forward-speed proxy: median over road pixels of vertical flow / (y - horizon)^2 (ground-plane flow law),
             relative units, NOT km/h (noisy: only used to tell a moving camera from a static one)
       yaw   horizontal flow of the skyline band, px/frame at 160 px wide (turning / panning)
       div   horizontal expansion rate of the road band (whole-scene looming)
       jerk  vertical flow of the far band (pitch dive / bump)
    """
    f = HERE / "_cache" / f"ego_{pk.name_of(name)}_r{rate:g}.npz"
    if f.exists() and not refresh:
        d = np.load(f)
        return {k: d[k] for k in d.files}
    ts, sp, yaw, dv, jk = [], [], [], [], []
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yh = HORIZON * H
    road = (yy > yh + 6) & (yy < 0.80 * H) & (xx > 0.12 * W) & (xx < 0.92 * W)
    far = (yy > yh - 14) & (yy < yh + 6)
    up = (yy < yh - 4) & (yy > 6)
    wgt = np.maximum(yy - yh, 1.0) ** 2
    cmd = [pk.FFMPEG, "-v", "error", "-i", str(pk.video_path(name)), "-vf", f"fps={rate},scale={W}:{H}:flags=area",
           "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    prev, i, fsz = None, 0, W * H
    while True:
        buf = proc.stdout.read(fsz)
        if len(buf) < fsz:
            break
        g = np.frombuffer(buf, np.uint8).reshape(H, W)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 13, 2, 5, 1.1, 0)
            u, v = fl[..., 0], fl[..., 1]
            sp.append(float(np.median(v[road] / wgt[road])) * 1000.0)
            yaw.append(float(np.median(u[up])))
            dv.append(float(np.median(u[road] / (xx[road] - W / 2 + 1e-3 * np.sign(xx[road] - W / 2 + 1e-6)))))
            jk.append(float(np.median(v[far])))
            ts.append(i / rate)
        prev, i = g, i + 1
    proc.wait()
    out = {"rate": np.array(rate), "t": np.array(ts), "speed": np.array(sp), "yaw": np.array(yaw), "div": np.array(dv),
           "jerk": np.array(jk)}
    np.savez_compressed(f, **out)
    return out


if __name__ == "__main__":
    import time
    n = sys.argv[1]
    t0 = time.time()
    e = ego_signals(n, refresh=True)
    print(n, len(e["t"]), f"{time.time() - t0:.1f}s")
    for k in ("speed", "yaw", "div", "jerk"):
        a = e[k]
        print(k, np.percentile(a, [5, 50, 95]).round(3))
