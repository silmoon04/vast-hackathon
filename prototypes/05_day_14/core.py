"""Day 14 core: mine candidate incident moments from tracks (+ ego-motion flow). Pure functions, no GPU.

Geometry is a *proxy*, not a calibration: a level dashcam, camera height ~1.2 m, focal ~0.45 x frame width and the horizon
at 52% of the frame height give a ground-plane range Z and a lateral offset X for every vehicle's ground-contact point
(bottom centre of its box). Everything reported from it is labelled 'proxy' and is only used to RANK moments.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protokit as pk  # noqa: E402
from ego import HORIZON  # noqa: E402

VEHICLES = ("car", "truck", "bus", "motorcycle", "bicycle")
FOCAL = 0.45             # assumed focal length as a fraction of frame width (wide dashcam lens)
CAM_H = 1.2              # assumed camera height, metres
EGO_HALF = 0.9           # half width of the ego car, metres
VEH_W = {"car": 1.8, "truck": 2.5, "bus": 2.5, "motorcycle": 0.8, "bicycle": 0.6}
WIN = 0.6                # seconds: regression window for growth / lateral rate


def sat(x):
    return np.clip(x, 0.0, 1.0)


def slope(t: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Centred least-squares slope of y(t) over 2k+1 samples (edges use what exists)."""
    out = np.zeros(len(y))
    for i in range(len(y)):
        a, b = max(0, i - k), min(len(y), i + k + 1)
        if b - a >= 3:
            tt = t[a:b] - t[a:b].mean()
            out[i] = float((tt * (y[a:b] - y[a:b].mean())).sum() / max((tt * tt).sum(), 1e-9))
    return out


def ego_speed(ego: dict | None, src_fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward-speed proxy on the road band in 1/(px*s) (ground-plane law), smoothed over ~1 s. Noisy: only used as a
    reference for the relative-motion test and to tell a moving camera from a static one."""
    if ego is None or not len(ego["t"]):
        return np.array([0.0]), np.array([np.nan])
    fl_fps = float(ego["rate"])
    sp = uniform_filter1d(median_filter(ego["speed"], 5, mode="nearest"), max(3, int(fl_fps * 0.7)), mode="nearest")
    return ego["t"], sp / 1000.0 * fl_fps / 8.0


def camera_mode(ego: dict | None) -> dict:
    """'dashcam' when the whole scene expands (forward motion: median |horizontal expansion| of the road band), 'fixed'
    when it does not (bridge camera, overhead lot camera, or a car that never moves)."""
    if ego is None or not len(ego["t"]):
        return {"mode": "unknown", "median_abs_div": float("nan")}
    div = float(np.median(np.abs(ego["div"])))
    return {"mode": "dashcam" if div > 0.008 else "fixed", "median_abs_div": round(div, 4),
            "rule": "dashcam if median |scene expansion| > 0.008 px/px/frame (dashcam_urban ~0.03, fixed cameras ~0.0001)"}


def per_track_features(tr: pk.Tracks, ego: dict | None, names=VEHICLES, min_samples: int = 5) -> pd.DataFrame:
    """One row per (track, sampled frame). Columns:
    Z, X         range / lateral offset proxies in metres (ground contact point, bottom centre of the box)
    clear        lateral clearance proxy to the ego path: |X| - half vehicle width - half ego width (m); <0 = in the path
    g, ttc       d ln(size)/dt (1/s) and time-to-contact proxy 1/g (s) when approaching
    lat, toward  lateral speed not explained by yaw / ego expansion (m/s) and its component toward the ego centre line
    ratio        relative-motion ratio: object ground-plane flow / road ground-plane flow (static ~1.35, oncoming/crossing
                 >2.5, same-direction slower leader <0.7); NaN while the camera is (nearly) stopped
    """
    W, H = tr.w, tr.h
    r = tr.rows[(tr.rows["tid"] >= 0) & tr.rows["name"].isin(names)]
    r = r[(r["bw"] < 0.9 * W) & (r["y2"] < 0.97 * H)]
    yh = HORIZON * H
    et, esp = ego_speed(ego, tr.src_fps)
    out = []
    k = max(1, int(round(WIN * tr.fps / 2)))
    k2 = max(2, int(round(1.0 * tr.fps / 2)))
    for tid, g in r.groupby("tid"):
        g = g.sort_values("t")
        if len(g) < min_samples:
            continue
        t = g["t"].to_numpy()
        s = np.sqrt(g["bw"].to_numpy() * g["bh"].to_numpy())
        ls = np.log(median_filter(s, 5, mode="nearest"))
        cx = median_filter(g["cx"].to_numpy(), 3, mode="nearest")
        y2s = median_filter(g["y2"].to_numpy(), 3, mode="nearest")
        gr = slope(t, ls, k)
        dcx, dy2 = slope(t, cx, k), slope(t, y2s, k)
        yaw = np.zeros(len(t))
        if ego is not None and len(ego["t"]):
            yaw = np.interp(t, ego["t"], ego["yaw"]) * (W / 160.0) * float(ego["rate"])
        dh = np.maximum(y2s - yh, 15.0)
        rho = (cx - W / 2) / dh                                    # lateral offset / camera height
        res_px = dcx - yaw - rho * dy2                             # px/s not explained by ego motion about the FOE
        lat = CAM_H * res_px / dh
        toward = -np.sign(cx - W / 2) * lat
        Z = FOCAL * W * CAM_H / dh
        X = rho * CAM_H
        nm = g["name"].mode().iloc[0]
        wm = VEH_W.get(nm, 1.8)
        clear = np.abs(X) - 0.5 * wm - EGO_HALF
        # relative-motion ratio over ~1 s windows
        rg = slope(t, y2s, k2) / dh ** 2
        ve = np.interp(t, et, esp) if len(et) > 1 else np.full(len(t), np.nan)
        ratio = np.where(ve > 60e-5, rg / np.maximum(ve, 1e-9), np.nan)   # vs the optical-flow speed proxy (noisy)
        out.append(pd.DataFrame({
            "tid": tid, "name": nm, "t": t, "f": g["f"].to_numpy(), "s": s, "g": gr, "Z": Z, "X": X, "clear": clear,
            "lat": lat, "toward": toward, "ratio": ratio, "ve": ve * 1e5, "rg": rg,
            "cx": g["cx"].to_numpy(), "y2": g["y2"].to_numpy(), "x1": g["x1"].to_numpy(), "x2": g["x2"].to_numpy(),
            "y1": g["y1"].to_numpy(), "bh": g["bh"].to_numpy(), "conf": g["conf"].to_numpy()}))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["ratio2"] = peer_ratio(df)
    return df


def peer_ratio(df: pd.DataFrame, half_window: float = 0.6, min_peers: int = 3, dx: float = 1.0) -> np.ndarray:
    """Relative-motion test without any ego sensor: static objects at the same lateral offset share the same
    ground-plane flow rg = v_ego/(f*h) (wide-angle distortion makes it depend on the offset, so peers are taken from
    the same side within +-1 m of X, e.g. the other parked cars along that kerb).
    ratio2 = rg / median(rg of those peers): ~1 static, >>1 oncoming/crossing, <<1 slower leader.
    NaN with fewer than 3 peers (then the moving evidence is 'unknown', not 'static')."""
    ref = df[(df["rg"] > 0) & (df["Z"] < 25) & (df["X"].abs() < 10)][["t", "tid", "rg", "X"]].sort_values("t")
    tt, rr, ii, xx = ref["t"].to_numpy(), ref["rg"].to_numpy(), ref["tid"].to_numpy(), ref["X"].to_numpy()
    out = np.full(len(df), np.nan)
    ts, tids, rg, X = df["t"].to_numpy(), df["tid"].to_numpy(), df["rg"].to_numpy(), df["X"].to_numpy()
    for n in range(len(df)):
        a, b = np.searchsorted(tt, ts[n] - half_window), np.searchsorted(tt, ts[n] + half_window)
        m = (ii[a:b] != tids[n]) & (np.abs(xx[a:b] - X[n]) < dx)
        if m.sum() >= min_peers:
            med = float(np.median(rr[a:b][m]))
            if med > 0:
                out[n] = rg[n] / med
    return out


def score(feat: pd.DataFrame, H: int = 720) -> pd.DataFrame:
    """Severity 0..1 per (track, frame):  how close  x  how much in the ego path  x  how dynamic  x  moving-object evidence.
    kind: 'closing' (in the lane and approaching fast), 'cut-in' (moving toward the lane), 'close pass' (alongside, near)."""
    if feat.empty:
        return feat
    f = feat.copy()
    f["ttc"] = np.where(f["g"] > 0.05, 1.0 / np.maximum(f["g"], 1e-6), np.inf)
    size_f = sat((f["bh"] / H - 0.06) / 0.12)
    near = sat((9.0 - f["Z"]) / 6.0)                               # 1 at <= 3 m, 0 at >= 9 m
    path = sat((2.2 - f["clear"]) / 2.2)                           # 1 when the vehicle overlaps the ego path, 0 at 2.2 m clear
    close = np.sqrt(near * size_f)
    approach = sat((f["g"] - 0.10) / 0.50)                         # growth 0.6/s ~ TTC 1.7 s
    cutin = sat((f["toward"] - 0.4) / 1.6)
    dyn = np.maximum(approach, cutin)
    r = f["ratio2"]
    moving_ev = np.where(r.isna(), 0.5, sat((np.abs(r - 1.0) - 0.45) / 0.9))
    moving_ev = np.maximum(moving_ev, sat(f["lat"].abs() / 1.5) * 0.8)
    f["close"], f["path"], f["dyn"], f["moving_ev"] = close, path, dyn, moving_ev
    f["sev"] = close * path * np.maximum(dyn, 0.30) * (0.35 + 0.65 * moving_ev)
    f["kind"] = np.where(cutin > 0.35, "cut-in / crossing", np.where(f["clear"] < 0.3, "closing", "close pass"))
    return f


def risk_curve(sc: pd.DataFrame, dur: float, step: float = 1 / 15) -> tuple[np.ndarray, np.ndarray]:
    """max-over-tracks severity at a regular time grid (a 0.4 s running mean per track first, to ignore single-frame blips)."""
    grid = np.arange(0, dur + step, step)
    curve = np.zeros(len(grid))
    if sc.empty:
        return grid, curve
    for tid, g in sc.groupby("tid"):
        g = g.sort_values("t")
        sev = uniform_filter1d(g["sev"].to_numpy(), 5, mode="nearest")
        v = np.interp(grid, g["t"].to_numpy(), sev, left=0, right=0)
        v[(grid < g["t"].min() - step) | (grid > g["t"].max() + step)] = 0
        curve = np.maximum(curve, v)
    return grid, curve


def pick_peaks(grid: np.ndarray, curve: np.ndarray, k: int = 5, min_sep: float = 8.0, min_sev: float = 0.03) -> list[int]:
    """Greedy top-k peaks with a minimum separation (so the 10 s clips barely overlap)."""
    order = np.argsort(-curve)
    chosen: list[int] = []
    for i in order:
        if curve[i] < min_sev:
            break
        if all(abs(grid[i] - grid[j]) >= min_sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= k:
            break
    return chosen
