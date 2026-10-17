"""SIMULATED fault copies of real footage, used as ground truth for the blind-window detector.

Everything produced here is synthetic and is labelled SIMULATED wherever it is shown:
  cover  black lens cover        cloth  flat grey cloth over the lens (+ sensor noise)
  freeze last good frame repeated blur   defocus (Gaussian sigma 12)
  damage a scratch + dent painted on the target car after a time (not a fault of the camera: a thing that happened to the car)
The detections for a simulated copy are derived from the real cached tracks of the same frames: nothing is detected
in cover/cloth/blur windows (a covered/defocused lens detects nothing; SIMULATED assumption) and the last
detections are repeated during a freeze.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402


def paint_damage(fr: np.ndarray, box) -> np.ndarray:
    """A visible scratch (3 light diagonal strokes) and a dent shadow on the car's flank. Returns fr (in place)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ov = fr.copy()
    cx, cy = int(x1 + 0.42 * w), int(y1 + 0.68 * h)
    cv2.ellipse(ov, (cx, cy), (max(4, int(0.16 * w)), max(3, int(0.10 * h))), -15, 0, 360, (25, 25, 30), -1)
    cv2.addWeighted(ov, 0.55, fr, 0.45, 0, fr)
    for k in range(3):
        a = (int(x1 + (0.12 + 0.03 * k) * w), int(y1 + (0.52 + 0.07 * k) * h))
        b = (int(x1 + (0.72 + 0.03 * k) * w), int(y1 + (0.86 + 0.02 * k) * h))
        cv2.line(fr, a, b, (215, 215, 220), max(1, int(w / 90)), cv2.LINE_AA)
    return fr


def make(base: str, t0: float, t1: float, faults: list[dict[str, Any]], out_path: Path, *, damage: dict[str, Any] | None = None
         ) -> dict[str, Any]:
    """Write the fault copy of base[t0, t1) to out_path (mp4, original size/fps). faults use times relative to t0."""
    meta = pk.info(base)
    fps = meta["fps"]
    rng = np.random.default_rng(7)
    last_good = None
    last_out = None
    with pk.Mp4Writer(out_path, fps) as w:
        for i, t, fr in pk.iter_frames(base, start=t0, end=t1):
            ts = t - t0
            fr = fr.copy()
            if damage and ts >= damage["t"]:
                paint_damage(fr, damage["box"])
            out = fr
            for f in faults:
                if f["start"] <= ts < f["end"]:
                    if f["kind"] == "cover":
                        out = np.zeros_like(fr)
                    elif f["kind"] == "cloth":
                        out = np.clip(np.array([100, 108, 112], np.float32) + rng.normal(0, 2.0, fr.shape), 0, 255).astype(np.uint8)
                    elif f["kind"] == "blur":
                        out = cv2.GaussianBlur(fr, (0, 0), 12)
                    elif f["kind"] == "freeze":
                        out = last_out if last_out is not None else fr
                    break
            if not any(f["start"] <= ts < f["end"] and f["kind"] == "freeze" for f in faults):
                last_out = out
            w.write(out)
    return {"base": base, "t0": t0, "t1": t1, "faults": faults, "damage": damage, "path": str(out_path),
            "gt_blind": [(f["start"], f["end"], f["kind"]) for f in faults], "simulated": True}


def sim_tracks(base_tr: pk.Tracks, meta: dict[str, Any], name: str) -> pk.Tracks:
    """Tracks for the fault copy: real tracks of the same frames, shifted to the copy's clock, with fault windows applied."""
    t0, t1 = meta["t0"], meta["t1"]
    r = base_tr.rows
    fshift = int(round(t0 * base_tr.src_fps))
    seg = r[(r["t"] >= t0) & (r["t"] < t1)].copy()
    seg["t"] = (seg["t"] - t0).round(3)
    seg["f"] = seg["f"] - fshift
    keep = np.ones(len(seg), bool)
    extra = []
    for f in meta["faults"]:
        win = (seg["t"] >= f["start"]) & (seg["t"] < f["end"])
        if f["kind"] in ("cover", "cloth", "blur"):
            keep &= ~win.to_numpy()
        elif f["kind"] == "freeze":
            keep &= ~win.to_numpy()
            ts = np.sort(seg["t"].unique())
            before = ts[ts < f["start"]]
            if len(before):
                last = seg[seg["t"] == before[-1]]
                for tt in ts[(ts >= f["start"]) & (ts < f["end"])]:
                    fi = int(seg.loc[seg["t"] == tt, "f"].iloc[0])
                    c = last.copy()
                    c["t"], c["f"] = tt, fi
                    extra.append(c)
    seg = pd.concat([seg[keep]] + extra).sort_values(["f"]).reset_index(drop=True)
    return dataclasses.replace(base_tr, video=name, duration=t1 - t0, rows=seg)
