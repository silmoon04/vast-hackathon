"""Stopped-vehicle detector (online, spatial matching: robust to tracker ID switches)."""
from __future__ import annotations
import numpy as np, pandas as pd

VEH = ("car", "truck", "bus", "motorcycle")


def iou_mat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (n,4), B (m,4) -> (n,m) IoU."""
    ix1 = np.maximum(A[:, None, 0], B[None, :, 0]); iy1 = np.maximum(A[:, None, 1], B[None, :, 1])
    ix2 = np.minimum(A[:, None, 2], B[None, :, 2]); iy2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


class Cand:
    __slots__ = ("box0", "box", "t0", "last", "n", "fired", "cls", "conf", "tids", "hist")

    def __init__(self, box, t, cls, conf, tid):
        self.box0 = np.array(box, float); self.box = self.box0.copy(); self.hist = [self.box0]
        self.t0 = t; self.last = t; self.n = 1; self.fired = False; self.cls = [cls]; self.conf = [conf]; self.tids = {tid}

    def add(self, box, t, cls, conf, tid):
        self.hist.append(np.array(box, float)); self.n += 1; self.last = t
        self.cls.append(cls); self.conf.append(conf); self.tids.add(tid)
        if len(self.hist) % 3 == 0:
            self.box = np.median(np.array(self.hist[-15:]), axis=0)


def stopped_alerts(rows: pd.DataFrame, fps: float, *, S: float = 5.0, k_rel: float = 0.12, min_iou: float = 0.55,
                   gap: float = 1.2, min_presence: float = 0.6, min_box: float = 28.0, min_conf: float = 0.3):
    """rows: tracker rows (f,t,tid,cls,name,conf,x1..y2). Streams frame by frame.
    A vehicle is 'stopped' when a spatially matched candidate keeps its centre within k = k_rel*box_diag px of its
    anchor for >= S seconds (detections may flicker: presence >= min_presence of sampled frames; gaps <= `gap` s).
    Matching is by position (not track id), so ByteTrack ID switches do not reset the clock.
    Returns alert dicts fired at the moment the S-th second is reached (t_alert), with t_stop = when it stopped."""
    r = rows[rows["name"].isin(VEH) & (rows["conf"] >= min_conf)].sort_values(["t", "f"])
    alerts, cands = [], []
    for t, g in r.groupby("t", sort=True):
        D = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        names, confs, tids = g["name"].tolist(), g["conf"].tolist(), g["tid"].tolist()
        used = set()
        if cands:
            B = np.array([c.box for c in cands]); B0 = np.array([c.box0 for c in cands])
            diag = np.hypot(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]); k = np.maximum(4.0, k_rel * diag)
            c0 = np.stack([(B0[:, 0] + B0[:, 2]) / 2, (B0[:, 1] + B0[:, 3]) / 2], 1)
            cd = np.stack([(D[:, 0] + D[:, 2]) / 2, (D[:, 1] + D[:, 3]) / 2], 1)
            dc = np.hypot(c0[:, None, 0] - cd[None, :, 0], c0[:, None, 1] - cd[None, :, 1])
            ov = iou_mat(B, D)
            ok = (dc <= k[:, None]) & (ov >= min_iou)
            score = np.where(ok, ov, -1.0)
            for i in np.argsort(-score.max(axis=1)):
                j = int(np.argmax(score[i]))
                if score[i, j] < 0 or j in used:
                    continue
                used.add(j); cands[i].add(D[j], t, names[j], confs[j], tids[j]); score[:, j] = -1.0
        for j in range(len(D)):
            if j not in used and (D[j, 2] - D[j, 0]) >= min_box and (D[j, 3] - D[j, 1]) >= min_box * 0.6:
                cands.append(Cand(D[j], t, names[j], confs[j], tids[j]))
        keep = []
        for c in cands:
            if t - c.last > gap:
                continue
            dur = c.last - c.t0
            if not c.fired and dur >= S:
                pres = c.n / max(1.0, dur * fps + 1)
                if pres >= min_presence:
                    c.fired = True
                    alerts.append({"t_alert": float(c.last), "t_stop": float(c.t0), "box": c.box.tolist(),
                                   "cls": max(set(c.cls), key=c.cls.count), "conf": float(np.mean(c.conf)),
                                   "tids": sorted(int(x) for x in c.tids), "presence": float(pres), "_c": c})
            keep.append(c)
        cands = keep
        if len(cands) > 1:  # same vehicle re-seeded by flicker: keep the oldest
            cands.sort(key=lambda c: c.t0)
            B = np.array([c.box for c in cands]); M = iou_mat(B, B)
            drop = set()
            for i in range(len(cands)):
                if i in drop:
                    continue
                for j in range(i + 1, len(cands)):
                    if M[i, j] > 0.8:
                        drop.add(j)
            cands = [c for i, c in enumerate(cands) if i not in drop]
    for a in alerts:
        c = a.pop("_c")
        a["t_end"] = float(c.last); a["dur"] = float(c.last - c.t0)
    return dedupe_alerts(alerts)


def dedupe_alerts(alerts: list[dict], min_iou: float = 0.5, slack: float = 1.5, same_spot_iou: float = 0.85,
                  occl_gap: float = 3.5) -> list[dict]:
    """One alert per vehicle: a flickering detector can re-seed a candidate on a car that is already alerted. Two alerts
    are the same vehicle when their boxes overlap (IoU > min_iou) AND their stopped intervals overlap (+- slack s).
    A different car that later pulls into the same spot has a disjoint interval and keeps its own alert."""
    order = sorted(alerts, key=lambda a: (a["t_stop"], -a["conf"]))
    kept: list[dict] = []
    for a in order:
        b = np.array([a["box"]], float)
        dup = False
        for k in kept:
            iou = iou_mat(b, np.array([k["box"]], float))[0, 0]
            if a["t_stop"] <= k["t_end"] + slack and k["t_stop"] <= a["t_end"] + slack and iou > min_iou:
                dup = True
                break
            # a stopped car hidden for a few seconds by a truck passing in front is re-seeded at the SAME pixels: same car
            if iou > same_spot_iou and a["t_stop"] <= k["t_end"] + occl_gap and k["t_stop"] <= a["t_end"] + occl_gap:
                dup = True
                break
        if not dup:
            kept.append(a)
    return sorted(kept, key=lambda a: (a["t_alert"], a["box"][0]))
