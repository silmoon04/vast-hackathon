"""Tiny greedy multi-object tracker for raw per-frame boxes that arrive without ids.

`track(df)` takes rows with f, t, cls, name, conf, x1, y1, x2, y2 (any existing `tid` is ignored) and returns
the same rows with a fresh `tid` column. Matching is greedy per frame on IoU plus normalised centre distance,
within a class group (YOLO flips car/truck/bus from frame to frame, so vehicles share a group). A track that
is unseen for more than `max_age` seconds is closed. Tracks with fewer than `min_hits` boxes get tid -1
(untracked), like ByteTrack's unconfirmed tracks.

`python -m stm.tracker [cams...]` drops the cached ByteTrack ids, re-tracks, and scores the result against them:
purity = fraction of our track's frames whose reference id is that track's majority reference id,
fragmentation = mean number of our tracks covering one reference track. Writes out/tracker_eval.json and
out/tracker_summary.txt.
"""
from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from stm import data

# coarse groups: detections only match tracks in the same group
_GROUPS = {"person": 0, "bicycle": 1, "motorcycle": 1,
           "car": 2, "truck": 2, "bus": 2, "train": 2, "airplane": 2, "boat": 2}


def class_group(name: str, cls: int, by: str = "group") -> int:
    if by == "cls":
        return int(cls)
    return _GROUPS.get(str(name), 100 + int(cls))


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (n,4) and b (m,4)."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-6)


def track(df: pd.DataFrame, max_age: float = 1.0, iou_min: float = 0.15, dist_max: float = 0.6,
          min_hits: int = 3, by: str = "group") -> pd.DataFrame:
    """Return a copy of df (sorted by f, original index labels kept so it aligns back) with a new `tid` column.

    max_age: seconds a lost track is kept for re-matching. iou_min / dist_max: a pair is a candidate when
    IoU >= iou_min or centre distance / box size <= dist_max (box size = sqrt of the larger area).
    by: "group" (coarse class groups, default) or "cls" (strict per class id).
    """
    df = df.drop(columns=[c for c in ("tid",) if c in df.columns]).sort_values(["f", "t"], kind="stable")
    n = len(df)
    tid = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return df.assign(tid=tid)
    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(float)
    ts = df["t"].to_numpy(float)
    grp = np.array([class_group(nm, c, by) for nm, c in zip(df["name"].astype(str), df["cls"].astype(int))])
    conf = df["conf"].to_numpy(float)
    fr = df["f"].to_numpy()
    # per-frame slices (df is sorted by f)
    cuts = np.flatnonzero(np.diff(fr)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, n]

    # active tracks as parallel lists
    t_box: list[np.ndarray] = []
    t_vel: list[np.ndarray] = []   # centre velocity px/s
    t_last: list[float] = []
    t_grp: list[int] = []
    t_id: list[int] = []
    next_id = 0
    for s, e in zip(starts, ends):
        now = ts[s]
        # drop tracks lost for too long
        keep = [k for k in range(len(t_id)) if now - t_last[k] <= max_age]
        if len(keep) != len(t_id):
            t_box = [t_box[k] for k in keep]; t_vel = [t_vel[k] for k in keep]
            t_last = [t_last[k] for k in keep]; t_grp = [t_grp[k] for k in keep]; t_id = [t_id[k] for k in keep]
        idx = np.arange(s, e)
        db = boxes[idx]
        assigned = np.full(len(idx), -1)
        if t_id:
            tb = np.stack(t_box)
            dt = now - np.asarray(t_last)
            shift = np.stack(t_vel) * dt[:, None]
            pred = tb + np.hstack([shift, shift])
            iou = _iou(pred, db)
            pc = (pred[:, :2] + pred[:, 2:]) / 2
            dc = (db[:, :2] + db[:, 2:]) / 2
            dist = np.hypot(pc[:, None, 0] - dc[None, :, 0], pc[:, None, 1] - dc[None, :, 1])
            pa = np.clip((pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]), 1, None)
            da = np.clip((db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1]), 1, None)
            size = np.sqrt(np.maximum(pa[:, None], da[None, :]))
            dn = dist / size
            ratio = np.maximum(pa[:, None], da[None, :]) / np.minimum(pa[:, None], da[None, :])
            same = np.asarray(t_grp)[:, None] == grp[idx][None, :]
            ok = same & (ratio <= 3.0) & ((iou >= iou_min) | (dn <= dist_max))
            score = np.where(ok, iou + 0.5 * np.clip(1 - dn / dist_max, 0, None), -1.0)
            # greedy: best pair first, higher-confidence detections win ties
            order = np.argsort(-(score + 1e-4 * conf[idx][None, :]), axis=None)
            used_t: set[int] = set()
            for flat in order:
                ti, di = divmod(int(flat), len(idx))
                if score[ti, di] < 0:
                    break
                if ti in used_t or assigned[di] >= 0:
                    continue
                used_t.add(ti)
                assigned[di] = ti
        for di, ti in enumerate(assigned):
            b = db[di]
            if ti >= 0:
                dt = max(now - t_last[ti], 1e-3)
                c_old = (t_box[ti][:2] + t_box[ti][2:]) / 2
                v = ((b[:2] + b[2:]) / 2 - c_old) / dt
                t_vel[ti] = 0.6 * t_vel[ti] + 0.4 * v
                t_box[ti] = b
                t_last[ti] = now
                tid[idx[di]] = t_id[ti]
            else:
                t_box.append(b); t_vel.append(np.zeros(2)); t_last.append(now)
                t_grp.append(int(grp[idx[di]])); t_id.append(next_id)
                tid[idx[di]] = next_id
                next_id += 1
    # unconfirmed (too short) tracks -> -1, renumber the rest 1..N in order of first appearance
    ids, first_pos, counts = np.unique(tid, return_index=True, return_counts=True)
    keep = (ids >= 0) & (counts >= min_hits)
    remap = np.full(next_id, -1, dtype=np.int64)
    kept = ids[keep][np.argsort(first_pos[keep], kind="stable")]
    remap[kept] = np.arange(1, len(kept) + 1)
    return df.assign(tid=remap[tid].astype(int))


def evaluate(ref: pd.DataFrame, ours: pd.DataFrame) -> dict[str, Any]:
    """Compare our ids with reference ids on the same rows (aligned on f + box)."""
    key = ["f", "x1", "y1", "x2", "y2"]
    m = ours[key + ["tid"]].merge(ref[key + ["tid"]].rename(columns={"tid": "ref"}), on=key, how="inner")
    m = m.drop_duplicates(key)
    both = m[(m.tid >= 0) & (m.ref >= 0)]
    if both.empty:
        return {"purity": None, "fragmentation": None, "rows_compared": 0}
    maj = both.groupby("tid").ref.agg(lambda s: s.value_counts().iat[0])
    purity = float(maj.sum() / len(both))
    frag = both.groupby("ref").tid.nunique()
    # also the reverse view: how many reference ids does one of our tracks swallow
    merges = both.groupby("tid").ref.nunique()
    return {"purity": round(purity, 4), "fragmentation": round(float(frag.mean()), 3),
            "fragmentation_median": float(frag.median()), "ref_ids_per_our_track": round(float(merges.mean()), 3),
            "rows_compared": int(len(both))}


def eval_cam(cam: str) -> dict[str, Any]:
    ref = data.tracks(cam)[data.COLS].copy()
    raw = ref.drop(columns=["tid"])
    t0 = time.perf_counter()
    ours = track(raw)
    secs = time.perf_counter() - t0
    nframes = int(raw.f.nunique())
    res = {"rows": int(len(ref)), "frames": nframes,
           "tracks_ours": int(ours.loc[ours.tid >= 0, "tid"].nunique()),
           "tracks_ref": int(ref.loc[ref.tid >= 0, "tid"].nunique()),
           "untracked_rows_ours": int((ours.tid < 0).sum()),
           "untracked_rows_ref": int((ref.tid < 0).sum()),
           "seconds": round(secs, 3), "frames_per_s": round(nframes / max(secs, 1e-9), 1)}
    res.update(evaluate(ref, ours))
    # long tracks only (what objects() would keep, >= 8 frames)
    lo = ours[ours.tid >= 0].tid.value_counts()
    lr = ref[ref.tid >= 0].tid.value_counts()
    res["tracks_ours_ge8"] = int((lo >= 8).sum())
    res["tracks_ref_ge8"] = int((lr >= 8).sum())
    return res


def main(argv: list[str]) -> int:
    cams = argv or list(data.CAMS)   # every cam with a cached track file; missing ones are reported as UNKNOWN
    report: dict[str, Any] = {}
    for cam in cams:
        try:
            report[cam] = eval_cam(cam)
        except FileNotFoundError as e:
            report[cam] = {"error": f"UNKNOWN: {e}"}
        print(cam, report[cam], flush=True)
    data.write_json(data.OUT / "tracker_eval.json", report)
    lines = ["tracker eval: cached ByteTrack ids dropped, rows re-tracked with stm.tracker.track (greedy IoU+centre,",
             "class groups, lost buffer 1.0 s, min 3 hits). purity = share of our track frames on that track's",
             "majority reference id; fragmentation = mean number of our tracks per reference track.",
             "The reference is the cached YOLO+ByteTrack ids, not hand-labelled ground truth: where the two",
             "disagree (e.g. ours merging two reference ids) it is UNKNOWN which one is right.", ""]
    for cam, r in report.items():
        if "error" in r:
            lines.append(f"{cam}: {r['error']}")
            continue
        lines.append(f"{cam}: rows {r['rows']}, frames {r['frames']}, tracks ours {r['tracks_ours']} vs ref "
                     f"{r['tracks_ref']} (>=8 frames: {r['tracks_ours_ge8']} vs {r['tracks_ref_ge8']}), "
                     f"purity {r['purity']}, fragmentation {r['fragmentation']}, ref ids per our track "
                     f"{r['ref_ids_per_our_track']}, untracked rows ours {r['untracked_rows_ours']}, "
                     f"{r['frames_per_s']} frames/s")
    (data.OUT / "tracker_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
