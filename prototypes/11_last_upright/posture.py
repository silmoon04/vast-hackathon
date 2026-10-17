"""LAST UPRIGHT core: pose keypoints -> posture per person per frame -> events -> long-lie clock.

Pure numpy/pandas (no GPU, no network): the whole decision loop is testable on cached poses.

Posture states
  UPRIGHT  standing (torso near vertical, knees well below the hips)
  SITTING  torso near vertical, knees about level with the hips (or upper body only, legs hidden)
  DOWN     torso near horizontal (shoulder->hip vector) and a wide box: lying / collapsed
  PARTIAL  body axis near vertical but shoulders+hips not both seen (legs hidden by a desk, torso cut by the frame)
  UNKNOWN  too few visible keypoints to say

Events (per stitched person track)
  last_upright   last frame before the DOWN run whose smoothed state is UPRIGHT (or PARTIAL: body axis near vertical, only
                 legs / head+shoulders in view = still on their feet) with the body within UP_DEG_REF (30) of vertical.
                 A SITTING frame never counts: a sprawled person with a near-vertical torso is already on the ground.
  contact_t      first ground contact: first frame after last_upright with the ankles within 0.5 torso lengths of the hips
                 (hips at foot level) or the state DOWN
  flat_t / first_down   first second the torso is >= FLAT_DEG (70) from vertical / first second of the smoothed DOWN run
  interval       flat_t - last_upright (a detection gap counts in full); to_ground = contact_t - last_upright
  speed          FAST (<= FAST_S): FALL   |  MEDIUM/SLOW: deliberate lie-down  |  UNOBSERVED (gap) / PARTIAL (no torso
                 keypoints at the flat end): not called a fall
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

UP, SIT, DOWN, UNK, PART = "UPRIGHT", "SITTING", "DOWN", "UNKNOWN", "PARTIAL"
KP_VIS = 0.35          # keypoint confidence treated as visible
UP_DEG_REF = 30.0      # body tilt at the 'last upright' frame (smoothed state UPRIGHT or PARTIAL and within this of vertical)
UP_DEG = 25.0          # torso within this of vertical = upright reference (start of the transition clock)
FLAT_DEG = 70.0        # torso at least this far from vertical = flat (end of the transition clock)
DOWN_ENTER_DEG = 60.0  # torso angle that starts DOWN
DOWN_EXIT_DEG = 42.0   # hysteresis: torso must come back below this to leave DOWN
FAST_S = 1.5           # transition upper bound <= this -> FALL
SLOW_S = 4.0           # above this -> clearly deliberate
MIN_DWELL = {UP: 0.3, SIT: 0.3, DOWN: 0.4, UNK: 0.3, PART: 0.3}
GAP_S = 1.2            # a hole in the detections longer than this ends a track segment

# long-lie ladder (CONFIGURABLE demo defaults, not clinical guidance; the BGS poster found care-home limits
# from 1 h to over 6 h, so the last rung is the low end of that range)
LADDER = [(0, "ON FLOOR", "log"), (60, "CHECK NOW", "check"), (300, "ESCALATE", "urgent"), (3600, "LONG LIE", "long-lie")]


# ------------------------------------------------------------------------------------------------ features
def kp_array(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(df["kp"].tolist(), float).reshape(-1, 17, 3)


def _mid(kp: np.ndarray, idx: tuple[int, ...]) -> np.ndarray | None:
    pts = [kp[i, :2] for i in idx if kp[i, 2] >= KP_VIS]
    return np.mean(pts, axis=0) if pts else None


def features_row(kp: np.ndarray, box: tuple[float, float, float, float]) -> dict[str, float]:
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    f: dict[str, float] = {"aspect": bw / bh, "bh": bh, "bw": bw, "nvis": float((kp[:, 2] >= KP_VIS).sum()),
                           "theta": np.nan, "axis": np.nan, "tlen": np.nan, "thigh_v": np.nan, "leg_v": np.nan,
                           "head_v": np.nan, "hip_x": np.nan, "hip_y": np.nan, "tilt": np.nan,
                           "nface": float((kp[:5, 2] >= KP_VIS).sum())}
    S, H = _mid(kp, (5, 6)), _mid(kp, (11, 12))
    if S is not None and H is not None:
        v = H - S
        tl = float(np.hypot(*v))
        if tl > 4:
            f["tlen"] = tl
            f["theta"] = math.degrees(math.atan2(abs(v[0]), v[1]))   # 0 = shoulders straight above hips, 90 = horizontal
            f["hip_x"], f["hip_y"] = float(H[0]), float(H[1])
    vis = kp[kp[:, 2] >= KP_VIS][:, :2]
    f["tilt"] = np.nan
    if len(vis) >= 4:   # principal axis of all visible keypoints: works when shoulders or hips are hidden
        c = vis - vis.mean(0)
        w, vec = np.linalg.eigh(c.T @ c)
        d = vec[:, 1]
        if w[1] > 1.5 * max(w[0], 1e-6):   # elongated enough for an axis to mean something
            f["axis"] = math.degrees(math.atan2(abs(d[0]), abs(d[1])))
    if f["tlen"] == f["tlen"]:
        K, A = _mid(kp, (13, 14)), _mid(kp, (15, 16))
        if K is not None:
            f["thigh_v"] = float((K[1] - H[1]) / f["tlen"])      # knees below hips, in torso lengths (stand ~0.9, sit ~0)
        if A is not None:
            f["leg_v"] = float((A[1] - H[1]) / f["tlen"])        # ankles below hips (stand ~1.8, sit ~1, lying ~0)
        head = _mid(kp, (0, 1, 2, 3, 4))
        if head is not None:
            f["head_v"] = float((H[1] - head[1]) / f["tlen"])    # head above hips (upright ~1.2+, lying ~0)
    # tilt = how far the body is from vertical: torso vector when shoulders+hips are seen, else the principal axis of
    # whatever keypoints are visible (partial view: legs only, head and shoulders only, ...)
    f["tilt"] = f["theta"] if f["theta"] == f["theta"] else f["axis"]
    return f


def feature_table(poses: pd.DataFrame) -> pd.DataFrame:
    kps = kp_array(poses)
    rows = [features_row(kps[i], (r.x1, r.y1, r.x2, r.y2)) for i, r in enumerate(poses.itertuples(index=False))]
    ft = pd.DataFrame(rows, index=poses.index)
    for c in ("f", "t", "tid", "conf", "x1", "y1", "x2", "y2"):
        ft[c] = poses[c].to_numpy()
    return ft


def raw_posture(f: dict[str, float] | pd.Series, was_down: bool) -> str:
    """One-frame posture with a Schmitt trigger on the torso angle (enter at 60 deg, leave below 42 deg)."""
    th, ar, ax = f["theta"], f["aspect"], f["axis"]
    enter = DOWN_EXIT_DEG if was_down else DOWN_ENTER_DEG
    min_ar = 0.6 if was_down else 0.8
    if th == th:
        if (th >= enter and ar >= min_ar) or th >= 78:
            return DOWN
    else:   # torso keypoints hidden (close-up, legs/torso cut by the frame): box shape + body axis, never trusted for speed
        wide = 1.15 if was_down else 1.45
        if ar >= wide and f["nface"] >= 2:   # a wide box with a visible head; arms reaching sideways have no head
            return DOWN
    # not down: stand or sit?
    tv, lv = f["thigh_v"], f["leg_v"]
    if th == th:
        if lv == lv and lv >= 1.45 or tv == tv and tv >= 0.7:
            return UP
        if tv == tv and tv < 0.45 or lv == lv and lv < 1.2:
            return SIT
        if tv == tv or lv == lv:
            return UP if th < 25 else SIT
        # upper body only (legs hidden by a desk or the frame): cannot be a stand we can prove -> SITTING
        return SIT if th < 55 else UNK
    if ax == ax and f["nvis"] >= 4 and ax < UP_DEG and ar < 1.1:
        return PART
    return UNK


# --------------------------------------------------------------------------------------------- smoothing
def smooth_states(t: np.ndarray, raw: list[str], min_dwell: dict[str, float] = MIN_DWELL) -> list[str]:
    """Hysteresis part 2: runs shorter than their min dwell are absorbed by the neighbour that keeps the longer run."""
    st = list(raw)
    n = len(st)
    if n == 0:
        return st
    for _ in range(4):
        runs = []
        i = 0
        while i < n:
            j = i
            while j + 1 < n and st[j + 1] == st[i]:
                j += 1
            runs.append([i, j, st[i]])
            i = j + 1
        changed = False
        for k, (i, j, s) in enumerate(runs):
            dur = (t[j] - t[i]) + (t[1] - t[0] if n > 1 else 0.07)
            if dur >= min_dwell.get(s, 0.4) or len(runs) == 1:
                continue
            prev_s = runs[k - 1][2] if k > 0 else None
            next_s = runs[k + 1][2] if k + 1 < len(runs) else None
            if prev_s is None and next_s is None:
                continue
            if prev_s is None:
                new = next_s
            elif next_s is None:
                new = prev_s
            elif prev_s == next_s:
                new = prev_s
            else:
                pl = runs[k - 1][1] - runs[k - 1][0]
                nl = runs[k + 1][1] - runs[k + 1][0]
                if s != DOWN and DOWN in (prev_s, next_s):
                    # a short non-DOWN run next to DOWN must not be eaten by DOWN: that would move the onset earlier
                    new = next_s if prev_s == DOWN else prev_s
                else:   # absorb into the longer neighbour
                    new = prev_s if pl >= nl else next_s
            for q in range(i, j + 1):
                st[q] = new
            changed = True
        if not changed:
            break
    return st


# ------------------------------------------------------------------------------------------ track stitching
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def stitch_tracks(ft: pd.DataFrame, max_gap: float = 1.5, reach: float = 1.6) -> dict[int, int]:
    """tid -> person id. (1) A track that starts shortly after another one ended, close to where it ended, is the same
    person (ByteTrack drops IDs exactly when someone falls or is occluded). (2) Two tracks alive at the same time whose
    boxes mostly coincide are duplicate detections of one person."""
    d0 = ft[ft["tid"] >= 0].sort_values("t")
    g = d0.groupby("tid")
    info = {}
    for tid, d in g:
        a, b = d.iloc[0], d.iloc[-1]
        info[tid] = dict(first=a.t, last=b.t, sx=(a.x1 + a.x2) / 2, sy=(a.y1 + a.y2) / 2, ex=(b.x1 + b.x2) / 2,
                         ey=(b.y1 + b.y2) / 2, ew=b.x2 - b.x1, eh=b.y2 - b.y1, sw=a.x2 - a.x1, sh=a.y2 - a.y1, n=len(d))
    parent = {tid: tid for tid in info}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    order = sorted(info, key=lambda k: info[k]["first"])
    used_end: set[int] = set()
    for b in order:
        best, bd = None, 1e9
        for a in order:
            if a == b or a in used_end or info[a]["last"] >= info[b]["first"] + 0.01:
                continue
            gap = info[b]["first"] - info[a]["last"]
            if gap > max_gap:
                continue
            dist = math.hypot(info[b]["sx"] - info[a]["ex"], info[b]["sy"] - info[a]["ey"])
            size = max(info[a]["ew"], info[a]["eh"], info[b]["sw"], info[b]["sh"])
            if dist <= reach * size and dist < bd:
                best, bd = a, dist
        if best is not None:
            parent[find(b)] = find(best)
            used_end.add(best)
    # duplicates: overlapping lifetimes with coinciding boxes
    boxes = {tid: d.set_index("f")[["x1", "y1", "x2", "y2"]] for tid, d in g}
    tids = list(info)
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            if info[a]["last"] < info[b]["first"] or info[b]["last"] < info[a]["first"]:
                continue
            common = boxes[a].index.intersection(boxes[b].index)
            if len(common) < 3:
                continue
            ious = [_iou(boxes[a].loc[f].to_numpy(float), boxes[b].loc[f].to_numpy(float)) for f in common]
            if float(np.median(ious)) > 0.55:
                parent[find(b)] = find(a)
    return {tid: find(tid) for tid in info}


# --------------------------------------------------------------------------------------------- the analysis
@dataclass
class PersonResult:
    pid: int
    rows: pd.DataFrame                  # per-frame: t, f, theta, aspect, raw, state, box
    events: list[dict[str, Any]] = field(default_factory=list)


def analyse(poses: pd.DataFrame, min_frames: int = 4) -> tuple[list[PersonResult], pd.DataFrame]:
    """poses -> (per person results, feature table with posture columns)."""
    ft = feature_table(poses)
    ft["person"] = ft["tid"].map(stitch_tracks(ft)).fillna(-1).astype(int)
    ft["raw"], ft["state"] = UNK, UNK
    results: list[PersonResult] = []
    for pid, d in ft[ft["person"] >= 0].sort_values("t").groupby("person"):
        if len(d) < min_frames:
            continue
        # per-frame the best detection for this person (a stitched person can have 2 tids in one frame)
        d = d.sort_values(["t", "conf"], ascending=[True, False]).drop_duplicates("t")
        was, raw = False, []
        for _, r in d.iterrows():
            s = raw_posture(r, was)
            was = s == DOWN
            raw.append(s)
        st = smooth_states(d["t"].to_numpy(), raw)
        ft.loc[d.index, "raw"] = raw
        ft.loc[d.index, "state"] = st
        rows = ft.loc[d.index].copy()
        results.append(PersonResult(pid, rows, find_events(rows)))
    return results, ft


def find_events(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Posture transitions of one person. DOWN onsets get last_upright / interval / speed; UP->SIT is a 'sat' event."""
    rows = rows.sort_values("t")
    t = rows["t"].to_numpy()
    st = rows["state"].to_numpy()
    th = rows["tilt"].to_numpy()          # torso angle, or body-axis angle when the torso is not visible
    src = np.where(rows["theta"].to_numpy() == rows["theta"].to_numpy(), "torso", "axis")
    ev: list[dict[str, Any]] = []
    if len(rows) == 0:
        return ev
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.07
    # runs of DOWN
    i, n = 0, len(rows)
    while i < n:
        if st[i] != DOWN:
            i += 1
            continue
        j = i
        while j + 1 < n and st[j + 1] == DOWN and t[j + 1] - t[j] <= GAP_S * 2:
            j += 1
        k = i
        first_down = float(t[k])
        # last_upright = the last frame BEFORE the DOWN run in the smoothed UPRIGHT state, or in PARTIAL (body axis near
        # vertical but only legs / head+shoulders in view: the person is still on their feet). SITTING never counts: a
        # sprawled person with the torso near vertical is on the ground already.
        lu = None
        for q in range(k - 1, -1, -1):
            if st[q] in (UP, PART) and th[q] == th[q] and th[q] < UP_DEG_REF:
                lu = q
                break
        onset_torso = bool(np.any(rows["theta"].to_numpy()[k:min(k + 4, j + 1)] == rows["theta"].to_numpy()[k:min(k + 4, j + 1)]))
        e: dict[str, Any] = {"kind": "down", "first_down": first_down, "down_end": float(t[j]) + dt, "onset_torso": onset_torso,
                             "end_state": "lost" if j == n - 1 else str(st[j + 1]),
                             "down_s": float(t[j]) + dt - first_down}
        if lu is None:
            e.update(last_upright=None, interval=None, speed="NOT_OBSERVED", transition="already down when first seen",
                     pre_state=str(st[k - 1]) if k > 0 else None, contact_t=first_down, flat_t=first_down)
        else:
            kf = k   # the transition clock stops when the torso is flat (>= FLAT_DEG), not merely past the DOWN threshold
            while kf < j and not (th[kf] == th[kf] and th[kf] >= FLAT_DEG):
                kf += 1
            if not (th[kf] == th[kf] and th[kf] >= FLAT_DEG):
                kf = k
            flat_t = float(t[kf])
            # first ground contact: first frame after the last upright where the hips are at foot level (ankles within half
            # a torso length of the hips: sitting on the ground / lying) or the posture is DOWN. Never later than flat.
            lv = rows["leg_v"].to_numpy() if "leg_v" in rows else np.full(n, np.nan)
            kc = kf
            for q in range(lu + 1, kf + 1):
                if st[q] == DOWN or (lv[q] == lv[q] and lv[q] < 0.5):
                    kc = q
                    break
            iv = flat_t - float(t[lu])
            seen = [x for x in range(lu, kf + 1) if th[x] == th[x]]     # samples where the body tilt was measurable
            gap = float(np.max(np.diff(t[seen]))) if len(seen) > 1 else iv
            e.update(last_upright=float(t[lu]), flat_t=flat_t, contact_t=float(t[kc]), interval=iv,
                     to_ground=float(t[kc] - t[lu]), max_gap=gap, pre_state=str(st[lu]), evidence=str(src[lu]),
                     upright_view="legs only" if st[lu] == PART else "full body")
            # a legs-only upright end is accepted as evidence only when it is a clean near-vertical axis with no hole before
            # the flat frame (checked through max_gap below); a torso-less END is never accepted (onset_torso)
            if not onset_torso:
                e["speed"] = "PARTIAL"    # a transition whose end we only saw through partial keypoints is never called a fall
            elif iv <= FAST_S:
                e["speed"] = "FAST"
            elif gap > FAST_S:
                e["speed"] = "UNOBSERVED"
            elif iv <= SLOW_S:
                e["speed"] = "MEDIUM"
            else:
                e["speed"] = "SLOW"
        ev.append(e)
        i = j + 1
    # sat events: stand (or partial upright view) -> sit. The speed is only reported when the knees were seen at both ends.
    tv = rows["thigh_v"].to_numpy()
    prev, pt, pq = None, None, None
    for q in range(n):
        s = st[q]
        if s in (UP, SIT, PART):
            if prev in (UP, PART) and s == SIT:
                e2 = q
                while e2 + 1 < n and st[e2 + 1] == SIT:
                    e2 += 1
                legs = tv[pq] == tv[pq] and tv[q] == tv[q]
                ev.append({"kind": "sat", "t": float(t[q]), "from_t": float(pt), "interval": float(t[q] - pt),
                           "sit_s": float(t[e2] - t[q] + dt),
                           "speed": ("FAST" if t[q] - pt <= FAST_S else "SLOW") if legs else "n/a (legs not seen)"})
            prev, pt, pq = s, t[q], q
        elif s == DOWN:
            prev = None
    return ev


# ------------------------------------------------------------------------------------------------ verdicts
def verdict(results: list[PersonResult], duration: float) -> dict[str, Any]:
    """Clip-level verdict from the worst event of any person. Classes: FALL > LIE > NO_ALARM."""
    downs = [(r.pid, e) for r in results for e in r.events if e["kind"] == "down" and e["down_s"] >= 0.5]
    sats = [(r.pid, e) for r in results for e in r.events if e["kind"] == "sat" and e["sit_s"] >= 1.0]
    falls = [(p, e) for p, e in downs if e["speed"] == "FAST"]
    if falls:
        p, e = max(falls, key=lambda x: x[1]["down_s"])
        return {"label": "FALL", "pid": p, "event": e}
    if downs:
        p, e = max(downs, key=lambda x: x[1]["down_s"])
        e = dict(e)
        # other people-tracks lying down at the same time without a seen transition are the same lying scene (a tracker
        # ID switch in a close-up splits one resident in two): report the earliest start and the latest end of the group
        if e.get("last_upright") is None:
            grp = [(q, x) for q, x in downs if x.get("last_upright") is None and x["first_down"] <= e["down_end"] and x["down_end"] >= e["first_down"]]
            e["first_down"] = min(x["first_down"] for _, x in grp)
            e["contact_t"] = e["flat_t"] = e["first_down"]
            e["down_end"] = max(x["down_end"] for _, x in grp)
            e["down_s"] = e["down_end"] - e["first_down"]
            e["merged_pids"] = sorted({q for q, _ in grp})
        return {"label": "LIE", "pid": p, "event": e}
    if sats:
        p, e = max(sats, key=lambda x: x[1]["interval"])
        return {"label": "NO_ALARM", "detail": "SAT" + (" (slow)" if e["speed"] == "SLOW" else ""), "pid": p, "event": e}
    return {"label": "NO_ALARM", "detail": "no posture event", "pid": None, "event": None}


def clock_state(elapsed_s: float, ladder=LADDER) -> dict[str, Any]:
    """Long-lie clock: the rung reached after `elapsed_s` seconds on the floor."""
    rung = ladder[0]
    for r in ladder:
        if elapsed_s >= r[0]:
            rung = r
    nxt = next((r for r in ladder if r[0] > elapsed_s), None)
    return {"elapsed_s": elapsed_s, "rung": rung[1], "level": rung[2],
            "next_in_s": (nxt[0] - elapsed_s) if nxt else None, "next_rung": nxt[1] if nxt else None}


def fmt_dur(s: float) -> str:
    s = max(0.0, s)
    if s < 60:
        return f"{s:.1f} s"
    if s < 3600:
        return f"{int(s // 60)} min {int(s % 60):02d} s"
    return f"{int(s // 3600)} h {int((s % 3600) // 60):02d} min"


def short_dur(s: float) -> str:
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s // 60)}m"
    return f"{int(s // 3600)}h"
