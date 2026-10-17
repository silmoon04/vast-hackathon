"""ON THE LINE scene-state engine.

Consumes a *live feed* (frame, tracker rows for that frame only) and keeps, up to "now", everything an RVRC
operator is asked on the phone: who is on scene, who arrived when, where each one went, what they look like.
The engine never sees the full track table: LiveFeed hands it one sampled frame at a time (ByteTrack is an online
tracker, so replaying the cached tracks is identical to running it live).

Stand-ins on the day: tracker rows come from the DataEngine YOLO11s stream (+ a tracker), the "description of the
best crop" from Cosmos3-Reason; colours come from pixels (no model).
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import protokit as pk  # noqa: E402

GROUP = {"person": "person", "car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
         "bicycle": "bicycle"}
GROUP_WORD = {"person": "person", "vehicle": "vehicle", "bicycle": "bicycle"}
GROUP_PLURAL = {"person": "people", "vehicle": "vehicles", "bicycle": "bicycles"}

# ------------------------------------------------------------------------------------------------ colour naming


def colour_name(bgr: tuple[float, float, float]) -> str:
    """Plain-English colour of one BGR value (HSV rules, tuned for CCTV)."""
    px = np.uint8([[[int(np.clip(c, 0, 255)) for c in bgr]]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    hue, s, v = float(h) * 2.0, s / 255.0, v / 255.0
    if v < 0.16:
        return "black"
    if s < 0.13:
        if v > 0.80:
            return "white"
        if v > 0.58:
            return "silver"
        if v > 0.34:
            return "grey"
        return "charcoal"
    if s < 0.30 and 20 <= hue < 70 and v > 0.55:
        return "cream"
    if hue < 12 or hue >= 345:
        return "red" if v > 0.35 else "maroon"
    if hue < 38:
        return "orange" if v > 0.55 else "brown"
    if hue < 70:
        return "yellow" if v > 0.55 else "olive"
    if hue < 165:
        return "green"
    if hue < 200:
        return "teal"
    if hue < 268:      # CCTV sensors push dark blue towards violet, so "blue" runs up to 268 deg
        return "blue" if v > 0.42 else "navy"
    if hue < 295:
        return "purple"
    return "pink" if v > 0.6 else "maroon"


CLOTHING_DISPLAY = {"silver": "light grey", "charcoal": "dark grey", "navy": "dark blue"}


def display_colour(group: str, name: str | None) -> str | None:
    """Wording for the card: clothing is never 'silver' (a near-grey sweater is 'light grey'); vehicles keep paint names."""
    if name is None:
        return None
    return CLOTHING_DISPLAY.get(name, name) if group == "person" else name


def _region(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = frame.shape[:2]
    a, b, c, d = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    if c - a < 2 or d - b < 2:
        return np.empty((0, 3), np.uint8)
    return frame[b:d, a:c].reshape(-1, 3)


def _med(px: np.ndarray, drop_dark: float) -> tuple[float, float, float]:
    if drop_dark > 0 and len(px) >= 60:
        v = px.max(axis=1)
        px = px[v >= np.quantile(v, drop_dark)]   # windows, tyres and shadows would drag a car towards black
    return tuple(float(v) for v in np.median(px, axis=0))


def fg_colour(frame: np.ndarray, box: tuple[float, float, float, float], rx: tuple[float, float],
              ry: tuple[float, float], drop_dark: float = 0.0) -> tuple[float, float, float] | None:
    """Median BGR of the foreground pixels inside a sub-region of `box` (foreground = differs from the ring just
    outside the box). Falls back to the plain region median when the region is the same colour as its surroundings."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    reg = _region(frame, x1 + rx[0] * w, y1 + ry[0] * h, x1 + rx[1] * w, y1 + ry[1] * h)
    if len(reg) < 12:
        return None
    pad = 0.25
    ring = np.concatenate([_region(frame, x1 - pad * w, y1 - pad * h, x2 + pad * w, y1),
                           _region(frame, x1 - pad * w, y2, x2 + pad * w, y2 + pad * h),
                           _region(frame, x1 - pad * w, y1, x1, y2), _region(frame, x2, y1, x2 + pad * w, y2)])
    if len(ring) < 12:
        return _med(reg, drop_dark)
    ring_m = np.median(ring.astype(np.float32), axis=0)
    lab_r = cv2.cvtColor(reg.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lab_ring = cv2.cvtColor(np.uint8([[ring_m]]), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
    dist = np.linalg.norm(lab_r - lab_ring, axis=1)
    keep = dist > 16
    if keep.sum() >= max(15, 0.2 * len(reg)):
        return _med(reg[keep], drop_dark)
    return _med(reg, drop_dark)


def person_colours(frame: np.ndarray, box) -> dict[str, tuple[float, float, float]]:
    out = {}
    c = fg_colour(frame, box, (0.22, 0.78), (0.16, 0.52))
    if c:
        out["top"] = c
    c = fg_colour(frame, box, (0.28, 0.72), (0.58, 0.92))
    if c:
        out["bottom"] = c
    return out


def vehicle_colours(frame: np.ndarray, box) -> dict[str, tuple[float, float, float]]:
    c = fg_colour(frame, box, (0.2, 0.8), (0.25, 0.75), drop_dark=0.3)
    return {"body": c} if c else {}


# ------------------------------------------------------------------------------------------------------ live feed


class LiveFeed:
    """Replays a video + its cached tracker rows as a live stream: yields (t, frame_idx, frame, rows_of_this_frame)."""

    def __init__(self, name: str, tr: pk.Tracks, start: float, end: float):
        self.name, self.tr, self.start, self.end = name, tr, start, end
        self._by_f = {int(f): g for f, g in tr.rows.groupby("f")}

    def __iter__(self) -> Iterator[tuple[float, int, np.ndarray, list[dict[str, Any]]]]:
        empty = self.tr.rows.iloc[0:0]
        for i, t, frame in pk.iter_frames(self.name, stride=self.tr.stride, start=self.start, end=self.end):
            g = self._by_f.get(i, empty)
            rows = [dict(tid=int(r.tid), name=r.name, conf=float(r.conf), box=(float(r.x1), float(r.y1), float(r.x2), float(r.y2)))
                    for r in g.itertuples()]
            yield t, i, frame, rows


# ------------------------------------------------------------------------------------------------ data classes


@dataclass
class Raw:
    tid: int
    group: str
    first_t: float
    pts: list = field(default_factory=list)      # (t, x1, y1, x2, y2, conf)
    cls: Counter = field(default_factory=Counter)
    eid: int | None = None
    max_conf: float = 0.0
    last_t: float = 0.0


@dataclass
class Entity:
    eid: int
    first_t: float
    pts: list = field(default_factory=list)      # (t, x1, y1, x2, y2, conf)
    cls: Counter = field(default_factory=Counter)
    raw_tids: list = field(default_factory=list)
    colour_votes: dict = field(default_factory=lambda: {"top": [], "bottom": [], "body": []})
    best: dict | None = None                      # {score, t, box, img}
    last_img: np.ndarray | None = None
    last_img_t: float = 0.0
    desc: dict | None = None                      # {text, req_t, avail_t, seconds}
    desc_requested: bool = False
    carried_votes: int = 0
    carried: bool = False
    at_alarm: bool = False
    confirmed_t: float = 0.0
    first_box_edge: str | None = None

    @property
    def group(self) -> str:
        return GROUP[self.cls_name]

    @property
    def last_t(self) -> float:
        return self.pts[-1][0]

    @property
    def cls_name(self) -> str:
        return self.cls.most_common(1)[0][0]

    @property
    def label(self) -> str:
        return f"{GROUP_WORD[self.group].upper()} #{self.eid}"


@dataclass
class Event:
    t: float
    kind: str       # arrived | departed | on_scene_at_alarm | stopped
    eid: int
    text: str


# ------------------------------------------------------------------------------------------------------ engine

EDGE_NAMES = {"left": "LEFT", "right": "RIGHT", "top": "TOP", "bottom": "BOTTOM"}


class SceneState:
    """Scene state of one camera, advanced one sampled frame at a time with feed()."""

    def __init__(self, w: int, h: int, fps: float, alarm_t: float, *, vlm_budget: int = 28, use_vlm: bool = True,
                 hold: float = 0.8, confirm_n: int = 4, confirm_conf: float = 0.40):
        self.w, self.h, self.fps, self.alarm_t = w, h, fps, alarm_t
        self.hold, self.confirm_n, self.confirm_conf = hold, confirm_n, confirm_conf
        self.now = alarm_t
        self.raw: dict[int, Raw] = {}
        self.entities: dict[int, Entity] = {}
        self.events: list[Event] = []
        self.snaps: list[tuple[float, np.ndarray]] = []   # evidence reel: (t, small frame)
        self._next_snap = alarm_t
        self._eid = 0
        self.vlm_budget, self.use_vlm = vlm_budget, use_vlm
        self.vlm_calls = 0
        self.vlm_log: list[dict[str, Any]] = []
        self.frame_now: np.ndarray | None = None
        self.rows_now: list[dict[str, Any]] = []
        self.truck_hist: dict[float, list] = {}
        self.count_log: list[tuple[float, dict[str, int]]] = []
        self.n_frames = 0
        self.first_frame_t: float | None = None

    # ---------------------------------------------------------------------------------------- geometry helpers
    def edge_of(self, box, margin: float = 0.045) -> str | None:
        x1, y1, x2, y2 = box
        d = {"left": x1 / self.w, "right": (self.w - x2) / self.w, "top": y1 / self.h, "bottom": (self.h - y2) / self.h}
        k = min(d, key=d.get)
        return k if d[k] <= margin else None

    # --------------------------------------------------------------------------------------------- ingestion
    def feed(self, t: float, frame: np.ndarray, rows: list[dict[str, Any]]) -> None:
        self.now = t
        self.n_frames += 1
        if self.first_frame_t is None:
            self.first_frame_t = t
        self.frame_now, self.rows_now = frame, rows
        if t >= self._next_snap - 1e-6:
            self.snaps.append((t, pk.fit(frame, 420)))
            self._next_snap = t + 0.5
        seen_eids: list[tuple[Entity, tuple, float]] = []
        for r in rows:
            g = GROUP.get(r["name"])
            if g is None or r["tid"] < 0:
                continue
            rw = self.raw.get(r["tid"])
            if rw is None:
                rw = self.raw[r["tid"]] = Raw(r["tid"], g, t)
            rw.pts.append((t, *r["box"], r["conf"]))
            rw.cls[r["name"]] += 1
            rw.max_conf = max(rw.max_conf, r["conf"])
            rw.last_t = t
            if rw.eid is None and len(rw.pts) >= self.confirm_n and rw.max_conf >= self.confirm_conf:
                self._confirm(rw, t)
            if rw.eid is not None:
                e = self.entities[rw.eid]
                if len(rw.pts) == 1 or e.pts[-1][0] < t:
                    e.pts.append((t, *r["box"], r["conf"]))
                e.cls[r["name"]] += 1
                seen_eids.append((e, r["box"], r["conf"]))
        for e, box, conf in seen_eids:
            self._observe(e, box, conf, frame, rows)
        self._carried_check(rows)
        self._maybe_describe()
        cnt = Counter(e.group for e in self.entities.values() if not e.carried and t - e.last_t <= self.hold)
        self.count_log.append((t, {g: cnt.get(g, 0) for g in GROUP_WORD}))

    def _confirm(self, rw: Raw, t: float) -> None:
        """Raw track just became real: attach it to an entity that vanished a moment ago, or open a new entity."""
        best: tuple[float, Entity] | None = None
        rw.group = GROUP[rw.cls.most_common(1)[0][0]]
        gap_max = {"person": 2.5, "vehicle": 1.5, "bicycle": 2.0}[rw.group]
        first = rw.pts[0]
        fcx, fcy = (first[1] + first[3]) / 2, (first[2] + first[4]) / 2
        for e in self.entities.values():
            if e.group != rw.group or e.carried:
                continue
            owner_active = any(self.raw[x].last_t > rw.first_t + 1e-6 and x != rw.tid for x in e.raw_tids)
            if owner_active:
                continue
            dt = rw.first_t - e.last_t
            if dt < -1e-6 or dt > gap_max:
                continue
            lt = e.pts[-1]
            if self.edge_of(e.pts[-1][1:5]) is not None:
                # Left through an edge: a new arrival is normally a new object. Exception (found on the car_detection
                # held-out clip): a half-visible object at the edge gets a fresh tracker id within ~0.6 s on an
                # overlapping box, and that must not count twice.
                ix0 = max(0.0, min(lt[3], first[3]) - max(lt[1], first[1]))
                iy0 = max(0.0, min(lt[4], first[4]) - max(lt[2], first[2]))
                small0 = max(1.0, min((lt[3] - lt[1]) * (lt[4] - lt[2]), (first[3] - first[1]) * (first[4] - first[2])))
                if not (dt <= 0.6 and ix0 * iy0 / small0 >= 0.4):
                    continue
            lcx, lcy = (lt[1] + lt[3]) / 2, (lt[2] + lt[4]) / 2
            vx, vy = self._velocity(e)
            px, py = lcx + vx * dt, lcy + vy * dt
            size = max(lt[3] - lt[1], lt[4] - lt[2], first[3] - first[1], first[4] - first[2])
            dist = float(np.hypot(px - fcx, py - fcy))
            norm = dist / (0.8 * size + 0.3 * float(np.hypot(vx, vy)) * dt + 15)
            ix = max(0.0, min(lt[3], first[3]) - max(lt[1], first[1]))
            iy = max(0.0, min(lt[4], first[4]) - max(lt[2], first[2]))
            small = max(1.0, min((lt[3] - lt[1]) * (lt[4] - lt[2]), (first[3] - first[1]) * (first[4] - first[2])))
            if ix * iy / small >= 0.4:      # boxes overlap: same object, tracker just re-issued the id
                norm = min(norm, 0.5)
            if norm <= 1.0 and (best is None or norm < best[0]):
                best = (norm, e)
        if best:
            e = best[1]
            e.raw_tids.append(rw.tid)
            rw.eid = e.eid
            for p in rw.pts:
                if p[0] > e.last_t:
                    e.pts.append(p)
            for c, n in rw.cls.items():
                e.cls[c] += n
            return
        self._eid += 1
        e = Entity(self._eid, rw.first_t, raw_tids=[rw.tid], confirmed_t=t)
        e.pts = list(rw.pts)
        e.cls = Counter(rw.cls)
        e.at_alarm = self.first_frame_t is not None and rw.first_t <= self.first_frame_t + 1.5 / self.fps
        e.first_box_edge = self.edge_of(e.pts[0][1:5])
        self.entities[e.eid] = e
        rw.eid = e.eid
        self.events.append(Event(rw.first_t, "on_scene_at_alarm" if e.at_alarm else "arrived", e.eid, ""))

    def _velocity(self, e: Entity, span: float = 0.8) -> tuple[float, float]:
        pts = [p for p in e.pts if p[0] >= e.last_t - span]
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 1e-6:
            return 0.0, 0.0
        a, b = pts[0], pts[-1]
        dt = b[0] - a[0]
        return ((b[1] + b[3]) / 2 - (a[1] + a[3]) / 2) / dt, ((b[2] + b[4]) / 2 - (a[2] + a[4]) / 2) / dt

    def _observe(self, e: Entity, box, conf: float, frame: np.ndarray, rows) -> None:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        edge = self.edge_of(box, 0.012)
        occl = 0.0
        for r in rows:
            if r["box"] == box:
                continue
            ix = max(0.0, min(x2, r["box"][2]) - max(x1, r["box"][0]))
            iy = max(0.0, min(y2, r["box"][3]) - max(y1, r["box"][1]))
            occl = max(occl, ix * iy / max(1.0, bw * bh))
        score = conf * np.sqrt(max(bw * bh, 1.0)) * (0.55 if edge else 1.0) * (0.55 if occl > 0.25 else 1.0)
        pad = 0.12
        cx1, cy1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
        cx2, cy2 = int(min(self.w, x2 + pad * bw)), int(min(self.h, y2 + pad * bh))
        if cx2 - cx1 >= 6 and cy2 - cy1 >= 6:
            crop = frame[cy1:cy2, cx1:cx2]
            e.last_img, e.last_img_t = crop.copy(), self.now
            if e.best is None or score > e.best["score"]:
                e.best = {"score": float(score), "t": self.now, "box": tuple(box), "img": crop.copy(), "edge": edge,
                          "occl": occl}
        if conf >= 0.4 and not edge and occl < 0.3 and self.n_frames % 2 == 0:
            cols = person_colours(frame, box) if e.group == "person" else (
                vehicle_colours(frame, box) if e.group == "vehicle" else {})
            for k, c in cols.items():
                v = e.colour_votes[k]
                v.append(c)
                if len(v) > 40:
                    v.pop(0)

    def _carried_check(self, rows) -> None:
        """A car whose box sits inside a truck's box for most recent frames is cargo (recovery truck), not a vehicle.
        Truck boxes are kept for the last few seconds (any detection, confirmed or not) so a car that only just
        became confirmed can be checked retroactively."""
        trucks = [r["box"] for r in rows if r["name"] == "truck"]
        if trucks:
            self.truck_hist[self.now] = trucks
        for k in [k for k in self.truck_hist if k < self.now - 8.0]:
            del self.truck_hist[k]
        if not self.truck_hist:
            return
        for e in self.entities.values():
            if e.carried or e.group != "vehicle" or e.cls_name == "truck" or self.now - e.last_t > 0.2:
                continue
            recent = [p for p in e.pts if p[0] >= self.now - 1.5]
            inside = 0
            for p in recent:
                for tb in self.truck_hist.get(p[0], []):
                    ix = max(0.0, min(p[3], tb[2]) - max(p[1], tb[0]))
                    iy = max(0.0, min(p[4], tb[3]) - max(p[2], tb[1]))
                    if ix * iy / max(1.0, (p[3] - p[1]) * (p[4] - p[2])) >= 0.65:
                        inside += 1
                        break
            if inside >= 4 and inside >= 0.6 * len(recent):
                e.carried = True

    # ------------------------------------------------------------------------------------------------- VLM
    describe_cb: Callable[[Entity], dict[str, Any] | None] | None = None

    def _maybe_describe(self) -> None:
        if not (self.use_vlm and self.describe_cb) or self.vlm_calls >= self.vlm_budget:
            return
        for e in self.entities.values():
            if e.desc_requested or e.carried or e.best is None or e.group == "bicycle":
                continue
            age = e.last_t - e.first_t
            gone = self.now - e.last_t > self.hold
            bh = e.best["box"][3] - e.best["box"][1]
            ok_size = bh >= (60 if e.group == "person" else 40)
            if (age >= 1.2 and not gone and ok_size and e.best["score"] > 0) or (gone and age >= 0.5 and ok_size):
                e.desc_requested = True
                self.vlm_calls += 1
                res = self.describe_cb(e)
                if res and res.get("text"):
                    e.desc = {"text": res["text"], "req_t": self.now, "seconds": res["seconds"],
                              "avail_t": self.now + res["seconds"]}
                self.vlm_log.append({"eid": e.eid, "t": self.now, "seconds": (res or {}).get("seconds"),
                                     "measured_seconds": (res or {}).get("measured_seconds"),
                                     "text": (res or {}).get("text", "")})
                if self.vlm_calls >= self.vlm_budget:
                    return

    # --------------------------------------------------------------------------------------------- queries
    def confirmed(self, group: str | None = None, include_carried: bool = False) -> list[Entity]:
        return [e for e in self.entities.values() if (group is None or e.group == group)
                and (include_carried or not e.carried)]

    def present(self, group: str | None = None) -> list[Entity]:
        return [e for e in self.confirmed(group) if self.now - e.last_t <= self.hold]

    def colours(self, e: Entity) -> dict[str, str]:
        out = {}
        for k, v in e.colour_votes.items():
            if v:
                out[k] = colour_name(tuple(np.median(np.array(v), axis=0)))
                if k == "body" and out[k] == "pink":
                    out[k] = "red"      # glare desaturates red paint to pink; pink vehicles are rare (found on car_detection)
        return out

    def heading(self, e: Entity, span: float = 1.2) -> dict[str, Any]:
        """Direction of travel from the last `span` s of the path (image frame)."""
        pts = [p for p in e.pts if p[0] >= e.last_t - span] or e.pts[-2:]
        if len(pts) < 2:
            return {"label": "unknown", "speed": 0.0, "dx": 0.0, "dy": 0.0}
        a, b = pts[0], pts[-1]
        dt = max(b[0] - a[0], 1e-6)
        dx = ((b[1] + b[3]) / 2 - (a[1] + a[3]) / 2)
        dy = ((b[2] + b[4]) / 2 - (a[2] + a[4]) / 2)
        size = max(b[3] - b[1], b[4] - b[2])
        speed = float(np.hypot(dx, dy) / dt)
        if np.hypot(dx, dy) < 0.35 * size or dt < 0.3:
            return {"label": "stationary", "speed": speed, "dx": dx, "dy": dy}
        horiz = "right" if dx > 0 else "left"
        vert = "down" if dy > 0 else "up"
        if abs(dx) >= 2.0 * abs(dy):
            lab = horiz
        elif abs(dy) >= 2.0 * abs(dx):
            lab = vert
        else:
            lab = f"{horiz}+{vert}"
        return {"label": lab, "speed": speed, "dx": dx, "dy": dy}

    def exit_side(self, e: Entity) -> str | None:
        """Frame edge the entity left through, from its last box (or its heading when it vanished near an edge)."""
        edge = self.edge_of(e.pts[-1][1:5], 0.06)
        if edge:
            return edge
        hd = self.heading(e)
        if hd["label"] != "stationary":
            cx, cy = (e.pts[-1][1] + e.pts[-1][3]) / 2, (e.pts[-1][2] + e.pts[-1][4]) / 2
            # extrapolate one second along the heading; if that point is out of frame, that is the exit side
            vx, vy = self._velocity(e, 1.0)
            px, py = cx + vx * 0.8, cy + vy * 0.8
            if px < 0:
                return "left"
            if px > self.w:
                return "right"
            if py < 0:
                return "top"
            if py > self.h:
                return "bottom"
        return None

    def facts(self, e: Entity) -> dict[str, Any]:
        gone = self.now - e.last_t > self.hold
        hd = self.heading(e)
        cols = self.colours(e)
        desc = e.desc["text"] if (e.desc and e.desc["avail_t"] <= self.now) else None
        stat_since = None
        if not gone:
            st = [p for p in e.pts if p[0] >= self.now - 2.5]
            if len(st) >= 3 and (e.pts[-1][0] - e.pts[0][0]) >= 2.0:
                size = max(st[-1][3] - st[-1][1], st[-1][4] - st[-1][2])
                d = np.hypot((st[-1][1] + st[-1][3]) / 2 - (st[0][1] + st[0][3]) / 2,
                             (st[-1][2] + st[-1][4]) / 2 - (st[0][2] + st[0][4]) / 2)
                if d < 0.35 * size and st[-1][0] - st[0][0] >= 2.0:
                    stat_since = st[0][0]
        return {"eid": e.eid, "group": e.group, "cls": e.cls_name, "first_t": e.first_t, "last_t": e.last_t,
                "present": not gone, "last_seen_ago": float(self.now - e.last_t),
                "entry_side": e.first_box_edge, "exit_side": self.exit_side(e) if gone else None,
                "heading": hd["label"], "speed_px_s": round(hd["speed"], 1), "colours": cols, "desc": desc,
                "desc_pending": bool(e.desc and e.desc["avail_t"] > self.now),
                "at_alarm": e.at_alarm, "stationary_since": stat_since, "n_frames": len(e.pts),
                "dwell": float(e.last_t - e.first_t)}

    def snap_at(self, t: float) -> np.ndarray | None:
        if not self.snaps:
            return None
        ts = np.array([s[0] for s in self.snaps])
        return self.snaps[int(np.argmin(np.abs(ts - t)))][1]
