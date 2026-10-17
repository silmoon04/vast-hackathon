"""follow: join fragment tracks into visits, then visits into identities across cameras.

merge_fragments(tracks)   same camera: b continues a when it starts <= 4 s after a ends, near where a was heading,
                          with a small colour distance. One visit = one track; `fragments` lists the joined ids.
handoffs(tracks, cams)    cross-camera hops from the cams.follow.json topology. A track that leaves camera X by an
                          exit zone at t is matched to tracks on a linked camera entering within
                          [t + min_s - 2, t + max_s + 5], ranked by colour distance. Blind links (no camera between)
                          get wider windows and a lower score. Nothing below MIN_SCORE is linked: those stay separate.
identities(tracks, hops)  chains of hops -> [{id, tracks, hops}]; every person track is in exactly one identity.
item_owner(item, ...)     the identity whose box overlaps the item most while the item is first visible.

Zones are frame edges: east = right, west = left, north = top, south = bottom (boxes normalised 0..1).
Times are scene seconds once the builder has shifted each camera by its offset (shift()).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from stm.fastindex import hsv_distance, look_of

MERGE_GAP_S = 4.0       # same-camera fragments: max unseen gap
MERGE_OVERLAP_S = 0.6   # ...or overlap by one sample (tracker doubled an id)
MERGE_DIST = 0.8        # start of b within this many box heights of where a was heading (+0.35 per second unseen)
MERGE_HSV = 0.35        # max colour distance for a same-camera merge
MERGE_SIZE = 1.7        # max box-height ratio
ZONE_M = 0.08           # a box within this share of the frame from an edge touches that edge's zone
LOOK_S = 0.75           # zones are also checked this many seconds ahead (exit) / back (entry) along the motion
DIRECT_PAD = (2.0, 5.0)     # window padding below min_s / above max_s
BLIND_PAD = (4.0, 10.0)     # blind links: longer window
BLIND_FACTOR = 0.85     # blind links score lower
ZONE_MISS = 0.85        # entering by another zone than expected costs this factor
MIN_SCORE = 0.6         # never link below this
CLOSE_RUNNER = 0.1      # mention the runner-up when it is this close
EDGES = ("east", "west", "north", "south")


def _c(b: list[float]) -> np.ndarray:
    return np.array([b[1] + b[3] / 2, b[2] + b[4] / 2])


def _vel(boxes: list[list[float]], end: bool = True) -> np.ndarray:
    """Centre velocity (frame units / s) over the last (or first) three boxes."""
    seg = boxes[-3:] if end else boxes[:3]
    if len(seg) < 2 or seg[-1][0] - seg[0][0] <= 0:
        return np.zeros(2)
    return (_c(seg[-1]) - _c(seg[0])) / (seg[-1][0] - seg[0][0])


def zones(box: list[float] | np.ndarray, m: float = ZONE_M) -> set[str]:
    """Frame-edge zones a normalised [x, y, w, h] box touches."""
    x, y, w, h = [float(v) for v in box]
    z = set()
    if x + w >= 1 - m:
        z.add("east")
    if x <= m:
        z.add("west")
    if y <= m:
        z.add("north")
    if y + h >= 1 - m:
        z.add("south")
    return z


def exit_zones(tr: dict[str, Any]) -> set[str]:
    b = tr["boxes"][-1]
    ahead = np.array(b[1:]) + np.r_[_vel(tr["boxes"]) * LOOK_S, 0, 0]
    return zones(b[1:]) | zones(ahead)


def entry_zones(tr: dict[str, Any]) -> set[str]:
    b = tr["boxes"][0]
    back = np.array(b[1:]) - np.r_[_vel(tr["boxes"], end=False) * LOOK_S, 0, 0]
    return zones(b[1:]) | zones(back)


def shift(tr: dict[str, Any], dt: float) -> dict[str, Any]:
    """Copy of a track with every time moved by dt (video seconds -> scene seconds: dt = camera offset)."""
    out = dict(tr)
    out["boxes"] = [[round(b[0] + dt, 2), *b[1:]] for b in tr["boxes"]]
    out["t0"], out["t1"] = out["boxes"][0][0], out["boxes"][-1][0]
    if tr.get("crop_t") is not None:
        out["crop_t"] = round(tr["crop_t"] + dt, 2)
    return out


# ---------------------------------------------------------------- same camera
def _combine(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if len(parts) == 1:
        return parts[0]
    by_t: dict[float, list[float]] = {}
    for p in parts:
        for b in p["boxes"]:
            if b[0] not in by_t or b[3] * b[4] > by_t[b[0]][3] * by_t[b[0]][4]:
                by_t[b[0]] = b
    boxes = [by_t[t] for t in sorted(by_t)]
    n = np.array([p["n"] for p in parts], float)
    sig = {k: [round(float(v), 4) for v in np.average([p["sig"][k] for p in parts], axis=0, weights=n)]
           for k in ("upper", "lower")}
    look, desc = look_of(sig)
    best = max(parts, key=lambda p: p.get("_crop_score", -1))
    tall = float(np.average([p.get("tall", 1.0) for p in parts], weights=n))
    return {**parts[0], "kind": "person" if tall >= 0.5 else "object", "t0": boxes[0][0], "t1": boxes[-1][0],
            "n": len(boxes), "tall": round(tall, 2), "boxes": boxes, "sig": sig, "look": look, "desc": desc,
            "crop_t": best.get("crop_t"), "_crop": best.get("_crop"), "_crop_score": best.get("_crop_score", -1),
            "fragments": [f for p in parts for f in p.get("fragments", [p["id"]])]}


def _continues(a: dict[str, Any], b: dict[str, Any], max_gap: float, max_dist: float, max_hsv: float) -> float | None:
    """Cost of b continuing a on the same camera, or None when it cannot."""
    gap = b["t0"] - a["t1"]
    if not (-MERGE_OVERLAP_S <= gap <= max_gap) or b["t1"] <= a["t1"]:
        return None
    la, fb = a["boxes"][-1], b["boxes"][0]
    ha, hb = max(la[4], 1e-4), max(fb[4], 1e-4)
    if max(ha, hb) / min(ha, hb) > MERGE_SIZE:
        return None
    pred = _c(la) + _vel(a["boxes"]) * max(gap, 0.0)
    dist = min(np.hypot(*(_c(fb) - pred)), np.hypot(*(_c(fb) - _c(la)))) / max(ha, hb)
    allowed = max_dist + 0.35 * max(gap, 0.0)
    if dist > allowed:
        return None
    d = hsv_distance(a["sig"], b["sig"])
    if d > max_hsv:
        return None
    return d + 0.15 * dist / allowed + 0.03 * max(gap, 0.0)


def merge_fragments(tracks: list[dict[str, Any]], max_gap: float = MERGE_GAP_S, max_dist: float = MERGE_DIST,
                    max_hsv: float = MERGE_HSV) -> list[dict[str, Any]]:
    """Join person fragments of one visit on the same camera (greedy, cheapest pair first). Returns merged tracks
    (other kinds passed through) sorted by (cam, t0); each keeps the first fragment's id and lists all joined ids in
    `fragments`."""
    out: list[dict[str, Any]] = [t for t in tracks if t.get("kind") != "person"]
    for cam in sorted({t["cam"] for t in tracks}):
        ts = sorted([t for t in tracks if t["cam"] == cam and t.get("kind") == "person"], key=lambda t: t["t0"])
        pairs = []
        for i, a in enumerate(ts):
            for j, b in enumerate(ts):
                if i != j:
                    cost = _continues(a, b, max_gap, max_dist, max_hsv)
                    if cost is not None:
                        pairs.append((cost, i, j))
        nxt: dict[int, int] = {}
        prv: dict[int, int] = {}
        for cost, i, j in sorted(pairs):
            if i in nxt or j in prv:
                continue
            k = i                       # no cycles
            while k in prv:
                k = prv[k]
            if k == j:
                continue
            nxt[i], prv[j] = j, i
        for i in range(len(ts)):
            if i in prv:
                continue
            chain = [i]
            while chain[-1] in nxt:
                chain.append(nxt[chain[-1]])
            out.append(_combine([ts[k] for k in chain]))
    return sorted(out, key=lambda t: (t["cam"], t["t0"]))


# ---------------------------------------------------------------- across cameras
def _zone_to(cam: dict[str, Any], to: str) -> str | None:
    """The exit zone of `cam` that its `exits` map sends to camera `to`."""
    for z, target in (cam.get("exits") or {}).items():
        if target == to:
            return z
    return None


def handoffs(tracks: list[dict[str, Any]], cameras: list[dict[str, Any]],
             min_score: float = MIN_SCORE) -> list[dict[str, Any]]:
    """Cross-camera hops {from, to, gap, score, why} between person tracks (scene seconds), one out and one in per
    track at most, best score first. cameras: the topology's camera entries (links, blind, exits)."""
    cams = {c["id"]: c for c in cameras}
    by_cam: dict[str, list[dict[str, Any]]] = {}
    for t in tracks:
        if t.get("kind") == "person" and t["boxes"]:
            by_cam.setdefault(t["cam"], []).append(t)
    cands = []
    for cam in cameras:
        routes = [(r, False) for r in cam.get("links") or []] + [(r, True) for r in cam.get("blind") or []]
        for r, blind in routes:
            to = r["to"]
            if to not in by_cam or cam["id"] not in by_cam:
                continue
            fz = r.get("from_zone") or _zone_to(cam, to)
            tz = r.get("to_zone") or _zone_to(cams.get(to, {}), cam["id"])
            lo, hi = float(r.get("min_s", 0.0)), float(r.get("max_s", 30.0))
            plo, phi = BLIND_PAD if blind else DIRECT_PAD
            for a in by_cam[cam["id"]]:
                ez = exit_zones(a)
                if (fz and fz not in ez) or not ez:
                    continue
                for b in by_cam[to]:
                    dt = b["t0"] - a["t1"]
                    if not (lo - plo <= dt <= hi + phi) or b["t1"] <= a["t1"] or b["t0"] <= a["t0"]:
                        continue            # b must start and end after a: identity tracks stay in time order
                    sim = 1.0 - hsv_distance(a["sig"], b["sig"])
                    out = max(lo - dt, dt - hi, 0.0)
                    timing = 1.0 if out == 0 else max(0.0, 1 - out / (plo if dt < lo else phi))
                    nz = entry_zones(b)
                    zone_ok = (tz in nz) if tz else bool(nz)
                    score = (sim * (0.8 + 0.2 * timing) * (1.0 if zone_ok else ZONE_MISS)
                             * (BLIND_FACTOR if blind else 1.0))
                    cands.append({"from": a["id"], "to": b["id"], "score": round(score, 3), "sim": sim, "dt": dt,
                                  "blind": blind, "overlap": bool(r.get("overlap")),
                                  "ez": fz or "/".join(sorted(ez)),
                                  "nz": tz if zone_ok and tz else "/".join(sorted(nz)) or "mid-frame"})
    cands.sort(key=lambda h: -h["score"])
    used_out: set[str] = set()
    used_in: set[str] = set()
    hops = []
    for h in cands:
        if h["score"] < min_score:
            break
        if h["from"] in used_out or h["to"] in used_in:
            continue
        used_out.add(h["from"])
        used_in.add(h["to"])
        rivals = [c["score"] for c in cands if c is not h and (c["from"] == h["from"] or c["to"] == h["to"])]
        hops.append({"from": h["from"], "to": h["to"], "gap": round(max(0.0, h["dt"]), 1), "score": round(h["score"], 2),
                     "why": _why(h, max(rivals) if rivals else None), "blind": h["blind"]})
    return hops


def _why(h: dict[str, Any], rival: float | None) -> str:
    s = f"colour match {h['sim']:.2f}, exit {h['ez']} -> entry {h['nz']}, "
    if h["dt"] <= 0:
        s += f"on both cameras for {-h['dt']:.0f} s (views overlap)" if h["overlap"] else "no time unseen"
    elif h["blind"]:
        s += f"{h['dt']:.0f} s with no camera between"
    else:
        s += f"{h['dt']:.0f} s"
    if rival is not None and h["score"] - rival <= CLOSE_RUNNER:
        s += f"; next best {rival:.2f}, close call"
    return s


def identities(tracks: list[dict[str, Any]], hops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chains of hops -> identities {id: "P-k", tracks: [ids in time order], hops: [...]}, numbered by first sighting.
    Every person track is in exactly one identity (a track no hop joined is its own)."""
    persons = {t["id"]: t for t in tracks if t.get("kind") == "person"}
    nxt = {h["from"]: h for h in hops if h["from"] in persons and h["to"] in persons}
    has_prev = {h["to"] for h in nxt.values()}
    heads = sorted((t for t in persons.values() if t["id"] not in has_prev), key=lambda t: (t["t0"], t["id"]))
    out = []
    for k, head in enumerate(heads, 1):
        chain, links = [head["id"]], []
        while chain[-1] in nxt and len(chain) <= len(persons):
            h = nxt[chain[-1]]
            links.append({key: h[key] for key in ("from", "to", "gap", "score", "why")})
            chain.append(h["to"])
        out.append({"id": f"P-{k}", "tracks": chain, "hops": links})
    return out


def follow(tracks: list[dict[str, Any]], cameras: list[dict[str, Any]], min_score: float = MIN_SCORE
           ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """merge_fragments -> handoffs -> identities. tracks in scene seconds. Returns (merged, hops, identities)."""
    merged = merge_fragments(tracks)
    hops = handoffs(merged, cameras, min_score)
    return merged, hops, identities(merged, hops)


# ---------------------------------------------------------------- items
def _box_at(tr: dict[str, Any], t: float) -> list[float] | None:
    b = tr["boxes"]
    if not b or t < b[0][0] or t > b[-1][0]:
        return None
    k = int(np.searchsorted([x[0] for x in b], t))
    if k < len(b) and b[k][0] == t or k == 0:
        return b[k][1:]
    lo, hi = b[k - 1], b[k]
    f = (t - lo[0]) / max(1e-6, hi[0] - lo[0])
    return [lo[i] + (hi[i] - lo[i]) * f for i in range(1, 5)]


def item_owner(item: dict[str, Any], tracks: list[dict[str, Any]], ident_of: dict[str, str],
               window: float = 5.0, grow: float = 1.0) -> str | None:
    """Identity whose person box overlaps the item box (grown by `grow` x its larger side on every side) most during
    [t0 - 1, t0 + window], the moments the item is first visible. None when no box overlaps it."""
    x, y, w, h = item["box"]
    g = grow * max(w, h)
    rx1, ry1, rx2, ry2 = x - g, y - g, x + w + g, y + h + g
    t0 = float(item.get("t0") or 0.0)
    best, owner = 0.0, None
    for tr in tracks:
        if tr["cam"] != item["cam"] or tr.get("kind") != "person" or tr["id"] not in ident_of:
            continue
        for t in np.arange(t0 - 1.0, t0 + window + 1e-6, 0.5):
            b = _box_at(tr, float(t))
            if b is None:
                continue
            ov = max(0.0, min(rx2, b[0] + b[2]) - max(rx1, b[0])) * max(0.0, min(ry2, b[1] + b[3]) - max(ry1, b[1]))
            if ov > best:
                best, owner = ov, ident_of[tr["id"]]
    return owner
