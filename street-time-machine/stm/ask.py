"""Question engine over tracks: still_there, when_left, which_way, who_near, count_since, what_changed.

`answer(cam, question, t_now)` parses the question with plain keyword rules, picks the target object
(colour words matched against the crop's dominant colour, names, "#id"), and answers from detections.

Honesty rules:
* A camera that cuts, pans or goes dark is "blind". When out/<cam>/ledger.json exists its blind spans are the
  ONLY source (one blind-time truth across the app); otherwise a cheap scan of our own is used (2-5 Hz frame
  differences: cuts, camera moves, dark, frozen).
* The scan also groups samples into "views" (camera framings). A box from one view says nothing about
  another view, so presence is only ever checked inside the target's own view.
* Departures next to blind time are UNKNOWN (with the bounds we do know), never "gone".
* People are never described or searched by how they look; blame questions ("who dented ...") get who was
  near plus "it can't say who did it". Every time is printed as mm:ss.
* Day words (yesterday, today, first/second day, Day A, '1 Sep') map onto the camera's days in order; day ids
  and labels are read from meta.json, never assumed. Day-to-day answers come from diff.json when it exists.

The per-camera scan (colours, views, blind spans, best-frame crops) is cached in out/<cam>/ask_index.json.
CLI: python -m stm.ask [cams...]  -> out/<cam>/qa.json + out/<cam>/ask_summary.txt
"""
from __future__ import annotations

import json
import re
import sys
import time
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm import data

TYPES = ("still_there", "when_left", "which_way", "who_near", "count_since", "what_changed")
DEMO_CAMS = ("ptz_parking", "person_bicycle_car_detection", "car_detection")
INDEX_VERSION = 7
CROPS = "ask_crops"     # own folder: other modules rewrite crops/

COLOUR_WORDS = {"red": "red", "maroon": "red", "orange": "orange", "yellow": "yellow", "green": "green",
                "blue": "blue", "navy": "blue", "purple": "purple", "white": "white", "silver": "grey",
                "grey": "grey", "gray": "grey", "black": "black", "dark": "black"}
NAME_WORDS = {"car": ("car", "truck"), "cars": ("car", "truck"), "vehicle": data.VEHICLES, "vehicles": data.VEHICLES,
              "suv": ("car", "truck"), "van": ("truck", "car"), "vans": ("truck", "car"), "truck": ("truck", "car"),
              "trucks": ("truck", "car"), "pickup": ("truck", "car"), "bus": ("bus",), "buses": ("bus",),
              "motorcycle": ("motorcycle",), "motorbike": ("motorcycle",), "bicycle": ("bicycle",), "bike": ("bicycle",),
              "bikes": ("bicycle",), "bicycles": ("bicycle",), "cyclist": ("bicycle",), "cyclists": ("bicycle",),
              "person": ("person",), "people": ("person",), "pedestrian": ("person",), "pedestrians": ("person",),
              "man": ("person",), "woman": ("person",), "someone": ("person",), "walker": ("person",)}
PRESENT_IOU = 0.45     # a detection at the target's box counts as "the target is there"
NEAR_REL = 0.5         # who_near: edge-to-edge gap / target height
BLIND_PAD = 2.0        # a sighting this close to blind time cannot prove a departure

# plain words for blind reasons (ledger + scan vocabularies)
REASON_WORDS = {"turned_away": "camera turned away", "cut": "camera turned away", "camera_moving": "camera turned away",
                "view_changed": "camera turned away", "blocked": "view blocked", "dark": "too dark", "blur": "blurred",
                "frozen": "picture frozen", "no_signal": "no signal", "blind": "camera blind"}
REFUSE_LOOKS = ("Street Time Machine never describes or searches people by how they look: no clothing, skin, gender "
                "or face. Ask about vehicles, or who came near a car.")
PERSON_WORDS = {"person", "persons", "people", "man", "men", "woman", "women", "guy", "guys", "girl", "girls", "boy",
                "boys", "lady", "ladies", "bloke", "kid", "kids", "child", "children", "someone", "somebody",
                "anyone", "anybody", "pedestrian", "pedestrians", "walker", "walkers", "he", "she", "him", "her"}
GENDER_WORDS = {"man", "men", "woman", "women", "guy", "guys", "girl", "girls", "boy", "boys", "lady", "ladies",
                "bloke", "he", "she", "him", "her"}
LOOK_WORDS = {"shirt", "tshirt", "jacket", "coat", "hoodie", "hood", "hat", "cap", "dress", "skirt", "jeans",
              "trousers", "pants", "shorts", "jumper", "sweater", "scarf", "shoes", "trainers", "boots", "hair",
              "haired", "beard", "bearded", "glasses", "skin", "skinned", "wearing", "dressed", "clothes", "clothing",
              "outfit", "uniform", "face", "faces", "tall", "fat", "thin", "blonde", "blond", "brunette", "ginger",
              "race", "ethnicity", "asian", "african", "caucasian", "hijab", "tattoo", "tattoos", "backpack"}
LOOK_COLOURS = sorted((set(COLOUR_WORDS) - {"dark"}) | {"brown", "pink", "beige"})
VEHICLE_NOUNS = {w for w, v in NAME_WORDS.items() if v and v[0] != "person"}
BLAME_RE = re.compile(r"\b(dent(?:ed|s)?|scratch(?:ed|es)?|hit|hits|stole|stolen|steal(?:s|ing)?|damag(?:e|ed|es|ing)|"
                      r"keyed|broke|broken|smash(?:ed)?|vandali[sz](?:e|ed)|crash(?:ed)?(?:\s+into)?|scrap(?:e|ed)|"
                      r"bump(?:ed)?(?:\s+into)?|rob(?:bed)?|nicked|blame|fault|responsible|culprit|suspect|guilty|"
                      r"did (?:it|this|that))\b")
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MON = "|".join(MONTHS)


# ----------------------------------------------------------------------------- small helpers

def mmss(t: float) -> str:
    """mm:ss, floored like the UI's clock, never negative ('-0.00' cannot happen)."""
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def span(a: float, b: float) -> str:
    return f"{mmss(a)}-{mmss(b)}"


def reason_word(reason: str | None) -> str:
    first = str(reason or "blind").split("+")[0].strip().replace(" ", "_")
    return REASON_WORDS.get(first, first.replace("_", " "))


@lru_cache(maxsize=None)
def _meta(cam: str) -> dict[str, Any]:
    p = data.OUT / cam / "meta.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


@lru_cache(maxsize=None)
def duration(cam: str) -> float:
    """Video length. Falls back to meta.json when the video itself is not on this machine (Pack D built elsewhere)."""
    d = float(data.info(cam)["duration"] or 0.0) if cam in data.CAMS else 0.0
    return d if d > 0 else float(_meta(cam).get("duration") or 0.0)


@lru_cache(maxsize=None)
def days_of(cam: str) -> tuple[dict[str, Any], ...]:
    """The camera's days in order, ids and labels from meta.json (else stm.data). Never assumes 'A'/'B'."""
    ds = _meta(cam).get("days")
    if not ds:
        ds = data.days(cam)
    return tuple({"id": str(d["id"]), "label": str(d.get("label") or d["id"]), "start": float(d["start"]),
                  "end": float(d["end"])} for d in ds)


def _json_file(cam: str, name: str) -> Any:
    p = data.OUT / cam / name
    try:
        return _json_cached(str(p), p.stat().st_mtime)
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=32)
def _json_cached(path: str, mtime: float) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def r2(x: float) -> float:
    return round(float(x), 2)


def iou_many(b: np.ndarray, B: np.ndarray) -> np.ndarray:
    """IoU of box b [4] against boxes B [n,4]."""
    if len(B) == 0:
        return np.zeros(0)
    ix = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = ix * iy
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-6)


def gap_many(b: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Edge-to-edge pixel gap between box b and boxes B (0 when they overlap)."""
    dx = np.maximum(0, np.maximum(b[0] - B[:, 2], B[:, 0] - b[2]))
    dy = np.maximum(0, np.maximum(b[1] - B[:, 3], B[:, 1] - b[3]))
    return np.hypot(dx, dy)


def colour_of(img: np.ndarray, name: str) -> dict[str, float]:
    """Fraction of pixels per colour word in the body of the crop (vehicles: centre band; people: torso)."""
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return {}
    if name == "person":
        sub = img[int(h * 0.2):int(h * 0.55), int(w * 0.25):int(w * 0.75)]
    else:
        sub = img[int(h * 0.3):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    if sub.size == 0:
        return {}
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lab = np.full(len(H), "grey", dtype=object)      # grey/silver: low saturation, mid brightness
    lab[(S < 50) & (V >= 175)] = "white"
    lab[V < 70] = "black"
    chroma = (S >= 70) & (V >= 60)
    lab[chroma & ((H < 9) | (H >= 165))] = "red"
    lab[chroma & (H >= 9) & (H < 21)] = "orange"
    lab[chroma & (H >= 21) & (H < 35)] = "yellow"
    lab[chroma & (H >= 35) & (H < 85)] = "green"
    lab[chroma & (H >= 85) & (H < 130)] = "blue"
    lab[chroma & (H >= 130) & (H < 165)] = "purple"
    vals, cnt = np.unique(lab, return_counts=True)
    return {str(v): round(float(c) / len(lab), 3) for v, c in zip(vals, cnt)}


def dominant(fr: dict[str, float]) -> str | None:
    if not fr:
        return None
    chrom = {k: v for k, v in fr.items() if k in ("red", "orange", "yellow", "green", "blue", "purple")}
    if chrom:
        k = max(chrom, key=chrom.get)
        if chrom[k] >= 0.25:     # paint colour wins over windows/shadow when it covers a quarter of the body
            return k
    return max(fr, key=fr.get)


# ----------------------------------------------------------------------------- per-camera index

def _merge(spans: list[list[Any]], gap: float = 0.6) -> list[list[Any]]:
    spans = sorted(spans)
    out: list[list[Any]] = []
    for s in spans:
        if out and s[0] <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], s[1])
            if s[2] not in out[-1][2]:
                out[-1][2] = out[-1][2] + "+" + s[2]
        else:
            out.append(list(s))
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    za, zb = (a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)
    return float((za * zb).mean())


def _scan(cam: str) -> dict[str, Any]:
    """One sequential pass over the video: blind spans, views and best-frame crops/colours."""
    info = data.info(cam)
    if not info.get("frames"):
        raise FileNotFoundError(f"video for {cam} is not on this machine: build out/{cam}/ask_index.json where it is "
                                f"(python -m stm.ask {cam}) and copy out/{cam}/ here")
    fps, W, H = info["fps"], info["w"], info["h"]
    objs = data.objects(cam)
    tr = data.tracks(cam)
    tr = tr[tr.tid.isin(objs.id)] if len(objs) else tr.iloc[:0]
    best = tr.loc[(tr.bw * tr.bh).groupby(tr.tid).idxmax()] if len(tr) else tr
    want: dict[int, list[Any]] = {}
    for r in best.itertuples():
        want.setdefault(int(r.f), []).append(r)
    step = max(1, int(round(fps / 2.5)))   # ~2.5 Hz samples for cut / camera-move detection
    sub = max(1, int(min(W / 96, H / 54) // 2))   # cheap pre-decimation before the area resize
    nthreads = cv2.getNumThreads()
    cv2.setNumThreads(2)        # small images: thread fan-out costs more than it saves on a busy CPU
    cap = cv2.VideoCapture(str(data.CAMS[cam]["video"]))
    smp_t, smp_mean, smp_frac, smp_mad, smp_edge = [], [], [], [], []
    sub2 = max(1, int(min(W / 192, H / 108) // 2))
    colours: dict[str, Any] = {}
    crops_dir = data.out_dir(cam) / CROPS
    crops_dir.mkdir(parents=True, exist_ok=True)
    i, prev = 0, None
    while True:
        if not cap.grab():
            break
        need = want.get(i)
        if i % step == 0 or need:
            ok, img = cap.retrieve()
            if ok:
                if i % step == 0:
                    g = cv2.cvtColor(cv2.resize(img[::sub, ::sub], (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                    g = cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)
                    dd = np.abs(g - prev) if prev is not None else np.zeros_like(g)
                    cells = dd.reshape(6, 9, 6, 16).mean(axis=(1, 3))     # 6x6 grid of mean change
                    smp_t.append(i / fps); smp_mean.append(float(g.mean()))
                    smp_frac.append(float((cells > 8.0).mean())); smp_mad.append(float(dd.mean()))
                    g2 = cv2.cvtColor(cv2.resize(img[::sub2, ::sub2], (192, 108), interpolation=cv2.INTER_AREA),
                                      cv2.COLOR_BGR2GRAY).astype(np.float32)
                    smp_edge.append(np.hypot(cv2.Sobel(g2, cv2.CV_32F, 1, 0), cv2.Sobel(g2, cv2.CV_32F, 0, 1)).astype(np.float16))
                    prev = g
                for r in need or []:
                    px, py = r.bw * 0.15, r.bh * 0.15
                    x1, y1 = int(max(0, r.x1 - px)), int(max(0, r.y1 - py))
                    x2, y2 = int(min(W, r.x2 + px)), int(min(H, r.y2 + py))
                    c = img[y1:y2, x1:x2]
                    if c.size == 0:
                        continue
                    cv2.imwrite(str(crops_dir / f"{int(r.tid)}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    fr = colour_of(img[int(r.y1):int(r.y2), int(r.x1):int(r.x2)], r.name)
                    colours[str(int(r.tid))] = {"colour": dominant(fr), "fracs": fr, "t": r2(r.t)}
        i += 1
    cap.release()
    cv2.setNumThreads(nthreads)
    t = np.array(smp_t); mean = np.array(smp_mean); frac = np.array(smp_frac); mad = np.array(smp_mad)
    dt = step / fps
    spans: list[list[Any]] = []
    # a still street changes < 2 grey levels per sample; a cut or a camera move changes 10-50 over the WHOLE
    # picture, while a car driving close to the lens changes only part of it -> need >= 80 % of grid cells
    moving = (mad > 8.0) & (frac >= 0.80)
    for k in np.flatnonzero(moving):
        spans.append([t[max(0, k - 1)], t[k], "cut"])
    dark = mean < 22
    for k in np.flatnonzero(dark):
        spans.append([t[k] - dt / 2, t[k] + dt / 2, "dark"])
    still = np.r_[False, mad[1:] < 0.05]     # bit-identical pictures (a real street always flickers a little)
    run = 0
    for k in range(len(still)):
        run = run + 1 if still[k] else 0
        if run * dt >= 3.0:
            spans.append([t[k] - run * dt, t[k], "frozen"])
    # Motion spans (big global change) are kept only when they really are the camera:
    #  * a tracked object covering > 8 % of the frame explains it (a car right under the lens + auto-exposure);
    #  * a multi-sample change whose before/after pictures have the same edges is an exposure ramp, not a move.
    # Edge-map correlation (calibrated on ptz_parking): same framing 0.61-0.95, different framing <= 0.13.
    big = tr.assign(a=tr.bw * tr.bh / float(W * H)) if len(tr) else tr

    def explained(s0: float, s1: float) -> bool:
        if not len(big):
            return False
        m = (big.t >= s0 - 0.3) & (big.t <= s1 + 0.3)
        return bool(m.any() and big.a[m].max() > 0.08)

    motion: list[list[Any]] = []
    for s0, s1, _ in _merge([sp for sp in spans if sp[2] == "cut"], gap=dt * 1.5):
        if explained(s0, s1):
            continue
        kb = int(max(0, np.searchsorted(t, s0, "right") - 1))
        ka = int(min(len(t) - 1, np.searchsorted(t, s1, "left")))
        single = (s1 - s0) <= dt * 1.5
        if not single and _ncc(smp_edge[kb], smp_edge[ka]) > 0.45:
            continue
        motion.append([s0, s1, "cut" if single else "camera_moving"])
    moving = np.zeros(len(t), bool)
    for s0, s1, _ in motion:
        moving |= (t > s0 - 1e-6) & (t <= s1 + 1e-6)
    merged = _merge(motion + [sp for sp in spans if sp[2] != "cut"])
    # views: stable stretches between motion spans; a stretch whose edges match an earlier view reuses its id
    view = np.full(len(t), -1, dtype=int)
    keys: list[np.ndarray] = []
    blind_mask = np.zeros(len(t), bool)
    for s, e, rs in motion:
        if rs == "camera_moving":
            blind_mask |= (t > s) & (t < e)
    cur = -1
    for k in range(len(t)):
        if blind_mask[k]:
            cur = -1
            continue
        if cur < 0 or moving[k]:
            sims = [_ncc(smp_edge[k], kz) for kz in keys]
            if sims and max(sims) > 0.45:
                cur = int(np.argmax(sims))
            else:
                keys.append(smp_edge[k]); cur = len(keys) - 1
        view[k] = cur
    return {"version": INDEX_VERSION, "camera": cam, "fps": fps, "w": W, "h": H, "duration": info["duration"],
            "samples": {"t": [r2(x) for x in t], "view": view.tolist()},
            "scan_blind": [{"start": r2(s), "end": r2(e), "reason": rs} for s, e, rs in merged],
            "colours": colours, "n_views": len(keys)}


@lru_cache(maxsize=None)
def index(cam: str) -> dict[str, Any]:
    p = data.out_dir(cam) / "ask_index.json"
    idx = None
    if p.exists():
        try:
            idx = json.loads(p.read_text(encoding="utf-8"))
            if idx.get("version") != INDEX_VERSION:
                idx = None
        except Exception:
            idx = None
    if idx is None:
        idx = _scan(cam)
        data.write_json(p, idx)
    idx["_t"] = np.array(idx["samples"]["t"], dtype=float)
    idx["_v"] = np.array(idx["samples"]["view"], dtype=int)
    T, V = idx["_t"], idx["_v"]
    # one blind-time truth: the ledger's spans when it exists, our own scan only as a fallback
    blind = [dict(b, source="scan") for b in idx["scan_blind"]]
    idx["ledger"] = False
    idx["blind_total_s"] = None
    lp = data.out_dir(cam) / "ledger.json"
    if lp.exists():
        try:
            L = json.loads(lp.read_text(encoding="utf-8"))
            # "home" views = framings the camera holds for >= 15 % of the clip; a 'blocked' ledger span during
            # which the camera shows some other stable framing is the camera pointing elsewhere, not a block
            vv, cnt = np.unique(V[V >= 0], return_counts=True)
            home = {int(v) for v, c in zip(vv, cnt) if c >= 0.15 * max(1, len(V))}
            lb = []
            for b in L.get("blind", []):
                s, e, rs = float(b["start"]), float(b["end"]), b.get("reason") or "blind"
                if rs == "blocked" and len(home) > 1:
                    m = (T > s) & (T < e)
                    if m.any() and float(((V[m] >= 0) & ~np.isin(V[m], list(home))).mean()) >= 0.5:
                        rs = "turned_away"
                lb.append({"start": s, "end": e, "reason": rs, "source": "ledger"})
            blind = lb
            idx["ledger"] = True
            if L.get("blind_total_s") is not None:
                idx["blind_total_s"] = float(L["blind_total_s"])
        except Exception:
            pass
    idx["blind"] = sorted(blind, key=lambda b: b["start"])
    # moments the camera itself moved (scan): not counted as blind time when the ledger rules, but a departure
    # or a "new" track right at one is the camera turning, never a seen exit or a real arrival
    idx["view_changes"] = [dict(b, source="scan") for b in idx["scan_blind"]
                           if idx["ledger"] and any(k in b["reason"] for k in ("cut", "camera_moving"))]
    bm = np.zeros(len(T), bool)
    for b in idx["blind"]:
        bm |= (T >= b["start"]) & (T <= b["end"])
    idx["_blind"] = bm
    # a FIXED camera (diff.json lined the days up with < 3 px shift over the whole picture) has one framing, even if
    # the scan split it into two "views" at a light change (e.g. day 2 in infrared): boxes compare across them
    D = _json_file(cam, "diff.json")
    al = D.get("alignment") if isinstance(D, dict) else None
    idx["fixed"] = bool(isinstance(al, dict) and al.get("ok") and float(al.get("corner_shift_px", 1e9)) <= 3.0
                        and float(al.get("overlap_frac", 0.0)) >= 0.99)
    if idx["fixed"]:
        idx["_v"] = np.where(idx["_v"] >= 0, 0, idx["_v"])
    idx["_dt"] = float(np.median(np.diff(T))) if len(T) > 1 else 0.2
    return idx


@lru_cache(maxsize=None)
def dets(cam: str) -> dict[str, np.ndarray]:
    df = data.tracks(cam)
    df = df[df.name.isin(data.MOVERS)]
    return {"t": df.t.to_numpy(float), "f": df.f.to_numpy(int), "tid": df.tid.to_numpy(int),
            "name": df.name.to_numpy(str), "box": df[["x1", "y1", "x2", "y2"]].to_numpy(float),
            "veh": df.name.isin(data.VEHICLES).to_numpy()}


@lru_cache(maxsize=None)
def objs(cam: str) -> pd.DataFrame:
    o = data.objects(cam)
    if o.empty:
        return o
    col = index(cam)["colours"]
    o = o.copy()
    o["colour"] = [col.get(str(i), {}).get("colour") for i in o.id]
    o["bw"] = o.x2 - o.x1
    # moved = the centre really went somewhere (5-95 % spread), not summed jitter of a parked box
    tr = data.tracks(cam)
    g = tr[tr.tid.isin(o.id)].groupby("tid")
    q = g[["cx", "cy"]].quantile(0.95) - g[["cx", "cy"]].quantile(0.05)
    spread = np.hypot(q.cx, q.cy)
    o["spread_px"] = [float(spread.get(i, 0.0)) for i in o.id]
    o["moved"] = o.spread_px >= 0.5 * np.maximum(o.bw, o.y2 - o.y1)
    o["view"] = [view_at(cam, (a + b) / 2) for a, b in zip(o.first_t, o.last_t)]
    return o


def view_at(cam: str, t: float) -> int:
    idx = index(cam)
    T, V = idx["_t"], idx["_v"]
    if len(T) == 0:
        return 0
    k = int(np.clip(np.searchsorted(T, t), 0, len(T) - 1))
    if k > 0 and abs(T[k - 1] - t) < abs(T[k] - t):
        k -= 1
    return int(V[k])


def views_of(cam: str, ts: np.ndarray) -> np.ndarray:
    idx = index(cam)
    T, V = idx["_t"], idx["_v"]
    if len(T) == 0:
        return np.zeros(len(ts), int)
    k = np.clip(np.searchsorted(T, ts), 0, len(T) - 1)
    return V[k]


def blind_between(cam: str, a: float, b: float) -> list[dict[str, Any]]:
    return [x for x in index(cam)["blind"] if x["end"] > a and x["start"] < b]


def _ir(cam: str, t: float) -> bool:
    """Is the picture at t in infrared (diff.json 'infrared' per day id)? False when unknown."""
    D = _json_file(cam, "diff.json")
    ir = D.get("infrared") if isinstance(D, dict) else None
    if not isinstance(ir, dict):
        return False
    for d in days_of(cam):
        if d["start"] <= t <= d["end"]:
            return bool(ir.get(d["id"], False))
    return False


def unseen_between(cam: str, a: float, b: float) -> list[dict[str, Any]]:
    """Blind spans plus camera-move moments overlapping [a, b] (for 'was the exit/arrival really seen')."""
    idx = index(cam)
    return sorted([x for x in idx["blind"] + idx["view_changes"] if x["end"] > a and x["start"] < b],
                  key=lambda x: x["start"])


def blind_seconds(cam: str, a: float, b: float) -> float:
    iv = sorted([max(a, x["start"]), min(b, x["end"])] for x in blind_between(cam, a, b))
    tot, cur = 0.0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur:
        tot += cur[1] - cur[0]
    return tot


def blind_runs(spans: list[dict[str, Any]], a: float | None = None, b: float | None = None,
               gap: float = 1.0) -> list[dict[str, Any]]:
    """Touching blind spans merged into runs, clipped to [a, b], each named by its longest piece in plain words
    (ledger fragments 'blocked 156-157, blur 157-159, turned_away 159-237 ...' -> 'camera turned away 02:36-04:11')."""
    iv = []
    for x in spans:
        s, e = float(x["start"]), float(x["end"])
        if a is not None:
            s = max(s, a)
        if b is not None:
            e = min(e, b)
        if e > s:
            iv.append((s, e, reason_word(x.get("reason"))))
    iv.sort()
    runs: list[dict[str, Any]] = []
    for s, e, w in iv:
        if runs and s <= runs[-1]["end"] + gap:
            runs[-1]["end"] = max(runs[-1]["end"], e)
            runs[-1]["_w"][w] = runs[-1]["_w"].get(w, 0.0) + (e - s)
        else:
            runs.append({"start": s, "end": e, "_w": {w: e - s}})
    return [{"start": r2(max(0.0, r["start"])), "end": r2(r["end"]), "reason": max(r["_w"], key=r["_w"].get)}
            for r in runs]


def describe_blind(spans: list[dict[str, Any]]) -> str:
    """'camera turned away 02:36-04:11 (95 s); no signal 01:08-01:10 (2 s)'. Accepts raw spans or runs."""
    runs = spans if spans and all(x.get("reason") in REASON_WORDS.values() for x in spans) else blind_runs(spans)
    def dur(x: dict[str, Any]) -> str:
        d = x["end"] - x["start"]
        return f"{d:.0f} s" if d >= 1.0 else "<1 s"
    txt = "; ".join(f"{x['reason']} {span(x['start'], x['end'])} ({dur(x)})" for x in runs[:3])
    return txt + (f"; and {len(runs) - 3} more" if len(runs) > 3 else "")


# ----------------------------------------------------------------------------- parsing

def _day_refs(q: str) -> list[tuple[int, str, Any]]:
    """Day words in the question, as (position, kind, value); resolved against a camera's days later."""
    refs: list[tuple[int, str, Any]] = []
    pats = [(r"\byesterday\b", "rel", -2), (r"\b(?:today|tonight|this morning|this evening)\b", "rel", -1),
            (r"\b(?:first|1st)\s+day\b|\bday\s+(?:1|one)\b", "ord", 0),
            (r"\b(?:second|2nd)\s+day\b|\bday\s+(?:2|two)\b", "ord", 1),
            (r"\b(?:third|3rd)\s+day\b|\bday\s+(?:3|three)\b", "ord", 2),
            (r"\b(?:last|final|latest)\s+day\b", "rel", -1), (r"\b(?:previous|earlier)\s+day\b", "rel", -2)]
    for pat, kind, val in pats:
        for m in re.finditer(pat, q):
            refs.append((m.start(), kind, val))
    for m in re.finditer(r"\bday\s+([a-h])\b", q):
        refs.append((m.start(), "letter", m.group(1).upper()))
    for m in re.finditer(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MON})[a-z]*\b", q):
        refs.append((m.start(), "date", (int(m.group(1)), m.group(2))))
    for m in re.finditer(rf"\b({_MON})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?\b", q):
        refs.append((m.start(), "date", (int(m.group(2)), m.group(1))))
    for m in re.finditer(r"\b(20\d{6})\b", q):
        refs.append((m.start(), "id", m.group(1)))
    return sorted(refs, key=lambda x: x[0])


def resolve_days(cam: str, p: dict[str, Any]) -> list[int]:
    """Indices into days_of(cam) for the day words in the question, in the order asked (duplicates dropped)."""
    ds = days_of(cam)
    n = len(ds)

    def key(d: dict[str, Any]) -> set[Any]:
        ks: set[Any] = {d["id"].lower(), d["label"].lower()}
        m = re.fullmatch(r"(20\d{2})(\d{2})(\d{2})", d["id"])
        if m and 1 <= int(m.group(2)) <= 12:
            ks.add((int(m.group(3)), MONTHS[int(m.group(2)) - 1]))
        m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON})", d["label"].lower()) or None
        if m:
            ks.add((int(m.group(1)), m.group(2)))
        m = re.search(rf"\b({_MON})[a-z]*\s+(\d{{1,2}})", d["label"].lower())
        if m:
            ks.add((int(m.group(2)), m.group(1)))
        return ks
    keys = [key(d) for d in ds]
    out: list[int] = []
    for _, kind, val in p.get("day_refs") or []:
        k = None
        if kind == "rel" and n >= (2 if val == -2 else 1):
            k = n + val
        elif kind == "ord" and val < n:
            k = val
        elif kind == "letter":
            hit = [i for i, d in enumerate(ds) if d["id"].upper() == val or d["label"].upper() == f"DAY {val}"]
            k = hit[0] if hit else (ord(val) - 65 if ord(val) - 65 < n else None)
        elif kind in ("date", "id"):
            hit = [i for i, ks in enumerate(keys) if (val if kind == "date" else val.lower()) in ks]
            k = hit[0] if hit else None
        if k is not None and k not in out:
            out.append(k)
    return out


def _appearance(q: str, words: list[str]) -> bool:
    """True when the question describes a person by how they look (clothing, colour, body, face)."""
    if not any(w in PERSON_WORDS for w in words):
        return False
    if any(w in LOOK_WORDS for w in words):
        return True
    cols = "|".join(LOOK_COLOURS)
    pers = "|".join(sorted(PERSON_WORDS))
    vehs = "|".join(sorted(VEHICLE_NOUNS))
    if re.search(rf"\b({cols})(?:-|\s+)(?:skinned\s+)?({pers})\b", q):
        return True
    if re.search(rf"\b({pers})\b.*?\b(?:in|wearing|with|dressed in)\s+(?:a\s+|an\s+|the\s+|all\s+)?({cols})\b(?!\s+({vehs})\b)", q):
        return True
    has_col = any(w in LOOK_COLOURS for w in words)
    return has_col and not any(w in VEHICLE_NOUNS for w in words)


def parse(question: str) -> dict[str, Any]:
    q = " " + question.lower().replace("?", " ").replace(",", " ").replace("-", " ") + " "
    blame = BLAME_RE.search(q)
    if blame and not re.search(r"\b(who|whose|which|what|anyone|anybody|someone|somebody|culprit|suspect|blame|"
                               r"fault|responsible|guilty)\b", q):
        blame = None
    if blame:
        typ = "who_near"      # "who dented the red car" -> who was near it, plus "it can't say who did it"
    elif "still there" in q or "still here" in q or "still parked" in q or re.search(r"\b(is|was) (the|a|that) .* there\b", q):
        typ = "still_there"
    elif ("when" in q and ("leave" in q or "left" in q or "go" in q.split() or "drive off" in q)):
        typ = "when_left"
    elif "which way" in q or "direction" in q or "where did" in q and ("go" in q or "head" in q):
        typ = "which_way"
    elif " near " in q or "close to" in q or "touch" in q or "came by the" in q or "approach" in q:
        typ = "who_near"
    elif "how many" in q or "count" in q or "number of" in q:
        typ = "count_since"
    elif "what changed" in q or "different" in q or "difference" in q or "change" in q:
        typ = "what_changed"
    else:
        typ = None
    words = re.findall(r"[a-z]+", q)
    focus = q
    if blame:
        focus = q[blame.end():]
    elif typ == "who_near":   # the target is what follows "near"/"close to"/"touch"
        m = re.search(r"(near|close to|touch\w*|approach\w*|came by)(.*)", q)
        focus = m.group(2) if m else q
    fw = re.findall(r"[a-z]+", focus)
    colour = next((COLOUR_WORDS[w] for w in fw if w in COLOUR_WORDS), None)
    names = next((NAME_WORDS[w] for w in fw if w in NAME_WORDS), None)
    if typ in ("count_since", "what_changed"):
        names = next((NAME_WORDS[w] for w in words if w in NAME_WORDS), None)
        colour = next((COLOUR_WORDS[w] for w in words if w in COLOUR_WORDS and w != "dark"), None)
    m = re.search(r"(?:#|\b(?:id|object|track)\s*#?)\s*(\d+)", q)
    oid = int(m.group(1)) if m else None
    since = at = None
    for m in re.finditer(r"\b(since|after|from|at|by|before)\s+(\d{1,2}):(\d{2})\b", q):
        v = int(m.group(2)) * 60 + int(m.group(3))
        if m.group(1) in ("since", "after", "from"):
            since = float(v)
        else:
            at = float(v)
    m = re.search(r"\b(since|after|from|at|by)\s+(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?|m|mins?|minutes?)\b", q)
    if m:
        v = float(m.group(2)) * (60 if m.group(3).startswith("m") else 1)
        if m.group(1) in ("since", "after", "from"):
            since = v
        else:
            at = v
    m = re.search(r"\b(?:last|past)\s+(\d+(?:\.\d+)?)?\s*(s|sec|secs|seconds?|m|mins?|minutes?)\b", q)
    last = None
    if m:
        last = float(m.group(1) or 1) * (60 if m.group(2).startswith("m") else 1)
    refs = _day_refs(q)
    since_day = bool(refs) and bool(re.search(r"\b(since|after|from)\b", q[:refs[0][0] + 1]))
    return {"type": typ, "colour": colour, "names": names, "object_id": oid, "since": since, "at": at,
            "last": last, "day_refs": refs, "since_day": since_day, "blame": bool(blame),
            "appearance": _appearance(q, words), "gender": any(w in GENDER_WORDS for w in words), "text": question}


# ----------------------------------------------------------------------------- target selection

def _crop_path(cam: str, oid: int) -> str | None:
    p = data.out_dir(cam) / CROPS / f"{oid}.jpg"
    return f"{CROPS}/{oid}.jpg" if p.exists() else None


def _box(r) -> list[float]:
    return [r2(r.x1), r2(r.y1), r2(r.x2), r2(r.y2)]


def _row(df: pd.DataFrame):
    return next(df.iloc[[0]].itertuples(index=False))


def pick(cam: str, p: dict[str, Any], t_now: float, want: str) -> tuple[Any, str | None]:
    """Return (object row, problem). want: 'parked' (still_there/when_left/who_near) or 'mover' (which_way)."""
    o = objs(cam)
    if o.empty:
        return None, "no tracked objects in this camera"
    if p["object_id"] is not None:
        hit = o[o.id == p["object_id"]]
        return (_row(hit), None) if len(hit) else (None, f"no object #{p['object_id']} in this camera")
    c = o[o.first_t <= t_now]
    names = p["names"] or (data.VEHICLES if want == "parked" else None)
    if names:
        c = c[c.name.isin(names)]
    if want == "parked":
        long_ = c[c.parked | (c.n >= 45) & ~c.moved]
        c = long_ if len(long_) else c
        if p["type"] == "when_left":      # "when did X leave": prefer ones that are no longer tracked
            gone = c[c.last_t < t_now - 1.0]
            c = gone if len(gone) else c
    else:
        mv = c[c.moved]
        c = mv if len(mv) else c
    what = " ".join(x for x in (p["colour"], (p["names"] or ["object"])[0]) if x)
    if c.empty:
        return None, f"no {what} seen before {mmss(t_now)}"
    if p["colour"]:
        fr = [index(cam)["colours"].get(str(i), {}).get("fracs", {}).get(p["colour"], 0.0) for i in c.id]
        c = c.assign(cscore=fr, dom=(c.colour == p["colour"]).astype(float))
        c = c[(c.cscore >= 0.15) | (c.dom > 0)]
        if c.empty:
            return None, f"no {what} found (colour judged from each crop's dominant colour)"
        c = c.sort_values(["dom", "n", "cscore"], ascending=False)
        return _row(c), None
    c = c.assign(seen=np.minimum(c.last_t, t_now)).sort_values(["seen", "n"], ascending=False)
    return _row(c), None


# ----------------------------------------------------------------------------- presence at a box

def presence(cam: str, box: list[float], view: int, t0: float, t1: float, names=data.VEHICLES) -> tuple[np.ndarray, np.ndarray]:
    """Times and tids of detections at `box` (IoU >= PRESENT_IOU) in [t0, t1] inside camera view `view`."""
    d = dets(cam)
    lo, hi = np.searchsorted(d["t"], t0, "left"), np.searchsorted(d["t"], t1, "right")
    nm = d["veh"][lo:hi] if tuple(names) == tuple(data.VEHICLES) else np.isin(d["name"][lo:hi], names)
    m = np.zeros(len(d["t"]), bool)
    m[lo:hi] = nm
    if not m.any():
        return np.zeros(0), np.zeros(0, int)
    idx = np.flatnonzero(m)
    hit = iou_many(np.array(box, float), d["box"][idx]) >= PRESENT_IOU
    idx = idx[hit]
    if len(idx):
        idx = idx[views_of(cam, d["t"][idx]) == view]
    return d["t"][idx], d["tid"][idx]


def seen_time(cam: str, view: int, a: float, b: float) -> float:
    """Seconds in (a, b] where the camera showed `view` and was not blind."""
    idx = index(cam)
    T = idx["_t"]
    m = (T > a) & (T <= b) & (idx["_v"] == view) & ~idx["_blind"]
    return float(m.sum()) * idx["_dt"]


def first_seen_after(cam: str, view: int, a: float, b: float) -> float | None:
    idx = index(cam)
    T = idx["_t"]
    m = (T > a) & (T <= b) & (idx["_v"] == view) & ~idx["_blind"]
    return float(T[m].min()) if m.any() else None


def chain(cam: str, box: list[float], view: int, t0: float, t_now: float, names) -> tuple[float, int, list[list[float]]]:
    """Follow detections at `box` from t0: the run ends at the first gap with > 3 s of visible, in-view, empty spot.
    Gaps the camera did not see are bridged and returned so the answer can say so."""
    ts, tids = presence(cam, box, view, t0, t_now, names)
    if len(ts) == 0:
        return -1.0, -1, []
    o = np.argsort(ts, kind="stable")
    ts, tids = ts[o], tids[o]
    cur, cur_tid, bridged = float(ts[0]), int(tids[0]), []
    for t, tid in zip(ts[1:], tids[1:]):
        if t - cur > 3.0:
            if seen_time(cam, view, cur, t) > 3.0:
                break
            # across unseen time: only the same-looking vehicle continues the run (a look-alike swap stays possible)
            sim, _ = look_same(cam, box, cur, float(t))
            if not (sim == sim and sim >= SAME_LOOK):
                break
            bridged.append([r2(cur), r2(t)])
        cur, cur_tid = float(t), int(tid)
    return cur, cur_tid, bridged


def exit_edge(cam: str, row, dx: float, dy: float) -> str | None:
    W, H = index(cam)["w"], index(cam)["h"]
    mx, my = 0.03 * W, 0.03 * H
    if row.x1 <= mx and dx <= 0:
        return "left"
    if row.x2 >= W - mx and dx >= 0:
        return "right"
    if row.y2 >= H - my and dy >= 0:
        return "bottom"
    if row.y1 <= my and dy <= 0:
        return "top"
    return None


def track_end(cam: str, r, t_now: float) -> dict[str, Any]:
    """Last observed motion of a moving object up to t_now: end time, box, heading, exit edge."""
    tr = data.tracks(cam)
    g = tr[(tr.tid == r.id) & (tr.t <= t_now)]
    last = g.iloc[-1]
    t_end = float(last.t)
    tail = g[g.t >= t_end - 1.5]
    if len(tail) < 2:
        tail = g.iloc[-2:]
    dx, dy = float(tail.cx.iat[-1] - tail.cx.iat[0]), float(tail.cy.iat[-1] - tail.cy.iat[0])
    if abs(dx) < 3 and abs(dy) < 3:
        heading = "nowhere (standing still)"
    else:
        heading = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else (
            "down (toward the camera)" if dy > 0 else "up (away from the camera)")
    on = float(r.last_t) > t_now + 0.25 or (t_now - t_end <= 0.5 and float(r.last_t) >= t_now)
    return {"t_end": t_end, "box": [float(last.x1), float(last.y1), float(last.x2), float(last.y2)],
            "t_start": float(g.t.iat[0]),
            "box0": [float(g.x1.iat[0]), float(g.y1.iat[0]), float(g.x2.iat[0]), float(g.y2.iat[0])],
            "heading": heading, "edge": exit_edge(cam, last, dx, dy), "on_screen": on,
            "blind": unseen_between(cam, t_end - 0.5, t_end + BLIND_PAD)}


@lru_cache(maxsize=24)
def _frame(cam: str, t: float) -> np.ndarray:
    return data.frame_at(cam, t)


def look_same(cam: str, box: list[float], ta: float, tb: float, tag: str | None = None) -> tuple[float, str | None]:
    """Grey-level correlation of the box region at two times (1 = identical picture, ~0 = something else there).
    Calibrated on ptz_parking: same parked car 0.70-0.99, car gone 0.08. Optionally saves the tb patch."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    try:
        a = _frame(cam, round(ta, 2))[max(0, y1):y2, max(0, x1):x2]
        b = _frame(cam, round(tb, 2))[max(0, y1):y2, max(0, x1):x2]
    except ValueError:
        return float("nan"), None
    if a.size == 0 or b.size == 0:
        return float("nan"), None
    ga, gb = [cv2.cvtColor(cv2.resize(x, (32, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
              for x in (a, b)]
    za, zb = (ga - ga.mean()) / (ga.std() + 1e-6), (gb - gb.mean()) / (gb.std() + 1e-6)
    rel = None
    if tag:
        d = data.out_dir(cam) / CROPS
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{tag}_{tb:.1f}.jpg"), b, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rel = f"{CROPS}/{tag}_{tb:.1f}.jpg"
    return round(float((za * zb).mean()), 3), rel


SAME_LOOK, DIFF_LOOK = 0.6, 0.3


def _status(cam: str, r, t_now: float) -> dict[str, Any]:
    """Where things stand for a parked (static) target at t_now, judged only inside its own camera view."""
    box = _box(r)
    view = int(r.view)
    names = data.VEHICLES if r.name in data.VEHICLES else (r.name,)
    v_now = view_at(cam, t_now)
    blind_now = blind_between(cam, t_now - 0.25, t_now + 0.25)
    st: dict[str, Any] = {"box": box, "view": view, "v_now": v_now, "blind_now": blind_now}
    t_last, tid_last, bridged = chain(cam, box, view, float(r.first_t), t_now, names)
    if t_last < 0:
        st["state"] = "none"
        return st
    st.update(t_last=t_last, tid_last=tid_last, bridged=bridged)
    if v_now == view and not blind_now and t_now - t_last <= 2.0:
        st["state"] = "present"
        return st
    seen = seen_time(cam, view, t_last, t_now)
    st["seen_after"] = seen
    if seen < 2.0:
        st["state"] = "unknown_away"      # the spot was never seen again (camera elsewhere / blind)
        st["blind_after"] = blind_between(cam, t_last, max(t_now, t_last + 0.5))
        return st
    # the detector can lose a parked car (occlusion, low confidence): look at the spot itself before saying it left
    idx = index(cam)
    T = idx["_t"]
    ok = (T > t_last) & (T <= t_now) & (idx["_v"] == view) & ~idx["_blind"]
    t_chk = float(T[ok].max())
    sim, patch = look_same(cam, box, t_last, t_chk, tag=f"spot{int(r.id)}")
    st.update(t_check=t_chk, look=sim, patch=patch)
    if sim == sim and sim >= SAME_LOOK:
        st["state"] = "present_undetected"
        return st
    if (sim == sim and sim > DIFF_LOOK) or _ir(cam, t_last) != _ir(cam, t_chk) or sim != sim:
        # partly matching, or the picture switched between colour and infrared (pixels can't be compared),
        # or the frames could not be read: never call that a departure
        st["state"] = "unclear"
        st["ir"] = _ir(cam, t_last) != _ir(cam, t_chk)
        return st
    after = unseen_between(cam, t_last - 0.25, t_last + BLIND_PAD)
    if after:
        st["state"] = "unknown_gap"       # empty spot seen later, but the departure fell in blind time
        st["blind_after"] = after
        st["by"] = first_seen_after(cam, view, max(x["end"] for x in after), t_now) or t_now
        return st
    st["state"] = "left"
    return st


# ----------------------------------------------------------------------------- answerers

def _ev(cam: str, oid: int, t: float, box, crop: str | None = None) -> dict[str, Any]:
    return {"object_id": int(oid), "t": r2(t), "crop": crop if crop is not None else _crop_path(cam, int(oid)),
            "box": [r2(x) for x in box] if box is not None else None}


def _label(r) -> str:
    """'red car #21'; people are never described by how they look: 'person #3370'."""
    if r.name not in data.VEHICLES:
        return f"{r.name} #{int(r.id)}"
    col = r.colour if isinstance(r.colour, str) else None
    return " ".join(x for x in (col, r.name) if x) + f" #{int(r.id)}"


def _res(answer: str, sentence: str, evidence: list[dict[str, Any]], honesty: str | None,
         last_seen: float | None = None, blind: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"answer": answer, "sentence": sentence, "evidence": evidence, "honesty": honesty,
            "last_seen": r2(last_seen) if last_seen is not None else None, "blind": blind or []}


def _mover_status(cam: str, r, t_now: float) -> dict[str, Any]:
    e = track_end(cam, r, t_now)
    if e["on_screen"]:
        e["state"] = "present"
    elif e["edge"]:
        e["state"] = "left"               # it walked/drove out through a frame edge: that exit was seen
    elif e["blind"]:
        e["state"] = "unknown_gap"
    else:
        e["state"] = "lost"               # vanished mid-frame with the camera fine: occlusion or a missed detection
    return e


def _unclear_why(st: dict[str, Any]) -> str:
    if st.get("ir"):
        return "after that the picture switched between colour and infrared, so its spot can't be compared by its pixels"
    if st.get("look") != st.get("look"):
        return "the spot's picture could not be checked on this machine"
    return f"the spot only partly matches later (picture match {st['look']:.2f}): maybe hidden behind something"


def a_still_there(cam: str, p, t_now: float) -> dict[str, Any]:
    r, why = pick(cam, p, t_now, "parked")
    if r is None:
        return _res("UNKNOWN", why[0].upper() + why[1:] + ".", [], why)
    lab = _label(r)
    if r.moved:
        e = _mover_status(cam, r, t_now)
        ev = [_ev(cam, r.id, e["t_start"], e["box0"]), _ev(cam, r.id, e["t_end"], e["box"])]
        if e["state"] == "present":
            return _res("YES", f"Yes. The {lab} is still in view at {mmss(t_now)}, moving {e['heading']}.", ev, None,
                        e["t_end"])
        if e["state"] == "left":
            return _res("NO", f"No. The {lab} left the picture through the {e['edge']} edge at {mmss(e['t_end'])}.",
                        ev, None, e["t_end"])
        if e["state"] == "unknown_gap":
            bl = describe_blind(e["blind"])
            return _res("UNKNOWN", f"Can't tell. The {lab} was last seen at {mmss(e['t_end'])}, just before the camera "
                        f"could not see ({bl}).", ev, f"the camera could not see when it disappeared: {bl}",
                        e["t_end"], blind_runs(e["blind"]))
        return _res("NO", f"Not in view: the {lab} was last tracked mid-frame at {mmss(e['t_end'])}.", ev,
                    "it vanished away from the frame edges, so it may be hidden behind something rather than gone",
                    e["t_end"])
    st = _status(cam, r, t_now)
    ev = [_ev(cam, r.id, r.first_t, st["box"])]
    if st["state"] == "none":
        return _res("UNKNOWN", f"The {lab} was never re-detected at its spot.", ev,
                    "no detections at that box in the same camera view")
    ev.append(_ev(cam, st["tid_last"], st["t_last"], st["box"]))
    t_last = st["t_last"]
    bridged = st.get("bridged") or []
    if st["state"] == "present":
        hon = []
        if st["tid_last"] != int(r.id):
            hon.append(f"tracker id changed to #{st['tid_last']}; matched by the same parking box (IoU >= {PRESENT_IOU})")
        if bridged:
            hon.append(f"the camera did not see the spot during {', '.join(span(a, b) for a, b in bridged[:3])}; "
                       "a look-alike swap then can't be ruled out")
        return _res("YES", f"Yes. The {lab} is still in its spot at {mmss(t_now)} (detected there at {mmss(t_last)}).",
                    ev, "; ".join(hon) or None, t_last)
    if st.get("patch"):
        ev.append(_ev(cam, r.id, st["t_check"], st["box"], crop=st["patch"]))
    if st["state"] == "present_undetected":
        return _res("YES", f"Yes, most likely. The detector lost the {lab} at {mmss(t_last)}, but its spot at "
                    f"{mmss(st['t_check'])} looks the same as when it was detected (picture match {st['look']:.2f}).",
                    ev, "not detected at the end; judged by comparing the spot's pixels, not by the detector", t_last)
    if st["state"] == "unclear":
        return _res("UNKNOWN", f"Can't tell. The detector lost the {lab} at {mmss(t_last)} and " + _unclear_why(st) + ".",
                    ev, "spot partly changed; could be occlusion, a person in front, or a different car", t_last)
    if st["state"] == "unknown_away":
        runs = blind_runs(blind_between(cam, t_last, max(t_now, t_last + 0.5)), t_last, max(t_now, t_last + 0.5))
        bl = describe_blind(runs)
        other = st["v_now"] != st["view"]
        tail = (f", then it pointed at a different view until {mmss(t_now)}"
                if other and runs and t_now - runs[-1]["end"] > 2.0 else "")
        return _res("UNKNOWN", f"Can't tell. The {lab} was last seen at {mmss(t_last)}. After that the camera never "
                    f"showed its spot again" + (f" ({bl}{tail})." if bl else f" before {mmss(t_now)}."), ev,
                    f"From {mmss(t_last)} to {mmss(t_now)} the camera was pointed elsewhere or could not see, and it "
                    "never answers 'gone' for time it could not see.", t_last, runs)
    if st["state"] == "unknown_gap":
        runs = blind_runs(st["blind_after"])
        bl = describe_blind(runs)
        return _res("NO", f"No. The {lab} was last seen at {mmss(t_last)} and its spot is seen empty from "
                    f"{mmss(st['by'])}; it left while the camera could not see ({bl}).", ev,
                    f"the departure itself was not seen ({bl})", t_last, runs)
    return _res("NO", f"No. The {lab} was last seen at {mmss(t_last)}; its spot stayed in view for "
                f"{st['seen_after']:.0f} s after and now looks different (picture match {st['look']:.2f}).", ev, None,
                t_last)


def a_when_left(cam: str, p, t_now: float) -> dict[str, Any]:
    r, why = pick(cam, p, t_now, "parked")
    if r is None:
        return _res("UNKNOWN", why[0].upper() + why[1:] + ".", [], why)
    lab = _label(r)
    if r.moved:
        e = _mover_status(cam, r, t_now)
        ev = [_ev(cam, r.id, e["t_start"], e["box0"]), _ev(cam, r.id, e["t_end"], e["box"])]
        if e["state"] == "present":
            return _res("still there", f"It hasn't left: the {lab} is still in view at {mmss(t_now)}.", ev, None,
                        e["t_end"])
        if e["state"] == "left":
            return _res(mmss(e["t_end"]), f"The {lab} left the picture through the {e['edge']} edge at "
                        f"{mmss(e['t_end'])}.", ev, None, e["t_end"])
        if e["state"] == "unknown_gap":
            bl = describe_blind(e["blind"])
            return _res("UNKNOWN", f"Some time after {mmss(e['t_end'])}: the {lab} was last seen then, right before "
                        f"the camera could not see ({bl}).", ev, f"the camera could not see when it disappeared: {bl}",
                        e["t_end"], blind_runs(e["blind"]))
        return _res(mmss(e["t_end"]), f"The {lab} was last tracked at {mmss(e['t_end'])}, mid-frame.", ev,
                    "it vanished away from the frame edges (maybe hidden behind something), so this is the last "
                    "sighting, not a seen exit", e["t_end"])
    st = _status(cam, r, t_now)
    ev = [_ev(cam, r.id, r.first_t, st["box"])]
    if st["state"] == "none":
        return _res("UNKNOWN", f"No sightings of the {lab} at its spot.", ev, "no detections at that box")
    ev.append(_ev(cam, st["tid_last"], st["t_last"], st["box"]))
    t_last = st["t_last"]
    if st["state"] == "present":
        return _res("still there", f"It hasn't left: the {lab} is still in its spot at {mmss(t_now)}.", ev, None,
                    t_last)
    if st.get("patch"):
        ev.append(_ev(cam, r.id, st["t_check"], st["box"], crop=st["patch"]))
    if st["state"] == "present_undetected":
        return _res("still there", f"It probably hasn't left: the detector lost the {lab} at {mmss(t_last)} but its "
                    f"spot still looks the same at {mmss(st['t_check'])} (picture match {st['look']:.2f}).", ev,
                    "judged by comparing the spot's pixels, not by the detector", t_last)
    if st["state"] == "unclear":
        return _res("UNKNOWN", f"Can't tell. The {lab} was last detected at {mmss(t_last)} and " + _unclear_why(st) + ".",
                    ev, "spot partly changed; could be occlusion or a different car", t_last)
    if st["state"] == "left":
        return _res(mmss(t_last), f"The {lab} left at about {mmss(t_last)}: its last detection in its spot, which "
                    f"looks different afterwards (picture match {st['look']:.2f}, {st['seen_after']:.0f} s of "
                    "visible spot after).", ev, None, t_last)
    if st["state"] == "unknown_gap":
        runs = blind_runs(st["blind_after"])
        bl = describe_blind(runs)
        by = st["by"]
        return _res("UNKNOWN", f"Between {mmss(t_last)} and {mmss(by)}: it is in its spot until then and the spot is "
                    f"empty after, but the camera could not see in between ({bl}).", ev,
                    f"it left while the camera could not see ({bl}); the exact moment is unknown", t_last, runs)
    runs = blind_runs(blind_between(cam, t_last, max(t_now, t_last + 0.5)), t_last, max(t_now, t_last + 0.5))
    bl = describe_blind(runs)
    return _res("UNKNOWN", f"Can't tell. The {lab} was still there at {mmss(t_last)}; the camera never showed its "
                f"spot again before {mmss(t_now)}" + (f" ({bl})." if bl else "."), ev,
                "no visible time after the last sighting; it may not have left at all", t_last, runs)


def a_which_way(cam: str, p, t_now: float) -> dict[str, Any]:
    if p["names"] is None and p["object_id"] is None:
        p = dict(p, names=("person",) if (objs(cam).name == "person").any() else None)
    r, why = pick(cam, p, t_now, "mover")
    if r is None:
        return _res("UNKNOWN", why[0].upper() + why[1:] + ".", [], why)
    lab = _label(r)
    e = _mover_status(cam, r, t_now)
    ev = [_ev(cam, r.id, e["t_start"], e["box0"]), _ev(cam, r.id, e["t_end"], e["box"])]
    word = e["heading"].split(" ")[0]
    if e["state"] == "present":
        return _res(word, f"The {lab} is still in view at {mmss(t_now)}, moving {e['heading']}.", ev, None, e["t_end"])
    if e["state"] == "left":
        return _res(e["edge"], f"The {lab} left through the {e['edge']} edge of the picture at {mmss(e['t_end'])}, "
                    f"heading {e['heading']}.", ev, None, e["t_end"])
    if e["state"] == "unknown_gap":
        bl = describe_blind(e["blind"])
        return _res("UNKNOWN", f"The {lab} was heading {e['heading']} when the camera lost the view at "
                    f"{mmss(e['t_end'])} ({bl}), so its exit is not seen.", ev, f"track ends in blind time: {bl}",
                    e["t_end"], blind_runs(e["blind"]))
    return _res(word, f"The {lab} was last tracked mid-frame at {mmss(e['t_end'])} heading {e['heading']}; it did not "
                "reach a frame edge.", ev,
                "track ended away from the frame edge (hidden behind something or missed by the detector)", e["t_end"])


def a_who_near(cam: str, p, t_now: float) -> dict[str, Any]:
    r, why = pick(cam, p, t_now, "parked")
    if r is None:
        return _res("UNKNOWN", why[0].upper() + why[1:] + ".", [], why)
    lab = _label(r)
    box = np.array(_box(r), float)
    t0, t1 = float(r.first_t), min(float(r.last_t), t_now)
    d = dets(cam)
    o = objs(cam)
    movers = [int(i) for i in o.id[o.moved] if int(i) != int(r.id)]   # parked neighbours are not "visitors"
    m = (d["t"] >= t0) & (d["t"] <= t1) & np.isin(d["tid"], movers)
    idx = np.flatnonzero(m)
    idx = idx[views_of(cam, d["t"][idx]) == int(r.view)] if len(idx) else idx
    hb = box[3] - box[1]
    gaps = gap_many(box, d["box"][idx]) if len(idx) else np.zeros(0)
    rel = gaps / max(hb, 1.0)
    close = idx[rel <= NEAR_REL]
    vis = []
    for tid in np.unique(d["tid"][close]):
        k = close[d["tid"][close] == tid]
        kk = k[np.argmin(gap_many(box, d["box"][k]))]
        vis.append({"tid": int(tid), "name": str(d["name"][kk]), "t": float(d["t"][kk]),
                    "gap": float(gap_many(box, d["box"][kk:kk + 1])[0]),
                    "enter": float(d["t"][k].min()), "exit": float(d["t"][k].max()), "box": d["box"][kk]})
    vis.sort(key=lambda v: v["t"])
    runs = blind_runs(blind_between(cam, t0, t1), t0, t1)
    bs = blind_seconds(cam, t0, t1)
    hon = ["close means within half a car-height in the picture, so it may have been in front or behind"]
    if bs >= 0.5:
        hon.append(f"the camera could not see for {bs:.0f} s of the {t1 - t0:.0f} s watch window "
                   f"({describe_blind(runs)}); anyone near then is missed")
    ev = [_ev(cam, r.id, t0, box)] + [_ev(cam, v["tid"], v["t"], v["box"]) for v in vis[:6]]

    def how(v: dict[str, Any]) -> str:
        passing = v["name"] in data.VEHICLES and v["exit"] - v["enter"] <= 4.0
        return f"{v['name']} #{v['tid']} {'drove past' if passing else 'came close'} at {mmss(v['t'])}"
    if not vis:
        return _res("0", f"Nothing moving came within half a car-height of the {lab} between {mmss(t0)} and "
                    f"{mmss(t1)}.", ev, "; ".join(hon), None, runs)
    parts = ", ".join(how(v) for v in vis[:5])
    more = f" and {len(vis) - 5} more" if len(vis) > 5 else ""
    return _res(str(len(vis)), f"{len(vis)} came near the {lab} ({span(t0, t1)}): {parts}{more}. "
                "This says who was close, not who did anything.", ev, "; ".join(hon), None, runs)


def _window(cam: str, p, t_now: float) -> tuple[float, float, str | None]:
    """(start, end, day label or None) for count/change questions."""
    if p["since"] is not None:
        return float(p["since"]), t_now, None
    if p["last"] is not None:
        return max(0.0, t_now - p["last"]), t_now, None
    refs = p.get("days_idx") or []
    if refs:
        ds = days_of(cam)
        a, b = ds[min(refs)], ds[max(refs)]
        if p.get("since_day"):
            return a["start"], t_now, f"since {a['label']}"
        lab = a["label"] if a is b else f"{a['label']} to {b['label']}"
        return a["start"], min(t_now, b["end"]), f"on {lab}"
    return 0.0, t_now, None


@lru_cache(maxsize=None)
def _track_ends(cam: str) -> dict[int, tuple[float, float, float, float]]:
    """tid -> (first cx, first cy, last cx, last cy), computed once per camera (tracks are sorted by t)."""
    g = data.tracks(cam).groupby("tid")
    f, l = g[["cx", "cy"]].first(), g[["cx", "cy"]].last()
    return {int(t): (float(f.cx[t]), float(f.cy[t]), float(l.cx[t]), float(l.cy[t])) for t in f.index}


def a_count_since(cam: str, p, t_now: float) -> dict[str, Any]:
    since, until, daylab = _window(cam, p, t_now)
    o = objs(cam)
    names = p["names"] or tuple(data.MOVERS)
    what = (p["names"] or ["object"])[0]
    plural = {"person": "people", "bus": "buses"}.get(what, what + "s")
    if o.empty:
        return _res("0", f"No {plural} tracked.", [], None)
    c = o[o.name.isin(names) & o.moved & (o.first_t >= since) & (o.first_t <= until)].sort_values("first_t")
    if p["colour"] and what in data.VEHICLES:
        c = c[c.colour == p["colour"]]
    # births right after a cut / camera move are things coming into view, not arrivals
    bl = index(cam)["blind"] + index(cam)["view_changes"]

    def after_blind(t: float) -> bool:
        return any(b["start"] - 0.25 <= t <= b["end"] + 1.0 for b in bl)
    reborn = c[[after_blind(t) for t in c.first_t]]
    c = c[[not after_blind(t) for t in c.first_t]]
    # stitch fragments: a same-class track that starts within 2 s of another ending, close by, is the same thing
    ends = _track_ends(cam)
    chains: list[list[Any]] = []
    for r in c.itertuples():
        sx, sy, ex, ey = ends[int(r.id)]
        start = (sx, sy)
        item = {"id": r.id, "name": r.name, "last_t": r.last_t, "end": (ex, ey),
                "bw": r.bw, "first_t": r.first_t, "box": _box(r)}
        for ch in chains:
            e = ch[-1]
            if e["name"] == r.name and 0 <= r.first_t - e["last_t"] <= 2.0 and \
                    np.hypot(start[0] - e["end"][0], start[1] - e["end"][1]) <= 1.5 * max(r.bw, e["bw"]):
                ch.append(item)
                break
        else:
            chains.append([item])
    n = len(chains)
    whole = since <= 0.5 and until >= duration(cam) - 0.5
    tb = index(cam).get("blind_total_s")
    bs = tb if (whole and tb is not None) else blind_seconds(cam, since, until)
    runs = blind_runs(blind_between(cam, since, until), since, until)
    notes = []
    if bs >= 0.5:
        notes.append(f"the camera could not see for {bs:.0f} s of {until - since:.0f} s ({describe_blind(runs)}), "
                     "so this is a lower bound")
    if len(reborn):
        notes.append(f"{len(reborn)} track(s) that only appeared because the view changed were not counted")
    frag = sum(len(ch) - 1 for ch in chains)
    if frag:
        notes.append(f"{frag} broken track fragment(s) merged")
    col = f"{p['colour']} " if p["colour"] and what in data.VEHICLES else ""
    ev = [_ev(cam, ch[0]["id"], ch[0]["first_t"], ch[0]["box"]) for ch in chains[:8]]
    when = f"{daylab} ({span(since, until)})" if daylab else f"between {mmss(since)} and {mmss(until)}"
    return _res(str(n), f"{n} moving {col}{plural if n != 1 else what} came into view {when}.", ev,
                "; ".join(notes) or None, None, runs)


def _parked_boxes(cam: str, view: int, t0: float, t1: float) -> list[Any]:
    o = objs(cam)
    if o.empty:
        return []
    c = o[(o.view == view) & (o.name.isin(data.VEHICLES)) & ~o.moved & (o.n >= 45) &
          (o.last_t >= t0) & (o.first_t <= t1)].sort_values("n", ascending=False)
    keep: list[Any] = []
    for r in c.itertuples():
        b = np.array(_box(r), float)
        if all(iou_many(b, np.array([_box(k)], float))[0] < 0.5 for k in keep):
            keep.append(r)
    return keep


def _diff_answer(cam: str, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """Day-to-day answer straight from diff.json (the same numbers the Diff view shows), when it covers these days
    and its alignment is ok. None = fall back to the in-video comparison."""
    D = _json_file(cam, "diff.json")
    if not isinstance(D, dict) or "counts" not in D:
        return None
    if {str(D.get("day_a", a["id"])), str(D.get("day_b", b["id"]))} != {a["id"], b["id"]}:
        return None
    al = D.get("alignment")
    if isinstance(al, dict) and not al.get("ok", True):
        return None
    ch = D.get("changes") or []
    cnt = D.get("counts") or {}
    n = {k: int(cnt.get(k, sum(1 for c in ch if c.get("kind") == k)))
         for k in ("left", "arrived", "swapped", "moved", "same")}
    words = {"left": "left", "arrived": "arrived", "swapped": "a different car", "moved": "moved"}
    moves = [f"{n[k]} {w}" for k, w in words.items() if n[k]]
    ans = ", ".join(moves) if moves else "no change"
    unconf = sum(1 for c in ch if c.get("kind") == "same" and c.get("unconfirmed"))
    sure = max(0, n["same"] - unconf)
    parts = moves + ([f"{sure} the same car"] if sure or not unconf else []) +         ([f"{unconf} still taken by a vehicle (not confirmed as the same car: "
          + ("night infrared, no colour)" if D.get("infrared") and any(D["infrared"].values()) else "can't compare looks)")]
         if unconf else [])
    nc = D.get("not_compared") or []
    why = {"outside the view both days shared": 0, "only partly visible": 0, "probably missed by the detector": 0,
           "seen too briefly to count as parked": 0}
    other = 0
    for x in nc:
        rs = str(x.get("reason") or "")
        if "outside" in rs or "cut off" in rs:
            why["outside the view both days shared"] += 1
        elif "partly visible" in rs:
            why["only partly visible"] += 1
        elif "detector miss" in rs or "missed" in rs:
            why["probably missed by the detector"] += 1
        elif "not long enough" in rs or "briefly" in rs:
            why["seen too briefly to count as parked"] += 1
        else:
            other += 1
    nc_txt = ""
    if nc:
        bits = [f"{v} {k}" for k, v in why.items() if v] + ([f"{other} other"] if other else [])
        nc_txt = f"; {len(nc)} not compared ({', '.join(bits)})"
    la, lb = a["label"], b["label"]
    sentence = f"{la} vs {lb}: {', '.join(parts)}{nc_txt}."
    ev = []
    for c in ch:
        if c.get("kind") == "same" or len(ev) >= 8:
            continue
        for side, bkey in (("a", "box"), ("b", "box_b")):
            x = c.get(side)
            if not isinstance(x, dict) or x.get("object_id") is None:
                continue
            crop = x.get("crop")
            crop = crop if crop and (data.OUT / cam / crop).exists() else None
            bx = c.get(bkey) if side == "b" else c.get("box")
            ev.append({"object_id": int(x["object_id"]), "t": r2(min(max(0.0, float(x.get("t") or 0.0)), duration(cam))),
                       "crop": crop, "box": [r2(v) for v in bx] if bx and bx[2] > bx[0] and bx[3] > bx[1] else None})
    hon = []
    if isinstance(al, dict) and al.get("overlap_frac") is not None and float(al.get("overlap_frac", 1.0)) < 0.99:
        hon.append(f"the camera framing differs between the days; {lb} was lined up onto {la} "
                   f"({al.get('inliers', '?')} matching points) and only the {float(al['overlap_frac']) * 100:.0f}% "
                   "of the picture seen on both days is compared")
    gap0 = float((al or {}).get("view_a", {}).get("end", a["end"])) if isinstance(al, dict) else a["end"]
    gap1 = float((al or {}).get("view_b", {}).get("start", b["start"])) if isinstance(al, dict) else b["start"]
    runs = blind_runs(blind_between(cam, gap0, gap1), gap0, gap1) if gap1 > gap0 else []
    if runs:
        hon.append(f"between the two views the camera could not see ({describe_blind(runs)})")
    sim = list(_meta(cam).get("simulated") or [])
    if sim:
        hon.append("SIMULATED: " + sim[0])
    return _res(ans, sentence, ev, "; ".join(hon) or None, None, runs)


def a_what_changed(cam: str, p, t_now: float) -> dict[str, Any]:
    ds = days_of(cam)
    refs = p.get("days_idx") or []
    pair = None
    if len(ds) > 1 and p["since"] is None and p["last"] is None:
        if len(refs) >= 2:
            pair = (min(refs), max(refs))
        elif len(refs) == 1:
            k = refs[0]
            if p.get("since_day") and k < len(ds) - 1:
                pair = (k, len(ds) - 1)
            elif k > 0:
                pair = (k - 1, k)
        else:
            pair = (0, len(ds) - 1)
    if pair is not None:
        a, b = ds[pair[0]], ds[pair[1]]
        if t_now <= b["start"]:
            return _res("UNKNOWN", f"{b['label']} hasn't started at {mmss(t_now)}.", [], "asked before that day")
        hit = _diff_answer(cam, a, b)
        if hit is not None:
            return hit
        ta, tb = a["end"] - 3.0, min(t_now, b["end"]) - 1.0
        # compare the last stable view of the first day with the view at the end of the second
        va, vb = view_at(cam, ta), view_at(cam, tb)
        for k in range(30):
            if va >= 0:
                break
            va = view_at(cam, ta - k)
        if va != vb or va < 0:
            runs = blind_runs(blind_between(cam, a["start"], tb), a["start"], tb)
            return _res("UNKNOWN", f"Can't compare {a['label']} with {b['label']}: the camera framing is different, "
                        "so the parking spots don't line up.", [],
                        f"camera moved between the days ({describe_blind(runs)}); a box-by-box comparison would "
                        "invent arrivals and departures", None, runs)
        t0, t1, label = ta, tb, f"{a['label']} vs {b['label']}"
    else:
        t0, t1, daylab = _window(cam, p, t_now)
        label = f"{mmss(t0)} vs {mmss(t1)}"
    v1 = view_at(cam, t1)
    v0 = view_at(cam, t0)
    if v0 != v1 or v1 < 0:
        return _res("UNKNOWN", f"Can't compare {label}: the camera shows a different view at the two times.", [],
                    f"the camera pointed somewhere else at {mmss(t1)} than at {mmss(t0)}")
    arrived, left, same, unknown, missed, unclear = [], [], [], [], [], []
    looks: dict[int, float] = {}
    for r in _parked_boxes(cam, v1, t0, t1):
        box = _box(r)
        names = data.VEHICLES
        at0, _ = presence(cam, box, v1, t0 - 2.0, t0 + 2.0, names)
        at1, _ = presence(cam, box, v1, t1 - 2.0, t1 + 0.5, names)
        if len(at0) and len(at1):
            same.append(r)
        elif len(at0) or len(at1):
            # the detector misses cars behind other cars: only call it a change if the spot's pixels changed too
            sim, _ = look_same(cam, box, t0, t1)
            if sim == sim and sim >= SAME_LOOK:
                same.append(r); missed.append(r)
            elif sim == sim and sim > DIFF_LOOK:
                unclear.append(r)
            else:
                (arrived if len(at1) else left).append(r)
                looks[int(r.id)] = sim
        else:
            unknown.append(r)
    runs = blind_runs(blind_between(cam, t0, t1), t0, t1)
    if not (arrived or left or same):
        o = objs(cam)
        moved = int((o.moved & (o.first_t >= t0) & (o.first_t <= t1)).sum()) if len(o) else 0
        return _res("no parked changes", f"Nothing parked here ({label}); {moved} moving object(s) passed through.",
                    [], None, None, runs)
    ev = [_ev(cam, r.id, r.first_t, _box(r)) for r in (arrived + left)[:8]]
    txt = f"{len(arrived)} arrived, {len(left)} left, {len(same)} stayed"
    detail = "; ".join([f"arrived: {', '.join(_label(r) for r in arrived[:4])}" if arrived else "",
                        f"left: {', '.join(_label(r) for r in left[:4])}" if left else ""]).strip("; ")
    hon = []
    if runs:
        hon.append(f"the camera could not see for part of the time in between ({describe_blind(runs)}); states "
                   "before and after are seen, the moments of change may not be")
    if missed:
        hon.append(f"{len(missed)} spot(s) the detector missed at one end look unchanged in the picture, counted as stayed")
    if unclear:
        hon.append(f"{len(unclear)} spot(s) partly changed (occlusion?), not counted")
    if unknown:
        hon.append(f"{len(unknown)} spot(s) undetected at both times")
    if looks:
        hon.append("changes confirmed by picture match " + ", ".join(f"#{k} {v:.2f}" for k, v in looks.items()))
    return _res(txt, f"{label}: {txt}." + (f" {detail}." if detail else ""), ev, "; ".join(hon) or None, None, runs)


ANSWERERS = {"still_there": a_still_there, "when_left": a_when_left, "which_way": a_which_way,
             "who_near": a_who_near, "count_since": a_count_since, "what_changed": a_what_changed}


def answer(cam: str, question: str, t_now: float | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    if cam not in data.CAMS:
        raise KeyError(f"unknown camera {cam}")
    p = parse(question)
    dur = duration(cam)
    p["days_idx"] = resolve_days(cam, p)
    if p["at"] is not None:
        t_now = p["at"]
    elif t_now is None and p["days_idx"] and p["type"] not in ("what_changed", "count_since"):
        t_now = days_of(cam)[p["days_idx"][-1]]["end"]     # "... yesterday": as of the end of that day
    t_now = dur if t_now is None else float(min(max(0.0, t_now), dur))
    refused = None
    looks_person = p["colour"] and p["names"] in (("person",), ("bicycle",)) and p["type"] != "count_since"
    if p["appearance"] or looks_person:
        refused = "appearance"
        res = _res("UNKNOWN", REFUSE_LOOKS, [], "by design")
    elif p["type"] is None:
        res = _res("UNKNOWN", "I can answer: is X still there, when did X leave, which way did X go, who came near X, "
                   "how many X since a time or day, what changed between the days.", [], "question not understood")
    else:
        res = ANSWERERS[p["type"]](cam, p, t_now)
        if p["blame"]:
            refused = "blame"
            near = res["sentence"].replace(" This says who was close, not who did anything.", "")
            res = dict(res, answer="UNKNOWN", sentence="It can't say who did it. It shows who was near: " + near,
                       honesty="by design: being near is not doing it; it shows who was near, never who did it"
                       + (f"; {res['honesty']}" if res["honesty"] else ""))
        if p["gender"] and p["names"] == ("person",):
            note = "people are not told apart by gender or looks: this means any person"
            res = dict(res, honesty=f"{res['honesty']}; {note}" if res["honesty"] else note)
    out = {"type": p["type"] or "unknown", "answer": str(res["answer"]), "sentence": res["sentence"],
           "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2), "evidence": res["evidence"],
           "honesty": res["honesty"], "last_seen": res.get("last_seen"), "blind": res.get("blind") or [],
           "refused": refused}
    return out


# ----------------------------------------------------------------------------- demo questions + CLI

def _phrase(cam: str, r, t_now: float, typ: str) -> str:
    """'the red car' when that phrase resolves to this object, else 'car #12'."""
    noun = "car" if r.name in ("car", "truck") else r.name
    if isinstance(r.colour, str):
        p = parse(f"{typ} the {r.colour} {noun}")
        p["type"] = typ
        got, _ = pick(cam, p, t_now, "parked" if typ != "which_way" else "mover")
        if got is not None and int(got.id) == int(r.id):
            return f"the {r.colour} {noun}"
    return f"{noun} #{int(r.id)}"


CHROMA_RANK = {"red": 0, "blue": 1, "yellow": 2, "green": 3, "orange": 4, "purple": 5, "white": 6, "black": 7, "grey": 8}


def _demo_target(cam: str) -> tuple[Any, str] | None:
    """A parked vehicle with the most distinctive colour whose phrase ('the red car') resolves back to it."""
    o = objs(cam)
    if o.empty:
        return None
    dur = duration(cam)
    veh = o[o.name.isin(data.VEHICLES)]
    longv = veh[(veh.parked | (veh.n >= 45)) & ~veh.moved]
    pool = longv if len(longv) else veh
    if pool.empty:
        return None
    rank = [CHROMA_RANK.get(c, 9) if isinstance(c, str) else 9 for c in pool.colour]
    for r in pool.assign(cr=rank).sort_values(["cr", "n"], ascending=[True, False]).head(12).itertuples(index=False):
        ph = _phrase(cam, r, dur, "who_near")
        if ph.startswith("the "):
            return r, ph
    r = _row(pool.sort_values("n", ascending=False))
    return r, _phrase(cam, r, dur, "who_near")


def demo_questions(cam: str) -> list[tuple[str, float | None]]:
    """8 questions built from what this camera actually contains, in demo order. Day names come from the
    camera's own day labels; colour words are used only when they resolve back to the intended vehicle."""
    o = objs(cam)
    dur = duration(cam)
    ds = days_of(cam)
    two = len(ds) > 1
    has_person = bool(len(o) and (o.name == "person").any())
    qs: list[tuple[str, float | None]] = []
    tgt = _demo_target(cam)
    if tgt:
        r, ph = tgt
        qs.append((f"Who came near {ph}?", None))
        qs.append((f"Is {ph} still there?", None))
        qs.append((f"Who dented {ph}?", None))
    if two:
        qs.append((f"What changed between {ds[0]['label']} and {ds[-1]['label']}?", None))
    else:
        qs.append((f"What changed since {mmss(dur * 0.1)}?", None))
    qs.append(("How many people walked by?", None) if has_person else ("How many cars came by?", None))
    qs.append(("Which way did the person go?", None) if has_person else ("Which way did the car go?", None))
    qs.append((f"How many cars came by since {mmss(dur * 0.25)}?", None))
    qs.append(("Find the man in the black shirt", None))     # the appearance refusal, on purpose
    while len(qs) < 8:
        qs.append(("How many cars came by?", None))
    return qs[:8]


def run(cam: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    for f in (index, objs, dets, _meta, duration, days_of, _track_ends):
        f.cache_clear()
    idx = index(cam)
    qs = demo_questions(cam)
    for q, _ in qs:                 # warm pass: qa.json records warm latencies, as the live server gives them
        answer(cam, q)
    items = []
    for q, _ in qs:
        a = answer(cam, q)          # no t: the question text alone must give this answer (the UI sends only q)
        items.append({"question": q, "t": parse(q)["at"], "type": a["type"], "answer": a})
    total = idx.get("blind_total_s")
    if total is None:
        total = blind_seconds(cam, 0.0, duration(cam))
    res = {"camera": cam, "ledger_used": idx["ledger"], "blind_total_s": r2(total),
           "days": [{"id": d["id"], "label": d["label"]} for d in days_of(cam)],
           "blind": [{k: b[k] for k in ("start", "end", "reason", "source")} for b in idx["blind"]],
           "items": items}
    d = data.out_dir(cam)
    data.write_json(d / "qa.json", res)
    by: dict[str, int] = {}
    for it in items:
        k = it["answer"]["answer"] if it["answer"]["answer"] in ("YES", "NO", "UNKNOWN") else "value"
        by[k] = by.get(k, 0) + 1
    lat = sorted(it["answer"]["latency_ms"] for it in items)
    lines = [f"ask: {cam}", f"objects indexed: {len(idx['colours'])} (colour from best-frame crop)",
             f"camera views found: {idx['n_views']}; blind spans: {len(idx['blind'])} from "
             f"{'ledger.json only' if idx['ledger'] else 'own scan (no ledger.json)'}; blind total {total:.1f} s",
             f"days: {', '.join(d['label'] for d in days_of(cam))}",
             f"qa.json: {len(items)} questions; answers {by}; warm latency p50 {lat[len(lat) // 2]:.0f} ms, "
             f"max {lat[-1]:.0f} ms",
             f"elapsed: {time.perf_counter() - t0:.1f} s"]
    lines += [f"  [{it['type']}] {it['question']} -> {it['answer']['answer']}  ({it['answer']['latency_ms']:.0f} ms)"
              for it in items]
    (d / "ask_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return res


def default_cams() -> list[str]:
    """Demo cams, Pack D first when it is registered on this machine."""
    extra = [c for c in data.CAMS if c not in DEMO_CAMS and c not in ("rochester_night", "yonge_dundas")
             and (data.OUT / c / "meta.json").exists()]
    return extra + list(DEMO_CAMS)


if __name__ == "__main__":
    for c in sys.argv[1:] or default_cams():
        run(c)
