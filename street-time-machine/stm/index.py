"""index: everyone who appeared, with a dominant colour name for vehicles, plus text search.

out/<cam>/index.json (schema in CONTRACTS.md "index") and stm.index.search(cam, "white car") -> [object_id].

People are never described by how they look. Person rows carry `colour: null, rgb: null`; bicycles and
motorcycles too (their box is mostly the rider). search() refuses any query that pairs a person word with a
colour, clothing or appearance word ("man in black", "white people", "red hoodie"), or uses a gender / age word
("woman", "kids"), and sets `last_reason` (= REFUSAL). It shows who was near, never who did it.

Vehicles (car / truck / bus) get a colour only when it is confident: box big enough, not too dark, and neither the
frame it came from nor its day is a night scene. Otherwise `colour: null` and `colour_why` says why ("night",
"dark box", "too small"). A dark car under street lights is never called red.

Crops: crops/<id>.jpg, the frame where the track's box is biggest, padded 15 %, JPEG q85, i.e. exactly what
data.save_crop(cam, id) writes. data.save_crop seeks the video once per crop (~1.6 s each), which would blow the
3-minute budget for 150 objects, so the frames are read in one sequential pass and cut with the same geometry.

Colour: HSV over the body of the box (borders dropped), darkest and brightest 10 % of pixels by V ignored. A
chromatic hue wins when enough of the remaining body is saturated; otherwise the colour is achromatic and named
black / grey / silver / white by brightness.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from stm import data

CAP = 150                       # objects per camera, the ones with most frames
PAD = 0.15                      # same padding as data.crop
COLOURS = ("white", "black", "grey", "silver", "red", "blue", "green", "yellow", "orange", "brown")

# tuned by eye on ptz_parking / yonge_dundas / car_detection crops
S_MIN, V_MIN, C_MIN = 70, 45, 20   # a pixel counts as "painted" (chromatic) above these (C = S*V/255)
CHROM_FRAC = 0.30               # share of body pixels that must be chromatic for a hue name
BLUE_FRAC = 0.45                # blue needs more: sky reflections tint black/grey paint blue
DARK_P95 = 50                   # box p95 brightness below this -> too dark to name a colour reliably
NIGHT_V = 60                    # scene (background plate) median V below this -> night, colours unreliable
NIGHT_DARK_FRAC = 0.35          # frame (object masked out) with this share of pixels V<50 is dark ...
NIGHT_S = 60                    # ... and it is night only if street lighting tints it (median S >= 60). A sunny
                                # car park with dark tarmac is dark but unsaturated (S 18-49 on ptz_parking): day.
                                # rochester_night frames: S 62-83.
IR_S90 = 4                      # 90th-pct saturation below this = a greyscale infrared night frame (any brightness):
                                # the real pack's 2nd day is IR (p90 S = 0); the greyest daytime dev clip has 9.
MIN_AREA = 400                  # box smaller than this (px^2, ~20x20) -> too few pixels to name a colour
COLOURED = ("car", "truck", "bus")   # the only classes that ever get a colour name
RIDERS = ("bicycle", "motorcycle")   # box is mostly the rider: never coloured (it would describe a person)

REFUSAL = "people are not searchable by appearance"
REFUSAL_TEXT = ("Street Time Machine never describes or searches people by how they look: no clothing, skin, "
                "gender or face. Ask about vehicles, or who came near a car.")
last_reason: str | None = None  # why the last search() returned what it did (None = an ordinary result)

# a query pairing a person word with a colour / garment / looks word is refused
PERSON_WORDS = {"person", "persons", "people", "pedestrian", "pedestrians", "someone", "somebody", "anyone",
                "anybody", "individual", "individuals", "human", "humans", "walker", "walkers", "jogger", "joggers",
                "runner", "runners", "cyclist", "cyclists", "rider", "riders", "suspect", "suspects", "stranger",
                "strangers", "thief", "thieves", "intruder", "visitor", "neighbour", "neighbor", "resident",
                "passerby", "passersby", "kid", "kids", "guy", "guys"}
# gender / age words describe how a person looks: refused even on their own
GENDER_AGE_WORDS = {"man", "men", "woman", "women", "guy", "guys", "lady", "ladies", "gentleman", "gentlemen",
                    "boy", "boys", "girl", "girls", "kid", "kids", "child", "children", "teen", "teens",
                    "teenager", "teenagers", "male", "males", "female", "females", "elderly", "toddler", "baby",
                    "young", "youth"}
# garments imply a person: refused with a colour word or a person word
GARMENT_WORDS = {"shirt", "shirts", "tshirt", "jacket", "jackets", "coat", "coats", "hoodie", "hoodies", "jumper",
                 "sweater", "sweatshirt", "hat", "hats", "cap", "beanie", "dress", "skirt", "trousers", "pants",
                 "jeans", "shorts", "shoes", "trainers", "sneakers", "clothes", "clothing", "outfit", "uniform",
                 "vest", "gilet", "tracksuit", "scarf", "gloves", "blouse", "top", "tops", "leggings"}
# other looks words: refused with a person word
LOOKS_WORDS = {"wearing", "wears", "wore", "dressed", "skin", "skinned", "face", "faces", "hair", "haired",
               "blonde", "blond", "brunette", "ginger", "bald", "beard", "bearded", "moustache", "glasses",
               "sunglasses", "masked", "tall", "fat", "thin", "skinny", "race", "ethnicity", "ethnic", "asian",
               "african", "caucasian", "hispanic", "latino", "arab", "tattoo", "tattooed", "pale", "gender"}

# words people type -> canonical names
COLOUR_WORDS = {"white": "white", "black": "black", "grey": "grey", "gray": "grey", "silver": "silver",
                "red": "red", "maroon": "red", "blue": "blue", "navy": "blue", "green": "green",
                "yellow": "yellow", "gold": "yellow", "orange": "orange", "brown": "brown", "beige": "brown",
                "tan": "brown"}
CLASS_WORDS = {"car": ["car"], "van": ["car"], "suv": ["car"], "sedan": ["car"], "taxi": ["car"],
               "truck": ["truck"], "lorry": ["truck"], "pickup": ["truck"], "bus": ["bus"],
               "motorcycle": ["motorcycle"], "motorbike": ["motorcycle"], "scooter": ["motorcycle"],
               "person": ["person"], "people": ["person"], "pedestrian": ["person"], "walker": ["person"],
               "someone": ["person"], "anyone": ["person"],
               "bike": ["bicycle"], "bicycle": ["bicycle"], "cyclist": ["bicycle"],
               "vehicle": list(data.VEHICLES)}
# when a strict colour match finds nothing, these are the honest "close enough" neighbours
NEAR = {"white": ["silver"], "silver": ["grey", "white"], "grey": ["silver", "black"], "black": ["grey"],
        "red": ["orange", "brown"], "orange": ["red", "yellow", "brown"], "yellow": ["orange"],
        "brown": ["orange", "red"], "blue": [], "green": []}


def _hue_name(h: float) -> str:
    """OpenCV hue (0..179) -> colour word."""
    if h < 8 or h >= 160:
        return "red"
    if h < 20:
        return "orange"
    if h < 34:
        return "yellow"
    if h < 92:                  # teal counts as green
        return "green"
    if h < 135:
        return "blue"
    return "red"                # magenta / purple: nearest word in the closed list


def body(box_img: np.ndarray, name: str = "car") -> np.ndarray:
    """The part of a vehicle box that carries its paint colour: borders dropped."""
    h, w = box_img.shape[:2]
    y1, y2, x1, x2 = 0.20, 0.80, 0.12, 0.88
    sub = box_img[int(h * y1):max(int(h * y2), int(h * y1) + 1), int(w * x1):max(int(w * x2), int(w * x1) + 1)]
    return sub if sub.size >= 3 * 16 else box_img


NO_COLOUR = {"colour": None, "rgb": None, "frac": None, "chrom": None, "v": None, "reliable": False}


def colour_of(box_img: np.ndarray, name: str = "car") -> dict[str, Any]:
    """Dominant colour of one vehicle box image (BGR, unpadded). Returns colour, rgb, frac (share of voting
    pixels), chrom (share of chromatic pixels), v (median brightness) and reliable (box not too dark).
    People and riders are never described by how they look: colour and rgb are None for them."""
    if name not in COLOURED:
        return dict(NO_COLOUR)
    sub = body(box_img, name)
    p95 = float(np.percentile(cv2.cvtColor(box_img, cv2.COLOR_BGR2HSV)[..., 2], 95))
    bgr = sub.reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    lo, hi = np.percentile(V, [10, 90])
    keep = (V >= lo) & (V <= hi)
    if keep.sum() < 12:
        keep = np.ones_like(V, dtype=bool)
    H, S, V, bgr = H[keep], S[keep], V[keep], bgr[keep]
    chrom = (S >= S_MIN) & (V >= V_MIN) & (S * V / 255.0 >= C_MIN)
    chrom_frac = float(chrom.mean())
    v_med = float(np.median(V))
    colour = None
    if chrom_frac >= CHROM_FRAC:
        names = np.array([_hue_name(h) for h in H[chrom]])
        vals, cnt = np.unique(names, return_counts=True)
        colour = str(vals[cnt.argmax()])
        vote = np.zeros_like(chrom)
        vote[np.flatnonzero(chrom)[names == colour]] = True
        # dark orange/yellow = brown (true reds stay red even in shadow: maroon is still "red")
        if colour in ("orange", "yellow") and np.median(V[vote]) < 120:
            colour = "brown"
        if colour == "blue" and chrom_frac < BLUE_FRAC:
            colour = None
    if colour is None:
        grey = ~chrom
        vg = V[grey] if grey.any() else V
        # the body of a car is its brightest large surface; glass, tyres and shadow are darker, so use
        # a high percentile of the (already trimmed) brightness rather than the median
        b = float(np.percentile(vg, 80))
        colour = "black" if b < 100 else "grey" if b < 130 else "silver" if b < 172 else "white"
        vote = grey if grey.any() else np.ones_like(chrom)
        # the swatch must show the shade that was named: take it from the same bright band the name came from
        # (~p80), not the median of all grey pixels, which glass / tyres / shadow pull far darker
        swatch = vote & (V >= np.percentile(vg, 60))
    else:
        swatch = vote
    rgb = np.median(bgr[swatch if swatch.any() else vote], axis=0)[::-1]
    return {"colour": colour, "rgb": [int(round(c)) for c in rgb], "frac": round(float(vote.mean()), 3),
            "chrom": round(chrom_frac, 3), "v": round(v_med, 1), "reliable": bool(p95 >= DARK_P95)}


def _box_in_crop(r: Any, W: int, H: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """(crop rect, box rect inside the crop) with data.crop's exact geometry."""
    px, py = r.bw * PAD, r.bh * PAD
    x1, y1 = int(max(0, r.x1 - px)), int(max(0, r.y1 - py))
    x2, y2 = int(min(W, r.x2 + px)), int(min(H, r.y2 + py))
    bx1, by1 = int(max(0, r.x1)) - x1, int(max(0, r.y1)) - y1
    bx2, by2 = int(min(W, r.x2)) - x1, int(min(H, r.y2)) - y1
    return (x1, y1, x2, y2), (bx1, by1, bx2, by2)


def frame_light(img: np.ndarray, box: tuple[float, float, float, float]) -> dict[str, float]:
    """Median brightness and share of dark pixels (V<50) of a full frame with the object's own box masked out,
    so a big dark lorry filling the frame does not make a sunny frame look like night."""
    H, W = img.shape[:2]
    small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
    m = np.ones(small.shape[:2], bool)
    x1, y1, x2, y2 = box
    m[int(max(0.0, y1) * 90 / H):int(np.ceil(min(H, y2) * 90 / H)),
      int(max(0.0, x1) * 160 / W):int(np.ceil(min(W, x2) * 160 / W))] = False
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    V, S = hsv[..., 2], hsv[..., 1]
    if m.sum() < 200:
        m[:] = True
    v = V[m]
    return {"v": float(np.median(v)), "dark": float((v < 50).mean()), "s": float(np.median(S[m])),
            "s90": float(np.percentile(S[m], 90))}


def is_night(light: dict[str, float]) -> bool:
    """Infrared (greyscale) frame, or a dark frame tinted by street lighting."""
    if light.get("s90", 255.0) < IR_S90:
        return True
    dark = light["v"] < NIGHT_V or light["dark"] >= NIGHT_DARK_FRAC
    return bool(dark and light.get("s", NIGHT_S) >= NIGHT_S)


def judge(name: str, col: dict[str, Any], area: float, night: bool) -> str | None:
    """Why a row has no colour (None = it keeps its colour). People and riders never get one."""
    if name == "person":
        return "person"
    if name in RIDERS:
        return "rider"
    if name not in COLOURED or col.get("colour") is None:
        return "class"
    if night:
        return "night"
    if not col["reliable"]:
        return "dark box"
    if area < MIN_AREA:
        return "too small"
    return None


def select(cam: str, cap: int = CAP) -> pd.DataFrame:
    """The `cap` objects with most frames (ties: earlier first)."""
    o = data.objects(cam)
    if o.empty:
        return o
    return o.sort_values(["n", "first_t"], ascending=[False, True]).head(cap).reset_index(drop=True)


def build(cam: str, cap: int = CAP) -> dict[str, Any]:
    t0 = time.time()
    objs = select(cam, cap)
    inf = data.info(cam)
    W, H = inf["w"], inf["h"]
    od = data.out_dir(cam)
    (od / "crops").mkdir(parents=True, exist_ok=True)
    scene_v = float(np.median(cv2.cvtColor(data.background(cam), cv2.COLOR_BGR2HSV)[..., 2]))
    day_lab = {d["id"]: d["label"] for d in data.days(cam)}
    # each recorded day judged on its own plate: one may be a night (dark + lamp-tinted, or greyscale infrared)
    plates = {d: data.background(cam, d) for d in day_lab} if len(day_lab) > 1 else {d: data.background(cam)
                                                                                     for d in day_lab}
    day_v = {d: float(np.median(cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 2])) for d, b in plates.items()}
    night_day = {d: bool(day_v[d] < NIGHT_V or is_night(frame_light(b, (0, 0, 1, 1)))) for d, b in plates.items()}
    found: dict[int, dict[str, Any]] = {}
    if len(objs):
        tr = data.tracks(cam)
        tr = tr[tr.tid.isin(objs.id)]
        best = tr.loc[(tr.bw * tr.bh).groupby(tr.tid).idxmax()]      # same frame data.crop(t=None) picks
        want: dict[int, list[Any]] = {}
        for r in best.itertuples():
            want.setdefault(int(r.f), []).append(r)
        last = max(want)
        cap_ = cv2.VideoCapture(str(data.CAMS[cam]["video"]))
        i = 0
        while i <= last and cap_.grab():
            need = want.get(i)
            if need:
                ok, img = cap_.retrieve()
                if ok:
                    for r in need:
                        (x1, y1, x2, y2), (bx1, by1, bx2, by2) = _box_in_crop(r, W, H)
                        c = img[y1:y2, x1:x2]
                        if c.size == 0 or bx2 <= bx1 or by2 <= by1:
                            continue
                        cv2.imwrite(str(od / "crops" / f"{int(r.tid)}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        col = colour_of(c[by1:by2, bx1:bx2], objs.loc[objs.id == r.tid, "name"].iat[0])
                        found[int(r.tid)] = {**col, "crop_t": round(float(r.t), 2), "area": float(r.bw * r.bh),
                                             "light": frame_light(img, (r.x1, r.y1, r.x2, r.y2))}
            i += 1
        cap_.release()
        # anything the pass missed (decoder ended early): fall back to the seeking helper
        for oid in objs.id:
            if int(oid) not in found:
                try:
                    rel = data.save_crop(cam, int(oid))
                    img = cv2.imread(str(od / rel))
                    r = best[best.tid == oid].iloc[0]
                    _, (bx1, by1, bx2, by2) = _box_in_crop(r, W, H)
                    nm = objs.loc[objs.id == oid, "name"].iat[0]
                    try:
                        light = frame_light(data.frame_at(cam, float(r.t)), (r.x1, r.y1, r.x2, r.y2))
                    except ValueError:
                        light = {"v": 0.0, "dark": 1.0, "s": 0.0, "s90": 0.0}   # unreadable: no colour
                    found[int(oid)] = {**colour_of(img[by1:by2, bx1:bx2], nm), "crop_t": round(float(r.t), 2),
                                       "area": float(r.bw * r.bh), "light": light}
                except (KeyError, ValueError):
                    pass

    rows = []
    for r in objs.itertuples():
        c = found.get(int(r.id))
        if c is None:
            continue                     # no frame could be read for it: left out, counted below
        night = bool(night_day.get(r.day, False) or is_night(c["light"]))
        why = judge(r.name, c, c["area"], night)
        keep = why is None
        rows.append({"object_id": int(r.id), "name": r.name, "colour": c["colour"] if keep else None,
                     "rgb": c["rgb"] if keep else None,
                     "first_t": round(r.first_t, 2), "last_t": round(r.last_t, 2), "parked": bool(r.parked),
                     "n": int(r.n), "crop": f"crops/{int(r.id)}.jpg", "crop_t": c["crop_t"],
                     "colour_frac": c["frac"] if keep else None, "colour_reliable": keep, "colour_why": why,
                     "night": night, "day": r.day})
    colours = {k: 0 for k in COLOURS}
    for r in rows:
        if r["colour"] is not None:
            colours[r["colour"]] += 1
    uncoloured: dict[str, int] = {}
    for r in rows:
        if r["colour_why"]:
            uncoloured[r["colour_why"]] = uncoloured.get(r["colour_why"], 0) + 1
    n_all = len(data.objects(cam))
    n_veh = sum(r["name"] in COLOURED for r in rows)
    notes = []
    if n_all > len(objs):
        notes.append(f"capped: {len(objs)} of {n_all} objects kept (most frames first)")
    if len(objs) > len(rows):
        notes.append(f"{len(objs) - len(rows)} objects had no readable frame and were left out")
    if uncoloured.get("person", 0) + uncoloured.get("rider", 0):
        notes.append(f"people are never described by how they look: {uncoloured.get('person', 0)} people and "
                     f"{uncoloured.get('rider', 0)} bikes/motorbikes (box is mostly the rider) have no colour")
    for d, dark in night_day.items():
        if dark:
            notes.append(f"{day_lab[d]} is a night recording (dark street-lit or greyscale infrared; background "
                         f"brightness {day_v[d]:.0f}/255): colours can't be read, so its vehicles get no colour")
    for why, what in (("night", "were seen in a night frame"),
                      ("dark box", f"are too dark to read (box p95 brightness < {DARK_P95})"),
                      ("too small", f"are too small (< {MIN_AREA} px)")):
        if uncoloured.get(why):
            notes.append(f"{uncoloured[why]} of {n_veh} vehicles {what}: no colour given")
    out = {"camera": cam, "objects": rows, "colours": {k: v for k, v in colours.items() if v},
           "n_coloured": sum(colours.values()), "n_vehicles": n_veh, "uncoloured": uncoloured,
           "n_objects_total": n_all, "cap": cap, "notes": notes, "scene_brightness": round(scene_v, 1),
           "scene_brightness_by_day": {d: round(v, 1) for d, v in day_v.items()},
           "night_days": [d for d, dark in night_day.items() if dark],
           "method": ("vehicles (car/truck/bus) only: dominant paint colour of the box body in HSV, borders and "
                      "darkest/brightest 10% ignored; no colour at night or for dark or tiny boxes. People, "
                      "bicycles and motorbikes are never given a colour or any appearance attribute.")}
    data.write_json(od / "index.json", out)
    _summary(cam, out, time.time() - t0)
    return out


def _summary(cam: str, idx: dict[str, Any], secs: float) -> None:
    rows = idx["objects"]
    by_name = pd.Series([r["name"] for r in rows]).value_counts().to_dict() if rows else {}
    lines = [f"index: {cam}",
             f"objects indexed: {len(rows)} (of {idx['n_objects_total']} tracked objects with >= 8 frames; cap {idx['cap']})",
             f"by class: {json.dumps(by_name)}",
             f"parked: {sum(r['parked'] for r in rows)}",
             f"colours (vehicles only): {json.dumps(idx['colours'])} = {idx['n_coloured']} of "
             f"{idx['n_vehicles']} vehicles",
             f"no colour, by reason: {json.dumps(idx['uncoloured'])}",
             f"night days: {json.dumps(idx['night_days'])}; brightness by day: "
             f"{json.dumps(idx['scene_brightness_by_day'])}",
             f"crops written: {len(rows)} under crops/<id>.jpg",
             f"search examples: " + "; ".join(f"'{q}' -> {len(_match(rows, q))}" for q in _demo_queries(rows)),
             f"seconds: {secs:.1f}"]
    lines += [f"note: {n}" for n in idx["notes"]]
    (data.out_dir(cam) / "index_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _demo_queries(rows: list[dict[str, Any]]) -> list[str]:
    pairs = pd.Series([f"{r['colour']} {r['name']}" for r in rows if r["colour"]], dtype=object).value_counts()
    qs = list(pairs.index[:3])
    if any(r["name"] == "person" for r in rows):
        qs += ["people", "man in black"]          # the second one must be refused (0)
    return qs


# ---------------------------------------------------------------- search

def parse(text: str) -> dict[str, Any]:
    words = re.findall(r"[a-z]+", text.lower())
    colours: list[str] = []
    names: list[str] = []
    parked = None
    for w in words:
        for cand in (w, w[:-1] if w.endswith("s") else w, w[:-2] if w.endswith("es") else w):
            if cand in COLOUR_WORDS:
                colours.append(COLOUR_WORDS[cand])
                break
            if cand in CLASS_WORDS:
                names += CLASS_WORDS[cand]
                break
        if w in ("parked", "stationary"):
            parked = True
        elif w in ("moving", "passing", "driving", "walking"):
            parked = False
    return {"colours": sorted(set(colours)), "names": sorted(set(names)), "parked": parked}


def refusal(text: str) -> str | None:
    """REFUSAL when `text` asks for people by how they look, else None.

    Refused: a person word with a colour / garment / looks word ("person in white", "black people", "anyone
    with a beard"), a garment with a colour ("red hoodie"), and any gender or age word ("woman", "kids")."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    colour = any(w in COLOUR_WORDS or (w.endswith("s") and w[:-1] in COLOUR_WORDS) for w in words)
    person = bool(words & PERSON_WORDS)
    garment = bool(words & GARMENT_WORDS)
    if words & GENDER_AGE_WORDS or (garment and (colour or person)) or (person and (colour or words & LOOKS_WORDS)):
        return REFUSAL
    return None


def _match_why(rows: list[dict[str, Any]], text: str, near: bool = True) -> tuple[list[int], str | None]:
    """(object ids, reason). reason is REFUSAL for appearance queries about people, a short note when the
    answer is empty because colours were withheld or when a neighbouring shade was used, else None."""
    why = refusal(text)
    if why:
        return [], why
    p = parse(text)

    def hits(cols: list[str]) -> list[dict[str, Any]]:
        return [r for r in rows if (not cols or (r.get("colour") is not None and r["colour"] in cols
                                                 and r["name"] != "person"))    # never a person by colour
                and (not p["names"] or r["name"] in p["names"])
                and (p["parked"] is None or r["parked"] == p["parked"])]

    if not p["colours"] and not p["names"] and p["parked"] is None:
        return [], None                   # nothing we understand: do not return everyone
    got, note = hits(p["colours"]), None
    if not got and near and p["colours"]:
        shades = sorted({n for c in p["colours"] for n in NEAR.get(c, [])})
        got = hits(shades) if shades else []     # no neighbour (blue/green): an empty list must not mean "any"
        if got:
            note = f"no {'/'.join(p['colours'])} found; showing the nearest shade ({', '.join(shades)})"
    if not got and p["colours"]:
        if p["names"] and all(n in RIDERS or n == "person" for n in p["names"]):
            note = "bikes and motorbikes have no colour: the box is mostly the rider, who is never described"
        else:
            unk = [r for r in rows if r.get("colour") is None and r["name"] in COLOURED
                   and (not p["names"] or r["name"] in p["names"])
                   and (p["parked"] is None or r["parked"] == p["parked"])]
            if unk:
                note = (f"can't tell: {len(unk)} matching vehicle{'s have' if len(unk) != 1 else ' has'} no "
                        "confident colour (night, dark or tiny)")
    return [r["object_id"] for r in sorted(got, key=lambda r: (-r["n"], r["first_t"]))], note


def _match(rows: list[dict[str, Any]], text: str, near: bool = True) -> list[int]:
    return _match_why(rows, text, near)[0]


_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def load(cam: str) -> dict[str, Any]:
    p = data.OUT / cam / "index.json"
    if not p.exists():
        return build(cam)
    return json.loads(p.read_text(encoding="utf-8"))


def search(cam: str, text: str) -> list[int]:
    """Object ids whose colour and class match the words in `text`, most-seen first.

    "white car" -> white cars; "people" -> every person; "parked silver car" -> parked silver cars.
    If the exact colour finds nothing, neighbouring shades are tried (white->silver, grey->silver/black, ...).
    Unknown words are ignored; text with no colour/class/parked word returns [].
    People are never matched by appearance: "man in black", "person in white", "woman" -> [] and the module-level
    `last_reason` is set to REFUSAL (show REFUSAL_TEXT). After any call, `last_reason` holds the reason or None."""
    global last_reason
    ids, last_reason = search_why(cam, text)
    return ids


def search_why(cam: str, text: str) -> tuple[list[int], str | None]:
    """search() plus its reason, without touching module state (safe for a threaded server)."""
    p = data.OUT / cam / "index.json"
    mt = p.stat().st_mtime if p.exists() else -1.0
    if cam not in _CACHE or _CACHE[cam][0] != mt:
        rows = load(cam)["objects"]
        _CACHE[cam] = (p.stat().st_mtime if p.exists() else mt, rows)
    return _match_why(_CACHE[cam][1], text)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Index every object per camera (vehicles get a colour name).")
    ap.add_argument("cams", nargs="*", default=list(data.CAMS))
    ap.add_argument("--cap", type=int, default=CAP)
    a = ap.parse_args(argv)
    for cam in a.cams:
        t0 = time.time()
        idx = build(cam, a.cap)
        print(f"{cam}: {len(idx['objects'])} objects, colours {idx['colours']}, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
