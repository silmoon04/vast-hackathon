"""Caption lie detector: check the pipeline's AI captions against detector evidence.

Inputs (VSS row shape; caption = reasoning_content, detector = object_counts):
  * the practice corpus (vast-hackathon/data/mock_corpus.json, a list of rows) -> out/captions/captions.json.
    For person-bicycle-car-detection clips our own YOLO+ByteTrack run on the original video is a second witness:
    our_counts = max simultaneous tracked objects per class within the segment.
  * the event pack (data/packd/segments.json, {"chunks": [...], "segments": [...]}) -> out/captions/<camera_id>.json.
    Segment times are put on the camera timeline with data/packd/chunk_offsets.json (offset of each chunk, day
    ranges). Its detector counts come from the same YOLO sidecars our tracks are built from, so there is ONE
    witness there, not two: our_counts stays null.

Everything is plain rules, listed below, so a judge can read exactly why a row got its verdict.
  * claim  = a COCO noun (or synonym; "vehicle" = car+truck+bus+motorcycle) in the caption; its count comes from
             the word before it: number words and digits are exact ("two cars"), "a"/"an"/"another" only say
             "at least one", "a few"/"several"/plurals give no count, "the"/"her"/... add no new count.
  * negation: "no pedestrians", "without any people", "no other vehicles or pedestrians" claim zero. Relative
             negations ("no MOVING vehicles", "no other cars") are not checked, since parked cars still count.
  * person = the explicit number on a group noun ("three people") if there is one, else the sum of the
             individually introduced people ("a man and a woman" = at least 2).
  * verdict: contradiction if a claimed MOVER class (person, vehicle, animal) was seen by no detector, an exact
             or zero count is off by >= 2 from every detector, an "at least" count is 2 or more above every
             detector, or the caption describes an object that should show up as a COCO lookalike (e.g. a drone
             -> airplane/bird/kite) while every detector saw nothing at all; consistent if every claim is
             supported (off by one = "count differs"; static things such as tables or cups the detector did not
             report are "unconfirmed", since silence about clutter is weak evidence); unverifiable if nothing in
             the caption could be checked.
  * our_counts: each track takes its majority class inside the segment (ByteTrack ids flip labels), tracks
             with fewer than 3 detections in the segment are dropped as blips, then max per frame.
  * detector_suspect: a detector class the caption never mentions AND a second signal says it is wrong:
             the caption names a non-COCO object the detector is known to force into that class (LOOKALIKE),
             the class is implausible for the scene the caption describes (SCENE), or the other detector
             did not see it either.
Output: out/captions/captions.json (practice corpus), out/captions/<camera_id>.json (event pack),
        out/captions/frames/..., out/captions/*summary.txt. Contradiction rows get a 6-frame strip.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import argparse

import cv2
import numpy as np

from stm import data

CORPUS = Path(os.environ.get("STM_CORPUS", str(data.ROOT.parent / "data" / "mock_corpus.json"))).expanduser()
VIDEOS = data.INTEL
PACKD = data.ROOT / "data" / "packd" / "segments.json"
OUT = data.OUT / "captions"

# clip name in the corpus -> dev camera in stm.data.CAMS whose tracks cover the same video
CLIP_TO_CAM = {"person-bicycle-car-detection": "person_bicycle_car_detection", "car-detection": "car_detection"}

# ---------------------------------------------------------------- claim-extraction vocabulary
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"]   # "orange" left out: it is mostly a colour word
SYNONYMS = {
    "person": ["people", "persons", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child",
               "children", "kid", "kids", "pedestrian", "pedestrians", "lady", "ladies", "guy", "guys",
               "gentleman", "adult", "adults", "individual", "individuals", "worker", "workers", "shopper",
               "shoppers", "customer", "customers", "cyclist", "cyclists", "walker", "walkers"],
    "bicycle": ["bike", "bikes"], "car": ["sedan", "sedans", "suv", "suvs", "taxi", "taxis", "hatchback",
                                          "hatchbacks", "minivan", "minivans", "jeep", "jeeps", "coupe", "coupes"],
    "vehicle": ["vehicle", "vehicles", "automobile", "automobiles"],
    "motorcycle": ["motorbike", "motorbikes", "scooter", "scooters"],
    "airplane": ["aeroplane", "aeroplanes", "plane", "planes", "jet", "jets", "aircraft"],
    "truck": ["lorry", "lorries", "van", "vans", "pickup"], "boat": ["ship", "ships"],
    "fire hydrant": ["hydrant", "hydrants"], "handbag": ["purse", "purses"], "suitcase": ["luggage"],
    "sports ball": ["ball", "balls"], "tennis racket": ["racket", "rackets"], "cup": ["mug", "mugs"],
    "couch": ["sofa", "sofas"], "potted plant": ["plant", "plants"], "dining table": ["table", "tables"],
    "tv": ["television", "televisions", "tvs", "monitor", "monitors"],
    "cell phone": ["phone", "phones", "smartphone", "smartphones", "mobile phone"],
    "refrigerator": ["fridge", "fridges"], "hair drier": ["hair dryer"],
}
GROUP_NOUNS = {"people", "persons", "men", "women", "children", "kids", "adults", "individuals", "pedestrians",
               "vehicles", "cars"}
VEHICLE = ("car", "truck", "bus", "motorcycle")      # what a "vehicle" claim is checked against (summed)
INDEFINITE = {"a", "an", "another"}                  # "at least one", not a count
QUANT = {"few", "several", "some", "many", "multiple", "numerous", "various", "couple", "lot", "lots", "group",
         "number", "handful", "pair", "dozens", "row", "line", "series", "bunch", "any"}
NEGATORS = {"no", "not", "without", "nor", "none", "nobody", "absence", "lack", "lacking", "devoid", "zero",
            "neither", "empty"}
HEDGE = {"unseen", "invisible", "hidden", "possibly", "possible", "perhaps", "maybe", "might", "potentially",
         "potential", "unclear", "if", "imagined"}            # "possibly moved by an unseen person": not in view
PREDICATIVE = {"is", "was", "be", "being", "been", "becomes", "remains"}   # "the truck is a pickup": no new object
CLAUSE_END = {".", ";", ":", "!", "?", "but", "while", "although", "though", "except", "however", "yet",
              "whereas", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "can", "could",
              "appears", "appear", "seems", "seem", "shows", "show", "includes", "include", "including"}
RELATIVE = {"other", "moving", "driving", "passing", "additional", "more", "further", "approaching", "oncoming",
            "active", "else", "new", "speeding", "traveling", "travelling"}
_RELATIVE_SOFT = {"other", "additional", "more", "else", "further"}   # "no other people" = zero if none named
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12, "both": 2, "a": 1, "an": 1, "another": 1, "single": 1}
DEFINITE = {"the", "this", "that", "these", "those", "his", "her", "their", "its", "same", "other"}
STOP = {"and", "or", "with", "of", "in", "on", "at", "to", "from", "by", "is", "are", "was", "were", "while",
        "into", "over", "under", "near", "behind", "next", "towards", "toward", "as", "for", "has", "have"}
COLOURS = {"black", "white", "red", "green", "blue", "yellow", "brown", "gray", "grey", "orange", "pink",
           "purple", "silver", "dark", "light", "beige", "golden"}
# non-COCO things a caption names -> COCO classes YOLO tends to force them into
LOOKALIKE = {
    "bolt": {"toothbrush", "baseball bat", "remote", "skateboard", "tennis racket", "knife", "scissors",
             "bottle", "cell phone", "spoon"},
    "tube": {"bottle", "baseball bat", "toothbrush", "remote", "skateboard"},
    "drone": {"airplane", "bird", "kite"},
    "window": {"tv", "laptop", "refrigerator", "microwave", "oven"},
    "plate": {"toilet", "sink", "frisbee", "clock"},
    "crockery": {"toilet", "sink", "frisbee", "clock"},
    "teapot": {"toilet", "sink", "vase"},
    "pottery": {"toilet", "sink", "vase"},
    "shelf": {"refrigerator", "oven", "microwave"},
}
LOOKALIKE_WORDS = {"bolts": "bolt", "tubes": "tube", "drones": "drone", "windows": "window", "plates": "plate",
                   "teapots": "teapot", "shelves": "shelf", "screw": "bolt", "screws": "bolt", "pipe": "tube",
                   "rod": "tube"} | {k: k for k in LOOKALIKE}
_BIG = {"airplane", "bus", "train", "truck", "boat", "car", "motorcycle", "elephant", "bear", "zebra", "giraffe",
        "horse", "cow", "sheep"}
_STREET_ODD = {"airplane", "boat", "train", "toilet", "bed", "oven", "microwave", "refrigerator", "sink", "toaster",
               "tv", "laptop", "keyboard", "mouse", "couch", "toothbrush", "hair drier", "teddy bear", "donut",
               "pizza", "cake", "sandwich", "hot dog", "broccoli", "carrot", "banana", "apple", "dining table",
               "elephant", "bear", "zebra", "giraffe", "surfboard", "skis", "snowboard"}
_INDOOR_ODD = _BIG | {"traffic light", "stop sign", "fire hydrant", "parking meter", "surfboard", "skis",
                      "snowboard"}
# scene words in the caption -> (scene label, COCO classes implausible there)
SCENES = {"parking lot": ("street", _STREET_ODD), "street": ("street", _STREET_ODD),
          "road": ("street", _STREET_ODD), "sidewalk": ("street", _STREET_ODD),
          "room": ("indoor", _INDOOR_ODD), "shop": ("indoor", _INDOOR_ODD), "store": ("indoor", _INDOOR_ODD),
          "aisle": ("indoor", _INDOOR_ODD), "kitchen": ("indoor", _INDOOR_ODD),
          "surface": ("tabletop", _BIG | {"person", "bicycle", "bench", "chair", "couch", "bed", "dining table",
                                          "refrigerator", "toilet"})}

# classes the detector reliably reports when present; for anything else, silence is weak evidence
STRICT_ABSENCE = {"person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "boat", "airplane", "bird",
                  "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "vehicle"}
# non-COCO nouns that a COCO detector still shows as a lookalike class when the thing is really there
SHOWS_AS = {"drone": {"airplane", "bird", "kite"}}
MIN_TRACK_DETS = 3

_NOUN: dict[tuple[str, ...], str] = {}
for _c in COCO:
    _NOUN[tuple(_c.split())] = _c
    _NOUN[tuple((_c + ("es" if _c.endswith(("s", "sh", "ch")) else "s")).split())] = _c
for _c, _ws in SYNONYMS.items():
    for _w in _ws:
        _NOUN[tuple(_w.split())] = _c
_NOUN[("knives",)] = "knife"
_NOUN[("benches",)] = "bench"
_NOUN[("buses",)] = "bus"
_NOUN[("pickup", "truck")] = _NOUN[("pickup", "trucks")] = "truck"
_MAXN = max(len(k) for k in _NOUN)
_IRREGULAR_PLURAL = {"people", "men", "women", "children", "persons"}
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*|\d+|[.,;:!?]")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("t-shirt", "tshirt"))


def _alpha(w: str) -> bool:
    return w.replace("-", "").isalpha()                  # "dark-colored" is an adjective, not a stop


def _is_plural(key: tuple[str, ...]) -> bool:
    last, cls = key[-1], _NOUN[key]
    return last in _IRREGULAR_PLURAL or (last.endswith("s") and not cls.endswith("s") and last != cls)


def _count_before(tok: list[str], i: int) -> tuple[int | None, bool, bool]:
    """Look back from the noun at tok[i] over adjectives for a determiner. Returns (count, exact, definite)."""
    j = i - 1
    while j >= 0 and i - j <= 5:
        w = tok[j]
        if w.isdigit():
            return int(w), True, False
        if w in QUANT:
            return None, False, False
        if w in NUMBERS:
            if w in INDEFINITE and j > 0 and tok[j - 1] in PREDICATIVE:
                return None, False, True                       # "is a standard pickup" re-describes
            return NUMBERS[w], w not in INDEFINITE, False
        if w in DEFINITE:
            return None, False, True
        if w == "and" and j > 0 and tok[j - 1] in COLOURS:   # "a black and white cat"
            j -= 1
            continue
        if w in STOP or not _alpha(w):
            return None, False, False
        j -= 1
    return None, False, False


def _negation(tok: list[str], i: int) -> tuple[bool, set[str]]:
    """Is the noun at tok[i] negated ("no pedestrians or moving vehicles")? Also returns the words between the
    negator and the noun, so "no OTHER vehicles" / "no MOVING vehicles" can be recognised as relative."""
    j, seen = i - 1, set()
    while j >= 0 and i - j <= 8:
        w = tok[j]
        if w in NEGATORS:
            return True, seen
        if w in CLAUSE_END or w.isdigit() or (w in NUMBERS and w != "single") or (w in DEFINITE and w != "other"):
            return False, set()
        seen.add(w)
        j -= 1
    return False, set()


def extract_claims(caption: str) -> list[dict[str, Any]]:
    """Every COCO noun mention in the caption as {"text", "object", "count", "exact", "negated", "relative",
    "group"} (count None = unspecified, 0 = negated)."""
    # video models sometimes loop and repeat whole sentences; count each distinct sentence once
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", caption.strip()) if x.strip()]
    tok = _tokens(" ".join(dict.fromkeys(sents)))
    out, i = [], 0
    while i < len(tok):
        for n in range(_MAXN, 0, -1):
            key = tuple(tok[i:i + n])
            if len(key) == n and key in _NOUN:
                cnt, exact, definite = _count_before(tok, i)
                if cnt is not None and not exact and _is_plural(key):
                    cnt = None                                # "a few cars" / "a row of cars" give no count
                neg, quals = _negation(tok, i)
                hedged = any(w in HEDGE for w in tok[max(0, i - 6):i] if w not in CLAUSE_END)
                if hedged and not neg:
                    cnt, exact, definite = None, False, False
                j = i - 1
                while j >= 0 and i - j <= 5 and _alpha(tok[j]) and tok[j] not in STOP | DEFINITE \
                        and tok[j] not in NUMBERS and tok[j] not in NEGATORS:
                    j -= 1
                start = j if j >= 0 and (tok[j] in NUMBERS or tok[j] in DEFINITE or tok[j].isdigit()
                                         or tok[j] in NEGATORS) else j + 1
                out.append({"text": " ".join(tok[start:i + n]), "object": _NOUN[key],
                            "count": 0 if neg else cnt, "exact": True if neg else exact, "negated": neg,
                            "relative": sorted((set(tok[start:i]) & RELATIVE) | (quals & _RELATIVE_SOFT)),
                            "group": " ".join(key) in GROUP_NOUNS,
                            "asserted": neg or cnt is not None or definite})
                i += n
                break
        else:
            i += 1
    return out


def claim_detail(claims: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per class: {"count", "exact"}. Explicit group number if any ("three people"), else the sum of counted
    mentions (exact only for a single number word), else None. A class mentioned ONLY in a negation claims 0
    ("no pedestrians"); relative negations ("no moving vehicles", "no other cars") are dropped, except
    "no other people" for a class the caption never named positively."""
    res: dict[str, dict[str, Any]] = {}
    pos_classes = {x["object"] for x in claims if not x["negated"]}
    vehicle_named = bool(pos_classes & (set(VEHICLE) | {"vehicle"}))
    for c in sorted({x["object"] for x in claims}):
        cs = [x for x in claims if x["object"] == c]
        pos = [x for x in cs if not x["negated"]]
        if c == "vehicle" and pos_classes & set(VEHICLE):
            continue                     # "two vehicles: an SUV and a sedan" is checked through car/truck/...
        if pos and not any(x.get("asserted", True) for x in pos) and any(x["negated"] for x in cs):
            pos = []                     # "no pedestrians ... interactions with people": the bare plural adds nothing
        if pos:
            asserted = any(x.get("asserted", True) for x in pos)
            groups = [x["count"] for x in pos if x["group"] and x["count"] is not None and x["exact"]]
            singles = [x for x in pos if not (x["group"] and x["exact"]) and x["count"] is not None]
            if groups:
                res[c] = {"count": max(groups), "exact": True, "asserted": True}
            elif singles:
                res[c] = {"count": sum(x["count"] for x in singles),
                          "exact": len(singles) == 1 and singles[0]["exact"], "asserted": True}
            else:
                res[c] = {"count": None, "exact": False, "asserted": asserted}
            continue
        is_vehicle = c in VEHICLE or c == "vehicle"
        for x in cs:
            rel = set(x.get("relative") or ())
            if not rel or (rel <= _RELATIVE_SOFT and not (is_vehicle and vehicle_named)):
                res[c] = {"count": 0, "exact": True, "asserted": True}
                break
    return res


def claimed_counts(claims: list[dict[str, Any]]) -> dict[str, int | None]:
    """Per class: the claimed count (None = unspecified, 0 = the caption says there is none)."""
    return {c: d["count"] for c, d in claim_detail(claims).items()}


def scene_and_lookalikes(caption: str) -> tuple[list[tuple[str, str, set[str]]], list[str]]:
    low = caption.lower()
    toks = set(_tokens(caption))
    scenes = [(w, lab, odd) for w, (lab, odd) in SCENES.items() if re.search(rf"\b{w}s?\b", low)]
    looks = sorted({LOOKALIKE_WORDS[w] for w in toks if w in LOOKALIKE_WORDS})
    return scenes, looks


# ---------------------------------------------------------------- evidence
def _segment_tracks(cam: str, start: float, end: float):
    """Our detections in [start, end): tracked only, majority class per track, blips (< 3 dets) dropped."""
    df = data.tracks(cam)
    s = df[(df.t >= start) & (df.t < end) & (df.tid >= 0)]
    if s.empty:
        return s
    n = s.groupby("tid").t.transform("size")
    s = s[n >= MIN_TRACK_DETS]
    if s.empty:
        return s
    major = s.groupby("tid").name.agg(lambda x: x.mode().iat[0])
    return s.assign(cls_major=s.tid.map(major))


def our_counts(cam: str, start: float, end: float) -> dict[str, int]:
    """Max simultaneous tracked objects per class in [start, end) from our own tracks."""
    s = _segment_tracks(cam, start, end)
    if s.empty:
        return {}
    per = s.groupby(["f", "cls_major"]).tid.nunique().unstack(fill_value=0).max()
    return {str(k): int(v) for k, v in per.items() if v > 0}


def _grab(videos: list[tuple[Path, float]], times: list[float]) -> list[np.ndarray] | None:
    """Frames at camera-timeline `times` from the first video that has them all. videos = [(path, offset)]:
    the frame for time t is read at t - offset in that file."""
    for p, off in videos:
        if not p or not Path(p).exists():
            continue
        cap = cv2.VideoCapture(str(p))
        got = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t - off) * 1000.0)
            ok, im = cap.read()
            if not ok:
                break
            got.append(im)
        cap.release()
        if len(got) == len(times):
            return got
    return None


def _strip(frames: list[np.ndarray], w: int = 320) -> np.ndarray:
    tiles = [cv2.resize(f, (w, max(1, round(f.shape[0] * w / f.shape[1]))), interpolation=cv2.INTER_AREA)
             for f in frames]
    h = min(t.shape[0] for t in tiles)
    return cv2.hconcat([t[:h] for t in tiles])


def _draw_ours(cam: str, start: float, end: float) -> tuple[np.ndarray, float] | None:
    """Our tracker's boxes on the tracked frame nearest the segment midpoint (the frame they were found on)."""
    s = _segment_tracks(cam, start, end)
    if s.empty:
        return None
    mid = (start + end) / 2
    f = s.f.iat[int((s.t - mid).abs().argmin())]
    g = s[s.f == f]
    t = float(g.t.iat[0])
    out = data.frame_at(cam, t)
    for r in g.itertuples():
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(out, p1, p2, (40, 220, 255), 2)
        cv2.putText(out, f"{r.cls_major} #{r.tid}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (40, 220, 255), 1, cv2.LINE_AA)
    return out, t


_MASS = {"crockery", "pottery"}
_PLURAL = {"person": "people", "bus": "buses", "knife": "knives", "bench": "benches", "skis": "skis",
           "scissors": "scissors", "sheep": "sheep"}


def _a(word: str) -> str:
    return word if word in _MASS else ("an " if word[:1] in "aeiou" else "a ") + word


def _n(k: int, cls: str) -> str:
    return f"{'no' if k == 0 else k} {cls if k == 1 else _PLURAL.get(cls, cls + 's')}"


# ---------------------------------------------------------------- judging
def _evidence_count(src: dict[str, int], cls: str) -> int:
    if cls == "vehicle":
        return int(sum(src.get(c, 0) for c in VEHICLE))
    return int(src.get(cls, 0))


def judge(claims: list[dict[str, Any]], pipe: dict[str, int], ours: dict[str, int] | None,
          caption: str) -> dict[str, Any]:
    sources = [("pipeline", pipe)] + ([("our tracker", ours)] if ours is not None else [])
    detail = claim_detail(claims)
    want = {c: d["count"] for c, d in detail.items()}
    checks, bad, good, weak, differ = [], [], [], [], []
    scenes, looks = scene_and_lookalikes(caption)
    everything_empty = all(not any(v > 0 for v in src.values()) for _, src in sources)
    for w in looks:
        if w in SHOWS_AS and everything_empty:
            alts = "/".join(sorted(SHOWS_AS[w]))
            empty = ", ".join(f"{n} saw nothing" for n, _ in sources)
            bad.append(f"caption describes {_a(w)}, which a COCO detector would show as {alts}; "
                       f"no detector reported any object in this segment ({empty})")
            checks.append({"object": w, "claimed": None, "exact": False, "seen": {n: 0 for n, _ in sources},
                           "status": "unseen"})
    for cls, d in detail.items():
        k, exact = d["count"], d["exact"]
        seen = {name: _evidence_count(src, cls) for name, src in sources}
        seen_txt = ", ".join(f"{name} {n}" for name, n in seen.items()) + (
            " (car+truck+bus+motorcycle)" if cls == "vehicle" else "")
        claim_txt = _n(k, cls) if k is not None else cls
        if k == 0:
            off = min(seen.values())
            status = "supported" if off == 0 else ("count_differs" if off == 1 else "count_off")
            msg = f"caption says {claim_txt}; {seen_txt}"
            (good if off == 0 else differ if off == 1 else bad).append(msg + (f" (off by {off})" if off else ""))
            if off == 1:
                differ[-1] = msg + " (off by one)"
        elif not any(seen.values()) and cls not in STRICT_ABSENCE:
            status = "unconfirmed"
            weak.append(f"{cls} not reported ({seen_txt}); detectors often miss static objects, so this "
                        "neither confirms nor refutes the caption")
        elif not any(seen.values()) and not d.get("asserted", True):
            status = "unconfirmed"
            weak.append(f"caption only mentions {cls} in passing ('people', 'pedestrians'), not as something "
                        f"in view ({seen_txt})")
        elif not any(seen.values()) and cls in VEHICLE and any(
                _evidence_count(src, "vehicle") for _, src in sources):
            status = "class_differs"
            fam = ", ".join(f"{name} " + " + ".join(f"{c} {src[c]}" for c in VEHICLE if src.get(c))
                            for name, src in sources if _evidence_count(src, "vehicle"))
            differ.append(f"caption says {_a(cls) if k in (None, 1) else claim_txt}; the detector saw no {cls} but "
                          f"other vehicles ({fam}); cars, vans and pickups are often labelled differently")
        elif not any(seen.values()):
            status = "unseen"
            bad.append(f"caption mentions {cls}{f' x{k}' if k else ''}; no detector saw one ({seen_txt})")
        elif k is None:
            status = "supported"
            good.append(f"{cls} seen ({seen_txt})")
        else:
            # exact: distance either way; "at least k" (a/an mentions): only claiming MORE than seen is wrong
            off = min(abs(k - n) if exact else max(0, k - n) for n in seen.values())
            same = any(k == n for n in seen.values())
            status = "count_off" if off >= 2 else ("supported" if same else "count_differs")
            msg = f"caption says {'' if exact else 'at least '}{claim_txt}; {seen_txt}"
            if off >= 2:
                bad.append(msg + f" (off by {off})")
            elif same:
                good.append(msg)
            else:
                more = all(n > k for n in seen.values())
                differ.append(msg + (" (the detector counted more; the caption need not name every one)"
                                     if more and not exact else " (off by one)"))
        checks.append({"object": cls, "claimed": k, "exact": exact, "seen": seen, "status": status})
    if bad:
        verdict = "contradiction"
    elif good or differ:
        verdict = "consistent"
    else:
        verdict = "unverifiable"

    # detector side: classes the caption never mentions
    notes = []
    mentioned = set(want) | (set(VEHICLE) if "vehicle" in want else set())
    for name, src in sources:
        other = [s for n, s in sources if n != name]
        for cls, n in sorted(src.items()):
            if cls in mentioned or n <= 0:
                continue
            why = []
            src_words = [w for w in looks if cls in LOOKALIKE[w]]
            if src_words:
                why.append(f"caption describes {' / '.join(_a(w) for w in src_words)}, which COCO has no class "
                           f"for; YOLO often calls such things {cls}")
            for w, lab, odd in scenes:
                if cls in odd:
                    why.append(f"{_a(cls)} is implausible in a {lab} scene ('{w}')")
            if other and all(_evidence_count(s, cls) == 0 for s in other):
                why.append(f"{'our tracker' if name == 'pipeline' else 'the pipeline'} did not see it either")
            notes.append({"source": name, "object": cls, "count": int(n), "suspect": bool(why),
                          "reason": "; ".join(why) if why else "not mentioned in the caption"})
    omitted = [f"{x['object']} x{x['count']}" for x in notes if x["source"] == "pipeline" and not x["suspect"]
               and x["object"] == "person"]

    if verdict == "unverifiable":
        why_txt = "nothing in the caption can be checked against COCO detections" + (
            f" (it describes: {', '.join(looks)})" if looks else "")
    elif verdict == "contradiction":
        why_txt = "; ".join(bad)
    else:
        why_txt = "every checkable claim is backed: " + "; ".join(good + differ)
    if weak:
        why_txt += ". Unconfirmed: " + "; ".join(weak)
    if omitted:
        why_txt += f". Caption never mentions the detected {', '.join(omitted)}"
    sus = [x for x in notes if x["suspect"]]
    return {"checks": checks, "verdict": verdict, "why": why_txt, "detector_notes": notes,
            "detector_suspect": bool(sus), "count_differs": any(c["status"] == "count_differs" for c in checks),
            "class_differs": any(c["status"] == "class_differs" for c in checks),
            "detector_reason": "; ".join(f"{x['source']} {x['object']}: {x['reason']}" for x in sus) or None}


def clip_of(row: dict[str, Any]) -> str:
    return re.sub(r"_segment_\d+(_of_\d+)?$", "", Path(row["filename"]).stem)


# ---------------------------------------------------------------- corpus shapes
def _mock_segments(rows: list[dict[str, Any]], cams: list[str] | None) -> list[dict[str, Any]]:
    allowed = set(cams) if cams else set(CLIP_TO_CAM.values())
    segs = []
    for r in rows:
        clip = clip_of(r)
        start, end = round(float(r["segment_start_sec"]), 2), round(float(r["segment_end_sec"]), 2)
        cam = CLIP_TO_CAM.get(clip)
        videos = [(VIDEOS / f"{clip}.mp4", 0.0)]
        if r.get("_local_path"):
            videos.append((Path(r["_local_path"]), float(r["segment_start_sec"])))
        segs.append({"id": f"{clip}_{int(r.get('segment_number') or 0):03d}", "pk": r.get("pk"), "clip": clip,
                     "start": start, "end": end, "row": r, "videos": videos,
                     "ours_cam": cam if cam in allowed and cam in data.CAMS else None})
    return segs


def _packd_offsets(obj: dict[str, Any], corpus: Path) -> tuple[dict[str, float], list[list[Any]]]:
    """Chunk -> start on the camera timeline, and the day table [[id, label, start, end|None], ...]."""
    p = Path(corpus).parent / "chunk_offsets.json"
    off = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    offsets = {str(k): float(v) for k, v in (off.get("offsets") or {}).items()}
    # fallback for chunks the offsets file does not list: chain them by their longest segment end
    seg_end: dict[str, float] = {}
    for r in obj.get("segments", []):
        ch = r.get("chunk") or clip_of(r)
        seg_end[ch] = max(seg_end.get(ch, 0.0), float(r.get("segment_end_sec") or 0.0))
    order = [Path(str(c)).stem for c in obj.get("chunks") or []] + sorted(seg_end)
    cursor = max((offsets[c] + seg_end.get(c, 0.0) for c in offsets), default=0.0)
    for ch in dict.fromkeys(order):
        if ch not in offsets and ch in seg_end:
            offsets[ch], cursor = cursor, cursor + seg_end[ch]
    return offsets, [list(d) for d in off.get("days") or []]


def _day_of(days: list[list[Any]], t: float, chunk: str) -> tuple[str | None, str | None]:
    for d in days:
        lo, hi = float(d[2]), (float("inf") if d[3] is None else float(d[3]))
        if lo <= t < hi:
            return str(d[0]), str(d[1])
    m = re.search(r"_(\d{8})_chunk_", chunk)
    return (m.group(1), m.group(1)) if m else (None, None)


def _packd_segments(obj: dict[str, Any], corpus: Path, cams: list[str] | None) -> list[dict[str, Any]]:
    offsets, days = _packd_offsets(obj, corpus)
    segs = []
    for r in obj.get("segments", []):
        cam = str(r.get("camera_id") or "packd")
        if cams and cam not in cams:
            continue
        chunk = r.get("chunk") or clip_of(r)
        num = int(float(r.get("segment_number") or 0))
        s0, s1 = float(r["segment_start_sec"]), float(r["segment_end_sec"])
        base = offsets.get(chunk, 0.0)
        start, end = round(base + s0, 2), round(base + s1, 2)
        day, label = _day_of(days, start, chunk)
        m = re.search(r"_(\d{8})_chunk_(\d+)$", chunk)
        rid = f"{m.group(1)}_c{m.group(2)}_s{num:02d}" if m else f"{chunk}_s{num:02d}"
        videos = []
        if cam in data.CAMS:
            videos.append((Path(data.CAMS[cam]["video"]), 0.0))
        local = Path(corpus).parent / "segments" / Path(str(r.get("filename") or "")).name
        if r.get("filename"):
            videos.append((local, start))
        segs.append({"id": rid, "pk": r.get("pk"), "clip": chunk, "camera": cam, "start": start, "end": end,
                     "seg_start": round(s0, 2), "seg_end": round(s1, 2), "segment": num, "day": day,
                     "day_label": label, "segment_source": r.get("source"), "row": r, "videos": videos,
                     "ours_cam": None})
    return segs


def _judge_segments(segs: list[dict[str, Any]], root: Path, frames: bool, sub: str = "") -> list[dict[str, Any]]:
    fdir = f"frames/{sub}/" if sub else "frames/"
    if frames:
        (root / fdir).mkdir(parents=True, exist_ok=True)
    res = []
    for sg in segs:
        r, rid, start, end = sg["row"], sg["id"], sg["start"], sg["end"]
        caption = (r.get("reasoning_content") or "").strip()
        pipe = {str(k): int(v) for k, v in json.loads(r.get("object_counts") or "{}").items()}
        cam = sg["ours_cam"]
        ours = our_counts(cam, start, end) if cam else None
        claims = extract_claims(caption)
        j = judge(claims, pipe, ours, caption)
        frame = frame_ours = frame_ours_t = strip = None
        strip_t: list[float] = []
        if frames:
            got = _grab(sg["videos"], [(start + end) / 2])
            if got:
                frame = f"{fdir}{rid}.jpg"
                cv2.imwrite(str(root / frame), got[0], [cv2.IMWRITE_JPEG_QUALITY, 85])
            if j["verdict"] == "contradiction":
                dur = end - start
                ts = [round(start + (i + 0.5) * dur / 6, 2) for i in range(6)]
                got = _grab(sg["videos"], ts)
                if got:
                    strip, strip_t = f"{fdir}{rid}_strip.jpg", ts
                    cv2.imwrite(str(root / strip), _strip(got), [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ours is not None:
                ov = _draw_ours(cam, start, end)
                if ov is not None:
                    frame_ours = f"{fdir}{rid}_ours.jpg"
                    frame_ours_t = round(ov[1], 2)
                    cv2.imwrite(str(root / frame_ours), ov[0], [cv2.IMWRITE_JPEG_QUALITY, 85])
        row = {"id": rid, "pk": sg.get("pk"), "clip": sg["clip"], "start": start, "end": end, "caption": caption,
               "claims": [{"text": c["text"], "object": c["object"], "count": c["count"], "negated": c["negated"]}
                          for c in claims],
               "claimed_counts": claimed_counts(claims), "checks": j["checks"],
               "pipeline_counts": pipe, "our_counts": ours, "our_camera": cam if ours is not None else None,
               "verdict": j["verdict"], "why": j["why"], "count_differs": j["count_differs"],
               "class_differs": j["class_differs"], "repeats": _repeats(caption),
               "detector_suspect": j["detector_suspect"], "detector_reason": j["detector_reason"],
               "detector_notes": j["detector_notes"], "model": r.get("cosmos_model"),
               "detector": r.get("perception_source"),
               "frame": frame, "frame_t": round((start + end) / 2, 2), "frame_ours": frame_ours,
               "frame_ours_t": frame_ours_t, "strip": strip, "strip_t": strip_t}
        for k in ("camera", "day", "day_label", "seg_start", "seg_end", "segment", "segment_source"):
            if k in sg:
                row[k] = sg[k]
        res.append(row)
    return res


def _repeats(caption: str) -> int:
    """How many sentences the caption repeats verbatim (a looping video model)."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", caption.strip()) if x.strip()]
    return len(sents) - len(set(sents))


def _totals(res: list[dict[str, Any]]) -> dict[str, int]:
    return {k: sum(x["verdict"] == k for x in res) for k in ("consistent", "contradiction", "unverifiable")}


def _write(root: Path, name: str, out: dict[str, Any], title: str) -> None:
    data.write_json(root / f"{name}.json", out)
    res = out["rows"]
    lines = [f"captions: {len(res)} rows from {title}",
             "totals: " + ", ".join(f"{k} {v}" for k, v in out["totals"].items()),
             f"count differs by one: {out['count_differs']}",
             f"detector_suspect rows: {out['detector_suspect']}",
             f"rows cross-checked with our tracks: {sum(x['our_counts'] is not None for x in res)}",
             f"frames saved: {sum(x['frame'] is not None for x in res)}",
             f"strips saved: {sum(x['strip'] is not None for x in res)}", out.get("witness_note") or "", ""]
    lines += [f"{x['id']}: {x['verdict']}{' [count differs]' if x['count_differs'] else ''}"
              f"{' [detector suspect]' if x['detector_suspect'] else ''} - {x['why']}" for x in res]
    summary = "captions_summary.txt" if name == "captions" else f"{name}_summary.txt"
    (root / summary).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _models(res: list[dict[str, Any]]) -> list[str]:
    return sorted({x["model"] for x in res if x.get("model")})


def run(cams: list[str] | None = None, corpus: Path = CORPUS, frames: bool = True,
        out_root: Path | None = None) -> dict[str, Any]:
    """Judge one corpus. A list of rows (practice corpus) -> captions.json. A {"segments": [...]} pack (event
    data) -> <camera_id>.json per camera; returns that camera's result (or {camera: result} for several)."""
    root = Path(out_root) if out_root else OUT
    root.mkdir(parents=True, exist_ok=True)
    obj = json.loads(Path(corpus).read_text(encoding="utf-8"))
    rules = __doc__.split("Output:")[0].strip()
    if isinstance(obj, dict) and "segments" in obj:
        segs = _packd_segments(obj, Path(corpus), cams)
        outs = {}
        for cam in dict.fromkeys(s["camera"] for s in segs):
            res = _judge_segments([s for s in segs if s["camera"] == cam], root, frames, sub=cam)
            days = list(dict.fromkeys((x["day"], x["day_label"]) for x in res if x.get("day")))
            out = {"camera": cam, "source": str(corpus), "practice": False, "models": _models(res),
                   "detectors": sorted({x["detector"] for x in res if x.get("detector")}),
                   "days": [{"id": d, "label": lab} for d, lab in days],
                   "witness_note": "One witness, not two: these detector counts come from the same YOLO detection "
                                   "sidecars our tracks are built from, so our tracker is not an independent "
                                   "check on this camera.",
                   "rules": rules, "rows": res, "totals": _totals(res),
                   "count_differs": sum(x["count_differs"] for x in res),
                   "detector_suspect": sum(x["detector_suspect"] for x in res)}
            _write(root, cam, out, f"{Path(corpus).name} ({cam})")
            outs[cam] = out
        return next(iter(outs.values())) if len(outs) == 1 else outs
    res = _judge_segments(_mock_segments(obj, cams), root, frames)
    out = {"camera": None, "source": str(corpus), "practice": True, "models": _models(res),
           "witness_note": "Practice corpus from the organisers' sample clips. For person-bicycle-car-detection our "
                           "own YOLO + ByteTrack run on the original video is a second, independent witness.",
           "rules": rules, "rows": res, "totals": _totals(res),
           "count_differs": sum(x["count_differs"] for x in res),
           "detector_suspect": sum(x["detector_suspect"] for x in res)}
    _write(root, "captions", out, Path(corpus).name)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cams", nargs="*", help="limit to these cameras (default: all)")
    ap.add_argument("--corpus", action="append", help="corpus file(s); default: practice corpus and "
                                                       "data/packd/segments.json, whichever exist")
    ap.add_argument("--no-frames", action="store_true", help="skip frames and strips (no video needed)")
    a = ap.parse_args(argv)
    corpora = [Path(c) for c in a.corpus] if a.corpus else [c for c in (CORPUS, PACKD) if c.exists()]
    for c in corpora:
        t0 = time.time()
        o = run(a.cams or None, corpus=c, frames=not a.no_frames)
        for x in (o.values() if "rows" not in o else [o]):
            print(f"captions {Path(c).name} {x.get('camera') or 'practice'}: {x['totals']} count_differs "
                  f"{x['count_differs']} detector_suspect {x['detector_suspect']} "
                  f"strips {sum(r['strip'] is not None for r in x['rows'])} {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
