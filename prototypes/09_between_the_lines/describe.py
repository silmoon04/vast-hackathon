"""Visual side of Between The Lines: slots, novelty, YOLO evidence, Cosmos description, fitting and AMBER flags."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
import protokit as pk  # noqa: E402

WPS = 2.5            # brief budget: at most 2.5 words per second of usable slot (the real TTS duration is still the final fit check)
MAX_SLOT = 10.0      # a long silence is cut into slots of at most this many seconds
MAX_WORDS = 8        # AD-style short lines: at most 8 words, whatever the slot length
HEDGES = re.compile(r"\b(appears? to|seems?|possibly|perhaps|maybe|might|likely|looks like|could be|unclear|unknown)\b", re.I)

# --------------------------------------------------------------------------------------------- slots


def make_slots(gaps: list[dict], max_slot: float = MAX_SLOT, min_slot: float = 1.5) -> list[dict]:
    """Cut each gap's usable window into slots <= max_slot s. Each slot: start, end (voice window), gap index."""
    slots = []
    for gi, g in enumerate(gaps):
        s, e = g["usable_start"], g["usable_end"]
        n = max(1, math.ceil((e - s) / max_slot))
        step = (e - s) / n
        for k in range(n):
            a, b = s + k * step, s + (k + 1) * step
            if b - a >= min_slot:
                slots.append({"gap": gi, "start": round(a, 3), "end": round(b, 3), "kind": g["kind"]})
    for i, sl in enumerate(slots):
        sl["id"] = i
    return slots


# --------------------------------------------------------------------------------------------- shots / novelty


def shot_cuts(video: str, t0: float, t1: float, step: float = 0.25, thr: float = 0.45) -> list[float]:
    """Hard cuts by HSV-histogram distance between frames `step` s apart (cheap, no model)."""
    fps = pk.info(video)["fps"]
    stride = max(1, int(round(fps * step)))
    prev, cuts = None, []
    for _, t, fr in pk.iter_frames(video, stride=stride, start=t0, end=t1):
        h = cv2.calcHist([cv2.cvtColor(cv2.resize(fr, (96, 54)), cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
        h = cv2.normalize(h, h).flatten()
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > thr:
            cuts.append(round(t, 2))
        prev = h
    return cuts


def _small(fr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 36)), (5, 5), 0).astype(np.float32)


def scene_change(a: np.ndarray, b: np.ndarray) -> float:
    """0..1 mean absolute luminance difference between two frames (after blur): scene-level change."""
    return float(np.abs(_small(a) - _small(b)).mean() / 255.0)


def new_since(tr: pk.Tracks, t_from: float, t_to: float, seen_before: set[int]) -> dict:
    """Confirmed tracks that first appear in (t_from, t_to] and were not seen earlier: the 'what is new' evidence."""
    s = tr.summary(min_frames=4)
    if s.empty:
        return {"new_tracks": 0, "new_by_class": {}, "classes_now": {}}
    new = s[(s["first"] > t_from) & (s["first"] <= t_to) & (~s["tid"].isin(seen_before))]
    cur = s[(s["last"] >= t_from) & (s["first"] <= t_to)]
    return {"new_tracks": int(len(new)), "new_by_class": {k: int(v) for k, v in new["name"].value_counts().items()},
            "classes_now": {k: int(v) for k, v in cur["name"].value_counts().items()}}


def window_counts(tr: pk.Tracks, t0: float, t1: float) -> dict[str, dict[str, float]]:
    """Per class over [t0, t1]: median and peak simultaneous count per sampled frame (YOLO26s + ByteTrack, cached)."""
    r = tr.rows[(tr.rows["t"] >= t0) & (tr.rows["t"] <= t1)]
    if r.empty:
        return {}
    nfr = r["f"].nunique()
    out = {}
    for nm, g in r.groupby("name"):
        per = g.groupby("f").size().reindex(r["f"].unique(), fill_value=0)
        out[nm] = {"median": float(per.median()), "peak": int(per.max()), "frames_present": round(float((per > 0).sum() / nfr), 2)}
    return out


# --------------------------------------------------------------------------------------------- claims / AMBER

NUMW = {"a": 1, "an": 1, "one": 1, "single": 1, "lone": 1, "two": 2, "pair": 2, "couple": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "dozen": 12}
PLURAL_MIN = {"several": 3, "many": 4, "crowd": 6, "crowds": 6, "group": 3, "groups": 3, "dozens": 12, "few": 3, "some": 2,
              "numerous": 5, "hundreds": 20, "throng": 6}
NOUNS = {  # surface form -> (COCO class, plural?)
    "person": ("person", 0), "people": ("person", 1), "man": ("person", 0), "men": ("person", 1), "woman": ("person", 0),
    "women": ("person", 1), "boy": ("person", 0), "boys": ("person", 1), "girl": ("person", 0), "girls": ("person", 1),
    "child": ("person", 0), "children": ("person", 1), "kid": ("person", 0), "kids": ("person", 1),
    "pedestrian": ("person", 0), "pedestrians": ("person", 1), "figure": ("person", 0), "figures": ("person", 1),
    "shopper": ("person", 0), "shoppers": ("person", 1), "passersby": ("person", 1), "commuters": ("person", 1),
    "crowd": ("person", 1), "crowds": ("person", 1), "cyclist": ("person", 0), "cyclists": ("person", 1),
    "driver": ("person", 0), "customer": ("person", 0), "customers": ("person", 1), "worker": ("person", 0),
    "workers": ("person", 1), "couple": ("person", 1), "pair": ("person", 1),
    "car": ("car", 0), "cars": ("car", 1), "taxi": ("car", 0), "taxis": ("car", 1), "vehicle": ("car", 0), "vehicles": ("car", 1),
    "van": ("car", 0), "vans": ("car", 1), "suv": ("car", 0), "sedan": ("car", 0),
    "truck": ("truck", 0), "trucks": ("truck", 1), "lorry": ("truck", 0), "lorries": ("truck", 1),
    "bus": ("bus", 0), "buses": ("bus", 1), "bicycle": ("bicycle", 0), "bicycles": ("bicycle", 1), "bike": ("bicycle", 0),
    "bikes": ("bicycle", 1), "motorcycle": ("motorcycle", 0), "motorbike": ("motorcycle", 0),
    "dog": ("dog", 0), "dogs": ("dog", 1), "bird": ("bird", 0), "birds": ("bird", 1), "horse": ("horse", 0),
    "backpack": ("backpack", 0), "umbrella": ("umbrella", 0), "umbrellas": ("umbrella", 1),
}
# classes that YOLO may legitimately report for a described noun (a van is often 'car' or 'truck', a cyclist is a person...)
ALIASES = {"car": ["car", "truck", "bus"], "truck": ["truck", "car", "bus"], "bus": ["bus", "truck"], "person": ["person"],
           "bicycle": ["bicycle"], "motorcycle": ["motorcycle", "bicycle"]}
SKIP_PREV = {"of", "with", "in", "on", "by", "near", "into", "past", "behind", "beside", "from"}


def parse_claims(line: str) -> list[dict]:
    """Count claims in a draft: [{cls, n, exact, text}]. 'two men' -> person n=2 exact; 'a car' -> car n=1 exact;
    'people' / 'cars' alone -> plural, n>=2 (not exact); 'a crowd' / 'several people' -> n>=min, not exact.
    A noun preceded by 'the' or by nothing singular is not a numeric claim (definite reference to a known thing)."""
    words = re.findall(r"[a-z]+|\d+", line.lower())
    claims: list[dict] = []
    for i, w in enumerate(words):
        if w not in NOUNS:
            continue
        cls, plural = NOUNS[w]
        prev = words[max(0, i - 3): i]
        n, exact, text = None, False, w
        for j in range(len(prev) - 1, -1, -1):  # nearest determiner / number before the noun (adjectives allowed between)
            p = prev[j]
            if p in NUMW and not (p in ("pair", "couple") and plural == 0):
                n, exact, text = NUMW[p], p not in ("pair", "couple", "dozen", "a", "an"), " ".join(prev[j:] + [w])
                break
            if p.isdigit():
                n, exact, text = int(p), True, " ".join(prev[j:] + [w])
                break
            if p in PLURAL_MIN:
                n, exact, text = PLURAL_MIN[p], False, " ".join(prev[j:] + [w])
                break
            if p in ("the", "his", "her", "their", "its", "this", "that", "these", "those"):
                break
        if n is None and plural and w not in ("pair", "couple"):
            n, exact = (PLURAL_MIN.get(w, 2)), False
        if n is None and w in ("crowd", "crowds"):
            n, exact = 6, False
        if n is not None:
            claims.append({"cls": cls, "n": int(n), "exact": bool(exact), "text": text})
    # 'another' / 'the other' adds one more of the last-mentioned class
    for m in re.finditer(r"\b(another|an other|the other)\b", line.lower()):
        if claims:
            claims.append({"cls": claims[-1]["cls"], "n": 1, "exact": claims[-1]["exact"], "text": m.group(0)})
    # same class mentioned several times ('a man ... a woman') = people add up; any vague mention makes the total vague
    merged: dict[str, dict] = {}
    for c in claims:
        m = merged.get(c["cls"])
        if m is None:
            merged[c["cls"]] = dict(c)
        else:
            m["n"] += c["n"]
            m["exact"] = m["exact"] and c["exact"]
            m["text"] += " + " + c["text"]
    return list(merged.values())


def evidence_range(cls: str, ev_s: list[dict[str, int]], ev_l: list[dict[str, int]]) -> tuple[int, int]:
    """(lo, hi) of per-keyframe counts for a class (aliases summed) across both detectors: YOLO26s tracks + YOLO26l."""
    vals = []
    for ev in ev_s + ev_l:
        vals.append(sum(ev.get(a, 0) for a in ALIASES.get(cls, [cls])))
    return (min(vals), max(vals)) if vals else (0, 0)


def check_line(line: str, ev_s: list[dict[str, int]], ev_l: list[dict[str, int]], confidence: str | None,
               prev_line: str | None) -> dict:
    """AMBER when a count word disagrees with YOLO, when the model hedges/low-confidence, or when the line repeats."""
    claims = parse_claims(line)
    reasons, checked, agreed = [], 0, 0
    for c in claims:
        lo, hi = evidence_range(c["cls"], ev_s, ev_l)
        tol = 1 if max(c["n"], hi) <= 4 else int(math.ceil(0.3 * max(c["n"], hi)))
        checked += 1
        if c["exact"]:
            ok = hi >= 1 and (lo - tol) <= c["n"] <= (hi + tol)  # a claimed class the detectors never see is a disagreement
        else:  # plural / vague: needs at least that many, within tolerance, and not zero
            ok = hi + tol >= c["n"] and hi >= 1
        if ok:
            agreed += 1
        else:
            reasons.append(f"line says '{c['text']}' ({c['cls']} n{'=' if c['exact'] else '>='}{c['n']}) but YOLO counts "
                           f"{lo}-{hi} across keyframes (s+l)")
    if confidence == "low":
        reasons.append("model reported low confidence")
    if HEDGES.search(line):
        reasons.append("hedged wording (" + HEDGES.search(line).group(0).lower() + ")")
    if prev_line and _jaccard(line, prev_line) > 0.7:
        reasons.append("near-repeat of the previous line")
    return {"flag": final_flag(bool(reasons), checked), "reasons": reasons, "claims": claims,
            "claims_checked": checked, "claims_agreed": agreed, "verified": checked > 0}


def final_flag(has_reasons: bool, n_checked: int) -> str:
    """AMBER = something disagrees or the model is unsure. GREEN = at least one claim in the line was CHECKED against the
    detectors (count word or colour word) and every checked claim agreed. UNVERIFIED = nothing checkable: no trust signal."""
    if has_reasons:
        return "AMBER"
    return "GREEN" if n_checked > 0 else "UNVERIFIED"


def _jaccard(a: str, b: str) -> float:
    A, B = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    return len(A & B) / max(1, len(A | B))


# --------------------------------------------------------------------------------------------- text fitting

LEAD = re.compile(r"^((in|from) (this|the) (image|frame|scene|video|shot)[, ]+|the (following|subsequent|next|first|second|last) "
                  r"(image|frame|picture)s?( is| are| shows?| depicts?)?[^.]*?(based on [^.,]*[,.]\s*)?|the (image|frame|video|scene) "
                  r"(shows|depicts|features)\s+|this (image|frame|scene|video) (shows|depicts)\s+|we (see|can see)\s+|there (is|are)\s+)", re.I)
META = re.compile(r"\b(image|images|frame|frames|picture|photograph|based on|confidence|the sentence|json|the camera captures|"
                  r"camera captures|the video)\b", re.I)


def is_meta(line: str) -> bool:
    """Replies that talk about the task or the pictures instead of describing the scene."""
    return bool(META.search(line))


def clean(line: str) -> str:
    line = line.strip().strip('"').strip("'").replace("\n", " ")
    line = re.sub(r"\s+", " ", line)
    line = LEAD.sub("", line).strip()
    line = re.sub(r"\([^)]*\)", "", line).strip()
    if line:
        line = line[0].upper() + line[1:]
    if line and line[-1] not in ".!?":
        line += "."
    return line


def n_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'\-]+", s))


STOP_END = {"a", "an", "the", "in", "on", "of", "and", "with", "to", "at", "by", "from", "as", "while", "his", "her", "their", "its",
            "into", "onto", "across", "through", "along", "beside", "near", "behind", "over", "under", "up", "down", "toward",
            "towards", "that", "which", "who", "is", "are", "but", "or", "for", "than", "then", "some", "two", "three", "one"}
BOUNDARY = {"in", "on", "with", "beside", "near", "behind", "across", "through", "along", "as", "while", "and", "then", "who", "which",
            "that", "toward", "towards", "into", "onto", "from", "at", "by", "over", "under", "before", "after", "but", "against"}


def trim_to_words(line: str, w: int) -> str | None:
    """Shorten to <= w words by cutting at a clause / prepositional boundary. None when no cut leaves a sentence with a verb
    and no dangling article or preposition (better to stay silent than to voice 'A man in a.')."""
    if n_words(line) <= w:
        return line
    words = line.rstrip(".").split()
    for i in range(min(w, len(words)), 1, -1):
        last = words[i - 1].rstrip(",;:").lower()
        nxt = words[i].lower() if i < len(words) else ""
        if last in STOP_END or not (nxt in BOUNDARY or words[i - 1].endswith(",")):
            continue
        cand = " ".join(words[:i]).rstrip(",;:") + "."
        if has_verb(cand) and not ends_badly(cand):
            return cand
    return None


VERBS = set("""stands stand walks walk runs run climbs climb looks look sits sit holds hold turns turn moves move leans lean reaches reach
falls fall jumps jump crosses cross enters enter exits exit approaches approach waits wait faces face points point stares stare
glances glance pulls pull pushes push carries carry lifts lift opens open closes close drives drive passes pass stops stop
smiles smile talks talk speaks speak shouts shout waves wave dances dance flies fly floats float swings swing steps step
rushes rush hurries hurry crawls crawl kneels kneel lies lie sleeps sleep watches watch gazes gaze glows glow appears
emerges emerge descends descend rises rise comes come goes go leaves leave arrives arrive gathers gather streams stream
pans cuts zooms fades flashes flicker flickers hangs hang stretches stretch spreads spread surrounds surround fills fill
rolls roll slides slide heads head follows follow chases chase grabs grab drops drop throws throw""".split())


NOT_VERB_PREV = set("""a an the his her their its this that these those some many several more other another two three four five six seven eight nine
ten of in on with by at from to into onto over under near beside behind across through along and or dark white black red blue green grey gray silver brown
yellow orange pink purple tall large small big little tattered metallic bright busy old young long short wide narrow distant tangled illuminated""".split())
BE = {"is", "are", "was", "were", "remains", "remain", "stays", "stay", "seems", "becomes", "stand", "walk", "run", "sit", "wait"}


def has_verb(line: str) -> bool:
    """Generous verb test (lexicon, be-forms, -ing/-ed, or an -s word that does not follow a determiner/adjective/preposition)."""
    ws = re.findall(r"[a-z]+", line.lower())
    for i, w in enumerate(ws):
        if w in VERBS or w in BE or (w.endswith("ing") and len(w) > 5) or (w.endswith("ed") and len(w) > 4 and i > 1 and ws[i - 1] not in NOT_VERB_PREV):
            return True
        if (i > 0 and len(w) >= 4 and w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")) and ws[i - 1] not in NOT_VERB_PREV
                and w not in NOUNS):
            return True
    return False


def ends_badly(line: str) -> bool:
    ws = re.findall(r"[a-z]+", line.lower())
    return not ws or (ws[-1] in STOP_END and ws[-1] != "her") or ws[-1] in {"narrow", "wide", "tall", "dark", "bright", "large", "small", "big", "long", "short", "old", "young",
                                                      "illuminated", "distant", "cracked", "shattered", "tattered", "tiled", "metal", "metallic"}


def shorten(line: str, w: int) -> str | None:
    """Text-only rewrite by the same model: keep subject + action, at most w words. None if it fails or does not comply."""
    for k, extra in enumerate(("", " Hard limit: count the words, there must be no more than %d. Example of the style: 'A woman crosses a bridge.'" % w)):
        raw = pk.vlm(f"Shorten this audio-description sentence to at most {w} words. Keep the main subject and the action, drop the details. "
                     f"Keep it a complete sentence in the present tense.{extra} Reply with the sentence only. Sentence: {line}", max_tokens=60)
        out = clean(raw.splitlines()[0]) if raw.strip() else ""
        if out and 2 <= n_words(out) <= w and has_verb(out) and not ends_badly(out):
            return out
    return None


def fallback_line(ev: dict[str, int]) -> str:
    """Template line from detector counts only (used when the VLM is down or returns nothing)."""
    nm = {"person": "person", "car": "car", "truck": "truck", "bus": "bus", "bicycle": "bicycle", "dog": "dog"}
    parts = []
    for k, v in sorted(ev.items(), key=lambda kv: -kv[1]):
        if k in nm and v > 0:
            parts.append(f"{v} {nm[k]}{'s' if v > 1 else ''}")
    return ("Scene with " + " and ".join(parts[:2]) + ".") if parts else "The scene changes."


# --------------------------------------------------------------------------------------------- VLM

PROMPT = ("You are writing a first draft of audio description for blind and partially sighted viewers (UK broadcast style). "
          "The {k} image{s} are consecutive moments, in order, from one short stretch of a {genre}. "
          "Write ONE sentence in the present tense, AT MOST {w} WORDS: who or what, doing what, and where. {style}"
          "Describe only what you can clearly see. If you are unsure of a detail (gender, clothing, objects, colours), leave it out and keep the sentence simple. "
          "No filler: never write 'remains static', 'throughout the scene', 'appears', 'seems' or 'in the scene'. "
          "Do not mention sound or dialogue, do not guess names, and never mention images, frames, the camera or the video. "
          "Give a number only if you can count it clearly. "
          "{prev}"
          'Reply as JSON: {{"line": "<sentence>", "confidence": "high" or "medium" or "low"}}')
STYLE_SHORT = "The time is very short: give only the main subject and what it is doing. "
STYLE_LONG = ""

# VLM filler that an AD writer would delete; stripped from drafts (a draft that is nothing but filler is dropped)
FILLER = re.compile(r"[,;]?\s*\b(remains?|stays?|is|are) (static|still|stationary|motionless|unchanged)( throughout( the)?( scene| video| clip| shot)?)?\b"
                    r"|[,;]?\s*\bthroughout( the)?( entire| whole)?( scene| video| clip| shot| frame)\b|[,;]?\s*\bin the scene\b", re.I)


def is_title(line: str) -> bool:
    """On-screen text read out ('Title: ...'): a valid AD line without a verb."""
    return bool(re.match(r"^\s*(title|on screen|caption)\s*:", line, re.I))


def strip_filler(line: str) -> str:
    """Delete VLM filler ('remains static throughout the scene'). Returns '' when what is left is not a complete sentence."""
    if not FILLER.search(line):
        return line
    out = re.sub(r"\s+", " ", FILLER.sub("", line)).strip(" ,;")
    out = re.sub(r"[,;]?\s*\b(and|while|but|as)( the (scene|video|clip|shot))?\.?$", "", out).strip(" ,;")
    out = clean(out) if out else ""
    return out if out and has_verb(out) and not ends_badly(out) else ""


def describe(frames: list[np.ndarray], w: int, prev_line: str | None, genre: str) -> tuple[str | None, str | None, str]:
    prev = f'The previous description was "{prev_line}"; say what is NEW or has changed, do not repeat it. ' if prev_line else ""
    p = PROMPT.format(k=len(frames), s="s" if len(frames) > 1 else "", w=w, prev=prev, genre=genre,
                      style=STYLE_SHORT if w <= 7 else STYLE_LONG)
    if len(frames) == 1:  # one picture: no "consecutive moments", and no motion verbs the picture cannot show
        p = p.replace("The 1 image are consecutive moments, in order, from one short stretch of a", "The image is one moment from a short stretch of a")
        p = p.replace("Describe only what you can clearly see.", "Describe only what you can clearly see; give an action only if the picture shows it clearly (otherwise say where someone is or what they hold).")
    raw = pk.vlm(p + " Reply with JSON only.", frames, max_tokens=120)  # ONE call (vlm_json would repeat it)
    if not raw:
        return None, None, ""
    d = None
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
        except Exception:
            d = None
    if isinstance(d, dict):  # the model sometimes names the key 'sentence' / 'description'
        d = {**d, "line": next((d[k] for k in ("line", "sentence", "description", "text") if d.get(k)), "")}
    if isinstance(d, dict) and d.get("line"):
        ln = strip_filler(clean(str(d["line"])))
        return (None if (not ln or is_meta(ln) or not (has_verb(ln) or is_title(ln))) else ln), str(d.get("confidence", "")).lower() or None, raw
    m = re.search(r'"line"\s*:\s*"([^"]+)', raw)
    if m:
        ln = strip_filler(clean(m.group(1)))
        return (ln or None), None, raw
    ln = strip_filler(clean(raw.splitlines()[0].strip("{} ")))
    return (None if (not ln or is_meta(ln)) else ln), None, raw


TEXT_PROMPT = ("Read the words that are written on this picture (a title, caption or sign that fills part of the picture). "
               "Reply with the exact words only, in reading order, on one line. If there are no readable words, reply NONE.")


def read_text(frame: np.ndarray) -> str | None:
    """On-screen title/caption text via the VLM (one call), or None. A real describer voices a title card first.
    Rejects replies that are not plain words (JSON, sentences about the picture, 'NONE')."""
    raw = pk.vlm(TEXT_PROMPT, [frame], max_tokens=40)
    if not raw:
        return None
    txt = re.sub(r"\s+", " ", raw.strip().splitlines()[0]).strip(" \"'`*.")
    if not txt or re.search(r"\bnone\b|no (readable )?(words|text)|not visible|cannot|can't|unable|sorry", txt, re.I) or len(txt) > 70 or is_meta(txt):
        return None
    if n_words(txt) > 9 or not re.search(r"[A-Za-z]{3}", txt):
        return None
    return txt


def title_line(txt: str) -> str:
    """'the orange open movie project presents' -> 'Title: The orange open movie project presents.'"""
    txt = txt.strip(" .")
    return "Title: " + txt[0].upper() + txt[1:] + "."


# --------------------------------------------------------------------------------------------- colour words vs pixels

CAR_NOUNS = {"car", "cars", "van", "vans", "truck", "trucks", "taxi", "taxis", "bus", "buses", "vehicle", "vehicles", "suv", "sedan"}
GARMENTS = {"shirt", "top", "dress", "jacket", "coat", "t-shirt", "jumper", "hoodie", "blouse", "sweater", "cloak", "suit"}
COLOUR_WORDS = {"black", "white", "grey", "gray", "silver", "red", "blue", "green", "yellow", "orange", "pink", "purple", "brown"}


def colour_claims(line: str) -> list[dict]:
    """'a blue car' -> {cls: car, colour: blue}; 'a woman in a pink top' -> {cls: person, colour: pink}."""
    ws = re.findall(r"[a-z\-]+", line.lower())
    out = []
    for i, w in enumerate(ws):
        if w not in COLOUR_WORDS:
            continue
        nxt = ws[i + 1: i + 4]
        if any(n in CAR_NOUNS for n in nxt[:2]):
            out.append({"cls": "car", "colour": w, "text": " ".join(ws[max(0, i - 1): i + 3])})
        elif any(n in GARMENTS for n in nxt[:2]):
            out.append({"cls": "person", "colour": w, "text": " ".join(ws[max(0, i - 2): i + 3])})
    return out


def colour_fracs(frame: np.ndarray, box, cls: str) -> dict[str, float]:
    """Fraction of pixels per named colour inside a detection box (torso band for people, central body for vehicles)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if cls == "person":
        x1, x2, y1, y2 = x1 + 0.2 * bw, x2 - 0.2 * bw, y1 + 0.18 * bh, y1 + 0.55 * bh
    else:
        x1, x2, y1, y2 = x1 + 0.1 * bw, x2 - 0.1 * bw, y1 + 0.25 * bh, y2 - 0.05 * bh
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return {}
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    H, S, V = hsv[:, 0] * 2, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    n = len(H)
    sat = (S > 0.35) & (V > 0.25)
    f = {
        "black": float((V < 0.22).mean()), "white": float(((S < 0.18) & (V > 0.72)).mean()),
        "grey": float(((S < 0.2) & (V >= 0.22) & (V <= 0.72)).mean()),
        "red": float((sat & ((H < 12) | (H > 345))).mean()), "orange": float((sat & (H >= 12) & (H < 40) & (V > 0.55)).mean()),
        "brown": float((sat & (H >= 8) & (H < 40) & (V <= 0.55)).mean()), "yellow": float((sat & (H >= 40) & (H < 70)).mean()),
        "green": float((sat & (H >= 70) & (H < 170)).mean()), "blue": float((sat & (H >= 170) & (H < 260)).mean()),
        "purple": float((sat & (H >= 260) & (H < 300)).mean()), "pink": float((((S > 0.2) & (V > 0.5)) & ((H >= 300) | (H < 12))).mean()),
    }
    f["silver"] = max(f["grey"], f["white"] * 0.8)
    f["gray"] = f["grey"]
    return f


def check_colours(line: str, frames: list[np.ndarray], boxes_per_frame: list[list[tuple[str, list[float]]]]) -> tuple[list[str], int]:
    """For each colour claim: does ANY detected box of that class carry >= 18 % of that colour in some keyframe?
    Returns (reasons for disagreement, number of claims that could be checked). Claims with no detected box cannot be checked."""
    reasons, checked = [], 0
    for c in colour_claims(line):
        aliases = ALIASES.get(c["cls"], [c["cls"]])
        best, seen, n_boxes = 0.0, {}, 0
        for fr, boxes in zip(frames, boxes_per_frame):
            for nm, bx in boxes:
                if nm not in aliases:
                    continue
                f = colour_fracs(fr, bx, c["cls"])
                if not f:
                    continue
                n_boxes += 1
                best = max(best, f.get(c["colour"], 0.0))
                for k in ("black", "white", "grey", "red", "blue", "green", "yellow", "brown", "pink", "orange", "purple"):
                    seen[k] = max(seen.get(k, 0.0), f[k])
        if n_boxes == 0:
            continue
        checked += 1
        if best < 0.18:
            top = ", ".join(f"{k} {v:.0%}" for k, v in sorted(seen.items(), key=lambda kv: -kv[1])[:3])
            reasons.append(f"line says '{c['text']}' but no detected {c['cls']} is {c['colour']} (best {best:.0%}; seen {top})")
    return reasons, checked


# --------------------------------------------------------------------------------------------- left / right words vs boxes

SIDE = re.compile(r"\b(?:on|to|at) the (left|right)\b|\b(?:left|right) of the (?:frame|road|picture)\b", re.I)


def position_claims(line: str) -> list[dict]:
    """'A white car drives past on the right.' -> [{cls: car, side: right}]: the last class noun before 'on the left/right'."""
    out = []
    for m in SIDE.finditer(line):
        side = (m.group(1) or ("left" if "left" in m.group(0).lower() else "right")).lower()
        before = re.findall(r"[a-z]+", line[: m.start()].lower())
        cls = next((NOUNS[w][0] for w in reversed(before) if w in NOUNS), None)
        if cls:
            out.append({"cls": cls, "side": side, "text": f"{cls} on the {side}"})
    return out


def check_positions(line: str, boxes_per_frame: list[list[tuple[str, list[float]]]], width: float) -> tuple[list[str], int]:
    """Does ANY detected box of that class (aliases) sit on the claimed side (centre x < 45 % / > 55 % of the width) in some keyframe?
    Claims with no detected box of the class cannot be checked here (the count check already flags a class that is never seen)."""
    reasons, checked = [], 0
    for c in position_claims(line):
        xs = [((bx[0] + bx[2]) / 2) / width for boxes in boxes_per_frame for nm, bx in boxes if nm in ALIASES.get(c["cls"], [c["cls"]])]
        if not xs:
            continue
        checked += 1
        ok = any(x < 0.45 for x in xs) if c["side"] == "left" else any(x > 0.55 for x in xs)
        if not ok:
            reasons.append(f"line says '{c['text']}' but detected {c['cls']}s are at x={min(xs):.0%}-{max(xs):.0%} of the frame width")
    return reasons, checked
