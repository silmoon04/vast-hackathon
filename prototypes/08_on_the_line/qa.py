"""Transcript line -> intents -> answer card, from cached scene state only (no model call on the answer path).

On the day the transcript line comes from Canary ASR; here it is typed. Parsing is plain keyword rules so that a
card never depends on an LLM; Cosmos only adds the optional 'description of the best crop' line.
"""
from __future__ import annotations

# make the module importable on its own (protokit lives in prototypes/)
import sys as _sys  # noqa: E402
from pathlib import Path as _P  # noqa: E402
_HERE_PATHS = [str(_P(__file__).resolve().parents[1]), str(_P(__file__).resolve().parent)]
_sys.path[:0] = [x for x in _HERE_PATHS if x not in _sys.path]

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import protokit as pk
from engine import GROUP_PLURAL, GROUP_WORD, Entity, SceneState, display_colour

COLOUR_FAMILY = {"white": ["white"], "silver": ["silver", "grey", "gray", "white"], "grey": ["grey", "gray", "silver"],
                 "charcoal": ["grey", "gray", "dark", "black", "charcoal"], "black": ["black", "dark"],
                 "navy": ["navy", "dark blue", "blue", "dark"], "blue": ["blue"], "red": ["red"], "maroon": ["maroon", "red", "burgundy", "dark red"],
                 "orange": ["orange", "yellow"], "yellow": ["yellow", "orange"], "green": ["green"], "teal": ["teal", "green", "blue", "turquoise"],
                 "brown": ["brown", "tan", "beige", "bronze", "gold", "khaki", "grey", "gray"], "cream": ["cream", "beige", "white", "tan"],
                 "purple": ["purple", "violet"], "pink": ["pink"], "olive": ["olive", "green", "brown", "khaki", "tan"]}

PERSON_W = (r"person|people|persons|man|men|woman|women|guy|guys|lad|lads|suspect|suspects|pedestrian|pedestrians|individual|individuals|"
            r"someone|anyone|anybody|lady|ladies|kid|kids|child|children|male|males|female|females|rider|cyclist|customer|customers|body|bodies|"
            r"bloke|blokes|chap|chaps|fella|fellas|feller|geezer|geezers|gent|gents|youth|youths|offender|offenders|perp|perps|intruder|intruders|"
            r"burglar|burglars|thief|thieves|teen|teens|teenager|teenagers|boy|boys|girl|girls|folk|folks|crowd|heads|staff|shopper|shoppers")
VEH_W = (r"car|cars|vehicle|vehicles|van|vans|truck|trucks|lorry|lorries|taxi|taxis|motor|motors|saloon|saloons|hatchback|hatchbacks|driver|drivers|"
         r"motorbike|motorbikes|motorcycle|motorcycles|wagon|pickup|jeep|suv|4x4|bus|coach|minibus|cab|moped|scooter|wheels")
BIKE_W = r"bike|bikes|bicycle|bicycles"

RX = {
    "total": r"in total|altogether|since the alarm|so far|overall|all told|in all|total number|\btotal\b|since (it|the alarm) (went|started)|"
             r"have (there )?been|has there been|have (come|passed|gone|arrived|entered|turned up|driven|shown up|appeared)|"
             r"how many (different|separate|distinct|more)",
    "count": r"how many|number of|head ?count|\bhow big\b|more than one|just the one|just one|only one|(is|are) there (a |any |more )",
    "direction": r"which way|what (direction|way)|where did|where do|where have|where has|where (is|was|are|were|will|would) (he|she|it|they|them|the|that|those|this)|"
                 r"where\b.*\b(go|went|gone|going|head|headed|heading|run|ran|running|walk|walked|walking|drive|drove|driving|off|away)\b|"
                 r"headed|heading|direction|\bexit(ed)?\b|go(ne)? (off|towards|toward)|travelling|traveling|\btravel\b|head(ed)? off|made? off|make off|"
                 r"ran off|ran away|run(ning)? (off|away)|walk(ed|ing)? (off|away)|drove off|which (side|road|street|exit)",
    "appearance": r"wearing|\bwears?\b|\bwore\b|got on\b|dressed|look like|looks like|looking like|describe|description|colou?rs?|what kind|what type|"
                  r"what sort|clothes|clothing|outfit|what (make|model)|make and model|what vehicle|what car|"
                  r"what (is|was|are|were) (he|she|it|they)\b|what (does|did|do) (he|she|it|they)( \w+)? look",
    "still": r"\bstill\b|gone yet|left yet|has (he|she|it|they) gone|have (they|he|she) gone|\bis (he|she|it|the [a-z ]+) (gone|there)\b|"
             r"\bare they (gone|there)\b|\bany(one|body) (left|there|about|around)\b|(has|have) (he|she|it|they) (left|gone|cleared off)|"
             r"\bgone\b",
    "arrival": r"when did|what time (did|was|were)|arrive|arrived|turn(ed)? up|first (seen|appear)|how long (ago|has|have)|came in|come in|entered|"
               r"show(ed)? up|pull(ed)? up|got there|get there|appear(ed)?\b",
}

COL_MAP = {
    "white": {"white", "silver", "cream"}, "black": {"black", "charcoal"}, "grey": {"grey", "silver", "charcoal"},
    "gray": {"grey", "silver", "charcoal"}, "silver": {"silver", "grey", "white"}, "blue": {"blue", "navy", "teal"},
    "navy": {"navy", "blue"}, "red": {"red", "maroon"}, "maroon": {"maroon", "red"}, "burgundy": {"maroon", "red"},
    "green": {"green", "olive", "teal"}, "yellow": {"yellow", "orange", "cream"}, "orange": {"orange", "yellow", "brown"},
    "brown": {"brown", "olive", "orange", "maroon"}, "pink": {"pink", "red", "maroon"}, "purple": {"purple", "pink"},
    "beige": {"cream", "brown"}, "cream": {"cream", "white"}, "dark": {"black", "charcoal", "navy", "maroon", "olive"},
    "light": {"white", "silver", "cream", "grey"},
}
COL_RX = "|".join(sorted(COL_MAP, key=len, reverse=True))
TOP_G = r"jacket|coat|top|shirt|t-shirt|tshirt|tee|hoodie|jumper|sweater|sweatshirt|cardigan|blouse|dress|fleece|vest|parka|anorak"
BOT_G = r"trousers|jeans|pants|shorts|skirt|leggings|joggers|tracksuit bottoms|bottoms"
VEH_G = r"car|van|truck|lorry|taxi|motor|saloon|hatchback|vehicle|wagon|pickup|jeep|suv|bus|cab|coach"
ATTR_RX = re.compile(rf"\b({COL_RX})\s+(?:long[- ]sleeved?\s+|short[- ]sleeved?\s+)?({TOP_G}|{BOT_G}|{VEH_G}|one|ones)\b")
ATTR_IN_RX = re.compile(rf"\b(?:in|wearing)\s+(?:a\s+|the\s+)?({COL_RX})\b")
AGE_RX = re.compile(r"\b(child|children|kid|kids|boy|girl|toddler|baby|infant|elderly|old (man|woman|lady)|teen\w*|young|youth|middle-aged)\b")
KNOWN_COLOURS = set(COL_MAP) | {"tan", "khaki", "turquoise", "bronze", "gold"}


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def normalise(text: str) -> str:
    """Lower-case, expand the contractions ASR and police use (where'd, what's, they're), drop punctuation."""
    s = text.lower().replace("’", "'").replace("`", "'")
    s = re.sub(r"\bwon't\b", "will not", s)
    s = re.sub(r"\bcan't\b", "cannot", s)
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r"(?<=\w)'re\b", " are", s)
    s = re.sub(r"(?<=\w)'ve\b", " have", s)
    s = re.sub(r"(?<=\w)'ll\b", " will", s)
    s = re.sub(r"(?<=\w)'m\b", " am", s)
    s = re.sub(r"(?<=\w)'d\b", " did", s)      # where'd / how'd
    s = re.sub(r"(?<=\w)'s\b", " is", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Attr:
    cols: set[str]           # pixel colour names that satisfy the selector
    word: str                # what the caller said, e.g. "red jacket"
    part: str | None         # top | bottom | body | None (any)
    group: str | None


@dataclass
class Ask:
    text: str
    t: float
    intents: list[str]
    group: str | None
    selector: str            # first | latest | last_seen | last_departed | focus | attr
    pronoun: bool
    attr: Attr | None = None
    attr_missed: bool = False


def _attr_of(s: str) -> Attr | None:
    m = ATTR_RX.search(s)
    if m:
        col, noun = m.group(1), m.group(2)
        part = "top" if _has(rf"^({TOP_G})$", noun) else "bottom" if _has(rf"^({BOT_G})$", noun) else "body" if _has(rf"^({VEH_G})$", noun) else None
        grp = "vehicle" if part == "body" else "person" if part in ("top", "bottom") else None
        return Attr(COL_MAP[col], f"{col} {noun}" if part else col, part, grp)
    m = ATTR_IN_RX.search(s)
    if m:
        return Attr(COL_MAP[m.group(1)], m.group(1), None, "person")
    return None


def looks_like_question(text: str) -> bool:
    s = normalise(text)
    return text.strip().endswith("?") or _has(r"^(what|where|who|whom|which|when|how|is|are|was|were|did|do|does|can|could|any|tell|give|has|have|anyone|anybody)\b", s)


def parse(text: str, focus: Entity | None) -> Ask | None:
    s = normalise(text)
    intents = [k for k, rx in RX.items() if _has(rx, s)]
    if "total" in intents and "count" in intents:
        intents.remove("count")
    if "arrival" in intents and "appearance" in intents and not _has(r"wearing|colou?r|look like|describe", s):
        intents.remove("appearance")
    if "arrival" in intents and "total" in intents and not _has(r"\bwhen\b|what time|first|arrive", s):
        intents.remove("arrival")
    if _has(r"(is there|are there|can you see|is anyone|anybody|anyone) .*(anyone|anybody|someone|there|there now|on site|on scene|in view)", s) and not intents:
        intents.append("count")
    if _has(r"anyone (there|on site|on scene|around)|is there anyone|anybody there", s) and "count" not in intents and "still" not in intents:
        intents.append("count")
    if "still" in intents and "count" in intents and (_has(r"\bis (anyone|anybody)\b", s) or _has(r"\bhow many\b", s)):
        intents.remove("still")
    if "still" in intents and "count" not in intents and _has(r"\b(anyone|anybody|someone|somebody)\b", s):
        intents.remove("still")          # "is anybody still about?" asks how many are here now
        intents.append("count")
    if "still" in intents and "direction" in intents and _has(r"\bwhere\b", s):
        intents.remove("still")
    if "still" in intents and "arrival" in intents and _has(r"\bhow long\b", s):
        intents.remove("still")
    if not intents:
        return None
    attr = _attr_of(s)
    group = None
    if _has(rf"\b({PERSON_W})\b", s):
        group = "person"
    if _has(rf"\b({VEH_W})\b", s):
        group = "vehicle" if group is None else group
        if group == "person" and _has(r"\bdriver\b", s) and not _has(r"\b(man|woman|person|people)\b", s):
            group = "vehicle"
    if _has(rf"\b({BIKE_W})\b", s):
        group = "bicycle"
    if group is None and _has(r"\bwho\b", s):
        group = "person"
    if group is None and attr and attr.group:
        group = attr.group
    pronoun = _has(r"\b(he|she|it|they|him|her|them|his|their)\b", s)
    if group is None and pronoun and focus is not None and attr is None:
        group = focus.group
    if attr is not None:
        selector = "attr"
    elif _has(r"\b(first|initial|earliest)\b", s):
        selector = "first"
    elif _has(r"\bwho (just )?(left|went|has left)\b|\b(just|last) (to )?(left|leave|leaving|exit|exited)\b"
              r"|\b(just|last) (person|man|woman|one|car|vehicle|van) (to )?(left|leave|leaving)\b", s):
        selector = "last_departed"
    elif _has(r"\b(latest|newest|just arrived|most recent(ly)? arriv\w+|last (one )?to (arrive|come|turn up|show up))\b", s) \
            or ("arrival" in intents and _has(r"\blast (one|person|man|woman|car|vehicle|van|truck)\b", s)):
        selector = "latest"
    elif _has(r"\b(last|just|final)\b", s):
        selector = "last_seen"
    elif focus is not None and (pronoun or _has(r"\bthe (car|vehicle|van|truck|man|woman|person|bike|lady|suspect|driver|bloke|lad|guy)\b", s)) \
            and (group is None or focus.group == group):
        selector = "focus"
    else:
        selector = "last_seen"
    return Ask(text, 0.0, intents, group, selector, pronoun, attr)


# --------------------------------------------------------------------------------------------------------- cards


@dataclass
class Evidence:
    kind: str                 # crop | scene | strip
    t: float
    caption: str
    img: np.ndarray
    eid: int | None = None


@dataclass
class Card:
    question: str
    t: float
    intents: list[str]
    headline: str
    lines: list[str]
    evidence: list[Evidence]
    facts: dict[str, Any]
    latency_ms: float = 0.0
    caveats: list[str] = field(default_factory=list)
    target_eid: int | None = None


def _ago(x: float) -> str:
    return f"{x:.1f} s ago"


def _side(s: str | None) -> str:
    return {"left": "LEFT", "right": "RIGHT", "top": "TOP", "bottom": "BOTTOM"}.get(s or "", "?")


def _hd(label: str) -> str:
    return {"left": "LEFT", "right": "RIGHT", "up": "UP the frame (away)", "down": "DOWN the frame (towards camera)",
            "stationary": "not moving"}.get(label, label.replace("+", " and ").upper())


def _what(f: dict[str, Any]) -> str:
    c = f["colours"]
    if f["group"] == "person":
        top, bot = display_colour("person", c.get("top")), display_colour("person", c.get("bottom"))
        if top and bot:
            return f"person: {top} top, {bot} trousers"
        return f"person: {top or bot or 'colour unclear'}"
    if f["group"] == "vehicle":
        return f"{c.get('body', 'colour-unclear')} {f['cls']}"
    return "bicycle"


def _garment_colours(text: str, nouns: str) -> list[str]:
    """Colour words that Cosmos attaches to a garment / vehicle noun, e.g. 'a blue top' -> ['blue']."""
    out = []
    for m in re.finditer(rf"\b([a-z]+)(?:-colou?red)?\s+(?:(?:long|short)[- ]sleeved\s+|knit(?:ted)?\s+|striped\s+|checked\s+)?(?:{nouns})\b", text):
        w = m.group(1)
        if w in KNOWN_COLOURS and w not in ("dark", "light"):
            out.append(w)
    return out


def cosmos_conflict(desc: str, group: str, cols: dict[str, str]) -> str | None:
    """Cross-check a Cosmos sentence against the pixel colours (and the 'do not guess age' rule). Returns the reason the
    sentence should be withheld ('Cosmos says blue top, pixels say light grey'), or None. Cosmos is never shown contradicting
    the pixels without an amber flag; the check cannot tell which of the two is wrong."""
    d = desc.lower()
    if AGE_RX.search(d):
        return "Cosmos guessed an age"

    def clash(found: list[str], pixel: str | None, part: str) -> str | None:
        if not found or not pixel:
            return None
        fam = set(COLOUR_FAMILY.get(pixel, [pixel])) | {pixel}
        if any(w in fam for w in found):
            return None
        return f"Cosmos says {found[0]} {part}, pixels say {display_colour(group, pixel)}"
    mentioned = [w for w in sorted(KNOWN_COLOURS) if re.search(rf"\b{w}\b", d) and w not in ("dark", "light")]
    if group == "person":
        top_found, bot_found = _garment_colours(d, TOP_G), _garment_colours(d, BOT_G)
        why = clash(top_found, cols.get("top"), "top") or clash(bot_found, cols.get("bottom"), "trousers")
        if why is None and not top_found and not bot_found:
            why = clash(mentioned, cols.get("top"), "colour")
        return why
    if group == "vehicle":
        found = _garment_colours(d, VEH_G + "|hatchback|sedan|suv|pickup") or mentioned
        return clash(found, cols.get("body"), "car")
    return None


def cosmos_lines(tf: dict[str, Any], group: str) -> tuple[list[str], list[str], dict[str, Any]]:
    """The 'Cosmos' part of an appearance card: (lines, caveats, facts). Withheld when it contradicts the pixels."""
    if tf["desc"]:
        why = cosmos_conflict(tf["desc"], group, tf["colours"])
        if why:
            return [f"Cosmos text withheld ({why.replace('Cosmos ', '', 1)})"],                 ["descriptions disagree"], {"cosmos_withheld": why, "desc_raw": tf["desc"]}
        return [f"Cosmos: \"{tf['desc']}\""], [], {"desc": tf["desc"]}
    if tf["desc_pending"]:
        return ["Cosmos description still being generated."], [], {}
    if group != "bicycle":
        return ["No Cosmos description (colour from pixels only)."], [], {}
    return [], [], {}


def _crop_ev(e: Entity, cap_prefix: str) -> Evidence | None:
    if e.best is None:
        return None
    return Evidence("crop", e.best["t"], f"{cap_prefix} {pk.mmss(e.best['t'])}", e.best["img"].copy(), e.eid)


def _scene_ev(st: SceneState, ents: list[Entity], cap: str) -> Evidence | None:
    if st.frame_now is None:
        return None
    img = st.frame_now.copy()
    for e in ents:
        b = e.pts[-1][1:5]
        pk.box(img, b, pk.color(e.eid), f"#{e.eid}", 3)
    return Evidence("scene", st.now, cap, pk.fit(img, 420))


def _path_ev(st: SceneState, e: Entity, cap: str) -> Evidence | None:
    snap = st.snap_at(e.last_t)
    if snap is None:
        return None
    img = snap.copy()
    sx = img.shape[1] / st.w
    pts = np.array([[(p[1] + p[3]) / 2 * sx, (p[2] + p[4]) / 2 * sx] for p in e.pts], np.int32)
    import cv2
    if len(pts) >= 2:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.arrowedLine(img, tuple(pts[max(0, len(pts) - 8)]), tuple(pts[-1]), (0, 200, 255), 3, cv2.LINE_AA, tipLength=0.5)
    x1, y1, x2, y2 = [v * sx for v in e.pts[-1][1:5]]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    if len(pts) >= 2:  # crop to the path so the thumbnail is legible
        lo, hi = pts.min(axis=0) - 40, pts.max(axis=0) + 40
        ih, iw = img.shape[:2]
        lo, hi = np.clip(lo, 0, [iw, ih]), np.clip(hi, 0, [iw, ih])
        if hi[0] - lo[0] < 120:
            c = (lo[0] + hi[0]) / 2
            lo[0], hi[0] = max(0, c - 60), min(iw, c + 60)
        if hi[1] - lo[1] < 70:
            c = (lo[1] + hi[1]) / 2
            lo[1], hi[1] = max(0, c - 35), min(ih, c + 35)
        img = img[int(lo[1]):int(hi[1]), int(lo[0]):int(hi[0])]
    return Evidence("scene", e.last_t, cap, img, e.eid)


def attr_match(st: SceneState, e: Entity, attr: Attr) -> bool:
    """Does this entity's voted pixel colour satisfy 'the one in the red jacket'?"""
    if attr.group and e.group != attr.group:
        return False
    cols = st.colours(e)
    if attr.part:
        return cols.get(attr.part) in attr.cols
    return any(v in attr.cols for v in cols.values())


def pick_target(st: SceneState, ask: Ask, focus: Entity | None) -> Entity | None:
    cands = st.confirmed(ask.group) if ask.group else st.confirmed()
    if not cands:
        return None
    if ask.selector == "attr" and ask.attr is not None:
        hits = [e for e in cands if attr_match(st, e, ask.attr)]
        if hits:
            return max(hits, key=lambda e: (e.last_t, e.eid))
        ask.attr_missed = True          # nobody matches the colour: fall back to the latest, and say so on the card
    if ask.selector == "focus" and focus is not None and (ask.group is None or focus.group == ask.group):
        return focus
    if ask.selector == "first":
        return min(cands, key=lambda e: (e.first_t, e.eid))
    if ask.selector == "latest":
        return max(cands, key=lambda e: (e.first_t, e.eid))
    if ask.selector == "last_departed":
        gone = [e for e in cands if st.now - e.last_t > st.hold]
        if gone:
            return max(gone, key=lambda e: (e.last_t, e.eid))
    return max(cands, key=lambda e: (e.last_t, e.eid))


def _fallback_card(st: SceneState, text: str, t0: float) -> Card:
    """The transcript line was question-shaped but matched no intent: say so and show the latest scene summary instead of
    leaving the operator with a blank."""
    pres = {g: len(st.present(g)) for g in ("person", "vehicle", "bicycle") if any(e.group == g for e in st.confirmed())}
    tot = {g: len(st.confirmed(g)) for g in pres}
    gone = [e for e in st.confirmed() if st.now - e.last_t > st.hold]
    lines = ["In view now: " + (", ".join(f"{n} {GROUP_PLURAL[g] if n != 1 else GROUP_WORD[g]}" for g, n in pres.items()) or "nothing") + "."]
    if gone:
        last = max(gone, key=lambda e: e.last_t)
        f = st.facts(last)
        lines.append(f"Last to leave: {GROUP_WORD[last.group]} #{last.eid} at {pk.mmss(last.last_t)}"
                     + (f", via the {_side(f['exit_side'])} edge." if f["exit_side"] else "."))
    ev = []
    se = _scene_ev(st, st.present(), f"in view at {pk.mmss(st.now)}")
    if se:
        ev.append(se)
    card = Card(text, st.now, ["summary"], "Didn't catch that: here is the latest scene summary", lines, ev,
                {"summary": {"count_now": pres, "total": tot}}, caveats=["not understood: ask again"])
    card.latency_ms = (time.perf_counter() - t0) * 1000
    return card


def answer(st: SceneState, text: str, focus: Entity | None) -> tuple[Card | None, Entity | None]:
    """Answer one transcript line from cached state. Returns (card, new_focus)."""
    t0 = time.perf_counter()
    ask = parse(text, focus)
    if ask is None:
        if looks_like_question(text) and st.n_frames:
            return _fallback_card(st, text, t0), focus
        return None, focus
    ask.t = st.now
    facts: dict[str, Any] = {}
    lines: list[str] = []
    head: list[str] = []
    ev: list[Evidence] = []
    caveats: list[str] = []
    target: Entity | None = None
    needs_target = any(i in ask.intents for i in ("direction", "appearance", "still", "arrival"))
    if needs_target:
        target = pick_target(st, ask, focus)
        if target is None:
            word = GROUP_WORD.get(ask.group or "", "person or vehicle")
            card = Card(text, st.now, ask.intents, f"No {word} seen since the alarm", [
                f"Nothing confirmed on the camera since {pk.mmss(st.alarm_t)}."], [], {"none": True})
            card.latency_ms = (time.perf_counter() - t0) * 1000
            return card, focus
    if target is not None and ask.attr_missed and ask.attr is not None:
        caveats.append(f"no {ask.attr.word}: showing latest")
    tf = st.facts(target) if target else None
    who = None
    if target and tf:
        who = {"person": "person", "vehicle": tf["cls"], "bicycle": "bicycle"}[target.group]

    for intent in ask.intents:
        if intent == "count":
            groups = [ask.group] if ask.group else ["person", "vehicle"]
            parts = []
            for g in groups:
                pres = st.present(g)
                still = sum(1 for e in pres if st.facts(e)["stationary_since"] is not None)
                facts.setdefault("count_now", {})[g] = len(pres)
                parts.append(f"{len(pres)} {GROUP_PLURAL[g] if len(pres) != 1 else GROUP_WORD[g]}")
                if still:
                    lines.append("Not moving." if len(pres) == 1 else f"{still} of the {len(pres)} {GROUP_PLURAL[g]} not moving.")
                shown = sorted(pres, key=lambda e: -(e.last_img.shape[0] * e.last_img.shape[1] if e.last_img is not None else 0))[:4]
                for e in shown:
                    if e.last_img is not None:
                        ev.append(Evidence("crop", e.last_img_t, f"#{e.eid} now", e.last_img.copy(), e.eid))
                if not shown:
                    ev_e = _scene_ev(st, pres, f"in view at {pk.mmss(st.now)}")
                    if ev_e:
                        ev.append(ev_e)
            head.append(" and ".join(parts) + " in view now")
            if ask.group == "person" and facts["count_now"]["person"] >= 8:
                caveats.append("crowded: count +/-2")
        elif intent == "total":
            groups = [ask.group] if ask.group else ["person", "vehicle"]
            parts = []
            for g in groups:
                ents = st.confirmed(g)
                n_alarm = sum(1 for e in ents if e.at_alarm)
                n_new = len(ents) - n_alarm
                facts.setdefault("total", {})[g] = len(ents)
                facts.setdefault("total_new", {})[g] = n_new
                parts.append(f"{len(ents)} {GROUP_PLURAL[g] if len(ents) != 1 else GROUP_WORD[g]}")
                if n_alarm:
                    lines.append(f"{n_alarm} already on scene at the alarm ({pk.mmss(st.alarm_t)}), {n_new} arrived since.")
                elif ents:
                    lines.append(f"All {n_new} arrived after the alarm ({pk.mmss(st.alarm_t)}): "
                                 + ", ".join(f"#{e.eid} {pk.mmss(e.first_t)}" for e in sorted(ents, key=lambda x: x.first_t)[:6]))
                strip = [e for e in sorted(ents, key=lambda x: x.first_t) if e.best is not None][:6]
                for e in strip:
                    c = _crop_ev(e, f"#{e.eid} first seen")
                    c.caption = f"#{e.eid} {pk.mmss(e.first_t)}"
                    ev.append(c)
            carried = [e for e in st.confirmed("vehicle", include_carried=True) if e.carried]
            if carried and "vehicle" in groups:
                facts["carried"] = len(carried)
                lines.append(f"{len(carried)} vehicle(s) on a recovery truck not counted separately.")
            head.append(" and ".join(parts) + " since the alarm")
            if "person" in groups:
                caveats.append("double-count risk in crowds")
        elif intent == "direction" and target and tf:
            facts["direction"] = {"eid": target.eid, "heading": tf["heading"], "exit_side": tf["exit_side"],
                                  "entry_side": tf["entry_side"], "present": tf["present"]}
            if tf["present"]:
                head.append(f"The {who} is still in view, heading {_hd(tf['heading'])}")
            elif tf["exit_side"]:
                head.append(f"The {who} left via the {_side(tf['exit_side'])} edge, heading {_hd(tf['heading'])}")
            else:
                head.append(f"The {who} vanished mid-frame, last heading {_hd(tf['heading'])}")
                caveats.append("did not leave through an edge: may be hidden behind something, not confirmed gone")
            if tf["entry_side"]:
                lines.append(f"Came in via the {_side(tf['entry_side'])} edge at {pk.mmss(tf['first_t'])}.")
            if not tf["present"]:
                lines.append(f"Last seen {pk.mmss(tf['last_t'])} ({_ago(tf['last_seen_ago'])}).")
            c0 = _crop_ev(target, "who")
            if c0:
                ev.append(c0)
            e2 = _path_ev(st, target, f"path, last seen {pk.mmss(target.last_t)}")
            if e2:
                ev.append(e2)
        elif intent == "appearance" and target and tf:
            facts["appearance"] = {"eid": target.eid, "colours": tf["colours"], "cls": tf["cls"], "desc": None}
            if not tf["colours"]:
                head.append(f"A {who} (colour not yet clear)")
            else:
                head.append("A " + _what(tf))
            c_lines, c_cav, c_facts = cosmos_lines(tf, target.group)
            lines += c_lines
            caveats += c_cav
            facts["appearance"].update(c_facts)
            c = _crop_ev(target, "best crop")
            if c:
                ev.append(c)
            if st.frame_now is not None and float(st.frame_now.mean()) < 70:
                caveats.append("night: colours unreliable")
        elif intent == "still" and target and tf:
            facts["still"] = {"eid": target.eid, "present": tf["present"], "last_seen_ago": tf["last_seen_ago"]}
            if tf["present"]:
                extra = ""
                if tf["stationary_since"] is not None:
                    extra = f", not moving since {pk.mmss(tf['stationary_since'])}"
                head.append(f"YES: the {who} is in view now{extra}")
                lines.append(f"Last confirmed in frame {tf['last_seen_ago']:.1f} s ago, on scene {tf['dwell']:.0f} s.")
            else:
                side = f", left via the {_side(tf['exit_side'])} edge" if tf["exit_side"] else ", vanished mid-frame"
                head.append(f"NO: the {who} was last seen {_ago(tf['last_seen_ago'])}{side}")
                lines.append(f"Last seen {pk.mmss(tf['last_t'])}.")
                facts["still"]["exit_side"] = tf["exit_side"]
                if not tf["exit_side"]:
                    caveats.append("may still be on scene but hidden")
            c = _crop_ev(target, "last good view")
            if c:
                ev.append(c)
        elif intent == "arrival" and target and tf:
            facts["arrival"] = {"eid": target.eid, "first_t": tf["first_t"], "at_alarm": tf["at_alarm"]}
            if tf["at_alarm"]:
                head.append(f"The {who} was already on scene at the alarm ({pk.mmss(st.alarm_t)})")
            else:
                head.append(f"The {who} arrived at {pk.mmss(tf['first_t'])} (alarm +{tf['first_t'] - st.alarm_t:.0f} s)")
                if tf["entry_side"]:
                    lines.append(f"Came in via the {_side(tf['entry_side'])} edge.")
            lines.append(f"{'On scene for' if tf['present'] else 'Was in view for'} {tf['dwell']:.0f} s"
                         + ("" if tf["present"] else f", gone {_ago(tf['last_seen_ago'])}") + ".")
            c = _crop_ev(target, "first good view")
            if c and not any(x.kind == "crop" for x in ev):
                ev.append(c)
    card = Card(text, st.now, ask.intents, ". ".join(head), lines, ev, facts, caveats=caveats,
                target_eid=target.eid if target else None)
    card.latency_ms = (time.perf_counter() - t0) * 1000
    return card, target or focus
