"""Hide & Seek core: archive sampling, per-clip attribute extraction, and the information-gain question engine.

Everything here is deterministic given (seed, cached tracks, cached VLM replies). The question engine in
`Engine` is mirrored one-to-one by the JavaScript engine embedded in out/hide_and_seek.html; test_smoke.py runs
both and checks they agree.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import protokit as pk  # noqa: E402

OUT = HERE / "out"
WORK = OUT / "_work"

# the default test set (every clip named in the brief)
DEFAULT_VIDEOS = [
    "retail_cctv", "lisbon_parking", "ptz_parking", "rochester_night", "yonge_dundas", "motorway_a40",
    "dashcam_urban", "elephants_dream", "skate_fall", "floor_lie", "assist_elderly", "help_sit", "sit_down",
    "car_detection", "classroom", "face_demographics_walking_and_pause", "one_by_one_person_detection",
    "people_detection", "person_bicycle_car_detection", "store_aisle_detection", "worker_zone_detection",
]
SEG = 5.0
VEHICLES = ("car", "truck", "bus")
FURNITURE = ("chair", "couch", "bed", "dining table", "tv", "laptop", "potted plant", "book", "bottle", "cup",
             "bench", "refrigerator", "microwave", "oven", "sink", "toilet", "vase", "keyboard", "cell phone", "clock")
BAGS = ("handbag", "backpack", "suitcase", "umbrella")
TRAFFIC = ("traffic light", "stop sign", "fire hydrant", "parking meter")
ANIMALS = ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe")
SETTINGS = ["street", "road or motorway", "parking lot", "shop or store", "factory or workshop", "room at home or office",
            "classroom", "sports or outdoor activity", "nature", "animation or cartoon", "other"]

# --------------------------------------------------------------------------------------------- sampling


def seg_counts() -> dict[str, int]:
    out = {}
    for v in DEFAULT_VIDEOS:  # a trailing sliver (< 1 s) is not a clip
        d = pk.info(v)["duration"]
        out[v] = max(1, int(d // SEG) + (1 if d % SEG >= 1.0 else 0))
    return out


def allocate(n_total: int, counts: dict[str, int], cap: int = 8) -> dict[str, int]:
    """Per-video clip quota ~ sqrt(#segments), at least 1, at most `cap` or the number of segments (largest remainder)."""
    w = {v: math.sqrt(c) for v, c in counts.items()}
    scale = n_total / sum(w.values())
    q = {v: min(counts[v], cap, max(1, int(w[v] * scale))) for v in counts}
    while sum(q.values()) < n_total:
        room = [v for v in q if q[v] < min(counts[v], cap)]
        if not room:
            break
        v = max(room, key=lambda x: w[x] * scale - q[x])
        q[v] += 1
    while sum(q.values()) > n_total:
        v = max((v for v in q if q[v] > 1), key=lambda x: q[x] - w[x] * scale)
        q[v] -= 1
    return q


def sample_clips(n_total: int = 64, seed: int = 12) -> list[dict[str, Any]]:
    """Stratified random sample: each video's segments are split into `quota` equal time bins, one random segment per bin."""
    rng = np.random.default_rng(seed)
    counts = seg_counts()
    quota = allocate(n_total, counts)
    clips: list[dict[str, Any]] = []
    for v in DEFAULT_VIDEOS:
        n, q = counts[v], quota[v]
        edges = np.linspace(0, n, q + 1)
        for k in range(q):
            lo, hi = int(round(edges[k])), max(int(round(edges[k])) + 1, int(round(edges[k + 1])))
            seg = int(rng.integers(lo, min(hi, n)))
            dur = pk.info(v)["duration"]
            clips.append({"video": v, "seg": seg, "t0": seg * SEG, "t1": min((seg + 1) * SEG, dur)})
    for i, c in enumerate(clips):
        c["id"] = f"c{i + 1:02d}"
    return clips


# ------------------------------------------------------------------------------------------ track features


def _bucket(n: int, edges: tuple[int, ...], labels: tuple[str, ...]) -> str:
    for e, lab in zip(edges, labels):
        if n < e:
            return lab
    return labels[-1]


def track_features(tr: "pk.Tracks", t0: float, t1: float, cam_moving: bool) -> dict[str, Any]:
    """Everything the DataEngine YOLO table can give for one 5 s clip (confirmed tracks only)."""
    r = tr.rows[(tr.rows["t"] >= t0) & (tr.rows["t"] < t1) & (tr.rows["tid"] >= 0)]
    if not r.empty:
        n_by = r.groupby("tid").size()
        r = r[r["tid"].isin(n_by[n_by >= 3].index)]
    f: dict[str, Any] = {}
    if r.empty:
        peak = lambda *names: 0  # noqa: E731
    else:
        def peak(*names: str) -> int:
            s = r[r["name"].isin(names)]
            return int(s.groupby("f").size().max()) if len(s) else 0
    f["persons"] = _bucket(peak("person"), (1, 2, 4, 8), ("0", "1", "2-3", "4-7"))
    if peak("person") >= 8:
        f["persons"] = "8+"
    f["vehicles"] = _bucket(peak(*VEHICLES), (1, 2, 4), ("0", "1", "2-3"))
    if peak(*VEHICLES) >= 4:
        f["vehicles"] = "4+"
    f["bike"] = peak("bicycle", "motorcycle") > 0
    f["heavy_vehicle"] = peak("truck", "bus") > 0
    f["bags"] = peak(*BAGS) > 0
    f["furniture"] = peak(*FURNITURE) > 0
    f["traffic_furniture"] = peak(*TRAFFIC) > 0
    f["animal"] = peak(*ANIMALS) > 0
    f["n_tracks"] = int(r["tid"].nunique()) if len(r) else 0
    f["blind"] = len(r) == 0
    # motion: per moving track, displacement over the clip in units of its own size
    vec, mag = [], []
    still_veh = still_person = False
    if len(r):
        for tid, g in r.sort_values("t").groupby("tid"):
            dur = float(g["t"].iloc[-1] - g["t"].iloc[0])
            size = float(max(g["bw"].median(), g["bh"].median(), 1.0))
            dx, dy = float(g["cx"].iloc[-1] - g["cx"].iloc[0]), float(g["cy"].iloc[-1] - g["cy"].iloc[0])
            disp = math.hypot(dx, dy) / size
            name = g["name"].mode().iloc[0]
            if dur >= 1.0 and disp / max(dur, 1e-6) > 0.35:
                vec.append((dx, dy))
                mag.append(math.hypot(dx, dy))
            if dur >= 3.0 and disp < 0.3 and not cam_moving:
                still_veh |= name in VEHICLES
                still_person |= name == "person"
    f["stationary_vehicle"] = bool(still_veh)
    f["stationary_person"] = bool(still_person)
    if cam_moving:
        f["motion"] = "camera moving"
    elif not vec:
        f["motion"] = "nothing moves"
    else:
        sx, sy = sum(v[0] for v in vec), sum(v[1] for v in vec)
        if math.hypot(sx, sy) < 0.4 * sum(mag):
            f["motion"] = "mixed directions"
        elif abs(sx) >= abs(sy):
            f["motion"] = "left to right" if sx > 0 else "right to left"
        else:
            f["motion"] = "toward camera" if sy > 0 else "away from camera"
    return f


# ------------------------------------------------------------------------------------------ frame features

HUES = [(8, "red"), (20, "orange"), (35, "yellow"), (85, "green"), (100, "cyan"), (130, "blue"), (160, "purple"),
        (170, "pink"), (181, "red")]


def frame_features(video: str, t0: float, t1: float) -> tuple[dict[str, Any], np.ndarray]:
    """brightness class, dominant colour, camera-moving flag (median dense-flow magnitude) + the middle frame."""
    meta = pk.info(video)
    stride = max(1, int(round(meta["fps"] / 3)))
    frames = [f for _, _, f in pk.iter_frames(video, stride=stride, start=t0, end=t1)]
    if not frames:
        frames = [pk.frame_at(video, max(0.0, t0))]
    small = [cv2.resize(f, (160, int(160 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA) for f in frames]
    gray = [cv2.cvtColor(s, cv2.COLOR_BGR2GRAY) for s in small]
    luma = float(np.mean([g.mean() for g in gray]))
    hsv = np.concatenate([cv2.cvtColor(s, cv2.COLOR_BGR2HSV).reshape(-1, 3) for s in small])
    chroma = (hsv[:, 1] >= 60) & (hsv[:, 2] >= 50)
    if chroma.mean() < 0.2:
        colour = "dark" if luma < 60 else ("light" if luma > 170 else "grey")
    else:
        h = hsv[chroma, 0]
        hist = {name: 0 for _, name in HUES}
        for v in h:
            for e, name in HUES:
                if v < e:
                    hist[name] += 1
                    break
        colour = max(hist, key=hist.get)
    mags = []
    for a, b in zip(gray[:-1], gray[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 13, 2, 5, 1.1, 0)
        mags.append(float(np.median(np.hypot(fl[..., 0], fl[..., 1]))))
    cam = float(np.median(mags)) if mags else 0.0
    return ({"luma": round(luma, 1), "brightness": "dark" if luma < 62 else ("dim" if luma < 90 else "bright"),
             "colour": colour, "cam_flow": round(cam, 3), "camera_moving": cam > 0.5}, frames[len(frames) // 2])


# ------------------------------------------------------------------------------------------------ Cosmos

VLM_PROMPT = ("This is one frame from a short video clip. Reply as JSON with exactly these keys: "
              '"indoor_outdoor": "indoor" or "outdoor"; '
              '"setting": one of ' + json.dumps(SETTINGS) + '; "caption": one short factual sentence (max 15 words) '
              "describing what is in the frame.")


def _field(text: str, key: str) -> str:
    import re
    m = re.search(r'"%s"\s*:\s*"([^"]*)' % key, text)
    return m.group(1).strip() if m else ""


def cosmos_features(img: np.ndarray, use_vlm: bool = True) -> dict[str, Any]:
    """One Cosmos call per clip (mid-frame). Fields are pulled with a regex because the model sometimes truncates or
    mis-brackets its JSON (found on 6 of 64 clips with a strict parser); a truncated caption is kept as is."""
    out = {"indoor_outdoor": "unknown", "setting": "unknown", "caption": "", "vlm_ok": False}
    if not use_vlm:
        return out
    text = pk.vlm(VLM_PROMPT + "\nReply with JSON only.", [img], max_tokens=120)
    io_, st = _field(text, "indoor_outdoor").lower(), _field(text, "setting").lower()
    out["indoor_outdoor"] = io_ if io_ in ("indoor", "outdoor") else "unknown"
    out["setting"] = st if st in SETTINGS else "unknown"
    out["caption"] = _field(text, "caption")[:140]
    out["vlm_ok"] = out["indoor_outdoor"] != "unknown" or out["setting"] != "unknown"
    return out


# --------------------------------------------------------------------------------------------- questions

# attribute -> (kind, human label); ordinal attrs expose "at least" thresholds; the rest expose "is value"
ORD = {"persons": ["0", "1", "2-3", "4-7", "8+"], "vehicles": ["0", "1", "2-3", "4+"],
       "brightness": ["dark", "dim", "bright"]}
BOOL_Q = {"bike": "Does a bicycle or motorbike appear at any point?", "heavy_vehicle": "Does a truck or bus appear at any point?",
          "stationary_vehicle": "Does a vehicle stand still for most of the clip?", "stationary_person": "Does someone stand still for 3+ seconds?",
          "bags": "Does anyone carry a bag or umbrella?", "furniture": "Are chairs, tables, screens or bottles visible?",
          "traffic_furniture": "Is there a traffic light, stop sign or hydrant?", "animal": "Is there an animal?"}
# Caption keywords: a fixed lexicon of visible things (written after reading the corpus caption word counts, BEFORE looking at which
# clips are twins; a word becomes a question only if 3 <= clips mentioning it <= N-3). Matched against the Cosmos caption.
KEYWORDS = {
    "trees": (r"\btrees?\b|\bgreenery\b|\bgrass", "Are there trees or greenery?"),
    "buildings": (r"\bbuildings?\b", "Are buildings visible?"),
    "shelves": (r"\bshel(f|ves)\b|\bproducts?\b|\bdishes\b|\bpottery\b", "Are shelves or stocked products visible?"),
    "road": (r"\broads?\b|\blanes?\b|\bhighway\b|\bmotorway\b|\basphalt\b", "Is a road or tarmac surface prominent?"),
    "parking": (r"\bparking\b|\bparked\b", "Is it a parking lot or parked cars?"),
    "sky": (r"\bsky\b", "Is the sky visible?"),
    "sign": (r"\bsigns?\b", "Is a sign visible?"),
    "floor": (r"\bfloors?\b", "Is a floor prominent?"),
    "hallway": (r"\bhallway\b|\bcorridor\b", "Is it a hallway or corridor?"),
    "woman": (r"\bwom[ae]n\b|\bgirl\b|\blady\b", "Is a woman visible?"),
    "man": (r"\bm[ae]n\b|\bboy\b", "Is a man visible?"),
    "walking": (r"\bwalk", "Is someone walking?"),
    "sitting": (r"\bsit|\bseated\b|\bchairs?\b", "Is someone sitting?"),
    "lying": (r"\blying\b|\bbed\b|\blies\b", "Is someone lying down or in bed?"),
    "empty": (r"\bempty\b|\bdeserted\b|\bquiet\b", "Is the place empty or quiet?"),
    "lights": (r"\bstreetlights?\b|\blamps?\b|\blights\b", "Are street lights or lamps visible?"),
    "pedestrians": (r"\bpedestrians?\b|\bcrowd|\bcrosswalk\b|\bcrossing\b", "Is there a pedestrian crowd or crossing?"),
    "animation": (r"cartoon|animat|\b3d\b|\brender|illustration|\bcharacters?\b", "Does it look like an animation?"),
    "wall": (r"\bwalls?\b", "Is a wall prominent?"),
}


def caption_words(caption: str) -> list[str]:
    import re
    cap = caption.lower()
    return [k for k, (rx, _) in KEYWORDS.items() if re.search(rx, cap)]


def add_words(clips: list[dict[str, Any]], min_df: int = 3) -> None:
    """Set c['attrs']['words'] = caption keywords whose document frequency over this archive is in [min_df, N-min_df]."""
    from collections import Counter
    cnt = Counter(w for c in clips for w in caption_words(c["extra"]["caption"]))
    keep = {w for w, n in cnt.items() if min_df <= n <= len(clips) - min_df}
    for c in clips:
        c["attrs"]["words"] = ",".join(w for w in caption_words(c["extra"]["caption"]) if w in keep)
CAT_Q = {"motion": "Do things mostly move {v}?", "colour": "Is the dominant colour {v}?",  # colour: first version only
         
         "setting": "Is the setting: {v}?", "indoor_outdoor": "Is it {v}?"}
# "colour" is still measured (c["attrs"]["colour"], shown in the report) but is NOT an attribute the game asks about or the twin
# test uses: a judge cannot answer "is the dominant colour grey?" the way the index computed it (reviewer finding, eyeball-checked).
ATTRS = ["persons", "vehicles", "bike", "heavy_vehicle", "stationary_vehicle", "stationary_person", "bags", "furniture",
         "traffic_furniture", "animal", "motion", "brightness", "indoor_outdoor", "setting", "words"]
SOURCE = {"persons": "yolo", "vehicles": "yolo", "bike": "yolo", "heavy_vehicle": "yolo", "stationary_vehicle": "yolo",
          "stationary_person": "yolo", "bags": "yolo", "furniture": "yolo", "traffic_furniture": "yolo", "animal": "yolo",
          "motion": "yolo", "brightness": "frames", "indoor_outdoor": "cosmos", "setting": "cosmos", "words": "cosmos"}


def attr_vector(c: dict[str, Any], attrs: list[str] | None = None) -> tuple:
    return tuple(str(c["attrs"][a]) for a in (attrs or ATTRS))


def make_questions(clips: list[dict[str, Any]], attrs: list[str] | None = None) -> list[dict[str, Any]]:
    """Candidate yes/no questions: list of {id, attr, op ('is'|'ge'|'has'), val, text}. Only questions that split the archive."""
    attrs = attrs or ATTRS
    qs: list[dict[str, Any]] = []
    for a in attrs:
        vals = [str(c["attrs"][a]) for c in clips]
        if a == "words":
            vocab = sorted({w for v in vals for w in v.split(",") if w})
            for w in vocab:
                qs.append({"id": f"word:{w}", "attr": a, "op": "has", "val": w, "text": KEYWORDS[w][1]})
        elif a in ORD:
            order = ORD[a]
            for i in range(1, len(order)):
                if a == "persons":
                    txt = {"1": "Is anyone visible?", "2-3": "At any moment, are there 2 or more people?",
                           "4-7": "At any moment, are there 4 or more people?", "8+": "Is it a crowd (8+ people at once)?"}[order[i]]
                elif a == "vehicles":
                    txt = {"1": "Is any car, truck or bus visible?", "2-3": "At any moment, are there 2 or more vehicles?",
                           "4+": "At any moment, are there 4 or more vehicles?"}[order[i]]
                else:
                    txt = {"dim": "Is it brighter than night-dark?", "bright": "Is it brightly lit (daylight)?"}[order[i]]
                qs.append({"id": f"{a}>={order[i]}", "attr": a, "op": "ge", "val": order[i], "text": txt})
        elif a in BOOL_Q:
            qs.append({"id": f"{a}", "attr": a, "op": "is", "val": "True", "text": BOOL_Q[a]})
        else:
            for v in sorted(set(vals)):
                if v in ("unknown",):
                    continue
                txt = CAT_Q[a].format(v=v)
                if a == "motion" and v == "nothing moves":
                    txt = "Does everything stay still?"
                if a == "motion" and v == "camera moving":
                    txt = "Is the camera itself moving (pan, zoom, driving)?"
                if a == "motion" and v == "mixed directions":
                    txt = "Do things move in mixed directions?"
                qs.append({"id": f"{a}={v}", "attr": a, "op": "is", "val": v, "text": txt})
    keep = []
    for q in qs:
        k = sum(answer(q, c["attrs"]) for c in clips)
        if 0 < k < len(clips):
            keep.append(q)
    return keep


def answer(q: dict[str, Any], attrs: dict[str, Any]) -> bool:
    v = str(attrs[q["attr"]])
    if q["op"] == "has":
        return q["val"] in v.split(",")
    if q["op"] == "ge":
        order = ORD[q["attr"]]
        return order.index(v) >= order.index(q["val"])
    return v == q["val"]


def entropy_split(k: int, n: int) -> float:
    if k <= 0 or k >= n:
        return 0.0
    p = k / n
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


PROTO = {"eps": 0.1, "stop": 0.8, "max_guesses": 3, "max_show": 4, "max_questions": 30}


class Engine:
    """Greedy information-gain questioner over a candidate set. Ties broken by question order (deterministic)."""

    def __init__(self, clips: list[dict[str, Any]], questions: list[dict[str, Any]]):
        self.clips, self.qs = clips, questions
        self.ans = np.array([[answer(q, c["attrs"]) for q in questions] for c in clips], dtype=bool)  # clip x question

    def best(self, cand: list[int], asked: set[int]) -> int | None:
        best, bi = 1e-9, None
        n = len(cand)
        for qi in range(len(self.qs)):
            if qi in asked:
                continue
            k = int(self.ans[cand, qi].sum())
            g = entropy_split(k, n)
            if g > best + 1e-12:
                best, bi = g, qi
        return bi

    def play(self, secret: int, flip: float = 0.0, rng: np.random.Generator | None = None, recover: bool = False,
             max_q: int = 40) -> dict[str, Any]:
        """Answer truthfully (or flipped with prob `flip`) for clips[secret]. Returns questions asked and outcome."""
        cand = list(range(len(self.clips)))
        asked: set[int] = set()
        hist: list[tuple[int, bool]] = []
        n = 0
        while len(cand) > 1 and n < max_q:
            qi = self.best(cand, asked)
            if qi is None:
                break
            a = bool(self.ans[secret, qi])
            if flip and rng is not None and rng.random() < flip:
                a = not a
            asked.add(qi)
            hist.append((qi, a))
            n += 1
            new = [i for i in cand if bool(self.ans[i, qi]) == a]
            if not new:  # contradiction: only possible with noisy answers
                if recover:
                    cand = self.closest(hist)
                    break
                cand = []
                break
            cand = new
        if recover and len(cand) > 1 and flip:
            cand = self.closest(hist)
        found = len(cand) == 1 and cand[0] == secret
        return {"secret": secret, "n_questions": n, "final": cand, "found": found,
                "ambiguous": len(cand) > 1, "history": hist}

    # ---- the page protocol: hard elimination -> "Is it this one?" -> soft recovery (mirrored by engine.js nextStep) ----
    def weights(self, hist: list[tuple[int, bool | None]], rejected: set[int], eps: float) -> list[float]:
        """Soft weight per clip: 0 if rejected by a guess, else eps ** (#answers that contradict the clip). 'Not sure' is ignored."""
        w = []
        for i in range(len(self.clips)):
            if i in rejected:
                w.append(0.0)
                continue
            m = sum(1 for qi, a in hist if a is not None and bool(self.ans[i, qi]) != a)
            w.append(eps ** m)
        return w

    def next_step(self, hist: list[tuple[int, bool | None]], wrong: list[list[int]], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Next move given the answers so far: {'type': 'ask', 'qi'} | {'type': 'guess', 'ids'} | {'type': 'giveup'}.
        Phase A (no guess rejected yet, answers consistent): hard elimination, greedy information gain, ends by asking
        'Is it this one?' on the survivor(s). Phase B (a guess was wrong, or the answers contradicted every clip): soft posterior
        where each contradicting answer costs a factor eps; ask the most informative question until the top clip has posterior
        >= stop, then guess again. Same tie rule as `best` (first question wins within 1e-12)."""
        P = {**PROTO, **(params or {})}
        asked = {qi for qi, _ in hist}
        n_all = len(self.clips)
        rejected = {i for g in wrong for i in g}
        if len(wrong) >= P["max_guesses"]:
            return {"type": "giveup"}
        hard = list(range(n_all))
        for qi, a in hist:
            if a is not None:
                hard = [i for i in hard if bool(self.ans[i, qi]) == a]
        if not rejected and hard:
            if len(hard) == 1:
                return {"type": "guess", "ids": hard}
            qi = self.best(hard, asked)
            if qi is None:
                return {"type": "guess", "ids": hard[:P["max_show"]]}
            return {"type": "ask", "qi": qi}
        w = self.weights(hist, rejected, P["eps"])
        tot = sum(w)
        if tot <= 0:
            return {"type": "giveup"}
        post = [x / tot for x in w]
        top = max(post)
        group = [i for i in range(n_all) if post[i] >= top - 1e-9][:P["max_show"]]
        if top >= P["stop"] or len(asked) >= P["max_questions"]:
            return {"type": "guess", "ids": group}
        eps = P["eps"]
        he = -(eps * math.log2(eps) + (1 - eps) * math.log2(1 - eps))
        best, bi = 1e-6, None
        for qi in range(len(self.qs)):
            if qi in asked:
                continue
            py = sum(post[i] for i in range(n_all) if self.ans[i, qi])
            pr = py * (1 - eps) + (1 - py) * eps
            hq = 0.0 if pr <= 0 or pr >= 1 else -(pr * math.log2(pr) + (1 - pr) * math.log2(1 - pr))
            g = hq - he
            if g > best + 1e-12:
                best, bi = g, qi
        if bi is None:
            return {"type": "guess", "ids": group}
        return {"type": "ask", "qi": bi}

    def run_protocol(self, secret: int, flip: float = 0.0, rng: np.random.Generator | None = None, flip_at: set[int] | None = None,
                     params: dict[str, Any] | None = None, judge=None) -> dict[str, Any]:
        """Play the page protocol against clips[secret]. flip: random answer-flip probability; flip_at: deterministic flips at
        those question positions (0-based, for parity tests). Judge answers guesses truthfully (picks the secret if shown)."""
        P = {**PROTO, **(params or {})}
        hist: list[tuple[int, bool | None]] = []
        wrong: list[list[int]] = []
        rounds = 0
        first_group = None
        while True:
            st = self.next_step(hist, wrong, P)
            if st["type"] == "ask":
                a = bool(self.ans[secret, st["qi"]])
                if judge is not None:  # an outside judge (e.g. eyeballed truth); returns True/False/None (= "not sure")
                    a = judge(secret, st["qi"])
                pos = len(hist)
                if a is not None and (flip and rng is not None and rng.random() < flip) or (flip_at and pos in flip_at):
                    a = not a
                hist.append((st["qi"], a))
                continue
            if st["type"] == "giveup":
                return {"secret": secret, "found": False, "n_questions": len(hist), "guesses": rounds, "first_group": first_group}
            rounds += 1
            if first_group is None:
                first_group = len(st["ids"])
            if secret in st["ids"]:
                return {"secret": secret, "found": True, "n_questions": len(hist), "guesses": rounds, "first_group": first_group}
            wrong.append(list(st["ids"]))

    def closest(self, hist: list[tuple[int, bool]]) -> list[int]:
        mism = np.array([sum(bool(self.ans[i, qi]) != a for qi, a in hist) for i in range(len(self.clips))])
        return [int(i) for i in np.flatnonzero(mism == mism.min())]


def twins(clips: list[dict[str, Any]], attrs: list[str] | None = None) -> list[list[str]]:
    """Groups of clips with an identical attribute vector (>= 2 members)."""
    g: dict[tuple, list[str]] = defaultdict(list)
    for c in clips:
        g[attr_vector(c, attrs)].append(c["id"])
    return sorted([v for v in g.values() if len(v) > 1], key=lambda x: (-len(x), x[0]))


def evaluate(clips: list[dict[str, Any]], attrs: list[str] | None = None, flip: float = 0.0, reps: int = 1,
             recover: bool = False, seed: int = 0) -> dict[str, Any]:
    qs = make_questions(clips, attrs)
    eng = Engine(clips, qs)
    rng = np.random.default_rng(seed)
    plays = [eng.play(i, flip, rng, recover) for _ in range(reps) for i in range(len(clips))]
    found = [p for p in plays if p["found"]]
    nq = [p["n_questions"] for p in plays]
    tw = twins(clips, attrs)
    in_twin = sum(len(t) for t in tw)
    return {"n_clips": len(clips), "n_questions_available": len(qs), "success_rate": len(found) / len(plays),
            "mean_questions_all": float(np.mean(nq)), "mean_questions_found": float(np.mean([p["n_questions"] for p in found])) if found else None,
            "max_questions": int(max(nq)), "unique_vectors": len({attr_vector(c, attrs) for c in clips}),
            "twin_groups": len(tw), "clips_in_twin_groups": in_twin, "plays": len(plays)}


# ----------------------------------------------------------------------------------------------- archive


def build_archive(n: int = 64, seed: int = 12, use_vlm: bool = True, log=print, workers: int = 6) -> list[dict[str, Any]]:
    """Sample clips and extract every attribute. Saves thumbnails to out/_work/thumbs and returns clip dicts.
    Stage 1 (tracks + frames) is sequential; stage 2 (Cosmos, slow and shared) runs in a small thread pool."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    clips = sample_clips(n, seed)
    trs: dict[str, Any] = {}
    (WORK / "thumbs").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    mids = []
    for k, c in enumerate(clips):
        v = c["video"]
        if v not in trs:
            trs[v] = pk.tracks(v)
        ff, mid = frame_features(v, c["t0"], c["t1"])
        tf = track_features(trs[v], c["t0"], c["t1"], ff["camera_moving"])
        c["attrs"] = {**{a: tf[a] for a in ("persons", "vehicles", "bike", "heavy_vehicle", "stationary_vehicle",
                                              "stationary_person", "bags", "furniture", "traffic_furniture", "animal", "motion")},
                      "brightness": ff["brightness"], "colour": ff["colour"]}
        c["extra"] = {"luma": ff["luma"], "cam_flow": ff["cam_flow"], "camera_moving": ff["camera_moving"],
                      "n_tracks": tf["n_tracks"], "blind": tf["blind"]}
        pk.save_jpg(mid, WORK / "thumbs" / f"{c['id']}.jpg", 85)
        mids.append(mid)
        if log and (k % 16 == 0 or k == len(clips) - 1):
            log(f"  frames+tracks {k + 1}/{len(clips)} ({time.time() - t_start:.0f}s)")
    with ThreadPoolExecutor(workers) as ex:
        cfs = list(ex.map(lambda m: cosmos_features(m, use_vlm), mids))
    for c, cf in zip(clips, cfs):
        c["attrs"].update({"indoor_outdoor": cf["indoor_outdoor"], "setting": cf["setting"]})
        c["extra"].update({"caption": cf["caption"], "vlm_ok": cf["vlm_ok"]})
    if log:
        log(f"  cosmos done ({time.time() - t_start:.0f}s), vlm ok {sum(c['extra']['vlm_ok'] for c in clips)}/{len(clips)}")
    return clips
