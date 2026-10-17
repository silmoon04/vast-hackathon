"""The overnight briefing: what a person should know about one camera in 10 seconds.

Built by plain rules from whatever outputs already exist under out/<cam>/ (ledger, diff, near,
moments, flow, synopsis, index; meta for duration and SIMULATED notes). Missing or half-written
inputs are skipped and listed. Every number in the text is copied (or summed/rounded) from an
input file or from stm.data.objects(cam); nothing is guessed. Blind time is the ledger's own
blind_total_s (per day: blind_by_day) when it has one, so the brief prints the same number as every view.

    python -m stm.brief [cams...]      -> out/<cam>/brief.json + out/<cam>/brief_summary.txt
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from stm import data

INPUTS = ("ledger", "diff", "near", "moments", "flow", "synopsis", "index")
MAX_BULLETS = 6

REASON_WORDS = {
    "dark": "too dark", "blur": "blurry picture", "frozen": "picture froze", "blocked": "view blocked",
    "no_signal": "no picture", "camera_moving": "camera moving", "view_changed": "camera pointed elsewhere",
    "turned_away": "camera turned away", "moved": "camera moving",
}
NO_COLOUR = {None, "", "null", "none", "None", "unknown"}  # index rows with no readable colour (night infrared, ...)
PLURAL = {"person": "people", "bus": "buses", "stretch": "stretches"}


# ---------------------------------------------------------------- small text helpers
def noun(name: str, n: int | float) -> str:
    return name if n == 1 else PLURAL.get(name, name + "s")


def join_and(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def mmss(t: float | None) -> str:
    if t is None:
        return "?"
    s = int(round(float(t)))
    return f"{s // 60:02d}:{s % 60:02d}"


def dur(s: float) -> str:
    """Seconds as plain words: '40 s', '5 min 16 s'."""
    s = int(round(float(s)))
    if s < 90:
        return f"{s} s"
    m, r = divmod(s, 60)
    return f"{m} min" + (f" {r} s" if r else "")


def breakdown(names: list[str]) -> str:
    """['car', 'person', 'car'] -> '2 cars and 1 person' (most common first)."""
    c = Counter(names)
    return join_and([f"{n} {noun(k, n)}" for k, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))])


def _r(t: Any) -> float | None:
    try:
        return None if t is None else round(float(t), 2)
    except (TypeError, ValueError):
        return None


def _bullet(text: str, kind: str, t: Any = None, crop: str | None = None, prio: int = 50) -> dict[str, Any]:
    return {"text": text, "t": _r(t), "kind": kind, "crop": crop or None, "_prio": prio}


# ---------------------------------------------------------------- inputs
def load_inputs(cam: str, root: Path | None = None) -> tuple[dict[str, Any], list[str]]:
    """Read out/<cam>/<name>.json for each input. Returns (found, missing). Unreadable = missing."""
    d = Path(root) if root else data.OUT / cam
    found, missing = {}, []
    for name in INPUTS + ("meta",):
        p = d / f"{name}.json"
        if not p.exists():
            missing.append(f"{name}.json")
            continue
        try:
            found[name] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            missing.append(f"{name}.json (unreadable)")
    return found, missing


def object_counts(objs: Any) -> dict[str, Any] | None:
    """Counts from a stm.data.objects(cam) DataFrame: moving tracks by name, parked vehicles, id->name.

    "Moving" = not parked and its centre travelled at least its own box width; the other unparked
    tracks (short or still fragments, often of a parked car) are counted as "still", not as going by.
    """
    if objs is None or len(objs) == 0:
        return {"total": 0, "moving": {}, "still": 0, "parked": 0, "parked_names": {}, "names": {}, "last_t": None}
    is_parked = objs.parked.astype(bool)
    unparked = objs[~is_parked]
    if {"travel_px", "x1", "x2"} <= set(objs.columns):
        went = unparked.travel_px >= (unparked.x2 - unparked.x1)
    else:
        went = unparked.parked == unparked.parked  # no travel info: every unparked track counts as moving
    moving, parked = unparked[went], objs[is_parked]
    return {"total": int(len(objs)),
            "moving": {k: int(v) for k, v in moving.name.value_counts().items()},
            "still": int(len(unparked) - len(moving)),
            "parked": int(len(parked)),
            "parked_names": {k: int(v) for k, v in parked.name.value_counts().items()},
            "names": {int(i): str(n) for i, n in zip(objs.id, objs.name)},
            "last_t": float(objs.last_t.max())}


def merged_blind(ledger: dict[str, Any]) -> list[tuple[float, float, str]]:
    """The ledger's blind spans, sorted (one per reason; touching spans are NOT joined here)."""
    spans = sorted((float(b["start"]), float(b["end"]), str(b.get("reason") or "unknown"))
                   for b in ledger.get("blind", []) if float(b["end"]) > float(b["start"]))
    return spans


def stretches(spans: list[tuple[float, float, str]]) -> list[tuple[float, float]]:
    """Join overlapping or touching blind spans into continuous stretches (start, end)."""
    out: list[list[float]] = []
    for s, e, _ in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def ledger_blind_total(ledger: dict[str, Any]) -> float:
    """The ledger's own blind total (`blind_total_s`, the one number every view prints) when it has one,
    else the merged sum of its spans."""
    v = ledger.get("blind_total_s")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return blind_seconds(merged_blind(ledger))


def ledger_blind_by_reason(ledger: dict[str, Any]) -> Counter:
    """Blind seconds per reason: `blind_by_reason`, else summed over `blind_by_day`, else over the spans."""
    by: Counter = Counter()
    if isinstance(ledger.get("blind_by_reason"), dict):
        for r, v in ledger["blind_by_reason"].items():
            by[str(r)] += float(v or 0)
    elif isinstance(ledger.get("blind_by_day"), list):
        for d in ledger["blind_by_day"]:
            for r, v in (d.get("by_reason") or {}).items():
                by[str(r)] += float(v or 0)
    else:
        for s, e, r in merged_blind(ledger):
            by[r] += e - s
    return Counter({r: v for r, v in by.items() if v > 0})


def colour_counts(index: dict[str, Any]) -> tuple[list[tuple[str, int]], int]:
    """Vehicle colours, most common first, and how many vehicles have one. Null colours are skipped.

    Uses the index's own `colours` / `n_coloured` (vehicles only). An older index without `n_coloured`
    also coloured people, so its vehicle rows are counted directly."""
    objs = index.get("objects") or []
    if index.get("n_coloured") is None and any("colour" in o and "name" in o for o in objs):
        c = Counter(str(o["colour"]) for o in objs
                    if o.get("colour") not in NO_COLOUR and o.get("name") in data.VEHICLES)
        cols = list(c.items())
    else:
        cols = [(str(k), int(v)) for k, v in (index.get("colours") or {}).items() if k not in NO_COLOUR and int(v or 0)]
    cols.sort(key=lambda kv: (-kv[1], kv[0]))
    n = int(index["n_coloured"]) if index.get("n_coloured") is not None else sum(v for _, v in cols)
    return cols, n


def blind_seconds(spans: list[tuple[float, float, str]]) -> float:
    """Total blind time with overlaps merged."""
    tot, cur_s, cur_e = 0.0, None, None
    for s, e, _ in spans:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


# ---------------------------------------------------------------- the rules
def build(cam: str, inputs: dict[str, Any], counts: dict[str, Any] | None = None,
          simulated: list[str] | None = None, missing: list[str] | None = None,
          root: Path | None = None) -> dict[str, Any]:
    """Pure rules: inputs (name -> parsed JSON), counts (object_counts) -> brief dict."""
    ledger, diff, near = inputs.get("ledger"), inputs.get("diff"), inputs.get("near")
    moments, flow, syn, index, meta = (inputs.get(k) for k in ("moments", "flow", "synopsis", "index", "meta"))
    used: list[str] = []
    bullets: list[dict[str, Any]] = []
    honesty: list[str] = []
    first_notes: list[str] = []
    names = (counts or {}).get("names", {})

    def crop_ok(c: Any) -> str | None:
        if not c:
            return None
        if root is not None and not (Path(root) / str(c)).exists():
            return None
        return str(c)

    duration = None
    for src, key in ((meta, "duration"), (syn, "src_duration"), (ledger, "seconds")):
        if src and src.get(key) is not None:
            duration = float(src[key])
            break
    day_label = {d["id"]: d.get("label", d["id"]) for d in (meta or {}).get("days", []) if "id" in d}

    # ---- what changed among parked vehicles
    change = None
    if diff and isinstance(diff.get("changes"), list):
        used.append("diff.json")
        ch = diff["changes"]
        cnt = dict(diff.get("counts") or Counter(c.get("kind") for c in ch))
        la = day_label.get(diff.get("day_a", "A"), f"Day {diff.get('day_a', 'A')}")
        lb = day_label.get(diff.get("day_b", "B"), f"Day {diff.get('day_b', 'B')}")
        ids = [s["object_id"] for c in ch for s in (c.get("a"), c.get("b")) if s and "object_id" in s]
        kinds = {names.get(int(i)) for i in ids} - {None}
        veh = "car" if kinds == {"car"} else "vehicle"
        L, A, M, S, same = (int(cnt.get(k, 0)) for k in ("left", "arrived", "moved", "swapped", "same"))
        parts = []
        if L:
            parts.append(f"{L} {noun(veh, L)} left")
        if A:
            parts.append(f"{A} arrived" if parts else f"{A} {noun(veh, A)} arrived")
        elif L:
            parts.append("none arrived")
        if A and not L:
            parts.append("none left")
        if M:
            parts.append(f"{M} moved" if parts else f"{M} {noun(veh, M)} moved")
        if S:
            parts.append(f"{S} {noun('spot', S)} got a different {veh}")
        change = (join_and(parts) if parts else f"no parked {veh} left or arrived") + f" between {la} and {lb}"

        def first(kind: str) -> dict[str, Any] | None:
            return next((c for c in ch if c.get("kind") == kind), None)

        def side(c: dict[str, Any] | None, pref: str) -> dict[str, Any]:
            if not c:
                return {}
            return c.get(pref) or c.get("a" if pref == "b" else "b") or {}

        if L:
            s = side(first("left"), "a")
            bullets.append(_bullet(f"{L} {noun(veh, L)} parked on {la} {'was' if L == 1 else 'were'} gone on {lb}.",
                                   "left", s.get("t"), crop_ok(s.get("crop")), 10))
        if A:
            s = side(first("arrived"), "b")
            bullets.append(_bullet(f"{A} {noun(veh, A)} {'was' if A == 1 else 'were'} new on {lb}.",
                                   "arrived", s.get("t"), crop_ok(s.get("crop")), 11))
        if S:
            s = side(first("swapped"), "b")
            bullets.append(_bullet(f"{S} parking {noun('spot', S)} had a different {veh} on {lb} than on {la}.",
                                   "swapped", s.get("t"), crop_ok(s.get("crop")), 12))
        if M:
            s = side(first("moved"), "b")
            bullets.append(_bullet(f"{M} {noun(veh, M)} moved to a different spot.", "moved",
                                   s.get("t"), crop_ok(s.get("crop")), 13))
        if same:
            s = side(first("same"), "a")
            bullets.append(_bullet(f"{same} {noun(veh, same)} stayed in the same place.", "same",
                                   s.get("t"), crop_ok(s.get("crop")), 40))
        nc = diff.get("not_compared") or []
        if nc:
            why = Counter(str(x.get("reason") or "no reason given") for x in nc)
            top, n_top = why.most_common(1)[0]
            honesty.append(f"{len(nc)} parked {noun(veh, len(nc))} could not be compared between the two days"
                           f" ({n_top} because: {top}).")
    elif ledger and ledger.get("spots"):
        spot_changes = [(c, s) for s in ledger["spots"] for c in s.get("changes", [])]
        cnt = Counter(c.get("kind") for c, _ in spot_changes)
        A, L, S = cnt.get("arrived", 0), cnt.get("left", 0), cnt.get("swapped", 0)
        n_spots = len(ledger["spots"])
        if spot_changes:
            parts = []
            if A:
                parts.append(f"{A} {noun('vehicle', A)} arrived")
            if L:
                parts.append(f"{L} left" if parts else f"{L} {noun('vehicle', L)} left")
            if S:
                parts.append(f"{S} {noun('spot', S)} got a different vehicle")
            change = join_and(parts) + " in the parking spots"
            for kind, prio, img in (("arrived", 10, "after"), ("left", 11, "before"), ("swapped", 12, "after")):
                ex = next((c for c, _ in spot_changes if c.get("kind") == kind), None)
                if ex:
                    n = cnt[kind]
                    txt = {"arrived": f"{n} {noun('vehicle', n)} pulled into a parking spot.",
                           "left": f"{n} {noun('vehicle', n)} left a parking spot.",
                           "swapped": f"{n} {noun('time', n)} a spot was taken by a different vehicle."}[kind]
                    bullets.append(_bullet(txt + f" First at {mmss(ex.get('t'))}.", kind, ex.get("t"),
                                           crop_ok(ex.get(img)), prio))
        else:
            change = f"the {n_spots} parked {noun('vehicle', n_spots)} stayed put"

    # ---- who and what was seen (tracks)
    from_diff_or_ledger = change is not None  # else the headline already says these counts
    if counts is not None:
        used.append("tracks (stm.data.objects)")
        mv = counts.get("moving", {})
        top = sorted(mv.items(), key=lambda kv: (-kv[1], kv[0]))
        P = int(counts.get("parked", 0))
        seen = join_and([f"{n} {noun(k, n)}" for k, n in top[:3]])
        if counts.get("total", 0) == 0:
            if change is None:
                change = "nothing was seen moving"
        else:
            if change is None:
                change = (f"{seen} went by" if seen else "nothing went by") + \
                         (f" and {P} {noun('vehicle', P)} stayed parked" if P else "")
            txt = (f"In total {join_and([f'{n} {noun(k, n)}' for k, n in top])} moved through the view"
                   if top else "Nothing moved through the view")
            txt += f"; {P} {noun('track', P)} {'was a parked vehicle' if P == 1 else 'were parked vehicles'}." if P else "."
            bullets.append(_bullet(txt, "count", None, None, 22 if from_diff_or_ledger else 37))
            honesty.append("Counts come from an automatic detector; one person or car can be counted twice "
                           "if it was lost and found again.")
            still = int(counts.get("still", 0))
            if still:
                honesty.append(f"{still} more {noun('track', still)} hardly moved (less than {'its' if still == 1 else 'their'} "
                               "own width) and {} not counted as going by.".format("is" if still == 1 else "are"))

    # ---- who came close to parked vehicles
    if near and near.get("targets"):
        used.append("near.json")
        tg = max(near["targets"], key=lambda x: (len(x.get("visitors", [])), -float(x.get("start", 0))))
        vis = tg.get("visitors", [])
        if vis:
            with_vis = sum(1 for x in near["targets"] if x.get("visitors"))
            nm = tg.get("name", "vehicle")
            bullets.append(_bullet(
                f"Most visitors near one parked {nm}: {len(vis)} came close ({breakdown([v.get('name', 'thing') for v in vis])})"
                f" between {mmss(tg.get('start'))} and {mmss(tg.get('end'))}.",
                "near", min(vis, key=lambda v: float(v.get("gap_rel", 9))).get("closest_t"), crop_ok(tg.get("crop")), 20))
            honesty.append(f"'Came close' only means near: {with_vis} of the {len(near['targets'])} parked vehicles "
                           "checked had someone or something close by, which says nothing about who did anything.")

    # ---- highlight moments
    if moments and moments.get("moments"):
        used.append("moments.json")
        # "Person stays N s" (long_stay) is not news on its own: it is dropped here, at the source
        skip = {"blind", "long_stay"} | ({"arrival", "departure"} if diff else set())
        prio = {"crowd": 21, "close_pass": 23}
        best: dict[str, dict[str, Any]] = {}  # best-scoring moment of each kind, so kinds do not repeat
        for m in moments["moments"]:
            k = m.get("kind")
            if k in skip or str(m.get("title") or "").startswith("Person stays"):
                continue
            if k not in best or float(m.get("score") or 0) > float(best[k].get("score") or 0):
                best[k] = m
        ms = sorted(best.values(), key=lambda m: (prio.get(m.get("kind"), 35), -float(m.get("score") or 0)))
        for i, m in enumerate(ms[:3]):
            title = str(m.get("title") or m.get("detail") or m.get("kind")).strip().rstrip(".")
            bullets.append(_bullet(f"{title} at {mmss(m.get('t'))}.", str(m.get("kind")), m.get("t"),
                                   crop_ok(m.get("thumb")), prio.get(m.get("kind"), 35) + i))

    # ---- main lanes
    if flow and (flow.get("lanes") or flow.get("busiest")):
        used.append("flow.json")
        lanes = sorted(flow.get("lanes") or [], key=lambda x: -int(x.get("count", 0)))
        if lanes and int(lanes[0].get("count", 0)) > 0:
            ln = lanes[0]
            g = "person" if ln.get("group") == "person" else "vehicle"
            bullets.append(_bullet(f"Busiest route: {ln['count']} {noun(g, int(ln['count']))} went {ln.get('name', 'the same way')}.",
                                   "lane", None, crop_ok(flow.get("image")), 25))
        b = flow.get("busiest")
        if b and b.get("count"):
            who = (join_and([f"{int(b[g])} {noun(g if g == 'person' else 'vehicle', int(b[g]))}"
                             for g in ("person", "vehicle") if int(b[g])])
                   if "person" in b and "vehicle" in b else f"{b['count']} people and vehicles")
            bullets.append(_bullet(f"Busiest moment: {who} on screen at {mmss(b.get('t'))}.",
                                   "busiest", b.get("t"), crop_ok(b.get("frame")), 36))

    # ---- replay
    if syn and syn.get("ghosts") is not None:
        used.append("synopsis.json")
        G = len(syn["ghosts"])
        gn = [str(g.get("name")) for g in syn["ghosts"] if g.get("name")]
        what = breakdown(gn) if len(gn) == G and len(set(gn)) <= 3 else f"{G} people and vehicles"
        if G:
            src = float(syn.get("src_duration") or 0)
            frm = f"the {dur(src)} recording"
            win = syn.get("src_window")
            if isinstance(win, (list, tuple)) and len(win) == 2 and None not in win:
                span = max(0.0, float(win[1]) - float(win[0]))  # the replay only shows this stretch
                if src and span < src - 1.0:
                    lab = str(syn.get("view_label") or "").strip()
                    frm = f"{dur(span)} of the {dur(src)} recording" + (f" ({lab})" if lab else "")
                else:
                    frm = f"the {dur(span)} recording"
            bullets.append(_bullet(f"A {dur(syn.get('out_duration', 0))} replay shows {what} from {frm} at once.",
                                   "replay", None, None, 38))

    # ---- colours
    if index and (index.get("colours") or index.get("n_vehicles")):
        used.append("index.json")
        cols, n_col = colour_counts(index)
        if cols and n_col:
            bullets.append(_bullet(f"Most common colour: {join_and([f'{c} ({n})' for c, n in cols[:3]])}"
                                   f" of {n_col} {noun('vehicle', n_col)}.", "colour", None, None, 39))
        n_veh = index.get("n_vehicles")
        if n_veh is not None and int(n_veh) > n_col:
            nc = int(n_veh) - n_col
            honesty.append(f"{nc} of {int(n_veh)} vehicles {'has' if nc == 1 else 'have'} no colour (seen at night "
                           f"or too dark or small to read), so the colour count leaves {'it' if nc == 1 else 'them'} out.")

    # ---- can the camera see? (honesty + headline tail)
    tail = None
    if ledger is not None:
        used.append("ledger.json")
        spans = merged_blind(ledger)
        tot = ledger_blind_total(ledger)  # the ledger's own total when it has one, so every view agrees
        if tot >= 0.5:
            by = ledger_blind_by_reason(ledger)
            why = join_and([f"{dur(v)} {REASON_WORDS.get(r, r.replace('_', ' '))}" for r, v in by.most_common()])
            runs = stretches(spans)
            most = duration is not None and duration > 0 and tot >= 0.9 * duration
            tail = f"the camera could not see for {dur(tot)}" + (f" of the {dur(duration)}" if most else "")
            days = [d for d in ledger.get("blind_by_day") or [] if d.get("blind_s") is not None]
            per_day = (" (" + join_and([f"{dur(float(d['blind_s']))} on {d.get('label') or d.get('day')}"
                                        for d in days]) + ")") if len(days) > 1 else ""
            note = f"The camera could not see for {dur(tot)} in total{per_day}"
            if runs:
                longest = max(runs, key=lambda x: x[1] - x[0])
                note += (f", in {len(runs)} {noun('stretch', len(runs))} ({why}); the longest was "
                         f"{mmss(longest[0])} to {mmss(longest[1])}")
            elif why:
                note += f" ({why})"
            first_notes.append(note + ". Anything that happened then is unknown.")
            if most and counts and counts.get("total", 0) > 0:
                first_notes.append(f"The camera check says the view was hidden for most of the clip, but the detector "
                               f"still found {counts['total']} {noun('thing', counts['total'])}; one of the two is wrong.")
        else:
            tail = "the camera could see the whole time"
            first_notes.append("The camera check found no time when the camera could not see.")
    else:
        first_notes.append("Not checked yet whether the camera could see the whole time (no camera health report).")

    for s in simulated or []:
        first_notes.append(f"SIMULATED: {s.rstrip('.')}.")

    if change is None:
        change = "no results yet for this camera"
    head = change + (f"; {tail}" if tail else "") + "."
    head = head[0].upper() + head[1:]

    bullets.sort(key=lambda b: b["_prio"])
    bullets = bullets[:MAX_BULLETS]
    for b in bullets:
        b.pop("_prio")
    if ledger is not None:  # a bullet whose time falls while the camera could not see is flagged, not dropped
        for b in bullets:
            hit = next((sp for sp in merged_blind(ledger) if b["t"] is not None and sp[0] <= b["t"] < sp[1]), None)
            if hit:
                honesty.append(f"\"{b['text'].split(':')[0].rstrip('.')}\" at {mmss(b['t'])} falls in a stretch when the "
                               f"camera could not see well ({REASON_WORDS.get(hit[2], hit[2])}), so treat it with care.")
    return {"camera": cam, "headline": head, "bullets": bullets, "honesty": first_notes + honesty,
            "generated_from": used, "missing": list(missing or [])}


# ---------------------------------------------------------------- CLI
def run(cam: str) -> dict[str, Any]:
    t0 = time.time()
    root = data.out_dir(cam)
    inputs, missing = load_inputs(cam, root)
    try:
        counts = object_counts(data.objects(cam))
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        counts = None
        missing.append(f"tracks ({type(e).__name__})")
    sim = (inputs.get("meta") or {}).get("simulated")
    if sim is None:
        sim = data.CAMS.get(cam, {}).get("simulated", [])
    b = build(cam, inputs, counts, sim, missing, root)
    data.write_json(root / "brief.json", b)
    blind = ledger_blind_total(inputs["ledger"]) if "ledger" in inputs else None
    lines = [f"brief for {cam}",
             f"headline: {b['headline']}",
             f"bullets: {len(b['bullets'])} {dict(Counter(x['kind'] for x in b['bullets']))}",
             f"honesty notes: {len(b['honesty'])}",
             f"generated from: {', '.join(b['generated_from']) or 'nothing'}",
             f"missing: {', '.join(b['missing']) or 'none'}",
             f"blind seconds (ledger total): {'not available' if blind is None else round(blind, 2)}"]
    if "diff" in inputs:
        lines.append(f"diff counts: {inputs['diff'].get('counts')}")
    if counts is not None:
        lines.append(f"tracked objects: {counts['total']} (moving {counts['moving']}, parked {counts['parked']})")
    lines.append(f"runtime: {time.time() - t0:.1f} s")
    (root / "brief_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return b


def main(argv: list[str]) -> None:
    for cam in argv or list(data.CAMS):
        if cam not in data.CAMS:
            print(f"skip {cam}: unknown camera")
            continue
        b = run(cam)
        print(f"{cam}: {b['headline']}  [{len(b['bullets'])} bullets; from {len(b['generated_from'])}, missing {len(b['missing'])}]")


if __name__ == "__main__":
    main(sys.argv[1:])
