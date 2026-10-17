"""Demo freeze gate + presenter cue card.

    <py> scripts/freeze_demo.py [--server http://127.0.0.1:8780] [--no-export] [--cue CUECARD.txt]

Checks (FAIL blocks the freeze, WARN is for the lead to judge):
 1. every out/<cam>/synopsis.mp4 opens in ffprobe (moov present) with duration > 0
 2. every image path referenced by every JSON exists on disk
 3. numbers the UI shows from two files agree:
    - blind total: ledger.json (blind_total_s, else the sum of blind spans) vs every "blind_total_s" key in
      any other JSON, the brief's "could not see for ..." text, and qa.json's ledger blind spans
    - diff.json counts vs the brief headline/bullets and the Ask "what changed between <day> and <day>" answer
    - honesty: cams with real days never say "Day A"/"Day B" or "SIMULATED"; no city name for the event cam;
      people never get a colour (index.json) or "(in <colour>)" (qa.json)
 4. with --server (localhost only): GET /api/ask for every qa.json question; p50 must be < 100 ms; answers that
    differ from the precomputed qa.json are listed (stale qa.json)
 5. runs python -m stm.export (skip with --no-export) and prints the dist size
 6. writes CUECARD.txt: every number the presenter says, computed from out/, plus the run sheet
Exit code: 0 when no FAIL.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stm import export  # noqa: E402  (also gives data.OUT and the cam order)

OUT = export.OUT
FFPROBE_FALLBACK = Path("ffprobe")
NUM_WORDS = {"no": 0, "none": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
KINDS = ("same", "arrived", "left", "moved", "swapped")
COLOURS = ("white", "black", "grey", "gray", "silver", "red", "blue", "green", "yellow", "orange", "brown")
CITIES = ("London",)
TOL_S = 0.51  # rounding to whole seconds is fine; a 1 s disagreement is not


class Gate:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, cam: str, msg: str) -> None:
        self.rows.append((level, cam, msg))

    def ok(self, cam: str, msg: str) -> None:
        self.add("OK", cam, msg)

    def warn(self, cam: str, msg: str) -> None:
        self.add("WARN", cam, msg)

    def fail(self, cam: str, msg: str) -> None:
        self.add("FAIL", cam, msg)

    def count(self, level: str) -> int:
        return sum(1 for r in self.rows if r[0] == level)

    def text(self) -> str:
        return "\n".join(f"{lv:4} {cam:28} {msg}" for lv, cam, msg in self.rows)


# ---------- small helpers ----------

def load(cam: str, name: str) -> Any:
    p = OUT / cam / f"{name}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def mmss(t: float | None) -> str:
    if t is None:
        return "--:--"
    t = int(round(max(0.0, float(t))))
    return f"{t // 60:02d}:{t % 60:02d}"


def dur_text(s: float) -> str:
    s = int(round(s))
    return f"{s // 60} min {s % 60} s ({s} s)" if s >= 60 else f"{s} s"


def strings(obj: Any) -> list[str]:
    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str):
            out.append(o)

    walk(obj)
    return out


def find_key(obj: Any, key: str, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k == key:
                found.append((p, v))
            found += find_key(v, key, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += find_key(v, key, f"{path}[{i}]")
    return found


def merge_spans(spans: list[dict], gap: float = 1.0) -> list[dict]:
    out: list[dict] = []
    for s in sorted(spans, key=lambda x: (x["start"], x["end"])):
        if out and s["start"] - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
            out[-1]["reasons"].add(s.get("reason") or "?")
        else:
            out.append({"start": s["start"], "end": s["end"], "reasons": {s.get("reason") or "?"}})
    return out


def span_union(spans: list[dict]) -> float:
    tot, cur_s, cur_e = 0.0, None, None
    for s in sorted(spans, key=lambda x: x["start"]):
        if cur_e is None or s["start"] > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s["start"], s["end"]
        else:
            cur_e = max(cur_e, s["end"])
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def ledger_blind_total(L: Any) -> float | None:
    if not isinstance(L, dict):
        return None
    if isinstance(L.get("blind_total_s"), (int, float)):
        return float(L["blind_total_s"])
    return round(span_union(L.get("blind") or []), 2)


DUR_RE = re.compile(r"(?:could not see|couldn't see|was blind|blind) (?:for )?(?:a total of )?"
                    r"(?:(\d+) ?min(?:utes?)?)?(?: ?(\d+(?:\.\d+)?) ?s(?:ec(?:onds?)?)?)?\b", re.I)


def text_blind_totals(texts: list[str]) -> list[tuple[str, float]]:
    """'could not see for 1 min 38 s' -> 98.0. Only sentences that talk about the whole recording ('in total')
    or the headline form are used, so per-car blind seconds are not mistaken for the total."""
    out = []
    for s in texts:
        for m in DUR_RE.finditer(s):
            if m.group(1) is None and m.group(2) is None:
                continue
            tail = s[m.end(): m.end() + 40].lower()
            if "in total" not in tail and "could not see" not in m.group(0).lower():
                continue
            out.append((s[:90], int(m.group(1) or 0) * 60 + float(m.group(2) or 0)))
    return out


def text_counts(s: str) -> dict[str, int]:
    """'2 vehicles left and none arrived' -> {'left': 2, 'arrived': 0}."""
    out: dict[str, int] = {}
    num = r"(\d+|" + "|".join(NUM_WORDS) + r")"
    for m in re.finditer(num + r"\s+(?:[A-Za-z]+\s+){0,3}?(" + "|".join(KINDS) + r")\b", s, re.I):
        n = m.group(1).lower()
        out.setdefault(m.group(2).lower(), int(n) if n.isdigit() else NUM_WORDS[n])
    return out


def day_info(meta: Any) -> tuple[list[str], list[str]]:
    days = (meta or {}).get("days") or []
    return [str(d.get("id")) for d in days], [str(d.get("label") or d.get("id")) for d in days]


def qa_items(Q: Any) -> list[dict]:
    if isinstance(Q, dict):
        return [x for x in Q.get("items") or Q.get("questions") or [] if isinstance(x, dict)]
    return [x for x in Q or [] if isinstance(x, dict)]


def ans(item: dict) -> dict:
    a = item.get("answer")
    return a if isinstance(a, dict) else item


def between_days_item(items: list[dict], labels: list[str]) -> dict | None:
    for it in items:
        q = (it.get("question") or "").lower()
        if ans(it).get("type", it.get("type")) == "what_changed" and it.get("t") is None and "between" in q:
            return it
    for it in items:
        q = (it.get("question") or "")
        if len(labels) >= 2 and labels[0] in q and labels[1] in q:
            return it
    return None


class _Soft:
    """A Gate view that turns FAIL into WARN."""
    def __init__(self, g: Gate) -> None:
        self.g = g

    def ok(self, cam: str, msg: str) -> None:
        self.g.ok(cam, msg)

    def warn(self, cam: str, msg: str) -> None:
        self.g.warn(cam, msg)

    def fail(self, cam: str, msg: str) -> None:
        self.g.warn(cam, "(not in demo) " + msg)


# ---------- checks ----------

def check_videos(g: Gate, cams: list[str], demo: set[str] = frozenset()) -> None:
    probe = shutil.which("ffprobe") or (str(FFPROBE_FALLBACK) if FFPROBE_FALLBACK.exists() else None)
    for cam in cams:
        v = OUT / cam / "synopsis.mp4"
        if not v.is_file():
            (g.fail if cam in demo else g.warn)(cam, "no synopsis.mp4")
            continue
        if not probe:
            g.warn(cam, "ffprobe not found; synopsis.mp4 not probed")
            continue
        r = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)],
                           capture_output=True, text=True, timeout=60)
        try:
            d = float(r.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            d = 0.0
        if r.returncode != 0 or d <= 0:
            g.fail(cam, f"synopsis.mp4 does not probe (moov missing?): {r.stderr.strip()[:120]}")
        else:
            g.ok(cam, f"synopsis.mp4 {d:.1f} s, {v.stat().st_size / 2**20:.2f} MB")


def check_images(g: Gate, cams: list[str], demo: set[str]) -> None:
    for cam in cams + ["captions"]:
        folder = f"out/{cam}"
        names = [p.stem for p in sorted((OUT / cam).glob("*.json"))] if (OUT / cam).is_dir() else []
        bad: dict[str, list[str]] = {}
        n = 0
        for name in names:
            obj = load(cam, name)
            for s in export.image_strings(obj):
                u = export._url_for(folder, s)
                if not u:
                    continue
                n += 1
                if not (export.ROOT / u).is_file():
                    bad.setdefault(name, []).append(s)
        if not bad:
            g.ok(cam, f"all {n} image references exist")
            continue
        for name, miss in bad.items():
            msg = f"{name}.json references {len(miss)} missing images, e.g. {', '.join(miss[:3])}"
            (g.fail if cam in demo and name != "synopsis" else g.warn)(cam, msg)


def check_numbers(g: Gate, cam: str, strict: bool = True) -> None:
    """strict: problems are FAIL (demo cams); otherwise WARN (cams the demo does not show)."""
    if not strict:
        g = _Soft(g)
    meta, L, D, B, Q, N = (load(cam, n) for n in ("meta", "ledger", "diff", "brief", "qa", "near"))
    ids, labels = day_info(meta)
    # blind total: one number everywhere
    total = ledger_blind_total(L)
    if total is not None:
        claims: list[tuple[str, float]] = []
        for name in ("near", "brief", "qa", "moments", "synopsis", "diff", "index", "flow"):
            obj = load(cam, name)
            if name == "synopsis" or obj is None:
                continue
            for p, v in find_key(obj, "blind_total_s"):
                if isinstance(v, (int, float)):
                    claims.append((f"{name}.json {p}", float(v)))
        if isinstance(B, dict):
            for s, v in text_blind_totals([B.get("headline") or ""] + [str(x) for x in B.get("honesty") or []]):
                claims.append((f"brief: '{s}'", v))
        if isinstance(Q, dict) and isinstance(Q.get("blind"), list) and Q["blind"]:
            led = [b for b in Q["blind"] if b.get("source", "ledger") == "ledger"]
            if len(led) != len(Q["blind"]):
                g.warn(cam, f"qa.json mixes {len(Q['blind']) - len(led)} non-ledger blind spans "
                            f"(scan) with the ledger's; Ask honesty can print a different total")
            claims.append(("qa.json blind (ledger spans)", round(span_union(led), 2)))
        bad = [(w, v) for w, v in claims if abs(v - total) > TOL_S]
        if bad:
            for w, v in bad:
                g.fail(cam, f"blind total {v:g} s in {w} != ledger {total:g} s")
        else:
            g.ok(cam, f"blind total {total:g} s agrees in {len(claims)} places")
    # diff counts vs brief and Ask
    counts = D.get("counts") if isinstance(D, dict) else None
    if isinstance(counts, dict):
        if isinstance(B, dict):
            texts = [B.get("headline") or ""] + [b.get("text", "") for b in B.get("bullets") or [] if isinstance(b, dict)
                                                 and b.get("kind") in ("left", "arrived", "swapped", "moved", "same", "diff")]
            seen = {}
            for s in texts:
                for k, v in text_counts(s).items():
                    seen.setdefault(k, v)
            bad = {k: (v, counts.get(k)) for k, v in seen.items() if counts.get(k) is not None and v != counts.get(k)}
            if bad:
                g.fail(cam, f"brief vs diff.counts disagree: {bad} (brief, diff)")
            elif seen:
                g.ok(cam, f"brief agrees with diff.counts on {sorted(seen)}")
        it = between_days_item(qa_items(Q), labels)
        if it is not None:
            a = ans(it)
            if str(a.get("answer", "")).upper() == "UNKNOWN":
                ok_align = (D.get("alignment") or {}).get("ok", True) if isinstance(D.get("alignment"), dict) else True
                (g.fail if ok_align else g.warn)(
                    cam, f"Ask '{it.get('question')}' says UNKNOWN while the diff tab shows counts {counts}")
            else:
                got = text_counts(f"{a.get('answer', '')}. {a.get('sentence', '')}")
                bad = {k: (v, counts.get(k)) for k, v in got.items() if counts.get(k) is not None and v != counts.get(k)}
                (g.fail if bad else g.ok)(cam, f"Ask between-days answer vs diff.counts: "
                                               f"{'disagree ' + str(bad) if bad else 'agree ' + str(got)}")
    # honesty / naming
    real_days = bool(ids) and not set(ids) <= {"A", "B"}
    all_text: list[tuple[str, str]] = []
    for p in sorted((OUT / cam).glob("*.json")):
        if p.stem in ("synopsis", "ask_index"):
            continue
        obj = load(cam, p.stem)
        all_text += [(p.name, s) for s in strings(obj)]
    if real_days:
        hits = [(f, s) for f, s in all_text if re.search(r"\bDay [AB]\b", s)]
        if hits:
            g.fail(cam, f"real day ids {ids} but {len(hits)} strings say 'Day A/B', e.g. {hits[0][0]}: '{hits[0][1][:70]}'")
        sim = (meta or {}).get("simulated")
        hits = [(f, s) for f, s in all_text if "SIMULATED" in s.upper()]
        if sim:
            g.fail(cam, f"real days but meta.simulated = {sim}")
        if hits:
            g.fail(cam, f"real days but {len(hits)} strings say SIMULATED, e.g. {hits[0][0]}: '{hits[0][1][:70]}'")
    if cam not in export.DEV_CAMS:
        hits = [(f, s) for f, s in all_text if any(c.lower() in s.lower() for c in CITIES)]
        if hits:
            g.fail(cam, f"event cam names a city in {hits[0][0]}: '{hits[0][1][:70]}'")
    I = load(cam, "index")
    if isinstance(I, dict):
        coloured = [o.get("object_id") for o in I.get("objects") or [] if o.get("name") == "person" and o.get("colour")]
        if coloured:
            g.fail(cam, f"index.json gives {len(coloured)} people a colour (e.g. #{coloured[0]})")
    pat = re.compile(r"\b(person|people|pedestrian)\b[^.]{0,25}\(in (" + "|".join(COLOURS) + r")\)", re.I)
    hits = [s for f, s in all_text if f == "qa.json" and pat.search(s)]
    if hits:
        g.fail(cam, f"qa.json describes a person by colour: '{pat.search(hits[0]).group(0)}'")


def check_server(g: Gate, server: str, cams: list[str]) -> dict[str, dict]:
    """GET /api/ask for every qa.json question. Returns {cam: {question: live answer}}."""
    host = urllib.parse.urlparse(server).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        g.fail("server", f"refusing non-local server {server}")
        return {}
    live: dict[str, dict] = {}
    try:
        urllib.request.urlopen(f"{server}/api/cams", timeout=3).read()
    except Exception as e:  # noqa: BLE001
        g.warn("server", f"{server} not reachable ({type(e).__name__}); latency check skipped")
        return {}
    walls: list[float] = []
    for cam in cams:
        live[cam] = {}
        for it in qa_items(load(cam, "qa")):
            q = it.get("question")
            if not q:
                continue
            params = {"cam": cam, "q": q}
            if it.get("t") is not None:
                params["t"] = it["t"]
            t0 = time.perf_counter()
            try:
                r = json.loads(urllib.request.urlopen(f"{server}/api/ask?{urllib.parse.urlencode(params)}",
                                                      timeout=30).read())
            except Exception as e:  # noqa: BLE001
                g.fail(cam, f"/api/ask failed for '{q}': {type(e).__name__}")
                continue
            walls.append((time.perf_counter() - t0) * 1000)
            live[cam][q] = r
            if str(r.get("answer")) != str(ans(it).get("answer")):
                g.warn(cam, f"live answer differs from qa.json for '{q}': {r.get('answer')!r} vs "
                            f"{ans(it).get('answer')!r} (rerun qa.json)")
    if walls:
        p50, mx = statistics.median(walls), max(walls)
        (g.ok if p50 < 100 else g.fail)("server", f"/api/ask x{len(walls)}: p50 {p50:.0f} ms, max {mx:.0f} ms")
    return live


def run_export(g: Gate) -> None:
    r = subprocess.run([sys.executable, "-m", "stm.export"], cwd=str(ROOT), capture_output=True, text=True,
                       timeout=600)
    head = [ln for ln in r.stdout.splitlines() if ln.startswith(("index.html", "images", "redacted", "capped"))]
    for ln in head:
        print("  export:", ln)
    (g.ok if r.returncode == 0 else g.fail)("export", head[0] if head else f"exit {r.returncode}: {r.stderr[-200:]}")


# ---------- cue card ----------

def cue_for_cam(cam: str, live: dict[str, dict]) -> list[str]:
    meta, L, D, B, Q, N = (load(cam, n) for n in ("meta", "ledger", "diff", "brief", "qa", "near"))
    ids, labels = day_info(meta)
    m = meta or {}
    out = [f"== {cam}: {m.get('title') or ''} ==",
           "Days: " + (", ".join(f"{lab} ({i})" for i, lab in zip(ids, labels)) or "one clip")
           + ("   SIMULATED: " + "; ".join(m.get("simulated")) if m.get("simulated") else "   (real recordings)")]
    if isinstance(B, dict) and B.get("headline"):
        out.append(f"Brief headline (key 0): {B['headline']}")
    total = ledger_blind_total(L)
    blind = merge_spans((L or {}).get("blind") or []) if isinstance(L, dict) else []
    if total is not None:
        longest = max(blind, key=lambda s: s["end"] - s["start"]) if blind else None
        out.append(f"Camera could not see: {dur_text(total)} in {len(blind)} stretches"
                   + (f"; longest {mmss(longest['start'])}-{mmss(longest['end'])} "
                      f"({'/'.join(sorted(longest['reasons']))})" if longest else ""))
    # Near hook: the parked car with most visitors on foot
    if isinstance(N, dict) and N.get("targets"):
        foot = lambda t: [v for v in t.get("visitors") or [] if v.get("name") in ("person", "bicycle")]
        best = max(N["targets"], key=lambda t: (len(foot(t)), -t.get("object_id", 0)))
        f = foot(best)
        veh = [v for v in best.get("visitors") or [] if v not in f]
        bl = sum(b["end"] - b["start"] for b in best.get("blind") or [])
        out.append(f"Near hook (key 1): {best.get('name')} #{best.get('object_id')} parked "
                   f"{mmss(best.get('start'))}-{mmss(best.get('end'))}: {len(f)} on foot, {len(veh)} vehicles came near; "
                   f"blind {bl:.1f} s while it was there")
        if f:
            out.append("   on foot: " + ", ".join(f"{v.get('name')} #{v.get('object_id')} at {mmss(v.get('closest_t'))}"
                                               for v in sorted(f, key=lambda v: v.get("closest_t") or 0)))
    # Ask beats
    items = qa_items(Q)
    for it in items:
        a = ans(it)
        typ = a.get("type") or it.get("type")
        q = it.get("question")
        a = live.get(q, a)
        if typ == "still_there" and str(a.get("answer")).upper() == "UNKNOWN":
            ev = [e.get("t") for e in a.get("evidence") or [] if isinstance(e.get("t"), (int, float))]
            last = a.get("last_seen") if isinstance(a.get("last_seen"), (int, float)) else (max(ev) if ev else None)
            after = [s for s in blind if last is not None and s["end"] > last]
            span = after[0] if after else None
            out.append(f"Ask CAN'T TELL (key 2): '{q}' -> UNKNOWN; last seen {mmss(last)}"
                       + (f"; camera could not see {mmss(span['start'])}-{mmss(span['end'])} "
                          f"({span['end'] - span['start']:.0f} s, {'/'.join(sorted(span['reasons']))})" if span else ""))
            break
    if "__refusal__" in live:
        r = live["__refusal__"]
        out.append(f"Ask 'Who dented the car?' (live) -> {r.get('answer')}: {r.get('sentence')}")
    for it in items:
        a = live.get(it.get("question"), ans(it))
        if (a.get("type") or it.get("type")) == "who_near":
            out.append(f"Ask who came near: '{it.get('question')}' -> {a.get('answer')}")
            break
    it = between_days_item(items, labels)
    counts = D.get("counts") if isinstance(D, dict) else None
    if isinstance(counts, dict):
        a, b = (labels + ["day 1", "day 2"])[:2]
        out.append(f"Diff {a} vs {b} (key 4): " + ", ".join(f"{counts.get(k, 0)} {k}" for k in KINDS)
                   + (f"; {len(D.get('not_compared') or [])} not compared" if isinstance(D.get("not_compared"), list) else ""))
    if it is not None:
        a = live.get(it.get("question"), ans(it))
        out.append(f"Ask '{it.get('question')}' -> {a.get('answer')}")
    return out


def cue_card(cams: list[str], g: Gate, live: dict[str, dict]) -> str:
    demo = [c for c in cams[:1] + ["ptz_parking", "yonge_dundas"] if c in cams]
    demo = list(dict.fromkeys(demo))
    lines = [f"STREET TIME MACHINE - CUE CARD   generated {time.strftime('%Y-%m-%d %H:%M')} from out/",
             f"Gate: {g.count('FAIL')} FAIL, {g.count('WARN')} WARN (scripts/freeze_demo.py). "
             f"Read numbers from here, not from memory.",
             f"Default cam: {cams[0] if cams else '-'}   fallback: ptz_parking   rewind wow shot: yonge_dundas", ""]
    for cam in demo:
        lines += cue_for_cam(cam, live.get(cam, {})) + [""]
    C = load("captions", "captions")
    if isinstance(C, dict):
        bad = [r for r in C.get("rows") or [] if r.get("verdict") == "contradiction"]
        tot = C.get("totals") or {}
        lines.append(f"Check the AI (key 5): {tot.get('consistent', 0)} consistent, {tot.get('contradiction', 0)} "
                     f"contradiction, {tot.get('unverifiable', 0)} unverifiable")
        for r in bad[:2]:
            lines.append(f"   {r.get('clip')} {r.get('start', 0):.1f}-{r.get('end', 0):.1f} s: "
                         f"\"{(r.get('caption') or '')[:90]}\" -> detector and tracker saw nothing"
                         + (f" (model {r['cosmos_model']})" if r.get("cosmos_model") else ""))
    seg = export.ROOT / "data" / "packd" / "segments.json"
    if seg.is_file():
        try:
            S = json.loads(seg.read_text(encoding="utf-8"))
            segs = S.get("segments") if isinstance(S, dict) else S
            n = sum(1 for s in segs or [] if isinstance(s, dict) and s.get("reasoning_content"))
            lines.append(f"Event camera captions: {n} Cosmos captions, one per 5 s segment")
        except (OSError, json.JSONDecodeError):
            pass
    lines += ["",
              "RUN SHEET (?demo=1)",
              "0:00 key 1 Who came near (hook). 'It tells you who was near, never who did it.'",
              "0:25 key 2 Ask: who came near (fast) -> 'Is the ... still there?' -> CAN'T TELL -> 'Who dented the car?' -> refusal",
              "1:00 key 6 Rewind yonge_dundas, spotlight people",
              "1:20 key 5 Check the AI: the drone (pause for the laugh)",
              "1:40 key 4 Yesterday vs today / Blink (skip if the aligned version is not merged)",
              "1:52 key 0 brief headline and close",
              "",
              "SCHEDULE",
              "15:00 feature freeze, Pack D go/no-go",
              "15:30 stop all writes to out/; commit out/ and ui/ ('demo freeze')",
              "15:45 and 16:10 timed rehearsals",
              "16:05 screen recording as last-resort backup",
              "16:30 submit",
              "",
              "Never: name a city for the event camera, describe a person by appearance, say who did it."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default=None, help="running server to time, e.g. http://127.0.0.1:8780 (localhost only)")
    ap.add_argument("--no-export", action="store_true", help="do not run python -m stm.export")
    ap.add_argument("--cue", default=str(ROOT / "CUECARD.txt"))
    a = ap.parse_args(argv)
    cams = export.available_cams()
    demo = set(cams[:1]) | {"ptz_parking", "yonge_dundas"}
    g = Gate()
    check_videos(g, cams, demo)
    check_images(g, cams, demo)
    for cam in cams:
        check_numbers(g, cam, strict=cam in demo)
    live = check_server(g, a.server, cams) if a.server else {}
    if live and cams:
        try:  # the blame refusal, as the server says it now
            q = urllib.parse.urlencode({"cam": cams[0], "q": "Who dented the car?"})
            r = json.loads(urllib.request.urlopen(f"{a.server}/api/ask?{q}", timeout=30).read())
            live.setdefault(cams[0], {})["__refusal__"] = r
            (g.ok if str(r.get("answer")).upper() in ("UNKNOWN", "REFUSED", "NO") or "never" in
             str(r.get("sentence", "")).lower() else g.warn)(cams[0], f"'Who dented the car?' -> {r.get('answer')}: "
                                                                     f"{str(r.get('sentence'))[:80]}")
        except Exception as e:  # noqa: BLE001
            g.warn(cams[0], f"'Who dented the car?' failed on the server ({type(e).__name__})")
    if not a.no_export:
        run_export(g)
    Path(a.cue).write_text(cue_card(cams, g, live), encoding="utf-8")
    print(g.text())
    print(f"\n{g.count('FAIL')} FAIL, {g.count('WARN')} WARN, {g.count('OK')} OK; cue card -> {a.cue}")
    return 1 if g.count("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
