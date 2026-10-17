"""Auto-drafted incident log at hang-up: timeline with times, built only from engine events + the call transcript."""
from __future__ import annotations

# make the module importable on its own (protokit lives in prototypes/)
import sys as _sys  # noqa: E402
from pathlib import Path as _P  # noqa: E402
_HERE_PATHS = [str(_P(__file__).resolve().parents[1]), str(_P(__file__).resolve().parent)]
_sys.path[:0] = [x for x in _HERE_PATHS if x not in _sys.path]

from dataclasses import dataclass
from typing import Any

import numpy as np

import protokit as pk
from engine import GROUP_PLURAL, GROUP_WORD, Entity, SceneState, display_colour
from qa import Card, _side


@dataclass
class LogLine:
    t: float
    kind: str          # alarm | arrived | departed | qa | summary
    text: str
    eid: int | None = None


def _who(st: SceneState, e: Entity) -> str:
    f = st.facts(e)
    c = f["colours"]
    if e.group == "vehicle":
        base = f"{c.get('body', '')} {e.cls_name}".strip()
    elif e.group == "person":
        base = "person" + (f" ({display_colour('person', c['top'])} top, {display_colour('person', c['bottom'])} trousers)" if c.get("top") and c.get("bottom") else "")
    else:
        base = "bicycle"
    return base


def draft(st: SceneState, cards: list[Card], title: str, hangup_t: float, max_lines: int = 16) -> dict[str, Any]:
    """Returns {header, summary, lines:[LogLine], text}. Everything is derived from state at hang-up."""
    asked = {c.target_eid for c in cards if c.target_eid}
    ents = sorted(st.confirmed(), key=lambda e: e.first_t)
    lines: list[LogLine] = [LogLine(st.alarm_t, "alarm", f"Alarm received, camera: {title}.")]
    at_alarm = [e for e in ents if e.at_alarm]
    for g in ("person", "vehicle", "bicycle"):
        n = sum(1 for e in at_alarm if e.group == g)
        if n:
            lines.append(LogLine(st.alarm_t, "alarm", f"{n} {GROUP_PLURAL[g] if n != 1 else GROUP_WORD[g]} already on scene."))
    ev_lines: list[LogLine] = []
    crowded = len(ents) > 9          # a crowd is summarised; only the people the police asked about get their own log lines
    for e in ents:
        if crowded and e.eid not in asked and e.group == "person":
            continue
        f = st.facts(e)
        dwell = e.last_t - e.first_t
        interesting = e.eid in asked or f["exit_side"] or (e.group != "person") or dwell >= 3.0
        if not interesting or (dwell < 1.0 and e.eid not in asked):
            continue
        if not e.at_alarm:
            side = f" via the {_side(f['entry_side'])} edge" if f["entry_side"] else ""
            ev_lines.append(LogLine(e.first_t, "arrived", f"{_who(st, e).capitalize()} arrived{side}. [#{e.eid}]", e.eid))
        if not f["present"]:
            how = f"left via the {_side(f['exit_side'])} edge" if f["exit_side"] else "vanished from view mid-frame"
            ev_lines.append(LogLine(e.last_t, "departed", f"{_who(st, e).capitalize()} {how}. [#{e.eid}]", e.eid))
    # keep the log readable: prefer entities that police asked about, then vehicles, then the rest by time
    keep = ev_lines
    if len(keep) > max_lines:
        pri = sorted(keep, key=lambda l: (0 if l.eid in asked else (1 if "person" not in l.text.lower()[:10] else 2), l.t))
        keep = sorted(pri[:max_lines], key=lambda l: l.t)
    lines += keep
    for c in cards:
        lines.append(LogLine(c.t, "qa", f"Police asked \"{c.question}\" -> {c.headline}."))
    lines.sort(key=lambda l: (l.t, 0 if l.kind == "alarm" else 1))
    still = st.present()
    tot = {g: len(st.confirmed(g)) for g in ("person", "vehicle", "bicycle")}
    summary = [f"Seen since alarm: " + ", ".join(f"{n} {GROUP_PLURAL[g] if n != 1 else GROUP_WORD[g]}" for g, n in tot.items() if n) + ".",
               f"On scene at hang-up ({pk.mmss(hangup_t)}): {len(still)} ({', '.join(f'{sum(1 for e in still if e.group == g)} {g}' for g in ('person', 'vehicle', 'bicycle') if any(e.group == g for e in still)) or 'none'})."]
    if crowded:
        others = sum(1 for e in ents if e.group == "person" and e.eid not in asked)
        summary.append(f"{others} other people moved through the scene (not itemised: crowd).")
    left = [e for e in ents if now_gone(st, e) and st.facts(e)["exit_side"]]
    if left:
        summary.append(f"{len(left)} left through a frame edge: " + ", ".join(sorted({_side(st.facts(e)['exit_side']) for e in left})) + ".")
    header = f"INCIDENT LOG (auto-draft, operator to review) | {title} | alarm {pk.mmss(st.alarm_t)} | hang-up {pk.mmss(hangup_t)}"
    text = header + "\n" + "\n".join(f"{pk.mmss(l.t)}  {l.text}" for l in lines) + "\n" + "\n".join(summary)
    return {"header": header, "summary": summary, "lines": lines, "text": text}


def now_gone(st: SceneState, e: Entity) -> bool:
    return st.now - e.last_t > st.hold
