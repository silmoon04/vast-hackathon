"""Render the pitch workflow result to PITCHES.md (ranked), and print a compact digest.

usage: python render_pitches.py PITCH_WORKFLOW_OUTPUT.json [OUT.md]
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
r = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
r = r.get("result", r)
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "PITCHES.md"
pitches = {p["n"]: p for p in r["pitches"]}
rank = r["rank"]
order = sorted(rank["ranking"], key=lambda x: x["rank"])

out = ["# 15 prototypes, pitched from their real outputs", "",
       "Built and tested on local stand-ins and openly licensed clips. "
       "Each pitch was written from the prototype's README, results and hero image, plus a "
       "skeptical reviewer's verdict. Visual gallery: gallery.html.", "",
       "## The decision", "", f"**Plan A.** {rank['plan_a']}", "", f"**Plan B.** {rank['plan_b']}", "",
       f"**Fold in.** {rank['combine']}", "", f"**Overall.** {rank['overall']}", "",
       "| Rank | Idea | Verdict | Win score | Why |", "|---|---|---|---|---|"]
for x in order:
    p = pitches.get(x["n"], {})
    out.append(f"| {x['rank']} | {x['name']} | {p.get('verdict', '?')} | {p.get('win_score', '?')}/10 | {x['why']} |")
for x in order:
    p = pitches.get(x["n"])
    if not p:
        continue
    out += ["", f"## {x['rank']}. {p['name']}: {p['tagline']}", "",
            f"*{p['verdict']}, win score {p['win_score']}/10. Folder: {p['n']}_*, hero: {p['best_hero']}*", "",
            p["pitch"], "", "**Proof (measured):**"] + [f"- {b}" for b in p["proof"]] + [
            "", f"**What you see:** {p['what_you_see']}", "", f"**The catch:** {p['catch']}", "",
            f"**On the day:** {p['on_the_day']}"]
if r.get("missing"):
    out += ["", "Missing pitches: " + ", ".join(r["missing"])]
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote", dst, len("\n".join(out)), "chars")
