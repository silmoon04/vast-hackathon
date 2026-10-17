# STREET TIME MACHINE: the 2-minute demo

This outline is for prepared video outputs. A fresh source checkout opens an illustrated synthetic fixture instead. Check every example against the current camera before presenting it.

**Rewind any street camera and ask it what happened, honestly.**

VAST "Real-Time Video Agents" builders hack, Saturday 17 October 2026. Submissions 16:30, awards 18:30.

## Before you go on stage

1. Start the app: `$PY -m uvicorn stm.server:app --port 8780`, open http://127.0.0.1:8780.
   Backup with no server: `dist/index.html` from `$PY -m stm.export`.
2. Pick the camera. Use Pack D if it is registered and its outputs exist. Otherwise use `ptz_parking`.
3. Get the real numbers. Run the check below (Git Bash, repo root) for that camera. Put what it prints into the `[slots]` in the
   script. **Never say a number it did not print.** If a line says MISSING, cut that beat.
4. Check the SIMULATED label. If the camera shows one, say "simulated" out loud.
5. Try each Ask question once before the demo. Use only questions whose answer you checked against the crop.

```bash
PY="python"
$PY - ptz_parking <<'EOF'
import json, statistics, sys
from pathlib import Path
cam = sys.argv[1]; o = Path("out") / cam
def j(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
s, d, n, l, q, c = (j(o / "synopsis.json"), j(o / "diff.json"), j(o / "near.json"),
                    j(o / "ledger.json"), j(o / "qa.json"), j("out/captions/captions.json"))
if s: print("synopsis:", s["src_duration"], "s ->", s["out_duration"], "s,", len(s["ghosts"]), "ghosts")
if d: print("diff:", d["counts"])
if n and n["targets"]: print("near:", len(n["targets"]), "targets; first:", n["targets"][0]["summary"])
if l: print("ledger: blind", sum(not x["visible"] for x in l["per_second"]), "of", len(l["per_second"]), "s")
if q: v = [r["latency_ms"] for r in q["items"]]; print("ask: median", round(statistics.median(v), 1), "ms, max", max(v), "ms")
if c: print("captions:", c["totals"])
for name, x in zip("synopsis diff near ledger qa captions".split(), (s, d, n, l, q, c)):
    if x is None: print(name, "MISSING")
EOF
```

## The script (2:00)

| Time | Beat |
|---|---|
| 0:00 | The problem |
| 0:15 | 1. Rewind |
| 0:40 | 2. Yesterday vs today |
| 1:00 | 3. Who came near my car |
| 1:20 | 4. Ask |
| 1:40 | 5. Check the AI caption |
| 1:55 | Close |

### 0:00 The problem

> "Street cameras record all day. Nobody watches. When something happens, someone scrubs through hours of
> video. Now AI writes a caption for every clip. But captions can be wrong, and they always sound sure.
> So we built a camera you can rewind and question. And it tells you when it could not see."

### 0:15 Money shot 1: Rewind

**Do:** open the camera. The synopsis plays. Click one ghost.

> "This is [src_duration] seconds of video in [out_duration] seconds. [ghosts] ghosts: every car and
> person, all on screen at once. Each one carries its real time. Click it, and we jump to that moment in
> the real footage."

**Proof on screen:** each ghost's time label comes from `synopsis.json` (`src_start`, `label`).

### 0:40 Money shot 2: Yesterday vs today

**Do:** open the diff view. Show the two plates. Hover one change.

> "Day A against Day B. [same] cars stayed. [left] left. [arrived] arrived. [swapped] spots now hold a
> different car. Every claim has a before and an after crop."

- On `ptz_parking` add: "These two days are SIMULATED. They are two halves of one clip, and the screen says so."
- On Pack D add: "Two real days: the first and second of September."
- If asked about the edges: the camera's two views do not overlap fully. Parts only one day can see are
  "not compared". They are never called "left".

### 1:00 Money shot 3: Who came near my car

**Do:** click a parked car. Point at the grey part of its strip.

> "Who came near my car? [near summary, read as printed]. The grey part is blind time. The camera could not
> see then, so who was near is UNKNOWN. We say that out loud. And this is who was near. It is not who did it."

### 1:20 Money shot 4: Ask

**Do:** type a checked question, such as "Is the red car still there?". Then "How many people walked by?".

> "Plain questions. Answers in milliseconds: the time is on each card. Every answer brings its proof: the
> crop, the time, the box. If the camera was blind, the answer is UNKNOWN, not NO."

**Proof on screen:** the latency chip and the evidence crops. Typical speed on this camera: [ask median] ms.

### 1:40 Money shot 5: Check the AI caption

**Do:** open the caption check. Click the contradiction row.

> "The pipeline writes a caption for every clip. We check each claim against what the detector counted.
> [consistent] agree. [contradiction] contradicts. [unverifiable] we cannot check, and we say so. This
> one says a drone flew over the car park. No detector saw anything in that clip."

- Check the row first. At the time of writing it was `person-bicycle-car-detection_005` (20-25 s).
- Say where the captions came from. In dev it is a local mock corpus in the pipeline's row shape. Say
  "pipeline captions" only if the rows really came from the event pipeline.

### 1:55 Close

> "Rewind it. Compare the days. Ask it. Check the AI. It never fills a gap with a guess."

## The trust story

- **UNKNOWN when it could not see.** Dark, blurred, frozen, blocked or turned-away time is grey. A car is
  never "gone" while the camera is blind.
- **"Who was near", never "who did it".** Being close is all the data shows. It names no one: no face or
  plate recognition.
- **SIMULATED labels.** Anything staged says SIMULATED on screen.
- **People can be blurred.** Only say this if the blur option is in the build you are showing. At the time
  of writing it was not built yet.
- **Every number is computed.** Each output folder has a `*_summary.txt` with its counts.
- **Footage stays local.** Internet test clips never go into the event's systems.

## How it uses the VAST pipeline

- **Detector sidecars.** The pipeline cuts video into short segments and runs YOLO on each frame. It
  writes one detection sidecar per segment: boxes, labels, confidence, `time_sec`.
- **Captions.** The caption model writes a description of each segment (`reasoning_content`).
- **VastDB rows.** One row per segment in the shared collection ties it together: segment start and end,
  `object_counts`, `detection_sidecar_uri`.
- **What we add.** `stm.packd` reads those rows and sidecars and puts segments on one timeline. Sidecars
  have no track ids, so `stm.tracker` links boxes into tracks. Then every feature runs unchanged, because
  each one reads only through `stm.data`. `stm.captions` checks the captions against the counts and our
  tracks. We only read; we never write to the shared tables.
- **Source on screen.** Say whether the current camera uses event sidecars, local detector tracks or the
  bundled synthetic fixture. Check its current outputs before describing integration or results.

## Three judge questions, honest answers

**"Isn't this just YOLO with a nice UI? Where is the VAST stack?"**
The boxes and captions are the pipeline's: its sidecars and VastDB rows are our input. We add what the
pipeline does not give you: tracks, blind time, parking spots, day-to-day matching, and a check on its
captions. If Pack D is not wired in at demo time, say so: "Today's demo runs on our local caches of public
clips. The same code reads Pack D through one adapter."

**"How accurate is it? How do you know?"**
We have no hand-labelled ground truth for Pack D, and our demo questions are written and checked by us. So
we do not claim an accuracy number. Instead every answer carries its proof: a crop, a time and a box. You
can check any answer in a second. When it cannot see, it says UNKNOWN. If `out/tracker_eval.json` exists,
you can also say how our tracker compares with ByteTrack (purity and fragmentation), using its numbers only.

**"Couldn't this be used to blame someone?"**
It is built not to. It says who was near, never who did it. It does no face or plate recognition. Blind
time is shown, not hidden. Staged data is labelled SIMULATED. Test footage stays on our laptop.
