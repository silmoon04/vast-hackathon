# 14 LANE CALL: stopped-vehicle alert triage

Local prototype. Every number below was measured by `run.py` on the local
clips and is in `results.json`; anything not measured is labelled. This is the second version, after a review that called the first
demo staged (slideshow trigger, hand-picked positives, self-drawn labels). What changed is in "Fixes since the review" at the end.

## What it is

A control-room operator gets a severity-ordered queue of stopped-vehicle alerts. Each card has an 8 s clip, the boxed vehicle
crop, the vehicle's ground-contact line (from a segmentation mask), a call IN LANE / AT KERB / CAN'T SEE with a confidence and a
reason (from lane polygons drawn once per camera preset), and a Cosmos second opinion that is shown but never decides. Stopped
vehicles that are stacked in one lane and appeared together collapse into ONE low-severity QUEUE card. National Highways verified
only 564 of 3,850 alerts: this is the triage step in front of the human.

Run: `python run.py --all --bonus` (about 2 min with the VLM / seg / detection caches warm; the first cold run in this session took
about 5 min on a laptop whose GPU was shared with other jobs). `python run.py --video NAME` runs one clip and MERGES into
`results.json` (it no longer overwrites the aggregate). Extra clip: `--video motorway_a40_inject`. Options: `--S 5` dwell seconds,
`--k 0.12` stillness tolerance (fraction of the box diagonal), `--no-vlm`, `--no-clips`. `python test_smoke.py` (about 7 s, 16 tests).
Outputs: `out/<video>/report.html` (the queue), `hero.jpg`, `cards/*` (crop, annotated frame, clip), `eyeball_*.jpg`,
`callview_*.jpg` (tiles with the contact line and the call), `merged_pairs.jpg` (PTZ), `results.json`.

Pipeline (`lanecall_core.py`, `lanecall.py`, `run.py`, `inject.py`):
1. Detector: YOLO26s boxes (cached `pk.tracks`). A vehicle is stopped when a spatially matched candidate keeps its centre within
   `k = 0.12 * box diagonal` of its anchor for >= S = 5 s with >= 60 % detection presence. Matching is by position, not track id, so
   ID switches do not reset the clock. A stopped car that is hidden for up to 3.5 s by a truck passing in front and re-seeded at the
   same pixels (IoU > 0.85) is the same car (found when the injected A40 car alerted twice).
2. Geometry: lane / kerb polygons drawn per camera preset on a reference frame (`zones.json`); the alert frame is registered to the
   preset by ORB + RANSAC homography so PTZ presets and small shifts still map; no match means CAN'T SEE.
3. Ground contact: YOLO26s-seg on the alert frame; the lowest mask pixel per column (5-column running max, so the underbody between
   the wheels is not taken as contact) is the real ground line, diagonal for an angled car. 7 points along it are mapped into the
   reference frame. If no seg mask matches the alert box (IoU >= 0.45) the box bottom edge is used (10 of 61 PTZ alerts, 1 of 33 Lisbon).
4. Call: >= 30 % of the contact line on a lane polygon is IN LANE (partly in lane still blocks it), else inside a kerb/bay polygon is
   AT KERB; CAN'T SEE if the ground contact is cut off by the frame bottom, a sliver is in frame, too dark/flat, occluded by a nearer
   vehicle, no registration, or outside every polygon. IN LANE confidence now follows the in-lane fraction (100 % -> 97 %, 71 % ->
   about 80 %, 57 % -> 70 %, 30 % -> 54 %), scaled down for visibility problems and weak registration.
5. Collapse: IN LANE alerts of the same camera and lane zone whose stop times are within 15 s (chained) and whose centres are within
   5 mean box widths (chained) form a group; a group of >= 3 becomes one QUEUE card (members listed as thumbnails) ranked below lone IN LANE and
   CAN'T SEE, above AT KERB. PTZ repeats: every alert's ground point is mapped into preset A's reference frame (homography between
   the preset reference frames) and alerts of the same car seen in another preset are merged (the better view is kept).
6. Queue order: IN LANE, CAN'T SEE, QUEUE, AT KERB, then confidence and dwell. The same order is reproduced with VastDB-style SQL on
   duckdb (`sql_queue_matches_python` is true for every video; the SQL avoids IN / GROUP BY / JOIN).
7. Cosmos second opinion (local Cosmos3-Edge, <= 28 calls per video, cached) on the red-boxed frame; collapsed under each card.

## Demo script (90 s)

1. 0:00 A40 hero (`out/motorway_a40_inject/hero.jpg`). Say first: "real motorway footage, three real cars cut out of the same clip and
   pasted in as stopped vehicles: SIMULATED". "The real detector fires after 5 s on all three. Two are in the running lanes: IN LANE,
   first in the queue, 97 %. The one on the hard shoulder drops to AT KERB. The control panel: the same clip without the pasted
   cars gives 0 alerts." Point at the white ground-contact lines.
2. 0:30 Lisbon hero (`out/lisbon_parking/hero.jpg`). "This clip is a photo slideshow, so the trigger is simulated and the polygons are
   mine (stills, no dwell time shown). 5 IN LANE first: a tow truck, a van and a car on zebra crossings, a car in the road, and the
   borderline one at 70 %: a car half up on the kerb. Say so: it is the honest limit." Open card Q01 in `report.html`.
3. 0:50 PTZ parking lot hero. "87 alerts from one lot; 26 are the same parked cars seen again by another preset, merged. 46 AT KERB are
   auto-lowered, nothing in a lane, 15 CAN'T SEE (car cut off at the frame edge: the system will not guess)." Open an 8 s clip.
4. 1:10 Yonge-Dundas hero (bonus). "7 cars stopped on the carriageway: all waiting at a red light. They collapse into 1 QUEUE card plus
   one lone IN LANE car and one CAN'T SEE: 7 alerts, 3 cards. IN LANE is not the same as an incident; the queue card is the first
   answer to that."
5. 1:25 "Alert to card: median 0.5 s including the segmentation pass, budget 2 s." Do not open the Cosmos row (see limits).

## How it maps to the VAST stack on the day

| This spike (local) | On the day | Note |
|---|---|---|
| YOLO26s detections (cached) | hosted YOLO11s (DataEngine) | RISK: the DataEngine table gives COCO counts per 5 s segment. The detector needs boxes per frame. If the endpoint only returns counts we must run our own YOLO on the stream (the same code), or fall back to count heuristics (no position, no lane call). Check on arrival. |
| YOLO26s-seg on the alert frame (local GPU) | own YOLO-seg on the alert frame | NEW dependency for the contact line: about 90 to 460 ms per frame here (shared laptop GPU). Not part of DataEngine. If we have no GPU, fall back to the box edge (older, worse on angled cars). |
| `pk.tracks` replayed | live stream into the detector | alert-to-card timing is measured on replayed detections (frame already decoded) |
| local Cosmos3-Edge second opinion | Cosmos3-Reason on the crop / frame | prompt and JSON contract in `lanecall.py` (`VLM_PROMPT`); also the natural place for a plain-English caption. Test before showing disagreement chips: the local model never says IN LANE |
| duckdb alerts table + ORDER BY | VastDB table `alerts` (call, conf, reason, vlm_call, queue_id, operator_verdict) | queue = `ORDER BY sev, conf DESC, dur DESC, t_alert, x1`; counts via `WHERE call = ?` |
| not built | Cosmos-Embed1 on the alert clip | idea: "3 similar past alerts were: at kerb" from operator-verified alerts |
| not built | Canary ASR | not relevant to this idea |

## Results on test footage

`python run.py --all --bonus`, S = 5 s. "Alerts" are stopped vehicles the detector raised; "cards" are what the operator sees after the
PTZ repeat merge and the queue collapse. Alert-to-card is detector output to finished card record (registration, call, crop and annotated
frame written, plus the seg pass on that frame). Not included: replay frame seek (median 0 to 109 ms), the S-second wait, Cosmos
(median about 7 s per call, shown asynchronously) and clip encoding (about 1 s each).

| video | role | alerts -> cards | IN LANE / CAN'T SEE / QUEUE / AT KERB (cards) | card median / p95 / max (ms) | eyeball right |
|---|---|---|---|---|---|
| lisbon_parking | primary, 1,955-slide photo montage, trigger SIMULATED | 33 -> 33 | 5 / 9 / 0 / 19 | 184 / 268 / 277 | 31/33 (not independent) |
| motorway_a40 | real clip, expect 0 | 0 -> 0 | 0 / 0 / 0 / 0 | n/a | 0 alerts (also 0 at S = 3, 2, 1.5 s in the first version) |
| motorway_a40_inject | the same clip + 3 SIMULATED pasted stops, real detector | 3 -> 3 | 2 / 0 / 0 / 1 | 289 / 375 / 384 | 3/3 (truth by construction, checked by eye) |
| ptz_parking | PTZ, 2 presets drawn | 61 (87 raw, 26 merged) -> 61 | 0 / 15 / 0 / 46 | 590 / 784 / 938 | 59/61 |
| yonge_dundas | bonus, red-light queue | 7 -> 3 | 1 / 1 / 1 (x5 vehicles) / 0 | 475 / 558 / 567 | 7/7 |
| all | | 104 -> 100 | | 530 / 770 / 938 | 100/104 = 96.2 % |

Latency moves with laptop load (the GPU was shared with other jobs; the per-frame seg time is cached from the run that computed it); it was
always far under 2 s (max 938 ms in this final run; 1,388 ms was the worst single frame including seek in the ptz run).

* Alert to card under 2 s: yes (max 938 ms).
* IN LANE precision 12/13, recall 12/12, harmful misses (truth IN LANE called AT KERB) 0, false IN LANE 1 (Lisbon, the angled car, now shown
  at 70 % instead of 97 %), abstained when judgeable 3 (1 Lisbon, 2 ptz).
* Baselines on the same 104 judged alerts: "always AT KERB" 67 %, "always IN LANE" 12 %. The 96.2 % is mostly AT KERB (70 of 104 labels),
  so read the IN LANE row. The IN LANE positives are still few: 12 = 4 slideshow stills, 6 cars at one red light, 2 pasted cars.
* Not-Lisbon subset (ptz + yonge + injected; geometry drawn on a reference frame, alerts judged on other frames or by construction): 69/71.
* Injected A40: all three pasted cars alert at t = 5.0 s (the S = 5 s rule, so the real detector saw them stationary the whole time);
  calls IN LANE 97 % (left carriageway outer lane), IN LANE 97 % (right carriageway lane), AT KERB 84 % (right hard shoulder, 86 % of the
  contact line inside the shoulder polygon). Control: the same clip without the paste gives 0 alerts.
* Red-light flood: Yonge 7 alerts become 3 cards (1 QUEUE of 5, 1 lone IN LANE, 1 CAN'T SEE).
* PTZ: 87 raw alerts become 61 unique cars; 46 AT KERB are auto-lowered and listed first on the hero.
* Cosmos second opinion vs my labels (judgeable alerts, cards that got a call): AT KERB 16/16 Lisbon, 15/15 ptz, 1/1 injected;
  IN LANE **0/7** (0/4 Lisbon, 0/1 Yonge, 0/2 injected). The local Cosmos3-Edge answers AT KERB for everything with this prompt. It never
  decides, and its row is now collapsed on the cards.

## Eyeball checks (frames I viewed, and what I saw)

* This round: the gridded A40 frame (t = 8 s) and a 2x crop of the right carriageway with the polygons, to place the pasted cars: the
  right hard shoulder is a dark strip about 65 to 70 px wide right of the solid line at y = 600 to 650; two lanes in the right
  carriageway. Real tracked cars give lane centres (outer left lane about x = 345 and right lane about x = 810 at y = 605), used to
  place the IN LANE cars so the position does not come from the system's output. My first left-car placement (x = 190) landed on the
  edge of the shoulder polygon and was called AT KERB; I moved it to the lane centre. That is a placement correction, not a rule change.
* `hero.jpg` and `callview_1.jpg` of the injected A40: three pasted cars look like real cars (shadows, perspective), white contact lines
  sit on the tyres, the two IN LANE cars are on lane paint, the third is on the shoulder. One 8 s clip frame (Q01): the car is boxed and
  traffic drives past it.
* Lisbon `callview_1.jpg`: tow truck, a car stopped in the road before a crossing, a car angled on a zebra, a van on a crossing: all four
  have their contact line in the carriageway (IN LANE correct). The fifth IN LANE card (E11, a silver car parked up on the pavement edge,
  contact line along a hand-drawn polygon border, 57 % in lane, 70 % confidence) is a car at the kerb by my label: still a false IN LANE.
* `merged_pairs.jpg` (8 PTZ pairs, kept vs merged): 7 are clearly the same parked car (same neighbours, same bay), 1 is uncertain (a roof at
  the frame bottom merged into a fully visible SUV through a chain of merges). The other 18 merges were not viewed.
* PTZ hero and Yonge hero (queue frame with the five purple member boxes); the A40 clean hero.
* Earlier rounds (first version): contact sheets of all clips, gridded reference frames, PTZ eyeball sheets (every real vehicle parked in
  a bay; 17 cars cut off at the frame bottom or a sliver), Yonge 8 tiles, 26 deterministic Lisbon slides plus 7 hand-picked ones.
  The `report.html` files were checked for missing media (none); I could not open them in a browser pane here, so their layout has not been
  looked at.
* Ground truth is in `ground_truth.json` (t, centre, truth label; the A40 injected entries are by construction). Labels: IN LANE, AT KERB,
  UNJUDGEABLE. A call is right when it equals the label, or CAN'T SEE for UNJUDGEABLE.

## Simulated or faked

* **The three A40 stops are SIMULATED** (`inject.py`): three real cars cut out of the same clip (seg mask, feathered edge, contact shadow,
  scaled to lane perspective) are pasted static for the whole clip, nearer real vehicles are pasted back in front. They are stationary from
  t = 0 (no arrival), so the stop time shown is "already stopped at view start" and the alert fires at 5.0 s. The footage, the YOLO detector
  run on the re-encoded pasted frames (no tracker), the S = 5 s rule, registration, geometry and call are real. The detector found 3/3.
* **Lisbon trigger is SIMULATED.** The clip is a ~1,955-slide photo montage (median 0.32 s per photo), so the real 5 s rule fires 0 times on it.
  Each selected slide stands in for a stopped vehicle (the card says "still from a slideshow", not a dwell time). Lisbon polygons and labels
  are both mine, drawn per slide; slides where I drew nothing (8) come out CAN'T SEE and are labelled UNJUDGEABLE. The 7 hand-picked slides
  were chosen because they looked in-lane.
* The 96 % headline is not independent: polygons and labels are mine, the ptz polygons were tuned on the same footage (one edge moved after
  the first call run), and two abstention rules were relaxed after the first pass (86 %).
* Local stand-ins: YOLO26s (+ YOLO26s-seg) for hosted YOLO11s, Cosmos3-Edge for Cosmos3-Reason, duckdb for VastDB.
* "Live" is a replay of cached detections: timing excludes decoding and the S-second wait. PTZ alerts with `t_stop = 0` mean "already
  stationary at the view start", not an observed stop.
* Timings were taken while many other jobs shared the CPU and GPU.

## What fails / limits

* Evidence for the part that matters is still thin: 12 IN LANE ground-truth alerts. No real vehicle stopped in a running lane or on a shoulder of
  a motorway exists in any footage I have; the two pasted cars are easy cases (100 % in lane, clean silhouettes, static, a visible lane). No
  pasted car straddles the line, is occluded, or arrives from the road.
* The angled-car case is reduced, not solved. The contact line is now the silhouette's lower edge, and that same Lisbon car moved from "IN LANE
  97 %" to "IN LANE 70 %", but it is still called IN LANE. The result depends on 3-pixel details of a hand-drawn polygon border: with the raw
  mask edge the car was AT KERB (Lisbon 32/33), with a 5-column mean smoothing 30/33, with the running max I kept on physical grounds 31/33.
* Red-light and congestion stops are IN LANE; the queue collapse hides them as ONE card but does not know signal phase. A real motorway jam
  would collapse the same way. The leftmost Yonge car stayed a lone IN LANE card because it is more than 5 box widths from the others (the rule was
  not tuned to merge it).
* The queue rule is geometric (same lane zone, 15 s, 5 widths), not a metre-based distance: it needs no scale but is camera-dependent.
* PTZ merge: spot-checked on 8 of 26 pairs; a chain of merges can join two different cars.
* CAN'T SEE rules are crude: a car cut off at the frame bottom abstains even when it is plainly in a bay (12 of the 15 ptz abstentions); 3 abstentions were
  judgeable by eye (1 Lisbon, 2 ptz: a nearer-vehicle overlap and an 86 px edge sliver).
* Needs polygons per camera and per PTZ preset (the slowest part: several attempts per preset). A real build needs a drawing tool and a
  "this preset has no geometry" state.
* Confidence is a heuristic: it now follows the in-lane fraction, but it is not calibrated.
* Cosmos3-Edge second opinion is 0/7 on IN LANE (it answers AT KERB for everything). Hosted Cosmos3-Reason is unknown; the row is collapsed on cards
  and not in the demo.
* The detector sometimes fires on non-vehicles (cartoon car, window reflection) and flickering far cars; there is no second detector check.
* Clips and Cosmos opinions exist only for the top of each queue (budgets: 12 IN LANE, 4 CAN'T SEE, 2 QUEUE, 4 AT KERB clips; 28 VLM calls).
* The seg pass is a new GPU step: median 94 ms (Lisbon) to 463 ms (PTZ) per frame on the shared laptop GPU.

## Kill test result

Criterion: calls right for >= 80 % of alerts by my eyeball check. **Passed: 100/104 = 96.2 %** (lisbon 31/33, ptz 59/61, yonge 7/7, injected A40
3/3); not-Lisbon subset 69/71. Caveats that matter more than the percentage: the trivial "always AT KERB" baseline scores 67 %, there are 12 IN LANE
positives (4 slideshow, 6 red-light, 2 pasted), Lisbon is not independent and its trigger is simulated, the A40 stops are pasted, and two rules were
adjusted after seeing first-pass misses. The first version scored 126/132 = 95.5 % on a different set (before the PTZ repeat merge and the detector
change) so the two numbers are not comparable one to one.

## Rebuild plan on the day (hours)

About 8 h with the real services if the hosted YOLO endpoint returns boxes and we have a camera feed with a real stop:
1.5 h polygon drawing tool (click polygons on a reference frame, save to VastDB); 1 h detector on the YOLO stream (the logic is about 100 lines,
unit-tested incl. ID switches and occlusion); 0.5 h YOLO-seg ground-contact line (port as is, needs a GPU); 1 h homography registration and call
logic (port as is); 0.5 h queue collapse; 1.5 h queue UI with clips and the Cosmos-Reason call; 0.5 h VastDB alerts table and SQL; 1.5 h footage,
polygons and rehearsal (include a real or pasted shoulder-straddling case). Add 2 to 3 h if the endpoint only returns counts (own YOLO). Biggest
unknowns: whether we get footage with an in-lane stop, and how Cosmos3-Reason behaves on this prompt.

## Fixes since the review (this round)

1. **Real motorway use case exercised**: `inject.py` pastes 3 real cars (cut from the clip, seg mask) into the A40 clip; the real detector, the S = 5 s
   rule, registration, geometry, call and cards run end to end (3/3 alerts, 2 IN LANE first, 1 AT KERB, control clip 0 alerts). Found and fixed on the way:
   a stopped car hidden by a passing truck re-alerted (duplicate); same-spot dedupe within 3.5 s now merges it.
2. **Ground-contact line from the seg mask** instead of the axis-aligned box bottom (fallback to the box edge when no mask matches); confidence now scales
   with the in-lane fraction (the 71 % car shows 70 %, not 97 %). The angled car is still called IN LANE (see limits).
3. **Demo surfaces**: Lisbon shows "still from a slideshow" instead of "dwell 0.4 s"; stop time "already stopped at view start" instead of 00:00.0;
   PTZ hero leads with the 46 auto-lowered AT KERB and the merge count; PTZ repeats merged across presets by homography (87 -> 61).
4. **Queue collapse**: >= 3 stacked IN LANE stops that appeared together become one low-severity QUEUE card; `run.py --video X` now merges into
   `results.json` and recomputes `overall` from all videos.
5. Cosmos row collapsed on cards (not in the fix list): its 0/7 on IN LANE is reported, not hidden.

## Files

`run.py` CLI; `lanecall_core.py` detector + dedupe; `lanecall.py` zones, registration, seg ground line, call, cards; `inject.py` simulated stops in
the A40 clip; `inject_assets/` cut-out cars; `zones.json` polygons (ptz A/B, A40, Yonge, 33 Lisbon slides); `lisbon_manual_picks.json`;
`ground_truth.json`; `test_smoke.py`; `results.json`; `results_before_fixes.json` and `ground_truth_before_fixes.json` (the reviewed version, for
comparison); `out/`. Footage credits are in each `report.html`.
