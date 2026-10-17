# 08 ON THE LINE: scene answers for the RVRC operator on the phone

## What it is
A scene-state engine that follows a live alarm camera, so that when police ask "how many, which way did they go, what are they wearing, are they still there, when did they arrive?" the operator gets a card with the answer and the evidence (thumbnail + timestamp) in milliseconds.
At hang-up it drafts the incident log with times. Everything is advanced through the clip with a simulated clock and answered from data before "now"; no answer calls a model.

## Demo script (what you show judges in 90 s)
1. (0:00) One sentence: "A remote video response centre operator is on the phone with police and has 10 seconds to look at a camera. These are the five questions police always ask." Open `out/rochester_night/demo.mp4` or `out/person_bicycle_car_detection/demo.mp4` (live replay left, answer cards right).
2. (0:10) Press play. The alarm clip runs, police lines appear in the caption bar (typed here, Canary ASR on the day). Each answer card pops on the next frame with its evidence crop and a latency chip (a few ms).
3. (0:30) Point at the follow-up chain: "What colour is the car?" -> "Is the car still there?" -> the engine resolves "it/she/the car" to the same entity and says "NO, last seen 2.8 s ago, left via the LEFT edge" with the path drawn on the frame. Then say something messy, "Where'd they go?" or "The man in the red jacket, where is he?": it still answers (and an unrecognised question shows a "Didn't catch that" card with the scene summary instead of nothing).
4. (0:50) Hang-up: the video panel flips to the auto-drafted incident log (arrivals, departures, every Q&A with times). "The operator reviews it instead of typing it."
5. (1:05) Show `out/<clip>/hero.jpg`: green verdict badge in the header ("5/5 RIGHT vs my eyeball truth"), slowest answer a few ms. Mention the held-out clips and the SQL-only baseline: the 5 s segment table alone gets totals badly wrong (13 vs 4) and cannot say direction at all, which is why the per-frame tracker is the point.
6. (1:20) One line on the VAST stack: Canary hears the police, DataEngine/VastDB hold the scene facts, Cosmos3-Reason describes the best crop, the answer path stays model-free. Be upfront that colour naming from pixels is the weakest part (see limits).

## How it maps to the VAST stack on the day
| Prototype piece | On the day | Status |
|---|---|---|
| Typed police lines | Canary ASR transcript of the call (operator microphone / call audio), same text-in contract | not built, input contract identical |
| YOLO26s + ByteTrack rows replayed per frame | **Own** YOLO11s (COCO) + ByteTrack on the live stream. As described in the brief DataEngine yields 5 s segment counts, so I assume it does not expose boxes or tracks (confirm on the day); direction/exit/arrival to the second need a per-frame tracker | design risk, see limits |
| DataEngine 5 s counts | Cross-check and fallback for count/total questions (SQL on the segment table), also the segment record a judge can see in VastDB | baseline measured below |
| Cosmos3-Edge (local) describing the best crop | Cosmos3-Reason on the best crop of each confirmed track, results cached; the card shows pixel colours first and the Cosmos sentence second, and withholds the sentence (amber chip) when it contradicts the pixels or guesses an age | stand-in, 1 call per entity, <= 28 per clip |
| In-memory entity table | VastDB `entities` / `events` tables written when a track confirms/leaves; cards read them with SQL (< 1 s) and the incident log is one `ORDER BY t` query | not built |
| Cosmos-Embed1 | Not used. Would add "show me the same person again" re-identification across cameras | out of scope |
| duckdb | VastDB SQL (only used here for the segment-table baseline; no IN/ILIKE/GROUP BY/JOIN/window) | stand-in |

## Results on test footage
Three tuned-on default clips (5 typed questions each at given video times; truth = frames I looked at, in `scripts.json`) plus three held-out clips (17 questions). Two of the held-out clips use **paraphrased police phrasing** ("Where'd she go?", "Has the red one gone yet?", "Is anybody still about?"); I wrote their questions and truth from contact sheets before running the engine on them. Latency = parse + answer + evidence crops, measured per question.

| clip | right | latency ms median / max | tracker ids -> entities | no-future-data re-run | SQL-only count/total Qs | Cosmos kept / withheld | live count vs yolo26l |
|---|---|---|---|---|---|---|---|
| person_bicycle_car_detection | 5/5 | 1.48 / 5.29 | 8 -> 7 | identical | 1/2 | 5/6 kept / 1/6 withheld | person: mean abs diff 0.25, within 1 on 100% of 12 frames; vehicle: mean abs diff 0.08, within 1 on 100% of 12 frames |
| rochester_night | 5/5 | 0.72 / 3.41 | 6 -> 5 | identical | 0/2 | 3/4 kept / 1/4 withheld | vehicle: mean abs diff 0.33, within 1 on 92% of 12 frames |
| retail_cctv | 5/5 | 2.72 / 4.34 | 48 -> 27 | identical | 2/2 | 15/27 kept / 12/27 withheld | person: mean abs diff 3.17, within 1 on 17% of 12 frames |
| one_by_one_person_detection (held-out 1) | 5/5 | 1.24 / 2.3 | 3 -> 3 | identical | 1/2 | 3/3 kept / 0/3 withheld | person: mean abs diff 0.0, within 1 on 100% of 12 frames |
| car_detection (held-out 2, paraphrased) | 6/6 | 1.35 / 2.41 | 5 -> 4 | identical | 0/2 | 3/4 kept / 1/4 withheld | vehicle: mean abs diff 0.17, within 1 on 100% of 12 frames |
| people_detection (held-out 3, paraphrased) | 5/6 | 1.77 / 3.37 | 7 -> 7 | identical | 0/3 | 1/7 kept / 6/7 withheld | person: mean abs diff 0.25, within 1 on 100% of 12 frames |

**Headline: 15/15 on the three default clips (tuned-on), 16/17 on the three held-out clips (one_by_one 5/5, car_detection 6/6, people_detection 5/6).** Slowest answer on any clip 5.3 ms. Kill test passed.

How honest the held-out number is:
- `car_detection` was **5/6 on its first run** (6/6 now). C6 "how many vehicles all told" said 5 vs 4: a car half out of the top edge got a second tracker id and the engine's "left through an edge, so a new track is a new object" rule counted it twice; and the red car was named "pink" so "Has the red one gone yet?" only passed by falling back to the latest car. I fixed both (merge a fresh id within 0.6 s on an overlapping box even at an edge; vehicles named "pink" become "red", glare desaturates red paint). Those two fixes were made after seeing this clip, so it is no longer clean held-out. The default clips stayed 5/5 after them (the shop's entity count went from 30 to 27).
- `people_detection` is **5/6 and I did not touch the engine after seeing it.** P3 "What's the lad in the black top got on?" is wrong: the card says "black top, maroon trousers" for a man in black trousers (the polished red-brown floor leaks into the trouser colour). Also P4 "The man in the black top, where did he go?" scored right, but the card carries an amber "no black top: showing latest" chip and its evidence crop shows the woman, not the man: by hang-up the man's top had drifted to "purple" in the pixel votes so the attribute selector found nobody. The exit side (LEFT) was the same for both, so the verdict is right and the card is only half right.
- `one_by_one_person_detection` is the easy case (one person at a time).
- The paraphrased questions still share a phrasing family with the reviewer's failure cases (blokes, where'd, what's), so they measure the engine more than the parser. `test_smoke.py::test_parser_paraphrases` covers 14 more phrasings; those were written after the first parser fixes.

Per-question detail:

**person_bicycle_car_detection** (alarm 0.0 s, hang-up 52.0 s)
| id | t | police asked | answer card | checks (engine value) | verdict | ms |
|---|---|---|---|---|---|---|
| A1 | 5.0 | How many people can you see right now? | 1 person in view now | count_now=1 | RIGHT | 5.286 |
| A2 | 10.0 | Which way did he go? | The person left via the RIGHT edge, heading RIGHT AND UP | exit_side=right | RIGHT | 1.483 |
| A3 | 18.0 | What colour is the car? | A white car | colour=white | RIGHT | 2.398 |
| A4 | 22.0 | Is the car still there? | NO: the car was last seen 2.8 s ago, left via the LEFT edge | present=False; exit_side=left | RIGHT | 0.885 |
| A5 | 51.0 | How many cars have come in since the alarm, and when did the first one arrive? | 2 vehicles since the alarm. The car arrived at 00:14.3 (alarm +14 s) | total=2; arrival=14.33 | RIGHT | 0.795 |

**rochester_night** (alarm 8.5 s, hang-up 21.0 s)
| id | t | police asked | answer card | checks (engine value) | verdict | ms |
|---|---|---|---|---|---|---|
| R1 | 11.0 | What vehicle just went past, what colour was it? | A navy car | colour=navy | RIGHT | 3.413 |
| R2 | 12.0 | Which way did it go? | The car left via the LEFT edge, heading LEFT | exit_side=left; heading=left | RIGHT | 0.932 |
| R3 | 12.5 | How many vehicles have passed since the alarm, and what time did the first one arrive? | 1 vehicle since the alarm. The car arrived at 00:09.5 (alarm +1 s) | total=1; arrival=9.47 | RIGHT | 0.693 |
| R4 | 19.5 | Is the last car still there? | NO: the car was last seen 1.3 s ago, left via the LEFT edge | present=False | RIGHT | 0.701 |
| R5 | 20.5 | How many vehicles in total since the alarm? | 4 vehicles since the alarm | total=4 | RIGHT | 0.716 |

**retail_cctv** (alarm 62.0 s, hang-up 88.0 s)
| id | t | police asked | answer card | checks (engine value) | verdict | ms |
|---|---|---|---|---|---|---|
| S1 | 65.0 | How many people can you see right now? | 16 people in view now | count_now=16 | RIGHT | 2.719 |
| S2 | 70.0 | Who just left, and what were they wearing? | A person: light grey top, dark blue trousers | colour=silver; colour=navy | RIGHT | 4.338 |
| S3 | 70.5 | Which way did she go? | The person left via the RIGHT edge, heading RIGHT | exit_side=right | RIGHT | 1.373 |
| S4 | 80.0 | Is she still there? | NO: the person was last seen 11.2 s ago, left via the RIGHT edge | present=False | RIGHT | 0.885 |
| S5 | 87.0 | How many people are in the shop now? | 11 people in view now | count_now=11 | RIGHT | 3.315 |

**one_by_one_person_detection** (held-out 1) (alarm 0.0 s, hang-up 62.0 s)
| id | t | police asked | answer card | checks (engine value) | verdict | ms |
|---|---|---|---|---|---|---|
| H1 | 12.0 | How many people are there now? | 1 person in view now | count_now=1 | RIGHT | 0.721 |
| H2 | 13.0 | What is she wearing? | A person: grey top, dark grey trousers | colour=grey; colour=charcoal | RIGHT | 2.297 |
| H3 | 16.0 | Is she still there? | YES: the person is in view now, not moving since 00:13.5 | present=True | RIGHT | 1.239 |
| H4 | 22.0 | Which way did she go? | The person left via the RIGHT edge, heading RIGHT | exit_side=right | RIGHT | 1.985 |
| H5 | 56.0 | How many people have been here in total since the alarm? | 3 people since the alarm | total=3 | RIGHT | 1.098 |

**car_detection** (held-out 2, paraphrased) (alarm 0.0 s, hang-up 29.8 s)
| id | t | police asked | answer card | checks (engine value) | verdict | ms |
|---|---|---|---|---|---|---|
| C1 | 6.0 | What sort of motor is that, colour and all? | A white car | colour=white | RIGHT | 2.334 |
| C2 | 9.5 | Which way did the vehicle go when it left? | The car left via the TOP edge, heading UP the frame (away) | exit_side=top | RIGHT | 1.264 |
| C3 | 17.0 | How many cars are there now? | 2 vehicles in view now | count_now=2 | RIGHT | 0.977 |
| C4 | 22.0 | Has the red one gone yet? | NO: the car was last seen 3.1 s ago, left via the TOP edge | present=False; exit_side=top | RIGHT | 2.405 |
| C5 | 29.0 | When did the last one turn up? | The car arrived at 00:25.8 (alarm +26 s) | arrival=25.84 | RIGHT | 1.159 |
| C6 | 29.5 | How many vehicles have there been all told? | 4 vehicles since the alarm | total=4 | RIGHT | 1.432 |

**people_detection** (held-out 3, paraphrased) (alarm 0.0 s, hang-up 49.0 s)
| id | t | police asked | answer card | checks (engine value) | verdict | ms |
|---|---|---|---|---|---|---|
| P1 | 3.0 | Right, how many blokes are we looking at? | 1 person in view now | count_now=1 | RIGHT | 0.882 |
| P2 | 8.0 | Where'd she go? | The person left via the RIGHT edge, heading RIGHT | exit_side=right | RIGHT | 1.406 |
| P3 | 17.5 | What's the lad in the black top got on? | A person: black top, maroon trousers | colour=black; colour=maroon | WRONG | 3.37 |
| P4 | 24.0 | The man in the black top, where did he go? | The person left via the LEFT edge, heading LEFT | exit_side=left | RIGHT | 2.132 |
| P5 | 36.5 | Altogether how many people have turned up since the alarm? | 6 people since the alarm | total=6 | RIGHT | 1.0 |
| P6 | 48.0 | Is anybody still about? | 0 people in view now | count_now=0 | RIGHT | 2.506 |

Other measured numbers (all in `results.json`):
- Engine ingest cost per sampled frame (mean, excluding frames that triggered a Cosmos call): 0.7 to 1.6 ms on the uncrowded clips, 13.1 ms on the crowded shop; the 15 fps budget is 66 ms.
- Segment-table SQL baseline (the 5 s DataEngine table, only segments complete at question time): it answers only 4 of the 13 count/total questions right overall: totals are wrong everywhere (cars 4 vs 2, vehicles 13 vs 4, people 9 vs 3, 12 vs 6, 5 vs 4) and the counts at P1, P6 (people_detection) and C3 (car_detection) are wrong too (I did not investigate why). It cannot answer direction, exit side, colours, "still there?" or sub-second arrival time at all.
- Cosmos conflict check: of the descriptions generated, **12/27 on the shop and 6/7 on people_detection were withheld** because they disagreed with the pixel colours or guessed an age. The check cannot tell which side is wrong: on people_detection Cosmos said "black top / black t-shirt" (correct by my eyeball) while the pixels said purple, so withholding hid the better source there. Raw text is kept in `results.json` (`desc_raw`) and the report.
- Measured local Cosmos latency was a median of 9 to 50 s per call on this shared GPU (max 186 s); the availability delay used for cards is a **SIMULATED 3 s**, not measured. Hero images are drawn at hang-up, so an appearance card that said "Cosmos description still being generated" at question time shows the Cosmos line that was ready by hang-up, labelled "(ready after the call)".
- `run.py --all` (6 clips, with demo mp4s, five no-future-data re-runs per clip, yolo26l cross-check, warm caches): **126 s** wall in the final run (198 s in the previous run; cold caches take far longer, Cosmos calls and the GPU lock dominate).
- Independent count check (yolo26l at conf 0.40 on 12 frames per clip): agrees on every clip except the shop (mean abs diff 3.17, within 1 on only 2 of 12 frames; 48 tracker ids merge into 27 entities, counts good to about +/-2).

## Eyeball checks (frames I viewed, what I saw)
- Contact sheets of all six clips before designing or scoring anything. Car park: fixed overhead view, a man walks in and out (right edge), a white car passes (bottom-left to left), cyclists cross the top-right, a teal car later. Rochester: busy road at night, camera re-aims around 20-24 s (call ended at 21 s). Shop: top-down CCTV, till at the left, 12-16 people.
- Car park frames 1.5 to 49.6: man in navy jacket and jeans leaves via the right edge at ~7 s; white saloon enters bottom-left ~14.3 s and is half out of the left edge at 19.0; two cars in total. Rochester frames 0.3 to 20.5: blue hatchback right to left 9.3-10.9 s, recovery truck carrying a car 13.0-14.7 s, grey hatchback and black saloon right to left, road empty at 18.6 and 20.5. Shop frames 62-110.5 s: the woman in a grey sweater and blue jeans leaves through the bottom-right corner at ~68.9 s; full-resolution counts 16 at 65 s and 12 at 87 s.
- Held-out 1 (one_by_one): three people arrive one at a time (grey knit cardigan woman 6 s, man in black jacket 23 s, woman in black 50 s), stand at the table and leave to the right.
- **New this round, car_detection** (sheets at 0-30 s, 4-10.6 s, 14.5-21.1 s, 25.4-29.8 s): white hatchback enters from the bottom ~4.6 s and leaves through the top ~8.3 s; light blue-grey car enters from the top ~14.9 s, red saloon from the bottom ~15.2 s, red leaves via the top ~18.9 s, blue-grey via the bottom ~18.4 s; a white sports car enters from the top ~25.2 s and leaves via the bottom ~27.8 s. Four vehicles in total. The blue-grey car is named "white" by the pixels (near-grey, bright).
- **New this round, people_detection** (sheets at 0-44 s, 1-7.6 s, 14.5-25.5 s, 27-38 s, 41-48.7 s): woman in dark top and skirt enters bottom ~1.6 s, exits right ~6.6 s; man in a black T-shirt plus woman in a blue top and striped skirt enter bottom ~15 s and exit left through the door ~21 s; three people (man in grey check shirt, woman in dark dress, person in white top) enter ~28.8 s, the man exits left ~33.6 s, the other two right ~34.5 s; man with grey cap and light-blue shirt enters ~43.5 s, exits left ~46.6 s. Six people by 36.5 s, seven in total, room empty at 48 s.
- My own outputs this round: hero.jpg of all six clips (verdict badge now in the header strip, log limited to the questioned entities, the car_detection and people_detection heroes show all six cards), demo frames (`out/_eyeball/demo_car_detection*.jpg`). Things I saw and fixed: the tilted badge covering the live frame, the retail log full of "Person arrived" noise, "Cosmos description still being generated" on finished heroes, the retail S2 card ("silver top, navy trousers" plus a Cosmos line about a child in a blue top), the sixth question card missing from the six-question heroes, a chip truncating at 30 characters.

## Simulated or faked (explicit)
- **The live feed is a replay**: recorded clips with cached YOLO26s+ByteTrack rows. ByteTrack is an online tracker, so the rows equal what a live run would have produced, and the engine only receives one frame's rows at a time (`LiveFeed`). The no-future-data check proves it per question.
- **Alarm times are mine** (Rochester 8.5 s, shop 62 s, others 0 s), not from a real alarm system; none of these clips is a real alarm camera.
- **Police questions are typed and scripted** at times I picked; there is no audio and no Canary ASR. Questions were chosen after looking at the clips, to be answerable and checkable. Paraphrases were written by me, not by an independent person.
- **Ground truth is my eyeballing**, with ranges for crowd counts (shop 14-18 and 11-13).
- **Engine thresholds were adjusted after seeing runs on the three default clips** (blue up to hue 268 for CCTV violet shift, drop the darkest 30% of vehicle pixels, link tracker fragments that overlap, count a car inside a truck's box as cargo, class by majority vote) and, this round, two rules after car_detection (edge-merge within 0.6 s, pink to red for vehicles). Only one_by_one and people_detection are untouched held-out evidence.
- **Cosmos3-Reason is the local Cosmos3-Edge**, and the time until a description is usable is **SIMULATED at 3 s** (hosted speed).
- **DataEngine / VastDB are not used for the core path.** Segment counts come from `pk.segments()` (my tracker rows), SQL runs in duckdb, and the engine state is plain Python objects.
- **Colours and directions are in image space**: "RIGHT edge" means right of the frame; there is no compass or street mapping.
- The "kept / withheld" Cosmos figures use a keyword conflict check I wrote; it is not an accuracy measure of Cosmos.

## What fails / limits
- **Colour naming from pixels is the weakest link.** On people_detection black clothing is named "purple" and black trousers "maroon" (floor and CCTV colour cast), the pale blue-grey car is "white", the Rochester grey hatchback was "brown" at night in the first run (not re-checked). Colour answers are right on 5 of 6 scripted colour questions, but only because the accepted sets are generous ("navy" for blue, "charcoal"/"grey"/"silver" interchangeable).
- **Attribute selection depends on those colours**: "the man in the red jacket" finds nobody if the pixel vote says maroon or purple, and then the card says "no ...: showing latest" and answers for the latest entity (flagged amber, but easy to miss).
- **Crowds**: the shop has 48 confirmed tracker ids for 27 entities after merging; counts only good to about +/-2 (cards say so). People hidden behind shelves are reported as "left via the TOP edge"; do not trust exit side there. In crowds the incident log itemises only the questioned people.
- **Cosmos can be wrong, and the check cannot tell who is**: it caught the shop "child in a blue top" and the age guess, but it also withheld a correct Cosmos sentence on people_detection. The conflict check is keyword-based.
- **Cyclists**: bicycle and person labels flip on the same track (car park summary: 4 people, 2 vehicles, 1 bicycle).
- **Parser is still regex**, now broader (contractions, police slang, colour/garment selectors, "last one", a fallback card), but unknown phrasing still falls to the "Didn't catch that" card, pronouns resolve to the last answered entity only, and the fallback fires only on question-shaped lines. ASR errors will hit this directly. All 32 questions in this folder are my own wording, none independent.
- **Fixed camera assumed**: entry/exit sides mean nothing on a moving camera. `ptz_parking` untested; Rochester's camera re-aim after ~20 s is avoided by ending the call at 21 s.
- **Small sample**: six clips, 32 questions, one rater (me).
- **DataEngine (as described) hands us 5 s counts, not boxes**, so the winning move depends on running our own YOLO11s+ByteTrack on the stream. If the event cannot provide the stream at frame rate, this idea degrades to the SQL-only baseline (count questions only).
- **Judging risk**: the provided services are thin here (Cosmos only describes crops; DataEngine counts are a fallback; no VastDB tables built). The story has to be "real RVRC pain, services as memory and voice", not "built on the stack".

## Kill test result
Criterion: >= 4 of 5 answers right on 2 clips, answer latency < 1 s.
**PASSED**: 5/5 on person_bicycle_car_detection, rochester_night and retail_cctv, slowest answer 5.3 ms (all clips), no-future-data re-runs identical for every question on all six clips. Held-out: 16/17 (one_by_one 5/5, car_detection 6/6 after two fixes made on it, people_detection 5/6). Read it with the caveats above: default clips are tuned-on, all 32 questions are phrased by me, and colour answers are the shaky part.

## Rebuild plan on the day (hours)
About **10 h** for two people with the real services: tracker on the stream + the entity/scene-state code (reuse this design) 3 h; Canary ASR hookup and intent rules with an LLM fallback for messy transcripts 1.5 h; VastDB entity/event tables, SQL answer path and incident-log query 1.5 h; Cosmos3-Reason crop descriptions and the "disagrees with pixels" check 1 h; cards UI + live video panel 2 h; rehearsal on 3-4 clips, camera orientation labels (left edge = which street) and the demo script 1 h.

## Files
`run.py` (CLI: `--all` = 3 default + 3 held-out clips, `--video NAME`, `--ask "text@mm:ss"`, `--no-vlm`, `--no-video`, `--no-causality`, `--vlm-latency`), `engine.py` (scene state), `qa.py` (intents, selectors, fallback, Cosmos conflict check, cards), `incident_log.py`, `render.py`, `sim.py` (clock + scoring), `sql_baseline.py`, `scripts.json` (questions + truth), `test_smoke.py` (16 checks), `results.json`, `out/<clip>/{hero.jpg,report.html,demo.mp4,evidence/}`, `out/_eyeball/` (contact sheets and frame grids). Every module now imports on its own (no PYTHONPATH needed).
