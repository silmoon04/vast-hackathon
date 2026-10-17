# 04 NIGHT EXIT: "who went through the exit since T?"

Local-footage prototype. Video time is the clock:
`00:00` is the start of the clip, `--since 00:21` means 21 s into the footage. These times are clip-relative;
this prototype does not map them to a camera wall clock.

## What it is (2 lines)

A night carer says "resident missing, last seen 00:12": NIGHT EXIT lists every person who crossed a drawn exit line after
that time, in order, with IN/OUT direction, best crop, clothing colours, a one-line Cosmos description and a rank against the
description ("grey top").
Output per clip: `hero.jpg` (card stack with crop + zoomed evidence frame per person, timeline), `report.html`, evidence frames, `results.json`.
Every default hero now headlines a person who went OUT (the missing-resident story); when several people match the description about equally the hero says "best of N close matches, not a sure thing".

Run from this folder with the project environment and local inputs prepared: `python run.py --all`.
`python test_smoke.py` runs the checks; `--no-recompute` skips the pipeline recomputation.

## Implementation notes

- YOLO26s + ByteTrack provide person tracks. Direction requires tracked boxes, which cannot be recovered from segment counts alone.
- duckdb queries `pk.segments()`. `vastdb_shortlist()` uses `SELECT ... WHERE "end" > T AND n_person > 0 ORDER BY start` to shortlist 5 s segments; the smoke test checks that every detected crossing lies inside a shortlisted segment.
- Local Cosmos3-Edge produces a cached one-line description from the best crop, one call per person. Descriptions can be wrong, including confusing a white hoodie with white hair.
- Cosmos-Embed1 compares query text with description text through `embed_text(query)` and `embed_text(description)`.
- Canary ASR is not used; this experiment has no audio path.

## Results on test footage (all numbers measured by `python run.py --all`, `results.json`)

| Clip | Exit line (drawn by me) | Tracks -> people | Crossings found | Eyeballed | Matched / direction right | Extra / missed |
|---|---|---|---|---|---|---|
| one_by_one_person_detection (139 s) | E door x=665 (exit), S door y=300 (entry) | 6 -> 6 | 12 (6 IN, 6 OUT) | 12 | 12 / 12 | 0 / 0 |
| rochester_night (85 s) | pavement gate x=1000 | 3 -> 2 | 1 (OUT 00:05, interpolated across a 2.6 s car occlusion) | 1 | 1 / 1 | 0 / 0 |
| retail_cctv (111 s) | diagonal across the mat corner | 180 -> 113 | 10 (7 IN, 3 OUT) | 10 | 10 / 10 | 0 / 0 |

Retail crossings: IN 00:03, 00:04, 00:06 (interpolated), 00:10, 00:20, 00:46, 01:02; OUT 00:19, 00:45 (back in 1.4 s later), 01:07.
Retail ground truth was assembled after I had seen the detections (door sweep plus evidence frames), so 10/10 there is
recall against what I could see, not an independent label set. One_by_one and rochester truth was eyeballed from frames first.

Description ranking (rank of the right person; "combined" = 0.55 pixel colour + 0.20 Cosmos-description words + 0.25 Embed1 similarity; the Embed1 part is min-maxed over the candidates, so the hero also prints an **evidence score** = colour + description words only, not min-maxed, and flags every candidate within 10 points of the best as a close match):

| Clip | since | Query (written from eyeballing) | Combined | Colour only | Embed1 only | Candidates |
|---|---|---|---|---|---|---|
| one_by_one | 00:12 | grey top (3 valid people, all 3 fill ranks 1-3) | 1 | 1 | 1 | 6 |
| one_by_one | 00:21 | light grey shirt and blue jeans | 1 | 1 | 1 | 5 |
| one_by_one | 00:40 | blonde woman in a black cardigan | 1 | **2** | 1 | 5 |
| one_by_one (post-fix) | 00:40 | black t-shirt and black trousers | 1 | **3** | 1 | 5 |
| rochester_night | 00:00 | dark clothes with a backpack (only one person exists) | 1 | 1 | 1 | 1 |
| retail_cctv | 00:00 | white coat | 1 (**first run: 2**) | 2 | 2 | 9 |
| retail_cctv | 00:30 | blue coat and pink top | 1 | 1 | 1 | 3 |
| retail_cctv | 00:00 | girl in a white hoodie and blue jeans | 1 | 1 | 2 | 9 |
| retail_cctv (post-fix) | 00:15 | grey sweater and blue jeans | 1 | 1 | 1 | 5 |

Default hero per clip (after the fixes; `results.json` -> `demo_query`):

| Clip | Default query | Best match | Rank / evidence score | Next candidate | Close matches (within 10 pts) |
|---|---|---|---|---|---|
| one_by_one | since 00:21 "light grey shirt and blue jeans" | OUT 02:15 | 73 % / 64 % | 56 % / 55 % | 1 (clear) |
| rochester_night | since 00:00 "dark clothes" | OUT 00:05 (interpolated) | single candidate (the 88 % is meaningless), evidence 100 % | none | labelled "only candidate" |
| retail_cctv | since 00:30 "blue coat and pink top" (was: "white coat" -> WENT IN 00:20) | OUT 00:45, back in 1 s later | 65 % / 54 % | 29 % / 38 % | 1 (clear) |

Close-match counts for the other queries: "grey top" 3 (all three really are grey tops), "white coat" 2 (white jacket woman 98 % vs white-hoodie man 90 %; this is the query that failed on first run), the rest 1.

Speed: warm caches `--all` 38 to 48 s on an idle machine, 71 to 78 s in the final runs on a busier one (three runs). Partly cold runs (Cosmos/pose/embed caches empty for that clip): rochester 156 s,
retail 98 s (2 of 9 Cosmos calls were already cached); one_by_one was not timed cold after the CPU-pose change (an earlier version waited
11 min for the shared GPU lock). Cold time is dominated by the shared Cosmos server (30 to 120 s per call under load, 4 calls overlapped),
so a fully cold `--all` should take about 5 to 6 minutes on a loaded machine, not measured end to end.

## Eyeball checks (frames I viewed, and what I saw)

- Contact sheets of all three clips (12 frames each). one_by_one: six people, one at a time, enter at the bottom edge, stand at the table, leave through the right edge. rochester: static view for about 21 s, then a pan to a different street view. retail: door at the lower right, out of frame, mat leads to it.
- one_by_one entry/exit frames for all six people (two sheets of 16 frames): each person appears at the bottom edge and disappears at the right edge; clothing noted in `gt.json` (P1 grey knit cardigan + blonde, P2 black jacket, P3 black cardigan + blonde, P4 black t-shirt + glasses, P5 grey cardigan + dark hair, P6 light grey shirt + jeans).
- Evidence frames of the first 8 one_by_one crossings plus the rochester one: the red dot sits where the person's centre crosses the yellow line, IN = walking up from the bottom, OUT = leaving at the right edge.
- rochester frames 4.4 to 8.2 s: a dark pedestrian with a backpack walks left to right behind a red car; he passes x=1000 while hidden (about 6.0 s). The tool says 00:05 (interpolated). Off by about 1 s, flagged "interpolated" on the card.
- retail 1 Hz sweep of the door region, 0 to 110 s (four sheets) and a 16-frame close-up sheet at 52 to 63 s: counted who steps over the line and which way. The blue-coat woman at 00:45 bends over the line to pick up a basket and returns 1.4 s later; the woman in the navy coat at 00:59 to 01:02 stands on the line with her feet before walking in (crossing logged 01:02).
- retail best-crop sheet (9 crops): all sharp and the right person; pid labels match the eyeballed people (girl in maroon, bow girl, glasses mum, hoodie girl, navy-fleece man, white-coat woman, blue-coat woman, navy-coat woman, grey-sweater woman).
- All three `hero.jpg` and `hero_q*.jpg` viewed after the last render; layout bugs I found and fixed: score text overlapping the swatches, cards running into the timeline, empty left panel.
- After the changes I re-viewed the three default heroes plus `one_by_one/hero_q1` ("grey top") and `retail_cctv/hero_q1` ("white coat"). Seen: one_by_one default "WENT OUT at 02:15", 73 % vs 56 %, each card has a zoomed crossing frame with the yellow line, boxed person and red dot; "grey top" shows 100 / 98 / 92 %, a "1 OF 3 CLOSE" badge and amber "best of 3 close matches, not a sure thing" (rank 1 = blonde woman in grey, rank 2 = woman in grey cardigan whose torso reads grey/black, rank 3 = man in grey shirt); retail default "WENT OUT 00:45, BACK IN 1s later" with the blue-coat woman boxed on the mat, rank 2 a woman walking IN at 01:02, rank 3 OUT 01:07; retail "white coat" "1 OF 2 CLOSE" (white jacket woman IN 00:20, white-hoodie man IN 00:10); rochester shows "ONLY CANDIDATE" and "only candidate in this window" (88 % rank score there is meaningless and the card says so).
- Colour checks: on the rank-2 card of "grey top" the torso colour reads grey 65 % / black 32 % while Cosmos says "grey long-sleeved top", consistent; the rank-1 one_by_one crop at 02:15 reads legs "black 48 %" while Cosmos (and I) see blue jeans in dim light: a known colour-naming weakness, not fixed.
- Independent check of rochester with the larger detector (yolo26l on CPU, conf 0.2, imgsz 960, 20 frames from 1 s to 80 s): a person at 1 s, 3 s and 7.5 s (the pedestrian, a different box each time), none in the other 17 frames (at 5.5 s he is behind the car). So the one crossing is probably the only pedestrian visible; this does not cover the panned part by eye.

## Simulated or faked (explicit)

- **The missing-resident story is simulated.** The footage is public clips (a one-by-one person test, a night street, a shop). Nobody is missing; "since T" is a time in the clip.
- **Exit lines are drawn by me**, not detected doors, and are overridable with `--line`. In retail the real door is outside the frame; the line is a convention across the mat corner.
- Cosmos3-Edge stands in for Cosmos3-Reason, local Embed1 for the hosted one, YOLO26s for YOLO11s, duckdb for VastDB. DataEngine counts are not used for directions (they cannot give any).
- Colour and ranking weights (0.55 / 0.20 / 0.25) were set once by judgment, then the 7 queries were run. The close-match threshold (10 points) and the evidence score were added later, not tuned on results. No injected faults anywhere.
- `test_smoke.py` recompute mode runs the pipeline with `--no-vlm` (no Cosmos descriptions), so it checks tracking, crossings, directions and cards, not the Cosmos part of the ranking.
- Retail ground truth is circular in part (see Results). Rochester has exactly one pedestrian, so its ranking test proves nothing about ranking.
- The ranking queries are small (5 to 9 candidates). The "right person first" claim is about this footage, not about a station full of residents.

## What fails / limits

- **Door out of frame (retail):** the line is a convention, so a person stepping over it and back is logged OUT then IN (blue coat woman). The card says "back in 1 s later" and the hero verdict says so, but it is still two events.
- **Tracker fragmentation:** 180 ByteTrack ids for about 15 people in the shop. A simple constant-velocity stitcher cuts it to 113 identities. A wrong merge can hide or invent a crossing, and there is no appearance-based re-identification. A person who is first tracked already on the inner side of the line is missed (no "appeared at the door" inference).
- **Mislabelled first frames:** in one_by_one the entering person was boxed as "dog"/"cat" for a few frames; I absorb such short fragments into the person track that starts at the frame edge (without it the 01:32 entry was lost).
- **Colour is rough:** torso/legs from pose keypoints (CPU YOLO26s-pose, cached). Long cardigans put torso colour into the "legs" region (P1 legs read grey, real: dark leggings); jeans in dim light read black (P6 legs black 48 %); the Rochester pedestrian's crop is mostly a yellow hoarding. No light/dark qualifiers ("light grey" = grey).
- **Colour alone is not enough** (rows marked bold above): black t-shirt versus black hoodie versus black cardigan needs the Cosmos description. The query parser is rule-based: colours and garment families (coat/jacket vs hoodie vs cardigan). Hair, gender, bags only reach the ranking through the Cosmos description and Embed1.
- **The 'white coat' query failed on first run** (rank 2 of 9: white-hoodie girl 1.00 vs white-coat woman 0.98, colour cannot separate coat from hoodie). I added garment-word matching after seeing that, so its pass is not clean evidence. The 2 post-fix queries were written afterwards and pass.
- **Embed1 text-vs-text** gives low raw cosines (0.0 to 0.5); I min-max them over the few candidates, which exaggerates small gaps and is 0.5 for a single candidate. Mitigation: the hero prints the not-min-maxed evidence score next to the rank score and says "close matches" when candidates are within 10 points. The rank score is still relative to the candidates in the window, not a probability.
- **Rochester is a weak night test:** the crop of the one pedestrian is a roughly 90 px tall blurred sliver; Cosmos "black hair" there is likely a guess, and the exit line is switched off for 75 % of the clip. The "night" claim is barely tested and the ranking there proves nothing (one candidate).
- **Smoke test scope:** the default smoke now recomputes one clip (about 10 s warm, no VLM); retail and rochester are still only checked on their saved outputs.
- **Camera motion guard** is one global shift against frame 0 (phase correlation, upper 65 % of the frame, plus a weak-peak test). It switches the line off for the whole rest of rochester (21.5 s on) and does not re-register a returning view. Zooming PTZ clips are out of scope.
- **Night footage is thin:** one small pedestrian (about 90 px tall) in 85 s and no crowd, so I cannot say how it behaves with several people at night.
- Persons only (no vehicles). Cold runs depend on the shared Cosmos server speed.

## Kill test result

Criterion: crossing directions correct for eyeballed people; the right person ranked first for 3 description queries written from eyeballing.

- Directions: 23 of 23 eyeballed crossings found, 23 of 23 with the right direction, 0 extra, 0 missed (retail part circular, see above).
- Implementation changes: default hero per clip now headlines an OUT (retail default changed from "white coat" -> WENT IN 00:20 to "blue coat and pink top" -> WENT OUT 00:45); near-ties flagged ("grey top": 3 close matches; "white coat": 2); zoomed evidence frame with line and crossing dot on each card; smoke test recomputes one clip. The numbers below did not change.
- Ranking: the 3 one_by_one queries: 3 of 3 rank 1 (also 1 of 1 post-fix there). All 7 pre-registered queries: 6 of 7 on the first run, 7 of 7 after the garment fix (post-hoc). The 2 queries written after the fix: 2 of 2.
- Verdict: **passed**, with the caveats above. The idea works on real footage; it is not a proof for a busy night ward.

## Files

`run.py` CLI, `nightexit.py` core, `render.py` hero/evidence drawing, `gt.json` eyeballed truth and queries, `test_smoke.py` (81 checks, about 17 s including the recompute; `--no-recompute` skips it), `results.json`,
`out/<clip>/{hero.jpg, hero_q*.jpg, report.html, crops/, events/}`, `out/eyeball/` (the contact sheets I viewed for the ground truth), `cache/` (pose results per frame).
