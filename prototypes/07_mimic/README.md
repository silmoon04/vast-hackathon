# MIMIC: find the object that moved like this

Run: `.venv/Scripts/python.exe prototypes/07_mimic/run.py --all`
(about 6 min on this loaded machine once the camera-motion caches exist; the first ever run adds about 10 min of camera analysis, `tools/cam.py` does it in parallel).
Other entry points: `--video NAME`, `--path "x,y;x,y;..." [--rot --pos-w 0.6]`, `test_smoke.py` (about 45 s incl. two real-browser checks), `out/draw.html` (interactive page), `tools/page_drive.py` (drives the page in headless Chrome/Edge over file://, screenshots in `out/_page/`).

## What it is

An investigator who remembers HOW something moved ("it came in, turned round and went back") draws that path, or picks a sketch; MIMIC returns the
tracked objects in the footage that moved like that: top 8 with thumbnails, camera-compensated trajectory overlays and a clip of the best one.
Every track becomes a 32-point motion descriptor in a table; a query is cosine kNN on the descriptors, then a DTW re-rank (shape, turning profile, optional position / size / pauses).
Direction: `left_to_right` and `diagonal` keep the drawn direction; `u_turn`, `stop_and_go` and `loiter` match in any orientation (12 rotations of the sketch), so the best hit often walks the other way round. This is no longer silent: every result records the rotation that matched (`rot_deg` in results.json), the hero shows an "as matched: turned 180 deg" copy of the sketch, and the page puts a "sketch turned 180 deg" chip on the card.
Weak results: anything above cost 0.62 is not a match. Heroes and the page hide such candidates (page: behind a "show them" toggle) or grey them out; when nothing is under 0.62 the hero / page shows a NO MATCH banner with the closest candidate greyed out.

## Demo script (90 s)

1. (0-15 s) Open `out/draw.html` (double-click, it is one file, 5 MB). "You remember how it moved, not what it looked like." The canvas holds the U-turn sketch and the page has already answered: 753 tracklets, 21 clips, 46 min of footage. The top hits are factory workers who walk in, turn round and walk out, each with their trajectory on the frame (green start, red end) and a grey chip "sketch turned 180 deg" (they walk the other way round than the sketch: orientation-free matching, and the page says so).
2. (15-40 s) Clear, then draw your own U-turn with the mouse (any size, any direction). Results re-rank in about 0.1-0.2 s in the page. Click "clip" on the best card for the 17 s video of that worker.
3. (40-60 s) The honest part, "it says NO when the answer is no": pick `yonge_dundas` in "Backdrop / search in" and draw the same U-turn. Red banner "NO MATCH: nothing in this clip moved like that (best cost 1.25, threshold 0.62)", the closest candidates greyed out. Backup picture: `out/hero_negative.jpg`.
4. (60-80 s) Pick "stop and go" (two blue pause rings on the sketch) or tick "my pauses count" and draw with a pause in the middle: matches show a ring and "x s" where the person halts, plus a speed-vs-sketch sparkline (blue rings with seconds = halts) (yonge pedestrian who walks, stands about 3 s and walks on; a woman in face_demographics who pauses).
5. (80-90 s) `out/hero.jpg`: the numbers (hand-checked P@3 1.00 / 1.00 / 0.67 / 1.00 for u-turn / left-to-right / stop-and-go / loiter) and `out/lisbon_parking/hero.jpg`: NO TRAJECTORIES, because that clip is a photo slideshow.

Pitch line that is true on this footage: "a U-turn anywhere in 46 minutes of video, found by drawing it, and a clear NO when there is none". Lines that are NOT true here: "the one car that U-turned in 13 minutes of Lisbon" (Lisbon is a slideshow, no car U-turn exists in any clip; the only hairpins are a few factory workers: see What fails).

## How it maps to the VAST stack on the day

| Here (local stand-in) | On the day | Note |
|---|---|---|
| YOLO26s TensorRT + ByteTrack, cached per frame boxes | hosted YOLO11s detections per frame + our own ByteTrack | **Risk**: DataEngine's per-5 s output is COUNTS, which cannot give trajectories. We need per-frame boxes from the YOLO service (or run ByteTrack on frames we send), so ingest must keep box coordinates. |
| duckdb `tracklets` table via `pk.vastdb` (SQL: SELECT / WHERE / OR chains / ORDER BY, no IN, GROUP BY, JOIN, window) | VastDB table, same columns (pos 64 floats, speed 32 floats, scalars) | Metadata filter in SQL (`video`, `cls`, `dur`, ...). kNN + DTW run in the app (numpy or the page's JS) over the pulled descriptors; no VastDB vector index is assumed. |
| local camera-motion estimation (LK / ORB, similarity transform) | same code on the frames | Needed whenever the camera moves or cuts; it is what makes ptz / handheld footage usable at all. |
| not used: Cosmos3-Reason | caption the best matches ("a white van reverses into a bay") and show it under the clip | optional garnish, never needed for correctness |
| not used: Cosmos-Embed1 | "motion AND look" queries: drawn path + text embedding ("red car") as a second filter | would fix the main limit (no appearance), not built here |
| not used: Canary ASR | none | |

## Results on test footage

All numbers measured on this machine (`results.json`, `ground_truth.json`). 21 default clips, 7,629 tracks from the cached YOLO26s+ByteTrack, of which 753 survive as tracklets (>= 1.5 s inside one
camera-stable window) and 309 are searchable (moved at least half a body height and 0.03 frame heights; the rest are parked / jittering boxes).

Hand-checked precision (labels by me, from the images, see Eyeball checks; sample is small):

| preset | P@3 | right in top 8 | best cost | where the best is |
|---|---|---|---|---|
| u_turn | 1.00 | 5 / 8 | 0.40 | worker_zone_detection (workers walk in, turn, walk back) |
| left_to_right | 1.00 | 8 / 8 | 0.02 | yonge_dundas, rochester_night |
| stop_and_go | 0.67 | 2 / 8 | 0.51 | yonge_dundas, face_demographics |
| loiter | 1.00 | 6 / 8 | 0.29 | face_demographics, elephants_dream (cartoon), retail_cctv |
| diagonal (extra) | 1.00 | 8 / 8 | 0.35 | ptz_parking, person_bicycle_car, retail_cctv |

Mean P@3 over the four kill-test presets: 0.92. Verdict thresholds (cost <= 0.45 MATCH, <= 0.62 MAYBE, else NO MATCH) were set by eye on those same 40 labelled results (29 right: median cost 0.43; 11 wrong: median 0.62, minimum 0.50), so they are optimistic and costs are not comparable between presets.

Negative control (the part a judge will test): a U-turn sketch searched clip by clip gives MATCH in 1 clip (worker_zone_detection, 0.40), MAYBE in 2 (face_demographics head sway 0.50, elephants_dream cartoon 0.60), NO MATCH in 12, and has no data in the 6 clips without searchable tracklets (`results.json` videos.*.presets.u_turn). The demo negative is `yonge_dundas` (a crowd, 161 tracklets): best cost 1.11 -> NO MATCH, `out/hero_negative.jpg`. Tracker-level negative only: no camera-stable tracklet of at least 1.5 s has a hairpin; it does not prove nobody in the crowd turned round.

Matched orientation (global top 3): u_turn 180 / 180 / 180 deg (the workers walk the other way round than the sketch, hence the "as matched" copy in the hero), stop_and_go 180 / 180 / 210 deg, loiter 0 / 210 / 0 deg, left_to_right and diagonal always 0 (direction is kept).

Speed: python query over all 753 tracklets 70-130 ms for straight sketches, 450-630 ms for rotation-free-any-orientation sketches (12 rotations); the same search in the page: the status line shows about 0.1-0.2 s for the U-turn preset (any orientation) in headless Chrome; parity with python checked in node (same top-3, same matched rotation, cost difference < 0.001).

Per clip (tracks = YOLO26s tracks with >= 3 frames; camera windows = camera-stable stretches after cuts / motion splitting):

| clip | length | tracks | camera windows | tracklets | searchable | best preset (cost, verdict) |
|---|---|---|---|---|---|---|
| retail_cctv | 111 s | 243 | 1 | 126 | 43 | diagonal 0.43 MATCH |
| lisbon_parking | 779 s | 6006 | 2177 | 12 | 0 | none: photo slideshow |
| ptz_parking | 316 s | 315 | 121 | 222 | 26 | left_to_right 0.29 MATCH |
| rochester_night | 85 s | 64 | 34 | 21 | 21 | left_to_right 0.03 MATCH |
| yonge_dundas | 57 s | 454 | 1 | 161 | 112 | left_to_right 0.02 MATCH |
| motorway_a40 | 24 s | 72 | 1 | 40 | 39 | stop_and_go 0.66 NO MATCH |
| dashcam_urban | 120 s | 153 | 307 | 4 | 2 | diagonal 0.61 MAYBE (moving camera, 2 searchable tracklets: not trustworthy) |
| elephants_dream | 654 s | 195 | 1635 | 94 | 19 | left_to_right 0.10 MATCH (cartoon) |
| skate_fall | 10 s | 6 | 10 | 1 | 0 | none |
| floor_lie | 3 s | 2 | 2 | 1 | 0 | none |
| assist_elderly | 10 s | 6 | 17 | 3 | 0 | none |
| help_sit | 14 s | 10 | 5 | 9 | 1 | stop_and_go 0.62 MAYBE |
| sit_down | 10 s | 4 | 1 | 3 | 0 | none |
| car_detection | 30 s | 5 | 1 | 4 | 4 | stop_and_go 0.73 NO MATCH |
| classroom | 33 s | 31 | 1 | 6 | 0 | none (seated) |
| face_demographics_walking_and_pause | 91 s | 7 | 2 | 7 | 7 | loiter 0.29 MATCH |
| one_by_one_person_detection | 139 s | 15 | 1 | 6 | 6 | stop_and_go 0.75 NO MATCH |
| people_detection | 50 s | 7 | 1 | 7 | 7 | stop_and_go 0.78 NO MATCH |
| person_bicycle_car_detection | 54 s | 9 | 1 | 8 | 8 | diagonal 0.36 MATCH |
| store_aisle_detection | 65 s | 19 | 1 | 12 | 8 | stop_and_go 0.70 NO MATCH |
| worker_zone_detection | 76 s | 6 | 1 | 6 | 6 | u_turn 0.40 MATCH |

The per-clip verdicts are NOT hand-checked beyond the global top 8 of each preset (the labelled keys are the only verified results).

Soft spots in the labels, stated plainly: all 40 labels are one rater (me) and the thresholds were tuned on the same 40. The loiter P@3 of 1.00 uses the preset's own definition ("hangs around inside a small area", a person who stands in place counts); under a stricter rubric "walks up, then stays" it is 0.67 (#1 and #3 are interviewees who walk up and stay 10-16 s, #2 is a cartoon character bobbing in place, see `out/_sheets/strips_loiter_0.jpg`). The same head sway of a standing person is labelled wrong for u_turn (no turn) and right for loiter (no travel): consistent with each preset's definition, but it is a judgement call.

SIMULATED robustness (labelled as such everywhere; queries are synthetic, footage is real):
- Self-retrieval: 60 real tracklets redrawn with a wobbly hand (smooth drift 3% of extent, jitter, 0.8-1.25x rescale); shape only, all clips: recall@1 0.70, recall@8 0.97 (curved paths only 0.87, n = 15); shape + "where it happened": recall@1 0.85, recall@8 0.98; same inside one clip: recall@8 0.98 (curved 1.00).
- 18 U-turn drawings with different leg gap / leg length / rotation: the top-1 is a hand-verified hairpin (worker) in 16 of 18, on average 2.67 of the top 3.
- 10 shaky redrawings of each preset: share of their top 3 that are in the clean preset's top 10: u_turn 1.00, left_to_right 0.50 (many near-identical straight tracks), stop_and_go 0.90, loiter 1.00, diagonal 1.00.

## Eyeball checks

Frames I actually viewed, and what I saw:
- Contact sheets (8 frames each) of lisbon_parking, ptz_parking, retail_cctv, rochester_night, yonge_dundas, motorway_a40, dashcam_urban, car_detection, store_aisle_detection, person_bicycle_car_detection, worker_zone_detection (all in `out/_sheets/`). The other default clips' sheets were generated but not looked at.
- **lisbon_parking is not a fixed camera**: 16 consecutive frames (`out/_sheets/lis_seq.jpg`) show a different still photo every ~0.4 s (a street sign, a yellow car, a van, black bars); the clip is a slideshow of photographs with 2,177 cuts in 779 s. There is no motion to find: the camera-motion step reports it and MIMIC says NO TRAJECTORIES. 12 jitter tracklets from the stills are rejected by the movement gate.
- one_by_one_person_detection at 17.9 / 18.4 / 50.8 / 51.0 / 51.2 / 51.8 s (`out/_sheets/obo.jpg`): locked-off camera, but a person fills the frame at 51 s and drags the camera-motion estimate. That found the foreground-bias bug; features are now taken from the background only (detected movers masked) and a locked camera needs strong evidence to be called moving.
- Top-8 sheets of every preset over all clips (`out/global/top8_*.jpg`): u_turn #1-4 and #6 are workers walking in, turning and walking out (real hairpins, boxes on the right person, arrow ends where they leave); #5, #7 are a standing person's head sway and #8 a cartoon sway (labelled wrong). left_to_right: eight straight left-to-right tracks (yonge pedestrians / cyclists, rochester cars at night). diagonal: eight oblique tracks, headings 14-30 degrees (the sketch itself is 25 degrees in pixel geometry).
- 6-frame filmstrips (frames, box, trail so far) of the top 8 of stop_and_go and loiter (`out/_sheets/strips_*.jpg`). stop_and_go: #1 a yonge pedestrian walks, stands about 3 s, walks on; #2 a woman in face_demographics walks in, pauses, steps aside; #3 is a cartoon; #4 one halt only; #5-#8 move continuously (wrong). loiter: #1, #3, #7 people who walk up and then stay in place 7-15 s; #2 a cartoon character standing; #5 and #8 retail customers staying in one small area for 26-44 s; #4 and #6 are cars driving across (wrong).
- Heroes viewed: global, yonge_dundas, worker_zone_detection, retail_cctv, ptz_parking, lisbon_parking, motorway_a40 (NO MATCH stamp, correct), skate_fall (no trajectories). First frames, a middle and a late frame of the global u_turn clip: box on the worker, trail and arrow follow him, the clip holds the whole track (up to 18 s): the worker walks in, stands at the turning point for several seconds, then walks out.
- **The page, in a real browser (new)**: `tools/page_drive.py` opens `out/draw.html` over file:// in headless Chrome (no server, no network), injects synthetic pointer events into the canvas (a U-turn, a straight line, a scribble, a walk-pause-walk with a 1.2 s hold) and screenshots the result (`out/_page/*.png`, 8 scenarios, all with 0 JS errors). Viewed: the initial page (U-turn preset: 8 cards, worker hairpins with trails, green/amber chips); a drawn U-turn (7 cards + "1 weaker candidate hidden"); the yonge_dundas negative (red NO MATCH banner, 3 greyed-out candidates, the sketch over the clip backdrop); the stop_and_go preset (two pause rings on the sketch, ring + seconds and a speed sparkline on the cards); the hand-drawn pause (a 1.2 s ring appears where I held the pointer; 4 of the 8 results then show a halt ring, the other 4 are straight cars / walkers that never stop, i.e. the pause weighting is soft). The canvas drawing, card layout and per-card canvases all rendered. Not done: a human mouse, touch, Firefox/Safari, and clicking the "clip" links (mp4 in a new tab).
- Heroes viewed after this round: global (hand-check RIGHT / WRONG tags on each runner-up: #5 and #7 are the head-sway false hits and say so), hero_negative (yonge, NO MATCH, closest candidate greyed), worker_zone_detection (MATCH, "turned 180 deg" copy of the sketch), yonge_dundas, retail_cctv (3 weaker candidates hidden), ptz_parking (1 match shown, 6 hidden), motorway_a40 (NO MATCH, greyed closest), plus ad-hoc preset heroes `out/global/hero_stop_and_go.jpg` (two pause rings on sketch and on the match, speed sparkline with dips: the yonge pedestrian) and `hero_loiter.jpg` (dwell halo on the sketch, 10 s ring on the standing interviewee).

## Simulated or faked

- All queries are synthetic: the presets are polylines I wrote; the "hand-drawn" tests are noisy copies of real tracks or of the presets. No human drew anything with a mouse.
- Detections and tracks are YOLO26s + ByteTrack (cached by protokit), standing in for the hosted YOLO11s. VastDB is duckdb. No Cosmos model and no ASR is used.
- Labels, and the verdict thresholds derived from them, come from one rater (me) on 40 results of one run, so P@3 and the thresholds are optimistic.
- The page's parity with python is tested in node (same top-3, same matched rotation) and its rendering and interaction in headless Chrome with SYNTHETIC pointer events. Real hand-drawn input (jitter, speed, touch) is still only simulated; the pause test uses a scripted 1.2 s hold.
- The "turned N deg" and pause / dwell cues are computed from the real tracks (rotation of the best DTW fit, halts = slower than 0.25 box heights per second for 0.6 s or more); the sketch-side pauses and the loiter halo are annotations of the preset, not measurements.

## What fails / limits

- **No car U-turn exists in this footage and the demo clip is unusable.** Lisbon is a photo slideshow (no motion). In the real video clips the only real hairpins are the factory workers (6 tracklets of a few workers: 5 of the 8 hits are the same few walks; the top one also stands about 6 s at the turn). ptz_parking cars are almost all parked (26 of 222 tracklets move enough); a vertical narrow hairpin drawn freehand found a creeping car (cost 0.47, MAYBE) which is not a U-turn.
- The motion data is only as good as the tracker: ID switches and occlusions cut tracklets (>= 1.5 s inside one camera window is required), and a crowd (yonge, retail) fragments tracks.
- stop_and_go is weak (P@3 0.67, 2 of 8 right): the halt signature comes from smoothed box-centre speed, which is noisy.
- Camera compensation is a 2D similarity transform from background features: it breaks on strong parallax, zoom-heavy PTZ, featureless asphalt and frames dominated by a close foreground object; those cases split the clip (ptz_parking 121 windows, dashcam 307, elephants_dream 1,635) and shrink the searchable set. Slow pans below 0.4 px/frame (at 256 px width) are treated as locked.
- Costs are not comparable between presets and the thresholds are tuned on the test sample.
- Appearance is not in the query: "a white car that U-turned" needs the Cosmos-Embed1 text filter that was not built.
- Camera chains were computed in two passes (static-camera and strict-mode rules were added late); a clean recompute from scratch may move a few window boundaries.
- Mirrors are not searched: an anticlockwise U-turn sketch will not match a clockwise hairpin unless a rotation makes them coincide (a hairpin is nearly symmetric, so it mostly does; spirals and L-turns are not). Rotation steps are 30 degrees.
- The hero's runner-up row can still show wrong-but-plausible hits (the global u_turn #5 and #7 are head sway at cost 0.49 / 0.59, labelled WRONG on the hero). The cost threshold cannot separate them from true hairpins on this sample.
- "my pauses count" is a soft preference (speed weight 3 against shape 1 + turn 0.8): with a hand-drawn walk-pause-walk half of the top 8 still never stop. Only the stop_and_go preset (halts + speed term) was hand-checked (P@3 0.67).
- Heroes hide results above cost 0.62, so a per-clip hero can show a single card (ptz_parking: one 2 s truck track). That is deliberate, not a bug.
- Sketch-in-any-orientation search costs 12x the DTW work (about 0.6 s in the page); fine for 753 tracklets, would need a faster index at 100k.

## Kill test result

Criterion: drawn-path queries return a plausible match for at least 3 of 4 presets. **PASS, 4 of 4**: u_turn, left_to_right, stop_and_go and loiter each have a hand-verified correct result in the top 3 (top-1 correct in all four; P@3 1.00 / 1.00 / 0.67 / 1.00). Caveats: the presets are synthetic sketches, one rater, and the U-turn evidence is one factory scene (6 tracklets, a few workers). Added this round: the negative control (U-turn on yonge_dundas -> NO MATCH, 12 of 15 clips with data say NO MATCH to a U-turn) and the page rendering check in a real browser; neither changes the verdict.

## Rebuild plan on the day (hours)

About 5 h with the real services: 1.0 h per-frame boxes from the YOLO endpoint + ByteTrack into tracklets; 1.0 h camera-motion windows and compensation (or skip it if the camera is fixed); 0.5 h descriptor table in VastDB; 1.0 h matching service (numpy kNN + DTW, ported from `mimic_core.match`); 1.0 h drawing page (port `mimic_search.js` + canvas); 0.5 h thumbnails, clip, polish. Plus 1 h of slack for the "only counts, no boxes" risk above.

## Files

`run.py` CLI, `mimic_core.py` (camera windows, tracklets, descriptors, DTW, match), `mimic_render.py` (cards, heroes, clips), `mimic_eval.py` (labels, simulated tests, index), `build_page.py` + `mimic_search.js` (page), `ground_truth.json` (hand labels with notes), `results.json`, `test_smoke.py`, `parity_check.js` / `page_check.js` (node checks), `tools/` (camera precompute, filmstrips, facts, `page_drive.py` real-browser driver, `rebuild_page.py`, `global_hero.py`, `preset_hero.py`), `out/<clip>/report.html|hero.jpg`, `out/hero.jpg`, `out/index.html`, `out/draw.html`, `out/global/`.
