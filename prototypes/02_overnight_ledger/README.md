# Overnight Ledger

## What it is
Pick one parked car on a fixed camera and get a second-by-second ledger: PRESENT (green), ABSENT (red), or CAN'T SEE (grey, with the cause: lens covered, frozen feed, out of focus, camera looking elsewhere, spot blocked). Every grey stretch gets a before/after crop comparison ("same car" or "CHANGED: check"), and every person whose feet enter a zone around the car is logged with a thumbnail and a TOUCH? score (the car crop just before vs just after the visit). The verdict is "same car", not "undamaged": damage is not checked. It never guesses through a blind stretch: the answer for those seconds is UNKNOWN.

Run: `../../.venv/Scripts/python.exe run.py --all` (about 5 min cold, about 2 min with caches), `--video ptz_parking --tid 3` for one car, then `../../.venv/Scripts/python.exe test_smoke.py` (about 1 s, 13 checks). Open `out/<video>/hero.jpg` first, then `out/<video>/report.html`.

## Demo script (90 s)
1. (0-15 s) `out/ptz_parking/hero.jpg`: "Was this dark sedan there all night?" Green strip, one grey block, green again. Stamp: THERE, SAME CAR (with the line "damage is NOT checked"). Point at the green badge in the header: the same detector found 36/36 seconds of SIMULATED camera faults with 0 false alarms.
2. (15-35 s) Point at the grey block: "The PTZ camera went off to a lamp post for 81 s. We do not guess: UNKNOWN 81 s (headline and strip agree). The car is there before and after, SSIM 0.69 vs the same car at a different angle and light, verdict SAME CAR." Show the before/after pair on the hero.
3. (35-50 s) The contact thumbnail: "One person walked up to the front-left corner at 02:18, 14 px from the car's footprint. Time-stamped, with a picture, and TOUCH? says the car looks the same before and after the visit (SSIM 0.87)." Say it plainly: feet near is not a touch.
4. (50-65 s) Second strip on the hero: "A black SUV is there until 01:08 and the spot is empty afterwards": red. This is a real jump cut in the clip; for a car park it is "the car left overnight".
5. (65-85 s) `out/sim_a/hero.jpg`: "Now we sabotage the camera (SIMULATED): black lens cover 10 s, frozen feed 8 s." Blue brackets (injected truth) sit exactly on the grey blocks: 18/18 seconds found, 0 false alarms, causes named.
6. (85-90 s) Honesty beat: `out/lisbon_parking/hero.jpg` says NO LEDGER POSSIBLE (it is a photo slideshow). Then `out/sim_b/hero.jpg` as the honesty slide: a scratch SIMULATED onto the car is visible in the "after" crop and the hero says in an amber box "NOT detected (known miss)". Fine damage is the Cosmos3-Reason job on the day, not something this prototype does.

## How it maps to the VAST stack on the day
| Step here (local stand-in) | On the day |
|---|---|
| YOLO26s + ByteTrack boxes, cached (`pk.tracks`) | DataEngine YOLO11s step. **Risk:** the DataEngine table described to us holds counts per 5 s segment, not boxes. Per-car matching needs boxes: either the hosted YOLO endpoint returns them, or we run our own YOLO on the clip. Counts still help: a drop in `peak_car` between two segments is a cheap "a car left" pre-filter (SQL on the segment table). |
| OpenCV: 5 fps luma / sharpness / frame-diff, ORB homography to a reference frame, 64x36 layout correlation (blind causes) | Cosmos-Embed1 video embeddings of 5 s segments as an independent "same view as the reference?" test (cosine distance to the reference segment). Not built or tested here. |
| SSIM / block-SSIM crop compare (`ledger.compare`) | Same code for gross change, plus **Cosmos3-Reason on the before/after crops** ("any new dent, scratch or missing part?") for the fine damage the SSIM test cannot see. Not built; no VLM is used anywhere in this prototype, so nothing depends on it. |
| `pk.vastdb` (duckdb) ledger table, VastDB-safe SQL only (`WHERE`, `ORDER BY`, `LIMIT`, plain aggregates) | VastDB table `ledger(video, car, sec, state, cause)`; queries are in `results.json -> sql_demo` (e.g. `SELECT MIN(sec) ... WHERE state = 'G' AND cause = 'covered'` returned 10 on sim_a, the true cover start). |
| Canary ASR | Not used (no audio in the test set). Possible add-on: car alarm / glass break as "look here" markers. |

## Results on test footage
All numbers are from `results.json` (run of this folder, nothing hand-edited).

| clip | what it really is | ledger | result |
|---|---|---|---|
| ptz_parking (316 s, real) | real PTZ lot camera: steady view 0-02:37, lamp-post close-up tour, back panned ~570 px at ~04:00; plus one hard cut at 01:08 | car A (dark sedan, tid 3, picked automatically): 235 s present, 0 absent, 81 s can't see (two 1 s alignment dropouts at 250 s and 256 s sit between identical present seconds and are counted as present, `summary.filled_gap_s` = 2) | THERE, SAME CAR; blind stretch 02:37-03:58 (81 s, "camera looking elsewhere"; the headline and the strip label both say 81 s), before/after SSIM 0.69 (worst block 0.29), verdict same car; 1 person came close (02:18, 14 px), TOUCH? SSIM 0.87 before vs after the visit (02:16 vs 02:20): same car |
| ptz_parking, second car | black SUV in the middle (tid 7), auto-chosen as the car with the longest absence | 68 s present, 166 s absent, 82 s can't see (81 s view + 1 s occluded) | present until 01:08, empty afterwards (matches eyeball, see below) |
| ptz_parking, all candidates | 40 candidate (car, shot) pairs, about 20 distinct cars | 27 before/after verdicts "same car", 4 "CHANGED: check" (2 distinct cars: tid 15 really left (seen in `evidence/tour_pairs.jpg`), tid 13's box overlaps a neighbour that moved) | |
| lisbon_parking (779 s) | **not a fixed camera**: title cards + photo slideshow, 2160 hard cuts, longest shot 6.4 s | none | NO LEDGER POSSIBLE (correct refusal) |
| car_detection (30 s) | empty overhead lot, cars drive through (max 0.2 s still) | none | NO LEDGER POSSIBLE (correct refusal) |
| sim_a (SIMULATED, from real ptz 69-154 s) | black cover 10 s at 10-20, freeze 8 s at 55-63 | 67 s present, 18 s can't see | blind seconds: precision 1.00, recall 1.00 (18/18), onset/offset error 0 s, cause correct 100% |
| sim_b (SIMULATED) | cloth over lens 7 s at 20-27, defocus 6 s at 40-46, freeze 5 s at 62-67, scratch painted on the car from 24 s | 67 s present, 18 s can't see | blind seconds: precision 1.00, recall 1.00 (18/18), onset/offset error 0 s, cause correct 100%. **The painted scratch was NOT detected: verdict stayed "same car"; the hero now says so in an amber "NOT detected (known miss)" box and its stamp reads "same car = appearance match; damage is NOT checked"** |
| false alarms on 10 clean real clips (764 s) | retail, street, dashcam, night street, classroom, factory, ... | | 15 s flagged, all in `retail_cctv` 98-111 s where frame 99 s vs 100 s differ by 0.05 (probable stalled stream, not verified as a hard freeze); 0 s flagged on the other 9 clips, including `rochester_night` |
| clean ptz window 0-156 s | the same real footage before the tour | | 0 false blind seconds (asserted in `test_smoke.py`) |

Runtime: `--all` cold 314 s (features 50 s, alignment about 40 s for ptz, sim files about 65 s each to write), warm about 2 min. Memory: JPEG-per-second alignment cache, `out/_cache` about 90 MB.

Crop comparison calibration (`calibrate.py -> calibration.json`, 11 cars of ptz_parking, one scene, thresholds tuned on the same data, no held-out set):
| pair type | n | flagged CHANGED |
|---|---|---|
| same car, same view, 45-75 s apart | 33 | 0.0% |
| same car, before vs after the tour (new angle/light) | 66 | 3.0% |
| a different car in the same box | 110 | 100% (SSIM p95 0.468 vs 0.642 for same-car p5) |
| SIMULATED scratch+dent painted onto same-view "after" crops | 33 | 0.0% |
| SIMULATED scratch+dent painted onto across-tour "after" crops | 66 | 10.6% |

So the crop test reliably answers "is it the same car?" and does not answer "is it scratched?".

Kill-test numbers shown on the main ptz hero as a header badge (computed from the two sim runs, `results.json -> ptz_parking.hero_badge`): "SIMULATED camera faults: 36/36 blind seconds caught (5 faults), 0 false alarms". These faults are injected into the same ptz scene the thresholds were tuned on, so read it as "the detector does what it was built for", not as a field accuracy.

## Eyeball checks
Frames I looked at myself (images are in `evidence/` and `out/`):
- `evidence/lisbon_parking_sheet.jpg`, `seq_lisbon_parking.jpg`: 16 frames across the clip plus 24 frames at 0.7 s steps. Every frame is a different street photo or a yellow title card. This is a slideshow, not "a fixed camera", so I dropped the planned lisbon window and made lisbon the refusal case.
- `evidence/car_detection_sheet.jpg`: overhead empty asphalt, a white car, then blue and red cars pass through. No parked car.
- `evidence/ptz_parking_sheet.jpg`, `edges.jpg`: view A until ~155 s; the camera is already moving at 155.4 s; lamp-post close-up 158-236 s; at 237.4 s the wide view is coming back dark; at 239-241 s it is settled and panned. Ledger blind stretch 157-238 s agrees within 2-3 s.
- `evidence/cut68.jpg`: frames 67.8/68.0/68.4/68.6: hard jump in exposure and scene content at 68.2 s. In `eye_B.jpg` the black SUV is there at 66 s and gone at 70 s.
- `evidence/eye_A.jpg`, `eye_B.jpg` (raw frames, box drawn at the car's place, 18 times each): car A (dark teal sedan) is present at 5, 30, 60, 66, 70, 90, 107, 125, 138, 150 s and again at 256, 270, 300, 312 s; at 162/200/236 s the car is not in the picture; at 138 s a man walks up to its front-left corner; 246 s is a dark transition frame I did not count. Car B (black SUV) is present at 5-66 s, and the spot is empty at 70, 90, 125, 138, 150, 256, 270, 300, 312 s (a car drives across the spot at 107 s, not counted). Written down in `ground_truth.json` before comparing.
- **Strip vs eyeball, 2 cars: 33 observations, 33 exact, 0 grey, 0 contradictions** (before the dropout fill it was 31 exact + 2 grey, both at second 256 where alignment dropped for 1 s). The fill only applies to 1-2 s gaps whose two neighbours have the same state (here 250 s and 256 s: present for car A, absent for car B); my eyeball frame at 256 s agrees with the filled state for both cars. The 250 s fill is not covered by an eyeball frame.
- `evidence/tour_pairs.jpg`: before/after crops of 12 cars across the tour: cars 1, 3, 4, 5, 6, 9, 10, 11 look identical (lighting differs); car 2 (far-left dark hatchback) is gone after the tour; car 7's box overlaps a neighbour.
- After the review fixes I viewed the new `out/ptz_parking/hero.jpg`, `out/sim_a/hero.jpg`, `out/sim_b/hero.jpg`: no grey slivers in either strip, headline 81 s = strip label 81 s, the footer is complete (ellipsized only if too long), the vertical gap between strips and the bottom row is gone, the header badge reads 36/36 with 0 false alarms, the sim_b scratch is visible in the "after" crop next to the amber "NOT detected (known miss)" box, and the TOUCH? line sits under the contact thumbnail (0.87 on ptz, 0.88 on sim_b where both crops already carry the scratch).
- `out/*/hero.jpg` and `out/ptz_parking/check.jpg` viewed after each fix. Real bugs found this way: the ORB homography was locking onto the camera's on-screen "ID: 001" text (identity transform, a 12 px error) and a false ABSENT for 17 s caused by it; fixed by dropping compact consensus blobs.

## Simulated or faked (explicit)
- **sim_a and sim_b are SIMULATED fault copies** of real ptz_parking 69-154 s, written by `sim.py`: black lens cover, flat grey cloth plus noise, Gaussian defocus (sigma 12), repeated last frame. Blue brackets on their heroes mark the injected truth.
- The painted scratch and dent in sim_b are SIMULATED.
- Detections for the sim clips are not re-detected: they are the real cached tracks of the same frames with detections removed in cover/cloth/blur windows (a covered lens detects nothing, my assumption) and repeated during the freeze.
- The car for the sim clips was chosen on the clean footage (what a user would do after being told which car to watch), then passed to the sim run.
- The ledger "overnight" is minutes of daylight footage: ptz_parking is a 5 min demo reel (with one jump cut), not a night.
- Nothing else is faked: ptz_parking, lisbon, car_detection and the 10 false-alarm clips are real, unmodified files.

## What fails / limits
- **Fine damage is not detected.** Scratch/dent recall 0% (same view) to 10.6% (across the tour) at the shipped false-alarm rate. The verdict is therefore worded "THERE, SAME CAR" with "damage is NOT checked" on the stamp, and sim_b shows the miss openly. The product answers "same car, still there"; "damaged?" needs the VLM pass on the day. TOUCH? has the same blind spot: it scores the car before/after a visit, so a scratch would not move it either.
- Only one real scene supports a parked-car ledger. lisbon_parking and car_detection give none; the three-clip default set is really one clip.
- ptz_parking is an edited reel: the 01:08 jump cut is what produced the only real ABSENT, and nothing in it is overnight.
- Contacts are geometry only (feet inside a zone: car box lower half grown by 30% of its width). One event on the main car; 16 across 40 candidate rows. Feet-near is not touching, and small distant people are missed by the detector.
- Alignment dropouts: 1 s gaps appear (250 s, 256 s in the view-B part). If a gap is at most 2 s and both neighbouring seconds have the same state, it takes that state (counted in `summary.filled_gap_s`, 2 on ptz) so the strip and headline agree; any other short gap stays grey and gets no before/after verdict (gaps under 3 s). The fill is an inference, not an observation.
- The only real "absent" is a video edit (jump cut at 01:08), and nothing here is overnight or at night: the ptz reel is 5 minutes of daylight.
- A car partly hidden by a neighbour reads "unsure" (grey): 17 s for tid 926 before the cut.
- A new PTZ preset that shows the spot only partly, or from a different place, is "view changed": the ledger gives up instead of comparing (correct, but it means a tour that never returns gives UNKNOWN).
- Freeze rule: at least 2 s of identical frames at 5 fps. A stream that repeats frames (low frame-rate source) reads as frozen: retail_cctv 98-111 s.
- Thresholds were tuned on the same scene they are reported on (no held-out set). Night footage was only tested for false alarms (`rochester_night` 0 s flagged), not for the ledger.
- A car replaced by a different one that fits the same box is caught by the crop test (100% of 110 pairs), but no real swap occurs in the test footage.
- The DataEngine stand-in problem above (counts, not boxes).

## Kill test result
Criterion: "catches the simulated lens cover and freeze at the right seconds; strip matches your eyeball check for 2 cars."
- **PASSED.** sim_a: lens cover found at 10-20 s (truth 10-20), freeze at 55-63 s (truth 55-63): onset and offset error 0 s. Over both sim clips five injected faults (cover, freeze, cloth, defocus, freeze; 36 blind seconds in 170 s of video): 36/36 found, 0 false blind seconds in 134 clear seconds, cause named correctly for all 36.
- Strip vs eyeball: 33/33 exact, 0 contradictions, on 2 cars (one stays, one is gone after a cut). Two of those 33 rest on the 1 s dropout fill (see limits).
- What the kill test does not cover: scratches (missed, and shown as a miss on sim_b), swapped cars (no real example), night (not tried). The planned lisbon fault copy was replaced by ptz windows because lisbon_parking is a photo slideshow. Thresholds were tuned on the same scene, so 36/36 with 0 false alarms in 134 clear seconds is optimistic.

## Rebuild plan on the day (hours)
About 8 h with the real services, to be rebuilt from scratch (event rule):
1. 1.0 h: ingest the clip, get per-frame boxes (hosted YOLO if it returns boxes, else our own), pick the parked car (stationary box centre, IDs broken, so match by box IoU).
2. 2.5 h: per-second ledger: 5 fps health scalars, reference-frame homography with on-screen-display rejection, presence by box IoU in reference coordinates with an appearance fallback.
3. 1.0 h: blind causes (covered, frozen, blurred, view changed) and the before/after SSIM compare.
4. 0.5 h: contacts (feet in zone) with thumbnails.
5. 1.0 h: Cosmos3-Reason on before/after crops for damage, Cosmos-Embed1 view-change cross-check.
6. 1.0 h: VastDB ledger table and the UI/strip, hero image.
7. 1.0 h: rehearsal on a fresh clip with injected faults.

## Files
`run.py` CLI, `ledger.py` core, `sim.py` simulated faults, `render.py` hero/strip drawing, `calibrate.py` + `calibration.json`, `ground_truth.json` (eyeball), `test_smoke.py`, `results.json`, `out/<video>/{hero.jpg, report.html, strip.png, check.jpg, ledger.csv, thumbs/}`, `evidence/` (contact sheets I viewed).
Footage credits are in each report footer (`pk.Report.credit`): ptz_parking by Supercircuits (public domain), lisbon_parking CC BY-SA 3.0, car_detection and the 8 other Intel clips CC BY 4.0, retail_cctv public domain, others as listed in protokit's catalogue. Local tests only.
