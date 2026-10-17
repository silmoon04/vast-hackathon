# DAY 14: dashcam recording in, Operation Snap packet out

## What it is
Give it a long dashcam file. It finds the few seconds where another vehicle comes close (close pass, cut-in, fast closing) without anyone scrubbing, ranks the top 5, and for each builds a 10 s clip (5 s before, 5 s after), a boxed key frame, a zoom crop, a Cosmos description and a Snap packet draft with a 14-day clock ("recorded DATE -> deadline DATE+14, N days left"). The recording date is read from the footage's burned-in overlay when there is one (`--recorded` overrides, today is the last fallback).
Why: West Midlands Police, April 2025: 1,321 of 2,017 Snap reports (65%) got no action before the 14-day window; the force said all but 12 were its own delay (one force; road.cc). Speed of reporting matters.

Run: `python run.py --all` (about 30 s with caches) or `python run.py --video dashcam_urban --recorded 2026-10-10 --start-time 18:40:00 --location "..."`. Outputs: `out/<video>/hero.jpg`, `report.html`, `cand_N/{clip.mp4,key.jpg,crop.jpg,storyboard.jpg,packet.json}`, top-level `results.json` (always merged, so the two sim runs survive `--all`). Simulated test: `python inject_test.py` then `python run.py --video out/sim/sim_cutin.mp4` and `--video out/sim/ctrl_highway.mp4`. Tests: `python -m pytest -q test_smoke.py` (10 tests, about 8 s). Options added in this revision: `--cosmos-triage` (turn the old "Cosmos says parked -> demote" back on, default off).

## Demo script (90 s)
1. (10 s) Open `out/dashcam_urban/hero.jpg`: 2 minutes of dashcam in, five numbered moments on the risk curve and five thumbnails, an amber "REVIEW: CLOSE INTERACTION" stamp (never a green "ready": the tool finds moments for a person to judge), and a clock bar that is red because the overlay says the footage was recorded on 18 Jun 2013 ("PASSED 4,855 days ago"). Say it out loud: the clock is read from the footage, and for this old clip it correctly says too late.
2. (25 s) Open `report.html`, go to #2 (the silver BMW at 00:48.4, passing about 1 m off the left front wing), play `cand_2/clip.mp4`, show the storyboard. #1 is a pass of two parked cars in a car park: say that the tool cannot tell parked from moving and a human swipes past it in two seconds.
3. (25 s) Show `cand_2/packet.json`: stream time, wall clock derived from the overlay time, SHA-256 of the original, statement draft with blanks the user must fill truthfully, plate field "NOT extracted", the clock.
4. (15 s) Change the clock live: `--recorded 2026-10-10` gives "7 days left" (amber bar, checked), `--recorded 2026-09-28` gives red "PASSED 5 days ago".
5. (15 s) Honesty beat: `out/motorway_a40/hero.jpg` (grey, "NOT A DASHCAM", no clock, no packet), `out/ctrl_highway/hero.jpg` ("NO INCIDENTS FOUND", a clean road gives zero) and `out/sim_cutin/hero.jpg` (SIMULATED cut-in found).
Do NOT claim that dashcam_urban contains a reportable offence: it does not (see below).

## How it maps to the VAST stack on the day
| Step here (local stand-in) | On the day |
|---|---|
| YOLO26s + ByteTrack per-frame boxes (`pk.tracks`, cached) | DataEngine YOLO11s. NOTE: the DataEngine table is per 5 s segment COUNTS, not boxes or tracks. This idea needs per-frame boxes, so we would run our own YOLO on the footage (or ask whether DataEngine exposes boxes). Biggest rebuild risk. |
| Ground-plane geometry + severity + peaks (`core.py`, CPU) | Same code, unchanged. |
| Optical-flow ego motion (`ego.py`, CPU, ffmpeg + Farneback) | Same code. |
| Local Cosmos3-Edge one-frame colour/body description and overlay date read | Cosmos3-Reason on the key frame, ideally on the 10 s clip (it can answer "is that car moving", which the stand-in cannot). Nothing depends on it except the recording date, which has the `--recorded` fallback. |
| Cosmos-Embed1 | Not used. Could index every candidate clip and let a user search "car cuts in front". |
| VastDB | Store candidates, packets and the clock (deadline per incident) as rows; query "what is expiring in 3 days". Not built here. |
| Canary ASR | Not used. Could transcribe the driver's spoken note in the audio track for the statement. |

## Results on test footage
Measured on this laptop (shared with other prototypes). Tracks and Cosmos answers were cached unless stated.

| clip | camera detected | verdict stamp | candidates | what the eyeball check says | wall time |
|---|---|---|---|---|---|
| dashcam_urban (120 s, primary) | dashcam | REVIEW: CLOSE INTERACTION (amber) | 5 | order: #1 parked dark estate in the car park (close_static), #2 silver BMW (real, moderate confidence), #3 parked white VW, #4 grey hatchback overtaken (unclear), #5 parked white SUV. Near-field precision@5 = 5/5, reportable precision@5 = 1/5, reportable in top 3 = 1 | 21 s warm; 74 s on the first run after the prompt change (date read 15 s, Cosmos 37 s cold, evidence 20 s) |
| motorway_a40 (24 s) | fixed | NOT A DASHCAM (red stamp, grey badge, grey clock, grey ticks) | 3 look-alikes, NO packet | correct refusal | 7 s |
| person_bicycle_car_detection (54 s) | fixed | NOT A DASHCAM | 2 look-alikes, NO packet | correct refusal | 1 s |
| ctrl_highway (20 s, unmodified excerpt of dashcam_urban 96-116 s) | dashcam | NO INCIDENTS FOUND (empty frame panel removed) | 0 (max pool severity 0.11 < floor 0.15) | correct: nothing happens | 1 s |
| sim_cutin (same excerpt + SIMULATED cut-in at 8.0 s) | dashcam | REVIEW | 1: cut-in at 9.27 s, severity 0.69, TTC proxy 1.0 s | found, ranked #1 | 5 s |

Overlay date reading (Cosmos stand-in, 3 frames at 10/50/90% of the clip, a date is accepted only if at least 2 reads agree and it is a real past date): dashcam_urban, sim_cutin and ctrl_highway all read 2013-06-18 with 3/3 agreeing reads, which matches the burned-in "2013/06/18" I saw in the frames. The same reads also gave the start time 19:40:09 (read times at t = 12, 60, 108 s were 19:40:22, 19:41:10, 19:41:58, consistent to the second), so packets now carry a wall-clock time for each candidate. Result for this clip: "PASSED 4,855 days ago (13.3 years)". Not tested on a clip without an overlay except the two fixed-camera clips, which are not read at all (dashcam mode only); a clip with no overlay would fall back to "assumed = today" and say so.

Ranking is geometry only: [parked car in car park 33.7 s, BMW 48.4 s, parked VW 71.5 s, grey hatchback 87.3 s, parked SUV 59.9 s]. Reportable events in top 3 and top 5: 1 and 1.

Cosmos stand-in measured again this round (8 pooled moments, my labels: 33.7 parked, 48.4 moving, 59.9 parked, 71.5 parked, 87.3 slow/unclear):
- one frame, "PARKED or DRIVING?": answered PARKED for 7 of 8, including the moving BMW (no discrimination).
- three frames 1 s apart, same question: answered DRIVING for 8 of 8 (no discrimination).
- free-text description asked to say parked or moving: right on 1 of the 4 labelled moments (parked for the BMW, moving for the two parked cars 59.9 and 71.5).
So the old "Cosmos says parked -> demote" rule (which looked like 2 correct, 1 wrong in the previous revision) was luck of wording, and it is now OFF by default. Consequence: the parked car at 33.7 s is back at #1 and the real BMW is #2.
Cosmos colour/body description of the 5 final candidates against my labels: colour right for 2 of 5 (#3 white VW and #5 white SUV). #1 dark grey estate called "silver sedan", #2 silver BMW saloon called "white small hatchback", #4 grey hatchback called "Reddred car". The prompt now forbids make/model and motion claims (the old "Honda Jazz" error is gone, replaced by a colour/body error), and the hero says "local stand-in, can be wrong".
The relative-motion test in the packet notes ("moving-like/static-like") marks the parked cars moving-like (0.92 and 1.0 for 33.7 and 71.5 s). It is shown labelled "unreliable" and is not used for anything.

## Fixes made in this revision (reviewer list)
1. Stamp logic: hero stamp comes from `verdict()`: fixed camera -> NOT A DASHCAM; no candidates -> NO INCIDENTS FOUND on a statement card with no empty frame panel; otherwise amber REVIEW: CLOSE INTERACTION. I first wrote a "SNAP-READY if severity >= 0.8, not parked, moving-like" rule as the reviewer suggested, and it stamped SNAP-READY on the parked car at 33.7 s (severity 0.84, moving-like 0.92: both signals wrong), so I deleted the green tier. A test asserts no heuristic can produce it.
2. Refused heroes: grey badge "look-alike only: not analysed for Snap" (moved to the bottom-left so it cannot collide with the stamp), grey box and ticks, grey dead clock bar "NO PACKET: NOT A DASHCAM, SO NO CLOCK". The same grey clock is used for the no-incident case.
3. Key frame: the hero main panel is now a context zoom (at least 3x the box and 40% of the frame width) on the sharpest un-clipped moment of the vehicle within 1.2 s of the peak, with a full-frame inset showing the window, plus a row of all five candidate thumbnails. I used this instead of the t-0.4/t0/t+0.4 strip (the strip exists per candidate as `storyboard.jpg` and in the report; in the hero it would have made every tile tiny). Cosmos prompt: colour and body type only, no make/model, no motion claim.
4. Recording date from the footage: read from the burned-in overlay (see above), used as the clock default, with `--recorded` taking priority and today as the last, labelled fallback. The hero says where the date came from. On this clip the clock is red and says PASSED, which is the honest answer.

## Eyeball checks (frames I viewed)
- Earlier rounds: contact sheets of all three default clips and 1 s / 0.2 s sheets of dashcam_urban 22-120 s (the ground truth in `ground_truth.json` comes from these): 0-22 s dark car-park exit; 33-35.5 s car park with parked estate and dark car; 47-48.4 s white van at the kerb then the BMW; 58.5-61 s white SUV; 70.5-72.5 s white VW; 86-88.6 s grey hatchback overtaken at about 1.2 m; 118.8 s brake lights on a lead car 15-20 m ahead (normal slowing).
- This round, all five new hero.jpg files viewed at full size: dashcam_urban (amber REVIEW, #1 is the dark estate with a clean un-clipped box and the inset, five thumbnails readable, red PASSED bar and the overlay-source line), ctrl_highway (a big 0 and NO INCIDENTS FOUND, no frame panel, grey clock), motorway_a40 (grey badge at bottom-left, grey box, grey ticks, grey clock; red NOT A DASHCAM stamp), person_bicycle_car_detection (overhead parking lot, same grey treatment), sim_cutin (SIMULATED banner, amber stamp, the composited MPV boxed and large). Also viewed the `--recorded 2026-10-10` variant of the dashcam hero: amber clock bar, 7 of 14 segments lit, "7 DAYS LEFT".
- Storyboard of cand_1 of the previous run (47.8-50.4 s): the BMW enters at the bottom-left, swings past the left front wing at t+0 and is gone by t+1; the ego car is turning at the junction. Moderate confidence that the BMW is moving.
- Overlay: the burned-in strip at the top reads 2013/06/18 and a clock; the bottom strip has km/h (shows "-") and G-sensor x/y/z. The G-sensor values were not read out.
- Not checked by eye: that the hero stamp state is correct for every possible clip; it is derived from five clips only.

## Simulated or faked
- `sim_cutin.mp4` is SIMULATED: a real car cut-out (YOLO-seg mask, from dashcam_urban at 119.7 s) composited into the real 96-116 s excerpt with the same ground-plane model the detector uses, so it tests ranking and plumbing, not real-world realism. `ctrl_highway.mp4` is the unmodified control (re-encoded the same way). Both inherit the 2013 overlay, so their clock says PASSED too.
- Dates: for a clip with a readable overlay the date is read, not assumed; for any other clip `--recorded` defaults to today (labelled "assumed" in the packet). The overlay read is by the local Cosmos3-Edge stand-in, 3 frames, accepted on 2-of-3 agreement; I verified it on the one overlay style I have (this camera).
- Cosmos is the local Cosmos3-Edge stand-in, one frame, not Cosmos3-Reason on a clip. Cached answers come from earlier runs.
- Camera height 1.2 m, focal 0.45 x width and horizon 0.49 x height are assumed or were fitted on dashcam_urban itself. Range, lateral offset and TTC are proxies for ranking, never measurements.
- Thresholds (severity floor 0.15, lane widths, pool of 8, min separation 8 s) were set while looking at this clip, so the numbers above are optimistic; there is no held-out dashcam footage.

## What fails / limits
- Precision is weak where it matters: dashcam_urban holds one plausibly reportable moment (the BMW) in 2 minutes, I am only moderately sure it is moving, and it does not clearly transgress anything. Headline: near-field precision 5/5, reportable precision 1/5, and the top-1 on the hero is a parked car.
- Monocular geometry cannot tell a parked car the driver passes closely from a moving car, and the local Cosmos cannot either (measured above, chance level). That is the product's core question ("is the other party at fault"). Only Cosmos3-Reason on the clip (to be tested on the day), ego speed, or a human can answer it. The honest UI is therefore "REVIEW", with five ranked thumbnails for a person to dismiss in seconds.
- Objects at the frame edge are distorted by the wide lens, so lateral offset is biased outward; the cut-in rule fired correctly only on the simulated case. No real cut-in was available to test.
- Sudden braking of the camera car is not detected (flow speed too noisy; the G-sensor in the overlay is the obvious fix). Pedestrians and cyclists are not scored.
- Motorway/overhead clips are refused by a simple rule (median scene expansion), which would also refuse a dashcam that never moves. Overlay dates are only read in dashcam mode.
- No plate OCR, no location. The statement draft has blanks the user must fill truthfully; the tool never asserts that an offence happened.
- Speed: 21 s for 2 minutes with everything cached; 74 s on the first run after the prompt change (cold Cosmos calls plus the 3 overlay reads, shared server). "Packet in under a minute" is only true warm; I did not show it cold.
- The Cosmos stand-in's colour/body text is right for 2 of 5 candidates; the hero shows it next to the true crop with a "can be wrong" label.

## Kill test result
Criterion: on dashcam_urban at least one of the top 3 is a real close interaction by eyeball. PASS, unchanged but now stated more plainly: the top 3 are a parked dark estate passed at about 1 m in a car park (#1, close_static), the silver BMW passing about 1 m off the left front wing (#2, moving, moderate confidence) and a parked white VW (#3, close_static). All five are real vehicles within about 1-2 m (near-field precision 5/5). Strict reading (a moving other party that a police portal would care about): 1 of 5, and it does not clearly transgress anything. Retrieval works, ranking by geometry alone cannot put the real one first, and the demo clip cannot show a convincing offence; with real submitted footage the value depends on how many true incidents are in it. If the user needs a hero with a real event at #1, that needs the hosted Cosmos3-Reason motion triage to work (not shown here).

## Rebuild plan on the day (hours)
About 8 h: 1 h get per-frame boxes (own YOLO on the footage if DataEngine only exposes 5 s counts), 1 h port `core.py`/`ego.py` as is, 2 h Cosmos3-Reason on the clip for "is it moving / what happened" plus triage (and re-test `--cosmos-triage` against hand labels before trusting it), 1 h packet + clock + overlay date + VastDB rows, 1 h clip/report/hero, 2 h find and label a few real incident clips to tune thresholds honestly.
