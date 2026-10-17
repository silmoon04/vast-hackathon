# LAST UPRIGHT

Local prototype. Run: `..\..\.venv\Scripts\python.exe run.py --all` (about 1 min with a warm Cosmos cache, up to about 4 min cold).

## What it is

For care-home night staff and falls reviews: from per-person pose keypoints it answers "when was the resident last upright, when were they first on the floor, how fast was the transition (FAST = fall, slower = deliberate lie/sit), and how long have they been down", then asks Cosmos to verify the key frame (floor vs bed).
Output per clip: a hero card (verdict stamp, last upright / on the floor / flat frames side by side, posture strip, observed long-lie clock, box-only baseline card), an HTML report, an annotated mp4 and a SIMULATED long-lie clock strip (clock_sim.jpg / .mp4).

## Implementation notes

- YOLO26s-pose + ByteTrack provide cached pose tracks through `pk.poses`. The box-only fallback is evaluated separately and is weaker.
- `pk.segments` provides YOLO26s counts per 5 s segment for the candidate funnel and box-only baseline.
- `pk.vastdb` provides duckdb queries limited to SELECT, WHERE and ORDER BY; the tests check the restricted dialect.
- Local Cosmos3-Edge (`pk.vlm`) can agree, disagree or add floor/bed context to a key frame. It does not decide the verdict.
- Cosmos-Embed1 and Canary ASR are not implemented.

## Results on test footage

Ground truth was written from contact sheets (`ground_truth.json`). The skate_fall times (last upright 6.4 s, ground contact 6.9 s, flat 7.25 s) come from a 0.2 s contact sheet that also had the algorithm overlay visible, so they are not blind. Classes: FALL (fast stand-to-floor), LIE (lying posture, no fast fall), NO_ALARM. All poses came from the protokit cache (`pose_source` in `results.json`).

Three times are reported for a fall. **Last upright** = last frame in the smoothed UPRIGHT state (or PARTIAL: body axis near vertical with only legs in view) before the DOWN run; a SITTING frame never counts. **Ground contact** = first frame where the hips are at foot level (or DOWN). **Flat** = first frame with the torso >= 70 degrees from vertical. FAST/slow is judged on last upright to flat (the longer, conservative interval); the clock starts at ground contact.

| clip | truth | verdict | last upright | ground contact (truth / ours) | flat (truth / ours) | last upright to flat | observed on floor | Cosmos "lying on floor?" / place | box-only baseline |
|---|---|---|---|---|---|---|---|---|---|
| skate_fall | FALL | **FALL** | 6.40 s (truth 6.4, legs only in view) | 6.9 / 6.96 s (+0.06) | 7.25 / 7.28 s (+0.03) | 0.88 s = FAST (ground contact after 0.56 s) | 1.04 s, then out of view | yes / floor | FALL (right), but fires at 6.48 s: 0.42 s before ground contact |
| floor_lie | LIE | **LIE** | not seen | down at first sight | - | not observed | 2.84 s (whole clip) | yes / floor | LIE |
| assist_elderly | LIE (in bed) | **LIE, shown as LYING IN BED** | not seen | down at first sight | - | not observed | 9.6 s (whole clip), no floor clock | no / bed | **FALL (wrong)** |
| help_sit | NO_ALARM | **NO_ALARM** (SAT event at 5.76 s) | n/a | none | none | speed not measurable (legs not seen) | 0 | no / other | NO_ALARM |
| sit_down | NO_ALARM | **NO_ALARM** (seated 100% of frames) | n/a | none | none | n/a | 0 | no / floor (place wrong, see limits) | NO_ALARM |
| classroom | NO_ALARM | **NO_ALARM**, 0 falls, 0 DOWN frames | n/a | none | none | n/a | 0 | no / sofa | NO_ALARM |

- Pexels clips labelled correctly: **5 of 5**; classroom falls: **0**. Cosmos answered the floor question correctly on 6 of 6.
- The skate_fall FALL rests on **legs-only evidence at the upright end**: from 5.3 to 6.4 s the camera is tilted down and only the legs are in view (feet on, then just off, the board), so the "upright" frame is a PARTIAL-view frame with the body axis at 4 degrees. The torso is measured at the flat end. A care-home camera that sees the whole body would not have this weakness.
- skate_fall is a knife edge on speed: stand-to-flat is 0.88 s against the 1.5 s FAST limit, and moving the limit to 0.8 s turns the clip into LIE (see sensitivity).
- classroom posture timeline matches what I saw: front man UPRIGHT from 4.1 s to 15.6 s then SITTING; a student further back UPRIGHT from about 21 s.
- VastDB funnel: segments flagged for a Cosmos call, skate_fall 1/3, floor_lie 1/1, assist_elderly 2/2, help_sit 0/3, sit_down 0/3, classroom 0/7.
- Extra negative, not tuned on: `one_by_one_person_detection` (139 s, 6 people tracks, walking): 0 DOWN events, 0 falls. `people_detection`, `retail_cctv`, `store_aisle_detection` were skipped (no cached poses; the GPU lock was held by other spikes and a CPU pose extraction was killed after 12 min).
- Threshold sensitivity (`run.py --sensitivity`, one threshold moved at a time, pose only, 15 configurations): 13 keep 5/5 and 0 classroom falls; `FAST_S 0.8` and `KP_VIS 0.5` drop skate_fall (4/5, the kill test still passes).
- Runtime: `run.py --all` about 60 s with a warm Cosmos cache (hero + report + annotated mp4 + clock_sim for 6 clips), up to about 4 min cold; the Cosmos server was shared and slow (calls of 5 to 70 s), so there is a 75 s timeout and a circuit breaker.

## Eyeball checks

- Contact sheets (12 frames each) of all six clips before designing. Saw: skate_fall is a rooftop skateboarder; floor_lie is a woman lying on a white floor for the whole 2.8 s; assist_elderly is an older man lying on a bed with a carer touching his forehead; help_sit is a man seen from behind (head out of frame until about 6 s) lowered into a wicker chair by two carers; sit_down is a portrait clip of a man seated on a wall the whole time (there is no sit-down in it); classroom is four seated people, the front man stands about 4.5 to 15.5 s.
- **Review fix, skate_fall 4.8 to 7.6 s every 0.2 s with the pose overlay** (this is what showed the first version was wrong): 5.0 crouched on the board (full body, UPRIGHT 31 deg), 5.2 popping up and cut by the top of the frame, 5.4 to 6.2 only the legs are in view (camera tilted down), 6.4 both feet in the air, 6.6 tumbling, 6.8 hand on the ground, 7.0 sprawled on the ground, 7.2 to 7.4 flat. The first version's "LAST UPRIGHT 00:07.1" card was the sprawl at 7.0 to 7.1 (state SITTING, torso 19 deg): wrong. The real stand-to-ground time is about 0.6 s and stand-to-flat about 0.9 s, not 0.16 s.
- New skate_fall hero viewed: left frame 00:06.4 shows two legs with shoes (feet just off the board, label "on feet, partial view: body axis 4 deg"), middle 00:07.0 shows him on the ground on hand and hip, right 00:07.3 shows him flat; the posture strip markers sit on a violet block (last upright, partial view) and a blue/red boundary (on floor). The strip is trimmed to the observed window (1.7 to 8.8 s). Numbers on the card agree: 0.9 s, 0.56 s + 0.32 s, 1.0 s on the floor.
- `clock_sim.jpg` (4 cards: 1.0 s OBSERVED, then SIMULATED 01:00 CHECK NOW, 05:00 ESCALATE, 1:00:00 LONG LIE) and a frame of `clock_sim.mp4` viewed (banner turns red as the clock runs). Annotated skate_fall mp4 at 6.9 s (skeleton on the hand/hip pose) and 7.6 s (red DOWN box, banner "FALL 00:07.0 on floor 0.6 s").
- help_sit 4.0 to 9.0 s every 0.4 s and classroom 3 to 27 s every 1.5 s: fixed the sit and stand windows in `ground_truth.json`.
- All six `hero.jpg` viewed after the redesign: skate_fall, assist_elderly and help_sit full size, floor_lie, sit_down, classroom in a montage. The hero panels now draw only the person under review (help_sit's clutter of carers' half-boxes is gone), the box labels moved to the caption under each frame, and the marker colour matches the block it sits on (help_sit "on feet 5.1 s" is now violet on the violet block).
- Not viewed: frames of `one_by_one_person_detection` (only counts: 0 DOWN events); full playback of the annotated mp4s.

## Simulated or faked

- **Long-lie clock beyond the observed time is SIMULATED.** The footage never shows anyone staying down more than 9.6 s (and in the one real fall, 1.0 s). `clock_sim.jpg` and `clock_sim.mp4` advance the clock to 1 min / 5 min / 1 h with `posture.clock_state` and carry a SIMULATED ribbon on every card but the first; the hero ladder says the same. Rung times are configurable demo defaults, not clinical guidance.
- VastDB is duckdb in-process; Cosmos3-Reason is the local Cosmos3-Edge; the Cosmos check is cached by content.
- No care-home footage: a skateboarder, stock photos of lying and assisted people, a classroom. No night, IR, bed rails, blankets over the body or ceiling cameras.
- No fault injection or synthetic video. The unit tests in `test_smoke.py` use synthetic skeletons (labelled as such there).
- Ground truth is one viewer (me) looking at frames, and the skate_fall times were fixed while the overlay was visible.
- Thresholds were set while looking at skate_fall, assist_elderly and help_sit, so the 5/5 is in-sample. The only out-of-sample number is the 0 false alarms on one 139 s walking clip.

## What fails / limits

- The kill test does not separate pose from boxes: a box-only baseline (what the DataEngine alone gives) also gets 4/5 and 0 classroom falls. What pose adds in these clips: the bed close-up is not called a fall (the box-only baseline calls `assist_elderly` a FALL), the ground-contact time is within 0.06 s of the truth where box-only fires 0.42 s early (while the skater is still airborne), stand vs sit vs down states, and a measurable transition speed. Cosmos is what turns the bed close-up into "lying in bed".
- Fall speed needs torso keypoints at the flat end and a near-vertical body at the other. With only partial keypoints at the flat end (bed close-up, head cut off, legs hidden behind a desk) the speed is reported as not measurable and the label can never be FALL; those cases end as LIE or NO_ALARM. skate_fall's upright end is legs-only (see results).
- A slow slump (upright to flat over more than 1.5 s) is labelled LIE ("check"), not FALL. The 1.5 s threshold is a guess, not validated on real elderly falls; skate_fall (0.88 s) is only 0.08 s above `FAST_S 0.8`, which flips it to LIE.
- `skate_fall` resident leaves the frame 1.0 s after landing, so the clock only observes 1.0 s. Anything out of view is not counted. There is no clip here where someone stays down after a fall, so the long-lie clock has never run on a real long lie.
- Bed vs floor depends on Cosmos, which is sometimes wrong: it called the wall in `sit_down` "floor" and the classroom chairs "sofa" (display only, hidden for no-alarm clips, but it shows how loose the one-word answer is).
- Tracker ID switches split one person (assist_elderly: 4 person tracks for one man and his carers, and one carer skeleton is drawn on the merged key frame); I merge overlapping lying tracks and stitch short gaps, which is a heuristic.
- One fixed viewpoint; an overhead or ceiling camera breaks the torso angle. Several people lying close together would merge.
- Pose is not supplied by the documented DataEngine YOLO11s boxes/counts path; this implementation depends on its own pose model.
- Cosmos latency on the shared local server was 5 to 70 s per clip; the real hosted service should be faster but is unmeasured.

## Kill test result

Criterion: correct label on at least 4 of the 5 Pexels clips and zero falls on classroom. **Passed: 5 of 5 and 0 falls** (re-run after the fixes). Read it with the caveats above: it passes for a box-only baseline too (4/5), the thresholds were tuned on the same clips, the one real fall depends on legs-only evidence and sits 0.08 s above a threshold that flips it, and only one unseen negative clip was available. Decision value: the idea works on real pose data and looks convincing, but the pose model is not a provided service, the footage is not care-home footage, and the long-lie clock is a simulated payoff, not a measured one.

## Files

`run.py` (CLI), `posture.py` (rules, hysteresis, events, clock), `render.py` (hero card, overlays, SIMULATED clock strip and mp4), `poses_src.py` (shared pose cache, CPU fallback if missing), `ground_truth.json`, `results.json`, `test_smoke.py` (13 checks, about 1.5 s), `out/<video>/{hero.jpg,report.html,annotated.mp4,key_left.jpg,key_mid.jpg,key_right.jpg,clock_sim.jpg,clock_sim.mp4}`.
Options: `--all`, `--video NAME`, `--no-vlm`, `--no-video`, `--negatives`, `--sensitivity`, `--clock SECONDS` (print the ladder state), `--clock-steps 60,300,3600` (SIMULATED cards in clock_sim).
