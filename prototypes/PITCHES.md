# 15 prototypes, pitched from their real outputs

Built and tested on local stand-ins and openly licensed clips. Each pitch was written from the prototype's README, results and hero image, plus a skeptical reviewer's verdict. Visual gallery: gallery.html.

## The decision

**Plan A.** ON THE LINE, rebuilt for Pack D as a voice-driven street watch. Someone plays the caller, Canary transcribes them, and Pack D replays at frame rate. Each question pops a card on the next frame: 'Is the silver car still there?' gets 'NO, last seen 2.8 s ago, left via LEFT edge', with crop, path, Cosmos3-Reason caption and a latency chip. A VastDB events table holds the answers and the incident log drafts itself. Times are clip-relative. Cut to 5 question types to fit roughly 7 h.

**Plan B.** OVERNIGHT LEDGER, if Canary or the voice loop is unreliable at 10:30. A judge clicks one parked Pack D car and gets a strip across both days: green present, red gone, grey UNKNOWN with the cause. Before/after crops are checked by Cosmos3-Reason for damage and Cosmos-Embed1 confirms the view is unchanged. The peak_car drop in SQL is the 'car left' pre-filter. The demo is the THERE, SAME CAR stamp plus thumbnails of people near the car. If there are no per-frame boxes at all, fall back to Blind Time plus UNCUT on segment SQL.

**Fold in.** Overnight Ledger becomes the 'how long has it been there / is it the same car' question type. Blind Time's health rows become the honesty rule: answer COULD NOT SEE, never NO, when the camera was blind. Night Exit becomes 'who crossed the kerb line since T', with Embed1 ranking. Day 14's SHA-256 evidence packet becomes the call-close export. If time allows, MIMIC's drawn-path search becomes an operator sidebar. Everything else stays in the pitch deck as roadmap.

**Overall.** Last night proved the pipelines run end to end on local stand-ins. Several engines refuse honestly: the slideshow was rejected, and blind or no-match states are reported instead of guessed. ON THE LINE answers quickly and correctly on a handful of held-out clips. It did not prove anything on Pack D. It also did not prove that hosted YOLO11s returns per-frame boxes, or that Cosmos3-Reason is good enough (local Cosmos was near chance). Most metrics are self-written, self-graded, in-sample and tiny. Pick ON THE LINE, with Overnight Ledger and Blind Time folded in. The 10:30 gate is per-frame boxes plus clean tracking of Pack D cars. Budget is tight: 10 h of planned work in about 7 h, so cut the scope early.

| Rank | Idea | Verdict | Win score | Why |
|---|---|---|---|---|
| 1 | ON THE LINE | STRONG BACKUP | 6/10 | Suits a fixed camera that mostly sees cars. Canary gives it a natural way in, and a live spoken question answered by a card is the strongest demo. Held-out 16/17, though self-graded. |
| 2 | OVERNIGHT LEDGER | STRONG BACKUP | 5/10 | The closest fit to Pack D: two days of parked cars. The honest UNKNOWN is memorable. All real evidence comes from one edited reel, and it has no clock. |
| 3 | RELEASE DESK | STRONG BACKUP | 6/10 | Real legal pain, uses the full stack, and catches its own leaks on stage. But it is people-heavy, Pack D's people are unknown, and the car cut-out case is thin. |
| 4 | MIMIC | STRONG BACKUP | 5.5/10 | Judges drawing the query themselves is great theatre, and a fixed camera helps. But the evidence rests on one factory scene, no car U-turn was found, and it barely uses the provided stack. |
| 5 | UNCUT | STRONG BACKUP | 5/10 | Uses Embed1 and VastDB naturally, and the misinformation story is clear. All results are in-sample, and two near-identical static days risk AMBIGUOUS matches and missed cuts. |
| 6 | BLIND TIME | FOLD INTO ANOTHER | 4/10 | An excellent trust layer that runs on pure segment SQL. A fixed Pack D camera may never go blind, so on its own the demo is flat. |
| 7 | LANE CALL | STRONG BACKUP | 5/10 | Double-parked versus kerb-parked fits a street camera. But no real in-lane stop has ever been seen, the polygons are fragile, and Cosmos scored 0/7. |
| 8 | NIGHT EXIT | FOLD INTO ANOTHER | 4/10 | Solid crossing detection and good use of the stack. But it handles people only, there is no door, no clock and no matching story. Best as the 'who crossed since T' feature. |
| 9 | PINCH POINT REGISTER | FOLD INTO ANOTHER | 4/10 | The zone back-test is a useful alert-rule feature. Pack D likely has no crowds, and the P95 result is near-tautological. |
| 10 | PRIVATE VIEW | FOLD INTO ANOTHER | 4/10 | A strong privacy idea built on our own SIFT, not the provided services. Pack D has no known drift, so it would need a simulated knock. |
| 11 | LEAK TEST | STRONG BACKUP | 5/10 | A rigorous face auditor, but Pack D's people are probably small and distant. The plate pivot is untested. Fold it into Release Desk. |
| 12 | HIDE & SEEK | FOLD INTO ANOTHER | 3/10 | Fails by design on one static camera: segments become lookalike twins. No human has ever played it. |
| 13 | LAST UPRIGHT | FOLD INTO ANOTHER | 3/10 | Nobody will fall on Pack D, pose is not a provided service, and the one real fall sits right on a threshold. |
| 14 | BETWEEN THE LINES | FOLD INTO ANOTHER | 3/10 | There is no dialogue or story on a street camera, and it failed its own 60% accuracy kill test. |
| 15 | DAY 14 | SKIP | 2/10 | It refuses fixed cameras by design, so it has nothing to run on with Pack D. Reuse only its evidence packet. |

## 1. ON THE LINE: Police ask five questions in ten seconds, and the camera answers every one in milliseconds, with the evidence attached.

*STRONG BACKUP, win score 6/10. Folder: 08_*, hero: 08_on_the_line/out/person_bicycle_car_detection/hero.jpg*

You are a remote video response operator. An alarm fires, police are on the phone, and they always ask the same things: how many, which way did they go, what colour, are they still there, when did they arrive? Today you squint and scrub through footage while they wait. ON THE LINE listens to the call and follows the camera live, so each question pops a card on the next frame: "NO: the car was last seen 2.8 s ago, left via the LEFT edge", with the crop, the drawn path and a latency chip. When you hang up, the incident log is already drafted. It is a real job with a real pain, the demo is visceral, and Canary is the obvious way in. On the held-out clips it scored 16/17, and the slowest answer took 5.3 ms.

**Proof (measured):**
- person_bicycle_car_detection: 5/5 right, median 1.48 ms, slowest 5.29 ms; 'Is the car still there?' answered NO, last seen 2.8 s ago, left via the LEFT edge
- Held-out: one_by_one 5/5, car_detection 6/6 (5/6 on first run, two fixes made on that clip), people_detection 5/6 with the engine untouched
- No-future-data re-runs were identical for every question on all six clips; the slowest answer on any clip was 5.3 ms
- The 5 s segment-table SQL baseline got only 4 of 13 count/total questions right (e.g. 13 vs 4 vehicles) and cannot say direction at all
- rochester_night: 5/5, slowest 3.41 ms; live vehicle count within 1 of yolo26l on 92% of 12 frames

**What you see:** A dark dashboard. On the left, an overhead car park frame with a boxed white car and coloured paths, plus count timelines and an auto-drafted incident log. On the right, five police-question cards with crops, answers, green "matches eyeball" chips and ms latencies, under a "5/5 RIGHT" badge.

**The catch:** The core path is our own per-frame tracker, not the provided services. DataEngine's 5 s counts badly miscount totals, Cosmos only captions crops, and no VastDB tables have been built yet, so a stack-focused judge may mark it down. Pixel colour naming is shaky (black was read as purple or maroon). All 32 questions are self-written, the scores are self-graded, and the parser is regex.

**On the day:** Pack D is a fixed neighbourhood camera with mostly cars, which suits this, but there is no capture clock (use relative times) and no alarm, so we stage one. Planned work is about 10 h for two people: own YOLO11s+ByteTrack, Canary, VastDB entities/events, Cosmos3-Reason crops, UI. 10:30 go/no-go: can we get the Pack D stream at frame rate and track cars entering and leaving correctly? If not, this falls back to count-only answers.

## 2. OVERNIGHT LEDGER: Your car, every second of the night: there, gone, or "we couldn't see", with the reason. Never a guess.

*STRONG BACKUP, win score 5/10. Folder: 02_*, hero: 02_overnight_ledger/out/ptz_parking/hero.jpg*

You park on the street and walk away. Next morning the question is simple: was it there all night, and did anyone go near it? Today's answer is hours of scrubbing footage. You click one car and get a second-by-second strip instead: green for present, red for gone, grey for "can't see" with the cause named (lens covered, frozen feed, blurred, camera pointing elsewhere). Every grey gap gets a before/after crop that says SAME CAR or CHANGED, and every person whose feet came near the car shows up as a time-stamped thumbnail. It never fills a blind stretch with a guess. Pack D is two days of a neighbourhood camera that mostly sees cars, which is the overnight-parking case this ledger was built for. The honest UNKNOWN is the part a judge will remember.

**Proof (measured):**
- ptz_parking (real): auto-picked sedan with 235 s present, 0 absent and 81 s UNKNOWN (camera looking elsewhere); before/after SSIM 0.69, verdict SAME CAR
- sim_a + sim_b (SIMULATED faults on ptz_parking): 36/36 blind seconds caught across 5 faults, onset/offset error 0 s, cause correct 100%, 0 false alarms
- ptz_parking strip vs eyeball on 2 cars: 33/33 exact, 0 contradictions (2 rest on a 1 s dropout fill)
- Crop compare (ptz_parking calibration): a different car was flagged CHANGED in 110/110 pairs; same car, same view in 0% of 33
- 10 clean real clips (764 s): 15 s false alarms, all in retail_cctv; lisbon_parking slideshow correctly refused (NO LEDGER POSSIBLE)

**What you see:** A real car-park camera frame with one sedan boxed in green, a big green stamp reading THERE, SAME CAR, and a green strip broken by one grey 81 s UNKNOWN block. Below it: a red strip for a second car that left, a before/after crop pair, and a thumbnail of a person near the car.

**The catch:** It answers "same car, still there", not "damaged". Painted-scratch recall was 0% in the same view and 10.6% across the tour. All real evidence comes from one 5-minute edited daylight reel, and its only real ABSENT is a video jump cut. The thresholds were tuned on that same scene. On Pack D, a fixed camera may produce few blind stretches to show off.

**On the day:** About 8 h to rebuild. DataEngine gives counts per 5 s segment, not boxes, so we need a hosted YOLO that returns boxes, or our own YOLO. Use the peak_car drop in SQL as a "car left" pre-filter. Add Cosmos3-Reason on the before/after crops to check for damage and Cosmos-Embed1 to confirm the view is unchanged. With no capture clock, label times as clip time. 10:30 go/no-go: do we have per-frame boxes, and does one parked Pack D car stay put for 20 s or more?

## 3. RELEASE DESK: Release only the person who asked: everyone else is hidden by default, and every exported frame is checked before it goes out.

*STRONG BACKUP, win score 6/10. Folder: 01_*, hero: 01_release_desk/out/retail_cctv/hero.jpg*

You are a council information-governance officer. A resident wants every clip of themselves, and the law gives you one month. Redacting by hand takes about nine hours per hour of footage (Home Office figure). You type the request. SQL over the 5 s segment table narrows hours of footage to a list of candidates. You click the person, and the desk exports them sharp while everyone else is blurred. The exported file is then checked a second time, frame by frame, by detectors that played no part in making it. Any frame where a stranger overlaps the requester goes to a person for review. You end up with a release record: hashes, a leak result, a review list and a stopwatch. It wins because it solves a real legal problem, uses the whole provided stack, and shows the system catching its own mistakes on stage.

**Proof (measured):**
- retail_cctv: every one of the 346 export frames was leak-checked. Round 1 found 5 strangers the tracker had missed; 11 frames were patched and round 2 found 0.
- retail_cctv: 26.4 s of 110.9 s released for the auto-picked 294 px requester. 65 frames (7 intervals, 19%) went to human review. Strangers were detected in all 346 source frames.
- retail_cctv: 170 s of machine time end to end, against about 238 s from the scaled Home Office figure (scaled, not measured). The export step alone ran at 4.0x real time.
- Simulated fault injection (stranger pixels pasted back into the export): caught 12/12 on retail_cctv, 12/12 on lisbon_parking and 11/12 on person_bicycle_car_detection.
- lisbon_parking (black Mini): released as a clean car-shaped cut-out, 0 leaks in 61/61 frames, 14 s end to end vs 22 s scaled manual.

**What you see:** On the left, a busy shop as filmed, with 7 strangers boxed in red. On the right, the released frame: frosted blur except for the elderly woman in the beige coat. Underneath, a green banner reads "0 strangers detected in 346 / 346 export frames", next to tiles for 26.4 s released, 65 review frames and 170 s on the stopwatch.

**The catch:** The strong demo is shop CCTV crowded with people, and Pack D is a neighbourhood camera that shows mostly cars, with an unknown number of people. "0 leaks" only means the detectors found none; the gate (conf >= 0.35 plus a match in the source) was tuned on the test clip. The van case still releases the subject in two pieces. No VLM or ASR was actually used.

**On the day:** VastDB segment SQL finds the candidates. Hosted YOLO11s plus our own ByteTrack runs on those segments only. Cosmos3-Reason becomes the independent leak checker (untested), and Cosmos-Embed1 finds candidates from a text query. The core rebuild is about 7 h, with Embed and Canary about 1.5 h each on top, so it is tight. 10:30 go/no-go: does Pack D have one person or car at least 60 px tall with strangers passing near it, and do the hosted YOLO11s calls return per-frame boxes?

## 4. MIMIC: You remember how it moved, not what it looked like: draw the path and MIMIC finds who moved that way, or tells you plainly that nobody did.

*STRONG BACKUP, win score 5.5/10. Folder: 07_*, hero: 07_mimic/out/hero.jpg*

You're an investigator with two days of street-camera footage and one memory: "it came in, turned round and went back." You can't type that into a search box. With MIMIC you draw it. Sketch a U-turn on the canvas and about 0.1-0.2 s later you get eight cards. Each card shows the real trajectory drawn on the frame, green at the start and red at the end, along with a cost, a "turned 180 deg" chip and a clip to play. Draw something that never happened and you get a red NO MATCH banner, not a confident wrong answer. Judges get to grab the mouse and test it themselves, and that hands-on moment is what wins. Pack D is a fixed camera that sees mostly cars, which suits MIMIC: the camera-compensation step isn't needed, and a real car U-turn over 2 days is a plausible find. Cosmos-Embed1 can then add "white van" on top of the drawn path.

**Proof (measured):**
- worker_zone_detection: a drawn U-turn ranked real worker hairpins as the top 3 (cost 0.40 / 0.43 / 0.45, MATCH), all matched with the sketch rotated 180 deg. Hand-checked P@3 is 1.00 for u_turn, left_to_right and loiter and 0.67 for stop_and_go, with a mean of 0.92. All labels come from one rater.
- yonge_dundas hard negative: the same U-turn sketch on a 161-tracklet crowd returned best cost 1.11, so NO MATCH. Across all clips the U-turn sketch gave 1 MATCH, 2 MAYBE and 12 NO MATCH, and 6 clips had no data.
- All 21 clips (46 min): 753 tracklets, 309 searchable. The left_to_right best cost was 0.02 on yonge_dundas pedestrians and cyclists and 0.03 on rochester_night cars.
- draw.html (5.0 MB, single file) was rendered in headless Chrome with synthetic pointer events: 8 scenarios, 0 JS errors, 0.1-0.2 s per query. JS and python agree on the top 3 with a cost difference under 0.001.
- lisbon_parking: the camera-motion step correctly flagged the clip as a photo slideshow (2,177 cuts in 779 s) and MIMIC reports NO TRAJECTORIES rather than inventing matches.

**What you see:** On the left is a yellow U-turn sketch with a MATCH stamp at cost 0.40. In the centre, a hard-hat factory worker in an orange vest has his real hairpin trail drawn over the frame, green start and red arrow end. Below are runner-up cards tagged RIGHT or WRONG, and along the bottom a P@3 strip for each preset.

**The catch:** The U-turn result rests on one factory scene: about 6 tracklets of a few workers. No car U-turn was found in any test clip. On the day, DataEngine outputs counts every 5 s, which cannot give trajectories, so we must get per-frame YOLO boxes and run our own ByteTrack. It also barely uses Cosmos, ASR or the provided pipeline, which the head judge explicitly wants.

**On the day:** Rebuild takes about 5 h plus 1 h of slack. Swap in hosted YOLO11s per-frame boxes and our own ByteTrack, store the descriptor table in VastDB, and port the numpy kNN+DTW matcher and the drawing page. Camera compensation is probably skipped because Pack D is a fixed camera. Add Cosmos-Embed1 text filtering and Cosmos captions on the hits. The 10:30 go/no-go check: can we get per-frame box coordinates (not just counts) for Pack D, and does at least one car track in it show a hairpin or a stop-and-go?

## 5. UNCUT (cut down): "Only 50% of this clip is ours." UNCUT finds the exact source second of a viral clip and shows every second that was cut or spliced in.

*STRONG BACKUP, win score 5/10. Folder: 10_*, hero: 10_uncut/out/t4_splice_ptz_sitdown/hero.jpg*

You run comms for a council, and a clip "from your street cam" is all over social media. It has been cropped, sped up and cut, and the comments are getting angry. You drop it into UNCUT. Seconds later you see a stamp: MATCH, the source camera, the exact window, "CROPPED 80%" and "3.0 s REMOVED @ clip 00:07". A ribbon shows the missing seconds in red. When someone splices in footage from elsewhere, the amber PARTIAL stamp says "Only 50% of this clip is ours". UNCUT never guesses: Cosmos-Embed1 vectors in VastDB narrow down the candidates, then frame hashes are aligned and every number is measured. Judges see real camera footage used to fight misinformation, with numbers anyone can check.

**Proof (measured):**
- Kill test passed: 5/5 verdicts right, 4/4 located offsets with max error 0.05 s (t1-t5)
- t3_rochester_speed_cut: measured speed 1.25x and 3.05 s removed at clip 6.0 s (truth 3.0 s at 6.0 s); a reviewer re-ran it and got the same result
- t4_splice_ptz_sitdown: PARTIAL, 50% explained, foreign span 5.0-10.0 s, 0.00 s start/end error
- x13_dashcam_916_caption (9:16 crop with burned-in caption and @handle) and x14_retail_tiktok_bars (letterboxed) both matched with 0.00 s error
- Hash distance: true matches 0.004-0.05, best non-matches 0.25-0.34, threshold 0.18. Stage 1 ranked the true source video first for 14/14 clips

**What you see:** Side by side: the viral clip and our fixed PTZ car-park archive frame (ID:001 overlay), identical. An amber "PARTIAL: SPLICED, NOT ALL OURS" stamp, the headline "Only 50% of this clip is ours", and a ribbon that is half green and half red ("no source in archive").

**The catch:** Every result is in-sample: the edits were simulated with ffmpeg and the thresholds were tuned on the same clips. Pack D is one mostly static camera recorded over 2 days. Cut detection on near-static footage failed in a synthetic test, and repeated near-identical days could trigger the AMBIGUOUS path, which no real clip has ever exercised. It does not use YOLO, Cosmos3-Reason or Canary.

**On the day:** About 6 h to rebuild. DataEngine segments and Cosmos-Embed1 vectors in VastDB replace the local index; we add our own ffmpeg frame_hashes table; align.py and render.py port unchanged. Demo clips get re-edited from Pack D. 10:30 go/no-go: hash 2 days of Pack D, then check that a cut 15 s clip resolves to the right day and second with no AMBIGUOUS result on near-identical frames.

## 6. BLIND TIME: "Nothing found" is a lie if the camera wasn't looking. Blind Time says COULD NOT SEE.

*FOLD INTO ANOTHER, win score 4/10. Folder: 06_*, hero: 06_blind_time/out/ptz_parking/hero.jpg*

You're the investigator asking the footage "was anyone in the car park between 02:40 and 04:00?" Today's search says NOTHING FOUND, and you write it in your report. That's wrong: the PTZ swung away at 02:35 and never came back, so it was blind for all 80 seconds. Blind Time puts a health verdict on every second (OK, DARK, FROZEN, BLUR, MOVED, OCCLUDED) and answers in four states: SEEN, SEEN IN OTHER VIEW, NOTHING FOUND, or COULD NOT SEE, with the coverage % and the blind spans. It even catches a frozen feed where plain search says SEEN from a stale frame. Every verdict is plain WHERE/COUNT/SUM SQL next to the DataEngine segment table in VastDB. Fold it in as the trust layer under the winning search/query project. It answers the judge's real-world-impact question for the whole stack: "we never say nothing happened unless we could see."

**Proof (measured):**
- ptz_parking (real): person 02:40-04:00, plain search NOTHING FOUND vs Blind Time COULD NOT SEE, blind 80 s of 80 s; health states match frame-by-frame eyeball labels on 316 of 316 s
- rochester_night (real): camera pans away at ~22 s; person 00:10-01:00 NOTHING FOUND became COULD NOT SEE (coverage 24%); 84/84 s agree with eyeball, and 0 seconds were wrongly called DARK on the night scene
- ptz_parking 30 s battery: 3 of 4 plain NOTHING FOUND answers were actually blind
- Held-out SIMULATED faults (store_aisle + yonge_dundas, thresholds frozen): DARK/FROZEN/BLUR/MOVED pooled 46/46 s recall, 0 false alarms on 26 + 19 healthy s; OCCLUDED only 5/10
- SQL check: all 12 default queries give the same answer as the Python implementation using only WHERE/COUNT/SUM (fits the VastDB dialect)

**What you see:** A dark dashboard. A big orange COULD NOT SEE stamp sits beside a struck-through NOTHING FOUND marked MISLEADING. Thumbnails show the car-park home view next to a tree-and-pole view. Below is a per-second health bar, green then amber from 02:35.

**The catch:** Pack D is a fixed neighbourhood camera. It may never freeze, move or go dark during the demo window, and then every verdict is a boring SEEN or NOTHING FOUND. The strongest demo needs a PTZ or injected faults. It also leaves out Cosmos-Reason and Cosmos-Embed, so on its own it shows off little of the stack the judge asked for.

**On the day:** Hits come from the DataEngine YOLO11s 5 s segment counts, so blind spans get rounded to 5 s. Health rows are a new sidecar that writes to VastDB. Cosmos-Reason can caption the blind frame as an optional extra. Rebuild takes about 5 h (+1 h for PTZ presets). At 10:30, run the health extractor over Pack D: are there any non-OK spans (night, freeze, occlusion)? If there are none, fold it in or use the "pull the cable" fault button.

## 7. LANE CALL: Every stopped car makes an alert. Lane Call tells you in half a second which ones are actually blocking the road.

*STRONG BACKUP, win score 5/10. Folder: 14_*, hero: 14_lane_call/out/motorway_a40_inject/hero.jpg*

You run a traffic control room. Every parked car, every queue at a red light and every car seen again by another camera preset raises its own alert, and the one that matters is buried among them. National Highways checked only 564 of 3,850 alerts. Lane Call puts a triage step in front of you. Each stopped vehicle gets a white ground-contact line traced from its segmentation mask and a call: IN LANE, AT KERB or CAN'T SEE, with a stated reason. Cars blocking a lane go to the top. Kerb-parked cars are lowered automatically. A stack of cars at a red light becomes one QUEUE card. Each card comes with an 8 s clip in about half a second. When the ground contact is cut off by the frame edge, the system says so instead of guessing. It is a real operator workflow, and judges get a clear before-and-after: 104 alerts become 100 cards, with the in-lane ones first.

**Proof (measured):**
- motorway_a40_inject: the real detector flagged all 3 pasted stops at 5.0 s. Two were called IN LANE 97% and went first; the hard-shoulder car was AT KERB 84%. The same clip with nothing pasted gave 0 alerts.
- ptz_parking: 87 raw alerts became 61 unique cars after merging 26 repeats across presets. 46 were lowered automatically as AT KERB, 15 were CAN'T SEE and 0 were IN LANE. Eyeball check: 59/61 right.
- yonge_dundas: 7 cars stopped at a red light became 3 cards (1 QUEUE of 5, 1 IN LANE, 1 CAN'T SEE). Eyeball check: 7/7 right.
- Across all 5 clips: 100/104 = 96.2% right by eyeball (the always-AT-KERB baseline scores 67%). IN LANE precision 12/13, recall 12/12, 0 harmful misses.
- Alert to card: median 530 ms, p95 770 ms, max 938 ms including the segmentation pass, against a 2 s budget. The duckdb SQL queue order matched the Python order on every video.

**What you see:** A real A40 motorway frame stamped SIMULATED STOPS. The running lanes are shaded red and the shoulder blue. Two pasted cars are boxed IN LANE 97% and one AT KERB 84%. On the right is the queue with crops, and a control panel reads "0 alerts".

**The catch:** No footage, Pack D included as far as we know, has shown a real vehicle stopped in a lane. The 12 IN LANE truths are 4 slideshow stills, 6 cars at a red light and 2 pasted cars. The angled-car call changes with 3 px of a hand-drawn polygon border. Local Cosmos scored 0/7 on IN LANE. The polygons and labels were all drawn by the builder.

**On the day:** On a neighbourhood camera the story changes from motorway to double-parked or blocking versus kerb-parked. About 8 h to rebuild: polygon tool, detector, seg line, call, queue UI and the VastDB alerts table. Add 2-3 h if DataEngine YOLO11s returns only counts per 5 s segment. Go/no-go at 10:30: does the endpoint return per-frame boxes, and does Pack D show any car stopped in the roadway?

## 8. NIGHT EXIT: "Last seen 00:12" becomes a ranked list of who went out the door after that, with direction and proof.

*FOLD INTO ANOTHER, win score 4/10. Folder: 04_*, hero: 04_night_exit/out/one_by_one_person_detection/hero.jpg*

You're the night carer. A resident is missing, last seen 00:12, wearing "grey top". You don't have time to scrub hours of footage. You type the time and the description, and NIGHT EXIT gives you a card for every person who crossed the exit line after 00:12: OUT or IN, the time, a sharp crop, a zoomed frame with the crossing dot on the line, colours, a Cosmos one-liner, and a rank against your words. When three people really do wear grey, it says "best of 3 close matches, not a sure thing" instead of pretending to be sure. When someone steps out and comes straight back, it says "back in 1 s later", so nobody raises a false alarm. Every part of the stack has a job: VastDB shortlists the segments, Cosmos describes each person and Embed1 ranks them. Best used as the "who crossed the line since T?" feature of the main search idea.

**Proof (measured):**
- Crossings: 23 of 23 eyeballed found with the right direction, 0 extra, 0 missed (one_by_one 12/12, rochester_night 1/1, retail_cctv 10/10, where the retail ground truth is partly circular)
- one_by_one_person_detection, since 00:21 "light grey shirt and blue jeans": WENT OUT 02:15, rank 73% vs 56% for the next person, evidence 64%
- Ranking: 7 of 7 pre-registered queries put the right person first (6 of 7 on the first run; 'white coat' on retail_cctv was rank 2 of 9 until a later garment-word fix)
- one_by_one, "blonde woman in a black cardigan": colour alone ranks the right person 2nd; adding the Cosmos description and Embed1 moves her to 1st
- retail_cctv, since 00:30 "blue coat and pink top": WENT OUT 00:45, back in 1 s later, 65% vs 29%; VastDB-style SQL shortlist kept every crossing in its window

**What you see:** Dark dashboard: query pill "since 00:21 | light grey shirt and blue jeans", a big green "WENT OUT at 02:15" under the exit-line frame, four ranked OUT cards each with a crop and a zoomed crossing frame (73% best match), and a timeline below.

**The catch:** Pack D is a neighbourhood street camera that sees mostly cars, the number of people is unknown, there is no capture clock and there is no door. The prototype only handles people, its care-home story has no matching footage, and DataEngine gives counts rather than boxes, so the core still runs on our own tracker. The night test was one blurry 90 px pedestrian.

**On the day:** Swap duckdb for VastDB SQL, run Cosmos3-Reason on each crop, and compare Embed1 crop clips against the query text. Run our own detector and ByteTrack, since DataEngine gives counts only. Draw a kerb or gate line on Pack D and add vehicles. Rebuild takes about 6 h. At 10:30, decide go or no-go: does Pack D show at least a few people crossing a drawable line in a 5-minute sample? If not, keep it as a feature only.

## 9. PINCH POINT REGISTER: Draw a zone, replay the whole recording, and get a crowd trigger you can defend because the footage already tested it.

*FOLD INTO ANOTHER, win score 4/10. Folder: 13_*, hero: 13_pinch_point/out/retail_cctv/hero.jpg*

You manage a busy space and the regulator's finding stings: your pinch points have "no recorded, risk-assessed plan." You draw an amber polygon on the camera view. Pinch Point Register counts every person whose feet land inside it, second by second, and replays the whole recording against your rule. Nobody guesses N. On a real shop-till CCTV clip, N>=9 for 3 s would have fired twice, N>=10 never, and N>=6 fires 9 times. You can watch the threshold trade-off happen. It then exports a plan entry with the polygon, trigger, back-test, three evidence frames and blank action rows for the owner to fill in. The real deliverable is an auditable rule. On Pack D, fold this zone back-test and threshold sweep into the main neighbourhood-camera project as its alert-rule feature, counting cars or people in a zone.

**Proof (measured):**
- retail_cctv (lead, 111 s fixed ceiling CCTV): peak 10 in zone; the P95 rule gives N=9, and the literal >=9-for-3 s rule fires 2x (00:23-00:27, 00:48-00:51), 12 s over the line; N>=10 never fires
- retail_cctv sweep: N>=6 fires 9x / 68 s, N>=8 5x / 25 s, N>=9 2x / 12 s, so fire counts are not monotone in N and the sweep belongs in the plan
- Kill test on retail_cctv peak frames: detector vs hand count +20.0%, +11.1%, +5.9%; over all 8 hand-counted frames, 7/8 are within 20% (mean abs error 11.0%)
- store_aisle_detection: peak 6, N=5 fires 3x with a 15 s longest episode (26 s, 39% over); people_detection sparse control fires once for 4 s
- duckdb SQL check (COUNT/MAX with WHERE only, VastDB-safe dialect) matches the Python result; test_smoke.py 8/8 pass in about 10 s; run.py --all takes 30-60 s warm

**What you see:** A dark dashboard. The ceiling-CCTV till queue is tinted amber with people boxed green and "10 IN ZONE", and everything outside is greyed out. A red "WOULD HAVE FIRED 2x" stamp sits above KPI cards, a threshold-sweep table and a per-second timeline with two red fire bands.

**The catch:** Pack D is a fixed neighbourhood camera showing mostly cars, and it is unknown whether it has any crowds, so there may be no pinch point to register. The documented DataEngine table gives counts per 5 s segment, not boxes, so a polygon zone may need our own YOLO pass. It does not use Embed1 or Canary, and the P95 'would have fired' result is near-tautological.

**On the day:** The local YOLO26s+ByteTrack stand-in becomes DataEngine YOLO11s (if bboxes exist, otherwise our YOLO on keyframes), duckdb becomes VastDB, and Cosmos3-Edge becomes Cosmos3-Reason labelling the peak frame. README estimates about 6 h to rebuild. 10:30 go/no-go: does DataEngine output carry bboxes, and does Pack D show at least one zone with sustained clusters of people or cars? If neither, cut it to a feature.

## 10. PRIVATE VIEW: The camera moved, the privacy masks didn't, and nobody noticed. PRIVATE VIEW notices to the second.

*FOLD INTO ANOTHER, win score 4/10. Folder: 15_*, hero: 15_private_view/out/rochester_real/hero.jpg*

You run a control room that watches cameras pointed at people's homes. Static privacy masks hide their windows. Then a van clips the pole or the mount sags. The image shifts, the masks stay where they were drawn, and bedroom windows are on screen with nobody aware of it. PRIVATE VIEW compares every frame with the setup frame and moves each window outline to where the window really is now. It shows red where the mask still sits, green where the window is and yellow where the window is exposed, and logs a per-window breach with start and end times. On real hand-held night footage it caught 8 windows 21-27% uncovered, and a second independent estimator agreed to 0.22 px. It is a strong privacy-integrity layer. Fold it into the winning Pack D pipeline as its camera-health and privacy feature.

**Proof (measured):**
- rochester_night raw-clip knock (SIMULATED, truth = knock x real drift from independent ECC): 9/9 windows pass, start/end error 0.00 s, event-peak error <= 0.3 pp, 0 false alarms
- rochester_real (REAL drift, no injection): 8 windows exposed 21-27% from 00:15.8; ECC cross-check agrees to 0.22 px and 0.06 pp on average, 17/17 events matched
- rochester_night stabilised knock 8-18 s (SIMULATED, partly circular): 7/7 exposed windows detected, 0.00 s timing error, peak error <= 2.4 pp
- ptz_parking (REAL PTZ, stand-in zones): 4 eyeballed roof landmarks at 280 s were 1.9-3.3 px off after a 541 px camera move; the 80 s unrecognised view is flagged UNVERIFIED, not marked safe
- Cosmos3-Edge grounding hit 0 of 9 windows; Cosmos-Embed1 caught the PTZ zoom (cosine 0.11-0.56) but missed the pan (0.92-0.94)

**What you see:** A night street of flats, shown twice. Left is SETUP, with 9 windows under green-on-red masks. Right is NOW: the red masks have slid off the windows, yellow slivers show the exposed glass, and a BREACH stamp reads "7 windows exposed, worst W7 37%". Below is a per-window exposure timeline.

**The catch:** Pack D is a fixed neighbourhood camera with mostly cars and no known knock or drift, so the core demo would have to be a simulated knock. The engine is our own SIFT geometry, not a provided service, and Cosmos grounding failed (0/9). None of the footage has a real residential breach. Judges who want the VAST stack used would see little of it.

**On the day:** Rebuild in about 6 h: operator polygon setup, the SIFT/RANSAC homography and exposure maths, a VastDB seg_exposure table with OR-chain alert SQL, Cosmos3-Reason breach captions, and Cosmos-Embed1 as a view-change screen. As a folded feature, budget about 3 h. 10:30 check: does Pack D drift measurably across its 2 days? If it doesn't and Cosmos3-Reason can't ground windows, ship it only as a simulated-knock sidebar.

## 11. LEAK TEST: It looked blurred, but 178 of 262 frames showed a readable face. Leak Test catches what playback speed hides.

*STRONG BACKUP, win score 5/10. Folder: 03_*, hero: 03_leak_test/out/retail_cctv/hero.jpg*

You are the disclosure officer at a council or police force. You blur a CCTV clip, watch it once at full speed, and release it. It looked fine. Leak Test plays the motivated intruder. It takes your export and the original and checks every frame for a face that is still as sharp as in the original. Then it issues a signed PASS/FAIL certificate with SHA-256 hashes of both files. You get four columns with red boxes on the exposed faces, a timeline where a keyframe-only redactor shows up as a red comb, and a leak reel of cropped faces with timestamps. The auditor deliberately uses different models from the redactor it checks. It does not trust the redactor's log. Judges get a privacy-tech story with real consequences and a jaw-drop hero they can read in 5 seconds.

**Proof (measured):**
- retail_cctv, export A (keyframe-only): FAIL. 178 of 262 frames (68 %) and 13.6 s show readable faces, and the auditor caught 625 of 692 exposed faces (90.3 %).
- Kill test passed across the clips: 93.8 % frame recall on A's leaks (1258/1341), 99.1 % person-level recall (109/110), and 90.9 % recall on an independent eyeball sample (30/33, 95 % interval 76-97 %).
- Specificity: 0 false flags out of 2008 faces audited on the full-redaction export C (0 of 742 frames flagged), and C gets a PASS on all 3 clips.
- retail_cctv, tight export D: FAIL on 2 frames. It caught a boy's head that the tight box missed at 00:10.6. That is a real leak a reviewer would not see at playback speed.
- Warm re-run of all 3 clips takes about 35 s, and test_smoke.py's 135 checks all pass.

**What you see:** A dark dashboard of a high-angle shop camera in four columns: A, B and D are stamped red FAIL with inset crops of readable faces, and C is stamped green PASS. Below are red comb timelines and three rows of leak-reel face crops with timestamps.

**The catch:** It is a face auditor, and Pack D is a car-heavy neighbourhood camera with an unknown number of people, who are probably small and distant. Faces under 16 px are not flagged by policy, and recall at 16-20 px is only 70 %. Precision is 75 %, so flags need a human look. The thresholds were tuned on these same 3 clips, and the independent recall check rests on just 33 Claude-labelled faces.

**On the day:** Rebuild is about 6 h. YOLO11s from DataEngine becomes the redactor under test, Cosmos3-Reason judges whether a face is identifiable in person crops, the leak ledger moves into VastDB SQL, and Cosmos-Embed1 clusters identities. The 10:30 go/no-go: in Pack D, are there persons with faces at least 20 px wide in enough segments? If not, pivot the target to licence plates (untested) or fold it into another idea.

## 12. HIDE & SEEK: Think of any clip and our video index finds it in about 6 questions, then shows you the clips it cannot tell apart.

*FOLD INTO ANOTHER, win score 3/10. Folder: 12_*, hero: 12_hide_and_seek/out/ALL/hero.jpg*

You run a video archive and trust your index, but you never check whether it can tell two clips apart. A judge picks a secret clip from a 64-tile grid. The page asks yes/no questions chosen by information gain ("Is it outdoor?", "Does a truck or bus appear?"), and the grid halves each time: 64, 32, 16, 8. One tile glows: "Is it this one?" If the judge lies once, the page notices the contradiction and still finds the clip. Then comes the honest part, the lookalike report: these clips are identical to your index, and these the detectors cannot see at all. It is a live stage moment and a real audit, and every question is a SQL COUNT over YOLO, Cosmos tags and caption keywords. Use it as the "audit your index" feature of the main search project.

**Proof (measured):**
- 64-clip archive across 21 test videos: strict engine finds 60/64 (93.8%) in 5.97 mean questions, worst 7. Kill test PASSED; v1 failed at 81.3%.
- Page protocol (questions + 'Is it this one?' + recovery) finds 64/64 at 6.97 steps. SIMULATED 5% wrong answers: 99.9% found, vs 69% for strict elimination.
- Ablation: YOLO tracks alone 57.8% found, + brightness 68.8%, + Cosmos tags 76.6%, + Cosmos caption keywords 93.8%. The provided services earn the result.
- Real twins: motorway_a40 0/2 findable and ptz_parking 4/6 (c13/c14 are the same static lot). 4 clips are invisible to detectors (Lisbon title card, credits, empty asphalt).
- Caption keyword 'no' answers are unreliable: recall is 0.15 for 'man' and 0.25 for 'woman', while precision is 1.0.

**What you see:** A dark dashboard. On the left, clip c22 (Yonge-Dundas) found in 6 questions, with its YES/NO trail halving 64>32>16>8>4>2>1. On the right, a 64-thumbnail grid with red twin borders and dashed cyan "invisible" tiles, a green KILL TEST PASSED stamp, and four stat tiles.

**The catch:** Pack D is one fixed neighbourhood camera, mostly cars, over 2 days. That is exactly the case where this fails: same-view clips became indistinguishable twins (motorway_a40 0/2, ptz_parking c13/c14). On Pack D most segments would be lookalikes, and the game falls apart. Also, no human has ever played it, and the 93.8% comes after a post-hoc fix on the same 64 clips.

**On the day:** YOLO11s gives counts only, so the motion and "stationary" questions are lost. Cosmos3-Reason captions feed LOWER(caption) LIKE keyword counts in VastDB. Rebuilding takes about 6 h, so as a feature budget about 2-3 h for the lookalike/invisible audit. Go/no-go at 10:30: run the twin count over 64 Pack D segments. If more than about 25% are twins, ship only the audit report.

## 13. LAST UPRIGHT: Not just "someone fell": when they were last standing, how fast they went down, and how long they've been on the floor.

*FOLD INTO ANOTHER, win score 3/10. Folder: 11_*, hero: 11_last_upright/out/skate_fall/hero.jpg*

You're on a night shift with 30 residents, and a camera alarm just said "FALL". Your first questions are when it happened and how long they've been down. LAST UPRIGHT answers both on a single card. It shows the last-upright frame, the ground-contact frame and the flat frame side by side, a stand-to-flat time that separates a fall from someone deliberately lying down, and a long-lie clock that moves through CHECK NOW, ESCALATE and LONG LIE. Cosmos checks the key frame, so a resident lying in bed doesn't raise a false alarm, which the box-only baseline does. VastDB filters the clips so Cosmos only looks at flagged segments. On Pack D, use it as the "person down" alert inside the winning street-watch project, not as the main project.

**Proof (measured):**
- skate_fall: correct FALL verdict. Ground contact at 6.96 s vs 6.9 s in the ground truth (+0.06 s), flat at 7.28 s vs 7.25 s (+0.03 s), 0.88 s from standing to flat = FAST. The box-only baseline fires 0.42 s before he touches the ground.
- assist_elderly: Cosmos answers 'bed', so it shows LYING IN BED and starts no floor clock. The box-only baseline calls the same clip a FALL.
- Kill test passed: all 5 Pexels clips labelled correctly, 0 falls on classroom (33 s, 5 people). Cosmos answered the floor question correctly on 6 of 6 clips.
- Unseen negative one_by_one_person_detection (139 s, 6 tracks): 0 DOWN events, 0 falls.
- VastDB funnel: only 1 of 3 segments on skate_fall and 0 of 7 on classroom need a Cosmos call. In the sensitivity run, 13 of 15 single-threshold changes keep 5/5.

**What you see:** A red FALL stamp and "0.9 s upright to flat". Three skeleton frames: legs at 00:06.4, hand and hip on the ground at 00:07.0, flat at 00:07.3. Under them, a coloured posture strip and cards for 1.0 s on the floor, the long-lie ladder, Cosmos "YES" and the box-only baseline.

**The catch:** Pack D is a neighbourhood camera that mostly shows cars, and we don't know whether it shows people. Nobody will fall on it, and it isn't a care home. Pose isn't a provided service. The box-only baseline scores 4 of 5 as well, and thresholds were tuned in-sample. The single real fall sits 0.08 s from a threshold that flips the verdict, and the long-lie clock has never run on a real long lie (it's simulated).

**On the day:** Swap in YOLO11-pose that we run ourselves, plus DataEngine counts as the funnel, VastDB tables and Cosmos3-Reason checking the key frame. A full rebuild takes about 6 h. As a "person down" feature inside another project it's roughly 2-3 h. Go/no-go at 10:30: does Pack D show pedestrians large enough for pose keypoints? If not, drop it.

## 14. BETWEEN THE LINES: The AI describer that knows when to shut up: it drafts audio description only in the pauses, and every line shows what the machine could check.

*FOLD INTO ANOTHER, win score 3/10. Folder: 09_*, hero: 09_between_the_lines/out/elephants_dream/hero.jpg*

You're an audio-description writer facing hours of programme: find every pause, write a line that fits, and never talk over a character. Between The Lines hands you a first draft. Red bars mark speech, coloured blocks sit only in the gaps, and each line is at most 8 words, timed against real TTS, with GREEN, UNVERIFIED or AMBER showing what YOLO could check. Press play and a voice speaks in the silences while the soundtrack drops about 15 dB and comes back. Ofcom estimated about 61,654 audio-described UK broadcast hours a year in 2010, so this is real-world impact judges feel, and on the day Canary, YOLO11s and Cosmos3-Reason fill exactly the slots. On Pack D, a silent-ish street camera, keep its gap-fitting, ducking narrator as the voice layer of the main project.

**Proof (measured):**
- elephants_dream (first 180 s): 10 lines in 10 dialogue gaps (55.8 s usable), 0.0 s over detected speech; by eye 4 accurate / 6 partly / 0 wrong; title card voiced exactly
- dashcam_urban (120 s): 11 lines in 12 gaps (75.9 s usable), programme ducked 15.7 dB mean; left/right check flagged 'black car on the right' AMBER (YOLO puts it centre-left)
- yonge_dundas (57 s): 5 lines in 5 gaps, 3 accurate / 2 partly / 0 wrong, 0.0 s over detected speech
- All 3 clips: 26 lines, 12 fully accurate (46%, up from 28%), 25 of 26 accurate or usable with an edit, 1 wrong
- SIMULATED speech injection (93 held-out TTS utterances in the real beds at -3/0/+6 dB): coverage 0.991-1.0, 0 missed, 0.0 s of gap over injected speech

**What you see:** A dark dashboard for Elephants Dream: a waveform striped red where speech was detected, coloured description blocks only in the gaps, a green "0 s over detected speech" stamp, and three keyframe cards with drafted lines, flags and by-eye verdicts.

**The catch:** Pack D is a fixed neighbourhood camera with no dialogue to work around and no story to describe, so the core idea has nothing to bite on. The strict kill test also failed: only 46% of lines were fully accurate against a 60% bar. GREEN is not proof (1 of the 12 GREEN lines was wrong), and the speech check runs on its own self-trained detector.

**On the day:** Canary word timestamps replace the self-trained speech detector. YOLO11s counts come from VastDB SQL, and Cosmos3-Reason writes the captions. The README estimates about 6 h to rebuild, and the ducking mix, VTT, player and fit loop port unchanged. 10:30 go/no-go: does Pack D have audio with speech or narration worth describing? If not, fold the narrator into the main project.

## 15. DAY 14: dashcam in, Operation Snap packet out: Two minutes of dashcam in, five close calls ranked, and a 14-day police clock that reads its date straight from the footage.

*SKIP, win score 2/10. Folder: 05_*, hero: 05_day_14/out/dashcam_urban/hero.jpg*

You are a driver who just got cut up. You have hours of dashcam footage and 14 days to send it to the police. One force got to 65% of Snap reports too late to act on them. You drop the file in. Nobody scrubs through it. You get a risk curve over the whole recording, the five closest moments ranked, a 10 s clip of each, a zoomed key frame, and a draft packet with a SHA-256 of the original. The clock is read from the footage's burned-in date, and it goes red when you are too late. It knows its limits: it stamps REVIEW, never "offence found", and it refuses footage that is not from a dashcam. That honesty is its strongest selling point. The problem is the event footage.

**Proof (measured):**
- dashcam_urban (120 s): 5 candidates with packets in 21 s warm (74 s cold). Near-field precision@5 = 5/5, but only 1/5 is reportable, and the real silver BMW at 00:48.4 ranks #2.
- dashcam_urban: overlay date read as 2013-06-18 with 3/3 agreeing reads, start time 19:40:09. The clock correctly shows 'PASSED 4,855 days ago'.
- motorway_a40 and person_bicycle_car_detection: both detected as fixed cameras, stamped NOT A DASHCAM, no packet produced.
- ctrl_highway (20 s unmodified control): 0 candidates (max severity 0.11 below the 0.15 floor). sim_cutin (SIMULATED): cut-in found at 9.27 s and ranked #1, severity 0.69, TTC proxy 1.0 s.
- Local Cosmos stand-in parked/moving triage was no better than chance (said PARKED for 7/8, then DRIVING for 8/8). Colour right for only 2/5 candidates, so triage is now off by default.

**What you see:** A dashcam frame in a car park: a dark estate boxed red as '#1 CLOSING FAST', an amber REVIEW stamp, a zoom crop, five numbered thumbnails, a blue risk curve with five ticks, and a fully red 14-day clock bar reading PASSED 4,855 days ago.

**The catch:** Pack D is a fixed neighbourhood camera, and this tool is built to refuse fixed cameras ('NOT A DASHCAM'), so it has nothing to run on with the event footage. Even on dashcam footage the #1 pick is a parked car (reportable 1/5). It also needs per-frame boxes, but DataEngine only provides 5 s counts. It does not use Cosmos-Embed1, VastDB or Canary.

**On the day:** The README estimates about 8 h to rebuild: run our own YOLO for boxes (DataEngine gives counts only), and test Cosmos3-Reason on clips for 'is it moving'. On Pack D it is a non-starter, because there is no dashcam footage. 10:30 go/no-go: only revisit it if organisers supply dashcam footage. Otherwise drop it and reuse just the evidence packet (clip, SHA-256, clock) in the main project.
