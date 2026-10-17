# 12 Hide & Seek: can our index tell our clips apart?

## What it is
A 64-clip archive (5 s segments sampled across the 21 test videos), each clip described by YOLO tracks, frame brightness and Cosmos (indoor/outdoor, setting, caption keywords). A 20-questions engine asks the yes/no question with the most information gain, then asks "Is it this one?", and recovers if an answer or the guess was wrong. Played on a stage (judge thinks of a clip) it is a game; played with every clip as the secret it is an audit that lists the lookalikes the index cannot separate and the clips the detectors cannot see.

Run: `python run.py --all` (about 85 s warm; a cold build takes minutes because Cosmos calls are slow when the box is shared). `python run.py --video lisbon_parking` re-renders one video's report and hero in about 4 s (it skips the corpus-wide sensitivity studies). `python test_smoke.py` (about 8 s). Open `out/hide_and_seek.html` through any static server (clips play from `out/clips/`, keep the folder together). Deep links: `#auto=c22` auto-plays that secret, `#look` opens the lookalike report.

## Demo script (90 s)
1. (0:00) Open `out/hide_and_seek.html`. "Pick any clip in your head from this grid, do not tell me." Hovering a tile shows what the index stores (chips + Cosmos caption), so the judge can answer the way the index would.
2. (0:10) Ask the questions the page shows; judge clicks YES / NO / "Not sure". The grid dims as candidates shrink (64 -> 32 -> 16 ...). About 6 questions later one tile glows and the page asks "Is it this one?" with the clip playing. Judge says yes. "Found it."
3. (0:50) Make one deliberate wrong answer, or say NO to the guess: the page says "one answer seems to disagree with the index, single wrong answers are now forgiven", keeps asking, and finds the clip anyway (shaded tiles = candidates one answer away). Then Auto-play with a Random secret to show it is not a trick.
4. (1:10) "Lookalike report": "now the honest part: these 4 clips (2 pairs) are identical to the index, we ask 'which of these is yours?'; these 4 are invisible to detection." Show `out/ALL/hero.jpg`.

## How it maps to the VAST stack on the day
| Here (local stand-in) | On the day |
|---|---|
| YOLO26s + ByteTrack counts per 5 s (`hs_core.track_features`) | DataEngine YOLO11s per-segment counts (persons, cars, bicycle, ...) in VastDB. Peak simultaneous counts, motion direction and "stationary" need per-frame boxes or tracks, not only counts; with counts only, person/vehicle/bike/furniture/bags/animal/traffic attributes work, motion and stationary do not. |
| frame brightness / camera-moving by optical flow | Compute from the segment frames in our own code (cheap, no service). |
| local Cosmos3-Edge, one frame, regex-parsed JSON (indoor/outdoor, setting, caption) | Cosmos3-Reason caption + two tagged fields per segment. The caption keyword lexicon (`hs_core.KEYWORDS`) is matched against that caption; with the real service a keyword question becomes a `LOWER(caption) LIKE '%tree%'` count. |
| duckdb attribute table + Python/JS engine | VastDB table, one SQL `COUNT` per candidate question over the remaining candidates (use `OR` chains and `LOWER(...) LIKE`, no `IN`/`GROUP BY`/`JOIN`). The engine itself is about 100 lines (`engine.js`) and stays client side. |
| Cosmos-Embed1 | not used. Natural extension: embedding distance to break attribute twins (the remaining c13/c14 and c25/c26 twins are genuinely the same camera view, so it would not help there). |

## Results on test footage
64 clips, seed 12, stratified random sample (about sqrt of the segment count per video, max 8, min 1, one random segment per time bin; no cherry picking). 49 candidate questions. Every clip played once as the secret.

What changed after the review (first version: 81.3% found, kill test FAILED): dominant-colour questions removed (a judge cannot answer them like the index; the colour is still measured and shown in the report), count questions reworded to "At any moment, ...", 17 Cosmos caption-keyword questions added (a fixed lexicon of visible things; a word becomes a question only if 3 to 61 of the 64 captions mention it), "Is it this one?" confirmation plus soft recovery added to the page, hover cards added.

| metric | value |
|---|---|
| **kill test engine** (strict: truthful answers, hard elimination, no guess) found uniquely | **60/64 = 93.8%** (was 81.3% in v1) |
| mean questions of that engine | **5.97** (median 6, worst 7; 5 Q: 6 clips, 6 Q: 54, 7 Q: 4; floor for these group sizes 5.94) |
| **page protocol** (questions + "Is it this one?" + recovery), truthful judge | **100% found**, 5.97 questions + 1.00 guess = 6.97 per game; 6% of games end in a "which of these two is yours?" pick |
| attribute twins left | 4 clips in 2 pairs: c13/c14 (ptz_parking, same static lot), c25/c26 (motorway_a40, same highway view). Both are real twins by eye. |
| invisible to detectors (zero confirmed tracks) | 4: c05 Lisbon title card, c38 Elephants Dream credits (dark), c44 and c58 empty asphalt |

Which pipeline layer buys discrimination (strict engine, same logic, fewer attributes):

| attributes | questions | found uniquely | mean Q | clips in twin groups |
|---|---|---|---|---|
| YOLO tracks only | 22 | 57.8% | 5.58 | 27 |
| + frame brightness | 24 | 68.8% | 5.75 | 20 |
| + Cosmos tags (indoor/outdoor, setting) | 32 | 76.6% | 5.80 | 15 |
| + Cosmos caption keywords = the page | 49 | **93.8%** | 5.97 | 4 |
| first version (tags + dominant colour, no keywords) | 40 | 81.3% | 5.84 | 12 |

Caption keywords did most of the work. The old twins are now split by a single caption word each: c41/c42 by "woman, sitting" (c41's caption names no kept keyword), c48/c50 by "woman, man", c60/c61 by "walking", c44/c58 by "road" vs "empty". Several of those splits hang on one word in one Cosmos caption, so they are fragile.

Archive size (random sub-archives of the same 64 rows, strict engine, 60 draws): 16 clips 99.4% / 4.0 Q; 32 clips 96.7% / 5.0 Q; 48 clips 94.9% / 5.6 Q; 64 clips 93.8% / 6.0 Q. Questions grow with log2(N); the lookalike share still creeps up with N, so a real archive of thousands of segments will have many more attribute twins than this one.

Per video (clips in the archive / found uniquely): retail_cctv 4/4, lisbon_parking 8/8 (1 invisible), ptz_parking 6/4, rochester_night 3/3, yonge_dundas 3/3, motorway_a40 2/0, dashcam_urban 4/4, elephants_dream 8/8 (1 invisible), car_detection 2/2 (1 invisible), person_bicycle_car_detection 2/2 (1 invisible), all other videos 1/1 to 4/4. Full table in `results.json` and `out/ALL/report.html`.

Index attributes vs my own eyeballed ground truth (`gt.json`, mid-frame of every clip):

| check | agree |
|---|---|
| indoor/outdoor (Cosmos) | 53/53 |
| setting (Cosmos) | 44/60 (all 8 animated Elephants Dream clips are never called "animation", c46/c47 classroom called "room at home or office", Lisbon title cards, c39 skate fall called "parking lot", 2 unknown) |
| night clips -> brightness dark | 3/3 |
| daylight clips -> bright | 49/50 |
| person visible at mid-frame -> persons >= 1 | 29/29 (recall only; the count is a 5 s peak) |
| vehicle visible at mid-frame -> vehicles >= 1 | 17/17 (same caveat) |

**Caption keyword questions vs what is visible** (new; I labelled 7 visual keywords on all 64 mid-frames: clearly visible / unsure / clearly not):

| keyword | index says yes and visible | index yes, not visible | visible, index says no | precision | recall |
|---|---|---|---|---|---|
| trees or greenery | 13 | 0 | 4 | 1.00 | 0.76 |
| sky | 3 | 0 | 10 | 1.00 | 0.23 |
| a woman | 4 | 0 | 12 | 1.00 | 0.25 |
| a man | 3 | 0 | 17 | 1.00 | 0.15 |
| parking lot / parked cars | 14 | 2 | 1 | 0.88 | 0.93 |
| shelves / products | 6 | 0 | 1 | 1.00 | 0.86 |
| hallway / corridor | 3 | 1 | 0 | 0.75 | 1.00 |

A caption only names what is salient in one frame, so "yes" is trustworthy and "no" is not: a judge who sees a woman answers YES where the index says no for 12 of 16 clips. That is the main fragility of this version. Test: a judge who answers these 7 questions from what is visible (unsure -> "Not sure") and every other question from the index. The page protocol still finds **64/64** (6.2 questions + 1.05 guesses, 95% on the first guess); the strict engine without confirmation would find only **87.5%**. The misses are absorbed by the soft recovery, not avoided.

Simulated wrong answers (SIMULATED: each answer flipped with probability p, 20 plays per clip, truthful guess confirmation):

| engine | p=0 | p=0.05 | p=0.10 |
|---|---|---|---|
| strict hard elimination, no confirmation (v1 page) | 93.8% / 6.0 Q | 69.0% / 6.0 Q | 50.7% / 6.0 Q |
| page protocol (confirm + soft recovery, max 3 guesses) | 100% / 6.0 Q + 1.0 guess | 99.9% / 7.8 Q + 1.3 guesses | 99.1% / 10.3 Q + 1.5 guesses |

Protocol parameters (`hs_core.PROTO`): eps 0.1 (each contradicting answer costs a factor 0.1), stop at posterior 0.8, at most 3 wrong guesses, at most 4 clips shown in a "which of these" step, at most 30 questions. Swept stop 0.5 to 0.9 and eps 0.05 to 0.2 on this same archive; all settings found 92% to 100%, so the choice is not delicate, but it is tuned on these 64 clips.

## Eyeball checks
I viewed all four 16-tile contact sheets (mid-frame of every one of the 64 clips) twice (first pass for the attribute ground truth, second pass for the keyword labels), the corpus hero, the ptz_parking and lisbon_parking heroes, and drove the page in the built-in browser.
- lisbon_parking is a photo slideshow (still photos with pan/zoom, Portuguese title cards, a poster), not live traffic. The index calls it "camera moving" (digital pan/zoom), gives 4+ vehicles for photos of parked cars, and calls the police-car photo and the poster (c11, c12) "bike present" (false positives). The title card c05 is correctly invisible to YOLO.
- c13/c14 (same static ptz lot, different time) and c25/c26 (same highway, different traffic) are visibly near identical: the remaining twins are real. c15 is the same view but 1 tracked person separates it.
- c41/c42 (man in bed vs man being helped to sit) were a false twin in v1 (identical counts, motion, brightness); they are now split only by the caption keywords "woman, sitting" (c42). c48 (empty corridor) vs c50 (two close faces) are split by "woman, man". c60/c61 (same aisle) are split only by "walking" in one caption: a thin separation.
- c44 (car_detection) and c58 (empty parking) are empty asphalt: truly invisible to COCO detectors; they are now told apart only by caption words (c44 "road", c58 "empty"), which is as thin as it sounds.
- Rochester c19-c21 are night street scenes (dark); Elephants Dream c31-c38 are animated (YOLO finds "people" in puppets, Cosmos calls them street/shop/factory).
- Hero: clip labels now sit on dark pills (readable over bright sky in ptz_parking), invisible clips get a dashed cyan border plus grey-out, the stamp no longer overlaps the legend. The page in the browser: the guess screen, a deliberately wrong answer (c22 with the 2nd answer flipped: first guess c23 rejected, found c22 after 11 questions + 2 guesses), a twin pick (c13), auto-play of c26 and the lookalike tab all worked; no console errors besides a missing favicon. The screenshots were small (pane size), so the visual layout of the guess panel was checked at thumbnail level and by reading the DOM text, not at full resolution.

## Simulated or faked
- Wrong answers (flip table above) are SIMULATED; no human played the page. The kill-test numbers come from the engine answering for the clip's own index attributes, i.e. a perfectly consistent judge. The eyeball-judge test covers only the 7 caption keywords I labelled; the other 42 questions (YOLO counts, motion, brightness, Cosmos tags) were still answered from the index, and I labelled the keywords knowing what the index said, so it is optimistic.
- The kill test was FAILED in v1 and PASSES now because I changed the attribute set after seeing why it failed (removed colour, added caption keywords). The keyword lexicon was written after reading the caption word counts but before looking at which clips were twins, and I dropped one unanswerable keyword ("clothing described") for answerability, but it is still a post-hoc fix on the same 64 clips. A new sample could land lower; the archive-size table shows the trend.
- Footage for "judges" is these 21 public test clips; nothing was shown to a real judge.
- Cosmos3-Edge stands in for Cosmos3-Reason; YOLO26s for YOLO11s; duckdb for VastDB. No real VAST service was called.
- Thresholds (brightness dark < 62, dim < 90; camera moving = median Farneback flow > 0.5 px at 160 px width; moving object > 0.35 own-sizes per second) were fixed before the first run except the two brightness thresholds, tuned after eyeballing (so the "night"/"daylight" checks are optimistic).

## What fails / limits
- Caption-keyword "no" answers are unreliable (recall 0.15 to 0.76 for most keywords); the page survives only because of "Not sure", the guess confirmation and soft recovery, at the price of about 1 extra question and 1 guess in the clean case and 8 to 10 questions when answers are wrong.
- Kill test: 94% found, mean 5.97 questions: passes. But the pass counts only the strict engine; the "found" figure of the page itself includes one guess per game (6.97 steps on average, still under 7). The lookalike share still grows with archive size.
- The first questions that win information gain (indoor/outdoor, truck or bus, trees) are easy for a judge, but later ones ("someone stands still for 3+ s", "mostly move left to right", "4+ vehicles at any moment") are about 5 s peaks and motion, not what a thumbnail shows; hover cards help but do not remove the mismatch.
- Camera-moving is wrong for slideshows (digital pan), correct for dashcam/PTZ/handheld.
- Cosmos setting is 73% accurate vs my eyeball and never says "animation"; one frame per 5 s segment, so a scene change inside the clip is not seen.
- 64 clips from 21 videos (several videos have 1-2 clips); one seed.

## Kill test result
Criterion: mean questions <= 7 AND >= 90% found, every clip as the secret (strict engine, truthful answers).
- Mean questions 5.97: PASS.
- Found uniquely 93.8% (60/64): PASS (v1: 81.3%, FAIL). The page protocol finds 100% at 6.97 questions + guesses; with an eyeball judge on the keyword questions 100% at 7.2; with 5% random wrong answers 99.9% at 9.1 steps.
- Verdict: PASSED, with the caveats above (post-hoc attribute change, keyword answers unreliable, no human tested). The report half ("these 4 clips are lookalikes, these 4 are invisible") remains the strongest story.

## Rebuild plan on the day (hours)
About 6 h with the real services: 1 h pull DataEngine segment rows (counts per class) and pick the 64+ archive from the event footage; 1 h per-clip attributes (Cosmos tagging prompt, brightness from frames, motion from tracks if available); 1 h VastDB table plus the per-question SQL counts (including `LIKE` keyword counts on the real captions); 1.5 h port `engine.js` + page to the real thumbnails and clips; 1 h lookalike/invisible report with real captions; 0.5 h rehearsal with a person answering. Files to reuse: `engine.js` (strict engine + page protocol), `game_template.html`, `hs_core.make_questions/Engine/KEYWORDS`, `render.py`.
