# 01 RELEASE DESK

## What it is

A council information-governance officer answers a CCTV subject access request (SAR) in minutes instead of days: the desk lists the candidate people from a SQL query over the 5 s segment table, trims to the requester's seconds, exports with **everything except the requester pixelated by default**, then checks **every exported frame** with independent detectors (different models, run on the exported file) and forces a human review of every frame where a stranger overlaps the released region. The output is a release record (hashes, leak result, review list, stopwatch), not just a video.

Run: `python run.py --all` (about 3-4 min with the GPU idle), `python run.py --video retail_cctv [--tid 1368] [--window 0:111] [--cls person]`, `python run.py --video X --list` (step 1 only), `python run.py --hard` (appendix: the lisbon van case), `python eyeball.py` (contact sheets), `python test_smoke.py` (13 tests, about 10 s). Outputs: `out/<video>/{hero.jpg, report.html, export.mp4, release_record.json, regions.json, leaks.json, risky/*.jpg, candidates.jpg, fault_injection.jpg, eyeball.jpg}` and `results.json`.

## Demo script (90 s)

1. (0-15 s) Open `out/retail_cctv/hero.jpg`: left is the busy shop as filmed with 7 other people boxed in red; right is the frame that would be released, only the elderly shopper in the beige coat is visible. Say: "ICO gives us one month; the Home Office measured about nine hours to redact one hour of footage by hand."
2. (15-35 s) Open `report.html`, section 1: the SQL that listed the candidates (real SQL over the segment table plus a track table) and the thumbnail strip. "The officer clicks the shopper; tests auto-pick."
3. (35-55 s) Play `export.mp4`: the requester walks through the aisle, everything else is frosted glass. It is mask-by-default: a person the detector misses is still hidden.
4. (55-75 s) The banner under the panels says "0 strangers detected in 346 / 346 export frames" and next to it "caught and fixed: round 1 found 5 leaks, cut out and re-exported, round 2 found 0". Say it as it is: that is detector evidence on every exported frame, not a proof, and round 1 shows the tracker really does miss people. Show `fault_injection.jpg` (SIMULATED: a stranger pasted back in, caught 12 of 12).
5. (75-90 s) Forced-review list with crops (`risky/r06_*.jpg`: a stranger walks beside the requester, only the requester's silhouette is released). Finish on the stopwatch tile: 170 s of machine time end to end against about 4.0 min scaled from the Home Office figure, officer sign-off PENDING.

## How it maps to the VAST stack on the day

| Local stand-in here | On the day |
|---|---|
| `pk.segments()` (YOLO26s + ByteTrack counts per 5 s) + duckdb | DataEngine YOLO11s counts per 5 s segment in **VastDB**; the same simple SQL (no JOIN/GROUP BY/IN/window) narrows hours of footage to the candidate segments for the requested time window and class |
| per-frame boxes and track ids (YOLO26s + ByteTrack, cached) | The segment table has counts only, no boxes. Call the hosted YOLO11s on the candidate segments only and run ByteTrack ourselves |
| YOLO26s-seg instance mask of the requester | No seg model is listed for the event: either run a local seg model on the few overlap frames or fall back to the box (more of the requester lost) |
| Independent leak check: yolo26l + yolo26s-pose | **Cosmos3-Reason** on export frames ("is any person other than the one in the box visible? JSON") plus a local YOLO as a second opinion. Not built or tested here: the local VLM stand-in was not used at all |
| Candidate finding by thumbnail | **Cosmos-Embed1** text query ("woman in a beige coat with a brown bag") over segment video embeddings to rank segments; embed the requester's crop clip to find their other appearances and merge track ids |
| Audio: dropped | **Canary ASR** on the audio track: flag segments with speech, drop or redact audio of others (not built here) |
| Release record JSON on disk | A VastDB audit table (one row per release: hashes, leak result, review list, sign-off) |

## Results on test footage

All numbers measured by `run.py --all --hard` on this laptop (shared with other jobs; "net" excludes time queueing for the shared GPU lock, which was 0 s in this run). Source: `results.json`. The default set is the first three columns; the last column is the appendix hard case.

| | retail_cctv | lisbon_parking (black Mini) | person_bicycle_car_detection | APPENDIX lisbon van |
|---|---|---|---|---|
| Request (SIMULATED text, real window) | a shopper, 0:00-1:51 | my parked car, 0:15-0:19 | a person crossing the car park, 0:38-0:54 | my red van, 8:49-8:55 |
| Requester | tid 1368, **auto-picked**, 294 px tall | tid 2 (the Mini), **auto-picked**, 221 px | tid 33, **auto-picked**, 77 px | tid 8179 (van), auto-picked, 314 px |
| Released | 26.4 s of 110.9 s (24%), 346 frames | 2.4 s of 779 s, 61 frames | 6.2 s of 53.9 s, 75 frames | 3.4 s of 779 s, 86 frames |
| Frames tracked / interp / extrap (pad) / none | 336 / 5 / 5 / 0 | 25 / 24 / 12 / 0 | 68 / 0 / 7 / 0 | 43 / 42 / 1 / 0 |
| Export frames leak-checked | **346 of 346 (every frame)** | 61 of 61 | 75 of 75 | 86 of 86 |
| Leaks round 1 -> final | 5 -> **0** (11 frames patched) | 0 -> 0 | 0 -> 0 | 86 -> 0 (the silhouette holes, see limits) |
| Strangers detected in the same source frames | 5205 detections, 346 of 346 frames | 122 in 61 of 61 (the poster face) | 44 in 29 of 75 | 258 in 86 of 86 |
| Requester re-detected in export (control) | 93.1% of frames (source 99.4%) | 67.2% (82.0%) | 86.7% (97.3%) | 100% (100%) |
| Weak hits below the 0.35 gate / unmatched hits (export) | 17 / 182 | 0 / 0 | 0 / 0 | 0 / 0 |
| Simulated fault injection caught | 12 / 12 | 12 / 12 | 11 / 12 | 12 / 12 |
| Forced human review | 65 frames, 7 intervals (19%) | 60 frames, 1 interval (98%) | 0 | 86 frames, 1 interval (100%) |
| Export (decode + mask + encode) | 6.6 s = **4.0x real time** | 0.3 s = 7.9x | 0.4 s = 14.2x | 0.4 s = 9.1x |
| **Whole pipeline, end to end, net** | **170 s** (0.16x real time) | 14 s (0.17x) | 17 s (0.38x) | 36 s (0.09x) |
| Home Office 9 h per 60 min, scaled (not measured) | 4.0 min for the released seconds, 16.6 min for the whole clip | 0.4 min / 116.9 min | 0.9 min / 8.1 min | 0.5 min / 116.9 min |
| Verdict | REVIEW | REVIEW | CLEAR | REVIEW |

Reading it honestly: the headline number is now the end-to-end one. Against the scaled Home Office figure (seconds released x 9) the machine is 1.4x faster on retail (170 s vs 238 s), 1.6x on the Mini (14 s vs 22 s), 3.4x on pbc (17 s vs 56 s) and 1.2x **slower** on the appendix van (36 s vs 31 s, fixed overhead on a 3.4 s release). The hero's stopwatch tile turns amber when the machine is slower than the scaled manual figure. None of it includes the officer's review time for the flagged frames. The export step alone runs at 4-14x real time.

## Eyeball checks (frames I viewed and what I saw)

- Contact sheets of each clip (made before designing): **retail_cctv** is real HD overhead CCTV in a shop, 8-15 people at once. **person_bicycle_car_detection** is an almost empty car park seen from above with one pedestrian (about 0:40), a cyclist and a car that arrives later. **lisbon_parking is not CCTV**: it is a slideshow of about 2150 stills and a few short moving shots (2144 shot cuts found); I scanned all 17 shots longer than 1.5 s and picked two: the black Mini parked on a kerb at night (15.3-17.3 s, a moving shot with a political poster of a man's face behind it) and the red van of the earlier version (8:50, a still photo with an elderly woman walking in front).
- `out/retail_cctv/hero.jpg` (after the fixes): left, 7 strangers boxed; right, only the woman in the beige coat with a brown shoulder bag is sharp, the rest is frosted blur. The verdict is now a banner **under** the panels, so it no longer covers her feet. Readable in about 5 s.
- `out/retail_cctv/eyeball.jpg` (6 evenly spaced frames): in all 6 export tiles I see only the beige-coat woman (same coat, bag, posture). The first tile (1:24.5, extrapolated pad) now releases nothing visible instead of a stray sliver, and I saw no thin unmasked lines in any of the 6 tiles.
- `out/retail_cctv/risky/r06_095.17s`-style crop (right image, 1:35.2): a woman with a black backpack walks beside the requester; the release is the requester's silhouette only, the stranger is gone. First crop (1:27.8): a white-coat stranger stands next to her; the release shows a grey rectangular hole where his box was and loses part of her sleeve (box holes cost evidence). Third crop (1:24.6, entry frame): only a ragged fragment is released.
- `out/lisbon_parking/hero.jpg`: the released frame shows the black Mini as a clean car-shaped cut-out, the rest is dark blur; no grey hole in the subject. The only "stranger" is the man's face on the poster, boxed and labelled "(a face on a poster)"; the source label now says "1 other person/vehicle" and matches the one red box drawn. I checked the first export frame (extrapolated pad) at full size: the poster appears from the chin downwards, the face is blurred away; det_l and pose found nothing at conf 0.10 in the first 3 export frames.
- `out/lisbon_parking__van/eyeball.jpg` (appendix): in all 6 tiles the van is released in two pieces with a dark grey rectangle between them where the woman stood. This is still the old problem. See limits.
- `out/person_bicycle_car_detection/hero.jpg`: the small pedestrian (about 40x77 px) in the released frame, the rest is dark blur; the source panel says "as filmed (no other person or vehicle in this frame)" and the tile says "strangers in 29 source frames, none in the frame shown", so the numbers are explained. It is still a weak test (requester alone).
- `out/retail_cctv/fault_injection.jpg`: a man in a navy jumper pasted back into the blurred frame, boxed red "LEAK", labelled SIMULATED.

Ground truth I recorded by looking: retail requester tid 1368 = one woman, beige coat, brown bag, entering the left aisle at about 1:25 and standing by the right-hand shelf to the end; lisbon (Mini) tid 2 = the black Mini, the poster man is the only other "person"; van tid 8179 = the red van, tid 8233 = the elderly woman (stranger); pbc tid 33 = the pedestrian, the car and cyclist are strangers.

## Simulated or faked

- The request text, the case references (SAR-DEMO-000N), the received date (2026-10-01) and the one-month due date are made up.
- The officer's choice: `--tid` is auto-picked in every test (rule in `release_record.json`: highest `min(dur,30) * conf * (1 - overlap) * (1 - edge) * min(height,250)/250` among tracks at least 60 px tall with at least 80% coverage and the clip's minimum duration; 3 s everywhere except 1.5 s for the Mini shot).
- **Fault injection** is SIMULATED: stranger pixels pasted back into the export, then the same checker is asked. It tests the checker's sensitivity, not real redaction failures.
- Home Office comparison: the published figure scaled to the clip, never measured. Officer review time is not measured or estimated.
- "DataEngine" is YOLO26s + ByteTrack, not YOLO11s; "VastDB" is duckdb; there is no VLM and no ASR in this prototype; audio is dropped, not redacted.
- Officer sign-off is recorded as PENDING; nothing was actually released to anyone.
- The lisbon "footage" is a slideshow with a few moving shots, not CCTV, and its "stranger" is a poster.

## What fails / limits

- **"0 strangers detected" is a detector claim, not a proof.** It now covers every exported frame on all three default clips (346/346, 61/61, 75/75), but only for what yolo26l and yolo26s-pose can see at conf >= 0.35. Round 1 on retail released 5 people the tracker had missed; any person that both detectors miss at once would still be released if they stand inside the released region. The banner and `results.json` say "detected in N / N export frames" on purpose.
- **The leak gate was set after seeing the data.** At confidence 0.20 the pose and detection models "found" one giant person covering most of a blurred retail frame. I count a hit as a leak only if conf >= 0.35 **and** it overlaps a stranger the same detectors saw in the unredacted frame. 17 weak and 182 unmatched export hits on retail are reported but not gating. A stricter reader would call that tuning on the test clip.
- **Silhouette holes failed the checker on the van.** I tried the reviewer's suggestion (for a vehicle request, hide only the stranger's silhouette instead of her box). On the van, round 1 flagged a person in 86 of 86 frames, so the patch loop cut the box holes back in and the van is again released in two pieces. I did not investigate whether the detector fires on the blurred shape or on residual pixels at the dilation margin. On the Mini the same silhouette code was used for the poster (125 holes) and round 1 found 0 leaks. So that fix is only half valid, and the main lisbon case was replaced instead.
- **Strangers are hidden by their whole box** on person requests, so requester pixels inside a stranger's box are lost (65 of 346 retail frames overlap; the officer may want those pixels back).
- Requester fragments: a same-class detection that is at least half inside the released pixels counts as part of the requester (172 hits on the van, 0 elsewhere).
- A stranger fully inside the released rectangle on a frame with no tracked overlap and not found by the checker is released. No face detector beyond pose keypoints; no plate or text detector.
- Heavy blur is not a proof of irreversibility, only of non-detection by these models.
- One requester per run, one shot per release (`--tid a,b` merges tracker ids inside one shot only).
- The trim pad is 0.4 s, not 1 s (changed after review): the extra seconds were extrapolated boxes outside the frame that released only slivers. Extrapolated frames whose box is less than 25% inside the frame are dropped, and a 5x5 morphological opening removes unmasked slivers up to 4 px wide (unit-tested).
- Mini case: 60 of 61 frames go to human review, because the poster face overlaps the released rectangle for the whole shot.
- Only 3 default clips: one real CCTV clip with one hard stranger case, one trivial clip (pbc), one non-CCTV clip with a poster. No audio, no night CCTV, no crowds beyond 15 people.
- Export hashes are not reproducible bit for bit (x264 threading); the hash is of the file that was checked.

## Kill test result

Criterion: requester at least about 60 px tall **and** the export runs at least real time. **Passed** on all four runs: requester 294 / 221 / 77 / 314 px, export step 4.0x / 7.9x / 14.2x / 9.1x real time (decode + mask + encode, no GPU). The whole pipeline including the all-frame leak check is 0.09-0.38x real time, so "real time" holds for the export step only.

## Rebuild plan on the day (hours)

About 7 h with the real services:

- 1.0 h: ingest, VastDB segment table query for window + class, candidate list UI.
- 1.5 h: call hosted YOLO11s on candidate segments, ByteTrack, shot-cut handling, stranger overlap logic.
- 1.0 h: region planning (trim, interpolation, dilation, opening), pixelate/mask export with ffmpeg, burn-in reference.
- 1.0 h: seg masks (local model) on overlap frames and stranger-box holes, or the box-only fallback.
- 1.5 h: independent leak check with Cosmos3-Reason (prompt, JSON parsing, sampling) plus a local YOLO second opinion, patch-and-recheck loop.
- 1.0 h: release record, review list with crops, report page, audit row in VastDB.

Cosmos-Embed1 candidate search and Canary audio flagging are extra (about 1.5 h each) and not counted.
