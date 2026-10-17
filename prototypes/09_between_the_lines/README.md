# Between The Lines

![Between the Lines animated demo](../../docs/demos/between-the-lines.gif)

[Animation details and footage credits](../../docs/DEMOS.md#09-between-the-lines).

## What it is
For an audio-description (AD) writer: it finds the gaps in a programme's dialogue and drafts a short description line (at most 8 words, at most 2.5 words per usable second) that fits each gap, tells the writer how far each line can be trusted (GREEN / UNVERIFIED / AMBER), and produces a WebVTT track, an HTML player and an audio-described MP4 (voice mixed in at each gap start, programme ducked under it).
It is a first-draft tool for a human writer (UK AD: Ofcom 2010 estimated about 61,654 audio-described broadcast hours a year), not an auto-describer.

Run: `python run.py --all` (about 75 s warm; 2-3 min cold with the local VLM at 0.3-6 s per call), `python test_smoke.py` (about 5 s), `python run.py --validate` (SIMULATED speech-detector test, about 2 min), `python run.py --video NAME --end S`, `--no-vlm` for detector-only template lines, `--demo NAME` re-cuts the 60 s demo window.

Files per clip in `out/<video>/`: `hero.jpg`, `report.html`, `player.html` (loudness waveform + speech + lines timeline, with/without-description toggle), `ad.mp4`, `plain.mp4`, `descriptions.vtt`, `demo_60s.mp4` (Elephants Dream and dashcam), `lines.json`, `audit_audio.png` (full spectrogram), `audit_lines_N.jpg` (spectrogram strip per line, for my by-eye speech audit), `eyeball_N.jpg` (keyframes + line, for my by-eye accuracy audit). `results.json` has every number below; `eyeball.json` and `spectro_audit.json` hold my by-eye verdicts.

## Implementation notes

- ffmpeg silencedetect and the small CNN speech detector in `vad_net.py` / `train_vad.py` estimate dialogue spans.
- YOLO26s + ByteTrack supply tracks, with YOLO26l as a second opinion.
- Local Cosmos3-Edge describes three keyframes and makes a second call to read on-screen text.
- Frame differences, shot cuts and new tracks determine what is new.
- Windows SAPI en-GB Hazel (`System.Speech` via PowerShell), at rate -1, supplies speech; ffmpeg mixes it into the output.
- duckdb is not used by the core loop. Canary word timestamps and hosted captioning are not implemented here.

## Results on test footage
Real measured numbers (`results.json`). "Eyeball" = my verdicts after viewing the keyframes each line was written from: accurate = every claim checks out; partly = subject/action right but a wrong, invented or missing detail; wrong. I am one rater and a few calls are generous (marked in `eyeball.json`).

| Clip (analysed) | Gaps (usable s) | Lines | GREEN / UNVERIFIED / AMBER | Over detected speech | Programme ducked | Eyeball accurate / partly / wrong | Spectrogram audit (clear / unclear / speech-like) |
|---|---|---|---|---|---|---|---|
| elephants_dream (first 180 s) | 10 (55.8 s) | 10 | 4 / 1 / 5 | 0.0 s | 15.1 dB mean | 4 / 6 / 0 | 6 / 4 / 0 |
| yonge_dundas (57 s, ambient) | 5 (33.3 s) | 5 | 3 / 0 / 2 | 0.0 s | 15.1 dB | 3 / 2 / 0 | 2 / 3 / 0 |
| dashcam_urban (120 s, engine noise) | 12 (75.9 s) | 11 | 5 / 4 / 2 | 0.0 s | 15.7 dB | 5 / 5 / 1 | 11 / 0 / 0 |
| Total | 27 (165 s) | 26 | 12 / 5 / 9 | 0.0 s | | 12 / 13 / 1 | 19 / 7 / 0 |

Before this revision (previous version, same footage): 7 of 25 lines accurate (28 %), 6 wrong; Elephants Dream was 1 of 7 accurate. What changed it: AD-style prompt (at most 8 words, no filler, leave out unsure details), three keyframes instead of two, an on-screen-text read for title cards, and word budgets tied to 2.5 words per usable second.

Flag calibration (by eye, 26 lines):

| Flag | Lines | Accurate | Partly | Wrong |
|---|---|---|---|---|
| GREEN (a checked claim agreed) | 12 | 4 | 7 | 1 |
| UNVERIFIED (nothing checkable) | 5 | 4 | 1 | 0 |
| AMBER (disagreement / doubt) | 9 | 4 | 5 | 0 |

More measured facts:
- Pace: at most 8 words per line; words per usable slot-second 0.8-1.1 mean per clip (max 2.44, budget 2.5). Spoken pace (words over measured TTS seconds) is 2.1-2.6 words/s on average, 64.1 s of voice in total. Every voice clip fits inside its slot, measured with the real TTS.
- Where do lines sit? Only 3.1 s of those 64.1 s fall inside ffmpeg-silencedetect silence; the rest is over music, effects or ambience. If a gap had to be silencedetect-silent, usable gap time would be 0 s (Elephants Dream), 0 s (Yonge-Dundas) and 4.9 s (dashcam).
- Speech detector, SIMULATED validation (`out/vad_validation.json`, unchanged): known synthetic speech (held-out Zira voice, held-out sentences, 93 utterances) mixed into the real beds of the three clips at -3, 0 and +6 dB relative to the bed. Mean coverage of injected speech 0.991-1.0 in every bed and gain, 0 utterances missed, 0.0 s of gap overlapping injected speech.
- AMBER quality: 9 AMBER lines, 5 of them on lines I judged not fully accurate; of the 14 not-fully-accurate lines only 5 were flagged (recall 36 %). The one wrong line (dashcam "A car turns right at intersection.") is GREEN: YOLO can confirm that a car exists, not that it turns.
- Shots: 24 hard cuts in the Elephants Dream window; 2 of its 12 slots were held (no new information).
- VLM calls: 11 + 6 + 17 (describe plus on-screen-text reads), under the 30-per-video cap, plus text-only shorten calls.

## Eyeball checks
Frames I viewed (Read tool): 30 s spectrogram tiles of Elephants Dream 0-180 s, then every line's 3 keyframes in `out/*/eyeball_N.jpg` (10 + 5 + 11 lines), every line's spectrogram strip in `out/*/audit_lines_N.jpg`, and the three `hero.jpg`. Verdicts and notes per line: `eyeball.json`, `spectro_audit.json`. What I saw:
- Elephants Dream: line 0 is the title card and now reads it exactly ("the orange open movie project presents"); lines 1-2 "A man in a brown coat stands." are right (bald man at the console, a second figure lies on the platform and is not mentioned). Line 7 "A figure walks on a narrow bridge." is right (two figures walking). Lines 3-6, 8, 9 are partly: "runs" for a figure that walks, two men peering round a curtain described as one man stepping forward, a walkway described with a figure that is not clearly visible. No line is wrong; most are thin ("A figure walks.").
- Yonge-Dundas: "Pedestrians cross the intersection; cars drive past building." and "A woman walks." (pink blouse) are right; "A woman walks." three times is flagged as near-repeat.
- Dashcam: garage lines, "A black car drives past on the left." and "A traffic light is on the left." are right. "A car turns right at intersection." is wrong: nothing turns, the road ahead is empty. "A black car drives ahead on the right." is partly (the car is centre-left) and AMBER caught that with the left/right check.
- Spectrograms: ED 0-14 s is near-silent title bed, so the detector's red block at 0.2-4.1 s is a false alarm. ED 14.5-30 s and 150-168 s look like effects (broadband bursts, gated pulses), 168.5-180 s is the only stretch where I saw gliding-pitch structure that looks like voices, and no line sits there. Dashcam is engine hum with turn-signal ticks. Yonge has crossing-signal clicks and crowd murmur. I did not listen to any audio.
- Hero images: the stamp text, counts and cards are readable at 1280 px; I fixed an overlap of the subtitle under the stamp.

## Simulated or faked
- No real dialogue ground truth exists locally (no ASR weights). "0 s over detected speech" is measured against my own CNN detector, which chose the gaps, so it is circular by itself. What is not circular: the SIMULATED injection test (Windows TTS mixed into the real beds), and my spectrogram audit of each line's window (by eye, no listening): 0 of 26 lines show speech-like structure, 19 clear, 7 unclear (dense harmonic bed in ED 109-127 s, murmur in Yonge). A hand-marked full dialogue ground truth of ED 0-180 s was NOT produced: I cannot listen, and the dense effects in the film do not let me mark speech from a spectrogram reliably.
- Voice: Windows SAPI, not a broadcast AD voice. Descriptions: Cosmos3-Edge (a weaker stand-in for Cosmos3-Reason). Detector counts: YOLO26s/26l, not the hosted YOLO11s.
- The speech detector (`vad_model.pt`, 68k parameters) was trained on synthetic speech from two TTS voices over beds from other clips plus synthetic noises; the three test programmes, the Zira voice and the test sentences were held out. It has never been checked against real human speech.
- `eyeball.json` and `spectro_audit.json` are one rater (me), not a panel.

## What fails / limits
- Accuracy: 46 % of lines fully accurate (12 of 26), 96 % accurate or partly, 1 wrong. Strict reading of the kill test still fails. The lines are safe but thin: the local VLM drops the second character, says "runs" for a walk, and cannot confirm verbs.
- GREEN does not mean right. GREEN only means a count, colour or left/right claim in the line was checked against YOLO and agreed: 4 of 12 GREEN lines were accurate, 1 was wrong. UNVERIFIED lines (nothing checkable) were 4 of 5 accurate because they are plain statements. AMBER has false alarms (cartoon coat colour, white car box covering the bonnet, near-repeats of a correct line). So the colours say "what the machine could check", not "trust". The writer reads every line.
- Dialogue detection on busy tracks: 63 % of the first 180 s of Elephants Dream is flagged speech-like, which leaves 10 gaps (55.8 s) and several 1.5-3 s slots. Some flagged regions are effects, not speech (e.g. 0.2-4.1 s).
- Title text: the film's second card ("elephants dream") falls in a region flagged as speech, so it is not voiced. The text read is only tried where YOLO sees no person or vehicle (so a billboard in a crowd is not read as a title).
- Slots where no complete sentence fits are left silent (Yonge 1, dashcam 3) rather than voicing a fragment.
- Draft lines of dashcam are first person ("I drive...") because the model sees a windscreen view.
- English only, one voice, no character names (a real describer gets them from the script).

## Kill test result
Criterion: zero lines overlap dialogue AND at least 60 % of lines judged accurate by eye.
- Dialogue: PASSED on every measure I have: 0 lines and 0.0 s over the detector's speech map (26 lines), 0.0 s over injected speech (SIMULATED), 0 of 26 lines speech-like in my spectrogram audit (7 unclear). Not provable against real dialogue without a transcript.
- Accuracy: FAILED on the strict reading: 12 of 26 lines (46 %) fully accurate (previous version 28 %). Counting "partly" as usable-with-edit it is 25 of 26 (96 %).
- Overall: NOT passed. Plumbing (timing, fit, ducking, never over detected speech) is sound and the content is much safer than before (1 wrong line in 26), but still thin and below the 60 % bar with the local model.
