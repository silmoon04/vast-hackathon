# 15 PRIVATE VIEW: privacy-mask drift monitor

Local prototype. Run: `python run.py --all` (about 6-7.5 min here with other prototypes sharing the CPU: 4 outputs plus the raw-clip knock test), then `python test_smoke.py` (4 s, reads the stored outputs plus a few live maths checks).

## What it is

Council CCTV hides residents' windows with static privacy masks. If the camera is knocked or a PTZ moves, the masks stay put and the windows are exposed, and nobody notices. PRIVATE VIEW estimates camera motion against the setup frame (SIFT + RANSAC homography), warps the window polygons to where the windows really are now, and logs per window how much of it sticks out of the static mask (px misalignment, % exposed, start/end, max %).

## Implementation notes

- Zone proposals combine local Cosmos3-Edge grounding, a classical contour detector and operator polygons. Grounding produced one unusable box; the operator polygons are the ones used.
- OpenCV SIFT + RANSAC estimates camera motion and homography.
- YOLO counts are not needed by the core loop. A VLM breach opinion is not used.
- Cosmos-Embed1 segment similarity supplies the view-change screen; see its limits below.
- `pk.vastdb` provides a duckdb `seg_exposure` table. Queries use OR chains with no GROUP BY, IN or JOIN; example output is in `results.json` under `videos.*.vastdb_demo`.
- ASR is not used.

## Results on test footage

Numbers from the last `python run.py --all` (results.json). Sampling 1.5 Hz (PTZ), 5 Hz (Rochester), 25 Hz (Lisbon); mask margin 0.6% of frame width (7.7 px at 1280); breach = at least 5% of the window exposed. Timing on this loaded laptop: ptz 192 s, rochester sim 42-78 s, rochester real 52 s, lisbon 23-32 s, raw-knock test 33 s; the full `--all` took 358 s (results.json `generated_s` is from the last single-clip re-render).

| Clip (segment) | What is injected | Result |
|---|---|---|
| rochester_night (4-21 s) | SIMULATED step knock 8.0-18.0 s: dx +28 px, dy +12 px, roll 1.0 deg, applied to frames the monitor first stabilised | 7 of 9 windows exposed in truth, 7/7 detected, 0 false alarms. Start/end error 0.00 s on all. Peak exposure truth vs measured (pp): W1 8.4/9.0, W4 24.2/24.4, W5 27.6/27.6, W6 31.5/33.9, W7 37.4/37.4, W8 28.6/28.6, W9 30.1/30.1. Mean per-sample error 0.1 pp, worst single sample 2.3 pp. Partly circular, see the next row. |
| rochester_night RAW knock (4-21 s, `checks.raw_knock`) | SIMULATED step knock 8.0-14.0 s, other direction (dx -26, dy +14, roll -1.2 deg), injected into the RAW unstabilised clip. Truth = knock matrix composed with the clip's real drift measured by ECC (intensity-based alignment, no feature matching, independent of the SIFT monitor) | 9/9 windows pass (all nine exposed, peaks 28.6-43.4%: partial exposure, not 100%): start/end error 0.00 s, event-peak error at most 0.3 pp. The 17 real-drift-only events outside the knock also match (17/17). 0 false alarms, 0 samples lost. Per-sample error vs truth: mean 0.19 pp, worst 4.2 pp (inside the knock: mean 0.37, worst 4.2). Truth that ignored the real drift would be off by 5.7 pp on average, so the drift matters and is accounted for. |
| rochester_real (4-21 s, REAL, no injection, no stabilisation) | nothing | 17 events from real hand-held drift: W1-W3 from 00:15.8 and W4-W9 from 00:16.2 to the end of the segment, peaks W5 27%, W8 26%, W9 26%, W1 24%, W7 24%, W2 23%, W6 23%, W3 21%, W4 21% (partial exposure, windows moved up to 33 px); plus early blips of 6-9% at 4.0-6.4 s. Cross-check with ECC on the same 85 frames: window outlines agree to 0.22 px on average (max 2.3 px), exposure to 0.06 pp on average (max 1.8 pp), 17/17 events matched, start error 0.0 s, end error at most 0.2 s, peak at most 0.4 pp. 52 s to process. |
| ptz_parking (0-316 s, REAL PTZ) | nothing | aligned 0-155 s; PTZ leaves preset 02:35 (exposed 100% for 3.3 s in transit); view not recognised 02:38-03:58 (80 s, 25% of samples); exposed 100% 03:58-05:16 (77 s) for Z1, Z2, Z3. Landmark check: 4 roof landmarks read by eye at 280 s vs warped reference: errors 3.2, 3.1, 3.3, 1.9 px (camera moved 541 px right). Stand-in zones. 192 s to process. |
| lisbon_parking (531.2-533.8 s) | SIMULATED decaying shake 532.2-533.4 s, 14 px amplitude at 2 Hz | V3 25.4% for 0.92 s (08:52.2-08:53.1) and V2 9.5% for 0.36 s (08:52.2-08:52.6) in truth and measured, reported as two separate events; start/end error 0.00 s; per-sample error 0.0 pp (max 0.1). V1 never breached (its one sub-minimum blip is hidden in the strips and absent from the log). Clip is a photo-slideshow hold, see limits. |

Kill-test tolerance read as: start/end within 0.5 s, exposure within 10 percentage points, at event level.

Accuracy sweep (`checks.shift_sweep`, SIMULATED shifts of the reference frame itself, 7 sizes from 6 px to 90 px with roll up to 2.5 deg): SIFT corner error 0.03-0.21 px, exposure error at most 1.2 pp. ORB (the first thing I tried) is 0.7-10.6 px off and up to 4.9 pp, so SIFT is the default (`--feat orb` exists). This sweep has no real motion or moving objects: an upper bound on accuracy.

Zone setup on the Rochester reference frame: Cosmos3-Edge grounding returned 1 box, not on any window (0 of 9 hit; now labelled UNUSABLE on `zones_setup.jpg` and in the report instead of "ok"). Classical contour detector: 30 proposals, 6 of 9 operator windows at IoU 0.3 (parameters tuned on this one frame, so optimistic). Zones used = operator polygons.

Cosmos-Embed1 cross-check on the PTZ clip (63 five-second segments, cosine to the 5-10 s segment): aligned segments mean 0.96, zoomed-in (unrecognised) segments 0.11-0.56, but the shifted wide view after 03:58 still scores 0.92-0.94. It catches the zoom, not the pan: it cannot replace the geometry (`checks.embed_view_change`, from an earlier run).

## Eyeball checks (frames I viewed)

- Contact sheets of all three clips, before designing anything. Found that `lisbon_parking` is not a fixed camera: it is a ~1 photo/s slideshow with yellow title cards and posters. The longest photo hold with a usable object is 3.2 s (08:51-08:54, red van, old woman). Held-shot search used frame differencing; longest holds were title/poster cards.
- `rochester_night` is hand-held, not fixed: zoom/pan wobble from about 21 s (scale 0.9-1.8), so I used the calmest segment and measured its real drift (above).
- PTZ phases by eye (2 s contact sheets): preset A to 02:34; fast move and zoom 02:36-02:40; zoomed on a pole and grass island to ~03:56; transition 03:56-03:58; wide view with a different framing 03:58-04:11 (darker), another re-frame at 04:12 (tilted down). The pipeline's 02:35 / 02:38 / 03:58 boundaries agree with this.
- PTZ landmarks (hotel chimney tops, roof end, gable apex) read from 4x crops of frames at 5 s and 280 s: shift 537 px x, -127 px y, scale 0.987 by my fit; estimator says 541, -132.
- Rochester window polygons drawn by eye on the reference frame (3 lit windows on the right tower, 6 balcony windows on the front block), overlay on the frame confirmed they sit on the panes.
- Round 1: viewed hero.jpg of all three, overlay frames at 4-6 time points each, and `zones_setup.jpg`. Fixed on the way: the stamp covering the windows, an overlapping legend, a transit frame chosen as the "now" picture.
- Round 2 (this revision): viewed all four final heroes. Rochester sim hero: headline reads `W7 37%, W6 34%, W9 30%, W8 29%, W5 28%, W4 24%, W1 9% exposed for 10 s`, stamp bottom-right over the road (0 window pixels covered, computed and asserted in the smoke test), both truth lines fit the canvas. Lisbon hero: two separate event lines (V3 25% 0.9 s, V2 10% 0.4 s), no stray amber cell in the V1 row, the NOW frame shows V3 and V2 exposed, matching the stamp. PTZ hero: per-window headline, otherwise unchanged.
- `rochester_real` hero, 3x crop of the NOW panel at 18.2 s: the yellow outlines of the three right-tower windows (W1-W3) enclose the lit panes while the red masks sit lower and to the left of them; the six balcony windows (W4-W9) show yellow outlines on the dark window squares with the red masks offset down-left. This matches the 0.2 px ECC agreement. I cannot judge sub-5 px agreement by eye, only that the outlines are on the windows and the masks are visibly off them.

## Simulated or faked (be explicit)

- Rochester knock (8-18 s), the raw-clip knock (8-14 s) and the Lisbon shake are SIMULATED: real frames warped by a known matrix.
- The 8-18 s test is circular in part: the clip is first stabilised with the monitor's own pass-1 motion, and truth is the injected matrix through the same exposure raster. The raw-clip test removes the stabilisation and takes the real drift in the truth from ECC instead. It is still not fully independent: the exposure definition (mask margin, 5% threshold, polygons) is shared, and ECC stands in for a truth nobody measured.
- There is no real residential-window breach in any footage. The only real camera move (PTZ) uses stand-in zones (hotel roofline, utility mast) and shows only 0% or 100%. Real partial exposure exists only in `rochester_real` (drift), where the reference is a second estimator, not a measurement.
- Lisbon V1-V3 (van windscreen, side windows, plate) are stand-ins on a photo.
- "Operator polygons" are mine, by eye; there is no setup UI.
- Mask margin and the 5% breach threshold are my choices; a real deployment would set them per site.
- Cosmos grounding ran on the local Edge model, not Cosmos3-Reason. Cosmos-Embed1 ran locally.

## What fails / limits

- Unrecognised views (zoomed PTZ) are reported as UNVERIFIED, not exposed or safe: the system cannot say where the windows are. 25% of PTZ samples.
- Direct match to a far-away PTZ preset is weak (about 70 inliers) and up to 10-17 px off; the tracker chains through intermediate frames and re-anchors at full resolution, which gave 3 px at 280 s but is state-dependent and drift-prone.
- Single global homography: near and far planes differ for translating cameras (fine for pan/tilt/zoom and small knocks, and for a near-planar facade).
- Moving cars and lamp glare reduce inliers; night windows are low texture (still 500+ inliers on Rochester).
- Exposure assumes the mask stays where it was drawn in image space (software masks). Cameras with native PTZ mask memory behave differently.
- Domain gap on PTZ: real PTZ cameras usually keep privacy masks per preset (re-projected by pan/tilt), so a PTZ moving on purpose does not expose masks the way this clip's static masks do. The realistic customer case is the knocked or drifting fixed camera, which is only simulated or small real drift here.
- Cost: SIFT extraction is about 0.25-0.6 s per sample on this loaded laptop (single thread), so not real time; the PTZ clip needed 1.5 Hz. A GPU/optimised matcher would be needed for many cameras.
- Cosmos Edge grounding is unusable (1 box, wrong); the classical fallback finds about two thirds of windows after tuning on one frame; operator polygons are the real setup path. Only 9 of about 30 visible Rochester windows are masked.
- Lisbon test is only 2.6 s on a still photo and has no real-footage value beyond sub-second tracking.
- The PTZ hero shows all three zones as one 100% block because the camera left the scene entirely, and there is no partially-exposed real PTZ case in this footage (this clip cannot provide partial real exposure; Rochester supplies that case). The PTZ images show a parking lot with slivers on the top edge, so Rochester provides the clearer partial-exposure example.
- ECC, the cross-check estimator, only works for moderate motion from the previous solution and needs a rigid region (it masks the road with the moving car); it is a test tool here, not part of the monitor.

## Kill test result

Criterion: on the simulated shift, exposure start/end within 0.5 s and % exposed within 10% of truth. Result: PASS, in both forms.

- Original (stabilised frames, knock 8-18 s): 7/7 exposed windows detected, start/end error 0.00 s, peak exposure error at most 2.4 pp (event-mean at most 0.6 pp), 0 false alarms, mean per-sample error 0.1 pp. Partly circular (see above).
- Non-circular (RAW unstabilised clip, knock 8-14 s in the other direction, truth = knock x real drift measured by ECC): 9/9 windows pass, start/end error 0.00 s, event-peak error at most 0.3 pp, 0 false alarms; worst single sample 4.2 pp, mean 0.19 pp. Window exposures were 28.6-43.4%, a partial-exposure case.
- Real footage with no injection: the two independent estimators agree to 0.22 px and 0.06 pp on average, so the real-drift numbers are not an artefact of one algorithm.

Caveats: the tolerance is read as 10 percentage points; a knock of the size tested exposes 8-47% of a window, not 100%; the exposure definition is shared by monitor and truth; the 7-size sweep (up to 100% exposure, 90 px) stays within 1.2 pp for SIFT.

## Files

`run.py` CLI (clips: ptz_parking, rochester_night, rochester_real, lisbon_parking; `raw_knock_check`, `ecc_crosscheck`), `pview.py` exposure/log/drawing, `pview_motion.py` SIFT tracker plus `EccTracker` (cross-check only), `pview_zones.py` proposals, `zones/*.json` operator polygons, `out/<video>/` outputs, `results.json`, `test_smoke.py`.
