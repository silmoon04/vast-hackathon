# Time Search: module contracts

Data interfaces for analysing a fixed street camera across recordings, with explicit unknown states when visibility is insufficient.

## Data and execution conventions

- Python: install `requirements.txt` in a virtual environment. Run from this folder: `python -m stm.<module> <cam>`.
- `stm.data` supplies tracks, objects, path, frame_at, background, crop, save_crop, days, info, out_dir, write_json and CAMS.
- Analysis modules write their named outputs under `out/<cam>/`, using the schemas below.
- Output values must come from computation. If something cannot be seen because the image is dark, frozen or blocked, return UNKNOWN with a reason.
- Dev cameras: `ptz_parking` (main: a fixed car park, Day A/B = SIMULATED halves of one clip), `car_detection`, `person_bicycle_car_detection`, `rochester_night`, `yonge_dundas`.
- Tracks are YOLO + ByteTrack at ~15 fps. Columns: f, t, tid, cls, name, conf, x1, y1, x2, y2, cx, cy, bw, bh, day. `objects(cam)` gives one row per track with `parked`.
- Modules expose a CLI `python -m stm.<module> [cams...]`; the default camera selection depends on the module.
- All JSON times are seconds into the video (float, 2 dp). Paths inside JSON are relative to `out/<cam>/`.
- Module summaries use `out/<cam>/<module>_summary.txt` and report what was computed, with counts.

## Outputs (the UI reads exactly these)

### health -> `out/<cam>/ledger.json`  (module `stm.ledger`)
```
{"camera": str, "seconds": int,
 "blind": [{"start": float, "end": float, "reason": "dark|blur|frozen|blocked|no_signal", "score": float}],
 "spots": [{"spot_id": "S1", "box": [x1,y1,x2,y2],
            "spans": [{"start": float, "end": float, "state": "present|gone|unknown", "object_id": int|null,
                       "reason": str|null}],
            "changes": [{"t": float, "kind": "arrived|left|swapped", "before": "crops/..jpg"|null, "after": "crops/..jpg"|null,
                         "same_vehicle": true|false|null, "similarity": float|null}]}],
 "per_second": [{"t": int, "visible": bool, "reason": str|null}]}
```
Spots = clusters of parked-vehicle boxes (IoU >= 0.5). `unknown` spans come from `blind` seconds (never "gone" while blind).

Added keys (all optional for readers; a ledger synced from an older run may lack them):
- `blind[].reason` and `per_second[].reason` may also be `moved` (the camera itself is moving: PTZ swing, zoom, shake) or `turned_away` (the camera holds still on a view that is not its home view: "camera pointed elsewhere"). Full set: `no_signal|dark|frozen|moved|turned_away|blocked|blur`.
- `blind_total_s: float` - THE one blind total (sum of `blind` spans, 1 dp). Every view prints this number.
- `blind_by_reason: {reason: float}` - the same total split by reason, largest first.
- `blind_by_day: [{"day": id, "label": str, "start": float, "end": float, "blind_s": float, "by_reason": {reason: float}}]` - per day of meta.days (ids never assumed "A"/"B").
- `home_views: [{"id", "day", "label", "view": int, "start", "end", "seen_s": int}]` - when each day's home view was on screen.
- `reason_words: {reason: str}` - plain words for each reason ("dark" -> "too dark", "turned_away" -> "camera pointed elsewhere"); UIs print these, not the raw ids.

### synopsis -> `out/<cam>/synopsis.mp4` + `out/<cam>/synopsis.json`  (module `stm.synopsis`)
Video synopsis: every moving object from the whole recording pasted onto the background at once, time-compressed into <= 60 s, with a small time label on each ghost. H.264 mp4 (`avc1` fourcc or ffmpeg on PATH, overridable with `STM_FFMPEG`), 960 px wide, 12 fps.
```
{"camera": str, "src_duration": float, "out_duration": float, "fps": float, "w": int, "h": int,
 "ghosts": [{"object_id": int, "name": str, "day": str, "src_start": float, "src_end": float,
             "out_start": float, "out_end": float, "label": "mm:ss",
             "boxes": [[t_out, x1, y1, x2, y2], ...]}]}   # boxes in synopsis-video pixel coords, one per 0.25 s
```
Parked vehicles appear as one static ghost each (not repeated every frame). Schedule start times so ghosts overlap as little as possible.

Added keys:
- `view: str` - the camera view this render shows: `"fixed"` for a fixed camera, else the day id of the home view (a PTZ camera that turns away is rendered on its first home view only). `view_label: str` - e.g. `"Day A view"`, `""` for fixed.
- `src_window: [float, float]` - the source seconds the render covers.
- `plate: str` - background of the first segment (`background_<day id>.jpg` for a multi-day camera, else `background.jpg`).
- `segments: [{"day", "label", "out_start", "out_end", "src_start", "src_end", "plate"}]` - one output segment per day, in day order, each on that day's own plate; ghosts from different days never share a segment.
- `skipped: {"other_view_tracks", "short", "too_close_people", "parked_fragments_merged", "cap"}` - counts of tracks left out and why.
- `redacted: {"person_detections_blurred": int, "lens_close_person_steps_hidden": int}` - people's heads are blurred before pasting; steps of a person right at the lens are not drawn.
- `ghosts[].parked: bool` - true = one static, unlabelled, half-transparent ghost per parked stay (then `object_ids: [int]` lists the merged tracker fragments and `speed` is null); false = a moving ghost with `speed: float` (playback speed-up). Keys starting with `_` are internal.

### diff -> `out/<cam>/diff.json`  (module `stm.diff`)
Day A vs Day B for cameras with two days (`ptz_parking`). Match parked vehicles across days by box IoU and appearance (colour histogram + small-image correlation).
```
{"camera": str, "day_a": "A", "day_b": "B",
 "changes": [{"kind": "same|arrived|left|moved|swapped", "box": [x1,y1,x2,y2],
              "a": {"object_id": int, "crop": str, "t": float}|null, "b": {...}|null,
              "similarity": float|null, "note": str}],
 "counts": {"same": int, "arrived": int, "left": int, "moved": int, "swapped": int},
 "background_a": "background_A.jpg", "background_b": "background_B.jpg"}
```
`day_a` / `day_b` are the camera's real day ids from meta.json (`"A"`/`"B"` on ptz_parking, `"20260901"`/`"20260902"` on neighborhood_cam-1), and the background names follow them. Added keys:
- `infrared: {day_id: bool}` - which day's plate is night infrared (no colour). When exactly one day is infrared, looks can't be compared across days.
- `unconfirmed: int` - how many changes carry `"unconfirmed": true`: kind `same` from a spot that holds a parked vehicle on both days while one day is infrared, so it can't be confirmed to be the same vehicle (`similarity` null). Readers must not count these as confirmed "same".
- `alignment: {"method", "ok", "matches", "inliers", "reproj_px", "corner_shift_px", "overlap_frac", "view_a", "view_b", "H_b_to_a", "fixed_view"}` - how day B was lined up onto day A. `fixed_view` is null when SIFT found enough matches; else the fixed-camera fallback `{"ok": bool, "dx": float, "dy": float, "ncc": float, "ncc_off": float}` (phase correlation of edge maps; ok = shift <= 6 px and edge match beats every 12-40 px shift). `background_b_aligned` = `background_<day b>_aligned.jpg`, day B warped into day A pixels.

### near -> `out/<cam>/near.json`  (module `stm.near`)
"Hit-and-run detective": for each parked vehicle, everyone/everything that came close while it was there.
```
{"camera": str,
 "targets": [{"object_id": int, "name": str, "box": [..], "start": float, "end": float, "crop": str,
              "visitors": [{"object_id": int, "name": str, "closest_t": float, "gap_px": float, "gap_rel": float,
                            "enter_t": float, "exit_t": float, "crop": str}],
              "blind": [{"start": float, "end": float, "reason": str}],
              "summary": str}]}
```
gap_rel = edge-to-edge gap divided by the target's box height (scale-free "how close"). Visitors = gap_rel <= 0.5. Read blind spans from `ledger.json` if it exists; otherwise compute a simple brightness/frozen check yourself. The summary says "who was near", never "who did it".

### ask -> `stm/ask.py` + `stm/server.py`  (module `stm.ask`)
Question engine over tracks for 6 question types: still_there, when_left, which_way, who_near, count_since, what_changed. `answer(cam, question:str, t_now:float|None) -> dict`:
```
{"type": str, "answer": "YES|NO|UNKNOWN|<number>|<text>", "sentence": str, "latency_ms": float,
 "evidence": [{"object_id": int, "t": float, "crop": str|null, "box": [..]|null}], "honesty": str|null}
```
Rule-based parsing for now (keywords + colour words matched against the crop's dominant colour + names). Also precompute `out/<cam>/qa.json` with 8 demo questions per camera and their answers. `stm/server.py`: FastAPI app serving `ui/` at `/`, `out/` at `/out`, `GET /api/cams`, `GET /api/ask?cam=&q=&t=`. Run: `<py> -m uvicorn stm.server:app --port 8780`.

Added answer keys (also in qa.json answers and `/api/ask`):
- `last_seen: float|null` - last second the target was actually seen (null when there is no target or it was never seen).
- `blind: [{"start": float, "end": float, "reason": str, "source": "ledger"|"scan"}]` - blind spans that bear on this answer (e.g. right after the last sighting); empty list when none. With a ledger.json its spans are the only blind source.
- `refused: "appearance"|"blame"|null` - `appearance`: the question describes or searches a person by looks (clothing, skin, gender, face), answer UNKNOWN by design; `blame`: "who did it" questions get who was near plus "it can't say who did it", answer UNKNOWN.

`GET /api/cams` returns a list (event camera first, then ptz_parking, yonge_dundas; `?all=1` adds the other prepared cams) of
```
{"camera": str, "title": str|null, "credit": str|null, "duration": float|null, "w": int|null, "h": int|null,
 "days": [{"id": str, "label": str, "start": float, "end": float}], "simulated": [str], "has_qa": bool,
 "ask": "live|precomputed|none", "default": bool}
```
`ask`: `live` = tracks + video on this machine; `precomputed` = answered from qa.json by question match (latency_ms null, header `X-STM-Source: precomputed`); `none` = no answers. `default`: true for the first listed camera (the one the UI opens). The baked export's `api/cams` has the same keys (no credit/w/h; `ask` is never `live` there).

`GET /api/health` -> `{"ok": true, "warm_enabled": bool, "warm": {cam: {"state": "queued|running|done|failed", "mode": ask mode, "questions": int, "load_ms"?, "n"?, "p50_ms"?, "max_ms"?, "drift"?: [...], "error"?: str}}, "warm_total": {"n", "p50_ms", "max_ms"}|null, "cams": {cam: "live|precomputed|none"}}` - the background warm-up (every qa.json question asked once at startup; `STM_WARM=0` skips it). `drift` lists questions whose live answer differs from qa.json.

### captions -> `out/captions/captions.json`  (module `stm.captions`)
"Caption lie detector": check the pipeline's AI descriptions against detector evidence. Input rows (VSS row shape) from `STM_CORPUS` or the parent toolkit's `data/mock_corpus.json` ( fields reasoning_content, object_classes, object_counts, segment_start_sec, segment_end_sec, filename). Also cross-check against our own tracks where the clip matches a dev camera (person-bicycle-car-detection -> `person_bicycle_car_detection`).
```
{"rows": [{"id": str, "clip": str, "start": float, "end": float, "caption": str,
           "claims": [{"text": str, "object": str, "count": int|null}],
           "pipeline_counts": {..}, "our_counts": {..}|null,
           "verdict": "consistent|contradiction|unverifiable", "why": str, "frame": str|null}],
 "totals": {"consistent": int, "contradiction": int, "unverifiable": int}}
```
Claims are extracted with plain rules (number words + COCO nouns + synonyms). Also flag detector-side errors (e.g. "toothbrush" on a bolt clip) as `detector_suspect: true` with a reason. Save one frame per row as `out/captions/frames/<id>.jpg` (frame at segment midpoint; video paths: `<STM_VIDEO_ROOT>/intel/<clip>.mp4`).

The event pack (`data/packd/segments.json`) goes to `out/captions/<camera_id>.json` (e.g. `out/captions/neighborhood_cam-1.json`) with the same row shape plus `camera`, `day`, `day_label`, `seg_start`, `seg_end`, `segment`, and top-level `days: [{"id", "label"}]`, `detectors`, `witness_note` (one witness there: its detector counts and our tracks come from the same YOLO sidecars, so `our_counts` stays null).

Appearance: descriptions of people's clothing and looks are cut from every caption before it is judged, saved or shown ("a man in a red jacket and a cap is walking" -> "a man is walking"; a sentence that only describes clothing is dropped). `caption` is the scrubbed text; the raw text is never written. `appearance_removed: int` per row = how many phrases were cut. `stm.captions.scrub_appearance(text) -> (text, n)` and `scrub_doc(captions_json)` are the shared helpers; `stm.export` runs `scrub_doc` on every captions JSON it bakes, so files written before the scrub existed are clean in the page too.

## Additional analysis modules

### flow -> `out/<cam>/flow.json` + `out/<cam>/flow.jpg`  (module `stm.flow`)
Where things move: a heat trail of every moving track drawn over `background.jpg` (people amber #F2A93B, vehicles blue #7FB6FF), plus the main lanes as arrows.
```
{"camera": str, "image": "flow.jpg", "w": int, "h": int,
 "lanes": [{"lane_id": "L1", "name": "left to right along the bottom", "group": "vehicle|person", "count": int,
            "from": [x,y], "to": [x,y], "path": [[x,y], ...], "median_speed_px_s": float, "example_ids": [int]}],
 "per_minute": [{"t": int, "person": int, "vehicle": int}],
 "busiest": {"t": float, "count": int, "frame": "frames/busiest.jpg"}}
```
Lanes = clusters of tracks by (entry point, exit point), path = mean trajectory resampled to 20 points. Parked/stationary tracks are excluded. Coordinates in source-video pixels.

### moments -> `out/<cam>/moments.json`  (module `stm.moments`)
The highlight reel: up to 12 notable moments per camera, sorted by t.
```
{"camera": str, "moments": [{"t": float, "end": float, "kind": "arrival|departure|crowd|close_pass|long_stay|blind",
   "title": str, "detail": str, "object_ids": [int], "thumb": "moments/<n>.jpg", "box": [x1,y1,x2,y2]|null, "score": float}]}
```
arrival/departure = a parked vehicle appears/leaves; crowd = peak count of people; close_pass = a person and a MOVING vehicle with edge gap < 0.3 x vehicle height; long_stay = longest-staying person; blind = from `ledger.json` blind spans if that file exists. Thumbs are full frames downscaled to 480 wide with the box drawn.

### index -> `out/<cam>/index.json` + `stm.index.search(cam, text)`  (module `stm.index`)
Everyone who appeared, with a dominant colour name from the crop (HSV, ignore borders and the darkest/brightest 10%).
```
{"camera": str, "objects": [{"object_id": int, "name": str, "colour": "white|black|grey|silver|red|blue|green|yellow|orange|brown",
  "rgb": [r,g,b], "first_t": float, "last_t": float, "parked": bool, "n": int, "crop": "crops/<id>.jpg"}],
 "colours": {colour: count}}
```
`colour` and `rgb` may be null: always for people, bicycles and motorcycles (never described by looks), and for a vehicle seen in a night or greyscale INFRARED frame (e.g. neighborhood_cam-1's 2 Sep night), a box too dark or too small to read. `colour_why: "person|rider|class|night|dark box|too small"|null` says why; `colour_reliable: bool`, `colour_frac: float|null`, `night: bool`, `day: str`, `crop_t: float` per object; top-level `night_days: [day id]`, `uncoloured: {why: count}`, `notes: [str]`. `colours` counts only objects that got a colour. Readers must treat a null colour as unknown, never as "black".
Cap at the 150 objects with most frames per camera. `search(cam, "white car")` -> list of object_ids matching colour words + class words (synonyms: van/suv->car, people/pedestrian->person, bike->bicycle).

### tracker + packd  (modules `stm.tracker`, `stm.packd`)
`stm.tracker.track(df) -> df` with a `tid` column, for raw per-frame boxes with no ids (IoU + centre-distance greedy matching per class, lost-track buffer ~1 s). `python -m stm.tracker` evaluates on cached cams by dropping tid and re-tracking: writes `out/tracker_eval.json` {cam: {tracks_ours, tracks_ref, purity, fragmentation, frames_per_s}}.
`stm.packd`: adapter for the event's neighbourhood-camera pack (VastDB detection rows / bbox sidecars, see the parent toolkit's `docs/API_NOTES.md` and `vastkit` package). `register(cam, video_path, det_rows, days=None, title=..., credit=...)` converts rows to the track-cache row shape [f,t,tid,cls,name,conf,x1,y1,x2,y2] (running stm.tracker when ids are missing), writes `<cam>__det__s1__c25.json.gz` into data.TRACK_CACHE and adds the cam to data.CAMS at runtime. Tested with a fake fixture only (no network).

### export -> `dist/`  (module `stm.export`)
Bakes a static, single-file demo for publishing: `dist/index.html` = `ui/index.html` with `<script>window.__STM__={...};window.__STM_FILES__={...}</script>` injected right before the page's main `<script>`. `__STM__` maps every JSON URL the UI fetches (`api/cams` -> list of cam entries in the `/api/cams` shape, `out/<cam>/<name>.json`, `out/captions/captions.json`, `out/captions/<cam>.json` for each exported cam) to its parsed content (captions JSONs pass through `stm.captions.scrub_doc` first). `__STM_FILES__` maps every image path referenced in those JSONs or by the UI (`out/<cam>/background.jpg`, `background_<day id>.jpg` for every `meta.days` entry, `flow.jpg`, crops, thumbs, frames) to a JPEG data URI (crops/thumbs <= 220 px long side q70; plates <= 960 px q72). Videos are copied, not inlined: `dist/out/<cam>/synopsis.mp4`. Keep `dist/index.html` <= 14 MB: if over, drop the least important images first (yonge_dundas crops, then other dev cams' crops) and log what was dropped. Prints sizes.

### brief -> `out/<cam>/brief.json`  (module `stm.brief`)
The overnight briefing: what a person should know in 10 seconds. Built by rules from whatever outputs exist (ledger, diff, near, moments, flow, synopsis, index); missing inputs are skipped and listed.
```
{"camera": str, "headline": str, "bullets": [{"text": str, "t": float|null, "kind": str, "crop": str|null}],
 "honesty": [str], "generated_from": [str], "missing": [str]}
```
Max 6 bullets, every number copied from an input file. honesty lists blind time and anything SIMULATED.
