# Street Time Machine: module contracts

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

### synopsis -> `out/<cam>/synopsis.mp4` + `out/<cam>/synopsis.json`  (module `stm.synopsis`)
Video synopsis: every moving object from the whole recording pasted onto the background at once, time-compressed into <= 60 s, with a small time label on each ghost. H.264 mp4 (`avc1` fourcc or ffmpeg on PATH, overridable with `STM_FFMPEG`), 960 px wide, 12 fps.
```
{"camera": str, "src_duration": float, "out_duration": float, "fps": float, "w": int, "h": int,
 "ghosts": [{"object_id": int, "name": str, "day": str, "src_start": float, "src_end": float,
             "out_start": float, "out_end": float, "label": "mm:ss",
             "boxes": [[t_out, x1, y1, x2, y2], ...]}]}   # boxes in synopsis-video pixel coords, one per 0.25 s
```
Parked vehicles appear as one static ghost each (not repeated every frame). Schedule start times so ghosts overlap as little as possible.

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

### captions -> `out/captions/captions.json`  (module `stm.captions`)
"Caption lie detector": check the pipeline's AI descriptions against detector evidence. Input rows use the VSS row shape and come from `STM_CORPUS` or the parent toolkit's `data/mock_corpus.json` (fields reasoning_content, object_classes, object_counts, segment_start_sec, segment_end_sec, filename). Also cross-check against our own tracks where the clip matches a dev camera (person-bicycle-car-detection -> `person_bicycle_car_detection`).
```
{"rows": [{"id": str, "clip": str, "start": float, "end": float, "caption": str,
           "claims": [{"text": str, "object": str, "count": int|null}],
           "pipeline_counts": {..}, "our_counts": {..}|null,
           "verdict": "consistent|contradiction|unverifiable", "why": str, "frame": str|null}],
 "totals": {"consistent": int, "contradiction": int, "unverifiable": int}}
```
Claims are extracted with plain rules (number words + COCO nouns + synonyms). Also flag detector-side errors (e.g. "toothbrush" on a bolt clip) as `detector_suspect: true` with a reason. Save one frame per row as `out/captions/frames/<id>.jpg` (frame at segment midpoint; video paths: `<STM_VIDEO_ROOT>/intel/<clip>.mp4`, default `videos/intel/`).

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
Cap at the 150 objects with most frames per camera. `search(cam, "white car")` -> list of object_ids matching colour words + class words (synonyms: van/suv->car, people/pedestrian->person, bike->bicycle).

### tracker + packd  (modules `stm.tracker`, `stm.packd`)
`stm.tracker.track(df) -> df` with a `tid` column, for raw per-frame boxes with no ids (IoU + centre-distance greedy matching per class, lost-track buffer ~1 s). `python -m stm.tracker` evaluates on cached cams by dropping tid and re-tracking: writes `out/tracker_eval.json` {cam: {tracks_ours, tracks_ref, purity, fragmentation, frames_per_s}}.
`stm.packd`: adapter for the event's neighbourhood-camera pack (VastDB detection rows / bbox sidecars; see the parent toolkit's `docs/API_NOTES.md` and `vastkit` package). `register(cam, video_path, det_rows, days=None, title=..., credit=...)` converts rows to the track-cache row shape [f,t,tid,cls,name,conf,x1,y1,x2,y2] (running stm.tracker when ids are missing), writes `<cam>__det__s1__c25.json.gz` into data.TRACK_CACHE and adds the cam to data.CAMS at runtime. This source import runs only offline smoke checks; saved Pack D results are separate reference snapshots.

### export -> `dist/`  (module `stm.export`)
Bakes a static, single-file demo for publishing: `dist/index.html` = `ui/index.html` with `<script>window.__STM__={...};window.__STM_FILES__={...}</script>` injected right before the page's main `<script>`. `__STM__` maps every JSON URL the UI fetches (`api/cams` -> list of cam ids, `out/<cam>/<name>.json`, `out/captions/captions.json`) to its parsed content. `__STM_FILES__` maps every image path referenced in those JSONs or by the UI (`out/<cam>/background.jpg`, `background_A/B.jpg`, `flow.jpg`, crops, thumbs, frames) to a JPEG data URI (crops/thumbs <= 220 px long side q70; plates <= 960 px q72). Videos are copied, not inlined: `dist/out/<cam>/synopsis.mp4`. Keep `dist/index.html` <= 14 MB: if over, drop the least important images first (yonge_dundas crops, then other dev cams' crops) and log what was dropped. Prints sizes.

### brief -> `out/<cam>/brief.json`  (module `stm.brief`)
The overnight briefing: what a person should know in 10 seconds. Built by rules from whatever outputs exist (ledger, diff, near, moments, flow, synopsis, index); missing inputs are skipped and listed.
```
{"camera": str, "headline": str, "bullets": [{"text": str, "t": float|null, "kind": str, "crop": str|null}],
 "honesty": [str], "generated_from": [str], "missing": [str]}
```
Max 6 bullets, every number copied from an input file. honesty lists blind time and anything SIMULATED.
