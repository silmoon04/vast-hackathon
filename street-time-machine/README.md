# Street Time Machine

Browse a camera recording by event, compare parked vehicles across days, and ask questions with a timestamp and crop attached to the answer. A visibility ledger records dark, blurred, frozen and blocked intervals so the app can answer UNKNOWN when it could not see.

This folder contains the Python analysis modules, FastAPI server, browser UI, tests, output contracts and demo preparation tools. It also includes a small illustrated fixture so a fresh checkout can open the interface without event footage, credentials or model inference.

## Run the source checkout

Use Python 3.10 or newer. Run these commands from this folder:

```sh
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn stm.server:app --host 127.0.0.1 --port 8780
```

Open <http://127.0.0.1:8780>. Windows users can also run `serve.cmd`; it uses this folder's virtual environment, the parent toolkit environment, or `python` on PATH, in that order. `STM_PY` selects another interpreter.

When no prepared `out/<camera>/meta.json` files exist, the server serves `examples/synthetic/`. The camera is labelled SYNTHETIC DEMO. Its Near view and two sample Ask answers use preset events in an illustration. The sample has no video, inferred detections, measured accuracy or inference latency. Rewind, comparison, flow and caption views show their unavailable states until their outputs are prepared.

If prepared outputs exist, the server uses `out/` instead. Set `STM_OUT` to select a different output directory explicitly.

## Analyse your own footage

The analysis modules need video plus tracked detector boxes. No event videos, detection caches, credentials, model weights or generated media exports are included here.

For a camera configuration, copy `cams.example.json` to `cams.local.json`, then set its video path, days and footage credit. Relative video paths resolve beside that config file. `STM_CAMERAS` selects another config. Missing-video cameras remain available as remote metadata but cannot overwrite their synced outputs through local analysis.

The default folders are `videos/online/`, `videos/intel/`, `_cache/tracks/` and `out/`. Override them with `STM_VIDEO_ROOT`, `STM_TRACK_CACHE` and `STM_OUT`. The caption checker accepts `STM_CORPUS`; its default points to the parent toolkit's `data/mock_corpus.json`, which you can generate with that toolkit's corpus builder. `STM_FFMPEG` selects an ffmpeg executable; otherwise ffmpeg is found on PATH.

`stm.packd` reads local detection sidecars and links boxes without IDs using `stm.tracker`. With downloaded video and sidecars:

```sh
python -m stm.packd my_camera videos/my-camera.mp4 path/to/sidecars
```

For later processes, keep the same camera ID and video in `cams.local.json`. The adapter writes the track cache, and the config makes the camera discoverable by each subsequent command.

Run these commands for a configured camera with video and tracks:

```sh
python -m stm.data my_camera
python -m stm.ledger my_camera
python -m stm.synopsis my_camera
python -m stm.diff my_camera
python -m stm.near my_camera
python -m stm.flow my_camera
python -m stm.moments my_camera
python -m stm.index my_camera
python -m stm.ask my_camera
python -m stm.brief my_camera
```

Run ledger before Near and Ask. Run the briefing after the other modules. Synopsis generation and Pack D video assembly need ffmpeg on PATH. These commands process footage; starting the bundled fixture does not run them.

## Pack D setup helpers

The helpers in `scripts/` can fetch and assemble the event's neighbourhood camera when you have access to its team VSS endpoint. They are optional and are never run by server startup.

```sh
python scripts/fetch_packd.py --config /path/to/team.config --output data/packd
python scripts/build_packd.py --pack data/packd
```

The fetcher reads `INGRESS_URL`, `USERNAME` and `PASSWORD` from the selected config. Alternatively set `VSS_INGRESS_URL`, `VSS_USERNAME` and `VSS_PASSWORD`; `STM_VSS_CONFIG` can select the config file. Credentials stay in memory. The endpoint must use HTTPS. The fetcher logs in and downloads chunks, segment rows and sidecars; it does not upload or re-ingest footage. Keep the downloaded files and config private.

The builder uses ffprobe and ffmpeg, joins chunks in recording order, offsets segment timestamps, creates tracks, updates `cams.local.json` and writes metadata. Its `--camera` option selects the camera ID. These helpers target the documented Pack D filename and API formats. They have not been run as part of this source import, and downloading or transcoding an entire pack may take time and disk space.

## Saved analysis snapshots

`examples/reference-results/` preserves structured JSON outputs from local runs on Pack D and the public sample clips: visibility ledgers, comparison results, question answers, colour indexes, motion summaries, synopsis manifests and caption checks. `examples/packd/` contains the downloaded segment metadata and chunk offsets used by that run. Local absolute file paths have been replaced with relative references; credentials and token-bearing URLs are checked before importing these snapshots.

These are saved results, not outputs recomputed by this checkout. Their video, crops and background images are omitted, so some referenced assets are unavailable. They are kept outside the server's default output directory so the initial interface does not display missing-media cameras. Read them to inspect the work already done or use them as schema examples. To process the actual footage, obtain it through your authorised VSS access and run the setup helpers.

## Tests and demo tools

The source checkout smoke check makes no network requests and loads no models:

```sh
python -m pytest tests/test_source_checkout.py -q
```

The other tests cover module logic, synthetic fixtures and prepared video outputs. Several expect local footage or generated `out/` files, so running the entire test folder requires those inputs. `CONTRACTS.md` describes the output formats. `scripts/freeze_demo.py` checks prepared outputs and can export a static build; review its `--help` before using it.

`PITCH.md` is a demo outline for prepared footage. `CUECARD.txt` is a generated reference from a separate local run, including failed gates. Its numbers do not describe the bundled synthetic fixture and should be regenerated against the footage being shown.

## Modules

| Module | Purpose |
| --- | --- |
| `stm.data`, `stm.packd`, `stm.tracker` | Camera configuration, common data access and detection-to-track conversion |
| `stm.ledger` | Per-second visibility, blind spans and parking changes |
| `stm.synopsis` | Time-compressed object replay with source timestamps |
| `stm.diff`, `stm.near` | Parked vehicle comparison and nearby trajectories |
| `stm.flow`, `stm.moments`, `stm.index` | Motion patterns, highlights and colour/class lookup |
| `stm.ask`, `stm.brief` | Rule-based questions and summaries from computed outputs |
| `stm.captions` | Rule-based comparison of caption claims and detector evidence |
| `stm.server`, `stm.export` | Local API/UI serving and static export |

Nearness alone does not establish responsibility. Detector agreement is not independent ground truth. Label staged material and keep footage credits with each camera.
