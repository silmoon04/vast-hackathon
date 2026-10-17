# Find Desk

Search indexed footage for an adult's clothing, review the matching clips, and confirm sightings to build a route. The person using the app confirms each sighting. Similar clothing alone does not establish identity, and gaps between cameras stay unknown.

The server uses Python's standard library. When configured, DeepSeek V4.1 Flash
turns the description and refinements into one text-search tool call through the
provided W&B API. The app validates that call, adds its fixed camera and dataset
scope, then executes VSS `POST /api/v1/agent/search-and-answer`. VSS retrieves
indexed captions and embeddings and generates an answer from those captions.
If planning fails, VAST searches the original description. Optional push-to-talk
uses the provided Canary service separately. This app does not track a person live.

Set `VSS_URL`, `VSS_USERNAME`, and `VSS_PASSWORD` on the server, using the existing team Secret. `PORT` defaults to `8080`. Keep credentials out of the browser and `dataset.json`.

For speech input, set `CANARY_URL` to the provided endpoint's base URL and `GPU_BEARER_TOKEN` from a server-side Secret. Speech stays unavailable if either value is missing. The app sends a multipart `file` and `language=en-US` to `/v1/audio/transcriptions`, without a `model` field. It keeps audio in memory, accepts only mono 16 kHz PCM16 WAV recordings of up to 20 seconds and 1 MiB, processes one recording at a time, and waits up to 30 seconds. It does not follow provider redirects. The returned text is editable before the user searches.

The flat deployment needs `main.py`, `index.html`, `app.js`, `style.css`, and a `dataset.json` manifest. The manifest defines the footage scope:

```json
{
  "title": "A short route",
  "ready": false,
  "auto_ready": true,
  "location": "route-demo-20261017",
  "license": {
    "name": "License name",
    "url": "https://example.org/rights",
    "attribution": "Required attribution",
    "summary": "What this footage permits"
  },
  "cameras": [
    {
      "id": "cam-a",
      "label": "Entrance",
      "original_video": "s3://team/demo/entrance.mp4",
      "duration_seconds": 60,
      "clock_start_seconds": 36000,
      "source_url": "https://example.org/footage"
    }
  ]
}
```

Use each camera's `id` as its VSS `camera_id`. Ingest every clip with the manifest's `location` and the same value in its tags. With `auto_ready: true`, a background worker checks the scoped VSS complete catalogue every 30 seconds. Search becomes available only after every declared video is complete at that location. A changed manifest or a failed check keeps search unavailable until a successful check. These checks only read the index; they do not start ingestion or write the manifest. Without `auto_ready`, set `ready` only after the clips have finished indexing. Until then the UI shows "Footage preparing".

The search request always includes that location and tag. The server also checks every returned `original_video` against the manifest before showing an answer or caching playback. An unexpected source rejects the result. A failed answer-generation step still permits review of scoped clips and displays an error. Video URLs contain opaque IDs; the server keeps storage keys and tokens private and checks manifest membership again on playback.

API routes:

- `GET /health`: process health and whether VSS credentials are configured.
- `GET /api/dataset`: public camera and license details. Storage keys are removed. `/dataset.json` returns the same sanitized view.
- `GET /api/speech`: whether Canary push-to-talk is configured. This is independent of footage readiness.
- `POST /api/transcribe`: a raw `audio/wav` body with a known `Content-Length`. Returns `text`, `latency_ms`, `provider: "Canary"`, and `editable: true`. The browser sends a separate search after the user reviews the text.
- `POST /api/search`: `{"description":"Adult wearing blue","refinements":["carrying a bag"],"camera_id":"cam-a"}`. The camera is optional. Returns HTTP 202 with `job_id`.
- `GET /api/jobs/<job_id>`: `running`, `done` with a `result`, or `error` with a short `detail`. Results include candidates, an answer, answer-generation status, and platform timings.
- `GET /video?id=<candidate_id>`: a server-side video proxy with byte-range support. Candidate times are relative to this full video.

Two searches may run at once. The server keeps the newest 64 jobs and 512 playback IDs in memory. Restarting it clears those caches. Descriptions are limited to 400 characters, with up to six refinements of 200 characters each.

Run the scoped backend checks from the repository root:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_find_desk.py
```

For deployment, copy this directory to the lab VM and provide a runtime manifest
containing the indexed source URIs. The checked-in example stays unready until
real indexed sources are supplied. The runtime `dataset.json` is ignored by Git.

```sh
python3 deploy.py --dataset dataset.json --host video-lab-team-13.cosmos.vastdata.com
```

The deployment uses the existing `atlas-field-desk-vss-creds` Secret and publishes
only `/find/`, leaving Field Desk and Street Time Machine on their own routes.
It runs one CPU container with a 256 MiB memory limit. It does not deploy a model
or change ingestion. Kubernetes restart clears search jobs, so saved browser
routes need a new search to refresh their playback links after a restart.

Optional server credentials are read from `atlas-find-desk-agent`
(`WANDB_API_KEY`, `WANDB_PROJECT_PATH`) and `atlas-find-desk-speech`
(`CANARY_URL`, `GPU_BEARER_TOKEN`). The router pins
`deepseek-ai/DeepSeek-V4.1-Flash` and the W&B inference endpoint. Its only tool
argument is search text; generated SQL, source URLs and scope changes are never
accepted. These credentials stay on the server. VSS and the two provided model
services remain separate parts of the application.

The first demo uses the first 60 seconds of cameras 0001 and 0009 from
`MTMC_Tracking_2025/val/Lab_000`, resized to 720p. NVIDIA publishes this synthetic
dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Keep the [dataset credit and source](https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces)
visible when recording or broadcasting the demo. Ground truth is used separately
for validation and is not sent to the agent as evidence.
