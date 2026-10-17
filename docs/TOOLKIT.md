# vastkit reference

Python clients, mocks, local model servers, media helpers and evaluation utilities for the VAST video stack.
Use this guide for SDK setup and API behaviour. The [project README](../README.md) lists the applications and experiments in this repository.

## Layout

| Path | What it is |
|---|---|
| `vastkit/config.py` | Settings from env, `VAST_ENV_FILE`, the single `/config/*.config` on the lab VM, or `./.env`. `describe()` shows set/unset only. |
| `vastkit/http.py` | httpx client factory: optional SOCKS proxy that loopback hosts always bypass, bearer header, bounded retries; keeps W&B hosts out of env proxies. |
| `vastkit/vss.py` | Client for the documented VSS routes (auth, search, ask, tools, explore, stream, upload, delete, reingest, metadata); `synthesis_error()` (a failed Cosmos synthesis inside a 200), `strip_for_log()` (drops `sql_query`). |
| `vastkit/gpu/` | Cosmos3-Reason chat (NVIDIA's official caption and temporal-localisation prompts, `events()` with `EVENTS_SCHEMA` for the latter's output format), Cosmos-Embed1 embeddings plus a plain `cosine()`, YOLO `/v1/infer` with `unique_counts()` / `with_times()`, Canary ASR. |
| `vastkit/llm.py` | OpenAI-compatible chat / JSON-schema / vision helper: W&B Inference by default, `LLM_BASE_URL` + `LLM_API_KEY` for other servers. `MODELS` maps role names (`agent`, `fast`, `vlm`, `vlm_cheap`, `judge`) to default model ids only. |
| `vastkit/evals.py` | Textbook metrics (hit@k, MRR, temporal IoU, `time_match`, P/R/F1), Wilson, bootstrap and `paired_bootstrap_delta` intervals, Cohen's kappa, thin Weave wrappers (`guarded_call`: Weave's guardrail pattern with a caller-supplied scorer). |
| `vastkit/vdb.py` | VastDB session helper, read-only Query Engine SQL lint, `cosine_distance_sql` (the `array_cosine_distance(...)` expression only, no ORDER BY or LIMIT), accident guards for the shared `vss-*` tables, an outbox for failed writes, DuckDB stand-in. |
| `vastkit/s3.py`, `vastkit/naming.py` | S3 `fetch` (local cache keyed by URI), `head`, `list_keys` and a sidecar loader; blueprint segment and sidecar key naming. |
| `vastkit/video.py` | ffprobe, `prep_clip` (cut + downscale + fps cap for Cosmos), browser-safe H.264, 5 s segmenting, lossless `concat`, frame grabs, contact sheets, box burn-in (class and score), WAV extract. |
| `vastkit/frames.py` | Webcam/file frame iterator, raw YOLO detections, FPS benchmark. |
| `vastkit/util.py` | Secret scrubbing, content-hash disk cache, scrubbed JSONL log, timer. |
| `vastkit/mocks/` | Shape-only VSS mock and GPU mock, plus a bland synthetic corpus. |
| `vastkit/local/` | Local servers with the same API as the hosted Cosmos, Embed1 and YOLO endpoints (outage fallback). |
| `python -m vastkit` | `doctor`, `models`, `mock`, `local` (see below); also the `vastkit` command once the package itself is installed (`pip install -e .`; `-r requirements.txt` installs only the dependencies). |
| `deploy/` | `/app` deploy script for the team namespace, a stdlib allowlist proxy and an unstyled deploy-check page. |
| `scripts/`, `.githooks/` | Secret scanner and its pre-commit hook, Windows SSH/SOCKS tunnel, local smoke test, mock-corpus builder. |
| `templates/` | SSH configuration example. |
| `docs/` | `API_NOTES.md` (request/response shapes and quirks from the kit and the blueprint source, and where the mocks differ) and hardware benchmarks. |

## Setup

The package needs Python 3.10+. The exact pins in `requirements.txt` need Python 3.12+ (numpy 2.5.2 and
av 19.0.1 do) and are tested on 3.12 with [uv](https://docs.astral.sh/uv/); on 3.10 or 3.11 install the
pyproject ranges instead (`pip install -e ".[all]"`). In a checkout, on a Windows laptop:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements.txt
```

Install the package from this checkout with `pip install -e ".[all]"`. The core install is
`httpx[socks]` only; the extras are `mock`, `llm`, `vdb`, `media`, `dev` and `all`. The wheel holds
`vastkit` (with `vastkit.gpu`, `vastkit.mocks`, `vastkit.local`) and the `vastkit` command. `deploy/`,
`scripts/`, `templates/` and `docs/` are not installed: use them from a checkout of this repository.

Local GPU stand-ins (optional, NVIDIA GPU with CUDA 12.8; not pyproject extras, they need the CUDA index):

```bash
uv pip install --python .venv -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

```bash
uv venv .venv-embed --python 3.12
```

```bash
uv pip install --python .venv-embed -r requirements-embed.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

Cosmos-Embed1's remote code needs transformers 4.x, so it has its own venv.

On the lab VM, reuse what the image already ships (vastdb, the ADBC driver). If its `python3` is 3.12 or
newer:

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt
```

If it is older (check with `python3 --version`), use the ranges instead:

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e ".[all]"
```

Config: copy `.env.example` to `.env` (gitignored) or to a file outside the repo and point
`VAST_ENV_FILE` at it. On the VM nothing is needed; the single `/config/*.config` is read automatically.
For each setting the env beats the file, and a primary name beats its aliases (a file `VSS_BASE_URL` beats an
env `INGRESS_URL`). On Windows `%USERNAME%` is your Windows account and is ignored, so set `VSS_USERNAME`.

## Commands

```bash
python -m vastkit doctor
```

Checks tools, the local GPU (skipped without torch), which config keys are set (names only; the `env <NAME>`
rows say `env`, `file` or `-` for where a name is set) and endpoint reachability. `--offline` skips the
network. Exits 1 only when a check fails.

```bash
python -m vastkit mock
```

VSS mock on 127.0.0.1:8765 and GPU mock on :8766 in one process (needs the `mock` extra). Set `VSS_MOCK=1`
in the client shell or `.env`: the clients then use the mock URLs and the mock login (`VSS_MOCK_TEAM` /
`VSS_MOCK_PASSWORD` from the shell, default `team-mock` / `mock-password`, the same variables the mock
reads) whatever else is configured, and ignore `VAST_SOCKS`. `GPU_BEARER_TOKEN` and the S3/VastDB access and
secret keys are cleared, so no real token reaches a mock or real storage. With `--vss-port` / `--gpu-port`, also
set the `MOCK_VSS_URL` / `MOCK_GPU_URL` lines it prints.
A busy port is reported and the command exits 1. Each mock also runs alone:
`python -m vastkit.mocks.vss_mock --port 8765`, `python -m vastkit.mocks.gpu_mock --port 8766`.

```bash
python -m vastkit local
```

Starts the local mirrors: Cosmos3-Edge on :8001, YOLO on :8002, Cosmos-Embed1 on :8003 (`--only cosmos yolo
embed` picks some). `LOCAL_*` and `HF_*` keys come from the shell, else from the config file (`--env-file`,
a global option: `python -m vastkit --env-file team.env local`; else the usual lookup; an empty `KEY=` is
ignored): model locations
`HF_HOME`, `LOCAL_YOLO_WEIGHTS`, `LOCAL_COSMOS_MODEL`, and the interpreters `LOCAL_PYTHON` (cosmos, yolo)
and `LOCAL_EMBED_PYTHON` (embed). Without those, it uses `.venv` / `.venv-embed` in the checkout if
present, else the current interpreter. A missing interpreter exits 2 before anything starts; Ctrl+C stops
every child (terminate, then kill after 10 s). The YOLO mirror reports `perception_json.source` from
`PERCEPTION_SOURCE` (shell only, default `yolo11_coco` as hosted). All three together use about 9.5 GB of VRAM.

```bash
python -m vastkit models
```

Lists the model ids served by Cosmos and W&B Inference. Ids change, so check on the day.

```bash
python -m vastkit.frames --source clip.mp4 --weights yolo26s.engine --imgsz 640
```

FPS and latency summary; `--jsonl` also writes per-frame raw detections (its folder is created).

```bash
.venv/Scripts/python.exe -m pytest
```

Runs the tests against the mocks, with no network or credentials needed.

## Client behaviour worth knowing

- `VSSClient`: a 401 triggers one re-login. Reads and searches retry on 502/503/504; `upload`, `reingest`
  and `delete_video` retry only on connect errors, so after a 5xx or timeout check the server state yourself.
- `CosmosReason`: the model id comes from `/v1/models`. `ask_video` answers are cached in
  `~/.vastkit/cache/cosmos` (the key includes the base URL), except for the mock and loopback servers;
  `cache=True/False` overrides. `CosmosResult.dropped` lists requested options (`response_format`,
  `media_io_kwargs`, ...) the server rejected; those answers are never cached. A "fps exceeds video fps"
  400 is retried once just below the clip fps.
- `vastkit.llm`: `WANDB_API_KEY` and the project header go only to `https://api.inference.wandb.ai/v1`.
  Another `base_url` (or `LLM_BASE_URL`) needs `api_key=` or `LLM_API_KEY`; loopback hosts get a placeholder.
- `vastkit.util.scrub`, and so every `JsonlLog` record, redacts values by key name (`password`, `secret`,
  `token`, `api_key`, `access_key`, ... with any prefix and a `_key`/`_id` suffix), bearer tokens, JWTs and
  `sk_`/`wandb_`/`hf_` keys.
- `vastkit.video`: `frames_at`, `keyframes` and `contact_sheet` times are offsets from the stream's first
  frame; odd frame sizes are rounded down to even; `burn_boxes` takes the score as `conf` or `confidence`.

## Mock fidelity

The VSS mock follows the public blueprint source for route names, request validation, response keys,
status codes and error details, JWT behaviour (HS256, 1-day expiry, 401s) and Range/206 streaming. Its
search score is a naive word-overlap placeholder (the share of query words found in a caption):
**ranking and similarity values are not representative**, so never tune anything against it. Streaming
serves bytes only for corpus rows with a local file (`scripts/build_mock_corpus.py` output or uploads); the
default synthetic corpus has none, so it returns 404. The `dashboard/reingest` routes are guessed from the
kit docs and log a warning when hit. The GPU mock returns fixed outputs (hash-seeded embeddings, canned
answers and detections) and reproduces the NIM 400s for `media_io_kwargs`. Fault-injection switches are in
the module docstrings; per-route differences are in [API_NOTES.md](API_NOTES.md).

## Deploy (/app)

On the lab VM: `NS=<team> APP_HOST=video-lab-team-<N>.cosmos.vastdata.com bash deploy/deploy.sh [APP_DIR]`
(default `deploy/app`). It follows the kit's `deploy-app-no-registry` contract; the VSS credentials go in a
Secret, never in the ConfigMap.

`deploy/app/main.py` keeps the VSS JWT server-side and forwards only the (method, route) pairs in
`READ_ROUTES`: auth/me, search, tools/\*, agent/ask, agent/search-and-answer, the videos read routes,
dashboard/stats, metadata/\* and suggestions. Everything else gets 403. DELETE, PUT and PATCH are never
forwarded, and `config`, `auth/login`, `streaming/*`, `batch-sync/*` and any route with a `prefill` segment
(such as `tools/prefill`) are always refused. `WRITE_ROUTES` (upload, dashboard re-ingest and its status
poll) also need `ALLOW_WRITES=1`, `APP_TOKEN` set in the pod and a matching `X-App-Token` header (the
deploy-check page has a field for it):
`ALLOW_WRITES=1 APP_TOKEN=<token> EXTRA_SECRET_KEYS="APP_TOKEN" bash deploy/deploy.sh`.
`/video?source=s3://...` streams a segment with Range support, only from the buckets in `VIDEO_BUCKETS`
(comma-separated); unset, `/video` is 403. The backend's stream route has no ACL (it streams any key the
backend's S3 credentials can read), so list only `<team>-vss-chunks-segments`: `deploy.sh` defaults
`VIDEO_BUCKETS` to `$S3_SEGMENTS_BUCKET` from the sourced config and puts it in the Secret. Allowlisted
reads and `/video` need no token: anyone who can reach the `/app` Ingress can read the team's indexed
captions and the segments in those buckets. A route your app needs
during the event goes into `READ_ROUTES`. Calls that can exceed nginx's 60 s timeout go through
`POST /jobs/api/v1/...` (same guard; returns a 32-hex `job_id`) and `GET /jobs/<id>`. Only the newest 256
jobs are kept in memory; an older id polls as 404.

## Lab access from a laptop

- `templates/ssh_config.example`: the `bc-<username>` host through `cloudflared access ssh` (cloudflared on PATH).
- `scripts/tunnel.ps1 -Alias bc-<username>` opens `ssh -N -D 127.0.0.1:1080` in its own window
  (Windows OpenSSH has no `-f`). Set `VAST_SOCKS=socks5h://127.0.0.1:1080` in the client shell or `.env`:
  vastkit's HTTP clients then use it for every host except loopback ones (local mirrors, mocks), and the
  W&B client never does. Never set it as a global `HTTPS_PROXY`.
- VastDB SQL needs the Linux ADBC driver: run it on the VM.

## Measured (RTX 4080 Laptop, 12 GB, Windows 11)

| What | Result |
|---|---|
| yolo26s TensorRT FP16, pure inference | 2.3–3.1 ms; 106.6 fps at 1080p, 146.3 fps at 360p end to end ([bench](bench_yolo_rtx4080.json); its `track=True` rows time ultralytics' built-in tracker, which vastkit does not use) |
| `vastkit.frames` on a 1080p60 file | 56.9 fps every frame; 60.1 fps paced at source rate with 2/900 dropped ([bench](bench_frames.json)) |
| Cosmos3-Edge (local) | ~11 s load, 5.8 GB VRAM, 7–9 tokens/s |

## Rules this repo follows

- Credentials never go in the repo, logs or output. `/config` output, `sql_query` and stream URLs (they
  carry the token) are never logged.
- `scripts/check_secrets.py` scans the staged content (what `git commit` records) plus untracked, non-ignored
  files; once files are committed, run `scripts/check_secrets.py --all` (every tracked and untracked,
  non-ignored file) before zipping or pushing a submission. A directory argument follows `.gitignore` inside
  the repo. It prints file:line and the kind of hit only (one per line). A line that holds no secret can
  carry the marker `secret-scan: allow` in a comment (`# secret-scan: allow`, `<!-- secret-scan: allow -->`)
  and is then skipped. The scan also fails on unreadable files and on files over 5 MB ("too large to
  scan"), except a `.pdf` over 5 MB (noted, and the scan passes) and the image, video, audio, weight,
  compressed-archive and Office suffixes listed in `SKIP_SUFFIX`, which are never read. Smaller `.pdf` files
  are scanned like text, an uncompressed `.tar` member by member (hits print as `bundle.tar:conf/team.env:1`).
  Credential names are any `NAME` ending in `PASSWORD`, `PASSWD`, `SECRET`, `SECRET_KEY`, `ACCESS_KEY`,
  `API_KEY` or `TOKEN` (`APP_TOKEN`, `HF_TOKEN`, ...); in env-style files also `_PASS`, `_PW`, `_PWD` and
  `_KEY` (with a 16+ character value of letters and digits). How an unquoted value is read depends on the
  file. In env-style files (`.env*`, config, INI, TOML and YAML files, shell and batch scripts, Dockerfiles),
  and in any file on a line that starts with `export`, `%env`, batch `set`/`setx`, Dockerfile `ENV`/`ARG` or
  a YAML list item `- NAME=value`, or after `-e`/`--env`, `--from-literal=` or `echo`, it counts as a literal
  unless it expands something: it is one whole `$X`, starts with `$(...)`, `%X%` or `%(x)s`, contains
  `${...}`, or is one whole call such as `lookup(x)`. Only in `.sh`/`.ps1` scripts, Dockerfiles and on those
  prefixed lines (not `set`/`setx` or YAML list items) does a longer value starting with `$X` (`$HOME/pw`)
  count too; in a `.env` or batch file it is a literal (write `${HOME}/pw`). In source code (`.py`, `.js`,
  `.ts`, ...) an unquoted value is code, so only quoted values count there; in other files (Markdown, text,
  notebooks) a short code heuristic decides. In any file it also flags `NAME: type = "..."`,
  `os.environ[...] = "..."` and `setenv`/`setdefault` with a literal, `user:password@` in a URL,
  `curl -u user:password`, JWTs, bearer tokens and AWS, HF, W&B, NVIDIA (`nvapi-`) and OpenAI-style (`sk-`)
  keys. The full rules are in the script's docstring (`scripts/check_secrets.py --help`).
  Enable the hook with `git config core.hooksPath .githooks` (it needs `python3` or `python` on PATH).
  Commits made on Windows drop the exec bit: after the first `git add`, run
  `git update-index --chmod=+x deploy/deploy.sh .githooks/pre-commit`; on a clone without it, run
  `chmod +x .githooks/pre-commit`, or git silently skips the hook.
- Never ingest internet video into VSS. `VSSClient.upload()` needs `i_own_this_video=True`, and even then
  refuses a path where a folder or the file name contains a sample-footage hint (`INTERNET_SAMPLE_HINTS`,
  such as pexels, youtube or sample-videos, matched as substrings, so expect the odd false positive; the
  short `intel` only as a whole word, so `intelligence` or `IntelliJ` folders pass),
  anything under the toolkit's `data/` when vastkit runs from a source checkout, and the mock corpus named by
  `VSS_MOCK_CORPUS`. With a wheel install a clone's `data/` (where `scripts/build_mock_corpus.py` writes) is
  not covered: set `VSS_MOCK_CORPUS` to that corpus in the client's shell, or `pip install -e` the clone.
  `scripts/build_mock_corpus.py` output stays local: by default it goes to the checkout's gitignored `data/`
  whatever the working directory.
- Never write to the shared `vss-collection` / `vss-prompts-events` tables. vastkit guards against doing it
  by accident (not a security boundary: a determined caller can still reach the SDK table). `vdb.insert` and
  `vdb.ensure_table` refuse them, any other `vss-*` table, the `vss-schema` schema and whatever
  `VDB_SCHEMA` / `VDB_COLLECTION` / `VDB_PROMPTS_COLLECTION` name, comparing table and schema names
  case-insensitively; `vdb.table()` returns those behind a proxy that passes only the read attributes in
  `vdb.READ_ONLY_ATTRS` (other public names raise PermissionError), and `vdb.qe_query` / `vdb.lint_sql` accept
  one read-only SELECT/WITH statement: no `;` (not even a trailing one) or `$` outside string literals, so
  dollar-quoted strings are not supported, and no backslash inside quotes, letter right before a quote
  (`E''` / `U&''` escape strings; write `TIMESTAMP '...'` with the space) or nested `/*` comment.
- Do not redeploy the DataEngine functions or the graph; stay within the team namespace.

## Source references

The VAST build kit and NVIDIA VSS blueprint are API references for the client routes, segment and sidecar naming,
perception counts and mock response shapes. Code comments cite the source files; [API_NOTES.md](API_NOTES.md)
records the request and response details.

`vastkit/gpu/cosmos.py` uses NVIDIA's official caption and temporal-localisation prompts from the Cosmos Reason2
repository, plus the model card's `<think>` / `<answer>` output format. `EVENTS_SCHEMA` describes the temporal
prompt's start/end/caption response. `scan_events()` can recover those items from broken JSON. Other generic
instructions cover continuation after a truncated answer, mock synthesis, local JSON formatting and smoke checks.

The VastDB vector-predicate workaround and ADBC driver discovery paths also cite the blueprint source. These
helpers do not create a project ranking policy or evaluation rubric; those belong to the application.

## Licences

- `vastkit` itself has no LICENSE file and no licence field in `pyproject.toml`: no licence is granted,
  so all rights are reserved by its author. The repository does not grant a software licence.
- `ultralytics` (used by `vastkit.frames` and `vastkit/local/yolo_server.py`) is AGPL-3.0. It is optional:
  the hosted YOLO endpoint needs only `vastkit.gpu.yolo`.
- Model weights (Cosmos3-Edge, Cosmos-Embed1, YOLO) are not in this repo; each has its own licence on its model card.
