# Local video search

This configurable example carries the local prototype pipeline into a portable folder: prepare clips from a camera manifest, run optional YOLO tracking, create Cosmos Embed1 video and reviewed-caption vectors, retrieve with SQLite FTS5 plus vector similarity and reciprocal-rank fusion, then optionally send only selected caption evidence to DeepSeek V4.1 Flash through SSH. The bundled fixture captions are invented; no real footage or model output is included.

Copy `config.example.json` to `config.json` and `camera-manifest.example.json` to `camera-manifest.json`. Edit paths relative to each config file. The manifest lists each camera's media file, duration, optional FPS, and optional expected SHA-256; source offsets and segment duration are configurable. Prepare media and captions, then run the stages as needed:

```sh
python ingest_video.py prepare --config config.json
python ingest_video.py detect --config config.json --weights /path/to/yolo.pt
```

`prepare` requires ffmpeg and hashes each source file before cutting it into configurable clips. The camera manifest's declared duration and source clock offset control segment times. `detect` requires the repository's optional `media` dependencies plus Ultralytics, PyTorch, and the configured YOLO checkpoint; it records per-frame detections and camera-local ByteTrack IDs. Track IDs do not represent cross-camera identities.

Create captions with a local vision language model or another reviewed workflow. `luna-ingest-prompt.txt` is a reusable offline prompt for segment-by-segment draft captions. Review every row against the media and keep its exact ID, camera, and clip/source interval from `segments.json`. A caption file must set `reviewed: true` on each row or use a filename containing `reviewed`; a directory can contain per-camera `*-reviewed.json` files. The source segment record must include `source_file` and the SHA-256 generated from the original input. Indexing re-hashes that original media and checks the manifest value. `captions-reviewed.example.json` documents the expected shape (its invented IDs and hashes are only a schema fixture). Caption caches record both source hashes and hashes of the reviewed caption text, so any caption edit requires re-embedding. Then build vectors using the configured local OpenAI-compatible embedding service:

```sh
python embed_videos.py --config config.json
python embed_captions.py --config config.json
python search.py --config config.json --rebuild
```

For Cosmos Embed1, start the repository's local service in its dedicated environment with `python -m vastkit.local.embed_server --host 127.0.0.1 --port 8003`; point `embedding.url` at its `/v1/embeddings` endpoint. That endpoint receives source clips, reviewed captions, and query text, so use a trusted local service; use HTTPS and suitable access controls for a remote endpoint. Set `embedding.model` and `embedding.checkpoint` to match the service. The service processes text and video in the same checkpoint space; remote or differently revised embeddings must not be mixed. The scripts save ordered IDs, source hashes, model/checkpoint provenance, dimensions, and normalized vectors. Search validates IDs, order, hashes, dimensions, and provenance before using caches. If an embedding service or cache is missing or fails, retrieval falls back to FTS5 and labels that mode in the response.

The index is built beside the current SQLite DB and atomically swapped after a successful build. Existing embedding rows are retained only if IDs, source hashes, and caption text match and every vector has valid dimensions and model/revision provenance. A source/caption mismatch or stale vector cache stops indexing rather than silently falling back to raw captions. To intentionally discard vectors, remove the local ignored index and vector cache before rebuilding.

To play selected clips, set `videos` to an explicit map from segment IDs to local MP4 paths. The UI requests `/video/<segment-id>` and supports byte ranges for seeking; it never accepts a client-supplied filesystem path. It seeks to the evidence's `clip_start` within that prepared clip. Paths may be absolute or relative to `config.json`.

To use the real remote model, copy `deepseek_worker.py` to the configured SSH host, set `WANDB_API_KEY` and W&B project attribution in that host's environment, and configure `ssh.target` and `ssh.command` locally. The default worker calls W&B Inference using `deepseek-ai/DeepSeek-V4.1-Flash`; `WANDB_MODEL` and `WANDB_INFERENCE_ENDPOINT` can select a compatible deployment. The local interface binds only to loopback and reports lexical/hybrid retrieval if remote reasoning is unavailable. The worker prompt treats captions as untrusted evidence, cites segment IDs and source-clock intervals, and avoids claims about identity, intent, or theft unsupported by captions.

Install the repository's `vastkit` package for `vastkit.local.embed_server`; use its `media` extra and the separately documented local embed environment for model operations. FTS5 search and the demo server otherwise use the Python standard library. This is a research pipeline, not an accuracy guarantee: captions can omit details, lexical retrieval is literal, visual similarity depends on the checkpoint and footage, and no claim is made that an answer establishes a real-world event. No public benchmark result is claimed by this generic sample.
