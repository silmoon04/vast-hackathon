# VAST stack API notes (request/response shapes, compiled Fri 16 Oct 2026)

Compiled from the public VAST build kit, the VSS blueprint source and the event docs. Each entry gives the request/response shape, quirks and a confidence tag (confirmed-in-source / documented-in-skill / inferred). Verify anything tagged inferred against the configured deployment before relying on it. MOCK notes describe the shipped mocks (`vastkit/mocks/vss_mock.py`, `vastkit/mocks/gpu_mock.py`) and say where they differ from the source.

### vss-backend POST /api/v1/auth/login | auth: none | confirmed-in-source

REQ: JSON {username: str (required), password: str (required)}. No tenant or VMS fields; the backend takes those from its own config (vast_host, tenant_name).

RESP: 200 {access_token: str, token_type: "bearer", username: str (echo of request)}. access_token is an HS256 JWT with claims {sub: username, exp: iat + 1 day, iat, role: "user"}. There is no refresh endpoint.

QUIRKS: 401 {"detail":"Invalid username or password"}. 422 is the FastAPI validation error when a field is missing.
The backend checks credentials against VMS: first /api/token/{tenant}/, then /api/token/.
If the backend vast_host is empty, ANY credentials succeed (auth_service.py:40-42).
On 401 "Token has expired", log in again (the token lasts 1 day).
MOCK: same shape. Signs an HS256 JWT {sub, exp, iat, role: "user"} with a local secret (VSS_MOCK_JWT_SECRET); expiry is VSS_MOCK_TOKEN_TTL seconds (default 1 day). Any username works; a password other than VSS_MOCK_PASSWORD (default mock-password) gets the 401 unless VSS_MOCK_ANY_PASSWORD=1.

SRC: Path keys: SP = <ref> (read-only clones of the public VAST kit and VSS blueprint repos, used as citations; not shipped); BE = SP/repo_vssbp/src/source-code/retrieval/video-backend/src; SK = SP/repo_vbc/src/.cursor/skills; MOCK = vastkit/mocks/vss_mock.py; GPU MOCK = vastkit/mocks/gpu_mock.py. Sources: BE/api/v1/auth.py:14-39; BE/schemas/auth.py:7-17; BE/services/auth_service.py:39-66 (VMS check), 130-139 (JWT); SK/retrieval/login/SKILL.md:15-40

### vss-backend GET /api/v1/auth/me | auth: Bearer JWT. The "Bearer " prefix is optional: a raw token in the Authorization header also works (auth_service.py:145-163). | confirmed-in-source

REQ: No body. Header: Authorization: Bearer <jwt>.

RESP: 200 {username: str, email: str|null, auth_type: str}. auth_type is the JWT role claim, default "user".

QUIRKS: 401 responses carry the header WWW-Authenticate: Bearer. Details:
- "Could not validate credentials" (no header)
- "Invalid token"
- "Token has expired"
This is the cheapest authenticated health check.
normalized_username (models/user.py:15-22) lowercases and replaces '.' and ' ' with '_'. The ACL SQL and the upload object_key use the raw username.
MOCK: matches (sub as username, email null, auth_type from the role claim; same 401 details and header, raw tokens accepted).

SRC: BE/api/v1/auth.py:42-51; BE/schemas/auth.py:20-24; BE/services/auth_service.py:68-96,145-163; BE/models/user.py:8-22

### vss-backend GET /health  and  / | auth: none | confirmed-in-source

REQ: none

RESP: /health returns {"status":"healthy"}.
/ returns {name: app_name (default "Video Reasoning API"), version: "1.0.0", status: "running"}.

QUIRKS: The backend Ingress only routes the path prefix /api (backend-deployment.yaml:91-92). So $INGRESS_URL/health is probably not routed; only in-cluster port 8000 serves it.
From a client, use GET /api/v1/auth/me or /api/v1/config as the health probe.
FastAPI docs were not disabled, so /openapi.json exists, but only in-cluster (via kubectl port-forward svc/video-backend-service 8000).
MOCK: both routes, same bodies, no auth.

SRC: BE/main.py:51-64; SP/repo_vssbp/src/deployments/vss-k8s-application/backend-deployment.yaml:39-53,86-97

### vss-backend GET /api/v1/config | auth: Bearer JWT | confirmed-in-source

REQ: No params. Authorization: Bearer <jwt>.

RESP: 200 with these sections:
- s3: {... keys masked}
- vastdb: {...}
- embedding: {...}
- synthesis: {cosmos_host, cosmos_port, cosmoshttpscheme, cosmos_model, cosmos_temperature, synthesis_max_tokens, synthesis_timeout_seconds, default_system_prompt}
- app: {app_name, max_upload_size_mb, allowed_video_extensions, cors_origins, display_timezone}

QUIRKS: Secrets are masked (config.py:14), but the response still exposes internal hostnames, so never log or print it.
max_upload_size_mb: code default 25 (config.py:87); the secret template uses 100 (backend-secret.yaml.example:67).
allowed_video_extensions = [.mp4, .mov, .webm, .avi, .mkv].
Use app.max_upload_size_mb to pre-check uploads (SK/ingest/upload-video/SKILL.md).
MOCK: same sections with "***" secrets and placeholder hostnames; max_upload_size_mb is VSS_MOCK_MAX_UPLOAD_MB (default 25).

SRC: BE/api/v1/config.py:14-78; BE/config.py:86-113

### vss-backend POST /api/v1/search | auth: Bearer JWT | confirmed-in-source

REQ: JSON VideoSearchRequest:
- query: str (required)
- top_k: int, default 15, range 1..100
- tags: list[str] = []
- include_public: bool = true
- public_only: bool = false
- system_prompt: str|null (overrides the synthesis system prompt)
- time_filter: str = "all". One of all|5m|15m|1h|24h|7d|custom. Unknown values silently apply no filter.
- custom_start_date, custom_end_date: ISO str|null. Used when time_filter = custom. 'Z', '+00:00' and milliseconds are stripped, then compared as a naive TIMESTAMP.
- metadata_filters: dict[str, scalar] = {}
- min_similarity: float, default 0.1, range 0..1
- llm_top_n: int, default 3, range 1..100. Number of top chunks sent to Cosmos.
- hybrid_text_weight: float|null. null means the server default 0.6; values are clamped to 0..1.

RESP: 200 VideoSearchResponse:
- results: VideoSearchResult[] (≤top_k)
- chunk_results: ChunkSearchResult[] (≤top_k)
- total: int (= len(results))
- chunk_total: int
- query: str
- embedding_time_ms: float
- search_time_ms: float
- permission_filtered: int
- llm_synthesis: {response: str (markdown), segments_used: int, segments_analyzed: str[] (labels like '<file> (best moment M:SS–M:SS)'), model: str, tokens_used: int, processing_time: float, error: str|null} | null
- sql_query: str|null

VideoSearchResult:
- filename
- source: s3 URI of the segment mp4
- reasoning_content
- is_public
- upload_timestamp: naive ISO datetime
- duration
- segment_number, total_segments
- segment_start_sec, segment_end_sec
- original_video: parent chunk s3 URI
- tags[]
- similarity_score
- optional: cosmos_model, tokens_used, cached_prompt_tokens, camera_id, capture_type, location, perception_json, object_classes (comma-joined str), object_counts (JSON str), max_detection_conf, perception_ok, perception_source, detection_sidecar_uri, detection_frame_count, detection_count, extra_metadata (JSON str)

ChunkSearchResult:
- original_video, filename
- chunk_duration_sec, total_segments
- similarity_score
- best_segment_number, best_match_start_sec, best_match_end_sec
- preview_source: the segment URI to play
- reasoning_content
- is_public, upload_timestamp, tags
- matched_segment_count
- query
- timeline: TimelineSegment[], each {segment_number, segment_start_sec, segment_end_sec, source, reasoning_content?, object_classes?, object_counts?, perception_ok?, similarity_score = 0, is_search_match = false, query_highlight = false, is_best_match = false}
- optional: camera_id, capture_type, location, cosmos_model, tokens_used, cached_prompt_tokens, stream_id, chunk_index, stream_chunk_total, prev_chunk_index, next_chunk_index

QUIRKS: Errors:
- Any exception returns 500 {"detail":"Search failed: ..."} (search.py:128-133). This includes embed and Query Engine outages.
- 422 on validation.

Synthesis failure is NOT an HTTP error. You get 200 with llm_synthesis.error set and response "Failed to generate AI synthesis: ..." (llm_service.py:135-151, search.py:104-113). Clients must check llm_synthesis.error.
Synthesis runs only when chunk_results is non-empty; with no results, llm_synthesis is null.

Hybrid scoring:
- The query text is embedded once per branch, both times as a Cosmos-Embed1 TEXT query.
- The text branch is cosine against `vectors`; the visual branch is cosine against `vectors_visual`.
- similarity = 1 - distance.
- min_similarity is applied per branch BEFORE the merge (vastdb_service.py:452-460).
- Merge: w*text + (1-w)*visual, with a missing branch counting as 0. Final scores can therefore fall below min_similarity.
- Segment fetch k = min(max(top_k*5, top_k), 100) (search.py:55).

SQL quirks:
- Tags are interpolated unescaped (vastdb_service.py:307-309), so a tag containing ' causes a 500.
- metadata_filters accepts ANY column name. Keys are injected unquoted after stripping quotes and ';'.
  - bool becomes TRUE/FALSE; numbers are bare.
  - object_classes uses a LOWER(...) LIKE '%v%' substring match.
  - List values are unsupported.
- ACL: public_only gives is_public = TRUE; include_public = false gives allowed_users && ARRAY['<username>']; otherwise both are OR'ed.
- sql_query embeds the full 256-float vector literal, so it is large; do not log it.

The kit says there are no /tags or /locations routes (SK/retrieval/README.md:33-35).
MOCK: same request validation and response keys, chunk grouping and timeline, ACL, time_filter and metadata_filters handling (unknown column → 500, object_classes substring match), tag with ' → 500. Differences:
- similarity_score is a placeholder: the share of query words (minus stop words) found in reasoning_content + object_classes. min_similarity applies to it; hybrid_text_weight is accepted and ignored. Ordering and values are not representative.
- Synthesis is a fixed template, or a call to MOCK_LLM_URL (OpenAI-compatible, model MOCK_LLM_MODEL) when set; MOCK_SYNTH_FAIL=1 gives the 200-with-error shape.
- sql_query is a short placeholder with a zero vector; embedding_time_ms is fixed.

SRC: BE/schemas/search.py:10-55; BE/api/v1/search.py:17-133; BE/models/video.py:9-91; BE/services/vastdb_service.py:263-309 (SQL), 312-350 (time), 353-371 (metadata filters), 452-460 (similarity), 566-583 (ACL), 586-623 (hybrid merge), 634-781 (chunk grouping), 802-824 (time presets); BE/services/llm_service.py:66-151; BE/services/embedding_service.py:21-37; SK/retrieval/search/SKILL.md:17-67

### vss-backend POST /api/v1/agent/ask | auth: Bearer JWT | confirmed-in-source

REQ: JSON {question: str (min_length 1), original_video: str|null (parent s3 URI), top_k: int, default 10, range 1..50}.

RESP: 200 {answer: str, tool_used: "video_segments"|"search_hybrid", evidence: object}.
- With original_video: evidence = {segment_count: int, segments_used: int}.
- Global: evidence = {result_count: int, top_sources: str[≤5]}.

QUIRKS: 404 "No accessible segments found for video: <uri>" (agent.py:45-49).

LIKELY BUG (inferred; verify against the deployed version):
- In global mode, agent.py:81-82 calls search_response.llm_synthesis.get("response").
- But llm_synthesis is a pydantic v2 model: pydantic 2.5.3 coerces the dict, and the model has no .get.
- So any query with at least one hit raises AttributeError and an unhandled 500. The body is plain text "Internal Server Error", not JSON.
- Clients should prefer /agent/search-and-answer, which handles both shapes (agent.py:98-110), and must not assume error bodies are JSON.

In original_video mode:
- It uses the FIRST top_k segments by segment_number (not ranked), each with similarity_score fixed at 1.0 (agent.py:51-61). Long videos are truncated.
- A synthesis failure still returns 200 with answer "Failed to generate AI synthesis: ...".

There is no /videos/ask route (SK/retrieval/agent-qa/SKILL.md:12).
MOCK: shapes, the 404 and the original_video mode (first top_k segments, similarity_score 1.0, real segment_count) match. Global mode answers normally unless MOCK_ASK_BUG=1, which reproduces the plain-text 500 whenever there is a hit.

SRC: BE/api/v1/agent.py:20-32,35-95; BE/schemas/search.py:53 (llm_synthesis typed Optional[LLMSynthesisResponse]); SK/retrieval/agent-qa/SKILL.md:14-28

### vss-backend POST /api/v1/agent/search-and-answer | auth: Bearer JWT | confirmed-in-source

REQ: Identical to POST /api/v1/search (VideoSearchRequest).

RESP: 200 {answer: str, tool_used: "search_hybrid", evidence: {...}}. evidence fields:
- query
- segment_hit_count: int
- chunk_count: int
- permission_filtered: int
- embedding_time_ms: float
- search_time_ms: float
- chunks: up to 10 items of {original_video, filename, similarity_score, best_match_start_sec, best_match_end_sec, matched_segment_count, preview_source}
- top_segment_sources: str[≤5]
- llm_synthesis: LLMSynthesisResponse as a dict, or null

QUIRKS: Answer fallback order: llm_synthesis.response, then chunk_results[0].reasoning_content, then results[0].reasoning_content, then "No matching segments found."
If synthesis failed, the answer is the "Failed to generate AI synthesis: ..." string, so check evidence.llm_synthesis.error.
500 "Search failed: ..." propagates from search.

llm_top_n conflict:
- The source has ge=1, so llm_top_n = 0 gives 422.
- The kit's reingest-chunk skill sends llm_top_n:0 to /search (SK/ingest/reingest-chunk/SKILL.md:124), so the event backend may be newer.
- Client: send ≥1, or on 422 retry with 1.

MOCK: matches, including the answer fallback order, auth and the 422 for llm_top_n = 0 (ge=1, as in the source).

SRC: BE/api/v1/agent.py:98-159; SK/retrieval/agent-qa/SKILL.md:30-44

### vss-backend GET /api/v1/tools/segments | auth: Bearer JWT | confirmed-in-source

REQ: Query original_video=<parent s3 URI> (required).

RESP: 200 {original_video: str, total: int, segments: [dict]}. Each segment dict has:
- filename, source, reasoning_content
- segment_number, total_segments
- segment_start_sec, segment_end_sec
- original_video
- object_classes, object_counts
- perception_ok, perception_source
- detection_sidecar_uri, detection_frame_count, detection_count
- is_public, upload_timestamp, tags
- camera_id, capture_type, location
- cosmos_model, tokens_used, cached_prompt_tokens
- duration
Segments are sorted by segment_number and deduped.

QUIRKS: ACL-filtered. No access or no rows gives total 0 (inferred; no 404 seen).
The timeline is fetched with OR chains because IN is rejected by the Query Engine (vastdb_service.py:662).
It returns every caption of one video in a single call.
MOCK: tools/segments, segment, detections, search, synthesize and explore are implemented over the mock rows, with the same 404s.

SRC: BE/api/v1/tools.py:40-52; BE/services/vastdb_service.py:1395-1435 (dict keys), 1446-1495 (OR-based fetch, sort, dedupe)

### vss-backend GET /api/v1/tools/segment | auth: Bearer JWT | confirmed-in-source

REQ: Query source=<segment s3 URI> (required).

RESP: 200 {segment: VideoSearchResult as dict}.

QUIRKS: 404 "Segment not found".
The lookup is ADBC `WHERE source = '<source>' LIMIT 1`, unescaped, and returns None on ANY error (vastdb_service.py:1068-1197). So a Query Engine outage looks like a 404.

SRC: BE/api/v1/tools.py:55-65

### vss-backend GET /api/v1/tools/detections  |  POST /api/v1/tools/search  |  POST /api/v1/tools/synthesize  |  GET /api/v1/tools/explore | auth: Bearer JWT | confirmed-in-source

REQ: Same as the corresponding routes:
- tools/detections → /videos/detections (source=)
- tools/search → /search body
- tools/synthesize → /videos/synthesize body
- tools/explore: scope=all|mine|public, date, location, limit = 48 (1..100), offset. There is NO `indexed` param.

RESP: Same response models as /videos/detections, /search (VideoSearchResponse), /videos/synthesize (VideoSynthesizeResponse) and /videos/explore (ExploreResponse).

QUIRKS: tools/explore calls the videos explore function directly without `indexed`. The FastAPI Query() default object is then passed through instead of the string "complete" (inferred). It may behave as partial, or fail.
Prefer /videos/explore?indexed=complete.
MOCK: tools/explore has no `indexed` param either and always behaves as indexed=complete.

SRC: BE/api/v1/tools.py:31-37,68-74,77-100,103-109

### vss-backend GET /api/v1/videos/explore | auth: Bearer JWT | confirmed-in-source

REQ: Query:
- scope = all|mine|public (default all; pattern-validated, else 422)
- date: str|null, YYYY-MM-DD matched against the upload day
- location: str|null, matched against the location label; blank locations get the label "(empty)"
- indexed = complete|partial (default complete)
- limit = 48, range 1..100
- offset = 0, ≥0

RESP: 200 ExploreResponse:
- chunks: ChunkSearchResult[], built from rows; similarity_score and query are not meaningful here
- total: int
- uploads_by_day: [{date, chunk_count}], newest first
- locations: [{label, chunk_count}]
- scope
- selected_date: str|null
- selected_location: str|null
- indexed: str
- limit, offset
- table_available: bool = true
- table_message: str|null

QUIRKS: indexed=complete means that, after dedupe, the rows cover every slot 1..max(total_segments) (segment_timeline.py:31-55). New or re-ingesting parents appear ONLY under indexed=partial until they are complete.
Results are sorted by upload_timestamp descending, then paged. Use limit=100 and keep increasing offset (SK/ingest/reingest-videos).
Rows come from a full SDK scan cached 45 s per worker (row_cache.py:12, 75-77). With uvicorn --workers 4 (Dockerfile), consecutive calls can disagree for up to 45 s.
Duplicate segment rows left by re-ingest are deduped: newest upload_timestamp wins, then the longest reasoning_content.
Stream labels (stream_id, chunk_index, stream_chunk_total, prev_chunk_index, next_chunk_index) come from extra_metadata.
On a missing table you get 200 with table_available = false, not an error.
MOCK: same params, shape, complete/partial rule, sort and paging. No 45 s cache, and it never holds duplicate rows (mock re-ingest rewrites rows in place).

SRC: BE/api/v1/videos.py:475-515; BE/schemas/explore.py (ExploreResponse); BE/utils/explore_index.py:12-86; BE/utils/browse_chunk.py:55-82; BE/utils/segment_timeline.py:7-55; BE/utils/stream_index.py:70-110; BE/utils/row_cache.py:12, 75-77; BE/services/vastdb_service.py:1629-1717

### vss-backend GET /api/v1/videos/stream | auth: ?token=<jwt> query param | confirmed-in-source

REQ: Query: source = s3://bucket/key (required), token = <jwt> (required; the Authorization header is NOT used).
Optional headers: Range: bytes=a-b, If-None-Match.

RESP: Video bytes. 200 for the full file, or 206 with Content-Range and Content-Length.
Headers: Accept-Ranges: bytes; Cache-Control: private, max-age=3600; Content-Disposition: inline; ETag.
304 when If-None-Match matches. 416 with Content-Range: bytes */N for an unsatisfiable range.

QUIRKS: Errors:
- 401 "Invalid or expired token"
- 400 "Invalid S3 source URL format"
- 400 "Invalid S3 URL format. Expected: s3://bucket/key"

There is NO ACL check on source: any authenticated user can stream any key the backend's S3 credentials can read. Do not rely on it for privacy.
The token in the URL ends up in proxy logs, so never log built URLs.
Laptop and judge browsers cannot reach INGRESS_URL. A deployed app should proxy this route same-origin, adding the token server-side and passing Range through.
MOCK: same auth, errors, headers, Range/206/416 and ETag/304, and no ACL check. It serves bytes only for rows with a local file (rows from scripts/build_mock_corpus.py, or mock uploads); anything else is 404 "Object not found in mock ...". The default synthetic corpus has no files, so every stream is 404 there.

SRC: BE/api/v1/videos.py:166-277; BE/services/s3_service.py:32-58,410-484

### vss-backend GET /api/v1/videos/playback-url | auth: ?token=<jwt> query param | confirmed-in-source

REQ: Query: source = s3://bucket/key, token = <jwt>, expires_in = 3600 (range 60..86400).

RESP: 200 {url: str (presigned S3 GET), expires_in: int}.

QUIRKS: 401, 400 (same messages as /stream), and 500 "Failed to generate playback URL: ...".
The presigned host is the backend's s3_endpoint (the private data VIP), so a laptop browser cannot fetch it. <video> also needs bucket CORS.
Prefer /videos/stream through a proxy.
MOCK: same params and 401/400s; the url is a fake http://mock-data-vip/... link that cannot be fetched.

SRC: BE/api/v1/videos.py:280-329; BE/services/s3_service.py:265-289

### vss-backend GET /api/v1/videos/metadata | auth: Bearer JWT | confirmed-in-source

REQ: Query source = <segment s3 URI> (required).

RESP: 200: one VideoSearchResult as a dict, with all fields listed under /search. similarity_score is not meaningful here.

QUIRKS: 404 "Video not found or you don't have permission to access it". This covers no row, no access, AND any Query Engine error (the lookup swallows exceptions).
The source is interpolated into SQL unescaped.
The original scenario and custom_prompt are not in the row (SK/ingest/reingest-videos/SKILL.md:107-112).
MOCK: same shape and 404 (similarity_score 0.0).

SRC: BE/api/v1/videos.py:616-652; BE/services/vastdb_service.py:971-1066 (ACL), 1068-1197 (lookup)

### vss-backend GET /api/v1/videos/detections | auth: Bearer JWT | confirmed-in-source

REQ: Query source = <segment s3 URI> (required).

RESP: 200: the raw detection sidecar JSON object, as written by the ingest detector:
- source: "yolo11_coco"
- segment_source: str
- video_shape: [h, w]
- fps: float
- frame_count: int
- detection_count: int
- object_classes: str[]
- object_counts: {label: int}
- max_detection_conf: float
- frames: [{frame_index: int, time_sec: float, shape: [h, w], detections: [{label: str, confidence: float (4 dp), bbox: [x1, y1, x2, y2] int pixels, xyxy}]}]

QUIRKS: Errors:
- 404 "Segment not found"
- 404 "No detection sidecar for segment"
- 500 "Failed to load detections: ..." when the sidecar is not a JSON object

.gz sidecars are gunzipped server-side.
fps is estimated as frame_count / segment_duration (default 30), so time_sec is approximate.
The sidecar's object_counts is a unique-object estimate (max per frame), not raw detection totals.
MOCK: the same sidecar shape, synthesised (10 frames, labels from the row's object_counts, boxes and confidences made up); both 404s; auth required.

SRC: BE/api/v1/videos.py:581-613; SP/repo_vssbp/src/source-code/ingest/video-detector/common/clients.py:128-183 (normalize), 186-231 (sidecar payload), 234-249 (gzip, key = detections/<stem>.json.gz); SP/repo_vssbp/src/scripts/vss-blueprint-models/yolo-infer/main.py:77-91 (bbox)

### vss-backend POST /api/v1/videos/synthesize | auth: Bearer JWT | confirmed-in-source

REQ: JSON VideoSynthesizeRequest:
- original_video: str (min length 1)
- question: str, default: a short prompt asking for a chronological summary of the whole video (BE/schemas/explore.py, VideoSynthesizeRequest; the mock uses its own wording)
- max_segments: int, default 50, range 1..100
- system_prompt: str|null

RESP: 200 {original_video: str, segment_count: int, segments_used: int, answer: str, llm_synthesis: dict (LLMSynthesisResponse shape), generated_at: datetime ISO}.
With the default prompt, answer is markdown with **Overview** and **Timeline** sections (≤8 bullets).

QUIRKS: 404 when there are no accessible segments.
Uses the first max_segments segments by segment_number.
Text-only: Cosmos sees the reasoning captions, not the video.
A synthesis failure returns 200 with llm_synthesis.error set.
The Cosmos timeout is 120 s, which can exceed nginx's default 60 s proxy timeout on the /app ingress.
MOCK: same request validation, shape and 404; first max_segments segments; the default question is reworded; answer from the mock synthesis (template or MOCK_LLM_URL), not the Overview/Timeline format.

SRC: BE/api/v1/videos.py:332-343 (default prompt), 518-568; BE/schemas/explore.py (VideoSynthesizeRequest/Response); SK/retrieval/videos/SKILL.md:14-57

### vss-backend POST /api/v1/videos/upload | auth: Bearer JWT | confirmed-in-source

REQ: multipart/form-data:
- file (required). Extension in .mp4|.mov|.webm|.avi|.mkv, case-insensitive check.
- is_public: bool = false
- tags: CSV str = ""
- allowed_users: CSV str = ""
- scenario: str = "". An analysis_scenarios value; ignored when custom_prompt is set.
- custom_prompt: str = "". Stripped, TRUNCATED to 800, and URL-encoded into S3 metadata.
- camera_id, capture_type, location: str = ""
- stream_id: str = "". Must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$
- chunk_number: int|null. 1-based; required if and only if stream_id is set.

RESP: 200 {success: true, object_key: "<username>/<YYYYmmdd_HHMMSS>_<uuid8>.<ext>", stream_id: str|null, chunk_number: int|null, message: "Video uploaded successfully and will be processed shortly"}.
object_key is relative to S3_CHUNKS_BUCKET. The original filename is kept only in the S3 metadata key original-filename.

QUIRKS: Errors:
- 400 "Invalid file extension. Allowed: .mp4, .mov, .webm, .avi, .mkv"
- 413 "File too large. Max size: {N}MB", where N = max_upload_size_mb
- 400 stream errors:
  - "stream_id is required when chunk_number is set"
  - "chunk_number is required when stream_id is set"
  - "chunk_number starts at 1"
  - "stream_id must start with a letter or number and contain only letters, numbers, dots, underscores, or hyphens"
- 500 "Failed to upload video: ..."

Behaviour:
- The backend ingress body limit is 110m.
- Extra uploads wait on a semaphore (max_concurrent_uploads = 10) rather than being rejected.
- The uploader is always added to allowed_users.
- There is no job id. Track progress via /videos/explore?indexed=partial and /dashboard/stats.
- The kit skill says object_key is '<timestamp>_<filename>'; that is wrong per the source.
- The referenced architecture documents only re-ingest in the live path, so uploads may not be processed.

MOCK: same form fields, extension and stream_id/chunk_number checks with the same details, 413 above VSS_MOCK_MAX_UPLOAD_MB (default 25), object_key format with the extension's case kept (s3_service.py:209), custom_prompt stripped and truncated to 800, uploader added to allowed_users. Differences:
- The bytes are kept in VSS_MOCK_UPLOAD_DIR (default ~/.vastkit/mock_uploads), so the new rows can be streamed.
- Indexing is simulated: 4 placeholder segment rows, named from the object_key basename, appear one by one 25-40 s after the upload (listed under indexed=partial until all four are there). scenario is ignored.

SRC: BE/api/v1/videos.py:53-163; BE/services/s3_service.py:170-262; BE/utils/stream_upload.py:15-35; SP/repo_vssbp/src/source-code/shared/ingest_metadata.py:68-157; SK/ingest/upload-video/SKILL.md:85-124

### vss-backend GET /api/v1/videos/chunk | auth: Bearer JWT | confirmed-in-source

REQ: Query: stream_id: str (min length 1), chunk_index: int ≥0 (0-based), scope = all|mine|public.

RESP: 200 ChunkSearchResult (same shape as in /search chunk_results).

QUIRKS: 404 "Stream chunk not found".
Off-by-one trap: the upload API's chunk_number is 1-based; the stored and queried chunk_index is 0-based.
MOCK: same params, shape and 404; mock uploads store chunk_index = chunk_number - 1.

SRC: BE/api/v1/videos.py:445-472; BE/utils/stream_upload.py:1-6 (upload chunk_number 1-based maps to stored chunk_index 0-based)

### vss-backend DELETE /api/v1/videos | auth: Bearer JWT. Only the uploader may delete (allowed_users or the S3 key prefix). | confirmed-in-source

REQ: Query original_video = <parent s3 URI> (min length 1).

RESP: 200 VideoDeleteResponse {original_video: str, segments_deleted: int, prompts_deleted: int, objects_deleted: int, object_errors: list}.

QUIRKS: 404 "Video not found". 403 "You can only delete videos you uploaded".
IRREVERSIBLE: it deletes the S3 source, segments, sidecars, collection rows and prompt/event rows, then clears the read caches.
VSSClient.delete_video() requires confirm=True and retries only on connect errors. Do not expose it as an autonomous tool.
MOCK: same 404/403 and response keys; it removes the mock rows only (the counts are placeholders).

SRC: BE/api/v1/videos.py:346-442; BE/utils/video_delete.py; BE/services/vastdb_service.py:1853-1871 ($row_id deletes)

### vss-backend GET /api/v1/dashboard/stats | auth: Bearer JWT | confirmed-in-source

REQ: Query scope = all|mine|public (default all; pattern-validated).

RESP: 200 DashboardStatsResponse:
- table: str; table_available: bool; table_message: str|null
- generated_at: str; query_time_ms: float; scope: str
- overview: {total_rows, segment_rows, other_rows, unique_videos, fully_indexed_videos, indexed_clips, re_ingest_rows, re_ingest_clips, stream_sessions, duplicate_segment_slots, duplicate_segment_rows, public_segment_rows, private_segment_rows}
- quality: {reasoning_ok, reasoning_ok_pct, perception_ok, perception_ok_pct, with_object_classes, with_object_classes_pct}
- objects: [{label, segment_count, instance_count}]
- metadata: {field: [{label, count}]}
- uploads_by_day: [{date, segment_rows}]
- recent_videos: [{original_video, filename, stream_id?, segment_rows, indexed_clips, chunk_count, re_ingest_rows, stream_span_sec?, ingest_kind?, unique_segments, expected_segments, duplicate_rows, upload_timestamp?, camera_id, capture_type, location, is_public}]
- s3_inventory?: {chunks_bucket, chunks_mp4?, segments_bucket, segments_mp4?, segmenter_output_bucket, segmenter_output_mp4?, errors?}
- pipeline_alignment?: {segments_s3_mp4?, indexed_clips, segment_rows, pending_index?, re_ingest_excess, indexed_matches_segments_s3?, rows_match_segments_s3?, segments_bucket_matches_segmenter?, healthy?}

QUIRKS: 422 for a bad scope. 500 "Failed to load dashboard stats: ...".
S3 mp4 counts are cached 300 s and rows 45 s, per worker.
The kit uses this as the second half of its health check (SK/ask-cosmos/SKILL.md:20-24).
MOCK: the full DashboardStatsResponse shape, computed from the current mock rows. The mock never holds re-ingest duplicates, so the re-ingest and duplicate counts are 0; pending_index counts mock uploads not yet visible.

SRC: BE/api/v1/dashboard.py:11-30; BE/schemas/dashboard.py; BE/utils/dashboard_stats.py; BE/services/vastdb_service.py:1214-1300; SK/retrieval/dashboard/SKILL.md:14-33

### vss-backend POST /api/v1/dashboard/reingest | auth: Bearer JWT | documented-in-skill

REQ: JSON. Exactly one target: {stream_id: str} or {original_video: str (s3 URI)}. Then:
- chunk_count: int, 1..100. Use 1 for original_video; it re-ingests the N latest complete chunks.
- Optional overrides, sent only if non-empty: camera_id, capture_type, location, scenario (an ingest-config analysis_scenarios value), custom_prompt (≤800).
- Omitting scenario and custom_prompt keeps the original prompt. Sending scenario without custom_prompt removes the old custom prompt. Blank metadata means preserve.

RESP: Documented fields: {job_id: str, selected_chunks: int, copied_segments: int, ...}. The full shape is unknown.

QUIRKS: 404 if the parent is not indexed (reingest-chunk/SKILL.md:101). Other errors are unknown.
Re-ingest copies segments to NEW source URIs, so new rows coexist with old ones for the same segment_number slot. Readers dedupe by newest upload_timestamp.
It takes minutes.
Build the JSON with a serializer, never by shell interpolation.
MOCK (guessed route; logs a warning when hit): 422 for both or neither target and for custom_prompt > 800 (guesses; upload truncates instead), 404 "Video is not indexed", then {job_id, status: "running", selected_chunks, copied_segments}. When the job completes it rewrites the target's rows in place (new upload_timestamp, a "[re-ingested]" caption prefix, the metadata overrides) instead of adding rows.

SRC: SK/ingest/reingest-videos/SKILL.md:114-176; SK/ingest/reingest-chunk/SKILL.md:24,98-101,165-205. Route ABSENT from BE/api/v1/dashboard.py (only /stats exists in blueprint snapshot)

### vss-backend GET /api/v1/dashboard/reingest/{job_id} | auth: Bearer JWT | documented-in-skill

REQ: Path job_id from the POST.

RESP: Documented fields: {status ("completed" is the terminal success value; others unknown), completed_chunks: int, total_chunks: int, indexed_segments: int, total_segments: int, ...}.

QUIRKS: Poll every 4 s.
A 404 after a backend restart only means the in-memory record was lost; it does not mean processing stopped.
INFERRED: with --workers 4, an in-memory job record lives in one worker, so polls may randomly 404. Treat a 404 as 'unknown': keep polling and fall back to /videos/explore?indexed=partial|complete or /dashboard/stats.
A chunk is done when its timeline covers segments 1..total_segments.
MOCK: 'running' (a guess) with linear progress, then 'completed' after MOCK_REINGEST_SECONDS (default 30). A share MOCK_REINGEST_404 (default 0.25) of polls gets a deterministic 404 "Job not found" to exercise the fallback; set it to 0 to turn that off.

SRC: SK/ingest/reingest-videos/SKILL.md:178-196; SK/ingest/reingest-chunk/SKILL.md:209-216; BE Dockerfile CMD uvicorn --workers 4 (SP/repo_vssbp/src/source-code/retrieval/video-backend/Dockerfile)

### vss-backend GET /api/v1/metadata/schema | auth: Bearer JWT | confirmed-in-source

REQ: No params.

RESP: 200 {schema: [{name: str, type: str (arrow type), ui_type: "select"|"text", label: str, options?: str[]}], table: str}.

QUIRKS: Only FILTERABLE_METADATA_COLUMNS appear: camera_id, capture_type, location, object_classes.
A field is 'select' (with options) only when it has 1..100 distinct values.
list and fixed_size_list columns are skipped.
500 "Failed to discover metadata schema: ...".
MOCK: same shape and select rule over the visible mock rows; type is always "string".

SRC: BE/api/v1/metadata.py:25-54 (excluded cols), 67-128; SP/repo_vssbp/src/source-code/shared/ingest_metadata.py:14-21

### vss-backend GET /api/v1/metadata/values | auth: Bearer JWT | confirmed-in-source

REQ: Query: field: str (required; must be camera_id|capture_type|location|object_classes), prefix: str = "", limit: int = 50 (no bounds enforced).

RESP: 200 {field: str, values: str[], count: int}.

QUIRKS: 400 "Field '<field>' is not available for value lookup". 500 "Failed to fetch field values: ...".
The prefix match is a case-insensitive startswith.
Distinct values are cached 120 s per worker.
Use it to validate metadata_filters before searching (SK/ask-cosmos).
MOCK: same params, 400 and prefix rule.

SRC: BE/api/v1/metadata.py:131-175

### vss-backend GET /api/v1/metadata/ingest-config | auth: none (public) | confirmed-in-source

REQ: No params, no auth.

RESP: Exact payload of ingest_config_for_api():
- custom_prompt_max_length: 800
- filterable_fields: [{key, label}]
  - camera_id: "Camera ID"
  - capture_type: "Capture Type"
  - location: "Location"
  - object_classes: "Object class"
- capture_types: [{value, label}], 9 entries in the source snapshot; each label is the Title-case value
- capture_type_default_label: "-- Select Type --"
- analysis_scenarios: [{value, label}], 11 entries in the source snapshot (value is a short id, label a human-readable name)
- analysis_scenario_default_label: "-- Use Default (from settings) --"
- labels: {use_custom_prompt, metadata_section_title, metadata_section_description, camera_id, capture_type, location, object_classes}
- placeholders: {camera_id, location, tags, allowed_users, custom_prompt}

QUIRKS: ARCHITECTURE_REFERENCE.md:96 lists 9 scenario presets; the source has 11. Read the lists from this endpoint instead of hardcoding them.
MOCK: same keys, field labels and default labels; the use_custom_prompt / metadata_section_* labels and the placeholders are shorter neutral wording of its own, and capture_types and analysis_scenarios each hold one neutral entry ("general", with the source's labels). No auth, as in the source.

SRC: BE/api/v1/metadata.py:57-64; SP/repo_vssbp/src/source-code/shared/ingest_metadata.py:12-82,160-179

### vss-backend GET /api/v1/suggestions | auth: Bearer JWT | confirmed-in-source

REQ: Query chips_only: bool = false.

RESP: 200:
- batch_id: str|null
- generated_at: ISO-8601 'Z' str|null
- search_prompts: str[≤10]
- key_events: [{query_text, label ("" when equal to query_text), original_video, filename, segment_start_sec: float, segment_end_sec: float, generated_at: ISO Z, batch_id, upload_timestamp: ISO Z|null}]
- key_events_count: int
- table: "<bucket>/<schema>/<table>"
- prompts_table_available: bool
- table_message: str|null

QUIRKS: A missing or unreadable prompts table gives 200 with prompts_table_available = false, not an error.
500 "Failed to load suggestions: ...".
chips_only = true gives key_events = [] and skips the collection scan.
Key events are filtered to videos the user can access.
Cache: 60 s per user per worker.
Only rows with is_active are used.
MOCK: same shape; search_prompts is always empty and key_events are the first 5 visible rows (query_text = the first 60 characters of the caption, batch_id "mock-batch"); chips_only gives [].

SRC: BE/api/v1/suggestions.py:13-31; BE/services/suggestions_service.py:58-61,236-324; SK/retrieval/suggest-prompts/SKILL.md

### gpu:cosmos3-reason POST {COSMOS3_REASON_URL}/v1/chat/completions | auth: Unresolved:
- The gpu skills send Authorization: Bearer ${GPU_BEARER_TOKEN} (SK/gpu/README.md:24).
- config.example:44-46 and ARCHITECTURE_REFERENCE.md:20-21 say no token is needed.
Make it optional: send the header only when a token is set, accepting a value with or without 'Bearer ' (as llm_service.py:310-314 does).
Always use your own GPU_BEARER_TOKEN from .env. | confirmed-in-source

REQ: OpenAI chat JSON:
- model: str, discovered via GET /v1/models data[0].id. config.example gives nvidia/cosmos3-reason, which reportedly 404s.
- messages: [{role: system|user|assistant, content}]. content is a str, or a list [{"type":"text","text":str}, {"type":"video_url","video_url":{"url":"data:video/mp4;base64,<b64>"}}].
- max_tokens: int
- temperature: float
- stream: false

How the blueprint uses it:
- The ingest reasoner sends one user message with [text, video_url] (no system message).
- Backend synthesis is text-only (system + user), with temperature 0.2 and max_tokens 2000.
- When finish_reason is length or max_tokens, it continues up to 4 more rounds. Each round appends the assistant text plus a short user turn asking the model to carry on from where it stopped without repeating itself (vastkit's CosmosReason uses its own wording of the same idea).

RESP: Fields consumed: choices[0].message.content (str), choices[0].finish_reason, usage.total_tokens (cached prompt tokens are read from the usage details by the reasoner).

QUIRKS: 401/403 means a wrong or missing bearer (the smoke-test table).
The ingest clients treat 401, 403, 408, 429, 500, 502, 503 and 504 as transient and retry.
NIM docs: a requested fps above the clip's fps returns 400, and media_io_kwargs fps and num_frames are mutually exclusive. Whether this deployment supports either is unverified. vastkit sends media_io_kwargs as {"video": {fps | num_frames}} and leaves it out only when the server says the field itself is unknown (the result's `dropped` then names it). An "fps exceeds" 400 is retried once at the clip fps the error names; other media_io_kwargs 400s raise.
The video is base64 inside the JSON, so keep clips to segment length (~5 s).
The skills hardcode a GPU host on port 8001 from an earlier dry run; that host is stale for London. Read COSMOS3_REASON_URL instead.
GPU MOCK: GET /v1/models lists MOCK_COSMOS_MODEL (default nvidia/cosmos3-nano-reasoner); chat accepts that id or "local", else 404. It returns 400 for fps + num_frames and for fps above MOCK_SOURCE_FPS (default 30). Output is canned: a json_schema with an `events` property gets two neutral events, other schemas get type-default values, a prompt containing <think> gets the think/answer format, anything else a one-line echo. finish_reason is always 'stop'; usage is fixed.

SRC: SP/repo_vssbp/src/source-code/ingest/video-reasoner/common/clients.py:101-197; BE/services/llm_service.py:305-373; BE/config.py:65-74,135-152; SK/gpu/README.md:51-82; SK/gpu/model-smoke-test/SKILL.md:85-94

### gpu:cosmos3-reason + cosmos-embed1 GET /v1/models, /v1/health/ready, /v1/health/live | auth: optional Bearer (see chat entry) | documented-in-skill

REQ: none (optional Bearer)

RESP: /v1/models returns {data: [{id: str, ...}]}. /v1/health/ready and /v1/health/live return 200 when up.

QUIRKS: /v1/health returns 404 on these NIMs, which is normal.
Model ids: discover at startup and cache. Fall back to env COSMOS3_REASON_MODEL / COSMOS_EMBED1_MODEL only if discovery fails.
The first NIM start can take minutes (DEPLOY.md:100-105).
vastkit's ready() trusts /v1/health/ready and tries other routes only when it returns 404/405.
GPU MOCK: /v1/models, /v1/health/ready and /v1/health/live, no auth.

SRC: SK/gpu/README.md:38-47,60-63,116-119; SK/gpu/model-health/SKILL.md:19-35; KIT config.example:52-58

### gpu:yolo11 GET {YOLO_URL}/healthz | auth: optional Bearer (the detector sends detector_authorization if set) | confirmed-in-source

REQ: none

RESP: 200 {ok: true, model: "yolo11s.pt", device: "0", cuda_available: bool, model_loaded: bool, inputs: ["url","video_base64"]}.

QUIRKS: There is no /v1/models or /v1/health/* on this server. GET /v1/infer returns 405.
Healthy means ok and model_loaded are both true.
Port varies by source, so never hardcode it; use YOLO_URL:
- kit: 8002 (SK/gpu/README.md:21)
- blueprint deploy.sh: 8003 (DEPLOY.md:50-55)
- detector code default: 8022 (SP/repo_vssbp/src/source-code/ingest/video-detector/common/models.py:11-12)
GPU MOCK: same keys, device "mock", model_loaded true. The local mirror (vastkit/local/yolo_server.py) reports the weights file it loaded.

SRC: SP/repo_vssbp/src/scripts/vss-blueprint-models/yolo-infer/main.py:214-230; SK/gpu/README.md:45,103; SK/gpu/model-health/SKILL.md:30-35

### gpu:yolo11 POST {YOLO_URL}/v1/infer  (alias /v1/infer-base64 requires video_base64 and no url, else 400) | auth: optional Bearer | confirmed-in-source

REQ: JSON InferRequest:
- url: http(s) URL|null
- video_base64: str|null. An optional 'data:video/*;base64,' prefix is stripped.
- filename: str|null. Suffix hint; a non-video suffix becomes .mp4.
- include_frames: bool = false
Exactly one of url or video_base64 is required.
Limits: base64 ≤ 280 MiB chars; decoded or downloaded ≤ 200 MiB; decoded ≥ 12 bytes; download timeout 120 s; conf 0.4.

RESP: 200:
- ok: true
- perception_ok: bool (frames > 0)
- perception_json: {source: "yolo11_coco", frames: [...]|null, object_classes: str[] sorted, object_counts: {label: int}, max_detection_conf: float, frame_count: int, detection_count: int}
- object_classes: str[]
- object_counts: {label: int}
- max_detection_conf: float
- frames: [{frame_index: int, shape: [h, w], detections: [{label: str, confidence: float (4 dp), bbox: [x1, y1, x2, y2] int pixels}]}] | null

QUIRKS: Errors:
- 422 "Provide exactly one of: url, video_base64"
- 400 "Invalid base64 in video_base64: ...", "Decoded video is too small to be a valid MP4", or "Unsupported URL scheme: ..."
- 413 when over a size limit
- 502 "Presigned URL fetch failed: HTTP ..." or "Inference failed: ..."
- /v1/infer-base64 only: 400 "video_base64 is required for /v1/infer-base64" or "Use /v1/infer with url only, not both"

Differences from the stored data:
- EVERY frame is inferred (no stride), so include_frames on long clips is huge.
- The frames carry NO time_sec; the ingest sidecar adds it.
- object_counts here are raw totals across frames (one person in 150 frames counts 150). VastDB/sidecar object_counts are unique-object estimates.

A `url` must be reachable from the GPU host; presigned URLs on the private S3 VIP probably are not, so prefer base64.
The backend never calls YOLO (SK/deployment/build-yamls/SKILL.md:48).
GPU MOCK: the 422 and base64 400s match; any valid input gets the same canned 10 frames (one person, one truck per frame, raw totals as above). No /v1/infer-base64 alias.
Local mirror: same /v1/infer contract; its /v1/infer-base64 alias returns 422 "Provide video_base64 (and no url)" where the hosted route returns the 400s above, and its 413 details are its own. perception_json.source is PERCEPTION_SOURCE (default yolo11_coco) whatever the weights; optional `stride` and `conf` fields that the hosted server ignores.

SRC: SP/repo_vssbp/src/scripts/vss-blueprint-models/yolo-infer/main.py:31-37,52-134,137-175,187-211,233-272; SP/repo_vssbp/src/source-code/ingest/video-detector/common/clients.py:65-125 (client: {video_base64, filename, include_frames:true}); SK/gpu/README.md:86-103

### gpu:cosmos-embed1 POST {COSMOS_EMBED1_URL}/v1/embeddings | auth: Optional Bearer. The blueprint sends embedding_authorization if set, else nvidia_api_key when not a local NIM. | confirmed-in-source

REQ: JSON:
- input: one of:
  - str: one text, request_type "query"
  - str[]: texts, request_type "bulk_text"
  - "data:video/mp4;base64,<b64>": a video, request_type "query"
- model: "nvidia/cosmos-embed1"
- request_type: "query"|"bulk_text"|"bulk_video"
- encoding_format: "float"
There is NO `dimensions` param.

RESP: 200 {data: [{embedding: float[256], ...}], ...}. The order follows the input.

QUIRKS: Dims are always 256 (the 224p variant) and must match vss-collection.vectors and vectors_visual.
Use the SAME embed server as ingest, or vector spaces differ (backend-secret.yaml.example:29-31).
Video timeout is 180 s; text 60-120 s.
An empty data array is an error (ingest raises).
Ingest treats 401 and 403 as transient.
Port: kit 8003; blueprint default 8002.
GPU MOCK: deterministic hash-seeded 256-d unit vectors (keyed by the lowercased text, or a hash of a video data URI), in input order; 400 for an empty list. They are not semantic, so similarities between them mean nothing.

SRC: SP/repo_vssbp/src/source-code/ingest/video-embedder/common/cosmos_embed_client.py:1-8,30-125; BE/services/cosmos_embed_client.py:15-69; BE/services/embedding_service.py:21-37; SK/gpu/README.md:107-126

### gpu:canary-1b POST {CANARY_1B_URL}/v1/audio/transcriptions | auth: optional Bearer (unresolved; see chat entry) | documented-in-skill

REQ: multipart/form-data: file (required; an audio file, e.g. 16 kHz mono WAV extracted with ffmpeg) and model (the skill example uses "nvidia/canary-1b"). Optional language and response_format are inferred from the OpenAI-compatible Riva NIM and unverified.

RESP: INFERRED: OpenAI-style {"text": str}. No source shows the shape.
Health: /v1/health/ready returns {"status":"ready"}; /v1/health/live returns {"status":"live"}.

QUIRKS: /v1/models and /healthz return 404 on this NIM; that is normal.
CANARY_1B_MODEL is intentionally unset (config.example:59-62), and research suggests discovering it from /v1/models, which 404s here. So the model id is uncertain: try 'nvidia/canary-1b', then the field omitted.
Canary is not part of the default pipeline, so nothing indexes audio. Any audio feature must store its own results (e.g. a custom VastDB table).
GPU MOCK: returns {"text": "mock transcript of <filename> (<n> bytes)"}.

SRC: SK/gpu/README.md:130-152; SK/gpu/model-health/SKILL.md:19-35; KIT config.example:50,59-62

## vastdb

LAYOUT
- database (bucket) = VASTDB_BUCKET, e.g. team-x-vss-db. The backend template default is vss-db.
- schema VDB_SCHEMA = vss-schema.
- tables: VDB_COLLECTION = vss-collection (the search index) and VDB_PROMPTS_COLLECTION = vss-prompts-events.
- Sources: KIT config.example:21-30; SP/repo_vssbp/src/deployments/vss-k8s-application/backend-secret.yaml.example:10-15.

vss-collection
Schema: SP/repo_vssbp/src/source-code/ingest/vastdb-writer/common/vastdb_client.py:52-89. One row per ~5 s segment clip.

| column | type | meaning |
|---|---|---|
| pk | utf8 | md5(source) (vastdb-writer/common/segment_index.py:7-8) |
| source | utf8 | s3 URI of the segment mp4 |
| filename | utf8 | |
| segment_number | uint32 | 1-based |
| segment_start_sec | float64 | position in the parent |
| segment_end_sec | float64 | position in the parent |
| reasoning_content | utf8 | Cosmos caption |
| perception_json | string | JSON summary {source, object_classes[], object_counts{}, object_counts_mode, max_detection_conf, frame_count, detection_count} |
| object_classes | utf8 | comma-joined, lowercase |
| object_counts | utf8 | JSON dict of unique-object estimates |
| max_detection_conf | float32 | |
| perception_ok | bool | |
| perception_source | utf8 | 'yolo11_coco' |
| detection_sidecar_uri | utf8 | s3://<segments bucket>/detections/<stem>.json.gz |
| detection_frame_count | uint32 | |
| detection_count | uint32 | |
| vectors | fixed_size_list<float32 NOT NULL>[256] | text embedding of reasoning_content |
| vectors_visual | fixed_size_list<float32 NOT NULL>[256] | Cosmos-Embed1 embedding of the segment mp4; ZERO-FILLED when visual embedding failed (vastdb_client.py:203-204) |
| cosmos_model | utf8 | |
| embedding_model | utf8 | |
| visual_embedding_model | utf8 | |
| tokens_used | int32 | |
| cached_prompt_tokens | int32 | |
| processing_time | float64 | |
| timestamp | utf8 | ISO 'Z' write time |
| allowed_users | list<utf8> | |
| is_public | bool | |
| upload_timestamp | timestamp(ns) | naive; parsed from the pipeline's ISO string, effectively UTC |
| duration | float64 | segment length |
| total_segments | uint32 | |
| original_video | utf8 | parent chunk s3 URI |
| tags | list<utf8> | |
| camera_id | utf8 | |
| capture_type | utf8 | |
| location | utf8 | |
| extra_metadata | string | JSON {status, embedding_dimensions, visual_embedding_dimensions, visual_embedding_ok, stream_id?, chunk_index? (0-based), chunk_start_sec?, stream_position_sec?, ingest_kind?} |

Writer rules:
- Rows with empty reasoning_content are never stored (vastdb_client.py:142-143).
- Insert is skipped if a row with the same source exists (vastdb_client.py:282-292).
- Re-ingest therefore writes new source URIs, leaving several rows per (original_video, segment_number) slot. Readers dedupe by newest upload_timestamp, then longest reasoning (BE/utils/segment_timeline.py:7-28).
- The dashboard counts these as duplicate_segment_rows.
- Filter WHERE extra_metadata visual_embedding_ok, or skip all-zero vectors, before trusting visual kNN. Cosine against a zero vector is undefined.

vss-prompts-events
Schema: SP/repo_vssbp/src/source-code/enrichment/prompt-suggester/common/vastdb_client.py:17-29.

| column | type | meaning |
|---|---|---|
| pk | utf8 | md5('search:'+q) or md5('event:'+(source or ov)+':'+start+':'+q) |
| kind | utf8 | 'search_prompt' or 'key_event' |
| query_text | utf8 | |
| label | utf8 | |
| original_video | utf8 | |
| filename | utf8 | |
| segment_start_sec | float64 | |
| segment_end_sec | float64 | |
| generated_at | timestamp(ns) | |
| batch_id | utf8 | |
| is_active | bool | |

- config.example:27 says this table 'holds the ingestion prompts'. That is WRONG per the source: it holds LLM-generated search chips and key events.
- The original scenario and custom_prompt are not persisted in any table (SK/ingest/reingest-videos/SKILL.md:107-112).
- They live only as S3 object metadata on chunks (SP/repo_vssbp/src/source-code/shared/ingest_metadata.py:68-82). Keys:
  - camera-id, capture-type, location, scenario
  - custom-prompt (URL-encoded)
  - tags, allowed-users, is-public
  - original-filename, upload-timestamp, capture-timestamp, owner
  - plus stream_id, chunk_index (0-based) and ingest_kind=stream_chunk

WRITE RULES
- Never write, drop or alter vss-collection or vss-prompts-events. They are protected in the kit insert.py and SKILL. vastkit.vdb refuses writes to any vss-* table, the vss-schema schema and the configured VDB_SCHEMA / VDB_COLLECTION / VDB_PROMPTS_COLLECTION names (case-insensitively) and returns them read-only from table(); this guards against accidents and is not a security boundary.
- Writes go through the vastdb SDK on the DATA VIP:
  - vastdb.connect(endpoint=S3_ENDPOINT, prefixed with http:// if it has no scheme, access=ACCESS_KEY, secret=SECRET_KEY, ssl_verify=False)
  - inside `with s.transaction() as tx:`
  - bucket = tx.bucket(VASTDB_BUCKET)
  - schema = bucket.schema(name, fail_if_missing=False) or bucket.create_schema(name, fail_if_exists=False)
  - schema.table(name, fail_if_missing=False) or schema.create_table(name, columns=pa.schema)
  - table.insert(pa.Table.from_pylist(rows, schema=...))
- Use your own schema, e.g. 'hackathon'.
- A concurrent create can raise 409/Conflict/'already exists'; treat that as success and re-fetch (vastdb_client.py:113-128).
- Kit insert.py Arrow inference: bool→bool_, int→int64, float→float32, otherwise utf8. Declare explicit schemas for timestamps, lists and vectors.
- Vector column type for kNN compatibility: pa.list_(pa.field('item', pa.float32(), nullable=False), 256).
- Other teams' buckets return 403.

SDK QUIRKS
- Never select() with no columns on vss-collection.
- Exclude vectors and vectors_visual from SDK selects. Kit query.py --include-vectors 'often fails on SDK select'.
- The blueprint monkey-patches vastdb._internal.build_query_data_request to strip vector fields from predicate queries: fixed_size_list in the backend (BE/services/vastdb_service.py:38-72), fixed_size_list and list<float> in the ingest writer (SP/repo_vssbp/src/source-code/ingest/vastdb-writer/common/vastdb_patch.py). Apply such a patch before any predicate select on a table with vectors; vastkit's vdb.patch_vector_predicates() uses the writer's type check.
- Pass internal_row_id=False on selects. Deletes use the '$row_id' column (vastdb_service.py:1853-1871).
- Never name a script select.py; it shadows the stdlib select module.
- Pins:
  - backend: vastdb 1.1.0 and adbc-driver-manager 1.0.0 (requirements.txt)
  - writer: 1.3.2
  - PyPI vastdb: 2.1.0 (Python ≥3.10)
- Endpoints:
  - The backend vdb_endpoint is the Query Engine VIP. The backend also uses it for SDK scans (suggestions_service.py:73-83).
  - Kit query.py reads VDB_ENDPOINT (env, then file), else S3_ENDPOINT.
  - The ingest secret's vdbendpoint is the data VIP.

SQL (Query Engine via ADBC), as the backend actually emits it
- Connect: adbc_driver_manager.dbapi.connect(driver=<path to libadbc_driver_vastdb.so>, db_kwargs={'vast.db.endpoint': QE VIP, 'vast.db.access_key': ..., 'vast.db.secret_key': ...}).
  - The backend loads /opt/adbc-driver/libadbc_driver_vastdb.so with a /tmp fallback, and downloads the driver when neither exists (vastdb_service.py:109-154, VastDBService._setup_adbc). vastkit's vdb.find_adbc_driver() checks $VAST_ADBC_DRIVER first, then those two paths, then the adbc-driver-vastdb pip package; it never downloads.
  - pip adbc-driver-vastdb ships manylinux x86_64 and macOS wheels only, NO Windows. Run SQL on the VM, or in Linux/WSL.
- Table reference: "<bucket>/<schema>"."<table>", quoted, slash inside (vastdb_service.py:270).
- kNN: SELECT <cols>, array_cosine_distance(vectors::FLOAT[256], ARRAY[f1,...,f256]::FLOAT[256]) AS distance FROM ... WHERE ... ORDER BY distance LIMIT n. Use vectors_visual for the visual branch; similarity = 1 - distance.
- Working constructs:
  - array overlap: tags && ARRAY['a','b'] and allowed_users && ARRAY['user']
  - booleans: is_public = TRUE
  - time: upload_timestamp >= TIMESTAMP 'YYYY-MM-DD HH:MM:SS' (naive)
  - text: LOWER(object_classes) LIKE '%car%'
  - string equality with '' escaping
- 'IN is rejected by this QE' (vastdb_service.py:662). Use OR chains.
- No query parameters are used: everything is string-interpolated, so escape quotes yourself.
- GROUP BY, COUNT, DISTINCT, JOIN, window functions and ILIKE are untested. The backend does all aggregation in Python over SDK scans. vastkit/vdb.py lint_sql (run by qe_query) refuses IN (...), ILIKE, GROUP BY, DISTINCT, JOIN, HAVING and OVER (...), as well as anything but a single SELECT/WITH statement, ';' (even a trailing one) or '$' outside string literals (so no dollar-quoted strings), a backslash inside quotes, a letter right before a quote (E'' / U&'' escape strings) and nested /* comments. vdb.cosine_distance_sql builds only the array_cosine_distance(...) expression above.

LOCAL STAND-IN
- DuckDB with FLOAT[256] columns and array_cosine_distance runs the same SELECT shape. This is likely, since the QE is DuckDB-derived per the ADBC README.
- Keep the IN/OR linter so local SQL never uses what the QE rejects.

## env_vars

PATH KEYS used in citations
- SP = <ref> (read-only clones of the public VAST kit and VSS blueprint repos, used as citations; not shipped)
- KIT = SP/repo_vbc/src
- SK = KIT/.cursor/skills
- BE = SP/repo_vssbp/src/source-code/retrieval/video-backend/src
- MOCK = vastkit/mocks/vss_mock.py; GPU MOCK = vastkit/mocks/gpu_mock.py (both shipped in this repo)

TEAM CONFIG
Source: /config/<team>.config on the VM, exported into the VM environment; documented in KIT config.example:9-62.

| variable | meaning |
|---|---|
| USERNAME | team id, e.g. team-x. It is the VSS login, the K8s namespace and the S3 key prefix. |
| PASSWORD | VSS login (secret) |
| S3_ENDPOINT | data VIP for S3 and VastDB SDK writes |
| ACCESS_KEY, SECRET_KEY | secrets |
| S3_CHUNKS_BUCKET | <team>-vss-chunks: upload target and parent chunks |
| S3_SEGMENTS_BUCKET | <team>-vss-chunks-segments: segment mp4s and detections/ sidecars |
| VDB_ENDPOINT | VastDB endpoint. If unset, use S3_ENDPOINT. |
| VASTDB_BUCKET | <team>-vss-db |
| VDB_SCHEMA | vss-schema |
| VDB_COLLECTION | vss-collection |
| VDB_PROMPTS_COLLECTION | vss-prompts-events |
| PIPELINE | UI name; no skill reads it |
| INGRESS_URL | VSS backend base URL. Internal; skills call it $BACKEND. |
| WANDB_API_KEY | secret; often set in the env but not in the file |
| WANDB_TEAM, WANDB_PROJECT | W&B team and project |
| COSMOS3_REASON_URL, YOLO_URL, COSMOS_EMBED1_URL, CANARY_1B_URL | each its own host:port; never derive one from another |
| COSMOS3_REASON_MODEL | nvidia/cosmos3-reason |
| COSMOS_EMBED1_MODEL | nvidia/cosmos-embed1 |
| CANARY_1B_MODEL | intentionally unset |

Other files in /config may be team-prefixed: <team>-k8s.yaml, <team>-backend-secret.yaml, <team>-vss2-secret.yaml.

SKILL-ONLY VARIABLES
- GPU_BEARER_TOKEN: read from the team config by the gpu skills, but absent from config.example. Treat it as OPTIONAL.
- GPU_HOST: the skills' hardcoded (stale) GPU host. Ignore it.
- AUTH: the bash header array.
- TEAM_CONFIG: the single /config/*.config path.
- BACKEND = INGRESS_URL.
- TOKEN: the JWT.
- K8s deploy: NS = USERNAME; KUBECONFIG = /config/${NS}-k8s.yaml; APP_NAME (DNS-1123); APP_DIR; TEAM_N = USERNAME minus 'team-'; APP_HOST = video-lab-team-${TEAM_N}.cosmos.vastdata.com; APP_PORT = 8080.
- App pod: PORT, VSS_URL (= INGRESS_URL), VSS_USERNAME, VSS_PASSWORD.

KIT PYTHON (vastdb-read query.py, list_catalog.py, vastdb-write insert.py)
- VAST_ENV_FILE overrides the config path. This is the only way off-VM: _team_config() otherwise globs /config/*.config and exits unless exactly one exists (query.py:10-30).
- VDB_ENDPOINT from the process env wins over the file.
- Credential keys are read from the FILE, not os.environ (query.py:83-93):
  - VAST_ACCESS_KEY, else ACCESS_KEY
  - VAST_SECRET_KEY, else SECRET_KEY
  - VASTDB_BUCKET, VDB_SCHEMA, VDB_COLLECTION
- Endpoints without a scheme get http:// prepended.

BACKEND (not for clients; for the mock only)
- SECRET_PATH defaults to /etc/secrets/config.yaml. Without it, settings come from pydantic env or .env with lowercase keys (BE/config.py:11-117, 155-168): vdb_endpoint, vdb_bucket, vdb_schema, vdb_collection, vdb_prompts_collection, vdb_access_key, vdb_secret_key, s3_*, embedding_*, visual_embedding_*, hybrid_text_weight (0.6), cosmos_host / cosmos_port / cosmoshttpscheme / cosmos_authorization / cosmos_model, synthesis_*, max_upload_size_mb, max_concurrent_uploads, allowed_video_extensions, vast_host, tenant_name, jwt_secret, cors_origins, display_timezone.
- The ingest vss2-secret keys have no underscores: vdbendpoint, vdbbucket, vdbschema, vdbaccesskey, vdbsecretkey, vdbcollection, s3endpoint, s3accesskey, s3secretkey, embeddinghost, embeddingport, embeddingmodel, embeddingdimensions. Exceptions: yolo_infer_host, yolo_infer_port, detector_authorization, cosmos_authorization.

VASTKIT CLIENT VARIABLES (vastkit/config.py; not kit-defined)
- Lookup per key: process env, then the config file (VAST_ENV_FILE, else the single /config/*.config, else ./.env). Aliases are tried in order, env then file for each.
- VSS URL: VSS_BASE_URL, then VSS_URL, then INGRESS_URL. Login: VSS_USERNAME / VSS_PASSWORD, then USERNAME / PASSWORD (the env USERNAME is ignored on Windows).
- VAST_SOCKS = socks5h://127.0.0.1:1080, applied per client to every host except loopback (127.0.0.1, localhost, ::1).
- GPU_BEARER_TOKEN, optional. *_MODEL, optional with /v1/models discovery.
- VSS_MOCK=1: the VSS URL becomes MOCK_VSS_URL (default http://127.0.0.1:8765) and every GPU URL MOCK_GPU_URL (default :8766); the login comes from the shell's VSS_MOCK_TEAM / VSS_MOCK_PASSWORD (defaults team-mock / mock-password); VAST_SOCKS is ignored.
- vastkit.llm: WANDB_API_KEY (and the project header) go only to https://api.inference.wandb.ai/v1. LLM_BASE_URL / LLM_API_KEY select another OpenAI-compatible server; loopback servers get a placeholder key. AGENT_MODEL, FAST_MODEL, VLM_MODEL, VLM_CHEAP and JUDGE_MODEL override the default model ids.

WINDOWS CAVEATS
1. Windows always defines %USERNAME% (the local account, e.g. '<windows-user>'), so os.environ['USERNAME'] silently returns the Windows account, not team-x. Read USERNAME from the config file, or use VSS_USERNAME. The same applies in Git Bash and PowerShell.
2. Windows env names are case-insensitive, so `username` and `USERNAME` collide.
3. There is no /config on the laptop. Copy only the non-secret URLs, or set VAST_ENV_FILE to a local file kept outside the repo and .gitignored.
4. zsh reserves USERNAME, and the kit says not to `source` the config there. `set -a; source` is bash-only; in PowerShell, parse KEY=VALUE lines.
5. Never print env. Use `env | cut -d= -f1` only (KIT .cursor/rules/build-day.mdc). PASSWORD, ACCESS_KEY, SECRET_KEY, WANDB_API_KEY and GPU_BEARER_TOKEN are secrets.
6. Do not set HTTPS_PROXY globally to the SOCKS tunnel; it would also route W&B and PyPI traffic.
7. The adbc-driver-vastdb wheel does not exist for Windows.

## deploy

CONTRACT for /app, with no registry
Source: SK/deployment/deploy-app-no-registry/SKILL.md (line refs below).

1) ConfigMap ${APP_NAME}-code (L108-120):
   kubectl -n $NS create configmap ${APP_NAME}-code --from-file=$APP_DIR --dry-run=client -o yaml | kubectl apply -f -
   - Size: ~1 MiB total (L47, L106, L283). No videos, models or node_modules.
   - Inferred from kubectl behaviour: --from-file=<dir> takes only top-level regular files and ignores subdirectories. Keys must match [-._a-zA-Z0-9]+.
   - Keep the app FLAT: main.py, requirements.txt (optional), index.html, app.js, style.css.
   - Mounted read-only at /code.

2) Secret ${APP_NAME}-vss-creds (L122-130), with keys:
   - VSS_URL = $INGRESS_URL
   - VSS_USERNAME = $USERNAME
   - VSS_PASSWORD = $PASSWORD
   Add more --from-literal keys plus env entries for anything else (GPU URLs, WANDB_API_KEY). Never put secrets in the ConfigMap.

3) Deployment ${APP_NAME} (L137-202):
   - replicas 1
   - image python:3.12-slim, imagePullPolicy IfNotPresent
   - containerPort ${APP_PORT} = 8080
   - env PORT = "8080", plus VSS_URL / VSS_USERNAME / VSS_PASSWORD via secretKeyRef
   - volume 'code' from configMap ${APP_NAME}-code, mountPath /code, workingDir /code
   - command ["bash","-c"] with args: set -euo pipefail; if [ -f requirements.txt ]; then pip install --no-cache-dir -q -r requirements.txt; fi; exec python main.py
   - readinessProbe httpGet /health on ${APP_PORT}, initialDelaySeconds 10, periodSeconds 10
   - No liveness probe, no resources.
   - pip runs on EVERY pod start and needs PyPI egress (unverified). Prefer stdlib only (http.server, urllib, json) or very few deps. Readiness stays false until the install finishes.

4) Service ${APP_NAME} (L203-217): ClusterIP, port 80 to targetPort ${APP_PORT}.

5) Ingress ${APP_NAME} (L218-240):
   - ingressClassName nginx
   - annotation nginx.ingress.kubernetes.io/rewrite-target: /$2 (written \$2 inside the heredoc)
   - host ${APP_HOST} = video-lab-team-<N>.cosmos.vastdata.com (L31, L68-71)
   - path /app(/|$)(.*), pathType ImplementationSpecific
   - backend service ${APP_NAME}, port 80
   - There is no proxy-body-size or proxy-read-timeout annotation (the VSS backend uses 110m and 300 s; backend-deployment.yaml:81-84). That means nginx defaults:
     - 1m request body: browser uploads over 1 MB through /app return 413.
     - 60 s upstream timeout: Cosmos synthesis (120 s timeout) can return 504.
     Add those annotations if allowed, or use an async job plus polling.

APP RULES
- Listen on 0.0.0.0:$PORT.
- Serve all routes at / because the Ingress strips /app (L81, L94, L102).
- GET /health must return 200 quickly with no VSS call.
- All browser-side URLs must be RELATIVE with no leading slash ('api/search', 'app.js').
  - '/x' resolves to the VSS UI at / (L32).
  - '/api/...' hits the VSS backend ingress.
- /app without a trailing slash makes relative URLs resolve against '/'. The server cannot see the prefix (no X-Forwarded-Prefix is configured), so fix it client-side: if (location.pathname.endsWith('/app')) location.replace(location.pathname + '/'). Never send absolute-path redirects.

CALLING VSS FROM THE POD
- Base: VSS_URL + /api/v1/... (INGRESS_URL is reachable in-cluster).
- POST auth/login with VSS_USERNAME/VSS_PASSWORD. Cache the JWT for under 1 day and log in again on 401.
- Proxy video through the app (GET video?source=...): fetch /api/v1/videos/stream?source=&token= server-side and pass Range, 206 and Content-Range through. Browsers cannot reach INGRESS_URL or presigned S3 URLs, and the token stays server-side.

OPERATIONS
- Update: re-apply the ConfigMap, then kubectl -n $NS rollout restart deploy/$APP_NAME. Running pods do not pick up ConfigMap changes (L119-120, L268-273). The same applies to Secret changes.
- Verify on the VM: kubectl -n $NS rollout status deploy/$APP_NAME; curl http://$APP_HOST/app and http://$APP_HOST/app/health (L252-260).
- KUBECONFIG = /config/${NS}-k8s.yaml with NS = $USERNAME (L58-66). It may be <team>-k8s.yaml or a symlink (config.example:70-76).
- Human URL: https://workshop.thecosmoslabs.com, then the App button, never $INGRESS_URL (L33-41, L263-265).
- Hard rules: deploy on-cluster only; no docker build/push; no new hostname; no path other than /app (L25-33, L277-286). Web app, CLI or script only (build-day.mdc; SK/build-day-quickstart/SKILL.md:67).

VASTKIT deploy/ (this repo)
- `bash deploy/deploy.sh [APP_DIR]` follows the contract above with three changes: the ConfigMap uses `kubectl apply --server-side` (client-side apply stores the whole ConfigMap in an annotation capped at 256 KiB), the Secret values go in with `--from-file` from a private temp dir instead of `--from-literal` (so no value is on a command line, where ps shows it), and the script always restarts the rollout. INGRESS_TUNING=1 adds the body-size and timeout annotations, if allowed.
- EXTRA_SECRET_KEYS="APP_TOKEN ..." copies those variables into the Secret and the pod env. VIDEO_BUCKETS defaults to $S3_SEGMENTS_BUCKET and, when set, is added to it. ALLOW_WRITES is a plain env entry.
- deploy/app/main.py is a stdlib allowlist proxy: only the (method, route) pairs in READ_ROUTES are forwarded with the server-side JWT; WRITE_ROUTES (upload, dashboard re-ingest and its status) also need ALLOW_WRITES=1, APP_TOKEN and a matching X-App-Token header (never forwarded). DELETE/PUT/PATCH, config, auth/login, streaming/*, batch-sync/* and any route with a 'prefill' segment at any depth (tools/prefill included) are always 403. Paths are decoded and normalised before the check; ambiguous paths, chunked bodies and bad Content-Length get 400.
- GET /video?source=s3://... proxies /videos/stream (token added server-side, Range passed through) for the buckets in VIDEO_BUCKETS only; unset, /video is 403, because /videos/stream has no ACL and would stream any key the backend's S3 credentials can read. Set it to <team>-vss-chunks-segments. POST /jobs/api/v1/<route> runs a guarded call in the background and GET /jobs/<id> polls it, for calls longer than the 60 s nginx timeout. Job ids are 32 hex characters (uuid4); only the newest 256 jobs (MAX_JOBS) stay in memory, and an evicted id returns 404 "no such job".
- Allowlisted reads and /video need no token: anyone who can reach the Ingress can read the team's indexed captions, and the segments in VIDEO_BUCKETS, through it.

REFERENCE: the VSS backend itself (SP/repo_vssbp/src/deployments/vss-k8s-application/backend-deployment.yaml)
- Deployment video-backend, port 8000, SECRET_PATH = /etc/secrets/config.yaml from Secret video-backend-secret.
- Liveness and readiness probes on /health.
- Ingress path /api (Prefix) on video-lab.CLUSTER_NAME.vastdata.com.

## access

LAPTOP TO LAB VM
Source: the lab page's onboarding steps.
1. On the lab page, add an SSH public key ('Option 1 · Key') or use the password option, then 'Configure your laptop'.
2. Run `ssh <username>@bc-<username>`. The first connect opens a Cloudflare Access browser login.
3. Use the short alias bc-<username>. The full bc-pool-<NNN>.thecosmoslabs.com name fails without a ProxyCommand: it sits behind Cloudflare anycast and HTTPS 302s to thecosmoslabs.cloudflareaccess.com.
4. Fallback: 'Open desktop' in the browser.
5. Cursor connects with Remote-SSH (anysphere.remote-ssh) to the same alias. On Windows a console window may flash when the ProxyCommand runs.

SSH CONFIG
Template: templates/ssh_config.example (append it to %USERPROFILE%\.ssh\config):
  Host bc-<username>
      HostName bc-pool-<NNN>.thecosmoslabs.com
      User <username>
      IdentityFile ~/.ssh/vast_bc
      IdentitiesOnly yes
      ProxyCommand cloudflared access ssh --hostname %h
      ServerAliveInterval 30
- Create the key with: ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\vast_bc -C vast-bc-london
- cloudflared must be on PATH: `winget install --id Cloudflare.cloudflared -e` (or `winget upgrade` for an old copy). If ssh cannot find it, put its full path (with no spaces) in ProxyCommand.
- The Windows ssh-agent service is disabled by default. Rely on IdentityFile, or enable the agent yourself.
- Use the NNN assigned by the lab page.

PRIVATE: VM or cluster only, no public DNS or routing
- INGRESS_URL: the VSS backend API (video-lab-team-<N>.cosmos.vastdata.com); its /api/v1/* is the only routed path.
- The app Ingress host, same name, path /app.
- S3_ENDPOINT (data VIP) and therefore presigned playback URLs.
- VDB_ENDPOINT and the backend vdb_endpoint (Query Engine VIP).
- The K8s API (KUBECONFIG).
- In-cluster service DNS: video-backend-service:8000, video-stream-capture-service:5000, video-batch-sync-service:5000.

CLOUDFLARE ACCESS: laptop browser with SSO
- workshop.thecosmoslabs.com: VSS UI with Search, Explore and Dashboard tabs, plus the App button.
- bc-pool-<NNN>.thecosmoslabs.com: SSH only, via cloudflared.
- team-<N>-app.thecosmoslabs.com/app/: unverified, from an unmerged kit branch.
- Scripted calls: run `cloudflared access login <url>`, then send the header `cf-access-token: <output of cloudflared access token -app=<url>>` (unverified).

GPU ENDPOINTS: unresolved
- At the earlier dry run they were public IP:port (one host, ports 8001-8004) behind a bearer token.
- config.example says each has its own host:port with no token.
- Read the configured endpoint URLs without printing secrets:
  ssh bc-<u> "grep -E '^(COSMOS3_REASON_URL|COSMOS_EMBED1_URL|YOLO_URL|CANARY_1B_URL)=' /config/*.config"
  Then test reachability from both the laptop and the VM. W&B inference is public internet.

LAPTOP DEV AGAINST PRIVATE HOSTS
- Run `ssh -N -D 127.0.0.1:1080 bc-<username>` in its own terminal tab. Windows OpenSSH has no -f, so use a separate tab or Start-Process.
- Point the clients at socks5h://127.0.0.1:1080. The 'h' means remote DNS, which the video-lab-team-N names require. Python needs requests[socks] or httpx[socks].
- Apply the proxy per client, not globally: vastkit reads VAST_SOCKS and uses it in every client for every host except loopback ones; the W&B client never uses it.
- Do NOT use -L port forwards for INGRESS_URL: nginx routes by Host header and SNI.
- -L is fine for raw IP:port GPU endpoints: ssh -N -L 18001:<gpu-host>:8001 -L 18003:<gpu-host>:8003 bc-<u>
- VastDB Query Engine SQL needs the Linux ADBC driver, so run it on the VM, not through the tunnel.
- Never edit /etc/hosts (build-day.mdc).
