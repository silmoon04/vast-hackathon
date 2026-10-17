"""Keep Find Desk searches and playback inside the declared footage dataset."""
import copy
import importlib.util
import io
import json
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "find_desk", Path(__file__).resolve().parents[1] / "deploy" / "find-desk" / "main.py")
desk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(desk)


def test_planned_query_executes_inside_fixed_scope(monkeypatch):
    query = {"description": "Person in pink", "refinements": ["orange hardhat"],
             "camera_id": "cam-a", "query": "original query"}
    dataset = {"location": "route-demo-20261017"}
    monkeypatch.setattr(desk, "planner", SimpleNamespace(available=lambda: True,
        plan_search=lambda *args: {"query": "pink top orange hardhat", "latency_ms": 20,
                                  "model": "deepseek-ai/DeepSeek-V4.1-Flash", "tool": "search_vast_footage"}))
    calls = []
    monkeypatch.setattr(desk, "vss", lambda path, body: calls.append((path, body)) or {})
    monkeypatch.setattr(desk, "build_result", lambda *args: {"candidates": []})
    result = desk.run_search(query, dataset)
    assert calls[0][0] == "agent/search-and-answer"
    assert calls[0][1]["query"] == "pink top orange hardhat"
    assert calls[0][1]["metadata_filters"] == {"camera_id": "cam-a", "location": "route-demo-20261017"}
    assert calls[0][1]["tags"] == ["route-demo-20261017"]
    assert result["planner"]["executed"] is True
    assert query["query"] == "original query"


def test_planning_failure_searches_original_text(monkeypatch):
    def broken(*args):
        raise RuntimeError("provider unavailable")
    monkeypatch.setattr(desk, "planner", SimpleNamespace(available=lambda: True, plan_search=broken))
    calls = []
    monkeypatch.setattr(desk, "vss", lambda path, body: calls.append(body) or {})
    monkeypatch.setattr(desk, "build_result", lambda *args: {})
    query = {"description": "Person in pink", "refinements": [], "camera_id": "", "query": "original text"}
    result = desk.run_search(query, {"location": "route-demo-20261017"})
    assert calls[0]["query"] == "original text"
    assert result["planner"]["status"] == "fallback"
    assert result["planner"]["executed"] is False


@pytest.fixture
def dataset():
    return {"title": "A short route", "ready": True, "location": "route-demo-20261017",
            "license": {"name": "CC BY 4.0", "url": "https://example.org/license", "secret": "private"},  # secret-scan: allow (fixture tests field removal)
            "cameras": [{"id": "cam-a", "label": "Entrance", "original_video": "s3://team/a.mp4",
                         "duration_seconds": 60, "clock_start_seconds": 36000,
                         "source_url": "https://example.org/a"},
                        {"id": "cam-b", "label": "Corridor", "original_video": "s3://team/b.mp4",
                         "duration_seconds": 45, "clock_start_seconds": 36000}]}


@pytest.fixture
def query(dataset):
    return desk.validate_query({"description": "Adult wearing a blue top", "refinements": ["carrying a bag"]}, dataset)


@pytest.fixture
def response():
    return {"answer": "A person in a blue top passes the entrance.", "evidence": {
        "chunk_count": 1, "embedding_time_ms": 17, "search_time_ms": 12,
        "chunks": [{"original_video": "s3://team/a.mp4", "best_match_start_sec": 10,
                    "best_match_end_sec": 15, "similarity_score": .72}],
        "llm_synthesis": {"response": "A person in a blue top.", "error": None,
                          "processing_time": .9, "model": "caption-model"}}}


@pytest.fixture(autouse=True)
def clear_cache():
    desk.JOBS.clear()
    desk.VIDEOS.clear()
    desk.READINESS.update(key="", ready=False)
    yield


def test_public_manifest_never_exposes_storage_or_unlisted_metadata(dataset):
    result = desk.public_dataset(dataset)
    encoded = json.dumps(result)
    assert result["ready"] and len(result["cameras"]) == 2
    assert "s3://" not in encoded and "original_video" not in encoded and "private" not in encoded
    assert result["license"]["url"] == "https://example.org/license"
    dataset["cameras"][0]["source_url"] = "https://user:password@example.org/a"
    assert desk.public_dataset(dataset)["cameras"][0]["source_url"] == ""


def test_platform_agent_call_is_scoped_and_refinements_are_retained(dataset, query, response, monkeypatch):
    calls = []

    def upstream(path, body=None, params=None, timeout=None):
        calls.append((path, body, params))
        if path == "agent/search-and-answer":
            return response
        assert path == "tools/segments"
        assert params == {"original_video": "s3://team/a.mp4"}
        assert 0 < timeout <= desk.CAPTION_FETCH_SECONDS
        return {"segments": [{"original_video": "s3://team/a.mp4", "segment_start_sec": 10,
                              "segment_end_sec": 15, "reasoning_content": "A blue top and a bag."}]}

    monkeypatch.setattr(desk, "vss", upstream)
    result = desk.run_search(query, dataset)
    path, body, _ = calls[0]
    assert path == "agent/search-and-answer"
    assert body["metadata_filters"] == {"location": "route-demo-20261017"}
    assert body["tags"] == ["route-demo-20261017"] and body["include_public"] is False
    assert "Adult wearing a blue top" in body["query"] and "carrying a bag" in body["query"]
    candidate = result["candidates"][0]
    assert candidate["caption"] == "A blue top and a bag."
    assert candidate["camera_id"] == "cam-a" and candidate["clock_start_seconds"] == 36000
    assert candidate["video_url"] == "video?id=" + candidate["id"]
    assert desk.VIDEOS[candidate["id"]] == "s3://team/a.mp4"
    assert "s3://" not in json.dumps(result) and "sql_query" not in result
    assert result["platform"]["synthesis_ms"] == 900


def test_outside_dataset_evidence_fails_before_any_clip_is_cached(dataset, query, response, monkeypatch):
    response["evidence"]["chunks"].append({"original_video": "s3://another/private.mp4"})
    monkeypatch.setattr(desk, "caption_for", lambda *args: pytest.fail("must check all scopes first"))
    with pytest.raises(desk.ScopeError, match="outside this demo"):
        desk.build_result(response, dataset, query, time.perf_counter())
    assert not desk.VIDEOS


def test_selected_camera_is_enforced_in_request_and_response(dataset, query, response):
    query["camera_id"] = "cam-b"
    assert desk.search_body(query, dataset)["metadata_filters"]["camera_id"] == "cam-b"
    with pytest.raises(desk.ScopeError, match="selected camera"):
        desk.build_result(response, dataset, query, time.perf_counter())


def test_incomplete_or_missing_evidence_never_returns_generated_answer(dataset, query, response):
    response["evidence"]["chunk_count"] = 2
    with pytest.raises(desk.ScopeError, match="incomplete"):
        desk.build_result(response, dataset, query, time.perf_counter())
    with pytest.raises(desk.ScopeError, match="verifiable"):
        desk.build_result({"answer": "An unsupported answer."}, dataset, query, time.perf_counter())


def test_nested_synthesis_failure_preserves_scoped_candidates(dataset, query, response, monkeypatch):
    response["answer"] = "Failed to generate AI synthesis: secret upstream detail"
    response["evidence"]["llm_synthesis"]["error"] = "upstream private detail"
    monkeypatch.setattr(desk, "caption_for", lambda *args: "A person wearing blue.")
    result = desk.build_result(response, dataset, query, time.perf_counter())
    assert len(result["candidates"]) == 1
    assert result["synthesis_ok"] is False and result["answer_error"]
    assert "private" not in json.dumps(result) and "secret" not in json.dumps(result)
    assert "still review" in result["answer"]


def test_invalid_times_are_dropped_and_clip_end_is_bounded(dataset, query, response, monkeypatch):
    chunks = response["evidence"]["chunks"]
    chunks[0].update(best_match_start_sec=58, best_match_end_sec=65)
    for start, end in [(-1, 5), (8, 4), (60, 70), (float("nan"), 5)]:
        row = copy.deepcopy(chunks[0])
        row.update(best_match_start_sec=start, best_match_end_sec=end)
        chunks.append(row)
    response["evidence"]["chunk_count"] = len(chunks)
    monkeypatch.setattr(desk, "caption_for", lambda *args: "A person.")
    result = desk.build_result(response, dataset, query, time.perf_counter())
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["end_seconds"] == 60


def test_stalled_caption_fetches_cannot_exceed_total_budget_or_hide_playable_matches(dataset, query, response, monkeypatch):
    release = threading.Event()
    calls = []
    monkeypatch.setattr(desk, "CAPTION_BUDGET_SECONDS", .12)
    monkeypatch.setattr(desk, "CAPTION_FETCH_SECONDS", .1)

    def stalled_caption(source, start, end, timeout):
        calls.append(timeout)
        release.wait(2)
        return "Late caption."

    monkeypatch.setattr(desk, "caption_for", stalled_caption)
    chunks = response["evidence"]["chunks"]
    for start in (20, 30):
        row = dict(chunks[0], best_match_start_sec=start, best_match_end_sec=start + 5)
        chunks.append(row)
    response["evidence"]["chunk_count"] = len(chunks)
    began = time.monotonic()
    try:
        result = desk.build_result(response, dataset, query, time.perf_counter())
        assert time.monotonic() - began < .8
        assert len(result["candidates"]) == 3
        assert all(candidate["caption"] == "Open this clip to check the clothing and direction."
                   for candidate in result["candidates"])
        assert len(calls) <= 2 and all(0 < timeout <= .1 for timeout in calls)
        assert all(candidate["id"] in desk.VIDEOS for candidate in result["candidates"])
    finally:
        release.set()
    # Both abandoned workers must return their slots before another search.
    assert desk.CAPTION_SLOTS.acquire(timeout=1)
    assert desk.CAPTION_SLOTS.acquire(timeout=1)
    desk.CAPTION_SLOTS.release()
    desk.CAPTION_SLOTS.release()


@pytest.mark.parametrize("body", [None, [], {}, {"description": " "}, {"description": "a" * 401},
                                {"description": "blue", "refinements": "red"},
                                {"description": "blue", "refinements": [""]},
                                {"description": "blue", "refinements": ["x"] * 7},
                                {"description": "blue", "refinements": ["x" * 201]},
                                {"description": "blue\x00top"}, {"description": "blue", "camera_id": "other"}])
def test_invalid_queries_are_rejected(dataset, body):
    with pytest.raises(ValueError):
        desk.validate_query(body, dataset)


def test_manifest_scope_is_required_and_untrusted_scope_values_never_reach_platform(dataset, query):
    for value in [None, "", "x' OR 1=1", {"location": "anything"}]:
        dataset["location"] = value
        with pytest.raises(desk.PreparingError):
            desk.search_body(query, dataset)


def test_removed_manifest_clip_cannot_still_be_streamed(dataset, monkeypatch):
    cid = "a" * 20
    desk.VIDEOS[cid] = "s3://team/a.mp4"
    monkeypatch.setattr(desk, "load_dataset", lambda: dataset)
    assert desk.video_source(cid) == "s3://team/a.mp4"
    dataset["cameras"] = []
    with pytest.raises(ValueError, match="expired"):
        desk.video_source(cid)


def test_async_jobs_keep_errors_private_and_release_search_slot(dataset, monkeypatch):
    monkeypatch.setattr(desk, "load_dataset", lambda: dataset)
    monkeypatch.setattr(desk, "run_search", lambda *args: (_ for _ in ()).throw(RuntimeError("password=private")))
    accepted = desk.start_search({"description": "A blue coat"})
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = desk.public_job(accepted["job_id"])
        if job["status"] != "running":
            break
        time.sleep(.01)
    assert job["status"] == "error" and "private" not in json.dumps(job)
    assert desk.SEARCH_SLOTS.acquire(blocking=False)
    assert desk.SEARCH_SLOTS.acquire(blocking=False)
    desk.SEARCH_SLOTS.release()
    desk.SEARCH_SLOTS.release()


def test_two_running_searches_are_the_limit_and_jobs_are_bounded(dataset, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(desk, "load_dataset", lambda: dataset)
    monkeypatch.setattr(desk, "run_search", lambda *args: release.wait(2) or {})
    try:
        desk.start_search({"description": "A blue coat"})
        desk.start_search({"description": "A red coat"})
        with pytest.raises(desk.PreparingError, match="Two searches"):
            desk.start_search({"description": "A green coat"})
    finally:
        release.set()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(job["status"] == "running" for job in desk.JOBS.values()):
        time.sleep(.01)
    monkeypatch.setattr(desk, "MAX_JOBS", 2)
    monkeypatch.setattr(desk, "run_search", lambda *args: {})
    for _ in range(3):
        job = desk.start_search({"description": "A blue coat"})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and desk.public_job(job["job_id"])["status"] == "running":
            time.sleep(.01)
    assert len(desk.JOBS) == 2


def test_empty_manifest_has_preparing_state_and_does_not_start_search(dataset, monkeypatch):
    dataset.update(ready=False, cameras=[])
    monkeypatch.setattr(desk, "load_dataset", lambda: dataset)
    assert desk.public_dataset()["status"] == "Footage preparing"
    with pytest.raises(desk.PreparingError, match="Footage preparing"):
        desk.start_search({"description": "A blue top"})
    assert not desk.JOBS


def test_http_boundary_has_json_body_limit_and_static_allowlist(dataset, monkeypatch):
    monkeypatch.setattr(desk, "load_dataset", lambda: dataset)
    server = desk.ThreadingHTTPServer(("127.0.0.1", 0), desk.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/dataset") as result:
            assert "original_video" not in result.read().decode()
        for path in ["/main.py", "/../main.py", "/.env"]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(base + path)
            assert error.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base + "/api/search", data=b"a" * 16385))
        assert error.value.code == 413
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(base + "/api/search", data=b"[]"))
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def wav_audio(channels=1, sample_width=2, rate=16000, seconds=.1):
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(channels)
        recording.setsampwidth(sample_width)
        recording.setframerate(rate)
        recording.writeframes(b"\0" * (int(seconds * rate) * channels * sample_width))
    return output.getvalue()


@pytest.fixture
def speech_config(monkeypatch):
    monkeypatch.setattr(desk, "CANARY_URL", "https://provided.example.org/canary")
    monkeypatch.setattr(desk, "GPU_BEARER_TOKEN", "test-private-token")


@pytest.mark.parametrize("audio", [b"not a recording", wav_audio(channels=2), wav_audio(rate=48000),
                                  wav_audio(sample_width=1), wav_audio(seconds=20.1), wav_audio(seconds=0),
                                  wav_audio()[:-2], wav_audio() + b"extra"],
                         ids=["not-wav", "stereo", "48khz", "pcm8", "over20s", "empty", "truncated", "trailing"])
def test_speech_rejects_wrong_format_duration_and_truncation(audio, speech_config, monkeypatch):
    monkeypatch.setattr(desk, "open_speech_request", lambda *args: pytest.fail("invalid audio must not leave the app"))
    with pytest.raises(ValueError, match="mono 16 kHz PCM16"):
        desk.transcribe(audio)


def test_speech_multipart_uses_provided_endpoint_and_keeps_headers_private(speech_config, monkeypatch):
    audio = wav_audio(seconds=20)
    sent = []

    def fake_request(request):
        sent.append(request)
        return io.BytesIO(b'{"text":"  Adult wearing blue.  ","language_code":"en-US"}')

    monkeypatch.setattr(desk, "open_speech_request", fake_request)
    result = desk.transcribe(audio)
    request = sent[0]
    assert request.full_url == "https://provided.example.org/canary/v1/audio/transcriptions"
    assert request.get_header("Authorization") == "Bearer test-private-token"
    assert request.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert b'name="language"\r\n\r\nen-US\r\n' in request.data
    assert b'name="file"; filename="speech.wav"' in request.data and audio in request.data
    assert b'name="model"' not in request.data
    assert result["text"] == "Adult wearing blue." and result["provider"] == "Canary" and result["editable"]
    assert result["latency_ms"] >= 0
    assert "test-private-token" not in json.dumps(result) and "provided.example.org" not in json.dumps(result)


@pytest.mark.parametrize("reply", [b'{"error":"private provider error"}', b'not json', b'{"text":" "}',
                                  json.dumps({"text": "x" * 4001}).encode(), b"x" * 65537],
                         ids=["error-only", "invalid-json", "empty-text", "long-text", "oversized"])
def test_speech_bad_provider_responses_are_generic_and_release_slot(reply, speech_config, monkeypatch, capsys):
    monkeypatch.setattr(desk, "open_speech_request", lambda request: io.BytesIO(reply))
    with pytest.raises(desk.SpeechError, match="could not transcribe") as error:
        desk.transcribe(wav_audio())
    assert "private" not in str(error.value) and not capsys.readouterr().out
    assert desk.SPEECH_SLOTS.acquire(blocking=False)
    desk.SPEECH_SLOTS.release()


def test_speech_transport_failure_does_not_log_or_return_url_token_or_audio(speech_config, monkeypatch, capsys):
    def failing_request(request):
        raise urllib.error.URLError("https://provided.example.org test-private-token raw-recording")

    monkeypatch.setattr(desk, "open_speech_request", failing_request)
    with pytest.raises(desk.SpeechError) as error:
        desk.transcribe(wav_audio())
    assert "provided.example.org" not in str(error.value) and "test-private-token" not in str(error.value)
    assert "raw-recording" not in str(error.value) and not capsys.readouterr().out
    assert desk.SPEECH_SLOTS.acquire(blocking=False)
    desk.SPEECH_SLOTS.release()


def test_speech_is_independent_of_dataset_and_needs_both_config_values(speech_config, monkeypatch):
    monkeypatch.setattr(desk, "load_dataset", lambda: pytest.fail("speech must not read the video manifest"))
    assert desk.speech_available()
    monkeypatch.setattr(desk, "GPU_BEARER_TOKEN", "")
    assert not desk.speech_available()
    with pytest.raises(desk.PreparingError, match="not configured"):
        desk.transcribe(wav_audio())
    monkeypatch.setattr(desk, "GPU_BEARER_TOKEN", "a")
    for url in ["", "file:///tmp/audio", "https://user:password@example.org", "https://example.org?token=private"]:
        monkeypatch.setattr(desk, "CANARY_URL", url)
        assert not desk.speech_available()


def test_speech_allows_only_one_recording_at_once(speech_config, monkeypatch):
    monkeypatch.setattr(desk, "open_speech_request", lambda *args: pytest.fail("no second provider call"))
    assert desk.SPEECH_SLOTS.acquire(blocking=False)
    try:
        with pytest.raises(desk.PreparingError, match="already processing"):
            desk.transcribe(wav_audio())
    finally:
        desk.SPEECH_SLOTS.release()


def test_speech_redirects_cannot_forward_bearer_to_another_host():
    assert desk.NoSpeechRedirect().redirect_request(None, None, 302, "Found", {}, "https://other.example.org") is None


def test_speech_http_routes_validate_audio_before_forwarding(speech_config, monkeypatch):
    calls = []

    def provider(request):
        calls.append(request)
        return io.BytesIO(b'{"text":"A blue coat"}')

    monkeypatch.setattr(desk, "open_speech_request", provider)
    server = desk.ThreadingHTTPServer(("127.0.0.1", 0), desk.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/speech") as response:
            assert json.load(response) == {"available": True, "provider": "Canary", "mode": "push-to-talk"}
        for data, content_type, expected in [(b"a" * ((1 << 20) + 1), "audio/wav", 413),
                                             (wav_audio(), "audio/webm", 400),
                                             (b"not wav", "audio/wav", 400)]:
            request = urllib.request.Request(base + "/api/transcribe", data=data, headers={"Content-Type": content_type})
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == expected
        assert not calls
        request = urllib.request.Request(base + "/api/transcribe", data=wav_audio(), headers={"Content-Type": "audio/wav"})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["text"] == "A blue coat"
        assert len(calls) == 1
        monkeypatch.setattr(desk, "GPU_BEARER_TOKEN", "")
        with urllib.request.urlopen(base + "/api/speech") as response:
            assert json.load(response)["available"] is False
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 503 and len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()


def write_manifest(folder, dataset):
    (folder / "dataset.json").write_text(json.dumps(dataset), encoding="utf-8")


def test_auto_readiness_starts_false_until_every_declared_clip_is_complete(dataset, monkeypatch, tmp_path):
    dataset.update(auto_ready=True, ready=False)
    monkeypatch.setattr(desk, "ROOT", tmp_path)
    write_manifest(tmp_path, dataset)
    rows = [{"original_video": "s3://team/a.mp4", "location": dataset["location"]}]
    calls = []

    def catalogue(path, params=None, timeout=None):
        calls.append((path, params, timeout))
        return {"chunks": list(rows), "total": len(rows), "table_available": True}

    monkeypatch.setattr(desk, "vss", catalogue)
    assert desk.load_dataset()["ready"] is False and not calls
    desk.refresh_readiness()
    assert desk.load_dataset()["ready"] is False
    rows.append({"original_video": "s3://team/b.mp4", "location": "different-location"})
    desk.refresh_readiness()
    assert desk.load_dataset()["ready"] is False
    rows[-1]["location"] = dataset["location"]
    desk.refresh_readiness()
    assert desk.load_dataset()["ready"] is True
    assert calls[-1] == ("videos/explore", {"scope": "mine", "location": "route-demo-20261017",
                                          "indexed": "complete", "limit": 100, "offset": 0}, 30)
    assert json.loads((tmp_path / "dataset.json").read_text())["ready"] is False


def test_auto_readiness_resets_when_manifest_changes_or_check_fails(dataset, monkeypatch, tmp_path):
    dataset.update(auto_ready=True, ready=False)
    monkeypatch.setattr(desk, "ROOT", tmp_path)
    write_manifest(tmp_path, dataset)
    monkeypatch.setattr(desk, "check_dataset_ready", lambda value: True)
    desk.refresh_readiness()
    assert desk.load_dataset()["ready"]
    dataset["cameras"][0]["original_video"] = "s3://team/replacement.mp4"
    write_manifest(tmp_path, dataset)
    assert desk.load_dataset()["ready"] is False
    monkeypatch.setattr(desk, "check_dataset_ready", lambda value: (_ for _ in ()).throw(OSError("private failure")))
    desk.refresh_readiness()
    assert desk.load_dataset()["ready"] is False


def test_readiness_result_for_replaced_manifest_is_discarded(dataset, monkeypatch, tmp_path):
    dataset.update(auto_ready=True, ready=False)
    monkeypatch.setattr(desk, "ROOT", tmp_path)
    write_manifest(tmp_path, dataset)

    def replace_while_query_runs(value):
        value["cameras"][0]["original_video"] = "s3://team/new.mp4"
        write_manifest(tmp_path, value)
        return True

    monkeypatch.setattr(desk, "check_dataset_ready", replace_while_query_runs)
    desk.refresh_readiness()
    assert desk.load_dataset()["ready"] is False


def test_readiness_missing_table_and_empty_manifest_stay_false(dataset, monkeypatch):
    monkeypatch.setattr(desk, "vss", lambda *args, **kwargs: {"table_available": False, "chunks": []})
    assert desk.check_dataset_ready(dataset) is False
    dataset["cameras"] = []
    assert desk.check_dataset_ready(dataset) is False
