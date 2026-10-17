import json
import time
import uuid
from pathlib import Path

import httpx
import pytest

from vastkit import vss as vss_mod
from vastkit.config import Settings
from vastkit.http import APIError
from vastkit.vss import VSSClient, jwt_claims, strip_for_log, synthesis_error


def _offline_client(handler) -> VSSClient:
    """VSSClient on an httpx.MockTransport with a pre-set token (no server, no login)."""
    c = VSSClient(Settings(vss_url="http://vss.test", vss_username="team-test"))
    c.client.close()
    c.client = httpx.Client(base_url="http://vss.test", transport=httpx.MockTransport(handler))
    c._token, c._exp = f"t-{uuid.uuid4().hex}", time.time() + 3600
    return c


def _unreachable(req: httpx.Request) -> httpx.Response:
    raise AssertionError(f"guard should have refused before {req.method} {req.url.path}")


@pytest.fixture()
def no_sleep(monkeypatch):
    monkeypatch.setattr(vss_mod.time, "sleep", lambda s: None)
    monkeypatch.delenv("VSS_MOCK_CORPUS", raising=False)


def test_login_returns_real_jwt(vss):
    tok = vss.login()
    claims = jwt_claims(tok)
    assert claims["sub"] == "team-mock" and claims["role"] == "user"
    assert claims["exp"] - claims["iat"] == 24 * 3600


def test_bad_password_is_401(vss_mock_url):
    cfg = Settings(vss_url=vss_mock_url, vss_username="team-mock", vss_password="wrong")
    with VSSClient(cfg) as c, pytest.raises(APIError) as e:
        c.login()
    assert e.value.status_code == 401 and "Invalid username or password" in e.value.detail


def test_routes_require_auth(vss_mock_url):
    r = httpx.get(vss_mock_url + "/api/v1/auth/me")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert r.json()["detail"] == "Could not validate credentials"
    r = httpx.get(vss_mock_url + "/api/v1/auth/me", headers={"Authorization": "Bearer nope.nope.nope"})
    assert r.json()["detail"] == "Invalid token"
    # ingest-config is public
    assert httpx.get(vss_mock_url + "/api/v1/metadata/ingest-config").status_code == 200


def test_raw_token_without_bearer_prefix(vss, vss_mock_url):
    tok = vss.login()
    r = httpx.get(vss_mock_url + "/api/v1/auth/me", headers={"Authorization": tok})
    assert r.status_code == 200 and r.json()["auth_type"] == "user"


def test_relogin_on_401(vss):
    vss.login()
    vss._token = vss._token[:-3] + "abc"  # corrupt signature -> 401 Invalid token
    vss._exp = time.time() + 3600
    assert vss.me()["username"] == "team-mock"


def test_search_shapes(vss):
    r = vss.search("person walks", top_k=5)
    assert r["total"] == len(r["results"]) <= 5
    assert r["results"], "default corpus should match"
    top = r["results"][0]
    for k in ("source", "original_video", "segment_start_sec", "similarity_score", "reasoning_content"):
        assert k in top
    c = r["chunk_results"][0]
    assert any(t["is_best_match"] for t in c["timeline"])
    assert r["llm_synthesis"]["error"] is None and synthesis_error(r) is None
    assert "sql_query" not in strip_for_log(r)


def test_search_validation(vss):
    with pytest.raises(ValueError):
        vss.search("x", top_k=0)
    with pytest.raises(ValueError):
        vss.search("x", tags=["it's"])
    with pytest.raises(ValueError):
        vss.search("x", time_filter="1w")


def test_synthesis_failure_is_200(vss, monkeypatch):
    monkeypatch.setenv("MOCK_SYNTH_FAIL", "1")
    r = vss.search("person")
    assert synthesis_error(r)
    a = vss.search_and_answer("person")
    assert synthesis_error(a)


def test_ask_bug_falls_back(vss, monkeypatch):
    monkeypatch.setenv("MOCK_ASK_BUG", "1")
    a = vss.ask("person walks")
    assert a["tool_used"] == "search_hybrid" and "segment_hit_count" in a["evidence"]


def test_per_video_tools(vss):
    r = vss.search("car drives along the road", top_k=3)
    ov = r["chunk_results"][0]["original_video"]
    segs = vss.segments(ov)
    assert segs["total"] == len(segs["segments"]) > 0
    nums = [s["segment_number"] for s in segs["segments"]]
    assert nums == sorted(nums)
    src = segs["segments"][0]["source"]
    assert vss.segment(src)["source"] == src
    assert vss.metadata(src)["source"] == src
    det = vss.detections(src)
    assert det["source"] == "yolo11_coco" and det["frames"][0]["detections"][0]["bbox"]
    assert vss.segment("s3://nope/missing.mp4") is None
    syn = vss.synthesize(ov)
    assert syn["segments_used"] == segs["total"]
    a = vss.ask("what happened?", original_video=ov)
    assert a["tool_used"] == "video_segments"


def test_explore_and_metadata(vss):
    chunks = list(vss.explore_all())
    assert len(chunks) == 3
    assert vss.explore(indexed="partial")["total"] == 0
    cfg = vss.ingest_config()
    assert cfg["custom_prompt_max_length"] == 800
    assert {"value": "general", "label": "General Video Analysis"} in cfg["analysis_scenarios"]
    assert "person" in vss.metadata_values("object_classes")
    with pytest.raises(APIError):
        vss.metadata_values("filename")
    st = vss.dashboard_stats()
    assert st["overview"]["unique_videos"] == 3
    assert vss.suggestions()["prompts_table_available"] is True


def test_upload_validation(vss, tmp_path, monkeypatch):
    from vastkit.mocks import vss_mock
    monkeypatch.setattr(vss_mock, "UPLOAD_DIR", tmp_path / "uploads")  # not ~/.vastkit/mock_uploads
    bad = tmp_path / "x.txt"
    bad.write_bytes(b"0" * 20)
    with pytest.raises(PermissionError):
        vss.upload(bad)  # ownership not confirmed
    with pytest.raises(ValueError):
        vss.upload(bad, i_own_this_video=True)
    clip = tmp_path / "Clip.MP4"
    clip.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError):
        vss.upload(clip, stream_id="cam1", i_own_this_video=True)  # chunk_number missing
    with pytest.raises(ValueError):
        vss.upload(clip, stream_id="-bad", chunk_number=1, i_own_this_video=True)
    r = vss.upload(clip, tags=["demo"], custom_prompt="describe the scene", i_own_this_video=True)
    assert r["success"] and r["object_key"].endswith(".MP4") and r["object_key"].startswith("team-mock/")
    assert list((tmp_path / "uploads").iterdir())


def test_reingest_with_flaky_404s(vss):
    ov = vss.search("person")["chunk_results"][0]["original_video"]
    with pytest.raises(ValueError):
        vss.reingest(original_video=ov, stream_id="x", custom_prompt="p")
    with pytest.raises(ValueError):
        vss.reingest(original_video=ov, custom_prompt=" ")
    job = vss.reingest(original_video=ov, custom_prompt="Describe the scene in detail")
    st = vss.wait_reingest(job["job_id"], timeout=30, poll=0.3)
    assert st and st["status"] == "completed"
    segs = vss.segments(ov)["segments"]
    assert segs
    assert all(s["reasoning_content"].startswith("[re-ingested] ") for s in segs)
    assert vss.dashboard_stats()["overview"]["duplicate_segment_rows"] == 0  # updated in place


def test_upload_refuses_sample_folders(vss, tmp_path):
    d = tmp_path / "Intel"
    d.mkdir()
    clip = d / "clip.mp4"
    clip.write_bytes(b"\x00" * 64)
    with pytest.raises(PermissionError):
        vss.upload(clip, i_own_this_video=True)


@pytest.mark.parametrize("rel", ["sample-videos-master/clip-a.mp4", "Pexels Videos 1234.mp4",
                                 "my_YouTube_rips/clip.mp4", "intel-iot/clip.mp4", "Intel_samples/clip.mp4"])
def test_upload_refuses_hint_substrings(tmp_path, no_sleep, rel):
    with _offline_client(_unreachable) as c, pytest.raises(PermissionError, match="contains"):
        c.upload(tmp_path / rel, i_own_this_video=True)


@pytest.mark.parametrize("rel", ["video-intelligence-hack/recordings/own.mp4", "Intelligent_cams/rec.mp4",
                                 "IntelliJProjects/x/own.mp4"])
def test_short_hints_match_whole_words_only(tmp_path, no_sleep, rel):
    vss_mod._check_upload_path(tmp_path / rel)  # 'intel' inside a longer word is not a sample folder


def test_upload_refuses_repo_data_and_mock_corpus(tmp_path, no_sleep, monkeypatch):
    assert vss_mod.REPO_DATA == Path(__file__).resolve().parents[1] / "data"
    seg = vss_mod.REPO_DATA / "segments" / "seg-a" / "seg-a_segment_001.mp4"
    work = tmp_path / "work" / "clipA"
    work.mkdir(parents=True)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"source": "s", "_local_path": str(work / "clipA_segment_001.mp4")}]),
                      encoding="utf-8")
    with _offline_client(_unreachable) as c:
        with pytest.raises(PermissionError, match="under"):
            c.upload(seg, i_own_this_video=True)
        monkeypatch.setenv("VSS_MOCK_CORPUS", str(corpus))  # corpus JSON: its segment folders
        with pytest.raises(PermissionError, match="under"):
            c.upload(work / "clipA_segment_002.mp4", i_own_this_video=True)
        vss_mod._check_upload_path(tmp_path / "mine" / "own.mp4")  # elsewhere is fine
        monkeypatch.setenv("VSS_MOCK_CORPUS", str(tmp_path / "work"))  # a corpus folder
        with pytest.raises(PermissionError, match="under"):
            c.upload(tmp_path / "work" / "other" / "x.mp4", i_own_this_video=True)
        corpus.write_text("{not json", encoding="utf-8")
        monkeypatch.setenv("VSS_MOCK_CORPUS", str(corpus))  # unreadable: its whole folder
        with pytest.raises(PermissionError, match="under"):
            c.upload(tmp_path / "mine" / "own.mp4", i_own_this_video=True)


@pytest.mark.parametrize("fail", ["504", "read-timeout"])
def test_non_idempotent_calls_not_replayed(tmp_path, no_sleep, fail):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        if fail == "read-timeout":
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(504, text="gateway timeout")

    clip = tmp_path / "own.mp4"
    clip.write_bytes(b"\x00" * 64)
    exc = httpx.ReadTimeout if fail == "read-timeout" else APIError
    with _offline_client(handler) as c:
        with pytest.raises(exc):
            c.upload(clip, i_own_this_video=True)
        with pytest.raises(exc):
            c.reingest(original_video="v.mp4", custom_prompt="p")
        with pytest.raises(exc):
            c.delete_video("v.mp4", confirm=True)
    assert seen == [("POST", "/api/v1/videos/upload"), ("POST", "/api/v1/dashboard/reingest"),
                    ("DELETE", "/api/v1/videos")]


def test_upload_retried_when_never_sent(tmp_path, no_sleep):
    bodies = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(req.content)
        if len(bodies) == 1:
            raise httpx.ConnectError("refused", request=req)
        return httpx.Response(200, json={"success": True})

    clip = tmp_path / "own.mp4"
    clip.write_bytes(b"\x07" * 64)
    with _offline_client(handler) as c:
        assert c.upload(clip, i_own_this_video=True)["success"]
    assert len(bodies) == 2 and b"\x07" * 64 in bodies[1]


def test_reads_and_search_still_retry(no_sleep):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(503) if len(seen) % 2 else httpx.Response(200, json={"ok": True})

    with _offline_client(handler) as c:
        assert c.me() == {"ok": True} and c.search("x") == {"ok": True}
    assert seen == ["/api/v1/auth/me"] * 2 + ["/api/v1/search"] * 2


def test_delete_requires_confirm(vss):
    with pytest.raises(PermissionError):
        vss.delete_video("s3://x/y.mp4")


def test_stream_range(vss, tmp_path):
    from vastkit.mocks import vss_mock
    f = tmp_path / "seg.mp4"
    f.write_bytes(bytes(range(256)) * 4)
    row = vss_mock.ROWS[0]
    row["_local_path"] = str(f)
    full = vss.stream(row["source"])
    assert full.status_code == 200 and len(full.content) == 1024
    part = vss.stream(row["source"], "bytes=10-19")
    assert part.status_code == 206 and part.content == bytes(range(10, 20))
    assert part.headers["content-range"] == "bytes 10-19/1024"
    with pytest.raises(APIError) as e:
        vss.stream(row["source"], "bytes=5000-")
    assert e.value.status_code == 416
