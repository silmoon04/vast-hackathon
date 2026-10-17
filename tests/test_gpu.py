import json
import shutil
import subprocess

import httpx
import pytest

from vastkit.config import Settings
from vastkit.gpu import CanaryASR, CosmosEmbed, CosmosReason, CosmosResult, YoloClient, cosine
from vastkit.gpu.cosmos import _clamp_fps, extract_json, mmss_to_sec, split_reasoning
from vastkit.gpu.yolo import unique_counts, with_times
from vastkit.http import APIError


@pytest.fixture()
def cfg(gpu_mock_url):
    return Settings(cosmos_reason_url=gpu_mock_url, cosmos_embed_url=gpu_mock_url, yolo_url=gpu_mock_url,
                    canary_url=gpu_mock_url)


@pytest.fixture()
def clip(tmp_path):
    p = tmp_path / "c.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)
    return p


def test_parsers():
    t, a = split_reasoning("<think>\nbecause\n</think>\n\n<answer>\nyes\n</answer>")
    assert (t, a) == ("because", "yes")
    assert split_reasoning("plain")[1] == "plain"
    assert split_reasoning("because\n</think>\nyes") == ("because", "yes")  # opening tag in the template
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('noise [1, 2] tail') == [1, 2]
    assert extract_json('Result: [{"a": 1}] done') == [{"a": 1}]  # '[' before '{': the list, not the inner dict
    assert extract_json('x {"a": [1]} y') == {"a": [1]}
    assert mmss_to_sec("01:02.50") == 62.5 and mmss_to_sec("<3.5 seconds>") == 3.5


def test_mmss_to_sec_prose():
    assert mmss_to_sec("3.5 s.") == 3.5 and mmss_to_sec("00:12.5 (approx.)") == 12.5
    assert mmss_to_sec("unknown.") == 0.0 and mmss_to_sec("...") == 0.0 and mmss_to_sec("") == 0.0
    assert mmss_to_sec("from 3 to 5 s") == 3.0 and mmss_to_sec("1:02:03") == 3723.0


def test_cosmos_discovers_model_and_parses(cfg, clip):
    c = CosmosReason(cfg=cfg, cache=False)
    assert c.model == "nvidia/cosmos3-nano-reasoner"  # discovered, not the 404ing config id
    r = c.ask_video(clip, "What is in the video?", reasoning=True, fps=4)
    assert r.think and r.answer.startswith("Mock answer")
    ev = c.events(clip, fps=2)
    assert ev[0]["start_sec"] == 0.5 and ev[1]["end_sec"] == 4.5


def _sent_bodies(c: CosmosReason) -> list:
    sent: list = []
    c.client.event_hooks = {"request": [lambda req: sent.append(json.loads(req.content))
                                        if req.method == "POST" else None], "response": []}
    return sent


def test_cosmos_guards(cfg, clip):
    c = CosmosReason(cfg=cfg, cache=False)
    with pytest.raises(ValueError):
        c.ask_video(clip, "x", fps=2, num_frames=8)
    # fps above the source is clamped client-side when source_fps is known
    assert c.ask_video(clip, "x", fps=60, source_fps=30).answer
    # unknown source fps: the server's 'exceeds video fps 30.0' 400 is retried once at the clip fps
    sent = _sent_bodies(c)
    r = c.ask_video(clip, "x", fps=60)
    assert r.answer and not r.dropped and not c.unsupported
    assert [b["media_io_kwargs"]["video"]["fps"] for b in sent] == [60, 29.9]
    # a value error is never remembered: later calls still send their own sampling
    c.ask_video(clip, "x", fps=2)
    assert sent[-1]["media_io_kwargs"] == {"video": {"fps": 2}}
    # fps with num_frames is a value error: it raises instead of silently dropping the sampling
    with pytest.raises(APIError):
        c.chat([{"role": "user", "content": "x"}], extra={"media_io_kwargs": {"video": {"fps": 2, "num_frames": 8}}})
    assert not c.unsupported


def test_clamp_fps():
    assert _clamp_fps({"video": {"fps": 8}}, "requested fps 8 exceeds video fps 4") == {"video": {"fps": 4.0}}
    assert _clamp_fps({"fps": 30}, "requested fps 30 exceeds video fps 23.98") == {"fps": 23.97}
    assert _clamp_fps({"video": {"fps": 2}}, "requested fps 2 exceeds video fps 4") is None
    assert _clamp_fps({"video": {"num_frames": 8}}, "exceeds video fps 4") is None
    assert _clamp_fps({"video": {"fps": 8}}, "fps too high") is None


def _fake(handler) -> CosmosReason:
    """CosmosReason over an in-memory transport (no network)."""
    c = CosmosReason("http://fake.invalid", model="m", cfg=Settings(), cache=False)
    c.client = httpx.Client(base_url="http://fake.invalid", transport=httpx.MockTransport(handler))
    return c


def _ok(content: str = "ok") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})


def test_cosmos_unknown_field_is_remembered():
    seen: list = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        if "top_k" in body:
            return httpx.Response(422, json={"detail": [{"type": "extra_forbidden", "loc": ["body", "top_k"],
                                                         "msg": "Extra inputs are not permitted"}]})
        return _ok()

    c = _fake(handler)
    r = c.ask_text("hi", extra={"top_k": 5})
    assert r.answer == "ok" and r.dropped == ["top_k"] and c.unsupported == {"top_k"}
    c.ask_text("hi", extra={"top_k": 5})
    assert len(seen) == 3 and "top_k" not in seen[-1]


def test_cosmos_value_error_drops_for_one_request_and_skips_cache(clip, tmp_path, monkeypatch):
    import vastkit.gpu.cosmos as cm
    monkeypatch.setattr(cm, "CACHE_DIR", tmp_path / "cache")
    seen: list = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"detail": "invalid json_schema: bad type"})
        return _ok('{"a": 1}')

    c = _fake(handler)
    c.cache = True
    r = c.ask_video(clip, "q", json_schema={"type": "object"})
    assert r.json == {"a": 1} and r.dropped == ["response_format"] and not c.unsupported
    assert not (tmp_path / "cache").exists()  # an answer made without the requested schema is not cached
    assert not c.ask_video(clip, "q", json_schema={"type": "object"}).cached
    assert "response_format" in seen[-2]  # the second call asked for the schema again


def test_cosmos_fps_clamp_retries_once():
    calls: list = []

    def handler(req):
        fps = json.loads(req.content)["media_io_kwargs"]["video"]["fps"]
        calls.append(fps)
        return httpx.Response(400, json={"detail": f"Requested fps {fps} exceeds video fps 4.00"})

    c = _fake(handler)
    with pytest.raises(APIError):
        c.ask_video(b"\x00" * 64, "q", fps=8)
    assert calls == [8, 3.99] and not c.unsupported


def test_cosmos_events_parsing(cfg):
    c = CosmosReason(cfg=cfg, cache=False)

    def events_for(answer: str) -> list:
        c.ask_video = lambda *a, **k: CosmosResult(answer=answer)
        return c.events(b"")

    one = {"start_sec": 1.0, "end_sec": 3.0, "caption": "a"}
    assert events_for('Here is the result: [{"start": "00:01.00", "end": "00:03.00", "caption": "a"}]') == [one]
    assert events_for('{"start": "00:01.00", "end": "00:03.00", "caption": "a"}') == [one]
    assert events_for('{"events": [{"start": "1 s.", "end": "3.0 sec.", "caption": "a"}]}') == [one]
    # parsed dict with no usable items: fall back to the regex salvage
    assert events_for('{"note": 1} "start": "00:01", "end": "00:03", "caption": "a"') == [one]
    assert events_for('{"note": 1}') == []


def test_cosmos_ready_uses_health_ready():
    def server(ready_code: int):
        return lambda req: httpx.Response(ready_code if req.url.path == "/v1/health/ready" else 200, json={})

    assert _fake(server(200)).ready() is True
    assert _fake(server(503)).ready() is False  # loading: /v1/models answering 200 does not count
    assert _fake(server(404)).ready() is True  # no health route: fall back to /v1/models
    e = CosmosEmbed("http://fake.invalid", cfg=Settings())
    for code, want in ((200, True), (503, False), (405, True)):
        e.client = httpx.Client(base_url="http://fake.invalid", transport=httpx.MockTransport(server(code)))
        assert e.ready() is want


def test_cosmos_bad_model_rediscovers(cfg):
    c = CosmosReason(cfg=cfg, model="nvidia/cosmos3-reason", cache=False)
    assert c.ask_text("hello").answer
    assert c.model == "nvidia/cosmos3-nano-reasoner"


def test_cosmos_cache(cfg, clip, tmp_path, monkeypatch):
    import vastkit.gpu.cosmos as cm
    monkeypatch.setattr(cm, "CACHE_DIR", tmp_path / "cache")
    c = CosmosReason(cfg=cfg, cache=True)
    a = c.ask_video(clip, "cache me")
    b = c.ask_video(clip, "cache me")
    assert not a.cached and b.cached and a.answer == b.answer
    # the key includes base_url: the same model id behind another URL (mock vs hosted) misses
    other = CosmosReason(cfg=cfg, cache=True)
    other.base_url = "https://cosmos.example.invalid"
    assert other.model == c.model and not other.ask_video(clip, "cache me").cached


def test_cosmos_cache_default(cfg):
    assert CosmosReason(cfg=cfg).cache is False  # loopback (mock / local mirror): off by default
    assert CosmosReason("http://[::1]:9", cfg=Settings()).cache is False
    assert CosmosReason("https://cosmos.example.invalid", cfg=Settings()).cache is True
    assert CosmosReason("https://cosmos.example.invalid", cfg=Settings(mock=True)).cache is False


def test_embed(cfg, clip):
    e = CosmosEmbed(cfg=cfg)
    v = e.text("a dog")
    assert len(v) == 256 and abs(sum(x * x for x in v) - 1) < 1e-6
    assert cosine(v, e.text("a dog")) == pytest.approx(1.0) and cosine(v, [0.0] * 256) == 0.0
    assert len(e.texts(["a", "b"])) == 2 and len(e.video(clip)) == 256
    assert not hasattr(e, "rank")  # no ranking helpers in the toolkit


def test_embed_dim_check(cfg):
    e = CosmosEmbed(cfg=cfg, dim=768)
    with pytest.raises(RuntimeError):
        e.text("x")


def test_yolo(cfg, clip):
    y = YoloClient(cfg=cfg)
    assert y.ready()
    r = y.infer(clip)
    assert r["perception_ok"] and r["frames"]
    assert unique_counts(r["frames"]) == {"person": 1, "truck": 1}
    assert with_times(r["frames"], 30.0)[3]["time_sec"] == 0.1
    with pytest.raises(ValueError):
        y.infer(b"tiny")


def test_canary(cfg, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)
    c = CanaryASR(cfg=cfg)
    assert c.ready()
    assert "mock transcript" in c.transcribe(wav)["text"]
    assert wav.exists()  # the caller's own audio file is never deleted


def test_canary_deletes_its_temp_wav(cfg, clip, tmp_path, monkeypatch):
    from vastkit.gpu import canary
    made = []

    def fake_wav(video, out=None, *, rate=16000):
        made.append(tmp_path / "extracted.wav")
        made[-1].write_bytes(b"RIFF" + b"\x00" * 100)
        return made[-1]

    monkeypatch.setattr(canary, "extract_wav", fake_wav)
    assert "mock transcript" in CanaryASR(cfg=cfg).transcribe(clip)["text"]
    assert made and not made[0].exists() and clip.exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_canary_extract_wav_temp_file_is_closed(tmp_path):
    from vastkit.gpu import canary
    src = tmp_path / "tone.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440", "-t", "1",
                    "-c:a", "aac", str(src)], check=True)
    wav = canary.extract_wav(src)
    assert wav.read_bytes()[:4] == b"RIFF"
    wav.unlink()  # an fd left open by mkstemp makes this fail on Windows


def test_scan_events_salvages_broken_json():
    from vastkit.gpu.cosmos import scan_events
    raw = '```json\n[{"type": "string", "start": "00:02.51",\n "end": "00:07.50",\n "caption": "A man \\"walks\\" in."}\n ]}'
    assert scan_events(raw) == [{"start": "00:02.51", "end": "00:07.50", "caption": 'A man \\"walks\\" in.'}]
