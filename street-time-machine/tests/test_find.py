"""stm.find: request routing parses tool calls; speech is validated and forwarded to the Find Desk, never called live."""
from __future__ import annotations

import io
import json
import os
import struct
import urllib.error

os.environ.setdefault("STM_WARM", "0")

import pytest
from fastapi.testclient import TestClient

from stm import find, server


def wav(seconds: float = 0.5) -> bytes:
    n = int(16000 * seconds)
    return (b"RIFF" + struct.pack("<I", 36 + n * 2) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
            + b"data" + struct.pack("<I", n * 2) + b"\0\0" * n)


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def client():
    return TestClient(server.app)


def test_parse_call_structured_and_text():
    msg = {"tool_calls": [{"function": {"name": "pick_candidate", "arguments": '{"number": 2}'}}]}
    assert find.parse_call(msg) == ("pick_candidate", {"number": 2})
    text = {"content": 'I will call {"name": "search_people", "arguments": {"description": "red hoodie"}}'}
    assert find.parse_call(text) == ("search_people", {"description": "red hoodie"})
    assert find.parse_call({"content": "no idea"}) is None


def test_transcribe_forwards_wav(client, monkeypatch):
    seen = {}

    def fake(req, timeout):
        seen.update(url=req.full_url, ctype=req.headers.get("Content-type"), n=len(req.data), timeout=timeout)
        return Resp(json.dumps({"text": " red hoodie ", "latency_ms": 431, "provider": "Canary"}).encode())

    monkeypatch.setattr(find.urllib.request, "urlopen", fake)
    r = client.post("/api/find/transcribe", content=wav(), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "red hoodie" and body["provider"] == "Canary" and body["provider_ms"] == 431
    assert seen["url"] == find.STT_URL and seen["ctype"] == "audio/wav" and seen["n"] == len(wav())


@pytest.mark.parametrize("audio", [b"", b"not a recording", b"RIFF" + b"\0" * 60, wav(40)],
                         ids=["empty", "not-wav", "riff-not-wave", "over-1MiB"])
def test_transcribe_rejects_bad_audio(client, monkeypatch, audio):
    monkeypatch.setattr(find.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call the service"))
    r = client.post("/api/find/transcribe", content=audio, headers={"Content-Type": "audio/wav"})
    assert r.status_code == 400


def test_transcribe_reports_upstream_errors(client, monkeypatch):
    def refuse(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"detail": "Record mono 16 kHz"}'))

    monkeypatch.setattr(find.urllib.request, "urlopen", refuse)
    r = client.post("/api/find/transcribe", content=wav(), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 400 and "mono 16 kHz" in r.json()["detail"]["error"]

    def down(req, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(find.urllib.request, "urlopen", down)
    r = client.post("/api/find/transcribe", content=wav(), headers={"Content-Type": "audio/wav"})
    assert r.status_code == 503 and "unreachable" in r.json()["detail"]["error"]


def test_speech_status(client, monkeypatch):
    monkeypatch.setattr(find.urllib.request, "urlopen",
                        lambda url, timeout: Resp(b'{"available": true, "provider": "Canary", "mode": "push-to-talk"}'))
    assert client.get("/api/find/speech").json() == {"available": True, "provider": "Canary"}

    def down(url, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(find.urllib.request, "urlopen", down)
    assert client.get("/api/find/speech").json()["available"] is False
