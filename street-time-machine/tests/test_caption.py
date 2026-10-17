"""Tests for stm.caption. The HTTP call is always faked: no network."""
from __future__ import annotations

import base64
import io
import json
import socket
import threading
import time
import urllib.request

import pytest

from stm import caption


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("tests must not open a network connection")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


def _crop(tmp_path, name: str, data: bytes = b"\xff\xd8 fake jpeg \xff\xd9") -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def _reply(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _fake(monkeypatch, text_or_exc, seen: list | None = None):
    def post(url, body, timeout):
        if seen is not None:
            seen.append((url, body, timeout))
        if isinstance(text_or_exc, BaseException):
            raise text_or_exc
        return _reply(text_or_exc)
    monkeypatch.setattr(caption, "_post", post)


def test_good_json(tmp_path, monkeypatch):
    seen: list = []
    _fake(monkeypatch, '{"person": true, "top": "red hoodie", "bottom": "black jeans", "items": ["grey backpack"]}', seen)
    p = _crop(tmp_path, "a.jpg")
    [r] = caption.describe_crops([p], url="http://relay.test/v1/chat/completions")
    assert r["ok"] is True and r["person"] is True
    assert r["desc"] == "red hoodie, black jeans, grey backpack"
    assert r["look"] == {"top": "red", "bottom": "black", "items": ["backpack"]}
    url, body, timeout = seen[0]
    assert url == "http://relay.test/v1/chat/completions" and timeout == 8
    assert body["model"] == "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
    assert body["temperature"] == 0 and body["max_tokens"] == 120
    img = body["messages"][1]["content"][1]["image_url"]["url"]
    assert img.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(img.split(",", 1)[1]) == open(p, "rb").read()
    prompt = (body["messages"][0]["content"] + body["messages"][1]["content"][0]["text"]).lower()
    for w in ("identity", "age", "gender", "ethnicity", "face", "no person", "json"):
        assert w in prompt or (w == "no person" and '"person": false' in prompt)


def test_json_inside_prose_and_normalised_words(tmp_path, monkeypatch):
    _fake(monkeypatch, 'Sure! Here is what I see:\n```json\n{"person": true, "top": "Gray Sweater", '
                       '"bottom": "dark blue pants", "items": []}\n```\nHope this helps {not json}.')
    [r] = caption.describe_crops([_crop(tmp_path, "a.jpg")])
    assert r["ok"] and r["desc"] == "grey hoodie, navy trousers"
    assert r["look"] == {"top": "grey", "bottom": "navy", "items": []}


def test_person_words_are_dropped(tmp_path, monkeypatch):
    _fake(monkeypatch, '<think>looks like a young man</think>{"person": true, "top": "young man in a red jacket", '
                       '"bottom": {"colour": "black", "garment": "jeans"}, "items": ["brown handbag", "none"]}')
    [r] = caption.describe_crops([_crop(tmp_path, "a.jpg")])
    assert r["ok"] and r["desc"] == "red jacket, black jeans, brown bag"
    assert r["look"] == {"top": "red", "bottom": "black", "items": ["bag"]}
    assert not {"man", "young"} & set(r["desc"].replace(",", " ").split())


def test_no_person(tmp_path, monkeypatch):
    _fake(monkeypatch, '{"person": false}')
    [r] = caption.describe_crops([_crop(tmp_path, "a.jpg")])
    assert r["ok"] is True and r["person"] is False and r["desc"] == ""
    _fake(monkeypatch, "There is no person in this crop, only a bench.")
    [r] = caption.describe_crops([_crop(tmp_path, "b.jpg")])
    assert r["ok"] is True and r["person"] is False and r["desc"] == ""


def test_timeout_and_failures_fall_back(tmp_path, monkeypatch):
    for exc in (socket.timeout("timed out"), TimeoutError(), ConnectionRefusedError(), ValueError("bad body")):
        _fake(monkeypatch, exc)
        [r] = caption.describe_crops([_crop(tmp_path, "a.jpg")], timeout=0.5)
        assert r["ok"] is False and r["desc"] == "" and r["error"].startswith("request")
    _fake(monkeypatch, "I'd rather not say.")
    [r] = caption.describe_crops([_crop(tmp_path, "a.jpg")])
    assert r["ok"] is False and r["desc"] == "" and r["error"] == "no JSON in reply"
    _fake(monkeypatch, '{"person": true, "top": null, "bottom": null, "items": []}')
    [r] = caption.describe_crops([_crop(tmp_path, "a.jpg")])
    assert r["ok"] is False and r["desc"] == ""
    [r] = caption.describe_crops([str(tmp_path / "missing.jpg")])
    assert r["ok"] is False and r["error"].startswith("read")


def test_thread_pool_keeps_order(tmp_path, monkeypatch):
    colours = ["red", "blue", "green", "black", "white", "yellow", "pink"]
    paths = [_crop(tmp_path, f"{c}.jpg", c.encode()) for c in colours]
    active, peak, lock = [0], [0], threading.Lock()

    def post(url, body, timeout):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        c = base64.b64decode(body["messages"][1]["content"][1]["image_url"]["url"].split(",", 1)[1]).decode()
        time.sleep(0.02 * (len(c) % 3))
        with lock:
            active[0] -= 1
        return _reply(json.dumps({"person": True, "top": f"{c} coat", "bottom": None, "items": []}))
    monkeypatch.setattr(caption, "_post", post)
    out = caption.describe_crops(paths, workers=3)
    assert [r["look"]["top"] for r in out] == colours
    assert 1 < peak[0] <= 3
    assert caption.describe_crops([]) == []


def test_default_url_from_env_and_real_post_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("STM_FIND_LLM", "http://relay.test:9/v1/chat/completions")
    seen: list = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        seen.append((req.full_url, req.get_header("Content-type"), json.loads(req.data), timeout))
        return Resp(json.dumps(_reply('{"person": true, "top": "white shirt", "bottom": "beige shorts", '
                                      '"items": []}')).encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    [r] = caption.describe_crops([_crop(tmp_path, "a.jpg")], timeout=3)
    assert r["ok"] and r["desc"] == "white shirt, beige shorts"
    url, ctype, body, timeout = seen[0]
    assert url == "http://relay.test:9/v1/chat/completions" and ctype == "application/json" and timeout == 3
    assert "authorization" not in {k.lower() for k in body} and body["max_tokens"] == 120


def test_parse_json_helper():
    assert caption.parse_json('noise {"a": 1} more {"b": 2}') == {"a": 1}
    assert caption.parse_json("{broken {\"person\": false}") == {"person": False}
    assert caption.parse_json("nothing here") is None
