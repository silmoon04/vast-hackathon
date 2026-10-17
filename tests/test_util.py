import base64
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from vastkit import util

FAKE = "zq" + "7" * 6  # fake value, built at runtime


def test_scrub_key_names():
    v = FAKE
    cases = [
        f"PASSWORD={v} SECRET_KEY={v} ACCESS_KEY={v}",
        f"AWS_SECRET_ACCESS_KEY={v}\nAWS_ACCESS_KEY_ID={v}\nAWS_SESSION_TOKEN={v}",
        json.dumps({"s3_access_key": v, "s3_secret_key": v, "vdb_secret_key": v, "access_key": v,
                    "token": v, "secret": v, "gpu_bearer_token": v}),
        str({"password": v, "token": v}),  # single-quoted Python repr
        json.dumps({"resp": json.dumps({"s3_secret_key": v})}),  # JSON inside a JSON string field
        f"X-Api-Key: {v}",
        f'SECRET_KEY="{v} {v}"',
        f"https://h/x?a=1&X-Amz-Signature={v}&X-Amz-Credential={v}",
    ]
    for c in cases:
        assert v not in util.scrub(c), c
    assert util.scrub(f"SECRET_KEY={v}&n=1") == "SECRET_KEY=<redacted>&n=1"
    assert util.scrub(str({"token": v})) == "{'token': '<redacted>'}"


def test_scrub_keeps_json_valid_and_non_secrets():
    rec = {"token": 123, "secret": None, "api_key": FAKE, "nested": {"password": FAKE},
           "max_tokens": 512, "prompt_tokens": 9, "token_type": "bearer", "tokenizer": "x"}
    got = json.loads(util.scrub(json.dumps(rec)))
    assert got["token"] == got["api_key"] == got["nested"]["password"] == "<redacted>"
    assert got["secret"] is None and FAKE not in json.dumps(got)
    assert {k: got[k] for k in ("max_tokens", "prompt_tokens", "token_type", "tokenizer")} == \
        {"max_tokens": 512, "prompt_tokens": 9, "token_type": "bearer", "tokenizer": "x"}
    inner = json.loads(json.loads(util.scrub(json.dumps({"body": json.dumps({"token": 5, "s3_access_key": FAKE})})))["body"])
    assert inner == {"token": "<redacted>", "s3_access_key": "<redacted>"}


def test_jsonl_log_scrubs_key_names(tmp_path):
    log = util.JsonlLog(tmp_path / "log.jsonl")
    log.write("prefill", resp={"s3_access_key": FAKE, "s3_secret_key": FAKE}, text=json.dumps({"token": FAKE}), n=2)
    assert FAKE not in (tmp_path / "log.jsonl").read_text(encoding="utf-8")
    rec = log.read()[0]
    assert rec["resp"] == {"s3_access_key": "<redacted>", "s3_secret_key": "<redacted>"} and rec["n"] == 2


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def test_scrub():
    pw, tok = "hun" + "ter2", ".".join(("abc", "def", "ghi"))  # fake values, built at runtime
    jwt = ".".join(_b64url(b) for b in (json.dumps({"alg": "HS256"}).encode(), b'{"sub":"team"}', os.urandom(12)))
    s = util.scrub(f'GET /api/v1/videos/stream?source=s3://b/k.mp4&token={tok} "password": "{pw}" '
                   f"Authorization: Bearer {jwt}")
    assert tok not in s and pw not in s and not any(part in s for part in jwt.split("."))
    assert "token=<redacted>" in s
    assert "<url>" in util.scrub("see https://host.example/x?y=1", urls=True)


def test_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "CACHE_ROOT", tmp_path)
    calls = []

    @util.disk_cache("t")
    def f(x, y=1):
        calls.append(x)
        return {"v": x + y}

    assert f(1) == {"v": 2} and f(1) == {"v": 2} and calls == [1]
    assert f(1, y=2) == {"v": 3} and calls == [1, 1]
    f(1, _refresh=True)
    assert calls == [1, 1, 1]
    clip = tmp_path / "a.bin"
    clip.write_bytes(b"one")
    f2 = util.disk_cache("t2")(lambda p: p.read_bytes() if hasattr(p, "read_bytes") else open(p, "rb").read())
    assert f2(str(clip)) == b"one"
    clip.write_bytes(b"two")  # content changed -> new key
    assert f2(str(clip)) == b"two"


def test_disk_cache_concurrent_identical_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "CACHE_ROOT", tmp_path)
    gate = threading.Barrier(8)

    @util.disk_cache("race")
    def f(x):
        gate.wait(timeout=5)  # all writers finish together and race on the same entry
        return b"x" * 200_000

    for _ in range(5):
        with ThreadPoolExecutor(8) as ex:
            assert list(ex.map(lambda _: f(1, _refresh=True), range(8))) == [b"x" * 200_000] * 8
    assert f(1) == b"x" * 200_000
    assert [p.suffix for p in f.cache_dir.iterdir()] == [".pkl"]  # no stray temp files


def test_content_key_paths(tmp_path):
    missing, clip = tmp_path / "out.mp4", tmp_path / "a.bin"
    assert util.content_key(missing) == util.content_key(str(missing))  # not hashed by content, no error
    util.content_key(tmp_path)  # a directory Path is keyed by name
    clip.write_bytes(b"one")
    k1 = util.content_key(clip)
    clip.write_bytes(b"two")
    assert util.content_key(clip) != k1 and util.content_key(clip) == util.content_key(str(clip))


def test_jsonl_log(tmp_path):
    log = util.JsonlLog(tmp_path / "x" / "log.jsonl")
    log.write("call", url="https://h/stream?token=secret123", n=1)
    recs = log.read()
    assert recs[0]["event"] == "call" and "secret123" not in recs[0]["url"]


def test_timed():
    sink = {}
    with util.timed("a", sink) as t:
        pass
    assert t["s"] >= 0 and "a" in sink
