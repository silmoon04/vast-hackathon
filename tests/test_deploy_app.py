"""deploy/app/main.py against the VSS mock: health, static, proxy, Range video, route guard, jobs, keep-alive, logs.
Also deploy/deploy.sh against a kubectl stub: Secret values never reach a command line."""
import http.client
import importlib.util
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

APP = Path(__file__).resolve().parents[1] / "deploy" / "app" / "main.py"


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _start(vss_url, tmp, **extra):
    """Run main.py as a subprocess on a free port -> (process, url, log path)."""
    port = _port()
    env = {k: v for k, v in os.environ.items() if k not in ("APP_TOKEN", "VIDEO_BUCKETS")}
    env.update(PORT=str(port), VSS_URL=vss_url, VSS_USERNAME="team-mock",
               VSS_PASSWORD=os.environ.get("VSS_MOCK_PASSWORD", "mock-password"), ALLOW_WRITES="0")
    env.update(extra)
    log = tmp / "app.log"
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, "-I", str(APP)], env=env, stdout=fh, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    return p, url, log


def _stop(p, log) -> str:
    p.terminate()
    p.wait(5)
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "token=" not in text and "?" not in text, "query strings (tokens) must never be logged"
    return text


def _raw(url, method, path, body=None, headers=None):
    """Send `path` byte for byte (httpx would normalise it) -> (status, body bytes)."""
    host, port = url.removeprefix("http://").split(":")
    c = http.client.HTTPConnection(host, int(port), timeout=30)
    try:
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


@pytest.fixture(scope="module")
def app(vss_mock_url, tmp_path_factory):
    p, url, log = _start(vss_mock_url, tmp_path_factory.mktemp("app"))
    yield url, log
    _stop(p, log)


@pytest.fixture(scope="module")
def app_url(app):
    return app[0]


@pytest.fixture(scope="module")
def writer(vss_mock_url, tmp_path_factory):
    """ALLOW_WRITES=1 with an APP_TOKEN made at runtime; VIDEO_BUCKETS limited to the mock's bucket."""
    from vastkit.mocks import vss_mock
    tok = secrets.token_urlsafe(24)
    bucket = vss_mock.ROWS[0]["source"].removeprefix("s3://").split("/", 1)[0]
    p, url, log = _start(vss_mock_url, tmp_path_factory.mktemp("writer"), ALLOW_WRITES="1", APP_TOKEN=tok,
                         VIDEO_BUCKETS=bucket)
    yield url, tok
    assert tok not in _stop(p, log)


@pytest.fixture(scope="module")
def main_mod():
    spec = importlib.util.spec_from_file_location("deploy_app_main", APP)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_health_and_static(app_url):
    assert httpx.get(app_url + "/health").json() == {"ok": True}
    r = httpx.get(app_url + "/")
    assert r.status_code == 200 and "Deploy check" in r.text
    assert httpx.get(app_url + "/app.js").headers["content-type"].startswith(("text/javascript", "application/javascript"))
    assert httpx.get(app_url + "/../main.py").status_code == 404
    assert httpx.get(app_url + "/main.py").status_code == 404


def test_proxy_search_and_guards(app_url):
    r = httpx.post(app_url + "/api/v1/search", json={"query": "person", "top_k": 3}, timeout=30)
    assert r.status_code == 200 and "results" in r.json()
    assert httpx.get(app_url + "/api/v1/auth/me").status_code == 200
    assert httpx.get(app_url + "/api/v1/dashboard/stats").status_code == 200
    # not on the read allowlist: secrets (config, */prefill), captures, logins, deletes, unknown routes
    for method, route in [("GET", "config"), ("POST", "auth/login"), ("GET", "streaming/prefill"),
                          ("GET", "batch-sync/prefill"), ("POST", "streaming/start"), ("POST", "streaming/stop"),
                          ("POST", "batch-sync/start"), ("POST", "batch-sync/check-objects"),
                          ("GET", "videos/playback-url"), ("DELETE", "videos?original_video=x"), ("PUT", "videos"),
                          ("PATCH", "videos"), ("GET", "search"), ("POST", "videos/explore"),
                          ("GET", "tools/prefill"), ("GET", "metadata/prefill")]:  # listed prefixes, denied segment
        assert httpx.request(method, f"{app_url}/api/v1/{route}", json={}).status_code == 403, (method, route)
    for route in ("videos/upload", "dashboard/reingest"):  # writes stay off with ALLOW_WRITES=0
        r = httpx.post(f"{app_url}/api/v1/{route}", json={})
        assert r.status_code == 403 and "writes disabled" in r.json()["detail"]


def test_encoded_paths_are_canonicalised(app_url):
    for path in ("/api/v1/%63onfig", "/api/v1/auth/%6cogin", "/api/v1/videos/%75pload", "/api/v1/dashboard/%72eingest",
                 "/api/v1//config", "/api/v1/search/../config", "/api/v1/./streaming/prefill",
                 "/api/v1/streaming%2Fprefill", "/jobs/api/v1/videos/%75pload", "/jobs/api/v1/%63onfig"):
        assert _raw(app_url, "POST", path, b"{}")[0] == 403, path
    for path in ("/api/v1/%2563onfig", "/api/v1/videos/%2575pload", "/api/v1/con%20fig", "/api/v1/config%00",
                 "/api/v1/search%3Fq=1"):
        assert _raw(app_url, "POST", path, b"{}")[0] == 400, path
    # the canonical path is what goes upstream: both reach /api/v1/search (raw '//' would 404 there)
    body = json.dumps({"query": "car", "top_k": 1}).encode()
    for path in ("/api/v1/%73earch", "/api/v1//search"):
        st, out = _raw(app_url, "POST", path, body, {"Content-Type": "application/json"})
        assert st == 200 and "results" in json.loads(out), path


def test_keepalive_drains_rejected_bodies(app):
    url, log = app
    host, port = url.removeprefix("http://").split(":")
    pw = "pw-" + secrets.token_hex(8)
    body = json.dumps({"username": "a", "password": pw}).encode()
    c, sock = http.client.HTTPConnection(host, int(port), timeout=30), None
    try:  # without the drain, the unread body is parsed as the next request line (501/400, body in the log)
        for method, path, want in [("POST", "/api/v1/auth/login", 403), ("GET", "/health", 200),
                                   ("POST", "/not-here", 404), ("GET", "/health", 200),
                                   ("DELETE", "/api/v1/videos", 403), ("GET", "/health", 200)]:
            c.request(method, path, body=body if method != "GET" else None, headers={"Content-Type": "application/json"})
            r = c.getresponse()
            assert r.status == want, (method, path, r.status)
            r.read()
            sock = sock or c.sock
            assert c.sock is sock, "every request must reuse the one keep-alive connection"
    finally:
        c.close()
    text = log.read_text(encoding="utf-8", errors="replace")
    assert pw not in text and "POST /api/v1/auth/login -> 403" in text


def test_bad_body_framing_closes(app_url):
    host, port = app_url.removeprefix("http://").split(":")
    for header, value in (("Content-Length", "-1"), ("Content-Length", "x"), ("Transfer-Encoding", "chunked")):
        c = http.client.HTTPConnection(host, int(port), timeout=30)
        try:
            c.putrequest("POST", "/api/v1/search")
            c.putheader(header, value)
            c.endheaders()
            if header == "Transfer-Encoding":
                c.send(b"2\r\n{}\r\n0\r\n\r\n")
            r = c.getresponse()
            assert r.status == 400, (header, value)
            r.read()
        finally:
            c.close()


def test_jobs(app_url):
    r = httpx.post(app_url + "/jobs/api/v1/search", json={"query": "car", "top_k": 2})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    for _ in range(100):
        j = httpx.get(f"{app_url}/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "done" and j["http_status"] == 200 and "results" in j["result"]
    assert httpx.post(app_url + "/jobs/api/v1/streaming/start", json={}).status_code == 403
    assert httpx.post(app_url + "/jobs/api/v1/dashboard/reingest", json={}).status_code == 403


def _poll(url: str, jid: str) -> dict:
    for _ in range(300):
        j = httpx.get(f"{url}/jobs/{jid}", timeout=10).json()
        if j["status"] != "running":
            return j
        time.sleep(0.1)
    return j


def test_unreachable_vss_is_502(tmp_path):
    """Nothing listens on VSS_URL, so the login fails: 502, not a 500."""
    p, url, log = _start(f"http://127.0.0.1:{_port()}", tmp_path)
    try:
        r = httpx.post(url + "/api/v1/search", json={"query": "car"}, timeout=60)
    finally:
        text = _stop(p, log)
    assert r.status_code == 502 and r.json()["detail"].startswith("upstream login failed"), r.text
    assert "Traceback" not in text


def test_dropped_login_is_502_and_jobs_end(tmp_path):
    """VSS closes every connection without a reply: 502 on the proxy and /video, and a job ends as an error."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()

    def drop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:  # closed at the end of the test
                return
            with conn:
                conn.recv(65536)

    threading.Thread(target=drop, daemon=True).start()
    p, url, log = _start(f"http://127.0.0.1:{srv.getsockname()[1]}", tmp_path, VIDEO_BUCKETS="b")
    try:
        r = httpx.post(url + "/api/v1/search", json={"query": "car"}, timeout=60)
        assert r.status_code == 502 and r.json()["detail"].startswith("upstream login failed"), r.text
        assert httpx.get(url + "/video", params={"source": "s3://b/k.mp4"}, timeout=60).status_code == 502
        jid = httpx.post(url + "/jobs/api/v1/search", json={"query": "car"}, timeout=60).json()["job_id"]
        j = _poll(url, jid)
        assert j["status"] == "error" and j["http_status"] == 502, j
    finally:
        text = _stop(p, log)
        srv.close()
    assert "Traceback" not in text


def test_job_worker_exception_ends_job(main_mod, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(main_mod, "upstream", boom)
    srv = main_mod.ThreadingHTTPServer(("127.0.0.1", 0), main_mod.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        jid = httpx.post(url + "/jobs/api/v1/search", json={"query": "car"}, timeout=10).json()["job_id"]
        j = _poll(url, jid)
    finally:
        srv.shutdown()
        srv.server_close()
    assert j["status"] == "error" and j["http_status"] == 502 and j["result"] == {"detail": "upstream error: RuntimeError"}
    assert "job error on POST /jobs/api/v1/search: RuntimeError" in capsys.readouterr().out


def test_video_range(writer, tmp_path):
    from vastkit.mocks import vss_mock  # the mock runs in-process (conftest), so a row can point at a local file
    app_url, _ = writer  # VIDEO_BUCKETS = the mock's bucket
    f = tmp_path / "seg.mp4"
    f.write_bytes(bytes(range(256)) * 4)
    row = vss_mock.ROWS[0]
    row["_local_path"] = str(f)
    r = httpx.get(app_url + "/video", params={"source": row["source"]}, headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == bytes(range(10, 20))
    assert r.headers["content-range"] == "bytes 10-19/1024" and r.headers["accept-ranges"] == "bytes"
    assert httpx.get(app_url + "/video", params={"source": row["source"]}).content == f.read_bytes()
    assert httpx.get(app_url + "/video", params={"source": "http://x"}).status_code == 400
    assert httpx.get(app_url + "/video", params={"source": "s3:///key"}).status_code == 400
    assert httpx.post(app_url + "/video", params={"source": row["source"]}).status_code == 404


def test_video_is_off_without_video_buckets(app_url):
    """/videos/stream has no ACL, so without VIDEO_BUCKETS anonymous /video would read any S3 key: 403."""
    from vastkit.mocks import vss_mock
    for src in (vss_mock.ROWS[0]["source"], "s3://other-bucket/k.mp4"):
        r = httpx.get(app_url + "/video", params={"source": src})
        assert r.status_code == 403 and r.json()["detail"] == "VIDEO_BUCKETS unset: /video is off", src
    assert httpx.get(app_url + "/video", params={"source": "http://x"}).status_code == 400


def test_writes_need_app_token(writer):
    url, tok = writer
    r = httpx.post(url + "/api/v1/dashboard/reingest", json={})
    assert r.status_code == 403 and "X-App-Token" in r.json()["detail"]
    assert httpx.post(url + "/api/v1/dashboard/reingest", json={}, headers={"X-App-Token": tok + "x"}).status_code == 403
    r = httpx.post(url + "/api/v1/dashboard/reingest", json={}, headers={"X-App-Token": tok})
    assert r.status_code == 422  # reached the mock, which wants exactly one of original_video / stream_id
    r = httpx.get(url + "/api/v1/dashboard/reingest/nope", headers={"X-App-Token": tok})
    assert r.status_code == 404 and r.json()["detail"] == "Job not found"
    assert httpx.get(url + "/api/v1/dashboard/reingest/nope").status_code == 403
    assert httpx.post(url + "/jobs/api/v1/dashboard/reingest", json={}).status_code == 403
    for method, route in [("DELETE", "videos"), ("PUT", "videos"), ("PATCH", "videos"), ("POST", "streaming/start"),
                          ("GET", "streaming/prefill"), ("GET", "batch-sync/prefill"), ("GET", "config")]:
        r = httpx.request(method, f"{url}/api/v1/{route}", headers={"X-App-Token": tok})
        assert r.status_code == 403, (method, route)
    assert httpx.post(url + "/api/v1/search", json={"query": "car", "top_k": 1}, timeout=30).status_code == 200


def test_video_buckets(writer):
    from vastkit.mocks import vss_mock
    url, _ = writer
    assert httpx.get(url + "/video", params={"source": "s3://other-bucket/k.mp4"}).status_code == 403
    assert httpx.get(url + "/video", params={"source": vss_mock.ROWS[0]["source"]}).status_code != 403


def test_canonical(main_mod):
    c = main_mod.canonical
    assert c("/api/v1/%63onfig") == "/api/v1/config"
    assert c("/api/v1//x/./y/../search/") == "/api/v1/x/search"
    assert c("/../main.py") == "/main.py" and c("/") == "/"
    for bad in ("/api/v1/%2563onfig", "/api/v1/a%20b", "/api/v1/x%00", "/api/v1/search%3Fq=1", "",
                "http://h/api/v1/config"):
        assert c(bad) is None, bad


def test_guard(main_mod, monkeypatch):
    g = main_mod.guard
    for method, route in [("GET", "auth/me"), ("HEAD", "dashboard/stats"), ("POST", "search"), ("GET", "tools/segments"),
                          ("POST", "tools/synthesize"), ("GET", "metadata/schema"), ("POST", "agent/ask"),
                          ("GET", "videos/stream")]:
        assert g(method, route, "") is None, (method, route)
    for method, route in [("GET", "tools"), ("POST", "videos/explore"), ("DELETE", "videos"), ("GET", "configx"),
                          ("GET", "config/x"), ("POST", "batch-sync/stop")]:
        assert g(method, route, "") == "not exposed through /app", (method, route)
    tok = secrets.token_urlsafe(16)
    monkeypatch.setattr(main_mod, "ALLOW_WRITES", True)
    monkeypatch.setattr(main_mod, "APP_TOKEN", "")
    assert "writes disabled" in g("POST", "videos/upload", "")  # no APP_TOKEN: writes stay off
    monkeypatch.setattr(main_mod, "APP_TOKEN", tok)
    assert g("POST", "videos/upload", tok) is None
    assert g("POST", "videos/upload", "") == g("POST", "videos/upload", tok[:-1]) == "missing or wrong X-App-Token"
    monkeypatch.setattr(main_mod, "ALLOW_WRITES", False)
    assert "writes disabled" in g("POST", "videos/upload", tok)
    monkeypatch.setattr(main_mod, "READ_ROUTES", main_mod.READ_ROUTES | {("GET", "streaming/*")})
    assert g("GET", "streaming/prefill", tok) == "not exposed through /app"  # deny wins over the lists
    assert "x-app-token" in main_mod.HOP  # never forwarded upstream


def test_guard_denies_prefill_segment(main_mod, monkeypatch):
    g = main_mod.guard
    monkeypatch.setattr(main_mod, "READ_ROUTES", main_mod.READ_ROUTES | {("GET", "videos/*")})
    for route in ("tools/prefill", "metadata/prefill", "tools/x/prefill", "videos/prefill", "prefill"):
        assert g("GET", route, "") == "not exposed through /app", route
    for route in ("tools/prefilled", "metadata/schema", "videos/x"):  # whole segments only
        assert g("GET", route, "") is None, route


def test_jobs_are_bounded(main_mod, monkeypatch):
    monkeypatch.setattr(main_mod, "MAX_JOBS", 3)
    monkeypatch.setattr(main_mod, "_jobs", {})
    jobs = [main_mod.new_job() for _ in range(5)]
    assert list(main_mod._jobs) == [j["job_id"] for j in jobs[-3:]]  # newest kept, oldest evicted first
    assert len({j["job_id"] for j in jobs}) == 5 and all(len(j["job_id"]) == 32 for j in jobs)
    assert all(main_mod._jobs[j["job_id"]] is j for j in jobs[-3:])  # workers update the stored dict
    jobs[0].update(status="done")  # a worker finishing an evicted job does not re-add it
    assert jobs[0]["job_id"] not in main_mod._jobs and main_mod.MAX_JOBS == len(main_mod._jobs)
    monkeypatch.undo()
    assert main_mod.MAX_JOBS == 256


KUBECTL_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_DIR/argv.log"
for a in "$@"; do
  case "$a" in --from-file=*=*) kv=${a#--from-file=}; cp "${kv#*=}" "$STUB_DIR/${kv%%=*}.val";; esac
done
case "$*" in *apply*) cat > /dev/null;; esac
"""


@pytest.mark.skipif(not shutil.which("bash") or "system32" in (shutil.which("bash") or "").lower(),
                    reason="needs a native bash (not the WSL launcher)")
@pytest.mark.parametrize("extra", ["APP_TOKEN UNSET_KEY_X", "APP_TOKEN VIDEO_BUCKETS UNSET_KEY_X"])
def test_deploy_script_keeps_secrets_off_the_command_line(tmp_path, extra):
    stubs, tmp, app_dir = tmp_path / "bin", tmp_path / "tmp", tmp_path / "app"
    for d in (stubs, tmp, app_dir):
        d.mkdir()
    (app_dir / "main.py").write_bytes(b"print('hi')\n")
    (app_dir / "__pycache__").mkdir()  # left by importing main.py locally; not worth the subdirectory warning
    (app_dir / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"\0")
    (stubs / "kubectl").write_bytes(KUBECTL_STUB.encode("utf-8"))  # bytes: sh needs LF line ends
    (stubs / "curl").write_bytes(b"#!/bin/sh\nprintf 200\n")
    for stub in ("kubectl", "curl"):
        os.chmod(stubs / stub, 0o755)  # a PATH lookup needs +x on Linux/macOS (MSYS bash runs them without it)
    vals = {"VSS_URL": "http://127.0.0.1:9/" + secrets.token_hex(4), "VSS_USERNAME": "u" + secrets.token_hex(4),
            "VSS_PASSWORD": "p w$x'\"" + secrets.token_hex(8), "APP_TOKEN": "t" + secrets.token_hex(8)}
    env = {**os.environ, **vals, "NS": "team", "APP_HOST": "app.test", "KUBECONFIG": str(tmp_path / "kc.yaml"),
           "TMPDIR": tmp.as_posix(), "STUB_DIR": tmp_path.as_posix(), "EXTRA_SECRET_KEYS": extra,
           "S3_SEGMENTS_BUCKET": "team-vss-chunks-segments",
           "PATH": str(stubs) + os.pathsep + os.environ.get("PATH", "")}
    env.pop("UNSET_KEY_X", None)
    env.pop("VIDEO_BUCKETS", None)
    r = subprocess.run([shutil.which("bash"), (APP.parents[1] / "deploy.sh").as_posix(), app_dir.as_posix()],
                       env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                       timeout=120)
    assert r.returncode == 0, r.stderr[-500:]
    assert "subdirectories" not in r.stderr
    argv = (tmp_path / "argv.log").read_text(encoding="utf-8")
    assert "create secret generic team-app-vss-creds" in argv and "--from-literal" not in argv
    for key, v in vals.items():
        assert v not in argv and v not in r.stdout + r.stderr, key
        assert (tmp_path / f"{key}.val").read_bytes() == v.encode("utf-8"), key  # exact bytes, no newline added
    assert "UNSET_KEY_X is empty; skipped" in r.stderr and not (tmp_path / "UNSET_KEY_X.val").exists()
    # VIDEO_BUCKETS defaults to the config's segments bucket and goes into the Secret once, listed or not
    assert (tmp_path / "VIDEO_BUCKETS.val").read_bytes() == b"team-vss-chunks-segments"
    assert argv.count("--from-file=VIDEO_BUCKETS=") == 1
    assert not list(tmp.iterdir())  # the temp secret dir is gone
