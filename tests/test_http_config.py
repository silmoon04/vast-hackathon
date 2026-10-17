"""vastkit.config parsing/precedence/mock mode and vastkit.http proxy routing."""
import os
import re
import socket
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from vastkit import config
from vastkit.config import MOCK_GPU_URL, MOCK_VSS_URL, Settings, load, parse_env_file
from vastkit.http import make_client

ROOT = Path(__file__).resolve().parents[1]
GPU_ATTRS = ("cosmos_reason_url", "cosmos_embed_url", "yolo_url", "canary_url")
_CONFIG_ENV = re.compile(r"^(VSS_|VAST_|COSMOS|YOLO_|CANARY_|GPU_|S3_|VDB_|VASTDB_|WANDB_|MOCK_VSS_URL$|"
                         r"MOCK_GPU_URL$|INGRESS_URL$|USERNAME$|PASSWORD$|ACCESS_KEY$|SECRET_KEY$)")


@pytest.fixture()
def clean_env(monkeypatch):
    """No config keys from the developer's shell; restored after the test."""
    for k in list(os.environ):
        if _CONFIG_ENV.match(k.upper()):
            monkeypatch.delenv(k, raising=False)
    config.reset()
    yield monkeypatch
    config.reset()


def _env_file(tmp_path: Path, text: str, encoding: str = "utf-8") -> str:
    p = tmp_path / "team.env"
    p.write_text(text, encoding=encoding)
    return str(p)


def _dead_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------- parse_env_file ----------------

def test_bom_does_not_drop_first_key(tmp_path):
    vals = parse_env_file(_env_file(tmp_path, "VSS_URL=http://10.0.0.5:8000\nYOLO_URL=x\n", "utf-8-sig"))
    assert vals == {"VSS_URL": "http://10.0.0.5:8000", "YOLO_URL": "x"}


def test_comment_after_empty_value_is_empty(tmp_path):
    vals = parse_env_file(_env_file(tmp_path, "VAST_SOCKS=      # socks5h://127.0.0.1:1080 when tunnelling\n"
                                              "VSS_MOCK=0   # 1 = mock\n"
                                              "A=\t#\n"
                                              "B='x # y'\n"
                                              "C=#Abc123xyz\n"  # a value that starts with '#' (bash keeps it)
                                              "D=#\n"))
    assert vals == {"VAST_SOCKS": "", "VSS_MOCK": "0", "A": "", "B": "x # y", "C": "#Abc123xyz", "D": "#"}


def test_env_example_parses_to_empty_or_plain_values():
    path = ROOT / ".env.example"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            assert "#" not in line, f"inline comment on a key line: {line.split('=', 1)[0]}"
    vals = parse_env_file(path)
    assert vals and all(not v.startswith("#") for v in vals.values())
    assert vals["VAST_SOCKS"] == "" and vals["VSS_MOCK"] == "0"


def test_env_example_template_loads_clean(clean_env):
    s = load(str(ROOT / ".env.example"))
    assert not s.mock and not s.socks_proxy and not s.gpu_bearer_token and not s.vss_url


# ---------------- precedence ----------------

def test_primary_key_in_file_beats_fallback_alias_in_env(clean_env, tmp_path):
    clean_env.setenv("S3_ENDPOINT", "http://s3-vip:80")
    clean_env.setenv("INGRESS_URL", "http://ingress:80")
    s = load(_env_file(tmp_path, "VDB_ENDPOINT=http://qe-vip:80\nVSS_BASE_URL=http://vss:80\n"))
    assert s.vdb_endpoint == "http://qe-vip:80"
    assert s.vss_url == "http://vss:80"
    assert s.s3_endpoint == "http://s3-vip:80"


def test_env_still_beats_file_for_the_same_key(clean_env, tmp_path):
    clean_env.setenv("VDB_ENDPOINT", "http://env-qe:80")
    s = load(_env_file(tmp_path, "VDB_ENDPOINT=http://file-qe:80\nS3_ENDPOINT=http://s3:80\n"))
    assert s.vdb_endpoint == "http://env-qe:80"


def test_fallback_alias_used_when_primary_unset(clean_env, tmp_path):
    s = load(_env_file(tmp_path, "S3_ENDPOINT=http://s3:80\nINGRESS_URL=http://ingress:80\n"))
    assert s.vdb_endpoint == "http://s3:80" and s.vss_url == "http://ingress:80"


# ---------------- mock mode ----------------

def test_mock_overrides_urls_credentials_and_socks(clean_env, tmp_path):
    pw = uuid.uuid4().hex
    text = (f"VSS_MOCK=1\nVSS_BASE_URL=http://127.0.0.1:9000\nVSS_USERNAME=team-7\nVSS_PASSWORD={pw}\n"
            "VAST_SOCKS=socks5h://127.0.0.1:1080\n" + "".join(
                f"{k}=http://10.0.0.9:9{i}00\n" for i, k in
                enumerate(("COSMOS3_REASON_URL", "COSMOS_EMBED1_URL", "YOLO_URL", "CANARY_1B_URL"))))
    s = load(_env_file(tmp_path, text))
    assert s.mock and s.vss_url == MOCK_VSS_URL
    assert (s.vss_username, s.vss_password) == ("team-mock", "mock-password")
    assert all(getattr(s, a) == MOCK_GPU_URL for a in GPU_ATTRS)
    assert s.socks_proxy == ""


def test_mock_clears_gpu_token_and_storage_keys(clean_env, tmp_path):
    for k in ("GPU_BEARER_TOKEN", "VAST_ACCESS_KEY", "VAST_SECRET_KEY"):
        clean_env.setenv(k, uuid.uuid4().hex)
    file_keys = "".join(f"{k}={uuid.uuid4().hex}\n" for k in ("ACCESS_KEY", "SECRET_KEY"))
    s = load(_env_file(tmp_path, "VSS_MOCK=1\n" + file_keys))
    assert s.mock and (s.gpu_bearer_token, s.access_key, s.secret_key) == ("", "", "")
    s = load(_env_file(tmp_path, file_keys))  # without mock mode the same keys load
    assert s.gpu_bearer_token and s.access_key and s.secret_key


def test_mock_urls_and_credentials_follow_overrides(clean_env, tmp_path):
    team, pw = "team-x", uuid.uuid4().hex
    clean_env.setenv("MOCK_VSS_URL", "http://127.0.0.1:9765")
    clean_env.setenv("VSS_MOCK_TEAM", team)
    clean_env.setenv("VSS_MOCK_PASSWORD", pw)
    s = load(_env_file(tmp_path, "VSS_MOCK=1\nMOCK_GPU_URL=http://127.0.0.1:9766\n"))
    assert s.vss_url == "http://127.0.0.1:9765"
    assert all(getattr(s, a) == "http://127.0.0.1:9766" for a in GPU_ATTRS)
    assert (s.vss_username, s.vss_password) == (team, pw)


def test_mock_mode_reaches_mock_on_custom_port_despite_dead_socks(clean_env, tmp_path, vss_mock_url):
    from vastkit.mocks import vss_mock
    from vastkit.vss import VSSClient
    clean_env.setenv("VSS_MOCK", "1")
    clean_env.setenv("MOCK_VSS_URL", vss_mock_url)
    clean_env.setenv("VAST_SOCKS", f"socks5h://127.0.0.1:{_dead_port()}")
    clean_env.setenv("VSS_MOCK_TEAM", vss_mock.TEAM)
    clean_env.setenv("VSS_MOCK_PASSWORD", vss_mock.PASSWORD)
    with VSSClient(load(_env_file(tmp_path, ""))) as c:
        assert c.me()["username"] == vss_mock.TEAM


# ---------------- describe ----------------

def test_describe_shows_only_set_or_unset():
    userinfo = "u:" + uuid.uuid4().hex
    s = Settings(vss_url="http://10.0.0.5:8000", vss_username="team-7",
                 socks_proxy=f"socks5h://{userinfo}@10.0.0.1:1080", s3_endpoint="http://172.16.0.1:80",
                 access_key=uuid.uuid4().hex, mock=True, source_file="/config/t.config")
    view = s.describe()
    assert set(view.values()) <= {"<set>", "<unset>"}
    assert view["vss_url"] == view["socks_proxy"] == view["mock"] == "<set>"
    assert view["canary_url"] == "<unset>"
    assert userinfo not in repr(view) and "10.0.0.5" not in repr(view)


# ---------------- make_client proxy routing ----------------

class _Ok(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


@pytest.fixture()
def local_http():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_loopback_bypasses_socks_proxy(local_http):
    with make_client(local_http, proxy=f"socks5h://127.0.0.1:{_dead_port()}", timeout=5) as c:
        assert c.get("/").text == "ok"


def test_proxy_still_used_for_remote_hosts():
    with make_client("http://10.0.0.5:8000", proxy="socks5h://127.0.0.1:1080") as c:
        assert c._transport_for_url(httpx.URL("http://10.0.0.5:8000/x")) is not c._transport
        for u in ("http://127.0.0.1:8001/", "http://localhost:8002/", "http://[::1]:8003/"):
            assert c._transport_for_url(httpx.URL(u)) is c._transport
