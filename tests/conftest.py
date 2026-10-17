import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(app, port: int):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return server


@pytest.fixture(scope="session")
def vss_mock_url():
    os.environ.setdefault("MOCK_REINGEST_SECONDS", "2")
    from vastkit.mocks import vss_mock
    port = _free_port()
    server = _serve(vss_mock.app, port)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture(scope="session")
def gpu_mock_url():
    from vastkit.mocks import gpu_mock
    port = _free_port()
    server = _serve(gpu_mock.app, port)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture()
def vss(vss_mock_url):
    from vastkit.config import Settings
    from vastkit.vss import VSSClient
    cfg = Settings(vss_url=vss_mock_url, vss_username="team-mock", vss_password="mock-password")
    with VSSClient(cfg) as c:
        yield c
