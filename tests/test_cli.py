import argparse
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from vastkit import __main__ as cli
from vastkit.config import Settings

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------- doctor
def test_doctor_without_torch_is_skip_not_fail(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    monkeypatch.setattr(cli, "load", lambda env_file=None: Settings())  # every endpoint unset -> skip
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert cli.doctor(argparse.Namespace(env_file=None, offline=False)) == 0
    out = capsys.readouterr().out
    assert "[skip] local GPU" in out and "torch not installed" in out and "0 failed" in out


def test_doctor_env_rows_show_source_not_value(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)
    for k in ("WANDB_API_KEY", "HF_TOKEN", "HF_HOME", "VAST_ENV_FILE", "VSS_MOCK", "WEAVE_PARALLELISM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WEAVE_PARALLELISM", "4")
    f = tmp_path / "team.env"
    f.write_text(f"HF_HOME={tmp_path / 'hf-cache'}\nHF_TOKEN=\nWEAVE_PARALLELISM=8\n", encoding="utf-8")
    assert cli.doctor(argparse.Namespace(env_file=str(f), offline=True)) == 0
    out = capsys.readouterr().out
    rows = {p[1]: p[2] for p in (line.split() for line in out.splitlines()) if len(p) == 3 and p[0] == "env"}
    assert rows == {"WANDB_API_KEY": "-", "HF_TOKEN": "-", "HF_HOME": "file", "VAST_ENV_FILE": "-", "VSS_MOCK": "-",
                    "WEAVE_PARALLELISM": "env"}
    assert "hf-cache" not in out and str(tmp_path) not in out


# ---------------------------------------------------------------- mock
def _run_mock(args: list[str], needle: str, timeout: float = 60.0) -> tuple[int | None, str]:
    """Run `python -m vastkit mock ...` until it exits or prints `needle`, then stop it."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen([sys.executable, "-m", "vastkit", "mock", *args], cwd=ROOT, env=env, text=True,
                         encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines: list[str] = []

    def pump() -> None:
        for line in p.stdout:
            lines.append(line)
    t = threading.Thread(target=pump, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while time.time() < deadline and p.poll() is None and not any(needle in x for x in lines):
        time.sleep(0.1)
    rc = p.poll()
    if rc is None:
        p.terminate()
        p.wait(timeout=10)
    t.join(timeout=10)
    return rc, "".join(lines)


def test_mock_exits_nonzero_when_port_in_use():
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen()
        busy = held.getsockname()[1]
        rc, out = _run_mock(["--vss-port", str(busy), "--gpu-port", str(_free_port())], needle="Ctrl+C")
    assert rc == 1, out
    assert f"VSS mock on 127.0.0.1:{busy}" in out
    assert "port in use?" in out
    assert "Ctrl+C to stop" not in out


def test_mock_on_other_ports_prints_client_urls():
    vp, gp = _free_port(), _free_port()
    rc, out = _run_mock(["--vss-port", str(vp), "--gpu-port", str(gp)], needle="MOCK_GPU_URL=")
    assert rc is None, out  # still serving until stopped
    assert f"MOCK_VSS_URL=http://127.0.0.1:{vp}" in out and f"MOCK_GPU_URL=http://127.0.0.1:{gp}" in out
    assert "VSS_MOCK=1" in out


def test_mock_returns_when_a_server_thread_dies(monkeypatch, capsys):
    import uvicorn

    class Dies:  # starts, then its thread ends (as after a crash inside uvicorn)
        def __init__(self, config):
            self.started, self.should_exit = False, False

        def run(self):
            self.started = True
            time.sleep(0.2)
    monkeypatch.setattr(uvicorn, "Server", Dies)
    assert cli.mock(argparse.Namespace(vss_port=cli.MOCK_PORTS[0], gpu_port=cli.MOCK_PORTS[1])) == 1
    cap = capsys.readouterr()
    assert "set in the client shell: VSS_MOCK=1" in cap.out and "MOCK_VSS_URL" not in cap.out
    assert "stopped unexpectedly" in cap.err


# ---------------------------------------------------------------- local
class FakeProc:
    def __init__(self, cmd, cwd=None, env=None, exits=True, stubborn=False):
        self.cmd, self.env, self.stubborn = cmd, env, stubborn
        self.returncode = 0 if exits else None
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.returncode = 1

    def kill(self):
        self.killed, self.returncode = True, -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        return self.returncode


@pytest.fixture()
def local_env(monkeypatch):
    for k in ("LOCAL_PYTHON", "LOCAL_EMBED_PYTHON", "LOCAL_YOLO_WEIGHTS", "LOCAL_COSMOS_MODEL", "HF_HOME",
              "VAST_ENV_FILE"):
        monkeypatch.delenv(k, raising=False)

    def no_server(*a, **kw):
        raise httpx.ConnectError("no server")
    monkeypatch.setattr(httpx, "get", no_server)  # never probe the real :8001-8003


def test_local_reads_env_file_and_interpreter_override(local_env, monkeypatch, tmp_path):
    f = tmp_path / "team.env"
    f.write_text(f"LOCAL_YOLO_WEIGHTS=file.pt\nHF_HOME={tmp_path}\nLOCAL_PYTHON={sys.executable}\n"
                 "LOCAL_EMBED_PYTHON=from-file\nOTHER_KEY=x\n", encoding="utf-8")
    monkeypatch.setenv("LOCAL_EMBED_PYTHON", sys.executable)  # the shell wins over the file
    started = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **kw: started.append(FakeProc(cmd, **kw)) or started[-1])
    assert cli.local(argparse.Namespace(env_file=str(f), only=["yolo", "embed"])) == 0
    assert [p.cmd[:3] for p in started] == [[sys.executable, "-m", "vastkit.local.yolo_server"],
                                             [sys.executable, "-m", "vastkit.local.embed_server"]]
    env = started[0].env
    assert env["LOCAL_YOLO_WEIGHTS"] == "file.pt" and env["HF_HOME"] == str(tmp_path)
    assert env["LOCAL_EMBED_PYTHON"] == sys.executable and "OTHER_KEY" not in env


def test_local_skips_empty_env_file_values(local_env, monkeypatch, tmp_path):
    f = tmp_path / "team.env"
    f.write_text("HF_HOME=\nLOCAL_YOLO_WEIGHTS=   # unset\nLOCAL_COSMOS_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("LOCAL_PYTHON", sys.executable)
    monkeypatch.setenv("LOCAL_COSMOS_MODEL", "")  # an empty shell value does not hide the file's
    started = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **kw: started.append(FakeProc(cmd, **kw)) or started[-1])
    assert cli.local(argparse.Namespace(env_file=str(f), only=["yolo"])) == 0
    env = started[0].env
    assert "HF_HOME" not in env and "LOCAL_YOLO_WEIGHTS" not in env and env["LOCAL_COSMOS_MODEL"] == "from-file"


def test_local_checks_interpreters_before_starting(local_env, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LOCAL_PYTHON", sys.executable)
    monkeypatch.setenv("LOCAL_EMBED_PYTHON", str(tmp_path / "missing" / "python.exe"))
    started = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **kw: started.append(cmd))
    assert cli.local(argparse.Namespace(env_file=None, only=["cosmos", "embed"])) == 2
    assert started == [] and "interpreter not found" in capsys.readouterr().err


def test_local_stops_started_children_when_a_start_fails(local_env, monkeypatch):
    monkeypatch.setenv("LOCAL_PYTHON", sys.executable)
    started = []

    def popen(cmd, **kw):
        if started:
            raise FileNotFoundError(cmd[0])
        started.append(FakeProc(cmd, exits=False, stubborn=True))  # ignores terminate -> must be killed
        return started[-1]
    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    with pytest.raises(FileNotFoundError):
        cli.local(argparse.Namespace(env_file=None, only=["cosmos", "yolo"]))
    assert started[0].terminated and started[0].killed


def test_local_interpreter_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)  # installed layout: no repo venvs
    assert cli._python({}, "LOCAL_PYTHON", ".venv") == sys.executable
    venv_py = tmp_path / ".venv-embed" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    venv_py.parent.mkdir(parents=True)
    venv_py.write_bytes(b"")
    assert cli._python({}, "LOCAL_EMBED_PYTHON", ".venv-embed") == str(venv_py)
    assert cli._python({"LOCAL_EMBED_PYTHON": "py-x"}, "LOCAL_EMBED_PYTHON", ".venv-embed") == "py-x"
