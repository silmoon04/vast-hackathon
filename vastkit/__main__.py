"""vastkit CLI.

    python -m vastkit doctor            # tools, GPU, config (names only) and live endpoint checks
    python -m vastkit models            # model ids served by Cosmos / W&B Inference
    python -m vastkit mock              # VSS mock :8765 + GPU mock :8766 in one process
    python -m vastkit local             # start the local GPU mirrors (cosmos :8001, yolo :8002, embed :8003)

Output never includes URLs, tokens or passwords: config is shown as set/unset (env rows: where it is set,
env / file / -), errors are scrubbed.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .config import find_config_file, load, parse_env_file
from .util import scrub

ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, fn: Callable[[], Any], results: list) -> None:
    t0 = time.perf_counter()
    try:
        out = fn()
        status = "skip" if isinstance(out, str) and out.startswith("skip:") else "ok"
        detail = out[5:].strip() if status == "skip" else out
    except Exception as e:
        status, detail = "FAIL", f"{type(e).__name__}: {e}"
    dt = time.perf_counter() - t0
    results.append((name, status))
    print(f"  [{status:4s}] {name:30s} {dt:5.1f}s  {scrub(detail, urls=True)[:220]}")


def doctor(a: argparse.Namespace) -> int:
    cfg = load(a.env_file)
    results: list = []
    print(f"vastkit {__version__} | python {platform.python_version()} | {platform.system()} {platform.release()}")
    print(f"config file: {'found' if cfg.source_file else 'none'}  mock: {cfg.mock}")

    print("\nTools")
    for tool in ("ffmpeg", "ffprobe", "ssh", "kubectl", "git", "nvidia-smi", "uv"):
        _check(tool, lambda t=tool: "found" if shutil.which(t) else "skip: not on PATH", results)

    def gpu():
        try:
            import torch
        except ImportError:  # optional (requirements-gpu.txt); not a failure
            return "skip: torch not installed"
        if not torch.cuda.is_available():
            return "skip: CUDA not available"
        p = torch.cuda.get_device_properties(0)
        return f"{p.name}, {p.total_memory / 2**30:.1f} GiB, torch {torch.__version__}"
    _check("local GPU", gpu, results)

    print("\nConfig (names only)")
    for k in ("vss_url", "vss_username", "vss_password", "cosmos_reason_url", "cosmos_embed_url", "yolo_url",
              "canary_url", "gpu_bearer_token", "s3_endpoint", "access_key", "secret_key", "vdb_endpoint",
              "vastdb_bucket", "wandb_api_key", "wandb_entity", "wandb_project", "socks_proxy"):
        v = getattr(cfg, k)
        print(f"  {k:22s} {'set' if v else '-'}")
    in_file = {k for k, v in parse_env_file(cfg.source_file).items() if v} if cfg.source_file else set()
    for envname in ("WANDB_API_KEY", "HF_TOKEN", "HF_HOME", "VAST_ENV_FILE", "VSS_MOCK", "WEAVE_PARALLELISM"):
        where = "env" if os.environ.get(envname) else "file" if envname in in_file else "-"  # source, never value
        print(f"  env {envname:18s} {where}")

    if a.offline:
        return 0
    print("\nEndpoints")

    def vss():
        if not cfg.vss_url:
            return "skip: VSS URL unset"
        from .vss import VSSClient
        with VSSClient(cfg=cfg) as c:
            me = c.me()
            return f"login ok as {me.get('username', '?')}"
    _check("VSS login + /auth/me", vss, results)

    def cosmos():
        if not cfg.cosmos_reason_url:
            return "skip: COSMOS3_REASON_URL unset"
        from .gpu import CosmosReason
        c = CosmosReason(cfg=cfg)
        return f"ready={c.ready()} model={c.model}"
    _check("Cosmos reason", cosmos, results)

    def embed():
        if not cfg.cosmos_embed_url:
            return "skip: COSMOS_EMBED1_URL unset"
        from .gpu import CosmosEmbed
        e = CosmosEmbed(cfg=cfg)
        return f"ready={e.ready()} dim={len(e.text('test'))}"  # neutral probe text
    _check("Cosmos embed1", embed, results)

    def yolo():
        if not cfg.yolo_url:
            return "skip: YOLO_URL unset"
        from .gpu import YoloClient
        h = YoloClient(cfg=cfg).health()
        return f"ok={h.get('ok')} model={h.get('model')} loaded={h.get('model_loaded')}"
    _check("YOLO", yolo, results)

    def canary():
        if not cfg.canary_url:
            return "skip: CANARY_1B_URL unset"
        from .gpu import CanaryASR
        return f"ready={CanaryASR(cfg=cfg).ready()}"
    _check("Canary ASR", canary, results)

    def wandb():
        if not (cfg.wandb_api_key or os.environ.get("WANDB_API_KEY")):
            return "skip: WANDB_API_KEY unset"
        from .llm import LLM
        ids = LLM(cfg=cfg).models()
        return f"{len(ids)} models"
    _check("W&B Inference", wandb, results)

    def vdb():
        if not (cfg.vdb_endpoint and cfg.access_key):
            return "skip: VDB endpoint/keys unset"
        from . import vdb as _vdb
        with _vdb.table(cfg=cfg) as t:
            cols = getattr(t, "arrow_schema", None)
            return f"opened; columns={len(cols) if cols is not None else '?'}"
    _check("VastDB vss-collection", vdb, results)

    fails = [n for n, s in results if s == "FAIL"]
    print(f"\n{len(results) - len(fails)} ok/skip, {len(fails)} failed" + (f": {', '.join(fails)}" if fails else ""))
    return 1 if fails else 0


def models(a: argparse.Namespace) -> int:
    cfg = load(a.env_file)
    if cfg.cosmos_reason_url:
        from .gpu import CosmosReason
        print("cosmos:", CosmosReason(cfg=cfg).models())
    if cfg.wandb_api_key or os.environ.get("WANDB_API_KEY"):
        from .llm import LLM
        for m in LLM(cfg=cfg).models():
            print("wandb:", m)
    return 0


MOCK_PORTS = (8765, 8766)  # VSS, GPU: where config's default MOCK_VSS_URL / MOCK_GPU_URL point


def mock(a: argparse.Namespace) -> int:
    import uvicorn
    from vastkit.mocks import gpu_mock, vss_mock
    ports = {"VSS": a.vss_port, "GPU": a.gpu_port}
    apps = {"VSS": vss_mock.app, "GPU": gpu_mock.app}
    servers = {k: uvicorn.Server(uvicorn.Config(apps[k], host="127.0.0.1", port=p, log_level="warning"))
               for k, p in ports.items()}
    threads = {k: threading.Thread(target=s.run, daemon=True) for k, s in servers.items()}
    for t in threads.values():
        t.start()
    # uvicorn's bind failure only ends its own thread (sys.exit in a thread), so check startup here
    deadline = time.time() + 5
    while time.time() < deadline and not all(s.started for s in servers.values()) \
            and all(t.is_alive() for t in threads.values()):
        time.sleep(0.05)
    bad = [k for k in servers if not (servers[k].started and threads[k].is_alive())]
    if bad:
        for s in servers.values():
            s.should_exit = True
        print("could not start " + ", ".join(f"{k} mock on 127.0.0.1:{ports[k]}" for k in bad)
              + " (port in use?); stop the other process or pick --vss-port/--gpu-port", file=sys.stderr, flush=True)
        return 1
    print(f"VSS mock :{a.vss_port}  GPU mock :{a.gpu_port}   Ctrl+C to stop", flush=True)
    if (a.vss_port, a.gpu_port) == MOCK_PORTS:
        print("set in the client shell: VSS_MOCK=1", flush=True)
    else:
        print(f"set in the client shell (or .env):\n  VSS_MOCK=1\n  MOCK_VSS_URL=http://127.0.0.1:{a.vss_port}\n"
              f"  MOCK_GPU_URL=http://127.0.0.1:{a.gpu_port}", flush=True)
    try:
        while all(t.is_alive() for t in threads.values()):
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    print("a mock server stopped unexpectedly", file=sys.stderr, flush=True)
    return 1


def _local_key(k: str) -> bool:
    return k.startswith(("LOCAL_", "HF_"))


def _python(env: dict[str, str], override: str, venv: str) -> str:
    """Interpreter for a local server: env override, else the repo venv if present, else this one."""
    if env.get(override):
        return env[override]
    p = ROOT / venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(p) if p.is_file() else sys.executable


def _stop(procs: list[subprocess.Popen], grace: float = 10.0) -> None:
    """Terminate the children, then kill any still running after `grace` seconds."""
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + grace
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def local(a: argparse.Namespace) -> int:
    env = dict(os.environ)
    cfg_file = a.env_file or find_config_file()
    if cfg_file:  # model locations (and interpreter overrides) may live in .env rather than the shell
        for k, v in parse_env_file(cfg_file).items():
            if v and _local_key(k) and not env.get(k):  # `KEY=` never replaces a server default with ''
                env[k] = v
    py = _python(env, "LOCAL_PYTHON", ".venv")
    py_embed = _python(env, "LOCAL_EMBED_PYTHON", ".venv-embed")
    plan = {"cosmos": ([py, "-m", "vastkit.local.cosmos_edge_server", "--port", "8001"], "/v1/health/ready", 8001),
            "yolo": ([py, "-m", "vastkit.local.yolo_server", "--port", "8002"], "/healthz", 8002),
            "embed": ([py_embed, "-m", "vastkit.local.embed_server", "--port", "8003"], "/v1/health/ready", 8003)}
    names = list(a.only or plan)
    exes = {plan[n][0][0] for n in names}
    missing = sorted(p for p in exes if not (Path(p).is_file() or shutil.which(p)))
    if missing:  # check before starting anything, so nothing is left running
        print(f"[local] interpreter not found: {', '.join(missing)} (set LOCAL_PYTHON / LOCAL_EMBED_PYTHON)",
              file=sys.stderr)
        return 2
    if "embed" in names and not env.get("LOCAL_EMBED_PYTHON") and py_embed == sys.executable:
        print("[local] note: no .venv-embed and LOCAL_EMBED_PYTHON unset; embed runs on this interpreter "
              "(needs transformers 4.x)", flush=True)
    import httpx
    procs: dict[str, subprocess.Popen] = {}
    try:
        for name in names:
            procs[name] = subprocess.Popen(plan[name][0], cwd=ROOT, env=env)
        t0 = time.time()
        pending = set(procs)
        while pending and time.time() - t0 < 300:
            for name in list(pending):
                _, path, port = plan[name]
                try:
                    if httpx.get(f"http://127.0.0.1:{port}{path}", timeout=2).status_code == 200:
                        print(f"[local] {name} ready on :{port} after {time.time() - t0:.0f}s", flush=True)
                        pending.discard(name)
                except httpx.HTTPError:
                    pass
                if procs[name].poll() is not None:
                    print(f"[local] {name} exited with {procs[name].returncode}", flush=True)
                    pending.discard(name)
            time.sleep(2)
        print("set: COSMOS3_REASON_URL=http://127.0.0.1:8001 YOLO_URL=http://127.0.0.1:8002 "
              "COSMOS_EMBED1_URL=http://127.0.0.1:8003   Ctrl+C to stop", flush=True)
        while any(p.poll() is None for p in procs.values()):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        _stop(list(procs.values()))
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="vastkit", description=__doc__.split("\n\n")[0])
    ap.add_argument("--env-file", help="explicit config file (default: VAST_ENV_FILE, /config/*.config, ./.env)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--offline", action="store_true", help="skip network checks")
    sub.add_parser("models")
    m = sub.add_parser("mock")
    m.add_argument("--vss-port", type=int, default=MOCK_PORTS[0])
    m.add_argument("--gpu-port", type=int, default=MOCK_PORTS[1])
    lo = sub.add_parser("local")
    lo.add_argument("--only", nargs="*", choices=["cosmos", "yolo", "embed"])
    a = ap.parse_args(argv)
    return {"doctor": doctor, "models": models, "mock": mock, "local": local}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
