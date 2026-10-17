"""Street Time Machine web server: ui/ at /, out/ at /out, the camera list at /api/cams, the question engine at /api/ask.

Run: serve.cmd   (or: <py> -m uvicorn stm.server:app --host 127.0.0.1 --port 8780)

* /api/cams lists the demo cameras in a fixed order: the event camera (neighborhood_cam-1) first when its
  out/<cam>/meta.json exists, then ptz_parking, then yonge_dundas. Other prepared cameras only with ?all=1.
  Day ids and labels come from each camera's meta.json (never assumed to be "A"/"B").
* /api/ask answers live from the tracks when this machine has them ("live"). A camera without local tracks
  (e.g. the event camera built on the lab VM) is answered from out/<cam>/qa.json by question match
  ("precomputed", latency_ms null because nothing was computed here), else 404 with a message.
  The X-STM-Source response header says which ("live" | "precomputed").
* At startup a background thread asks every qa.json question of every listed camera once, so the first
  live question does not pay the cold load. The server serves immediately; the thread logs p50/max.
  GET /api/health shows the warm-up state. Set STM_WARM=0 to skip it.
* Every response carries Cache-Control: no-store, so a re-run module shows up on reload.
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from stm import ask, data

PREFERRED = ("neighborhood_cam-1", "ptz_parking", "yonge_dundas", "synthetic_demo")
# A source checkout can show the interface without event footage or model inference.
OUT: Path = data.OUT
if "STM_OUT" not in os.environ and not any(OUT.glob("*/meta.json")):
    OUT = data.ROOT / "examples" / "synthetic"
_CAM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LOCKS: dict[str, threading.RLock] = {}   # one question at a time per camera (a cold load blocks only its own cam)
_LOCKS_LOCK = threading.Lock()
_WARM_LOCK = threading.Lock()
_WARM: dict[str, Any] = {"enabled": False, "cams": {}, "total": None}
_LIVE: set[str] = set()                    # cams confirmed to have local tracks + video


def _log(msg: str) -> None:
    print(f"[stm.server] {msg}", flush=True)


def _engine(cam: str) -> threading.RLock:
    with _LOCKS_LOCK:
        return _LOCKS.setdefault(cam, threading.RLock())


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


def _read(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean(o: Any) -> Any:
    """Plain JSON: numpy scalars to Python, NaN/inf to null (the browser's JSON.parse rejects NaN)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
    if isinstance(o, int):
        return int(o)
    if isinstance(o, float):
        return float(o) if math.isfinite(o) else None
    if hasattr(o, "item"):                 # numpy scalar
        return _clean(o.item())
    if hasattr(o, "tolist"):               # numpy array
        return _clean(o.tolist())
    return str(o)


# ----------------------------------------------------------------------------- cameras

def known(cam: str) -> bool:
    return bool(_CAM_RE.match(cam)) and (cam in data.CAMS or (OUT / cam / "meta.json").exists())


SCAN_MAX_S = 600.0     # without a fresh ask_index.json the engine scans the whole video first; only short clips


def _index_ok(cam: str) -> bool:
    """ask_index.json is current, or the clip is short enough to scan in the background warm-up."""
    idx = _read(data.OUT / cam / "ask_index.json")
    if isinstance(idx, dict) and idx.get("version") == getattr(ask, "INDEX_VERSION", None):
        return True
    dur = (_read(data.OUT / cam / "meta.json") or {}).get("duration") or 0.0
    return float(dur) <= SCAN_MAX_S


def ask_mode(cam: str) -> str:
    """'live' (tracks + video on this machine), 'precomputed' (qa.json only) or 'none'."""
    if cam in _LIVE:
        return "live"
    if cam in data.CAMS and any(data.TRACK_CACHE.glob(f"{cam}__det__*.json.gz")) \
            and Path(data.CAMS[cam]["video"]).exists() and _index_ok(cam):
        _LIVE.add(cam)
        return "live"
    return "precomputed" if (OUT / cam / "qa.json").exists() else "none"


def listed_cams(all_: bool = False) -> list[str]:
    def has(c: str) -> bool:
        return bool(_CAM_RE.match(c)) and (OUT / c / "meta.json").exists()
    cams = [c for c in PREFERRED if has(c)]
    if all_ or not cams:
        cams += [c for c in data.CAMS if c not in cams and has(c)]
        if OUT.is_dir():
            cams += sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name not in cams and has(p.name))
    return cams


def cam_entry(cam: str, i: int) -> dict[str, Any]:
    m = _read(OUT / cam / "meta.json") or {}
    days = [{"id": str(d.get("id")), "label": d.get("label") or str(d.get("id")),
             "start": d.get("start"), "end": d.get("end")} for d in (m.get("days") or []) if isinstance(d, dict)]
    return _clean({"camera": cam, "title": m.get("title"), "credit": m.get("credit"), "duration": m.get("duration"),
                   "w": m.get("w"), "h": m.get("h"), "days": days, "simulated": m.get("simulated") or [],
                   "has_qa": (OUT / cam / "qa.json").exists(), "ask": ask_mode(cam), "default": i == 0})


# ----------------------------------------------------------------------------- precomputed answers

def _norm(s: str | None) -> str:
    """Same normalisation as the UI's offline fallback: lower case, letters/digits/spaces only."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (s or "").lower())).strip()


def qa_items(cam: str) -> list[dict[str, Any]]:
    q = _read(OUT / cam / "qa.json")
    if isinstance(q, dict):
        q = q.get("items") or q.get("questions") or []
    return [it for it in (q or []) if isinstance(it, dict) and (it.get("question") or it.get("q"))]


def qa_match(cam: str, q: str, t: float | None) -> dict[str, Any] | None:
    want = _norm(q)
    for it in qa_items(cam):
        if _norm(it.get("question") or it.get("q")) != want:
            continue
        it_t = it.get("t")
        if t is not None and it_t is not None and abs(float(it_t) - float(t)) > 0.5:
            continue          # same words asked at another moment: not the stored answer
        a = it.get("answer")
        return a if isinstance(a, dict) else None
    return None


# ----------------------------------------------------------------------------- warm-up

def warm_cam(cam: str) -> dict[str, Any]:
    """Ask every qa.json question once (no t, as qa.json was built). Returns the stats it logged."""
    mode = ask_mode(cam)
    items = qa_items(cam)
    st: dict[str, Any] = {"state": "running", "mode": mode, "questions": len(items)}
    with _WARM_LOCK:
        _WARM["cams"][cam] = st
    if mode != "live":
        st["state"] = "done"
        _log(f"warm {cam}: {mode} ({len(items)} stored answers), nothing to load")
        return st
    qs = [it.get("question") or it.get("q") for it in items] or ["How many cars came by?"]
    t0 = time.perf_counter()
    try:
        with _engine(cam):
            ask.answer(cam, qs[0])
    except Exception as e:
        st.update(state="failed", error=f"{type(e).__name__}: {e}")
        _log(f"warm {cam}: FAILED to load: {type(e).__name__}: {e}")
        return st
    st["load_ms"] = _ms(t0)
    lat: list[float] = []
    drift: list[dict[str, Any]] = []
    for it in items:
        q = it.get("question") or it.get("q")
        t1 = time.perf_counter()
        try:
            with _engine(cam):
                a = ask.answer(cam, q)
        except Exception as e:
            drift.append({"question": q, "error": f"{type(e).__name__}: {e}"})
            continue
        lat.append(_ms(t1))
        want = (it.get("answer") or {}).get("answer")
        if want is not None and str(a.get("answer")) != str(want):
            drift.append({"question": q, "qa_json": want, "live": a.get("answer")})
    st.update(state="done", n=len(lat), latencies_ms=lat, drift=drift,
              p50_ms=round(statistics.median(lat), 1) if lat else None, max_ms=round(max(lat), 1) if lat else None)
    _log(f"warm {cam}: loaded in {st['load_ms'] / 1000:.1f} s; {len(lat)} answers, "
         f"p50 {st['p50_ms']} ms, max {st['max_ms']} ms")
    errs = [d for d in drift if "error" in d]
    diff = [d for d in drift if "error" not in d]
    if errs:
        _log(f"warm {cam}: WARNING {len(errs)} question(s) FAILED in the engine: "
             + "; ".join(f"{d['question']!r} ({d['error']})" for d in errs[:4]))
    if diff:
        _log(f"warm {cam}: WARNING {len(diff)} answer(s) differ from qa.json (re-run: python -m stm.ask {cam}): "
             + "; ".join(f"{d['question']!r} qa.json {d['qa_json']} vs live {d['live']}" for d in diff[:4]))
    return st


def _warm_run(cams: list[str]) -> None:
    for cam in cams:
        try:
            warm_cam(cam)
        except Exception:
            _log(f"warm {cam}: crashed\n{traceback.format_exc()}")
    with _WARM_LOCK:
        lat = [x for s in _WARM["cams"].values() for x in s.get("latencies_ms", [])]
    if lat:
        p50, mx = statistics.median(lat), max(lat)
        _WARM["total"] = {"n": len(lat), "p50_ms": round(p50, 1), "max_ms": round(mx, 1)}
        _log(f"warm: {len(lat)} answers, p50 {p50:.0f} ms, max {mx:.0f} ms")
        if p50 >= 100:
            _log(f"WARNING WARNING: p50 {p50:.0f} ms >= 100 ms. Live answers will not feel instant.")


def start_warm(cams: list[str]) -> threading.Thread | None:
    """Warm cams not seen before, in a daemon thread. Safe to call repeatedly."""
    with _WARM_LOCK:
        todo = [c for c in cams if c not in _WARM["cams"]]
        for c in todo:
            _WARM["cams"][c] = {"state": "queued"}
    if not todo:
        return None
    th = threading.Thread(target=_warm_run, args=(todo,), name="stm-warm", daemon=True)
    th.start()
    return th


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("STM_WARM", "1") != "0":
        _WARM["enabled"] = True
        ids = listed_cams()
        _log(f"serving now; warming {ids} in the background")
        start_warm(ids)
    yield


app = FastAPI(title="Street Time Machine", lifespan=lifespan)


@app.middleware("http")
async def no_store(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/cams")
def cams(all: int = Query(0)) -> list[dict[str, Any]]:
    """Prepared cameras (out/<cam>/meta.json exists), event camera first. ?all=1 adds the other dev cameras."""
    ids = listed_cams(bool(all))
    if _WARM["enabled"]:
        start_warm(ids)          # a camera synced after startup gets warmed too
    return [cam_entry(c, i) for i, c in enumerate(ids)]


@app.get("/api/health")
def health() -> dict[str, Any]:
    with _WARM_LOCK:
        w = {c: {k: v for k, v in s.items() if k != "latencies_ms"} for c, s in _WARM["cams"].items()}
    return _clean({"ok": True, "warm_enabled": _WARM["enabled"], "warm": w, "warm_total": _WARM["total"],
                   "cams": {c: ask_mode(c) for c in listed_cams(True)}})


@app.get("/api/ask")
def api_ask(cam: str = Query(..., min_length=1, max_length=64), q: str = Query(..., min_length=1, max_length=300),
            t: float | None = Query(None)):
    t0 = time.perf_counter()
    if not known(cam):
        raise HTTPException(404, f"unknown camera {cam!r}")
    mode = ask_mode(cam)
    failed = None
    if mode == "live":
        try:
            with _engine(cam):
                res = ask.answer(cam, q, t)
            out = _clean({k: res.get(k) for k in ("type", "answer", "sentence", "latency_ms", "evidence", "honesty")})
            engine = out["latency_ms"]
            out["latency_ms"] = _ms(t0)     # what the user waited, measured here (includes any queueing)
            return JSONResponse(out, headers={"X-STM-Source": "live", "X-STM-Engine-Ms": str(engine)})
        except Exception as e:              # fall back to a stored answer if there is one
            failed = f"{type(e).__name__}: {e}"
            _log(f"ask {cam} {q!r}: engine failed: {failed}")
    a = qa_match(cam, q, t)
    if a is not None:
        out = _clean({k: a.get(k) for k in ("type", "answer", "sentence", "evidence", "honesty")})
        out["latency_ms"] = None            # stored answer: nothing was computed here, so no speed claim
        return JSONResponse(out, headers={"X-STM-Source": "precomputed", "X-STM-Lookup-Ms": str(_ms(t0))})
    if failed:
        raise HTTPException(500, "The question engine could not answer that one. Try one of the sample questions.")
    raise HTTPException(404, "This machine has no local tracks for this camera, so it can only answer the "
                             "sample questions. Pick one of them.")


OUT.mkdir(parents=True, exist_ok=True)
app.mount("/out", StaticFiles(directory=str(OUT)), name="out")
app.mount("/", StaticFiles(directory=str(data.ROOT / "ui"), html=True), name="ui")
