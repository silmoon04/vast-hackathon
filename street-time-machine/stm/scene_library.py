"""Local footage catalogue and bounded, lazy Find & Follow indexing."""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
FOOTAGE_ROOT = Path(os.environ.get("STM_FOOTAGE_ROOT", str(ROOT.parent / "vast-hackathon")))
OUT = ROOT / "out" / "find"
_DEFAULT_CATALOGUES: tuple[str, ...] = ()


def _catalogue_paths() -> tuple[Path, ...]:
    """Read an optional OS path-list of catalogue JSON files.

    Keeping this empty by default makes a clean checkout safe and portable;
    operators can opt in to local footage catalogues with STM_SCENE_CATALOGS.
    Relative entries are resolved from the application root.
    """
    raw = os.environ.get("STM_SCENE_CATALOGS", "")
    entries = [part.strip() for part in raw.split(os.pathsep) if part.strip()]
    if not entries:
        entries = list(_DEFAULT_CATALOGUES)
    return tuple((Path(part) if Path(part).is_absolute() else ROOT / part).resolve()
                 for part in entries)
router = APIRouter(prefix="/api/find/library", tags=["footage library"])
_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="scene-library")


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-")[:54]


def sources() -> dict[str, dict[str, Any]]:
    """Only exact catalogue members can be served; arbitrary paths never enter the API."""
    result = {}
    for filename in _catalogue_paths():
        for raw in _read(filename) or []:
            item = dict(raw)
            path = Path(item["file"])
            if not path.is_absolute():
                path = FOOTAGE_ROOT / path
            item["path"] = path.resolve()
            result[str(item["id"])] = item
    return result


def groups() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in sources().values():
        group = str(item["group"])
        # Capture 1, 2 and 3 are different events, despite sharing one gallery group.
        if group == "citic":
            match = re.match(r"citic-(\d+)-", item["id"])
            if match and match[1] != "1":
                group += "-" + match[1]
        name = "library-" + _slug(group)
        entry = result.setdefault(name, {"name": name, "group": group, "items": [],
                                        "title": item.get("group_name", group)})
        entry["items"].append(item)
    for entry in result.values():
        if entry["group"].startswith("citic"):
            entry["title"] = "CITIC indoor route · capture " + (entry["group"].split("-")[-1] if "-" in entry["group"] else "1")
        entry["items"].sort(key=lambda item: str(item.get("camera", item["id"])))
    priority = {"citic": 0, "citic-2": 1, "citic-3": 2, "wisenet": 3,
                "hospital": 4, "tocada-1": 5, "tocada-2": 6}
    return dict(sorted(result.items(), key=lambda pair: (priority.get(pair[1]["group"], 20), pair[0])))


def _group(name: str) -> dict[str, Any]:
    entry = groups().get(name)
    if entry is None:
        raise HTTPException(404, "Unknown footage scene")
    return entry


def _status(name: str) -> dict[str, Any]:
    with _lock:
        job = dict(_jobs.get(name, {}))
    if job.get("status") in {"queued", "indexing"}:
        return job
    scene = _read(OUT / name / "scene.json")
    if scene and not (scene.get("build") or {}).get("problems"):
        return {"status": "ready", "tracks": len(scene.get("tracks", [])),
                "indexed_at": (OUT / name / "scene.json").stat().st_mtime}
    return job or {"status": "unindexed"}


def _clock_note(entry: dict[str, Any]) -> str:
    if entry["group"].startswith("citic"):
        return "Synchronized cameras within this capture; captures are separate events."
    return "Clip-relative clock within this footage group. Cross-camera timing and identities require verification."


def metadata(entry: dict[str, Any]) -> dict[str, Any]:
    items = entry["items"]
    return {"name": entry["name"], "title": entry["title"], "group": entry["group"],
            "kind": "synthetic" if all(item.get("synthetic") for item in items) else "real",
            "cameras": len(items), "duration": max(float(item.get("duration") or 0) for item in items),
            "base": f"/out/find/{entry['name']}/", "clock_note": _clock_note(entry),
            "available_cameras": sum(item["path"].is_file() for item in items),
            "analysis": "Motion tracks and clothing colours computed at up to native 720p; no ground-truth identities imported.",
            **_status(entry["name"])}


@router.get("")
def catalogue() -> dict[str, Any]:
    return {"scenes": [metadata(entry) for entry in groups().values()], "index_concurrency": 2}


@router.get("/video/{source_id}")
def video(source_id: str) -> FileResponse:
    item = sources().get(source_id)
    if item is None or not item["path"].is_file():
        raise HTTPException(404, "Footage source unavailable")
    return FileResponse(item["path"], media_type="video/mp4", filename=item["path"].name,
                        content_disposition_type="inline")


@router.get("/{name}/scene")
def scene(name: str) -> dict[str, Any]:
    entry = _group(name)
    meta = metadata(entry)
    loaded = _read(OUT / name / "scene.json") if meta["status"] == "ready" else None
    cams = []
    for k, item in enumerate(entry["items"]):
        cam_id = chr(65 + k)
        cams.append({"id": cam_id, "name": item.get("label", cam_id),
                     "pos": [round(40 + 420 * k / max(1, len(entry['items']) - 1)), 75],
                     "links": [], "blindLinks": [], "props": [], "offset": 0,
                     "exits": None, "exitZones": {}, "video": f"/api/find/library/video/{item['id']}",
                     "width": item.get("width"), "height": item.get("height"),
                     "source_id": item["id"], "credit": item.get("credit", ""),
                     "source_url": item.get("source_url"), "license": item.get("license"),
                     "license_url": item.get("license_url"), "route_note": item.get("route_note", "")})
    sc = copy.deepcopy(loaded) if loaded else {
        "version": 1, "simulated": all(item.get("synthetic") for item in entry["items"]),
        "title": entry["title"], "clock0": "2026-10-17T00:00:00", "duration": meta["duration"],
        "cameras": cams, "tracks": [], "identities": [], "items": [], "build": {"seconds": None}}
    source_cams = {cam["id"]: cam for cam in cams}
    for cam in sc["cameras"]:
        if str(cam["id"]) in source_cams:
            cam.update(source_cams[str(cam["id"])])
    for track in sc.get("tracks", []):
        if track.get("crop") and not track["crop"].startswith("/"):
            track["crop"] = meta["base"] + track["crop"]
    sc.update(name=name, base=meta["base"], status=meta["status"], index_status=meta["status"],
              simulated=all(item.get("synthetic") for item in entry["items"]),
              clock_note=meta["clock_note"], credit=" | ".join(dict.fromkeys(item.get("credit", "") for item in entry["items"])))
    sc["notices"] = list(dict.fromkeys(filter(None, [item.get("route_note", "") for item in entry["items"]]
                                              + ["Source license: " + item["license"] for item in entry["items"] if item.get("license")])))
    sc["analysis_note"] = "Analysis preserves native resolution up to 720 pixels high; lower resolution sources are never upscaled."
    if meta.get("error"):
        sc["index_error"] = meta["error"]
    return sc


def _index(name: str) -> None:
    folder = OUT / name
    try:
        entry = _group(name)
        missing = [item["id"] for item in entry["items"] if not item["path"].is_file()]
        if missing:
            raise FileNotFoundError("Missing source videos: " + ", ".join(missing))
        folder.mkdir(parents=True, exist_ok=True)
        topo = {"title": entry["title"], "clock0": "2026-10-17T00:00:00",
                "credit": " | ".join(dict.fromkeys(item.get("credit", "") for item in entry["items"])),
                "cameras": [{"id": chr(65 + k), "name": item.get("label", item["id"]),
                             "video": str(item["path"]), "offset_s": 0, "links": [], "blind": []}
                            for k, item in enumerate(entry["items"])]}
        topology = folder / "topology.json"
        topology.write_text(json.dumps(topo, indent=2), encoding="utf-8")
        with _lock:
            _jobs[name] = {"status": "indexing", "started_at": time.time()}
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (folder / "index.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "index_library_scene.py"),
                                   "--cams", str(topology), "--scene", name, "--out", str(OUT)],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        if proc.returncode:
            error = (folder / "index.log").read_text(encoding="utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"Indexer exited {proc.returncode}: {error}")
        with _lock:
            _jobs[name] = {"status": "ready", "completed_at": time.time()}
    except Exception as exc:
        with _lock:
            _jobs[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@router.post("/{name}/index")
def start_index(name: str) -> dict[str, Any]:
    entry = _group(name)
    with _lock:
        if _status(name)["status"] not in {"queued", "indexing", "ready"}:
            _jobs[name] = {"status": "queued", "queued_at": time.time()}
            _pool.submit(_index, name)
    return metadata(entry)
