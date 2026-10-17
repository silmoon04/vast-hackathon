"""Video Atlas Field Desk: real catalogue, video, Pinch Point analysis and questions.

Deploy as a flat ConfigMap at /app. Uses VSS credentials from a Kubernetes Secret.
No additional dependencies, model weights, uploads or ingestion changes.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import analysis

ROOT = Path(__file__).resolve().parent
VSS_URL = os.environ.get("VSS_URL", "").rstrip("/")
VSS_USERNAME = os.environ.get("VSS_USERNAME", "")
VSS_PASSWORD = os.environ.get("VSS_PASSWORD", "")
TOKEN_LOCK = threading.Lock()
CATALOG_LOCK = threading.Lock()
CACHE_LOCK = threading.Lock()
TOKEN = {"value": "", "at": 0}
CATALOG = {"at": 0, "clips": {}}
SEGMENTS = OrderedDict()
ANALYSES = OrderedDict()
ANALYSIS_SLOTS = threading.BoundedSemaphore(3)
STARTED = time.time()


def open_request(request, timeout=35):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def token(force=False):
    with TOKEN_LOCK:
        if force or not TOKEN["value"] or time.time() - TOKEN["at"] > 20 * 3600:
            if not VSS_URL or not VSS_USERNAME or not VSS_PASSWORD:
                raise RuntimeError("VSS credentials are not configured.")
            body = json.dumps({"username": VSS_USERNAME, "password": VSS_PASSWORD}).encode()
            with open_request(urllib.request.Request(VSS_URL + "/api/v1/auth/login", data=body, headers={"Content-Type": "application/json"})) as response:
                TOKEN.update(value=json.load(response)["access_token"], at=time.time())
        return TOKEN["value"]


def vss(path, params=None):
    query = "?" + urllib.parse.urlencode(params) if params else ""
    for attempt in range(2):
        request = urllib.request.Request(VSS_URL + "/api/v1/" + path + query, headers={"Authorization": "Bearer " + token(force=attempt == 1)})
        try:
            with open_request(request) as response:
                raw = response.read((12 << 20) + 1)
                if len(raw) > 12 << 20:
                    raise ValueError("The detection response is too large for this demo.")
                return json.loads(raw)
        except urllib.error.HTTPError as error:
            if error.code == 401 and attempt == 0:
                continue
            raise


def catalogue():
    with CATALOG_LOCK:
        if time.time() - CATALOG["at"] < 300 and CATALOG["clips"]:
            return CATALOG["clips"]
        clips = {}
        for offset in range(0, 10000, 100):
            page = vss("videos/explore", {"indexed": "complete", "limit": 100, "offset": offset})
            if page.get("table_available") is False:
                raise RuntimeError("The video index is unavailable.")
            for row in page.get("chunks", []):
                source = row.get("original_video")
                if not source or not source.startswith("s3://"):
                    continue
                cid = hashlib.sha256(source.encode()).hexdigest()[:20]
                clips[cid] = {"id": cid, "source": source, "filename": row.get("filename", "Video clip"),
                              "camera": row.get("camera_id") or "Unlabelled camera", "location": row.get("location") or "",
                              "caption": str(row.get("reasoning_content") or ""), "duration": row.get("chunk_duration_sec"),
                              "segment_count": row.get("total_segments", 0)}
            if offset + 100 >= page.get("total", 0) or not page.get("chunks"):
                break
        CATALOG.update(at=time.time(), clips=clips)
        return clips


def public_clip(clip):
    return {key: value for key, value in clip.items() if key != "source"}


def clip_by_id(cid):
    if not re.fullmatch(r"[0-9a-f]{20}", str(cid)):
        raise ValueError("Choose a clip from the catalogue.")
    clip = catalogue().get(cid)
    if not clip:
        raise ValueError("This clip is no longer in the catalogue. Refresh the list.")
    return clip


def filtered_catalog(mode, query="", camera=""):
    clips = catalogue()
    terms = {"people": ["people", "pedestrian", "crosswalk", "crossing", "crowd", "sidewalk", "queue"],
             "vehicles": ["vehicle", "car", "traffic", "intersection", "truck", "parking", "road"]}.get(mode, [])
    words = query.lower().split()
    ranked = []
    for clip in clips.values():
        text = " ".join(str(clip[k]) for k in ("filename", "camera", "location", "caption")).lower()
        if camera and clip["camera"] != camera:
            continue
        if any(word not in text for word in words):
            continue
        score = sum(text.count(term) for term in terms)
        if terms and not score:
            continue
        ranked.append((score, clip))
    ranked.sort(key=lambda item: (-item[0], item[1]["filename"]))
    return {"total": len(clips), "matches": len(ranked), "clips": [public_clip(c) for _, c in ranked[:60]],
            "cameras": sorted({c["camera"] for c in clips.values()}), "filter": mode,
            "filter_method": "Catalogue caption and camera keywords; detections are checked when a clip is opened."}


def segment_rows(clip):
    with CACHE_LOCK:
        cached = SEGMENTS.get(clip["id"])
        if cached:
            SEGMENTS.move_to_end(clip["id"])
            return cached
    rows = vss("tools/segments", {"original_video": clip["source"]}).get("segments", [])
    rows.sort(key=lambda row: float(row.get("segment_start_sec") or 0))
    if not rows:
        raise ValueError("No indexed segments are available for this clip.")
    with CACHE_LOCK:
        SEGMENTS[clip["id"]] = rows
        while len(SEGMENTS) > 64:
            SEGMENTS.popitem(last=False)
    return rows


def analyse_clip(cid, region, confidence):
    clip = clip_by_id(cid)
    region = analysis.validate_region(region)
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between zero and one.")
    aid = hashlib.sha256(json.dumps([cid, region, confidence]).encode()).hexdigest()[:24]
    with CACHE_LOCK:
        if aid in ANALYSES:
            ANALYSES.move_to_end(aid)
            return ANALYSES[aid]
    if not ANALYSIS_SLOTS.acquire(blocking=False):
        raise RuntimeError("Three clips are already being analysed. Try again shortly.")
    try:
        started = time.perf_counter()
        rows = segment_rows(clip)
        if len(rows) > 120:
            raise ValueError("Choose a shorter clip for this demo.")
        duration = max(float(row.get("segment_end_sec") or 0) for row in rows)
        if duration <= 0:
            duration = float(clip.get("duration") or 0)
        def fetch(row):
            if not row.get("detection_sidecar_uri"):
                return None
            try:
                side = vss("videos/detections", {"source": row["source"]})
                start = float(row.get("segment_start_sec") or 0)
                end = float(row.get("segment_end_sec") or start + float(row.get("duration") or 5))
                return start, end, side
            except (urllib.error.URLError, OSError, ValueError):
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            sidecars = [side for side in pool.map(fetch, rows) if side is not None]
        fetched = time.perf_counter()
        result = analysis.build(sidecars, duration, region, confidence)
        result.update(analysis_id=aid, clip=public_clip(clip), segments_total=len(rows), segments_loaded=len(sidecars),
                      fetch_ms=round((fetched - started) * 1000), compute_ms=round((time.perf_counter() - fetched) * 1000),
                      total_ms=round((time.perf_counter() - started) * 1000), source="VSS YOLO11 detection sidecars", computed_at=time.time())
        with CACHE_LOCK:
            ANALYSES[aid] = result
            while len(ANALYSES) > 16:
                ANALYSES.popitem(last=False)
        return result
    finally:
        ANALYSIS_SLOTS.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "VideoAtlas/1.0"

    def log_message(self, fmt, *args):
        print(json.dumps({"method": self.command, "path": urllib.parse.urlsplit(self.path).path, "status": str(args[1]) if len(args) > 1 else ""}), flush=True)

    def send_data(self, data, status=200, content_type="application/json; charset=utf-8"):
        body = json.dumps(data, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        get = lambda key, default="": params.get(key, [default])[0]
        try:
            if parsed.path in {"/health", "/api/health"}:
                return self.send_data({"ok": True, "app": "Video Atlas Field Desk", "configured": bool(VSS_URL and VSS_USERNAME and VSS_PASSWORD), "uptime_s": round(time.time() - STARTED)})
            if parsed.path == "/api/catalog":
                return self.send_data(filtered_catalog(get("mode", "people"), get("q")[:200], get("camera")[:200]))
            if parsed.path == "/video":
                return self.stream(clip_by_id(get("clip")))
            files = {
                "/": "index.html", "/index.html": "index.html",
                "/app.js": "app.js", "/style.css": "style.css",
                "/project-guide.js": "project-guide.js",
                "/project-guide.css": "project-guide.css",
                "/project-guide.json": "project-guide.json",
            }
            if parsed.path in files:
                path = ROOT / files[parsed.path]
                return self.send_data(path.read_bytes(), content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return self.send_data({"detail": "Route not found."}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as error:
            self.send_data({"detail": str(error)}, 400)
        except Exception as error:
            print(json.dumps({"error": type(error).__name__, "route": parsed.path}), flush=True)
            self.send_data({"detail": "The catalogue or video service did not respond. Try again shortly."}, 502)

    def do_POST(self):
        route = urllib.parse.urlsplit(self.path).path
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                self.close_connection = True
                return self.send_data({"detail": "Request must be between 1 and 65536 bytes."}, 413)
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Send a JSON object.")
            if route == "/api/analyse":
                return self.send_data(analyse_clip(body.get("clip", ""), body.get("region", []), body.get("confidence", .25)))
            if route == "/api/ask":
                question = body.get("question", "")
                if not isinstance(question, str) or not 1 <= len(question.strip()) <= 400:
                    raise ValueError("Enter a question of up to 400 characters.")
                with CACHE_LOCK:
                    item = ANALYSES.get(str(body.get("analysis_id", "")))
                if item is None:
                    return self.send_data({"detail": "This analysis expired. Open the clip again."}, 409)
                started = time.perf_counter()
                result = analysis.answer(item, question.strip(), body.get("mode", "people"))
                result["compute_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return self.send_data(result)
            return self.send_data({"detail": "Route not found."}, 404)
        except (ValueError, TypeError, KeyError) as error:
            self.send_data({"detail": str(error)}, 400)
        except RuntimeError as error:
            self.send_data({"detail": str(error)}, 503)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            print(json.dumps({"error": type(error).__name__, "route": route}), flush=True)
            self.send_data({"detail": "The analysis service could not complete this request. Try another clip."}, 502)

    def stream(self, clip):
        for attempt in range(2):
            url = VSS_URL + "/api/v1/videos/stream?" + urllib.parse.urlencode({"source": clip["source"], "token": token(force=attempt == 1)})
            headers = {}
            requested_range = self.headers.get("Range", "")
            if re.fullmatch(r"bytes=\d*-\d*", requested_range):
                headers["Range"] = requested_range
            try:
                response = open_request(urllib.request.Request(url, headers=headers), timeout=60)
            except urllib.error.HTTPError as error:
                if error.code == 401 and attempt == 0:
                    continue
                if error.code == 416:
                    return self.send_data({"detail": "Video range is unavailable."}, 416)
                raise
            with response:
                self.send_response(response.status)
                for name in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "ETag"):
                    if response.headers.get(name):
                        self.send_header(name, response.headers[name])
                self.send_header("Cache-Control", "private, max-age=600")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                if self.command != "HEAD":
                    while data := response.read(65536):
                        self.wfile.write(data)
            return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler)
    server.daemon_threads = True
    print("Video Atlas Field Desk listening", flush=True)
    server.serve_forever()
