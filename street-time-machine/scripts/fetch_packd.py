"""Pull Pack D (neighborhood_cam-1) chunk videos, segment rows and YOLO sidecars from the team VSS backend.

Uses an explicitly selected team config file or VSS_INGRESS_URL, VSS_USERNAME and VSS_PASSWORD.
Never prints the credentials or the token. Read-only: GET/POST login only, no writes, no re-ingest.
Output: data/packd/{chunks/<chunk>.mp4, sidecars/<segment>.json, segments.json}, or --output.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "packd"


def load_cfg(config_path=None):
    cfg = {}
    lines = Path(config_path).expanduser().read_text(encoding="utf-8").splitlines() if config_path else []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.replace("export ", "").strip()
        cfg[k] = v.strip().strip('"').strip("'")
    for key in ("INGRESS_URL", "USERNAME", "PASSWORD"):
        if os.environ.get("VSS_" + key):
            cfg[key] = os.environ["VSS_" + key]
    missing = [key for key in ("INGRESS_URL", "USERNAME", "PASSWORD") if not cfg.get(key)]
    if missing:
        raise SystemExit("Set --config or VSS_ environment variables for: " + ", ".join(missing))
    return cfg


CFG = {}
B = ""
_tok = None


def login():
    global _tok
    body = json.dumps({"username": CFG["USERNAME"], "password": CFG["PASSWORD"]}).encode()
    req = urllib.request.Request(B + "/api/v1/auth/login", body, {"Content-Type": "application/json"})
    _tok = json.load(urllib.request.urlopen(req, timeout=60))["access_token"]


def get(path, **params):
    for attempt in range(4):
        req = urllib.request.Request(B + path + "?" + urllib.parse.urlencode(params),
                                     headers={"Authorization": "Bearer " + _tok})
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code == 401:
                login()
                continue
            if e.code == 404:
                return None
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError("GET failed " + path)


def stream(src, dest):
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest.stat().st_size
    for attempt in range(4):
        url = B + "/api/v1/videos/stream?" + urllib.parse.urlencode({"source": src, "token": _tok})
        try:
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            tmp.rename(dest)
            return dest.stat().st_size
        except urllib.error.HTTPError as e:
            if e.code == 401:
                login()
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError("stream failed " + src)


def first_list(d):
    if isinstance(d, list):
        return d
    for k in ("segments", "results", "items", "rows", "chunks", "data"):
        if isinstance(d.get(k), list):
            return d[k]
    for v in d.values():
        if isinstance(v, list):
            return v
    return []


def main():
    global CFG, B, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("STM_VSS_CONFIG"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CFG = load_cfg(args.config)
    B = CFG["INGRESS_URL"].rstrip("/")
    if not B.startswith("https://"):
        raise SystemExit("The VSS ingress must use HTTPS.")
    OUT = args.output.expanduser().resolve()
    (OUT / "chunks").mkdir(parents=True, exist_ok=True)
    (OUT / "sidecars").mkdir(parents=True, exist_ok=True)
    login()
    print("login ok")
    chunks, off = [], 0
    while True:
        d = get("/api/v1/videos/explore", scope="all", indexed="complete", location="neighborhood", limit=100,
                offset=off)
        got = first_list(d or {})
        chunks += got
        off += len(got)
        if not got or off >= int((d or {}).get("total", 0)):
            break
    srcs = sorted({c.get("original_video") or c.get("source") for c in chunks
                   if "_neighborhood_2026" in str(c.get("original_video") or c.get("source"))})
    print("chunks", len(srcs))

    def one_chunk(src):
        name = src.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        segs = first_list(get("/api/v1/tools/segments", original_video=src) or {})
        rows = []
        for s in segs:
            sid = s.get("source") or s.get("segment_source")
            side = get("/api/v1/videos/detections", source=sid)
            seg_name = sid.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if side is not None:
                side["segment_start_sec"] = s.get("segment_start_sec")
                side["segment_end_sec"] = s.get("segment_end_sec")
                side["chunk"] = name
                (OUT / "sidecars" / (seg_name + ".json")).write_text(json.dumps(side))
            keep = {k: v for k, v in s.items() if k not in ("embedding", "vector", "embeddings")}
            keep["chunk"], keep["has_sidecar"] = name, side is not None
            rows.append(keep)
        size = stream(src, OUT / "chunks" / (name + ".mp4"))
        return name, rows, size

    allrows = []
    with cf.ThreadPoolExecutor(max(1, min(args.workers, 8))) as ex:
        for name, rows, size in ex.map(one_chunk, srcs):
            allrows += rows
            print(name, len(rows), "segments", sum(r["has_sidecar"] for r in rows), "sidecars",
                  round(size / 1e6, 1), "MB", flush=True)
    (OUT / "segments.json").write_text(json.dumps({"chunks": srcs, "segments": allrows}, default=str))
    print("done", len(srcs), "chunks", len(allrows), "segments")


if __name__ == "__main__":
    sys.exit(main())
