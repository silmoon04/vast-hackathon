"""protokit: shared helpers for the 15 video prototypes in prototypes/.

Footage: Intel IoT sample clips (CC BY 4.0), Wikimedia Commons clips (PD / CC0 / CC BY / CC BY-SA) and Pexels
clips (Pexels licence). Local tests only: never commit, never ingest into VSS, never write to team S3 buckets,
never send to hosted endpoints. Every model here runs on this laptop.

Stand-ins for the event stack (same shapes, local models):
  DataEngine YOLO11s counts per 5 s segment -> segments()   (YOLO26s TensorRT + ByteTrack, cached)
  VastDB SQL over the segment table         -> vastdb()     (duckdb in-process; same SQL dialect subset)
  Cosmos3-Reason                            -> vlm()        (local Cosmos3-Edge on :8001, cached; weaker + slower)
  Cosmos-Embed1                             -> embed_*()    (local Cosmos-Embed1-224p on :8003, cached)

Usage from a prototype folder:
    import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import protokit as pk
    tr = pk.tracks("retail_cctv")          # DataFrame-backed Tracks, cached under prototypes/_cache
"""
from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import html
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "_cache"
MODELS = Path(os.environ.get("PK_MODELS", "S:/vast-hack/models"))
INTEL = Path(os.environ.get("PK_INTEL", "S:/vast-hack/videos/intel"))
COSMOS_URL = os.environ.get("PK_COSMOS_URL", "http://127.0.0.1:8001")
EMBED_URL = os.environ.get("PK_EMBED_URL", "http://127.0.0.1:8003")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------------------------------- footage


def videos() -> dict[str, dict[str, Any]]:
    """name -> {path, source, title, license, author, origin}. Intel clips + Commons + Pexels."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(INTEL.glob("*.mp4")):
        out[p.stem.replace("-", "_")] = {"path": str(p).replace("\\", "/"), "title": p.name, "license": "CC BY 4.0",
                                         "author": "Intel Corporation (intel-iot-devkit/sample-videos)",
                                         "source": "https://github.com/intel-iot-devkit/sample-videos", "origin": "intel"}
    for f in ("videos.json", "videos_extra.json"):
        if (ROOT / f).exists():
            out.update(json.loads((ROOT / f).read_text(encoding="utf-8")))
    return out


def video_path(src: str | Path) -> Path:
    """Accept a catalogue name (e.g. 'retail_cctv', 'people_detection') or a path."""
    p = Path(src)
    if p.suffix and p.exists():
        return p
    v = videos()
    if str(src) not in v:
        raise KeyError(f"unknown video {src!r}; known: {', '.join(sorted(v))}")
    return Path(v[str(src)]["path"])


def name_of(src: str | Path) -> str:
    p = Path(src)
    if not p.suffix:
        return str(src)
    for k, v in videos().items():
        if Path(v["path"]).resolve() == p.resolve():
            return k
    return p.stem


def credit(src: str | Path) -> str:
    """One-line attribution for reports."""
    v = videos().get(name_of(src))
    if not v:
        return f"{Path(str(src)).name} (local file)"
    return f"{v['title']} by {v['author']}, {v['license']}, via {v['origin']} ({v['source']})"


def info(src: str | Path) -> dict[str, Any]:
    """{fps, w, h, frames, duration} via OpenCV (fast, no ffprobe)."""
    p = video_path(src)
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"path": str(p), "fps": float(fps), "w": w, "h": h, "frames": n, "duration": n / fps if fps else 0.0}


def iter_frames(src: str | Path, *, stride: int = 1, start: float = 0.0, end: float | None = None
                ) -> Iterator[tuple[int, float, np.ndarray]]:
    """(frame_index, seconds, BGR frame) for every `stride`-th frame in [start, end)."""
    p = video_path(src)
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
        i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            t = i / fps
            if end is not None and t >= end:
                break
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield i, t, frame
            i += 1
    finally:
        cap.release()


def frame_at(src: str | Path, t: float) -> np.ndarray:
    """One BGR frame at time t (seconds)."""
    p = video_path(src)
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"no frame at {t:.2f}s in {p.name}")
    return frame


def frames_at(src: str | Path, times: Sequence[float]) -> list[np.ndarray]:
    return [frame_at(src, t) for t in times]


# ---------------------------------------------------------------------------------------------- GPU lock


@contextlib.contextmanager
def gpu_lock(timeout: float = 3600.0) -> Iterator[None]:
    """Cross-process lock so concurrent prototypes never load several YOLO models onto the 12 GB GPU at once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = CACHE / "gpu.lock"
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            with contextlib.suppress(Exception):  # stale lock: owner gone or older than 20 min
                if time.time() - lock.stat().st_mtime > 1200:
                    lock.unlink()
                    continue
            if time.time() - t0 > timeout:
                raise TimeoutError("gpu.lock held too long")
            time.sleep(0.5)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


_MODELS: dict[str, Any] = {}


def _yolo(kind: str):
    """kind: 'det' (yolo26s TensorRT), 'det_l' (yolo26l), 'pose', 'seg'."""
    if kind not in _MODELS:
        from ultralytics import YOLO
        f = {"det": "yolo26s.engine", "det_l": "yolo26l.pt", "pose": "yolo26s-pose.pt", "seg": "yolo26s-seg.pt"}[kind]
        _MODELS[kind] = YOLO(str(MODELS / f), task={"det": "detect", "det_l": "detect", "pose": "pose", "seg": "segment"}[kind])
    return _MODELS[kind]


# ---------------------------------------------------------------------------------------------- tracks

COLS = ["f", "t", "tid", "cls", "name", "conf", "x1", "y1", "x2", "y2"]


@dataclass
class Tracks:
    """Tracked detections for one video. rows: DataFrame with COLS (+ cx, cy, bw, bh). Pixel coords of the source."""
    video: str
    src_fps: float
    stride: int
    w: int
    h: int
    duration: float
    names: dict[int, str]
    rows: pd.DataFrame
    model: str = "yolo26s"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        """Effective sampled frame rate."""
        return self.src_fps / self.stride

    def of(self, *names: str) -> pd.DataFrame:
        return self.rows[self.rows["name"].isin(names)] if names else self.rows

    def track(self, tid: int) -> pd.DataFrame:
        return self.rows[self.rows["tid"] == tid].sort_values("t")

    def summary(self, min_frames: int = 3) -> pd.DataFrame:
        """One row per track: tid, name (majority), first, last, dur, n, mean conf, box size, start/end centre."""
        r = self.rows[self.rows["tid"] >= 0]
        if r.empty:
            return pd.DataFrame(columns=["tid", "name", "first", "last", "dur", "n", "conf", "bw", "bh",
                                         "x0", "y0", "x1e", "y1e"])
        g = r.sort_values("t").groupby("tid")
        s = pd.DataFrame({
            "name": g["name"].agg(lambda x: x.value_counts().index[0]),
            "first": g["t"].min(), "last": g["t"].max(), "n": g.size(), "conf": g["conf"].mean(),
            "bw": g["bw"].median(), "bh": g["bh"].median(),
            "x0": g["cx"].first(), "y0": g["cy"].first(), "x1e": g["cx"].last(), "y1e": g["cy"].last()})
        s["dur"] = s["last"] - s["first"]
        s = s[s["n"] >= min_frames].reset_index()
        return s[["tid", "name", "first", "last", "dur", "n", "conf", "bw", "bh", "x0", "y0", "x1e", "y1e"]]

    def at(self, t: float, tol: float | None = None) -> pd.DataFrame:
        """Detections in the sampled frame nearest to t."""
        if self.rows.empty:
            return self.rows
        tol = tol if tol is not None else 0.75 / self.fps
        ts = self.rows["t"].unique()
        near = ts[np.argmin(np.abs(ts - t))]
        return self.rows[(self.rows["t"] == near)] if abs(near - t) <= tol else self.rows.iloc[0:0]

    def counts_per_second(self, *names: str) -> pd.DataFrame:
        """Per whole second: max simultaneous objects of each class (wide DataFrame, index = second)."""
        r = self.of(*names).copy()
        r["sec"] = r["t"].astype(int)
        per_frame = r.groupby(["sec", "f", "name"]).size().reset_index(name="n")
        wide = per_frame.groupby(["sec", "name"])["n"].max().unstack(fill_value=0)
        return wide.reindex(range(int(self.duration) + 1), fill_value=0)


def _with_geom(df: pd.DataFrame) -> pd.DataFrame:
    df["cx"], df["cy"] = (df["x1"] + df["x2"]) / 2, (df["y1"] + df["y2"]) / 2
    df["bw"], df["bh"] = df["x2"] - df["x1"], df["y2"] - df["y1"]
    return df


def tracks(src: str | Path, *, fps_target: float = 15.0, model: str = "det", conf: float = 0.25,
           refresh: bool = False, verbose: bool = True) -> Tracks:
    """YOLO26s (TensorRT) + ByteTrack over the whole video at ~fps_target, cached as gzip JSON.

    tid == -1 means detected but not yet confirmed by the tracker. Classes: COCO-80 (as the hosted YOLO11s).
    """
    name = name_of(src)
    meta = info(src)
    stride = max(1, int(round(meta["fps"] / fps_target)))
    f = CACHE / "tracks" / f"{name}__{model}__s{stride}__c{int(conf * 100)}.json.gz"
    if f.exists() and not refresh:
        d = json.loads(gzip.decompress(f.read_bytes()))
        df = pd.DataFrame(d["rows"], columns=COLS)
        return Tracks(d["video"], d["src_fps"], d["stride"], d["w"], d["h"], d["duration"],
                      {int(k): v for k, v in d["names"].items()}, _with_geom(df), d.get("model", model), d.get("extra", {}))
    rows: list[list[Any]] = []
    t0 = time.time()
    with gpu_lock():
        if f.exists() and not refresh:  # another process computed it while we waited for the lock
            return tracks(src, fps_target=fps_target, model=model, conf=conf)
        m = _yolo(model)
        names = {int(k): v for k, v in m.names.items()}
        m.predictor = None  # fresh tracker state per video
        for i, t, frame in iter_frames(src, stride=stride):
            res = m.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=640, verbose=False)[0]
            b = res.boxes
            if b is None or len(b) == 0:
                continue
            ids = b.id.int().tolist() if b.id is not None else [-1] * len(b)
            for tid, c, cf, xy in zip(ids, b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist()):
                rows.append([i, round(t, 3), tid, c, names[c], round(cf, 3), *[round(v, 1) for v in xy]])
        m.predictor = None
        _unload()
    d = {"video": name, "src_fps": meta["fps"], "stride": stride, "w": meta["w"], "h": meta["h"],
         "duration": meta["duration"], "names": names, "rows": rows, "model": model,
         "extra": {"seconds_to_compute": round(time.time() - t0, 1)}}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(gzip.compress(json.dumps(d).encode()))
    if verbose:
        print(f"[protokit] tracked {name}: {len(rows)} boxes, {meta['frames'] // stride} frames in {time.time() - t0:.0f}s")
    return tracks(src, fps_target=fps_target, model=model, conf=conf)


def poses(src: str | Path, *, fps_target: float = 15.0, conf: float = 0.25, refresh: bool = False) -> pd.DataFrame:
    """YOLO26s-pose + ByteTrack: rows f, t, tid, conf, x1..y2, kp (17x3 list: x, y, visibility). Cached."""
    name = name_of(src)
    meta = info(src)
    stride = max(1, int(round(meta["fps"] / fps_target)))
    f = CACHE / "poses" / f"{name}__s{stride}.json.gz"
    if f.exists() and not refresh:
        return _with_geom(pd.DataFrame(json.loads(gzip.decompress(f.read_bytes()))))
    rows = []
    with gpu_lock():
        if f.exists() and not refresh:  # computed by another process while we waited
            return _with_geom(pd.DataFrame(json.loads(gzip.decompress(f.read_bytes()))))
        m = _yolo("pose")
        m.predictor = None
        for i, t, frame in iter_frames(src, stride=stride):
            res = m.track(frame, persist=True, tracker="bytetrack.yaml", conf=conf, imgsz=640, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            ids = res.boxes.id.int().tolist() if res.boxes.id is not None else [-1] * len(res.boxes)
            kps = res.keypoints.data.tolist()
            for tid, cf, xy, kp in zip(ids, res.boxes.conf.tolist(), res.boxes.xyxy.tolist(), kps):
                rows.append({"f": i, "t": round(t, 3), "tid": tid, "conf": round(cf, 3),
                             "x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3],
                             "kp": [[round(a, 1), round(b, 1), round(c, 2)] for a, b, c in kp]})
        m.predictor = None
        _unload()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(gzip.compress(json.dumps(rows).encode()))
    return _with_geom(pd.DataFrame(rows))


def _unload() -> None:
    """Free GPU memory after every locked block: 15 prototypes may run at once on a 12 GB laptop GPU."""
    _MODELS.clear()
    with contextlib.suppress(Exception):
        import torch
        torch.cuda.empty_cache()


def seg_masks_many(frames: Sequence[np.ndarray], *, classes: Sequence[str] | None = None, conf: float = 0.25
                   ) -> list[list[dict[str, Any]]]:
    """Instance masks with YOLO26s-seg for many frames (one model load): per frame [{name, conf, box, mask HxW bool}]."""
    out: list[list[dict[str, Any]]] = []
    with gpu_lock():
        m = _yolo("seg")
        for frame in frames:
            res = m.predict(frame, conf=conf, imgsz=640, verbose=False, retina_masks=True)[0]
            dets: list[dict[str, Any]] = []
            if res.masks is not None:
                masks = res.masks.data.cpu().numpy() > 0.5
                for k, (c, cf, xy) in enumerate(zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(),
                                                    res.boxes.xyxy.tolist())):
                    if classes and m.names[c] not in classes:
                        continue
                    mk = masks[k]
                    if mk.shape != frame.shape[:2]:
                        mk = cv2.resize(mk.astype(np.uint8), (frame.shape[1], frame.shape[0])) > 0
                    dets.append({"name": m.names[c], "conf": cf, "box": xy, "mask": mk})
            out.append(dets)
        _unload()
    return out


def seg_masks(frame: np.ndarray, **kw: Any) -> list[dict[str, Any]]:
    """Single-frame seg_masks_many (reloads the model each call: batch with seg_masks_many when you can)."""
    return seg_masks_many([frame], **kw)[0]


def detect_many(frames: Sequence[np.ndarray], *, model: str = "det", conf: float = 0.25) -> list[list[dict[str, Any]]]:
    """Untracked detections for many frames (one model load): per frame [{name, conf, box}].
    model 'det' (yolo26s TensorRT) or 'det_l' (yolo26l, a different, larger model: use it as an independent check)."""
    out = []
    with gpu_lock():
        m = _yolo(model)
        for frame in frames:
            res = m.predict(frame, conf=conf, imgsz=640, verbose=False)[0]
            out.append([{"name": m.names[c], "conf": cf, "box": xy} for c, cf, xy in
                        zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(), res.boxes.xyxy.tolist())])
        _unload()
    return out


def detect(frame: np.ndarray, **kw: Any) -> list[dict[str, Any]]:
    return detect_many([frame], **kw)[0]


def pose_many(frames: Sequence[np.ndarray], *, conf: float = 0.25) -> list[list[dict[str, Any]]]:
    """Untracked YOLO26s-pose on many frames: per frame [{conf, box, kp 17x3 (x, y, visibility)}]. COCO keypoint
    order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles."""
    out = []
    with gpu_lock():
        m = _yolo("pose")
        for frame in frames:
            res = m.predict(frame, conf=conf, imgsz=640, verbose=False)[0]
            kps = res.keypoints.data.tolist() if res.keypoints is not None else []
            out.append([{"conf": cf, "box": xy, "kp": kp} for cf, xy, kp in
                        zip(res.boxes.conf.tolist(), res.boxes.xyxy.tolist(), kps)])
        _unload()
    return out


# ------------------------------------------------------------------------------------ DataEngine stand-in


def segments(src: str | Path, seconds: float = 5.0, tr: Tracks | None = None) -> pd.DataFrame:
    """One row per 5 s segment, like the DataEngine table: seg, start, end, unique tracks per class, peak
    simultaneous count per class, mean brightness. Columns n_<class> and peak_<class> for every class seen."""
    tr = tr or tracks(src)
    dur = tr.duration
    nseg = int(np.ceil(dur / seconds))
    r = tr.rows[tr.rows["tid"] >= 0].copy()
    r["seg"] = (r["t"] // seconds).astype(int)
    out = pd.DataFrame({"seg": range(nseg)})
    out["start"] = out["seg"] * seconds
    out["end"] = np.minimum(out["start"] + seconds, dur)
    for nm in sorted(r["name"].unique()):
        sub = r[r["name"] == nm]
        out[f"n_{nm}"] = out["seg"].map(sub.groupby("seg")["tid"].nunique()).fillna(0).astype(int)
        out[f"peak_{nm}"] = out["seg"].map(sub.groupby(["seg", "f"]).size().groupby("seg").max()).fillna(0).astype(int)
    out.insert(0, "video", tr.video)
    return out


def vastdb(tables: dict[str, pd.DataFrame]):
    """In-process duckdb with the given DataFrames registered as tables: the local stand-in for VastDB SQL."""
    import duckdb
    con = duckdb.connect()
    for k, df in tables.items():
        con.register(f"_{k}", df)
        con.execute(f'CREATE TABLE "{k}" AS SELECT * FROM _{k}')
    return con


# ----------------------------------------------------------------------------------------------- models


def _jpg_b64(img: np.ndarray | bytes, max_side: int = 640) -> str:
    if isinstance(img, bytes):
        return base64.b64encode(img).decode()
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def _key(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:32]


def vlm(prompt: str, images: Sequence[np.ndarray | bytes] = (), *, video: str | Path | None = None,
        max_tokens: int = 200, system: str | None = None, timeout: float = 240.0) -> str:
    """Ask local Cosmos3-Edge (stand-in for hosted Cosmos3-Reason). Cached by content. Returns '' if the
    server is down or PK_VLM=off (prototypes must still work without it, just with less text)."""
    if os.environ.get("PK_VLM", "on") == "off":
        return ""
    import httpx
    content: list[dict[str, Any]] = []
    for im in images:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _jpg_b64(im)}})
    if video is not None:
        b = Path(video).read_bytes()
        content.append({"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(b).decode()}})
    content.append({"type": "text", "text": prompt})
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    key = _key("vlm", msgs, max_tokens)
    f = CACHE / "vlm" / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["text"]
    body = {"model": "nvidia/Cosmos3-Edge", "messages": msgs, "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        t0 = time.time()
        r = httpx.post(f"{COSMOS_URL}/v1/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
        dt = time.time() - t0
    except Exception as e:  # server down or slow: degrade, never crash a prototype
        print(f"[protokit] vlm unavailable: {type(e).__name__}", file=sys.stderr)
        return ""
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"text": text, "seconds": round(dt, 1), "prompt": prompt}), encoding="utf-8")
    return text


def vlm_json(prompt: str, images: Sequence[np.ndarray | bytes] = (), **kw: Any) -> Any:
    """vlm() then parse the first JSON object/array in the reply; None if it doesn't parse."""
    import re
    text = vlm(prompt + "\nReply with JSON only.", images, **kw)
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, text, re.S)
        if m:
            with contextlib.suppress(Exception):
                return json.loads(m.group(0))
    return None


def _embed(inp: list[str], kind: str) -> np.ndarray:
    import httpx
    out, todo = [None] * len(inp), []
    for i, s in enumerate(inp):
        f = CACHE / "embed" / f"{_key(kind, s)}.npy"
        if f.exists():
            out[i] = np.load(f)
        else:
            todo.append(i)
    if todo:
        r = httpx.post(f"{EMBED_URL}/v1/embeddings", json={"model": "nvidia/cosmos-embed1", "input": [inp[i] for i in todo],
                                                            "request_type": kind, "encoding_format": "float"},
                       timeout=600)
        r.raise_for_status()
        for i, d in zip(todo, r.json()["data"]):
            v = np.asarray(d["embedding"], dtype=np.float32)
            (CACHE / "embed").mkdir(parents=True, exist_ok=True)
            np.save(CACHE / "embed" / f"{_key(kind, inp[i])}.npy", v)
            out[i] = v
    return np.stack(out)


def embed_text(texts: Sequence[str]) -> np.ndarray:
    """Cosmos-Embed1 text embeddings (N x 256, L2-normalised by the server). Cached."""
    return _embed(list(texts), "query")


def embed_video(clips: Sequence[str | Path]) -> np.ndarray:
    """Cosmos-Embed1 video embeddings for short mp4 clips (N x 256). Cached by file content."""
    uris = ["data:video/mp4;base64," + base64.b64encode(Path(c).read_bytes()).decode() for c in clips]
    return _embed(uris, "videos")


# ------------------------------------------------------------------------------------------ media output


PALETTE = [(66, 135, 245), (245, 130, 48), (60, 180, 75), (230, 25, 75), (145, 30, 180), (70, 240, 240),
           (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]


def color(i: int) -> tuple[int, int, int]:
    return PALETTE[int(i) % len(PALETTE)]


def label(img: np.ndarray, text: str, xy: tuple[int, int] = (8, 8), *, scale: float = 0.55,
          fg=(255, 255, 255), bg=(0, 0, 0)) -> np.ndarray:
    """Draw text with a solid background box, top-left at xy. In place; returns img."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(xy[0]), int(xy[1])
    cv2.rectangle(img, (x, y), (x + tw + 8, y + th + base + 6), bg, -1)
    cv2.putText(img, text, (x + 4, y + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)
    return img


def box(img: np.ndarray, xyxy: Sequence[float], col=(0, 255, 0), text: str | None = None, thick: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
    if text:
        label(img, text, (x1, max(0, y1 - 22)), scale=0.45, bg=col)
    return img


def fit(img: np.ndarray, w: int) -> np.ndarray:
    h0, w0 = img.shape[:2]
    return cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)


def sheet(images: Sequence[np.ndarray], labels: Sequence[str] | None = None, *, cols: int = 4, tile_w: int = 320,
          pad: int = 4) -> np.ndarray:
    """Contact sheet of BGR images (each resized to tile_w wide), optional label per tile."""
    tiles = [fit(im, tile_w) for im in images]
    if labels:
        for t, s in zip(tiles, labels):
            label(t, s, (4, 4), scale=0.45)
    th = max(t.shape[0] for t in tiles)
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * (th + pad) + pad, cols * (tile_w + pad) + pad, 3), 24, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        y, x = pad + r * (th + pad), pad + c * (tile_w + pad)
        canvas[y:y + t.shape[0], x:x + tile_w] = t
    return canvas


def save_jpg(img: np.ndarray, path: str | Path, q: int = 88) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    p.write_bytes(buf.tobytes())
    return p


def save_png(img: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    p.write_bytes(buf.tobytes())
    return p


class Mp4Writer:
    """Browser-playable H.264 writer (ffmpeg pipe). with Mp4Writer(path, fps) as w: w.write(bgr_frame)."""

    def __init__(self, path: str | Path, fps: float, width: int | None = None):
        self.path, self.fps, self.width, self.proc = Path(path), fps, width, None

    def write(self, frame: np.ndarray) -> None:
        if self.width and frame.shape[1] != self.width:
            frame = fit(frame, self.width)
        h, w = frame.shape[:2]
        if self.proc is None:
            self.size = (w - w % 2, h - h % 2)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen(
                [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.size[0]}x{self.size[1]}",
                 "-r", f"{self.fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)
        self.proc.stdin.write(np.ascontiguousarray(frame[:self.size[1], :self.size[0]]).tobytes())

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *a: Any) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()


def cut(src: str | Path, start: float, end: float, dst: str | Path, *, width: int | None = 960,
        audio: bool = False) -> Path:
    """Re-encode [start, end) of a video to a browser mp4 (frame-accurate)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale='trunc(min({width},iw)/2)*2':-2" if width else "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video_path(src)), "-t", f"{end - start:.3f}",
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    cmd += ["-c:a", "aac"] if audio else ["-an"]
    subprocess.run(cmd + [str(dst)], check=True)
    return dst


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def mmss(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


# ----------------------------------------------------------------------------------------------- reports

STATE_COLORS = {"seen": "#2e9e5b", "present": "#2e9e5b", "ok": "#2e9e5b", "green": "#2e9e5b",
                "nothing": "#9aa5b1", "absent": "#d64545", "red": "#d64545", "alert": "#d64545",
                "blind": "#555", "unknown": "#555", "grey": "#777", "amber": "#e0a100", "warn": "#e0a100"}


def timeline_svg(spans: Sequence[tuple[float, float, str, str]], duration: float, *, width: int = 900,
                 height: int = 26, ticks: int = 10) -> str:
    """spans: (start, end, state, tooltip). state picks a colour from STATE_COLORS (or is a #hex)."""
    sx = width / max(duration, 1e-6)
    parts = [f'<svg viewBox="0 0 {width} {height + 16}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg">',
             f'<rect x="0" y="0" width="{width}" height="{height}" fill="#e8ecef"/>']
    for s, e, st, tip in spans:
        c = STATE_COLORS.get(st, st if st.startswith("#") else "#888")
        parts.append(f'<rect x="{s * sx:.1f}" y="0" width="{max(1.0, (e - s) * sx):.1f}" height="{height}" fill="{c}">'
                     f'<title>{html.escape(tip)}</title></rect>')
    for k in range(ticks + 1):
        x = k * width / ticks
        parts.append(f'<text x="{min(x, width - 30):.0f}" y="{height + 13}" font-size="10" fill="#666">'
                     f'{mmss(k * duration / ticks)}</text>')
    parts.append("</svg>")
    return "".join(parts)


CSS = """
:root{--bg:#f7f8fa;--fg:#1d232a;--muted:#5d6b78;--card:#fff;--line:#dfe3e8;--accent:#2f6fde}
@media (prefers-color-scheme: dark){:root{--bg:#14171b;--fg:#e6e9ec;--muted:#9aa5b1;--card:#1d2227;--line:#2c333a;--accent:#6ea0ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 60px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:28px 0 8px}
.sub{color:var(--muted);margin:0 0 16px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.kv div{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.kv b{display:block;font-size:22px}.kv span{color:var(--muted);font-size:12px}
img,video{max-width:100%;border-radius:6px;display:block}figure{margin:12px 0}figcaption{color:var(--muted);font-size:13px;margin-top:4px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}
pre{white-space:pre-wrap;font-size:12px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;overflow:auto}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}
.foot{color:var(--muted);font-size:12px;margin-top:40px;border-top:1px solid var(--line);padding-top:10px}
"""


class Report:
    """Minimal HTML report. Media paths are made relative to the report file."""

    def __init__(self, path: str | Path, title: str, subtitle: str = ""):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.title, self.parts = title, [f"<h1>{html.escape(title)}</h1>"]
        if subtitle:
            self.parts.append(f'<p class="sub">{html.escape(subtitle)}</p>')
        self.credits: list[str] = []

    def _rel(self, p: str | Path) -> str:
        return os.path.relpath(Path(p).resolve(), self.path.parent.resolve()).replace("\\", "/")

    def h2(self, text: str) -> "Report":
        self.parts.append(f"<h2>{html.escape(text)}</h2>")
        return self

    def text(self, text: str, *, raw: bool = False) -> "Report":
        self.parts.append(f"<p>{text if raw else html.escape(text)}</p>")
        return self

    def html(self, snippet: str) -> "Report":
        self.parts.append(snippet)
        return self

    def kv(self, items: dict[str, Any]) -> "Report":
        self.parts.append('<div class="kv">' + "".join(
            f"<div><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>" for k, v in items.items()) + "</div>")
        return self

    def image(self, path: str | Path, caption: str = "") -> "Report":
        self.parts.append(f'<figure><img src="{self._rel(path)}" loading="lazy"><figcaption>{html.escape(caption)}'
                          "</figcaption></figure>")
        return self

    def video(self, path: str | Path, caption: str = "") -> "Report":
        self.parts.append(f'<figure><video src="{self._rel(path)}" controls muted playsinline preload="metadata"></video>'
                          f"<figcaption>{html.escape(caption)}</figcaption></figure>")
        return self

    def table(self, rows: Sequence[dict[str, Any]] | pd.DataFrame, cols: Sequence[str] | None = None,
              max_rows: int = 200) -> "Report":
        if isinstance(rows, pd.DataFrame):
            rows = rows.to_dict("records")
        rows = list(rows)[:max_rows]
        cols = list(cols or (rows[0].keys() if rows else []))
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        self.parts.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>" + "".join(
            "<tr>" + "".join(f"<td>{html.escape(fmt(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
            + "</table>")
        return self

    def timeline(self, spans: Sequence[tuple[float, float, str, str]], duration: float, caption: str = "") -> "Report":
        self.parts.append(f'<figure>{timeline_svg(spans, duration)}<figcaption>{html.escape(caption)}</figcaption></figure>')
        return self

    def json(self, obj: Any, caption: str = "") -> "Report":
        if caption:
            self.parts.append(f'<p class="sub">{html.escape(caption)}</p>')
        self.parts.append(f"<pre>{html.escape(json.dumps(obj, indent=1, default=str, ensure_ascii=False))}</pre>")
        return self

    def credit(self, src: str | Path) -> "Report":
        c = credit(src)
        if c not in self.credits:
            self.credits.append(c)
        return self

    def save(self) -> Path:
        foot = ('<div class="foot">Video prototype. All models ran '
                "locally: YOLO26s + ByteTrack for the DataEngine YOLO step, local Cosmos3-Edge standing in for Cosmos3-Reason, "
                "local Cosmos-Embed1, duckdb standing in for VastDB. Footage: " + "; ".join(html.escape(c) for c in self.credits)
                + ". Local tests only.</div>")
        self.path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                             f'content="width=device-width,initial-scale=1"><title>{html.escape(self.title)}</title>'
                             f"<style>{CSS}</style></head><body><main>{''.join(self.parts)}{foot}</main></body></html>",
                             encoding="utf-8")
        return self.path


def write_results(path: str | Path, obj: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False), encoding="utf-8")
    return p


if __name__ == "__main__":
    for k, v in videos().items():
        print(f"{k:<38} {v['license']:<14} {v['origin']:<18} {Path(v['path']).name}")
