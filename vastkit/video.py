"""ffmpeg / PyAV / OpenCV video utilities.

Browser and Weave playback need H.264 yuv420p (+faststart). OpenCV's 'mp4v' FourCC is NOT
browser-playable, so everything here writes through ffmpeg.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
H264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
EVEN = "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # yuv420p needs even width and height


def _scale(width: int | None) -> str:
    """Downscale to at most `width` (never up), keeping aspect; both sides even for libx264/yuv420p."""
    return f"scale='trunc(min({width},iw)/2)*2':-2" if width else EVEN


def _run(cmd: list[str]) -> None:
    # ffmpeg writes UTF-8; the Windows locale default (cp1252) mangles non-ASCII paths in its errors
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed: {p.stderr.strip()[-600:]}")


def probe(path: str | Path) -> dict[str, Any]:
    """{duration, fps, width, height, codec, frames, has_audio} from ffprobe."""
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                          str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace",
                         check=True).stdout  # UTF-8 JSON: cp1252 decoding garbles or fails on non-ASCII paths
    j = json.loads(out)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        fps = float(Fraction(rate)) if rate != "0/0" else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    dur = float(j.get("format", {}).get("duration") or v.get("duration") or 0)
    return {"duration": dur, "fps": fps, "width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "codec": v.get("codec_name"), "frames": int(v.get("nb_frames") or 0) or round(dur * fps),
            "has_audio": any(s.get("codec_type") == "audio" for s in j.get("streams", []))}


def _out(dst: str | Path | None, suffix: str = ".mp4") -> Path:
    if dst:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        return Path(dst)
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # an open fd keeps the file undeletable on Windows (and leaks one per call)
    return Path(p)


def prep_clip(src: str | Path, dst: str | Path | None = None, *, start: float | None = None,
              duration: float | None = None, width: int = 640, fps: float | None = 4.0, crf: int = 28,
              burn_timestamp: bool = False, audio: bool = False) -> Path:
    """Cut + downscale + cap fps (never above the source: Cosmos returns 400) -> small H.264 clip.

    ~0.3-1 MB per 10 s at 640px/4fps; keeps Cosmos under ~16k multimodal tokens.
    """
    info = probe(src)
    out = _out(dst)
    vf = [_scale(width)]
    if fps and info["fps"]:
        vf.append(f"fps={min(fps, info['fps'])}")
    if burn_timestamp:
        off = start or 0
        vf.append("drawtext=text='%{pts\\:hms\\:" + str(off) + "}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5")
    cmd = [FFMPEG, "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vf", ",".join(vf), *H264, "-preset", "veryfast", "-crf", str(crf)]
    cmd += ["-c:a", "aac", "-b:a", "64k"] if audio and info["has_audio"] else ["-an"]
    _run(cmd + [str(out)])
    return out


def to_browser_mp4(src: str | Path, dst: str | Path | None = None, *, crf: int = 23) -> Path:
    out = _out(dst)
    _run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-vf", EVEN, *H264, "-preset", "veryfast",
          "-crf", str(crf), "-c:a", "aac", str(out)])
    return out


def segment(src: str | Path, out_dir: str | Path, *, seconds: float = 5.0, width: int | None = None,
            prefix: str | None = None) -> list[dict[str, Any]]:
    """Split into fixed-length re-encoded segments (exact cut points, like the DataEngine segmenter).

    Returns [{path, segment_number (1-based), start_sec, end_sec}].
    """
    info = probe(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or Path(src).stem
    n = max(1, int(-(-info["duration"] // seconds)))  # ceil
    segs = []
    for i in range(n):
        start = i * seconds
        end = min(info["duration"], start + seconds)
        if end - start < 0.2:
            break
        p = out_dir / f"{stem}_segment_{i + 1:03d}.mp4"
        _run([FFMPEG, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
              "-vf", _scale(width), *H264, "-preset", "veryfast", "-crf", "23", "-an", str(p)])
        segs.append({"path": str(p), "segment_number": i + 1, "start_sec": start, "end_sec": end})
    return segs


def _decode_at(c: Any, s: Any, target: int, start: int, step: int) -> Any:
    """First frame with pts >= target (the last one past EOF). MPEG-TS seeks land on a packet, not a
    keyframe, so decoding can begin after target: then seek further back and retry. The last retry may
    go before the stream start (TS with B-frames needs it to reach the first GOP); a container that
    refuses that seek (AVI, whose first frame can come after start_time) keeps the frame already decoded."""
    import av
    back, frame = 0, None
    while True:
        at = target - back
        try:
            c.seek(at, stream=s, any_frame=False, backward=True)
        except (av.error.FFmpegError, OSError):
            if frame is None:
                raise
            return frame
        frame, n = None, 0
        for f in c.decode(s):
            frame, n = f, n + 1
            if f.pts is not None and f.pts >= target:
                break
        if at < start or (frame is not None and (n > 1 or frame.pts is None or frame.pts <= target)):
            return frame
        back = back * 2 or step


def _frames_at(src: str | Path, times: Sequence[float], *, max_side: int = 768,
               quality: int = 85) -> Iterator[tuple[float, bytes]]:
    import av
    import cv2
    with av.open(str(src)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        start = s.start_time or 0  # pts of the first frame (MPEG-TS, HLS, offset mp4s): times are relative to it
        for t in times:
            frame = _decode_at(c, s, start + int(max(0.0, t) / tb), start, max(1, int(1.0 / tb)))
            if frame is None:
                continue
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                yield t, buf.tobytes()


def frames_at(src: str | Path, times: Sequence[float], *, max_side: int = 768, quality: int = 85) -> list[bytes]:
    """JPEG bytes for the first frame at or after each time (seconds from the stream start; PyAV seek + decode)."""
    return [jpg for _, jpg in _frames_at(src, times, max_side=max_side, quality=quality)]


def keyframes(src: str | Path, n: int = 3, **kw: Any) -> list[tuple[float, bytes]]:
    """n uniformly spaced frames -> [(t_sec, jpeg_bytes)] (a frame that fails to decode is left out)."""
    dur = probe(src)["duration"] or 1.0
    times = [(i + 0.5) * dur / n for i in range(n)]
    return list(_frames_at(src, times, **kw))


def contact_sheet(src: str | Path, n: int = 9, cols: int = 3, tile_w: int = 320, label: bool = True) -> bytes:
    """Grid of n frames with timestamps (JPEG bytes): one image to send to a VLM or a slide."""
    import cv2
    import numpy as np
    frames = keyframes(src, n, max_side=tile_w)
    tiles = []
    for t, jpg in frames:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1])))
        if label:
            cv2.putText(img, f"{t:5.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, f"{t:5.1f}s", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(img)
    if not tiles:
        raise RuntimeError("no frames decoded")
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    ok, buf = cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def burn_boxes(src: str | Path, detections: dict[int, list[dict[str, Any]]], dst: str | Path | None = None,
               *, colors: dict[str, tuple[int, int, int]] | None = None) -> Path:
    """Draw {frame_index: [{label, conf (or confidence), bbox:[x1,y1,x2,y2]}]} onto the video -> H.264 mp4.

    frames.Detector output (``conf``) can be passed as is. Each box is tagged with its class and score only.
    """
    import av
    import cv2
    out = _out(dst)
    info = probe(src)
    w, h = info["width"] - info["width"] % 2, info["height"] - info["height"] % 2
    fps = info["fps"] or 30
    enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", *H264, "-preset", "veryfast",
                            "-crf", "23", str(out)], stdin=subprocess.PIPE)
    colors = colors or {}
    assert enc.stdin
    with av.open(str(src)) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            img = f.to_ndarray(format="bgr24")[:h, :w].copy()
            for d in detections.get(i, []):
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                col = colors.get(d["label"], (0, 200, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                conf = d.get("conf", d.get("confidence", 0))
                cv2.putText(img, f"{d['label']} {conf:.2f}",(x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            enc.stdin.write(img.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg encode failed")
    return out


def data_uri(data: bytes | str | Path, mime: str) -> str:
    raw = data if isinstance(data, bytes) else Path(data).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def extract_wav(video: str | Path, dst: str | Path | None = None, rate: int = 16000) -> Path:
    out = _out(dst, ".wav")
    _run([FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(rate),
          "-c:a", "pcm_s16le", str(out)])
    return out


def concat(paths: Iterable[str | Path], dst: str | Path) -> Path:
    """Lossless concat of same-codec clips (ffmpeg concat demuxer)."""
    lst = _out(None, ".txt")
    lst.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    out = _out(dst)
    try:
        _run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
              str(out)])
    finally:
        lst.unlink(missing_ok=True)
    return out
