import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vastkit import video
from vastkit.s3 import load_sidecar, split_uri

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    p = tmp_path_factory.mktemp("v") / "test.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440", "-t", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


def test_probe(src):
    info = video.probe(src)
    assert info["width"] == 1280 and abs(info["fps"] - 30) < 0.1 and info["has_audio"]
    assert 11.5 < info["duration"] < 12.5


def test_prep_clip_caps_fps_and_width(src, tmp_path):
    out = video.prep_clip(src, tmp_path / "c.mp4", start=2, duration=5, width=640, fps=60)
    info = video.probe(out)
    assert info["width"] == 640 and abs(info["fps"] - 30) < 0.1  # 60 requested, capped to source
    assert 4.5 < info["duration"] < 5.5 and not info["has_audio"] and info["codec"] == "h264"
    out2 = video.prep_clip(src, tmp_path / "d.mp4", duration=3, fps=4, burn_timestamp=True)
    assert abs(video.probe(out2)["fps"] - 4) < 0.1


def test_segment(src, tmp_path):
    segs = video.segment(src, tmp_path / "segs", seconds=5)
    assert [s["segment_number"] for s in segs] == [1, 2, 3]
    assert segs[-1]["end_sec"] == pytest.approx(12, abs=0.2)
    assert Path(segs[0]["path"]).name == "test_segment_001.mp4"


def test_frames_and_sheet(src):
    kf = video.keyframes(src, n=3)
    assert len(kf) == 3 and kf[0][1][:2] == b"\xff\xd8"
    sheet = video.contact_sheet(src, n=4, cols=2)
    assert sheet[:2] == b"\xff\xd8"


def test_burn_boxes_and_wav(src, tmp_path):
    short = video.prep_clip(src, tmp_path / "s.mp4", duration=1, fps=None)
    out = video.burn_boxes(short, {0: [{"label": "person", "confidence": 0.9, "bbox": [10, 10, 100, 100], "id": 3}]},
                           tmp_path / "b.mp4")
    assert video.probe(out)["codec"] == "h264"
    wav = video.extract_wav(src, tmp_path / "a.wav")
    assert wav.read_bytes()[:4] == b"RIFF"


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *map(str, args)], check=True)


def test_odd_dimensions_encode_even(tmp_path):
    odd = tmp_path / "odd.mp4"  # yuv444p allows odd sizes; our yuv420p outputs must not inherit them
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=639x479:rate=10", "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv444p", odd)
    dims = lambda p: (video.probe(p)["width"], video.probe(p)["height"])  # noqa: E731
    assert dims(video.prep_clip(odd, tmp_path / "p.mp4", width=640)) == (638, 478)
    assert dims(video.to_browser_mp4(odd, tmp_path / "b.mp4")) == (638, 478)
    assert {dims(s["path"]) for s in video.segment(odd, tmp_path / "s", seconds=1)} == {(638, 478)}
    assert {dims(s["path"]) for s in video.segment(odd, tmp_path / "w", seconds=1, width=321)} == {(320, 240)}


def test_temp_outputs_are_closed_and_cleaned(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "t"))
    (tmp_path / "t").mkdir()
    p = video._out(None)
    p.unlink()  # an fd left open by mkstemp makes this fail on Windows
    clip = tmp_path / "c.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", clip)
    out = video.concat([clip, clip], tmp_path / "cc.mp4")
    assert 1.8 < video.probe(out)["duration"] < 2.3
    assert list((tmp_path / "t").iterdir()) == []  # concat's list file is removed


def test_frames_at_is_relative_to_stream_start(tmp_path):
    plain, off, ts = tmp_path / "plain.mp4", tmp_path / "off.mp4", tmp_path / "plain.ts"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "6", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "10", plain)
    _ffmpeg("-i", plain, "-c", "copy", "-output_ts_offset", "3", off)  # mp4 whose first pts is 3 s
    _ffmpeg("-i", plain, "-c", "copy", "-f", "mpegts", ts)  # MPEG-TS: non-zero start, packet-level seeks
    times = [0.0, 0.5, 2.0, 4.5]
    want = video.frames_at(plain, times)
    assert len(set(want)) == len(times)
    assert video.frames_at(off, times) == want and video.frames_at(ts, times) == want
    assert [t for t, _ in video.keyframes(off, n=3)] == pytest.approx([1.0, 3.0, 5.0], abs=0.05)


def test_frames_at_before_the_first_frame(tmp_path):
    """h264 copied into AVI: start_time 0 but the first frame at 0.1 s, and seeking before 0 is refused."""
    plain, avi = tmp_path / "plain.mp4", tmp_path / "a.avi"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", "10", plain)
    _ffmpeg("-i", plain, "-c", "copy", avi)
    assert video.frames_at(avi, [0.0, 0.05]) == video.frames_at(plain, [0.0, 0.0])
    assert len(video.frames_at(avi, [0.0, 0.5, 2.0])) == 3
    assert len(video.keyframes(avi, n=30)) == 30  # the first time (0.05 s) falls before the first frame


_RUN = subprocess.run


def _child(monkeypatch, out: bytes, err: bytes = b"", code: int = 0):
    """subprocess.run runs a Python child that writes these raw bytes instead of ffmpeg/ffprobe."""
    import sys
    script = f"import sys; sys.stdout.buffer.write({out!r}); sys.stderr.buffer.write({err!r}); sys.exit({code})"
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _RUN([sys.executable, "-c", script], **kw))


def test_probe_and_errors_decode_utf8(monkeypatch, tmp_path):
    """ffprobe/ffmpeg write UTF-8 whatever the locale; cp1252 failed on byte 0x9d of a CJK file name."""
    import re
    name = "東京 café.mp4"
    info = {"streams": [{"codec_type": "video", "width": 64, "height": 48, "avg_frame_rate": "10/1"}],
            "format": {"filename": name, "duration": "1.0"}}
    _child(monkeypatch, json.dumps(info, ensure_ascii=False).encode("utf-8"))
    assert video.probe(name)["height"] == 48
    _child(monkeypatch, b"", f"{name}: Invalid data\n".encode("utf-8"), code=1)
    with pytest.raises(RuntimeError, match=re.escape(f"{name}: Invalid data")):
        video.to_browser_mp4(name, tmp_path / "o.mp4")


def test_burn_boxes_reads_detector_conf(src, tmp_path, monkeypatch):
    import cv2
    tags, put = [], cv2.putText
    monkeypatch.setattr(cv2, "putText", lambda img, text, *a, **kw: (tags.append(text), put(img, text, *a, **kw))[1])
    short = video.prep_clip(src, tmp_path / "s.mp4", duration=0.5, fps=None, width=320)
    video.burn_boxes(short, {0: [{"label": "person", "conf": 0.87, "bbox": [1, 1, 50, 50], "id": 3}],
                             1: [{"label": "car", "confidence": 0.5, "bbox": [1, 1, 50, 50]}]}, tmp_path / "b.mp4")
    assert tags == ["person 0.87", "car 0.50"]  # class and score only; an "id" key is not drawn


def test_sidecar_and_uri(tmp_path):
    import gzip
    p = tmp_path / "x.json.gz"
    p.write_bytes(gzip.compress(json.dumps({"frames": []}).encode()))
    assert load_sidecar(p) == {"frames": []}
    assert split_uri("s3://b/k/x.mp4") == ("b", "k/x.mp4")
    with pytest.raises(ValueError):
        split_uri("http://x")
