import json
import threading
import time

import numpy as np

from vastkit import frames

FRAME = np.zeros((4, 4, 3), np.uint8)


class _BlockingCap:
    """Fake live capture whose read() takes a while; records a release() that overlaps a read()."""

    def __init__(self):
        self.reading = self.released = self.raced = False

    def get(self, prop):
        return 30.0

    def read(self):
        self.reading = True
        time.sleep(0.02)
        self.reading = False
        return True, FRAME

    def release(self):
        self.raced = self.raced or self.reading
        self.released = True


def test_live_close_releases_after_read_returns(monkeypatch):
    cap, readers = _BlockingCap(), []

    class _Spy(frames._Latest):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            readers.append(self)

    monkeypatch.setattr(frames, "open_source", lambda source: cap)
    monkeypatch.setattr(frames, "_Latest", _Spy)
    gen = frames.iter_frames("0")  # live: background latest-frame reader
    next(gen)
    gen.close()  # consumer breaks early (bench --max-frames)
    assert cap.released and not cap.raced and not readers[0].th.is_alive()


class _FileCap:
    def __init__(self, n):
        self.n, self.released = n, False

    def read(self):
        if self.n <= 0:
            return False, None
        self.n -= 1
        return True, FRAME

    def release(self):
        self.released = True


class _ScriptedReader:
    """Stands in for _Latest: each lock acquisition reveals the next index the reader got to."""

    def __init__(self, cap, paced):
        self.seq = iter([0, 5, 6, 9, 10])
        self.idx, self.frame, self.ok, self.stopped = -1, None, True, False
        self.th = threading.Thread(target=lambda: None)
        self.th.start()
        self.lock = self

    def __enter__(self):
        try:
            self.idx, self.frame = next(self.seq), FRAME
        except StopIteration:
            self.ok = False
        return self

    def __exit__(self, *exc):
        return False


def test_stride_skips_carry_dropped_frames(monkeypatch):
    monkeypatch.setattr(frames, "open_source", lambda source: _FileCap(0))
    monkeypatch.setattr(frames, "_Latest", _ScriptedReader)
    got = [(i, d) for i, _, _, d in frames.iter_frames("0", stride=2)]
    # 1-4 never read before 5 (skipped by stride) -> counted at 6; 7-8 lost before 9 -> counted at 10
    assert got == [(0, 0), (6, 4), (10, 2)]


def test_bench_creates_jsonl_parent(monkeypatch, tmp_path):
    cap = _FileCap(3)
    monkeypatch.setattr(frames, "open_source", lambda source: cap)
    det = lambda frame: [{"label": "x", "conf": 0.5, "bbox": [0, 0, 1, 1]}]  # noqa: E731
    out = tmp_path / "runs" / "dets.jsonl"
    res = frames.bench("clip.mp4", det, paced=False, jsonl=out)
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert res["frames"] == 3 and res["dropped"] == 0 and len(rows) == 3 and cap.released
    assert rows[0]["detections"][0]["conf"] == 0.5
