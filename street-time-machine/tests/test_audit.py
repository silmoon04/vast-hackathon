"""Known-truth pipeline audit: offline rules on a 12-row fixture, the clock reader on rendered text, the
clock-vs-picture checks on frames built in memory. No video, no network."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from stm import audit as A

SPANS = [
    {"start": 0, "end": 20, "state": "live", "src": [585, 605], "light": "day colour", "clock_start": "2026-09-01 10:15:00"},
    {"start": 20, "end": 30, "state": "frozen", "src": [605, 605], "clock_start": "2026-09-01 10:15:20"},
    {"start": 30, "end": 35, "state": "black", "clock_start": "2026-09-01 10:15:30"},
    {"start": 35, "end": 50, "state": "live", "src": [605, 620], "light": "day colour", "clock_start": "2026-09-01 10:15:35"},
    {"start": 50, "end": 60, "state": "live", "src": [1834.088, 1844.088], "light": "night infrared, black and white",
     "clock_start": "2026-09-02 21:40:00"},
]
TRUTH_SEG = ["live"] * 4 + ["frozen"] * 2 + ["black"] + ["live"] * 3 + ["ir"] * 2
CAPTIONS = [
    "A white SUV drives along a quiet residential street. Parked cars line the road.",
    "A white SUV drives from left to right. No pedestrians or other moving vehicles are present.",
    "A gray SUV is driving down the street at a moderate speed.",
    "A gray SUV drives across the frame. The scene appears calm and quiet.",
    "The scene is a quiet residential street with parked cars. There are no moving vehicles in the scene.",
    "A quiet street with a silver SUV parked by the curb. There are no moving vehicles visible.",
    "The video shows a completely black screen. A timestamp reads 2026-09-01 10:15:30, then 2026-09-01 10:15:34.",
    "A silver pickup truck drives from right to left across the frame.",
    "A silver pickup truck is seen driving along a residential street.",
    "A white pickup truck is driving along the street.",
    "A nighttime street captured in black and white. The timestamp reads 2026-09-02 21:40:00.",
    "A nighttime street in black and white. The street is quiet, with no visible movement or activity.",
]


def _frames(frozen: bool, n: int = 100) -> list[dict]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        j = 0 if frozen else int(rng.integers(0, 3))
        out.append({"frame_index": i, "detections": [{"label": "car", "bbox": [100 + j, 400, 200 + j, 470]},
                                                     {"label": "car", "bbox": [700, 500, 820, 590]}]})
    return out


def fixture_segments(captions: list[str] = CAPTIONS) -> list[dict]:
    segs = []
    for k, cap in enumerate(captions):
        frozen = TRUTH_SEG[k] == "frozen"
        black = TRUTH_SEG[k] == "black"
        frames = [{"frame_index": i, "detections": []} for i in range(100)] if black else _frames(frozen)
        segs.append({"segment_number": k + 1, "segment_start_sec": 5.0 * k, "segment_end_sec": 5.0 * (k + 1),
                     "reasoning_content": cap, "object_counts": "{}" if black else '{"car": 2}',
                     "detection_count": 0 if black else 200, "detections": {"frames": frames}})
    return segs


FLAGS = [
    {"kind": "clock_runs_on_frozen", "start": 20.0, "end": 30.0, "severity": "alarm", "text": "clock ran 9 s on a held frame"},
    {"kind": "clock_runs_on_black", "start": 30.0, "end": 35.0, "severity": "info", "text": "clock ran over black"},
    {"kind": "splice", "start": 35.0, "end": 35.0, "severity": "alarm", "text": "picture continues, clock +15 s"},
    {"kind": "date_change_at_cut", "start": 50.05, "end": 50.05, "severity": "alarm", "text": "date changes at the cut"},
]


def run_fixture(captions: list[str] = CAPTIONS, ours_errors: dict[int, str] | None = None) -> tuple[list, list, dict]:
    truth_sec, truth_clock = A.truth_seconds(SPANS, 60)
    ours = list(truth_sec)
    for s, lab in (ours_errors or {}).items():
        ours[s] = lab
    flags = [dict(f) for f in FLAGS]
    rows = A.build_rows(fixture_segments(captions), TRUTH_SEG, ours, truth_clock, flags)
    events = []
    for e in A.truth_events(SPANS):
        f = A.ours_caught(e, flags)
        e = {**e, "ours": f is not None, "ours_flag": f}
        e["cosmos"] = A.cosmos_noticed(e, rows)
        e["text"] = A.EVENT_WORDS[e["kind"]]
        events.append(e)
    A.add_event_notes(rows, events)
    return rows, events, A.score(rows, events, flags, ours, truth_sec)


# ---------------------------------------------------------------- ground truth
def test_truth_seconds_and_clock():
    lab, clk = A.truth_seconds(SPANS, 60)
    assert lab[:20] == ["live"] * 20 and lab[20:30] == ["frozen"] * 10 and lab[30:35] == ["black"] * 5
    assert lab[35:50] == ["live"] * 15 and lab[50:] == ["ir"] * 10
    assert clk[0] == "2026-09-01 10:15:00" and clk[34] == "2026-09-01 10:15:34" and clk[35] == "2026-09-01 10:15:35"
    assert clk[59] == "2026-09-02 21:40:09"


def test_truth_events_from_spans():
    ev = A.truth_events(SPANS)
    kinds = [e["kind"] for e in ev]
    assert kinds == ["clock_runs_on_frozen", "clock_runs_on_black", "splice", "date_change_at_cut"]
    sp = ev[2]
    assert sp["start"] == 35 and sp["ref_t"] == 20 and sp["missing_s"] == 15     # src 605 first shown at 10:15:20
    dc = ev[3]
    assert dc["start"] == 50 and dc["clock_from"] == "2026-09-01 10:15:49" and dc["clock_to"] == "2026-09-02 21:40:00"
    # a clip that is one live span has nothing to find
    assert A.truth_events([{"start": 0, "end": 15, "state": "live", "src": [675, 690],
                            "clock_start": "2026-09-01 10:16:30"}]) == []


# ---------------------------------------------------------------- our per-second ledger
def test_label_seconds_rules():
    luma = np.array([80.0] * 4 + [80.0] * 4 + [3.0] * 2 + [80.0] * 2)
    sat = np.array([100.0] * 10 + [4.0] * 2)
    diff = np.array([np.nan, 2, 2, 2, 1.5, 0.0, 0.08, 0.0, 70, 0.0, 60, 2.0])
    lab = A.label_seconds(luma, sat, diff)
    # 4..7 are four identical samples (the first of the run included); 8..9 black wins; 10..11 grey = ir
    assert lab == ["live"] * 4 + ["frozen"] * 4 + ["black"] * 2 + ["ir"] * 2


def test_frozen_needs_three_samples():
    still = np.array([False, True, False, False, True, True, False])
    assert A.held_runs(still, 3).tolist() == [False, False, False, True, True, True, False]


# ---------------------------------------------------------------- captions
@pytest.mark.parametrize("text, want", [
    ("The video shows a completely black screen throughout its duration.", {"black": True, "frozen": False}),
    ("A nighttime residential area captured in black and white.", {"ir": True, "dark": True, "black": False}),
    ("A gray SUV drives along a residential street.", {"movement": True, "still": False}),
    ("There are no moving vehicles or pedestrians in the scene.", {"movement": False, "still": True}),
    ("No pedestrians or other moving vehicles are present in the scene.", {"still": False}),
    ("The environment appears quiet and still, with no visible movement or activity.", {"still": True}),
    ("The picture is frozen on a single frame.", {"frozen": True}),
    ("A dark-colored sedan is parked in a driveway.", {"dark": False, "black": False}),
    ("The timestamp reads 2026-09-02 21:40:00.", {"clock": True, "clock_values": ["2026-09-02 21:40:00"]}),
    ("The scene suddenly cuts to a different day.", {"cut": True}),
])
def test_caption_claims(text, want):
    got = A.caption_claims(text)
    for k, v in want.items():
        assert got[k] == v, (k, got)


def test_excerpt_limit_and_people():
    cap = ("A man in a red jacket and blue jeans walks a small dog along the pavement. The street is quiet. "
           "No pedestrians are visible near the parked cars. " + "Trees line the road on both sides. " * 6 +
           "The video shows a completely black screen at the end.")
    ex = A.excerpt(cap)
    assert len(ex) <= A.EXCERPT_MAX
    assert "jacket" not in ex and "man " not in ex.lower()       # a sentence about a person is dropped
    assert "black screen" in ex                                   # the checked claim survives the cut
    assert len(A.excerpt("word " * 200)) <= A.EXCERPT_MAX


def test_clock_quotes_ok():
    a, b = datetime(2026, 9, 2, 21, 40, 5), datetime(2026, 9, 2, 21, 40, 9)
    assert A.clock_quotes_ok(["2026-09-02 21:40:04"], a, b) is True          # within 1 s
    assert A.clock_quotes_ok(["2026-09-02 21:40:30"], a, b) is False
    assert A.clock_quotes_ok(["21:40:07"], a, b) is True
    assert A.clock_quotes_ok([], a, b) is None


def test_yolo_summary_sees_identical_boxes():
    frozen = A.yolo_summary(fixture_segments()[4])
    live = A.yolo_summary(fixture_segments()[0])
    assert frozen["boxes_frozen"] and frozen["static_pairs"] == frozen["pairs"] == 99
    assert not live["boxes_frozen"]
    assert not A.yolo_summary(fixture_segments()[6])["boxes_frozen"]     # no boxes at all is not "frozen boxes"


# ---------------------------------------------------------------- the 12-row fixture
def test_twelve_rows_scores_are_counted():
    rows, events, sc = run_fixture(ours_errors={10: "frozen", 11: "frozen", 12: "frozen"})
    assert len(rows) == 12 and sc["segments"] == 12
    honest = [r["caption_honest"] for r in rows]
    assert honest == [True] * 4 + [False] * 2 + [True] * 6
    assert sc["cosmos_honest"] == 10
    assert sc["ours_correct"] == 11 and not rows[2]["ours_correct"]       # seg 3 tampered: 3 of 5 seconds wrong
    assert sc["ours_seconds_correct"] == 57 and sc["seconds"] == 60
    assert sc["events"] == 4 and sc["events_ours"] == 4 and sc["false_flags_ours"] == 0
    assert sc["events_cosmos"] == 1                                       # only the black segment's ticking clock
    assert sc["splice"] == {"t": 35.0, "ours": True, "cosmos": False}
    assert sc["date_change"]["ours"] and not sc["date_change"]["cosmos"]
    assert any("missed" in n for n in rows[7]["notes"])                  # seg 8 starts on the splice
    assert any("YOLO sidecar" in n for n in rows[4]["notes"])
    assert rows[6]["flags"] == ["clock_runs_on_black"] and rows[4]["flags"] == ["clock_runs_on_frozen"]
    assert all(len(r["caption"]) <= A.EXCERPT_MAX for r in rows)


def test_scores_follow_the_captions():
    caps = list(CAPTIONS)
    caps[4] = "The picture is frozen: the same frame repeats for the whole segment."
    caps[7] = "The video resumes after a gap; a silver pickup truck drives across."
    caps[10] = "The timestamp reads 2026-09-02 21:41:30 on a black and white night street."   # wrong clock quote
    rows, events, sc = run_fixture(caps)
    assert sc["cosmos_honest"] == 10           # +1 frozen admitted, -1 misread clock
    assert rows[4]["caption_honest"] and not rows[10]["caption_honest"]
    assert sc["events_cosmos"] == 3 and sc["splice"]["cosmos"] is True


def test_false_alarm_is_counted():
    truth_sec, truth_clock = A.truth_seconds(SPANS, 60)
    flags = [dict(f) for f in FLAGS] + [{"kind": "clock_jump", "start": 12.0, "end": 12.0, "severity": "alarm",
                                         "text": "bogus"}]
    rows = A.build_rows(fixture_segments(), TRUTH_SEG, truth_sec, truth_clock, flags)
    ev = []
    for e in A.truth_events(SPANS):
        f = A.ours_caught(e, flags)
        ev.append({**e, "ours": f is not None, "ours_flag": f, "cosmos": False})
    assert A.score(rows, ev, flags, truth_sec, truth_sec)["false_flags_ours"] == 1


def test_timing_block():
    res = {"events": [{"t": 0.2, "msg": "uploaded a.mp4"}, {"t": 0.3, "msg": "uploaded b.mp4"},
                      {"t": 578.9, "msg": "poller restarted (poll only)"}],
           "uploads": {"a.mp4": {"upload_s": 0.24, "bytes": 10, "resp": {},
                                 "chunks": {"a": {"seen_any": 1558.2, "seen_complete": 1558.4, "segments_done": 1558.6,
                                                  "total_segments": 12}}},
                       "b.mp4": {"upload_s": 0.1, "bytes": 5, "resp": {},
                                 "chunks": {"b": {"seen_any": 1558.2, "seen_complete": 1558.4, "segments_done": 1560.4,
                                                  "total_segments": 3}}}}}
    t = A.timing(res, "a.mp4", 60.1)
    assert t["indexed_after_s"] == 1558.2 and t["indexed_after_min"] == 26.0 and t["batch"] is True
    assert t["x_realtime"] == 25.9 and t["poller_restarts_s"] == [578.9]


# ---------------------------------------------------------------- clock reader (rendered, monospace grid)
def _render(text: str) -> np.ndarray:
    img = np.full((80, 420), 110, np.uint8)        # the dark overlay box ...
    img[:10] = 235                                  # ... with bright sky above and below it
    img[62:] = 235
    for i, ch in enumerate(text):
        cv2.putText(img, ch, (26 + 20 * i, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    return img


def test_clock_reader_reads_unseen_digits():
    t0 = datetime(2026, 9, 1, 10, 15, 0)
    texts = [(t0 + timedelta(seconds=s)).strftime(A.CLOCK_FMT) for s in range(20)]
    g = A.fit_glyphs([_render(t) for t in texts], texts)
    assert 19.5 < g.pitch < 20.5 and set("0123456789-:") <= set(g.tmpl)
    for want in ("2026-09-02 21:40:05", "2026-09-01 10:15:35", "2027-12-31 23:59:58"):
        r = A.read_clock(_render(want), g)
        assert r["text"] == want, r
    assert A.read_clock(np.full((80, 420), 110, np.uint8), g)["text"] is None    # no text = unreadable, not a guess


# ---------------------------------------------------------------- clock vs picture on frames built in memory
FPS = 4.0


def _texture() -> np.ndarray:
    t = cv2.GaussianBlur(np.random.default_rng(7).random((24, 400)).astype(np.float32), (0, 0), 6)
    return ((t - t.min()) / (t.max() - t.min()) * 200 + 28).astype(np.uint8)


TEX = _texture()


def _scene(k: int, inverted: bool = False) -> np.ndarray:
    """A smooth street-like texture panning 1 px per frame: ~3 grey levels per frame, ~46 after 48 frames."""
    win = TEX[:, k:k + 32].copy()
    return 255 - win if inverted else win


def _clip(splice: bool = True) -> tuple[np.ndarray, list[datetime | None]]:
    frames = [_scene(i) for i in range(32)]                       # 0-8 s live
    frames += [_scene(32)] * 32                                   # 8-16 s frozen on frame 32
    frames += [np.zeros((24, 32), np.uint8)] * 16                 # 16-20 s black
    frames += [_scene(32 + i) for i in range(32)]                 # 20-28 s resumes where it froze
    frames += [_scene(i, inverted=True) for i in range(16)]       # 28-32 s hard cut to another day
    day1, day2 = datetime(2026, 9, 1, 10, 0, 0), datetime(2026, 9, 2, 21, 0, 0)
    reads = [day1 + timedelta(seconds=s) for s in range(28)] + [day2 + timedelta(seconds=s) for s in range(4)]
    return np.stack(frames), reads


def _flags(grey: np.ndarray, reads: list) -> dict:
    keep = np.ones(grey.shape[1:], bool)
    luma = grey.reshape(len(grey), -1).mean(axis=1)
    return A.clock_picture_flags(grey, keep, A.f2f_series(grey, keep), luma, FPS, reads)


def test_clock_vs_picture_finds_all_four():
    grey, reads = _clip()
    out = _flags(grey, reads)
    kinds = [f["kind"] for f in out["flags"]]
    assert kinds == ["clock_runs_on_frozen", "clock_runs_on_black", "splice", "date_change_at_cut"], kinds
    fz, bk, sp, dc = out["flags"]
    assert (fz["start"], fz["end"], fz["advanced_s"]) == (8.0, 16.0, 7.0)
    assert bk["severity"] == "info" and bk["advanced_s"] == 3.0
    assert sp["ref_t"] == 8.0 and sp["start"] == 20.0 and sp["clock_elapsed_s"] == 12.0
    assert sp["picture_diff"] == 0.0 and sp["missing_min_s"] >= 11
    assert dc["start"] == 28.0 and dc["clock_to"].startswith("2026-09-02")
    assert [c["kind"] for c in out["cuts"]] == ["picture_lost", "picture_returns", "hard_cut"]


def test_honest_resume_is_not_a_splice():
    """Picture and clock both skip ahead by the same 12 s: a gap in recording, but no lie between the two."""
    grey, reads = _clip()
    grey = grey.copy()
    for i in range(32):
        grey[80 + i] = _scene(32 + 48 + i)       # the scene moved on during the 12 s blind stretch
    kinds = [f["kind"] for f in _flags(grey, reads)["flags"]]
    assert "splice" not in kinds and "resume_consistent" in kinds


def test_steady_clip_has_no_alarms_and_a_clock_jump_is_caught():
    frames = np.stack([_scene(i) for i in range(64)])
    t0 = datetime(2026, 9, 1, 10, 0, 0)
    reads = [t0 + timedelta(seconds=s) for s in range(16)]
    assert _flags(frames, reads)["flags"] == []
    reads[10:] = [r + timedelta(seconds=5) for r in reads[10:]]
    fl = _flags(frames, reads)["flags"]
    assert [f["kind"] for f in fl] == ["clock_jump"] and fl[0]["jump_s"] == 5.0


# ---------------------------------------------------------------- the real output, when it has been built
def test_written_audit_is_self_consistent():
    p = A.OUT / "audit.json"
    if not p.exists():
        pytest.skip("run python -m stm.audit first")
    doc = json.loads(p.read_text(encoding="utf-8"))
    for key in ("clip", "baseline"):
        c = doc[key]
        rows, sc = c["segments"], c["scores"]
        assert sc["cosmos_honest"] == sum(r["caption_honest"] for r in rows)
        assert sc["ours_correct"] == sum(r["ours_correct"] for r in rows)
        assert sc["events_ours"] == sum(e["ours"] for e in c["events"])
        assert all(len(r["caption"]) <= A.EXCERPT_MAX for r in rows)
        assert all((A.OUT / r["thumb"]).exists() for r in rows)
    assert len(doc["clip"]["segments"]) == 12
