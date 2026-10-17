"""stm.seams on a 3-chunk fake day (offline: no video, no network)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from stm import seams

DAY = "20260901"
A, B, C = (f"20261001_07233{i}_neighborhood_{DAY}_chunk_000{i}" for i in range(3))
DUR = {A: 35.0, B: 40.0, C: 20.0}
STRIDES = {A: 25.2, B: None}          # A -> B measured, B -> C not
REASONS = {B: "static or frozen picture (0 of 6 frames match distinctly)"}
FPS = 20.0


def _segs(chunk, upto, step=5.0):
    n = int(np.ceil(upto / step))
    return [{"chunk": chunk, "filename": f"{chunk}_segment_{i + 1:03d}_of_{n:03d}.mp4", "segment_number": i + 1,
             "segment_start_sec": i * step, "segment_end_sec": min(upto, (i + 1) * step)} for i in range(n)]


SEGS = _segs(A, 30.0) + _segs(B, 30.0) + _segs(C, 15.0)
COV = seams.coverage(SEGS)
CHUNKS = [{"chunk": k, "duration": v} for k, v in DUR.items()]


@pytest.fixture
def p():
    return seams.plan(list(reversed(CHUNKS)), COV, STRIDES, REASONS, fps=FPS)


def test_keep_end_offsets_and_totals(p):
    rows = p["rows"]
    assert [r["chunk"] for r in rows] == [A, B, C]                      # recording order, whatever the input order
    assert [r["keep_end"] for r in rows] == [25.2, 30.0, 15.0]
    assert [r["keep_by"] for r in rows] == ["stride", "covered", "covered"]
    assert [r["offset"] for r in rows] == pytest.approx([0.0, 25.2, 55.2])
    assert seams.offsets(p) == {A: 0.0, B: pytest.approx(25.2), C: pytest.approx(55.2)}
    t = p["totals"]
    assert t["raw_s"] == 95.0 and t["kept_s"] == pytest.approx(70.2)
    assert t["removed_s"] == pytest.approx(24.8)
    assert t["removed_uncovered_s"] == pytest.approx(5.0 + 10.0 + 5.0)  # tails with no sidecars
    assert t["removed_overlap_s"] == pytest.approx(4.8)                 # A's 25.2-30 s repeats as B's first 4.8 s
    assert t["indexed_s"] == pytest.approx(t["kept_s"])                 # every kept second has sidecars
    assert seams.uncovered(p, COV) == []
    assert p["days"] == [[DAY, "1 Sep", 0.0, None]]


def test_unmeasured_seams_are_listed(p):
    assert [(s["a"], s["b"], s["reason"]) for s in p["unmeasured"]] == [(B, C, REASONS[B])]
    assert p["totals"]["seams"] == 2 and p["totals"]["measured_seams"] == 1
    assert "unmeasured" in seams.summary(p)


def test_keep_end_rules():
    assert seams.keep_end(35.0, 30.0, 33.0, FPS) == (30.0, "covered")    # stride past the covered end
    assert seams.keep_end(35.0, 30.0, 25.23, FPS) == (25.2, "stride")    # snapped down to the 20 fps grid
    assert seams.keep_end(6.352941, 10.0, None, FPS) == (6.35, "duration")
    assert seams.keep_end(35.0, 30.0, None) == (30.0, "covered")


def test_day_change_is_not_a_seam():
    d2 = f"20261001_073619_neighborhood_20260902_chunk_0000"
    order = seams.order_chunks([d2, C, A, B, A])
    assert order == [A, B, C, d2]
    assert seams.same_day_seams(order) == [(A, B), (B, C)]
    cov = seams.coverage(SEGS + _segs(d2, 30.0))
    p2 = seams.plan(CHUNKS + [{"chunk": d2, "duration": 30.1}], cov, dict(STRIDES, **{C: 12.0}), fps=FPS)
    assert p2["rows"][2]["keep_end"] == 15.0                            # C is its day's last chunk: no stride
    assert p2["days"] == [[DAY, "1 Sep", 0.0, 70.2], ["20260902", "2 Sep", 70.2, None]]
    with pytest.raises(ValueError):
        seams.chunk_key("not_a_chunk.mp4")


def test_encoded_lengths_move_the_offsets():
    p2 = seams.plan(CHUNKS, COV, STRIDES, REASONS, fps=FPS, actual={A: 25.25})
    assert p2["rows"][0]["keep_end"] == 25.25 and p2["rows"][0]["planned_keep_end"] == 25.2
    assert [r["offset"] for r in p2["rows"]] == pytest.approx([0.0, 25.25, 55.25])


def test_internal_gap_is_reported_on_the_new_timeline():
    segs = [s for s in SEGS if not (s["chunk"] == B and s["segment_number"] == 3)]   # B loses 10-15 s
    cov = seams.coverage(segs)
    p2 = seams.plan(CHUNKS, cov, STRIDES, REASONS, fps=FPS)
    assert seams.uncovered(p2, cov) == [[35.2, 40.2]]
    assert p2["totals"]["indexed_s"] == pytest.approx(65.2)


def test_removed_spans_on_the_old_timeline(p):
    old = {A: 0.0, B: 35.0, C: 75.0}
    assert seams.removed_spans(p, old) == [[25.2, 35.0], [65.0, 75.0], [90.0, 95.0]]


def _side(start, n=100, fps=FPS):
    return {"chunk": A, "segment_start_sec": start, "segment_end_sec": start + n / fps, "fps": fps, "frame_count": n,
            "detection_count": n, "frames": [{"frame_index": i, "time_sec": round(i / fps, 3),
                                              "detections": [{"label": "car", "confidence": 0.9,
                                                              "bbox": [10, 10, 50, 40]}]} for i in range(n)]}


def test_trim_sidecar():
    side = _side(25.0)
    t = seams.trim_sidecar(side, 25.2)
    assert [f["time_sec"] for f in t["frames"]] == [0.0, 0.05, 0.1, 0.15]   # frames at/after keep_end are gone
    assert t["frame_count"] == 4 and t["detection_count"] == 4
    assert t["segment_end_sec"] == 25.2 and t["trimmed_at"] == 25.2
    assert side["frame_count"] == 100 and len(side["frames"]) == 100     # input untouched
    assert seams.trim_sidecar(side, 30.0) is side
    assert seams.trim_sidecar(_side(30.0), 30.0) is None


def test_trim_segments(p):
    keep = {r["chunk"]: r["keep_end"] for r in p["rows"]}
    kept, dropped = seams.trim_segments(SEGS, keep)
    assert [(s["chunk"], s["segment_number"]) for s in dropped] == [(A, 6)]   # 25-30 s keeps only 0.2 s
    assert len(kept) == len(SEGS) - 1 and not any("segment_end_sec_orig" in s for s in kept)
    kept2, _ = seams.trim_segments(SEGS, dict(keep, **{A: 27.0}))
    clipped = [s for s in kept2 if "segment_end_sec_orig" in s]
    assert [(s["segment_start_sec"], s["segment_end_sec"], s["segment_end_sec_orig"], s["duration"])
            for s in clipped] == [(25.0, 27.0, 30.0, 2.0)]


def _frame(n):
    """160x90 grey frame number n of a busy street: every frame differs from every other one."""
    return (80 + 30 * np.random.default_rng(n).standard_normal((90, 160))).astype(np.float32)


def test_frame_match_finds_the_overlap():
    ns = range(int(12 * FPS), int(35 * FPS))                            # chunk A from 12 s to its end at 35 s
    ta, fa = [n / FPS for n in ns], np.stack([_frame(n) for n in ns])
    tb = [0.3, 0.8, 1.3, 1.8, 2.5, 3.5, 12.0]                           # 12 s into B lies past A's end
    matches = []
    for t in tb:
        n = int(round((25.2 + t) * FPS))                                # B's frame t was A's frame 25.2 + t
        q = _frame(n) + np.random.default_rng(1000 + n).normal(0, 0.1, (90, 160)).astype(np.float32)
        hit, cost, runner_up = seams.best_match(q, fa, ta)
        matches.append((hit, t, cost, runner_up))
    assert sum(m[2] <= seams.MAX_COST for m in matches) == len(tb) - 1  # only the frame past A's end fails
    assert seams.stride_from_matches(matches, 35.0) == (25.2, "clean")
    frozen = [(12.0 + i, t, 0.0, 0.01) for i, t in enumerate(tb)]      # identical frames everywhere: no stride
    assert seams.stride_from_matches(frozen, 35.0)[0] is None


def test_stride_rules():
    ok = [(25.5, 0.3, 0.0, 5.0), (25.9, 0.8, 0.0, 4.0), (26.4, 1.3, 0.01, 3.0)]
    assert seams.stride_from_matches(ok, 35.0) == (25.1, "clean")
    assert seams.stride_from_matches([], 35.0) == (None, "no frames")
    assert seams.stride_from_matches(ok[:1], 35.0)[0] is None                           # one match is not enough
    assert seams.stride_from_matches([(x[0], x[1], 2.0, 3.0) for x in ok], 35.0)[0] is None   # not distinct
    assert seams.stride_from_matches([(x[0], x[1], 0.25, 4.5) for x in ok], 35.0)[0] is None  # look-alike, not exact
    split = [(25.5, 0.3, 0.0, 5.0), (31.9, 0.8, 0.0, 4.0), (20.4, 1.3, 0.0, 3.0)]
    assert "disagree" in seams.stride_from_matches(split, 35.0)[1]
    assert "outside" in seams.stride_from_matches(ok, 20.0)[1]                         # stride past chunk A's end


def test_cli_replans_a_stored_build(tmp_path, p, capsys):
    rec = {"inputs": {"chunks": CHUNKS, "coverage": COV, "strides": STRIDES, "reasons": REASONS, "fps": FPS},
           "plan": p}
    path = tmp_path / "seams.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert seams.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "stored plan matches" in out and "kept 70.20 s" in out
    rec["plan"]["rows"][1]["offset"] = 30.0
    path.write_text(json.dumps(rec), encoding="utf-8")
    assert seams.main([str(path)]) == 1


def test_build_script_trims_and_shifts_sidecars(p):
    script = Path(__file__).resolve().parent.parent / "scripts" / "build_packd_v2.py"
    spec = importlib.util.spec_from_file_location("build_packd_v2", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                         # import has no side effects
    keep = {r["chunk"]: r["keep_end"] for r in p["rows"]}
    sides = [dict(_side(0.0), chunk=B), dict(_side(25.0), chunk=A), dict(_side(25.0), chunk=B)]
    rows, st = mod.sidecar_rows(sides, keep, seams.offsets(p))
    assert st == {"in": 3, "kept": 2, "trimmed": 1, "dropped": 0, "frames_dropped": 96}
    assert [round(r["segment_start_sec"], 3) for r in rows] == [25.2, 25.0, 50.2]      # B starts at 25.2
    assert len(rows[1]["frames"]) == 4
