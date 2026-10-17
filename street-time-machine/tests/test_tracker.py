import numpy as np
import pandas as pd
import pytest

from stm import data, tracker


def _rows(spec):
    """spec: list of (f, name, x1, y1, x2, y2) -> raw detections frame at 15 fps."""
    recs = [{"f": f, "t": f / 15.0, "cls": {"car": 2, "truck": 7, "person": 0}[n], "name": n, "conf": 0.8,
             "x1": x1, "y1": y1, "x2": x2, "y2": y2} for f, n, x1, y1, x2, y2 in spec]
    return pd.DataFrame(recs)


def test_two_movers_stay_separate():
    spec = []
    for f in range(30):
        spec.append((f, "car", 10 + 5 * f, 100, 60 + 5 * f, 140))      # left to right
        spec.append((f, "car", 400 - 5 * f, 300, 450 - 5 * f, 340))    # right to left, other lane
    out = tracker.track(_rows(spec))
    assert out.tid.nunique() == 2
    assert (out.groupby("tid").y1.nunique() == 1).all()


def test_gap_within_buffer_keeps_id_longer_gap_splits():
    short = [(f, "car", 100, 100, 160, 140) for f in list(range(10)) + list(range(18, 28))]   # 0.53 s gap
    assert tracker.track(_rows(short)).tid.nunique() == 1
    long = [(f, "car", 100, 100, 160, 140) for f in list(range(10)) + list(range(50, 60))]    # 2.7 s gap
    assert tracker.track(_rows(long)).tid.nunique() == 2


def test_class_groups():
    flip = [(f, "car" if f % 2 else "truck", 100, 100, 160, 140) for f in range(20)]
    assert tracker.track(_rows(flip)).tid.nunique() == 1          # car/truck flicker is one vehicle
    assert tracker.track(_rows(flip), by="cls").tid.nunique() == 2
    mixed = [(f, n, 100, 100, 160, 140) for f in range(20) for n in ("car", "person")]
    assert tracker.track(_rows(mixed)).tid.nunique() == 2         # a person never joins a car track


def test_short_tracks_untracked_and_empty_input():
    out = tracker.track(_rows([(0, "car", 0, 0, 10, 10), (1, "car", 0, 0, 10, 10)]))
    assert (out.tid == -1).all()
    empty = tracker.track(_rows([]).reindex(columns=["f", "t", "cls", "name", "conf", "x1", "y1", "x2", "y2"]))
    assert len(empty) == 0 and "tid" in empty


def test_evaluate_identity_is_perfect():
    ref = data.tracks("person_bicycle_car_detection")[data.COLS]
    r = tracker.evaluate(ref, ref)
    assert r["purity"] == 1.0 and r["fragmentation"] == 1.0


@pytest.mark.parametrize("cam", ["car_detection", "person_bicycle_car_detection"])
def test_retrack_matches_bytetrack(cam):
    r = tracker.eval_cam(cam)
    assert r["purity"] >= 0.85
    assert r["fragmentation"] <= 2.0
    assert 0.3 * r["tracks_ref"] <= r["tracks_ours"] <= 2 * r["tracks_ref"]


def test_retrack_ptz_first_minute():
    ref = data.tracks("ptz_parking")[data.COLS]
    ref = ref[ref.t < 60].copy()
    ours = tracker.track(ref.drop(columns=["tid"]))
    assert len(ours) == len(ref)
    assert set(ours.columns) >= set(data.COLS)
    r = tracker.evaluate(ref, ours)
    assert r["purity"] >= 0.9
    assert r["fragmentation"] <= 1.5
    # the parked cars of the car park should come out as long, stable tracks
    lens = ours[ours.tid >= 0].tid.value_counts()
    assert (lens >= 0.9 * ref.f.nunique()).sum() >= 5


def test_index_labels_kept_for_alignment():
    spec = [(f, "car", 10 + 5 * f, 100, 60 + 5 * f, 140) for f in range(10)]
    spec += [(f, "person", 300, 300, 320, 360) for f in range(10)]
    raw = _rows(spec).sample(frac=1.0, random_state=0)            # shuffled, non-contiguous labels
    raw.index = raw.index * 7 + 3
    out = tracker.track(raw)
    back = raw.assign(tid=out.tid)                                 # aligns by label
    assert back.groupby("name").tid.nunique().to_dict() == {"car": 1, "person": 1}
    assert back.tid.nunique() == 2 and (back.tid >= 1).all()
