"""Occupancy correctness at the hosted application's sidecar boundary."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("field_desk_analysis", Path(__file__).resolve().parents[1] / "deploy" / "field-desk" / "analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def frame(index, people=1, **kwargs):
    detections = [{"label": "person", "confidence": .9, "bbox": [10 + i, 10, 30 + i, 80]} for i in range(people)]
    return {"frame_index": index, "shape": [100, 100], "detections": detections, **kwargs}


def side(frames, fps=2):
    return {"fps": fps, "video_shape": [100, 100], "frames": frames}


def test_median_is_half_up_and_empty_frames_are_observed_zeroes():
    result = analysis.build([(0, 2, side([frame(0, 1), frame(1, 2), frame(2, 0), frame(3, 0)]))], 2)
    assert [s["counts"]["person"] for s in result["series"]] == [2, 0]
    assert result["coverage_pct"] == 100


def test_missing_sidecar_time_stays_unknown_and_breaks_an_event():
    result = analysis.build([(0, 1, side([frame(0, 5)])), (2, 3, side([frame(0, 5)]))], 3)
    assert result["series"][1]["counts"] is None
    assert result["coverage_pct"] == 66.7
    assert analysis.threshold_events(result["series"], "person", 5, 2) == []


def test_segment_offsets_and_duplicate_frames_are_handled():
    result = analysis.build([(0, 1, side([frame(0, 1), frame(0, 1)])), (1, 2, side([frame(0, 3)]))], 2)
    assert result["frame_count"] == 2
    assert result["frames"][1]["t"] == 1
    assert [s["counts"]["person"] for s in result["series"]] == [1, 3]


def test_zone_checks_bottom_centre_not_box_centre():
    result = analysis.build([(0, 1, side([frame(0)]))], 1, [[0, .7], [1, .7], [1, 1], [0, 1]])
    assert result["series"][0]["counts"]["person"] == 1
    excluded = analysis.build([(0, 1, side([frame(0)]))], 1, [[.5, .5], [1, .5], [1, 1], [.5, 1]])
    assert excluded["series"][0]["counts"]["person"] == 0


def test_confidence_filter_and_vehicle_group_do_not_double_count():
    f = frame(0, 0)
    f["detections"] = [{"label": "car", "confidence": .9, "bbox": [0, 0, 20, 20]}, {"label": "truck", "confidence": .1, "bbox": [30, 0, 50, 20]}]
    result = analysis.build([(0, 1, side([f]))], 1)
    assert result["series"][0]["counts"]["vehicle"] == 1
    assert result["series"][0]["counts"]["truck"] == 0


def test_threshold_is_literal_and_partial_last_second_is_not_full():
    series = [{"t": i, "end": i + 1, "counts": {"person": n}} for i, n in enumerate([5, 5, 4, 5, 5, 5])]
    events = analysis.threshold_events(series, "person", 5, 3)
    assert len(events) == 1
    assert events[0]["start"] == 3 and events[0]["fire_at"] == 6
    assert analysis.threshold_events([{"t": 0, "end": .3, "counts": {"person": 5}}], "person", 5, 1) == []


def test_question_threshold_range_and_unique_intents():
    result = analysis.build([(0, 5, side([frame(i, 5 if i < 6 else 2) for i in range(10)]))], 5)
    response = analysis.answer(result, "Were at least 5 people visible for 3 seconds?")
    assert response["intent"] == "threshold" and len(response["events"]) == 1
    assert response["events"][0]["start"] == 0
    assert analysis.answer(result, "How many people between 3 and 5 seconds?")["evidence"][0]["count"] == 2
    assert analysis.answer(result, "How many different people?")["intent"] == "unique_unavailable"
    assert analysis.answer(result, "Who owns the car?")["intent"] == "help"


@pytest.mark.parametrize("region", [[[0, 0], [1, 1], [2, 0]], [[0, 0], [.5, .5], [1, 1]], [[0, 0], [1, 1]]])
def test_invalid_regions_are_rejected(region):
    with pytest.raises(ValueError):
        analysis.validate_region(region)


def test_empty_sidecar_is_not_a_zero_count_success():
    with pytest.raises(ValueError, match="no usable"):
        analysis.build([], 30)
