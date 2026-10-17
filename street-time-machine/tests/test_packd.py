import gzip
import json

import numpy as np
import pytest

from stm import data, packd

CAM = "fake_packd"
SPLIT = 15.0


def _sidecar(g, start, fps):
    frames = []
    for f, fr in g.groupby("f"):
        dets = [{"label": r.name, "confidence": round(float(r.conf), 4),
                 "bbox": [int(round(r.x1)), int(round(r.y1)), int(round(r.x2)), int(round(r.y2))]}
                for r in fr.itertuples()]
        frames.append({"frame_index": int(f - round(start * fps)), "time_sec": round(float(fr.t.iat[0]) - start, 3),
                       "shape": [432, 768], "detections": dets})
    n = sum(len(fr["detections"]) for fr in frames)
    return {"source": "yolo11_coco", "segment_source": f"s3://b/segments/car-detection_segment_00{1 + int(start > 0)}_of_002.mp4",
            "video_shape": [432, 768], "fps": fps, "frame_count": len(frames), "detection_count": n,
            "max_detection_conf": max((d["confidence"] for fr in frames for d in fr["detections"]), default=0.0),
            "frames": frames}


@pytest.fixture
def fixture_rows():
    src = data.tracks("car_detection")[data.COLS]
    fps = data.info("car_detection")["fps"]
    a, b = src[src.t < SPLIT], src[src.t >= SPLIT]
    seg1 = dict(_sidecar(a, 0.0, fps), segment_start_sec=0.0)                 # bare sidecar carrying its start
    seg2 = {"segment_number": 2, "segment_start_sec": SPLIT, "segment_end_sec": 30.16,
            "sidecar": _sidecar(b, SPLIT, fps)}                               # VastDB-row style
    return src, [seg2, seg1]                                                  # out of order on purpose


@pytest.fixture
def registered(fixture_rows):
    if not data.CAMS["car_detection"]["video"].exists():
        pytest.skip("car_detection video not available")
    src, rows = fixture_rows
    path = packd.register(CAM, data.CAMS["car_detection"]["video"], rows, title="Fake pack D")
    try:
        yield src, path
    finally:
        packd.unregister(CAM)
        assert not path.exists()
        assert CAM not in data.CAMS


def test_cls_ids():
    assert [packd.cls_id(n) for n in ("person", "bicycle", "car", "motorcycle", "bus", "truck")] == [0, 1, 2, 3, 5, 7]
    assert packd.cls_id("toothbrush") == 79 and packd.cls_id("Traffic Light") == 9
    assert packd.cls_id("unicorn") == -1


def test_register_and_objects(registered):
    src, path = registered
    raw = json.load(gzip.open(path, "rt", encoding="utf-8"))
    assert raw["extra"]["source"] == "stm.packd" and raw["extra"]["retracked"] is True
    assert [s["start"] for s in raw["extra"]["segments"]] == [0.0, SPLIT]
    df = data.tracks(CAM)
    assert len(df) == len(src)
    # times survive the segment offset + merge (time order, rounding to 3 dp)
    assert np.allclose(np.sort(df.t.to_numpy()), np.sort(src.t.to_numpy()), atol=0.002)
    assert df.f.is_monotonic_increasing or df.sort_values("f").t.is_monotonic_increasing
    assert set(df.loc[df.name == "car", "cls"]) == {2}
    assert (df.tid >= 0).any()
    obj = data.objects(CAM)
    assert len(obj) >= 1
    assert {"id", "name", "first_t", "last_t", "n", "parked"} <= set(obj.columns)
    assert "car" in set(obj.name)
    # a track that crosses the segment boundary is one id, not two
    ref_objs = data.objects("car_detection")
    crossing = ref_objs[(ref_objs.first_t < SPLIT - 1) & (ref_objs.last_t > SPLIT + 1)]
    if len(crossing):
        ours = obj[(obj.first_t < SPLIT - 1) & (obj.last_t > SPLIT + 1)]
        assert len(ours) >= 1


def test_chained_starts_without_offsets(fixture_rows):
    _, (seg2, seg1) = fixture_rows
    s1 = {k: v for k, v in seg1.items() if k != "segment_start_sec"}
    s2 = dict(seg2["sidecar"])
    df, meta = packd.sidecars_to_rows([s1, s2], fps=s1["fps"])
    assert [m["how"] for m in meta["segments"]] == ["chained", "chained"]
    assert meta["segments"][0]["start"] == 0.0
    assert meta["segments"][1]["start"] == pytest.approx(s1["frame_count"] / s1["fps"], abs=1e-3)
    assert (df.tid == -1).all() and len(df) == s1["detection_count"] + s2["detection_count"]


def test_s3_refused():
    with pytest.raises(ValueError):
        packd.load_sidecar("s3://bucket/detections/x.json.gz")


def test_boxes_scaled_to_video_size():
    side = {"fps": 30.0, "frame_count": 1, "video_shape": [360, 640],
            "frames": [{"frame_index": 0, "time_sec": 0.0, "shape": [360, 640],
                        "detections": [{"label": "car", "confidence": 0.9, "bbox": [64, 36, 128, 72]}]}]}
    df, meta = packd.sidecars_to_rows([dict(side, segment_start_sec=0.0)], fps=30.0, size=(1280, 720))
    assert df[["x1", "y1", "x2", "y2"]].iloc[0].tolist() == [128.0, 72.0, 256.0, 144.0]
    assert meta["segments"][0]["scaled_from"] == [640, 360]
    same, meta2 = packd.sidecars_to_rows([dict(side, segment_start_sec=0.0)], fps=30.0, size=(640, 360))
    assert same.x2.iat[0] == 128.0 and "scaled_from" not in meta2["segments"][0]


def test_refuses_to_overwrite_foreign_cache():
    hits = sorted(data.TRACK_CACHE.glob("car_detection__det__*.json.gz"))
    if not hits:
        pytest.skip("no car_detection cache")
    before, cam_before = hits[0].read_bytes(), dict(data.CAMS["car_detection"])
    try:
        with pytest.raises(FileExistsError):
            packd.register("car_detection", data.CAMS["car_detection"]["video"], [])
        assert hits[0].read_bytes() == before
    finally:  # restore the real cache even if the guard ever regresses
        if hits[0].read_bytes() != before:
            hits[0].write_bytes(before)
        data.CAMS["car_detection"] = cam_before
        data.tracks.cache_clear()
