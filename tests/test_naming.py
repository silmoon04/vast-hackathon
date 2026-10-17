from vastkit.mocks.corpus import default_corpus
from vastkit.naming import parse_segment, segment_key, sidecar_key


def test_segment_key_matches_blueprint():
    assert segment_key("clip.mp4", 1, 6) == "segments/clip_segment_001_of_006.mp4"
    assert segment_key("my.clip.MOV", 12, 120) == "segments/my.clip_segment_012_of_120.MOV"
    assert segment_key("noext", 2, 3) == "segments/noext_segment_002_of_003.mp4"


def test_sidecar_key_strips_only_mp4():
    assert sidecar_key("segments/clip_segment_001_of_006.mp4") == "detections/clip_segment_001_of_006.json.gz"
    assert sidecar_key("s3://b/segments/clip_segment_001_of_006.MP4") == "detections/clip_segment_001_of_006.json.gz"
    assert sidecar_key("segments/clip_segment_001_of_006.mov") == "detections/clip_segment_001_of_006.mov.json.gz"
    assert sidecar_key("x.mp4", prefix="/dets") == "dets/x.json.gz"
    assert sidecar_key("x.mp4", prefix="") == sidecar_key("a/x.mp4", prefix="detections/") == "detections/x.json.gz"
    assert sidecar_key("segments/a.mp4.mov") == "detections/a.mp4.mov.json.gz"  # .mp4 only as the suffix


def test_parse_round_trip():
    for name, n, total in (("clip.mp4", 1, 6), ("a_segment_b.mov", 7, 9), ("cam.01.mkv", 100, 120)):
        got = parse_segment("s3://bucket/" + segment_key(name, n, total))
        assert got == {"base": name.rsplit(".", 1)[0], "segment_number": n, "total_segments": total,
                       "ext": name.rsplit(".", 1)[1]}
    assert parse_segment("s3://bucket/originals/clip.mp4") is None


def test_mock_corpus_uses_blueprint_names():
    for r in default_corpus():
        seg = parse_segment(r["source"])
        assert seg and seg["segment_number"] == r["segment_number"] and seg["total_segments"] == r["total_segments"]
        assert r["detection_sidecar_uri"].endswith("/" + sidecar_key(r["source"]))
        assert r["filename"] == r["source"].rsplit("/", 1)[-1]
