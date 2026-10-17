import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_same_filename_uploads_get_distinct_segments(vss, tmp_path, monkeypatch):
    from vastkit.mocks import vss_mock
    monkeypatch.setattr(vss_mock, "UPLOAD_DIR", tmp_path / "uploads")
    keys, mine = [], []
    try:
        for n, payload in enumerate((b"first-take" * 8, b"second-take" * 8)):
            d = tmp_path / f"take{n}"
            d.mkdir()
            clip = d / "clip.mp4"  # same filename both times
            clip.write_bytes(payload)
            keys.append(vss.upload(clip, i_own_this_video=True)["object_key"])
        for key in keys:
            rows = [r for r in vss_mock.PENDING if r["original_video"].endswith("/" + key)]
            assert len(rows) == 4
            stem = Path(key).stem
            assert all(f"/segments/{stem}_segment_" in r["source"] for r in rows)
            mine.append(rows)
        first, second = ({r["source"] for r in rows} for rows in mine)
        assert not first & second
        assert not {r["pk"] for r in mine[0]} & {r["pk"] for r in mine[1]}
        # promote now and check the second upload's segment streams the second file, not the first
        for rows in mine:
            for r in rows:
                vss_mock.PENDING.remove(r)
                vss_mock.ROWS.append(r)
        src = mine[1][0]["source"]
        assert vss.segment(src)["original_video"].endswith("/" + keys[1])
        assert vss.stream(src).content == b"second-take" * 8
    finally:
        for rows in mine:
            for r in rows:
                for pool in (vss_mock.PENDING, vss_mock.ROWS):
                    if r in pool:
                        pool.remove(r)


def test_synthesize_rejects_empty_question(vss):
    import pytest
    from vastkit.http import APIError
    with pytest.raises(APIError) as e:  # the blueprint schema has min_length=1 too
        vss._json("POST", "/videos/synthesize", json={"original_video": "s3://b/v.mp4", "question": ""})
    assert e.value.status_code == 422


def test_vss_mock_runs_as_a_script(tmp_path):
    # run by path, the corpus import fallback must find the repo root from vastkit/mocks/
    r = subprocess.run([sys.executable, str(ROOT / "vastkit" / "mocks" / "vss_mock.py"), "--help"], cwd=tmp_path,
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert "Shape-only local VSS backend mock" in r.stdout
