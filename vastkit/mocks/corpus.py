"""Mock corpus for the VSS mock: rows shaped like vss-collection (minus the vector columns).

A corpus is a JSON list of segment rows. ``default_corpus()`` builds a small synthetic one
so the mock works with zero setup; ``scripts/build_mock_corpus.py`` builds one from local
videos (captions from the local Cosmos3-Edge mirror if it is running).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from vastkit.naming import segment_key, sidecar_key
except ImportError:  # run as a script without the package on sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (vastkit/mocks/ -> ../..)
    from vastkit.naming import segment_key, sidecar_key

SEGMENT_SEC = 5.0

# Deliberately bland placeholder captions: enough to exercise search/explore/stats shapes.
_SYNTH = {
    "sample_a.mp4": ("general", "site-a", "cam-01", [
        "A person walks from the left side of the frame to the right.",
        "Two people stand near a table and talk.",
        "A person picks up a bag from the table and leaves the frame.",
        "The scene is empty; a chair is visible in the background.",
        "A person enters carrying a box and places it on the table.",
        "Two people sit down at the table.",
    ], ["person", "chair", "handbag"]),
    "sample_b.mp4": ("general", "site-b", "cam-02", [
        "A car drives slowly along the road from right to left.",
        "A person on a bicycle passes the parked cars.",
        "A van stops at the side of the road.",
        "Several cars pass in both directions.",
        "The road is quiet; a person walks along the pavement.",
    ], ["car", "person", "bicycle", "truck"]),
    "sample_c.mp4": ("general", "site-c", "cam-03", [
        "A person opens a door and walks into a room.",
        "A person sits at a desk and uses a laptop.",
        "A second person enters and hands over a cup.",
        "Both people look at the laptop screen.",
        "The room is empty; the laptop remains on the desk.",
    ], ["person", "laptop", "cup"]),
}


def _pk(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()


def make_rows(filename: str, captions: list[str], *, team: str, capture_type: str = "general",
              location: str = "", camera_id: str = "", classes: list[str] | None = None,
              local_path: str | None = None, uploaded: datetime | None = None,
              is_public: bool = True, segment_sec: float = SEGMENT_SEC,
              chunks_bucket: str | None = None, segments_bucket: str | None = None) -> list[dict[str, Any]]:
    chunks_bucket = chunks_bucket or f"{team}-vss-chunks"
    segments_bucket = segments_bucket or f"{team}-vss-chunks-segments"
    uploaded = uploaded or (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=3))
    parent = f"s3://{chunks_bucket}/{team}/{uploaded:%Y%m%d_%H%M%S}_{_pk(filename)[:8]}{Path(filename).suffix}"
    classes = classes or []
    rows = []
    for i, cap in enumerate(captions, start=1):
        start, end = (i - 1) * segment_sec, i * segment_sec
        seg_key = segment_key(Path(filename).name, i, len(captions))
        source = f"s3://{segments_bucket}/{seg_key}"
        seen = [c for j, c in enumerate(classes) if (i + j) % 2 == 0 or c in cap.lower()] or classes[:1]
        counts = {c: 1 + (i + k) % 3 for k, c in enumerate(seen)}
        rows.append({
            "pk": _pk(source), "source": source, "filename": seg_key.rsplit("/", 1)[-1],
            "segment_number": i, "segment_start_sec": start, "segment_end_sec": end,
            "reasoning_content": cap, "object_classes": ",".join(sorted(seen)),
            "object_counts": json.dumps(counts), "max_detection_conf": 0.62 + 0.03 * (i % 10),
            # default perception source: blueprint scripts/vss-blueprint-models/yolo-infer/main.py PERCEPTION_SOURCE
            "perception_ok": bool(seen), "perception_source": "yolo11_coco",
            "detection_sidecar_uri": f"s3://{segments_bucket}/{sidecar_key(seg_key)}",
            "detection_frame_count": int(segment_sec * 30), "detection_count": sum(counts.values()) * 30,
            "cosmos_model": "nvidia/cosmos3-reason", "embedding_model": "nvidia/cosmos-embed1",
            "visual_embedding_model": "nvidia/cosmos-embed1", "tokens_used": 900 + 7 * i,
            "cached_prompt_tokens": 0, "processing_time": 4.2,
            "timestamp": (uploaded + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "allowed_users": [team], "is_public": is_public,
            "upload_timestamp": uploaded.isoformat(), "duration": segment_sec,
            "total_segments": len(captions), "original_video": parent, "tags": [capture_type],
            "camera_id": camera_id, "capture_type": capture_type, "location": location,
            "extra_metadata": json.dumps({"status": "ok", "embedding_dimensions": 256,
                                          "visual_embedding_dimensions": 256,
                                          "visual_embedding_ok": True}),
            "_local_path": local_path,
        })
    return rows


def default_corpus(team: str = "team-mock") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None, microsecond=0)
    for n, (fname, (ctype, loc, cam, caps, classes)) in enumerate(_SYNTH.items()):
        rows += make_rows(fname, caps, team=team, capture_type=ctype, location=loc, camera_id=cam,
                          classes=classes, uploaded=base + timedelta(minutes=7 * n))
    return rows


def load_corpus(path: str | None, team: str) -> list[dict[str, Any]]:
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return default_corpus(team)
