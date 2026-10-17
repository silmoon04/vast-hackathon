"""S3 key naming used by the VSS blueprint pipeline (vss-blueprint d0e58da).

segmenter: source-code/ingest/video-segmenter/common/handler_utils.py  get_segment_key()
detector:  source-code/ingest/video-detector/common/clients.py         sidecar_key_for_segment()
Only the key formats are mirrored (tests/test_naming.py pins them); the code here is the toolkit's own.
"""
from __future__ import annotations

import re

_SEG = re.compile(r"^(?:.*/)?(?:segments/)?(?P<base>.+)_segment_(?P<n>\d+)_of_(?P<total>\d+)\.(?P<ext>[A-Za-z0-9]+)$")


def segment_key(original_filename: str, segment_number: int, total_segments: int) -> str:
    """'clip.mp4', 1, 6 -> 'segments/clip_segment_001_of_006.mp4' (no extension -> mp4)."""
    parts = original_filename.rsplit(".", 1)
    ext = parts[1] if len(parts) > 1 else "mp4"
    return f"segments/{parts[0]}_segment_{segment_number:03d}_of_{total_segments:03d}.{ext}"


def sidecar_key(segment_key_or_uri: str, prefix: str = "detections/") -> str:
    """Segment key/URI -> 'detections/<name>.json.gz'; only a case-insensitive .mp4 is stripped."""
    name = re.sub(r"(?i)\.mp4\Z", "", segment_key_or_uri.rpartition("/")[2])
    folder = (prefix or "detections/").lstrip("/")
    return f"{folder}{'' if folder.endswith('/') else '/'}{name}.json.gz"


def parse_segment(key_or_uri: str) -> dict | None:
    """'s3://b/segments/clip_segment_002_of_006.mp4' -> {base, segment_number, total_segments, ext}."""
    m = _SEG.match(key_or_uri)
    if not m:
        return None
    return {"base": m["base"], "segment_number": int(m["n"]), "total_segments": int(m["total"]), "ext": m["ext"]}
