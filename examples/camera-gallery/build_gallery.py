"""Validate a local camera-gallery catalog before serving it."""
import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlparse


def load_catalog(catalog_path: Path, media_root: Path) -> list[dict]:
    root = media_root.resolve(strict=True)
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("catalog must be a JSON array")
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each catalog item must be an object")
        for field in ("id", "group", "group_name", "label", "file"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"each item needs a non-empty {field}")
        if item["id"] in seen:
            raise ValueError(f"duplicate clip id: {item['id']}")
        seen.add(item["id"])
        for field in ("duration", "width", "height", "preview_start", "preview_end"):
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field} must be a finite number: {item['id']}")
        if item["duration"] <= 0 or item["width"] <= 0 or item["height"] <= 0:
            raise ValueError(f"duration and dimensions must be positive: {item['id']}")
        if not (0 <= item["preview_start"] < item["preview_end"] <= item["duration"]):
            raise ValueError(f"preview interval must fall within duration: {item['id']}")
        if "synthetic" in item and not isinstance(item["synthetic"], bool):
            raise ValueError(f"synthetic must be a boolean: {item['id']}")
        for field in ("source_url", "license_url"):
            value = item.get(field)
            if value is not None and (not isinstance(value, str) or urlparse(value).scheme.lower() not in ("http", "https")):
                raise ValueError(f"{field} must use http or https: {item['id']}")
        rel = Path(item["file"])
        if rel.is_absolute():
            raise ValueError(f"clip path must be relative: {item['id']}")
        try:
            resolved = (root / rel).resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            raise ValueError(f"clip path is missing or outside media root: {item['id']}")
        if not resolved.is_file():
            raise ValueError(f"clip is not a file: {item['id']}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("catalog.example.json"))
    parser.add_argument("--root", type=Path, required=True, help="directory containing local media")
    args = parser.parse_args()
    items = load_catalog(args.catalog, args.root)
    print(f"Valid catalog: {len(items)} clips in {len({i['group'] for i in items})} scenes")


if __name__ == "__main__":
    main()
