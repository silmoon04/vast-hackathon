"""Build a realistic VSS-mock corpus from local videos you own or that are openly licensed.

Each video is cut into 5 s segments (like the DataEngine segmenter), then each segment gets a Cosmos
caption, YOLO unique-object counts and (optionally) Embed1 text + visual vectors, using the local
mirrors by default (python -m vastkit local) or any URLs you pass.

    .venv/Scripts/python.exe scripts/build_mock_corpus.py <folder-of-local-videos> --max-videos 4 \
        --out data/mock_corpus.json --vectors data/mock_vectors.parquet
    then  VSS_MOCK_CORPUS=data/mock_corpus.json python -m vastkit mock

The output stays local (data/ is gitignored): it is only ever served by the mock, never uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
DATA = Path(__file__).resolve().parents[1] / "data"  # the defaults land in the repo's gitignored data/ from any cwd

from vastkit.mocks.corpus import make_rows  # noqa: E402
from vastkit.gpu import CosmosEmbed, CosmosReason, YoloClient  # noqa: E402
from vastkit.gpu.cosmos import PROMPTS  # noqa: E402
from vastkit.gpu.yolo import unique_counts  # noqa: E402
from vastkit.video import probe, segment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=str(DATA / "mock_corpus.json"))
    ap.add_argument("--vectors", help="also write vectors (text + visual) to this parquet for duckdb_standin")
    ap.add_argument("--work", default=str(DATA / "segments"),
                    help="where segment mp4s are written (served by the mock)")
    ap.add_argument("--team", default="team-mock")
    ap.add_argument("--max-videos", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=40.0, help="only the first N seconds of each video")
    ap.add_argument("--cosmos", default="http://127.0.0.1:8001")
    ap.add_argument("--yolo", default="http://127.0.0.1:8002")
    ap.add_argument("--embed", default="http://127.0.0.1:8003")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--prompt", default=PROMPTS["caption"])
    a = ap.parse_args()

    vids = sorted(p for p in Path(a.folder).iterdir() if p.suffix.lower() in (".mp4", ".mov", ".mkv"))[:a.max_videos]
    cos, yolo = CosmosReason(a.cosmos), YoloClient(a.yolo)
    emb = None if a.no_embed else CosmosEmbed(a.embed)
    rows_all, vec_rows = [], []
    base = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None, microsecond=0)
    t0 = time.time()
    for n, v in enumerate(vids):
        info = probe(v)
        work = Path(a.work) / v.stem
        segs = [s for s in segment(v, work, seconds=5.0, width=960) if s["start_sec"] < a.max_seconds]
        caps, classes_per, counts_per, conf_per = [], [], [], []
        for s in segs:
            r = cos.ask_video(s["path"], a.prompt, fps=min(2.0, info["fps"] or 2.0), max_tokens=220)
            caps.append(" ".join((r.answer or "").split()))
            det = yolo.infer(s["path"], include_frames=True)
            uc = unique_counts(det.get("frames") or [])
            classes_per.append(sorted(uc)); counts_per.append(uc); conf_per.append(det.get("max_detection_conf", 0.0))
            print(f"  {v.name} seg {s['segment_number']}: {caps[-1][:90]!r} {uc}", flush=True)
        all_classes = sorted({c for cl in classes_per for c in cl})
        rows = make_rows(v.name, caps, team=a.team, capture_type="general", location="", camera_id=v.stem[:24],
                         classes=all_classes, uploaded=base + timedelta(minutes=7 * n))
        for row, s, cl, uc, mc in zip(rows, segs, classes_per, counts_per, conf_per):
            row.update(object_classes=",".join(cl), object_counts=json.dumps(uc), max_detection_conf=round(mc, 4),
                       perception_ok=bool(cl), detection_frame_count=int(round(5 * (info["fps"] or 30))),
                       cosmos_model=cos.model, segment_end_sec=s["end_sec"],
                       _local_path=str(Path(s["path"]).resolve()))
            if emb:
                vec_rows.append({"source": row["source"], "vectors": emb.text(row["reasoning_content"]),
                                 "vectors_visual": emb.video(s["path"])})
        rows_all += rows
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows_all, indent=1), encoding="utf-8")
    print(f"wrote {len(rows_all)} rows from {len(vids)} videos to {out} in {time.time() - t0:.0f}s")
    if a.vectors and vec_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vastkit.vdb import VECTOR_TYPE
        tbl = pa.Table.from_pylist(vec_rows, schema=pa.schema([("source", pa.utf8()), ("vectors", VECTOR_TYPE),
                                                               ("vectors_visual", VECTOR_TYPE)]))
        Path(a.vectors).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, a.vectors)
        print(f"wrote {len(vec_rows)} vector rows to {a.vectors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
