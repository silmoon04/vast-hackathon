"""Smoke-test the local model mirrors (or the hosted endpoints) through the same vastkit clients.

    .venv/Scripts/python.exe scripts/smoke_local.py <local-clip.mp4>
    .venv/Scripts/python.exe scripts/smoke_local.py CLIP --hosted     # use COSMOS3_REASON_URL etc. from config

Prints timings and short outputs only (no URLs, no tokens).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vastkit.gpu import CosmosEmbed, CosmosReason, YoloClient  # noqa: E402
from vastkit.gpu.yolo import unique_counts  # noqa: E402
from vastkit.gpu.embed import cosine  # noqa: E402
from vastkit.video import prep_clip, probe  # noqa: E402

LOCAL = {"cosmos": "http://127.0.0.1:8001", "yolo": "http://127.0.0.1:8002", "embed": "http://127.0.0.1:8003"}


def step(name, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        print(f"[ok]   {name:28s} {time.perf_counter() - t0:6.1f}s  {out}")
        return True
    except Exception as e:  # keep going; report every endpoint
        print(f"[FAIL] {name:28s} {time.perf_counter() - t0:6.1f}s  {type(e).__name__}: {str(e)[:300]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--hosted", action="store_true", help="use configured hosted URLs instead of localhost")
    ap.add_argument("--skip", nargs="*", default=[], choices=["cosmos", "yolo", "embed"])
    a = ap.parse_args()
    clip = prep_clip(a.video, duration=8, width=640, fps=4)  # a new temp file (no dst given)
    try:
        return _run(a, clip)
    finally:
        clip.unlink(missing_ok=True)


def _run(a, clip: Path) -> int:
    url = (lambda k: None) if a.hosted else LOCAL.get
    info = probe(clip)
    print(f"clip: {info['width']}x{info['height']} {info['fps']:.1f}fps {info['duration']:.1f}s "
          f"{clip.stat().st_size / 1e3:.0f} kB")
    ok = True
    if "yolo" not in a.skip:
        y = YoloClient(url("yolo"))
        ok &= step("yolo ready", y.ready)
        ok &= step("yolo infer (<=8 s clip)", lambda: (lambda r: {"frames": r["perception_json"]["frame_count"],
                                                                 "raw": r["object_counts"],
                                                                 "unique": unique_counts(r["frames"])})(y.infer(clip)))
    if "embed" not in a.skip:
        e = CosmosEmbed(url("embed"))
        ok &= step("embed ready", e.ready)
        state = {}

        def emb():  # shape check only: two neutral texts, raw cosines, no relevance judgement
            state["v"] = e.video(clip)
            state["t"] = e.texts(["a moving object", "a blank screen"])
            return {"dim": len(state["v"]), "text_dim": len(state["t"][0]),
                    "cosines": [round(cosine(t, state["v"]), 3) for t in state["t"]]}
        ok &= step("embed video + 2 texts", emb)
    if "cosmos" not in a.skip:
        c = CosmosReason(url("cosmos"), cache=False)
        ok &= step("cosmos ready", lambda: (c.ready(), c.model))
        ok &= step("cosmos caption", lambda: c.ask_video(clip, "Describe what happens in this clip in 2 sentences.",
                                                         fps=2, max_tokens=160).answer[:240])
        ok &= step("cosmos events (json)", lambda: json.dumps(c.events(clip, fps=2, max_tokens=400))[:300])
        ok &= step("cosmos text-only", lambda: c.ask_text("Reply with the single word: ready", max_tokens=8).answer)
        ok &= step("cosmos fps+num_frames -> 400", lambda: _expect_400(c, clip))
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


def _expect_400(c, clip):
    from vastkit.http import APIError
    body = {"model": c.model, "max_tokens": 4, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "x"}, {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}],
        "media_io_kwargs": {"video": {"fps": 2, "num_frames": 8}}}
    r = c.client.post("/v1/chat/completions", json=body)
    if r.status_code != 400:
        raise APIError(r, "expected 400")
    return "400 as expected"


if __name__ == "__main__":
    raise SystemExit(main())
