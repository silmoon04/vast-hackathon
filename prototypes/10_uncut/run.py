"""UNCUT CLI.  python run.py --all | --video NAME_OR_PATH [--no-fullscan] [--rebuild-index]

NAME is a test clip from tests/ (t1_..., x7_...) or the path of any mp4 to verify against the archive."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_tests as mt  # noqa: E402
import render  # noqa: E402
import uncut as U  # noqa: E402
import verify as V  # noqa: E402
from uncut import pk  # noqa: E402

OUT = HERE / "out"


def evaluate(gt: dict, r: dict) -> dict:
    """Compare a verdict with the ground truth declared in make_tests.GT."""
    ev = {"verdict_ok": r["verdict"] == gt["verdict"]}
    b = r.get("best")
    if gt.get("source") and b and r["verdict"] in ("MATCH", "PARTIAL"):
        segs = b["segments"]
        ev["source_ok"] = b["video"] == gt["source"]
        ev["start_err"] = abs(segs[0]["src_start"] - gt["src_start"])
        ev["end_err"] = abs(segs[-1]["src_end"] - gt["src_end"])
        ev["offset_ok"] = bool(ev["source_ok"] and max(ev["start_err"], ev["end_err"]) <= 1.0)
        ev["speed_err_pct"] = abs(b["speed"] - gt["speed"]) / gt["speed"] * 100
        ev["mirror_ok"] = b["mirrored"] == gt["mirrored"]
        if gt.get("variant"):
            ev["crop_ok"] = b["variant"] == gt["variant"]
        else:
            ev["crop_ok"] = (abs(b["crop"] - gt["crop"]) < 0.01 and b.get("crop_kind") != "p916") if isinstance(gt["crop"], float) else None
        gr = gt["removed"]
        mr = [x for x in b["removed"] if x["seconds"] > 0]
        ev["removed_ok"] = len(gr) == len(mr)
        if gr and len(mr) == len(gr):
            ev["removed_err_s"] = abs(mr[0]["seconds"] - gr[0][1])
            ev["removed_pos_err_s"] = abs(mr[0]["at_src"] - gr[0][0])
            ev["removed_clip_pos_err_s"] = abs(mr[0]["at_q"] - (gr[0][0] - gt["src_start"]) / gt["speed"])
        gf = gt["foreign"]
        ev["foreign_ok"] = len(gf) == len(b["foreign"])
        if gf and len(b["foreign"]) == len(gf):
            ev["foreign_err_s"] = max(abs(b["foreign"][0]["q_start"] - gf[0][0]), abs(b["foreign"][0]["q_end"] - gf[0][1]))
    elif gt.get("source"):
        ev["offset_ok"] = False
    else:
        ev["offset_ok"] = None      # nothing to locate
    return ev


def clean(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def process(arc, key: str, clip: Path, log=print, fullscan=True) -> dict:
    gt = mt.GT.get(key)
    r = V.verify(arc, clip, key)
    od = OUT / key
    od.mkdir(parents=True, exist_ok=True)
    hero = render.hero(r, clip, od / "hero.jpg", len(arc.seg), len(arc.names))
    plot = render.alignment_plot(r, od / "alignment.png")
    strip = render.filmstrip(r, clip, od / "filmstrip.jpg")
    shutil.copy(clip, od / "query.mp4")
    rep_path = render.report(r, gt, clip, od, hero, plot, strip)
    # embed the playable query into the report (inserted after the hero by simple string edit)
    html = rep_path.read_text(encoding="utf-8")
    html = html.replace("<h2>Ground truth", '<h2>The clip as received</h2><figure><video src="query.mp4" controls muted playsinline '
                        'preload="metadata"></video><figcaption>Query clip (local copy, never uploaded)</figcaption></figure><h2>Ground truth', 1) \
        if "<h2>Ground truth" in html else html
    rep_path.write_text(html, encoding="utf-8")
    res = {**clean(r), "headline": V.headline(r), "ground_truth": gt, "eval": evaluate(gt, r) if gt else None}
    if gt:
        res["ground_truth"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in gt.items()}
    log(f"  {key:<26} {r['verdict']:<15} {V.headline(r)}   [{r['seconds']:.0f}s]")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="the 5 kill-test clips + stress extras")
    ap.add_argument("--kill-only", action="store_true", help="with --all: only the 5 kill-test clips")
    ap.add_argument("--video", help="test clip name (t1_lisbon_crop80 ...) or any mp4 path")
    ap.add_argument("--rebuild-index", action="store_true", help="delete the cached archive index first (about 12 min)")
    args = ap.parse_args()
    if not args.all and not args.video:
        ap.print_help()
        return 2
    t0 = time.time()
    if args.rebuild_index:
        for f in ("index_vec.npy", "index_seg.json"):
            (U.WORK / f).unlink(missing_ok=True)
    tests = mt.make()
    arc = U.Archive.open()
    print(f"[uncut] archive: {len(arc.seg)} five-second segments, {len(arc.names)} videos, "
          f"{sum(len(h['1.0']) for h in arc.hash.values())} hashed frames x {len(U.VARIANTS)} variants", flush=True)
    jobs: list[tuple[str, Path]] = []
    if args.all:
        keys = mt.KILL if args.kill_only else mt.KILL + mt.EXTRAS
        jobs += [(k, tests[k]) for k in keys]
    if args.video:
        if args.video in tests:
            jobs.append((args.video, tests[args.video]))
        elif Path(args.video).exists():
            jobs.append((Path(args.video).stem, Path(args.video)))
        else:
            print(f"unknown clip {args.video!r}; test clips: {', '.join(tests)}")
            return 2
    results = {}
    for key, clip in jobs:
        results[key] = process(arc, key, clip)
    kill = [results[k] for k in mt.KILL if k in results]
    summary = {}
    if len(kill) == len(mt.KILL):
        ok_v = sum(r["eval"]["verdict_ok"] for r in kill)
        located = [r for r in kill if r["ground_truth"].get("source")]
        ok_o = sum(bool(r["eval"].get("offset_ok")) for r in located)
        summary["kill_test"] = {"criterion": "5/5 verdicts right and offsets within 1 s", "verdicts_right": f"{ok_v}/5",
                                "offsets_within_1s": f"{ok_o}/{len(located)}", "passed": ok_v == 5 and ok_o == len(located)}
    ex = [results[k] for k in mt.EXTRAS if k in results]
    if ex:
        summary["extras"] = {"verdicts_right": f"{sum(r['eval']['verdict_ok'] for r in ex)}/{len(ex)}",
                             "offsets_within_1s": f"{sum(bool(r['eval'].get('offset_ok')) for r in ex if r['ground_truth'].get('source'))}/"
                                                  f"{sum(1 for r in ex if r['ground_truth'].get('source'))}"}
    summary["archive"] = {"videos": len(arc.names), "segments": len(arc.seg), "holdout": U.HOLDOUT,
                          "hashed_frames": int(sum(len(h['1.0']) for h in arc.hash.values()))}
    summary["wall_seconds"] = round(time.time() - t0, 1)
    out = {"summary": summary, "clips": results}
    path = HERE / "results.json" if args.all else OUT / f"results_{jobs[-1][0]}.json"
    pk.write_results(path, out)
    print(json.dumps(summary, indent=1, default=str))
    print(f"[uncut] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
