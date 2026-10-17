"""NIGHT EXIT: "who went through the exit since T?" (video time is the clock).

    python run.py --all                                   # default test set, demo query per clip
    python run.py --video retail_cctv --since 00:20 --query "white coat" --dir out
    python run.py --video one_by_one_person_detection --line "Gate:690,0,690,432:300,200:centre"
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nightexit as ne  # noqa: E402
import protokit as pk  # noqa: E402
import render  # noqa: E402

OUT = HERE / "out"          # main() may point this at another folder (--out-dir) so tests do not touch the saved outputs
CLOSE = 0.10                # candidates within this much (0..1 combined score) of the best are "close matches"
GT_FILE = HERE / "gt.json"
TOL = 3.0     # seconds: tolerance when matching a detected crossing to an eyeballed one


def parse_line(spec: str) -> ne.Line:
    """'name:x1,y1,x2,y2:inside_x,inside_y[:foot|centre]'"""
    parts = spec.split(":")
    p = [float(v) for v in parts[1].split(",")]
    ins = [float(v) for v in parts[2].split(",")]
    return ne.Line(parts[0], (p[0], p[1]), (p[2], p[3]), (ins[0], ins[1]), parts[3] if len(parts) > 3 else "foot")


# ------------------------------------------------------------------------------------------------ analysis


def analyse(name: str, *, lines: list[ne.Line] | None = None, use_vlm: bool = True) -> dict[str, Any]:
    """Everything that does not depend on the query: crossings of all persons, crops, colours, descriptions."""
    cfg = ne.CLIPS.get(name, {})
    lines = lines or cfg.get("lines")
    if not lines:
        raise SystemExit(f"no exit line known for {name}: pass --line 'name:x1,y1,x2,y2:inside_x,inside_y'")
    t0 = time.time()
    r, tr = ne.person_rows(name)
    n_tracks = int(r["tid"].nunique())
    r = ne.stitch(r, tr.fps, cfg.get("stitch_gap", 2.0))
    blind, motion = ([], {})
    if cfg.get("motion_guard"):
        blind, motion = ne.camera_blind_spans(name)
    ev = ne.find_crossings(r, lines, blind=blind)
    ev["returned"] = np.nan
    for k, e in ev.iterrows():          # OUT followed by IN of the same person at the same door within 30 s
        if e.dir == "OUT":
            back = ev[(ev.pid == e.pid) & (ev.line == e.line) & (ev.dir == "IN") & (ev.t > e.t) & (ev.t <= e.t + 30)]
            if len(back):
                ev.loc[k, "returned"] = float(back.t.iloc[0] - e.t)
    seg = pk.segments(name, 5.0, tr)                 # the DataEngine table stand-in (5 s segments, YOLO counts)
    t1 = time.time()
    print(f"[night_exit] {name}: {len(ev)} crossings from {n_tracks} tracks ({t1 - t0:.1f} s), computing appearance...", flush=True)
    people = ne.appearance(name, r, tr, ev, OUT / name / "crops", use_vlm=use_vlm) if len(ev) else {}
    return {"name": name, "lines": lines, "r": r, "tr": tr, "ev": ev, "blind": blind, "motion": motion, "people": people, "segments": seg,
            "n_tracks": n_tracks, "n_identities": int(r["pid"].nunique()),
            "t_events": t1 - t0, "t_appearance": time.time() - t1}


def vastdb_shortlist(st: dict[str, Any], since: float) -> dict[str, Any]:
    """The VastDB step on the day: SQL over the DataEngine segment table to shortlist the 5 s segments after T that
    contain people (duckdb stand-in; only the dialect subset the real VastDB accepts: no IN / GROUP BY / JOIN)."""
    seg = st["segments"]
    if "n_person" not in seg.columns:
        return {"sql": None, "segments_total": int(len(seg)), "segments_after_T": 0, "segments_with_people": 0, "all_crossings_inside": True}
    con = pk.vastdb({"segments": seg})
    sql = f'SELECT seg, start, "end", n_person, peak_person FROM segments WHERE "end" > {since:.1f} AND n_person > 0 ORDER BY start'
    short = con.execute(sql).fetchdf()
    after = int(con.execute(f'SELECT count(*) FROM segments WHERE "end" > {since:.1f}').fetchone()[0])
    ids = set(short["seg"].astype(int))
    inside = all(int(e.t // 5) in ids for e in st["ev"].itertuples() if e.t >= since)
    return {"sql": sql, "segments_total": int(len(seg)), "segments_after_T": after, "segments_with_people": int(len(short)),
            "all_crossings_inside": bool(inside)}


def build_cards(st: dict[str, Any], since: float, query: str, direction: str = "any", top: int = 6,
                mode: str = "combined") -> list[dict[str, Any]]:
    ev = st["ev"]
    sel = ev[(ev.t >= since)]
    if direction in ("in", "out"):
        sel = sel[sel.dir == direction.upper()]
    sel = sel[sel.pid.isin(st["people"])]
    cards = []
    for pid, g in sel.groupby("pid"):
        events = [dict(e._asdict()) for e in g.sort_values("t").itertuples(index=False)]
        outs = [e for e in events if e["dir"] == "OUT"]
        primary = outs[0] if outs else events[0]
        p = st["people"][int(pid)]
        ret = primary["returned"] if primary["dir"] == "OUT" and primary["returned"] == primary["returned"] else None
        cards.append({"pid": int(pid), "events": events, "primary": primary, "top": p["top"], "legs": p["legs"],
                      "desc": p["desc"], "crop": p["crop"], "crop_bgr": p["crop_bgr"], "returned": ret, "score": None,
                      "parts": {}, "abs": None, "gap": None, "n_close": 1})
    if query.strip() and cards:
        ranked = ne.rank_people({c["pid"]: st["people"][c["pid"]] for c in cards}, query, mode)
        by = {x["pid"]: x for x in ranked}
        for c in cards:
            c["score"] = by[c["pid"]]["score"]
            c["parts"] = {k: v for k, v in by[c["pid"]].items() if k.startswith("s_")}
        cards.sort(key=lambda c: -c["score"])
        best = cards[0]["score"]
        n_close = sum(1 for c in cards if best - c["score"] <= CLOSE)
        for c in cards:
            c["abs"] = c["parts"].get("s_abs")
            c["gap"] = best - c["score"]
            c["n_close"] = n_close
    else:
        cards.sort(key=lambda c: c["primary"]["t"])
    return cards[:top]


# ------------------------------------------------------------------------------------------ ground truth


def load_gt() -> dict[str, Any]:
    return json.loads(GT_FILE.read_text(encoding="utf-8")) if GT_FILE.exists() else {}


def match_events(pred: pd.DataFrame, truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Match detected crossings to eyeballed ones. Pass 1: same direction and |dt| <= tol (nearest first). Pass 2: leftover
    truth events may match any leftover detection within tol, which then counts as a wrong direction."""
    used: set[int] = set()
    pairs: list[tuple[dict[str, Any], Any]] = []
    done: set[int] = set()
    for same_dir in (True, False):
        for gi, g in sorted(enumerate(truth), key=lambda z: z[1]["t"]):
            if gi in done:
                continue
            best, bd = None, 1e9
            for k, e in pred.iterrows():
                if k in used or (same_dir and e.dir != g["dir"]):
                    continue
                dt = abs(e.t - g["t"])
                if dt <= g.get("tol", TOL) and dt < bd:
                    best, bd = k, dt
            if best is not None:
                used.add(best)
                done.add(gi)
                pairs.append((g, pred.loc[best]))
    tp = len(pairs)
    dir_ok = sum(1 for g, e in pairs if e.dir == g["dir"])
    return {"truth_events": len(truth), "detected_events": int(len(pred)), "matched": tp, "missed": len(truth) - tp,
            "extra": int(len(pred)) - tp, "direction_correct": dir_ok,
            "direction_accuracy": round(dir_ok / tp, 3) if tp else None,
            "recall": round(tp / len(truth), 3) if truth else None,
            "precision": round(tp / len(pred), 3) if len(pred) else None,
            "missed_list": [g for gi, g in enumerate(truth) if gi not in done],
            "extra_list": [{"t": float(e.t), "dir": e.dir} for k, e in pred.iterrows() if k not in used]}


def eval_query(st: dict[str, Any], q: dict[str, Any]) -> dict[str, Any]:
    """Rank of the eyeballed valid person(s) for one description query, in three scoring modes.
    q["targets"]: crossing times (+-tol) of every valid answer; pass = the rank-1 card is a valid answer."""
    since = ne.parse_clock(q["since"])
    targets = q["targets"]
    k_need = int(q.get("need_top_k", 0))
    res: dict[str, Any] = {"since": q["since"], "query": q["query"], "valid_answers": len(targets), "note": q.get("note", "")}

    def valid(c: dict[str, Any]) -> bool:
        return any(abs(e["t"] - t["t"]) <= t.get("tol", TOL) and (not t.get("dir") or e["dir"] == t["dir"])
                   for e in c["events"] for t in targets)

    for mode in ("combined", "colour", "embed"):
        cards = build_cards(st, since, q["query"], "any", 50, mode)
        ranks = [i + 1 for i, c in enumerate(cards) if valid(c)]
        res[mode] = {"close_matches_within_10pts": sum(1 for c in cards if c["score"] is not None and cards[0]["score"] - c["score"] <= CLOSE),
                     "rank_of_first_valid": ranks[0] if ranks else None, "n_candidates": len(cards),
                     "top1": ne.clock(cards[0]["primary"]["t"]) if cards else None,
                     "top1_score": round(cards[0]["score"], 3) if cards and cards[0]["score"] is not None else None,
                     "top_k_all_valid": (len(ranks) >= k_need and ranks[:k_need] == list(range(1, k_need + 1))) if k_need else None}
    res["pass"] = res["combined"]["rank_of_first_valid"] == 1
    res["post_fix"] = bool(q.get("post_fix", False))
    return res


# ------------------------------------------------------------------------------------------------ outputs


def event_frames(st: dict[str, Any], folder: Path) -> list[dict[str, Any]]:
    r, ev = st["r"], st["ev"]
    fr = ne.read_frames(st["name"], list(ev.f))
    out = []
    for k, e in enumerate(ev.itertuples()):
        d = r[(r.pid == e.pid) & ((r.f - e.f).abs() <= 2)]
        box = None
        if len(d):
            row = d.iloc[(d.f - e.f).abs().argmin()]
            box = (row.x1, row.y1, row.x2, row.y2)
        title = f"{ne.clock(e.t)}  {e.dir}  {e.line}  (person #{e.pid})"
        im = render.evidence_frame(fr[e.f], st["lines"], box, e, title)
        im = pk.fit(im, 1280) if im.shape[1] > 1280 else im
        p = pk.save_jpg(im, folder / f"ev_{k:02d}_{ne.clock(e.t).replace(':', 'm')}s_{e.dir}.jpg", 82)
        out.append({"k": k, "path": p, "t": float(e.t), "dir": e.dir, "pid": int(e.pid), "box": box, "frame": int(e.f)})
    return out


def make_hero(st: dict[str, Any], since: float, query: str, cards: list[dict[str, Any]], path: Path) -> None:
    ref, box = None, None
    if cards:
        p = cards[0]["primary"]
        ref = ne.read_frames(st["name"], [p["f"]]).get(p["f"])
        d = st["r"][(st["r"].pid == cards[0]["pid"]) & ((st["r"].f - p["f"]).abs() <= 2)]
        if len(d):
            row = d.iloc[(d.f - p["f"]).abs().argmin()]
            box = (row.x1, row.y1, row.x2, row.y2)
    ev = st["ev"]
    # per-card evidence: the frame of the crossing, zoomed on the line, person boxed, red dot = crossing point
    fr_all = ne.read_frames(st["name"], [c["primary"]["f"] for c in cards])
    for c in cards:
        p = c["primary"]
        fr = fr_all.get(p["f"])
        d = st["r"][(st["r"].pid == c["pid"]) & ((st["r"].f - p["f"]).abs() <= 2)]
        bx = None
        if len(d):
            row = d.iloc[(d.f - p["f"]).abs().argmin()]
            bx = (row.x1, row.y1, row.x2, row.y2)
        c["ev_bgr"] = render.evidence_thumb(fr, st["lines"], bx, p) if fr is not None else None
    render.hero(path, name=st["name"], title=st["name"].replace("_", " "), since=since,
                query=query, cards=cards, all_events=ev, duration=st["tr"].duration, blind=st["blind"], lines=st["lines"],
                ref_frame=ref, focus_box=box, n_before=int((ev.t < since).sum()))


def card_json(c: dict[str, Any]) -> dict[str, Any]:
    p = c["primary"]
    return {"person": c["pid"], "primary": {"t": p["t"], "clock": ne.clock(p["t"]), "dir": p["dir"], "door": p["line"],
                                            "interpolated": bool(p["inferred"])},
            "events": [{"t": e["t"], "dir": e["dir"], "door": e["line"]} for e in c["events"]],
            "score": None if c["score"] is None else round(c["score"], 3),
            "abs_score": None if c.get("abs") is None else round(c["abs"], 3),
            "gap_to_best": None if c.get("gap") is None else round(c["gap"], 3),
            "close_matches": c.get("n_close", 1) if c["score"] is not None else None,
            "parts": {k: (None if v is None else round(v, 3)) for k, v in c["parts"].items()},
            "top_colours": c["top"], "legs_colours": c["legs"], "description": c["desc"],
            "back_in_after_s": c["returned"]}


def make_report(st: dict[str, Any], since: float, query: str, cards: list[dict[str, Any]], folder: Path,
                evs: list[dict[str, Any]], hero_path: Path, qres: list[dict[str, Any]]) -> Path:
    name = st["name"]
    ev = st["ev"]
    rep = pk.Report(folder / "report.html", f"NIGHT EXIT: {name}",
                    "Who went through the exit since T? Video time is the clock (00:00 = start of the clip).")
    rep.credit(name)
    vs = vastdb_shortlist(st, since)
    rep.kv({"query": f"since {ne.clock(since)}" + (f' + "{query}"' if query else ""), "crossings in clip": len(ev),
            "persons since T": len(cards), "person tracks -> identities": f"{st['n_tracks']} -> {st['n_identities']}",
            "exit doors": ", ".join(sorted({(ln.group or ln.name) for ln in st["lines"]})),
            "VastDB shortlist": f"{vs['segments_with_people']} of {vs['segments_after_T']} segments after T"})
    rep.image(hero_path, "The answer card: best match first, ranked against the description.")
    rep.h2("VastDB step (DataEngine segment table)")
    rep.html(f"<pre>{html.escape(vs['sql'] or 'n/a: no person class in the segment table')}</pre>")
    rep.text(f"{vs['segments_with_people']} of {vs['segments_after_T']} five-second segments after T contain people; every crossing "
             f"after T lies inside one of them: {vs['all_crossings_inside']}. The table gives counts only (no boxes), so it narrows "
             "the search but cannot give a direction: the exit line needs the tracked boxes.")
    rep.h2("Card stack")
    html_cards = ['<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px">']
    for k, c in enumerate(cards):
        p = c["primary"]
        parts = " ".join(f"{kk[2:]}={vv:.2f}" for kk, vv in c["parts"].items() if vv is not None and kk != "s_emb_raw")
        colours = lambda h: ", ".join(f"{a} {int(b * 100)}%" for a, b in list(h.items())[:3])  # noqa: E731
        html_cards.append(
            f'<div class="card"><img src="{rep._rel(c["crop"])}" style="float:left;margin-right:10px;height:150px">'
            f'<b style="font-size:20px">#{k + 1} {p["dir"]} {ne.clock(p["t"])}</b><br>door: {html.escape(p["line"])}'
            f'{" (interpolated across a detection gap)" if p["inferred"] else ""}'
            f'{"<br>back IN after %.0f s" % c["returned"] if c["returned"] is not None else ""}<br>'
            f'top: {html.escape(colours(c["top"]))}<br>legs: {html.escape(colours(c["legs"]))}<br>'
            f'<i>"{html.escape(c["desc"] or "(no description)")}"</i><br>'
            f'<span style="color:var(--muted)">{"rank score %.2f, evidence (colour+text) %s (%s)%s" % (c["score"], "n/a" if c.get("abs") is None else "%.2f" % c["abs"], parts, " | CLOSE MATCH: %d candidates within %d points" % (c["n_close"], CLOSE * 100) if c.get("n_close", 1) > 1 and c["gap"] <= CLOSE else "") if c["score"] is not None else ""}</span>'
            f'<div style="clear:both"></div></div>')
    html_cards.append("</div>")
    rep.html("".join(html_cards) if cards else "<p>No crossings in this window.</p>")
    rep.h2("Timeline")
    spans = [(0, since, "#c9ced4", "before T: ignored")] if since > 0 else []
    spans += [(a, b, "blind", "camera moving: exit line not valid") for a, b in st["blind"]]
    top_keys = {(c["pid"], e["t"]) for c in cards[:1] for e in c["events"]} if query else set()
    for e in ev.itertuples():
        col = "#d64545" if (e.pid, e.t) in top_keys else ("#2e9e5b" if e.dir == "IN" else "#e0a100")
        spans.append((e.t - 0.6, e.t + 0.6, col, f"{ne.clock(e.t)} {e.dir} {e.line} person #{e.pid}"))
    rep.timeline(spans, st["tr"].duration, "green = IN, amber = OUT, red = best match, brown = camera moving (line invalid)")
    rep.h2("All crossings in the clip")
    rep.table([{"t": ne.clock(e.t), "dir": e.dir, "door": e.line, "person": e.pid, "interpolated": e.inferred,
                "back_in_s": "" if e.returned != e.returned else round(e.returned, 1)} for e in ev.itertuples()])
    rep.h2("Evidence frames (line yellow, person boxed, red dot = crossing point)")
    rep.html('<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px">' + "".join(
        f'<figure><img src="{rep._rel(x["path"])}" loading="lazy"></figure>' for x in evs) + "</div>")
    if qres:
        rep.h2("Eyeballed ranking queries")
        rep.table([{"since": q["since"], "query": q["query"], "valid answers": q["valid_answers"],
                    "rank (combined)": q["combined"]["rank_of_first_valid"], "rank (colour only)": q["colour"]["rank_of_first_valid"],
                    "rank (embed only)": q["embed"]["rank_of_first_valid"], "of": q["combined"]["n_candidates"],
                    "pass": q["pass"]} for q in qres])
    rep.h2("Method")
    rep.text("YOLO26s+ByteTrack person tracks (the DataEngine YOLO stand-in) -> constant-velocity stitching of fragments -> "
             "foot/centre point crosses the exit line (debounced) -> best crop (size x sharpness, no overlap) -> pose keypoints "
             "-> torso/leg colour names -> local Cosmos3-Edge one-liner -> rank = colour match + description match + "
             "Cosmos-Embed1 text similarity. Camera motion guard switches the line off while the view is not the reference view.")
    return rep.save()


def process(name: str, args: argparse.Namespace, gt: dict[str, Any]) -> dict[str, Any]:
    cfg = ne.CLIPS.get(name, {})
    lines = [parse_line(s) for s in args.line] if args.line else None
    t_start = time.time()
    st = analyse(name, lines=lines, use_vlm=not args.no_vlm)
    folder = OUT / name
    since_s = args.since if args.since is not None else cfg.get("since", "00:00")
    query = args.query if args.query is not None else cfg.get("query", "")
    since = ne.parse_clock(since_s)
    cards = build_cards(st, since, query, args.dir, args.top)
    make_hero(st, since, query, cards, folder / "hero.jpg")
    evs = event_frames(st, folder / "events")
    g = gt.get(name, {})
    qres = [eval_query(st, q) for q in g.get("queries", [])] if not args.line else []
    for i, q in enumerate(g.get("queries", [])):
        if not args.line:
            make_hero(st, ne.parse_clock(q["since"]), q["query"], build_cards(st, ne.parse_clock(q["since"]), q["query"], "any", 4),
                      folder / f"hero_q{i + 1}.jpg")
    rep = make_report(st, since, query, cards, folder, evs, folder / "hero.jpg", qres)
    ev = st["ev"]
    res: dict[str, Any] = {
        "clip": name, "duration_s": round(st["tr"].duration, 1), "clock": "video time (00:00 = start of clip)",
        "exit_lines": [ln.as_dict() for ln in st["lines"]], "camera_blind_spans": [[round(a, 1), round(b, 1)] for a, b in st["blind"]],
        "person_tracks": st["n_tracks"], "identities_after_stitching": st["n_identities"],
        "crossings": [{"t": e.t, "clock": ne.clock(e.t), "dir": e.dir, "door": e.line, "person": int(e.pid),
                       "interpolated": bool(e.inferred), "back_in_after_s": None if e.returned != e.returned else round(e.returned, 1)}
                      for e in ev.itertuples()],
        "vastdb_shortlist": vastdb_shortlist(st, since),
        "demo_query": {"since": since_s, "query": query, "persons_listed": len(cards), "cards": [card_json(c) for c in cards]},
        "seconds": {"crossings": round(st["t_events"], 1), "appearance_and_vlm": round(st["t_appearance"], 1),
                    "total": round(time.time() - t_start, 1)},
        "report": str(rep.relative_to(OUT.parent)).replace("\\", "/"), "hero": f"{OUT.name}/{name}/hero.jpg"}
    if g.get("events") is not None and not args.line:
        res["vs_eyeball"] = match_events(ev, g["events"])
        res["eyeball_note"] = g.get("note", "")
    if qres:
        res["ranking_queries"] = qres
    print(f"[night_exit] {name}: {len(ev)} crossings, {len(cards)} persons since {since_s}"
          + (f' for "{query}" -> best: {card_json(cards[0])["primary"]["dir"]} {card_json(cards[0])["primary"]["clock"]}' if cards and query else "")
          + f" ({res['seconds']['total']} s)")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help=f"run the default test set: {', '.join(ne.DEFAULT_SET)}")
    ap.add_argument("--video", help="catalogue name or path of one clip")
    ap.add_argument("--since", help="clock time, mm:ss or hh:mm:ss or seconds (video time). default per clip")
    ap.add_argument("--query", help='description, e.g. "grey top, dark trousers" (empty = time order only)')
    ap.add_argument("--dir", choices=["any", "in", "out"], default="any", help="only list this crossing direction")
    ap.add_argument("--line", action="append", help="override exit line 'name:x1,y1,x2,y2:inside_x,inside_y[:foot|centre]'")
    ap.add_argument("--top", type=int, default=4, help="cards on the stack")
    ap.add_argument("--out-dir", help="write out/<clip>/ here instead of ./out (used by test_smoke.py --recompute)")
    ap.add_argument("--results", help="write the results json here instead of ./results.json")
    ap.add_argument("--no-vlm", action="store_true", help="skip the Cosmos one-line descriptions")
    args = ap.parse_args()
    names = ne.DEFAULT_SET if args.all else [args.video] if args.video else []
    if not names:
        ap.error("pass --all or --video NAME")
    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir).resolve()
    res_path = Path(args.results).resolve() if args.results else HERE / "results.json"
    gt = load_gt()
    results: dict[str, Any] = {}
    prev = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    results.update(prev.get("videos", {}))
    t0 = time.time()
    for n in names:
        results[n] = process(n, args, gt)
    out = {"idea": "04 NIGHT EXIT", "clock_note": "video time is the clock", "videos": results,
           "last_run_seconds": round(time.time() - t0, 1)}
    kt = [q for v in results.values() for q in v.get("ranking_queries", [])]
    if kt:
        pre = [q for q in kt if not q["post_fix"]]
        out["kill_test"] = {
            "ranking_queries_preregistered": len(pre), "ranking_passed_now": sum(1 for q in pre if q["pass"]),
            "ranking_first_run": "6 of 7 passed: 'white coat' on retail_cctv failed (rank 2 of 9, hoodie girl 1.00 vs white-coat woman 0.98)",
            "post_hoc_fix": "added garment-word matching (coat/jacket vs hoodie) after seeing that failure: it now passes, so that "
                            "pass is NOT clean evidence; the 2 queries flagged post_fix were written afterwards",
            "ranking_post_fix_queries": sum(1 for q in kt if q["post_fix"]),
            "ranking_post_fix_passed": sum(1 for q in kt if q["post_fix"] and q["pass"]),
            "events_truth": sum(v["vs_eyeball"]["truth_events"] for v in results.values() if "vs_eyeball" in v),
            "events_matched": sum(v["vs_eyeball"]["matched"] for v in results.values() if "vs_eyeball" in v),
            "direction_correct": sum(v["vs_eyeball"]["direction_correct"] for v in results.values() if "vs_eyeball" in v)}
    pk.write_results(res_path, out)
    print(f"[night_exit] done in {time.time() - t0:.0f} s -> results.json")


if __name__ == "__main__":
    main()
