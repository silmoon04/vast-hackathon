#!/usr/bin/env python
"""BLIND TIME CLI.

  run.py --all                         ptz_parking + rochester_night + retail_cctv with SIMULATED faults (+ sensitivity clip)
                                       + HELD-OUT SIMULATED faults in store_aisle_detection and yonge_dundas
  run.py --video heldout               only the SIMULATED clips (sim + held-out)
  run.py --video NAME [--ref-t S]      one catalogue clip (or 'sim')
  run.py --video ptz_parking --query "person 02:00-04:00" --query "car 00:30-01:00 zone=0,0.5,1,1"
  run.py --force                       ignore the feature cache
Outputs: out/<video>/report.html, out/<video>/hero.jpg, results.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import blind as bl  # noqa: E402
import protokit as pk  # noqa: E402
import render as rd  # noqa: E402
import sim  # noqa: E402

OUT = HERE / "out"
SIM_NAME = "retail_cctv_SIM"

# queries asked of each clip: head (hero) + others. Windows were chosen after looking at the contact sheets.
REAL: dict[str, dict[str, Any]] = {
    "ptz_parking": {"ref_t": 0.0, "badge": "PTZ camera",
                    "head": "person 02:40-04:00",
                    "also": ["person 02:00-04:00", "car 00:30-01:30", "car 04:20-05:10"]},
    "rochester_night": {"ref_t": 0.0, "badge": "night, camera pans away",
                        "head": "person 00:10-01:00",
                        "also": ["person 00:00-00:10", "car 00:00-00:20", "car 00:30-00:50"]},
}
# the head query is the FROZEN one: plain search says SEEN (stale frame), Blind Time says COULD NOT SEE
SIM_QUERIES = {"head": "person 00:37-00:44", "also": ["person 00:15-00:23", "person 01:00-01:06", "person 01:17-01:26"]}
EYEBALL = HERE / "eyeball.json"


# ------------------------------------------------------------------------------------------------ helpers


def span_rows(an: dict[str, Any]) -> list[dict[str, Any]]:
    rows = an["rows"]
    out = []
    for a, b, st in bl.spans([r["state"] for r in rows]):
        mid = rows[(a + b - 1) // 2]
        out.append({"t0": a, "t1": b, "dur": b - a, "state": st, "why": mid["why"]})
    return out


def state_seconds(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {st: sum(1 for r in rows if r["state"] == st) for st in bl.STATES}


def battery(an: dict[str, Any], win: int = 30) -> dict[str, Any]:
    """Tile the clip into 30 s windows and ask 'person' / 'car' in each: how often does a plain search mislead?"""
    rows, dets, ref = an["rows"], an["dets"], an["ref"]
    n = len(rows)
    tot = nf = mis = stale = other = seen_ok = 0
    for cls in ("person", "car"):
        if not any(d["name"] == cls for fr in dets for d in fr):
            continue  # the clip has no such class at all: not a meaningful question
        for t0 in range(0, n - win + 1, win):
            a = bl.answer(rows, dets, bl.Query(cls, t0, t0 + win), ref)
            tot += 1
            nf += a["naive"] == "NOTHING FOUND"
            mis += a["naive_misleading"]
            stale += a["naive_stale"]
            other += a["verdict"] == "SEEN IN OTHER VIEW"
            seen_ok += a["verdict"] == "SEEN"
    return {"windows": tot, "window_s": win, "naive_nothing_found": nf, "naive_nothing_found_but_blind": mis,
            "naive_seen_but_only_stale_or_black_frames": stale, "seen_in_other_view": other, "seen": seen_ok}


def frame_for(src: str | Path, s: float) -> np.ndarray:
    return pk.frame_at(src, s + 0.5)


def render_queries(label_dir: Path, src: str | Path, an: dict[str, Any], queries: list[bl.Query]) -> list[dict[str, Any]]:
    rows, dets, ref = an["rows"], an["dets"], an["ref"]
    out = []
    con = pk.vastdb(bl.health_tables(rows, dets))  # duckdb standing in for VastDB
    for k, q in enumerate(queries):
        a = bl.answer(rows, dets, q, ref)
        if q.zone is None:  # the same answer from SQL over the health + hits tables must agree exactly
            sq = bl.answer_sql(con, q, len(rows))
            a["sql_check"] = {"agrees": sq["verdict"] == a["verdict"] and abs(sq["coverage"] - a["coverage"]) < 1e-9
                              and (sq["hit_seconds"] > 0) == bool(a["evidence"])
                              and (sq["other_view_seconds"] > 0) == bool(a["evidence_off_view"]), "sql": sq["sql"]}
        imgs: list[dict[str, Any]] = []
        for e in sorted(a["evidence"] or a["evidence_off_view"], key=lambda e: -e["conf"])[:3]:
            fr = frame_for(src, e["s"])
            d = [x for x in dets[e["s"]] if x["name"] == q.cls]
            im = rd.annotate(pk.fit(fr, 640), [{**x, "box": [v * 640 / fr.shape[1] for v in x["box"]]} for x in d], q.cls, None,
                             f"{pk.mmss(e['s'])[:5]}  {q.cls} x{e['n']}  [{e['state']}{' = other view' if not a['evidence'] else ''}]")
            p = pk.save_jpg(im, label_dir / f"q{k}_ev{e['s']}.jpg", 82)
            imgs.append({"path": str(p), "cap": f"{pk.mmss(e['s'])[:5]} {q.cls} x{e['n']} (camera state {e['state']}"
                        f"{'; hit in another field of view' if not a['evidence'] else ''})"})
        for sp in sorted(a["blind_spans"], key=lambda s: -(s["t1"] - s["t0"]))[:2]:
            s = (sp["t0"] + sp["t1"] - 1) // 2
            im = rd.annotate(pk.fit(frame_for(src, s), 640), None, None, sp["state"],
                             f"{pk.mmss(s)[:5]}  {sp['state']}: {bl.STATE_WHY[sp['state']]}")
            p = pk.save_jpg(im, label_dir / f"q{k}_blind{s}.jpg", 82)
            imgs.append({"path": str(p), "cap": f"{pk.mmss(s)[:5]} blind ({sp['state']})", "blind": True})
        a["images"] = imgs
        out.append(a)
    return out


def hero_for(path: Path, *, name: str, src: str | Path, an: dict[str, Any], answers: list[dict[str, Any]], badge: str,
             subtitle: str, tiles: list[tuple[str, str]] | None = None, footnote: str = "") -> Path:
    rows, dets = an["rows"], an["dets"]
    head = answers[0]
    states = [r["state"] for r in rows]
    ref = an["ref"]
    thumbs = [(rd.annotate(ref.color.copy(), None, None, "OK", "reference view"), "REFERENCE VIEW (home)", rd.STATE_RGB["OK"])]
    if head["verdict"] == "SEEN IN OTHER VIEW":
        e = max(head["evidence_off_view"], key=lambda e: e["conf"])
        fr = frame_for(src, e["s"])
        d = [x for x in dets[e["s"]] if x["name"] == head["cls"]]
        thumbs.append((rd.annotate(fr, d, head["cls"]), f"{pk.mmss(e['s'])[:5]}  {head['cls']} x{e['n']} (other view)",
                       rd.STATE_RGB["MOVED"]))
    elif head["blind_spans"] and head["verdict"] != "SEEN":
        sp = max(head["blind_spans"], key=lambda s: s["t1"] - s["t0"])
        s = (sp["t0"] + sp["t1"] - 1) // 2
        thumbs.append((frame_for(src, s), f"{pk.mmss(s)[:5]}  {sp['state']}", rd.STATE_RGB[sp["state"]]))
    elif head["evidence"]:
        e = max(head["evidence"], key=lambda e: e["conf"])
        fr = frame_for(src, e["s"])
        d = [x for x in dets[e["s"]] if x["name"] == head["cls"]]
        thumbs.append((rd.annotate(fr, d, head["cls"]), f"{pk.mmss(e['s'])[:5]}  {head['cls']} x{e['n']}", rd.STATE_RGB["OK"]))
    else:
        s = (head["t0"] + head["t1"]) // 2
        thumbs.append((frame_for(src, min(s, len(rows) - 1)), f"{pk.mmss(s)[:5]}  camera OK", rd.STATE_RGB["OK"]))
    sec = state_seconds(rows)
    n = len(rows)
    bt = battery(an)
    tiles = tiles or [(f"{sec['OK'] / n * 100:.0f}%", "of the clip the camera\nshowed the reference view"),
             (f"{n - sec['OK']} s", f"blind or degraded\n(of {n} s)"),
             (f"{bt['naive_nothing_found_but_blind']} of {bt['naive_nothing_found']}",
              "plain 'NOTHING FOUND'\nanswers that were blind")]
    return rd.hero(path, video=name, duration=an["duration"], states=states, head=head, also=answers[1:4], thumbs=thumbs,
                   tiles=tiles, subtitle=subtitle, badge=badge, footnote=footnote)


def verdict_tag(v: str) -> str:
    rgb = rd.VERDICT_RGB[v]
    return f'<span class="tag" style="background:rgb{rgb}">{html.escape(v)}</span>'


def query_card(rep: pk.Report, label_dir: Path, a: dict[str, Any]) -> None:
    sp = ", ".join(f"{s['state']} {pk.mmss(s['t0'])[:5]}-{pk.mmss(s['t1'])[:5]}" for s in a["blind_spans"][:6]) or "none"
    more = f" (+{len(a['blind_spans']) - 6} more)" if len(a["blind_spans"]) > 6 else ""
    ev = f"{len(a['evidence'])} s with hits" if a["evidence"] else "no hits on the reference view"
    off = f"; {len(a['evidence_off_view'])} s of hits in another field of view (not counted)" if a["evidence_off_view"] else ""
    rep.html(
        f'<div class="card"><b style="font-size:17px">{html.escape(a["query"])}</b> &nbsp; {verdict_tag(a["verdict"])}'
        f' &nbsp; <span style="color:var(--muted)">plain search says: <b>{html.escape(a["naive"])}</b>'
        f'{" (misleading)" if a["naive_misleading"] else ""}</span><br>'
        f'coverage <b>{a["coverage"] * 100:.0f}%</b> of {bl.fmt_dur(a["window_s"])} &middot; blind {bl.fmt_dur(a["blind_s"])}'
        f' &middot; blind spans: {html.escape(sp)}{more} &middot; {ev}{off}</div>')
    for im in a["images"][:4]:
        rep.image(im["path"], im["cap"])


def write_report(path: Path, *, name: str, src: str | Path, an: dict[str, Any], answers: list[dict[str, Any]], hero: Path | None,
                 extra: list[tuple[str, Any]] | None = None, credit: str | None = None, subtitle: str = "") -> None:
    rows = an["rows"]
    n = len(rows)
    sec = state_seconds(rows)
    rep = pk.Report(path, f"Blind Time: {name}", subtitle)
    rep.credit(credit or src)
    if hero is not None:
        rep.image(hero, "Hero: the first answer, in one picture")
    rep.kv({"seconds analysed": n, "camera OK": f"{sec['OK'] / n * 100:.0f}%", "blind or degraded": f"{n - sec['OK']} s",
            **{k: f"{v} s" for k, v in sec.items() if k != "OK" and v}})
    rep.h2("Camera health, per second")
    spans = [(a, b, bl.STATE_HEX[st], f"{st} {pk.mmss(a)[:5]}-{pk.mmss(b)[:5]}: {bl.STATE_WHY[st]}") for a, b, st in
             bl.spans([r["state"] for r in rows])]
    rep.timeline(spans, an["duration"], "green = OK, amber = camera looks elsewhere, purple = blur, blue = frozen, black = dark, red = occluded")
    rep.table([{"from": pk.mmss(s["t0"])[:5], "to": pk.mmss(s["t1"])[:5], "dur_s": s["dur"], "state": s["state"], "evidence": s["why"]}
               for s in span_rows(an)], ["from", "to", "dur_s", "state", "evidence"], 60)
    rep.h2("Query answers (three states)")
    for a in answers:
        query_card(rep, path.parent, a)
    chk = [a for a in answers if a.get("sql_check")]
    if chk:
        rep.h2("The same answers from VastDB-style SQL")
        rep.html(f"<p>Health and hits live in two small tables next to the DataEngine segment table; every verdict above is "
                 f"recomputed with plain <code>WHERE</code> + <code>COUNT/SUM</code> (no GROUP BY, JOIN or IN). "
                 f"<b>{sum(a['sql_check']['agrees'] for a in chk)} of {len(chk)}</b> answers are identical to the Python "
                 f"implementation. SQL for the first query:</p><pre>{html.escape(chr(10).join(chk[0]['sql_check']['sql']))}</pre>")
    for title, obj in extra or []:
        rep.h2(title)
        if isinstance(obj, str):
            rep.html(obj)
        else:
            rep.json(obj)
    rep.h2("Method")
    rep.html("<p>Per second: mean luminance (DARK), identical consecutive frames (FROZEN), Crete blur metric (BLUR), ORB + RANSAC "
             "similarity transform against a median-background reference view (MOVED: share of the reference view still in "
             "frame, zoom, rotation; no registrable scene), largest region differing from the registered background (OCCLUDED). "
             "Queries: SEEN needs a detection while the camera covered the asked-about view; SEEN IN OTHER VIEW = detections exist "
             "but only while the camera looked elsewhere (the asked-about place was not watched); NOTHING FOUND needs coverage of at "
             f"least {bl.CFG['coverage_for_nothing'] * 100:.0f}%; everything else is COULD NOT SEE.</p>")
    rep.json({k: (list(v) if isinstance(v, tuple) else v) for k, v in bl.CFG.items()}, "Thresholds used")
    rep.save()


# ------------------------------------------------------------------------------------------------ eyeball


def compare_eyeball(name: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not EYEBALL.exists():
        return None
    gt = json.loads(EYEBALL.read_text(encoding="utf-8")).get(name)
    if not gt:
        return None
    n = len(rows)
    lab = ["HOME"] * n
    for t0, t1 in gt["away"]:
        for s in range(n):  # a second counts as AWAY when at least half of it is inside an away span
            if min(s + 1, t1) - max(s, t0) >= 0.5:
                lab[s] = "AWAY"
    pred = ["HOME" if r["state"] == "OK" else "AWAY" for r in rows]
    agree = sum(p == g for p, g in zip(pred, lab))
    gt_away = sum(g == "AWAY" for g in lab)
    pred_away = sum(p == "AWAY" for p in pred)
    tp = sum(p == g == "AWAY" for p, g in zip(pred, lab))
    return {"seconds": n, "agree": agree, "agreement": agree / n, "gt_away_s": gt_away, "pred_away_s": pred_away,
            "away_precision": tp / pred_away if pred_away else None, "away_recall": tp / gt_away if gt_away else None,
            "pred_spans_non_ok": [[a, b, s] for a, b, s in bl.spans([r["state"] for r in rows]) if s != "OK"],
            "gt_away": gt["away"], "gt_note": gt.get("note", "")}


# ------------------------------------------------------------------------------------------------ real clips


def auto_queries(an: dict[str, Any]) -> tuple[str, list[str]]:
    dets, rows = an["dets"], an["rows"]
    cnt: dict[str, int] = {}
    for fr in dets:
        for d in fr:
            cnt[d["name"]] = cnt.get(d["name"], 0) + 1
    cls = max(cnt, key=lambda k: cnt[k]) if cnt else "person"
    sp = [s for s in span_rows(an) if s["state"] != "OK"]
    n = len(rows)
    if sp:
        b = max(sp, key=lambda s: s["dur"])
        head = f"{cls} {max(0, b['t0'] - 15)}-{min(n, b['t1'] + 15)}"
    else:
        head = f"{cls} 0-{min(60, n)}"
    return head, [f"{cls} 0-{min(30, n)}"]


def spans_sheet(out: Path, src: str | Path, an: dict[str, Any]) -> Path:
    """One frame (middle second) from each state span, border = detected state: my eyeball sheet."""
    sp = span_rows(an)
    sel = sp if len(sp) <= 12 else sorted(sorted(sp, key=lambda s: -s["dur"])[:12], key=lambda s: s["t0"])
    sheet = pk.sheet([rd.annotate(pk.fit(frame_for(src, (s["t0"] + s["t1"] - 1) // 2), 480), None, None, s["state"],
                                  f"{pk.mmss((s['t0'] + s['t1'] - 1) // 2)[:5]} {s['state']}") for s in sel],
                     cols=4, tile_w=320)
    return pk.save_jpg(sheet, out / "spans_sheet.jpg", 85)


def run_real(name: str, args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    cfg = REAL.get(name, {})
    ref_t = args.ref_t if args.ref_t is not None else cfg.get("ref_t", 0.0)
    out = OUT / name
    an = bl.analyze(name, out / "_work", ref_t, force=args.force)
    bl.classify(an)
    head, also = (cfg["head"], cfg["also"]) if cfg else auto_queries(an)
    qtexts = list(args.query or []) or [head] + also
    queries = [bl.Query.parse(t.replace("m", ":") if False else t) for t in qtexts]
    answers = render_queries(out, name, an, queries)
    sub = f"{pk.mmss(an['duration'])[:5]} long | {an['w']}x{an['h']} | reference view = {pk.mmss(ref_t)[:5]}"
    hero = hero_for(out / "hero.jpg", name=name, src=name, an=an, answers=answers, badge=cfg.get("badge", ""), subtitle=sub)
    # frames at the middle of each span: the eyeball sheet
    spans_sheet(out, name, an)
    eye = compare_eyeball(name, an["rows"])
    extra: list[tuple[str, Any]] = [("One frame from each state span", None)]
    rep_extra = [("Frames from the detected spans", f'<figure><img src="spans_sheet.jpg"><figcaption>middle second of each span, '
                                                    "border colour = detected state</figcaption></figure>")]
    if eye:
        rep_extra.append(("Eyeball check (my own reading of frames) vs detector", eye))
    write_report(out / "report.html", name=name, src=name, an=an, answers=answers, hero=hero, extra=rep_extra, subtitle=sub)
    res = {"video": name, "duration_s": an["duration"], "seconds_analysed": len(an["rows"]), "ref_t": ref_t,
           "state_seconds": state_seconds(an["rows"]), "ok_pct": 100 * state_seconds(an["rows"])["OK"] / len(an["rows"]),
           "spans": span_rows(an), "queries": [{k: v for k, v in a.items() if k != "images"} for a in answers],
           "battery_30s_windows": battery(an), "eyeball": eye, "seconds_to_compute": an.get("seconds_to_compute"),
           "wall_s": round(time.time() - t0, 1)}
    pk.write_results(out / "results.json", res)
    return res


# ------------------------------------------------------------------------------------------------ simulated faults


def ensure_sim(plan: list[dict[str, Any]], fname: str, force: bool, base: str | None = None) -> Path:
    d = OUT / "sim"
    p = d / fname
    meta = d / (fname + ".plan.json")
    want = json.dumps(plan, sort_keys=True)
    if p.exists() and meta.exists() and meta.read_text() == want and not force:
        return p
    t = time.time()
    sim.make_faulty(base or sim.BASE, plan, p)
    meta.write_text(want)
    print(f"  [sim] wrote {p.name} in {time.time() - t:.0f}s")
    return p


def fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def run_sim(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = OUT / SIM_NAME
    p = ensure_sim(sim.DEFAULT_PLAN, "retail_cctv_faulty.mp4", args.force)
    an = bl.analyze(p, out / "_work", 0.0, force=args.force)
    bl.classify(an)
    rows = an["rows"]
    n = len(rows)
    nat_file = out / "_work" / "natural_stalls.json"
    if nat_file.exists() and not args.force:
        nat = json.loads(nat_file.read_text())
    else:
        nat = sim.natural_stall_seconds(sim.BASE)
        nat_file.write_text(json.dumps(nat))
    gt = sim.gt_labels(n, sim.DEFAULT_PLAN, nat)
    pred = [r["state"] for r in rows]
    raw = [r["raw_state"] for r in rows]
    sc = sim.score(pred, gt)
    sc_raw = sim.score(raw, gt)
    conf = sim.confusion(pred, gt)
    spec = ("DARK", "FROZEN", "BLUR", "MOVED")
    rec_min = min(sc[s]["recall"] for s in spec)
    rec_tol_min = min(sc[s]["recall_tol"] for s in spec)
    # same but only inside injected spans, excluding the two boundary seconds
    queries = [bl.Query.parse(t) for t in [SIM_QUERIES["head"]] + SIM_QUERIES["also"]]
    answers = render_queries(out, p, an, queries)
    sub = "SIMULATED faults injected into retail_cctv (ground truth known) | not a real outage"
    inj_gt = sim.gt_labels(n, sim.DEFAULT_PLAN, None)  # injected faults only (natural stalls left out)
    sc_inj = sim.score(pred, inj_gt)
    rec_inj = {st: sc_inj[st]["recall"] for st in ("DARK", "FROZEN", "BLUR", "MOVED", "OCCLUDED")}
    tiles = [(f"{rec_min * 100:.0f}%", "lowest recall of the 4\nspec faults (per second)"),
             (f"{(sc['OK']['precision'] or 0) * 100:.0f}%", "OK precision: clean\nseconds called OK"),
             (f"{sum(1 for v in rec_inj.values() if v and v >= 0.9)} of 5", "fault types caught\n(recall >= 90%)")]
    hero = hero_for(out / "hero.jpg", name="retail_cctv + SIMULATED faults", src=p, an=an, answers=answers, badge="SIMULATED",
                    subtitle=sub, tiles=tiles,
                    footnote="blue at 0:00-0:03 and 1:38-1:51 is a REAL stall already in the source clip, not simulated")
    sweep = run_sweep(args, nat)
    rows_tbl = [{"state": st, "gt_s": sc[st]["gt_s"], "pred_s": sc[st]["pred_s"], "precision": fmt_pct(sc[st]["precision"]),
                 "recall": fmt_pct(sc[st]["recall"]), "recall (+-1 s)": fmt_pct(sc[st]["recall_tol"]),
                 "precision (+-1 s)": fmt_pct(sc[st]["precision_tol"])} for st in bl.STATES]
    plan_tbl = [{"state": f["state"], "from": pk.mmss(f["t0"]), "to": pk.mmss(f["t1"]), "what (SIMULATED)": f["what"]} for f in sim.DEFAULT_PLAN]
    rep_extra: list[tuple[str, Any]] = [
        ("SIMULATED fault plan (ground truth)", None), ("", None)]
    rep_extra = [("SIMULATED fault plan (ground truth)", None)]
    # build html pieces with a throwaway Report to reuse the table renderer
    tmp = pk.Report(out / "_tmp.html", "x")
    tmp.table(plan_tbl)
    t1 = tmp.parts[-1]
    tmp.table(rows_tbl)
    t2 = tmp.parts[-1]
    tmp.table(sweep["table"])
    t3 = tmp.parts[-1]
    (out / "_tmp.html").unlink(missing_ok=True)
    nat_txt = ("The untouched retail_cctv clip already contains real stalls (native-resolution frame pairs identical to within "
               "codec noise; I confirmed by eye that people stand in the same places at 99.0, 102.0 and 104.5 s): "
               f"{len(nat)} s ({', '.join(pk.mmss(s)[:5] for s in nat[:4])} ... {pk.mmss(nat[-1])[:5] if nat else ''}). "
               "They count as FROZEN ground truth, so a clean detector must report them.")
    spans_sheet(out, p, an)
    rep_extra = [("SIMULATED fault plan (ground truth)", t1 + f"<p>{html.escape(nat_txt)}</p>"),
                 ("Frames from the detected spans", '<figure><img src="spans_sheet.jpg"><figcaption>middle second of each span '
                                                    "of the SIMULATED clip, border colour = detected state</figcaption></figure>"),
                 ("Per-state precision / recall (per second, strict = exact second)", t2),
                 ("Sensitivity: how strong must a SIMULATED fault be?", t3),
                 ("Confusion (rows = ground truth)", conf)]
    write_report(out / "report.html", name="retail_cctv + SIMULATED faults", src=p, an=an, answers=answers, hero=hero,
                 extra=rep_extra, credit=sim.BASE, subtitle=sub)
    res = {"video": SIM_NAME, "SIMULATED": True, "base": sim.BASE, "duration_s": an["duration"], "seconds_analysed": n,
           "plan": sim.DEFAULT_PLAN, "natural_frozen_s": nat, "scores_smoothed": sc, "scores_raw": sc_raw, "confusion": conf,
           "recall_injected_only": rec_inj, "recall_min_4_spec_faults": rec_min, "recall_tol_min_4_spec_faults": rec_tol_min,
           "pred_spans": span_rows(an), "queries": [{k: v for k, v in a.items() if k != "images"} for a in answers],
           "battery_30s_windows": battery(an), "sweep": sweep, "seconds_to_compute": an.get("seconds_to_compute"),
           "wall_s": round(time.time() - t0, 1)}
    pk.write_results(out / "results.json", res)
    return res


def run_sweep(args: argparse.Namespace, nat: list[int]) -> dict[str, Any]:
    plan = sim.sweep_plan()
    p = ensure_sim(plan, "retail_cctv_sweep.mp4", args.force)
    an = bl.analyze(p, OUT / SIM_NAME / "_work_sweep", 0.0, force=args.force, with_dets=False)
    bl.classify(an)
    rows = an["rows"]
    tbl = []
    for f in plan:
        secs = [s for s in range(int(np.ceil(f["t0"])), int(np.floor(f["t1"]))) if s < len(rows)]  # seconds fully inside the fault
        got = [rows[s]["state"] for s in secs]
        exp = f["state"]
        tbl.append({"fault (SIMULATED)": f["what"], "expected": exp, "seconds": len(secs),
                    "detected as expected": f"{sum(g == exp for g in got)}/{len(secs)}",
                    "flagged as any non-OK": f"{sum(g != 'OK' for g in got)}/{len(secs)}",
                    "states seen": ",".join(sorted(set(got)))})
    ok_secs = [s for s in range(len(rows)) if s not in nat and not any(f["t0"] - 1 <= s < f["t1"] + 1 for f in plan)]
    fp = sum(rows[s]["state"] != "OK" for s in ok_secs)
    return {"table": tbl, "clean_seconds": len(ok_secs), "false_alarm_seconds": fp}


# ------------------------------------------------------------------------------------------------ held-out simulation


def run_heldout(base: str, args: argparse.Namespace) -> dict[str, Any]:
    """Re-inject the 5 fault types at NEW times / strengths into another healthy clip. Thresholds are the frozen
    blind.CFG ones: scored once, no re-tuning. SIMULATED."""
    t0 = time.time()
    plan = sim.HELDOUT[base]
    out = OUT / f"heldout_{base}"
    p = ensure_sim(plan, f"{base}_heldout.mp4", args.force, base=base)
    an = bl.analyze(p, out / "_work", 0.0, force=args.force, with_dets=False)
    bl.classify(an)
    rows = an["rows"]
    n = len(rows)
    nat_file = out / "_work" / "natural_stalls.json"
    out.mkdir(parents=True, exist_ok=True)
    (out / "_work").mkdir(parents=True, exist_ok=True)
    if nat_file.exists() and not args.force:
        nat = json.loads(nat_file.read_text())
    else:
        nat = sim.natural_stall_seconds(base)
        nat_file.write_text(json.dumps(nat))
    pred = [r["state"] for r in rows]
    gt = sim.gt_labels(n, plan, nat)
    gt_inj = sim.gt_labels(n, plan, None)
    sc, sc_inj = sim.score(pred, gt), sim.score(pred, gt_inj)
    # per injected fault: seconds fully inside the fault (strict) that carry the expected state
    per_fault = []
    for f in plan:
        secs = [s for s in range(int(np.ceil(f["t0"])), int(np.floor(f["t1"]))) if s < n]
        got = [pred[s] for s in secs]
        per_fault.append({"fault (SIMULATED)": f["what"], "expected": f["state"], "seconds": len(secs),
                          "detected as expected": f"{sum(g == f['state'] for g in got)}/{len(secs)}",
                          "flagged as any non-OK": f"{sum(g != 'OK' for g in got)}/{len(secs)}", "states seen": ",".join(sorted(set(got)))})
    healthy = [s for s in range(n) if s not in nat and not any(f["t0"] - 1 <= s < f["t1"] + 1 for f in plan)]
    false_alarm = sum(pred[s] != "OK" for s in healthy)
    rec = {st: sc[st]["recall"] for st in ("DARK", "FROZEN", "BLUR", "MOVED", "OCCLUDED")}
    rec_inj = {st: sc_inj[st]["recall"] for st in ("DARK", "FROZEN", "BLUR", "MOVED", "OCCLUDED")}
    spans_sheet(out, p, an)
    tmp = pk.Report(out / "_tmp.html", "x")
    tmp.table(per_fault)
    t1 = tmp.parts[-1]
    tmp.table([{"state": st, "gt_s": sc[st]["gt_s"], "pred_s": sc[st]["pred_s"], "precision": fmt_pct(sc[st]["precision"]),
                "recall": fmt_pct(sc[st]["recall"]), "recall (+-1 s)": fmt_pct(sc[st]["recall_tol"])} for st in bl.STATES])
    t2 = tmp.parts[-1]
    (out / "_tmp.html").unlink(missing_ok=True)
    rep_extra = [("HELD-OUT fault plan, per injected fault (SIMULATED)", t1),
                 ("Frames from the detected spans", '<figure><img src="spans_sheet.jpg"><figcaption>middle second of each span, '
                                                    "border colour = detected state</figcaption></figure>"),
                 ("Per-state precision / recall (frozen thresholds)", t2), ("Confusion (rows = ground truth)", sim.confusion(pred, gt))]
    write_report(out / "report.html", name=f"{base} + HELD-OUT SIMULATED faults", src=p, an=an, answers=[], hero=None, extra=rep_extra,
                 credit=base, subtitle="SIMULATED faults, new times/strengths, thresholds frozen (held-out check)")
    res = {"video": f"heldout_{base}", "SIMULATED": True, "base": base, "duration_s": an["duration"], "seconds_analysed": n,
           "plan": plan, "natural_frozen_s": nat, "scores": sc, "recall": rec, "recall_injected_only": rec_inj,
           "recall_min_5": min(v for v in rec.values() if v is not None), "per_fault": per_fault,
           "ok_precision": sc["OK"]["precision"], "healthy_seconds": len(healthy), "false_alarm_seconds": false_alarm,
           "confusion": sim.confusion(pred, gt), "pred_spans": span_rows(an), "wall_s": round(time.time() - t0, 1)}
    pk.write_results(out / "results.json", res)
    return res


# ------------------------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="default test set")
    ap.add_argument("--video", help="catalogue name, or 'sim' for the SIMULATED-fault clip")
    ap.add_argument("--query", action="append", help="e.g. 'person 02:00-04:00' (optional 'zone=x1,y1,x2,y2', normalised)")
    ap.add_argument("--ref-t", type=float, default=None, help="reference (home) view time in seconds (default 0)")
    ap.add_argument("--force", action="store_true", help="recompute features (ignore cache)")
    args = ap.parse_args(argv)
    if not args.all and not args.video:
        ap.print_help()
        return 2
    t0 = time.time()
    heldout = ["heldout_" + b for b in sim.HELDOUT]
    names = ["ptz_parking", "rochester_night", "sim"] + heldout if args.all else ["sim"] + heldout if args.video == "heldout" else [args.video]
    summary: dict[str, Any] = {}
    for nm in names:
        print(f"== {nm}")
        if nm.startswith("heldout_"):
            res = run_heldout(nm[len("heldout_"):], args)
        else:
            res = run_sim(args) if nm in ("sim", SIM_NAME) else run_real(nm, args)
        summary[res["video"]] = res
        if "ok_pct" in res:
            print(f"   ok {res['ok_pct']:.0f}% of seconds")
        elif "recall_min_5" in res:
            print(f"   HELD-OUT min recall of 5 fault types {res['recall_min_5'] * 100:.0f}%, "
                  f"false alarms {res['false_alarm_seconds']}/{res['healthy_seconds']} healthy s")
        else:
            print(f"   spec-fault min recall {res['recall_min_4_spec_faults'] * 100:.0f}%")
        for q in res.get("queries", []):
            print(f"   {q['query']:<24} {q['verdict']:<14} (plain search: {q['naive']}; coverage {q['coverage'] * 100:.0f}%)")
    if args.all:
        pk.write_results(HERE / "results.json", {"videos": summary, "total_wall_s": round(time.time() - t0, 1),
                                                 "thresholds": {k: (list(v) if isinstance(v, tuple) else v) for k, v in bl.CFG.items()}})
        print(f"results.json written, {time.time() - t0:.0f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
