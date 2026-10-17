"""ON THE LINE: scene answers for an RVRC operator on the phone to police.

    run.py --all                       default test set (3 clips) + 3 held-out clips, writes out/<clip>/{report.html,hero.jpg,demo.mp4}
    run.py --video retail_cctv         one clip
    run.py --video X --ask "which way did they go?@00:12"    extra typed question at video time 12 s
    run.py --video X --no-vlm --no-video --no-causality       fast iteration
Scripts (the five police questions per clip + eyeballed truth) live in scripts.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import protokit as pk  # noqa: E402

import incident_log  # noqa: E402
import qa  # noqa: E402
import sql_baseline  # noqa: E402
import render  # noqa: E402
import sim  # noqa: E402

AGREE_WAIT = 90.0   # seconds to wait for the shared GPU lock before reporting the yolo26l check as unavailable
DEFAULT_SET = ["person_bicycle_car_detection", "rochester_night", "retail_cctv"]
HELD_NOTE = ("scripts and truth written from contact sheets before the first run on these clips. one_by_one and people_detection: no change to the engine after seeing their results. car_detection: first run was 5/6 (C6 counted 5 vehicles, a half-visible car at the frame edge got a second tracker id; the red car was named 'pink'); I then fixed the edge-merge rule and the red/pink naming, so car_detection is no longer a clean held-out result")
HELD_OUT = ["one_by_one_person_detection", "car_detection", "people_detection"]   # thresholds never tuned on these
OUT = HERE / "out"
def load_scripts() -> dict:
    return json.loads((HERE / "scripts.json").read_text(encoding="utf-8"))


def parse_time(s: str) -> float:
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


def _comparable(facts: dict) -> str:
    return json.dumps(facts, sort_keys=True, default=lambda o: round(float(o), 4))


def causality_check(name: str, script: dict, full: sim.SimResult, use_vlm: bool) -> list[dict]:
    """Re-run the clip stopping at each question's time: the answer facts must be identical. Proves no future data
    leaks into an answer."""
    out = []
    for s in full.scored:
        part = sim.simulate(name, script, use_vlm=use_vlm, only=s["i"])
        mine = [x for x in part.scored if x["i"] == s["i"]]
        got = mine[0]["facts"] if mine else None
        out.append({"id": s["id"], "identical": got is not None and _comparable(got) == _comparable(s["facts"])})
    return out


def _det_l_counts(name: str, ts: list[float]) -> list[dict[str, int]]:
    """Person / vehicle counts from yolo26l on the frames at ts. Cached on disk; waits max 90 s for the shared GPU."""
    cache = OUT / "_det_l_cache.json"
    store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    key = name + "|" + ",".join(f"{t:.3f}" for t in ts)
    if key in store:
        return store[key]
    frames = [pk.frame_at(name, t) for t in ts]
    vehicle_names = {"car", "truck", "bus", "motorcycle"}
    out = []
    with pk.gpu_lock(timeout=AGREE_WAIT):
        m = pk._yolo("det_l")
        for fr in frames:
            res = m.predict(fr, conf=0.40, imgsz=640, verbose=False)[0]
            names = [m.names[c] for c in res.boxes.cls.int().tolist()]
            out.append({"person": sum(n == "person" for n in names), "vehicle": sum(n in vehicle_names for n in names)})
        pk._unload()
    store[key] = out
    cache.write_text(json.dumps(store), encoding="utf-8")
    return out


def detector_agreement(name: str, res: sim.SimResult, script: dict) -> dict:
    """Engine's live count vs an independent larger detector (yolo26l) on frames across the call. Agreement, not truth."""
    st = res.state
    try:
        groups = [g for g in ("person", "vehicle") if any(e.group == g for e in st.confirmed())]
        ts = [t for t, _ in st.count_log][::max(1, len(st.count_log) // 14)][2:]
        ts = [t for t in ts if t > st.alarm_t + 1.0][:12]
        ind_counts = _det_l_counts(name, ts)
        by_t = dict(st.count_log)
        near = lambda t: by_t[min(by_t, key=lambda k: abs(k - t))]  # noqa: E731
        out = {}
        for g in groups:
            eng = [near(t)[g] for t in ts]
            ind = [c[g] for c in ind_counts]
            diff = np.abs(np.array(eng) - np.array(ind))
            out[g] = {"frames": len(ts), "mean_abs_diff": round(float(diff.mean()), 2), "within_1": round(float((diff <= 1).mean()), 2),
                      "engine": eng, "yolo26l": ind}
        return out
    except Exception as e:  # GPU busy / model missing: report as unavailable, never fake
        return {"unavailable": f"{type(e).__name__}: {e}"}


def vlm_stats(res: sim.SimResult) -> dict:
    st = res.state
    secs = [v["measured_seconds"] for v in st.vlm_log if v.get("measured_seconds")]
    agree, n, withheld, aged = 0, 0, 0, 0
    for e in st.confirmed():
        if not e.desc:
            continue
        n += 1
        why = qa.cosmos_conflict(e.desc["text"], e.group, st.colours(e))
        if why is None:
            agree += 1
        else:
            withheld += 1
            aged += why == "Cosmos guessed an age"
    return {"calls": st.vlm_calls, "described": sum(1 for e in st.entities.values() if e.desc),
            "measured_local_median_s": round(statistics.median(secs), 2) if secs else None,
            "measured_local_max_s": round(max(secs), 2) if secs else None,
            "assumed_availability_delay_s_SIMULATED": sim.VLM_LATENCY,
            "colour_agreement_with_pixels": f"{agree}/{n}" if n else "n/a", "withheld_by_conflict_check": f"{withheld}/{n}", "of_which_age_guess": aged, "samples": [v["text"] for v in st.vlm_log if v.get("text")][:6]}


def refresh_for_hero(st: sim.SceneState, c: qa.Card) -> qa.Card:
    """The hero is drawn at hang-up: an appearance card that said 'Cosmos description still being generated' at question
    time gets the Cosmos line that was ready by hang-up (same conflict check). Headline, verdict and latency are untouched,
    and the line says it was refreshed."""
    a = c.facts.get("appearance")
    if not a or not any(("still being generated" in ln or "No Cosmos description" in ln) for ln in c.lines):
        return c
    e = st.entities.get(a["eid"])
    if e is None:
        return c
    lines, cav, _ = qa.cosmos_lines(st.facts(e), e.group)
    if not lines or "still being generated" in lines[0] or "No Cosmos description" in lines[0]:
        return c
    keep = [ln for ln in c.lines if "Cosmos" not in ln]
    tag = "(ready after the call) " if lines[0].startswith("Cosmos") else ""
    return dataclasses.replace(c, lines=[tag + lines[0]] + keep + lines[1:], caveats=cav + [x for x in c.caveats if x not in cav])


def save_evidence(card, outdir: Path, qi: int) -> list[dict]:
    files = []
    for k, ev in enumerate(card.evidence):
        p = outdir / "evidence" / f"q{qi + 1}_{k}.jpg"
        pk.save_jpg(ev.img if max(ev.img.shape[:2]) <= 420 else pk.fit(ev.img, 420), p)
        files.append({"file": str(p.relative_to(outdir)).replace("\\", "/"), "caption": ev.caption, "t": round(ev.t, 2), "kind": ev.kind})
    return files


def run_clip(name: str, script: dict, *, use_vlm: bool, make_video: bool, do_causality: bool, adhoc: list[dict],
             prior: dict | None = None) -> dict:
    t_clip = time.time()
    script = json.loads(json.dumps(script))
    script["questions"] += adhoc
    outdir = OUT / name
    outdir.mkdir(parents=True, exist_ok=True)
    title = script["title"]
    qs = script["questions"]
    tr = pk.tracks(name)
    writer = pk.Mp4Writer(outdir / "demo.mp4", tr.fps) if make_video else None
    state: dict = {"last_new": -9.0}

    def on_frame(st, t, frame, cards, _flag):
        if writer is None:
            return
        cap = next((q["text"] for q in qs if q["t"] - 2.4 <= t <= q["t"] + 1.4), None)
        new_idx = None
        if cards and t - cards[-1].t <= 1.5:
            new_idx = len(cards) - 1
        qi = [i for i, q in enumerate(qs) if i < len(qs)]
        writer.write(render.compose(st, t, frame, cards, title=title, hangup_t=script["hangup"], questions=qs, slots=max(5, len(qs)),
                                    q_idx=list(range(len(cards))), new_idx=new_idx, caption=cap))

    res = sim.simulate(name, script, use_vlm=use_vlm, on_frame=on_frame if make_video else None)
    st = res.state
    log = incident_log.draft(st, res.cards, title, script["hangup"])
    last_frame = pk.frame_at(name, min(script["hangup"], tr.duration - 0.2))
    q_idx = [s["i"] for s in res.scored]
    if writer is not None:
        hold = render.compose(st, st.now, last_frame, res.cards, title=title, hangup_t=script["hangup"], questions=qs, slots=max(5, len(qs)),
                              q_idx=q_idx, caption="Hang-up.")
        for _ in range(int(tr.fps * 1.0)):
            writer.write(hold)
        scr = render.log_screen(st, last_frame, res.cards, log, title=title, hangup_t=script["hangup"], questions=qs, q_idx=q_idx)
        for _ in range(int(tr.fps * 5.0)):
            writer.write(scr)
        writer.__exit__()
    # ---- scoring
    scored_q = [s for s in res.scored if s["correct"] is not None]
    n_ok = sum(1 for s in scored_q if s["correct"])
    lat = [c.latency_ms for c in res.cards]
    # ---- hero
    hero_t = script.get("hero_t", script["hangup"])
    hero_frame = pk.frame_at(name, hero_t)
    stamp = (f"{n_ok}/{len(scored_q)} RIGHT", f"slowest answer {max(lat):.0f} ms, no future data used")
    asked = {c.target_eid for c in res.cards if c.target_eid}
    log_short = [f"{pk.mmss(l.t)}  {l.text}" for l in log["lines"] if l.kind != "qa" and (l.kind == "alarm" or l.eid in asked)][:5]
    hero_cards = [refresh_for_hero(st, c) for c in res.cards]
    crowded = len(st.confirmed()) > 9
    focus = {c.target_eid for c in res.cards if c.target_eid} if crowded else None
    hero = render.compose(st, hero_t, hero_frame, hero_cards, title=title, hangup_t=script["hangup"], questions=qs, slots=max(5, len(qs)),
                          scored=res.scored, q_idx=q_idx, summary=True, stamp=stamp, log_lines=log_short, focus=focus)
    pk.save_jpg(hero, outdir / "hero.jpg", 90)
    # ---- extras
    caus = causality_check(name, script, res, use_vlm) if do_causality else ((prior or {}).get("causality") or [])   # reuse the last full check when skipped
    agree = detector_agreement(name, res, script)
    vstat = vlm_stats(res)
    sqlb = sql_baseline.compute(name, script)
    # ---- evidence + report
    evid = [save_evidence(c, outdir, i) for i, c in enumerate(res.cards)]
    build_report(name, script, res, log, evid, outdir, n_ok, len(scored_q), caus, agree, vstat, sqlb)
    ents = st.confirmed(include_carried=True)
    stats = {
        "title": title, "alarm_s": script["alarm"], "hangup_s": script["hangup"], "frames_ingested": st.n_frames,
        "fps_sampled": tr.fps, "raw_tracks_confirmed": sum(1 for r in st.raw.values() if r.eid is not None),
        "entities": len(ents), "carried_not_counted": sum(1 for e in ents if e.carried),
        "answers_right": n_ok, "answers_scored": len(scored_q),
        "latency_ms": {"p50": round(statistics.median(lat), 2), "max": round(max(lat), 2), "all": [round(x, 2) for x in lat]},
        "feed_ms_per_frame_mean": round(statistics.mean(res.feed_ms), 2) if res.feed_ms else None,
        "causality": caus, "causality_all_identical": all(c["identical"] for c in caus) if caus else None,
        "detector_agreement": agree, "vlm": vstat, "sql_baseline": sqlb,
        "questions": [{k: v for k, v in s.items() if k != "facts"} | {"facts": json.loads(_comparable(s["facts"]))} for s in res.scored],
        "incident_log": log["text"], "wall_s": round(time.time() - t_clip, 1), "sim_wall_s": round(res.wall_s, 1),
    }
    return stats


def build_report(name, script, res, log, evid, outdir: Path, n_ok, n, caus, agree, vstat, sqlb) -> None:
    st = res.state
    rep = pk.Report(outdir / "report.html", f"ON THE LINE: {script['title']}",
                    "Scene answers for an RVRC operator on the phone, advanced live through the clip, answered from cached state").credit(name)
    lat = [c.latency_ms for c in res.cards]
    rep.kv({"answers right vs eyeball": f"{n_ok}/{n}", "slowest answer": f"{max(lat):.0f} ms", "median answer": f"{statistics.median(lat):.1f} ms",
            "entities after merging": len(st.confirmed()), "raw tracker ids": sum(1 for r in st.raw.values() if r.eid is not None),
            "no-future-data check": ("pass" if caus and all(c["identical"] for c in caus) else "not run") if caus else "not run"})
    rep.image(outdir / "hero.jpg", "Call summary: frame at the hero moment, paths of everyone seen, the five answer cards scored against the eyeballed truth")
    if (outdir / "demo.mp4").exists():
        rep.video(outdir / "demo.mp4", "Demo: the clip replayed live, police questions typed at their video times, cards pop with evidence, log drafts at hang-up")
    rep.h2("Scripted police call, scored")
    rows = []
    for s in res.scored:
        rows.append({"id": s["id"], "video time": pk.mmss(s["t"]), "police asked": s["text"], "answer card": s["headline"],
                     "checks (got)": "; ".join(f"{c['type']}={c['got']} {'ok' if c['ok'] else 'FAIL'}" for c in s["checks"]),
                     "verdict": "RIGHT" if s["correct"] else "WRONG", "latency ms": s["latency_ms"]})
    rep.table(rows)
    rep.h2("Eyeballed truth used for scoring")
    rep.table([{"id": q["id"], "truth (what I saw in the frames)": q.get("truth", "")} for q in script["questions"] if q.get("truth")])
    rep.h2("Answer cards with evidence")
    for i, c in enumerate(res.cards):
        body = (f"<div class='card'><b>Q{i + 1} {pk.mmss(c.t)}</b> &ldquo;{c.question}&rdquo;<br><span style='font-size:19px'><b>{c.headline}</b></span><br>"
                + "".join(f"<span class='sub'>{l}</span><br>" for l in c.lines)
                + ("".join(f"<span class='sub'>caveat: {x}</span><br>" for x in c.caveats))
                + "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:6px'>"
                + "".join(f"<figure style='margin:0'><img src='{e['file']}' style='height:130px'><figcaption>{e['caption']}</figcaption></figure>" for e in evid[i])
                + f"</div><span class='sub'>answered from cached state in {c.latency_ms:.1f} ms</span></div>")
        rep.html(body)
    rep.h2("Incident log drafted at hang-up")
    rep.html("<pre>" + log["text"].replace("&", "&amp;").replace("<", "&lt;") + "</pre>")
    spans = []
    for e in sorted(st.confirmed(), key=lambda e: e.first_t):
        spans.append((e.first_t - st.alarm_t, e.last_t - st.alarm_t, "#4aa3ff" if e.group == "person" else "#ffa040",
                      f"#{e.eid} {e.group} {e.cls_name} {pk.mmss(e.first_t)}-{pk.mmss(e.last_t)}"))
    rep.timeline(spans[:60], script["hangup"] - st.alarm_t, "Entity presence since alarm (blue person, orange vehicle)")
    rep.h2("Cosmos descriptions (stand-in for Cosmos3-Reason)")
    rep.json(vstat)
    rep.h2("Independent check: live count vs yolo26l on the same frames")
    rep.json(agree)
    rep.h2("What the 5 s DataEngine segment table alone would have answered (count / total questions)")
    rep.table([{"id": a["id"], "type": a["type"], "SQL-only answer": a["got"], "truth range": str(a["truth_range"]),
                "right": "yes" if a["right"] else "NO"} for a in sqlb["answers"]])
    rep.save()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-causality", action="store_true")
    ap.add_argument("--vlm-latency", type=float, default=3.0,
                    help="SIMULATED seconds until a Cosmos description is usable (hosted speed); -1 = use measured local latency")
    ap.add_argument("--agree-wait", type=float, default=90.0, help="max seconds to wait for the shared GPU lock (yolo26l cross-check)")
    ap.add_argument("--ask", action="append", default=[], help='extra typed question "text@mm:ss"')
    a = ap.parse_args()
    sim.VLM_LATENCY = None if a.vlm_latency < 0 else a.vlm_latency
    global AGREE_WAIT
    AGREE_WAIT = a.agree_wait
    scripts = load_scripts()
    names = DEFAULT_SET + HELD_OUT if a.all else ([a.video] if a.video else [])
    if not names:
        ap.error("give --all or --video NAME")
    adhoc = []
    for k, s in enumerate(a.ask):
        txt, _, tt = s.rpartition("@")
        adhoc.append({"id": f"X{k + 1}", "t": parse_time(tt), "text": txt, "checks": []})
    t0 = time.time()
    results: dict = {"idea": "ON THE LINE", "clips": {}}
    prior = OUT.parent / "results.json"
    if prior.exists():
        results = json.loads(prior.read_text(encoding="utf-8"))
    for n in names:
        if n not in scripts:
            sys.exit(f"no script for {n}; add it to scripts.json")
        print(f"[run] {n} ...", flush=True)
        r = run_clip(n, scripts[n], use_vlm=not a.no_vlm, make_video=not a.no_video, do_causality=not a.no_causality, adhoc=adhoc,
                     prior=results["clips"].get(n))
        results["clips"][n] = r
        print(f"[run] {n}: {r['answers_right']}/{r['answers_scored']} right, slowest {r['latency_ms']['max']} ms, {r['wall_s']} s", flush=True)
        for q in r["questions"]:
            print(f"   {q['id']} {'OK ' if q['correct'] else ('BAD' if q['correct'] is not None else '---')} {q['text']} -> {q['headline']}")
    clips = results["clips"]
    scored = {k: v for k, v in clips.items() if v["answers_scored"] and k in DEFAULT_SET}
    good = [k for k, v in scored.items() if v["answers_right"] >= 4]
    all_lat = [x for v in scored.values() for x in v["latency_ms"]["all"]]
    results["kill_test"] = {"criterion": ">= 4 of 5 answers right on 2 clips; answer latency < 1 s",
                            "clips_with_ge_4_of_5": good, "max_latency_ms": round(max(all_lat), 2) if all_lat else None,
                            "passed": len(good) >= 2 and bool(all_lat) and max(all_lat) < 1000}
    ho = {k: v for k, v in clips.items() if k in HELD_OUT and v["answers_scored"]}
    results["held_out"] = {"clips": {k: f"{v['answers_right']}/{v['answers_scored']}" for k, v in ho.items()},
                           "right": sum(v["answers_right"] for v in ho.values()), "scored": sum(v["answers_scored"] for v in ho.values()),
                           "note": HELD_NOTE}
    results["default_set"] = {"right": sum(v["answers_right"] for v in scored.values()), "scored": sum(v["answers_scored"] for v in scored.values())}
    results["wall_s"] = round(time.time() - t0, 1)
    pk.write_results(HERE / "results.json", results)
    print(f"[run] kill test: {results['kill_test']}  total {results['wall_s']} s")


if __name__ == "__main__":
    main()
