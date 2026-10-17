"""HIDE & SEEK: can the index tell our clips apart? (prototype 12)

  python run.py --all                 build the 64-clip archive, play the game with every clip as the secret, write
                                      out/hide_and_seek.html, out/ALL/*, out/<video>/*, results.json
  python run.py --video lisbon_parking   re-render one video's report (+ hero) from the cached archive
  options: --n-clips 64 --seed 12 --no-vlm --rebuild --noise 0.05 --secret c27
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hs_core as hc  # noqa: E402
import render  # noqa: E402

pk = hc.pk
OUT, WORK = hc.OUT, hc.WORK


# ------------------------------------------------------------------------------------------------- archive


def load_archive(n: int, seed: int, use_vlm: bool, rebuild: bool) -> list[dict[str, Any]]:
    f = WORK / f"archive_n{n}_s{seed}_{'vlm' if use_vlm else 'novlm'}.json"
    if f.exists() and not rebuild:
        return json.loads(f.read_text(encoding="utf-8"))
    t0 = time.time()
    clips = hc.build_archive(n, seed, use_vlm)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(clips, indent=1), encoding="utf-8")
    print(f"archive built in {time.time() - t0:.0f}s -> {f.name}")
    return clips


def cut_clips(clips: list[dict[str, Any]]) -> None:
    d = OUT / "clips"
    for c in clips:
        dst = d / f"{c['id']}.mp4"
        if not dst.exists():
            pk.cut(c["video"], c["t0"], c["t1"], dst, width=480)


# ------------------------------------------------------------------------------------------------ analysis


def trace(eng: hc.Engine, secret: int) -> list[dict[str, Any]]:
    """Full question/answer ladder for one secret, with candidate counts."""
    cand = list(range(len(eng.clips)))
    asked: set[int] = set()
    out = []
    while len(cand) > 1:
        qi = eng.best(cand, asked)
        if qi is None:
            break
        a = bool(eng.ans[secret, qi])
        new = [i for i in cand if bool(eng.ans[i, qi]) == a]
        out.append({"qi": qi, "text": eng.qs[qi]["text"], "a": a, "before": len(cand), "after": len(new)})
        asked.add(qi)
        cand = new
    return out


def blind_reason(c: dict[str, Any]) -> str:
    if not c["extra"]["blind"]:
        return ""
    return "too dark" if c["attrs"]["brightness"] == "dark" else "no COCO objects"


ATTRS_V1 = [a for a in hc.ATTRS if a != "words"] + ["colour"]  # the first version: 15 attributes incl. dominant colour, no caption keywords


def protocol_stats(eng: hc.Engine, n: int, flip: float = 0.0, reps: int = 1, seed: int = 5) -> dict[str, Any]:
    """Play the page protocol (hard elimination -> 'Is it this one?' -> soft recovery) for every clip as the secret."""
    rng = np.random.default_rng(seed)
    rs = [eng.run_protocol(i, flip, rng) for _ in range(reps) for i in range(n)]
    return {"answer_flip_prob": flip, "plays": len(rs), "found_rate": float(np.mean([r["found"] for r in rs])),
            "mean_questions": float(np.mean([r["n_questions"] for r in rs])),
            "mean_guesses": float(np.mean([r["guesses"] for r in rs])),
            "mean_questions_plus_guesses": float(np.mean([r["n_questions"] + r["guesses"] for r in rs])),
            "found_on_first_guess": float(np.mean([r["found"] and r["guesses"] == 1 for r in rs])),
            "first_guess_was_a_twin_pick": float(np.mean([r["first_group"] is not None and r["first_group"] > 1 for r in rs])),
            "worst_questions": int(max(r["n_questions"] for r in rs))}


def keyword_eyeball(clips: list[dict[str, Any]], eng: hc.Engine) -> dict[str, Any] | None:
    """Caption-keyword questions vs my eyeballed truth, and the page protocol played by a judge who answers those questions from what
    is visible (unsure -> 'Not sure') instead of from the index."""
    f = HERE / "gt.json"
    if not f.exists() or "keywords" not in json.loads(f.read_text(encoding="utf-8")):
        return None
    gt = json.loads(f.read_text(encoding="utf-8"))["keywords"]
    ids = [c["id"] for c in clips]
    out: dict[str, Any] = {"per_keyword": {}}
    truth: dict[tuple[int, int], bool | None] = {}
    for qi, q in enumerate(eng.qs):
        if q["attr"] != "words" or q["val"] not in gt:
            continue
        g = gt[q["val"]]
        tp = fp = fn = tn = unsure = 0
        for i, cid in enumerate(ids):
            if cid in g["unsure"]:
                truth[(i, qi)] = None
                unsure += 1
                continue
            vis = cid in g["true"]
            truth[(i, qi)] = vis
            idx = bool(eng.ans[i, qi])
            tp += vis and idx
            fp += (not vis) and idx
            fn += vis and (not idx)
            tn += (not vis) and (not idx)
        out["per_keyword"][q["val"]] = {"question": q["text"], "index_yes_and_visible": tp, "index_yes_but_not_visible": fp,
                                        "visible_but_index_no": fn, "unsure_clips": unsure,
                                        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    judge = lambda i, qi: truth.get((i, qi), bool(eng.ans[i, qi]))  # noqa: E731
    rs = [eng.run_protocol(i, judge=judge) for i in range(len(clips))]
    out["eyeball_judge_protocol"] = {
        "labelled_keywords": sorted(gt), "found_rate": float(np.mean([r["found"] for r in rs])),
        "mean_questions": float(np.mean([r["n_questions"] for r in rs])), "mean_guesses": float(np.mean([r["guesses"] for r in rs])),
        "mean_questions_plus_guesses": float(np.mean([r["n_questions"] + r["guesses"] for r in rs])),
        "found_on_first_guess": float(np.mean([r["found"] and r["guesses"] == 1 for r in rs])),
        "not_found": [clips[r["secret"]]["id"] for r in rs if not r["found"]]}
    # same judge, but strict hard elimination without confirmation (what v1 would have done)
    ok = 0
    for i in range(len(clips)):
        cand, asked = list(range(len(clips))), set()
        while len(cand) > 1:
            qi = eng.best(cand, asked)
            if qi is None:
                break
            a = judge(i, qi)
            asked.add(qi)
            if a is None:
                continue
            cand = [k for k in cand if bool(eng.ans[k, qi]) == a]
        ok += cand == [i]
    out["eyeball_judge_strict_found_rate"] = ok / len(clips)
    return out


def analyse(clips: list[dict[str, Any]], noise: float, full: bool = True) -> dict[str, Any]:
    hc.add_words(clips)
    qs = hc.make_questions(clips)
    eng = hc.Engine(clips, qs)
    plays = [eng.play(i) for i in range(len(clips))]
    for c, p in zip(clips, plays):
        c["play"] = {"n_questions": p["n_questions"], "found": p["found"],
                     "final": [clips[i]["id"] for i in p["final"]]}
    n_found = sum(p["found"] for p in plays)
    nq = [p["n_questions"] for p in plays]
    tw = hc.twins(clips)
    expected_hit = float(np.mean([(1 / len(p["final"])) if p["secret"] in p["final"] else 0.0 for p in plays]))
    proto = protocol_stats(eng, len(clips))
    stats = {"n_clips": len(clips), "n_questions_available": len(qs), "n_found": n_found,
             "success_rate": n_found / len(clips), "expected_hit_rate_if_guessing_inside_twin_group": expected_hit,
             "mean_questions_all": float(np.mean(nq)),
             "mean_questions_found": float(np.mean([p["n_questions"] for p in plays if p["found"]])) if n_found else None,
             "median_questions": float(np.median(nq)), "max_questions": int(max(nq)),
             "question_histogram": {str(k): int(v) for k, v in sorted(zip(*np.unique(nq, return_counts=True)))},
             "unique_vectors": len({hc.attr_vector(c) for c in clips}), "twin_groups": len(tw),
             "clips_in_twin_groups": sum(len(t) for t in tw),
             "blind": sum(c["extra"]["blind"] for c in clips), "floor_log2": math.log2(len(clips)),
             "protocol_found": proto["found_rate"], "protocol_mean_total": proto["mean_questions_plus_guesses"],
             "protocol_mean_questions": proto["mean_questions"]}
    first_q = eng.best(list(range(len(clips))), set())
    stats["first_question"] = qs[first_q]["text"]
    use: dict[str, int] = {}
    for i in range(len(clips)):
        for t in trace(eng, i):
            use[t["text"]] = use.get(t["text"], 0) + 1
    stats["question_usage_top"] = sorted(use.items(), key=lambda kv: -kv[1])[:10]
    out = {"qs": qs, "eng": eng, "plays": plays, "stats": stats, "twins": tw, "protocol": proto, "ablation": {}, "noise": {}, "scaling": {},
           "keyword_eyeball": keyword_eyeball(clips, eng)}
    if not full:  # quick single-video render: skip the corpus-wide sensitivity studies
        return out
    # ablation: which pipeline layer buys what (strict engine, same questions logic, fewer attributes)
    layers = (("YOLO tracks only", [a for a in hc.ATTRS if hc.SOURCE[a] == "yolo"]),
              ("+ frame brightness", [a for a in hc.ATTRS if hc.SOURCE[a] in ("yolo", "frames")]),
              ("+ Cosmos tags (indoor/outdoor, setting)", [a for a in hc.ATTRS if a != "words"]),
              ("+ Cosmos caption keywords = full (page)", list(hc.ATTRS)),
              ("first version: tags + dominant colour, no keywords", ATTRS_V1))
    for name, attrs in layers:
        e = hc.evaluate(clips, attrs)
        out["ablation"][name] = {k: e[k] for k in ("n_questions_available", "success_rate", "mean_questions_all", "max_questions",
                                                   "unique_vectors", "clips_in_twin_groups")}
    # noisy-answer robustness (SIMULATED wrong answers): strict hard elimination vs the page protocol with confirmation + recovery
    reps = 20
    for flip in (0.0, noise, 2 * noise):
        e = hc.evaluate(clips, None, flip=flip, reps=reps, seed=7)
        out["noise"][f"strict_flip{flip:.2f}"] = {"engine": "strict hard elimination, no confirmation", "answer_flip_prob": flip,
                                                  "success_rate": e["success_rate"], "mean_questions_all": e["mean_questions_all"],
                                                  "mean_guesses": 0.0}
        pr = protocol_stats(eng, len(clips), flip, reps if flip else 1, seed=11)
        out["noise"][f"page_flip{flip:.2f}"] = {"engine": "page protocol (confirm + soft recovery, max 3 guesses)", "answer_flip_prob": flip,
                                                "success_rate": pr["found_rate"], "mean_questions_all": pr["mean_questions"],
                                                "mean_guesses": pr["mean_guesses"]}
    # archive-size sensitivity: random sub-archives of the same 64 attribute rows (strict engine)
    rng = np.random.default_rng(3)
    for m in (16, 32, 48, 64):
        rs = []
        for _ in range(1 if m == len(clips) else 60):
            idx = sorted(rng.choice(len(clips), m, replace=False))
            rs.append(hc.evaluate([clips[i] for i in idx], None))
        out["scaling"][str(m)] = {"success_rate": float(np.mean([x["success_rate"] for x in rs])),
                                  "mean_questions_all": float(np.mean([x["mean_questions_all"] for x in rs])),
                                  "floor_log2": math.log2(m), "draws": len(rs)}
    return out


def attribute_accuracy(clips: list[dict[str, Any]]) -> dict[str, Any] | None:
    f = HERE / "gt.json"
    if not f.exists():
        return None
    gt = json.loads(f.read_text(encoding="utf-8"))
    by = {c["id"]: c for c in clips}
    res: dict[str, Any] = {}

    def rate(name: str, pairs: list[tuple[bool, str]]) -> None:
        if pairs:
            res[name] = {"n": len(pairs), "agree": sum(ok for ok, _ in pairs), "rate": sum(ok for ok, _ in pairs) / len(pairs),
                         "misses": [m for ok, m in pairs if not ok]}

    io, st, night, night2, day, person, veh = [], [], [], [], [], [], []
    for cid, g in gt["clips"].items():
        if cid not in by:
            continue
        a = by[cid]["attrs"]
        if "indoor_outdoor" in g:
            io.append((a["indoor_outdoor"] == g["indoor_outdoor"], f"{cid}: index {a['indoor_outdoor']} / eyeball {g['indoor_outdoor']}"))
        if "setting" in g:
            st.append((a["setting"] == g["setting"], f"{cid}: index {a['setting']} / eyeball {g['setting']}"))
        if g.get("night") is True:
            night.append((a["brightness"] == "dark", f"{cid}: index {a['brightness']} (luma {by[cid]['extra']['luma']}) / eyeball night"))
            night2.append((a["brightness"] in ("dark", "dim"), f"{cid}: index {a['brightness']} / eyeball night"))
        if g.get("night") is False:
            day.append((a["brightness"] == "bright", f"{cid}: index {a['brightness']} (luma {by[cid]['extra']['luma']}) / eyeball daylight"))
        if g.get("person_visible"):
            person.append((a["persons"] != "0", f"{cid}: person visible at mid-frame, index persons={a['persons']}"))
        if g.get("vehicle_visible"):
            veh.append((a["vehicles"] != "0", f"{cid}: vehicle visible at mid-frame, index vehicles={a['vehicles']}"))
    rate("indoor_outdoor (Cosmos)", io)
    rate("setting (Cosmos)", st)
    rate("night clips -> brightness 'dark' (recall)", night)
    rate("night clips -> 'dark' or 'dim' (recall)", night2)
    rate("daylight clips -> 'bright'", day)
    rate("person visible -> persons>0 (recall)", person)
    rate("vehicle visible -> vehicles>0 (recall)", veh)
    return res


# ------------------------------------------------------------------------------------------------- outputs


def chips_of(c: dict[str, Any]) -> list[str]:
    a = c["attrs"]
    out = []
    out.append(f"{a['persons']} people" if a["persons"] != "0" else "no people")
    if a["vehicles"] != "0":
        out.append(f"{a['vehicles']} vehicles")
    if a["bike"] in (True, "True"):
        out.append("bike")
    out.append(a["motion"])
    out.append(a["brightness"])
    if a["setting"] != "unknown":
        out.append(a["setting"])
    out += [w for w in str(a.get("words", "")).split(",") if w][:4]
    return out


def small_thumb(p: Path) -> str:
    import base64
    import cv2
    im = cv2.imread(str(p))
    s = 240 / max(im.shape[:2]) if im.shape[0] > im.shape[1] else 240 / im.shape[1]
    im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_game(clips, ana, thumbs_dir: Path) -> Path:
    import base64
    qs = ana["qs"]
    data = {
        "clips": [{"id": c["id"], "label": f"{c['video'].replace('_', ' ')} {c['t0']:.0f}-{c['t1']:.0f}s",
                   "thumb": small_thumb(thumbs_dir / f"{c['id']}.jpg"),
                   "video": f"clips/{c['id']}.mp4", "attrs": {k: str(c["attrs"][k]) for k in hc.ATTRS},
                   "chips": chips_of(c), "caption": c["extra"]["caption"]} for c in clips],
        "params": hc.PROTO,
        "questions": [{**{k: q[k] for k in ("id", "attr", "op", "val", "text")}, "ans": [bool(x) for x in ana["eng"].ans[:, i]]}
                      for i, q in enumerate(qs)],
        "stats": {**{k: ana["stats"][k] for k in ("success_rate", "mean_questions_all", "max_questions", "twin_groups",
                                                   "clips_in_twin_groups", "blind", "protocol_found", "protocol_mean_total")}},
        "twins": [[{"id": cid, "label": next(c for c in clips if c["id"] == cid)["video"].replace("_", " "),
                    "thumb": None,
                    "chips": chips_of(next(c for c in clips if c["id"] == g[0]))} for cid in g] for g in ana["twins"]],
        "blind": [],
    }
    byid = {c["id"]: c for c in data["clips"]}
    for grp in data["twins"]:
        for m in grp:
            m["thumb"] = byid[m["id"]]["thumb"]
            m["label"] = byid[m["id"]]["label"]
    for c in clips:
        if c["extra"]["blind"]:
            data["blind"].append({"id": c["id"], "label": byid[c["id"]]["label"], "thumb": byid[c["id"]]["thumb"], "why": blind_reason(c)})
    html = (HERE / "game_template.html").read_text(encoding="utf-8")
    html = html.replace("__ENGINE__", (HERE / "engine.js").read_text(encoding="utf-8"))
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__N__", str(len(clips)))
    (OUT / "hide_and_seek.html").write_text(html, encoding="utf-8")
    (WORK / "game_data.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return OUT / "hide_and_seek.html"


def contact_sheet(clips: list[dict[str, Any]], ids: list[str], path: Path, cols: int = 4, tile_w: int = 300) -> None:
    import cv2
    by = {c["id"]: c for c in clips}
    imgs = [cv2.imread(str(WORK / "thumbs" / f"{i}.jpg")) for i in ids]
    lab = [f"{i} {by[i]['video'][:16]} {by[i]['t0']:.0f}s" for i in ids]
    pk.save_jpg(pk.sheet(imgs, lab, cols=cols, tile_w=tile_w), path, 82)


def render_all(clips, ana, only: str | None, noise: float, acc) -> dict[str, Any]:
    thumbs = WORK / "thumbs"
    stats = ana["stats"]
    plays = ana["plays"]
    eng = ana["eng"]
    twin_ids = {cid for g in ana["twins"] for cid in g}
    blind_ids = {c["id"] for c in clips if c["extra"]["blind"]}
    by = {c["id"]: c for c in clips}
    kill = {"criterion": "mean questions <= 7 AND >= 90% found (every clip as the secret)",
            "mean_questions": stats["mean_questions_all"], "found_rate": stats["success_rate"]}
    kill["passed"] = bool(kill["mean_questions"] <= 7 and kill["found_rate"] >= 0.9)
    kill["short"] = f"mean {kill['mean_questions']:.2f} Q (<=7), found {100 * kill['found_rate']:.0f}% (>=90%)"
    kill["page_protocol"] = ana["protocol"]
    v1 = ana["ablation"].get("first version: tags + dominant colour, no keywords")
    kill["v1_found"] = v1["success_rate"] if v1 else None
    # choose the showcase secret: found, with the median question count, most people/vehicles visible
    med = int(round(stats["median_questions"]))
    cands = [i for i, p in enumerate(plays) if p["found"] and p["n_questions"] == med] or [i for i, p in enumerate(plays) if p["found"]] or [0]
    secret = max(cands, key=lambda i: by[clips[i]["id"]]["extra"]["n_tracks"])
    if getattr(only, "secret", None):
        pass
    per_video: dict[str, Any] = {}
    videos = hc.DEFAULT_VIDEOS if only is None else [only]
    for v in videos:
        items, nf, qlist = [], 0, []
        for i, c in enumerate(clips):
            if c["video"] != v:
                continue
            p = plays[i]
            partners = [x for x in c["play"]["final"] if x != c["id"]]
            it = {"id": c["id"], "t0": c["t0"], "t1": c["t1"], "chips": chips_of(c), "n_q": p["n_questions"], "found": p["found"],
                  "twins": partners, "blind": bool(c["extra"]["blind"]), "why": blind_reason(c)}
            items.append(it)
            nf += p["found"]
            qlist.append(p["n_questions"])
        if not items:
            continue
        summ = {"n": len(items), "n_found": nf, "mean_q": float(np.mean(qlist))}
        vdir = OUT / v
        render.hero_video(vdir / "hero.jpg", v, items, thumbs, summ, pk.credit(v))
        write_video_report(v, items, clips, ana, vdir, summ)
        per_video[v] = {"n_clips": len(items), "n_found_uniquely": nf, "found_rate": nf / len(items),
                        "mean_questions": summ["mean_q"], "max_questions": max(qlist),
                        "clips_with_attribute_twin": sum(1 for it in items if it["twins"]),
                        "invisible_clips": sum(1 for it in items if it["blind"]),
                        "verdict": ("ALL CLIPS FINDABLE" if nf == len(items) else f"{len(items) - nf} OF {len(items)} CLIPS AMBIGUOUS"),
                        "clips": [{"id": it["id"], "t0": it["t0"], "n_questions": it["n_q"], "found": it["found"], "twins": it["twins"],
                                   "blind": it["blind"]} for it in items]}
    if only is None:
        tr = trace(eng, secret)
        (OUT / "ALL").mkdir(parents=True, exist_ok=True)
        render.hero_corpus(OUT / "ALL" / "hero.jpg", clips, thumbs, tr, secret, stats, twin_ids, blind_ids, kill)
        write_corpus_report(clips, ana, kill, acc, tr, secret, per_video)
    return {"per_video": per_video, "kill": kill, "showcase_secret": clips[secret]["id"]}


def write_video_report(v: str, items, clips, ana, vdir: Path, summ) -> None:
    by = {c["id"]: c for c in clips}
    for f in vdir.glob("twins_*.jpg"):
        f.unlink()
    rep = pk.Report(vdir / "report.html", f"Hide & Seek: {v.replace('_', ' ')}",
                    "Can the index find each of this video's archive clips, and tell it from every other clip?")
    rep.credit(v)
    rep.image(vdir / "hero.jpg", "Hero")
    rep.kv({"clips in archive": summ["n"], "found uniquely": f"{summ['n_found']}/{summ['n']}",
            "mean questions": f"{summ['mean_q']:.2f}", "corpus mean": f"{ana['stats']['mean_questions_all']:.2f}"})
    rep.h2("Clips and what the index knows about them")
    rows = []
    for it in items:
        c = by[it["id"]]
        a = c["attrs"]
        rows.append({"clip": f"{it['id']} ({c['t0']:.0f}-{c['t1']:.0f}s)", "people": a["persons"], "vehicles": a["vehicles"],
                     "motion": a["motion"], "light": a["brightness"], "keywords": a["words"], "indoor/outdoor": a["indoor_outdoor"],
                     "setting": a["setting"], "questions": it["n_q"], "result": ("found" if it["found"] else
                     ("INVISIBLE " + blind_reason(c) + "; " if it["blind"] else "") + "twin of " + ", ".join(it["twins"])),
                     "Cosmos caption": c["extra"]["caption"]})
    rep.table(rows)
    rep.h2("How the engine found each one (secret = this clip)")
    eng = ana["eng"]
    for it in items:
        i = next(k for k, c in enumerate(clips) if c["id"] == it["id"])
        tr = trace(eng, i)
        rep.html("<p><b>" + it["id"] + "</b>: " + (" &rarr; ".join(f"{t['text']} <i>{'yes' if t['a'] else 'no'}</i> ({t['after']})" for t in tr)
                                                   if tr else "no useful question") + "</p>")
    tw = [it for it in items if it["twins"]]
    if tw:
        rep.h2("Attribute twins (the index cannot tell these apart)")
        for it in tw:
            ids = [it["id"]] + it["twins"]
            p = vdir / f"twins_{it['id']}.jpg"
            contact_sheet(clips, ids[:6], p, cols=min(3, len(ids)), tile_w=380)
            rep.image(p, f"{', '.join(ids)}: identical on all {len(hc.ATTRS)} attributes")
    rep.save()


def write_corpus_report(clips, ana, kill, acc, tr, secret, per_video) -> None:
    stats = ana["stats"]
    d = OUT / "ALL"
    for f in d.glob("twin_group_*.jpg"):  # stale contact sheets from an earlier run with more twin groups
        f.unlink()
    by = {c["id"]: c for c in clips}
    rep = pk.Report(d / "report.html", "Hide & Seek: can our index tell our clips apart?",
                    f"{len(clips)} five-second clips sampled from {len(per_video)} videos; every clip played as the secret.")
    for v in per_video:
        rep.credit(v)
    rep.image(d / "hero.jpg", "Hero")
    rep.html('<p><a href="../hide_and_seek.html"><b>Open the interactive game</b></a> (out/hide_and_seek.html; clips play from out/clips/).</p>')
    rep.kv({"found uniquely": f"{stats['n_found']}/{stats['n_clips']} ({100 * stats['success_rate']:.0f}%)",
            "mean questions": f"{stats['mean_questions_all']:.2f}", "max questions": stats["max_questions"],
            "attribute twins": f"{stats['clips_in_twin_groups']} clips / {stats['twin_groups']} groups",
            "invisible": stats["blind"], "kill test": "PASS" if kill["passed"] else "FAIL"})
    rep.text(f"Kill test: {kill['criterion']} -> {kill['short']} -> {'PASSED' if kill['passed'] else 'FAILED'}.")
    rep.h2("Where the questions go")
    rep.table([{"questions needed": k, "clips": v} for k, v in stats["question_histogram"].items()])
    rep.table([{"question": q, "used (clip games)": n} for q, n in stats["question_usage_top"]])
    rep.h2("Which layer of the pipeline buys discrimination (ablation)")
    rep.table([{"attributes": k, **v} for k, v in ana["ablation"].items()])
    rep.h2("Lookalikes: identical attribute vectors")
    for k, g in enumerate(ana["twins"]):
        p = d / f"twin_group_{k + 1}.jpg"
        contact_sheet(clips, g[:8], p, cols=min(4, len(g)), tile_w=300)
        rep.image(p, f"Group {k + 1} ({len(g)} clips): " + ", ".join(chips_of(by[g[0]])))
    if not ana["twins"]:
        rep.text("None.")
    rep.h2("Invisible to search: detectors find nothing")
    bl = [c["id"] for c in clips if c["extra"]["blind"]]
    if bl:
        p = d / "blind.jpg"
        contact_sheet(clips, bl[:16], p, cols=4, tile_w=300)
        rep.image(p, "; ".join(f"{i}: {blind_reason(by[i])}" for i in bl[:16]))
    else:
        rep.text("None.")
    pr = ana["protocol"]
    rep.h2("The page protocol: questions, then 'Is it this one?', then soft recovery")
    rep.text(f"Truthful judge, every clip as the secret: found {100 * pr['found_rate']:.0f}% within {hc.PROTO['max_guesses']} guesses, "
             f"{pr['mean_questions']:.2f} questions + {pr['mean_guesses']:.2f} guesses on average; "
             f"{100 * pr['found_on_first_guess']:.0f}% on the first guess, {100 * pr['first_guess_was_a_twin_pick']:.0f}% of games end in a "
             f"pick between lookalikes (one extra guess question, e.g. c13 or c14?).")
    ke = ana["keyword_eyeball"]
    if ke:
        rep.h2("Caption-keyword questions vs what is visible (my eyeballed labels)")
        rep.table([{"keyword": k, **{kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}} for k, v in ke["per_keyword"].items()])
        e = ke["eyeball_judge_protocol"]
        rep.text(f"A judge who answers those {len(e['labelled_keywords'])} questions from what is visible (unsure -> Not sure) and every other question "
                 f"from the index: page protocol finds {100 * e['found_rate']:.0f}% ({e['mean_questions']:.1f} questions + {e['mean_guesses']:.1f} guesses); "
                 f"the strict engine without confirmation would find {100 * ke['eyeball_judge_strict_found_rate']:.0f}%. Not found: {', '.join(e['not_found']) or 'none'}.")
    rep.h2("Robustness to wrong answers (SIMULATED answer flips)")
    rep.table([{"case": k, **{kk: vv for kk, vv in v.items()}} for k, v in ana["noise"].items()])
    rep.h2("Archive size (random sub-archives of the same 64 rows)")
    rep.table([{"clips": k, **v} for k, v in ana["scaling"].items()])
    if acc:
        rep.h2("Index attributes vs my own eyeballed ground truth")
        rep.table([{"check": k, "n": v["n"], "agree": v["agree"], "rate": v["rate"]} for k, v in acc.items()])
        for k, v in acc.items():
            if v["misses"]:
                rep.html(f"<p><b>{k}</b> misses: " + "; ".join(v["misses"][:12]) + "</p>")
    rep.h2("Per video")
    rep.table([{"video": v, "clips": r["n_clips"], "found": r["n_found_uniquely"], "mean Q": r["mean_questions"],
                "twins": r["clips_with_attribute_twin"], "invisible": r["invisible_clips"], "verdict": r["verdict"]}
               for v, r in per_video.items()])
    rep.save()


# ------------------------------------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--n-clips", type=int, default=64)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--no-vlm", action="store_true", help="skip Cosmos attributes (indoor/outdoor, setting, caption)")
    ap.add_argument("--rebuild", action="store_true", help="recompute the archive instead of using out/_work cache")
    ap.add_argument("--noise", type=float, default=0.05, help="answer-flip probability for the SIMULATED robustness test")
    args = ap.parse_args()
    if not args.all and not args.video:
        ap.error("pass --all or --video NAME")
    if args.video and args.video not in hc.DEFAULT_VIDEOS:
        ap.error(f"unknown video {args.video!r}; one of {', '.join(hc.DEFAULT_VIDEOS)}")
    t0 = time.time()
    clips = load_archive(args.n_clips, args.seed, not args.no_vlm, args.rebuild)
    ana = analyse(clips, args.noise, full=args.all)
    acc = attribute_accuracy(clips)
    cut_clips(clips)
    build_game(clips, ana, WORK / "thumbs")
    res = render_all(clips, ana, None if args.all else args.video, args.noise, acc)
    stats = ana["stats"]
    results = {"idea": "12_hide_and_seek", "n_clips": len(clips), "seed": args.seed, "vlm": not args.no_vlm,
               "corpus": stats, "kill_test": res["kill"], "ablation": ana["ablation"], "protocol": ana["protocol"], "protocol_params": hc.PROTO, "keyword_eyeball": ana["keyword_eyeball"], "caption_lexicon": sorted(hc.KEYWORDS),
               "noise_simulated": ana["noise"], "archive_size_scaling": ana["scaling"],
               "attribute_vs_eyeball": acc, "per_video": res["per_video"],
               "twin_groups": [{"clips": g, "attrs": {a: by_attr for a, by_attr in
                                                      next(c for c in clips if c["id"] == g[0])["attrs"].items()}}
                               for g in ana["twins"]],
               "invisible_clips": [{"id": c["id"], "video": c["video"], "t0": c["t0"], "why": blind_reason(c)}
                                   for c in clips if c["extra"]["blind"]],
               "clips": [{"id": c["id"], "video": c["video"], "seg": c["seg"], "t0": c["t0"], "t1": c["t1"],
                          "attrs": c["attrs"], "extra": c["extra"], "play": c["play"]} for c in clips],
               "showcase_secret": res["showcase_secret"], "seconds": round(time.time() - t0, 1)}
    if args.all:
        pk.write_results(HERE / "results.json", results)
    print(f"mean questions {stats['mean_questions_all']:.2f} | found {stats['n_found']}/{stats['n_clips']} | "
          f"twins {stats['clips_in_twin_groups']} in {stats['twin_groups']} groups | blind {stats['blind']} | "
          f"kill test {'PASS' if res['kill']['passed'] else 'FAIL'} | {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
