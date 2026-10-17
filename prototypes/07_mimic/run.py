#!/usr/bin/env python
"""MIMIC: find the object that moved like this.  python run.py --all | --video NAME | --path "x,y;x,y;..." [--rot]

Pipeline: cached YOLO26s+ByteTrack tracks -> camera-motion windows (cuts + compensation) -> tracklets -> 32-point motion
descriptors -> table (duckdb = VastDB stand-in, SQL for metadata filters) -> query (drawn path / preset) -> cosine kNN
shortlist -> DTW re-rank -> top 8 with thumbnails, trajectory overlays and a clip of the best.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mimic_core as mc  # noqa: E402
import mimic_render as mr  # noqa: E402

pk = mc.pk
OUT = HERE / "out"
PRESET_ORDER = ["u_turn", "left_to_right", "stop_and_go", "loiter", "diagonal"]
EVAL4 = ["u_turn", "left_to_right", "stop_and_go", "loiter"]      # the four presets of the kill test
GT_FILE = HERE / "ground_truth.json"


def load_gt() -> dict:
    return json.loads(GT_FILE.read_text(encoding="utf-8")) if GT_FILE.exists() else {"labels": {}, "calibration": {}, "notes": {}}


def verdict(cost: float, cal: dict) -> tuple[str, tuple]:
    s, p = cal.get("strong"), cal.get("plausible")
    if s is None:
        return "UNCALIBRATED", mr.AMBER
    if cost <= s:
        return "MATCH", mr.GREEN
    if cost <= p:
        return "MAYBE", mr.AMBER
    return "NO MATCH", mr.RED


# ------------------------------------------------------------------------------------------------ the table
def make_db(d: dict):
    df = d["df"]
    light = df.drop(columns=["pos", "shape", "speed", "turn_curve"])
    clips = pd.DataFrame([{"video": c, **{k: v for k, v in m.items()}} for c, m in d["meta"].items()])
    return pk.vastdb({"tracklets": light, "clips": clips})


def sql_keys(con, video: str | None = None, classes: tuple | None = None) -> list[str]:
    """Metadata filter in the VastDB SQL subset (no IN / GROUP BY / JOIN / window): the candidates for the kNN."""
    where = ["dur >= 1.5", "len >= 0.02"]
    if video:
        where.append(f"video = '{video}'")
    if classes:
        where.append("(" + " OR ".join(f"cls = '{c}'" for c in classes) + ")")
    return con.execute("SELECT key FROM tracklets WHERE " + " AND ".join(where) + " ORDER BY t0").df()["key"].tolist()


# ------------------------------------------------------------------------------------------------ composition
def letterbox(img: np.ndarray, w: int, h: int) -> np.ndarray:
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), mr.PANEL, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def paste(canvas, img, x, y):
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img


def make_hero(d: dict, q: mc.Query, res: pd.DataFrame, title: str, subtitle: str, cal: dict, foot: str,
              timeline: tuple | None = None, note: str = "", video: str | None = None, badges: list | None = None,
              labels: dict | None = None) -> np.ndarray:
    """1280x720 'product' image: what you drew -> what MIMIC found, verdict stamp, runner-ups, where in the footage."""
    W, H = 1280, 720
    img = np.full((H, W, 3), mr.BG, np.uint8)
    mr.text(img, "MIMIC", (24, 52), 1.5, mr.ACC, 2)
    mr.text(img, "draw how it moved  >  find who moved that way", (210, 52), 0.7, mr.MUTED)
    ts = mr.tsize(title, 0.7)[0]
    mr.text(img, title, (W - 24 - ts, 34), 0.7, mr.FG)
    ss = mr.tsize(subtitle, 0.5)[0]
    mr.text(img, subtitle, (W - 24 - ss, 60), 0.5, mr.MUTED)
    plaus = cal.get("plausible", 1e9)

    def hc(key):                                             # my hand-check of a result (only when labels were passed)
        return "" if not labels or key not in labels else (" | hand-check: RIGHT" if labels[key] else " | hand-check: WRONG")
    best_rot = int(res.rot_deg.iloc[0]) if (len(res) and "rot_deg" in res and q.any_rot) else 0
    temporal = q.name in ("stop_and_go", "loiter")
    sk = mr.sketch_panel(q.pts, 340, 252, "YOU DRAW", q.name.replace("_", " "), q.aspect, marks=mc.preset_marks(q.name) if q.name in mc.PRESETS else None,
                         speed=q.speed if q.speed_w > 0 else None, rot=best_rot)
    paste(img, sk, 24, 100)
    cv2.arrowedLine(img, (372, 226), (410, 226), mr.ACC, 4, cv2.LINE_AA, tipLength=0.5)
    if len(res):
        best = res.iloc[0]
        v, col = verdict(best.cost, cal)
        weak = v == "NO MATCH"
        rot_txt = f"  turned {best_rot} deg" if best_rot else ""
        big = mr.card(d, best.key, 640, tag=(f"CLOSEST (not a match)  {best.cls}  {pk.mmss(best.t0)}-{pk.mmss(best.t1)}" if weak
                                             else f"TOP MATCH  {best.cls}  {pk.mmss(best.t0)}-{pk.mmss(best.t1)}") + rot_txt,
                      sub=f"{best.video}  cost {best.cost:.2f}{hc(best.key)}", pauses=temporal, dim=weak)
        big = letterbox(big, 540, 318)
        if q.name == "stop_and_go":
            mr.spark(big, 8, 318 - 86, 170, 50, best.speed, (60, 200, 255), "speed: this match (grey = sketch)", ref=q.speed)
        paste(img, big, 424, 100)
        cv2.rectangle(img, (424, 100), (964, 418), col, 3)
        shown = 1
        for k in (1, 2):
            if len(res) > k and res.iloc[k].cost <= plaus:
                r = res.iloc[k]
                c = mr.card(d, r.key, 400, tag=f"#{k + 1} {r.cls} {pk.mmss(r.t0)}", sub=f"cost {r.cost:.2f}{hc(r.key)}", pauses=temporal)
                paste(img, letterbox(c, 280, 152), 976, 100 + (k - 1) * 166)
                shown += 1
        mr.stamp(img, v, col, (14, 352), 0.85, -6)
        mr.text(img, f"cost {best.cost:.2f}", (200, 392), 0.6, col)
        mr.text(img, f"top of {len(res)}, {best.dur:.0f} s track", (200, 414), 0.45, mr.MUTED)
        slot = 0
        for k in range(3, 7):
            if len(res) > k and res.iloc[k].cost <= plaus:
                r = res.iloc[k]
                c = mr.card(d, r.key, 300, tag=f"#{k + 1} {r.cls} {pk.mmss(r.t0)}", sub=f"{r.video[:12]} cost {r.cost:.2f}{hc(r.key)}", pauses=temporal)
                paste(img, letterbox(c, 296, 158), 24 + slot * 312, 452)
                slot += 1
                shown += 1
        n_weak = int((res.cost.iloc[:7] > plaus).sum())
        if weak:
            mr.text(img, f"Nothing in this footage moved like that: even the closest candidate is above cost {plaus:.2f}.", (24, 480), 0.7, mr.RED)
            mr.text(img, "(weaker candidates are hidden on purpose; the closest one is shown greyed out above)", (24, 508), 0.5, mr.MUTED)
        elif n_weak and slot < 4:
            x0 = 24 + slot * 312 + 12
            mr.text(img, f"{n_weak} weaker candidate{'s' if n_weak > 1 else ''} hidden:", (x0, 486), 0.55, mr.MUTED)
            mr.text(img, f"all above cost {plaus:.2f} (not a match)", (x0, 510), 0.55, mr.MUTED)
    else:
        mr.stamp(img, "NO TRAJECTORIES", mr.RED, (440, 104), 1.3, -4)
        words, line, y = note.split(), "", 252
        for wd in words:                                     # wrap the explanation
            if mr.tsize(line + " " + wd, 0.55)[0] > 800:
                mr.text(img, line, (424, y), 0.55, mr.MUTED)
                line, y = wd, y + 24
            else:
                line = (line + " " + wd).strip()
        mr.text(img, line, (424, y), 0.55, mr.MUTED)
        if video:
            m = d["meta"][video]
            slideshow = m["windows"] > m["duration"]
            t0 = m["duration"] * 0.3
            ts = [t0 + k * (0.45 if slideshow else m["duration"] * 0.1) for k in range(6)]
            for k, t in enumerate(ts):
                fr = mr.frame(video, float(min(t, m["duration"] - 0.3)))
                paste(img, letterbox(fr, 200, 112), 424 + (k % 3) * 212, 316 + (k // 3) * 124)
            mr.text(img, ("6 pictures 0.45 s apart: a photo slideshow, every picture is a new scene" if slideshow
                          else "6 frames spread over the clip"), (424, 576), 0.5, mr.MUTED)
    if badges:
        for k, (name, line1, line2) in enumerate(badges):
            x = 24 + k * 312
            cv2.rectangle(img, (x, 630), (x + 296, 696), mr.PANEL, -1)
            cv2.rectangle(img, (x, 630), (x + 296, 696), (90, 80, 70), 1)
            mr.text(img, name, (x + 10, 654), 0.55, mr.ACC, halo=mr.PANEL)
            mr.text(img, line1, (x + 10, 676), 0.5, mr.FG, halo=mr.PANEL)
            mr.text(img, line2, (x + 10, 692), 0.38, mr.MUTED, halo=mr.PANEL)
    if timeline:
        dur, spans, marks = timeline
        mr.text(img, "WHERE IN THE FOOTAGE  (blue = camera-stable, trackable; coloured = top matches)", (24, 646), 0.45, mr.MUTED)
        mr.timeline_strip(img, 24, 656, W - 48, 16, dur, spans, marks)
    mr.text(img, foot, (24, H - 10), 0.42, mr.MUTED)
    return img


def timeline_for(d: dict, video: str, res: pd.DataFrame):
    meta = d["meta"][video]
    w = d["wins"][video]
    spans = [(float(a), float(b)) for a, b in zip(w.t0, w.t1) if b - a >= 0.5]
    cols = [(0, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 220), (180, 180, 255), (255, 160, 90), (90, 220, 255), (200, 255, 200)]
    marks = [(r.t0, r.t1, cols[i % len(cols)]) for i, r in enumerate(res.itertuples())]
    return meta["duration"], spans, marks


# ------------------------------------------------------------------------------------------------- main runs
def run_presets(d, df, arr, keys=None):
    out = {}
    for p in PRESET_ORDER:
        q = mc.preset_query(p)
        t0 = time.time()
        if keys is None:
            sub, a = df, arr
        else:
            msk = df["key"].isin(keys).values
            sub = df[msk].reset_index(drop=True)
            a = {k: v[msk] for k, v in arr.items()}
        out[p] = (mc.match(q, sub, a, k=8) if len(sub) else sub.iloc[0:0].assign(cost=[]), time.time() - t0)
    return out


def video_report(d, name: str, res: dict, cal: dict, clips_budget: int, do_clips: bool) -> dict:
    meta = d["meta"][name]
    vdir = OUT / name
    vdir.mkdir(parents=True, exist_ok=True)
    df = d["df"]
    n_tl = int((df.video == name).sum())
    cover = meta["cam_cover"] / max(meta["duration"], 1e-9)
    title = pk.videos()[name]["title"]
    rep = pk.Report(vdir / "report.html", f"MIMIC: {name}", f"{title}  |  {meta['w']}x{meta['h']}, {meta['duration']:.0f} s")
    rep.credit(name)
    rep.kv({"tracks (YOLO26s+ByteTrack)": meta["n_tracks"], "usable tracklets (>=1.5 s, camera-stable)": n_tl,
            "camera windows": meta["windows"], "footage in windows": f"{100 * cover:.0f}%"})
    summary = {"tracks": meta["n_tracks"], "tracklets": n_tl, "camera_windows": meta["windows"], "camera_cover_pct": round(100 * cover, 1),
               "presets": {}}
    # best preset for the hero = lowest top-1 cost
    ranked = sorted(((r[0].cost.iloc[0], p) for p, r in res.items() if len(r[0])), key=lambda x: x[0])
    hero_p = ranked[0][1] if ranked else "u_turn"
    n_clips = 0
    for p in PRESET_ORDER:
        r, secs = res[p]
        q = mc.preset_query(p)
        rep.h2(f"{p.replace('_', ' ')}: {mc.PRESETS[p]['desc']}")
        sk_path = pk.save_jpg(mr.sketch_panel(q.pts, 360, 200, "QUERY", p, marks=mc.preset_marks(p), speed=q.speed if q.speed_w > 0 else None), vdir / f"q_{p}.jpg")
        if not len(r):
            rep.text("No tracklets to search in this clip.")
            summary["presets"][p] = {"top1_cost": None, "verdict": "NO DATA", "top3": []}
            continue
        cards = []
        for i, row in enumerate(r.itertuples()):
            v, _ = verdict(row.cost, cal)
            rt = f" turned {row.rot_deg}" if (mc.PRESETS[p]["any_rot"] and row.rot_deg) else ""
            cards.append(mr.card(d, row.key, 380, tag=f"#{i + 1} {row.cls} {pk.mmss(row.t0)}{rt}", sub=f"cost {row.cost:.2f} {v}",
                                 pauses=p in ("stop_and_go", "loiter"), dim=(v == "NO MATCH")))
        sh = pk.save_jpg(pk.sheet(cards, cols=4, tile_w=380), vdir / f"top8_{p}.jpg", 80)
        rep.html(f'<div class="kv"><div><img src="{vdir.name and (Path("q_" + p + ".jpg"))}"></div></div>')
        rep.image(sh, f"top 8 by DTW cost (lower is better; greyed = above the match threshold, i.e. NOT a match); white box = the object, orange line = camera-compensated trajectory, green dot = start, red arrow = end, blue ring = halt; query answered in {secs * 1000:.0f} ms")
        v1, _ = verdict(r.cost.iloc[0], cal)
        if do_clips and v1 in ("MATCH", "MAYBE") and n_clips < clips_budget:
            cp = vdir / f"clip_{p}.mp4"
            mr.write_clip(d, r.key.iloc[0], cp, tag=f"{p} #1 cost {r.cost.iloc[0]:.2f}")
            rep.video(cp, f"best match for '{p}' (cost {r.cost.iloc[0]:.2f}, {v1})")
            n_clips += 1
        rep.table(r[["key", "cls", "t0", "t1", "dur", "cost", "c_shape", "c_turn"]].round(3).head(8))
        summary["presets"][p] = {"top1_cost": round(float(r.cost.iloc[0]), 3), "verdict": v1, "ms": round(secs * 1000, 1),
                                 "top3": [{"key": k, "cost": round(float(c), 3), "cls": cl, "t0": round(float(a), 1), "rot_deg": int(rd)}
                                          for k, c, cl, a, rd in zip(r.key[:3], r.cost[:3], r.cls[:3], r.t0[:3], r.rot_deg[:3])]}
    # hero
    q = mc.preset_query(hero_p)
    r = res[hero_p][0]
    foot = (f"{n_tl} tracklets searched | cosine kNN shortlist + DTW re-rank | cost calibrated on hand-checked results"
            if cal.get("strong") is not None else f"{n_tl} tracklets searched | cosine kNN + DTW | thresholds not calibrated")
    note = ""
    if not len(r):
        if n_tl == 0 or meta["windows"] > meta["duration"]:
            note = (f"{meta['n_tracks']} tracks detected but no usable trajectory: {meta['windows']} camera windows in {meta['duration']:.0f} s "
                    f"(cuts or camera motion), so almost nothing stays camera-stable for 1.5 s, and what does barely moves.")
        else:
            note = (f"{n_tl} tracklet(s), none moving at least half a body height: this clip is too short or too static for a path search "
                    f"({meta['duration']:.0f} s).")
    hero = make_hero(d, q, r, name, f"{meta['duration']:.0f} s | {meta['windows']} camera windows | {n_tl} tracklets", cal, foot,
                     timeline=timeline_for(d, name, r) if len(r) else (meta["duration"], [(float(a), float(b)) for a, b in zip(d['wins'][name].t0, d['wins'][name].t1) if b - a >= 0.5], []),
                     note=note, video=name)
    pk.save_jpg(hero, vdir / "hero.jpg", 88)
    rep.parts.insert(2, f'<figure><img src="hero.jpg"><figcaption>hero: preset "{hero_p}" (lowest top-1 cost in this clip)</figcaption></figure>')
    rep.save()
    summary["hero_preset"] = hero_p
    return summary


def parse_path(s: str) -> np.ndarray:
    return np.array([[float(a) for a in p.split(",")] for p in s.split(";") if p.strip()], float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="default test set: every cached clip")
    ap.add_argument("--video", help="one clip by catalogue name")
    ap.add_argument("--path", help="a drawn path in normalised canvas coordinates: 'x,y;x,y;...' (origin top-left)")
    ap.add_argument("--rot", action="store_true", help="with --path: match in any orientation")
    ap.add_argument("--pos-w", type=float, default=0.0, help="with --path: how much WHERE it happened matters (0..1)")
    ap.add_argument("--size-w", type=float, default=0.0, help="with --path: how much the extent of the path matters (0..1)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the tracklet table")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--no-page", action="store_true")
    a = ap.parse_args()
    if not (a.all or a.video or a.path):
        ap.error("give --all, --video NAME or --path")
    T0 = time.time()
    d = mc.build_all(rebuild=a.rebuild)
    df = d["df"]
    arr = mc.tl_arrays(df)
    con = make_db(d)
    gt = load_gt()
    cal = gt.get("calibration", {})
    print(f"[mimic] table: {len(df)} tracklets from {len(d['clips'])} clips ({time.time() - T0:.0f}s)")

    if a.path:
        q = mc.Query(parse_path(a.path), 16 / 9, a.pos_w, a.size_w, a.rot, None, 0.0, "drawn")
        keys = sql_keys(con, a.video)
        msk = df["key"].isin(keys).values
        sub = df[msk].reset_index(drop=True)
        r = mc.match(q, sub, {k: v[msk] for k, v in arr.items()}, k=8)
        cards = [mr.card(d, k, 380, tag=f"#{i + 1} {row.cls} {row.video[:14]} {pk.mmss(row.t0)}", sub=f"cost {row.cost:.2f} {verdict(row.cost, cal)[0]}")
                 for i, (k, row) in enumerate(zip(r.key, r.itertuples()))]
        qd = OUT / "query"
        pk.save_jpg(pk.sheet(cards, cols=4, tile_w=380), qd / "result.jpg", 82)
        pk.save_jpg(make_hero(d, q, r, "drawn query", f"{len(sub)} tracklets searched", cal, "drawn path query"), qd / "hero.jpg", 88)
        print(r[["key", "cls", "t0", "t1", "cost"]].round(3).to_string())
        print("wrote", qd / "result.jpg")
        return

    videos = d["clips"] if a.all else [a.video]
    for v in videos:
        if v not in d["clips"]:
            sys.exit(f"unknown video {v!r}")
    results_path = HERE / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if (results_path.exists() and not a.all) else {}
    results.setdefault("videos", {})

    # ---- per video
    per_video = {}
    n_clips_total = 0
    for v in videos:
        t0 = time.time()
        keys = sql_keys(con, v)
        res = run_presets(d, df, arr, keys)
        s = video_report(d, v, res, cal, 3, not a.no_clips)
        results["videos"][v] = s
        per_video[v] = res
        print(f"[mimic] {v:38s} tracklets={s['tracklets']:4d} hero={s['hero_preset']:<13s} "
              + " ".join(f"{p}={s['presets'][p]['top1_cost'] if s['presets'][p]['top1_cost'] is not None else '-'}" for p in PRESET_ORDER)
              + f"  ({time.time() - t0:.0f}s)", flush=True)
    # ---- hard negative for the demo: draw a U-turn on a busy scene that has no hairpin -> NO MATCH, greyed-out closest candidate
    if "yonge_dundas" in per_video:
        r = per_video["yonge_dundas"]["u_turn"][0]
        meta = d["meta"]["yonge_dundas"]
        foot = f"hard negative: u_turn sketch searched in yonge_dundas only ({int((df.video == 'yonge_dundas').sum())} tracklets); a crowd, but nobody walks a hairpin"
        hn = make_hero(d, mc.preset_query("u_turn"), r, "yonge_dundas (hard negative)", f"{meta['duration']:.0f} s | {meta['windows']} camera window | crowd scene", cal, foot,
                       timeline=timeline_for(d, "yonge_dundas", r) if len(r) else None)
        pk.save_jpg(hn, OUT / "hero_negative.jpg", 88)
        results["negative_test"] = {"clip": "yonge_dundas", "preset": "u_turn", "top1_cost": round(float(r.cost.iloc[0]), 3), "verdict": verdict(float(r.cost.iloc[0]), cal)[0],
                                    "threshold_plausible": cal.get("plausible"), "note": "tracker-level negative: no >=1.5 s camera-stable tracklet has a hairpin; it does not prove nobody turned round"}
    if not a.all:
        results["updated"] = time.strftime("%Y-%m-%d %H:%M")
        pk.write_results(results_path, results)
        return

    # ---- global (all clips): the 'judge draws, one match pops up' view
    import mimic_eval as me
    gdir = OUT / "global"
    gdir.mkdir(parents=True, exist_ok=True)
    keys = sql_keys(con)
    gres = {}
    glob = {}
    for p in PRESET_ORDER:
        q = mc.preset_query(p)
        t0 = time.time()
        r = mc.match(q, df, arr, k=8)
        ms = (time.time() - t0) * 1000
        gres[p] = r
        cards = [mr.card(d, k, 380, tag=f"#{i + 1} {row.cls} {row.video[:16]} {pk.mmss(row.t0)}" + (f" turned {row.rot_deg}" if (mc.PRESETS[p]["any_rot"] and row.rot_deg) else ""),
                         sub=f"cost {row.cost:.2f} {verdict(row.cost, cal)[0]}", pauses=p in ("stop_and_go", "loiter"), dim=verdict(row.cost, cal)[0] == "NO MATCH")
                 for i, (k, row) in enumerate(zip(r.key, r.itertuples()))]
        pk.save_jpg(pk.sheet(cards, cols=4, tile_w=380), gdir / f"top8_{p}.jpg", 82)
        glob[p] = {"ms": round(ms, 1), "top8": [{"key": k, "video": vv, "cls": cl, "t0": round(float(t0_), 1), "t1": round(float(t1_), 1), "cost": round(float(c), 3), "rot_deg": int(rd)}
                                              for k, vv, cl, t0_, t1_, c, rd in zip(r.key, r.video, r.cls, r.t0, r.t1, r.cost, r.rot_deg)]}
    ev = me.evaluate(d, df, arr, gres, gt, cal)
    results["global"] = glob
    results["eval"] = ev
    results["calibration"] = cal
    # global hero: the headline preset is u_turn (the demo moment)
    hp = "u_turn"
    badges = []
    for p in EVAL4:
        pv = ev["precision"][p]
        badges.append((p.replace("_", " "), f"hand-checked P@3 {pv['p_at_3']:.2f}  ({pv['correct_in_top8']}/8 right)", f"best cost {gres[p].cost.iloc[0]:.2f} from {gres[p].video.iloc[0]}"))
    hero = make_hero(d, mc.preset_query(hp), gres[hp], "all clips", f"{len(df)} tracklets | {sum(m['duration'] for m in d['meta'].values()) / 60:.0f} min of footage", cal,
                     "global search across 21 clips | cosine kNN shortlist + DTW re-rank | thresholds set by eye on the same hand-checked results", badges=badges,
                     labels=gt.get("labels", {}).get(hp, {}))
    pk.save_jpg(hero, OUT / "hero.jpg", 88)
    pk.save_jpg(hero, gdir / "hero.jpg", 88)
    # clips for the best of every preset (global)
    if not a.no_clips:
        for p in PRESET_ORDER:
            r = gres[p]
            if len(r) and verdict(r.cost.iloc[0], cal)[0] in ("MATCH", "MAYBE", "UNCALIBRATED"):
                mr.write_clip(d, r.key.iloc[0], gdir / f"clip_{p}.mp4", tag=f"{p} #1 cost {r.cost.iloc[0]:.2f}")
    results["meta"] = {k: v for k, v in d["meta"].items()}
    results["totals"] = {"clips": len(d["clips"]), "tracklets": int(len(df)), "seconds_total": round(time.time() - T0, 1)}
    results["updated"] = time.strftime("%Y-%m-%d %H:%M")
    pk.write_results(results_path, results)
    if not a.no_page:
        import build_page
        build_page.build(d, df, cal, gres)
    me.write_index(d, results, cal)
    if not a.no_page:                                        # the page in a real browser (headless Chrome/Edge, file://, synthetic pointer events)
        try:
            sys.path.insert(0, str(HERE / "tools"))
            import page_drive as pdv
            names = ["draw_u_turn", "draw_line", "yonge_u_turn_negative", "draw_walk_stop_walk"]
            brs = [pdv.run(sc) for sc in pdv.SCENARIOS if sc["name"] in names]
            results["page_browser_check"] = {"browser": Path(pdv.find_browser()).name, "note": "synthetic pointer events, file:// page, no server",
                                             "scenarios": [{k: v for k, v in r.items() if k not in ("first", "png")} for r in brs]}
            print("[mimic] page in a real browser:", [(r.get("scen"), r.get("cards"), len(r.get("errors", ["x"]))) for r in brs])
        except Exception as ex:  # noqa: BLE001
            results["page_browser_check"] = {"error": str(ex)[:200]}
        pk.write_results(results_path, results)
    print(f"[mimic] done in {time.time() - T0:.0f}s; kill test: {ev.get('kill_test')}")


if __name__ == "__main__":
    main()
