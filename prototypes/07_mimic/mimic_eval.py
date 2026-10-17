"""MIMIC evaluation: precision@3 from hand labels, kill test, SIMULATED self-retrieval and drawing-robustness tests,
and the global index page."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mimic_core as mc  # noqa: E402

pk = mc.pk
EVAL4 = ["u_turn", "left_to_right", "stop_and_go", "loiter"]


def wobble(pts: np.ndarray, rng: np.random.Generator, amp: float = 0.03) -> np.ndarray:
    """A plausible hand-drawn version of a path: resampled to ~24 vertices, low-frequency wobble plus jitter, slightly
    different size/position, and a mouse-speed-like point spacing."""
    p = mc.resample_arc(pts, 24)
    ext = float(np.linalg.norm(p.max(0) - p.min(0))) + 1e-6
    drift = np.cumsum(rng.normal(0, 1, (24, 2)), 0)
    drift = drift - np.linspace(drift[0], drift[-1], 24)
    drift = drift / (np.abs(drift).max() + 1e-9) * amp * ext
    jit = rng.normal(0, amp * 0.25 * ext, (24, 2))
    c = p.mean(0)
    sc = float(rng.uniform(0.8, 1.25))
    return (p - c) * sc + c + drift + jit + rng.normal(0, 0.03, 2)


def uturn_variants(df: pd.DataFrame, arr: dict) -> dict:
    """SIMULATED: 18 differently proportioned / oriented U-turn drawings (leg gap, leg length, rotation): do they all find the
    hand-verified hairpins (factory workers)?"""
    import itertools
    out = []
    for gap, length, ang in itertools.product([0.1, 0.2, 0.3], [0.25, 0.5], [0, 90, 200]):
        r_ = gap / 2
        th = np.linspace(-np.pi / 2, np.pi / 2, 20)
        arc = np.stack([length + r_ * np.cos(th) / (16 / 9), r_ + r_ * np.sin(th)], 1)
        pts = np.vstack([np.stack([np.linspace(0, length, 15), np.zeros(15)], 1), arc, np.stack([np.linspace(length, 0, 15), np.full(15, gap)], 1)])
        pts[:, 1] -= gap / 2
        pts = pts @ mc.rot2(np.deg2rad(ang)).T + 0.5
        r = mc.match(mc.Query(pts, 16 / 9, 0, 0, True, None, 0, "u"), df, arr, k=3)
        out.append({"gap": gap, "length": length, "angle": ang, "workers_in_top3": int(sum(k.startswith("worker_zone") for k in r.key)),
                    "top1": r.key.iloc[0], "top1_cost": round(float(r.cost.iloc[0]), 3)})
    return {"label": "SIMULATED: 18 differently proportioned / rotated U-turn drawings, hand-verified hairpins = worker_zone tracklets",
            "mean_workers_in_top3": round(float(np.mean([o["workers_in_top3"] for o in out])), 2),
            "top1_is_worker": int(sum(o["top1"].startswith("worker_zone") for o in out)), "n": len(out), "variants": out}


def evaluate(d: dict, df: pd.DataFrame, arr: dict, gres: dict, gt: dict, cal: dict) -> dict:
    labels = gt.get("labels", {})
    ev: dict = {"precision": {}, "kill_test": {}}
    n_pass, n_top1 = 0, 0
    for p in EVAL4 + ["diagonal"]:
        r = gres[p]
        lab = labels.get(p, {})
        top = [(k, lab.get(k)) for k in r.key[:8]]
        top3 = top[:3]
        ok3 = sum(1 for _, v in top3 if v == 1)
        unl = sum(1 for _, v in top3 if v is None)
        ok8 = sum(1 for _, v in top if v == 1)
        unl8 = sum(1 for _, v in top if v is None)
        ev["precision"][p] = {"p_at_3": round(ok3 / 3, 3), "correct_in_top3": ok3, "unlabelled_in_top3": unl,
                              "top1_correct": bool(top3 and top3[0][1] == 1), "correct_in_top8": ok8, "unlabelled_in_top8": unl8,
                              "top3": [{"key": k, "label": v} for k, v in top3]}
        if p in EVAL4:
            n_pass += int(ok3 >= 1)
            n_top1 += int(bool(top3 and top3[0][1] == 1))
    ev["kill_test"] = {"criterion": ">= 3 of 4 presets (u_turn, left_to_right, stop_and_go, loiter) return a plausible match (>= 1 hand-verified correct result in the top 3)",
                       "presets_passing": n_pass, "of": 4, "presets_with_correct_top1": n_top1, "passed": n_pass >= 3,
                       "labelled": bool(labels)}
    ev["mean_p_at_3_4presets"] = round(float(np.mean([ev["precision"][p]["p_at_3"] for p in EVAL4])), 3)

    # ---- SIMULATED self-retrieval: take real tracklets, 'redraw' them with a wobbly hand, find them again
    rng = np.random.default_rng(7)
    good = df[(df["len"] > 0.12) & (df["dur"] >= 2.0) & (df["rel_ext"] >= mc.MIN_REL) & (df["ext"] >= mc.MIN_EXT)].index.to_numpy()
    pick = rng.choice(good, size=min(60, len(good)), replace=False) if len(good) else np.array([], int)
    sim = {}
    lat = []
    for label, pos_w, one_clip in (("shape_only_all_clips", 0.0, False), ("shape_plus_where_all_clips", 0.6, False), ("shape_plus_where_one_clip", 0.6, True)):
        hit1 = hit8 = near = 0
        curved = [0, 0]
        for i in pick:
            row = df.loc[i]
            tl = d["tls"][row["key"]]
            pts = wobble(tl.xy, rng)
            q = mc.Query(pts, tl.W / tl.H, pos_w, 0.0, False, None, 0.0, "sim")
            if one_clip:
                m = (df["video"] == row["video"]).values
                sub, a_ = df[m].reset_index(drop=True), {k: v[m] for k, v in arr.items()}
            else:
                sub, a_ = df, arr
            t0 = time.time()
            r = mc.match(q, sub, a_, k=8)
            lat.append(time.time() - t0)
            keys = list(r.key)
            h1 = keys[:1] == [row["key"]]
            hit1 += int(h1)
            hit8 += int(row["key"] in keys)
            dup = False
            if keys and not h1:
                b_ = r.iloc[0]
                dup = bool(row["video"] == b_["video"] and row["cls"] == b_["cls"] and min(row["t1"], b_["t1"]) - max(row["t0"], b_["t0"]) > 0.3)
            near += int(h1 or dup)
            if row["straight"] < 0.8:
                curved[0] += 1
                curved[1] += int(row["key"] in keys)
        n = max(len(pick), 1)
        sim[label] = {"recall_at_1": round(hit1 / n, 3), "recall_at_8": round(hit8 / n, 3), "top1_is_same_object_or_overlapping_duplicate": round(near / n, 3),
                      "recall_at_8_curved_paths_only": round(curved[1] / max(curved[0], 1), 3), "n_curved": curved[0]}
    ev["simulated_self_retrieval"] = {"label": "SIMULATED hand-drawn queries (wobble + jitter + rescale of real tracklets)", "n": int(len(pick)), "variants": sim,
                                      "ms_per_query_median": round(1000 * float(np.median(lat)), 1) if lat else None}

    # ---- SIMULATED drawing robustness for the presets: do shaky re-drawings keep returning the clean results?
    rob = {}
    for p in EVAL4 + ["diagonal"]:
        clean10 = set(mc.match(mc.preset_query(p), df, arr, k=10).key)
        base = mc.preset_query(p)
        inside, total = 0, 0
        trials = 10
        for _ in range(trials):
            q = mc.Query(wobble(base.pts, rng, 0.04), base.aspect, base.pos_w, base.size_w, base.any_rot, base.speed, base.speed_w, p,
                         base.shape_w, base.turn_w, base.straight_w)
            r = mc.match(q, df, arr, k=3)
            inside += sum(1 for k in r.key if k in clean10)
            total += len(r)
        rob[p] = round(inside / max(total, 1), 2)
    ev["simulated_drawing_robustness"] = {"label": "SIMULATED shaky re-drawings of each preset: share of their top-3 results that are in the clean preset's top 10",
                                          "per_preset": rob}
    ev["simulated_uturn_variants"] = uturn_variants(df, arr)
    # ---- calibration support: cost of correct vs incorrect hand-labelled results
    cs, ws = [], []
    for p, lab in labels.items():
        if p not in gres:
            continue
        for k, v in lab.items():
            row = gres[p][gres[p].key == k]
            if len(row):
                (cs if v == 1 else ws).append(float(row.cost.iloc[0]))
    ev["labelled_cost"] = {"correct_n": len(cs), "correct_median": round(float(np.median(cs)), 3) if cs else None,
                           "correct_max": round(max(cs), 3) if cs else None, "wrong_n": len(ws),
                           "wrong_median": round(float(np.median(ws)), 3) if ws else None,
                           "wrong_min": round(min(ws), 3) if ws else None}
    return ev


def write_index(d: dict, results: dict, cal: dict) -> Path:
    out = HERE / "out"
    rep = pk.Report(out / "index.html", "MIMIC: find it by how it moved",
                    "Draw a path or pick a sketch; MIMIC finds the tracked objects that moved like that.")
    for c in d["clips"]:
        rep.credit(c)
    rep.image(out / "hero.jpg", "Global search over all clips with the u-turn sketch.")
    rep.html('<p><a href="draw.html"><b>Open the interactive page: draw a path with the mouse</b></a> (self-contained, runs in the browser)</p>')
    ev = results.get("eval", {})
    rep.h2("Kill test and precision (hand-checked)")
    rep.kv({"kill test": "PASS" if ev.get("kill_test", {}).get("passed") else "FAIL",
            "presets with a correct hit in top 3": f"{ev.get('kill_test', {}).get('presets_passing')} / 4",
            "mean P@3 (4 presets)": ev.get("mean_p_at_3_4presets"),
            "SIMULATED recall@8, shape + where": ev.get("simulated_self_retrieval", {}).get("variants", {}).get("shape_plus_where_all_clips", {}).get("recall_at_8")})
    rows = [{"preset": p, "P@3": v["p_at_3"], "correct top3": v["correct_in_top3"], "top-1 correct": v["top1_correct"],
             "correct in top 8": v["correct_in_top8"]} for p, v in ev.get("precision", {}).items()]
    rep.table(rows)
    for p, g in results.get("global", {}).items():
        rep.h2(f"{p.replace('_', ' ')}: top 8 across all clips")
        rep.image(out / "global" / f"top8_{p}.jpg", f"{g['ms']:.0f} ms for the query over {results['totals']['tracklets']} tracklets")
        cp = out / "global" / f"clip_{p}.mp4"
        if cp.exists():
            rep.video(cp, "best match")
    rep.h2("Per clip")
    rows = []
    for c, s in results["videos"].items():
        hp = s["hero_preset"]
        rows.append({"clip": c, "tracks": s["tracks"], "tracklets": s["tracklets"], "camera windows": s["camera_windows"],
                     "hero preset": hp, "top-1 cost": (s["presets"][hp] or {}).get("top1_cost"),
                     "verdict": (s["presets"][hp] or {}).get("verdict")})
    rep.table(rows)
    for c in results["videos"]:
        rep.html(f'<p><a href="{c}/report.html">{c}</a></p>')
    return rep.save()
