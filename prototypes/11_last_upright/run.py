"""LAST UPRIGHT: cached poses -> posture per person per frame -> last-upright / first-on-floor / speed / long-lie clock.

    python run.py --all                 # default test set (6 clips)
    python run.py --video skate_fall    # one clip
    python run.py --all --no-vlm        # skip the Cosmos verification calls
    python run.py --all --no-video      # skip the annotated mp4s (faster)
    python run.py --clock 3600          # print the long-lie ladder state `S` seconds after a fall (logic only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import protokit as pk  # noqa: E402
import posture as P  # noqa: E402
import render as R  # noqa: E402
import poses_src  # noqa: E402

DEFAULT_SET = ["skate_fall", "floor_lie", "assist_elderly", "help_sit", "sit_down", "classroom"]
OUT = HERE / "out"
GT = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8")) if (HERE / "ground_truth.json").exists() else {}
HEX = {P.UP: "#1fb5a0", P.SIT: "#4f8cff", P.DOWN: "#ff4d4f", P.PART: "#aa8ce6", P.UNK: "#828c96", "GONE": "#343c46"}


# ------------------------------------------------------------------------------------------- baselines / SQL
def box_only_verdict(name: str) -> dict:
    """What the hosted DataEngine alone could do: person boxes (no keypoints). Lying = wide box. Same smoothing/events."""
    tr = pk.tracks(name)
    r = tr.rows[(tr.rows["name"] == "person") & (tr.rows["tid"] >= 0)].copy()
    if r.empty:
        return {"label": "NO_ALARM", "detail": "no persons"}
    r["aspect"] = r["bw"] / r["bh"]
    r["conf"] = r["conf"]
    person = P.stitch_tracks(r)
    r["person"] = r["tid"].map(person)
    results = []
    for pid, d in r.sort_values("t").groupby("person"):
        if len(d) < 4:
            continue
        d = d.sort_values(["t", "conf"], ascending=[True, False]).drop_duplicates("t")
        was, raw = False, []
        for a in d["aspect"]:
            s = P.DOWN if a >= (0.95 if was else 1.25) else P.UP
            was = s == P.DOWN
            raw.append(s)
        st = P.smooth_states(d["t"].to_numpy(), raw)
        d = d.assign(thigh_v=np.nan, state=st, theta=[90.0 if s == P.DOWN else 0.0 for s in st], tilt=[90.0 if s == P.DOWN else 0.0 for s in st])
        results.append(P.PersonResult(int(pid), d, P.find_events(d)))
    v = P.verdict(results, tr.duration)
    return {"label": v["label"], "first_down": v["event"].get("first_down") if v.get("event") else None,
            "speed": v["event"].get("speed") if v.get("event") else None}


SQL = ("SELECT seg, start_s, end_s, people, down_frames, up_frames, sit_frames FROM posture_segments "
       "WHERE down_frames > 0 ORDER BY seg")


def vastdb_funnel(name: str, ft: pd.DataFrame, duration: float) -> tuple[list[dict], str]:
    """posture per 5 s segment next to the DataEngine person count, queried with the VastDB-safe SQL subset
    (no IN / GROUP BY / JOIN / window): the cheap pass that decides which segments get a Cosmos3-Reason call."""
    f = ft[ft["person"] >= 0].copy()
    f["seg"] = (f["t"] // 5).astype(int)
    nseg = int(np.ceil(duration / 5))
    rows = []
    try:
        de = pk.segments(name)
        de_people = dict(zip(de["seg"], de.get("n_person", pd.Series(0, index=de.index))))
    except Exception:
        de_people = {}
    for s in range(nseg):
        d = f[f["seg"] == s]
        rows.append({"video": name, "seg": s, "start_s": s * 5.0, "end_s": min((s + 1) * 5.0, duration),
                     "people": int(de_people.get(s, d["person"].nunique())),
                     "down_frames": int((d["state"] == P.DOWN).sum()), "up_frames": int((d["state"] == P.UP).sum()),
                     "sit_frames": int((d["state"] == P.SIT).sum())})
    con = pk.vastdb({"posture_segments": pd.DataFrame(rows)})
    return con.execute(SQL).fetchdf().to_dict("records"), SQL


# ------------------------------------------------------------------------------------------------- Cosmos
def parse_yn(text: str) -> bool | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return None


def parse_place(text: str) -> str:
    t = text.lower()
    for w in ("bed", "sofa", "couch", "floor", "ground"):
        if w in t:
            return {"couch": "sofa", "ground": "floor"}.get(w, w)
    return "other" if t.strip() else ""


VLM_FAILS = 0        # circuit breaker: two empty answers (server busy/down) and the remaining clips skip Cosmos
VLM_TIMEOUT = 75.0   # the local Cosmos server is shared with other jobs: a slow call must not stall the run


def cosmos_check(frame: np.ndarray, crop: np.ndarray) -> dict:
    """Verification on the key frame. Never decides the verdict: it can only agree, disagree or add context."""
    global VLM_FAILS
    t0 = time.time()
    if VLM_FAILS >= 2:
        return {"lying_on_floor_raw": "", "lying_on_floor": None, "place_raw": "", "place": "", "posture_text": "",
                "seconds": 0.0, "calls": 0, "skipped": "Cosmos server slow or down earlier in this run"}
    q1 = pk.vlm("Is a person lying down on the floor or ground in this image? Answer with one word: yes or no.", [frame],
                max_tokens=6, timeout=VLM_TIMEOUT)
    q2 = q3 = ""
    VLM_FAILS = 0 if q1 else VLM_FAILS + 1
    if q1:   # server answered: ask for context; if it is down or stuck, stop after one timeout
        q2 = pk.vlm("Where is the person lying or sitting: on the floor, on a bed, on a sofa or chair, or elsewhere? "
                    "Answer with one word: floor, bed, sofa or other.", [frame], max_tokens=6, timeout=VLM_TIMEOUT)
        if time.time() - t0 < 40:   # optional one-liner: short timeout, the verdict never waits on it
            q3 = pk.vlm("Describe the posture of the main person in one short sentence.", [crop], max_tokens=40,
                        timeout=25.0)
    return {"lying_on_floor_raw": q1, "lying_on_floor": parse_yn(q1), "place_raw": q2, "place": parse_place(q2),
            "posture_text": q3, "seconds": round(time.time() - t0, 1), "calls": 1 + bool(q2) + bool(q3)}


# ----------------------------------------------------------------------------------------------- one video
def pick_key(res: list[P.PersonResult], v: dict) -> tuple[P.PersonResult | None, dict | None]:
    if v.get("pid") is not None:
        r = next(r for r in res if r.pid == v["pid"])
        return r, v["event"]
    if res:
        return max(res, key=lambda r: len(r.rows)), None
    return None, None


def frame_with_overlay(name: str, t: float, ft: pd.DataFrame, kps: np.ndarray, pid: int | None):
    img = pk.frame_at(name, t)
    sub = ft[(ft["t"] - t).abs() < 1e-3]
    sub = sub[sub["person"] >= 0]
    pos = [ft.index.get_loc(i) for i in sub.index]
    return img, sub, kps[pos]


def process(name: str, use_vlm: bool = True, make_video: bool = True, clock_steps: tuple = (60, 300, 3600)) -> dict:
    t_start = time.time()
    meta = pk.info(name)
    dur = meta["duration"]
    po, pose_src = poses_src.load(name)
    res, ft = P.analyse(po)
    kps = P.kp_array(po)
    v = P.verdict(res, dur)
    key, ev = pick_key(res, v)
    label = v["label"]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)

    # ---- key frames: last upright | ground contact | flat on the floor (or the nearest equivalent)
    dt = float(np.median(np.diff(sorted(ft["t"].unique())))) if len(ft) > 1 else 0.07
    all_t = np.sort(ft["t"].unique())
    snap = lambda x: float(all_t[np.argmin(np.abs(all_t - x))])  # noqa: E731
    ts_key = sorted(key.rows["t"]) if key is not None else [0.0]
    times: list[tuple[float, str]] = []
    if label in ("FALL", "LIE") and ev is not None:
        if ev.get("last_upright") is not None:
            lu, ct, ft_ = ev["last_upright"], ev["contact_t"], ev["first_down"]
            times.append((lu, f"LAST UPRIGHT  {pk.mmss(lu)}"))
            if ct - lu >= dt * 0.9 and ft_ - ct >= dt * 0.9:
                times.append((ct, f"ON THE FLOOR  {pk.mmss(ct)}"))
                times.append((snap(ft_), f"FLAT  {pk.mmss(ft_)}"))
            else:
                times.append((snap(ft_), f"ON THE FLOOR  {pk.mmss(ct)}"))
        else:
            times.append((snap(ev["first_down"]), f"FIRST SEEN  {pk.mmss(snap(ev['first_down']))}"))
            times.append((snap(ev["down_end"] - dt), f"STILL DOWN  {pk.mmss(snap(ev['down_end'] - dt))}"))
    elif ev is not None and ev["kind"] == "sat":
        lt = ev["from_t"]
        rt = float(min(ts_key, key=lambda x: abs(x - (ev["t"] + 0.8))))
        times = [(lt, f"ON FEET  {pk.mmss(lt)}"), (rt, f"SEATED  {pk.mmss(rt)}")]
    else:
        lt, rt = ts_key[0], ts_key[len(ts_key) // 2] if len(ts_key) > 2 else ts_key[-1]
        times = [(lt, f"SEEN  {pk.mmss(lt)}"), (rt, f"SEEN  {pk.mmss(rt)}")]
    pid = key.pid if key is not None else None
    grp = set((ev or {}).get("merged_pids") or ([pid] if pid is not None else []))
    npan = len(times)
    pw_, ph_ = (1080, 608) if npan == 2 else (780, 608)
    panels_l: list[dict] = []
    raw_frames: list[np.ndarray] = []
    prev_t = None
    for t, cap in times:
        img, sub, kk = frame_with_overlay(name, t, ft, kps, pid)
        raw_frames.append(img)
        sub_k = sub[sub["person"].isin(grp)]
        ann = R.annotate_frame(img, sub, kk, only_pid=grp, label=False)
        boxes = [(r.x1, r.y1, r.x2, r.y2) for r in sub_k.itertuples()]
        st = str(sub_k["state"].iloc[0]) if len(sub_k) else P.UNK
        tl = float(sub_k["tilt"].iloc[0]) if len(sub_k) and sub_k["tilt"].iloc[0] == sub_k["tilt"].iloc[0] else None
        torso = bool(len(sub_k) and sub_k["theta"].iloc[0] == sub_k["theta"].iloc[0])
        nice = {P.PART: "on feet, partial view", P.UP: "upright", P.SIT: "sitting", P.DOWN: "down"}.get(st, st.lower())
        txt = nice if tl is None else f"{nice}: {'torso' if torso else 'body axis'} {tl:.0f}° from vertical"
        panels_l.append({"img": R.crop_panel(ann, boxes, pw_, ph_), "cap": cap, "state": st, "t": t, "sub": txt, "boxes": boxes,
                         "gap_txt": (f"+{t - prev_t:.2f} s" if prev_t is not None and npan == 3 else "")})
        prev_t = t
    for side, p in zip(("left", "mid", "right"), panels_l if npan == 3 else [panels_l[0], None, panels_l[1]]):
        if p is not None:
            pk.save_jpg(p["img"], out / f"key_{side}.jpg")
    rt_img, rt_boxes = raw_frames[-1], panels_l[-1]["boxes"]

    # ---- Cosmos verification on the key (last) frame
    vlm = None
    if use_vlm:
        crop = R.crop_panel(rt_img, rt_boxes, 800, 450)
        vlm = cosmos_check(rt_img, crop)

    # ---- level / wording
    place = (vlm or {}).get("place", "")
    level = label
    if label == "LIE" and place in ("bed", "sofa"):
        level = "LIE_BED"
    iv = ev.get("interval") if ev else None
    if label == "FALL":
        stamp = "FALL"
        headline = f"Last upright {pk.mmss(ev['last_upright'])}  to  on the floor {pk.mmss(ev['contact_t'])}"
        sub = (f"{iv:.1f} s from last upright to flat ({ev['to_ground']:.1f} s to first ground contact): FAST = a fall, not a deliberate "
               f"lie-down. Flat from {pk.mmss(ev['first_down'])}.")
        itxt, stxt, asub = f"{iv:.1f} s", "upright to flat: FAST = FALL", ""
    elif label == "LIE":
        stamp = "LYING IN BED" if level == "LIE_BED" else "LYING: CHECK"
        if ev.get("last_upright") is None:
            headline = f"Already down when first seen ({pk.mmss(ev['first_down'])})"
            sub = ("No upright-to-floor transition in the footage, so fall vs deliberate lie-down cannot be told. "
                   + ("Cosmos says bed: logged, no fall alarm." if level == "LIE_BED" else "Long-lie clock starts at first sight."))
            itxt, stxt, asub = "--", "onset not observed", "clock runs from first sight"
        else:
            headline = f"Went down {pk.mmss(ev['contact_t'])} after {iv:.1f} s ({ev['speed']})"
            sub = "Not a fast fall: possibly a deliberate lie-down or an unobserved stretch. Check on the resident."
            itxt, stxt, asub = f"{iv:.1f} s", f"{ev['speed']} transition", ""
    else:
        stamp = "NO ALARM"
        if ev is not None and ev["kind"] == "sat":
            headline = f"On feet until {pk.mmss(ev['from_t'])}, seated from {pk.mmss(ev['t'])}"
            sub = "A stand-to-sit is a SAT event, not a fall: the torso never swung to the floor posture."
            itxt, stxt, asub = "SAT", ("speed " + ev["speed"]) if not ev["speed"].startswith("n/a") else "speed not measurable", "no alarm"
        else:
            headline = f"{len(res)} {'person' if len(res) == 1 else 'people'} tracked, nobody went down"
            sit_frac = float((key.rows["state"] == P.SIT).mean()) if key is not None else 0.0
            sub = (f"Seated in {sit_frac:.0%} of the frames seen; no person reached the floor posture. " if sit_frac >= 0.8 else
                   "No person reached the floor posture. ") + "No stand-to-floor transition in the footage."
            itxt, stxt, asub = "0", "falls", ""
    # ---- long-lie clock: starts at first ground contact
    since = None
    obs_floor = None
    if label in ("FALL", "LIE") and ev is not None:
        c0 = ev["contact_t"]
        since = max(0.0, dur - c0)
        cs = P.clock_state(since)
        ev["down_end"] = min(ev["down_end"], dur)
        ev["down_s"] = ev["down_end"] - ev["first_down"]
        obs_floor = max(0.0, ev["down_end"] - c0)
        ev["on_floor_s"] = obs_floor
        at_end = ev["down_end"] >= dur - 0.3
        if at_end:
            csub = ("down from first sight to clip end" if ev.get("last_upright") is None else "still down at clip end")
        elif ev["end_state"] == "lost":
            csub = f"left the frame at {pk.mmss(ev['down_end'])}"
        elif ev["end_state"] in (P.UP, P.SIT, P.PART):
            csub = f"back upright at {pk.mmss(ev['down_end'])}"
        else:
            csub = f"posture unclear after {pk.mmss(ev['down_end'])}"
        bed = level == "LIE_BED"
        clock = {"title": "TIME LYING DOWN (observed)" if bed else "TIME ON FLOOR (observed)", "big": P.fmt_dur(obs_floor), "sub": csub}
        ladder = [((f"{P.short_dur(r[0])} " if r[0] else "") + r[1], (r[1] == cs["rung"]) and not bed, r[2]) for r in P.LADDER]
        lsub = ("on a bed: the floor long-lie clock is not started" if bed else
                f"measured: {P.fmt_dur(obs_floor)} on the floor, rung 1 only. Rungs 2-4 are SIMULATED (clock_sim.jpg / .mp4)")
    else:
        cs = None
        clock = {"title": "TIME ON FLOOR", "big": "none", "sub": "nobody reached the floor posture"}
        ladder = [(r[1], False, r[2]) for r in P.LADDER]
        lsub = "clock starts at the estimated fall"
    if vlm is None or vlm["lying_on_floor"] is None:
        vl = {"line1": "Cosmos: unavailable" if vlm is None or not vlm["lying_on_floor_raw"] else "Cosmos: unclear",
              "col": R.MUTED, "lines": ["server busy or down: verdict above is pose-only"]}
    else:
        yes, down_label = vlm["lying_on_floor"], label in ("FALL", "LIE")
        if yes and down_label:
            line1, col = "Lying on floor: YES (agrees)", (31, 181, 160)
        elif not yes and down_label and vlm["place"] in ("bed", "sofa"):
            line1, col = f"Lying on a {vlm['place']}, not the floor", (79, 140, 255)
        elif not yes and down_label:
            line1, col = "Lying on floor: NO (disagrees)", (240, 160, 30)
        elif not yes:
            line1, col = "Lying on floor: NO (agrees)", (31, 181, 160)
        else:
            line1, col = "Lying on floor: YES (disagrees)", (240, 160, 30)
        vl = {"line1": line1, "col": col,
              "lines": [(f"place: {vlm['place'] or '?'}  " if down_label else "") + f"({vlm['seconds']} s, {vlm['calls']} calls)", vlm["posture_text"] or ""]}
    ks = key.rows if key is not None else pd.DataFrame(columns=["t", "state"])
    if ev and ev.get("merged_pids") and len(ev["merged_pids"]) > 1:   # one resident split by a tracker ID switch: join the pieces
        parts = [r.rows.assign(_k=int(r.pid != key.pid)) for r in res if r.pid in ev["merged_pids"]]
        ks = pd.concat(parts).sort_values(["t", "_k"]).drop_duplicates("t")

    def state_at(t: float) -> str:
        return str(ks.loc[(ks["t"] - t).abs().idxmin(), "state"]) if len(ks) else P.UNK

    marks = []
    if label in ("FALL", "LIE") and ev is not None:
        if ev.get("last_upright") is not None:
            marks.append((ev["last_upright"], "last upright", R.COL.get(state_at(ev["last_upright"]), R.COL[P.UP])))
            marks.append((ev["contact_t"], "on floor", R.COL[P.DOWN]))
        else:
            marks.append((ev["first_down"], "down at first sight", R.COL[P.DOWN]))
    elif ev is not None and ev["kind"] == "sat":
        marks += [(ev["from_t"], "on feet", R.COL.get(state_at(ev["from_t"]), R.COL[P.UP])), (ev["t"], "sat", R.COL[P.SIT])]
    spans = R.spans_of(ks, dur) if key is not None else [(0, dur, "GONE")]
    seen_t = sorted(ks["t"]) if len(ks) else [0.0]
    strip_t0, strip_t1 = max(0.0, seen_t[0] - 0.4), min(dur, seen_t[-1] + 0.9)
    gtc = GT.get(name, {})
    cred = pk.credit(name)
    base = box_only_verdict(name)
    box_right = base["label"] == gtc.get("label") if gtc else None
    if base["label"] != gtc.get("label", base["label"]):
        bx_line, bx_col = f"{base['label']} (wrong)", (240, 160, 30)
        bx_txt = ("a wide box reads as a fall: it cannot tell a bed from the floor" if gtc.get("place") == "bed"
                  else "box shape alone misreads this clip")
    else:
        bx_line, bx_col = f"{base['label'].replace('_', ' ')} (right)", (31, 181, 160)
        ref = gtc.get("ground_contact_s", gtc.get("first_down_s"))
        if base.get("first_down") is not None and ref is not None:
            bx_txt = f"fires at {base['first_down']:.1f} s ({base['first_down'] - ref:+.1f} s vs ground contact); no stand/sit split, no skeleton"
        else:
            bx_txt = "same verdict here; no skeleton, no stand/sit split, no speed measure"
    hero = R.hero_card({
        "title": f"{name}  ·  person #{pid}" if pid is not None else name, "stamp": stamp, "level": level, "headline": headline,
        "sub": sub, "panels": panels_l, "big_txt": itxt, "speed_txt": stxt, "arrow_sub": asub,
        "strip": spans, "strip_t0": strip_t0, "duration": strip_t1, "marks": marks, "clock": clock, "ladder": ladder, "ladder_sub": lsub, "vlm": vl,
        "boxonly": {"line1": bx_line, "col": bx_col, "text": bx_txt}, "credit": cred[:150]}, out / "hero.jpg")

    # ---- SIMULATED long-lie clock (the observed floor time is real, the later rungs are logic only)
    clock_files = {}
    if label in ("FALL", "LIE") and ev is not None and level != "LIE_BED":
        steps = clock_steps
        R.clock_filmstrip(rt_img, obs_floor, out / "clock_sim.jpg", f"{name}: {P.fmt_dur(obs_floor)} observed, then the clock advanced", steps)
        clock_files["clock_sim"] = str((out / "clock_sim.jpg").relative_to(HERE))
        if make_video:
            R.clock_video(rt_img, obs_floor, out / "clock_sim.mp4", end_s=float(max(steps)))
            clock_files["clock_sim_video"] = str((out / "clock_sim.mp4").relative_to(HERE))

    # ---- annotated video
    vid_path = None
    if make_video:
        vid_path = out / "annotated.mp4"
        by_f = {}
        for i in ft.index[ft["person"] >= 0]:
            by_f.setdefault(int(ft.at[i, "f"]), []).append(i)
        stride = max(1, int(round(meta["fps"] / 15.0)))
        with pk.Mp4Writer(vid_path, meta["fps"], width=min(960, meta["w"])) as w:
            for i, t, frame in pk.iter_frames(name):
                near = int(round(i / stride)) * stride
                idx = by_f.get(near, [])
                if idx:
                    sub = ft.loc[idx]
                    frame = R.annotate_frame(frame, sub, kps[[ft.index.get_loc(j) for j in idx]], label=True)
                if label in ("FALL", "LIE") and ev is not None and t >= ev["contact_t"]:
                    col = (60, 60, 230) if label == "FALL" else (30, 160, 240)
                    bh_ = max(34, frame.shape[0] // 14)
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], bh_), col, -1)
                    head = "FALL" if label == "FALL" else "LYING"
                    if t <= ev["down_end"]:
                        txt = f"{head}  {pk.mmss(ev['contact_t'])}   on floor {t - ev['contact_t']:.1f} s"
                    else:
                        txt = f"{head}  {pk.mmss(ev['contact_t'])}   last seen down {pk.mmss(ev['down_end'])}, out of view"
                    cv2.putText(frame, txt, (10, int(bh_ * 0.72)), cv2.FONT_HERSHEY_DUPLEX, bh_ / 46, (255, 255, 255), 2, cv2.LINE_AA)
                w.write(frame)

    # ---- VastDB-style funnel
    funnel, sql = vastdb_funnel(name, ft, dur)

    # ---- report
    rep = pk.Report(out / "report.html", f"LAST UPRIGHT: {name}", f"{label}: {headline}").credit(name)
    rep.image(hero, "Hero card: what the reviewer sees first.")
    kvs = {"verdict": f"{label}" + (f" ({level})" if level != label else ""), "people tracked": len(res),
           "last upright": pk.mmss(ev["last_upright"]) + f" ({ev.get('upright_view')})" if ev and ev.get("last_upright") is not None else "-",
           "first ground contact": pk.mmss(ev["contact_t"]) if ev and "contact_t" in ev else "-",
           "flat (torso >= 70 deg)": pk.mmss(ev["first_down"]) if ev and "first_down" in ev else "-",
           "last upright to ground contact": f"{ev['to_ground']:.2f} s" if ev and ev.get("to_ground") is not None else "-",
           "last upright to flat": f"{iv:.2f} s" if iv is not None else "-",
           "observed on floor (from ground contact)": P.fmt_dur(obs_floor) if obs_floor is not None else "-",
           "box-only baseline": base["label"]}
    rep.kv(kvs)
    if vid_path:
        rep.video(vid_path, "Pose, posture state and torso angle per person, per frame (pose sampled at 15 fps, held between samples).")
    rep.h2("Key frames")
    for (t_, cap_), side in zip(times, ("left", "mid", "right") if npan == 3 else ("left", "right")):
        rep.image(out / f"key_{side}.jpg", cap_)
    if clock_files:
        rep.h2("Long-lie clock (observed time is real, later rungs SIMULATED)")
        rep.image(out / "clock_sim.jpg", "Card 1 is measured on the footage. Cards 2-4 advance the clock with posture.clock_state: SIMULATED.")
        if "clock_sim_video" in clock_files:
            rep.video(out / "clock_sim.mp4", "SIMULATED accelerated clock (log scale) from the observed time to 1 h.")
    rep.h2("Posture timelines")
    for r in res:
        sp = [(s, e, HEX.get(st, "#888"), f"{st} {pk.mmss(s)}-{pk.mmss(e)}") for s, e, st in R.spans_of(r.rows, dur)]
        rep.timeline(sp, dur, f"person #{r.pid}: {len(r.rows)} samples; teal=UPRIGHT, blue=SITTING, red=DOWN, violet=partial view, grey=unclear, dark=not in view")
    rep.h2("Events")
    rows = []
    for r in res:
        for e in r.events:
            rows.append({"person": r.pid, **{k: (round(x, 2) if isinstance(x, float) else x) for k, x in e.items()}})
    rep.table(rows) if rows else rep.text("No posture events.")
    rep.h2("Cosmos verification (local stand-in)")
    rep.json(vlm, "Key frame questions; never used to decide the verdict.") if vlm else rep.text("Skipped.")
    rep.h2("VastDB funnel: which 5 s segments need a Cosmos call")
    rep.text(sql)
    rep.table(funnel) if funnel else rep.text("No segment with a DOWN posture: zero Cosmos calls needed.")
    rep.h2("Box-only baseline (DataEngine detections, no keypoints)")
    rep.json(base)
    rep.save()

    rec = {
        "video": name, "pose_source": pose_src, "duration_s": round(dur, 2), "resolution": f"{meta['w']}x{meta['h']}", "people_tracked": len(res),
        "label": label, "level": level, "headline": headline,
        "event": {k: (round(x, 3) if isinstance(x, float) else x) for k, x in (ev or {}).items()},
        "ground_truth": gtc, "correct": (label == gtc.get("label")) if gtc else None,
        "clock": cs, "since_fall_at_clip_end_s": round(since, 2) if since is not None else None,
        "falls_flagged": sum(1 for r in res for e in r.events if e["kind"] == "down" and e["down_s"] >= 0.5 and e["speed"] == "FAST"),
        "down_events": sum(1 for r in res for e in r.events if e["kind"] == "down" and e["down_s"] >= 0.5),
        "sat_events": sum(1 for r in res for e in r.events if e["kind"] == "sat"),
        "vlm": vlm, "vlm_correct": (None if not vlm or vlm["lying_on_floor"] is None or "on_floor" not in gtc else vlm["lying_on_floor"] == gtc["on_floor"]),
        "box_only": base, "box_only_correct": (base["label"] == gtc.get("label")) if gtc else None,
        "vastdb_sql": sql, "vastdb_rows": funnel, "seconds": round(time.time() - t_start, 1),
        "files": {"hero": str((out / "hero.jpg").relative_to(HERE)), "report": str((out / "report.html").relative_to(HERE)),
                  "video": str(vid_path.relative_to(HERE)) if vid_path else None, **clock_files},
        "last_upright_s": ev.get("last_upright") if ev else None, "ground_contact_s": ev.get("contact_t") if ev else None,
        "flat_s": ev.get("first_down") if ev else None, "observed_on_floor_s": round(obs_floor, 2) if obs_floor is not None else None,
    }
    if gtc.get("first_down_s") is not None and ev and "first_down" in ev:
        rec["first_down_error_s"] = round(ev["first_down"] - gtc["first_down_s"], 2)          # flat time vs truth
        if base.get("first_down") is not None:
            rec["box_only_first_down_error_s"] = round(base["first_down"] - gtc["first_down_s"], 2)
    if gtc.get("ground_contact_s") is not None and ev and ev.get("contact_t") is not None:
        rec["ground_contact_error_s"] = round(ev["contact_t"] - gtc["ground_contact_s"], 2)
        if base.get("first_down") is not None:
            rec["box_only_vs_ground_contact_s"] = round(base["first_down"] - gtc["ground_contact_s"], 2)
    if gtc.get("last_upright_s") is not None and ev and ev.get("last_upright") is not None:
        rec["last_upright_error_s"] = round(ev["last_upright"] - gtc["last_upright_s"], 2)
    return rec


SENS = [("baseline", {}), ("DOWN_ENTER 50", {"DOWN_ENTER_DEG": 50.0}), ("DOWN_ENTER 70", {"DOWN_ENTER_DEG": 70.0}),
        ("DOWN_EXIT 35", {"DOWN_EXIT_DEG": 35.0}), ("DOWN_EXIT 55", {"DOWN_EXIT_DEG": 55.0}), ("UP_DEG 15", {"UP_DEG": 15.0}),
        ("UP_DEG 35", {"UP_DEG": 35.0}), ("UP_DEG_REF 20", {"UP_DEG_REF": 20.0}), ("UP_DEG_REF 40", {"UP_DEG_REF": 40.0}), ("FLAT_DEG 60", {"FLAT_DEG": 60.0}), ("FLAT_DEG 80", {"FLAT_DEG": 80.0}),
        ("FAST_S 0.8", {"FAST_S": 0.8}), ("FAST_S 2.5", {"FAST_S": 2.5}), ("KP_VIS 0.25", {"KP_VIS": 0.25}),
        ("KP_VIS 0.5", {"KP_VIS": 0.5})]


def sensitivity() -> list[dict]:
    """Re-run the pose-only verdicts with one threshold moved at a time: does the 5/5 survive, or was it tuned to a knife edge?"""
    data = {n: poses_src.load(n)[0] for n in DEFAULT_SET}
    rows = []
    for name, over in SENS:
        old = {k: getattr(P, k) for k in over}
        for k, v in over.items():
            setattr(P, k, v)
        try:
            labels = {}
            for n, po in data.items():
                res, _ = P.analyse(po)
                labels[n] = P.verdict(res, pk.info(n)["duration"])["label"]
        finally:
            for k, v in old.items():
                setattr(P, k, v)
        ok = {n: labels[n] == GT[n]["label"] for n in labels}
        rows.append({"config": name, "labels": labels, "pexels_correct": sum(ok[n] for n in DEFAULT_SET[:5]),
                     "classroom_falls_label": labels["classroom"] == "FALL", "wrong": [n for n in ok if not ok[n]]})
        print(f"  [sens] {name:<14} pexels {rows[-1]['pexels_correct']}/5  wrong={rows[-1]['wrong']}", flush=True)
    return rows


NEG_SET = ["people_detection", "one_by_one_person_detection", "retail_cctv", "store_aisle_detection"]


def negatives(names: list[str]) -> dict:
    """False-alarm check on footage with no falls (people walking, shopping). Every DOWN event is dumped as a frame
    pair under out/_negatives/ so it can be looked at; nothing here is tuned on these clips."""
    out = {}
    for n in names:
        t0 = time.time()
        meta = pk.info(n)
        try:
            po, src = poses_src.load(n, allow_cpu=False)
        except FileNotFoundError:
            print(f"  [neg] {n}: no cached poses, skipped", flush=True)
            continue
        res, ft = P.analyse(po)
        kps = P.kp_array(po)
        downs = [(r.pid, e) for r in res for e in r.events if e["kind"] == "down" and e["down_s"] >= 0.5]
        rec = {"duration_s": round(meta["duration"], 1), "pose_source": src, "people_tracked": len(res),
               "person_frames": int((ft["person"] >= 0).sum()),
               "frames_labelled_DOWN": int((ft["state"] == P.DOWN).sum()), "down_events": len(downs),
               "falls_flagged": sum(1 for _, e in downs if e["speed"] == "FAST"),
               "events": [{"person": p, "first_down": round(e["first_down"], 2), "down_s": round(e["down_s"], 2), "speed": e["speed"]}
                          for p, e in downs][:10]}
        for k, (p, e) in enumerate(downs[:3]):
            t = e["first_down"] + min(0.5, e["down_s"] / 2)
            near = ft.loc[(ft["t"] - t).abs().idxmin(), "t"]
            img, sub, kk = frame_with_overlay(n, float(near), ft, kps, p)
            pk.save_jpg(R.annotate_frame(img, sub, kk), OUT / "_negatives" / f"{n}_down{k}_{pk.mmss(near).replace(':', '-')}.jpg")
        rec["seconds"] = round(time.time() - t0, 1)
        out[n] = rec
        print(f"  [neg] {n}: {rec['people_tracked']} people, {rec['down_events']} DOWN events, {rec['falls_flagged']} falls ({rec['seconds']} s)", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video", action="append")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--negatives", action="store_true", help="false-alarm check on extra no-fall clips (walking, shopping)")
    ap.add_argument("--sensitivity", action="store_true", help="move each threshold once and re-score the 6 clips (pose only)")
    ap.add_argument("--clock-steps", default="60,300,3600", help="SIMULATED long-lie clock cards (seconds after the fall), default 60,300,3600")
    ap.add_argument("--clock", type=float, help="print the long-lie ladder state S seconds after a fall and exit")
    a = ap.parse_args()
    if a.clock is not None:
        print(json.dumps(P.clock_state(a.clock), indent=1))
        return
    names = DEFAULT_SET if a.all else (a.video or [])
    if not names and not a.negatives and not a.sensitivity:
        ap.error("give --all, --video NAME, --negatives or --sensitivity")
    results = {}
    old = json.loads((HERE / "results.json").read_text(encoding="utf-8")) if (HERE / "results.json").exists() else {"videos": {}}
    sens = sensitivity() if a.sensitivity else old.get("sensitivity", [])
    neg = old.get("extra_negatives", {})
    if a.negatives:
        neg = {**neg, **negatives(NEG_SET)}
    for n in names:
        print(f"[last_upright] {n} ...", flush=True)
        r = process(n, use_vlm=not a.no_vlm, make_video=not a.no_video, clock_steps=tuple(float(x) for x in a.clock_steps.split(',')))
        results[n] = r
        print(f"  {r['label']:<9} {r['headline']}  correct={r['correct']}  vlm_correct={r['vlm_correct']}  {r['seconds']} s", flush=True)
    merged = {**old.get("videos", {}), **results}
    pm = {k: merged[k] for k in DEFAULT_SET if k in merged}
    pex = ["skate_fall", "floor_lie", "assist_elderly", "help_sit", "sit_down"]
    scored = [pm[k]["correct"] for k in pex if k in pm and pm[k]["correct"] is not None]
    kill = {"criterion": ">= 4 of 5 Pexels clips labelled correctly AND zero falls on classroom",
            "pexels_correct": sum(1 for c in scored if c), "pexels_total": len(scored),
            "classroom_falls": pm["classroom"]["falls_flagged"] if "classroom" in pm else None}
    kill["passed"] = bool(len(scored) == 5 and kill["pexels_correct"] >= 4 and kill["classroom_falls"] == 0)
    pk.write_results(HERE / "results.json", {"idea": "LAST UPRIGHT", "videos": merged, "kill_test": kill, "extra_negatives": neg, "sensitivity": sens,
                                              "thresholds": {"UP_DEG": P.UP_DEG, "UP_DEG_REF": P.UP_DEG_REF, "DOWN_ENTER_DEG": P.DOWN_ENTER_DEG,
                                                             "DOWN_EXIT_DEG": P.DOWN_EXIT_DEG, "FAST_S": P.FAST_S, "SLOW_S": P.SLOW_S,
                                                             "KP_VIS": P.KP_VIS, "min_dwell_s": P.MIN_DWELL}})
    print("kill test:", json.dumps(kill))


if __name__ == "__main__":
    main()
