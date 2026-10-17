"""LANE CALL CLI.  python run.py --all [--bonus] | --video NAME [--S 5] [--no-vlm] [--no-clips]

Per video: stopped-vehicle detector (tracks, spatial matching) -> per alert: ground-contact line from the YOLO-seg silhouette ->
lane geometry call -> cards; IN LANE alerts that are stacked and appeared together collapse into ONE low-severity QUEUE card;
severity-ordered queue (report.html), hero.jpg, cards/*, eyeball sheets. results.json aggregates all videos (merged, never
overwritten by a single-video run).

Videos: the catalogue names, plus the virtual clip 'motorway_a40_inject' = the real A40 clip with 3 SIMULATED stopped cars pasted
in (see inject.py): the only way to exercise the headline motorway use case end to end.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import protokit as pk  # noqa: E402
import lanecall as L  # noqa: E402
import lanecall_core as lc  # noqa: E402
from lanecall import AT_KERB, CANT, IN_LANE, QUEUE  # noqa: E402

OUT = HERE / "out"
INJECT = "motorway_a40_inject"
DEFAULT_SET = ["lisbon_parking", "motorway_a40", INJECT, "ptz_parking"]
BONUS_SET = ["yonge_dundas"]
SLIDE_VIDEOS = {"lisbon_parking"}
BASE_OF = {INJECT: "motorway_a40"}           # catalogue clip behind a virtual clip (zones, credit)
VLM_BUDGET = 28            # <= 30 VLM calls per video
CLIP_BUDGET = {IN_LANE: 12, CANT: 4, QUEUE: 2, AT_KERB: 4}
CLIP_PRE, CLIP_POST = 3.0, 5.0     # 8 s clip around the alert
QUEUE_MIN = 3              # >= 3 IN LANE stops stacked in one lane zone that stopped together -> one QUEUE card
QUEUE_DT, QUEUE_GAP = 15.0, 5.0    # stop times within 15 s; centres within 5 mean box widths


# ------------------------------------------------------------------------------------------------ helpers
class FrameSource:
    """Sequential-ish random access to frames of one video, with a tiny cache (many alerts share a frame)."""

    def __init__(self, src):
        self.cap = cv2.VideoCapture(str(pk.video_path(src)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.cache: dict[int, np.ndarray] = {}

    def by_fid(self, fid: int) -> np.ndarray:
        if fid not in self.cache:
            if len(self.cache) > 8:
                self.cache.pop(next(iter(self.cache)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, fr = self.cap.read()
            if not ok:
                raise ValueError(f"cannot read frame {fid}")
            self.cache[fid] = fr
        return self.cache[fid]

    def at(self, t: float) -> tuple[int, np.ndarray]:
        fid = int(round(t * self.fps))
        return fid, self.by_fid(fid)


def vehicles_at(tr, t: float) -> list[list[float]]:
    d = tr.at(t)
    d = d[d["name"].isin(lc.VEH)] if len(d) else d
    return [[r.x1, r.y1, r.x2, r.y2] for r in d.itertuples()]


def stop_txt(c: dict) -> str:
    """'stopped 05:00' or, when the vehicle was already still at the start of the view, say so (no fake 00:00 timestamp)."""
    if c.get("slide_kind"):
        return "still from a slideshow"
    if c["t_stop"] < 0.5:
        return "already stopped at view start"
    return f"stopped {pk.mmss(c['t_stop'])}"


def draw_foot(img: np.ndarray, c: dict, col, scale: float = 1.0) -> None:
    pts = c.get("foot_px")
    if pts:
        p = (np.array(pts, float) * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [p], False, (255, 255, 255), max(2, int(3 * scale)), cv2.LINE_AA)
        for q in p.reshape(-1, 2):
            cv2.circle(img, tuple(int(v) for v in q), max(3, int(4 * scale)), col, -1, cv2.LINE_AA)


def annotated(frame: np.ndarray, cam: L.Cam | None, reg, card: dict, width: int = 960, extra: list | None = None) -> np.ndarray:
    """Frame with zones, the vehicle box, its ground-contact line and the call stamp. extra: other cards' boxes (queue)."""
    img = frame.copy()
    if cam is not None and reg is not None:
        img = L.draw_zones(img, cam, reg["H"])
    col = L.CALL_COL[card["call"]]
    for b in (extra or [card["box"]]):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 4)
    x1, y1, x2, y2 = (int(v) for v in card["box"])
    if not extra:
        draw_foot(img, card, col)
    s = width / img.shape[1]
    img = cv2.resize(img, (width, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    sx, sy = int(x1 * s), int(y1 * s)
    img = L.stamp(img, card["call"] if not extra else QUEUE, card["conf"], (max(4, min(sx, width - 260)), max(4, sy - 56)), size=26)
    return img


def write_clip(src, name: str, card: dict, dst: Path) -> None:
    """8 s clip (3 s before, 5 s after the alert) with the vehicle(s) boxed and the call as a caption."""
    meta = pk.info(src)
    t0 = max(0.0, card["t_alert"] - CLIP_PRE)
    t1 = min(meta["duration"], card["t_alert"] + CLIP_POST)
    col = L.CALL_COL[card["call"]]
    boxes = card.get("boxes") or [card["box"]]
    with pk.Mp4Writer(dst, meta["fps"] / 2, width=640) as w:
        for i, t, fr in pk.iter_frames(src, stride=2, start=t0, end=t1):
            if t >= card["t_stop"] - 0.5:
                for b in boxes:
                    x1, y1, x2, y2 = (int(v) for v in b)
                    cv2.rectangle(fr, (x1, y1), (x2, y2), col, 4)
            fr = pk.fit(fr, 640)
            pk.label(fr, f"{card['id']} {card['call']} {card['conf']:.0%}  t={pk.mmss(t)}", (6, 6), scale=0.55, bg=col)
            w.write(fr)


# ------------------------------------------------------------------------------------------ stream pipeline
def seg_for_alerts(alerts: list[dict], fs: FrameSource, cache: Path) -> dict:
    return L.seg_profiles([a["fid"] for a in alerts], fs.by_fid, cache)


def build_cards_stream(video: str, S: float, k_rel: float, out: Path) -> dict:
    inj = None
    if video == INJECT:
        import inject
        inj = inject.build(out)
        tr, src = inj["tracks"], inj["mp4"]
    else:
        tr, src = pk.tracks(video), video
    cams = L.load_cams(BASE_OF.get(video, video))
    fs = FrameSource(src)
    registrar = L.Registrar(cams)
    t_det = time.perf_counter()
    alerts = lc.stopped_alerts(tr.rows, tr.fps, S=S, k_rel=k_rel)
    det_s = time.perf_counter() - t_det
    alerts.sort(key=lambda a: (a["t_alert"], a["box"][0]))
    for a in alerts:
        t0 = time.perf_counter()
        a["fid"], a["_frame"] = fs.at(a["t_alert"])
        a["seek_ms"] = (time.perf_counter() - t0) * 1000   # offline replay only: a live system already holds the frame
    seg = seg_for_alerts(alerts, fs, out / "_seg_cache.json")
    cards = []
    for a in alerts:
        frame = fs.by_fid(a["fid"])
        t0 = time.perf_counter()
        cam, reg = registrar.lookup(a["fid"], frame)
        sg = seg[a["fid"]]
        prof = L.match_profile(sg["dets"], a["box"])
        c = L.make_call(frame, a["box"], cam, reg, vehicles_at(tr, a["t_alert"]), a["conf"], prof)
        a.update(c)
        a["seg_ms"] = sg["ms"]
        a["cam_obj"], a["reg"] = cam, reg
        a["_crop"], a["_frame"] = L.crop_box(frame, a["box"]), frame
        a["card_ms"] = (time.perf_counter() - t0) * 1000 + sg["ms"]   # + the seg pass on this frame (inference only)
        cards.append(a)
    return {"tr": tr, "cams": cams, "cards": cards, "det_s": det_s, "mode": "stream", "fs": fs, "src": src, "inject": inj,
            "registrar": registrar}


def merge_presets(cards: list[dict], ctx: dict) -> tuple[list[dict], list[dict]]:
    """PTZ: the same parked car is alerted once per preset. Map every alert's ground point into the first preset's reference
    frame (ORB+RANSAC homography between the preset reference frames) and keep ONE alert per physical car. Returns (kept, dropped)."""
    cams = ctx["cams"]
    if len(cams) < 2:
        return cards, []
    canon = cams[0]
    to_canon = {canon.id: np.eye(3)}
    for c in cams[1:]:
        r = L.register(c.ref, canon)             # H: canon ref -> c ref
        if r and r["inliers"] >= 60:
            to_canon[c.id] = np.linalg.inv(r["H"])
    ctx["preset_maps"] = {k: bool(v is not None) for k, v in to_canon.items()}

    def canon_pts(a):
        if a["cam_obj"] is None or a["reg"] is None or a["cam"] not in to_canon:
            return None
        P = np.array(a["foot_px"], np.float32).reshape(-1, 1, 2)
        P = cv2.perspectiveTransform(P, np.linalg.inv(a["reg"]["H"]))
        P = cv2.perspectiveTransform(P, to_canon[a["cam"]].astype(np.float64)).reshape(-1, 2)
        return P

    kept: list[dict] = []
    dropped: list[dict] = []
    for a in sorted(cards, key=lambda a: (a["t_alert"], a["box"][0])):
        pa = canon_pts(a)
        a["_cp"] = pa
        match = None
        if pa is not None:
            ga, wa = pa.mean(0), float(np.hypot(*(pa[-1] - pa[0])))
            for k in kept:
                if k["_cp"] is None or k["cam"] == a["cam"]:     # only ACROSS presets: within one preset the detector already dedupes
                    continue
                gk, wk = k["_cp"].mean(0), float(np.hypot(*(k["_cp"][-1] - k["_cp"][0])))
                if np.hypot(*(ga - gk)) < 0.6 * max(1.0, 0.5 * (wa + wk)) and a["cls"] == k["cls"]:
                    match = k
                    break
        if match is None:
            a["seen_in"] = [a["cam"]]
            kept.append(a)
        else:
            match["seen_in"] = sorted(set(match.get("seen_in", [match["cam"]]) + [a["cam"]]))
            if match["call"] == CANT and a["call"] != CANT:       # keep the better view of the same car
                a["seen_in"] = match["seen_in"]
                kept[kept.index(match)] = a
                match["dup_obj"] = a
                dropped.append(match)
            else:
                a["dup_obj"] = match
                dropped.append(a)
    for a in kept + dropped:
        a.pop("_cp", None)
    return sorted(kept, key=lambda a: (a["t_alert"], a["box"][0])), dropped


def group_queues(cards: list[dict]) -> list[list[dict]]:
    """IN LANE alerts that are stacked in the same lane zone of the same camera and stopped within QUEUE_DT s of each other.
    Connected components with >= QUEUE_MIN members are a queue (red light / stop-go), not separate incidents."""
    inl = [c for c in cards if c["call"] == IN_LANE]
    par = list(range(len(inl)))

    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    for i, a in enumerate(inl):
        for j in range(i + 1, len(inl)):
            b = inl[j]
            if a["cam"] != b["cam"] or a["zone"] != b["zone"] or abs(a["t_stop"] - b["t_stop"]) > QUEUE_DT:
                continue
            ca = ((a["box"][0] + a["box"][2]) / 2, (a["box"][1] + a["box"][3]) / 2)
            cb = ((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2)
            wm = 0.5 * ((a["box"][2] - a["box"][0]) + (b["box"][2] - b["box"][0]))
            if np.hypot(ca[0] - cb[0], ca[1] - cb[1]) <= QUEUE_GAP * wm:
                par[f(i)] = f(j)
    comp: dict[int, list[dict]] = {}
    for i, a in enumerate(inl):
        comp.setdefault(f(i), []).append(a)
    return [m for m in comp.values() if len(m) >= QUEUE_MIN]


def finish_cards(video: str, ctx: dict, out: Path, use_vlm: bool, use_clips: bool) -> tuple[list[dict], list[dict]]:
    """Dedupe, queue collapse, ordering, images (timed into card_ms), second opinions and clips.
    Returns (units, alerts): units are the cards an operator sees; alerts are every stopped vehicle the detector raised."""
    alerts = ctx["cards"]
    ctx["merged"] = []
    if ctx["mode"] == "stream":
        alerts, ctx["merged"] = merge_presets(alerts, ctx)
    ctx["cards"] = alerts
    for e, c in enumerate(sorted(alerts, key=lambda c: (c["t_alert"], c["box"][0])), 1):
        c["eid"] = f"E{e:02d}"
    cdir = out / "cards"
    cdir.mkdir(parents=True, exist_ok=True)
    for old in list(cdir.glob("*")):
        old.unlink()
    # queue collapse
    units: list[dict] = []
    in_q: set[int] = set()
    for gi, grp in enumerate(group_queues(alerts), 1):
        grp.sort(key=lambda c: (c["t_alert"], c["box"][0]))
        lead = grp[-1]                      # the stop that completed the queue: all earlier members are still in view
        for m in grp:
            in_q.add(id(m))
            m["queue_id"] = f"QU{gi}"
        t_sort = sorted(m["t_alert"] for m in grp)
        span = max(m["t_stop"] for m in grp) - min(m["t_stop"] for m in grp)
        u = {"call": QUEUE, "is_unit": True, "members": [m["eid"] for m in grp], "n_members": len(grp), "boxes": [m["box"] for m in grp],
             "box": lead["box"], "cls": f"{len(grp)} vehicles", "t_stop": min(m["t_stop"] for m in grp), "t_alert": t_sort[-1],
             "dur": float(np.mean([m["dur"] for m in grp])), "conf": round(float(np.mean([m["conf"] for m in grp])), 2),
             "zone": lead["zone"], "cam": lead["cam"], "lane_frac": round(float(np.mean([m["lane_frac"] for m in grp])), 2),
             "kerb_frac": 0.0, "vis": [], "reg_inliers": lead["reg_inliers"], "fid": lead["fid"], "cam_obj": lead["cam_obj"],
             "reg": lead["reg"], "_frame": lead["_frame"], "_crop": lead["_crop"], "seek_ms": lead["seek_ms"],
             "card_ms": max(m["card_ms"] for m in grp), "foot_px": None, "slide_kind": lead.get("slide_kind"),
             "reason": (f"{len(grp)} vehicles stopped in '{lead['zone']}' within {span:.0f} s of each other, stacked along the lane: "
                        f"looks like a queue (signal or stop-go), not {len(grp)} incidents. Collapsed into one low-severity card."),
             "mem_cards": grp}
        units.append(u)
    units += [c for c in alerts if id(c) not in in_q]
    units.sort(key=L.severity_key)
    for i, c in enumerate(units, 1):
        c["id"] = f"Q{i:02d}"
        t0 = time.perf_counter()
        pk.save_jpg(c["_crop"] if max(c["_crop"].shape[:2]) <= 360 else pk.fit(c["_crop"], 360), cdir / f"{c['id']}_crop.jpg", 85)
        ann = annotated(c["_frame"], c["cam_obj"], c["reg"], c, extra=c.get("boxes"))
        pk.save_jpg(ann, cdir / f"{c['id']}_frame.jpg", 82)
        c["card_ms"] += (time.perf_counter() - t0) * 1000
        c["crop_path"], c["frame_path"] = f"cards/{c['id']}_crop.jpg", f"cards/{c['id']}_frame.jpg"
        c["_ann"] = ann
        if c.get("is_unit"):
            for m in c["mem_cards"]:
                th = m["_crop"] if max(m["_crop"].shape[:2]) <= 200 else pk.fit(m["_crop"], 200)
                pk.save_jpg(th, cdir / f"{m['eid']}_crop.jpg", 80)
                m["crop_path"] = f"cards/{m['eid']}_crop.jpg"
    # Cosmos second opinion: IN LANE and CAN'T SEE first, then the most confident AT KERB, within the call budget
    for c in units:
        c["vlm"] = None
    if use_vlm:
        order = [c for c in units if c["call"] in (IN_LANE, CANT)] + [c for c in units if c["call"] == AT_KERB]
        for c in order[:VLM_BUDGET]:
            c["vlm"] = L.cosmos_second_opinion(c["_frame"], c["box"])
    # clips
    c_used = {k: 0 for k in L.SEV}
    for c in units:
        c["clip_path"] = None
        if use_clips and ctx["mode"] == "stream" and c_used[c["call"]] < CLIP_BUDGET[c["call"]]:
            c_used[c["call"]] += 1
            try:
                write_clip(ctx["src"], video, c, out / "cards" / f"{c['id']}_clip.mp4")
                c["clip_path"] = f"cards/{c['id']}_clip.mp4"
            except Exception as e:  # clip is a convenience, never fatal
                print(f"[clip] {c['id']} failed: {e}", file=sys.stderr)
    return units, alerts


# ------------------------------------------------------------------------------------------- slide pipeline
def lisbon_slides(video: str, out: Path) -> list[dict]:
    """The Lisbon clip is a ~1600-slide photo montage (0.3-0.5 s per photo), not a fixed camera: segment it into
    slides by hard cuts (mean abs diff of 64x48 thumbs > 45) and cache."""
    f = out / "_slides.json"
    if f.exists():
        return json.loads(f.read_text())
    tr = pk.tracks(video)
    prev, ts, cuts = None, [], []
    for i, t, fr in pk.iter_frames(video, stride=tr.stride):
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None and np.abs(g - prev).mean() > 45:
            cuts.append(t)
        prev = g
        ts.append(t)
    edges = [0.0] + cuts + [tr.duration]
    slides = [{"t0": a, "t1": b} for a, b in zip(edges[:-1], edges[1:])]
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(slides))
    return slides


def pick_primary(tr, t0: float, t1: float, W: int, H: int):
    """Largest confident vehicle that persists (IoU-matched) through the slide; its median box."""
    r = tr.rows[(tr.rows["t"] >= t0 + 0.04) & (tr.rows["t"] < t1 - 0.04) & tr.rows["name"].isin(lc.VEH) & (tr.rows["conf"] >= 0.5)]
    if r.empty:
        return None
    nfr = r["t"].nunique()
    best = None
    for rr in r.sort_values("bw", ascending=False).head(6).itertuples():
        box = [rr.x1, rr.y1, rr.x2, rr.y2]
        area = (rr.x2 - rr.x1) * (rr.y2 - rr.y1) / (W * H)
        if area < 0.012:
            continue
        m = r[(np.abs(r["x1"] - rr.x1) < 12) & (np.abs(r["y1"] - rr.y1) < 12) & (np.abs(r["x2"] - rr.x2) < 12) &
              (np.abs(r["y2"] - rr.y2) < 12)]
        if m["t"].nunique() >= max(2, 0.6 * nfr):
            sc = area * float(m["conf"].mean())
            if best is None or sc > best[0]:
                best = (sc, m[["x1", "y1", "x2", "y2"]].median().tolist(), str(rr.name), float(m["conf"].mean()), m["t"].nunique())
    return best


def build_cards_slides(video: str, out: Path, window: float = 30.0) -> dict:
    """SIMULATED trigger: a slide is treated as a stopped vehicle. One slide per `window` seconds (the first slide whose
    primary vehicle persists), chosen BEFORE any geometry was drawn; geometry per selected slide is in zones.json."""
    tr = pk.tracks(video)
    fs = FrameSource(video)
    slides = lisbon_slides(video, out)
    sel = []
    for k in range(int(np.ceil(tr.duration / window))):
        for s in slides:
            if s["t0"] < k * window or s["t0"] >= (k + 1) * window or s["t1"] - s["t0"] < 0.25:
                continue
            p = pick_primary(tr, s["t0"], s["t1"], tr.w, tr.h)
            if p:
                sel.append({"win": k, **s, "primary": p})
                break
    data = json.loads((HERE / "zones.json").read_text(encoding="utf-8"))
    refs = {d["id"]: d for d in data["cameras"] if d["video"] == video}
    items = [{"cid": f"lis_{s['win']:02d}", "kind": "deterministic sample", "t0": s["t0"], "t1": s["t1"],
              "primary": s["primary"]} for s in sel]
    mp = HERE / "lisbon_manual_picks.json"
    if mp.exists():
        for m in json.loads(mp.read_text(encoding="utf-8")):
            items.append({"cid": f"lisx_{m['pool_idx']:03d}", "kind": "hand-picked (looked in-lane-ish)", "t0": m["slide_t0"],
                          "t1": m["slide_t1"], "primary": (0.0, m["box"], m["cls"], m["conf"], 3)})
    todo = []
    for s in items:
        if s["cid"] not in refs:
            continue
        tm = round((s["t0"] + s["t1"]) / 2, 2)
        t_seek = time.perf_counter()
        fid, frame = fs.at(tm)
        todo.append((s, tm, fid, frame.copy(), (time.perf_counter() - t_seek) * 1000))
    frames = {fid: fr for _, _, fid, fr, _ in todo}
    seg = L.seg_profiles(list(frames), lambda f: frames[f], out / "_seg_cache.json")
    cams, cards = [], []
    for s, tm, fid, frame, a_seek in todo:
        cid = s["cid"]
        t_ref = round(s["t0"] + 0.1, 2)
        cam = L.Cam(refs[cid], pk.frame_at(video, t_ref))
        cams.append(cam)
        _, p_box, p_cls, p_conf, p_n = s["primary"]
        sg = seg[fid]
        t0 = time.perf_counter()
        reg = L.register(frame, cam)
        prof = L.match_profile(sg["dets"], p_box)
        c = L.make_call(frame, p_box, cam, reg, vehicles_at(tr, tm), p_conf, prof)
        a = {"t_alert": tm, "t_stop": s["t0"], "t_end": s["t1"], "dur": s["t1"] - s["t0"], "box": p_box, "cls": p_cls,
             "conf_det": p_conf, "tids": [], "presence": 1.0, "slide_kind": s["kind"], **c}
        crop = L.crop_box(frame, p_box)
        a["fid"], a["cam_obj"], a["reg"], a["_crop"], a["_frame"] = fid, cam, reg, crop, frame
        a["seek_ms"] = a_seek
        a["seg_ms"] = sg["ms"]
        a["card_ms"] = (time.perf_counter() - t0) * 1000 + sg["ms"]
        cards.append(a)
    return {"tr": tr, "cams": cams, "cards": cards, "det_s": 0.0, "mode": "slides", "fs": fs, "src": video,
            "n_slides": len(slides), "slides": slides}


# ------------------------------------------------------------------------------------------------- outputs
def chip(call: str, conf: float | None = None) -> str:
    t = call + (f" {conf:.0%}" if conf is not None else "")
    return f'<span class="tag" style="background:{L.CALL_HEX[call]}">{html.escape(t)}</span>'


QCSS = """<style>
.q{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.qc{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--c);border-radius:10px;padding:10px}
.qc img,.qc video{width:100%;border-radius:6px;margin:6px 0}.qc .m{color:var(--muted);font-size:12px}
.qc .r{font-size:13px;margin:4px 0}.agree{font-size:12px;padding:1px 6px;border-radius:8px;background:var(--line)}
.qc .th img{width:64px;display:inline-block;margin:2px 2px 0 0}
details>summary{cursor:pointer;font-weight:600;margin:10px 0}
.sim{background:#7a3b00;color:#fff;padding:6px 10px;border-radius:6px;font-weight:600}
</style>"""


def card_html(c: dict, show_clip: bool = True) -> str:
    v = c.get("vlm")
    if v and v.get("call"):
        agree = "agrees" if v["call"] == c["call"] else "differs"
        vl = (f'<details class="r"><summary class="m">Cosmos second opinion (advisory, collapsed: {agree})</summary>'
              f'{chip(v["call"])} <span class="agree">{agree}</span> '
              f'<span class="m">{html.escape(v["reason"])} ({v["seconds"]} s, advisory only)</span></details>')
    elif v:
        vl = '<div class="r m">Cosmos second opinion: no usable answer</div>'
    elif c["call"] == QUEUE:
        vl = '<div class="r m">Cosmos second opinion: not requested for a queue card</div>'
    else:
        vl = '<div class="r m">Cosmos second opinion: not requested (budget)</div>'
    media = (f'<video src="{c["clip_path"]}" controls muted playsinline preload="none" poster="{c["frame_path"]}"></video>'
             if show_clip and c.get("clip_path") else f'<a href="{c["frame_path"]}"><img src="{c["frame_path"]}" loading="lazy"></a>')
    extra = ""
    if c["call"] == QUEUE:
        extra = '<div class="th">' + "".join(f'<img src="{m["crop_path"]}" title="{m["eid"]}" loading="lazy">' for m in c["mem_cards"]) + "</div>"
        thumb = ""
    else:
        thumb = f'<img src="{c["crop_path"]}" style="width:120px;display:inline-block;margin:0" loading="lazy">'
    seen = f' &middot; same car seen in {len(c["seen_in"])} presets' if len(c.get("seen_in", [])) > 1 else ""
    return (f'<div class="qc" style="--c:{L.CALL_HEX[c["call"]]}"><b>{c["id"]}</b> {chip(c["call"], c["conf"])}'
            f'<div class="m">{html.escape(c["cls"])} &middot; {html.escape(stop_txt(c))} &middot; '
            f'{"" if c.get("slide_kind") or c["t_stop"] < 0.5 else f"dwell {c["dur"]:.1f} s &middot; "}'
            f'alert {pk.mmss(c["t_alert"])} &middot; card in {c["card_ms"]:.0f} ms{seen}</div>{media}'
            f'<div class="r">{html.escape(c["reason"])}</div>{vl}{extra}{thumb}</div>')


def _pil_text(bg, items):
    return L.text_pil(bg, items)


def hero(video: str, units: list[dict], alerts: list[dict], ctx: dict, summary: dict, out: Path) -> Path:
    """1280x720 product shot: the view with every alert boxed + the queue + counts + verdict stamp."""
    Wd, Ht = 1280, 720
    bg = np.full((Ht, Wd, 3), (24, 19, 14), np.uint8)
    items = [(28, 18, "LANE CALL", 34, (255, 255, 255), True),
             (230, 28, "stopped-vehicle alert triage for control rooms", 20, (160, 170, 180), False)]
    n = len(units)
    cnt = summary["by_card"]
    simulated = video == INJECT
    if len(alerts) == 0:
        fr = ctx["fs"].at(min(ctx["tr"].duration - 0.1, ctx["tr"].duration / 2))[1]
        cam = ctx["cams"][0] if ctx["cams"] else None
        reg = L.register(fr, cam) if cam else None
        img = L.draw_zones(fr, cam, reg["H"]) if reg else fr
        img = pk.fit(img, 756)
        bg[84:84 + img.shape[0], 28:28 + 756] = img
        from PIL import Image, ImageDraw
        im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((60, 110, 430, 190), 12, fill=(46, 158, 91))
        d.text((84, 118), "ALL CLEAR", font=L._font(52), fill=(255, 255, 255))
        bg = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        items += [(812, 96, "0 stopped-vehicle alerts", 34, (255, 255, 255), True),
                  (812, 146, f"{summary['n_vehicle_tracks']} vehicle tracks, {summary['duration_s']:.0f} s of traffic", 20, (170, 180, 190), False),
                  (812, 180, f"none held still for >= {summary['S']:g} s", 20, (170, 180, 190), False),
                  (812, 240, "Lane geometry loaded (red = running lanes,", 18, (150, 160, 170), False),
                  (812, 264, "blue = shoulder / verge). Expected result for", 18, (150, 160, 170), False),
                  (812, 288, "free-flowing motorway traffic: quiet queue.", 18, (150, 160, 170), False),
                  (812, 340, "Same clip + 3 SIMULATED stopped cars:", 18, (230, 190, 120), True),
                  (812, 364, "out/motorway_a40_inject (2 IN LANE, 1 AT KERB)", 18, (230, 190, 120), False)]
        cv2.rectangle(bg, (0, 652), (Wd, Ht), (36, 30, 24), -1)
        items += [(28, 662, "0 stopped-vehicle alerts  ->", 24, (255, 255, 255), True),
                  (360, 662, "0 IN LANE", 26, (235, 50, 45), True), (590, 662, "0 CAN'T SEE", 26, (235, 170, 0), True),
                  (840, 662, "0 AT KERB", 26, (60, 170, 60), True),
                  (28, 692, "quiet queue = no operator time spent  |  real footage, nothing stopped (nothing injected in this run)", 15,
                   (150, 160, 170), False)]
        bg = L.text_pil(bg, items)
        p = out / "hero.jpg"
        pk.save_jpg(bg, p, 90)
        return p
    top = units[0]
    qunit = next((u for u in units if u["call"] == QUEUE), None)
    if qunit is not None and cnt.get(IN_LANE, 0) <= 1:
        top = qunit                      # the story of this clip is the collapse: show the view the queue card stands for
    nothing_in_lane = cnt.get(IN_LANE, 0) == 0 and cnt.get(QUEUE, 0) == 0
    # left image: the view of the top unit with every alert in it boxed in its own call colour
    fid = top["fid"]
    grp = [a for a in alerts if a["fid"] == fid and a.get("cam") == top.get("cam")]
    if nothing_in_lane:
        fid = max({c["fid"] for c in alerts}, key=lambda f: sum(1 for c in alerts if c["fid"] == f))
        grp = [c for c in alerts if c["fid"] == fid]
    if len(grp) >= 2 and ctx["mode"] != "slides":
        fr = grp[0]["_frame"].copy()
        if grp[0]["cam_obj"] is not None and grp[0]["reg"] is not None:
            fr = L.draw_zones(fr, grp[0]["cam_obj"], grp[0]["reg"]["H"])
        for c in grp:
            col = L.CALL_COL[QUEUE if c.get("queue_id") else c["call"]]
            x1, y1, x2, y2 = (int(v) for v in c["box"])
            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 4 if c["call"] == IN_LANE and not c.get("queue_id") else 3)
            draw_foot(fr, c, col)
        img = pk.fit(fr, 756)
        sc = img.shape[1] / fr.shape[1]
        if not nothing_in_lane:
            for c in grp:
                if (c["call"] == IN_LANE or c["call"] == AT_KERB) and not c.get("queue_id"):
                    lab = c["call"]
                    sx, sy = int(c["box"][0] * sc), int(c["box"][1] * sc)
                    img = L.stamp(img, lab, c["conf"], (max(2, min(sx, 756 - 230)), max(2, sy - 40)), size=17)
        cap1 = (f"{len(grp)} stopped vehicles in this view: " + ", ".join(
            f"{sum(1 for c in grp if (QUEUE if c.get('queue_id') else c['call']) == k)} {k}" for k in (IN_LANE, QUEUE, CANT, AT_KERB)
            if any((QUEUE if c.get("queue_id") else c["call"]) == k for c in grp)))
        cap2 = ("white line = ground contact (seg mask); red = running lane, blue = shoulder"
                if not nothing_in_lane else
                "red band = running-lane polygon drawn once per camera preset; every stopped car is outside it")
    else:
        img = pk.fit(top["_ann"], 756)
        cap1 = f"{top['id']}  {top['cls']}, {stop_txt(top)}"
        cap2 = top["reason"][:92]
    if top["call"] == QUEUE:
        cap1 = f"{top['id']}  QUEUE of {top['n_members']}: one card instead of {top['n_members']} IN LANE alerts"
        cap2 = "stacked in one lane and stopped together: signal or stop-go, not separate incidents"
    if img.shape[0] > 504:
        img = cv2.resize(img, (int(756 * 504 / img.shape[0]), 504), interpolation=cv2.INTER_AREA)
    x0 = 28 + (756 - img.shape[1]) // 2
    bg[84:84 + img.shape[0], x0:x0 + img.shape[1]] = img
    banner = None
    if nothing_in_lane:
        banner, bcol = "NOTHING IN LANE", (46, 158, 91)
    elif simulated:
        banner, bcol = "SIMULATED STOPS", (20, 90, 170)
    elif top["call"] == QUEUE:
        banner, bcol = f"{top['n_members']} CARS = 1 QUEUE CARD", (140, 110, 200)
    if banner:
        from PIL import Image, ImageDraw
        im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
        wbn = 400 if nothing_in_lane else (430 if simulated else 560)
        d.rounded_rectangle((x0 + 12, 96, x0 + 12 + wbn, 96 + 62), 10, fill=bcol)
        d.text((x0 + 28, 102), banner, font=L._font(38), fill=(255, 255, 255))
        bg = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    items.append((28, 84 + img.shape[0] + 8, cap1[:100], 19, (230, 235, 240), True))
    items.append((28, 84 + img.shape[0] + 34, cap2[:118], 17, (160, 170, 180), False))
    y = 112
    if nothing_in_lane:
        # lead with the good news: the auto-lowered pile, then only what is left for the operator
        n_k, n_c = cnt.get(AT_KERB, 0), cnt.get(CANT, 0)
        from PIL import Image, ImageDraw
        im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((808, 84, 1252, 214), 12, fill=(30, 74, 48), outline=(60, 170, 60), width=3)
        bg = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        items += [(826, 92, f"{n_k} AT KERB", 44, (90, 215, 130), True),
                  (826, 148, "auto-lowered: parked in bays, nobody paged", 19, (200, 235, 210), False),
                  (826, 176, f"({len(summary.get('merged', []))} repeat alerts of the same cars merged)" if summary.get("merged") else "", 15,
                   (150, 190, 165), False),
                  (812, 234, f"{n_c} CAN'T SEE  needs a human glance", 22, (235, 170, 0), True),
                  (812, 264, "cars cut off by the frame edge: ground contact", 16, (170, 180, 190), False),
                  (812, 286, "not visible, so the system will not guess", 16, (170, 180, 190), False)]
        bg = L.text_pil(bg, items)
        items = []
        y = 318
        shown = [c for c in units if c["call"] == CANT][:3]
    else:
        items.append((812, 78, "ALERT QUEUE  (IN LANE first)", 20, (160, 170, 180), True))
        bg = L.text_pil(bg, items)
        items = []
        shown = units[:5]
    for c in shown:
        th = pk.fit(c["_crop"], 112)
        th = th[:84] if th.shape[0] >= 84 else np.pad(th, ((0, 84 - th.shape[0]), (0, 0), (0, 0)))
        bg[y:y + 84, 812:812 + 112] = th
        col = L.CALL_COL[c["call"]]
        cv2.rectangle(bg, (808, y - 2), (812 + 112 + 4, y + 86), col, 3)
        head = f"{c['id']}  {c['call']}  {c['conf']:.0%}" + (f"  x{c['n_members']}" if c["call"] == QUEUE else "")
        items += [(936, y - 2, head, 21, tuple(int(v) for v in L.CALL_COL[c["call"]][::-1]), True),
                  (936, y + 28, f"{c['cls']}, {stop_txt(c)}"[:38], 17, (200, 207, 214), False),
                  (936, y + 50, (c["reason"][:34] + ".." if len(c["reason"]) > 34 else c["reason"]), 15, (140, 150, 160), False)]
        y += 100 if not nothing_in_lane else 96
    rest = n - len(shown)
    if rest > 0 and not nothing_in_lane:
        k = {c_: sum(1 for c in units[len(shown):] if c["call"] == c_) for c_ in (IN_LANE, CANT, QUEUE, AT_KERB)}
        items.append((812, y + 4, f"+{rest} more: " + ", ".join(f"{v} {kk}" for kk, v in k.items() if v), 18, (180, 188, 196), True))
    bg = L.text_pil(bg, items)
    if simulated:
        from PIL import Image, ImageDraw
        im = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((808, 440, 1252, 636), 12, fill=(33, 40, 50), outline=(70, 90, 115), width=2)
        bg = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        bg = L.text_pil(bg, [(826, 450, "CONTROL: same clip, nothing pasted", 18, (170, 190, 215), True),
                             (826, 478, f"{summary.get('control_clean_alerts', 0)} alerts", 40, (90, 215, 130), True),
                             (826, 530, "real A40 traffic never holds still for 5 s", 16, (170, 180, 190), False),
                             (826, 556, "SIMULATED = 3 real cars cut out of this clip", 16, (230, 190, 120), False),
                             (826, 578, "and pasted static (inject.py). Real: footage,", 16, (230, 190, 120), False),
                             (826, 600, "detector, 5 s rule, geometry, call.", 16, (230, 190, 120), False)])
    # footer band
    cv2.rectangle(bg, (0, 652), (Wd, Ht), (36, 30, 24), -1)
    na = len(alerts)
    lead = f"{na} stopped vehicles" + (f" -> {n} cards" if n != na else "") + "  ->"
    f_items = [(28, 662, lead, 22, (255, 255, 255), True)]
    x = 400 if n != na else 360
    for call in (IN_LANE, CANT, QUEUE, AT_KERB):
        if call == QUEUE and not cnt.get(QUEUE):
            continue
        f_items.append((x, 662, f"{cnt.get(call, 0)} {call}", 24, tuple(int(v) for v in L.CALL_COL[call][::-1]), True))
        x += 200 if call != CANT else 230
    mode = ("trigger SIMULATED (slideshow stills)" if ctx["mode"] == "slides"
            else "3 pasted cars are SIMULATED, detector + geometry are real" if simulated else "live-style stream")
    f_items.append((28, 692, f"alert -> card: median {summary['card_ms_median']:.0f} ms incl. seg pass (budget 2000)  |  {mode}  |  Cosmos is advisory", 15,
                    (150, 160, 170), False))
    bg = L.text_pil(bg, f_items)
    p = out / "hero.jpg"
    pk.save_jpg(bg, p, 90)
    return p


def eyeball_sheets(video: str, cards: list[dict], out: Path) -> list[str]:
    """Numbered, call-free tiles (box only) so the true label can be judged without seeing the system's answer."""
    paths = []
    cards = sorted(cards, key=lambda c: c["eid"])
    for k in range(0, len(cards), 16):
        tiles, labs = [], []
        for c in cards[k:k + 16]:
            f = c["_frame"].copy()
            x1, y1, x2, y2 = (int(v) for v in c["box"])
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bw = max(x2 - x1, 80)
            half = int(bw * 1.7)
            hh = int(half * 0.75)
            a, b = max(0, cx - half), max(0, cy - hh)
            e, d = min(f.shape[1], cx + half), min(f.shape[0], cy + hh)
            tiles.append(f[b:d, a:e]); labs.append(c["eid"])
        p = out / f"eyeball_{k // 16 + 1}.jpg"
        pk.save_jpg(pk.sheet(tiles, labs, cols=4, tile_w=330), p, 82)
        paths.append(p.name)
    return paths


def callview_sheets(video: str, cards: list[dict], out: Path, only: tuple = (IN_LANE,)) -> list[str]:
    """Tiles WITH the ground-contact line and the call, for checking the rule's reasoning by eye (not the ground truth)."""
    sel = [c for c in sorted(cards, key=lambda c: c["eid"]) if c["call"] in only]
    paths = []
    for k in range(0, len(sel), 12):
        tiles, labs = [], []
        for c in sel[k:k + 12]:
            f = c["_frame"].copy()
            if c["cam_obj"] is not None and c["reg"] is not None:
                f = L.draw_zones(f, c["cam_obj"], c["reg"]["H"])
            x1, y1, x2, y2 = (int(v) for v in c["box"])
            col = L.CALL_COL[c["call"]]
            cv2.rectangle(f, (x1, y1), (x2, y2), col, 3)
            draw_foot(f, c, col)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = int(max(x2 - x1, 80) * 1.7); hh = int(half * 0.75)
            tiles.append(f[max(0, cy - hh):min(f.shape[0], cy + hh), max(0, cx - half):min(f.shape[1], cx + half)])
            labs.append(f"{c['eid']} {c['call']} {c['conf']:.0%} lane {c['lane_frac']}")
        p = out / f"callview_{k // 12 + 1}.jpg"
        pk.save_jpg(pk.sheet(tiles, labs, cols=4, tile_w=330), p, 82)
        paths.append(p.name)
    return paths


def kept_of(m: dict) -> dict:
    """The surviving alert a merged repeat ended up in (follow the chain: a kept alert can itself be replaced by a better view)."""
    o = m["dup_obj"]
    while "eid" not in o and o.get("dup_obj") is not None:
        o = o["dup_obj"]
    return o


def merge_sheet(ctx: dict, out: Path, limit: int = 8) -> list[str]:
    """Kept alert next to the repeat alert that was merged into it (same physical car in another PTZ preset), for eyeballing."""
    pairs = [m for m in ctx.get("merged", []) if m.get("dup_obj") is not None][:limit]
    if not pairs:
        return []
    tiles, labs = [], []
    for m in pairs:
        for c, tag in ((kept_of(m), "KEPT"), (m, "MERGED")):
            f = c["_frame"].copy()
            x1, y1, x2, y2 = (int(v) for v in c["box"])
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = int(max(x2 - x1, 80) * 2.2); hh = int(half * 0.75)
            tiles.append(f[max(0, cy - hh):min(f.shape[0], cy + hh), max(0, cx - half):min(f.shape[1], cx + half)])
            labs.append(f"{tag} {c['cam']} t={c['t_alert']:.0f}s")
    p = out / "merged_pairs.jpg"
    pk.save_jpg(pk.sheet(tiles, labs, cols=4, tile_w=330), p, 82)
    return [p.name]


def numbered_frames(video: str, cards: list[dict], out: Path) -> list[str]:
    """Full frames per camera with every alert boxed and numbered (no calls) for ground-truth review."""
    paths = []
    byc: dict = {}
    for c in cards:
        byc.setdefault((c["cam"], c["fid"]), []).append(c)
    for (cam, _), cs in byc.items():
        if len(cs) < 3:
            continue
        f = cs[0]["_frame"].copy()
        for c in cs:
            x1, y1, x2, y2 = (int(v) for v in c["box"])
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 0, 255), 2)
            pk.label(f, c["eid"][1:], (x1 + 2, y1 + 2), scale=0.5, bg=(0, 0, 160))
        p = out / f"numbered_{cam}_{cs[0]['fid']}.jpg"
        pk.save_jpg(f, p, 85)
        paths.append(p.name)
    return paths


def accuracy(video: str, alerts: list[dict]) -> dict | None:
    gt_f = HERE / "ground_truth.json"
    if not gt_f.exists():
        return None
    gt = json.loads(gt_f.read_text(encoding="utf-8")).get(video)
    if not gt:
        return None
    rows, ok, n, stopped_bad, per = [], 0, 0, 0, {}
    for c in alerts:
        cx, cy = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
        m = [g for g in gt["alerts"] if abs(g["t_alert"] - c["t_alert"]) < 1.0 and abs(g["cx"] - cx) < 12 and abs(g["cy"] - cy) < 12]
        if not m:
            continue
        g = m[0]
        truth = g["truth"]
        n += 1
        right = (truth == c["call"]) or (truth == "UNJUDGEABLE" and c["call"] == CANT)
        ok += int(right)
        stopped_bad += int(not g.get("really_stopped", True))
        p = per.setdefault(truth, [0, 0]); p[1] += 1; p[0] += int(right)
        rows.append({"id": c["eid"], "call": c["call"], "truth": truth, "right": right})
    return {"judged": n, "right": ok, "acc": round(ok / n, 3) if n else None, "coverage": f"{n}/{len(alerts)}",
            "false_stopped": stopped_bad, "by_truth": {k: f"{v[0]}/{v[1]}" for k, v in per.items()}, "rows": rows}


def write_report(video: str, units: list[dict], alerts: list[dict], ctx: dict, summary: dict, out: Path, sheets: list[str]) -> Path:
    tr = ctx["tr"]
    rep = pk.Report(out / "report.html", f"LANE CALL: {video}",
                    "Stopped-vehicle alert triage. Every card: 8 s clip, boxed vehicle, lane geometry call, Cosmos second opinion.")
    rep.credit(BASE_OF.get(video, video))
    rep.html(QCSS)
    bc = summary["by_card"]
    rep.kv({"stopped vehicles": summary["n_alerts"], "cards": summary["n_cards"], "IN LANE": bc.get(IN_LANE, 0),
            "CAN'T SEE": bc.get(CANT, 0), "QUEUE": bc.get(QUEUE, 0), "AT KERB": bc.get(AT_KERB, 0),
            "alert to card (median ms)": round(summary["card_ms_median"]), "p95 ms": round(summary["card_ms_p95"]),
            "clip": f"{tr.duration:.0f} s, S = {summary['S']:g} s"})
    rep.image(out / "hero.jpg", "Hero: top of the queue")
    if ctx["mode"] == "slides":
        rep.html('<p class="sim">The Lisbon clip is a ~%d-slide photo montage, not a camera feed. The alert trigger is SIMULATED (each '
                 'selected slide stands in for a stopped vehicle); geometry was drawn per slide; the ground-contact line, call, confidence, '
                 'registration and card logic are real.</p>' % ctx["n_slides"])
    elif video == INJECT:
        rep.html('<p class="sim">SIMULATED: 3 real cars cut out of this clip are pasted in as static stopped vehicles (inject.py). The '
                 'footage, the YOLO detector run on the pasted pixels, the 5 s stopped rule, the registration, the geometry and the call are real.</p>')
    else:
        spans = [(c["t_stop"], c["t_alert"], {IN_LANE: "alert", CANT: "amber", AT_KERB: "ok", QUEUE: "amber"}[c["call"]],
                  f"{c['id']} {c['call']} {c['cls']}") for c in units]
        rep.h2("Alert timeline (stop -> alert)").timeline(spans, tr.duration, "red = IN LANE, amber = CAN'T SEE / QUEUE, green = AT KERB")
    if video == INJECT:
        spans = [(c["t_stop"], c["t_alert"], {IN_LANE: "alert", CANT: "amber", AT_KERB: "ok", QUEUE: "amber"}[c["call"]],
                  f"{c['id']} {c['call']} {c['cls']}") for c in units]
        rep.h2("Alert timeline (stop -> alert)").timeline(spans, tr.duration, "red = IN LANE, green = AT KERB")
    pri = [c for c in units if c["call"] != AT_KERB]
    rep.h2(f"Needs an operator ({len(pri)})")
    rep.html('<div class="q">' + "".join(card_html(c) for c in pri) + "</div>" if pri else "<p>Nothing needs an operator.</p>")
    low = [c for c in units if c["call"] == AT_KERB]
    if low:
        rep.html(f"<details><summary>AT KERB: auto-lowered ({len(low)})</summary><div class=\"q\">"
                 + "".join(card_html(c) for c in low) + "</div></details>")
    if summary.get("merged"):
        rep.text(f"{len(summary['merged'])} repeat alerts of the same parked cars (seen again in another PTZ preset) were merged into the cards above.")
    if summary.get("accuracy"):
        a = summary["accuracy"]
        rep.h2("Eyeball check").text(f"{a['right']}/{a['judged']} calls match the labels I wrote from viewing frames "
                                     f"({a['acc']:.0%}); by true label: {a['by_truth']}.")
    return rep.save()


def process(video: str, S: float, k_rel: float, use_vlm: bool, use_clips: bool) -> dict:
    out = OUT / video
    out.mkdir(parents=True, exist_ok=True)
    for stale in list(out.glob("eyeball_*.jpg")) + list(out.glob("numbered_*.jpg")) + list(out.glob("callview_*.jpg")) + list(out.glob("merged_*.jpg")):
        stale.unlink()
    t_all = time.perf_counter()
    ctx = build_cards_slides(video, out) if video in SLIDE_VIDEOS else build_cards_stream(video, S, k_rel, out)
    units, alerts = finish_cards(video, ctx, out, use_vlm, use_clips)
    tr = ctx["tr"]
    ms = [c["card_ms"] for c in units]
    s = tr.summary() if (tr.rows["tid"] >= 0).any() else None
    if s is not None:
        n_tracks = int(len(s[s["name"].isin(lc.VEH)]))
    else:   # untracked detections (injected clip): count distinct vehicle-class boxes per frame as a rough scale
        n_tracks = int(tr.rows[tr.rows["name"].isin(lc.VEH)].groupby("f").size().max())
    kinds = (IN_LANE, CANT, QUEUE, AT_KERB)
    by_alert = {k: sum(1 for c in alerts if c["call"] == k) for k in (IN_LANE, CANT, AT_KERB)}
    by_card = {k: sum(1 for c in units if c["call"] == k) for k in kinds}
    summary = {"video": video, "mode": ctx["mode"], "duration_s": round(tr.duration, 1), "S": S, "k_rel": k_rel,
               "n_vehicle_tracks": n_tracks, "n_alerts": len(alerts), "by_call": by_alert,
               "n_cards": len(units), "by_card": by_card,
               "n_queue_collapsed": sum(c["n_members"] for c in units if c["call"] == QUEUE),
               "merged": [{"dropped": {"cam": m["cam"], "t_alert": round(m["t_alert"], 1), "call": m["call"]},
                           "kept": {"eid": kept_of(m)["eid"], "cam": kept_of(m)["cam"], "t_alert": round(kept_of(m)["t_alert"], 1)}}
                          for m in ctx.get("merged", [])],
               "footprint_src": {k: sum(1 for c in alerts if c.get("foot_src") == k) for k in ("seg-mask", "box-edge")},
               "card_ms_median": float(np.median(ms)) if ms else 0.0, "card_ms_p95": float(np.percentile(ms, 95)) if ms else 0.0,
               "card_ms_max": float(max(ms)) if ms else 0.0,
               "seg_ms_median": float(np.median([c.get("seg_ms", 0.0) for c in alerts])) if alerts else 0.0,
               "seek_ms_median": float(np.median([c["seek_ms"] for c in alerts])) if alerts else 0.0,
               "total_incl_seek_ms_max": float(max(c["seek_ms"] + c["card_ms"] for c in units)) if units else 0.0,
               "detector_s": round(ctx["det_s"], 2),
               "vlm_calls": sum(1 for c in units if c.get("vlm")),
               "vlm_agree": sum(1 for c in units if c.get("vlm") and c["vlm"].get("call") == c["call"]),
               "vlm_usable": sum(1 for c in units if c.get("vlm") and c["vlm"].get("call"))}
    if ctx["mode"] == "slides":
        summary["n_slides"] = ctx["n_slides"]
        nat = lc.stopped_alerts(tr.rows, tr.fps, S=S, k_rel=k_rel)
        summary["native_alerts_at_S"] = len(nat)
        summary["slide_len_median_s"] = round(float(np.median([x["t1"] - x["t0"] for x in ctx["slides"]])), 2)
    if ctx.get("inject"):
        base = pk.tracks(BASE_OF[video])
        summary["control_clean_alerts"] = len(lc.stopped_alerts(base.rows, base.fps, S=S, k_rel=k_rel))   # same clip, nothing pasted
        summary["simulated_injection"] = [{"id": sp["id"], "desc": sp["desc"], "expect_by_construction": sp["expect"], "box": b}
                                          for sp, b in zip(ctx["inject"]["spec"], ctx["inject"]["boxes"])]
    # the queue as VastDB-style SQL (duckdb stand-in; dialect kept to the safe subset: no IN / GROUP BY / JOIN / window)
    import pandas as pd
    df = pd.DataFrame([{"id": c["id"], "sev": L.SEV[c["call"]], "conf": c["conf"], "dur": c["dur"], "t_alert": c["t_alert"],
                        "x1": c["box"][0], "call": c["call"]} for c in units],
                      columns=["id", "sev", "conf", "dur", "t_alert", "x1", "call"])
    con = pk.vastdb({"alerts": df})
    q = [r[0] for r in con.execute("SELECT id FROM alerts ORDER BY sev, conf DESC, dur DESC, t_alert, x1").fetchall()]
    summary["sql_queue_matches_python"] = q == [c["id"] for c in units]
    summary["sql_counts"] = {k: int(con.execute("SELECT count(*) FROM alerts WHERE call = ?", [k]).fetchone()[0]) for k in kinds}
    sheets = (eyeball_sheets(video, alerts, out) + numbered_frames(video, alerts, out)
              + callview_sheets(video, alerts, out) + merge_sheet(ctx, out)) if alerts else []
    summary["accuracy"] = accuracy(video, alerts)
    hero(video, units, alerts, ctx, summary, out)
    write_report(video, units, alerts, ctx, summary, out, sheets)
    summary["wall_s"] = round(time.perf_counter() - t_all, 1)
    keep = ("id", "eid", "t_stop", "t_alert", "dur", "cls", "box", "call", "conf", "reason", "lane_frac", "kerb_frac", "zone", "cam",
            "reg_inliers", "vis", "card_ms", "seg_ms", "seek_ms", "vlm", "clip_path", "slide_kind", "foot_src", "queue_id", "seen_in",
            "n_members", "members")

    def rec(c):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k in keep}
    summary["cards"] = [rec(c) for c in units]
    summary["alerts"] = [rec(c) for c in sorted(alerts, key=lambda c: c["eid"])]
    return summary


def aggregate(videos: dict) -> dict:
    allc = [c for r in videos.values() for c in r["cards"]]
    ms = [c["card_ms"] for c in allc]
    o = {"videos": len(videos), "alerts": sum(r["n_alerts"] for r in videos.values()), "cards": len(allc),
         "card_ms_median": float(np.median(ms)) if ms else None, "card_ms_p95": float(np.percentile(ms, 95)) if ms else None,
         "card_ms_max": float(max(ms)) if ms else None, "under_2s": bool(ms and max(ms) < 2000)}
    judged = [(r["accuracy"]["right"], r["accuracy"]["judged"]) for r in videos.values() if r.get("accuracy")]
    if judged:
        o["eyeball_right"] = sum(j[0] for j in judged)
        o["eyeball_judged"] = sum(j[1] for j in judged)
        o["eyeball_acc"] = round(o["eyeball_right"] / o["eyeball_judged"], 3)
    rows = [(vn, r) for vn, v in videos.items() if v.get("accuracy") for r in v["accuracy"]["rows"]]
    if rows:
        pred_in = [r for _, r in rows if r["call"] == IN_LANE]
        true_in = [r for _, r in rows if r["truth"] == IN_LANE]
        o["in_lane_precision"] = f"{sum(r['truth'] == IN_LANE for r in pred_in)}/{len(pred_in)}"
        o["in_lane_recall"] = f"{sum(r['call'] == IN_LANE for r in true_in)}/{len(true_in)}"
        o["harmful_misses"] = sum(1 for _, r in rows if r["truth"] == IN_LANE and r["call"] == AT_KERB)
        o["false_in_lane"] = sum(1 for _, r in rows if r["truth"] == AT_KERB and r["call"] == IN_LANE)
        o["abstained_when_judgeable"] = sum(1 for _, r in rows if r["truth"] in (IN_LANE, AT_KERB) and r["call"] == CANT)
        o["baseline_always_at_kerb"] = round(sum(r["truth"] == AT_KERB for _, r in rows) / len(rows), 3)
        o["baseline_always_in_lane"] = round(sum(r["truth"] == IN_LANE for _, r in rows) / len(rows), 3)
        ind = [r for vn, r in rows if vn not in ("lisbon_parking",)]
        o["eyeball_independent_geometry"] = (f"{sum(r['right'] for r in ind)}/{len(ind)} (everything except Lisbon: geometry drawn on a "
                                             f"reference frame, alerts judged on other frames or injected by construction)")
        by_vid = {}
        for vn, r in rows:
            p = by_vid.setdefault(vn, [0, 0]); p[1] += 1; p[0] += int(r["right"])
        o["eyeball_by_video"] = {k: f"{v[0]}/{v[1]}" for k, v in by_vid.items()}
        o["in_lane_truth_by_video"] = {vn: sum(1 for v2, r in rows if v2 == vn and r["truth"] == IN_LANE) for vn in by_vid}
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description="LANE CALL: stopped-vehicle alert triage")
    ap.add_argument("--all", action="store_true", help="default test set: " + ", ".join(DEFAULT_SET))
    ap.add_argument("--bonus", action="store_true", help="also run bonus clips: " + ", ".join(BONUS_SET))
    ap.add_argument("--video", action="append", help="catalogue name or %s (repeatable)" % INJECT)
    ap.add_argument("--S", type=float, default=5.0, help="seconds a vehicle must hold still before an alert")
    ap.add_argument("--k", type=float, default=0.12, help="stillness tolerance as a fraction of the box diagonal")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-clips", action="store_true")
    a = ap.parse_args()
    vids = (DEFAULT_SET if a.all else []) + (BONUS_SET if a.bonus else []) + (a.video or [])
    if not vids:
        ap.error("give --all or --video NAME")
    rp = HERE / "results.json"
    old = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    merged = dict(old.get("videos", {}))          # a partial run updates its videos and keeps the rest
    ran = []
    for v in dict.fromkeys(vids):
        t0 = time.time()
        merged[v] = process(v, a.S, a.k, not a.no_vlm, not a.no_clips)
        ran.append(v)
        r = merged[v]
        print(f"[{v}] {r['n_alerts']} alerts -> {r['n_cards']} cards {r['by_card']} card median {r['card_ms_median']:.0f} ms, "
              f"wall {time.time() - t0:.0f}s" + (f", acc {r['accuracy']['right']}/{r['accuracy']['judged']}" if r.get("accuracy") else ""),
              flush=True)
    res = {"set": list(merged), "last_run": ran, "videos": merged, "overall": aggregate(merged)}
    pk.write_results(rp, res)
    print("wrote", rp)


if __name__ == "__main__":
    main()
