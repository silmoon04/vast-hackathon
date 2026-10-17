"""LANE CALL: triage engine. Stopped-vehicle alert -> lane geometry call (IN LANE / AT KERB / CAN'T SEE) + card.

Pieces: zone registry (per-camera polygons drawn on a reference frame, transferred to the live frame by ORB+RANSAC
homography so PTZ presets / small camera shifts still map), call logic, card builder, Cosmos second opinion.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import protokit as pk  # noqa: E402

IN_LANE, AT_KERB, CANT, QUEUE = "IN LANE", "AT KERB", "CAN'T SEE", "QUEUE"
CALL_COL = {IN_LANE: (40, 40, 235), AT_KERB: (60, 170, 60), CANT: (0, 170, 235), QUEUE: (200, 110, 140)}  # BGR
CALL_HEX = {IN_LANE: "#e5322d", AT_KERB: "#2e9e5b", CANT: "#e0a100", QUEUE: "#8c6ec8"}
# queue order: lone IN LANE, then CAN'T SEE (might hide a blocker), then a QUEUE card (stacked, appeared together:
# normal stop-go or a signal, low severity), then AT KERB (auto-lowered)
SEV = {IN_LANE: 0, CANT: 1, QUEUE: 2, AT_KERB: 3}

LANE_MIN = 0.30       # >= 30 % of the vehicle footprint on running-lane polygons -> obstructs the lane
LANE_SURE = 0.70
MIN_INLIERS = 25


# ------------------------------------------------------------------------------------------------ zone registry
class Cam:
    """One camera / preset: reference frame + polygons (normalised coords) + ORB features for registration."""

    def __init__(self, d: dict[str, Any], ref_img: np.ndarray):
        self.id, self.video, self.desc = d["id"], d["video"], d.get("desc", d["id"])
        self.ref_t = d.get("ref_t")
        self.h, self.w = ref_img.shape[:2]
        self.ref = ref_img
        self.zones = [{"name": z["name"], "kind": z["kind"],
                       "px": (np.array(z["pts"], float) * [self.w, self.h]).astype(np.float32)} for z in d["zones"]]
        self.kp, self.des, self.scale = _orb(ref_img)

    def zone_at(self, pt: tuple[float, float]) -> tuple[str | None, str | None]:
        """(kind, name) of the zone containing a ref-frame pixel; lanes win over kerb zones."""
        hit = None
        for z in self.zones:
            if cv2.pointPolygonTest(z["px"], (float(pt[0]), float(pt[1])), False) >= 0:
                if z["kind"] == "lane":
                    return "lane", z["name"]
                hit = hit or ("kerb", z["name"])
        return hit if hit else (None, None)


def _orb(img: np.ndarray):
    s = min(1.0, 960.0 / img.shape[1])
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(2500, fastThreshold=8)
    kp, des = orb.detectAndCompute(g, None)
    return kp, des, s


def register(frame: np.ndarray, cam: Cam, feats=None) -> dict[str, Any] | None:
    """Homography ref -> frame (full-resolution pixels), or None when the view does not match this preset."""
    kp2, des2, s2 = feats if feats is not None else _orb(frame)
    if cam.des is None or des2 is None or len(kp2) < 40 or len(cam.kp) < 40:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(cam.des, des2, k=2)
    good = [a for a, b in (p for p in m if len(p) == 2) if a.distance < 0.78 * b.distance]
    if len(good) < MIN_INLIERS:
        return None
    src = np.float32([cam.kp[g.queryIdx].pt for g in good]) / cam.scale
    dst = np.float32([kp2[g.trainIdx].pt for g in good]) / s2
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None
    inl = int(mask.sum())
    corners = np.float32([[0, 0], [cam.w, 0], [cam.w, cam.h], [0, cam.h]]).reshape(-1, 1, 2)
    wc = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    area = cv2.contourArea(wc.astype(np.float32)) / float(cam.w * cam.h)
    shift = float(np.mean(np.hypot(*(wc - corners.reshape(-1, 2)).T)))
    if inl < MIN_INLIERS or inl / len(good) < 0.35 or not (0.55 < area < 1.8):
        return None
    return {"H": H, "inliers": inl, "ratio": inl / len(good), "area": float(area), "shift_px": shift}


class Registrar:
    """Matches an alert frame to the camera/preset registry. Caches per frame (many alerts share one frame) and tries
    the previously matched preset first (a PTZ stays on a preset for minutes)."""

    def __init__(self, cams: list[Cam]):
        self.cams, self.cache, self.last = cams, {}, None

    def lookup(self, fid: int, frame: np.ndarray) -> tuple[Cam | None, dict[str, Any] | None]:
        if fid in self.cache:
            return self.cache[fid]
        feats = _orb(frame)
        best, bi, bs = None, None, -1.0
        for c in self.cams:
            r = register(frame, c, feats)
            if not r:
                continue
            # prefer the preset the camera is actually AT (small residual shift), not just one that shares scenery
            sc = r["inliers"] * float(np.exp(-r["shift_px"] / (0.12 * c.w)))
            if sc > bs:
                best, bi, bs = c, r, sc
        if best is not None and bi["shift_px"] > 0.30 * best.w:
            best, bi = None, None  # camera is far from every drawn preset: geometry would not transfer
        self.cache[fid] = (best, bi)
        return best, bi


def load_cams(video: str, ref_frames: dict[str, np.ndarray] | None = None) -> list[Cam]:
    data = json.loads((HERE / "zones.json").read_text(encoding="utf-8"))
    out = []
    for d in data["cameras"]:
        if d["video"] != video:
            continue
        img = (ref_frames or {}).get(d["id"])
        if img is None:
            img = pk.frame_at(video, d["ref_t"])
        out.append(Cam(d, img))
    return out


# ---------------------------------------------------------------------------------------------------- the call
def footprint(box, n: int = 7, prof=None) -> np.ndarray:
    """Ground-contact points of the vehicle in frame pixels.
    prof (from seg_profiles): the lowest mask pixel in each column of the YOLO-seg silhouette, i.e. the real ground-contact
    line, which is diagonal for an angled car. Without it: the bottom edge of the axis-aligned box (the old rule)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if prof is not None and len(prof) >= 4:
        P = np.array(prof, float)
        P = P[np.argsort(P[:, 0])]
        k = 5                                     # tyres touch the ground: the underbody between the wheels is NOT contact, so
        ypad = np.pad(P[:, 1], k // 2, mode="edge")  # take the lowest silhouette pixel in a small window (running max)
        P = np.stack([P[:, 0], np.max(np.lib.stride_tricks.sliding_window_view(ypad, k), axis=1)], 1)
        lo, hi = P[0, 0] + 0.06 * (P[-1, 0] - P[0, 0]), P[-1, 0] - 0.06 * (P[-1, 0] - P[0, 0])
        xs = np.linspace(lo, hi, n)
        ys = np.interp(xs, P[:, 0], P[:, 1]) - 0.02 * bh
        return np.stack([xs, ys], 1)
    xs = np.linspace(x1 + 0.08 * bw, x2 - 0.08 * bw, n)
    return np.stack([xs, np.full(n, y2 - 0.05 * bh)], 1)


# ---------------------------------------------------------------------------- segmentation ground-contact profile
VEH_SEG = ("car", "truck", "bus", "motorcycle")


def _profile(mask: np.ndarray, ncols: int = 33):
    """Bottom boundary of a bool mask: [(x, lowest y)] for ncols columns across the mask's x extent."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    xa, xb = int(xs.min()), int(xs.max())
    if xb - xa < 6:
        return None
    pts = []
    for x in np.linspace(xa, xb, ncols):
        xi = int(round(x))
        col = np.nonzero(mask[:, max(0, xi - 1):xi + 2].any(axis=1))[0]
        if len(col):
            pts.append([float(xi), float(col.max())])
    return pts if len(pts) >= 6 else None


def seg_profiles(fids, get_frame, cache_file: Path | None = None) -> dict[int, dict[str, Any]]:
    """YOLO26s-seg on each distinct alert frame (one model load, batched, GPU-locked). Per frame:
    {"ms": inference ms for that frame (excludes model load and GPU-lock wait), "dets": [{name, conf, box, prof}]}.
    Cached on disk (key = frame id) so reruns do not need the GPU. Workaround for protokit.seg_masks_many, which gives
    no per-frame timing (uses protokit's gpu_lock and model loader)."""
    cache: dict[str, Any] = {}
    if cache_file is not None and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    fids = list(dict.fromkeys(fids))
    need = [f for f in fids if str(f) not in cache]
    if need:
        with pk.gpu_lock():
            m = pk._yolo("seg")
            m.predict(get_frame(need[0]), conf=0.25, imgsz=640, verbose=False, retina_masks=True)   # warm-up, not timed
            for fid in need:
                fr = get_frame(fid)
                t0 = time.perf_counter()
                res = m.predict(fr, conf=0.25, imgsz=640, verbose=False, retina_masks=True)[0]
                dets = []
                if res.masks is not None:
                    masks = res.masks.data.cpu().numpy() > 0.5
                    for k, (c, cf, xy) in enumerate(zip(res.boxes.cls.int().tolist(), res.boxes.conf.tolist(),
                                                        res.boxes.xyxy.tolist())):
                        if m.names[c] not in VEH_SEG:
                            continue
                        mk = masks[k]
                        if mk.shape != fr.shape[:2]:
                            mk = cv2.resize(mk.astype(np.uint8), (fr.shape[1], fr.shape[0])) > 0
                        dets.append({"name": m.names[c], "conf": round(cf, 3), "box": [round(v, 1) for v in xy],
                                     "prof": _profile(mk)})
                cache[str(fid)] = {"ms": round((time.perf_counter() - t0) * 1000, 1), "dets": dets}
            pk._unload()
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
    return {f: cache[str(f)] for f in fids}


def match_profile(sdets: list[dict[str, Any]], box, min_iou: float = 0.45):
    """Seg silhouette profile of the vehicle whose seg box best overlaps the alert box (None: use the box edge)."""
    best, bi = None, min_iou
    b = np.array([box], float)
    for d in sdets:
        if not d.get("prof"):
            continue
        i = _iou(b, np.array([d["box"]], float))
        if i > bi:
            best, bi = d, i
    return best["prof"] if best else None


def _iou(A, B) -> float:
    ix = max(0.0, min(A[0, 2], B[0, 2]) - max(A[0, 0], B[0, 0])); iy = max(0.0, min(A[0, 3], B[0, 3]) - max(A[0, 1], B[0, 1]))
    inter = ix * iy
    ua = (A[0, 2] - A[0, 0]) * (A[0, 3] - A[0, 1]) + (B[0, 2] - B[0, 0]) * (B[0, 3] - B[0, 1]) - inter
    return inter / max(ua, 1e-9)


def visibility_issues(frame: np.ndarray, box, others: list[list[float]], det_conf: float) -> list[str]:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    out = []
    if y2 >= H - 3:
        out.append("ground contact cut off by the frame bottom")
    elif (x1 <= 3 or x2 >= W - 3) and bw < 0.07 * W:
        out.append("only a sliver of the vehicle is in frame")
    if bw < 24 or bh < 18:
        out.append("too small to judge")
    crop = frame[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)]
    if crop.size:
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if g.mean() < 25 or (np.percentile(g, 95) - np.percentile(g, 5)) < 22:
            out.append("too dark / blurry / low contrast")
    area = max(1.0, bw * bh)
    for o in others:
        ix = max(0.0, min(x2, o[2]) - max(x1, o[0])); iy = max(0.0, min(y2, o[3]) - max(y1, o[1]))
        if ix * iy / area > 0.35 and o[3] > y2 + 2:  # a nearer vehicle covers a big part of this one
            out.append("occluded by a nearer vehicle")
            break
    return out


def make_call(frame: np.ndarray, box, cam: Cam | None, reg: dict[str, Any] | None, others: list[list[float]],
              det_conf: float, prof=None) -> dict[str, Any]:
    """The core decision (cam, reg come from Registrar.lookup). Returns call, confidence, reason + the numbers."""
    vis = visibility_issues(frame, box, others, det_conf)
    fp = footprint(box, prof=prof)
    res: dict[str, Any] = {"cam": cam.id if cam else None, "vis": vis, "lane_frac": None, "kerb_frac": None,
                           "reg_inliers": reg["inliers"] if reg else 0, "zone": None,
                           "foot_src": "seg-mask" if prof is not None else "box-edge",
                           "foot_px": [[round(float(x), 1), round(float(y), 1)] for x, y in fp]}
    if cam is None or reg is None:
        res.update(call=CANT, conf=0.9, reason="view does not match any camera with lane geometry (camera moved / no preset)")
        return res
    Hinv = np.linalg.inv(reg["H"])
    pts = cv2.perspectiveTransform(fp.reshape(-1, 1, 2).astype(np.float32), Hinv).reshape(-1, 2)
    kinds = [cam.zone_at(p) for p in pts]
    lane = float(np.mean([k == "lane" for k, _ in kinds]))
    kerb = float(np.mean([k == "kerb" for k, _ in kinds]))
    lane_names = [n for k, n in kinds if k == "lane"]
    kerb_names = [n for k, n in kinds if k == "kerb"]
    names = lane_names if lane >= LANE_MIN else (kerb_names or lane_names)
    res.update(lane_frac=round(lane, 2), kerb_frac=round(kerb, 2), zone=max(set(names), key=names.count) if names else None)
    qual = min(1.0, reg["inliers"] / 80.0) ** 0.5
    vf = (0.75 if vis else 1.0) * (0.85 if det_conf < 0.35 else 1.0)  # a weak detector box lowers confidence, it does not abstain
    if lane >= LANE_MIN and (not vis or lane >= 0.5):
        part = lane < LANE_SURE
        # confidence scales with how much of the ground-contact line is on the lane (100 % -> 0.97, 70 % -> 0.80, 30 % -> 0.54)
        conf = (0.35 + 0.62 * lane) * vf * (0.85 + 0.15 * qual)
        res.update(call=IN_LANE, conf=round(min(conf, 0.97), 2),
                   reason=(f"{lane:.0%} of the ground-contact line is on '{res['zone']}'"
                           + (" (partly in lane: still blocks it)" if part else "")
                           + (f"; caution: {', '.join(vis)}" if vis else "")))
    elif vis:
        res.update(call=CANT, conf=round(0.85 - 0.1 * (len(vis) - 1), 2), reason=", ".join(vis) + " (footprint not reliable)")
    elif kerb >= 0.5:
        conf = (0.45 + 0.30 * kerb + 0.25 * min(1.0, (LANE_MIN - lane) / LANE_MIN)) * (0.8 + 0.2 * qual) * min(1.0, 0.6 + det_conf)
        res.update(call=AT_KERB, conf=round(min(conf, 0.98), 2),
                   reason=f"ground-contact line {kerb:.0%} inside '{res['zone']}', {lane:.0%} on the running lane")
    else:
        res.update(call=CANT, conf=0.7, reason="footprint falls outside the mapped zones")
    return res


# ------------------------------------------------------------------------------------------ drawing / cards
def _font(sz: int, bold: bool = True):
    from PIL import ImageFont
    for f in (["arialbd.ttf", "segoeuib.ttf"] if bold else ["arial.ttf", "segoeui.ttf"]):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            continue
    return ImageFont.load_default()


def text_pil(img: np.ndarray, items: list[tuple], ) -> np.ndarray:
    """items: (x, y, text, size, rgb, bold). Draw antialiased text onto a BGR image."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    for x, y, t, sz, col, bold in items:
        d.text((x, y), t, font=_font(sz, bold), fill=tuple(col))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def stamp(img: np.ndarray, call: str, conf: float, xy: tuple[int, int], size: int = 34) -> np.ndarray:
    from PIL import Image, ImageDraw
    txt = f"{call}  {conf:.0%}"
    f = _font(size)
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((0, 0), txt, font=f)
    x, y = xy
    col = tuple(int(v) for v in CALL_COL[call][::-1])
    d.rounded_rectangle((x, y, x + (r - l) + 28, y + (b - t) + 20), 8, fill=col)
    d.text((x + 14 - l, y + 8 - t), txt, font=f, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def draw_zones(img: np.ndarray, cam: Cam, H: np.ndarray | None) -> np.ndarray:
    """Overlay the camera's lane (red) and kerb/bay (blue, faint) polygons, transferred by homography H ref->frame."""
    polys = []
    for z in cam.zones:
        p = z["px"].reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(p, H) if H is not None else p
        polys.append((z["kind"], p.reshape(-1, 2).astype(np.int32)))
    kerb, lane = img.copy(), img.copy()
    for k, p in polys:
        cv2.fillPoly(kerb if k == "kerb" else lane, [p], (200, 140, 40) if k == "kerb" else (40, 40, 230))
    out = cv2.addWeighted(kerb, 0.14, img, 0.86, 0)
    out = cv2.addWeighted(lane, 0.38, out, 0.62, 0) if any(k == "lane" for k, _ in polys) else out
    # lane overlay must only tint lane polygons: rebuild from masks
    mask = np.zeros(img.shape[:2], np.uint8)
    for k, p in polys:
        if k == "lane":
            cv2.fillPoly(mask, [p], 255)
    base = cv2.addWeighted(kerb, 0.14, img, 0.86, 0)
    tinted = cv2.addWeighted(lane, 0.38, base, 0.62, 0)
    out = np.where(mask[..., None] > 0, tinted, base)
    for k, p in polys:
        cv2.polylines(out, [p], True, (40, 40, 230) if k == "lane" else (200, 140, 40), 2, cv2.LINE_AA)
    return out


def crop_box(frame: np.ndarray, box, pad: float = 0.35, min_side: int = 160) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = max(pad * bw, 0), max(pad * bh, 0)
    cw, ch = max(bw + 2 * px, min_side), max(bh + 2 * py, min_side * 0.75)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
    c, d = int(min(W, cx + cw / 2)), int(min(H, cy + ch / 2))
    return frame[b:d, a:c].copy()


VLM_PROMPT = ("This is a traffic camera frame. One vehicle is outlined with a red box and has been stationary for several "
              "seconds. Where is that boxed vehicle? Answer A if it is stopped on the carriageway / in a running traffic lane "
              "(blocking moving traffic), B if it is at the kerb: parked in a bay or beside the road, on the pavement, "
              "verge or hard shoulder, C if you cannot tell (hidden, dark, cut off). "
              'Reply as JSON: {"answer":"A|B|C","reason":"max 14 words"}')
VLM_MAP = {"A": IN_LANE, "B": AT_KERB, "C": CANT}


def cosmos_second_opinion(frame: np.ndarray, box) -> dict[str, Any]:
    """Local Cosmos3-Edge standing in for Cosmos3-Reason. Shown on the card, never decisive."""
    img = frame.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    t0 = time.perf_counter()
    j = pk.vlm_json(VLM_PROMPT, [img], max_tokens=80)
    dt = time.perf_counter() - t0
    if not isinstance(j, dict) or str(j.get("answer", "")).strip().upper()[:1] not in VLM_MAP:
        return {"call": None, "reason": "no usable answer", "seconds": round(dt, 1)}
    a = str(j["answer"]).strip().upper()[:1]
    return {"call": VLM_MAP[a], "reason": str(j.get("reason", ""))[:140], "seconds": round(dt, 1)}


def severity_key(c: dict[str, Any]) -> tuple:
    """Queue order: IN LANE, CAN'T SEE, QUEUE (collapsed stop-go), AT KERB; confident + long-dwell first."""
    return (SEV[c["call"]], -c["conf"], -c["dur"], c["t_alert"], c["box"][0])
