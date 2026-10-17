"""Human-labelled ground truth, independent of the auditor's models and of the redactor logs.

    python eyeball.py make     -> eyeball_sample.json (frozen keys) + out/_work/eyeball_recall.jpg + eyeball_precision.jpg
    labels are written by hand into eyeball_gt.json:  {"recall": "YNNY...", "precision": "YYNY..."}  (tile order)

RECALL sample: person boxes (YOLO26s tracks, conf >= 0.25, box height >= 60 px) in frames that redactor A SKIPPED
  (grid frame k % 3 != 0), so they are unblurred by construction. Tile = the head crop of the ORIGINAL frame.
  Label Y = I can see a face (eyes resolvable) that someone could identify; N = back of head / too small / blurred.
  score(): of the Y tiles, how many did the auditor flag in the export (a flagged face centred in that head crop)?
PRECISION sample: faces the auditor FLAGGED in exports A and B (crops from the export itself).
  Label Y = it really is a sharp human face; N = not a face / not sharp. -> auditor precision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import protokit as pk  # noqa: E402
import leaklib as L  # noqa: E402

SAMPLE = HERE / "eyeball_sample.json"
GT = HERE / "eyeball_gt.json"
DEFAULT = ["face_demographics_walking_and_pause", "retail_cctv", "yonge_dundas"]
PER_VIDEO = 24
PREC_PER_EXPORT = 30


def load_audit(video, kind):
    d = json.loads((HERE / "out" / video / "_work" / f"audit_{kind}.json").read_text())
    return d["res"] if isinstance(d, dict) else d


def grid_frames(name, meta):
    m = pk.info(name)
    stride = max(1, int(round(m["fps"] / 15.0)))
    return [fi for fi, t, fr in pk.iter_frames(name, stride=stride, start=meta["t0"], end=meta["t1"])]


def tile_of(frame, rect, size=150, grow=0.0):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = rect
    c = frame[int(max(0, y1)):int(min(H, y2)), int(max(0, x1)):int(min(W, x2))]
    s = size / max(c.shape[:2])
    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    t = np.full((size, size, 3), 30, np.uint8)
    t[:c.shape[0], :c.shape[1]] = c
    return t


def make(seed=7, keep_recall=False):
    """keep_recall=True keeps the frozen recall sample (its labels stay valid: it does not depend on the auditor) and
    re-draws only the precision sample (it depends on what the auditor flags, so it must be re-drawn and re-labelled
    whenever the auditor changes)."""
    rng = np.random.default_rng(seed)
    rec, prec = [], []
    old = json.loads(SAMPLE.read_text()) if (keep_recall and SAMPLE.exists()) else None
    for name in DEFAULT:
        meta = json.loads((HERE / "out" / name / "_work" / "window.json").read_text())
        m = pk.info(name)
        kof = {fi: k for k, fi in enumerate(grid_frames(name, meta))}
        tr = pk.tracks(name).of("person")
        tr = tr[(tr.conf >= 0.25) & (tr.bh >= 60)]
        cand = [(kof[int(f)], int(f), int(tid), [float(x1), float(y1), float(x2), float(y2)])
                for f, tid, x1, y1, x2, y2 in zip(tr.f, tr.tid, tr.x1, tr.y1, tr.x2, tr.y2)
                if int(f) in kof and kof[int(f)] % 3 != 0]
        pick = rng.choice(len(cand), size=min(PER_VIDEO, len(cand)), replace=False)
        for i in sorted(pick):
            k, f, tid, b = cand[i]
            rec.append({"video": name, "k": k, "f": f, "tid": tid, "box": [round(v, 1) for v in b]})
        rng_p = np.random.default_rng(seed + 1)
        for kd in ("a_keyframe", "b_conf"):
            aud = load_audit(name, kd)
            L.tag_in_person(aud, load_audit(name, "original"))
            fl = [(r["k"], f["rect"]) for r in aud for f in r["faces"] if L.is_flagged(f)]
            if not fl:
                continue
            for i in sorted(rng_p.choice(len(fl), size=min(PREC_PER_EXPORT // 3, len(fl)), replace=False)):
                prec.append({"video": name, "export": kd, "k": fl[i][0], "rect": [round(v, 1) for v in fl[i][1]]})
        orig_aud = load_audit(name, "original")
        for kd in ("c_good", "d_tight"):  # every flag on the two good exports is checked
            aud = load_audit(name, kd)
            L.tag_in_person(aud, orig_aud)
            for r in aud:
                for f in r["faces"]:
                    if L.is_flagged(f):
                        prec.append({"video": name, "export": kd, "k": r["k"], "rect": [round(v, 1) for v in f["rect"]]})
    if old is not None:
        rec = old["recall"]
    SAMPLE.write_text(json.dumps({"recall": rec, "precision": prec}, indent=0))
    tiles = []
    for j, it in enumerate(rec):
        m = pk.info(it["video"])
        fr = pk.frame_at(it["video"], it["f"] / m["fps"] + 1e-4)
        t = tile_of(fr, L.head_crop(it["box"], fr.shape[1], fr.shape[0]))
        pk.label(t, f"{j}", (2, 2), scale=0.5, bg=(0, 0, 160))
        tiles.append(t)
    pk.save_jpg(pk.sheet(tiles, cols=12, tile_w=150, pad=3), HERE / "out" / "_work" / "eyeball_recall.jpg", 90)
    tiles = []
    for j, it in enumerate(prec):
        path = HERE / "out" / it["video"] / f"export_{it['export']}.mp4"
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, it["k"])
        ok, fr = cap.read()
        cap.release()
        x1, y1, x2, y2 = it["rect"]
        a = max(x2 - x1, y2 - y1) * 0.7
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        t = tile_of(fr, [cx - a, cy - a, cx + a, cy + a])
        pk.label(t, f"{j}", (2, 2), scale=0.5, bg=(0, 0, 160))
        tiles.append(t)
    pk.save_jpg(pk.sheet(tiles, cols=12, tile_w=150, pad=3), HERE / "out" / "_work" / "eyeball_precision.jpg", 90)
    print(len(rec), "recall tiles,", len(prec), "precision tiles")


def score(audits_by_video):
    """audits_by_video: {video: {kind: audit list}}. Returns recall/precision from my labels (None if no labels)."""
    if not (SAMPLE.exists() and GT.exists()):
        return None
    s = json.loads(SAMPLE.read_text())
    lab = json.loads(GT.read_text())
    out = {}
    rec, rl = s["recall"], lab.get("recall", "")
    if len(rl) == len(rec):
        rows = []
        for it, y in zip(rec, rl):
            aud = audits_by_video.get(it["video"], {}).get("a_keyframe")
            if aud is None or it["k"] >= len(aud):
                continue
            x1, y1, x2, y2 = L.head_crop(it["box"], 10 ** 5, 10 ** 5)
            fl = [f for f in aud[it["k"]]["faces"] if L.is_flagged(f)
                  and x1 <= (f["rect"][0] + f["rect"][2]) / 2 <= x2 and y1 <= (f["rect"][1] + f["rect"][3]) / 2 <= y2]
            rows.append({"video": it["video"], "k": it["k"], "label": y, "flagged": bool(fl)})
        yy = [r for r in rows if r["label"] == "Y"]
        out["recall"] = {"tiles": len(rows), "labelled_face": len(yy), "caught": sum(r["flagged"] for r in yy),
                         "recall": round(sum(r["flagged"] for r in yy) / len(yy), 4) if yy else None,
                         "labelled_noface": len(rows) - len(yy),
                         "flagged_though_noface": sum(r["flagged"] for r in rows if r["label"] == "N"),
                         "by_video": {v: {"face": sum(1 for r in yy if r["video"] == v),
                                          "caught": sum(1 for r in yy if r["video"] == v and r["flagged"])} for v in DEFAULT}}
    prec, pl = s["precision"], lab.get("precision", "")
    if len(pl) == len(prec):
        out["precision"] = {"tiles": len(prec), "real_faces": pl.count("Y"),
                            "precision": round(pl.count("Y") / len(prec), 4) if prec else None,
                            "by_export": {kd: {"n": sum(1 for it in prec if it["export"] == kd),
                                               "real": sum(1 for it, y in zip(prec, pl) if it["export"] == kd and y == "Y")}
                                          for kd in ("a_keyframe", "b_conf", "c_good", "d_tight")}}
    return out or None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make":
        make(keep_recall="--keep-recall" in sys.argv)
