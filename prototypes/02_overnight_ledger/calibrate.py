"""Calibration of the before/after crop comparison on CLEAN ptz_parking footage (no simulation of the video involved).
 same_view  : same car, same camera view, 45-75 s apart                          (should read 'same car')
 across_tour: same car, before the PTZ tour vs after it returned (new light/angle) (should read 'same car')
 different  : crop of a different parked car placed in the same box             (should read 'CHANGED')
 paint_*    : SIMULATED scratch+dent painted onto the 'after' crop of the same_view / across_tour pairs
One car (tid 15) really left during the tour (seen by eye) and is excluded from across_tour.
Writes calibration.json with percentile tables and hit/false-alarm rates at the shipped thresholds."""
import json
from pathlib import Path
import numpy as np, cv2
import ledger as L, protokit as pk, sim

def run(out=L.HERE / "calibration.json"):
    name = "ptz_parking"; cd = L.HERE / "out" / "_cache"
    path = pk.video_path(name); tr = pk.tracks(name); F = L.features(path, cd)
    cands = L.candidate_spots(tr, L.split_runs(F))
    al = L.align_video(path, F, cands[0].t_ref, cache_dir=cd)
    img = {k: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for k, b in al.jpgs.items()}
    usable = [c for c in cands if len(L.visible_seconds(al, c.box)) >= 200 and c.stay >= 80 and 15 not in c.tids]
    rows = {k: [] for k in ("same_view", "across_tour", "different_car", "paint_same_view", "paint_across_tour")}
    for c in usable:
        m = L.margin_for_box(c.box)
        crop = lambda k, box=c.box: L._spot_crop(img[k], box, m)
        w, h = c.box[2] - c.box[0], c.box[3] - c.box[1]
        for a, b in ((70, 130), (80, 140), (100, 145)):
            if a not in img or b not in img:
                continue
            A, B = crop(a), crop(b)
            rows["same_view"].append(L.compare(A, B, m))
            B2 = B.copy(); sim.paint_damage(B2, (m, m, m + w, m + h)); rows["paint_same_view"].append(L.compare(A, B2, m))
        for a in (100, 125):
            for b in (255, 272, 300):
                if a not in img or b not in img:
                    continue
                A, B = crop(a), crop(b)
                rows["across_tour"].append(L.compare(A, B, m))
                B2 = B.copy(); sim.paint_damage(B2, (m, m, m + w, m + h)); rows["paint_across_tour"].append(L.compare(A, B2, m))
        for d in usable:
            if d is c:
                continue
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            box2 = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if box2[0] < 0 or box2[1] < 0 or box2[2] > al.W or box2[3] > al.H or L.iou(box2, c.box) > 0.2:
                continue
            rows["different_car"].append(L.compare(crop(100), L._spot_crop(img[110], box2, m), m))
    run.rows = rows
    res = {"cars_used": len(usable)}
    for nm, rr in rows.items():
        res[f"{nm}.n"] = len(rr)
        for key in ("ssim", "block_min", "hue"):
            res[f"{nm}.{key}.p5_50_95"] = [round(float(np.percentile([r[key] for r in rr], p)), 3) for p in (5, 50, 95)]
        res[f"{nm}.flagged_CHANGED_pct"] = round(100 * float(np.mean([L.looks_changed(r) for r in rr])), 1)
    out.write_text(json.dumps({"thresholds": {k: L.P[k] for k in ("cmp_ssim", "cmp_block", "cmp_hue", "app_present", "app_diff")},
                               "results": res}, indent=1))
    return res

if __name__ == "__main__":
    for k, v in run().items():
        print(k, v)
