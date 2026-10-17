"""Stage 2: frame-sequence alignment of a query clip against archive dHash sequences.

Estimates per candidate (video, variant): offset + speed (Hough vote over (speed, offset) then robust LS), jumps between
segments (removed source seconds), unmatched query spans (inserted foreign footage), mirror/crop (best variant)."""
from __future__ import annotations

import numpy as np

from uncut import HFPS, NB, VARIANTS, vname

DT = 1.0 / HFPS
TAU = 0.18           # per-frame normalised Hamming distance still counted as "same frame" (true-match noise 0.00-0.05)
MIN_FOREIGN = 1.0    # seconds of unmatched query before we call it inserted foreign footage
MARGIN = 0.035       # a line may only claim a frame whose distance is within this of the frame's best match anywhere
MIN_JUMP = 0.6       # seconds of source gap before we call it "removed"
SPEEDS = np.exp(np.linspace(np.log(0.5), np.log(2.0), 70))
OCCAM = [vname(v) for v in sorted(VARIANTS, key=lambda v: (v.kind == 'p916' or v.crop < 1.0, v.mirror, v.kind == 'p916',
                                                          v.ox != 0.5, -v.crop))]   # least-modified explanation first


def ham(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    sq, sa = q.astype(np.float32) * 2 - 1, a.astype(np.float32) * 2 - 1
    return (NB - sq @ sa.T) / 2 / NB


def _pairs(D: np.ndarray, k=5, sep=3):
    """candidate (query frame, archive frame, weight): up to k mutually-separated minima per query row with d <= TAU."""
    out = []
    for i in range(D.shape[0]):
        row = D[i]
        cand = np.argpartition(row, 60)[:60] if len(row) > 60 else np.arange(len(row))
        cand = cand[np.argsort(row[cand])]
        picked = []
        for j in cand:
            if row[j] > TAU:
                break
            if all(abs(j - p) >= sep for p in picked):
                picked.append(int(j))
                out.append((i, int(j), float(np.exp(-(row[j] - row[cand[0]]) / 0.02) * (1.0 - row[j] / TAU))))
                if len(picked) == k:
                    break
    return out


def _hough(pairs, tq, a_grid, bin_w=0.3):
    if not pairs:
        return []
    ii, jj, w = (np.array(x) for x in zip(*pairs))
    x, y = tq[ii], jj / HFPS
    peaks = []
    for a in a_grid:
        c = y - a * x
        lo = c.min()
        nb = int((c.max() - lo) / bin_w) + 3
        h = np.zeros(nb)
        np.add.at(h, ((c - lo) / bin_w).astype(int) + 1, w)
        hs = h + 0.5 * (np.roll(h, 1) + np.roll(h, -1))
        k = int(np.argmax(hs))
        peaks.append((float(hs[k]) * (1.0 - 0.03 * abs(np.log(a)) / np.log(2)), float(a), float(lo + (k - 0.5) * bin_w)))
    peaks.sort(reverse=True)
    return peaks


def _assign(D, tq, a, c, free, tol=0.45, rowmin=None):
    """each free query frame -> best archive frame within +-tol s of the line ts = a*tq + c (or -1)."""
    n = len(tq)
    js, ds = -np.ones(n, int), np.ones(n, np.float32)
    r = int(np.ceil(tol * HFPS))
    rowmin = D.min(1) if rowmin is None else rowmin
    for i in np.where(free)[0]:
        j0 = int(round((a * tq[i] + c) * HFPS))
        lo, hi = max(0, j0 - r), min(D.shape[1], j0 + r + 1)
        if hi > lo:
            k = lo + int(np.argmin(D[i, lo:hi]))
            if D[i, k] <= min(TAU, rowmin[i] + MARGIN):    # must also be near this frame's best match anywhere
                js[i], ds[i] = k, D[i, k]
    return js, ds



def _compete(D, tq, a, cs):
    """each query frame goes to the line (speed a, offsets cs) that explains it best, and only if that is near the
    frame's best match anywhere (static scenes otherwise let a line run on past a cut)."""
    rowmin = D.min(1)
    bestd, bestj, bests = np.ones(len(tq), np.float32), -np.ones(len(tq), int), -np.ones(len(tq), int)
    for si, c in enumerate(cs):
        js, ds = _assign(D, tq, a, c, np.ones(len(tq), bool))
        better = (js >= 0) & (ds < bestd)
        bestd[better], bestj[better], bests[better] = ds[better], js[better], si
    keep = (bests >= 0) & (bestd <= rowmin + MARGIN)
    out = []
    for si, c in enumerate(cs):
        sel = keep & (bests == si)
        out.append({"c": c, "idx": np.where(sel)[0], "j": bestj[sel], "d": bestd[sel]})
    return out


def _score(segs):
    return float(sum((1.0 - s["d"] / TAU).sum() for s in segs))


def align_one(q, tq, A, valid, region):
    """Align one query hash sequence to one archive hash array."""
    D = ham(q, A)
    D[:, ~(valid & region)] = 1.0
    res = {"D": D, "score": 0.0, "segments": [], "a": 1.0, "match": np.zeros(len(tq), bool),
           "d": np.ones(len(tq), np.float32), "j": -np.ones(len(tq), int)}
    pairs = _pairs(D)
    peaks = _hough(pairs, tq, SPEEDS)
    if not peaks or peaks[0][0] < 3.0:
        return res
    _, a, c = peaks[0]
    fit_a = a
    # speed prior: plain 1.0x (or a common speed within 6% of the free fit) wins when its Hough evidence is
    # within 5% of the best free fit; static scenes carry little speed evidence, so this avoids a bogus 1.16x
    snapped = False
    for v in [1.0] + [v for v in (0.5, 0.75, 1.25, 1.5, 2.0) if abs(v / fit_a - 1) <= 0.06]:
        pv = _hough(pairs, tq, [v])
        if pv and pv[0][0] >= 0.95 * peaks[0][0]:
            a, c, snapped = v, pv[0][2], True
            break
    free = np.ones(len(tq), bool)
    segs = []
    for it in range(6):
        if it > 0:                                   # next segment: same speed, new offset, from still-unmatched frames
            sub = [p for p in pairs if free[p[0]]]
            pk = _hough(sub, tq, [a])
            if not pk or pk[0][0] < 3.0:
                break
            c = pk[0][2]
        js, ds = _assign(D, tq, a, c, free)
        ok = js >= 0
        if ok.sum() < 4:
            break
        if it == 0 and not snapped and ok.sum() >= 8 and np.ptp(tq[ok]) > 2.0:        # refine speed + offset on the dominant segment
            x, y = tq[ok], js[ok] / HFPS
            for _ in range(3):
                aa, cc = np.polyfit(x, y, 1)
                if not 0.4 < aa < 2.5:
                    break
                r = np.abs(y - (aa * x + cc))
                keep = r <= max(0.25, 2.0 * np.median(r))
                if keep.sum() < 6:
                    break
                x, y = x[keep], y[keep]
            if 0.4 < aa < 2.5:
                a, c = float(aa), float(cc)
            js, ds = _assign(D, tq, a, c, free)
            ok = js >= 0
        else:
            c = float(np.median(js[ok] / HFPS - a * tq[ok]))
        segs.append({"c": c, "idx": np.where(ok)[0], "j": js[ok], "d": ds[ok]})
        free &= ~ok
        if free.sum() < 4:
            break
    if not segs:
        return res
    # joint refinement of one speed with a free offset per segment (needs >= 3 s of aligned span)
    allx = np.concatenate([tq[s["idx"]] for s in segs])
    if not snapped and np.ptp(allx) > 3.0 and sum(len(s["idx"]) for s in segs) >= 10:
        rows, ys = [], []
        for si, s in enumerate(segs):
            for i, j in zip(s["idx"], s["j"]):
                r = np.zeros(1 + len(segs))
                r[0], r[1 + si] = tq[i], 1.0
                rows.append(r)
                ys.append(j / HFPS)
        sol = np.linalg.lstsq(np.array(rows), np.array(ys), rcond=None)[0]
        if 0.4 < sol[0] < 2.5:
            a = float(sol[0])
            for si, s in enumerate(segs):
                s["c"] = float(sol[1 + si])
            free = np.ones(len(tq), bool)
            for s in segs:
                js, ds = _assign(D, tq, a, s["c"], free)
                ok = js >= 0
                s["idx"], s["j"], s["d"] = np.where(ok)[0], js[ok], ds[ok]
                free &= ~ok
    segs = _compete(D, tq, a, [s["c"] for s in segs])
    res["fit_speed"] = fit_a
    res["snapped"] = snapped
    segs = [s for s in segs if len(s["idx"]) >= 3]
    segs.sort(key=lambda s: tq[s["idx"][0]])
    merged = []                                      # merge neighbours whose offsets agree
    for s in segs:
        if merged and abs(merged[-1]["c"] - s["c"]) < 0.5:
            m = merged[-1]
            m["idx"] = np.concatenate([m["idx"], s["idx"]])
            m["j"] = np.concatenate([m["j"], s["j"]])
            m["d"] = np.concatenate([m["d"], s["d"]])
        else:
            merged.append(dict(s))
    match = np.zeros(len(tq), bool)
    dsel = np.ones(len(tq), np.float32)
    jsel = -np.ones(len(tq), int)
    for s in merged:
        match[s["idx"]] = True
        dsel[s["idx"]], jsel[s["idx"]] = s["d"], s["j"]
    # time-ambiguity: strongest alternative offset (same speed) far from every aligned stretch vs the aligned one
    ii, jj, w = (np.array(x) for x in zip(*pairs))
    cvals = jj / HFPS - a * tq[ii]
    def mass(c0, half=0.3):
        return float(w[np.abs(cvals - c0) <= half].sum())
    main = max(mass(s["c"]) for s in merged)
    far = np.ones(len(cvals), bool)
    for s in merged:
        far &= np.abs(cvals - s["c"]) > 1.5
    alt = 0.0
    if far.any():
        cf, wf = cvals[far], w[far]
        for c0 in cf[np.argsort(-wf)][:60]:
            alt = max(alt, float(wf[np.abs(cf - c0) <= 0.3].sum()))
    res.update(segments=merged, a=a, match=match, d=dsel, j=jsel, score=float((1.0 - dsel[match] / TAU).sum()),
               alt_ratio=alt / max(main, 1e-6))
    return res


def twin_fraction(res, margin=0.04, excl=2.0):
    """share of matched query frames that look equally good at an archive time >= excl s from the aligned one
    (static camera / repeated footage => the frame cannot be located uniquely)."""
    D, m, d, j = res["D"], res["match"], res["d"], res["j"]
    idx = np.where(m)[0]
    if len(idx) == 0:
        return 1.0
    r = int(excl * HFPS)
    tw = 0
    for i in idx:
        row = D[i].copy()
        row[max(0, j[i] - r):j[i] + r + 1] = 1.0
        if row.min() <= d[i] + margin:
            tw += 1
    return tw / len(idx)


def _run_bounds(idx, max_gap=2, min_run=4):
    """first/last frame index of the stretch's solid runs: isolated stray frames that happen to look like this stretch
    (static footage) do not move its boundary. A run = frames with gaps <= max_gap frames, at least min_run long."""
    idx = np.asarray(idx)
    cuts = np.where(np.diff(idx) > max_gap)[0] + 1
    runs = [r for r in np.split(idx, cuts) if len(r) >= min_run] or [idx]
    return int(runs[0][0]), int(runs[-1][-1])


def describe(res, tq, qdur):
    """Turn a raw alignment into the report: segments, removed seconds, inserted foreign spans, coverage."""
    segs = res["segments"]
    a = res["a"]
    out = {"speed": a, "segments": [], "removed": [], "foreign": [], "coverage": 0.0, "foreign_seconds": 0.0}
    if not segs:
        return out
    m = res["match"]
    out["coverage"] = float(m.sum() * DT / qdur)
    for s in segs:
        i0, i1 = _run_bounds(s["idx"])
        out["segments"].append({"q_start": float(tq[i0]), "q_end": float(tq[i1] + DT),
                                "src_start": float(a * tq[i0] + s["c"]), "src_end": float(a * (tq[i1] + DT) + s["c"]),
                                "n": int(len(s["idx"])), "mean_d": float(np.mean(s["d"]))})
    for p, n in zip(out["segments"][:-1], out["segments"][1:]):
        # source seconds skipped = jump of the line's offset (+ clip time spent on foreign frames in between). Using the
        # offsets, not the boundary frames, keeps it exact when static footage lets both stretches claim the same frames.
        c1, c2 = p["src_end"] - a * p["q_end"], n["src_start"] - a * n["q_start"]
        overlap = p["q_end"] - n["q_start"]               # > 0: frames both stretches can explain (static footage)
        skipped = (c2 - c1) + a * max(0.0, -overlap)
        at_q = p["q_end"] if overlap <= 0.2 else (p["q_end"] + n["q_start"]) / 2
        unc = 0.0 if overlap <= 0.2 else overlap / 2
        if overlap > 0.2:                                  # split the ambiguous frames at the midpoint: no overlapping stretches
            p["q_end"], n["q_start"] = at_q, at_q
            p["src_end"], n["src_start"] = a * at_q + c1, a * at_q + c2
        ent = {"at_q": float(at_q), "at_src": float(a * at_q + c1), "seconds": float(skipped),
               "seconds_r": round(skipped * 2) / 2, "pos_uncertain_s": float(unc)}
        if skipped >= MIN_JUMP:
            out["removed"].append(ent)
        elif skipped <= -MIN_JUMP:
            out["removed"].append({**ent, "note": "rewound/repeated"})
    n_ = len(tq)
    i = 0
    while i < n_:
        if m[i]:
            i += 1
            continue
        k = i
        while k < n_ and not m[k]:
            k += 1
        if (k - i) * DT >= MIN_FOREIGN:
            out["foreign"].append({"q_start": float(tq[i]), "q_end": float(tq[k - 1] + DT)})
        i = k
    out["foreign_seconds"] = float(sum(f["q_end"] - f["q_start"] for f in out["foreign"]))
    return out
