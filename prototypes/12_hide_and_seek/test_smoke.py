"""Fast invariants on the produced outputs (< 60 s). Run after `python run.py --all`:  python test_smoke.py"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hs_core as hc  # noqa: E402

OUT = HERE / "out"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
CLIPS = [{"id": c["id"], "video": c["video"], "attrs": c["attrs"], "extra": c["extra"]} for c in R["clips"]]
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        FAILS.append(name)


def hard_with_flip(eng: "hc.Engine", n: int, pos: int) -> float:
    """Strict engine (no confirmation, no recovery) with the answer at question position `pos` flipped; fraction ending on the secret."""
    ok = 0
    for secret in range(n):
        cand, asked, k = list(range(n)), set(), 0
        while len(cand) > 1:
            qi = eng.best(cand, asked)
            if qi is None:
                break
            a = bool(eng.ans[secret, qi]) != (k == pos)
            asked.add(qi)
            k += 1
            cand = [i for i in cand if bool(eng.ans[i, qi]) == a]
        ok += cand == [secret]
    return ok / n


def main() -> int:
    n = len(CLIPS)
    check("64 clips in the archive", n == 64, str(n))
    check("clips span every default video", {c["video"] for c in CLIPS} == set(hc.DEFAULT_VIDEOS))
    # 1. results are reproducible from the stored attributes (engine is deterministic)
    qs = hc.make_questions(CLIPS)
    eng = hc.Engine(CLIPS, qs)
    plays = [eng.play(i) for i in range(n)]
    found = sum(p["found"] for p in plays)
    mean_q = float(np.mean([p["n_questions"] for p in plays]))
    c = R["corpus"]
    check("recomputed found count equals results.json", found == c["n_found"], f"{found} vs {c['n_found']}")
    check("recomputed mean questions equals results.json", abs(mean_q - c["mean_questions_all"]) < 1e-9, f"{mean_q:.3f}")
    # 2. information-theoretic floor: mean depth of a binary question tree >= entropy of the leaf (attribute-vector group) sizes
    from collections import Counter
    sizes = Counter(hc.attr_vector(x) for x in CLIPS).values()
    floor = -sum(g / n * math.log2(g / n) for g in sizes)
    check("mean questions >= entropy of attribute-vector groups", mean_q >= floor - 1e-9, f"{mean_q:.3f} >= {floor:.3f}")
    # 3. a clip is 'found' iff its attribute vector is unique; twins are exactly the identical vectors
    tw = {cid for g in hc.twins(CLIPS) for cid in g}
    ok = all(p["found"] == (CLIPS[i]["id"] not in tw) for i, p in enumerate(plays))
    check("found <=> attribute vector unique (no spurious failures)", ok)
    check("twin groups share an identical vector", all(len({hc.attr_vector(next(x for x in CLIPS if x['id'] == i)) for i in g}) == 1
                                                         for g in hc.twins(CLIPS)))
    check("every question splits the archive", all(0 < sum(bool(a) for a in eng.ans[:, i]) < n for i in range(len(qs))))
    check("truthful answers never empty the candidate set", all(len(p["final"]) >= 1 and p["secret"] in p["final"] for p in plays))
    # 4. kill test arithmetic matches the criterion
    k = R["kill_test"]
    check("kill test verdict is consistent", k["passed"] == (k["mean_questions"] <= 7 and k["found_rate"] >= 0.9),
          f"mean {k['mean_questions']:.2f}, found {k['found_rate']:.2f}, passed={k['passed']}")
    # 5. JS engine in the HTML == Python engine: strict engine, page protocol (truthful) and page protocol with deliberately wrong answers
    node = shutil.which("node")
    if node:
        js = ("const fs=require('fs');const {makeEngine}=require(process.argv[1]);"
              "const D=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));const E=makeEngine(D);"
              "const out={strict:[],p0:[],p1:[],p2:[]};"
              "for(let s=0;s<E.N;s++){const r=E.simulate(s);out.strict.push([r.n,r.final]);"
              "for(const [k,f] of [['p0',[]],['p1',[1]],['p2',[0,3]]]){const q=E.playProtocol(s,f);out[k].push([q.found,q.n,q.guesses]);}}"
              "console.log(JSON.stringify(out));")
        r = subprocess.run([node, "-e", js, str(HERE / "engine.js"), str(OUT / "_work" / "game_data.json")], capture_output=True, text=True, timeout=50)
        if r.returncode == 0:
            res = json.loads(r.stdout)
            same = all(res["strict"][i][0] == plays[i]["n_questions"] and res["strict"][i][1] == plays[i]["final"] for i in range(n))
            check("JS strict engine reproduces Python engine for all 64 secrets", same)
            for key, fl in (("p0", None), ("p1", {1}), ("p2", {0, 3})):
                py = [eng.run_protocol(i, flip_at=fl) for i in range(n)]
                ok = all(res[key][i] == [py[i]["found"], py[i]["n_questions"], py[i]["guesses"]] for i in range(n))
                check(f"JS page protocol == Python protocol for all secrets (flipped answers at {sorted(fl) if fl else 'none'})", ok)
            check("page protocol finds every clip with a truthful judge", all(x[0] for x in res["p0"]))
            rec = np.mean([x[0] for x in res["p1"]])
            check("page protocol survives one wrong answer (>= 90% found)", rec >= 0.9, f"{rec:.2f}")
            check("hard-only engine would NOT survive it (the recovery is doing work)", hard_with_flip(eng, n, 1) < 0.9,
                  f"{hard_with_flip(eng, n, 1):.2f} found")
        else:
            check("JS engine runs under node", False, r.stderr[:200])
    else:
        check("node available for JS parity test", False, "node not found")
    gd = json.loads((OUT / "_work" / "game_data.json").read_text(encoding="utf-8"))
    qtxt = [q["text"] for q in gd["questions"]]
    check("no 'dominant colour' question on the page", not any("colour" in t.lower() for t in qtxt))
    check("count questions say 'at any moment'", any(t.startswith("At any moment") for t in qtxt))
    check("caption-keyword questions exist", sum(q["attr"] == "words" for q in gd["questions"]) >= 5)
    check("every clip carries chips + caption for the hover card", all(c["chips"] and "caption" in c for c in gd["clips"]))
    html = (OUT / "hide_and_seek.html").read_text(encoding="utf-8")
    check("game html is self-contained and filled", "__DATA__" not in html and "__ENGINE__" not in html and "makeEngine" in html
          and "Is it this one?" in html and len(html) > 200_000,
          f"{len(html) // 1024} KB")
    # 6. outputs per video
    for v in hc.DEFAULT_VIDEOS:
        h, rp = OUT / v / "hero.jpg", OUT / v / "report.html"
        if not (h.exists() and rp.exists()):
            check(f"{v}: hero + report exist", False)
            continue
        im = Image.open(h)
        arr = np.asarray(im.convert("L"), dtype=float)
        check(f"{v}: hero <=1280 px and not blank", im.width <= 1280 and arr.std() > 20, f"{im.width}x{im.height} std {arr.std():.0f}")
    h = OUT / "ALL" / "hero.jpg"
    im = Image.open(h)
    check("ALL/hero.jpg <=1280 px", im.width <= 1280, f"{im.size}")
    check("ALL/report.html exists", (OUT / "ALL" / "report.html").exists())
    check("64 clip mp4s cut", len(list((OUT / "clips").glob("*.mp4"))) >= 64)
    check("per-video results cover every video", set(R["per_video"]) == set(hc.DEFAULT_VIDEOS))
    check("blind clips really have zero tracks", all(c["extra"]["n_tracks"] == 0 for c in CLIPS if c["extra"]["blind"]))
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
