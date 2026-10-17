"""Re-render only out/hero.jpg (global u_turn hero with hand-check tags): python tools/global_hero.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
import mimic_core as mc  # noqa: E402

pk = mc.pk
d = mc.build_all()
df = d["df"]
arr = mc.tl_arrays(df)
gt = R.load_gt()
cal = gt["calibration"]
res = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
ev = res["eval"]
gres = {p: mc.match(mc.preset_query(p), df, arr, k=8) for p in R.PRESET_ORDER}
badges = [(p.replace("_", " "), f"hand-checked P@3 {ev['precision'][p]['p_at_3']:.2f}  ({ev['precision'][p]['correct_in_top8']}/8 right)",
           f"best cost {gres[p].cost.iloc[0]:.2f} from {gres[p].video.iloc[0]}") for p in R.EVAL4]
hero = R.make_hero(d, mc.preset_query("u_turn"), gres["u_turn"], "all clips", f"{len(df)} tracklets | {sum(m['duration'] for m in d['meta'].values()) / 60:.0f} min of footage", cal,
                   "global search across 21 clips | cosine kNN shortlist + DTW re-rank | thresholds set by eye on the same hand-checked results", badges=badges,
                   labels=gt.get("labels", {}).get("u_turn", {}))
pk.save_jpg(hero, R.OUT / "hero.jpg", 88)
pk.save_jpg(hero, R.OUT / "global" / "hero.jpg", 88)
print("ok")
