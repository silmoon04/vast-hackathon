"""Render the hero for any preset over all clips (or one): python tools/preset_hero.py stop_and_go [clip]  -> out/global/hero_<preset>.jpg"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
import mimic_core as mc  # noqa: E402

pk = mc.pk
p = sys.argv[1]
clip = sys.argv[2] if len(sys.argv) > 2 else None
d = mc.build_all()
df = d["df"]
arr = mc.tl_arrays(df)
gt = R.load_gt()
sub, a = df, arr
if clip:
    m = (df.video == clip).values
    sub, a = df[m].reset_index(drop=True), {k: v[m] for k, v in arr.items()}
r = mc.match(mc.preset_query(p), sub, a, k=8)
hero = R.make_hero(d, mc.preset_query(p), r, clip or "all clips", f"{len(sub)} tracklets", gt["calibration"], "preset hero", labels=None if clip else gt.get("labels", {}).get(p, {}))
out = R.OUT / "global" / f"hero_{p}{'_' + clip if clip else ''}.jpg"
pk.save_jpg(hero, out, 88)
print(out)
