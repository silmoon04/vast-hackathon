"""Rebuild out/draw.html only (no heroes / clips): python tools/rebuild_page.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mimic_core as mc  # noqa: E402
import build_page  # noqa: E402

d = mc.build_all()
df = d["df"]
arr = mc.tl_arrays(df)
cal = json.loads((HERE / "ground_truth.json").read_text(encoding="utf-8")).get("calibration", {})
gres = {p: mc.match(mc.preset_query(p), df, arr, k=8) for p in mc.PRESETS}
build_page.build(d, df, cal, gres)
