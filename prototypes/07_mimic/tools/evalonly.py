import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import mimic_core as mc, mimic_eval as me
d = mc.build_all(); df = d['df']; arr = mc.tl_arrays(df)
gt = json.loads((HERE/'ground_truth.json').read_text())
gres = {p: mc.match(mc.preset_query(p), df, arr, k=8) for p in mc.PRESETS}
t=time.time(); ev = me.evaluate(d, df, arr, gres, gt, gt['calibration']); print(time.time()-t)
print(json.dumps({k:v for k,v in ev.items() if k in ('simulated_self_retrieval','simulated_drawing_robustness','labelled_cost')}, indent=1))
