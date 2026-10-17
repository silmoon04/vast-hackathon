import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import numpy as np, cv2, mimic_core as mc, mimic_render as mr
pk = mc.pk
d = mc.build_all()
res = json.loads((HERE/'results.json').read_text())
pr, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows=[]
for r in res['global'][pr]['top8'][a:b]:
    s = mr.filmstrip(d, r['key'], n=6, w=210)
    pk.label(s, f"{pr} {r['key']} cost {r['cost']}", (s.shape[1]-420, s.shape[0]-24), scale=0.45)
    rows.append(s)
W = max(r.shape[1] for r in rows)
out = np.vstack([cv2.copyMakeBorder(r,0,0,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=(24,24,24)) for r in rows])
pk.save_jpg(out, HERE/'out'/'_sheets'/f'strips_{pr}_{a}.jpg', 80); print(out.shape)
