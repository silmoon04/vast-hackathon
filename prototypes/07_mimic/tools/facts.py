import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import numpy as np, mimic_core as mc, mimic_render as mr
pk = mc.pk
d = mc.build_all(); df = d['df']; arr = mc.tl_arrays(df)
res = json.loads((HERE/'results.json').read_text())
def heading(row):
    p = np.array(row['pos']).reshape(-1,2)*[d['meta'][row.video]['w']/d['meta'][row.video]['h'],1]
    v = p[-1]-p[0]; return np.degrees(np.arctan2(v[1], v[0]))
for pr in sys.argv[1:]:
    print('==', pr)
    for r in res['global'][pr]['top8']:
        row = df[df.key==r['key']].iloc[0]
        print(f"{r['key']:42s} cost {r['cost']:.2f} dur {row.dur:5.1f} len {row['len']:.2f} ext {row.ext:.2f} rel {row.rel_ext:.1f} heading {heading(row):6.0f} turn {row.turn_total:5.2f} stops {row.stops} stopfrac {row.stop_frac:.2f} straight {row.straight:.2f}")
