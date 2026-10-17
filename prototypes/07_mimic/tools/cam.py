import sys, time
sys.path.insert(0,'.')
import mimic_core as mc, numpy as np
names = sys.argv[1:] or mc.DEFAULT_CLIPS
for n in names:
    t0=time.time()
    w = mc.camera_windows(n)
    dur=(w.t1-w.t0).sum()
    ch = mc.camera_chain(n)
    print(f"{n:38s} windows={len(w):4d} covered={dur:7.1f}s moving_windows={int(w.moving.sum()):3d} cuts={int(ch['cut'].sum())} unrel={int((~ch['rel']).sum())} ({time.time()-t0:.0f}s)", flush=True)
