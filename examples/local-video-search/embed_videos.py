"""Embed prepared video segments through vastkit's local Cosmos Embed1 HTTP service."""
import argparse,base64,json,time
from pathlib import Path
from urllib.request import Request,urlopen
import search

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=search.DEFAULT_CONFIG);a=p.parse_args();cfg=search.load_config(a.config)
    import numpy as np
    rows=json.loads(cfg['segments'].read_text(encoding='utf-8'));emb=cfg['embedding'];allv=[];t0=time.perf_counter();batch=int(emb.get('video_batch_size',1))
    for start in range(0,len(rows),batch):
        pack=rows[start:start+batch];items=[]
        for r in pack:
            path=Path(r['video']).expanduser()
            if not path.is_file():raise FileNotFoundError(path)
            items.append('data:video/mp4;base64,'+base64.b64encode(path.read_bytes()).decode('ascii'))
        req=Request(emb['url'],data=json.dumps({'model':emb['model'],'input':items}).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=float(emb.get('video_timeout',300))) as resp:doc=json.loads(resp.read())
        data=sorted(doc.get('data',[]),key=lambda x:x.get('index',0))
        if len(data)!=len(pack) or [x.get('index') for x in data]!=list(range(len(pack))):raise ValueError('video embedding response order/count mismatch')
        allv.extend(x['embedding'] for x in data)
    vectors=np.asarray(allv,dtype=np.float32)
    if vectors.ndim!=2 or not np.isfinite(vectors).all():raise ValueError('invalid vectors')
    norms=np.linalg.norm(vectors,axis=1)
    if np.any(norms==0):raise ValueError('zero vector')
    vectors/=norms[:,None]
    out=cfg.get('visual_vectors') or cfg['index'].with_name('visual-vectors.npz')
    np.savez_compressed(out,vectors=vectors,ids=np.array([r['id'] for r in rows]),source_sha256=np.array([r['source_sha256'] for r in rows]),model=np.array(emb['model']),revision=np.array(emb.get('checkpoint') or doc.get('model') or emb['model']))
    print(json.dumps({'segment_count':len(rows),'dimensions':int(vectors.shape[1]),'model':emb['model'],'wall_seconds':time.perf_counter()-t0,'output':str(out)}))
if __name__=='__main__':main()
