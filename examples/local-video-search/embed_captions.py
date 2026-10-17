"""Create caption vector cache through the configured OpenAI-compatible embedding service."""
import argparse, hashlib, json, time
from pathlib import Path
from urllib.request import Request,urlopen
import search

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=search.DEFAULT_CONFIG);a=p.parse_args();cfg=search.load_config(a.config)
    import numpy as np
    rows=search.corpus(cfg); emb=cfg['embedding']; texts=[r['caption'] for r in rows]; allv=[];t0=time.perf_counter();batch=int(emb.get('batch_size',8))
    for start in range(0,len(texts),batch):
        items=texts[start:start+batch];body=json.dumps({'model':emb['model'],'input':items},ensure_ascii=False).encode()
        req=Request(emb['url'],data=body,headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=float(emb.get('timeout',120))) as resp:doc=json.loads(resp.read())
        data=sorted(doc.get('data',[]),key=lambda x:x.get('index',0))
        if len(data)!=len(items) or [x.get('index') for x in data]!=list(range(len(items))):raise ValueError('embedding response IDs/order/count mismatch')
        allv.extend(x['embedding'] for x in data)
    vectors=np.asarray(allv,dtype=np.float32)
    if vectors.ndim!=2 or not np.isfinite(vectors).all():raise ValueError('invalid vectors')
    norms=np.linalg.norm(vectors,axis=1)
    if np.any(norms==0):raise ValueError('zero vector')
    vectors/=norms[:,None]
    out=cfg.get('caption_vectors') or cfg['index'].with_name('caption-vectors.npz')
    revision=emb.get('checkpoint') or doc.get('model') or emb['model']
    np.savez_compressed(out,vectors=vectors,ids=np.array([r['id'] for r in rows]),source_sha256=np.array([r['source_sha256'] for r in rows]),caption_sha256=np.array([hashlib.sha256(r['caption'].encode('utf-8')).hexdigest() for r in rows]),model=np.array(emb['model']),revision=np.array(revision))
    print(json.dumps({'caption_count':len(rows),'dimensions':int(vectors.shape[1]),'model':emb['model'],'checkpoint':revision,'wall_seconds':time.perf_counter()-t0,'output':str(out)}))
if __name__=='__main__':main()
