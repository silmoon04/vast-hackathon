"""Local reviewed-caption search demo. Standard library only; optional embeddings use HTTP."""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, sqlite3, subprocess, tempfile, time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"

def load_config(path=DEFAULT_CONFIG):
    path=Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    def local(p):
        p=Path(p).expanduser()
        return p if p.is_absolute() else (path.parent/p).resolve()
    for key in ("segments", "captions", "index", "html"):
        if key not in cfg: raise ValueError(f"config missing {key}")
    cfg["segments"] = local(cfg["segments"])
    cfg["captions"] = local(cfg["captions"])
    cfg["index"] = local(cfg["index"])
    cfg["html"] = local(cfg["html"])
    for key in ("caption_vectors","visual_vectors"):
        cfg[key]=local(cfg[key]) if cfg.get(key) else cfg["index"].with_name("caption-vectors.npz" if key=="caption_vectors" else "visual-vectors.npz")
    for key in ("camera_manifest","prepared_dir","detections"):
        if cfg.get(key):cfg[key]=local(cfg[key])
    if cfg.get("yolo",{}).get("weights"):
        cfg["yolo"]["weights"]=str(local(cfg["yolo"]["weights"]))
    cfg.setdefault("videos", {})
    cfg.setdefault("embedding", {})
    cfg.setdefault("ssh", {})
    cfg.setdefault("port", 18806)
    cfg.setdefault("top_k", 8)
    cfg["videos"]={k:str(local(v)) for k,v in cfg["videos"].items()}
    return cfg

def corpus(cfg):
    src_doc=json.loads(cfg["segments"].read_text(encoding="utf-8"))
    src=src_doc.get("segments",src_doc) if isinstance(src_doc,dict) else src_doc
    cap_path=Path(cfg["captions"])
    cap_files=sorted(cap_path.glob("*-reviewed.json")) if cap_path.is_dir() else [cap_path]
    if not cap_files:raise ValueError("caption directory must contain explicitly reviewed *-reviewed.json records")
    caps=[]
    for file in cap_files:
        doc=json.loads(file.read_text(encoding="utf-8"));part=doc.get("segments",doc) if isinstance(doc,dict) else doc
        if not isinstance(part,list):raise ValueError(f"{file.name} must contain a caption list")
        reviewed_file="reviewed" in file.stem.casefold()
        for row in part:
            row=dict(row)
            if reviewed_file and row.get("reviewed") is not False:row["reviewed"]=True
            caps.append(row)
    if not isinstance(src,list) or not isinstance(caps,list): raise ValueError("segments and captions must be arrays or {segments: []}")
    def unique(rows,label):
        result={}
        for r in rows:
            sid=r.get("id")
            if not sid or sid in result: raise ValueError(f"missing/duplicate {label} id: {sid!r}")
            result[sid]=r
        return result
    source=unique(src,"source"); reviewed=unique(caps,"caption"); verified_files={}
    if set(source)!=set(reviewed): raise ValueError("source and reviewed caption IDs must match exactly")
    out=[]
    for sid,s in source.items():
        c=reviewed[sid]
        if c.get("reviewed") is not True: raise ValueError(f"caption {sid} is not explicitly reviewed; use a reviewed record or reviewed filename")
        for field in ("camera","clip_start","clip_end","source_start","source_end"):
            if c.get(field)!=s.get(field): raise ValueError(f"metadata or source hash mismatch for {sid}: {field}")
        sha=s.get("source_sha256","")
        if not re.fullmatch(r"[0-9a-fA-F]{64}",str(sha)): raise ValueError(f"invalid source SHA-256 for {sid}")
        source_file=s.get("source_file")
        if not source_file:raise ValueError(f"missing source_file for {sid}; source hash cannot be verified")
        source_path=Path(source_file).expanduser()
        if not source_path.is_absolute():source_path=(cfg["segments"].parent/source_path).resolve()
        key=str(source_path)
        if key not in verified_files:
            digest=hashlib.sha256()
            with source_path.open('rb') as fp:
                for chunk in iter(lambda:fp.read(1024*1024),b''):digest.update(chunk)
            verified_files[key]=digest.hexdigest()
        if verified_files[key]!=sha.lower():raise ValueError(f"source file hash mismatch for {sid}")
        if c.get("source_sha256") and c["source_sha256"].lower()!=sha.lower():raise ValueError(f"caption source hash mismatch for {sid}")
        caption=str(c.get("caption","")).strip()
        if not caption: raise ValueError(f"empty caption for {sid}")
        out.append(dict(id=sid,camera=s["camera"],clip_start=s["clip_start"],clip_end=s["clip_end"],source_start=s["source_start"],source_end=s["source_end"],source_sha256=sha.lower(),caption=caption))
    return out

def build_index(cfg, rows):
    """Build side-by-side and swap only after success; existing embeddings are not discarded."""
    target=cfg["index"]; target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_name(target.name+".building")
    if tmp.exists(): tmp.unlink()
    try:
        con=sqlite3.connect(tmp)
        con.executescript("CREATE TABLE captions(id TEXT PRIMARY KEY,camera TEXT,clip_start REAL,clip_end REAL,source_start REAL,source_end REAL,source_sha256 TEXT,caption TEXT); CREATE VIRTUAL TABLE captions_fts USING fts5(id UNINDEXED,camera UNINDEXED,caption,tokenize='unicode61'); CREATE TABLE index_metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);")
        con.executemany("INSERT INTO captions VALUES(:id,:camera,:clip_start,:clip_end,:source_start,:source_end,:source_sha256,:caption)",rows)
        con.executemany("INSERT INTO captions_fts VALUES(:id,:camera,:caption)",rows); con.commit(); con.close()
        con=sqlite3.connect(tmp)
        fingerprint=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        con.execute("INSERT INTO index_metadata VALUES('corpus_sha256',?)",(fingerprint,));con.commit();con.close()
        if target.exists():
            old=sqlite3.connect(target)
            try:
                tables={x[0] for x in old.execute("select name from sqlite_master where type='table'")}
                if "embeddings" in tables:
                    old_ids={r[0] for r in old.execute("SELECT id FROM captions")}
                    new_ids={r["id"] for r in rows}
                    if old_ids!=new_ids:
                        raise ValueError("existing vector rows may not align with rebuilt captions; remove the old index explicitly before rebuilding")
                    old_hashes=dict(old.execute("SELECT id,source_sha256 FROM captions"))
                    if any(old_hashes.get(r["id"]) != r["source_sha256"] for r in rows):
                        raise ValueError("source hashes changed since vectors were indexed; remove the old index explicitly before rebuilding")
                    old_text=dict(old.execute("SELECT id,caption FROM captions"))
                    if any(old_text.get(r["id"]) != r["caption"] for r in rows):
                        raise ValueError("caption text changed since vectors were indexed; remove the old index explicitly before rebuilding")
                    new=sqlite3.connect(tmp)
                    try:
                        new.execute("CREATE TABLE embeddings(id TEXT PRIMARY KEY,model TEXT,revision TEXT,dimension INTEGER,vector BLOB)")
                        vector_rows=list(old.execute("SELECT id,model,revision,dimension,vector FROM embeddings ORDER BY id"))
                        if {r[0] for r in vector_rows} != new_ids:
                            raise ValueError("existing vector IDs do not exactly match captions; remove the old index explicitly before rebuilding")
                        if any(not r[1] or not r[2] or not r[3] or len(r[4]) != r[3]*4 for r in vector_rows):
                            raise ValueError("existing vector provenance or dimensions are invalid")
                        if len({(r[1],r[2],r[3]) for r in vector_rows})>1:
                            raise ValueError("existing vectors mix models, checkpoints, or dimensions")
                        new.executemany("INSERT INTO embeddings VALUES(?,?,?,?,?)",vector_rows)
                        new.commit()
                    finally: new.close()
            finally: old.close()
        os.replace(tmp,target)
    finally:
        if tmp.exists(): tmp.unlink()

STOP={"the","a","an","to","is","of","in","at","by","on","and","with","from"}
def search(db, query, top_k=8):
    toks=list(dict.fromkeys(x for x in re.findall(r"[\w]+",query.casefold()) if x not in STOP))
    if not toks:return []
    match=" OR ".join('"'+x.replace('"','""')+'"' for x in toks)
    con=sqlite3.connect(db); con.row_factory=sqlite3.Row
    try: return [dict(r) for r in con.execute("SELECT c.*,bm25(captions_fts) score FROM captions_fts JOIN captions c USING(id) WHERE captions_fts MATCH ? ORDER BY score LIMIT ?",(match,top_k))]
    finally:con.close()

def vector_sources(cfg, rows):
    """Load normalized vectors only when IDs, source hashes, shape and provenance agree."""
    import numpy as np
    sources={}
    for name,key in (("caption","caption_vectors"),("visual","visual_vectors")):
        path=cfg.get(key)
        if not path or not Path(path).is_file():continue
        with np.load(path,allow_pickle=False) as z:
            v=np.asarray(z["vectors"],dtype=np.float32); ids=[str(x) for x in z["ids"].tolist()]
            hashes=[str(x) for x in z["source_sha256"].tolist()]
            caption_hashes=[str(x) for x in z["caption_sha256"].tolist()] if "caption_sha256" in z else None
            model=str(z["model"].item()); revision=str(z["revision"].item())
        expected=[r["id"] for r in rows]
        if ids!=expected or hashes!=[r["source_sha256"] for r in rows]:raise ValueError(f"{name} vector IDs/order/source hashes do not match the reviewed corpus")
        if name=="caption" and caption_hashes!=[hashlib.sha256(r["caption"].encode('utf-8')).hexdigest() for r in rows]:raise ValueError("caption vector content hashes do not match reviewed captions")
        if v.ndim!=2 or len(v)!=len(rows) or not model or not revision or not np.isfinite(v).all():raise ValueError(f"invalid {name} vector array or missing model provenance")
        emb=cfg.get("embedding",{})
        if emb.get("model") and model!=emb["model"]:raise ValueError(f"{name} cache model does not match configured query model")
        if emb.get("checkpoint") and revision!=emb["checkpoint"]:raise ValueError(f"{name} cache checkpoint does not match configured query checkpoint")
        norms=np.linalg.norm(v,axis=1)
        if np.any(norms==0):raise ValueError(f"zero vector in {name} cache")
        sources[name]=(v/norms[:,None],model,revision)
    return sources

def query_vector(cfg, query):
    import numpy as np
    emb=cfg.get("embedding",{}); url=emb.get("url"); model=emb.get("model")
    if not url or not model:raise RuntimeError("embedding service URL/model not configured")
    req=__import__("urllib.request",fromlist=["Request"]).Request(url,data=json.dumps({"model":model,"input":[query]},ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
    with __import__("urllib.request",fromlist=["urlopen"]).urlopen(req,timeout=float(emb.get("timeout",20))) as response:doc=json.loads(response.read())
    data=sorted(doc.get("data",[]),key=lambda x:x.get("index",0))
    if len(data)!=1 or not data[0].get("embedding"):raise ValueError("embedding response did not contain one query vector")
    v=np.asarray(data[0]["embedding"],dtype=np.float32); norm=float(np.linalg.norm(v))
    if v.ndim!=1 or norm==0 or not np.isfinite(v).all():raise ValueError("invalid query embedding")
    return v/norm

def retrieve(cfg, rows, query, top_k):
    import numpy as np
    lexical=search(cfg["index"],query,max(18,top_k*3)); indexed={r["id"]:r for r in rows}
    rankings={"fts":[(r["id"],float(r["score"])) for r in lexical]}; maps={"fts":dict(rankings["fts"])}
    mode="SQLite FTS5 lexical"
    try:
        vectors=vector_sources(cfg,rows)
        if vectors:
            qv=query_vector(cfg,query)
            for name,(arr,_model,_rev) in vectors.items():
                if arr.shape[1]!=len(qv):raise ValueError(f"{name} vector dimension differs from query embedding")
                scored=arr@qv; order=np.argsort(-scored)
                rankings[name]=[(rows[int(i)]["id"],float(scored[i])) for i in order]; maps[name]=dict(rankings[name])
            mode="SQLite FTS5 + "+cfg.get("embedding",{}).get("model","configured embedding model")+" "+"/".join(vectors)+" similarity; RRF"
        elif cfg.get("caption_vectors") or cfg.get("visual_vectors"):
            mode="SQLite FTS5 lexical (vector cache unavailable)"
    except Exception as exc:
        # The lexical result remains available, and the API names the fallback honestly.
        mode=f"SQLite FTS5 lexical (embedding unavailable: {type(exc).__name__})"
        rankings={"fts":rankings["fts"]};maps={"fts":maps["fts"]}
    ranks={name:{sid:i for i,(sid,_) in enumerate(items,1)} for name,items in rankings.items()}
    pool=set(sid for items in rankings.values() for sid,_ in items[:max(top_k,18)])
    results=[]
    for sid in pool:
        item=dict(indexed[sid]); item["rrf_score"]=sum(1/(60+ranks[n][sid]) for n in ranks if sid in ranks[n])
        for name in ("fts","caption","visual"):
            item[name+"_rank"]=ranks.get(name,{}).get(sid)
            if sid in maps.get(name,{}):item[name+"_score"]=maps[name][sid]
        results.append(item)
    return sorted(results,key=lambda x:(-x["rrf_score"],x["id"]))[:top_k],mode

def answer_remote(cfg, payload):
    ssh=cfg["ssh"]
    if not ssh.get("target") or not ssh.get("command"): raise RuntimeError("SSH target and remote command must be configured")
    proc=subprocess.run(["ssh",ssh["target"],ssh["command"]],input=json.dumps(payload,ensure_ascii=False),text=True,encoding="utf-8",capture_output=True,timeout=int(ssh.get("timeout",90)))
    if proc.returncode: raise RuntimeError("remote worker failed")
    result=json.loads(proc.stdout)
    if not result.get("results") or result["results"][0].get("status")!=200:raise RuntimeError("remote model request failed")
    return result

def handler_for(cfg, rows):
    allowed={k:Path(v).expanduser().resolve() for k,v in cfg["videos"].items()}
    class H(BaseHTTPRequestHandler):
        def log_message(self,*a): pass
        def send_json(self,status,obj):
            b=json.dumps(obj,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            path=urlsplit(self.path).path
            if path=="/":
                b=cfg["html"].read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
            if path.startswith("/video/"):
                sid=unquote(path[len('/video/'):]); p=allowed.get(sid)
                if not p or not p.is_file(): self.send_error(404); return
                size=p.stat().st_size; start,end,status=0,size-1,200; rh=self.headers.get("Range","")
                try:
                    if rh:
                        if not rh.startswith("bytes=") or "," in rh: raise ValueError
                        a,b=rh[6:].split("-",1)
                        if a: start=int(a); end=int(b) if b else size-1
                        else: start=max(0,size-int(b))
                        if start<0 or start>=size or end<start: raise ValueError
                        end=min(end,size-1); status=206
                except ValueError:self.send_error(416);return
                n=end-start+1; self.send_response(status); self.send_header("Content-Type","video/mp4"); self.send_header("Accept-Ranges","bytes"); self.send_header("Content-Length",str(n))
                if status==206:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
                self.end_headers()
                with p.open('rb') as f:f.seek(start);self.wfile.write(f.read(n))
                return
            self.send_error(404)
        def do_POST(self):
            if urlsplit(self.path).path!="/api/search":self.send_error(404);return
            try:
                n=int(self.headers.get("Content-Length","0"))
                if n<=0 or n>64000: self.send_json(413,{"error":"Request must be under 64 KB"});return
                body=json.loads(self.rfile.read(n)); q=body.get("query","")
                if not isinstance(q,str) or not q.strip() or len(q)>500:self.send_json(400,{"error":"Enter a query of 1–500 characters"});return
                hits,retrieval_mode=retrieve(cfg,rows,q.strip(),int(cfg["top_k"]))
                for hit in hits:hit["video_available"]=hit["id"] in allowed
                coverage=defaultdict(int)
                for r in rows:coverage[r["camera"]]+=1
                payload={"queries":[{"query":q.strip(),"retrieval_mode":retrieval_mode,"coverage":{"indexed_captions":len(rows),"retrieved_evidence":len(hits),"by_camera":coverage},"evidence":hits}]}
                try:result=answer_remote(cfg,payload); mode=retrieval_mode+"; DeepSeek via SSH"
                except Exception as e: result={"error":type(e).__name__}; mode=retrieval_mode+"; remote reasoning unavailable"
                answer=None
                if isinstance(result,dict):
                    answer=((result.get("results") or [{}])[0].get("answer"))
                self.send_json(200,{"query":q.strip(),"evidence":hits,"retrieval_mode":mode,"answer":answer,"result":result})
            except Exception:self.send_json(400,{"error":"Invalid search request"})
    return H

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=DEFAULT_CONFIG);p.add_argument("--rebuild",action="store_true");p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int);a=p.parse_args()
    if a.host not in ("127.0.0.1","localhost"):raise SystemExit("demo binds to loopback only")
    cfg=load_config(a.config); rows=corpus(cfg)
    stale=True
    if cfg["index"].exists():
        try:
            con=sqlite3.connect(cfg["index"]);meta=dict(con.execute("SELECT key,value FROM index_metadata"));con.close()
            stale=meta.get('corpus_sha256')!=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        except sqlite3.Error:stale=True
    if a.rebuild or stale:build_index(cfg,rows)
    server=ThreadingHTTPServer((a.host,a.port or int(cfg["port"])),handler_for(cfg,rows));print(f"http://{a.host}:{server.server_port}",flush=True);server.serve_forever()
if __name__=="__main__":main()
