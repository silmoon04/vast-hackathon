"""Prepare short clips and run optional YOLO tracking over a configurable camera manifest."""
from __future__ import annotations
import argparse,hashlib,json,re,subprocess,time
from pathlib import Path
import search

def manifest(cfg):
    path=cfg.get('camera_manifest')
    if not path:raise ValueError('set camera_manifest to an input JSON manifest')
    path=Path(path).resolve();doc=json.loads(path.read_text(encoding='utf-8'));cams=doc.get('cameras')
    if not isinstance(cams,list) or not cams:raise ValueError('camera manifest needs a nonempty cameras list')
    seen=set()
    for c in cams:
        if not isinstance(c.get('id'),str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',c['id']) or c['id'] in ('.','..') or c['id'] in seen:raise ValueError('camera IDs must be unique and match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}')
        seen.add(c['id']);c['path']=Path(c['file']).expanduser();c['path']=c['path'] if c['path'].is_absolute() else (path.parent/c['path']).resolve()
        if not c['path'].is_file():raise FileNotFoundError(c['path'])
        digest=hashlib.sha256()
        with c['path'].open('rb') as fp:
            for chunk in iter(lambda:fp.read(1024*1024),b''):digest.update(chunk)
        actual=digest.hexdigest()
        if c.get('sha256') and c['sha256'].lower()!=actual:raise ValueError(f"source hash mismatch for {c['id']}")
        c['sha256']=actual
    return doc,cams

def prepare(cfg):
    doc,cams=manifest(cfg);dest=Path(cfg.get('prepared_dir',cfg['index'].parent/'segments')).expanduser();dest.mkdir(parents=True,exist_ok=True)
    duration=float(cfg.get('segment_seconds',5));origin=float(doc.get('source_clock_offset_seconds',0)); rows=[];t=time.perf_counter()
    if duration<=0:raise ValueError('segment_seconds must be positive')
    for cam in cams:
        total=float(cam['duration_seconds']);n=0;start=0.0
        if total<=0:raise ValueError(f"duration_seconds must be positive for {cam['id']}")
        while start<total:
            length=min(duration,total-start);sid=f"{cam['id']}-s{n:04d}";out=dest/(sid+'.mp4')
            subprocess.run(['ffmpeg','-v','error','-y','-ss',str(start),'-i',str(cam['path']),'-t',str(length),'-c:v','libx264','-preset','veryfast','-crf','20','-an',str(out)],check=True)
            rows.append({'id':sid,'camera':cam['id'],'clip_start':0,'clip_end':length,'source_start':origin+start,'source_end':origin+start+length,'source_file':str(cam['path']),'source_sha256':cam['sha256'],'video':str(out.resolve())});n+=1;start+=duration
    cfg['segments'].parent.mkdir(parents=True,exist_ok=True);cfg['segments'].write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'segments':len(rows),'source_manifest':str(cfg['camera_manifest']),'wall_seconds':time.perf_counter()-t,'output':str(cfg['segments'])}))

def detect(cfg,weights,stride):
    import cv2,torch,numpy as np
    from ultralytics import YOLO
    doc,cams=manifest(cfg);torch.set_num_threads(int(cfg.get('yolo',{}).get('torch_threads',4)));cv2.setNumThreads(1)
    if not Path(weights).is_file():raise FileNotFoundError(weights)
    model=YOLO(str(weights),task='detect');rows=[];reports=[];base=float(doc.get('source_clock_offset_seconds',0));total=time.perf_counter()
    for cam in cams:
        if model.predictor and getattr(model.predictor,'trackers',None):
            for tracker in model.predictor.trackers:tracker.reset()
        cap=cv2.VideoCapture(str(cam['path']));fps=float(cam.get('fps') or cap.get(cv2.CAP_PROP_FPS) or 30);idx=0;processed=0;times=[];counts={};ids=set();start=time.perf_counter()
        while True:
            ok,frame=cap.read()
            if not ok:break
            if idx%stride:idx+=1;continue
            st=time.perf_counter();r=model.track(frame,persist=True,tracker=cfg.get('yolo',{}).get('tracker','bytetrack.yaml'),device=cfg.get('yolo',{}).get('device',0),imgsz=int(cfg.get('yolo',{}).get('image_size',640)),conf=float(cfg.get('yolo',{}).get('confidence',.25)),verbose=False)[0]
            torch.cuda.synchronize() if torch.cuda.is_available() else None;times.append(time.perf_counter()-st);det=[];b=r.boxes
            if b is not None:
                cls=b.cls.cpu().tolist();scores=b.conf.cpu().tolist();boxes=b.xyxy.cpu().tolist();tids=b.id.cpu().tolist() if b.id is not None else [None]*len(cls)
                for k,sc,box,tid in zip(cls,scores,boxes,tids):
                    label=r.names[int(k)];tid=int(tid) if tid is not None else None
                    if tid is not None:ids.add(tid)
                    counts[label]=counts.get(label,0)+1;det.append({'label':label,'confidence':sc,'bbox':box,'track_id':tid})
            rows.append({'camera':cam['id'],'frame':idx,'clip_time':idx/fps,'source_time':base+idx/fps,'detections':det});processed+=1;idx+=1
        cap.release();elapsed=time.perf_counter()-start;reports.append({'camera':cam['id'],'processed_frames':processed,'decoded_frames':idx,'wall_seconds':elapsed,'processed_fps':processed/elapsed if elapsed else 0,'stride':stride,'latency_p50_ms':float(np.percentile(times,50))*1000 if times else None,'latency_p95_ms':float(np.percentile(times,95))*1000 if times else None,'raw_detection_counts':counts,'within_camera_track_ids':len(ids)})
    out=Path(cfg.get('detections',cfg['index'].with_name('detections.json')));out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    report=out.with_name(out.stem+'-timing.json');report.write_text(json.dumps({'weights':str(weights),'model_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),'total_wall_seconds':time.perf_counter()-total,'cameras':reports,'tracking':'ByteTrack; track IDs are camera-local, not cross-camera identities'},indent=2),encoding='utf-8')
    print(json.dumps({'detections':str(out),'timing':str(report),'camera_count':len(cams)}))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','detect']);p.add_argument('--config',type=Path,default=search.DEFAULT_CONFIG);p.add_argument('--weights');p.add_argument('--stride',type=int);a=p.parse_args();cfg=search.load_config(a.config)
    if a.stage=='prepare':prepare(cfg)
    else:
        weights=a.weights or cfg.get('yolo',{}).get('weights')
        if not weights:raise SystemExit('set yolo.weights in config or pass --weights')
        stride=a.stride or int(cfg.get('yolo',{}).get('stride',3));detect(cfg,weights,stride)
if __name__=='__main__':main()
