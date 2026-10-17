import json, tempfile, unittest, hashlib
from pathlib import Path
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
import search

class SearchTests(unittest.TestCase):
    def cfg(self, d):
        d=Path(d); return {"segments":d/"s.json","captions":d/"c.json","index":d/"i.sqlite3","html":d/"index.html","videos":{},"ssh":{},"top_k":8}
    def records(self, root):
        media=Path(root)/'source.mp4';media.write_bytes(b'fictional test media');h=hashlib.sha256(media.read_bytes()).hexdigest()
        s={"id":"x","camera":"cam","clip_start":0,"clip_end":5,"source_start":9,"source_end":14,"source_file":str(media),"source_sha256":h}
        c={**s,"reviewed":True,"caption":"München: person beside bicycle"}
        return [s],[c]
    def test_review_hash_alignment_and_unicode_search(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.cfg(d); s,c=self.records(d); cfg['segments'].write_text(json.dumps(s));cfg['captions'].write_text(json.dumps(c))
            rows=search.corpus(cfg);search.build_index(cfg,rows)
            self.assertEqual(search.search(cfg['index'],'München')[0]['id'],'x')
            c[0]['source_sha256']='b'*64;cfg['captions'].write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'source hash mismatch'):search.corpus(cfg)
    def test_unreviewed_and_nonmatching_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.cfg(d);s,c=self.records(d);c[0]['reviewed']=False
            cfg['segments'].write_text(json.dumps(s));cfg['captions'].write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'not explicitly reviewed'):search.corpus(cfg)
    def test_config_paths_are_relative_to_config(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.json';p.write_text(json.dumps({"segments":"s","captions":"c","index":"db","html":"page.html"}))
            self.assertEqual(search.load_config(p)['index'],Path(d)/'db')
    def test_video_range_and_camera_allowlist(self):
        with tempfile.TemporaryDirectory() as d:
            video=Path(d)/'clip.mp4';video.write_bytes(b'0123456789')
            cfg={"html":Path(d)/'page.html',"index":Path(d)/'db','videos':{'cam':str(video)},'ssh':{},'top_k':3}
            cfg['html'].write_text('<p>demo</p>')
            server=ThreadingHTTPServer(('127.0.0.1',0),search.handler_for(cfg,[]))
            Thread(target=server.serve_forever,daemon=True).start()
            try:
                req=Request(f'http://127.0.0.1:{server.server_port}/video/cam',headers={'Range':'bytes=3-6'})
                with urlopen(req) as response:
                    self.assertEqual(response.status,206);self.assertEqual(response.read(),b'3456')
                    self.assertEqual(response.headers['Content-Range'],'bytes 3-6/10')
                with self.assertRaises(HTTPError):urlopen(f'http://127.0.0.1:{server.server_port}/video/../secret')
            finally:server.shutdown();server.server_close()
    def test_vector_order_hash_and_checkpoint_are_enforced(self):
        try:import numpy as np
        except ImportError:self.skipTest('numpy optional dependency is not installed')
        with tempfile.TemporaryDirectory() as d:
            import hashlib
            cfg=self.cfg(d); rows=[]
            for sid,h,text in [('a','a'*64,'one'),('b','b'*64,'two')]:rows.append({'id':sid,'source_sha256':h,'caption':text})
            path=Path(d)/'vectors.npz';np.savez_compressed(path,vectors=np.eye(2,dtype=np.float32),ids=np.array(['a','b']),source_sha256=np.array(['a'*64,'b'*64]),caption_sha256=np.array([hashlib.sha256(b'one').hexdigest(),hashlib.sha256(b'two').hexdigest()]),model=np.array('embed-v1'),revision=np.array('rev1'))
            cfg.update(caption_vectors=path,embedding={'model':'embed-v1','checkpoint':'rev1'})
            self.assertIn('caption',search.vector_sources(cfg,rows))
            cfg['embedding']['checkpoint']='rev2'
            with self.assertRaisesRegex(ValueError,'checkpoint'):search.vector_sources(cfg,rows)
            cfg['embedding']['checkpoint']='rev1';rows.reverse()
            with self.assertRaisesRegex(ValueError,'IDs/order'):search.vector_sources(cfg,rows)
            rows.reverse();rows[0]['caption']='changed'
            with self.assertRaisesRegex(ValueError,'content hashes'):search.vector_sources(cfg,rows)
    def test_rebuild_refuses_stale_vectors_and_keeps_old_database(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=self.cfg(d);s,c=self.records(d);cfg['segments'].write_text(json.dumps(s));cfg['captions'].write_text(json.dumps(c))
            rows=search.corpus(cfg);search.build_index(cfg,rows)
            con=__import__('sqlite3').connect(cfg['index']);con.execute('CREATE TABLE embeddings(id TEXT PRIMARY KEY,model TEXT,revision TEXT,dimension INTEGER,vector BLOB)');con.execute('INSERT INTO embeddings VALUES(?,?,?,?,?)',('x','m','r',1,b'\x00'*4));con.commit();con.close()
            c[0]['source_sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'hashes changed'):search.build_index(cfg,[{**rows[0],'source_sha256':'b'*64}])
            con=__import__('sqlite3').connect(cfg['index'])
            try:
                self.assertEqual(con.execute('SELECT source_sha256 FROM captions').fetchone()[0],rows[0]['source_sha256'])
                self.assertEqual(con.execute('SELECT count(*) FROM embeddings').fetchone()[0],1)
            finally:con.close()

if __name__=='__main__':unittest.main()
