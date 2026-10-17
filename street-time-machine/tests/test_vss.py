"""Offline tests for stm.vss: a fake transport stands in for the VSS backend (no network, no sleeping)."""
import io
import json
import logging
import urllib.parse

import pytest

from stm import vss

URL = "https://vss.example.test"
USER = "team-zz-user"
PW = "pw-S3cr3t-0f-team"
CAM = "neighborhood_cam-1"


def _resp(status, obj=None, headers=None, raw=None):
    body = raw if raw is not None else (b"" if obj is None else json.dumps(obj).encode())
    return vss.Response(status, {k.lower(): v for k, v in (headers or {}).items()}, io.BytesIO(body))


class _Dropping(io.BytesIO):
    """Serves `cut` bytes, then the connection drops."""

    def __init__(self, data, cut):
        super().__init__(data)
        self.cut = cut

    def read(self, n=-1):
        if self.tell() >= self.cut:
            raise ConnectionResetError("peer reset")
        n = self.cut - self.tell() if n < 0 else min(n, self.cut - self.tell())
        return super().read(n)


class FakeVSS:
    """Routes like the backend; tokens are 'jwt.SECRET-<n>'. `fail[route]` = statuses/exceptions served first."""

    def __init__(self, chunks=(), segs=None, sidecars=None, videos=None, results=()):
        self.chunks, self.segs = list(chunks), segs or {}
        self.sidecars, self.videos, self.results = sidecars or {}, videos or {}, list(results)
        self.calls, self.fail, self.valid, self.n = [], {}, set(), 0
        self.ignore_range, self.drop_at = False, None

    def expire(self):
        self.valid.clear()

    def routes(self, name):
        return [c for c in self.calls if c["route"] == name]

    def __call__(self, method, url, headers, body, timeout):
        u = urllib.parse.urlsplit(url)
        route = u.path.split("/api/v1/", 1)[1]
        q = dict(urllib.parse.parse_qsl(u.query))
        payload = json.loads(body) if body else None
        self.calls.append({"method": method, "route": route, "q": q, "headers": dict(headers), "body": payload})
        if self.fail.get(route):
            f = self.fail[route].pop(0)
            if isinstance(f, Exception):
                raise f
            return _resp(f, {"detail": "flaky"})
        if route == "auth/login":
            if payload != {"username": USER, "password": PW}:
                return _resp(401, {"detail": "Invalid credentials"})
            self.n += 1
            tok = f"jwt.SECRET-{self.n}"
            self.valid.add(tok)
            return _resp(200, {"access_token": tok, "token_type": "bearer"})
        tok = q.get("token") if route == "videos/stream" else headers.get("Authorization", "")[len("Bearer "):]
        if tok not in self.valid:
            return _resp(401, {"detail": "Invalid or expired token"})
        return getattr(self, "_" + route.replace("/", "_"))(q, payload, headers)

    def _videos_explore(self, q, payload, headers):
        rows = [c for c in self.chunks if q.get("location") in (None, c.get("location"))]
        off, lim = int(q["offset"]), int(q["limit"])
        return _resp(200, {"chunks": rows[off:off + lim], "total": len(rows), "scope": q["scope"]})

    def _tools_segments(self, q, payload, headers):
        segs = self.segs.get(q["original_video"], [])
        return _resp(200, {"original_video": q["original_video"], "total": len(segs), "segments": segs})

    def _videos_detections(self, q, payload, headers):
        side = self.sidecars.get(q["source"])
        return _resp(404, {"detail": "No detection sidecar for segment"}) if side is None else _resp(200, side)

    def _videos_metadata(self, q, payload, headers):
        for segs in self.segs.values():
            for s in segs:
                if s["source"] == q["source"]:
                    return _resp(200, dict(s, embedding=[0.1] * 4))
        return _resp(404, {"detail": "Video not found or you don't have permission to access it"})

    def _dashboard_stats(self, q, payload, headers):
        return _resp(200, {"scope": q["scope"], "overview": {"total_rows": 3}, "table_available": True})

    def _search(self, q, payload, headers):
        if not 1 <= payload["top_k"] <= 100 or payload.get("llm_top_n", 3) < 1:
            return _resp(422, {"detail": [{"msg": "bad"}]})
        cam = payload["metadata_filters"].get("camera_id")
        res = [r for r in self.results if cam in (None, r.get("camera_id"))][:payload["top_k"]]
        return _resp(200, {"results": res, "chunk_results": [], "total": len(res), "embedding_time_ms": 12.5,
                           "search_time_ms": 80.0, "sql_query": "SELECT ... vector [0.1, 0.2]",
                           "llm_synthesis": {"response": "x", "processing_time": 1.5, "error": None}})

    def _videos_stream(self, q, payload, headers):
        data = self.videos.get(q["source"])
        if data is None:
            return _resp(404, {"detail": "Object not found"})
        rng = headers.get("Range")
        start = int(rng[len("bytes="):].rstrip("-")) if rng and not self.ignore_range else 0
        if start >= len(data) and start:
            return _resp(416, headers={"Content-Range": f"bytes */{len(data)}"})
        body = data[start:]
        hdr = {"Content-Length": str(len(body))}
        if start:
            hdr["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        if self.drop_at is not None:
            cut, self.drop_at = self.drop_at, None
            return vss.Response(206 if start else 200, {k.lower(): v for k, v in hdr.items()}, _Dropping(body, cut))
        return _resp(206 if start else 200, headers=hdr, raw=body)


def _chunk(i, cam=CAM, segs=3, loc="neighborhood", public=True):
    name = f"20261001_0723{i:02d}_{loc}_20260901_chunk_{i:04d}"
    return {"original_video": f"s3://chunks/team/{name}.mp4", "filename": name + ".mp4", "camera_id": cam,
            "location": loc, "total_segments": segs, "is_public": public,
            "upload_timestamp": f"2026-10-01T07:23:{i % 60:02d}"}


def _seg(chunk, n, total, sec=5.0):
    name = vss.stem(chunk["original_video"])
    return {"source": f"s3://segs/segments/{name}_segment_{n:03d}_of_{total:03d}.mp4", "segment_number": n,
            "total_segments": total, "segment_start_sec": (n - 1) * sec, "segment_end_sec": n * sec,
            "original_video": chunk["original_video"], "camera_id": chunk["camera_id"],
            "object_counts": json.dumps({"car": n}), "object_classes": "car", "reasoning_content": f"seg {n}",
            "embedding": [0.0, 1.0]}


@pytest.fixture
def world():
    chunks = [_chunk(i) for i in range(3)] + [_chunk(10, cam="ptz_parking", segs=2, loc="parking")]
    segs = {c["original_video"]: [_seg(c, n, c["total_segments"]) for n in range(c["total_segments"], 0, -1)]
            for c in chunks}                                                    # reversed on purpose
    sidecars = {s["source"]: {"source": "yolo11_coco", "segment_source": s["source"], "fps": 20.0,
                              "frames": [{"frame_index": 0, "time_sec": 0.0, "detections": []}]}
                for ss in segs.values() for s in ss}
    missing = segs[chunks[1]["original_video"]][0]["source"]
    del sidecars[missing]                                                       # one segment has no sidecar
    videos = {c["original_video"]: bytes(range(256)) * (40 + i) for i, c in enumerate(chunks)}
    results = [dict(_seg(chunks[0], 2, 3), similarity_score=0.31),
               dict(_seg(chunks[1], 1, 3), similarity_score=0.52),
               dict(_seg(chunks[3], 1, 2), similarity_score=0.9)]
    fake = FakeVSS(chunks, segs, sidecars, videos, results)
    c = vss.Client(vss.Config(URL, USER, PW, "test"), transport=fake, sleep=lambda s: None)
    old = vss.set_client(c)
    try:
        yield fake, c
    finally:
        vss.set_client(old)


# ---------------------------------------------------------------- config

def _cfg_file(tmp_path, text):
    (tmp_path / "team-zz.config").write_text(text, encoding="utf-8")
    return str(tmp_path / "*.config")


def test_config_order(tmp_path):
    pat = _cfg_file(tmp_path, f"# team\nexport INGRESS_URL=\"{URL}/\"\nUSERNAME='{USER}'\nPASSWORD={PW}\nX=1=2\n")
    env_vss = {"VSS_URL": "https://a.test", "VSS_USERNAME": "u", "VSS_PASSWORD": "p"}
    env_ing = {"INGRESS_URL": "https://b.test", "USERNAME": "lin", "PASSWORD": "q"}
    assert vss.load_config(dict(env_vss, **env_ing), "linux", pat).host == "a.test"
    assert vss.load_config(env_ing, "linux", pat).origin == "env:INGRESS_URL"
    cfg = vss.load_config(env_ing, "win32", pat)                 # Windows: env USERNAME is the login name
    assert (cfg.url, cfg.username, cfg.password, cfg.origin) == (URL + "/", USER, PW, "file:team-zz.config")
    part = {"INGRESS_URL": "https://b.test", "PASSWORD": "q"}    # incomplete env sets are never mixed in
    assert vss.load_config(part, "linux", pat).origin.startswith("file:")
    assert vss.parse_config_text("X=1=2\n#C=3\nexport  Y = 'v' \n") == {"X": "1=2", "Y": "v"}


def test_config_errors_name_no_values(tmp_path):
    with pytest.raises(vss.VSSConfigError, match="found 0"):
        vss.load_config({}, "linux", str(tmp_path / "*.config"))
    (tmp_path / "a.config").write_text(f"PASSWORD={PW}\n")
    with pytest.raises(vss.VSSConfigError) as e:
        vss.load_config({}, "linux", str(tmp_path / "*.config"))
    assert PW not in str(e.value) and "lacks" in str(e.value)
    (tmp_path / "b.config").write_text("")
    with pytest.raises(vss.VSSConfigError, match="found 2"):
        vss.load_config({}, "linux", str(tmp_path / "*.config"))


# ---------------------------------------------------------------- auth, retries, redaction

def test_login_bearer_and_relogin_on_401(world):
    fake, c = world
    c.login()
    login = fake.routes("auth/login")[0]
    assert login["method"] == "POST" and login["body"] == {"username": USER, "password": PW}
    assert c.stats("mine")["overview"]["total_rows"] == 3
    assert fake.routes("dashboard/stats")[0]["headers"]["Authorization"] == "Bearer jwt.SECRET-1"
    fake.expire()
    assert c.stats()["scope"] == "all"
    assert c.logins == 2 and fake.routes("dashboard/stats")[-1]["headers"]["Authorization"] == "Bearer jwt.SECRET-2"


def test_bad_login_raises_without_secret(world):
    fake, _ = world
    c = vss.Client(vss.Config(URL, USER, "wrong-" + PW, "test"), transport=fake, sleep=lambda s: None)
    with pytest.raises(vss.VSSAuthError) as e:
        c.explore()
    assert "401" in str(e.value) and PW not in str(e.value) and len(fake.routes("auth/login")) == 1


def test_retries_then_gives_up(world):
    fake, c = world
    fake.fail["dashboard/stats"] = [503, ConnectionResetError("reset"), 502]
    assert c.stats()["scope"] == "all"                                   # 3 retries are enough
    fake.fail["dashboard/stats"] = [503, 503, 503, 503]
    with pytest.raises(vss.VSSError, match="gave up after 4 attempts"):
        c.stats()
    with pytest.raises(vss.VSSError, match="422"):                       # other 4xx: no retry
        c.call("POST", "search", payload={"query": "x", "top_k": 0, "metadata_filters": {}})
    assert len(fake.routes("search")) == 1


def test_404_is_none(world, tmp_path):
    assert vss.detections("s3://segs/segments/nope.mp4") is None
    assert vss.segment_meta("s3://segs/segments/nope.mp4") is None
    assert vss.segments("s3://chunks/nope.mp4") == []
    assert vss.stream("s3://chunks/nope.mp4", tmp_path / "nope.mp4") is None
    assert not (tmp_path / "nope.mp4").exists()


def test_secrets_never_in_repr_or_logs(world, tmp_path, caplog, capsys):
    fake, c = world
    caplog.set_level(logging.DEBUG, logger="stm.vss")
    c.login()
    c.explore(camera_id=CAM)
    c.search("a car", camera_id=CAM)
    fake.expire()
    c.stream(fake.chunks[0]["original_video"], tmp_path / "a.mp4")
    fake.fail["videos/explore"] = [400]
    with pytest.raises(vss.VSSError) as e:
        c.explore()
    assert vss.main(["cams", "-v"]) == 0
    seen = " ".join([caplog.text, repr(c), str(c), repr(c.config), str(c.config), str(e.value),
                     *capsys.readouterr()])
    tokens = [f"jwt.SECRET-{i}" for i in range(1, fake.n + 1)]
    assert fake.n >= 2 and c._token in tokens
    for secret in [PW, USER, *tokens, "token=jwt", "&token=", "?token="]:
        assert secret not in seen, secret
    assert "videos/explore" in caplog.text and "<redacted>" in repr(c)


def test_repr_blanks_username_inside_host(tmp_path):
    pat = _cfg_file(tmp_path, f"INGRESS_URL=https://lab-{USER}.example.test\nUSERNAME={USER}\nPASSWORD={PW}\n")
    cfg = vss.load_config({}, "win32", pat)
    c = vss.Client(cfg, transport=lambda *a: None)
    for text in (repr(cfg), str(cfg), repr(c), str(c)):
        assert USER not in text and PW not in text and "lab-***.example.test" in text


def test_scrub_covers_escaped_forms():
    c = vss.Client(vss.Config(URL, "u-1", 'p"w\\x y', "t"), transport=lambda *a: None)
    c._token = "jwt.T"
    text = 'bad {"input": "p\\"w\\\\x y"} p%22w%5Cx%20y p"w\\x y Bearer jwt.T'
    out = c._scrub(text)
    assert "jwt.T" not in out and 'p"w\\x y' not in out and "p%22w" not in out and 'p\\"w' not in out


# ---------------------------------------------------------------- API shapes

def test_explore_pages_and_filters_client_side(world):
    fake, c = world
    fake.chunks += [_chunk(20 + i, cam="busy_cam", segs=1, loc="street") for i in range(230)]
    allc = vss.explore()
    assert len(allc) == 234 and [x["original_video"] for x in allc] == sorted(x["original_video"] for x in allc)
    pages = fake.routes("videos/explore")
    assert [p["q"]["offset"] for p in pages] == ["0", "100", "200"] and all(p["q"]["limit"] == "100" for p in pages)
    mine = vss.explore(camera_id=CAM, scope="mine")
    assert len(mine) == 3 and {x["camera_id"] for x in mine} == {CAM}
    last = fake.routes("videos/explore")[-1]["q"]
    assert last["scope"] == "mine" and "camera_id" not in last                # server ignores camera_id
    assert len(vss.explore(location="parking")) == 1 and fake.routes("videos/explore")[-1]["q"]["location"] == "parking"
    with pytest.raises(ValueError):
        vss.explore(scope="everyone")


def test_cameras_counts(world):
    cams = {r["camera_id"]: r for r in vss.cameras()}
    assert cams[CAM]["chunks"] == 3 and cams[CAM]["segments"] == 9 and cams[CAM]["locations"] == ["neighborhood"]
    assert cams["ptz_parking"]["segments"] == 2 and cams[CAM]["first_upload"] < cams[CAM]["last_upload"]


def test_segments_sorted_and_counts_parsed(world):
    fake, _ = world
    src = fake.chunks[0]["original_video"]
    fake.segs[src][0]["object_counts"] = "not json"
    fake.segs[src][1]["object_counts"] = {"person": "2", "bus": None}
    rows = vss.segments(src)
    assert [r["segment_number"] for r in rows] == [1, 2, 3]
    assert rows[0]["object_counts"] == {"car": 1} and rows[1]["object_counts"] == {"person": 2}
    assert rows[2]["object_counts"] == {} and all("embedding" not in r for r in rows)
    meta = vss.segment_meta(rows[0]["source"])
    assert meta["object_counts"] == {"car": 1} and "embedding" not in meta
    assert vss.detections(rows[0]["source"])["source"] == "yolo11_coco"


def test_search_body_and_hits(world):
    fake, _ = world
    hits = vss.search("a car parked in a driveway", camera_id=CAM, top_k=10)
    body = fake.routes("search")[-1]["body"]
    assert body["metadata_filters"] == {"camera_id": CAM} and body["llm_top_n"] >= 1
    assert body["top_k"] == 10 and body["min_similarity"] == 0.25
    assert [h["similarity_score"] for h in hits] == [0.52, 0.31]
    assert set(hits[0]) >= {"source", "similarity_score", "reasoning_content", "original_video", "start_sec",
                            "end_sec"}
    assert (hits[0]["start_sec"], hits[0]["end_sec"]) == (0.0, 5.0)
    assert len(vss.search("cars")) == 3 and fake.routes("search")[-1]["body"]["metadata_filters"] == {}
    raw = vss.client().search_raw("cars", location="parking")
    assert "sql_query" not in raw and fake.routes("search")[-1]["body"]["metadata_filters"] == {"location": "parking"}
    for bad in ({"top_k": 0}, {"top_k": 101}, {"min_similarity": 1.5}, {"llm_top_n": 0}):
        with pytest.raises(ValueError):
            vss.search("cars", **bad)


def test_search_drops_other_camera_hits(world):
    fake, _ = world
    fake.results.append(dict(_seg(fake.chunks[3], 2, 2), camera_id="ptz_parking", similarity_score=0.99))
    fake._search = lambda q, p, h: _resp(200, {"results": fake.results})   # a server that ignored the filter
    assert {h["camera_id"] for h in vss.search("x", camera_id=CAM)} == {CAM}


# ---------------------------------------------------------------- stream

def test_stream_fresh_resume_and_restart(world, tmp_path):
    fake, c = world
    src = fake.chunks[0]["original_video"]
    data = fake.videos[src]
    assert c.stream(src, tmp_path / "a.mp4") == len(data) and (tmp_path / "a.mp4").read_bytes() == data
    call = fake.routes("videos/stream")[-1]
    assert call["q"]["source"] == src and call["q"]["token"].startswith("jwt.") and "Authorization" not in call["headers"]
    n = len(fake.routes("videos/stream"))
    assert c.stream(src, tmp_path / "a.mp4") == len(data) and len(fake.routes("videos/stream")) == n   # kept

    (tmp_path / "b.mp4.part").write_bytes(data[:1000])                   # resume an interrupted download
    assert c.stream(src, tmp_path / "b.mp4") == len(data) and (tmp_path / "b.mp4").read_bytes() == data
    assert fake.routes("videos/stream")[-1]["headers"]["Range"] == "bytes=1000-"
    assert not (tmp_path / "b.mp4.part").exists()

    fake.ignore_range = True                                             # server answers 200: rewrite
    (tmp_path / "c.mp4.part").write_bytes(b"x" * 500)
    assert c.stream(src, tmp_path / "c.mp4") == len(data) and (tmp_path / "c.mp4").read_bytes() == data
    fake.ignore_range = False

    (tmp_path / "d.mp4.part").write_bytes(data)                          # complete part: 416 -> finished
    assert c.stream(src, tmp_path / "d.mp4") == len(data) and (tmp_path / "d.mp4").read_bytes() == data


def test_stream_resumes_after_drop_and_relogin(world, tmp_path):
    fake, c = world
    src = fake.chunks[2]["original_video"]
    data = fake.videos[src]
    c.login()
    fake.expire()                                                        # first try: 401 -> re-login
    fake.drop_at = 3000                                                  # second: drops after 3000 bytes
    assert c.stream(src, tmp_path / "e.mp4") == len(data) and (tmp_path / "e.mp4").read_bytes() == data
    calls = fake.routes("videos/stream")
    assert len(calls) == 3 and calls[-1]["headers"]["Range"] == "bytes=3000-" and c.logins == 2


# ---------------------------------------------------------------- pull + CLI

def test_pull_writes_packd_shapes(world, tmp_path):
    fake, _ = world
    out = vss.pull(CAM, tmp_path / "p", video=False)
    assert (out["chunks"], out["segments"], out["expected_segments"]) == (3, 9, 9)
    assert (out["sidecars"], out["missing_sidecars"], out["videos"]) == (8, 1, 0)
    assert not fake.routes("videos/stream") and not (tmp_path / "p" / "chunks").exists()
    seg = json.loads((tmp_path / "p" / "segments.json").read_text())
    assert seg["camera_id"] == CAM and len(seg["chunks"]) == 3 and len(seg["segments"]) == 9
    row = seg["segments"][0]
    assert json.loads(row["object_counts"]) == {"car": 1}               # string, as stm.captions reads it
    assert row["chunk"] == vss.stem(row["original_video"]) and row["has_sidecar"] is True
    side_files = sorted((tmp_path / "p" / "sidecars").glob("*.json"))
    assert len(side_files) == 8
    side = json.loads(side_files[0].read_text())
    assert side["segment_start_sec"] == 0.0 and side["segment_end_sec"] == 5.0 and side["chunk"] == row["chunk"]
    n_det = len(fake.routes("videos/detections"))
    again = vss.pull(CAM, tmp_path / "p", video=True, workers=2)        # rerun: only the missing one refetched
    assert again["sidecars_fetched"] == 0 and len(fake.routes("videos/detections")) == n_det + 1
    assert again["videos"] == 3 and len(list((tmp_path / "p" / "chunks").glob("*.mp4"))) == 3


def test_cli_cams_search_pull(world, tmp_path, capsys):
    fake, _ = world
    assert vss.main(["cams"]) == 0
    out = capsys.readouterr().out
    assert CAM in out and "2 cameras, 4 chunks, 11 segments" in out
    assert vss.main(["search", CAM, "a car", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert [h["similarity_score"] for h in res["hits"]] == [0.52, 0.31] and res["server"]["search_time_ms"] == 80.0
    assert vss.main(["pull", "ptz_parking", "--dest", str(tmp_path / "q"), "--no-video", "--scope", "mine"]) == 0
    assert "2 segments" in capsys.readouterr().out and fake.routes("videos/explore")[-1]["q"]["scope"] == "mine"
    fake.fail["videos/explore"] = [400]
    assert vss.main(["cams"]) == 2 and "error: VSS GET videos/explore -> 400" in capsys.readouterr().err
    assert vss.main(["pull", "nobody_cam", "--dest", str(tmp_path / "none"), "--no-video"]) == 2
    assert "no chunks for camera 'nobody_cam'" in capsys.readouterr().err and not (tmp_path / "none").exists()
