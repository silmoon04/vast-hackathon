"""Tests for stm.export: run on the real out/ outputs (whatever exists now) plus a tiny synthetic tree."""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from stm import data, export

TAG_RE = re.compile(r"<script>window\.__STM__=(.*?);window\.__STM_FILES__=(.*?)</script>\n", re.S)


def _parse_page(page: str) -> tuple[dict, dict, re.Match]:
    m = TAG_RE.search(page)
    assert m, "data script not injected"
    un = lambda s: json.loads(s.replace("<\\!--", "<!--"))
    return un(m.group(1)), un(m.group(2)), m


def _decode(uri: str) -> np.ndarray:
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:3] == b"\xff\xd8\xff", "not a JPEG"
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    return img


# ---------- pure helpers ----------

def test_image_strings_walks_everything():
    obj = {"a": "crops/1.jpg", "b": [{"c": "near/2_3.4.PNG"}, ["x.jpg", "x.jpg"]], "d": 3, "e": "notes.txt",
           "f": None, "g": {"h": {"i": "frames/busiest.jpeg"}}}
    assert export.image_strings(obj) == ["crops/1.jpg", "near/2_3.4.PNG", "x.jpg", "frames/busiest.jpeg"]


def test_url_for_rejects_unsafe_paths():
    assert export._url_for("out/ptz_parking", "crops/1.jpg") == "out/ptz_parking/crops/1.jpg"
    assert export._url_for("out/ptz_parking", "./crops/1.jpg") == "out/ptz_parking/crops/1.jpg"
    for bad in ["../secret.jpg", "C:/x/y.jpg", "/abs.jpg", "https://e.com/a.jpg", "data:image/jpeg;base64,xx.jpg"]:
        assert export._url_for("out/ptz_parking", bad) is None


def test_inject_before_main_script_and_escapes():
    html = '<title>t</title>\n<script src="lib.js"></script>\n<p>x</p>\n<script>\nconst a=1;\n</script>\n'
    page = export.inject(html, {"api/cams": ["c"], "out/c/meta.json": {"s": "</script><!--"}}, {"out/c/a.jpg": "data:x"})
    stm, files, m = _parse_page(page)
    assert stm["out/c/meta.json"]["s"] == "</script><!--"
    assert files == {"out/c/a.jpg": "data:x"}
    assert page.index(m.group(0)) < page.index("<script>\nconst a=1;")
    assert page.index('<script src="lib.js">') < page.index(m.group(0))
    assert page.replace(m.group(0), "") == html  # original page untouched
    assert page.count("</script>") == html.count("</script>") + 1  # no raw </script> leaked from data


def test_tiers_and_drop_order():
    assert export.tier_of("out/ptz_parking/background_A.jpg") == "plate"
    assert export.tier_of("out/ptz_parking/flow.jpg") == "plate"
    assert export.tier_of("out/ptz_parking/frames/busiest.jpg") == "plate"
    assert export.tier_of("out/ptz_parking/crops/1.jpg") == "crop"
    assert export.tier_of("out/ptz_parking/moments/3.jpg") == "crop"
    assert export.tier_of("out/captions/frames/a.jpg") == "frame"
    r = export.drop_rank
    assert r("out/yonge_dundas/crops/1.jpg") < r("out/car_detection/crops/1.jpg") < r("out/ptz_parking/crops/1.jpg")
    assert r("out/ptz_parking/background.jpg") >= 99


# ---------- real outputs ----------

@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not export.available_cams():
        pytest.skip("no outputs under out/ yet")
    dist = tmp_path_factory.mktemp("dist")
    # other modules may be writing out/ right now: snapshot what existed before the build
    t0 = time.time() - 1.0
    before = {f"out/{c}/{n}.json" for c in export.available_cams() for n in export.CAM_JSONS
              if (data.OUT / c / f"{n}.json").exists()}
    ui = dist.parent / (dist.name + "_ui.html")  # UI owners edit ui/index.html live: snapshot it
    ui.write_bytes(export.UI.read_bytes())
    rep = export.build(dist=dist, ui=ui, quiet=True)
    rep["_before"], rep["_t0"], rep["_ui"] = before, t0, ui
    page = (dist / "index.html").read_bytes().decode("utf-8")  # keep CRLF if the UI has it
    return rep, dist, page


def test_real_page_contents(real):
    rep, dist, page = real
    stm, files, m = _parse_page(page)
    ui = rep["_ui"].read_bytes().decode("utf-8")
    assert page.replace(m.group(0), "") == ui
    main = re.search(r"<script>\s*const S =", page)
    assert main and page.index(m.group(0)) < main.start()

    entries = stm["api/cams"]
    cams = [e["camera"] for e in entries]
    assert cams == rep["cams"] and cams
    assert all(c in data.CAMS or (data.OUT / c / "meta.json").is_file() for c in cams)
    assert cams == export.demo_order(cams)  # event cam first, then ptz_parking, yonge_dundas
    assert all(set(e) >= {"camera", "title", "days", "simulated"} for e in entries)
    if (data.OUT / "neighborhood_cam-1" / "meta.json").is_file():
        assert cams[0] == "neighborhood_cam-1"
    assert any("redacted_people" in line for line in rep["log"])
    # every UI JSON that existed before the build is inlined, under exactly the URL the UI fetches
    skipped = " ".join(rep["log"])
    for url in rep["_before"]:
        assert url in stm or url in skipped, url
    for url in stm:
        assert url == "api/cams" or (export.ROOT / url).exists()
    if "out/ptz_parking/meta.json" in stm:
        disk = json.loads((data.OUT / "ptz_parking" / "meta.json").read_text(encoding="utf-8"))
        assert stm["out/ptz_parking/meta.json"]["camera"] == disk["camera"] == "ptz_parking"

    # every image reference in every inlined JSON resolves to an inlined (or logged-dropped) file
    for url, obj in stm.items():
        if url == "api/cams":
            continue
        folder = url.rsplit("/", 1)[0]
        for s in export.image_strings(obj):
            u = export._url_for(folder, s)
            src = export.ROOT / u if u else None
            if src is not None and src.is_file() and src.stat().st_mtime < rep["_t0"]:
                assert u in files or u in rep["dropped"], u
    for cam in cams:
        if (data.OUT / cam / "background.jpg").exists():
            assert f"out/{cam}/background.jpg" in files

    # data URIs are JPEGs within the size tier for their kind
    assert len(files) == rep["n_images_inlined"] > 0
    for url, uri in files.items():
        assert url.startswith("out/") and (export.ROOT / url).is_file()
        img = _decode(uri)
        assert max(img.shape[:2]) <= export.TIERS[export.tier_of(url)][0], url

    assert rep["index_html_bytes"] == (dist / "index.html").stat().st_size <= export.MAX_BYTES


def test_real_videos_copied(real):
    rep, dist, _ = real
    for cam in export.available_cams():
        src = data.OUT / cam / "synopsis.mp4"
        if src.exists():
            dst = dist / "out" / cam / "synopsis.mp4"
            assert dst.exists() and dst.stat().st_size > 0
    assert len(rep["videos"]) == sum((data.OUT / c / "synopsis.mp4").exists() for c in export.available_cams())


def test_real_over_budget_drops_least_important_first(real, tmp_path):
    rep, _, _ = real
    budget = rep["index_html_bytes"] - 1  # force at least one drop
    r2 = export.build(dist=tmp_path, max_bytes=budget, quiet=True)
    assert r2["n_images_dropped"] >= 1
    assert r2["index_html_bytes"] <= budget
    _, files, _ = _parse_page((tmp_path / "index.html").read_text(encoding="utf-8"))
    worst_dropped = max(export.drop_rank(u) for u in r2["dropped"])
    assert all(export.drop_rank(u) >= worst_dropped for u in files)
    assert any("dropped" in line for line in r2["log"])
    for u in r2["dropped"]:
        assert (tmp_path / u).is_file() and u not in files


# ---------- synthetic tree: drop order across cams ----------

def test_synthetic_drop_order(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    out = root / "out"
    rng = np.random.default_rng(0)

    def jpg(p: Path, w: int, h: int):
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))

    for cam in ("ptz_parking", "car_detection", "yonge_dundas"):
        jpg(out / cam / "background.jpg", 1280, 720)
        jpg(out / cam / "crops" / "1.jpg", 300, 200)
        (out / cam / "meta.json").write_text(json.dumps({"camera": cam, "background": "background.jpg"}))
        (out / cam / "near.json").write_text(json.dumps({"targets": [{"crop": "crops/1.jpg", "bad": "../x.jpg"}]}))
    (out / "ptz_parking" / "ledger.json").write_text(json.dumps({"changes": [{"before": "crops/404.jpg"}]}))
    (out / "ptz_parking" / "broken.json").write_text("{not json")
    ui = tmp_path / "ui.html"
    ui.write_text("<p>hi</p>\n<script>\ninit();\n</script>\n")
    monkeypatch.setattr(export, "ROOT", root)
    monkeypatch.setattr(export, "OUT", out)

    full = export.build(dist=tmp_path / "d0", ui=ui, quiet=True)
    assert full["cams"] == ["ptz_parking", "yonge_dundas", "car_detection"]  # demo order
    assert full["n_images_inlined"] == 6 and full["n_images_dropped"] == 0
    # a reference to a file that is not on disk is counted and logged, never silently lost
    assert full["missing"] == ["out/ptz_parking/crops/404.jpg"] and full["n_images_missing"] == 1
    assert any("not on disk" in line for line in full["log"]) and any("ledger.json" in line for line in full["log"])
    stm, files, _ = _parse_page((tmp_path / "d0" / "index.html").read_text(encoding="utf-8"))
    assert set(stm) == ({"api/cams", "out/ptz_parking/ledger.json"}
                        | {f"out/{c}/{n}.json" for c in full["cams"] for n in ("meta", "near")})
    assert "out/ptz_parking/crops/404.jpg" not in files
    big = _decode(files["out/ptz_parking/background.jpg"])
    small = _decode(files["out/ptz_parking/crops/1.jpg"])
    assert max(big.shape[:2]) == 960 and max(small.shape[:2]) == 220

    crop_cost = {u: len(u) + len(v) + 6 for u, v in files.items()}
    # budget that requires dropping exactly the yonge_dundas crop
    budget = full["index_html_bytes"] - crop_cost["out/yonge_dundas/crops/1.jpg"] + 10
    r = export.build(dist=tmp_path / "d1", ui=ui, max_bytes=budget, quiet=True)
    assert r["dropped"] == ["out/yonge_dundas/crops/1.jpg"]
    assert (tmp_path / "d1" / "out" / "yonge_dundas" / "crops" / "1.jpg").is_file()
    # rebuilding with room to spare inlines it again and removes the stale dropped file
    r = export.build(dist=tmp_path / "d1", ui=ui, quiet=True)
    assert r["dropped"] == [] and not (tmp_path / "d1" / "out" / "yonge_dundas" / "crops" / "1.jpg").exists()
    # tighter: yonge crop, then the other dev cam's crop, before any main-cam image
    budget2 = budget - crop_cost["out/car_detection/crops/1.jpg"]
    r = export.build(dist=tmp_path / "d2", ui=ui, max_bytes=budget2, quiet=True)
    assert r["dropped"][:2] == ["out/yonge_dundas/crops/1.jpg", "out/car_detection/crops/1.jpg"]
    assert "out/ptz_parking/background.jpg" not in r["dropped"]
    # impossible budget: main-cam plates are never dropped, and the overrun is logged
    r = export.build(dist=tmp_path / "d3", ui=ui, max_bytes=1000, quiet=True)
    assert "out/ptz_parking/background.jpg" not in r["dropped"]
    assert any("WARNING" in line for line in r["log"])


# ---------- event cam first, people blurred, per-cam crop cap ----------

def _lap(img: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_demo_order():
    assert export.demo_order(["car_detection", "yonge_dundas", "ptz_parking", "neighborhood_cam-1"]) == \
        ["neighborhood_cam-1", "ptz_parking", "yonge_dundas", "car_detection"]


def test_event_cam_first_people_blurred_and_cap(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    out = root / "out"
    rng = np.random.default_rng(1)

    def jpg(p: Path, w: int, h: int):
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))

    pd_cam = "neighborhood_cam-1"  # not in data.CAMS: found by its meta.json
    for cam in ("ptz_parking", pd_cam):
        jpg(out / cam / "background.jpg", 1280, 720)
        for i in (1, 2, 3, 4, 5):
            jpg(out / cam / "crops" / f"{i}.jpg", 120, 200)
        jpg(out / cam / "near" / "2_10.0.jpg", 120, 200)
        (out / cam / "meta.json").write_text(json.dumps(
            {"camera": cam, "title": "t", "days": [{"id": "20260901", "label": "1 Sep"}], "simulated": []}))
        (out / cam / "near.json").write_text(json.dumps({"targets": [
            {"object_id": 1, "name": "car", "crop": "crops/1.jpg",
             "visitors": [{"object_id": 2, "name": "person", "crop": "near/2_10.0.jpg"}]}]}))
        (out / cam / "index.json").write_text(json.dumps({"objects": [
            {"object_id": 3, "name": "person", "crop": "crops/3.jpg"},
            {"object_id": 4, "name": "car", "crop": "crops/4.jpg"}]}))
        (out / cam / "synopsis.json").write_text(json.dumps({"ghosts": [
            {"object_id": 5, "name": "bicycle"}, {"object_id": 2, "name": "person"}]}))
        (out / cam / "extra_view.json").write_text(json.dumps({"x": 1}))
    ui = tmp_path / "ui.html"
    ui.write_text("<p>hi</p>\n<script>\ngetJSON(`out/${cam}/extra_view.json`);\n</script>\n")
    monkeypatch.setattr(export, "ROOT", root)
    monkeypatch.setattr(export, "OUT", out)

    r = export.build(dist=tmp_path / "d", ui=ui, quiet=True)
    assert r["cams"] == [pd_cam, "ptz_parking"]
    stm, files, _ = _parse_page((tmp_path / "d" / "index.html").read_text(encoding="utf-8"))
    assert [e["camera"] for e in stm["api/cams"]] == [pd_cam, "ptz_parking"]
    assert stm["api/cams"][0]["days"][0]["label"] == "1 Sep" and stm["api/cams"][0]["simulated"] == []
    assert f"out/{pd_cam}/extra_view.json" in stm  # a JSON the UI names is inlined without editing export
    # people: person visitor crop (near), person index crop, bicycle ghost crop, person ghost crop crops/2
    want = {f"out/{c}/{p}" for c in (pd_cam, "ptz_parking")
            for p in ("near/2_10.0.jpg", "crops/2.jpg", "crops/3.jpg", "crops/5.jpg")}
    assert set(r["people"]) == want and r["redacted_people"] == len(want)
    assert any(line.startswith(f"redacted_people: {len(want)}") for line in r["log"])
    sharp = _lap(_decode(files[f"out/{pd_cam}/crops/4.jpg"]))  # a car: not blurred
    assert sharp > 1000
    for u in want:
        assert _lap(_decode(files[u])) < 0.05 * sharp, u
    # both demo cams' plates are protected
    assert export.drop_rank(f"out/{pd_cam}/background.jpg", set(r["protected"])) >= 99
    assert export.drop_rank("out/ptz_parking/background.jpg", set(r["protected"])) >= 99

    # cap: 3 crops per cam -> ghost-only crops (2, 5) go first, then index-only; the near ones stay
    r = export.build(dist=tmp_path / "d2", ui=ui, quiet=True, crop_cap=3)
    assert r["capped"] == {pd_cam: 3, "ptz_parking": 3}
    _, files, _ = _parse_page((tmp_path / "d2" / "index.html").read_text(encoding="utf-8"))
    for c in (pd_cam, "ptz_parking"):
        for gone in ("crops/2.jpg", "crops/5.jpg"):
            assert f"out/{c}/{gone}" in r["dropped"] and f"out/{c}/{gone}" not in files
        assert f"out/{c}/near/2_10.0.jpg" in files and f"out/{c}/crops/1.jpg" in files
        assert (tmp_path / "d2" / "out" / c / "crops" / "5.jpg").is_file()  # still reachable as a file
    assert any(line.startswith(f"capped {pd_cam}") for line in r["log"])


# ---------- scripts/freeze_demo.py ----------

def _freeze():
    import importlib.util
    spec = importlib.util.spec_from_file_location("freeze_demo", export.ROOT / "scripts" / "freeze_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_freeze_helpers():
    f = _freeze()
    assert f.text_counts("2 vehicles left and none arrived between Day A and Day B") == {"left": 2, "arrived": 0}
    assert f.text_counts("2 left, 16 the same car, 2 a different car") == {"left": 2, "same": 16}
    assert f.text_blind_totals(["The camera could not see for 1 min 38 s in total, in 2 stretches"])[0][1] == 98.0
    assert f.text_blind_totals(["camera was blind for 2.5 s, so who was near it then is UNKNOWN"]) == []
    assert f.mmss(155.0) == "02:35" and f.mmss(59.6) == "01:00" and f.mmss(-0.0) == "00:00"
    assert f.span_union([{"start": 0, "end": 2}, {"start": 1, "end": 3}, {"start": 5, "end": 6}]) == 4
    m = f.merge_spans([{"start": 156, "end": 157, "reason": "blocked"}, {"start": 157, "end": 237, "reason": "blur"}])
    assert len(m) == 1 and m[0]["end"] == 237


def test_freeze_gate_catches_disagreements(tmp_path, monkeypatch):
    f = _freeze()
    cam = "neighborhood_cam-1"
    d = tmp_path / cam
    d.mkdir()
    w = lambda n, o: (d / f"{n}.json").write_text(json.dumps(o))
    w("meta", {"camera": cam, "days": [{"id": "20260901", "label": "1 Sep"}, {"id": "20260902", "label": "2 Sep"}],
               "simulated": []})
    w("ledger", {"blind": [{"start": 10, "end": 40, "reason": "dark"}], "blind_total_s": 30.0})
    w("diff", {"counts": {"same": 5, "arrived": 1, "left": 2, "moved": 0, "swapped": 0}, "alignment": {"ok": True}})
    w("brief", {"headline": "3 vehicles left and 1 arrived between Day A and Day B; the camera could not see for 31 s.",
                "honesty": ["Seen in London."], "bullets": []})
    w("qa", {"items": [{"question": "What changed between 1 Sep and 2 Sep?", "t": None,
                        "answer": {"type": "what_changed", "answer": "UNKNOWN", "sentence": "x"}}]})
    w("index", {"objects": [{"object_id": 9, "name": "person", "colour": "red"}]})
    monkeypatch.setattr(f, "OUT", tmp_path)
    g = f.Gate()
    f.check_numbers(g, cam)
    fails = " | ".join(m for lv, _, m in g.rows if lv == "FAIL")
    for needle in ("blind total 31", "brief vs diff.counts", "says UNKNOWN", "Day A/B", "names a city", "a colour"):
        assert needle in fails, needle
    g2 = f.Gate()
    f.check_numbers(g2, cam, strict=False)
    assert g2.count("FAIL") == 0 and g2.count("WARN") >= 6
    card = f.cue_for_cam(cam, {})
    assert any("1 Sep" in line and "2 Sep" in line for line in card) and not any("SIMULATED" in x for x in card)


def test_freeze_server_check_with_stub(tmp_path, monkeypatch):
    """check_server on a local stub: p50 measured, a live answer that differs from qa.json is flagged."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps([] if self.path.startswith("/api/cams") else {"answer": "NO", "latency_ms": 1.0}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        f = _freeze()
        (tmp_path / "c").mkdir()
        (tmp_path / "c" / "qa.json").write_text(json.dumps({"items": [
            {"question": "Is it there?", "t": 3.0, "answer": {"answer": "YES"}},
            {"question": "Who came near?", "t": None, "answer": {"answer": "NO"}}]}))
        monkeypatch.setattr(f, "OUT", tmp_path)
        g = f.Gate()
        live = f.check_server(g, f"http://127.0.0.1:{srv.server_address[1]}", ["c"])
        assert set(live["c"]) == {"Is it there?", "Who came near?"}
        assert any(lv == "OK" and "p50" in m for lv, _, m in g.rows)
        assert [m for lv, _, m in g.rows if lv == "WARN" and "differs" in m][0].startswith("live answer differs")
        g2 = f.Gate()
        assert f.check_server(g2, "http://example.com:8780", ["c"]) == {} and g2.count("FAIL") == 1
    finally:
        srv.shutdown()
