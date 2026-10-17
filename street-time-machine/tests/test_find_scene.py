"""Find & Follow scene builder on synthetic cv2 videos (no network, no GPU, all files in a temp dir).

Three cameras: a red-top person walks A -> B (views overlap) -> C (no camera between, ~10 s unseen) and passes behind
a pillar on A (so the A visit fragments); a blue-top person wanders on A the whole time; a green-top decoy walks
through C just after the red person. C's video starts 20 s into the scene (offset_s 20).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import stm
from stm import fastindex, follow

ROOT = Path(__file__).resolve().parent.parent
W, H, VFPS = 480, 270, 10
RED, BLACK = (40, 40, 200), (35, 35, 35)          # BGR
BLUE, WHITE = (190, 90, 30), (230, 230, 230)
GREEN, GREY = (60, 170, 60), (128, 128, 128)
PILLAR = (0.40, 0.62)                             # camera A, x range, drawn in front of people
PHONE = [0.70, 0.62, 0.04, 0.03]                  # camera A, put down at 14.5 s, gone at 30 s


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_find_scene", ROOT / "scripts" / "build_find_scene.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


builder = _load_builder()


def walk(t: float, t0: float, t1: float, foot: float, h: float, x0: float = -0.05, x1: float = 1.05):
    if not t0 <= t <= t1:
        return None
    return (x0 + (x1 - x0) * (t - t0) / (t1 - t0), foot, h)


def wander(t: float, t0: float, lo: float, hi: float, foot: float, h: float, speed: float = 0.04):
    if t < t0:
        return None
    span = hi - lo
    p = (speed * (t - t0)) % (2 * span)
    return (lo + (p if p <= span else 2 * span - p), foot, h)


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (H // 6, W // 6, 3), dtype=np.uint8)
    bg = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    bg = cv2.addWeighted(bg, 0.35, np.full_like(bg, (118, 124, 120)), 0.65, 0)   # low-saturation street texture
    for y in range(int(H * 0.5), H, 18):
        cv2.line(bg, (0, y), (W, y), (96, 100, 98), 1)
    return bg


def _person(img: np.ndarray, pos, top, bottom) -> None:
    cx, foot, h = pos
    hp = int(h * H)
    wp = int(0.36 * hp)
    x, y1 = int(cx * W), int(foot * H) - hp
    cv2.circle(img, (x, y1 + int(0.11 * hp)), int(0.11 * hp), (120, 150, 190), -1)
    cv2.rectangle(img, (x - wp // 2, y1 + int(0.22 * hp)), (x + wp // 2, y1 + int(0.55 * hp)), top, -1)
    cv2.rectangle(img, (x - int(0.4 * wp), y1 + int(0.55 * hp)), (x + int(0.4 * wp), y1 + hp), bottom, -1)


def make_video(path: Path, seconds: float, seed: int, people, pillar=None, phone=None) -> Path:
    """people: [(fn(scene t) -> (cx, foot, h) | None, top, bottom)]; video t = scene t - offset handled by caller."""
    bg = _texture(seed)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VFPS, (W, H))
    for f in range(int(seconds * VFPS)):
        t = f / VFPS
        img = bg.copy()
        if phone and phone[1] <= t < phone[2]:
            x, y, w, h = phone[0]
            cv2.rectangle(img, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (20, 20, 25), -1)
        for fn, top, bottom in people:
            pos = fn(t)
            if pos:
                _person(img, pos, top, bottom)
        if pillar:
            cv2.rectangle(img, (int(pillar[0] * W), 0), (int(pillar[1] * W), H), (84, 92, 104), -1)
        wr.write(img)
    wr.release()
    return path


def make_set(folder: Path) -> Path:
    """Write a.mp4, b.mp4, c.mp4 and cams.follow.json into folder; returns the topology path."""
    folder.mkdir(parents=True, exist_ok=True)
    make_video(folder / "a.mp4", 60, 1, [(lambda t: walk(t, 2, 20, 0.93, 0.38), RED, BLACK),
                                         (lambda t: wander(t, 3, 0.06, 0.34, 0.45, 0.30, 0.05), BLUE, WHITE)],
               pillar=PILLAR, phone=(PHONE, 14.5, 30.0))
    make_video(folder / "b.mp4", 60, 2, [(lambda t: walk(t, 17, 33, 0.90, 0.40), RED, BLACK)])
    off_c = 20.0
    make_video(folder / "c.mp4", 45, 3, [(lambda t: walk(t + off_c, 43, 58, 0.92, 0.38), RED, BLACK),
                                         (lambda t: walk(t + off_c, 46, 62, 0.50, 0.30), GREEN, GREY)])
    topo = {"title": "Synthetic walk A -> B -> C", "clock0": "13:00:00", "date": "2026-10-17",
            "credit": "Synthetic test footage drawn with cv2",
            "cameras": [
                {"id": "A", "name": "Cafe", "video": "a.mp4", "offset_s": 0, "pos": [0.1, 0.5], "exits": {"east": "B"},
                 "links": [{"to": "B", "from_zone": "east", "to_zone": "west", "min_s": -4, "max_s": 2, "overlap": True}],
                 "props": [{"kind": "pillar", "label": "pillar", "box": [PILLAR[0], 0, PILLAR[1] - PILLAR[0], 1]}],
                 "items": [{"id": "phone", "kind": "phone", "label": "your phone", "box": PHONE}]},
                {"id": "B", "name": "Corridor", "video": "b.mp4", "offset_s": 0, "pos": [0.5, 0.5],
                 "exits": {"west": "A", "east": "C"},
                 "links": [{"to": "A", "from_zone": "west", "to_zone": "east", "min_s": -4, "max_s": 2, "overlap": True}],
                 "blind": [{"to": "C", "min_s": 6, "max_s": 15}]},
                {"id": "C", "name": "Street", "video": "c.mp4", "offset_s": off_c, "pos": [0.9, 0.5],
                 "exits": {"west": "B", "east": "street"}, "blind": [{"to": "B", "min_s": 6, "max_s": 15}]}]}
    p = folder / "cams.follow.json"
    p.write_text(json.dumps(topo, indent=1), encoding="utf-8")
    return p


def _fake_items(seen: list) -> types.ModuleType:
    mod = types.ModuleType("stm.items")

    def item_span(video_path, box, fps=2, offset_s=0.0):
        seen.append((Path(video_path).name, list(box), fps, offset_s))
        return {"t0": 14.5, "t1": 30.0, "evidence": "fake span for the test"}
    mod.item_span = item_span
    return mod


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("find_scene")
    cams = make_set(tmp / "src")
    seen: list = []
    logs: list[str] = []
    mp = pytest.MonkeyPatch()
    fake = _fake_items(seen)
    mp.setitem(sys.modules, "stm.items", fake)
    mp.setattr(stm, "items", fake, raising=False)
    t0 = time.perf_counter()
    try:
        sc = builder.build(cams, "synth", tmp / "out", log=logs.append)
    finally:
        mp.undo()
    secs = time.perf_counter() - t0
    print(f"\nfind scene build time: {secs:.1f} s ({sc['build']['index_s']} s indexing)")
    for line in logs:
        print(line)
    return {"scene": sc, "folder": tmp / "out" / "synth", "logs": logs, "seen": seen, "secs": secs}


def _by_id(sc):
    return {t["id"]: t for t in sc["tracks"]}


def _ident(sc, top: str):
    by = _by_id(sc)
    hits = [i for i in sc["identities"] if any(by[t]["look"]["top"] == top for t in i["tracks"])]
    return hits, by


def test_scene_json_valid_and_files_exist(built):
    sc, folder = built["scene"], built["folder"]
    assert builder.check_scene(sc, folder) == []
    disk = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    assert disk["version"] == 1 and disk["simulated"] is False and isinstance(disk["title"], str)
    assert disk["clock0"] == "2026-10-17T13:00:00" and isinstance(disk["credit"], str) and disk["credit"]
    assert disk["duration"] == pytest.approx(65.0, abs=0.2)
    for c in disk["cameras"]:
        assert set(c) >= {"id", "name", "pos", "links", "blindLinks", "props", "video", "offset", "exits"}
        assert (folder / c["video"]).stat().st_size > 0
        assert set(c["blindLinks"]) <= set(c["links"])
    cams = {c["id"]: c for c in disk["cameras"]}
    assert cams["C"]["offset"] == 20.0 and cams["C"]["exits"] == "street" and cams["A"]["exits"] is None
    assert cams["B"]["blindLinks"] == ["C"] and "C" in cams["B"]["links"]
    for t in disk["tracks"]:
        assert set(t) >= {"id", "cam", "kind", "t0", "t1", "boxes", "desc", "look", "crop"}
        assert t["kind"] == "person" and t["crop"] and (folder / t["crop"]).stat().st_size > 0
        assert all(len(b) == 5 for b in t["boxes"])
        assert {"top", "bottom", "items"} <= set(t["look"])
        dts = np.diff([b[0] for b in t["boxes"]])
        assert len(t["boxes"]) < 3 or np.median(dts) == pytest.approx(0.5, abs=0.06)      # ~2 boxes per second
    for it in disk["items"]:
        assert set(it) >= {"id", "kind", "label", "cam", "owner", "box", "t0", "t1"}
    if shutil.which("ffprobe"):
        codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=codec_name", "-of", "csv=p=0", str(folder / cams["A"]["video"])],
                               capture_output=True, text=True).stdout.strip()
        assert codec == "h264"


def test_red_person_followed_a_b_c(built):
    sc = built["scene"]
    hits, by = _ident(sc, "red")
    chains = [[by[t]["cam"] for t in i["tracks"]] for i in hits]
    assert chains == [["A", "B", "C"]], chains
    ident = hits[0]
    a, b, c = (by[t] for t in ident["tracks"])
    assert all(by[t]["look"]["top"] == "red" and by[t]["look"]["bottom"] == "black" for t in ident["tracks"])
    hop_ab, hop_bc = ident["hops"]
    assert (hop_ab["from"], hop_ab["to"]) == (a["id"], b["id"]) and hop_ab["gap"] == 0.0
    assert (hop_bc["from"], hop_bc["to"]) == (b["id"], c["id"])
    assert 8.0 <= hop_bc["gap"] <= 12.5, hop_bc
    assert "no camera between" in hop_bc["why"] and "overlap" in hop_ab["why"]
    assert hop_bc["score"] < hop_ab["score"]               # blind link scores lower
    assert 41 <= c["t0"] <= 45                             # C offset applied (video t 23 -> scene t 43)


def test_others_stay_separate(built):
    sc = built["scene"]
    red, by = _ident(sc, "red")
    blue, _ = _ident(sc, "blue")
    green, _ = _ident(sc, "green")
    assert blue and green
    assert not {i["id"] for i in blue} & {i["id"] for i in red}
    assert not {i["id"] for i in green} & {i["id"] for i in red}
    assert all(by[t]["cam"] == "A" for i in blue for t in i["tracks"])
    assert all(by[t]["cam"] == "C" and not i["hops"] for i in green for t in i["tracks"])
    tops = [by[t]["look"]["top"] for i in sc["identities"] for t in i["tracks"]]
    assert all(len({by[t]["look"]["top"] for t in i["tracks"]}) == 1 for i in sc["identities"]), tops


def test_fragments_merged(built):
    sc = built["scene"]
    reds_on_a = [t for t in sc["tracks"] if t["cam"] == "A" and t["look"]["top"] == "red"]
    assert len(reds_on_a) == 1
    assert len(reds_on_a[0]["fragments"]) >= 2            # split by the pillar, joined again
    assert sc["build"]["fragments"] > sc["build"]["person_tracks"]
    assert reds_on_a[0]["t1"] - reds_on_a[0]["t0"] >= 14


def test_item_span_and_owner(built):
    sc = built["scene"]
    (it,) = sc["items"]
    assert built["seen"] and built["seen"][0][0] == "a.mp4" and built["seen"][0][2] == 2
    assert it["t0"] == 14.5 and it["t1"] == 30.0 and it["gone"] is True
    red, _ = _ident(sc, "red")
    assert it["owner"] == red[0]["id"]


def test_build_time_printed(built):
    assert any(line.startswith("build time") for line in built["logs"])
    # wall time depends on what else the machine runs (4-37 s seen for these 3 short 480x270 clips); the work does not
    assert built["scene"]["build"]["index_cpu_s"] < 30          # ~7-9 CPU-s


def test_items_fallback_without_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "stm.items", None)     # import fails -> t0 0, t1 end of footage
    monkeypatch.delattr(stm, "items", raising=False)
    topo = {"cameras": [{"id": "A", "video": "x.mp4", "offset_s": 0.0,
                         "items": [{"id": "bag", "kind": "bag", "label": "bag", "box": [10, 20, 30, 40]}]}]}
    (it,) = builder.build_items(topo, {"A": [100, 200]}, [], [], 50.0, print)
    assert it["t0"] == 0.0 and it["t1"] == 50.0 and it["gone"] is False and it["owner"] is None
    assert it["box"] == [0.1, 0.1, 0.3, 0.2]


def test_item_times_use_scene_seconds(monkeypatch):
    seen: list = []
    fake = _fake_items(seen)
    monkeypatch.setitem(sys.modules, "stm.items", fake)
    monkeypatch.setattr(stm, "items", fake, raising=False)     # `from stm import items` reads the package attribute
    topo = {"cameras": [{"id": "C", "video": "c.mp4", "offset_s": 20.0,
                         "items": [{"id": "bag", "box": [0.1, 0.1, 0.2, 0.2]}]}]}
    (it,) = builder.build_items(topo, {"C": [100, 100]}, [], [], 80.0, print)
    assert seen[0][3] == -20.0             # stm.items returns video t - offset_s; the scene is video t + 20
    assert it["t0"] == 14.5 and it["t1"] == 30.0
    # --seconds cut camera C's footage at scene 25: the item went after that, which this footage cannot show
    (cut,) = builder.build_items(topo, {"C": [100, 100]}, [], [], 80.0, print, {"C": 25.0})
    assert cut["t1"] == 25.0 and cut["gone"] is False


def _trk(tid, cam, t0, t1, x0, x1, up, lo):
    ts = np.arange(t0, t1 + 1e-6, 0.5)
    boxes = [[round(float(t), 2), float(x0 + (x1 - x0) * (t - t0) / (t1 - t0)), 0.5, 0.08, 0.4] for t in ts]
    sig = {"upper": [0.0] * 12, "lower": [0.0] * 12}
    sig["upper"][fastindex.COLORS.index(up)] = 1.0
    sig["lower"][fastindex.COLORS.index(lo)] = 1.0
    return {"id": tid, "cam": cam, "kind": "person", "t0": boxes[0][0], "t1": boxes[-1][0], "n": len(boxes),
            "boxes": boxes, "sig": sig, "fragments": [tid]}


def test_no_link_below_threshold_and_merge_rules():
    cams = [{"id": "A", "exits": {"east": "B"}, "links": [{"to": "B", "min_s": 0, "max_s": 5}]},
            {"id": "B", "exits": {"west": "A"}}]
    a = _trk("a", "A", 0, 10, 0.3, 0.95, "red", "black")
    b_other = _trk("b", "B", 12, 20, 0.0, 0.6, "blue", "white")
    assert follow.handoffs([a, b_other], cams) == []
    b_same = _trk("b2", "B", 12, 20, 0.0, 0.6, "red", "black")
    (hop,) = follow.handoffs([a, b_other, b_same], cams)
    assert hop["to"] == "b2" and hop["score"] >= follow.MIN_SCORE
    # same camera: a 2 s gap continuing the walk merges; a different colour or a 6 s gap does not
    f1 = _trk("f1", "A", 0, 4, 0.1, 0.3, "red", "black")
    f2 = _trk("f2", "A", 6, 10, 0.4, 0.6, "red", "black")
    f3 = _trk("f3", "A", 6, 10, 0.4, 0.6, "green", "grey")
    f4 = _trk("f4", "A", 16, 20, 0.6, 0.8, "red", "black")
    out = follow.merge_fragments([f1, f2, f3, f4])
    frags = sorted(sorted(t["fragments"]) for t in out)
    assert frags == [["f1", "f2"], ["f3"], ["f4"]]


def test_hop_never_goes_back_in_time():
    # overlapping views allow b to start before a ends, but a b that started before a would put the identity's
    # tracks out of time order (find.html sorts them by t0 and then finds no hop between them)
    cams = [{"id": "A", "exits": {"east": "B"}, "links": [{"to": "B", "min_s": -4, "max_s": 2, "overlap": True}]},
            {"id": "B", "exits": {"west": "A"}}]
    a = _trk("a", "A", 10, 13, 0.6, 0.95, "red", "black")
    early = _trk("b", "B", 9, 20, 0.0, 0.6, "red", "black")
    assert follow.handoffs([a, early], cams) == []
    late = _trk("b2", "B", 11, 20, 0.0, 0.6, "red", "black")
    (hop,) = follow.handoffs([a, late], cams)
    (ident,) = follow.identities([a, late], [hop])
    assert ident["tracks"] == ["a", "b2"] and hop["gap"] == 0.0


@pytest.mark.skipif(not builder.ffmpeg_path() or not shutil.which("ffprobe"), reason="needs ffmpeg and ffprobe")
def test_variable_frame_rate_uses_timestamps(tmp_path):
    ff = builder.ffmpeg_path()
    for name, rate in (("p1", 5), ("p2", 20)):          # 4 s at 5 fps, then 4 s at 20 fps, joined without re-encoding
        subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate={rate}:duration=4",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_path / f"{name}.mp4")], check=True)
    (tmp_path / "list.txt").write_text("file 'p1.mp4'\nfile 'p2.mp4'\n", encoding="utf-8")
    vfr = tmp_path / "vfr.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.txt"), "-c", "copy",
                    str(vfr)], check=True)
    pts = [float(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "frame=pts_time", "-of", "csv=p=0", str(vfr)],
                                            capture_output=True, text=True, check=True).stdout.replace(",", " ").split()]
    cap = cv2.VideoCapture(str(vfr))
    vfps, ts = cap.get(cv2.CAP_PROP_FPS), []
    while cap.grab():
        ts.append(fastindex.frame_time(cap, len(ts), vfps))
    cap.release()
    assert len(ts) == len(pts)
    assert max(abs(i / vfps - p) for i, p in enumerate(pts)) > 0.5      # frame index / fps really drifts here
    assert max(abs(t - p) for t, p in zip(ts, pts)) < 0.01             # what the browser plays
    assert fastindex.index_video(vfr, "V")["duration"] == pytest.approx(pts[-1], abs=0.2)


def test_colour_words():
    hsv = cv2.cvtColor(np.array([[RED] * 50, [BLUE] * 50, [GREEN] * 50, [BLACK] * 50, [WHITE] * 50],
                                np.uint8), cv2.COLOR_BGR2HSV)
    words = [fastindex.colour_word(fastindex.colour_hist(row)) for row in hsv]
    assert words == ["red", "blue", "green", "black", "white"]
