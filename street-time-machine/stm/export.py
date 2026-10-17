"""Bake a static, single-file demo into dist/.

dist/index.html = ui/index.html with
    <script>window.__STM__={...};window.__STM_FILES__={...}</script>
injected right before the page's main (inline) <script>.

__STM__        maps every JSON URL the UI requests ("api/cams", "out/<cam>/<name>.json",
               "out/captions/captions.json") to its parsed content. Missing files are skipped.
__STM_FILES__  maps every image path the UI requests ("out/<cam>/<rel>", "out/captions/<rel>") to a
               JPEG data URI. Image paths are collected by walking every JSON value and keeping strings
               that end in .jpg/.png (relative to that JSON's folder), plus the ones the UI builds itself
               (background.jpg, crops/<ghost id>.jpg).
Videos are copied, not inlined: dist/out/<cam>/synopsis.mp4.
Cam order (also the order of "api/cams"): any event cam (e.g. Pack D "neighborhood_cam-1", found by its
out/<cam>/meta.json even when it is not in data.CAMS) first, then ptz_parking (the rehearsed fallback),
then yonge_dundas, then the other dev cams.
People: every crop of a person, bicycle or motorcycle (found by object id in any JSON) is Gaussian-blurred
before it becomes a data URI; the count is logged as "redacted_people: N".
Each cam keeps at most CROP_CAP crops (synopsis-ghost-only crops go first, then index-only ones).
If the page is still over the size budget (14 MB), the least important images are dropped first
(yonge_dundas crops, then other dev cams' crops, ...; the first cam and ptz_parking plates never) and
every drop is logged. Dropped images are still written as files under dist/out/ so a folder-served dist
keeps working.

CLI: python -m stm.export [cams...] [--dist DIR] [--max-mb 14]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2

from stm import data

ROOT = data.ROOT
OUT = data.OUT
UI = ROOT / "ui" / "index.html"
DIST = ROOT / "dist"
MAIN_CAM = "ptz_parking"          # the rehearsed fallback: its plates are never dropped
DEV_CAMS = ("ptz_parking", "car_detection", "person_bicycle_car_detection", "rochester_night", "yonge_dundas")
FRONT = ("ptz_parking", "yonge_dundas")   # after any event cam, in this order
MAX_BYTES = 14 * 1024 * 1024
CROP_CAP = 450                    # max crops per cam in the page
MAX_JSON_BYTES = 4 * 1024 * 1024  # a single JSON bigger than this is skipped (logged)
CAM_JSONS = ("meta", "synopsis", "ledger", "diff", "near", "qa", "moments", "flow", "index", "brief")
PEOPLE = ("person", "bicycle", "motorcycle")  # crops that show a person (riders too) are blurred
KEY_REFS = ("near", "qa", "brief", "moments", "diff", "ledger", "flow")  # images these use matter most
PERSON_IMG_RE = re.compile(r"^(?:crops|near|ask_crops)/(\d+)(?:_[^/]*)?\.(?:jpe?g|png)$", re.I)
CAPTIONS_URL = "out/captions/captions.json"
IMG_RE = re.compile(r"\.(jpe?g|png)$", re.I)
# (max long side px, JPEG quality)
TIERS = {"plate": (960, 72), "frame": (360, 70), "crop": (220, 70)}


# ---------- collecting ----------

def image_strings(obj: Any) -> list[str]:
    """Every string anywhere in obj that ends in .jpg/.jpeg/.png, in walk order, de-duplicated."""
    seen: dict[str, None] = {}

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str) and IMG_RE.search(o.strip()):
            seen.setdefault(o.strip(), None)

    walk(obj)
    return list(seen)


def _url_for(folder_url: str, rel: str) -> str | None:
    """'out/<cam>' + 'crops/1.jpg' -> 'out/<cam>/crops/1.jpg'. Rejects absolute paths, URLs and '..'."""
    rel = rel.replace("\\", "/").strip()
    if not rel or rel.startswith(("/", "data:", "http:", "https:")) or re.match(r"^[A-Za-z]:", rel):
        return None
    if ".." in rel.split("/"):
        return None
    while rel.startswith("./"):
        rel = rel[2:]
    return f"{folder_url}/{rel}"


def _read_json(p: Path, log: list[str]) -> Any:
    """Read a JSON written by another module; retry once in case it is mid-write."""
    for attempt in range(2):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
            if attempt == 0:
                time.sleep(0.5)
                continue
            log.append(f"skipped {p.relative_to(ROOT).as_posix()}: unreadable ({type(e).__name__})")
    return None


def _event_cams() -> list[str]:
    """out/<cam>/ folders with a meta.json that are not dev cams (e.g. Pack D synced from the lab VM)."""
    if not OUT.is_dir():
        return []
    return sorted(d.name for d in OUT.iterdir()
                  if d.is_dir() and d.name not in DEV_CAMS and d.name != "captions" and (d / "meta.json").is_file())


def demo_order(cams: list[str]) -> list[str]:
    """Event cams first, then ptz_parking, yonge_dundas, then the rest (stable)."""
    def key(ic: tuple[int, str]) -> tuple[int, int]:
        i, c = ic
        if c not in DEV_CAMS:
            return (0, i)
        if c in FRONT:
            return (1 + FRONT.index(c), i)
        return (9, i)
    return [c for _, c in sorted(enumerate(cams), key=key)]


def available_cams(cams: list[str] | None = None) -> list[str]:
    """Cams that have an out/<cam>/ folder with at least one UI JSON, in demo_order()."""
    want = list(cams) if cams else list(data.CAMS) + [c for c in _event_cams() if c not in data.CAMS]
    keep = []
    for c in want:
        d = OUT / c
        if c not in keep and d.is_dir() and any((d / f"{n}.json").exists() for n in CAM_JSONS):
            keep.append(c)
    return demo_order(keep)


def cam_entry(cam: str, meta: Any) -> dict[str, Any]:
    """The same shape stm.server's /api/cams returns for one cam."""
    m = meta if isinstance(meta, dict) else {}
    return {"camera": cam, "title": m.get("title"), "duration": m.get("duration"), "days": m.get("days"),
            "simulated": m.get("simulated", []), "has_qa": (OUT / cam / "qa.json").exists()}


def ui_json_names(cam: str, html: str) -> list[str]:
    """CAM_JSONS plus any other out/<cam>/<stem>.json whose stem the UI mentions ('stem' or stem.json),
    so a JSON a UI owner starts fetching today is inlined without touching this file."""
    names = list(CAM_JSONS)
    d = OUT / cam
    if d.is_dir() and html:
        quoted = set(re.findall(r"['\"`/]([A-Za-z0-9_\-]{2,60})(?:\.json)?['\"`]", html)) | \
            set(re.findall(r"([A-Za-z0-9_\-]{2,60})\.json", html))
        for p in sorted(d.glob("*.json")):
            if p.stem not in names and p.stem in quoted:
                names.append(p.stem)
    return names


def people_refs(obj: Any) -> set[str]:
    """object ids (as str) of every dict in obj whose name is a person/bicycle/motorcycle."""
    ids: set[str] = set()

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            n = o.get("name")
            if isinstance(n, str) and n.lower() in PEOPLE and o.get("object_id") is not None:
                ids.add(str(o["object_id"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return ids


def is_person_image(url: str, folder: str, person_ids: set[str]) -> bool:
    if not url.startswith(folder + "/"):
        return False
    m = PERSON_IMG_RE.match(url[len(folder) + 1:])
    return bool(m and m.group(1) in person_ids)


def collect(cams: list[str], log: list[str], missing: dict[str, str] | None = None,
            html: str = "", refs: dict[str, set[str]] | None = None, people: set[str] | None = None
            ) -> tuple[dict[str, Any], dict[str, Path], list[Path]]:
    """Returns (stm: url -> parsed json, images: url -> source file, videos: source files).

    `missing`, if given, is filled with image url -> what referenced it, for every reference whose file
    is not on disk. The page cannot show those (the UI hides a broken <img>), so build() logs them.
    `refs`, if given, gets image url -> JSON stems that reference it ("synopsis-ghost" for the ones the
    UI builds from ghost ids). `people`, if given, gets every image url that shows a person."""
    stm: dict[str, Any] = {"api/cams": []}
    images: dict[str, Path] = {}
    videos: list[Path] = []
    missing = {} if missing is None else missing
    refs = {} if refs is None else refs
    people = set() if people is None else people

    def add_img(url: str | None, ref: str, stem: str) -> None:
        if not url:
            return
        if url in images:
            refs.setdefault(url, set()).add(stem)
            return
        if (ROOT / url).is_file():
            images[url] = ROOT / url
            refs.setdefault(url, set()).add(stem)
        else:
            missing.setdefault(url, ref)

    for cam in cams:
        folder = f"out/{cam}"
        person_ids: set[str] = set()
        for name in ui_json_names(cam, html):
            p = OUT / cam / f"{name}.json"
            if not p.exists():
                continue
            if p.stat().st_size > MAX_JSON_BYTES:
                log.append(f"skipped {folder}/{name}.json: {p.stat().st_size / 2**20:.1f} MB is over the "
                           f"{MAX_JSON_BYTES / 2**20:.0f} MB per-JSON limit")
                continue
            obj = _read_json(p, log)
            if obj is None:
                continue
            url = f"{folder}/{name}.json"
            stm[url] = obj
            person_ids |= people_refs(obj)
            for s in image_strings(obj):
                add_img(_url_for(folder, s), url, name)
            if name == "synopsis" and isinstance(obj, dict):
                # the synopsis inspector builds crops/<object_id>.jpg itself
                for g in obj.get("ghosts") or []:
                    if isinstance(g, dict) and g.get("object_id") is not None:
                        add_img(f"{folder}/crops/{g['object_id']}.jpg", f"{url} (ghost crops)", "synopsis-ghost")
        add_img(f"{folder}/background.jpg", "ui (ask view plate)", "ui")
        stm["api/cams"].append(cam_entry(cam, stm.get(f"{folder}/meta.json")))
        people |= {u for u in images if is_person_image(u, folder, person_ids)}
        v = OUT / cam / "synopsis.mp4"
        if v.is_file():
            videos.append(v)

    cp = ROOT / CAPTIONS_URL
    if cp.exists():
        obj = _read_json(cp, log)
        if obj is not None:
            stm[CAPTIONS_URL] = obj
            for s in image_strings(obj):
                add_img(_url_for("out/captions", s), CAPTIONS_URL, "captions")
    return stm, images, videos


# ---------- encoding ----------

def tier_of(url: str) -> str:
    """plate = full street plates (background*, flow, frames/), frame = caption frames, crop = everything else."""
    parts = url.split("/")
    name = parts[-1].lower()
    if len(parts) >= 2 and parts[1] == "captions":
        return "frame"
    if name.startswith("background") or name.startswith("flow") or (len(parts) >= 4 and parts[2] == "frames"):
        return "plate"
    return "crop"


def blur_kernel(h: int, w: int) -> int:
    """Odd Gaussian kernel: 15 px at least, ~1/8 of the long side for bigger crops (a face stays unreadable)."""
    k = max(15, round(max(h, w) / 8))
    return k if k % 2 else k + 1


def encode(src: Path, max_side: int, quality: int, blur: bool = False) -> bytes | None:
    """Downscale (never upscale) to max_side on the long side, optionally Gaussian-blur, return JPEG bytes."""
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    if blur:
        k = blur_kernel(*img.shape[:2])
        img = cv2.GaussianBlur(img, (k, k), 0)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def data_uri(jpg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")


def drop_rank(url: str, protected: tuple[str, ...] | set[str] = (MAIN_CAM,),
              low: set[str] | frozenset = frozenset()) -> int:
    """Lower = dropped first. `protected` = the demo cams (first cam + ptz_parking): their plates are never
    dropped (rank 99). `low` = crops only the synopsis inspector uses (no JSON names them)."""
    parts = url.split("/")
    cam = parts[1] if len(parts) > 1 else ""
    t = tier_of(url)
    lo = url in low
    if t == "crop":
        if cam in protected or cam == "captions":
            return 6 if lo else 8
        if cam == "yonge_dundas":
            return 0 if lo else 1
        return 2 if lo else 3
    if t == "frame":
        return 4  # caption frames
    if cam not in protected:
        return 5 if not parts[-1].lower().startswith("background") else 7
    return 99


def crop_value(url: str, refs: dict[str, set[str]]) -> int:
    """2 = named by near/qa/brief/moments/diff/ledger/flow, 1 = by index (Find) or other, 0 = synopsis only."""
    r = refs.get(url, set())
    if any(x in KEY_REFS for x in r):
        return 2
    if r and r <= {"synopsis", "synopsis-ghost"}:
        return 0
    return 1


# ---------- page ----------

def _js(obj: Any) -> str:
    """JSON safe to put inside an inline <script> (NaN/Infinity, if any, are valid JS literals)."""
    s = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    return s.replace("</", "<\\/").replace("<!--", "<\\!--")


def inject(html: str, stm: dict[str, Any], files: dict[str, str]) -> str:
    """Insert the data script right before the first inline <script> (no src). Falls back to </body> or EOF."""
    tag = f"<script>window.__STM__={_js(stm)};window.__STM_FILES__={_js(files)}</script>\n"
    for m in re.finditer(r"<script\b([^>]*)>", html, re.I):
        if "src=" not in m.group(1).lower():
            return html[: m.start()] + tag + html[m.start():]
    m = re.search(r"</body\s*>", html, re.I)
    if m:
        return html[: m.start()] + tag + html[m.start():]
    return html + "\n" + tag


def _natural(u: str) -> list[Any]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", u)]


def build(cams: list[str] | None = None, dist: Path | str = DIST, ui: Path | str = UI,
          max_bytes: int = MAX_BYTES, quiet: bool = False, crop_cap: int = CROP_CAP) -> dict[str, Any]:
    """Write dist/index.html (+ videos, + files for any dropped image). Returns a report dict."""
    dist = Path(dist)
    log: list[str] = []
    cams = available_cams(cams)
    protected = {c for c in cams[:1] + [MAIN_CAM] if c in cams}
    html = Path(ui).read_bytes().decode("utf-8")  # keep the UI's own line endings byte for byte
    missing: dict[str, str] = {}
    refs: dict[str, set[str]] = {}
    people: set[str] = set()
    stm, images, videos = collect(cams, log, missing, html=html, refs=refs, people=people)
    if missing:  # referenced by a JSON (or by the UI) but not on disk: the page cannot show these
        by_ref: dict[str, list[str]] = {}
        for u, ref in missing.items():
            by_ref.setdefault(ref, []).append(u)
        log.append(f"{len(missing)} referenced images are not on disk, so they are not in the page:")
        for ref, us in sorted(by_ref.items()):
            log.append(f"  {len(us)} from {ref}, e.g. {', '.join(u.rsplit('/', 2)[-2] + '/' + u.rsplit('/', 1)[-1] for u in us[:3])}")

    encoded: dict[str, bytes] = {}
    redacted = 0
    for url, src in images.items():
        mx, q = TIERS[tier_of(url)]
        blur = url in people
        b = encode(src, mx, q, blur=blur)
        if b is None:
            log.append(f"skipped {url}: could not decode image")
            continue
        encoded[url] = b
        redacted += blur
    log.insert(0, f"redacted_people: {redacted} person/rider crops Gaussian-blurred before inlining")
    uris = {u: data_uri(b) for u, b in encoded.items()}

    # per-cam crop cap: keep the crops a JSON names (near/qa/brief/...), then index ones, then ghost-only ones
    dropped: list[str] = []
    capped: dict[str, int] = {}
    for cam in cams:
        pre = f"out/{cam}/"
        crops = [u for u in uris if u.startswith(pre) and tier_of(u) == "crop"]
        if len(crops) <= crop_cap:
            continue
        crops.sort(key=lambda u: (-crop_value(u, refs), _natural(u)))
        cut = crops[crop_cap:]
        by_v = {v: sum(1 for u in cut if crop_value(u, refs) == v) for v in (0, 1, 2)}
        log.append(f"capped {cam}: {len(crops)} crops > cap {crop_cap}, dropped {len(cut)} "
                   f"(ghost-only {by_v[0]}, index/other {by_v[1]}, named by near/qa/brief/... {by_v[2]})")
        capped[cam] = len(cut)
        for u in cut:
            dropped.append(u)
            del uris[u]

    # size of the page without images; each image costs len(key) + len(uri) + 6 (quotes, colon, comma)
    base_size = len(inject(html, stm, {}).encode("utf-8"))
    cost = {u: len(u) + len(v) + 6 for u, v in uris.items()}
    total = base_size + sum(cost.values())
    low = {u for u in uris if tier_of(u) == "crop" and crop_value(u, refs) == 0}
    rank = lambda u: drop_rank(u, protected, low)
    if total > max_bytes:
        over: list[str] = []
        order = sorted(uris, key=lambda u: (rank(u), -cost[u], u))
        for u in order:
            if total <= max_bytes:
                break
            if rank(u) >= 99:
                break
            total -= cost[u]
            over.append(u)
        for u in over:
            del uris[u]
        log.append(f"over budget: dropped {len(over)} images to fit {max_bytes / 2**20:.1f} MB")
        groups: dict[str, int] = {}
        for u in over:
            k = f"{u.split('/')[1]}/{tier_of(u)}"
            groups[k] = groups.get(k, 0) + 1
        for k, n in sorted(groups.items()):
            log.append(f"  dropped {n} {k} images")
        dropped += over

    page = inject(html, stm, uris)
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_bytes(page.encode("utf-8"))  # no newline translation, so the budget math holds
    page_bytes = (dist / "index.html").stat().st_size
    if page_bytes > max_bytes:
        log.append(f"WARNING: index.html is {page_bytes / 2**20:.2f} MB, over the {max_bytes / 2**20:.1f} MB budget "
                   f"even after dropping every droppable image")

    # image files under dist/out/ exist only for images dropped by THIS run; remove ones left by earlier runs
    # so a folder-served dist never shows a stale picture for a path the page no longer inlines
    keep = {(dist / u).resolve() for u in dropped}
    if (dist / "out").is_dir():
        for p in (dist / "out").rglob("*"):
            if p.is_file() and IMG_RE.search(p.name) and p.resolve() not in keep:
                p.unlink()
    for u in dropped:  # keep dropped images reachable when dist/ is served as a folder
        p = dist / u
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(encoded[u])

    video_out = []
    for v in videos:
        rel = v.relative_to(ROOT)
        dst = dist / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != v.stat().st_size or dst.stat().st_mtime < v.stat().st_mtime:
            shutil.copy2(v, dst)
        video_out.append({"path": rel.as_posix(), "bytes": dst.stat().st_size})

    per_cam: dict[str, dict[str, int]] = {}
    for c in cams + ["captions"]:
        pre = f"out/{c}/"
        js = [k for k in stm if k.startswith(pre)]
        im = [k for k in uris if k.startswith(pre)]
        if not js and not im:
            continue
        per_cam[c] = {"json": len(js), "json_bytes": sum(len(_js(stm[k])) for k in js), "images": len(im),
                      "image_bytes": sum(len(uris[k]) for k in im),
                      "dropped": sum(1 for d in dropped if d.startswith(pre))}
    report = {"dist": str(dist), "index_html_bytes": page_bytes, "max_bytes": max_bytes, "cams": cams,
              "redacted_people": redacted, "people": sorted(people & set(encoded)), "capped": capped,
              "protected": sorted(protected),
              "json_urls": sorted(stm), "n_json": len(stm), "n_images_found": len(images),
              "n_images_inlined": len(uris), "n_images_dropped": len(dropped), "dropped": dropped,
              "n_images_missing": len(missing), "missing": sorted(missing),
              "image_uri_bytes": sum(len(v) for v in uris.values()), "videos": video_out,
              "per_cam": per_cam, "log": log}
    if not quiet:
        print(summary_text(report))
    return report


def summary_text(r: dict[str, Any]) -> str:
    mb = lambda b: f"{b / 2**20:.2f} MB"
    lines = [f"stm.export -> {r['dist']}",
             f"index.html: {mb(r['index_html_bytes'])} (budget {mb(r['max_bytes'])}, "
             f"{'OK' if r['index_html_bytes'] <= r['max_bytes'] else 'OVER'})",
             f"cams: {len(r['cams'])} ({', '.join(r['cams'])})",
             f"JSON urls inlined: {r['n_json']}",
             f"images: {r['n_images_found']} found, {r['n_images_inlined']} inlined ({mb(r['image_uri_bytes'])} of data URIs), "
             f"{r['n_images_dropped']} dropped, {r.get('n_images_missing', 0)} referenced but missing on disk",
             f"videos copied: {len(r['videos'])} ({mb(sum(v['bytes'] for v in r['videos']))})"]
    for v in r["videos"]:
        lines.append(f"  {v['path']}: {mb(v['bytes'])}")
    lines.append(f"redacted_people: {r.get('redacted_people', 0)}")
    if r.get("capped"):
        lines.append("capped crops per cam: " + ", ".join(f"{c} -{n}" for c, n in r["capped"].items()))
    lines.append("per cam:")
    for c, s in r["per_cam"].items():
        lines.append(f"  {c}: {s['json']} json ({mb(s['json_bytes'])}), {s['images']} images ({mb(s['image_bytes'])}), "
                     f"{s['dropped']} dropped")
    lines.append("json urls: " + ", ".join(r["json_urls"]))
    if r["log"]:
        lines.append("log:")
        lines += [f"  {x}" for x in r["log"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stm.export", description=__doc__.splitlines()[0])
    ap.add_argument("cams", nargs="*", help="cams to include (default: every cam with outputs)")
    ap.add_argument("--dist", default=str(DIST))
    ap.add_argument("--max-mb", type=float, default=MAX_BYTES / 2**20)
    a = ap.parse_args(argv)
    unknown = [c for c in a.cams if c not in data.CAMS and not (OUT / c / "meta.json").is_file()]
    if unknown:
        print(f"unknown cams: {unknown}", file=sys.stderr)
        return 2
    r = build(a.cams or None, a.dist, max_bytes=int(a.max_mb * 2**20))
    (OUT / "export_summary.txt").write_text(summary_text(r) + "\n", encoding="utf-8")
    return 0 if r["index_html_bytes"] <= r["max_bytes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
