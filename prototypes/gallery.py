"""Build prototypes/gallery.html from the build workflow result: every idea's hero images (embedded), key results,
kill test and the reviewer's scores. Images are downscaled JPEG data URIs so the page is one portable file.

usage: python gallery.py WORKFLOW_OUTPUT.json [OUT.html]
"""
import base64
import html
import json
import sys
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
src = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
items = src.get("result", src)
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "gallery.html"
esc = html.escape


def img_uri(path: str, width: int = 960) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    im = cv2.imread(str(p))
    if im is None:
        return None
    if im.shape[1] > width:
        im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


cards = []
for it in items:
    fin, rev = it.get("f") or it.get("b"), it.get("r") or {}
    if not fin:
        cards.append(f'<section class="card"><h2>{esc(it["n"])}. {esc(it["name"])}</h2><p class="bad">Build did not finish.</p></section>')
        continue
    kt = fin.get("kill_test") or {}
    figs = []
    for v in fin.get("videos", [])[:3]:
        h = v.get("hero", "")
        uri = img_uri(h if Path(h).is_absolute() else str(Path(fin.get("folder", "")) / h))
        if uri:
            figs.append(f'<figure><img src="{uri}" loading="lazy"><figcaption><b>{esc(v["video"])}</b>: '
                        f'{esc(v.get("key_result", ""))}</figcaption></figure>')
    score = (f'<span class="pill">demo {rev.get("demo_score", "?")}/10</span><span class="pill">wow {rev.get("wow", "?")}/10</span>'
             if rev else "")
    status = fin.get("status", "?")
    cards.append(
        f'<section class="card" id="p{esc(it["n"])}"><h2>{esc(it["n"])}. {esc(it["name"])} '
        f'<span class="pill {esc(status)}">{esc(status)}</span>{score}</h2>'
        + (f'<p class="verdict">Reviewer: {esc(rev.get("one_liner", ""))}</p>' if rev else "")
        + "".join(figs)
        + '<ul>' + "".join(f"<li>{esc(m)}</li>" for m in fin.get("headline_metrics", [])[:6]) + "</ul>"
        + f'<p><b>Kill test</b> ({"passed" if kt.get("passed") else "not passed"}): {esc(kt.get("result", ""))}</p>'
        + f'<p><b>Demo:</b> {esc(fin.get("demo_script", ""))}</p>'
        + f'<p><b>Fails / limits:</b> {esc(fin.get("what_fails", ""))}</p>'
        + (f'<p><b>Simulated:</b> {esc("; ".join(fin.get("simulated", [])))}</p>' if fin.get("simulated") else "")
        + f'<p class="path">{esc(fin.get("folder", ""))}</p></section>')

toc = " · ".join(f'<a href="#p{esc(it["n"])}">{esc(it["n"])} {esc(it["name"])}</a>' for it in items)
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prototype Gallery</title><style>
:root{{--bg:#f6f7f9;--fg:#1c2127;--muted:#5f6b77;--card:#fff;--line:#dde2e7}}
@media (prefers-color-scheme: dark){{:root{{--bg:#121519;--fg:#e7eaed;--muted:#9aa6b2;--card:#1b2026;--line:#2b323a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:20px 16px 60px}}h1{{margin:0 0 6px}}.sub,.path,figcaption{{color:var(--muted);font-size:13px}}
.toc{{font-size:13px;line-height:1.9}}.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:18px 0}}
h2{{font-size:19px;margin:0 0 6px}}img{{max-width:100%;border-radius:8px;display:block}}figure{{margin:10px 0}}
.pill{{display:inline-block;font-size:12px;font-weight:600;border-radius:10px;padding:1px 8px;margin-left:6px;background:#e3e8ee;color:#1c2127}}
.working{{background:#2e9e5b;color:#fff}}.partial{{background:#e0a100;color:#fff}}.broken{{background:#d64545;color:#fff}}
.verdict{{font-style:italic}}.bad{{color:#d64545}}
</style></head><body><main><h1>15 prototypes, tested on real footage</h1>
<p class="sub">Video prototype experiments. All models ran locally:
YOLO26s + ByteTrack (DataEngine YOLO step), Cosmos3-Edge (standing in for Cosmos3-Reason), Cosmos-Embed1, duckdb (standing in for VastDB).
Footage: Intel IoT samples (CC BY 4.0), Wikimedia Commons (PD, CC0, CC BY, CC BY-SA) and Pexels clips; local tests only.</p>
<p class="toc">{toc}</p>{"".join(cards)}</main></body></html>"""
dst.write_text(page, encoding="utf-8")
print("wrote", dst, round(len(page) / 1e6, 2), "MB,", len(items), "ideas")
