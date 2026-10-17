"""Render out/draw.html in a REAL browser (headless Edge/Chrome, file:// only, no server, no network) and drive it with synthetic
pointer events: python tools/page_drive.py [--scenario all]   -> out/_page/*.png + out/_page/report.json
Each scenario injects a small driver script into a COPY of the page (out/_page/drive_*.html), waits for results, writes a DOM summary
into <pre id=__drive> and takes a screenshot. The page itself is not modified."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "out"
PG = OUT / "_page"
BROWSERS = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]

# drawing helper: synthetic pointer events (setPointerCapture is stubbed: a synthetic pointerId is not an active pointer)
DRIVER = r"""
<script>
(function(){
const log = []; window.addEventListener('error', e => log.push('ERR ' + e.message));
cv.setPointerCapture = () => {};
function ev(type, x, y){ const r = cv.getBoundingClientRect(); cv.dispatchEvent(new PointerEvent(type, {clientX: r.left + x*r.width, clientY: r.top + y*r.height, pointerId: 7, bubbles: true})); }
async function drawPath(P, gapMs, hold){ ev('pointerdown', P[0][0], P[0][1]); for (let i = 1; i < P.length; i++){ ev('pointermove', P[i][0], P[i][1]); if (gapMs) await new Promise(r => setTimeout(r, (hold && i === hold[0]) ? hold[1] : gapMs)); } ev('pointerup', 0, 0); }
const scen = __SCEN__;
async function go(){
  try {
    if (scen.preset) { usePreset(scen.preset); }
    if (scen.clipName) { sel.value = String(RAW.clips.findIndex(c => c.name === scen.clipName)); sizeCanvas(); }
    if (scen.timing) document.getElementById('timing').checked = true;
    if (scen.rot != null) document.getElementById('rot').checked = scen.rot;
    if (scen.draw) { const g = scen.gap || 0; await drawPath(scen.draw, g, scen.hold); }
    await new Promise(r => setTimeout(r, 800));
    const g = document.getElementById('grid');
    const cards = [...g.children].map(c => c.innerText.replace(/\s+/g,' ').slice(0,150));
    const cvs = [...g.querySelectorAll('.res canvas')];
    const nonblank = cvs.filter(c => { try { const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data; let s = 0; for (let i = 0; i < d.length; i += 400) s += d[i] + d[i+1] + d[i+2]; return s > 0; } catch(e) { return 'tainted'; } }).length;
    const out = {scen: scen.name, cards: g.querySelectorAll('.res').length, banner: !!g.querySelector('.banner'), weak_cards: g.querySelectorAll('.res.weak').length, turned_chips: (g.innerText.match(/sketch turned/g) || []).length, first: cards[0], nonblank_canvases: nonblank, status: document.getElementById('status').textContent, pts: pts.length, errors: log};
    const pre = document.createElement('pre'); pre.id = '__drive'; pre.style.display = 'none'; pre.textContent = JSON.stringify(out); document.body.appendChild(pre);
  } catch(e) { const pre = document.createElement('pre'); pre.id = '__drive'; pre.textContent = JSON.stringify({scen: scen.name, fatal: String(e && e.stack || e)}); document.body.appendChild(pre); }
}
setTimeout(go, 600);
})();
</script>
"""

U = [[0.15 + 0.45 * i / 14, 0.38] for i in range(15)] + [[0.60 + 0.10 * (1 - abs(2 * i / 10 - 1) ** 2) ** 0.5, 0.38 + 0.24 * i / 10] for i in range(1, 10)] + [[0.60 - 0.45 * i / 14, 0.62] for i in range(15)]
LINE = [[0.1 + 0.8 * i / 30, 0.5 + 0.01 * (i % 3)] for i in range(31)]
SCRIBBLE = [[0.5 + 0.08 * __import__("math").cos(i / 3) , 0.5 + 0.12 * __import__("math").sin(i / 2.3)] for i in range(40)]
SCENARIOS = [
    {"name": "initial_u_turn_preset"},
    {"name": "draw_u_turn", "draw": U, "rot": True},
    {"name": "draw_line", "draw": LINE, "rot": False},
    {"name": "draw_scribble", "draw": SCRIBBLE, "rot": True},
    {"name": "yonge_u_turn_negative", "clipName": "yonge_dundas", "draw": U, "rot": False},
    {"name": "preset_stop_and_go", "preset": "stop_and_go"},
    {"name": "preset_loiter", "preset": "loiter"},
    {"name": "draw_walk_stop_walk", "draw": [[0.1 + 0.8 * i / 30, 0.5] for i in range(31)], "rot": True, "timing": True, "gap": 40, "hold": [10, 1200]},
]


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    raise SystemExit("no Edge/Chrome found")


def run(scen: dict, w: int = 1280, h: int = 1100) -> dict:
    PG.mkdir(parents=True, exist_ok=True)
    html = (OUT / "draw.html").read_text(encoding="utf-8")
    page = PG / f"drive_{scen['name']}.html"
    page.write_text(html.replace("</body>", DRIVER.replace("__SCEN__", json.dumps(scen)) + "</body>"), encoding="utf-8")
    png = PG / f"{scen['name']}.png"
    udd = tempfile.mkdtemp(prefix="mimic_edge_")
    base = [find_browser(), "--headless=new", "--disable-gpu", "--no-first-run", "--disable-background-networking", "--disable-extensions",
            f"--user-data-dir={udd}", "--allow-file-access-from-files", f"--window-size={w},{h}", "--virtual-time-budget=9000"]
    url = page.as_uri()
    subprocess.run(base + [f"--screenshot={png}", url], capture_output=True, timeout=120)
    dom = subprocess.run(base + ["--dump-dom", url], capture_output=True, timeout=120, text=True, encoding="utf-8", errors="replace").stdout
    m = re.search(r'<pre id="__drive"[^>]*>(.*?)</pre>', dom, re.S)
    res = json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&")) if m else {"scen": scen["name"], "fatal": "driver output missing (page did not render?)"}
    res["png"] = str(png)
    return res


if __name__ == "__main__":
    which = sys.argv[1:] or [s["name"] for s in SCENARIOS]
    rep = [run(s) for s in SCENARIOS if s["name"] in which]
    (PG / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for r in rep:
        print(json.dumps({k: v for k, v in r.items() if k != "first"})[:400])
