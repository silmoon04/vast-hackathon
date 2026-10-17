"""Outputs of Between The Lines: WebVTT, audio-described MP4 (ffmpeg ducking mix), HTML player, hero image."""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
import audio as au  # noqa: E402
import protokit as pk  # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
AD_TARGET_RMS_DB = -19.0  # voice level before the mix
# review flags: GREEN = a checkable claim was confirmed by the detectors; AMBER = something disagrees / doubtful; UNVERIFIED = nothing checkable
FLAG_RGB = {"GREEN": (50, 190, 110), "AMBER": (240, 175, 20), "UNVERIFIED": (125, 145, 175)}
FLAG_HEX = {"GREEN": "#2e9e5b", "AMBER": "#e0a100", "UNVERIFIED": "#7d91af"}


# ------------------------------------------------------------------------------------------------------ VTT


def vtt_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


def write_vtt(path: Path, lines: list[dict]) -> Path:
    out = ["WEBVTT", "", "NOTE Between The Lines: AI first-draft audio description. Cue id carries the review flag (GREEN/AMBER/UNVERIFIED).",
           "NOTE Every cue sits in a gap found by our own speech detector. GREEN = a count/colour claim was checked against YOLO and agreed;",
           "NOTE AMBER = a claim disagrees or the model is unsure; UNVERIFIED = nothing checkable. Every cue is a draft for a human writer."]
    for ln in lines:
        out += ["", f"AD-{ln['id']:03d}-{ln['flag']}", f"{vtt_time(ln['start'])} --> {vtt_time(ln['end'])}", ln["text"]]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------------ mix


def _prep_wav(src: str, dst: Path) -> float:
    """Resample a SAPI wav to 44.1 kHz mono, normalise to the target RMS with a soft peak limit; return seconds."""
    x, sr = au.load_wav(src)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(44100, sr)
    x = resample_poly(x, 44100 // g, sr // g).astype(np.float32)
    idx = np.where(np.abs(x) > 0.005)[0]
    x = x[idx[0]: idx[-1] + 1] if len(idx) else x
    act = np.abs(x) > 0.02 * np.abs(x).max()
    rms = float(np.sqrt(np.mean(x[act] ** 2)) + 1e-9)
    x = x * (10 ** (AD_TARGET_RMS_DB / 20) / rms)
    x = np.tanh(x * 0.9) / 0.9 * 0.95 if np.abs(x).max() > 0.95 else x
    pad = np.zeros(int(0.02 * 44100), np.float32)
    x = np.concatenate([pad, x, pad])
    import scipy.io.wavfile as wf
    wf.write(str(dst), 44100, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return len(x) / 44100


def cut_plain(src: str, t0: float, t1: float, dst: Path, width: int = 960) -> Path:
    vf = f"scale='trunc(min({width},iw)/2)*2':-2"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(src), "-t", f"{t1 - t0:.3f}", "-vf", vf, "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
                    str(dst)], check=True)
    return dst


def mix_ad(plain: Path, lines: list[dict], out_mp4: Path, work: Path, duck_stem: Path) -> dict:
    """Audio-described MP4: each line's TTS is delayed to its gap start (adelay), summed into an AD bus, the programme is
    ducked with sidechaincompress keyed by that bus, then the two are mixed. Video stream is copied untouched."""
    work.mkdir(parents=True, exist_ok=True)
    if not lines:
        shutil.copy(plain, out_mp4)
        return {"lines_mixed": 0}
    cmd = [FFMPEG, "-y", "-v", "error", "-i", str(plain)]
    fc = []
    for k, ln in enumerate(lines):
        wav = work / f"ad_{ln['id']:03d}.wav"
        ln["voice_s"] = round(_prep_wav(ln["wav"], wav), 3)
        cmd += ["-i", str(wav)]
        ms = int(round(ln["start"] * 1000))
        fc.append(f"[{k + 1}:a]pan=stereo|c0=c0|c1=c0,adelay={ms}|{ms}[l{k}]")  # explicit upmix: no implicit -3 dB
    labels = "".join(f"[l{k}]" for k in range(len(lines)))
    fc.append(f"{labels}amix=inputs={len(lines)}:normalize=0:dropout_transition=0[bus]" if len(lines) > 1 else "[l0]anull[bus]")
    fc.append("[bus]asplit=2[sc][ad]")
    nch = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "csv=p=0", str(plain)],
                             capture_output=True, text=True).stdout.strip() or 2)
    fc.append("[0:a]aresample=44100," + ("pan=stereo|c0=c0|c1=c0" if nch == 1 else "aformat=channel_layouts=stereo") + "[prog]")  # mono stays level-neutral
    fc.append("[prog][sc]sidechaincompress=threshold=0.02:ratio=12:attack=40:release=450:makeup=1[duck0]")
    fc.append("[duck0]asplit=2[duck][duckstem]")
    fc.append("[duck][ad]amix=inputs=2:normalize=0:duration=first[mix]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "0:v", "-c:v", "copy", "-map", "[mix]", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(out_mp4), "-map", "[duckstem]", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(duck_stem)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg mix failed: " + r.stderr[-800:])
    return {"lines_mixed": len(lines)}


def ducking_depth(orig_wav: Path, duck_wav: Path, lines: list[dict]) -> dict:
    """Programme level inside each line window, original vs ducked stem (dB). Measured, not assumed."""
    a, sr = au.load_wav(orig_wav)
    b, _ = au.load_wav(duck_wav)
    d = []
    for ln in lines:
        i0, i1 = int((ln["start"] + 0.3) * sr), int((ln["end"] - 0.1) * sr)
        if i1 - i0 < sr // 5:
            continue
        ra = 20 * np.log10(np.sqrt(np.mean(a[i0:i1] ** 2)) + 1e-7)
        rb = 20 * np.log10(np.sqrt(np.mean(b[i0:min(i1, len(b))] ** 2)) + 1e-7)
        if ra > -70:
            d.append(ra - rb)
    return {"mean_db": round(float(np.mean(d)), 1) if d else None, "min_db": round(float(np.min(d)), 1) if d else None,
            "max_db": round(float(np.max(d)), 1) if d else None, "n": len(d)}


# ------------------------------------------------------------------------------------------------------ player


def write_player(path: Path, name: str, title: str, duration: float, lines: list[dict], speech: list, gaps: list, credit: str,
                 stats: dict, level: np.ndarray | None = None, eye: dict | None = None) -> Path:
    wave = []
    if level is not None and len(level):
        n = 700
        lv = np.clip((np.asarray(level) + 70) / 60, 0, 1)
        edges = np.linspace(0, len(lv), n + 1).astype(int)
        wave = [round(float(lv[a:max(b, a + 1)].max()), 3) for a, b in zip(edges[:-1], edges[1:])]
    data = {"duration": duration, "lines": [{"id": l["id"], "s": l["start"], "e": l["end"], "t": l["text"], "f": l["flag"],
                                             "r": l["reasons"], "v": l["verified"], "y": (eye or {}).get(l["id"], "")} for l in lines],
            "wave": wave,
            "speech": [[round(s, 2), round(e, 2)] for s, e in speech], "gaps": [[g["usable_start"], g["usable_end"]] for g in gaps]}
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Between The Lines player</title><style>
:root{{--bg:#f6f7f9;--fg:#1b2128;--mut:#5b6773;--card:#fff;--line:#dde2e8;--sp:#d9534f;--gap:#2e9e5b;--amb:#e0a100;--unv:#7d91af;--acc:#2f6fde}}
@media (prefers-color-scheme: dark){{:root{{--bg:#13161a;--fg:#e8ebee;--mut:#98a3ae;--card:#1c2127;--line:#2b323a;--acc:#6ea0ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:18px 16px 40px}}h1{{font-size:22px;margin:0}}.sub{{color:var(--mut);margin:2px 0 14px}}
video{{width:100%;border-radius:8px;background:#000;display:block}}
.bar canvas{{position:absolute;inset:0;width:100%;height:100%}}.bar{{position:relative;height:74px;margin-top:10px;background:var(--card);border:1px solid var(--line);border-radius:8px;cursor:pointer;overflow:hidden}}
.bar i{{position:absolute;top:0;height:12px;background:var(--sp)}}.bar b{{position:absolute;top:50px;height:24px;border-radius:2px}}
.bar u{{position:absolute;top:0;bottom:0;width:2px;background:var(--fg);pointer-events:none}}
.cap{{margin:12px 0;min-height:64px;padding:12px 14px;border-radius:8px;background:var(--card);border:1px solid var(--line);font-size:20px}}
.cap.on{{border-color:var(--gap)}}.cap.on.amber{{border-color:var(--amb)}}.cap.on.unv{{border-color:var(--unv)}}.cap small{{display:block;font-size:12px;color:var(--mut);margin-top:4px}}
.row{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}}.row label{{font-size:14px}}
button{{font:inherit;padding:5px 12px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}}
button.on{{background:var(--acc);color:#fff;border-color:var(--acc)}}
table{{border-collapse:collapse;width:100%;font-size:14px;margin-top:8px}}td,th{{border-bottom:1px solid var(--line);padding:5px 6px;text-align:left;vertical-align:top}}
tr.cur{{background:rgba(47,111,222,.12)}}tr{{cursor:pointer}}.tag{{display:inline-block;padding:0 8px;border-radius:9px;color:#fff;font-size:12px;font-weight:700}}
.G{{background:var(--gap)}}.A{{background:var(--amb)}}.U{{background:var(--unv)}}.leg{{font-size:12px;color:var(--mut)}}.leg span{{display:inline-block;width:10px;height:10px;margin:0 4px 0 10px;border-radius:2px}}
.foot{{color:var(--mut);font-size:12px;margin-top:22px;border-top:1px solid var(--line);padding-top:8px}}
</style></head><body><main>
<h1>Between The Lines <span style="font-weight:400;color:var(--mut)">: {html.escape(title)}</span></h1>
<p class="sub">{html.escape(stats['headline'])}</p>
<video id="v" controls playsinline preload="auto" src="ad.mp4"><track kind="descriptions" src="descriptions.vtt" srclang="en" label="Audio description draft" default></video>
<div class="row"><button id="b_ad" class="on">With audio description</button><button id="b_pl">Programme only</button>
<span class="leg"><span style="background:var(--sp)"></span>speech found by our own detector (do not talk here); grey = programme loudness<span style="background:var(--gap)"></span>description line
<span style="background:var(--unv)"></span>UNVERIFIED: nothing checkable<span style="background:var(--amb)"></span>AMBER: check first</span></div>
<div class="bar" id="bar"><canvas id="wv" width="1400" height="148"></canvas></div>
<div class="cap" id="cap">Press play: the description appears here while it is voiced in the pauses.</div>
<table id="tbl"><tr><th>Time</th><th>Draft line</th><th>Flag</th></tr></table>
<div class="foot">Footage: {html.escape(credit)}. Local prototype: Cosmos3-Edge stands in for Cosmos3-Reason, YOLO26s/26l for the DataEngine YOLO, Windows SAPI (en-GB Hazel) for the voice.</div>
</main><script>
const D={json.dumps(data)};
const v=document.getElementById('v'),cap=document.getElementById('cap'),bar=document.getElementById('bar'),tbl=document.getElementById('tbl');
const pct=t=>(100*t/D.duration).toFixed(3)+'%';
const COL={{GREEN:'var(--gap)',AMBER:'var(--amb)',UNVERIFIED:'var(--unv)'}},CLS={{GREEN:'',AMBER:' amber',UNVERIFIED:' unv'}},TG={{GREEN:'G',AMBER:'A',UNVERIFIED:'U'}};
D.speech.forEach(([s,e])=>{{const i=document.createElement('i');i.style.left=pct(s);i.style.width=pct(e-s);bar.appendChild(i)}});
D.lines.forEach(l=>{{const b=document.createElement('b');b.style.left=pct(l.s);b.style.width=pct(l.e-l.s);b.style.background=COL[l.f];b.title=l.t;bar.appendChild(b)}});
const u=document.createElement('u');bar.appendChild(u);
(function(){{const c=document.getElementById('wv'),g=c.getContext('2d'),n=D.wave.length;if(!n)return;g.fillStyle=getComputedStyle(document.body).color;g.globalAlpha=.45;
const w=c.width/n;D.wave.forEach((a,i)=>{{const h=a*88;g.fillRect(i*w,148-h-2,Math.max(1,w),h)}})}})();
bar.onclick=ev=>{{const r=bar.getBoundingClientRect();v.currentTime=D.duration*(ev.clientX-r.left)/r.width}};
const mmss=t=>Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
D.lines.forEach(l=>{{const r=tbl.insertRow();r.id='r'+l.id;r.onclick=()=>{{v.currentTime=Math.max(0,l.s-.5);v.play()}};
r.innerHTML='<td>'+mmss(l.s)+'</td><td>'+l.t.replace(/</g,'&lt;')+(l.r.length?'<br><small style="color:var(--mut)">'+l.r.join('; ').replace(/</g,'&lt;')+'</small>':'')+'</td><td><span class="tag '+TG[l.f]+'">'+l.f+'</span>'+(l.y?' <small style="color:var(--mut)">by eye: '+l.y+'</small>':'')+'</td>'}});
let cur=-1;
v.ontimeupdate=()=>{{const t=v.currentTime;u.style.left=pct(t);const l=D.lines.find(l=>t>=l.s-.05&&t<=l.e+.4);
 if(l){{cap.className='cap on'+CLS[l.f];cap.innerHTML=l.t.replace(/</g,'&lt;')+'<small>'+mmss(l.s)+' to '+mmss(l.e)+' · '+l.f+(l.r.length?' · '+l.r.join('; ').replace(/</g,'&lt;'):'')+'</small>'}}
 else if(cur!=-2){{cap.className='cap';cap.textContent=D.speech.some(([s,e])=>t>=s&&t<=e)?'Dialogue: describer stays silent.':'(pause: nothing to add)'}}
 const id=l?l.id:-1;if(id!=cur){{document.querySelectorAll('tr.cur').forEach(x=>x.classList.remove('cur'));if(l)document.getElementById('r'+l.id).classList.add('cur');cur=id}}}};
function sw(src,on){{const t=v.currentTime,p=!v.paused;v.src=src;v.currentTime=t;if(p)v.play();document.getElementById('b_ad').className=on?'on':'';document.getElementById('b_pl').className=on?'':'on'}}
document.getElementById('b_ad').onclick=()=>sw('ad.mp4',true);document.getElementById('b_pl').onclick=()=>sw('plain.mp4',false);
</script></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------------ hero

FONT_DIR = Path("C:/Windows/Fonts")


def _f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for n in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arialbd.ttf" if bold else "arial.ttf"):
        try:
            return ImageFont.truetype(str(FONT_DIR / n), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(d: ImageDraw.ImageDraw, text: str, font, width: int, max_lines: int = 4) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,") + " ..."
    return lines


def hero(path: Path, title: str, subtitle: str, duration: float, speech: list, gaps: list, lines: list[dict],
         level_curve: np.ndarray, picks: list[dict], stats: dict, frames: dict[int, np.ndarray], eye: dict | None = None,
         footer: str | None = None, spectro: tuple[int, int] | None = None) -> Path:
    W, H = 1280, 720
    BG, FG, MUT, CARD, LINE = (15, 19, 24), (240, 243, 246), (150, 162, 175), (27, 33, 40), (48, 56, 66)
    RED, GRN, AMB, BLU = (222, 84, 78), (50, 190, 110), (240, 175, 20), (96, 150, 255)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((36, 22), "BETWEEN THE LINES", font=_f(40, True), fill=FG)
    d.text((36, 72), subtitle, font=_f(17), fill=MUT)
    # verdict stamp: honest wording, the detector is our own
    over = stats["over_speech_s"]
    ok = over <= 0.0001
    sc = GRN if ok else RED
    x0, y0, x1, y1 = 640, 16, 1244, 100
    d.rounded_rectangle((x0, y0, x1, y1), 14, outline=sc, width=4, fill=(20, 40, 30) if ok else (50, 20, 20))
    d.text((x0 + 18, y0 + 4), (f"{over:.1f} s over detected speech" if not ok else "0 s over detected speech"), font=_f(30, True), fill=sc)
    sub = "own detector" + (f"  ·  spectrogram audit by eye: {spectro[0]} of {spectro[1]} lines clear, {spectro[1] - spectro[0]} unclear" if spectro else "  ·  not yet audited by eye")
    d.text((x0 + 20, y0 + 40), sub, font=_f(16), fill=FG)
    d.text((x0 + 20, y0 + 60), f"{stats['n_lines']} draft lines  ·  {stats['n_green']} GREEN  ·  {stats['n_unverified']} UNVERIFIED  ·  {stats['n_amber']} AMBER", font=_f(17, True), fill=FG)
    # timeline panel
    px0, px1, py = 36, W - 36, 126
    d.rounded_rectangle((px0 - 12, py - 10, px1 + 12, py + 188), 12, fill=CARD)
    sx = (px1 - px0) / duration
    d.text((px0, py), "PROGRAMME AUDIO   red = speech found by our own detector: the describer stays silent", font=_f(14, True), fill=MUT)
    # loudness curve + speech blocks
    ty, th = py + 22, 62
    d.rectangle((px0, ty, px1, ty + th), fill=(20, 25, 31))
    for s, e in speech:
        d.rectangle((px0 + s * sx, ty, px0 + max(px0 + s * sx + 2, e * sx + px0) - px0 if False else px0 + e * sx, ty + th), fill=(120, 44, 42))
    if len(level_curve):
        lv = np.clip((level_curve + 70) / 60, 0, 1)
        n = len(lv)
        pts = [(px0 + i / n * (px1 - px0), ty + th - 3 - lv[i] * (th - 8)) for i in range(0, n, max(1, n // 700))]
        d.line(pts, fill=(205, 212, 220), width=1)
    d.text((px0, ty + th + 8), "AI-DRAFTED DESCRIPTION, VOICED IN THE PAUSES", font=_f(14, True), fill=MUT)
    ly = ty + th + 30
    d.rectangle((px0, ly, px1, ly + 34), fill=(20, 25, 31))
    for g in gaps:
        d.rectangle((px0 + g["usable_start"] * sx, ly + 30, px0 + g["usable_end"] * sx, ly + 34), fill=(60, 80, 70))
    for l in lines:
        c = FLAG_RGB[l["flag"]]
        d.rounded_rectangle((px0 + l["start"] * sx, ly + 3, px0 + max(l["start"] * sx + 4, l["end"] * sx) + px0 - px0, ly + 28), 4, fill=c)
    for k in range(0, 11):
        t = duration * k / 10
        xx = px0 + t * sx
        d.line((xx, ly + 38, xx, ly + 44), fill=MUT)
        d.text((min(xx, px1 - 40), ly + 44), pk.mmss(t)[:5], font=_f(12), fill=MUT)
    # cards
    cw, ch, gap = 392, 352, 22
    cy = 336
    for i, p in enumerate(picks[:3]):
        cx = 36 + i * (cw + gap)
        d.rounded_rectangle((cx, cy, cx + cw, cy + ch), 14, fill=CARD)
        fr = frames.get(p["id"])
        if fr is not None:
            im = Image.fromarray(cv2.cvtColor(pk.fit(fr, cw - 20), cv2.COLOR_BGR2RGB))
            ih = min(im.size[1], 192)
            img.paste(im.crop((0, 0, im.size[0], ih)), (cx + 10, cy + 10))
        else:
            ih = 192
        d.rounded_rectangle((cx + 16, cy + 16, cx + 96, cy + 42), 6, fill=(0, 0, 0))
        d.text((cx + 24, cy + 17), pk.mmss(p["start"])[:5], font=_f(18, True), fill=FG)
        v = (eye or {}).get(p["id"])
        if v:
            vc = {"accurate": GRN, "partly": AMB, "wrong": RED}[v]
            txt = "by eye: " + v
            tw = d.textlength(txt, font=_f(16, True))
            d.rounded_rectangle((cx + cw - 30 - tw, cy + 16, cx + cw - 16, cy + 42), 6, fill=(0, 0, 0), outline=vc, width=2)
            d.text((cx + cw - 23 - tw, cy + 18), txt, font=_f(16, True), fill=vc)
        ty2 = cy + ih + 20
        for j, ln in enumerate(_wrap(d, p["text"], _f(21, True), cw - 32, 3)):
            d.text((cx + 16, ty2 + j * 27), ln, font=_f(21, True), fill=FG)
        col = FLAG_RGB[p["flag"]]
        label = ("AMBER  " + (p["reasons"][0] if p["reasons"] else "check")) if p["flag"] == "AMBER" else (
            "GREEN  a checked claim agrees with YOLO" if p["flag"] == "GREEN" else "UNVERIFIED  nothing here YOLO can check")
        d.rounded_rectangle((cx + 12, cy + ch - 46, cx + cw - 12, cy + ch - 12), 8, fill=(col[0] // 5, col[1] // 5, col[2] // 5), outline=col, width=2)
        lab = _wrap(d, label, _f(14, True), cw - 40, 1)[0]
        d.text((cx + 22, cy + ch - 40), lab, font=_f(14, True), fill=col)
    d.text((36, H - 26), footer or "Cosmos3 caption + YOLO count check + Windows en-GB voice, ducked under the programme. Local prototype.",
           font=_f(14), fill=MUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)
    return path
