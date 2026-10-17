"""BETWEEN THE LINES: first-draft audio description that fits the gaps in dialogue.

  python run.py --all                 default test set (elephants_dream first 180 s, yonge_dundas, dashcam_urban)
  python run.py --video NAME [--end S] [--vlm-cap N] [--no-vlm]
  python run.py --validate            SIMULATED dialogue-injection test of the speech detector -> out/vad_validation.json
  python run.py --report-only         rebuild results.json / reports from out/*/lines.json (+ eyeball.json) without recompute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
import protokit as pk  # noqa: E402

import audio as au  # noqa: E402
import describe as ds  # noqa: E402
import render as rd  # noqa: E402
import tts  # noqa: E402

OUT = HERE / "out"
DEFAULT_SET = [("elephants_dream", 180.0), ("yonge_dundas", None), ("dashcam_urban", None)]
GENRE = {"elephants_dream": "animated film", "yonge_dundas": "busy city street crossing, filmed from the pavement",
         "dashcam_urban": "drive through a city street, seen through the windscreen"}
VOICE_RATE = -1  # SAPI rate: Hazel at 0 speaks short lines at about 3.5 words/s; -1 is closer to AD pace


# ------------------------------------------------------------------------------------------------------ helpers


def keyframes(video: str, times: list[float]) -> list[np.ndarray]:
    out = []
    for t in times:
        try:
            out.append(pk.frame_at(video, max(0.0, t)))
        except ValueError:
            out.append(out[-1] if out else np.zeros((360, 640, 3), np.uint8))
    return out


def ev_at(tr: pk.Tracks, times: list[float]) -> list[dict[str, int]]:
    out = []
    for t in times:
        r = tr.at(t, tol=0.5)
        out.append({k: int(v) for k, v in r["name"].value_counts().items()} if len(r) else {})
    return out


def counts_from_det(dets: list[dict]) -> dict[str, int]:
    d: dict[str, int] = {}
    for x in dets:
        d[x["name"]] = d.get(x["name"], 0) + 1
    return d


def fit_line(text: str, slot_len: float, w0: int) -> tuple[str | None, str | None, float, int]:
    """Fit a draft into the slot, measured with the real TTS. Shorten (model rewrite, then clause-boundary trim) until it fits;
    returns text=None when no complete sentence fits. -> text, wav, seconds, passes"""
    w, passes = w0, 0
    if ds.n_words(text) > w:
        text = ds.shorten(text, w) or ds.trim_to_words(text, w)
        passes += 1
    for _ in range(5):
        if not text:
            return None, None, 0.0, passes
        wav, _w = tts.synth(text, rate=VOICE_RATE)
        dur = tts.active_duration(wav) + 0.05
        if dur <= slot_len - 0.05:
            return text, wav, dur, passes
        w = max(3, min(ds.n_words(text) - 2, int(ds.n_words(text) * (slot_len - 0.1) / dur)))
        text = ds.shorten(text, w) or ds.trim_to_words(text, w)
        passes += 1
    return None, None, 0.0, passes


def eyeball_sheet(path: Path, lines: list[dict], frames: dict[int, list[np.ndarray]], per_sheet: int = 6) -> list[Path]:
    """Review sheets (1280 px wide): one row per line = the 3 keyframes it was written from + flag + text."""
    paths = []
    fw, row_h = 280, 168
    f1, f2 = rd._f(17, True), rd._f(17)
    for sheet_i in range(0, len(lines), per_sheet):
        chunk = lines[sheet_i: sheet_i + per_sheet]
        img = Image.new("RGB", (1280, len(chunk) * row_h + 8), (18, 22, 27))
        d = ImageDraw.Draw(img)
        for k, ln in enumerate(chunk):
            y = 8 + k * row_h
            for j, fr in enumerate((frames.get(ln["id"]) or [])[:3]):
                im = cv2.resize(fr, (fw - 4, int((fw - 4) * fr.shape[0] / fr.shape[1])), interpolation=cv2.INTER_AREA)
                img.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (8 + j * fw, y))
            col = rd.FLAG_RGB[ln["flag"]]
            x0 = 8 + 3 * fw + 8
            d.text((x0, y), f"#{ln['id']}  {pk.mmss(ln['start'])[:5]}  {ln['flag']}", font=f1, fill=col)
            words, cur, yy = ln["text"].split(), "", y + 28
            for wd in words:
                t = (cur + " " + wd).strip()
                if d.textlength(t, font=f2) > 1280 - x0 - 8:
                    d.text((x0, yy), cur, font=f2, fill=(235, 238, 242))
                    yy += 24
                    cur = wd
                else:
                    cur = t
            d.text((x0, yy), cur, font=f2, fill=(235, 238, 242))
            d.text((x0, y + row_h - 36), "yolo26s " + json.dumps(ln["ev_s"][1]) + "  l " + json.dumps(ln["ev_l"][1]), font=rd._f(13), fill=(140, 152, 165))
        p = path.parent / f"{path.stem}_{sheet_i // per_sheet + 1}.jpg"
        img.save(p, quality=88)
        paths.append(p)
    return paths


def audit_figure(path: Path, wav16: Path, speech: list, silence: list, lines: list[dict], duration: float) -> Path:
    """Spectrogram (0-4 kHz) with detected speech (red) and drafted lines (green/amber) for the by-eye dialogue audit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.signal as sg
    x, sr = au.load_wav(wav16)
    f, t, S = sg.spectrogram(x, sr, nperseg=512, noverlap=384)
    fig, ax = plt.subplots(2, 1, figsize=(16, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    ax[0].pcolormesh(t, f[:130], 10 * np.log10(S[:130] + 1e-10), shading="auto", vmin=-105, vmax=-45, cmap="magma")
    ax[0].set_ylabel("Hz")
    ax[0].set_title("Programme spectrogram (0-4 kHz). Speech harmonics show as stacked stripes; red bar = detected speech")
    for s, e in speech:
        ax[1].axvspan(s, e, 0.55, 1.0, color="#d9534f")
    for ln in lines:
        ax[1].axvspan(ln["start"], ln["end"], 0.0, 0.45, color=rd.FLAG_HEX[ln["flag"]])
    for s, e in silence:
        ax[1].axvspan(s, e, 0.47, 0.53, color="#4a90d9")
    ax[1].set_yticks([0.22, 0.5, 0.78], ["lines", "silencedetect", "speech"])
    ax[1].set_xlim(0, duration)
    ax[1].set_xlabel("seconds")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def audit_lines_figure(od: Path, wav16: Path, speech: list, lines: list[dict], per_sheet: int = 7, pad: float = 2.0) -> list[Path]:
    """One spectrogram strip per line (line window +- 2 s, 0-3.4 kHz) for the by-eye 'is anything speech-like under this line' audit.
    Independent of the speech detector: I judge the strips myself and record verdicts in spectro_audit.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.signal as sg
    x, sr = au.load_wav(wav16)
    f, tt, S = sg.spectrogram(x, sr, nperseg=640, noverlap=480)
    paths = []
    for si in range(0, len(lines), per_sheet):
        chunk = lines[si: si + per_sheet]
        fig, axs = plt.subplots(len(chunk), 1, figsize=(16, 1.9 * len(chunk)), squeeze=False)
        for ax, ln in zip(axs[:, 0], chunk):
            a, b = max(0.0, ln["start"] - pad), ln["end"] + pad
            m = (tt >= a) & (tt <= b)
            ax.pcolormesh(tt[m], f[:110], 10 * np.log10(S[:110][:, m] + 1e-10), shading="auto", vmin=-100, vmax=-40, cmap="magma")
            ax.axvspan(ln["start"], ln["end"], ymin=0.93, ymax=1.0, color=rd.FLAG_HEX[ln["flag"]])
            ax.axvline(ln["start"], color="w", lw=0.8, ls="--")
            ax.axvline(ln["end"], color="w", lw=0.8, ls="--")
            for s, e in speech:
                if e > a and s < b:
                    ax.axvspan(max(s, a), min(e, b), ymin=0.0, ymax=0.06, color="#d9534f")
            ax.set_xticks(np.arange(np.ceil(a), b, 1.0))
            ax.set_xticklabels([f"{v:.0f}" for v in np.arange(np.ceil(a), b, 1.0)], fontsize=7)
            ax.set_yticks([])
            ax.set_ylabel(f"#{ln['id']}", rotation=0, labelpad=14, fontsize=11)
            ax.set_xlim(a, b)
        plt.tight_layout()
        pth = od / f"audit_lines_{si // per_sheet + 1}.jpg"
        fig.savefig(pth, dpi=70, pil_kwargs={"quality": 85})
        plt.close(fig)
        paths.append(pth)
    return paths


def spectro_verdicts(name: str, lines: list[dict]) -> dict[int, dict]:
    """My by-eye spectrogram verdicts per line (spectro_audit.json); stale ones (text changed) are ignored."""
    p = HERE / "spectro_audit.json"
    e = json.loads(p.read_text(encoding="utf-8")).get(name, {}) if p.exists() else {}
    out = {}
    for l in lines:
        v = e.get(str(l["start"]))
        if v and v.get("text_sha") == hashlib.sha256(l["text"].encode()).hexdigest()[:8]:
            out[l["id"]] = v
    return out


# ------------------------------------------------------------------------------------------------------ pipeline


def run_video(name: str, end: float | None = None, vlm_cap: int = 30, use_vlm: bool = True) -> dict:
    t_start = time.time()
    video = str(pk.video_path(name))
    meta = pk.info(name)
    T = min(meta["duration"], end) if end else meta["duration"]
    od = OUT / name
    od.mkdir(parents=True, exist_ok=True)
    work = od / "_work"
    work.mkdir(exist_ok=True)

    # 1. audio: speech + silence maps -> gaps
    wav16 = au.extract_wav(video, work / "prog16.wav", 0.0, T)
    x, _ = au.load_wav(wav16)
    r = au.find_gaps(x)
    gaps, speech, silence = r["gaps"], r["speech"], r["silence"]
    slots = ds.make_slots(gaps)

    # 2. visual evidence (cached YOLO26s tracks) + novelty gating
    tr = pk.tracks(name)
    cuts = ds.shot_cuts(video, 0.0, T)
    plan, prev_t_end, prev_frame, seen = [], 0.0, None, set()
    s_all = tr.summary(min_frames=4)
    for sl in slots:
        L = sl["end"] - sl["start"]
        w = min(ds.MAX_WORDS, max(3, int(L * ds.WPS)))  # brief budget: <= 2.5 words per usable slot second, <= 8 words
        dur_est = min(L, w / ds.WPS)
        times = [sl["start"] + 0.15, sl["start"] + dur_est / 2, sl["start"] + dur_est]
        frs = keyframes(video, times)
        nv = ds.new_since(tr, prev_t_end, sl["end"], seen)
        sc = ds.scene_change(prev_frame, frs[1]) if prev_frame is not None else 1.0
        ncut = sum(1 for c in cuts if prev_t_end < c <= sl["end"])
        novelty = min(1.0, 3.0 * sc) + 0.15 * min(ncut, 4) + 0.08 * min(nv["new_tracks"], 5)
        sl.update({"words": w, "times": [round(t, 2) for t in times], "novelty": round(novelty, 3), "scene_change": round(sc, 3),
                   "cuts": ncut, "new": nv, "frames": frs})
        dark = all(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()) < 10 for f in frs)  # black frame: nothing to describe
        sl["dark"] = bool(dark)
        describe_it = (prev_frame is None or novelty >= 0.35 or ncut >= 1) and not dark
        sl["described"] = bool(describe_it)
        if describe_it:
            prev_t_end, prev_frame = sl["end"], frs[1]
            if not s_all.empty:
                seen |= set(s_all[(s_all["first"] <= sl["end"])]["tid"].tolist())
    cand = [s for s in slots if s["described"]]
    dropped = 0
    if len(cand) > vlm_cap:
        keep = sorted(sorted(cand, key=lambda s: -s["novelty"])[:vlm_cap], key=lambda s: s["id"])
        for s in cand:
            if s not in keep:
                s["described"], dropped = False, dropped + 1
        cand = keep

    # 3. independent second detector (YOLO26l) on every keyframe of described slots (one batch, GPU lock)
    allf = [f for s in cand for f in s["frames"]]
    dlc = work / "detl_cache.json"
    key = hashlib.sha256(json.dumps([[round(t, 2) for t in s["times"]] for s in cand]).encode()).hexdigest()[:12]
    cached = json.loads(dlc.read_text()) if dlc.exists() else {}
    if cached.get("key") == key:
        dl = cached["dets"]
    else:
        dl = pk.detect_many(allf, model="det_l", conf=0.3) if allf else []
        dlc.write_text(json.dumps({"key": key, "dets": [[{k: (v if isinstance(v, (str, float)) else [float(z) for z in v]) for k, v in d.items()} for d in f] for f in dl]}))
    for i, s in enumerate(cand):
        s["dets_l"] = dl[3 * i: 3 * i + 3]
        s["ev_l"] = [counts_from_det(d) for d in dl[3 * i: 3 * i + 3]]
        s["ev_s"] = ev_at(tr, s["times"])

    # 4. describe -> fit -> flag
    lines, prev_line, vlm_calls, fits, unfit = [], None, 0, 0, []
    genre = GENRE.get(name, "video")
    n_titles = 0
    for si, s in enumerate(cand):
        L = s["end"] - s["start"]
        text, conf, raw = (None, None, "")
        if use_vlm:
            print(f"  [{name}] slot {s['id']} t={s['start']:.1f}-{s['end']:.1f} -> VLM (calls so far {vlm_calls}, slot {si + 1}/{len(cand)})", flush=True)
            seen_things = sum(v for ev in s["ev_s"] + s["ev_l"] for k, v in ev.items() if k in ("person", "car", "truck", "bus", "bicycle", "motorcycle"))
            # a title card / caption is voiced first; only probed where no person/vehicle is detected (a billboard in a crowd is not a title)
            if seen_things == 0 and vlm_calls + (len(cand) - si) + 1 <= vlm_cap:
                shown = ds.read_text(pk.fit(s["frames"][1], 768))
                vlm_calls += 1
                if shown:
                    text, conf, raw = ds.title_line(shown), "medium", "title"
                    n_titles += 1
                    s["on_screen_text"] = shown
            if not text:
                text, conf, raw = ds.describe([pk.fit(f, 512) for f in s["frames"]], s["words"], prev_line, genre)
                vlm_calls += 1
        fallback = False
        if not text and use_vlm and raw:  # the model answered but talked about the pictures instead of the scene: stay silent
            unfit.append({"slot": s["id"], "slot_len": round(L, 2), "draft": raw[:120], "trimmed": None})
            print(f"    slot {s['id']}: model reply was not a scene description -> left silent", flush=True)
            continue
        if not text:
            text, conf, fallback = ds.fallback_line(s["ev_s"][1]), "low", True
        text0 = text
        text, wav, dur, passes = fit_line(text, L, s["words"])
        if text is None or (use_vlm and not fallback and not ds.is_title(text) and (not ds.has_verb(text) or ds.ends_badly(text))):
            unfit.append({"slot": s["id"], "slot_len": round(L, 2), "draft": text0, "trimmed": text})
            print(f"    slot {s['id']}: no complete sentence fits {L:.1f} s -> left silent (draft: {text0!r})", flush=True)
            continue
        fits += passes > 0
        chk = ds.check_line(text, s["ev_s"], s["ev_l"], conf, prev_line)
        boxes = [[(b.name, [b.x1, b.y1, b.x2, b.y2]) for b in tr.at(t, tol=0.5).itertuples()] +
                 [(d["name"], d["box"]) for d in s["dets_l"][k]] for k, t in enumerate(s["times"])]
        c_reasons, c_checked = ds.check_colours(text, s["frames"], boxes)
        p_reasons, p_checked = ds.check_positions(text, boxes, float(s["frames"][0].shape[1]))
        c_reasons, c_checked = c_reasons + p_reasons, c_checked + p_checked
        if c_reasons:
            chk["reasons"] += c_reasons
            chk["flag"] = "AMBER"
        chk["claims_checked"] += c_checked
        chk["verified"] = chk["claims_checked"] > 0
        chk["colour_claims"] = ds.colour_claims(text)
        if fallback:
            chk["reasons"] = chk["reasons"] + ["template fallback (VLM unavailable)"]
        # AMBER if anything disagrees / is doubtful; GREEN only if a checkable claim was confirmed; else UNVERIFIED
        chk["flag"] = ds.final_flag(bool(chk["reasons"]), chk["claims_checked"])
        ln = {"id": len(lines), "slot": s["id"], "gap": s["gap"], "start": round(s["start"], 3), "end": round(s["start"] + dur, 3),
              "text": text, "draft": text0, "words": ds.n_words(text), "wps": round(ds.n_words(text) / dur, 2), "fit_passes": passes,
              "wav": wav, "slot_len": round(L, 2), "wps_slot": round(ds.n_words(text) / L, 2), "conf": conf, "novelty": s["novelty"], "new": s["new"], "cuts": s["cuts"],
              "ev_s": s["ev_s"], "ev_l": s["ev_l"], "kind": s["kind"], **chk}
        lines.append(ln)
        prev_line = text

    # 5. media: plain cut, ducked mix, VTT, player
    plain = rd.cut_plain(video, 0.0, T, od / "plain.mp4")
    ad = od / "ad.mp4"
    rd.mix_ad(plain, lines, ad, work, work / "duck16.wav")
    duck = rd.ducking_depth(wav16, work / "duck16.wav", lines) if lines else {}
    rd.write_vtt(od / "descriptions.vtt", lines)

    # 6. measurements
    over_speech = sum(au.overlap((l["start"], l["end"]), speech) for l in lines)
    in_silence = sum(au.overlap((l["start"], l["end"]), silence) for l in lines)
    line_s = sum(l["end"] - l["start"] for l in lines)
    over_inner = sum(au.overlap((l["start"], l["end"]), [(g["start"], g["end"])]) for l in lines for g in [gaps[l["gap"]]])
    gap_s = sum(g["usable"] for g in gaps)
    n_g = sum(l["flag"] == "GREEN" for l in lines)
    n_u = sum(l["flag"] == "UNVERIFIED" for l in lines)
    n_a = sum(l["flag"] == "AMBER" for l in lines)
    stats = {
        "video": name, "analysed_s": round(T, 1), "audio_speech_frac": r["speech_frac"], "gaps": len(gaps), "gap_usable_s": round(gap_s, 1),
        "strict_silence_only_gap_s": r["strict_gap_s"], "gaps_quiet": sum(g["kind"] == "quiet" for g in gaps), "gaps_bed": sum(g["kind"] == "bed" for g in gaps),
        "slots": len(slots), "slots_described": len(cand), "slots_held": len(slots) - len(cand), "slots_dropped_by_cap": dropped, "slots_unfittable": len(unfit), "unfittable": unfit,
        "n_lines": len(lines), "n_green": n_g, "n_amber": n_a, "n_unverified": n_u, "n_verified": sum(l["verified"] for l in lines),
        "words": sum(l["words"] for l in lines), "line_seconds": round(line_s, 1),
        "mean_wps": round(sum(l["words"] for l in lines) / max(line_s, 1e-6), 2),
        "max_words": max([l["words"] for l in lines], default=0), "max_wps_slot": max([l["wps_slot"] for l in lines], default=0),
        "mean_wps_slot": round(sum(l["words"] for l in lines) / max(sum(l["slot_len"] for l in lines), 1e-6), 2), "lines_trimmed_to_fit": int(fits),
        "over_speech_s": round(over_speech, 3), "lines_overlapping_speech": sum(
            au.overlap((l["start"], l["end"]), speech) > 0 for l in lines),
        "lines_inside_own_gap": sum(l["start"] >= gaps[l["gap"]]["usable_start"] - 1e-6 and l["end"] <= gaps[l["gap"]]["usable_end"] + 1e-6
                                    for l in lines),
        "line_seconds_in_silencedetect_silence": round(in_silence, 1), "line_seconds_over_music_or_ambience": round(line_s - in_silence, 1),
        "silencedetect_threshold_db": round(r["silence_thr_db"], 1), "ducking_db": duck, "vlm_calls": vlm_calls, "title_lines": n_titles,
        "n_shot_cuts": len(cuts), "seconds": round(time.time() - t_start, 1), "headline": ""}
    stats["headline"] = (f"{stats['n_lines']} lines drafted into {stats['gaps']} dialogue gaps ({stats['gap_usable_s']} s usable); "
                         f"{stats['over_speech_s']} s over detected speech; {stats['n_green']} GREEN / {stats['n_amber']} AMBER / {stats['n_unverified']} UNVERIFIED.")

    # 7. persist + visuals
    level = au.rms_db(x)
    picks = pick_hero_lines(lines)
    frame_by_id = {l["id"]: next(s for s in slots if s["id"] == l["slot"])["frames"][1] for l in lines}
    for l in lines:
        pk.save_jpg(pk.fit(frame_by_id[l["id"]], 480), od / "frames" / f"line_{l['id']:03d}.jpg", 85)
    save_lines = [{k: v for k, v in l.items() if k != "wav"} for l in lines]
    (od / "lines.json").write_text(json.dumps({"stats": stats, "lines": save_lines, "gaps": gaps, "speech": speech,
                                                "silence": silence, "cuts": cuts,
                                                "slots": [{k: v for k, v in s.items() if k not in ("frames",)} for s in slots]},
                                               indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    sheets = eyeball_sheet(od / "eyeball", lines, {l["id"]: next(s for s in slots if s["id"] == l["slot"])["frames"] for l in lines})
    audit_figure(od / "audit_audio.png", wav16, speech, silence, lines, T)
    for old_p in od.glob("audit_lines_*.jpg"):
        old_p.unlink()
    audit_lines_figure(od, wav16, speech, lines)
    rd.hero(od / "hero.jpg", {"elephants_dream": "Elephants Dream", "yonge_dundas": "Yonge-Dundas scramble", "dashcam_urban": "Urban dashcam"}.get(name, name),
            f"First-draft audio description fitted to dialogue pauses  ·  {name}", T, speech, gaps, lines,
            level, picks, stats, {p["id"]: frame_by_id[p["id"]] for p in picks})
    rd.write_player(od / "player.html", name, name.replace("_", " "), T, lines, speech, gaps, pk.credit(name), stats, level)
    build_report(od, name, T, stats, lines, gaps, speech, silence, sheets)
    rebuild(name)  # hero/player/report again, now with by-eye verdicts when eyeball.json covers these lines
    return stats


def rebuild(name: str) -> None:
    """Regenerate hero / player / report from out/<name>/lines.json (no recompute of audio, YOLO or VLM)."""
    od = OUT / name
    j = json.loads((od / "lines.json").read_text(encoding="utf-8"))
    stats, lines, gaps, speech, silence = j["stats"], j["lines"], j["gaps"], [tuple(s) for s in j["speech"]], [tuple(s) for s in j["silence"]]
    T = stats["analysed_s"]
    x, _ = au.load_wav(od / "_work" / "prog16.wav")
    eye = eye_verdicts(name, lines)
    picks = pick_hero_lines(lines, eye)
    frames = {p["id"]: cv2.imread(str(od / "frames" / f"line_{p['id']:03d}.jpg")) for p in picks}
    n = len(lines)
    sv = spectro_verdicts(name, lines)
    clear = sum(v["verdict"] == "clear" for v in sv.values())
    foot = (f"By eye on the keyframes: {sum(v == 'accurate' for v in eye.values())} accurate, {sum(v == 'partly' for v in eye.values())} partly, "
            f"{sum(v == 'wrong' for v in eye.values())} wrong of {n}.  Speech check: own detector"
            + (f" + my spectrogram audit ({clear}/{len(sv)} lines clear, {len(sv) - clear} unclear: no speech-like structure seen in any)." if sv else " only (no real dialogue transcript locally).")
            if eye else None)
    title = {"elephants_dream": "Elephants Dream", "yonge_dundas": "Yonge-Dundas scramble", "dashcam_urban": "Urban dashcam"}.get(name, name)
    rd.hero(od / "hero.jpg", title, f"First-draft audio description fitted to dialogue pauses  ·  {name}", T, speech, gaps, lines,
            au.rms_db(x), picks, stats, frames, eye, foot, spectro=(clear, len(sv)) if sv else None)
    rd.write_player(od / "player.html", name, name.replace("_", " "), T, lines, speech, gaps, pk.credit(name), stats, au.rms_db(x), eye)
    sheets = sorted(od.glob("eyeball_*.jpg"))
    build_report(od, name, T, stats, lines, gaps, speech, silence, sheets, sv)


def make_demo(name: str, secs: float = 60.0) -> dict | None:
    """Cut the `secs`-long window of ad.mp4 with the best by-eye record (accurate +3, partly +1, wrong -5; needs >= 2 lines; ties: more
    lines, then earliest) into demo_60s.mp4. Without verdicts it falls back to the most lines."""
    import subprocess
    od = OUT / name
    j = json.loads((od / "lines.json").read_text(encoding="utf-8"))
    T = j["stats"]["analysed_s"]
    if T < secs or not j["lines"]:
        return None
    eye = eye_verdicts(name, j["lines"])
    pts = {"accurate": 3, "partly": 1, "wrong": -5}
    cands = []
    for a in np.arange(0, T - secs + 1e-6, 1.0):
        inside = [l for l in j["lines"] if a <= l["start"] and l["end"] <= a + secs]
        if len(inside) >= 2:
            cands.append((sum(pts.get(eye.get(l["id"]), 0) for l in inside) if eye else 0, len(inside), -float(a)))
    if not cands:
        return None
    best = max(cands)
    a = -best[2]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.2f}", "-i", str(od / "ad.mp4"), "-t", f"{secs:.2f}", "-c:v", "libx264", "-preset",
                    "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    str(od / "demo_60s.mp4")], check=True)
    inside = [l for l in j["lines"] if a <= l["start"] and l["end"] <= a + secs]
    return {"window_start": round(float(a), 1), "lines": len(inside), "amber": sum(l["flag"] == "AMBER" for l in inside),
            "by_eye": {k: sum(eye.get(l["id"]) == k for l in inside) for k in ("accurate", "partly", "wrong")},
            "line_ids": [l["id"] for l in inside], "selected_by": "by-eye verdicts" if eye else "line count"}


def eye_verdicts(name: str, lines: list[dict]) -> dict[int, str]:
    """My by-eye verdicts (eyeball.json) for the current lines; stale ones (text changed) are ignored."""
    p = HERE / "eyeball.json"
    e = json.loads(p.read_text(encoding="utf-8")).get(name, {}) if p.exists() else {}
    out = {}
    for l in lines:
        v = e.get(str(l["start"]))
        if v and v.get("text_sha") == hashlib.sha256(l["text"].encode()).hexdigest()[:8]:
            out[l["id"]] = v["verdict"]
    return out


def pick_hero_lines(lines: list[dict], eye: dict | None = None) -> list[dict]:
    """Three cards, shown WITH the by-eye verdict so the hero cannot flatter: the best line (by eye, GREEN first), one flagged line
    (preferably an AMBER that is really wrong), and one more spread in time."""
    if not lines:
        return []
    eye = eye or {}
    rank = {"accurate": 0, "partly": 1, "wrong": 2}
    picks: list[dict] = []
    amber = sorted([l for l in lines if l["flag"] == "AMBER"], key=lambda l: (eye.get(l["id"], "accurate") == "accurate", -len(l["reasons"])))
    if amber:
        picks.append(amber[0])
    flagrank = {"GREEN": 0, "UNVERIFIED": 1, "AMBER": 2}
    acc = sorted([l for l in lines if l["flag"] != "AMBER"], key=lambda l: (rank.get(eye.get(l["id"], "partly"), 1), flagrank[l["flag"]], -l["words"]))
    for l in acc:
        if len(picks) >= 3:
            break
        if all(abs(l["start"] - p["start"]) > 8 for p in picks):
            picks.append(l)
    for l in lines:
        if len(picks) >= 3:
            break
        if l not in picks:
            picks.append(l)
    return sorted(picks, key=lambda l: l["id"])


def build_report(od: Path, name: str, T: float, stats: dict, lines: list[dict], gaps: list, speech: list, silence: list,
                 sheets: list[Path], sv: dict | None = None) -> Path:
    rep = pk.Report(od / "report.html", f"Between The Lines: {name}", stats["headline"])
    rep.credit(name)
    rep.image(od / "hero.jpg", "The 5-second story: dialogue in red, description lines in the pauses, flagged lines in amber.")
    rep.kv({"dialogue gaps": stats["gaps"], "usable gap seconds": stats["gap_usable_s"], "lines drafted": stats["n_lines"],
            "over detected speech (s, own detector)": stats["over_speech_s"], "GREEN": stats["n_green"], "UNVERIFIED": stats["n_unverified"], "AMBER": stats["n_amber"], "words per slot-second (mean / max)": f"{stats['mean_wps_slot']} / {stats['max_wps_slot']}", "spoken words/s": stats["mean_wps"],
            "programme ducked by (dB)": (stats["ducking_db"] or {}).get("mean_db", "n/a"), "VLM calls": stats["vlm_calls"]})
    rep.h2("Listen and watch")
    rep.html('<p><a href="player.html">Open the interactive player</a> (captions, timeline, with/without description toggle).</p>'
             f'<figure><video controls playsinline preload="metadata" src="ad.mp4"><track kind="descriptions" src="descriptions.vtt" '
             f'srclang="en" label="AD draft" default></video><figcaption>ad.mp4: programme audio ducked under each line (Windows en-GB voice, '
             f'mixed with ffmpeg). Plain programme: plain.mp4. WebVTT: descriptions.vtt.</figcaption></figure>')
    spans = [(s, e, "#d9534f", f"speech {pk.mmss(s)}-{pk.mmss(e)}") for s, e in speech]
    spans += [(l["start"], l["end"], rd.FLAG_HEX[l["flag"]], f"{pk.mmss(l['start'])} {l['flag']}: {l['text']}") for l in lines]
    rep.h2("Timeline")
    rep.timeline(sorted(spans), T, "Red = speech found by our own detector; green = line with a checked claim that agreed; grey-blue = UNVERIFIED (nothing checkable); amber = AMBER (check first).")
    rep.h2("Dialogue audit (spectrogram)")
    rep.image(od / "audit_audio.png", "Detected speech (red), silencedetect silence (blue), lines (green/amber) over the programme spectrogram.")
    rep.h2("Spectrogram audit of every line (my eyes, not the detector)")
    for pth in sorted(od.glob("audit_lines_*.jpg")):
        rep.image(pth, "Each strip: line window +- 2 s; coloured bar = the line; red base bar = detector speech. Verdicts: spectro_audit.json.")
    rep.h2("Drafted lines")
    rows = []
    for l in lines:
        rows.append({"#": l["id"], "time": f"{pk.mmss(l['start'])} - {pk.mmss(l['end'])}", "line": l["text"], "flag": l["flag"],
                     "spectrogram (by eye)": (sv or {}).get(l["id"], {}).get("verdict", ""),
                     "why": "; ".join(l["reasons"]) or ("counts verified" if l["verified"] else "no checkable counts"),
                     "yolo26s/26l (mid keyframe)": json.dumps({**l["ev_s"][1]}) + " / " + json.dumps(l["ev_l"][1]),
                     "novelty": l["novelty"], "words": l["words"]})
    rep.table(rows, max_rows=100)
    rep.h2("Keyframes behind each line (for review)")
    for s in sheets:
        rep.image(s, "")
    rep.h2("Stats")
    rep.json({k: v for k, v in stats.items() if k != "headline"})
    return rep.save()


# ------------------------------------------------------------------------------------------------------ results


def collect_results(extra: dict | None = None) -> dict:
    res = {"idea": "Between The Lines", "videos": {}, "generated": time.strftime("%Y-%m-%d %H:%M")}
    ey_path = HERE / "eyeball.json"
    eyes = json.loads(ey_path.read_text(encoding="utf-8")) if ey_path.exists() else {}
    for d in sorted(OUT.iterdir()) if OUT.exists() else []:
        f = d / "lines.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        st = j["stats"]
        e = eyes.get(d.name, {})
        ver = {}
        stale = 0
        for l in j["lines"]:
            v = e.get(str(l["start"]))
            if v and v.get("text_sha") == hashlib.sha256(l["text"].encode()).hexdigest()[:8]:
                ver[l["id"]] = v["verdict"]
            else:
                stale += 1
        n = len(j["lines"])
        acc = sum(v == "accurate" for v in ver.values())
        part = sum(v == "partly" for v in ver.values())
        wrong = sum(v == "wrong" for v in ver.values())
        amber_ids = {l["id"] for l in j["lines"] if l["flag"] == "AMBER"}
        flag_of = {l["id"]: l["flag"] for l in j["lines"]}
        by_flag = {f: {"n": sum(1 for l in j["lines"] if l["flag"] == f), "accurate": sum(1 for i, v in ver.items() if flag_of[i] == f and v == "accurate"),
                       "partly": sum(1 for i, v in ver.items() if flag_of[i] == f and v == "partly"),
                       "wrong": sum(1 for i, v in ver.items() if flag_of[i] == f and v == "wrong")} for f in ("GREEN", "UNVERIFIED", "AMBER")}
        st["eyeball"] = {"judged": len(ver), "unjudged_or_stale": stale, "accurate": acc, "partly": part, "wrong": wrong,
                         "accurate_frac": round(acc / n, 3) if n else None,
                         "accurate_or_partly_frac": round((acc + part) / n, 3) if n else None,
                         "amber_flags_on_inaccurate": sum(1 for i, v in ver.items() if v != "accurate" and i in amber_ids),
                         "inaccurate_total": sum(1 for v in ver.values() if v != "accurate"),
                         "green_lines_accurate": by_flag["GREEN"]["accurate"], "green_total": by_flag["GREEN"]["n"], "by_flag": by_flag}
        res["videos"][d.name] = st
    tot_lines = sum(v["n_lines"] for v in res["videos"].values())
    res["totals"] = {"lines": tot_lines, "over_speech_s": round(sum(v["over_speech_s"] for v in res["videos"].values()), 3),
                     "amber": sum(v["n_amber"] for v in res["videos"].values()),
                     "eyeball_accurate_frac": round(sum(v["eyeball"]["accurate"] for v in res["videos"].values()) / max(tot_lines, 1), 3)}
    ev = [v["eyeball"] for v in res["videos"].values()]
    n_l = max(tot_lines, 1)
    strict, lenient = sum(e["accurate"] for e in ev) / n_l, sum(e["accurate"] + e["partly"] for e in ev) / n_l
    sa = HERE / "spectro_audit.json"
    sp = json.loads(sa.read_text(encoding="utf-8")) if sa.exists() else {}
    sp_tot = {"clear": 0, "ambiguous": 0, "speech-like": 0, "judged": 0}
    for vn, vd in res["videos"].items():
        j = json.loads((OUT / vn / "lines.json").read_text(encoding="utf-8"))
        sv = spectro_verdicts(vn, j["lines"])
        vd["spectro_audit_by_eye"] = {k: sum(v["verdict"] == k for v in sv.values()) for k in ("clear", "ambiguous", "speech-like")}
        vd["spectro_audit_by_eye"]["judged"] = len(sv)
        for k in ("clear", "ambiguous", "speech-like"):
            sp_tot[k] += vd["spectro_audit_by_eye"][k]
        sp_tot["judged"] += len(sv)
    bf = {f: {"n": sum(v["eyeball"]["by_flag"][f]["n"] for v in res["videos"].values()),
              "accurate": sum(v["eyeball"]["by_flag"][f]["accurate"] for v in res["videos"].values()),
              "partly": sum(v["eyeball"]["by_flag"][f]["partly"] for v in res["videos"].values()),
              "wrong": sum(v["eyeball"]["by_flag"][f]["wrong"] for v in res["videos"].values())} for f in ("GREEN", "UNVERIFIED", "AMBER")}
    res["flag_calibration"] = bf
    res["spectrogram_audit_by_eye"] = sp_tot
    res["kill_test"] = {
        "criterion": "zero lines overlap dialogue AND >= 60 % of lines judged accurate by eye",
        "lines_overlapping_detected_speech": sum(v["lines_overlapping_speech"] for v in res["videos"].values()),
        "overlap_seconds_vs_speech_map": res["totals"]["over_speech_s"],
        "line_seconds_inside_silencedetect_silence": round(sum(v["line_seconds_in_silencedetect_silence"] for v in res["videos"].values()), 1),
        "line_seconds_total": round(sum(v["line_seconds"] for v in res["videos"].values()), 1),
        "eyeball_accurate_strict": round(strict, 3), "eyeball_accurate_or_partly": round(lenient, 3),
        "spectrogram_audit_by_eye": sp_tot,
        "dialogue_part_passed": res["totals"]["over_speech_s"] == 0.0 and sp_tot["speech-like"] == 0, "accuracy_part_passed_strict": strict >= 0.6,
        "passed": res["totals"]["over_speech_s"] == 0.0 and sp_tot["speech-like"] == 0 and strict >= 0.6}
    vv = OUT / "vad_validation.json"
    if vv.exists():
        res["vad_validation_SIMULATED"] = json.loads(vv.read_text(encoding="utf-8"))
    res.update(extra or {})
    return res


def run_validation() -> dict:
    import validate as va
    out = {}
    for nm, (src, t0, t1) in {"ED_music_sfx_27-147": ("elephants_dream", 27, 147), "yonge_ambient": ("yonge_dundas", 0, 57),
                              "dashcam_engine": ("dashcam_urban", 0, 120)}.items():
        p = HERE / "_work" / f"bed_{nm}.wav"
        au.extract_wav(pk.video_path(src), p, t0, t1)
        bed, _ = au.load_wav(p)
        base = au.find_gaps(bed)
        out[nm] = {"no_injection": {"speech_frac": base["speech_frac"], "gaps": len(base["gaps"])},
                   "injected": [va.evaluate(bed, g) for g in (-3, 0, 6)]}
    (OUT / "vad_validation.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                             encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--end", type=float, default=None, help="analyse only the first S seconds")
    ap.add_argument("--vlm-cap", type=int, default=30)
    ap.add_argument("--no-vlm", action="store_true", help="detector-only template lines")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--rebuild", help="comma list of analysed videos: regenerate hero/player/report from lines.json")
    ap.add_argument("--demo", help="cut the 60 s demo window for this analysed video")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    if a.validate:
        run_validation()
    if a.all:
        for name, end in DEFAULT_SET:
            st = run_video(name, end, a.vlm_cap, not a.no_vlm)
            print(f"{name}: {st['headline']} ({st['seconds']} s)")
            if name in ("elephants_dream", "dashcam_urban"):
                print("  demo window:", make_demo(name))
    elif a.video:
        st = run_video(a.video, a.end, a.vlm_cap, not a.no_vlm)
        print(f"{a.video}: {st['headline']} ({st['seconds']} s)")
    if a.rebuild:
        for n in a.rebuild.split(","):
            rebuild(n)
    if a.demo:
        print(a.demo, make_demo(a.demo))
    if a.all or a.video or a.report_only or a.validate or a.rebuild or a.demo:
        pk.write_results(HERE / "results.json", collect_results({"run_seconds": round(time.time() - t0, 1)}))
        print("results.json written")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
