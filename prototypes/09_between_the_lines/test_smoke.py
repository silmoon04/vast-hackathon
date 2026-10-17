"""Fast invariant checks (<60 s) on the produced outputs. Run: python test_smoke.py   (needs `python run.py --all` first)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))
import audio as au  # noqa: E402
import describe as ds  # noqa: E402

OUT = HERE / "out"
VIDEOS = ["elephants_dream", "yonge_dundas", "dashcam_urban"]
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def decode(mp4: Path, td: Path, tag: str) -> np.ndarray:
    w = td / f"{tag}.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(w)], check=True)
    return au.load_wav(w)[0]


def parse_vtt(p: Path) -> list[tuple[str, float, float, str]]:
    cues, txt = [], p.read_text(encoding="utf-8")
    for blk in re.split(r"\n\n+", txt):
        m = re.search(r"^(AD-\d+-\w+)\n(\d+):(\d+):([\d.]+) --> (\d+):(\d+):([\d.]+)\n(.+)$", blk.strip(), re.M | re.S)
        if m:
            g = m.groups()
            cues.append((g[0], int(g[1]) * 3600 + int(g[2]) * 60 + float(g[3]), int(g[4]) * 3600 + int(g[5]) * 60 + float(g[6]), g[7]))
    return cues


def unit_tests() -> None:
    c = ds.parse_claims("Two men walk past a white car.")
    check({(x["cls"], x["n"]) for x in c} == {("person", 2), ("car", 1)}, "parse_claims: 'two men ... a white car'")
    check(ds.parse_claims("The man leans out.") == [], "parse_claims: definite 'the man' is not a count claim")
    ev = [{"person": 1}, {"person": 2}, {"person": 1}]
    bad = ds.check_line("Five people cross the street.", ev, ev, "high", None)
    check(bad["flag"] == "AMBER" and bad["reasons"], "AMBER when line says five people and YOLO sees 1-2")
    good = ds.check_line("Two people cross the street.", ev, ev, "high", None)
    check(good["flag"] == "GREEN" and good["verified"], "GREEN when 'two people' matches YOLO 1-2")
    check(ds.check_line("The street is empty.", ev, ev, "high", None)["flag"] == "UNVERIFIED", "UNVERIFIED when the line has nothing YOLO can check")
    check(ds.check_line("A man waves.", ev, ev, "low", None)["flag"] == "AMBER", "AMBER when the model reports low confidence")
    check(ds.check_line("A car appears to turn.", [{"car": 1}] * 3, [{"car": 1}] * 3, "high", None)["flag"] == "AMBER", "AMBER on hedged wording")
    t = ds.trim_to_words("A man in a brown coat walks toward a woman on a platform.", 7)
    check(t is not None and ds.n_words(t) <= 7 and t.endswith(".") and ds.has_verb(t), f"trim_to_words cuts at a clause boundary and keeps a verb ({t!r})")
    check(ds.trim_to_words("A woman in a tattered cloak walks across a shattered bridge.", 3) is None, "trim_to_words refuses to voice a fragment (returns None)")
    nocar = ds.check_line("A car passes by.", [{"person": 2}] * 3, [{"person": 2}] * 3, "high", None)
    check(nocar["flag"] == "AMBER", "AMBER when the line names a car and neither YOLO model sees one")
    cl = ds.colour_claims("The blue car turns right while a woman in a pink top waits.")
    check({(c["cls"], c["colour"]) for c in cl} == {("car", "blue"), ("person", "pink")}, "colour_claims: blue car, pink top")
    fr = np.zeros((100, 100, 3), np.uint8)
    fr[:] = (30, 30, 200)  # BGR red
    why, n = ds.check_colours("A blue car.", [fr], [[("car", [0, 0, 100, 100])]])
    check(n == 1 and why, "colour check: a red box contradicts 'blue car' (pixel test)")
    why, n = ds.check_colours("A red car.", [fr], [[("car", [0, 0, 100, 100])]])
    check(n == 1 and not why, "colour check: a red box supports 'red car'")
    pc = ds.position_claims("A white car drives past on the right.")
    check(pc == [{"cls": "car", "side": "right", "text": "car on the right"}], "position_claims: 'white car ... on the right'")
    why, n = ds.check_positions("A car drives past on the right.", [[("car", [10, 0, 50, 40])]], 200.0)
    check(n == 1 and why, "position check: a car at the left edge contradicts 'on the right'")
    why, n = ds.check_positions("A car drives past on the right.", [[("car", [150, 0, 190, 40])]], 200.0)
    check(n == 1 and not why, "position check: a car at the right edge supports 'on the right'")
    check(ds.final_flag(False, 0) == "UNVERIFIED" and ds.final_flag(False, 2) == "GREEN" and ds.final_flag(True, 2) == "AMBER",
          "flags: nothing checkable = UNVERIFIED, checked and agreed = GREEN, any doubt = AMBER")
    check(ds.strip_filler("A man walks and the scene remains static throughout the scene.") == "A man walks." and
          ds.strip_filler("A textured floor with debris remains static throughout the scene.") == "", "filler 'remains static throughout the scene' is stripped / dropped")
    check(ds.is_title(ds.title_line("the orange open movie project presents")) and ds.title_line("elephants dream") == "Title: Elephants dream.", "title_line formats on-screen text")
    gaps = [{"usable_start": 4.0, "usable_end": 27.0, "kind": "bed"}]
    sl = ds.make_slots(gaps)
    check(all(s["end"] - s["start"] <= ds.MAX_SLOT + 1e-6 for s in sl) and len(sl) == 3, "long gap is cut into <= 10 s slots")


def video_checks(name: str) -> None:
    od = OUT / name
    check((od / "lines.json").exists(), f"{name}: lines.json exists")
    if not (od / "lines.json").exists():
        return
    j = json.loads((od / "lines.json").read_text(encoding="utf-8"))
    lines, gaps, speech = j["lines"], j["gaps"], [tuple(s) for s in j["speech"]]
    check(len(lines) >= 1, f"{name}: {len(lines)} lines drafted")
    over = sum(au.overlap((l["start"], l["end"]), speech) for l in lines)
    check(over == 0.0, f"{name}: lines overlap detected speech by {over:.3f} s (must be 0)")
    inside = all(gaps[l["gap"]]["usable_start"] - 1e-6 <= l["start"] and l["end"] <= gaps[l["gap"]]["usable_end"] + 1e-6 for l in lines)
    check(inside, f"{name}: every line sits inside its own dialogue gap (with guards)")
    check(all(lines[i]["end"] <= lines[i + 1]["start"] for i in range(len(lines) - 1)), f"{name}: lines do not overlap each other")
    check(all(l["words"] <= ds.MAX_WORDS for l in lines), f"{name}: every line has <= {ds.MAX_WORDS} words (max {max(l['words'] for l in lines)})")
    check(all(l["words"] / l["slot_len"] <= 2.5 + 1e-6 for l in lines),
          f"{name}: pace budget: max words per usable slot-second {max(l['words'] / l['slot_len'] for l in lines):.2f} <= 2.5")
    check(all(l["end"] - l["start"] <= l["slot_len"] for l in lines), f"{name}: the voice (real TTS, measured) fits inside every slot")
    check(all(l["flag"] in ("GREEN", "AMBER", "UNVERIFIED") for l in lines), f"{name}: every line has a flag")
    check(all((l["flag"] == "GREEN") == (l["verified"] and not l["reasons"]) for l in lines),
          f"{name}: GREEN only where a checkable claim was confirmed and nothing disagrees")
    check(all(l["flag"] != "UNVERIFIED" or not l["verified"] for l in lines), f"{name}: UNVERIFIED lines have no checked claim")
    check(not any(re.search(r"remains static|throughout the scene", l["text"], re.I) for l in lines), f"{name}: no 'remains static / throughout the scene' filler")
    # VTT
    cues = parse_vtt(od / "descriptions.vtt")
    check(len(cues) == len(lines), f"{name}: VTT has {len(cues)} cues == {len(lines)} lines")
    check(all(abs(c[1] - l["start"]) < 0.002 and abs(c[2] - l["end"]) < 0.002 for c, l in zip(cues, lines)), f"{name}: VTT times match lines")
    # images / html
    w = Image.open(od / "hero.jpg").size[0]
    check(w <= 1280, f"{name}: hero.jpg width {w} <= 1280")
    rep = (od / "report.html").read_text(encoding="utf-8")
    check("Footage:" in rep and "ad.mp4" in rep, f"{name}: report.html credits footage and embeds ad.mp4")
    check((od / "player.html").exists(), f"{name}: player.html exists")
    ph = (od / "player.html").read_text(encoding="utf-8")
    check(0 <= ph.find("const COL=") < ph.find("D.lines.forEach") and '"wave": [' in ph and "UNVERIFIED" in ph, f"{name}: player.html defines its flag colours before use and carries the waveform")
    # media: audio-described mp4 keeps video, programme untouched outside the lines, audible inside
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ad, pl = decode(od / "ad.mp4", td, "ad"), decode(od / "plain.mp4", td, "pl")
        n = min(len(ad), len(pl))
        ad, pl = ad[:n], pl[:n]
        la, lp = au.rms_db(ad), au.rms_db(pl)
        m = min(len(la), len(lp))
        mask_in = np.zeros(m, bool)
        mask_near = np.zeros(m, bool)
        for l in lines:
            mask_in[int((l["start"] + 0.2) / 0.05): int((l["end"] - 0.1) / 0.05)] = True
            mask_near[max(0, int((l["start"] - 1.0) / 0.05)): int((l["end"] + 1.5) / 0.05)] = True
        out_d = np.abs(la[:m] - lp[:m])[~mask_near & (lp[:m] > -60)]
        check(len(out_d) == 0 or float(np.percentile(out_d, 95)) < 1.5,
              f"{name}: programme unchanged outside lines (p95 level diff {float(np.percentile(out_d, 95)) if len(out_d) else 0:.2f} dB)")
        inside_d = (la[:m] - lp[:m])[mask_in]
        check(len(inside_d) == 0 or float(np.median(inside_d)) > -1.0, f"{name}: description is audible in the mix (median level change inside lines {float(np.median(inside_d)) if len(inside_d) else 0:+.1f} dB)")
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(od / "ad.mp4")], capture_output=True, text=True).stdout.split()
    check("video" in pr and "audio" in pr, f"{name}: ad.mp4 has video + audio streams")
    st = j["stats"]
    check(st["n_green"] + st["n_amber"] + st["n_unverified"] == st["n_lines"], f"{name}: GREEN + AMBER + UNVERIFIED = lines")
    d = (st.get("ducking_db") or {}).get("mean_db")
    check(d is None or d >= 5.0, f"{name}: programme ducked by mean {d} dB under the voice (>= 5)")


def global_checks() -> None:
    r = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    check(set(VIDEOS) <= set(r["videos"]), "results.json covers the default test set")
    check(r["totals"]["over_speech_s"] == 0.0, f"results.json: total seconds over detected speech = {r['totals']['over_speech_s']}")
    v = r.get("vad_validation_SIMULATED")
    check(v is not None, "results.json carries the SIMULATED speech-detector validation")
    if v:
        inj = [e for b in v.values() for e in b["injected"]]
        check(min(e["mean_coverage"] for e in inj) >= 0.9, f"VAD covers >= 90% of injected speech in every bed/gain (min {min(e['mean_coverage'] for e in inj):.3f})")
        check(sum(e["gap_overlap_with_injected_speech_s"] for e in inj) <= 0.5, "gaps overlap injected speech by <= 0.5 s in total")
    ey = [r["videos"][n]["eyeball"] for n in VIDEOS if n in r["videos"]]
    check(all(e["unjudged_or_stale"] == 0 for e in ey), "eyeball.json judges every current line (none stale)")
    sp = r["spectrogram_audit_by_eye"]
    n_lines = sum(r["videos"][n]["n_lines"] for n in VIDEOS)
    check(sp["judged"] == n_lines, f"spectro_audit.json judges every current line ({sp['judged']}/{n_lines})")
    check(sp["speech-like"] == 0, f"spectrogram audit by eye: {sp['clear']} clear, {sp['ambiguous']} ambiguous, {sp['speech-like']} speech-like lines")
    kt = r["kill_test"]
    check(kt["eyeball_accurate_strict"] == round(sum(e["accurate"] for e in ey) / n_lines, 3), "kill-test accuracy matches the by-eye verdict counts")
    ed = json.loads((OUT / "elephants_dream" / "lines.json").read_text(encoding="utf-8"))
    check(any(l["text"].startswith("Title:") for l in ed["lines"]), "Elephants Dream: the title card is voiced (on-screen text read)")
    demo = OUT / "elephants_dream" / "demo_60s.mp4"
    check(demo.exists() and demo.stat().st_size > 100_000, "Elephants Dream demo_60s.mp4 exists")


if __name__ == "__main__":
    unit_tests()
    for v in VIDEOS:
        video_checks(v)
    if (HERE / "results.json").exists():
        global_checks()
    print("\nFAILED:" if FAILS else "\nALL SMOKE CHECKS PASSED", *FAILS, sep="\n  ")
    sys.exit(1 if FAILS else 0)
