"""SIMULATED dialogue injection: the only ground truth we have for the speech detector.

Real dialogue timestamps are unavailable locally (no ASR weights). So we mix KNOWN synthetic speech (Windows TTS,
three voices) into real programme beds at known times and gains, then measure how much of it the speech map covers
and whether any gap handed to the describer overlaps it. TTS is cleaner and more regular than film dialogue, so
treat the recall numbers as optimistic; they are a sanity floor, not a guarantee.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
import audio as au  # noqa: E402
import tts  # noqa: E402

LINES = [
    "Come on, we have to keep moving before it gets any worse.",
    "I told you not to look down, and now you will not stop shaking.",
    "That is not what I meant, and you know it perfectly well.",
    "Wait here for me, I will go and see what is on the other side.",
    "Nobody ever said it would be easy, but this is ridiculous.",
    "Listen carefully, because I am only going to say this once.",
    "Look at that, it has been there the whole time and we missed it.",
    "Do you really believe that he would just leave us behind?",
    "Careful now, the whole thing could come down at any moment.",
    "I think we are going the wrong way, the map says otherwise.",
    "Please, just give me a minute to think about what to do next.",
    "You never listen to a single word that I say to you.",
]
VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop", "Microsoft Hazel Desktop"]


def _utt(p: str) -> tuple[np.ndarray, float, float]:
    x, sr = au.load_wav(p)
    if sr != au.SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(au.SR, sr)
        x = resample_poly(x, au.SR // g, sr // g).astype(np.float32)
    lv = np.abs(x) > 0.01 * np.abs(x).max()
    idx = np.where(lv)[0]
    return x, idx[0] / au.SR, idx[-1] / au.SR


def inject(bed: np.ndarray, gain_db: float, seed: int = 0, n_lines: int = 12) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Mix synthetic speech into `bed`. gain_db: speech RMS relative to the bed's overall RMS (0 = equal loudness)."""
    rng = np.random.default_rng(seed)
    items = [{"text": LINES[i % len(LINES)], "voice": VOICES[(i + seed) % 3]} for i in range(n_lines)]
    paths = tts.synth_many(items)
    out = bed.copy()
    bed_rms = float(np.sqrt(np.mean(bed ** 2)) + 1e-9)
    t = float(rng.uniform(1.0, 3.0))
    spans = []
    for p in paths:
        u, a, b = _utt(p)
        if t + len(u) / au.SR + 1 > len(bed) / au.SR:
            break
        sp_rms = float(np.sqrt(np.mean(u[int(a * au.SR): int(b * au.SR)] ** 2)) + 1e-9)
        g = bed_rms * 10 ** (gain_db / 20) / sp_rms
        i0 = int(t * au.SR)
        out[i0: i0 + len(u)] += u * g
        spans.append((t + a, t + b))
        t += len(u) / au.SR + float(rng.uniform(1.8, 5.0))
    return np.clip(out, -1, 1), spans


def evaluate(bed: np.ndarray, gain_db: float, seed: int = 0) -> dict:
    mix, spans = inject(bed, gain_db, seed)
    r = au.find_gaps(mix)
    cov = [au.overlap(s, r["speech"]) / (s[1] - s[0]) for s in spans]
    gap_over = sum(au.overlap((g["usable_start"], g["usable_end"]), spans) for g in r["gaps"])
    spk = sum(e - s for s, e in spans)
    return {"gain_db": gain_db, "utterances": len(spans), "speech_s": round(spk, 1),
            "mean_coverage": round(float(np.mean(cov)), 3), "utt_detected_80pct": int(sum(c >= 0.8 for c in cov)),
            "utt_missed_below_50pct": int(sum(c < 0.5 for c in cov)),
            "gaps": len(r["gaps"]), "gap_seconds": round(sum(g["usable"] for g in r["gaps"]), 1),
            "gap_overlap_with_injected_speech_s": round(gap_over, 2)}


if __name__ == "__main__":
    beds = {}
    for name, (src, t0, t1) in {"ED_music_sfx_27-147": ("elephants_dream", 27, 147), "yonge_ambient": ("yonge_dundas", 0, 57),
                                "dashcam_engine": ("dashcam_urban", 0, 120)}.items():
        import protokit as pk
        au.extract_wav(pk.video_path(src), HERE / "_work" / f"bed_{name}.wav", t0, t1)
        beds[name], _ = au.load_wav(HERE / "_work" / f"bed_{name}.wav")
    for n, b in beds.items():
        base = au.find_gaps(b)
        print(n, "baseline (no injection): speech_frac", base["speech_frac"], "gaps", len(base["gaps"]))
        for g in (-3, 0, 6):
            print("  ", evaluate(b, g))
