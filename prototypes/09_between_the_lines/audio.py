"""Dialogue-gap finding for Between The Lines.

Two independent maps of the programme audio:
  silence map : ffmpeg silencedetect with an ADAPTIVE threshold (noise floor + margin), the classic approach.
  speech map  : a heuristic voice-activity detector (voicing x syllable-rate modulation x speech-band energy) that
                tolerates music beds and ambience: a gap may contain music, it may not contain talking.
Gaps (where a description may be voiced) are the complement of the speech map, >= MIN_GAP seconds, with guards.

On the event day both are replaced by Canary ASR word timestamps (exact), see README. No ASR weights exist locally,
so this module is the stand-in and is validated with injected synthetic dialogue (labelled SIMULATED in results).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as sg

FFMPEG = "ffmpeg"
SR = 16000
HOP = 160  # 10 ms
MIN_GAP = 1.5
GUARD = 0.25  # seconds kept clear either side of a line


def extract_wav(video: str | Path, dst: str | Path, t0: float = 0.0, t1: float | None = None, sr: int = SR) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{t0:.3f}", "-i", str(video)]
    if t1 is not None:
        cmd += ["-t", f"{t1 - t0:.3f}"]
    subprocess.run(cmd + ["-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)], check=True)
    return dst


def load_wav(p: str | Path) -> tuple[np.ndarray, int]:
    sr, x = wavfile.read(str(p))
    if x.dtype != np.float32:
        x = x.astype(np.float32) / 32768.0
    return x, sr


def rms_db(x: np.ndarray, sr: int = SR, win: float = 0.05) -> np.ndarray:
    n = int(sr * win)
    m = len(x) // n
    return 20 * np.log10(np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(1)) + 1e-7)


# ---------------------------------------------------------------------------------------------- silence map


def silence_map(x: np.ndarray, sr: int = SR, min_dur: float = 0.4) -> tuple[list[tuple[float, float]], float]:
    """Adaptive silencedetect: threshold = 10th percentile of 50 ms loudness + 8 dB (floored -60, capped -30 dB).
    Runs the real ffmpeg silencedetect filter on the wav. Returns ([(start, end)], threshold_db)."""
    lv = rms_db(x, sr)
    thr = float(np.clip(np.percentile(lv[lv > -90], 10) + 8.0, -60.0, -30.0)) if (lv > -90).any() else -60.0
    tmp = Path(__file__).resolve().parent / "_work" / "_sd.wav"
    tmp.parent.mkdir(exist_ok=True)
    wavfile.write(str(tmp), sr, (x * 32767).astype(np.int16))
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", str(tmp), "-af", f"silencedetect=n={thr:.1f}dB:d={min_dur}",
                        "-f", "null", "-"], capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"silence_start: (-?[\d.]+)", r.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    dur = len(x) / sr
    if len(starts) > len(ends):
        ends.append(dur)
    return [(max(0.0, s), e) for s, e in zip(starts, ends)], thr


# ----------------------------------------------------------------------------------------------- speech map


def _frames(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    m = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(m)[:, None]
    return x[idx]


def speech_features(x: np.ndarray, sr: int = SR) -> dict[str, np.ndarray]:
    """Per 10 ms hop: voicing (0-1), speech-band loudness (dB), and a 1 s sliding syllable-rate modulation ratio."""
    x = x - np.mean(x)
    pre = np.append(x[0], x[1:] - 0.95 * x[:-1])
    # voicing: normalised autocorrelation peak for f0 in 70-400 Hz on 40 ms frames
    n = int(0.040 * sr)
    fr = _frames(pre, n, HOP) * np.hanning(n)
    nfft = 2048
    F = np.fft.rfft(fr, nfft)
    ac = np.fft.irfft(np.abs(F) ** 2, nfft)[:, : n]
    ac = ac / (ac[:, :1] + 1e-9)
    lo, hi = int(sr / 400), int(sr / 70)
    peak = ac[:, lo:hi].max(1)
    voicing = np.clip((peak - 0.30) / 0.45, 0, 1)
    # speech-band (300-3400 Hz) envelope at 100 Hz
    sos = sg.butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    band = sg.sosfilt(sos, x)
    env = np.sqrt(np.mean(_frames(band, int(0.025 * sr), HOP) ** 2, axis=1) + 1e-12)
    m = min(len(env), len(voicing))
    env, voicing = env[:m], voicing[:m]
    # modulation ratio: power in 2.5-8 Hz of the log envelope over total 0.3-20 Hz, 1 s windows hop 0.25 s, spread back
    le = np.log(env + 1e-6)
    w = 100
    mod = np.zeros(m)
    if m > w:
        starts = np.arange(0, m - w, 25)
        fr2 = le[starts[:, None] + np.arange(w)[None, :]]
        fr2 = (fr2 - fr2.mean(1, keepdims=True)) * np.hanning(w)
        P = np.abs(np.fft.rfft(fr2, 512)) ** 2
        fq = np.fft.rfftfreq(512, 0.01)
        syl = P[:, (fq >= 2.5) & (fq <= 8)].sum(1)
        tot = P[:, (fq >= 0.3) & (fq <= 20)].sum(1) + 1e-9
        depth = np.sqrt(fr2.var(1) + 1e-9)  # raw log-envelope variability (how "chopped" the band is)
        ratio = syl / tot
        score = ratio * np.clip(depth / 0.35, 0, 1)
        for s, v in zip(starts, score):
            mod[s: s + w] = np.maximum(mod[s: s + w], v)
    return {"voicing": voicing, "env_db": 20 * np.log10(env + 1e-7), "mod": mod}


def speech_map(x: np.ndarray, sr: int = SR, *, thr: float = 0.5, dilate: float = 0.35, min_speech: float = 0.25,
               bridge: float = 0.6, feats: dict | None = None) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Boolean speech activity per 10 ms hop -> segments [(start, end)] and the score curve.
    Conservative on purpose: a false alarm only removes a gap, a miss puts a line over dialogue."""
    f = feats or speech_features(x, sr)
    v, mod, env = f["voicing"], f["mod"], f["env_db"]
    # voiced fraction in 0.4 s windows, requires audible speech band (not digital silence)
    k = 40
    vf = np.convolve(v, np.ones(k) / k, mode="same")
    audible = env > (np.percentile(env, 5) + 10.0)
    score = (0.55 * np.clip(vf / 0.5, 0, 1) + 0.45 * np.clip(mod / 0.55, 0, 1)) * audible
    active = score >= thr
    segs = _runs(active)
    segs = [(s, e) for s, e in segs if (e - s) * 0.01 >= min_speech]
    segs = _merge([(s * 0.01, e * 0.01) for s, e in segs], bridge)
    segs = [(max(0.0, s - dilate), e + dilate) for s, e in segs]
    return _merge(segs, 0.0), score


_NET = None


def net_scores(x: np.ndarray) -> np.ndarray | None:
    """Per-10 ms speech probability from the tiny CNN (vad_model.pt, trained by train_vad.py); None if no weights."""
    global _NET
    wp = Path(__file__).resolve().parent / "vad_model.pt"
    if not wp.exists():
        return None
    import vad_net
    if _NET is None:
        _NET = vad_net.load_net(wp)
    return vad_net.predict(_NET, x)


def speech_map_net(x: np.ndarray, *, thr: float = 0.35, dilate: float = 0.30, min_speech: float = 0.20, bridge: float = 0.7,
                   scores: np.ndarray | None = None) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Speech segments from the CNN. Threshold 0.35 (below 0.5) and 0.3 s dilation are deliberately conservative."""
    p = scores if scores is not None else net_scores(x)
    p = np.convolve(p, np.ones(5) / 5, mode="same")
    segs = [(s, e) for s, e in _runs(p >= thr) if (e - s) * 0.01 >= min_speech]
    segs = _merge([(s * 0.01, e * 0.01) for s, e in segs], bridge)
    return _merge([(max(0.0, s - dilate), e + dilate) for s, e in segs], 0.0), p


def _runs(b: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _merge(segs: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def complement(segs: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    out, t = [], 0.0
    for s, e in sorted(segs):
        if s > t:
            out.append((t, s))
        t = max(t, e)
    if t < duration:
        out.append((t, duration))
    return out


def intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                out.append((s, e))
    return _merge(out, 0.0)


def overlap(seg: tuple[float, float], segs: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(seg[1], e) - max(seg[0], s)) for s, e in segs)


def find_gaps(x: np.ndarray, sr: int = SR, *, min_gap: float = MIN_GAP, guard: float = GUARD, method: str = "net", **kw) -> dict:
    """Return {gaps:[{start,end,usable_start,usable_end,usable,kind,level_db}], speech, silence, ...}.
    A gap is any >= min_gap window with no detected speech. kind = 'quiet' if silencedetect also calls it silence for
    most of its length, else 'bed' (music / ambience / effects under it)."""
    dur = len(x) / sr
    sc = net_scores(x) if method == "net" else None
    if sc is not None:
        speech, score = speech_map_net(x, scores=sc)
    else:
        speech, score = speech_map(x, sr, feats=speech_features(x, sr), **kw)
    silence, sil_thr = silence_map(x, sr)
    lv = rms_db(x, sr)
    gaps = []
    for s, e in complement(speech, dur):
        if e - s < min_gap + 2 * guard:
            continue
        us, ue = s + guard, e - guard
        if ue - us < min_gap:
            continue
        sil_frac = overlap((s, e), silence) / (e - s)
        seg_lv = lv[int(s / 0.05): int(e / 0.05)]
        gaps.append({"start": round(s, 3), "end": round(e, 3), "usable_start": round(us, 3), "usable_end": round(ue, 3),
                     "usable": round(ue - us, 3), "kind": "quiet" if sil_frac > 0.6 else "bed",
                     "silence_frac": round(sil_frac, 2), "level_db": round(float(np.mean(seg_lv)), 1) if len(seg_lv) else -90})
    strict = [(a, b) for a, b in intersect(complement(speech, dur), silence) if b - a >= min_gap + 2 * guard]
    return {"gaps": gaps, "strict_gap_s": round(sum(b - a - 2 * guard for a, b in strict), 1), "speech": speech, "silence": silence, "silence_thr_db": sil_thr, "score": score,
            "duration": dur, "speech_frac": round(sum(e - s for s, e in speech) / dur, 3)}
