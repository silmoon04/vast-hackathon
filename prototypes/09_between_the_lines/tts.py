"""Windows built-in TTS (System.Speech via PowerShell). No downloads. Batch API so PowerShell starts once.

synth_many([{text, wav, voice?, rate?}]) -> writes WAV files (16-bit PCM, mono, 22.05 kHz from SAPI) and returns
durations in seconds. Results are cached by (text, voice, rate) so reruns are instant.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_work" / "tts_cache"
DEFAULT_VOICE = "Microsoft Hazel Desktop"  # en-GB, matches a UK AD brief

_PS = r"""
param([string]$Job)
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -Encoding UTF8 $Job | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$cur = ''
foreach ($it in $items) {
  if ($it.voice -ne $cur) { $s.SelectVoice($it.voice); $cur = $it.voice }
  $s.Rate = [int]$it.rate
  $s.SetOutputToWaveFile($it.wav)
  $s.Speak($it.text)
  $s.SetOutputToNull()
}
$s.Dispose()
"""


def _key(text: str, voice: str, rate: int) -> str:
    return hashlib.sha256(f"{voice}|{rate}|{text}".encode("utf-8")).hexdigest()[:20]


def wav_duration(p: Path) -> float:
    with wave.open(str(p), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def synth_many(items: list[dict]) -> list[float]:
    """items: [{text, voice?, rate?}] -> list of cached wav paths (str) in the same order; durations via wav_duration."""
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs, paths = [], []
    for it in items:
        voice, rate = it.get("voice", DEFAULT_VOICE), int(it.get("rate", 0))
        p = CACHE / f"{_key(it['text'], voice, rate)}.wav"
        paths.append(p)
        if not (p.exists() and p.stat().st_size > 1000):
            jobs.append({"text": it["text"], "voice": voice, "rate": rate, "wav": str(p)})
    if jobs:
        with tempfile.TemporaryDirectory() as td:
            jf, sf = Path(td) / "job.json", Path(td) / "tts.ps1"
            jf.write_text(json.dumps(jobs), encoding="utf-8")
            sf.write_text(_PS, encoding="utf-8")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(sf), "-Job", str(jf)],
                           check=True, capture_output=True, timeout=600)
    return [str(p) for p in paths]


def synth(text: str, voice: str = DEFAULT_VOICE, rate: int = 0) -> tuple[str, float]:
    p = synth_many([{"text": text, "voice": voice, "rate": rate}])[0]
    return p, wav_duration(Path(p))


if __name__ == "__main__":
    for r in (0, 2):
        p, d = synth("A bald man in a dark jacket leans out of a doorway and shouts across a cavernous machine hall.", rate=r)
        print(r, p, round(d, 2))


def active_duration(p: str | Path) -> float:
    """Seconds of actual speech in a wav (leading/trailing SAPI silence removed)."""
    import numpy as np
    with wave.open(str(p), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    idx = np.where(np.abs(x) > 0.005)[0]
    return float((idx[-1] - idx[0]) / sr) if len(idx) else 0.0
