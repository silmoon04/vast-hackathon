"""Train the tiny speech detector used by audio.py (local, ~minutes). Run once: python train_vad.py

Training data is SYNTHETIC speech (Windows TTS voices David + Hazel, random rate, random pitch/formant resampling)
mixed at random gains into REAL non-dialogue beds taken from the audio of OTHER clips (lisbon_parking, motorway_a40,
ptz_parking, Intel samples). Held out entirely from training: the three test programmes (elephants_dream, yonge_dundas,
dashcam_urban), the Zira voice and every sentence in validate.LINES. Weights: vad_model.pt (about 60 KB).
"""
from __future__ import annotations

import random
import sys
from math import gcd
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audio as au  # noqa: E402
import tts  # noqa: E402
from vad_net import VadNet, logmel  # noqa: E402

SUBJ = ["the man", "she", "my brother", "the old woman", "we", "the captain", "nobody", "the children", "your friend", "I",
        "the doctor", "they", "the driver", "our neighbour", "the young boy"]
VERB = ["wants to know", "never understood", "keeps telling me", "will not forget", "asked about", "is worried about",
        "forgot all about", "does not care for", "has been waiting for", "decided against", "cannot believe", "laughed about"]
OBJ = ["the train that left early", "what happened last night", "the money on the table", "the road through the hills",
       "that terrible noise outside", "the letter from the bank", "where the keys went", "how long this will take",
       "the weather this week", "whether the shop is open", "the plan we made yesterday", "all of those old photographs"]
TAIL = ["", ", so please be quiet", ", and that is the end of it", ", which is hardly surprising", " right now",
        ", but nobody listens", " before it gets dark", ", can you imagine that"]
BEDS = ["lisbon_parking", "motorway_a40", "ptz_parking", "car_detection", "people_detection", "person_bicycle_car_detection",
        "store_aisle_detection", "worker_zone_detection", "face_demographics_walking_and_pause"]


def sentences(n: int, seed: int = 1) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        s = f"{rng.choice(SUBJ)} {rng.choice(VERB)} {rng.choice(OBJ)}{rng.choice(TAIL)}."
        out.add(s[0].upper() + s[1:])
    return sorted(out)


def load_utts(n: int = 260) -> list[np.ndarray]:
    items = []
    for i, s in enumerate(sentences(n)):
        items.append({"text": s, "voice": ["Microsoft David Desktop", "Microsoft Hazel Desktop"][i % 2], "rate": [-2, 0, 2][i % 3]})
    paths = tts.synth_many(items)
    out = []
    for p in paths:
        x, sr = au.load_wav(p)
        g = gcd(au.SR, sr)
        x = resample_poly(x, au.SR // g, sr // g).astype(np.float32)
        idx = np.where(np.abs(x) > 0.01 * np.abs(x).max())[0]
        out.append(x[idx[0]: idx[-1] + 1])
    return out


def synth_bed(rng, L: int) -> np.ndarray:
    """Non-speech training beds: coloured noise, engine/hum drones, vibrato chords with tremolo, filtered sweeps."""
    t = np.arange(L) / au.SR
    kind = rng.integers(0, 4)
    if kind == 0:  # coloured noise
        w = rng.standard_normal(L).astype(np.float32)
        W = np.fft.rfft(w)
        f = np.fft.rfftfreq(L, 1 / au.SR) + 1.0
        W *= f ** (-rng.uniform(0, 1.2)) * (f < rng.uniform(400, 7500))
        x = np.fft.irfft(W, L).astype(np.float32)
    elif kind == 1:  # drone
        f0 = rng.uniform(35, 220)
        x = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h ** rng.uniform(0.5, 1.5) * (1 + 0.3 * np.sin(2 * np.pi * rng.uniform(0.05, 3) * t))
                for h in range(1, rng.integers(3, 9))).astype(np.float32)
        x += 0.3 * rng.standard_normal(L).astype(np.float32) * 0.3
    elif kind == 2:  # chords / pads / pulses
        x = np.zeros(L, np.float32)
        beat = rng.uniform(0.5, 3.0)
        for _ in range(rng.integers(2, 5)):
            f0 = rng.uniform(90, 900)
            vib = 1 + 0.004 * np.sin(2 * np.pi * rng.uniform(4, 7) * t)
            for h in range(1, 7):
                x += (np.sin(2 * np.pi * np.cumsum(f0 * h * vib) / au.SR) / h).astype(np.float32)
        x *= (0.55 + 0.45 * np.sin(2 * np.pi * beat * t) ** 2).astype(np.float32) if rng.random() < 0.5 else 1
    else:  # swept noise (whoosh, wind, traffic)
        w = rng.standard_normal(L).astype(np.float32)
        from scipy.signal import butter, sosfilt
        out = np.zeros(L, np.float32)
        n = 8
        for k in range(n):
            a, b = k * L // n, (k + 1) * L // n
            fc = rng.uniform(150, 4000)
            out[a:b] = sosfilt(butter(2, fc, "low", fs=au.SR, output="sos"), w[a:b])
        x = out * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 1.5) * t))
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def make_batch(utts, beds, rng, n=24, secs=20.0):
    X, Y = [], []
    L = int(secs * au.SR)
    for _ in range(n):
        b = beds[rng.integers(len(beds))]
        o = rng.integers(0, max(1, len(b) - L))
        bed = b[o: o + L].copy()
        if len(bed) < L:
            bed = np.pad(bed, (0, L - len(bed)))
        r = rng.random()
        rms0 = float(np.sqrt(np.mean(bed ** 2)) + 1e-4)
        if r < 0.30:   # synthetic bed only
            bed = synth_bed(rng, L) * rms0 * 10 ** (rng.uniform(-14, 6) / 20)
        elif r < 0.55:  # real bed plus synthetic bed
            bed = bed + synth_bed(rng, L) * rms0 * 10 ** (rng.uniform(-12, 3) / 20)
        bed = bed * 10 ** (rng.uniform(-18, 6) / 20)
        lab = np.zeros(L // au.HOP, np.float32)
        brms = float(np.sqrt(np.mean(bed ** 2)) + 1e-4)
        t = rng.uniform(0.2, 2.0)
        if rng.random() > 0.15:  # 15 % pure-bed negatives
            while t < secs - 3:
                u = utts[rng.integers(len(utts))]
                f = rng.uniform(0.82, 1.22)  # pseudo new speaker: shifts pitch and formants together
                num = int(round(f * 100))
                u = resample_poly(u, 100, num).astype(np.float32)
                if t + len(u) / au.SR > secs - 0.2:
                    break
                g = brms * 10 ** (rng.uniform(-6, 9) / 20) / (np.sqrt(np.mean(u ** 2)) + 1e-9)
                i0 = int(t * au.SR)
                bed[i0: i0 + len(u)] += u * g
                lab[int(t * 100): int((t + len(u) / au.SR) * 100)] = 1
                t += len(u) / au.SR + rng.uniform(0.3, 4.0)
                if rng.random() < 0.25:
                    break
        X.append(logmel(np.clip(bed, -1, 1)))
        Y.append(lab)
    m = min(min(x.shape[1] for x in X), min(len(y) for y in Y))
    return torch.tensor(np.stack([x[:, :m] for x in X])), torch.tensor(np.stack([y[:m] for y in Y]))


def main(pool: int = 720, epochs: int = 28) -> None:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    utts = load_utts()
    beds = [au.load_wav(HERE / "_work" / "beds" / f"{b}.wav")[0] for b in BEDS]
    Xs, Ys = [], []
    for _ in range(pool // 24):
        for attempt in range(30):  # this laptop is shared by many jobs: ride out transient out-of-memory errors
            try:
                X, Y = make_batch(utts, beds, rng, n=24)
                break
            except MemoryError:
                import time; time.sleep(5)
        Xs.append(X[:, :, :1990]); Ys.append(Y[:, :1990])
    X, Y = torch.cat(Xs), torch.cat(Ys)
    print("pool", X.shape, "speech frac", round(float(Y.mean()), 3), flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = VadNet().to(dev)
    bs = 16
    steps = epochs * (len(X) // bs)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    lossf = nn.BCEWithLogitsLoss()
    it = 0
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for k in range(0, len(X) - bs + 1, bs):
            idx = perm[k: k + bs]
            xb, yb = X[idx].to(dev), Y[idx].to(dev)
            o = rng.integers(0, 90)  # random crop
            loss = lossf(net(xb[:, :, o: o + 1900]), yb[:, o: o + 1900])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss); it += 1
        if ep % 5 == 0 or ep == epochs - 1:
            print(ep, round(tot / (len(X) // bs), 4), flush=True)
    torch.save(net.cpu().state_dict(), HERE / "vad_model.pt")
    print("saved", sum(p.numel() for p in net.parameters()), "params")


if __name__ == "__main__":
    main()
