"""Tiny dilated 1-D CNN speech detector: log-mel (40 bands, 10 ms hop) -> per-hop speech probability."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import percentile_filter

SR, HOP, NMEL = 16000, 160, 40
_MEL: np.ndarray | None = None


def _mel_fb(nfft: int = 512) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(7600), NMEL + 2))
    bins = np.fft.rfftfreq(nfft, 1 / SR)
    fb = np.zeros((NMEL, len(bins)), np.float32)
    for i in range(NMEL):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)), 0, None)
    return fb


def logmel(x: np.ndarray) -> np.ndarray:
    """(NMEL, T) log-mel with the slowly varying background (6 s running 10th percentile per band) subtracted, so the net
    sees what is salient relative to the bed it sits on, (absolute level is deliberately NOT given: it let an earlier version cheat)."""
    global _MEL
    if _MEL is None:
        _MEL = _mel_fb()
    n = 400
    m = 1 + (len(x) - n) // HOP
    idx = np.arange(n)[None, :] + HOP * np.arange(m)[:, None]
    fr = x[idx] * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, 512)) ** 2
    M = np.log(S @ _MEL.T + 1e-8).T.astype(np.float32)  # (NMEL, T)
    sub = M[:, ::10]
    bg = percentile_filter(sub, 10, size=(1, 61), mode="nearest")
    bg = np.stack([np.interp(np.arange(M.shape[1]), np.arange(0, M.shape[1], 10)[: bg.shape[1]], b) for b in bg])
    rel = M - bg
    return np.concatenate([rel / 3.0, np.zeros((1, M.shape[1]), np.float32)], 0).astype(np.float32)  # last channel unused (kept for shape)


class VadNet(nn.Module):
    def __init__(self, ch: int = 48):
        super().__init__()
        self.inp = nn.Conv1d(NMEL + 1, ch, 5, padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8, 16)])
        self.norms = nn.ModuleList([nn.GroupNorm(4, ch) for _ in self.blocks])
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, NMEL+1, T) -> logits (B, T)
        h = torch.relu(self.inp(x))
        for c, n in zip(self.blocks, self.norms):
            h = h + torch.relu(n(c(h)))
        return self.out(h).squeeze(1)


def load_net(path) -> VadNet:
    net = VadNet()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()


@torch.no_grad()
def predict(net: VadNet, x: np.ndarray) -> np.ndarray:
    return torch.sigmoid(net(torch.tensor(logmel(x))[None]))[0].numpy()
