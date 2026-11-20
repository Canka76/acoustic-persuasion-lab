"""Speech quality metrics: segmental SNR, log-spectral distance.
(PESQ and ViSQOL not available on Windows — not used.)
"""
from __future__ import annotations

import numpy as np


def segmental_snr(ref: np.ndarray, deg: np.ndarray, sr: int, frame_ms: float = 30.0) -> float:
    """ITU-T P.56 segmental SNR (dB). Frames with ref energy < -50 dBov skipped."""
    hop = int(sr * frame_ms / 1000)
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    snrs = []
    for i in range(0, n - hop, hop):
        r = ref[i:i + hop]
        d = deg[i:i + hop]
        sig_pwr = np.mean(r ** 2)
        if sig_pwr < 1e-10:
            continue
        noise = r - d
        noise_pwr = np.mean(noise ** 2) + 1e-12
        snrs.append(10.0 * np.log10(sig_pwr / noise_pwr))
    if not snrs:
        return 0.0
    return float(np.mean(np.clip(snrs, -10, 35)))


def log_spectral_distance(ref: np.ndarray, deg: np.ndarray, sr: int, frame_ms: float = 30.0) -> float:
    """Mean log-spectral distance in dB over voiced frames."""
    from numpy.fft import rfft
    hop = int(sr * frame_ms / 1000)
    n = min(len(ref), len(deg))
    ref, deg = ref[:n], deg[:n]
    win = np.hanning(hop)
    lsds = []
    for i in range(0, n - hop, hop):
        r = np.abs(rfft(ref[i:i + hop] * win)) + 1e-12
        d = np.abs(rfft(deg[i:i + hop] * win)) + 1e-12
        if 20 * np.log10(r.max()) < -50:
            continue
        lsds.append(np.sqrt(np.mean((20 * np.log10(r / d)) ** 2)))
    return float(np.mean(lsds)) if lsds else 0.0
