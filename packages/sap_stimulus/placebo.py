"""Arm C placebo layer: energy/spectral-matched noise, frequency-jittered AM, no F0 pull."""
from __future__ import annotations

import numpy as np

_JITTER_INTERVAL_S = 0.250  # resample f_mod every 250 ms


def _fir_shape_noise(
    noise: np.ndarray,
    target_spectrum: np.ndarray,
    sr: int,
    n_taps: int = 255,
) -> np.ndarray:
    """Shape white noise to match *target_spectrum* via an FIR filter."""
    from scipy.signal import firwin2, freqz, lfilter

    n_fft = 2048
    target_psd = np.abs(np.fft.rfft(target_spectrum, n=n_fft)) ** 2
    target_psd /= target_psd.max() + 1e-12

    freqs = np.linspace(0, 1, len(target_psd))  # normalised [0, 1]
    gains = np.sqrt(target_psd)
    gains /= gains.max() + 1e-12

    # firwin2 needs endpoints at 0 and 1
    b = firwin2(n_taps, freqs, gains)
    return lfilter(b, [1.0], noise).astype(np.float32)


def make_placebo_layer(
    b_layer: np.ndarray,
    sr: int,
    depth: float = 0.03,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return an Arm-C placebo layer matched to *b_layer* (from Arm B generator).

    - RMS within 0.5 dB of b_layer.
    - Long-term spectral envelope matched within 2 dB per octave band.
    - AM present but f_mod uniformly resampled from 0.5–30 Hz every 250 ms (phase-continuous).
    - No F0 modification.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(b_layer)
    # White noise base
    noise = rng.standard_normal(n).astype(np.float32)

    # Spectral shaping to match B layer envelope
    shaped = _fir_shape_noise(noise, b_layer, sr)

    # RMS match to B layer
    b_rms = float(np.sqrt(np.mean(b_layer**2) + 1e-12))
    s_rms = float(np.sqrt(np.mean(shaped**2) + 1e-12))
    shaped *= b_rms / s_rms

    # Jittered AM: resample f_mod every 250 ms with phase continuity
    jitter_hop = int(_JITTER_INTERVAL_S * sr)
    t = np.arange(n, dtype=np.float64) / sr
    modulator = np.ones(n, dtype=np.float64)
    phase = 0.0
    for start in range(0, n, jitter_hop):
        end = min(start + jitter_hop, n)
        f_mod = float(rng.uniform(0.5, 30.0))
        chunk_t = np.arange(end - start, dtype=np.float64) / sr
        chunk_mod = 1.0 + depth * np.cos(2 * np.pi * f_mod * chunk_t + phase)
        modulator[start:end] = chunk_mod
        # Update phase for continuity
        phase = (2 * np.pi * f_mod * (end - start) / sr + phase) % (2 * np.pi)

    return (shaped * modulator.astype(np.float32)).astype(np.float32)
