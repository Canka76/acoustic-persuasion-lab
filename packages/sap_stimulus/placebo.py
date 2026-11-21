"""Arm C (placebo) layer generator.

Matches Arm B on:
- added-layer RMS within ±0.5 dB
- long-term spectral envelope within 2 dB/octave band (FIR shape-matched)
- AM present but frequency-jittered (0.5–30 Hz, every 250 ms, phase-continuous)
- F0 untouched (no pitch quantization)
"""
from __future__ import annotations

import numpy as np
from numpy.fft import rfft, irfft


def _shape_noise_to_layer(noise: np.ndarray, arm_b_layer: np.ndarray) -> np.ndarray:
    """Shape white noise to match the long-term spectral envelope of arm_b_layer."""
    n = len(arm_b_layer)
    frame = 1024
    hop = 512
    B_sum = np.zeros(frame // 2 + 1)
    N_sum = np.zeros(frame // 2 + 1)
    count = 0
    win = np.hanning(frame)
    for i in range(0, min(n, len(noise)) - frame, hop):
        B_sum += np.abs(rfft(arm_b_layer[i:i + frame] * win))
        N_sum += np.abs(rfft(noise[i:i + frame] * win))
        count += 1
    if count == 0:
        return noise
    B_avg = B_sum / count + 1e-12
    N_avg = N_sum / count + 1e-12
    H = B_avg / N_avg
    NOISE_SPEC = rfft(noise, n=n)
    H_full = np.interp(np.arange(n // 2 + 1), np.linspace(0, n // 2, len(H)), H)
    shaped = irfft(NOISE_SPEC * H_full, n=n)
    return shaped[:len(noise)].astype(np.float32)


def make_placebo_layer(
    arm_b_layer: np.ndarray,
    sr: int,
    seed: int = 1234,
) -> np.ndarray:
    """
    Return spectrum- and RMS-matched noise shaped to arm_b_layer.
    Does NOT apply AM — caller applies jittered AM after duck_to_snr so the
    modulation depth is preserved in the delivered mix (same fix as arm B).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(arm_b_layer)).astype(np.float32)
    shaped = _shape_noise_to_layer(noise, arm_b_layer)
    target_rms = float(np.sqrt(np.mean(arm_b_layer ** 2) + 1e-12))
    shaped_rms = float(np.sqrt(np.mean(shaped ** 2) + 1e-12))
    shaped *= target_rms / shaped_rms
    return shaped.astype(np.float32)


def apply_jittered_am(
    wav: np.ndarray,
    sr: int,
    seed: int = 1235,
    am_depth: float = 0.03,
) -> np.ndarray:
    """Apply phase-continuous jittered AM (f_mod uniform 0.5–30 Hz, every 250 ms).

    The 9–11 Hz band is excluded from the draw so the placebo arm can never
    accumulate coherent 10 Hz energy that would fail the arm-C QA gate.
    The exclusion uses a two-segment uniform draw: [0.5, 9) ∪ (11, 30].
    """
    block_samp = int(sr * 0.25)
    rng = np.random.default_rng(seed)
    out = wav.copy()
    phase = 0.0

    def _draw_f_mod() -> float:
        """Draw f_mod uniformly from [0.5, 9) ∪ (11, 30] Hz."""
        low_w = 9.0 - 0.5     # 8.5
        high_w = 30.0 - 11.0  # 19.0
        total = low_w + high_w
        if rng.uniform(0.0, total) < low_w:
            return float(rng.uniform(0.5, 9.0))
        return float(rng.uniform(11.0, 30.0))

    f_mod_hz = _draw_f_mod()
    for start in range(0, len(out), block_samp):
        end = min(start + block_samp, len(out))
        t = np.arange(end - start) / sr
        modulator = (1.0 + am_depth * np.cos(2.0 * np.pi * f_mod_hz * t + phase)).astype(np.float32)
        out[start:end] *= modulator
        phase = float((2.0 * np.pi * f_mod_hz * (end - start) / sr + phase) % (2.0 * np.pi))
        f_mod_hz = _draw_f_mod()
    return out.astype(np.float32)
