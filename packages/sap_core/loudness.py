"""Loudness measurement: pyloudnorm LUFS, true-peak."""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln


def integrated_lufs(wav: np.ndarray, sr: int) -> float:
    """Return integrated loudness in LUFS."""
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(wav))


def normalize_lufs(wav: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    """Return loudness-normalised copy of *wav* targeting *target_lufs* LUFS."""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    return pyln.normalize.loudness(wav, loudness, target_lufs)


def true_peak_dbtp(wav: np.ndarray, sr: int) -> float:
    """Return true-peak level in dBTP."""
    # 4× oversample via scipy
    from scipy.signal import resample_poly
    upsampled = resample_poly(wav, 4, 1)
    peak = float(np.max(np.abs(upsampled)))
    return 20.0 * np.log10(peak + 1e-12)


def peak_limit(wav: np.ndarray, sr: int, ceiling_dbtp: float = -1.0) -> np.ndarray:
    """Hard-limit true peak to *ceiling_dbtp* dBTP."""
    ceiling_linear = 10.0 ** (ceiling_dbtp / 20.0)
    peak = float(np.max(np.abs(wav)))
    if peak > ceiling_linear:
        wav = wav * (ceiling_linear / peak)
    return wav
