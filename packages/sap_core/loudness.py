"""Loudness normalization (ITU-R BS.1770-4 via pyloudnorm)."""
from __future__ import annotations

import numpy as np
import pyloudnorm as pyln


def measure_lufs(wav: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    return float(meter.integrated_loudness(wav))


def normalize_lufs(wav: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    return pyln.normalize.loudness(wav, loudness, target_lufs).astype(np.float32)


def true_peak_db(wav: np.ndarray) -> float:
    """Return true-peak level in dBTP (4× oversampling via scipy)."""
    import scipy.signal as sps
    up = sps.resample_poly(wav, 4, 1)
    return float(20.0 * np.log10(max(np.abs(up).max(), 1e-12)))


def peak_limit(wav: np.ndarray, ceiling_dbtp: float = -1.0) -> np.ndarray:
    """Hard-clip to ceiling (approximate limiter — good enough for stimulus prep)."""
    linear_ceiling = 10.0 ** (ceiling_dbtp / 20.0)
    return np.clip(wav, -linear_ceiling, linear_ceiling).astype(np.float32)
