"""Speech quality metrics: segmental SNR, log-spectral distance. (Stubs — implemented in Step 6.)"""
from __future__ import annotations

import numpy as np


def segmental_snr(ref: np.ndarray, deg: np.ndarray, sr: int, frame_s: float = 0.020) -> float:
    """Segmental SNR in dB (stub)."""
    raise NotImplementedError("Implemented in Step 6 (acoustic_firewall.quality)")


def log_spectral_distance(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Log-spectral distance in dB (stub)."""
    raise NotImplementedError("Implemented in Step 6 (acoustic_firewall.quality)")
