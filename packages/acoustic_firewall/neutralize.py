"""Neutralizer: AM removal + LF layer removal. Stub — implemented in Step 6."""
from __future__ import annotations

import numpy as np

from acoustic_firewall.model import DetectReport


def neutralize(wav: np.ndarray, sr: int, report: DetectReport) -> np.ndarray:
    """Remove detected acoustic manipulation from *wav*.

    AM removal: complex-demodulate broadband envelope at report.f_mod_hz,
    reconstruct coherent component, apply inverse gain (clamped depth 0.08,
    smoothed over 100 ms).

    LF layer removal: MMSE spectral subtraction restricted to 40–120 Hz,
    noise PSD from minimum statistics over 3 s window, over-subtraction 1.5,
    spectral floor −18 dB. Bypassed when LF energy is F0-correlated.

    F0 de-quantization is NOT attempted (would damage prosody more than the
    manipulation; detection reports it; neutralization does not touch it).
    """
    raise NotImplementedError("neutralize() implemented in Step 6")
