"""Feature extraction for the acoustic firewall detector.

Stubs only — real implementation in Step 5.
Inter-band modulation coherence is the discriminating feature:
natural speech AM is band-incoherent at a fixed rate; a globally injected AM
is perfectly coherent across bands.
"""
from __future__ import annotations

import numpy as np

_WINDOW_S = 8.0
_HOP_S = 1.0
_N_BANDS = 32
_FMIN = 50.0
_FMAX = 8000.0
_ENV_SR = 200  # Hz, envelope decimation rate


def extract(wav: np.ndarray, sr: int) -> dict[str, float]:
    """Extract detection features from *wav* over 8 s windows / 1 s hop.

    Returns a flat dict of scalar features (nan for unavailable features).

    Features (Step 5):
    1. 32-band gammatone filterbank Hilbert envelopes, decimated to 200 Hz.
    2. Inter-band modulation coherence (1–30 Hz) — primary discriminating feature.
    3. Peak frequency and prominence of coherent modulation component.
    4. 40–120 Hz band energy ratio and temporal stationarity.
    5. Cross-correlation of LF component with speech F0 track.
    6. F0 pitch-class histogram χ² vs uniform (needs ≥ 20 s voiced; else nan).
    7. Spectral kurtosis mean/variance in 40–120 Hz.
    """
    # Stub: return nan-filled dict for all features
    return {
        "inter_band_coherence_mean": float("nan"),
        "inter_band_coherence_peak_hz": float("nan"),
        "inter_band_coherence_prominence_db": float("nan"),
        "lf_band_energy_ratio": float("nan"),
        "lf_band_stationarity": float("nan"),
        "lf_f0_crosscorr": float("nan"),
        "f0_pitchclass_chi2": float("nan"),
        "spectral_kurtosis_mean": float("nan"),
        "spectral_kurtosis_var": float("nan"),
    }
