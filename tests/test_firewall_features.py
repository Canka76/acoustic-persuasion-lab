"""Gate test for acoustic_firewall.features.extract().

Synthesizes deterministic audio in-process; no corpus/stimuli required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "sap_core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "acoustic_firewall"))

from acoustic_firewall.features import extract

_SR = 22_050
_DURATION_S = 10.0
_RNG = np.random.default_rng(42)


def _make_audio() -> tuple[np.ndarray, np.ndarray]:
    """Return (clean, am_modulated) deterministic float32 arrays."""
    n = int(_DURATION_S * _SR)
    # Band-limited noise: white noise -> bandpass 100–7000 Hz (simulates speech band)
    from scipy.signal import butter, sosfiltfilt
    noise = _RNG.standard_normal(n).astype(np.float32)
    sos = butter(4, [100 / (_SR / 2), 7000 / (_SR / 2)], btype="band", output="sos")
    clean = sosfiltfilt(sos, noise).astype(np.float32)
    # Coherent AM at 10 Hz, 3% depth — mirrors sap_stimulus.layers.amplitude_modulate exactly
    t = np.arange(n) / _SR
    am = (clean * (1.0 + 0.03 * np.cos(2.0 * np.pi * 10.0 * t))).astype(np.float32)
    return clean, am


def test_extract_no_nan_clean():
    clean, _ = _make_audio()
    feats = extract(clean, _SR)
    nan_keys = [k for k, v in feats.items() if np.isnan(v)]
    # f0_pitchclass_chi2 is allowed to be nan when < 20 s voiced; exclude it
    disallowed = [k for k in nan_keys if k != "f0_pitchclass_chi2"]
    assert not disallowed, f"nan in clean features: {disallowed}"


def test_extract_no_nan_am():
    _, am = _make_audio()
    feats = extract(am, _SR)
    nan_keys = [k for k, v in feats.items() if np.isnan(v)]
    disallowed = [k for k in nan_keys if k != "f0_pitchclass_chi2"]
    assert not disallowed, f"nan in AM features: {disallowed}"


def test_am_coherence_exceeds_clean():
    """AM signal must yield higher inter_band_coherence_mean than clean noise."""
    clean, am = _make_audio()
    c_feats = extract(clean, _SR)
    a_feats = extract(am, _SR)
    assert a_feats["inter_band_coherence_mean"] > c_feats["inter_band_coherence_mean"], (
        f"AM coherence {a_feats['inter_band_coherence_mean']:.4f} not > "
        f"clean coherence {c_feats['inter_band_coherence_mean']:.4f}"
    )


def test_am_peak_hz_near_10():
    """Peak inter-band modulation frequency must be within 1 Hz of 10 Hz for AM input."""
    _, am = _make_audio()
    feats = extract(am, _SR)
    peak = feats["inter_band_coherence_peak_hz"]
    assert abs(peak - 10.0) <= 1.0, (
        f"inter_band_coherence_peak_hz = {peak:.2f} Hz, expected ≈ 10 Hz"
    )
