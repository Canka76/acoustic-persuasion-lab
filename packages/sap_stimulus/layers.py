"""Subharmonic drone layer and amplitude modulation."""
from __future__ import annotations

import numpy as np

from .dramaturgy import Act
from .harmony import chord_pitches_hz, C2_HZ


def subharmonic_layer(
    acts: list[Act],
    sr: int,
    n_samples: int,
    root_hz: float = C2_HZ,
) -> np.ndarray:
    """
    Sine-wave subharmonic layer (60–110 Hz range) following the chord progression.
    Act boundaries crossfade with 20 ms raised-cosine ramps.
    """
    out = np.zeros(n_samples, dtype=np.float32)
    xfade_samp = int(sr * 0.020)  # 20 ms

    for act in acts:
        start = int(act.start_s * sr)
        end = min(int(act.end_s * sr), n_samples)
        if end <= start:
            continue

        # Pick the lowest chord tone in 60–110 Hz (extend via octave if needed)
        pitches = chord_pitches_hz(act.chord, root_hz)
        candidates = [p for p in pitches if 60 <= p <= 110]
        if not candidates:
            for p in pitches:
                oct_p = p
                while oct_p < 60:
                    oct_p *= 2.0
                while oct_p > 110:
                    oct_p /= 2.0
                candidates.append(oct_p)
        freq = min(candidates)

        t = np.arange(end - start) / sr
        segment = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

        # Raised-cosine fade-in at start
        if start > 0 and xfade_samp > 0:
            ramp_len = min(xfade_samp, len(segment))
            ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
            segment[:ramp_len] *= ramp.astype(np.float32)

        # Raised-cosine fade-out at end
        if xfade_samp > 0:
            ramp_len = min(xfade_samp, len(segment))
            ramp = 0.5 * (1 + np.cos(np.pi * np.arange(ramp_len) / ramp_len))
            segment[-ramp_len:] *= ramp.astype(np.float32)

        out[start:end] += segment

    return out


def amplitude_modulate(
    wav: np.ndarray,
    sr: int,
    f_mod_hz: float = 10.0,
    depth: float = 0.03,
) -> np.ndarray:
    """
    Apply coherent AM: wav * (1 + depth * cos(2π f_mod t)).
    depth=0.03 → 3 % modulation depth.
    """
    t = np.arange(len(wav)) / sr
    modulator = 1.0 + depth * np.cos(2.0 * np.pi * f_mod_hz * t)
    return (wav * modulator).astype(np.float32)
