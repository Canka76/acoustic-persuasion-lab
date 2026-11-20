"""Subharmonic drone layer and amplitude modulation."""
from __future__ import annotations

import numpy as np

from sap_stimulus.dramaturgy import Act
from sap_stimulus.harmony import chord_pitches_hz

_CROSSFADE_S = 0.020  # 20 ms raised-cosine crossfade at act boundaries


def subharmonic_layer(
    acts: list[Act],
    sr: int,
    n_samples: int,
    root_hz: float = 65.406,  # C2
) -> np.ndarray:
    """Sine-based subharmonic layer (60–110 Hz chord root) with 20 ms raised-cosine crossfades."""
    out = np.zeros(n_samples, dtype=np.float32)
    xfade_samples = int(_CROSSFADE_S * sr)
    t_global = np.arange(n_samples, dtype=np.float64) / sr

    for act_idx, act in enumerate(acts):
        start = int(act.start_s * sr)
        end = int(act.end_s * sr)
        if start >= n_samples:
            break
        end = min(end, n_samples)

        # Use the lowest chord pitch as the drone frequency
        pitches = chord_pitches_hz(act.chord, root_hz)
        freq = min(pitches)  # lowest pitch in the chord, should be in 60–110 Hz

        t_act = t_global[start:end]
        # Phase continuity: start phase from last sample of previous act
        phase_offset = 0.0
        if act_idx > 0:
            prev_end = int(acts[act_idx - 1].end_s * sr)
            prev_freq_pitches = chord_pitches_hz(acts[act_idx - 1].chord, root_hz)
            prev_freq = min(prev_freq_pitches)
            phase_offset = 2 * np.pi * prev_freq * (prev_end / sr) % (2 * np.pi)

        sine = np.sin(2 * np.pi * freq * t_act + phase_offset).astype(np.float32)

        # Raised-cosine fade in
        fade_in_len = min(xfade_samples, len(sine))
        fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_in_len) / fade_in_len))
        sine[:fade_in_len] *= fade_in

        # Raised-cosine fade out
        if act_idx < len(acts) - 1:
            fade_out_len = min(xfade_samples, len(sine))
            fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_out_len) / fade_out_len))
            sine[-fade_out_len:] *= fade_out

        out[start:end] += sine

    return out


def amplitude_modulate(
    wav: np.ndarray,
    sr: int,
    f_mod_hz: float = 10.0,
    depth: float = 0.03,
) -> np.ndarray:
    """Apply sinusoidal amplitude modulation: wav * (1 + depth * cos(2π f_mod t))."""
    t = np.arange(len(wav), dtype=np.float64) / sr
    modulator = (1.0 + depth * np.cos(2 * np.pi * f_mod_hz * t)).astype(np.float32)
    return wav * modulator
