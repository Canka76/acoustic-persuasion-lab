"""Chord tables and pitch helpers."""
from __future__ import annotations

import numpy as np

# Semitone offsets from chord root
CHORD_TABLE: dict[str, tuple[int, ...]] = {
    "Dm7":        (0, 3, 7, 10),
    "G7":         (0, 4, 7, 10),
    "G7-tritone": (0, 4, 6, 10),   # tritone sub on the 5th
    "Cmaj7":      (0, 4, 7, 11),
    "tritone":    (0, 6),
}

C2_HZ: float = 65.406_195_8  # MIDI note 36


def chord_pitches_hz(chord: str, root_hz: float = C2_HZ) -> tuple[float, ...]:
    offsets = CHORD_TABLE[chord]
    return tuple(root_hz * (2.0 ** (s / 12.0)) for s in offsets)


def nearest_chord_tone_hz(f0_hz: float, chord: str, root_hz: float = C2_HZ) -> float:
    """Return the chord tone (any octave) closest in cents to f0_hz."""
    pitches = chord_pitches_hz(chord, root_hz)
    # Extend all chord tones across octaves spanning f0
    candidates: list[float] = []
    for p in pitches:
        oct_p = p
        while oct_p < f0_hz * 4:
            candidates.append(oct_p)
            oct_p *= 2.0
        oct_p = p / 2.0
        while oct_p > f0_hz / 4:
            candidates.append(oct_p)
            oct_p /= 2.0
    if not candidates:
        return f0_hz
    return min(candidates, key=lambda c: abs(1200 * np.log2(c / f0_hz)))
