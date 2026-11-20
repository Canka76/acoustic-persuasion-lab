"""Chord table and pitch-Hz helpers."""
from __future__ import annotations

CHORD_TABLE: dict[str, tuple[int, ...]] = {
    "Dm7":     (0, 3, 7, 10),
    "G7":      (0, 4, 7, 10),
    "Cmaj7":   (0, 4, 7, 11),
    "tritone": (0, 6),
}

_C2_HZ = 65.406  # MIDI C2


def chord_pitches_hz(chord: str, root_hz: float = _C2_HZ) -> tuple[float, ...]:
    """Return absolute pitch values in Hz for *chord* rooted at *root_hz*."""
    semitones = CHORD_TABLE[chord]
    return tuple(root_hz * (2 ** (s / 12)) for s in semitones)


def nearest_chord_tone_hz(f0_hz: float, chord: str, root_hz: float = _C2_HZ) -> float:
    """Return the chord-tone pitch (Hz) nearest to *f0_hz* (octave-invariant)."""
    import math

    pitches = chord_pitches_hz(chord, root_hz)
    # Fold all chord tones into the same octave as f0_hz
    best: float = pitches[0]
    best_cents = float("inf")
    for p in pitches:
        # Find octave-equivalent nearest to f0_hz
        ratio = f0_hz / p
        octave = round(math.log2(ratio))
        candidate = p * (2**octave)
        cents = abs(1200 * math.log2(f0_hz / candidate))
        if cents < best_cents:
            best_cents = cents
            best = candidate
    return best
