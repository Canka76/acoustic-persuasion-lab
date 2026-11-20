"""F0 pitch-pull toward chord tones via WORLD analysis/synthesis."""
from __future__ import annotations

import math

import numpy as np

from sap_stimulus.dramaturgy import Act
from sap_stimulus.harmony import nearest_chord_tone_hz

_MAX_SHIFT_CENTS = 60.0


def pull_f0(
    wav: np.ndarray,
    sr: int,
    acts: list[Act],
    strength: float = 0.175,
) -> np.ndarray:
    """Return resynthesised WAV with voiced F0 pulled toward the act's chord tones.

    Uses WORLD (pyworld): harvest → cheaptrick → d4c → synthesize.
    Per-frame change clamped to ≤ 60 cents.
    Unvoiced frames are untouched.
    """
    import pyworld as pw  # type: ignore[import]

    wav64 = wav.astype(np.float64)
    f0, t = pw.harvest(wav64, sr)
    sp = pw.cheaptrick(wav64, f0, t, sr)
    ap = pw.d4c(wav64, f0, t, sr)

    # Build a lookup: frame time → chord for the current act
    def _chord_for_time(time_s: float) -> str:
        for act in reversed(acts):
            if time_s >= act.start_s:
                return act.chord
        return acts[0].chord

    new_f0 = f0.copy()
    for i, (time_s, f0_val) in enumerate(zip(t, f0)):
        if f0_val <= 0.0:  # unvoiced
            continue
        chord = _chord_for_time(time_s)
        target = nearest_chord_tone_hz(f0_val, chord)

        cents_diff = 1200.0 * math.log2(target / f0_val)
        clamped_cents = max(-_MAX_SHIFT_CENTS, min(_MAX_SHIFT_CENTS, cents_diff * strength / 0.175))
        new_f0[i] = f0_val * (2 ** (clamped_cents / 1200.0))

    out = pw.synthesize(new_f0, sp, ap, sr)
    return out.astype(np.float32)
