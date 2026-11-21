"""F0 pitch-pull toward chord tones via WORLD analysis/synthesis."""
from __future__ import annotations

import numpy as np

from .dramaturgy import Act
from .harmony import nearest_chord_tone_hz


MAX_SHIFT_CENTS: float = 60.0
DEFAULT_STRENGTH: float = 0.175


def pull_f0(
    wav: np.ndarray,
    sr: int,
    acts: list[Act],
    strength: float = DEFAULT_STRENGTH,
    return_diag: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    WORLD analysis → shift F0 toward nearest chord tone → resynthesize.
    Unvoiced frames untouched. Per-frame shift clamped to MAX_SHIFT_CENTS.

    Args:
        return_diag: if True, return (wav, diag) where diag contains
            'f0_pre', 'f0_post', 'abs_cents' (ground-truth per-frame shift
            with no estimator noise — same frame grid, same voicing mask).
    """
    try:
        import pyworld as pw
    except ImportError as e:
        raise ImportError("pyworld required for f0_pull") from e

    frame_period_ms = 5.0
    wav64 = wav.astype(np.float64)
    f0, t = pw.harvest(wav64, sr, frame_period=frame_period_ms)
    sp = pw.cheaptrick(wav64, f0, t, sr)
    ap = pw.d4c(wav64, f0, t, sr)

    frame_period_s = frame_period_ms / 1000.0
    times_s = np.arange(len(f0)) * frame_period_s

    act_lookup = _make_act_lookup(acts)
    new_f0 = f0.copy()

    for i, t_s in enumerate(times_s):
        if f0[i] <= 0:
            continue  # unvoiced
        act = act_lookup(t_s)
        if act is None:
            continue
        target_hz = nearest_chord_tone_hz(f0[i], act.chord)
        f0_cand = f0[i] * (1.0 - strength) + target_hz * strength
        # Clamp to MAX_SHIFT_CENTS
        cents = 1200.0 * np.log2(f0_cand / f0[i])
        if abs(cents) > MAX_SHIFT_CENTS:
            f0_cand = f0[i] * (2.0 ** (np.sign(cents) * MAX_SHIFT_CENTS / 1200.0))
        new_f0[i] = f0_cand

    out64 = pw.synthesize(new_f0, sp, ap, sr, frame_period=frame_period_ms)
    result = out64.astype(np.float32)

    if not return_diag:
        return result

    voiced_mask = f0 > 0
    abs_cents = np.where(
        voiced_mask,
        np.abs(1200.0 * np.log2(np.where(voiced_mask, new_f0 / (f0 + 1e-9), 1.0))),
        np.nan,
    )
    diag = {
        "f0_pre":     f0,
        "f0_post":    new_f0,
        "abs_cents":  abs_cents,  # NaN for unvoiced frames
    }
    return result, diag


def _make_act_lookup(acts: list[Act]):
    def lookup(t: float) -> Act | None:
        for act in acts:
            if act.start_s <= t < act.end_s:
                return act
        return acts[-1] if acts else None
    return lookup
