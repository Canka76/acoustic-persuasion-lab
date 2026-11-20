"""Per-frame sidechain ducking: layer RMS sits target_snr_db below speech RMS."""
from __future__ import annotations

import numpy as np

_HOP_S = 0.020       # 20 ms hop
_WINDOW_S = 0.400    # 400 ms RMS window
_ATTACK_S = 0.050    # 50 ms attack
_RELEASE_S = 0.200   # 200 ms release


def duck_to_snr(
    speech: np.ndarray,
    layer: np.ndarray,
    sr: int,
    target_snr_db: float = -31.0,
) -> tuple[np.ndarray, float]:
    """Scale *layer* so it sits *target_snr_db* below speech RMS (per 20 ms frame).

    Returns (ducked_layer, achieved_broadband_snr_db).
    Gain is smoothed with 50 ms attack / 200 ms release.
    """
    hop = int(_HOP_S * sr)
    win = int(_WINDOW_S * sr)
    n = len(speech)

    # Pad to same length
    if len(layer) < n:
        layer = np.pad(layer, (0, n - len(layer)))
    else:
        layer = layer[:n]

    target_ratio = 10.0 ** (target_snr_db / 20.0)  # < 1

    # Compute per-frame gains
    frames = range(0, n, hop)
    gains: list[float] = []
    for start in frames:
        s_chunk = speech[max(0, start - win // 2) : start + win // 2]
        l_chunk = layer[max(0, start - win // 2) : start + win // 2]
        s_rms = float(np.sqrt(np.mean(s_chunk**2) + 1e-12))
        l_rms = float(np.sqrt(np.mean(l_chunk**2) + 1e-12))
        desired_l_rms = s_rms * target_ratio
        gains.append(desired_l_rms / l_rms)

    # Smooth gains: attack / release
    alpha_attack = 1.0 - np.exp(-1.0 / (sr * _ATTACK_S / hop))
    alpha_release = 1.0 - np.exp(-1.0 / (sr * _RELEASE_S / hop))
    smoothed = gains[:]
    for i in range(1, len(smoothed)):
        alpha = alpha_attack if smoothed[i] < smoothed[i - 1] else alpha_release
        smoothed[i] = alpha * smoothed[i] + (1 - alpha) * smoothed[i - 1]

    # Apply to layer sample-by-sample (linear interpolation between frames)
    gain_signal = np.interp(
        np.arange(n),
        [i * hop for i in range(len(smoothed))],
        smoothed,
    ).astype(np.float32)
    ducked = layer * gain_signal

    # Achieved broadband SNR
    speech_rms = float(np.sqrt(np.mean(speech**2) + 1e-12))
    layer_rms = float(np.sqrt(np.mean(ducked**2) + 1e-12))
    achieved_snr_db = 20.0 * np.log10(layer_rms / speech_rms + 1e-12)

    return ducked, achieved_snr_db
