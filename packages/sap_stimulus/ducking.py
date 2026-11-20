"""Per-frame energy ducking: set added layer SNR relative to speech."""
from __future__ import annotations

import numpy as np

HOP_MS = 20.0
RMS_WINDOW_MS = 400.0
ATTACK_MS = 50.0
RELEASE_MS = 200.0


def duck_to_snr(
    speech: np.ndarray,
    layer: np.ndarray,
    sr: int,
    target_snr_db: float = -31.0,
) -> tuple[np.ndarray, float]:
    """
    Scale `layer` per-frame so its RMS sits `target_snr_db` below speech RMS.
    Returns (scaled_layer, achieved_broadband_snr_db).

    Gain is smoothed with exponential attack/release envelopes.
    """
    hop = int(sr * HOP_MS / 1000)
    rms_win = int(sr * RMS_WINDOW_MS / 1000)
    n = max(len(speech), len(layer))
    speech_p = np.pad(speech, (0, max(0, n - len(speech))))
    layer_p = np.pad(layer.copy(), (0, max(0, n - len(layer))))

    target_linear = 10.0 ** (target_snr_db / 20.0)

    # Per-frame target gain
    gain_frames: list[float] = []
    n_frames = (n + hop - 1) // hop
    for fi in range(n_frames):
        i = fi * hop
        win_start = max(0, i - rms_win // 2)
        win_end = min(n, i + rms_win // 2 + 1)
        s_rms = float(np.sqrt(np.mean(speech_p[win_start:win_end] ** 2) + 1e-12))
        l_chunk = layer_p[i:min(i + hop, n)]
        l_rms = float(np.sqrt(np.mean(l_chunk ** 2) + 1e-12))
        desired_rms = s_rms * target_linear
        gain_frames.append(desired_rms / l_rms)

    # Smooth gain with attack/release (per-frame time constants)
    att = np.exp(-1.0 / max(1, int(ATTACK_MS / HOP_MS)))
    rel = np.exp(-1.0 / max(1, int(RELEASE_MS / HOP_MS)))
    smoothed: list[float] = []
    g = gain_frames[0] if gain_frames else 1.0
    for target_g in gain_frames:
        alpha = att if target_g < g else rel
        g = alpha * g + (1.0 - alpha) * target_g
        smoothed.append(g)

    # Apply gain sample-by-sample (piecewise constant per frame)
    for idx, g in enumerate(smoothed):
        start = idx * hop
        end = min(start + hop, n)
        layer_p[start:end] *= g

    scaled = layer_p[:len(layer)].astype(np.float32)

    # Measure achieved broadband SNR
    speech_rms_total = float(np.sqrt(np.mean(speech_p ** 2) + 1e-12))
    layer_rms_total = float(np.sqrt(np.mean(scaled ** 2) + 1e-12))
    achieved_snr = 20.0 * np.log10(layer_rms_total / speech_rms_total)

    return scaled, achieved_snr
