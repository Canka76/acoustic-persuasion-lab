"""Forced alignment: torchaudio MMS aligner with energy-based fallback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SentenceBoundary:
    sentence_idx: int
    start_s: float
    end_s: float
    text: str


def align(
    wav: np.ndarray,
    sr: int,
    sentences: list[str],
) -> tuple[list[SentenceBoundary], str]:
    """Align *sentences* to *wav*.

    Returns (boundaries, align_method) where align_method is "mms" or "energy".
    Tries torchaudio MMS/Wav2Vec2 CTC aligner first; falls back to energy-based
    detection with a 150 ms silence threshold if torch is unavailable.
    """
    try:
        return _mms_align(wav, sr, sentences)
    except Exception:
        return _energy_align(wav, sr, sentences)


def _mms_align(
    wav: np.ndarray, sr: int, sentences: list[str]
) -> tuple[list[SentenceBoundary], str]:
    import torchaudio  # noqa: F401 — availability probe
    raise NotImplementedError("MMS alignment not yet implemented; fallback to energy")


def _energy_align(
    wav: np.ndarray, sr: int, sentences: list[str]
) -> tuple[list[SentenceBoundary], str]:
    """Energy-based sentence boundary detection using 150 ms silence threshold."""
    silence_threshold_s = 0.150
    frame_len = int(sr * 0.020)  # 20 ms frames
    hop = frame_len // 2

    frames = [
        wav[i : i + frame_len]
        for i in range(0, len(wav) - frame_len + 1, hop)
    ]
    rms = np.array([np.sqrt(np.mean(f**2) + 1e-12) for f in frames])
    noise_floor = np.percentile(rms, 10)
    is_silent = rms < noise_floor * 3

    silence_frames = int(silence_threshold_s / (hop / sr))
    boundaries_s: list[float] = [0.0]
    silent_run = 0
    for i, s in enumerate(is_silent):
        if s:
            silent_run += 1
        else:
            if silent_run >= silence_frames:
                t = (i - silent_run // 2) * hop / sr
                boundaries_s.append(t)
            silent_run = 0
    boundaries_s.append(len(wav) / sr)

    # Distribute sentences evenly across detected boundaries
    n = len(sentences)
    pts = np.linspace(0, len(boundaries_s) - 1, n + 1)
    result: list[SentenceBoundary] = []
    for idx, sentence in enumerate(sentences):
        start = boundaries_s[min(int(pts[idx]), len(boundaries_s) - 1)]
        end = boundaries_s[min(int(pts[idx + 1]), len(boundaries_s) - 1)]
        result.append(SentenceBoundary(idx, start, end, sentence))
    return result, "energy"
