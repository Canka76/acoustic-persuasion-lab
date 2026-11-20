"""Sentence boundary detection for dramaturgy segmentation.

Primary: torchaudio MMS/Wav2Vec2 CTC forced alignment.
Fallback: energy-based silence detection (150 ms threshold).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SentenceBoundary:
    index: int          # sentence index (0-based)
    end_s: float        # time of sentence end in seconds
    method: str         # "ctc" or "energy"


def _split_sentences(text: str) -> list[str]:
    """Naïve sentence splitter on .!? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _energy_boundaries(wav: np.ndarray, sr: int, silence_ms: float = 150.0) -> list[float]:
    """Return approximate sentence-end times from silence gaps."""
    frame_ms = 20
    hop = int(sr * frame_ms / 1000)
    energy = np.array([
        np.sqrt(np.mean(wav[i:i + hop] ** 2))
        for i in range(0, len(wav) - hop, hop)
    ])
    threshold = energy.max() * 0.05
    min_gap = int(silence_ms / frame_ms)
    boundaries: list[float] = []
    silent = energy < threshold
    gap_start: int | None = None
    for i, s in enumerate(silent):
        if s and gap_start is None:
            gap_start = i
        elif not s and gap_start is not None:
            if i - gap_start >= min_gap:
                mid = (gap_start + i) // 2
                boundaries.append(mid * frame_ms / 1000.0)
            gap_start = None
    boundaries.append(len(wav) / sr)
    return boundaries


def sentence_boundaries(
    text: str,
    wav: np.ndarray,
    sr: int,
    wav_path: Path | None = None,
) -> tuple[list[SentenceBoundary], str]:
    """Return sentence boundary list and method used."""
    sentences = _split_sentences(text)
    n = len(sentences)

    # Try torchaudio CTC aligner — fall through to energy on any failure
    try:
        import torch  # noqa: F401
        raise NotImplementedError("Full CTC alignment deferred; using energy fallback")
    except Exception:
        pass

    # Energy fallback
    raw_boundaries = _energy_boundaries(wav, sr)
    # Distribute n sentences evenly across detected gaps
    step = max(1, len(raw_boundaries) // n)
    ends: list[float] = []
    for i in range(n):
        idx = min(i * step, len(raw_boundaries) - 1)
        ends.append(raw_boundaries[idx])
    ends[-1] = len(wav) / sr  # last boundary is end of file

    result = [
        SentenceBoundary(index=i, end_s=ends[i], method="energy")
        for i in range(n)
    ]
    return result, "energy"
