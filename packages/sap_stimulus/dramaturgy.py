"""Deterministic three-act segmentation: premise / conflict / resolution."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sap_core.align import SentenceBoundary, align


@dataclass(frozen=True)
class Act:
    name: Literal["premise", "conflict", "resolution"]
    start_s: float
    end_s: float
    chord: str  # e.g. "Dm7", "G7", "Cmaj7", "C-tritone"


_ACT_RATIOS = {"premise": (0.00, 0.35), "conflict": (0.35, 0.70), "resolution": (0.70, 1.00)}
_ACT_CHORDS: dict[str, str] = {
    "premise": "Dm7",
    "conflict": "G7",   # substituted with C-tritone at boundary — see harmony.py
    "resolution": "Cmaj7",
}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def segment(
    text: str,
    duration_s: float,
    wav: np.ndarray | None = None,
    sr: int | None = None,
) -> list[Act]:
    """Return three Acts with time boundaries snapped to nearest sentence end.

    If *wav* and *sr* are provided, uses forced alignment (MMS or energy fallback)
    to snap boundaries; otherwise distributes proportionally by character count.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    if wav is not None and sr is not None:
        boundaries, _ = align(wav, sr, sentences)
        sent_ends = [b.end_s for b in boundaries]
    else:
        # Proportional by character offset
        chars = [len(s) for s in sentences]
        total = sum(chars)
        cumulative = np.cumsum(chars) / total
        sent_ends = [float(c * duration_s) for c in cumulative]

    acts: list[Act] = []
    for act_name, (ratio_start, ratio_end) in _ACT_RATIOS.items():
        target_start = ratio_start * duration_s
        target_end = ratio_end * duration_s

        # Snap end to nearest sentence boundary
        if act_name != "resolution":
            dists = [abs(e - target_end) for e in sent_ends]
            snapped_end = sent_ends[int(np.argmin(dists))]
        else:
            snapped_end = duration_s

        actual_start = acts[-1].end_s if acts else 0.0
        chord = _ACT_CHORDS[act_name]  # type: ignore[literal-required]
        acts.append(Act(name=act_name, start_s=actual_start, end_s=snapped_end, chord=chord))  # type: ignore[arg-type]

    return acts
