"""Dramaturgy segmentation: split audio into premise / conflict / resolution acts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Act:
    name: Literal["premise", "conflict", "resolution"]
    start_s: float
    end_s: float
    chord: str  # e.g. "Dm7", "G7", "Cmaj7"


_DEFAULT_CHORDS: dict[str, str] = {
    "premise": "Dm7",
    "conflict": "G7-tritone",
    "resolution": "Cmaj7",
}

_DEFAULT_SPLITS: tuple[float, float] = (0.35, 0.35)  # premise, conflict — rest is resolution


def segment(
    text: str,
    duration_s: float,
    wav: np.ndarray | None = None,
    sr: int | None = None,
) -> list[Act]:
    """Return [premise, conflict, resolution] acts.

    Boundaries are snapped to sentence ends detected from forced alignment
    (or energy-based fallback). If wav/sr not provided, boundaries are purely
    proportional to duration_s.
    """
    p_frac, c_frac = _DEFAULT_SPLITS
    r_frac = 1.0 - p_frac - c_frac  # noqa: F841

    if wav is not None and sr is not None:
        from sap_core.align import sentence_boundaries
        bounds, _method = sentence_boundaries(text, wav, sr)
        # Find sentence end closest to each target fraction
        targets = [p_frac * duration_s, (p_frac + c_frac) * duration_s]
        snapped = []
        for t in targets:
            best = min(bounds, key=lambda b: abs(b.end_s - t))
            snapped.append(best.end_s)
        premise_end = snapped[0]
        conflict_end = snapped[1]
    else:
        premise_end = p_frac * duration_s
        conflict_end = (p_frac + c_frac) * duration_s

    return [
        Act("premise", 0.0, premise_end, _DEFAULT_CHORDS["premise"]),
        Act("conflict", premise_end, conflict_end, _DEFAULT_CHORDS["conflict"]),
        Act("resolution", conflict_end, duration_s, _DEFAULT_CHORDS["resolution"]),
    ]
