"""Audio I/O helpers: load/save WAV, resample."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load(path: Path | str, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV file. Returns (samples, sample_rate). Resamples to *sr* if given."""
    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr is not None and sr != file_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return wav, file_sr


def save(path: Path | str, wav: np.ndarray, sr: int) -> None:
    """Write a WAV file at 32-bit float."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype="FLOAT")


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample *wav* from *orig_sr* to *target_sr*."""
    if orig_sr == target_sr:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
