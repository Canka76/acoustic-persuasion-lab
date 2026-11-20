"""Audio I/O and basic DSP utilities."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


def load(path: Path | str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV (or any soundfile-readable) file.
    Returns (samples float32 mono, sample_rate).
    If target_sr given and differs from file SR, resample via scipy."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import scipy.signal as sps
        wav = sps.resample_poly(wav, target_sr, sr).astype(np.float32)
        sr = target_sr
    return wav, sr


def save(wav: np.ndarray, sr: int, path: Path | str, subtype: str = "PCM_24") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype=subtype)


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(wav ** 2)))


def db(x: float) -> float:
    """Linear amplitude → dB."""
    return 20.0 * np.log10(max(x, 1e-12))


def from_db(db_val: float) -> float:
    return 10.0 ** (db_val / 20.0)
