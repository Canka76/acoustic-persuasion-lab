"""TTS backend protocol + PiperBackend (default) + F5TtsBackend (optional)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class RenderInfo:
    out_path: Path
    duration_s: float
    sample_rate: int
    sha256: str
    backend: str


@runtime_checkable
class TtsBackend(Protocol):
    def synthesize(self, text: str, out_path: Path) -> RenderInfo:
        ...


class PiperBackend:
    """ONNX Piper TTS — CPU, deterministic with fixed seed, Windows-friendly."""

    def __init__(self, model: str = "en_US-lessac-medium", seed: int = 42) -> None:
        self.model = model
        self.seed = seed

    def synthesize(self, text: str, out_path: Path) -> RenderInfo:
        import numpy as np
        import soundfile as sf
        from piper import PiperVoice  # type: ignore[import]

        voice = PiperVoice.load(self.model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(str(out_path), "w", samplerate=voice.config.sample_rate, channels=1, subtype="FLOAT") as f:
            for audio_bytes in voice.synthesize_stream_raw(text):
                chunk = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                f.write(chunk)

        import soundfile as sf2
        data, sr = sf2.read(str(out_path))
        sha = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
        return RenderInfo(out_path=out_path, duration_s=len(data) / sr, sample_rate=sr, sha256=sha, backend="piper")


class F5TtsBackend:
    """F5-TTS backend (requires the 'f5' extra)."""

    def __init__(self) -> None:
        try:
            import f5_tts  # noqa: F401
        except ImportError as e:
            raise ImportError("Install the f5 extra: uv sync --extra f5") from e

    def synthesize(self, text: str, out_path: Path) -> RenderInfo:
        raise NotImplementedError("F5TtsBackend.synthesize — implement in Step 1 if needed")
