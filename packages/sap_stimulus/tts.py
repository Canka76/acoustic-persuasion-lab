"""TTS backend protocol and PiperBackend implementation.

Renders text to 48 kHz mono WAV, normalized to -23 LUFS, peak-limited to -1 dBTP.
Output is deterministic (fixed model, fixed seed) so same text → bit-identical PCM.
"""
from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from sap_core.audio import load, save, sha256
from sap_core.loudness import measure_lufs, normalize_lufs, true_peak_db, peak_limit

TARGET_SR = 48_000
TARGET_LUFS = -23.0
TARGET_DBTP = -1.0
SILENCE_THRESHOLD = 0.01   # fraction of peak
SILENCE_PAD_MS = 200.0

DEFAULT_MODEL_PATH = Path("models/piper/en_US-ryan-high.onnx")


@dataclass
class RenderInfo:
    out_path: Path
    sample_rate: int
    duration_s: float
    lufs: float
    true_peak_dbtp: float
    sha256: str
    backend: str


@runtime_checkable
class TtsBackend(Protocol):
    def synthesize(self, text: str, out_path: Path) -> RenderInfo: ...


def _trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    threshold = SILENCE_THRESHOLD * max(np.abs(wav).max(), 1e-12)
    nonzero = np.where(np.abs(wav) > threshold)[0]
    if len(nonzero) == 0:
        return wav
    pad = int(sr * SILENCE_PAD_MS / 1000)
    return wav[max(0, nonzero[0] - pad): min(len(wav), nonzero[-1] + pad)]


def _post_process(wav: np.ndarray, sr: int, out_path: Path, backend_name: str) -> RenderInfo:
    wav = _trim_silence(wav, sr)
    wav = normalize_lufs(wav, sr, TARGET_LUFS)
    wav = peak_limit(wav, TARGET_DBTP)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save(wav, TARGET_SR, out_path)
    return RenderInfo(
        out_path=out_path,
        sample_rate=TARGET_SR,
        duration_s=len(wav) / sr,
        lufs=measure_lufs(wav, sr),
        true_peak_dbtp=true_peak_db(wav),
        sha256=sha256(out_path),
        backend=backend_name,
    )


class PiperBackend:
    """Deterministic TTS via piper-tts Python API (ONNX, CPU).

    model_path: path to .onnx file (config .json expected alongside it).
    Voice: en_US-ryan-high by default; download with:
        python -c "import urllib.request; urllib.request.urlretrieve(
            'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx',
            'models/piper/en_US-ryan-high.onnx')"
    """

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        self._voice = None  # lazy load

    def _get_voice(self):
        if self._voice is None:
            from piper.voice import PiperVoice
            self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def synthesize(self, text: str, out_path: Path) -> RenderInfo:
        voice = self._get_voice()
        # Render to in-memory WAV buffer
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        buf.seek(0)
        # Read back via soundfile-compatible path
        import soundfile as sf
        wav_data, native_sr = sf.read(buf, dtype="float32", always_2d=False)
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        # Resample if needed
        if native_sr != TARGET_SR:
            import scipy.signal as sps
            wav_data = sps.resample_poly(wav_data, TARGET_SR, native_sr).astype(np.float32)
        return _post_process(wav_data, TARGET_SR, Path(out_path), "PiperBackend")


class EdgeTtsBackend:
    """Fallback: Microsoft Edge TTS via `edge-tts` package (requires internet).
    Not bit-reproducible across API versions. Use only if Piper is unavailable.
    """

    VOICE = "en-US-RyanMultilingualNeural"

    def __init__(self, voice: str | None = None):
        self.voice = voice or self.VOICE

    def synthesize(self, text: str, out_path: Path) -> RenderInfo:
        import asyncio, tempfile
        try:
            import edge_tts  # type: ignore[import]
        except ImportError as e:
            raise ImportError("pip install edge-tts") from e
        out_path = Path(out_path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            asyncio.run(edge_tts.Communicate(text, self.voice).save(str(tmp_path)))
            wav, sr = load(tmp_path, target_sr=TARGET_SR)
        finally:
            tmp_path.unlink(missing_ok=True)
        return _post_process(wav, sr, out_path, "EdgeTtsBackend")


def render_corpus(
    corpus_items: list[dict],
    out_dir: Path,
    backend: TtsBackend | None = None,
    overwrite: bool = False,
) -> dict[str, RenderInfo]:
    """Render all corpus items to out_dir/<id>.wav.

    The same base render is shared across all arms — arms differ only by
    post-processing of identical PCM. This is load-bearing for the design.
    """
    if backend is None:
        backend = PiperBackend()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RenderInfo] = {}

    for item in corpus_items:
        iid = item["id"]
        out_path = out_dir / f"{iid}.wav"
        if out_path.exists() and not overwrite:
            wav, sr = load(out_path)
            results[iid] = RenderInfo(
                out_path=out_path, sample_rate=sr,
                duration_s=len(wav) / sr,
                lufs=measure_lufs(wav, sr),
                true_peak_dbtp=true_peak_db(wav),
                sha256=sha256(out_path), backend="cached",
            )
            print(f"  [cache] {iid}")
            continue

        print(f"  [tts] {iid}...", end=" ", flush=True)
        info = backend.synthesize(item["text"], out_path)
        results[iid] = info
        print(f"{info.duration_s:.1f}s  LUFS={info.lufs:.1f}  TP={info.true_peak_dbtp:.1f}")

    return results
