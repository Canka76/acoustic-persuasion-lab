"""
converse.py — speak any text through the acoustic-persuasion-lab pipeline.

Synthesises speech with Piper TTS (en_US-ryan-high) then applies the
arm A / B / C post-processing used in the RCT.  Output is a WAV file you
can play immediately.

Usage
-----
Interactive loop (default arm B):
    uv run python converse.py

Single utterance:
    uv run python converse.py "If the economy is strong, taxes must be low."

Pick an arm explicitly:
    uv run python converse.py --arm A "..."    # clean speech, no manipulation
    uv run python converse.py --arm B "..."    # F0-pull + subharmonic drone + 10 Hz AM
    uv run python converse.py --arm C "..."    # placebo (matched energy, jittered AM)

Custom output path:
    uv run python converse.py --out my_clip.wav "..."

Auto-play (Windows):
    uv run python converse.py --play "..."
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── make all workspace packages importable ───────────────────────────────────
_ROOT = Path(__file__).parent
for _pkg in ("sap_core", "sap_stimulus", "acoustic_firewall"):
    sys.path.insert(0, str(_ROOT / "packages" / _pkg))

from sap_core.audio import load, save
from sap_core.loudness import normalize_lufs, peak_limit
from sap_stimulus.tts import PiperBackend

_MODEL = _ROOT / "models" / "piper" / "en_US-ryan-high.onnx"
_SR_TARGET = 48_000
_AM_FREQ   = 10.0    # Hz — the fixed entrainment frequency
_AM_DEPTH  = 0.03    # 3 % modulation depth
_SNR_DB    = -31.0   # drone-to-speech SNR used in the RCT
_PULL_STR  = 0.35    # F0 pull strength used in the RCT


# ── arm A ────────────────────────────────────────────────────────────────────

def _arm_a(wav: np.ndarray, sr: int) -> np.ndarray:
    """Clean control: WORLD pass-through only (matches vocoder residual of B/C)."""
    from sap_stimulus.f0_pull import pull_f0
    wav = pull_f0(wav, sr, acts=[], strength=0.0)
    wav = normalize_lufs(wav, sr)
    return peak_limit(wav).astype(np.float32)


# ── arm B ────────────────────────────────────────────────────────────────────

def _arm_b(wav: np.ndarray, sr: int, text: str) -> np.ndarray:
    """Full treatment: F0-pull → subharmonic drone → duck → coherent 10 Hz AM → watermark."""
    from sap_stimulus.dramaturgy import segment
    from sap_stimulus.ducking import duck_to_snr
    from sap_stimulus.f0_pull import pull_f0
    from sap_stimulus.layers import amplitude_modulate, subharmonic_layer
    from sap_stimulus.watermark import embed

    duration_s = len(wav) / sr
    acts = segment(text, duration_s, wav=wav, sr=sr)
    wav_pitched = pull_f0(wav, sr, acts, strength=_PULL_STR)

    layer = subharmonic_layer(acts, sr, len(wav_pitched))
    layer_ducked, _ = duck_to_snr(wav_pitched, layer, sr, target_snr_db=_SNR_DB)
    layer_ducked = layer_ducked[:len(wav_pitched)]
    layer_am = amplitude_modulate(layer_ducked, sr, f_mod_hz=_AM_FREQ, depth=_AM_DEPTH)

    mixed = wav_pitched + layer_am
    mixed = normalize_lufs(mixed, sr)
    mixed = peak_limit(mixed)
    mixed, _ = embed(mixed, sr)
    return mixed.astype(np.float32)


# ── arm C ────────────────────────────────────────────────────────────────────

def _arm_c(wav: np.ndarray, sr: int, text: str) -> np.ndarray:
    """Placebo: WORLD pass-through + matched-energy noise layer + jittered AM, no F0 pull."""
    from sap_stimulus.dramaturgy import segment
    from sap_stimulus.ducking import duck_to_snr
    from sap_stimulus.f0_pull import pull_f0
    from sap_stimulus.layers import amplitude_modulate, subharmonic_layer
    from sap_stimulus.placebo import apply_jittered_am, make_placebo_layer
    from sap_stimulus.watermark import embed

    duration_s = len(wav) / sr
    acts = segment(text, duration_s, wav=wav, sr=sr)

    # Need B-layer shape to match energy — build drone at constant amplitude
    layer_b = subharmonic_layer(acts, sr, len(wav))
    layer_b_am = amplitude_modulate(layer_b, sr, f_mod_hz=_AM_FREQ, depth=_AM_DEPTH)

    wav_world = pull_f0(wav, sr, acts=[], strength=0.0)   # vocoder pass-through, no pull
    placebo = make_placebo_layer(layer_b_am, sr)
    placebo_ducked, _ = duck_to_snr(wav_world, placebo, sr, target_snr_db=_SNR_DB)
    placebo_ducked = placebo_ducked[:len(wav_world)]
    placebo_jittered = apply_jittered_am(placebo_ducked, sr, am_depth=_AM_DEPTH)

    mixed = wav_world + placebo_jittered
    mixed = normalize_lufs(mixed, sr)
    mixed = peak_limit(mixed)
    mixed, _ = embed(mixed, sr)
    return mixed.astype(np.float32)


# ── synthesise ────────────────────────────────────────────────────────────────

def synthesise(text: str, arm: str = "B", out_path: Path | None = None) -> Path:
    """Render *text* through the requested arm pipeline and return the output path."""
    if not _MODEL.exists():
        raise FileNotFoundError(
            f"Piper model not found: {_MODEL}\n"
            "Download with:\n"
            "  python -c \"import urllib.request; urllib.request.urlretrieve("
            "'https://huggingface.co/rhasspy/piper-voices/resolve/main/"
            "en/en_US/ryan/high/en_US-ryan-high.onnx', 'models/piper/en_US-ryan-high.onnx')\""
        )

    arm = arm.upper()
    if arm not in ("A", "B", "C"):
        raise ValueError(f"arm must be A, B, or C — got {arm!r}")

    if out_path is None:
        out_path = Path(f"converse_arm{arm}.wav")

    tts = PiperBackend(_MODEL)

    # Step 1: TTS → temp WAV
    with tempfile.TemporaryDirectory() as tmp:
        base_wav_path = Path(tmp) / "base.wav"
        info = tts.synthesize(text, base_wav_path)
        wav, sr = load(base_wav_path)

    print(f"  TTS done  ({info.duration_s:.1f} s, {info.lufs:.1f} LUFS)")

    # Step 2: arm-specific post-processing
    if arm == "A":
        print("  Arm A: clean pass-through")
        processed = _arm_a(wav, sr)
    elif arm == "B":
        print("  Arm B: F0-pull + subharmonic drone + 10 Hz AM + watermark")
        processed = _arm_b(wav, sr, text)
    else:
        print("  Arm C: placebo (matched energy, jittered AM) + watermark")
        processed = _arm_c(wav, sr, text)

    save(processed, _SR_TARGET, out_path)
    print(f"  Saved → {out_path}")
    return out_path


# ── play ──────────────────────────────────────────────────────────────────────

def _play(path: Path) -> None:
    """Best-effort auto-play — works on Windows, macOS, Linux."""
    import platform
    import subprocess
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(str(path))
        elif system == "Darwin":
            subprocess.run(["afplay", str(path)], check=True)
        else:
            subprocess.run(["aplay", str(path)], check=True)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  [play] could not auto-play: {e}")
        print(f"  Open manually: {path.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("text", nargs="?", default=None, help="Text to synthesise")
    p.add_argument("--arm", default="B", choices=["A", "B", "C", "a", "b", "c"],
                   help="Processing arm: A=clean, B=entrainment (default), C=placebo")
    p.add_argument("--out", type=Path, default=None,
                   help="Output WAV path (default: converse_arm{X}.wav)")
    p.add_argument("--play", action="store_true",
                   help="Auto-play the WAV after rendering")
    p.add_argument("--scan", action="store_true",
                   help="Run the acoustic firewall scan on the output after rendering")
    return p.parse_args()


def main() -> None:
    args = _parse()

    arm_descriptions = {
        "A": "clean (no manipulation)",
        "B": "entrainment — F0-pull + 60-110 Hz subharmonic drone + 10 Hz coherent AM",
        "C": "placebo — matched energy, jittered AM, no F0 pull",
    }
    arm = args.arm.upper()

    if args.text:
        # Single-shot mode
        print(f"\nArm {arm}: {arm_descriptions[arm]}")
        out = synthesise(args.text, arm=arm, out_path=args.out)
        if args.play:
            _play(out)
        if args.scan:
            _scan(out)
    else:
        # Interactive loop
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  acoustic-persuasion-lab · TTS conversation                 ║")
        print("║                                                              ║")
        print("║  Commands:                                                   ║")
        print("║    :a  switch to Arm A (clean)                               ║")
        print("║    :b  switch to Arm B (entrainment — default)               ║")
        print("║    :c  switch to Arm C (placebo)                             ║")
        print("║    :scan  run firewall on last output                        ║")
        print("║    :play  play last output                                   ║")
        print("║    :q  quit                                                  ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"\nActive arm: {arm} — {arm_descriptions[arm]}\n")

        last_out: Path | None = None
        session_n = 0

        while True:
            try:
                raw = input("say> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not raw:
                continue
            if raw == ":q":
                print("Bye.")
                break
            if raw in (":a", ":b", ":c"):
                arm = raw[1].upper()
                print(f"→ Arm {arm}: {arm_descriptions[arm]}")
                continue
            if raw == ":scan":
                if last_out:
                    _scan(last_out)
                else:
                    print("No output yet.")
                continue
            if raw == ":play":
                if last_out:
                    _play(last_out)
                else:
                    print("No output yet.")
                continue

            session_n += 1
            out_path = Path(f"converse_{session_n:03d}_arm{arm}.wav")
            print()
            try:
                last_out = synthesise(raw, arm=arm, out_path=out_path)
                if args.play:
                    _play(last_out)
            except (RuntimeError, OSError, ValueError) as e:
                print(f"  [error] {e}")
            print()


def _scan(path: Path) -> None:
    """Run the acoustic firewall on *path* and print the result."""
    try:
        from acoustic_firewall.features import extract, _WINDOW_S
        from acoustic_firewall.model import load_model, predict
        wav, sr = load(path)
        duration_s = len(wav) / sr
        feats = extract(wav, sr)
        model_path = _ROOT / "packages" / "acoustic_firewall" / "models" / "detector_v1.joblib"
        if not model_path.exists():
            print("  [scan] model not trained yet — run: uv run python -m acoustic_firewall.train")
            return
        model = load_model(model_path)
        report = predict(feats, model)
        coherence_nan = report.f_mod_hz is None or (
            isinstance(report.f_mod_hz, float) and __import__("math").isnan(report.f_mod_hz)
        )
        print(f"  [scan] verdict={report.verdict}  score={report.score:.3f}"
              f"  f_mod_hz={report.f_mod_hz}  threshold={report.threshold_used}")
        if duration_s < _WINDOW_S:
            print(f"  [scan] ⚠  clip is {duration_s:.1f} s — shorter than the {_WINDOW_S:.0f} s "
                  f"analysis window, so inter_band_coherence_* features are nan and imputed "
                  f"to 0. Verdict is based on LF energy / kurtosis only. "
                  f"Use a sentence ≥ {_WINDOW_S:.0f} s for a coherence-based result.")
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"  [scan] error: {e}")

if __name__ == "__main__":
    main()
