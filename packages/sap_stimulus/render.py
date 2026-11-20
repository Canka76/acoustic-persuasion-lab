"""CLI entry point: python -m sap_stimulus.render --corpus ... --arms A B C --out ..."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import uuid
from pathlib import Path

import numpy as np

from sap_core.audio import load, save
from sap_core.loudness import integrated_lufs, normalize_lufs, peak_limit
from sap_stimulus.dramaturgy import segment
from sap_stimulus.ducking import duck_to_snr
from sap_stimulus.f0_pull import pull_f0
from sap_stimulus.layers import amplitude_modulate, subharmonic_layer
from sap_stimulus.placebo import make_placebo_layer
from sap_stimulus.tts import PiperBackend
from sap_stimulus.watermark import embed

_TARGET_LUFS = -23.0
_TARGET_PEAK_DBTP = -1.0
_TARGET_SNR_DB = -31.0
_AM_F_MOD = 10.0
_AM_DEPTH = 0.03
_SR = 48_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def render_base(item: dict, base_dir: Path) -> Path:
    """Render base TTS audio for *item* if not already present."""
    out = base_dir / f"{item['id']}.wav"
    if out.exists():
        return out
    backend = PiperBackend()
    info = backend.synthesize(item["text"], out)
    # Trim silence, loudness-normalise, peak-limit
    wav, sr = load(out, sr=_SR)
    # Trim 200 ms silence at head/tail
    from scipy.signal import find_peaks
    rms_window = int(0.020 * sr)
    energy = np.array([
        np.sqrt(np.mean(wav[i:i+rms_window]**2))
        for i in range(0, len(wav) - rms_window, rms_window)
    ])
    threshold = np.percentile(energy, 10) * 3
    voiced = np.where(energy > threshold)[0]
    if len(voiced):
        head = max(0, voiced[0] * rms_window - int(0.2 * sr))
        tail = min(len(wav), (voiced[-1] + 1) * rms_window + int(0.2 * sr))
        wav = wav[head:tail]
    wav = normalize_lufs(wav, sr, _TARGET_LUFS)
    wav = peak_limit(wav, sr, _TARGET_PEAK_DBTP)
    save(out, wav, sr)
    sha = _sha256(out)
    (out.with_suffix(".sha256")).write_text(sha)
    return out


def render_arm(
    item: dict,
    arm: str,
    base_path: Path,
    out_dir: Path,
    operating_point_path: Path | None = None,
) -> None:
    target_snr = _TARGET_SNR_DB
    if operating_point_path and operating_point_path.exists():
        op = json.loads(operating_point_path.read_text())
        target_snr = op.get("target_snr_db", _TARGET_SNR_DB)

    wav, sr = load(base_path, sr=_SR)
    n_samples = len(wav)

    stim_uuid = uuid.uuid4()
    manifest: dict = {
        "id": item["id"],
        "arm": arm,
        "source_sha256": _sha256(base_path),
        "git_commit": _git_commit(),
        "generator_version": "0.1.0",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item['id']}.wav"

    if arm == "A":
        # Control: bit-identical copy of base render
        import shutil
        shutil.copy2(base_path, out_path)
        manifest["f_mod_hz"] = None
        manifest["depth"] = None
        manifest["achieved_snr_db"] = None
        manifest["chord_plan"] = None
    else:
        acts = segment(item["text"], len(wav) / sr, wav, sr)
        sub_layer = subharmonic_layer(acts, sr, n_samples)

        if arm == "B":
            wav_pulled = pull_f0(wav, sr, acts)
            am_layer = amplitude_modulate(sub_layer, sr, _AM_F_MOD, _AM_DEPTH)
            ducked, achieved_snr = duck_to_snr(wav_pulled, am_layer, sr, target_snr)
            final = wav_pulled + ducked
            manifest["f_mod_hz"] = _AM_F_MOD
            manifest["depth"] = _AM_DEPTH
            manifest["achieved_snr_db"] = achieved_snr
            manifest["chord_plan"] = [{"name": a.name, "chord": a.chord, "start_s": a.start_s, "end_s": a.end_s} for a in acts]
        elif arm == "C":
            rng = np.random.default_rng(42)
            placebo = make_placebo_layer(sub_layer, sr, _AM_DEPTH, rng)
            ducked, achieved_snr = duck_to_snr(wav, placebo, sr, target_snr)
            final = wav + ducked
            manifest["f_mod_hz"] = "jittered_0.5-30"
            manifest["depth"] = _AM_DEPTH
            manifest["achieved_snr_db"] = achieved_snr
            manifest["chord_plan"] = None
        else:
            raise ValueError(f"Unknown arm: {arm}")

        final = normalize_lufs(final, sr, _TARGET_LUFS)
        final = peak_limit(final, sr, _TARGET_PEAK_DBTP)
        watermarked, stim_uuid = embed(final, sr, stim_uuid)
        save(out_path, watermarked, sr)

    manifest["stimulus_uuid"] = str(stim_uuid)
    (out_dir / f"{item['id']}.manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render stimulus arms from corpus.")
    parser.add_argument("--corpus", required=True, help="Path to corpus_v1.json")
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--out", required=True, help="Output directory root")
    parser.add_argument("--base-renders", default="corpus/base_renders")
    parser.add_argument("--operating-point", default=None)
    args = parser.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    base_dir = Path(args.base_renders)
    out_root = Path(args.out)
    op_path = Path(args.operating_point) if args.operating_point else None

    items = corpus.get("items", [])
    for item in items:
        base_path = render_base(item, base_dir)
        for arm in args.arms:
            render_arm(item, arm, base_path, out_root / arm, op_path)
            print(f"[render] {arm}/{item['id']} → done")


if __name__ == "__main__":
    main()
