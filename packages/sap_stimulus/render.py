"""CLI entry-point for stimulus generation.

Usage:
    python -m sap_stimulus.render \\
        --corpus corpus/scripts/corpus_v1.json \\
        --arms A B C \\
        --out corpus/stimuli/

Arm A: bit-identical copy of base render (control).
Arm B: F0-pull + subharmonic drone + coherent AM + DSSS watermark.
Arm C: placebo (matched energy/spectrum, jittered AM, no F0 pull) + DSSS watermark.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from sap_core.audio import load, save, sha256
from sap_core.loudness import normalize_lufs, peak_limit


def _build_arm_b(
    base_wav: np.ndarray,
    sr: int,
    text: str,
    item_id: str,
    snr_db: float,
    f_mod_hz: float,
    am_depth: float,
    pull_strength: float,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Return (mixed_wav, manifest_info, raw_layer_before_ducking)."""
    from sap_stimulus.dramaturgy import segment
    from sap_stimulus.f0_pull import pull_f0
    from sap_stimulus.layers import subharmonic_layer, amplitude_modulate
    from sap_stimulus.ducking import duck_to_snr
    from sap_stimulus.watermark import embed

    acts = segment(text, len(base_wav) / sr, wav=base_wav, sr=sr)
    wav_pitched = pull_f0(base_wav, sr, acts, strength=pull_strength)
    layer = subharmonic_layer(acts, sr, len(wav_pitched))
    layer_am = amplitude_modulate(layer, sr, f_mod_hz=f_mod_hz, depth=am_depth)
    layer_scaled, achieved_snr = duck_to_snr(wav_pitched, layer_am, sr, target_snr_db=snr_db)
    layer_scaled = layer_scaled[:len(wav_pitched)]
    mixed = wav_pitched + layer_scaled
    mixed = normalize_lufs(mixed, sr)
    mixed = peak_limit(mixed)
    mixed, stim_id = embed(mixed, sr)
    # Save the scaled layer as a sidecar for QA verification

    info = {
        "arm": "B",
        "item_id": item_id,
        "stimulus_id": stim_id,
        "achieved_snr_db": round(achieved_snr, 2),
        "f_mod_hz": f_mod_hz,
        "am_depth": am_depth,
        "pull_strength": pull_strength,
        "acts": [{"name": a.name, "start_s": round(a.start_s, 3),
                  "end_s": round(a.end_s, 3), "chord": a.chord} for a in acts],
    }
    return mixed, info, layer_scaled  # caller saves layer_scaled as sidecar


def _build_arm_c(
    base_wav: np.ndarray,
    sr: int,
    arm_b_layer: np.ndarray,
    item_id: str,
    snr_db: float,
) -> tuple[np.ndarray, dict]:
    """Arm C: WORLD pass-through (strength=0) + placebo layer.

    The WORLD pass-through matches the vocoder residual of Arm B so that
    rms(B-A) / rms(C-A) reflects only the difference in added-layer energy,
    not the WORLD reconstruction artifact. Without this, C-A contains only
    the placebo layer while B-A contains layer + large vocoder residual,
    making the ±0.5 dB check meaningless.
    """
    from sap_stimulus.f0_pull import pull_f0
    from sap_stimulus.placebo import make_placebo_layer
    from sap_stimulus.ducking import duck_to_snr
    from sap_stimulus.watermark import embed

    # WORLD pass-through: analyze/synthesize with no pitch change (strength=0)
    # This matches the vocoder residual of Arm B without any harmonic manipulation.
    wav_world = pull_f0(base_wav, sr, acts=[], strength=0.0)

    placebo = make_placebo_layer(arm_b_layer, sr)
    placebo_scaled, achieved_snr = duck_to_snr(wav_world, placebo, sr, target_snr_db=snr_db)
    placebo_scaled = placebo_scaled[:len(wav_world)]
    mixed = wav_world + placebo_scaled
    mixed = normalize_lufs(mixed, sr)
    mixed = peak_limit(mixed)
    mixed, stim_id = embed(mixed, sr)

    info = {
        "arm": "C",
        "item_id": item_id,
        "stimulus_id": stim_id,
        "achieved_snr_db": round(achieved_snr, 2),
        "world_passthrough": True,
    }
    return mixed, info


def render_all(
    corpus_path: Path,
    base_renders_dir: Path,
    out_dir: Path,
    arms: list[str],
    snr_db: float = -31.0,
    f_mod_hz: float = 10.0,
    am_depth: float = 0.03,
    pull_strength: float = 0.175,
) -> None:
    corpus = json.loads(corpus_path.read_text())
    items = corpus["items"]

    for item in items:
        iid = item["id"]
        base_path = base_renders_dir / f"{iid}.wav"
        if not base_path.exists():
            print(f"[WARN] base render missing: {base_path} — skipping")
            continue

        base_wav, sr = load(base_path, target_sr=48_000)
        base_hash = sha256(base_path)
        is_processable = item["block"] in ("fallacy", "practice")

        # Pre-build B layer once so C can reuse it (avoid double WORLD analysis)
        arm_b_layer: np.ndarray | None = None

        for arm in arms:
            arm_dir = out_dir / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            out_wav = arm_dir / f"{iid}.wav"

            if arm == "A":
                shutil.copy2(base_path, out_wav)
                info: dict = {"arm": "A", "item_id": iid, "source_sha256": base_hash}

            elif arm == "B" and is_processable:
                mixed, info, arm_b_layer = _build_arm_b(
                    base_wav, sr, item["text"], iid,
                    snr_db, f_mod_hz, am_depth, pull_strength,
                )
                save(mixed, sr, out_wav)
                # Save the scaled layer as a sidecar for QA AM verification
                layer_sidecar = arm_dir / f"{iid}.layer.wav"
                save(arm_b_layer, sr, layer_sidecar)
                info["source_sha256"] = base_hash

            elif arm == "C" and is_processable:
                if arm_b_layer is None:
                    # B not yet built in this run; build layer only
                    from sap_stimulus.dramaturgy import segment
                    from sap_stimulus.layers import subharmonic_layer, amplitude_modulate
                    acts = segment(item["text"], len(base_wav) / sr)
                    arm_b_layer = amplitude_modulate(
                        subharmonic_layer(acts, sr, len(base_wav)), sr,
                        f_mod_hz=f_mod_hz, depth=am_depth,
                    )
                mixed, info = _build_arm_c(base_wav, sr, arm_b_layer, iid, snr_db)
                save(mixed, sr, out_wav)
                info["source_sha256"] = base_hash

            else:
                # Trust items and unprocessable blocks: same as A
                shutil.copy2(base_path, out_wav)
                info = {"arm": arm, "item_id": iid, "source_sha256": base_hash,
                        "note": "no_processing"}

            manifest = arm_dir / f"{iid}.manifest.json"
            manifest.write_text(json.dumps(info, indent=2))
            snr_str = f"  SNR={info.get('achieved_snr_db', 'n/a')} dB" if arm != "A" else ""
            print(f"  [{arm}] {iid}{snr_str}")

    print(f"\nDone. Stimuli in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render SAP stimulus arms A/B/C")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-renders", type=Path, default=Path("corpus/base_renders"))
    ap.add_argument("--snr-db", type=float, default=-31.0,
                    help="Target layer SNR relative to speech (dB). Override after threshold pilot.")
    ap.add_argument("--f-mod-hz", type=float, default=10.0)
    ap.add_argument("--am-depth", type=float, default=0.03)
    ap.add_argument("--pull-strength", type=float, default=0.175)
    args = ap.parse_args()

    render_all(
        corpus_path=args.corpus,
        base_renders_dir=args.base_renders,
        out_dir=args.out,
        arms=args.arms,
        snr_db=args.snr_db,
        f_mod_hz=args.f_mod_hz,
        am_depth=args.am_depth,
        pull_strength=args.pull_strength,
    )


if __name__ == "__main__":
    main()
