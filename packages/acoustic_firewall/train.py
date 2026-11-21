"""Train the acoustic firewall detector (GBM) on rendered stimuli.

Usage:
    uv run python -m acoustic_firewall.train \
        --stimuli corpus/stimuli \
        --corpus corpus/scripts/corpus_v1.json \
        [--out packages/acoustic_firewall/models/detector_v1.joblib]

Arm A WAVs -> label 0 (clean).
Arm B WAVs -> label 1 (entrainment_detected).
Trust items (block == "trust") are excluded (no ground-truth label).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _collect(stimuli_dir: Path, corpus_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): feature matrix and binary label vector."""
    from acoustic_firewall.features import extract
    from sap_core.audio import load

    with corpus_path.open() as f:
        items = [
            it for it in json.load(f)["items"]
            if it["block"] in ("fallacy", "practice")  # exclude trust
        ]

    X_rows: list[list[float]] = []
    y_labels: list[int] = []
    for item in items:
        for arm, label in [("A", 0), ("B", 1)]:
            wav_path = stimuli_dir / arm / f"{item['id']}.wav"
            if not wav_path.exists():
                print(f"  skip: {wav_path} not found", file=sys.stderr)
                continue
            wav, sr = load(wav_path)
            feats = extract(wav, sr)
            feat_vec = list(feats.values())
            # Warn on unexpected nan (f0_pitchclass_chi2 nan is expected for short audio)
            unexpected_nan = [
                k for k, v in feats.items()
                if np.isnan(v) and k != "f0_pitchclass_chi2"
            ]
            if unexpected_nan:
                print(f"  warn: unexpected nan in {wav_path}: {unexpected_nan}", file=sys.stderr)
            # Replace nan with 0 (matches predict() imputation)
            X_rows.append([0.0 if np.isnan(v) else v for v in feat_vec])
            y_labels.append(label)

    return np.array(X_rows, dtype=np.float32), np.array(y_labels, dtype=np.int32)


def train(stimuli_dir: Path, corpus_path: Path, out_path: Path) -> None:
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier

    print("Extracting features…")
    X, y = _collect(stimuli_dir, corpus_path)
    if len(X) == 0:
        print("No stimuli found. Run sap_stimulus.render first.", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(X)} samples ({int(y.sum())} positive / {int((y == 0).sum())} negative)")

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
    clf.fit(X, y)
    print(f"  train accuracy: {clf.score(X, y):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)
    print(f"Model saved -> {out_path}")


def main() -> None:
    _DEFAULT_STIMULI = Path("corpus/stimuli")
    _DEFAULT_CORPUS = Path("corpus/scripts/corpus_v1.json")
    _DEFAULT_OUT = Path(__file__).parent / "models" / "detector_v1.joblib"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimuli", type=Path, default=_DEFAULT_STIMULI)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()
    train(args.stimuli, args.corpus, args.out)


if __name__ == "__main__":
    main()
