"""GBM classifier stub — trained in Step 5."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "detector_v1.joblib"


@dataclass
class DetectReport:
    verdict: str          # "entrainment_detected" | "clean"
    score: float          # classifier probability [0, 1]
    f_mod_hz: float | None
    threshold_used: float


def load_model(path: Path = _DEFAULT_MODEL_PATH) -> Any:
    """Load persisted sklearn model from *path*."""
    import joblib
    return joblib.load(path)


def predict(features: dict[str, float], model: Any, threshold: float = 0.5) -> DetectReport:
    """Run classifier on feature dict. Returns DetectReport."""
    feature_vec = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
    # Replace nan with 0 for inference
    feature_vec = np.nan_to_num(feature_vec, nan=0.0)
    score = float(model.predict_proba(feature_vec)[0, 1])
    verdict = "entrainment_detected" if score >= threshold else "clean"
    f_mod = features.get("inter_band_coherence_peak_hz")
    return DetectReport(verdict=verdict, score=score, f_mod_hz=f_mod, threshold_used=threshold)
