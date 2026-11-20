"""Speech quality metrics: pystoi ESTOI, segmental SNR, log-spectral distance.

CLI: python -m acoustic_firewall.quality <cleaned> <reference>
Acceptance thresholds (Step 6): ESTOI >= 0.85, segSNR >= 15 dB, LSD <= 1.0 dB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def estoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    """Extended short-time objective intelligibility (ESTOI) in [0, 1]."""
    from pystoi import stoi as _stoi  # type: ignore[import]
    min_len = min(len(ref), len(deg))
    return float(_stoi(ref[:min_len], deg[:min_len], sr, extended=True))


def segmental_snr(ref: np.ndarray, deg: np.ndarray, sr: int, frame_s: float = 0.020) -> float:
    """Segmental SNR in dB (mean over voiced 20 ms frames, floor −10 dB, ceil 35 dB)."""
    frame_len = int(frame_s * sr)
    n = min(len(ref), len(deg))
    snrs: list[float] = []
    for start in range(0, n - frame_len, frame_len):
        r = ref[start : start + frame_len]
        d = deg[start : start + frame_len]
        noise = d - r
        signal_power = float(np.mean(r**2))
        noise_power = float(np.mean(noise**2))
        if signal_power < 1e-10:
            continue
        snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))
        snrs.append(float(np.clip(snr, -10.0, 35.0)))
    return float(np.mean(snrs)) if snrs else float("nan")


def log_spectral_distance(ref: np.ndarray, deg: np.ndarray, sr: int, frame_s: float = 0.020) -> float:
    """Log-spectral distance in dB (mean over frames)."""
    frame_len = int(frame_s * sr)
    n = min(len(ref), len(deg))
    lsds: list[float] = []
    for start in range(0, n - frame_len, frame_len):
        r = np.abs(np.fft.rfft(ref[start : start + frame_len])) + 1e-12
        d = np.abs(np.fft.rfft(deg[start : start + frame_len])) + 1e-12
        lsd = float(np.sqrt(np.mean((20.0 * np.log10(r / d))**2)))
        lsds.append(lsd)
    return float(np.mean(lsds)) if lsds else float("nan")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m acoustic_firewall.quality <cleaned> <reference>")
        sys.exit(1)

    from sap_core.audio import load
    cleaned_path, ref_path = Path(sys.argv[1]), Path(sys.argv[2])
    deg, sr = load(cleaned_path)
    ref, sr_ref = load(ref_path, sr=sr)

    e = estoi(ref, deg, sr)
    s = segmental_snr(ref, deg, sr)
    l = log_spectral_distance(ref, deg, sr)

    print(f"ESTOI:   {e:.4f}  ({'PASS' if e >= 0.85 else 'FAIL'} >= 0.85)")
    print(f"segSNR:  {s:.2f} dB  ({'PASS' if s >= 15.0 else 'FAIL'} >= 15 dB)")
    print(f"LSD:     {l:.2f} dB  ({'PASS' if l <= 1.0 else 'FAIL'} <= 1.0 dB)")

    ok = e >= 0.85 and s >= 15.0 and l <= 1.0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
