"""Feature extraction for the acoustic firewall detector.

Inter-band modulation coherence is the discriminating feature:
natural speech AM is band-incoherent at a fixed rate; a globally injected AM
is perfectly coherent across bands.
"""
from __future__ import annotations

import librosa  # available via sap_core dep chain
import numpy as np
from scipy.signal import butter, decimate, hilbert as _hilbert, sosfilt, sosfiltfilt
from scipy.stats import kurtosis as _scipy_kurtosis

_WINDOW_S = 8.0
_HOP_S = 1.0
_N_BANDS = 32
_FMIN = 50.0
_FMAX = 8000.0
_ENV_SR = 200  # Hz, envelope decimation rate


def extract(wav: np.ndarray, sr: int) -> dict[str, float]:
    """Extract detection features from *wav* over 8 s windows / 1 s hop.

    Returns a flat dict of scalar features (nan for unavailable features).

    Features (Step 5):
    1. 32-band gammatone filterbank Hilbert envelopes, decimated to 200 Hz.
    2. Inter-band modulation coherence (1–30 Hz) — primary discriminating feature.
    3. Peak frequency and prominence of coherent modulation component.
    4. 40–120 Hz band energy ratio and temporal stationarity.
    5. Cross-correlation of LF component with speech F0 track.
    6. F0 pitch-class histogram χ² vs uniform (needs ≥ 20 s voiced; else nan).
    7. Spectral kurtosis mean/variance in 40–120 Hz.
    """
    wav = np.asarray(wav, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Gammatone filterbank → Hilbert envelopes → decimate to _ENV_SR
    # ------------------------------------------------------------------
    erb_lo = 21.4 * np.log10(1 + 0.00437 * _FMIN)
    erb_hi = 21.4 * np.log10(1 + 0.00437 * _FMAX)
    erb_centers = np.linspace(erb_lo, erb_hi, _N_BANDS)
    cf_hz = (10 ** (erb_centers / 21.4) - 1) / 0.00437  # shape (_N_BANDS,)

    decim_factor = max(1, sr // _ENV_SR)
    envelopes: list[np.ndarray] = []

    for fc in cf_hz:
        # scipy.signal.gammatone IIR returns b:(5,) a:(9,) on scipy ≥1.7 — the
        # single-biquad SOS path requires (3,)/(3,), so we go straight to the
        # Butterworth bandpass fallback (plan assumption A1).
        bw_lo = max(fc * 0.8, _FMIN)
        bw_hi = min(fc * 1.25, _FMAX)
        nyq = sr / 2.0
        sos_bw = butter(4, [bw_lo / nyq, bw_hi / nyq], btype="band", output="sos")
        filtered = sosfilt(sos_bw, wav)

        env = np.abs(_hilbert(filtered))

        # Chain decimation if factor is large (decimate handles ≤13 safely)
        if decim_factor > 13:
            step1 = max(2, decim_factor // 10)
            step2 = max(1, decim_factor // step1)
            env = decimate(env, step1, zero_phase=True)
            if step2 > 1:
                env = decimate(env, step2, zero_phase=True)
        elif decim_factor > 1:
            env = decimate(env, decim_factor, zero_phase=True)

        envelopes.append(env.astype(np.float32))

    # ------------------------------------------------------------------
    # 2. Inter-band modulation coherence (vector-strength synchrony, 1–30 Hz)
    # ------------------------------------------------------------------
    win_len = int(_WINDOW_S * _ENV_SR)   # frames per window
    hop_len = int(_HOP_S * _ENV_SR)      # frames per hop
    env_len = min(len(e) for e in envelopes)
    env_mat = np.stack([e[:env_len] for e in envelopes], axis=0)  # (N_BANDS, env_len)

    n_wins = max(1, (env_len - win_len) // hop_len + 1)
    coherence_scores: list[np.ndarray] = []
    freqs: np.ndarray | None = None

    for w in range(n_wins):
        start = w * hop_len
        seg = env_mat[:, start : start + win_len]
        if seg.shape[1] < win_len // 2:
            break
        seg = seg - seg.mean(axis=1, keepdims=True)  # remove DC per band
        fft_seg = np.fft.rfft(seg, axis=1)           # (N_BANDS, n_freq)
        if freqs is None:
            freqs = np.fft.rfftfreq(seg.shape[1], d=1.0 / _ENV_SR)
        mean_complex = fft_seg.mean(axis=0)
        mean_abs = np.abs(fft_seg).mean(axis=0)
        sync = np.abs(mean_complex) / (mean_abs + 1e-12)  # vector strength ∈ [0, 1]
        coherence_scores.append(sync)

    if len(coherence_scores) == 0 or freqs is None:
        inter_band_coherence_mean = float("nan")
        inter_band_coherence_peak_hz = float("nan")
        inter_band_coherence_prominence_db = float("nan")
    else:
        mean_sync = np.mean(coherence_scores, axis=0)
        mask = (freqs >= 1.0) & (freqs <= 30.0)
        sync_band = mean_sync[mask]
        freqs_band = freqs[mask]

        if len(sync_band) == 0:
            inter_band_coherence_mean = float("nan")
            inter_band_coherence_peak_hz = float("nan")
            inter_band_coherence_prominence_db = float("nan")
        else:
            peak_idx = int(np.argmax(sync_band))
            inter_band_coherence_mean = float(sync_band[peak_idx])
            inter_band_coherence_peak_hz = float(freqs_band[peak_idx])
            non_peak = np.delete(sync_band, peak_idx)
            baseline = float(np.median(non_peak)) if len(non_peak) > 0 else 1e-6
            inter_band_coherence_prominence_db = float(
                20.0 * np.log10((sync_band[peak_idx] + 1e-12) / (baseline + 1e-12))
            )

    # ------------------------------------------------------------------
    # 3. LF band energy ratio and temporal stationarity (40–120 Hz)
    # ------------------------------------------------------------------
    nyq = sr / 2.0
    sos_lf = butter(4, [40.0 / nyq, 120.0 / nyq], btype="band", output="sos")
    wav_lf = sosfiltfilt(sos_lf, wav).astype(np.float32)

    total_power = float(np.mean(wav ** 2)) + 1e-12
    lf_power = float(np.mean(wav_lf.astype(np.float64) ** 2))
    lf_band_energy_ratio = lf_power / total_power

    frame_hop = int(0.1 * sr)  # 100 ms frames
    lf_frame_powers = [
        float(np.mean(wav_lf[i : i + frame_hop] ** 2))
        for i in range(0, len(wav_lf) - frame_hop, frame_hop)
    ]
    if len(lf_frame_powers) > 1:
        cv = float(np.std(lf_frame_powers)) / (float(np.mean(lf_frame_powers)) + 1e-12)
        lf_band_stationarity = float(1.0 / (1.0 + cv))
    else:
        lf_band_stationarity = float("nan")

    # ------------------------------------------------------------------
    # 4. LF–F0 cross-correlation
    # ------------------------------------------------------------------
    frame_len_f0 = 2048
    hop_len_f0 = 512
    wav_f32 = wav.astype(np.float32)
    f0 = librosa.yin(wav_f32, fmin=60.0, fmax=400.0, sr=sr,
                     frame_length=frame_len_f0, hop_length=hop_len_f0)

    lf_env = np.array([
        float(np.mean(np.abs(wav_lf[i * hop_len_f0 : i * hop_len_f0 + frame_len_f0])))
        for i in range(len(f0))
    ], dtype=np.float64)

    voiced = (f0 > 65.0) & (f0 < 380.0)
    if voiced.sum() > 10:
        from scipy.stats import pearsonr
        corr, _ = pearsonr(f0[voiced].astype(np.float64), lf_env[voiced])
        lf_f0_crosscorr = float(corr)
    else:
        lf_f0_crosscorr = float("nan")

    # ------------------------------------------------------------------
    # 5. F0 pitch-class histogram χ² vs uniform
    # ------------------------------------------------------------------
    voiced_f0 = f0[voiced]
    min_voiced_frames = int(20.0 * sr / hop_len_f0)
    if len(voiced_f0) >= min_voiced_frames:
        midi = 12.0 * np.log2(voiced_f0.astype(np.float64) / 440.0) + 69.0
        pitch_classes = np.round(midi).astype(int) % 12
        counts = np.bincount(pitch_classes, minlength=12).astype(float)
        from scipy.stats import chisquare
        chi2_stat, _ = chisquare(counts)
        f0_pitchclass_chi2 = float(chi2_stat)
    else:
        f0_pitchclass_chi2 = float("nan")

    # ------------------------------------------------------------------
    # 6. Spectral kurtosis in 40–120 Hz
    # ------------------------------------------------------------------
    kurtoses = [
        float(_scipy_kurtosis(wav_lf[i : i + frame_hop].astype(np.float64)))
        for i in range(0, len(wav_lf) - frame_hop, frame_hop)
        if len(wav_lf[i : i + frame_hop]) == frame_hop
    ]
    if len(kurtoses) >= 2:
        spectral_kurtosis_mean = float(np.mean(kurtoses))
        spectral_kurtosis_var = float(np.var(kurtoses))
    else:
        spectral_kurtosis_mean = float("nan")
        spectral_kurtosis_var = float("nan")

    # Key names and insertion order are frozen — model.py uses list(features.values())
    return {
        "inter_band_coherence_mean": inter_band_coherence_mean,
        "inter_band_coherence_peak_hz": inter_band_coherence_peak_hz,
        "inter_band_coherence_prominence_db": inter_band_coherence_prominence_db,
        "lf_band_energy_ratio": lf_band_energy_ratio,
        "lf_band_stationarity": lf_band_stationarity,
        "lf_f0_crosscorr": lf_f0_crosscorr,
        "f0_pitchclass_chi2": f0_pitchclass_chi2,
        "spectral_kurtosis_mean": spectral_kurtosis_mean,
        "spectral_kurtosis_var": spectral_kurtosis_var,
    }
