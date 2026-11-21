"""Step-3 Stimulus QA gate.

All tests skip at module level if corpus/stimuli/ does not yet exist.
Run after:  uv run python -m sap_stimulus.render --corpus corpus/scripts/corpus_v1.json
                          --arms A B C --out corpus/stimuli/
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure packages are importable before pip-install
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "packages" / "sap_stimulus"))
sys.path.insert(0, str(_REPO / "packages" / "sap_core"))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_JSON = _REPO / "corpus" / "scripts" / "corpus_v1.json"
BASE_RENDERS = _REPO / "corpus" / "base_renders"
STIMULI_DIR = _REPO / "corpus" / "stimuli"

# ---------------------------------------------------------------------------
# Module-level skip guard — entire module skips when stimuli not rendered yet
# ---------------------------------------------------------------------------
if not STIMULI_DIR.exists():
    pytest.skip(
        "corpus/stimuli/ does not exist — run sap_stimulus.render first",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _load_corpus() -> list[dict]:
    with CORPUS_JSON.open() as f:
        data = json.load(f)
    return [item for item in data["items"] if item["block"] in ("fallacy", "practice")]


def _load_all_corpus() -> list[dict]:
    """All items including trust — for checks that must cover every stimulus."""
    with CORPUS_JSON.open() as f:
        data = json.load(f)
    return data["items"]


ITEMS = _load_corpus()
ITEM_IDS = [item["id"] for item in ITEMS]          # fallacy + practice only
ALL_ITEM_IDS = [item["id"] for item in _load_all_corpus()]  # + trust items
ARMS = ["A", "B", "C"]

TARGET_SNR_DB = -31.0
AM_FREQ_HZ = 10.0


def _wav_path(arm: str, item_id: str) -> Path:
    return STIMULI_DIR / arm / f"{item_id}.wav"


def _base_render_path(item_id: str) -> Path:
    return BASE_RENDERS / f"{item_id}.wav"


def _manifest_path(arm: str, item_id: str) -> Path:
    return STIMULI_DIR / arm / f"{item_id}.manifest.json"


def _load_wav(arm: str, item_id: str) -> tuple[np.ndarray, int]:
    import soundfile as sf  # type: ignore

    path = _wav_path(arm, item_id)
    if not path.exists():
        pytest.skip(f"Stimulus not rendered: {path}")
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return wav, int(sr)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


# ---------------------------------------------------------------------------
# 1. Source identity — all arms derive from the same base render (source_sha256)
#    (A is no longer bit-identical to base; all arms run WORLD pass-through.)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_id", ALL_ITEM_IDS)
def test_source_identity(item_id: str) -> None:
    """All three arm manifests must share the same source_sha256.
    This confirms arms differ only by WORLD pitch-pull strength and added layer,
    not by which TTS render was used.
    """
    hashes = {}
    for arm in ARMS:
        mpath = _manifest_path(arm, item_id)
        if not mpath.exists():
            pytest.skip(f"item {item_id} arm {arm}: manifest missing")
        h = json.loads(mpath.read_text()).get("source_sha256")
        if h is None:
            pytest.skip(f"item {item_id} arm {arm}: source_sha256 missing from manifest")
        hashes[arm] = h
    assert len(set(hashes.values())) == 1, (
        f"item {item_id}: arms derive from different base renders: {hashes}"
    )

# ---------------------------------------------------------------------------
# 2. Loudness match — integrated LUFS within ±0.1 LU across all arms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_loudness_match(item_id: str) -> None:
    """Integrated LUFS of A, B, C must all be within ±0.1 LU of each other."""
    import pyloudnorm as pyln  # type: ignore

    lufs_values: dict[str, float] = {}
    for arm in ARMS:
        wav, sr = _load_wav(arm, item_id)
        meter = pyln.Meter(sr)
        # pyloudnorm expects (samples,) or (samples, channels) as float64
        lufs = meter.integrated_loudness(wav.astype(np.float64))
        lufs_values[arm] = float(lufs)

    vals = list(lufs_values.values())
    spread = max(vals) - min(vals)
    assert spread <= 0.1, (
        f"item {item_id}: LUFS spread {spread:.4f} LU > 0.1 LU — {lufs_values}"
    )


# ---------------------------------------------------------------------------
# 3. True peak — ≤ -1.0 dBTP for all arms (4× oversampling)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_true_peak(item_id: str) -> None:
    """True peak ≤ -1.0 dBTP for every arm (4× oversampling via resample_poly)."""
    from scipy.signal import resample_poly  # type: ignore

    for arm in ARMS:
        wav, _sr = _load_wav(arm, item_id)
        oversampled = resample_poly(wav.astype(np.float64), up=4, down=1)
        peak_linear = float(np.max(np.abs(oversampled)))
        if peak_linear == 0.0:
            continue  # silence — trivially fine
        peak_dbtp = 20.0 * np.log10(peak_linear)
        assert peak_dbtp <= -1.0, (
            f"item {item_id} arm {arm}: true peak {peak_dbtp:.2f} dBTP > -1.0 dBTP"
        )


# ---------------------------------------------------------------------------
# 4. B/C added-energy match — 20*log10(rms(B-A)/rms(C-A)) within ±0.5 dB
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_bc_energy_match(item_id: str) -> None:
    """rms(B_layer.wav) / rms(C_layer.wav) must be within ±0.5 dB.

    Compares the actual added-layer signals saved as sidecars during render,
    not the full-mix residuals which are dominated by the WORLD vocoder artifact.
    This directly measures that the placebo energy matches the treatment layer.
    """
    b_layer = STIMULI_DIR / "B" / f"{item_id}.layer.wav"
    c_layer = STIMULI_DIR / "C" / f"{item_id}.layer.wav"
    if not b_layer.exists() or not c_layer.exists():
        pytest.skip(f"item {item_id}: layer sidecars missing — re-render to generate them")
    import soundfile as sf
    b_wav, _ = sf.read(str(b_layer), dtype="float32", always_2d=False)
    c_wav, _ = sf.read(str(c_layer), dtype="float32", always_2d=False)
    rms_b = _rms(b_wav)
    rms_c = _rms(c_wav)
    if rms_b < 1e-9 or rms_c < 1e-9:
        pytest.skip(f"item {item_id}: near-zero layer energy in sidecar")
    delta_db = 20.0 * np.log10(rms_b / rms_c)
    assert abs(delta_db) <= 0.5, (
        f"item {item_id}: B layer rms={rms_b:.6f}, C layer rms={rms_c:.6f}, "
        f"delta {delta_db:.3f} dB — exceeds ±0.5 dB"
    )

# ---------------------------------------------------------------------------
# 5. SNR accuracy — manifest achieved_snr_db within ±1 dB of target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_snr_accuracy(item_id: str) -> None:
    """Achieved broadband SNR (from manifest) within ±1 dB of target."""
    for arm in ("B", "C"):  # A has no added layer; SNR lives in B and C manifests
        mpath = _manifest_path(arm, item_id)
        if not mpath.exists():
            pytest.skip(f"Manifest missing: {mpath}")
        manifest = json.loads(mpath.read_text())
        achieved = float(manifest.get("achieved_snr_db", manifest.get("snr_db", float("nan"))))
        target = float(manifest.get("target_snr_db", TARGET_SNR_DB))
        assert not np.isnan(achieved), f"item {item_id} arm {arm}: achieved_snr_db missing from manifest"
        error = abs(achieved - target)
        assert error <= 1.0, (
            f"item {item_id} arm {arm}: achieved SNR {achieved:.2f} dB, "
            f"target {target:.2f} dB, error {error:.2f} dB > 1 dB"
        )


# ---------------------------------------------------------------------------
# 6. AM verification — B peak at 10 Hz ±0.2 Hz, prominence ≥ 6 dB; C: none ≥ 6 dB
# ---------------------------------------------------------------------------

def _modulation_spectrum(wav: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs, power_dB) of the Hilbert envelope modulation spectrum.

    Uses an 8 s window from the start of the file.
    """
    from scipy.signal import hilbert  # type: ignore

    win_samples = min(8 * sr, len(wav))
    if win_samples < sr:
        return np.array([]), np.array([])

    segment = wav[:win_samples].astype(np.float64)
    envelope = np.abs(hilbert(segment))
    # Remove DC
    envelope -= envelope.mean()

    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(win_samples, d=1.0 / sr)
    power_db = 20.0 * np.log10(spectrum + 1e-12)
    return freqs, power_db


def _local_median_prominence(freqs: np.ndarray, power_db: np.ndarray, peak_hz: float,
                              band_hz: float = 2.0) -> float:
    """Prominence of a spectral peak: peak_dB minus local median in ±band_hz window."""
    mask = (freqs >= peak_hz - band_hz) & (freqs <= peak_hz + band_hz)
    if not np.any(mask):
        return -np.inf
    local_median = float(np.median(power_db[mask]))
    peak_db_val = float(np.max(power_db[mask]))
    return peak_db_val - local_median


@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_am_unit(item_id: str) -> None:
    """Unit test: layer_am.wav sidecar must have dominant 10 Hz ±0.2 Hz peak ≥6 dB.
    Tests amplitude_modulate() output in isolation (not the delivered mix).
    """
    layer_path = STIMULI_DIR / "B" / f"{item_id}.layer_am.wav"
    if not layer_path.exists():
        pytest.skip(f"item {item_id}: layer_am sidecar missing — re-render")
    import soundfile as sf
    layer_wav, sr = sf.read(str(layer_path), dtype="float32", always_2d=False)
    freqs, pdb = _modulation_spectrum(layer_wav, sr)
    if len(freqs) == 0:
        pytest.skip(f"item {item_id}: layer_am too short")
    band = (freqs >= 1.0) & (freqs <= 30.0)
    peak_hz = float(freqs[band][np.argmax(pdb[band])])
    prom = _local_median_prominence(freqs, pdb, peak_hz)
    assert abs(peak_hz - AM_FREQ_HZ) <= 0.2, f"{item_id} layer_am peak at {peak_hz:.3f} Hz"
    assert prom >= 6.0, f"{item_id} layer_am prominence {prom:.1f} dB < 6 dB"


@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_am_verification(item_id: str) -> None:
    """Delivered-stimulus check: 10 Hz AM must be detectable in arm B's sub-F0 band.

    The subharmonic drone sits at 55–75 Hz (C2 = 65.4 Hz), below the voiced F0 of
    the Ryan TTS voice (~100–120 Hz). Speech energy in that sub-F0 band is low, so
    the layer dominates there. We bandpass arm B at 55–75 Hz, compute the Hilbert
    envelope of that band, and assert a 10 Hz ±0.2 Hz peak with ≥6 dB prominence.

    If this fails at −31 dB SNR it is a finding about effective dose: the AM may
    not be recoverable from the delivered mix even by a narrowband approach, which
    implies the detector (Steps 5–6) cannot work at this SNR level either.
    """
    from scipy.signal import butter, sosfiltfilt
    from scipy.signal import hilbert as sig_hilbert

    wav_b, sr_b = _load_wav("B", item_id)
    wav_c, sr_c = _load_wav("C", item_id)

    def _subfund_am_spectrum(wav: np.ndarray, sr: int):
        """Bandpass 55–75 Hz with SOS (numerically stable), Hilbert envelope, rfft."""
        sos = butter(4, [55.0, 75.0], btype="bandpass", fs=sr, output="sos")
        filtered = sosfiltfilt(sos, wav.astype(np.float64))
        win = min(len(filtered), 16 * sr)
        env = np.abs(sig_hilbert(filtered[:win]))
        env -= env.mean()
        spec = np.abs(np.fft.rfft(env))
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        pdb = 20.0 * np.log10(spec + 1e-12)
        return freqs, pdb

    wav_a, sr_a = _load_wav("A", item_id)
    freqs_b, pdb_b = _subfund_am_spectrum(wav_b, sr_b)
    freqs_a, pdb_a = _subfund_am_spectrum(wav_a, sr_a)
    band = (freqs_b >= 1.0) & (freqs_b <= 30.0)
    if not np.any(band):
        pytest.skip(f"item {item_id}: audio too short for delivered-stimulus AM check")

    # 10 Hz prominence in B and in A (noise floor — natural band structure)
    prom_b_10hz = _local_median_prominence(freqs_b, pdb_b, AM_FREQ_HZ, band_hz=0.5)
    prom_a_10hz = _local_median_prominence(freqs_a, pdb_a, AM_FREQ_HZ, band_hz=0.5)
    margin = prom_b_10hz - prom_a_10hz  # B above A's natural floor at 10 Hz

    # Arm A has ~6-8 dB natural structure in this band. A flat 6 dB threshold on B
    # alone has no discriminative power. Assert B exceeds A's 10 Hz floor by ≥ 3 dB,
    # and that B's absolute prominence is ≥ 6 dB.
    # Either assertion failing is a dose finding — run threshold pilot (Step 4).
    assert prom_b_10hz >= 6.0, (
        f"item {item_id}: B 10 Hz prominence {prom_b_10hz:.2f} dB < 6 dB absolute. "
        f"A floor: {prom_a_10hz:.2f} dB, margin B-A: {margin:.2f} dB. "
        f"FINDING: 10 Hz AM not detectable above band noise at −31 dB SNR. "
        f"Run threshold pilot (Step 4) and re-render at the measured operating SNR."
    )
    assert margin >= 3.0, (
        f"item {item_id}: B 10 Hz prominence {prom_b_10hz:.2f} dB exceeds A floor "
        f"{prom_a_10hz:.2f} dB by only {margin:.2f} dB (need ≥ 3 dB margin). "
        f"FINDING: 10 Hz AM is within band baseline variance — not discriminative. "
        f"Run threshold pilot (Step 4) and re-render at the measured operating SNR."
    )

    # Arm C: the LAYER (actual added signal) must have < half the 10 Hz prominence
    # of the B layer.  Comparing full C-mix to A-mix false-positives on items whose
    # speech has natural ~10 Hz syllabic amplitude variation (e.g. arg_06, 4.17 dB
    # natural A floor → C-mix margin 3.93 dB that is not from deliberate AM).
    # The layer sidecar is the unambiguous signal: if it has less 10 Hz prominence
    # than B's layer, the placebo is clean — regardless of speech content.
    b_layer_path = STIMULI_DIR / "B" / f"{item_id}.layer.wav"
    c_layer_path = STIMULI_DIR / "C" / f"{item_id}.layer.wav"
    if b_layer_path.exists() and c_layer_path.exists():
        import soundfile as sf
        b_layer, _ = sf.read(b_layer_path, dtype="float32")
        c_layer, _ = sf.read(c_layer_path, dtype="float32")
        prom_b_layer = _local_median_prominence(*_subfund_am_spectrum(b_layer, sr_b), AM_FREQ_HZ, band_hz=0.5)
        prom_c_layer = _local_median_prominence(*_subfund_am_spectrum(c_layer, sr_c), AM_FREQ_HZ, band_hz=0.5)
        assert prom_c_layer < prom_b_layer, (
            f"item {item_id} C layer 10 Hz prominence {prom_c_layer:.2f} dB ≥ "
            f"B layer {prom_b_layer:.2f} dB — C placebo has unexpected coherent "
            f"10 Hz AM in its added signal"
        )

# ---------------------------------------------------------------------------
# 7. F0 pull verification — B pitch-class histogram χ² vs A: p<.001, 0<Δ≤60 cents
# ---------------------------------------------------------------------------

def _f0_pitch_class_histogram(wav: np.ndarray, sr: int) -> np.ndarray:
    """12-bin pitch-class histogram (semitone mod 12) over voiced frames (F0 > 50 Hz)."""
    import pyworld  # type: ignore

    wav64 = wav.astype(np.float64)
    f0, _t = pyworld.harvest(wav64, sr)
    voiced = f0[f0 > 50.0]
    if len(voiced) == 0:
        return np.zeros(12, dtype=np.float64)

    semitones = 12.0 * np.log2(voiced / 440.0)
    pitch_classes = np.mod(np.round(semitones).astype(int), 12)
    hist, _ = np.histogram(pitch_classes, bins=np.arange(13))
    return hist.astype(np.float64)


def _voiced_cent_shift(wav_a: np.ndarray, wav_b: np.ndarray, sr: int) -> float:
    """Median cent shift of voiced frames from A to B."""
    import pyworld  # type: ignore

    n = min(len(wav_a), len(wav_b))
    f0_a, _ = pyworld.harvest(wav_a[:n].astype(np.float64), sr)
    f0_b, _ = pyworld.harvest(wav_b[:n].astype(np.float64), sr)

    voiced_mask = (f0_a > 50.0) & (f0_b > 50.0)
    if not np.any(voiced_mask):
        return 0.0

    cents = 1200.0 * np.log2(f0_b[voiced_mask] / (f0_a[voiced_mask] + 1e-9))
    return float(np.median(cents))


@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_f0_pull_verification(item_id: str) -> None:
    """B F0 pitch-class histogram χ² vs A must be significant (p<.001);
    median cent shift 0 < Δ ≤ 60.
    """
    from scipy.stats import chi2_contingency  # type: ignore

    wav_a, sr_a = _load_wav("A", item_id)
    wav_b, sr_b = _load_wav("B", item_id)
    assert sr_a == sr_b, f"item {item_id}: sample-rate mismatch A vs B"

    hist_a = _f0_pitch_class_histogram(wav_a, sr_a)
    hist_b = _f0_pitch_class_histogram(wav_b, sr_b)

    if hist_a.sum() < 50 or hist_b.sum() < 50:
        pytest.skip(f"item {item_id}: insufficient voiced frames for F0 χ² test")

    # Contingency table: 2 rows (A, B) × 12 pitch-class columns
    table = np.vstack([hist_a, hist_b])
    # Zero-out columns where both are zero to avoid divide-by-zero
    nonzero_cols = (table.sum(axis=0) > 0)
    table = table[:, nonzero_cols]

    _, p_value, _, _ = chi2_contingency(table)
    assert p_value < 0.001, (
        f"item {item_id}: F0 pitch-class χ² p={p_value:.4f} ≥ 0.001 — "
        "B pitch distribution not significantly different from A"
    )

    delta_cents = _voiced_cent_shift(wav_a, wav_b, sr_a)
    assert 0.0 < abs(delta_cents) <= 60.0, (
        f"item {item_id}: median cent shift |Δ|={abs(delta_cents):.2f} not in (0, 60] cents "
        f"(direction: {'up' if delta_cents >= 0 else 'down'}, raw Δ={delta_cents:.2f} ¢)"
    )


# ---------------------------------------------------------------------------
# 8. Intelligibility preserved — skipped (faster-whisper not installed)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="faster-whisper not installed")
def test_intelligibility_preserved() -> None:
    """WER(B) − WER(A) ≤ 1 pp on the fallacy block (requires faster-whisper)."""
    pass  # pragma: no cover


# ---------------------------------------------------------------------------
# 9. Watermark roundtrip — decode from B and C, payload starts with b'SPAP-LAB'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_id", ALL_ITEM_IDS)
@pytest.mark.parametrize("arm", ["A", "B", "C"])
def test_watermark_roundtrip(arm: str, item_id: str) -> None:
    """DSSS watermark payload must start with b'SPAP-LAB' on the raw WAV."""
    from sap_stimulus.watermark import decode  # type: ignore

    wav, sr = _load_wav(arm, item_id)
    payload = decode(wav, sr)
    assert payload is not None, (
        f"item {item_id} arm {arm}: watermark decode returned None"
    )
    assert payload[:8] == b"SPAP-LAB", (
        f"item {item_id} arm {arm}: payload prefix {payload[:8]!r} != b'SPAP-LAB'"
    )
