# Acoustic Firewall — Technical Guide

Step 5 of the acoustic-persuasion-lab: how the detector works, how to run it,
and what its current limitations are. All scores in this document are from
actual runs on the rendered corpus, not predictions.

---

## Background: what the detector is trying to find

Arm B stimuli are built by mixing three manipulations into clean speech:

```
arm_B = (speech_F0_pulled + subharmonic_drone_ducked) × (1 + 0.03·cos(2π·10·t))
```

1. **F0 pull** — speech pitch nudged toward chord tones (Dm7 → G7 → Cmaj7 progression)
2. **Subharmonic drone** — 60–110 Hz sine waves following the chord, ducked to a fixed SNR
3. **Coherent AM** — the full mix multiplied by `1 + 0.03·cos(2π·10·t)` (3 % depth, 10 Hz)

Arm A is the same speech with no manipulation. Arm C (placebo) has a matched-energy
layer and jittered (incoherent) AM — the same layer energy as arm B, but no fixed-rate
coherent modulation and no F0 pull.

The firewall's job is to flag arm B as `entrainment_detected` and leave arm A as `clean`.

---

## Feature extraction — `acoustic_firewall/features.py`

`extract(wav, sr)` returns nine scalar features from a single WAV array.

### 1. 32-band ERB-spaced Butterworth filterbank

The human cochlea resolves frequency logarithmically. The Equivalent Rectangular Bandwidth
(ERB) scale captures this:

```
ERB_rate(f) = 21.4 · log₁₀(1 + 0.00437 · f)
```

Thirty-two center frequencies are spaced evenly in ERB-rate from 50 Hz to 8000 Hz, then
converted back to Hz. Each band uses a 4th-order Butterworth bandpass filter at ±25 % of
its center frequency (`scipy.signal.butter + sosfilt`).

> **Note on gammatone:** `scipy.signal.gammatone(fc, "iir")` was intended here but returns
> `b:(5,) a:(9,)` on scipy ≥ 1.7, incompatible with the single-biquad SOS construction the
> plan assumed. Butterworth is the confirmed fallback (plan assumption A1). The discrimination
> is unaffected — both filter types reveal the same envelope synchrony pattern.

### 2. Hilbert envelope → decimate to 200 Hz

For each band: `envelope = |hilbert(filtered)|`. The Hilbert transform gives the analytic
signal; its magnitude is the **instantaneous amplitude** — how loud that frequency band is
at every moment. Decimated to 200 Hz (from 22050 or 48000 Hz) since modulation of interest
tops out at ~100 Hz. Result: a 32 × N matrix of envelope time series.

### 3. Inter-band vector-strength synchrony (the primary discriminator)

For each 8-second window (1-second hop):

1. Remove DC from each band's envelope segment.
2. Compute the FFT of each band's envelope → 32 complex spectra.
3. At each frequency bin `f` in 1–30 Hz, compute **vector strength**:

```
synchrony(f) = |mean_over_32_bands(FFT_band(f))| / mean_over_32_bands(|FFT_band(f)|)
```

This equals **1.0** when all 32 bands have the same phase at frequency `f` (perfectly
coherent modulation), and **0.0** when phases are random (incoherent — natural speech).

**Why this detects arm B:** the `× (1 + 0.03·cos(2π·10·t))` term multiplies the full
mixed signal, so every band's envelope gets an identical 10 Hz component with identical
phase → synchrony ≈ 1.0 at 10 Hz. Natural speech modulation differs by band.

The window average gives three features:
- `inter_band_coherence_mean` — peak synchrony in 1–30 Hz
- `inter_band_coherence_peak_hz` — which frequency that peak sits at
- `inter_band_coherence_prominence_db` — peak above median baseline in dB

### 4. LF band energy and stationarity (40–120 Hz)

The subharmonic drone layer occupies 60–110 Hz. A Butterworth bandpass extracts this range:

- `lf_band_energy_ratio` = power(40–120 Hz) / total power
- `lf_band_stationarity` = 1 / (1 + CV) where CV = std of 100 ms frame LF energies / mean
  — a steady sine drone is more stationary than sporadic low-frequency speech content.

### 5. LF–F0 cross-correlation

`librosa.yin` extracts the fundamental frequency (F0) track (fmin=60/fmax=400 Hz,
frame=2048, hop=512 samples). The LF envelope is computed at the same frame rate.

`lf_f0_crosscorr` = Pearson r(F0, LF envelope) over voiced frames only
(`voiced = (f0 > 65) & (f0 < 380) & ~isnan(lf_env)`).

> **Edge case handled:** `librosa.yin` emits one frame past the last full hop when
> `len(wav)` is an exact multiple of the hop length. That frame's slice is empty →
> `np.mean([])` = nan. Fixed by guarding `i * hop_len_f0 < len(wav_lf)` and excluding
> nan frames from the voiced mask.

### 6. F0 pitch-class histogram χ²

Requires ≥ 20 s of voiced audio. Maps voiced F0 values to semitone pitch classes (mod 12)
and runs `scipy.stats.chisquare` against a uniform distribution. The F0 pull quantizes pitch
toward 4 chord tones, producing a non-uniform histogram. Returns `nan` for shorter stimuli.

### 7. Spectral kurtosis in 40–120 Hz

`scipy.stats.kurtosis` over 100 ms frames of the LF-filtered signal. A steady sine drone
(arm B layer) has near-Gaussian amplitude distribution (low kurtosis); impulsive natural LF
content (plosives, clicks) has higher kurtosis.

---

## Classifier — `acoustic_firewall/model.py`

```python
GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
```

Trained on 20 stimuli: 10 arm A WAVs (label 0, clean) and 10 arm B WAVs (label 1,
entrainment). Trust items excluded (no ground-truth label). `nan` → 0.0 imputation matches
training and inference. Train accuracy: **1.0000**.

The 9-feature vector order is fixed (insertion order in `extract()` return dict); `predict()`
uses `np.array(list(features.values()))` positionally — do not reorder keys.

---

## Actual scan results (verified on rendered corpus)

All scans run as `uv run acoustic-firewall scan corpus/stimuli/<arm>/<id>.wav`.

### prac_01 (practice item, sound argument)

| Arm | Verdict | Score | f_mod_hz |
|-----|---------|-------|----------|
| A   | `clean` | 0.00  | 5.5 Hz   |
| B   | `entrainment_detected` | 1.00 | 21.75 Hz |
| C   | `entrainment_detected` | 1.00 | 21.75 Hz |

### Cross-item summary (arg_01, arg_02, arg_05, arg_06)

| Item   | A verdict / score | B verdict / score | C verdict / score |
|--------|-------------------|-------------------|-------------------|
| arg_01 | clean / 0.0       | detected / 1.0    | **detected / 1.0** |
| arg_02 | clean / 0.0       | detected / 1.0    | **detected / 1.0** |
| arg_05 | clean / 0.0       | detected / 1.0    | **detected / 1.0** |
| arg_06 | clean / 0.0       | detected / 1.0    | **detected / 1.0** |

**Arm A: 100 % correctly classified clean.**
**Arm B: 100 % correctly classified detected.**
**Arm C: 100 % false positives — consistently misclassified as detected.**

---

## Known limitation: arm C false positives

The classifier perfectly separates arm A from arms B and C, but cannot distinguish B
from C. The root cause:

Arm C's matched-energy layer occupies the same 60–110 Hz band as arm B's subharmonic
drone. The `lf_band_energy_ratio` and `lf_band_stationarity` features fire for C as
strongly as for B. The jittered AM in arm C also still produces *some* envelope synchrony
within each 8-second window (the jitter rate matters: slow jitter → still locally coherent).

The `f_mod_hz` value is identical between B and C for every item — both are driven by the
same dominant envelope modulation from their shared LF layer, not by the coherent AM
specifically.

**What the classifier actually detects:** "has an added low-frequency layer" rather than
"has coherent broadband AM". The inter-band coherence feature is necessary but the LF
energy features dominate the decision boundary at the current training scale (20 stimuli).

**To fix this in Step 6+ :** include arm C in training with label 0 (or as a third class),
or engineer a feature that explicitly measures phase stability of the synchrony peak across
windows (jittered AM produces an unstable peak; coherent AM produces a rock-steady one).

---

## Note on `f_mod_hz`

`f_mod_hz` in the scan output is `inter_band_coherence_peak_hz` — the frequency of the
**dominant** envelope synchrony peak across all 32 bands in the 1–30 Hz range. On real
speech stimuli this is not guaranteed to be 10 Hz, because:

- The syllabic rhythm of speech (4–8 Hz) also creates some inter-band coherence
- The subharmonic drone's amplitude variation follows speech segmentation
- The 10 Hz injected AM adds to, not replaces, these natural modulations

The peak at 10 Hz is most visible in isolated synthetic AM signals (as the gate test shows:
`inter_band_coherence_peak_hz = 10.00 Hz` on pure bandlimited noise × AM). On full stimuli
the dominant peak shifts with the stimulus content. The classifier uses all 9 features
together, not just the peak frequency.

---

## How to run it

### Prerequisites

```bash
# From repo root. Stimuli must already be rendered:
ls corpus/stimuli/A/prac_01.wav   # should exist

# Train the detector (takes ~2 min, processes 20 WAVs):
uv run python -m acoustic_firewall.train \
    --stimuli corpus/stimuli \
    --corpus corpus/scripts/corpus_v1.json

# Model written to:
# packages/acoustic_firewall/models/detector_v1.joblib
```

### Scan a single file

```bash
uv run acoustic-firewall scan corpus/stimuli/B/arg_01.wav
# Output:
# {
#   "verdict": "entrainment_detected",
#   "score": 1.0,
#   "f_mod_hz": 9.0,
#   "threshold_used": 0.5
# }
```

### Run the automated gate tests (no audio files needed)

```bash
uv run pytest tests/test_firewall_features.py -v
# 4 passed — fully synthetic, deterministic (seed=42), ~11 s
```

The four tests:

| Test | Input | Assertion |
|------|-------|-----------|
| `test_extract_no_nan_clean` | Bandlimited noise | 8 of 9 features non-nan |
| `test_extract_no_nan_am` | Same × (1+0.03·cos(2π·10·t)) | 8 of 9 features non-nan |
| `test_am_coherence_exceeds_clean` | Both above | AM synchrony > clean synchrony |
| `test_am_peak_hz_near_10` | AM signal | Peak within 1 Hz of 10.0 Hz |

### Inspect raw features on any WAV

```python
# uv run python  (from repo root)
import sys
sys.path.insert(0, "packages/sap_core")
sys.path.insert(0, "packages/acoustic_firewall")
from sap_core.audio import load
from acoustic_firewall.features import extract

wav, sr = load("corpus/stimuli/B/prac_01.wav")
feats = extract(wav, sr)
for k, v in feats.items():
    print(f"{k:45s} {v:.4f}")
```

### Neutralize a detected file (Step 6 — not yet implemented)

```bash
uv run acoustic-firewall clean corpus/stimuli/B/arg_01.wav cleaned_arg_01.wav
# Currently raises NotImplementedError — implemented in Step 6
```

---

## File map

```
packages/acoustic_firewall/
  features.py        — extract(wav, sr) → dict[str, float]   ← Step 5 (this document)
  model.py           — load_model(), predict(), DetectReport
  train.py           — GBM trainer on rendered stimuli        ← Step 5
  neutralize.py      — neutralize(wav, sr, report) → ndarray ← Step 6 (stub)
  cli.py             — acoustic-firewall scan / clean / verify-watermark
  quality.py         — ESTOI / segSNR / LSD metrics for Step 6 acceptance
  watermark_verify.py — DSSS decode (mirrors sap_stimulus.watermark, decode side only)
  models/
    detector_v1.joblib  — trained GBM (generated by train.py; gitignored)
```
