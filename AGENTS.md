# AGENTS.md — acoustic-persuasion-lab

Guidance for AI coding agents working in this repo. Read this before making any changes.

## Project Purpose

Acoustic persuasion RCT platform. Generates three-arm audio stimuli (A/control, B/entrainment,
C/placebo), serves a FastAPI study app, runs adaptive threshold detection, and ships an acoustic
firewall that detects and (eventually) neutralizes coherent amplitude-modulation entrainment.

**Active step:** Step 5 — `acoustic_firewall/features.py:extract()` (stub → real implementation).
Step 6 (`neutralize.py`) is the next dependency; do not start it until Step 5 is complete.

## Repository Layout

```
packages/
  sap_core/           Audio I/O, loudness, alignment, metrics (shared base)
  sap_stimulus/       LAB-ONLY stimulus generator — NEVER expose via network
  acoustic_firewall/  Detector + neutralizer (Steps 5-6 in progress)
apps/
  study/              FastAPI RCT platform (Step 7, complete)
  threshold_pilot/    2AFC adaptive staircase (Step 4, complete)
  firewall_worklet/   AudioWorklet demo (stub, not wired)
analysis/             Analysis pipeline + preregistration (Step 8, complete)
corpus/
  scripts/corpus_v1.json   14-item stimulus corpus (all experiment items)
  stimuli/A, B, C/         Rendered WAVs + manifests (run sap_stimulus.render to populate)
  base_renders/            TTS base renders before arm processing
tests/                pytest suite (run from repo root with uv run pytest)
graphify-out/         Knowledge graph outputs (GRAPH_REPORT.md, graph.html, graph.json)
```

## Architecture — Key Communities (from graphify)

Graph: **418 nodes · 819 edges · 21 communities**. Query with `graphify query "<question>"`.

| Community | Role | Key symbols |
|---|---|---|
| **Audio DSP Core** | Shared audio primitives — god node `load()` bridges everything | `sap_core.audio.load/save/rms`, `segment()`, `duck_to_snr()` |
| **Acoustic Firewall Detector** | Step 5 target | `features.extract()`, `model.predict()`, `DetectReport` |
| **Loudness Normalization** | Stimulus post-processing | `normalize_lufs()`, `peak_limit()`, `_post_process()` |
| **Study FastAPI App** | Participant-facing RCT server | `_get_conn()`, `get_stimulus()`, `create_participant()` |
| **Threshold Pilot App** | 2AFC adaptive staircase | `_build_trial_response()`, `post_trial()` |
| **Statistical Analysis** | Bayesian + frequentist outcomes | `bayes_factor_binom()`, `planned_contrast()`, `dprime_loglinear()` |
| **Pre-registration & Hypotheses** | H1–H5 as code | Lives in `analysis/preregistration.md` |
| **Test Harness & Fixtures** | QA gate for rendered stimuli | `test_stimulus_qa.py` (skips without corpus/stimuli/) |
| **Speech Quality Metrics** | ESTOI / segSNR / LSD thresholds | `acoustic_firewall/quality.py` |
| **Dependency & Boundary Tests** | Ethics gates | `acoustic_firewall` must never import `sap_stimulus` |

**God nodes** (highest betweenness — touch with extra care):

| Symbol | Degree | Why it matters |
|---|---|---|
| `render_all()` | 17 | Orchestrates all three stimulus arms; touching it breaks rendering |
| `load()` (sap_core.audio) | 14 | Used by every module that handles audio |
| `_post_process()` | 13 | TTS + loudness post-processing shared by all arms |
| `_build_arm_b()` | 12 | B-arm entrainment pipeline — source of training labels |
| `pull_f0()` | 11 | F0 pitch-pull shared between arms B and C |
| `extract()` | 10 | **Current work target** — acoustic_firewall feature extraction |

## Critical Rules

1. **Dependency boundary (enforced by test):** `acoustic_firewall` must never import `sap_stimulus`.
   `uv run pytest tests/test_dependency_boundary.py` enforces this. Any edit to
   `acoustic_firewall/` must pass this test.

2. **No network surface for stimulus generator:** `apps/` must never import `sap_stimulus`.
   `tests/test_no_generator_endpoint.py` enforces this.

3. **Watermark is non-optional:** every generated stimulus must carry the DSSS watermark
   (`sap_stimulus.watermark.embed`). Arm C (placebo) also embeds it.

4. **Trust items have no ground-truth label:** `block == "trust"` items use `sound: null`.
   Exclude them from firewall classifier training (labels 0/1 require `block in ("fallacy",
   "practice")`).

5. **Feature dict key order is load-bearing:** `model.py:predict()` calls
   `np.array(list(features.values()), ...)`. Do not add, remove, or reorder keys in
   `features.py:extract()`.

6. **sap_stimulus is LAB-ONLY:** guarded by ethics gate tests. Never relax these tests.

## Package Manager & Commands

```bash
uv run pytest                                          # full suite (QA tests skip without stimuli)
uv run pytest tests/test_dependency_boundary.py tests/test_no_generator_endpoint.py -q   # ethics gates (always runnable)
uv run pytest tests/test_firewall_features.py -q       # firewall feature gate (Step 5 target)
uv run python -m sap_stimulus.render --corpus corpus/scripts/corpus_v1.json --arms A B C --out corpus/stimuli/
uv run python -m acoustic_firewall.train               # train detector (requires rendered stimuli)
uv run acoustic-firewall scan <file.wav>               # run detector CLI
```

## Available Dependencies by Package

| Package | Key deps |
|---|---|
| `sap_core` | numpy, scipy, soundfile, pyloudnorm, librosa |
| `sap_stimulus` | sap_core, pyworld, piper-tts, torch, torchaudio, joblib |
| `acoustic_firewall` | sap_core, scikit-learn, joblib, pystoi |
| dev | pytest ≥ 8, pytest-cov, ruff |

`acoustic_firewall` reaches `librosa` and `scipy` transitively through `sap_core`.

## Step 5 Implementation Contract

**File to implement:** `packages/acoustic_firewall/features.py:extract(wav, sr)`

Returns a **9-key dict in fixed order** (key names and insertion order are frozen):
```python
{
    "inter_band_coherence_mean": float,        # synchrony peak in 1–30 Hz
    "inter_band_coherence_peak_hz": float,     # frequency of that peak
    "inter_band_coherence_prominence_db": float,
    "lf_band_energy_ratio": float,             # 40–120 Hz power / total power
    "lf_band_stationarity": float,             # 1 / (1 + CV of frame LF energies)
    "lf_f0_crosscorr": float,                  # Pearson r(F0, LF envelope)
    "f0_pitchclass_chi2": float,               # nan if < 20 s voiced
    "spectral_kurtosis_mean": float,
    "spectral_kurtosis_var": float,
}
```

Algorithm uses: 32-band ERB-spaced gammatone filterbank → Hilbert envelopes → inter-band
synchrony; Butterworth 40–120 Hz LF band; `librosa.yin` F0 track; `scipy.stats.kurtosis`.

**New files for Step 5:**
- `packages/acoustic_firewall/train.py` — GBM trainer (arm A → label 0, arm B → label 1)
- `tests/test_firewall_features.py` — deterministic synthetic-audio gate test

## Verification Gates (must all pass before marking Step 5 done)

1. `uv run pytest tests/test_firewall_features.py -q` → 4 passed
2. `grep -rn "sap_stimulus" packages/acoustic_firewall/` → zero hits
3. `uv run pytest tests/test_firewall_features.py tests/test_dependency_boundary.py tests/test_no_generator_endpoint.py -q` → 6 passed
4. Full suite delta = exactly +4 tests vs. pre-Step-5 baseline of 2 passed

## Knowledge Graph

Built with graphify; outputs in `graphify-out/`. Query live:
```bash
graphify query "how does arm B entrainment layer reach the final mixed audio?"
graphify path "extract" "DetectReport"
graphify explain "render_all"
```

Surprising cross-community connection discovered by graph: `test_am_coherence_exceeds_clean()`
already has an EXTRACTED `calls` edge to `extract()` — the test was written before the
implementation, confirming the TDD gate is correctly wired.

Rebuild graph after major structural changes: `graphify . --update` from repo root.
