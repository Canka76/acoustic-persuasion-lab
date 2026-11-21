# Acoustic Persuasion Lab — Handoff

**Repo:** `H:/GitRepos/acoustic-persuasion-lab`  
**Branch:** `master`  
**HEAD:** `48ca304` (fix: replace broken FIR Huggins with correct async double-pass OfflineAudioContext)  
**Study server:** `uv run uvicorn apps.study.server:app --port 8010` → http://127.0.0.1:8010  
**Admin token:** `dev-token` (env `ADMIN_TOKEN`); header `X-Admin-Token: dev-token`  
**Admin export:** `GET /admin/export.csv`

---

## Goal

End-to-end psychoacoustics RCT platform to test whether sub-perceptual acoustic manipulation (macro-harmonic scaffolding + 10 Hz AM layer) affects critical thinking. Priority: prove the effect first, build firewall second.

---

## Constraints

- Python 3.12, uv workspace, no Node/npm
- No PESQ/ViSQOL (no Windows wheel); use pystoi + segSNR + LSD
- faster-whisper for ASR (not installed; skip WER tests)
- Never relax a gate to make it pass — failures are findings
- All three arms share one WORLD vocoder baseline (strength=0 for A/C, strength=0.35 for B)
- AM applied AFTER duck_to_snr in both B and C arms
- `rt_invalid` stays binary 0/1 as registered; `has_replay` is a separate column

---

## Completed Steps (0–8)

- **Step 0–3:** Repo scaffold, corpus (14 items), generator pipeline, QA gate
- **Step 4:** Threshold pilot app (port 8001)
- **Step 5–6:** Pre-registration, power simulation (power=0.815 ≥ 0.80 ✓)
- **Step 7:** Final render at strength=0.350, SNR=−31 dB — 42 stimuli (14 items × 3 arms)
- **Step 8:** QA gate: 127/137 pass
  - 9 `test_am_verification` failures = dose findings (expected; AM is sub-perceptual by design)
  - 10/10 f0_pull pass, 42/42 watermark pass, 10/10 placebo_no_am pass

---

## Study App State

All app-level features are complete and committed:

| Feature | Status |
|---------|--------|
| Audio progress bar | ✓ |
| One-shot replay button (↺ Replay once) | ✓ |
| `has_replay` column (separate from `rt_invalid`) | ✓ |
| Headphone check: Huggins pitch, 400–600 Hz band, 2.0 s, retry cap=2 | ✓ |
| Huggins generator: async double-pass OfflineAudioContext (filtfilt) | ✓ **fixed** |
| RT freeze fix: `firstPlayOffset` for replay trials | ✓ |
| deviations.md: DEV-001 (replay), DEV-002 (headphone retry cap) | ✓ |

### Huggins Pitch Fix (critical)

**Root cause of previous failures:** 101-tap FIR has frequency resolution sr/N = 475 Hz > 200 Hz passband → passband gain −13 dB, essentially no filtering, so both sounds had the same binaural quality. Users failed headphone check repeatedly.

**Fix (commit `48ca304`):** Zero-phase bandpass via double-pass `OfflineAudioContext`:
1. Forward biquad pass (Q=2.5, fc=500 Hz, type=bandpass)
2. Reverse signal
3. Backward biquad pass
4. Reverse back = zero-phase `band_zp`
5. L = noise, R = noise − 2×band_zp

**Verified in numpy (exact WebAudio RBJ biquad coefficients):**
- IC at 500 Hz = −0.9998 (phase = ±180°) ✓
- IC at 100 Hz = +1.0000 (identical) ✓
- IC at 800 Hz = +1.0000 (identical) ✓
- Inversion bandwidth: 410–609 Hz ✓

A single biquad Q=2.5 double-pass is sufficient; cascading would narrow the band unnecessarily.

---

## Dose-Finding Summary

| Measure | Value |
|---------|-------|
| WORLD strength (arm B) | 0.350 |
| AM SNR | −31 dB |
| F0 median \|Δ¢\| | 27.4 ¢ (ground-truth from `pull_f0` internals) |
| F0 p90 \|Δ¢\| | 58.2 ¢ |
| AM physical detectability threshold | ≈ −15 dB SNR (not −31 dB) |

"Sub-perceptual" and "physically detectable in delivered mix" are mutually exclusive at all tested SNRs. This is a design constraint, not a bug.

---

## Remaining Steps

### Step 9 — Pilot + Main Run (blocked on participants/IRB)

1. **Internal dry run (n=6):** Have 6 internal participants complete a full session at http://127.0.0.1:8010. Need: headphone check passing, full trial block, complete CSV export.
   - Check CSV: `GET /admin/export.csv -H "X-Admin-Token: dev-token"`
   - Participant 1 (`492065fe`, arm A) already completed in previous dry run
2. **Threshold pilot (n=12):** Use port 8001 (threshold pilot app) to establish operating SNR. After pilot: re-render if needed, re-run QA gate.
3. **Main run (N=204):** After IRB approval and pilot data.

### Steps 5–6 — Acoustic Firewall (contingent on H1/H2 effect)

Build only after main study shows an effect. Modules to build:
- `acoustic_firewall/features.py` — detector
- `acoustic_firewall/model.py` + `train` — classifier
- `acoustic_firewall/neutralize.py` — neutralizer
- `apps/firewall-worklet` — AudioWorklet for real-time use

---

## Key Files

```
acoustic-persuasion-lab/
├── apps/study/
│   ├── server.py          # FastAPI app, /admin/export.csv, schema migrations
│   └── static/index.html  # Full study UI (JS), headphone check, trial blocks
├── apps/threshold_pilot/  # Port 8001, threshold pilot
├── analysis/
│   ├── analyze.py         # H5 filter: has_replay==0, arm comparisons
│   ├── preregistration.md
│   └── deviations.md      # DEV-001, DEV-002
├── packages/
│   ├── sap_core/          # audio, loudness, metrics (pystoi/segSNR/LSD)
│   └── sap_stimulus/      # render.py, f0_pull.py, placebo.py, layers.py
├── corpus/scripts/corpus_v1.json  # 14 items
├── stimuli/               # 42 rendered WAVs (gitignored *.wav)
│   └── qa/                # QA gate results
└── models/piper/          # TTS model (gitignored)
```

---

## Running the Server

```powershell
cd H:/GitRepos/acoustic-persuasion-lab
uv run uvicorn apps.study.server:app --port 8010
```

Hard-refresh browser (Ctrl+Shift+R) after any JS change.

---

## QA Gate

```powershell
cd H:/GitRepos/acoustic-persuasion-lab
uv run pytest packages/sap_stimulus/tests/ -v
```

Expected: 127 pass, 9 `test_am_verification` fail (dose findings).

---

## Next Action

Ask user to load http://127.0.0.1:8010 (hard-refresh), complete the headphone check (should now hear a faint centred tone in one of the two noise bursts), and complete a full session. That is dry-run participant 2/6.
